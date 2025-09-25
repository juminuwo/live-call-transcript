#!/usr/bin/env python3
"""
Reliable transcription engine using faster-whisper directly.
No more server/client WebSocket complexity - just local, direct transcription.
"""

import numpy as np
import threading
import queue
import time
import logging
import io
import wave
from typing import Optional, Callable, Dict, List, Tuple
from collections import deque
import hashlib

logger = logging.getLogger(__name__)


class AudioBuffer:
    """Smart audio buffer with VAD and overflow protection."""

    def __init__(self, sample_rate: int = 16000, max_duration: float = 10.0):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)
        self.buffer = deque(maxlen=self.max_samples)
        self.lock = threading.Lock()

    def add_audio(self, audio_data: np.ndarray):
        """Add audio data to buffer."""
        with self.lock:
            self.buffer.extend(audio_data)

    def get_audio(self, duration: float = 5.0) -> Optional[np.ndarray]:
        """Get audio data for transcription."""
        samples_needed = int(self.sample_rate * duration)

        with self.lock:
            if len(self.buffer) < samples_needed:
                return None

            # Get the audio data
            audio_data = np.array(list(self.buffer)[-samples_needed:])
            return audio_data

    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer.clear()

    def size(self) -> float:
        """Get buffer size in seconds."""
        with self.lock:
            return len(self.buffer) / self.sample_rate


class TranscriptionDeduplicator:
    """Smart deduplication for transcription results."""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.recent_texts = deque(maxlen=max_history)
        self.text_hashes = set()
        self.last_text = ""
        self.last_time = 0.0

    def is_duplicate(self, text: str, timestamp: float) -> bool:
        """Check if this text is likely a duplicate."""
        if not text or not text.strip():
            return True

        text = text.strip()
        current_time = timestamp

        # Quick hash check
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.text_hashes:
            logger.debug(f"Hash duplicate detected: {text}")
            return True

        # Check for substring duplicates (common with progressive transcription)
        if self.last_text:
            # If current text is contained in last text and recent
            if text in self.last_text and (current_time - self.last_time) < 3.0:
                logger.debug(f"Substring duplicate: '{text}' in '{self.last_text}'")
                return True

            # If last text is contained in current but very recent (< 2s)
            if (self.last_text in text and
                (current_time - self.last_time) < 2.0 and
                len(text) < len(self.last_text) * 1.3):  # Not significantly longer
                logger.debug(f"Minor extension: '{self.last_text}' -> '{text}'")
                return True

        # Check against recent texts for exact matches
        for recent_text in self.recent_texts:
            if text == recent_text:
                logger.debug(f"Recent duplicate: {text}")
                return True

        return False

    def add_text(self, text: str, timestamp: float):
        """Add text to history."""
        text = text.strip()
        if text:
            self.recent_texts.append(text)
            text_hash = hashlib.md5(text.encode()).hexdigest()
            self.text_hashes.add(text_hash)
            self.last_text = text
            self.last_time = timestamp

            # Clean old hashes if we have too many
            if len(self.text_hashes) > self.max_history * 2:
                self.text_hashes.clear()
                # Rebuild from recent texts
                for recent_text in self.recent_texts:
                    recent_hash = hashlib.md5(recent_text.encode()).hexdigest()
                    self.text_hashes.add(recent_hash)


class TranscriptionEngine:
    """Reliable transcription engine using faster-whisper."""

    def __init__(
        self,
        model_name: str = "small",
        language: str = "en",
        device: str = "auto",
        compute_type: str = "float16",
        callback: Optional[Callable] = None
    ):
        self.model_name = model_name
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.callback = callback

        # Model and processing
        self.model = None
        self.is_running = False

        # Audio buffers for each stream
        self.mic_buffer = AudioBuffer()
        self.desktop_buffer = AudioBuffer()

        # Deduplicators for each stream
        self.mic_dedup = TranscriptionDeduplicator()
        self.desktop_dedup = TranscriptionDeduplicator()

        # Processing threads
        self.mic_thread = None
        self.desktop_thread = None
        self.stop_event = threading.Event()

        # Stats
        self.mic_transcriptions = 0
        self.desktop_transcriptions = 0
        self.start_time = None

    def _initialize_model(self):
        """Initialize the faster-whisper model."""
        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading faster-whisper model: {self.model_name}")
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info("Model loaded successfully")

        except ImportError:
            logger.error("faster-whisper not installed. Installing fallback to openai-whisper")
            try:
                import whisper
                self.model = whisper.load_model(self.model_name)
                logger.info(f"Loaded OpenAI Whisper model: {self.model_name}")
            except ImportError:
                raise ImportError("Neither faster-whisper nor openai-whisper is installed")

    def _transcribe_audio(self, audio_data: np.ndarray) -> List[Dict]:
        """Transcribe audio data."""
        if self.model is None:
            return []

        try:
            # Convert audio to the format expected by the model
            audio_data = audio_data.astype(np.float32)

            # Use faster-whisper if available
            if hasattr(self.model, 'transcribe'):
                segments, info = self.model.transcribe(
                    audio_data,
                    language=self.language,
                    beam_size=1,  # Faster inference
                    best_of=1,    # Faster inference
                    vad_filter=True,  # Filter out non-speech
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=200
                    )
                )

                results = []
                for segment in segments:
                    results.append({
                        'text': segment.text.strip(),
                        'start': segment.start,
                        'end': segment.end,
                        'confidence': getattr(segment, 'avg_logprob', 0.0)
                    })
                return results

            else:
                # Fallback to OpenAI Whisper
                result = self.model.transcribe(audio_data, language=self.language)
                segments = []
                for segment in result.get('segments', []):
                    segments.append({
                        'text': segment.get('text', '').strip(),
                        'start': segment.get('start', 0.0),
                        'end': segment.get('end', 0.0),
                        'confidence': 0.0
                    })
                return segments

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return []

    def _process_stream(self, buffer: AudioBuffer, deduplicator: TranscriptionDeduplicator,
                       source: str, stats_attr: str):
        """Process audio stream for transcription."""
        logger.info(f"Started transcription thread for {source}")

        last_transcription_time = time.time()

        while self.is_running and not self.stop_event.wait(1.0):
            try:
                # Check if we have enough audio
                buffer_duration = buffer.size()
                if buffer_duration < 2.0:  # Need at least 2 seconds
                    continue

                # Get audio for transcription
                audio_data = buffer.get_audio(duration=5.0)  # Process 5-second chunks
                if audio_data is None:
                    continue

                # Check if audio has enough energy (basic VAD)
                rms = np.sqrt(np.mean(audio_data**2))
                if rms < 0.005:  # Too quiet, probably no speech
                    continue

                # Transcribe
                segments = self._transcribe_audio(audio_data)
                current_time = time.time()

                for segment in segments:
                    text = segment['text']
                    if not text:
                        continue

                    # Check for duplicates
                    if deduplicator.is_duplicate(text, current_time):
                        continue

                    # Add to deduplicator history
                    deduplicator.add_text(text, current_time)

                    # Call callback
                    if self.callback:
                        self.callback(current_time, source, text)

                    # Update stats
                    setattr(self, stats_attr, getattr(self, stats_attr) + 1)

                    logger.info(f"[{source.upper()}] {text}")

                last_transcription_time = current_time

                # Clear some old buffer data to prevent memory issues
                if buffer.size() > 8.0:  # If buffer has more than 8 seconds
                    # Get fresh audio to clear old data
                    buffer.get_audio(duration=3.0)

            except Exception as e:
                logger.error(f"Error processing {source} stream: {e}")
                time.sleep(1)

        logger.info(f"Transcription thread for {source} stopped")

    def start(self):
        """Start the transcription engine."""
        if self.is_running:
            logger.warning("Transcription engine already running")
            return

        logger.info("Starting reliable transcription engine...")

        try:
            # Initialize model
            self._initialize_model()

            # Start processing
            self.is_running = True
            self.stop_event.clear()
            self.start_time = time.time()

            # Start processing threads
            self.mic_thread = threading.Thread(
                target=self._process_stream,
                args=(self.mic_buffer, self.mic_dedup, "mic", "mic_transcriptions"),
                daemon=True
            )

            self.desktop_thread = threading.Thread(
                target=self._process_stream,
                args=(self.desktop_buffer, self.desktop_dedup, "desktop", "desktop_transcriptions"),
                daemon=True
            )

            self.mic_thread.start()
            self.desktop_thread.start()

            logger.info("Transcription engine started successfully")

        except Exception as e:
            logger.error(f"Failed to start transcription engine: {e}")
            self.stop()
            raise

    def stop(self):
        """Stop the transcription engine."""
        if not self.is_running:
            return

        logger.info("Stopping transcription engine...")
        self.is_running = False
        self.stop_event.set()

        # Wait for threads to finish
        if self.mic_thread:
            self.mic_thread.join(timeout=5)
        if self.desktop_thread:
            self.desktop_thread.join(timeout=5)

        # Clear buffers
        self.mic_buffer.clear()
        self.desktop_buffer.clear()

        logger.info("Transcription engine stopped")

    def process_mic_audio(self, audio_data: np.ndarray, source: str = "mic"):
        """Process microphone audio."""
        if self.is_running:
            self.mic_buffer.add_audio(audio_data)

    def process_desktop_audio(self, audio_data: np.ndarray, source: str = "desktop"):
        """Process desktop audio."""
        if self.is_running:
            self.desktop_buffer.add_audio(audio_data)

    def get_stats(self) -> Dict:
        """Get transcription statistics."""
        runtime = time.time() - self.start_time if self.start_time else 0
        return {
            'runtime_seconds': runtime,
            'mic_transcriptions': self.mic_transcriptions,
            'desktop_transcriptions': self.desktop_transcriptions,
            'mic_buffer_size': self.mic_buffer.size(),
            'desktop_buffer_size': self.desktop_buffer.size()
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def demo_transcription():
    """Demo the transcription engine."""
    print("=== Transcription Engine Demo ===")

    def transcription_callback(timestamp, source, text):
        time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
        print(f"[{time_str}] {source.upper()}: {text}")

    try:
        with TranscriptionEngine(callback=transcription_callback) as engine:
            print("Transcription engine started. Generating test audio...")

            # Generate some test audio (sine wave)
            sample_rate = 16000
            duration = 3
            t = np.linspace(0, duration, sample_rate * duration, False)
            test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

            # Send test audio to both streams
            for i in range(3):
                print(f"Sending test audio chunk {i+1}/3...")
                engine.process_mic_audio(test_audio)
                engine.process_desktop_audio(test_audio * 0.5)  # Quieter desktop audio
                time.sleep(4)

            print("\nFinal stats:")
            stats = engine.get_stats()
            print(f"Runtime: {stats['runtime_seconds']:.1f}s")
            print(f"Mic transcriptions: {stats['mic_transcriptions']}")
            print(f"Desktop transcriptions: {stats['desktop_transcriptions']}")

    except Exception as e:
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_transcription()