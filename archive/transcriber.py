#!/usr/bin/env python3
"""
Transcription manager using WhisperLive for dual audio stream processing.
Manages server startup and client connections for microphone and system audio.
"""

import subprocess
import time
import threading
import queue
import numpy as np
import logging
from typing import Optional, Callable, Dict
import websocket
import json
import socket
from contextlib import closing

# WhisperLive imports
import sys
import os
sys.path.append('./WhisperLive')

from whisper_live.client import TranscriptionClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperLiveServer:
    """Manages WhisperLive server process."""

    def __init__(self, port: int = 9090, backend: str = "faster_whisper", model: str = "small"):
        self.port = port
        self.backend = backend
        self.model = model
        self.server_process = None

    def is_port_available(self) -> bool:
        """Check if the server port is available."""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            return sock.connect_ex(('localhost', self.port)) != 0

    def start_server(self, timeout: int = 30):
        """Start the WhisperLive server process."""
        if not self.is_port_available():
            logger.info(f"Server already running on port {self.port}")
            return

        logger.info(f"Starting WhisperLive server on port {self.port}")

        # Build server command
        server_cmd = [
            'python', './WhisperLive/run_server.py',
            '--port', str(self.port),
            '--backend', self.backend
        ]

        try:
            logger.debug(f"Running server command: {' '.join(server_cmd)}")
            self.server_process = subprocess.Popen(
                server_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            logger.debug(f"Server process started with PID: {self.server_process.pid}")

            # Wait for server to start
            start_time = time.time()
            attempts = 0
            while time.time() - start_time < timeout:
                attempts += 1
                if not self.is_port_available():
                    logger.info(f"WhisperLive server started successfully after {attempts} attempts")
                    return

                # Check if process is still alive
                if self.server_process.poll() is not None:
                    stdout, stderr = self.server_process.communicate()
                    logger.error(f"Server process died unexpectedly. Exit code: {self.server_process.returncode}")
                    logger.error(f"Server stdout: {stdout.decode()}")
                    logger.error(f"Server stderr: {stderr.decode()}")
                    raise RuntimeError(f"Server process died with exit code {self.server_process.returncode}")

                if attempts % 5 == 0:  # Log every 5 seconds
                    logger.debug(f"Still waiting for server startup... attempt {attempts}")

                time.sleep(1)

            # Timeout reached - capture output for debugging
            stdout, stderr = self.server_process.communicate(timeout=5)
            logger.error(f"Server failed to start within {timeout}s. Output:")
            logger.error(f"Server stdout: {stdout.decode()}")
            logger.error(f"Server stderr: {stderr.decode()}")
            raise RuntimeError(f"Server failed to start within {timeout}s timeout")

        except Exception as e:
            logger.error(f"Failed to start WhisperLive server: {e}")
            self.stop_server()
            raise

    def stop_server(self):
        """Stop the WhisperLive server process."""
        if self.server_process:
            logger.info("Stopping WhisperLive server")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None

    def __enter__(self):
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_server()


class StreamingTranscriberClient:
    """Custom streaming client for real-time audio processing."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9090,
        source: str = "mic",
        lang: str = "en",
        model: str = "small",
        callback: Optional[Callable] = None
    ):
        self.host = host
        self.port = port
        self.source = source
        self.lang = lang
        self.model = model
        self.callback = callback

        # Audio processing
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.ws = None
        self.client_uid = None
        self.ws_connected = False
        self.server_ready = False
        self.connection_lock = threading.Lock()

        # Threading
        self.ws_thread = None
        self.audio_thread = None
        self.heartbeat_thread = None

        # Segment tracking to prevent duplicate processing
        self.processed_segments = set()
        self.max_processed_segments = 50  # Keep memory usage reasonable

        # Progressive segment tracking
        self.last_text = ""
        self.last_text_time = 0.0

        # Debug counters
        self.audio_sent_count = 0
        self.debug_counter = 0
        self.messages_received = 0
        self.last_message_time = time.time()

    def _on_open(self, ws):
        """WebSocket connection opened."""
        logger.info(f"WebSocket connection opened for {self.source}")

        # Send connection message with proper WhisperLive format
        connection_msg = {
            "uid": self.client_uid,
            "language": self.lang,
            "task": "transcribe",
            "model": self.model,
            "use_vad": True,
            "send_last_n_segments": 1,  # Reduce duplicate segments
            "no_speech_thresh": 0.45,
            "clip_audio": False,
            "same_output_threshold": 0.5,  # Reduce duplicate threshold
            "enable_translation": False,
            "target_language": "en"
        }
        try:
            ws.send(json.dumps(connection_msg))
            with self.connection_lock:
                self.ws_connected = True
            logger.info(f"Connection message sent for {self.source}, waiting for server ready...")
        except Exception as e:
            logger.error(f"Failed to send connection message for {self.source}: {e}")
            with self.connection_lock:
                self.ws_connected = False
                self.server_ready = False

    def _on_message(self, ws, message):
        """Handle WebSocket messages."""
        try:
            self.messages_received += 1
            self.last_message_time = time.time()

            data = json.loads(message)
            logger.debug(f"Received message #{self.messages_received} for {self.source}: {data}")

            # Check for server ready message
            if 'message' in data and data['message'] == 'SERVER_READY':
                with self.connection_lock:
                    self.server_ready = True
                logger.info(f"Server ready for {self.source}")
                return

            # Handle transcription segments
            if 'segments' in data:
                segments = data['segments']
                logger.debug(f"Processing {len(segments)} segments for {self.source}")
                for segment in segments:
                    if 'text' in segment and segment['text'].strip():
                        text = segment['text'].strip()

                        # Ensure timestamp is a float and handle relative/absolute timing
                        raw_timestamp = segment.get('start', 0.0)
                        try:
                            segment_timestamp = float(raw_timestamp)
                            # Use current time as base if segment timestamp seems relative
                            if segment_timestamp < 100000:  # Likely relative timestamp
                                timestamp = time.time()
                            else:
                                timestamp = segment_timestamp
                        except (ValueError, TypeError):
                            timestamp = time.time()

                        # Smart deduplication for progressive segments
                        current_time = timestamp

                        # Skip if this text is a subset of what we just processed
                        if (self.last_text and
                            text in self.last_text and
                            abs(current_time - self.last_text_time) < 5.0):
                            logger.debug(f"Skipping subset text for {self.source}: '{text}' (contained in '{self.last_text}')")
                            continue

                        # Skip if the last text is a subset of current text and very recent
                        if (self.last_text and
                            self.last_text in text and
                            abs(current_time - self.last_text_time) < 3.0):
                            logger.debug(f"Detected text extension for {self.source}: '{self.last_text}' -> '{text}'")
                            # Only send if it's significantly longer (20% more)
                            if len(text) > len(self.last_text) * 1.2:
                                logger.debug(f"Text significantly extended, allowing through")
                                # Update tracking and allow this to be processed
                                pass
                            else:
                                logger.debug(f"Text minimally extended, skipping")
                                continue

                        # Skip exact duplicates within a short time window
                        if (text == self.last_text and
                            abs(current_time - self.last_text_time) < 2.0):
                            logger.debug(f"Skipping exact duplicate for {self.source}: '{text}'")
                            continue

                        logger.debug(f"Processing new transcription for {self.source}: '{text}' at {timestamp}")

                        # Update tracking for next comparison
                        self.last_text = text
                        self.last_text_time = current_time

                        if self.callback:
                            self.callback(timestamp, self.source, text)

            # Handle simple text messages (fallback) - less common with proper segment handling
            elif 'text' in data and data.get('text', '').strip():
                text = data['text'].strip()
                current_time = time.time()

                # Apply same smart deduplication to fallback messages
                if (text == self.last_text and
                    abs(current_time - self.last_text_time) < 2.0):
                    logger.debug(f"Skipping duplicate fallback text for {self.source}: '{text}'")
                    return

                logger.debug(f"Processing fallback transcription for {self.source}: '{text}'")

                # Update tracking
                self.last_text = text
                self.last_text_time = current_time

                if self.callback:
                    self.callback(current_time, self.source, text)
            else:
                # Log when we receive a message that doesn't contain segments or text
                if 'message' not in data or data['message'] != 'SERVER_READY':
                    logger.debug(f"Received non-transcription message for {self.source}: {data}")

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {self.source}: {e}")
        except Exception as e:
            logger.error(f"Message handling error for {self.source}: {e}")

    def _on_error(self, ws, error):
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error for {self.source}: {error}")
        with self.connection_lock:
            self.ws_connected = False
            self.server_ready = False

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket closure."""
        logger.warning(f"WebSocket connection closed for {self.source}: {close_status_code} - {close_msg}")
        with self.connection_lock:
            self.ws_connected = False
            self.server_ready = False

    def _websocket_thread(self):
        """WebSocket connection thread."""
        try:
            import uuid
            self.client_uid = str(uuid.uuid4())

            # Create WebSocket connection
            ws_url = f"ws://{self.host}:{self.port}"
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )

            self.ws.run_forever()

        except Exception as e:
            logger.error(f"WebSocket thread error for {self.source}: {e}")

    def _audio_processing_thread(self):
        """Audio processing thread."""
        logger.info(f"Audio processing thread started for {self.source}")
        while self.is_running:
            try:
                # Get audio chunk from queue
                audio_data = self.audio_queue.get(timeout=1.0)
                logger.debug(f"Got audio data for {self.source}, queue size: {self.audio_queue.qsize()}")

                # Check connection and server ready status safely
                with self.connection_lock:
                    ready = self.ws_connected and self.server_ready and self.ws is not None

                if ready:
                    try:
                        # Convert to bytes if numpy array
                        if isinstance(audio_data, np.ndarray):
                            audio_bytes = audio_data.astype(np.float32).tobytes()
                        else:
                            audio_bytes = audio_data

                        # Send audio data to server
                        self.ws.send(audio_bytes, websocket.ABNF.OPCODE_BINARY)
                        self.audio_sent_count += 1

                        # Log every 10th audio send to avoid spam
                        if self.audio_sent_count % 10 == 0:
                            logger.debug(f"Sent {self.audio_sent_count} audio chunks for {self.source}")

                    except websocket.WebSocketConnectionClosedException:
                        logger.warning(f"WebSocket connection lost for {self.source}")
                        with self.connection_lock:
                            self.ws_connected = False
                        # Attempt reconnection
                        if self.is_running:
                            self._reconnect()
                    except Exception as send_error:
                        logger.error(f"Failed to send audio data for {self.source}: {send_error}")
                        with self.connection_lock:
                            self.ws_connected = False
                            self.server_ready = False
                        # Attempt reconnection on connection errors
                        if self.is_running and "Connection reset" in str(send_error):
                            self._reconnect()
                else:
                    # Connection not ready, skip this audio chunk
                    with self.connection_lock:
                        status = f"connected={self.ws_connected}, ready={self.server_ready}, ws_exists={self.ws is not None}"
                    logger.debug(f"Connection not ready for {self.source}: {status}")
                    continue

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio processing error for {self.source}: {e}")
                break

        logger.info(f"Audio processing thread stopped for {self.source}")

    def _heartbeat_monitor(self):
        """Monitor thread and connection health."""
        logger.info(f"Heartbeat monitor started for {self.source}")
        while self.is_running:
            try:
                # Check every 5 seconds
                time.sleep(5)

                if not self.is_running:
                    break

                with self.connection_lock:
                    ws_status = f"connected={self.ws_connected}, ready={self.server_ready}"

                queue_size = self.audio_queue.qsize()
                thread_status = f"ws_alive={self.ws_thread and self.ws_thread.is_alive()}, audio_alive={self.audio_thread and self.audio_thread.is_alive()}"

                # Check if we're still receiving messages
                time_since_last_msg = time.time() - self.last_message_time

                logger.info(f"[HEARTBEAT] {self.source}: {ws_status}, queue={queue_size}, {thread_status}, audio_sent={self.audio_sent_count}, msgs_received={self.messages_received}, last_msg_ago={time_since_last_msg:.1f}s")

                # If we haven't received a message in 20 seconds but are still sending audio, reconnect
                if (time_since_last_msg > 20.0 and
                    self.ws_connected and
                    self.server_ready and
                    self.audio_sent_count > 0):
                    logger.warning(f"WhisperLive server appears stuck for {self.source} - attempting reconnect")
                    self._reconnect()

            except Exception as e:
                logger.error(f"Heartbeat monitor error for {self.source}: {e}")

        logger.info(f"Heartbeat monitor stopped for {self.source}")

    def start(self):
        """Start the streaming client."""
        if self.is_running:
            return

        self.is_running = True

        # Start WebSocket thread
        self.ws_thread = threading.Thread(target=self._websocket_thread, daemon=True)
        self.ws_thread.start()

        # Wait for WebSocket to connect and server to be ready
        max_wait = 15  # seconds
        wait_time = 0
        while wait_time < max_wait:
            with self.connection_lock:
                if self.ws_connected and self.server_ready:
                    break
            time.sleep(0.5)
            wait_time += 0.5

        with self.connection_lock:
            ready = self.ws_connected and self.server_ready

        if not ready:
            logger.error(f"Failed to establish ready WebSocket connection for {self.source} within {max_wait}s")
            return

        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self._audio_processing_thread, daemon=True)
        self.audio_thread.start()

        # Start heartbeat monitor
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
        self.heartbeat_thread.start()

        logger.info(f"Started streaming transcription client for {self.source}")

    def _reconnect(self):
        """Attempt to reconnect WebSocket."""
        logger.info(f"Attempting to reconnect WebSocket for {self.source}")

        # Close existing connection
        if self.ws:
            try:
                self.ws.close()
            except:
                pass

        with self.connection_lock:
            self.ws_connected = False
            self.server_ready = False

        # Reset counters for fresh tracking
        self.messages_received = 0
        self.last_message_time = time.time()

        # Wait a bit before reconnecting
        time.sleep(2)

        # Start new WebSocket thread
        self.ws_thread = threading.Thread(target=self._websocket_thread, daemon=True)
        self.ws_thread.start()

        # Wait for connection and server ready
        max_wait = 15
        wait_time = 0
        while wait_time < max_wait and self.is_running:
            with self.connection_lock:
                if self.ws_connected and self.server_ready:
                    logger.info(f"Successfully reconnected WebSocket for {self.source}")
                    return True
            time.sleep(0.5)
            wait_time += 0.5

        logger.error(f"Failed to reconnect WebSocket for {self.source}")
        return False

    def restart_required(self):
        """Check if this client needs a server restart."""
        return not self.ws_connected and not self.server_ready

    def stop(self):
        """Stop the streaming client."""
        if not self.is_running:
            return

        self.is_running = False

        with self.connection_lock:
            self.ws_connected = False
            self.server_ready = False

        # Close WebSocket
        if self.ws:
            self.ws.close()

        # Wait for threads to finish
        if self.ws_thread:
            self.ws_thread.join(timeout=5)
        if self.audio_thread:
            self.audio_thread.join(timeout=5)
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5)

        logger.info(f"Stopped streaming transcription client for {self.source}")

    def send_audio(self, audio_data: np.ndarray):
        """Send audio data for transcription."""
        if self.is_running:
            self.audio_queue.put(audio_data)
            logger.debug(f"Queued audio data for {self.source}, queue size now: {self.audio_queue.qsize()}")
        else:
            logger.warning(f"Cannot queue audio for {self.source} - client not running")


class DualStreamTranscriber:
    """Manages dual-stream transcription for microphone and system audio."""

    def __init__(
        self,
        port: int = 9090,
        lang: str = "en",
        model: str = "small",
        callback: Optional[Callable] = None
    ):
        self.port = port
        self.lang = lang
        self.model = model
        self.callback = callback

        # Components
        self.server = None
        self.mic_client = None
        self.system_client = None
        self.is_restarting = False

    def _transcription_callback(self, timestamp: float, source: str, text: str):
        """Handle transcription results from clients."""
        if self.callback:
            self.callback(timestamp, source, text)

    def start(self):
        """Start dual-stream transcription system."""
        try:
            # Start WhisperLive server
            logger.info("Starting WhisperLive server...")
            self.server = WhisperLiveServer(port=self.port, model=self.model)
            self.server.start_server()

            # Create transcription clients
            self.mic_client = StreamingTranscriberClient(
                port=self.port,
                source="mic",
                lang=self.lang,
                model=self.model,
                callback=self._transcription_callback
            )

            self.system_client = StreamingTranscriberClient(
                port=self.port,
                source="desktop",
                lang=self.lang,
                model=self.model,
                callback=self._transcription_callback
            )

            # Start clients
            self.mic_client.start()
            self.system_client.start()

            logger.info("Dual-stream transcription system started")

        except Exception as e:
            logger.error(f"Failed to start transcription system: {e}")
            self.stop()
            raise

    def stop(self):
        """Stop dual-stream transcription system."""
        logger.info("Stopping transcription system...")

        # Stop clients
        if self.mic_client:
            self.mic_client.stop()
        if self.system_client:
            self.system_client.stop()

        # Stop server
        if self.server:
            self.server.stop_server()

        logger.info("Transcription system stopped")

    def process_mic_audio(self, audio_data: np.ndarray):
        """Process microphone audio data."""
        if not self.is_restarting and self.mic_client:
            self.mic_client.send_audio(audio_data)

    def process_system_audio(self, audio_data: np.ndarray):
        """Process system audio data."""
        if not self.is_restarting and self.system_client:
            self.system_client.send_audio(audio_data)

    def __enter__(self):
        self.start()
        return self

    def restart_server_if_needed(self):
        """Restart the WhisperLive server if both clients have failed."""
        if (self.mic_client and self.mic_client.restart_required() and
            self.system_client and self.system_client.restart_required()):

            logger.warning("Both clients disconnected - restarting WhisperLive server")
            self.is_restarting = True

            try:
                # Stop all clients first
                logger.info("Stopping existing clients...")
                if self.mic_client:
                    self.mic_client.stop()
                if self.system_client:
                    self.system_client.stop()

                # Give clients time to stop completely
                time.sleep(2)

                # Restart server
                logger.info("Restarting WhisperLive server...")
                if self.server:
                    logger.info("Stopping existing server...")
                    self.server.stop_server()
                    logger.info("Waiting for server cleanup...")
                    time.sleep(5)  # Give more time for cleanup

                    logger.info("Starting server with reduced timeout for restart...")
                    self.server.start_server(timeout=15)  # Shorter timeout for restart
                else:
                    # Create new server if none exists
                    logger.info("Creating new server instance...")
                    self.server = WhisperLiveServer(port=self.port, model=self.model)
                    self.server.start_server(timeout=15)  # Shorter timeout

                logger.info("Server restart completed, waiting for stabilization...")
                time.sleep(2)

                # Recreate and restart clients
                logger.info("Creating new clients...")
                try:
                    self.mic_client = StreamingTranscriberClient(
                        port=self.port,
                        source="mic",
                        lang=self.lang,
                        model=self.model,
                        callback=self._transcription_callback
                    )
                    logger.info("Created new mic client")

                    self.system_client = StreamingTranscriberClient(
                        port=self.port,
                        source="desktop",
                        lang=self.lang,
                        model=self.model,
                        callback=self._transcription_callback
                    )
                    logger.info("Created new desktop client")

                    # Start clients
                    logger.info("Starting new clients...")
                    self.mic_client.start()
                    logger.info("Started mic client")

                    self.system_client.start()
                    logger.info("Started desktop client")

                    # Give clients time to establish connections
                    logger.info("Waiting for client connections to establish...")
                    time.sleep(3)

                    # Validate that clients are actually connected
                    mic_ok = self.mic_client and hasattr(self.mic_client, 'ws_connected') and self.mic_client.ws_connected
                    desktop_ok = self.system_client and hasattr(self.system_client, 'ws_connected') and self.system_client.ws_connected

                    if mic_ok and desktop_ok:
                        logger.info("Successfully restarted WhisperLive server and clients - both clients connected")
                        self.is_restarting = False
                        return True
                    else:
                        logger.warning(f"Restart completed but clients not fully connected: mic={mic_ok}, desktop={desktop_ok}")
                        self.is_restarting = False
                        return False

                except Exception as client_error:
                    logger.error(f"Failed to create/start new clients: {client_error}")
                    raise

            except Exception as e:
                logger.error(f"Failed to restart server: {e}")
                logger.exception("Detailed error information:")
                self.is_restarting = False
                return False

        return False

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def test_transcriber():
    """Test function for the transcriber system."""
    def transcription_callback(timestamp, source, text):
        print(f"[{timestamp:.2f}] {source}: {text}")

    logger.info("Testing transcriber system...")

    try:
        with DualStreamTranscriber(callback=transcription_callback) as transcriber:
            logger.info("Transcriber started, generating test audio...")

            # Generate some test audio data
            sample_rate = 16000
            duration = 2  # seconds
            frequency = 440  # A4 note

            for i in range(5):
                # Generate test sine wave
                t = np.linspace(0, duration, sample_rate * duration, False)
                test_audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

                # Send to both streams
                transcriber.process_mic_audio(test_audio)
                transcriber.process_system_audio(test_audio)

                time.sleep(3)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")


if __name__ == "__main__":
    test_transcriber()