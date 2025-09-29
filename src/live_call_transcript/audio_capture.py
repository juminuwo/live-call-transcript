"""
Reliable audio capture system for dual-stream transcription.
Simplified and more robust audio handling with better error recovery.
"""

import sounddevice as sd
import numpy as np
import threading
import queue
import time
from typing import Optional, Callable, Dict, List
import platform
import logging
import os

logger = logging.getLogger(__name__)


class AudioCapture:
    """Dual audio capture with proper error handling and recovery."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 0.3,
        mic_device: Optional[int] = None,
        system_device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)

        # Audio devices
        self.mic_device = mic_device
        self.system_device = system_device

        # Audio streams and threads
        self.mic_stream = None
        self.system_stream = None
        self.mic_thread = None
        self.system_thread = None

        # Control
        self.is_running = False
        self.stop_event = threading.Event()

        # Callbacks
        self.mic_callback: Optional[Callable] = None
        self.system_callback: Optional[Callable] = None

        # Error tracking
        self.mic_errors = 0
        self.system_errors = 0
        self.max_errors = 5

    @staticmethod
    def list_audio_devices() -> Dict[str, List[Dict]]:
        """List available audio devices."""
        try:
            devices = sd.query_devices()
            input_devices = []
            output_devices = []

            for i, device in enumerate(devices):
                device_info = {
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'] if device['max_input_channels'] > 0 else device['max_output_channels'],
                    'sample_rate': device['default_samplerate']
                }

                if device['max_input_channels'] > 0:
                    input_devices.append(device_info)
                if device['max_output_channels'] > 0:
                    output_devices.append(device_info)

            return {'input': input_devices, 'output': output_devices}
        except Exception as e:
            logger.error(f"Failed to list audio devices: {e}")
            return {'input': [], 'output': []}

    @staticmethod
    def find_system_audio_device() -> Optional[int]:
        """Find system audio device with improved detection."""
        try:
            devices = sd.query_devices()
            system = platform.system().lower()
            is_wsl = 'microsoft' in platform.uname().release.lower()

            for i, device in enumerate(devices):
                name = device['name'].lower()

                if system == 'windows' or is_wsl:
                    # Look for Windows system audio devices
                    if any(keyword in name for keyword in [
                        'stereo mix', 'system sounds', 'speakers',
                        'headphones', 'what u hear', 'loopback', 'wasapi'
                    ]):
                        if device['max_input_channels'] > 0:
                            logger.info(f"Found system audio device: {device['name']} (ID: {i})")
                            return i

                elif system == 'linux':
                    # Look for PulseAudio monitor devices
                    if 'monitor' in name or 'output' in name:
                        if device['max_input_channels'] > 0:
                            logger.info(f"Found system audio device: {device['name']} (ID: {i})")
                            return i

                elif system == 'darwin':  # macOS
                    # Look for macOS system audio devices
                    if any(keyword in name for keyword in [
                        'system audio', 'soundflower', 'blackhole', 'aggregate'
                    ]):
                        if device['max_input_channels'] > 0:
                            logger.info(f"Found system audio device: {device['name']} (ID: {i})")
                            return i

            logger.warning("No system audio device found automatically")
            return None

        except Exception as e:
            logger.error(f"Error finding system audio device: {e}")
            return None

    def _setup_wsl_audio(self):
        """Setup WSL audio environment if needed."""
        is_wsl = 'microsoft' in platform.uname().release.lower()
        if is_wsl:
            os.environ['PULSE_SERVER'] = 'unix:/mnt/wslg/PulseServer'
            os.environ['PULSE_DEVICE'] = 'capture_sink.monitor'
            logger.info("Configured WSL audio environment")

    def _create_stream(self, device_id: Optional[int], callback: Callable, name: str):
        """Create an audio stream with error handling."""
        try:
            stream = sd.InputStream(
                device=device_id,
                channels=1,  # Always use mono for simplicity
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=callback,
                dtype=np.float32
            )
            logger.info(f"Created {name} stream (device: {device_id})")
            return stream
        except Exception as e:
            logger.error(f"Failed to create {name} stream: {e}")
            raise

    def _mic_callback(self, indata, frames, time_info, status):
        """Microphone audio callback."""
        try:
            if status:
                logger.warning(f"Mic audio status: {status}")

            # Convert to mono and normalize
            if len(indata.shape) > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata.flatten()

            # Call user callback
            if self.mic_callback and self.is_running:
                self.mic_callback(audio_data.copy(), 'mic')

        except Exception as e:
            self.mic_errors += 1
            logger.error(f"Mic callback error ({self.mic_errors}/{self.max_errors}): {e}")
            if self.mic_errors >= self.max_errors:
                logger.error("Too many mic errors, stopping mic stream")
                if self.mic_stream:
                    self.mic_stream.stop()

    def _system_callback(self, indata, frames, time_info, status):
        """System audio callback."""
        try:
            if status:
                logger.warning(f"System audio status: {status}")

            # Convert to mono and normalize
            if len(indata.shape) > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata.flatten()

            # Call user callback
            if self.system_callback and self.is_running:
                self.system_callback(audio_data.copy(), 'desktop')

        except Exception as e:
            self.system_errors += 1
            logger.error(f"System callback error ({self.system_errors}/{self.max_errors}): {e}")
            if self.system_errors >= self.max_errors:
                logger.error("Too many system audio errors, stopping system stream")
                if self.system_stream:
                    self.system_stream.stop()

    def set_callbacks(self, mic_callback: Callable, system_callback: Callable):
        """Set audio processing callbacks."""
        self.mic_callback = mic_callback
        self.system_callback = system_callback

    def start(self):
        """Start audio capture."""
        if self.is_running:
            logger.warning("Audio capture already running")
            return

        logger.info("Starting reliable audio capture...")

        # Setup environment
        self._setup_wsl_audio()

        # Auto-detect system device if needed
        if self.system_device is None:
            self.system_device = self.find_system_audio_device()
            if self.system_device is None:
                logger.error("No system audio device found. Please specify manually or enable Stereo Mix.")
                raise RuntimeError("System audio device not available")

        self.is_running = True
        self.stop_event.clear()

        # Reset error counters
        self.mic_errors = 0
        self.system_errors = 0

        try:
            # Create and start microphone stream
            self.mic_stream = self._create_stream(
                self.mic_device, self._mic_callback, "microphone"
            )
            self.mic_stream.start()

            # Create and start system audio stream
            self.system_stream = self._create_stream(
                self.system_device, self._system_callback, "system audio"
            )
            self.system_stream.start()

            logger.info("Audio capture started successfully")
            logger.info(f"Microphone device: {self.mic_device}")
            logger.info(f"System audio device: {self.system_device}")

        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            self.stop()
            raise

    def stop(self):
        """Stop audio capture."""
        if not self.is_running:
            return

        logger.info("Stopping audio capture...")
        self.is_running = False
        self.stop_event.set()

        # Stop streams
        if self.mic_stream:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
            except Exception as e:
                logger.error(f"Error stopping mic stream: {e}")
            finally:
                self.mic_stream = None

        if self.system_stream:
            try:
                self.system_stream.stop()
                self.system_stream.close()
            except Exception as e:
                logger.error(f"Error stopping system stream: {e}")
            finally:
                self.system_stream = None

        logger.info("Audio capture stopped")

    def is_healthy(self) -> Dict[str, bool]:
        """Check if audio streams are healthy."""
        return {
            'mic': self.mic_stream is not None and self.mic_errors < self.max_errors,
            'system': self.system_stream is not None and self.system_errors < self.max_errors,
            'running': self.is_running
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def demo_audio_capture():
    """Demo the audio capture system."""
    print("=== Audio Capture Demo ===")

    # List devices
    devices = AudioCapture.list_audio_devices()
    print("\nInput devices:")
    for device in devices['input']:
        print(f"  [{device['id']:2d}] {device['name']}")

    # Find system audio
    system_device = AudioCapture.find_system_audio_device()
    if system_device is not None:
        print(f"\nAuto-detected system audio: {system_device}")
    else:
        print("\nNo system audio device found")

    # Test callbacks
    def mic_callback(audio_data, source):
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > 0.01:  # Only show when there's actual audio
            print(f"MIC: RMS={rms:.4f}")

    def system_callback(audio_data, source):
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > 0.01:  # Only show when there's actual audio
            print(f"DESKTOP: RMS={rms:.4f}")

    # Test capture
    try:
        with AudioCapture() as capture:
            capture.set_callbacks(mic_callback, system_callback)
            print("\nCapturing audio for 10 seconds...")
            print("Make some noise into your microphone and play some system audio!")
            time.sleep(10)
    except Exception as e:
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_audio_capture()