#!/usr/bin/env python3
"""
Audio capture utilities for dual-stream transcription.
Handles microphone and system audio capture simultaneously.
"""

import sounddevice as sd
import numpy as np
import threading
import queue
import time
from typing import Optional, Callable, Dict, List, Tuple
import platform
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioCapture:
    """Handles simultaneous capture of microphone and system audio."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 0.5,
        mic_device: Optional[int] = None,
        system_device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)

        # Audio devices
        self.mic_device = mic_device
        self.system_device = system_device

        # Audio queues for each stream
        self.mic_queue = queue.Queue()
        self.system_queue = queue.Queue()

        # Control flags
        self.is_recording = False
        self.threads = []

        # Callbacks for processed audio chunks
        self.mic_callback: Optional[Callable] = None
        self.system_callback: Optional[Callable] = None

    @staticmethod
    def list_audio_devices() -> Dict[str, List[Dict]]:
        """List available audio devices categorized by type."""
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

        return {
            'input': input_devices,
            'output': output_devices
        }

    @staticmethod
    def find_system_audio_device() -> Optional[int]:
        """
        Attempt to find system audio device.
        Platform-specific logic for capturing system/desktop audio.
        """
        devices = sd.query_devices()
        system = platform.system().lower()

        # Check if we're in WSL environment
        is_wsl = platform.uname().release.lower().find('microsoft') >= 0

        for i, device in enumerate(devices):
            name = device['name'].lower()

            if system == 'windows':
                # Look for stereo mix, system sounds, or WASAPI loopback
                if any(keyword in name for keyword in [
                    'stereo mix', 'system sounds', 'speakers',
                    'headphones', 'what u hear', 'loopback'
                ]):
                    if device['max_input_channels'] > 0:
                        logger.info(f"Found potential system audio device: {device['name']} (ID: {i})")
                        return i

            elif system == 'linux':
                if is_wsl:
                    # In WSL with WSLg, check for PulseAudio capture_sink.monitor
                    import subprocess
                    import os
                    try:
                        env = os.environ.copy()
                        env['PULSE_SERVER'] = 'unix:/mnt/wslg/PulseServer'
                        result = subprocess.run(['pactl', 'list', 'sources', 'short'],
                                              capture_output=True, text=True, env=env)
                        if 'capture_sink.monitor' in result.stdout:
                            # In WSLg, we use the default ALSA device to access PulseAudio
                            # and rely on setting the PULSE_DEVICE environment variable
                            logger.info(f"Found WSLg system audio via capture_sink.monitor, using default device (ID: 1)")
                            return 1  # Use default device for WSLg
                    except Exception as e:
                        logger.debug(f"WSL PulseAudio check failed: {e}")

                # Look for PulseAudio monitor devices (standard Linux)
                if 'monitor' in name or 'output' in name:
                    if device['max_input_channels'] > 0:
                        logger.info(f"Found potential system audio device: {device['name']} (ID: {i})")
                        return i

            elif system == 'darwin':  # macOS
                # Look for system audio or aggregate devices
                if any(keyword in name for keyword in [
                    'system audio', 'soundflower', 'blackhole', 'aggregate'
                ]):
                    if device['max_input_channels'] > 0:
                        logger.info(f"Found potential system audio device: {device['name']} (ID: {i})")
                        return i

        logger.warning("No system audio device found automatically")
        return None

    def _mic_audio_callback(self, indata, frames, time_info, status):
        """Callback for microphone audio stream."""
        if status:
            logger.warning(f"Microphone audio callback status: {status}")

        # Convert to mono if stereo
        if indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata.flatten()

        # Add to queue
        self.mic_queue.put(audio_data.copy())

    def _system_audio_callback(self, indata, frames, time_info, status):
        """Callback for system audio stream."""
        if status:
            logger.warning(f"System audio callback status: {status}")

        # Convert to mono if stereo
        if indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata.flatten()

        # Add to queue
        self.system_queue.put(audio_data.copy())

    def _process_mic_queue(self):
        """Process microphone audio queue in separate thread."""
        while self.is_recording:
            try:
                # Get audio chunk from queue
                audio_data = self.mic_queue.get(timeout=1.0)

                # Call callback if set
                if self.mic_callback:
                    self.mic_callback(audio_data, 'mic')

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing mic queue: {e}")

    def _process_system_queue(self):
        """Process system audio queue in separate thread."""
        while self.is_recording:
            try:
                # Get audio chunk from queue
                audio_data = self.system_queue.get(timeout=1.0)

                # Call callback if set
                if self.system_callback:
                    self.system_callback(audio_data, 'system')

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing system queue: {e}")

    def set_callbacks(self, mic_callback: Callable, system_callback: Callable):
        """Set callbacks for processed audio chunks."""
        self.mic_callback = mic_callback
        self.system_callback = system_callback

    def start_recording(self):
        """Start dual audio capture."""
        if self.is_recording:
            logger.warning("Recording already in progress")
            return

        self.is_recording = True

        # Set up WSL environment if needed
        import os
        is_wsl = platform.uname().release.lower().find('microsoft') >= 0
        if is_wsl:
            os.environ['PULSE_SERVER'] = 'unix:/mnt/wslg/PulseServer'
            os.environ['PULSE_DEVICE'] = 'capture_sink.monitor'
            logger.info("Set WSLg PulseAudio environment variables")

        # Auto-detect system audio device if not specified
        if self.system_device is None:
            self.system_device = self.find_system_audio_device()
            if self.system_device is None:
                logger.error("Could not find system audio device. Please specify manually.")
                raise RuntimeError("System audio device not found")

        try:
            # Start microphone stream
            self.mic_stream = sd.InputStream(
                device=self.mic_device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self._mic_audio_callback,
                dtype=np.float32
            )

            # Start system audio stream
            self.system_stream = sd.InputStream(
                device=self.system_device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self._system_audio_callback,
                dtype=np.float32
            )

            # Start audio streams
            self.mic_stream.start()
            self.system_stream.start()

            # Start processing threads
            mic_thread = threading.Thread(target=self._process_mic_queue, daemon=True)
            system_thread = threading.Thread(target=self._process_system_queue, daemon=True)

            mic_thread.start()
            system_thread.start()

            self.threads = [mic_thread, system_thread]

            logger.info("Dual audio capture started")
            logger.info(f"Microphone device: {self.mic_device}")
            logger.info(f"System audio device: {self.system_device}")

        except Exception as e:
            self.is_recording = False
            logger.error(f"Failed to start audio capture: {e}")
            raise

    def stop_recording(self):
        """Stop dual audio capture."""
        if not self.is_recording:
            return

        self.is_recording = False

        # Stop streams
        if hasattr(self, 'mic_stream'):
            self.mic_stream.stop()
            self.mic_stream.close()

        if hasattr(self, 'system_stream'):
            self.system_stream.stop()
            self.system_stream.close()

        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2.0)

        logger.info("Dual audio capture stopped")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_recording()


def demo_audio_devices():
    """Demo function to list available audio devices."""
    print("=== Available Audio Devices ===")

    devices = AudioCapture.list_audio_devices()

    print("\n--- Input Devices (Microphones) ---")
    for device in devices['input']:
        print(f"ID {device['id']}: {device['name']} "
              f"({device['channels']} channels, {device['sample_rate']} Hz)")

    print("\n--- Output Devices (Speakers/Headphones) ---")
    for device in devices['output']:
        print(f"ID {device['id']}: {device['name']} "
              f"({device['channels']} channels, {device['sample_rate']} Hz)")

    # Try to find system audio automatically
    system_device = AudioCapture.find_system_audio_device()
    if system_device is not None:
        print(f"\nAuto-detected system audio device: ID {system_device}")
    else:
        print("\nNo system audio device auto-detected")
        print("You may need to:")
        print("- Enable 'Stereo Mix' in Windows Sound settings")
        print("- Install PulseAudio monitor on Linux")
        print("- Install BlackHole or SoundFlower on macOS")


if __name__ == "__main__":
    demo_audio_devices()