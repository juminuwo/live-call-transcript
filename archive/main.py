#!/usr/bin/env python3
"""
Live Call Transcription System
Main entry point for dual-stream real-time transcription.
"""

import argparse
import signal
import sys
import time
import threading
from typing import Optional
import logging

# Local imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from live_call_transcript.audio_utils import AudioCapture
from live_call_transcript.transcriber import DualStreamTranscriber
from live_call_transcript.logger import TranscriptionLogger, RealTimeDisplay

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LiveCallTranscriber:
    """Main application class for live call transcription."""

    def __init__(
        self,
        mic_device: Optional[int] = None,
        system_device: Optional[int] = None,
        output_file: Optional[str] = None,
        format_type: str = "json",
        sample_rate: int = 16000,
        lang: str = "en",
        model: str = "small",
        console_output: bool = True,
        real_time_display: bool = False
    ):
        self.mic_device = mic_device
        self.system_device = system_device
        self.output_file = output_file
        self.format_type = format_type
        self.sample_rate = sample_rate
        self.lang = lang
        self.model = model
        self.console_output = console_output
        self.real_time_display = real_time_display

        # Components
        self.audio_capture: Optional[AudioCapture] = None
        self.transcriber: Optional[DualStreamTranscriber] = None
        self.logger: Optional[TranscriptionLogger] = None
        self.display: Optional[RealTimeDisplay] = None

        # Control flags
        self.is_running = False
        self.shutdown_event = threading.Event()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()

    def _transcription_callback(self, timestamp: float, source: str, text: str):
        """Handle transcription results."""
        if not text.strip():
            return

        # Map source names
        source_map = {"mic": "mic", "desktop": "desktop", "system": "desktop"}
        mapped_source = source_map.get(source, source)

        # Log to file
        if self.logger:
            self.logger.log_transcription(timestamp, mapped_source, text)

        # Update real-time display
        if self.display:
            self.display.add_transcription(
                timestamp, mapped_source, text, self.logger.session_start if self.logger else time.time()
            )

    def _mic_audio_callback(self, audio_data, source):
        """Handle microphone audio data."""
        if self.transcriber:
            self.transcriber.process_mic_audio(audio_data)

    def _system_audio_callback(self, audio_data, source):
        """Handle system audio data."""
        if self.transcriber:
            self.transcriber.process_system_audio(audio_data)

    def setup(self):
        """Setup all components."""
        try:
            logger.info("Setting up live call transcription system...")

            # Setup logger
            self.logger = TranscriptionLogger(
                output_file=self.output_file,
                format_type=self.format_type,
                console_output=self.console_output and not self.real_time_display
            )

            # Setup real-time display if requested
            if self.real_time_display:
                self.display = RealTimeDisplay(max_lines=15, show_timestamps=True)

            # Setup transcriber
            self.transcriber = DualStreamTranscriber(
                lang=self.lang,
                model=self.model,
                callback=self._transcription_callback
            )

            # Setup audio capture
            self.audio_capture = AudioCapture(
                sample_rate=self.sample_rate,
                mic_device=self.mic_device,
                system_device=self.system_device
            )

            # Set audio callbacks
            self.audio_capture.set_callbacks(
                self._mic_audio_callback,
                self._system_audio_callback
            )

            logger.info("Setup completed successfully")

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise

    def start(self):
        """Start the transcription system."""
        if self.is_running:
            logger.warning("System already running")
            return

        try:
            logger.info("Starting live call transcription...")

            # Start logger
            if self.logger:
                self.logger.start()

            # Start transcriber
            if self.transcriber:
                self.transcriber.start()

            # Start audio capture
            if self.audio_capture:
                self.audio_capture.start_recording()

            self.is_running = True
            logger.info("Live call transcription started successfully")

            # Display status
            print("\n" + "="*60)
            print("LIVE CALL TRANSCRIPTION ACTIVE")
            print("="*60)
            if self.logger:
                if hasattr(self.logger, 'json_path'):
                    print(f"JSON output: {self.logger.json_path}")
                if hasattr(self.logger, 'csv_path'):
                    print(f"CSV output: {self.logger.csv_path}")
            print("Press Ctrl+C to stop transcription")
            print("="*60)

            # Clear display if using real-time mode
            if self.display:
                self.display.clear()

        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            self.shutdown()
            raise

    def run(self):
        """Run the main transcription loop."""
        if not self.is_running:
            logger.error("System not started")
            return

        try:
            # Main loop - just wait for shutdown signal
            while self.is_running and not self.shutdown_event.is_set():
                self.shutdown_event.wait(1.0)

                # Check if any component has failed and attempt restart if needed
                if self.transcriber:
                    if not hasattr(self.transcriber, 'server'):
                        logger.error("Transcriber server failed")
                        break

                    # Check if server restart is needed
                    if self.transcriber.restart_server_if_needed():
                        logger.info("Server restart completed - continuing operation")

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        """Shutdown all components gracefully."""
        if not self.is_running:
            return

        logger.info("Shutting down transcription system...")
        self.is_running = False
        self.shutdown_event.set()

        # Stop components in reverse order
        try:
            if self.audio_capture:
                self.audio_capture.stop_recording()

            if self.transcriber:
                self.transcriber.stop()

            if self.logger:
                stats = self.logger.get_statistics()
                logger.info(f"Session statistics: {stats}")
                self.logger.stop()

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

        logger.info("Shutdown completed")

    def __enter__(self):
        self.setup()
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


def list_audio_devices():
    """List available audio devices."""
    print("Scanning audio devices...")
    devices = AudioCapture.list_audio_devices()

    print("\n=== AUDIO DEVICES ===")
    print("\nMICROPHONE DEVICES (Input):")
    for device in devices['input']:
        print(f"  [{device['id']:2d}] {device['name']}")
        print(f"       {device['channels']} channels, {device['sample_rate']:.0f} Hz")

    print(f"\nSYSTEM AUDIO DEVICES:")
    system_device = AudioCapture.find_system_audio_device()
    if system_device is not None:
        for device in devices['input']:
            if device['id'] == system_device:
                print(f"  [{device['id']:2d}] {device['name']} *** AUTO-DETECTED ***")
                print(f"       {device['channels']} channels, {device['sample_rate']:.0f} Hz")
    else:
        print("  No system audio device auto-detected")
        print("\n  TROUBLESHOOTING:")
        print("  - Windows: Enable 'Stereo Mix' in Sound settings")
        print("  - Linux: Install PulseAudio with monitor devices")
        print("  - macOS: Install BlackHole or SoundFlower")

    print(f"\nALL OUTPUT DEVICES (for reference):")
    for device in devices['output']:
        print(f"  [{device['id']:2d}] {device['name']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Live Call Transcription System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available audio devices
  python main.py --list-devices

  # Start transcription with auto-detected devices
  python main.py

  # Specify devices manually
  python main.py --mic-device 1 --system-device 2

  # Output to specific file in CSV format
  python main.py --output call_transcript --format csv

  # Use real-time display mode
  python main.py --real-time-display

  # Use different language and model
  python main.py --lang es --model medium
        """
    )

    # Device selection
    parser.add_argument('--list-devices', '-l', action='store_true',
                        help='List available audio devices and exit')
    parser.add_argument('--mic-device', '-m', type=int,
                        help='Microphone device ID (use --list-devices to see options)')
    parser.add_argument('--system-device', '-s', type=int,
                        help='System audio device ID (auto-detected if not specified)')

    # Output options
    parser.add_argument('--output', '-o', type=str,
                        help='Output file base name (timestamp added if not specified)')
    parser.add_argument('--format', '-f', choices=['json', 'csv', 'both'], default='json',
                        help='Output format (default: json)')

    # Audio settings
    parser.add_argument('--sample-rate', '-r', type=int, default=16000,
                        help='Audio sample rate in Hz (default: 16000)')

    # Transcription settings
    parser.add_argument('--lang', type=str, default='en',
                        help='Language code for transcription (default: en)')
    parser.add_argument('--model', type=str, default='small',
                        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                        help='Whisper model to use (default: small)')

    # Display options
    parser.add_argument('--no-console', action='store_true',
                        help='Disable console output')
    parser.add_argument('--real-time-display', action='store_true',
                        help='Enable real-time console display (clears screen)')

    # Debug options
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # List devices and exit
    if args.list_devices:
        list_audio_devices()
        return 0

    try:
        # Create and run transcription system
        with LiveCallTranscriber(
            mic_device=args.mic_device,
            system_device=args.system_device,
            output_file=args.output,
            format_type=args.format,
            sample_rate=args.sample_rate,
            lang=args.lang,
            model=args.model,
            console_output=not args.no_console,
            real_time_display=args.real_time_display
        ) as transcriber:
            transcriber.run()

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Application failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())