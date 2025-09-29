"""
Reliable Live Call Transcription System
Main application using the new reliable architecture.
"""

import argparse
import signal
import sys
import time
import threading
import json
import csv
import os
from typing import Optional
import logging
from datetime import datetime

cuda_version = "13.0"
cuda_bin_path = fr"C:\Program Files\NVIDIA\CUDNN\v9.13\bin\{cuda_version}"

# Add to PATH (so DLL loader can find it)
os.environ["PATH"] = cuda_bin_path + os.pathsep + os.environ["PATH"]
# Local imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from live_call_transcript.audio_capture import AudioCapture
from live_call_transcript.transcription_engine import TranscriptionEngine
from live_call_transcript.logger import TranscriptionLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveTranscriber:
    """Live call transcription system."""

    def __init__(
        self,
        mic_device: Optional[int] = None,
        system_device: Optional[int] = None,
        output_file: Optional[str] = None,
        output_format: str = "json",
        model_name: str = "small",
        language: str = "en",
        sample_rate: int = 16000,
    ):
        self.mic_device = mic_device
        self.system_device = system_device
        self.output_file = output_file
        self.output_format = output_format
        self.model_name = model_name
        self.language = language
        self.sample_rate = sample_rate

        # Components
        self.audio_capture = None
        self.transcription_engine = None
        self.logger = None

        # Control
        self.is_running = False
        self.shutdown_event = threading.Event()

        # Output files
        self.json_file = None
        self.csv_file = None
        self.csv_writer = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Stats
        self.session_start = None
        self.transcription_count = 0

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()

    def _setup_output_files(self):
        """Setup output files for logging."""
        if self.output_file:
            base_name = self.output_file
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"data/transcript_{timestamp}"

        output_paths = []

        # JSON output
        if self.output_format in ['json', 'both']:
            self.json_path = f"{base_name}.json"
            self.json_file = open(self.json_path, 'w', encoding='utf-8')
            output_paths.append(self.json_path)
        else:
            self.json_file = None
            self.json_path = None

        # CSV output
        if self.output_format in ['csv', 'both']:
            self.csv_path = f"{base_name}.csv"
            self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['timestamp', 'source', 'text', 'session_time'])
            output_paths.append(self.csv_path)
        else:
            self.csv_file = None
            self.csv_writer = None
            self.csv_path = None

        logger.info(f"Output files: {', '.join(output_paths)}")

    def _transcription_callback(self, timestamp: float, source: str, text: str):
        """Handle transcription results."""
        if not text.strip():
            return

        self.transcription_count += 1
        session_time = timestamp - self.session_start if self.session_start else 0

        # Create transcription record
        record = {
            "timestamp": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "source": source,
            "text": text.strip(),
            "session_time": f"{session_time:.2f}s"
        }

        # Write to JSON file
        if self.json_file:
            json.dump(record, self.json_file, ensure_ascii=False)
            self.json_file.write('\n')
            self.json_file.flush()

        # Write to CSV file
        if self.csv_writer:
            self.csv_writer.writerow([
                record["timestamp"],
                record["source"],
                record["text"],
                record["session_time"]
            ])
            self.csv_file.flush()

        # Console output
        time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
        print(f"[{time_str}] {source.upper():8s}: {text}")

    def start(self):
        """Start the transcription system."""
        if self.is_running:
            logger.warning("System already running")
            return

        logger.info("Starting reliable live call transcription...")
        self.session_start = time.time()

        try:
            # Setup output files
            self._setup_output_files()

            # Initialize transcription engine
            self.transcription_engine = TranscriptionEngine(
                model_name=self.model_name,
                language=self.language,
                callback=self._transcription_callback
            )

            # Initialize audio capture
            self.audio_capture = AudioCapture(
                sample_rate=self.sample_rate,
                mic_device=self.mic_device,
                system_device=self.system_device
            )

            # Set audio callbacks
            self.audio_capture.set_callbacks(
                self.transcription_engine.process_mic_audio,
                self.transcription_engine.process_desktop_audio
            )

            # Start components
            self.transcription_engine.start()
            self.audio_capture.start()

            self.is_running = True

            # Display status
            print("\n" + "="*60)
            print("RELIABLE LIVE CALL TRANSCRIPTION ACTIVE")
            print("="*60)
            if self.json_path:
                print(f"JSON output: {self.json_path}")
            if self.csv_path:
                print(f"CSV output: {self.csv_path}")
            print(f"Output format: {self.output_format}")
            print(f"Model: {self.model_name} ({self.language})")
            print(f"Microphone device: {self.mic_device}")
            print(f"System audio device: {self.system_device}")
            print("Press Ctrl+C to stop transcription")
            print("="*60)

            logger.info("System started successfully")

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
            # Main loop - monitor system health
            while self.is_running and not self.shutdown_event.is_set():
                self.shutdown_event.wait(5.0)

                if not self.is_running:
                    break

                # Check system health
                if self.audio_capture:
                    health = self.audio_capture.is_healthy()
                    if not health['running']:
                        logger.error("Audio capture stopped unexpectedly")
                        break

                # Print periodic stats
                if self.transcription_engine:
                    stats = self.transcription_engine.get_stats()
                    logger.info(f"Stats: {self.transcription_count} total, "
                              f"mic_buf: {stats['mic_buffer_size']:.1f}s, "
                              f"desktop_buf: {stats['desktop_buffer_size']:.1f}s")

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        """Shutdown the system gracefully."""
        if not self.is_running:
            return

        logger.info("Shutting down transcription system...")
        self.is_running = False
        self.shutdown_event.set()

        # Stop components
        try:
            if self.audio_capture:
                self.audio_capture.stop()

            if self.transcription_engine:
                stats = self.transcription_engine.get_stats()
                logger.info(f"Final stats: {stats}")
                self.transcription_engine.stop()

            # Close output files
            if self.json_file:
                self.json_file.close()
            if self.csv_file:
                self.csv_file.close()

            runtime = time.time() - self.session_start if self.session_start else 0
            logger.info(f"Session completed: {self.transcription_count} transcriptions "
                       f"in {runtime:.1f}s")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

        logger.info("Shutdown completed")

    def __enter__(self):
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
        description="Reliable Live Call Transcription System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available audio devices
  python main.py --list-devices

  # Start transcription with auto-detected devices
  python main.py

  # Specify devices manually
  python main.py --mic-device 1 --system-device 2

  # Output to specific file
  python main.py --output my_call

  # Use different language and model
  python main.py --lang es --model medium

  # Use tiny model for faster processing
  python main.py --model tiny
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
    parser.add_argument('--output-format', type=str, default='json',
                        choices=['json', 'csv', 'both'],
                        help='Output format: json, csv, or both (default: json)')

    # Audio settings
    parser.add_argument('--sample-rate', '-r', type=int, default=16000,
                        help='Audio sample rate in Hz (default: 16000)')

    # Transcription settings
    parser.add_argument('--lang', type=str, default='en',
                        help='Language code for transcription (default: en)')
    parser.add_argument('--model', type=str, default='small',
                        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                        help='Whisper model to use (default: small)')

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
        with LiveTranscriber(
            mic_device=args.mic_device,
            system_device=args.system_device,
            output_file=args.output,
            output_format=args.output_format,
            model_name=args.model,
            language=args.lang,
            sample_rate=args.sample_rate
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