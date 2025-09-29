"""
Logging utilities for live call transcription.
Handles JSON formatting, file output, and real-time display of transcription results.
"""

import json
import csv
import time
import threading
import queue
from datetime import datetime, timezone
from typing import Optional, Dict, List, TextIO, Union
from pathlib import Path
import logging
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptionEntry:
    """Represents a single transcription entry."""

    def __init__(self, timestamp: float, source: str, text: str, session_start: Optional[float] = None):
        self.timestamp = timestamp
        self.source = source
        self.text = text.strip()
        self.session_start = session_start or time.time()

        # Calculate relative time from session start
        # Handle cases where timestamp might be relative or absolute
        if timestamp < self.session_start and (self.session_start - timestamp) > 86400:
            # If timestamp is much smaller than session start, treat it as relative
            self.relative_time = timestamp
        else:
            # Normal case: timestamp is absolute
            self.relative_time = max(0, timestamp - self.session_start)

    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "time": self.format_time(self.relative_time),
            "timestamp": self.timestamp,
            "source": self.source,
            "text": self.text
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_csv_row(self) -> List[str]:
        """Convert to CSV row format."""
        return [
            self.format_time(self.relative_time),
            str(self.timestamp),
            self.source,
            self.text
        ]

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time in HH:MM:SS.ms format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    def __str__(self) -> str:
        """String representation for console display."""
        time_str = self.format_time(self.relative_time)
        return f"[{time_str}] {self.source:7s}: {self.text}"


class TranscriptionLogger:
    """Handles logging and formatting of transcription results."""

    def __init__(
        self,
        output_file: Optional[str] = None,
        format_type: str = "json",
        console_output: bool = True,
        buffer_size: int = 100,
        auto_flush: bool = True
    ):
        """
        Initialize the transcription logger.

        Args:
            output_file: Path to output file. If None, creates timestamped filename.
            format_type: Output format ("json", "csv", or "both").
            console_output: Whether to display results in console.
            buffer_size: Number of entries to buffer before writing to disk.
            auto_flush: Whether to auto-flush after each write.
        """
        self.format_type = format_type
        self.console_output = console_output
        self.buffer_size = buffer_size
        self.auto_flush = auto_flush

        # Session management
        self.session_start = time.time()
        self.entry_count = 0
        self.entries_buffer: List[TranscriptionEntry] = []

        # File handles
        self.json_file: Optional[TextIO] = None
        self.csv_file: Optional[TextIO] = None
        self.csv_writer = None

        # Threading
        self.write_queue = queue.Queue()
        self.is_running = False
        self.write_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # Setup output files
        self._setup_output_files(output_file)

    def _setup_output_files(self, output_file: Optional[str]):
        """Setup output files based on format type."""
        # Generate base filename if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"transcription_{timestamp}"
        else:
            base_name = Path(output_file).stem

        # Setup files based on format type
        if self.format_type in ["json", "both"]:
            self.json_path = f"{base_name}.jsonl"
            logger.info(f"JSON output: {self.json_path}")

        if self.format_type in ["csv", "both"]:
            self.csv_path = f"{base_name}.csv"
            logger.info(f"CSV output: {self.csv_path}")

    def _open_files(self):
        """Open output files for writing."""
        try:
            # Open JSON file
            if self.format_type in ["json", "both"]:
                self.json_file = open(self.json_path, 'w', encoding='utf-8')

            # Open CSV file
            if self.format_type in ["csv", "both"]:
                self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
                self.csv_writer = csv.writer(self.csv_file)
                # Write CSV header
                self.csv_writer.writerow(["Time", "Timestamp", "Source", "Text"])
                if self.auto_flush:
                    self.csv_file.flush()

        except Exception as e:
            logger.error(f"Failed to open output files: {e}")
            raise

    def _close_files(self):
        """Close output files."""
        if self.json_file:
            self.json_file.close()
            self.json_file = None

        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def _write_worker(self):
        """Background thread for writing entries to files."""
        while self.is_running or not self.write_queue.empty():
            try:
                entry = self.write_queue.get(timeout=1.0)
                self._write_entry_to_files(entry)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Write worker error: {e}")

    def _write_entry_to_files(self, entry: TranscriptionEntry):
        """Write a single entry to output files."""
        try:
            # Write to JSON file
            if self.json_file:
                self.json_file.write(entry.to_json() + '\n')
                if self.auto_flush:
                    self.json_file.flush()

            # Write to CSV file
            if self.csv_writer:
                self.csv_writer.writerow(entry.to_csv_row())
                if self.auto_flush:
                    self.csv_file.flush()

        except Exception as e:
            logger.error(f"Failed to write entry: {e}")

    def start(self):
        """Start the logger."""
        if self.is_running:
            return

        self.is_running = True
        self._open_files()

        # Start background write thread
        self.write_thread = threading.Thread(target=self._write_worker, daemon=True)
        self.write_thread.start()

        logger.info("Transcription logger started")

    def stop(self):
        """Stop the logger and ensure all data is written."""
        if not self.is_running:
            return

        self.is_running = False

        # Flush any remaining entries
        self.flush_buffer()

        # Wait for write thread to finish
        if self.write_thread:
            self.write_thread.join(timeout=5.0)

        self._close_files()
        logger.info("Transcription logger stopped")

    def log_transcription(self, timestamp: float, source: str, text: str):
        """Log a transcription result."""
        if not self.is_running:
            logger.warning("Logger not started, ignoring entry")
            return

        # Create entry
        entry = TranscriptionEntry(timestamp, source, text, self.session_start)

        with self.lock:
            self.entry_count += 1
            self.entries_buffer.append(entry)

        # Console output
        if self.console_output:
            print(str(entry))

        # Add to write queue
        self.write_queue.put(entry)

        # Auto-flush buffer if it's full
        if len(self.entries_buffer) >= self.buffer_size:
            self.flush_buffer()

    def flush_buffer(self):
        """Flush the entries buffer."""
        with self.lock:
            if self.entries_buffer:
                logger.debug(f"Flushing {len(self.entries_buffer)} entries")
                self.entries_buffer.clear()

    def get_statistics(self) -> Dict:
        """Get logging statistics."""
        with self.lock:
            return {
                "session_start": self.session_start,
                "session_duration": time.time() - self.session_start,
                "total_entries": self.entry_count,
                "buffered_entries": len(self.entries_buffer),
                "queue_size": self.write_queue.qsize()
            }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class RealTimeDisplay:
    """Real-time console display for transcription results."""

    def __init__(self, max_lines: int = 20, show_timestamps: bool = True):
        self.max_lines = max_lines
        self.show_timestamps = show_timestamps
        self.lines: List[str] = []
        self.lock = threading.Lock()

    def add_transcription(self, timestamp: float, source: str, text: str, session_start: float):
        """Add a transcription result to the display."""
        entry = TranscriptionEntry(timestamp, source, text, session_start)

        with self.lock:
            # Add new line
            if self.show_timestamps:
                line = str(entry)
            else:
                line = f"{source:7s}: {text}"

            self.lines.append(line)

            # Keep only the last max_lines
            if len(self.lines) > self.max_lines:
                self.lines.pop(0)

            # Clear screen and redraw
            self._redraw()

    def _redraw(self):
        """Redraw the console display."""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')

        print("=" * 80)
        print("LIVE CALL TRANSCRIPTION")
        print("=" * 80)
        print()

        for line in self.lines:
            print(line)

        print()
        print("Press Ctrl+C to stop...")

    def clear(self):
        """Clear the display."""
        with self.lock:
            self.lines.clear()
            self._redraw()


def demo_logger():
    """Demonstrate the logging functionality."""
    print("Testing transcription logger...")

    def generate_test_data():
        """Generate test transcription data."""
        test_transcriptions = [
            ("mic", "Hello, can you hear me?"),
            ("desktop", "Yes, I can hear you clearly."),
            ("mic", "Great! Let's start the meeting."),
            ("desktop", "Sounds good. I have the agenda ready."),
            ("mic", "Perfect. Let's begin with the first item."),
            ("desktop", "The first item is about the project timeline."),
            ("mic", "According to our estimates, we need three weeks."),
            ("desktop", "That seems reasonable given the scope."),
        ]

        start_time = time.time()
        for i, (source, text) in enumerate(test_transcriptions):
            timestamp = start_time + (i * 2.0)  # 2 seconds apart
            yield timestamp, source, text

    # Test with different formats
    for format_type in ["json", "csv", "both"]:
        print(f"\nTesting {format_type} format...")

        with TranscriptionLogger(
            output_file=f"test_{format_type}",
            format_type=format_type,
            console_output=True
        ) as logger_instance:

            for timestamp, source, text in generate_test_data():
                logger_instance.log_transcription(timestamp, source, text)
                time.sleep(0.5)  # Brief pause for demonstration

            stats = logger_instance.get_statistics()
            print(f"\nStatistics: {stats}")

    print("\nDemo completed. Check the output files.")


if __name__ == "__main__":
    demo_logger()