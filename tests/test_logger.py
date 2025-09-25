"""
Tests for logger module.
"""

import pytest
import json
import csv
import time
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from live_call_transcript.logger import TranscriptionEntry, TranscriptionLogger, RealTimeDisplay


class TestTranscriptionEntry:
    """Test cases for TranscriptionEntry class."""

    def test_initialization(self):
        """Test TranscriptionEntry initialization."""
        timestamp = time.time()
        session_start = timestamp - 10.0

        entry = TranscriptionEntry(timestamp, "mic", "Hello world", session_start)

        assert entry.timestamp == timestamp
        assert entry.source == "mic"
        assert entry.text == "Hello world"
        assert entry.session_start == session_start
        assert entry.relative_time == 10.0

    def test_initialization_default_session_start(self):
        """Test TranscriptionEntry with default session start."""
        timestamp = time.time()

        entry = TranscriptionEntry(timestamp, "desktop", "Test message")

        assert entry.timestamp == timestamp
        assert entry.source == "desktop"
        assert entry.text == "Test message"
        assert entry.session_start <= timestamp
        assert entry.relative_time >= 0

    def test_text_stripping(self):
        """Test that text is stripped of whitespace."""
        timestamp = time.time()

        entry = TranscriptionEntry(timestamp, "mic", "  Hello world  \n")

        assert entry.text == "Hello world"

    def test_to_dict(self):
        """Test conversion to dictionary format."""
        timestamp = time.time()
        session_start = timestamp - 65.123  # 1 minute, 5.123 seconds

        entry = TranscriptionEntry(timestamp, "mic", "Hello world", session_start)
        result = entry.to_dict()

        expected_keys = {"time", "timestamp", "source", "text"}
        assert set(result.keys()) == expected_keys
        assert result["time"] == "00:01:05.123"
        assert result["timestamp"] == timestamp
        assert result["source"] == "mic"
        assert result["text"] == "Hello world"

    def test_to_json(self):
        """Test conversion to JSON string."""
        timestamp = time.time()
        session_start = timestamp - 30.5

        entry = TranscriptionEntry(timestamp, "desktop", "Test message", session_start)
        json_str = entry.to_json()

        # Parse back to verify
        parsed = json.loads(json_str)
        assert parsed["time"] == "00:00:30.500"
        assert parsed["source"] == "desktop"
        assert parsed["text"] == "Test message"

    def test_to_csv_row(self):
        """Test conversion to CSV row format."""
        timestamp = time.time()
        session_start = timestamp - 125.75  # 2 minutes, 5.75 seconds

        entry = TranscriptionEntry(timestamp, "mic", "CSV test", session_start)
        row = entry.to_csv_row()

        assert len(row) == 4
        assert row[0] == "00:02:05.750"
        assert row[1] == str(timestamp)
        assert row[2] == "mic"
        assert row[3] == "CSV test"

    def test_format_time(self):
        """Test time formatting function."""
        # Test various time formats
        assert TranscriptionEntry.format_time(0.0) == "00:00:00.000"
        assert TranscriptionEntry.format_time(1.5) == "00:00:01.500"
        assert TranscriptionEntry.format_time(65.123) == "00:01:05.123"
        assert TranscriptionEntry.format_time(3665.999) == "01:01:05.999"

    def test_string_representation(self):
        """Test string representation for console display."""
        timestamp = time.time()
        session_start = timestamp - 42.123

        entry = TranscriptionEntry(timestamp, "mic", "String test", session_start)
        str_repr = str(entry)

        assert "00:00:42.123" in str_repr
        assert "mic" in str_repr
        assert "String test" in str_repr
        assert str_repr.startswith("[00:00:42.123]")


class TestTranscriptionLogger:
    """Test cases for TranscriptionLogger class."""

    def test_initialization_defaults(self):
        """Test TranscriptionLogger initialization with defaults."""
        logger = TranscriptionLogger()

        assert logger.format_type == "json"
        assert logger.console_output is True
        assert logger.buffer_size == 100
        assert logger.auto_flush is True
        assert logger.is_running is False
        assert logger.entry_count == 0

    def test_initialization_custom_params(self, temp_dir):
        """Test TranscriptionLogger initialization with custom parameters."""
        output_file = os.path.join(temp_dir, "test_output")

        logger = TranscriptionLogger(
            output_file=output_file,
            format_type="csv",
            console_output=False,
            buffer_size=50,
            auto_flush=False
        )

        assert logger.format_type == "csv"
        assert logger.console_output is False
        assert logger.buffer_size == 50
        assert logger.auto_flush is False

    def test_setup_output_files_json(self):
        """Test output file setup for JSON format."""
        with patch('builtins.open', mock_open()):
            logger = TranscriptionLogger(format_type="json")
            logger._setup_output_files("test")

            assert hasattr(logger, 'json_path')
            assert logger.json_path == "test.jsonl"

    def test_setup_output_files_csv(self):
        """Test output file setup for CSV format."""
        logger = TranscriptionLogger(format_type="csv")
        logger._setup_output_files("test")

        assert hasattr(logger, 'csv_path')
        assert logger.csv_path == "test.csv"

    def test_setup_output_files_both(self):
        """Test output file setup for both formats."""
        logger = TranscriptionLogger(format_type="both")
        logger._setup_output_files("test")

        assert hasattr(logger, 'json_path')
        assert hasattr(logger, 'csv_path')
        assert logger.json_path == "test.jsonl"
        assert logger.csv_path == "test.csv"

    def test_setup_output_files_auto_name(self):
        """Test automatic filename generation."""
        with patch('live_call_transcript.logger.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20231225_120000"

            logger = TranscriptionLogger(format_type="json")
            logger._setup_output_files(None)

            assert "transcription_20231225_120000" in logger.json_path

    def test_log_transcription(self, temp_dir):
        """Test logging a transcription entry."""
        output_file = os.path.join(temp_dir, "test_log")
        logger = TranscriptionLogger(output_file=output_file, format_type="json")

        with patch.object(logger, '_write_entry_to_files') as mock_write:
            logger.start()

            timestamp = time.time()
            logger.log_transcription(timestamp, "mic", "Test message")

            # Check that entry was created and queued
            assert logger.entry_count == 1
            assert len(logger.entries_buffer) == 1

            # Verify the entry
            entry = logger.entries_buffer[0]
            assert entry.timestamp == timestamp
            assert entry.source == "mic"
            assert entry.text == "Test message"

            logger.stop()

    def test_context_manager(self, temp_dir):
        """Test TranscriptionLogger as context manager."""
        output_file = os.path.join(temp_dir, "test_context")

        with patch.object(TranscriptionLogger, '_open_files'):
            with patch.object(TranscriptionLogger, '_close_files'):
                with TranscriptionLogger(output_file=output_file) as logger:
                    assert logger.is_running is True

                assert logger.is_running is False

    def test_flush_buffer(self):
        """Test buffer flushing."""
        logger = TranscriptionLogger()
        logger.start()

        # Add some entries to buffer
        timestamp = time.time()
        logger.log_transcription(timestamp, "mic", "Message 1")
        logger.log_transcription(timestamp + 1, "desktop", "Message 2")

        assert len(logger.entries_buffer) == 2

        logger.flush_buffer()

        assert len(logger.entries_buffer) == 0
        logger.stop()

    def test_get_statistics(self):
        """Test getting logger statistics."""
        logger = TranscriptionLogger()
        logger.start()

        stats = logger.get_statistics()

        expected_keys = {
            "session_start", "session_duration", "total_entries",
            "buffered_entries", "queue_size"
        }
        assert set(stats.keys()) == expected_keys
        assert stats["total_entries"] == 0
        assert stats["buffered_entries"] == 0
        assert stats["session_duration"] >= 0

        logger.stop()

    def test_write_entry_to_files_json(self, temp_dir):
        """Test writing entry to JSON file."""
        output_file = os.path.join(temp_dir, "test_write")
        logger = TranscriptionLogger(output_file=output_file, format_type="json")

        # Setup files manually for testing
        logger._setup_output_files(output_file)
        logger._open_files()

        # Create and write entry
        timestamp = time.time()
        entry = TranscriptionEntry(timestamp, "mic", "Test write", timestamp - 10)
        logger._write_entry_to_files(entry)

        logger._close_files()

        # Verify file contents
        with open(logger.json_path, 'r') as f:
            content = f.read().strip()
            parsed = json.loads(content)
            assert parsed["source"] == "mic"
            assert parsed["text"] == "Test write"

    def test_write_entry_to_files_csv(self, temp_dir):
        """Test writing entry to CSV file."""
        output_file = os.path.join(temp_dir, "test_write_csv")
        logger = TranscriptionLogger(output_file=output_file, format_type="csv")

        # Setup files manually for testing
        logger._setup_output_files(output_file)
        logger._open_files()

        # Create and write entry
        timestamp = time.time()
        entry = TranscriptionEntry(timestamp, "desktop", "CSV write test", timestamp - 5)
        logger._write_entry_to_files(entry)

        logger._close_files()

        # Verify file contents
        with open(logger.csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 2  # Header + 1 data row
            assert rows[0] == ["Time", "Timestamp", "Source", "Text"]
            assert rows[1][2] == "desktop"
            assert rows[1][3] == "CSV write test"


class TestRealTimeDisplay:
    """Test cases for RealTimeDisplay class."""

    def test_initialization(self):
        """Test RealTimeDisplay initialization."""
        display = RealTimeDisplay(max_lines=10, show_timestamps=False)

        assert display.max_lines == 10
        assert display.show_timestamps is False
        assert len(display.lines) == 0

    def test_add_transcription_with_timestamps(self):
        """Test adding transcription with timestamps."""
        display = RealTimeDisplay(max_lines=5, show_timestamps=True)

        timestamp = time.time()
        session_start = timestamp - 30

        with patch.object(display, '_redraw') as mock_redraw:
            display.add_transcription(timestamp, "mic", "Test message", session_start)

            assert len(display.lines) == 1
            assert "00:00:30.000" in display.lines[0]
            assert "mic" in display.lines[0]
            assert "Test message" in display.lines[0]
            mock_redraw.assert_called_once()

    def test_add_transcription_without_timestamps(self):
        """Test adding transcription without timestamps."""
        display = RealTimeDisplay(max_lines=5, show_timestamps=False)

        timestamp = time.time()
        session_start = timestamp - 15

        with patch.object(display, '_redraw') as mock_redraw:
            display.add_transcription(timestamp, "desktop", "No timestamp", session_start)

            assert len(display.lines) == 1
            assert "desktop" in display.lines[0]
            assert "No timestamp" in display.lines[0]
            assert "00:00:15" not in display.lines[0]  # No timestamp
            mock_redraw.assert_called_once()

    def test_max_lines_limit(self):
        """Test that display respects max_lines limit."""
        display = RealTimeDisplay(max_lines=3, show_timestamps=False)

        timestamp = time.time()
        session_start = timestamp

        with patch.object(display, '_redraw'):
            # Add more lines than the limit
            for i in range(5):
                display.add_transcription(
                    timestamp + i, "mic", f"Message {i}", session_start
                )

            # Should only keep the last 3 lines
            assert len(display.lines) == 3
            assert "Message 2" in display.lines[0]
            assert "Message 3" in display.lines[1]
            assert "Message 4" in display.lines[2]

    @patch('live_call_transcript.logger.os.system')
    def test_redraw(self, mock_system):
        """Test console redraw functionality."""
        display = RealTimeDisplay(max_lines=2, show_timestamps=True)
        display.lines = ["Line 1", "Line 2"]

        display._redraw()

        # Should call system clear command
        mock_system.assert_called_once()

    def test_clear(self):
        """Test clearing the display."""
        display = RealTimeDisplay()
        display.lines = ["Line 1", "Line 2", "Line 3"]

        with patch.object(display, '_redraw') as mock_redraw:
            display.clear()

            assert len(display.lines) == 0
            mock_redraw.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])