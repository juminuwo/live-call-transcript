"""
Tests for main module.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock, call
import argparse

# Add main module to path for testing
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import LiveCallTranscriber, list_audio_devices, main


class TestLiveCallTranscriber:
    """Test cases for LiveCallTranscriber class."""

    def test_initialization(self):
        """Test LiveCallTranscriber initialization."""
        transcriber = LiveCallTranscriber(
            mic_device=1,
            system_device=2,
            output_file="test_output",
            format_type="csv",
            sample_rate=48000,
            lang="es",
            model="medium",
            console_output=False,
            real_time_display=True
        )

        assert transcriber.mic_device == 1
        assert transcriber.system_device == 2
        assert transcriber.output_file == "test_output"
        assert transcriber.format_type == "csv"
        assert transcriber.sample_rate == 48000
        assert transcriber.lang == "es"
        assert transcriber.model == "medium"
        assert transcriber.console_output is False
        assert transcriber.real_time_display is True
        assert transcriber.is_running is False

    def test_initialization_defaults(self):
        """Test LiveCallTranscriber initialization with defaults."""
        transcriber = LiveCallTranscriber()

        assert transcriber.mic_device is None
        assert transcriber.system_device is None
        assert transcriber.output_file is None
        assert transcriber.format_type == "json"
        assert transcriber.sample_rate == 16000
        assert transcriber.lang == "en"
        assert transcriber.model == "small"
        assert transcriber.console_output is True
        assert transcriber.real_time_display is False

    def test_transcription_callback(self):
        """Test transcription callback handling."""
        transcriber = LiveCallTranscriber()

        # Mock logger and display
        mock_logger = MagicMock()
        mock_logger.session_start = 1234567890.0
        mock_display = MagicMock()

        transcriber.logger = mock_logger
        transcriber.display = mock_display

        # Test callback
        transcriber._transcription_callback(1234567895.5, "desktop", "Test message")

        # Verify logger was called with mapped source
        mock_logger.log_transcription.assert_called_once_with(
            1234567895.5, "desktop", "Test message"
        )

        # Verify display was updated
        mock_display.add_transcription.assert_called_once_with(
            1234567895.5, "desktop", "Test message", 1234567890.0
        )

    def test_transcription_callback_empty_text(self):
        """Test transcription callback with empty text."""
        transcriber = LiveCallTranscriber()
        mock_logger = MagicMock()
        transcriber.logger = mock_logger

        # Callback should return early for empty text
        transcriber._transcription_callback(1234567890.0, "mic", "   ")

        mock_logger.log_transcription.assert_not_called()

    def test_transcription_callback_source_mapping(self):
        """Test source name mapping in transcription callback."""
        transcriber = LiveCallTranscriber()
        mock_logger = MagicMock()
        transcriber.logger = mock_logger

        # Test different source mappings
        transcriber._transcription_callback(1234567890.0, "mic", "Mic test")
        transcriber._transcription_callback(1234567891.0, "system", "System test")
        transcriber._transcription_callback(1234567892.0, "desktop", "Desktop test")

        calls = mock_logger.log_transcription.call_args_list
        assert calls[0][0] == (1234567890.0, "mic", "Mic test")
        assert calls[1][0] == (1234567891.0, "desktop", "System test")  # mapped to desktop
        assert calls[2][0] == (1234567892.0, "desktop", "Desktop test")

    def test_audio_callbacks(self):
        """Test audio callback handling."""
        import numpy as np
        transcriber = LiveCallTranscriber()

        # Mock transcriber component
        mock_transcriber = MagicMock()
        transcriber.transcriber = mock_transcriber

        test_audio = np.array([0.1, 0.2, 0.3])

        # Test mic callback
        transcriber._mic_audio_callback(test_audio, "mic")
        mock_transcriber.process_mic_audio.assert_called_once_with(test_audio)

        # Test system callback
        transcriber._system_audio_callback(test_audio, "system")
        mock_transcriber.process_system_audio.assert_called_once_with(test_audio)

    @patch('live_call_transcript.logger.TranscriptionLogger')
    @patch('live_call_transcript.logger.RealTimeDisplay')
    @patch('live_call_transcript.transcriber.DualStreamTranscriber')
    @patch('live_call_transcript.audio_utils.AudioCapture')
    def test_setup(self, mock_audio, mock_transcriber, mock_display, mock_logger):
        """Test system setup."""
        transcriber = LiveCallTranscriber(real_time_display=True)

        transcriber.setup()

        # Verify all components were created
        mock_logger.assert_called_once()
        mock_display.assert_called_once()
        mock_transcriber.assert_called_once()
        mock_audio.assert_called_once()

        # Verify components are assigned
        assert transcriber.logger is not None
        assert transcriber.display is not None
        assert transcriber.transcriber is not None
        assert transcriber.audio_capture is not None

    def test_setup_without_display(self):
        """Test setup without real-time display."""
        with patch('live_call_transcript.logger.TranscriptionLogger'):
            with patch('live_call_transcript.transcriber.DualStreamTranscriber'):
                with patch('live_call_transcript.audio_utils.AudioCapture'):
                    transcriber = LiveCallTranscriber(real_time_display=False)

                    transcriber.setup()

                    assert transcriber.display is None

    @patch.object(LiveCallTranscriber, 'setup')
    @patch.object(LiveCallTranscriber, 'start')
    def test_context_manager(self, mock_start, mock_setup):
        """Test LiveCallTranscriber as context manager."""
        with patch.object(LiveCallTranscriber, 'shutdown') as mock_shutdown:
            with LiveCallTranscriber() as transcriber:
                assert isinstance(transcriber, LiveCallTranscriber)

            mock_setup.assert_called_once()
            mock_start.assert_called_once()
            mock_shutdown.assert_called_once()

    def test_start(self):
        """Test starting the transcription system."""
        transcriber = LiveCallTranscriber()

        # Mock components
        mock_logger = MagicMock()
        mock_transcriber = MagicMock()
        mock_audio = MagicMock()

        transcriber.logger = mock_logger
        transcriber.transcriber = mock_transcriber
        transcriber.audio_capture = mock_audio

        transcriber.start()

        # Verify components were started
        mock_logger.start.assert_called_once()
        mock_transcriber.start.assert_called_once()
        mock_audio.start_recording.assert_called_once()

        assert transcriber.is_running is True

    def test_shutdown(self):
        """Test shutting down the transcription system."""
        transcriber = LiveCallTranscriber()
        transcriber.is_running = True

        # Mock components
        mock_logger = MagicMock()
        mock_transcriber = MagicMock()
        mock_audio = MagicMock()
        mock_logger.get_statistics.return_value = {"total_entries": 42}

        transcriber.logger = mock_logger
        transcriber.transcriber = mock_transcriber
        transcriber.audio_capture = mock_audio

        transcriber.shutdown()

        # Verify components were stopped
        mock_audio.stop_recording.assert_called_once()
        mock_transcriber.stop.assert_called_once()
        mock_logger.get_statistics.assert_called_once()
        mock_logger.stop.assert_called_once()

        assert transcriber.is_running is False

    def test_run_loop(self):
        """Test main run loop."""
        transcriber = LiveCallTranscriber()
        transcriber.is_running = True

        # Mock shutdown event to trigger loop exit
        transcriber.shutdown_event.set()

        with patch.object(transcriber, 'shutdown') as mock_shutdown:
            transcriber.run()
            mock_shutdown.assert_called_once()

    def test_signal_handler(self):
        """Test signal handler for graceful shutdown."""
        transcriber = LiveCallTranscriber()

        with patch.object(transcriber, 'shutdown') as mock_shutdown:
            transcriber._signal_handler(2, None)  # SIGINT
            mock_shutdown.assert_called_once()


@patch('live_call_transcript.audio_utils.AudioCapture.list_audio_devices')
@patch('live_call_transcript.audio_utils.AudioCapture.find_system_audio_device')
def test_list_audio_devices(mock_find_system, mock_list_devices, audio_device_list):
    """Test listing audio devices function."""
    mock_list_devices.return_value = audio_device_list
    mock_find_system.return_value = 2

    # Capture stdout for testing
    with patch('builtins.print') as mock_print:
        list_audio_devices()

        # Verify print was called multiple times
        assert mock_print.call_count > 5  # Should print device info

        # Check that device info was printed
        call_args = [call[0][0] for call in mock_print.call_args_list if call[0]]
        combined_output = ' '.join(str(arg) for arg in call_args)

        assert 'Default Microphone' in combined_output
        assert 'USB Headset Microphone' in combined_output
        assert 'AUTO-DETECTED' in combined_output


class TestMainFunction:
    """Test cases for main function."""

    @patch('main.list_audio_devices')
    def test_main_list_devices(self, mock_list_devices):
        """Test main function with --list-devices option."""
        test_args = ['main.py', '--list-devices']

        with patch('sys.argv', test_args):
            result = main()

        mock_list_devices.assert_called_once()
        assert result == 0

    @patch('main.LiveCallTranscriber')
    def test_main_normal_operation(self, mock_transcriber_class):
        """Test main function normal operation."""
        mock_transcriber = MagicMock()
        mock_transcriber_class.return_value.__enter__.return_value = mock_transcriber

        test_args = [
            'main.py',
            '--mic-device', '1',
            '--system-device', '2',
            '--output', 'test_output',
            '--format', 'csv',
            '--lang', 'es',
            '--model', 'medium'
        ]

        with patch('sys.argv', test_args):
            result = main()

        # Verify transcriber was created with correct arguments
        mock_transcriber_class.assert_called_once()
        call_kwargs = mock_transcriber_class.call_args[1]

        assert call_kwargs['mic_device'] == 1
        assert call_kwargs['system_device'] == 2
        assert call_kwargs['output_file'] == 'test_output'
        assert call_kwargs['format_type'] == 'csv'
        assert call_kwargs['lang'] == 'es'
        assert call_kwargs['model'] == 'medium'

        # Verify transcriber run was called
        mock_transcriber.run.assert_called_once()

        assert result == 0

    @patch('main.LiveCallTranscriber')
    def test_main_keyboard_interrupt(self, mock_transcriber_class):
        """Test main function with keyboard interrupt."""
        mock_transcriber = MagicMock()
        mock_transcriber.run.side_effect = KeyboardInterrupt()
        mock_transcriber_class.return_value.__enter__.return_value = mock_transcriber

        test_args = ['main.py']

        with patch('sys.argv', test_args):
            result = main()

        assert result == 0

    @patch('main.LiveCallTranscriber')
    def test_main_exception(self, mock_transcriber_class):
        """Test main function with general exception."""
        mock_transcriber_class.side_effect = RuntimeError("Test error")

        test_args = ['main.py']

        with patch('sys.argv', test_args):
            result = main()

        assert result == 1

    def test_main_argument_parsing(self):
        """Test argument parsing in main function."""
        test_args = [
            'main.py',
            '--mic-device', '1',
            '--system-device', '2',
            '--output', 'test_output',
            '--format', 'both',
            '--sample-rate', '48000',
            '--lang', 'fr',
            '--model', 'large',
            '--no-console',
            '--real-time-display',
            '--verbose'
        ]

        with patch('sys.argv', test_args):
            with patch('main.LiveCallTranscriber') as mock_transcriber_class:
                mock_transcriber_class.return_value.__enter__.return_value = MagicMock()

                main()

                # Verify all arguments were parsed correctly
                call_kwargs = mock_transcriber_class.call_args[1]

                assert call_kwargs['mic_device'] == 1
                assert call_kwargs['system_device'] == 2
                assert call_kwargs['output_file'] == 'test_output'
                assert call_kwargs['format_type'] == 'both'
                assert call_kwargs['sample_rate'] == 48000
                assert call_kwargs['lang'] == 'fr'
                assert call_kwargs['model'] == 'large'
                assert call_kwargs['console_output'] is False  # --no-console
                assert call_kwargs['real_time_display'] is True


if __name__ == "__main__":
    pytest.main([__file__])