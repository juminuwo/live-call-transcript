"""
Tests for audio_utils module.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import threading
import time

from live_call_transcript.audio_utils import AudioCapture


class TestAudioCapture:
    """Test cases for AudioCapture class."""

    def test_initialization(self):
        """Test AudioCapture initialization with default parameters."""
        capture = AudioCapture()

        assert capture.sample_rate == 16000
        assert capture.channels == 1
        assert capture.chunk_duration == 0.5
        assert capture.chunk_size == 8000  # 16000 * 0.5
        assert capture.is_recording is False
        assert capture.mic_device is None
        assert capture.system_device is None

    def test_initialization_custom_params(self):
        """Test AudioCapture initialization with custom parameters."""
        capture = AudioCapture(
            sample_rate=48000,
            channels=2,
            chunk_duration=1.0,
            mic_device=1,
            system_device=2
        )

        assert capture.sample_rate == 48000
        assert capture.channels == 2
        assert capture.chunk_duration == 1.0
        assert capture.chunk_size == 48000  # 48000 * 1.0
        assert capture.mic_device == 1
        assert capture.system_device == 2

    @patch('live_call_transcript.audio_utils.sd.query_devices')
    def test_list_audio_devices(self, mock_query_devices, audio_device_list):
        """Test listing audio devices."""
        # Mock device data
        mock_devices = [
            {'name': 'Default Microphone', 'max_input_channels': 1, 'max_output_channels': 0, 'default_samplerate': 44100},
            {'name': 'Default Speakers', 'max_input_channels': 0, 'max_output_channels': 2, 'default_samplerate': 44100},
            {'name': 'USB Headset', 'max_input_channels': 1, 'max_output_channels': 2, 'default_samplerate': 48000},
        ]
        mock_query_devices.return_value = mock_devices

        devices = AudioCapture.list_audio_devices()

        assert 'input' in devices
        assert 'output' in devices
        assert len(devices['input']) == 2  # Default Microphone and USB Headset
        assert len(devices['output']) == 2  # Default Speakers and USB Headset

        # Check input device structure
        input_device = devices['input'][0]
        assert 'id' in input_device
        assert 'name' in input_device
        assert 'channels' in input_device
        assert 'sample_rate' in input_device

    @patch('live_call_transcript.audio_utils.sd.query_devices')
    @patch('live_call_transcript.audio_utils.platform.system')
    def test_find_system_audio_device_windows(self, mock_system, mock_query_devices):
        """Test finding system audio device on Windows."""
        mock_system.return_value = 'Windows'
        mock_devices = [
            {'name': 'Default Microphone', 'max_input_channels': 1},
            {'name': 'Stereo Mix', 'max_input_channels': 2},
            {'name': 'Default Speakers', 'max_input_channels': 0},
        ]
        mock_query_devices.return_value = mock_devices

        device_id = AudioCapture.find_system_audio_device()

        assert device_id == 1  # Stereo Mix device

    @patch('live_call_transcript.audio_utils.sd.query_devices')
    @patch('live_call_transcript.audio_utils.platform.system')
    def test_find_system_audio_device_linux(self, mock_system, mock_query_devices):
        """Test finding system audio device on Linux."""
        mock_system.return_value = 'Linux'
        mock_devices = [
            {'name': 'Default Microphone', 'max_input_channels': 1},
            {'name': 'Monitor of Built-in Audio', 'max_input_channels': 2},
            {'name': 'Default Speakers', 'max_input_channels': 0},
        ]
        mock_query_devices.return_value = mock_devices

        device_id = AudioCapture.find_system_audio_device()

        assert device_id == 1  # Monitor device

    @patch('live_call_transcript.audio_utils.sd.query_devices')
    @patch('live_call_transcript.audio_utils.platform.system')
    def test_find_system_audio_device_not_found(self, mock_system, mock_query_devices):
        """Test when no system audio device is found."""
        mock_system.return_value = 'Windows'
        mock_devices = [
            {'name': 'Default Microphone', 'max_input_channels': 1},
            {'name': 'Default Speakers', 'max_input_channels': 0},
        ]
        mock_query_devices.return_value = mock_devices

        device_id = AudioCapture.find_system_audio_device()

        assert device_id is None

    def test_set_callbacks(self):
        """Test setting audio callbacks."""
        capture = AudioCapture()

        def mock_mic_callback(data, source):
            pass

        def mock_system_callback(data, source):
            pass

        capture.set_callbacks(mock_mic_callback, mock_system_callback)

        assert capture.mic_callback == mock_mic_callback
        assert capture.system_callback == mock_system_callback

    def test_audio_callback_processing(self):
        """Test audio callback data processing."""
        capture = AudioCapture()

        # Test stereo to mono conversion
        stereo_data = np.array([[0.5, -0.5], [0.3, -0.3]], dtype=np.float32)
        capture._mic_audio_callback(stereo_data, None, None, None)

        # Check that data was added to queue
        assert not capture.mic_queue.empty()

        # Get and verify processed data
        processed_data = capture.mic_queue.get()
        expected_mono = np.mean(stereo_data, axis=1)
        np.testing.assert_array_almost_equal(processed_data, expected_mono)

    def test_audio_callback_mono_data(self):
        """Test audio callback with mono data."""
        capture = AudioCapture()

        # Test mono data (no conversion needed)
        mono_data = np.array([[0.5], [0.3]], dtype=np.float32)
        capture._system_audio_callback(mono_data, None, None, None)

        # Check that data was added to queue
        assert not capture.system_queue.empty()

        # Get and verify processed data
        processed_data = capture.system_queue.get()
        expected_flat = mono_data.flatten()
        np.testing.assert_array_almost_equal(processed_data, expected_flat)

    def test_queue_processing_thread(self):
        """Test audio queue processing in separate thread."""
        capture = AudioCapture()
        processed_data = []

        def mock_callback(data, source):
            processed_data.append((data, source))

        capture.mic_callback = mock_callback
        capture.is_recording = True

        # Add test data to queue
        test_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        capture.mic_queue.put(test_data)

        # Run processing thread briefly
        thread = threading.Thread(target=capture._process_mic_queue, daemon=True)
        thread.start()
        time.sleep(0.1)  # Let it process
        capture.is_recording = False
        thread.join(timeout=1.0)

        # Verify callback was called
        assert len(processed_data) == 1
        np.testing.assert_array_equal(processed_data[0][0], test_data)
        assert processed_data[0][1] == 'mic'

    @patch('live_call_transcript.audio_utils.sd.InputStream')
    def test_start_recording_success(self, mock_input_stream):
        """Test successful start of recording."""
        capture = AudioCapture(system_device=2)  # Specify system device

        # Mock stream instances
        mock_mic_stream = MagicMock()
        mock_system_stream = MagicMock()
        mock_input_stream.side_effect = [mock_mic_stream, mock_system_stream]

        capture.start_recording()

        assert capture.is_recording is True
        assert mock_input_stream.call_count == 2
        mock_mic_stream.start.assert_called_once()
        mock_system_stream.start.assert_called_once()

    def test_start_recording_no_system_device(self):
        """Test start recording fails when no system device found."""
        capture = AudioCapture()

        with patch.object(capture, 'find_system_audio_device', return_value=None):
            with pytest.raises(RuntimeError, match="System audio device not found"):
                capture.start_recording()

    def test_context_manager(self):
        """Test AudioCapture as context manager."""
        with patch.object(AudioCapture, 'start_recording') as mock_start:
            with patch.object(AudioCapture, 'stop_recording') as mock_stop:
                with AudioCapture() as capture:
                    assert isinstance(capture, AudioCapture)

                mock_stop.assert_called_once()

    def test_stop_recording(self):
        """Test stopping recording."""
        capture = AudioCapture()
        capture.is_recording = True

        # Mock streams
        mock_mic_stream = MagicMock()
        mock_system_stream = MagicMock()
        capture.mic_stream = mock_mic_stream
        capture.system_stream = mock_system_stream

        # Mock threads
        mock_thread1 = MagicMock()
        mock_thread2 = MagicMock()
        capture.threads = [mock_thread1, mock_thread2]

        capture.stop_recording()

        assert capture.is_recording is False
        mock_mic_stream.stop.assert_called_once()
        mock_mic_stream.close.assert_called_once()
        mock_system_stream.stop.assert_called_once()
        mock_system_stream.close.assert_called_once()
        mock_thread1.join.assert_called_once()
        mock_thread2.join.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])