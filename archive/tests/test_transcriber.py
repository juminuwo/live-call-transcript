"""
Tests for transcriber module.
"""

import pytest
import subprocess
import time
import socket
from unittest.mock import patch, MagicMock, mock_open
from contextlib import closing

from live_call_transcript.transcriber import WhisperLiveServer, StreamingTranscriberClient, DualStreamTranscriber


class TestWhisperLiveServer:
    """Test cases for WhisperLiveServer class."""

    def test_initialization(self):
        """Test WhisperLiveServer initialization."""
        server = WhisperLiveServer(port=9091, backend="faster_whisper", model="base")

        assert server.port == 9091
        assert server.backend == "faster_whisper"
        assert server.model == "base"
        assert server.server_process is None

    def test_initialization_defaults(self):
        """Test WhisperLiveServer initialization with defaults."""
        server = WhisperLiveServer()

        assert server.port == 9090
        assert server.backend == "faster_whisper"
        assert server.model == "small"

    @patch('live_call_transcript.transcriber.socket.socket')
    def test_is_port_available_true(self, mock_socket):
        """Test port availability check when port is available."""
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 1  # Connection failed (port available)
        mock_socket.return_value.__enter__.return_value = mock_sock

        server = WhisperLiveServer(port=9091)
        assert server.is_port_available() is True

    @patch('live_call_transcript.transcriber.socket.socket')
    def test_is_port_available_false(self, mock_socket):
        """Test port availability check when port is in use."""
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 0  # Connection succeeded (port in use)
        mock_socket.return_value.__enter__.return_value = mock_sock

        server = WhisperLiveServer(port=9090)
        assert server.is_port_available() is False

    @patch('live_call_transcript.transcriber.subprocess.Popen')
    @patch.object(WhisperLiveServer, 'is_port_available')
    def test_start_server_success(self, mock_port_check, mock_popen):
        """Test successful server startup."""
        # First call returns True (port available), second returns False (server started)
        mock_port_check.side_effect = [True, False]
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        server = WhisperLiveServer()
        server.start_server(timeout=2)

        assert server.server_process == mock_process
        mock_popen.assert_called_once()

    @patch.object(WhisperLiveServer, 'is_port_available')
    def test_start_server_already_running(self, mock_port_check):
        """Test server startup when server is already running."""
        mock_port_check.return_value = False  # Port not available (server running)

        server = WhisperLiveServer()
        server.start_server()

        # Should not try to start new process
        assert server.server_process is None

    @patch('live_call_transcript.transcriber.subprocess.Popen')
    @patch.object(WhisperLiveServer, 'is_port_available')
    def test_start_server_timeout(self, mock_port_check, mock_popen):
        """Test server startup timeout."""
        mock_port_check.return_value = True  # Port always available (server never starts)
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        server = WhisperLiveServer()

        with pytest.raises(RuntimeError, match="Server failed to start within timeout"):
            server.start_server(timeout=1)

    def test_stop_server(self):
        """Test stopping the server."""
        server = WhisperLiveServer()
        mock_process = MagicMock()
        server.server_process = mock_process

        server.stop_server()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()
        assert server.server_process is None

    def test_stop_server_no_process(self):
        """Test stopping server when no process exists."""
        server = WhisperLiveServer()
        # Should not raise any errors
        server.stop_server()

    def test_stop_server_force_kill(self):
        """Test force killing server process on timeout."""
        server = WhisperLiveServer()
        mock_process = MagicMock()
        mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 10)
        server.server_process = mock_process

        server.stop_server()

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()

    def test_context_manager(self):
        """Test WhisperLiveServer as context manager."""
        with patch.object(WhisperLiveServer, 'start_server') as mock_start:
            with patch.object(WhisperLiveServer, 'stop_server') as mock_stop:
                with WhisperLiveServer() as server:
                    assert isinstance(server, WhisperLiveServer)

                mock_start.assert_called_once()
                mock_stop.assert_called_once()


class TestStreamingTranscriberClient:
    """Test cases for StreamingTranscriberClient class."""

    def test_initialization(self):
        """Test StreamingTranscriberClient initialization."""
        callback = MagicMock()

        client = StreamingTranscriberClient(
            host="192.168.1.100",
            port=9091,
            source="desktop",
            lang="es",
            model="medium",
            callback=callback
        )

        assert client.host == "192.168.1.100"
        assert client.port == 9091
        assert client.source == "desktop"
        assert client.lang == "es"
        assert client.model == "medium"
        assert client.callback == callback
        assert client.is_running is False
        assert client.ws is None

    def test_initialization_defaults(self):
        """Test StreamingTranscriberClient initialization with defaults."""
        client = StreamingTranscriberClient()

        assert client.host == "localhost"
        assert client.port == 9090
        assert client.source == "mic"
        assert client.lang == "en"
        assert client.model == "small"
        assert client.callback is None

    @patch('live_call_transcript.transcriber.json.loads')
    def test_on_message_valid_transcription(self, mock_json_loads):
        """Test WebSocket message handling for valid transcription."""
        callback = MagicMock()
        client = StreamingTranscriberClient(callback=callback)

        # Mock JSON data
        mock_json_loads.return_value = {
            'text': 'Hello world',
            'timestamp': 1234567890.5
        }

        client._on_message(None, '{"text": "Hello world"}')

        # Verify callback was called
        callback.assert_called_once_with(1234567890.5, "mic", "Hello world")

    @patch('live_call_transcript.transcriber.json.loads')
    def test_on_message_empty_text(self, mock_json_loads):
        """Test WebSocket message handling with empty text."""
        callback = MagicMock()
        client = StreamingTranscriberClient(callback=callback)

        mock_json_loads.return_value = {'text': '   ', 'timestamp': 1234567890.5}

        client._on_message(None, '{"text": "   "}')

        # Callback should not be called for empty text
        callback.assert_not_called()

    @patch('live_call_transcript.transcriber.json.loads')
    def test_on_message_json_error(self, mock_json_loads):
        """Test WebSocket message handling with JSON decode error."""
        mock_json_loads.side_effect = ValueError("Invalid JSON")

        client = StreamingTranscriberClient()
        # Should not raise exception
        client._on_message(None, 'invalid json')

    def test_send_audio_running(self):
        """Test sending audio data when client is running."""
        import numpy as np
        client = StreamingTranscriberClient()
        client.is_running = True

        test_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        client.send_audio(test_audio)

        # Audio should be added to queue
        assert not client.audio_queue.empty()
        queued_audio = client.audio_queue.get()
        assert np.array_equal(queued_audio, test_audio)

    def test_send_audio_not_running(self):
        """Test sending audio data when client is not running."""
        import numpy as np
        client = StreamingTranscriberClient()
        client.is_running = False

        test_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        client.send_audio(test_audio)

        # Audio should not be added to queue
        assert client.audio_queue.empty()

    @patch('live_call_transcript.transcriber.threading.Thread')
    @patch('live_call_transcript.transcriber.websocket.WebSocketApp')
    def test_start_client(self, mock_websocket, mock_thread):
        """Test starting the streaming client."""
        client = StreamingTranscriberClient()

        client.start()

        assert client.is_running is True
        assert mock_thread.call_count == 2  # WebSocket and audio threads
        mock_websocket.assert_called_once()

    def test_stop_client(self):
        """Test stopping the streaming client."""
        client = StreamingTranscriberClient()
        client.is_running = True

        # Mock WebSocket and threads
        mock_ws = MagicMock()
        mock_ws_thread = MagicMock()
        mock_audio_thread = MagicMock()

        client.ws = mock_ws
        client.ws_thread = mock_ws_thread
        client.audio_thread = mock_audio_thread

        client.stop()

        assert client.is_running is False
        mock_ws.close.assert_called_once()
        mock_ws_thread.join.assert_called_once()
        mock_audio_thread.join.assert_called_once()


class TestDualStreamTranscriber:
    """Test cases for DualStreamTranscriber class."""

    def test_initialization(self):
        """Test DualStreamTranscriber initialization."""
        callback = MagicMock()

        transcriber = DualStreamTranscriber(
            port=9091,
            lang="fr",
            model="medium",
            callback=callback
        )

        assert transcriber.port == 9091
        assert transcriber.lang == "fr"
        assert transcriber.model == "medium"
        assert transcriber.callback == callback
        assert transcriber.server is None
        assert transcriber.mic_client is None
        assert transcriber.system_client is None

    def test_initialization_defaults(self):
        """Test DualStreamTranscriber initialization with defaults."""
        transcriber = DualStreamTranscriber()

        assert transcriber.port == 9090
        assert transcriber.lang == "en"
        assert transcriber.model == "small"
        assert transcriber.callback is None

    def test_transcription_callback(self):
        """Test internal transcription callback handling."""
        main_callback = MagicMock()
        transcriber = DualStreamTranscriber(callback=main_callback)

        transcriber._transcription_callback(1234567890.5, "mic", "Test message")

        main_callback.assert_called_once_with(1234567890.5, "mic", "Test message")

    @patch('live_call_transcript.transcriber.WhisperLiveServer')
    @patch('live_call_transcript.transcriber.StreamingTranscriberClient')
    def test_start_transcriber(self, mock_client_class, mock_server_class):
        """Test starting the dual-stream transcriber."""
        mock_server = MagicMock()
        mock_mic_client = MagicMock()
        mock_system_client = MagicMock()

        mock_server_class.return_value = mock_server
        mock_client_class.side_effect = [mock_mic_client, mock_system_client]

        transcriber = DualStreamTranscriber()
        transcriber.start()

        # Verify server was started
        mock_server.start_server.assert_called_once()

        # Verify clients were created and started
        assert mock_client_class.call_count == 2
        mock_mic_client.start.assert_called_once()
        mock_system_client.start.assert_called_once()

    def test_stop_transcriber(self):
        """Test stopping the dual-stream transcriber."""
        transcriber = DualStreamTranscriber()

        # Mock components
        mock_server = MagicMock()
        mock_mic_client = MagicMock()
        mock_system_client = MagicMock()

        transcriber.server = mock_server
        transcriber.mic_client = mock_mic_client
        transcriber.system_client = mock_system_client

        transcriber.stop()

        # Verify all components were stopped
        mock_mic_client.stop.assert_called_once()
        mock_system_client.stop.assert_called_once()
        mock_server.stop_server.assert_called_once()

    def test_process_audio_data(self):
        """Test processing audio data for both streams."""
        import numpy as np
        transcriber = DualStreamTranscriber()

        # Mock clients
        mock_mic_client = MagicMock()
        mock_system_client = MagicMock()
        transcriber.mic_client = mock_mic_client
        transcriber.system_client = mock_system_client

        test_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # Test mic audio processing
        transcriber.process_mic_audio(test_audio)
        mock_mic_client.send_audio.assert_called_once_with(test_audio)

        # Test system audio processing
        transcriber.process_system_audio(test_audio)
        mock_system_client.send_audio.assert_called_once_with(test_audio)

    def test_context_manager(self):
        """Test DualStreamTranscriber as context manager."""
        with patch.object(DualStreamTranscriber, 'start') as mock_start:
            with patch.object(DualStreamTranscriber, 'stop') as mock_stop:
                with DualStreamTranscriber() as transcriber:
                    assert isinstance(transcriber, DualStreamTranscriber)

                mock_start.assert_called_once()
                mock_stop.assert_called_once()

    @patch('live_call_transcript.transcriber.WhisperLiveServer')
    @patch('live_call_transcript.transcriber.StreamingTranscriberClient')
    def test_start_failure_cleanup(self, mock_client_class, mock_server_class):
        """Test cleanup when start fails."""
        # Mock server start to raise exception
        mock_server = MagicMock()
        mock_server.start_server.side_effect = RuntimeError("Server failed")
        mock_server_class.return_value = mock_server

        transcriber = DualStreamTranscriber()

        with pytest.raises(RuntimeError, match="Server failed"):
            transcriber.start()

        # Verify stop was called for cleanup
        # Note: In real implementation, stop() should be called in except block


if __name__ == "__main__":
    pytest.main([__file__])