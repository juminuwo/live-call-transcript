"""
Pytest configuration and fixtures for Live Call Transcription tests.
"""

import pytest
import tempfile
import os
import sys
import numpy as np
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from live_call_transcript.logger import TranscriptionEntry


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    sample_rate = 16000
    duration = 1.0  # 1 second
    frequency = 440  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    return audio_data, sample_rate


@pytest.fixture
def sample_transcription_entries():
    """Generate sample transcription entries for testing."""
    import time
    session_start = time.time()

    entries = [
        TranscriptionEntry(session_start + 1.0, "mic", "Hello world", session_start),
        TranscriptionEntry(session_start + 2.5, "desktop", "How are you?", session_start),
        TranscriptionEntry(session_start + 4.0, "mic", "I'm doing well, thanks!", session_start),
        TranscriptionEntry(session_start + 6.2, "desktop", "That's great to hear", session_start),
    ]

    return entries, session_start


@pytest.fixture
def mock_websocket():
    """Mock WebSocket for testing transcription clients."""
    class MockWebSocket:
        def __init__(self):
            self.connected = True
            self.sent_messages = []
            self.binary_messages = []

        def send(self, data, opcode=None):
            if opcode is not None:
                self.binary_messages.append(data)
            else:
                self.sent_messages.append(data)

        def close(self):
            self.connected = False

        @property
        def sock(self):
            return self if self.connected else None

    return MockWebSocket()


@pytest.fixture
def audio_device_list():
    """Mock audio device list for testing."""
    return {
        'input': [
            {'id': 0, 'name': 'Default Microphone', 'channels': 1, 'sample_rate': 44100.0},
            {'id': 1, 'name': 'USB Headset Microphone', 'channels': 1, 'sample_rate': 48000.0},
            {'id': 2, 'name': 'Stereo Mix (Monitor)', 'channels': 2, 'sample_rate': 44100.0},
        ],
        'output': [
            {'id': 10, 'name': 'Default Speakers', 'channels': 2, 'sample_rate': 44100.0},
            {'id': 11, 'name': 'USB Headset Speakers', 'channels': 2, 'sample_rate': 48000.0},
        ]
    }