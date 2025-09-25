"""
Live Call Transcription System

A reliable real-time transcription system that captures and transcribes both
microphone and system audio during live calls using faster-whisper directly.
"""

__version__ = "2.0.0"
__author__ = "Live Call Transcript Team"
__description__ = "Reliable real-time dual-stream call transcription using faster-whisper"

from .audio_capture import AudioCapture
from .transcription_engine import TranscriptionEngine
from .logger import TranscriptionLogger

__all__ = [
    "AudioCapture",
    "TranscriptionEngine",
    "TranscriptionLogger"
]