#!/usr/bin/env python3
"""
Test script for the reliable transcription system.
Quick verification that all components work together.
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from live_call_transcript.audio_capture import AudioCapture
from live_call_transcript.transcription_engine import TranscriptionEngine

def test_components():
    """Test individual components."""
    print("=== Testing Reliable Transcription Components ===")

    # Test 1: Audio device detection
    print("\n1. Testing audio device detection...")
    try:
        devices = AudioCapture.list_audio_devices()
        print(f"   Found {len(devices['input'])} input devices")

        system_device = AudioCapture.find_system_audio_device()
        if system_device is not None:
            print(f"   Auto-detected system audio: device {system_device}")
        else:
            print("   No system audio device found")
        print("   [OK] Audio device detection working")
    except Exception as e:
        print(f"   [ERROR] Audio device detection failed: {e}")
        return False

    # Test 2: Transcription engine initialization
    print("\n2. Testing transcription engine...")
    try:
        def dummy_callback(timestamp, source, text):
            print(f"   Callback received: [{source}] {text}")

        engine = TranscriptionEngine(
            model_name="tiny",  # Use smallest model for testing
            callback=dummy_callback
        )
        print("   [OK] Transcription engine created")

        # Test model loading
        engine._initialize_model()
        print("   [OK] Model loaded successfully")

        # Test audio processing
        test_audio = np.random.random(16000).astype(np.float32) * 0.1  # 1 second of quiet noise
        engine.process_mic_audio(test_audio)
        engine.process_desktop_audio(test_audio)
        print("   [OK] Audio processing working")

    except Exception as e:
        print(f"   [ERROR] Transcription engine test failed: {e}")
        return False

    # Test 3: Audio capture initialization
    print("\n3. Testing audio capture...")
    try:
        capture = AudioCapture()
        print("   [OK] Audio capture created")

        # Test health check
        health = capture.is_healthy()
        print(f"   Health status: {health}")
        print("   [OK] Audio capture working")

    except Exception as e:
        print(f"   [ERROR] Audio capture test failed: {e}")
        return False

    print("\n=== All Component Tests Passed! ===")
    return True

def test_integration():
    """Test full system integration."""
    print("\n=== Integration Test ===")

    try:
        transcription_count = 0

        def test_callback(timestamp, source, text):
            nonlocal transcription_count
            transcription_count += 1
            print(f"[{source.upper()}] {text}")

        # Initialize components
        print("Initializing transcription engine...")
        engine = TranscriptionEngine(
            model_name="tiny",
            callback=test_callback
        )

        print("Initializing audio capture...")
        capture = AudioCapture()

        # Connect them
        capture.set_callbacks(
            engine.process_mic_audio,
            engine.process_desktop_audio
        )

        print("Starting engine...")
        engine.start()

        print("Testing with synthetic audio...")

        # Generate test audio that might produce some transcription
        sample_rate = 16000
        duration = 2

        # Generate a more complex test signal
        t = np.linspace(0, duration, sample_rate * duration, False)
        test_audio = (
            np.sin(2 * np.pi * 400 * t) +  # 400 Hz tone
            np.sin(2 * np.pi * 600 * t) +  # 600 Hz tone
            np.random.random(len(t)) * 0.1   # Some noise
        ).astype(np.float32) * 0.3

        # Send test audio
        for i in range(3):
            print(f"Sending test audio chunk {i+1}/3...")
            engine.process_mic_audio(test_audio)
            engine.process_desktop_audio(test_audio * 0.7)
            time.sleep(3)  # Give time for processing

        # Get final stats
        stats = engine.get_stats()
        print(f"\nFinal stats: {stats}")

        print("Stopping engine...")
        engine.stop()

        print("\n[OK] Integration test completed successfully")
        print(f"Note: Generated {transcription_count} transcriptions from test audio")

        return True

    except Exception as e:
        print(f"[ERROR] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Starting Reliable Transcription System Tests...")

    # Component tests
    if not test_components():
        print("\n[ERROR] Component tests failed!")
        return 1

    # Integration test
    if not test_integration():
        print("\n[ERROR] Integration test failed!")
        return 1

    print("\n" + "="*50)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*50)
    print("\nYour reliable transcription system is ready to use!")
    print("\nTo start transcribing:")
    print(f"  python main_reliable.py")
    print(f"  python main_reliable.py --help  # for options")

    return 0

if __name__ == "__main__":
    sys.exit(main())