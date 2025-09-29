"""
Test script to verify audio device setup for live_call_transcript_312 project.
Tests both microphone and system audio capture capabilities.
"""

import sounddevice as sd
import numpy as np
import os

# Set PulseAudio server for WSL
os.environ['PULSE_SERVER'] = 'unix:/mnt/wslg/PulseServer'

def test_audio_devices():
    """Test and display available audio devices."""
    print("=== Audio Device Test ===")
    print(f"sounddevice version: {sd.__version__}")
    print(f"Default sample rate: {sd.default.samplerate}")
    print(f"Default device: {sd.default.device}")

    print("\nAvailable devices:")
    devices = sd.query_devices()
    print(devices)

    # Try to identify microphone and system audio sources
    print("\n=== Device Analysis ===")
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"Input Device {i}: {device['name']} ({device['max_input_channels']} channels)")

    return devices

def test_microphone_recording(duration=3):
    """Test microphone recording capability."""
    print(f"\n=== Testing Microphone (RDPSource) ===")
    try:
        # Use RDPSource for microphone input
        recording = sd.rec(int(duration * sd.default.samplerate),
                          channels=1, dtype='float64',
                          device=None)  # Use default input
        print(f"Recording for {duration} seconds... Speak into microphone!")
        sd.wait()

        # Check if we captured audio
        max_amplitude = np.max(np.abs(recording))
        print(f"Max amplitude captured: {max_amplitude:.4f}")

        if max_amplitude > 0.001:
            print("✓ Microphone recording successful!")
            return True
        else:
            print("⚠ No significant audio detected from microphone")
            return False

    except Exception as e:
        print(f"✗ Microphone test failed: {e}")
        return False

def test_system_audio_capture():
    """Test system audio capture via loopback."""
    print(f"\n=== Testing System Audio Capture ===")
    try:
        # List sources to find our loopback
        print("Available PulseAudio sources:")
        import subprocess
        result = subprocess.run(['pactl', 'list', 'sources', 'short'],
                              capture_output=True, text=True,
                              env={'PULSE_SERVER': 'unix:/mnt/wslg/PulseServer'})
        print(result.stdout)

        if 'capture_sink.monitor' in result.stdout:
            print("✓ System audio loopback configured correctly!")
            print("To capture system audio, your applications should output to 'capture_sink'")
            return True
        else:
            print("⚠ System audio loopback not found")
            return False

    except Exception as e:
        print(f"✗ System audio test failed: {e}")
        return False

def show_usage_instructions():
    """Show instructions for using the audio setup."""
    print(f"\n=== Usage Instructions ===")
    print("1. Microphone: Use default input device (RDPSource)")
    print("2. System Audio: Applications should output to 'capture_sink'")
    print("   - You can set this as default: pactl set-default-sink capture_sink")
    print("   - Or redirect specific apps to capture_sink")
    print("3. In your Python code:")
    print("   - Mic input: sd.rec(..., device=None)  # Uses default")
    print("   - System audio: Use source 'capture_sink.monitor'")
    print("\n=== PulseAudio Commands ===")
    print("- List sources: pactl list sources short")
    print("- List sinks: pactl list sinks short")
    print("- Set default sink: pactl set-default-sink capture_sink")

if __name__ == "__main__":
    print("Testing audio setup for live_call_transcript_312...")

    devices = test_audio_devices()
    mic_ok = test_microphone_recording()
    sys_ok = test_system_audio_capture()

    print(f"\n=== Summary ===")
    print(f"✓ Audio system: WSLg PulseAudio")
    print(f"{'✓' if mic_ok else '⚠'} Microphone: {'Working' if mic_ok else 'Check connection'}")
    print(f"{'✓' if sys_ok else '⚠'} System Audio: {'Configured' if sys_ok else 'Needs setup'}")

    if mic_ok and sys_ok:
        print("\n🎉 Audio setup is ready for live call transcription!")
    else:
        print("\n⚠ Some audio components need attention")

    show_usage_instructions()