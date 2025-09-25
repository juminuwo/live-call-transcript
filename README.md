# Live Call Transcription System

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-orange.svg)](tests/)

A real-time transcription system that simultaneously captures and transcribes both microphone and system audio during live calls using WhisperLive. Perfect for recording meetings, calls, and conversations with dual-stream audio processing.

## ✨ Features

- **🎤 Dual Audio Capture**: Simultaneously record microphone and system/desktop audio
- **⚡ Real-time Transcription**: Stream processing using WhisperLive for low-latency results
- **📝 Multiple Output Formats**: JSON, CSV, or both with accurate timestamps
- **🖥️ Cross-platform Support**: Works on Windows, Linux, and macOS
- **🔧 Auto Device Detection**: Automatically finds system audio devices when possible
- **💬 Live Display**: Optional real-time console updates during transcription
- **🎯 Source Identification**: Clear labeling of microphone vs system audio
- **⚙️ Configurable**: Customizable audio devices, models, languages, and output formats
- **🛡️ Graceful Shutdown**: Handles interruptions properly with data preservation

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Configuration](#-configuration)
- [Output Format](#-output-format)
- [Platform-Specific Setup](#-platform-specific-setup)
- [API Reference](#-api-reference)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## 🚀 Installation

### Prerequisites

- **Python 3.10+**
- **Conda** (Miniconda or Anaconda)
- **Git** with submodules support

### Automatic Setup

1. **Clone the repository with submodules:**
   ```bash
   git clone --recursive https://github.com/your-username/live_call_transcript.git
   cd live_call_transcript
   ```

2. **Run the automated setup script:**
   ```bash
   ./setup.sh
   ```

   This script will:
   - Create the conda environment `live_call_transcript_312`
   - Install all dependencies
   - Set up WhisperLive from the submodule
   - Test the installation
   - Display platform-specific configuration tips

### Manual Setup

If you prefer manual installation:

```bash
# 1. Create conda environment
conda env create -f environment.yml

# 2. Activate environment
conda activate live_call_transcript_312

# 3. Install WhisperLive submodule
pip install -e ./WhisperLive

# 4. Test installation
python main.py --list-devices
```

## ⚡ Quick Start

1. **List available audio devices:**
   ```bash
   python main.py --list-devices
   ```

2. **Start basic transcription** (uses auto-detected devices):
   ```bash
   python main.py
   ```

3. **View real-time transcription in console:**
   ```bash
   python main.py --real-time-display
   ```

4. **Stop transcription:**
   Press `Ctrl+C` to gracefully stop and save all transcriptions.

## 📚 Usage Examples

### Basic Usage
```bash
# Auto-detect devices and save as JSON
python main.py

# Specify output file and format
python main.py --output meeting_notes --format csv

# Use specific audio devices
python main.py --mic-device 1 --system-device 2
```

### Advanced Configuration
```bash
# Spanish transcription with medium model
python main.py --lang es --model medium

# Both JSON and CSV output with real-time display
python main.py --format both --real-time-display --output call_transcript

# High sample rate for better quality
python main.py --sample-rate 48000 --model large
```

### Complete Example Session
```bash
# 1. Check available devices
python main.py --list-devices

# 2. Start transcription with specific setup
python main.py \\
  --mic-device 1 \\
  --system-device 2 \\
  --output "team_meeting_2024" \\
  --format both \\
  --lang en \\
  --model small \\
  --real-time-display

# 3. During the call, you'll see live transcription like:
# [00:01:23.450] mic    : Hello everyone, welcome to today's meeting
# [00:01:25.120] desktop: Hi there, thanks for having me
# [00:01:28.890] mic    : Let's start with the first agenda item
```

## ⚙️ Configuration

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--list-devices, -l` | List available audio devices | |
| `--mic-device, -m` | Microphone device ID | Auto-detect |
| `--system-device, -s` | System audio device ID | Auto-detect |
| `--output, -o` | Output file base name | Timestamp-based |
| `--format, -f` | Output format (json/csv/both) | json |
| `--sample-rate, -r` | Audio sample rate (Hz) | 16000 |
| `--lang` | Language code (en, es, fr, etc.) | en |
| `--model` | Whisper model size | small |
| `--no-console` | Disable console output | False |
| `--real-time-display` | Enable live console display | False |
| `--verbose, -v` | Enable verbose logging | False |

### Supported Models
- `tiny`: Fastest, least accurate (~39 MB)
- `base`: Good balance (~74 MB)
- `small`: **Recommended** (~244 MB)
- `medium`: Better accuracy (~769 MB)
- `large`: Best accuracy (~1550 MB)
- `large-v2`: Improved large model
- `large-v3`: Latest large model

### Supported Languages
Common language codes: `en` (English), `es` (Spanish), `fr` (French), `de` (German), `it` (Italian), `pt` (Portuguese), `ru` (Russian), `ja` (Japanese), `ko` (Korean), `zh` (Chinese)

## 📄 Output Format

### JSON Format (.jsonl)
Each line contains a timestamped transcription entry:
```json
{"time": "00:01:23.450", "timestamp": 1694567890.230, "source": "mic", "text": "Hello everyone, welcome to today's meeting"}
{"time": "00:01:25.120", "timestamp": 1694567892.450, "source": "desktop", "text": "Hi there, thanks for having me"}
{"time": "00:01:28.890", "timestamp": 1694567895.340, "source": "mic", "text": "Let's start with the first agenda item"}
```

### CSV Format (.csv)
Structured data with headers:
```csv
Time,Timestamp,Source,Text
00:01:23.450,1694567890.230,mic,"Hello everyone, welcome to today's meeting"
00:01:25.120,1694567892.450,desktop,"Hi there, thanks for having me"
00:01:28.890,1694567895.340,mic,"Let's start with the first agenda item"
```

### Field Descriptions
- **Time**: Relative timestamp from session start (HH:MM:SS.mmm)
- **Timestamp**: Absolute Unix timestamp with milliseconds
- **Source**: Audio source (`mic` for microphone, `desktop` for system audio)
- **Text**: Transcribed speech content

## 🖥️ Platform-Specific Setup

### Windows
For system audio capture on Windows:

1. **Enable Stereo Mix:**
   - Right-click sound icon → "Open Sound settings"
   - Click "Sound Control Panel"
   - Go to "Recording" tab
   - Right-click and "Show Disabled Devices"
   - Enable "Stereo Mix" and set as default

2. **Alternative: Use virtual audio cables:**
   - Install VB-Cable or similar virtual audio driver
   - Configure applications to output to virtual cable
   - Select virtual cable as system device

### Linux
For system audio capture on Linux:

1. **Install PulseAudio tools:**
   ```bash
   sudo apt-get update
   sudo apt-get install pulseaudio pavucontrol
   ```

2. **Configure audio routing:**
   ```bash
   # Launch PulseAudio Volume Control
   pavucontrol

   # In the "Recording" tab, set applications to monitor desktop audio
   ```

3. **Find monitor devices:**
   ```bash
   pactl list sources | grep -E "(Name:|Description:)"
   ```

### macOS
For system audio capture on macOS:

1. **Install BlackHole (recommended):**
   ```bash
   brew install blackhole-2ch
   ```

2. **Configure Audio MIDI Setup:**
   - Open "Audio MIDI Setup" app
   - Create "Multi-Output Device" combining speakers + BlackHole
   - Create "Aggregate Device" combining mic + BlackHole
   - Set system output to Multi-Output Device

3. **Alternative: Use SoundFlower** (older option)

## 📖 API Reference

### Core Classes

#### `AudioCapture`
Handles simultaneous audio capture from multiple devices.

```python
from live_call_transcript import AudioCapture

# Create capture instance
capture = AudioCapture(
    sample_rate=16000,
    channels=1,
    mic_device=1,
    system_device=2
)

# Set callbacks for processed audio
capture.set_callbacks(mic_callback, system_callback)

# Start/stop capture
capture.start_recording()
capture.stop_recording()
```

#### `DualStreamTranscriber`
Manages WhisperLive server and dual transcription clients.

```python
from live_call_transcript import DualStreamTranscriber

def transcription_callback(timestamp, source, text):
    print(f"[{timestamp}] {source}: {text}")

# Create transcriber
transcriber = DualStreamTranscriber(
    lang="en",
    model="small",
    callback=transcription_callback
)

# Start transcription system
with transcriber:
    # Send audio data
    transcriber.process_mic_audio(audio_data)
    transcriber.process_system_audio(audio_data)
```

#### `TranscriptionLogger`
Handles output formatting and file writing.

```python
from live_call_transcript import TranscriptionLogger

# Create logger
logger = TranscriptionLogger(
    output_file="meeting_notes",
    format_type="both",  # json, csv, or both
    console_output=True
)

# Use as context manager
with logger:
    logger.log_transcription(timestamp, "mic", "Hello world")
```

### Utility Functions

```python
from live_call_transcript.audio_utils import AudioCapture

# List available devices
devices = AudioCapture.list_audio_devices()
print(devices['input'])  # Microphones
print(devices['output']) # Speakers

# Auto-detect system audio device
system_device = AudioCapture.find_system_audio_device()
```

## 🧪 Testing

The project includes comprehensive tests covering all major functionality.

### Running Tests
```bash
# Activate environment
conda activate live_call_transcript_312

# Run all tests
pytest

# Run with coverage
pytest --cov=src/live_call_transcript --cov-report=html

# Run specific test modules
pytest tests/test_audio_utils.py
pytest tests/test_logger.py -v

# Run tests with output
pytest -s tests/test_transcriber.py
```

### Test Coverage
- **Audio capture and device detection**
- **Transcription pipeline and WebSocket handling**
- **Output formatting (JSON/CSV) and file I/O**
- **Main application logic and CLI parsing**
- **Error handling and edge cases**

## 🔧 Troubleshooting

### Common Issues

#### "No system audio device found"
**Problem**: System cannot detect desktop audio capture device.

**Solutions**:
- **Windows**: Enable "Stereo Mix" in sound settings
- **Linux**: Install PulseAudio and use `pavucontrol` to configure monitors
- **macOS**: Install BlackHole or SoundFlower virtual audio driver
- **Manual**: Use `--system-device N` to specify device ID manually

#### "ModuleNotFoundError: No module named 'sounddevice'"
**Problem**: Audio dependencies not installed.

**Solution**:
```bash
conda activate live_call_transcript_312
conda install portaudio -c conda-forge
pip install sounddevice
```

#### "WhisperLive server failed to start"
**Problem**: Server port already in use or dependencies missing.

**Solutions**:
- Check if port 9090 is available: `netstat -an | grep 9090`
- Kill existing processes: `pkill -f whisper_live`
- Try different port: `python main.py --port 9091` (not currently supported, modify code)
- Reinstall WhisperLive: `pip install -e ./WhisperLive --force-reinstall`

#### Poor transcription quality
**Problem**: Transcriptions are inaccurate or incomplete.

**Solutions**:
- Use larger model: `--model medium` or `--model large`
- Increase sample rate: `--sample-rate 48000`
- Check audio device selection with `--list-devices`
- Ensure good microphone positioning and minimal background noise
- Verify system audio is actually being captured

#### High CPU/Memory usage
**Problem**: System resources consumed heavily during transcription.

**Solutions**:
- Use smaller model: `--model tiny` or `--model base`
- Reduce sample rate: `--sample-rate 8000` (may affect quality)
- Close unnecessary applications
- Consider using GPU acceleration if available

### Debug Mode
Enable verbose logging for troubleshooting:
```bash
python main.py --verbose
```

### Getting Help
1. Check the [Issues](https://github.com/your-username/live_call_transcript/issues) page
2. Enable verbose mode and check logs
3. Test with `--list-devices` to verify audio setup
4. Try different audio devices and settings

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone with submodules
git clone --recursive https://github.com/your-username/live_call_transcript.git
cd live_call_transcript

# Create development environment
conda env create -f environment.yml
conda activate live_call_transcript_312

# Install in development mode
pip install -e .

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Project Structure
```
live_call_transcript/
├── src/live_call_transcript/    # Main package code
│   ├── __init__.py
│   ├── audio_utils.py           # Audio capture utilities
│   ├── transcriber.py           # WhisperLive integration
│   └── logger.py                # Output formatting
├── tests/                       # Test suite
│   ├── test_audio_utils.py
│   ├── test_transcriber.py
│   ├── test_logger.py
│   └── test_main.py
├── WhisperLive/                 # Submodule
├── main.py                      # CLI entry point
├── environment.yml              # Conda dependencies
├── setup.sh                     # Setup script
└── README.md                    # This file
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [WhisperLive](https://github.com/collabora/WhisperLive) - Real-time speech transcription
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [SoundDevice](https://python-sounddevice.readthedocs.io/) - Audio I/O for Python

## 📊 Performance Notes

### Recommended System Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 4GB+ (8GB+ for large models)
- **Storage**: 2GB+ free space for models
- **Network**: Not required (runs locally)

### Model Performance Comparison
| Model | Size | Relative Speed | Accuracy | Use Case |
|-------|------|---------------|----------|----------|
| tiny | 39 MB | Fastest | Basic | Quick testing |
| base | 74 MB | Fast | Good | Light usage |
| small | 244 MB | **Balanced** | **Good** | **Recommended** |
| medium | 769 MB | Slower | Better | High accuracy needs |
| large | 1550 MB | Slowest | Best | Professional use |

---

**Happy transcribing! 🎉**

For questions, issues, or suggestions, please visit our [GitHub repository](https://github.com/your-username/live_call_transcript).