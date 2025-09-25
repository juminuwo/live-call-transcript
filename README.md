# Live Call Transcription

A real-time transcription system that captures and transcribes both microphone and system audio simultaneously during live calls.

## Features

- **Dual Audio Capture**: Records microphone and system/desktop audio in parallel.
- **Real-time Transcription**: Utilizes `faster-whisper` for efficient and accurate local transcription.
- **CPU and GPU Support**: Includes scripts for running on both CPU and GPU.
- **Automatic Device Detection**: Attempts to automatically find the system audio loopback device.
- **Multiple Output Formats**: Saves transcriptions in both JSON and CSV formats.
- **Command-Line Interface**: Provides a simple CLI for configuration.
- **Robust and Reliable**: Designed for stability with graceful shutdown and error handling.

## Installation

### Prerequisites

- Python 3.10+
- Git

### Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/live_call_transcript.git
    cd live_call_transcript
    ```

2.  **Install dependencies:**

    It is recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements_reliable.txt
    ```

## Configuration

The PowerShell scripts `run_transcription_cpu.ps1` and `run_transcription_gpu.ps1` use a configuration file `config.ps1` to set the path to the Python executable and the transcription scripts.

You can edit `config.ps1` to match your system's setup.

-   `$pythonPath`: The full path to the Python executable within your conda environment.
-   `$scriptPathCpu`: The full path to the `main_cpu.py` script.
-   `$scriptPathGpu`: The full path to the `main.py` script.

## Usage

### Listing Audio Devices

To see a list of available audio devices, run:

```bash
python main.py --list-devices
```

This will output the available microphone and system audio devices with their corresponding IDs.

### Running the Transcription

-   **For GPU (Recommended):**

    ```bash
    python main.py
    ```

-   **For CPU only:**

    ```bash
    python main_cpu.py
    ```

You can also use the PowerShell scripts on Windows, which handle the environment setup. See the "Configuration" section to configure the paths for the scripts.

-   **For GPU:**

    ```powershell
    ./run_transcription_gpu.ps1
    ```

-   **For CPU:**

    ```powershell
    ./run_transcription_cpu.ps1
    ```

### Command-Line Options

| Option | Description | Default |
| --- | --- | --- |
| `--list-devices`, `-l` | List available audio devices and exit. | |
| `--mic-device`, `-m` | Microphone device ID. | Auto-detect |
| `--system-device`, `-s` | System audio device ID. | Auto-detect |
| `--output`, `-o` | Base name for the output files. | `data/transcript_{timestamp}` |
| `--sample-rate`, `-r` | Audio sample rate in Hz. | 16000 |
| `--lang` | Language code for transcription. | `en` |
| `--model` | Whisper model to use. | `small` |
| `--verbose`, `-v` | Enable verbose logging. | `False` |

### Example

```bash
python main.py --mic-device 1 --system-device 5 --output my_meeting --lang en --model medium
```

This command will start the transcription with microphone device 1 and system audio device 5, save the output to `my_meeting.json` and `my_meeting.csv`, use the English language, and the `medium` Whisper model.

## Output Format

The application generates two output files in the `data` directory: a JSON file and a CSV file.

### JSON Format (`.json`)

Each line in the JSON file is a JSON object representing a single transcription segment.

```json
{"timestamp": "2025-09-25 10:00:01.123", "source": "mic", "text": "Hello, this is a test.", "session_time": "1.23s"}
{"timestamp": "2025-09-25 10:00:02.456", "source": "desktop", "text": "I can hear you.", "session_time": "2.56s"}
```

### CSV Format (`.csv`)

The CSV file contains the same information in a tabular format.

```csv
timestamp,source,text,session_time
2025-09-25 10:00:01.123,mic,"Hello, this is a test.",1.23s
2025-09-25 10:00:02.456,desktop,"I can hear you.",2.56s
```

## Platform-Specific Setup

### Windows

For system audio capture, you may need to enable "Stereo Mix" in your sound settings.

1.  Right-click the sound icon in the taskbar and select "Sounds".
2.  Go to the "Recording" tab.
3.  Right-click in the empty space and select "Show Disabled Devices".
4.  Right-click on "Stereo Mix" and select "Enable".
5.  Set it as the default device if needed.

If "Stereo Mix" is not available, you can use a virtual audio cable like [VB-CABLE](https://vb-audio.com/Cable/).

### Linux

System audio capture on Linux usually requires PulseAudio. You can find the monitor device by running:

```bash
pactl list sources | grep 'Monitor'
```

### macOS

On macOS, you will need to use a virtual audio device like [BlackHole](https://github.com/ExistentialAudio/BlackHole) or [Soundflower](https://github.com/mattingalls/Soundflower) to capture system audio.

## Troubleshooting

-   **No system audio device found:** Make sure you have configured your system as described in the "Platform-Specific Setup" section.
-   **cuDNN errors on Windows:** If you encounter errors related to cuDNN, try running the `main_cpu.py` script or the `run_transcription_cpu.ps1` PowerShell script to use the CPU for transcription.
-   **Poor transcription quality:** Try using a larger model (e.g., `--model medium` or `--model large`) for better accuracy, at the cost of higher resource usage.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

No license. Free to use.

## Acknowledgements

This project relies on the following open-source libraries:

-   **[faster-whisper](https://github.com/guillaumekln/faster-whisper)**: For providing a faster, more efficient implementation of the Whisper model.

A big thank you to the developers and maintainers of these projects.