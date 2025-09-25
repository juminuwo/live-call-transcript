# Live Call Transcription Launcher with GPU/CUDA Support
# This PowerShell script sets up CUDA environment and attempts GPU transcription

Write-Host "Setting up CUDA/GPU environment..." -ForegroundColor Green

# Set CUDA environment variables
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$env:CUDNN_PATH = "C:\Program Files\NVIDIA\CUDNN\v9.13"
$env:PATH = "C:\Program Files\NVIDIA\CUDNN\v9.13\bin\13.0;" + $env:PATH
$env:TORCH_CUDNN_V8_API_ENABLED = "1"
$env:CUDA_VISIBLE_DEVICES = "0"

Write-Host "CUDA environment configured:" -ForegroundColor Yellow
Write-Host "  CUDA_PATH: $($env:CUDA_PATH)" -ForegroundColor Cyan
Write-Host "  CUDNN_PATH: $($env:CUDNN_PATH)" -ForegroundColor Cyan
Write-Host "  CUDA_VISIBLE_DEVICES: $($env:CUDA_VISIBLE_DEVICES)" -ForegroundColor Cyan
Write-Host ""

# Check if conda environment exists
$condaEnvs = conda info --envs 2>$null
if (-not ($condaEnvs -match "live_call_transcript_312")) {
    Write-Host "ERROR: Conda environment 'live_call_transcript_312' not found" -ForegroundColor Red
    Write-Host "Please run: conda activate live_call_transcript_312" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Source the configuration file
. "./config.ps1"

# Check if Python exists
if (-not (Test-Path $pythonPath)) {
    Write-Host "ERROR: Python not found at $pythonPath" -ForegroundColor Red
    Write-Host "Please check your conda environment path" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if NVIDIA GPU is available
Write-Host "Checking GPU availability..." -ForegroundColor Yellow
try {
    $gpuCheck = & $pythonPath -c "
import torch
print('PyTorch CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU count:', torch.cuda.device_count())
    print('GPU name:', torch.cuda.get_device_name(0))
else:
    print('No CUDA GPU detected')
" 2>$null

    Write-Host $gpuCheck -ForegroundColor Cyan
} catch {
    Write-Host "Could not check GPU availability" -ForegroundColor Yellow
}

Write-Host ""

# Get the script path - use original GPU version
$scriptPath = $scriptPathGpu

Write-Host "Starting transcription with GPU acceleration..." -ForegroundColor Green
Write-Host "Python: $pythonPath" -ForegroundColor Gray
Write-Host "Script: $scriptPath" -ForegroundColor Gray
Write-Host ""
Write-Host "WARNING: If you get cuDNN errors, use run_transcription_cpu.ps1 instead" -ForegroundColor Red
Write-Host ""

# Run the transcription script with all passed arguments
try {
    & $pythonPath $scriptPath $args
} catch {
    Write-Host ""
    Write-Host "GPU transcription failed. Consider using the CPU version:" -ForegroundColor Red
    Write-Host ".\run_transcription_cpu.ps1 $args" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Transcription ended." -ForegroundColor Yellow
Read-Host "Press Enter to exit"