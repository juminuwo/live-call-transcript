# Live Call Transcription Launcher with cuDNN Fix
# This PowerShell script sets the required CUDA environment variables before running

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

# Get the script path - use CPU-only version to avoid cuDNN issues
$scriptPath = $scriptPathCpu

Write-Host "Starting transcription with CPU processing..." -ForegroundColor Green
Write-Host "Python: $pythonPath" -ForegroundColor Gray
Write-Host "Script: $scriptPath" -ForegroundColor Gray
Write-Host ""

# Run the transcription script with all passed arguments
& $pythonPath $scriptPath $args

Write-Host ""
Write-Host "Transcription ended." -ForegroundColor Yellow
Read-Host "Press Enter to exit"