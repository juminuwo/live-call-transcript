#!/bin/zsh

powershell.exe -ExecutionPolicy Bypass -File "$(wslpath -w "$(pwd)/run_transcription_cpu.ps1")" "$@"
