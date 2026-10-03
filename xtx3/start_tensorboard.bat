@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\tensorboard.ps1" -Action start %*
if errorlevel 1 (
    pause
    exit /b 1
)
