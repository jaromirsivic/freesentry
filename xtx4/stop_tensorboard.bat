@echo off
rem XTX4 Pose - stop the TensorBoard started by tensorboard.bat (closes its console window).
taskkill /FI "WINDOWTITLE eq XTX4 TensorBoard*" /T /F >nul 2>nul
if errorlevel 1 (
    echo No running "XTX4 TensorBoard" window found.
) else (
    echo TensorBoard stopped.
)
