@echo off
rem XTX4 Pose - training launcher (double-clickable, or pass train.py flags).
rem Creates the virtual environment on first run via init.bat, then starts train.py.
cd /d %~dp0

if not exist ".venv\.deps_installed" (
    echo Environment not initialised - running init.bat...
    call init.bat
    if errorlevel 1 (
        echo init.bat failed.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" -c "import tensorboard" >nul 2>nul
if errorlevel 1 (
    uv pip install --python ".venv\Scripts\python.exe" "tensorboard>=2.20,<3"
    if errorlevel 1 (
        echo Failed to install TensorBoard.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" train.py %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
