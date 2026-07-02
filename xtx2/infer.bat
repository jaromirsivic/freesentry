@echo off
rem XTX2 Pose - interactive inference launcher (double-clickable).
cd /d %~dp0

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create the virtual environment. Is Python on PATH?
        pause
        exit /b 1
    )
)

if not exist ".venv\.deps_installed" (
    echo Installing requirements ^(first run^)...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install requirements.
        pause
        exit /b 1
    )
    type nul > ".venv\.deps_installed"
)

call ".venv\Scripts\activate.bat"
python infer.py
pause
