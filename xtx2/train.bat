@echo off
rem XTX2 Pose - interactive training launcher (double-clickable). Uses uv.
cd /d %~dp0

where uv >nul 2>nul
if errorlevel 1 (
    echo uv not found - downloading uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "Path=%USERPROFILE%\.local\bin;%Path%"
)
where uv >nul 2>nul
if errorlevel 1 (
    echo Failed to install uv. See https://docs.astral.sh/uv/
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment with uv...
    uv venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

if not exist ".venv\.deps_installed" (
    echo Installing requirements ^(first run^)...
    uv pip install -r requirements.txt --python .venv\Scripts\python.exe
    if errorlevel 1 (
        echo Failed to install requirements.
        pause
        exit /b 1
    )
    type nul > ".venv\.deps_installed"
)

call ".venv\Scripts\activate.bat"
python train.py %*
pause
