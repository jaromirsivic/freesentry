@echo off
rem XTX2 Pose - interactive training launcher (double-clickable). Uses uv.
cd /d %~dp0

call :ensure_env
if errorlevel 1 exit /b 1

call ".venv\Scripts\activate.bat"
python train.py %*
pause
exit /b 0

rem --------------------------------------------------------------------------
rem Ensure uv, the virtualenv and dependencies (incl. the right torch build).
rem --------------------------------------------------------------------------
:ensure_env
where uv >nul 2>nul
if errorlevel 1 (
    echo uv not found - downloading uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "Path=%USERPROFILE%\.local\bin;%Path%"
)
where uv >nul 2>nul
if errorlevel 1 (
    echo Failed to install uv. See https://docs.astral.sh/uv/
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment with uv...
    uv venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        exit /b 1
    )
)

if not exist ".venv\.deps_installed" call :install_deps
exit /b 0

:install_deps
echo Installing requirements ^(first run^)...
uv pip install -r requirements.txt --python .venv\Scripts\python.exe
if errorlevel 1 (
    echo Failed to install requirements.
    exit /b 1
)
rem Install torch separately so a CUDA build is not clobbered by the CPU wheel.
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo No NVIDIA GPU detected - installing CPU torch...
    uv pip install torch torchvision --python .venv\Scripts\python.exe
) else (
    echo NVIDIA GPU detected - installing CUDA torch ^(cu130^)...
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 --python .venv\Scripts\python.exe
)
if errorlevel 1 (
    echo Failed to install torch.
    exit /b 1
)
type nul > ".venv\.deps_installed"
exit /b 0
