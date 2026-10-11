@echo off
rem XTX4 - create .venv with uv and install dependencies (CUDA torch when an NVIDIA GPU is present).
cd /d %~dp0
where uv >nul 2>nul
if errorlevel 1 (
    echo uv not found - downloading uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "Path=%USERPROFILE%\.local\bin;%Path%"
)
if not exist ".venv\Scripts\python.exe" uv venv --python 3.12
rem Install torch FIRST so that requirements (pnnx depends on torch) do not pull the CPU-only wheel.
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    uv pip install torch torchvision --python .venv\Scripts\python.exe
) else (
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 --python .venv\Scripts\python.exe
)
if errorlevel 1 exit /b 1
uv pip install -r requirements.txt --python .venv\Scripts\python.exe
if errorlevel 1 exit /b 1
rem pnnx may have replaced torch with a CPU build; force the CUDA build back when a GPU is present.
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    uv pip install --reinstall-package torch torch torchvision --index-url https://download.pytorch.org/whl/cu130 --python .venv\Scripts\python.exe
)
type nul > ".venv\.deps_installed"
echo XTX4 environment ready.
