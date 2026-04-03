powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" && set Path=C:\Users\User\.local\bin;%Path%
uv init
git init
IF NOT EXIST .venv\Scripts\activate (uv venv)
call .venv\Scripts\activate
rem uv pip install -U ultralytics
rem nvidia-smi
rem download drivers for your version of CUDA https://developer.nvidia.com/cuda-downloads
rem install drivers and cuda toolkit
rem nvcc --version
rem CUDA-enabled torch for NVIDIA runtime is pinned in pyproject.toml via the cu130 index.
rem uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
rem uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
rem python
rem import torch
rem print(torch.cuda.is_available())
uv add fastapi --extra standard
uv add uvicorn --extra standard
uv add gpiozero pigpio
uv add numpy
uv add opencv-python
uv add ultralytics
uv add debugpy
uv add tensorrt
uv add picamera2
uv pip install python-periphery
uv add torch torchvision torchaudio
python -c "import sys, torch; print(sys.executable); print(torch.__file__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
pause