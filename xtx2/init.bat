@echo off

choice /C YN /M "Freesentry project in this folder will be initialized from scratch. Do you want to continue? (Y=Yes N=No)"
if errorlevel 2 exit /b 0
echo Preparing to initialize the project in this folder from scratch
echo This may take a few minutes...
rmdir /s /q .venv
del uv.lock
del pyproject.toml
del .python-version

echo Downloading uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" && set Path=C:\Users\User\.local\bin;%Path%
echo Initializing python environment
uv init
uv venv
IF NOT EXIST .venv\Scripts\activate (uv venv)
call .venv\Scripts\activate
uv add fastapi --extra standard
uv add uvicorn --extra standard
uv add gpiozero pigpio
uv add numpy
uv add debugpy
uv add opencv-python
uv add ultralytics 

@echo on
echo Current user: %username%
@echo off


echo --------------------------------
echo If you have AMD, Intel, NVidia or other GPU,
echo you can use the Vulkan driver to run the app.
echo Download the Vulkan driver for your GPU 
echo from the vendor's website and install it.
echo NVidia: https://developer.nvidia.com/vulkan-driver
echo Nvidia: https://www.google.com/search?q=nvidia+vulkan+driver+download
echo AMD: https://www.google.com/search?q=amd+vulkan+driver+download
echo Intel: https://www.google.com/search?q=intel+vulkan+driver+download
echo Then run the following command to test if
echo the Vulkan driver is installed correctly:
echo vulkaninfo
echo --------------------------------
choice /C YN /M "Should I run the vulkaninfo command to test if the Vulkan driver is installed correctly? (Y=Yes N=No)"
if errorlevel 2 (
    echo OK, skipping vulkaninfo command
) else (
    echo Running vulkaninfo command
    call vulkaninfo
)

echo --------------------------------
echo If you have a NVIDIA GPU with CUDA support,
echo you can use the CUDA version of this app.
echo To install the CUDA drivers download them from
echo https://developer.nvidia.com/cuda-downloads
echo and install the drivers and cuda toolkit.
echo Then run the following command to test if
echo the CUDA drivers are installed correctly:
echo nvcc --version
echo --------------------------------

choice /C YN /M "Should I run the nvcc --version command to test if the CUDA drivers are installed correctly? (Y=Yes N=No)"
if errorlevel 2 (
    echo OK, skipping nvcc --version command
) else (
    echo Running nvcc --version command
    call nvcc --version
)

echo --------------------------------
choice /C YN /M "Do you want to setup the app to use a NVIDIA GPU with CUDA support? (Y=Yes N=No)"
if errorlevel 2 (
    echo Installing CPU only version of torch
    uv add torch torchvision torchaudio
) else (
    echo Installing CUDA version of torch
    echo --------------------------------
    choice /C YN /M "Do you want to install CUDA 13.X (Recommended - say Y), or CUDA 12.X (say N)?"
    if errorlevel 2 (
        echo Installing CUDA 124
        uv pip uninstall torch
        uv pip uninstall torchvision
        uv pip uninstall torchaudio
        uv pip install torch --index-url https://download.pytorch.org/whl/cu124
        uv pip install torchvision --index-url https://download.pytorch.org/whl/cu124
        uv pip install torchaudio --index-url https://download.pytorch.org/whl/cu124
    ) else (
        echo Installing CUDA 130
        uv pip uninstall torch
        uv pip uninstall torchvision
        uv pip uninstall torchaudio
        uv pip install torch --index-url https://download.pytorch.org/whl/cu130
        uv pip install torchvision --index-url https://download.pytorch.org/whl/cu130
        uv pip install torchaudio --index-url https://download.pytorch.org/whl/cu130
    )
    echo --------------------------------
    echo Support for CUDA installed.
    echo Now I will test if the CUDA is available.
    timeout /t 10 /nobreak
    echo Testing if the CUDA drivers are installed correctly
    echo this may take a few minutes...
    python -c "import torch; print(f'CUDA is available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"
)
echo --------------------------------
echo Environment initialized successfully
echo --------------------------------
pause