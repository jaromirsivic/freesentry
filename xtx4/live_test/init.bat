@echo off
rem XTX4 live test - initialise the Python virtual environment with uv.
rem Usage: init.bat            (interactive)
rem        init.bat --auto     (no questions, no pause; used by start.bat)
setlocal
cd /d "%~dp0"

set "AUTO=0"
if /i "%~1"=="--auto" set "AUTO=1"
set "PYTHON_VERSION=3.12"

echo --------------------------------
echo XTX4 live test - environment setup
echo --------------------------------

rem ---- uv -------------------------------------------------------------------
where uv >nul 2>nul
if errorlevel 1 (
    echo uv not found - installing uv from astral.sh ...
    powershell -NoProfile -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "Path=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%Path%"
)
where uv >nul 2>nul
if errorlevel 1 (
    echo ERROR: uv is still not available. Install it manually: https://docs.astral.sh/uv/
    goto :fail
)
uv --version

rem ---- virtual environment ----------------------------------------------------
if exist ".venv\Scripts\python.exe" (
    if "%AUTO%"=="1" goto :install
    choice /C YN /M "A virtual environment already exists. Recreate it from scratch? (Y=recreate, N=only update packages)"
    if errorlevel 2 goto :install
    echo Removing the existing environment...
    rmdir /s /q ".venv"
)

echo Creating virtual environment (Python %PYTHON_VERSION%, downloaded by uv if needed)...
uv venv --no-project --python %PYTHON_VERSION% .venv
if errorlevel 1 (
    echo ERROR: failed to create the virtual environment.
    goto :fail
)

:install
echo Installing packages from requirements.txt ...
uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
if errorlevel 1 (
    echo ERROR: package installation failed.
    goto :fail
)

rem ---- verification -------------------------------------------------------------
echo Verifying the installation...
".venv\Scripts\python.exe" -c "import ncnn, onnxruntime, cv2, numpy, psutil, PIL, tkinter; print('ncnn', ncnn.__version__, '| onnxruntime', onnxruntime.__version__, '| opencv', cv2.__version__, '| numpy', numpy.__version__)"
if errorlevel 1 (
    echo ERROR: verification failed - see the message above.
    goto :fail
)
".venv\Scripts\python.exe" -c "import sys; sys.path.insert(0, '..'); import xtx4.runtime; assert 'torch' not in sys.modules; print('xtx4.runtime OK (torch-free)')"
if errorlevel 1 (
    echo ERROR: the xtx4 package was not found next to live_test\ - keep live_test inside the xtx4 repository.
    goto :fail
)
".venv\Scripts\python.exe" -c "import ncnn; n = ncnn.get_gpu_count(); print('Vulkan GPUs usable by ncnn:', n); ncnn.destroy_gpu_instance()" 2>nul

echo --------------------------------
echo Environment ready. Run start.bat to launch the XTX4 test app.
echo --------------------------------
if "%AUTO%"=="0" pause
endlocal & exit /b 0

:fail
if "%AUTO%"=="0" pause
endlocal & exit /b 1
