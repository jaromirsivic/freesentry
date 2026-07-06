@echo off
rem XTX2 Pose - single-image inference with fixed outputs. Uses uv.
rem
rem Usage:
rem   infer_image.bat model input_image output_image output_pose
rem
rem   model         : n, m, or l
rem   input_image   : path to input .jpg or .png
rem   output_image  : path to output .png with green boxes and red keypoints
rem   output_pose   : path to output .json with detected poses
rem
rem Example:
rem   infer_image.bat n photo.jpg result.png result.json
cd /d %~dp0

set "MODEL=%~1"
set "INPUT_IMAGE=%~2"
set "OUTPUT_IMAGE=%~3"
set "OUTPUT_POSE=%~4"

if "%MODEL%"=="" goto usage
if "%INPUT_IMAGE%"=="" goto usage
if "%OUTPUT_IMAGE%"=="" goto usage
if "%OUTPUT_POSE%"=="" goto usage

if /I not "%MODEL%"=="n" if /I not "%MODEL%"=="m" if /I not "%MODEL%"=="l" (
    echo Invalid model "%MODEL%". Use n, m, or l.
    exit /b 1
)

if not exist "%INPUT_IMAGE%" (
    echo Input image not found: %INPUT_IMAGE%
    exit /b 1
)

set "WEIGHTS=runs\xtx2-%MODEL%\best.pt"
if not exist "%WEIGHTS%" (
    echo Checkpoint not found: %WEIGHTS%
    echo Train the model first, or place best.pt in runs\xtx2-%MODEL%\
    exit /b 1
)

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

if not exist ".venv\.deps_installed" (
    echo Installing requirements ^(first run^)...
    uv pip install -r requirements.txt --python .venv\Scripts\python.exe
    if errorlevel 1 (
        echo Failed to install requirements.
        exit /b 1
    )
    type nul > ".venv\.deps_installed"
)

call ".venv\Scripts\activate.bat"

rem Prefer CUDA when available; infer.py falls back gracefully if not.
python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
if errorlevel 1 (
    set "DEVICE=cpu"
) else (
    set "DEVICE=cuda"
)

python infer.py ^
    --weights "%WEIGHTS%" ^
    --variant %MODEL% ^
    --image "%INPUT_IMAGE%" ^
    --save "%OUTPUT_IMAGE%" ^
    --json "%OUTPUT_POSE%" ^
    --viz-green-red ^
    --device %DEVICE%

if errorlevel 1 (
    echo Inference failed.
    exit /b 1
)

echo Done.
echo Annotated image: %OUTPUT_IMAGE%
echo Poses JSON:      %OUTPUT_POSE%
exit /b 0

:usage
echo Usage: infer_image.bat model input_image output_image output_pose
echo.
echo   model         : n, m, or l
echo   input_image   : path to input .jpg or .png
echo   output_image  : path to output .png with green boxes and red keypoints
echo   output_pose   : path to output .json with detected poses
exit /b 1
