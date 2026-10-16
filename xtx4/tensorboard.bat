@echo off
rem XTX4 Pose - TensorBoard launcher (double-clickable).
rem   tensorboard.bat                      -> all runs under D:\xtx4_runs (run.output_root), port 6006
rem   tensorboard.bat D:\xtx4_runs\f6_u_b_300ep        -> one run
rem   tensorboard.bat D:\xtx4_runs 6007    -> custom port
rem Starts TensorBoard in its own console window ("XTX4 TensorBoard") and opens the browser.
rem Stop it with stop_tensorboard.bat (or close that window).
cd /d %~dp0
setlocal

set "LOGDIR=%~1"
if "%LOGDIR%"=="" set "LOGDIR=D:\xtx4_runs"
set "PORT=%~2"
if "%PORT%"=="" set "PORT=6006"

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

if not exist "%LOGDIR%" (
    echo Log directory not found: %LOGDIR%
    pause
    exit /b 1
)

echo Starting TensorBoard on http://localhost:%PORT%/  (logdir %LOGDIR%)
start "XTX4 TensorBoard" /min ".venv\Scripts\python.exe" -m tensorboard.main --logdir "%LOGDIR%" --port %PORT% --bind_all --reload_interval 30
rem Give the server a moment, then open the browser.
timeout /t 4 /nobreak >nul
start "" "http://localhost:%PORT%/"
echo TensorBoard runs in the window "XTX4 TensorBoard"; stop it with stop_tensorboard.bat.
endlocal
