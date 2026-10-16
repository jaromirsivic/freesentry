@echo off
rem XTX4 Pose - train u_b and then u_c (300 epochs each), one after another.
rem
rem After a crash / power cut / reboot just run this file again: finished variants are
rem skipped and the interrupted one continues from the last epoch that fully completed
rem (D:\xtx4_runs\f6_<variant>_300ep\epochs\epoch_NNNN.pt). Progress log:
rem D:\xtx4_runs\f6_combined.log. Extra options are passed on, e.g.
rem   train_combined.bat --dry-run                 (only show what would happen)
rem   train_combined.bat --set training.batch_size=48
rem Variants are trained sequentially on purpose: running them in parallel exhausted
rem the virtual memory and rebooted the PC, and was not faster (EXPERIMENTS.md).
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

".venv\Scripts\python.exe" scripts\train_combined.py --variants u_b u_c --epochs 300 %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
