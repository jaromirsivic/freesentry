@echo off
rem XTX3 live test - launcher. Extra arguments are passed to main.py, e.g.:
rem   start.bat --network xtx3-u --size 320 --camera 0 --no-dialog
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo The Python environment is not initialised yet - running init.bat ...
    call init.bat --auto
    if errorlevel 1 (
        echo Initialisation failed.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" main.py %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo live test exited with code %RC%
    pause
)
endlocal & exit /b %RC%
