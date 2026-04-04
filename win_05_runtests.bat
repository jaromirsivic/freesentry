@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Missing virtual environment at .venv\Scripts\python.exe
    echo Run win_01_init.bat first.
    set "TEST_EXIT=1"
    goto summary
)

call .venv\Scripts\activate

echo Running Python test suite...
".venv\Scripts\python.exe" -m unittest discover -s tests -v
set "TEST_EXIT=%ERRORLEVEL%"

:summary
echo.
if "%TEST_EXIT%"=="0" (
    echo Test run PASSED.
) else (
    echo Test run FAILED with exit code %TEST_EXIT%.
)

pause
exit /b %TEST_EXIT%
