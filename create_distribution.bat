@echo off
setlocal

set "SOURCE_DIR=%~dp0"
if "%SOURCE_DIR:~-1%"=="\" set "SOURCE_DIR=%SOURCE_DIR:~0,-1%"

for %%I in ("%SOURCE_DIR%\..") do set "PARENT_DIR=%%~fI"
set "TARGET_DIR=%PARENT_DIR%\freesentry_distribution"

echo Source: "%SOURCE_DIR%"
echo Target: "%TARGET_DIR%"
pause

if exist "%TARGET_DIR%" (
    choice /C YN /M "Target directory already exists. Do you want to delete it?"
    if errorlevel 2 exit /b 0
    echo deleting target directory
    echo this may take a few minutes...
    rmdir /S /Q "%TARGET_DIR%"
    if exist "%TARGET_DIR%" (
        echo Failed to delete target directory.
        exit /b 1
    )
)

if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

robocopy "%SOURCE_DIR%" "%TARGET_DIR%" /E /R:2 /W:2 ^
 /XD "%SOURCE_DIR%\.venv" "%SOURCE_DIR%\server\data" "%SOURCE_DIR%\server\data_startup" "%SOURCE_DIR%\server\wwwroot\node_modules" ".git" "__pycache__" ^
 /XF "command.md" "todo.txt"
set "ROBOCOPY_EXIT=%ERRORLEVEL%"
if %ROBOCOPY_EXIT% GEQ 8 (
    echo Root copy failed with code %ROBOCOPY_EXIT%.
    exit /b %ROBOCOPY_EXIT%
)

@REM mkdir "%TARGET_DIR%\server\data\logs" 2>nul
@REM mkdir "%TARGET_DIR%\server\data\project" 2>nul

@REM if exist "%SOURCE_DIR%\server\data\cursor" (
@REM     robocopy "%SOURCE_DIR%\server\data\cursor" "%TARGET_DIR%\server\data\cursor" /E /R:2 /W:2 /XD ".git" "__pycache__"
@REM     set "ROBOCOPY_EXIT=%ERRORLEVEL%"
@REM     if %ROBOCOPY_EXIT% GEQ 8 (
@REM         echo Cursor copy failed with code %ROBOCOPY_EXIT%.
@REM         exit /b %ROBOCOPY_EXIT%
@REM     )
@REM ) else (
@REM     echo Source folder not found: "%SOURCE_DIR%\server\data\cursor"
@REM )

@REM mkdir "%TARGET_DIR%\server\data\project\cache" 2>nul
@REM mkdir "%TARGET_DIR%\server\data\project\input" 2>nul
@REM mkdir "%TARGET_DIR%\server\data\project\work" 2>nul
@REM mkdir "%TARGET_DIR%\server\data\project\output" 2>nul
@REM mkdir "%TARGET_DIR%\server\data\project\temp" 2>nul

@REM if exist "%SOURCE_DIR%\server\data_startup" (
@REM     robocopy "%SOURCE_DIR%\server\data_startup" "%TARGET_DIR%\server\data" /E /R:2 /W:2 /XD ".git" "__pycache__"
@REM     set "ROBOCOPY_EXIT=%ERRORLEVEL%"
@REM     if %ROBOCOPY_EXIT% GEQ 8 (
@REM         echo Startup data copy failed with code %ROBOCOPY_EXIT%.
@REM         exit /b %ROBOCOPY_EXIT%
@REM     )
@REM ) else (
@REM     echo Source folder not found: "%SOURCE_DIR%\server\data_startup"
@REM )

echo Distribution prepared in "%TARGET_DIR%"
exit /b 0
