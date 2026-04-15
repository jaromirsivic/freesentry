@echo off
echo --------------------------------------
echo Starting the server. This action 
echo typically takes 3 to 5 minutes to complete.
echo Make sure you have initialized the project
echo by running init.bat
echo --------------------------------------
@echo on
timeout /t 3 /nobreak
call .venv\Scripts\activate
uvicorn server.main:app --workers 1 --limit-concurrency 128 --host 0.0.0.0 --port 80
rem .venv\Scripts\python.exe -m uvicorn server.main:app --workers 1 --limit-concurrency 128 --host 0.0.0.0 --port 80
pause