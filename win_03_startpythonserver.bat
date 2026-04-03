call .venv\Scripts\activate
.venv\Scripts\python.exe -m uvicorn server.main:app --reload --host 0.0.0.0 --port 80
pause