call .venv\Scripts\activate
uvicorn server.main:app --workers 1 --limit-concurrency 128 --host 0.0.0.0 --port 80
# .venv\Scripts\python.exe -m uvicorn server.main:app --workers 1 --limit-concurrency 128 --host 0.0.0.0 --port 80
pause