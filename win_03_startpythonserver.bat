call .venv\Scripts\activate
uvicorn server.main:app --reload --host 0.0.0.0 --port 80
pause