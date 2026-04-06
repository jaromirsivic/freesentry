#!/bin/bash
cd /home/freesentry/freesentry
source .venv/bin/activate
uvicorn server.main:app --workers 1 --limit-concurrency 128 --host 0.0.0.0 --port 80