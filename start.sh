#!/bin/bash
cd /home/freesentry/freesentry
source .venv/bin/activate
uvicorn server.main:app --host 0.0.0.0 --port 80