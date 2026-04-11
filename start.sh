#!/bin/bash
echo "--------------------------------------"
echo "Starting the server. This action"
echo "typically takes 3 to 5 minutes to complete."
echo "Make sure you have initialized the project"
echo "by running init.sh"
echo "--------------------------------------"
#cd /home/freesentry/freesentry
source .venv/bin/activate
uvicorn server.main:app --workers 1 --limit-concurrency 128 --host 0.0.0.0 --port 80