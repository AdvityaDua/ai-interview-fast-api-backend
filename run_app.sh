#!/bin/bash

# Navigate to the script directory
cd "$(dirname "$0")"

# Activate the virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment 'venv' not found. Please create it first."
    exit 1
fi

# Fix macOS fork safety issue for Gunicorn workers
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Run the app using Gunicorn with Uvicorn worker (with --reload for development)
echo "Starting Ai for job Backend on port 8001..."
gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8001 --workers 2 --timeout 120 --reload
