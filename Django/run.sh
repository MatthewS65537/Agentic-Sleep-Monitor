#!/bin/bash

# Function to kill all background processes
cleanup() {
    echo "[Django/run.sh] Stopping Django process..."
    kill $DJANGO_PID 2>/dev/null
    wait $DJANGO_PID 2>/dev/null
    echo "[Django/run.sh] Django process stopped."
    exit
}

# Set up trap to catch SIGINT (kill signal)
trap cleanup SIGINT

# Start Django server in the background
python3 manage.py runserver 0.0.0.0:8000 &
DJANGO_PID=$!

echo "[Django/run.sh] Django process started. Press Ctrl+C to stop."

# Wait for the background process
wait
