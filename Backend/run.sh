#!/bin/bash

# Function to kill all background processes
cleanup() {
    echo "[Backend/run.sh] Stopping all processes..."
    kill $FLASK_PID $AUDIO_RECORD_PID $INFERENCE_ENGINE_PID 2>/dev/null
    wait $FLASK_PID $AUDIO_RECORD_PID $INFERENCE_ENGINE_PID 2>/dev/null
    echo "[Backend/run.sh] All processes stopped."
    exit
}

# Set up trap to catch SIGINT (kill signal)
trap cleanup SIGINT

# Start Flask app in the background
python3 Flask/app.py &
FLASK_PID=$!

# Wait for Flask to start up (adjust sleep time if needed)
sleep 2

# Run audio recording
# python3 "Data Streaming/Audio Streaming/audio_record.py" --silent &
# AUDIO_RECORD_PID=$!

# Run Inference Engine
python3 "Inference Engine/InferenceBackend.py" --engine "Lightning" --live &
INFERENCE_ENGINE_PID=$!

echo "[Backend/run.sh] All processes started. Press Ctrl+C to stop."

# Wait for all background processes
wait
