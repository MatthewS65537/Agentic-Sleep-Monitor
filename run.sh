#!/bin/bash

# Function to kill all background processes
cleanup() {
    echo "[MASTER/run.sh] Stopping all processes..."
    pkill -P $FLASK_PID
    pkill -P $AUDIO_RECORD_PID
    pkill -P $INFERENCE_ENGINE_PID
    pkill -P $DJANGO_PID
    kill $FLASK_PID $AUDIO_RECORD_PID $INFERENCE_ENGINE_PID $DJANGO_PID 2>/dev/null
    wait $FLASK_PID $AUDIO_RECORD_PID $INFERENCE_ENGINE_PID $DJANGO_PID 2>/dev/null
    echo "[MASTER/run.sh] All processes stopped."
    exit
}

# Set up trap to catch SIGINT (Ctrl+C)
trap cleanup SIGINT

cd Backend
# Start Flask app in the background
python3 Flask/app.py &
FLASK_PID=$!

# Wait for Flask to start up (adjust sleep time if needed)
sleep 2

# Run audio recording
python3 "Data Streaming/Audio Streaming/audio_record.py" --silent &
AUDIO_RECORD_PID=$!

# Run Inference Engine
python3 "Inference Engine/InferenceBackend.py" --engine "Lightning" &
INFERENCE_ENGINE_PID=$!

echo "[Master/run.sh] Backend processes started. Press Ctrl+C to stop."

cd ../Django
# Start Django server in the background
python3 manage.py runserver 0.0.0.0:8000 &
DJANGO_PID=$!

echo "[Master/run.sh] Django process started. Press Ctrl+C to stop."

cd ../
echo "[MASTER/run.sh] All processes started. Press Ctrl+C to stop."
# Wait for the background process
wait
