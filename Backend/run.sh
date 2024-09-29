#!/bin/bash

# Start Flask app in the background
python3 Flask/app.py &
FLASK_PID=$!

# Wait for Flask to start up (adjust sleep time if needed)
sleep 2

# Run request generator
python3 request_generator.py

# Run Inference Engine
python3 Inference\ Engine/InferenceBackend.py --engine "Lightning"

# Terminate Flask app
kill $FLASK_PID

# Wait for Flask to shut down gracefully
wait $FLASK_PID
