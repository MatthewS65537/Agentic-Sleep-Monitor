import requests
import time
import argparse
import csv
import os
from Audio.Engine import AudioEngine
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

# Argument parser for front and backend URLs
parser = argparse.ArgumentParser(description='Inference Backend for Audio and Vision')
parser.add_argument('--backend_url', type=str, default="http://127.0.0.1:5000", help='Backend server URL')
parser.add_argument('--data_path', type=str, default="./data/", help='Path to data')
parser.add_argument('--engine', type=str, default="Lightning", help='Inference engine to use')

def get_audio():
    response = requests.get(f"{args.backend_url}/audio/get_wav")
    return response.json()

def get_vision():
    response = requests.get(f"{args.backend_url}/vision/get_jpg")
    return response.json()

def log_audio(csv_writer, tstamp, prediction):
    """Log the timestamp and prediction to the audio CSV file."""
    csv_writer.writerow([tstamp, prediction])
    # Removed flush() as it is not a valid method for csv.writer

def log_vision(csv_writer, tstamp, prediction):
    """Log the timestamp and prediction to the vision CSV file."""
    csv_writer.writerow([tstamp, prediction])
    # Removed flush() as it is not a valid method for csv.writer

if __name__ == "__main__":
    args = parser.parse_args()  # Moved argparser to the main loop

    audio_engine = AudioEngine(
        platform=args.engine,
        model_name="AudioClassificationTransformer",
    )

    # Create results directory if it doesn't exist
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'audio'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'vision'), exist_ok=True)

    # Create a CSV file to log audio results
    program_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_csv_file_path = os.path.join(results_dir, f'audio/audio_{program_start_time}.csv')
    vision_csv_file_path = os.path.join(results_dir, f'vision/vision_{program_start_time}.csv')

    audio_csv_file = open(audio_csv_file_path, mode='w', newline='')
    audio_csv_writer = csv.writer(audio_csv_file)
    audio_csv_writer.writerow(['Timestamp', 'Prediction'])  # Write header
    audio_csv_file.close()

    # vision_csv_file = open(vision_csv_file_path, mode='w', newline='')
    # vision_csv_writer = csv.writer(vision_csv_file)
    # vision_csv_writer.writerow(['Timestamp', 'Prediction'])  # Write header

    while True:
        wav = get_audio()
        msg = wav['info']
        if not msg == "Queue is empty":
            tstamp = wav['timestamp']
            prediction = audio_engine.predict(args.data_path + 'audio/' + msg)
            print(prediction)

            # Log the timestamp and prediction to the audio CSV file
            audio_csv_file = open(audio_csv_file_path, mode='a', newline='')
            audio_csv_writer = csv.writer(audio_csv_file)
            log_audio(audio_csv_writer, tstamp, prediction)
            # Manually flush to ensure data is written
            audio_csv_file.close()

        # TODO: VISION COMPONENT

        time.sleep(5)
