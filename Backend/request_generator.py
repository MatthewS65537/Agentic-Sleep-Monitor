import requests
import json
import time
import os

# Flask server URL
base_url = "http://127.0.0.1:5000"

# Get list of audio files in the data/audio directory
audio_files = os.listdir("./data/audio")

audio_files = [file for file in audio_files if file.endswith(".wav")]
audio_files = sorted(audio_files, key=lambda x : int(x.split('.')[0][9:]))

st_time = time.time()
# Send audio requests for each file in the directory
for i, audio_file in enumerate(audio_files):
    print(f"Sending audio request for {audio_file}")
    audio_data = {
        "audio_string": audio_file,
        "timestamp": time.time() + i * 10
    }
    response = requests.post(f"{base_url}/audio/post_wav", json=audio_data)
    print(f"Audio POST response: {response.json()}")
    time.sleep(0.005)