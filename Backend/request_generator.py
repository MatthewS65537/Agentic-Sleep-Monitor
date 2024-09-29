import requests
import json
import time
import random

# Flask server URL
base_url = "http://localhost:5000"

# Function to generate fake vision data
def generate_fake_vision():
    return {
        "vision_string": f"fake_vision_{random.randint(1000, 9999)}",
        "timestamp": int(time.time())
    }

# Send fake audio requests
for i in range(1, 11):
    audio_data = {
        "audio_string": f"recording{i % 5 + 1}.wav",
        "timestamp": int(time.time())
    }
    response = requests.post(f"{base_url}/audio/post_wav", json=audio_data)
    print(f"Audio POST response: {response.json()}")
    time.sleep(0.1)

# # Send fake vision requests
# for _ in range(5):
#     vision_data = generate_fake_vision()
#     response = requests.post(f"{base_url}/vision/post_jpg", json=vision_data)
#     print(f"Vision POST response: {response.json()}")
#     time.sleep(1)

# # Get audio data
# for _ in range(5):
#     response = requests.get(f"{base_url}/audio/get_wav")
#     print(f"Audio GET response: {response.json()}")
#     time.sleep(1)

# # Get vision data
# for _ in range(5):
#     response = requests.get(f"{base_url}/vision/get_jpg")
#     print(f"Vision GET response: {response.json()}")
#     time.sleep(1)
