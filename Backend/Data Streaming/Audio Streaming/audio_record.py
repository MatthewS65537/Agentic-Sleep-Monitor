import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import threading
import os
from tqdm import tqdm
import requests
import time
import argparse

def save_recording(path, freq, recording, silent=False):
    write(path, freq, recording)
    if not silent:
        print(f"Saved recording: {path}")

    url = 'http://127.0.0.1:5000/audio/post_wav'
    data = {
        'audio_string': f"{path.split('/')[-1]}",
        'timestamp' : time.time()
    }
    response = requests.post(url, json=data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio recording script')
    parser.add_argument('--device_name', type=str, default="MacBook Pro Microphone", help='Name of the audio device to use')
    parser.add_argument('--silent', action='store_true', help='Run in silent mode without printing progress')
    args = parser.parse_args()

    # Create a directory for recordings if it doesn't exist
    os.makedirs("../../data/recordings", exist_ok=True)

    devices = sd.query_devices()
    device_idx = 0
    freq = 0

    for device in devices:
        if not args.silent:
            print(device["name"])
        if device["name"] == args.device_name:
            device_idx = device["index"]
            freq = int(device["default_samplerate"])

    sd.default.device = device_idx

    # time.sleep(30)

    try:
        i = 0
        while True:
        # for i in tqdm(range(100)):
            if not args.silent:
                print(f"Started Recording No. {i}")
            duration = 10
            recording = sd.rec(int(duration * freq), 
                            samplerate=freq, channels=1)
            sd.wait()
            
            # Create a new thread for saving the recording
            thread = threading.Thread(target=save_recording, 
                                      args=(f"./data/audio/recording{i}.wav", freq, recording, args.silent))
            thread.start()
            i += 1
            # Don't wait for the thread to finish; continue with the next recording
    except KeyboardInterrupt:
        if not args.silent:
            print("[INFO/audio_record.py] Terminated by user.")
    finally:
        # Wait for all threads to complete before exiting
        for thread in threading.enumerate():
            if thread != threading.current_thread():
                thread.join()

        if not args.silent:
            print("[INFO/audio_record.py] All recordings completed and saved.")