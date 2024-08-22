import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import threading
import os
from tqdm import tqdm
import requests
import time

def save_recording(path, freq, recording):
    write(path, freq, recording)
    print(f"Saved recording: {path}")

    url = 'http://127.0.0.1:5000/audio/post_wav'
    data = {
        'audio_string': f"{path.split('/')[-1]}",
        'timestamp' : time.time()
    }
    response = requests.post(url, json=data)

if __name__ == "__main__":
    # Create a directory for recordings if it doesn't exist
    os.makedirs("../../data/recordings", exist_ok=True)

    devices = sd.query_devices()
    device_idx = 0
    freq = 0

    for device in devices:
        if device["name"] == "USBAudio1.0":
            device_idx = device["index"]
            freq = int(device["default_samplerate"])

    sd.default.device = device_idx

    # time.sleep(30)

    try:
        i = 0
        while True:
        # for i in tqdm(range(100)):
            print(f"Started Recording No. {i}")
            duration = 10
            recording = sd.rec(int(duration * freq), 
                            samplerate=freq, channels=1)
            sd.wait()
            
            # Create a new thread for saving the recording
            thread = threading.Thread(target=save_recording, 
                                      args=(f"../../data/recordings/recording{i}.wav", freq, recording))
            thread.start()
            i += 1
            # Don't wait for the thread to finish; continue with the next recording
    except KeyboardInterrupt:
        print("[INFO/audio_record.py] Terminated by user.")
    finally:
        # Wait for all threads to complete before exiting
        for thread in threading.enumerate():
            if thread != threading.current_thread():
                thread.join()

        print("[INFO/audio_record.py] All recordings completed and saved.")