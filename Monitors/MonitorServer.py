import os
import sys
import platform
import time
import argparse
from tqdm import tqdm

sys.path.append("./Audio Monitor")

import torch
import numpy as np
from AudioMonitorModules import *

import requests

if "macOS" in platform.platform() and "arm64" in platform.platform():
    import coremltools as ct

def get_inference_platform():
    platform_str = platform.platform()
    if "Windows" in platform_str or "Linux" in platform_str:
        return "Lightning"
    elif "macOS" in platform_str:
        if "arm64" in platform_str:
            # Apple Silicon Mac, has NPU
            return "ANE"
        else:
            return "Lightning"
    else:
        return "Lightning"
    
class UniversalInferenceEngine():
    def __init__(self, platform, model_name, checkpoint_dir = "../model_checkpoints", model_obj = None):
        self.platform = platform
        path_no_ext = os.path.join(checkpoint_dir, model_name)
        if self.platform == "Lightning":
            # Pytorch Lightning Inference
            self.model = model_obj.load_from_checkpoint(path_no_ext + ".ckpt")
            self.model.eval()
        elif self.platform == "ANE":
            # CoreML Inference
            self.model = ct.models.MLModel(path_no_ext + ".mlpackage")
        else:
            if self.platform in ["cpu", "cuda", "mps"]:
                # Pytorch Inference
                self.model = model_obj()
                # Surpress Argument with weights_only=True
                self.model.load_state_dict(torch.load(path_no_ext + ".pt", weights_only=True))
                self.model.to(self.platform, dtype=torch.float32)
                self.model.eval()
            else:
                assert(False)
    
    def predict(self, data):
        # data -> numpy array of shape [1, 32, 937]
        if self.platform == "Lightning":
            data = torch.tensor(data).to(dtype=torch.float32)
            logits = self.model(data)
            # print(f"[DEBUG] {logits}")
            label_idx = torch.argmax(logits)
        elif self.platform == "ANE":
            data = data.detach().numpy().astype(np.float32)
            logits = self.model.predict({'x_1' : data})["linear_25"][0]
            # print(f"[DEBUG] {logits}")
            label_idx = np.argmax(logits)
        else:
            if self.platform in ["cpu", "cuda", "mps"]: 
                data = torch.tensor(data)
                logits = self.model(data.to(self.platform, dtype=torch.float32))
                # print(f"[DEBUG] {logits}")
                label_idx = torch.argmax(logits)
            else:
                assert(False)
        return label_idx.item()

import csv
import os

def log_to_csv(timestamp, label, log_path):
    """
    Append an entry to a CSV log file.
    
    :param timestamp: float, the timestamp of the audio classification
    :param label: str, the predicted label
    :param log_file: str, the name of the log file (default: 'audio_classification_log.csv')
    """
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Check if the file exists, if not, create it with a header
    file_exists = os.path.isfile(log_path)
    
    with open(log_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header if the file is newly created
        if not file_exists:
            writer.writerow(['Timestamp', 'Label'])
        
        # Append the new entry
        writer.writerow([timestamp, label])


if __name__ == "__main__":
    start_time = time.time()
    logpath = f'../../data/audio_logs/audio_classification_log_{start_time}.csv'
    parser = argparse.ArgumentParser(description="Audio Classification Model")
    parser.add_argument("--device", type=str, default="Torch", help="Specify the device to use (e.g., 'cpu', 'cuda', 'mps')")
    args = parser.parse_args()
    
    inf_platform = args.device
    if inf_platform == None:
        inf_platform = get_inference_platform()
    
    if inf_platform == "Torch":
        inf_platform = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"[INFO] Selected Inference Platform \"{inf_platform}\"")

    AudioEndpoint = "http://127.0.0.1:5000/audio/get_wav"
    AudioModel = UniversalInferenceEngine(
        inf_platform,
        model_name = "AudioClassificationTransformer",
        model_obj = AudioTransformer if inf_platform == "Lightning" else AudioClassificationTransformer
    )

    print("[INFO] Loaded Audio Model.")

    reading_no = 0
    # while True:
    #     time.sleep(5)

    #     response = requests.get(AudioEndpoint)
    #     # print(response.json())
    #     msg = response.json()["info"]
    #     if msg == "Queue is empty":
    #         print(time.time(), msg)
    #         continue
    #     audio_time = response.json()["timestamp"]
    #     start_time = time.time()
    #     MFCC = load_wav_into_tensor(f"../data/recordings/{msg}")
    #     MFCC = MFCC.reshape([1, 32, 937]).to(dtype=torch.float32)
    #     label_idx = AudioModel.predict(MFCC)
    #     end_time = time.time()
        
    #     labels = ["NormalSleep", "Hypopnea", "Snore", "ObstructiveApnea", "MixedApnea"]
    #     label = labels[label_idx]

    #     print(f"Reading No. {reading_no}; {audio_time:.2f}; {label}")
    #     log_to_csv(audio_time, label, logpath)
    #     reading_no += 1

    logpath = f'../data/audio_logs/audio_classification_log_AUG13DEMO.csv'
    for i in tqdm(range(1, 2319)):
        start_time = time.time()
        MFCC = load_wav_into_tensor(f"../data/demo_recordings/recording{i}.wav")
        MFCC = MFCC.reshape([1, 32, 937]).to(dtype=torch.float32)
        label_idx = AudioModel.predict(MFCC)
        end_time = time.time()

        audio_time = int(os.path.getctime(f"../data/demo_recordings/recording{i}.wav"))
        
        labels = ["NormalSleep", "Hypopnea", "Snore", "ObstructiveApnea", "MixedApnea"]
        label = labels[label_idx]

        # print(f"Reading No. {i}; {audio_time:.2f}; {label}")
        log_to_csv(audio_time, label, logpath)
        