import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
from model import *

st = time.time()

device = "cpu"
precision = torch.float32
model = ImageClassifierCNN(num_classes=4)
model.load_state_dict(torch.load("VisionCNN.pt"))
model = model.to(device, dtype=precision)

en = time.time()

print(f"[INFO] Loaded Model {en - st : .2f} seconds")

dataset = []
print("Generating Dataset")
for _ in tqdm(range(10000)):
	dataset.append(torch.rand(1, 1, 112, 112))

print("Starting Inference")
for data in tqdm(dataset):
	predictions = model(data.to(device, dtype=precision))