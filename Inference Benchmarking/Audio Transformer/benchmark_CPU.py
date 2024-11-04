import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
from modules.AudioTransformer import *

st = time.time()

device = "cpu"
precision = torch.float32
model = AudioClassificationTransformer(num_classes=5)
model.load_state_dict(torch.load("AudioTransformer.pt"))
model = model.to(device, dtype=precision)

en = time.time()

print(f"[INFO] Loaded Model {en - st : .2f} seconds")

dataset = []
for _ in range(100):
	dataset.append(torch.rand(1, 32, 937))

for data in tqdm(dataset):
	predictions = model(data.to(device, dtype=precision))