import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import coremltools as ct
import numpy as np
from tqdm import tqdm
import time
from modules.AudioTransformer import *

model_ANE = ct.models.MLModel('AudioTransformer.mlpackage')

precision = torch.float32

model_CPU = AudioClassificationTransformer(num_classes=5)
model_MPS = AudioClassificationTransformer(num_classes=5)
model_CPU.load_state_dict(torch.load("AudioTransformer.pt"))
model_MPS.load_state_dict(torch.load("AudioTransformer.pt"))
model_CPU = model_CPU.to("cpu", dtype=precision)
model_MPS = model_MPS.to("mps", dtype=precision)
model_CPU.eval()
model_MPS.eval()

dataset = []
for _ in range(10):
	dataset.append(np.random.randn(1, 32, 937))

with torch.no_grad():
	for data in dataset:
		tensor_data = torch.tensor(data).to(dtype=precision)
		predictions_CPU = model_CPU(tensor_data.to("cpu"))
		print("CPU:", predictions_CPU, torch.argmax(predictions_CPU).item())
		predictions_MPS = model_MPS(tensor_data.to("mps"))
		print("MPS:", predictions_MPS, torch.argmax(predictions_MPS).item())
		predictions_ANE = model_ANE.predict({'x_1' : data})
		print("ANE:", predictions_ANE, np.argmax(predictions_ANE["linear_25"]))