import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import coremltools as ct
import numpy as np
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioClassificationCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassificationCNN, self).__init__()
        # Convolutional layer (input channels, output channels, kernel size)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(256, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        # Pooling layer to reduce dimensions
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # Adaptive pooling to make it size-independent
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Apply convolutional layers followed by pooling
        x = x.unsqueeze(1)
        # x = x.permute(1, 0, 2, 3)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        # Adaptive pooling and flattening
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        # Fully connected layer for classification
        x = self.fc(x)
        return x

model_ANE = ct.models.MLModel('AudioClassificationCNN.mlpackage')

precision = torch.float32

model_CPU = AudioClassificationCNN(num_classes=5)
model_MPS = AudioClassificationCNN(num_classes=5)
model_CPU.load_state_dict(torch.load("AudioCNNLarge.pt"))
model_MPS.load_state_dict(torch.load("AudioCNNLarge.pt"))
model_CPU = model_CPU.to("cpu", dtype=precision)
model_MPS = model_MPS.to("mps", dtype=precision)
model_CPU.eval()
model_MPS.eval()

np.random.seed(42)

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
		print("ANE:", predictions_ANE, np.argmax(predictions_ANE["linear_0"]))