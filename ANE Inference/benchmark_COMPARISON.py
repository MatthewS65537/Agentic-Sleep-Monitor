import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import coremltools as ct
import numpy as np
from tqdm import tqdm
import time

class AudioClassificationTransformer(nn.Module):
    def __init__(self, num_classes, d_model=512, nhead=16, num_layers=6, dim_feedforward=1024, dropout=0.25):
        super(AudioClassificationTransformer, self).__init__()

        # 32 MFCC Channels
        self.input_projection = nn.Linear(32, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        output = self.classifier(x)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

model_ANE = ct.models.MLModel('AudioClassificationTransformer.mlmodel')

precision = torch.float32

model_CPU = AudioClassificationTransformer(num_classes=5)
model_MPS = AudioClassificationTransformer(num_classes=5)
model_CPU.load_state_dict(torch.load("AudioClassifier.pt"))
model_MPS.load_state_dict(torch.load("AudioClassifier.pt"))
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