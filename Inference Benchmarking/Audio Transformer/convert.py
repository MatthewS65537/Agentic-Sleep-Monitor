import torch
import coremltools as ct
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from modules.AudioTransformer import *

precision = torch.float32

hparams = {
    "grokking": "None",
    "learning_rate": 3e-4,
    "optimizer": "AdamW",
    "scheduler": "ReduceLROnPlateau",
    "scheduler_factor": 0.67,
    "scheduler_patience": 10,
    "min_lr": 5e-7,
    "num_classes": 5,
    "model_name": "AudioTransformer",
    "precision": "16-mixed",
    "weight_decay": 0.003,
    "beta_1": 0.84,
    "beta_2": 0.86,
    "bsz": 64,
    "hardware" : "4 x RTX 4090"
}

model_lightning = AudioTransformer.load_from_checkpoint("AudioTransformer.ckpt")
torch.save(model_lightning.model.state_dict(), "AudioTransformer.pt")

# Load the model
num_classes = 5  # Adjust this based on your specific use case
model_CPU = AudioClassificationTransformer(num_classes)
model_CPU.load_state_dict(torch.load("AudioTransformer.pt"))
model_CPU = model_CPU.to("cpu", dtype=precision)
model_CPU.eval()

# Convert the model to Core ML
test_input = torch.rand(1, 32, 937)  # Adjust input size based on your audio data
traced_model = torch.jit.trace(model_CPU, test_input)
model_ANE = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=test_input.shape)],
)
model_ANE.save("AudioTransformer.mlpackage")
# model_ANE = ct.models.MLModel("AudioClassificationTransformer.mlpackage")

# # Create a dataset for benchmarking
# dataset = [torch.rand(1, 32, 937) for _ in range(10)]  # Adjust input size

# with torch.no_grad():
#     for data in dataset:
#         # Benchmark CPU
#         predictions_CPU = model_CPU(data.to(dtype=precision))
#         print("CPU Predictions:", predictions_CPU, torch.argmax(predictions_CPU).item())
        
#         # Benchmark CoreML
#         predictions_ANE = model_ANE.predict({'x_1': data.numpy()})
#         print("CoreML Predictions:", predictions_ANE, np.argmax(predictions_ANE["linear_25"]))
