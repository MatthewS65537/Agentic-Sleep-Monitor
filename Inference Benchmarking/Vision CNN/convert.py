import torch
import coremltools as ct
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from model import *

precision = torch.float32

hparams = {
    'num_classes': 4,
    'learning_rate': 5e-4,
    'weight_decay': 1e-5,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'scheduler_factor': 0.5,
    'scheduler_patience': 5,
    'min_lr': 1e-6,
    "bsz" : 4,
    'hardware' : "Mac M1 Pro (4E+4P+8GPU) ",
}

model_lightning = ImageClassifierLightning.load_from_checkpoint("VisionCNN.ckpt")
torch.save(model_lightning.model.state_dict(), "VisionCNN.pt")

# Load the model
num_classes = 4  # Adjust this based on your specific use case
model_CPU = ImageClassifierCNN(num_classes)
model_CPU.load_state_dict(torch.load("VisionCNN.pt"))
model_CPU = model_CPU.to("cpu", dtype=precision)
model_CPU.eval()

# Convert the model to Core ML
test_input = torch.rand(1, 1, 112, 112)  # Adjust input size based on your audio data
traced_model = torch.jit.trace(model_CPU, test_input)
model_ANE = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=test_input.shape)],
)
model_ANE.save("VisionCNN.mlpackage")
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
