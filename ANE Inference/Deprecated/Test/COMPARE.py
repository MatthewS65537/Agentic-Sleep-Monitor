import torch
import coremltools as ct
import numpy as np
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

precision = torch.float32

# Load the model
num_classes = 5  # Adjust this based on your specific use case
model_CPU = AudioClassificationCNN(num_classes)
model_CPU.load_state_dict(torch.load("model.pt"))
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
model_ANE.save("AudioClassificationCNN.mlpackage")
model_ANE = ct.models.MLModel("AudioClassificationCNN.mlpackage")
# Create a dataset for benchmarking
dataset = [torch.rand(1, 32, 937) for _ in range(10)]  # Adjust input size

with torch.no_grad():
    for data in dataset:
        # Benchmark CPU
        predictions_CPU = model_CPU(data.to(dtype=precision))
        print("CPU Predictions:", predictions_CPU, torch.argmax(predictions_CPU).item())
        
        # Benchmark CoreML
        predictions_ANE = model_ANE.predict({'x_1': data.numpy()})
        print("CoreML Predictions:", predictions_ANE, np.argmax(predictions_ANE["linear_0"]))