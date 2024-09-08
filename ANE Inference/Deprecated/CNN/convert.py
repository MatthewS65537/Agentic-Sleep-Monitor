import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

torch.manual_seed(42)
precision = torch.float32

test_input = torch.rand(1, 32, 937)
model = AudioClassificationCNN(num_classes=5)
model = model.to("cpu", dtype=precision)
model.eval()

traced_model = torch.jit.trace(model, test_input)
# out = traced_model(test_input)

import coremltools as ct

# Using image_input in the inputs parameter:
# Convert to Core ML program using the Unified Conversion API.
model = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=test_input.shape)],
 )

model.save("AudioCNN.mlpackage")

# Print model description
# print(model.description)

# Print input and output descriptions
# print("Inputs:", model.input_description)
# print("Outputs:", model.output_description)
