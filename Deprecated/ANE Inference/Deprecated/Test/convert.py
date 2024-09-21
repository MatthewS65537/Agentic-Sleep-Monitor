import torch
import torch.nn as nn
import coremltools as ct

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

test_input = torch.rand(1, 10)  # Adjusted input size for SimpleMLP
model = SimpleMLP(input_size=10, hidden_size=20, output_size=5)
model.eval()

traced_model = torch.jit.trace(model, test_input)

# Convert to Core ML program using the Unified Conversion API.
model = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=test_input.shape)],
)

model.save("SimpleMLP.mlpackage")