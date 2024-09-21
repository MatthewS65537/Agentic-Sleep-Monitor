import torch
import torch.nn as nn

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

num_classes = 10  # Adjust this based on your specific use case
model = AudioClassificationCNN(num_classes)
torch.save(model.state_dict(), "model.pt")