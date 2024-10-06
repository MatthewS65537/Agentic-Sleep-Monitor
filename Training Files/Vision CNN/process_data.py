import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root_dirs):
        self.image_paths = []
        self.labels = []
        
        for label, path in enumerate(root_dirs):
            for file in os.listdir(path):
                if file.endswith(".jpg"):
                    self.image_paths.append(os.path.join(path, file))
                    self.labels.append(label)
        
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Define paths
paths = ["./raw_data/down", "./raw_data/up", "./raw_data/left", "./raw_data/right"]

# Create dataset
dataset = ImageDataset(paths)

# Process all images
all_images = []
all_labels = []
for i in range(len(dataset)):
    image, label = dataset[i]
    all_images.append(image)
    all_labels.append(label)

# Convert to tensors
X = torch.stack(all_images)
y = torch.tensor(all_labels)

# Split the data
torch.manual_seed(42)
shuffle = torch.randperm(len(X))
X = X[shuffle]
y = y[shuffle]

train_split = int(0.8 * len(X))
X_train, X_test = X[:train_split], X[train_split:]
y_train, y_test = y[:train_split], y[train_split:]

# Save the processed data
torch.save(X_train, './data/X_train.pt')
torch.save(y_train, './data/y_train.pt')
torch.save(X_test, './data/X_test.pt')
torch.save(y_test, './data/y_test.pt')

print("Data processing and splitting complete.")
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Print label distributions
def print_label_distribution(labels, set_name):
    unique_labels, counts = torch.unique(labels, return_counts=True)
    print(f"\n{set_name} set label distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} ({count/len(labels)*100:.2f}%)")

print_label_distribution(y_train, "Training")
print_label_distribution(y_test, "Test")
