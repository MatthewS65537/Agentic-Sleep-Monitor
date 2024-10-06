import torch
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.clone().detach(), dtype=torch.float32)
        self.y = torch.tensor(y.clone().detach(), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]