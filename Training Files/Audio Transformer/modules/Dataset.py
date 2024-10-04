from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class MFCCDataset(Dataset):
    """
    Dataset class for MFCC (Mel-frequency cepstral coefficients) features.

    This dataset is designed to work with MFCC features for audio classification tasks.

    Args:
        X (torch.Tensor): Input MFCC features.
        y (torch.Tensor): Corresponding labels.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = torch.tensor(X.clone().detach(), dtype=torch.float32)
        self.y = torch.tensor(y.clone().detach(), dtype=torch.float32)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the MFCC features and the corresponding label.
        """
        return self.X[idx], self.y[idx]


class AudioDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for audio data.

    This DataModule handles the creation of DataLoaders for training and validation datasets.

    Args:
        train_data (Dataset): Training dataset.
        val_data1 (Dataset): First validation dataset.
        val_data2 (Dataset): Second validation dataset.
        batch_size (int): Batch size for DataLoaders. Default is 16.
    """

    def __init__(self, train_data: Dataset, val_data1: Dataset, val_data2: Dataset, batch_size: int = 16):
        super().__init__()
        self.train_data = train_data
        self.val_data1 = val_data1
        self.val_data2 = val_data2
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        """
        Creates the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> List[DataLoader]:
        """
        Creates DataLoaders for the validation datasets.

        Returns:
            List[DataLoader]: A list containing DataLoaders for both validation datasets.
        """
        return [
            DataLoader(self.val_data1, batch_size=self.batch_size),
            DataLoader(self.val_data2, batch_size=self.batch_size)
        ]
