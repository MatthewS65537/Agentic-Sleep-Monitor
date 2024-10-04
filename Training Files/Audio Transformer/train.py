import pytorch_lightning as pl
import torch
from modules.AudioTransformer import *
from modules.Dataset import *
from torch.utils.data import DataLoader

# Load data
MFCC_Tensors = torch.load("./data/MFCC_Tensors.pt")
Label_Tensors = torch.load("./data/Label_Tensors.pt")
MFCC_Val_Tensors = torch.load("./data/MFCC_Val_Tensors.pt")
Label_Val_Tensors = torch.load("./data/Label_Val_Tensors.pt")

train_split = int(MFCC_Tensors.shape[0] * 0.8)
X_train = MFCC_Tensors[:train_split]
y_train = Label_Tensors[:train_split]
X_dev = MFCC_Tensors[train_split:]
y_dev = Label_Tensors[train_split:]
X_test = MFCC_Val_Tensors
y_test = Label_Val_Tensors

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define hyperparameters
hparams = {
    "grokking": "None",
    "learning_rate": 1e-4,
    "optimizer": "AdamW",
    "scheduler": "ReduceLROnPlateau",
    "scheduler_factor": 0.5,
    "scheduler_patience": 5,
    "min_lr": 1e-6,
    "num_classes": 5,
    "model_name": "AudioTransformer",
    "precision": "16-mixed",
    "weight_decay": 0.01,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "bsz": 256
}

# Create model
model = AudioTransformer(hparams)

# Prepare datasets and data module
bsz = hparams["bsz"]
train_dataset = MFCCDataset(X_train, y_train)
dev_dataset = MFCCDataset(X_dev, y_dev)
test_dataset = MFCCDataset(X_test, y_test)
data_module = AudioDataModule(train_dataset, dev_dataset, test_dataset, batch_size=bsz)

# Set up trainer
trainer = pl.Trainer(
    logger=True,
    enable_checkpointing=True,
    enable_progress_bar=True,
    max_epochs=50,
    log_every_n_steps=1,
    precision=hparams['precision'],
)

# Train the model
trainer.fit(model, datamodule=data_module)

# Evaluate the model
test_result = trainer.test(model, datamodule=data_module)

print("Test Result:")
print(f"  Accuracy: {test_result[0]['test/accuracy']:.4f}")

print("Hyperparameters:")
for key, value in hparams.items():
    print(f"  {key}: {value}")