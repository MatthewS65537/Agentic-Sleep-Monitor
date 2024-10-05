import pytorch_lightning as pl
import torch
from modules.AudioTransformer import AudioTransformer
from modules.Dataset import MFCCDataset, AudioDataModule
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

# Define hyperparameters (same as in train.py)
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

# Load the trained model
model = AudioTransformer.load_from_checkpoint("Checkpoint.ckpt", hparams=hparams)
model.eval()

# Prepare datasets and data module
bsz = hparams["bsz"]
train_dataset = MFCCDataset(X_train, y_train)
dev_dataset = MFCCDataset(X_dev, y_dev)
test_dataset = MFCCDataset(X_test, y_test)
data_module = AudioDataModule(train_dataset, dev_dataset, test_dataset, batch_size=bsz)

# GPU-specific settings
trainer_args = {
    "precision": hparams["precision"],
}

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')
    if torch.cuda.device_count() > 1:
        trainer_args["strategy"] = 'ddp'
    else:
        trainer_args["accelerator"] = "gpu"
        trainer_args["devices"] = [0]

# Create the trainer
trainer = pl.Trainer(**trainer_args, deterministic=True)

# Perform validation on all splits
print("Validating on training set:")
trainer.validate(model, dataloaders=DataLoader(train_dataset, batch_size=bsz))

print("\nValidating on development set:")
trainer.validate(model, dataloaders=DataLoader(dev_dataset, batch_size=bsz))

print("\nValidating on test set:")
trainer.validate(model, dataloaders=DataLoader(test_dataset, batch_size=bsz))
