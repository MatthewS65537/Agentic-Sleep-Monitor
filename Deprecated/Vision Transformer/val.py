import pytorch_lightning as pl
import torch
from modules.VisionTransformer import ViTLightning
from modules.Dataset import ImageDataset
from torch.utils.data import DataLoader

# Load data with weights_only=True to address the FutureWarning
X_train = torch.load("./data/images_train.pt", weights_only=True)
y_train = torch.load("./data/labels_train.pt", weights_only=True)
X_test = torch.load("./data/images_test.pt", weights_only=True)
y_test = torch.load("./data/labels_test.pt", weights_only=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define hyperparameters (same as in train.py)
hparams = {
    'num_classes': 4,
    'in_channels': 1,
    'patch_size': 8,
    'img_size': (96, 56),
    'd_model': 512,
    'num_layers': 6,
    'nhead': 16,
    'dim_feedforward': 1024,
    'dropout': 0.25,
    'precision': '16-mixed',
    'learning_rate': 5e-3,
    'scheduler': 'Exponential Decay',
    'scheduler_step_size': 20,
    'scheduler_gamma': 0.8,
    'weight_decay': 5e-5,
    'beta_1': 0.8,
    'beta_2': 0.85,
    'bsz': 4,  # 16, technically (4 x GPUs)
    'grokking': 'None',
    'hardware': '4 x RTX 4090'
}

# Create model
model = ViTLightning(hparams)

# Prepare datasets
train_dataset = ImageDataset(X_train, y_train)
test_dataset = ImageDataset(X_test, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=hparams['bsz'])
test_loader = DataLoader(test_dataset, batch_size=hparams['bsz'])

# Common trainer arguments
trainer_args = {
    "precision": hparams["precision"],
}

# GPU-specific settings
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')
    if torch.cuda.device_count() > 1:
        trainer_args["strategy"] = 'ddp'
    else:
        trainer_args["accelerator"] = "gpu"
        trainer_args["devices"] = [0]

# Create the trainer
trainer = pl.Trainer(**trainer_args, deterministic=True)

# Load the best model checkpoint
model = ViTLightning.load_from_checkpoint("path/to/best_model_checkpoint.ckpt", hparams=hparams)

# Run the validation on train set
print("Validating on training set:")
train_result = trainer.validate(model, dataloaders=train_loader)

# Run the test
print("\nValidating on test set:")
test_result = trainer.test(model, dataloaders=test_loader)

print(f"Train Result: {train_result}")
print(f"Val Result: {test_result}")
