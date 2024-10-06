import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from model import ImageClassifierLightning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }

# Load the preprocessed data
X_train = torch.load('./data/X_train.pt')
y_train = torch.load('./data/y_train.pt')
X_test = torch.load('./data/X_test.pt')
y_test = torch.load('./data/y_test.pt')

# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Define hyperparameters (same as training script)
hparams = {
    'num_classes': 4,
    'learning_rate': 5e-4,
    'weight_decay': 1e-5,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'scheduler_factor': 0.5,
    'scheduler_patience': 5,
    'min_lr': 1e-6,
    "bsz" : 4,
    'hardware' : "Mac M1 Pro (4E+4P+8GPU) ",
}

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Load the trained model
try:
    model = ImageClassifierLightning.load_from_checkpoint("Checkpoint_MPS.ckpt", hparams=hparams)
    model.eval()
except FileNotFoundError:
    print("Model checkpoint not found. Please ensure the model has been trained and saved correctly.")
    exit(1)

# Initialize Trainer for evaluation
trainer = pl.Trainer()

# Evaluate on train set
train_results = trainer.validate(model, train_loader)[0]
y_train_pred = model(X_train).argmax(dim=1).cpu().numpy()
train_metrics = compute_metrics(y_train.cpu().numpy(), y_train_pred)

# Evaluate on test set
test_results = trainer.validate(model, test_loader)[0]
y_test_pred = model(X_test).argmax(dim=1).cpu().numpy()
test_metrics = compute_metrics(y_test.cpu().numpy(), y_test_pred)

# Print results
print("Train Metrics:")
for k, v in train_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nTest Metrics:")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")