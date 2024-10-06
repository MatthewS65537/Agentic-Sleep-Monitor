import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from model import ImageClassifierLightning
from pytorch_lightning.callbacks import EarlyStopping

# Load the preprocessed data
X_train = torch.load('./data/X_train.pt')
y_train = torch.load('./data/y_train.pt')
X_test = torch.load('./data/X_test.pt')
y_test = torch.load('./data/y_test.pt')

# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Define hyperparameters
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
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


import time

start = time.time()
# Initialize the model
model = ImageClassifierLightning(hparams)

# Define early stopping callback
early_stop_callback = EarlyStopping(
    monitor='val/accuracy',
    mode='max',
    patience=5,
    verbose=True
)

# Initialize Trainer with early stopping
trainer = pl.Trainer(
    max_epochs=20,
    callbacks=[early_stop_callback],
    # accelerator='cpu',
)

# Train the model
trainer.fit(model, train_loader, test_loader)

end = time.time()
print("Training complete.")
print(f"Time taken: {end - start} seconds")