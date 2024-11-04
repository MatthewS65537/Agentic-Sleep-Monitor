import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics

class ImageClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Changed kernel_size to 3
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Changed kernel_size to 3
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ImageClassifierLightning(pl.LightningModule):
    def __init__(self, hparams):
        super(ImageClassifierLightning, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = ImageClassifierCNN(num_classes=self.hparams.num_classes)
        self.criterion = nn.CrossEntropyLoss()

        # Metrics for training
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.train_precision = torchmetrics.Precision(task="multiclass", num_classes=self.hparams.num_classes)
        self.train_recall = torchmetrics.Recall(task="multiclass", num_classes=self.hparams.num_classes)
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes)
        self.train_loss = torchmetrics.MeanMetric()

        # Metrics for validation
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_loss = torchmetrics.MeanMetric()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # Update metrics
        self.train_loss(loss)
        self.train_accuracy(y_hat, y)
        self.train_precision(y_hat, y)
        self.train_recall(y_hat, y)
        self.train_f1(y_hat, y)

        # Log metrics
        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/accuracy', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/precision', self.train_precision, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/recall', self.train_recall, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # Update metrics
        self.val_loss(loss)
        self.val_accuracy(y_hat, y)
        self.val_precision(y_hat, y)
        self.val_recall(y_hat, y)
        self.val_f1(y_hat, y)

        # Log metrics
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val/accuracy', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val/recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(self.hparams.beta_1, self.hparams.beta_2)
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience,
            min_lr=self.hparams.min_lr
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/accuracy',
                'interval': 'epoch',
                'frequency': 1
            }
        }
