import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
import torchmetrics

class AudioClassificationTransformer(nn.Module):
    def __init__(self, num_classes=5, d_model=512, nhead=16, num_layers=6, dim_feedforward=1024, dropout=0.25):
        super(AudioClassificationTransformer, self).__init__()

        # 32 MFCC Channels
        self.input_projection = nn.Linear(32, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        output = self.classifier(x)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class AudioTransformer(pl.LightningModule):
    def __init__(self):
        super(AudioTransformer, self).__init__()
        self.model = AudioClassificationTransformer(num_classes=5)
        self.criterion = nn.CrossEntropyLoss()

        # Metrics for training
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", average="weighted", num_classes=5)
        self.train_precision = torchmetrics.classification.Precision(task="multiclass", average="weighted", num_classes=5)
        self.train_recall = torchmetrics.classification.Recall(task="multiclass", average="weighted", num_classes=5)
        self.train_f1 = torchmetrics.classification.F1Score(task="multiclass", average="weighted", num_classes=5)
        self.train_loss = torchmetrics.MeanMetric()

        # Metrics for validation
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", average="weighted", num_classes=5)
        self.val_precision = torchmetrics.classification.Precision(task="multiclass", average="weighted", num_classes=5)
        self.val_recall = torchmetrics.classification.Recall(task="multiclass", average="weighted", num_classes=5)
        self.val_f1 = torchmetrics.classification.F1Score(task="multiclass", average="weighted", num_classes=5)
        self.val_loss = torchmetrics.MeanMetric()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.train_loss(loss)
        self.train_accuracy(y_hat, y)
        self.train_precision(y_hat, y)
        self.train_recall(y_hat, y)
        self.train_f1(y_hat, y)

        # Log metrics
        self.log('train_loss', self.train_loss.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', self.train_accuracy.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_precision', self.train_precision.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_recall', self.train_recall.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1.compute(), on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        self.train_loss.reset()
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.val_loss(loss)
        self.val_accuracy(y_hat, y)
        self.val_precision(y_hat, y)
        self.val_recall(y_hat, y)
        self.val_f1(y_hat, y)

        # Log metrics
        self.log('val_loss', self.val_loss.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', self.val_accuracy.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_precision', self.val_precision.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recall', self.val_recall.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1.compute(), on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        self.val_loss.reset()
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=2e-5)
        return optimizer