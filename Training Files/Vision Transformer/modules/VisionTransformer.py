import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics


class PatchEmbedding(nn.Module):
    """Converts an image into patches and then into linear embeddings."""

    def __init__(self, in_channels: int = 1, patch_size: int = 8, emb_size: int = 512):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        b, e, h, w = x.shape
        x = x.permute(0, 2, 3, 1)  # b, h, w, e
        x = x.reshape(b, h * w, e)
        return x


class PositionalEncoding(nn.Module):
    """Adds positional encoding to the token embeddings."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ViT(nn.Module):
    """Vision Transformer (ViT) model."""

    def __init__(self, num_classes: int, in_channels: int = 1, patch_size: int = 8, 
                 img_size: Tuple[int, int] = (56, 92), d_model: int = 512, num_layers: int = 6, 
                 nhead: int = 16, dim_feedforward: int = 1024, dropout: float = 0.25):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, d_model)
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer(x)

        x = x[:, 0]

        x = self.ln(x)
        x = self.fc(x)
        return x

    def count_params(self) -> float:
        """Count the number of trainable parameters in millions."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6


def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """Apply Exponential Moving Average (EMA) to gradients."""
    if grads is None:
        grads = {n: p.grad.data.detach() for n, p in m.named_parameters() if p.requires_grad}

    for n, p in m.named_parameters():
        if p.requires_grad:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb

    return grads


class ViTLightning(pl.LightningModule):
    """PyTorch Lightning module for Vision Transformer."""

    def __init__(self, hparams):
        super(ViTLightning, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = ViT(
            num_classes=self.hparams.num_classes,
            in_channels=self.hparams.in_channels,
            patch_size=self.hparams.patch_size,
            img_size=self.hparams.img_size,
            d_model=self.hparams.d_model,
            num_layers=self.hparams.num_layers,
            nhead=self.hparams.nhead,
            dim_feedforward=self.hparams.dim_feedforward,
            dropout=self.hparams.dropout,
        )
        self.criterion = nn.CrossEntropyLoss()

        if self.hparams.get("grokking") == "GrokFastEMA":
            self.automatic_optimization = False
        
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.type(torch.int64)
        loss = self.criterion(y_hat, y)

        # Apply gradient modification if grokking is set to "GrokFastEMA"
        if self.hparams.get("grokking") == "GrokFastEMA":
            self.manual_backward(loss)
            if not hasattr(self, "grads"):
                self.grads = None
            self.grads = gradfilter_ema(self.model, grads=self.grads)
            
            # Update parameters
            self.optimizers().step()
            self.optimizers().zero_grad()

        # Metrics update and logging
        self.train_loss(loss)
        self.train_accuracy(y_hat, y)
        self.train_precision(y_hat, y)
        self.train_recall(y_hat, y)
        self.train_f1(y_hat, y)
        
        # Log Metrics
        self.log('train/loss', self.train_loss.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/accuracy', self.train_accuracy.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/precision', self.train_precision.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/recall', self.train_recall.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/f1', self.train_f1.compute(), on_step=False, on_epoch=True, prog_bar=True)

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
        y = y.type(torch.int64)
        loss = self.criterion(y_hat, y)

        self.val_loss(loss)
        self.val_accuracy(y_hat, y)
        self.val_precision(y_hat, y)
        self.val_recall(y_hat, y)
        self.val_f1(y_hat, y)
        self.log('val/loss', self.val_loss.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/accuracy', self.val_accuracy.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/precision', self.val_precision.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/recall', self.val_recall.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/f1', self.val_f1.compute(), on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        self.val_loss.reset()
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
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
