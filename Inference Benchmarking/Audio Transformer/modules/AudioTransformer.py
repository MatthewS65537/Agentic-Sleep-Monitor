import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
class AudioClassificationTransformer(nn.Module):
    def __init__(self, num_classes, d_model=512, nhead=16, num_layers=6, dim_feedforward=1024, dropout=0.25):
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
        x = x.clone()
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
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> Dict[str, torch.Tensor]:
    if grads is None:
        grads = {n: p.grad.data.detach() for n, p in m.named_parameters() if p.requires_grad}

    for n, p in m.named_parameters():
        if p.requires_grad:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb

    return grads

class AudioTransformer(pl.LightningModule):
    def __init__(self, hparams):
        super(AudioTransformer, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = AudioClassificationTransformer(num_classes=self.hparams.num_classes)
        self.criterion = nn.CrossEntropyLoss()

        self.grokking_mode = self.hparams.get("grokking", None)
        if self.grokking_mode in ["GrokFastEMA", "DelayedGrokFastEMA"]:
            self.automatic_optimization = False
            self.grads = None
        
        if self.grokking_mode == "DelayedGrokFastEMA":
            self.delay_epoch = self.hparams.get("delay_epoch", 0)

        # Metrics for training
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", average="weighted", num_classes=self.hparams.num_classes)
        self.train_precision = torchmetrics.classification.Precision(task="multiclass", average="weighted", num_classes=self.hparams.num_classes)
        self.train_recall = torchmetrics.classification.Recall(task="multiclass", average="weighted", num_classes=self.hparams.num_classes)
        self.train_f1 = torchmetrics.classification.F1Score(task="multiclass", average="weighted", num_classes=self.hparams.num_classes)
        self.train_loss = torchmetrics.MeanMetric()

        # Metrics for validation
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", average="weighted", num_classes=self.hparams.num_classes)
        self.val_precision = torchmetrics.classification.Precision(task="multiclass", average="weighted", num_classes=self.hparams.num_classes)
        self.val_recall = torchmetrics.classification.Recall(task="multiclass", average="weighted", num_classes=self.hparams.num_classes)
        self.val_f1 = torchmetrics.classification.F1Score(task="multiclass", average="weighted", num_classes=self.hparams.num_classes)
        self.val_loss = torchmetrics.MeanMetric()

        # Metrics for testing
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", average="weighted", num_classes=self.hparams.num_classes)
        self.test_precision = torchmetrics.classification.Precision(task="multiclass", average="weighted", num_classes=self.hparams.num_classes)
        self.test_recall = torchmetrics.classification.Recall(task="multiclass", average="weighted", num_classes=self.hparams.num_classes)
        self.test_f1 = torchmetrics.classification.F1Score(task="multiclass", average="weighted", num_classes=self.hparams.num_classes)
        self.test_loss = torchmetrics.MeanMetric()

    def forward(self, x):
        return self.model(x)

    def is_grokfastema_active(self):
        if self.grokking_mode == "GrokFastEMA":
            return True
        elif self.grokking_mode == "DelayedGrokFastEMA":
            return self.current_epoch >= self.delay_epoch
        return False

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.long()
        loss = self.criterion(y_hat, y)

        if self.is_grokfastema_active():
            self.manual_backward(loss)
            self.grads = gradfilter_ema(self.model, grads=self.grads, alpha=self.hparams.alpha, lamb=self.hparams.lamb)

            # Update parameters
            self.optimizers().step()
            self.optimizers().zero_grad()
        elif self.grokking_mode == "DelayedGrokFastEMA":
            self.manual_backward(loss)
            self.optimizers().step()
            self.optimizers().zero_grad()

        # Update metrics
        self.train_loss(loss)
        self.train_accuracy(y_hat, y)
        self.train_precision(y_hat, y)
        self.train_recall(y_hat, y)
        self.train_f1(y_hat, y)

        # Log metrics
        self.log('train/loss', self.train_loss.compute(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/accuracy', self.train_accuracy.compute(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/precision', self.train_precision.compute(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/recall', self.train_recall.compute(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/f1', self.train_f1.compute(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        self.train_loss.reset()
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx = 0):
        if dataloader_idx == 0:
            x, y = batch
            y_hat = self(x)
            y = y.long()
            loss = self.criterion(y_hat, y)

            self.val_loss(loss)
            self.val_accuracy(y_hat, y)
            self.val_precision(y_hat, y)
            self.val_recall(y_hat, y)
            self.val_f1(y_hat, y)

            # Log metrics
            self.log('val/loss', self.val_loss.compute(), on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False, sync_dist=True)
            self.log('val/accuracy', self.val_accuracy.compute(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False, sync_dist=True)
            self.log('val/precision', self.val_precision.compute(), on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False, sync_dist=True)
            self.log('val/recall', self.val_recall.compute(), on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False, sync_dist=True)
            self.log('val/f1', self.val_f1.compute(), on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False, sync_dist=True)

        elif dataloader_idx == 1:
            x, y = batch
            y_hat = self(x)
            y = y.type(torch.int64)
            loss = self.criterion(y_hat, y)

            self.test_loss(loss)
            self.test_accuracy(y_hat, y)
            self.test_precision(y_hat, y)
            self.test_recall(y_hat, y)
            self.test_f1(y_hat, y)

            # Log Metrics
            self.log('test/loss', self.test_loss.compute(), on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False, sync_dist=True)
            self.log('test/accuracy', self.test_accuracy.compute(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False, sync_dist=True)
            self.log('test/precision', self.test_precision.compute(), on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False, sync_dist=True)
            self.log('test/recall', self.test_recall.compute(), on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False, sync_dist=True)
            self.log('test/f1', self.test_f1.compute(), on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False, sync_dist=True)

        return loss

    def on_validation_epoch_end(self):
        self.val_loss.reset()
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
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
