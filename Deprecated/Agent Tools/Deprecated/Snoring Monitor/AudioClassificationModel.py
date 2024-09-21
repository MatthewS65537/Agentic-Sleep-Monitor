import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AudioClassificationTransformer(nn.Module):
    def __init__(self, num_classes, d_model=256, nhead=16, num_layers=6, dim_feedforward=1024, dropout=0.25):
        super(AudioClassificationTransformer, self).__init__()

        # Input shape: [batch_size, 14, 2000]
        self.input_projection = nn.Linear(14, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: [batch_size, 14, 2000]
        x = x.permute(2, 0, 1)  # Shape: [2000, batch_size, 14]

        # Project input to d_model dimensions
        x = self.input_projection(x)  # Shape: [2000, batch_size, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)

        # Global average pooling
        x = x.mean(dim=0)  # Shape: [batch_size, d_model]

        # Classification
        output = self.classifier(x)  # Shape: [batch_size, num_classes]

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