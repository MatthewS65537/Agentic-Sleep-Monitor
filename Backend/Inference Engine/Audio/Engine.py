import math
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torchmetrics
import os
import platform
import torch
import numpy as np

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
    
def resize_spectrogram(spectrogram, target_length):
    spectrogram_np = spectrogram.squeeze().numpy()  # Convert to numpy array
    current_length = spectrogram_np.shape[-1]
    resized = np.zeros((spectrogram_np.shape[0], target_length))
    for i in range(spectrogram_np.shape[0]):
        resized[i, :] = np.interp(
            np.linspace(0, current_length, target_length),
            np.arange(current_length),
            spectrogram_np[i, :]
        )
    return torch.tensor(resized)  # Convert back to tensor

def load_wav_into_tensor(wav_file):
    waveform, sample_rate = torchaudio.load(wav_file)
    waveform = waveform.repeat(1, 10)

    # Compute the mel-spectrogram
    target_duration = 10
    sample_rate = 48000
    hop_length = 512
    mel_spectrogram = T.MelSpectrogram(sample_rate=sample_rate, n_mels=32, hop_length=hop_length)(waveform)
    mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
    mel_spectrogram_db = torch.mean(mel_spectrogram_db, dim = 0)


    # Calculate the number of frames for 10 seconds
    num_frames_10s = int(10 * sample_rate / hop_length)

    # Resize the mel-spectrogram
    resized_mel_spectrogram_db = resize_spectrogram(mel_spectrogram_db, num_frames_10s)
    return resized_mel_spectrogram_db


if "macOS" in platform.platform() and "arm64" in platform.platform():
    import coremltools as ct

def get_inference_platform():
    platform_str = platform.platform()
    if "Windows" in platform_str or "Linux" in platform_str:
        return "Lightning"
    elif "macOS" in platform_str:
        if "arm64" in platform_str:
            # Apple Silicon Mac, has NPU
            return "ANE"
        else:
            return "Lightning"
    else:
        return "Lightning"
    
class AudioInferenceEngine():
    def __init__(self, platform, model_name, checkpoint_dir = "../checkpoints"):
        self.platform = platform
        path_no_ext = os.path.join(checkpoint_dir, model_name)
        if self.platform == "Lightning":
            # Pytorch Lightning Inference
            self.model = AudioTransformer.load_from_checkpoint(path_no_ext + ".ckpt")
            self.model.eval()
        elif self.platform == "ANE":
            # CoreML Inference
            self.model = ct.models.MLModel(path_no_ext + ".mlpackage")
        else:
            if self.platform in ["cpu", "cuda", "mps"]:
                # Pytorch Inference
                self.model = AudioClassificationTransformer()
                # Surpress Argument with weights_only=True
                self.model.load_state_dict(torch.load(path_no_ext + ".pt", weights_only=True))
                self.model.to(self.platform, dtype=torch.float32)
                self.model.eval()
            else:
                assert(False)
    
    def predict(self, data):
        # data -> numpy array of shape [1, 32, 937]
        if self.platform == "Lightning":
            data = torch.tensor(data).to(dtype=torch.float32)
            logits = self.model(data)
            # print(f"[DEBUG] {logits}")
            label_idx = torch.argmax(logits)
        elif self.platform == "ANE":
            data = data.detach().numpy().astype(np.float32)
            logits = self.model.predict({'x_1' : data})["linear_25"][0]
            # print(f"[DEBUG] {logits}")
            label_idx = np.argmax(logits)
        else:
            if self.platform in ["cpu", "cuda", "mps"]: 
                data = torch.tensor(data)
                logits = self.model(data.to(self.platform, dtype=torch.float32))
                # print(f"[DEBUG] {logits}")
                label_idx = torch.argmax(logits)
            else:
                assert(False)
        return label_idx.item()

class AudioEngine:
    def __init__(self, platform, model_name, checkpoint_dir="./model_checkpoints"):
        self.inference_engine = AudioInferenceEngine(platform, model_name, checkpoint_dir)
    
    def predict(self, wav_file_path, return_label=True):
        """
        Method that takes a WAV file path and returns a prediction using the Audio Inference Engine.
        
        Args:
        wav_file_path (str): Path to the WAV file
        
        Returns:
        int: Predicted label index
        """
        # Load and preprocess the WAV file
        mfcc = load_wav_into_tensor(wav_file_path)
        
        # Reshape the MFCC to match the expected input shape [1, 32, 937]
        mfcc = mfcc.unsqueeze(0)  # Add batch dimension
        
        # Make prediction using the inference engine
        prediction = self.inference_engine.predict(mfcc)
        
        return prediction if not return_label else self.get_label(prediction)

    def get_label(self, prediction):
        """
        Convert the numerical prediction to a human-readable label.
        
        Args:
        prediction (int): Predicted label index
        
        Returns:
        str: Human-readable label
        """
        labels = ["NormalSleep", "Hypopnea", "Snore", "ObstructiveApnea", "MixedApnea"]
        return labels[prediction]

