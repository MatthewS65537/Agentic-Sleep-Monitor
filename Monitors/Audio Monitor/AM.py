import time
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from AudioClassificationTransformer import *

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

if __name__ == "__main__":
    model = AudioTransformer.load_from_checkpoint("../../model_checkpoints/AudioClassificationModel.ckpt")
    model.eval()

    start_time = time.time()
    MFCC = load_wav_into_tensor("../../data/recording0.wav")
    MFCC = MFCC.reshape([1, 32, 937]).to(dtype=torch.float32)
    logits = model(MFCC)
    label_idx = torch.argmax(logits)
    end_time = time.time()
    
    labels = ["Normal Sleep", "Hypopnea", "Snore", "ObstructiveApnea", "MixedApnea"]

    print(labels[label_idx])
    print(end_time - start_time, "s")