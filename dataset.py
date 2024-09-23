from torch.utils.data import Dataset
import torch
import torchaudio
import os

class SpecVocoderDataset(Dataset):
    def __init__(self, training_files) -> None:
        super().__init__()
        self.training_files = training_files
        self.sample_rate = 22050  # static value
        self.n_fft = 1024  # static value
        self.hop_length = 275  # static value
        self.n_mels = 80  # static value
    
    def __len__(self):
        return len(self.training_files)
    
    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.training_files[idx])

        if sr != self.sample_rate:
            resample_transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resample_transform(waveform)
        
        mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )
        mel_spec = mel_spec_transform(waveform)

        mel_spec_db_transform = torchaudio.transforms.AmplitudeToDB()
        mel_spec_db = mel_spec_db_transform(mel_spec)

        return mel_spec_db, waveform
