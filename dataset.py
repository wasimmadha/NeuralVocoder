from torch.utils.data import Dataset
import librosa
import torch
import numpy as np

class SpecVocoderDataset(Dataset):
    def __init__(self, training_files) -> None:
        super().__init__()
        self.training_files = training_files
        self.sample_rate = 22050 # static value
        self.n_fft = 1025 # static value
        self.hop_length = 275 # static value
        self.n_mels = 80 # static value
    
    def __len__(self):
        return len(self.training_files)
    
    def __getitem__(self, idx):
        waveform, sr = librosa.load(self.training_files[idx], sr=self.sample_rate)
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, 
            sr=self.sample_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db_tensor = torch.tensor(mel_spec_db)
        waveform_tensor = torch.tensor(waveform)
        return mel_spec_db_tensor, waveform_tensor