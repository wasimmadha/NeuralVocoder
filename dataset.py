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
    
def collate_fn(batch):
    mel_specs, waveforms = zip(*batch)  # Unzip the batch

    # Find max length for padding
    max_mel_length = max(mel_spec.size(1) for mel_spec in mel_specs)
    max_waveform_length = max(waveform.size(0) for waveform in waveforms)

    # Pad mel-spectrograms
    padded_mel_specs = []
    for mel_spec in mel_specs:
        pad_length = max_mel_length - mel_spec.size(1)
        padded_mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_length), 'constant', float('-inf'))  # Padding with -inf for log scale
        padded_mel_specs.append(padded_mel_spec)

    # Pad waveforms
    padded_waveforms = []
    for waveform in waveforms:
        pad_length = max_waveform_length - waveform.size(0)
        padded_waveform = torch.nn.functional.pad(waveform, (0, pad_length), 'constant', 0)  # Padding with 0
        padded_waveforms.append(padded_waveform)

    return torch.stack(padded_mel_specs), torch.stack(padded_waveforms)
