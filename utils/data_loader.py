# utils/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torchaudio

class SpectrogramDataset(Dataset):
    def __init__(self, metadata_file, transform=None):
        """
        Args:
            metadata_file (string): Path to the metadata CSV file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        spectrogram_path = self.metadata.iloc[idx]['spectrogram_path']
        label = self.metadata.iloc[idx]['emotion_idx']
        
        # Load the precomputed spectrogram
        spectrogram = torch.load(spectrogram_path)
        
        # Add a channel dimension if needed (for CNN)
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)  # (1, n_mels, time)
        
        if self.transform:
            spectrogram = self.transform(spectrogram)
            
        return spectrogram, label

class AudioDataset(Dataset):
    def __init__(self, metadata_file, target_sample_rate=16000, target_duration=3.0, transform=None):
        """
        Args:
            metadata_file (string): Path to the metadata CSV file.
            target_sample_rate (int): Target sample rate for the audio.
            target_duration (float): Target duration in seconds. Audio will be padded or trimmed to this length.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata = pd.read_csv(metadata_file)
        self.target_sample_rate = target_sample_rate
        self.target_duration = target_duration
        self.transform = transform
        self.target_length = int(target_duration * target_sample_rate)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        audio_path = self.metadata.iloc[idx]['file_path']
        label = self.metadata.iloc[idx]['emotion_idx']
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)
        
        # Ensure consistent duration (pad or trim)
        if waveform.shape[1] < self.target_length:
            # Pad with zeros
            pad_length = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        elif waveform.shape[1] > self.target_length:
            # Trim to target length
            waveform = waveform[:, :self.target_length]
            
        if self.transform:
            waveform = self.transform(waveform)
            
        return waveform, label

def get_dataloader(metadata_file, dataset_type='spectrogram', batch_size=32, shuffle=True, transform=None):
    if dataset_type == 'spectrogram':
        dataset = SpectrogramDataset(metadata_file, transform=transform)
    elif dataset_type == 'audio':
        dataset = AudioDataset(metadata_file, transform=transform)
    else:
        raise ValueError("dataset_type must be 'spectrogram' or 'audio'")
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader