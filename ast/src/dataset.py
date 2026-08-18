import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, metadata_csv_path, audio_dir, is_train=False, target_sr=16000, duration=10, num_mel_bins=128):
        """
        Custom PyTorch Dataset for loading log-Mel spectrograms from AudioSet clips.
        
        Args:
            metadata_csv_path (str): Path to the train/test metadata.csv
            audio_dir (str): Base directory containing audio files (downloads/train or downloads/test)
            is_train (bool): If True, apply SpecAugment data augmentation
            target_sr (int): Sampling rate to resample audio to
            duration (int): Duration of clip in seconds (default 10s)
            num_mel_bins (int): Number of Mel frequency bins
        """
        self.audio_dir = audio_dir
        self.is_train = is_train
        self.target_sr = target_sr
        self.duration = duration
        self.num_samples = target_sr * duration
        self.num_mel_bins = num_mel_bins
        
        # Load metadata
        self.df = pd.read_csv(metadata_csv_path)
        # Filter for successful rows only
        self.df = self.df[self.df['status'] == 'Success'].reset_index(drop=True)
        
        self.classes = ['Crash_Collision', 'Siren', 'Scream_Shout', 'Gunshot_Explosion']
        
        # Log-Mel Spectrogram transform
        # Window length 25ms (400 samples @16kHz), Hop length 10ms (160 samples @16kHz)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=self.num_mel_bins,
            f_min=0,
            f_max=8000
        )
        
        # SpecAugment data augmentation
        if self.is_train:
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=24)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=96)
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        group = row['primary_group']
        
        # Check if preprocessed .pt file exists
        base_name, _ = os.path.splitext(filename)
        pt_path = os.path.join(self.audio_dir, group, base_name + ".pt")
        
        if os.path.exists(pt_path):
            try:
                log_mel = torch.load(pt_path)
                # Slice time steps if a shorter duration is requested
                expected_time_steps = (self.target_sr * self.duration) // 160 + 1
                if log_mel.shape[2] > expected_time_steps:
                    log_mel = log_mel[:, :, :expected_time_steps]
                elif log_mel.shape[2] < expected_time_steps:
                    # Pad if for some reason it's shorter
                    pad_len = expected_time_steps - log_mel.shape[2]
                    log_mel = torch.nn.functional.pad(log_mel, (0, pad_len))
            except Exception:
                # Fallback to normal loading if .pt is corrupted
                log_mel = self._process_raw_wav(os.path.join(self.audio_dir, group, filename))
        else:
            log_mel = self._process_raw_wav(os.path.join(self.audio_dir, group, filename))
            
        # Apply SpecAugment for training
        if self.is_train:
            log_mel = self.freq_mask(log_mel)
            log_mel = self.time_mask(log_mel)
            
        # Multi-label binary labels: [Crash_Collision, Siren, Scream_Shout, Gunshot_Explosion]
        labels = torch.tensor([
            float(row['Crash_Collision']),
            float(row['Siren']),
            float(row['Scream_Shout']),
            float(row['Gunshot_Explosion'])
        ], dtype=torch.float32)
        
        return log_mel, labels

    def _process_raw_wav(self, audio_path):
        num_samples = self.target_sr * self.duration
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample to 16kHz if necessary
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)
                
            # Convert stereo/multichannel to mono by averaging channels
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Pad or trim waveform to standard length
            if waveform.shape[1] < num_samples:
                pad_len = num_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            elif waveform.shape[1] > num_samples:
                waveform = waveform[:, :num_samples]
                
        except Exception:
            # Fallback to zeros in case of read errors
            waveform = torch.zeros((1, num_samples))
            
        # Convert to Mel Spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to Log-scale
        log_mel = torch.log(mel_spec + 1e-6)
        
        # Instance Normalization (ensuring mean=0, std=0.5)
        mean = log_mel.mean()
        std = log_mel.std()
        log_mel = 0.5 * (log_mel - mean) / (std + 1e-6)
        
        return log_mel

