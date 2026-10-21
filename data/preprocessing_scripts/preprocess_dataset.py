import os
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
import argparse

def preprocess_set(metadata_csv, audio_dir, target_sr=16000, duration=10, num_mel_bins=128):
    if not os.path.exists(metadata_csv):
        print(f"Metadata file {metadata_csv} not found. Skipping...")
        return
        
    df = pd.read_csv(metadata_csv)
    df = df[df['status'] == 'Success'].reset_index(drop=True)
    num_samples = target_sr * duration
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=num_mel_bins,
        f_min=0,
        f_max=8000
    )
    
    print(f"Preprocessing files in {audio_dir}...")
    for idx in tqdm(range(len(df))):
        row = df.iloc[idx]
        filename = row['filename']
        group = row['primary_group']
        
        audio_path = os.path.join(audio_dir, group, filename)
        base_name, _ = os.path.splitext(filename)
        pt_path = os.path.join(audio_dir, group, base_name + ".pt")
        
        if os.path.exists(pt_path):
            continue
            
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)
                
            # Mono conversion
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Pad or Trim
            if waveform.shape[1] < num_samples:
                pad_len = num_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            elif waveform.shape[1] > num_samples:
                waveform = waveform[:, :num_samples]
                
        except Exception:
            waveform = torch.zeros((1, num_samples))
            
        # Log-Mel conversion
        mel_spec = mel_transform(waveform)
        log_mel = torch.log(mel_spec + 1e-6)
        
        # Instance Normalization (mean=0, std=0.5)
        mean = log_mel.mean()
        std = log_mel.std()
        log_mel = 0.5 * (log_mel - mean) / (std + 1e-6)
        
        # Save as PyTorch tensor
        torch.save(log_mel, pt_path)

def main():
    parser = argparse.ArgumentParser(description="Preprocess AudioSet WAV files to Log-Mel Spectrogram PyTorch Tensors")
    parser.add_argument("--duration", type=int, default=10, help="Standard audio duration in seconds (default: 10)")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_files_dir = os.path.abspath(os.path.join(base_dir, "..", "dataset_files"))
    preprocess_set(os.path.join(dataset_files_dir, "train", "metadata.csv"), os.path.join(dataset_files_dir, "train"), duration=args.duration)
    preprocess_set(os.path.join(dataset_files_dir, "test", "metadata.csv"), os.path.join(dataset_files_dir, "test"), duration=args.duration)
    print("Preprocessing finished successfully!")

if __name__ == "__main__":
    main()
