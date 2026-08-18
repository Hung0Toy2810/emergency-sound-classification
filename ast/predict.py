import os
import argparse
import torch
import torchaudio
import numpy as np
from src.model import ViTAST

CLASSES = ['Crash_Collision', 'Siren', 'Scream_Shout', 'Gunshot_Explosion']

def preprocess_audio(audio_path, target_sr=16000, duration=10, num_mel_bins=128):
    """
    Loads, resamples, pads/trims, and extracts log-Mel spectrogram features 
    from a single audio file, matching training preprocessing.
    """
    num_samples = target_sr * duration
    waveform, sr = torchaudio.load(audio_path)
    
    # 1. Resample to 16kHz
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        
    # 2. Convert multi-channel to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # 3. Pad/Trim to exact duration
    if waveform.shape[1] < num_samples:
        pad_len = num_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))
    elif waveform.shape[1] > num_samples:
        waveform = waveform[:, :num_samples]
        
    # 4. Extract Mel Spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=num_mel_bins,
        f_min=0,
        f_max=8000
    )
    mel_spec = mel_transform(waveform)
    log_mel = torch.log(mel_spec + 1e-6)
    
    # 5. Instance Normalization (z-score per spectrogram)
    mean = log_mel.mean()
    std = log_mel.std()
    log_mel = 0.5 * (log_mel - mean) / (std + 1e-6)
    
    # Add batch and channel dimensions: [1, 1, num_mel_bins, time_steps]
    return log_mel.unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description="Run AST inference on a single audio file")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (.wav)")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_checkpoint = os.path.join(base_dir, "model_checkpoints", "best_model.pth")
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint, help="Path to model checkpoint")
    parser.add_argument("--vit_model", type=str, default="google/vit-base-patch16-224", help="Pretrained ViT model name")
    parser.add_argument("--duration", type=int, default=10, help="Audio duration in seconds")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"Error: Audio file '{args.audio}' not found.")
        return
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found. Please train the model first.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Preprocess
    print(f"Preprocessing audio file '{args.audio}'...")
    input_tensor = preprocess_audio(args.audio, duration=args.duration).to(device)
    time_steps = input_tensor.shape[-1]
    
    # 2. Initialize Model with dynamic time patches
    num_freq_patches = 12
    num_time_patches = (time_steps - 16) // 10 + 1
    model = ViTAST(
        model_name=args.vit_model,
        num_classes=4,
        num_freq_patches=num_freq_patches,
        num_time_patches=num_time_patches
    )
    
    # 3. Load Checkpoint
    print(f"Loading weights from '{args.checkpoint}'...")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 4. Inference
    print("Running inference...")
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
    print("\n" + "="*35)
    print("       AST PREDICTION RESULTS       ")
    print("="*35)
    for cls_name, prob in zip(CLASSES, probs):
        bar_len = int(prob * 20)
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"{cls_name:<18}: {prob*100:>6.2f}%  [{bar}]")
    print("="*35)

if __name__ == "__main__":
    main()
