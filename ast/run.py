import os
import argparse
import torch
from src.dataset import AudioDataset
from src.model import ViTAST
from src.train import run_training

def main():
    parser = argparse.ArgumentParser(description="Train Audio Spectrogram Transformer (AST) with ViT")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16 since AMP FP16 is enabled)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--vit_model", type=str, default="google/vit-base-patch16-224", help="Pretrained ViT model from Hugging Face")
    parser.add_argument("--dry_run", action="store_true", help="Perform a dry-run sanity check on dataset and model without training")
    parser.add_argument("--duration", type=int, default=10, help="Audio duration in seconds (default: 10)")
    parser.add_argument("--max_batches", type=int, default=None, help="Maximum number of batches per epoch (for quick testing)")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess all audio files to .pt tensors before training")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint if available")
    
    args = parser.parse_args()
    
    # Define paths dynamically relative to run.py location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(base_dir, "..", "ast_data"))
    
    train_metadata = os.path.join(data_dir, "dataset_files", "train", "metadata.csv")
    train_audio_dir = os.path.join(data_dir, "dataset_files", "train")
    test_metadata = os.path.join(data_dir, "dataset_files", "test", "metadata.csv")
    test_audio_dir = os.path.join(data_dir, "dataset_files", "test")
    
    # 0. Offline Preprocessing if requested
    if args.preprocess:
        print("Running offline preprocessing on train and test datasets...")
        import sys
        sys.path.append(data_dir)
        from preprocessing_scripts.preprocess_dataset import preprocess_set
        preprocess_set(train_metadata, train_audio_dir, duration=args.duration)
        preprocess_set(test_metadata, test_audio_dir, duration=args.duration)
        print("Offline preprocessing completed!\n")
    
    # 1. Load datasets
    print("Initializing datasets...")
    if not os.path.exists(train_metadata) or not os.path.exists(test_metadata):
        print("Error: Train/test metadata files not found! Please run split_metadata.py and reorganize_dataset.py first.")
        return
        
    train_dataset = AudioDataset(
        metadata_csv_path=train_metadata,
        audio_dir=train_audio_dir,
        is_train=True,
        duration=args.duration
    )
    
    val_dataset = AudioDataset(
        metadata_csv_path=test_metadata,
        audio_dir=test_audio_dir,
        is_train=False,
        duration=args.duration
    )
    
    print(f"Loaded {len(train_dataset)} training samples.")
    print(f"Loaded {len(val_dataset)} validation samples.")
    
    # 2. Check shapes and perform dry-run
    print("\nPerforming sanity checks...")
    sample_spec, sample_label = train_dataset[0]
    print(f"Sample Spectrogram Shape: {sample_spec.shape} (Expected: [1, 128, time_steps])")
    print(f"Sample Label: {sample_label} (Shape: {sample_label.shape})")
    
    # 3. Initialize Model
    # Stride=10, kernel=16 -> (128-16)/10 + 1 = 12 frequency patches, (time_steps-16)/10 + 1 time patches.
    num_freq_patches = 12
    num_time_patches = (sample_spec.shape[-1] - 16) // 10 + 1
    print(f"Dynamic patches calculated: frequency={num_freq_patches}, time={num_time_patches}")
    
    print(f"\nInitializing model '{args.vit_model}' adapted to AST...")
    model = ViTAST(
        model_name=args.vit_model,
        num_classes=4,
        num_freq_patches=num_freq_patches,
        num_time_patches=num_time_patches
    )
    print("Model initialized successfully!")
    
    # Model dry run
    dummy_input = torch.randn(2, 1, 128, sample_spec.shape[-1])
    with torch.no_grad():
        dummy_output = model(dummy_input)
    print(f"Model Dry-Run Output Shape: {dummy_output.shape} (Expected: [Batch, 4])")
    assert dummy_output.shape == (2, 4), "Model output shape mismatch!"
    print("Sanity checks passed!")
    
    if args.dry_run:
        print("\nDry-run flag set. Exiting without training.")
        return
        
    # 4. Start Training
    print("\nStarting training pipeline...")
    run_training(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_batches=args.max_batches,
        resume=args.resume
    )


if __name__ == "__main__":
    main()

