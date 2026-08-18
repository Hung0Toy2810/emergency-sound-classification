import torch
import torch.nn as nn
from train_custom_cnn import create_data_loaders, evaluate, get_device, DATA_PATH
from adaptive_controller import GrowableCNN

def main():
    device = get_device()
    checkpoint_path = "models/custom_cnn_best.pth"
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        print(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")
        return
        
    channels = checkpoint['channels']
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    
    model = GrowableCNN(num_classes, channels=channels).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    criterion = nn.CrossEntropyLoss()
    
    # Đánh giá trên data/mels-urbansound8k-split/
    print("\n=== Đánh giá trên data/mels-urbansound8k-split/ ===")
    try:
        train_loader, val_loader, test_loader, _, _ = create_data_loaders("data/mels-urbansound8k-split/", batch_size=16)
        
        train_loss, train_acc = evaluate(model, train_loader, criterion, device, desc="  Train Split")
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, desc="  Val Split")
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc="  Test Split")
        
        print(f"data/mels-urbansound8k-split/ -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"data/mels-urbansound8k-split/ -> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"data/mels-urbansound8k-split/ -> Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    except Exception as e:
        print(f"Lỗi khi đánh giá trên mels-urbansound8k-split: {e}")

if __name__ == '__main__':
    main()
