"""
train_custom_cnn.py - Custom CNN from Scratch for UrbanSound8K Classification
============================================================================
Mục tiêu: Implement CNN thuần túy KHÔNG dùng pretrained, lộ rõ kiến trúc
từng layer để giảng viên và sinh viên hiểu được tầng thấp nhất.

Kiến trúc: VGG-style CNN (5 Conv Blocks + Classifier)
References:
  [1] LeCun et al. (1998) - Gradient-Based Learning Applied to Document Recognition
  [2] Simonyan & Zisserman (2014) - Very Deep Convolutional Networks (VGGNet)
  [3] Ioffe & Szegedy (2015) - Batch Normalization
  [4] Srivastava et al. (2014) - Dropout: A Simple Way to Prevent Overfitting
  [5] Lin et al. (2013) - Network In Network (Global Average Pooling)

Usage:
    python3 train_custom_cnn.py
    python3 train_custom_cnn.py --data data/mels-urbansound8k-split/ --epochs 30

====================== TOÁN HỌC CỐT LÕI ======================

1. CONVOLUTION 2D (Tích chập 2 chiều):
   Công thức:
     y[i,j] = Σ_m Σ_n x[i+m, j+n] * w[m, n] + b
   Trong đó:
     x = input feature map, w = kernel (bộ lọc), b = bias
     Kích thước output: H_out = (H_in + 2*padding - kernel_size) / stride + 1

2. BATCH NORMALIZATION (Chuẩn hóa theo batch):
   Công thức:
     x̂ = (x - μ_B) / √(σ²_B + ε)     # Chuẩn hóa
     y = γ * x̂ + β                     # Scale và Shift
   Trong đó:
     μ_B = mean của batch, σ²_B = variance của batch
     γ, β = tham số học được (learnable parameters)
     ε = hằng số nhỏ tránh chia cho 0 (thường = 1e-5)

3. ReLU (Rectified Linear Unit):
   Công thức:
     f(x) = max(0, x)
   Đạo hàm:
     f'(x) = 1 nếu x > 0, 0 nếu x ≤ 0

4. MAX POOLING 2x2:
   Công thức:
     y[i,j] = max(x[2i:2i+2, 2j:2j+2])
   Tác dụng: Giảm kích thước spatial xuống 1/2, giữ lại đặc trưng mạnh nhất.

5. GLOBAL AVERAGE POOLING:
   Công thức:
     y_c = (1/H*W) * Σ_i Σ_j x_c[i,j]
   Tác dụng: Nén toàn bộ feature map HxW thành 1 số duy nhất cho mỗi channel.

6. CROSS-ENTROPY LOSS:
   Công thức:
     L = -Σ_c y_c * log(ŷ_c)
   Trong đó:
     y_c = one-hot label, ŷ_c = softmax probability

7. SOFTMAX:
   Công thức:
     P(class=c) = exp(z_c) / Σ_k exp(z_k)
========================================================================
"""

import os
import ssl
import copy
import time
import argparse
import gc
import numpy as np

# Fix SSL on macOS
if not os.environ.get('PYTHONHTTPSVERIFY', '') and hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from adaptive_controller import (LagrangeCapacityController, GrowableCNN,
                                 grow_model, setup_cosine_lr,
                                 compute_gradient_pressures)

# ============================================================================
# Cấu hình
# ============================================================================
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 2e-3
MIN_LR = 1e-5
LR_PATIENCE = 5
LR_FACTOR = 0.2
DROPOUT = 0.5

# Path to the spectrogram images produced by preprocess.py
DATA_PATH = "data/mels-urbansound8k-split/"

# Where to save model checkpoints and plots
SAVE_DIR = "models/"
CHECKPOINT_NAME = "custom_cnn_best.pth"

# Train / Validation / Test split ratios
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def get_device():
    """Select the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    return device


# ============================================================================
# KIẾN TRÚC CNN TỰ XÂY DỰNG TỪ ĐẦU (giữ lại cho reference/print)
# ============================================================================

class ConvBlock(nn.Module):
    """Khối tích chập: Conv→BN→ReLU→Conv→BN→ReLU→MaxPool"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(x)
        return x


class UrbanSoundCNN(nn.Module):
    """Kiến trúc CNN 5 blocks (giữ lại cho print_model_summary)."""
    def __init__(self, num_classes, dropout_conv=0.25, dropout_fc=0.5):
        super().__init__()
        self.block1 = ConvBlock(3, 32, dropout=0.0)
        self.block2 = ConvBlock(32, 64, dropout=0.0)
        self.block3 = ConvBlock(64, 128, dropout=dropout_conv)
        self.block4 = ConvBlock(128, 256, dropout=dropout_conv)
        self.block5 = ConvBlock(256, 512, dropout=dropout_conv)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_fc),
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Dropout(p=0.3), nn.Linear(128, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        x = self.block4(x); x = self.block5(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.classifier(x)


# ============================================================================
# Data Loading
# ============================================================================

def create_data_loaders(data_path, batch_size=BATCH_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # Removed RandomHorizontalFlip (reverses time) and heavy RandomAffine 
        # to preserve temporal and frequency structure of spectrograms.
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.0), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        # SpecAugment proxy: Randomly zero out blocks in frequency/time
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.1, 3.3)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")
    test_dir = os.path.join(data_path, "test")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
        test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform) if os.path.isdir(test_dir) else None
    else:
        full_dataset = datasets.ImageFolder(data_path, transform=train_transform)
        targets = full_dataset.targets
        indices = list(range(len(full_dataset)))
        train_val_idx, test_idx = train_test_split(indices, test_size=TEST_RATIO, stratify=targets, random_state=42)
        train_val_targets = [targets[i] for i in train_val_idx]
        rel_val = VAL_RATIO / (1 - TEST_RATIO)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=rel_val, stratify=train_val_targets, random_state=42)
        train_dataset = Subset(full_dataset, train_idx)
        eval_full = datasets.ImageFolder(data_path, transform=eval_transform)
        val_dataset = Subset(eval_full, val_idx)
        test_dataset = Subset(eval_full, test_idx)
        print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # FIX RAM: num_workers=0 trên macOS/MPS để tránh worker processes tích lũy
    # RSS (Resident Set Size) qua nhiều epochs. Mỗi worker fork Python process
    # riêng (~300-500MB mỗi cái) và KHÔNG được giải phóng giữa các epoch.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if test_dataset else None

    classes = train_dataset.classes if hasattr(train_dataset, 'classes') else train_dataset.dataset.classes
    if hasattr(train_dataset, 'targets'):
        train_targets = train_dataset.targets
    elif hasattr(train_dataset, 'indices'):
        train_targets = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
    else:
        train_targets = []

    return train_loader, val_loader, test_loader, classes, train_targets


# ============================================================================
# Mixup functions
# ============================================================================

def mixup_data(x, y, alpha=0.2, device='cpu'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================================
# Training & Evaluation
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device,
                    pressure_every=50, mps_flush_every=200):
    """Train một epoch.

    Args:
        pressure_every:  Chỉ đo gradient pressure mỗi N batch (không phải mỗi batch)
                         -> giảm 50x số MPS temp tensors được tạo ra.
        mps_flush_every: Gọi torch.mps.empty_cache() mỗi N batch trong epoch
                         -> giữ MPS allocator Free List nhỏ, không phình to.
    """
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    # Khởi tạo mảng cộng dồn pressure
    epoch_pressures = [0.0] * 5
    num_batches = 0
    pressure_samples = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply Mixup
        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.2, device=device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()

        # FIX RAM: Chỉ đo pressure mỗi `pressure_every` batches thay vì mỗi batch.
        # compute_gradient_pressures tạo 50 MPS temp tensors mỗi lần gọi.
        # 2456 batches/epoch -> giảm từ 122,800 xuống còn ~2,456 MPS alloc/epoch.
        if num_batches % pressure_every == 0:
            batch_pressures = compute_gradient_pressures(model)
            for i in range(5):
                epoch_pressures[i] += batch_pressures[i]
            pressure_samples += 1

        optimizer.step()

        # FIX RAM: Trích xuất scalar ngay, rồi del tensor MPS tường minh.
        # Không để outputs/loss sống đến lần lặp tiếp theo (tránh 2-batch peak).
        loss_val = loss.item()                                   # scalar CPU
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            # Mixup-aware accuracy: tính đúng theo tỷ lệ trộn cả 2 nhãn
            n_correct = (lam * (preds == targets_a).sum().item()
                         + (1 - lam) * (preds == targets_b).sum().item())
        n_total = labels.size(0)                                 # scalar
        del outputs, loss, inputs, labels, targets_a, targets_b  # giải phóng MPS buffer

        loss_sum += loss_val * n_total
        correct += n_correct
        total += n_total
        num_batches += 1
        pbar.set_postfix(loss=f"{loss_sum/total:.4f}", acc=f"{correct/total:.4f}")

        # FIX RAM: Flush MPS allocator cache định kỳ trong epoch.
        # Tránh MPS Free List phình to suốt 60 giây của một epoch.
        if num_batches % mps_flush_every == 0 and device.type == 'mps':
            torch.mps.empty_cache()

    # Trung bình pressure trên toàn epoch
    if pressure_samples == 0:
        pressure_samples = 1
    avg_pressures = [p / pressure_samples for p in epoch_pressures]
    total_p = sum(avg_pressures) + 1e-10
    avg_pressures = [p / total_p for p in avg_pressures]

    return loss_sum / total, correct / total, avg_pressures


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="  Val"):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        loss_sum += loss.item() * inputs.size(0)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)
    return loss_sum / total, correct / total


# ============================================================================
# Main — Gradient-Informed Targeted Growth
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Custom CNN + Gradient-Targeted Growth")
    parser.add_argument("--data", type=str, default=DATA_PATH)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    device = get_device()
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Data
    print(f"\n--- Loading data from: {args.data} ---")
    train_loader, val_loader, test_loader, class_names, train_targets = create_data_loaders(args.data, args.batch_size)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Controller — per-stage widths, gradient-informed
    # PES (Parameter Efficiency Score) = ΔAcc / (ΔParams/Params_cũ)
    # Phạt nặng khi params tăng nhiều mà acc tăng không tương xứng
    controller = LagrangeCapacityController(
        min_width=0.50, delta_width=0.05, max_width=1.0,
        epsilon=0.15, alpha=0.15, lambda_threshold=1.5, plateau_patience=5,
        grow_cooldown=8,
        min_pes=0.02,   # PES < 0.02 → phạt nặng ngay (block)
        good_pes=0.10,  # PES ≥ 0.10 → grow hiệu quả, không phạt
    )

    # Model — start at 25% channels per stage
    channels = controller.get_channels()
    model = GrowableCNN(num_classes, channels=channels).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"\nInitial: channels={channels} | params={total_p:,}")
    print(f"Stage widths: {[f'{w:.0%}' for w in controller.stage_widths]}")

    # Class weights
    weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=np.array(train_targets))
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    # Train
    print(f"\n--- Training with Gradient-Informed Targeted Growth ---")
    best_val_loss = float("inf")
    start_epoch = 0
    checkpoint_path = os.path.join(SAVE_DIR, CHECKPOINT_NAME)

    if args.resume and os.path.exists(checkpoint_path):
        print(f"  [Resume] Found checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        initial_channels = controller.get_channels()
        # Phục hồi cấu hình (vì model có thể đã grow)
        controller.stage_widths = checkpoint['stage_widths']
        channels = checkpoint['channels']
        
        # Tạo lại model và optimizer với cấu hình mới
        model = GrowableCNN(num_classes, channels=channels).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # Khởi tạo Optimizer và Scheduler đúng với trạng thái (đã Grow hay chưa)
        if channels != initial_channels:
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.7, weight_decay=1e-4)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
        
        best_val_loss = checkpoint['val_loss']
        start_epoch = checkpoint['epoch']
            
        print(f"  [Resume] Resuming from epoch {start_epoch} | channels {channels} | val_loss {best_val_loss:.4f}")
        print(f"  [Resume] Khôi phục Learning Rate\n")

    best_wts = copy.deepcopy(model.state_dict())
    best_channels = list(channels)
    tl_hist, vl_hist, ta_hist, va_hist, param_hist = [], [], [], [], []
    
    # Phục hồi lịch sử vẽ biểu đồ nếu có file recovered
    import json
    if args.resume and os.path.exists("recovered_logs_full.json"):
        with open("recovered_logs_full.json", "r") as f:
            h = json.load(f)
            # Cắt mảng lịch sử đúng bằng start_epoch để không bị trùng lặp
            tl_hist = h['tl'][:start_epoch]
            vl_hist = h['vl'][:start_epoch]
            ta_hist = h['ta'][:start_epoch]
            va_hist = h['va'][:start_epoch]
            param_hist = h['param'][:start_epoch]
            print(f"  [Resume] Đã nạp lại biểu đồ chính xác {len(tl_hist)} epochs từ lịch sử chat!")

    for epoch in range(start_epoch, args.epochs):
        try:
            total_p = sum(p.numel() for p in model.parameters())
            print(f"\nEpoch {epoch+1}/{args.epochs} [params={total_p:,}]")
            t0 = time.time()

            tl, ta, pressures = train_one_epoch(model, train_loader, criterion, optimizer, device)

            controller.update_pressures(pressures)
            stage_names = ["Block1", "Block2", "Block3", "Block4", "Block5"]
            pressure_str = " | ".join(f"{n}:{p:.2%}" for n, p in zip(stage_names, pressures))
            print(f"  Gradient Pressure: {pressure_str}")

            vl, va = evaluate(model, val_loader, criterion, device)
            scheduler.step(vl)

            lr = optimizer.param_groups[0]["lr"]
            print(f"  Train Loss: {tl:.4f} | Acc: {ta:.4f}")
            print(f"  Val   Loss: {vl:.4f} | Acc: {va:.4f}")
            print(f"  LR: {lr:.6f} | Time: {time.time()-t0:.1f}s")

            should_grow, info = controller.step(tl, vl, epoch, val_acc=va)
            print(f"  λ={info['lambda']:.4f} | gap={info['gap']:.4f} | converged={info['converged']}")
            pes_str = f"{info['last_pes']:.4f}" if info['last_pes'] is not None else "N/A"
            print(f"  est_params={info['est_params']:,} | cooldown={info['cooldown_remaining']}"
                  f" | bad_grows={info['bad_grows']} | last_PES={pes_str}")
            print(f"  Stage widths: {[f'{w:.0%}' for w in controller.stage_widths]}")

            tl_hist.append(tl); vl_hist.append(vl)
            ta_hist.append(ta); va_hist.append(va)
            param_hist.append(total_p)

            if vl < best_val_loss:
                best_val_loss = vl
                best_wts = copy.deepcopy(model.state_dict())
                best_channels = list(model.channels)
                torch.save({'epoch': epoch+1, 'model_state_dict': best_wts,
                            'val_loss': best_val_loss, 'val_acc': va,
                            'class_names': class_names, 'channels': best_channels,
                            'stage_widths': list(controller.stage_widths)},
                           os.path.join(SAVE_DIR, CHECKPOINT_NAME))
                print(f"  ★ Best model saved! (val_loss: {best_val_loss:.4f})")

            if should_grow:
                old_channels = list(model.channels)
                target = controller.grow_targeted()
                new_channels = controller.get_channels()
                # FIX RAM: Giữ ref cũ, del tường minh trước khi tạo model mới.
                # Cho phép Metal allocator thu hồi buffer của model cũ ngay,
                # tránh window thời gian model cũ + model mới cùng tồn tại.
                old_model = model
                model = grow_model(old_model, new_channels, num_classes, device)
                del old_model                                     # Metal buffer released
                new_p = sum(p.numel() for p in model.parameters())
                print(f"  ⚡ GREW Stage {target} ({stage_names[target]}): "
                      f"{old_channels} → {new_channels} | params: {total_p:,} → {new_p:,}")
                optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.7, weight_decay=1e-4)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
                # Flush MPS cache ngay sau grow (shape mới -> allocator cần làm sạch)
                if device.type == 'mps':
                    torch.mps.empty_cache()
                gc.collect()

            if controller.converged and controller.no_improve_count >= 6:
                print(f"\n  ⚠ Converged. Stopping.")
                break

            if device.type == 'mps':
                torch.mps.empty_cache()
            gc.collect()

        except KeyboardInterrupt:
            print(f"\n\n  ⏹ Ctrl+C detected at epoch {epoch+1}. Saving results...")
            break

    # Plot
    if tl_hist:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(tl_hist, label="Train"); axes[0].plot(vl_hist, label="Val")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(ta_hist, label="Train"); axes[1].plot(va_hist, label="Val")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Acc"); axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        axes[2].plot(param_hist, color='orange', linewidth=2)
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Parameters"); axes[2].set_title("Parameter Growth"); axes[2].grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(SAVE_DIR, "custom_cnn_curves.png"), dpi=150); plt.close()
        print(f"  Plots saved to: {SAVE_DIR}custom_cnn_curves.png")

    if test_loader and best_wts:
        print(f"\n--- Test set evaluation ---")
        test_model = GrowableCNN(num_classes, channels=best_channels).to(device)
        test_model.load_state_dict(best_wts)
        test_loss, test_acc = evaluate(test_model, test_loader, criterion, device, "  Test")
        print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    print(f"\n--- Done! Best Val Loss: {best_val_loss:.4f} ---")
    print(f"Final widths: {[f'{w:.0%}' for w in controller.stage_widths]}")
    print(f"Grows: {controller.grow_count} | Converged: {controller.converged}")


if __name__ == "__main__":
    main()
