import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

def calculate_metrics(targets, preds):
    """
    Calculates Mean Average Precision (mAP), unified classification accuracy,
    and standard F1, Precision, Recall, and ROC AUC metrics.
    """
    num_classes = targets.shape[1]
    class_aps = []
    
    # Calculate Average Precision (AP) for each class
    for i in range(num_classes):
        try:
            ap = average_precision_score(targets[:, i], preds[:, i])
            class_aps.append(ap)
        except Exception:
            class_aps.append(0.0)  # Safe fallback in case of single-class edge cases
            
    mAP = np.mean(class_aps)
    
    # Unified Accuracy: class with highest predicted prob vs. class with highest target
    # (Since target labels are one-hot represented as multi-hot binary arrays)
    try:
        targets_argmax = np.argmax(targets, axis=1)
        preds_argmax = np.argmax(preds, axis=1)
        accuracy = accuracy_score(targets_argmax, preds_argmax)
    except Exception:
        accuracy = 0.0
        
    binary_preds = (preds >= 0.5).astype(float)
    try:
        f1_macro = f1_score(targets, binary_preds, average='macro', zero_division=0.0)
        f1_micro = f1_score(targets, binary_preds, average='micro', zero_division=0.0)
        precision_macro = precision_score(targets, binary_preds, average='macro', zero_division=0.0)
        precision_micro = precision_score(targets, binary_preds, average='micro', zero_division=0.0)
        recall_macro = recall_score(targets, binary_preds, average='macro', zero_division=0.0)
        recall_micro = recall_score(targets, binary_preds, average='micro', zero_division=0.0)
    except Exception:
        f1_macro = f1_micro = precision_macro = precision_micro = recall_macro = recall_micro = 0.0

    try:
        roc_auc_macro = roc_auc_score(targets, preds, average='macro')
    except Exception:
        roc_auc_macro = 0.0
        
    return {
        'class_aps': class_aps,
        'mAP': mAP,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'precision_micro': precision_micro,
        'recall_macro': recall_macro,
        'recall_micro': recall_micro,
        'roc_auc_macro': roc_auc_macro
    }

try:
    from torch.amp import autocast, GradScaler
    _amp_available = True
except ImportError:
    try:
        from torch.cuda.amp import autocast, GradScaler
        _amp_available = True
    except ImportError:
        _amp_available = False

def train_one_epoch(model, dataloader, optimizer, criterion, device, max_batches=None, scaler=None):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training")
    
    use_amp = _amp_available and (device.type == 'cuda') and (scaler is not None)
    
    for i, (specs, labels) in enumerate(progress_bar):
        if max_batches is not None and i >= max_batches:
            break
        specs = specs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(specs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    num_steps = max_batches if max_batches is not None else len(dataloader)
    return running_loss / num_steps

def validate(model, dataloader, criterion, device, max_batches=None, use_amp=False):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    
    amp_active = _amp_available and (device.type == 'cuda') and use_amp
    
    with torch.no_grad():
        for i, (specs, labels) in enumerate(tqdm(dataloader, desc="Validation")):
            if max_batches is not None and i >= max_batches:
                break
            specs = specs.to(device)
            labels = labels.to(device)
            
            if amp_active:
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(specs)
                    loss = criterion(logits, labels)
            else:
                logits = model(specs)
                loss = criterion(logits, labels)
                
            running_loss += loss.item()
            
            # Convert logits to probabilities
            probs = torch.sigmoid(logits)
            
            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())
            
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    metrics = calculate_metrics(all_targets, all_preds)
    num_steps = max_batches if max_batches is not None else len(dataloader)
    metrics['val_loss'] = running_loss / num_steps
    
    return metrics

def run_training(model, train_dataset, val_dataset, epochs=10, batch_size=8, lr=2e-5, save_dir=None, max_batches=None, resume=False):
    if save_dir is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.abspath(os.path.join(src_dir, "..", "model_checkpoints"))
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize AMP GradScaler
    use_amp = _amp_available and (device.type == 'cuda')
    scaler = None
    if use_amp:
        try:
            scaler = GradScaler(device_type='cuda')
        except TypeError:
            scaler = GradScaler()
        print("AMP (Automatic Mixed Precision) FP16 is enabled.")
        
    # Create DataLoaders
    # Pin memory for faster GPU transfer if CUDA is available
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)
    
    model = model.to(device)
    
    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    start_epoch = 1
    best_mAP = -1.0
    
    # Resume checkpoint logic
    if resume:
        latest_path = os.path.join(save_dir, "latest_model.pth")
        best_path = os.path.join(save_dir, "best_model.pth")
        checkpoint_path = None
        
        if os.path.exists(latest_path):
            checkpoint_path = latest_path
        elif os.path.exists(best_path):
            checkpoint_path = best_path
            
        if checkpoint_path is not None:
            print(f"\n>>> Resuming training from checkpoint: {checkpoint_path} <<<")
            try:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                except TypeError:
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_mAP = checkpoint.get('mAP', -1.0)
                print(f"Successfully loaded checkpoint. Resuming from Epoch {start_epoch}...")
            except Exception as e:
                print(f"Failed to load checkpoint ({e}). Starting from scratch.")
        else:
            print("\nNo checkpoint found. Starting training from scratch.")
            
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, max_batches, scaler)
        print(f"Train Loss: {train_loss:.4f}")
        
        val_metrics = validate(model, val_loader, criterion, device, max_batches, use_amp)
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"Val mAP: {val_metrics['mAP']:.4f} | Val Accuracy: {val_metrics['accuracy']:.4f} | Val F1 (Macro): {val_metrics['f1_macro']:.4f}")
        print(f"Val Precision (Macro): {val_metrics['precision_macro']:.4f} | Val Recall (Macro): {val_metrics['recall_macro']:.4f} | Val ROC AUC (Macro): {val_metrics['roc_auc_macro']:.4f}")
        
        for idx, cat in enumerate(train_dataset.classes):
            print(f"  - {cat} AP: {val_metrics['class_aps'][idx]:.4f}")
            
        # 1. Always save latest checkpoint
        latest_checkpoint_path = os.path.join(save_dir, "latest_model.pth")
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mAP': best_mAP,
            }, latest_checkpoint_path)
        except Exception as e:
            print(f"Warning: Failed to save latest checkpoint ({e})")
            
        # 2. Checkpoint based on best mAP
        if val_metrics['mAP'] > best_mAP:
            best_mAP = val_metrics['mAP']
            checkpoint_path = os.path.join(save_dir, "best_model.pth")
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'mAP': best_mAP,
                }, checkpoint_path)
                print(f"*** New best validation mAP: {best_mAP:.4f}! Saved weights to {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Failed to save best checkpoint ({e})")

            
    print("\nTraining completed successfully!")

