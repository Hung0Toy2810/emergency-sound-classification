import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from src.dataset import AudioDataset
from src.model import ViTAST
from src.train import calculate_metrics
from tqdm import tqdm

CLASSES = ['Crash_Collision', 'Siren', 'Scream_Shout', 'Gunshot_Explosion']

def plot_precision_recall_curves(all_targets, all_preds, save_path):
    """
    Plots the Precision-Recall Curve for each of the 4 classes.
    """
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(CLASSES):
        precision, recall, _ = precision_recall_curve(all_targets[:, i], all_preds[:, i])
        ap = average_precision_score(all_targets[:, i], all_preds[:, i])
        plt.plot(recall, precision, label=f'{class_name} (AP = {ap:.4f})', lw=2)
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves (AST Model)')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Precision-Recall Curves to '{save_path}'")

def plot_roc_curves(all_targets, all_preds, save_path):
    """
    Plots the ROC Curve (TPR vs. FPR) for each class and calculates Area Under Curve (AUC).
    """
    from sklearn.metrics import roc_curve, auc
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(all_targets[:, i], all_preds[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved ROC Curves to '{save_path}'")

def plot_class_performance(all_targets, all_preds, save_path):
    """
    Plots a side-by-side bar chart showing AP and F1 scores for each class.
    """
    aps = []
    f1s = []
    
    binary_preds = (all_preds >= 0.5).astype(float)
    
    for i in range(len(CLASSES)):
        ap = average_precision_score(all_targets[:, i], all_preds[:, i])
        f1 = f1_score(all_targets[:, i], binary_preds[:, i], zero_division=0.0)
        aps.append(ap)
        f1s.append(f1)
        
    x = np.arange(len(CLASSES))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, aps, width, label='Average Precision (AP)', color='#3498db')
    rects2 = ax.bar(x + width/2, f1s, width, label='F1-Score', color='#2ecc71')
    
    ax.set_ylabel('Scores')
    ax.set_title('Per-Class Average Precision (AP) vs F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=15)
    ax.legend(loc='lower left')
    ax.set_ylim([0, 1.15])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Class Performance Bar Chart to '{save_path}'")

def plot_confusion_matrix(all_targets, all_preds, test_dataset, save_path):
    """
    Constructs a 4x4 Confusion Matrix by mapping each audio clip's true primary group
    to the class index with the highest predicted probability.
    """
    true_labels = []
    pred_labels = []
    
    # Mapping class names to indices
    class_to_idx = {name: idx for idx, name in enumerate(CLASSES)}
    
    for idx in range(len(test_dataset.df)):
        row = test_dataset.df.iloc[idx]
        primary_group = row['primary_group']
        if primary_group in class_to_idx:
            true_labels.append(class_to_idx[primary_group])
            # Find which class has the highest prediction probability for this sample
            pred_idx = np.argmax(all_preds[idx])
            pred_labels.append(pred_idx)
            
    # Generate Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(CLASSES))))
    
    # Display Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title('Confusion Matrix (Primary Group Classes)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix to '{save_path}'")

def generate_specaugment_demo(test_dataset, save_path):
    """
    Generates a side-by-side comparison of a clean Log-Mel spectrogram vs.
    its SpecAugment-masked equivalent, suitable for reporting.
    """
    # Pick the first sample
    log_mel, _ = test_dataset[0]  # original (non-augmented) log-Mel spectrogram
    
    # Apply Frequency and Time Masking manually
    freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=16)
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=80)
    augmented = time_mask(freq_mask(log_mel.clone()))
    
    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Original spectrogram plot
    im1 = axes[0].imshow(log_mel[0].numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Original Log-Mel Spectrogram')
    axes[0].set_xlabel('Time Frames')
    axes[0].set_ylabel('Mel Frequency Bins')
    fig.colorbar(im1, ax=axes[0])
    
    # SpecAugment spectrogram plot
    im2 = axes[1].imshow(augmented[0].numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('SpecAugment Applied (Time & Frequency Masking)')
    axes[1].set_xlabel('Time Frames')
    axes[1].set_ylabel('Mel Frequency Bins')
    fig.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved SpecAugment Demo plot to '{save_path}'")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(base_dir, "..", "ast_data"))
    
    checkpoint_path = os.path.join(base_dir, "model_checkpoints", "best_model.pth")
    test_metadata = os.path.join(data_dir, "dataset_files", "test", "metadata.csv")
    test_audio_dir = os.path.join(data_dir, "dataset_files", "test")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: '{checkpoint_path}' not found. Please train the model first.")
        return
    if not os.path.exists(test_metadata):
        print(f"Error: '{test_metadata}' not found. Please run preprocessing first.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize validation/test dataset (is_train=False)
    print("Loading test dataset...")
    test_dataset = AudioDataset(
        metadata_csv_path=test_metadata,
        audio_dir=test_audio_dir,
        is_train=False,
        duration=10
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    sample_spec, _ = test_dataset[0]
    time_steps = sample_spec.shape[-1]
    
    # Initialize AST adapted ViT Model
    num_freq_patches = 12
    num_time_patches = (time_steps - 16) // 10 + 1
    model = ViTAST(
        model_name="google/vit-base-patch16-224",
        num_classes=4,
        num_freq_patches=num_freq_patches,
        num_time_patches=num_time_patches
    )
    
    # Load weights
    print(f"Loading model checkpoint from '{checkpoint_path}'...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Run predictions on all samples
    all_targets = []
    all_preds = []
    
    print("Evaluating all test samples...")
    with torch.no_grad():
        for specs, labels in tqdm(test_loader):
            specs = specs.to(device)
            logits = model(specs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_targets.append(labels.numpy())
            all_preds.append(probs)
            
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    # Create evaluation directory
    eval_results_dir = os.path.join(base_dir, "plots_and_metrics")
    os.makedirs(eval_results_dir, exist_ok=True)
    
    # Calculate metrics
    metrics = calculate_metrics(all_targets, all_preds)
    
    # Save text report
    report_path = os.path.join(eval_results_dir, "metrics_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write("             AST EVALUATION REPORT              \n")
        f.write("="*50 + "\n\n")
        f.write(f"Unified Accuracy       : {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Mean Average Precision : {metrics['mAP']*100:.2f}%\n")
        f.write(f"Macro F1-Score         : {metrics['f1_macro']*100:.2f}%\n")
        f.write(f"Micro F1-Score         : {metrics['f1_micro']*100:.2f}%\n")
        f.write(f"Macro Precision        : {metrics['precision_macro']*100:.2f}%\n")
        f.write(f"Micro Precision        : {metrics['precision_micro']*100:.2f}%\n")
        f.write(f"Macro Recall           : {metrics['recall_macro']*100:.2f}%\n")
        f.write(f"Micro Recall           : {metrics['recall_micro']*100:.2f}%\n")
        f.write(f"Macro ROC AUC          : {metrics['roc_auc_macro']:.4f}\n\n")
        f.write("Class-Wise Average Precision (AP):\n")
        f.write("-"*50 + "\n")
        for idx, class_name in enumerate(CLASSES):
            f.write(f"{class_name:<18} : {metrics['class_aps'][idx]*100:.2f}%\n")
        f.write("="*50 + "\n")
    print(f"Saved detailed metrics report to '{report_path}'")
    
    # Generate and save plots
    print("\nGenerating evaluation graphics...")
    plot_precision_recall_curves(all_targets, all_preds, os.path.join(eval_results_dir, "precision_recall_curves.png"))
    plot_roc_curves(all_targets, all_preds, os.path.join(eval_results_dir, "roc_curves.png"))
    plot_class_performance(all_targets, all_preds, os.path.join(eval_results_dir, "metrics_bar_chart.png"))
    plot_confusion_matrix(all_targets, all_preds, test_dataset, os.path.join(eval_results_dir, "confusion_matrix.png"))
    generate_specaugment_demo(test_dataset, os.path.join(eval_results_dir, "specaugment_demo.png"))
    
    print("\n" + "="*50)
    print("             AST TEST EVALUATION METRICS             ")
    print("="*50)
    print(f"Unified Accuracy             : {metrics['accuracy']*100:.2f}%")
    print(f"Mean Average Precision (mAP) : {metrics['mAP']*100:.2f}%")
    print(f"Macro F1-Score               : {metrics['f1_macro']*100:.2f}%")
    print(f"Macro Precision              : {metrics['precision_macro']*100:.2f}%")
    print(f"Macro Recall                 : {metrics['recall_macro']*100:.2f}%")
    print(f"Macro ROC AUC                : {metrics['roc_auc_macro']:.4f}")
    print("="*50)
    print(f"\nSUCCESS: All evaluation graphics successfully saved to '{eval_results_dir}'")
    print("="*50)

if __name__ == "__main__":
    main()
