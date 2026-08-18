"""
Load cả 2 mô hình SoundCNN và ViTAST một lần khi server khởi động.
"""

import sys
import os
import torch

# ── Đường dẫn project ────────────────────────────────────────────────
CNN_CHECKPOINT = os.path.join(os.path.dirname(__file__), "models", "custom_cnn_best.pth")
AST_CHECKPOINT = os.path.join(os.path.dirname(__file__), "models", "best_model.pth")


def get_device():
    """Chọn GPU nếu có, ngược lại dùng CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cnn_model(device=None):
    """
    Load SoundCNN từ checkpoint.
    Returns: (model, class_names)
    """
    if device is None:
        device = get_device()

    from src.cnn_model import SoundCNN

    checkpoint = torch.load(CNN_CHECKPOINT, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    img_size = checkpoint.get("img_size", 224)

    model = SoundCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[CNN] Loaded SoundCNN — {len(class_names)} classes, img_size={img_size}")
    print(f"      Classes: {class_names}")
    return model, class_names


def load_ast_model(device=None):
    """
    Load ViTAST (Audio Spectrogram Transformer) từ checkpoint.
    Cần internet lần đầu để tải ViT pretrained từ Hugging Face.
    Returns: (model, class_names)
    """
    if device is None:
        device = get_device()

    from src.ast_model import ViTAST

    classes = ["Crash_Collision", "Siren", "Scream_Shout", "Gunshot_Explosion"]

    # Tham số patches cho audio 10s @16kHz, hop=160
    # time_steps = (16000 * 10) // 160 + 1 = 1001
    # num_time_patches = (1001 - 16) // 10 + 1 = 99
    NUM_FREQ_PATCHES = 12
    NUM_TIME_PATCHES = 99

    print("[AST] Đang khởi tạo ViTAST (có thể tải ViT pretrained lần đầu)...")
    model = ViTAST(
        model_name="google/vit-base-patch16-224",
        num_classes=4,
        num_freq_patches=NUM_FREQ_PATCHES,
        num_time_patches=NUM_TIME_PATCHES,
    )

    checkpoint = torch.load(AST_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"[AST] Loaded ViTAST — {len(classes)} classes")
    print(f"      Classes: {classes}")
    return model, classes
