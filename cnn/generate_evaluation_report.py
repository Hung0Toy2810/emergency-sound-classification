import os

import matplotlib
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from config import CHECKPOINT_NAME, DATA_PATH, SAVE_DIR
from data_module import SoundDataModule
from evaluator import ModelEvaluator
from model import SoundCNN
from utils import get_device

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    device = get_device()
    out_dir = "evaluation_report"
    checkpoint_path = os.path.join(SAVE_DIR, CHECKPOINT_NAME)
    os.makedirs(out_dir, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    model = SoundCNN(len(class_names)).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError:
        print("Checkpoint is not compatible with the simplified SoundCNN. Please train again first.")
        return

    data = SoundDataModule(DATA_PATH, batch_size=16, device=device).setup()
    _, _, test_loader = data.loaders()
    if test_loader is None:
        print("No test split found.")
        return

    y_true, y_pred = ModelEvaluator(device).collect_predictions(model, test_loader)
    report = classification_report(y_true, y_pred, target_names=class_names)
    if not isinstance(report, str):
        raise TypeError("Expected a text classification report.")
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    print(f"Report saved to {out_dir}")


if __name__ == "__main__":
    main()
