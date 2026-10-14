import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrainingPlotter:
    """Vẽ biểu đồ loss và accuracy sau khi train."""

    def plot_history(self, history, output_path, eval_name="Eval"):
        epochs = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, history["train_loss"], label="Train")
        axes[0].plot(epochs, history["eval_loss"], label=eval_name)
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[1].plot(epochs, history["train_acc"], label="Train")
        axes[1].plot(epochs, history["eval_acc"], label=eval_name)
        axes[1].set_title("Accuracy")
        axes[1].legend()
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
