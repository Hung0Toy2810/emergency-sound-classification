"""Check evaluator reductions and the retained demo-compatible checkpoint."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

CNN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CNN_ROOT))
from evaluator import ModelEvaluator


def test_evaluation_weights_the_partial_last_batch():
    logits = torch.tensor([[3.0, 1.0], [1.0, 3.0], [4.0, 0.0]])
    labels = torch.tensor([0, 1, 1])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    evaluator = ModelEvaluator(torch.device("cpu"))
    model = torch.nn.Identity()
    loss, accuracy = evaluator.evaluate(model, loader, torch.nn.CrossEntropyLoss())
    assert loss == pytest.approx(torch.nn.functional.cross_entropy(logits, labels).item())
    assert accuracy == pytest.approx(2 / 3)
    assert evaluator.collect_predictions(model, loader) == ([0, 1, 1], [0, 1, 0])


def test_empty_evaluation_fails_clearly():
    loader = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
    with pytest.raises(ValueError, match="empty"):
        ModelEvaluator(torch.device("cpu")).evaluate(
            torch.nn.Identity(), loader, torch.nn.CrossEntropyLoss()
        )


def test_retained_checkpoint_loads_into_demo_architecture():
    path = CNN_ROOT.parent / "demo" / "src" / "cnn_model.py"
    spec = importlib.util.spec_from_file_location("soundcnn_demo_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = torch.load(
        CNN_ROOT / "models" / "custom_cnn_best.pth",
        map_location="cpu",
        weights_only=True,
    )
    assert set(checkpoint["class_names"]) == {
        "Crash_Collision", "Gunshot_Explosion", "Scream_Shout", "Siren"
    }
    model = module.SoundCNN(num_classes=4)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        prediction = model(torch.zeros(1, 3, 224, 224))
    assert prediction.shape == (1, 4)
    assert torch.isfinite(prediction).all()
