"""
Flask backend cho demo so sánh SoundCNN vs ViTAST.
"""

import sys
import io

# Force UTF-8 for standard streams on Windows to avoid UnicodeEncodeErrors
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import os
import time
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from model_loader import load_cnn_model, load_ast_model, get_device
from audio_processor import prepare_audio_inputs

# ── Paths cho confusion matrix images ─────────────────────────────────
CNN_CM_PATH = os.path.join(os.path.dirname(__file__), "static", "images", "cnn_confusion_matrix.png")
AST_CM_PATH = os.path.join(os.path.dirname(__file__), "static", "images", "ast_confusion_matrix.png")

# ── Khởi tạo Flask ───────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

# ── Load models (1 lần khi server start) ──────────────────────────────
device = get_device()
print(f"\n{'='*50}")
print(f"  Device: {device}")
print(f"{'='*50}\n")

cnn_model = None
cnn_classes = []
ast_model = None
ast_classes = []
model_status = {
    "cnn": {"ready": False, "error": None},
    "ast": {"ready": False, "error": None},
}


def _safe_load_models():
    """Load từng model độc lập để demo vẫn chạy nếu một model lỗi."""
    global cnn_model, cnn_classes, ast_model, ast_classes

    print("Loading SoundCNN...")
    try:
        cnn_model, cnn_classes = load_cnn_model(device)
        model_status["cnn"] = {"ready": True, "error": None}
    except Exception as exc:
        model_status["cnn"] = {"ready": False, "error": str(exc)}
        print(f"[CNN] Load failed: {exc}")

    print("\nLoading ViTAST (AST)...")
    try:
        ast_model, ast_classes = load_ast_model(device)
        model_status["ast"] = {"ready": True, "error": None}
    except Exception as exc:
        model_status["ast"] = {"ready": False, "error": str(exc)}
        print(f"[AST] Load failed: {exc}")

    ready = [name.upper() for name, status in model_status.items() if status["ready"]]
    print(f"\n{'='*50}")
    print(f"  Models ready: {', '.join(ready) if ready else 'none'}")
    print(f"{'='*50}\n")


_safe_load_models()


# ── Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Trạng thái server và model để kiểm tra nhanh trước khi demo."""
    status_code = 200 if any(status["ready"] for status in model_status.values()) else 503
    return jsonify({
        "device": str(device),
        "models": model_status,
        "limits": {
            "max_upload_mb": app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
            "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
        },
    }), status_code


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(_error):
    return jsonify({"error": "File quá lớn. Giới hạn upload là 50 MB."}), 413


def _extension_allowed(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _model_unavailable_payload(model_name, status):
    return {
        "model": model_name,
        "available": False,
        "error": status["error"] or "Model chưa sẵn sàng.",
    }


def _run_cnn_prediction(cnn_input):
    if cnn_model is None or cnn_input is None:
        return _model_unavailable_payload("SoundCNN", model_status["cnn"])

    t0 = time.perf_counter()
    with torch.inference_mode():
        cnn_logits = cnn_model(cnn_input.to(device))
        cnn_probs = F.softmax(cnn_logits, dim=1).cpu().numpy()[0]
    cnn_time = (time.perf_counter() - t0) * 1000

    pred_idx = int(cnn_probs.argmax())
    return {
        "model": "SoundCNN",
        "available": True,
        "predicted_class": cnn_classes[pred_idx],
        "confidence": float(cnn_probs[pred_idx]) * 100,
        "inference_ms": round(cnn_time, 1),
        "probabilities": {
            cls: round(float(prob) * 100, 2) for cls, prob in zip(cnn_classes, cnn_probs)
        },
    }


def _run_ast_prediction(ast_input):
    if ast_model is None or ast_input is None:
        return _model_unavailable_payload("ViTAST (AST)", model_status["ast"])

    t0 = time.perf_counter()
    with torch.inference_mode():
        ast_logits = ast_model(ast_input.to(device))
        ast_probs = torch.sigmoid(ast_logits).cpu().numpy()[0]
    ast_time = (time.perf_counter() - t0) * 1000

    pred_idx = int(ast_probs.argmax())
    return {
        "model": "ViTAST (AST)",
        "available": True,
        "predicted_class": ast_classes[pred_idx],
        "confidence": float(ast_probs[pred_idx]) * 100,
        "inference_ms": round(ast_time, 1),
        "probabilities": {
            cls: round(float(prob) * 100, 2) for cls, prob in zip(ast_classes, ast_probs)
        },
    }


@app.route("/predict", methods=["POST"])
def predict():
    """Nhận file audio, chạy cả 2 model, trả JSON kết quả."""
    if "audio" not in request.files:
        return jsonify({"error": "Không tìm thấy file audio"}), 400

    audio_file = request.files["audio"]
    filename = audio_file.filename
    if not filename:
        return jsonify({"error": "Chưa chọn file"}), 400
    if not _extension_allowed(filename):
        return jsonify({
            "error": "Định dạng không hỗ trợ. Hãy dùng wav, mp3, flac, ogg, m4a, aac hoặc wma."
        }), 400
    if not any(status["ready"] for status in model_status.values()):
        return jsonify({
            "error": "Không có model nào load thành công. Kiểm tra /health và log server."
        }), 503

    # Lưu file tạm
    suffix = os.path.splitext(filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Prepare inputs once to avoid redundant decoding of the same audio file
        need_cnn = cnn_model is not None
        need_ast = ast_model is not None
        inputs = prepare_audio_inputs(tmp_path, need_cnn=need_cnn, need_ast=need_ast, need_visual=True)

        cnn_result = _run_cnn_prediction(inputs["cnn"])
        ast_result = _run_ast_prediction(inputs["ast"])

        return jsonify({
            "cnn": cnn_result,
            "ast": ast_result,
            "spectrogram": inputs["spectrogram"],
            "filename": audio_file.filename,
            "device": str(device),
        })

    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý: {str(e)}"}), 500

    finally:
        # Xóa file tạm
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/metrics")
def metrics():
    """Trả về dữ liệu metrics so sánh (hardcoded từ evaluation reports)."""
    return jsonify({
        "cnn": {
            "name": "SoundCNN",
            "architecture": "5-block SoundCNN (32→64→128→256→512)",
            "params": "4.78M",
            "size_mb": "18.3",
            "accuracy": 64.21,
            "macro_precision": 65.78,
            "macro_recall": 66.14,
            "macro_f1": 65.88,
            "per_class": {
                "Crash_Collision":    {"precision": 57, "recall": 53, "f1": 55},
                "Gunshot_Explosion":  {"precision": 58, "recall": 61, "f1": 59},
                "Scream_Shout":      {"precision": 64, "recall": 71, "f1": 68},
                "Siren":             {"precision": 84, "recall": 80, "f1": 82},
            },
        },
        "ast": {
            "name": "ViTAST (Transformer)",
            "architecture": "ViT-Base/16 for audio",
            "params": "86.6M",
            "size_mb": "~330",
            "accuracy": 83.67,
            "macro_precision": 85.94,
            "macro_recall": 84.03,
            "macro_f1": 84.69,
            "mAP": 92.04,
            "roc_auc": 96.22,
            "per_class": {
                "Crash_Collision":    {"ap": 87.83, "f1": 80.0},
                "Siren":             {"ap": 98.57, "f1": 91.0},
                "Scream_Shout":      {"ap": 91.99, "f1": 85.0},
                "Gunshot_Explosion":  {"ap": 89.77, "f1": 83.0},
            },
        },
    })


@app.route("/images/cnn_cm")
def cnn_confusion_matrix():
    """Serve confusion matrix image của CNN."""
    if os.path.exists(CNN_CM_PATH):
        return send_file(CNN_CM_PATH, mimetype="image/png")
    return "Not found", 404


@app.route("/images/ast_cm")
def ast_confusion_matrix():
    """Serve confusion matrix image của AST."""
    if os.path.exists(AST_CM_PATH):
        return send_file(AST_CM_PATH, mimetype="image/png")
    return "Not found", 404


if __name__ == "__main__":
    print("\n🚀 Server đang chạy tại http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
