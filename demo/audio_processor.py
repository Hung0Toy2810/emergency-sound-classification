"""
Audio preprocessing for both SoundCNN and ViTAST inference pipelines.
"""

import base64
import io
from functools import lru_cache
from typing import TypedDict, cast

import librosa
import librosa.display
import matplotlib
import numpy as np
import torch
import torchaudio
from PIL import Image
from torchvision import transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class AudioInputs(TypedDict):
    cnn: torch.Tensor | None
    ast: torch.Tensor | None
    spectrogram: str | None


CNN_SR = 22050
CNN_DURATION = 4.0
CNN_SAMPLES = int(CNN_SR * CNN_DURATION)
CNN_N_MELS = 128
CNN_IMG_SIZE = 224

AST_SR = 16000
AST_DURATION = 10
AST_SAMPLES = AST_SR * AST_DURATION
AST_N_MELS = 128


@lru_cache(maxsize=1)
def _cnn_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((CNN_IMG_SIZE, CNN_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


@lru_cache(maxsize=1)
def _ast_mel_transform():
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=AST_SR,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=AST_N_MELS,
        f_min=0,
        f_max=8000,
    )


def _fixed_length_np(y, target_samples):
    if len(y) < target_samples:
        return np.pad(y, (0, target_samples - len(y)))
    return y[:target_samples]


def _load_audio(audio_path, sr, duration=None):
    max_duration = duration if duration is not None else None
    y, _ = librosa.load(audio_path, sr=sr, mono=True, duration=max_duration)
    return y.astype(np.float32, copy=False)


def _mel_db_from_audio(y, sr, n_mels=128):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=2048,
        hop_length=512,
        fmin=0,
        fmax=sr // 2,
    )
    return librosa.power_to_db(mel, ref=np.max)


def _cnn_tensor_from_mel_db(mel_db):
    # Keep the image-rendering step because the CNN checkpoint was trained on
    # rendered mel images, not raw mel tensors.
    buf = io.BytesIO()
    fig = plt.figure(figsize=(CNN_IMG_SIZE / 100, CNN_IMG_SIZE / 100), dpi=100)
    ax = Axes(fig, (0.0, 0.0, 1.0, 1.0))
    ax.set_axis_off()
    fig.add_axes(ax)
    librosa.display.specshow(
        mel_db,
        sr=CNN_SR,
        hop_length=512,
        fmin=0,
        fmax=CNN_SR // 2,
        ax=ax,
    )
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB")
    # Compose changes PIL input into a Tensor through ToTensor.
    tensor = cast(torch.Tensor, _cnn_eval_transform()(img))
    return tensor.unsqueeze(0)


def _ast_tensor_from_audio(y):
    y = _fixed_length_np(y, AST_SAMPLES)
    waveform = torch.from_numpy(y).unsqueeze(0)
    mel_spec = _ast_mel_transform()(waveform)
    log_mel = torch.log(mel_spec + 1e-6)
    mean = log_mel.mean()
    std = log_mel.std()
    log_mel = 0.5 * (log_mel - mean) / (std + 1e-6)
    return log_mel.unsqueeze(0)


def prepare_audio_inputs(audio_path, need_cnn=True, need_ast=True, need_visual=True) -> AudioInputs:
    """
    Decode the uploaded audio once per target sample rate and build all requested
    model inputs. This avoids the old path that decoded the same file three times.
    """
    outputs: AudioInputs = {"cnn": None, "ast": None, "spectrogram": None}

    cnn_audio = None
    if need_cnn or need_visual:
        cnn_audio = _load_audio(audio_path, sr=CNN_SR)

    if need_cnn:
        cnn_clip = _fixed_length_np(cnn_audio, CNN_SAMPLES)
        cnn_mel_db = _mel_db_from_audio(cnn_clip, CNN_SR, CNN_N_MELS)
        outputs["cnn"] = _cnn_tensor_from_mel_db(cnn_mel_db)

    if need_ast:
        ast_audio = _load_audio(audio_path, sr=AST_SR, duration=AST_DURATION)
        outputs["ast"] = _ast_tensor_from_audio(ast_audio)

    if need_visual:
        outputs["spectrogram"] = generate_spectrogram_visual_from_audio(cnn_audio, CNN_SR)

    return outputs


def preprocess_for_cnn(audio_path):
    return prepare_audio_inputs(audio_path, need_cnn=True, need_ast=False, need_visual=False)["cnn"]


def preprocess_for_ast(audio_path):
    return prepare_audio_inputs(audio_path, need_cnn=False, need_ast=True, need_visual=False)["ast"]


def generate_spectrogram_visual(audio_path):
    y = _load_audio(audio_path, sr=CNN_SR)
    return generate_spectrogram_visual_from_audio(y, CNN_SR)


def generate_spectrogram_visual_from_audio(y, sr):
    """
    Generate a base64 encoded waveform + mel spectrogram PNG for the UI.
    """
    duration = max(len(y) / sr, 1e-6)
    mel_db = _mel_db_from_audio(y, sr, n_mels=128)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={"height_ratios": [1, 2]})
    fig.patch.set_facecolor("#12121a")

    times = np.linspace(0, duration, num=len(y))
    axes[0].plot(times, y, color="#00d4ff", linewidth=0.4, alpha=0.8)
    axes[0].fill_between(times, y, alpha=0.15, color="#00d4ff")
    axes[0].set_xlim(0, duration)
    axes[0].set_ylabel("Amplitude", color="#a0a0b0", fontsize=9)
    axes[0].set_title("Waveform & Mel Spectrogram", color="#f0f0f0", fontsize=11, fontweight="bold")
    axes[0].tick_params(colors="#a0a0b0", labelsize=8)
    axes[0].set_facecolor("#12121a")
    for spine in axes[0].spines.values():
        spine.set_color("#2a2a3a")

    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        ax=axes[1],
        cmap="magma",
    )
    axes[1].set_ylabel("Hz", color="#a0a0b0", fontsize=9)
    axes[1].set_xlabel("Time (s)", color="#a0a0b0", fontsize=9)
    axes[1].tick_params(colors="#a0a0b0", labelsize=8)
    axes[1].set_facecolor("#12121a")
    for spine in axes[1].spines.values():
        spine.set_color("#2a2a3a")

    fig.tight_layout(pad=1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor="#12121a", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
