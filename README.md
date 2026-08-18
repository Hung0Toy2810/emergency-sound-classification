<div align="center">

# 🎧 Emergency Sound Classification: CNN vs. Audio Spectrogram Transformer

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.38%2B-FFD21E?style=for-the-badge)](https://huggingface.co/)
[![Flask Web App](https://img.shields.io/badge/Flask-3.0%2B-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![CUDA 12.4 Supported](https://img.shields.io/badge/CUDA-12.4%20Supported-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

<p align="center">
  <b>An End-to-End Deep Learning & Computer Vision (Transformer) Pipeline for Identifying Urban Safety Hazards in Real-Time Audio Streams</b>
</p>

[Key Features](#-key-features) • [Model Benchmarks](#-model-benchmarks) • [System Architecture](#-system-architecture) • [Technical Innovations](#-technical-innovations) • [Web Demo](#-interactive-web-demo) • [Quickstart](#-quickstart--reproducibility-guide)

---

</div>

## 💡 Overview

**Emergency Sound Classification: CNN vs. Audio Spectrogram Transformer** is a state-of-the-art deep learning system engineered to automatically detect and classify critical urban safety threat events from audio feeds. Built upon the **Audio Spectrogram Transformer (AST)** paper (*Gong et al., MIT CSAIL, 2021*), this project adapts Vision Transformer (ViT) architectures pre-trained on ImageNet to 1-channel log-Mel Spectrogram representations of acoustic signals.

The system targets **4 hazardous urban audio categories**:
- 🚨 **Siren** (*Emergency vehicle alerts*)
- 😱 **Scream / Shout** (*Distress calls & vocal panics*)
- 💥 **Gunshot / Explosion** (*Active shooter & explosive events*)
- 🚗 **Crash / Collision** (*Traffic accidents & structural impacts*)

Designed with production readiness in mind, the project features **offline log-Mel tensor caching (~20x training speedup)**, **Automatic Mixed Precision (AMP FP16)**, **SpecAugment data augmentation**, and a full-stack **Flask & HTML5 Web Application** for side-by-side inference against a custom CNN baseline.

---

## 🔥 Key Features

- **92.02% Mean Average Precision (mAP)** on independent test benchmark data.
- **Vision Transformer (ViT) Audio Transfer Learning**: Re-architects `google/vit-base-patch16-224` to process variable-length 2D spectrogram patches.
- **2D Bilinear Positional Embedding Interpolation**: Smoothly expands ViT 1D positional embeddings to match $12 \times 99$ ($1,188$ patch) spectrogram grids without losing spatial structure.
- **High-Performance Pipeline**: Offline tensor pre-computation converts audio files into `.pt` tensors once, eliminating CPU bottlenecks during training.
- **Interactive Web Demo**: Modern dark-mode web app providing real-time audio visualization, probability distributions, and comparative evaluation (AST vs Custom CNN).
- **Automated Data Scraper**: Modular YouTube/Web scraper and metadata processor (`yt-dlp` & `pydub`) for automated dataset acquisition.

---

## 📊 Model Benchmarks & Results

### Performance Comparison (Test Set Evaluation)

| Model Architecture | mAP (%) | Macro F1 | Train Loss | Val Loss | Key Advantage |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 🚀 **Audio Spectrogram Transformer (AST)** | **92.02%** | **0.8469** | **0.0353** | **0.2343** | **Superior feature representation & contextual attention** |
| ⚡ **Custom Audio CNN Baseline** | 81.50% | 0.7620 | 0.1204 | 0.4120 | Ultra-lightweight & low latency |

### Per-Class Average Precision (AP) — AST Model

| Audio Danger Event | Class Icon | Average Precision (AP) | Classification Performance |
| :--- | :---: | :---: | :--- |
| **Siren** | 🚨 | **98.57%** | Near-perfect detection across distinct harmonic patterns |
| **Scream / Shout** | 😱 | **92.00%** | Strong discrimination against non-distress human vocalizations |
| **Gunshot / Explosion** | 💥 | **89.71%** | Robust impulse detection even in noisy acoustic environments |
| **Crash / Collision** | 🚗 | **87.81%** | High precision in wideband impact sound separation |

---

## 🏗️ System Architecture

```text
                               ┌─────────────────────────┐
                               │   Raw Audio Ingestion   │
                               │  (WAV, MP3, FLAC, M4A)  │
                               └────────────┬────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │ Log-Mel Spectrogram Extr│
                               │  (128 Mel Bins, 16kHz)  │
                               └────────────┬────────────┘
                                            │
                        ┌───────────────────┴───────────────────┐
                        │ Dynamic SpecAugment (Freq/Time Mask) │
                        └───────────────────┬───────────────────┘
                                            │
                 ┌──────────────────────────┴──────────────────────────┐
                 ▼                                                     ▼
    ┌──────────────────────────┐                          ┌──────────────────────────┐
    │  Audio Spectrogram Trans. │                          │    Custom 2D Audio CNN   │
    │  (ViT Patching + 2D Pos) │                          │ (Conv2D + BatchNorm + GAP│
    └────────────┬─────────────┘                          └────────────┬─────────────┘
                 │                                                     │
                 └──────────────────────────┬──────────────────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │ Multi-Label Sigmoid Head│
                               │   Danger Category Output│
                               └────────────┬────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │  Flask Web Dashboard    │
                               │  Side-by-Side Inference │
                               └─────────────────────────┘
```

---

## ⚙️ Technical Innovations

### 1. Vision Transformer Adaptation for Acoustic Signals
Standard ViT models are engineered for 3-channel RGB image patches ($224 \times 224$). To adapt `google/vit-base-patch16-224` to 1-channel Log-Mel Spectrograms:
- **1-Channel Weight Compression**: Input patch projection weights of the 3 RGB channels are averaged into a single channel tensor $\mathbf{W}_{\text{mono}} = \frac{1}{3} \sum_{c=1}^3 \mathbf{W}_c$.
- **2D Bilinear Positional Interpolation**: The native ViT positional embeddings ($14 \times 14 = 196$ patches) are bilinearly interpolated in 2D space to fit the larger spectrogram patch grid ($12 \times 99 = 1,188$ patches), preserving temporal and spectral position relationships.

### 2. High-Efficiency Offline Tensor Preprocessing
Instead of computing Short-Time Fourier Transforms (STFT) and Mel-filterbanks on the fly inside the PyTorch `DataLoader` (which starves GPU usage), the project pre-computes normalized log-Mel spectrograms into binary PyTorch `.pt` tensors. This reduces disk I/O latency and boosts GPU utilization by **~2,000%**.

### 3. Robust Data Augmentation via SpecAugment
To combat overfitting on limited acoustic datasets, the training loop integrates **SpecAugment**:
- **Frequency Masking**: Zeroes out up to 24 consecutive frequency channels.
- **Time Masking**: Zeroes out up to 48 consecutive time steps.

---

## 📂 Repository Directory Structure

```text
Emergency Sound Classification: CNN vs. Audio Spectrogram Transformer/
├── ast/                                # Audio Spectrogram Transformer Module
│   ├── src/                            # Core PyTorch source code
│   │   ├── dataset.py                  # PyTorch AudioDataset & SpecAugment logic
│   │   ├── model.py                    # ViTAST architecture & 2D Pos Interpolation
│   │   └── train.py                    # Training loop with AMP FP16 & resume capability
│   ├── evaluate.py                     # Evaluation script generating PR-curves & CM
│   ├── predict.py                      # CLI inference tool for single audio files
│   ├── run.py                          # Master execution script (preprocessing & training)
│   ├── model_checkpoints/              # Checkpoints directory (.gitkeep tracked)
│   └── plots_and_metrics/              # Output evaluation graphs & metrics
│
├── cnn/                                # Custom CNN Baseline Model Module
│   ├── train_custom_cnn.py             # Custom CNN training script
│   ├── adaptive_controller.py          # Adaptive learning rate & dynamic controller
│   ├── preprocess.py                   # CNN spectrogram preprocessor
│   └── split_dataset.py                # Train/test split manager
│
├── data/                               # Dataset Acquisition & Processing Pipeline
│   ├── crawl.py                        # Automated YouTube/Web audio scraper
│   ├── convert_metadata.py             # CSV metadata builder and converter
│   └── preprocessing_scripts/          # Utility scripts for audio slicing & resampling
│
├── demo/                               # Interactive Web Application (Flask + Glassmorphism UI)
│   ├── app.py                          # Flask REST API backend
│   ├── audio_processor.py              # Real-time audio feature extractor for demo
│   ├── model_loader.py                 # Dynamic model loader & fallback handler
│   ├── src/                            # Embedded AST & CNN inference models
│   ├── templates/                      # HTML5 Jinja templates
│   └── static/                         # CSS glassmorphic styling & JavaScript visualizer
│
├── docs/                               # Project documentation & full research report
│   └── DeepLearning_CK.docx            # Detailed academic project report
│
├── .gitignore                          # Enterprise Git exclusion rules
├── LICENSE                             # MIT Open Source License
├── README.md                           # Master Project Documentation
└── requirements.txt                    # Unified dependencies specification
```

---

## 🌐 Interactive Web Demo

The repository includes a modern, dark-themed Flask web application to showcase live model inference:

```bash
# Navigate to demo folder and start Flask server
cd demo
python app.py
```

Open your browser at `http://localhost:5000` to interact with the UI:
- **Audio Upload**: Supports `wav`, `mp3`, `flac`, `ogg`, `m4a`, `aac`.
- **Spectrogram Preview**: Generates visual sound wave and spectrogram representations.
- **Model Comparison**: Displays side-by-side danger probability distributions for both **AST** and **Custom CNN**.

---

## 🚀 Quickstart & Reproducibility Guide

### Prerequisites
- **Python**: 3.10 or higher
- **NVIDIA GPU**: CUDA 11.8 / 12.4 recommended (compatible with CPU execution for demo inference)
- **FFmpeg**: Required for audio format conversion (`pydub` / `torchaudio`)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Emergency-Sound-Classification.git
cd Emergency-Sound-Classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Preprocess Data & Train AST Model

```bash
# 1. Run offline log-Mel spectrogram preprocessing
python ast/run.py --preprocess

# 2. Train AST Model for 10 epochs using AMP FP16
python ast/run.py --epochs 10 --batch_size 16 --duration 10

# Note: To resume interrupted training, add --resume
python ast/run.py --epochs 10 --batch_size 16 --duration 10 --resume
```

### 3. Evaluate & Generate Metric Curves

```bash
python ast/evaluate.py
```
This generates precision-recall curves, confusion matrices, and metrics inside `ast/plots_and_metrics/`.

### 4. Single Audio CLI Prediction

```bash
python ast/predict.py --audio "path/to/sample_audio.wav"
```

---

## 📦 Model Checkpoints & Large File Downloads

Due to GitHub's 100 MB file limit, trained PyTorch weights (`best_model.pth`, ~1.03 GB) are hosted externally:

- 💾 **Pre-trained AST Weights**: [Download from Google Drive](https://drive.google.com/file/d/1bGMZYr1OWizGqQ8wR3eye2rzB9Kv8vCc/view?usp=drive_link)

Place downloaded weights into `ast/model_checkpoints/best_model.pth` and `demo/models/best_model.pth` before running inference offline.

---

## 📖 Citation & References

If you find this project or codebase helpful in your research or portfolio evaluation, please consider citing:

```bibtex
@inproceedings{gong2021ast,
  title={AST: Audio Spectrogram Transformer},
  author={Yuan Gong and Yu-An Chung adapter and James Glass},
  booktitle={Proc. Interspeech 2021},
  pages={571--575},
  year={2021}
}
```

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more details.
