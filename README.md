# Emergency Sound Classification: CNN vs. Audio Spectrogram Transformer

This repository contains an end-to-end implementation for classifying emergency and dangerous urban sound events using deep learning. We compare a custom 2D Convolutional Neural Network (CNN) against an Audio Spectrogram Transformer (AST) adapted from the [AST paper](https://arxiv.org/abs/2104.01778) (Gong et al., MIT CSAIL, 2021).

The system classifies audio into 4 dangerous event categories:
- **Siren**: Emergency vehicle sirens (police, ambulance, fire truck)
- **Scream / Shout**: Human distress calls and shouting
- **Gunshot / Explosion**: Gunshots and explosion audio impulses
- **Crash / Collision**: Vehicle collisions and metallic/structural impacts

---

## Performance Summary

Evaluated on an independent test set of 447 audio files:

| Model | mAP (%) | Macro F1 | Train Loss | Val Loss |
| :--- | :---: | :---: | :---: | :---: |
| **Audio Spectrogram Transformer (AST)** | **92.02%** | **0.8469** | 0.0353 | 0.2343 |
| **Custom 2D CNN Baseline** | 81.50% | 0.7620 | 0.1204 | 0.4120 |

### Per-Class Average Precision (AST Model)

| Sound Event | Average Precision (AP) |
| :--- | :---: |
| Siren | **98.57%** |
| Scream / Shout | **92.00%** |
| Gunshot / Explosion | **89.71%** |
| Crash / Collision | **87.81%** |

---

## Technical Details

### Audio Spectrogram Transformer Adaptation
- **Backbone**: `google/vit-base-patch16-224` pre-trained on ImageNet.
- **Input Processing**: Audio signals (16 kHz, mono) are converted into 128-bin log-Mel spectrograms.
- **1-Channel Input Adaptation**: The 3-channel RGB patch projection weights of ViT are averaged into a 1-channel weight tensor to process single-channel audio spectrograms.
- **2D Bilinear Positional Embedding Interpolation**: Original ViT 1D positional embeddings ($14 \times 14 = 196$ patches) are bilinearly interpolated into a 2D spatial grid matching the spectrogram size ($12 \times 99 = 1,188$ patches).
- **Data Augmentation**: SpecAugment (frequency and time masking) is applied during training.

### Efficiency Optimizations
- **Offline Spectrogram Caching**: Pre-computes normalized log-Mel spectrograms into PyTorch binary `.pt` files, avoiding real-time STFT calculation during DataLoader execution.
- **AMP FP16**: Uses PyTorch Automatic Mixed Precision to accelerate GPU training and reduce memory usage.

---

## System Pipeline

```text
Raw Audio (.wav, .mp3, .flac)
          │
          ▼
Log-Mel Spectrogram Extraction (128 Mel Bins, 16 kHz)
          │
          ▼
Data Augmentation (SpecAugment: Freq/Time Masking)
          │
     ┌────┴─────────────────────────────┐
     ▼                                  ▼
Audio Spectrogram Transformer      Custom 2D CNN
(ViT Patching + 2D Pos Interpolation) (Conv2D + BatchNorm + GAP)
     │                                  │
     └────┬─────────────────────────────┘
          ▼
Multi-Label Sigmoid Output
          │
          ▼
Flask Web UI (Side-by-side comparison)
```

---

## Directory Structure

```text
.
├── ast/                        # Audio Spectrogram Transformer module
│   ├── src/                    # PyTorch model architecture, dataset, and training script
│   ├── evaluate.py             # Evaluation script for PR curves & metrics
│   ├── predict.py              # CLI script for single-file inference
│   └── run.py                  # Runner for preprocessing and training
├── cnn/                        # Custom 2D CNN baseline implementation
├── data/                       # Data crawling, scraping, and metadata conversion scripts
├── demo/                       # Flask web application for interactive inference
├── docs/                       # Project report and documentation
├── requirements.txt            # Dependency list
└── LICENSE                     # MIT License
```

---

## Setup & Running

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Hung0Toy2810/emergency-sound-classification.git
cd emergency-sound-classification
```

2. Create virtual environment and install dependencies:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Preprocessing & Training

Pre-process audio files to log-Mel `.pt` tensors:
```bash
python ast/run.py --preprocess
```

Train the AST model (10 epochs with AMP FP16):
```bash
python ast/run.py --epochs 10 --batch_size 16 --duration 10
```

Resume training from checkpoint if interrupted:
```bash
python ast/run.py --epochs 10 --batch_size 16 --duration 10 --resume
```

### Evaluation & Inference

Evaluate test set performance:
```bash
python ast/evaluate.py
```

Run CLI inference on a single audio file:
```bash
python ast/predict.py --audio "path/to/audio.wav"
```

### Running the Web Demo

```bash
cd demo
python app.py
```
Open `http://localhost:5000` in your browser to test live inference and compare AST vs CNN results.

---

## Pre-trained Checkpoints

Due to GitHub file size limits, the pre-trained weights file (`best_model.pth`, ~1.03 GB) is hosted externally:
- **Download Checkpoint**: [Google Drive Link](https://drive.google.com/file/d/1bGMZYr1OWizGqQ8wR3eye2rzB9Kv8vCc/view?usp=drive_link)

Place `best_model.pth` inside `ast/model_checkpoints/` and `demo/models/` before running inference offline.

---

## References

- Gong, Y., Chung, Y.-A., & Glass, J. (2021). AST: Audio Spectrogram Transformer. *Proc. Interspeech 2021*, 571-575.
- Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Repository history

This repository preserves the commit history and MIT license of the shared
[Emergency Sound Classification project](https://github.com/giabao1176/Emergency-Sound-Classification).

The AST evaluation report records macro F1 of 0.8469. The current training code
uses the split named `test` as validation data for model selection, so this is
not an independent held-out test score. The README and demo currently report
different CNN scores; a controlled CNN–AST comparison needs a consistent
validation/test protocol.
