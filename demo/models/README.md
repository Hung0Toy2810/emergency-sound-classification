# Model Files

This directory contains the trained model weights for the Sound Classification Demo.

## Required Models

Due to GitHub file size limitations, model files are not included in this repository.

### Expected Files:

1. **best_model.pth** (~988 MB)
   - Audio Spectrogram Transformer (AST) model
   - Fine-tuned for 4-class sound classification

2. **custom_cnn_best.pth** (~19 MB)
   - Custom CNN model
   - 5-layer convolutional architecture

## Download Instructions

Please download the model files from one of the following sources:

- [Google Drive Link] - *Add your link here*
- [OneDrive Link] - *Add your link here*
- [Hugging Face] - *Add your link here*

## Placement

After downloading, place the model files in this directory:
```
models/
├── best_model.pth
├── custom_cnn_best.pth
└── README.md
```

## Training Your Own Models

If you want to train your own models, please refer to the training scripts in the training documentation.

### CNN Model Architecture:
- Input: Mel-spectrogram (224×224)
- Layers: 5 Conv blocks (32→64→128→256→512)
- Output: 4 classes

### AST Model:
- Base: MIT/ast-finetuned-audioset-10-10-0.4593
- Fine-tuned on ESC-50 subset
- Input: Log-Mel spectrogram (128×T)
- Output: 4 classes (multi-label)
