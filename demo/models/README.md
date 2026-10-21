# Demo checkpoints

## SoundCNN

The retained four-class checkpoint is tracked at
`../../cnn/models/custom_cnn_best.pth`. The demo uses it automatically unless
`custom_cnn_best.pth` is placed in this directory as a local override.

The model has five convolution blocks with widths 32, 64, 128, 256 and 512,
uses 224 x 224 RGB mel images, and returns the class order stored in the
checkpoint. See [the CNN documentation](../../cnn/README.md).

## Audio Spectrogram Transformer

Place `best_model.pth` in this directory. It is downloaded separately using
the link in the [main README](../../README.md#pre-trained-checkpoints).

The implementation adapts `google/vit-base-patch16-224` pretrained on ImageNet
to single-channel log-Mel spectrograms. It uses four multi-label outputs:
Crash_Collision, Siren, Scream_Shout, Gunshot_Explosion.
