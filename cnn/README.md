# Four-class SoundCNN

This is the supplied SoundCNN project corresponding to the CNN architecture
and confusion matrix already used by the web demo. It replaces the previous
UrbanSound8K/GrowableCNN files.

## Model and checkpoint

- Five convolution blocks, with widths 32, 64, 128, 256 and 512.
- Input: RGB mel-spectrogram images, 224 x 224.
- Audio preprocessing: 22,050 Hz, four seconds, 128 mel bins.
- Checkpoint: `models/custom_cnn_best.pth` (18.3 MiB, 4,780,388 parameters).
- Output order: Crash_Collision, Gunshot_Explosion, Scream_Shout, Siren.

The checkpoint loads into `demo/src/cnn_model.py` without remapping weights.
The demo falls back to this checkpoint when `demo/models/` has no CNN weights.

## Run

From this directory, install `requirements.txt` in your Python environment.
Organize input audio as `data/audio/<split>/<class>/<clip>` for the four
classes listed above, where split is `train`, `val` or `test`.

```bash
python preprocess.py --input data/audio --output data/mels
python train_custom_cnn.py --data data/mels --epochs 50
python generate_evaluation_report.py
```

The retained trainer selects checkpoints using `test` if that folder exists,
otherwise `val`. This preserves the supplied training behavior. A split used
for model selection is development data, not an independent final test.
The original evaluation audio is not included in this repository.

## Saved results

`evaluation_report/classification_report.txt` and `confusion_matrix.png` cover
447 clips. The matrix gives accuracy 0.642058 and macro F1 0.658834, consistent
with the saved checkpoint accuracy and the report rounded to two decimals.
These were checked against the supplied artifacts, not recomputed by running
the original audio through the checkpoint.

`evaluator.py` supplies the missing `ModelEvaluator` dependency required by the
training and report scripts; it does not alter the retained checkpoint.

## Verification

From the repository root:

```bash
python -m pytest cnn/tests -q
```
