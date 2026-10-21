import argparse
import os
from multiprocessing import Pool, cpu_count

import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

SR = 22050
DURATION = 4.0
SAMPLES = int(SR * DURATION)
N_MELS = 128
IMG_SIZE = 224
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def fixed_length(y):
    if len(y) < SAMPLES:
        return np.pad(y, (0, SAMPLES - len(y)))
    return y[:SAMPLES]


def save_mel(audio_path, output_path):
    import librosa
    import librosa.display

    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    y = fixed_length(y)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=N_MELS,
        n_fft=2048,
        hop_length=512,
        fmin=0,
        fmax=SR // 2,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig = plt.figure(figsize=(IMG_SIZE / 100, IMG_SIZE / 100), dpi=100)
    ax = Axes(fig, (0.0, 0.0, 1.0, 1.0))
    ax.set_axis_off()
    fig.add_axes(ax)
    librosa.display.specshow(mel_db, sr=SR, hop_length=512, fmin=0, fmax=SR // 2, ax=ax)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def collect_tasks(input_dir, output_dir):
    tasks = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in AUDIO_EXTS:
                continue
            audio_path = os.path.join(root, name)
            rel_path = os.path.relpath(audio_path, input_dir)
            output_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".png")
            if not os.path.exists(output_path):
                tasks.append((audio_path, output_path))
    return tasks


def process(task):
    audio_path, output_path = task
    try:
        save_mel(audio_path, output_path)
    except Exception as exc:
        return f"{audio_path}: {exc}"
    return None


def main():
    parser = argparse.ArgumentParser(description="Convert audio folders to mel-spectrogram images.")
    parser.add_argument("--input", default="data/metadata", help="Audio root: split/class/file.wav")
    parser.add_argument("--output", default="data/mels", help="Output root for mel images")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = parser.parse_args()

    tasks = collect_tasks(args.input, args.output)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Files to process: {len(tasks)}")
    if not tasks:
        return

    with Pool(args.workers) as pool:
        errors = [err for err in tqdm(pool.imap_unordered(process, tasks), total=len(tasks)) if err]

    if errors:
        print(f"Finished with {len(errors)} errors.")
        for err in errors[:10]:
            print(f"- {err}")
    else:
        print("Preprocess finished successfully.")


if __name__ == "__main__":
    main()
