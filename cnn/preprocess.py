import os
import csv
import warnings
import matplotlib
# Headless mode MUST be set before importing pyplot
matplotlib.use('Agg')

import librosa  # type: ignore
import librosa.display  # type: ignore
import matplotlib.pyplot as plt
import noisereduce as nr  # type: ignore
import sklearn.preprocessing
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")

# ======================== CONFIG ========================
DATA_DIR = "data/"
CSV_PATH = os.path.join(DATA_DIR, "UrbanSound8K.csv")
OUTPUT_BASE_PATH = os.path.join(DATA_DIR, "mels-urbansound8k/")

SR = 22050              # 22050 Hz is sufficient for urban sounds (Nyquist = 11025 Hz)
FIXED_DURATION = 4.0    # Pad/trim all clips to exactly 4 seconds for consistency
FIXED_SAMPLES = int(SR * FIXED_DURATION)

# Noise reduction: set to False by default (see docstring below for reasoning)
# UrbanSound8K clips are very short (<= 4s), many lack a "quiet" segment for
# spectral gating to estimate noise profile. Classes like air_conditioner and
# engine_idling have spectra similar to stationary noise and may be suppressed.
# Recommend running an ablation study (with vs without) on 1 fold before enabling.
APPLY_NOISE_REDUCTION = False

# UrbanSound8K Class Mapping (for reference/logging only — labels come from CSV)
CLASS_NAMES = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music"
}
# ========================================================


def load_metadata(csv_path):
    """
    Load UrbanSound8K.csv and return a dict mapping:
        filename -> { 'fold': int, 'classID': int, 'class': str }
    This is the ground truth — we use this instead of parsing filenames
    to guarantee label correctness.
    """
    metadata = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row['slice_file_name']] = {
                'fold': int(row['fold']),
                'classID': int(row['classID']),
                'class': row['class'],
            }
    return metadata


def pad_or_trim(y, target_length):
    """
    Ensure all audio clips have the same length.
    - If shorter: zero-pad at the end (silence padding)
    - If longer: trim from the end
    This is critical because UrbanSound8K clips range from <0.5s to 4s.
    Without this, mel spectrograms have different time dimensions,
    which causes issues for CNN input.
    """
    if len(y) < target_length:
        # Pad with zeros (silence) at the end
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    elif len(y) > target_length:
        y = y[:target_length]
    return y


def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


def save_mel_spec(y, sr, directory, filename_base):
    """
    Generate and save a pure mel-spectrogram image.
    No axes, no margins, no waveform subplots. Just the spectrogram 
    to prevent CNN from learning non-physical artifacts.

    Key fix: uses power_to_db (not amplitude_to_db) because
    librosa.feature.melspectrogram() returns a POWER spectrogram (|S|^2).
    """
    try:
        if len(y) == 0:
            return

        S = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=128,
            fmin=0,        # Keep low frequencies (important for AC, engine)
            fmax=None,     # Auto = Nyquist = SR/2
            n_fft=2048,
            hop_length=512
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        # Create figure with exact dimensions (2.24x2.24 inches @ 100 DPI = 224x224 pixels)
        fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
        
        # Add an axes that fills the entire figure (no margins)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # Draw spectrogram on this axes
        librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=512,
            fmin=0,
            ax=ax
        )

        out_path = os.path.join(directory, f"{filename_base}.png")
        fig.savefig(out_path, dpi=100)
        plt.close(fig)

    except Exception as e:
        print(f"[WARN] save_mel_spec {filename_base}: {e}")


def process_single_file(args):
    """
    Worker function to process a single .wav file.
    Receives a tuple: (file_path, fold, classID, class_name)
    Labels come from CSV metadata, not from filename parsing.
    """
    file_path, fold, class_id, class_name = args

    try:
        filename = os.path.basename(file_path)
        file_id = filename[:-4]  # remove .wav

        # Build output directory: mels-urbansound8k/fold{N}/{classID}_{className}/
        folder_name = f"{class_id}_{class_name}"
        output_dir = os.path.join(OUTPUT_BASE_PATH, f"fold{fold}", folder_name)
        os.makedirs(output_dir, exist_ok=True)

        # Resume check BEFORE expensive loading
        check_file = os.path.join(output_dir, f"{file_id}.png")
        if os.path.exists(check_file):
            return None

        # Load audio
        y, sr = librosa.load(file_path, sr=SR, mono=True)

        # Pad or trim to fixed duration for consistent spectrogram dimensions
        y = pad_or_trim(y, FIXED_SAMPLES)

        # Optional noise reduction
        if APPLY_NOISE_REDUCTION:
            y = nr.reduce_noise(y=y, sr=sr)

        save_mel_spec(y, sr, output_dir, file_id)

    except Exception as e:
        return f"Error processing {file_path}: {e}"
    return None


def main():
    print("=" * 60)
    print("  UrbanSound8K Preprocessing Pipeline")
    print("=" * 60)
    print(f"  SR:                 {SR} Hz")
    print(f"  Fixed duration:     {FIXED_DURATION}s ({FIXED_SAMPLES} samples)")
    print(f"  Noise reduction:    {'ON' if APPLY_NOISE_REDUCTION else 'OFF'}")
    print(f"  dB conversion:      power_to_db (correct for melspectrogram)")
    print(f"  Label source:       {CSV_PATH}")
    print("=" * 60)

    # 1. Load metadata from CSV (ground truth labels)
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV file not found at {CSV_PATH}")
        return

    metadata = load_metadata(CSV_PATH)
    print(f"Loaded metadata for {len(metadata)} files from CSV.")

    # 2. Build task list by cross-referencing CSV with actual .wav files
    task_list = []
    missing_files = []
    orphan_files = []

    # Collect all .wav files on disk
    wav_files_on_disk = {}
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith(".wav"):
                wav_files_on_disk[f] = os.path.join(root, f)

    # Match CSV entries to files on disk
    for filename, info in metadata.items():
        if filename in wav_files_on_disk:
            task_list.append((
                wav_files_on_disk[filename],
                info['fold'],
                info['classID'],
                info['class'],
            ))
        else:
            missing_files.append(filename)

    # Check for .wav files NOT in CSV (orphans)
    csv_filenames = set(metadata.keys())
    for wav_name in wav_files_on_disk:
        if wav_name not in csv_filenames:
            orphan_files.append(wav_name)

    print(f"Matched {len(task_list)} files (CSV ↔ disk).")
    if missing_files:
        print(f"WARNING: {len(missing_files)} files in CSV but NOT found on disk!")
        print(f"  Sample: {missing_files[:3]}")
    if orphan_files:
        print(f"WARNING: {len(orphan_files)} .wav files on disk but NOT in CSV!")
        print(f"  Sample: {orphan_files[:3]}")

    if not task_list:
        print("No files to process. Check your data directory and CSV.")
        return

    # 3. Print class distribution
    class_counts = {}
    for _, _, cid, cname in task_list:
        key = f"{cid}_{cname}"
        class_counts[key] = class_counts.get(key, 0) + 1
    print("\nClass distribution:")
    for key in sorted(class_counts.keys()):
        print(f"  {key}: {class_counts[key]} files")
    print()

    # 4. Parallel processing
    num_workers = cpu_count()
    print(f"Spawning {num_workers} workers for parallel processing...")

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_file, task_list),
            total=len(task_list)
        ))

    # 5. Error reporting
    errors = [r for r in results if r is not None]
    if errors:
        print(f"\nProcessing finished with {len(errors)} errors.")
        for e in errors[:5]:
            print(f"  - {e}")
    else:
        print("\nAll files processed successfully!")

    # 6. Verify output count
    output_count = sum(
        len(files) for _, _, files in os.walk(OUTPUT_BASE_PATH) if files
    )
    print(f"\nTotal output images: {output_count}")
    print(f"Expected:            {len(task_list)}")
    if output_count == len(task_list):
        print("✅ Output count matches expected count.")
    else:
        print("⚠️  Mismatch! Some files may have failed silently.")


if __name__ == "__main__":
    main()
