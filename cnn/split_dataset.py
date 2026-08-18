import os
import shutil
from tqdm import tqdm

SOURCE_DIR = "data/mels-urbansound8k"
DEST_DIR = "data/mels-urbansound8k-split"

# UrbanSound8K prevents data leakage by grouping slices of the same original
# recording into the same fold. A random 80-10-10 split across all files 
# would cause leakage.
# To achieve an 80-10-10 split without leakage, we split by folds:
# Train: Folds 1 to 8 (80%)
# Val:   Fold 9 (10%)
# Test:  Fold 10 (10%)

SPLIT_MAP = {
    "train": ["fold1", "fold2", "fold3", "fold4", "fold5", "fold6", "fold7", "fold8"],
    "val":   ["fold9"],
    "test":  ["fold10"]
}

def main():
    print("=========================================================")
    print("  Splitting UrbanSound8K Dataset (Avoiding Data Leakage) ")
    print("=========================================================")
    print(f"Source:      {SOURCE_DIR}")
    print(f"Destination: {DEST_DIR}\n")
    
    if os.path.exists(DEST_DIR):
        print(f"Cleaning up existing directory: {DEST_DIR}")
        shutil.rmtree(DEST_DIR)
        
    for split_name, folds in SPLIT_MAP.items():
        print(f"Processing '{split_name}' set ({len(folds)} folds)...")
        
        for fold in tqdm(folds, desc=f"  Copying to {split_name}", leave=False):
            fold_path = os.path.join(SOURCE_DIR, fold)
            if not os.path.exists(fold_path):
                print(f"  [WARN] Fold directory not found: {fold_path}")
                continue
                
            classes = os.listdir(fold_path)
            for cls in classes:
                cls_src = os.path.join(fold_path, cls)
                if not os.path.isdir(cls_src):
                    continue
                    
                cls_dest = os.path.join(DEST_DIR, split_name, cls)
                os.makedirs(cls_dest, exist_ok=True)
                
                files = [f for f in os.listdir(cls_src) if f.endswith('.png')]
                for f in files:
                    src_file = os.path.join(cls_src, f)
                    dest_file = os.path.join(cls_dest, f)
                    shutil.copy2(src_file, dest_file)
                    
        print(f"  -> Finished {split_name} set.")

    print("\nVerifying splits...")
    for split_name in ["train", "val", "test"]:
        split_dir = os.path.join(DEST_DIR, split_name)
        if not os.path.exists(split_dir):
            continue
        total_files = sum(len(files) for _, _, files in os.walk(split_dir) if files)
        print(f"  {split_name.ljust(5)}: {total_files} images")

    print("\nDataset split complete. Ready for training!")

if __name__ == "__main__":
    main()
