import os
import csv
import shutil

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_files_dir = os.path.abspath(os.path.join(base_dir, "..", "dataset_files"))
metadata_csv = os.path.join(dataset_files_dir, "metadata.csv")

categories = ["Crash_Collision", "Siren", "Scream_Shout", "Gunshot_Explosion"]

# Define paths for train and test splits
train_dir = os.path.join(dataset_files_dir, "train")
test_dir = os.path.join(dataset_files_dir, "test")

# Create subdirectories
for cat in categories:
    os.makedirs(os.path.join(train_dir, cat), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cat), exist_ok=True)

if not os.path.exists(metadata_csv):
    print(f"Error: {metadata_csv} does not exist!")
    exit()

print("Reorganizing files and creating split metadata...")

# Headers
headers = None
train_rows = []
test_rows = []

files_moved = 0
files_not_found = 0

with open(metadata_csv, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    headers = next(reader)
    
    # Indexes of columns
    filename_idx = headers.index("filename")
    group_idx = headers.index("primary_group")
    status_idx = headers.index("status")
    subset_idx = headers.index("subset")
    
    for row in reader:
        if len(row) < len(headers):
            continue
            
        filename = row[filename_idx]
        group = row[group_idx]
        status = row[status_idx]
        subset = row[subset_idx]
        
        # Only process successfully downloaded files
        if status == "Success":
            # Current location of the wav file
            src_path = os.path.join(dataset_files_dir, group, filename)
            
            if os.path.exists(src_path):
                # Determine target split directory
                split_name = "test" if subset == "eval" else "train"
                dest_dir = os.path.join(dataset_files_dir, split_name, group)
                dest_path = os.path.join(dest_dir, filename)
                
                # Move physical file
                shutil.move(src_path, dest_path)
                files_moved += 1
                
                # Add to split-specific row list
                if split_name == "train":
                    train_rows.append(row)
                else:
                    test_rows.append(row)
            else:
                files_not_found += 1

# Write split metadata CSV files
train_meta_path = os.path.join(train_dir, "metadata.csv")
test_meta_path = os.path.join(test_dir, "metadata.csv")

with open(train_meta_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(train_rows)

with open(test_meta_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(test_rows)

# Clean up empty original category folders
for cat in categories:
    cat_path = os.path.join(dataset_files_dir, cat)
    if os.path.exists(cat_path) and not os.listdir(cat_path):
        os.rmdir(cat_path)
        print(f"Removed empty directory: dataset_files/{cat}")

print("\n--- Reorganization Summary ---")
print(f"Total successful files moved: {files_moved}")
if files_not_found > 0:
    print(f"Files marked as Success but not found on disk: {files_not_found}")
print(f"Train metadata created at: {train_meta_path} ({len(train_rows)} samples)")
print(f"Test metadata created at: {test_meta_path} ({len(test_rows)} samples)")
