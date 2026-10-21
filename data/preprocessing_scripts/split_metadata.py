import os
import csv

base_dir = os.path.dirname(os.path.abspath(__file__))
metadata_dir = os.path.abspath(os.path.join(base_dir, "..", "raw_segments"))
downloads_dir = os.path.abspath(os.path.join(base_dir, "..", "dataset_files"))
metadata_csv = os.path.join(downloads_dir, "metadata.csv")
temp_csv = os.path.join(downloads_dir, "metadata_temp.csv")

def parse_set(filename):
    filepath = os.path.join(metadata_dir, filename)
    ids = set()
    if not os.path.exists(filepath):
        print(f"Warning: {filename} not found in {metadata_dir}.")
        return ids
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                ids.add((parts[0].strip(), float(parts[1].strip())))
    return ids

print("Loading AudioSet split mappings...")
balanced_ids = parse_set("balanced_train_segments.csv")
eval_ids = parse_set("eval_segments.csv")
unbalanced_ids = parse_set("unbalanced_train_segments.csv")

if not os.path.exists(metadata_csv):
    print(f"Error: {metadata_csv} does not exist!")
    exit()

print("Updating metadata.csv to add 'subset' column...")
rows_updated = 0
success_counts = {"balanced_train": 0, "unbalanced_train": 0, "eval": 0, "unknown": 0}

with open(metadata_csv, 'r', encoding='utf-8') as infile, \
     open(temp_csv, 'w', newline='', encoding='utf-8') as outfile:
     
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    header = next(reader, None)
    if header:
        # Check if 'subset' is already in header
        if "subset" not in header:
            header.append("subset")
        writer.writerow(header)
        
        subset_idx = header.index("subset") if "subset" in header else len(header) - 1
        
        for row in reader:
            if len(row) < 4:
                continue
            ytid = row[1]
            try:
                start_seconds = float(row[2])
            except ValueError:
                continue
                
            key = (ytid, start_seconds)
            
            # Determine subset
            if key in eval_ids:
                subset = "eval"
            elif key in balanced_ids:
                subset = "balanced_train"
            elif key in unbalanced_ids:
                subset = "unbalanced_train"
            else:
                subset = "unknown"
                
            # Append or update subset column
            if len(row) < len(header):
                row.append(subset)
            else:
                row[subset_idx] = subset
                
            writer.writerow(row)
            rows_updated += 1
            if row[10] == "Success": # status column
                success_counts[subset] += 1

# Replace old metadata with the updated one
if os.path.exists(metadata_csv):
    backup_path = os.path.join(downloads_dir, "metadata_before_split.csv")
    if os.path.exists(backup_path):
        os.remove(backup_path)
    os.rename(metadata_csv, backup_path)
os.rename(temp_csv, metadata_csv)

print(f"\nSuccessfully updated {rows_updated} rows in metadata.csv.")
print(f"A backup of the old file has been saved to: dataset_files/metadata_before_split.csv")
print("\nSuccessful downloads summary:")
for k, v in success_counts.items():
    print(f"  - {k}: {v} files")
