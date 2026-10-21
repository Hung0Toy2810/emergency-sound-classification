import os
import csv

# Define target groups configuration
TARGET_GROUPS = {
    "Crash_Collision": {
        "mids": ["/m/07pjjrj", "/m/07pc8lb", "/m/07rknqz", "/m/0h9mv"],
        "names": ["Smash, crash", "Breaking", "Skidding", "Tire squeal"]
    },
    "Siren": {
        "mids": ["/m/03kmc9", "/m/04qvtq", "/m/012n7d", "/m/012ndj", "/m/0dgbq"],
        "names": ["Siren", "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)", "Civil defense siren"]
    },
    "Scream_Shout": {
        "mids": ["/m/03qc9zr", "/m/07p6fty", "/m/07sr1lc", "/t/dd00135"],
        "names": ["Screaming", "Shout", "Yell", "Children shouting"]
    },
    "Gunshot_Explosion": {
        "mids": ["/m/032s66", "/m/014zdl", "/m/07qqyl4"],
        "names": ["Gunshot, gunfire", "Explosion", "Boom"]
    }
}

def parse_audioset_csv(file_path, subset):
    if not os.path.exists(file_path):
        print(f"Metadata file {file_path} does not exist. Skipping.")
        return {}
    
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f if not line.startswith('#')]
        reader = csv.reader(lines, skipinitialspace=True)
        for row in reader:
            if len(row) >= 4:
                ytid = row[0]
                try:
                    start_seconds = float(row[1])
                except ValueError:
                    continue
                # Labels are comma-separated and might contain double quotes
                labels = [label.strip().replace('"', '') for label in row[3:]]
                
                # Split any internal commas if present
                labels_list = []
                for label in labels:
                    labels_list.extend([l.strip() for l in label.split(',') if l.strip()])
                data[(ytid, start_seconds)] = (labels_list, subset)
    return data

def main():
    metadata_dir = "./metadata"
    download_dir = "./downloads"
    
    # Use metadata_backup.csv if it exists to restore original data, otherwise metadata.csv
    backup_path = os.path.join(download_dir, "metadata_backup.csv")
    old_metadata_path = os.path.join(download_dir, "metadata.csv")
    
    if os.path.exists(backup_path):
        print("Found backup metadata file, using it as source...")
        source_path = backup_path
    elif os.path.exists(old_metadata_path):
        print("Using metadata.csv as source...")
        source_path = old_metadata_path
    else:
        print("No metadata source file found to convert.")
        return

    new_metadata_path = os.path.join(download_dir, "metadata_multilabel.csv")

    # 1. Load all segment mappings from AudioSet CSV files
    print("Loading AudioSet segment metadata...")
    audioset_data = {}
    for filename in ["balanced_train_segments.csv", "eval_segments.csv", "unbalanced_train_segments.csv"]:
        filepath = os.path.join(metadata_dir, filename)
        if os.path.exists(filepath):
            print(f"  Parsing {filename}...")
            subset = filename.split('_segments')[0]
            audioset_data.update(parse_audioset_csv(filepath, subset))
            
    print(f"Loaded {len(audioset_data)} segment mappings from AudioSet metadata.")

    # 2. Open metadata source and write to new file
    print("Converting metadata to multi-label schema...")
    
    success_count = 0
    converted_count = 0
    
    with open(source_path, 'r', encoding='utf-8') as infile, \
         open(new_metadata_path, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Read header
        header = next(reader, None)
        if not header:
            print("Empty source file.")
            return
            
        # Check if already migrated (should not happen if using backup)
        if "Crash_Collision" in header and "Siren" in header:
            print("Source metadata is already in multi-label format! Aborting conversion.")
            outfile.close()
            os.remove(new_metadata_path)
            return

        # Write new header
        new_header = [
            "filename", "ytid", "start_seconds", "end_seconds", 
            "primary_group", "class_name", 
            "Crash_Collision", "Siren", "Scream_Shout", "Gunshot_Explosion", 
            "status", "message", "subset"
        ]
        writer.writerow(new_header)
        
        for row in reader:
            # Format in old metadata: filename, ytid, start_seconds, end_seconds, group, class_name, status, message
            if len(row) < 8:
                continue
            
            filename = row[0]
            ytid = row[1]
            try:
                start_seconds = float(row[2])
                end_seconds = float(row[3])
            except ValueError:
                continue
            
            primary_group = row[4]
            class_name = row[5]
            status = row[6]
            message = row[7]
            
            # Lookup original labels and subset
            mapping = audioset_data.get((ytid, start_seconds))
            if mapping:
                labels, subset = mapping
            else:
                labels, subset = [], "unknown"
            
            binary_labels = {group: 0 for group in TARGET_GROUPS.keys()}
            
            if labels:
                for group, config in TARGET_GROUPS.items():
                    if any(label in config['mids'] for label in labels):
                        binary_labels[group] = 1
            else:
                # Fallback: if we can't find it in original AudioSet metadata, use its primary group
                if primary_group in binary_labels:
                    binary_labels[primary_group] = 1
            
            writer.writerow([
                filename,
                ytid,
                start_seconds,
                end_seconds,
                primary_group,
                class_name,
                binary_labels['Crash_Collision'],
                binary_labels['Siren'],
                binary_labels['Scream_Shout'],
                binary_labels['Gunshot_Explosion'],
                status,
                message,
                subset
            ])
            converted_count += 1
            if status == "Success":
                success_count += 1

    # Overwrite the old metadata file and keep backup if we used it
    if source_path == old_metadata_path:
        os.rename(old_metadata_path, backup_path)
    else:
        # If we used backup, we can safely overwrite the metadata.csv
        if os.path.exists(old_metadata_path):
            os.remove(old_metadata_path)
            
    os.rename(new_metadata_path, old_metadata_path)
    
    print(f"\nMigration completed successfully!")
    print(f"  - Converted rows: {converted_count}")
    print(f"  - Successful downloads: {success_count}")
    print(f"  - Source file: {source_path}")
    print(f"  - Destination file: {old_metadata_path}")

if __name__ == "__main__":
    main()
