import os
import csv
import json
import requests
import urllib.request
from tqdm import tqdm
import yt_dlp
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import shutil

# Target categories and their exact AudioSet MIDs and names
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

# Reverse mapping for quick lookups
mid_to_group = {}
mid_to_name = {}
for group_name, group_info in TARGET_GROUPS.items():
    for mid, name in zip(group_info["mids"], group_info["names"]):
        mid_to_group[mid] = group_name
        mid_to_name[mid] = name

# Metadata download URLs
METADATA_URLS = {
    "class_labels_indices.csv": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
    "balanced_train_segments.csv": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv",
    "eval_segments.csv": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv",
    "unbalanced_train_segments.csv": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv"
}

def download_metadata_file(filename, url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    filepath = os.path.join(dest_dir, filename)
    if os.path.exists(filepath):
        print(f"Metadata file {filename} already exists at {filepath}.")
        return filepath

    print(f"Downloading {filename} from {url}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            total_size = int(response.info().get('Content-Length', 0))
            block_size = 1024 * 1024  # 1MB
            
            with open(filepath, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=filename
            ) as pbar:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    pbar.update(len(buffer))
        print(f"Successfully downloaded {filename}.")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
    return filepath

def parse_audioset_csv(file_path):
    """
    Parses AudioSet segment CSV files, skipping header lines starting with #.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return []

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # Gather non-comment lines
        lines = [line for line in f if not line.startswith('#')]
        
        reader = csv.reader(lines, skipinitialspace=True)
        for row in reader:
            if len(row) >= 4:
                ytid = row[0]
                start_seconds = float(row[1])
                end_seconds = float(row[2])
                # Labels are comma-separated and might contain double quotes
                labels = [label.strip().replace('"', '') for label in row[3:]]
                
                # Split any internal commas if present
                labels_list = []
                for label in labels:
                    labels_list.extend([l.strip() for l in label.split(',') if l.strip()])
                
                data.append({
                    'ytid': ytid,
                    'start_seconds': start_seconds,
                    'end_seconds': end_seconds,
                    'labels': labels_list
                })
    return data

def filter_dataset(segments):
    """
    Filters segments to find those containing target MIDs.
    """
    matched = []
    for seg in segments:
        matching_labels = [label for label in seg['labels'] if label in mid_to_group]
        if matching_labels:
            # We still need a primary label/group for downloading folder
            first_label = matching_labels[0]
            seg_copy = seg.copy()
            seg_copy['matched_label'] = first_label
            seg_copy['matched_name'] = mid_to_name[first_label]
            seg_copy['matched_group'] = mid_to_group[first_label]
            # Maintain the list of all matched target groups
            seg_copy['matched_groups'] = list(set([mid_to_group[label] for label in matching_labels]))
            matched.append(seg_copy)
    return matched

def get_ffmpeg_path():
    # Use local ffmpeg.exe if copied
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe")
    if os.path.exists(local_path):
        return local_path
    return "ffmpeg"

def download_and_trim_clip(ytid, start_seconds, end_seconds, output_path, shutdown_event):
    """
    Gets the stream URL via yt-dlp and extracts the segment using ffmpeg.
    """
    import random
    
    # 1. Introduce random delay to prevent triggering rate limits, checking shutdown frequently
    sleep_time = random.uniform(3.0, 6.5)
    steps = int(sleep_time / 0.5)
    for _ in range(steps):
        if shutdown_event.is_set():
            return False, "Cancelled by user"
        time.sleep(0.5)
        
    if shutdown_event.is_set():
        return False, "Cancelled by user"
    
    # Get audio stream URL with standard browser headers
    ydl_opts = {
        'format': 'bestaudio/best',
        'js_runtimes': {'node': {}},
        'remote_components': ['ejs:github'],
        'quiet': True,
        'no_warnings': True,
        'no_color': True,
        'socket_timeout': 15,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Sec-Fetch-Mode': 'navigate',
        }
    }
    
    # Check if cookies.txt exists in the script directory to bypass YouTube bot detection
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cookie_path = os.path.join(script_dir, "cookies.txt")
    if os.path.exists(cookie_path):
        ydl_opts['cookiefile'] = cookie_path
    
    stream_url = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={ytid}", download=False)
            stream_url = info.get('url')
    except Exception as e:
        err_str = str(e)
        # If rate limited (bot challenge), perform cooldown sleep
        if "not a bot" in err_str.lower() or "confirm you're not a bot" in err_str.lower():
            print(f"\n[Rate Limit] YouTube bot check detected for {ytid}. Cooling down thread for 30 seconds...")
            time.sleep(30)
        return False, f"yt-dlp error: {err_str[:100]}"

    if not stream_url:
        return False, "Could not retrieve audio stream URL"

    # 2. Extract and format using ffmpeg
    ffmpeg_bin = get_ffmpeg_path()
    duration = end_seconds - start_seconds
    
    # We do input seek (-ss BEFORE -i) for speed.
    cmd = [
        ffmpeg_bin,
        "-y",                   # Overwrite without asking
        "-i", stream_url,
        "-ss", str(start_seconds),
        "-t", str(duration),
        "-ar", "16000",         # AST standard: 16kHz
        "-ac", "1",             # AST standard: mono
        output_path
    ]
    
    try:
        # Run with timeout to prevent ffmpeg from hanging on slow connections
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            timeout=25, 
            check=True
        )
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
            if os.path.exists(output_path):
                os.remove(output_path)
            return False, "FFmpeg produced empty audio"
        return True, "Success"
    except subprocess.TimeoutExpired:
        return False, "FFmpeg timeout expired"
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode('utf-8', errors='ignore').strip()
        # Keep only the last line or brief summary
        last_line = err_msg.split('\n')[-1] if '\n' in err_msg else err_msg
        return False, f"FFmpeg error: {last_line}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def process_download_job(job, download_dir, metadata_csv_path, lock, shutdown_event):
    if shutdown_event.is_set():
        return {'ytid': job['ytid'], 'status': 'Cancelled', 'message': 'Shutdown requested'}
        
    ytid = job['ytid']
    start_seconds = job['start_seconds']
    end_seconds = job['end_seconds']
    group = job['matched_group']
    class_name = job['matched_name']
    
    group_dir = os.path.join(download_dir, group)
    os.makedirs(group_dir, exist_ok=True)
    
    filename = f"{ytid}_{int(start_seconds)}.wav"
    output_path = os.path.join(group_dir, filename)
    
    # If already downloaded, skip
    if os.path.exists(output_path):
        return {'ytid': ytid, 'status': 'Already Downloaded', 'message': 'File exists'}

    if shutdown_event.is_set():
        return {'ytid': ytid, 'status': 'Cancelled', 'message': 'Shutdown requested'}

    success, msg = download_and_trim_clip(ytid, start_seconds, end_seconds, output_path, shutdown_event)
    
    status_str = "Success" if success else "Failed"
    
    # Calculate binary labels
    labels = job.get('labels', [])
    binary_labels = {g: 0 for g in TARGET_GROUPS.keys()}
    for g, config in TARGET_GROUPS.items():
        if any(label in config['mids'] for label in labels):
            binary_labels[g] = 1
            
    # Append to metadata.csv securely using a thread lock
    with lock:
        with open(metadata_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                filename, 
                ytid, 
                start_seconds, 
                end_seconds, 
                group, 
                class_name,
                binary_labels['Crash_Collision'],
                binary_labels['Siren'],
                binary_labels['Scream_Shout'],
                binary_labels['Gunshot_Explosion'],
                status_str, 
                msg,
                job.get('subset', 'unknown')
            ])
            
    return {'ytid': ytid, 'status': status_str, 'message': msg}

def load_permanent_failures(metadata_csv_path):
    failures = set()
    if not os.path.exists(metadata_csv_path):
        return failures
        
    perm_keywords = [
        "Video unavailable", 
        "no longer available", 
        "account associated with this video has been terminated", 
        "removed for violating", 
        "Private video",
        "This video is not available",
        "Copyright claim"
    ]
    
    try:
        with open(metadata_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return failures
            
            try:
                ytid_idx = header.index("ytid")
                status_idx = header.index("status")
                msg_idx = header.index("message")
            except ValueError:
                return failures
                
            for row in reader:
                if len(row) > max(ytid_idx, status_idx, msg_idx):
                    ytid = row[ytid_idx]
                    status = row[status_idx]
                    msg = row[msg_idx]
                    if status == "Failed":
                        if any(kw in msg for kw in perm_keywords):
                            failures.add(ytid)
    except Exception as e:
        print(f"Error loading permanent failures: {e}")
    return failures

def main():
    print("=" * 60)
    print("              AUDIOSET SAFETY-CRITICAL DOWNLOADER")
    print("=" * 60)
    
    metadata_dir = "./metadata"
    download_dir = "./downloads"
    os.makedirs(download_dir, exist_ok=True)
    
    # Check for cookies.txt
    cookie_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cookies.txt")
    if os.path.exists(cookie_path):
        print("[Status] Found 'cookies.txt'. Custom session cookies will be used to bypass bot checks!")
    else:
        print("[Notice] No 'cookies.txt' found in directory.")
        print("  -> Tip: If downloads fail with 'Sign in to confirm you're not a bot', export your YouTube cookies")
        print("     using a browser extension (like 'Get cookies.txt LOCALLY') to a file named 'cookies.txt'")
        print("     and place it in this folder to allow uninterrupted downloading.")
    
    # 1. Download base metadata
    print("\n--- Step 1: Downloading AudioSet Metadata ---")
    download_metadata_file("class_labels_indices.csv", METADATA_URLS["class_labels_indices.csv"], metadata_dir)
    download_metadata_file("balanced_train_segments.csv", METADATA_URLS["balanced_train_segments.csv"], metadata_dir)
    download_metadata_file("eval_segments.csv", METADATA_URLS["eval_segments.csv"], metadata_dir)
    
    # Ask if user wants to download unbalanced dataset metadata (95MB)
    print("\nDo you want to download and search the unbalanced train set metadata (95MB)?")
    print("This will give you access to thousands of extra samples.")
    choice = input("Enter 'y' to download unbalanced set, or press Enter to skip: ").strip().lower()
    
    files_to_parse = [
        ("balanced_train", os.path.join(metadata_dir, "balanced_train_segments.csv")),
        ("eval", os.path.join(metadata_dir, "eval_segments.csv"))
    ]
    
    if choice == 'y':
        unbal_path = download_metadata_file("unbalanced_train_segments.csv", METADATA_URLS["unbalanced_train_segments.csv"], metadata_dir)
        if unbal_path:
            files_to_parse.append(("unbalanced_train", unbal_path))
            
    # 2. Parse and Filter
    print("\n--- Step 2: Parsing & Filtering Target Sound Events ---")
    all_matched = []
    
    for name, filepath in files_to_parse:
        print(f"Parsing {name} dataset...")
        segments = parse_audioset_csv(filepath)
        matched = filter_dataset(segments)
        print(f"  Found {len(matched)} matching clips in {name}.")
        # Add subset label
        for m in matched:
            m['subset'] = name
        all_matched.extend(matched)
        
    print(f"\nTotal matches found across selected sets: {len(all_matched)}")
    
    # Show stats by group
    stats = {}
    for seg in all_matched:
        group = seg['matched_group']
        stats[group] = stats.get(group, 0) + 1
        
    print("\nSummary of available clips by group:")
    for group, count in stats.items():
        print(f"  - {group}: {count} clips")
        
    if not all_matched:
        print("No matching clips found. Exiting.")
        return

    # 3. Setup download parameters
    print("\n--- Step 3: Configure Downloads ---")
    
    limit_per_group = input("\nLimit downloads per category (e.g. 50, 100, 500) [Default: 10]: ").strip()
    if not limit_per_group:
        limit_per_group = 10
    else:
        limit_per_group = int(limit_per_group)
        
    max_threads = input("Number of parallel download threads [Default: 4]: ").strip()
    if not max_threads:
        max_threads = 4
    else:
        max_threads = int(max_threads)
        
    # Initialize download metadata path
    metadata_csv_path = os.path.join(download_dir, "metadata.csv")

    # Load permanent failures to avoid retrying them
    perm_failures = load_permanent_failures(metadata_csv_path)
    if perm_failures:
        print(f"Loaded {len(perm_failures)} unique permanent video failures to skip.")

    # Pre-count existing successful downloads from metadata.csv (multi-label support)
    success_counts = {group: 0 for group in TARGET_GROUPS.keys()}
    success_filenames = set()
    
    if os.path.exists(metadata_csv_path):
        try:
            with open(metadata_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('status') == 'Success':
                        filename = row.get('filename')
                        success_filenames.add(filename)
                        for group in TARGET_GROUPS.keys():
                            if row.get(group) == '1':
                                success_counts[group] += 1
        except Exception as e:
            print(f"Warning: could not read success counts from metadata.csv: {e}")
            
    # Physically check folders for any additional files not recorded as Success in metadata.csv
    print("\nPre-counting existing downloads:")
    for group in TARGET_GROUPS.keys():
        group_dir = os.path.join(download_dir, group)
        if os.path.exists(group_dir):
            existing_files = [f for f in os.listdir(group_dir) if f.endswith('.wav')]
            for f in existing_files:
                if f not in success_filenames:
                    success_filenames.add(f)
                    success_counts[group] += 1
                    
        print(f"  - {group}: {success_counts[group]} files already downloaded successfully.")
        
    # Greedy set-cover job selection
    current_counts = success_counts.copy()
    jobs_to_run = []
    
    for seg in all_matched:
        ytid = seg['ytid']
        start_seconds = seg['start_seconds']
        filename = f"{ytid}_{int(start_seconds)}.wav"
        
        # Check if already downloaded
        if filename in success_filenames:
            continue
            
        # Check if it was a permanent failure
        if ytid in perm_failures:
            continue
            
        # Determine all target groups this segment belongs to
        seg_groups = []
        for group, config in TARGET_GROUPS.items():
            if any(label in config['mids'] for label in seg['labels']):
                seg_groups.append(group)
                
        if not seg_groups:
            continue
            
        # Check if we need this segment for at least one of its groups
        needed = False
        for group in seg_groups:
            if current_counts[group] < limit_per_group:
                needed = True
                break
                
        if not needed:
            continue
            
        # Otherwise, select it!
        for group in seg_groups:
            current_counts[group] += 1
        jobs_to_run.append(seg)
            
    # Print selection stats
    selected_by_group = {group: 0 for group in TARGET_GROUPS.keys()}
    for job in jobs_to_run:
        for group, config in TARGET_GROUPS.items():
            if any(label in config['mids'] for label in job['labels']):
                selected_by_group[group] += 1
                
    for group in TARGET_GROUPS.keys():
        needed = max(0, limit_per_group - success_counts[group])
        print(f"Selected {selected_by_group[group]} new clips matching {group} (target new needed: {needed}).")
        
    print(f"\nTotal selected new clips to download: {len(jobs_to_run)}")
    
    if len(jobs_to_run) == 0:
        print("No new clips need to be downloaded to reach the target limit. Exiting.")
        return
        
    confirm = input("Start downloading? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Download aborted.")
        return
        
    # Initialize metadata.csv
    metadata_csv_path = os.path.join(download_dir, "metadata.csv")
    write_header = not os.path.exists(metadata_csv_path)
    
    with open(metadata_csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["filename", "ytid", "start_seconds", "end_seconds", "primary_group", "class_name", "Crash_Collision", "Siren", "Scream_Shout", "Gunshot_Explosion", "status", "message", "subset"])
            
    import threading
    lock = threading.Lock()
    
    shutdown_event = threading.Event()
    
    # 4. Parallel Download
    print(f"\nStarting parallel downloads using {max_threads} threads...")
    print("Press Ctrl+C at any time to stop the process and save progress gracefully.\n")
    success_count = 0
    fail_count = 0
    already_downloaded = 0
    cancelled_count = 0
    
    start_time = time.time()
    
    try:
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {
                executor.submit(process_download_job, job, download_dir, metadata_csv_path, lock, shutdown_event): job
                for job in jobs_to_run
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading clips"):
                if shutdown_event.is_set():
                    cancelled_count += 1
                    continue
                try:
                    result = future.result()
                    status = result['status']
                    if status == "Success":
                        success_count += 1
                    elif status == "Already Downloaded":
                        already_downloaded += 1
                    elif status == "Cancelled":
                        cancelled_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    print(f"\nJob exception for {futures[future]['ytid']}: {e}")
                    fail_count += 1
    except KeyboardInterrupt:
        print("\n\n[Interrupt] Ctrl+C detected. Cancelling remaining downloads... Please wait for active threads to finish.")
        shutdown_event.set()
        # Cancel any pending futures that haven't run yet
        for future in futures:
            future.cancel()
            
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("                          DOWNLOAD REPORT")
    print("=" * 60)
    print(f"Total time elapsed: {elapsed:.2f} seconds")
    print(f"Successfully downloaded: {success_count} clips")
    print(f"Already downloaded/skipped: {already_downloaded} clips")
    print(f"Failed downloads (deleted/geoblocked): {fail_count} clips")
    if cancelled_count > 0:
        print(f"Cancelled by user: {cancelled_count} clips")
    print(f"Metadata summary written to: {os.path.abspath(metadata_csv_path)}")
    print(f"Audio files saved in directory: {os.path.abspath(download_dir)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
