import os
import sys
import tarfile
import subprocess
import time

def main():
    url = "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz"
    dest_dir = "data"
    os.makedirs(dest_dir, exist_ok=True)
    tar_path = os.path.join(dest_dir, "UrbanSound8K.tar.gz")
    
    print(f"Downloading UrbanSound8K dataset from {url}...")
    
    max_retries = 20
    download_success = False
    
    for attempt in range(max_retries):
        try:
            # -C - tells curl to automatically resume the transfer from where it left off
            # -L follows redirects
            subprocess.run(["curl", "-C", "-", "-L", "-o", tar_path, url], check=True)
            download_success = True
            print("\nDownload completed successfully.")
            break
        except subprocess.CalledProcessError as e:
            print(f"\nDownload interrupted (curl exited with code {e.returncode}).")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds... (Attempt {attempt+2}/{max_retries})")
                time.sleep(5)
            else:
                print("Max retries reached. Download failed.")
                sys.exit(1)
        
    if not download_success:
        sys.exit(1)

    print("Extracting dataset (this may take a few minutes)...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=dest_dir)
        print("Extraction completed successfully.")
    except Exception as e:
        print(f"Error extracting archive (file might be corrupted): {e}")
        sys.exit(1)
        
    print("Cleaning up compressed file...")
    try:
        os.remove(tar_path)
        print("Cleanup completed.")
    except Exception as e:
        print(f"Error removing archive: {e}")

    print("Dataset is ready in the 'data' directory.")

if __name__ == "__main__":
    main()
