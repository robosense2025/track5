#!/usr/bin/env python3
"""
download_track5.py - Improved version with better rate limit handling
"""

import argparse
from huggingface_hub import snapshot_download, login, HfFileSystem
from huggingface_hub.utils import LocalEntryNotFoundError, HfHubHTTPError
import time
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the track5-cross-platform-3d-object-detection subfolder "
                    "from the robosense/datasets Hugging Face repo."
    )
    parser.add_argument(
        "output_dir",
        help="Local directory where files will be saved"
    )
    parser.add_argument(
        "--repo-id",
        default="robosense/datasets",
        help="Hugging Face dataset repo id (default: %(default)s)"
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch or commit to download from (default: %(default)s)"
    )
    parser.add_argument(
        "--subfolder",
        default="track5-cross-platform-3d-object-detection",
        help="Subfolder in the repo to download (default: %(default)s)"
    )
    parser.add_argument(
        "--token",
        help="Hugging Face authentication token (get from https://huggingface.co/settings/tokens)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing directory before download"
    )
    return parser.parse_args()

def check_repo_access(repo_id, token=None, max_retries=3):
    fs = HfFileSystem(token=token)
    for attempt in range(max_retries):
        try:
            # Just try listing the root to check access
            fs.ls(repo_id, maxdepth=1)
            return True
        except HfHubHTTPError as e:
            if e.response.status_code == 429:
                wait = min(60, (attempt + 1) * 20)
                print(f"Rate limited. Waiting {wait} seconds before checking access again...")
                time.sleep(wait)
            else:
                raise
    return False

def robust_download(repo_id, revision, output_dir, subfolder="", token=None, clean=False, max_tries=5):
    if clean and os.path.exists(output_dir):
        print(f"Cleaning existing directory: {output_dir}")
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    if not check_repo_access(repo_id, token):
        raise RuntimeError("Could not establish connection to Hugging Face Hub after multiple attempts")

    patterns = [f"{subfolder}/*"] if subfolder else ["*"]
    
    for attempt in range(1, max_tries + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                local_dir=output_dir,
                allow_patterns=patterns,
                resume_download=True,
                token=token,
                max_workers=4,  # Reduced to be gentler on servers
                local_dir_use_symlinks="auto",
                ignore_patterns=["*.md", "*.txt"],  # Skip documentation files if they cause issues
            )
            return True
        except LocalEntryNotFoundError as e:
            print(f"[Attempt {attempt}/{max_tries}] Cache error: {e}")
            if attempt == max_tries:
                raise
            time.sleep(10 * attempt)
        except HfHubHTTPError as e:
            if e.response.status_code == 429:
                wait = min(300, 30 * (attempt + 1))
                print(f"[Attempt {attempt}/{max_tries}] Rate limited. Waiting {wait} seconds...")
                time.sleep(wait)
            else:
                print(f"[Attempt {attempt}/{max_tries}] HTTP Error: {e}")
                if attempt == max_tries:
                    raise
                time.sleep(20)
        except Exception as e:
            print(f"[Attempt {attempt}/{max_tries}] Unexpected error: {e}")
            if attempt == max_tries:
                raise
            time.sleep(30)
    return False

def main():
    args = parse_args()

    print(f"Downloading '{args.subfolder}' from '{args.repo_id}@{args.revision}'...")
    print(f"Target directory: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        success = robust_download(
            repo_id=args.repo_id,
            revision=args.revision,
            output_dir=args.output_dir,
            subfolder=args.subfolder,
            token=args.token,
            clean=args.clean
        )
        
        if success:
            print("Download completed successfully.")
            sys.exit(0)
        else:
            print("Download failed after multiple attempts.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()