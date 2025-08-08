#!/usr/bin/env python3
"""
download_track5.py

Usage:
    python download_track5.py /your/target/path
"""
import sys
import argparse
from huggingface_hub import snapshot_download, login
import time
import os
from huggingface_hub.utils import LocalEntryNotFoundError, HfHubHTTPError

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
        "--use-symlinks",
        action="store_true",
        help="Allow snapshot_download to create symlinks for files (saves space)"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face authentication token (default: None)"
    )
    return parser.parse_args()

def robust_snapshot_download(repo_id, revision, output_dir, subfolder="", use_symlinks=False, token=None, max_tries=5):
    if token:
        login(token=token)

    patterns = []
    if subfolder:
        patterns = [f"{subfolder}/*", f"{subfolder}/**"]
    else:
        patterns = ["**"]  # whole repo

    for attempt in range(1, max_tries+1):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                local_dir=output_dir,
                local_dir_use_symlinks=use_symlinks,
                allow_patterns=patterns,
                max_workers=8,
                resume_download=True,  # Explicitly enable resume
                token=token
            )
            print("Download complete.")
            return True
        except LocalEntryNotFoundError as e:
            print(f"[Attempt {attempt}/{max_tries}] LocalEntryNotFoundError: {e}")
            if attempt == max_tries:
                raise
            time.sleep(min(60, 10*attempt))  # Longer wait time
        except HfHubHTTPError as e:
            if e.response.status_code == 429:
                wait_time = min(300, 30*attempt)  # Longer wait for rate limits
                print(f"[Attempt {attempt}/{max_tries}] Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"[Attempt {attempt}/{max_tries}] HTTP Error: {e}")
                if attempt == max_tries:
                    raise
                time.sleep(10)
        except Exception as e:
            print(f"[Attempt {attempt}/{max_tries}] {type(e).__name__}: {e}")
            if attempt == max_tries:
                raise
            time.sleep(min(60, 10*attempt))
    return False

def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Downloading '{args.subfolder}' from '{args.repo_id}@{args.revision}' into '{args.output_dir}'...")
    success = robust_snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        output_dir=args.output_dir,
        subfolder=args.subfolder,
        use_symlinks=args.use_symlinks,
        token=args.token,
    )
    
    if success:
        print("Download completed successfully.")
    else:
        print("Download failed after multiple attempts.")
        sys.exit(1)

if __name__ == "__main__":
    main()