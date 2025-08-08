#!/usr/bin/env python3
"""
download_track5.py

Usage:
    python download_track5.py /your/target/path
"""

import argparse
from huggingface_hub import snapshot_download, login
import time, sys

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
                max_workers=8,   # speed + some robustness
                # force_download=False (default) will resume if partial cache exists
            )
            print("Download complete.")
            return
        except Exception as e:
            print(f"[Attempt {attempt}/{max_tries}] {type(e).__name__}: {e}")
            if attempt == max_tries:
                raise
            time.sleep(min(30, 5*attempt))

def main():
    args = parse_args()

    print(f"Downloading '{args.subfolder}' from '{args.repo_id}@{args.revision}' into '{args.output_dir}'...")
    robust_snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        output_dir=args.output_dir,
        subfolder=args.subfolder,
        use_symlinks=args.use_symlinks,
        token=None,  # or os.environ["HUGGINGFACE_HUB_TOKEN"]
    )
    print("Download complete.")

if __name__ == "__main__":
    main()
