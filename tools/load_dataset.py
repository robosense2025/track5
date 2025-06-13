#!/usr/bin/env python3
"""
download_track5.py

Usage:
    python download_track5.py /your/target/path
"""

import argparse
from huggingface_hub import snapshot_download

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

def main():
    args = parse_args()

    print(f"Downloading '{args.subfolder}' from '{args.repo_id}@{args.revision}' into '{args.output_dir}'...")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output_dir,
        local_dir_use_symlinks=args.use_symlinks,
        allow_patterns=[ f"{args.subfolder}/*" ],
    )
    print("Download complete.")

if __name__ == "__main__":
    main()
