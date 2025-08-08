#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download a subfolder from a Hugging Face dataset repo to a user-specified path.

Repo:    robosense/datasets  (repo_type=dataset)
Subdir:  track5-cross-platform-3d-object-detection
Usage:
    python download_track5.py /path/to/save

Optionally:
    HUGGINGFACE_TOKEN=hf_xxx python download_track5.py /path/to/save
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download


REPO_ID = "robosense/datasets"
REPO_TYPE = "dataset"
SUBFOLDER = "track5-cross-platform-3d-object-detection"


def download_track5(dest_dir: str,
                    revision: Optional[str] = None,
                    token: Optional[str] = None) -> Path:
    """
    下载 Hugging Face 数据集 robosense/datasets 中的
    track5-cross-platform-3d-object-detection 子目录到 dest_dir。

    Parameters
    ----------
    dest_dir : str
        想保存到的本地路径（若不存在会创建）。
    revision : Optional[str]
        可选的版本（如特定分支名/commit/tag），默认最新。
    token : Optional[str]
        若需要私有读取权限，可传入 HF token（否则可用环境变量 HUGGINGFACE_TOKEN）。

    Returns
    -------
    Path
        实际保存数据的本地目录路径（dest_dir/SUBFOLDER）。
    """
    dest_dir = Path(dest_dir).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 只下载目标子目录（包括其下所有文件/子目录）
    cache_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=[f"{SUBFOLDER}/**"],
        revision=revision,
        token=token,
        # 把实际文件放到普通目录而不是建立到 cache 的符号链接，便于直接使用/移动
        local_dir=None,
        local_dir_use_symlinks=False,
    )

    # snapshot_download 返回的是整个（受 allow_patterns 过滤后的）快照根目录
    src = Path(cache_dir) / SUBFOLDER
    if not src.exists():
        raise FileNotFoundError(
            f"子目录未在快照中找到：{src}。请检查 allow_patterns 或仓库结构是否变化。"
        )

    out_dir = dest_dir / SUBFOLDER
    out_dir.mkdir(parents=True, exist_ok=True)

    # 将缓存中的子目录复制/同步到目标路径
    # 若目标已存在则合并覆盖（Python 3.8+）
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (out_dir / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            src_f = Path(root) / f
            dst_f = out_dir / rel / f
            # 若文件已存在则覆盖
            shutil.copy2(src_f, dst_f)

    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description=f"Download {REPO_ID}/{SUBFOLDER} to a local path.")
    parser.add_argument("dest", help="下载保存到的目录（会自动创建）")
    parser.add_argument("--revision", default=None,
                        help="可选：指定分支/tag/commit（默认最新）")
    parser.add_argument("--token", default=None,
                        help="可选：HF 访问令牌（也可用环境变量 HUGGINGFACE_TOKEN）")
    args = parser.parse_args()

    out_dir = download_track5(args.dest, revision=args.revision, token=args.token)
    print(f"✅ 下载完成，已保存到：{out_dir}")


if __name__ == "__main__":
    main()
