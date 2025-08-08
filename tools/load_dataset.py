#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import time
import random
from pathlib import Path
from typing import Optional, List

from huggingface_hub import (
    snapshot_download,
    hf_hub_download,
    HfApi,
)
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

REPO_ID = "robosense/datasets"
REPO_TYPE = "dataset"
SUBFOLDER = "track5-cross-platform-3d-object-detection"

def _copy_tree(src: Path, dst: Path):
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(Path(root) / f, dst / rel / f)

def _list_repo_files_in_subfolder(repo_id: str, subfolder: str, revision: Optional[str], token: Optional[str]) -> List[str]:
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, revision=revision, repo_type=REPO_TYPE)
    # 只保留子目录下的文件
    subfolder = subfolder.strip("/")+ "/"
    return [f for f in files if f.startswith(subfolder)]

def _sequential_download_with_retry(dest_dir: Path, files: List[str], revision: Optional[str], token: Optional[str]):
    """逐文件下载，带指数退避，尽量避免429。"""
    for rel_path in files:
        out_path = dest_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 已存在则跳过（resume）
        if out_path.exists() and out_path.stat().st_size > 0:
            continue

        # 最多重试 8 次，退避上限 ~ 60s
        delay = 1.5
        for attempt in range(8):
            try:
                cached = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=rel_path,
                    repo_type=REPO_TYPE,
                    revision=revision,
                    token=token,
                    local_files_only=False,
                    # 避免走 symlink，直接落到 cache 文件
                    force_download=False,
                    resume_download=True,
                )
                # 拷贝到目标位置
                shutil.copy2(cached, out_path)
                break
            except HfHubHTTPError as e:
                # 429 / 5xx 做退避重试
                status = getattr(e.response, "status_code", None)
                if status in (429, 500, 502, 503, 504):
                    sleep_s = delay + random.uniform(0, 0.5*delay)
                    time.sleep(sleep_s)
                    delay = min(delay * 2, 60)
                    if attempt == 7:
                        raise
                else:
                    raise
            except LocalEntryNotFoundError:
                # 网络波动/HEAD失败也重试
                sleep_s = delay + random.uniform(0, 0.5*delay)
                time.sleep(sleep_s)
                delay = min(delay * 2, 60)
                if attempt == 7:
                    raise

def download_track5(dest_dir: str,
                    revision: Optional[str] = None,
                    token: Optional[str] = None) -> Path:
    dest_dir = Path(dest_dir).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    out_dir = dest_dir / SUBFOLDER
    out_dir.mkdir(parents=True, exist_ok=True)

    # 先尝试稳妥的 snapshot_download，降低并发避免429
    try:
        cache_dir = snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            allow_patterns=[f"{SUBFOLDER}/**"],
            revision=revision,
            token=token,
            local_dir=None,
            local_dir_use_symlinks=False,
            # 关键：降低并发
            max_workers=2,
            # 断点续传
            resume_download=True,
        )
        src = Path(cache_dir) / SUBFOLDER
        if not src.exists():
            raise FileNotFoundError(f"子目录未在快照中找到：{src}")
        _copy_tree(src, out_dir)
        return out_dir
    except (HfHubHTTPError, LocalEntryNotFoundError):
        # 如果还是被限流，退而求其次：逐文件顺序下载 + 重试
        files = _list_repo_files_in_subfolder(REPO_ID, SUBFOLDER, revision, token)
        if not files:
            raise RuntimeError("在仓库中没有找到目标子目录的文件，请检查路径或权限。")
        _sequential_download_with_retry(dest_dir=dest_dir, files=files, revision=revision, token=token)
        return out_dir

def main():
    parser = argparse.ArgumentParser(
        description=f"Download {REPO_ID}/{SUBFOLDER} to a local path."
    )
    parser.add_argument("dest", help="下载保存到的目录（会自动创建）")
    parser.add_argument("--revision", default=None, help="可选：分支/tag/commit（默认最新）")
    parser.add_argument("--token", default=os.environ.get("HUGGINGFACE_TOKEN"), help="可选：HF 访问令牌")
    args = parser.parse_args()

    out_dir = download_track5(args.dest, revision=args.revision, token=args.token)
    print(f"✅ 下载完成，已保存到：{out_dir}")

if __name__ == "__main__":
    main()
