"""Decode every video frame once into 224x224 uint8 arrays read through np.memmap.

Each frame goes through exactly the training path's operations: LeRobot's pyav
decoder, the float -> uint8 conversion of data.py, then the same resize_with_pad
(openpi_client, PIL bilinear) that transforms.ResizeImages applies.
Training then reads arrays instead of seeking in mp4 files. Build once per
machine (per node on a cluster, onto local NVMe); rebuilding is skipped when a
complete cache for the same source files already exists.
"""

import argparse
import hashlib
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np

from .common import DEFAULT_CONFIG, HARDWARE_KEY, read_config, write_json

CACHE_VERSION = 2  # v2: PIL resize, as transforms.ResizeImages (v1 used JAX)
IMAGE_SIZE = 224
# Dataset camera key -> pi0.5 image slot (also used by data.py).
CAMERAS = {
    "base_0_rgb": "observation.images.image_front",
    "left_wrist_0_rgb": "observation.images.image_wrist",
    "right_wrist_0_rgb": "observation.images.image_side",
}
DECODE_CHUNK = 64  # frames decoded per call; bounds worker memory to about 1 GB


def cache_root(c):
    return Path(c[HARDWARE_KEY]["frame_cache_dir"])


def source_fingerprint(c):
    """Identifies the exact parquet/video files (hashes recorded by prepare.py)."""
    manifest = json.loads((Path(c["dataset_root"]) / "splits.json").read_text())
    files = json.dumps(manifest["files"], sort_keys=True).encode()
    return hashlib.sha256(files).hexdigest()


def expected_manifest(c):
    return {
        "version": CACHE_VERSION,
        "dataset_revision": c["dataset_revision"],
        "source_fingerprint": source_fingerprint(c),
        "image_size": IMAGE_SIZE,
        "cameras": CAMERAS,
        "tasks": sorted(c["tasks"]),
    }


def load_manifest(c):
    """The cache manifest if a complete cache matches this config, else None."""
    if not c[HARDWARE_KEY].get("frame_cache_dir"):
        return None
    path = cache_root(c) / "manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    expected = expected_manifest(c)
    if not manifest.get("complete") or any(
        manifest.get(k) != v for k, v in expected.items()
    ):
        return None
    return manifest


def _open_dataset(c, task):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        c["dataset_repo"],
        root=Path(c["dataset_root"]) / c["tasks"][task],
        revision=c["dataset_revision"],
        video_backend="pyav",
        download_videos=False,
    )


def _decode_episode(job):
    """Decode one (task, camera, episode) video into its rows of the camera array."""
    import torch
    from lerobot.common.datasets.video_utils import decode_video_frames
    from openpi_client import image_tools  # the module transforms.ResizeImages uses

    array_path, video_path, timestamps, rows, tolerance_s = job
    out = np.load(array_path, mmap_mode="r+")
    for start in range(0, len(timestamps), DECODE_CHUNK):
        ts = timestamps[start : start + DECODE_CHUNK]
        frames = decode_video_frames(video_path, ts, tolerance_s, "pyav")
        with torch.no_grad():
            # Same conversion as data.py, then the same resize as transforms.ResizeImages.
            images = (frames.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
        resized = np.asarray(
            image_tools.resize_with_pad(images, IMAGE_SIZE, IMAGE_SIZE)
        )
        out[rows[start : start + len(ts)]] = resized
    out.flush()
    return len(timestamps)


def build(c, workers):
    if not c[HARDWARE_KEY].get("frame_cache_dir"):
        raise SystemExit(
            "hardware.frame_cache_dir is null in this config; nothing to build"
        )
    root = cache_root(c)
    if load_manifest(c) is not None:
        print(f"Frame cache already complete: {root}")
        return
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").unlink(missing_ok=True)
    manifest = {**expected_manifest(c), "complete": False, "frames": {}}
    prompts = json.loads((Path(c["dataset_root"]) / "splits.json").read_text())[
        "prompts"
    ]
    jobs = []
    for task in sorted(c["tasks"]):
        ds = _open_dataset(c, task)
        columns = ds.hf_dataset.with_format("numpy")[:]
        # Arrays are keyed by row position in the full task dataset (the released
        # "index" column has gaps after trimming); episodes are stored in order.
        episode = np.asarray(columns["episode_index"], dtype=np.int64)
        if np.any(np.diff(episode) < 0):
            raise ValueError(f"{task}: episodes are not stored in ascending order")
        task_dir = root / task
        task_dir.mkdir(exist_ok=True)
        np.save(
            task_dir / "state.npy",
            np.stack(columns["observation.state"]).astype(np.float32),
        )
        np.save(task_dir / "action.npy", np.stack(columns["action"]).astype(np.float32))
        np.save(task_dir / "episode_index.npy", episode)
        np.save(
            task_dir / "frame_index.npy",
            np.asarray(columns["frame_index"], dtype=np.int64),
        )
        timestamps = np.asarray(columns["timestamp"], dtype=np.float64)
        for slot, video_key in CAMERAS.items():
            path = task_dir / f"{slot}.npy"
            np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.uint8,
                shape=(len(episode), IMAGE_SIZE, IMAGE_SIZE, 3),
            ).flush()
            for ep in np.unique(episode):
                rows = np.flatnonzero(episode == ep)
                video = ds.root / ds.meta.get_video_file_path(int(ep), video_key)
                jobs.append(
                    (
                        str(path),
                        str(video),
                        timestamps[rows].tolist(),
                        rows,
                        ds.tolerance_s,
                    )
                )
        manifest["frames"][task] = int(len(episode))
        manifest.setdefault("prompts", {})[task] = prompts[task]
    jobs.sort(key=lambda j: -len(j[2]))  # longest first for better load balance
    started, done = time.time(), 0
    context = multiprocessing.get_context("spawn")  # JAX is not fork-safe
    with context.Pool(workers) as pool:
        for count in pool.imap_unordered(_decode_episode, jobs):
            done += count
            if done % 20000 < count:
                print(
                    f"decoded {done} camera-frames in {time.time() - started:.0f}s",
                    flush=True,
                )
    write_json(root / "manifest.json", {**manifest, "complete": True})
    print(
        f"Frame cache complete: {root} ({done} camera-frames, {time.time() - started:.0f}s)"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument(
        "--workers", type=int, help="defaults to hardware.cache_build_workers"
    )
    args = p.parse_args()
    c = read_config(args.config)
    build(c, args.workers or c[HARDWARE_KEY]["cache_build_workers"])
