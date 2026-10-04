"""Fetch the base model and the pinned atomic release; split by episode before statistics."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import av
from huggingface_hub import snapshot_download

from .common import (
    ANGLE_EDGE_MARGIN,
    ANGLE_WRAP_SCHEME,
    DEFAULT_CONFIG,
    SPLIT_SCHEME,
    angle_centers,
    check_angle_margin,
    wrap_angles,
    TASK_NUMBERS,
    read_config,
    sha256,
    write_json,
)


def fetch_base_checkpoint(c):
    """Download the converted pi0.5 base model if missing; always verify its SHA256."""
    from huggingface_hub import hf_hub_download

    target = Path(c["base_checkpoint"])
    weights = target / "model.safetensors"
    if not weights.exists():
        for name in ("config.json", "model.safetensors"):
            hf_hub_download(
                c["base_checkpoint_hf_repo"],
                name,
                revision=c["base_checkpoint_hf_revision"],
                local_dir=target,
            )
    if sha256(weights) != c["base_checkpoint_sha256"]:
        raise ValueError(f"{weights} does not match base_checkpoint_sha256")
    print(f"Base checkpoint verified: {weights}")


def prepare(c):
    fetch_base_checkpoint(c)
    root = Path(c["dataset_root"])
    snapshot_download(
        c["dataset_repo"],
        repo_type="dataset",
        revision=c["dataset_revision"],
        local_dir=root,
        allow_patterns=[p + "/**" for p in c["tasks"].values()],
        max_workers=4,
    )
    splits, prompts, arrays, files = {}, {}, {"state": [], "actions": []}, {}
    every_frame = []  # state and actions of all episodes, for the angle-margin check
    video_counts = {}
    for task, directory in c["tasks"].items():
        assert task in ("i1", "i5")
        path = root / directory
        info = json.loads((path / "meta/info.json").read_text())
        tasks = [
            json.loads(x) for x in (path / "meta/tasks.jsonl").read_text().splitlines()
        ]
        assert len(tasks) == 1
        prompts[task] = tasks[0]["task"]
        episodes = [
            json.loads(x)
            for x in (path / "meta/episodes.jsonl").read_text().splitlines()
        ]
        video_counts[task] = 0
        ids = sorted(x["episode_index"] for x in episodes)
        assert (
            len(ids)
            == c["train_episodes_per_task"]
            + c["validation_episodes_per_task"]
            + c["test_episodes_per_task"]
        )
        # Independent split per task: both tasks have episode IDs 0..99, so a
        # shared seed would give them identical splits.
        rng = np.random.default_rng([c["data_seed"], TASK_NUMBERS[task]])
        ids = rng.permutation(ids).tolist()
        n, v = c["train_episodes_per_task"], c["validation_episodes_per_task"]
        splits[task] = {
            "train": sorted(ids[:n]),
            "validation": sorted(ids[n : n + v]),
            "test": sorted(ids[n + v :]),
        }
        for episode in episodes:
            ep = episode["episode_index"]
            file = path / info["data_path"].format(
                episode_chunk=ep // info["chunks_size"], episode_index=ep
            )
            table = pq.read_table(file).to_pydict()
            length = episode["length"]
            frames = np.asarray(table["frame_index"])
            # Released trajectories are trimmed and may begin at frame 13 etc.
            assert len(frames) == length and np.all(np.diff(frames) == 1), (
                task,
                ep,
                "non-consecutive frame indices",
            )
            assert set(table["episode_index"]) == {ep}
            assert set(table["task_index"]) == {tasks[0]["task_index"]}
            np.testing.assert_allclose(
                np.diff(table["timestamp"]), 1 / info["fps"], atol=1e-4
            )
            for key, source in (("state", "observation.state"), ("actions", "action")):
                values = np.asarray(table[source], dtype=np.float32)
                assert values.shape == (length, c["physical_action_dim"])
                assert np.isfinite(values).all()
                every_frame.append(values)
                if ep in splits[task]["train"]:
                    arrays[key].append(values)
            files[str(file.relative_to(root))] = sha256(file)
            for key, feature in info["features"].items():
                if feature["dtype"] != "video":
                    continue
                video = path / info["video_path"].format(
                    episode_chunk=ep // info["chunks_size"],
                    episode_index=ep,
                    video_key=key,
                )
                with av.open(str(video)) as container:
                    stream = container.streams.video[0]
                    assert stream.frames == length, (
                        task,
                        ep,
                        key,
                        "video/data frame count mismatch",
                    )
                    assert float(stream.average_rate) == info["fps"]
                video_counts[task] += 1
        for file in path.rglob("*.mp4"):
            files[str(file.relative_to(root))] = sha256(file)
    # Angles (e.g. end-effector roll near +-pi) are wrapped into a window centered
    # on the training data, so equal orientations get equal values before statistics.
    dims = c["angle_dims"]
    centers = angle_centers(np.concatenate(arrays["state"] + arrays["actions"]), dims)
    check_angle_margin(np.concatenate(every_frame), dims, centers)
    stats = {}
    for key, chunks in arrays.items():
        x = wrap_angles(np.concatenate(chunks), dims, centers)
        stats[key] = {
            "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(),
            "q01": np.quantile(x, 0.01, axis=0).tolist(),
            "q99": np.quantile(x, 0.99, axis=0).tolist(),
        }
    # NormStats file schema is the native OpenPI serialization.
    write_json(root / "norm_stats.json", {"norm_stats": stats})
    write_json(
        root / "splits.json",
        {
            "data_seed": c["data_seed"],
            "split_scheme": SPLIT_SCHEME,
            "splits": splits,
            "prompts": prompts,
            "revision": c["dataset_revision"],
            "files": files,
            "verified_video_counts": video_counts,
            "angle_wrap": {
                "dims": dims,
                "centers": centers,
                "scheme": ANGLE_WRAP_SCHEME,
                "edge_margin": ANGLE_EDGE_MARGIN,
            },
            "normalization": "pooled i1+i5 TRAIN episodes only; absolute 14-D targets; "
            "angle dims wrapped (angle_wrap); quantile",
        },
    )
    print(
        json.dumps(
            {
                "prompts": prompts,
                "episodes": {
                    t: {s: len(v) for s, v in d.items()} for t, d in splits.items()
                },
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    prepare(read_config(p.parse_args().config))
