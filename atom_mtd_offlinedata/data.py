"""LeRobot v2.1 data and native OpenPI transforms, with explicit query pairs.

A pair is (t, t + execution_horizon): two consecutive policy queries when
config["execution_horizon"] actions are executed between them.

Frames come from the frame cache (cache_frames.py) when hardware.frame_cache_dir
is set, otherwise from the mp4 files through LeRobot. Both paths produce
bit-identical records (tests/test_actual_data.py).
"""

import json
import os
from pathlib import Path

import jax
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi import transforms
from openpi.models import model as model_lib
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import normalize

from .cache_frames import CAMERAS, cache_root, load_manifest
from .common import (
    HARDWARE_KEY,
    SPLIT_SCHEME,
    TASK_NUMBERS,
    distributed_rank,
    wrap_angles,
)
from .losses import assert_pairs


class EpisodeSubset(LeRobotDataset):
    def _get_query_indices(self, idx, ep_idx):
        # This installed LeRobot version indexes subset boundaries by original
        # episode ID. Translate IDs to subset positions for noncontiguous splits.
        position = self.episodes.index(ep_idx) if self.episodes is not None else ep_idx
        return super()._get_query_indices(idx, position)


class AtomicPairs:
    def __init__(self, config, split, tasks=("i1", "i5"), *, use_cache=None):
        """use_cache: None = use the frame cache iff configured; False = decode video."""
        self.config, self.split = config, split
        self.execution_horizon = config["execution_horizon"]
        if not 1 <= self.execution_horizon < config["action_horizon"]:
            raise ValueError("Require 1 <= execution_horizon < action_horizon")
        if split == "train" and any(t not in ("i1", "i5") for t in tasks):
            raise ValueError("x1 is strictly evaluation-only")
        self.root = Path(config["dataset_root"])
        self.manifest = json.loads((self.root / "splits.json").read_text())
        assert self.manifest["revision"] == config["dataset_revision"]
        if (
            self.manifest.get("data_seed") != config["data_seed"]
            or self.manifest.get("split_scheme") != SPLIT_SCHEME
        ):
            raise ValueError("splits.json does not match data_seed; rerun prepare.py")
        # Angle windows were fixed from the training data by prepare.py.
        angle_wrap = self.manifest.get("angle_wrap", {})
        if angle_wrap.get("dims") != config["angle_dims"]:
            raise ValueError("splits.json does not match angle_dims; rerun prepare.py")
        self.angle_dims, self.angle_centers = angle_wrap["dims"], angle_wrap["centers"]
        self.cache_dir = None
        if use_cache is not False and config[HARDWARE_KEY].get("frame_cache_dir"):
            if load_manifest(config) is None:
                raise FileNotFoundError(
                    f"No complete frame cache at {cache_root(config)}; run cache_frames.py"
                )
            self.cache_dir = cache_root(config)
        self._arrays, self._arrays_pid = {}, None
        self.stats = normalize.load(self.root)
        self.tokenizer = PaligemmaTokenizer(config["max_token_len"])
        steps = [transforms.Normalize(self.stats, use_quantiles=True)]
        if self.cache_dir is None:
            # Cached frames are already resize_with_pad'ed; resizing 224x224 is identity.
            steps.append(transforms.ResizeImages(224, 224))
        steps += [
            transforms.TokenizePrompt(self.tokenizer, discrete_state_input=True),
            transforms.PadStatesAndActions(config["action_dim"]),
        ]
        self.transform = transforms.compose(steps)
        self.datasets, self.pairs = {}, []
        for task in tasks:
            if task == "x1":
                if split != "test" or not config.get("x1_dataset_root"):
                    raise ValueError(
                        "x1 requires valid released trajectories and test-only access"
                    )
                if self.cache_dir is not None:
                    raise ValueError("x1 trajectories are not part of the frame cache")
                path = Path(config["x1_dataset_root"])
                episodes = [
                    json.loads(s)["episode_index"]
                    for s in (path / "meta/episodes.jsonl").read_text().splitlines()
                ]
                prompts = [
                    json.loads(s)["task"]
                    for s in (path / "meta/tasks.jsonl").read_text().splitlines()
                ]
                if prompts != [
                    "Pick up exactly two red cubes and place them into the basket."
                ]:
                    raise ValueError(
                        "x1 released prompt does not match verified Franka x1"
                    )
                self.manifest["prompts"][task] = prompts[0]
            else:
                path = self.root / config["tasks"][task]
                episodes = self.manifest["splits"][task][split]
            info = json.loads((path / "meta/info.json").read_text())
            self.datasets[task] = ds = EpisodeSubset(
                config["dataset_repo"],
                root=path,
                episodes=episodes,
                revision=config["dataset_revision"],
                video_backend="pyav",
                download_videos=False,
                delta_timestamps={
                    "action": [i / info["fps"] for i in range(config["action_horizon"])]
                },
            )
            if self.cache_dir is not None:
                # Row of each episode's first frame in the full task dataset (cache key).
                cached_episodes = np.load(self.cache_dir / task / "episode_index.npy")
            offset = 0
            for ep in sorted(episodes):
                length = ds.meta.episodes[ep]["length"]
                full_row = (
                    int(np.searchsorted(cached_episodes, ep))
                    if self.cache_dir is not None
                    else None
                )
                # Both H-step chunks (at t and t + execution_horizon) must be real:
                # no end-of-episode padding, so t + execution_horizon + H <= length.
                first_frame = int(ds.hf_dataset[offset]["frame_index"])
                last_start = length - config["action_horizon"] - self.execution_horizon
                self.pairs.extend(
                    (
                        task,
                        offset + t,
                        ep,
                        first_frame + t,
                        None if full_row is None else full_row + t,
                    )
                    for t in range(last_start + 1)
                )
                offset += length
        if not self.pairs:
            raise ValueError("No pairs with two complete action chunks")

    def __len__(self):
        return len(self.pairs)

    def _cached(self, task):
        # Open memory maps lazily in each process (loader workers fork from the trainer).
        if self._arrays_pid != os.getpid():
            self._arrays, self._arrays_pid = {}, os.getpid()
        if task not in self._arrays:
            names = ("state", "action", "episode_index", "frame_index", *CAMERAS)
            self._arrays[task] = {
                n: np.load(self.cache_dir / task / f"{n}.npy", mmap_mode="r")
                for n in names
            }
        return self._arrays[task]

    def _raw_record(self, task, row, cache_row, episode, frame):
        """Untransformed record of one frame (angles wrapped); same for cache and video."""
        record = self._read_frame(task, row, cache_row, episode, frame)
        for key in ("state", "actions"):
            record[key] = wrap_angles(record[key], self.angle_dims, self.angle_centers)
        return record

    def _read_frame(self, task, row, cache_row, episode, frame):
        horizon = self.config["action_horizon"]
        if self.cache_dir is not None:
            a = self._cached(task)
            assert int(a["episode_index"][cache_row]) == episode
            assert int(a["frame_index"][cache_row]) == frame
            # The whole action chunk must lie inside the episode (no padding).
            assert int(a["episode_index"][cache_row + horizon - 1]) == episode
            return {
                "image": {k: np.array(a[k][cache_row]) for k in CAMERAS},
                "image_mask": {k: np.asarray(True) for k in CAMERAS},
                "state": np.array(a["state"][cache_row]),
                "actions": np.array(a["action"][cache_row : cache_row + horizon]),
                "prompt": self.manifest["prompts"][task],
            }
        raw = self.datasets[task][row]
        assert int(raw["episode_index"]) == episode
        assert int(raw["frame_index"]) == frame
        assert not raw["action_is_pad"].any()
        assert raw["task"] == self.manifest["prompts"][task]
        return {
            "image": {
                k: (raw[v].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                for k, v in CAMERAS.items()
            },
            "image_mask": {k: np.asarray(True) for k in CAMERAS},
            "state": raw["observation.state"].numpy(),
            "actions": raw["action"].numpy(),
            "prompt": raw["task"],
        }

    def __getitem__(self, index):
        task, row, episode, timestep, cache_row = self.pairs[index]
        samples = []
        for delta in (0, self.execution_horizon):
            record = self._raw_record(
                task,
                row + delta,
                None if cache_row is None else cache_row + delta,
                episode,
                timestep + delta,
            )
            record = self.transform(record)
            # Quantile statistics deserialize as float64; native Torch training
            # uses float32 actions. Keep direct offline evaluation identical.
            record["actions"] = record["actions"].astype(np.float32)
            record["state"] = record["state"].astype(np.float32)
            # Language KD tokens: instruction and discretized state, i.e. every real
            # prompt token except BOS and the trailing "\nAction: " cue. Length
            # follows each prompt, so any instruction length is supported.
            ids = record["tokenized_prompt"]
            length = int(record["tokenized_prompt_mask"].sum())
            cue = self.tokenizer._tokenizer.encode("\nAction: ")
            if length <= len(cue) + 1 or list(ids[length - len(cue) : length]) != cue:
                raise ValueError(
                    "Unable to locate the action cue; prompt may be truncated"
                )
            language_mask = np.zeros_like(record["tokenized_prompt_mask"])
            language_mask[1 : length - len(cue)] = True  # exclude BOS and action cue
            samples.append((record, language_mask))
        task_num = TASK_NUMBERS[task]
        times = [timestep, timestep + self.execution_horizon]
        return samples, [task_num] * 2, [episode] * 2, times


def batch(dataset, indices):
    records, language_masks, tasks, episodes, times = [], [], [], [], []
    for index in indices:
        samples, task, ep, time = dataset[int(index)]
        for record, mask in samples:
            records.append(record)
            language_masks.append(mask)
        tasks.append(task)
        episodes.append(ep)
        times.append(time)
    stacked = jax.tree.map(lambda *x: torch.from_numpy(np.stack(x)), *records)
    actions = stacked.pop("actions")
    observation = model_lib.Observation.from_dict(stacked)
    metadata = {
        "observation": observation,
        "language_mask": torch.from_numpy(np.stack(language_masks)),
        "task": torch.tensor(tasks),
        "episode": torch.tensor(episodes),
        "timestep": torch.tensor(times),
    }
    assert_pairs(
        metadata["task"],
        metadata["episode"],
        metadata["timestep"],
        execution_horizon=dataset.execution_horizon,
    )
    return metadata, actions


class _StepBatches(torch.utils.data.Dataset):
    """Batch for a given step, so DataLoader workers stay step-addressable."""

    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return self.loader.steps

    def __getitem__(self, step):
        return batch(self.loader.dataset, self.loader.step_indices(step))


class PairLoader:
    """One batch per optimizer step; identical global batches for any GPU count.

    The global batch (batch_size observations = batch_size // 2 pairs) is drawn
    from SeedSequence([seed, step]); each rank takes its contiguous slice.
    """

    def __init__(self, dataset, batch_size, seed, steps, *, start_step=0, workers=0):
        if batch_size % 2:
            raise ValueError("batch_size counts observations and must be even")
        rank, world_size = distributed_rank()
        self.pairs_per_batch = batch_size // 2
        if self.pairs_per_batch % world_size:
            raise ValueError(
                f"{self.pairs_per_batch} pairs per batch do not split over {world_size} GPUs"
            )
        self.local_pairs = self.pairs_per_batch // world_size
        self.offset = rank * self.local_pairs
        self.dataset, self.seed, self.steps = dataset, seed, steps
        self.start_step = start_step
        # Video decoding in forked workers is unsafe (LeRobot); cached arrays are fine.
        self.workers = workers if dataset.cache_dir is not None else 0

    def step_indices(self, step):
        # Step-addressable sampler; independent of teacher forwards and model RNG.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
        global_indices = rng.integers(len(self.dataset), size=self.pairs_per_batch)
        return global_indices[self.offset : self.offset + self.local_pairs]

    def __iter__(self):
        steps = range(self.start_step, self.steps)
        if not self.workers:
            for step in steps:
                yield batch(self.dataset, self.step_indices(step))
            return
        yield from torch.utils.data.DataLoader(
            _StepBatches(self),
            batch_size=None,
            sampler=steps,
            num_workers=self.workers,
            prefetch_factor=4,
            multiprocessing_context="fork",
        )

    def __len__(self):
        return self.steps - self.start_step


def to_physical_actions(
    normalized, stats, physical_dim, angle_dims, output_angle_center=None
):
    """Model output (normalized, padded to action_dim) -> physical actions [..., physical_dim].

    Inverts OpenPI's quantile normalization (transforms.Unnormalize with quantiles) on
    the physical dimensions. Angles come out in their training windows (see
    splits.json "angle_wrap"); all windows describe the same orientations. Pass
    output_angle_center (e.g. 0.0 for a controller expecting (-pi, pi]) to re-wrap them.
    Pose convention of this dataset: flange x, y, z in meters in the robot base frame;
    rx, ry, rz roll-pitch-yaw in radians, R = Rz(rz) @ Ry(ry) @ Rx(rx).
    """
    y = np.asarray(normalized, dtype=np.float64)[..., :physical_dim]
    q01 = np.asarray(stats.q01, dtype=np.float64)[:physical_dim]
    q99 = np.asarray(stats.q99, dtype=np.float64)[:physical_dim]
    actions = (y + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    if output_angle_center is not None:
        actions = wrap_angles(
            actions, angle_dims, [output_angle_center] * len(angle_dims)
        )
    return actions
