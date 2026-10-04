import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "atom_mtd_offlinedata/configs/local.json"
TASK_NUMBERS = {"i1": 1, "i5": 5, "x1": 101}
# Recorded in splits.json; a split made by any other scheme must be regenerated.
SPLIT_SCHEME = "per-task default_rng([data_seed, task_number])"
# Per-step seed is seed * STEP_SEED_STRIDE + step: unique while steps < stride.
STEP_SEED_STRIDE = 1_000_003
# Where and how a run executes; never changes results, so it is excluded from
# config_hash and run matching. local.json and aws.json may differ only here
# and in ENVIRONMENT_EXPERIMENT_KEYS.
HARDWARE_KEY = "hardware"
ENVIRONMENT_EXPERIMENT_KEYS = ("batch_size",)


def read_config(path=DEFAULT_CONFIG):
    c = json.loads(Path(path).read_text())
    for key in ("dataset_root", "base_checkpoint", "x1_dataset_root"):
        if c.get(key) and not Path(c[key]).is_absolute():
            c[key] = str(ROOT / c[key])
    hardware = c[HARDWARE_KEY]
    for key in ("output_root", "frame_cache_dir", "benchmark_file"):
        if hardware.get(key) and not Path(hardware[key]).is_absolute():
            hardware[key] = str(ROOT / hardware[key])
    return c


def output_root(c):
    return Path(c[HARDWARE_KEY]["output_root"])


def experiment_settings(c):
    """The config without hardware settings: everything that can change results."""
    return {k: v for k, v in c.items() if k != HARDWARE_KEY}


def eval_seed(c):
    """Test pairs and evaluation noise; shared by every model and training seed."""
    return c["data_seed"] + 1


def probe_seed(c):
    """Fixed train/validation probe batches and noise."""
    return c["data_seed"] + 2


def audit_seed(c):
    """One-time Stage-B representation audit."""
    return c["data_seed"] + 3


def derive_seed(*entropy):
    """Independent 32-bit seed for a tuple of non-negative integers."""
    return int(np.random.SeedSequence(list(entropy)).generate_state(1)[0])


def stage_a_steps(c, pair_count):
    """Updates for the configured epochs; one epoch = one pass over valid pairs."""
    pairs_per_batch = c["batch_size"] // 2
    return math.ceil(c["stage_a"]["epochs"] * pair_count / pairs_per_batch)


def distributed_rank():
    """(rank, world_size); valid before and after process-group initialization."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


# Angle dims (config "angle_dims") are wrapped into (center - pi, center + pi] with
# center = circular mean of the training data, snapped to a multiple of pi/2.
# Data must keep at least this distance from the window edge, else no window works.
ANGLE_EDGE_MARGIN = np.pi / 4
ANGLE_WRAP_SCHEME = (
    "(center - pi, center + pi]; center = train circular mean snapped to pi/2"
)


def wrap_angles(values, dims, centers):
    """Copy of values with each angle column mapped into (center - pi, center + pi].

    Adding multiples of 2*pi never changes the orientation. Values already inside
    their window are returned bit-identical; only values outside move.
    """
    out = np.array(values, copy=True)
    for dim, center in zip(dims, centers):
        x = out[..., dim].astype(np.float64)
        high = center + np.pi
        outside = (x <= center - np.pi) | (x > high)
        x[outside] = high - np.mod(high - x[outside], 2 * np.pi)
        out[..., dim] = x
    return out


def angle_centers(values, dims):
    """Per angle column of values [N, D]: circular mean snapped to a multiple of pi/2."""
    centers = []
    for dim in dims:
        mean = np.angle(np.exp(1j * values[:, dim].astype(np.float64)).mean())
        centers.append(
            float(np.round(mean / (np.pi / 2)) * (np.pi / 2)) + 0.0
        )  # no -0.0
    return centers


def check_angle_margin(values, dims, centers):
    """Raise if wrapped angles come within ANGLE_EDGE_MARGIN of their window edge."""
    wrapped = wrap_angles(values, dims, centers)
    for dim, center in zip(dims, centers):
        deviation = float(np.abs(wrapped[..., dim] - center).max())
        if deviation > np.pi - ANGLE_EDGE_MARGIN:
            raise ValueError(
                f"Angle dim {dim} spans {deviation:.2f} rad from center {center:.2f}: "
                "too close to the wrap edge for any window; use a sin/cos encoding"
            )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def config_hash(c):
    return hashlib.sha256(
        json.dumps(experiment_settings(c), sort_keys=True).encode()
    ).hexdigest()


def matches_run_config(saved, current, seed):
    """Allow removing planned repetitions and changing hardware.

    Every training/data setting and the actual run seed must still match.
    """
    return saved.get("seed") == seed and all(
        saved.get(key) == value
        for key, value in experiment_settings(current).items()
        if key != "training_seeds"
    )
