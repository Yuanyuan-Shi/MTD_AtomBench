import json
from pathlib import Path

import pytest
import torch

from atom_mtd_offlinedata.common import read_config
from atom_mtd_offlinedata.data import AtomicPairs, batch


def test_x1_never_enters_training():
    with pytest.raises(ValueError, match="evaluation-only"):
        AtomicPairs(read_config(), "train", ("x1",))


def test_released_split_and_consecutive_batch():
    c = read_config()
    manifest = Path(c["dataset_root"]) / "splits.json"
    if not manifest.exists():
        pytest.skip("Run prepare.py to enable real trajectory integration test")
    data = json.loads(manifest.read_text())
    for task in ("i1", "i5"):
        train, val, test = (
            set(data["splits"][task][s]) for s in ("train", "validation", "test")
        )
        assert (len(train), len(val), len(test)) == (80, 10, 10)
        assert not (train & val or train & test or val & test)
    # Per-task split seeds: tasks share episode IDs 0..99 but not their splits.
    assert data["data_seed"] == c["data_seed"]
    assert data["splits"]["i1"]["test"] != data["splits"]["i5"]["test"]
    ds = AtomicPairs(c, "validation")
    # Noncontiguous episode subsets and both ends exercise the LeRobot boundary fix.
    metadata, actions = batch(ds, [0, len(ds) - 1])
    assert actions.shape == (4, 50, 32)
    assert actions.dtype == torch.float32
    assert metadata["observation"].state.dtype == torch.float32
    gap = c["execution_horizon"]
    assert torch.equal(metadata["timestep"][:, 1], metadata["timestep"][:, 0] + gap)
    assert torch.equal(metadata["task"], torch.tensor([[1, 1], [5, 5]]))
    assert metadata["language_mask"].any(1).all()
    assert not (
        metadata["language_mask"] & ~metadata["observation"].tokenized_prompt_mask
    ).any()
    # Instruction and state tokens: every real token except BOS and "\nAction: ".
    real = metadata["observation"].tokenized_prompt_mask.sum(1)
    assert not metadata["language_mask"][:, 0].any()
    assert torch.equal(metadata["language_mask"].sum(1), real - 1 - 4)
    assert (actions[:, :, 14:] == 0).all()


def test_pairs_follow_configured_execution_horizon():
    c = read_config()
    if not (Path(c["dataset_root"]) / "splits.json").exists():
        pytest.skip("Run prepare.py to enable real trajectory integration test")
    gap = c["execution_horizon"]
    ds = AtomicPairs(c, "validation", ("i1",))
    lengths = [
        ds.datasets["i1"].meta.episodes[ep]["length"]
        for ep in ds.datasets["i1"].episodes
    ]
    assert len(ds) == sum(n - c["action_horizon"] - gap + 1 for n in lengths)
    (record_t, _), (record_next, _) = ds[len(ds) // 2][0]
    # The second query's chunk starts where the first chunk's step `gap` is.
    assert (record_t["actions"][gap] == record_next["actions"][0]).all()


def test_frame_cache_records_are_bit_identical_to_video_decoding():
    import numpy as np

    from atom_mtd_offlinedata.cache_frames import load_manifest

    c = read_config()
    if load_manifest(c) is None:
        pytest.skip("Run cache_frames.py to enable the cache equality test")
    cached = AtomicPairs(c, "validation")
    video = AtomicPairs(c, "validation", use_cache=False)
    assert cached.cache_dir is not None and video.cache_dir is None
    assert [p[:4] for p in cached.pairs] == [p[:4] for p in video.pairs]
    rng = np.random.default_rng(0)
    picks = [0, len(cached) - 1, *rng.integers(len(cached), size=6).tolist()]
    for index in picks:
        a, b = cached[index], video[index]
        assert a[1:] == b[1:]  # task, episode, timesteps
        for (record_a, mask_a), (record_b, mask_b) in zip(a[0], b[0]):
            assert np.array_equal(mask_a, mask_b)
            flat_a, flat_b = flatten(record_a), flatten(record_b)
            assert flat_a.keys() == flat_b.keys()
            for key in flat_a:
                assert np.asarray(flat_a[key]).dtype == np.asarray(flat_b[key]).dtype, (
                    key
                )
                assert np.array_equal(flat_a[key], flat_b[key]), key


def flatten(record, prefix=""):
    out = {}
    for key, value in record.items():
        if isinstance(value, dict):
            out.update(flatten(value, prefix + key + "."))
        else:
            out[prefix + key] = value
    return out


def test_angle_wrap_removes_fake_jumps_in_real_trajectories():
    import glob

    import numpy as np
    import pyarrow.parquet as pq

    from atom_mtd_offlinedata.common import wrap_angles

    c = read_config()
    root = Path(c["dataset_root"])
    manifest = json.loads((root / "splits.json").read_text())
    wrap = manifest["angle_wrap"]
    stats = json.loads((root / "norm_stats.json").read_text())["norm_stats"]
    q01, q99 = (np.array(stats["actions"][k]) for k in ("q01", "q99"))
    for directory in c["tasks"].values():
        for f in sorted(glob.glob(str(root / directory / "data/chunk-000/*.parquet"))):
            table = pq.read_table(
                f, columns=["observation.state", "action"]
            ).to_pydict()
            for key in ("observation.state", "action"):
                values = wrap_angles(
                    np.array(table[key]), wrap["dims"], wrap["centers"]
                )
                angles = values[:, wrap["dims"]]
                # Real wrist rotation per frame is < 0.1 rad; no 2*pi jumps remain.
                assert np.abs(np.diff(angles, axis=0)).max() < 0.2, f
                normalized = (values - q01) / (q99 - q01 + 1e-6) * 2 - 1
                assert np.abs(normalized[:, wrap["dims"]]).max() < 5, f  # was ~26


def test_physical_actions_round_trip_on_real_records():
    import numpy as np

    from atom_mtd_offlinedata.data import to_physical_actions

    c = read_config()
    ds = AtomicPairs(c, "validation", ("i5",), use_cache=False)
    task, row, episode, frame, cache_row = ds.pairs[len(ds) // 3]
    raw = ds._raw_record(task, row, cache_row, episode, frame)
    (record, _), _ = ds[len(ds) // 3][0]
    back = to_physical_actions(
        record["actions"], ds.stats["actions"], c["physical_action_dim"], ds.angle_dims
    )
    assert np.allclose(back, raw["actions"], atol=1e-4)
