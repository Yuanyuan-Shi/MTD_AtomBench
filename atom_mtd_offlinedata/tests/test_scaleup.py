"""Hardware-independence of results, planning and launch commands (no GPU needed)."""

import json
import math
from types import SimpleNamespace

import pytest
import torch

from atom_mtd_offlinedata.common import (
    ENVIRONMENT_EXPERIMENT_KEYS,
    HARDWARE_KEY,
    ROOT,
    config_hash,
    matches_run_config,
    read_config,
    stage_a_steps,
)

CONFIGS = ROOT / "atom_mtd_offlinedata/configs"


def test_local_and_aws_configs_differ_only_in_hardware_and_batch_size():
    local = json.loads((CONFIGS / "local.json").read_text())
    aws = json.loads((CONFIGS / "aws.json").read_text())
    allowed = {HARDWARE_KEY, *ENVIRONMENT_EXPERIMENT_KEYS}
    assert set(local) == set(aws)
    assert {k for k in local if local[k] != aws[k]} <= allowed


def test_config_hash_ignores_hardware_but_not_experiment_settings():
    c = read_config(CONFIGS / "aws.json")
    moved = {**c, HARDWARE_KEY: {**c[HARDWARE_KEY], "nodes": 1, "gpus_per_node": 4}}
    assert config_hash(moved) == config_hash(c)
    assert matches_run_config({**c, "seed": 7}, moved, 7)
    assert config_hash({**c, "batch_size": 32}) != config_hash(c)
    assert not matches_run_config({**c, "seed": 7}, {**c, "batch_size": 32}, 7)


def test_stage_a_steps_follow_epochs_over_pairs():
    c = {"batch_size": 64, "stage_a": {"epochs": 10}}
    assert stage_a_steps(c, 36188) == math.ceil(10 * 36188 / 32)


def test_plan_reproduces_option_b_on_three_nodes():
    from atom_mtd_offlinedata.plan import make_plan

    c = read_config(CONFIGS / "aws.json")
    c[HARDWARE_KEY]["benchmark_file"] = None  # use the documented estimates
    phases = {p["name"]: p for p in make_plan(c)["phases"]}
    gpus = {j["variant"]: j["gpus"] for j in phases["stage_a"]["jobs"]}
    assert gpus == {"joint_bc": 16, "teacher_i1": 16, "teacher_i5": 8}
    assert {j["gpus"] for j in phases["stage_b"]["jobs"]} == {8}
    assert len({tuple(j["nodes"]) for j in phases["stage_b"]["jobs"]}) == 3


def test_plan_runs_sequentially_on_one_gpu():
    from atom_mtd_offlinedata.plan import make_plan

    c = read_config(CONFIGS / "local.json")
    c[HARDWARE_KEY]["benchmark_file"] = None
    for phase in make_plan(c, smoke=True)["phases"]:
        jobs = sorted(phase["jobs"], key=lambda j: j["start_s"])
        assert all(j["gpus"] == 1 for j in jobs)
        assert all(
            b["start_s"] == pytest.approx(a["end_s"]) for a, b in zip(jobs, jobs[1:])
        )


def _with_world(monkeypatch, rank, world_size):
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", str(world_size))


def test_pair_loader_global_batch_is_identical_for_any_gpu_count(monkeypatch):
    from atom_mtd_offlinedata.data import PairLoader

    dataset = type("D", (), {"cache_dir": None, "__len__": lambda self: 1000})()
    reference = None
    for world_size in (1, 2, 4, 8, 16):
        parts = []
        for rank in range(world_size):
            _with_world(monkeypatch, rank, world_size)
            parts.extend(PairLoader(dataset, 64, seed=7, steps=10).step_indices(3))
        reference = parts if reference is None else reference
        assert parts == reference
    with pytest.raises(ValueError):
        _with_world(monkeypatch, 0, 3)
        PairLoader(dataset, 64, seed=7, steps=10)


def test_step_tau_and_noise_are_identical_for_any_gpu_count(monkeypatch):
    from atom_mtd_offlinedata.model import DistillationPolicy
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    policy = SimpleNamespace(seed=7, update_index=42)
    policy.sample_time = lambda n, device: PI0Pytorch.sample_time(None, n, device)
    policy.sample_noise = lambda shape, device: PI0Pytorch.sample_noise(
        None, shape, device
    )
    full = None
    for world_size in (1, 2, 4):
        taus, noises = [], []
        for rank in range(world_size):
            _with_world(monkeypatch, rank, world_size)
            local = torch.zeros(32 // world_size, 50, 32)
            tau, noise = DistillationPolicy.step_randomness(policy, local)
            taus.append(tau)
            noises.append(noise)
        tau, noise = torch.cat(taus), torch.cat(noises)
        full = full or (tau, noise)
        assert torch.equal(tau, full[0]) and torch.equal(noise, full[1])
    assert torch.equal(full[0][0::2], full[0][1::2])  # one tau per pair
    assert not torch.equal(full[1][0::2], full[1][1::2])  # independent noise


def test_evaluation_randomness_does_not_depend_on_batching():
    from atom_mtd_offlinedata.evaluate import pair_randomness
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    model = SimpleNamespace(
        sample_time=lambda n, device: PI0Pytorch.sample_time(None, n, device),
        sample_noise=lambda shape, device: PI0Pytorch.sample_noise(None, shape, device),
    )
    c = {"data_seed": 20261001}
    actions = torch.zeros(4, 50, 32)
    together = pair_randomness(model, c, "i1", [3, 9], actions)
    first = pair_randomness(model, c, "i1", [3], actions[:2])
    second = pair_randomness(model, c, "i1", [9], actions[:2])
    for joint, a, b in zip(together, first, second):
        assert torch.equal(joint, torch.cat([a, b]))


def test_slurm_commands_place_jobs_on_planned_nodes_and_gpus(monkeypatch):
    from atom_mtd_offlinedata.launch import SlurmLauncher

    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    launcher = SlurmLauncher(read_config(CONFIGS / "aws.json"), dry_run=True)
    two_nodes = {
        "variant": "joint_bc",
        "kind": "train",
        "gpus": 16,
        "nodes": [0, 1],
        "gpu_ids": list(range(8)),
    }
    argv, _ = launcher.command(two_nodes, 2, ["atom_mtd_offlinedata.train"])
    text = " ".join(argv)
    assert "--nodelist=node0,node1" in text and "--nnodes 2" in text
    assert "--nproc_per_node 8" in text and "--rdzv_endpoint node0:29502" in text
    half_node = {
        "variant": "teacher_i1",
        "kind": "eval",
        "gpus": 4,
        "nodes": [2],
        "gpu_ids": [4, 5, 6, 7],
    }
    argv, _ = launcher.command(half_node, 0, ["atom_mtd_offlinedata.evaluate"])
    text = " ".join(argv)
    assert "ATOM_GPU_IDS=4,5,6,7" in text and "--standalone" in text
    assert "--nodelist=node2" in text
