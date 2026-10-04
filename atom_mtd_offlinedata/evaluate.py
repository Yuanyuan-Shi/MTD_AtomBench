"""Held-out trajectories only; sampled actions are open-loop chunk predictions.

Runs on one GPU or under torchrun: test pairs are split across ranks and the
per-batch rows are gathered on rank 0. Every pair's tau and noise come from its
own seed, so results do not depend on batch size or GPU count.
"""

import argparse
import json
import os

import jax
import numpy as np
import torch
import torch.distributed as dist

from .common import (
    DEFAULT_CONFIG,
    HARDWARE_KEY,
    TASK_NUMBERS,
    derive_seed,
    distributed_rank,
    eval_seed,
    output_root,
    read_config,
    write_json,
)
from .data import AtomicPairs, batch
from .losses import distillation_losses, assert_pairs, mse
from .model import DistillationPolicy, STAGE_A, STAGE_B
from .train import checkpoint_path, model_config, run_directory
from . import tracking

COMPARED = ("joint_bc",) + STAGE_B  # Stage-B starting point and the KD variants
EVALUATED = STAGE_A + STAGE_B  # specialists are evaluated as baselines on their task


def evaluated_tasks(variant):
    return (
        (variant.removeprefix("teacher_"),)
        if variant.startswith("teacher_")
        else ("i1", "i5", "x1")
    )


def pair_randomness(model, c, task, pair_ids, actions):
    """tau [2P], flow noise [2P,H,D] and sampling noise [2P,H,D] from per-pair seeds."""
    taus, noises, initial = [], [], []
    for pair_id in pair_ids:
        torch.manual_seed(derive_seed(eval_seed(c), TASK_NUMBERS[task], int(pair_id)))
        taus.append(model.sample_time(1, actions.device).repeat(2))
        noises.append(model.sample_noise((2, *actions.shape[1:]), actions.device))
        initial.append(model.sample_noise((2, *actions.shape[1:]), actions.device))
    return torch.cat(taus), torch.cat(noises), torch.cat(initial)


@torch.no_grad()
def evaluate(c, variant, seed, smoke=False):
    if variant not in EVALUATED:
        raise ValueError(f"Unknown variant {variant}")
    rank, world_size = distributed_rank()
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    compared = variant in COMPARED
    teachers = {
        t: checkpoint_path(c, "teacher_" + t, seed, smoke) for t in ("i1", "i5")
    }
    model = (
        DistillationPolicy(
            model_config(c),
            c,
            "joint_bc",  # never trained here; "joint_bc" only disables Stage-B checks
            seed,
            checkpoint_path(c, variant, seed, smoke),
            teachers if compared else {},
        )
        .to(device)
        .eval()
    )
    pairs_per_batch = c[HARDWARE_KEY]["eval_batch_pairs"]
    rows = {}
    for task in evaluated_tasks(variant):
        if task == "x1" and not c.get("x1_dataset_root"):
            rows[task] = {"status": "unavailable", "reason": c["x1_status"]}
            continue
        ds = AtomicPairs(c, "test", (task,), use_cache=False if task == "x1" else None)
        count = min(c["evaluation_pairs_per_task"], len(ds))
        if smoke:
            count = min(count, 4)
        ids = np.random.default_rng(eval_seed(c)).choice(
            len(ds), size=count, replace=False
        )
        chunks = [
            ids[i : i + pairs_per_batch] for i in range(0, count, pairs_per_batch)
        ]
        local = []
        with torch.random.fork_rng(devices=[device.index]):
            for selected in chunks[rank::world_size]:
                metadata, actions = batch(ds, selected)
                metadata = jax.tree.map(lambda x: x.to(device), metadata)
                actions = actions.to(device)
                tau, noise, initial_noise = pair_randomness(
                    model, c, task, selected, actions
                )
                if task != "x1":
                    student, teacher, masks = model.compare(
                        metadata,
                        actions,
                        with_teachers=compared,
                        tau=tau,
                        noise=noise,
                    )
                    row = {}
                    if compared:
                        losses = distillation_losses(
                            student,
                            teacher,
                            masks,
                            c,
                            repr_enabled=True,
                            temp_enabled=True,
                        )
                        row = {
                            "output_teacher_mse": float(losses["output"]),
                            "image_representation_error": float(losses["img"]),
                            "language_representation_error": float(losses["lang"]),
                            "action_representation_error": float(losses["action"]),
                            "temporal_residual_error": float(losses["temp"]),
                        }
                else:
                    assert_pairs(
                        metadata["task"],
                        metadata["episode"],
                        metadata["timestep"],
                        execution_horizon=c["execution_horizon"],
                    )
                    student = model.forward_features(
                        metadata["observation"],
                        actions,
                        noise=noise,
                        time=tau,
                        train=False,
                    )
                    row = {}  # deliberately no x1 teacher metrics
                row["task_flow_loss"] = float(student["task_loss"].mean())
                predicted = model.sample_actions(
                    actions.device,
                    metadata["observation"],
                    noise=initial_noise,
                    num_steps=c["denoise_steps"],
                )
                d = c["physical_action_dim"]
                row["first_action_mse"] = float(
                    mse(predicted[:, 0, :d], actions[:, 0, :d])
                )
                row["action_chunk_mse"] = float(
                    mse(predicted[:, :, :d], actions[:, :, :d])
                )
                # Also provide original dataset units, with heterogeneous physical dimensions disclosed.
                stats = ds.stats["actions"]
                scale = torch.as_tensor(
                    (stats.q99 - stats.q01 + 1e-6) / 2,
                    device=actions.device,
                    dtype=torch.float32,
                )
                row["first_action_mse_original_units"] = float(
                    ((predicted[:, 0, :d] - actions[:, 0, :d]) * scale).square().mean()
                )
                row["action_chunk_mse_original_units"] = float(
                    ((predicted[:, :, :d] - actions[:, :, :d]) * scale).square().mean()
                )
                local.append((row, len(selected)))
        gathered = [local]
        if world_size > 1:
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local)
        values = [entry for part in gathered for entry in part]
        rows[task] = {
            "status": "evaluated",
            "pairs": count,
            "metrics": {
                k: float(
                    np.average(
                        [v[k] for v, _ in values], weights=[w for _, w in values]
                    )
                )
                for k in values[0][0]
            },
            "pair_indices": ids.tolist(),
            "action_metric_space": "train-quantile-normalized physical dimensions 0:14; original-units metrics separately labeled",
        }
    if compared:
        rows["mean_atomic_temporal_residual_error"] = float(
            np.mean(
                [rows[t]["metrics"]["temporal_residual_error"] for t in ("i1", "i5")]
            )
        )
    if rank != 0:
        return None
    tracked = json.loads(
        (run_directory(c, variant, seed, smoke) / "wandb.json").read_text()
    )
    tracking.log_evaluation(c, tracked["path"], rows)
    # Written last: its existence marks the evaluation complete (run_experiment skips it).
    write_json(
        run_directory(c, variant, seed, smoke) / "evaluation.json",
        {"variant": variant, "seed": seed, "eval_seed": eval_seed(c), "results": rows},
    )
    return rows


def summarize(c, smoke=False):
    if len(c["training_seeds"]) != 1:
        raise ValueError("Current protocol requires one matched seed")
    seed = c["training_seeds"][0]
    summary = {}
    for variant in EVALUATED:
        results = json.loads(
            (run_directory(c, variant, seed, smoke) / "evaluation.json").read_text()
        )["results"]
        summary[variant] = {}
        for task in evaluated_tasks(variant):
            if results[task]["status"] != "evaluated":
                summary[variant][task] = {"status": "unavailable"}
                continue
            summary[variant][task] = {
                metric: {"value": float(value), "seed": seed}
                for metric, value in results[task]["metrics"].items()
            }
        if variant in COMPARED:
            summary[variant]["mean_atomic_temporal_residual_error"] = {
                "value": float(results["mean_atomic_temporal_residual_error"]),
                "seed": seed,
            }
    root = output_root(c) / ("smoke" if smoke else "main")
    write_json(root / "comparison.json", summary)
    lines = [
        "# Offline comparison",
        "",
        f"Single-seed results for seed {seed}. No across-seed uncertainty estimate. "
        "These are offline errors, not success rates. Specialists (teacher_*) are "
        "evaluated on their own task only, without teacher-matching metrics.",
        "",
    ]
    for task in ("i1", "i5", "x1"):
        lines += [f"## {task}", "", "| Model | Metric | Value |", "|---|---|---|"]
        for variant in summary:
            for metric, value in summary[variant].get(task, {}).items():
                lines.append(
                    f"| {variant} | {metric} | "
                    + (
                        "unavailable"
                        if isinstance(value, str)
                        else f"{value['value']:.7g}"
                    )
                    + " |"
                )
        lines += [""]
    lines += [
        "## Mean atomic temporal residual error",
        "",
        "| Model | Value |",
        "|---|---|",
    ]
    for variant in COMPARED:
        value = summary[variant]["mean_atomic_temporal_residual_error"]
        lines.append(f"| {variant} | {value['value']:.7g} |")
    (root / "comparison.md").write_text("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--variant", choices=EVALUATED)
    p.add_argument("--seed", type=int)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    c = read_config(args.config)
    if args.summarize:
        summarize(c, args.smoke)
    elif args.variant is None or args.seed is None:
        p.error("--variant and --seed are required unless --summarize")
    else:
        evaluate(c, args.variant, args.seed, args.smoke)
    if dist.is_initialized():
        dist.destroy_process_group()
