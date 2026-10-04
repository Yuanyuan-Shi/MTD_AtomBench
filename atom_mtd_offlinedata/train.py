"""Configure hooks into openpi/scripts/train_pytorch.py; no second optimizer loop.

Runs as one process (local) or under torchrun with any number of GPUs/nodes: the
global batch, tau and noise do not depend on the GPU count (see data.PairLoader and
model.DistillationPolicy.forward). Rank 0 owns every file, probe and W&B call.
"""

import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import wandb
from openpi.models.pi0_config import Pi0Config
from openpi.training.config import DataConfig, TrainConfig
from openpi.training.optimizer import AdamW, CosineDecaySchedule

from .common import (
    DEFAULT_CONFIG,
    HARDWARE_KEY,
    ROOT,
    STEP_SEED_STRIDE,
    audit_seed,
    config_hash,
    distributed_rank,
    matches_run_config,
    output_root,
    probe_seed,
    read_config,
    sha256,
    stage_a_steps,
    write_json,
)
from .data import AtomicPairs, PairLoader, batch
from .model import DistillationPolicy, STAGE_A, STAGE_B
from . import tracking


def run_directory(c, name, seed, smoke=False):
    return output_root(c) / ("smoke" if smoke else "main") / f"{name}_seed{seed}"


def benchmark_directory(c, name, world_size):
    return output_root(c) / "benchmark" / f"{name}_gpus{world_size}"


def checkpoint_path(c, name, seed, smoke=False):
    return run_directory(c, name, seed, smoke) / "checkpoint"


def model_config(c):
    return Pi0Config(
        pi05=True,
        action_horizon=c["action_horizon"],
        action_dim=c["action_dim"],
        max_token_len=c["max_token_len"],
        discrete_state_input=True,
        pytorch_compile_mode=None,
    )


def unwrap(model):
    """The DistillationPolicy inside a DistributedDataParallel wrapper, if any."""
    return (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )


def latest_resume_step(directory):
    """Step of the newest complete resume checkpoint written by OpenPI, else 0."""
    root = directory / "upstream" / "atom_mtd_offlinedata" / directory.name
    steps = [int(d.name) for d in root.glob("*") if d.is_dir() and d.name.isdigit()]
    return max(steps, default=0)


@torch.no_grad()
def task_probe(model, dataset, c):
    mode = model.training
    device = next(model.parameters()).device
    values = []
    model.eval()
    try:
        with torch.random.fork_rng(
            devices=[device.index] if device.type == "cuda" else []
        ):
            torch.manual_seed(probe_seed(c))
            rng = np.random.default_rng(probe_seed(c))
            cache_key = (c["batch_size"], c["validation_batches"])
            if getattr(dataset, "_probe_cache_key", None) != cache_key:
                dataset._probe_cache = [
                    batch(
                        dataset, rng.integers(len(dataset), size=c["batch_size"] // 2)
                    )
                    for _ in range(c["validation_batches"])
                ]
                dataset._probe_cache_key = cache_key
            # Keep fixed CPU batches so intermediate checks avoid reloading frames.
            # Reset after loading too, so cached and fresh probes use identical noise.
            torch.manual_seed(probe_seed(c))
            for metadata, actions in dataset._probe_cache:
                metadata = jax.tree.map(lambda x: x.to(device), metadata)
                student, _, _ = model.compare(
                    metadata, actions.to(device), with_teachers=False
                )
                values.append(float(student["task_loss"].mean()))
    finally:
        model.train(mode)
    return float(np.mean(values))


def train(c, variant, seed, smoke=False, benchmark_steps=None):
    """Train one variant. benchmark_steps: time that many updates; no W&B or results."""
    if variant not in STAGE_A + STAGE_B:
        raise ValueError("Unsupported variant (there is no Joint BC continuation)")
    benchmark = benchmark_steps is not None
    rank, world_size = distributed_rank()  # from torchrun's environment
    is_main = rank == 0
    directory = (
        benchmark_directory(c, variant, world_size)
        if benchmark
        else run_directory(c, variant, seed, smoke)
    )
    if (directory / "result.json").exists():
        raise FileExistsError(f"Completed result already exists: {directory}")
    if benchmark and is_main and directory.exists():
        shutil.rmtree(directory)
    is_b = variant in STAGE_B
    schedule = c["stage_b" if is_b else "stage_a"]
    assert c["stage_b"]["peak_lr"] < c["stage_a"]["peak_lr"]
    initialization = checkpoint_path(c, "joint_bc", seed, smoke) if is_b else None
    teachers = (
        {t: checkpoint_path(c, "teacher_" + t, seed, smoke) for t in ("i1", "i5")}
        if is_b
        else {}
    )
    if benchmark and is_b:
        # Timing only: untrained base-plus-LoRA teachers cost the same as trained ones.
        initialization, teachers = None, {"i1": None, "i5": None}
    if is_b and not benchmark:
        for name in STAGE_A:
            result = json.loads(
                (run_directory(c, name, seed, smoke) / "result.json").read_text()
            )
            if not result["trusted"]:
                raise RuntimeError(
                    f"Stage A {name} did not pass train/validation decrease checks"
                )
            if result["config_hash"] != config_hash(c):
                saved = json.loads(
                    (run_directory(c, name, seed, smoke) / "config.json").read_text()
                )
                if not matches_run_config(saved, c, seed):
                    raise RuntimeError(
                        "Stage A config changed; use a separate output directory"
                    )
    task_names = (
        (variant.removeprefix("teacher_"),)
        if variant.startswith("teacher_")
        else ("i1", "i5")
    )
    dataset = AtomicPairs(c, "train", task_names)
    validation = AtomicPairs(c, "validation", task_names)
    if benchmark:
        steps = benchmark_steps
    elif smoke:
        steps = c["smoke_steps"]
    else:
        steps = schedule["steps"] if is_b else stage_a_steps(c, len(dataset))
    # seed * stride + step is unique per (seed, step) only while step < stride.
    assert steps < STEP_SEED_STRIDE, "STEP_SEED_STRIDE must exceed steps"
    hardware = c[HARDWARE_KEY]
    start_step = 0 if benchmark else latest_resume_step(directory)
    base_hash = sha256(Path(c["base_checkpoint"]) / "model.safetensors")
    init_hash = (
        sha256(initialization / "trainable.safetensors")
        if initialization
        else base_hash
    )
    warmup_steps = max(1, round(steps * schedule["warmup_fraction"]))
    provenance = {
        **c,
        "variant": variant,
        "seed": seed,
        "smoke": smoke,
        "world_size": world_size,
        "train_pairs": len(dataset),
        "total_steps": steps,
        "fixed_probe_interval": 10 if smoke else c["validation_interval"],
        "initialization_checkpoint": str(initialization)
        if initialization
        else c["base_checkpoint"],
        "initialization_sha256": init_hash,
        "base_sha256": base_hash,
        "split_manifest_sha256": sha256(Path(c["dataset_root"]) / "splits.json"),
        "norm_stats_sha256": sha256(Path(c["dataset_root"]) / "norm_stats.json"),
        "teacher_checkpoints": {t: str(p) for t, p in teachers.items() if p},
        "teacher_sha256": {
            t: sha256(p / "trainable.safetensors") for t, p in teachers.items() if p
        },
        "code_sha256": {
            p: sha256(ROOT / p)
            for p in (
                "openpi/scripts/train_pytorch.py",
                "openpi/src/openpi/models_pytorch/pi0_pytorch.py",
                "pi05_lora_sft/lora.py",
                "atom_mtd_offlinedata/model.py",
                "atom_mtd_offlinedata/data.py",
                "atom_mtd_offlinedata/losses.py",
                "atom_mtd_offlinedata/train.py",
            )
        },
        "active_wandb_base_url": tracking.settings(c)["base_url"],
        "warmup_steps": warmup_steps,
        "active_schedule": schedule,
        "config_hash": config_hash(c),
        "derived_seeds": {
            "step_seed_stride": STEP_SEED_STRIDE,
            "probe": probe_seed(c),
            "audit": audit_seed(c),
        },
    }
    if start_step:
        saved = json.loads((directory / "config.json").read_text())
        if not matches_run_config(saved, c, seed) or saved["total_steps"] != steps:
            raise RuntimeError(f"Cannot resume {directory}: its settings changed")
        provenance = saved
    elif is_main:
        write_json(directory / "config.json", provenance)
    spec = importlib.util.spec_from_file_location(
        "atom_openpi_trainer", ROOT / "openpi/scripts/train_pytorch.py"
    )
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    resume_interval = hardware["resume_interval"]
    cfg = TrainConfig(
        name="atom_mtd_offlinedata",
        project_name=c["project"],
        exp_name=directory.name,
        model=model_config(c),
        seed=seed,
        batch_size=c["batch_size"],
        num_train_steps=steps,
        lr_schedule=CosineDecaySchedule(
            warmup_steps=warmup_steps,
            peak_lr=schedule["peak_lr"],
            decay_steps=max(2, steps - 1),
            decay_lr=schedule["end_lr"],
        ),
        optimizer=AdamW(**{k: v for k, v in c["optimizer"].items() if k != "name"}),
        checkpoint_base_dir=str(directory / "upstream"),
        ema_decay=None,
        log_interval=1,
        save_interval=resume_interval,
        resume=bool(start_step),
        wandb_enabled=not benchmark,
    )
    run_holder, probes, step_times = {}, {}, []

    def init_wandb(*args, **kwargs):
        if benchmark:
            return
        previous = (
            json.loads((directory / "wandb.json").read_text()) if start_step else {}
        )
        run = tracking.initialize(
            c, directory.name, directory, provenance, run_id=previous.get("id")
        )
        run_holder.update(
            run=run,
            id=run.id,
            path=run.path if isinstance(run.path, str) else "/".join(run.path),
            url=tracking.run_url(c, run),
        )
        write_json(
            directory / "wandb.json",
            {k: v for k, v in run_holder.items() if k != "run"},
        )

    def ready(model):
        policy = unwrap(model)
        if start_step:
            probes.update(json.loads((directory / "initial_probes.json").read_text()))
        elif not benchmark:
            if is_main:
                probes["initial_train"] = task_probe(policy, dataset, c)
                probes["initial_validation"] = task_probe(policy, validation, c)
            if is_b:
                device = next(policy.parameters()).device
                # Audit both specialists BEFORE the first update, on fixed train pairs.
                # Every rank runs it, which also builds that rank's frozen teachers.
                try:
                    with torch.no_grad(), torch.random.fork_rng(devices=[device.index]):
                        torch.manual_seed(audit_seed(c))
                        for task in ("i1", "i5"):
                            ix = next(
                                i for i, p in enumerate(dataset.pairs) if p[0] == task
                            )
                            metadata, actions = batch(dataset, [ix])
                            policy.compare(
                                jax.tree.map(lambda x: x.to(device), metadata),
                                actions.to(device),
                            )
                finally:
                    if is_main:
                        write_json(
                            directory / "representation_audit.json",
                            policy.representation_audit,
                        )
            if is_main:
                write_json(directory / "initial_probes.json", probes)
        if is_b and not benchmark:
            assert policy.initialization_sha256 == init_hash

    def on_step(model, step, lr, grad_norm):
        policy = unwrap(model)
        losses = dict(policy.last_metrics)
        if world_size > 1:
            # Logged losses are global-batch means, like the averaged gradients.
            values = torch.tensor(list(losses.values()), device="cuda")
            dist.all_reduce(values)
            losses = dict(zip(losses, (values / world_size).tolist()))
        if not np.isfinite(grad_norm):
            raise FloatingPointError("Nonfinite gradients")
        if not is_main:
            return
        step_times.append(time.perf_counter())
        metrics = {
            **losses,
            "train/learning_rate": lr,
            "train/grad_norm": grad_norm,
            "optimizer_step": step,
        }
        if benchmark:
            return
        if step == 1:
            write_json(
                directory / "first_batch.json",
                {
                    "metrics": metrics,
                    "tensor_shapes": policy.tensor_shapes,
                    "lora": policy.lora_info,
                    "initialization_sha256": init_hash,
                    "world_size": world_size,
                },
            )
        if step % provenance["fixed_probe_interval"] == 0 or step == steps:
            metrics["validation/loss_task"] = task_probe(policy, validation, c)
            metrics["probe/train_loss_task"] = task_probe(policy, dataset, c)
            if step == steps:
                probes["final_train"] = metrics["probe/train_loss_task"]
                probes["final_validation"] = metrics["validation/loss_task"]
        # Upstream commits at this same zero-based step; no backwards W&B steps.
        wandb.log(metrics, step=step - 1, commit=False)
        with (directory / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics) + "\n")

    def save(model, optimizer, step, config, is_main, data_config):
        if not benchmark and step % resume_interval == 0 and step < steps:
            # Full student + optimizer state for resuming; keep only the newest.
            upstream.save_checkpoint(
                model, optimizer, step, config, is_main, data_config
            )
            if is_main:
                for old in config.checkpoint_dir.glob("*"):
                    if old.is_dir() and old.name.isdigit() and int(old.name) < step:
                        shutil.rmtree(old)
        if step != steps or not is_main:
            return
        if benchmark:
            gaps = np.diff(step_times)[min(5, len(step_times) // 2) :]
            write_json(
                directory / "benchmark.json",
                {
                    "variant": variant,
                    "world_size": world_size,
                    "steps": steps,
                    "seconds_per_step": float(np.median(gaps)),
                    "batch_size": c["batch_size"],
                },
            )
            return
        checkpoint = directory / "checkpoint"
        checkpoint.mkdir(exist_ok=True)
        tensors = {
            n: p.detach().cpu().contiguous()
            for n, p in unwrap(model).named_parameters()
            if p.requires_grad
        }
        safetensors.torch.save_file(tensors, str(checkpoint / "trainable.safetensors"))
        write_json(checkpoint / "provenance.json", provenance)
        trusted = (
            probes["final_train"] < probes["initial_train"]
            and probes["final_validation"] < probes["initial_validation"]
        )
        write_json(
            directory / "training_result.json",
            {
                "variant": variant,
                "seed": seed,
                "probes": probes,
                "trusted": trusted,
                "config_hash": config_hash(c),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint / "trainable.safetensors"),
            },
        )

    def make_model(m):
        policy = DistillationPolicy(m, c, variant, seed, initialization, teachers)
        policy.update_index = start_step  # step-addressed tau/noise continue on resume
        return policy

    upstream.train_loop(
        cfg,
        model_factory=make_model,
        loader_factory=lambda _: (
            PairLoader(
                dataset,
                c["batch_size"],
                seed,
                steps,
                start_step=start_step,
                workers=hardware["loader_workers"],
            ),
            DataConfig(norm_stats=dataset.stats, asset_id="atom_franka"),
        ),
        on_step=on_step,
        on_ready=ready,
        wandb_initializer=init_wandb,
        checkpoint_saver=save,
    )
    if not is_main:
        return None
    if benchmark:
        return json.loads((directory / "benchmark.json").read_text())
    tracking.verify_history(c, run_holder["path"], "train/loss_total")
    result = json.loads((directory / "training_result.json").read_text())
    result.update(wandb_url=run_holder["url"], wandb_verified=True)
    write_json(directory / "result.json", result)
    # Resume checkpoints (full student + optimizer) are only needed until completion.
    shutil.rmtree(directory / "upstream", ignore_errors=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--variant", choices=STAGE_A + STAGE_B, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--benchmark-steps", type=int, help="time updates only (benchmark.py)"
    )
    args = p.parse_args()
    train(
        read_config(args.config),
        args.variant,
        args.seed,
        args.smoke,
        benchmark_steps=args.benchmark_steps,
    )
