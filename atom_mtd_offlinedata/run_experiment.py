"""Stage orchestration only. Every training job uses OpenPI's train_loop.

Schedules the run for the configured hardware (launch/schedule.py), builds the frame cache on
every node, then runs Stage A, Stage B and evaluation through the configured
launcher (local processes or SLURM job steps). Finished jobs are skipped, so a
rerun continues where a previous one stopped.

    python -m atom_mtd_offlinedata.run_experiment --config <cfg> --phase smoke
    python -m atom_mtd_offlinedata.run_experiment --config <cfg> --phase main [--dry-run]
"""

import argparse
import json
import subprocess
import sys

from .common import (
    DEFAULT_CONFIG,
    HARDWARE_KEY,
    ROOT,
    config_hash,
    matches_run_config,
    output_root,
    read_config,
    write_json,
)
from .launch import make_launcher
from .model import STAGE_A, STAGE_B
from .launch.schedule import describe_schedule, make_schedule
from .train import run_directory


def job_args(config_path, job, seed, smoke):
    module = (
        "atom_mtd_offlinedata.train"
        if job["kind"] == "train"
        else "atom_mtd_offlinedata.evaluate"
    )
    args = [
        module,
        "--config",
        str(config_path),
        "--variant",
        job["variant"],
        "--seed",
        str(seed),
    ]
    return args + (["--smoke"] if smoke else [])


def job_done(c, job, seed, smoke):
    name = "result.json" if job["kind"] == "train" else "evaluation.json"
    return (run_directory(c, job["variant"], seed, smoke) / name).exists()


def check_training(c, variants, seed, smoke):
    for variant in variants:
        directory = run_directory(c, variant, seed, smoke)
        result = json.loads((directory / "result.json").read_text())
        saved = json.loads((directory / "config.json").read_text())
        if not matches_run_config(saved, c, seed) or not result["wandb_verified"]:
            raise RuntimeError(f"Stale or unverified run: {directory}")
        if variant in STAGE_A and not result["trusted"]:
            raise RuntimeError(
                f"{variant} losses did not decrease; result recorded, do not trust this checkpoint"
            )


def check_smoke_gate(c):
    gate = output_root(c) / "smoke_passed.json"
    approval = json.loads(gate.read_text())
    if not approval["passed"]:
        raise RuntimeError("The current config has not passed the smoke protocol")
    if approval["config_hash"] != config_hash(c):
        # A smoke may have started before future seed repetitions were removed.
        # Check its complete per-run settings and actual seed.
        seed = c["training_seeds"][0]
        if approval["training_seeds"] != [seed]:
            raise RuntimeError("Smoke seed differs from the current protocol")
        for variant in STAGE_A + STAGE_B:
            saved = json.loads(
                (run_directory(c, variant, seed, True) / "config.json").read_text()
            )
            if not matches_run_config(saved, c, seed):
                raise RuntimeError("Training settings changed since the smoke protocol")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--phase", choices=("smoke", "main"), required=True)
    p.add_argument(
        "--dry-run", action="store_true", help="print the schedule and commands only"
    )
    args = p.parse_args()
    c = read_config(args.config)
    smoke = args.phase == "smoke"
    if not smoke and len(c["training_seeds"]) != 1:
        raise ValueError("The current protocol trains one matched seed")
    launcher = make_launcher(c, args.dry_run)
    schedule = make_schedule(c, smoke)
    print(describe_schedule(schedule), flush=True)
    phase_root = output_root(c) / args.phase
    if c[HARDWARE_KEY].get("frame_cache_dir"):
        # Before preflight, which checks the cache; skipped where already complete.
        launcher.run_on_every_node(
            ["atom_mtd_offlinedata.cache_frames", "--config", str(args.config)]
        )
    if not args.dry_run:
        write_json(phase_root / "schedule.json", schedule)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "atom_mtd_offlinedata.preflight",
                "--config",
                str(args.config),
                "--wandb-smoke",
            ],
            check=True,
            cwd=ROOT,
        )
        if not smoke:
            check_smoke_gate(c)
    for seed in c["training_seeds"][:1] if smoke else c["training_seeds"]:
        for phase in schedule["phases"]:
            jobs = [
                j
                for j in phase["jobs"]
                if args.dry_run or not job_done(c, j, seed, smoke)
            ]
            print(f"\n== {phase['name']}: {len(jobs)} job(s) to run", flush=True)
            launcher.run_jobs(
                jobs,
                lambda j: job_args(args.config, j, seed, smoke),
                phase_root / "logs",
            )
            if args.dry_run:
                continue
            if phase["name"] == "stage_a":
                check_training(c, STAGE_A, seed, smoke)
            elif phase["name"] == "stage_b":
                check_training(c, STAGE_B, seed, smoke)
                inits = {
                    json.loads(
                        (run_directory(c, v, seed, smoke) / "config.json").read_text()
                    )["initialization_sha256"]
                    for v in STAGE_B
                }
                assert len(inits) == 1, "Stage B initialization mismatch"
    if args.dry_run:
        return
    if smoke:
        write_json(
            output_root(c) / "smoke_passed.json",
            {
                "passed": True,
                "config_hash": config_hash(c),
                "training_seeds": c["training_seeds"][:1],
            },
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "atom_mtd_offlinedata.evaluate",
            "--config",
            str(args.config),
            "--summarize",
        ]
        + (["--smoke"] if smoke else []),
        check=True,
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
