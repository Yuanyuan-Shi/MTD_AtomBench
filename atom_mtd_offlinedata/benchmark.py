"""Measure seconds per training step at each usable GPU count, for plan.py.

Times a few dozen updates of a Stage-A and a Stage-B variant (Stage B with
untrained teachers, which cost the same) through the normal training path, with
no W&B or results. Writes hardware.benchmark_file, which plan.py then prefers
over the hardware.step_seconds estimates. Run inside the same allocation the
experiment will use (e.g. the sbatch job on AWS).
"""

import argparse
import json

from .common import DEFAULT_CONFIG, HARDWARE_KEY, write_json, read_config
from .launch import make_launcher
from .plan import gpu_options
from .train import benchmark_directory

PROBES = {"stage_a": "joint_bc", "stage_b": "output_repr_temp_mtd"}


def placement(c, gpus):
    per_node = c[HARDWARE_KEY]["gpus_per_node"]
    if gpus >= per_node:
        return list(range(gpus // per_node)), list(range(per_node))
    return [0], list(range(gpus))


def run(c, config_path, steps, dry_run=False):
    launcher = make_launcher(c, dry_run)
    seed = c["training_seeds"][0]
    results = {}
    for cost_key, variant in PROBES.items():
        results[cost_key] = {}
        options = gpu_options(c, "train")
        # Whole-node counts are what the plan uses for training; others extrapolate.
        for gpus in [
            g for g in options if g >= c[HARDWARE_KEY]["gpus_per_node"]
        ] or options[-1:]:
            nodes, gpu_ids = placement(c, gpus)
            job = {
                "variant": variant,
                "kind": f"benchmark{gpus}",
                "gpus": gpus,
                "nodes": nodes,
                "gpu_ids": gpu_ids,
                "start_s": 0.0,
            }
            args = [
                "atom_mtd_offlinedata.train",
                "--config",
                str(config_path),
                "--variant",
                variant,
                "--seed",
                str(seed),
                "--benchmark-steps",
                str(steps),
            ]
            launcher.run_jobs(
                [job],
                lambda _: args,
                benchmark_directory(c, variant, gpus).parent / "logs",
            )
            if dry_run:
                continue
            measured = json.loads(
                (benchmark_directory(c, variant, gpus) / "benchmark.json").read_text()
            )
            results[cost_key][str(gpus)] = measured["seconds_per_step"]
            print(
                f"{cost_key} on {gpus} GPU(s): {measured['seconds_per_step']:.3f} s/step",
                flush=True,
            )
    if not dry_run:
        write_json(
            c[HARDWARE_KEY]["benchmark_file"],
            {**results, "steps": steps, "batch_size": c["batch_size"]},
        )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    run(read_config(args.config), args.config, args.steps, args.dry_run)
