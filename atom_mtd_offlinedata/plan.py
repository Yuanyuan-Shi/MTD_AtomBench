"""Allocate training/evaluation jobs to nodes and GPUs for the shortest wall-clock time.

Phases run in order (Stage A, Stage B, evaluation) because each needs the previous
phase's checkpoints. Within a phase, every combination of job order and GPU count
is simulated and the shortest schedule is kept. GPU counts are those that split
the global batch evenly and either fit inside one node or use whole nodes.
Seconds per step come from benchmark.py when measured, else from the estimates
in hardware.step_seconds (scaled by GPU count, with an efficiency loss for
multi-node jobs).

    python -m atom_mtd_offlinedata.plan --config atom_mtd_offlinedata/configs/aws.json
"""

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .common import DEFAULT_CONFIG, HARDWARE_KEY, read_config, stage_a_steps
from .model import STAGE_A, STAGE_B

EVALUATED = STAGE_A + STAGE_B
BRUTE_FORCE_LIMIT = (
    200_000  # simulated schedules per phase before falling back to greedy
)


@dataclass
class Job:
    variant: str
    kind: str  # "train" or "eval"
    cost_key: str  # "stage_a", "stage_b" or "eval_model"
    steps: int  # optimizer steps (training) or 1 (evaluation)
    gpus: int = 0
    nodes: list = field(default_factory=list)  # node indices in the allocation
    gpu_ids: list = field(default_factory=list)  # GPU indices on those nodes
    start_s: float = 0.0
    end_s: float = 0.0


def train_pair_count(c, tasks):
    """Valid (t, t + execution_horizon) training pairs, from metadata only."""
    root = Path(c["dataset_root"])
    splits = json.loads((root / "splits.json").read_text())["splits"]
    total = 0
    for task in tasks:
        meta = root / c["tasks"][task] / "meta/episodes.jsonl"
        lengths = {
            json.loads(line)["episode_index"]: json.loads(line)["length"]
            for line in meta.read_text().splitlines()
        }
        total += sum(
            max(0, lengths[ep] - c["action_horizon"] - c["execution_horizon"] + 1)
            for ep in splits[task]["train"]
        )
    return total


def phase_jobs(c, smoke):
    def steps_a(variant):
        if smoke:
            return c["smoke_steps"]
        tasks = (
            (variant.removeprefix("teacher_"),)
            if variant.startswith("teacher_")
            else ("i1", "i5")
        )
        return stage_a_steps(c, train_pair_count(c, tasks))

    stage_b = c["smoke_steps"] if smoke else c["stage_b"]["steps"]
    return [
        ("stage_a", [Job(v, "train", "stage_a", steps_a(v)) for v in STAGE_A]),
        ("stage_b", [Job(v, "train", "stage_b", stage_b) for v in STAGE_B]),
        ("evaluation", [Job(v, "eval", "eval_model", 1) for v in EVALUATED]),
    ]


def gpu_options(c, kind="train"):
    """GPU counts that split the batch evenly and fit one node or use whole nodes.

    Training jobs also keep at most hardware.max_train_obs_per_gpu observations per
    GPU (memory: Stage B holds the student plus two frozen teachers).
    """
    hw = c[HARDWARE_KEY]
    per_node, nodes = hw["gpus_per_node"], hw["nodes"]
    pairs = c["batch_size"] // 2
    options = [
        g for g in range(1, per_node + 1) if per_node % g == 0 and pairs % g == 0
    ]
    options += [
        k * per_node for k in range(2, nodes + 1) if pairs % (k * per_node) == 0
    ]
    if kind == "train":
        fits = [
            g for g in options if c["batch_size"] / g <= hw["max_train_obs_per_gpu"]
        ]
        if not fits:
            raise ValueError(
                "No GPU count keeps the batch within max_train_obs_per_gpu"
            )
        options = fits
    return options


def load_benchmarks(c):
    path = c[HARDWARE_KEY].get("benchmark_file")
    if not path or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text())


def seconds_per_step(c, cost_key, gpus, benchmarks):
    hw = c[HARDWARE_KEY]
    measured = benchmarks.get(cost_key, {})
    if str(gpus) in measured:
        return measured[str(gpus)]
    if measured:  # extrapolate from the nearest measured GPU count
        ref_gpus = min(measured, key=lambda g: abs(int(g) - gpus))
        ref_seconds, ref_gpus = measured[ref_gpus], int(ref_gpus)
    else:
        ref_seconds = hw["step_seconds"][cost_key]
        ref_gpus = hw["step_seconds"]["reference_gpus"]
    seconds = ref_seconds * ref_gpus / gpus
    per_node = hw["gpus_per_node"]
    if gpus > per_node >= ref_gpus:
        seconds /= hw["internode_efficiency"]
    return seconds


def simulate(c, jobs, counts, benchmarks):
    """List-schedule jobs (in order) with the given GPU counts; returns makespan."""
    hw = c[HARDWARE_KEY]
    per_node = hw["gpus_per_node"]
    free_at = [[0.0] * per_node for _ in range(hw["nodes"])]
    placed = []
    for job, gpus in zip(jobs, counts):
        duration = job.steps * seconds_per_step(c, job.cost_key, gpus, benchmarks)
        if gpus >= per_node:  # whole nodes, the earliest-free ones
            k = gpus // per_node
            ready = sorted(range(len(free_at)), key=lambda n: max(free_at[n]))[:k]
            start = max(max(free_at[n]) for n in ready)
            nodes, gpu_ids = sorted(ready), list(range(per_node))
        else:  # within one node: the earliest-free GPUs of the best node
            best = None
            for n, times in enumerate(free_at):
                ids = sorted(range(per_node), key=lambda i: times[i])[:gpus]
                start_n = max(times[i] for i in ids)
                if best is None or start_n < best[0]:
                    best = (start_n, n, sorted(ids))
            start, node, gpu_ids = best
            nodes = [node]
        for n in nodes:
            for i in gpu_ids:
                free_at[n][i] = start + duration
        placed.append(
            Job(
                **{
                    **asdict(job),
                    "gpus": gpus,
                    "nodes": nodes,
                    "gpu_ids": gpu_ids,
                    "start_s": start,
                    "end_s": start + duration,
                }
            )
        )
    return max(j.end_s for j in placed), placed


def schedule_phase(c, jobs, benchmarks):
    options = gpu_options(c, jobs[0].kind)
    combos = math.factorial(len(jobs)) * len(options) ** len(jobs)
    best = None
    if combos <= BRUTE_FORCE_LIMIT:
        for order in itertools.permutations(jobs):
            for counts in itertools.product(options, repeat=len(jobs)):
                makespan, placed = simulate(c, order, counts, benchmarks)
                if best is None or makespan < best[0] - 1e-9:
                    best = (makespan, placed)
    else:  # greedy: equal GPU share per job, longest job first
        total = c[HARDWARE_KEY]["nodes"] * c[HARDWARE_KEY]["gpus_per_node"]
        share = max(g for g in options if g <= max(1, total // len(jobs)))
        order = sorted(
            jobs,
            key=lambda j: -j.steps * seconds_per_step(c, j.cost_key, share, benchmarks),
        )
        best = simulate(c, order, [share] * len(order), benchmarks)
    return best


def make_plan(c, smoke=False):
    benchmarks = load_benchmarks(c)
    phases, total = [], 0.0
    for name, jobs in phase_jobs(c, smoke):
        makespan, placed = schedule_phase(c, jobs, benchmarks)
        phases.append(
            {"name": name, "makespan_s": makespan, "jobs": [asdict(j) for j in placed]}
        )
        total += makespan
    hw = c[HARDWARE_KEY]
    return {
        "nodes": hw["nodes"],
        "gpus_per_node": hw["gpus_per_node"],
        "smoke": smoke,
        "timing_source": "benchmark"
        if benchmarks
        else "hardware.step_seconds estimates",
        "phases": phases,
        "total_s": total,
    }


def describe(plan):
    lines = [
        f"{plan['nodes']} node(s) x {plan['gpus_per_node']} GPU(s); timing from "
        f"{plan['timing_source']}; estimated total {plan['total_s'] / 3600:.1f} h"
    ]
    for phase in plan["phases"]:
        lines.append(f"\n{phase['name']}: ~{phase['makespan_s'] / 3600:.2f} h")
        for j in sorted(phase["jobs"], key=lambda j: (j["start_s"], j["nodes"])):
            lines.append(
                f"  {j['variant']:22s} {j['gpus']:3d} GPU(s) on node(s) {j['nodes']} "
                f"GPUs {j['gpu_ids']}  {j['start_s'] / 3600:5.2f} h -> {j['end_s'] / 3600:5.2f} h"
                + (f"  ({j['steps']} steps)" if j["kind"] == "train" else "")
            )
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    print(describe(make_plan(read_config(args.config), args.smoke)))
