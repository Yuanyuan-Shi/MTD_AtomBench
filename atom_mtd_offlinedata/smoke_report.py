"""Plot recorded smoke metrics; optionally follow an existing orchestration PID."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import DEFAULT_CONFIG, ROOT, output_root, read_config, write_json

VARIANTS = (
    "teacher_i1",
    "teacher_i5",
    "joint_bc",
    "output_mtd",
    "output_repr_mtd",
    "output_repr_temp_mtd",
)


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def render(c):
    seed = c["training_seeds"][0]
    report_dir = ROOT / "atom_mtd_offlinedata/reports"
    report_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    probes_fig, probes_axes = plt.subplots(
        2, 3, figsize=(15, 8), constrained_layout=True
    )
    rows = {}
    for variant, ax, probe_ax in zip(
        VARIANTS, axes.flat, probes_axes.flat, strict=True
    ):
        directory = output_root(c) / "smoke" / f"{variant}_seed{seed}"
        history = []
        if (directory / "metrics.jsonl").exists():
            for line in (directory / "metrics.jsonl").read_text().splitlines():
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # A concurrent writer may not yet have finished its last line.
        result = read_json(directory / "result.json")
        training_result = read_json(directory / "training_result.json")
        probes = {
            **read_json(directory / "initial_probes.json"),
            **training_result.get("probes", {}),
        }
        steps = [h["optimizer_step"] for h in history]
        losses = np.array([h["train/loss_total"] for h in history])
        state = (
            "complete"
            if result
            else "training complete; finalizing"
            if training_result
            else "running"
            if history
            else "pending"
        )
        row = {"state": state, "updates": max(steps, default=0), "probes": probes}
        if len(losses):
            ax.plot(steps, losses, alpha=0.4, label="Raw batch total")
            rolling = [
                float(losses[max(0, i - 4) : i + 1].mean()) for i in range(len(losses))
            ]
            ax.plot(steps, rolling, label="Trailing 5-batch mean")
            row.update(
                first_5_mean=float(losses[:5].mean()),
                last_5_mean=float(losses[-5:].mean()),
            )
            ax.legend(fontsize=8)
        for split, metric, color in (
            ("train", "probe/train_loss_task", "tab:blue"),
            ("validation", "validation/loss_task", "tab:orange"),
        ):
            points = {}
            if "initial_" + split in probes:
                points[0] = probes["initial_" + split]
            points.update(
                {h["optimizer_step"]: h[metric] for h in history if metric in h}
            )
            if "final_" + split in probes:
                points[c["smoke_steps"]] = probes["final_" + split]
            if points:
                xs = sorted(points)
                probe_ax.plot(
                    xs, [points[x] for x in xs], "o-", label=split, color=color
                )
                probe_ax.legend(fontsize=8)
        for chart in (ax, probe_ax):
            chart.set_title(f"{variant} · {state}", fontsize=10)
            chart.set_xlabel("Optimizer updates")
            chart.set_ylabel("Loss")
            chart.set_xlim(0, c["smoke_steps"])
            chart.grid(alpha=0.2)
        row["wandb"] = read_json(directory / "wandb.json").get("url")
        row["trusted"] = result.get("trusted", training_result.get("trusted"))
        rows[variant] = row
    fig.suptitle(f"Offline smoke test · seed {seed} · recorded batch losses")
    probes_fig.suptitle("Fixed-data, fixed-noise task loss · measured points only")
    fig.savefig(report_dir / "training_loss_curves.png", dpi=140)
    probes_fig.savefig(report_dir / "fixed_probe_curves.png", dpi=140)
    plt.close(fig)
    plt.close(probes_fig)
    gate = read_json(output_root(c) / "smoke_passed.json")
    write_json(
        report_dir / "smoke_status.json",
        {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "smoke_passed": gate.get("passed", False),
            "runs": rows,
        },
    )
    lines = [
        "# Smoke test progress",
        "",
        "One seed; 30 updates per model. No full experiment is launched by this report.",
        "",
        "| Model | State | Updates | First 5 mean | Last 5 mean | Fixed train initial → final | Fixed validation initial → final |",
        "|---|---|---:|---:|---:|---|---|",
    ]

    def value(x):
        return "pending" if x is None else f"{x:.6f}"

    for variant, row in rows.items():
        p = row["probes"]
        lines.append(
            f"| {variant} | {row['state']} | {row['updates']} | {value(row.get('first_5_mean'))} | {value(row.get('last_5_mean'))} | {value(p.get('initial_train'))} → {value(p.get('final_train'))} | {value(p.get('initial_validation'))} → {value(p.get('final_validation'))} |"
        )
    lines += [
        "",
        "Batch means use the first/last up to five recorded updates. For short histories these windows overlap. Fixed probes use the same observations and flow noise, without training augmentation. Only measured probe points are plotted. The first already-running teacher has initial/final probes; later jobs also probe every ten updates.",
        "",
        "Loss decay supports a learning sanity check; routing, masks, gradients, noise sharing and checkpoint assertions provide separate correctness checks.",
    ]
    (report_dir / "smoke_status.md").write_text("\n".join(lines) + "\n")


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--watch-pid", type=int)
    args = parser.parse_args()
    config = read_config(args.config)
    while True:
        render(config)
        if not args.watch_pid or not alive(args.watch_pid):
            break
        time.sleep(30)
