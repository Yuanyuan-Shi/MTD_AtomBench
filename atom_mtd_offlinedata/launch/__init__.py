"""Running the experiment's jobs on hardware.

launch/__init__.py  shared Launcher base class and make_launcher (picks by hardware.launcher)
launch/local.py     LocalLauncher: plain processes on this machine
launch/slurm.py     SlurmLauncher: srun job steps inside one SLURM allocation
launch/schedule.py  which job runs on which nodes/GPUs, in what order (fastest schedule)
launch/measure_speed.py  seconds per training step per GPU count, used by schedule.py

Launchers run a phase's jobs in schedule order: a job starts as soon as every
earlier job that shares one of its GPUs has finished, which reproduces the
schedule simulated by schedule.py. Multi-GPU jobs run under torchrun.
"""

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from ..common import HARDWARE_KEY, ROOT

POLL_SECONDS = 10
RENDEZVOUS_PORT = 29500  # + job index, so concurrent multi-node jobs never collide


class Launcher:
    def __init__(self, c, dry_run=False):
        self.c, self.dry_run = c, dry_run
        self.per_node = c[HARDWARE_KEY]["gpus_per_node"]

    def command(self, job, index, module_args):
        raise NotImplementedError

    def every_node_command(self, argv):
        raise NotImplementedError

    def _torchrun(self, job, index, module_args, *, rendezvous_host=None):
        python = [sys.executable]
        if job["gpus"] == 1 and len(job["nodes"]) == 1:
            return python + ["-m", *module_args]
        distributed = [
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(len(job["gpu_ids"])),
        ]
        if len(job["nodes"]) == 1:
            distributed += ["--standalone"]
        else:
            distributed += [
                "--nnodes",
                str(len(job["nodes"])),
                "--rdzv_backend",
                "c10d",
                "--rdzv_endpoint",
                f"{rendezvous_host}:{RENDEZVOUS_PORT + index}",
                "--rdzv_id",
                f"atom-{job['variant']}-{job['kind']}",
            ]
        return python + distributed + ["-m", *module_args]

    def run_on_every_node(self, argv):
        command = self.every_node_command(argv)
        print("$", shlex.join(command), flush=True)
        if not self.dry_run:
            subprocess.run(command, check=True, cwd=ROOT)

    def run_jobs(self, jobs, args_for, log_dir):
        """Run jobs (dicts from schedule.py) in schedule order; raise if any job fails."""
        order = sorted(jobs, key=lambda j: (j["start_s"], j["nodes"], j["gpu_ids"]))
        slots = [{(n, g) for n in j["nodes"] for g in j["gpu_ids"]} for j in order]
        commands = [self.command(j, i, args_for(j)) for i, j in enumerate(order)]
        if self.dry_run:
            for job, (argv, env) in zip(order, commands):
                visible = env.get("CUDA_VISIBLE_DEVICES", "")
                print(f"$ CUDA_VISIBLE_DEVICES={visible} {shlex.join(argv)}")
            return
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        state = ["pending"] * len(order)  # pending -> running -> done
        running = {}
        try:
            while "pending" in state or running:
                for i, job in enumerate(order):
                    blocked = any(
                        state[k] != "done" and slots[k] & slots[i] for k in range(i)
                    )
                    if state[i] != "pending" or blocked:
                        continue
                    argv, env = commands[i]
                    log = Path(log_dir) / f"{job['variant']}_{job['kind']}.log"
                    print(
                        f"start {job['variant']} ({job['kind']}) on nodes {job['nodes']} "
                        f"GPUs {job['gpu_ids']}; log {log}",
                        flush=True,
                    )
                    handle = log.open("a")
                    running[i] = (
                        subprocess.Popen(
                            argv,
                            cwd=ROOT,
                            env={**os.environ, **env},
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                        ),
                        handle,
                    )
                    state[i] = "running"
                for i, (process, handle) in list(running.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    handle.close()
                    del running[i]
                    if code:
                        raise RuntimeError(
                            f"{order[i]['variant']} ({order[i]['kind']}) failed with exit code {code}"
                        )
                    state[i] = "done"
                    print(
                        f"done  {order[i]['variant']} ({order[i]['kind']})", flush=True
                    )
                if running:
                    time.sleep(POLL_SECONDS)
        finally:
            for process, handle in running.values():
                process.terminate()
                handle.close()


def make_launcher(c, dry_run=False):
    kind = c[HARDWARE_KEY]["launcher"]
    if kind == "local":
        from .local import LocalLauncher

        return LocalLauncher(c, dry_run)
    if kind == "slurm":
        from .slurm import SlurmLauncher

        return SlurmLauncher(c, dry_run)
    raise ValueError(f"Unknown hardware.launcher {kind!r}")
