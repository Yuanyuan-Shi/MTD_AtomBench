"""SLURM launcher: every job is an srun step on its planned nodes of one allocation."""

import os
import subprocess
import sys

from ..common import HARDWARE_KEY
from . import Launcher


class SlurmLauncher(Launcher):
    """Job steps (srun --overlap) inside one sbatch allocation of hardware.nodes nodes."""

    def __init__(self, c, dry_run=False):
        super().__init__(c, dry_run)
        nodes = c[HARDWARE_KEY]["nodes"]
        if dry_run and "SLURM_JOB_NODELIST" not in os.environ:
            self.hosts = [f"node{i}" for i in range(nodes)]
        else:
            listing = subprocess.run(
                ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
                check=True,
                capture_output=True,
                text=True,
            )
            self.hosts = listing.stdout.split()
        if len(self.hosts) != nodes:
            raise RuntimeError(
                f"Allocation has {len(self.hosts)} nodes but hardware.nodes is {nodes}"
            )

    def command(self, job, index, module_args):
        hosts = [self.hosts[n] for n in job["nodes"]]
        visible = ",".join(map(str, job["gpu_ids"]))
        srun = [
            "srun",
            "--overlap",
            "--kill-on-bad-exit=1",
            f"--nodes={len(hosts)}",
            f"--ntasks={len(hosts)}",
            "--ntasks-per-node=1",
            f"--gpus-per-node={self.per_node}",
            f"--nodelist={','.join(hosts)}",
            f"--export=ALL,ATOM_GPU_IDS={visible}",
        ]
        # SLURM sets CUDA_VISIBLE_DEVICES for the step; narrow it to this job's GPUs.
        select = [
            "bash",
            "-c",
            'export CUDA_VISIBLE_DEVICES="$ATOM_GPU_IDS"; exec "$@"',
            "_",
        ]
        argv = (
            srun
            + select
            + self._torchrun(job, index, module_args, rendezvous_host=hosts[0])
        )
        return argv, {"CUDA_VISIBLE_DEVICES": visible}

    def every_node_command(self, argv):
        n = len(self.hosts)
        return [
            "srun",
            "--overlap",
            f"--nodes={n}",
            f"--ntasks={n}",
            "--ntasks-per-node=1",
            sys.executable,
            "-m",
            *argv,
        ]
