"""Local launcher: every job is a plain process on this machine."""

import sys

from . import Launcher


class LocalLauncher(Launcher):
    """One machine; GPUs selected with CUDA_VISIBLE_DEVICES."""

    def command(self, job, index, module_args):
        env = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, job["gpu_ids"]))}
        return self._torchrun(job, index, module_args), env

    def every_node_command(self, argv):
        return [sys.executable, "-m", *argv]
