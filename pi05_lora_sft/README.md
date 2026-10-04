# Local pi0.5 LoRA SFT

This directory supplies the PyTorch equivalent of the copied OpenPI LoRA
configuration and the three authorized Stage-A jobs: `teacher_i1`, `teacher_i5`,
and `joint_bc`. It calls `../openpi/scripts/train_pytorch.py` through the experiment
configuration; it does not define a separate optimizer or training loop.

See [the offline experiment](../atom_mtd_offlinedata/README.md) for exact data,
settings, status and commands. `reference/` preserves the pre-existing JAX
full-parameter SFT script/config for inspection; those are not LoRA scripts and
are not invoked by this experiment.
