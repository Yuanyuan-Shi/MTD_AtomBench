#!/usr/bin/env bash
# One SLURM allocation for a whole phase (cache build, Stage A, Stage B, evaluation).
# Submit from the repository root on the shared filesystem:
#   sbatch atom_mtd_offlinedata/launch/slurm_experiment.sbatch            # PHASE=main
#   PHASE=smoke sbatch atom_mtd_offlinedata/launch/slurm_experiment.sbatch
#   MODE=benchmark sbatch atom_mtd_offlinedata/launch/slurm_experiment.sbatch
# --nodes and --gpus-per-node must match hardware.nodes / gpus_per_node in the config.
#SBATCH --job-name=atom-mtd
#SBATCH --nodes=3
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=atom_mtd_offlinedata/reports/slurm-%x-%j.out
set -euo pipefail
source atom_mtd_offlinedata/env.sh
CONFIG="${CONFIG:-atom_mtd_offlinedata/configs/aws.json}"
if [[ "${MODE:-experiment}" == benchmark ]]; then
  srun --overlap --nodes="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
    "$ATOM_PYTHON" -m atom_mtd_offlinedata.cache_frames --config "$CONFIG"
  "$ATOM_PYTHON" -m atom_mtd_offlinedata.benchmark --config "$CONFIG"
else
  "$ATOM_PYTHON" -m atom_mtd_offlinedata.run_experiment --config "$CONFIG" --phase "${PHASE:-main}"
fi
