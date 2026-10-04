"""Stage-A pi05 LoRA SFT using the copied OpenPI trainer and ATOM loader."""

import argparse
from atom_mtd_offlinedata.common import DEFAULT_CONFIG, read_config
from atom_mtd_offlinedata.train import train

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument(
        "--variant", choices=("teacher_i1", "teacher_i5", "joint_bc"), required=True
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    train(read_config(args.config), args.variant, args.seed, args.smoke)
