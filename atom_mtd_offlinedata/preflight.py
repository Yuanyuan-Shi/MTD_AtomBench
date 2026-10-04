import argparse
import json
from pathlib import Path

import torch

from .common import DEFAULT_CONFIG, HARDWARE_KEY, ROOT, read_config, sha256, write_json
from . import tracking


def check(c, *, remote_smoke=False):
    report = {"checks": {}, "blockers": []}

    def attempt(name, fn):
        try:
            report["checks"][name] = fn()
        except Exception as exc:
            # Exceptions can contain auth data from external libraries: record only controlled messages.
            report["checks"][name] = {
                "status": "failed",
                "exception": type(exc).__name__,
            }
            if name == "wandb_endpoint" and isinstance(exc, RuntimeError):
                report["checks"][name]["reason"] = str(exc)
            report["blockers"].append(name)

    def cuda():
        x = torch.randn(8, 8, device="cuda", requires_grad=True)
        x.square().mean().backward()
        torch.cuda.synchronize()
        return {
            "status": "passed",
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
        }

    def dataset():
        p = Path(c["dataset_root"]) / "splits.json"
        data = json.loads(p.read_text())
        for task in ("i1", "i5"):
            sets = [
                set(data["splits"][task][s]) for s in ("train", "validation", "test")
            ]
            assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        return {
            "status": "passed",
            "prompts": data["prompts"],
            "revision": data["revision"],
        }

    def checkpoint():
        p = Path(c["base_checkpoint"]) / "model.safetensors"
        assert p.is_file(), f"{p} missing; run prepare.py"
        assert sha256(p) == c["base_checkpoint_sha256"], (
            "base checkpoint SHA256 mismatch"
        )
        return {"status": "passed", "path": str(p), "bytes": p.stat().st_size}

    def frame_cache():
        from .cache_frames import cache_root, load_manifest

        if not c[HARDWARE_KEY].get("frame_cache_dir"):
            return {
                "status": "passed",
                "mode": "video decoding (no frame cache configured)",
            }
        manifest = load_manifest(c)
        assert manifest is not None, f"no complete frame cache at {cache_root(c)}"
        return {
            "status": "passed",
            "path": str(cache_root(c)),
            "frames": manifest["frames"],
        }

    attempt("cuda", cuda)
    attempt("dataset", dataset)
    attempt("base_checkpoint", checkpoint)
    if (
        c[HARDWARE_KEY]["launcher"] == "local"
    ):  # on SLURM the cache is built per node first
        attempt("frame_cache", frame_cache)
    attempt(
        "wandb_endpoint", lambda: (tracking.probe_endpoint(c) or {"status": "passed"})
    )
    if remote_smoke and "wandb_endpoint" not in report["blockers"]:

        def smoke():
            run = tracking.initialize(
                c, "infrastructure_smoke", ROOT / "atom_mtd_offlinedata", c
            )
            run.log(
                {
                    "smoke/cuda_verified": int("cuda" not in report["blockers"]),
                    "smoke/logging_probe": 1,
                }
            )
            path, url = run.path, tracking.run_url(c, run)
            run.finish()
            tracking.verify_history(c, path, "smoke/logging_probe")
            return {"status": "passed", "url": url}

        attempt("wandb_remote_metrics", smoke)
    report["status"] = "blocked" if report["blockers"] else "passed"
    write_json(ROOT / "atom_mtd_offlinedata/reports/preflight.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--wandb-smoke", action="store_true")
    args = p.parse_args()
    result = check(read_config(args.config), remote_smoke=args.wandb_smoke)
    raise SystemExit(int(result["status"] != "passed"))
