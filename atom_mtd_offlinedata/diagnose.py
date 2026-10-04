"""Real-data forward/backward diagnostic; no optimizer update, no trained claims."""

import argparse
from pathlib import Path

import jax
import torch

from .common import DEFAULT_CONFIG, ROOT, read_config, sha256, write_json
from .data import AtomicPairs, batch
from .model import build_policy
from .train import model_config


def diagnose(c, observations):
    torch.manual_seed(c["training_seeds"][0])
    ds = AtomicPairs(c, "validation")
    candidates = [
        next(i for i, p in enumerate(ds.pairs) if p[0] == t) for t in ("i1", "i5")
    ]
    indices = [candidates[i % 2] for i in range(observations // 2)]
    metadata, actions = batch(ds, indices)
    metadata = jax.tree.map(lambda x: x.cuda(), metadata)
    actions = actions.cuda()
    model = build_policy(model_config(c), c).cuda().train()
    model.gradient_checkpointing_enable()
    tau = model.sample_time(actions.shape[0] // 2, actions.device).repeat_interleave(2)
    noise = model.sample_noise(actions.shape, actions.device)
    features = model.forward_features(
        metadata["observation"], actions, noise=noise, time=tau, train=False
    )
    expected = (noise - actions - features["flow"]).square()
    torch.testing.assert_close(features["task_loss"], expected, rtol=0, atol=0)
    loss = features["task_loss"].mean()
    loss.backward()
    grads = {
        n: p.grad
        for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }
    assert grads and all(torch.isfinite(g).all() for g in grads.values())
    assert any("lora_" in n and g.abs().sum() > 0 for n, g in grads.items())
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None
    result = {
        "status": "passed",
        "kind": "untrained base-plus-LoRA forward/backward; no optimizer updates",
        "observations": observations,
        "task_loss": float(loss.detach()),
        "shapes": {
            k: list(features[k].shape) for k in ("flow", "img", "lang", "action")
        },
        "representation_requires_grad": {
            k: features[k].requires_grad for k in ("img", "lang", "action")
        },
        "language_kd_tokens": metadata["language_mask"].sum(1).tolist(),
        "pair_tasks": metadata["task"].tolist(),
        "pair_episodes": metadata["episode"].tolist(),
        "pair_timesteps": metadata["timestep"].tolist(),
        "same_tau": torch.equal(tau[::2], tau[1::2]),
        "independent_noise_not_equal": not torch.equal(noise[::2], noise[1::2]),
        "lora": model.lora_info,
        "gradient_tensors": len(grads),
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "base_sha256": sha256(Path(c["base_checkpoint"]) / "model.safetensors"),
    }
    write_json(ROOT / "atom_mtd_offlinedata/reports/model_diagnostic.json", result)
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--observations", type=int, default=4)
    args = p.parse_args()
    if args.observations < 2 or args.observations % 2:
        p.error("--observations must be positive and even")
    diagnose(read_config(args.config), args.observations)
