"""Match OpenPI's per-head einsum LoRA, rather than flattened PEFT ranks.

The upstream freeze filter freezes only non-LoRA llm parameters: SigLIP,
the multimodal projector, action projections and time MLP remain trainable.
"""

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, *, heads=1, kind="dense"):
        super().__init__()
        self.weight = base.weight
        self.bias = base.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.kind, self.heads, self.scale = kind, heads, alpha / rank
        if kind == "qkv":
            shapes = (
                (heads, self.in_features, rank),
                (heads, rank, self.out_features // heads),
            )
        elif kind == "out":
            shapes = (
                (heads, self.in_features // heads, rank),
                (heads, rank, self.out_features),
            )
        else:
            shapes = (self.in_features, rank), (rank, self.out_features)
        self.lora_a = nn.Parameter(
            torch.empty(shapes[0], device=base.weight.device, dtype=torch.float32)
        )
        self.lora_b = nn.Parameter(
            torch.empty(shapes[1], device=base.weight.device, dtype=torch.float32)
        )
        nn.init.normal_(self.lora_a, std=0.01)
        nn.init.normal_(self.lora_b, std=0.01)

    def forward(self, x):
        a, b = self.lora_a.to(x.dtype), self.lora_b.to(x.dtype)
        if self.kind == "qkv":
            update = torch.einsum("...d,ndr,nrh->...nh", x, a, b).flatten(-2)
        elif self.kind == "out":
            update = torch.einsum(
                "...nh,nhr,nrd->...d", x.unflatten(-1, (self.heads, -1)), a, b
            )
        else:
            update = x @ a @ b
        return F.linear(x, self.weight, self.bias) + self.scale * update


def install_lora(model, settings):
    if settings["dropout"] != 0:
        raise ValueError(
            "Upstream OpenPI LoRA has no adapter dropout; only 0 is supported"
        )
    for name, parameter in model.named_parameters():
        is_llm = (
            ".language_model." in name
            or ".gemma_expert." in name
            or ".lm_head." in name
        )
        parameter.requires_grad_(not is_llm and settings["train_non_llm"])
    count = 0
    for name, module in list(model.named_modules()):
        if (
            not isinstance(module, nn.Linear)
            or name.rsplit(".", 1)[-1] not in settings["targets"]
        ):
            continue
        if ".language_model." not in name and ".gemma_expert." not in name:
            continue
        expert = ".gemma_expert." in name
        key = "action_expert" if expert else "paligemma"
        target = name.rsplit(".", 1)[-1]
        kind = (
            "qkv"
            if target in ("q_proj", "k_proj", "v_proj")
            else "out"
            if target == "o_proj"
            else "dense"
        )
        heads = 8 if target in ("q_proj", "o_proj") else 1
        parent_path, leaf = name.rsplit(".", 1)
        replacement = LoRALinear(
            module,
            settings[key + "_rank"],
            settings[key + "_alpha"],
            heads=heads,
            kind=kind,
        )
        setattr(model.get_submodule(parent_path), leaf, replacement)
        count += 1
    if count != 2 * 18 * 7:
        raise ValueError(f"Expected 252 pi05 LoRA projections, found {count}")
    return {
        "projection_count": count,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }
