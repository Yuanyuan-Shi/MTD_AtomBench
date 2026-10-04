import json
from pathlib import Path

import jax
import torch
import safetensors.torch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch

from pi05_lora_sft.lora import install_lora
from .common import STEP_SEED_STRIDE, derive_seed, distributed_rank, sha256
from .losses import assert_pairs, distillation_losses, representation_error

STAGE_A = ("teacher_i1", "teacher_i5", "joint_bc")
STAGE_B = ("output_mtd", "output_repr_mtd", "output_repr_temp_mtd")


def restore_trainable(model, directory, c):
    directory = Path(directory)
    meta = json.loads((directory / "provenance.json").read_text())
    if meta["base_sha256"] != sha256(Path(c["base_checkpoint"]) / "model.safetensors"):
        raise ValueError("Checkpoint was trained from a different base")
    if meta["lora"] != c["lora"]:
        raise ValueError("Checkpoint LoRA configuration mismatch")
    state = safetensors.torch.load_file(directory / "trainable.safetensors")
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError("Trainable checkpoint keys mismatch")
    model.load_state_dict(state, strict=False)
    # Exact equality check after loading; never accept a partial Joint BC restore.
    for name, value in state.items():
        if not torch.equal(model.get_parameter(name).detach().cpu(), value):
            raise ValueError(f"Checkpoint tensor did not load exactly: {name}")
    return sha256(directory / "trainable.safetensors")


def build_policy(model_config, c, checkpoint=None):
    model = PI0Pytorch(model_config)
    safetensors.torch.load_model(
        model, str(Path(c["base_checkpoint"]) / "model.safetensors"), strict=True
    )
    model.lora_info = install_lora(model, c["lora"])
    if checkpoint:
        model.initialization_sha256 = restore_trainable(model, checkpoint, c)
    return model


class DistillationPolicy(PI0Pytorch):
    def __init__(
        self, model_config, c, variant, seed, initialization=None, teacher_paths=None
    ):
        torch.manual_seed(seed)
        super().__init__(model_config)
        self.experiment, self.variant, self.seed = c, variant, seed
        self.update_index = 0
        safetensors.torch.load_model(
            self, str(Path(c["base_checkpoint"]) / "model.safetensors"), strict=True
        )
        self.lora_info = install_lora(self, c["lora"])
        self.initialization_sha256 = (
            restore_trainable(self, initialization, c) if initialization else None
        )
        # Plain dict intentionally excludes teachers from student optimizer/checkpoints.
        self.teachers = {}
        self.teacher_paths = teacher_paths or {}
        self.representation_audit = {}
        self.last_metrics = {}

    def get_teacher(self, task, device):
        if task not in (1, 5):
            raise ValueError("There is no composition teacher")
        if task not in self.teachers:
            teacher = build_policy(
                self.config, self.experiment, self.teacher_paths[f"i{task}"]
            )
            teacher.requires_grad_(False).eval().to(device)
            self.teachers[task] = teacher
        teacher = self.teachers[task]
        assert not teacher.training and all(
            not p.requires_grad for p in teacher.parameters()
        )
        return teacher

    def compare(
        self,
        metadata,
        actions,
        *,
        augment=False,
        with_teachers=True,
        tau=None,
        noise=None,
    ):
        """Matched student/teacher features; tau/noise are sampled here unless given."""
        assert_pairs(
            metadata["task"],
            metadata["episode"],
            metadata["timestep"],
            execution_horizon=self.experiment["execution_horizon"],
        )
        tasks = metadata["task"].flatten()
        if not torch.isin(tasks, torch.tensor([1, 5], device=tasks.device)).all():
            raise ValueError(
                "Atomic training/evaluation only; x1 needs teacher-free evaluation"
            )
        observation = preprocess_observation_pytorch(
            metadata["observation"], train=augment
        )
        # One tau per pair; separate Gaussian draws for every [observation,H,D].
        if tau is None:
            tau = self.sample_time(
                actions.shape[0] // 2, actions.device
            ).repeat_interleave(2)
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if tau.shape != actions.shape[:1] or noise.shape != actions.shape:
            raise ValueError("tau/noise do not match the batch")
        assert torch.equal(tau[0::2], tau[1::2])
        if torch.equal(noise[0::2], noise[1::2]):
            raise RuntimeError("Temporal noise was unexpectedly tied")
        student = self.forward_features(
            observation, actions, noise=noise, time=tau, processed=True
        )
        masks = {k: student[k + "_mask"] for k in ("img", "lang", "action")}
        masks["lang"] = masks["lang"] & metadata["language_mask"]
        if not with_teachers:
            return student, None, masks
        teacher_result = {
            k: torch.empty_like(student[k]) for k in ("flow", "img", "lang", "action")
        }
        with torch.no_grad():
            for task in tasks.unique().tolist():
                indices = (tasks == task).nonzero().flatten()
                teacher = self.get_teacher(task, actions.device)
                # Slice exactly the same already augmented tensors; never re-augment.
                fields = {
                    k: getattr(observation, k)
                    for k in (
                        "images",
                        "image_masks",
                        "state",
                        "tokenized_prompt",
                        "tokenized_prompt_mask",
                        "token_ar_mask",
                        "token_loss_mask",
                    )
                }
                from types import SimpleNamespace

                obs = SimpleNamespace(**jax.tree.map(lambda x: x[indices], fields))
                target = teacher.forward_features(
                    obs,
                    actions[indices],
                    noise=noise[indices],
                    time=tau[indices],
                    processed=True,
                )
                assert torch.equal(
                    target["noisy_actions"], student["noisy_actions"][indices]
                )
                assert torch.equal(target["noise"], student["noise"][indices])
                for key in teacher_result:
                    teacher_result[key][indices] = target[key]
                if task not in self.representation_audit:
                    values = {
                        k: float(
                            representation_error(
                                student[k][indices].detach(),
                                target[k],
                                masks[k][indices],
                                self.experiment["repr_eps"],
                            )
                        )
                        for k in ("img", "lang", "action")
                    }
                    self.representation_audit[task] = values
                    if self.variant in STAGE_B and any(v <= 0 for v in values.values()):
                        raise RuntimeError(
                            f"Meaningless/identical representation for teacher i{task}: {values}"
                        )
        assert all(not x.requires_grad for x in teacher_result.values())
        return student, teacher_result, masks

    def step_randomness(self, actions):
        """This rank's rows of the global-batch tau/noise for the current update.

        Draws tau (one per pair) and noise for the whole global batch from the
        per-update seed and slices this rank's rows, so any GPU count gets the same
        values. Then reseeds per rank for the augmentation that follows.
        """
        rank, world_size = distributed_rank()
        local = actions.shape[0]
        rows = slice(rank * local, (rank + 1) * local)
        step_seed = self.seed * STEP_SEED_STRIDE + self.update_index
        torch.manual_seed(step_seed)
        tau = self.sample_time(local * world_size // 2, actions.device)
        tau = tau.repeat_interleave(2)[rows]
        noise = self.sample_noise(
            (local * world_size, *actions.shape[1:]), actions.device
        )
        torch.manual_seed(derive_seed(step_seed, rank))
        return tau, noise[rows]

    def forward(self, metadata, actions, noise=None, time=None):
        if noise is not None or time is not None:
            raise ValueError("Experiment samples matched pair noise/tau internally")
        # Matched per-update RNG independent of lazy teacher initialization and
        # validation. tau/noise are drawn for the global batch and sliced, so every
        # GPU count sees the same values; augmentation is drawn per GPU.
        with torch.random.fork_rng(
            devices=[actions.device.index] if actions.is_cuda else []
        ):
            tau, noise = self.step_randomness(actions)
            student, teacher, masks = self.compare(
                metadata,
                actions,
                augment=self.training,
                with_teachers=self.variant in STAGE_B,
                tau=tau,
                noise=noise,
            )
        task_loss = student["task_loss"].mean()
        if teacher is None:
            losses = {
                k: task_loss.new_zeros(())
                for k in (
                    "output",
                    "img",
                    "lang",
                    "action",
                    "repr",
                    "temp",
                    "weighted_output",
                    "weighted_repr",
                    "weighted_temp",
                )
            }
        else:
            losses = distillation_losses(
                student,
                teacher,
                masks,
                self.experiment,
                repr_enabled=self.variant != "output_mtd",
                temp_enabled=self.variant == "output_repr_temp_mtd",
            )
        total = task_loss + sum(
            losses["weighted_" + k] for k in ("output", "repr", "temp")
        )
        self.last_metrics = {
            "train/loss_task": float(task_loss.detach()),
            "train/loss_total": float(total.detach()),
        }
        self.last_metrics.update(
            {
                "train/" + (k if k.startswith("weighted_") else "loss_" + k): float(
                    v.detach()
                )
                for k, v in losses.items()
            }
        )
        self.tensor_shapes = {
            k: list(student[k].shape) for k in ("flow", "img", "lang", "action")
        }
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite training objective")
        self.update_index += 1
        return total
