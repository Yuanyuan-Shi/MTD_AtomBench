"""All KD reductions in FP32. Task loss is the unchanged OpenPI reduction."""

import torch


def mse(a, b):
    return (a.float() - b.float()).square().mean()


def representation_error(student, teacher, mask, eps=1e-8):
    if student.shape != teacher.shape or mask.shape != student.shape[:-1]:
        raise ValueError("Representation/mask shape mismatch")
    if not mask.any():
        raise ValueError("No valid representation tokens")
    s, t = student.float(), teacher.float()
    s = s / (s.norm(dim=-1, keepdim=True) + eps)
    t = t / (t.norm(dim=-1, keepdim=True) + eps)
    return (s - t).square()[mask].mean()


def temporal_error(student, teacher, *, execution_horizon):
    # [pairs, 2 queries execution_horizon steps apart, horizon, action_dim]
    if (
        student.shape != teacher.shape
        or student.ndim != 4
        or student.shape[1] != 2
        or not 1 <= execution_horizon < student.shape[2]
    ):
        raise ValueError("Expected [P,2,H,D] with 1 <= execution_horizon < H")
    shift = execution_horizon  # chunk_t and chunk_{t+shift} overlap on H - shift steps
    rs = student[:, 0, shift:] - student[:, 1, :-shift]
    rt = teacher[:, 0, shift:] - teacher[:, 1, :-shift]
    return mse(rs, rt)


def assert_pairs(task, episode, timestep, *, execution_horizon):
    if task.ndim != 2 or task.shape[1] != 2:
        raise ValueError("Pair metadata must have shape [P,2]")
    if not torch.equal(episode[:, 0], episode[:, 1]):
        raise ValueError("Temporal pair crosses episodes")
    if not torch.equal(task[:, 0], task[:, 1]):
        raise ValueError("Temporal pair crosses tasks")
    if not torch.equal(timestep[:, 1], timestep[:, 0] + execution_horizon):
        raise ValueError("Temporal pair is not execution_horizon steps apart")


def distillation_losses(
    student, teacher, masks, coefficients, *, repr_enabled, temp_enabled
):
    zero = student["flow"].new_zeros((), dtype=torch.float32)
    losses = {key: zero for key in ("img", "lang", "action", "repr", "temp")}
    losses["output"] = mse(student["flow"], teacher["flow"])
    if repr_enabled:
        for key in ("img", "lang", "action"):
            losses[key] = representation_error(
                student[key], teacher[key], masks[key], coefficients["repr_eps"]
            )
        losses["repr"] = losses["img"] + losses["lang"] + losses["action"]
    if temp_enabled:
        shape = (-1, 2, *student["flow"].shape[1:])
        losses["temp"] = temporal_error(
            student["flow"].reshape(shape),
            teacher["flow"].reshape(shape),
            execution_horizon=coefficients["execution_horizon"],
        )
    for key in ("output", "repr", "temp"):
        coefficient = coefficients["lambda_out" if key == "output" else "lambda_" + key]
        losses["weighted_" + key] = coefficient * losses[key]
    return losses
