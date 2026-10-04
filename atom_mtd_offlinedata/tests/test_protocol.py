import pytest
import torch

from atom_mtd_offlinedata.losses import (
    assert_pairs,
    distillation_losses,
    representation_error,
    temporal_error,
)
from pi05_lora_sft.lora import LoRALinear


def test_temporal_target_is_teacher_residual_not_smoothing():
    teacher = torch.tensor([[[[0.0], [3.0], [7.0]], [[1.0], [2.0], [4.0]]]])
    student = (teacher + 19).requires_grad_()
    # Nonzero temporal changes are perfectly correct when they match the teacher.
    assert temporal_error(student, teacher, execution_horizon=1) == 0
    assert (student[:, 0, 1:] - student[:, 1, :-1]).square().mean() > 0
    wrong = student.clone()
    wrong[:, 1] += 2
    loss = temporal_error(wrong, teacher, execution_horizon=1)
    assert loss == 4
    loss.backward()
    assert student.grad is not None


def test_masked_normalization_ignores_padding_and_scale():
    student = torch.tensor([[[3.0, 4.0], [900.0, -900.0]]], requires_grad=True)
    teacher = torch.tensor([[[6.0, 8.0], [-100.0, 900.0]]])
    mask = torch.tensor([[True, False]])
    loss = representation_error(student, teacher, mask)
    assert float(loss) < 1e-12
    loss.backward()
    assert torch.equal(student.grad[:, 1], torch.zeros_like(student.grad[:, 1]))
    with pytest.raises(ValueError):
        representation_error(student, teacher, torch.zeros_like(mask))


@pytest.mark.parametrize("field", ["task", "episode", "timestep"])
def test_temporal_boundary_rejection(field):
    data = {
        "task": torch.tensor([[1, 1]]),
        "episode": torch.tensor([[9, 9]]),
        "timestep": torch.tensor([[20, 21]]),
    }
    assert_pairs(**data, execution_horizon=1)
    data[field][0, 1] += 1
    with pytest.raises(ValueError):
        assert_pairs(**data, execution_horizon=1)


def test_temporal_error_compares_overlap_of_shifted_chunks():
    # Pair queried 2 steps apart: chunk_t[2:] and chunk_{t+2}[:-2] cover the same steps.
    teacher = torch.randn(3, 2, 6, 4)
    student = teacher + torch.randn(3, 1, 1, 4)  # same offset in both chunks
    assert temporal_error(student, teacher, execution_horizon=2) < 1e-12
    student[:, 1, :4] += 1.0  # chunk_{t+2} overlap off by 1 everywhere
    torch.testing.assert_close(
        temporal_error(student, teacher, execution_horizon=2), torch.tensor(1.0)
    )
    student[:, 1, 4:] += 5.0  # non-overlapping tail is not compared
    torch.testing.assert_close(
        temporal_error(student, teacher, execution_horizon=2), torch.tensor(1.0)
    )
    for bad in (0, 6):
        with pytest.raises(ValueError):
            temporal_error(student, teacher, execution_horizon=bad)


def test_pairs_must_be_execution_horizon_apart():
    data = {
        "task": torch.tensor([[1, 1]]),
        "episode": torch.tensor([[9, 9]]),
        "timestep": torch.tensor([[20, 25]]),
    }
    assert_pairs(**data, execution_horizon=5)
    with pytest.raises(ValueError):
        assert_pairs(**data, execution_horizon=1)


def test_weighting_and_gradients():
    s = {
        k: torch.randn(4, 3, 5, requires_grad=True)
        for k in ("flow", "img", "lang", "action")
    }
    t = {k: torch.randn_like(v) for k, v in s.items()}
    masks = {k: torch.ones(4, 3, dtype=torch.bool) for k in ("img", "lang", "action")}
    c = {
        "lambda_out": 0.5,
        "lambda_repr": 0.1,
        "lambda_temp": 0.1,
        "repr_eps": 1e-8,
        "execution_horizon": 1,
    }
    losses = distillation_losses(s, t, masks, c, repr_enabled=True, temp_enabled=True)
    assert torch.equal(
        losses["repr"], sum(losses[k] for k in ("img", "lang", "action"))
    )
    total = sum(losses["weighted_" + k] for k in ("output", "repr", "temp"))
    torch.testing.assert_close(
        total, 0.5 * losses["output"] + 0.1 * losses["repr"] + 0.1 * losses["temp"]
    )
    total.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in s.values())
    assert all(v.grad is None for v in t.values())


@pytest.mark.parametrize(
    "kind,heads,inputs,outputs",
    [("qkv", 2, 5, 6), ("out", 2, 6, 5), ("dense", 1, 5, 6)],
)
def test_lora_matches_explicit_openpi_einsum_weight(kind, heads, inputs, outputs):
    base = torch.nn.Linear(inputs, outputs, bias=False)
    layer = LoRALinear(base, rank=2, alpha=2, heads=heads, kind=kind)
    a, b = layer.lora_a, layer.lora_b
    if kind == "qkv":
        update = torch.einsum("ndr,nrh->nhd", a, b).reshape(outputs, inputs)
    elif kind == "out":
        update = torch.einsum("nhr,nrd->dnh", a, b).reshape(outputs, inputs)
    else:
        update = (a @ b).T
    x = torch.randn(3, 4, inputs, requires_grad=True)
    torch.testing.assert_close(
        layer(x),
        torch.nn.functional.linear(x, base.weight + update),
        atol=1e-6,
        rtol=1e-6,
    )
    layer(x).square().mean().backward()
    assert base.weight.grad is None
    assert a.grad is not None and b.grad is not None


def test_stage_b_protocol_is_matched_and_no_continuation():
    from atom_mtd_offlinedata.common import read_config
    from atom_mtd_offlinedata.model import STAGE_B

    c = read_config()
    assert c["training_seeds"] == [7]
    assert c["stage_b"]["peak_lr"] < c["stage_a"]["peak_lr"]
    assert set(c["tasks"]) == {"i1", "i5"}
    assert STAGE_B == ("output_mtd", "output_repr_mtd", "output_repr_temp_mtd")


def test_fixed_probes_cache_preserves_noise_and_training_rng():
    from unittest.mock import patch
    from atom_mtd_offlinedata.train import task_probe

    class Dataset:
        def __len__(self):
            return 20

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def compare(self, metadata, actions, with_teachers):
            assert not self.training and not with_teachers
            return (
                {"task_loss": (actions + torch.randn_like(actions)).square()},
                None,
                None,
            )

    def decode(dataset, ids):
        torch.rand(1)  # Even decoding consuming RNG must not change probe noise.
        return {"ids": torch.tensor(ids)}, torch.tensor(ids, dtype=torch.float32)

    model, dataset = Model(), Dataset()
    c = {"batch_size": 4, "validation_batches": 2, "data_seed": 20261001}
    before = torch.get_rng_state().clone()
    with patch("atom_mtd_offlinedata.train.batch", side_effect=decode) as get_batch:
        fresh = task_probe(model, dataset, c)
        cached = task_probe(model, dataset, c)
        assert get_batch.call_count == 2
    assert fresh == cached
    assert torch.equal(before, torch.get_rng_state())
    assert model.training


def test_seed_repetition_change_does_not_allow_training_config_change():
    from atom_mtd_offlinedata.common import matches_run_config

    saved = {"seed": 7, "training_seeds": [7, 17, 27], "batch_size": 32}
    current = {"training_seeds": [7], "batch_size": 32}
    assert matches_run_config(saved, current, 7)
    assert not matches_run_config(saved, {**current, "batch_size": 16}, 7)
    assert not matches_run_config(saved, current, 17)


def test_pair_noise_contract_with_actual_comparison_code():
    """Spy teachers verify routing, shared augmented input, tau and independent noise."""
    from types import SimpleNamespace
    from unittest.mock import patch
    from atom_mtd_offlinedata.model import DistillationPolicy

    class Toy(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset
            self.calls = []

        def forward_features(self, observation, actions, noise, time, processed):
            self.calls.append(
                (observation, noise.clone(), time.clone(), torch.is_grad_enabled())
            )
            flow = noise + self.offset
            hidden = torch.stack([flow[..., 0], flow[..., 1] + 1], dim=-1)
            mask = torch.ones(hidden.shape[:2], dtype=torch.bool)
            return {
                "flow": flow,
                "img": hidden,
                "lang": hidden,
                "action": hidden,
                "task_loss": flow.square(),
                "img_mask": mask,
                "lang_mask": mask,
                "action_mask": mask,
                "noisy_actions": time[:, None, None] * noise
                + (1 - time[:, None, None]) * actions,
                "noise": noise,
                "time": time,
            }

    toy = Toy(0)
    toy.sample_time = lambda n, d: torch.rand(n, device=d)
    toy.sample_noise = lambda shape, device: torch.randn(shape, device=device)
    teachers = {1: Toy(1).eval(), 5: Toy(5).eval()}
    toy.get_teacher = lambda task, device: teachers[task]
    toy.representation_audit = {}
    toy.experiment = {"repr_eps": 1e-8, "execution_horizon": 1}
    toy.variant = "joint_bc"
    obs = SimpleNamespace(
        images={"image": torch.arange(4)[:, None]},
        image_masks={"image": torch.ones(4, dtype=torch.bool)},
        state=torch.zeros(4, 2),
        tokenized_prompt=torch.zeros(4, 3, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(4, 3, dtype=torch.bool),
        token_ar_mask=None,
        token_loss_mask=None,
    )
    metadata = {
        "observation": obs,
        "task": torch.tensor([[1, 1], [5, 5]]),
        "episode": torch.tensor([[0, 0], [0, 0]]),
        "timestep": torch.tensor([[2, 3], [7, 8]]),
        "language_mask": torch.ones(4, 3, dtype=torch.bool),
    }
    with patch(
        "atom_mtd_offlinedata.model.preprocess_observation_pytorch",
        side_effect=lambda o, train: o,
    ) as preprocess:
        s, t, _ = DistillationPolicy.compare(
            toy, metadata, torch.zeros(4, 3, 2), with_teachers=True
        )
        assert preprocess.call_count == 1
    assert torch.equal(s["time"][::2], s["time"][1::2])
    assert not torch.equal(s["noise"][::2], s["noise"][1::2])
    torch.testing.assert_close(t["flow"][:2], s["flow"][:2] + 1)
    torch.testing.assert_close(t["flow"][2:], s["flow"][2:] + 5)
    for i, task in enumerate((1, 5)):
        observation, noise, tau, grad_enabled = teachers[task].calls[0]
        assert not grad_enabled
        assert torch.equal(noise, s["noise"][2 * i : 2 * i + 2])
        assert torch.equal(
            observation.images["image"], obs.images["image"][2 * i : 2 * i + 2]
        )


def test_wrap_angles_keeps_orientation_and_maps_into_window():
    import numpy as np

    from atom_mtd_offlinedata.common import wrap_angles

    pi = np.pi
    raw = np.array([[pi], [-pi], [3.15], [-3.13], [2.9], [7.0]], dtype=np.float32)
    for center in (pi, 0.0, -pi / 2):
        out = wrap_angles(raw, [0], [center])[:, 0]
        assert np.all((out > center - pi - 1e-6) & (out <= center + pi + 1e-6))
        # Same orientation: only multiples of 2*pi were added.
        assert np.allclose(np.cos(out), np.cos(raw[:, 0]), atol=1e-6)
        assert np.allclose(np.sin(out), np.sin(raw[:, 0]), atol=1e-5)
    near_pi = wrap_angles(raw, [0], [pi])[:, 0]
    assert abs(near_pi[0] - near_pi[1]) < 1e-6  # +pi and -pi: one value (float32 ulp)
    assert near_pi[2] == raw[2, 0] and near_pi[4] == raw[4, 0]  # inside: bit-identical
    assert abs(near_pi[3] - 3.153185) < 1e-5  # -3.13 lands next to 3.15


def test_angle_centers_follow_the_data_and_snap_to_quarter_turns():
    import numpy as np

    from atom_mtd_offlinedata.common import angle_centers

    rng = np.random.default_rng(0)
    roll_near_pi = np.concatenate(
        [rng.uniform(2.9, 3.14, 500), rng.uniform(-3.14, -3.0, 300)]
    )
    data = np.stack(
        [roll_near_pi, rng.normal(-0.08, 0.1, 800), rng.normal(-1.5, 0.1, 800)], axis=1
    )
    assert angle_centers(data, [0, 1, 2]) == pytest.approx([np.pi, 0.0, -np.pi / 2])


def test_angle_margin_check_rejects_angles_sweeping_the_full_circle():
    import numpy as np

    from atom_mtd_offlinedata.common import check_angle_margin

    spinning = np.random.default_rng(0).uniform(-np.pi, np.pi, (1000, 1))
    with pytest.raises(ValueError, match="sin/cos"):
        check_angle_margin(spinning, [0], [0.0])
    check_angle_margin(np.full((10, 1), 0.3), [0], [0.0])


def test_to_physical_actions_inverts_openpi_quantile_normalization():
    import numpy as np
    from openpi import transforms
    from openpi.shared.normalize import NormStats

    from atom_mtd_offlinedata.data import to_physical_actions

    rng = np.random.default_rng(0)
    q01, q99 = rng.uniform(-1, 0, 14), rng.uniform(1, 2, 14)
    q01[11], q99[11] = 2.95, 3.29  # rx window around pi
    stats = NormStats(mean=np.zeros(14), std=np.ones(14), q01=q01, q99=q99)
    actions = rng.uniform(q01, q99, (50, 14))
    normalized = transforms.Normalize({"actions": stats}, use_quantiles=True)(
        {"actions": actions}
    )["actions"]
    padded = np.concatenate([normalized, np.zeros((50, 18))], axis=1)
    back = to_physical_actions(padded, stats, 14, [11, 12, 13])
    assert np.allclose(back, actions, atol=1e-6)
    robot = to_physical_actions(
        padded, stats, 14, [11, 12, 13], output_angle_center=0.0
    )
    assert np.all(robot[:, 11] <= np.pi) and np.all(robot[:, 11] > -np.pi)
    assert np.allclose(np.cos(robot[:, 11]), np.cos(actions[:, 11]), atol=1e-6)
