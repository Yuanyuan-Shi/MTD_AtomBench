"""A small, runnable lesson in multi-teacher knowledge distillation."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


torch.manual_seed(7)
np.random.seed(7)
torch.set_num_threads(1)  # Keep this tiny CPU experiment predictable and quick.

HIDDEN = 32
BATCH_SIZE = 128
LAMBDA_KD = 0.5
LAMBDA_REPR = 0.1


# =========================================================
# 1. Generate synthetic demonstrations
# =========================================================
def make_demonstrations(skill_id, count):
    observations = 2 * torch.rand(count, 2) - 1  # x, y in [-1, 1]
    x, y = observations[:, 0], observations[:, 1]

    if skill_id == 0:  # Pick: mostly move with x and y.
        ax = x + 0.5 * y + 0.2 * torch.sin(torch.pi * x)
        ay = y - 0.25 * x + 0.15 * torch.cos(torch.pi * y)
    else:  # Push: rotate the direction and add a different curve.
        ax = -y + 0.2 * x + 0.2 * torch.sin(torch.pi * y)
        ay = x + 0.4 * y - 0.15 * torch.cos(torch.pi * x)

    actions = torch.stack([ax, ay], dim=1)
    actions += 0.02 * torch.randn_like(actions)  # Small demonstration noise.
    return observations, actions


pick_train_o, pick_train_a = make_demonstrations(0, 1024)
push_train_o, push_train_a = make_demonstrations(1, 1024)
pick_test_o, pick_test_a = make_demonstrations(0, 256)
push_test_o, push_test_a = make_demonstrations(1, 256)

train_o = torch.cat([pick_train_o, push_train_o])
train_a = torch.cat([pick_train_a, push_train_a])
train_skill = torch.cat([torch.zeros(1024), torch.ones(1024)]).long()


# =========================================================
# 2. Small networks and test metric
# =========================================================
class SmallMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, HIDDEN)
        self.layer2 = nn.Linear(HIDDEN, HIDDEN)
        self.action_head = nn.Linear(HIDDEN, 2)

    def forward(self, inputs, return_hidden=False):
        hidden = F.relu(self.layer1(inputs))
        hidden = F.relu(self.layer2(hidden))
        action = self.action_head(hidden)
        return (action, hidden) if return_hidden else action


def student_input(observations, skill_ids):
    skill_one_hot = F.one_hot(skill_ids, num_classes=2).float()
    return torch.cat([observations, skill_one_hot], dim=1)


def student_test_mse(student):
    student.eval()
    with torch.no_grad():
        pick_ids = torch.zeros(len(pick_test_o), dtype=torch.long)
        push_ids = torch.ones(len(push_test_o), dtype=torch.long)
        pick_prediction = student(student_input(pick_test_o, pick_ids))
        push_prediction = student(student_input(push_test_o, push_ids))
        pick_mse = F.mse_loss(pick_prediction, pick_test_a).item()
        push_mse = F.mse_loss(push_prediction, push_test_a).item()
    return pick_mse, push_mse, (pick_mse + push_mse) / 2


# =========================================================
# 3. Train specialist teachers (atomic-policy fine-tuning)
# =========================================================
# Each teacher learns from its own demonstrations, analogous to separately
# fine-tuning one atomic robot policy for pick and one for push.
teachers = []
for name, observations, demonstrations, test_o, test_a in [
    ("Pick", pick_train_o, pick_train_a, pick_test_o, pick_test_a),
    ("Push", push_train_o, push_train_a, push_test_o, push_test_a),
]:
    print(f"\n{'=' * 50}\nTraining {name} Teacher\nTarget: recorded {name.lower()} demonstrations\n{'=' * 50}")
    teacher = SmallMLP(input_dim=2)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=0.01)
    for epoch in range(50):
        teacher.train()
        for indices in torch.randperm(len(observations)).split(BATCH_SIZE):
            prediction = teacher(observations[indices])
            loss = F.mse_loss(prediction, demonstrations[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    teacher.eval()
    with torch.no_grad():
        test_mse = F.mse_loss(teacher(test_o), test_a).item()
    print(f"{name} teacher test MSE: {test_mse:.6f}")
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)  # Distillation must update only students.
    teachers.append(teacher)

pick_teacher, push_teacher = teachers


# =========================================================
# 4. Joint behavior cloning (Joint SFT / Joint BC)
# =========================================================
print(f"\n{'=' * 50}\nJoint BC / Joint SFT\nTarget: recorded pick and push demonstrations\n{'=' * 50}")
joint_student = SmallMLP(input_dim=4)  # [x, y, one-hot pick, one-hot push]
optimizer = torch.optim.Adam(joint_student.parameters(), lr=0.01)
for epoch in range(45):
    joint_student.train()
    for indices in torch.randperm(len(train_o)).split(BATCH_SIZE):
        inputs = student_input(train_o[indices], train_skill[indices])
        student_actions = joint_student(inputs)
        loss_bc = F.mse_loss(student_actions, train_a[indices])
        optimizer.zero_grad()
        loss_bc.backward()
        optimizer.step()
joint_results = student_test_mse(joint_student)
print(f"Joint BC test MSE: pick={joint_results[0]:.6f}, push={joint_results[1]:.6f}, average={joint_results[2]:.6f}")


# =========================================================
# 5. Multi-teacher output distillation
# =========================================================
print(f"\n{'=' * 50}\nMulti-Teacher Output Distillation\nBC target: demonstration action\nKD target: selected specialist teacher\n{'=' * 50}")
output_student = SmallMLP(input_dim=4)
optimizer = torch.optim.Adam(output_student.parameters(), lr=0.01)
for epoch in range(45):
    output_student.train()
    for batch_number, indices in enumerate(torch.randperm(len(train_o)).split(BATCH_SIZE)):
        observations = train_o[indices]
        skill_ids = train_skill[indices]
        demonstration_actions = train_a[indices]

        # Route each sample to the teacher for its known skill ID.
        pick_rows, push_rows = skill_ids == 0, skill_ids == 1
        teacher_actions = torch.empty_like(demonstration_actions)
        teacher_hidden = torch.empty(len(indices), HIDDEN)
        with torch.no_grad():  # Teacher outputs are targets; no teacher gradients.
            teacher_actions[pick_rows], teacher_hidden[pick_rows] = pick_teacher(
                observations[pick_rows], return_hidden=True
            )
            teacher_actions[push_rows], teacher_hidden[push_rows] = push_teacher(
                observations[push_rows], return_hidden=True
            )

        student_actions, student_hidden = output_student(
            student_input(observations, skill_ids), return_hidden=True
        )
        if epoch == 0 and batch_number == 0:
            for label, tensor in [
                ("observations", observations), ("skill_ids", skill_ids),
                ("demonstration_actions", demonstration_actions),
                ("teacher_actions", teacher_actions), ("student_actions", student_actions),
                ("teacher_hidden", teacher_hidden), ("student_hidden", student_hidden),
            ]:
                print(f"{label:23s} shape: {tuple(tensor.shape)}")

        loss_bc = F.mse_loss(student_actions, demonstration_actions)  # Dataset target.
        loss_kd = F.mse_loss(student_actions, teacher_actions)  # Learned teacher target.
        loss_total = loss_bc + LAMBDA_KD * loss_kd
        optimizer.zero_grad()
        loss_total.backward()  # Only output_student has trainable parameters here.
        optimizer.step()
output_results = student_test_mse(output_student)
print(f"Output MTD test MSE: pick={output_results[0]:.6f}, push={output_results[1]:.6f}, average={output_results[2]:.6f}")


# =========================================================
# 6. Optional: also distill hidden representations
# =========================================================
print(f"\n{'=' * 50}\nOutput + Representation Distillation (optional)\nTargets: demonstrations, teacher actions, teacher hidden states\n{'=' * 50}")
repr_student = SmallMLP(input_dim=4)
optimizer = torch.optim.Adam(repr_student.parameters(), lr=0.01)
for epoch in range(45):
    repr_student.train()
    for indices in torch.randperm(len(train_o)).split(BATCH_SIZE):
        observations = train_o[indices]
        skill_ids = train_skill[indices]
        demonstration_actions = train_a[indices]

        # The same per-sample teacher routing is used for actions and hidden states.
        pick_rows, push_rows = skill_ids == 0, skill_ids == 1
        teacher_actions = torch.empty_like(demonstration_actions)
        teacher_hidden = torch.empty(len(indices), HIDDEN)
        with torch.no_grad():
            teacher_actions[pick_rows], teacher_hidden[pick_rows] = pick_teacher(
                observations[pick_rows], return_hidden=True
            )
            teacher_actions[push_rows], teacher_hidden[push_rows] = push_teacher(
                observations[push_rows], return_hidden=True
            )

        student_actions, student_hidden = repr_student(
            student_input(observations, skill_ids), return_hidden=True
        )
        loss_bc = F.mse_loss(student_actions, demonstration_actions)
        loss_kd = F.mse_loss(student_actions, teacher_actions)
        loss_repr = F.mse_loss(student_hidden, teacher_hidden)
        loss_total = loss_bc + LAMBDA_KD * loss_kd + LAMBDA_REPR * loss_repr
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
repr_results = student_test_mse(repr_student)
print(f"Output + representation MTD test MSE: pick={repr_results[0]:.6f}, push={repr_results[1]:.6f}, average={repr_results[2]:.6f}")


# =========================================================
# 7. Compare held-out results
# =========================================================
print("\nTest MSE (lower is better)")
print(f"{'Method':31s} {'Pick':>10s} {'Push':>10s} {'Average':>10s}")
for name, scores in [
    ("Joint BC", joint_results),
    ("Output MTD", output_results),
    ("Output + Representation MTD", repr_results),
]:
    print(f"{name:31s} {scores[0]:10.6f} {scores[1]:10.6f} {scores[2]:10.6f}")

positions = np.arange(2)
width = 0.25
for offset, (name, scores) in enumerate([
    ("Joint BC", joint_results),
    ("Output MTD", output_results),
    ("Output + Representation MTD", repr_results),
]):
    plt.bar(positions + (offset - 1) * width, scores[:2], width, label=name)
plt.xticks(positions, ["Pick", "Push"])
plt.ylabel("Test MSE (lower is better)")
plt.title("One student, two skills")
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.savefig("mtd_results.png", dpi=150)
plt.close()
print("\nSaved plot: mtd_results.png")
