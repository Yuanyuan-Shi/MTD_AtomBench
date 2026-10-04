# Multi-teacher distillation, in one small PyTorch tutorial

This tutorial consolidates two learned specialists into one student on a tiny
continuous-control problem. “Pick” and “push” are names for two different
synthetic mappings from an observation `[x, y]` to an action `[ax, ay]`.
There is no robot or external dataset.

## Run it

From the project root:

```bash
python mtd_tutorial/tutorial.py
```

The only dependencies are `torch`, `numpy`, and `matplotlib`. The script prints
teacher and student test MSE, shows the shapes of the first distillation batch,
and saves `mtd_results.png` in the directory where you run it.

## 1. Demonstrations and specialist teachers

We sample `x` and `y` uniformly from `[-1, 1]`. Pick and push use different,
slightly nonlinear action functions, then receive a little Gaussian noise.
Each skill has its own train and test data. We train `T_pick` only on pick
demonstrations and `T_push` only on push demonstrations. This resembles
separately fine-tuning atomic robot policies. Their supervised target is the
recorded demonstration action.

All networks have two 32-unit ReLU hidden layers and a 2-D action output.
Teachers receive `[x, y]`. The student receives `[x, y, pick_bit, push_bit]`.
The second hidden layer is returned when we want to compare representations.

## 2. Joint behavior cloning (Joint SFT / Joint BC)

One student sees the union of both demonstration datasets and the skill ID.
It learns directly from the recorded action:

\[
L_{BC} = \|a_{student} - a_{demo}\|^2.
\]

In code, `torch.nn.functional.mse_loss` averages squared differences across
the batch and both action coordinates. This is the baseline: it does not query
teachers.

## 3. Multi-teacher output distillation

Every training sample has a known skill ID, so its teacher is selected by a
simple mask:

```text
pick sample -> T_pick
push sample -> T_push
```

The student still uses the demonstration target for BC. It also tries to
match the selected *learned* teacher's action:

\[
L_{KD} = \|a_{student} - a_{teacher}\|^2,
\qquad
L_{total} = L_{BC} + \lambda_{KD}L_{KD}.
\]

We use `lambda_KD = 0.5`. Teacher parameters are frozen and teacher queries
run under `torch.no_grad()`, so only the student learns during distillation.
BC and KD can disagree: a teacher prediction is learned from demonstrations,
whereas a demonstration action is the recorded target for that sample.

## 4. Optional hidden-representation distillation

The student has a wider input than a teacher, but both have a 32-dimensional
second hidden layer. We can compare those activations for each routed sample:

\[
L_{repr} = \|h_{student} - h_{teacher}\|^2,
\]

\[
L_{total} = L_{BC} + \lambda_{KD}L_{KD} + \lambda_{repr}L_{repr}.
\]

We use `lambda_repr = 0.1`. This asks the student to use an internal
representation similar to its specialist teacher, in addition to matching
the final action. Hidden units in independently trained networks have no
guaranteed one-to-one meaning, so this extra loss is an experiment, not a
promise of lower test error. Compare the printed scores and plot to see what
happened with this seed.

## What this demonstrates—and what it does not

This is **skill consolidation**:

```text
{T_pick, T_push} -> one universal student
```

The student performs a selected atomic skill when given its skill ID. This
does **not** demonstrate automatically executing a new temporal composition,
such as `push -> pick`, from a long-horizon instruction. That would require
additional data or machinery for sequencing skills over time.

## How this maps to our ATOM-Bench / VLA project

| In this tutorial | In the project |
| --- | --- |
| synthetic observation | robot RGB/proprioception |
| skill ID | atomic task / routed skill |
| demonstration action | robot demonstration action chunk |
| `T_pick` / `T_push` | separately fine-tuned atomic VLA teachers |
| student | universal VLA |
| output KD | action / flow-field distillation |
| representation KD | hidden VLA/action-expert representation |
| synthetic dataset | ATOM-Bench atomic datasets |

The mapping is conceptual. This script does not implement ATOM-Bench or a VLA.
