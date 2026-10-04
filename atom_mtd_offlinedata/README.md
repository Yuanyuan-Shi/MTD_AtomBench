# Offline ATOM-Bench multi-teacher distillation

This experiment extends the copied `../openpi/scripts/train_pytorch.py`. It uses
OpenPI's pi0.5 model, flow loss, preprocessing, AdamW convention and training loop;
LeRobot reads the released parquet/video trajectories. There is no simulator,
rollout controller, online data collection, or success-rate evaluation here.

The same code runs on one local GPU (debugging, `configs/local.json`) and on a
SLURM cluster of multi-GPU nodes (`configs/aws.json`, 3 nodes × 8 H100). The full
experiment is: Stage A (two specialists and Joint BC, 10 epochs each, batch 64),
Stage B (three KD variants, 5,000 updates each), then held-out evaluation of all
six models.
The implementation is not a claim that the requested training runs have completed.

## Layout and provenance

- `../openpi/`: OpenPI source vendored at upstream commit
  `215abfb217dbac7d5f1273282331b9b1866c0479` (its git history is not kept), with local
  edits in `scripts/train_pytorch.py` and `models_pytorch/pi0_pytorch.py` (this
  experiment) and `models/gemma.py`, `models/lora.py`, `models/ecot_pi05.py` (an earlier
  project). `openpi/third_party/` (ALOHA, LIBERO) is not in git; `../setup.sh` clones it
  at OpenPI's pinned commits.
- `../pi05_lora_sft/`: LoRA implementation and Stage-A entry point. The reference
  directory contains the previous full-parameter JAX SFT configuration for audit;
  it is not used by this experiment.
- `configs/local.json`, `configs/aws.json`: one complete config per environment
  (see "Configs and hardware").
- `prepare.py`, `data.py`: pinned download, episode splits, normalization and pairs.
- `cache_frames.py`: one-time decode of all video frames into 224x224 uint8 arrays.
- `model.py`, `losses.py`: teacher routing, representation audit and KD losses.
- `train.py`: configuration and callbacks into OpenPI's existing training loop.
- `evaluate.py`: held-out flow/action/teacher/temporal errors and single-seed tables.
- `run_experiment.py`: orchestration of a whole phase, with no continuation baseline.
- `plan.py`, `benchmark.py`, `launch/`: GPU allocation, step timing, and local or
  SLURM job launch (`launch/slurm_experiment.sbatch`).
- `preflight.py`, `diagnose.py`, `tests/`: environment, real-data gradient and protocol checks.

Only the OpenPI trainer and `PI0Pytorch` have experiment-specific edits. `env.sh`
selects the environment `openpi/.venv`, the vendored OpenPI source and
`runtime/python/transformers`: transformers 4.53.2 with OpenPI's patched model files
(adaRMS, activation precision, KV cache), kept as a separate copy instead of
overwriting the environment as OpenPI's README suggests. `../setup.sh` builds both;
neither is stored in git. CUDA 12.8 PyTorch is required by Blackwell GPUs and also
used on H100.

Large inputs are downloaded, not stored in git: the converted pi0.5 base model from
Hugging Face [`yyshi0619/pi05-base-pytorch`](https://huggingface.co/yyshi0619/pi05-base-pytorch)
(pinned revision and SHA256 in the configs; a format conversion of
`gs://openpi-assets/checkpoints/pi05_base` with OpenPI's
`examples/convert_jax_model_to_pytorch.py --precision float32`), and the dataset from
`AtomBench/FrankaPanda` at the pinned revision. `prepare.py` fetches and verifies both.

## Data and fixed protocol

Default track: Franka Panda. Source:
[AtomBench/FrankaPanda](https://huggingface.co/datasets/AtomBench/FrankaPanda), revision
`4c2d4ed5e86b7967cc048a3bfe935d5daef6c663`.

| Task | Directory | Actual metadata instruction |
|---|---|---|
| i1 | `I1_Pick_the_blue_cube_and_place_it_on_the_basket` | Pick up the blue cube and place it into the basket. |
| i5 | `I5_Pick_two_blocks_and_place_them_in_the_basket` | Pick up exactly two blocks and place them into the basket. |
| x1 | No released observation/action trajectory directory in this revision | Pick up exactly two red cubes and place them into the basket. |

The x1 instruction is verified against the
[benchmark's published page source](https://github.com/flageval-baai/AtomBenchPage/blob/main/index.html),
not against unavailable x1 trajectories. Video examples alone are insufficient
for action-MSE evaluation. A valid released x1 LeRobot trajectory root can be
configured later; its prompt is checked, all of its episodes are evaluation-only,
and no teacher is constructed for it. Its normalization always uses atomic train data.

Each atomic task has 100 released episodes. `data_seed` (default 20261001) fixes 80 train,
10 validation and 10 test episodes per task. `data/FrankaPanda/splits.json`
records exact IDs, prompts, revision and parquet/video SHA256s. Normalization
uses only pooled i1+i5 **training** frames, with OpenPI quantile normalization.
Task membership is attached explicitly because both source datasets use
`task_index=0`. Episode identity is `(task_id, episode_index)`.

A pair is two consecutive policy queries, at frames t and t + `execution_horizon`
(the number of actions executed between queries; config default 1, must be
< H). Released frame indices can start above zero after trimming. Pair
construction uses their actual indices, checks the `execution_horizon` gap and
episode/task equality, and keeps only pairs for which **both** H-step chunks are
fully available. There is no episode-end action padding. Every Stage-B variant
uses the same pair sampler. `batch_size` counts observations (64 on AWS = 32 pairs).

All 14 released absolute action coordinates are used and padded to 32 for pi0.5.
They comprise seven joints (rad), gripper (0..1, about 1 = open) and six end-effector
pose coordinates. Neither the paper nor the dataset card documents the pose
convention; forward kinematics of the Panda from the recorded joints reproduces it
(orientation within 0.11 deg, position within 1.3 mm): x, y, z is the **flange**
position in meters in the robot base frame (not the Robotiq fingertip), and rx, ry, rz
is roll-pitch-yaw in radians, R = Rz(rz) Ry(ry) Rx(rx). Data was collected by
SpaceMouse teleoperation at 30 Hz.

Angle dims (`angle_dims`: rx, ry, rz) are wrapped into (center - pi, center + pi]
before statistics and training; the center is the circular mean of the training data
snapped to a multiple of pi/2 and recorded in `splits.json` (`angle_wrap`). Here rx
(gripper pointing down, roll about pi) gets center pi and ry, rz get 0. Two i5 episodes
(50 train, 66 test) store roll as about -pi instead of +pi: the same orientation, but
it normalized to about -26 and caused loss spikes; wrapping maps it next to the other
episodes. A fixed (-pi, pi] window would instead split rx at its typical value (3,537
fake jumps across 196 of 200 episodes). `prepare.py` refuses angle data that comes
within pi/4 of its window edge (a sin/cos encoding would be needed). For a robot,
`data.to_physical_actions` undoes normalization and can re-wrap angles, e.g.
`output_angle_center=0.0` for controllers expecting (-pi, pi].
No speculative delta-action transform is applied. Front/wrist/side images map to
OpenPI's base/left-wrist/right-wrist image slots, respectively. The third slot is
the side camera, not a claim that this robot has a second wrist camera.

Default H=50; max prompt length=200. Native pi0.5 tokenization includes discrete
state. Language KD masks select instruction and discretized-state tokens, i.e.
every real prompt token except BOS and the trailing "\nAction: " cue (padding is
excluded); the image KD mask excludes absent camera tokens.
The tokenizer runs on the 14-D normalized state before native padding to 32.

## LoRA and training tree

Base source: `gs://openpi-assets/checkpoints/pi05_base/params`.
The existing local Orbax checkpoint is specified by `base_checkpoint_jax`.
The local converted checkpoint is `checkpoints/pi05_base_pytorch/model.safetensors`.

OpenPI's existing LoRA is implemented in JAX; its PyTorch trainer does not install
adapters when a `_lora` config is selected. `pi05_lora_sft/lora.py` translates the
existing per-head einsum parameterization into PyTorch:

| Component | Rank | Alpha | Dropout | Targets |
|---|---:|---:|---:|---|
| PaliGemma text transformer | 16 | 16 | 0 | q/k/v/o, gate/up/down |
| Action expert | 32 | 32 | 0 | q/k/v/o, gate/up/down |

Attention adapters retain per-head factors, including the output projection;
they do not flatten heads into one rank-16/32 update. Both factors use normal
initialization with std=0.01, matching this OpenPI checkout. The scale is 1 for
these settings. The vision encoder, multimodal projector, action projections and
time MLP remain trainable, matching `Pi0Config.get_freeze_filter()`; non-adapter
LLM parameters are frozen. EMA is disabled as in the upstream LoRA configuration.

AdamW: betas=(0.9,0.95), eps=1e-8, weight decay=1e-10, gradient clip=1.
Stage A: `stage_a.epochs` (10) passes over the valid training pairs of each model's
data, i.e. ceil(10 × pairs / (batch_size / 2)) updates (with batch 64: T1 11,309,
T2 17,430, Joint BC 28,739), 5% warmup, cosine 5e-5 -> 5e-6.
Stage B: 5,000 updates, 5% warmup, cosine 1e-5 -> 1e-6.
The smoke run uses 30 updates per model and does not substitute for the main run.
These counts are initial fixed choices, not results of hyperparameter tuning.

For seed 7: the two specialists and Joint BC independently initialize
from the same base and LoRA initialization. Specialists see their task only.
Each Stage-B model loads exactly that seed's single Joint BC checkpoint; SHA256,
the complete set of trainable parameter keys, and tensor equality are checked.
The frozen parameters are restored from the same base. All three Stage-B models
use the same pairs, seed, optimizer, schedule, batch size and update count.
Checkpoint files contain all trained parameters (including non-LLM parameters),
with base identity recorded. The runner skips fully completed, matching jobs. Every
`hardware.resume_interval` updates a resume checkpoint (student and optimizer state,
newest only) is written; rerunning an interrupted job continues from it with the same
step-addressed batches, tau and noise and the same W&B run.

## Exact losses and tensors

`PI0Pytorch.forward_features()` performs the original flow forward. With
`x_tau = tau*noise + (1-tau)*actions`, its `task_loss` is the unreduced original
`MSE(noise-actions, flow)`, averaged across the full OpenPI [B,H,32] tensor.

| Tensor key | Hidden state location | Shape |
|---|---|---|
| `flow` | `action_out_proj(suffix_out)` | [B,50,32] |
| `img` | final contextualized prefix output, image positions | [B,768,2048] |
| `lang` | final contextualized prefix output, text positions | [B,200,2048], language mask applied |
| `action` | final expert output immediately before `action_out_proj` | [B,50,1024] |

Raw SigLIP embeddings and token embedding lookups are not used as KD targets.
Before a refinement run, the runner checks both specialists against Joint BC on
fixed training pairs. Identical representations prevent representation KD.
Values are written to `representation_audit.json`; this audit cannot be completed
until trained specialist and Joint BC checkpoints exist.

All KD arithmetic and reductions use FP32. Define `N(h)=h/(||h||2+eps)` over the
hidden dimension and `masked_MSE` over valid tokens and their features:

```text
L_output = mean((student.flow - teacher.flow)^2)
L_img    = masked_MSE(N(student.img),    N(teacher.img),    image_mask)
L_lang   = masked_MSE(N(student.lang),   N(teacher.lang),   language_mask)
L_action = masked_MSE(N(student.action), N(teacher.action), action_mask)
L_repr   = L_img + L_lang + L_action
k        = execution_horizon   (chunks at t and t+k overlap on H-k steps)
r_s      = student_t[:,k:,:] - student_tk[:,:-k,:]
r_t      = teacher_t[:,k:,:] - teacher_tk[:,:-k,:]
L_temp   = mean((r_s-r_t)^2)
L_total  = L_task + lambda_out*L_output
                  + lambda_repr*L_repr + lambda_temp*L_temp
```

Only the terms in a variant's name are enabled; other training-loss logs are zero.
Initial coefficients are 0.5/0.1/0.1. Teachers are frozen, in eval mode and under
`torch.no_grad()`. Teacher/student share one already augmented observation, prompt,
noise and tau at each state. Temporal neighbors share tau and get independent
standard Gaussian noise tensors. There is no standalone student smoothing loss.

Training logs raw and weighted terms, total/task loss, LR and gradient norm.
Each run saves `first_batch.json`, `metrics.jsonl`, `config.json`, initial/final
fixed-RNG train/validation probes, checkpoint provenance and W&B links. Stage-A
checkpoints are not trusted unless both fixed-probe losses decrease. A failed
smoke is recorded and stops the pipeline, rather than changing seeds or ranking.

## W&B and offline evaluation

Project: `atom-mtd-offlinedata`. The requested UI is
<https://forge.coreweave.com/wandb/home>. The verified API base is
`https://forge.coreweave.com/api/wandb`, configured in `wandb_base_url`.
This gateway accepts the existing W&B cloud credential from the local netrc;
the integration loads it into the process environment for this specific gateway.
API keys are never stored in experiment files or printed.
There is no silent offline W&B fallback. The preflight makes an online smoke run
and reads its metric back from the server; training metrics are also read back.

Validation and test use disjoint episodes. Default test protocol samples 256
valid pairs per task without replacement using `data_seed + 1`, shared by
all models and training seeds. Ten native flow integration steps produce the
first-action and full-chunk MSE; a velocity prediction is not mislabeled as an
action prediction. Action MSE uses the 14 physical dimensions in normalized space;
separately labeled original-coordinate MSE is also saved. Flow/KD retain the
original 32-D model output. Atomic errors include the matching specialist's
output, normalized representations and overlapping temporal residuals. x1 has
only teacher-free metrics when trajectories are available. No offline error is
reported as success rate.

`<output_root>/<smoke|main>/comparison.json` and `comparison.md` contain one result
per model (seed 7). The specialists are evaluated on their own task only, as baselines
without teacher-matching metrics. Test pairs are split across GPUs; each pair's tau
and noise come from its own seed, so results do not depend on GPU count or batch size.
No across-seed standard deviation is estimated. Missing results cause aggregation
to fail. Atomic temporal errors also have an equally weighted i1/i5 mean.

## Configs and hardware

`configs/local.json` and `configs/aws.json` are complete and identical except for
`batch_size` (32 locally, 64 on AWS) and the `hardware` block; a test enforces this.
`hardware` holds settings that never change results: `launcher` (`local` or `slurm`),
`nodes`, `gpus_per_node`, `output_root` (`runs/local`, `runs/aws`), `frame_cache_dir`,
loader/cache workers, `max_train_obs_per_gpu`, `resume_interval` and timing estimates
for the planner. It is excluded from `config_hash` and run matching, so moving a run to
different hardware does not mark it stale. Commands default to `configs/local.json`.

Results do not depend on the GPU count: each step's global batch of pairs, tau and
noise are drawn from the step seed and sliced per GPU (augmentation is drawn per GPU).
Logged losses are averaged over GPUs.

Video decoding dominated step time (13–16 s per batch of 32 observations).
`cache_frames.py` decodes every frame once with the training path's exact operations
(LeRobot pyav decoding, uint8 conversion, and the PIL `resize_with_pad` from
`openpi_client` that `transforms.ResizeImages` applies, to 224x224), about 57 GB,
read through `np.memmap`. A test checks that cached records are bit-identical to
video decoding. With `frame_cache_dir: null`, frames are decoded from video as before.

`plan.py` assigns jobs to nodes and GPUs: phases run in order, and within a phase every
job order and GPU count (counts that split the batch, fit one node or use whole nodes,
and respect `max_train_obs_per_gpu`) is simulated; the shortest schedule wins. On
3 × 8 GPUs it runs T1 then Joint BC on 16 GPUs while T2 uses the third node, one KD
variant per node, and evaluation on 4 GPUs per model. `benchmark.py` measures seconds
per step; the planner uses those numbers instead of the estimates once they exist.

## Reproduction

Run from the workspace root (paths containing spaces are quoted):

```bash
cd '/mnt1/Multi-Teacher Distillation'
source atom_mtd_offlinedata/env.sh

# Base model (Hugging Face, if missing) and the pinned i1/i5 task directories;
# both hash-checked. Then fixed splits, angle windows and normalization statistics.
"$ATOM_PYTHON" -m atom_mtd_offlinedata.prepare

# Local (1 GPU): frame cache, tests, smoke.
"$ATOM_PYTHON" -m atom_mtd_offlinedata.cache_frames
"$ATOM_PYTHON" -m pytest -q atom_mtd_offlinedata/tests
"$ATOM_PYTHON" -m atom_mtd_offlinedata.diagnose --observations 32
"$ATOM_PYTHON" -m atom_mtd_offlinedata.run_experiment --phase smoke
"$ATOM_PYTHON" -m atom_mtd_offlinedata.smoke_report

# Preview the AWS schedule and every srun command without running anything.
"$ATOM_PYTHON" -m atom_mtd_offlinedata.run_experiment \
  --config atom_mtd_offlinedata/configs/aws.json --phase main --dry-run

# Individual jobs (Stage-A results must exist before Stage B):
"$ATOM_PYTHON" -m atom_mtd_offlinedata.train --variant teacher_i1 --seed 7 --smoke
"$ATOM_PYTHON" -m atom_mtd_offlinedata.evaluate --variant output_repr_temp_mtd --seed 7 --smoke
"$ATOM_PYTHON" -m atom_mtd_offlinedata.evaluate --summarize --smoke
```

On the AWS SLURM cluster, clone the repository onto the shared filesystem and set it
up once (no Hugging Face token is needed; both downloads are public). The frame cache
is built on each node's local NVMe automatically, and finished jobs are skipped on
resubmission:

```bash
git clone https://github.com/Yuanyuan-Shi/MTD_AtomBench.git && cd MTD_AtomBench
bash setup.sh                       # third_party, openpi/.venv, patched transformers
source atom_mtd_offlinedata/env.sh
"$ATOM_PYTHON" -m atom_mtd_offlinedata.prepare --config atom_mtd_offlinedata/configs/aws.json
git status                          # splits.json / norm_stats.json must be unchanged
MODE=benchmark sbatch atom_mtd_offlinedata/launch/slurm_experiment.sbatch   # optional timing
PHASE=smoke sbatch atom_mtd_offlinedata/launch/slurm_experiment.sbatch      # smoke gate
sbatch atom_mtd_offlinedata/launch/slurm_experiment.sbatch                  # full run
```

All random seeds come from two config keys.
- `training_seeds` (per replicate): LoRA init and batch order; shared by all six
  models of a replicate. Per-step augmentation/tau/noise use
  `seed * STEP_SEED_STRIDE + step` (`STEP_SEED_STRIDE = 1_000_003` in `common.py`;
  training asserts steps < stride so values stay unique).
- `data_seed` (global, every model and training seed): the episode split, drawn
  independently per task as `default_rng([data_seed, task_number])` by `prepare.py`;
  evaluation pairs and noise use `data_seed + 1`, fixed probes `data_seed + 2` and
  the Stage-B audit `data_seed + 3` (helpers in `common.py`). Changing `data_seed`
  requires rerunning `prepare.py`; training refuses a `splits.json` made with another
  seed or scheme.

Each run records the derived values in its `config.json` (`derived_seeds`).


All commands accept `--config`. Change the protocol in a new config/output root;
the smoke gate checks all per-run settings and the actual seed. Removing only
planned seed repetitions preserves a matching smoke result. Runtime dependencies are
`openpi/uv.lock` plus torch==2.7.1+cu128 and torchvision==0.22.1+cu128 from the official
CUDA 12.8 wheel index, installed by `../setup.sh`.
