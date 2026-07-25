# μVLA on LIBERO

[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) is the standard manipulation
benchmark inherited from OpenVLA-OFT. It is fully observable, so memory is not required to
solve it. It is included here for the opposite reason: it is the control condition. A
memory mechanism that helps on [MIKASA-Robo](MIKASA.md) should not *hurt* on LIBERO, and
the LIBERO numbers are what makes that checkable against a well-known baseline.

This page is self-contained — install, data, the exact training commands, and the exact
evaluation commands. The shared base install lives in [SETUP.md](SETUP.md).

## Relevant files

Training
* `vla-scripts/finetune.py` — fine-tuning script (shared with MIKASA-Robo)
* `prismatic/vla/datasets/` — `LIBEROVLAEpisodicDataset`, the episodic dataloader

Evaluation
* `experiments/robot/libero/run_libero_eval.py` — single-suite eval
* `experiments/robot/libero/run_libero_eval_suites.sh` — launcher over a suite preset
* `experiments/robot/libero/libero_utils.py` — LIBERO env helpers
* `experiments/robot/libero/regenerate_libero_dataset.py` — dataset regeneration
* `experiments/robot/openvla_utils.py` — checkpoint loading, memory auto-detection

## 1. Install

Complete [SETUP.md](SETUP.md) first, then install LIBERO itself into the same environment:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

# Required. See the note below — without this the editable install resolves to nothing.
touch LIBERO/libero/__init__.py

uv pip install -e LIBERO
uv pip install -r experiments/robot/libero/libero_requirements.txt

# First import writes ~/.libero/config.yaml and prompts for a dataset path.
# Answering "N" accepts the defaults, which is what the eval scripts expect.
echo N | uv run --no-sync python -c "import libero"
```

Verify:

```bash
uv run --no-sync python -c "
from libero.libero.envs import OffScreenRenderEnv
print('LIBERO OK')
"
```

Three things about that block are not obvious, and all three fail in ways that look like
something else:

* **`touch LIBERO/libero/__init__.py`.** LIBERO's `setup.py` uses `find_packages()`, but
  upstream ships no `libero/__init__.py`, so nothing is found. A legacy
  `pip install -e` tolerated this — it put the repository root on `sys.path` and `libero`
  resolved as a namespace package. A PEP 660 editable install, which is what `uv` does,
  builds an explicit import mapping instead, and that mapping comes out **empty**. The
  install reports success and `uv pip list` shows `libero 0.1.0`; only `import libero`
  fails, with `ModuleNotFoundError`.
* **The `echo N` line.** LIBERO's first import is interactive. Left to itself it blocks on
  a `(Y/N)` prompt, which in a batch job or under `pytest` surfaces as a hang or as
  `OSError: reading from stdin while output is captured`.
* **`--no-sync` after this point.** Plain `uv run` re-syncs the environment from
  `uv.lock` and prunes anything installed with `uv pip install`, which silently removes
  LIBERO and its requirements again. Use `uv run --no-sync` for every LIBERO command, or
  re-run the two `uv pip install` lines after any `uv sync`.

`libero_requirements.txt` pins `mujoco==3.8.0` deliberately: `robosuite` 1.4.0 calls
`mujoco.mj_fullM(model, dst, qM)`, and mujoco 3.9 changed that signature, so an unpinned
install dies with `TypeError: mj_fullM(): incompatible function arguments` on the first
environment reset.

## 2. Data

The modified LIBERO RLDS datasets (~10 GB) are the same ones used by OpenVLA and
OpenVLA-OFT: <https://huggingface.co/datasets/openvla/modified_libero_rlds>

```bash
export LIBERO_ROOT="$PWD/data/modified_libero_rlds"

uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('openvla/modified_libero_rlds',
                  repo_type='dataset', local_dir='$LIBERO_ROOT')
"
```

Expected layout:

```
$LIBERO_ROOT/
├── libero_spatial_no_noops/1.0.0/
├── libero_object_no_noops/1.0.0/
├── libero_goal_no_noops/1.0.0/
└── libero_10_no_noops/1.0.0/
```

`_no_noops` means training samples with near-zero actions have been filtered out.

## 3. Training

μVLA trains a **single policy over all four suites at once** rather than one policy per
suite, because a memory state carried across an episode is only comparable if the four
suites go through the same model. Normalization statistics are computed jointly and stored
under `libero_combined`; the evaluation script resolves the un-norm key automatically
(suite name → `<suite>_no_noops` → `libero_combined`), so a combined checkpoint works
without passing `--unnorm_key`.

Export this block first:

```bash
export EXP_DIR="$PWD/my_checkpoints/exp_libero"
export MODEL_SNAPSHOT="${EXP_DIR}/base_model_snapshot"   # created in SETUP.md
export LIBERO_ROOT="$PWD/data/modified_libero_rlds"
export LIBERO_SUITES="libero_spatial_no_noops,libero_object_no_noops,libero_goal_no_noops,libero_10_no_noops"
```

Replace `--nproc-per-node 8` with your GPU count; the learning rate was tuned for the
8-GPU setting.

### 3.1. μVLA with memory — the canonical configuration

64 memory tokens, TBPTT with `--tbptt_length 8`, cosine schedule. The reported LIBERO
results come from this configuration.

Note the difference from MIKASA-Robo, which uses `--tbptt_length 2`. LIBERO episodes are
longer and fully observable, so a longer carry costs nothing in stability.

```bash
uv run python -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node 8 \
  vla-scripts/finetune.py \
    --vla_path "${MODEL_SNAPSHOT}" \
    --use_libero_episodic True \
    --data_root_dir "${LIBERO_ROOT}" \
    --libero_suite_names "${LIBERO_SUITES}" \
    --run_root_dir "${EXP_DIR}/runs" \
    --dataset_name libero_combined \
    --batch_size 4 \
    --use_l1_regression True \
    --use_proprio True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100_000 \
    --max_steps 150_005 \
    --save_freq 2500 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --run_id_note mu_vla_libero_64mem_tbptt8 \
    --use_memory true \
    --num_mem_tokens 64 \
    --tbptt_length 8 \
    --use_gradient_checkpointing True \
    --grad_accumulation_steps 1 \
    --memory_log_freq 10 \
    --memory_expensive_log_freq 100 \
    --lr_schedule cosine \
    --lr_warmup_steps 2000 \
    --lr_min_ratio 0.1
```

### 3.2. OpenVLA-OFT baseline — no memory

```bash
uv run python -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node 8 \
  vla-scripts/finetune.py \
    --vla_path "${MODEL_SNAPSHOT}" \
    --use_libero_episodic True \
    --data_root_dir "${LIBERO_ROOT}" \
    --libero_suite_names "${LIBERO_SUITES}" \
    --run_root_dir "${EXP_DIR}/runs" \
    --dataset_name libero_combined \
    --batch_size 4 \
    --use_l1_regression True \
    --use_proprio True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100_000 \
    --max_steps 150_005 \
    --save_freq 2500 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --run_id_note openvla_oft_libero_baseline
```

For reference, OpenVLA-OFT reports 97.1% average success for four independently trained
policies and 96.8% for one policy trained on all four suites combined, so the combined
setting costs almost nothing on this benchmark.

The memory-flag table and the resume instructions are shared with MIKASA-Robo; see
[MIKASA.md](MIKASA.md#memory-flags).

## 4. Evaluation

Memory hyperparameters are auto-detected from the checkpoint, so the same command
evaluates a memory checkpoint and a baseline checkpoint.

```bash
export CKPT_DIR="${EXP_DIR}/runs/<...>--150000_chkpt"
```

### 4.1. Canonical evaluation — all four suites

Both inference modes were reported, so both are run:

```bash
bash experiments/robot/libero/run_libero_eval_suites.sh \
    --checkpoint "${CKPT_DIR}" \
    --preset libero4 \
    --num_trials_per_task 50 \
    --receding_horizon true \
    --results_note "rh-true"

bash experiments/robot/libero/run_libero_eval_suites.sh \
    --checkpoint "${CKPT_DIR}" \
    --preset libero4 \
    --num_trials_per_task 50 \
    --receding_horizon false \
    --results_note "rh-false"
```

Presets: `libero4` (all four canonical suites, default), `libero_spatial`,
`libero_object`, `libero_goal`, `libero_10`, `libero_90`. `--suites <comma-list>`
overrides the preset; `--task_id <n>` restricts to one task within a suite.

Defaults: 50 episodes per task, `--seed 7`. With 10 tasks per suite that is 500 trials
per suite.

`--receding_horizon` defaults to `true` for memory checkpoints and `false` otherwise; see
[MIKASA.md](MIKASA.md#receding-horizon) for why both were measured.

### 4.2. A single suite

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python \
  experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "${CKPT_DIR}" \
    --task_suite_name libero_spatial \
    --num_trials_per_task 50 \
    --receding_horizon true
```

### 4.3. An upstream OpenVLA-OFT checkpoint, through the same path

```bash
bash experiments/robot/libero/run_libero_eval_suites.sh \
    --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --preset libero4
```

Results land in `./eval_results/<CKPT_TAG>_<results_note>/<suite>/<timestamp>--seed<seed>/`,
one directory per suite per run. `scripts/collect_eval_results.py` aggregates MIKASA-Robo
runs only; it has no LIBERO code path.

## Notes

* Train until L1 loss drops below ~0.01 and plateaus. With the configuration above it
  reaches ~0.006 on LIBERO-Spatial by 150K steps. The schedule is `cosine`: linear warmup
  over `--lr_warmup_steps`, then cosine decay to `--lr_min_ratio` of the peak by
  `--max_steps`. `--num_steps_before_decay` is carried through from the original launch
  scripts but is read only by the `multistep` schedule, which is what the §3.2 baseline
  uses.
* Robosuite tears down its EGL context at interpreter shutdown, which can print
  `EGLError(EGL_NOT_INITIALIZED)` *after* the results are already on screen. It is
  cosmetic; the exit code is still 0.
