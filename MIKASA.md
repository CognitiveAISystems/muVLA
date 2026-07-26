# μVLA on MIKASA-Robo

[MIKASA-Robo](https://github.com/CognitiveAISystems/MIKASA-Robo) is a suite of
memory-intensive manipulation tasks. Unlike LIBERO, the tasks are partially observable by
construction: the information needed to act correctly is shown early in the episode and is
no longer visible when the action must be taken. This is the benchmark the memory module
is aimed at.

This page is self-contained — data, the exact training commands, and the exact evaluation
commands. Only the shared install steps live elsewhere, in [SETUP.md](SETUP.md).

## Relevant files

Training
* `vla-scripts/finetune.py` — fine-tuning script (shared with LIBERO)
* `prismatic/vla/datasets/` — `MIKASARoboVLAEpisodicDataset`, the episodic dataloader

Evaluation
* `experiments/robot/mikasa_robo/run_mikasa_robo_eval.py` — single-environment eval
* `experiments/robot/mikasa_robo/run_mikasa_robo_eval_envs.sh` — launcher over an env preset
* `experiments/robot/mikasa_robo/mikasa_robo_utils.py` — env registry and the canonical
  32-environment list
* `experiments/robot/openvla_utils.py` — checkpoint loading, memory auto-detection

## 1. Install

Follow [SETUP.md](SETUP.md) first. `mikasa-robo-suite` is installed by `uv sync` from the
`v1.0.0` git tag, so this benchmark needs no extra install step.

Do not float that pin. v1.0.0 renamed two observation keys relative to earlier releases —
`prompt` became `task_cue` and `joints` became `proprio` — and the evaluation wrapper in
`experiments/robot/mikasa_robo/mikasa_robo_utils.py` is written against the v1.0.0
spelling of both.

## 2. Data

The training data is published as RLDS on the Hugging Face Hub:
<https://huggingface.co/datasets/mikasa-robo/mikasa-robo-vla-rlds>

The repository holds one directory per environment, 90 in total and roughly 190 MB each,
so downloading all of it is about 17 GB. **Fetch only the environments you intend to train
on.** The five that produce the reported results:

```bash
export MIKASA_ROOT="$PWD/data/mikasa_robo_vla_rlds"

uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('mikasa-robo/mikasa-robo-vla-rlds',
                  repo_type='dataset', local_dir='$MIKASA_ROOT',
                  allow_patterns=['shell_game_push_vla_v0/*',
                                  'intercept_medium_vla_v0/*',
                                  'remember_color_5_vla_v0/*',
                                  'take_it_back_vla_v0/*',
                                  'remember_shape_and_color_3x3_vla_v0/*'])
"
```

A single environment is the same call with one pattern:

```bash
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('mikasa-robo/mikasa-robo-vla-rlds',
                  repo_type='dataset', local_dir='$MIKASA_ROOT',
                  allow_patterns=['remember_color_5_vla_v0/*'])
"
```

Drop `allow_patterns` entirely to fetch all 90.

Resulting layout — one directory per environment, each holding a TFDS version directory:

```
$MIKASA_ROOT/
├── shell_game_push_vla_v0/1.0.0/
├── intercept_medium_vla_v0/1.0.0/
├── remember_color_5_vla_v0/1.0.0/
├── take_it_back_vla_v0/1.0.0/
└── remember_shape_and_color_3x3_vla_v0/1.0.0/
```

Each environment holds **250 demonstrations**.

Note the naming: the dataset directories are `snake_case`, while the Gym environment ids
used at evaluation time are `CamelCase-VLA-v0`. `--mikasa_env_names` is resolved as a
directory name under `--data_root_dir`, so it takes the `snake_case` spelling; `--env_id`
at evaluation takes the Gym id. The two are not interchangeable.

## 3. Training

Both commands below are complete and copy-pasteable once this block is exported:

```bash
export EXP_DIR="$PWD/my_checkpoints/exp_mikasa"
export MODEL_SNAPSHOT="${EXP_DIR}/base_model_snapshot"   # created in SETUP.md
export MIKASA_ROOT="$PWD/data/mikasa_robo_vla_rlds"
export MIKASA_ENVS="shell_game_push_vla_v0,intercept_medium_vla_v0,remember_color_5_vla_v0,take_it_back_vla_v0,remember_shape_and_color_3x3_vla_v0"
```

The five training environments are always passed as one comma-separated list.
Normalization statistics are computed jointly across them and stored under the key
`mikasa_combined`, so a checkpoint trained on this set must be unnormalized with
`mikasa_combined` rather than a per-environment key. The evaluation script resolves that
automatically.

Replace `--nproc-per-node 8` with your GPU count; the learning rate was tuned for the
8-GPU setting.

### 3.1. μVLA with memory — the canonical configuration

64 memory tokens, TBPTT with `--tbptt_length 2`, cosine schedule. The reported
MIKASA-Robo results come from this configuration.

```bash
uv run python -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node 8 \
  vla-scripts/finetune.py \
    --vla_path "${MODEL_SNAPSHOT}" \
    --use_mikasa_episodic True \
    --data_root_dir "${MIKASA_ROOT}" \
    --mikasa_env_names "${MIKASA_ENVS}" \
    --run_root_dir "${EXP_DIR}/runs" \
    --dataset_name mikasa_five \
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
    --run_id_note mu_vla_mikasa_64mem_tbptt2 \
    --use_memory true \
    --num_mem_tokens 64 \
    --tbptt_length 2 \
    --use_gradient_checkpointing True \
    --grad_accumulation_steps 1 \
    --memory_log_freq 10 \
    --memory_expensive_log_freq 100 \
    --lr_schedule cosine \
    --lr_warmup_steps 2000 \
    --lr_min_ratio 0.1
```

### 3.2. OpenVLA-OFT baseline — no memory

The same code path with the memory block removed. This is what the memory runs are
compared against.

```bash
uv run python -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node 8 \
  vla-scripts/finetune.py \
    --vla_path "${MODEL_SNAPSHOT}" \
    --use_mikasa_episodic True \
    --data_root_dir "${MIKASA_ROOT}" \
    --mikasa_env_names "${MIKASA_ENVS}" \
    --run_root_dir "${EXP_DIR}/runs" \
    --dataset_name mikasa_five \
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
    --run_id_note openvla_oft_mikasa_baseline
```

### Why `--use_mikasa_episodic True` is required with memory

`finetune.py` asserts it. Memory only means something if consecutive batch elements are
consecutive timesteps of the same episode, which the standard frame-shuffled RLDS pipeline
does not provide. The episodic dataloader emits `is_first` / `is_last` so the memory state
can be reset at episode boundaries.

### Memory flags

| Flag | Default | Meaning |
|---|---|---|
| `--use_memory` | `False` | Enable the memory module. `False` reproduces OpenVLA-OFT. |
| `--num_mem_tokens` | `4` | Memory tokens prepended to the multimodal prefix. Reported runs use 64. |
| `--memory_update` | `tbptt` | `tbptt` truncates the gradient every `tbptt_length` steps; `ema` blends instead. |
| `--tbptt_length` | `1` | Steps carried before truncation. MIKASA-Robo uses 2. |
| `--ema_alpha` | `0.1` | Only used when `--memory_update ema`. |
| `--attention_mask_mode` | `custom` | `custom` is the memory-aware mask; `full` is the ablation. |
| `--use_gradient_checkpointing` | `False` | Required to fit 64 memory tokens on an 80 GB card. |

### Resuming

```bash
    --vla_path "${CKPT_DIR}" \
    --resume True \
    --resume_step 52500 \
```

`CKPT_DIR` is the `...--<step>_chkpt` directory under `${EXP_DIR}/runs` — note the double
hyphen before the step. All other flags must match the original run.

## 4. Evaluation

Memory hyperparameters (`--use_memory`, `--num_mem_tokens`, `--memory_update`,
`--ema_alpha`) are **auto-detected from the checkpoint** — they are written into the
checkpoint directory at save time. The same command therefore evaluates a memory
checkpoint and a baseline checkpoint. Pass them explicitly only to override, for example
to run a memory checkpoint with memory disabled as an ablation.

```bash
export CKPT_DIR="${EXP_DIR}/runs/<...>--150000_chkpt"
```

### 4.0. Released checkpoints

To evaluate without training first, both reported MIKASA-Robo runs are on the Hub under the
[**mu-vla**](https://huggingface.co/mu-vla/models) organization. Both are step 150000,
64 memory tokens, cosine scheduling, trained on the five environments from section 2; they
differ only in the TBPTT truncation length.

| Checkpoint | TBPTT |
|---|---|
| [`mu-vla-openvla-oft-mikasa-robo-5-tasks-m64-k8-tbptt`](https://huggingface.co/mu-vla/mu-vla-openvla-oft-mikasa-robo-5-tasks-m64-k8-tbptt) | 8 |
| [`mu-vla-openvla-oft-mikasa-robo-5-tasks-m64-k2-tbptt`](https://huggingface.co/mu-vla/mu-vla-openvla-oft-mikasa-robo-5-tasks-m64-k2-tbptt) | 2 |

```bash
export CKPT_DIR="$PWD/my_checkpoints/mu-vla-mikasa-5-m64-k8"

uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('mu-vla/mu-vla-openvla-oft-mikasa-robo-5-tasks-m64-k8-tbptt',
                  local_dir='$CKPT_DIR')
"
```

Each is about 16 GB and includes `dataset_statistics.json` and `memory_meta.json`, so the
commands below work against the downloaded directory unchanged — the memory
hyperparameters are picked up from the checkpoint.

### 4.1. Canonical evaluation

Both inference modes were reported for every checkpoint, so both are run:

```bash
# Receding horizon: re-plan at every step from the current observation.
bash experiments/robot/mikasa_robo/run_mikasa_robo_eval_envs.sh \
    --checkpoint "${CKPT_DIR}" \
    --preset mikasa32 \
    --num_trials 100 \
    --receding_horizon true \
    --results_note "rh-true"

# Open loop: execute the whole 8-action chunk before re-planning.
bash experiments/robot/mikasa_robo/run_mikasa_robo_eval_envs.sh \
    --checkpoint "${CKPT_DIR}" \
    --preset mikasa32 \
    --num_trials 100 \
    --receding_horizon false \
    --results_note "rh-false"
```

Swap `--preset mikasa32` for `--preset mikasa5` to evaluate only the five training
environments, or `--envs A-VLA-v0,B-VLA-v0` for an arbitrary subset. `--envs` takes Gym
environment ids, not dataset directory names.

### 4.2. A single environment

This is the script the launcher calls per environment; useful for a quick check.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python \
  experiments/robot/mikasa_robo/run_mikasa_robo_eval.py \
    --pretrained_checkpoint "${CKPT_DIR}" \
    --env_id RememberColor5-VLA-v0 \
    --num_trials 100 \
    --receding_horizon true
```

### Protocol

All reported MIKASA numbers use:

* **100 episodes per environment** (`--num_trials 100`)
* **starting seed 4242424242**; episode *i* uses env seed `4242424242 + i`
* **`success_once`** as the metric — the task counts as solved if the success condition
  held at any point in the episode
* mean ± standard error over episodes, then averaged across environments

Presets:

| Preset | Contents |
|---|---|
| `mikasa5` | the five training environments (default) |
| `mikasa32` | the evaluation set: 23 of the 32 environments in `MIKASA_ROBO_32_ENV_IDS` |

`mikasa32` excludes the nine 400-step environments — `BunchOfColors{3,5,7}`,
`SeqOfColors{3,5,7}` and `ChainOfColors{3,5,7}` — which are present in
`MIKASA_ROBO_32_ENV_IDS` but commented out of the preset. The 23 it does run are the ones
`scripts/collect_eval_results.py` tabulates.

### Receding horizon

`--receding_horizon true` re-plans at every step from the current observation, using only
the first action of each predicted chunk. `false` executes the whole chunk of 8 actions
open-loop before re-planning.

The default is auto: `true` when the checkpoint has memory, `false` otherwise. Both modes
were measured for every checkpoint, because the choice interacts with memory — under
open-loop execution the memory state is not updated within a chunk, so it is stale for 7
of every 8 environment steps.

### Collecting results

Each run writes into `./eval_results/<CKPT_TAG>_<results_note>/<env_id>/<timestamp>--seed<seed>/`,
where `CKPT_TAG` is the basename of `--checkpoint`. The aggregator walks that tree, so point
it at the top:

```bash
uv run python scripts/collect_eval_results.py \
    --eval-results-dir eval_results \
    --output eval_results_summary.csv
```
