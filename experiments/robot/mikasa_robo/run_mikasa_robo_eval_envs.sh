#!/usr/bin/env bash
# Run MIKASA-Robo evaluation for one checkpoint across a chosen env preset.
#
# Despite the historical name, this script can run against any of:
#   --preset mikasa5      the 5 training envs (default; used for exp_id 1..6)
#   --preset mikasa32     the 23-env evaluation set (see PRESET_MIKASA32 below)
#                         (mirrors `MIKASA_ROBO_32_ENV_IDS` in
#                         `experiments/robot/mikasa_robo/mikasa_robo_utils.py`)
# `--envs <comma-list>` always wins over `--preset` when set.
#
# Usage:
#   bash experiments/robot/mikasa_robo/run_mikasa_robo_eval_envs.sh \
#       --checkpoint /path/to/<...>_chkpt \
#       [--num_trials 100] \
#       [--seed 4242424242] \
#       [--cuda_device 0] \
#       [--use_memory true] \
#       [--num_mem_tokens 64] \
#       [--memory_update tbptt|ema] \
#       [--ema_alpha 0.1] \
#       [--receding_horizon true|false] \    # default: auto (true if memory else false)
#       [--unnorm_key mikasa_combined] \
#       [--use_wandb false] \
#       [--wandb_entity X] \
#       [--wandb_project Y] \
#       [--run_id_note tag] \
#       [--results_dir ./eval_results] \
#       [--results_note ""]                   # appended to <CKPT_TAG> dir name (default "")
#       [--preset mikasa5|mikasa32] \
#       [--envs ShellGamePush-VLA-v0,RememberColor5-VLA-v0]   # comma-separated subset; overrides preset
#
# Examples:
#   # exp_id_1 / exp_id_2 (no memory)
#   bash .../run_mikasa_robo_eval_envs.sh \
#       --checkpoint .../base_model_snapshot+...--exp_id_1...--50000_chkpt
#
#   # exp_id_3 (1 mem token, TBPTT)
#   bash .../run_mikasa_robo_eval_envs.sh \
#       --checkpoint .../base_model_snapshot+...--exp_id_3...--50000_chkpt \
#       --use_memory true --num_mem_tokens 1 --memory_update tbptt
#
#   # exp_id_4 / exp_id_5 (64 mem tokens, TBPTT)
#   bash .../run_mikasa_robo_eval_envs.sh \
#       --checkpoint .../base_model_snapshot+...--exp_id_4...--50000_chkpt \
#       --use_memory true --num_mem_tokens 64 --memory_update tbptt
#
#   # exp_id_6 (64 mem tokens, EMA)
#   bash .../run_mikasa_robo_eval_envs.sh \
#       --checkpoint .../base_model_snapshot+...--exp_id_6...--50000_chkpt \
#       --use_memory true --num_mem_tokens 64 --memory_update ema --ema_alpha 0.1
#
# The 5 envs are the same set passed to finetune.py via --mikasa_env_names:
#   ShellGamePush-VLA-v0
#   InterceptMedium-VLA-v0
#   RememberColor5-VLA-v0
#   TakeItBack-VLA-v0
#   RememberShapeAndColor3x3-VLA-v0

set -euo pipefail

# Silence noisy backends inherited by every per-env Python invocation.
# Mirrors the suppression in run_mikasa_robo_eval.py — exports here ensure
# the env vars also reach any child processes spawned before Python's
# own `os.environ.setdefault` lines run.
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ── Defaults ──────────────────────────────────────────────────────────────
CHECKPOINT=""
NUM_TRIALS=100
SEED=4242424242   # MIKASA validation protocol: per-episode env seeds = SEED + i
CUDA_DEVICE=0
# Memory hyperparameters are auto-detected from the checkpoint by default
# (see detect_memory_config in experiments/robot/openvla_utils.py).
# Set these to non-empty values only to override auto-detection.
USE_MEMORY=""
NUM_MEM_TOKENS=""
MEMORY_UPDATE=""
EMA_ALPHA=""
# Inference mode: "" (auto: receding-horizon iff use_memory), "true" or "false".
# Set explicitly to override regardless of memory state — works for both
# memory and non-memory checkpoints.
RECEDING_HORIZON=""
UNNORM_KEY=mikasa_combined
USE_WANDB=false
WANDB_ENTITY=""
WANDB_PROJECT=""
RUN_ID_NOTE=""
RESULTS_DIR="./eval_results"
# Optional suffix appended to the checkpoint folder name in eval_results/.
# Useful when running the same checkpoint under different inference settings
# (e.g. --results_note "receding_horizon=false_exp2").
RESULTS_NOTE=""
ENVS_ARG=""    # comma-separated subset; empty = preset
PRESET="mikasa5"   # one of: mikasa5 | mikasa32

# ── Arg parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)       CHECKPOINT="$2";      shift 2 ;;
        --num_trials)       NUM_TRIALS="$2";      shift 2 ;;
        --seed)             SEED="$2";            shift 2 ;;
        --cuda_device)      CUDA_DEVICE="$2";     shift 2 ;;
        --use_memory)       USE_MEMORY="$2";      shift 2 ;;
        --num_mem_tokens)   NUM_MEM_TOKENS="$2";  shift 2 ;;
        --memory_update)    MEMORY_UPDATE="$2";   shift 2 ;;
        --ema_alpha)        EMA_ALPHA="$2";       shift 2 ;;
        --receding_horizon) RECEDING_HORIZON="$2"; shift 2 ;;
        --unnorm_key)       UNNORM_KEY="$2";      shift 2 ;;
        --use_wandb)        USE_WANDB="$2";       shift 2 ;;
        --wandb_entity)     WANDB_ENTITY="$2";    shift 2 ;;
        --wandb_project)    WANDB_PROJECT="$2";   shift 2 ;;
        --run_id_note)      RUN_ID_NOTE="$2";     shift 2 ;;
        --results_dir)      RESULTS_DIR="$2";     shift 2 ;;
        --results_note)     RESULTS_NOTE="$2";    shift 2 ;;
        --envs)             ENVS_ARG="$2";        shift 2 ;;
        --preset)           PRESET="$2";          shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: --checkpoint is required" >&2
    exit 2
fi
# Accept either a local checkpoint directory or a HuggingFace repo id. The two are
# only distinguishable by shape: an HF id is exactly "owner/name", with no leading
# "/" or "./" and no second slash. Anything else that is not an existing directory
# is a typo, and saying so here beats a stack trace forty seconds into Python start-up.
if [[ -d "$CHECKPOINT" ]]; then
    :   # local checkpoint directory
elif [[ "$CHECKPOINT" == /* || "$CHECKPOINT" == .* || "$CHECKPOINT" == */*/* ]]; then
    echo "ERROR: --checkpoint directory does not exist: $CHECKPOINT" >&2
    exit 2
elif [[ "$CHECKPOINT" == */* ]]; then
    :   # HuggingFace repo id, e.g. moojink/openvla-7b-oft-... -- let Python resolve it
else
    echo "ERROR: --checkpoint is neither an existing directory nor a HuggingFace" >&2
    echo "       repo id (owner/name): $CHECKPOINT" >&2
    exit 2
fi

# ── Resolve repo root (this script lives at experiments/robot/mikasa_robo) ─
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EVAL_PY="$SCRIPT_DIR/run_mikasa_robo_eval.py"
DEFAULT_VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ -n "${EVAL_PYTHON:-}" ]]; then
    PYTHON_BIN="$EVAL_PYTHON"
elif [[ -x "$DEFAULT_VENV_PYTHON" ]]; then
    # Standalone runs should prefer the repo-managed uv environment when it
    # already exists, even if EVAL_PYTHON was not set.
    PYTHON_BIN="$DEFAULT_VENV_PYTHON"
else
    PYTHON_BIN="python"
fi

# ── Preflight: verify the project venv has the required Python deps ───────
# Failing here once with a helpful message is much better than failing in the
# per-env loop 32 times in a row with a cryptic ModuleNotFoundError.
# This catches the common mistake of running the script from a shell where
# `uv` resolves to some other environment than the project's own .venv.
if ! ( cd "$REPO_ROOT" && uv run --no-sync python -c "import draccus, torch, transformers, mani_skill" ) >/dev/null 2>&1; then
    echo "ERROR: project venv at '$REPO_ROOT' is missing core deps (draccus / torch / transformers / mani_skill)." >&2
    echo "       Likely 'uv run' is not pointing at the project venv. Try one of:" >&2
    echo "         - export EVAL_PYTHON='$DEFAULT_VENV_PYTHON'   (use the project venv explicitly)" >&2
    echo "         - cd '$REPO_ROOT' && uv sync              (one-time setup)" >&2
    echo "       Then re-run this script." >&2
    exit 3
fi

# ── Presets ───────────────────────────────────────────────────────────────
# mikasa5 = the 5 training envs (must match --mikasa_env_names from finetune.py).
PRESET_MIKASA5=(
    "ShellGamePush-VLA-v0"
    "InterceptMedium-VLA-v0"
    "RememberColor5-VLA-v0"
    "TakeItBack-VLA-v0"
    "RememberShapeAndColor3x3-VLA-v0"
)
# mikasa32 = the evaluation set: 23 of the 32 ids in MIKASA_ROBO_32_ENV_IDS
# (experiments/robot/mikasa_robo/mikasa_robo_utils.py). The nine commented-out
# entries below are the 400-step environments and are not part of the reported
# results; scripts/collect_eval_results.py tabulates the same 23.
PRESET_MIKASA32=(
    "ShellGameTouch-VLA-v0"
    "ShellGamePush-VLA-v0"
    "ShellGamePick-VLA-v0"
    "InterceptSlow-VLA-v0"
    "InterceptMedium-VLA-v0"
    "InterceptFast-VLA-v0"
    "InterceptGrabSlow-VLA-v0"
    "InterceptGrabMedium-VLA-v0"
    "InterceptGrabFast-VLA-v0"
    "RotateLenientPos-VLA-v0"
    "RotateLenientPosNeg-VLA-v0"
    "RotateStrictPos-VLA-v0"
    "RotateStrictPosNeg-VLA-v0"
    "TakeItBack-VLA-v0"
    "RememberColor3-VLA-v0"
    "RememberColor5-VLA-v0"
    "RememberColor9-VLA-v0"
    "RememberShape3-VLA-v0"
    "RememberShape5-VLA-v0"
    "RememberShape9-VLA-v0"
    "RememberShapeAndColor3x2-VLA-v0"
    "RememberShapeAndColor3x3-VLA-v0"
    "RememberShapeAndColor5x3-VLA-v0"
    # "BunchOfColors3-VLA-v0"
    # "BunchOfColors5-VLA-v0"
    # "BunchOfColors7-VLA-v0"
    # "SeqOfColors3-VLA-v0"
    # "SeqOfColors5-VLA-v0"
    # "SeqOfColors7-VLA-v0"
    # "ChainOfColors3-VLA-v0"
    # "ChainOfColors5-VLA-v0"
    # "ChainOfColors7-VLA-v0"
)

# `--envs` (explicit comma-list) wins over `--preset`. Otherwise resolve preset.
if [[ -n "$ENVS_ARG" ]]; then
    IFS=',' read -r -a ENVS <<< "$ENVS_ARG"
    # Trim whitespace around each entry
    for i in "${!ENVS[@]}"; do
        ENVS[$i]="$(echo -n "${ENVS[$i]}" | xargs)"
    done
else
    case "$PRESET" in
        mikasa5)  ENVS=("${PRESET_MIKASA5[@]}") ;;
        mikasa32) ENVS=("${PRESET_MIKASA32[@]}") ;;
        *)
            echo "ERROR: unknown --preset='$PRESET' (expected: mikasa5|mikasa32)" >&2
            exit 2
            ;;
    esac
fi

if [[ ${#ENVS[@]} -eq 0 ]]; then
    echo "ERROR: --envs resolved to empty list" >&2
    exit 2
fi

echo "Will evaluate ${#ENVS[@]} env(s) sequentially: ${ENVS[*]}"
echo "Using Python: $PYTHON_BIN"

# ── Per-run results dir ──────────────────────────────────────────────────
# Each env gets its own directory:
#
#   $RESULTS_DIR/<CKPT_TAG>/<ENV_ID>/<RUN_STAMP>/
#       logs/EVAL-mikasa_robo-<env_id>-...txt   (clean log written by Python)
#       videos/episode_*_<env_id>_*.mp4
#
# RUN_STAMP includes seed for context; bumping the seed across runs gives
# clearly distinguished directories.
CKPT_TAG="$(basename "$CHECKPOINT")"
# Append optional user note so the same checkpoint evaluated under different
# inference settings doesn't collide in eval_results/.
[[ -n "$RESULTS_NOTE" ]] && CKPT_TAG="${CKPT_TAG}_${RESULTS_NOTE}"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)--seed${SEED}"

# ── Build common args ─────────────────────────────────────────────────────
COMMON_ARGS=(
    --pretrained_checkpoint "$CHECKPOINT"
    --num_trials "$NUM_TRIALS"
    --seed "$SEED"
    --unnorm_key "$UNNORM_KEY"
    --use_wandb "$USE_WANDB"
)
# Only forward memory flags if user explicitly set them. Otherwise the eval
# script auto-detects from the checkpoint via detect_memory_config().
[[ -n "$USE_MEMORY"       ]] && COMMON_ARGS+=(--use_memory       "$USE_MEMORY")
[[ -n "$NUM_MEM_TOKENS"   ]] && COMMON_ARGS+=(--num_mem_tokens   "$NUM_MEM_TOKENS")
[[ -n "$MEMORY_UPDATE"    ]] && COMMON_ARGS+=(--memory_update    "$MEMORY_UPDATE")
[[ -n "$EMA_ALPHA"        ]] && COMMON_ARGS+=(--ema_alpha        "$EMA_ALPHA")
[[ -n "$RECEDING_HORIZON" ]] && COMMON_ARGS+=(--receding_horizon "$RECEDING_HORIZON")
[[ -n "$WANDB_ENTITY"    ]] && COMMON_ARGS+=(--wandb_entity    "$WANDB_ENTITY")
[[ -n "$WANDB_PROJECT"   ]] && COMMON_ARGS+=(--wandb_project   "$WANDB_PROJECT")
[[ -n "$RUN_ID_NOTE"     ]] && COMMON_ARGS+=(--run_id_note     "$RUN_ID_NOTE")

# ── Run each env sequentially ─────────────────────────────────────────────
cd "$REPO_ROOT"

for ENV_ID in "${ENVS[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  ENV: $ENV_ID"
    echo "  CKPT: $CKPT_TAG"
    echo "════════════════════════════════════════════════════════════"

    ENV_OUT_DIR="$RESULTS_DIR/$CKPT_TAG/$ENV_ID/$RUN_STAMP"
    mkdir -p "$ENV_OUT_DIR"

    set +e
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
        "$PYTHON_BIN" "$EVAL_PY" \
            --env_id "$ENV_ID" \
            --output_dir "$ENV_OUT_DIR" \
            "${COMMON_ARGS[@]}"
    EXIT_CODE=$?
    set -e

    # Extract final SR line ("Success rate: 0.NNNN ± 0.NNNN (k/N)") from the
    # clean .txt log file written by Python into the env-specific logs dir.
    SR_LINE=""
    LATEST_LOG="$(ls -1t "$ENV_OUT_DIR/logs/"*.txt 2>/dev/null | head -n1 || true)"
    if [[ -n "$LATEST_LOG" ]]; then
        SR_LINE="$(grep -E "Success rate:" "$LATEST_LOG" | tail -n1 || true)"
    fi
    echo "[$ENV_ID] exit=$EXIT_CODE  ${SR_LINE:-(no SR line found)}"
done

echo ""
echo "finished: $(date -Iseconds)"
