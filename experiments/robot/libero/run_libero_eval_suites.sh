#!/usr/bin/env bash
# Run LIBERO evaluation for one checkpoint across one or more task suites.
#
# Mirrors the MIKASA-Robo launcher
# (`experiments/robot/mikasa_robo/run_mikasa_robo_eval_envs.sh`):
# memory hyperparameters and inference mode are auto-detected from the
# checkpoint by default; override only when needed.
#
# Usage:
#   bash experiments/robot/libero/run_libero_eval_suites.sh \
#       --checkpoint /path/to/<...>_chkpt \
#       [--num_trials_per_task 50] \
#       [--seed 7] \
#       [--cuda_device 2] \
#       [--use_memory true] \
#       [--num_mem_tokens 64] \
#       [--memory_update tbptt|ema] \
#       [--ema_alpha 0.1] \
#       [--receding_horizon true|false] \    # default: auto (true if memory else false)
#       [--unnorm_key libero_combined] \     # default: auto-resolved (suite, suite_no_noops, libero_combined)
#       [--task_id 3] \                       # restrict to one task within a suite
#       [--use_wandb false] \
#       [--wandb_entity X] \
#       [--wandb_project Y] \
#       [--run_id_note tag] \
#       [--results_dir ./eval_results] \
#       [--results_note ""]                   # appended to <CKPT_TAG> dir name
#       [--preset libero4|libero_spatial|libero_object|libero_goal|libero_10|libero_90] \
#       [--suites libero_spatial,libero_object]   # comma-separated; overrides preset
#
# Presets:
#   libero4           — the four canonical LIBERO suites (default; matches the LIBERO
#                       training configurations in LIBERO.md section 3)
#                       libero_spatial, libero_object, libero_goal, libero_10
#   libero_spatial    — only libero_spatial
#   libero_object     — only libero_object
#   libero_goal       — only libero_goal
#   libero_10         — only libero_10 (LIBERO-Long)
#   libero_90         — only libero_90
#
# Examples:
#   # Original OpenVLA-OFT (no memory) on all 4 suites
#   bash .../run_libero_eval_suites.sh \
#       --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
#
#   # mu-VLA with 64 mem tokens, TBPTT (auto-detected from checkpoint)
#   bash .../run_libero_eval_suites.sh \
#       --checkpoint .../base_model_snapshot+...exp_id_7...--150000_chkpt
#
#   # Single suite, single task within it
#   bash .../run_libero_eval_suites.sh \
#       --checkpoint .../my_chkpt \
#       --preset libero_spatial \
#       --task_id 0

set -euo pipefail

# Silence noisy backends inherited by every per-suite Python invocation.
# Mirrors the suppression in run_libero_eval.py — exports here ensure the
# env vars also reach any child processes spawned before Python's own
# `os.environ.setdefault` lines run.
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

# ── Defaults ──────────────────────────────────────────────────────────────
CHECKPOINT=""
NUM_TRIALS_PER_TASK=50
SEED=7
CUDA_DEVICE=0     # tests run on CUDA_VISIBLE_DEVICES=1 by default
# Memory hyperparameters are auto-detected from the checkpoint by default
# (see detect_memory_config in experiments/robot/openvla_utils.py).
USE_MEMORY=""
NUM_MEM_TOKENS=""
MEMORY_UPDATE=""
EMA_ALPHA=""
# Inference mode: "" (auto), "true", "false". Set explicitly to override.
RECEDING_HORIZON=""
# Action un-norm key: "" → eval script resolves (suite name → +_no_noops → libero_combined).
UNNORM_KEY=""
TASK_ID=""
USE_WANDB=false
WANDB_ENTITY=""
WANDB_PROJECT=""
RUN_ID_NOTE=""
RESULTS_DIR="./eval_results"
RESULTS_NOTE=""
SUITES_ARG=""   # comma-separated subset; empty = preset
PRESET="libero4"

# ── Arg parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)         CHECKPOINT="$2";          shift 2 ;;
        --num_trials_per_task) NUM_TRIALS_PER_TASK="$2"; shift 2 ;;
        --seed)               SEED="$2";                shift 2 ;;
        --cuda_device)        CUDA_DEVICE="$2";         shift 2 ;;
        --use_memory)         USE_MEMORY="$2";          shift 2 ;;
        --num_mem_tokens)     NUM_MEM_TOKENS="$2";      shift 2 ;;
        --memory_update)      MEMORY_UPDATE="$2";       shift 2 ;;
        --ema_alpha)          EMA_ALPHA="$2";           shift 2 ;;
        --receding_horizon)   RECEDING_HORIZON="$2";    shift 2 ;;
        --unnorm_key)         UNNORM_KEY="$2";          shift 2 ;;
        --task_id)            TASK_ID="$2";             shift 2 ;;
        --use_wandb)          USE_WANDB="$2";           shift 2 ;;
        --wandb_entity)       WANDB_ENTITY="$2";        shift 2 ;;
        --wandb_project)      WANDB_PROJECT="$2";       shift 2 ;;
        --run_id_note)        RUN_ID_NOTE="$2";         shift 2 ;;
        --results_dir)        RESULTS_DIR="$2";         shift 2 ;;
        --results_note)       RESULTS_NOTE="$2";        shift 2 ;;
        --suites)             SUITES_ARG="$2";          shift 2 ;;
        --preset)             PRESET="$2";              shift 2 ;;
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

# ── Resolve repo root ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EVAL_PY="$SCRIPT_DIR/run_libero_eval.py"

# ── Preflight: verify the project venv has the required Python deps ──────
# Failing here once with a helpful message is much better than failing in
# the per-suite loop with a cryptic ModuleNotFoundError.
if ! ( cd "$REPO_ROOT" && uv run --no-sync python -c "import draccus, torch, transformers, libero, bddl.parsing; from libero.libero.envs import OffScreenRenderEnv" ) >/dev/null 2>&1; then
    echo "ERROR: project venv at '$REPO_ROOT' is missing or has incompatible LIBERO deps" \
         "(expected draccus / torch / transformers / libero / bddl.parsing / libero.libero.envs)." >&2
    echo "       See LIBERO.md section 1 for installation instructions." >&2
    exit 3
fi

# ── Presets ───────────────────────────────────────────────────────────────
PRESET_LIBERO4=(
    "libero_spatial"
    "libero_object"
    "libero_goal"
    "libero_10"
)

# `--suites` (explicit comma-list) wins over `--preset`. Otherwise resolve preset.
if [[ -n "$SUITES_ARG" ]]; then
    IFS=',' read -r -a SUITES <<< "$SUITES_ARG"
    for i in "${!SUITES[@]}"; do
        SUITES[$i]="$(echo -n "${SUITES[$i]}" | xargs)"
    done
else
    case "$PRESET" in
        libero4)        SUITES=("${PRESET_LIBERO4[@]}") ;;
        libero_spatial) SUITES=("libero_spatial") ;;
        libero_object)  SUITES=("libero_object") ;;
        libero_goal)    SUITES=("libero_goal") ;;
        libero_10)      SUITES=("libero_10") ;;
        libero_90)      SUITES=("libero_90") ;;
        *)
            echo "ERROR: unknown --preset='$PRESET' " \
                 "(expected: libero4|libero_spatial|libero_object|libero_goal|libero_10|libero_90)" >&2
            exit 2
            ;;
    esac
fi

if [[ ${#SUITES[@]} -eq 0 ]]; then
    echo "ERROR: --suites resolved to empty list" >&2
    exit 2
fi

echo "Will evaluate ${#SUITES[@]} suite(s) sequentially: ${SUITES[*]}"

# ── Per-run results dir ──────────────────────────────────────────────────
# Each suite gets its own directory:
#
#   $RESULTS_DIR/<CKPT_TAG>/<SUITE>/<RUN_STAMP>/
#       logs/EVAL-libero-<suite>-...txt
#       videos/<rollout>.mp4
#
CKPT_TAG="$(basename "${CHECKPOINT%/}")"
[[ -n "$RESULTS_NOTE" ]] && CKPT_TAG="${CKPT_TAG}_${RESULTS_NOTE}"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)--seed${SEED}"

# ── Build common args ─────────────────────────────────────────────────────
COMMON_ARGS=(
    --pretrained_checkpoint "$CHECKPOINT"
    --num_trials_per_task "$NUM_TRIALS_PER_TASK"
    --seed "$SEED"
    --use_wandb "$USE_WANDB"
)
# Only forward optional flags if user explicitly set them. Otherwise the eval
# script auto-detects from the checkpoint.
[[ -n "$USE_MEMORY"       ]] && COMMON_ARGS+=(--use_memory       "$USE_MEMORY")
[[ -n "$NUM_MEM_TOKENS"   ]] && COMMON_ARGS+=(--num_mem_tokens   "$NUM_MEM_TOKENS")
[[ -n "$MEMORY_UPDATE"    ]] && COMMON_ARGS+=(--memory_update    "$MEMORY_UPDATE")
[[ -n "$EMA_ALPHA"        ]] && COMMON_ARGS+=(--ema_alpha        "$EMA_ALPHA")
[[ -n "$RECEDING_HORIZON" ]] && COMMON_ARGS+=(--receding_horizon "$RECEDING_HORIZON")
[[ -n "$UNNORM_KEY"       ]] && COMMON_ARGS+=(--unnorm_key       "$UNNORM_KEY")
[[ -n "$TASK_ID"          ]] && COMMON_ARGS+=(--task_id          "$TASK_ID")
[[ -n "$WANDB_ENTITY"     ]] && COMMON_ARGS+=(--wandb_entity     "$WANDB_ENTITY")
[[ -n "$WANDB_PROJECT"    ]] && COMMON_ARGS+=(--wandb_project    "$WANDB_PROJECT")
[[ -n "$RUN_ID_NOTE"      ]] && COMMON_ARGS+=(--run_id_note      "$RUN_ID_NOTE")

# ── Run each suite sequentially ───────────────────────────────────────────
cd "$REPO_ROOT"

for SUITE in "${SUITES[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  SUITE: $SUITE"
    echo "  CKPT:  $CKPT_TAG"
    echo "════════════════════════════════════════════════════════════"

    SUITE_OUT_DIR="$RESULTS_DIR/$CKPT_TAG/$SUITE/$RUN_STAMP"
    mkdir -p "$SUITE_OUT_DIR"

    set +e
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
        uv run --no-sync python "$EVAL_PY" \
            --task_suite_name "$SUITE" \
            --output_dir "$SUITE_OUT_DIR" \
            "${COMMON_ARGS[@]}"
    EXIT_CODE=$?
    set -e

    # Extract final SR line ("Success rate: 0.NNNN ± 0.NNNN (k/N)") from the
    # clean .txt log file written by Python into the suite-specific logs dir.
    SR_LINE=""
    LATEST_LOG="$(ls -1t "$SUITE_OUT_DIR/logs/"*.txt 2>/dev/null | head -n1 || true)"
    if [[ -n "$LATEST_LOG" ]]; then
        SR_LINE="$(grep -E "Success rate:" "$LATEST_LOG" | tail -n1 || true)"
    fi
    echo "[$SUITE] exit=$EXIT_CODE  ${SR_LINE:-(no SR line found)}"
done

echo ""
echo "finished: $(date -Iseconds)"
