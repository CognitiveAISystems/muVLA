"""
run_libero_eval.py

Evaluates a trained OpenVLA-OFT / mu-OpenVLA-OFT policy on a LIBERO task suite.

Mirrors the MIKASA-Robo eval (`experiments/robot/mikasa_robo/run_mikasa_robo_eval.py`)
one-to-one in terms of UX:
  * Auto-detects mu-VLA memory (num_mem_tokens, memory_update, ema_alpha) from
    the checkpoint via `detect_memory_config()`. The user only points at the
    run dir; CLI overrides take priority.
  * Forces receding-horizon (per-step requery) when memory is enabled, so that
    eval matches training (mu-VLA rolls memory once per sim step).
  * Rich Panel header + Progress + final Panel; per-run text log captures the
    full transcript for forensic debugging (sanity prints, mem-check, per-
    episode results).
  * Mean ± SE summary over `success` (1 if the task was completed, else 0).
  * Per-run output_dir layout:
        <output_dir>/logs/EVAL-libero-<suite>-...txt
        <output_dir>/videos/<rollout>.mp4

Usage:
    cd ~/mu-vla
    uv run python experiments/robot/libero/run_libero_eval.py \\
        --pretrained_checkpoint runs/my_checkpoint \\
        --task_suite_name libero_spatial \\
        [--num_trials_per_task 50] \\
        [--task_id 3]                 # restrict to one task within the suite
"""

import os  # noqa: E402  — keep before TF/torch import
import sys

# ── Silence noisy third-party backends BEFORE they are imported ──────────────
# These env vars must be set BEFORE TensorFlow / transformers / robosuite imports
# fire (TF reads them at module-load time). Setting them here once means a
# clean console for the whole eval.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")        # suppress TF info/warn/error
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")        # suppress oneDNN banner
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # suppress fork warning
os.environ.setdefault("ABSL_LOGGING_VERBOSITY", "0")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import contextlib
import io
import json
import logging
import warnings
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

# Suppress chatty loggers and import-time UserWarnings/FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
for _noisy in ("transformers", "tensorflow", "absl", "h5py", "matplotlib", "PIL"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

import draccus
import numpy as np
import torch
import wandb

# Rich UI
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Append project root so that interpreter can find experiments.robot
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Heavy project imports below also print at module-load time (e.g.
# `prismatic.vla.constants` prints "Using ... constants:" on import).
# Swallow that into a buffer so the console stays clean.
_startup_stdout = io.StringIO()
with contextlib.redirect_stdout(_startup_stdout):
    from libero.libero import benchmark

    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        get_libero_wrist_image,
        quat2axisangle,
        save_rollout_video,
    )
    from experiments.robot.openvla_utils import (
        detect_memory_config,
        get_action_head,
        get_memory_module,
        get_noisy_action_projector,
        get_processor,
        get_proprio_projector,
        resize_image_for_policy,
    )
    from experiments.robot.robot_utils import (
        DATE_TIME,
        get_action,
        get_image_resize_size,
        get_model,
        invert_gripper_action,
        normalize_gripper_action,
        set_seed_everywhere,
    )
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Single rich console for all human-facing output. Mirrors the mikasa eval
# setup. `force_terminal=True` keeps ANSI sequences flowing under `tee` so
# bash launchers can capture stdout without breaking the live progress bar.
console = Console(highlight=False, force_terminal=True)

# File-only logger (forensic debugging). All eval transcripts go here.
logger = logging.getLogger("libero_eval")
logger.setLevel(logging.INFO)
logger.propagate = False  # never echo to root → no stdout duplication


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite (per the original openvla-oft repo).
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,   # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,     # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,       # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,       # longest training demo has 373 steps
}


@dataclass
class GenerateConfig:
    # fmt: off

    ###########################################################################
    # Model-specific parameters
    ###########################################################################
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""

    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2           # base camera + wrist camera
    use_proprio: bool = True

    center_crop: bool = True               # Must be True when model was trained with image_aug
    num_open_loop_steps: int = 8           # Should match NUM_ACTIONS_CHUNK
    # `receding_horizon=None` means "auto": enabled when memory is on (must
    # mirror the per-step memory rollout used during training), disabled
    # otherwise (matching standard OpenVLA-OFT chunked inference).
    # Pass --receding_horizon=true|false to override.
    receding_horizon: Optional[bool] = None

    lora_rank: int = 32

    # Action un-normalization key. Empty → resolved by `check_unnorm_key()`:
    #   1. cfg.task_suite_name (e.g. 'libero_spatial')
    #   2. cfg.task_suite_name + '_no_noops' (training key for OXE pipeline)
    #   3. 'libero_combined' (mu-VLA episodic dataset key — multi-suite training)
    # Pass --unnorm_key explicitly to override.
    unnorm_key: Union[str, Path] = ""

    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # mu-VLA recurrent memory.
    # By default, all memory hyperparameters are auto-detected from the
    # checkpoint via `detect_memory_config()`. The user only needs to point
    # `--pretrained_checkpoint` at the run directory; if a `memory_module--*`
    # file exists, memory is enabled with the right `num_mem_tokens` and
    # update rule. CLI overrides take priority.
    use_memory: Optional[bool] = None      # None → auto-detect from checkpoint
    num_mem_tokens: Optional[int] = None   # None → auto-detect from checkpoint
    memory_update: Optional[str] = None    # "tbptt"|"ema"; None → auto-detect
    ema_alpha: Optional[float] = None      # None → from memory_meta.json or 0.1
    # Attention mask mode used by the LLM when memory is active.
    #   "custom" — mu-VLA context/action split (default for old/new mu-VLA ckpts).
    #   "full"   — standard OpenVLA-OFT bidirectional (memory + full mask ablation).
    # None → auto-detect from `memory_meta.json` (default "custom" for old ckpts).
    attention_mask_mode: Optional[str] = None

    ###########################################################################
    # LIBERO environment-specific parameters
    ###########################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # one of TaskSuite values
    num_steps_wait: int = 10                         # warmup so objects stabilize
    num_trials_per_task: int = 50                    # episodes per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT" or path to JSON
    env_img_res: int = 256                           # raw env-rendered resolution
    # If set, only run the task at this index within the suite (0..n_tasks-1).
    # Default: run all tasks in the suite.
    task_id: Optional[int] = None

    ###########################################################################
    # Utils
    ###########################################################################
    run_id_note: Optional[str] = None

    # Per-run output directory. If set, the eval writes everything (text logs +
    # rollout mp4s) under this single directory:
    #     <output_dir>/logs/<run_id>.txt
    #     <output_dir>/videos/*.mp4
    # If None, falls back to the legacy split (`local_log_dir` for text,
    # `./rollouts/<DATE>` for video).
    output_dir: Optional[str] = None
    local_log_dir: str = "./experiments/logs"

    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"

    seed: int = 7   # global torch/numpy RNG seed (for any model-side stochasticity)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration and auto-detect memory hyperparameters."""
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must not be empty!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, (
            "Expecting `center_crop==True` because model was trained with image augmentations!"
        )

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), (
        "Cannot use both 8-bit and 4-bit quantization!"
    )

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], (
        f"Invalid task suite: {cfg.task_suite_name}"
    )

    # ── Auto-detect mu-VLA memory hyperparameters from the checkpoint ──
    # CLI values (anything not None) take priority over auto-detected values.
    detected = detect_memory_config(str(cfg.pretrained_checkpoint))

    if cfg.use_memory is None:
        cfg.use_memory = bool(detected.get("use_memory", False))
    elif cfg.use_memory and not detected.get("use_memory", False):
        console.print(
            "[yellow]warning:[/yellow] use_memory=True passed via CLI but no "
            "memory_module--*_checkpoint.pt found in checkpoint dir. "
            "Memory loading WILL fail."
        )

    if cfg.use_memory:
        if cfg.num_mem_tokens is None:
            cfg.num_mem_tokens = int(detected.get("num_mem_tokens", 4))
        elif "num_mem_tokens" in detected and cfg.num_mem_tokens != detected["num_mem_tokens"]:
            console.print(
                f"[yellow]warning:[/yellow] num_mem_tokens override: "
                f"CLI={cfg.num_mem_tokens} but checkpoint has "
                f"{detected['num_mem_tokens']}. Using CLI value, this will "
                f"fail loading unless intentional."
            )
        if cfg.memory_update is None:
            cfg.memory_update = str(detected.get("memory_update", "tbptt"))
        if cfg.ema_alpha is None:
            cfg.ema_alpha = float(detected.get("ema_alpha", 0.1))
        if cfg.attention_mask_mode is None:
            cfg.attention_mask_mode = str(detected.get("attention_mask_mode", "custom"))
        elif "attention_mask_mode" in detected and cfg.attention_mask_mode != detected["attention_mask_mode"]:
            console.print(
                f"[yellow]warning:[/yellow] attention_mask_mode override: "
                f"CLI={cfg.attention_mask_mode!r} but checkpoint trained with "
                f"{detected['attention_mask_mode']!r}. This is a TRAIN/EVAL "
                f"mismatch and will likely degrade performance."
            )
        assert cfg.memory_update in ("tbptt", "ema"), (
            f"Invalid memory_update={cfg.memory_update!r} (must be 'tbptt' or 'ema')"
        )
        assert 0.0 <= cfg.ema_alpha <= 1.0, (
            f"ema_alpha must be in [0, 1], got {cfg.ema_alpha}"
        )
        assert cfg.attention_mask_mode in ("custom", "full"), (
            f"Invalid attention_mask_mode={cfg.attention_mask_mode!r} (must be 'custom' or 'full')"
        )
    else:
        # Fill defaults so downstream code doesn't trip over None values.
        cfg.num_mem_tokens = cfg.num_mem_tokens or 0
        cfg.memory_update = cfg.memory_update or "tbptt"
        cfg.ema_alpha = cfg.ema_alpha if cfg.ema_alpha is not None else 0.1
        # No memory → mask mode is irrelevant; the LLM uses default OpenVLA-OFT path.
        cfg.attention_mask_mode = cfg.attention_mask_mode or "custom"

    # ── Resolve receding_horizon ──────────────────────────────────────────
    # Memory-enabled checkpoints REQUIRE receding-horizon execution. During
    # training, mu-VLA rolls memory forward by exactly one step per dataloader
    # yield (one sim step). Default chunked inference (num_open_loop_steps=8)
    # would only update memory once every 8 sim steps, feeding the model
    # severely out-of-distribution memory states from step 8 onward.
    if cfg.receding_horizon is None:
        cfg.receding_horizon = bool(cfg.use_memory)
    elif cfg.use_memory and not cfg.receding_horizon:
        console.print(
            "[yellow]warning:[/yellow] use_memory=True but "
            "receding_horizon=False explicitly set. This MISMATCHES training "
            "(which rolls memory once per sim step) and will likely produce "
            "near-random behaviour. Strongly recommend "
            "--receding_horizon=true for memory-enabled checkpoints."
        )


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Resolve the action un-normalization key against the model's norm_stats.

    Three-level fallback (covers all training-mode → eval-suite combinations):
      1. cfg.unnorm_key as-is, if non-empty (CLI override)
      2. cfg.task_suite_name (e.g. 'libero_spatial' — original OXE pipeline)
      3. cfg.task_suite_name + '_no_noops' (training-time RLDS dataset name)
      4. 'libero_combined' (mu-VLA episodic dataset — joint multi-suite stats)

    Also: if the resolved stats are missing the LIBERO action mask (the older
    LIBEROVLAEpisodicDataset releases did not persist it), inject the canonical
    `[True]*6 + [False]` so `_unnormalize_actions` passes the gripper dim
    through untouched. Without this the model's near-binary {0, 1} gripper
    output gets remapped through q01/q99 and the gripper can never close
    (→ 0% SR on every LIBERO suite).
    """
    candidates = []
    if cfg.unnorm_key:
        candidates.append(str(cfg.unnorm_key))
    candidates.extend([
        cfg.task_suite_name,
        f"{cfg.task_suite_name}_no_noops",
        "libero_combined",
    ])

    resolved_key = None
    for key in candidates:
        if key in model.norm_stats:
            if key != cfg.unnorm_key:
                console.print(
                    f"[cyan]info:[/cyan] resolved unnorm_key='{key}' "
                    f"(checked: {candidates})"
                )
            cfg.unnorm_key = key
            resolved_key = key
            break

    if resolved_key is None:
        raise AssertionError(
            f"None of unnorm_key candidates {candidates} found in model norm_stats. "
            f"Available keys: {list(model.norm_stats.keys())}"
        )

    # Inject canonical LIBERO action mask if missing. This is a no-op for the
    # released openvla-7b-oft-libero-* checkpoints (which already store
    # mask=[True]*6+[False]) and a critical fix for mu-VLA episodic
    # checkpoints whose stats were saved without the mask.
    action_stats = model.norm_stats[resolved_key].get("action")
    if isinstance(action_stats, dict):
        existing_mask = action_stats.get("mask")
        needs_mask = (
            existing_mask is None
            or (isinstance(existing_mask, np.ndarray) and existing_mask.size == 0)
        )
        if needs_mask:
            q01 = action_stats.get("q01")
            if q01 is not None and len(q01) == 7:
                libero_mask = np.array([True] * 6 + [False], dtype=bool)
                action_stats["mask"] = libero_mask
                console.print(
                    "[cyan]info:[/cyan] injected canonical LIBERO action mask "
                    "[True]*6+[False] (gripper passthrough) into "
                    f"norm_stats[{resolved_key!r}]['action'] "
                    "(was missing — without this the gripper never closes)."
                )


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    model = get_model(cfg)

    # Load proprio projector — LIBERO uses 8D proprio
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    # Load action head
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # [mu-VLA] Load recurrent memory module if checkpoint contains one
    memory_module = None
    if cfg.use_memory:
        memory_module = get_memory_module(cfg, model.llm_dim)

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor, memory_module


def setup_logging(cfg: GenerateConfig):
    """Set up the per-run text log file (and optionally wandb).

    Mirrors the mikasa eval. Console output is handled separately by rich
    Panels + Progress in `main()`; everything else (sanity prints, mem-check,
    per-episode results, final summary) lands in this text file.
    """
    run_id = f"EVAL-libero-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    if cfg.output_dir is not None:
        log_dir = os.path.join(cfg.output_dir, "logs")
    else:
        log_dir = cfg.local_log_dir
    os.makedirs(log_dir, exist_ok=True)
    local_log_filepath = os.path.join(log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")

    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Append a message to the per-run text log. Console output is handled
    separately by the rich Panel/Progress in `main()`."""
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    initial_states = task_suite.get_task_init_states(task_id)
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    log_message("Using default initial states", log_file)
    return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
    }
    return observation, img


def process_action(action, model_family):
    """Process model-predicted action before sending to LIBERO env.

    Unlike MIKASA, LIBERO requires:
      - normalize_gripper_action: gripper [0,1] → [-1,+1]
      - invert_gripper_action: dataloader flipped sign (0=close → -1=open)
    """
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    memory_module=None,
    initial_state=None,
    log_file=None,
    debug_first_step: bool = False,
    progress: Optional[Progress] = None,
    step_task_id: Optional[int] = None,
):
    """Run a single LIBERO episode.

    Returns (success, replay_images, ep_stats).
    `success` follows LIBERO convention: True iff `done` was True at any step
    after the warmup (`num_steps_wait`).

    If `progress` and `step_task_id` are provided, the per-episode step bar
    is reset to 0/max_steps at the start and advanced once per simulator step.
    """
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue (only used when receding_horizon=False)
    if not cfg.receding_horizon and cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        log_message(
            f"WARNING: num_open_loop_steps ({cfg.num_open_loop_steps}) != "
            f"NUM_ACTIONS_CHUNK ({NUM_ACTIONS_CHUNK})",
            log_file,
        )
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # [mu-VLA] Initialize memory state at episode start. Each LIBERO episode
    # is an independent rollout, so memory must be reset to initial_memory.
    mem_state = None
    if cfg.use_memory and memory_module is not None:
        mem_state = memory_module.get_initial_state(batch_size=1)
        mem_state = mem_state.to(
            dtype=torch.bfloat16,
            device=next(memory_module.parameters()).device,
        )

    max_steps = TASK_MAX_STEPS[TaskSuite(cfg.task_suite_name)]
    total_steps = max_steps + cfg.num_steps_wait

    # Reset per-episode step bar (rich UI)
    if progress is not None and step_task_id is not None:
        progress.reset(step_task_id, total=total_steps)
        progress.update(step_task_id, completed=0, visible=True)

    replay_images = []
    success = False

    # Cache norm stats for debug-first-step sanity print (only the first time
    # this is set inside the run we hit the debug path).
    proprio_norm_stats = None
    if debug_first_step and cfg.use_proprio:
        try:
            proprio_norm_stats = model.norm_stats[cfg.unnorm_key]["proprio"]
        except (AttributeError, KeyError):
            proprio_norm_stats = None

    debug_logged = not debug_first_step

    # Per-episode invariant counters. With receding-horizon + memory, all
    # three must agree at episode end (one forward → one mem update → one
    # action executed per sim step). With chunked execution they don't.
    forward_calls = 0
    mem_updates = 0
    sim_steps = 0
    mem_state_delta_l1_max = 0.0
    mem_state_delta_l1_min = float("inf")

    t = 0
    try:
        while t < total_steps:
            # Warmup: do nothing while objects settle
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                if progress is not None and step_task_id is not None:
                    progress.update(step_task_id, advance=1)
                continue

            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            if cfg.receding_horizon:
                # Receding horizon: re-query model every step, execute only first action.
                mem_input = mem_state  # save M^in_t for EMA blend
                result = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                    memory_state=mem_state,
                    use_memory_mask=cfg.use_memory,
                    attention_mask_mode=cfg.attention_mask_mode,
                )
                forward_calls += 1
                if cfg.use_memory and mem_state is not None:
                    actions, new_mem = result
                    if cfg.memory_update == "ema":
                        new_mem_state = (
                            cfg.ema_alpha * new_mem
                            + (1.0 - cfg.ema_alpha) * mem_input
                        )
                    else:
                        new_mem_state = new_mem
                    delta = (new_mem_state.float() - mem_input.float()).abs().mean().item()
                    mem_state_delta_l1_max = max(mem_state_delta_l1_max, delta)
                    mem_state_delta_l1_min = min(mem_state_delta_l1_min, delta)
                    mem_state = new_mem_state
                    mem_updates += 1
                else:
                    actions = result
                assert len(actions) == NUM_ACTIONS_CHUNK, (
                    f"Expected chunk of length {NUM_ACTIONS_CHUNK}, got {len(actions)}"
                )
                action = actions[0]
            else:
                # Open-loop chunked execution (default for non-memory checkpoints).
                if len(action_queue) == 0:
                    mem_input = mem_state
                    result = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        use_film=cfg.use_film,
                        memory_state=mem_state,
                        use_memory_mask=cfg.use_memory,
                        attention_mask_mode=cfg.attention_mask_mode,
                    )
                    forward_calls += 1
                    if cfg.use_memory and mem_state is not None:
                        actions, new_mem = result
                        if cfg.memory_update == "ema":
                            mem_state = (
                                cfg.ema_alpha * new_mem
                                + (1.0 - cfg.ema_alpha) * mem_input
                            )
                        else:
                            mem_state = new_mem
                        mem_updates += 1
                    else:
                        actions = result
                    action_queue.extend(actions)
                action = action_queue.popleft()

            # Sanity-print on the very first action step of the very first
            # debug episode. Capture proprio BEFORE get_action runs above —
            # get_vla_action() mutates `observation["state"]` in place.
            if not debug_logged:
                debug_logged = True
                proprio_raw = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )
                proprio_arr = np.asarray(proprio_raw)
                proprio_norm_str = "n/a"
                if proprio_norm_stats is not None:
                    try:
                        from experiments.robot.openvla_utils import normalize_proprio
                        proprio_norm = normalize_proprio(proprio_raw, proprio_norm_stats)
                        proprio_norm_str = (
                            f"min={float(np.min(proprio_norm)):.3f} "
                            f"max={float(np.max(proprio_norm)):.3f}"
                        )
                    except Exception as ex:
                        proprio_norm_str = f"normalize_proprio failed: {ex}"
                log_message(
                    f"[sanity] task={task_description!r}",
                    log_file,
                )
                log_message(
                    f"[sanity] proprio_raw shape={proprio_arr.shape} "
                    f"min={float(np.min(proprio_arr)):.3f} "
                    f"max={float(np.max(proprio_arr)):.3f} | "
                    f"proprio_norm: {proprio_norm_str}",
                    log_file,
                )
                action_arr = np.asarray(action)
                log_message(
                    f"[sanity] action[0] (post-process) shape={action_arr.shape} "
                    f"min={float(np.min(action_arr)):.3f} "
                    f"max={float(np.max(action_arr)):.3f} "
                    f"gripper={float(action_arr[-1]):.3f} (expected in [-1, 1])",
                    log_file,
                )
                if mem_state is not None:
                    log_message(
                        f"[sanity] mem_state shape={tuple(mem_state.shape)} "
                        f"dtype={mem_state.dtype} device={mem_state.device}",
                        log_file,
                    )
                _b = np.asarray(get_libero_image(obs))
                _w = np.asarray(get_libero_wrist_image(obs))
                log_message(
                    f"[sanity] base_image shape={_b.shape} dtype={_b.dtype} "
                    f"min={int(_b.min())} max={int(_b.max())} mean={float(_b.mean()):.1f}",
                    log_file,
                )
                log_message(
                    f"[sanity] wrist_image shape={_w.shape} dtype={_w.dtype} "
                    f"min={int(_w.min())} max={int(_w.max())} mean={float(_w.mean()):.1f}",
                    log_file,
                )

            # Process action and step env
            action = process_action(action, cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            sim_steps += 1
            t += 1
            if progress is not None and step_task_id is not None:
                progress.update(step_task_id, advance=1)
            if done:
                success = True
                break

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)
        import traceback
        log_message(traceback.format_exc(), log_file)

    # ── Invariant check: receding-horizon + memory means
    #     forward_calls == sim_steps == mem_updates,
    #     i.e. one model query and one memory update per simulator tick. ──
    if cfg.receding_horizon and cfg.use_memory:
        if forward_calls != sim_steps or mem_updates != sim_steps:
            log_message(
                f"[mem-check] FAIL: forward_calls={forward_calls} "
                f"mem_updates={mem_updates} sim_steps={sim_steps} — "
                f"these must be equal in receding-horizon + memory mode",
                log_file,
            )
            assert forward_calls == sim_steps == mem_updates, (
                "Memory rollout broken: forward/mem-update count != sim steps"
            )
        if mem_state_delta_l1_min == 0.0 and sim_steps > 1:
            log_message(
                "[mem-check] WARNING: mem_state delta hit zero at some step — "
                "memory may be frozen for that step.",
                log_file,
            )

    if debug_first_step:
        log_message(
            f"[mem-check] sim_steps={sim_steps} forward_calls={forward_calls} "
            f"mem_updates={mem_updates} | mem_delta_L1 per step "
            f"min={mem_state_delta_l1_min if mem_state_delta_l1_min != float('inf') else 'n/a'} "
            f"max={mem_state_delta_l1_max}",
            log_file,
        )

    ep_stats = {"steps": sim_steps}
    return success, replay_images, ep_stats


def _build_header_panel(cfg: GenerateConfig, num_tasks: int) -> Panel:
    """Pre-run config panel — replaces the old multi-line `log_message` dump."""
    if cfg.use_memory:
        if cfg.memory_update == "ema":
            mem_line = f"[green]on[/]  · {cfg.num_mem_tokens} tokens · EMA α={cfg.ema_alpha} (auto)"
        else:
            mem_line = f"[green]on[/]  · {cfg.num_mem_tokens} tokens · TBPTT (auto)"
    else:
        mem_line = "[dim]off[/]"

    inference_line = (
        "receding-horizon (per-step requery)" if cfg.receding_horizon
        else f"open-loop chunk ({cfg.num_open_loop_steps} actions)"
    )

    if cfg.task_id is not None:
        scope = f"task {cfg.task_id} only · {cfg.num_trials_per_task} episodes"
    else:
        scope = f"{num_tasks} tasks × {cfg.num_trials_per_task} episodes = {num_tasks * cfg.num_trials_per_task} total"

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column()
    table.add_row("Checkpoint", os.path.basename(str(cfg.pretrained_checkpoint).rstrip("/")))
    table.add_row("Memory", mem_line)
    table.add_row("Inference", inference_line)
    table.add_row("Suite", cfg.task_suite_name)
    table.add_row("Validation", scope)
    table.add_row("Episode max", f"{TASK_MAX_STEPS[TaskSuite(cfg.task_suite_name)]} sim-steps (+{cfg.num_steps_wait} warmup)")
    table.add_row("Unnorm key", str(cfg.unnorm_key) if cfg.unnorm_key else "[dim](resolved at load time)[/]")
    table.add_row(
        "Metric",
        "[dim]success rate = mean(success) ± SE over all episodes; "
        "success = 1 iff env.done was True at any post-warmup step[/]",
    )
    return Panel(
        table,
        title=f"[bold]mu-VLA eval · LIBERO[/]",
        title_align="left",
        border_style="cyan",
    )


def _build_results_panel(suite_name: str, n: int, k: int, mean: float, std: float, se: float) -> Panel:
    """Post-run results panel — single line per metric, no banners."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column()
    table.add_row("Episodes", f"{n}")
    table.add_row("Successes", f"{k}  [dim](episodes where env.done was True at any post-warmup step)[/]")
    table.add_row("Success rate", f"[bold]{mean:.4f} ± {se:.4f}[/]  ({k}/{n})  [dim]= mean(success) ± SE  (SE = std/√N)[/]")
    table.add_row("Std", f"{std:.4f}  [dim]= sample std (Bernoulli variance, ddof=1)[/]")
    return Panel(
        table,
        title=f"[bold green]{suite_name} · final[/]",
        title_align="left",
        border_style="green",
    )


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main entry: evaluate a trained policy on a LIBERO task suite."""
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    log_file, local_log_filepath, run_id = setup_logging(cfg)
    log_message(f"Run ID: {run_id}", log_file)
    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Checkpoint: {cfg.pretrained_checkpoint}", log_file)
    log_message(f"Num trials per task: {cfg.num_trials_per_task}", log_file)
    log_message(
        f"Memory: use={cfg.use_memory} num_mem_tokens={cfg.num_mem_tokens} "
        f"update={cfg.memory_update} ema_alpha={cfg.ema_alpha}",
        log_file,
    )
    log_message(
        f"Inference mode: {'receding-horizon' if cfg.receding_horizon else f'chunked ({cfg.num_open_loop_steps})'}",
        log_file,
    )

    # Resolve videos directory up-front so we can show it to the user.
    if cfg.output_dir is not None:
        videos_root = os.path.join(cfg.output_dir, "videos")
    else:
        from experiments.robot.robot_utils import DATE
        videos_root = f"./rollouts/{DATE}"

    # Load model + heads + memory under a status spinner — same UX as mikasa.
    with console.status("[cyan]Loading model + heads…[/]", spinner="dots"):
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            model, action_head, proprio_projector, noisy_action_projector, processor, memory_module = (
                initialize_model(cfg)
            )

    resize_size = get_image_resize_size(cfg)
    log_message(f"Image resize size: {resize_size}", log_file)

    # Initialize LIBERO benchmark + task suite under a spinner too — the
    # benchmark loader prints a few "Loading benchmark XYZ…" lines.
    with console.status(f"[cyan]Loading LIBERO suite '{cfg.task_suite_name}'…[/]", spinner="dots"):
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    if cfg.task_id is not None:
        assert 0 <= cfg.task_id < num_tasks, (
            f"--task_id={cfg.task_id} out of range for suite "
            f"'{cfg.task_suite_name}' which has {num_tasks} tasks (0..{num_tasks - 1})"
        )
        task_indices = [cfg.task_id]
    else:
        task_indices = list(range(num_tasks))

    # ── Console: opening config panel + paths ────────────────────────────
    console.print()
    console.print(_build_header_panel(cfg, num_tasks))
    log_dir_for_print = os.path.dirname(local_log_filepath)
    console.print(
        f"  [bold cyan]Logs:[/]   [dim]{log_dir_for_print}[/]  "
        f"[dim](this run: {os.path.basename(local_log_filepath)})[/]"
    )
    console.print(f"  [bold cyan]Videos:[/] [dim]{videos_root}/<rollout>.mp4[/]")
    console.print()

    # Aggregated results: one bool per episode (success).
    successes: list = []

    # Per-task SR breakdown for the file log.
    per_task_sr: dict = {}

    # ── Console: rich.Progress with three task rows ──────────────────────
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description:>10}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("• ETA"),
        TimeRemainingColumn(),
        TextColumn("• [cyan]{task.fields[running]}[/]"),
        console=console,
        transient=False,
    )
    suite_task_id = progress.add_task(
        "Tasks",
        total=len(task_indices) * cfg.num_trials_per_task,
        running="0/0 = 0.00%",
    )
    episode_task_id = progress.add_task(
        "Episode",
        total=cfg.num_trials_per_task,
        running="",
    )
    step_task_id = progress.add_task(
        "Step",
        total=TASK_MAX_STEPS[TaskSuite(cfg.task_suite_name)] + cfg.num_steps_wait,
        running="",
    )

    overall_first = True  # whether we still owe a sanity log

    with progress:
        for task_id in task_indices:
            task = task_suite.get_task(task_id)
            initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

            # Build the env for this task. Suppress its prints.
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

            log_message(f"\n=== Task {task_id}: {task_description} ===", log_file)

            # Reset per-task episode bar
            progress.reset(episode_task_id, total=cfg.num_trials_per_task)
            progress.update(episode_task_id, completed=0, visible=True, running="")

            task_episodes, task_successes = 0, 0
            for episode_idx in range(cfg.num_trials_per_task):
                # Resolve initial state
                if cfg.initial_states_path == "DEFAULT":
                    initial_state = initial_states[episode_idx]
                else:
                    initial_states_task_key = task_description.replace(" ", "_")
                    episode_key = f"demo_{episode_idx}"
                    if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                        log_message(
                            f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!",
                            log_file,
                        )
                        continue
                    initial_state = np.array(
                        all_initial_states[initial_states_task_key][episode_key]["initial_state"]
                    )

                log_message(
                    f"Episode {episode_idx + 1}/{cfg.num_trials_per_task} — {task_description}",
                    log_file,
                )

                debug_first_step = overall_first
                if overall_first:
                    overall_first = False

                success, replay_images, ep_stats = run_episode(
                    cfg,
                    env,
                    task_description,
                    model,
                    resize_size,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    memory_module=memory_module,
                    initial_state=initial_state,
                    log_file=log_file,
                    debug_first_step=debug_first_step,
                    progress=progress,
                    step_task_id=step_task_id,
                )

                successes.append(int(bool(success)))
                task_episodes += 1
                if success:
                    task_successes += 1

                # Save replay video.
                try:
                    save_rollout_video(
                        replay_images,
                        len(successes),
                        success=success,
                        task_description=task_description,
                        log_file=log_file,
                        output_dir=videos_root,
                    )
                except Exception as e:
                    log_message(
                        f"WARNING: failed to save rollout video for episode "
                        f"{len(successes)}: {e}",
                        log_file,
                    )

                # Running aggregate update
                total_episodes = len(successes)
                total_successes = sum(successes)
                running_mean = total_successes / total_episodes
                log_message(
                    f"  → Success: {success} | steps: {ep_stats['steps']} | "
                    f"running SR {total_successes}/{total_episodes} = {running_mean:.2%}",
                    log_file,
                )

                progress.update(
                    suite_task_id,
                    advance=1,
                    running=f"SR {total_successes}/{total_episodes} = {running_mean:.2%}",
                )
                progress.update(
                    episode_task_id,
                    advance=1,
                    running=f"task {task_id}: {task_successes}/{task_episodes}",
                )

                if cfg.use_wandb:
                    wandb.log({
                        "global_episode": total_episodes,
                        "success": int(success),
                        "running_mean": running_mean,
                    })

            # Per-task SR
            task_sr = task_successes / task_episodes if task_episodes > 0 else 0.0
            per_task_sr[task_description] = (task_successes, task_episodes, task_sr)
            log_message(
                f"Task {task_id} ({task_description}) SR: {task_sr:.4f} "
                f"({task_successes}/{task_episodes})",
                log_file,
            )
            if cfg.use_wandb:
                wandb.log({
                    f"success_rate/{task_description}": task_sr,
                    f"num_episodes/{task_description}": task_episodes,
                })

    # ── Final mean ± SE summary ───────────────────────────────────────────
    s_arr = np.asarray(successes, dtype=np.float64)
    n = int(s_arr.size)
    k = int(s_arr.sum())
    mean = float(s_arr.mean()) if n > 0 else 0.0
    std = float(s_arr.std(ddof=1)) if n > 1 else 0.0
    se = float(std / np.sqrt(n)) if n > 0 else 0.0

    # Per-task breakdown (kept in file log only).
    log_message("\n" + "=" * 60, log_file)
    log_message(f"Per-task breakdown for {cfg.task_suite_name}", log_file)
    for desc, (succ, ep, sr) in per_task_sr.items():
        log_message(f"  {desc}: {sr:.4f} ({succ}/{ep})", log_file)
    log_message("=" * 60, log_file)

    log_message(
        f"\n{'=' * 60}\n"
        f"Final Results for {cfg.task_suite_name}\n"
        f"  Total episodes:   {n}\n"
        f"  Total successes:  {k}\n"
        f"  Success rate:     {mean:.4f} ± {se:.4f} ({k}/{n})  [SE = std/sqrt(N)]\n"
        f"  Std:              {std:.4f}\n"
        f"{'=' * 60}",
        log_file,
    )

    # ── Console: closing summary panel ───────────────────────────────────
    console.print()
    console.print(_build_results_panel(cfg.task_suite_name, n, k, mean, std, se))
    console.print()
    # Plain echo so bash launchers parsing stdout (`tee` + grep) still see a
    # `Success rate:` line.
    console.print(f"  Success rate:     {mean:.4f} ± {se:.4f} ({k}/{n})  [SE = std/sqrt(N)]")

    if cfg.use_wandb:
        wandb.log({
            f"final_success_rate/{cfg.task_suite_name}": mean,
            f"final_success_std/{cfg.task_suite_name}": std,
            f"total_episodes/{cfg.task_suite_name}": n,
        })
        wandb.save(local_log_filepath)

    log_file.close()
    return mean


if __name__ == "__main__":
    eval_libero()
