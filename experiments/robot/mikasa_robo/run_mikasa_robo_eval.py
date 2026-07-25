"""
run_mikasa_robo_eval.py

Evaluates a trained OpenVLA-OFT policy in a MIKASA-Robo simulation environment.
Supports mu-VLA recurrent memory: when --use_memory is True, a MemoryModule is loaded
from the checkpoint and memory state is maintained across steps within each episode.

Usage:
    cd ~/mu-vla
    uv run python experiments/robot/mikasa_robo/run_mikasa_robo_eval.py \
        --pretrained_checkpoint runs/my_checkpoint \
        --env_id TakeItBack-VLA-v0 \
        --num_trials 50 \
        --use_memory true \
        --num_mem_tokens 4
"""

import os  # noqa: E402  — keep before TF/torch import
import sys

# ── Silence noisy third-party backends BEFORE they are imported ──────────────
# These env vars must be set BEFORE TensorFlow / transformers / sapien imports
# fire (TF reads them at module-load time). Setting them here once means a
# clean console for the whole eval.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")        # suppress TF info/warn/error
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")        # suppress oneDNN banner
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # suppress fork warning
os.environ.setdefault("ABSL_LOGGING_VERBOSITY", "0")

import contextlib
import io
import logging
import warnings

# Suppress chatty loggers and import-time UserWarnings/FutureWarnings from
# torch.cuda/sapien/etc — they pollute the console without telling the user
# anything actionable.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
for _noisy in ("transformers", "tensorflow", "absl", "h5py", "matplotlib", "PIL"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

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

# Heavy project imports below also print at module-load time
# (e.g. `prismatic.vla.constants` prints "Using MIKASA constants:" etc.).
# Swallow that into a buffer so the console stays clean.
_startup_stdout = io.StringIO()
with contextlib.redirect_stdout(_startup_stdout):
    from experiments.robot.mikasa_robo.mikasa_robo_utils import (
        DATE,
        DATE_TIME,
        get_language_instruction,
        get_mikasa_images,
        get_mikasa_proprio,
        make_eval_env,
        render_scene_frame,
        save_composed_video,
    )
    from experiments.robot.openvla_utils import (
        detect_memory_config,
        get_action_head,
        get_memory_module,
        get_processor,
        get_proprio_projector,
    )
    from experiments.robot.robot_utils import (
        get_action,
        get_image_resize_size,
        get_model,
        set_seed_everywhere,
    )
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Single rich console for all human-facing output. Tables + panels +
# progress bars are routed through here. Plain `logger.info` calls are
# *not* echoed to stdout — they go only to the per-run text log file.
#
# `force_terminal=True` is critical: the bash launcher
# `run_mikasa_robo_eval_envs.sh` pipes stdout through `tee` to capture a
# per-env stdout log, which makes rich detect "not a TTY" and silently
# disable the live Progress bar (it falls back to one static line per
# render, which then collapse into blank space). force_terminal=True keeps
# the ANSI cursor-up/clear-line sequences flowing so the user actually
# sees the spinner + progress bar even under `tee`.
PLAIN_PROGRESS = os.environ.get("MU_VLA_PLAIN_PROGRESS", "0") == "1" or not sys.stdout.isatty()
console = Console(highlight=False, force_terminal=not PLAIN_PROGRESS)

# File-only logger. Anything we want to keep for forensic debugging
# (sanity prints, mem-checks, per-episode results) is appended here.
logger = logging.getLogger("mikasa_eval")
logger.setLevel(logging.INFO)
logger.propagate = False  # never echo to root → no stdout duplication


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

    unnorm_key: str = "mikasa_combined"    # Key in dataset_statistics.json

    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # mu-VLA recurrent memory
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
    # MIKASA-Robo environment-specific parameters
    ###########################################################################
    env_id: str = "TakeItBack-VLA-v0"
    num_trials: int = 100                  # MIKASA validation protocol: 100 episodes
    max_episode_steps: Optional[int] = None  # None → auto-detect from gym.spec(env_id)

    ###########################################################################
    # Utils
    ###########################################################################
    run_id_note: Optional[str] = None

    # Per-run output directory. If set, the eval writes everything (text logs,
    # scene-render mp4s, composed mp4s) under this single directory:
    #     <output_dir>/logs/<run_id>.txt
    #     <output_dir>/videos/<env_id>_scene_raw/*.mp4
    #     <output_dir>/videos/episode_*_<env_id>_*.mp4
    # If None, falls back to the legacy split (`local_log_dir` for text,
    # `./rollouts/<DATE_TIME>` for video).
    output_dir: Optional[str] = None
    local_log_dir: str = "./experiments/logs"

    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"

    # MIKASA validation protocol: episodes are seeded with cfg.seed + trial_idx,
    # so a fixed starting seed makes the whole 100-episode sweep reproducible.
    # 4242424242 is the seed all reported numbers were produced with; change it
    # only if you intend to report a different sample.
    seed: int = 4242424242

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters and auto-detect memory hyperparameters."""
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must not be empty!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, (
            "Expecting `center_crop==True` because model was trained with image augmentations!"
        )

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), (
        "Cannot use both 8-bit and 4-bit quantization!"
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
        cfg.attention_mask_mode = cfg.attention_mask_mode or "custom"

    # ── Resolve receding_horizon ──────────────────────────────────────────
    # Memory-enabled checkpoints REQUIRE receding-horizon execution. During
    # training, mu-VLA rolls memory forward by exactly one step per simulator
    # tick (see vla-scripts/finetune.py:1303-1366 — every dataloader yield is
    # a single timestep). Default chunked inference (num_open_loop_steps=8)
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


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    model = get_model(cfg)

    # Load proprio projector — MIKASA uses 7D proprio (xyz + rpy + gripper)
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg, model.llm_dim, proprio_dim=7,
        )

    # Load action head
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load memory module
    memory_module = None
    if cfg.use_memory:
        memory_module = get_memory_module(cfg, model.llm_dim)

    # Load processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        # Verify unnorm_key is present in model norm_stats. Three-level
        # fallback (covers all training-mode → eval-env combinations):
        #   1. cfg.unnorm_key as-is (e.g. 'mikasa_combined' for episodic dataset).
        #   2. env-id-derived key (e.g. 'shell_game_push_vla_v0') — RLDSDataset
        #      checkpoints store one key per training env.
        #   3. synthetic average across all per-env keys — for evaluating an
        #      RLDSDataset checkpoint on envs it WASN'T trained on (out-of-
        #      distribution generalization study). All MIKASA envs share the
        #      same robot/action space, so averaging is a reasonable proxy.
        if cfg.unnorm_key not in model.norm_stats:
            derived_key = _env_id_to_dataset_name(cfg.env_id)
            if derived_key in model.norm_stats:
                console.print(
                    f"[yellow]info:[/yellow] unnorm_key='{cfg.unnorm_key}' not in "
                    f"norm_stats; auto-falling back to env-derived key '{derived_key}'."
                )
                cfg.unnorm_key = derived_key
            elif len(model.norm_stats) > 0:
                # Build synthetic combined stats by averaging over all training envs
                synthetic_key = "_auto_combined_for_eval"
                if synthetic_key not in model.norm_stats:
                    model.norm_stats[synthetic_key] = _build_combined_norm_stats(
                        {k: v for k, v in model.norm_stats.items() if not k.startswith("_")}
                    )
                console.print(
                    f"[cyan]info:[/cyan] env '{cfg.env_id}' was not in the training set "
                    f"(tried '{cfg.unnorm_key}' and derived '{derived_key}'). "
                    f"Denormalising actions with AVG of {len(model.norm_stats) - 1} per-env "
                    f"training stats. All MIKASA-Robo envs share Panda 7-DoF actions in "
                    f"[-1, 1], so this is a valid OOD-generalisation eval."
                )
                cfg.unnorm_key = synthetic_key
            else:
                raise AssertionError(
                    f"Action un-norm key '{cfg.unnorm_key}' not found in model "
                    f"norm_stats and no fallback keys available."
                )

    return model, action_head, proprio_projector, processor, memory_module


def setup_logging(cfg: GenerateConfig):
    """Set up the per-run text log file (and optionally wandb).

    The text log captures the full transcript (sanity prints, mem-check,
    per-episode results, final summary) for forensic debugging. The console
    only shows the rich Panel + Progress + final Panel — see `main()`.

    If `cfg.output_dir` is set, text logs are written to `<output_dir>/logs/`.
    Otherwise we fall back to the legacy `cfg.local_log_dir`.
    """
    run_id = f"EVAL-mikasa_robo-{cfg.env_id}-{cfg.model_family}-{DATE_TIME}"
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
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Append a message to the per-run text log. Console output is handled
    separately by the rich Panel/Progress in `main()`."""
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def _env_id_to_dataset_name(env_id: str) -> str:
    """Convert MIKASA env_id to RLDSDataset key format.

    Example: 'ShellGamePush-VLA-v0' → 'shell_game_push_vla_v0'.
    Used to fall back to per-env unnorm keys when the combined-dataset key
    (e.g. 'mikasa_combined') isn't present in the checkpoint's norm_stats.
    """
    import re
    s = env_id.replace("-", "_")
    # Insert '_' between lowercase/digit and uppercase to split CamelCase.
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", s)
    return s.lower()


def _build_combined_norm_stats(per_env_stats: dict) -> dict:
    """Aggregate per-env norm_stats into a synthetic 'combined' stat.

    Used as a last-resort fallback when validating an RLDSDataset-trained
    checkpoint on an env it wasn't trained on. All MIKASA envs share the
    same Panda 7-DoF action space, so averaging q01/q99/mean/std across
    training envs gives a reasonable denormalization range — comparable
    to what `mikasa_combined` would have been if the same model were
    trained with MIKASARoboVLAEpisodicDataset.

    `mask` fields are AND-aggregated (a dim is valid only if valid in
    every training env).
    """
    if not per_env_stats:
        return {}
    keys = list(per_env_stats.keys())
    template = per_env_stats[keys[0]]
    combined: dict = {}
    for field, val in template.items():
        if isinstance(val, dict):
            combined[field] = {}
            for sub in val:
                try:
                    arrs = [np.asarray(per_env_stats[k][field][sub]) for k in keys]
                except (KeyError, TypeError):
                    combined[field][sub] = val[sub]
                    continue
                if sub == "mask" or arrs[0].dtype == bool:
                    combined[field][sub] = np.logical_and.reduce(arrs).tolist()
                elif np.issubdtype(arrs[0].dtype, np.number):
                    combined[field][sub] = np.mean(arrs, axis=0).tolist()
                else:
                    combined[field][sub] = val[sub]
        else:
            combined[field] = val
    return combined


def prepare_observation(obs):
    """
    Prepare observation for policy input.

    Extracts images and proprio from the MIKASA env observation dict
    and returns them in the format expected by get_vla_action:
        obs["full_image"]:  np.ndarray (H, W, 3) uint8
        obs["wrist_image"]: np.ndarray (H, W, 3) uint8
        obs["state"]:       np.ndarray (7,) float32
    """
    base_image, wrist_image = get_mikasa_images(obs)
    proprio = get_mikasa_proprio(obs)

    observation = {
        "full_image": base_image,
        "wrist_image": wrist_image,
        "state": proprio,
    }
    return observation, base_image


def process_action(action):
    """
    Process model-predicted action before sending to MIKASA env.

    Unlike LIBERO, MIKASA does NOT require:
      - normalize_gripper_action: gripper is already in [-1, 1] after denormalization
      - invert_gripper_action: training data did not invert gripper sign
    """
    return np.clip(action, -1.0, 1.0)


def _env_reset_with_seed(env, seed: int):
    """Reset env using the most permissive seeding interface available.

    ManiSkill GPU envs accept `seed=` directly on `reset()`. If the underlying
    wrapper rejects the kwarg (older API), fall back to `options={"seed": ...}`,
    then to `env.unwrapped.reset(seed=...)`. Returns (obs, info).
    """
    try:
        return env.reset(seed=seed)
    except TypeError:
        pass
    try:
        return env.reset(options={"seed": seed})
    except TypeError:
        pass
    return env.unwrapped.reset(seed=seed)


def run_episode(
    cfg: GenerateConfig,
    env,
    model,
    resize_size,
    episode_timeout: int,
    processor=None,
    action_head=None,
    proprio_projector=None,
    memory_module=None,
    log_file=None,
    episode_seed: Optional[int] = None,
    debug_first_step: bool = False,
    progress: Optional[Progress] = None,
    step_task_id: Optional[int] = None,
):
    """Run a single evaluation episode.

    Returns `(success_once, top_frames, wrist_frames, scene_frames)`:
      - `success_once`: True iff `info["success"]` was True at any point
        during the episode — the metric required by the MIKASA validation
        protocol.
      - `top_frames` / `wrist_frames`: per-step camera images from the obs
        dict.
      - `scene_frames`: per-step scene renders captured via `env.render()`.
        Collecting in-memory replaces the previous `RecordEpisode` wrapper —
        see `mikasa_robo_utils.save_composed_video` docstring for the why.

    If `progress` and `step_task_id` are provided, the per-episode step
    counter is reset to 0/`episode_timeout` at the start and advanced by 1
    on every simulator step — this drives the second progress bar in the
    rich UI.
    """
    if episode_seed is not None:
        obs, info = _env_reset_with_seed(env, episode_seed)
    else:
        obs, info = env.reset()

    # Extract language instruction from info dict
    language_instruction = get_language_instruction(env, info)
    if debug_first_step:
        log_message(
            f"[sanity] language_instruction={language_instruction!r}",
            log_file,
        )

    # Initialize action queue (used only when receding_horizon=False)
    if not cfg.receding_horizon and cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        log_message(
            f"WARNING: num_open_loop_steps ({cfg.num_open_loop_steps}) != "
            f"NUM_ACTIONS_CHUNK ({NUM_ACTIONS_CHUNK})",
            log_file,
        )
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # [mu-VLA] Initialize memory state at episode start
    mem_state = None
    if cfg.use_memory and memory_module is not None:
        mem_state = memory_module.get_initial_state(batch_size=1)
        mem_state = mem_state.to(dtype=torch.bfloat16, device=next(memory_module.parameters()).device)

    top_frames = []
    wrist_frames = []
    scene_frames: list = []
    success_once = False

    # Reset the per-episode step bar (if the rich UI is driving us) so the
    # bar shows 0 / episode_timeout at the start of every episode.
    if progress is not None and step_task_id is not None:
        progress.reset(step_task_id, total=episode_timeout)
        progress.update(step_task_id, completed=0, visible=True)

    # Cache norm stats for debug-first-step sanity print
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
    total_reward = 0.0
    # Track per-step mem-state delta to confirm memory is actually changing.
    mem_state_delta_l1_max = 0.0
    mem_state_delta_l1_min = float("inf")

    try:
        for step in range(episode_timeout):
            # Prepare observation
            observation, base_img = prepare_observation(obs)

            if cfg.receding_horizon:
                # Receding horizon control: re-query model every step, execute only first action
                mem_input = mem_state  # save M^in_t for EMA blend
                result = get_action(
                    cfg,
                    model,
                    observation,
                    language_instruction,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    memory_state=mem_state,
                    use_memory_mask=cfg.use_memory,
                    attention_mask_mode=cfg.attention_mask_mode,
                )
                forward_calls += 1
                if cfg.use_memory and mem_state is not None:
                    actions, new_mem = result
                    # Mirror training-time memory update rule.
                    if cfg.memory_update == "ema":
                        new_mem_state = cfg.ema_alpha * new_mem + (1.0 - cfg.ema_alpha) * mem_input
                    else:
                        new_mem_state = new_mem
                    # Track L1 delta between mem_input (M^in_t) and the new
                    # mem_state (M^in_{t+1}) so we can prove memory really
                    # rolls forward each step.
                    delta = (new_mem_state.float() - mem_input.float()).abs().mean().item()
                    mem_state_delta_l1_max = max(mem_state_delta_l1_max, delta)
                    mem_state_delta_l1_min = min(mem_state_delta_l1_min, delta)
                    mem_state = new_mem_state
                    mem_updates += 1
                else:
                    actions = result
                # Execute ONLY the first action of the predicted chunk
                # (matches training: chunk[t][0] is the action AT time t).
                assert len(actions) == NUM_ACTIONS_CHUNK, (
                    f"Expected chunk of length {NUM_ACTIONS_CHUNK}, got {len(actions)}"
                )
                action = process_action(actions[0])
            else:
                # Open-loop chunking: requery model when action queue is empty
                if len(action_queue) == 0:
                    mem_input = mem_state  # save M^in_t for EMA blend
                    result = get_action(
                        cfg,
                        model,
                        observation,
                        language_instruction,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        memory_state=mem_state,
                        use_memory_mask=cfg.use_memory,
                        attention_mask_mode=cfg.attention_mask_mode,
                    )
                    if cfg.use_memory and mem_state is not None:
                        actions, new_mem = result
                        if cfg.memory_update == "ema":
                            mem_state = cfg.ema_alpha * new_mem + (1.0 - cfg.ema_alpha) * mem_input
                        else:
                            mem_state = new_mem
                    else:
                        actions = result
                    action_queue.extend(actions)

                action = process_action(action_queue.popleft())

            # Sanity-print on the very first step of the very first debug episode.
            # IMPORTANT: capture proprio BEFORE get_action() runs above —
            # get_vla_action() mutates `observation["state"]` in place
            # (normalizes proprio). We saved a snapshot earlier in this loop
            # iteration via `prepare_observation`'s return.
            if not debug_logged:
                debug_logged = True
                # Reconstruct raw proprio from `obs` (env output, untouched).
                proprio_raw = get_mikasa_proprio(obs)
                proprio_arr = np.asarray(proprio_raw)
                # apply same normalization that get_vla_action uses to confirm bounds
                proprio_norm_str = "n/a"
                proprio_norm = None
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
                    f"[sanity] proprio_raw shape={proprio_arr.shape} "
                    f"min={float(np.min(proprio_arr)):.3f} "
                    f"max={float(np.max(proprio_arr)):.3f} | "
                    f"proprio_norm: {proprio_norm_str}",
                    log_file,
                )
                log_message(
                    f"[sanity] action[0] shape={np.asarray(action).shape} "
                    f"min={float(np.min(action)):.3f} "
                    f"max={float(np.max(action)):.3f} "
                    f"gripper={float(action[-1]):.3f} (expected in [-1, 1])",
                    log_file,
                )
                if mem_state is not None:
                    log_message(
                        f"[sanity] mem_state shape={tuple(mem_state.shape)} "
                        f"dtype={mem_state.dtype} device={mem_state.device}",
                        log_file,
                    )
                # Sanity-describe the obs the policy *just* acted on. The
                # composed-video frame buffers are appended post-step now,
                # so we re-derive base/wrist directly from the current
                # (pre-step) `obs` here rather than reading buffer slots
                # that haven't been filled yet for this iteration.
                _sanity_base, _sanity_wrist = get_mikasa_images(obs)
                _b = np.asarray(_sanity_base)
                _w = np.asarray(_sanity_wrist)
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

            # Convert to torch tensor for ManiSkill GPU env
            action_tensor = torch.from_numpy(action).float().unsqueeze(0)  # (1, 7)
            action_tensor = action_tensor.to(
                device=obs["rgb"].device if torch.is_tensor(obs["rgb"]) else "cuda"
            )

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_tensor)
            sim_steps += 1
            r = reward.cpu().numpy() if torch.is_tensor(reward) else np.asarray(reward)
            total_reward += float(r.sum())
            if progress is not None and step_task_id is not None:
                progress.update(step_task_id, advance=1)

            # Capture the post-step state for the composed video.
            # Capturing BEFORE the step (the previous behaviour) recorded
            # the post-reset state at frame 0, where Sapien's visual mesh
            # transforms had not yet synced — the cups looked sunken into
            # the table for one frame and then snapped to the correct pose
            # at frame 1. Post-step capture reflects what physics actually
            # produces, with no transient sync artefact.
            base_image, wrist_image = get_mikasa_images(obs)
            top_frames.append(base_image)
            wrist_frames.append(wrist_image)
            scene_frames.append(render_scene_frame(env))

            # Check success
            if "success" in info:
                s = info["success"]
                if torch.is_tensor(s):
                    s = s.cpu().numpy()
                s = np.asarray(s)
                if s.any():
                    success_once = True
                    break  # early exit on first success

            # Check if episode is done
            term = terminated.cpu().numpy() if torch.is_tensor(terminated) else np.asarray(terminated)
            trunc = truncated.cpu().numpy() if torch.is_tensor(truncated) else np.asarray(truncated)
            if np.logical_or(term, trunc).any():
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
        # Sanity that memory actually changed every step (not frozen).
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

    episode_stats = {
        "steps": sim_steps,
        "total_reward": total_reward,
        "mean_reward": total_reward / sim_steps if sim_steps > 0 else 0.0,
    }
    return success_once, top_frames, wrist_frames, scene_frames, episode_stats


def _build_header_panel(cfg: GenerateConfig, episode_timeout: Optional[int] = None) -> Panel:
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

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column()
    table.add_row("Checkpoint", os.path.basename(str(cfg.pretrained_checkpoint).rstrip("/")))
    table.add_row("Memory", mem_line)
    table.add_row("Inference", inference_line)
    table.add_row(
        "Validation",
        f"{cfg.num_trials} episodes · seeds [{cfg.seed} .. {cfg.seed + cfg.num_trials - 1}]",
    )
    if episode_timeout is not None:
        table.add_row("Episode max", f"{episode_timeout} sim-steps")
    table.add_row("Unnorm key", cfg.unnorm_key)
    table.add_row(
        "Metric",
        f"[dim]success rate = mean(success_once) ± std over {cfg.num_trials} episodes; "
        "success_once = 1 iff info['success'] was True at any step[/]",
    )
    return Panel(
        table,
        title=f"[bold]mu-VLA eval · {cfg.env_id}[/]",
        title_align="left",
        border_style="cyan",
    )


def _build_results_panel(env_id: str, n: int, k: int, mean: float, std: float, se: float) -> Panel:
    """Post-run results panel — single line per metric, no banners."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column()
    table.add_row("Episodes", f"{n}")
    table.add_row("Successes", f"{k}  [dim](episodes where info['success'] was True at any step)[/]")
    table.add_row("Success rate", f"[bold]{mean:.4f} ± {se:.4f}[/]  ({k}/{n})  [dim]= mean(success_once) ± SE  (SE = std/√N)[/]")
    table.add_row("Std", f"{std:.4f}  [dim]= sample std (Bernoulli variance, ddof=1)[/]")
    return Panel(
        table,
        title=f"[bold green]{env_id} · final[/]",
        title_align="left",
        border_style="green",
    )


@draccus.wrap()
def main(cfg: GenerateConfig) -> None:
    # Validate configuration (may emit yellow CLI-warning lines via console)
    validate_config(cfg)

    # Per-episode env seeding follows the MIKASA validation protocol
    # (cfg.seed = 4242424242 by default; episode i uses cfg.seed + i).
    # Seed torch/numpy with the same starting value for reproducibility of
    # any model-side stochasticity (sampling, dropout if leftover, etc.).
    set_seed_everywhere(cfg.seed)

    # Set up file logging (the per-run text log captures the full transcript).
    log_file, local_log_filepath, run_id = setup_logging(cfg)
    log_message(f"Run ID: {run_id}", log_file)
    log_message(f"Environment: {cfg.env_id}", log_file)
    log_message(f"Checkpoint: {cfg.pretrained_checkpoint}", log_file)
    log_message(f"Num trials: {cfg.num_trials}", log_file)
    log_message(
        f"Memory: use={cfg.use_memory} num_mem_tokens={cfg.num_mem_tokens} "
        f"update={cfg.memory_update} ema_alpha={cfg.ema_alpha}",
        log_file,
    )
    log_message(
        f"Inference mode: {'receding-horizon' if cfg.receding_horizon else f'chunked ({cfg.num_open_loop_steps})'}",
        log_file,
    )

    # ── Console: opening config panel ────────────────────────────────────
    console.print()
    console.print(_build_header_panel(cfg))

    # Resolve output paths up-front so we can show them to the user right
    # after the header panel. The per-run text log lives in
    # <output_dir>/logs/, composed videos in <output_dir>/videos/.
    if cfg.output_dir is not None:
        videos_root = os.path.join(cfg.output_dir, "videos")
    else:
        videos_root = f"./rollouts/{DATE_TIME}"
    composed_video_dir = videos_root
    log_dir_for_print = os.path.dirname(local_log_filepath)
    console.print(
        f"  [bold cyan]Logs:[/]   [dim]{log_dir_for_print}[/]  "
        f"[dim](this run: {os.path.basename(local_log_filepath)})[/]"
    )
    console.print(f"  [bold cyan]Videos:[/] [dim]{composed_video_dir}/episode_<N>_{cfg.env_id}_<success|failure>.mp4[/]")
    console.print()

    # Load model + heads + memory module under a status spinner so the user
    # gets visible feedback during the (~10s) checkpoint-shard load. Shared
    # utility functions (`get_model`, `from_pretrained`) emit progress prints
    # and a HuggingFace shard-loading tqdm bar; we redirect both to the
    # per-run log file so the console stays clean.
    if PLAIN_PROGRESS:
        console.print("Loading model + heads...")
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            model, action_head, proprio_projector, processor, memory_module = initialize_model(cfg)
    else:
        with console.status("[cyan]Loading model + heads…[/]", spinner="dots"):
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                model, action_head, proprio_projector, processor, memory_module = initialize_model(cfg)

    resize_size = get_image_resize_size(cfg)
    log_message(f"Image resize size: {resize_size}", log_file)

    log_message(f"Creating environment: {cfg.env_id}", log_file)
    if PLAIN_PROGRESS:
        console.print(f"Creating environment {cfg.env_id}...")
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            env, episode_timeout = make_eval_env(
                env_id=cfg.env_id,
                num_envs=1,
                max_episode_steps=cfg.max_episode_steps,
            )
    else:
        with console.status(f"[cyan]Creating environment {cfg.env_id}…[/]", spinner="dots"):
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                env, episode_timeout = make_eval_env(
                    env_id=cfg.env_id,
                    num_envs=1,
                    max_episode_steps=cfg.max_episode_steps,
                )
    log_message(f"Episode timeout: {episode_timeout}", log_file)
    log_message(
        f"Validation protocol: {cfg.num_trials} episodes, env seeds "
        f"[{cfg.seed}, {cfg.seed + cfg.num_trials - 1}]",
        log_file,
    )

    # ── Console: rich.Progress with two task rows (episodes + steps) ─────
    successes: list = []  # one bool/int per episode (success_once)
    progress = None
    episode_task_id = None
    step_task_id = None
    progress_context = contextlib.nullcontext()
    if not PLAIN_PROGRESS:
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
        episode_task_id = progress.add_task(
            "Episodes",
            total=cfg.num_trials,
            running="0/0 = 0.00%",
        )
        # Second row: per-episode step counter. `running` is unused on this row
        # (the column TextColumn requires the field on every task to avoid a
        # KeyError, so we pass an empty string). We reset it at the start of
        # every episode inside `run_episode`.
        step_task_id = progress.add_task(
            "Step",
            total=episode_timeout,
            running="",
        )
        progress_context = progress

    with progress_context:
        for trial_idx in range(cfg.num_trials):
            episode_seed = cfg.seed + trial_idx
            log_message(
                f"\nStarting episode {trial_idx + 1}/{cfg.num_trials} (seed={episode_seed})...",
                log_file,
            )
            if PLAIN_PROGRESS:
                console.print(
                    f"Starting episode {trial_idx + 1}/{cfg.num_trials} "
                    f"(seed={episode_seed})..."
                )

            success_once, top_frames, wrist_frames, scene_frames, ep_stats = run_episode(
                cfg,
                env,
                model,
                resize_size,
                episode_timeout=episode_timeout,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                memory_module=memory_module,
                log_file=log_file,
                episode_seed=episode_seed,
                debug_first_step=(trial_idx == 0),
                progress=progress,
                step_task_id=step_task_id,
            )

            successes.append(int(bool(success_once)))
            total_episodes = len(successes)
            total_successes = sum(successes)

            # Save composed video for this episode unconditionally — every
            # episode gets a [scene | top / wrist] mp4. `scene_frames` was
            # collected in-memory by `run_episode` (one frame per sim step),
            # so we no longer depend on RecordEpisode's async-flush behaviour
            # which silently dropped the LAST episode and produced mp4v files
            # that VS Code can't decode.
            success_str = "success" if success_once else "failure"
            composed_path = os.path.join(
                composed_video_dir,
                f"episode_{total_episodes}_{cfg.env_id}_{success_str}.mp4",
            )
            try:
                save_composed_video(
                    scene_frames=scene_frames,
                    top_frames=top_frames,
                    wrist_frames=wrist_frames,
                    output_path=composed_path,
                    fps=30.0,
                    log_file=log_file,
                )
            except Exception as e:
                log_message(
                    f"WARNING: failed to save composed video for episode "
                    f"{total_episodes}: {e}",
                    log_file,
                )

            running_mean = total_successes / total_episodes
            log_message(
                f"Episode {total_episodes}/{cfg.num_trials} | "
                f"success_once: {bool(success_once)} | "
                f"steps: {ep_stats['steps']} | "
                f"total_reward: {ep_stats['total_reward']:.4f} | "
                f"mean_reward: {ep_stats['mean_reward']:.4f} | "
                f"Running mean SR: {total_successes}/{total_episodes} = {running_mean:.2%}",
                log_file,
            )
            if progress is not None and episode_task_id is not None:
                progress.update(
                    episode_task_id,
                    advance=1,
                    running=f"SR {total_successes}/{total_episodes} = {running_mean:.2%}",
                )
            elif PLAIN_PROGRESS:
                console.print(
                    f"Episode {total_episodes}/{cfg.num_trials} | "
                    f"success_once={bool(success_once)} | "
                    f"steps={ep_stats['steps']} | "
                    f"running_mean={running_mean:.2%}"
                )

            if cfg.use_wandb:
                wandb.log({
                    "episode": total_episodes,
                    "success": int(success_once),
                    "running_mean": running_mean,
                })

    # Final results: mean ± SE (standard error = std/sqrt(N)) of `success_once` over N episodes.
    # SE quantifies precision of the mean estimate; std (Bernoulli variance) is shown separately.
    s_arr = np.asarray(successes, dtype=np.float64)
    n = int(s_arr.size)
    k = int(s_arr.sum())
    mean = float(s_arr.mean()) if n > 0 else 0.0
    std = float(s_arr.std(ddof=1)) if n > 1 else 0.0
    se = float(std / np.sqrt(n)) if n > 0 else 0.0

    # Per-env summary line — kept verbatim in the file log so the bash
    # launcher's `grep -E "Success rate:" ... | tail -n1` keeps working.
    log_message(
        f"\n{'=' * 60}\n"
        f"Final Results for {cfg.env_id}\n"
        f"  Total episodes:   {n}\n"
        f"  Total successes:  {k}\n"
        f"  Success rate:     {mean:.4f} ± {se:.4f} ({k}/{n})  [SE = std/sqrt(N)]\n"
        f"  Std:              {std:.4f}\n"
        f"{'=' * 60}",
        log_file,
    )

    # ── Console: closing summary panel ──────────────────────────────────
    console.print()
    console.print(_build_results_panel(cfg.env_id, n, k, mean, std, se))
    console.print()
    # Plain echo so that bash launchers parsing stdout (`tee` + grep) still
    # see a `Success rate: ...` line. Indent matches the file-log block.
    console.print(f"  Success rate:     {mean:.4f} ± {se:.4f} ({k}/{n})  [SE = std/sqrt(N)]")

    if cfg.use_wandb:
        wandb.log({
            f"final_success_rate/{cfg.env_id}": mean,
            f"final_success_std/{cfg.env_id}": std,
            f"total_episodes/{cfg.env_id}": n,
        })

    # Cleanup
    env.close()
    log_file.close()


if __name__ == "__main__":
    main()
