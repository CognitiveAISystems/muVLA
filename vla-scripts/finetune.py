"""
finetune.py

Fine-tunes OpenVLA via LoRA.
"""

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.memory import MemoryModule
from prismatic.models.memory_diagnostics import MemoryDiagnostics, TokenLayout
from prismatic.models.projectors import (
    NoisyActionProjector,
    ProprioProjector,
)
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "libero_spatial_no_noops"    # RLDS dataset name. In episodic mode (--use_*_episodic) this is
                                                     #   only a label for the run directory; in plain RLDS mode it must
                                                     #   name a real dataset under --data_root_dir.
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_schedule: str = "multistep"                   # LR schedule: "multistep" (original) or "cosine"
    lr_warmup_steps: int = 0                         # Number of warmup steps (multistep: manual 10%→100%; cosine: integrated)
    num_steps_before_decay: int = 100_000            # (multistep only) Number of steps before LR decays by 10x
    lr_min_ratio: float = 0.0                        # (cosine only) Min LR as fraction of peak LR (0.0 = decay to zero)
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200_000                         # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

    # MIKASA-Robo episodic dataloader
    use_mikasa_episodic: bool = False                # If True, uses POMDP-aware episodic dataloader for MIKASA-Robo
    mikasa_env_names: str = "shell_game_push_vla_v0,intercept_medium_vla_v0,remember_color_5_vla_v0,take_it_back_vla_v0,remember_shape_and_color_3x3_vla_v0"
    # Directory names under --data_root_dir, matching the layout of the published
    # RLDS dataset. These are NOT the Gym environment ids used at evaluation time.

    # LIBERO episodic dataloader (multi-task)
    use_libero_episodic: bool = False                # If True, uses POMDP-aware episodic dataloader for LIBERO
    libero_suite_names: str = "libero_spatial_no_noops,libero_object_no_noops,libero_goal_no_noops,libero_10_no_noops"

    # mu-VLA recurrent memory
    use_memory: bool = False                         # If True, enables recurrent memory tokens (mu-VLA)
    num_mem_tokens: int = 4                          # Number of memory tokens injected into the sequence
    memory_update: str = "tbptt"                     # Cross-step memory update rule: "tbptt" or "ema".
                                                     #   "tbptt": keep graph within tbptt_length window, detach at boundary.
                                                     #   "ema":   no cross-step gradient flow; M^in_{t+1} = alpha*M^out_t + (1-alpha)*M^in_t.
    ema_alpha: float = 0.1                           # Mixing coefficient for memory_update="ema". Higher = faster update.
                                                     #   alpha=1 recovers TBPTT-length-1; alpha=0 freezes memory.
    tbptt_length: int = 1                            # TBPTT truncation length: gradients flow through this many steps.
                                                     #   Ignored when memory_update="ema".
    attention_mask_mode: str = "custom"              # Attention mask used by the LLM when memory is active:
                                                     #   "custom" — mu-VLA context/action split (MEM cannot see ACTION).
                                                     #   "full"   — standard OpenVLA-OFT bidirectional (last-row trick).
                                                     #   Ignored when use_memory=False (full mask is always used).
                                                     #   Saved to memory_meta.json so eval auto-detects the right mode.
    use_gradient_checkpointing: bool = False          # If True, enables gradient checkpointing for the LLM backbone
                                                     #   Trades ~30-40% slower backward for much lower VRAM usage.
                                                     #   Recommended when using TBPTT with length > 1.
    memory_log_freq: int = 0                          # mu-VLA diagnostics: lightweight metrics every N grad steps
                                                     #   (memory norms, grad norms). 0 = disabled. Recommended: 10.
                                                     #   Only active when use_memory=True.
    memory_expensive_log_freq: int = 0                # mu-VLA diagnostics: expensive metrics every N grad steps
                                                     #   (attention heatmap, episode plots). 0 = disabled. Recommended: 200.
                                                     #   Attention heatmaps work both with and without use_memory.

    # LoRA
    use_lora: bool = True                            # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = True          # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!

    # Logging
    use_wandb: bool = False                          # If True, log to WandB (needs `wandb login`)
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # fmt: on


def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.vla_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    Loads a checkpoint for a given module.

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wrap a module with DistributedDataParallel.

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        None.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device_id (str): Device ID.
        module_args (dict): Args for initializing the module.
        to_bf16 (bool): Whether to convert to torch.bfloat16 data type.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)


def run_forward_pass(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_diffusion,
    use_proprio,
    use_film,
    num_patches,
    compute_diffusion_l1=False,
    num_diffusion_steps_train=None,
    memory_state=None,
    use_memory_mask=False,
    attention_mask_mode="custom",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        use_l1_regression (bool): Whether to use L1 regression.
        use_diffusion (bool): Whether to use diffusion.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.
        num_patches (int): Number of vision patches.
        compute_diffusion_l1 (bool): Whether to sample actions and compute L1 loss for diffusion (do this once every
                                    diffusion_sample_freq steps during training; do it every batch for validation)
        num_diffusion_steps_train (int): Number of diffusion steps for training (only used for diffusion).

    Returns:
        tuple: (loss, metrics_dict)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
    """
    metrics = {}

    # Get ground-truth action labels
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)

    # [Only for diffusion] Sample noisy actions used as input for noise predictor network
    if use_diffusion:
        noisy_dict = action_head.module.sample_noisy_actions(ground_truth_actions)
        noise, noisy_actions, diffusion_timestep_embeddings = (
            noisy_dict["noise"],
            noisy_dict["noisy_actions"],
            noisy_dict["diffusion_timestep_embeddings"],
        )
    else:
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    # VLA forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=noisy_actions if use_diffusion else None,
            noisy_action_projector=noisy_action_projector if use_diffusion else None,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
            use_film=use_film,
            memory_state=memory_state,
            use_memory_mask=use_memory_mask,
            attention_mask_mode=attention_mask_mode,
        )

    # Get action masks needed for logging
    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    # Compute metrics for discrete action representation (next-token prediction)
    if not (use_l1_regression or use_diffusion):
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
            }
        )
    # Compute metrics for continuous action representations (L1 regression | diffusion)
    else:
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        # Get hidden states for text portion of prompt+response (after the vision patches)
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        # Get hidden states for action portion of response
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )  # (B, act_chunk_len, D)

        if use_l1_regression:
            # Predict action
            predicted_actions = action_head.module.predict_action(actions_hidden_states)
            # Get full L1 loss
            loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)

        if use_diffusion:
            # Predict noise
            noise_pred = action_head.module.predict_noise(actions_hidden_states)
            # Get diffusion noise prediction MSE loss
            noise_pred = noise_pred.reshape(noise.shape)
            loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")

            # Only sample actions and compute L1 losses if specified
            if compute_diffusion_l1:
                with torch.no_grad():
                    predicted_actions = run_diffusion_sampling(
                        vla=vla,
                        action_head=action_head,
                        noisy_action_projector=noisy_action_projector,
                        proprio_projector=proprio_projector,
                        batch=batch,
                        batch_size=batch_size,
                        num_patches=num_patches,
                        actions_shape=ground_truth_actions.shape,
                        device_id=device_id,
                        current_action_mask=current_action_mask,
                        next_actions_mask=next_actions_mask,
                        use_proprio=use_proprio,
                        use_film=use_film,
                    )

        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
            }
        )

        # Get detailed L1 losses for logging
        should_log_l1_loss = not use_diffusion or (use_diffusion and compute_diffusion_l1)
        if should_log_l1_loss:
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            metrics.update(
                {
                    "curr_action_l1_loss": curr_action_l1_loss.item(),
                    "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )

    # Extract new memory state if available
    new_memory_state = getattr(output, "new_memory_state", None)

    # Return loss, metrics, and optionally new memory state
    return loss, metrics, new_memory_state


def run_diffusion_sampling(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    batch_size,
    num_patches,
    actions_shape,
    device_id,
    current_action_mask,
    next_actions_mask,
    use_proprio,
    use_film,
) -> torch.Tensor:
    """
    Run diffusion sampling (reverse diffusion) to generate actions.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        batch_size (int): Batch size.
        num_patches (int): Number of vision patches.
        actions_shape (tuple): Shape of ground-truth actions.
        device_id (str): Device ID.
        current_action_mask (torch.Tensor): Mask for current action.
        next_actions_mask (torch.Tensor): Mask for next actions.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.

    Returns:
        torch.Tensor: Predicted actions.
    """
    # Sample random noisy action, used as the starting point for reverse diffusion
    noise = torch.randn(
        size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM),
        device=device_id,
        dtype=torch.bfloat16,
    )  # (B, chunk_len, action_dim)

    # Set diffusion timestep values
    action_head.module.noise_scheduler.set_timesteps(action_head.module.num_diffusion_steps_train)

    # Reverse diffusion: Iteratively denoise to generate action, conditioned on observation
    curr_noisy_actions = noise
    for t in action_head.module.noise_scheduler.timesteps:
        # Get diffusion model's noise prediction (conditioned on VLA latent embedding, current noisy action embedding,
        # and diffusion timestep embedding)
        timesteps = torch.Tensor([t]).repeat(batch_size).to(device_id)
        diffusion_timestep_embeddings = (
            action_head.module.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
        )  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                labels=batch["labels"],
                output_hidden_states=True,
                proprio=batch["proprio"] if use_proprio else None,
                proprio_projector=proprio_projector if use_proprio else None,
                noisy_actions=curr_noisy_actions,
                noisy_action_projector=noisy_action_projector,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings,
                use_film=use_film,
            )
            # Get last layer hidden states
            last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
            # Get hidden states for text portion of prompt+response (after the vision patches)
            text_hidden_states = last_hidden_states[:, num_patches:-1]
            # Get hidden states for action portion of response
            actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(
                batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1
            )  # (B, act_chunk_len, D)
            actions_hidden_states = actions_hidden_states.to(torch.bfloat16)
            # Predict noise
            noise_pred = action_head.module.predict_noise(actions_hidden_states)

        # Compute the action at the previous diffusion timestep: x_t -> x_{t-1}
        curr_noisy_actions = action_head.module.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    return curr_noisy_actions.reshape(actions_shape)


def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics


def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity (str): W&B entity instance

    Returns:
        None.
    """
    log_dict = {}
    for name, value in metrics.items():
        # Map loss_value to Loss for better readability in W&B
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        # Keep other metrics as is
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)


def echo_metrics_to_stdout(metrics, step: int, learning_rate: float) -> None:
    """Print, on one line, the metrics that were just sent to W&B.

    Every training metric in this script is reported through `wandb.log` and nowhere
    else. Training without a W&B account is a documented, supported path
    (`WANDB_MODE=disabled`), and on that path the run would otherwise be unobservable:
    the only thing on screen is a progress bar, with no loss, no L1 and no memory
    metrics. `tqdm.write` is used instead of `print` so the line is emitted above the
    active progress bar rather than through it.

    Args:
        metrics (dict): The same smoothened metrics passed to `log_metrics_to_wandb`.
        step (int): Training step.
        learning_rate (float): Current learning rate, read from the optimizer so it
            reflects both the scheduler and any manual warmup.

    Returns:
        None.
    """
    parts = [f"step {step}"]
    for name, value in sorted(metrics.items()):
        label = "loss" if name == "loss_value" else name
        parts.append(f"{label} {value:.4f}" if isinstance(value, (int, float)) else f"{label} {value}")
    parts.append(f"lr {learning_rate:.2e}")
    tqdm.tqdm.write("[train] " + " | ".join(parts))


def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    action_head,
    train_dataset,
    distributed_state,
    memory_module=None,
    optimizer=None,
    scheduler=None,
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, dataset statistics,
    and optimizer/scheduler state for correct resume.

    Args:
        cfg (FinetuneConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        action_head (nn.Module): Action head module.
        train_dataset (RLDSDataset): Training dataset.
        distributed_state (PartialState): Distributed training state.
        memory_module (nn.Module): Memory module (only used for mu-VLA).
        optimizer (torch.optim.Optimizer): Optimizer (saved for resume).
        scheduler (torch.optim.lr_scheduler._LRScheduler): LR scheduler (saved for resume).

    Returns:
        None.
    """
    # Determine checkpoint paths and naming
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    # Create directories and save dataset statistics (main process only)
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    # Wait for directories to be created
    dist.barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)

        # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}"
            )

        if (cfg.use_l1_regression or cfg.use_diffusion) and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

        if cfg.use_film:
            # To be safe, just save the entire vision backbone (not just FiLM components)
            torch.save(
                vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
            )

        if cfg.use_memory and memory_module is not None:
            torch.save(memory_module.state_dict(), checkpoint_dir / f"memory_module--{checkpoint_name_suffix}")
            # Persist memory hyperparameters for downstream eval auto-detection
            # (see experiments/robot/openvla_utils.py:detect_memory_config).
            import json as _json
            memory_meta = {
                "num_mem_tokens": int(cfg.num_mem_tokens),
                "memory_update": str(cfg.memory_update),
                "ema_alpha": float(cfg.ema_alpha),
                "tbptt_length": int(cfg.tbptt_length),
                "attention_mask_mode": str(cfg.attention_mask_mode),
            }
            with open(checkpoint_dir / "memory_meta.json", "w") as f:
                _json.dump(memory_meta, f, indent=2)

        # Save optimizer and scheduler state for correct resume
        if optimizer is not None:
            torch.save(optimizer.state_dict(), checkpoint_dir / f"optimizer--{checkpoint_name_suffix}")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), checkpoint_dir / f"scheduler--{checkpoint_name_suffix}")

    # Wait for model components to be saved
    dist.barrier()

    # Merge LoRA weights into base model and save resulting model checkpoint
    # Note: Can be very slow on some devices; if so, we recommend merging offline
    if cfg.use_lora and cfg.merge_lora_during_training:
        if distributed_state.is_main_process:
            # Load base model on CPU to avoid GPU OOM (training model is still in GPU memory)
            base_vla = AutoModelForVision2Seq.from_pretrained(
                cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                trust_remote_code=True, device_map="cpu"
            )
            merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir, device_map="cpu")
            merged_vla = merged_vla.merge_and_unload()
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")
            del merged_vla, base_vla

        # Wait for merged model to be saved
        dist.barrier()


def run_validation(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    val_dataloader,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,
    log_step,
    distributed_state,
    val_time_limit,
) -> None:
    """
    Compute validation set metrics for logging.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        val_dataloader (DataLoader): Validation data loader.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        cfg (FinetuneConfig): Training configuration.
        num_patches (int): Number of vision patches.
        log_step (int): Current logging step.
        distributed_state (PartialState): Distributed training state.
        val_time_limit (int): Time limit for computing validation metrics.

    Returns:
        None.
    """
    val_start_time = time.time()
    vla.eval()
    val_batches_count = 0

    # List to store validation metrics
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            # Always compute L1 loss for validation, even for diffusion
            _, metrics, _ = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
                compute_diffusion_l1=True,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
            )

            # Add the loss value to the metrics
            metrics["loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1

            # Cut testing on validation set short if it exceeds time limit
            if time.time() - val_start_time > val_time_limit:
                break

    # Compute average validation metrics
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    # Add batch count to metrics
    avg_val_metrics["val_batches_count"] = val_batches_count

    # Log validation metrics to W&B
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA.

    Allows toggling different action representations (discrete vs. continuous), different learning objectives
    (next-token prediction vs. L1 regression vs. diffusion), FiLM. Also allows for additional model inputs,
    such as additional camera images and robot proprioceptive state. Assumes parallel action generation with
    action chunking.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        None.
    """
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Initialize wandb logging. A disabled run is still a run: `wandb.log` remains a
    # no-op call rather than an error, so every logging site below is unconditional.
    # Without this branch, a machine with no API key dies here instead of training.
    if distributed_state.is_main_process:
        if cfg.use_wandb:
            wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")
        else:
            wandb.init(mode="disabled")

    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes
    # In both cases, we want to check whether any changes have been made to
    # the `modeling_prismatic.py` file in this codebase; if so, we will copy
    # the file to the downloaded or locally stored checkpoint directory so
    # that the user's changes to the VLA class logic go into effect
    if model_is_on_hf_hub(cfg.vla_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        # Overwrite VLA path
        cfg.vla_path = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Update config.json and sync model files
    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)

    # Wait for model files to be synced
    dist.barrier()

    # Load processor and VLA
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)

    # Set number of images in VLA input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # LoRA setup
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        # Wrap vision backbone with FiLM wrapper
        # Important: For this, must specify `vla.model.vision_backbone` instead of just `vla.vision_backbone`, since the
        # latter would cause the new wrapped backbone to be saved as a new attribute of `vla` instead of overwriting the
        # original one (due to the LoRA wrapper)
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # [mu-VLA] Enable gradient checkpointing for the LLM backbone to reduce VRAM usage.
    # This is especially important for TBPTT with K > 1, where K computation graphs are held simultaneously.
    # use_reentrant=False is required for TBPTT: reentrant checkpointing does not support
    # retaining multiple computation graphs simultaneously (raises RuntimeError).
    if cfg.use_gradient_checkpointing:
        vla.language_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("[mu-VLA] Gradient checkpointing enabled for LLM backbone")

    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # If applicable, instantiate proprio projector
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
        )

    # If applicable, instantiate continuous action head for L1 regression
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead,
            "action_head",
            cfg,
            device_id,
            {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
            to_bf16=True,
        )

    # If applicable, instantiate diffusion action head and noisy action projector
    if cfg.use_diffusion:
        action_head = init_module(
            DiffusionActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": vla.module.llm_dim,
                "action_dim": ACTION_DIM,
                "num_diffusion_steps_train": cfg.num_diffusion_steps_train,
            },
            to_bf16=True,
        )
        noisy_action_projector = init_module(
            NoisyActionProjector, "noisy_action_projector", cfg, device_id, {"llm_dim": vla.module.llm_dim}
        )

    # Get number of vision patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings
    if cfg.use_proprio:
        NUM_PATCHES += 1
    # For diffusion, a single diffusion timestep embedding is appended to the end of the vision patch embeddings
    if cfg.use_diffusion:
        NUM_PATCHES += 1

    # [mu-VLA] Instantiate memory module
    memory_module = None
    if cfg.use_memory:
        assert cfg.use_mikasa_episodic or cfg.use_libero_episodic, (
            "use_memory=True requires use_mikasa_episodic=True or use_libero_episodic=True "
            "(needs is_first flags)"
        )
        memory_module = MemoryModule(
            num_mem_tokens=cfg.num_mem_tokens,
            hidden_dim=vla.module.llm_dim,
        )
        if cfg.resume:
            state_dict = load_checkpoint("memory_module", cfg.vla_path, cfg.resume_step)
            memory_module.load_state_dict(state_dict)
        memory_module = memory_module.to(torch.bfloat16)
        memory_module = memory_module.to(device_id)
        NUM_PATCHES += cfg.num_mem_tokens
        if cfg.memory_update == "ema":
            print(f"[mu-VLA] MemoryModule: {cfg.num_mem_tokens} tokens, update=EMA, alpha={cfg.ema_alpha}")
        else:
            print(f"[mu-VLA] MemoryModule: {cfg.num_mem_tokens} tokens, update=TBPTT, length={cfg.tbptt_length}")

    # [mu-VLA] Initialize memory diagnostics
    memory_diagnostics = None
    if distributed_state.is_main_process:
        # Diagnostics are available in two modes:
        #   - With memory (use_memory=True): lightweight metrics (norms, grads) + expensive (attention heatmaps, episode plots)
        #   - Without memory: only expensive metrics (attention heatmaps) are meaningful
        has_memory = cfg.use_memory and memory_module is not None
        want_lightweight = has_memory and cfg.memory_log_freq > 0
        want_expensive = cfg.memory_expensive_log_freq > 0
        if want_lightweight or want_expensive:
            num_vision_patches = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
            token_layout = TokenLayout(
                num_bos=1,
                num_vision=num_vision_patches,
                num_proprio=1 if cfg.use_proprio else 0,
                num_diffusion=1 if cfg.use_diffusion else 0,
                num_mem=cfg.num_mem_tokens if has_memory else 0,
                num_action_and_stop=ACTION_DIM * NUM_ACTIONS_CHUNK + 1,
            )
            memory_diagnostics = MemoryDiagnostics(
                log_freq=cfg.memory_log_freq if has_memory else 0,
                expensive_log_freq=cfg.memory_expensive_log_freq,
                token_layout=token_layout,
                wandb_module=wandb,
            )
            print(f"[mu-VLA] Diagnostics enabled: lightweight every {cfg.memory_log_freq if has_memory else 0}, "
                  f"expensive every {cfg.memory_expensive_log_freq} grad steps"
                  f"{' (attention only, no memory)' if not has_memory else ''}")

    # Instantiate optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression or cfg.use_diffusion:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    if cfg.use_diffusion:
        trainable_params += [param for param in noisy_action_projector.parameters() if param.requires_grad]
    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    if cfg.use_memory and memory_module is not None:
        trainable_params += [param for param in memory_module.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Record original learning rate
    original_lr = optimizer.param_groups[0]["lr"]

    # Create learning rate scheduler
    if cfg.lr_schedule == "cosine":
        import math

        warmup_steps = cfg.lr_warmup_steps
        total_steps = cfg.max_steps
        min_ratio = cfg.lr_min_ratio

        def _cosine_with_warmup(step: int) -> float:
            """Returns LR multiplier in [min_ratio, 1.0]."""
            if warmup_steps > 0 and step < warmup_steps:
                # Linear warmup: 10% → 100% (matches legacy warmup range)
                return 0.1 + 0.9 * (step / warmup_steps)
            # Cosine decay from 1.0 → min_ratio
            decay_steps = max(total_steps - warmup_steps, 1)
            progress = min((step - warmup_steps) / decay_steps, 1.0)
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=_cosine_with_warmup)
    elif cfg.lr_schedule == "multistep":
        scheduler = MultiStepLR(
            optimizer,
            milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
            gamma=0.1,  # Multiplicative factor of learning rate decay
        )
    else:
        raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule!r}. Use 'multistep' or 'cosine'.")

    # [Resume] Restore optimizer and scheduler state so that momentum/variance and LR
    # continue from where they left off instead of restarting from scratch.
    if cfg.resume and cfg.resume_step is not None:
        optimizer_ckpt_path = os.path.join(cfg.vla_path, f"optimizer--{cfg.resume_step}_checkpoint.pt")
        scheduler_ckpt_path = os.path.join(cfg.vla_path, f"scheduler--{cfg.resume_step}_checkpoint.pt")
        if os.path.exists(optimizer_ckpt_path):
            print(f"Loading optimizer state from {optimizer_ckpt_path}")
            optimizer.load_state_dict(torch.load(optimizer_ckpt_path, weights_only=True, map_location="cpu"))
        else:
            print(f"WARNING: Optimizer checkpoint not found at {optimizer_ckpt_path}, "
                  "optimizer state will be reinitialized (momentum/variance lost)")
        if os.path.exists(scheduler_ckpt_path):
            print(f"Loading scheduler state from {scheduler_ckpt_path}")
            scheduler.load_state_dict(torch.load(scheduler_ckpt_path, weights_only=False, map_location="cpu"))
        else:
            print(f"WARNING: Scheduler checkpoint not found at {scheduler_ckpt_path}, "
                  "LR schedule will restart from step 0")

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # train_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder,
    # )
    # ---

    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    use_wrist_image = cfg.num_images_in_input > 1

    # Create training and optional validation datasets
    assert not (cfg.use_mikasa_episodic and cfg.use_libero_episodic), (
        "use_mikasa_episodic and use_libero_episodic are mutually exclusive"
    )
    if cfg.use_mikasa_episodic:
        from prismatic.vla.datasets.mikasa_episodic_dataset import (
            MikasaBatchTransform,
            MikasaEpisodicCollator,
            MIKASARoboVLAEpisodicDataset,
        )

        mikasa_batch_transform = MikasaBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
            use_wrist_image=use_wrist_image,
            use_proprio=cfg.use_proprio,
            image_aug=cfg.image_aug,
            resize_resolution=tuple(vla.module.config.image_sizes),
        )
        env_names = [e.strip() for e in cfg.mikasa_env_names.split(",")]
        train_dataset = MIKASARoboVLAEpisodicDataset(
            data_root_dir=cfg.data_root_dir,
            env_names=env_names,
            batch_transform=mikasa_batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            batch_size=cfg.batch_size,
        )
        collator = MikasaEpisodicCollator(
            processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
        )
    elif cfg.use_libero_episodic:
        from prismatic.vla.datasets.libero_episodic_dataset import (
            LiberoBatchTransform,
            LiberoEpisodicCollator,
            LIBEROVLAEpisodicDataset,
        )

        libero_batch_transform = LiberoBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
            use_wrist_image=use_wrist_image,
            use_proprio=cfg.use_proprio,
            image_aug=cfg.image_aug,
            resize_resolution=tuple(vla.module.config.image_sizes),
        )
        suite_names = [s.strip() for s in cfg.libero_suite_names.split(",")]
        train_dataset = LIBEROVLAEpisodicDataset(
            data_root_dir=cfg.data_root_dir,
            suite_names=suite_names,
            batch_transform=libero_batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            batch_size=cfg.batch_size,
        )
        collator = LiberoEpisodicCollator(
            processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
        )
    else:
        batch_transform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
            use_wrist_image=use_wrist_image,
            use_proprio=cfg.use_proprio,
        )
        train_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            image_aug=cfg.image_aug,
        )
        collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
        )

    if cfg.use_val_set and not (cfg.use_mikasa_episodic or cfg.use_libero_episodic):
        batch_transform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
            use_wrist_image=use_wrist_image,
            use_proprio=cfg.use_proprio,
        )
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )

    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create dataloader
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important: Set to 0 if using RLDS / MIKASA episodic, which use their own parallelism
    )
    if cfg.use_val_set and not (cfg.use_mikasa_episodic or cfg.use_libero_episodic):
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        )

    # [mu-VLA] Compute effective accumulation steps for TBPTT / EMA.
    # For TBPTT: backward runs once per K batches (one K-step graph per backward),
    #   so one optimizer step spans K * grad_accumulation_steps batch iterations.
    # For EMA:   no cross-step graph, backward every batch (effective_tbptt=1).
    # NOTE: TBPTT K > 1 keeps K computation graphs in GPU memory simultaneously.
    if cfg.use_memory and memory_module is not None:
        assert cfg.memory_update in ("tbptt", "ema"), f"Invalid memory_update: {cfg.memory_update}"
        assert 0.0 <= cfg.ema_alpha <= 1.0, f"ema_alpha must be in [0, 1], got {cfg.ema_alpha}"
    assert cfg.attention_mask_mode in ("custom", "full"), (
        f"Invalid attention_mask_mode: {cfg.attention_mask_mode!r} (must be 'custom' or 'full')"
    )
    use_ema = cfg.use_memory and memory_module is not None and cfg.memory_update == "ema"
    if cfg.use_memory and memory_module is not None:
        effective_tbptt = 1 if cfg.memory_update == "ema" else cfg.tbptt_length
    else:
        effective_tbptt = 1
    effective_accum_steps = effective_tbptt * cfg.grad_accumulation_steps

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    metrics_window = effective_accum_steps
    recent_metrics = {
        "loss_value": deque(maxlen=metrics_window),
        "curr_action_accuracy": deque(maxlen=metrics_window),
        "curr_action_l1_loss": deque(maxlen=metrics_window),
        "next_actions_accuracy": deque(maxlen=metrics_window),
        "next_actions_l1_loss": deque(maxlen=metrics_window),
    }

    # [mu-VLA] Initialize memory state and TBPTT loss accumulator
    mem_state = None
    tbptt_loss_accum = None
    tbptt_count = 0
    if cfg.use_memory and memory_module is not None:
        mem_state = memory_module.get_initial_state(cfg.batch_size).to(device_id)
        tbptt_loss_accum = torch.tensor(0.0, device=device_id)

    # Start training
    resume_start = cfg.resume_step if cfg.resume and cfg.resume_step is not None else 0
    with tqdm.tqdm(total=cfg.max_steps, initial=resume_start, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):

            # [mu-VLA] Handle memory reset at episode boundaries
            current_mem_state = None
            if cfg.use_memory and memory_module is not None and mem_state is not None:
                is_first = batch["is_first"].to(device_id)
                if use_ema:
                    # EMA: mem_state is already detached between steps (see ema_update below).
                    # Default should_detach=~is_first is a no-op for continuing elements and
                    # correctly routes initial_memory into the graph for is_first elements.
                    current_mem_state = memory_module.reset_episodes(mem_state, is_first)
                else:
                    # TBPTT window: keep graph on continuing elements; detach happens after backward.
                    no_detach = torch.zeros_like(is_first)
                    current_mem_state = memory_module.reset_episodes(mem_state, is_first, no_detach)
                # Retain grad so we can log ∂L/∂mem_input after backward
                if memory_diagnostics is not None:
                    current_mem_state.retain_grad()

            # Compute training metrics and loss
            compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
            loss, metrics, new_mem_state = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
                compute_diffusion_l1=compute_diffusion_l1,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
                memory_state=current_mem_state,
                use_memory_mask=cfg.use_memory,
                attention_mask_mode=cfg.attention_mask_mode,
            )

            # [mu-VLA] Record step data for memory diagnostics (before detach)
            if memory_diagnostics is not None and new_mem_state is not None:
                memory_diagnostics.record_step(
                    mem_state, new_mem_state, is_first, batch_idx,
                    pixel_values=batch["pixel_values"],
                )
                # Retain grad on output memory so we can log ∂L/∂mem_output after backward
                new_mem_state.retain_grad()

            # [mu-VLA] Memory update: TBPTT (K-step graph) or EMA (no cross-step graph)
            if use_ema:
                # EMA ablation: backward every step; no loss accumulation across steps.
                # Within-step gradient flow reaches initial_memory via is_first elements only.
                normalized_loss = loss / effective_accum_steps
                normalized_loss.backward()

                # [mu-VLA] Sync initial_memory gradient across DDP processes
                if distributed_state.num_processes > 1 and memory_module.initial_memory.grad is not None:
                    dist.all_reduce(memory_module.initial_memory.grad, op=dist.ReduceOp.SUM)
                    memory_module.initial_memory.grad.div_(distributed_state.num_processes)

                # EMA blend for next step's memory input. Both operands detached by construction.
                mem_state = MemoryModule.ema_update(current_mem_state, new_mem_state, cfg.ema_alpha)
            elif cfg.use_memory and memory_module is not None:
                # TBPTT: accumulate losses over K steps, then backward once.
                tbptt_loss_accum = tbptt_loss_accum + loss
                tbptt_count += 1
                mem_state = new_mem_state  # Keep in computation graph for TBPTT!

                if tbptt_count >= cfg.tbptt_length:
                    # TBPTT boundary: backward through K timesteps of memory.
                    # Gradient flows: loss_K → action_hidden → attention → MEM_K → MEM_{K-1} → ... → MEM_1
                    normalized_loss = tbptt_loss_accum / effective_accum_steps
                    normalized_loss.backward()

                    # [mu-VLA] Sync initial_memory gradient across DDP processes
                    if distributed_state.num_processes > 1 and memory_module.initial_memory.grad is not None:
                        dist.all_reduce(memory_module.initial_memory.grad, op=dist.ReduceOp.SUM)
                        memory_module.initial_memory.grad.div_(distributed_state.num_processes)

                    # Truncate: detach memory for the next TBPTT window
                    mem_state = mem_state.detach()
                    tbptt_loss_accum = torch.tensor(0.0, device=device_id)
                    tbptt_count = 0
            else:
                # Standard path (no memory): backward every step
                normalized_loss = loss / cfg.grad_accumulation_steps
                normalized_loss.backward()

            # Store recent train metrics
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            # Compute gradient step index (accounts for TBPTT: K batches per backward)
            gradient_step_idx = batch_idx // effective_accum_steps

            # Compute smoothened train metrics
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)

            # Push Metrics to W&B (every wandb_log_freq gradient steps)
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)
                # W&B collapses repeated writes to the same step; stdout does not. With
                # TBPTT there are `effective_accum_steps` batches per gradient step, all
                # sharing one `log_step`, so echo on the last batch of the window only.
                if (batch_idx + 1) % effective_accum_steps == 0:
                    echo_metrics_to_stdout(smoothened_metrics, log_step, optimizer.param_groups[0]["lr"])

            # [If applicable] Linearly warm up learning rate from 10% to 100% of original
            # NOTE: Only used with "multistep" schedule. For "cosine", warmup is integrated into the scheduler.
            if cfg.lr_schedule == "multistep" and cfg.lr_warmup_steps > 0:
                # Use log_step (not gradient_step_idx) so warmup accounts for resume
                lr_progress = min((log_step + 1) / cfg.lr_warmup_steps, 1.0)  # Cap at 1.0
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                # Log the learning rate (use actual param group LR to capture both scheduler and manual warmup)
                wandb.log(
                    {
                        "VLA Train/Learning Rate": optimizer.param_groups[0]["lr"],
                    },
                    step=log_step,
                )

            # Diagnostics logging (memory metrics + attention heatmaps)
            if memory_diagnostics is not None:
                if current_mem_state is not None and memory_diagnostics.should_log(log_step):
                    # Gradients are populated after backward() if retain_grad() was called.
                    # ∂L/∂mem_input: does the model READ from memory?
                    # ∂L/∂mem_output: gradient signal that TRAINS memory writing.
                    mem_input_grad = current_mem_state.grad if current_mem_state.grad is not None else None
                    mem_output_grad = new_mem_state.grad if (new_mem_state is not None and new_mem_state.grad is not None) else None
                    memory_diagnostics.log_lightweight(
                        mem_state, memory_module, log_step,
                        mem_state_input_grad=mem_input_grad,
                        mem_state_output_grad=mem_output_grad,
                    )
                if memory_diagnostics.should_log_expensive(log_step):
                    memory_diagnostics.log_expensive(
                        vla=vla,
                        batch=batch,
                        current_mem_state=current_mem_state,  # None when use_memory=False
                        device_id=device_id,
                        log_step=log_step,
                        use_proprio=cfg.use_proprio,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        use_film=cfg.use_film,
                        use_diffusion=cfg.use_diffusion,
                        noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                        attention_mask_mode=cfg.attention_mask_mode,
                    )

            # Optimizer and LR scheduler step
            if (batch_idx + 1) % effective_accum_steps == 0:
                # Compute gradient norm before optimizer step (gradients are zeroed after step)
                if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                    total_norm_sq = 0.0
                    for p in trainable_params:
                        if p.grad is not None:
                            total_norm_sq += p.grad.data.float().norm(2).item() ** 2
                    grad_norm = total_norm_sq ** 0.5
                    wandb.log({"VLA Train/Gradient Norm": grad_norm}, step=log_step)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

                # Save model checkpoint: either keep latest checkpoint only or all checkpoints
                if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                    save_training_checkpoint(
                        cfg=cfg,
                        run_dir=run_dir,
                        log_step=log_step,
                        vla=vla,
                        processor=processor,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                        action_head=action_head if (cfg.use_l1_regression or cfg.use_diffusion) else None,
                        train_dataset=train_dataset,
                        distributed_state=distributed_state,
                        memory_module=memory_module if cfg.use_memory else None,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )

                # Test model on validation set
                if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                    run_validation(
                        vla=vla,
                        action_head=action_head,
                        noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        val_dataloader=val_dataloader,
                        action_tokenizer=action_tokenizer,
                        device_id=device_id,
                        cfg=cfg,
                        num_patches=NUM_PATCHES,
                        log_step=log_step,
                        distributed_state=distributed_state,
                        val_time_limit=cfg.val_time_limit,
                    )
                    # Set model back to training mode after validation
                    vla.train()

            # Stop training when max_steps is reached
            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
