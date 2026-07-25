"""
memory_diagnostics.py

Diagnostics and wandb logging for mu-VLA recurrent memory during training.
All heavy visualization (attention heatmaps, per-episode plots) runs every N steps
in torch.no_grad(), keeping the training loop clean and fast.

Usage in finetune.py:
    diagnostics = MemoryDiagnostics(cfg, token_layout, wandb)
    # in training loop:
    diagnostics.record_step(mem_state, new_mem_state, is_first, batch_idx)
    if diagnostics.should_log(batch_idx):
        diagnostics.log_lightweight(mem_state, memory_module, batch_idx)
    if diagnostics.should_log_expensive(batch_idx):
        diagnostics.log_expensive(vla, batch, memory_state, ...)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TokenLayout:
    """Describes the position of each token group in the multimodal sequence."""
    num_bos: int = 1
    num_vision: int = 512        # e.g. 256 patches * 2 images
    num_proprio: int = 0         # 1 if use_proprio else 0
    num_diffusion: int = 0       # 1 if use_diffusion else 0
    num_mem: int = 0             # num_mem_tokens
    # num_text and num_action are dynamic per-batch — set before use
    num_text: int = 0
    num_action_and_stop: int = 57  # ACTION_DIM * NUM_ACTIONS_CHUNK + 1

    @property
    def group_names(self) -> List[str]:
        names = ["BOS", "VISION", "PROPRIO"]
        if self.num_diffusion > 0:
            names.append("DIFF_T")
        if self.num_mem > 0:
            names.append("MEM")
        names.extend(["TEXT", "ACTION", "STOP"])
        return names

    @property
    def group_sizes(self) -> List[int]:
        sizes = [self.num_bos, self.num_vision, self.num_proprio]
        if self.num_diffusion > 0:
            sizes.append(self.num_diffusion)
        if self.num_mem > 0:
            sizes.append(self.num_mem)
        num_action = self.num_action_and_stop - 1
        sizes.extend([self.num_text, num_action, 1])
        return sizes

    @property
    def total_prefix_tokens(self) -> int:
        """Total multimodal prefix tokens (everything before text in the sequence)."""
        return (self.num_bos + self.num_vision + self.num_proprio
                + self.num_diffusion + self.num_mem)

    @property
    def num_context_tokens(self) -> int:
        """Total context tokens (everything before ACTION+STOP)."""
        return self.total_prefix_tokens + self.num_text

    def update_text_from_input(self, input_ids_seq_len: int):
        """Compute num_text from input_ids shape (includes BOS + text + action + stop)."""
        self.num_text = input_ids_seq_len - self.num_bos - self.num_action_and_stop


class MemoryDiagnostics:
    """Collects and logs mu-VLA memory diagnostics to wandb."""

    def __init__(
        self,
        log_freq: int = 10,
        expensive_log_freq: int = 200,
        token_layout: Optional[TokenLayout] = None,
        wandb_module=None,
    ):
        self.log_freq = log_freq
        self.expensive_log_freq = expensive_log_freq
        self.layout = token_layout or TokenLayout()
        self.wandb = wandb_module

        # Episode tracking for batch element 0.
        # We keep two buffers: the current (in-progress) episode and the last
        # completed episode. Plots always use the *completed* episode so they
        # show the full trajectory from t=0 to T-1.
        self._current_norms: List[torch.Tensor] = []
        self._current_states: List[torch.Tensor] = []
        self._current_frames: List[np.ndarray] = []
        self._completed_norms: List[torch.Tensor] = []
        self._completed_states: List[torch.Tensor] = []
        self._completed_frames: List[np.ndarray] = []

    # ── Frequency checks ──────────────────────────────────────────────

    def should_log(self, batch_idx: int) -> bool:
        return self.log_freq > 0 and batch_idx % self.log_freq == 0

    def should_log_expensive(self, batch_idx: int) -> bool:
        return self.expensive_log_freq > 0 and batch_idx % self.expensive_log_freq == 0

    # ── Per-step data recording ───────────────────────────────────────

    def record_step(
        self,
        mem_state: torch.Tensor,
        new_mem_state: Optional[torch.Tensor],
        is_first: torch.Tensor,
        batch_idx: int,
        pixel_values: Optional[torch.Tensor] = None,
    ):
        """
        Record memory state for episode-level plots.
        Only tracks batch element 0. Accumulates ALL steps in an episode;
        when a new episode starts (is_first=True), the previous episode becomes
        the "completed" episode used for plotting.

        Args:
            pixel_values: (B, C, H, W) preprocessed images from the batch.
                          For 2-image input C=6 (base + wrist concatenated along channels).
                          Stored as uint8 numpy for the rollout strip.
        """
        if is_first[0].item():
            # Episode boundary: current episode (if any) becomes completed
            if len(self._current_norms) > 0:
                self._completed_norms = self._current_norms
                self._completed_states = self._current_states
                self._completed_frames = self._current_frames
            self._current_norms = []
            self._current_states = []
            self._current_frames = []

        if new_mem_state is not None:
            with torch.no_grad():
                state_b0 = new_mem_state[0].detach().float().cpu()  # (num_mem, D)
                norms = state_b0.norm(dim=-1)  # (num_mem,)
                self._current_norms.append(norms)
                self._current_states.append(state_b0)

        # Store observation frame for batch element 0
        if pixel_values is not None:
            with torch.no_grad():
                frame = self._pixel_values_to_frame(pixel_values[0])
                self._current_frames.append(frame)

    # ── Lightweight metrics (every log_freq steps) ────────────────────

    def log_lightweight(
        self,
        mem_state: torch.Tensor,
        memory_module: torch.nn.Module,
        log_step: int,
        mem_state_input_grad: Optional[torch.Tensor] = None,
        mem_state_output_grad: Optional[torch.Tensor] = None,
    ):
        """Log cheap per-step metrics: memory norms, gradient norms.

        Args:
            mem_state: current memory state (B, num_mem, D)
            memory_module: MemoryModule with initial_memory parameter
            log_step: wandb logging step
            mem_state_input_grad: ∂L/∂mem_input (B, num_mem, D).
                How much the loss depends on what was READ from memory.
                Near zero → model ignores memory content in attention.
            mem_state_output_grad: ∂L/∂mem_output (B, num_mem, D).
                Gradient signal that TRAINS memory writing. This is the
                signal that flows backward through TBPTT across timesteps.
                Near zero → loss doesn't care what's written to MEM positions.
                Growing → model is learning to use memory for future predictions.
        """
        if self.wandb is None:
            return

        logs = {}

        with torch.no_grad():
            # Mean memory token norm (batch-averaged, all tokens averaged)
            norms = mem_state.detach().float().norm(dim=-1)  # (B, num_mem)
            logs["mu-VLA Memory/mem_tokens_mean_norm"] = norms.mean().item()

        # Gradient norm of initial_memory parameter
        if memory_module.initial_memory.grad is not None:
            grad_norm = memory_module.initial_memory.grad.detach().float().norm().item()
            logs["mu-VLA Memory/initial_memory_grad_norm"] = grad_norm

        # initial_memory parameter norm
        param_norm = memory_module.initial_memory.detach().float().norm().item()
        logs["mu-VLA Memory/initial_memory_param_norm"] = param_norm

        # ∂L/∂mem_output — gradient signal that trains memory WRITING
        # This is the primary signal: it flows through TBPTT across timesteps
        # and tells us whether the model is learning to write useful info to memory.
        if mem_state_output_grad is not None:
            self._log_mem_grad(logs, mem_state_output_grad, prefix="mem_output")

        # ∂L/∂mem_input — does the model READ from memory?
        if mem_state_input_grad is not None:
            self._log_mem_grad(logs, mem_state_input_grad, prefix="mem_input")

        self.wandb.log(logs, step=log_step)

    @staticmethod
    def _log_mem_grad(logs: Dict, grad: torch.Tensor, prefix: str):
        """Log gradient norm metrics for a memory state tensor."""
        with torch.no_grad():
            grad_t = grad.detach().float()
            per_elem_norm = grad_t.norm(dim=(-2, -1))  # (B,)
            logs[f"mu-VLA Memory/{prefix}_grad_norm_mean"] = per_elem_norm.mean().item()
            logs[f"mu-VLA Memory/{prefix}_grad_norm_max"] = per_elem_norm.max().item()

    # ── Expensive metrics (every expensive_log_freq steps) ────────────

    def log_expensive(
        self,
        vla: torch.nn.Module,
        batch: Dict,
        current_mem_state: Optional[torch.Tensor],
        device_id: str,
        log_step: int,
        use_proprio: bool = False,
        proprio_projector: Optional[torch.nn.Module] = None,
        use_film: bool = False,
        use_diffusion: bool = False,
        noisy_action_projector: Optional[torch.nn.Module] = None,
        attention_mask_mode: str = "custom",
    ):
        """
        Run a separate no_grad forward pass with output_attentions=True,
        then log attention heatmap + episode-level plots.

        Works both with and without memory: if current_mem_state is None,
        the forward pass runs without memory tokens and the custom mask.
        """
        if self.wandb is None:
            return

        logs = {}

        # 1. Attention analysis (separate no_grad forward pass)
        try:
            attn_heatmap, group_attention_logs = self._compute_attention_analysis(
                vla, batch, current_mem_state, device_id,
                use_proprio, proprio_projector,
                use_film, use_diffusion, noisy_action_projector,
                attention_mask_mode=attention_mask_mode,
            )
            if attn_heatmap is not None:
                logs["mu-VLA Attention/block_attention_heatmap"] = attn_heatmap
            logs.update(group_attention_logs)
        except Exception as e:
            print(f"[mu-VLA Diagnostics] Attention analysis failed: {e}")

        # 2. Episode-level plots from LAST COMPLETED episode
        if len(self._completed_norms) > 1:
            try:
                norm_plot = self._plot_episode_norms(self._completed_norms)
                if norm_plot is not None:
                    logs["mu-VLA Memory/episode_mem_norms"] = norm_plot
            except Exception as e:
                print(f"[mu-VLA Diagnostics] Norm plot failed: {e}")

            try:
                cosine_plot = self._plot_episode_cosine_distance(self._completed_states)
                if cosine_plot is not None:
                    logs["mu-VLA Memory/episode_cosine_distance"] = cosine_plot
            except Exception as e:
                print(f"[mu-VLA Diagnostics] Cosine distance plot failed: {e}")

        # 3. Episode rollout strip from LAST COMPLETED episode
        if len(self._completed_frames) > 0:
            try:
                rollout_img = self._plot_episode_rollout(self._completed_frames)
                if rollout_img is not None:
                    logs["mu-VLA Memory/episode_rollout"] = rollout_img
            except Exception as e:
                print(f"[mu-VLA Diagnostics] Rollout plot failed: {e}")

        if logs:
            self.wandb.log(logs, step=log_step)

    # ── Attention analysis ────────────────────────────────────────────

    def _compute_attention_analysis(
        self,
        vla: torch.nn.Module,
        batch: Dict,
        current_mem_state: Optional[torch.Tensor],
        device_id: str,
        use_proprio: bool,
        proprio_projector: Optional[torch.nn.Module],
        use_film: bool,
        use_diffusion: bool,
        noisy_action_projector: Optional[torch.nn.Module],
        attention_mask_mode: str = "custom",
    ) -> Tuple[Optional[object], Dict]:
        """
        Run a no_grad forward pass with output_attentions=True to extract
        attention weights. Returns (heatmap_wandb_image, group_attention_logs).

        If current_mem_state is None, runs without memory (standard OpenVLA-OFT attention).
        """
        vla_module = vla.module if hasattr(vla, 'module') else vla

        # Update layout from this batch
        self.layout.update_text_from_input(batch["input_ids"].shape[1])

        # Determine memory arguments
        use_memory = current_mem_state is not None
        mem_state_arg = current_mem_state.detach() if use_memory else None

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = vla_module(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                labels=batch["labels"].to(device_id),  # needed for action mask processing
                output_attentions=True,
                output_hidden_states=False,
                proprio=batch["proprio"].to(device_id) if use_proprio else None,
                proprio_projector=proprio_projector.module if (
                    use_proprio and proprio_projector is not None and hasattr(proprio_projector, 'module')
                ) else (proprio_projector if use_proprio else None),
                noisy_actions=None,
                noisy_action_projector=None,
                diffusion_timestep_embeddings=None,
                use_film=use_film,
                memory_state=mem_state_arg,
                use_memory_mask=use_memory,
                attention_mask_mode=attention_mask_mode,
            )

        # output.attentions: tuple of (B, num_heads, seq_len, seq_len) per layer
        if output.attentions is None or len(output.attentions) == 0:
            return None, {}

        # Aggregate block attention: average over layers and heads, take batch element 0
        block_attn = self._aggregate_block_attention(output.attentions)
        heatmap = self._plot_block_attention(block_attn)

        # Group attention fractions (scalar metrics for wandb line charts)
        group_logs = self._compute_group_attention_fractions(block_attn)

        return heatmap, group_logs

    def _aggregate_block_attention(
        self,
        attentions: Tuple[torch.Tensor, ...],
    ) -> np.ndarray:
        """
        Aggregate per-position attention weights into block-level attention matrix.

        Args:
            attentions: tuple of (B, num_heads, seq_len, seq_len), one per layer

        Returns:
            block_attn: (num_groups, num_groups) numpy array,
                        block_attn[i, j] = mean attention mass from group i queries to group j keys
        """
        group_sizes = self.layout.group_sizes
        num_groups = len(group_sizes)

        # Compute group boundaries
        boundaries = np.cumsum([0] + group_sizes)

        # Accumulate over layers, take batch element 0, average over heads
        block_attn_sum = np.zeros((num_groups, num_groups), dtype=np.float64)
        num_layers = len(attentions)

        for layer_attn in attentions:
            # layer_attn: (B, num_heads, seq_len, seq_len)
            # Take batch 0, average over heads
            attn_b0 = layer_attn[0].float().mean(dim=0).cpu().numpy()  # (seq_len, seq_len)

            for qi in range(num_groups):
                q_start, q_end = boundaries[qi], boundaries[qi + 1]
                num_queries = q_end - q_start
                if num_queries == 0:
                    continue
                for ki in range(num_groups):
                    k_start, k_end = boundaries[ki], boundaries[ki + 1]
                    if k_end <= k_start:
                        continue
                    # Total attention mass from query group qi to key group ki,
                    # averaged over query positions (so each row sums to 1.0).
                    block = attn_b0[q_start:q_end, k_start:k_end]
                    block_attn_sum[qi, ki] += block.sum() / num_queries

        block_attn = block_attn_sum / num_layers
        return block_attn

    def _plot_block_attention(self, block_attn: np.ndarray):
        """Create a wandb Image of the block attention heatmap."""
        group_names = self.layout.group_names
        num_groups = len(group_names)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(block_attn, cmap="viridis", aspect="auto")

        # Annotate cells
        for i in range(num_groups):
            for j in range(num_groups):
                val = block_attn[i, j]
                color = "white" if val < 0.5 * block_attn.max() else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=8)

        ax.set_xticks(range(num_groups))
        ax.set_xticklabels(group_names, rotation=45, ha="right")
        ax.set_yticks(range(num_groups))
        ax.set_yticklabels(group_names)
        ax.set_xlabel("Key Group")
        ax.set_ylabel("Query Group")
        ax.set_title("Block attention mass (mean over layers & heads)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()

        img = self.wandb.Image(fig)
        plt.close(fig)
        return img

    def _compute_group_attention_fractions(self, block_attn: np.ndarray) -> Dict:
        """Compute per-query-group attention distribution as scalar metrics."""
        group_names = self.layout.group_names
        logs = {}

        for qi, q_name in enumerate(group_names):
            row_sum = block_attn[qi].sum()
            if row_sum < 1e-10:
                continue
            for ki, k_name in enumerate(group_names):
                frac = block_attn[qi, ki] / row_sum
                logs[f"mu-VLA Attention/{q_name}_to_{k_name}_frac"] = float(frac)

        return logs

    # ── Episode-level plots ───────────────────────────────────────────

    def _plot_episode_norms(self, norms_list: List[torch.Tensor]):
        """Plot memory token norms over timesteps for a completed episode."""
        if len(norms_list) < 2:
            return None

        norms = torch.stack(norms_list).numpy()  # (T, num_mem)
        T, num_mem = norms.shape

        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(num_mem):
            ax.plot(range(T), norms[:, i], label=f"MEM_{i}", alpha=0.8)
        ax.set_xlabel("Timestep in episode")
        ax.set_ylabel("L2 norm")
        ax.set_title(f"Memory token norms over episode ({T} steps, batch_idx=0)")
        ax.set_xticks(range(T))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        img = self.wandb.Image(fig)
        plt.close(fig)
        return img

    def _plot_episode_cosine_distance(self, states_list: List[torch.Tensor]):
        """Plot cosine distance between consecutive memory states over episode."""
        if len(states_list) < 2:
            return None

        states = torch.stack(states_list)  # (T, num_mem, D)
        T, num_mem, D = states.shape

        # Cosine distance between consecutive steps: 1 - cos_sim(t, t-1)
        cos_sim = F.cosine_similarity(states[1:], states[:-1], dim=-1)  # (T-1, num_mem)
        cos_dist = (1 - cos_sim).numpy()

        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(num_mem):
            ax.plot(range(1, T), cos_dist[:, i], label=f"MEM_{i}", alpha=0.8)
        ax.set_xlabel("Timestep in episode")
        ax.set_ylabel("Cosine distance (t vs t-1)")
        ax.set_title(f"Memory token cosine distance over episode ({T} steps, batch_idx=0)")
        ax.set_xticks(range(1, T))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        img = self.wandb.Image(fig)
        plt.close(fig)
        return img

    # ── Observation rollout ──────────────────────────────────────────

    @staticmethod
    def _pixel_values_to_frame(pv: torch.Tensor) -> np.ndarray:
        """
        Convert a single preprocessed pixel_values tensor to a column of camera images.

        Args:
            pv: (C, H, W) tensor. For a dual-backbone with 2 cameras:
                C=12 = 2 cameras × 2 backbone views × 3 channels.
                Layout: [base_view1(3), base_view2(3), wrist_view1(3), wrist_view2(3)].
                For single camera: C=6 = 1 camera × 2 views × 3 channels.
                Normalized with mean≈0.5, std≈0.5.

        Returns:
            (H*num_cameras, W, 3) uint8 numpy — one image per camera stacked vertically.
            Only the first backbone view is shown (the second is a different
            resolution/preprocessing of the same image).
        """
        pv = pv.detach().float().cpu()
        C, H, W = pv.shape

        # Dual-backbone: each camera produces 6 channels (2 views × 3 ch).
        # We display only the first view (channels 0:3) per camera.
        channels_per_camera = 6  # 2 backbone views × 3 RGB channels
        num_cameras = C // channels_per_camera

        # Denormalize: raw = pixel * 0.5 + 0.5, clamp to [0, 1]
        pv = pv * 0.5 + 0.5
        pv = pv.clamp(0, 1)

        panels = []
        for i in range(num_cameras):
            # Take only the first 3 channels (first backbone view) of each camera
            img = pv[i * channels_per_camera : i * channels_per_camera + 3]  # (3, H, W)
            img = img.permute(1, 2, 0).numpy()  # (H, W, 3)
            panels.append(img)

        # Stack vertically: (H*num_cameras, W, 3)
        combined = np.concatenate(panels, axis=0)
        return (combined * 255).astype(np.uint8)

    _MAX_ROLLOUT_STEPS = 100

    def _plot_episode_rollout(self, frames: List[np.ndarray]):
        """
        Create a grid of observation frames for a completed episode.
        Each column is one timestep with images stacked vertically
        (base camera on top, wrist on bottom).

        If the episode has more than _MAX_ROLLOUT_STEPS frames, returns a placeholder
        image with a skip message instead.
        """
        T = len(frames)
        if T == 0:
            return None

        if T > self._MAX_ROLLOUT_STEPS:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(
                0.5, 0.5,
                f"Episode too long ({T} steps > {self._MAX_ROLLOUT_STEPS}), rollout skipped",
                ha="center", va="center", fontsize=12, color="red",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            fig.tight_layout()
            img = self.wandb.Image(fig)
            plt.close(fig)
            return img

        # Pick evenly spaced frames (up to ~20) if episode is long-ish
        max_panels = 20
        if T <= max_panels:
            indices = list(range(T))
        else:
            indices = np.linspace(0, T - 1, max_panels, dtype=int).tolist()

        num_panels = len(indices)
        # Each frame is (H_total, W, 3) where H_total = H * num_images (stacked vertically)
        H_total, W, _ = frames[0].shape

        cell_w = 1.5
        cell_h = cell_w * H_total / W
        fig, axes = plt.subplots(
            1, num_panels,
            figsize=(cell_w * num_panels, cell_h + 0.4),
        )
        if num_panels == 1:
            axes = [axes]

        for ax, idx in zip(axes, indices):
            ax.imshow(frames[idx])
            ax.set_title(f"t={idx}", fontsize=7)
            ax.axis("off")

        fig.suptitle(f"Memory rollout observations ({T} steps, batch_idx=0)", fontsize=10)
        fig.tight_layout()

        img = self.wandb.Image(fig)
        plt.close(fig)
        return img
