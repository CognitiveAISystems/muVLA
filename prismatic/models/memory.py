"""
memory.py

MemoryModule for mu-VLA: learnable recurrent memory tokens injected into the
multimodal prefix of the transformer sequence.

At t=0 (episode start): memory is initialized from a learnable nn.Parameter.
At t>0: memory is the hidden state extracted from MEM positions at the previous step.
TBPTT: continuing episodes get detached memory; new episodes keep gradient flow
through initial_memory so the model learns to initialize memory correctly.
"""

import torch
import torch.nn as nn


class MemoryModule(nn.Module):
    def __init__(self, num_mem_tokens: int, hidden_dim: int):
        super().__init__()
        self.num_mem_tokens = num_mem_tokens
        self.hidden_dim = hidden_dim
        self.initial_memory = nn.Parameter(
            torch.randn(num_mem_tokens, hidden_dim) * 0.02
        )

    def get_initial_state(self, batch_size: int) -> torch.Tensor:
        """Returns initial memory state for the entire batch: (B, num_mem_tokens, hidden_dim)."""
        return self.initial_memory.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def reset_episodes(
        self,
        mem_state: torch.Tensor,
        is_first: torch.Tensor,
        should_detach: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Per-element episode reset with TBPTT.

        Args:
            mem_state: (B, num_mem_tokens, hidden_dim) - current memory state
            is_first: (B,) bool - True where a new episode begins
            should_detach: (B,) bool or None - True where TBPTT truncation applies.
                If None, defaults to ~is_first (TBPTT length 1: detach every step).

        Returns:
            Updated mem_state where each element [i] is:
              - is_first[i]=True  -> initial_memory (in compute graph)
              - should_detach[i]=True (and not first) -> detached mem_state
              - otherwise -> mem_state as-is (in compute graph, within TBPTT window)
        """
        if should_detach is None:
            should_detach = ~is_first

        B = mem_state.shape[0]
        initial = self.initial_memory.unsqueeze(0).expand(B, -1, -1)

        # Use torch.where for clean gradient routing:
        # Priority: is_first > should_detach > keep in graph
        is_first_3d = is_first[:, None, None]
        should_detach_3d = (should_detach & ~is_first)[:, None, None]

        result = torch.where(
            is_first_3d,
            initial,
            torch.where(
                should_detach_3d,
                mem_state.detach(),
                mem_state,
            ),
        )
        return result

    @staticmethod
    def ema_update(
        mem_input: torch.Tensor,
        mem_output: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        EMA blend producing next step's memory input from current step's input and output.

            M^in_{t+1} = alpha * M^out_t + (1 - alpha) * M^in_t

        Both operands are detached: EMA ablation intentionally breaks cross-step
        gradient flow through memory. Gradients reach `initial_memory` only via
        within-step attention on `is_first=True` steps.

        Args:
            mem_input:  (B, num_mem_tokens, hidden_dim) — memory fed to the model
                        at step t (output of `reset_episodes`).
            mem_output: (B, num_mem_tokens, hidden_dim) — hidden states read from
                        MEM positions at step t.
            alpha:      Weight on the fresh output. alpha=1 recovers TBPTT-length-1
                        behavior (overwrite with output); alpha=0 freezes memory.

        Returns:
            Detached tensor of shape (B, num_mem_tokens, hidden_dim) to feed into
            the next step (before per-element `reset_episodes`).
        """
        return alpha * mem_output.detach() + (1.0 - alpha) * mem_input.detach()
