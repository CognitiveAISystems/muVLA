<div align="center">

# μVLA: On Recurrent Memory for Partially Observable Manipulation in VLA Models

[![arXiv](https://img.shields.io/badge/arXiv-2606.12497-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.12497)
[![Project page](https://img.shields.io/badge/Project-Website-4c6ef5?style=for-the-badge&logo=googlechrome&logoColor=white)](https://avanturist322.github.io/mu-vla/)
[![Checkpoints](https://img.shields.io/badge/🤗%20Checkpoints-TBA-lightgrey?style=for-the-badge)](#checkpoints)
[![License](https://img.shields.io/badge/License-MIT-2f9e44?style=for-the-badge)](LICENSE)

<img src="https://avanturist322.github.io/project-websites/mu-vla/figures/visual_abstract_v5.png" alt="μVLA visual abstract" width="100%">

</div>

μVLA extends [OpenVLA-OFT](https://github.com/moojink/openvla-oft) with a recurrent
memory mechanism for partially observable manipulation. A small set of learnable memory
tokens is carried across timesteps within an episode, so the policy can condition on
observations that are no longer visible in the current frame.

The repository supports two benchmarks:

* **MIKASA-Robo** — memory-focused manipulation tasks, deliberately partially observable
  ([MIKASA-Robo](https://github.com/CognitiveAISystems/MIKASA-Robo)).
* **LIBERO** — the standard four-suite manipulation benchmark, used here with an episodic
  dataloader so the memory path can be checked against a familiar baseline.

## Method

<div align="center">
<img src="https://avanturist322.github.io/project-websites/mu-vla/figures/method_v3.png" alt="μVLA method overview" width="100%">
</div>

A small bank of learnable memory tokens is carried across timesteps inside the backbone
self-attention and updated end-to-end with TBPTT — no auxiliary losses, no architectural
additions.

### Attention mask with the memory-action guard

<div align="center">
<img src="https://avanturist322.github.io/project-websites/mu-vla/figures/attention_v1.png" alt="μVLA attention mask with the memory-action guard" width="100%">
</div>

Memory tokens attend only to observations, proprioception, language, and previous memory
state, but cannot read action tokens. This prevents the recurrent state from trivially
copying demonstrated actions and encourages encoding of task-relevant observations
instead.

## What is different from OpenVLA-OFT

| | OpenVLA-OFT | μVLA |
|---|---|---|
| Observation model | MDP (single frame + wrist) | POMDP (episodic, memory carried across steps) |
| Extra parameters | — | `MemoryModule`: learnable memory tokens in the multimodal prefix |
| Memory update | — | TBPTT truncation, or EMA (`M_in[t+1] = a * M_out[t] + (1-a) * M_in[t]`) |
| Dataloader | frame-shuffled RLDS | episodic (`is_first` / `is_last`), sequential within an episode |
| Attention | causal over the prefix | memory-aware mask (`--attention_mask_mode custom`) |

Memory is off by default; `--use_memory False` reproduces the OpenVLA-OFT baseline through
the same code path, which is how the baselines in both benchmark pages are produced.

## Documentation

Read these in order. Each benchmark page is self-contained past the shared install: it
covers its own data, supervised fine-tuning, and in-environment evaluation.

| Page | Contents |
|---|---|
| **[SETUP.md](SETUP.md)** | Install with `uv` (including the required `transformers` fork), download the `openvla-7b` base model, logging |
| **[MIKASA.md](MIKASA.md)** | MIKASA-Robo: dataset, training with and without memory, the evaluation protocol |
| **[LIBERO.md](LIBERO.md)** | LIBERO: the extra LIBERO install steps, dataset, combined-suite training, four-suite evaluation |

Shortest path from a clean machine to a trained-and-evaluated policy:

```bash
git clone https://github.com/CognitiveAISystems/muVLA.git && cd muVLA
uv sync --python 3.10                       # SETUP.md
# then follow MIKASA.md (sections 2, 3, 4) or LIBERO.md (sections 1, 2, 3, 4)
```

## System requirements

Inference:
* 1 GPU with ~16 GB VRAM, for either benchmark

Training:
* 1-8 GPUs with 40-80 GB (bfloat16). Runs with 64 memory tokens and
  `--use_gradient_checkpointing True` fit on a single 80 GB card at `--batch_size 4`.

## Checkpoints

Fine-tuned μVLA checkpoints will be released on the Hugging Face Hub: **TBA.**

Until then, both benchmark pages contain the exact training configurations the reported
numbers were produced with, so the checkpoints can be reproduced from the base
`openvla/openvla-7b` model.

## Acknowledgements

This repository builds directly on [OpenVLA-OFT](https://github.com/moojink/openvla-oft)
by Moo Jin Kim, Chelsea Finn and Percy Liang (MIT-licensed), see [LICENSE](LICENSE). If you use μVLA, please also cite OpenVLA-OFT
([arXiv:2502.19645](https://arxiv.org/abs/2502.19645)).

## Citation

```bibtex
@article{cherepanov2026muvla,
  title={{$\mu$}VLA: On Recurrent Memory for Partially Observable Manipulation in VLA Models},
  author={Cherepanov, Egor and Kachaev, Nikita and Zelezetsky, Daniil and Bulatov, Aydar and Pshenitsyn, Artem and Kuratov, Yuri and Skrynnik, Alexey and Panov, Aleksandr I and Kovalev, Alexey K},
  journal={arXiv preprint arXiv:2606.12497},
  year={2026}
}
```
