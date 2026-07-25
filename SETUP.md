# Setup

Everything on this page is shared by both benchmarks. Once it is done, continue with
[MIKASA.md](MIKASA.md) or [LIBERO.md](LIBERO.md), each of which is self-contained from
that point on (data, training, evaluation).

## Requirements

* Linux, x86-64
* Python 3.10 (the dependency set does not resolve on 3.11+)
* CUDA 12.x, one or more NVIDIA GPUs with >= 40 GB VRAM for training
* [`uv`](https://docs.astral.sh/uv/) for dependency management

## 1. Install μVLA

```bash
git clone https://github.com/CognitiveAISystems/muVLA.git
cd muVLA

# Creates .venv and installs the pinned dependency set from uv.lock
uv sync --python 3.10
```

`--python 3.10` lets `uv` use a system CPython 3.10 if there is one and download a managed
build if there is not, so this works on a machine that has no 3.10 installed. Pass an
explicit path (`--python /usr/bin/python3.10`) only if you need one particular interpreter.

That is the whole base install. It already includes the `transformers` fork and
`mikasa-robo-suite`; only LIBERO needs extra steps, and those are in
[LIBERO.md](LIBERO.md#1-install).

In particular, **`flash-attn` is not required.** Upstream OpenVLA-OFT asks for it, but
nothing in μVLA requests `attn_implementation="flash_attention_2"` — the one call site
is commented out, and the environment the reported results were produced in has no
`flash_attn` in it. Installing it is a long source build that buys nothing here.

Verify:

```bash
uv run python -c "import torch, transformers, prismatic; print(torch.__version__, transformers.__version__)"
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
uv run python -c "import mikasa_robo_suite; print('mikasa-robo-suite OK')"
```

### The `transformers` fork is required, not optional

`pyproject.toml` pins `transformers` to
[`CognitiveAISystems/transformers-mu-openvla-oft`](https://github.com/CognitiveAISystems/transformers-mu-openvla-oft)
at commit `9dbc09f5`. That fork adds the memory-aware attention mask that μVLA needs.
`uv sync` installs it for you; nothing else is required.

This matters because **substituting the stock PyPI `transformers` fails silently.** The
install succeeds, training starts, and losses look reasonable — but the memory tokens are
attended to as ordinary prefix tokens, so `--use_memory True` produces a model with no
working memory and memory metrics that mean nothing. If you manage dependencies by hand,
check that the fork is what actually got installed:

```bash
uv run python -c "import transformers, inspect; print(inspect.getsourcefile(transformers))"
uv run python -c "from transformers import LlamaConfig; assert hasattr(LlamaConfig(), 'use_mu_vla_memory_mask'); print('fork OK')"
```

### MIKASA-Robo is pinned to a git tag

`mikasa-robo-suite` is installed from the `v1.0.0` git tag, which is the exact revision
the reported results were produced against. Do not float this pin: v1.0.0 renamed two
observation keys (`prompt` -> `task_cue` and `joints` -> `proprio`) relative to earlier
releases, and `experiments/robot/mikasa_robo/mikasa_robo_utils.py` is written against the
v1.0.0 spelling of both.

**Known upstream gap.** MIKASA-Robo v1.0.0 declares no `package-data` rule in its
`pyproject.toml`, so `low_poly_light_bulb.glb` is dropped from *every* built
distribution — the PyPI wheel and a wheel built from the git tag alike. Eight lamp
environments load that asset at reset and will raise. None of them are part of the
32-environment μVLA benchmark, so this does not affect anything in [MIKASA.md](MIKASA.md).
To use the lamp environments, install MIKASA-Robo from a source checkout, or fix it
upstream with:

```toml
[tool.setuptools.package-data]
"mikasa_robo_suite" = ["**/*.glb"]
```

## 2. Download the base model

All fine-tuning starts from the `openvla/openvla-7b` checkpoint. Download it once and
keep a clean copy; every experiment then works on its own snapshot, so a crashed run can
never corrupt the shared baseline.

```bash
export BASE_MODELS_DIR="$PWD/artifacts/base_models"
mkdir -p "$BASE_MODELS_DIR"

SNAPSHOT_PATH=$(uv run python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('openvla/openvla-7b'))
")
cp -aL "$SNAPSHOT_PATH" "$BASE_MODELS_DIR/openvla-7b-clean"
```

Then make a per-experiment working copy. `MODEL_SNAPSHOT` is what the training commands
in [MIKASA.md](MIKASA.md#3-training) and [LIBERO.md](LIBERO.md#3-training) expect:

```bash
export EXP_DIR="$PWD/my_checkpoints/exp_mikasa"     # or exp_libero
mkdir -p "$EXP_DIR"
cp -aL "$BASE_MODELS_DIR/openvla-7b-clean/." "$EXP_DIR/base_model_snapshot"
export MODEL_SNAPSHOT="$EXP_DIR/base_model_snapshot"
```

## 3. Get the data

Both benchmarks are consumed as RLDS, and both datasets are on the Hugging Face Hub. The
download commands and the expected directory layouts are on the benchmark pages:

| Benchmark | Dataset | Where |
|---|---|---|
| MIKASA-Robo | [`mikasa-robo/mikasa-robo-vla-rlds`](https://huggingface.co/datasets/mikasa-robo/mikasa-robo-vla-rlds) | [MIKASA.md](MIKASA.md#2-data) |
| LIBERO | [`openvla/modified_libero_rlds`](https://huggingface.co/datasets/openvla/modified_libero_rlds) | [LIBERO.md](LIBERO.md#2-data) |

## 4. Logging

Weights & Biases is off by default: `finetune.py` starts a disabled run, so no account, no
API key and no `WANDB_MODE` export are needed. Every metric it would have logged is also
printed to stdout, one line per logged step, so a run is fully observable without it:

```
[train] step 0  | curr_action_l1_loss 0.7695 | loss 0.8164 | next_actions_l1_loss 0.8232 | lr 5.00e-05
[train] step 2  | curr_action_l1_loss 2.6836 | loss 2.6191 | next_actions_l1_loss 2.6094 | lr 5.04e-05
```

To use W&B instead, run `wandb login` once and pass `--use_wandb True --wandb_entity
<your-entity> --wandb_project <your-project>`.

## Next

* [MIKASA.md](MIKASA.md) — MIKASA-Robo: data, training, evaluation
* [LIBERO.md](LIBERO.md) — LIBERO: extra install steps, data, training, evaluation
