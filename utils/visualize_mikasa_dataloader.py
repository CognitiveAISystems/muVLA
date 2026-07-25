"""
visualize_mikasa_dataloader.py

Builds a PNG grid showing observations over time for each batch stream.
Rows = batch positions (streams), columns = timesteps.
Episode boundaries are highlighted: red border = is_first, green border = is_last.

Uses MIKASARoboVLAEpisodicDataset directly (same __init__ and _make_stream as training),
so the visualization exercises the exact same dataloader logic.
The only difference: batch_transform is replaced with an identity pass-through
(skips tokenization — not needed for visualization).

Usage:
    python utils/visualize_mikasa_dataloader.py [options]

Examples:
    # Default: 4 streams, 40 steps, all 3 envs
    python utils/visualize_mikasa_dataloader.py

    # One env, more steps
    python utils/visualize_mikasa_dataloader.py --batch_size 3 --n_steps 60 \\
        --env_names "remember_color_5_vla_v0" --output out/vis_rc5.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from prismatic.vla.datasets.mikasa_episodic_dataset import MIKASARoboVLAEpisodicDataset


# ── identity batch_transform ──────────────────────────────────────────────────

class _IdentityTransform:
    """
    Replaces MikasaBatchTransform for visualization purposes.
    Passes the raw step dict through unchanged — no tokenization needed.
    The dataset's __init__ (episode loading, normalization) and _make_stream
    are identical to training; only the transform at the end is skipped.
    """
    def __call__(self, step: dict) -> dict:
        return step


# ── visual constants ──────────────────────────────────────────────────────────
THUMB_W, THUMB_H = 96, 96
CELL_W = THUMB_W + 4             # 2px border on each side
CELL_H = THUMB_H + 20            # room for text label below
ROW_HEADER_W = 80
COL_HEADER_H = 24
PAD = 6

COLOR_IS_FIRST = (220, 60,  60)  # red   — episode start
COLOR_IS_LAST  = (60,  180, 60)  # green — episode end
COLOR_BOTH     = (220, 160, 40)  # yellow — single-step episode (both flags)
COLOR_NORMAL   = (180, 180, 180) # grey  — middle of episode

ENV_SHORT = {
    "shell_game_push_vla_v0":               "SGP",
    "intercept_medium_vla_v0":              "IM",
    "remember_color_5_vla_v0":              "RC5",
    "take_it_back_vla_v0":                  "TIB",
    "remember_shape_and_color_3x3_vla_v0":  "RSC3",
}


def _try_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _render_cell(step: dict) -> Image.Image:
    """Render one timestep as a (CELL_W × CELL_H) PIL image with a colored border."""
    img = Image.fromarray(step["image"]).resize((THUMB_W, THUMB_H), Image.BILINEAR)
    cell = Image.new("RGB", (CELL_W, CELL_H), (40, 40, 40))

    if step["is_first"] and step["is_last"]:
        border_color = COLOR_BOTH
    elif step["is_first"]:
        border_color = COLOR_IS_FIRST
    elif step["is_last"]:
        border_color = COLOR_IS_LAST
    else:
        border_color = COLOR_NORMAL

    border = Image.new("RGB", (THUMB_W + 4, THUMB_H + 4), border_color)
    border.paste(img, (2, 2))
    cell.paste(border, (0, 0))

    draw = ImageDraw.Draw(cell)
    dataset_name = step.get("dataset_name", "?")
    env_key = dataset_name.replace("mikasa_", "") if dataset_name != "?" else "?"
    env_short = ENV_SHORT.get(env_key, env_key[:3])
    t = step.get("t", "?")
    label = f"{env_short} t={t}"
    draw.text((2, THUMB_H + 5), label, fill=(220, 220, 220), font=_try_font(9))

    return cell


def build_visualization(
    data_root_dir: str,
    env_names: list,
    n_steps: int,
    batch_size: int,
    seed: int,
    output_path: str,
) -> None:

    # ── Create the dataset exactly as in training ─────────────────────────────
    # __init__: loads all episodes via tfds, computes + applies normalization
    # _make_stream: same round-robin infinite generator used in __iter__
    # Only difference: batch_transform is _IdentityTransform (no tokenization)
    print("Creating MIKASARoboVLAEpisodicDataset (same as training)...")
    dataset = MIKASARoboVLAEpisodicDataset(
        data_root_dir=Path(data_root_dir),
        env_names=env_names,
        batch_transform=_IdentityTransform(),
        resize_resolution=(224, 224),   # unused without real transform
        batch_size=batch_size,
        seed=seed,
    )

    # ── Collect n_steps per stream using dataset._make_stream ────────────────
    master_rng = np.random.default_rng(seed)

    def _with_step_counter(stream):
        """Wraps _make_stream to inject per-episode step counter `t`."""
        t = 0
        for step in stream:
            if step["is_first"]:
                t = 0
            yield dict(step, t=t)
            t += 1

    streams = [
        _with_step_counter(dataset._make_stream(np.random.default_rng(master_rng.integers(2**63))))
        for _ in range(batch_size)
    ]

    # round-robin collection: same order as MIKASARoboVLAEpisodicDataset.__iter__
    rows: list = [[] for _ in range(batch_size)]
    for global_idx in range(n_steps * batch_size):
        stream_idx = global_idx % batch_size
        rows[stream_idx].append(next(streams[stream_idx]))

    # ── Render grid ───────────────────────────────────────────────────────────
    grid_w = ROW_HEADER_W + n_steps * (CELL_W + PAD) + PAD
    grid_h = COL_HEADER_H + batch_size * (CELL_H + PAD) + PAD + 16  # +16 for legend
    canvas = Image.new("RGB", (grid_w, grid_h), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    font_hdr = _try_font(11)
    font_sm  = _try_font(9)

    # column headers
    for col in range(n_steps):
        x = ROW_HEADER_W + PAD + col * (CELL_W + PAD) + CELL_W // 2 - 8
        draw.text((x, 4), f"t{col}", fill=(160, 160, 160), font=font_sm)

    # rows
    for row_idx, row_steps in enumerate(rows):
        y_top = COL_HEADER_H + PAD + row_idx * (CELL_H + PAD)
        draw.text((4, y_top + CELL_H // 2 - 6), f"str{row_idx}",
                  fill=(200, 200, 200), font=font_hdr)
        for col, step in enumerate(row_steps):
            x_left = ROW_HEADER_W + PAD + col * (CELL_W + PAD)
            canvas.paste(_render_cell(step), (x_left, y_top))

    # legend
    legend_y = grid_h - 14
    draw.rectangle([(4, legend_y), (14, legend_y + 10)], fill=COLOR_IS_FIRST)
    draw.text((18, legend_y), "is_first (episode start)", fill=(200, 200, 200), font=font_sm)
    draw.rectangle([(175, legend_y), (185, legend_y + 10)], fill=COLOR_IS_LAST)
    draw.text((189, legend_y), "is_last (episode end)", fill=(200, 200, 200), font=font_sm)
    draw.rectangle([(340, legend_y), (350, legend_y + 10)], fill=COLOR_BOTH)
    draw.text((354, legend_y), "both (1-step episode)", fill=(200, 200, 200), font=font_sm)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"\nSaved → {out_path}  ({grid_w}×{grid_h} px)")

    # ── Episode transition summary ────────────────────────────────────────────
    print("\nEpisode transitions per stream:")
    for row_idx, row_steps in enumerate(rows):
        segments = []
        for col, step in enumerate(row_steps):
            if step["is_first"]:
                dataset_name = step.get("dataset_name", "?")
                env_key = dataset_name.replace("mikasa_", "") if dataset_name != "?" else "?"
                env_short = ENV_SHORT.get(env_key, env_key[:3])
                segments.append(f"col{col}:{env_short}")
        print(f"  stream {row_idx}: {' → '.join(segments)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_root_dir", default="data/mikasa_robo_vla_rlds")
    parser.add_argument("--env_names",
                        default="remember_color_5_vla_v0,take_it_back_vla_v0")
    parser.add_argument("--n_steps",    type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output",     default="out/mikasa_dataloader_vis.png")
    args = parser.parse_args()

    build_visualization(
        data_root_dir=args.data_root_dir,
        env_names=[e.strip() for e in args.env_names.split(",")],
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
