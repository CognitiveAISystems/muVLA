"""
mikasa_episodic_dataset.py

POMDP-aware episodic dataset for MIKASA-Robo environments.
Preserves temporal ordering within episodes and streams complete episodes
through fixed batch positions — critical for partially observable environments.

Episodes are loaded lazily one-at-a-time from TFDS (not all into RAM).
Normalization statistics are computed in a streaming pass (action+proprio only,
no images) and cached to disk.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type

import numpy as np
import tensorflow_datasets as tfds
import torch
from PIL import Image
from torch.utils.data import IterableDataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from prismatic.models.backbones.llm.prompting import PromptBuilder
    from prismatic.models.backbones.vision import ImageTransform
    from prismatic.vla.action_tokenizer import ActionTokenizer

# Import lightweight constants directly (no heavy deps)
from prismatic.vla.constants import IGNORE_INDEX, NUM_ACTIONS_CHUNK


# === Helper Functions ===


def _resolve_versioned_dir(env_dir: Path) -> Path:
    """Resolve versioned subdirectory (e.g., remember_color_5_vla_v0/1.0.0/).

    The argument is a dataset directory, not a Gym environment id: the published RLDS
    repository spells its directories snake_case.
    """
    version_dirs = sorted([d for d in env_dir.iterdir() if d.is_dir()])
    if version_dirs:
        return version_dirs[-1]
    return env_dir


def _load_episodes_from_rlds(data_dir: str, env_name: str) -> List[Dict[str, Any]]:
    """
    Loads all episodes from an RLDS dataset directory into RAM.

    .. deprecated::
        Kept for backward compatibility with scripts that monkey-patch
        ``self.all_episodes``.  New code should use the streaming path.
    """
    data_path = _resolve_versioned_dir(Path(data_dir))
    builder = tfds.builder_from_directory(str(data_path))
    dataset = builder.as_dataset(split="train")

    episodes = []
    for episode in dataset:
        steps = list(episode["steps"])
        T = len(steps)
        if T == 0:
            continue

        images = np.stack([step["observation"]["image"].numpy() for step in steps])
        wrist_images = np.stack([step["observation"]["wrist_image"].numpy() for step in steps])
        proprio = np.stack([step["observation"]["proprio"].numpy() for step in steps])
        actions = np.stack([step["action"].numpy() for step in steps])

        lang = steps[0]["language_instruction"].numpy()
        if isinstance(lang, bytes):
            lang = lang.decode("utf-8")

        episodes.append({
            "images": images,
            "wrist_images": wrist_images,
            "proprio": proprio,
            "actions": actions,
            "language_instruction": lang,
            "env_name": env_name,
        })

    return episodes


def _load_single_episode_from_tf(tf_episode) -> Dict[str, Any]:
    """Extract a single episode from a TF dataset element into numpy arrays."""
    steps = list(tf_episode["steps"])
    T = len(steps)
    if T == 0:
        return None

    images = np.stack([step["observation"]["image"].numpy() for step in steps])
    wrist_images = np.stack([step["observation"]["wrist_image"].numpy() for step in steps])
    proprio = np.stack([step["observation"]["proprio"].numpy() for step in steps])
    actions = np.stack([step["action"].numpy() for step in steps])

    lang = steps[0]["language_instruction"].numpy()
    if isinstance(lang, bytes):
        lang = lang.decode("utf-8")

    return {
        "images": images,
        "wrist_images": wrist_images,
        "proprio": proprio,
        "actions": actions,
        "language_instruction": lang,
    }


def _compute_dataset_statistics(
    all_episodes: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Computes combined normalization statistics across all episodes (in-memory variant).
    Returns dict compatible with save_dataset_statistics().
    """
    all_actions = np.concatenate([ep["actions"] for ep in all_episodes], axis=0)
    all_proprio = np.concatenate([ep["proprio"] for ep in all_episodes], axis=0)

    return _compute_stats_from_arrays(all_actions, all_proprio, len(all_episodes))


def _compute_stats_from_arrays(
    all_actions: np.ndarray,
    all_proprio: np.ndarray,
    num_trajectories: int,
) -> Dict[str, Any]:
    """Compute statistics dict from concatenated arrays."""
    stats = {
        "action": {
            "mean": np.mean(all_actions, axis=0).astype(np.float32),
            "std": np.std(all_actions, axis=0).astype(np.float32),
            "min": np.min(all_actions, axis=0).astype(np.float32),
            "max": np.max(all_actions, axis=0).astype(np.float32),
            "q01": np.quantile(all_actions, 0.01, axis=0).astype(np.float32),
            "q99": np.quantile(all_actions, 0.99, axis=0).astype(np.float32),
        },
        "proprio": {
            "mean": np.mean(all_proprio, axis=0).astype(np.float32),
            "std": np.std(all_proprio, axis=0).astype(np.float32),
            "min": np.min(all_proprio, axis=0).astype(np.float32),
            "max": np.max(all_proprio, axis=0).astype(np.float32),
            "q01": np.quantile(all_proprio, 0.01, axis=0).astype(np.float32),
            "q99": np.quantile(all_proprio, 0.99, axis=0).astype(np.float32),
        },
        "num_transitions": int(all_actions.shape[0]),
        "num_trajectories": int(num_trajectories),
    }
    return stats


def _compute_statistics_streaming(
    env_dirs: Dict[str, Path],
) -> Dict[str, Any]:
    """
    Compute normalization statistics by streaming through TFDS datasets.

    Only loads action + proprio per episode (no images), so RAM usage is
    O(total_steps * action_dim * 4 bytes) — ~250 MB for 90 tasks.
    """
    all_actions: List[np.ndarray] = []
    all_proprio: List[np.ndarray] = []
    num_trajectories = 0

    for env_name, env_dir in sorted(env_dirs.items()):
        versioned_dir = _resolve_versioned_dir(env_dir)
        builder = tfds.builder_from_directory(str(versioned_dir))
        dataset = builder.as_dataset(split="train")

        for episode in dataset:
            steps = list(episode["steps"])
            if len(steps) == 0:
                continue
            actions = np.stack([s["action"].numpy() for s in steps])
            proprio = np.stack([s["observation"]["proprio"].numpy() for s in steps])
            all_actions.append(actions)
            all_proprio.append(proprio)
            num_trajectories += 1

    all_actions = np.concatenate(all_actions, axis=0)
    all_proprio = np.concatenate(all_proprio, axis=0)

    return _compute_stats_from_arrays(all_actions, all_proprio, num_trajectories)


def _stats_cache_path(data_root_dir: Path, env_names: List[str]) -> Path:
    """Deterministic cache path based on sorted env names."""
    key = ",".join(sorted(env_names))
    h = hashlib.sha256(key.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return data_root_dir / f"mikasa_statistics_{h}.json"


def _load_or_compute_statistics(
    data_root_dir: Path,
    env_names: List[str],
    env_dirs: Dict[str, Path],
) -> Dict[str, Any]:
    """
    Load cached statistics from disk, or compute them by streaming through
    all TFDS datasets (action + proprio only, no images).
    """
    cache_path = _stats_cache_path(data_root_dir, env_names)

    if cache_path.exists():
        print(f"[MIKASARoboVLAEpisodicDataset] Loading cached statistics from {cache_path}")
        with open(cache_path, "r") as f:
            stats = json.load(f)
        # Convert lists back to numpy arrays
        for group in ["action", "proprio"]:
            for k, v in stats[group].items():
                if isinstance(v, list):
                    stats[group][k] = np.array(v, dtype=np.float32)
        return stats

    print("[MIKASARoboVLAEpisodicDataset] Computing statistics (streaming, no images)...")
    stats = _compute_statistics_streaming(env_dirs)

    # Save to disk (convert numpy to lists for JSON)
    stats_json = {}
    for group in ["action", "proprio"]:
        stats_json[group] = {}
        for k, v in stats[group].items():
            if isinstance(v, np.ndarray):
                stats_json[group][k] = v.tolist()
            else:
                stats_json[group][k] = v
    stats_json["num_transitions"] = stats["num_transitions"]
    stats_json["num_trajectories"] = stats["num_trajectories"]

    with open(cache_path, "w") as f:
        json.dump(stats_json, f, indent=2)
    print(f"[MIKASARoboVLAEpisodicDataset] Cached statistics to {cache_path}")

    return stats


def _normalize_bounds_q99(
    data: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Normalize to [-1, 1] using q01/q99 bounds. Optionally mask dimensions."""
    denom = q99 - q01 + 1e-8
    normalized = 2.0 * (data - q01) / denom - 1.0
    normalized = np.clip(normalized, -1.0, 1.0)
    if mask is not None:
        normalized = np.where(mask, normalized, data)
    return normalized.astype(np.float32)


def _chunk_actions_numpy(
    actions: np.ndarray,
    timestep: int,
    num_actions_chunk: int,
) -> np.ndarray:
    """
    Returns action chunk [action_t, action_{t+1}, ..., action_{t+chunk-1}].
    Pads by repeating last valid action if chunk extends beyond episode end.
    """
    T = actions.shape[0]
    chunk = np.zeros((num_actions_chunk, actions.shape[1]), dtype=np.float32)
    for i in range(num_actions_chunk):
        idx = min(timestep + i, T - 1)
        chunk[i] = actions[idx]
    return chunk


# === Batch Transform ===


# Image augmentation config — must exactly match RLDSDataset (prismatic/vla/datasets/datasets.py)
_IMAGE_AUGMENT_KWARGS = dict(
    random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
    random_brightness=[0.2],
    random_contrast=[0.8, 1.2],
    random_saturation=[0.8, 1.2],
    random_hue=[0.05],
    augment_order=[
        "random_resized_crop",
        "random_brightness",
        "random_contrast",
        "random_saturation",
        "random_hue",
    ],
)


def _apply_image_augmentation(img_np: np.ndarray, resize_size: Tuple[int, int]) -> np.ndarray:
    """
    Apply the exact same image augmentations as the RLDS pipeline.

    RLDS order: decode -> resize (Lanczos3) -> augment.
    We replicate this by resizing first, then augmenting at the target resolution.

    Uses dlimp.transforms (TensorFlow ops) to guarantee identical behavior:
    same Lanczos3 resize, same random_resized_crop, same HSV-based saturation
    and hue adjustments, same brightness/contrast math.

    Args:
        img_np: (H, W, 3) uint8 numpy array.
        resize_size: (height, width) target resolution matching RLDS resize_size.

    Returns:
        (H, W, 3) uint8 numpy array, resized and augmented.
    """
    import tensorflow as tf
    from dlimp.transforms import augment_image, resize_image

    tf_img = tf.constant(img_np)
    tf_img = resize_image(tf_img, resize_size)
    tf_img = augment_image(tf_img, **_IMAGE_AUGMENT_KWARGS)
    return tf_img.numpy()


@dataclass
class MikasaBatchTransform:
    """Converts a single MIKASA step dict to the format expected by the collator/model."""

    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    image_aug: bool = False
    resize_resolution: Tuple[int, int] = (224, 224)

    def __call__(self, step: Dict[str, Any]) -> Dict[str, Any]:
        # Apply image augmentation (operates on numpy uint8).
        # RLDS order: decode -> resize (Lanczos3, to resize_resolution) -> augment.
        # We replicate this: resize first, then augment at target resolution.
        # image_transform afterwards (Resize/CenterCrop are no-ops on 224x224 -> ToTensor -> Normalize).
        image_np = step["image"]
        wrist_np = step.get("wrist_image")
        if self.image_aug:
            image_np = _apply_image_augmentation(image_np, self.resize_resolution)
            if wrist_np is not None:
                wrist_np = _apply_image_augmentation(wrist_np, self.resize_resolution)

        img = Image.fromarray(image_np)
        lang = step["language_instruction"].lower()
        action_chunk = step["action"]  # [NUM_ACTIONS_CHUNK, ACTION_DIM]

        # Build prompt (same logic as RLDSBatchTransform)
        prompt_builder = self.prompt_builder_fn("openvla")

        current_action = action_chunk[0]
        future_actions = action_chunk[1:]

        current_action_string = self.action_tokenizer(current_action)
        future_actions_string = "".join(self.action_tokenizer(future_actions))
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # Image transform
        pixel_values = self.image_transform(img)

        # Only predict action tokens (and optionally stop token)
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=step["dataset_name"],
            actions=action_chunk,
            is_first=step["is_first"],
            is_last=step["is_last"],
        )

        if self.use_wrist_image and wrist_np is not None:
            img_wrist = Image.fromarray(wrist_np)
            return_dict["pixel_values_wrist"] = self.image_transform(img_wrist)

        if self.use_proprio:
            return_dict["proprio"] = step["proprio"]

        return return_dict


# === Collator ===


@dataclass
class MikasaEpisodicCollator:
    """Extends PaddedCollatorForActionPrediction with episode boundary flags."""

    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        from prismatic.util.data_utils import PaddedCollatorForActionPrediction

        base_collator = PaddedCollatorForActionPrediction(
            self.model_max_length, self.pad_token_id, self.padding_side, self.pixel_values_dtype
        )
        output = base_collator(instances)

        if "is_first" in instances[0]:
            output["is_first"] = torch.tensor([inst["is_first"] for inst in instances], dtype=torch.bool)
        if "is_last" in instances[0]:
            output["is_last"] = torch.tensor([inst["is_last"] for inst in instances], dtype=torch.bool)

        return output


# === Main Dataset ===


class MIKASARoboVLAEpisodicDataset(IterableDataset):
    """
    POMDP-aware episodic dataset for MIKASA-Robo environments.

    Each batch position is a persistent "stream" that yields sequential
    timesteps from complete episodes. When one episode ends, the next
    begins immediately from the same stream.

    Episodes are loaded lazily one-at-a-time from TFDS — only one episode
    per stream is in RAM at any time.  Normalization statistics are computed
    via a streaming pass (action + proprio only) and cached to disk.

    Yields samples in round-robin order across batch_size streams, so that
    DataLoader(batch_size=B) naturally groups them into correct batches.
    """

    def __init__(
        self,
        data_root_dir: Path,
        env_names: List[str],
        batch_transform: MikasaBatchTransform,
        resize_resolution: Tuple[int, int],
        batch_size: int = 4,
        seed: int = 42,
    ) -> None:
        self.batch_transform = batch_transform
        self.batch_size = batch_size
        self.seed = seed
        self.resize_resolution = resize_resolution

        data_root_dir = Path(data_root_dir)
        self.env_names = list(env_names)
        self.env_dirs: Dict[str, Path] = {
            env_name: data_root_dir / env_name for env_name in env_names
        }

        # Load or compute normalization statistics (streaming, no images loaded)
        combined_stats = _load_or_compute_statistics(data_root_dir, env_names, self.env_dirs)

        # Store as dataset_statistics in the format expected by save_dataset_statistics
        self.dataset_statistics = {"mikasa_combined": combined_stats}

        # Store normalization params for on-the-fly normalization in streams
        self._q01_action = np.asarray(combined_stats["action"]["q01"], dtype=np.float32)
        self._q99_action = np.asarray(combined_stats["action"]["q99"], dtype=np.float32)
        self._q01_proprio = np.asarray(combined_stats["proprio"]["q01"], dtype=np.float32)
        self._q99_proprio = np.asarray(combined_stats["proprio"]["q99"], dtype=np.float32)

        # Approximate dataset length (for progress bars)
        self._dataset_length = int(combined_stats["num_transitions"])

        print(
            f"[MIKASARoboVLAEpisodicDataset] {len(env_names)} envs, "
            f"~{self._dataset_length} steps, streaming mode"
        )

    def _make_env_iterator(self, env_name: str, seed: int):
        """
        Create a shuffled, infinitely-repeating TFDS iterator for one env.

        The shuffle buffer operates at the episode level — episodes are
        reordered within each pass through the dataset.
        """
        env_dir = self.env_dirs[env_name]
        versioned_dir = _resolve_versioned_dir(env_dir)
        builder = tfds.builder_from_directory(str(versioned_dir))
        dataset = builder.as_dataset(split="train")
        dataset = dataset.shuffle(buffer_size=200, seed=seed).repeat()
        return iter(dataset)

    def _make_stream(self, rng: np.random.Generator) -> Iterator[Dict[str, Any]]:
        """
        Infinite generator for a single batch position's stream.
        Yields one step dict at a time from sequentially sampled episodes.

        Each call creates independent TFDS iterators per env, so multiple
        streams do not interfere with each other.
        """
        # One shuffled iterator per env, independent for this stream
        env_iters = {
            env_name: self._make_env_iterator(env_name, int(rng.integers(2**31)))
            for env_name in self.env_names
        }

        while True:
            # Pick a random environment for the next episode
            env_name = rng.choice(self.env_names)
            env_iter = env_iters[env_name]

            # Load ONE episode from the TFDS iterator
            tf_episode = next(env_iter)
            ep = _load_single_episode_from_tf(tf_episode)
            if ep is None:
                continue

            T = ep["actions"].shape[0]

            # Normalize on the fly
            ep["actions"] = _normalize_bounds_q99(ep["actions"], self._q01_action, self._q99_action)
            ep["proprio"] = _normalize_bounds_q99(ep["proprio"], self._q01_proprio, self._q99_proprio)

            for t in range(T):
                action_chunk = _chunk_actions_numpy(ep["actions"], t, NUM_ACTIONS_CHUNK)

                yield {
                    "image": ep["images"][t],
                    "wrist_image": ep["wrist_images"][t],
                    "proprio": ep["proprio"][t],
                    "action": action_chunk,
                    "language_instruction": ep["language_instruction"],
                    "is_first": (t == 0),
                    "is_last": (t == T - 1),
                    "dataset_name": f"mikasa_{env_name}",
                }

            # Episode data goes out of scope here -> GC can reclaim RAM

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Yields individual step dicts in round-robin order across batch_size streams.

        DataLoader with batch_size=B will group consecutive B yields into one batch,
        so batch[i] always corresponds to stream[i], preserving temporal coherence.

        In DDP each rank gets a different seed offset so that ranks sample different
        episodes and the effective batch diversity equals batch_size * world_size.
        """
        rank = 0
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                rank = dist.get_rank()
        except Exception:
            pass

        master_rng = np.random.default_rng(self.seed + rank)
        streams = [
            self._make_stream(np.random.default_rng(master_rng.integers(2**63)))
            for _ in range(self.batch_size)
        ]

        while True:
            for stream in streams:
                yield self.batch_transform(next(stream))

    def __len__(self) -> int:
        return self._dataset_length
