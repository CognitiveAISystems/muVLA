"""
libero_episodic_dataset.py

Episodic dataset for multi-task LIBERO training that mirrors the streaming
logic of ``MIKASARoboVLAEpisodicDataset`` (POMDP-aware episode streams), but
uses the LIBERO RLDS data format (``image``/``wrist_image`` 256x256x3, 8-D
``state`` proprio, 7-D action with the gripper dim inverted post-load).

Episodes are loaded lazily one-at-a-time from TFDS (not all into RAM).
Normalization statistics are computed in a streaming pass (action+proprio only,
no images) and cached to disk. Action gripper is inverted before stats are
computed, matching the on-the-fly behavior of ``libero_dataset_transform``
in the standard OXE pipeline.

The collator and batch transform mirror their MIKASA counterparts so that
``finetune.py`` can plug the LIBERO episodic dataset in via the same code path
(``is_first`` / ``is_last`` boundary flags, identical step dict layout).
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
    """Resolve versioned subdirectory (e.g., libero_spatial_no_noops/1.0.0/)."""
    version_dirs = sorted([d for d in env_dir.iterdir() if d.is_dir()])
    if version_dirs:
        return version_dirs[-1]
    return env_dir


def _libero_action_transform(actions: np.ndarray) -> np.ndarray:
    """
    Apply the LIBERO standardization to actions: clip the last (gripper) dim
    to ``[0, 1]`` and invert (so ``+1 = open, 0 = close``). Mirrors
    ``libero_dataset_transform`` in
    ``prismatic/vla/datasets/rlds/oxe/transforms.py``.
    """
    actions = actions.astype(np.float32, copy=True)
    gripper = np.clip(actions[:, -1:], 0.0, 1.0)
    actions[:, -1:] = 1.0 - gripper
    return actions


def _load_episodes_from_rlds(data_dir: str, suite_name: str) -> List[Dict[str, Any]]:
    """
    Load all episodes from a LIBERO RLDS dataset directory into RAM.

    Used by tests and by streaming-statistics fallbacks. Production training
    relies on ``_make_suite_iterator`` to stream episodes lazily.
    """
    data_path = _resolve_versioned_dir(Path(data_dir))
    builder = tfds.builder_from_directory(str(data_path))
    dataset = builder.as_dataset(split="train")

    episodes = []
    for episode in dataset:
        ep = _load_single_episode_from_tf(episode)
        if ep is None:
            continue
        ep["suite_name"] = suite_name
        episodes.append(ep)

    return episodes


def _load_single_episode_from_tf(tf_episode) -> Optional[Dict[str, Any]]:
    """
    Extract a single LIBERO episode from a TF dataset element into numpy arrays.

    Applies the LIBERO action gripper inversion at load time so that statistics
    and on-the-fly normalization see the standardized action distribution.
    """
    steps = list(tf_episode["steps"])
    T = len(steps)
    if T == 0:
        return None

    images = np.stack([step["observation"]["image"].numpy() for step in steps])
    wrist_images = np.stack([step["observation"]["wrist_image"].numpy() for step in steps])
    # LIBERO proprio = full 8-D state (6D EEF pose + 2D gripper).
    proprio = np.stack([step["observation"]["state"].numpy() for step in steps]).astype(np.float32)
    actions = np.stack([step["action"].numpy() for step in steps]).astype(np.float32)
    actions = _libero_action_transform(actions)

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
) -> Dict[str, Any]:
    """Compute combined normalization statistics across all episodes (in-memory)."""
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
    suite_dirs: Dict[str, Path],
) -> Dict[str, Any]:
    """
    Compute normalization statistics by streaming through TFDS datasets.
    Only loads action + proprio per episode (no images).
    """
    all_actions: List[np.ndarray] = []
    all_proprio: List[np.ndarray] = []
    num_trajectories = 0

    for suite_name, suite_dir in sorted(suite_dirs.items()):
        versioned_dir = _resolve_versioned_dir(suite_dir)
        builder = tfds.builder_from_directory(str(versioned_dir))
        dataset = builder.as_dataset(split="train")

        for episode in dataset:
            steps = list(episode["steps"])
            if len(steps) == 0:
                continue
            actions = np.stack([s["action"].numpy() for s in steps]).astype(np.float32)
            actions = _libero_action_transform(actions)
            proprio = np.stack([s["observation"]["state"].numpy() for s in steps]).astype(np.float32)
            all_actions.append(actions)
            all_proprio.append(proprio)
            num_trajectories += 1

    all_actions = np.concatenate(all_actions, axis=0)
    all_proprio = np.concatenate(all_proprio, axis=0)

    return _compute_stats_from_arrays(all_actions, all_proprio, num_trajectories)


def _stats_cache_path(data_root_dir: Path, suite_names: List[str]) -> Path:
    """Deterministic cache path based on sorted suite names."""
    key = ",".join(sorted(suite_names))
    h = hashlib.sha256(key.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return data_root_dir / f"libero_statistics_{h}.json"


def _load_or_compute_statistics(
    data_root_dir: Path,
    suite_names: List[str],
    suite_dirs: Dict[str, Path],
) -> Dict[str, Any]:
    """Load cached statistics from disk, or compute them by streaming."""
    cache_path = _stats_cache_path(data_root_dir, suite_names)

    if cache_path.exists():
        print(f"[LIBEROVLAEpisodicDataset] Loading cached statistics from {cache_path}")
        with open(cache_path, "r") as f:
            stats = json.load(f)
        for group in ["action", "proprio"]:
            for k, v in stats[group].items():
                if isinstance(v, list):
                    stats[group][k] = np.array(v, dtype=np.float32)
        return stats

    print("[LIBEROVLAEpisodicDataset] Computing statistics (streaming, no images)...")
    stats = _compute_statistics_streaming(suite_dirs)

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
    print(f"[LIBEROVLAEpisodicDataset] Cached statistics to {cache_path}")

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
    """Returns action chunk [a_t, ..., a_{t+chunk-1}], padding by repeating last action."""
    T = actions.shape[0]
    chunk = np.zeros((num_actions_chunk, actions.shape[1]), dtype=np.float32)
    for i in range(num_actions_chunk):
        idx = min(timestep + i, T - 1)
        chunk[i] = actions[idx]
    return chunk


# === Batch Transform ===


# Image augmentation config — must exactly match RLDSDataset
# (prismatic/vla/datasets/datasets.py) for parity with the standard pipeline.
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
    """Resize (Lanczos3) then augment, matching the RLDS pipeline order."""
    import tensorflow as tf
    from dlimp.transforms import augment_image, resize_image

    tf_img = tf.constant(img_np)
    tf_img = resize_image(tf_img, resize_size)
    tf_img = augment_image(tf_img, **_IMAGE_AUGMENT_KWARGS)
    return tf_img.numpy()


def _resize_only(img_np: np.ndarray, resize_size: Tuple[int, int]) -> np.ndarray:
    """Resize without augmentation (used when image_aug=False; matches RLDS Lanczos3 path)."""
    import tensorflow as tf
    from dlimp.transforms import resize_image

    tf_img = tf.constant(img_np)
    tf_img = resize_image(tf_img, resize_size)
    return tf_img.numpy()


@dataclass
class LiberoBatchTransform:
    """Converts a single LIBERO step dict to the format expected by the collator/model."""

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
        # LIBERO raw images are 256x256x3 — always resize to resize_resolution to
        # match the RLDS pipeline (decode -> resize Lanczos3 -> [augment]).
        image_np = step["image"]
        wrist_np = step.get("wrist_image")
        if self.image_aug:
            image_np = _apply_image_augmentation(image_np, self.resize_resolution)
            if wrist_np is not None:
                wrist_np = _apply_image_augmentation(wrist_np, self.resize_resolution)
        else:
            image_np = _resize_only(image_np, self.resize_resolution)
            if wrist_np is not None:
                wrist_np = _resize_only(wrist_np, self.resize_resolution)

        img = Image.fromarray(image_np)
        lang = step["language_instruction"].lower()
        action_chunk = step["action"]  # [NUM_ACTIONS_CHUNK, ACTION_DIM]

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

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        pixel_values = self.image_transform(img)

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
class LiberoEpisodicCollator:
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


class LIBEROVLAEpisodicDataset(IterableDataset):
    """
    Episodic dataset for multi-task LIBERO training.

    Each batch position is a persistent stream that yields sequential timesteps
    from complete LIBERO episodes. When one episode ends, the next begins
    immediately from the same stream — possibly from a different task suite.

    Episodes are loaded lazily one-at-a-time from TFDS — only one episode
    per stream is in RAM at any time. Normalization statistics are computed
    via a streaming pass (action + proprio only) and cached to disk.

    Yields samples in round-robin order across batch_size streams, so that
    ``DataLoader(batch_size=B)`` naturally groups them into correct batches
    and batch position ``i`` corresponds to stream ``i``.
    """

    def __init__(
        self,
        data_root_dir: Path,
        suite_names: List[str],
        batch_transform: LiberoBatchTransform,
        resize_resolution: Tuple[int, int],
        batch_size: int = 4,
        seed: int = 42,
    ) -> None:
        self.batch_transform = batch_transform
        self.batch_size = batch_size
        self.seed = seed
        self.resize_resolution = resize_resolution

        data_root_dir = Path(data_root_dir)
        self.suite_names = list(suite_names)
        self.suite_dirs: Dict[str, Path] = {
            suite_name: data_root_dir / suite_name for suite_name in suite_names
        }

        combined_stats = _load_or_compute_statistics(data_root_dir, suite_names, self.suite_dirs)

        # LIBERO gripper dim (last action dim) is binary-like and must NOT be
        # rescaled — mirrors the OXE pipeline's normalization mask behavior.
        # Persist the mask in dataset_statistics so eval-time `_unnormalize_actions`
        # can pass the gripper through untouched (otherwise model output ≈ {0, 1}
        # gets remapped through q01/q99 and the gripper can never close).
        self._action_norm_mask = np.array([True] * 6 + [False], dtype=bool)
        combined_stats["action"]["mask"] = self._action_norm_mask.tolist()

        self.dataset_statistics = {"libero_combined": combined_stats}

        self._q01_action = np.asarray(combined_stats["action"]["q01"], dtype=np.float32)
        self._q99_action = np.asarray(combined_stats["action"]["q99"], dtype=np.float32)
        self._q01_proprio = np.asarray(combined_stats["proprio"]["q01"], dtype=np.float32)
        self._q99_proprio = np.asarray(combined_stats["proprio"]["q99"], dtype=np.float32)

        self._dataset_length = int(combined_stats["num_transitions"])

        print(
            f"[LIBEROVLAEpisodicDataset] {len(suite_names)} suites, "
            f"~{self._dataset_length} steps, streaming mode"
        )

    def _make_suite_iterator(self, suite_name: str, seed: int):
        """Create a shuffled, infinitely-repeating TFDS iterator for one suite."""
        suite_dir = self.suite_dirs[suite_name]
        versioned_dir = _resolve_versioned_dir(suite_dir)
        builder = tfds.builder_from_directory(str(versioned_dir))
        dataset = builder.as_dataset(split="train")
        dataset = dataset.shuffle(buffer_size=200, seed=seed).repeat()
        return iter(dataset)

    def _make_stream(self, rng: np.random.Generator) -> Iterator[Dict[str, Any]]:
        """Infinite generator for one batch position's stream."""
        suite_iters = {
            suite_name: self._make_suite_iterator(suite_name, int(rng.integers(2**31)))
            for suite_name in self.suite_names
        }

        while True:
            suite_name = rng.choice(self.suite_names)
            suite_iter = suite_iters[suite_name]

            tf_episode = next(suite_iter)
            ep = _load_single_episode_from_tf(tf_episode)
            if ep is None:
                continue

            T = ep["actions"].shape[0]

            ep["actions"] = _normalize_bounds_q99(
                ep["actions"], self._q01_action, self._q99_action, mask=self._action_norm_mask
            )
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
                    "dataset_name": f"libero_{suite_name}",
                }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
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
