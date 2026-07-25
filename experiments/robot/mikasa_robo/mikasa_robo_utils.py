"""Utils for evaluating policies in MIKASA-Robo simulation environments."""

import os
import time
import warnings
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

# Width of the MIKASA proprio vector: xyz(3) + rpy(3) + gripper(1). Structural, so it
# is stated here rather than read from prismatic.vla.constants.PROPRIO_DIM, which is
# resolved from sys.argv and would be wrong in any process that is not an eval run.
MIKASA_PROPRIO_DIM = 7


# ── Task registry ────────────────────────────────────────────────────────────
# Mirrors get_mikasa_robo_datasets.py:env_info() so that the eval wrapper chain
# is identical to the data-collection chain. Each set drives:
#   • whether CurriculumPhaseNoopActionWrapper is added (any cue-phase task)
#   • which task-specific render-info wrapper is added (oracle/cup overlay on video)
SHELL_GAME_CORE_ENVS = frozenset({
    "ShellGameTouch-VLA-v0",
    "ShellGamePush-VLA-v0",
    "ShellGamePick-VLA-v0",
})

INTERCEPT_ENVS = frozenset({
    "InterceptSlow-VLA-v0",
    "InterceptMedium-VLA-v0",
    "InterceptFast-VLA-v0",
    "InterceptGrabSlow-VLA-v0",
    "InterceptGrabMedium-VLA-v0",
    "InterceptGrabFast-VLA-v0",
})

ROTATE_ENVS = frozenset({
    "RotateLenientPos-VLA-v0",
    "RotateLenientPosNeg-VLA-v0",
    "RotateStrictPos-VLA-v0",
    "RotateStrictPosNeg-VLA-v0",
})

COLOR_MEMORY_ENVS = frozenset({
    "RememberColor3-VLA-v0",
    "RememberColor5-VLA-v0",
    "RememberColor9-VLA-v0",
    "FindImposterColor3-VLA-v0",
    "FindImposterColor5-VLA-v0",
    "FindImposterColor9-VLA-v0",
})

SHAPE_MEMORY_ENVS = frozenset({
    "RememberShape3-VLA-v0",
    "RememberShape5-VLA-v0",
    "RememberShape9-VLA-v0",
    "FindImposterShape3-VLA-v0",
    "FindImposterShape5-VLA-v0",
    "FindImposterShape9-VLA-v0",
})

SHAPE_COLOR_MEMORY_ENVS = frozenset({
    "RememberShapeAndColor3x2-VLA-v0",
    "RememberShapeAndColor3x3-VLA-v0",
    "RememberShapeAndColor5x3-VLA-v0",
    "FindImposterShapeAndColor3x2-VLA-v0",
    "FindImposterShapeAndColor3x3-VLA-v0",
    "FindImposterShapeAndColor5x3-VLA-v0",
})

MEMORY_CAPACITY_ENVS = frozenset({
    "BunchOfColors3-VLA-v0",
    "BunchOfColors5-VLA-v0",
    "BunchOfColors7-VLA-v0",
    "SeqOfColors3-VLA-v0",
    "SeqOfColors5-VLA-v0",
    "SeqOfColors7-VLA-v0",
    "ChainOfColors3-VLA-v0",
    "ChainOfColors5-VLA-v0",
    "ChainOfColors7-VLA-v0",
})

SHELL_GAME_SHUFFLE_ENVS = frozenset({
    "ShellGameShuffleTouch-VLA-v0",
})

SHELL_GAME_LAMP_ENVS = frozenset({
    "ShellGameShuffleColorLampTouch-VLA-v0",
    "ShellGameColorLampTouch-VLA-v0",
})

# Tasks whose data was collected with CurriculumPhaseNoopActionWrapper
# (cue + empty phase actions are silently zeroed during collection).
CUE_PHASE_TASKS = frozenset(
    SHELL_GAME_CORE_ENVS
    | COLOR_MEMORY_ENVS
    | SHAPE_MEMORY_ENVS
    | SHAPE_COLOR_MEMORY_ENVS
    | MEMORY_CAPACITY_ENVS
    | SHELL_GAME_SHUFFLE_ENVS
    | SHELL_GAME_LAMP_ENVS
)


# ── Canonical 32-env benchmark for mu-VLA validation ─────────────────────────
# Source: MIKASA-Robo
# `mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets.py`.
# Order is the user's id 0..31, with the three ShellGame variants
# (Touch / Push / Pick) grouped at the top to match
# `SHELL_GAME_CORE_ENVS`. `max_episode_steps` is the value embedded in
# `@register_env(...)` for each env in `mikasa_robo_suite/vla/memory_envs/*.py`,
# and matches what `gym.spec(env_id).max_episode_steps` returns at runtime.
# `make_eval_env` auto-detects this value when `max_episode_steps=None`.
MIKASA_ROBO_32_ENVS = (
    ("ShellGameTouch-VLA-v0", 30),
    ("ShellGamePush-VLA-v0", 30),
    ("ShellGamePick-VLA-v0", 30),
    ("InterceptSlow-VLA-v0", 60),
    ("InterceptMedium-VLA-v0", 60),
    ("InterceptFast-VLA-v0", 60),
    ("InterceptGrabSlow-VLA-v0", 60),
    ("InterceptGrabMedium-VLA-v0", 60),
    ("InterceptGrabFast-VLA-v0", 60),
    ("RotateLenientPos-VLA-v0", 60),
    ("RotateLenientPosNeg-VLA-v0", 60),
    ("RotateStrictPos-VLA-v0", 90),
    ("RotateStrictPosNeg-VLA-v0", 90),
    ("TakeItBack-VLA-v0", 60),
    ("RememberColor3-VLA-v0", 25),
    ("RememberColor5-VLA-v0", 25),
    ("RememberColor9-VLA-v0", 25),
    ("RememberShape3-VLA-v0", 25),
    ("RememberShape5-VLA-v0", 25),
    ("RememberShape9-VLA-v0", 25),
    ("RememberShapeAndColor3x2-VLA-v0", 25),
    ("RememberShapeAndColor3x3-VLA-v0", 25),
    ("RememberShapeAndColor5x3-VLA-v0", 25),
    ("BunchOfColors3-VLA-v0", 400),
    ("BunchOfColors5-VLA-v0", 400),
    ("BunchOfColors7-VLA-v0", 400),
    ("SeqOfColors3-VLA-v0", 400),
    ("SeqOfColors5-VLA-v0", 400),
    ("SeqOfColors7-VLA-v0", 400),
    ("ChainOfColors3-VLA-v0", 400),
    ("ChainOfColors5-VLA-v0", 400),
    ("ChainOfColors7-VLA-v0", 400),
)
MIKASA_ROBO_32_ENV_IDS = tuple(eid for eid, _ in MIKASA_ROBO_32_ENVS)


def get_eval_wrapper_chain(env_id: str):
    """Return the ordered list of (wrapper_label, applies_to_obs) tuples
    that `make_eval_env` will install for `env_id`.

    Inner-to-outer ordering. Matches the dataset-collection chain in
    `mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets.py:env_info`
    plus the eval-only `RecordEpisode` + observation flatteners. The labels
    are documentation-only — the actual install happens in `make_eval_env`.

    `applies_to_obs=True` wrappers can change what the policy sees;
    `applies_to_obs=False` wrappers only modify the recorded scene video
    or `info` dict.
    """
    obs_changing = {
        "StateOnlyTensorToDictWrapper",
        "CurriculumPhaseNoopActionWrapper",
        "FlattenRGBDObservationWrapper(rgb=True, joints=True)",
        "ConvertJointsToEEFXyzRpyGripperWrapper",
    }
    inner = ["StateOnlyTensorToDictWrapper", *_get_dataset_wrapper_order(env_id)]
    outer = [
        "RecordEpisode",
        "FlattenRGBDObservationWrapper(rgb=True, joints=True)",
        "ConvertJointsToEEFXyzRpyGripperWrapper",
    ]
    return [(name, name in obs_changing) for name in inner + outer]


def _get_render_info_wrapper_cls(env_id: str):
    """Return the task-specific render-info wrapper class (or None) for env_id.

    These wrappers only override env.render() to draw oracle/cup/target overlays
    on the recorded scene video — they DO NOT modify obs["rgb"] consumed by the
    policy. Including them keeps eval videos visually identical to the videos
    produced during dataset collection.
    """
    from mikasa_robo_suite.vla.utils.wrappers import (
        MemoryCapacityInfoWrapper,
        RememberColorInfoWrapper,
        RememberShapeAndColorInfoWrapper,
        RememberShapeInfoWrapper,
        RotateRenderAngleInfoWrapper,
        ShellGameRenderCupInfoWrapper,
    )

    if env_id in SHELL_GAME_CORE_ENVS or env_id in SHELL_GAME_SHUFFLE_ENVS or env_id in SHELL_GAME_LAMP_ENVS:
        return ShellGameRenderCupInfoWrapper
    if env_id in COLOR_MEMORY_ENVS:
        return RememberColorInfoWrapper
    if env_id in SHAPE_MEMORY_ENVS:
        return RememberShapeInfoWrapper
    if env_id in SHAPE_COLOR_MEMORY_ENVS:
        return RememberShapeAndColorInfoWrapper
    if env_id in MEMORY_CAPACITY_ENVS:
        return MemoryCapacityInfoWrapper
    if env_id in ROTATE_ENVS:
        return RotateRenderAngleInfoWrapper
    return None


# ── Custom wrapper (copied from MIKASA-Robo baselines/ppo/ppo_memtasks.py) ──
# The stock mani_skill FlattenRGBDObservationWrapper does NOT support
# oracle= / joints= parameters that are required to reproduce the exact
# observation format used during data collection.  We vendor the wrapper
# here to avoid pulling in the entire baselines module and its heavy deps.
class FlattenRGBDObservationWrapper:
    """Lazy-loaded wrapper — see _FlattenRGBDObservationWrapper for the real impl."""

    _cls = None

    def __new__(cls, env, **kwargs):
        if cls._cls is None:
            import gymnasium as gym
            from mani_skill.envs.sapien_env import BaseEnv
            from mani_skill.utils import common
            from mikasa_robo_suite.vla.utils.wrappers import StateOnlyTensorToDictWrapper

            class _FlattenRGBDObservationWrapper(gym.ObservationWrapper):
                """Flattens rgbd observations into a dict with rgb/proprio keys.

                Reproduced from MIKASA-Robo baselines/ppo/ppo_memtasks.py to match
                the exact observation format used during dataset collection.
                """

                def __init__(self, env, rgb=True, depth=True, state=True, oracle=False, joints=False):
                    self.base_env: BaseEnv = StateOnlyTensorToDictWrapper(env.unwrapped)
                    super().__init__(env)
                    self.include_rgb = rgb
                    self.include_depth = depth
                    self.include_state = state
                    self.include_oracle = oracle
                    self.include_joints = joints

                    sample_obs, _ = env.reset()
                    new_obs = self.observation(sample_obs)
                    self.base_env.update_obs_space(new_obs)

                def observation(self, observation: Dict):
                    ret = dict()

                    # MIKASA-Robo renamed this observation key from `prompt` to
                    # `task_cue` in v1.0.0. Normalize to `prompt` so the rest of
                    # this wrapper works against either version. Pop rather than
                    # alias: leaving `task_cue` in the dict would flatten it into
                    # the state vector below and silently change its dimension.
                    if "task_cue" in observation and "prompt" not in observation:
                        observation["prompt"] = observation.pop("task_cue")

                    if self.include_rgb or self.include_depth:
                        ret["oracle_info"] = observation["oracle_info"]
                        ret["prompt"] = observation["prompt"]
                        sensor_data = observation.pop("sensor_data")

                        del observation["sensor_param"]
                        images = []
                        for cam_data in sensor_data.values():
                            if self.include_rgb:
                                images.append(cam_data["rgb"])
                            if self.include_depth:
                                images.append(cam_data["depth"])

                        if len(images) > 0:
                            images = torch.concat(images, axis=-1)

                    if self.include_state and not (self.include_rgb or self.include_depth):
                        if not self.include_oracle:
                            observation.pop("oracle_info")
                    else:
                        if not self.include_joints:
                            filtered_obs = {k: v for k, v in observation.items() if k not in ["prompt", "oracle_info"]}
                        else:
                            extra_agent = {}
                            for key in ["extra", "agent"]:
                                if key in observation:
                                    extra_agent[key] = observation.pop(key)

                            extra_agent_flat = common.flatten_state_dict(extra_agent, use_torch=True, device=self.base_env.device)
                            # MIKASA-Robo renamed this output key from `joints` to
                            # `proprio` in v1.0.0 (the `joints=` *parameter* kept its
                            # name). `ConvertJointsToEEFXyzRpyGripperWrapper`, applied
                            # next in make_eval_env, reads `proprio` and silently
                            # returns the observation untouched if it is absent — so
                            # emitting `joints` here leaves the raw flattened vector
                            # in place instead of the 7-D eef pose.
                            ret["proprio"] = extra_agent_flat

                            filtered_obs = {k: v for k, v in observation.items() if k not in ["prompt", "oracle_info", "extra"]}

                        observation = common.flatten_state_dict(filtered_obs, use_torch=True, device=self.base_env.device)

                    if self.include_state and not (self.include_rgb or self.include_depth):
                        ret = observation
                    else:
                        ret["state"] = observation
                    if self.include_rgb and not self.include_depth:
                        ret["rgb"] = images
                    elif self.include_rgb and self.include_depth:
                        ret["rgbd"] = images
                    elif self.include_depth and not self.include_rgb:
                        ret["depth"] = images

                    if "state" in ret.keys() and not self.include_state:
                        ret.pop("state")

                    if "oracle_info" in ret.keys() and not self.include_oracle and ret["oracle_info"] is not None:
                        ret.pop("oracle_info")

                    if "oracle_info" in ret.keys() and (ret["oracle_info"] == 4242424242).any().item():
                        ret.pop("oracle_info")

                    if "prompt" in ret.keys() and (ret["prompt"] == 4242424242).any().item():
                        ret.pop("prompt")

                    if "proprio" in ret.keys() and not self.include_joints:
                        ret.pop("proprio")

                    return ret

            cls._cls = _FlattenRGBDObservationWrapper
        return cls._cls(env, **kwargs)


def _get_dataset_wrapper_order(env_id: str):
    """Return the ordered list of wrapper-class names that the dataset
    collector applies for `env_id`. Mirrors `env_info()` in
    `mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets.py`.

    The order matters for `cv2.putText` reliability: the *innermost* info
    wrapper that actually overlays text must either (a) do its own
    tensor→uint8 conversion, or (b) sit above a wrapper that already produced
    a uint8 numpy frame. `RotateRenderAngleInfoWrapper` does NOT do (a) and
    therefore must be placed AFTER `RenderRewardInfoWrapper` to satisfy (b).
    """
    info_wrapper_cls = _get_render_info_wrapper_cls(env_id)
    info_name = info_wrapper_cls.__name__ if info_wrapper_cls is not None else None
    cue = env_id in CUE_PHASE_TASKS

    # ── ShellGame core / shuffle / lamp ───────────────────────────────
    # dataset: [cue], RenderStep, ShellGameRenderCup, RenderReward, DebugReward
    if info_name == "ShellGameRenderCupInfoWrapper":
        chain = []
        if cue:
            chain.append("CurriculumPhaseNoopActionWrapper")
        chain += ["RenderStepInfoWrapper", info_name, "RenderRewardInfoWrapper"]
        chain.append("DebugRewardWrapper")
        return chain

    # ── Rotate (lenient + strict) ─────────────────────────────────────
    # dataset: RenderStep, RenderReward, RotateRenderAngle, DebugReward
    # NB. RotateRenderAngleInfoWrapper has no tensor→uint8 conversion of
    # its own; placing it innermost (as we did before) crashes cv2.putText.
    if info_name == "RotateRenderAngleInfoWrapper":
        chain = []
        if cue:
            chain.append("CurriculumPhaseNoopActionWrapper")
        chain += ["RenderStepInfoWrapper", "RenderRewardInfoWrapper", info_name]
        chain.append("DebugRewardWrapper")
        return chain

    # ── Color / Shape / ShapeColor / MemoryCapacity memory tasks ──────
    # dataset: cue, InfoWrapper, RenderStep, RenderReward, DebugReward
    if info_name in {
        "RememberColorInfoWrapper",
        "RememberShapeInfoWrapper",
        "RememberShapeAndColorInfoWrapper",
        "MemoryCapacityInfoWrapper",
    }:
        chain = []
        if cue:
            chain.append("CurriculumPhaseNoopActionWrapper")
        chain += [info_name, "RenderStepInfoWrapper", "RenderRewardInfoWrapper"]
        chain.append("DebugRewardWrapper")
        return chain

    # ── Intercept / TakeItBack (no info wrapper) ──────────────────────
    # dataset: [cue], RenderStep, RenderReward, DebugReward
    chain = []
    if cue:
        chain.append("CurriculumPhaseNoopActionWrapper")
    chain += ["RenderStepInfoWrapper", "RenderRewardInfoWrapper"]
    chain.append("DebugRewardWrapper")
    return chain


def make_eval_env(env_id: str, num_envs: int = 1, max_episode_steps: Optional[int] = None,
                  video_dir: Optional[str] = None):
    """
    Create MIKASA-Robo evaluation environment with the same wrapper stack
    used during dataset collection.

    Wrapper order (mirrors get_mikasa_robo_datasets.py:env_info() exactly):
        1. gym.make(env_id, obs_mode="rgb")
        2. StateOnlyTensorToDictWrapper
        3. <env-specific chain — see _get_dataset_wrapper_order>
        4. FlattenRGBDObservationWrapper(rgb=True, joints=True)
        5. ConvertJointsToEEFXyzRpyGripperWrapper

    The env-specific chain (step 3) varies because the dataset collector
    places different InfoWrapper variants in different positions; we mirror
    that exactly. In particular, `RotateRenderAngleInfoWrapper` MUST sit
    above `RenderRewardInfoWrapper` (it relies on the latter to produce a
    uint8 numpy frame, and crashes otherwise on raw torch tensors).

    Note: scene-render recording is no longer done by `RecordEpisode` here —
    the eval loop calls `env.render()` itself once per simulator step and
    collects the frames in-memory. This sidesteps three issues with
    RecordEpisode's async buffering: (a) the buffer for episode N only
    flushes on `reset()` of episode N+1 (the LAST episode therefore never
    landed on disk), (b) the saved file used the `mp4v` FOURCC which VS
    Code's HTML5 player cannot decode, and (c) reading frames back from
    disk to compose was wasteful.

    Argument `video_dir` is kept for backward compatibility but ignored.

    Returns (wrapped_env, episode_timeout).
    """
    del video_dir  # backward-compat shim; see docstring
    import gymnasium as gym

    import mani_skill.envs  # noqa: F401 — registers ManiSkill envs
    import mikasa_robo_suite.vla.memory_envs  # noqa: F401 — registers MIKASA envs
    from mikasa_robo_suite.vla.utils import wrappers as mr_wrappers
    from mikasa_robo_suite.vla.utils.wrappers import (
        ConvertJointsToEEFXyzRpyGripperWrapper,
        StateOnlyTensorToDictWrapper,
    )

    warnings.filterwarnings("ignore", message=".*env\\.\\w+ to get variables from other wrappers is deprecated.*")

    # Auto-detect episode length from the registered env spec
    episode_timeout = max_episode_steps or gym.spec(env_id).max_episode_steps

    env = gym.make(
        env_id,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        num_envs=num_envs,
        max_episode_steps=episode_timeout,
        render_mode="rgb_array",  # pure 3D scene; camera feeds come from obs_mode="rgb"
        sim_backend="gpu",
        reward_mode="normalized_dense",
    )

    env = StateOnlyTensorToDictWrapper(env)

    # Apply env-class-specific wrapper chain, mirroring the dataset collector
    # exactly (in particular, position of RotateRenderAngleInfoWrapper above
    # RenderRewardInfoWrapper — see _get_dataset_wrapper_order docstring).
    for wrapper_name in _get_dataset_wrapper_order(env_id):
        wrapper_cls = getattr(mr_wrappers, wrapper_name)
        env = wrapper_cls(env)

    env = FlattenRGBDObservationWrapper(
        env, rgb=True, depth=False, state=False, oracle=False, joints=True
    )
    env = ConvertJointsToEEFXyzRpyGripperWrapper(env)

    return env, episode_timeout


def render_scene_frame(env) -> np.ndarray:
    """Return the current scene render (with all overlay wrappers applied)
    as a uint8 RGB HxWx3 numpy array. Drops the leading num_envs dim — eval
    runs with num_envs=1.
    """
    frame = env.render()
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    arr = np.asarray(frame)
    if arr.ndim == 4:  # (num_envs, H, W, 3)
        arr = arr[0]
    return _to_uint8_rgb(arr)


def get_language_instruction(env, info=None):
    """
    Extract language instruction for the current episode.

    The instruction is available in the info dict returned by reset()/step(),
    under the key "language_instruction".
    """
    if info is not None and isinstance(info, dict) and "language_instruction" in info:
        lang = info["language_instruction"]
        if isinstance(lang, (list, np.ndarray)):
            return str(lang[0])
        return str(lang)

    # Fallback: try env attributes
    try:
        return env.unwrapped.get_language_instruction()
    except AttributeError:
        pass
    try:
        return env.unwrapped.prompt_str
    except AttributeError:
        pass

    return ""


def get_mikasa_images(obs):
    """
    Split 6-channel RGB observation into base and wrist camera images.

    obs["rgb"] has shape (num_envs, 128, 128, 6) — two cameras concatenated
    along the channel dimension.

    Returns:
        base_image:  np.ndarray of shape (128, 128, 3)  — base camera (channels 0..2)
        wrist_image: np.ndarray of shape (128, 128, 3)  — wrist camera (channels 3..5)
    """
    rgb_raw = obs["rgb"]
    if torch.is_tensor(rgb_raw):
        rgb_raw = rgb_raw.cpu().numpy()
    # Take first env (num_envs=1 for eval)
    base_image = rgb_raw[0, ..., :3]
    wrist_image = rgb_raw[0, ..., 3:]
    return base_image, wrist_image


def get_mikasa_proprio(obs):
    """
    Extract 7D proprioception from observation.

    obs["proprio"] has shape (num_envs, 7): [x, y, z, roll, pitch, yaw, gripper],
    produced by ConvertJointsToEEFXyzRpyGripperWrapper.

    Returns:
        np.ndarray of shape (7,)

    Raises:
        RuntimeError: if the vector is not 7-D, which means the conversion wrapper
            did not run. It no-ops when its input key is missing, so the failure is
            otherwise silent: the policy would be fed the raw flattened joint state.
    """
    proprio = obs["proprio"]
    if torch.is_tensor(proprio):
        proprio = proprio.cpu().numpy()
    if proprio.shape[-1] != MIKASA_PROPRIO_DIM:
        raise RuntimeError(
            f"expected {MIKASA_PROPRIO_DIM}-D proprio, got {proprio.shape[-1]}-D. "
            "ConvertJointsToEEFXyzRpyGripperWrapper did not convert the observation; "
            "check that the wrapper chain in make_eval_env matches the installed "
            "MIKASA-Robo version."
        )
    return proprio[0]


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert RGB image (float/uint8, torch/numpy) to contiguous uint8 RGB."""
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    x = np.asarray(arr)
    if x.dtype == np.uint8:
        return np.ascontiguousarray(x)
    x = x.astype(np.float32, copy=False)
    max_val = float(np.nanmax(x)) if x.size > 0 else 0.0
    if max_val <= 1.0 + 1e-5:
        x = x * 255.0
    return np.clip(x, 0.0, 255.0).astype(np.uint8, copy=False)


def compose_eval_video(
    general_frames: Sequence[np.ndarray],
    top_frames: Sequence[np.ndarray],
    wrist_frames: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """Compose frames: [ general render | top (upper-right) / wrist (lower-right) ].

    Camera images (128x128) are scaled so that two stacked vertically
    exactly match the height of the scene render, preserving aspect ratio.
    """
    n = min(len(general_frames), len(top_frames), len(wrist_frames))
    if n <= 0:
        return []

    first = _to_uint8_rgb(general_frames[0])
    g_h, g_w = first.shape[:2]

    # Each camera panel gets exactly half the scene height.
    # Width is derived from the camera's original aspect ratio so it
    # doesn't look squashed.
    first_cam = _to_uint8_rgb(top_frames[0])
    cam_h_orig, cam_w_orig = first_cam.shape[:2]

    top_h = g_h // 2
    wrist_h = g_h - top_h
    # Scale width proportionally to keep the camera's aspect ratio
    side_w = max(1, int(cam_w_orig * top_h / cam_h_orig))

    out: List[np.ndarray] = []
    for idx in range(n):
        g = _to_uint8_rgb(general_frames[idx])
        if g.shape[:2] != (g_h, g_w):
            g = cv2.resize(g, (g_w, g_h), interpolation=cv2.INTER_AREA)

        top = cv2.resize(_to_uint8_rgb(top_frames[idx]), (side_w, top_h), interpolation=cv2.INTER_AREA)
        wrist = cv2.resize(_to_uint8_rgb(wrist_frames[idx]), (side_w, wrist_h), interpolation=cv2.INTER_AREA)

        cv2.putText(top, "top", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(wrist, "wrist", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        right = np.concatenate([top, wrist], axis=0)
        composed = np.concatenate([g, right], axis=1)
        out.append(np.ascontiguousarray(composed))
    return out


def save_composed_video(
    scene_frames: Sequence[np.ndarray],
    top_frames: Sequence[np.ndarray],
    wrist_frames: Sequence[np.ndarray],
    output_path: str,
    fps: float = 30.0,
    log_file=None,
):
    """Compose [scene | top / wrist] and write to MP4 via imageio + FFmpeg
    using the H.264 (`libx264`) codec with `yuv420p` pixel format.

    The previous writer used `cv2.VideoWriter(..., FOURCC='mp4v')` which
    produces MPEG-4 part 2 streams that VS Code's HTML5 video player
    cannot decode (file opens with "an error occurred while loading the
    video file"). H.264 with yuv420p is the lowest-common-denominator
    format that plays everywhere a browser-derived player runs.

    `scene_frames` is the in-memory list of scene-render frames collected
    by the eval loop (one `env.render()` per simulator step). The previous
    interface read them back from a `<scene_video_dir>/<i>.mp4` written by
    the `RecordEpisode` wrapper, which had two problems: the wrapper
    flushed buffers asynchronously on the next `reset()` (so the LAST
    episode's mp4 never landed on disk before our save call ran), and
    reading back from disk to compose was wasted I/O.
    """
    composed = compose_eval_video(scene_frames, top_frames, wrist_frames)
    if not composed:
        msg = f"WARNING: No frames to compose for {output_path}"
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # libx264 + yuv420p require even width and height. Pad if necessary.
    h, w = composed[0].shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        composed = [
            np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            for f in composed
        ]

    arr = np.stack(composed, axis=0)  # (N, H, W, 3) uint8 RGB

    # imageio.v3 picks the FFmpeg backend automatically for *.mp4 outputs.
    import imageio.v3 as iio

    iio.imwrite(
        output_path,
        arr,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
    )

    msg = f"Saved composed video ({len(composed)} frames) to {output_path}"
    if log_file:
        log_file.write(msg + "\n")
        log_file.flush()
