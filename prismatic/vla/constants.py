"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import sys
from enum import Enum

# Llama 2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

MIKASA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


# Function to detect robot platform from command line arguments
def detect_robot_platform():
    """Detect robot platform from CLI args.

    Priority order:
      1. Explicit episodic-loader flags (`--use_libero_episodic True` or
         `--use_mikasa_episodic True`). These are the most reliable signal
         because they are set by the user with intent, and survive even when
         path substrings (e.g. checkpoint dirs named ``my_checkpoints/mikasa-robo``)
         would otherwise mislead the substring fallback.
      2. Explicit named-mixture / suite flags (`--libero_suite_names`,
         `--mikasa_env_names`).
      3. Substring fallback on the full argv (``--data_root_dir``,
         ``--dataset_name``, etc.).

    Default to LIBERO when nothing matches.
    """
    cmd_args = " ".join(sys.argv).lower()

    def _flag_is_true(name: str) -> bool:
        # Match `--<name> true` and `--<name>=true`. argv is space-joined, so
        # both forms collapse into the same haystack.
        return f"--{name} true" in cmd_args or f"--{name}=true" in cmd_args

    def _flag_is_present(name: str) -> bool:
        return f"--{name} " in cmd_args or f"--{name}=" in cmd_args

    # 1) Explicit episodic-loader flags take absolute priority.
    if _flag_is_true("use_libero_episodic"):
        return "LIBERO"
    if _flag_is_true("use_mikasa_episodic"):
        return "MIKASA"

    # 2) Explicit suite / env-name flags imply the corresponding platform
    #    even if the user did not (yet) set --use_*_episodic True.
    if _flag_is_present("libero_suite_names"):
        return "LIBERO"
    if _flag_is_present("mikasa_env_names"):
        return "MIKASA"

    # 3) Fallback: substring search across argv.
    if "mikasa" in cmd_args:
        return "MIKASA"
    elif "libero" in cmd_args:
        return "LIBERO"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    else:
        # Default to LIBERO if unclear
        return "LIBERO"


# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()

# Set the appropriate constants based on the detected platform
if ROBOT_PLATFORM == "MIKASA":
    constants = MIKASA_CONSTANTS
elif ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS

# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print("If needed, manually set the correct constants in `prismatic/vla/constants.py`!")
