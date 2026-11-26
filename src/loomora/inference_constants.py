# Copyright (c) Prior Labs GmbH 2026.
# Modified for Loomora V1.0.
# Original notices for adapted implementations are in licenses/.
from __future__ import annotations

from typing import Any, Literal, TypeAlias
import os
import numpy as np
import joblib
from packaging import version

MemorySavingMode: TypeAlias = bool | Literal["auto"] | float | int
TaskType: TypeAlias = Literal["multiclass"]
XType: TypeAlias = Any
YType: TypeAlias = Any
DEFAULT_SAVE_PEAK_MEMORY_FACTOR = 8
MAX_INFERENCE_CHUNK_CELLS = 2048 * 768
MAX_ENSEMBLE_BATCH_CELLS = 768 * 1_000_000

NA_PLACEHOLDER = "__MISSING__"

SKLEARN_16_DECIMAL_PRECISION = 16
PROBABILITY_EPSILON_ROUND_ZERO = 1e-3
AUTOCAST_DTYPE_BYTE_SIZE = 2  # bfloat16
DEFAULT_DTYPE_BYTE_SIZE = 4  # float32

# Otherwise, yoa-johnson double power can end up causing a lot of overflows...
DEFAULT_NUMPY_PREPROCESSING_DTYPE = np.float64

# TODO(eddiebergman): Maybe make these a parameter
MEMORY_SAFETY_FACTOR = 5.0  # Taken as default from function


# TODO(eddiebergman): Pulled from `def get_ensemble_configurations()`
ENSEMBLE_CONFIGURATION_MAX_STEP = 2
MAXIMUM_FEATURE_SHIFT = 1_000
CLASS_SHUFFLE_OVERESTIMATE_FACTOR = 3

FEATURE_IMPORTANCE_MAX_SAMPLES = 100_000

AUTO_FEATURE_SUBSAMPLING_TOP_K = 150
AUTO_FEATURE_SUBSAMPLING_TOP_K_MIN_FEATURES = 200
AUTO_FEATURE_SUBSAMPLING_IMPORTANCE_MIN_SAMPLES = 100_000

# 1) Figure out whether this Joblib version supports "generator_unordered".
# For example, assume "generator_unordered" is officially supported in joblib >= 1.4.0
SUPPORTS_GENERATOR_UNORDERED = version.parse(joblib.__version__) >= version.parse(
    "1.4.0",
)

# scikit-learn has a minimum of "1.2.0", thus we need to work also without return_as.
SUPPORTS_RETURN_AS = version.parse(joblib.__version__) >= version.parse(
    "1.3.0",
)
# 2) Define a mapping from your custom parallel mode to joblib's "return_as" parameter.
if SUPPORTS_GENERATOR_UNORDERED:
    # If the installed Joblib is new enough, allow "generator_unordered"
    PARALLEL_MODE_TO_RETURN_AS = {
        "block": "list",
        "in-order": "generator",
        "as-ready": "generator_unordered",
    }
else:
    # If the installed Joblib is older, fall back to "generator"
    PARALLEL_MODE_TO_RETURN_AS = {
        "block": "list",
        "in-order": "generator",
        # fallback to "generator" instead of "generator_unordered"
        "as-ready": "generator",
    }

ALLOW_CPU_LARGE_DATASET = os.getenv("LOOMORA_ALLOW_CPU_LARGE_DATASET", "0").lower() in {"1", "true", "yes", "on"}
MAX_BATCHED_TEST_ROWS = int(os.getenv("LOOMORA_MAX_BATCHED_TEST_ROWS", "32768"))
if MAX_BATCHED_TEST_ROWS < 0:
    raise ValueError("LOOMORA_MAX_BATCHED_TEST_ROWS must be nonnegative")
