# Copyright (c) Prior Labs GmbH 2026.
# Modified for Loomora V1.0.
# Original notices for adapted implementations are in licenses/.
from __future__ import annotations

import contextlib
import os
import warnings
from collections.abc import Sequence
from typing import Literal
import pandas as pd
import numpy as np
import torch
from .inference_errors import LoomoraValidationError
from .inference_constants import AUTOCAST_DTYPE_BYTE_SIZE, DEFAULT_DTYPE_BYTE_SIZE

MAXINT_RANDOM_SEED = int(np.iinfo(np.int32).max)

def get_autocast_context(
    device: torch.device, *, enabled: bool
) -> contextlib.AbstractContextManager:
    """Returns a torch.autocast context manager.

    Args:
        device: The torch device being used.
        enabled: Whether to enable autocast.

    Returns:
        A context manager for autocasting.
    """
    return torch.autocast(device.type, enabled=enabled)


DevicesSpecification = (
    torch.device | str | Sequence[torch.device | str] | Literal["auto"]
)


def infer_devices(devices: DevicesSpecification) -> tuple[torch.device, ...]:
    """Selects the appropriate PyTorch devices for inference.

    If `device` is "auto" then the devices are selected as follows:
    1. If CUDA is available and not excluded, returns all available "cuda" devices
    2. Otherwise, if MPS is available and not excluded, returns the "mps" device
    3. Otherwise, returns the "cpu" device

    CUDA and MPS can be excluded from the "auto" selection by specifying the
    LOOMORA_EXCLUDE_DEVICES environment variable. This can be either "cuda", "mps", or
    "cuda,mps". This is useful for excluding device classes in CI.

    Args:
        devices: The device specification. One of:
            - "auto": the device will be selected as described above
            - a PyTorch device string like "cuda", "mps", or "cpu": this single device
                will be selected by parsing the string to a torch.device
            - a torch.device: this single device will be selected
            - a list of PyTorch device strings or torch.device: each item will be
                converted to a torch.device, and all of the devices selected

    Returns:
        A tuple of at least one device.
    """
    exclude_devices = {
        d.strip()
        for d in os.getenv("LOOMORA_EXCLUDE_DEVICES", "").split(",")
        if d.strip()
    }

    if devices == "auto":
        if "cuda" not in exclude_devices and torch.cuda.is_available():
            return tuple(
                torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
            )

        if "mps" not in exclude_devices and torch.backends.mps.is_available():
            if _is_torch_mps_supported():
                return (torch.device("mps"),)
            warnings.warn(
                "An MPS device is available, but Loomora disables MPS for "
                "PyTorch < 2.6 (earlier versions can give poor accuracy and "
                "lack bfloat16 autocast support on the MPS backend). Falling "
                "back to CPU. Install torch>=2.6 to use MPS.",
                stacklevel=2,
            )

        return (torch.device("cpu"),)

    if isinstance(devices, (str, torch.device)):
        devices = (devices,)

    devices = tuple(_parse_device(device) for device in devices)

    if len(set(devices)) != len(devices):
        raise ValueError(
            "The list of devices for inference cannot contain the same device more "
            f"than once. It contained: {devices}"
        )

    if any(d.type == "mps" for d in devices):
        if not torch.backends.mps.is_available():
            raise ValueError(
                "The MPS device was selected, but MPS is not available on this system."
            )
        if not _is_torch_mps_supported():
            raise ValueError(
                "The MPS device was selected, "
                "but Loomora requires PyTorch >= 2.6 for MPS. "
                "Upgrade PyTorch, or set device='cpu' instead."
            )

    return devices


def _parse_device(device: str | torch.device) -> torch.device:
    # This is safe because torch.device(torch.device(...)) is a noop.
    # torch.device(device) returns a fairly informative error message if `device` is not
    # a valid device, thus do no extra error reporting.
    device = torch.device(device)

    # In older versions of PyTorch, some torch.cuda functions fail if the device has no
    # index. 0 is implicit if no index is specified, so add it.
    if device == torch.device("cuda"):
        device = torch.device("cuda:0")

    return device


def _is_torch_mps_supported() -> bool:
    """Return True if the MPS device is supported, otherwise False.

    We require PyTorch >= 2.6 for MPS to support all used operations.
    """
    return torch.__version__ >= "2.6"


def is_autocast_available(device_type: str) -> bool:
    """Infer whether autocast is available for the given device type.

    Args:
        device_type: The device type to check for autocast availability.

    Returns:
        Whether autocast is available for the given device type.
    """
    # Try to use PyTorch's built-in function first
    try:
        # Check if the function is available in torch
        if hasattr(torch.amp.autocast_mode, "is_autocast_available"):
            # Use function directly
            torch_is_autocast_available = torch.amp.autocast_mode.is_autocast_available
            return bool(torch_is_autocast_available(device_type))
        # Fall back to custom implementation
        raise AttributeError("is_autocast_available not found")
    except (ImportError, AttributeError):
        # Fall back to custom implementation if the function isn't available
        return bool(
            hasattr(torch.cuda, "amp")
            and hasattr(torch.cuda.amp, "autocast")
            and (
                device_type == torch.device("cuda").type
                or (
                    device_type == torch.device("cpu").type
                    and hasattr(torch.cpu, "amp")
                )
            ),
        )


def _cpu_supports_fast_bf16() -> bool:
    """Whether the CPU accelerates bfloat16 (Intel AMX / AVX512-BF16, AMD Zen 4+).

    Requires a torch build with oneDNN, which provides the fast bf16 kernels
    (absent e.g. on macOS wheels, where CPU bf16 falls back to slow reference
    kernels). AMX CPUs also enumerate AVX512-BF16, so this one check covers
    both instruction sets.
    """
    # bf16 without oneDNN's fast kernels is far slower than float32. Official
    # wheels always ship oneDNN; this guards distro/self-built torch without it.
    if not torch.backends.mkldnn.is_available():
        return False
    # Private torch API with no public equivalent; if a torch release removes
    # it, warn and stay on float32 rather than risk slow emulated bf16.
    avx512_bf16 = getattr(torch.cpu, "_is_avx512_bf16_supported", None)
    if avx512_bf16 is None:
        warnings.warn(
            "torch.cpu._is_avx512_bf16_supported() does not exist in this torch"
            " version, so Loomora cannot detect CPU bf16 support and disables"
            " CPU bf16 autocast. Check whether this PyTorch build exposes the"
            " native CPU bf16 capability API.",
            stacklevel=2,
        )
        return False
    return bool(avx512_bf16())


def infer_autocast_inference_mode(
    devices: Sequence[torch.device], *, enable: bool | None
) -> bool:
    """Infer whether reduced-precision (autocast) inference should be enabled.

    On GPU this is fp16 autocast; on CPU ``torch.autocast`` runs in bfloat16,
    which is enabled only on CPUs with native bf16 support (see
    :func:`_cpu_supports_fast_bf16`).

    Args:
        devices: The devices to validate against.
        enable:
            Whether it should be enabled, `True` or `False`, otherwise if `None`,
            detect if it's possible and use it if so.

    Returns:
        Whether to use autocast inference or not.

    Raises:
        ValueError: If autocast was enabled and any of the selected devices do
            not support it.
    """
    is_cpu = any(device.type.lower() == "cpu" for device in devices)
    if is_cpu:
        # CPU autocast runs in bfloat16, which is only faster than float32 on CPUs
        # with native bf16 support.
        autocast_available = (
            all(device.type.lower() == "cpu" for device in devices)
            and is_autocast_available("cpu")
            and _cpu_supports_fast_bf16()
        )
    else:
        autocast_available = any(
            is_autocast_available(device.type) for device in devices
        )

    if enable is None:
        return autocast_available

    if enable is True:
        if not autocast_available:
            raise ValueError(
                'You specified `inference_precision="autocast"`, however one or'
                f" more of the selected devices ({devices=}) does not support it."
                " On CPU, autocast requires hardware-accelerated bfloat16"
                " (Intel AMX / AVX512-BF16, AMD Zen 4+)."
                '\nSet `inference_precision="auto"` to fall back to full'
                " precision automatically.",
            )
        return True

    if enable is False:
        return False

    raise ValueError(f"Unrecognized argument '{enable}'")


def infer_random_state(
    random_state: int | np.random.RandomState | np.random.Generator | None,
) -> tuple[int, np.random.Generator]:
    """Infer the random state from the given input.

    Args:
        random_state: The random state to infer.

    Returns:
        A static integer seed and a random number generator.
    """
    if isinstance(random_state, (int, np.integer)):
        np_rng = np.random.default_rng(random_state)
        static_seed = int(random_state)
    elif isinstance(random_state, np.random.RandomState):
        static_seed = int(random_state.randint(0, MAXINT_RANDOM_SEED))
        np_rng = np.random.default_rng(static_seed)
    elif isinstance(random_state, np.random.Generator):
        np_rng = random_state
        static_seed = int(np_rng.integers(0, MAXINT_RANDOM_SEED))
    elif random_state is None:
        np_rng = np.random.default_rng()
        static_seed = int(np_rng.integers(0, MAXINT_RANDOM_SEED))
    else:
        raise ValueError(f"Invalid random_state {random_state}")

    return static_seed, np_rng



def determine_precision(
    inference_precision: torch.dtype | Literal["autocast", "auto"],
    devices_: Sequence[torch.device],
) -> tuple[bool, torch.dtype | None, int]:
    """Decide whether to use autocast or a forced precision dtype.

    Args:
        inference_precision:

            - If `"auto"`, decide automatically based on the device.
            - If `"autocast"`, explicitly use PyTorch autocast (mixed precision).
            - If a `torch.dtype`, force that precision.

        devices_: The devices which will be used for inference.

    Returns:
        use_autocast_:
            True if mixed-precision autocast will be used.
        forced_inference_dtype_:
            If not None, the forced precision dtype for the model.
        byte_size:
            The byte size per element for the chosen precision.
    """
    if inference_precision in ["autocast", "auto"]:
        use_autocast_ = infer_autocast_inference_mode(
            devices=devices_,
            enable=True if (inference_precision == "autocast") else None,
        )
        forced_inference_dtype_ = None
        byte_size = (
            AUTOCAST_DTYPE_BYTE_SIZE if use_autocast_ else DEFAULT_DTYPE_BYTE_SIZE
        )
    elif isinstance(inference_precision, torch.dtype):
        use_autocast_ = False
        forced_inference_dtype_ = inference_precision
        byte_size = inference_precision.itemsize
    else:
        raise LoomoraValidationError(
            f"Unknown inference_precision={inference_precision}"
        )

    return use_autocast_, forced_inference_dtype_, byte_size


def create_inference_engine(  # noqa: PLR0913
    *,
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache", "batched"],
    X_train: np.ndarray,
    y_train: np.ndarray,
    ensemble_preprocessor: LoomoraEnsemblePreprocessor,
    models: list[Architecture],
    devices_: Sequence[torch.device],
    byte_size: int,
    forced_inference_dtype_: torch.dtype | None,
    memory_saving_mode: MemorySavingMode,
    use_autocast_: bool,
    task_type: str,
    inference_mode: bool = True,
    keep_cache_on_device: bool = True,
    kv_cache_precision: Literal["auto", "int8", "fp8"] | None = None,
    inference_profile: Literal["memory_save", "speed"] = "memory_save",
    estimator_batch_size: int = 8,
) -> InferenceEngine:
    """Create the appropriate Loomora inference engine based on `fit_mode`.

    Each execution mode will perform slightly different operations based on the mode
    specified by the user. In the case where preprocessors will be fit during
    initialization, we will use them to further transform the associated borders with
    each ensemble config member.

    Args:
        fit_mode: Determines how we prepare inference (pre-cache or not).
        X_train: Training features
        y_train: Training target
        ensemble_preprocessor: The ensemble preprocessor to use.
        models: The loaded Loomora models.
        devices_: The devices for inference.
        byte_size: Byte size for the chosen inference precision.
        forced_inference_dtype_: If not None, the forced dtype for inference.
        memory_saving_mode: GPU/CPU memory saving settings.
        use_autocast_: Whether we use torch.autocast for inference.
        task_type: The task type, e.g. "multiclass" or "regression". Only used
            for ``fit_mode="fit_with_cache"``, where the cache is built during
            initialization and is task-specific.
        inference_mode: Whether to use torch.inference_mode (set False if
            backprop is needed)
        inference_profile: Preserve the original path with ``memory_save`` or
            use estimator batching with OOM retry under ``speed``.
        estimator_batch_size: Starting estimator batch for ``speed``; total
            ensemble size and preprocessing remain unchanged.
        keep_cache_on_device: Only relevant for ``fit_mode="fit_with_cache"``.
            If True (default), each per-estimator KV cache stays on the
            inference device. If False, caches are offloaded to CPU as they
            are built and moved back on demand during inference, lowering
            resident device memory at the cost of per-call transfers.
        kv_cache_precision: Only for ``fit_mode="fit_with_cache"``. Resolved
            against what the architecture supports. ``None`` (default) picks the
            architecture default (``"int8"`` when it can quantize, else
            ``"auto"``); ``"int8"`` quantizes the KV cache to save memory;
            ``"fp8"`` stores it as 8-bit floats (same size, float rounding
            semantics); ``"auto"`` keeps the computed dtype.
    """
    from .inference_engine import InferenceEngineOnDemand, InferenceEngineCachePreprocessing, InferenceEngineExplicitKVCache

    if inference_profile == "speed" and len(devices_) != 1:
        raise ValueError("speed currently uses one device per process; select one GPU")

    if fit_mode == "low_memory":
        return InferenceEngineOnDemand(
            X_train=X_train,
            y_train=y_train,
            ensemble_preprocessor=ensemble_preprocessor,
            models=models,
            devices=devices_,
            dtype_byte_size=byte_size,
            force_inference_dtype=forced_inference_dtype_,
            save_peak_mem=memory_saving_mode,
        )
    if fit_mode == "fit_preprocessors":
        return InferenceEngineCachePreprocessing(
            X_train=X_train,
            y_train=y_train,
            ensemble_preprocessor=ensemble_preprocessor,
            models=models,
            devices=devices_,
            dtype_byte_size=byte_size,
            force_inference_dtype=forced_inference_dtype_,
            save_peak_mem=memory_saving_mode,
            inference_mode=inference_mode,
            inference_profile=inference_profile,
            estimator_batch_size=estimator_batch_size,
        )
    if fit_mode == "fit_with_cache":
        return InferenceEngineExplicitKVCache(
            X_train=X_train,
            y_train=y_train,
            ensemble_preprocessor=ensemble_preprocessor,
            models=models,
            devices=devices_,
            dtype_byte_size=byte_size,
            force_inference_dtype=forced_inference_dtype_,
            save_peak_mem=memory_saving_mode,
            autocast=use_autocast_,
            task_type=task_type,
            keep_cache_on_device=keep_cache_on_device,
            kv_cache_precision=kv_cache_precision,
            inference_profile=inference_profile,
            estimator_batch_size=estimator_batch_size,
        )
    if fit_mode == "batched":
        raise ValueError(
            "InferenceEngineBatchedNoPreprocessing should be initialized directly "
            "rather than through create_inference_engine."
        )

    raise ValueError(f"Invalid fit_mode: {fit_mode}")


def resolve_categorical_features_indices(
    X: XType,
    categorical_features_indices: Sequence[int] | None,
) -> list[int] | None:
    """Validate declared indices and merge pandas `category` column positions.

    A `category` dtype states the same intent as an entry in
    `categorical_features_indices`, so the two are merged here, before any column
    moves. The merged declarations drive date/text expansion and modality
    detection; numeric columns remain subject to the categorical cardinality cap.

    Returns:
        The sorted union of both, or `None` when neither names a column.
    """
    from .inference_validation import validate_categorical_features_indices

    validate_categorical_features_indices(categorical_features_indices)
    declared = set(categorical_features_indices or ())
    if isinstance(X, pd.DataFrame):
        typed = {
            i
            for i, dtype in enumerate(X.dtypes)
            if isinstance(dtype, pd.CategoricalDtype)
        }
        declared.update(typed)
    return sorted(declared) if declared else None



def balance_probas_by_class_counts(
    probas: torch.Tensor,
    class_counts: np.ndarray,
) -> torch.Tensor:
    """Balance probabilities by class counts.

    Args:
        probas: The probabilities to balance.
        class_counts: The class counts to use for balancing.

    Returns:
        The balanced probabilities.
    """
    class_prob_in_train = class_counts / class_counts.sum()
    balanced_probas = probas / torch.from_numpy(class_prob_in_train).float().to(
        probas.device
    )
    return balanced_probas / balanced_probas.sum(dim=-1, keepdim=True)



def resolved_softmax_temperature(estimator) -> float:
    return estimator.softmax_temperature_

def resolved_n_estimators(estimator) -> int:
    return estimator.inference_config_.N_ESTIMATORS if estimator.n_estimators == "auto" else estimator.n_estimators

def expand_dates_and_text(
    X: XType,
    *,
    categorical_features_indices: Sequence[int] | None,
    inference_config: InferenceConfig,
) -> tuple[XType, DateTransformer, TextTransformer, list[str] | None, list[int] | None]:
    """Expand the datetime and text columns of a fit input, before validation.

    An expanded column is dropped and its features appended, so every column
    after it moves down. The returned labels and categorical positions describe
    the returned input, so no caller needs to know which transformer ran last.

    Returns:
        The expanded input, the two fitted transformers, the expanded input's
        column labels (`None` when `X` is not a `DataFrame`), and the declared
        categorical positions in it (`None` when none were declared).
    """
    from .preprocessing.datetimes import DateTransformer
    from .preprocessing.text import TextTransformer

    date_transformer = DateTransformer(
        categorical_indices=categorical_features_indices,
        transform_dates=inference_config.TRANSFORM_DATES,
    )
    X = date_transformer.fit_transform(X)
    categorical_indices = date_transformer.output_indices(categorical_features_indices)
    text_transformer = TextTransformer(
        categorical_indices=categorical_indices,
        transform_text=inference_config.TRANSFORM_TEXT,
        min_cardinality_for_text=inference_config.MIN_CARDINALITY_FOR_TEXT,
        n_components=inference_config.TEXT_N_COMPONENTS,
    )
    X = text_transformer.fit_transform(X)
    return (
        X,
        date_transformer,
        text_transformer,
        text_transformer.feature_names_out_,
        text_transformer.output_indices(categorical_indices),
    )
