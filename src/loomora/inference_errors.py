# Copyright (c) Prior Labs GmbH 2026.
# Modified for Loomora V1.0.
# Original notices for adapted implementations are in licenses/.
from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
import torch
from .inference_constants import XType, MAX_BATCHED_TEST_ROWS

class LoomoraError(Exception):
    """Base class for all Loomora-specific exceptions."""


class LoomoraUserError(LoomoraError):
    """Base class for errors caused by invalid user input (safe to map to HTTP 400)."""


class LoomoraValidationError(ValueError, LoomoraUserError):
    """User provided invalid data (shape, NaNs, categories, etc.)."""


class LoomoraOutOfMemoryError(LoomoraError):
    """Base class for GPU out-of-memory errors during prediction.

    This error provides guidance on how to handle large test sets that exceed
    available GPU memory.
    """

    device_name: str = "GPU"  # Override in subclasses

    def __init__(
        self,
        original_error: Exception | None = None,
        *,
        n_train_samples: int | None = None,
        n_test_samples: int | None = None,
        n_features: int | None = None,
        model_type: str = "classifier",
    ):
        predict_method = "predict_proba" if model_type == "classifier" else "predict"

        size_info = f" with {n_test_samples:,} test samples" if n_test_samples else ""

        size_line = ""
        if n_train_samples is not None and n_test_samples is not None:
            size_line = (
                f"Your sizes: {n_train_samples:,} train / "
                f"{n_test_samples:,} test samples"
            )
            if n_features is not None:
                size_line += f", {n_features} features"
            size_line += ".\n"

        max_batched_test_rows = MAX_BATCHED_TEST_ROWS
        message = (
            f"{self.device_name} out of memory{size_info}.\n\n"
            f"This is issue is usually caused by one of the following two reasons:\n\n"
            f"1) Large test set — split into batches:\n\n"
            f"    predictions = []\n"
            f"    for i in range(0, len(X_test), 100):\n"
            f"        pred = model.{predict_method}("
            f"X_test[i:i + 100])\n"
            f"        predictions.append(pred)\n"
            f"    predictions = np.vstack(predictions)\n\n"
            f"   Only with fit_mode='fit_with_cache', test rows are chunked\n"
            f"   automatically — lower the LOOMORA_MAX_BATCHED_TEST_ROWS\n"
            f"   environment variable (currently {max_batched_test_rows}) to\n"
            f"   reduce peak memory without changing your code.\n\n"
            f"2) Large training set — batching won't help.\n"
            f"   Subsample your training data; see https://docs.priorlabs.ai\n"
            f"   for further guidance.\n\n"
            f"{size_line}"
            f"Not sure which? If model.{predict_method}(X_test[:1]) "
            f"also fails, it's (2)."
        )
        if original_error is not None:
            message += f"\n\nOriginal error: {original_error}"
        super().__init__(message)
        self.original_error = original_error


class LoomoraCUDAOutOfMemoryError(LoomoraOutOfMemoryError):
    """Error raised when CUDA GPU runs out of memory during prediction."""

    device_name = "CUDA"


class LoomoraMPSOutOfMemoryError(LoomoraOutOfMemoryError):
    """Error raised when MPS (Apple Silicon) runs out of memory during prediction."""

    device_name = "MPS"


def is_oom_error(e: Exception) -> bool:
    """Return True if *e* is a GPU out-of-memory error (CUDA or MPS)."""
    return isinstance(e, torch.OutOfMemoryError) or (
        isinstance(e, RuntimeError) and "out of memory" in str(e).lower()
    )


def _sizes_for_message(X: XType) -> tuple[int, int | None]:
    """Read the row and column counts an OOM message should report."""
    shape = getattr(X, "shape", None)
    if shape is None:
        return len(X), None
    return shape[0], shape[1] if len(shape) > 1 else None


@contextmanager
def handle_oom_errors(
    devices: tuple[torch.device, ...],
    X: XType,
    model_type: str,
    n_train_samples: int | None = None,
) -> Generator[None, None, None]:
    """Context manager to catch OOM errors and raise helpful Loomora exceptions.

    Args:
        devices: The devices the model is running on.
        X: The input data as the model sees it, i.e. already converted, whose
            sample and feature counts the error message reports. The width is
            deliberately read from here rather than from `n_features_in_`, which
            describes the caller's frame and can be narrower (date expansion) --
            the message is about memory, so it has to name what the transformer
            actually runs on.
        model_type: Either "classifier" or "regressor".
        n_train_samples: Number of training samples (for the error message).

    Raises:
        LoomoraCUDAOutOfMemoryError: If a CUDA OOM error occurs.
        LoomoraMPSOutOfMemoryError: If an MPS OOM error occurs.
    """
    try:
        yield
    except torch.OutOfMemoryError as e:
        n_test_samples, n_features = _sizes_for_message(X)
        raise LoomoraCUDAOutOfMemoryError(
            e,
            n_train_samples=n_train_samples,
            n_test_samples=n_test_samples,
            n_features=n_features,
            model_type=model_type,
        ) from None
    except RuntimeError as e:
        is_mps = any(d.type == "mps" for d in devices)
        is_oom = "out of memory" in str(e).lower()
        if is_mps and is_oom:
            n_test_samples, n_features = _sizes_for_message(X)
            raise LoomoraMPSOutOfMemoryError(
                e,
                n_train_samples=n_train_samples,
                n_test_samples=n_test_samples,
                n_features=n_features,
                model_type=model_type,
            ) from None
        raise
