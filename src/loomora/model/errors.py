"""Small runtime errors used by the model's memory fallback."""

import torch


def is_oom_error(error: Exception) -> bool:
    """Detect CUDA or MPS memory exhaustion."""
    return isinstance(error, torch.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )
