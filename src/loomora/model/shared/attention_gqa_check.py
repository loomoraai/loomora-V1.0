#  Copyright (c) Prior Labs GmbH 2026.
# Modified for Loomora V1.0.

"""Check for the enable_gqa parameter of PyTorch scaled_dot_product_attention."""

from __future__ import annotations

import functools

import torch
from packaging.version import Version


@functools.cache
def gqa_is_supported(device: torch.device | None = None, dtype: torch.dtype | None = None) -> bool:
    """Check if PyTorch's scaled_dot_product_attention supports enable_gqa parameter.

    An explicit device checks the inference backend, including CPU in PyTorch
    2.9+. Calls without a device retain the CUDA training capability check.
    """
    if device is not None:
        release = Version(torch.__version__).release
        if device.type == "cpu":
            return release >= (2, 9)
        return (
            device.type == "cuda" and release >= (2, 5)
            and dtype in {torch.float16, torch.bfloat16}
            and torch.cuda.get_device_capability(device)[0] >= 8
        )

    if not torch.cuda.is_available():
        return False

    # torch.__version__ supports string comparison.
    has_enable_gqa = torch.__version__ >= "2.5"
    if not has_enable_gqa:
        return False

    # Check compute capability only if CUDA is available
    # We need compute capability >= 8.0 for efficient GQA
    device = torch.cuda.current_device()
    nvidia_compute_capability = torch.cuda.get_device_capability(device)
    return nvidia_compute_capability[0] >= 8
