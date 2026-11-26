# Copyright (c) Prior Labs GmbH 2026.
# Modified for Loomora V1.0.

"""Load and cache weights for the shared inference implementation."""

import json
from functools import lru_cache
from pathlib import Path

import torch
from safetensors import safe_open


@lru_cache(maxsize=1)
def load_checkpoint_cached(path: str, identity: tuple[int, int]) -> dict:
    """Load weights, invalidating the cache when the file changes."""
    if Path(path).suffix == ".safetensors":
        with safe_open(path, framework="pt", device="cpu") as weights:
            metadata = weights.metadata() or {}
            if "config" not in metadata or "model_version" not in metadata:
                raise ValueError("Expected a Loomora weight file with architecture metadata")
            return {
                "config": json.loads(metadata["config"]),
                "model_version": metadata["model_version"],
                "model": {name: weights.get_tensor(name) for name in weights.keys()},
            }
    if Path(path).suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=True)
    raise ValueError("Expected a .safetensors or .pt weight file")
