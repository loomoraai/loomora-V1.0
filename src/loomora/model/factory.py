"""Construct Loomora V1.0 and check the compatibility of serialized weights."""

from dataclasses import asdict
from typing import Any

import torch

from .architecture import LoomoraConfig, LoomoraModel


def build_config(config: dict[str, Any] | LoomoraConfig | None = None) -> LoomoraConfig:
    if isinstance(config, LoomoraConfig):
        return config
    values = dict(config or {})
    values["name"] = "Loomora V1.0"
    return LoomoraConfig(**values)


def build_model(
    config: dict[str, Any] | LoomoraConfig | None = None, *,
    device: torch.device | str | None = None, dtype: torch.dtype | None = None,
) -> LoomoraModel:
    return LoomoraModel(config=build_config(config), device=device, dtype=dtype)


def validate_checkpoint(payload: dict[str, Any], config: LoomoraConfig | None = None) -> None:
    if payload.get("model_version") != LoomoraModel.model_version:
        raise ValueError("Weights must match the Loomora V1.0 architecture")
    if config is not None and "config" in payload:
        saved = asdict(build_config(payload["config"]))
        expected = asdict(config)
        runtime = {"name", "cell_embed_row_chunk_size", "inference_row_chunk_size", "inference_col_chunk_size"}
        different = [key for key in saved if key not in runtime and saved[key] != expected[key]]
        if different:
            raise ValueError(f"Weight configuration differs in {', '.join(different)}")
