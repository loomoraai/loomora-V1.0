#  Copyright (c) Prior Labs GmbH 2026.
# Modified for Loomora V1.0.

"""KV cache data structures for explicit cache passing through architectures.

Provides cache containers for storing key-value projections from attention
layers, enabling efficient inference by reusing computed values across
different test sets without storing state inside the model.

Includes optional quantization (int8 or fp8) via
:class:`QuantizedKVCacheEntry` for reduced memory footprint with per-tensor
symmetric quantization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace

import torch
from torch import Tensor

QUANTIZED_KV_DTYPE: torch.dtype = torch.int8  # default
FP8_KV_DTYPE: torch.dtype = torch.float8_e4m3fn

#: Storage dtype for each quantized ``kv_cache_precision`` value.
KV_CACHE_PRECISION_DTYPES: dict[str, torch.dtype] = {
    "int8": QUANTIZED_KV_DTYPE,
    "fp8": FP8_KV_DTYPE,
}

# Low, high, max-magnitude value for each integer dtype.
# int8 uses the symmetric range [-127, 127] (one code below the full int8
# range) so that ``-max * scale`` equals ``+max * scale`` and dequantization
# cannot exceed the original absmax in magnitude.
_QUANTIZATION_RANGES: dict[torch.dtype, tuple[int, int, int]] = {
    torch.int8: (-127, 127, 127),
}

# Float dtypes are scaled onto the representable range and rounded by the
# dtype cast. e4m3fn has no infinity, so the clamp is required: casting a
# value above the max would produce NaN.
_FLOAT_QUANTIZATION_DTYPES = (torch.float8_e4m3fn,)


def _quantize_tensor(
    t: Tensor, dtype: torch.dtype = torch.int8
) -> tuple[Tensor, Tensor]:
    """Per-tensor symmetric quantization to the given *dtype*.

    Returns ``(quantized, scale)`` where ``scale = absmax / max_val`` and
    ``quantized ~= t / scale``, so ``float = quantized * scale``.
    """
    if dtype in _QUANTIZATION_RANGES:
        lo, hi, max_val = _QUANTIZATION_RANGES[dtype]
    elif dtype in _FLOAT_QUANTIZATION_DTYPES:
        max_val = torch.finfo(dtype).max
        lo, hi = -max_val, max_val
    else:
        raise ValueError(
            f"Unsupported quantization dtype {dtype}. Supported: "
            f"{list(_QUANTIZATION_RANGES) + list(_FLOAT_QUANTIZATION_DTYPES)}"
        )
    absmax = t.abs().amax()
    scale = absmax / float(max_val)
    # Avoid division by zero for all-zero tensors; floor at scale.dtype's
    # smallest positive normal so the clamp is representable in any dtype.
    scale = torch.clamp(scale, min=torch.finfo(scale.dtype).tiny)
    scaled = t / scale
    if dtype in _QUANTIZATION_RANGES:
        scaled = scaled.round()
    quantized = scaled.clamp(lo, hi).to(dtype)
    return quantized, scale


def _dequantize_tensor(t: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
    """Dequantize a quantized tensor back to floating-point *dtype*."""
    return t.to(dtype) * scale.to(dtype)


@dataclass
class KVCacheEntry:
    """A single key-value cache entry for one attention layer.

    Attributes:
        key: Cached key projections, shape ``(B, N_train, num_kv_heads, head_dim)``.
        value: Cached value projections, shape ``(B, N_train, num_kv_heads, head_dim)``.
    """

    key: Tensor | None = None
    value: Tensor | None = None

    def is_valid(self) -> bool:
        """Check if this cache entry contains valid data."""
        return self.key is not None and self.value is not None

    def to(self, device: torch.device | str) -> KVCacheEntry:
        """Move this entry to the given device. Returns a new KVCacheEntry."""
        if not self.is_valid():
            return KVCacheEntry()
        return KVCacheEntry(key=self.key.to(device), value=self.value.to(device))

    def quantize(
        self, dtype: torch.dtype = QUANTIZED_KV_DTYPE
    ) -> QuantizedKVCacheEntry:
        """Quantize this entry with per-tensor symmetric scaling.

        Args:
            dtype: Target quantization dtype (default `QUANTIZED_KV_DTYPE`).
        """
        assert self.is_valid()
        k_q, k_s = _quantize_tensor(self.key, dtype)
        v_q, v_s = _quantize_tensor(self.value, dtype)
        return QuantizedKVCacheEntry(key=k_q, value=v_q, key_scale=k_s, value_scale=v_s)


@dataclass
class QuantizedKVCacheEntry:
    """Quantized key-value cache entry with scale factors.

    Stores K/V as quantized tensors alongside scale factors for symmetric
    quantization: ``float_value = quantized_value * scale``. The quantized
    dtype is implicit in the stored tensors (see ``self.key.dtype``);
    the scale already encodes the dtype's quantization range, so dequantizing
    requires no extra dtype bookkeeping.

    Attributes:
        key: Quantized key projections, shape ``(B, N_train, num_kv_heads, head_dim)``.
        value: Quantized value projections, shape ``(B, N_train, num_kv_heads,
        head_dim)``.
        key_scale: Scalar scale factor for keys.
        value_scale: Scalar scale factor for values.
    """

    key: Tensor | None = None
    value: Tensor | None = None
    key_scale: Tensor | None = None
    value_scale: Tensor | None = None

    def is_valid(self) -> bool:
        """Check if this cache entry contains valid data."""
        return (
            self.key is not None
            and self.value is not None
            and self.key_scale is not None
            and self.value_scale is not None
        )

    def to(self, device: torch.device | str) -> QuantizedKVCacheEntry:
        """Move this entry to the given device."""
        if not self.is_valid():
            return QuantizedKVCacheEntry()
        return QuantizedKVCacheEntry(
            key=self.key.to(device),
            value=self.value.to(device),
            key_scale=self.key_scale.to(device),
            value_scale=self.value_scale.to(device),
        )

    def dequantize(self, dtype: torch.dtype) -> KVCacheEntry:
        """Dequantize back to a full-precision :class:`KVCacheEntry`."""
        assert self.is_valid()
        return KVCacheEntry(
            key=_dequantize_tensor(self.key, self.key_scale, dtype),
            value=_dequantize_tensor(self.value, self.value_scale, dtype),
        )


@dataclass
class KVCache(ABC):
    """Maps layer indices to KVCacheEntry or QuantizedKVCacheEntry objects.

    This is the base class for the architecture-specific caches. These
    store the per-layer key/value projections in ``kv`` and add their own fitted
    preprocessing / embedding state as extra fields.

    Attributes:
        kv: Maps layer/block index to cached key-value projections.
    """

    kv: dict[int, KVCacheEntry | QuantizedKVCacheEntry] = field(default_factory=dict)

    def is_populated(self) -> bool:
        """True when the cache contains valid data."""
        return any(entry.is_valid() for entry in self.kv.values())

    def is_empty(self) -> bool:
        """True when the cache has not been populated yet."""
        return not self.is_populated()

    @abstractmethod
    def to(self, device: torch.device | str) -> KVCache:
        """Move all entries to the given device. Returns a new KVCache."""
        return KVCache(kv=self._kv_to(device))

    def _kv_to(
        self, device: torch.device | str
    ) -> dict[int, KVCacheEntry | QuantizedKVCacheEntry]:
        """Move the per-layer KV entries to the given device."""
        return {idx: entry.to(device) for idx, entry in self.kv.items()}

    @staticmethod
    def _dict_of_tensors_to(
        state: dict[str, Tensor] | None, device: torch.device | str
    ) -> dict[str, Tensor] | None:
        """Move a dict of tensors to the given device (passing through ``None``)."""
        if state is None:
            return None
        return {k: v.to(device) for k, v in state.items()}

    @staticmethod
    def _list_of_tensors_to(
        tensors: list[Tensor] | None, device: torch.device | str
    ) -> list[Tensor] | None:
        """Move a list of tensors to the given device (passing through ``None``)."""
        if tensors is None:
            return None
        return [t.to(device) for t in tensors]


@dataclass
class LoomoraCache(KVCache):
    """Cached inputs and activations for Loomora V1.0.

    Stores everything needed to skip stages 0-2 for train rows and reuse
    cached K/V in the ICL transformer.

    Attributes:
        kv: Per-layer KV cache for the ICL transformer blocks.
        decoder_keys: Projected many-class decoder keys of shape
            `(B, N_train, H_dec, D_dec)`, i.e. the decoder's `k_projection`
            already applied to the post-ICL, post-norm train embeddings. Caching the keys rather than the embeddings they come from is smaller
            (`H_dec * D_dec` is below the ICL width) and keeps the projection off
            the predict path.
        train_shape: `(batch_size, num_train)` for validation.
        scaler_cache: Fitted standard-scaler statistics (`mean`, `std`). Allows
            standardising test-only data without train rows present.
        ecdf_context: ECDF bucket edges and their rank bounds per column, `(3,
            B, C, K)` at `ECDF_CONTEXT_DTYPE`, against which test cells are
            ranked. `K` is `min(cell_ecdf_num_buckets, n_train)`, so on a tall
            table this stops growing with the row count.
        inducing_hidden: Per-block inducing hidden states from the
            distribution embedder, each of shape `(B*C_out, n_ind, E)`.
            Allows running `cross_attn_block2` on test rows without
            recomputing `cross_attn_block1` from train rows.
    """

    decoder_keys: torch.Tensor | None = None
    train_shape: tuple[int, int] = (0, 0)
    scaler_cache: dict[str, torch.Tensor] | None = None
    ecdf_context: torch.Tensor | None = None
    inducing_hidden: list[torch.Tensor] | None = None
    inducing_hidden_round2: list[torch.Tensor] | None = None

    def to(self, device: torch.device | str) -> LoomoraCache:
        """Move all cached tensors to the given device."""
        return replace(self,
            kv=self._kv_to(device),
            decoder_keys=(
                self.decoder_keys.to(device) if self.decoder_keys is not None else None
            ),
            train_shape=self.train_shape,
            scaler_cache=self._dict_of_tensors_to(self.scaler_cache, device),
            ecdf_context=(
                self.ecdf_context.to(device) if self.ecdf_context is not None else None
            ),
            inducing_hidden=self._list_of_tensors_to(self.inducing_hidden, device),
            inducing_hidden_round2=self._list_of_tensors_to(self.inducing_hidden_round2, device),
        )

    def quantize(self, dtype: torch.dtype = QUANTIZED_KV_DTYPE) -> LoomoraCache:
        """Return a new cache with quantized ICL KV entries.

        Only the ICL KV cache is quantized; `decoder_keys`, `scaler_cache`,
        `ecdf_context` and `inducing_hidden` stay at the precision they were built
        at. `InferenceEngineExplicitKVCache`
        calls this whenever the resolved `kv_cache_precision` is not `"auto"`, which
        is why `get_supported_kv_cache_precisions` has to advertise the dtypes this
        handles.

        Args:
            dtype: Target quantization dtype (default `QUANTIZED_KV_DTYPE`, int8;
                `FP8_KV_DTYPE` is the other one the engine can ask for).
        """
        quantized_kv = {
            idx: (entry.quantize(dtype) if isinstance(entry, KVCacheEntry) else entry)
            for idx, entry in self.kv.items()
        }
        return replace(self, kv=quantized_kv)
