#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class KVAccounting:
    codec: str
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    bits: int
    group_size: int
    tensor_count: int


class KVCodec:
    name = "base"

    def compress_cache(self, cache: Any) -> dict[str, Any]:
        raise NotImplementedError

    def decompress_cache(self, payload: dict[str, Any]) -> Any:
        raise NotImplementedError

    def accounting(self, payload: dict[str, Any]) -> KVAccounting:
        raise NotImplementedError


class TurboQuantStyleKVCodec(KVCodec):
    """CPU-side grouped int4 KV quantization for memory-object storage.

    This is intentionally not a TurboQuant attention kernel. It is a clean
    memory-object codec boundary with basic grouped quantization so normal live
    chat can continue to use model-supported KV caches.
    """

    name = "turboquant-style-grouped-int4-v1"

    def __init__(self, *, bits: int = 4, group_size: int = 64) -> None:
        if bits != 4:
            raise ValueError("v1 supports 4-bit quantization only")
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        self.bits = bits
        self.group_size = group_size

    def compress_cache(self, cache: Any) -> dict[str, Any]:
        tensors = list(self._iter_tensors(cache))
        entries = [self._quantize_tensor(tensor.detach().to("cpu")) for tensor in tensors]
        original_bytes = sum(entry["original_bytes"] for entry in entries)
        compressed_bytes = sum(entry["packed"].numel() for entry in entries)
        compressed_bytes += sum(entry["scale"].numel() * entry["scale"].element_size() for entry in entries)
        compressed_bytes += sum(entry["shape_bytes"] for entry in entries)
        return {
            "codec": self.name,
            "bits": self.bits,
            "group_size": self.group_size,
            "entries": entries,
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "todo": "Add TurboQuant-style rotations and sign-residual correction after the live chat path is stable.",
        }

    def decompress_cache(self, payload: dict[str, Any]) -> list[torch.Tensor]:
        if payload.get("codec") != self.name:
            raise ValueError(f"unsupported_codec:{payload.get('codec')}")
        return [self._dequantize_tensor(entry) for entry in payload.get("entries", [])]

    def accounting(self, payload: dict[str, Any]) -> KVAccounting:
        original_bytes = int(payload.get("original_bytes") or 0)
        compressed_bytes = int(payload.get("compressed_bytes") or 0)
        ratio = (original_bytes / compressed_bytes) if compressed_bytes else 0.0
        return KVAccounting(
            codec=str(payload.get("codec") or self.name),
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=ratio,
            bits=int(payload.get("bits") or self.bits),
            group_size=int(payload.get("group_size") or self.group_size),
            tensor_count=len(payload.get("entries", [])),
        )

    def _iter_tensors(self, cache: Any) -> list[torch.Tensor]:
        if cache is None:
            return []
        if torch.is_tensor(cache):
            return [cache]
        if hasattr(cache, "layers"):
            tensors: list[torch.Tensor] = []
            for layer in getattr(cache, "layers", []):
                for name in ("keys", "values", "key_cache", "value_cache"):
                    tensor = getattr(layer, name, None)
                    if torch.is_tensor(tensor) and tensor.numel():
                        tensors.append(tensor)
            return tensors
        if hasattr(cache, "to_legacy_cache"):
            cache = cache.to_legacy_cache()
        tensors: list[torch.Tensor] = []
        if isinstance(cache, (list, tuple)):
            for item in cache:
                tensors.extend(self._iter_tensors(item))
        elif isinstance(cache, dict):
            for item in cache.values():
                tensors.extend(self._iter_tensors(item))
        return tensors

    def _quantize_tensor(self, tensor: torch.Tensor) -> dict[str, Any]:
        source = tensor.to(dtype=torch.float16, device="cpu").contiguous()
        flat = source.flatten()
        original_numel = flat.numel()
        pad = (-original_numel) % self.group_size
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        grouped = flat.reshape(-1, self.group_size).to(dtype=torch.float32)
        max_abs = grouped.abs().amax(dim=1).clamp_min(1e-8)
        scale = max_abs / 7.0
        quantized = torch.round(grouped / scale[:, None]).clamp(-8, 7).to(torch.int16)
        unsigned = (quantized + 8).to(torch.uint8).flatten()
        if unsigned.numel() % 2:
            unsigned = torch.nn.functional.pad(unsigned, (0, 1))
        packed = unsigned[0::2] | (unsigned[1::2] << 4)
        return {
            "shape": tuple(source.shape),
            "dtype": str(source.dtype),
            "numel": original_numel,
            "pad": pad,
            "scale": scale.to(dtype=torch.float16),
            "packed": packed.contiguous(),
            "original_bytes": source.numel() * source.element_size(),
            "shape_bytes": len(source.shape) * 8,
        }

    def _dequantize_tensor(self, entry: dict[str, Any]) -> torch.Tensor:
        packed = entry["packed"].to(dtype=torch.uint8, device="cpu")
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        unsigned = torch.empty(packed.numel() * 2, dtype=torch.uint8)
        unsigned[0::2] = low
        unsigned[1::2] = high
        numel = int(entry["numel"])
        pad = int(entry.get("pad") or 0)
        quantized = unsigned[: numel + pad].to(torch.int16) - 8
        grouped = quantized.reshape(-1, self.group_size).to(torch.float32)
        scale = entry["scale"].to(dtype=torch.float32, device="cpu")
        restored = (grouped * scale[:, None]).flatten()[:numel]
        return restored.reshape(tuple(entry["shape"])).to(dtype=torch.float16)


def codec_name() -> str:
    return TurboQuantStyleKVCodec.name
