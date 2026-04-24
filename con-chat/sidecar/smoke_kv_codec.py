#!/usr/bin/env python3
from __future__ import annotations

import torch

from kv_codec import TurboQuantStyleKVCodec


def main() -> None:
    torch.manual_seed(7)
    key = torch.randn(1, 4, 32, 16, dtype=torch.float16)
    value = torch.randn(1, 4, 32, 16, dtype=torch.float16)
    codec = TurboQuantStyleKVCodec(bits=4, group_size=64)

    payload = codec.compress_cache(((key, value),))
    restored_key, restored_value = codec.decompress_cache(payload)
    accounting = codec.accounting(payload)

    source = torch.cat([key.flatten().float(), value.flatten().float()])
    restored = torch.cat([restored_key.flatten().float(), restored_value.flatten().float()])
    relative_error = torch.linalg.vector_norm(source - restored) / torch.linalg.vector_norm(source).clamp_min(1e-8)

    print(f"codec={accounting.codec}")
    print(f"original_bytes={accounting.original_bytes}")
    print(f"compressed_bytes={accounting.compressed_bytes}")
    print(f"compression_ratio={accounting.compression_ratio:.4f}")
    print(f"reconstruction_relative_error={float(relative_error):.6f}")


if __name__ == "__main__":
    main()
