"""Minimal writer for llama.cpp control-vector GGUF files (--control-vector / --control-vector-scaled)."""
from __future__ import annotations

import struct

import numpy as np

_ALIGN = 32


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def write_cvector_gguf(path: str, directions: dict, model_hint: str = "") -> str:
    """directions: {N: 1-D float array}, using llama.cpp's `direction.N` convention
    (added to the output of 0-indexed block N). Layers without an entry are left untouched."""
    n_max = max(directions)
    tensors = [(f"direction.{i}", np.asarray(directions[i], dtype=np.float32))
               for i in range(1, n_max + 1) if i in directions]
    kv = _s("general.architecture") + struct.pack("<I", 8) + _s("controlvector")
    kv += _s("controlvector.model_hint") + struct.pack("<I", 8) + _s(model_hint or "unknown")
    kv += _s("controlvector.layer_count") + struct.pack("<I", 4) + struct.pack("<I", len(tensors))
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 3) + kv

    tinfo, data = b"", b""
    for name, vec in tensors:
        data += b"\x00" * ((-len(data)) % _ALIGN)          # each tensor's data offset is aligned
        tinfo += _s(name) + struct.pack("<I", 1) + struct.pack("<Q", vec.shape[0])
        tinfo += struct.pack("<IQ", 0, len(data))           # type 0 = F32, offset into the data section
        data += vec.tobytes()
    blob = header + tinfo
    blob += b"\x00" * ((-len(blob)) % _ALIGN) + data
    with open(path, "wb") as f:
        f.write(blob)
    return path
