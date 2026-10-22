#!/usr/bin/env python3
"""
Dump KV cache data from a GGSQ v3 binary file.

Usage:
    python3 dump_kv_data.py <path_to_ggsq_file>

Parses the llama.cpp GGSQ v3 sequence-state file format and prints
header fields: magic, version, n_tokens, n_stream, attention layer
info, recurrent layer info, and file size.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from qwen38_ggsq import inspect_qwen38_ggsq_v3


def dump_kv_data(path: str) -> None:
    p = Path(path)
    if not p.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    file_size = p.stat().st_size

    info = inspect_qwen38_ggsq_v3(path)
    print(f"File: {path}")
    print(f"File size: {file_size} bytes")
    print(f"Magic: 0x{info['magic']:08x}")
    print(f"Version: {info['version']}")
    print(f"n_tokens: {info['n_tokens']}")
    print(f"n_stream: {info['n_stream']}")
    print(f"Attention layers: {info['attention_layers']}")
    print(f"Recurrent layers: {info['recurrent_layers']}")
    print(f"v_trans: {info['v_trans']}")
    print(f"cache_type_k: {info['cache_type_k']}")
    print(f"cache_type_v: {info['cache_type_v']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_ggsq_file>", file=sys.stderr)
        sys.exit(1)

    dump_kv_data(sys.argv[1])
