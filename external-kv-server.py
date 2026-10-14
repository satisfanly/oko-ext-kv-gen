#!/usr/bin/env python3
"""
External KV Cache Generator Server (PoC)

Implements the endpoint specified in /tmp/external-kv-cache-generator.md:
  POST http://127.0.0.1:11654/kv
  Request:  JSON array of token IDs (application/json)
  Response: binary KV cache data in llama.cpp GGSQ v3 format (application/octet-stream)

Uses brain_runner_qwen3_8_27b_q8_ggml_cuda.py to generate real KV caches
from the Qwen3.8-27B Q8_0 GGUF model, then serializes to llama.cpp's
GGSQ v3 single-sequence state format using qwen38_ggsq.py.

The output is directly compatible with llama.cpp's llama_state_seq_set_data_ext()
for the Qwen3.8-27B hybrid architecture (16 full-attention + 48 DeltaNet layers).

IMPORTANT: The target llama.cpp context must match:
  - Same model (Qwen3.8-27B)
  - Same KV cache types (--cache-type-k/--cache-type-v)
  - Same Flash Attention mode (--llama-flash-attn)
  - Same stream count (default: 1)

Usage:
    python3 external-kv-server.py \
        --model /path/to/Qwen3.8-27B-...-Q8_0.gguf \
        [--host 127.0.0.1] [--port 11654] \
        [--mlp-chunk 4096] [--linear-attn-chunk 8192] [--full-attn-chunk 512] \
        [--cache-type-k f16] [--cache-type-v f16] \
        [--llama-flash-attn | --no-llama-flash-attn]
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import torch
from http.server import BaseHTTPRequestHandler, HTTPServer

# Add the ext-kv-cache-generator directory to the path so we can import
# brain_runner and its dependencies.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from brain_runner_qwen3_8_27b_q8_ggml_cuda import brain_runner
from qwen38_ggsq import write_qwen38_ggsq_v3, QWEN38_27B

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ext-kv-server")


class KVCacheServer:
    """Holds the loaded model runner and configuration."""

    def __init__(
        self,
        model_path: str,
        mlp_chunk_tokens: int = 4096,
        linear_attn_chunk_tokens: int = 8192,
        full_attn_chunk_tokens: int = 512,
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
        llama_flash_attn: bool = True,
    ):
        self.model_path = model_path
        self.runner = None
        self.mlp_chunk_tokens = mlp_chunk_tokens
        self.linear_attn_chunk_tokens = linear_attn_chunk_tokens
        self.full_attn_chunk_tokens = full_attn_chunk_tokens
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.llama_flash_attn = llama_flash_attn

    def load_model(self):
        """Load the GGUF model via brain_runner. Called once at startup."""
        logger.info("Loading model: %s", self.model_path)
        t0 = time.perf_counter()

        self.runner = brain_runner(
            self.model_path,
            mlp_chunk_tokens=self.mlp_chunk_tokens,
            linear_attn_chunk_tokens=self.linear_attn_chunk_tokens,
            full_attn_chunk_tokens=self.full_attn_chunk_tokens,
        )

        elapsed = time.perf_counter() - t0
        logger.info("Model loaded in %.2fs", elapsed)

    @staticmethod
    def _state0(d, name, layer_idx):
        """Extract state tensor at state_idx=0 from HF cache layer."""
        if isinstance(d, dict):
            t = d.get(0)
            if t is None and d:
                t = next(iter(d.values()))
        elif isinstance(d, (tuple, list)):
            t = d[0] if d else None
        elif torch.is_tensor(d):
            t = d
        else:
            t = None
        if not torch.is_tensor(t):
            raise RuntimeError(
                f"cache.layers[{layer_idx}].{name} has no state tensor at state_idx=0"
            )
        return t

    @staticmethod
    def extract_qwen38_cache(cache):
        """Extract attention and recurrent states from HF DynamicCache."""
        layers = getattr(cache, "layers", None)
        if layers is None:
            raise RuntimeError(
                f"expected a Transformers DynamicCache-like object with .layers, got {type(cache)!r}"
            )
        if len(layers) != QWEN38_27B.n_layers:
            raise RuntimeError(
                f"cache has {len(layers)} layers, expected {QWEN38_27B.n_layers}"
            )

        attention = {}
        recurrent = {}

        full = set(QWEN38_27B.full_attention_layers)
        for il, layer in enumerate(layers):
            if il in full:
                k = getattr(layer, "keys", None)
                v = getattr(layer, "values", None)
                if not torch.is_tensor(k) or not torch.is_tensor(v):
                    raise RuntimeError(
                        f"full-attention cache layer {il} has no .keys/.values tensors"
                    )
                attention[il] = (k, v)
            else:
                conv = KVCacheServer._state0(
                    getattr(layer, "conv_states", None), "conv_states", il
                )
                state = KVCacheServer._state0(
                    getattr(layer, "recurrent_states", None), "recurrent_states", il
                )
                recurrent[il] = (conv, state)

        return attention, recurrent

    def generate_kv_cache(self, tokens: list[int]) -> bytes:
        """
        Generate KV cache for the given token sequence.

        Args:
            tokens: List of token IDs (integers).

        Returns:
            Binary KV cache data in llama.cpp GGSQ v3 format.
        """
        if self.runner is None:
            raise RuntimeError("Model not loaded")

        logger.info("Generating KV cache for %d tokens", len(tokens))
        t0 = time.perf_counter()

        # brain_runner expects an example dict with input_ids, attention_mask, labels
        attention_mask = [1] * len(tokens)
        example = {
            "input_ids": tokens,
            "attention_mask": attention_mask,
            "labels": tokens,
        }

        # Build the KV cache (returns a DynamicCache object)
        cache = self.runner.build_kv_cache(example, verbose=True)

        # Extract attention and recurrent states
        attention, recurrent = self.extract_qwen38_cache(cache)

        # Serialize to GGSQ v3 format (llama.cpp compatible)
        # write_qwen38_ggsq_v4 expects a Path, so we write to a temp file
        # and read it back.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            tmp_path = tmp.name
        try:
            write_qwen38_ggsq_v3(
                tmp_path,
                tokens,
                attention,
                recurrent,
                cache_type_k=self.cache_type_k,
                cache_type_v=self.cache_type_v,
                v_trans=not self.llama_flash_attn,
                n_stream=1,
            )
            with open(tmp_path, "rb") as f:
                kv_data = f.read()
        finally:
            os.unlink(tmp_path)

        elapsed = time.perf_counter() - t0
        logger.info(
            "KV cache generated in %.2fs (%d bytes, seq_len=%d)",
            elapsed,
            len(kv_data),
            cache.get_seq_length(),
        )

        return kv_data


class KVCacheHandler(BaseHTTPRequestHandler):
    """HTTP handler for the external KV cache generator endpoint."""

    server_version = "ExtKVCacheServer/1.0"
    kv_server: KVCacheServer = None  # Set by main()

    def log_message(self, format, *args):
        logger.info(format % args)

    def do_POST(self):
        if self.path != "/kv":
            self.send_error(404, "Not Found")
            return

        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_error(400, "Bad Request: empty body")
            return

        body = self.rfile.read(content_length)

        # Parse JSON array of token IDs
        try:
            tokens = json.loads(body)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON: %s", e)
            self.send_error(400, "Bad Request: invalid JSON")
            return

        if not isinstance(tokens, list):
            logger.error("Expected JSON array, got %s", type(tokens).__name__)
            self.send_error(400, "Bad Request: expected JSON array")
            return

        # Validate tokens are integers
        for i, t in enumerate(tokens):
            if not isinstance(t, int):
                logger.error("Token at index %d is not an integer: %s", i, t)
                self.send_error(400, "Bad Request: all tokens must be integers")
                return

        logger.info(
            "Received %d tokens (first 10: %s)",
            len(tokens),
            tokens[:10],
        )

        # Generate KV cache
        try:
            kv_data = self.kv_server.generate_kv_cache(tokens)
        except Exception as e:
            logger.exception("Failed to generate KV cache")
            self.send_error(500, f"Internal Server Error: {e}")
            return

        # Send response
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(kv_data)))
        self.end_headers()
        self.wfile.write(kv_data)

        logger.info("Sent %d bytes of KV cache data", len(kv_data))

    def do_GET(self):
        """Health check endpoint."""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status = {
                "status": "ok",
                "model_loaded": self.kv_server.runner is not None,
                "model_path": self.kv_server.model_path,
            }
            self.wfile.write(json.dumps(status).encode())
        else:
            self.send_error(404, "Not Found")


def main():
    parser = argparse.ArgumentParser(
        description="External KV cache generator server (PoC)"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the GGUF model file",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=11654,
        help="Port to listen on (default: 11654)",
    )
    parser.add_argument(
        "--mlp-chunk",
        type=int,
        default=4096,
        help="MLP chunk size in tokens (default: 4096)",
    )
    parser.add_argument(
        "--linear-attn-chunk",
        type=int,
        default=8192,
        help="Linear attention chunk size in tokens (default: 8192)",
    )
    parser.add_argument(
        "--full-attn-chunk",
        type=int,
        default=512,
        help="Full attention chunk size in tokens (default: 512)",
    )
    parser.add_argument(
        "--cache-type-k",
        choices=["f16", "f32"],
        default="f16",
        help="KV cache K type (default: f16)",
    )
    parser.add_argument(
        "--cache-type-v",
        choices=["f16", "f32"],
        default="f16",
        help="KV cache V type (default: f16)",
    )
    parser.add_argument(
        "--llama-flash-attn",
        action="store_true",
        default=True,
        help="Target llama.cpp uses Flash Attention (default: True)",
    )
    parser.add_argument(
        "--no-llama-flash-attn",
        action="store_true",
        help="Target llama.cpp uses classic transposed V layout",
    )
    args = parser.parse_args()

    if args.no_llama_flash_attn:
        args.llama_flash_attn = False

    # Create and load the KV cache server
    kv_server = KVCacheServer(
        model_path=args.model,
        mlp_chunk_tokens=args.mlp_chunk,
        linear_attn_chunk_tokens=args.linear_attn_chunk,
        full_attn_chunk_tokens=args.full_attn_chunk,
        cache_type_k=args.cache_type_k,
        cache_type_v=args.cache_type_v,
        llama_flash_attn=args.llama_flash_attn,
    )

    # Load model before starting the server
    kv_server.load_model()

    # Set up HTTP server
    KVCacheHandler.kv_server = kv_server
    server = HTTPServer((args.host, args.port), KVCacheHandler)

    logger.info(
        "External KV cache server listening on http://%s:%d/kv",
        args.host,
        args.port,
    )
    logger.info("Health check: http://%s:%d/health", args.host, args.port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server")
        server.shutdown()


if __name__ == "__main__":
    main()
