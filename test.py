#!/usr/bin/env python3
from pathlib import Path

import torch
from transformers import AutoTokenizer

from brain_runner_qwen3_8_27b_q8_ggml_cuda import brain_runner


# =============================================================================
# EDIT THIS
# =============================================================================

MODEL_PATH = "../models/Qwen3.8-27B-TTURBO-Fable-C-Fusion-709-L-Uncen-TOOLS-NM-DAU-NEO-MAX-MTP-Q8_0.gguf"

# =============================================================================

CACHE_PATH = "/tmp/qwen38_test_kv_cache.pt"
PROMPT = "Explain in one paragraph why the sky is blue." * 5000

def load_tokenizer(gguf_path: str):
    """
    Load tokenizer metadata from the same GGUF.

    Recent Transformers versions can reconstruct the tokenizer directly from
    supported GGUF metadata, so no separate HF tokenizer directory is required.
    """
    gguf_path = Path(gguf_path).resolve()

    return AutoTokenizer.from_pretrained(
        str(gguf_path.parent),
        gguf_file=str(gguf_path),
        local_files_only=True,
        trust_remote_code=True,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test")

    model_path = Path(MODEL_PATH).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"MODEL_PATH does not exist: {model_path}\n"
            "Edit MODEL_PATH at the top of test.py."
        )

    print(f"Model: {model_path}")
    print(f"Cache: {CACHE_PATH}")
    print(f"CUDA : {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------------------------------
    # Tokenizer + prompt
    # -------------------------------------------------------------------------

    tokenizer = load_tokenizer(str(model_path))

    encoded = tokenizer(
        PROMPT,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = encoded["input_ids"][0].tolist()
    attention_mask = encoded["attention_mask"][0].tolist()

    # brain_runner.step() still expects labels even though this test only needs
    # logits. Reusing input_ids keeps the test simple.
    example = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids,
    }

    print(f"Prompt tokens: {len(input_ids)}")
    print(f"Prompt: {PROMPT!r}")

    # -------------------------------------------------------------------------
    # Load Q8_0 GGUF model through the GGML-CUDA runner
    # -------------------------------------------------------------------------

    runner = brain_runner(str(model_path))

    print("\n=== Build + save KV cache ===")
    cache = runner.build_and_save_kv_cache(
        example,
        CACHE_PATH,
        verbose=True,
    )

    original_cache_len = int(cache.get_seq_length())
    print(f"Built cache sequence length: {original_cache_len}")
    print(f"Saved cache to: {CACHE_PATH}")

    if original_cache_len != len(input_ids):
        raise RuntimeError(
            f"Built cache length is {original_cache_len}, "
            f"but prompt length is {len(input_ids)}"
        )

if __name__ == "__main__":
    main()
