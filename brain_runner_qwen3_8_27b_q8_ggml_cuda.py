#!/usr/bin/env python3
from pathlib import Path

import torch
import time
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from transformers.masking_utils import (
    create_causal_mask,
    create_recurrent_attention_mask,
)
from transformers.cache_utils import DynamicCache

from q8_ggml_cuda import load_model_from_gguf_without_q8_dequant

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Workaround to address assert: "object has no attribute '__name__'. Did you mean: '__ne__'?"
import transformers.utils as tf_utils
def _noop_auto_docstring(obj=None, *args, **kwargs):
    # Transformers may call auto_docstring with metadata such as
    # checkpoint=..., custom_intro=..., etc. Ignore all of it.
    if obj is None:
        def decorator(real_obj):
            return real_obj
        return decorator
    return obj
tf_utils.auto_docstring = _noop_auto_docstring
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from trainer_utils import modules as mu

SEED = 42
MLP_CHUNK_TOKENS = 4096
LINEAR_ATTN_CHUNK_TOKENS = 8192
FULL_ATTN_CHUNK_TOKENS = 512
GPU_COMPUTE_DTYPE = torch.float16

torch.manual_seed(SEED)

def _ram_used_mb():
    """Current resident RAM (RSS) used by this Python process, in MiB."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # /proc reports KiB.
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def _cuda_mem_mb():
    """Current PyTorch CUDA allocator usage as (allocated_MiB, reserved_MiB)."""
    return (
        torch.cuda.memory_allocated("cuda") / 1024**2,
        torch.cuda.memory_reserved("cuda") / 1024**2,
    )


def _cuda_peak_mem_mb():
    """PyTorch CUDA peak usage since the last reset_peak_memory_stats()."""
    return (
        torch.cuda.max_memory_allocated("cuda") / 1024**2,
        torch.cuda.max_memory_reserved("cuda") / 1024**2,
    )


def _log_cuda_stage(label):
    """Synchronize and print current + peak PyTorch CUDA allocator usage."""
    torch.cuda.synchronize()
    allocated, reserved = _cuda_mem_mb()
    peak_allocated, peak_reserved = _cuda_peak_mem_mb()
    print(
        f"... {label}: "
        f"GPU-allocated={allocated:.2f} MB, "
        f"GPU-reserved={reserved:.2f} MB, "
        f"GPU-peak-allocated={peak_allocated:.2f} MB, "
        f"GPU-peak-reserved={peak_reserved:.2f} MB"
    )


def _nested_cuda_tensors(value, path, out, seen):
    """Collect CUDA tensors reachable through ordinary Python containers/objects."""
    obj_id = id(value)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            out.append((path, tuple(value.shape), str(value.dtype), value.numel() * value.element_size()))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _nested_cuda_tensors(item, f"{path}[{key!r}]", out, seen)
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _nested_cuda_tensors(item, f"{path}[{idx}]", out, seen)
        return


def _module_cuda_tensors(module, prefix="module"):
    """Find registered parameters/buffers and plain tensor attrs left on CUDA."""
    out = []
    seen = set()

    for name, param in module.named_parameters(recurse=True):
        if param.device.type == "cuda":
            out.append((f"{prefix}.param:{name}", tuple(param.shape), str(param.dtype), param.numel() * param.element_size()))
            seen.add(id(param))

    for name, buf in module.named_buffers(recurse=True):
        if buf.device.type == "cuda":
            out.append((f"{prefix}.buffer:{name}", tuple(buf.shape), str(buf.dtype), buf.numel() * buf.element_size()))
            seen.add(id(buf))

    # Custom Q8 wrappers keep packed bytes as plain attrs rather than Parameters.
    # Inspect ordinary attrs recursively too, while avoiding module recursion loops.
    for attr_name, value in vars(module).items():
        if attr_name in {"_modules", "_parameters", "_buffers"}:
            continue
        _nested_cuda_tensors(value, f"{prefix}.{attr_name}", out, seen)

    return out


class brain_runner:
    def __prep_model(self, model_dir):
        if not model_dir.endswith(".gguf"):
            raise ValueError(
                f"This debug runner is GGUF-only for now, got: {model_dir}"
            )

        # Do NOT use HF's normal GGUF loader here: on CUDA it currently
        # dequantizes Q8_0 matrices into dense PyTorch tensors.
        #
        # This loader instantiates the HF model on META, materializes only
        # non-Q8 tensors, and leaves Q8_0 Linear/Embedding weights packed.
        # Their forward ops are executed by GGML's CUDA backend.
        model = load_model_from_gguf_without_q8_dequant(
            model_dir,
            dtype=torch.bfloat16,
            strict_q8=True,
            verbose=True,
        )

        # Force Transformers full-attention layers through SDPA rather than an
        # eager implementation that may materialize an O(S^2) attention matrix.
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("sdpa")
        elif hasattr(model, "set_attention_implementation"):
            model.set_attention_implementation("sdpa")
        else:
            model.config._attn_implementation = "sdpa"
            if hasattr(model, "model") and hasattr(model.model, "config"):
                model.model.config._attn_implementation = "sdpa"

        attn_impl = getattr(model.config, "_attn_implementation", None)
        if attn_impl != "sdpa":
            raise RuntimeError(
                "Failed to force Transformers full attention to SDPA: "
                f"config._attn_implementation={attn_impl!r}"
            )

        if hasattr(model, "model") and hasattr(model.model, "config"):
            model.model.config._attn_implementation = "sdpa"
            inner_attn_impl = getattr(
                model.model.config, "_attn_implementation", None
            )
            if inner_attn_impl != "sdpa":
                raise RuntimeError(
                    "Inner model did not accept SDPA attention implementation: "
                    f"{inner_attn_impl!r}"
                )

        print(
            "[attention] transformers implementation=sdpa; "
            "full-attention CUDA kernel=EFFICIENT_ATTENTION "
            "(forced, persistent FP16 K/V workspace, BF16 cache in CPU RAM)"
        )

        model.eval()

        # Debug invariant: the model stays CPU-resident. Scan registered
        # Parameters/Buffers plus plain tensor attrs used by custom Q8 wrappers.
        cuda_tensors = _module_cuda_tensors(model, "model")
        if cuda_tensors:
            preview = ", ".join(item[0] for item in cuda_tensors[:20])
            raise RuntimeError(
                "Model unexpectedly has persistent CUDA tensors after load: "
                + preview
            )

        print(
            f"[memory] model ready (CPU-resident): "
            f"RAM={_ram_used_mb():.2f} MB, "
            f"PyTorch-GPU={torch.cuda.memory_allocated('cuda') / 1024**2:.2f} MB"
        )

        return model

    @staticmethod
    def _full_attention_chunk_sm75_efficient_fp16(
        attn,
        hidden_states,
        position_embeddings,
        key_workspace_fp16,
        value_workspace_fp16,
        key_cache_cpu,
        value_cache_cpu,
        chunk_start,
    ):
        """
        Run ONE full-attention prefill chunk using persistent K/V workspaces.

        Persistent for the duration of this full-attention layer:
          * key_workspace_fp16/value_workspace_fp16 on CUDA, preallocated for
            the complete sequence exactly once
          * key_cache_cpu/value_cache_cpu in BF16 RAM, also preallocated once

        Per chunk:
          1. project Q/K/V in the native model dtype (BF16)
          2. apply absolute RoPE for this chunk
          3. copy ONLY the new K/V chunk into:
               a) the final BF16 CPU HF-cache backing tensors
               b) the persistent FP16 CUDA SDPA workspace
          4. run attention against workspace[..., :chunk_end, :]
          5. return the chunk output in the model's native dtype

        No accumulated BF16 K/V cache is kept on CUDA and no previous FP16 K/V
        prefix is recast/copied on later chunks.
        """
        input_dtype = hidden_states.dtype
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query_states, gate = torch.chunk(
            attn.q_proj(hidden_states).view(
                *input_shape, -1, attn.head_dim * 2
            ),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = attn.q_norm(
            query_states.view(hidden_shape)
        ).transpose(1, 2)

        key_states = attn.k_norm(
            attn.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)

        value_states = attn.v_proj(hidden_states).view(
            hidden_shape
        ).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )

        q_len = query_states.shape[-2]
        chunk_end = chunk_start + q_len

        if chunk_end > key_workspace_fp16.shape[-2]:
            raise RuntimeError(
                f"Full-attention workspace overflow: chunk_end={chunk_end}, "
                f"capacity={key_workspace_fp16.shape[-2]}"
            )

        # Final persistent HF cache lives in CPU RAM. copy_ performs the
        # CUDA->CPU transfer directly into the preallocated destination.
        key_cache_cpu[
            :, :, chunk_start:chunk_end, :
        ].copy_(key_states)
        value_cache_cpu[
            :, :, chunk_start:chunk_end, :
        ].copy_(value_states)

        # Persistent CUDA workspace lives in FP16 for sm75 memory-efficient
        # attention. copy_ performs BF16->FP16 conversion into the existing
        # workspace without allocating/casting the entire accumulated prefix.
        key_workspace_fp16[
            :, :, chunk_start:chunk_end, :
        ].copy_(key_states)
        value_workspace_fp16[
            :, :, chunk_start:chunk_end, :
        ].copy_(value_states)

        # Q exists only for this chunk. K/V are views into the fixed workspace.
        query_sdpa = query_states.to(torch.float16).contiguous()
        key_sdpa = key_workspace_fp16[:, :, :chunk_end, :]
        value_sdpa = value_workspace_fp16[:, :, :chunk_end, :]

        kv_len = chunk_end

        # q is the newest chunk while k/v are the full prefix, therefore causal
        # alignment must be lower-right rather than the upper-left alignment of
        # plain is_causal=True for a non-square Q x KV matrix.
        causal_bias = causal_lower_right(q_len, kv_len)

        del query_states
        del key_states
        del value_states

        try:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                attn_output = F.scaled_dot_product_attention(
                    query_sdpa,
                    key_sdpa,
                    value_sdpa,
                    attn_mask=causal_bias,
                    dropout_p=0.0,
                    scale=attn.scaling,
                    is_causal=False,
                    enable_gqa=(attn.num_key_value_groups > 1),
                )
        except RuntimeError as exc:
            raise RuntimeError(
                f"layer {attn.layer_idx}: forced memory-efficient SDPA failed "
                f"for chunk q_len={q_len}, kv_len={kv_len}, "
                f"Q/K/V dtype={query_sdpa.dtype}, "
                f"shapes={tuple(query_sdpa.shape)}/"
                f"{tuple(key_sdpa.shape)}/{tuple(value_sdpa.shape)}, "
                f"K stride={key_sdpa.stride()}, V stride={value_sdpa.stride()}. "
                "No math fallback was allowed; inspect the preceding PyTorch "
                "warning for the remaining kernel constraint."
            ) from exc
        finally:
            del query_sdpa
            del key_sdpa
            del value_sdpa
            del causal_bias

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        gate_fp16 = torch.sigmoid(gate).to(torch.float16)
        del gate

        attn_output = attn_output * gate_fp16
        del gate_fp16

        attn_output = attn.o_proj(attn_output)
        if attn_output.dtype != input_dtype:
            attn_output = attn_output.to(input_dtype)

        return attn_output, None

    def save_model(self, target_dir):
        if hasattr(self.model, "_gguf_q8_reader"):
            raise RuntimeError(
                "This inference-only Q8 runner keeps Q8_0 weights in the source GGUF; "
                "save_pretrained() would not serialize those packed weights."
            )
        self.model.save_pretrained(target_dir)

    @staticmethod
    def _move_cache_to(cache, device):
        """
        Move every tensor stored by a HF Cache to `device`.

        Qwen3.8/Qwen3.5 hybrid caches are not plain K/V caches:
          * full-attention layers store keys/values
          * linear-attention layers store conv_states/recurrent_states

        Moving all tensor attributes keeps the loaded cache directly usable as
        `past_key_values` by the normal whole-model forward.
        """
        device = torch.device(device)

        def move_value(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, list):
                return [move_value(v) for v in value]
            if isinstance(value, tuple):
                return tuple(move_value(v) for v in value)
            if isinstance(value, dict):
                return {k: move_value(v) for k, v in value.items()}
            return value

        if hasattr(cache, "layers"):
            for cache_layer in cache.layers:
                for name, value in list(vars(cache_layer).items()):
                    setattr(cache_layer, name, move_value(value))

                # Dynamic cache layers remember the device used at lazy init.
                # Keep that metadata consistent with the moved tensors.
                if hasattr(cache_layer, "device") and cache_layer.device is not None:
                    cache_layer.device = device

        return cache

    @staticmethod
    def _move_cache_layer_to(cache, layer_idx, device):
        """Move only one cache layer's tensor state to `device`."""
        device = torch.device(device)
        cache_layer = cache.layers[layer_idx]

        def move_value(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, list):
                return [move_value(v) for v in value]
            if isinstance(value, tuple):
                return tuple(move_value(v) for v in value)
            if isinstance(value, dict):
                return {k: move_value(v) for k, v in value.items()}
            return value

        for name, value in list(vars(cache_layer).items()):
            setattr(cache_layer, name, move_value(value))

        # HF cache layers may remember the device used for lazy/state init.
        if hasattr(cache_layer, "device") and cache_layer.device is not None:
            cache_layer.device = device

        return cache_layer

    def save_kv_cache(self, cache, target_file):
        """
        Save a complete Qwen3.8 inference cache.

        The serialized object is the actual HF DynamicCache, so after loading it
        can be passed directly to a normal, non-layered model forward as
        `past_key_values=cache`.

        NOTE: torch serialization pickles the cache class. Restore it with the
        same (or API-compatible) Transformers version; this code targets 5.9.0.
        """
        # Persist cache on CPU. This keeps serialization independent of GPU
        # residency and avoids retaining the full cache in VRAM just for saving.
        self._move_cache_to(cache, "cpu")

        target_file = Path(target_file)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, target_file)

    def load_kv_cache(self, source_file, device="cpu"):
        """
        Restore a cache saved by save_kv_cache().

        By default the cache is restored on CPU to minimize VRAM residency.
        For a conventional whole-model CUDA forward, explicitly load with
        device="cuda". The layer-by-layer path can instead move only the current
        cache slot to CUDA and immediately offload it again.
        """
        cache = torch.load(
            source_file,
            map_location=device,
            weights_only=False,
        )
        if not isinstance(cache, DynamicCache):
            raise TypeError(
                f"Expected transformers DynamicCache, got {type(cache).__name__}"
            )

        expected_layers = len(self.model.model.layers)
        if len(cache.layers) != expected_layers:
            raise ValueError(
                f"Cache has {len(cache.layers)} layers, model has {expected_layers}"
            )

        return self._move_cache_to(cache, device)

    @staticmethod
    def _validate_complete_cache(cache, config, expected_seq_len):
        """
        Validate the current Transformers hybrid cache layout.

        Full-attention cache layers expose `keys` / `values` as tensors.

        Current LinearAttentionLayer stores its states in dictionaries keyed by
        state index (normally state 0 for Qwen3.5/Qwen3.8):
            conv_states      = {0: Tensor | None}
            recurrent_states = {0: Tensor | None}
        Older implementations may expose the tensor directly, so accept both.
        """
        if len(cache.layers) != len(config.layer_types):
            raise RuntimeError(
                f"Cache layer count mismatch: "
                f"{len(cache.layers)} != {len(config.layer_types)}"
            )

        errors = []

        def get_state_tensor(value, state_idx=0):
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, dict):
                return value.get(state_idx)
            if isinstance(value, (list, tuple)):
                if 0 <= state_idx < len(value):
                    return value[state_idx]
            return None

        for i, (cache_layer, layer_type) in enumerate(
            zip(cache.layers, config.layer_types)
        ):
            if layer_type == "full_attention":
                keys = getattr(cache_layer, "keys", None)
                values = getattr(cache_layer, "values", None)

                if not isinstance(keys, torch.Tensor) or keys.numel() == 0:
                    errors.append(f"layer {i}: full_attention keys missing/empty")

                if not isinstance(values, torch.Tensor) or values.numel() == 0:
                    errors.append(f"layer {i}: full_attention values missing/empty")

            elif layer_type == "linear_attention":
                conv_states = getattr(cache_layer, "conv_states", None)
                recurrent_states = getattr(cache_layer, "recurrent_states", None)

                conv_state = get_state_tensor(conv_states, 0)
                recurrent_state = get_state_tensor(recurrent_states, 0)

                if not isinstance(conv_state, torch.Tensor) or conv_state.numel() == 0:
                    errors.append(
                        f"layer {i}: linear_attention conv_state[0] missing/empty"
                    )

                if (
                    not isinstance(recurrent_state, torch.Tensor)
                    or recurrent_state.numel() == 0
                ):
                    errors.append(
                        f"layer {i}: linear_attention recurrent_state[0] missing/empty"
                    )

                # Current HF cache tracks explicit initialization flags too.
                conv_init = getattr(cache_layer, "is_conv_states_initialized", None)
                if isinstance(conv_init, dict) and not conv_init.get(0, False):
                    errors.append(
                        f"layer {i}: linear_attention conv_state[0] not initialized"
                    )

                recurrent_init = getattr(
                    cache_layer, "is_recurrent_states_initialized", None
                )
                if (
                    isinstance(recurrent_init, dict)
                    and not recurrent_init.get(0, False)
                ):
                    errors.append(
                        f"layer {i}: linear_attention recurrent_state[0] not initialized"
                    )

            else:
                errors.append(
                    f"layer {i}: unexpected layer type {layer_type!r}"
                )

        seq_len = cache.get_seq_length()

        if seq_len != expected_seq_len:
            errors.append(
                f"cache seq_len={seq_len}, expected={expected_seq_len}"
            )

        if errors:
            raise RuntimeError(
                "Incomplete KV cache:\n  " + "\n  ".join(errors)
            )

    @torch.no_grad()
    def build_kv_cache(self, e, verbose=False):
        """
        Build a normal HF-compatible Qwen3.8/Qwen3.5 hybrid cache while keeping
        everything in host RAM unless it is actively needed by the current layer.

        Persistent between layers (CPU):
          * hidden_states
          * position ids
          * RoPE position embeddings
          * DeltaNet recurrent attention mask (full attention uses mask=None)
          * all completed cache slots

        Temporary on CUDA:
          attention phase:
            * input_layernorm
            * current token mixer (DeltaNet or full attention)
            * cache.layers[i]
            * for DeltaNet: one sequence chunk of hidden state / mask at a time
            * for full attention: one query chunk + persistent FP16 K/V workspace
            * RoPE position data only for full-attention layers

          FFN phase:
            * post_attention_layernorm
            * MLP
            * one token chunk of hidden state at a time

        DeltaNet prefill is evaluated sequentially in `linear_attn_chunk_tokens`
        chunks. Full attention is also evaluated sequentially in
        `full_attn_chunk_tokens` query chunks using one fixed FP16 K/V CUDA
        workspace while the final BF16 cache is filled directly in CPU RAM.
        The cache slot is evacuated immediately after
        the attention phase. The attention output returns to CPU before the MLP
        starts. The MLP is evaluated in `mlp_chunk_tokens` sequence chunks.
        """
        model = self.model
        core = model.model

        # Keep all setup tensors on CPU. In the Q8 GGML embedding wrapper the
        # matrix operation still executes through GGML CUDA internally, but the
        # returned PyTorch tensor follows input_ids.device, so CPU IDs give us a
        # CPU embedding result without permanently moving embed_tokens to CUDA.
        input_ids_cpu = torch.tensor(
            e["input_ids"], dtype=torch.long
        ).unsqueeze(0)
        attention_mask_cpu = torch.tensor(
            e["attention_mask"], dtype=torch.long
        ).unsqueeze(0)

        t0 = time.perf_counter()

        # Measure setup independently from decoder-layer phases.
        torch.cuda.reset_peak_memory_stats("cuda")

        input_embeds_cpu = core.embed_tokens(input_ids_cpu)
        if input_embeds_cpu.device.type != "cpu":
            raise RuntimeError(
                f"embed_tokens unexpectedly returned {input_embeds_cpu.device}"
            )

        seq_len = input_ids_cpu.shape[1]
        del input_ids_cpu

        # RoPE preparation is cheap and can run entirely in RAM.
        position_ids_cpu = torch.arange(
            seq_len,
            device="cpu",
            dtype=torch.long,
        ).unsqueeze(0)

        qwen_position_ids_cpu = position_ids_cpu[None, ...].expand(
            4, input_embeds_cpu.shape[0], -1
        )

        text_position_ids_cpu = qwen_position_ids_cpu[0]
        rope_position_ids_cpu = qwen_position_ids_cpu[1:]

        position_embeddings_cpu = [
            x.detach()
            for x in core.rotary_emb(
                input_embeds_cpu,
                rope_position_ids_cpu,
            )
        ]

        # One shared full-model cache. Empty/completed slots remain CPU-backed.
        cache = DynamicCache(config=core.config)

        # Flash SDPA + native GQA requires attention_mask=None. For our current
        # batch-1 KV prefill this is correct only when there is no padding.
        #
        # Qwen3.5 attention is causal (`self.is_causal=True`), and Transformers'
        # SDPA integration derives `is_causal=True` from q_len > 1 together with
        # attention_mask=None. Therefore we do NOT build/copy a 4-D causal mask
        # for full-attention layers.
        has_padding = not bool(attention_mask_cpu.bool().all().item())
        if has_padding:
            raise RuntimeError(
                "Forced Flash SDPA prefill currently requires an unpadded "
                "batch (all attention_mask values must be 1). "
                "An explicit padding mask would disable the native Flash/GQA "
                "path we are testing."
            )

        # DeltaNet still consumes its recurrent 2-D mask, so build only that.
        mask_kwargs = {
            "config": core.config,
            "inputs_embeds": input_embeds_cpu,
            "attention_mask": attention_mask_cpu,
            "past_key_values": cache,
            "position_ids": text_position_ids_cpu,
        }

        linear_attention_mask = create_recurrent_attention_mask(
            **mask_kwargs
        )

        attention_masks_cpu = {
            "full_attention": None,
            "linear_attention": linear_attention_mask,
        }

        # Drop setup references. No full-attention causal mask exists anymore.
        del mask_kwargs
        del linear_attention_mask
        del attention_mask_cpu
        del position_ids_cpu
        del qwen_position_ids_cpu
        del rope_position_ids_cpu

        # Hidden state is already CPU-resident and stays there between layers.
        hidden_states_cpu = input_embeds_cpu.detach()
        del input_embeds_cpu

        if verbose:
            torch.cuda.synchronize()
            allocated, reserved = _cuda_mem_mb()
            peak_allocated, peak_reserved = _cuda_peak_mem_mb()
            ram = _ram_used_mb()
            print(
                f"... KV setup complete: "
                f"GPU-allocated={allocated:.2f} MB, "
                f"GPU-reserved={reserved:.2f} MB, "
                f"GPU-peak-allocated={peak_allocated:.2f} MB, "
                f"GPU-peak-reserved={peak_reserved:.2f} MB, "
                f"RAM={ram:.2f} MB; model/setup/activations/masks=CPU"
            )

        for i, layer in enumerate(core.layers):
            tb = time.perf_counter()
            layer_type = core.config.layer_types[i]

            # ------------------------------------------------------------------
            # Phase 1: input norm + token mixer only.
            #
            # Do NOT move the complete decoder layer to CUDA. Only the modules
            # participating in the token-mixing half of this block are staged.
            # The cache slot is needed only here, so it returns to CPU before
            # the FFN phase starts.
            # ------------------------------------------------------------------
            token_mixer = (
                layer.linear_attn
                if layer_type == "linear_attention"
                else layer.self_attn
            )

            # Track the true peak of the attention/DeltaNet half independently.
            torch.cuda.reset_peak_memory_stats("cuda")
            if verbose:
                _log_cuda_stage(f"block {i} ATTENTION baseline")

            norm_staged = False
            mixer_staged = False
            cache_staged = False

            hidden_states_cuda = None
            normed_states_cuda = None
            mixer_output_cuda = None
            attn_aux_cuda = None
            attention_hidden_cpu = None
            layer_attention_mask_cuda = None
            layer_position_ids_cuda = None
            layer_position_embeddings_cuda = None

            try:
                mu._tmp_module_to(layer.input_layernorm, "cuda")
                norm_staged = True

                mu._tmp_module_to(token_mixer, "cuda")
                mixer_staged = True

                self._move_cache_layer_to(cache, i, "cuda")
                cache_staged = True


                if verbose:
                    _log_cuda_stage(
                        f"block {i} ATTENTION modules+cache staged"
                    )

                if layer_type == "linear_attention":
                    # ----------------------------------------------------------
                    # DeltaNet outer prefill chunking.
                    #
                    # Qwen3.5's GatedDeltaNet is recurrent: every call consumes
                    # the previous recurrent/conv state from this cache layer and
                    # writes the new state back. Keeping cache.layers[i] on CUDA
                    # across these chunks therefore gives a true sequential
                    # prefill while bounding temporary activation memory.
                    #
                    # The fallback delta-rule kernel itself chunks at 64 tokens.
                    # We enforce an outer chunk size divisible by 64 so outer
                    # boundaries align with the same internal recurrence grouping.
                    # ----------------------------------------------------------
                    attention_hidden_cpu = torch.empty_like(
                        hidden_states_cpu,
                        device="cpu",
                    )

                    seq_tokens = hidden_states_cpu.shape[1]
                    chunk_tokens = min(
                        self.linear_attn_chunk_tokens,
                        seq_tokens,
                    )
                    linear_mask_cpu = attention_masks_cpu["linear_attention"]

                    if isinstance(linear_mask_cpu, torch.Tensor):
                        if linear_mask_cpu.ndim != 2:
                            raise RuntimeError(
                                "Expected Qwen3.5 recurrent attention mask to be "
                                f"2D [batch, seq], got shape "
                                f"{tuple(linear_mask_cpu.shape)}"
                            )
                        if linear_mask_cpu.shape[1] != seq_tokens:
                            raise RuntimeError(
                                "Linear-attention mask length mismatch: "
                                f"{linear_mask_cpu.shape[1]} != {seq_tokens}"
                            )

                    for chunk_start in range(0, seq_tokens, chunk_tokens):
                        chunk_end = min(
                            chunk_start + chunk_tokens,
                            seq_tokens,
                        )

                        chunk_hidden_cuda = None
                        chunk_normed_cuda = None
                        chunk_mask_cuda = None
                        chunk_mixer_cuda = None
                        chunk_output_cuda = None

                        try:
                            chunk_hidden_cuda = hidden_states_cpu[
                                :, chunk_start:chunk_end, :
                            ].to("cuda")

                            chunk_normed_cuda = layer.input_layernorm(
                                chunk_hidden_cuda
                            )

                            if isinstance(linear_mask_cpu, torch.Tensor):
                                chunk_mask_cuda = linear_mask_cpu[
                                    :, chunk_start:chunk_end
                                ].to("cuda")
                            else:
                                chunk_mask_cuda = linear_mask_cpu

                            chunk_mixer_cuda = layer.linear_attn(
                                hidden_states=chunk_normed_cuda,
                                cache_params=cache,
                                attention_mask=chunk_mask_cuda,
                            )

                            chunk_output_cuda = (
                                chunk_hidden_cuda + chunk_mixer_cuda
                            )

                            attention_hidden_cpu[
                                :, chunk_start:chunk_end, :
                            ].copy_(chunk_output_cuda.to("cpu"))

                            if verbose:
                                _log_cuda_stage(
                                    f"block {i} ATTENTION DeltaNet chunk "
                                    f"{chunk_start}:{chunk_end} computed"
                                )

                        finally:
                            # Do not let any chunk-local CUDA reference survive
                            # into the next chunk.
                            del chunk_hidden_cuda
                            del chunk_normed_cuda
                            del chunk_mask_cuda
                            del chunk_mixer_cuda
                            del chunk_output_cuda

                    del linear_mask_cpu

                elif layer_type == "full_attention":
                    # ----------------------------------------------------------
                    # Full-attention sequential prefill with persistent workspace.
                    #
                    # GPU:
                    #   one fixed FP16 K workspace + one fixed FP16 V workspace
                    #   for this layer, reused by every query chunk.
                    #
                    # CPU:
                    #   final BF16 K/V cache tensors are preallocated once and
                    #   filled chunk-by-chunk. DynamicCache.update() is NOT used
                    #   here because DynamicLayer grows via torch.cat(), which
                    #   would repeatedly reallocate/copy the accumulated prefix.
                    # ----------------------------------------------------------
                    attention_hidden_cpu = torch.empty_like(
                        hidden_states_cpu,
                        device="cpu",
                    )

                    seq_tokens = hidden_states_cpu.shape[1]
                    chunk_tokens = min(
                        self.full_attn_chunk_tokens,
                        seq_tokens,
                    )

                    attn = layer.self_attn
                    batch_size = hidden_states_cpu.shape[0]
                    kv_heads = int(attn.config.num_key_value_heads)
                    head_dim = int(attn.head_dim)

                    workspace_shape = (
                        batch_size,
                        kv_heads,
                        seq_tokens,
                        head_dim,
                    )

                    # Final HF cache backing tensors: native BF16 in RAM.
                    key_cache_cpu = torch.empty(
                        workspace_shape,
                        dtype=hidden_states_cpu.dtype,
                        device="cpu",
                    )
                    value_cache_cpu = torch.empty(
                        workspace_shape,
                        dtype=hidden_states_cpu.dtype,
                        device="cpu",
                    )

                    # SDPA workspace: allocated ONCE for the whole layer.
                    key_workspace_fp16 = torch.empty(
                        workspace_shape,
                        dtype=torch.float16,
                        device="cuda",
                    )
                    value_workspace_fp16 = torch.empty(
                        workspace_shape,
                        dtype=torch.float16,
                        device="cuda",
                    )

                    if verbose:
                        workspace_mb = (
                            key_workspace_fp16.numel()
                            * key_workspace_fp16.element_size()
                            * 2
                            / 1024**2
                        )
                        print(
                            f"... block {i} full-attn persistent FP16 K/V "
                            f"workspace allocated once: {workspace_mb:.2f} MB, "
                            f"shape={workspace_shape}"
                        )
                        _log_cuda_stage(
                            f"block {i} ATTENTION full workspace ready"
                        )

                    cos_cpu, sin_cpu = position_embeddings_cpu

                    for chunk_start in range(0, seq_tokens, chunk_tokens):
                        chunk_end = min(
                            chunk_start + chunk_tokens,
                            seq_tokens,
                        )

                        chunk_hidden_cuda = None
                        chunk_normed_cuda = None
                        chunk_cos_cuda = None
                        chunk_sin_cuda = None
                        chunk_mixer_cuda = None
                        chunk_output_cuda = None
                        chunk_attn_aux_cuda = None

                        try:
                            chunk_hidden_cuda = hidden_states_cpu[
                                :, chunk_start:chunk_end, :
                            ].to("cuda")

                            chunk_normed_cuda = layer.input_layernorm(
                                chunk_hidden_cuda
                            )

                            chunk_cos_cuda = cos_cpu[
                                :, chunk_start:chunk_end, :
                            ].to("cuda")
                            chunk_sin_cuda = sin_cpu[
                                :, chunk_start:chunk_end, :
                            ].to("cuda")

                            (
                                chunk_mixer_cuda,
                                chunk_attn_aux_cuda,
                            ) = self._full_attention_chunk_sm75_efficient_fp16(
                                attn,
                                chunk_normed_cuda,
                                (chunk_cos_cuda, chunk_sin_cuda),
                                key_workspace_fp16,
                                value_workspace_fp16,
                                key_cache_cpu,
                                value_cache_cpu,
                                chunk_start,
                            )

                            chunk_output_cuda = (
                                chunk_hidden_cuda + chunk_mixer_cuda
                            )

                            attention_hidden_cpu[
                                :, chunk_start:chunk_end, :
                            ].copy_(chunk_output_cuda.to("cpu"))

                            if verbose:
                                _log_cuda_stage(
                                    f"block {i} ATTENTION full chunk "
                                    f"{chunk_start}:{chunk_end} computed "
                                    f"(kv_len={chunk_end}, persistent FP16 workspace)"
                                )

                        finally:
                            del chunk_hidden_cuda
                            del chunk_normed_cuda
                            del chunk_cos_cuda
                            del chunk_sin_cuda
                            del chunk_mixer_cuda
                            del chunk_output_cuda
                            del chunk_attn_aux_cuda

                    # Install the completed CPU tensors directly into the HF
                    # DynamicLayer. Current DynamicLayer sequence length is
                    # derived from keys.shape[-2], so no GPU-grown cache is
                    # needed during prefill.
                    cache_layer = cache.layers[i]

                    if (
                        getattr(cache_layer, "is_initialized", False)
                        and isinstance(getattr(cache_layer, "keys", None), torch.Tensor)
                        and cache_layer.keys.numel() != 0
                    ):
                        raise RuntimeError(
                            f"full-attention cache layer {i} unexpectedly already "
                            "contained K/V before direct CPU cache installation"
                        )

                    cache_layer.keys = key_cache_cpu
                    cache_layer.values = value_cache_cpu
                    cache_layer.is_initialized = True
                    cache_layer.dtype = key_cache_cpu.dtype
                    cache_layer.device = torch.device("cpu")

                    # The cache now owns these CPU tensors; drop only local refs.
                    del key_cache_cpu
                    del value_cache_cpu

                    # The FP16 workspace is no longer needed after this layer.
                    del key_workspace_fp16
                    del value_workspace_fp16
                    del cos_cpu
                    del sin_cpu

                    if verbose:
                        _log_cuda_stage(
                            f"block {i} ATTENTION full workspace released"
                        )

                else:
                    raise RuntimeError(
                        f"Unsupported layer type {layer_type!r} at layer {i}"
                    )

            finally:
                # Cleanup is exception-safe even if staging or compute failed
                # halfway through the phase.
                if cache_staged:
                    self._move_cache_layer_to(cache, i, "cpu")
                if mixer_staged:
                    mu._tmp_module_reset(token_mixer)
                if norm_staged:
                    mu._tmp_module_reset(layer.input_layernorm)

                # Delete every CUDA local created by phase 1. In particular, do
                # not leave residual aliases or ignored attention outputs alive.
                del hidden_states_cuda
                del normed_states_cuda
                del mixer_output_cuda
                del attn_aux_cuda
                del layer_attention_mask_cuda
                del layer_position_ids_cuda
                del layer_position_embeddings_cuda

            # Verify phase-1 modules and cache really left CUDA. This scans plain
            # tensor attrs as well as registered Parameters/Buffers.
            leftover_cuda = []
            leftover_cuda.extend(
                _module_cuda_tensors(layer.input_layernorm, "input_layernorm")
            )
            leftover_cuda.extend(
                _module_cuda_tensors(token_mixer, "token_mixer")
            )
            if leftover_cuda:
                preview = ", ".join(item[0] for item in leftover_cuda[:20])
                raise RuntimeError(
                    f"layer {i} attention phase left CUDA tensors after reset: "
                    + preview
                )

            if verbose:
                torch.cuda.synchronize()
                allocated, reserved = _cuda_mem_mb()
                peak_allocated, peak_reserved = _cuda_peak_mem_mb()
                print(
                    f"... block {i} ATTENTION complete/cleaned: "
                    f"GPU-allocated={allocated:.2f} MB, "
                    f"GPU-reserved={reserved:.2f} MB, "
                    f"GPU-peak-allocated={peak_allocated:.2f} MB, "
                    f"GPU-peak-reserved={peak_reserved:.2f} MB"
                )

            if attention_hidden_cpu is None:
                raise RuntimeError(
                    f"layer {i} token-mixer phase did not produce hidden states"
                )

            # ------------------------------------------------------------------
            # Phase 2: post-attention norm + MLP only.
            #
            # MLP is token-independent, so process the sequence in chunks to cap
            # gate_proj/up_proj intermediate activation memory. This preserves
            # exact per-token MLP semantics; only execution order is chunked.
            # ------------------------------------------------------------------
            # FFN gets its own peak window, independent of attention.
            torch.cuda.reset_peak_memory_stats("cuda")
            if verbose:
                _log_cuda_stage(f"block {i} FFN baseline")

            post_norm_staged = False
            mlp_staged = False

            next_hidden_states_cpu = torch.empty_like(
                attention_hidden_cpu,
                device="cpu",
            )

            try:
                mu._tmp_module_to(layer.post_attention_layernorm, "cuda")
                post_norm_staged = True

                mu._tmp_module_to(layer.mlp, "cuda")
                mlp_staged = True

                if verbose:
                    _log_cuda_stage(
                        f"block {i} FFN modules staged"
                    )

                seq_tokens = attention_hidden_cpu.shape[1]
                chunk_tokens = min(self.mlp_chunk_tokens, seq_tokens)

                for chunk_start in range(0, seq_tokens, chunk_tokens):
                    chunk_end = min(chunk_start + chunk_tokens, seq_tokens)

                    chunk_cuda = None
                    normed_chunk_cuda = None
                    mlp_chunk_cuda = None
                    output_chunk_cuda = None
                    try:
                        chunk_cuda = attention_hidden_cpu[
                            :, chunk_start:chunk_end, :
                        ].to("cuda")

                        # Exact decoder FFN residual order. Avoid a separate
                        # residual alias: chunk_cuda itself is the residual.
                        normed_chunk_cuda = layer.post_attention_layernorm(
                            chunk_cuda
                        )
                        mlp_chunk_cuda = layer.mlp(normed_chunk_cuda)
                        output_chunk_cuda = chunk_cuda + mlp_chunk_cuda

                        next_hidden_states_cpu[
                            :, chunk_start:chunk_end, :
                        ].copy_(output_chunk_cuda.to("cpu"))

                        if verbose:
                            _log_cuda_stage(
                                f"block {i} FFN chunk "
                                f"{chunk_start}:{chunk_end} computed"
                            )
                    finally:
                        del chunk_cuda
                        del normed_chunk_cuda
                        del mlp_chunk_cuda
                        del output_chunk_cuda

            finally:
                if mlp_staged:
                    mu._tmp_module_reset(layer.mlp)
                if post_norm_staged:
                    mu._tmp_module_reset(layer.post_attention_layernorm)

            # Verify phase-2 modules also evacuated correctly, including plain
            # tensor attributes in custom modules.
            leftover_cuda = []
            leftover_cuda.extend(
                _module_cuda_tensors(
                    layer.post_attention_layernorm,
                    "post_attention_layernorm",
                )
            )
            leftover_cuda.extend(
                _module_cuda_tensors(layer.mlp, "mlp")
            )
            if leftover_cuda:
                preview = ", ".join(item[0] for item in leftover_cuda[:20])
                raise RuntimeError(
                    f"layer {i} FFN phase left CUDA tensors after reset: "
                    + preview
                )

            hidden_states_cpu = next_hidden_states_cpu
            del next_hidden_states_cpu
            del attention_hidden_cpu

            if verbose:
                torch.cuda.synchronize()
                allocated, reserved = _cuda_mem_mb()
                peak_allocated, peak_reserved = _cuda_peak_mem_mb()
                print(
                    f"... block {i} FFN complete/cleaned: "
                    f"GPU-allocated={allocated:.2f} MB, "
                    f"GPU-reserved={reserved:.2f} MB, "
                    f"GPU-peak-allocated={peak_allocated:.2f} MB, "
                    f"GPU-peak-reserved={peak_reserved:.2f} MB"
                )

            if verbose:
                mem = torch.cuda.memory_allocated("cuda") / 1024**2
                ram = _ram_used_mb()

                cache_desc = f"cache[{i}]=CPU"
                if layer_type == "linear_attention":
                    cache_layer = cache.layers[i]
                    conv_states = getattr(cache_layer, "conv_states", {})
                    recurrent_states = getattr(
                        cache_layer, "recurrent_states", {}
                    )
                    conv0 = (
                        conv_states.get(0)
                        if isinstance(conv_states, dict)
                        else conv_states
                    )
                    rec0 = (
                        recurrent_states.get(0)
                        if isinstance(recurrent_states, dict)
                        else recurrent_states
                    )
                    conv_shape = (
                        tuple(conv0.shape)
                        if isinstance(conv0, torch.Tensor)
                        else None
                    )
                    rec_shape = (
                        tuple(rec0.shape)
                        if isinstance(rec0, torch.Tensor)
                        else None
                    )
                    cache_desc += (
                        f", conv0={conv_shape}, recurrent0={rec_shape}"
                    )

                print(
                    f"... cache block {i} took "
                    f"{time.perf_counter() - tb:.6f}s, "
                    f"GPU={mem:.2f} MB, RAM={ram:.2f} MB, "
                    f"hidden=CPU, {cache_desc}, "
                    f"split=attn+ffn, "
                    f"linear_attn_chunk={self.linear_attn_chunk_tokens}, "
                    f"full_attn_chunk={self.full_attn_chunk_tokens}, "
                    f"mlp_chunk={self.mlp_chunk_tokens}"
                )

        if len(cache.layers) != len(core.layers):
            raise RuntimeError(
                f"Incomplete cache: {len(cache.layers)} cache layers for "
                f"{len(core.layers)} model layers"
            )

        cached_len = cache.get_seq_length()
        if cached_len != seq_len:
            raise RuntimeError(
                f"Unexpected cache length {cached_len}; expected {seq_len}"
            )

        self._validate_complete_cache(
            cache,
            core.config,
            seq_len,
        )

        # Strong invariant: every tensor nested anywhere in the returned cache
        # must be CPU-backed. LinearAttentionLayer stores state tensors inside
        # dictionaries, so a shallow top-level check is insufficient.
        def assert_cpu(value, path):
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu":
                    raise RuntimeError(
                        f"{path} unexpectedly left on {value.device}"
                    )
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    assert_cpu(item, f"{path}[{key!r}]")
                return
            if isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    assert_cpu(item, f"{path}[{idx}]")

        for i, cache_layer in enumerate(cache.layers):
            for name, value in vars(cache_layer).items():
                assert_cpu(value, f"cache[{i}].{name}")

        if hidden_states_cpu.device.type != "cpu":
            raise RuntimeError(
                f"hidden_states unexpectedly left on {hidden_states_cpu.device}"
            )

        if verbose:
            allocated, reserved = _cuda_mem_mb()
            peak_allocated, peak_reserved = _cuda_peak_mem_mb()
            ram = _ram_used_mb()
            print(
                f"... complete CPU-backed cache built: "
                f"layers={len(cache.layers)}, tokens={cached_len}, "
                f"time={time.perf_counter() - t0:.6f}s, "
                f"GPU-allocated={allocated:.2f} MB, "
                f"GPU-reserved={reserved:.2f} MB, "
                f"last-phase-peak-allocated={peak_allocated:.2f} MB, "
                f"last-phase-peak-reserved={peak_reserved:.2f} MB, "
                f"RAM={ram:.2f} MB"
            )

        return cache

    def build_and_save_kv_cache(self, e, target_file, verbose=False):
        cache = self.build_kv_cache(e, verbose=verbose)
        self.save_kv_cache(cache, target_file)
        return cache

    def __init__(
        self,
        model_dir,
        mlp_chunk_tokens=MLP_CHUNK_TOKENS,
        linear_attn_chunk_tokens=LINEAR_ATTN_CHUNK_TOKENS,
        full_attn_chunk_tokens=FULL_ATTN_CHUNK_TOKENS,
    ):
        self.model = self.__prep_model(model_dir)

        self.mlp_chunk_tokens = int(mlp_chunk_tokens)
        if self.mlp_chunk_tokens <= 0:
            raise ValueError(
                f"mlp_chunk_tokens must be > 0, got {self.mlp_chunk_tokens}"
            )

        self.linear_attn_chunk_tokens = int(linear_attn_chunk_tokens)
        if self.linear_attn_chunk_tokens <= 0:
            raise ValueError(
                "linear_attn_chunk_tokens must be > 0, got "
                f"{self.linear_attn_chunk_tokens}"
            )

        self.full_attn_chunk_tokens = int(full_attn_chunk_tokens)
        if self.full_attn_chunk_tokens <= 0:
            raise ValueError(
                "full_attn_chunk_tokens must be > 0, got "
                f"{self.full_attn_chunk_tokens}"
            )

        # HF's PyTorch fallback DeltaNet kernel internally uses 64-token chunks.
        # Align outer boundaries with those chunks to preserve the same grouping
        # as an unchunked prefill as closely as possible.
        if self.linear_attn_chunk_tokens % 64 != 0:
            raise ValueError(
                "linear_attn_chunk_tokens must be a multiple of 64 "
                f"(got {self.linear_attn_chunk_tokens})"
            )

        # Kept only for compatibility with older caller code; KV generation now
        # stages attention and FFN independently within each decoder layer.
        n_layers = len(self.model.model.layers)
        self.bindexes = [1] * n_layers

