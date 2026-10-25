from __future__ import annotations

import gc
import re
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load


@dataclass(frozen=True)
class Qwen35Layout:
    key_heads: int
    value_heads: int
    key_head_dim: int
    value_head_dim: int

    @classmethod
    def from_model(cls, model):
        candidates = [getattr(getattr(model, "model", None), "config", None),
                      getattr(getattr(model, "config", None), "text_config", None),
                      getattr(model, "config", None)]
        for config in candidates:
            if config is not None and all(hasattr(config, attr) for attr in (
                "linear_num_key_heads", "linear_num_value_heads",
                "linear_key_head_dim", "linear_value_head_dim",
            )):
                return cls(*(int(getattr(config, attr)) for attr in (
                    "linear_num_key_heads", "linear_num_value_heads",
                    "linear_key_head_dim", "linear_value_head_dim",
                )))
        raise ValueError("Cannot determine Qwen3.5 linear-attention head layout")

    def __post_init__(self):
        if (self.key_heads < 1 or self.value_heads < 1 or
                self.value_heads % self.key_heads or
                self.key_head_dim < 1 or self.value_head_dim < 1):
            raise ValueError(f"Invalid Gated DeltaNet head layout: {self}")

    @property
    def key_width(self):
        return self.key_heads * self.key_head_dim

    @property
    def value_width(self):
        return self.value_heads * self.value_head_dim

    @property
    def conv_width(self):
        return 2 * self.key_width + self.value_width

    def gguf_to_hf_index(self, head_dim: int) -> torch.Tensor:
        """For GGUF (tiled) index j, return its HF (grouped) source index."""
        ratio = self.value_heads // self.key_heads
        return (torch.arange(self.value_heads * head_dim, dtype=torch.long)
                .reshape(self.key_heads, ratio, head_dim)
                .permute(1, 0, 2).reshape(-1))

    def hf_to_gguf_index(self, head_dim: int) -> torch.Tensor:
        """For HF index i, return its GGUF (tiled) source index."""
        return torch.argsort(self.gguf_to_hf_index(head_dim))


def _is_delta(name: str) -> bool:
    return ".linear_attn." in name


def _is_plain_norm(name: str) -> bool:
    # Qwen3.5 ordinary RMSNorm expects (1 + parameter). The gated DeltaNet
    # output normalization expects *parameter*, without a +1. The GGUF
    # converter adds one to all the former and deliberately excludes the latter.
    return name.endswith("norm.weight") and not name.endswith(".linear_attn.norm.weight")


def _row_index(name: str, layout: Qwen35Layout) -> torch.Tensor | None:
    if not _is_delta(name):
        return None
    if name.endswith(".in_proj_qkv.weight"):
        qk = 2 * layout.key_width
        return torch.cat((torch.arange(qk, dtype=torch.long),
                          qk + layout.hf_to_gguf_index(layout.value_head_dim)))
    if name.endswith((".in_proj_z.weight", ".conv1d.weight")):
        idx = layout.hf_to_gguf_index(layout.value_head_dim)
        if name.endswith(".conv1d.weight"):
            qk = 2 * layout.key_width
            return torch.cat((torch.arange(qk, dtype=torch.long), qk + idx))
        return idx
    if name.endswith((".in_proj_b.weight", ".in_proj_a.weight",
                      ".A_log", ".dt_bias")):
        return layout.hf_to_gguf_index(1)
    return None


def q8_runtime_permutations(name: str, layout: Qwen35Layout) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """(input_index, output_index) applied around GGML's unchanged packed Q8 op.

    For out_proj: GGUF weights take tiled input; pass x_hf[GGUF->HF idx].
    For V-producing projections: GGUF outputs tiled; return y_gguf[HF->GGUF idx].
    This avoids dequantizing/repacking GGML Q8_0 blocks.
    """
    input_idx = None
    output_idx = _row_index(name, layout)
    if name.endswith(".linear_attn.out_proj.weight"):
        input_idx = layout.gguf_to_hf_index(layout.value_head_dim)
    return input_idx, output_idx


def restore_hf_non_q8(weights: torch.Tensor, gguf_name: str, hf_name: str, layout: Qwen35Layout) -> torch.Tensor:
    """Inverse GGUF conversion, operating on CPU PyTorch tensors.

    Should be applied exactly ONCE before materializing a HF parameter.
    """
    # llama.cpp converted A_log using -exp(A_log), the original parameter
    # needs log(-GGUF_A). Use F32 to avoid FP16 underflow/precision loss.
    if hf_name.endswith(".linear_attn.A_log"):
        a = weights.float()
        if not bool(torch.isfinite(a).all()) or not bool((a < 0).all()):
            raise ValueError(f"Invalid GGUF SSM A (must be finite negative): {gguf_name}")
        weights = (-a).log()
    if _is_plain_norm(hf_name):
        weights = weights.float() - 1.0
    row_index = _row_index(hf_name, layout)
    if row_index is not None:
        if weights.shape[0] != row_index.numel():
            raise ValueError(f"{gguf_name} -> {hf_name}: head reorder requires dimension 0 "
                             f"of {row_index.numel()}, got {tuple(weights.shape)}")
        weights = weights.index_select(0, row_index)
    if hf_name.endswith(".linear_attn.out_proj.weight"):
        idx = layout.hf_to_gguf_index(layout.value_head_dim)
        if weights.shape[1] != idx.numel():
            raise ValueError(f"{gguf_name} -> {hf_name}: expected input width {idx.numel()}")
        weights = weights.index_select(1, idx)
    return weights


_EXT = None
_BACKEND_READY = False


def _find_build_bin(llama_cpp_dir: Path) -> Path:
    candidates = [
        llama_cpp_dir / "build" / "bin",
        llama_cpp_dir / "build",
    ]
    for p in candidates:
        if (p / "libggml.so").exists() or list(p.glob("libggml*.so")):
            return p
    hits = list((llama_cpp_dir / "build").rglob("libggml.so"))
    if hits:
        return hits[0].parent
    raise FileNotFoundError(
        f"Could not find llama.cpp shared GGML libraries under {llama_cpp_dir / 'build'}"
    )


def load_extension(
    llama_cpp_dir: str | os.PathLike | None = None,
    device: int = 0,
    verbose: bool = False,
):
    """JIT-build the tiny PyTorch<->GGML bridge and initialize CUDA backend."""
    global _EXT, _BACKEND_READY

    if llama_cpp_dir is None:
        llama_cpp_dir = os.environ.get("LLAMA_CPP_DIR", "../llama.cpp-s")
    if not llama_cpp_dir:
        raise RuntimeError("Set LLAMA_CPP_DIR=/path/to/llama.cpp")

    llama_cpp_dir = Path(llama_cpp_dir).resolve()
    build_bin = _find_build_bin(llama_cpp_dir)

    if _EXT is None:
        src = Path(__file__).with_name("ggml_q8_cuda_ext.cpp")
        include_dir = llama_cpp_dir / "ggml" / "include"

        # Link only the core GGML shared library. CUDA is loaded dynamically from
        # build/bin by ggml_backend_load_all_from_path().
        libggml = build_bin / "libggml.so"
        if not libggml.exists():
            # Some builds put the core .so one level elsewhere.
            hits = list((llama_cpp_dir / "build").rglob("libggml.so"))
            if not hits:
                raise FileNotFoundError("libggml.so not found; build llama.cpp with BUILD_SHARED_LIBS=ON")
            libggml = hits[0]
            build_bin = libggml.parent

        # The host-side PyTorch CUDA-stream header includes CUDA runtime
        # headers even though this extension does not compile CUDA kernels.
        # With `with_cuda=False`, cpp_extension does not automatically add the
        # CUDA toolkit include directory, so resolve it explicitly.
        cuda_include_dirs = []
        cuda_candidates = [
            os.environ.get("CUDA_HOME"),
            os.environ.get("CUDA12_ROOT"),
            os.environ.get("CUDA_PATH"),
            "/opt/cuda-12.8", "/opt/cuda", "/usr/local/cuda",
        ]
        for candidate in cuda_candidates:
            if candidate and (Path(candidate) / "include" / "cuda_runtime_api.h").exists():
                cuda_include_dirs.append(str(Path(candidate) / "include"))
                break
        if not cuda_include_dirs:
            raise RuntimeError(
                "CUDA headers required by c10/cuda/CUDAStream.h were not found. "
                "Set CUDA_HOME or CUDA12_ROOT to your CUDA toolkit directory."
            )

        _EXT = load(
            name="ggml_q8_cuda_ext",
            sources=[str(src)],
            extra_include_paths=[str(include_dir), *cuda_include_dirs],
            extra_cflags=["-O3"],  # PyTorch already supplies the required -std=c++20
            extra_ldflags=[
                # The CUDA input path uses c10's current-stream API but does
                # not compile any custom CUDA kernels (with_cuda=False).
                "-lc10_cuda", "-ltorch_cuda",
                str(libggml),
                f"-Wl,-rpath,{build_bin}",
            ],
            with_cuda=False,
            verbose=verbose,
        )

    if not _BACKEND_READY:
        _EXT.init_backend(str(build_bin), int(device))
        _BACKEND_READY = True
        if verbose:
            print(f"[q8-ggml] backend={_EXT.backend_name()} libs={build_bin}")

    return _EXT


def _raw_u8_view(np_data: np.ndarray) -> torch.Tensor:
    """Zero-copy uint8 view over GGUFReader's mmap-backed tensor payload."""
    arr = np.asarray(np_data)
    raw = arr.view(np.uint8).reshape(-1)
    # GGUFReader is read-only; torch warns that the numpy array is non-writable.
    # We never write it, so this is deliberate.
    return torch.from_numpy(raw)


class GGMLQ8Linear(nn.Module):
    """
    nn.Linear-compatible inference module using original GGUF Q8_0 blocks.

    Q8_0 GGML CUDA projection without any additional CUDA-resident weights.

    Activations are copied/cast DIRECTLY into GGML's existing CUDA input
    buffer for the common (non-permuted) case. No CPU activation round-trip,
    additional GPU allocations, or altered matmul dtype are required.
    Permuted inputs retain the original CPU staging path to avoid an extra
    CUDA index_select allocation. Output handling retains the original CPU
    staging to avoid overlapping a second CUDA output with GGML buffers.

    GGML_Q8_DIRECT_INPUT=0 restores the original input path for A/B tests.
    """

    def __init__(self, packed: torch.Tensor, in_features: int, out_features: int, bias=None,
                 input_permutation=None, output_permutation=None):
        super().__init__()
        self._packed_q8 = packed
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.input_permutation = input_permutation
        self.output_permutation = output_permutation
        if bias is not None:
            self.register_buffer("bias", bias.detach().cpu(), persistent=False)
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, module: nn.Linear, packed: torch.Tensor):
        return cls(
            packed,
            module.in_features,
            module.out_features,
            module.bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ext = load_extension()
        shape = x.shape
        if shape[-1] != self.in_features:
            raise RuntimeError(f"Q8 linear expected last dim {self.in_features}, got {shape[-1]}")

        x2 = x.detach().reshape(-1, self.in_features)
        direct_input = (
            x2.is_cuda
            and self.input_permutation is None
            and os.environ.get("GGML_Q8_DIRECT_INPUT", "1").lower()
            not in ("0", "false")
        )
        if direct_input:
            # The extension casts into its *already allocated* F32 input
            # buffer. Do not make a separate GPU F32 activation here.
            x2 = x2.contiguous()
        else:
            x2 = x2.to(device="cpu", dtype=torch.float32).contiguous()
            if self.input_permutation is not None:
                x2 = x2.index_select(-1, self.input_permutation)

        y = ext.q8_mul_mat(
            self._packed_q8,
            self.in_features,
            self.out_features,
            x2,
        )
        y = y.reshape(*shape[:-1], self.out_features)
        if self.output_permutation is not None:
            y = y.index_select(-1, self.output_permutation)
        y = y.to(device=x.device, dtype=x.dtype)

        if self.bias is not None:
            y = y + self.bias.to(device=y.device, dtype=y.dtype)
        return y


class GGMLQ8Embedding(nn.Module):
    """nn.Embedding-compatible Q8_0 get_rows using GGML CUDA."""

    def __init__(self, packed: torch.Tensor, num_embeddings: int, embedding_dim: int,
                 output_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self._packed_q8 = packed
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.output_dtype = output_dtype

    @classmethod
    def from_embedding(cls, module: nn.Embedding, packed: torch.Tensor):
        return cls(packed, module.num_embeddings, module.embedding_dim,
                   output_dtype=module.weight.dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ext = load_extension()
        ids = input_ids.detach().reshape(-1).to(device="cpu", dtype=torch.int32).contiguous()
        y = ext.q8_get_rows(
            self._packed_q8,
            self.embedding_dim,
            self.num_embeddings,
            ids,
        )
        y = y.reshape(*input_ids.shape, self.embedding_dim)

        # Keep embeddings consistent with the configured HF compute dtype.
        target_device = input_ids.device
        target_dtype = self.output_dtype
        return y.to(device=target_device, dtype=target_dtype)


def _get_module(root: nn.Module, path: str) -> nn.Module:
    if not path:
        return root
    cur = root
    for piece in path.split("."):
        cur = getattr(cur, piece)
    return cur


def _replace_module(root: nn.Module, path: str, replacement: nn.Module) -> None:
    if "." in path:
        parent_path, attr = path.rsplit(".", 1)
        parent = _get_module(root, parent_path)
    else:
        parent, attr = root, path
    setattr(parent, attr, replacement)



def _is_extra_block_tensor(gguf_name: str, model: nn.Module) -> bool:
    """
    Return True for GGUF block tensors that are outside the normal HF causal-LM
    decoder trunk.

    Qwen3.5/Qwen3.8 GGUF files may contain appended MTP/NextN blocks after the
    main decoder stack. Example for the 27B dense model:
        normal trunk: blk.0 ... blk.63
        extra MTP:    blk.64

    AutoModelForCausalLM exposes only the normal trunk, so those extra tensors
    intentionally have no corresponding HF module and should be ignored by the
    strict Q8 replacement check.
    """
    match = re.match(r"^blk\.(\d+)\.", gguf_name)
    if match is None:
        return False

    try:
        trunk_layers = len(model.model.layers)
    except Exception:
        return False

    return int(match.group(1)) >= trunk_layers



def _augment_qwen35_gguf_mapping(model: nn.Module, gguf_to_hf: dict[str, str]) -> None:
    """
    Add Qwen3.5/Qwen3.8 mappings that current gguf-py does not infer.

    llama.cpp stores the Gated-DeltaNet time-step bias as:
        blk.<N>.ssm_dt.bias

    Hugging Face Qwen3.5 exposes the same tensor as:
        model.layers.<N>.linear_attn.dt_bias

    Only add entries for dt_bias parameters that actually exist in this model,
    so full-attention layers are never given a fake mapping.
    """
    pattern = re.compile(r"^model\.layers\.(\d+)\.linear_attn\.dt_bias$")
    for hf_name in model.state_dict().keys():
        match = pattern.match(hf_name)
        if match is None:
            continue

        layer_idx = int(match.group(1))
        gguf_to_hf.setdefault(
            f"blk.{layer_idx}.ssm_dt.bias",
            hf_name,
        )


def patch_model_q8_0_from_gguf(
    model: nn.Module,
    gguf_path: str | os.PathLike,
    *,
    strict: bool = True,
    verbose: bool = True,
):
    """
    Replace every mapped GGUF Q8_0 nn.Linear / nn.Embedding with a GGML-CUDA
    backed module while retaining the already-loaded HF modules for non-Q8 data.

    The GGUF numpy payloads are mmap-backed; the patched modules keep zero-copy
    torch.uint8 views of those packed bytes.
    """
    from gguf import GGUFReader
    from transformers.modeling_gguf_pytorch_utils import (
        TensorProcessor,
        get_gguf_hf_weights_map,
    )

    # Ensure the extension/backend is ready before discarding dense weights.
    load_extension(verbose=verbose)

    gguf_path = str(Path(gguf_path).resolve())
    reader = GGUFReader(gguf_path)

    processor = TensorProcessor(config=model.config.to_dict())
    gguf_to_hf = get_gguf_hf_weights_map(model, processor, model_type="qwen35")
    _augment_qwen35_gguf_mapping(model, gguf_to_hf)
    layout = Qwen35Layout.from_model(model)

    replaced = []
    unsupported = []
    unmapped = []
    ignored_extra = []

    for t in reader.tensors:
        if t.tensor_type.name != "Q8_0":
            continue

        hf_name = gguf_to_hf.get(t.name)
        if hf_name is None:
            if _is_extra_block_tensor(t.name, model):
                ignored_extra.append(t.name)
            else:
                unmapped.append(t.name)
            continue

        if not hf_name.endswith(".weight"):
            unsupported.append((t.name, hf_name, "not a weight"))
            continue

        module_path = hf_name[:-len(".weight")]
        try:
            module = _get_module(model, module_path)
        except Exception as exc:
            unsupported.append((t.name, hf_name, f"module lookup failed: {exc}"))
            continue

        packed = _raw_u8_view(t.data)

        if isinstance(module, nn.Linear):
            inp_perm, out_perm = q8_runtime_permutations(hf_name, layout)
            replacement = GGMLQ8Linear(
                packed, module.in_features, module.out_features, module.bias,
                input_permutation=inp_perm, output_permutation=out_perm,
            )
        elif isinstance(module, nn.Embedding):
            replacement = GGMLQ8Embedding.from_embedding(module, packed)
        else:
            unsupported.append((t.name, hf_name, type(module).__name__))
            continue

        _replace_module(model, module_path, replacement)
        replaced.append((t.name, hf_name, t.n_bytes))

    # Keep the reader alive: its np.memmap owns the packed tensor storage.
    model._gguf_q8_reader = reader
    model._gguf_q8_path = gguf_path

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        total = sum(n for _, _, n in replaced)
        print(
            f"[q8-ggml] replaced {len(replaced)} Q8_0 modules "
            f"({total / 1024**3:.2f} GiB packed)"
        )
        if ignored_extra:
            print(
                f"[q8-ggml] ignored {len(ignored_extra)} Q8 tensors from "
                f"extra MTP/NextN block(s) outside the {len(model.model.layers)}-layer trunk"
            )
        if unmapped:
            print(f"[q8-ggml] unmapped Q8 trunk tensors: {len(unmapped)}")
        if unsupported:
            print(f"[q8-ggml] unsupported mapped Q8 tensors: {len(unsupported)}")
            for item in unsupported[:20]:
                print("  ", item)

    if strict and (unsupported or unmapped):
        details = []
        if unsupported:
            details.append(f"unsupported={len(unsupported)}")
        if unmapped:
            details.append(f"unmapped={len(unmapped)}")
        raise RuntimeError(
            "Not all Q8_0 tensors were replaced (" + ", ".join(details) + "). "
            "Set Q8_GGML_STRICT=0 only if you intentionally accept partial parity."
        )

    return {
        "replaced": replaced,
        "unsupported": unsupported,
        "unmapped": unmapped,
        "ignored_extra": ignored_extra,
    }


def load_model_from_gguf_without_q8_dequant(
    gguf_path: str | os.PathLike,
    *,
    dtype: torch.dtype = torch.bfloat16,
    strict_q8: bool = True,
    verbose: bool = True,
):
    """
    Instantiate the HF text model on META, load only the non-Q8_0 GGUF tensors
    into ordinary PyTorch Parameters, and replace Q8_0 Linear/Embedding weights
    with GGML CUDA-backed modules.

    Unlike AutoModelForCausalLM.from_pretrained(..., gguf_file=...), this never
    dequantizes Q8_0 matrices into dense PyTorch weights.
    """
    from accelerate import init_empty_weights
    from gguf import GGUFReader, dequantize
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.modeling_gguf_pytorch_utils import (
        TensorProcessor,
        get_gguf_hf_weights_map,
    )

    load_extension(verbose=verbose)

    gguf_path = Path(gguf_path).resolve()
    model_dir = gguf_path.parent

    # Config/tokenizer metadata are parsed from GGUF only; no tensor conversion.
    config = AutoConfig.from_pretrained(
        str(model_dir),
        gguf_file=str(gguf_path),
        local_files_only=True,
        trust_remote_code=True,
    )

    # Keep ordinary buffers (RoPE metadata etc.) real, but create all parameters
    # on meta so the 27B dense model is never allocated in BF16 first.
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
        )

    reader = GGUFReader(str(gguf_path))
    processor = TensorProcessor(config=model.config.to_dict())
    gguf_to_hf = get_gguf_hf_weights_map(model, processor, model_type="qwen35")
    _augment_qwen35_gguf_mapping(model, gguf_to_hf)
    layout = Qwen35Layout.from_model(model)

    non_q8_state = {}
    replacements = []
    unsupported_q8 = []
    unmapped_q8 = []
    ignored_extra_q8 = []

    # First replace Q8 modules. This removes their meta Parameters from the model
    # before load_state_dict(assign=True) materializes the remaining parameters.
    for t in reader.tensors:
        if t.tensor_type.name != "Q8_0":
            continue

        hf_name = gguf_to_hf.get(t.name)
        if hf_name is None:
            if _is_extra_block_tensor(t.name, model):
                ignored_extra_q8.append(t.name)
            else:
                unmapped_q8.append(t.name)
            continue
        if not hf_name.endswith(".weight"):
            unsupported_q8.append((t.name, hf_name, "not a weight"))
            continue

        module_path = hf_name[:-len(".weight")]
        try:
            module = _get_module(model, module_path)
        except Exception as exc:
            unsupported_q8.append((t.name, hf_name, f"lookup failed: {exc}"))
            continue

        packed = _raw_u8_view(t.data)
        if isinstance(module, nn.Linear):
            # Keep the GGUF Q8_0 bytes intact; adapt V-head ordering at runtime.
            inp_perm, out_perm = q8_runtime_permutations(hf_name, layout)
            replacement = GGMLQ8Linear(
                packed, module.in_features, module.out_features, bias=None,
                input_permutation=inp_perm, output_permutation=out_perm,
            )
        elif isinstance(module, nn.Embedding):
            replacement = GGMLQ8Embedding(
                packed,
                module.num_embeddings,
                module.embedding_dim,
                output_dtype=dtype,
            )
        else:
            unsupported_q8.append((t.name, hf_name, type(module).__name__))
            continue

        _replace_module(model, module_path, replacement)
        replacements.append((t.name, hf_name, t.n_bytes))

    if verbose and ignored_extra_q8:
        extra_blocks = sorted({
            int(name.split(".")[1])
            for name in ignored_extra_q8
            if name.startswith("blk.")
        })
        print(
            f"[q8-ggml] ignored {len(ignored_extra_q8)} Q8 tensor(s) from "
            f"extra MTP/NextN block(s) {extra_blocks}; "
            f"HF causal trunk has {len(model.model.layers)} layers"
        )

    if strict_q8 and (unsupported_q8 or unmapped_q8):
        raise RuntimeError(
            "Q8_0 trunk replacement incomplete before materialization: "
            f"unsupported={len(unsupported_q8)}, unmapped={len(unmapped_q8)}. "
            f"First unsupported: {unsupported_q8[:5]}; first unmapped: {unmapped_q8[:5]}"
        )

    # Materialize only the tensors which were NOT Q8_0 in the GGUF.
    # This follows the same GGUF name mapping / tensor processor used by HF's
    # loader, but deliberately skips the expensive Q8 dequantization branch.
    for t in reader.tensors:
        if t.tensor_type.name == "Q8_0":
            continue

        mapped_name = gguf_to_hf.get(t.name)
        if mapped_name is None:
            continue

        weights = dequantize(t.data, t.tensor_type)
        result = processor.process(
            weights=weights,
            name=t.name,
            tensor_key_mapping=gguf_to_hf,
            parsed_parameters={"tensors": non_q8_state},
        )
        if result.name is None:
            continue

        target_name = gguf_to_hf.get(result.name)
        if target_name is None:
            continue

        tensor = torch.from_numpy(np.copy(result.weights))
        tensor = restore_hf_non_q8(tensor, t.name, target_name, layout)

        # Qwen3.5/Qwen3.8 DeltaNet depthwise Conv1d is stored in GGUF as
        # [channels, kernel_size], while Hugging Face nn.Conv1d expects
        # [channels, 1, kernel_size] because groups == channels.
        #
        # Keep this conversion deliberately narrow instead of generically
        # reshaping tensors: it applies only to linear_attn.conv1d.weight and
        # requires the element count / expected dimensions to match exactly.
        if target_name.endswith(".linear_attn.conv1d.weight"):
            target_tensor = model.state_dict().get(target_name)
            if target_tensor is None:
                raise RuntimeError(
                    f"Could not find HF target tensor for {target_name}"
                )

            expected_shape = tuple(target_tensor.shape)
            if (
                tensor.ndim == 2
                and len(expected_shape) == 3
                and expected_shape[1] == 1
                and tensor.shape[0] == expected_shape[0]
                and tensor.shape[1] == expected_shape[2]
            ):
                tensor = tensor.unsqueeze(1)
            elif tuple(tensor.shape) != expected_shape:
                raise RuntimeError(
                    f"Unexpected DeltaNet Conv1d shape for {target_name}: "
                    f"GGUF/HF-processed={tuple(tensor.shape)}, expected={expected_shape}"
                )

        if tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        non_q8_state[target_name] = tensor

    incompatible = model.load_state_dict(non_q8_state, strict=False, assign=True)

    # A bias belonging to a Q8 linear would have been removed when replacing the
    # module. Qwen3.5 text projections are bias=False, so this is expected. If a
    # future checkpoint has Q8 linear biases, fail visibly rather than silently.
    meta_params = [name for name, p in model.named_parameters() if p.is_meta]
    meta_buffers = [name for name, b in model.named_buffers() if getattr(b, "is_meta", False)]
    if meta_params or meta_buffers:
        raise RuntimeError(
            "Model still has unmaterialized META tensors. "
            f"params={meta_params[:20]} buffers={meta_buffers[:20]} "
            f"missing_keys={incompatible.missing_keys[:20]}"
        )

    model._gguf_q8_reader = reader
    model._gguf_q8_path = str(gguf_path)
    model._gguf_q8_replacements = replacements
    model.eval()

    if verbose:
        packed_gib = sum(n for _, _, n in replacements) / 1024**3
        print(
            f"[q8-ggml] META load complete: {len(replacements)} Q8_0 modules "
            f"kept packed ({packed_gib:.2f} GiB); "
            f"{len(non_q8_state)} non-Q8 tensors materialized"
        )
        if incompatible.unexpected_keys:
            print("[q8-ggml] unexpected non-Q8 keys:", incompatible.unexpected_keys[:20])
        # missing keys inside Q8 modules are gone from model.state_dict(), so any
        # remaining missing keys are worth surfacing.
        if incompatible.missing_keys:
            print("[q8-ggml] missing keys:", incompatible.missing_keys[:20])

    return model


