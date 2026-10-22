from __future__ import annotations

import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


# llama.cpp GGSQ sequence-state file format v3 (e.g. commit 19e28a2 / b11277).
# This project intentionally targets v3, not current v4.
LLAMA_STATE_SEQ_MAGIC = 0x67677371  # 'ggsq'
LLAMA_STATE_SEQ_VERSION = 3

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


@dataclass(frozen=True)
class GGMLFloatType:
    name: str
    enum: int
    torch_dtype: torch.dtype
    numpy_dtype: np.dtype
    itemsize: int


FLOAT_TYPES = {
    "f16": GGMLFloatType("f16", GGML_TYPE_F16, torch.float16, np.dtype("<f2"), 2),
    "f32": GGMLFloatType("f32", GGML_TYPE_F32, torch.float32, np.dtype("<f4"), 4),
}


@dataclass(frozen=True)
class Qwen38Spec:
    # Production values below describe Qwen/Qwen3.8-27B's text backbone exactly.
    n_layers: int = 64
    full_attention_interval: int = 4
    kv_heads: int = 4
    attention_head_dim: int = 256
    linear_conv_kernel_dim: int = 4
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        # Qwen3.8 pattern is 3 Gated DeltaNet layers, then 1 full-attention layer.
        return tuple(range(self.full_attention_interval - 1, self.n_layers, self.full_attention_interval))

    @property
    def recurrent_layers(self) -> tuple[int, ...]:
        full = set(self.full_attention_layers)
        return tuple(i for i in range(self.n_layers) if i not in full)

    @property
    def attention_k_width(self) -> int:
        return self.kv_heads * self.attention_head_dim

    @property
    def attention_v_width(self) -> int:
        return self.kv_heads * self.attention_head_dim

    @property
    def conv_channels(self) -> int:
        # Qwen3.5/3.8 Gated DeltaNet conv input is Q/K/V concatenated:
        # K-like channels twice + V-like channels once.
        return (
            2 * self.linear_num_key_heads * self.linear_key_head_dim
            + self.linear_num_value_heads * self.linear_value_head_dim
        )

    @property
    def recurrent_r_width(self) -> int:
        # llama.cpp stores only convolution history, not the current token.
        return (self.linear_conv_kernel_dim - 1) * self.conv_channels

    @property
    def recurrent_s_width(self) -> int:
        return self.linear_num_value_heads * self.linear_key_head_dim * self.linear_value_head_dim


QWEN38_27B = Qwen38Spec()


def _gguf_to_hf_value_index(spec: Qwen38Spec, head_dim: int) -> torch.Tensor:
    """GGUF tiled V-head order -> HF grouped heads, for cache export.
    """
    key_heads = spec.linear_num_key_heads
    value_heads = spec.linear_num_value_heads
    if key_heads < 1 or value_heads < 1 or value_heads % key_heads or head_dim < 1:
        raise ValueError("Invalid Qwen3.5 value/key head layout")
    return (torch.arange(value_heads * head_dim, dtype=torch.long)
            .reshape(key_heads, value_heads // key_heads, head_dim)
            .permute(1, 0, 2).reshape(-1))


def _u32(x: int) -> bytes:
    return struct.pack("<I", int(x))


def _i32(x: int) -> bytes:
    return struct.pack("<i", int(x))


def _u64(x: int) -> bytes:
    return struct.pack("<Q", int(x))


def attention_token_rows(
    t: torch.Tensor,
    *,
    typ: GGMLFloatType,
    expected_heads: int,
    expected_head_dim: int,
) -> np.ndarray:
    """HF [1,H,T,D] -> llama.cpp token-major [T,H*D]."""
    if t.ndim != 4 or t.shape[0] != 1:
        raise ValueError(f"expected attention cache [1,H,T,D], got {tuple(t.shape)}")
    if t.shape[1] != expected_heads or t.shape[3] != expected_head_dim:
        raise ValueError(
            f"unexpected attention cache shape {tuple(t.shape)}; expected "
            f"H={expected_heads}, D={expected_head_dim}"
        )
    t = t.detach().to(device="cpu", dtype=typ.torch_dtype)
    t = t[0].permute(1, 0, 2).contiguous()  # [T,H,D]
    return np.asarray(t.numpy().reshape(t.shape[0], -1), dtype=typ.numpy_dtype, order="C")


def recurrent_r_row(conv_state: torch.Tensor, *, spec: Qwen38Spec = QWEN38_27B) -> np.ndarray:
    """Map HF Gated DeltaNet conv state to llama.cpp R row.

    HF stores [B, conv_channels, kernel_size]. llama.cpp R is a GGML tensor whose
    logical shape is [kernel_size-1, conv_channels, cell], with dim0 fastest.
    Therefore one serialized cell row is GGUF[:, :, -history:].flatten(C).
    GGUF has tiled V-head order, while HF keeps grouped V-head order.
    """
    if conv_state.ndim != 3 or conv_state.shape[0] != 1:
        raise ValueError(f"expected conv state [1,C,K], got {tuple(conv_state.shape)}")
    if conv_state.shape[1] != spec.conv_channels:
        raise ValueError(
            f"unexpected conv channel count {conv_state.shape[1]}; expected {spec.conv_channels}"
        )
    if conv_state.shape[2] != spec.linear_conv_kernel_dim:
        raise ValueError(
            f"unexpected conv kernel state length {conv_state.shape[2]}; "
            f"expected {spec.linear_conv_kernel_dim}"
        )

    history = spec.linear_conv_kernel_dim - 1
    # [C,history] in C order => history (GGML dim0) is fastest.
    # Reorder the V channels (leave Q/K channels intact) from HF grouped to
    # GGUF tiled order, matching llama.cpp's _LinearAttentionVReorderBase.
    qk_width = 2 * spec.linear_num_key_heads * spec.linear_key_head_dim
    v_idx = _gguf_to_hf_value_index(spec, spec.linear_value_head_dim)
    idx = torch.cat((torch.arange(qk_width, dtype=torch.long), qk_width + v_idx))
    row = conv_state.detach()[0, :, -history:].to(device="cpu", dtype=torch.float32)
    row = row.index_select(0, idx).contiguous()
    a = np.asarray(row.numpy().reshape(-1), dtype=np.dtype("<f4"), order="C")
    if a.size != spec.recurrent_r_width:
        raise AssertionError(f"internal R width mismatch: {a.size} != {spec.recurrent_r_width}")
    return a


def recurrent_s_row(recurrent_state: torch.Tensor, *, spec: Qwen38Spec = QWEN38_27B) -> np.ndarray:
    """Map HF Gated DeltaNet recurrent state to llama.cpp S row.

    HF:       [B, value_heads, key_dim, value_dim]
    llama.cpp views S as GGML [key_dim, value_dim, value_heads, cell].

    GGML dim0 is fastest, so serialized order must be:
        GGUF_tiled_head -> value_dim -> key_dim
    which is HF[0][GGUF_to_HF_heads].permute(0, 2, 1) before flattening.
    """
    expected = (
        1,
        spec.linear_num_value_heads,
        spec.linear_key_head_dim,
        spec.linear_value_head_dim,
    )
    if tuple(recurrent_state.shape) != expected:
        raise ValueError(f"unexpected recurrent state {tuple(recurrent_state.shape)}; expected {expected}")

    head_idx = _gguf_to_hf_value_index(spec, 1)
    row = (
        recurrent_state.detach()[0]
        .to(device="cpu", dtype=torch.float32)
        .index_select(0, head_idx)
        .permute(0, 2, 1)  # [H,V,K], K becomes fastest after C flatten
        .contiguous()
    )
    a = np.asarray(row.numpy().reshape(-1), dtype=np.dtype("<f4"), order="C")
    if a.size != spec.recurrent_s_width:
        raise AssertionError(f"internal S width mismatch: {a.size} != {spec.recurrent_s_width}")
    return a


def _validate_layer_sets(
    attention: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    recurrent: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    spec: Qwen38Spec,
) -> None:
    got_a = tuple(sorted(attention))
    got_r = tuple(sorted(recurrent))
    if got_a != spec.full_attention_layers:
        raise ValueError(
            f"full-attention cache layer ids mismatch: got {got_a}, expected {spec.full_attention_layers}"
        )
    if got_r != spec.recurrent_layers:
        raise ValueError(
            f"recurrent cache layer ids mismatch: got {got_r}, expected {spec.recurrent_layers}"
        )



def _validate_tensor_shapes(
    token_count: int,
    attention: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    recurrent: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    spec: Qwen38Spec,
) -> None:
    """Validate every HF cache tensor without materializing CPU copies."""
    attn_shape = (1, spec.kv_heads, token_count, spec.attention_head_dim)
    conv_shape = (1, spec.conv_channels, spec.linear_conv_kernel_dim)
    state_shape = (
        1,
        spec.linear_num_value_heads,
        spec.linear_key_head_dim,
        spec.linear_value_head_dim,
    )

    for il in spec.full_attention_layers:
        k, v = attention[il]
        if not torch.is_tensor(k) or not torch.is_tensor(v):
            raise ValueError(f"layer {il}: K/V must be torch tensors")
        if tuple(k.shape) != attn_shape:
            raise ValueError(f"layer {il}: K shape {tuple(k.shape)} != expected {attn_shape}")
        if tuple(v.shape) != attn_shape:
            raise ValueError(f"layer {il}: V shape {tuple(v.shape)} != expected {attn_shape}")

    for il in spec.recurrent_layers:
        conv, state = recurrent[il]
        if not torch.is_tensor(conv) or not torch.is_tensor(state):
            raise ValueError(f"layer {il}: recurrent R/S sources must be torch tensors")
        if tuple(conv.shape) != conv_shape:
            raise ValueError(f"layer {il}: conv shape {tuple(conv.shape)} != expected {conv_shape}")
        if tuple(state.shape) != state_shape:
            raise ValueError(f"layer {il}: recurrent state {tuple(state.shape)} != expected {state_shape}")


def _write_attention_rows(
    f,
    tensor: torch.Tensor,
    *,
    typ: GGMLFloatType,
    token_count: int,
    width: int,
    transposed: bool,
    spec: Qwen38Spec,
) -> None:
    rows = attention_token_rows(
        tensor,
        typ=typ,
        expected_heads=spec.kv_heads,
        expected_head_dim=spec.attention_head_dim,
    )
    if rows.shape != (token_count, width):
        raise ValueError(f"attention rows {rows.shape} != expected {(token_count, width)}")
    if transposed:
        # llama.cpp classic V layout: embedding component outermost, cells innermost.
        f.write(rows.T.copy(order="C").tobytes(order="C"))
    else:
        f.write(rows.tobytes(order="C"))


def write_qwen38_ggsq_v3(
    path: str | Path,
    token_ids: Sequence[int],
    attention: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    recurrent: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    *,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    v_trans: bool = False,
    n_stream: int = 1,
    spec: Qwen38Spec = QWEN38_27B,
) -> None:
    """Write a Qwen3.8-27B text sequence cache in llama.cpp GGSQ v3 format.

    The memory payload mirrors current llama.cpp's hybrid memory order:
      1) llama_kv_cache state for 16 full-attention layers
      2) llama_memory_recurrent state for 48 Gated DeltaNet layers

    F16/F32 attention K/V are supported. Recurrent R/S are always F32, matching
    llama.cpp's Qwen3.5/Qwen3.8 recurrent memory allocation.

    Important v3 detail: unlike GGSQ v4, v3 does NOT serialize n_rot_k/n_rot_v
    after the attention v_trans/n_layer header.
    """
    if cache_type_k not in FLOAT_TYPES or cache_type_v not in FLOAT_TYPES:
        raise ValueError("cache_type_k/cache_type_v must be f16 or f32")
    if n_stream < 1:
        raise ValueError("n_stream must be >= 1")
    if not token_ids:
        raise ValueError("cannot export an empty sequence")
    if any(not isinstance(x, (int, np.integer)) for x in token_ids):
        raise ValueError("token_ids must be integers")

    _validate_layer_sets(attention, recurrent, spec)

    k_typ = FLOAT_TYPES[cache_type_k]
    v_typ = FLOAT_TYPES[cache_type_v]
    n_tokens = len(token_ids)

    # Validate the complete cache before creating output. Data conversion itself is
    # streamed below, so exporting a long context does not duplicate the entire
    # attention + recurrent cache in host RAM.
    _validate_tensor_shapes(n_tokens, attention, recurrent, spec)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
    os.close(fd)
    tmp = Path(tmp_name)

    try:
        with tmp.open("wb") as f:
            # llama_context::state_seq_save_file wrapper.
            f.write(_u32(LLAMA_STATE_SEQ_MAGIC))
            f.write(_u32(LLAMA_STATE_SEQ_VERSION))
            f.write(_u32(n_tokens))
            for tok in token_ids:
                f.write(_i32(tok))

            # ------------------------------------------------------------------
            # Attention component: llama_kv_cache::state_write(seq_id >= 0)
            # ------------------------------------------------------------------
            f.write(_u32(n_stream))
            for stream in range(n_stream):
                # A single exported sequence is placed in source stream 0. Other
                # streams are serialized empty; llama.cpp requires exact n_stream.
                if stream != 0:
                    f.write(_u32(0))
                    continue

                f.write(_u32(n_tokens))

                # Qwen3.8 uses M-RoPE => has_cell_ext()==true. For text-only input,
                # all three positional planes are identical. llama_kv_cell_ext is
                # {x, y, tok}, each int32.
                for pos, tok in enumerate(token_ids):
                    f.write(_i32(pos))       # primary position
                    f.write(_u32(1))         # exactly one saved seq id
                    f.write(_i32(pos))       # ext.x
                    f.write(_i32(pos))       # ext.y
                    f.write(_i32(tok))       # ext.tok
                    f.write(_i32(0))         # saved source seq id; ignored on seq restore

                # GGSQ v3 attention header. v4 added n_rot_k/n_rot_v here;
                # they MUST NOT be present in a v3 file, even if both would be zero.
                f.write(_u32(1 if v_trans else 0))
                f.write(_u32(len(spec.full_attention_layers)))

                # K: all layers first.
                for il in spec.full_attention_layers:
                    k, _ = attention[il]
                    f.write(_i32(k_typ.enum))
                    f.write(_u64(spec.attention_k_width * k_typ.itemsize))
                    _write_attention_rows(
                        f, k, typ=k_typ, token_count=n_tokens, width=spec.attention_k_width,
                        transposed=False, spec=spec,
                    )

                # V: all layers after K.
                if not v_trans:
                    for il in spec.full_attention_layers:
                        _, v = attention[il]
                        f.write(_i32(v_typ.enum))
                        f.write(_u64(spec.attention_v_width * v_typ.itemsize))
                        _write_attention_rows(
                            f, v, typ=v_typ, token_count=n_tokens, width=spec.attention_v_width,
                            transposed=False, spec=spec,
                        )
                else:
                    for il in spec.full_attention_layers:
                        _, v = attention[il]
                        f.write(_i32(v_typ.enum))
                        f.write(_u32(v_typ.itemsize))
                        f.write(_u32(spec.attention_v_width))
                        _write_attention_rows(
                            f, v, typ=v_typ, token_count=n_tokens, width=spec.attention_v_width,
                            transposed=True, spec=spec,
                        )

            # ------------------------------------------------------------------
            # Recurrent component: llama_memory_recurrent::state_write(seq_id >= 0)
            # ------------------------------------------------------------------
            f.write(_u32(1))                  # recurrent cell_count
            f.write(_i32(n_tokens - 1))       # cell.pos == latest sequence position
            f.write(_u32(0))                  # seq-specific save stores no seq ids here

            f.write(_u32(0))                  # s_trans; current writer uses 0
            f.write(_u32(spec.n_layers))      # total model layer count, incl. null full-attn layers

            # R tensors first, only for recurrent layers. llama.cpp Qwen3.8 R/S are F32.
            for il in spec.recurrent_layers:
                conv, _ = recurrent[il]
                rr = recurrent_r_row(conv, spec=spec)
                f.write(_i32(GGML_TYPE_F32))
                f.write(_u64(spec.recurrent_r_width * 4))
                f.write(rr.tobytes(order="C"))
                del rr

            # Then S tensors.
            for il in spec.recurrent_layers:
                _, state = recurrent[il]
                sr = recurrent_s_row(state, spec=spec)
                f.write(_i32(GGML_TYPE_F32))
                f.write(_u64(spec.recurrent_s_width * 4))
                f.write(sr.tobytes(order="C"))
                del sr

            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


class _Reader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.f = self.path.open("rb")
        self.size = self.path.stat().st_size

    def close(self) -> None:
        self.f.close()

    @property
    def pos(self) -> int:
        return self.f.tell()

    def unpack(self, fmt: str):
        s = struct.Struct(fmt)
        b = self.f.read(s.size)
        if len(b) != s.size:
            raise ValueError(f"truncated GGSQ at offset {self.pos}")
        v = s.unpack(b)
        return v[0] if len(v) == 1 else v

    def skip(self, n: int) -> None:
        if n < 0 or self.pos + n > self.size:
            raise ValueError(f"invalid/truncated payload at offset {self.pos}, skip={n}")
        self.f.seek(n, os.SEEK_CUR)


def inspect_qwen38_ggsq_v3(
    path: str | Path,
    *,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    v_trans: bool = False,
    n_stream: int = 1,
    spec: Qwen38Spec = QWEN38_27B,
) -> dict:
    """Streaming structural verifier for files produced by write_qwen38_ggsq_v3."""
    k_typ = FLOAT_TYPES[cache_type_k]
    v_typ = FLOAT_TYPES[cache_type_v]
    r = _Reader(path)
    try:
        magic = r.unpack("<I")
        version = r.unpack("<I")
        if magic != LLAMA_STATE_SEQ_MAGIC or version != LLAMA_STATE_SEQ_VERSION:
            raise ValueError(f"not GGSQ v3: magic=0x{magic:08x}, version={version}")

        n_tokens = r.unpack("<I")
        if n_tokens < 1:
            raise ValueError("zero-token sequence")
        tokens = [r.unpack("<i") for _ in range(n_tokens)]

        ns = r.unpack("<I")
        if ns != n_stream:
            raise ValueError(f"n_stream mismatch in file: {ns} != expected {n_stream}")

        nonempty_streams = 0
        for s in range(ns):
            cell_count = r.unpack("<I")
            if cell_count == 0:
                continue
            nonempty_streams += 1
            if s != 0 or cell_count != n_tokens:
                raise ValueError(f"unexpected attention stream {s} cell_count={cell_count}")

            for pos in range(cell_count):
                p = r.unpack("<i")
                nseq = r.unpack("<I")
                x = r.unpack("<i")
                y = r.unpack("<i")
                tok = r.unpack("<i")
                seqid = r.unpack("<i")
                if (p, x, y, tok, nseq, seqid) != (pos, pos, pos, tokens[pos], 1, 0):
                    raise ValueError(f"bad cell metadata at token {pos}")

            # GGSQ v3 has exactly these two fields here. Do not read the
            # n_rot_k/n_rot_v fields introduced by v4.
            vt = bool(r.unpack("<I"))
            nl = r.unpack("<I")
            if vt != v_trans or nl != len(spec.full_attention_layers):
                raise ValueError(f"attention header mismatch: v_trans={vt}, layers={nl}")

            for _ in spec.full_attention_layers:
                gt = r.unpack("<i")
                rowb = r.unpack("<Q")
                expected = spec.attention_k_width * k_typ.itemsize
                if gt != k_typ.enum or rowb != expected:
                    raise ValueError(f"K descriptor mismatch: type={gt}, row={rowb}, expected={expected}")
                r.skip(cell_count * rowb)

            for _ in spec.full_attention_layers:
                gt = r.unpack("<i")
                if gt != v_typ.enum:
                    raise ValueError(f"V type mismatch: {gt} != {v_typ.enum}")
                if not v_trans:
                    rowb = r.unpack("<Q")
                    expected = spec.attention_v_width * v_typ.itemsize
                    if rowb != expected:
                        raise ValueError(f"V row mismatch: {rowb} != {expected}")
                    r.skip(cell_count * rowb)
                else:
                    el = r.unpack("<I")
                    width = r.unpack("<I")
                    if el != v_typ.itemsize or width != spec.attention_v_width:
                        raise ValueError(f"transposed V descriptor mismatch: el={el}, width={width}")
                    r.skip(cell_count * width * el)

        if nonempty_streams != 1:
            raise ValueError(f"expected exactly one non-empty attention stream, got {nonempty_streams}")

        # Recurrent component.
        rec_cells = r.unpack("<I")
        if rec_cells != 1:
            raise ValueError(f"unexpected recurrent cell_count={rec_cells}")
        rec_pos = r.unpack("<i")
        rec_nseq = r.unpack("<I")
        if rec_pos != n_tokens - 1 or rec_nseq != 0:
            raise ValueError(f"bad recurrent meta: pos={rec_pos}, nseq={rec_nseq}")

        s_trans = r.unpack("<I")
        rec_n_layer = r.unpack("<I")
        if s_trans != 0 or rec_n_layer != spec.n_layers:
            raise ValueError(f"bad recurrent header: s_trans={s_trans}, n_layer={rec_n_layer}")

        for _ in spec.recurrent_layers:
            gt = r.unpack("<i")
            rowb = r.unpack("<Q")
            expected = spec.recurrent_r_width * 4
            if gt != GGML_TYPE_F32 or rowb != expected:
                raise ValueError(f"R descriptor mismatch: type={gt}, row={rowb}, expected={expected}")
            r.skip(rowb)

        for _ in spec.recurrent_layers:
            gt = r.unpack("<i")
            rowb = r.unpack("<Q")
            expected = spec.recurrent_s_width * 4
            if gt != GGML_TYPE_F32 or rowb != expected:
                raise ValueError(f"S descriptor mismatch: type={gt}, row={rowb}, expected={expected}")
            r.skip(rowb)

        if r.pos != r.size:
            raise ValueError(f"trailing bytes: parsed={r.pos}, file_size={r.size}")

        return {
            "magic": magic,
            "version": version,
            "n_tokens": n_tokens,
            "n_stream": ns,
            "attention_layers": len(spec.full_attention_layers),
            "recurrent_layers": len(spec.recurrent_layers),
            "v_trans": v_trans,
            "cache_type_k": cache_type_k,
            "cache_type_v": cache_type_v,
            "file_size": r.size,
        }
    finally:
        r.close()
