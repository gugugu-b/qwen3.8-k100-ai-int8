"""Tuning3 Runtime Sitecustomize for Qwen3.8-27B on K100-AI (TP=2).

Integrates:
1. Native LDS-x GEMV for M=1 decode shapes:
   - qkv_proj:     K5120 -> N8192 (TP=2) / N14336
   - gate_up_proj: K5120 -> N17408 (TP=2) / N34816
   - down_proj:    K8704 -> N5120
   - out_proj:     K3072 -> N5120
2. Clean passthrough for M > 1 to lightop/lmslim matmul_int8
3. High-performance Triton Long-KV GQA (BM128/BN64/w8) for Full Attention prefill (layers 3, 7, 11, ..., 63)
"""

from __future__ import annotations
import os
import sys
import importlib.util
from typing import Optional
from pathlib import Path
import torch
import triton

print("[Tuning3 Patch] Initializing Qwen3.8-27B optimizations on K100-AI...", flush=True)

# ---------------------------------------------------------------------------
# 1. Load Native Unified Multi-M LDS-x Module
# ---------------------------------------------------------------------------
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SO_CANDIDATES = [
    os.path.join(_CURRENT_DIR, "k100_int8_gemv_unified.so"),
    "/data/wangzy/native_ext/k100_int8_gemv_unified.so",
    "/tmp/k100_int8_gemv_unified.so",
]
_UNIFIED_SO = None
for _cand in _SO_CANDIDATES:
    if os.path.exists(_cand):
        _UNIFIED_SO = _cand
        break

_lds_unified = None
if _UNIFIED_SO is not None:
    try:
        spec = importlib.util.spec_from_file_location("k100_int8_gemv_unified", _UNIFIED_SO)
        if spec and spec.loader:
            _lds_unified = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_lds_unified)
            print(f"[Tuning3 Patch] Loaded Unified Multi-M LDS-x from {_UNIFIED_SO}", flush=True)
    except Exception as e:
        print(f"[Tuning3 Patch] Warning: Failed to load Unified LDS-x from {_UNIFIED_SO}: {e}", flush=True)
else:
    print("[Tuning3 Patch] Warning: k100_int8_gemv_unified.so not found in candidate paths", flush=True)

# ---------------------------------------------------------------------------
# 2. Intercept scaled_mm for Multi-M (M in [1, 2, 3, 4]) LDS-x GEMV
# ---------------------------------------------------------------------------
_seen_mm = set()

def _tuning3_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
    best_config: Optional[list] = None,
    orig_fn = None,
) -> torch.Tensor:
    if (
        bias is None
        and out_dtype is torch.bfloat16
        and isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and a.is_cuda
        and b.is_cuda
        and a.dtype is torch.int8
        and b.dtype is torch.int8
        and a.ndim == 2
        and b.ndim == 2
        and _lds_unified is not None
    ):
        try:
            M = a.shape[0]
            K = a.shape[1]

            if M in (1, 2, 3, 4):
                # Check weight layout
                if int(b.shape[0]) == K: # [K, N] from triton_scaled_mm
                    N = b.shape[1]
                    w = getattr(b, "_ldsx_w", None)
                    if w is None or w.device != b.device or not w.is_contiguous():
                        w = b.t().contiguous()
                        b._ldsx_w = w
                elif int(b.shape[1]) == K: # [N, K] from blaslt_scaled_mm
                    N = b.shape[0]
                    w = b if b.is_contiguous() else b.contiguous()
                else:
                    N = None

                if N is not None:
                    # Defensive memory, contiguity, and 16-byte alignment checks
                    a_c = a if a.is_contiguous() else a.contiguous()
                    if a_c.data_ptr() % 16 == 0 and w.data_ptr() % 16 == 0:
                        sa = scale_a.to(device=a.device, dtype=torch.float32).reshape(-1)
                        sb = scale_b.to(device=b.device, dtype=torch.float32).reshape(-1)
                        if sa.numel() == 1 and M > 1:
                            sa = sa.expand(M).contiguous()
                        else:
                            sa = sa.contiguous()
                        if sb.numel() == 1 and N > 1:
                            sb = sb.expand(N).contiguous()
                        else:
                            sb = sb.contiguous()

                        if sa.numel() >= M and sb.numel() >= N:
                            if K == 5120 and (N in (8192, 14336, 17408, 34816)):
                                key = ("lds_k5120", K, N)
                                if key not in _seen_mm:
                                    _seen_mm.add(key)
                                    print(f"[Tuning3 Patch] Active K5120 Unified LDS-x for M={M} K={K}->N={N}", flush=True)
                                return _lds_unified.gemv_k5120(a_c, w, sa, sb, 4)

                            if (K == 8704 or K == 3072) and N == 5120:
                                key = ("lds_row", K, N)
                                if key not in _seen_mm:
                                    _seen_mm.add(key)
                                    print(f"[Tuning3 Patch] Active TP2 Row Unified LDS-x for M={M} K={K}->N={N}", flush=True)
                                return _lds_unified.gemv_row(a_c, w, sa, sb, 4)
        except Exception:
            pass

    if orig_fn is not None:
        return orig_fn(a, b, scale_a, scale_b, out_dtype, bias, best_config)
    from lmslim.layers.gemm.int8_utils import matmul_int8
    return matmul_int8(a, scale_a, b, scale_b, out_dtype, best_config)

try:
    from lmslim.quantize import quant_ops as _lmslim_quant_ops
    _orig_lmslim_mm = _lmslim_quant_ops.triton_scaled_mm
    _lmslim_quant_ops.triton_scaled_mm = lambda *args, **kwargs: _tuning3_scaled_mm(*args, orig_fn=_orig_lmslim_mm, **kwargs)
    print("[Tuning3 Patch] Patched lmslim.quantize.quant_ops.triton_scaled_mm", flush=True)
except Exception as e:
    print(f"[Tuning3 Patch] Warning: lmslim patch failed ({e})", flush=True)

try:
    from lightop import gemm_ops as _lightop_gemm_ops
    _orig_lightop_mm = _lightop_gemm_ops.triton_scaled_mm
    _lightop_gemm_ops.triton_scaled_mm = lambda *args, **kwargs: _tuning3_scaled_mm(*args, orig_fn=_orig_lightop_mm, **kwargs)
    print("[Tuning3 Patch] Patched lightop.gemm_ops.triton_scaled_mm", flush=True)
except Exception as e:
    print(f"[Tuning3 Patch] Note: lightop gemm_ops patch skipped ({e})", flush=True)

try:
    from sglang.srt.layers.quantization.compressed_tensors import quant_ops as _sgl_quant_ops
    if hasattr(_sgl_quant_ops, "triton_scaled_mm"):
        _orig_sgl_tri_mm = _sgl_quant_ops.triton_scaled_mm
        _sgl_quant_ops.triton_scaled_mm = lambda *args, **kwargs: _tuning3_scaled_mm(*args, orig_fn=_orig_sgl_tri_mm, **kwargs)
        print("[Tuning3 Patch] Patched sglang compressed_tensors quant_ops.triton_scaled_mm", flush=True)
    if hasattr(_sgl_quant_ops, "blaslt_scaled_mm"):
        _orig_sgl_blaslt = _sgl_quant_ops.blaslt_scaled_mm
        def _wrapped_blaslt(a, b, scale_a, scale_b, out_dtype, bias=None):
            return _tuning3_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=bias, orig_fn=lambda *a_args, **k_kwargs: _orig_sgl_blaslt(a, b, scale_a, scale_b, out_dtype, bias=bias))
        _sgl_quant_ops.blaslt_scaled_mm = _wrapped_blaslt
        print("[Tuning3 Patch] Patched sglang compressed_tensors quant_ops.blaslt_scaled_mm", flush=True)
except Exception as e:
    print(f"[Tuning3 Patch] Note: sglang quant_ops patch skipped ({e})", flush=True)

# ---------------------------------------------------------------------------
# 3. Triton Long-KV GQA (BM128/BN64/w8) for Full Attention Prefill
# ---------------------------------------------------------------------------
_VENDOR = "/usr/local/lib/python3.10/dist-packages/flash_attn/flash_attn_triton_mqa_gqa.py"
_tri_longkv_fn = None

if os.path.exists(_VENDOR):
    try:
        _raw = Path(_VENDOR).read_text()
        _old_start = "            start_m = MAX_SEQLENS_Q // BLOCK_M -1 - tl.program_id(0)\n"
        _new_start = "            start_m = (MAX_SEQLENS_Q + BLOCK_M - 1) // BLOCK_M - 1 - tl.program_id(0)\n"
        _old_guard = "            if start_m * BLOCK_M > seqlen_q:\n                return\n"
        _new_guard = "            if start_m * BLOCK_M > seqlen_q:\n                start_m = 0\n"
        if _old_start in _raw and _old_guard in _raw:
            _raw = _raw.replace(_old_start, _new_start).replace(_old_guard, _new_guard)
        
        _tmp_py = f"/tmp/q38_tp2_longkv_{os.getpid()}.py"
        Path(_tmp_py).write_text(_raw)
        spec = importlib.util.spec_from_file_location(f"q38_tp2_longkv_{os.getpid()}", _tmp_py)
        if spec and spec.loader:
            _mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_mod)
            _mod.attn_fwd.configs = [
                triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 0, "PRE_LOAD_V": False}, num_stages=1, num_warps=8)
            ]
            if hasattr(_mod.attn_fwd, "cache"):
                _mod.attn_fwd.cache.clear()
            if hasattr(_mod.attn_fwd, "best_config"):
                _mod.attn_fwd.best_config = None
            _tri_longkv_fn = _mod.flash_attn_varlen_func
            print("[Tuning3 Patch] Compiled and loaded Triton Long-KV GQA (BM128/BN64/w8)", flush=True)
    except Exception as e:
        print(f"[Tuning3 Patch] Warning: Failed to load Triton Long-KV: {e}", flush=True)

if _tri_longkv_fn is not None:
    try:
        from sglang.srt.layers.attention import pack_paged_kv_to_varlen as _packmod
        from sglang.srt.layers.attention import flashattention_backend as _fabackend

        _parent_try = _packmod.try_pack_paged_kv_to_varlen_attention
        _longkv_hits = 0

        def _first(x):
            return x[0] if isinstance(x, (tuple, list)) else x

        def _longkv_eligible(*, q, forward_batch, metadata, layer, window_size, sinks, key_cache, value_cache, page_size):
            if int(getattr(layer, "tp_q_head_num", -1)) != 12:
                return False
            if int(getattr(layer, "tp_k_head_num", -1)) != 2 or int(getattr(layer, "tp_v_head_num", -1)) != 2:
                return False
            if int(getattr(layer, "head_dim", -1)) != 256 or int(getattr(layer, "v_head_dim", -1)) != 256:
                return False
            if q.dtype != torch.bfloat16 or key_cache.dtype != torch.bfloat16 or value_cache.dtype != torch.bfloat16:
                return False
            if tuple(window_size) != (-1, -1) or sinks is not None:
                return False
            if bool(getattr(forward_batch, "mha_return_lse", False)):
                return False
            if float(getattr(layer, "logit_cap", 0.0)) != 0.0:
                return False
            
            max_q = int(getattr(metadata, "max_seq_len_q", -1))
            max_k = int(getattr(metadata, "max_seq_len_k", -1))
            if max_q < 512 or max_k < 2048:
                return False
                
            if getattr(metadata, "page_table", None) is None:
                return False
            if getattr(metadata, "cu_seqlens_q", None) is None or getattr(metadata, "cu_seqlens_k", None) is None:
                return False
            return True

        def _longkv_try(
            *, q, forward_batch, metadata, layer, window_size, sinks,
            key_cache, value_cache, page_size, k_descale, v_descale, **kwargs
        ):
            global _longkv_hits
            if not _longkv_eligible(
                q=q, forward_batch=forward_batch, metadata=metadata, layer=layer,
                window_size=window_size, sinks=sinks, key_cache=key_cache,
                value_cache=value_cache, page_size=page_size
            ):
                return _parent_try(
                    q=q, forward_batch=forward_batch, metadata=metadata, layer=layer,
                    window_size=window_size, sinks=sinks, key_cache=key_cache,
                    value_cache=value_cache, page_size=page_size,
                    k_descale=k_descale, v_descale=v_descale, **kwargs
                )

            batch_size = int(getattr(forward_batch, "batch_size", 1))
            pk, pv = _packmod.pack_paged_kv_to_varlen(
                key_cache.view(-1, layer.tp_k_head_num, page_size, layer.head_dim),
                value_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim, page_size),
                metadata.page_table,
                forward_batch.seq_lens_cpu[:batch_size],
                page_size,
            )

            _longkv_hits += 1
            if _longkv_hits <= 4 or _longkv_hits % 32 == 0:
                print(
                    f"[Tuning3 Patch] Active Triton Long-KV hit={_longkv_hits} layer={getattr(layer, 'layer_id', -1)} "
                    f"max_q={metadata.max_seq_len_q} max_k={metadata.max_seq_len_k} B={batch_size}",
                    flush=True
                )

            out = _tri_longkv_fn(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                pk,
                pv,
                metadata.cu_seqlens_q,
                metadata.cu_seqlens_k,
                int(metadata.max_seq_len_q),
                int(metadata.max_seq_len_k),
                dropout_p=0.0,
                softmax_scale=layer.scaling,
                causal=True,
                return_attn_probs=False,
            )
            return _first(out)

        _packmod.try_pack_paged_kv_to_varlen_attention = _longkv_try
        _fabackend.try_pack_paged_kv_to_varlen_attention = _longkv_try
        print("[Tuning3 Patch] Patched try_pack_paged_kv_to_varlen_attention with Triton Long-KV", flush=True)
    except Exception as e:
        print(f"[Tuning3 Patch] Warning: Failed to hook pack_paged_kv_to_varlen_attention: {e}", flush=True)

print("[Tuning3 Patch] Initialization complete.", flush=True)


# The service uses Lightop's private LMSlim copy. Patch both distinct identities.
import json as _v2_json
import sys as _v2_sys
from pathlib import Path as _V2Path
from lmslim.layers.gemm import int8_utils as _v2_int8_utils
from lightop._lmslim_native.layers.gemm import int8_utils as _v2_lightop_int8_utils

_v2_tile_rows = _v2_json.loads((_V2Path(__file__).parent / 'triton_tiles.json').read_text())
_v2_tile_configs = {(r['M'], r['K'], r['N']): r['config'] for r in _v2_tile_rows}
assert len(_v2_tile_configs) == 18
assert all(c['SPLIT_K'] == 1 for c in _v2_tile_configs.values())
_v2_seen_tiles = set()
_v2_seen_by_backend = set()
_v2_originals = {
    'lmslim': _v2_int8_utils.matmul_int8,
    'lightop-native': _v2_lightop_int8_utils.matmul_int8,
}
assert _v2_originals['lmslim'] is not _v2_originals['lightop-native']

def _v2_make_matmul(original, backend):
    def matmul(a, a_scale, b, b_scale, out_dtype, config):
        if (out_dtype is torch.bfloat16
                and a.dtype is torch.int8 and b.dtype is torch.int8
                and a.ndim == b.ndim == 2 and a.is_cuda and b.is_cuda):
            m, k = a.shape
            if (b.shape[0] == k and b.stride() == (1, k)
                    and a_scale.numel() >= m and b_scale.numel() >= b.shape[1]):
                key = (m, k, b.shape[1])
                selected = _v2_tile_configs.get(key)
                if selected is not None:
                    previous_config = config
                    config = selected
                    _v2_seen_tiles.add(key)
                    if (backend, key) not in _v2_seen_by_backend:
                        _v2_seen_by_backend.add((backend, key))
                        print(f'[v2 Triton] Active MKN={key} backend={backend} previous_config={previous_config} config={config}', flush=True)
        return original(a, a_scale, b, b_scale, out_dtype, config)
    return matmul

_v2_wrappers = {name: _v2_make_matmul(fn, name) for name, fn in _v2_originals.items()}
_v2_alias_originals = {}
_v2_alias_wrappers = {}
for _v2_name, _v2_module in list(_v2_sys.modules.items()):
    if not _v2_name.startswith(('lmslim', 'lightop', 'sglang')) or _v2_module is None:
        continue
    _v2_fn = getattr(_v2_module, 'matmul_int8', None)
    for _v2_backend, _v2_original in _v2_originals.items():
        if _v2_fn is _v2_original:
            _v2_alias_originals[_v2_name] = _v2_original
            _v2_alias_wrappers[_v2_name] = _v2_wrappers[_v2_backend]
            _v2_module.matmul_int8 = _v2_wrappers[_v2_backend]
            break
assert 'lmslim.layers.gemm.int8_utils' in _v2_alias_originals
assert 'lightop._lmslim_native.layers.gemm.int8_utils' in _v2_alias_originals
assert 'lightop._lmslim_native.quantize.quant_ops' in _v2_alias_originals
print(f'[v2 Triton] Loaded {len(_v2_tile_configs)} measured configurations; aliases={list(_v2_alias_originals)}', flush=True)

# Independent communication item; unchanged tile base.
from native_bf16_ar_guard import install as _native_bf16_ar_install
_native_bf16_ar_install()
