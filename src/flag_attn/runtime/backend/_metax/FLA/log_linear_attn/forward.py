# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import lru_cache

import torch
import triton

from .kernels import (
    _fused_output_kernel,
    _fused_reduce_output_kernel,
    _inter_fwd_kernel,
    _local_fwd_kernel,
    _merge_states_kernel,
    _paired_states_kernel,
    _prepare_weights_kernel,
    _quad_states_kernel,
    _reduce_output_flat_kernel,
    _scores_kernel,
)

CHUNK_SIZE = 64
BLOCK_KEY = 64
BLOCK_VALUE = 64
REDUCTION_BLOCK = 4096


@lru_cache(maxsize=None)
def _level_lut(device_index: int) -> torch.Tensor:
    device = torch.device("cuda", device_index)
    lut = torch.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=torch.int32, device=device)
    rows = torch.arange(CHUNK_SIZE, device=device)[:, None]
    cols = torch.arange(CHUNK_SIZE, device=device)[None, :]
    for level in range(1, int(math.log2(CHUNK_SIZE)) + 1):
        half = 1 << (level - 1)
        anchor = rows - rows % half
        selected = (rows % (2 * half) >= half) & (cols + half >= anchor) & (cols < anchor)
        lut = torch.where(selected, level, lut)
    return lut


def _chunk_local_cumsum(g: torch.Tensor) -> torch.Tensor:
    batch, sequence, heads = g.shape
    return g.view(batch, sequence // CHUNK_SIZE, CHUNK_SIZE, heads).cumsum(dim=2).view_as(g)


def _validate_inputs(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, level_scales: torch.Tensor
) -> tuple[int, int, int, int, int, int]:
    tensors = (q, k, v, g, level_scales)
    if not all((tensor.is_cuda for tensor in tensors)):
        raise ValueError("chunk_log_linear_attn requires CUDA tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all inputs must be on the same CUDA device")
    if any((tensor.requires_grad for tensor in tensors)):
        raise RuntimeError("chunk_log_linear_attn is forward-only and does not support autograd")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("chunk_log_linear_attn supports BF16 q, k, and v")
    if g.dtype != torch.float32:
        raise ValueError("g must use float32")
    if level_scales.dtype != q.dtype:
        raise ValueError("level_scales must have the same dtype as q")
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 tensors with matching q/k shapes")
    if q.shape[2] != 1 or k.shape[2] != 1:
        raise ValueError("chunk_log_linear_attn requires one shared Q/K head")
    batch, sequence, _, key_dim = q.shape
    heads, value_dim = (v.shape[2], v.shape[-1])
    if v.shape[:2] != (batch, sequence):
        raise ValueError("v must have shape [B, T, H, V]")
    if g.shape != (batch, sequence, heads):
        raise ValueError("g must have shape [B, T, H]")
    if batch < 1 or sequence < 1 or heads < 1:
        raise ValueError("batch, sequence, and heads must be positive")
    if sequence % CHUNK_SIZE != 0 or sequence & (sequence - 1):
        raise ValueError("sequence must be a power of two and divisible by 64")
    if key_dim not in (64, 128, 256) or value_dim not in (64, 128, 256):
        raise ValueError("chunk_log_linear_attn supports K/V in {64, 128, 256}")
    if level_scales.ndim != 4:
        raise ValueError("level_scales must be rank 4")
    levels = level_scales.shape[-1]
    if level_scales.shape[:3] != (batch, sequence, heads):
        raise ValueError("level_scales must have shape [B, T, H, L]")
    if levels != math.ceil(math.log2(sequence)) + 1:
        raise ValueError("invalid level_scales level count")
    return (batch, sequence, heads, key_dim, value_dim, levels)


def chunk_log_linear_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, level_scales: torch.Tensor
) -> torch.Tensor:
    batch, sequence, heads, key_dim, value_dim, levels = _validate_inputs(q=q, k=k, v=v, g=g, level_scales=level_scales)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    level_scales = level_scales.contiguous()
    num_chunks = sequence // CHUNK_SIZE
    cache_states = num_chunks > 1
    fuse_forward = cache_states and num_chunks >= 4
    fuse_quad_states = fuse_forward and num_chunks >= 8
    first_level = 2 if fuse_quad_states else int(fuse_forward)
    num_levels = max(1, math.ceil(math.log2(num_chunks)))
    state_levels = num_levels - int(cache_states)
    num_key_tiles = key_dim // BLOCK_KEY
    num_value_tiles = value_dim // BLOCK_VALUE
    num_tiles = num_key_tiles * num_value_tiles
    tile_first = fuse_forward and num_tiles > 1
    state_tile_first = tile_first and num_tiles >= 8
    output_programs = batch * num_chunks * heads * num_tiles
    precompute_scores = heads * num_value_tiles >= 16 and output_programs >= 4096
    precompute_weighted = fuse_forward and precompute_scores and (num_value_tiles > 1)
    precompute_query = fuse_forward and output_programs >= 4096
    fuse_reduction = precompute_weighted and precompute_query and (num_key_tiles == 2)
    local_g = _chunk_local_cumsum(g=g.contiguous())
    level_lut = _level_lut(device_index=q.device.index or 0)
    partial_shape = (batch, sequence, heads, num_key_tiles, value_dim)
    if fuse_reduction:
        local_output = None
        output = torch.empty_like(v)
        output_partial = output
    else:
        local_output = torch.empty(partial_shape, device=v.device, dtype=v.dtype)
        output_partial = local_output
        output = output_partial.squeeze(3) if num_key_tiles == 1 else torch.empty_like(v)
    state_nodes = max(1, 2 * num_chunks - 2)
    if cache_states:
        first_nodes = num_chunks >> first_level
        state_nodes = first_nodes + (first_nodes // 2 if state_levels > first_level + 1 else 0)
        if fuse_forward and state_levels <= first_level:
            state_nodes = 0
    states = torch.empty(state_nodes * batch * heads * key_dim * value_dim, device=v.device, dtype=torch.float32)
    cached_states = (
        torch.empty((num_chunks - 1) * batch * heads * key_dim * value_dim, device=v.device, dtype=torch.bfloat16)
        if cache_states
        else None
    )
    chunk_gates = torch.empty(batch, num_chunks, heads, device=v.device, dtype=torch.float32)
    scores = (
        torch.empty(batch, num_chunks, num_key_tiles, CHUNK_SIZE, CHUNK_SIZE, device=q.device, dtype=torch.float32)
        if precompute_scores
        else None
    )
    weighted_scores = (
        torch.empty(batch, num_chunks, heads, num_key_tiles, CHUNK_SIZE, CHUNK_SIZE, device=q.device, dtype=q.dtype)
        if precompute_weighted
        else None
    )
    query_factors = (
        torch.empty(batch, num_chunks, heads, num_levels, CHUNK_SIZE, device=q.device, dtype=torch.float32)
        if precompute_query
        else None
    )
    state_kernel = _quad_states_kernel if fuse_quad_states else _paired_states_kernel
    output_kernel = _fused_reduce_output_kernel if fuse_reduction else _fused_output_kernel
    output_tiles = num_value_tiles if fuse_reduction else num_tiles
    output_grid = (
        (num_chunks * batch * heads * output_tiles,) if tile_first else (num_chunks, batch * heads, output_tiles)
    )
    state_grid = (
        ((num_chunks >> first_level) * batch * heads * num_tiles,)
        if state_tile_first
        else (num_chunks >> first_level, batch * heads, num_tiles)
    )
    if precompute_scores:
        _scores_kernel[num_chunks, batch, num_key_tiles](
            q=q, k=k, scores=scores, T=sequence, K=key_dim, NT=num_chunks, NK=num_key_tiles, BK=BLOCK_KEY, num_warps=4
        )
    if not fuse_forward:
        _local_fwd_kernel[num_chunks, batch * heads, num_tiles](
            q=q,
            k=k,
            v=v,
            g=local_g,
            level_scales=level_scales,
            level_lut=level_lut,
            local_output=local_output,
            states=states,
            chunk_gates=chunk_gates,
            scores=scores,
            cached_states=cached_states,
            T=sequence,
            H=heads,
            K=key_dim,
            V=value_dim,
            L=levels,
            NT=num_chunks,
            BK=BLOCK_KEY,
            BV=BLOCK_VALUE,
            NK=num_key_tiles,
            NV=num_value_tiles,
            PRECOMPUTE_SCORES=precompute_scores,
            CACHE_STATES=cache_states,
            num_warps=4,
            num_stages=2,
        )
    if fuse_forward:
        state_kernel[state_grid](
            k=k,
            v=v,
            g=local_g,
            states=states,
            cached_states=cached_states,
            chunk_gates=chunk_gates,
            B=batch,
            T=sequence,
            H=heads,
            K=key_dim,
            V=value_dim,
            NT=num_chunks,
            BK=BLOCK_KEY,
            BV=BLOCK_VALUE,
            NV=num_value_tiles,
            STORE_STATES=first_level < state_levels,
            TILE_FIRST=state_tile_first,
            num_warps=4,
            num_stages=2,
        )
    for level in range(first_level + 1, num_levels):
        block = 1 << level
        half = block >> 1
        store_states = level < state_levels
        merge_nodes = num_chunks // block if store_states else num_chunks // (2 * block)
        merge_grid = (
            (merge_nodes * batch * heads * num_tiles,) if state_tile_first else (merge_nodes, batch * heads, num_tiles)
        )
        _merge_states_kernel[merge_grid](
            states=states,
            cached_states=cached_states,
            chunk_gates=chunk_gates,
            B=batch,
            H=heads,
            K=key_dim,
            V=value_dim,
            NT=num_chunks,
            LEVEL=level,
            HALF=half,
            BLOCK=block,
            BK=BLOCK_KEY,
            BV=BLOCK_VALUE,
            NV=num_value_tiles,
            CACHE_STATES=cache_states,
            FIRST_LEVEL=first_level,
            STORE_STATES=store_states,
            TILE_FIRST=state_tile_first,
            num_warps=4,
            num_stages=2,
        )
    if precompute_weighted or precompute_query:
        _prepare_weights_kernel[num_chunks, batch * heads](
            g=local_g,
            level_scales=level_scales,
            level_lut=level_lut,
            scores=scores,
            chunk_gates=chunk_gates,
            weighted_scores=weighted_scores,
            query_factors=query_factors,
            T=sequence,
            H=heads,
            L=levels,
            NT=num_chunks,
            NK=num_key_tiles,
            NUM_LEVELS=num_levels,
            MAX_BLOCK=1 << (num_levels - 1),
            WEIGHTED_SCORES=precompute_weighted,
            PRECOMPUTE_QUERY=precompute_query,
            num_warps=4,
            num_stages=2,
        )
    if fuse_forward:
        output_kernel[output_grid](
            q=q,
            k=k,
            v=v,
            g=local_g,
            level_scales=level_scales,
            level_lut=level_lut,
            scores=weighted_scores if precompute_weighted else scores,
            states=cached_states,
            chunk_gates=chunk_gates,
            output=output_partial,
            query_factors=query_factors,
            B=batch,
            T=sequence,
            H=heads,
            K=key_dim,
            V=value_dim,
            L=levels,
            NT=num_chunks,
            NUM_LEVELS=num_levels,
            MAX_BLOCK=1 << (num_levels - 1),
            BK=BLOCK_KEY,
            BV=BLOCK_VALUE,
            NK=num_key_tiles,
            NV=num_value_tiles,
            PRECOMPUTE_SCORES=precompute_scores,
            CACHE_STATES=cache_states,
            WEIGHTED_SCORES=precompute_weighted,
            PRECOMPUTE_QUERY=precompute_query,
            TILE_FIRST=tile_first,
            num_warps=4,
            num_stages=2,
        )
    else:
        _inter_fwd_kernel[num_chunks, batch * heads, num_tiles](
            q=q,
            g=local_g,
            level_scales=level_scales,
            local_output=local_output,
            states=cached_states if cache_states else states,
            chunk_gates=chunk_gates,
            output=output_partial,
            B=batch,
            T=sequence,
            H=heads,
            K=key_dim,
            V=value_dim,
            L=levels,
            NT=num_chunks,
            NUM_LEVELS=num_levels,
            MAX_BLOCK=1 << (num_levels - 1),
            BK=BLOCK_KEY,
            BV=BLOCK_VALUE,
            NK=num_key_tiles,
            NV=num_value_tiles,
            CACHE_STATES=cache_states,
            num_warps=4,
            num_stages=2,
        )
    if num_key_tiles > 1 and (not fuse_reduction):
        _reduce_output_flat_kernel[triton.cdiv(batch * sequence * heads * value_dim, REDUCTION_BLOCK),](
            partial=output_partial,
            output=output,
            N=batch * sequence * heads * value_dim,
            V=value_dim,
            NK=num_key_tiles,
            BLOCK=REDUCTION_BLOCK,
            num_warps=4,
        )
    return output
