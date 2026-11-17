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

import triton
import triton.language as tl

BLOCK_T = tl.constexpr(64)


@triton.jit
def _scores_kernel(
    q, k, scores, T: tl.constexpr, K: tl.constexpr, NT: tl.constexpr, NK: tl.constexpr, BK: tl.constexpr
):
    chunk = tl.program_id(0).to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    key_tile = tl.program_id(2)
    times = chunk * BLOCK_T + tl.arange(0, BLOCK_T)
    keys = key_tile * BK + tl.arange(0, BK)
    query = tl.load(q + (batch * T + times[:, None]) * K + keys[None, :])
    key = tl.load(k + (batch * T + times[:, None]) * K + keys[None, :])
    score = tl.dot(query, tl.trans(key))
    rows = tl.arange(0, BLOCK_T)
    columns = tl.arange(0, BLOCK_T)
    offset = ((batch * NT + chunk) * NK + key_tile) * BLOCK_T * BLOCK_T
    tl.store(scores + offset + rows[:, None] * BLOCK_T + columns[None, :], score)


@triton.jit
def _local_fwd_kernel(
    q,
    k,
    v,
    g,
    level_scales,
    level_lut,
    local_output,
    states,
    chunk_gates,
    scores,
    cached_states,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    NT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    PRECOMPUTE_SCORES: tl.constexpr,
    CACHE_STATES: tl.constexpr,
    RETURN_TILE: tl.constexpr = False,
    KEY_TILE: tl.constexpr = -1,
    WEIGHTED_SCORES: tl.constexpr = False,
    TILE_FIRST: tl.constexpr = False,
):
    if TILE_FIRST:
        tiles: tl.constexpr = NV if KEY_TILE >= 0 else NK * NV
        index = tl.program_id(0).to(tl.int64)
        chunk_index = index // tiles % NT
        batch_head = index // (tiles * NT)
        tile_index = (index % tiles).to(tl.int32)
    else:
        chunk_index = tl.program_id(0).to(tl.int64)
        batch_head = tl.program_id(1).to(tl.int64)
        tile_index = tl.program_id(2)
    key_tile = tile_index // NV if KEY_TILE < 0 else KEY_TILE
    value_tile = tile_index % NV
    batch_index = batch_head // H
    head_index = batch_head % H
    offsets_t = tl.arange(0, BLOCK_T)
    offsets_k = key_tile * BK + tl.arange(0, BK)
    offsets_v = value_tile * BV + tl.arange(0, BV)
    token_offsets = chunk_index * BLOCK_T + offsets_t
    q_ptrs = q + (batch_index * T + token_offsets[:, None]) * K + offsets_k[None, :]
    k_ptrs = k + (batch_index * T + token_offsets[:, None]) * K + offsets_k[None, :]
    v_ptrs = v + ((batch_index * T + token_offsets[:, None]) * H + head_index) * V + offsets_v[None, :]
    b_k = tl.load(k_ptrs).to(tl.float32)
    b_v = tl.load(v_ptrs).to(tl.float32)
    b_g = tl.load(g + (batch_index * T + token_offsets) * H + head_index)
    row = offsets_t[:, None]
    col = offsets_t[None, :]
    if WEIGHTED_SCORES:
        score_base = (((batch_index * NT + chunk_index) * H + head_index) * NK + key_tile) * BLOCK_T * BLOCK_T
        b_score = tl.load(scores + score_base + row * BLOCK_T + col)
    else:
        b_level = tl.load(level_lut + row * BLOCK_T + col)
        scale_ptrs = level_scales + ((batch_index * T + token_offsets[:, None]) * H + head_index) * L + b_level
        b_level_scale = tl.load(scale_ptrs).to(tl.float32)
        if PRECOMPUTE_SCORES:
            score_base = ((batch_index * NT + chunk_index) * NK + key_tile) * BLOCK_T * BLOCK_T
            b_score = tl.load(scores + score_base + row * BLOCK_T + col)
        else:
            b_q = tl.load(q_ptrs).to(tl.float32)
            b_score = tl.dot(b_q.to(q.dtype.element_ty), tl.trans(b_k.to(k.dtype.element_ty)))
        b_score *= tl.exp(b_g[:, None] - b_g[None, :]) * b_level_scale
        b_score = tl.where(row >= col, b_score, 0.0)
    b_output = tl.dot(b_score.to(v.dtype.element_ty), b_v.to(v.dtype.element_ty))
    if RETURN_TILE:
        b_q = tl.load(q_ptrs).to(tl.float32)
        return (b_output.to(v.dtype.element_ty).to(tl.float32), b_q, b_g)
    else:
        output_ptrs = (
            local_output
            + (((batch_index * T + token_offsets[:, None]) * H + head_index) * NK + key_tile) * V
            + offsets_v[None, :]
        )
        tl.store(output_ptrs, b_output.to(local_output.dtype.element_ty))
        gate_last = tl.load(g + (batch_index * T + chunk_index * BLOCK_T + BLOCK_T - 1) * H + head_index)
        b_v_scaled = b_v * tl.exp(gate_last - b_g)[:, None]
        b_state = tl.dot(tl.trans(b_k.to(k.dtype.element_ty)), b_v_scaled.to(v.dtype.element_ty))
        state_base = (
            ((batch_index * NT + chunk_index) * H + head_index) * K * V + offsets_k[:, None] * V + offsets_v[None, :]
        )
        tl.store(states + state_base, b_state)
        if CACHE_STATES:
            if NT > 1 and chunk_index % 2 == 0:
                cache_base = (
                    ((batch_index * (NT // 2) + chunk_index // 2) * H + head_index) * K * V
                    + offsets_k[:, None] * V
                    + offsets_v[None, :]
                )
                tl.store(cached_states + cache_base, b_state.to(cached_states.dtype.element_ty))
        if key_tile == 0 and value_tile == 0:
            tl.store(chunk_gates + (batch_index * NT + chunk_index) * H + head_index, gate_last)


@triton.jit
def _merge_states_kernel(
    states,
    cached_states,
    chunk_gates,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NT: tl.constexpr,
    LEVEL: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    CACHE_STATES: tl.constexpr,
    FIRST_LEVEL: tl.constexpr,
    STORE_STATES: tl.constexpr = True,
    TILE_FIRST: tl.constexpr = False,
):
    if TILE_FIRST:
        tiles: tl.constexpr = K // BK * NV
        nodes: tl.constexpr = NT >> LEVEL if STORE_STATES else NT >> LEVEL + 1
        index = tl.program_id(0).to(tl.int64)
        node_index = index // tiles % nodes
        batch_head = index // (tiles * nodes)
        tile_index = (index % tiles).to(tl.int32)
    else:
        node_index = tl.program_id(0).to(tl.int64)
        batch_head = tl.program_id(1).to(tl.int64)
        tile_index = tl.program_id(2)
    if not STORE_STATES:
        node_index *= 2
    key_tile = tile_index // NV
    value_tile = tile_index % NV
    batch_index = batch_head // H
    head_index = batch_head % H
    right_end = (node_index + 1) * BLOCK - 1
    left_end = right_end - HALF
    offsets_k = key_tile * BK + tl.arange(0, BK)
    offsets_v = value_tile * BV + tl.arange(0, BV)
    if CACHE_STATES:
        previous_nodes = NT >> LEVEL - 1
        previous_level_base = tl.full((), (LEVEL - 1 - FIRST_LEVEL) % 2, tl.int64) * (NT >> FIRST_LEVEL) * B * H * K * V
        left_node = left_end >> LEVEL - 1
        right_node = right_end >> LEVEL - 1
    else:
        previous_nodes = NT >> LEVEL - 1
        previous_level_base = tl.full((), 2 * NT - 2 * previous_nodes, tl.int64) * B * H * K * V
        left_node = left_end >> LEVEL - 1
        right_node = right_end >> LEVEL - 1
    left_base = (
        previous_level_base
        + ((batch_index * previous_nodes + left_node) * H + head_index) * K * V
        + offsets_k[:, None] * V
        + offsets_v[None, :]
    )
    right_base = (
        previous_level_base
        + ((batch_index * previous_nodes + right_node) * H + head_index) * K * V
        + offsets_k[:, None] * V
        + offsets_v[None, :]
    )
    left_state = tl.load(states + left_base)
    right_state = tl.load(states + right_base)
    offsets_chunk = tl.arange(0, HALF)
    gate_ptrs = chunk_gates + (batch_index * NT + right_end - HALF + 1 + offsets_chunk) * H + head_index
    right_decay = tl.exp(tl.sum(tl.load(gate_ptrs), axis=0))
    merged = left_state * right_decay + right_state
    if CACHE_STATES:
        output_nodes = NT >> LEVEL
        output_level_base = tl.full((), (LEVEL - FIRST_LEVEL) % 2, tl.int64) * (NT >> FIRST_LEVEL) * B * H * K * V
        output_node = right_end >> LEVEL
    else:
        output_nodes = NT >> LEVEL
        output_level_base = tl.full((), 2 * NT - 2 * output_nodes, tl.int64) * B * H * K * V
        output_node = right_end >> LEVEL
    output_base = (
        output_level_base
        + ((batch_index * output_nodes + output_node) * H + head_index) * K * V
        + offsets_k[:, None] * V
        + offsets_v[None, :]
    )
    if STORE_STATES:
        tl.store(states + output_base, merged)
    if CACHE_STATES:
        if node_index % 2 == 0:
            cache_nodes = NT >> LEVEL + 1
            cache_level_base = tl.full((), NT - (NT >> LEVEL), tl.int64) * B * H * K * V
            cache_node = right_end >> LEVEL + 1
            cache_base = (
                cache_level_base
                + ((batch_index * cache_nodes + cache_node) * H + head_index) * K * V
                + offsets_k[:, None] * V
                + offsets_v[None, :]
            )
            tl.store(cached_states + cache_base, merged.to(cached_states.dtype.element_ty))


@triton.jit
def _inter_fwd_kernel(
    q,
    g,
    level_scales,
    local_output,
    states,
    chunk_gates,
    output,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    NT: tl.constexpr,
    NUM_LEVELS: tl.constexpr,
    MAX_BLOCK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    CACHE_STATES: tl.constexpr,
    initial_output=None,
    initial_q=None,
    initial_g=None,
    USE_INITIAL_TILES: tl.constexpr = False,
    KEY_TILE: tl.constexpr = -1,
    RETURN_TILE: tl.constexpr = False,
    query_factors=None,
    PRECOMPUTE_QUERY: tl.constexpr = False,
    TILE_FIRST: tl.constexpr = False,
):
    if TILE_FIRST:
        tiles: tl.constexpr = NV if KEY_TILE >= 0 else NK * NV
        index = tl.program_id(0).to(tl.int64)
        chunk_index = index // tiles % NT
        batch_head = index // (tiles * NT)
        tile_index = (index % tiles).to(tl.int32)
    else:
        chunk_index = tl.program_id(0).to(tl.int64)
        batch_head = tl.program_id(1).to(tl.int64)
        tile_index = tl.program_id(2)
    key_tile = tile_index // NV if KEY_TILE < 0 else KEY_TILE
    value_tile = tile_index % NV
    batch_index = batch_head // H
    head_index = batch_head % H
    offsets_t = tl.arange(0, BLOCK_T)
    offsets_k = key_tile * BK + tl.arange(0, BK)
    offsets_v = value_tile * BV + tl.arange(0, BV)
    token_offsets = chunk_index * BLOCK_T + offsets_t
    if USE_INITIAL_TILES:
        b_q = initial_q
        b_output = initial_output
        b_g = initial_g
    else:
        q_ptrs = q + (batch_index * T + token_offsets[:, None]) * K + offsets_k[None, :]
        local_ptrs = (
            local_output
            + (((batch_index * T + token_offsets[:, None]) * H + head_index) * NK + key_tile) * V
            + offsets_v[None, :]
        )
        b_q = tl.load(q_ptrs).to(tl.float32)
        b_output = tl.load(local_ptrs).to(tl.float32)
        b_g = tl.load(g + (batch_index * T + token_offsets) * H + head_index)
    current_end = chunk_index - 1
    later_decay = tl.zeros([], dtype=tl.float32)
    for level in tl.static_range(NUM_LEVELS):
        block_size = 1 << level
        if chunk_index & block_size != 0:
            if CACHE_STATES:
                level_nodes = NT >> level + 1
                level_base = tl.full((), NT - (NT >> level), tl.int64) * B * H * K * V
                state_node = current_end >> level + 1
            else:
                level_nodes = NT >> level
                level_base = tl.full((), 2 * NT - 2 * level_nodes, tl.int64) * B * H * K * V
                state_node = current_end >> level
            state_base = (
                level_base
                + ((batch_index * level_nodes + state_node) * H + head_index) * K * V
                + offsets_k[:, None] * V
                + offsets_v[None, :]
            )
            b_state = tl.load(states + state_base)
            if PRECOMPUTE_QUERY:
                factor_base = (((batch_index * NT + chunk_index) * H + head_index) * NUM_LEVELS + level) * BLOCK_T
                query_factor = tl.load(query_factors + factor_base + offsets_t)
                b_q_scaled = b_q * query_factor[:, None]
            else:
                scale_ptrs = level_scales + ((batch_index * T + token_offsets) * H + head_index) * L + 7 + level
                b_scale = tl.load(scale_ptrs).to(tl.float32)
                b_q_scaled = b_q * (b_scale * tl.exp(b_g + later_decay))[:, None]
            b_output += tl.dot(b_q_scaled.to(q.dtype.element_ty), b_state.to(q.dtype.element_ty))
            if not PRECOMPUTE_QUERY:
                offsets_chunk = tl.arange(0, MAX_BLOCK)
                gate_ptrs = (
                    chunk_gates + (batch_index * NT + current_end - block_size + 1 + offsets_chunk) * H + head_index
                )
                later_decay += tl.sum(tl.load(gate_ptrs, mask=offsets_chunk < block_size, other=0.0), axis=0)
            current_end -= block_size
    if RETURN_TILE:
        return b_output
    else:
        output_ptrs = (
            output
            + (((batch_index * T + token_offsets[:, None]) * H + head_index) * NK + key_tile) * V
            + offsets_v[None, :]
        )
        tl.store(output_ptrs, b_output.to(output.dtype.element_ty))


@triton.jit
def _reduce_output_flat_kernel(
    partial, output, N: tl.constexpr, V: tl.constexpr, NK: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // V
    column = offsets % V
    accumulator = tl.zeros((BLOCK,), tl.float32)
    for key_tile in tl.static_range(NK):
        value = tl.load(partial + (row * NK + key_tile) * V + column, offsets < N, other=0)
        accumulator += value.to(tl.float32)
    tl.store(output + offsets, accumulator.to(output.dtype.element_ty), offsets < N)


@triton.jit
def _state_tile(
    k,
    v,
    g,
    batch_index,
    head_index,
    chunk_index,
    key_tile,
    value_tile,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    tokens = chunk_index * BLOCK_T + tl.arange(0, BLOCK_T)
    keys = key_tile * BK + tl.arange(0, BK)
    values = value_tile * BV + tl.arange(0, BV)
    key = tl.load(k + (batch_index * T + tokens[:, None]) * K + keys[None, :])
    value = tl.load(v + ((batch_index * T + tokens[:, None]) * H + head_index) * V + values[None, :]).to(tl.float32)
    gates = tl.load(g + (batch_index * T + tokens) * H + head_index)
    last_gate = tl.load(g + (batch_index * T + (chunk_index + 1) * BLOCK_T - 1) * H + head_index)
    scaled_value = value * tl.exp(last_gate - gates)[:, None]
    state = tl.dot(tl.trans(key), scaled_value.to(v.dtype.element_ty))
    return (state, last_gate)


@triton.jit
def _paired_states_kernel(
    k,
    v,
    g,
    states,
    cached_states,
    chunk_gates,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    STORE_STATES: tl.constexpr = True,
    TILE_FIRST: tl.constexpr = False,
):
    if TILE_FIRST:
        tiles: tl.constexpr = K // BK * NV
        groups: tl.constexpr = NT // 2
        index = tl.program_id(0).to(tl.int64)
        pair = index // tiles % groups
        batch_head = index // (tiles * groups)
        tile = (index % tiles).to(tl.int32)
    else:
        pair = tl.program_id(0).to(tl.int64)
        batch_head = tl.program_id(1).to(tl.int64)
        tile = tl.program_id(2)
    batch_index = batch_head // H
    head_index = batch_head % H
    key_tile = tile // NV
    value_tile = tile % NV
    keys = key_tile * BK + tl.arange(0, BK)
    values = value_tile * BV + tl.arange(0, BV)
    tile_offsets = keys[:, None] * V + values[None, :]
    left, left_gate = _state_tile(
        k=k,
        v=v,
        g=g,
        batch_index=batch_index,
        head_index=head_index,
        chunk_index=pair * 2,
        key_tile=key_tile,
        value_tile=value_tile,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
    )
    cache_base = ((batch_index * (NT // 2) + pair) * H + head_index) * K * V
    tl.store(cached_states + cache_base + tile_offsets, left.to(cached_states.dtype.element_ty))
    right, right_gate = _state_tile(
        k=k,
        v=v,
        g=g,
        batch_index=batch_index,
        head_index=head_index,
        chunk_index=pair * 2 + 1,
        key_tile=key_tile,
        value_tile=value_tile,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
    )
    merged = tl.fma(left, tl.exp(right_gate), right)
    state_base = ((batch_index * (NT // 2) + pair) * H + head_index) * K * V
    if STORE_STATES:
        tl.store(states + state_base + tile_offsets, merged)
    if pair % 2 == 0:
        cache_level_base = tl.full((), NT // 2, tl.int64) * B * H * K * V
        cache_base = ((batch_index * (NT // 4) + pair // 2) * H + head_index) * K * V
        tl.store(
            cached_states + cache_level_base + cache_base + tile_offsets, merged.to(cached_states.dtype.element_ty)
        )
    if key_tile == 0 and value_tile == 0:
        tl.store(chunk_gates + (batch_index * NT + pair * 2) * H + head_index, left_gate)
        tl.store(chunk_gates + (batch_index * NT + pair * 2 + 1) * H + head_index, right_gate)


@triton.jit
def _quad_states_kernel(
    k,
    v,
    g,
    states,
    cached_states,
    chunk_gates,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    STORE_STATES: tl.constexpr = True,
    TILE_FIRST: tl.constexpr = False,
):
    if TILE_FIRST:
        tiles: tl.constexpr = K // BK * NV
        groups: tl.constexpr = NT // 4
        index = tl.program_id(0).to(tl.int64)
        group = index // tiles % groups
        batch_head = index // (tiles * groups)
        tile = (index % tiles).to(tl.int32)
    else:
        group = tl.program_id(0).to(tl.int64)
        batch_head = tl.program_id(1).to(tl.int64)
        tile = tl.program_id(2)
    batch = batch_head // H
    head = batch_head % H
    key_tile = tile // NV
    value_tile = tile % NV
    keys = key_tile * BK + tl.arange(0, BK)
    values = value_tile * BV + tl.arange(0, BV)
    offsets = keys[:, None] * V + values[None, :]
    first, first_gate = _state_tile(
        k=k,
        v=v,
        g=g,
        batch_index=batch,
        head_index=head,
        chunk_index=group * 4,
        key_tile=key_tile,
        value_tile=value_tile,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
    )
    leaf_base = ((batch * (NT // 2) + group * 2) * H + head) * K * V
    tl.store(cached_states + leaf_base + offsets, first.to(cached_states.dtype.element_ty))
    second, second_gate = _state_tile(
        k=k,
        v=v,
        g=g,
        batch_index=batch,
        head_index=head,
        chunk_index=group * 4 + 1,
        key_tile=key_tile,
        value_tile=value_tile,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
    )
    left = tl.fma(first, tl.exp(second_gate), second)
    level_one_base = tl.full((), NT // 2, tl.int64) * B * H * K * V
    pair_base = ((batch * (NT // 4) + group) * H + head) * K * V
    tl.store(cached_states + level_one_base + pair_base + offsets, left.to(cached_states.dtype.element_ty))
    third, third_gate = _state_tile(
        k=k,
        v=v,
        g=g,
        batch_index=batch,
        head_index=head,
        chunk_index=group * 4 + 2,
        key_tile=key_tile,
        value_tile=value_tile,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
    )
    tl.store(cached_states + leaf_base + H * K * V + offsets, third.to(cached_states.dtype.element_ty))
    fourth, fourth_gate = _state_tile(
        k=k,
        v=v,
        g=g,
        batch_index=batch,
        head_index=head,
        chunk_index=group * 4 + 3,
        key_tile=key_tile,
        value_tile=value_tile,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
    )
    right = tl.fma(third, tl.exp(fourth_gate), fourth)
    merged = tl.fma(left, tl.exp(third_gate + fourth_gate), right)
    if STORE_STATES:
        tl.store(states + pair_base + offsets, merged)
    if group % 2 == 0:
        level_two_base = tl.full((), 3 * NT // 4, tl.int64) * B * H * K * V
        cache_base = ((batch * (NT // 8) + group // 2) * H + head) * K * V
        tl.store(cached_states + level_two_base + cache_base + offsets, merged.to(cached_states.dtype.element_ty))
    if key_tile == 0 and value_tile == 0:
        gate_base = (batch * NT + group * 4) * H + head
        tl.store(chunk_gates + gate_base, first_gate)
        tl.store(chunk_gates + gate_base + H, second_gate)
        tl.store(chunk_gates + gate_base + 2 * H, third_gate)
        tl.store(chunk_gates + gate_base + 3 * H, fourth_gate)


@triton.jit
def _prepare_weights_kernel(
    g,
    level_scales,
    level_lut,
    scores,
    chunk_gates,
    weighted_scores,
    query_factors,
    T: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    NT: tl.constexpr,
    NK: tl.constexpr,
    NUM_LEVELS: tl.constexpr,
    MAX_BLOCK: tl.constexpr,
    WEIGHTED_SCORES: tl.constexpr,
    PRECOMPUTE_QUERY: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    batch = batch_head // H
    head = batch_head % H
    rows = tl.arange(0, BLOCK_T)
    tokens = chunk * BLOCK_T + rows
    gates = tl.load(g + (batch * T + tokens) * H + head)
    if WEIGHTED_SCORES:
        columns = tl.arange(0, BLOCK_T)
        levels = tl.load(level_lut + rows[:, None] * BLOCK_T + columns[None, :])
        scales = tl.load(level_scales + ((batch * T + tokens[:, None]) * H + head) * L + levels).to(tl.float32)
        factors = tl.exp(gates[:, None] - gates[None, :]) * scales
        tile_offsets = rows[:, None] * BLOCK_T + columns[None, :]
        for key_tile in tl.static_range(NK):
            score_base = ((batch * NT + chunk) * NK + key_tile) * BLOCK_T * BLOCK_T
            score = tl.load(scores + score_base + tile_offsets)
            weighted = tl.where(rows[:, None] >= columns[None, :], score * factors, 0.0)
            output_base = (((batch * NT + chunk) * H + head) * NK + key_tile) * BLOCK_T * BLOCK_T
            tl.store(weighted_scores + output_base + tile_offsets, weighted.to(weighted_scores.dtype.element_ty))
    if PRECOMPUTE_QUERY:
        current_end = chunk - 1
        later_decay = tl.zeros([], tl.float32)
        for level in tl.static_range(NUM_LEVELS):
            block_size = 1 << level
            if chunk & block_size != 0:
                scale = tl.load(level_scales + ((batch * T + tokens) * H + head) * L + 7 + level).to(tl.float32)
                factor = scale * tl.exp(gates + later_decay)
                factor_base = (((batch * NT + chunk) * H + head) * NUM_LEVELS + level) * BLOCK_T
                tl.store(query_factors + factor_base + rows, factor)
                offsets = tl.arange(0, MAX_BLOCK)
                gate_offsets = (batch * NT + current_end - block_size + 1 + offsets) * H + head
                later_decay += tl.sum(tl.load(chunk_gates + gate_offsets, offsets < block_size, 0.0), axis=0)
                current_end -= block_size


@triton.jit
def _fused_reduce_output_kernel(
    q,
    k,
    v,
    g,
    level_scales,
    level_lut,
    scores,
    states,
    chunk_gates,
    output,
    query_factors,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    NT: tl.constexpr,
    NUM_LEVELS: tl.constexpr,
    MAX_BLOCK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    PRECOMPUTE_SCORES: tl.constexpr,
    CACHE_STATES: tl.constexpr,
    WEIGHTED_SCORES: tl.constexpr,
    PRECOMPUTE_QUERY: tl.constexpr,
    TILE_FIRST: tl.constexpr,
):
    accumulator = tl.zeros((BLOCK_T, BV), tl.float32)
    for key_tile in tl.static_range(NK):
        local_output, query, gates = _local_fwd_kernel(
            q=q,
            k=k,
            v=v,
            g=g,
            level_scales=level_scales,
            level_lut=level_lut,
            local_output=None,
            states=None,
            chunk_gates=chunk_gates,
            scores=scores,
            cached_states=None,
            T=T,
            H=H,
            K=K,
            V=V,
            L=L,
            NT=NT,
            BK=BK,
            BV=BV,
            NK=NK,
            NV=NV,
            PRECOMPUTE_SCORES=PRECOMPUTE_SCORES,
            CACHE_STATES=CACHE_STATES,
            RETURN_TILE=True,
            KEY_TILE=key_tile,
            WEIGHTED_SCORES=WEIGHTED_SCORES,
            TILE_FIRST=TILE_FIRST,
        )
        partial = _inter_fwd_kernel(
            q=q,
            g=g,
            level_scales=level_scales,
            local_output=None,
            states=states,
            chunk_gates=chunk_gates,
            output=output,
            B=B,
            T=T,
            H=H,
            K=K,
            V=V,
            L=L,
            NT=NT,
            NUM_LEVELS=NUM_LEVELS,
            MAX_BLOCK=MAX_BLOCK,
            BK=BK,
            BV=BV,
            NK=NK,
            NV=NV,
            CACHE_STATES=CACHE_STATES,
            initial_output=local_output,
            initial_q=query,
            initial_g=gates,
            query_factors=query_factors,
            PRECOMPUTE_QUERY=PRECOMPUTE_QUERY,
            TILE_FIRST=TILE_FIRST,
            USE_INITIAL_TILES=True,
            KEY_TILE=key_tile,
            RETURN_TILE=True,
        )
        accumulator += partial.to(output.dtype.element_ty).to(tl.float32)
    if TILE_FIRST:
        index = tl.program_id(0).to(tl.int64)
        chunk = index // NV % NT
        batch_head = index // (NV * NT)
        value_tile = (index % NV).to(tl.int32)
    else:
        chunk = tl.program_id(0).to(tl.int64)
        batch_head = tl.program_id(1).to(tl.int64)
        value_tile = tl.program_id(2)
    batch = batch_head // H
    head = batch_head % H
    tokens = chunk * BLOCK_T + tl.arange(0, BLOCK_T)
    values = value_tile * BV + tl.arange(0, BV)
    offsets = ((batch * T + tokens[:, None]) * H + head) * V + values[None, :]
    tl.store(output + offsets, accumulator.to(output.dtype.element_ty))


@triton.jit
def _fused_output_kernel(
    q,
    k,
    v,
    g,
    level_scales,
    level_lut,
    scores,
    states,
    chunk_gates,
    output,
    query_factors,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    NT: tl.constexpr,
    NUM_LEVELS: tl.constexpr,
    MAX_BLOCK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    PRECOMPUTE_SCORES: tl.constexpr,
    CACHE_STATES: tl.constexpr,
    WEIGHTED_SCORES: tl.constexpr,
    PRECOMPUTE_QUERY: tl.constexpr,
    TILE_FIRST: tl.constexpr,
):
    local_output, query, gates = _local_fwd_kernel(
        q=q,
        k=k,
        v=v,
        g=g,
        level_scales=level_scales,
        level_lut=level_lut,
        local_output=None,
        states=None,
        chunk_gates=chunk_gates,
        scores=scores,
        cached_states=None,
        T=T,
        H=H,
        K=K,
        V=V,
        L=L,
        NT=NT,
        BK=BK,
        BV=BV,
        NK=NK,
        NV=NV,
        PRECOMPUTE_SCORES=PRECOMPUTE_SCORES,
        CACHE_STATES=CACHE_STATES,
        RETURN_TILE=True,
        WEIGHTED_SCORES=WEIGHTED_SCORES,
        TILE_FIRST=TILE_FIRST,
    )
    _inter_fwd_kernel(
        q=q,
        g=g,
        level_scales=level_scales,
        local_output=None,
        states=states,
        chunk_gates=chunk_gates,
        output=output,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        L=L,
        NT=NT,
        NUM_LEVELS=NUM_LEVELS,
        MAX_BLOCK=MAX_BLOCK,
        BK=BK,
        BV=BV,
        NK=NK,
        NV=NV,
        CACHE_STATES=CACHE_STATES,
        initial_output=local_output,
        initial_q=query,
        initial_g=gates,
        query_factors=query_factors,
        PRECOMPUTE_QUERY=PRECOMPUTE_QUERY,
        TILE_FIRST=TILE_FIRST,
        USE_INITIAL_TILES=True,
    )
