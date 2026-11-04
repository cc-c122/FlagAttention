# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Device-only packed metadata and fused score pooling / exact TopK."""
import torch
import triton
import triton.language as tl
from .forward import _compress_k_kernel, stage1, _topk_float_to_ordered_key, _topk_index_to_key, _topk_key_to_index


@triton.jit
def _compress_dual_c550(
    K,
    CK,
    KFine,
    KCoarse,
    C1,
    C2,
    B: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    N1: tl.constexpr,
    K1: tl.constexpr,
    S1: tl.constexpr,
    K2: tl.constexpr,
    S2: tl.constexpr,
    BB: tl.constexpr,
    W1: tl.constexpr,
    W2: tl.constexpr,
    BD: tl.constexpr,
):
    chunk, hk, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bi = tl.arange(0, BB)
    starts = tl.load(CK + bi, bi < B, 0)
    ends = tl.load(CK + bi + 1, bi < B, 0)
    counts1 = tl.where((bi < B) & (ends - starts >= K1), (ends - starts - K1) // S1 + 1, 0)
    counts2 = tl.where((bi < B) & (ends - starts >= K2), (ends - starts - K2) // S2 + 1, 0)
    base1 = tl.sum(tl.where(bi < batch, counts1, 0), 0)
    base2 = tl.sum(tl.where(bi < batch, counts2, 0), 0)
    length1 = tl.sum(tl.where(bi == batch, counts1, 0), 0)
    length2 = tl.sum(tl.where(bi == batch, counts2, 0), 0)
    if chunk == 0 and hk == 0:
        tl.store(C1 + batch, base1)
        tl.store(C2 + batch, base2)
        if batch == B - 1:
            tl.store(C1 + B, base1 + length1)
            tl.store(C2 + B, base2 + length2)
    start = tl.load(CK + batch)
    end = tl.load(CK + batch + 1)
    d = tl.arange(0, BD)
    if chunk < N1:
        if chunk < length1:
            w = tl.arange(0, W1)
            token = start + chunk * S1 + w
            values = tl.load(
                K + token[:, None] * HK * D + hk * D + d[None, :],
                (w[:, None] < K1) & (token[:, None] < end) & (d[None, :] < D),
                0,
            ).to(tl.float32)
            mean = tl.sum(values, 0) / K1
            tl.store(KFine + (base1 + chunk) * HK * D + hk * D + d, mean, d < D)
    else:
        coarse = chunk - N1
        if coarse < length2:
            w = tl.arange(0, W2)
            token = start + coarse * S2 + w
            values = tl.load(
                K + token[:, None] * HK * D + hk * D + d[None, :],
                (w[:, None] < K2) & (token[:, None] < end) & (d[None, :] < D),
                0,
            ).to(tl.float32)
            mean = tl.sum(values, 0) / K2
            tl.store(KCoarse + (base2 + coarse) * HK * D + hk * D + d, mean, d < D)


@triton.jit
def _compressed_metadata(
    CK,
    C1,
    C2,
    B: tl.constexpr,
    K1: tl.constexpr,
    S1: tl.constexpr,
    K2: tl.constexpr,
    S2: tl.constexpr,
    BB: tl.constexpr,
):
    i = tl.arange(0, BB)
    a = tl.load(CK + i, i < B, 0)
    b = tl.load(CK + i + 1, i < B, 0)
    n1 = tl.where((i < B) & (b - a >= K1), (b - a - K1) // S1 + 1, 0)
    n2 = tl.where((i < B) & (b - a >= K2), (b - a - K2) // S2 + 1, 0)
    tl.store(C1 + i + 1, tl.cumsum(n1), i < B)
    tl.store(C2 + i + 1, tl.cumsum(n2), i < B)
    tl.store(C1, 0)
    tl.store(C2, 0)


@triton.jit
def _pooled_keys(
    Score,
    blocks,
    valid,
    forced,
    fine_len,
    qindex,
    hk,
    TQ: tl.constexpr,
    N1: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    WINDOW: tl.constexpr,
):
    value = tl.full(blocks.shape, -float("inf"), tl.float32)
    for j in tl.static_range(WINDOW):
        fine = blocks * STRIDE - PAD + j
        x = tl.load(
            Score + hk * TQ * N1 + qindex * N1 + fine,
            valid & (~forced) & (fine >= 0) & (fine < fine_len),
            -float("inf"),
        )
        value = tl.maximum(value, x)
    value = tl.where(forced, float("inf"), value)
    value = tl.where(value == value, value, -float("inf"))
    score_key = _topk_float_to_ordered_key(value.to(tl.uint32, bitcast=True))
    key = (score_key.to(tl.uint64) << 16) | _topk_index_to_key(blocks).to(tl.uint64)
    return tl.where(valid, key, tl.zeros_like(key))


@triton.jit
def _pool_select_c550(
    Score,
    CQ,
    CK,
    C1,
    Selected,
    TQ: tl.constexpr,
    N1: tl.constexpr,
    NB: tl.constexpr,
    BS: tl.constexpr,
    TOPK: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    WINDOW: tl.constexpr,
    INIT: tl.constexpr,
    LOCAL: tl.constexpr,
    BB: tl.constexpr,
    IDENTITY: tl.constexpr,
):
    qi, hk, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qs, qe = tl.load(CQ + batch), tl.load(CQ + batch + 1)
    if qi >= qe - qs:
        return
    kl = tl.load(CK + batch + 1) - tl.load(CK + batch)
    position = kl - (qe - qs) + qi
    count = tl.minimum(NB, tl.minimum(tl.cdiv(kl, BS), tl.cdiv(position + 1, BS)))
    ranks = tl.arange(0, TOPK)
    dest = Selected + ((hk * TQ + qs + qi) * TOPK) + ranks
    if IDENTITY:
        tl.store(dest, tl.where(ranks < count, ranks, -1))
    else:
        if count <= TOPK:
            tl.store(dest, tl.where(ranks < count, ranks, -1))
            return
        fine_len = tl.load(C1 + batch + 1) - tl.load(C1 + batch)
        tiles = tl.cdiv(count, BB)
        start = (tiles - 1) * BB
        offsets = tl.arange(0, BB)
        blocks = start + offsets
        qblock = position // BS
        forced = (blocks < INIT) | ((qblock >= blocks) & (qblock <= blocks + LOCAL))
        keys = _pooled_keys(Score, blocks, blocks < count, forced, fine_len, qs + qi, hk, TQ, N1, STRIDE, PAD, WINDOW)
        best = tl.max(keys, 0, keep_dims=True) if TOPK == 1 else tl.topk(keys, TOPK)
        for _ in tl.range(0, tiles - 1):
            if TOPK > 1:
                best = tl.bitonic_merge(best)
            start -= BB
            blocks = start + offsets
            forced = (blocks < INIT) | ((qblock >= blocks) & (qblock <= blocks + LOCAL))
            keys = _pooled_keys(
                Score, blocks, blocks < count, forced, fine_len, qs + qi, hk, TQ, N1, STRIDE, PAD, WINDOW
            )
            top = tl.max(keys, 0, keep_dims=True) if TOPK == 1 else tl.topk(keys, TOPK)
            best = tl.maximum(best, top)
        best = (best << 48) | (best >> 16)
        if TOPK > 1:
            best = tl.sort(best, descending=True)
        selected = _topk_key_to_index((best >> 48).to(tl.uint32))
        tl.store(dest, tl.where(selected < count, selected, -1))


def select(q, k, cq, ck, max_q, max_k, config, scale):
    batch = cq.numel() - 1
    tq, hq, d = q.shape
    hk = k.shape[1]
    bs, topk = config.block_size, config.topk
    nb = triton.cdiv(max_k, bs)
    if nb > 4096:
        raise ValueError("TopK currently supports at most 4096 KV blocks")
    if topk <= 0 or topk > 2048 or topk & (topk - 1):
        raise ValueError("TopK must be a power of two no greater than 2048")
    selected = torch.empty((hk, tq, topk), device=q.device, dtype=torch.int32)
    n1 = max(0, (max_k - config.k1_kernel_size) // config.k1_stride + 1)
    if nb <= topk:
        # The canonical selection is the identity, independent of scores.
        c1 = ck
        score = q
    else:
        n2 = max(0, (max_k - config.k2_kernel_size) // config.k2_stride + 1)
        c1, c2 = torch.empty_like(ck), torch.empty_like(ck)
        if batch > 32:
            _compressed_metadata[(1,)](
                ck,
                c1,
                c2,
                batch,
                config.k1_kernel_size,
                config.k1_stride,
                config.k2_kernel_size,
                config.k2_stride,
                triton.next_power_of_2(batch),
                num_warps=4,
            )
        compressed = []
        for n, window, stride, cu in [
            (n1, config.k1_kernel_size, config.k1_stride, c1),
            (n2, config.k2_kernel_size, config.k2_stride, c2),
        ]:
            capacity = min(batch * n, k.shape[0] // stride + batch)
            output = torch.empty((capacity, hk, d), device=k.device, dtype=k.dtype)
            if n and batch > 32:
                _compress_k_kernel[(n, hk, batch)](
                    k,
                    output,
                    ck,
                    cu,
                    hk,
                    d,
                    window,
                    stride,
                    triton.next_power_of_2(window),
                    triton.next_power_of_2(d),
                    num_warps=4,
                )
            compressed.append(output)
        if batch <= 32:
            # One launch populates both means and metadata. Each CTA computes
            # its own packed prefix, so no cross-CTA synchronization is needed.
            _compress_dual_c550[(max(1, n1 + n2), hk, batch)](
                k,
                ck,
                *compressed,
                c1,
                c2,
                batch,
                hk,
                d,
                n1,
                config.k1_kernel_size,
                config.k1_stride,
                config.k2_kernel_size,
                config.k2_stride,
                triton.next_power_of_2(batch),
                triton.next_power_of_2(config.k1_kernel_size),
                triton.next_power_of_2(config.k2_kernel_size),
                triton.next_power_of_2(d),
                num_warps=4,
                num_stages=1,
            )
        score = stage1(
            q,
            *compressed,
            cq,
            c1,
            c2,
            max_q,
            config.k1_stride,
            config.k2_stride,
            scale,
            config.causal,
            ck,
            _max_k1=n1,
            _selection_topk=topk,
            _selection_block_size=bs,
            _selection_local_blocks=config.local_blocks,
        )
    bb = max(64, topk, triton.next_power_of_2(min(nb, 1024)))
    _pool_select_c550[(max_q, hk, batch)](
        score,
        cq,
        ck,
        c1,
        selected,
        tq,
        n1,
        nb,
        bs,
        topk,
        bs // config.k1_stride,
        config.k1_kernel_size // config.k1_stride - 1,
        config.k1_kernel_size // config.k1_stride + bs // config.k1_stride - 1,
        config.init_blocks,
        config.local_blocks,
        bb,
        nb <= topk,
        num_warps=2 if bb <= 64 else 4 if bb <= 128 else 8,
        num_stages=1,
    )
    return selected
