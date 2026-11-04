# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Packed dense attention for MACA's 64 KiB shared-memory budget.

The short-sequence branch must not call the legacy CUDA FlashAttention D128
configuration: it needs 72 KiB on C550. This kernel keeps packed variable
lengths and GQA on device, and uses the same suffix-aligned causal mask as
InfLLM-V2. Gradients reuse InfLLM-V2's independently tested sparse backward
with every KV block selected.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _dense_kernel(Q, K, V, CQ, CK, O, LSE,
                  HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                  SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                  BD: tl.constexpr, STORE_LSE: tl.constexpr):
    tile, h, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qs, qe = tl.load(CQ + batch), tl.load(CQ + batch + 1)
    ks, ke = tl.load(CK + batch), tl.load(CK + batch + 1)
    rm = tile * BM + tl.arange(0, BM)
    if tile * BM >= qe - qs:
        return
    rn, rd = tl.arange(0, BN), tl.arange(0, BD)
    hk = h // (HQ // HK)
    qoff = (qs + rm[:, None]) * HQ * D + h * D + rd[None, :]
    q = tl.load(Q + qoff, (rm[:, None] < qe - qs) & (rd[None, :] < D), 0)
    absolute = ke - ks - (qe - qs) + rm
    mi = tl.full((BM,), -float('inf'), tl.float32)
    li = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    end = tl.minimum(ke - ks, ke - ks - (qe - qs) + (tile + 1) * BM)
    for start in range(0, end, BN):
        pos = start + rn
        off = (ks + pos[:, None]) * HK * D + hk * D + rd[None, :]
        mask = (pos[:, None] < ke - ks) & (rd[None, :] < D)
        k, v = tl.load(K + off, mask, 0), tl.load(V + off, mask, 0)
        score = tl.dot(q, tl.trans(k)) * (SCALE * log2e)
        score = tl.where((pos[None, :] < ke - ks) & (pos[None, :] <= absolute[:, None]), score, -float('inf'))
        mn = tl.maximum(mi, tl.max(score, 1))
        alpha = tl.exp2(mi - mn)
        p = tl.exp2(score - mn[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        li = li * alpha + tl.sum(p, 1)
        mi = mn
    tl.store(O + qoff, acc / li[:, None], (rm[:, None] < qe - qs) & (rd[None, :] < D))
    if STORE_LSE:
        tl.store(LSE + (qs + rm) * HQ + h, (mi + tl.log2(li)) / log2e, rm < qe - qs)


def _forward(q, k, v, cq, ck, max_q, scale, store_lse):
    tq, hq, d = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((tq, hq), device=q.device, dtype=torch.float32) if store_lse else out
    _dense_kernel[(triton.cdiv(max_q, 32), hq, cq.numel()-1)](
        q, k, v, cq, ck, out, lse, hq, k.shape[1], d,
        d**-0.5 if scale is None else scale, 32, 64 if d <= 128 else 32,
        max(16, triton.next_power_of_2(d)), store_lse, num_warps=4, num_stages=1,
    )
    return out, lse


class _DenseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cq, ck, max_q, scale):
        out, lse = _forward(q, k, v, cq, ck, max_q, scale, True)
        ctx.save_for_backward(q, k, v, cq, ck, out, lse)
        ctx.max_q, ctx.scale = max_q, scale
        return out

    @staticmethod
    def backward(ctx, dout):
        from .backward import sparse_backward
        q, k, v, cq, ck, out, lse = ctx.saved_tensors
        max_k = int((ck[1:] - ck[:-1]).max().item())
        blocks = triton.cdiv(max_k, 64)
        # IDs beyond a shorter sample's length are masked by sparse_backward.
        ids = torch.arange(blocks, device=q.device, dtype=torch.int32)
        selected = ids.view(1, 1, blocks).expand(k.shape[1], q.shape[0], blocks).contiguous()
        dq, dk, dv = sparse_backward(q,k,v,out,dout.contiguous(),lse,selected,cq,ck,ctx.max_q,64,ctx.scale,True)
        return dq, dk, dv, None, None, None, None


def dense_attention(q, k, v, cq, ck, max_q, scale=None):
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return _DenseAttention.apply(q, k, v, cq, ck, max_q, scale)
    return _forward(q, k, v, cq, ck, max_q, scale, False)[0]
