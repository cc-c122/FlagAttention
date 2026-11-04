# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""Independent FP32 reference checks for C550 generalized tiles and gradients."""
import math
import pytest
import torch

from flag_attn import infllmv2_attention
from flag_attn.infllmv2 import c550_reuse as REUSE
from flag_attn.infllmv2.forward import _is_metax
from flag_attn.infllmv2.reference import dense_attention_ref

pytestmark = [
    pytest.mark.infllmv2_attention,
    pytest.mark.skipif(not torch.cuda.is_available() or not _is_metax(), reason="requires MetaX C550"),
]


def packed(lengths, g, d, dtype, hk=2):
    torch.manual_seed(4419)
    cq = [0]
    ck = [0]
    for m, n in lengths:
        cq.append(cq[-1] + m)
        ck.append(ck[-1] + n)
    q = torch.randn(cq[-1], g * hk, d, device="cuda", dtype=dtype)
    k = torch.randn(ck[-1], hk, d, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    return (
        q,
        k,
        v,
        torch.tensor(cq, device="cuda", dtype=torch.int32),
        torch.tensor(ck, device="cuda", dtype=torch.int32),
    )


def reference(q, k, v, ids, cq, ck, bs):
    """FP32 dense softmax masked by actual selected block membership."""
    hq, d = q.shape[1:]
    hk = k.shape[1]
    g = hq // hk
    out = torch.zeros_like(q)
    lse = torch.full(q.shape[:2], -torch.inf, device=q.device)
    qc = cq.tolist()
    kc = ck.tolist()
    for b in range(len(qc) - 1):
        qs, qe = qc[b : b + 2]
        ks, ke = kc[b : b + 2]
        n = ke - ks
        m = qe - qs
        pos = torch.arange(n, device=q.device)
        absolute = n - m + torch.arange(m, device=q.device)
        for h in range(hk):
            selected = ids[h, qs:qe].long()
            counts = torch.zeros((m, max(1, math.ceil(n / bs))), device=q.device, dtype=torch.int32)
            valid = (selected >= 0) & (selected < counts.shape[1])
            counts.scatter_add_(1, selected.clamp(0, counts.shape[1] - 1), valid.int())
            mask = (counts[:, pos // bs] > 0) & (pos[None, :] <= absolute[:, None])
            qh = q[qs:qe, h * g : (h + 1) * g].float().transpose(0, 1)
            score = (qh @ k[ks:ke, h].float().T) * (d**-0.5)
            score = score.masked_fill(~mask[None, :, :], -torch.inf)
            probability = torch.nan_to_num(score.softmax(-1), nan=0.0)
            out[qs:qe, h * g : (h + 1) * g] = (probability @ v[ks:ke, h].float()).transpose(0, 1).to(q.dtype)
            lse[qs:qe, h * g : (h + 1) * g] = score.logsumexp(-1).T
    return out, lse


SEMANTIC_CASES = [
    (1, 64, 64, 64),
    (2, 128, 32, 32),
    (3, 96, 128, 32),
    (4, 32, 64, 128),
    (5, 80, 64, 64),
    (8, 128, 64, 64),
    (12, 160, 32, 128),
    (16, 192, 64, 64),
    (32, 256, 128, 64),
    (64, 128, 64, 64),
    (4, 224, 64, 64),
    (4, 64, 16, 32),
    (4, 64, 64, 16),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("g,d,bs,topk", SEMANTIC_CASES)
def test_reference_membership_padding_and_empty(dtype, g, d, bs, topk):
    init, local = 1, 8
    boundary = (init + local + 1) * bs
    lengths = [(7, boundary + 3), (5, boundary - 1), (5, 3), (9, 4099)]
    q, k, v, cq, ck = packed(lengths, g, d, dtype)
    ids = torch.full((2, q.shape[0], topk), -1, dtype=torch.int32)
    offset = 0
    for m, n in lengths:
        for qi in range(m):
            for h in range(2):
                blocks = [b for b in range(max(0, (n - m + qi) // bs + 1)) if (b + qi + h) % 3 != 0][-topk:]
                if qi % 5 == 0:
                    blocks = []
                ids[h, offset + qi, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        offset += m
    ids = ids.cuda()
    cfg = REUSE.reuse_config(g, d, dtype, bs, topk, (init, local), 1024, 1024)
    assert cfg is not None
    out, lse = REUSE.sparse_reuse(
        q, k, v, ids, cq, ck, 9, bs, store_lse=True, init_blocks=init, local_blocks=local, **cfg
    )
    expected, el = reference(q, k, v, ids, cq, ck, bs)
    torch.testing.assert_close(
        out, expected, atol=0.012 if dtype == torch.bfloat16 else 0.002, rtol=0.03 if dtype == torch.bfloat16 else 0.01
    )
    torch.testing.assert_close(lse, el, atol=0.0001, rtol=0.0001)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "g,d,m,n",
    [
        (1, 32, 65, 193),
        (4, 96, 65, 193),
        (4, 192, 129, 257),
        (32, 256, 33, 193),
        (4, 96, 1, 193),
        (4, 256, 1, 193),
        (64, 128, 17, 129),
        (64, 256, 17, 129),
        (24, 192, 17, 129),
    ],
)
def test_dense_padded_dim_reference_gradients(dtype, g, d, m, n):
    inputs = packed([(m, n)], g, d, dtype)
    for tensor in inputs[:3]:
        tensor.mul_(0.2).requires_grad_(True)
    actual = infllmv2_attention(*inputs, m, n)
    expected = dense_attention_ref(*inputs)
    grad = torch.randn_like(actual)
    ag = torch.autograd.grad(actual, inputs[:3], grad)
    eg = torch.autograd.grad(expected, inputs[:3], grad)
    for a, b in zip((actual, *ag), (expected, *eg)):
        torch.testing.assert_close(
            a, b, atol=0.008 if dtype == torch.bfloat16 else 0.002, rtol=0.05 if dtype == torch.bfloat16 else 0.02
        )
