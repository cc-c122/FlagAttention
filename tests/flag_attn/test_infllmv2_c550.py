# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""C550 coverage beyond the upstream small-shape correctness suite."""
import pytest
import torch

from flag_attn import InfLLMV2Config, infllmv2_attention, infllmv2_decode
from flag_attn.infllmv2.forward import _is_metax, select_blocks, sparse_attention
from flag_attn.infllmv2.reference import dense_attention_ref, select_blocks_ref
from test_infllmv2_attention import _packed_inputs, _reference_attention

pytestmark = [
    pytest.mark.infllmv2_attention,
    pytest.mark.skipif(not torch.cuda.is_available() or not _is_metax(), reason="requires MetaX C550"),
]


def _vector_sparse_reference(q, k, v, ids, cq, ck, block_size=64, softmax_scale=None, causal=True):
    # Independent FP32 dense formula with an explicit mask for the selected union.
    # Avoid the upstream reference's per-query/per-head device-to-host loop on
    # long tests, while leaving all compression/ranking references unchanged.
    chunks = []
    group = q.shape[1] // k.shape[1]
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    qc, kc = cq.cpu().tolist(), ck.cpu().tolist()
    for qs, qe, ks, ke in zip(qc[:-1], qc[1:], kc[:-1], kc[1:]):
        qh = q[qs:qe].float().transpose(0, 1)
        kh = k[ks:ke].float().repeat_interleave(group, 1).transpose(0, 1)
        vh = v[ks:ke].float().repeat_interleave(group, 1).transpose(0, 1)
        pos = torch.arange(ke - ks, device=q.device)
        allowed = (
            (ids[:, qs:qe, :, None] == (pos // block_size)[None, None, None, :]).any(2).repeat_interleave(group, 0)
        )
        if causal:
            allowed = allowed & (
                pos[None, None, :] <= (ke - ks - (qe - qs) + torch.arange(qe - qs, device=q.device))[None, :, None]
            )
        scores = (qh @ kh.transpose(1, 2)) * scale
        prob = torch.softmax(scores.masked_fill(~allowed, -torch.inf), -1).nan_to_num(0.0)
        chunks.append((prob @ vh).transpose(0, 1).to(q.dtype))
    return torch.cat(chunks)


def _pipeline_reference(q, k, v, cq, ck, maxq, maxk, cfg):
    from flag_attn.infllmv2.reference import compress_k_ref, stage1_ref, pool_scores_ref

    with torch.no_grad():
        k1, c1 = compress_k_ref(k, ck, cfg.k1_kernel_size, cfg.k1_stride)
        k2, c2 = compress_k_ref(k, ck, cfg.k2_kernel_size, cfg.k2_stride)
        s = stage1_ref(q, k1, k2, cq, c1, c2, cfg.k1_stride, cfg.k2_stride, cu_seqlens_k=ck)
        p = pool_scores_ref(
            s,
            cq,
            c1,
            maxq,
            cfg.k1_kernel_size,
            cfg.k1_stride,
            cfg.block_size,
            cfg.init_blocks,
            cfg.local_blocks,
            ck,
            maxk,
        )
        ids = select_blocks_ref(p, cq, cfg.topk, cfg.block_size, ck)
    return _vector_sparse_reference(q, k, v, ids, cq, ck, cfg.block_size)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d,group", [(32, 1), (64, 3), (64, 16), (128, 4), (128, 16)])
@pytest.mark.parametrize("lengths,kv", [((1, 3), (17, 31)), ((17, 33), (63, 129)), ((65, 97), (129, 257))])
def test_all_blocks_against_independent_dense(dtype, d, group, lengths, kv):
    q, k, v, cq, ck = _packed_inputs(lengths, kv, d=d, hq=2 * group, hkv=2, dtype=dtype, seed=113)
    # All visible blocks fit in topk; this is exactly causal dense attention,
    # regardless of the selector's floating-point ranking.
    config = InfLLMV2Config(topk=8, local_blocks=2, dense_len=0)
    out = infllmv2_attention(q, k, v, cq, ck, max(lengths), max(kv), config=config)
    ref = dense_attention_ref(q, k, v, cq, ck)
    torch.testing.assert_close(out, ref, atol=4e-3 if dtype == torch.bfloat16 else 5e-4, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("m", [1, 17, 65])
@pytest.mark.parametrize("group", [4, 16])
def test_gradients_both_backward_paths(dtype, d, m, group):
    q, k, v, cq, ck = _packed_inputs((m,), (193,), d=d, hq=2 * group, hkv=2, dtype=dtype, requires_grad=True, seed=211)
    refs = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
    ids = torch.tensor([0, 2, 3, -1], device=q.device, dtype=torch.int32).expand(2, m, 4).contiguous()
    out = sparse_attention(q, k, v, ids, cq, ck, m)
    ref = _vector_sparse_reference(*refs, ids, cq, ck)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    tol = 1e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)
    for x, y in zip((q, k, v), refs):
        torch.testing.assert_close(x.grad, y.grad, atol=tol, rtol=tol)


@pytest.mark.parametrize("d,dtype", [(64, torch.float16), (128, torch.bfloat16)])
def test_long_sparse_pipeline(d, dtype):
    q, k, v, cq, ck = _packed_inputs((1024,), (2049,), d=d, hq=32, hkv=2, dtype=dtype, seed=383)
    config = InfLLMV2Config(topk=8, local_blocks=2, dense_len=0)
    out = infllmv2_attention(q, k, v, cq, ck, 1024, 2049, config=config)
    ref = _pipeline_reference(q, k, v, cq, ck, 1024, 2049, config)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=4e-2)


@pytest.mark.parametrize("blocks,topk", [(65, 1), (257, 4), (1025, 64), (4096, 64)])
@pytest.mark.parametrize("ties", [False, True])
def test_topk_multitile_and_ties(blocks, topk, ties):
    torch.manual_seed(733)
    score = (
        torch.randint(-3, 4, (2, 3, blocks), device="cuda").float()
        if ties
        else torch.randn((2, 3, blocks), device="cuda")
    )
    score[:, :, 0] = float("nan")
    score[:, :, 1] = float("-inf")
    score[:, :, 2] = float("inf")
    cq = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    ck = torch.tensor([0, blocks * 64 - 17], device="cuda", dtype=torch.int32)
    out = select_blocks(score, cq, topk, cu_seqlens_k=ck)
    # The production selector explicitly treats NaN as -inf.
    ref = select_blocks_ref(torch.where(torch.isnan(score), -torch.inf, score), cq, topk, cu_seqlens_k=ck)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@pytest.mark.parametrize("dense", [False, True])
def test_noncontiguous_inputs_and_nondefault_stream(dense):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        q, k, v, cq, ck = _packed_inputs((17, 31), (97, 193), d=128, hq=8, hkv=2, seed=557)
        q, k, v = [torch.stack([x, x], dim=-1)[..., 0] for x in (q, k, v)]
        cq, ck = [torch.stack([x, x], dim=-1)[:, 0] for x in (cq, ck)]
        config = InfLLMV2Config(topk=4, dense_len=8192 if dense else 0)
        out = infllmv2_attention(q, k, v, cq, ck, 31, 193, config=config)
        ref = dense_attention_ref(q, k, v, cq, ck)
        torch.testing.assert_close(out, ref, atol=4e-3, rtol=2e-2)
    stream.synchronize()


def test_decode_default_configuration_long_cache():
    q, k, v, cq, ck = _packed_inputs((1, 1), (8192, 8257), d=128, hq=32, hkv=2, seed=887)
    config = InfLLMV2Config()
    out = infllmv2_decode(q, k, v, ck, 8257, config=config)
    ref = _reference_attention(q, k, v, cq, ck, 1, 8257, config)
    torch.testing.assert_close(out, ref, atol=4e-3, rtol=2e-2)


def test_fresh_tensors_on_repeated_calls():
    config = InfLLMV2Config(topk=4, dense_len=0)
    for seed in [13, 29, 41]:
        q, k, v, cq, ck = _packed_inputs((17,), (193,), d=64, hq=8, hkv=2, seed=seed)
        out = infllmv2_attention(q, k, v, cq, ck, 17, 193, config=config)
        ref = dense_attention_ref(q, k, v, cq, ck)
        torch.testing.assert_close(out, ref, atol=4e-3, rtol=2e-2)


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("m", [1, 17, 65])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dense_packed_gradients(d, m, dtype):
    q, k, v, cq, ck = _packed_inputs(
        (m, m + 2), (129, 193), d=d, hq=8, hkv=2, dtype=dtype, requires_grad=True, seed=991
    )
    refs = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
    out = infllmv2_attention(q, k, v, cq, ck, m + 2, 193)
    ref = dense_attention_ref(*refs, cq, ck)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    tol = 1e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)
    for x, y in zip((q, k, v), refs):
        torch.testing.assert_close(x.grad, y.grad, atol=tol, rtol=tol)


@pytest.mark.parametrize("d", [60, 100, 256])
@pytest.mark.parametrize("dense", [False, True])
def test_masked_head_dimensions(d, dense):
    q, k, v, cq, ck = _packed_inputs((17,), (129,), d=d, hq=4, hkv=1, dtype=torch.float16, seed=1201)
    cfg = InfLLMV2Config(topk=4, dense_len=8192 if dense else 0)
    out = infllmv2_attention(q, k, v, cq, ck, 17, 129, config=cfg, softmax_scale=0.23)
    ref = dense_attention_ref(q, k, v, cq, ck, softmax_scale=0.23)
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=1e-2)


@pytest.mark.parametrize("dense", [False, True])
def test_gqa16_large_logits_and_gradients(dense):
    q, k, v, cq, ck = _packed_inputs(
        (65, 97), (193, 257), d=128, hq=32, hkv=2, dtype=torch.bfloat16, requires_grad=True, seed=2003
    )
    with torch.no_grad():
        q.mul_(5)
        k.mul_(5)
    refs = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
    out = infllmv2_attention(q, k, v, cq, ck, 97, 257, config=InfLLMV2Config(topk=8, dense_len=8192 if dense else 0))
    ref = dense_attention_ref(*refs, cq, ck)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=2e-2)
    for x, y in zip((q, k, v), refs):
        torch.testing.assert_close(x.grad, y.grad, atol=2e-2, rtol=3e-2)
