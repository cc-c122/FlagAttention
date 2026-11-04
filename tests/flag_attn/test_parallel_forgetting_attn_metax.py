# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""C550 regressions for the registered parallel_forgetting_attn operator.

The shared test_forgetting_attention module already covers all 90 historical
shapes with two seeds. These cases cover the C550 scan tree, MMA ordering,
pruning boundaries, dispatch, live inputs, streams, and launch-cache behavior.
All references and fixtures come from this checkout; no external workdir is
required. Non-MetaX runs collect these tests and skip execution.
"""
import importlib
import math

import pytest
import torch

from flag_attn import runtime
from flag_attn.testing import backend as test_backend
import test_forgetting_attention as ref

pytestmark = [
    pytest.mark.parallel_forgetting_attn,
    pytest.mark.skipif(
        runtime.device.vendor_name != "metax" or not test_backend.is_available(),
        reason="requires the MetaX MACA backend",
    ),
]


@pytest.fixture(scope="module")
def c550():
    return importlib.import_module("flag_attn.runtime.backend._metax.parallel_forgetting_attn")


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("m,n", [(1, 1), (1, 129), (2, 63), (65, 129), (129, 257)])
@pytest.mark.parametrize("d", [16, 32, 60, 64, 100, 128])
@torch.inference_mode()
def test_gqa_mqa_tail_and_default_scale(dtype, m, n, d):
    case = dict(B=2, M=m, N=n, HQ=4, H=1, D=d, dtype=dtype, scale=1 / math.sqrt(d))
    inputs = ref.make_inputs(case, 13)
    out = ref.optimized_entry()(*inputs, adaptive_threshold=-10.0)
    assert out.is_contiguous()
    ref.assert_bitwise(out, ref.call_official(inputs, case))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("threshold", [0.0, -1.0, -10.0, -100000.0])
@torch.inference_mode()
def test_pruning_boundaries(dtype, threshold, c550):
    prepare = c550.prepare
    for d, qm in [(60, 64), (100, 128)]:
        case = dict(B=2, M=129, N=257, HQ=4, H=2, D=d, dtype=dtype, scale=0.125)
        inputs = ref.make_inputs(case, 71)
        prefix, starts = prepare(inputs[3], 129, qm, 32, threshold)
        expected_prefix = inputs[3].transpose(1, 2).cumsum(-1)
        assert torch.equal(prefix, expected_prefix)
        qp = 128 + torch.arange(0, 129, qm, device="cuda")
        kp = torch.arange(31, 288, 32, device="cuda").clamp_max(256)
        expected = (prefix[..., qp, None] - prefix[..., None, kp] < threshold).sum(-1) * 32
        assert torch.equal(starts, expected)
        ref.assert_bitwise(ref.call_optimized(inputs, case, threshold), ref.call_official(inputs, case, threshold))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@torch.inference_mode()
def test_live_inputs_scales_thresholds_and_batch(dtype):
    ref.test_forgetting_attention_launch_reuse(dtype)
    for b in (1, 3, 2, 1):
        case = dict(B=b, M=65, N=129, HQ=4, H=2, D=60, dtype=dtype, scale=0.125)
        inputs = ref.make_inputs(case, b + 15)
        ref.assert_bitwise(ref.call_optimized(inputs, case), ref.call_official(inputs, case))


@pytest.mark.parametrize("offset", [0, 1, 2, 3])
@torch.inference_mode()
def test_gate_alignment(offset):
    ref.test_forgetting_attention_gate_alignment(offset)


@torch.inference_mode()
def test_stream_and_tensor_lifetime():
    ref.test_forgetting_attention_nondefault_stream_and_lifetime()


def test_inference_required():
    with torch.inference_mode(False), pytest.raises(NotImplementedError, match="inference"):
        ref.optimized_entry()(*ref.make_inputs(ref._reuse_case()), adaptive_threshold=-10.0)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("m", [1, 65])
@pytest.mark.parametrize("d", [60, 128])
@torch.inference_mode()
def test_dense_math_oracle(dtype, m, d):
    case = dict(B=1, M=m, N=129, HQ=4, H=1, D=d, dtype=dtype, scale=1 / math.sqrt(d))
    q, k, v, gate = ref.make_inputs(case, 81)
    prefix = gate.transpose(1, 2).cumsum(-1)
    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).float().repeat_interleave(4, dim=1)
    vh = v.transpose(1, 2).float().repeat_interleave(4, dim=1)
    scores = qh @ kh.transpose(-1, -2) * case["scale"]
    scores += prefix[..., -m:, None] - prefix[..., None, :]
    causal = torch.arange(129, device="cuda")[None, :] <= (129 - m + torch.arange(m, device="cuda"))[:, None]
    expected = (scores.masked_fill(~causal, -torch.inf).softmax(-1) @ vh).transpose(1, 2)
    out = ref.call_optimized((q, k, v, gate), case, threshold=-100000.0)
    tol = 0.015 if dtype == "bfloat16" else 0.002
    torch.testing.assert_close(out.float(), expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("n", [8257, 16385])
@torch.inference_mode()
def test_beyond_hopper_profile(n):
    case = dict(B=1, M=65, N=n, HQ=4, H=2, D=64, dtype="bfloat16", scale=0.125)
    inputs = ref.make_inputs(case, 31)
    ref.assert_bitwise(ref.call_optimized(inputs, case), ref.call_official(inputs, case))


@pytest.mark.parametrize(
    "n", [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097]
)
@torch.inference_mode()
def test_fused_prefix_exact(n, c550):
    for b, h in [(1, 2), (2, 8), (4, 32)]:
        generator = torch.Generator(device="cuda").manual_seed(n)
        gate = -torch.rand((b, n, h), device="cuda", generator=generator)
        gate[:, ::7] = 0
        decode_bn = min(128, max(16, 1 << (n - 1).bit_length()))
        for m, qm, bn in [(1, 1, decode_bn), (n, 128, 32)]:
            p, s = c550.prepare(gate, m, qm, bn, -10.0)
            expected = gate.transpose(1, 2).cumsum(-1)
            assert torch.equal(p, expected)
            qp = n - m + torch.arange(0, m, qm, device="cuda")
            kp = torch.arange(bn - 1, n + bn - 1, bn, device="cuda").clamp_max(n - 1)
            starts = (expected[..., qp, None] - expected[..., None, kp] < -10.0).sum(-1) * bn
            assert torch.equal(s, starts)


@torch.inference_mode()
def test_warm_cache_respects_kernel_hooks(c550):
    case = dict(B=1, M=65, N=129, HQ=4, H=2, D=128, dtype="bfloat16", scale=0.125)
    inputs = ref.make_inputs(case, 31)
    ref.call_optimized(inputs, case)

    def fail(*a, **kw):
        raise RuntimeError("C550 pre-run hook sentinel")

    c550._c550_attention_d128.pre_run_hooks.append(fail)
    try:
        with pytest.raises(RuntimeError, match="C550 pre-run hook sentinel"):
            ref.call_optimized(inputs, case)
    finally:
        c550._c550_attention_d128.pre_run_hooks.remove(fail)
    ref.assert_bitwise(ref.call_optimized(inputs, case), ref.call_official(inputs, case))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@torch.inference_mode()
def test_d128_subtile_boundaries(dtype):
    for m, n in [(31, 63), (32, 65), (33, 129), (63, 129), (64, 129), (65, 257), (127, 257), (128, 257), (129, 257)]:
        c = dict(B=1, M=m, N=n, HQ=4, H=1, D=128, dtype=dtype, scale=0.125)
        x = ref.make_inputs(c, m)
        for threshold in (0.0, -1.0, -10.0, -100000.0):
            ref.assert_bitwise(ref.call_optimized(x, c, threshold), ref.call_official(x, c, threshold))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("d", [100, 128])
@torch.inference_mode()
def test_mma_pack_boundaries(dtype, d):
    for m, n in [(31, 63), (65, 129), (129, 257)]:
        c = dict(B=1, M=m, N=n, HQ=4, H=1, D=d, dtype=dtype, scale=0.3125)
        for seed in [17, 29]:
            x = ref.make_inputs(c, seed)
            for threshold in [0.0, -1.0, -10.0, -100000.0]:
                ref.assert_bitwise(ref.call_optimized(x, c, threshold), ref.call_official(x, c, threshold))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("n", [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129])
@torch.inference_mode()
def test_fused_decode_scan_chunks(dtype, n):
    for b, h, hk, d in [(1, 2, 1, 60), (2, 8, 8, 64), (4, 32, 4, 100), (1, 2, 1, 16), (1, 2, 1, 32), (4, 32, 4, 128)]:
        c = dict(B=b, M=1, N=n, HQ=h, H=hk, D=d, dtype=dtype, scale=0.3125)
        x = ref.make_inputs(c, n)
        for threshold in [0.0, -10.0]:
            ref.assert_bitwise(ref.call_optimized(x, c, threshold), ref.call_official(x, c, threshold))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("d", [32, 60, 64])
@torch.inference_mode()
def test_two_warp_prefill(dtype, d):
    c = dict(B=1, M=1024, N=1025, HQ=4, H=2, D=d, dtype=dtype, scale=0.3125)
    x = ref.make_inputs(c, 37)
    for threshold in [0.0, -10.0]:
        ref.assert_bitwise(ref.call_optimized(x, c, threshold), ref.call_official(x, c, threshold))


@pytest.mark.parametrize("n", [257, 1025, 4096])
@pytest.mark.parametrize("h", [2, 32])
@pytest.mark.parametrize("pattern", ["zeros", "mixed_magnitude"])
@torch.inference_mode()
def test_native_prefix_search(n, h, pattern, c550):
    torch.manual_seed(n + h)
    gate = -torch.rand((2, n, h), device="cuda", dtype=torch.float32)
    if pattern == "zeros":
        gate[:, 1::3] = 0.0
        gate[:, n // 3 : 2 * n // 3] = 0.0
    else:
        gate *= torch.exp2(torch.randint(-12, 12, gate.shape, device="cuda").float())
    expected = gate.transpose(1, 2).cumsum(-1)
    for m, qm, bn in [(1, 1, 128), (n, 64, 32), (n // 2, 128, 32)]:
        qp = n - m + torch.arange(0, m, qm, device="cuda")
        kp = torch.arange(bn - 1, n + bn - 1, bn, device="cuda").clamp_max(n - 1)
        delta = expected[..., qp, None] - expected[..., None, kp]
        boundary = delta[0, 0, -1, 0]
        thresholds = [
            0.0,
            -10.0,
            boundary.item(),
            torch.nextafter(boundary, torch.full_like(boundary, float("-inf"))).item(),
        ]
        for threshold in thresholds:
            prefix, starts = c550.prepare(gate, m, qm, bn, threshold)
            assert torch.equal(prefix.contiguous().view(torch.int32), expected.contiguous().view(torch.int32))
            assert torch.equal(starts, (delta < threshold).sum(-1) * bn)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@torch.inference_mode()
def test_batch_outer_partial_tiles(dtype):
    case = dict(B=2, M=2049, N=2051, HQ=4, H=2, D=128, dtype=dtype, scale=0.125)
    x = ref.make_inputs(case, 71)
    for threshold in [0.0, -10.0]:
        ref.assert_bitwise(ref.call_optimized(x, case, threshold), ref.call_official(x, case, threshold))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 1025, 1056, 4, 2, 64),
        (2, 1025, 1057, 4, 2, 64),
        (1, 2048, 2048, 8, 8, 64),
        (2, 1025, 1056, 4, 2, 32),
        (2, 1025, 1057, 4, 2, 16),
        (2, 1024, 1056, 4, 2, 60),
        (2, 1025, 1056, 4, 2, 60),
        (2, 1025, 1056, 4, 2, 100),
        (2, 1025, 1056, 4, 2, 128),
    ],
)
@pytest.mark.parametrize("gate_pattern", ["random", "zero"])
@torch.inference_mode()
def test_aligned_key_tail_and_compute_tiles(dtype, shape, gate_pattern):
    b, m, n, h, hk, d = shape
    case = dict(B=b, M=m, N=n, HQ=h, H=hk, D=d, dtype=dtype, scale=1.3)
    x = list(ref.make_inputs(case, 103))
    if gate_pattern == "zero":
        x[-1].zero_()
    for threshold in (0.0, -10.0, -10000.0):
        ref.assert_bitwise(ref.call_optimized(x, case, threshold), ref.call_official(x, case, threshold))


def test_dispatch_and_error_propagation(c550, monkeypatch):
    import flag_attn
    import flag_attn.forgetting_attention as api

    assert flag_attn.parallel_forgetting_attn is c550.parallel_forgetting_attn
    assert api.parallel_forgetting_attn is c550.parallel_forgetting_attn
    assert api.forgetting_attention is c550.parallel_forgetting_attn

    def fail(*args, **kwargs):
        raise RuntimeError("sentinel C550 kernel failure")

    monkeypatch.setattr(c550, "c550_forward", fail)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="sentinel C550 kernel failure"):
        api.forgetting_attention(*ref.make_inputs(ref._reuse_case()), adaptive_threshold=-10.0)


@pytest.mark.parametrize("entry", ["registry", "namespace"])
@pytest.mark.parametrize(
    "problem",
    [
        "head_first",
        "seq_start",
        "threshold_none",
        "threshold_positive",
        "threshold_nan",
        "scale_inf",
        "gate_dtype",
        "q_dtype",
        "noncontiguous",
    ],
)
@torch.inference_mode()
def test_reject_unsupported(problem, entry, monkeypatch):
    if entry == "namespace":
        from flag_attn.forgetting_attention import forgetting_attention

        monkeypatch.setattr(ref, "optimized_entry", lambda: forgetting_attention)
    ref.test_forgetting_attention_rejects_unsupported(problem)


@pytest.mark.parametrize("h", [1, 2, 3, 16, 32])
@pytest.mark.parametrize("n", [257, 1025, 4096])
@pytest.mark.parametrize("offset", [0, 1])
@torch.inference_mode()
def test_native_scan_lane_mapping(h, n, offset, c550):
    torch.manual_seed(h + n + offset)
    storage = -torch.rand(2 * n * h + offset, device="cuda", dtype=torch.float32)
    gate = storage[offset:].view(2, n, h)
    gate[:, ::5] = 0.0
    expected = gate.transpose(1, 2).cumsum(-1)
    for m, qm, bn in [(1, 1, 128), (n, 64, 32), (n // 2, 128, 32)]:
        qi = n - m + torch.arange(0, m, qm, device="cuda")
        ki = torch.arange(bn - 1, n + bn - 1, bn, device="cuda").clamp_max(n - 1)
        for threshold in [0.0, -10.0]:
            prefix, starts = c550.prepare(gate, m, qm, bn, threshold)
            assert torch.equal(prefix.contiguous().view(torch.int32), expected.contiguous().view(torch.int32))
            assert torch.equal(starts, ((expected[..., qi, None] - expected[..., None, ki]) < threshold).sum(-1) * bn)
