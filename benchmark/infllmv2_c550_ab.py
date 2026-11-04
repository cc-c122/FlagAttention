"""Compare an archived C550 adaptation with the current InfLLM-V2 implementation.

This is a whole-API, alternating A/B CUDA Event benchmark. Supply an unchanged
baseline source directory and a JSON case matrix; both sources and all eleven
samples are recorded. Unsupported baseline cases have no speedup value.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys
import time
import types

import torch


def load_operator(name, directory):
    package = types.ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    return importlib.import_module(name + ".operator")


def hashes(directory):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.glob("*.py"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--optimized-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7411)
    parser.add_argument("--reps", type=int, default=11)
    args = parser.parse_args()
    torch.set_num_threads(1)
    modules = {
        "baseline": load_operator("infllmv2_pr_baseline", args.baseline_dir.resolve()),
        "optimized": load_operator("infllmv2_pr_optimized", args.optimized_dir.resolve()),
    }
    cases = json.loads(args.cases.read_text())
    manifest = {
        "baseline": "V0: initial working C550 adaptation before performance optimization",
        "optimized": "V5: final generalized C550 optimization",
        "baseline_sha256": hashes(args.baseline_dir),
        "optimized_sha256": hashes(args.optimized_dir),
        "matrix_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "seed": args.seed,
        "reps": args.reps,
        "torch": torch.__version__,
        "device": str(torch.cuda.get_device_properties(0)),
        "physical_gpu": 1,
        "timing": "whole API; CUDA Event median; alternating A/B; no CUDA Graph",
        "started": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_name(args.output.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    rows = []
    for index, case in enumerate(cases):
        row = dict(case, seed=args.seed)
        rows.append(row)
        dtype = getattr(torch, case["dtype"])
        torch.manual_seed(args.seed)
        batch, m, n, d = (case[key] for key in ("batch", "q", "kv", "d"))
        grad = case.get("backward", False)
        q = (torch.randn(batch * m, case["hq"], d, device="cuda", dtype=dtype) * 0.2).requires_grad_(grad)
        k = (torch.randn(batch * n, case["hk"], d, device="cuda", dtype=dtype) * 0.2).requires_grad_(grad)
        v = (torch.randn_like(k) * 0.2).requires_grad_(grad)
        cq = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * m
        ck = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * n
        dout = torch.randn_like(q) if grad else None
        funcs = {}
        for key, module in modules.items():
            config = module.InfLLMV2Config(**case["config"])

            def run(module=module, config=config):
                out = module.infllmv2_attention(q, k, v, cq, ck, m, n, config=config)  # noqa: F821 -- inputs are deleted only after all timed calls
                return (out, *torch.autograd.grad(out, (q, k, v), dout)) if grad else (out,)  # noqa: F821 -- inputs are deleted only after all timed calls

            funcs[key] = run
        values = {}
        for key in ("baseline", "optimized"):
            try:
                values[key] = funcs[key]()
                torch.cuda.synchronize()
            except Exception as error:
                row[key + "_error"] = type(error).__name__ + ": " + str(error)[-2000:]
        if "optimized" not in values:
            row["status"] = "optimized_failed"
        else:
            if "baseline" in values:
                try:
                    atol, rtol = (0.008, 0.05) if dtype == torch.bfloat16 else (0.002, 0.02)
                    for actual, expected in zip(values["optimized"], values["baseline"]):
                        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                    row["max_abs_errors"] = [
                        float((a.float() - b.float()).abs().max())
                        for a, b in zip(values["optimized"], values["baseline"])
                    ]
                    row["status"] = "compared"
                except AssertionError as error:
                    row["status"] = "correctness_failed"
                    row["correctness_error"] = str(error)[-2000:]
            else:
                row["status"] = "baseline_unsupported"
            active = [key for key in funcs if key in values]
            for _ in range(3):
                for key in active:
                    funcs[key]()
            torch.cuda.synchronize()
            samples = {key: [] for key in active}
            wall = {key: [] for key in active}
            events = {
                key: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for key in active
            }
            for rep in range(args.reps):
                for key in active if rep % 2 == 0 else list(reversed(active)):
                    start, end = events[key]
                    wall_start = time.perf_counter_ns()
                    start.record()
                    funcs[key]()
                    end.record()
                    end.synchronize()
                    samples[key].append(start.elapsed_time(end))
                    wall[key].append((time.perf_counter_ns() - wall_start) / 1e6)
            row["measurements"] = {
                key: {
                    "median_ms": statistics.median(samples[key]),
                    "samples_ms": samples[key],
                    "wall_median_ms": statistics.median(wall[key]),
                    "wall_samples_ms": wall[key],
                }
                for key in active
            }
            if row["status"] == "compared":
                row["speedup"] = (
                    row["measurements"]["baseline"]["median_ms"] / row["measurements"]["optimized"]["median_ms"]
                )
        print(
            index + 1,
            len(cases),
            row["status"],
            case["label"],
            case["dtype"],
            "G",
            case["g"],
            "D",
            d,
            "Q/KV",
            m,
            n,
            "x",
            round(row.get("speedup", 0), 3),
            flush=True,
        )
        args.output.write_text(json.dumps(rows, indent=2))
        del values, funcs, q, k, v, cq, ck, dout
    assert manifest["baseline_sha256"] == hashes(args.baseline_dir)
    assert manifest["optimized_sha256"] == hashes(args.optimized_dir)
    manifest.update(
        finished=time.time(),
        cases=len(rows),
        status_counts={
            status: sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})
        },
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("AB_COMPLETE", manifest["status_counts"], flush=True)


if __name__ == "__main__":
    main()
