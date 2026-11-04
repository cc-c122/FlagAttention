# InfLLM-V2：C550 优化与性能对比

保留仓库已注册的 `infllmv2_attention` mark、公共 prefill/decode API 和算子清单。
实现位于 `src/flag_attn/infllmv2`，通过 C550 分派使用对应 kernel。

## 版本与读表说明

**适配原版 = V0**：最初已完成 C550 正确性适配的版本；**最终版 = V5**：完成后续优化及 shape 泛化的版本。两列耗时单位均为毫秒（ms），**加速比 = 适配原版耗时 ÷ 最终版耗时**，每行比较相同的输入配置。Query 头数、KV 头数和每头维度分别对应 Hq、Hkv 和 D；Q、KV 长度均指每条序列的 token 数。

V0 是 2026-11-02 留存的初始 C550 适配快照，保存在 [baseline_v0](baseline_v0/)。
6 个文件的 SHA256 与原始归档和第一轮优化的基线清单一致。V0/V5 是说明演进使用的编号，不是 Git tag。

## 测量方法与总体结果

在同一张物理 C550 GPU1 上直接做 V0/V5 A/B：相同输入、seed=7411、预热 3 次，
交错测量 11 轮，取完整算子 API 的 CUDA Event 中位数。计时没有使用 CUDA Graph 或 profiler 插桩。
源码哈希、软件版本和时间戳见 [measurement_manifest.json](measurement_manifest.json)。
所有加速比均来自这次直接比较。

230 组前向均可直接比较，等权几何平均加速 **6.92×**，230 组加速比均超过 1.05。另测 18 组前向加反向，其中 6 组可比较，12 组适配原版无法运行；这 12 组单列最终版耗时，不计算加速比。

## 代表性性能表

以下按场景和精度分表，优化前后耗时分别列出。

### 8K 完整预填充（prefill）

共同条件：Batch=1，Q 长度=8192，KV 长度=8192；Block=64、TopK=64、初始块=1、局部块=32。

**BF16**

| Query 头数 | KV 头数 | 每头维度 | 适配原版（ms） | 最终版（ms） | 加速比 |
|---:|---:|---:|---:|---:|---:|
| 4 | 4 | 64 | 20.014 | 5.286 | 3.79× |
| 16 | 4 | 96 | 371.886 | 11.391 | 32.65× |
| 16 | 4 | 128 | 371.011 | 11.424 | 32.48× |
| 16 | 4 | 256 | 1262.532 | 37.698 | 33.49× |
| 32 | 2 | 64 | 11.988 | 4.720 | 2.54× |
| 32 | 2 | 128 | 185.381 | 9.193 | 20.16× |

**FP16**

| Query 头数 | KV 头数 | 每头维度 | 适配原版（ms） | 最终版（ms） | 加速比 |
|---:|---:|---:|---:|---:|---:|
| 4 | 4 | 64 | 19.896 | 4.671 | 4.26× |
| 16 | 4 | 96 | 371.907 | 11.068 | 33.60× |
| 16 | 4 | 128 | 367.172 | 11.060 | 33.20× |
| 16 | 4 | 256 | 1244.920 | 37.003 | 33.64× |
| 32 | 2 | 64 | 11.566 | 4.190 | 2.76× |
| 32 | 2 | 128 | 183.564 | 8.872 | 20.69× |

### 1K Dense 注意力

共同条件：Batch=1，Q 长度=1024，KV 长度=1024，Query 头数=64，KV 头数=2，每头维度=256。

Dense 分支不使用 Block、TopK 等选块参数。

**BF16**

| 适配原版（ms） | 最终版（ms） | 加速比 |
|---:|---:|---:|
| 18.760 | 1.384 | 13.56× |

**FP16**

| 适配原版（ms） | 最终版（ms） | 加速比 |
|---:|---:|---:|
| 19.161 | 1.342 | 14.28× |

### Chunk 注意力：128 条 query / 8K KV

共同条件：Batch=1，Q 长度=128，KV 长度=8192，Query 头数=16，KV 头数=4，每头维度=96。

Block=64、TopK=64、初始块=1、局部块=32。

**BF16**

| 适配原版（ms） | 最终版（ms） | 加速比 |
|---:|---:|---:|
| 9.649 | 0.878 | 10.99× |

**FP16**

| 适配原版（ms） | 最终版（ms） | 加速比 |
|---:|---:|---:|
| 9.739 | 0.875 | 11.13× |

完整 248 组结果见 [performance.md](performance.md)，每组原始样本见
[comparison_results.json](comparison_results.json)。完整表按精度、计算阶段和场景分组，
将共同参数放在表格上方，只在列中展示变化的参数。

覆盖范围包括 BF16/FP16、GQA1–64（含非二次幂分组）、D32–256、Batch 1/2/4，
以及不同序列边界、Block、TopK 和窗口。这些是已测组合，不是所有参数的笛卡尔积。

## 主要实现

- 为 C550 的共享内存预算调整 dense/sparse MMA 布局和启动配置，解决原 CUDA tile 超过 64 KiB 的问题。
- 融合 K 压缩准备与池化/选块，保留确定的块排序，并减少设备长度元数据的主机回读。
- 在 query 和 GQA 头之间复用选中的局部 KV；对补齐的头和维度加掩码，按资源用量选择 tile。
  显式 Selected 输入继续使用通用路径，保留重复 ID 的语义。
- 拆分较宽的反向头分组，使用 FP32 累加 dK/dV，保留 suffix causal、packed 变长输入、自定义 scale 和 Q/K/V 梯度。

## 验证与复现

沿用 `infllmv2_attention` 并启用 `--strict-markers` 的集成验证为 **151 passed、4 skipped**，
包含原算子测试、C550 回归测试和 44 组独立泛化 reference 测试。4 个跳过项为 NVIDIA/Hopper 专用测试。
另有 7 个现有 benchmark 用例通过，涵盖 dense、8K prefill/decode 和 forward+backward。
日志见 [validation.json](validation.json)。

```bash
pytest tests/flag_attn/test_infllmv2_attention.py tests/flag_attn/test_infllmv2_c550.py tests/flag_attn/test_infllmv2_c550_generalization.py -m infllmv2_attention --strict-markers -q
pytest benchmark/test_infllmv2_attention_benchmark.py -m infllmv2_attention --strict-markers -q
python benchmark/infllmv2_c550_ab.py --baseline-dir docs/infllmv2_c550/baseline_v0 --optimized-dir src/flag_attn/infllmv2 --cases docs/infllmv2_c550/comparison_cases.json --output results/infllmv2_v0_v5_repeat.json
```

在 C550/MACA 环境的仓库根目录执行，并设置 `PYTHONPATH=src`。
第二条命令包含比已选 7 个集成 benchmark 更长的序列；A/B 性能表由第三条命令对应的脚本测量。

## 保留问题

大维度反向仍有寄存器 spill；FP32 原子累加不保证逐位确定性。
两路注意力需要 FP32 partial 缓冲，部分场景虽然指令减少，显存流量仍会增加。
短 query、显式 Selected 和不满足专用路径条件的配置使用通用回退，性能结论限于已测 shape 和类型。
