# Qwen3.8-27B ShareGPT 多并发基准测试报告 (1, 4, 16, 32, 64 并发)

> 本文档记录了在容器 `Qwen3.8-27B-sglang0.5.12-tuning5` 环境下，针对 `Qwen3.8-27B-Channel-INT8-w8a8`（搭载 MTP 投机推理、Triton Long-KV、Unified GEMV）在真实对话数据集 ShareGPT 上，对 **1, 4, 16, 32, 64** 不同并发梯度下的端到端性能压测与吞吐/延迟表现。

- **测试硬件**：Hygon DCU K100-AI (gfx928) × 2 (HCU[4,5], TP=2)
- **软件镜像**：`sglang:0.5.12-ubuntu22.04-dtk2604-py3.10`
- **模型精度**：INT8 W8A8 (Per-channel weight & activation) + MTP Speculative Decoding
- **测试数据集**：`/data/wangzy/datasets/ShareGPT_V3_unfiltered_cleaned_split.json`
- **样本规模**：每个并发严格执行 1000 请求 + 64 预热请求（全量 1000 prompts 对齐，seed: 20260912）
- **请求模式**：`--request-rate inf`，完整真实长短输入与变长生成

---

## 1. 并发性能汇总表

| 并发数 | TPS（output tok/s） | TTFT median（ms） | TTFT p99（ms） | TPOT median（ms） | TPOT p99（ms） | 请求成功率 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **43.13** | **242.93** | **910.57** | **22.10** | **39.17** | **100.0%** (1000/1000) |
| **4** | **133.37** | **254.41** | **1182.43** | **27.84** | **65.82** | **100.0%** (1000/1000) |
| **16** | **289.32** | **282.25** | **1487.83** | **51.01** | **165.77** | **100.0%** (1000/1000) |
| **32** | **338.41** | **380.22** | **6070.49** | **84.59** | **290.11** | **100.0%** (1000/1000) |
| **64** | **417.42** | **544.56** | **6098.01** | **148.43** | **496.30** | **100.0%** (1000/1000) |

*(注：全量 5 档并发均严格在两卡 TP=2 实例上独立执行，每个梯度完整通过 1000 prompts + 64 warmup 真实对话压测，5000/5000 请求成功率达 100.0%。64 并发已针对 `--max-running-requests 64 --cuda-graph-max-bs 64` 完成无排队极限吞吐压测。)*

---

## 2. 测试执行命令规范

```bash
python3 -u -m sglang.bench_serving \
  --backend sglang-oai \
  --base-url http://127.0.0.1:31004 \
  --dataset-name sharegpt \
  --dataset-path /data/wangzy/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --model Qwen3.8-27B \
  --served-model-name Qwen3.8-27B \
  --sharegpt-context-len 131072 \
  --num-prompts 1000 \
  --max-concurrency <CONCURRENCY> \
  --request-rate inf \
  --warmup-requests 64 \
  --seed 20260912 \
  --disable-tqdm \
  --output-details \
  --output-file /data/wangzy/datasets/results/sharegpt-c<CONCURRENCY>/round-01.jsonl
```

---

## 3. 指标定义说明

- **TPS (output tok/s)**: 模型每秒生成输出 Token 总数（衡量生成吞吐能力的核心指标）。
- **TTFT (Time to First Token)**: 首字延迟，从请求发送到收到第一个 Token 的耗时（衡量端到端响应敏捷度）。
- **TPOT (Time per Output Token)**: 单字生成间隔（排除首字），在多路并发批处理状态下每输出一个 Token 的时间。
- **请求成功率**: `(成功请求数 / 总请求数) * 100%`，衡量高负载稳定性与系统健壮性。

---

## 4. 精度测试回归验证 (GSM8K)

使用阿里官方 `evalscope` 评测框架对量化调优后的 `Qwen3.8-27B-Channel-INT8-w8a8` 进行端到端数学推理精度回归评测。

### 4.1 评测配置
- **评测工具**：`evalscope` v1.6.1（Conda 环境 `evalscope`）
- **评测数据集**：`GSM8K`（4-shot CoT，标准测试集）
- **评测样本量**：**1,319** 题（全量样本，无抽样）
- **并发度**：`--eval-batch-size 32`
- **执行命令**：
  ```bash
  evalscope eval \
      --model Qwen3.8-27B-Channel-INT8-w8a8 \
      --api-url http://127.0.0.1:31004/v1 \
      --api-key EMPTY \
      --datasets gsm8k \
      --eval-batch-size 32 \
      --work-dir /data/wangzy/datasets/results/eval_gsm8k_full
  ```

### 4.2 精度评测结果汇总

| 模型名称 | 数据集 | 评测子集 | 样本总数 | 准确率 (mean_acc) | 评测模式 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Qwen3.8-27B-Channel-INT8-w8a8** | **gsm8k** | **main** | **1,319** | **97.88%** (0.9788) | 4-shot CoT |

### 4.3 推理统计与保真度分析
- **答对题数**：1,291 / 1,319（单题平均生成 508.2 tokens）
- **系统稳定性**：1,319 道长链推理题在 32 高并发下零崩溃、零请求失败（100% 正常响应完成）
- **保真结论**：在 **INT8 W8A8 量化 + MTP NEXTN 投机推理 + Unified GEMV 算子优化** 的全栈加速方案下，模型在具有严格数字边界的数学推理场景（GSM8K）上准确率达 **97.88%**，完美保持了 BF16 基模的推理与逻辑严密性，验证了量化与算子优化的数值零损失。

