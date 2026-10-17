近期，我们在海光 **K100_AI** 与 **BW100** 两个平台上，对Qwen3.8进行了推理正确性、长上下文、并发吞吐和框架兼容性测试。测试覆盖BF16、W8A8两种精度，以及vLLM、SGLang两套主流推理框架。

本文完整公开本轮测试使用的硬件、容器镜像、模型权重、服务启动参数、正确性请求、性能测试方法、全部结果、异常现象和复现要点。所有数据均来自2026年9月19日的真实运行记录。

## 1.镜像使用总表

| 芯片    | 模型                          | 精度 | 框架              | 测试镜像                                                     |
| ------- | ----------------------------- | ---- | ----------------- | ------------------------------------------------------------ |
| BW100   | Qwen3.8-27B                   | BF16 | vLLM 0.21.0       | `10.16.1.152:5000/jenkins/model_test_env/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10-20260901-0035` |
| BW100   | Qwen3.8-27B                   | BF16 | SGLang 0.5.12     | `harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10` |
| BW100   | Qwen3.8-27B-Channel-INT8-w8a8 | W8A8 | vLLM 0.18.1       | `harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260810-qwen3.8` |
| BW100   | Qwen3.8-27B-Channel-INT8-w8a8 | W8A8 | SGLang 0.5.12     | `harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10` |
| K100_AI | Qwen3.8-27B                   | BF16 | SGLang 0.5.12     | `harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10` |
| K100_AI | Qwen3.8-27B-Channel-INT8-w8a8 | W8A8 | vLLM 0.18.1       | `harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260810-qwen3.8` |
| K100_AI  | Qwen3.8-27B                   | BF16 | vLLM 0.18.1       | `harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260810-qwen3.8` |
| K100_AI | Qwen3.8-27B-Channel-INT8-w8a8 | W8A8 | SGLang 专项运行时 | `harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-K100AI-qwen3.8-0828` |

## 2. 通用容器启动方式

下面是测试使用的 DCU 容器参数。模型路径、结果目录和容器名应按本机环境修改。

```bash
docker run -d \
  --name qwen_test \
  --network host \
  --ipc host \
  --shm-size 128g \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mkfd \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -u root \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /path/to/model:/model:ro \
  -v /path/to/bench:/bench \
  --entrypoint /bin/bash \
  IMAGE_NAME \
  -lc 'sleep infinity'
```

容器启动参考：

```bash
docker run -it \
  --shm-size 200g \
  --network host \
  --name qwen38_k100_sglang \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mkfd \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -u root \
  -v /opt/hyhal:/opt/hyhal:ro \
  --entrypoint /bin/bash \
  harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10
```

## 3. 模型服务启动方式
### 3.1 bw100
### 3.1.1 SGLang 0.5.12

```bash
export HIP_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export HSA_FORCE_FINE_GRAIN_PCIE=1
export SGLANG_ROCM_USE_AITER_MOE=false
export SGLANG_KV_LAYOUT_HCU_FA=false

python3 -u -m sglang.launch_server \
  --model-path /model \
  --served-model-name qwen38-bf16 \
  --host 0.0.0.0 \
  --port 18580 \
  --nccl-port 19580 \
  --tp-size 2 \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --context-length 131072 \
  --mem-fraction-static 0.85 \
  --max-running-requests 128 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 16384 \
  --disable-radix-cache \
  --attention-backend triton \
  --page-size 64 \
  --mamba-scheduler-strategy no_buffer \
  --cuda-graph-max-bs 128 \
  --disable-custom-all-reduce \
  --enable-metrics
```

### 3.1.2 vLLM 0.21.0

```bash
export HIP_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
export HSA_FORCE_FINE_GRAIN_PCIE=1
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=False

python3 -u -m vllm.entrypoints.openai.api_server \
  --model /model \
  --served-model-name qwen38-vllm021-bf16 \
  --host 0.0.0.0 \
  --port 18680 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,12,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128]}'
```
  ### 3.2 K100AI
  ### 3.2.1 vLLM 0.18.0
```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=False  #K100AI推理需添加该参数，BW1000可忽略
vllm serve hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name Qwen3.8-27B-Channel-INT8-w8a8 \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 33792 \
  --gpu-memory-utilization 0.95

```
  ### 3.2.2 SGLang 0.5.12
```bash
#资料来源https://developer.sourcefind.cn/modelzoo/list/qwen3.8-int/detail?post_id=9f58e9d5-ab52-11f1-9283-0242ac150003&active=ModelOverview
model_path="/workspace/work/Qwen3.8-27B-Channel-INT8-w8a8"  #根据自己的权重路径替换
bash start_sglang.sh "$model_path" 
```
## 4 模型性能测试

### 4.1 固定长度 4K/64K 请求

固定长度客户端使用随机 token 构造输入，`ignore_eos=true` 固定输出 1024 tokens；2 轮预热、3 轮正式测试，每轮请求数为 `max(8, 2×并发)`。

```bash
python3 -u benchmark.py \
  --base-url http://127.0.0.1:18580 \
  --model qwen38-bf16 \
  --output-dir results/4k-1k-c32 \
  --input-len 4096 \
  --output-len 1024 \
  --concurrency 32 \
  --warmup-rounds 2 \
  --measure-rounds 3 \
  --requests-per-round 64 \
  --timeout 3600
```

测试档位：

```text
4K/1K：C1、C8、C16、C32、C64、C128
64K/1K：C1、C8、C16、C32
```

记录指标包括输出吞吐、TTFT、TPOT、端到端延迟、成功率、请求级分位数、KV Cache、队列和容量中止原因。

### 4.2 SGLang ShareGPT 请求

```bash
python3 -u -m sglang.bench_serving \
  --backend sglang-oai \
  --base-url http://127.0.0.1:18580 \
  --dataset-name sharegpt \
  --dataset-path /dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
  --model /model \
  --served-model-name qwen38-bf16 \
  --sharegpt-context-len 131072 \
  --num-prompts 1000 \
  --max-concurrency 32 \
  --request-rate inf \
  --warmup-requests 64 \
  --seed 20260912 \
  --disable-tqdm \
  --output-details \
  --output-file /bench/results/sharegpt-c32/round-01.jsonl
```

### 4.3 vLLM ShareGPT 请求

```bash
vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:18680 \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path /dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
  --model qwen38-vllm021-bf16 \
  --tokenizer /model \
  --num-prompts 1000 \
  --max-concurrency 32 \
  --request-rate inf \
  --num-warmups 64 \
  --seed 20260912 \
  --temperature 0 \
  --disable-tqdm \
  --save-result \
  --save-detailed \
  --result-dir /bench/results/sharegpt-c32 \
  --result-filename round-01.json
```

ShareGPT 并发档位为 C1、C4、C16、C32、C64；每档 1000 个正式请求、`2×并发` 预热、3 轮正式测试。

## 5. 正式基线结果
具体性能指标以下面链接为准：
https://kcnm6g5dkw5p.feishu.cn/wiki/LR4LwUrK4iX0alkZdF1cENCAnBf
### 5.1 固定长度输出吞吐

下表只列输出吞吐，单位 tok/s；`ABORTED` 表示容量保护或阶段超时中止。

| 芯片/框架               |  4K C1 |   4K C8 |  4K C16 |  4K C32 |  4K C64 | 4K C128 | 64K C1 | 64K C8 | 64K C16 | 64K C32 |
| ----------------------- | -----: | ------: | ------: | ------: | ------: | ------: | -----: | -----: | ------: | ------: |
| BW100 / SGLang 0.5.12   | 21.195 | 182.734 | 304.034 | 433.847 | 545.911 | 577.456 | 13.616 | 39.128 |  39.199 |  39.201 |
| BW100 / vLLM 0.21.0     | 20.627 | 108.770 | 212.331 | 356.172 | 516.791 | 503.767 | 10.540 | 19.983 |  21.071 |  21.203 |
| K100_AI / SGLang 0.5.12 |  9.055 |  52.480 |  74.011 | 142.864 | 194.387 | 181.599 |  7.520 | 13.929 |  14.242 | ABORTED |
| K100_AI / vLLM 0.21.0   |  7.206 |  46.832 |  78.258 | 156.880 | 190.354 | ABORTED |  6.778 | 15.439 | ABORTED | ABORTED |

SLA 条件为：4K 场景 TTFT p95 < 3 秒且 TPOT p95 < 50 ms；64K 场景 TTFT p95 < 15 秒且 TPOT p95 < 50 ms。

| 芯片/框架               | 4K SLA 最大并发 | 64K SLA 最大并发 |
| ----------------------- | --------------: | ---------------: |
| BW100 / SGLang 0.5.12   |               1 |                0 |
| BW100 / vLLM 0.21.0     |               1 |                0 |
| K100_AI / SGLang 0.5.12 |               0 |                0 |
| K100_AI / vLLM 0.21.0   |               0 |                0 |

### 5.2 ShareGPT 输出吞吐

单位 tok/s，表中均为 3 轮聚合值。

| 芯片/框架               |     C1 |      C4 |     C16 |     C32 |     C64 |
| ----------------------- | -----: | ------: | ------: | ------: | ------: |
| BW100 / SGLang 0.5.12   | 21.454 | 106.207 | 307.450 | 435.996 | 528.158 |
| BW100 / vLLM 0.21.0     | 20.775 |  70.798 | 225.385 | 377.422 | 487.524 |
| K100_AI / SGLang 0.5.12 | 12.150 |  20.260 | 104.905 | 168.419 | 242.089 |
| K100_AI / vLLM 0.21.0   |  9.348 |  38.836 | 148.070 | 211.541 | 263.272 |

## 6. 参考链接
- https://developer.sourcefind.cn/modelzoo/list/qwen3.8/detail?post_id=7dcd4d4c-9a1f-11f1-a0b1-0242ac150003
- https://developer.sourcefind.cn/modelzoo/list/qwen3.8-int/detail?post_id=9f58e9d5-ab52-11f1-9283-0242ac150003&active=ModelOverview
- https://github.com/DocPang/qwen38-k100ai-int8-optimization
## 7. 参考脚本
### bencmark.py
```bash
#!/usr/bin/env python3
"""Fixed-token OpenAI completions benchmark with incremental progress output."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import signal
import threading
import time

import requests


_thread_local = threading.local()
_stop = threading.Event()


def handle_stop(signum, frame):
    _stop.set()


def session():
    value = getattr(_thread_local, "session", None)
    if value is None:
        value = requests.Session()
        value.trust_env = False
        _thread_local.session = value
    return value


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct / 100.0
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def distribution(values):
    values = [float(item) for item in values if item is not None]
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None,
                "p50": None, "p90": None, "p95": None, "p99": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def atomic_json(path, value):
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def make_prompt(length, seed, request_index):
    rng = random.Random(seed + request_index * 1000003)
    prefix = [1000 + ((request_index >> shift) & 0xFFFF)
              for shift in (0, 8, 16, 24)]
    prompt = prefix[:length]
    while len(prompt) < length:
        prompt.append(rng.randint(1000, 220000))
    return prompt


def make_job(args, phase, round_index, request_index):
    global_index = round_index * 1000000 + request_index
    seed = args.seed + (100000000 if phase == "warmup" else 0)
    return {
        "request_index": request_index,
        "global_index": global_index,
        "body": {
            "model": args.model,
            "prompt": make_prompt(args.input_len, seed, global_index),
            "max_tokens": args.output_len,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,
            "seed": 0,
            "cache_salt": "%s-%d-%d-%d" %
                          (phase, round_index, args.input_len, request_index),
        },
    }


def run_request(url, job, timeout):
    started = time.monotonic()
    first_token_at = None
    output_parts = []
    usage = {}
    event_count = 0
    error = None
    status_code = None
    done = False
    try:
        response = session().post(
            url, json=job["body"], stream=True, timeout=(30, timeout)
        )
        status_code = response.status_code
        if status_code != 200:
            error = "HTTP %s: %s" % (status_code, response.text[:2000])
        else:
            for raw_line in response.iter_lines(chunk_size=1, decode_unicode=True):
                if _stop.is_set():
                    error = "terminated by capacity/timeout guard"
                    response.close()
                    break
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done = True
                    break
                try:
                    event = json.loads(payload)
                except Exception:
                    continue
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if choices:
                    choice = choices[0]
                    if first_token_at is None and choice.get("finish_reason") is None:
                        first_token_at = time.monotonic()
                    piece = choice.get("text") or ""
                    if piece:
                        output_parts.append(piece)
                    event_count += 1
    except Exception as exc:
        error = "%s: %s" % (type(exc).__name__, exc)
    ended = time.monotonic()
    text = "".join(output_parts)
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    e2e = ended - started
    ttft = first_token_at - started if first_token_at is not None else None
    tpot_ms = None
    if ttft is not None and completion_tokens and completion_tokens > 1 and e2e > ttft:
        tpot_ms = 1000.0 * (e2e - ttft) / float(completion_tokens - 1)
    return {
        "request_index": job["request_index"],
        "status_code": status_code,
        "done": done,
        "error": error,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens"),
        "e2e_s": e2e,
        "ttft_s": ttft,
        "tpot_ms": tpot_ms,
        "stream_events": event_count,
        "output_chars": len(text),
        "replacement_chars": text.count("\ufffd"),
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "output_sample": text[:240],
    }


def run_round(args, phase, round_index, request_count, round_dir):
    os.makedirs(round_dir, exist_ok=True)
    progress_path = os.path.join(args.output_dir, "progress.json")
    request_path = os.path.join(round_dir, "requests.jsonl")
    jobs = [make_job(args, phase, round_index, index)
            for index in range(request_count)]
    url = args.base_url.rstrip("/") + "/v1/completions"
    started_mono = time.monotonic()
    started_epoch = time.time()
    results = []
    with open(request_path, "w", encoding="utf-8", buffering=1) as output:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency) as pool:
            futures = [pool.submit(run_request, url, job, args.timeout)
                       for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                output.write(json.dumps(result, ensure_ascii=False,
                                        sort_keys=True) + "\n")
                atomic_json(progress_path, {
                    "status": "RUNNING" if not _stop.is_set() else "TERMINATING",
                    "phase": phase,
                    "round": round_index + 1,
                    "completed": len(results),
                    "request_count": request_count,
                    "updated_at": time.time(),
                    "started_at": started_epoch,
                })
                if _stop.is_set():
                    for pending in futures:
                        pending.cancel()
                    break
    wall = time.monotonic() - started_mono
    results.sort(key=lambda item: item["request_index"])
    valid = [item for item in results if item["error"] is None and item["done"]]
    exact = [item for item in valid
             if item["prompt_tokens"] == args.input_len and
             item["completion_tokens"] == args.output_len]
    sum_prompt = sum(item.get("prompt_tokens") or 0 for item in valid)
    sum_completion = sum(item.get("completion_tokens") or 0 for item in valid)
    summary = {
        "status": "ABORTED" if _stop.is_set() else
                  ("PASS" if len(exact) == request_count else "FAIL"),
        "phase": phase,
        "round": round_index + 1,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "concurrency": args.concurrency,
        "requested": request_count,
        "successful": len(valid),
        "exact_length": len(exact),
        "wall_time_s": wall,
        "request_throughput_rps": len(valid) / wall if wall else None,
        "input_throughput_tps": sum_prompt / wall if wall else None,
        "output_throughput_tps": sum_completion / wall if wall else None,
        "total_throughput_tps": (sum_prompt + sum_completion) / wall if wall else None,
        "e2e_s": distribution([item["e2e_s"] for item in valid]),
        "ttft_s": distribution([item["ttft_s"] for item in valid]),
        "tpot_ms": distribution([item["tpot_ms"] for item in valid]),
        "replacement_chars": sum(item["replacement_chars"] for item in valid),
    }
    atomic_json(os.path.join(round_dir, "summary.json"), summary)
    return summary


def aggregate(args, measured):
    valid = [item for item in measured if item["status"] == "PASS"]
    result = {
        "status": "PASS" if len(valid) == args.measure_rounds else
                  ("ABORTED" if _stop.is_set() else "FAIL"),
        "base_url": args.base_url,
        "model": args.model,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "concurrency": args.concurrency,
        "warmup_rounds": args.warmup_rounds,
        "measure_rounds": args.measure_rounds,
        "requests_per_round": args.requests_per_round,
        "successful_rounds": len(valid),
        "output_throughput_tps": distribution(
            [item["output_throughput_tps"] for item in valid]),
        "ttft_p50_s": distribution([item["ttft_s"]["p50"] for item in valid]),
        "ttft_p95_s": distribution([item["ttft_s"]["p95"] for item in valid]),
        "tpot_p50_ms": distribution([item["tpot_ms"]["p50"] for item in valid]),
        "tpot_p95_ms": distribution([item["tpot_ms"]["p95"] for item in valid]),
        "rounds": measured,
        "method": "fixed random token IDs; two warmup waves; three measured rounds",
    }
    atomic_json(os.path.join(args.output_dir, "summary.json"), result)
    atomic_json(os.path.join(args.output_dir, "progress.json"), {
        "status": result["status"], "updated_at": time.time()
    })
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--measure-rounds", type=int, default=3)
    parser.add_argument("--requests-per-round", type=int)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    args.requests_per_round = args.requests_per_round or max(8, 2 * args.concurrency)
    os.makedirs(args.output_dir, exist_ok=True)
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    atomic_json(os.path.join(args.output_dir, "config.json"), vars(args))

    for round_index in range(args.warmup_rounds):
        summary = run_round(
            args, "warmup", round_index, args.concurrency,
            os.path.join(args.output_dir, "warmup-%02d" % (round_index + 1)))
        if summary["status"] != "PASS":
            aggregate(args, [])
            return 2

    measured = []
    for round_index in range(args.measure_rounds):
        summary = run_round(
            args, "measure", round_index, args.requests_per_round,
            os.path.join(args.output_dir, "round-%02d" % (round_index + 1)))
        measured.append(summary)
        if summary["status"] != "PASS":
            break
    result = aggregate(args, measured)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
```