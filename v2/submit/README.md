# Qwen3.8-27B：K100-AI／SGLang／TP2 提交与复现

本目录是提交入口，按“**环境准备 → 依赖安装 → 模型启动 → Benchmark → 得到结果**”执行。复现对象为已测 **INT8 tile＋BF16 native AllReduce 候选**；优化代码、源码、分项 Diff、配置及已测二进制见本目录。整理日期：2026-10-17，北京时间。

已完成随机性能矩阵及服务64／128配置复测，主表展示八档、SLA边界诊断、ShareGPT复核和干净容器复跑；GSM8K本地全量协议为 **1288/1319＝97.6497%**。64K SLA存在轮间波动，部分随机4K输出含U+FFFD标记；另外三项精度和最终资产准入仍待完成。[当前材料状态](docs/提交材料状态-20261017.md)列出这些限制。

| 入口 | 内容 |
|---|---|
| [补丁与完整源码](patch/README.md) | 修改项、源码、分项Diff、上游依赖、编译与回退 |
| [当前性能报告 §4](docs/性能优化报告.md#4-性能结果) | 预期结果、全部三正式轮、原始数据与均值 |
| [本轮性能与SLA报告](docs/本轮性能与SLA结果-20261017.md) | 八档主展示TPS／TTFT／TPOT、逐轮P95边界与文本标记 |
| [GSM8K报告](docs/GSM8K精度验证-20261017.md) | 完整1319题、原始响应、独立重评分及历史对照 |
| [优化与复现说明](docs/优化与复现说明.md) | 注入方式、运行条件、源码身份、复现证据 |
| [Demo说明](docs/Demo说明.md)／[视频](docs/demo/demo.mp4) | 启动与访问；基于真实API时间流的终端回放视频 |

## 1. 环境准备

以下命令在K100-AI Linux宿主执行，以 `/data/wangzy/v2/submit` 为例；按顺序在同一终端执行。评审入口使用**物理卡0、1，TP=2**，容器名 `qwen38-v2-review01`。

### 1.1 镜像与宿主

光合官方镜像自带SGLang 0.5.12+das185、Torch 2.10、DTK 26.04与Python 3.10，无需另装推理框架。按指定地址拉取：

```bash
export REVIEW_IMAGE='harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10'
export REVIEW_IMAGE_ID='sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad'
docker pull "$REVIEW_IMAGE"
test "$(docker image inspect "$REVIEW_IMAGE" --format '{{.Id}}')" = "$REVIEW_IMAGE_ID"
```

本机成绩来自同版本 `hcu/admin/base/sglang` 镜像，固定image ID如上、RepoDigest为 `sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68`。2026-10-17复核 `dcu` 地址时服务器DNS解析失败，尚未证实两个命名空间的镜像身份相同；上面identity检查必须通过才对应本包成绩。若无法拉取，可载入已交付镜像 `docker load -i /已交付镜像.tar`，或拉取实测的固定引用：

```bash
# 仅在上面的拉取失败时使用；成功后仍检查同一image ID。
export REVIEW_IMAGE='harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68'
docker pull "$REVIEW_IMAGE"
test "$(docker image inspect "$REVIEW_IMAGE" --format '{{.Id}}')" = "$REVIEW_IMAGE_ID"
```

宿主实测驱动为 **hycu 6.3.30-V1.4.1a**，`/opt/hyhal` 存在；需要Bash、Python 3.9以上、Docker、`flock`、`/dev/kfd`、`/dev/mkfd`、`/dev/dri`。完整版本和来源见 [env_declare.yaml](env_declare.yaml)。驱动版本可读取 `/sys/module/hycu/version`。

### 1.2 路径与权重

```bash
cd /data/wangzy/v2/submit
export SUBMIT_ROOT="$PWD"
export REVIEW_ID="review01-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export MODEL_PATH=/data/model/Qwen3.8-27B-Channel-INT8-w8a8
export DATASET_PATH=/data/wangzy/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
export LIBRARY_DIR="$SUBMIT_ROOT/config/library_gpu4_int8tuned_stock_clean"
export PORT=31014
export NCCL_PORT=31015
export LOCK_FILE=/data/wangzy/submit1009/result/gpu01-owner.lock
export PATCH_DIR="$SUBMIT_ROOT/result/runs/${REVIEW_ID}-runtime/patch"
```

| 资产 | 准备与核验 |
|---|---|
| 光合官方发布权重 `Qwen3.8-27B-Channel-INT8-w8a8` | 本机模型README标识为 `hygon/Qwen3.8-27B-Channel-INT8-w8a8`；完整模型／tokenizer／config共31文件、30,902,757,820字节，保持官方INT8权重。用下节资产检查与归档SHA逐项核对 |
| ShareGPT | 准备 `ShareGPT_V3_unfiltered_cleaned_split.json`；SHA256为 `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4` |
| rocBLAS完整库 | 162文件，身份见 `config/library-sha256.json`。服务器版已包含；本地小文件归档不含约790MiB完整库，需交付同一清单对应的完整目录 |

模型与数据可调整上述路径。若库在包外，直接设置 `LIBRARY_DIR=/实际完整库目录`；也可在包内目标不存在时执行 `python3 scripts/prepare_assets.py --library-source /实际完整库目录`。

### 1.3 启动前检查与卡0、1私有补丁

```bash
command -v docker python3 flock
cat /sys/module/hycu/version
test -d /opt/hyhal && test -e /dev/kfd && test -e /dev/mkfd && test -d /dev/dri
/opt/dtk/bin/rocm-smi -d 0 1 --showuse --showmeminfo vram
# 确认0、1没有其他计算任务；下面检查协作锁及两个端口。
mkdir -p "$(dirname "$LOCK_FILE")"
touch "$LOCK_FILE"
flock -n "$LOCK_FILE" true
python3 -I - "$PORT" "$NCCL_PORT" <<'CHECK_PORTS'
import socket,sys
for port in map(int,sys.argv[1:]):
    with socket.socket() as s:
        s.bind(('0.0.0.0',port))
print('PORTS_FREE')
CHECK_PORTS
bash scripts/run_review01.sh check
python3 -I scripts/verify_patch_bundle.py
python3 -I scripts/verify_package.py --runtime --library "$LIBRARY_DIR"
mkdir -p "result/runs/${REVIEW_ID}-assets"
python3 -I scripts/check_assets.py \
  --model-path "$MODEL_PATH" --dataset-path "$DATASET_PATH" \
  --output "result/runs/${REVIEW_ID}-assets/assets.json"
python3 -I scripts/prepare_review01.py --output "$PATCH_DIR"
```

预期依次包含 `PACKAGE_INTEGRITY_PASS`、补丁检查 `PASS`、`LIBRARY_SHA256_PASS`、`EXTERNAL_ASSET_IDENTITY_PASS`、`PRIVATE_REVIEW01_PATCH_PREPARED`。资产检查顺序读取约30.9GB模型，应在测量前完成。锁的前置检查完成后即释放，服务进程启动时再次独占持锁至退出。

原已测 `patch/native_bf16_ar_guard.py` 的物理设备断言固定6、7；准备工具只在本次运行私有目录中把该断言改为0、1，保存Diff、原／新SHA和完整运行副本。原始候选源码与二进制保持归档身份，具体见 [补丁说明](patch/README.md)。

### 1.4 创建容器（物理卡0、1，host网络）

先准备上述路径和私有补丁，再执行以下完整命令。同名容器应不存在；失败记录保留，重试使用新的 `REVIEW_ID`。

```bash
mounts=()
for config in "$SUBMIT_ROOT"/config/aiter/*/*.json; do
  stem=$(basename "$(dirname "$config")")
  mounts+=(-v "$config:/usr/local/lib/python3.10/dist-packages/aiter/ops/triton/configs/$stem/$(basename "$config"):ro")
done

docker run -d --name qwen38-v2-review01 --label qwen38.v2.review01=20261017 \
  --network host --ipc host \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -e HIP_VISIBLE_DEVICES=0,1 -e GPU_PAIR=0,1 \
  -e PORT="$PORT" -e NCCL_PORT="$NCCL_PORT" \
  -e MODEL_PATH="$MODEL_PATH" -e DATASET_PATH="$DATASET_PATH" \
  -e PATCH_DIR="$PATCH_DIR" -e VARIANT=candidate \
  -e RUN_ID="${REVIEW_ID}-server" -e MAX_WAIT=1800 -e MAX_GATE_WAIT=5 \
  -e TVM_FFI_CACHE_DIR=/run/qwen38-tvm-ffi \
  -v "$SUBMIT_ROOT:$SUBMIT_ROOT" \
  -v "$MODEL_PATH:$MODEL_PATH:ro" -v "$DATASET_PATH:$DATASET_PATH:ro" \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$LIBRARY_DIR:/run/qwen38-library:ro" \
  -v "$LOCK_FILE:/run/qwen38-gpu-pair.lock" \
  -v "$SUBMIT_ROOT/patch/runtime/tvm-ffi:/run/qwen38-tvm-ffi:ro" \
  -v "$SUBMIT_ROOT/patch/runtime/direct_register_custom_op.py:/usr/local/lib/python3.10/dist-packages/lightop/_lmslim_native/vllm_compat/direct_register_custom_op.py:ro" \
  "${mounts[@]}" "$REVIEW_IMAGE_ID" sleep infinity
```

`HIP_VISIBLE_DEVICES`与`GPU_PAIR`同时设为0、1，容器内映射为逻辑卡0、1。命令创建待启动容器；模型由第3节启动。主机网络供本机压测使用；模型、库、运行态overlay和缓存均只读挂载。

## 2. 依赖安装与身份校验

安装方式是拉取／载入第1节固定厂商镜像。`requirements.txt`列出实际已安装版本；保留镜像自带Torch、Triton、SGLang与Aiter。启动模型前在新容器中核对依赖、DCU可见性及私有补丁：

```bash
docker exec -i qwen38-v2-review01 bash -s <<'CHECK_RUNTIME'
set -euo pipefail
source /opt/dtk/env.sh
python3 -I - <<'PY_CHECK'
import hashlib,importlib.metadata as m,json,os
from pathlib import Path
import torch
for name in ('sglang','torch','triton','aiter','transformers','aiohttp','requests'):
    print(name, m.version(name))
assert os.environ['HIP_VISIBLE_DEVICES'] == os.environ['GPU_PAIR'] == '0,1'
assert torch.cuda.is_available() and torch.cuda.device_count() == 2
for i in range(2):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).gcnArchName)
p = Path(os.environ['PATCH_DIR'])
manifest = json.loads((p/'native-ar-manifest.json').read_text())
for name, digest in manifest['files'].items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest() == digest, name
assert "assert os.environ['HIP_VISIBLE_DEVICES'] == '0,1'" in (p/'native_bf16_ar_guard.py').read_text()
print('REVIEW01_RUNTIME_PASS')
PY_CHECK
CHECK_RUNTIME
bash scripts/run_review01.sh check-runtime
```

预期输出 `REVIEW01_RUNTIME_PASS`与`LIBRARY_SHA256_PASS`，可见两张gfx928卡。以上是环境／文件身份核验；精度、SLA和性能以对应测试结果判定。正常复现直接使用已测 `.so`，源码另行重建见 [patch/README.md](patch/README.md)。

本机已按上述卡0、1步骤完成依赖、资产、模型启动及真实API验证，容器已停止并释放锁；见 [实际执行记录](result/evidence/readme-review01-20261017/runtime-review.json)。该验证使用已测image ID，不表示dcu镜像别名已通过在线核验。

## 3. 模型启动

```bash
RUN_ID="${REVIEW_ID}-server" bash scripts/run_review01.sh start
```

成功后输出 `Server ready` 和 `Service: http://127.0.0.1:31014`。容器为 `qwen38-v2-review01`，日志及API配置在 `logs/<REVIEW_ID>-server/server.log`、`server-info.json`。

服务固定为TP2／PP1、context67584、FA3、page64、静态显存比例0.85、运行数32、Graph32、prefill65536／chunk8192、关闭radix、Mamba no_buffer、NEXTN steps1／draft2、native BF16 one-stage AR；完整命令在 `server.log`。客户端C64／C128不会把服务运行数改成64／128。

第1节创建独立容器，模型、rocBLAS库、Aiter配置、Lightop完整文件overlay和TVM FFI缓存均按记录只读挂载。`serve.sh`通过进程级 `PYTHONPATH` 加载 `patch/sitecustomize.py`，并在启动时检查扩展和AR guard实际安装。正常复现直接使用随包已测 `.so`；源码重建说明见 [patch/README.md](patch/README.md)。

启动日志目录必须为新目录。重试前检查日志，用第8节命令停止自有容器，再从第1节换新的 `REVIEW_ID` 创建，保留失败记录。

## 4. Benchmark

在同一个服务上串行执行测试。每次使用唯一 `RUN_ID`；脚本拒绝覆盖结果目录。正式性能测量时关闭Profiler和trace，并保持服务参数不变。

### 4.1 单点复现

```bash
RUN_ID="${REVIEW_ID}-4k-c8" bash scripts/run_review01.sh benchmark 4k 8
RUN_ID="${REVIEW_ID}-64k-c8" bash scripts/run_review01.sh benchmark 64k 8
RUN_ID="${REVIEW_ID}-sharegpt-c32" bash scripts/run_review01.sh sharegpt 32
```

随机请求使用已归档的论坛参考客户端：seed20260912，输入4096／65536 Token，输出1024且ignore_eos，两轮各C请求预热、三轮各 `max(8,2*C)` 请求正式测试。

ShareGPT使用固定镜像中的 `sglang.bench_serving`：seed20260912、context67584、1000请求、一次完整1000请求预热、三轮正式；每次另有64条短预热。官方教程的示例上下文和预热文字存在差异，当前口径及差异见[口径对齐说明](docs/官方教程口径对齐-20261017.md)，不混写为环境完全相同的因果收益。

### 4.2 当前展示矩阵

按评审范围选择单点组或完整矩阵组。当前展示组的命令如下；4K客户端C64／C128不在本组：

```bash
for c in 1 8 16 32; do
  RUN_ID="${REVIEW_ID}-matrix-4k-c${c}" bash scripts/run_review01.sh benchmark 4k "$c"
done
for c in 1 8 16 32; do
  RUN_ID="${REVIEW_ID}-matrix-64k-c${c}" bash scripts/run_review01.sh benchmark 64k "$c"
done
for c in 1 4 16 32 64; do
  RUN_ID="${REVIEW_ID}-matrix-sharegpt-c${c}" bash scripts/run_review01.sh sharegpt "$c"
done
```

### 4.3 SLA边界点复核

本次已测4K边界为C7通过、C8失败；64K在不同卡对／运行阶段的C1结果不一致，尚不能认定稳定达标。可在本次独立服务上复跑这些点：

```bash
RUN_ID="${REVIEW_ID}-sla-4k-c7" bash scripts/run_review01.sh benchmark 4k 7
RUN_ID="${REVIEW_ID}-sla-4k-c8" bash scripts/run_review01.sh benchmark 4k 8
RUN_ID="${REVIEW_ID}-sla-64k-c1" bash scripts/run_review01.sh benchmark 64k 1
RUN_ID="${REVIEW_ID}-sla-64k-c2" bash scripts/run_review01.sh benchmark 64k 2
```

每个并发点必须三轮分别满足P95阈值：4K TTFT＜10秒且TPOT＜80ms；64K TTFT＜30秒且TPOT＜100ms。边界点复跑只证明这些点的情况；最大并发结论还需从低到高加压、补边界和重复稳定性记录。已完成整数点证据见 `result/summary/community-official-alignment-20261017-d.json`。`execution/`保存历史调度源码和原路径；新评审以这里的新 `RUN_ID` 单实例命令为入口。

## 5. 预期结果（本机实测，官方口径三正式轮均值；详见报告 §4 与 logs/v2_benchmark_raw/final/）

以下为本机已测参考值，吞吐为三次正式轮TPS的算术均值，不含预热；TTFT／TPOT为对应三轮mean的均值。完整三轮、尾延迟、逐轮SLA与条件见 [报告 §4](docs/性能优化报告.md#4-性能结果)，原始文件和SHA见 [logs/v2_benchmark_raw/final/](logs/v2_benchmark_raw/final/README.md)。

随机主表八档采用论坛reference口径，4K仅展示客户端C1／8／16／32。服务运行数固定为32，历史4K客户端C64／C128仅作排队诊断归档，不列入预期结果。ShareGPT单列安装版工具口径，context67584与完整1000请求预热等差异沿用第4节说明；标题中的“官方口径”不表示这些差异已消除或全部赛事准入通过。

| 场景 | 客户端C | 输出TPS | TTFT mean s | TPOT mean ms |
| --- | --- | --- | --- | --- |
| 4K/1K | 1 | 57.3612 | 1.3027 | 16.1767 |
| 4K/1K | 8 | 237.0503 | 4.8507 | 28.0799 |
| 4K/1K | 16 | 307.6920 | 7.8928 | 43.0938 |
| 4K/1K | 32 | 375.7735 | 13.8868 | 69.8337 |
| 64K/1K | 1 | 22.4353 | 26.3437 | 18.8655 |
| 64K/1K | 8 | 31.6825 | 75.8293 | 177.1380 |
| 64K/1K | 16 | 32.5044 | 152.4424 | 332.0561 |
| 64K/1K | 32 | 32.7705 | 522.0759 | 375.7277 |

| ShareGPT 客户端C | 输出TPS | TTFT mean s | TPOT mean ms |
| --- | --- | --- | --- |
| 1 | 54.5359 | 0.2862 | 17.7850 |
| 4 | 144.5965 | 0.3006 | 27.8083 |
| 16 | 326.2182 | 0.3326 | 49.0756 |
| 32 | 419.7877 | 0.5167 | 76.7629 |
| 64 | 416.8612 | 14.8590 | 77.2397 |

主展示4K C16／32正式输出分别有1／3条U+FFFD标记；未展示的高客户端并发诊断原始标记也保留。SLA须逐轮判定：4K已测C7通过、C8失败；64K C1跨阶段不一致，尚不能认定稳定达标。不同卡对同机并行、ShareGPT跨日期条件及TTFT长尾见报告，表中TPS不代表SLA或精度通过。

```bash
# 只读CPU复核：身份、预热与正式轮分离、原始测量及三轮均值。
python3 -I scripts/verify_benchmark_archive.py
```

### 服务64／128与Graph上限复测（另组条件）

这两组已使用新的服务配置重测三正式轮；独立于上面的服务32／Graph32矩阵。原始数据、采样标签、KV预算与全量逐轮数据见 [复测报告](docs/服务端64与128并发复测-20261017.md) 和 [本次原始归档](logs/v2_benchmark_raw/server64-128-20261017/README.md)。

| 4K/1K客户端C | 服务／Graph上限 | 正式期间采样运行峰值 | 输出TPS（三轮均值） | TTFT mean s | TPOT mean ms | 结果定位 |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 64／64 | 64 | 398.9657 | 25.6184 | 133.0500 | 观测达到64；另组性能参考 |
| 128 | 128／128 | 76 | 378.3503 | 151.0970 | 160.2615 | 排队诊断，实际128并发未验证 |

复跑这两种服务配置，使用 [独立复跑入口](execution/server64-128-20261017/README.md#服务器复跑入口)；该入口在原服务器路径分别占用物理4、5和6、7，不使用本页默认的32上限服务。

峰值取三正式轮绝对时间窗内部（前后各向内5秒）、每5秒一次 `sglang:num_running_reqs` 的同一原标签系列，不跨rank／标签相加。C128的KV预算333,504 tokens，不能把配置128称为实际并发128通过。两组正式响应分别有3／5条U+FFFD标记，均不满足本地4K SLA；文本标记与请求完整性分别保留。两TP2实例在同机并行，吞吐不作为隔离A/B收益。

```bash
# CPU复核本次591个源文件及两组原始数据，输出到新的运行目录。
python3 -I scripts/verify_server_capacity.py \
  --collection-root logs/v2_benchmark_raw/server64-128-20261017/server \
  --output "result/runs/${REVIEW_ID}-server-capacity-review.json"
```

## 6. 得到与核对结果

```bash
python3 -I scripts/show_results.py "result/runs/${REVIEW_ID}-4k-c8/4k-c8"
python3 -I scripts/show_results.py "result/runs/${REVIEW_ID}-64k-c8/64k-c8"
python3 -I scripts/show_results.py "result/runs/${REVIEW_ID}-sharegpt-c32/sharegpt-c32"
```

完整矩阵／SLA测试时，把参数改成对应 `RUN_ID` 下的场景目录即可。

| 结果文件 | 含义 |
|---|---|
| 随机：`result/runs/<RUN_ID>/<4k或64k>-c<C>/command.txt`、`server-info.json` | 完整客户端命令与当时服务配置 |
| 随机：`warmup-*/`、`round-*/requests.jsonl`、`round-*/summary.json` | 预热、逐请求记录（含输出SHA与片段）、时间、长度、错误及逐轮统计 |
| 随机：`reference_performance_review.json` | 三轮TPS均值、TTFT／TPOT、完整性和单列文本标记 |
| ShareGPT：`result/runs/<RUN_ID>/sharegpt-c<C>/` | 完整预热及三轮 `.jsonl`／日志、命令和服务配置 |
| ShareGPT：`validated_summary.json` | 三轮TPS、TTFT／TPOT median和P99等，3000正式请求的完整性检查 |

TPS单位为输出tokens/s；随机TTFT为秒、TPOT为毫秒。`*_p95_mean_of_rounds`是逐轮P95的均值，不能代替“三轮分别过线”的SLA判定；`show_results.py`同时打印三轮独立判定。ShareGPT的ITL P95不是TPOT P95。

随机参考客户端记录的U+FFFD标记与性能请求完整性分开呈现，不能据性能PASS宣称文本或模型能力全部通过；原有额外文本门禁失败保留原判定。ShareGPT沿用严格校验，失败时检查原始日志，保留目录并换RUN_ID重试。

已归档报告见本页顶部。历史七场景报告和失败尝试保留在 `result/evidence/`、`result/summary/`；旧 `review_summary.py`仅复核旧七场景，旧 `summarize_sla.py`采用P99诊断，当前P95结果以上述入口和逐轮原件为准。

## 7. Demo与GSM8K

服务已启动时直接演示：

```bash
bash scripts/run_review01.sh demo "请用中文简短解释张量并行。"
```

`demo`使用本机OpenAI兼容流接口。随包 [demo.mp4](docs/demo/demo.mp4)为实际SSE时间流的终端回放，原始请求／事件／SHA见[Demo说明](docs/Demo说明.md)。

GSM8K在性能测试结束后单独执行。离线评分依赖从包内固定wheel解包到私有目录，原版EvalScope 1.6.1评分器和冻结四样例提示已随包提供：

```bash
export QUALITY_OUTPUT="$SUBMIT_ROOT/result/runs/${REVIEW_ID}-gsm8k"
docker exec qwen38-v2-review01 python3 -I \
  "$SUBMIT_ROOT/execution/gsm8k-community-20261017/prepare_quality.py"
docker exec qwen38-v2-review01 python3 -I \
  "$SUBMIT_ROOT/execution/gsm8k-community-20261017/evaluate.py" \
  --base-url "http://127.0.0.1:$PORT" --output "$QUALITY_OUTPUT"
docker exec qwen38-v2-review01 python3 -I \
  "$SUBMIT_ROOT/execution/gsm8k-community-20261017/review.py" "$QUALITY_OUTPUT"
python3 -I scripts/show_results.py "$QUALITY_OUTPUT"
```

该流程为完整1319题、temperature=0、并发32，沿用历史不传max_tokens，评分 `message.content`。执行脚本保留历史并行负载标签，新评审须另行记录实际并行／隔离条件。结果含逐题响应、请求SHA、得分、截断／错误／文本标记及重评分记录。95%门槛需至少1254题正确。最终官方精度协议一致性仍待核实；LiveCodeBench v6、RealWorldQA、MathVision Without CI尚未运行。

## 8. 停止与回退

```bash
bash scripts/run_review01.sh stop
```

该命令核对专属标签、镜像ID和卡对后，只停止并删除 `qwen38-v2-review01`。容器退出后服务锁释放，日志和结果留在包内。只读挂载和进程内补丁随容器撤销。

历史6、7入口 `scripts/run.sh` 及其v1回退保留用于溯源；本页0、1入口固定candidate。停止0、1容器即撤销本次进程内补丁，运行副本和Diff保留在 `result/runs/`。

## 9. 补丁审核与包结构

| 修改类型 | 交付入口 |
|---|---|
| 自定义HIP Kernel与构建 | `patch/k100_int8_gemv_unified.hip`、`patch/build_unified.py`、`scripts/rebuild_kernel.py` |
| Python／Triton路由、LongKV、tile | `patch/sitecustomize.py`、`patch/triton_tiles.json`、`patch/diffs/`及完整输入／运行态源快照 |
| BF16 native AR资格与身份检查 | `patch/native_bf16_ar_guard.py`；底层厂商kernel保持原样 |
| 框架兼容overlay | `patch/runtime/direct_register_custom_op.py`与原始文件Diff |
| TVM缓存与上游依赖 | `patch/runtime/tvm-ffi/`、源码／构建来源及固定镜像身份 |
| 来源／校验 | `patch/bundle-manifest.json`、`manifest/package-sha256.json`、原件来源清单 |

详见[补丁说明](patch/README.md)。评审正常启动直接使用完整生效源码与已测二进制，无需手工修改共享安装。源码另行重建只写新目录，生成的新二进制须重新验证后才能对应成绩。

`config/`放服务／测试配置和外置库清单；`scripts/`放入口与校验；`patch/`放源码和补丁；`logs/`放服务日志；`result/`放原始证据及汇总；`docs/`放报告和历史快照；`execution/`放已执行批次的调度源码与GSM8K客户端。旧README和执行过程已移至 `docs/history/`，不作为当前评审步骤。

本次材料核验范围及记录见 [README与补丁整理记录](docs/README与补丁整理记录-20261017.md)。
