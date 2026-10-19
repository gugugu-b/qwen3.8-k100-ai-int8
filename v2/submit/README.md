# Qwen3.8-27B：K100-AI／SGLang／TP2 提交与复现

<!-- V2_MEASURED_ENV:START -->
**实测环境声明：硬件、软件、框架、模型、运行、采集。** 本报告对应K100-AI／SGLang／两卡TP=2赛道。以下将原测量记录与2026-10-17只读补采的环境信息一并声明；补采未运行GPU计算，也未重测或改写性能数据。

| 类别 | 本机实测声明 |
|---|---|
| **硬件** | 宿主8×**K100-AI（gfx928）**，单卡65520 MiB显存；每个服务只用2卡，**TP=2／PP=1**。互联为**PCIe，端点16.0 GT/s×16（PCIe 4.0×16）**；`rocm-smi --showtopo`显示所有卡对可P2P访问，HCU0–3组内／4–7组内2跳，跨组3跳。CPU为**2×Hygon C86 7490 64-core**，共128核／128逻辑CPU、8个NUMA节点；系统可见内存**1007.62 GiB**。系统盘为`nvme2n1`／**INTEL SSDPF2KX038T1，3.84 TB**；根目录、模型和结果都在同一`/dev/mapper/ubuntu--vg-ubuntu--lv` **ext4/LVM**文件系统，文件系统容量3,776,075,399,168字节。卡对、同机并行与各场景配置按性能表逐行区分。[宿主／CPU／存储原件](result/evidence/environment-declaration-20261017/host-inventory.json)、[拓扑／PCIe原件](result/evidence/environment-declaration-20261017/details.json) |
| **软件** | 宿主／容器OS均为**Ubuntu 22.04.5 LTS**；宿主内核**5.15.0-119-generic**，驱动**hycu 6.3.30-V1.4.1a**，`/opt/hyhal`存在。宿主DTK为**25.04**；实际推理容器DTK为**26.04-rc4**，DAS构建标识为**`das185`**（按厂商wheel原样记录）。实测镜像及tag：`harbor.sourcefind.cn:5443/hcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10`；image ID **`sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad`**，RepoDigest **`sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68`**。复现时按[镜像拉取与身份核验](README.md#11-镜像与宿主)执行。 |
| **框架** | 推理框架为**SGLang `0.5.12+das185.dtk2604.torch2100.2608121934.gcbf025`**；Python **3.10.12**。关键依赖：Torch `2.10.0+das.opt1.dtk2604.2608131652.g28498c`；Triton `3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1`；Aiter `0.1.5+das185.dtk2604.torch2100.2608131714.gb0979d`；Transformers **5.6.0**；Lightop `0.6.0+das.dtk2604.torch2100.2608121706.g09a4f3`；LMSlim `0.3.1+das.opt4.dtk2604.torch2100.2607241157.gf9a687`；apache-tvm-ffi **0.1.9**；aiohttp **3.14.3**；requests **2.34.2**。实际版本来自固定镜像的发行包元数据，[版本原件](result/evidence/environment-declaration-20261017/details.json)。 |
| **模型** | 使用光合官方发布的**`hygon/Qwen3.8-27B-Channel-INT8-w8a8`**，服务器路径`/data/model/Qwen3.8-27B-Channel-INT8-w8a8`，未在本地重新量化。精度为**Channel INT8 W8A8**，未量化张量保留BF16／FP32；配置`text_config.dtype=bfloat16`、压缩格式`compressed-tensors`，权重头包含I8／BF16／F32。完整模型共31文件、30,902,757,820字节，其中18个权重分片`model-000NN-of-00018.safetensors`的**SHA256前8位（NN=01→18）**依次为：`61cf64c4`、`2a6c100e`、`1e22bff9`、`2d83707e`、`a3f7143f`、`26fa317b`、`b7e34fb9`、`fb3a8f9d`、`09f95ad0`、`5eb23579`、`5319d76e`、`0f997fca`、`142e9e31`、`8cd71184`、`26b78bb4`、`17f3023f`、`7ceb03d7`、`faf47d69`。[逐文件完整SHA256](result/evidence/official-alignment-20261017/community-official-20261017-c/model-sha256.json)中取INT8模型路径对应项；[精度元数据原件](result/evidence/environment-declaration-20261017/model-and-benchmark.json)。 |
| **运行** | 下方给出**完整模型启动命令和全部显式优化环境变量**；复现入口为`bash scripts/run_review01.sh start`，当前评审卡对**0,1**，服务／Graph默认**32／32**。历史性能使用0,1／2,3／4,5／6,7各TP2实例，同机并行条件保留。容量复测用4,5、端口31540／31541、服务／Graph **64／64**，以及6,7、端口31560／31561、**128／128**；其完整原始命令和私有补丁路径见[启动命令记录](result/evidence/environment-declaration-20261017/launch-commands.json)。原ShareGPT服务64失败测量单独保留。关键参数：context67584、FA3、page64、显存比例0.85、prefill65536／chunk8192、关闭radix、Mamba `no_buffer`、NEXTN steps1／topk1／draft2、native BF16 one-stage AllReduce。 |
| **采集** | 随机负载采用[官方论坛教程](https://forum.sourcefind.cn/thread/d006a0e2-b624-11f1-93f4-0242ac150003)原版[benchmark.reference.py](scripts/benchmark.reference.py)，版本以SHA256 **`08d7e43d2b347f2a2cfef3a577f7db4ab2472b3139939db8a6731b07369c5fc6`**固定；ShareGPT采用上述固定SGLang镜像内的`sglang.bench_serving`，脚本SHA256 **`f2206dcda74932e1b148cdf7b5ef07443bb5d9e0f2729b7a67d602182e3aa050`**。[早期内部benchmark.py](scripts/benchmark.py)的SHA前8位为`1d8eca3e`，总表已区分旧口径。全部已测并发：**4K C1/2/3/4/5/6/7/8/16/32/64/128；64K C1/2/8/16/32；ShareGPT C1/4/16/32/64**。参考矩阵／SLA／干净容器／容量复测：**2轮预热（每轮C请求）＋3轮正式（每轮max(8,2C)）**；旧初测4K预热2轮、64K预热1轮，正式均3轮、每轮max(16,2C)；旧交付4K预热2轮、旧候选64K预热1轮。随机seed=20260912，输入4096／65536、输出1024、ignore_eos。ShareGPT为**1轮1000请求完整预热＋3轮各1000正式**，每次调用另有64条短预热，数据SHA固定为`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`。**总表38组／114正式轮，TPS取三正式轮算术均值、排除预热**；容量监控按5秒采样并取正式窗口内缩5秒，未开启Profiler／trace。 |

**完整启动命令与关键环境变量（默认服务32／Graph32）。** 先按[复现README](README.md)准备镜像、只读依赖挂载、卡对锁及私有`PATCH_DIR`，再由受锁入口`bash scripts/run_review01.sh start`启动；下面展开的是容器内实际模型进程，`PATCH_DIR`继承README中已准备的私有目录。原始32／64／128命令、环境及来源SHA均保存在[启动命令记录](result/evidence/environment-declaration-20261017/launch-commands.json)。

```bash
source /opt/dtk/env.sh
export CXX=/opt/dtk/bin/hipcc
export GPU_PAIR=0,1 HIP_VISIBLE_DEVICES=0,1 HSA_FORCE_FINE_GRAIN_PCIE=1
export SGLANG_USE_LIGHTOP=1 SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_ROCM_USE_AITER_MOE=False SGLANG_USE_CUDA_IPC_TRANSPORT=1
export SGLANG_USE_AITER_LINEAR_ATTN=1 SGLANG_USE_QUICK_CUSTOM_ALLREDUCE=false
export SGLANG_USE_1STAGE_ALLREDUCE=1
export ROCBLAS_TENSILE_LIBPATH=/run/qwen38-library
export TVM_FFI_CACHE_DIR=/run/qwen38-tvm-ffi
export PATCH_DIR="${PATCH_DIR:?先按复现README准备私有补丁目录}"
export PYTHONPATH="$PATCH_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 -u -m sglang.launch_server \
  --model-path /data/model/Qwen3.8-27B-Channel-INT8-w8a8 \
  --tp-size 2 --pp-size 1 \
  --attention-backend fa3 --mm-attention-backend fa3 \
  --page-size 64 --served-model-name Qwen3.8-27B-Channel-INT8-w8a8 \
  --host 0.0.0.0 --port 31014 --nccl-port 31015 \
  --context-length 67584 --disable-radix-cache --mem-fraction-static 0.85 \
  --max-running-requests 32 --max-prefill-tokens 65536 --chunked-prefill-size 8192 \
  --mamba-scheduler-strategy no_buffer --cuda-graph-max-bs 32 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-metrics \
  --speculative-algorithm NEXTN --speculative-num-steps 1 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 \
  --custom-all-reduce-backend native
```

环境补采时间与原测量时间分开记录；宿主／拓扑信息取自只读采集，镜像版本和benchmark文件哈希取自**不挂GPU设备**的只读镜像检查。各轮当时的API参数、原始命令、卡对、文本状态与成绩以总表所链原件为准。[机器可读环境声明](env_declare.yaml)与此段同步。
<!-- V2_MEASURED_ENV:END -->

本目录是提交入口，按“**环境准备 → 依赖安装 → 模型启动 → Benchmark → 得到结果**”执行。复现对象为已测 **INT8 tile＋BF16 native AllReduce 候选**；优化代码、源码、分项 Diff、配置及已测二进制见本目录。整理日期：2026-10-17，北京时间。

全部性能已汇总为38组候选测量／114正式轮的一张总表，包含所有已测并发、复测、排队诊断与原失败记录；GSM8K本地全量协议为 **1288/1319＝97.6497%**。64K SLA存在轮间波动，部分随机4K输出含U+FFFD标记；另外三项精度和最终资产准入仍待完成。[当前材料状态](docs/提交材料状态-20261017.md)列出这些限制。

| 入口 | 内容 |
|---|---|
| [补丁与完整源码](patch/README.md) | 修改项、源码、分项Diff、上游依赖、编译与回退 |
| [当前性能报告 §4](docs/性能优化报告.md#4-性能结果) | 预期结果、全部三正式轮、原始数据与均值 |
| [本轮性能与SLA报告](docs/本轮性能与SLA结果-20261017.md) | 38组统一总表入口、逐轮SLA与文本标记 |
| [GSM8K报告](docs/GSM8K精度验证-20261017.md) | 完整1319题、原始响应、独立重评分及历史对照 |
| [优化与复现说明](docs/优化与复现说明.md) | 注入方式、运行条件、源码身份、复现证据 |
| [Demo说明](docs/Demo说明.md)／[视频](docs/demo/demo.mp4) | 启动与访问；基于真实API时间流的终端回放视频 |

## 1. 环境准备

以下命令在K100-AI Linux宿主执行。完整克隆GitHub仓库后，先在仓库根目录执行 `python3 -I v2/prepare_library.py`，再执行 `cd v2/submit`；已按仓库首页完成这两步则保持当前目录继续。后续按顺序在同一终端执行，`SUBMIT_ROOT`取当前实际目录，模型与数据路径按宿主实际位置填写。评审入口使用**物理卡0、1，TP=2**，容器名 `qwen38-v2-review01`。

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
test -f env_declare.yaml && test -f scripts/run_review01.sh
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
| rocBLAS完整库 | 162文件，身份见 `config/library-sha256.json`。完整GitHub仓库已附 `v1/submit1005/config/library_gpu4_int8tuned_stock_clean.tar.gz`；上述 `v2/prepare_library.py` 将其解包至当前包的 `config/library_gpu4_int8tuned_stock_clean/` 并逐文件核对SHA。单独交付 `submit` 时应同时提供该完整目录或压缩包 |

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

### 4.2 服务32标准矩阵组

按评审范围选择单点组或此服务32标准矩阵组。全部已测并发见第5节总表；以下仅复现该配置的一组场景，4K客户端C64／C128排队诊断和服务64／128配置的独立复跑入口见总表说明：

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

## 5. 性能总表（全部已测并发与复测批次）

**全部38组v2候选测量、114正式轮合并在下表**，按场景和客户端C升序排列。同一次测量的归档副本按SHA去重；早期／最新矩阵、SLA、干净容器、高并发配置与失败测量均保留，不同条件分别一行。另有6组v1／v1-return对照不计入v2，详见 [总表说明](docs/性能总表.md) 与 [完整JSON／源SHA](result/summary/performance-all-concurrency.json)。

覆盖：4K C1、2、3、4、5、6、7、8、16、32、64、128；64K C1、2、8、16、32；ShareGPT C1、4、16、32、64。

TPS为三正式轮输出TPS算术均值，排除预热；TTFT用秒、TPOT用毫秒，延迟三项依次为mean／P95／P99的三轮均值，缺失填“—”。“服务／Graph／实测峰值”前两项来自API，第三项只填写已有正式窗口采样证据；“正式轮×请求”为轮数×每轮请求数。

<!-- V2_ALL_PERFORMANCE_TABLE:START -->
| 场景 | C | 批次／卡对 | 服务／Graph／实测峰值 | 正式轮×请求 | TPS R1／R2／R3 | **均值TPS** | TTFT mean／P95／P99（s） | TPOT mean／P95／P99（ms） | SLA通过轮 | U+FFFD正式／预热；文本状态 | 原始数据 |
|---|---:|---|---|---|---|---:|---|---|---|---|---|
| 4K/1K | 1 | 初测A（旧口径）；0,1 | 32／32／— | 3×16 | 56.8625／57.2476／57.2887 | **57.1329** | 1.3082／1.3277／1.3670 | 16.2411／17.2472／17.6080 | 3/3 | 0/0；无U+FFFD标记 | [原件](result/runs/community-v2-20261016-a-p01-matrix-4k-c1/4k-c1/) |
| 4K/1K | 1 | 参考矩阵C；0,1 | 32／32／— | 3×8 | 57.5940／57.1559／57.3337 | **57.3612** | 1.3027／1.3088／1.3099 | 16.1767／17.2370／17.4965 | 3/3 | 0/0；无U+FFFD标记 | [原件](logs/v2_benchmark_raw/final/random/4k-c1/) |
| 4K/1K | 2 | SLA复测D；0,1 | 32／32／— | 3×8 | 99.3960／99.6213／98.5990 | **99.2055** | 1.7282／2.5759／2.5799 | 18.2772／20.2407／20.5358 | 3/3 | 0/0；无U+FFFD标记 | [原件](logs/v2_benchmark_raw/final/sla/4k-c2/) |
| 4K/1K | 3 | SLA复测D；0,1 | 32／32／— | 3×8 | 108.0944／109.4428／109.6867 | **109.0746** | 2.4823／3.8628／3.8636 | 22.8353／26.3081／26.5723 | 3/3 | 0/1；文本标记未解决 | [原件](logs/v2_benchmark_raw/final/sla/4k-c3/) |
| 4K/1K | 4 | SLA复测D；0,1 | 32／32／— | 3×8 | 142.5647／145.7291／142.3393 | **143.5443** | 3.3016／5.1451／5.1515 | 24.0863／26.9638／27.1694 | 3/3 | 0/1；文本标记未解决 | [原件](logs/v2_benchmark_raw/final/sla/4k-c4/) |
| 4K/1K | 5 | SLA复测D；0,1 | 32／32／— | 3×10 | 166.6608／166.8267／169.2010 | **167.5628** | 3.7820／6.4296／6.4314 | 25.5204／29.6344／30.0432 | 3/3 | 0/1；文本标记未解决 | [原件](logs/v2_benchmark_raw/final/sla/4k-c5/) |
| 4K/1K | 6 | SLA复测D；0,1 | 32／32／— | 3×12 | 184.1519／185.4305／186.7025 | **185.4283** | 3.9745／7.7024／7.7101 | 27.6831／32.2434／32.8096 | 3/3 | 0/0；无U+FFFD标记 | [原件](logs/v2_benchmark_raw/final/sla/4k-c6/) |
| 4K/1K | 7 | SLA复测D；0,1 | 32／32／— | 3×14 | 202.6679／204.4836／206.0424 | **204.3980** | 4.4781／8.9921／8.9939 | 29.1765／34.2911／35.7012 | 3/3 | 0/0；无U+FFFD标记 | [原件](logs/v2_benchmark_raw/final/sla/4k-c7/) |
| 4K/1K | 8 | 交付复测10-10；6,7 | 32／32／— | 3×16 | 233.0752／239.8198／241.8319 | **238.2423** | 4.8125／10.1719／10.1832 | 27.9617／33.7793／34.4132 | 0/3 | 0/0；无U+FFFD标记 | [原件](result/evidence/4k-delivery/submit1009/result/delivery-native-ar-4k1k-20261010-b-native-bf16-ar/4k/) |
| 4K/1K | 8 | 参考矩阵C；0,1 | 32／32／— | 3×16 | 231.1512／238.9557／241.0439 | **237.0503** | 4.8507／10.2532／10.2686 | 28.0799／33.7680／34.8980 | 0/3 | 0/0；无U+FFFD标记 | [原件](logs/v2_benchmark_raw/final/random/4k-c8/) |
| 4K/1K | 8 | SLA复测D；0,1 | 32／32／— | 3×16 | 230.7496／237.5780／239.5601 | **235.9626** | 4.8540／10.2577／10.2688 | 28.2238／34.3219／35.4609 | 0/3 | 0/0；无U+FFFD标记 | [原件](logs/v2_benchmark_raw/final/sla/4k-c8/) |
| 4K/1K | 8 | 干净容器D；2,3 | 32／32／— | 3×16 | 232.8307／239.5053／241.6763 | **238.0041** | 4.8196／10.1853／10.1947 | 27.9834／34.0271／35.1562 | 0/3 | 0/0；无U+FFFD标记 | [原件](logs/v2_benchmark_raw/final/clean-runtime/4k-c8/) |
| 4K/1K | 16 | 初测A（旧口径）；0,1 | 32／32／— | 3×32 | 307.7341／306.4367／309.1953 | **307.7887** | 7.8893／20.5006／20.5255 | 43.0771／58.2247／62.1407 | 0/3 | 1/0；文本标记未解决 | [原件](result/runs/community-v2-20261016-a-p01-matrix-4k-c16/4k-c16/) |
| 4K/1K | 16 | 参考矩阵C；0,1 | 32／32／— | 3×32 | 307.3908／306.6908／308.9944 | **307.6920** | 7.8928／20.4734／20.5000 | 43.0938／58.7031／61.8307 | 0/3 | 1/0；文本标记未解决 | [原件](logs/v2_benchmark_raw/final/random/4k-c16/) |
| 4K/1K | 32 | 初测A（旧口径）；4,5 | 32／32／— | 3×64 | 373.6388／372.9632／378.6180 | **375.0733** | 13.8753／40.0780／41.0889 | 69.9568／107.0323／115.1116 | 0/3 | 3/0；文本标记未解决 | [原件](result/runs/community-v2-20261016-a-p45-matrix-4k-c32/4k-c32/) |
| 4K/1K | 32 | 参考矩阵C；4,5 | 32／32／— | 3×64 | 373.5505／373.7362／380.0340 | **375.7735** | 13.8868／40.7032／41.1433 | 69.8337／107.4367／115.1667 | 0/3 | 3/0；文本标记未解决 | [原件](logs/v2_benchmark_raw/final/random/4k-c32/) |
| 4K/1K | 64 | 参考矩阵C（排队）；4,5 | 32／32／— | 3×128 | 378.3249／379.3977／380.5158 | **379.4128** | 77.0132／120.2056／128.5294 | 75.5164／103.4186／113.8426 | 0/3 | 4/1；文本标记未解决 | [原件](logs/v2_benchmark_raw/final/random/4k-c64/) |
| 4K/1K | 64 | 容量复测；4,5 | 64／64／64 | 3×128 | 397.5390／400.0357／399.3224 | **398.9657** | 25.6184／77.2760／82.1669 | 133.0500／200.6000／219.5265 | 0/3 | 3/0；文本标记未解决 | [原件](logs/v2_benchmark_raw/server64-128-20261017/server/submit/result/runs/high-concurrency-20261017-a-p45-server64-4k-c64/4k-c64/) |
| 4K/1K | 128 | 参考矩阵C（排队）；4,5 | 32／32／— | 3×256 | 380.3697／379.3167／382.1848 | **380.6238** | 204.0955／281.7888／298.7418 | 78.4118／97.6487／111.2150 | 0/3 | 5/1；文本标记未解决 | [原件](logs/v2_benchmark_raw/final/random/4k-c128/) |
| 4K/1K | 128 | 容量复测（未达128）；6,7 | 128／128／76 | 3×256 | 377.9551／377.8335／379.2622 | **378.3503** | 151.0970／247.0728／262.2991 | 160.2615／226.1501／244.9978 | 0/3 | 5/6；文本标记未解决 | [原件](logs/v2_benchmark_raw/server64-128-20261017/server/submit/result/runs/high-concurrency-20261017-a-p67-server128-4k-c128/4k-c128/) |
| 64K/1K | 1 | 初测A（旧口径）；2,3 | 32／32／— | 3×16 | 22.2136／22.4248／22.4108 | **22.3497** | 26.4029／26.5296／26.7737 | 18.9781／20.2386／20.6595 | 3/3 | 0/0；无U+FFFD标记 | [原件](result/runs/community-v2-20261016-a-p23-matrix-64k-c1/64k-c1/) |
| 64K/1K | 1 | 参考矩阵C；2,3 | 32／32／— | 3×8 | 22.4074／22.5846／22.3137 | **22.4353** | 26.3437／26.3732／26.3768 | 18.8655／19.8221／19.9588 | 3/3 | 0/0；无U+FFFD标记 | [原件](logs/v2_benchmark_raw/final/random/64k-c1/) |
| 64K/1K | 1 | SLA复测D；4,5 | 32／32／— | 3×8 | 22.2528／22.0203／22.2295 | **22.1675** | 26.8825／28.1087／28.7323 | 18.8776／19.9470／20.0803 | 2/3 | 0/0；无U+FFFD标记 | [原件](logs/v2_benchmark_raw/final/sla/64k-c1/) |
| 64K/1K | 2 | SLA复测D；4,5 | 32／32／— | 3×8 | 26.6100／26.5643／26.6068 | **26.5937** | 30.3912／44.7986／51.5218 | 45.3200／65.9445／73.0279 | 0/3 | 0/0；无U+FFFD标记 | [原件](logs/v2_benchmark_raw/final/sla/64k-c2/) |
| 64K/1K | 8 | 候选复测10-16；4,5 | 32／32／— | 3×16 | 31.1472／31.5545／31.4529 | **31.3849** | 76.6476／195.0341／209.5087 | 178.7371／345.9634／362.1596 | 0/3 | 0/0；无U+FFFD标记 | [原件](result/evidence/64k-validation/submit1009/result/perf-64k-20261016-a/variants/candidate/64k/) |
| 64K/1K | 8 | 参考矩阵C；2,3 | 32／32／— | 3×16 | 31.5583／31.7749／31.7143 | **31.6825** | 75.8293／192.8001／207.0493 | 177.1380／340.2861／371.7856 | 0/3 | 0/0；无U+FFFD标记 | [原件](logs/v2_benchmark_raw/final/random/64k-c8/) |
| 64K/1K | 8 | 干净容器D；2,3 | 32／32／— | 3×16 | 31.4337／31.7779／31.7298 | **31.6471** | 75.8993／192.9987／207.3433 | 177.3519／328.0635／349.0278 | 0/3 | 0/0；无U+FFFD标记 | [原件](logs/v2_benchmark_raw/final/clean-runtime/64k-c8/) |
| 64K/1K | 16 | 初测A（旧口径）；2,3 | 32／32／— | 3×32 | 32.5345／32.5852／32.5609 | **32.5602** | 152.3757／381.7710／461.3019 | 331.2748／603.4444／680.3698 | 0/3 | 0/0；无U+FFFD标记 | [原件](result/runs/community-v2-20261016-a-p23-matrix-64k-c16/64k-c16/) |
| 64K/1K | 16 | 参考矩阵C；2,3 | 32／32／— | 3×32 | 32.5349／32.5340／32.4442 | **32.5044** | 152.4424／381.9429／462.0651 | 332.0561／600.2013／687.2172 | 0/3 | 0/0；无U+FFFD标记 | [原件](logs/v2_benchmark_raw/final/random/64k-c16/) |
| 64K/1K | 32 | 初测A（旧口径）；6,7 | 32／32／— | 3×64 | 32.8873／32.8444／32.7626 | **32.8314** | 521.5284／832.1655／961.1045 | 374.1360／617.5398／682.1557 | 0/3 | 0/0；无U+FFFD标记 | [原件](result/runs/community-v2-20261016-a-p67-matrix-64k-c32/64k-c32/) |
| 64K/1K | 32 | 参考矩阵C；6,7 | 32／32／— | 3×64 | 32.8267／32.8293／32.6554 | **32.7705** | 522.0759／833.5132／962.2725 | 375.7277／600.7631／689.2660 | 0/3 | 0/0；无U+FFFD标记 | [原件](logs/v2_benchmark_raw/final/random/64k-c32/) |
| ShareGPT | 1 | ShareGPT原测；4,5 | 32／32／— | 3×1000 | 54.7578／54.7302／54.1196 | **54.5359** | 0.2862／—／0.9446 | 17.7850／—／31.4189 | — | 0/0；无U+FFFD标记 | [原件](logs/v2_benchmark_raw/final/sharegpt/c1/) |
| ShareGPT | 4 | ShareGPT原测；6,7 | 32／32／— | 3×1000 | 145.0806／144.5570／144.1519 | **144.5965** | 0.3006／—／0.9457 | 27.8083／—／61.9548 | — | 0/0；无U+FFFD标记 | [原件](logs/v2_benchmark_raw/final/sharegpt/c4/) |
| ShareGPT | 16 | ShareGPT原测；6,7 | 32／32／— | 3×1000 | 325.8117／326.8923／325.9508 | **326.2182** | 0.3326／—／0.9819 | 49.0756／—／118.3995 | — | 0/0；无U+FFFD标记 | [原件](logs/v2_benchmark_raw/final/sharegpt/c16/) |
| ShareGPT | 32 | ShareGPT原测；6,7 | 32／32／— | 3×1000 | 420.0578／419.7860／419.5194 | **419.7877** | 0.5167／—／2.7345 | 76.7629／—／246.4288 | — | 0/0；无U+FFFD标记 | [原件](logs/v2_benchmark_raw/final/sharegpt/c32/) |
| ShareGPT | 64 | ShareGPT原测（排队）；6,7 | 32／32／— | 3×1000 | 420.4584／415.8154／415.0012 | **417.0917** | 14.8478／—／21.3840 | 77.2219／—／252.3542 | — | 0/0；无U+FFFD标记 | [原件](result/evidence/sharegpt-c4-c16-c64/submit1009/experiments/performance-validation-20261016/result/submit-current-sharegpt-pair67-20261016-a-c64-candidate/sharegpt-c64/) |
| ShareGPT | 64 | 服务64尝试（失败）；0,1 | 64／64／— | 3×1000 | 471.1975／474.6636／475.2242 | **473.6951** | 0.8999／—／6.1777 | 146.1575／—／456.1811 | — | 6/2；原文本门禁FAIL | [原件](result/evidence/sharegpt-c64-server64-failed/submit1009/experiments/performance-validation-c64-server64-20261016/result/submit-current-c64-server64-pair01-20261016-a-c64-candidate/sharegpt-c64/) |
| ShareGPT | 64 | ShareGPT复核D（排队）；6,7 | 32／32／— | 3×1000 | 415.8142／417.3176／417.4518 | **416.8612** | 14.8590／—／21.3600 | 77.2397／—／249.0410 | — | 0/0；无U+FFFD标记 | [原件](logs/v2_benchmark_raw/final/sharegpt/c64/) |
<!-- V2_ALL_PERFORMANCE_TABLE:END -->

SLA通过轮按本地阈值逐轮P95判定：4K TTFT＜10秒、TPOT＜80ms；64K＜30秒、＜100ms。ShareGPT缺同口径P95，SLA填“—”，不拿ITL P95替代。服务32的C64／128是排队负载；新C128服务／Graph虽设128，正式运行采样峰值仅76。ShareGPT服务64尝试的473.6951 TPS来自完整三轮，但原文本门禁FAIL（正式6条U+FFFD），保留失败判定。

各批次协议、预热／请求量和同机并行条件分别保留，分位数均值不是合并分位数，性能数字不代表精度或赛事准入。4K C7三轮通过／C8失败；64K C1跨批次结果不一致，不能从本表推断全局最大并发。

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
