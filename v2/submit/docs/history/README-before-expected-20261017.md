# Qwen3.8-27B：K100-AI／SGLang／TP2 提交与复现

本目录是提交入口，按“**环境准备 → 依赖安装 → 模型启动 → Benchmark → 得到结果**”执行。复现对象为已测 **INT8 tile＋BF16 native AllReduce 候选**；优化代码、源码、分项 Diff、配置及已测二进制见本目录。整理日期：2026-10-17，北京时间。

已完成十档随机性能矩阵、SLA边界诊断、ShareGPT复核和干净容器复跑；GSM8K本地全量协议为 **1288/1319＝97.6497%**。64K SLA存在轮间波动，部分随机4K输出含U+FFFD标记；另外三项精度和最终资产准入仍待完成。[当前材料状态](docs/提交材料状态-20261017.md)列出这些限制。

| 入口 | 内容 |
|---|---|
| [补丁与完整源码](patch/README.md) | 修改项、源码、分项Diff、上游依赖、编译与回退 |
| [本轮性能与SLA报告](docs/本轮性能与SLA结果-20261017.md) | 十档TPS／TTFT／TPOT、逐轮P95边界与文本标记 |
| [GSM8K报告](docs/GSM8K精度验证-20261017.md) | 完整1319题、原始响应、独立重评分及历史对照 |
| [优化与复现说明](docs/优化与复现说明.md) | 注入方式、运行条件、源码身份、复现证据 |
| [Demo说明](docs/Demo说明.md)／[视频](docs/demo/demo.mp4) | 启动与访问；基于真实API时间流的终端回放视频 |

## 1. 环境准备

以下命令在K100-AI Linux服务器上执行，以 `/data/wangzy/v2/submit` 为例。整个 `submit` 可迁移，脚本从自身位置解析包路径。宿主需要 Bash、Python 3.9以上、Docker、`flock`、可用的DCU驱动、`/opt/hyhal` 和相应设备访问权限。

```bash
cd /data/wangzy/v2/submit
export SUBMIT_ROOT="$PWD"
export REVIEW_ID="review-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export MODEL_PATH=/data/model/Qwen3.8-27B-Channel-INT8-w8a8
export DATASET_PATH=/data/wangzy/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
export LIBRARY_DIR="$SUBMIT_ROOT/config/library_gpu4_int8tuned_stock_clean"
export PORT=31014
export NCCL_PORT=31015
```

评审需准备以下外置资产；模型、数据可按实际位置修改上述两个路径。

| 资产 | 准备与核验 |
|---|---|
| 已测模型 `Qwen3.8-27B-Channel-INT8-w8a8` | 来源记录为 `hygon/Qwen3.8-27B-Channel-INT8-w8a8`；完整模型／tokenizer／config共31文件、30,902,757,820字节。用下一节的 `check_assets.py` 与已归档SHA逐项核对。原BF16权重不是本入口的依赖 |
| ShareGPT | `ShareGPT_V3_unfiltered_cleaned_split.json`；SHA256为 `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4` |
| rocBLAS完整库 | 162文件，身份见 `config/library-sha256.json`。服务器版已包含；本地小文件归档不含约790MiB完整库，需提供同一清单对应的完整目录 |

若库由外部交付，可直接把 `LIBRARY_DIR` 指向它；也可在包内目标目录尚不存在时复制成完整目录：

```bash
# 将 /实际完整库目录 改为已交付库的位置；脚本拒绝覆盖已有目标。
python3 scripts/prepare_assets.py --library-source /实际完整库目录
```

默认评审入口固定使用 **物理卡6、7，TP=2**。启动前检查这两张卡无其他计算任务：

```bash
/opt/dtk/bin/rocm-smi -d 6 7 --showuse --showmeminfo vram
```

启动脚本使用宿主 `/data/wangzy/submit1009/result/gpu67-owner.lock` 协作互斥；该目录可由脚本创建，只用于卡对锁，无需安装旧 `submit1009` 包。GPU空闲情况仍需启动前检查。默认guard也断言6、7，单独设置 `GPU_PAIR` 不会把该入口变成任意卡对版本。历史八卡测试使用独立记录、仅改设备断言的运行副本，见 `execution/` 和对应证据。

## 2. 依赖安装与身份校验

依赖由固定厂商镜像提供。先拉取镜像，再运行检查，避免全新机器在 `check-runtime` 阶段找不到镜像。

```bash
export REVIEW_IMAGE='harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68'
export REVIEW_IMAGE_ID='sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad'
docker pull "$REVIEW_IMAGE"
test "$(docker image inspect "$REVIEW_IMAGE" --format '{{.Id}}')" = "$REVIEW_IMAGE_ID"

bash scripts/run.sh check
python3 -I scripts/verify_patch_bundle.py
mkdir -p "result/runs/${REVIEW_ID}-assets"
python3 -I scripts/check_assets.py \
  --model-path "$MODEL_PATH" --dataset-path "$DATASET_PATH" \
  --output "result/runs/${REVIEW_ID}-assets/assets.json"
bash scripts/run.sh check-runtime
```

离线评审可先 `docker load -i /已交付镜像.tar`，再检查同一image ID。`requirements.txt`记录镜像已安装版本；环境安装通过拉取／载入该镜像完成，保留其厂商Torch、Triton、SGLang和Aiter版本。模型身份检查会顺序读取约30.9GB文件，应在GPU测量之前执行。

预期：`check`输出 `PACKAGE_INTEGRITY_PASS`；补丁检查输出 `PASS`；资产检查输出 `EXTERNAL_ASSET_IDENTITY_PASS`；`check-runtime`同时输出 `LIBRARY_SHA256_PASS`。这些检查的范围是文件与运行库身份，模型精度和SLA另见各自报告。

## 3. 模型启动

```bash
RUN_ID="${REVIEW_ID}-server" bash scripts/run.sh start
```

成功后输出 `Server ready` 和 `Service: http://127.0.0.1:31014`。容器为 `qwen38-v2-submit`，日志及API配置在 `logs/<REVIEW_ID>-server/server.log`、`server-info.json`。

服务固定为TP2／PP1、context67584、FA3、page64、静态显存比例0.85、运行数32、Graph32、prefill65536／chunk8192、关闭radix、Mamba no_buffer、NEXTN steps1／draft2、native BF16 one-stage AR；完整命令在 `server.log`。客户端C64／C128不会把服务运行数改成64／128。

`run.sh`创建独立容器，模型、rocBLAS库、Aiter配置、Lightop完整文件overlay和TVM FFI缓存均按记录只读挂载。`serve.sh`通过进程级 `PYTHONPATH` 加载 `patch/sitecustomize.py`，并在启动时检查扩展和AR guard实际安装。正常复现直接使用随包已测 `.so`；源码重建说明见 [patch/README.md](patch/README.md)。

同名容器存在时入口拒绝覆盖。重试前检查日志，用第7节命令停止自有容器，再换新的 `REVIEW_ID`，保留失败记录。

## 4. Benchmark

在同一个服务上串行执行测试。每次使用唯一 `RUN_ID`；脚本拒绝覆盖结果目录。正式性能测量时关闭Profiler和trace，并保持服务参数不变。

### 4.1 单点复现

```bash
RUN_ID="${REVIEW_ID}-4k-c8" bash scripts/run.sh benchmark 4k 8
RUN_ID="${REVIEW_ID}-64k-c8" bash scripts/run.sh benchmark 64k 8
RUN_ID="${REVIEW_ID}-sharegpt-c32" bash scripts/run.sh sharegpt 32
```

随机请求使用已归档的论坛参考客户端：seed20260912，输入4096／65536 Token，输出1024且ignore_eos，两轮各C请求预热、三轮各 `max(8,2*C)` 请求正式测试。

ShareGPT使用固定镜像中的 `sglang.bench_serving`：seed20260912、context67584、1000请求、一次完整1000请求预热、三轮正式；每次另有64条短预热。官方教程的示例上下文和预热文字存在差异，当前口径及差异见[口径对齐说明](docs/官方教程口径对齐-20261017.md)，不混写为环境完全相同的因果收益。

### 4.2 完整性能矩阵

按评审范围选择单点组或完整矩阵组。完整组的命令如下：

```bash
for c in 1 8 16 32 64 128; do
  RUN_ID="${REVIEW_ID}-matrix-4k-c${c}" bash scripts/run.sh benchmark 4k "$c"
done
for c in 1 8 16 32; do
  RUN_ID="${REVIEW_ID}-matrix-64k-c${c}" bash scripts/run.sh benchmark 64k "$c"
done
for c in 1 4 16 32 64; do
  RUN_ID="${REVIEW_ID}-matrix-sharegpt-c${c}" bash scripts/run.sh sharegpt "$c"
done
```

### 4.3 SLA边界点复核

本次已测4K边界为C7通过、C8失败；64K在不同卡对／运行阶段的C1结果不一致，尚不能认定稳定达标。可在本次独立服务上复跑这些点：

```bash
RUN_ID="${REVIEW_ID}-sla-4k-c7" bash scripts/run.sh benchmark 4k 7
RUN_ID="${REVIEW_ID}-sla-4k-c8" bash scripts/run.sh benchmark 4k 8
RUN_ID="${REVIEW_ID}-sla-64k-c1" bash scripts/run.sh benchmark 64k 1
RUN_ID="${REVIEW_ID}-sla-64k-c2" bash scripts/run.sh benchmark 64k 2
```

每个并发点必须三轮分别满足P95阈值：4K TTFT＜10秒且TPOT＜80ms；64K TTFT＜30秒且TPOT＜100ms。边界点复跑只证明这些点的情况；最大并发结论还需从低到高加压、补边界和重复稳定性记录。已完成整数点证据见 `result/summary/community-official-alignment-20261017-d.json`。`execution/`保存历史调度源码和原路径；新评审以这里的新 `RUN_ID` 单实例命令为入口。

## 5. 得到与核对结果

```bash
python3 -I scripts/show_results.py "result/runs/${REVIEW_ID}-4k-c8/4k-c8"
python3 -I scripts/show_results.py "result/runs/${REVIEW_ID}-64k-c8/64k-c8"
python3 -I scripts/show_results.py "result/runs/${REVIEW_ID}-sharegpt-c32/sharegpt-c32"
```

完整矩阵／SLA测试时，把参数改成对应 `RUN_ID` 下的场景目录即可。

| 结果文件 | 含义 |
|---|---|
| 随机：`result/runs/<RUN_ID>/<4k或64k>-c<C>/command.txt`、`server-info.json` | 完整客户端命令与当时服务配置 |
| 随机：`warmup-*/`、`round-*/requests.jsonl`、`round-*/summary.json` | 预热、逐请求原始输出、时间、长度、错误及逐轮统计 |
| 随机：`reference_performance_review.json` | 三轮TPS均值、TTFT／TPOT、完整性和单列文本标记 |
| ShareGPT：`result/runs/<RUN_ID>/sharegpt-c<C>/` | 完整预热及三轮 `.jsonl`／日志、命令和服务配置 |
| ShareGPT：`validated_summary.json` | 三轮TPS、TTFT／TPOT median和P99等，3000正式请求的完整性检查 |

TPS单位为输出tokens/s；随机TTFT为秒、TPOT为毫秒。`*_p95_mean_of_rounds`是逐轮P95的均值，不能代替“三轮分别过线”的SLA判定；`show_results.py`同时打印三轮独立判定。ShareGPT的ITL P95不是TPOT P95。

随机参考客户端记录的U+FFFD标记与性能请求完整性分开呈现，不能据性能PASS宣称文本或模型能力全部通过；原有额外文本门禁失败保留原判定。ShareGPT沿用严格校验，失败时检查原始日志，保留目录并换RUN_ID重试。

已归档报告见本页顶部。历史七场景报告和失败尝试保留在 `result/evidence/`、`result/summary/`；旧 `review_summary.py`仅复核旧七场景，旧 `summarize_sla.py`采用P99诊断，当前P95结果以上述入口和逐轮原件为准。

## 6. Demo与GSM8K

服务已启动时直接演示：

```bash
bash scripts/run.sh demo "请用中文简短解释张量并行。"
```

`demo`使用本机OpenAI兼容流接口。随包 [demo.mp4](docs/demo/demo.mp4)为实际SSE时间流的终端回放，原始请求／事件／SHA见[Demo说明](docs/Demo说明.md)。

GSM8K在性能测试结束后单独执行。离线评分依赖从包内固定wheel解包到私有目录，原版EvalScope 1.6.1评分器和冻结四样例提示已随包提供：

```bash
export QUALITY_OUTPUT="$SUBMIT_ROOT/result/runs/${REVIEW_ID}-gsm8k"
docker exec qwen38-v2-submit python3 -I \
  "$SUBMIT_ROOT/execution/gsm8k-community-20261017/prepare_quality.py"
docker exec qwen38-v2-submit python3 -I \
  "$SUBMIT_ROOT/execution/gsm8k-community-20261017/evaluate.py" \
  --base-url "http://127.0.0.1:$PORT" --output "$QUALITY_OUTPUT"
docker exec qwen38-v2-submit python3 -I \
  "$SUBMIT_ROOT/execution/gsm8k-community-20261017/review.py" "$QUALITY_OUTPUT"
python3 -I scripts/show_results.py "$QUALITY_OUTPUT"
```

该流程为完整1319题、temperature=0、并发32，沿用历史不传max_tokens，评分 `message.content`。执行脚本保留历史并行负载标签，新评审须另行记录实际并行／隔离条件。结果含逐题响应、请求SHA、得分、截断／错误／文本标记及重评分记录。95%门槛需至少1254题正确。最终官方精度协议一致性仍待核实；LiveCodeBench v6、RealWorldQA、MathVision Without CI尚未运行。

## 7. 停止与回退

```bash
bash scripts/run.sh stop
```

该命令核对专属标签、镜像ID和卡对后，只停止并删除 `qwen38-v2-submit`。容器退出后服务锁释放，日志和结果留在包内。只读挂载和进程内补丁随容器撤销。

内部v1回退：停止当前容器，使用新的启动RUN_ID执行 `VARIANT=baseline-v1 bash scripts/run.sh start`。它加载 `patch/baseline-v1/` 并关闭custom AR；这是本项目历史v1对照，官方Baseline另见报告。评审默认使用candidate。

## 8. 补丁审核与包结构

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
