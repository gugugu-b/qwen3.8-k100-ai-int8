# Qwen3.8-27B-Channel-INT8-w8a8 · K100-AI / SGLang

<!-- V2_MEASURED_ENV:START -->
**实测环境声明：硬件、软件、框架、模型、运行、采集。** 本报告对应K100-AI／SGLang／两卡TP=2赛道。以下将原测量记录与2026-10-17只读补采的环境信息一并声明；补采未运行GPU计算，也未重测或改写性能数据。

| 类别 | 本机实测声明 |
|---|---|
| **硬件** | 宿主8×**K100-AI（gfx928）**，单卡65520 MiB显存；每个服务只用2卡，**TP=2／PP=1**。互联为**PCIe，端点16.0 GT/s×16（PCIe 4.0×16）**；`rocm-smi --showtopo`显示所有卡对可P2P访问，HCU0–3组内／4–7组内2跳，跨组3跳。CPU为**2×Hygon C86 7490 64-core**，共128核／128逻辑CPU、8个NUMA节点；系统可见内存**1007.62 GiB**。系统盘为`nvme2n1`／**INTEL SSDPF2KX038T1，3.84 TB**；根目录、模型和结果都在同一`/dev/mapper/ubuntu--vg-ubuntu--lv` **ext4/LVM**文件系统，文件系统容量3,776,075,399,168字节。卡对、同机并行与各场景配置按性能表逐行区分。[宿主／CPU／存储原件](v2/submit/result/evidence/environment-declaration-20261017/host-inventory.json)、[拓扑／PCIe原件](v2/submit/result/evidence/environment-declaration-20261017/details.json) |
| **软件** | 宿主／容器OS均为**Ubuntu 22.04.5 LTS**；宿主内核**5.15.0-119-generic**，驱动**hycu 6.3.30-V1.4.1a**，`/opt/hyhal`存在。宿主DTK为**25.04**；实际推理容器DTK为**26.04-rc4**，DAS构建标识为**`das185`**（按厂商wheel原样记录）。实测镜像及tag：`harbor.sourcefind.cn:5443/hcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10`；image ID **`sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad`**，RepoDigest **`sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68`**。复现时按[镜像拉取与身份核验](v2/submit/README.md#11-镜像与宿主)执行。 |
| **框架** | 推理框架为**SGLang `0.5.12+das185.dtk2604.torch2100.2608121934.gcbf025`**；Python **3.10.12**。关键依赖：Torch `2.10.0+das.opt1.dtk2604.2608131652.g28498c`；Triton `3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1`；Aiter `0.1.5+das185.dtk2604.torch2100.2608131714.gb0979d`；Transformers **5.6.0**；Lightop `0.6.0+das.dtk2604.torch2100.2608121706.g09a4f3`；LMSlim `0.3.1+das.opt4.dtk2604.torch2100.2607241157.gf9a687`；apache-tvm-ffi **0.1.9**；aiohttp **3.14.3**；requests **2.34.2**。实际版本来自固定镜像的发行包元数据，[版本原件](v2/submit/result/evidence/environment-declaration-20261017/details.json)。 |
| **模型** | 使用光合官方发布的**`hygon/Qwen3.8-27B-Channel-INT8-w8a8`**，服务器路径`/data/model/Qwen3.8-27B-Channel-INT8-w8a8`，未在本地重新量化。精度为**Channel INT8 W8A8**，未量化张量保留BF16／FP32；配置`text_config.dtype=bfloat16`、压缩格式`compressed-tensors`，权重头包含I8／BF16／F32。完整模型共31文件、30,902,757,820字节，其中18个权重分片`model-000NN-of-00018.safetensors`的**SHA256前8位（NN=01→18）**依次为：`61cf64c4`、`2a6c100e`、`1e22bff9`、`2d83707e`、`a3f7143f`、`26fa317b`、`b7e34fb9`、`fb3a8f9d`、`09f95ad0`、`5eb23579`、`5319d76e`、`0f997fca`、`142e9e31`、`8cd71184`、`26b78bb4`、`17f3023f`、`7ceb03d7`、`faf47d69`。[逐文件完整SHA256](v2/submit/result/evidence/official-alignment-20261017/community-official-20261017-c/model-sha256.json)中取INT8模型路径对应项；[精度元数据原件](v2/submit/result/evidence/environment-declaration-20261017/model-and-benchmark.json)。 |
| **运行** | 下方给出**完整模型启动命令和全部显式优化环境变量**；复现入口为`bash scripts/run_review01.sh start`，当前评审卡对**0,1**，服务／Graph默认**32／32**。历史性能使用0,1／2,3／4,5／6,7各TP2实例，同机并行条件保留。容量复测用4,5、端口31540／31541、服务／Graph **64／64**，以及6,7、端口31560／31561、**128／128**；其完整原始命令和私有补丁路径见[启动命令记录](v2/submit/result/evidence/environment-declaration-20261017/launch-commands.json)。原ShareGPT服务64失败测量单独保留。关键参数：context67584、FA3、page64、显存比例0.85、prefill65536／chunk8192、关闭radix、Mamba `no_buffer`、NEXTN steps1／topk1／draft2、native BF16 one-stage AllReduce。 |
| **采集** | 随机负载采用[官方论坛教程](https://forum.sourcefind.cn/thread/d006a0e2-b624-11f1-93f4-0242ac150003)原版[benchmark.reference.py](v2/submit/scripts/benchmark.reference.py)，版本以SHA256 **`08d7e43d2b347f2a2cfef3a577f7db4ab2472b3139939db8a6731b07369c5fc6`**固定；ShareGPT采用上述固定SGLang镜像内的`sglang.bench_serving`，脚本SHA256 **`f2206dcda74932e1b148cdf7b5ef07443bb5d9e0f2729b7a67d602182e3aa050`**。[早期内部benchmark.py](v2/submit/scripts/benchmark.py)的SHA前8位为`1d8eca3e`，总表已区分旧口径。全部已测并发：**4K C1/2/3/4/5/6/7/8/16/32/64/128；64K C1/2/8/16/32；ShareGPT C1/4/16/32/64**。参考矩阵／SLA／干净容器／容量复测：**2轮预热（每轮C请求）＋3轮正式（每轮max(8,2C)）**；旧初测4K预热2轮、64K预热1轮，正式均3轮、每轮max(16,2C)；旧交付4K预热2轮、旧候选64K预热1轮。随机seed=20260912，输入4096／65536、输出1024、ignore_eos。ShareGPT为**1轮1000请求完整预热＋3轮各1000正式**，每次调用另有64条短预热，数据SHA固定为`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`。**总表38组／114正式轮，TPS取三正式轮算术均值、排除预热**；容量监控按5秒采样并取正式窗口内缩5秒，未开启Profiler／trace。 |

**完整启动命令与关键环境变量（默认服务32／Graph32）。** 先按[复现README](v2/submit/README.md)准备镜像、只读依赖挂载、卡对锁及私有`PATCH_DIR`，再由受锁入口`bash scripts/run_review01.sh start`启动；下面展开的是容器内实际模型进程，`PATCH_DIR`继承README中已准备的私有目录。原始32／64／128命令、环境及来源SHA均保存在[启动命令记录](v2/submit/result/evidence/environment-declaration-20261017/launch-commands.json)。

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

环境补采时间与原测量时间分开记录；宿主／拓扑信息取自只读采集，镜像版本和benchmark文件哈希取自**不挂GPU设备**的只读镜像检查。各轮当时的API参数、原始命令、卡对、文本状态与成绩以总表所链原件为准。[机器可读环境声明](v2/submit/env_declare.yaml)与此段同步。
<!-- V2_MEASURED_ENV:END -->

在海光 **K100-AI（gfx928）、两卡TP=2、SGLang 0.5.12** 上的INT8推理优化、Benchmark与复现材料。当前版本为 **[v2](v2/README.md)**，更新于2026-10-17；历史交付保留在 [v1](v1/README.md)。

[复现步骤](v2/submit/README.md) · [性能报告](v2/submit/docs/性能优化报告.md) · [补丁与完整源码](v2/submit/patch/README.md) · [当前材料状态](v2/submit/docs/提交材料状态-20261017.md)

## v2性能总表：全部已测并发

**全部38组v2候选测量、114正式轮合并在下表**，按场景和客户端C升序排列。同一次测量的归档副本按SHA去重；早期／最新矩阵、SLA、干净容器、高并发配置与失败测量均保留，不同条件分别一行。另有6组v1／v1-return对照不计入v2，详见 [总表说明](v2/submit/docs/性能总表.md) 与 [完整JSON／源SHA](v2/submit/result/summary/performance-all-concurrency.json)。

覆盖：4K C1、2、3、4、5、6、7、8、16、32、64、128；64K C1、2、8、16、32；ShareGPT C1、4、16、32、64。

TPS为三正式轮输出TPS算术均值，排除预热；TTFT用秒、TPOT用毫秒，延迟三项依次为mean／P95／P99的三轮均值，缺失填“—”。“服务／Graph／实测峰值”前两项来自API，第三项只填写已有正式窗口采样证据；“正式轮×请求”为轮数×每轮请求数。

<!-- V2_ALL_PERFORMANCE_TABLE:START -->
| 场景 | C | 批次／卡对 | 服务／Graph／实测峰值 | 正式轮×请求 | TPS R1／R2／R3 | **均值TPS** | TTFT mean／P95／P99（s） | TPOT mean／P95／P99（ms） | SLA通过轮 | U+FFFD正式／预热；文本状态 | 原始数据 |
|---|---:|---|---|---|---|---:|---|---|---|---|---|
| 4K/1K | 1 | 初测A（旧口径）；0,1 | 32／32／— | 3×16 | 56.8625／57.2476／57.2887 | **57.1329** | 1.3082／1.3277／1.3670 | 16.2411／17.2472／17.6080 | 3/3 | 0/0；无U+FFFD标记 | [原件](v2/submit/result/runs/community-v2-20261016-a-p01-matrix-4k-c1/4k-c1/) |
| 4K/1K | 1 | 参考矩阵C；0,1 | 32／32／— | 3×8 | 57.5940／57.1559／57.3337 | **57.3612** | 1.3027／1.3088／1.3099 | 16.1767／17.2370／17.4965 | 3/3 | 0/0；无U+FFFD标记 | [原件](v2/submit/logs/v2_benchmark_raw/final/random/4k-c1/) |
| 4K/1K | 2 | SLA复测D；0,1 | 32／32／— | 3×8 | 99.3960／99.6213／98.5990 | **99.2055** | 1.7282／2.5759／2.5799 | 18.2772／20.2407／20.5358 | 3/3 | 0/0；无U+FFFD标记 | [原件](v2/submit/logs/v2_benchmark_raw/final/sla/4k-c2/) |
| 4K/1K | 3 | SLA复测D；0,1 | 32／32／— | 3×8 | 108.0944／109.4428／109.6867 | **109.0746** | 2.4823／3.8628／3.8636 | 22.8353／26.3081／26.5723 | 3/3 | 0/1；文本标记未解决 | [原件](v2/submit/logs/v2_benchmark_raw/final/sla/4k-c3/) |
| 4K/1K | 4 | SLA复测D；0,1 | 32／32／— | 3×8 | 142.5647／145.7291／142.3393 | **143.5443** | 3.3016／5.1451／5.1515 | 24.0863／26.9638／27.1694 | 3/3 | 0/1；文本标记未解决 | [原件](v2/submit/logs/v2_benchmark_raw/final/sla/4k-c4/) |
| 4K/1K | 5 | SLA复测D；0,1 | 32／32／— | 3×10 | 166.6608／166.8267／169.2010 | **167.5628** | 3.7820／6.4296／6.4314 | 25.5204／29.6344／30.0432 | 3/3 | 0/1；文本标记未解决 | [原件](v2/submit/logs/v2_benchmark_raw/final/sla/4k-c5/) |
| 4K/1K | 6 | SLA复测D；0,1 | 32／32／— | 3×12 | 184.1519／185.4305／186.7025 | **185.4283** | 3.9745／7.7024／7.7101 | 27.6831／32.2434／32.8096 | 3/3 | 0/0；无U+FFFD标记 | [原件](v2/submit/logs/v2_benchmark_raw/final/sla/4k-c6/) |
| 4K/1K | 7 | SLA复测D；0,1 | 32／32／— | 3×14 | 202.6679／204.4836／206.0424 | **204.3980** | 4.4781／8.9921／8.9939 | 29.1765／34.2911／35.7012 | 3/3 | 0/0；无U+FFFD标记 | [原件](v2/submit/logs/v2_benchmark_raw/final/sla/4k-c7/) |
| 4K/1K | 8 | 交付复测10-10；6,7 | 32／32／— | 3×16 | 233.0752／239.8198／241.8319 | **238.2423** | 4.8125／10.1719／10.1832 | 27.9617／33.7793／34.4132 | 0/3 | 0/0；无U+FFFD标记 | [原件](v2/submit/result/evidence/4k-delivery/submit1009/result/delivery-native-ar-4k1k-20261010-b-native-bf16-ar/4k/) |
| 4K/1K | 8 | 参考矩阵C；0,1 | 32／32／— | 3×16 | 231.1512／238.9557／241.0439 | **237.0503** | 4.8507／10.2532／10.2686 | 28.0799／33.7680／34.8980 | 0/3 | 0/0；无U+FFFD标记 | [原件](v2/submit/logs/v2_benchmark_raw/final/random/4k-c8/) |
| 4K/1K | 8 | SLA复测D；0,1 | 32／32／— | 3×16 | 230.7496／237.5780／239.5601 | **235.9626** | 4.8540／10.2577／10.2688 | 28.2238／34.3219／35.4609 | 0/3 | 0/0；无U+FFFD标记 | [原件](v2/submit/logs/v2_benchmark_raw/final/sla/4k-c8/) |
| 4K/1K | 8 | 干净容器D；2,3 | 32／32／— | 3×16 | 232.8307／239.5053／241.6763 | **238.0041** | 4.8196／10.1853／10.1947 | 27.9834／34.0271／35.1562 | 0/3 | 0/0；无U+FFFD标记 | [原件](v2/submit/logs/v2_benchmark_raw/final/clean-runtime/4k-c8/) |
| 4K/1K | 16 | 初测A（旧口径）；0,1 | 32／32／— | 3×32 | 307.7341／306.4367／309.1953 | **307.7887** | 7.8893／20.5006／20.5255 | 43.0771／58.2247／62.1407 | 0/3 | 1/0；文本标记未解决 | [原件](v2/submit/result/runs/community-v2-20261016-a-p01-matrix-4k-c16/4k-c16/) |
| 4K/1K | 16 | 参考矩阵C；0,1 | 32／32／— | 3×32 | 307.3908／306.6908／308.9944 | **307.6920** | 7.8928／20.4734／20.5000 | 43.0938／58.7031／61.8307 | 0/3 | 1/0；文本标记未解决 | [原件](v2/submit/logs/v2_benchmark_raw/final/random/4k-c16/) |
| 4K/1K | 32 | 初测A（旧口径）；4,5 | 32／32／— | 3×64 | 373.6388／372.9632／378.6180 | **375.0733** | 13.8753／40.0780／41.0889 | 69.9568／107.0323／115.1116 | 0/3 | 3/0；文本标记未解决 | [原件](v2/submit/result/runs/community-v2-20261016-a-p45-matrix-4k-c32/4k-c32/) |
| 4K/1K | 32 | 参考矩阵C；4,5 | 32／32／— | 3×64 | 373.5505／373.7362／380.0340 | **375.7735** | 13.8868／40.7032／41.1433 | 69.8337／107.4367／115.1667 | 0/3 | 3/0；文本标记未解决 | [原件](v2/submit/logs/v2_benchmark_raw/final/random/4k-c32/) |
| 4K/1K | 64 | 参考矩阵C（排队）；4,5 | 32／32／— | 3×128 | 378.3249／379.3977／380.5158 | **379.4128** | 77.0132／120.2056／128.5294 | 75.5164／103.4186／113.8426 | 0/3 | 4/1；文本标记未解决 | [原件](v2/submit/logs/v2_benchmark_raw/final/random/4k-c64/) |
| 4K/1K | 64 | 容量复测；4,5 | 64／64／64 | 3×128 | 397.5390／400.0357／399.3224 | **398.9657** | 25.6184／77.2760／82.1669 | 133.0500／200.6000／219.5265 | 0/3 | 3/0；文本标记未解决 | [原件](v2/submit/logs/v2_benchmark_raw/server64-128-20261017/server/submit/result/runs/high-concurrency-20261017-a-p45-server64-4k-c64/4k-c64/) |
| 4K/1K | 128 | 参考矩阵C（排队）；4,5 | 32／32／— | 3×256 | 380.3697／379.3167／382.1848 | **380.6238** | 204.0955／281.7888／298.7418 | 78.4118／97.6487／111.2150 | 0/3 | 5/1；文本标记未解决 | [原件](v2/submit/logs/v2_benchmark_raw/final/random/4k-c128/) |
| 4K/1K | 128 | 容量复测（未达128）；6,7 | 128／128／76 | 3×256 | 377.9551／377.8335／379.2622 | **378.3503** | 151.0970／247.0728／262.2991 | 160.2615／226.1501／244.9978 | 0/3 | 5/6；文本标记未解决 | [原件](v2/submit/logs/v2_benchmark_raw/server64-128-20261017/server/submit/result/runs/high-concurrency-20261017-a-p67-server128-4k-c128/4k-c128/) |
| 64K/1K | 1 | 初测A（旧口径）；2,3 | 32／32／— | 3×16 | 22.2136／22.4248／22.4108 | **22.3497** | 26.4029／26.5296／26.7737 | 18.9781／20.2386／20.6595 | 3/3 | 0/0；无U+FFFD标记 | [原件](v2/submit/result/runs/community-v2-20261016-a-p23-matrix-64k-c1/64k-c1/) |
| 64K/1K | 1 | 参考矩阵C；2,3 | 32／32／— | 3×8 | 22.4074／22.5846／22.3137 | **22.4353** | 26.3437／26.3732／26.3768 | 18.8655／19.8221／19.9588 | 3/3 | 0/0；无U+FFFD标记 | [原件](v2/submit/logs/v2_benchmark_raw/final/random/64k-c1/) |
| 64K/1K | 1 | SLA复测D；4,5 | 32／32／— | 3×8 | 22.2528／22.0203／22.2295 | **22.1675** | 26.8825／28.1087／28.7323 | 18.8776／19.9470／20.0803 | 2/3 | 0/0；无U+FFFD标记 | [原件](v2/submit/logs/v2_benchmark_raw/final/sla/64k-c1/) |
| 64K/1K | 2 | SLA复测D；4,5 | 32／32／— | 3×8 | 26.6100／26.5643／26.6068 | **26.5937** | 30.3912／44.7986／51.5218 | 45.3200／65.9445／73.0279 | 0/3 | 0/0；无U+FFFD标记 | [原件](v2/submit/logs/v2_benchmark_raw/final/sla/64k-c2/) |
| 64K/1K | 8 | 候选复测10-16；4,5 | 32／32／— | 3×16 | 31.1472／31.5545／31.4529 | **31.3849** | 76.6476／195.0341／209.5087 | 178.7371／345.9634／362.1596 | 0/3 | 0/0；无U+FFFD标记 | [原件](v2/submit/result/evidence/64k-validation/submit1009/result/perf-64k-20261016-a/variants/candidate/64k/) |
| 64K/1K | 8 | 参考矩阵C；2,3 | 32／32／— | 3×16 | 31.5583／31.7749／31.7143 | **31.6825** | 75.8293／192.8001／207.0493 | 177.1380／340.2861／371.7856 | 0/3 | 0/0；无U+FFFD标记 | [原件](v2/submit/logs/v2_benchmark_raw/final/random/64k-c8/) |
| 64K/1K | 8 | 干净容器D；2,3 | 32／32／— | 3×16 | 31.4337／31.7779／31.7298 | **31.6471** | 75.8993／192.9987／207.3433 | 177.3519／328.0635／349.0278 | 0/3 | 0/0；无U+FFFD标记 | [原件](v2/submit/logs/v2_benchmark_raw/final/clean-runtime/64k-c8/) |
| 64K/1K | 16 | 初测A（旧口径）；2,3 | 32／32／— | 3×32 | 32.5345／32.5852／32.5609 | **32.5602** | 152.3757／381.7710／461.3019 | 331.2748／603.4444／680.3698 | 0/3 | 0/0；无U+FFFD标记 | [原件](v2/submit/result/runs/community-v2-20261016-a-p23-matrix-64k-c16/64k-c16/) |
| 64K/1K | 16 | 参考矩阵C；2,3 | 32／32／— | 3×32 | 32.5349／32.5340／32.4442 | **32.5044** | 152.4424／381.9429／462.0651 | 332.0561／600.2013／687.2172 | 0/3 | 0/0；无U+FFFD标记 | [原件](v2/submit/logs/v2_benchmark_raw/final/random/64k-c16/) |
| 64K/1K | 32 | 初测A（旧口径）；6,7 | 32／32／— | 3×64 | 32.8873／32.8444／32.7626 | **32.8314** | 521.5284／832.1655／961.1045 | 374.1360／617.5398／682.1557 | 0/3 | 0/0；无U+FFFD标记 | [原件](v2/submit/result/runs/community-v2-20261016-a-p67-matrix-64k-c32/64k-c32/) |
| 64K/1K | 32 | 参考矩阵C；6,7 | 32／32／— | 3×64 | 32.8267／32.8293／32.6554 | **32.7705** | 522.0759／833.5132／962.2725 | 375.7277／600.7631／689.2660 | 0/3 | 0/0；无U+FFFD标记 | [原件](v2/submit/logs/v2_benchmark_raw/final/random/64k-c32/) |
| ShareGPT | 1 | ShareGPT原测；4,5 | 32／32／— | 3×1000 | 54.7578／54.7302／54.1196 | **54.5359** | 0.2862／—／0.9446 | 17.7850／—／31.4189 | — | 0/0；无U+FFFD标记 | [原件](v2/submit/logs/v2_benchmark_raw/final/sharegpt/c1/) |
| ShareGPT | 4 | ShareGPT原测；6,7 | 32／32／— | 3×1000 | 145.0806／144.5570／144.1519 | **144.5965** | 0.3006／—／0.9457 | 27.8083／—／61.9548 | — | 0/0；无U+FFFD标记 | [原件](v2/submit/logs/v2_benchmark_raw/final/sharegpt/c4/) |
| ShareGPT | 16 | ShareGPT原测；6,7 | 32／32／— | 3×1000 | 325.8117／326.8923／325.9508 | **326.2182** | 0.3326／—／0.9819 | 49.0756／—／118.3995 | — | 0/0；无U+FFFD标记 | [原件](v2/submit/logs/v2_benchmark_raw/final/sharegpt/c16/) |
| ShareGPT | 32 | ShareGPT原测；6,7 | 32／32／— | 3×1000 | 420.0578／419.7860／419.5194 | **419.7877** | 0.5167／—／2.7345 | 76.7629／—／246.4288 | — | 0/0；无U+FFFD标记 | [原件](v2/submit/logs/v2_benchmark_raw/final/sharegpt/c32/) |
| ShareGPT | 64 | ShareGPT原测（排队）；6,7 | 32／32／— | 3×1000 | 420.4584／415.8154／415.0012 | **417.0917** | 14.8478／—／21.3840 | 77.2219／—／252.3542 | — | 0/0；无U+FFFD标记 | [原件](v2/submit/result/evidence/sharegpt-c4-c16-c64/submit1009/experiments/performance-validation-20261016/result/submit-current-sharegpt-pair67-20261016-a-c64-candidate/sharegpt-c64/) |
| ShareGPT | 64 | 服务64尝试（失败）；0,1 | 64／64／— | 3×1000 | 471.1975／474.6636／475.2242 | **473.6951** | 0.8999／—／6.1777 | 146.1575／—／456.1811 | — | 6/2；原文本门禁FAIL | [原件](v2/submit/result/evidence/sharegpt-c64-server64-failed/submit1009/experiments/performance-validation-c64-server64-20261016/result/submit-current-c64-server64-pair01-20261016-a-c64-candidate/sharegpt-c64/) |
| ShareGPT | 64 | ShareGPT复核D（排队）；6,7 | 32／32／— | 3×1000 | 415.8142／417.3176／417.4518 | **416.8612** | 14.8590／—／21.3600 | 77.2397／—／249.0410 | — | 0/0；无U+FFFD标记 | [原件](v2/submit/logs/v2_benchmark_raw/final/sharegpt/c64/) |
<!-- V2_ALL_PERFORMANCE_TABLE:END -->

SLA通过轮按本地阈值逐轮P95判定：4K TTFT＜10秒、TPOT＜80ms；64K＜30秒、＜100ms。ShareGPT缺同口径P95，SLA填“—”，不拿ITL P95替代。服务32的C64／128是排队负载；新C128服务／Graph虽设128，正式运行采样峰值仅76。ShareGPT服务64尝试的473.6951 TPS来自完整三轮，但原文本门禁FAIL（正式6条U+FFFD），保留失败判定。

各批次协议、预热／请求量和同机并行条件分别保留，分位数均值不是合并分位数，性能数字不代表精度或赛事准入。4K C7三轮通过／C8失败；64K C1跨批次结果不一致，不能从本表推断全局最大并发。

## 从仓库开始复现

```bash
git clone https://github.com/gugugu-b/qwen3.8-k100-ai-int8.git
cd qwen3.8-k100-ai-int8

# 从仓库已有压缩包准备v2所需rocBLAS库；逐项校验162文件。
python3 -I v2/prepare_library.py
cd v2/submit
bash scripts/run_review01.sh check
```

随后按 **[v2提交README](v2/submit/README.md)** 顺序执行：

1. 环境准备：镜像拉取／载入、hycu驱动与设备、官方Channel-INT8权重及ShareGPT、资产SHA、端口和卡对锁检查、完整 `docker run`。
2. 依赖安装：使用固定厂商镜像，核对Torch、Triton、SGLang、Aiter及两张gfx928可见性。
3. 模型启动：默认物理卡0、1；私有guard副本仅改设备断言并记录Diff与SHA。
4. Benchmark：单点／完整矩阵／SLA复核，使用唯一RUN_ID。
5. 得到结果：读取逐轮原件、均值、文本标记与逐轮SLA，停止自有容器。

本机已按卡0、1步骤实际验证依赖、模型／数据SHA、模型health和真实API，两个rank的AR guard生效；自有容器已停止并释放锁。见 [运行验证](v2/submit/result/evidence/readme-review01-20261017/runtime-review.json)。

README采用用户指定的 `harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10` 拉取地址；本机验证时该地址因服务器DNS失败未完成在线身份核验。实测镜像来自同版本 `hcu/admin/base/sglang`，固定image ID为 `sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad`。提交README保留固定身份门禁与实测引用；不能仅按相同tag认定两个地址等价。

## v2改动与交付

v2沿用INT8 Multi-M LDS GEMV、权重转置缓存、Triton Long-KV、Aiter配置和rocBLAS库，增加 **18个INT8 Triton tile** 与 **BF16 native one-stage AllReduce资格筛选**。底层厂商native AR kernel保持原样；完整自定义源码、五份分项Diff、实际运行overlay、上游快照、构建入口和来源边界见 [补丁说明](v2/submit/patch/README.md)。卡0、1的设备断言Diff与完整guard也已归档。

```text
.
├── README.md                    # 当前v2总览与入口
├── PERFORMANCE.md               # v1历史性能记录
├── LICENSE
├── v1/                          # 历史交付与原运行库压缩包
└── v2/
    ├── README.md                # v2版本说明与资产准备
    ├── prepare_library.py       # 从已有压缩包准备并核验完整库
    ├── release.json             # 版本、源包身份与清单
    ├── tools/seal_manifest.py   # 材料有意变更后的封存工具
    └── submit/
        ├── README.md            # 环境→依赖→启动→Benchmark→结果
        ├── env_declare.yaml
        ├── scripts/             # 运行与CPU校验入口
        ├── patch/               # 完整源码、Diff、已测扩展与来源
        ├── config/              # 配置及完整库SHA清单
        ├── execution/           # 实际调度与精度脚本
        ├── logs/                # 服务日志、原始性能归档
        ├── result/              # 原始证据、逐请求与汇总
        ├── manifest/            # 包文件SHA
        └── docs/                # 性能、精度、Demo与材料状态
```

原始逐请求、日志和失败判定保持不变，新增完整总表与来源核验，包身份见当前manifest，运行历史继续保留。模型权重与ShareGPT大文件按README在包外准备；展开的rocBLAS库不重复纳入Git，由已有压缩包准备。完整原始日志和已测二进制随v2提供。

## 精度与当前限制

GSM8K本地冻结EvalScope 1.6.1四样例全量协议：**1288/1319＝97.6497%**，达到本地95%门槛，逐题响应与独立重评分通过，见 [GSM8K报告](v2/submit/docs/GSM8K精度验证-20261017.md)。最终官方协议等价性尚待核对；v2的LiveCodeBench v6、RealWorldQA、MathVision Without CI尚未运行。v1精度记录不能代替v2结果。

当前仍需解决64K SLA稳定性、其余精度、最终协议和组合验证，不宣称全部赛事准入通过。历史构建日志及供应商native AR匹配完整C++／HIP源码的交付边界见补丁说明。[Demo视频](v2/submit/docs/demo/demo.mp4)来自真实API事件时间流的终端回放。

项目代码许可见 [Apache 2.0](LICENSE)；随包上游源码与依赖的许可见对应目录。
