# Qwen3.8-27B K100-AI SGLang性能优化报告

版本：v2/submit，2026-10-16。按社区§6.2五段式编排。**本报告是已有实测的可追溯草稿；随机矩阵、SLA、净室性能与准入材料未齐，不能声称最终获准提交。**

## 第一部分：测试环境

- 硬件：K100-AI（gfx928），每服务两卡TP=2、PP=1。4K/ShareGPT C4/16/32/64在物理HCU6,7；64K、ShareGPT C1在4,5，具体以每场景原证据为准。全部不使用HCU0–3。CPU/内存/NUMA/驱动/互联拓扑待净室采集。
- 软件：官方指定SGLang 0.5.12厂商镜像，DTK2604、Ubuntu22.04、Python3.10.12。固定image ID/digest、torch2.10、Triton3.6、Aiter0.1.5的完整厂商版本见 `env_declare.yaml`；不把通用PyPI版本当实测依赖。
- 模型：`/data/model/Qwen3.8-27B-Channel-INT8-w8a8`，Channel INT8 W8A8投影、BF16相关输出；官方原模型为Qwen/Qwen3.8-27B。权重、tokenizer SHA与量化复现待补。
- 服务：context67584，FA3，page64，mem0.85，max_running_requests32，Graph max bs32，prefill65536/chunk8192，radix cache关闭，no_buffer，NEXTN steps1/draft2/topk1，native AR及小BF16 eligibility guard。保留full_nvlink=false，未合入失败候选。
- rocBLAS：使用完整stock clean库，GemmGeneralityAdd.dat SHA `c3e295b24494507ede49ee2b330a96ddc9fe1654c58e5fd9611b3adf8b227c55`；162项库清单见config，不能混用INT8/BF16选优产物。Aiter五项实测配置已收入config/aiter。
- 数据：ShareGPT文件SHA `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`；随机输入为自带benchmark的固定长度token序列，4096或65536输入、1024输出、ignore_eos=true。seed20260912。

## 第二部分：Baseline

### 社区官方K100-AI SGLang参考

以下原数值转录自社区§4.2.3；TTFT/TPOT沿用原表数字，表头没有单位，不据此猜测最终评分统计。原社区快照的ShareGPT K100-AI表为空。2026-10-17取得的官方教程补充了C1/4/16/32/64参考值12.150/20.260/104.905/168.419/242.089 tok/s；教程镜像与本候选镜像差异继续保留，不直接当同环境因果收益。

| 输入/输出 | C | 官方TPS | 官方TTFT原值 | 官方TPOT原值 |
|---|---:|---:|---:|---:|
| 4K/1K | 1 | 9.055 | 3.49 | 107.128 |
| 4K/1K | 8 | 52.48 | 16.292 | 136.651 |
| 4K/1K | 16 | 74.011 | 29.264 | 187.778 |
| 4K/1K | 32 | 142.864 | 39.483 | 185.586 |
| 4K/1K | 64 | 194.387 | 76.489 | 254.519 |
| 4K/1K | 128 | 181.599 | 178.759 | 449.788 |
| 64K/1K | 1 | 7.52 | 51.219 | 83.045 |
| 64K/1K | 8 | 13.929 | 230.054 | 349.988 |
| 64K/1K | 16 | 14.242（新教程） | 未列 | 未列 |
| 64K/1K | 32 | ABORTED（新教程） | 未列 | 未列 |

本候选4K/64K C8绝对TPS为238.2423/31.3849。官方工具、权重、请求量、流事件计时和单位尚未对齐，暂不将它们与官方52.48/13.929相除后声明正式优化收益。官方SLA最大并发两场景均为0，不能以0作分母计算百分比提升；候选最大并发仍未知。

### 用户指定历史v1报告对照

以下百分比仅为 `(候选三轮均值 / 用户历史报告值 − 1) ×100`，参考原文在docs/source，非官方评分，也非同期同卡对因果收益。

| 场景 | 历史v1 tok/s | 候选 tok/s | 历史参考变化 |
|---|---:|---:|---:|
| 4K/1K C8 | 206.8400 | 238.2423 | +15.1819% |
| 64K/1K C8 | 30.3900 | 31.3849 | +3.2736% |
| ShareGPT C1 | 43.1300 | 54.5359 | +26.4453% |
| ShareGPT C4 | 133.3700 | 144.5965 | +8.4176% |
| ShareGPT C16 | 289.3200 | 326.2182 | +12.7534% |
| ShareGPT C32 | 338.4100 | 419.7877 | +24.0471% |
| ShareGPT C64 | 417.4200 | 417.0917 | -0.0787% |

另有4K同期v1→候选→返回v1的三轮对照：前后v1合并208.6413 tok/s，候选238.2423 tok/s，+14.1875%。这个分母与历史报告206.84不同，因此不混用+15.1819%。其小样本尾指标见result/evidence/4k-delivery，不能泛化至其它并发或全模型精度。

## 第三部分：优化方法

| 优化项 | 瓶颈与链路 | 做法及机制 | 证据与适用范围 |
|---|---|---|---|
| 原v1 LDS-x INT8 GEMV | 小M decode投影；通用GEMM在小batch时开销偏高 | HIP特化小M形状、LDS复用；保持原有形状/数据类型路由与回退 | patch/HIP源码与so；已继承，不把历史收益归为本次新增 |
| 原v1 Long-KV GQA | 长上下文Full Attention prefill | 动态载入镜像中的Triton实现并应用补丁，BM128/BN64/warps8 | sitecustomize中明确变更代码；依赖固定镜像与硬件 |
| 原v1 Aiter配置/MTP | linear attention及decode生成链路 | 五项实测gfx928配置；NEXTN steps1、draft2保留 | config/aiter、API与服务日志；不冒称本次逐项新增收益 |
| v2精确shape tile | INT8投影通用tile对实测M/K/N效率不足；首版服务激活为0暴露接入缺口 | 对18个实测shape覆写已选tile，同时覆盖Lightop私有LMSlim与独立LMSlim身份；保留其它形状原分派 | triton_tiles.json、v2-vs-v1.diff、包装/Graph检查及激活日志。历史tile固定4K约+9%、64K约+3%、C32约+4%，仅属于当时条件 |
| v2 BF16 native copy-in AR | TP2同步开销 | 使用厂商已有确定性one-stage native实现，仅BF16/TP2且bytes<16MiB走copy-in，FP32/大tensor/不合格输入保留RCCL | guard校验SGLang源码和sgl_kernel二进制SHA，数值门禁摘要与实测源码完整保留；full_nvlink=false |

理论预期是降低小batch矩阵与跨卡同步开销，不能据此推导所有场景收益。18个tile的算子实测/Graph证据与native AR数值检查各有范围；最终tile+AR组合实际模型Graph/compiler关联仍REVIEW_REQUIRED，历史tile-only PASS不能移植为组合PASS。未合入SiLU融合、Fill、Split-K、kpack、pipeline等失败/无收益实验。自有HIP、注入代码、回滚源码和差异已随包提供；厂商内核由固定镜像提供，未修改该二进制。

## 第四部分：性能结果

每个场景三轮；下表延迟统一换算成ms，均为对应轮统计量的算术均值，**P99列不是合并总体P99**。固定长度每轮16请求，ShareGPT每轮1000请求；各场景正式请求均零错误。

| 场景 | 输出TPS | TTFT mean ms | TPOT mean ms | TTFT P99 ms | TPOT P99 ms |
|---|---:|---:|---:|---:|---:|
| 4K/1K C8 | 238.2423 | 4812.4610 | 27.9617 | 10183.2424 | 34.4132 |
| 64K/1K C8 | 31.3849 | 76647.6033 | 178.7371 | 209508.6883 | 362.1596 |
| ShareGPT C1 | 54.5359 | 286.1929 | 17.7850 | 944.5735 | 31.4189 |
| ShareGPT C4 | 144.5965 | 300.5613 | 27.8083 | 945.6639 | 61.9548 |
| ShareGPT C16 | 326.2182 | 332.5501 | 49.0756 | 981.8911 | 118.3995 |
| ShareGPT C32 | 419.7877 | 516.7407 | 76.7629 | 2734.5376 | 246.4288 |
| ShareGPT C64 | 417.0917 | 14847.8117 | 77.2219 | 21384.0426 | 252.3542 |

固定4K于10月10日测量；64K及ShareGPT于10月16日。固定4K两轮各8请求预热、64K一轮8请求预热；ShareGPT每档一轮完整1000请求预热，完整预热和每正式轮另带64短预热。逐轮TPS、全部保存延迟、成功数及SHA见result/summary同名JSON，原始请求/输出与完整命令、服务参数在result/evidence。

吞吐使用输出Token总数/正式墙钟时长；未除以并发冒充单路速率。固定负载TTFT从请求到首有效输出事件、TPOT由输出长度和首尾时间计算；MTP一次流事件可能包含多Token，ITL不能直接等同TPOT。ShareGPT保留安装版benchmark统计并独立汇总；text_chunks及逐请求最终latency未保存，因此TPOT/ITL/E2E不能声称全部按请求精确重建。

ShareGPT C64：TPS相对历史−0.0787%，TTFT P99相对历史6098.01ms回退+250.6725%；C1 TTFT P99相对历史910.57ms回退+3.7343%。其它改善与这两项回退同时呈现。C64是客户端并发64、服务最大运行数32，尚未验证服务运行数64配置，根因未证明。

随机其它并发档已有上一轮部分三轮数据，完整状态及失败见新增复核说明；SLA最大并发尚未完成。当前按官方参考协议重跑十档，再补用户确认阈值下的P95边界。`scripts/plan_benchmarks.py missing`提供建议8个缺档计划。按P99临时观察，4K C8的TTFT超过10s；64K C8两个延迟均超阈值。它们不能构成SLA最大并发结论。

## 第五部分：适用范围与Trade-off

- 硬件/ABI绑定：gfx928、指定厂商镜像、Python3.10、torch2.10，so不能直接用于其它平台；新编译产物需重新门禁。
- 精确shape tile只对命中形状生效；native AR仅限BF16/TP2/<16MiB，不假定其它batch或精度有收益。
- 显存与时延：MTP、Graph、max_running_requests、chunked prefill共同影响显存和排队；尚无同口径显存增量/长稳态报告，不填写推测数值。
- 测试限制：三轮极差不是统计噪声上界；部分场景两卡对任务重叠；历史v1预热条件不同，不能据此证明隔离因果显著性。
- 工程限制：本地完整库、权重量化材料、官方脚本、Demo视频和当前目录净室性能待补。历史新镜像启动检查只证明当时能启动。
- 内部20%门槛未完成，不等于社区要求的统一准入条件；当前未冻结、原工作区默认v1保持。
- 精度：GSM8K独立TP2实例已完成1288/1319（97.6497%），本地≥95%数值门槛达到，详见[GSM8K报告](GSM8K精度验证-20261017.md)；LiveCodeBench v6、RealWorldQA、MathVision（Without CI）仍NOT_RUN并后置；分别必须≥95.0/88.3/83.9/88.0，未验证前不能称精度通过。任何因精度修复导致的实现变化都要重验相关性能。


## 2026-10-16 后续执行更新

用户已授权8卡四个TP2实例。执行源码和分工见 `execution/community-v2-20261016-a/`；默认单实例run.sh仍固定6,7。四实例控制器持有每个卡对的宿主锁，并仅给自有容器传入外部锁已持有标记，serve.sh才跳过容器内重复加锁；0–7启动前都须检查空闲。算术不变，各卡对仅复制修改guard物理卡断言。正式数据保留四实例并行条件，不冒称隔离成绩。

另已收入晚间服务64/Graph64的C64失败归档 `result/evidence/sharegpt-c64-server64-failed/`：三轮473.6951tok/s、TTFT P99 6177.7480ms，但U+FFFD文本完整性检查失败，不能替换原七场景的合格数据或称修复通过。原始判定保留。本轮缺档诊断继续使用服务32/Graph32。
