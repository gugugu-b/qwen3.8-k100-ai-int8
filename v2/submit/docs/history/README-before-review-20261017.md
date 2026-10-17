# Qwen3.8-27B · K100-AI / SGLang · v2/submit

> **当前材料状态**：[提交材料状态](docs/提交材料状态-20261017.md)；[GSM8K完整结果](docs/GSM8K精度验证-20261017.md)。旧执行记录按日期保留，当前结论以这两份报告为准。

> **GSM8K当前结果**：完整1319题得分1288，**97.6497%**；0请求错误、0截断、0文本异常，原始响应重评分通过。见[GSM8K精度报告](docs/GSM8K精度验证-20261017.md)。正式官方精度协议仍待对齐，其余三项精度后置。

> **2026-10-17 当前执行口径**：已取得官方教程。`run.sh benchmark`现使用论坛参考客户端：两轮预热、三轮正式、每轮max(8,2C)请求。SLA阈值按用户确认的本地赛事文档，P95统计取自教程。见[官方教程口径对齐](docs/官方教程口径对齐-20261017.md)。旧额外文本门禁失败原样保留；新性能复核单列U+FFFD，不宣称文本完整性或精度通过。


整理日期：2026-10-16（北京时间）。本目录按社区文档第6节整理，以已测 **tile＋BF16 native AllReduce 候选**为复跑版本。**当前为待补测、待准入的提交草稿，不能直接声明满足全部参赛条件。** 工作区原 `submit1009` 保留原样。

- [社区要求对照与补测清单](docs/社区要求对照与补测清单.md)：性能矩阵与SLA、材料与复现持续推进；GSM8K按追加要求独立实例并行，其余精度最后。
- [五段式性能优化报告](docs/性能优化报告.md)：环境、Baseline、方法、结果、适用边界。
- [优化与复现说明](docs/优化与复现说明.md)：源码、二进制、运行依赖、回滚和证据。
- [Demo说明](docs/Demo说明.md)：启动、访问、演示与视频录制要求。
- [七场景完整原报告](result/summary/final-performance-report-20261016.md)及同名JSON：保留逐轮值与来源SHA。

## 目录

```text
submit/
├── README.md
├── env_declare.yaml             # JSON语法的合法YAML，含已知版本与待补项
├── requirements.txt             # 使用厂商镜像预装依赖
├── config/                      # 实测Aiter配置、测试矩阵、162项库SHA清单
├── scripts/                     # 宿主入口、容器启动、benchmark、校验、Demo
├── patch/                       # 实测v2源码/二进制、v1回滚、注入Diff、运行时overlay
├── logs/                        # 已归档服务日志；新运行日志独立RUN_ID
├── result/                      # 原始证据、审核、七场景汇总；新结果放runs/
├── manifest/                    # 文件SHA及原件来源
└── docs/                        # 报告、复现、缺项与社区原文
```

## 环境准备 → 依赖 → 启动 → Benchmark → 结果

以下命令在K100-AI服务器上执行，从本目录进入。容器固定 **HCU6,7 / TP=2**；沿用工作区gpu67互斥锁，HCU0–3禁止使用。需要Docker、Python3、flock及已挂载的 `/opt/hyhal`。

1. 将本目录同步到服务器，例如 `/data/wangzy/v2/submit`。仅此目录是本次提交入口。
2. 使用 `env_declare.yaml` 指定的SGLang 0.5.12厂商镜像和digest。启动脚本在缺少指定image ID时拉取固定digest；不使用PyPI替换厂商torch/triton。`requirements.txt` 是环境记录，不执行升级安装。
3. 准备模型 `/data/model/Qwen3.8-27B-Channel-INT8-w8a8` 和ShareGPT数据。量化流程、权重SHA、与官方模型资产的一致性仍需补交。ShareGPT实测SHA已记录，启动后跑ShareGPT前强制校验。
4. **本地归档没有完整rocBLAS配置库**。在原服务器选择下述两种方式之一：

```bash
# 方式A：服务器已有完整库，只读挂载，不复制大文件。
export LIBRARY_DIR=/data/wangzy/submit1009/config/library_gpu4_int8tuned_stock_clean
# 方式B：补成可携带的完整目录（CPU复制，约束为全部162项SHA一致）。
# python3 scripts/prepare_assets.py --library-source /data/wangzy/submit1009/config/library_gpu4_int8tuned_stock_clean
# unset LIBRARY_DIR

bash scripts/run.sh check
bash scripts/run.sh check-runtime
bash scripts/run.sh start
RUN_ID=review-4k-c8-unique bash scripts/run.sh benchmark 4k 8
RUN_ID=review-64k-c8-unique bash scripts/run.sh benchmark 64k 8
RUN_ID=review-sharegpt-c32-unique bash scripts/run.sh sharegpt 32
bash scripts/run.sh demo
bash scripts/run.sh stop
```

`MODEL_PATH` / `DATASET_PATH` 可按实际路径设置；`PORT` / `NCCL_PORT` 默认31014/31015。所有启动参数见 `scripts/serve.sh`，`config/serving_config.yaml` 为声明副本。新启动在 `logs/<RUN_ID>/`，新测试在 `result/runs/<RUN_ID>/`，已有目录拒绝覆盖。请记录实际启动RUN_ID；历史目录中的PID只作证据。

`check` 只验证包内SHA与证据，**不代表运行条件齐全**；`check-runtime` 另外检查完整库、镜像及模型路径。当前整理只做CPU校验，新增入口尚未在K100-AI新容器实跑；历史交付启动检查不能冒充本目录的净室复现。

## 查看结果和后续测试

```bash
python3 scripts/review_summary.py
python3 scripts/plan_benchmarks.py missing
python3 scripts/plan_benchmarks.py sla-probe
# 计划脚本只打印命令，不启动测试；运行前先读缺项清单。
```

固定长度脚本携带了实测使用的自带benchmark，只用于延续历史性能诊断。官方教程包的脚本和SLA统计口径尚未提供，正式提交前需要对齐。ShareGPT调用镜像内 `sglang.bench_serving`。本包不包含伪造的官方成绩、SLA最大并发、视频或精度结果。

回滚：先 `bash scripts/run.sh stop`，再 `VARIANT=baseline-v1 bash scripts/run.sh start`。v1是内部优化基线，不等于官方镜像默认Baseline。

## 当前主要结果与限制

4K C8 238.2423 tok/s，64K C8 31.3849 tok/s；ShareGPT C1/4/16/32/64分别54.5359/144.5965/326.2182/419.7877/417.0917 tok/s。数据为每场景三轮实测。C64 TTFT P99为21384.0426 ms，相对用户历史v1报告回退250.6725%，不能隐藏。GSM8K独立实例已完成1288/1319（97.6497%），本地95%数值门槛达到；其余三项精度最后；社区四项均仍须达到准入阈值。


## 2026-10-16 后续执行更新

用户已授权8卡四个TP2实例。执行源码和分工见 `execution/community-v2-20261016-a/`；默认单实例run.sh仍固定6,7。四实例控制器持有每个卡对的宿主锁，并仅给自有容器传入外部锁已持有标记，serve.sh才跳过容器内重复加锁；0–7启动前都须检查空闲。算术不变，各卡对仅复制修改guard物理卡断言。正式数据保留四实例并行条件，不冒称隔离成绩。

另已收入晚间服务64/Graph64的C64失败归档 `result/evidence/sharegpt-c64-server64-failed/`：三轮473.6951tok/s、TTFT P99 6177.7480ms，但U+FFFD文本完整性检查失败，不能替换原七场景的合格数据或称修复通过。原始判定保留。本轮缺档诊断继续使用服务32/Graph32。


## 2026-10-16 23:39 社区补测已按8卡四实例启动

用户明确授权全部8卡，已核实启动前无KFD GPU进程、各卡对锁空闲；未停止或修改其它容器。新运行 `community-v2-20261016-a`：0,1测4K C1/C16；2,3测64K C1/C16；4,5测4K C32/C64/C128；6,7测64K C32，均TP2、服务32/Graph32。四实例全通过各162项数值/Graph与20项路由排除检查，API health均200，正在执行预热/正式测试；尚无本轮完整三轮场景结果，不写新增收益。

服务器 `/data/wangzy/v2/submit` 的完整rocBLAS库162项SHA已核；INT8和原始模型分别31/34项文件SHA已采集。候选site保持da5b59...，各物理卡对只改guard断言，原默认v1/共享DTK不变。服务64/Graph64的C64晚间文本完整性失败已补入提交证据，本轮不放宽其门禁。

后续独立进程 `community-v2-followup-20261016-b` 等待矩阵全部完成且完整性复核通过，再做mean/P99双口径SLA诊断、干净容器复跑与实际API Demo采集；矩阵失败即停在复核处。官方教程包尚待用户提供，诊断不能冒充官方成绩；精度四项仍安排最后。每档独立目录，失败原件保留，完成后只停本轮自有实例并释放对应锁。

执行源码：`02-测试与结果/community-v2-execution-20261016/`；实时状态：`02-测试与结果/results-community-v2-execution-20261016-a/执行状态.md`。本地续作进程会按显式inventory下载小于16MiB的证据、核SHA、补入v2/submit并更新记录；Demo视频仅据真实SSE时间流渲染并标明终端回放。并行主机竞争条件如实保留，不声明隔离显著性。


## 社区补测流水线执行归档 2026-10-17T01:31:17.053153+08:00

状态：EXECUTION_HAS_FAILURES。归档与SHA核验见 `02-测试与结果/results-community-v2-execution-20261016-a/`；新增数值如完整通过则见 `v2/submit/result/summary/community-v2-execution-20261016-b.md`。精度NOT_RUN、官方口径待确认，不代表最终准入。失败原文、未完成档位及资源释放记录均保留，不自动放宽门禁。


## GSM8K并行验证授权更新（2026-10-17）

按用户追加要求，GSM8K改为一个独立TP2实例与其他性能实例并行。卡对0,1完成已启动4K队列并释放锁后，由新容器在31400端口验证1319题；其余卡对继续性能。后续SLA等待GSM8K卡对释放。四样例冻结提示及原EvalScope 1.6.1评分器在固定镜像CPU容器中已复现历史1319/1319评分；启动时尚无新模型精度成绩；该记录已由上述完整GSM8K结果替代。按赛事≥95%判定（至少1254题正确），保留逐题原响应、SHA、错误/截断/文本标记及独立重评分。正式精度协议等同关系未核实，其他三项精度仍后置。脚本及执行说明见 `v2/submit/execution/gsm8k-community-20261017/`、`v2/submit/docs/GSM8K执行安排-20261017.md`。


## GSM8K独立实例补测 2026-10-17T05:32:04.018125+08:00

完整精度 97.6497%，95%数值门槛达到。本地协议和原始证据见 `v2/submit/docs/GSM8K精度验证-20261017.md`。其他三项精度与最终官方协议仍未完结。


## 空闲卡接续调度更新（2026-10-17）

按用户“空闲的卡都用起来加速任务进度”的追加要求，后续任务改为每个卡对完成自身前序、独立核验请求完整性并释放锁后立即接续。原矩阵测试不停止、不覆盖；只替换了尚未启动GPU工作的等待协调器。

| 卡对 | 当前/接续工作 | HTTP端口 |
|---|---|---:|
| 0,1 | 立即启动4K SLA边界测试；GSM8K已完成并释放卡对 | 31200 |
| 4,5 | 立即启动64K SLA边界测试，同一卡对先重测C1，再找边界 | 31220 |
| 2,3 | 保持当前64K C16；完成释放后接续真实Demo和干净容器4K/64K C8复跑 | 31310 → 31210 |
| 6,7 | 保持当前64K C32；完成释放后接续ShareGPT C64、服务32/Graph32长尾复核 | 31330 → 31230 |

调度源码：`execution/official-alignment-20261017-c/after_matrix_parallel.py`。完整任务分配、旧等待进程交接、逐卡对前序核验、锁和容器身份均存入原后续控制目录`community-official-followup-20261017-d`。最终汇总与本地收集入口不变。各实例均TP2；同机随机矩阵、SLA和复现/ShareGPT混合负载条件必须随结果披露，不声称隔离成绩。其他三项精度仍后置。


## 社区补测流水线执行归档 2026-10-17T07:40:05.247732+08:00

状态：DIAGNOSTICS_COLLECTED_OFFICIAL_AND_ACCURACY_PENDING。归档与SHA核验见 `02-测试与结果/results-official-alignment-20261017-c/`；新增数值如完整通过则见 `v2/submit/result/summary/community-official-alignment-20261017-d.md`。GSM8K结果见独立精度报告，其余三项精度NOT_RUN；性能口径已按教程对齐，SLA阈值按用户选择，最终准入仍未验证，不代表最终准入。失败原文、未完成档位及资源释放记录均保留，不自动放宽门禁。


本轮实际归档报告：[性能与SLA结果](docs/本轮性能与SLA结果-20261017.md)。GSM8K结果见独立精度报告，其余三项尚未验证。
