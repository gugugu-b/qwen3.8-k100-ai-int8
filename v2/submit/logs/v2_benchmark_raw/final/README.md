# 原始性能数据：最终展示索引

本目录是本机已测数据的只读归档，原始文件按来源逐字节复制；final表示本次报告选用的数据集，不表示全部赛事准入通过。没有重跑或改写历史测量。

从提交包根目录执行 `python3 -I scripts/verify_benchmark_archive.py` 可核SHA、重算三轮及原始数据校验；`summary.json`为机器可读结果，`manifest.json`给出每个文件的原路径、大小和SHA。

报告入口：[性能优化报告 §4](../../../docs/性能优化报告.md#4-性能结果)。README预期结果仅采用primary角色：随机8档，ShareGPT5档；4K客户端C64／C128仅作排队诊断归档，不进入预期结果和报告主表。SLA与clean-runtime单列。

| 角色 | 场景 | 卡对 | 三正式轮均值 TPS | 原路径（相对submit） |
|---|---|---|---:|---|
| primary_random | [random/4k-c1](random/4k-c1/) | 0,1 | 57.3612 | `result/runs/community-official-20261017-c-p01-matrix-4k-c1/4k-c1` |
| primary_random | [random/4k-c8](random/4k-c8/) | 0,1 | 237.0503 | `result/runs/community-official-20261017-c-p01-matrix-4k-c8/4k-c8` |
| primary_random | [random/4k-c16](random/4k-c16/) | 0,1 | 307.6920 | `result/runs/community-official-20261017-c-p01-matrix-4k-c16/4k-c16` |
| primary_random | [random/4k-c32](random/4k-c32/) | 4,5 | 375.7735 | `result/runs/community-official-20261017-c-p45-matrix-4k-c32/4k-c32` |
| primary_random | [random/64k-c1](random/64k-c1/) | 2,3 | 22.4353 | `result/runs/community-official-20261017-c-p23-matrix-64k-c1/64k-c1` |
| primary_random | [random/64k-c8](random/64k-c8/) | 2,3 | 31.6825 | `result/runs/community-official-20261017-c-p23-matrix-64k-c8/64k-c8` |
| primary_random | [random/64k-c16](random/64k-c16/) | 2,3 | 32.5044 | `result/runs/community-official-20261017-c-p23-matrix-64k-c16/64k-c16` |
| primary_random | [random/64k-c32](random/64k-c32/) | 6,7 | 32.7705 | `result/runs/community-official-20261017-c-p67-matrix-64k-c32/64k-c32` |
| primary_sharegpt | [sharegpt/c1](sharegpt/c1/) | 4,5 | 54.5359 | `result/evidence/sharegpt-c1/submit1009/experiments/performance-validation-20261016/result/submit-current-c1-pair45-20261016-a-c1-candidate/sharegpt-c1` |
| primary_sharegpt | [sharegpt/c4](sharegpt/c4/) | 6,7 | 144.5965 | `result/evidence/sharegpt-c4-c16-c64/submit1009/experiments/performance-validation-20261016/result/submit-current-sharegpt-pair67-20261016-a-c4-candidate/sharegpt-c4` |
| primary_sharegpt | [sharegpt/c16](sharegpt/c16/) | 6,7 | 326.2182 | `result/evidence/sharegpt-c4-c16-c64/submit1009/experiments/performance-validation-20261016/result/submit-current-sharegpt-pair67-20261016-a-c16-candidate/sharegpt-c16` |
| primary_sharegpt | [sharegpt/c32](sharegpt/c32/) | 6,7 | 419.7877 | `result/evidence/sharegpt-c32/submit1009/experiments/performance-validation-20261016/result/perf-sharegpt-20261016-a-c32-candidate/sharegpt-c32` |
| primary_sharegpt | [sharegpt/c64](sharegpt/c64/) | 6,7 | 416.8612 | `result/runs/community-official-followup-20261017-d-p67-sharegpt-c64-recheck/sharegpt-c64` |
| sla | [sla/4k-c2](sla/4k-c2/) | 0,1 | 99.2055 | `result/runs/community-official-followup-20261017-d-p01-sla-4k-c2/4k-c2` |
| sla | [sla/4k-c3](sla/4k-c3/) | 0,1 | 109.0746 | `result/runs/community-official-followup-20261017-d-p01-sla-4k-c3/4k-c3` |
| sla | [sla/4k-c4](sla/4k-c4/) | 0,1 | 143.5443 | `result/runs/community-official-followup-20261017-d-p01-sla-4k-c4/4k-c4` |
| sla | [sla/4k-c5](sla/4k-c5/) | 0,1 | 167.5628 | `result/runs/community-official-followup-20261017-d-p01-sla-4k-c5/4k-c5` |
| sla | [sla/4k-c6](sla/4k-c6/) | 0,1 | 185.4283 | `result/runs/community-official-followup-20261017-d-p01-sla-4k-c6/4k-c6` |
| sla | [sla/4k-c7](sla/4k-c7/) | 0,1 | 204.3980 | `result/runs/community-official-followup-20261017-d-p01-sla-4k-c7/4k-c7` |
| sla | [sla/4k-c8](sla/4k-c8/) | 0,1 | 235.9626 | `result/runs/community-official-followup-20261017-d-p01-sla-4k-c8/4k-c8` |
| sla | [sla/64k-c1](sla/64k-c1/) | 4,5 | 22.1675 | `result/runs/community-official-followup-20261017-d-p45-sla-64k-c1/64k-c1` |
| sla | [sla/64k-c2](sla/64k-c2/) | 4,5 | 26.5937 | `result/runs/community-official-followup-20261017-d-p45-sla-64k-c2/64k-c2` |
| clean_runtime | [clean-runtime/4k-c8](clean-runtime/4k-c8/) | 2,3 | 238.0041 | `result/runs/community-official-followup-20261017-d-p23-clean-runtime-4k-c8/4k-c8` |
| clean_runtime | [clean-runtime/64k-c8](clean-runtime/64k-c8/) | 2,3 | 31.6471 | `result/runs/community-official-followup-20261017-d-p23-clean-runtime-64k-c8/64k-c8` |

## 原始文件与条件

- `random/`：论坛reference协议保留10档原件，主展示8档；4K客户端C64／C128的服务运行数仍为32，仅作排队诊断，原始目录不移动、不删除，也不在本索引展示其性能数字。seed20260912，两轮各C预热、三轮max(8,2C)正式。请求级requests.jsonl保留时间、Token数量、状态、输出SHA和片段；未保存完整输出或每Token IDs／原始SSE，不能声称可以逐Token重放。
- `sharegpt/`：安装版sglang.bench_serving，C1/4/16/32取10月16日已验证候选，C64取10月17日最新复核；均context67584、1000完整预热、3×1000正式，每次64短预热。教程context131072及预热描述差异另见报告。
- `sla/`：4K C2–8、64K C1/2；4K C1复用random/4k-c1，未重复复制。4K C7通过、C8失败；64K C1阶段间不一致，不能称稳定达标或全局最大0。
- `clean-runtime/`：独立新容器的4K／64K C8；与矩阵C8分开，未择优替换。
- `environment/`：两批各卡对的原始服务日志／API、容器命令／身份／依赖，及已有宿主、模型记录；ShareGPT历史服务证据另存各c目录service/。所有PID均为历史记录。
- 两卡TP2、服务运行数32／Graph32；不同卡对同机并行负载保留，不能视为单一隔离实验。
- 固定负载U+FFFD保留，正式／预热分别计数；三轮均值不用于代替逐轮P95 SLA。旧失败与旧数据仍在原result/evidence目录，不随本次整理覆盖。
