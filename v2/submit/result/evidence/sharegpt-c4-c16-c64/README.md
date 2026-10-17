# 最新submit ShareGPT C16/C64/C4最终性能归档（2026-10-16）

**C4三轮144.5965 tok/s，相对用户v1综合报告133.37提升8.4176%；6、7卡整条队列已结束并释放。**

只测最新submit候选，物理HCU6、7/TP2，没有新v1测量或精度测试。C4于15:31:43结束，队列15:31:53完成，原host1621555退出、专有容器停止、gpu67锁FREE。15:47:39实读物理6、7利用率均0.0%、各6MiB驱动常驻，原服务/TP/benchmark PID均不存在。4、5卡的C1原队列继续第三轮，不能据67完成停止45或删除全部监控。

| 场景 | 正式三轮输出 tok/s | 均值 tok/s | v1报告 tok/s | 历史参考变化 |
|---|---|---:|---:|---:|
| C16 | 325.811686 / 326.892252 / 325.950773 | 326.218237 | 289.32 | +12.7534% |
| C64 | 420.458438 / 415.815355 / 415.001226 | 417.091673 | 417.42 | −0.0787% |
| C4 | 145.080622 / 144.557031 / 144.151948 | 144.596534 | 133.37 | +8.4176% |

三个场景共9000正式及3000完整预热零错误，另12次各64短热；原seed20260912、context67584、Graph/MTP与API native/disable=false保留。C4极差率0.6422521%，非统计噪声上界。吞吐变化公式为(最新三轮均值/v1报告−1)×100，非旧同期v1分母。

| C4延迟指标 | v1报告 ms | 最新三轮百分位均值 ms | 变化 |
|---|---:|---:|---:|
| TTFT median | 254.41 | 244.2794 | −3.9820% |
| TTFT P99 | 1182.43 | 945.6639 | −20.0237% |
| TPOT median | 27.84 | 25.5924 | −8.0731% |
| TPOT P99 | 65.82 | 61.9548 | −5.8724% |

C64实测仍为客户端并发64、服务max_running_requests32、cuda_graph_max_bs32；不是服务端上限64的配置。其TTFT P99从报告6098.01ms变21384.04ms（+250.67%）的回退完整保留，不能宣称全部场景尾延迟合格。根因未证明；没有在本轮新增服务64参数测试。历史报告预热/日期/并行条件与当前不同，非同期同卡对A/B或纯补丁因果验收。MTP JSON缺text_chunks和逐请求最终latency，吞吐/TTFT P99精确复算，其余保留安装benchmark统计独立汇总；轮P99均值不是全部请求合并P99。精度NOT_RUN、actualGraph最终门禁REVIEW_REQUIRED、默认v1及stock DAT保持，未冻结。

唯一CPU final helper仅执行一次，严格检查真实退出/own停止/FREE锁、原始manifest精确覆盖和全部SHA后，冻结70项artifact与59项control的并集、两个manifest、外部state/job及控制证据。共98文件45,156,776字节，最大3,152,062字节，全<16MiB；按inventory显式同步。两份manifest本地/远端全部SHA核验通过；OUT最终摘要、producer review-final、远端独立、本地独立四份完整summary字节相同，complete=true。未重跑C16/C64原快照或已完成GPU测试。

- 原始manifest SHA：299018c8c569f917ac2ff2523ad51a2129892b2bd861921658037f28e7245930
- control SHA：4794805da2c75895144f468513d02b9105a80a1edd20148516964ba0c8d03ca8
- 最终summary SHA：421f90f4aa597f7e2e9ddc480f838de9ccf013bf566960a14433c09b344bb382
- 远端完整快照：/data/wangzy/submit1009/result/submit-current-sharegpt-pair67-20261016-a-final-postcheck/
- 本地完整镜像：submit1009/
- 本地严格独立审核：local-independent-review/performance-summary.json
- 单独实卡释放证明：pair67-resource-release-proof.json

现有final归档包含C4全部原始与最终控制，未另建重复的C4 scene POST。旧C16/C64快照保持原样。只有剩余C1完成、其最终审核/归档及四文档齐备、4–7本任务占用均释放后才删除监控。
