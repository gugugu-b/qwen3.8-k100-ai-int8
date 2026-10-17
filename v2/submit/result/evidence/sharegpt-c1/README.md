# 最新submit ShareGPT C1最终性能验证（2026-10-16）

**三轮平均54.5359 tok/s，相对用户v1综合评测报告43.13提升26.4453%；4、5卡服务16:38:50停止，4–7全部释放。**

物理HCU4、5/TP2，只测最新submit，原candidate45仅将物理卡对断言由6,7适配4,5，算术/路由保持；不能称整个补丁SHA与67相同。默认v1和stock DAT不动，不再测v1、未验证精度。3轮3000正式+1000完整预热零错误，另4次各64短热，原seed20260912/context67584/Graph/MTP/native和服务上限32/Graph32保持。

| 正式轮次 | 输出吞吐 tok/s | 请求 | 错误 |
|---|---:|---:|---:|
| 1 | 54.757796 | 1000 | 0 |
| 2 | 54.730236 | 1000 | 0 |
| 3 | 54.119596 | 1000 | 0 |
| 均值 | 54.535876 | 3000 | 0 |

极差率1.1702390%只为观测波动，不是统计噪声上界。变化=(54.5358758691/43.13−1)×100=+26.4453417%，只作用户指定历史报告参考，非同期因果A/B或正式20%验收。

| 延迟指标 | v1报告 ms | 最新三轮百分位均值 ms | 变化 |
|---|---:|---:|---:|
| TTFT median | 242.93 | 238.0083 | −2.0260% |
| TTFT P99 | 910.57 | 944.5735 | +3.7343% |
| TPOT median | 22.10 | 17.1458 | −22.4171% |
| TPOT P99 | 39.17 | 31.4189 | −19.7883% |

吞吐和TTFT P99可精确重算；原MTP JSON缺text_chunks/逐请求最终latency，ITL/TPOT/E2E保留安装benchmark统计并独立汇总，不能冒称全部逐请求精确重建。轮P99均值不是全部请求合并P99。C1 TTFT P99小幅历史回退保留，C64仍有TTFT P99回退250.67%、客户端64/服务上限32/Graph32条件，不声明全部尾延迟或实际server64配置通过。精度NOT_RUN、实际Graph最终门禁REVIEW_REQUIRED，默认未晋级/未FROZEN_PASS。

队列16:39:00正常结束；host1621554实际不存在、own45容器停止、gpu45锁FREE。16:40:34实核两own20261016容器均停止、全部相关queue/TP/server/bench PID退出、45/67锁FREE；物理4、5、6、7利用率各0.0%，VRAM各2MiB驱动常驻，没有本任务占卡。资源证据all-card-release-proof.json，未触碰0–3或他人容器。

CPU final helper --pair45只执行一次，严格冻结40项artifact和41项control及其并集/外部state/job/own控制。62文件17,979,775B，最大3,914,979B，全<16MiB，显式逐文件同步并核全部SHA/size。producer review-final、OUT final summary、远端独立、本地独立四份完整summary逐字节相同，complete=true；strict reviewer不加allow-partial，-B没有生成源pycache，原数据/旧快照不覆盖。

- 最终summary SHA：53c7aa9de50d63c8f34caa5d411742f109d29ae53565870d054045eb696fd36c
- artifact SHA：b81b791f7f5f954ca87878756fb271543268803030a93e0610e5b32e7a86e499
- control SHA：9ace9b24369ba214d6bc47c67fe9eeb5bde70ebe19b71d6736eb96bdb8c0ab44
- 远端完整快照：/data/wangzy/submit1009/result/submit-current-c1-pair45-20261016-a-final-postcheck/
- 本地完整原始镜像：submit1009/
- 本地严格独立审核：local-independent-review/performance-summary.json

最终七场景报告见../results-submit-current-performance-summary-20261016-a/final-performance-report-20261016.md及JSON。所有本轮性能与归档/四文档完成后关闭本监控；本轮结束不以精度或20%PASS为前提。
