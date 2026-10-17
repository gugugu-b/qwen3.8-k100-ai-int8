# 社区v2缺档与后续执行结果

生成时间：2026-10-16T23:37:31.320434+00:00（UTC时间戳）。来源为官方教程参考客户端；SLA阈值按用户选择的本地赛事文档，P95取自教程。尚非主办方最终成绩。

| 类型 | 卡对 | 场景 | C | 三轮TPS均值 | 完整性 |
|---|---|---|---:|---:|---|
| matrix | 0,1 | 4k | 1 | 57.3612 | PERFORMANCE_MEASUREMENTS_PASS |
| matrix | 0,1 | 4k | 8 | 237.0503 | PERFORMANCE_MEASUREMENTS_PASS |
| matrix | 0,1 | 4k | 16 | 307.6920 | PERFORMANCE_MEASUREMENTS_COMPLETE_TEXT_FLAGGED |
| matrix | 2,3 | 64k | 1 | 22.4353 | PERFORMANCE_MEASUREMENTS_PASS |
| matrix | 2,3 | 64k | 8 | 31.6825 | PERFORMANCE_MEASUREMENTS_PASS |
| matrix | 2,3 | 64k | 16 | 32.5044 | PERFORMANCE_MEASUREMENTS_PASS |
| matrix | 4,5 | 4k | 32 | 375.7735 | PERFORMANCE_MEASUREMENTS_COMPLETE_TEXT_FLAGGED |
| matrix | 4,5 | 4k | 64 | 379.4128 | PERFORMANCE_MEASUREMENTS_COMPLETE_TEXT_FLAGGED |
| matrix | 4,5 | 4k | 128 | 380.6238 | PERFORMANCE_MEASUREMENTS_COMPLETE_TEXT_FLAGGED |
| matrix | 6,7 | 64k | 32 | 32.7705 | PERFORMANCE_MEASUREMENTS_PASS |
| sla-4k | 0,1 | 4k | 1 | 57.3612 | PERFORMANCE_MEASUREMENTS_PASS |
| sla-4k | 0,1 | 4k | 2 | 99.2055 | PERFORMANCE_MEASUREMENTS_PASS |
| sla-4k | 0,1 | 4k | 4 | 143.5443 | PERFORMANCE_MEASUREMENTS_COMPLETE_TEXT_FLAGGED |
| sla-4k | 0,1 | 4k | 8 | 235.9626 | PERFORMANCE_MEASUREMENTS_PASS |
| sla-4k | 0,1 | 4k | 3 | 109.0746 | PERFORMANCE_MEASUREMENTS_COMPLETE_TEXT_FLAGGED |
| sla-4k | 0,1 | 4k | 5 | 167.5628 | PERFORMANCE_MEASUREMENTS_COMPLETE_TEXT_FLAGGED |
| sla-4k | 0,1 | 4k | 6 | 185.4283 | PERFORMANCE_MEASUREMENTS_PASS |
| sla-4k | 0,1 | 4k | 7 | 204.3980 | PERFORMANCE_MEASUREMENTS_PASS |
| sla-64k | 4,5 | 64k | 1 | 22.1675 | PERFORMANCE_MEASUREMENTS_PASS |
| sla-64k | 4,5 | 64k | 2 | 26.5937 | PERFORMANCE_MEASUREMENTS_PASS |
| demo-and-clean-runtime | 2,3 | 4k | 8 | 238.0041 | PERFORMANCE_MEASUREMENTS_PASS |
| demo-and-clean-runtime | 2,3 | 64k | 8 | 31.6471 | PERFORMANCE_MEASUREMENTS_PASS |
| sharegpt-c64-recheck | 6,7 | sharegpt | 64 | 416.8612 | PASS |

用户已追加一个独立TP2实例并行GSM8K，结果见community-gsm8k-20261017-e；其他三项精度后置。SLA按本地赛事阈值、教程P95给出观察边界；教程更严阈值只作附加对照。
完整命令、服务参数、原始请求、预热及正式轮见同名JSON列出的各路径。新目录干净容器复现保留并行主机条件；不作隔离显著性声明。
Demo只使用实际API时间流记录，终端回放视频仍需据此渲染；不将日志或截图冒充已交视频。
实际模型Graph/compiler关联、最终模型资产准入与完整四项精度仍未完结；不冻结或晋级默认。
