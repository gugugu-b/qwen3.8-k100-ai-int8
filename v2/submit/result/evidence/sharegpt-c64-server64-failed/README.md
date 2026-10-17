# 最新submit服务端64并发补测：测量齐全，原文本完整性检查失败

截至2026-10-16 22:45（北京时间）。物理HCU0、1，TP=2，客户端并发64，真实API服务max_running_requests=64、cuda_graph_max_bs=64，Graph/MTP/native通信启用。

**原队列状态为FAILED，未标记PASS。** 三轮正式请求及1000条完整预热全部结束，HTTP/benchmark请求错误数0；原validate_sharegpt.py第26行拒绝generated_texts中的U+FFFD替换字符，收尾退出1。以下为完整原始测量的独立数值复核，不改变原失败结果。

| 指标 | 最新服务64三轮均值 | 用户v1综合报告 | 相对历史报告变化 |
|---|---:|---:|---:|
| 输出吞吐 tok/s | 473.6951 | 417.42 | +13.4816% |
| TTFT median ms | 476.2042 | 544.56 | -12.5525% |
| TTFT P99 ms | 6177.7480 | 6098.01 | +1.3076% |
| TPOT median ms | 128.7106 | 148.43 | -13.2853% |
| TPOT P99 ms | 456.1811 | 496.30 | -8.0836% |

三轮输出吞吐分别为471.1974911336 / 474.6635736484 / 475.2241773802 tok/s，均值473.6950807207 tok/s，观测极差率0.8500587%。正式3000条、完整预热1000条，每次另64短预热。每轮输入331978、输出196501 tokens。

旧6、7卡服务32/Graph32的C64为417.0916731672 tok/s；本次相对该数值+13.5710%。卡对、运行时间、服务及Graph上限均不同，这只是历史描述，不能归因于单一参数或补丁，也没有重测v1。用户指定v1报告417.42始终作为主要分母。

## 原失败与边界

- 完整预热及每轮正式输出均在请求索引24、663各有一个末尾U+FFFD，共8条响应含该字符；正式轮共6条。原raw、failed-state、validator source/log及全部SHA完整保留。
- 已确认失败发生于客户端收尾文本完整性校验；服务器日志无新VMFault/Watchdog/OOM/Traceback，所有请求成功。U+FFFD的生成来源仍UNPROVEN，原始token IDs和逐条SSE text_chunks未保存，不推断是算子、模型数值或传输解码故障。
- 从停止的自有容器只读复制了安装版bench_serving.py，SHA f2206dcda74932e1b148cdf7b5ef07443bb5d9e0f2729b7a67d602182e3aa050；其对应客户端使用严格UTF-8解码并拼接choices text。这项源码事实不能恢复未保存的逐块响应或最终token IDs。
- 原strict reviewer及成功final归档入口未运行，因为原RUN未finished/COMPLETE；没有伪造stage、validated_summary、final-state或成功marker，没有改原断言或用去除字符后的输出替换raw。
- 新独立CPU审计只复核原完整性能测量并显式返回MEASUREMENTS_COMPLETE_ORIGINAL_TEXT_INTEGRITY_FAILED；不是放宽原验收门禁。没有新GPU重跑或质量评测。
- TTFT P99及吞吐可精确重算。TPOT/ITL/E2E为原benchmark已保存统计的逐轮独立汇总；text_chunks与逐请求最终latency缺失，不能宣称精确重建这些逐请求尾指标。轮P99均值不是合并总体P99；极差不是统计噪声上界。
- 精度/GSM8K NOT_RUN、实际模型Graph门禁REVIEW_REQUIRED；默认site和stock DAT不变，未晋级/冻结/FROZEN_PASS。

## 证据与释放

- 原artifact manifest38项SHA `9c4c7b1a7514d84b3586eaefd50b3255af7f31a55dacbe9a4bf7564e87d7d637`；control manifest32项SHA `9da3465b597acb0bbd4d88b7c9da153388d5846d5634648b2e0a8d3188545942`。
- 远端一次性失败归档：`/data/wangzy/submit1009/result/submit-current-c64-server64-pair01-20261016-a-failure-archive-20261016-a/`；52份原始/控制/源码/CPU审核文件，合计14,775,307字节，最大3,076,883字节，均<16MiB，已按inventory显式逐文件同步、全部size/SHA一致。
- 远端及本地CPU measurement-report.json字节相同，SHA `d5ddcc2e70e2a6a91c0c3d43f5cac37dfd2701876566c2086e3680545e9f04a7`；完整逐轮测量、所有已保存延迟字段、失败索引、源SHA与比较在该JSON中。
- 自有容器22:32:03停止；原queue3224259和已记录server/TP/bench进程全部退出，gpu01锁FREE。2026-10-16 22:45实核物理0、1利用率0%、显存各2MiB常驻。详见pair01-final-release-proof.json。
- 旧七景、旧服务32 C64、原失败RUN及原strict脚本均不覆盖，不操作卡2–7或其它容器。

旧服务32参考来源：`../results-submit-current-sharegpt-pair67-final-20261016-a/local-independent-review/performance-summary.json`，SHA `421f90f4aa597f7e2e9ddc480f838de9ccf013bf566960a14433c09b344bb382`，JSON pointer `/scenarios/64/mean_output_tps`。
