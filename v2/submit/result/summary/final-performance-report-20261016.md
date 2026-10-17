# 最新submit七场景性能报告

截至 2026-10-16 16:42 Asia/Shanghai。仅性能，七场景均有三轮完整候选数据。4–7卡已全部释放，两个队列退出、自有容器停止、锁FREE。

以下变化统一相对用户指定v1报告固定分母，属于历史参照，不是同期因果验收。

| 场景 | 测量日期 | 报告v1 tok/s | 最新submit三轮均值 tok/s | 历史变化 | 三轮极差/均值 |
|---|---|---:|---:|---:|---:|
| 4K/1K C8 | 2026-10-10 | 206.84 | 238.2423 | +15.1819% | 3.6755% |
| 64K/1K C8 | 2026-10-16 | 30.39 | 31.3849 | +3.2736% | 1.2975% |
| ShareGPT C1 | 2026-10-16 | 43.13 | 54.5359 | +26.4453% | 1.1702% |
| ShareGPT C4 | 2026-10-16 | 133.37 | 144.5965 | +8.4176% | 0.6423% |
| ShareGPT C16 | 2026-10-16 | 289.32 | 326.2182 | +12.7534% | 0.3312% |
| ShareGPT C32 | 2026-10-16 | 338.41 | 419.7877 | +24.0471% | 0.1283% |
| ShareGPT C64 | 2026-10-16 | 417.42 | 417.0917 | -0.0787% | 1.3084% |

ShareGPT延迟为每轮对应统计的三轮均值，单位ms；括号为相对历史报告变化。

| 场景 | TTFT median | TTFT P99 | TPOT median | TPOT P99 |
|---|---:|---:|---:|---:|
| ShareGPT C1 | 238.0083 (-2.0260%) | 944.5735 (+3.7343%) | 17.1458 (-22.4171%) | 31.4189 (-19.7883%) |
| ShareGPT C4 | 244.2794 (-3.9820%) | 945.6639 (-20.0237%) | 25.5924 (-8.0731%) | 61.9548 (-5.8724%) |
| ShareGPT C16 | 258.5805 (-8.3860%) | 981.8911 (-34.0052%) | 45.9779 (-9.8649%) | 118.3995 (-28.5760%) |
| ShareGPT C32 | 348.0811 (-8.4527%) | 2734.5376 (-54.9536%) | 72.2717 (-14.5623%) | 246.4288 (-15.0568%) |
| ShareGPT C64 | 15151.0802 (+2682.2609%) | 21384.0426 (+250.6725%) | 72.7562 (-50.9828%) | 252.3542 (-49.1529%) |

C64 TTFT P99相对报告变化 **+250.6725%**；客户端C64仍保留服务max_running_requests=32和Graph最大batch32，未测试实际服务64并发。

C1 TTFT P99相对报告变化 **+3.7343%**。

完整逐轮TPS、全部已保存延迟字段及各自单位、三轮均值、源文件SHA和历史分母见同名JSON。固定4K/64K的TTFT与E2E用秒，TPOT用毫秒；ShareGPT全部延迟用毫秒。

资源释放采证：2026-10-16T08:40:34.975518+00:00，HCU4/5/6/7使用率均0.0%，各2MiB驱动常驻。

- 历史报告比较不是同期同卡对因果A/B，不使用本次已停止的v1/return分母，也不作为20%正式验收。
- 4K数据为2026-10-10已审核交付；64K和所有ShareGPT数据为2026-10-16。C32复用已完成候选，不重复压测。
- 历史ShareGPT每档1000条+64短预热；当前候选每档1000完整预热+三轮1000正式，每次另64短预热，条件不同。
- 64K/C32曾与另一卡对任务重叠，C1和C16/C64/C4也有独立卡对重叠；仅保留实际条件，不声称隔离显著性。
- 所有延迟统计逐轮保留并计算对应统计的三轮算术均值；P99均值不是合并所有请求的总体P99，std均值也不是总体std。
- MTP原始JSON未保存text_chunks和逐请求最终latency，TPOT/ITL/E2E保留原benchmark统计，不能声称按请求精确重建；缺失P95不补算冒充。
- C64客户端并发64，但服务max_running_requests=32、Graph最大batch=32；这不是服务实际64并发能力验证。
- C64相对历史报告TTFT P99明显增加，保留该回退；不能因吞吐接近而隐藏延迟变化。
- 精度/GSM8K未验证，不继承历史报告精度结论；未晋级默认补丁，未冻结，无FROZEN_PASS。

来源与校验：

- `files/review_4k.json` SHA256 `992df2f9d0f300f625fd72263754874630ce5716fb392c0cfb2a04b0441258bd`；原分类路径 `02-测试与结果/results-4k-v2-delivery-20261010-b/local-integrity-review/review.json`。
- `files/review_64k.json` SHA256 `d2478cf98f56373cbff16db5a7a0000d73ae12a13dcf86ef46717c541f329d9f`；原分类路径 `02-测试与结果/results-64k-v2-validation-20261016-a/local-integrity-review-b/review.json`。
- `files/review_c32.json` SHA256 `7104baebc1429811ae55ce18a0b45727ef8f7efdb893e14844e76b633ce49660`；原分类路径 `02-测试与结果/results-sharegpt-v2-validation-20261016-c32-a/local-independent-review/comparison.json`。
- `files/4k_validated.json` SHA256 `b245a7407f58649758676dcc6813f8fda772ad45a0fe37532108f1c411d060e4`；原分类路径 `02-测试与结果/results-4k-v2-delivery-20261010-b/submit1009/result/delivery-native-ar-4k1k-20261010-b-native-bf16-ar/4k/validated_summary.json`。
- `files/4k_round_01.json` SHA256 `b9fb075b84c2906ab3ad440ef962307402857cf42ab12940d481b540b4535403`；原分类路径 `02-测试与结果/results-4k-v2-delivery-20261010-b/submit1009/result/delivery-native-ar-4k1k-20261010-b-native-bf16-ar/4k/round-01/summary.json`。
- `files/4k_round_02.json` SHA256 `cf1f0cb14af28791aca266ef3a5a9022715348adcae5931ae6d54ac7795a034c`；原分类路径 `02-测试与结果/results-4k-v2-delivery-20261010-b/submit1009/result/delivery-native-ar-4k1k-20261010-b-native-bf16-ar/4k/round-02/summary.json`。
- `files/4k_round_03.json` SHA256 `ee3bfd49d64f3986632fba647cb9c1563a81666c673e095502893d1f29d67624`；原分类路径 `02-测试与结果/results-4k-v2-delivery-20261010-b/submit1009/result/delivery-native-ar-4k1k-20261010-b-native-bf16-ar/4k/round-03/summary.json`。
- `files/64k_validated.json` SHA256 `3d8902cc90a5f21488c033a6a589e31c35d92dc58741752e53015b630d3a6527`；原分类路径 `02-测试与结果/results-64k-v2-validation-20261016-a/submit1009/result/perf-64k-20261016-a/variants/candidate/64k/validated_summary.json`。
- `files/64k_round_01.json` SHA256 `2d9edaff0d813dabfa4bc49a8ae985fd64a4343eeb81ac4d7c9e72d48b8b48d8`；原分类路径 `02-测试与结果/results-64k-v2-validation-20261016-a/submit1009/result/perf-64k-20261016-a/variants/candidate/64k/round-01/summary.json`。
- `files/64k_round_02.json` SHA256 `3ebfc96ce8d8244df9c64bcc03905bab5110358d8c96178c088095a6059b3b84`；原分类路径 `02-测试与结果/results-64k-v2-validation-20261016-a/submit1009/result/perf-64k-20261016-a/variants/candidate/64k/round-02/summary.json`。
- `files/64k_round_03.json` SHA256 `a6d54d86e5410d0128a4da9a5bdec9f178c6351024089772c72c7161829a8ba7`；原分类路径 `02-测试与结果/results-64k-v2-validation-20261016-a/submit1009/result/perf-64k-20261016-a/variants/candidate/64k/round-03/summary.json`。
- `files/c32_validated.json` SHA256 `cb38bf0019a75011adbc0c8a17a7de2104afb507300a83ccb1512900ea8998b8`；原分类路径 `02-测试与结果/results-sharegpt-v2-validation-20261016-c32-a/submit1009/experiments/performance-validation-20261016/result/perf-sharegpt-20261016-a-c32-candidate/sharegpt-c32/validated_summary.json`。
- `files/c32_round_01.jsonl` SHA256 `2b4e5cc8dd7ec0bcccf1b3ef11b9f7ab394b8f8b9a4216ef93c3eb8c029fc1f4`；原分类路径 `02-测试与结果/results-sharegpt-v2-validation-20261016-c32-a/submit1009/experiments/performance-validation-20261016/result/perf-sharegpt-20261016-a-c32-candidate/sharegpt-c32/round-01.jsonl`。
- `files/c32_round_02.jsonl` SHA256 `62b7037e34edf305cc7a125c2c09e4d568150d5a04cfbc3adf7b6e305e99ff51`；原分类路径 `02-测试与结果/results-sharegpt-v2-validation-20261016-c32-a/submit1009/experiments/performance-validation-20261016/result/perf-sharegpt-20261016-a-c32-candidate/sharegpt-c32/round-02.jsonl`。
- `files/c32_round_03.jsonl` SHA256 `1a0509c4f37ac618c1b2d00b9c8656646bdbf88905618bec1a5b3d5e0c4a337d`；原分类路径 `02-测试与结果/results-sharegpt-v2-validation-20261016-c32-a/submit1009/experiments/performance-validation-20261016/result/perf-sharegpt-20261016-a-c32-candidate/sharegpt-c32/round-03.jsonl`。
- `files/existing_scene_map.json` SHA256 `ff38cacd4afff2809fa59ec95ca3d8847485e9add759aa36cc02d36f550d3ba5`；原分类路径 `02-测试与结果/results-submit-current-performance-summary-20261016-a/final-existing-scenes-source-map.json`。
- `files/final45.json` SHA256 `53c7aa9de50d63c8f34caa5d411742f109d29ae53565870d054045eb696fd36c`；原分类路径 `02-测试与结果/results-submit-current-sharegpt-c1-pair45-final-20261016-a/local-independent-review/performance-summary.json`。
- `files/final67.json` SHA256 `421f90f4aa597f7e2e9ddc480f838de9ccf013bf566960a14433c09b344bb382`；原分类路径 `02-测试与结果/results-submit-current-sharegpt-pair67-final-20261016-a/local-independent-review/performance-summary.json`。
- `files/reference.json` SHA256 `54c5ff586396dd90760eeb3fc3686f741927a5ef065294148ce293919ab4ef2d`；原分类路径 `02-测试与结果/performance-validation-20261016/v1-report-reference.json`。
- `files/reference_report.md` SHA256 `b36db6ec60fcac45d710fada2bdf74b1b4655ef631567e6c34fc4dc4a20e417d`；原分类路径 `02-测试与结果/performance-validation-20261016/v1-report-source.md`。
- `files/release.json` SHA256 `a469301507f195c945f719bae7838a19cf77a348c4d51aa96a69997ea19d3077`；原分类路径 `02-测试与结果/results-submit-current-sharegpt-c1-pair45-final-20261016-a/all-card-release-proof.json`。
- `files/c1_round_01.jsonl` SHA256 `eeca16141b568a130b2d6c5c0e237119a455aa205bfd30276802fa4f0061557c`；原分类路径 `02-测试与结果/results-submit-current-sharegpt-c1-pair45-final-20261016-a/submit1009/experiments/performance-validation-20261016/result/submit-current-c1-pair45-20261016-a-c1-candidate/sharegpt-c1/round-01.jsonl`。
- `files/c1_round_02.jsonl` SHA256 `8a87e8a50b5f5b2759fe8e5d9034693a735d37226f098666844ac1ed094e6510`；原分类路径 `02-测试与结果/results-submit-current-sharegpt-c1-pair45-final-20261016-a/submit1009/experiments/performance-validation-20261016/result/submit-current-c1-pair45-20261016-a-c1-candidate/sharegpt-c1/round-02.jsonl`。
- `files/c1_round_03.jsonl` SHA256 `8884d1608cb4dfb34961f15b322a07da38b0413e4af28b11914f88a4c3f0f23b`；原分类路径 `02-测试与结果/results-submit-current-sharegpt-c1-pair45-final-20261016-a/submit1009/experiments/performance-validation-20261016/result/submit-current-c1-pair45-20261016-a-c1-candidate/sharegpt-c1/round-03.jsonl`。
- `files/c4_round_01.jsonl` SHA256 `c4ff31902a4557387ad6e1a3bb7a290c8943226f06bfcb8358d976a3131e22ba`；原分类路径 `02-测试与结果/results-submit-current-sharegpt-pair67-final-20261016-a/submit1009/experiments/performance-validation-20261016/result/submit-current-sharegpt-pair67-20261016-a-c4-candidate/sharegpt-c4/round-01.jsonl`。
- `files/c4_round_02.jsonl` SHA256 `c053091f43c8f20d1f01db4060c5f1199aedb94f616ed29c51ff57484b48168d`；原分类路径 `02-测试与结果/results-submit-current-sharegpt-pair67-final-20261016-a/submit1009/experiments/performance-validation-20261016/result/submit-current-sharegpt-pair67-20261016-a-c4-candidate/sharegpt-c4/round-02.jsonl`。
- `files/c4_round_03.jsonl` SHA256 `3026e6057759c9289bd5b0ce53a4414e7ae5a9b347949458a0cdd53d86cae9cf`；原分类路径 `02-测试与结果/results-submit-current-sharegpt-pair67-final-20261016-a/submit1009/experiments/performance-validation-20261016/result/submit-current-sharegpt-pair67-20261016-a-c4-candidate/sharegpt-c4/round-03.jsonl`。
- `files/c16_round_01.jsonl` SHA256 `9d7950dced19b498c7be59ef73c0346753b72398f15117a925d808eac7a1684d`；原分类路径 `02-测试与结果/results-submit-current-sharegpt-pair67-final-20261016-a/submit1009/experiments/performance-validation-20261016/result/submit-current-sharegpt-pair67-20261016-a-c16-candidate/sharegpt-c16/round-01.jsonl`。
- `files/c16_round_02.jsonl` SHA256 `f26c17ae4d44678dd89c3d8424445e9cf02d7d645e33bcb0431f3a5b5e633d17`；原分类路径 `02-测试与结果/results-submit-current-sharegpt-pair67-final-20261016-a/submit1009/experiments/performance-validation-20261016/result/submit-current-sharegpt-pair67-20261016-a-c16-candidate/sharegpt-c16/round-02.jsonl`。
- `files/c16_round_03.jsonl` SHA256 `890e321878f194622b32a3c7fe324c425c4e55e51502c5284ed86f12e9108289`；原分类路径 `02-测试与结果/results-submit-current-sharegpt-pair67-final-20261016-a/submit1009/experiments/performance-validation-20261016/result/submit-current-sharegpt-pair67-20261016-a-c16-candidate/sharegpt-c16/round-03.jsonl`。
- `files/c64_round_01.jsonl` SHA256 `bd3f22075c29840455dd732625ccfa693a284b4c2784fa105c98688072d25201`；原分类路径 `02-测试与结果/results-submit-current-sharegpt-pair67-final-20261016-a/submit1009/experiments/performance-validation-20261016/result/submit-current-sharegpt-pair67-20261016-a-c64-candidate/sharegpt-c64/round-01.jsonl`。
- `files/c64_round_02.jsonl` SHA256 `a29b11f6907cfbd7a431ca8943550fff09771815753ee5ca00d8594e197a4c60`；原分类路径 `02-测试与结果/results-submit-current-sharegpt-pair67-final-20261016-a/submit1009/experiments/performance-validation-20261016/result/submit-current-sharegpt-pair67-20261016-a-c64-candidate/sharegpt-c64/round-02.jsonl`。
- `files/c64_round_03.jsonl` SHA256 `578dcd20ac977db95998d149f54cf59fc4d30ce4a2a4d0f9e617c90ce3e14770`；原分类路径 `02-测试与结果/results-submit-current-sharegpt-pair67-final-20261016-a/submit1009/experiments/performance-validation-20261016/result/submit-current-sharegpt-pair67-20261016-a-c64-candidate/sharegpt-c64/round-03.jsonl`。
