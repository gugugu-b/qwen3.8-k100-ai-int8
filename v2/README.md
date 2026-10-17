# v2：K100-AI／SGLang／TP2 社区提交

更新日期：2026-10-17。完整提交入口为 [submit/README.md](submit/README.md)，按 **环境准备 → 依赖安装 → 模型启动 → Benchmark → 得到结果** 执行；默认使用物理卡0、1。源码、完整运行文件及分项Diff见 [补丁说明](submit/patch/README.md)。

## 从GitHub仓库准备运行库

模型权重和ShareGPT按提交README准备。rocBLAS使用仓库已有的 `v1/submit1005/config/library_gpu4_int8tuned_stock_clean.tar.gz`，其全部162文件与v2清单逐项SHA一致。无需再次下载或重复存储展开后的827,205,083字节库：

```bash
# 在仓库根目录执行；不使用GPU，不修改已封存源码或证据。
python3 -I v2/prepare_library.py
cd v2/submit
bash scripts/run_review01.sh check
```

随后从 [提交README第1节](submit/README.md#1-环境准备)开始：拉取／载入固定镜像，设置当前包路径、权重与数据路径，核空闲卡和资产，创建容器，再检查依赖、启动及Benchmark。完整库目录准备好后默认LIBRARY_DIR即可使用；准备工具遇到已有库时只核验，拒绝覆盖不匹配内容。

## 本版结果与证据

| 项目 | 当前结果 |
|---|---|
| 随机负载主矩阵 | 服务32／Graph32，4K与64K各C1／8／16／32，共8场景、24正式轮 |
| ShareGPT | C1／4／16／32／64，各3正式轮；C32为419.7877 TPS |
| 新增C64容量配置 | 服务64／Graph64，398.9657 TPS；正式期间5秒采样运行峰值64 |
| 新增C128容量配置 | 服务128／Graph128，378.3503 TPS；正式采样峰值76，保留为排队诊断，实际128并发未验证 |
| SLA | 本地阈值逐轮P95：4K已测C7通过、C8失败；64K C1跨阶段不一致，稳定性待解决 |
| GSM8K | 本地冻结四样例全量1319题，1288正确，97.6497%；最终官方协议一致性待核对 |
| 物理卡0、1复现 | 新容器依赖、31项模型与数据SHA、模型health和真实API通过，两个rank的AR guard eligible；容器已停止、锁已释放 |

所有TPS为三正式轮输出TPS的算术均值，排除预热。不同服务上限、Graph和并行条件分别记录；随机输出U+FFFD标记保留，性能完整性通过不代表精度或赛事准入。

- [当前性能报告 §4](submit/docs/性能优化报告.md#4-性能结果)：预期结果、完整逐轮数据及适用范围。
- [原始矩阵／SLA／ShareGPT归档](submit/logs/v2_benchmark_raw/final/README.md)：26场景、78正式轮，附来源清单与CPU复核入口。
- [服务64／128独立复测](submit/docs/服务端64与128并发复测-20261017.md)：新增6正式轮、591个原始源文件、运行并发采样和KV预算。
- [GSM8K报告](submit/docs/GSM8K精度验证-20261017.md)、[Demo与视频](submit/docs/Demo说明.md)、[卡0、1运行验证](submit/result/evidence/readme-review01-20261017/runtime-review.json)。
- [材料状态](submit/docs/提交材料状态-20261017.md)：其余三项精度、最终官方协议、64K SLA稳定性与最终组合验证仍待完成，当前不标记全部赛事准入通过。

`submit/manifest/package-sha256.json`封存6192个文件；仓库保留完整原始日志及历史运行目录。模型与ShareGPT大文件在包外，完整库由上述已校验压缩包准备。`tools/seal_manifest.py`仅供有意编辑材料后重新封存，正常复现不运行它。
