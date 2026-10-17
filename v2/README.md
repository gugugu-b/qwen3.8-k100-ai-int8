# v2：K100-AI／SGLang／TP2 社区提交

更新日期：2026-10-17。完整提交入口为 [submit/README.md](submit/README.md)，按 **环境准备 → 依赖安装 → 模型启动 → Benchmark → 得到结果** 执行；默认使用物理卡0、1。源码、完整运行文件及分项Diff见 [补丁说明](submit/patch/README.md)。

[六类实测环境声明](submit/docs/实测环境声明.md)已填写硬件拓扑与存储、宿主／容器软件版本、框架依赖、18个权重分片SHA256前8位、完整启动命令和采集协议，并放在仓库首页与实测报告第一段。

## 从GitHub仓库准备运行库

模型权重和ShareGPT按提交README准备。rocBLAS使用仓库已有的 `v1/submit1005/config/library_gpu4_int8tuned_stock_clean.tar.gz`，其全部162文件与v2清单逐项SHA一致。无需再次下载或重复存储展开后的827,205,083字节库：

```bash
# 在仓库根目录执行；不使用GPU，不修改已封存源码或证据。
python3 -I v2/prepare_library.py
cd v2/submit
bash scripts/run_review01.sh check
```

随后从 [提交README第1节](submit/README.md#1-环境准备)开始：拉取／载入固定镜像，设置当前包路径、权重与数据路径，核空闲卡和资产，创建容器，再检查依赖、启动及Benchmark。完整库目录准备好后默认LIBRARY_DIR即可使用；准备工具遇到已有库时只核验，拒绝覆盖不匹配内容。

## 一张表查看全部性能

**[完整性能总表](submit/docs/性能总表.md)** 汇总38组v2候选测量、114正式轮。4K覆盖C1–8、16、32、64、128；64K覆盖C1、2、8、16、32；ShareGPT覆盖C1、4、16、32、64。相同并发下的不同协议、卡对、服务／Graph上限、复测和原失败记录分别列出，不再只选代表值展示。

仓库首页、提交README§5和性能报告§4使用同一张总表，含每轮TPS、均值、TTFT／TPOT mean与尾延迟、SLA通过轮数、U+FFFD标记及原始数据链接。旧服务32高客户端并发明确作排队诊断；新C128实际采样峰值76；ShareGPT服务64虽测得473.6951 TPS但原文本门禁FAIL不变。原始文件和历史判定保持原样，6组v1对照不计入v2总表。

- [当前性能报告 §4](submit/docs/性能优化报告.md#4-性能结果)：预期结果、完整逐轮数据及适用范围。
- [原始矩阵／SLA／ShareGPT归档](submit/logs/v2_benchmark_raw/final/README.md)：26场景、78正式轮，附来源清单与CPU复核入口。
- [服务64／128独立复测](submit/docs/服务端64与128并发复测-20261017.md)：新增6正式轮、591个原始源文件、运行并发采样和KV预算。
- [GSM8K报告](submit/docs/GSM8K精度验证-20261017.md)、[Demo与视频](submit/docs/Demo说明.md)、[卡0、1运行验证](submit/result/evidence/readme-review01-20261017/runtime-review.json)。
- [材料状态](submit/docs/提交材料状态-20261017.md)：其余三项精度、最终官方协议、64K SLA稳定性与最终组合验证仍待完成，当前不标记全部赛事准入通过。

`submit/manifest/package-sha256.json`封存当前源码、报告及证据；仓库保留完整原始日志及历史运行目录。模型与ShareGPT大文件在包外，完整库由上述已校验压缩包准备。`tools/seal_manifest.py`仅供有意编辑材料后重新封存，正常复现不运行它。
