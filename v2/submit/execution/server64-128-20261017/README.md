# 4K/1K 服务端64／128容量实测

用户要求重测此前仅客户端加压的两组4K/1K数据。本批分别使服务max_running_requests与Graph上限匹配客户端64／128，保留原候选、镜像、TP2、context67584、显存0.85、MTP、prefill及输入生成／计时口径。它们与旧服务32的排队测试分开保存。

- C64：物理4,5，HTTP31540、NCCL31541、门禁31542；服务64／Graph64。
- C128：物理6,7，HTTP31560、NCCL31561、门禁31562；服务128／Graph128。
- seed20260912，4096输入、1024输出、ignore_eos；两轮各C预热，三轮各max(8,2C)正式。每轮128／256请求，分别384／768正式请求。
- 每实例启动前核空闲、持卡对flock，运行原数值／Graph门禁；启动后核API真实服务参数和候选guard。OOM／启动失败保留证据，不降成Graph32冒充容量通过。
- 只操作带本run标签、对应镜像和物理卡对的自有容器。完成／失败均停止自有实例并释放锁；不修改共享安装或默认v2脚本。

服务器控制目录：`/data/wangzy/v2/retest-server64-128-20261017-a`。脚本不覆盖已有控制目录或同名容器。状态、源码、完整命令、GPU前后状态、日志、API与原始轮次保持可追溯；输出U+FFFD单列，不用性能完整性PASS代替文本／模型精度。

本批使用两个独立实例，保留同机并行负载条件；不能把不同卡对、服务32→64／128与Graph变化合称同卡单项因果收益。

采样：压测期每5秒GET `/metrics`原文，指标按原标签分别记录，不跨rank相加；另有每5秒CPU文件progress快照，以started_at与正式轮wall_time限定正式窗口并向内留5秒。参数上限与实测运行峰值分别报告。实际本批执行controller SHA为 `1f2f2247725832e8af04f666d494902951c38b0b0d30e3a90b78f9adb1c3fb1f`；本地后续仅补顶层异常终态写入的版本未替换活动进程，已保留实际执行副本与Diff。


## 服务器复跑入口

此入口依赖原服务器路径 `/data/wangzy/v2/submit`，固定使用空闲物理4、5与6、7，各TP2。与README默认物理0、1的32上限评审入口分开；停止其他自有性能请求后执行，控制器核空闲、端口与卡对锁，拒绝覆盖。当前controller只比实际已执行版多了顶层失败终态处理，Diff已归档。

```bash
cd /data/wangzy/v2/submit
export CAPACITY_ID="capacity-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export CAPACITY_CONTROL="/data/wangzy/v2/${CAPACITY_ID}-control"
python3 -I execution/server64-128-20261017/controller.py --dry-check --package "$PWD"
# 观察器等待控制器创建目录；只读进度文件，不发模型/API请求。
python3 -I execution/server64-128-20261017/observe_progress.py \
  --control-root "$CAPACITY_CONTROL" --run-id "$CAPACITY_ID" &
OBSERVER_PID=$!
python3 -I execution/server64-128-20261017/controller.py --run \
  --run-id "$CAPACITY_ID" --control-root "$CAPACITY_CONTROL"
wait "$OBSERVER_PID"
```

新生成绝对时间窗口与轻量metrics原件在控制目录；benchmark在 `result/runs/<CAPACITY_ID>-p45-server64-4k-c64/4k-c64/` 与对应p67目录，用 `scripts/show_results.py` 分别读取。测量完成／失败自动停止自有容器和释放锁。新运行独立保存，不能覆盖本次已归档591文件。参数化观察器为复现补充工具，原已执行观察器完整源码与SHA仍保留。
