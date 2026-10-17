# submit1009 — 4K输入 / 1K输出性能交付

本次候选平均 **238.2423 tok/s**，同期前后v1合并 **208.6413 tok/s**，实测提升 **14.1875%**。6、7卡TP=2，4096输入/1024输出、C8、seed20260912，无Profiler；v1→候选→返回v1分别两轮8请求完整预热和三轮16请求正式测量。144条正式请求及48条预热全部完成，零错误。

| 变体 | 三轮输出吞吐 tok/s | 均值 tok/s | 极差/均值 |
|---|---|---:|---:|
| v1前置 | 204.1400 / 209.9246 / 211.8121 | 208.6256 | 3.6775% |
| 本次候选 | 233.0752 / 239.8198 / 241.8319 | 238.2423 | 3.6755% |
| v1返回 | 203.9501 / 210.2339 / 211.7871 | 208.6570 | 3.7560% |

增益分母为前后v1三轮均值的平均；按轮编号配对增益为 14.2273%, 14.1569%, 14.1796%。v1前后漂移 +0.0151%，观测波动参考值 3.7560%（各组极差和漂移的最大值），增益超过该参考值：True。每轮输入不同、对应轮跨变体输入相同，因此该参考值包含输入差异，非统计噪声置信上界。测试按连续变体块执行，并以前后v1检查漂移。

| 尾延迟指标 | 相对同期前后v1均值变化 |
|---|---:|
| ttft_s_p95 | -3.5878% |
| ttft_s_p99 | -3.5698% |
| tpot_ms_p95 | -11.9635% |
| tpot_ms_p99 | -13.4295% |
| e2e_s_p95 | -11.5009% |
| e2e_s_p99 | -11.4620% |

尾延迟<=5%门禁在本次小规模复测中：True。完整初始v1对照、逐轮分位数和原始请求在result中，不能以本次144条代替ShareGPT长稳态。

48条同输入的候选输出SHA分别相对前置/返回v1的不同条数为 {'v1': 0, 'v1-return': 0}。这仅为本次固定合成输入的观察，不代替语义精度或GSM8K验收。

此包为限时性能交付，**未完成此前三场景各20%的最终验收，未冻结**。4K达到20%：False；64K/ShareGPT C32未重跑，新候选实际模型Graph回放/compiler关联仍REVIEW_REQUIRED，净室复现/GSM8K未运行，其他精度由用户手动。任务原默认v1保持不变。

## 补丁和环境

patch/sitecustomize.py原SHA `da5b59bd28ea434356c7c96744c47f19493e540da7ba6f9bb890410e168c1610`，候选为已验证tile基底加BF16/TP2/<16MiB copy-in确定性native one-stage AllReduce；FP32/大BF16/不合格路由回原RCCL，full_nvlink=false保持。包含原v1 LDS-x GEMV与Long-KV功能及独立native数值门禁摘要，不混入失败/无收益的RMS、SiLU、Fill、Split-K、kpack、pipeline等项。

原厂镜像、gfx928、模型路径、软件版本和服务参数见env_declare.yaml。config/library_gpu4_int8tuned_stock_clean为完整净室stock库，DAT SHA `c3e295b24494507ede49ee2b330a96ddc9fe1654c58e5fd9611b3adf8b227c55`，没有替换共享DTK。patch/runtime的duplicate-registration防护源与本次实测容器完全一致，新容器用只读单文件挂载，不修改宿主或共享安装文件。requirements.txt记录厂商预装依赖，不执行pip替换厂商torch/triton。

## 复跑

在同服务器、原厂镜像与模型路径可用时，从本目录执行：

```bash
bash scripts/run.sh check
bash scripts/run.sh start
RUN_ID=manual-4k-$(date -u +%Y%m%dT%H%M%SZ) bash scripts/run.sh benchmark
bash scripts/run.sh stop
```

默认仅用物理6、7，TP2，Graph/MTP均与实测一致。start持有本任务gpu67非阻塞锁；发现锁被占用或同名交付容器存在即拒绝启动，不抢卡。start使用独立容器qwen38-submit1009-delivery；model只读挂载。端口默认31014/31015，可用PORT/NCCL_PORT调整。每次生成唯一日志/结果目录，拒绝覆盖。服务启动后预期地址http://192.2.111.66:31014；**交付时所有本任务服务已停止，该地址当前未保留服务**。

回滚：先stop，再 `VARIANT=baseline-v1 bash scripts/run.sh start`，该入口使用patch/baseline-v1原SHA `f1789532d58daa66187ba901d3d7513ff142cb33fafc7793ce44f4e821464495` 和AR关闭参数。最终测量已执行返回v1三轮验证。

## 证据和存储

result/delivery-native-ar-4k1k-20261010-b含控制/原队列源码/预检/完成标记/原始SHA/比较；另外三变体包含完整warmup/round JSONL、summary、配置、server-info和完整服务日志。result/independent-review/review.json为逐请求、吞吐、尾延迟、服务参数及SHA独立CPU复核；scripts/review_delivery_4k.py可在全新输出目录复核，不覆盖原摘要。result/card-release.json证明本任务所有容器停止、队列退出及gpu45/gpu67锁释放。

完整交付目录 `/data/wangzy/submit1009/delivery/20261010-4k1k/submit1009`。完整配置库和压缩交付包留服务器；本地仅小文件镜像，config库依据manifest保留远端，不将本地小镜像宣称为完整可部署包。历史大DB/tensor按用户授权清理，旧SHA/日志/结论和删除清单保留，不在此包冒称原始大数据仍完整。

新容器运行依赖：`patch/runtime/tvm-ffi/` 含本次实测容器已有的Torch C DLPack缓存（原文件200624字节，SHA见env_declare.yaml）。入口以只读目录挂载并指定TVM_FFI_CACHE_DIR，保留已测运行依赖，不关闭TVM功能、不修改共享库。首次新镜像检查的递归构建失败完整保留，修正后使用独立RUN检查。

最终新镜像启动检查已通过，API核对TP2、Graph、MTP及native backend，两rank guard/eligible和包内LDS加载均有效。独立交付容器随后停止并删除，4–7卡使用率0.0%、VRAM各4MiB，45/67锁FREE。此项仅启动完整性检查，没有净室性能复测/GSM8K，不能替代实际模型Graph回放/compiler关联门禁。完整前后检查证据见result/delivery-package-startup-20261010-{a,b}。
