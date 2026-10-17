# Qwen3.8-27B-Channel-INT8-w8a8 · K100-AI / SGLang

在海光 **K100-AI（gfx928）、两卡TP=2、SGLang 0.5.12** 上的INT8推理优化、Benchmark与复现材料。当前版本为 **[v2](v2/README.md)**，更新于2026-10-17；历史交付保留在 [v1](v1/README.md)。

[复现步骤](v2/submit/README.md) · [性能报告](v2/submit/docs/性能优化报告.md) · [补丁与完整源码](v2/submit/patch/README.md) · [当前材料状态](v2/submit/docs/提交材料状态-20261017.md)

## v2性能结果

下表为本机实测，输出TPS取三正式轮的算术均值，排除预热。随机负载为4096／65536输入、1024输出、ignore_eos；ShareGPT使用固定数据与安装版工具，口径单列。完整矩阵、三轮延迟和条件见 [报告§4](v2/submit/docs/性能优化报告.md#4-性能结果)。

| 代表场景 | 客户端C | 服务／Graph上限 | 输出TPS | TTFT mean s | TPOT mean ms |
|---|---:|---|---:|---:|---:|
| 随机4K/1K | 8 | 32／32 | 237.0503 | 4.8507 | 28.0799 |
| 随机4K/1K | 32 | 32／32 | 375.7735 | 13.8868 | 69.8337 |
| 随机64K/1K | 8 | 32／32 | 31.6825 | 75.8293 | 177.1380 |
| ShareGPT | 32 | 32／32 | 419.7877 | 0.5167 | 76.7629 |
| ShareGPT | 64 | 32／32 | 416.8612 | 14.8590 | 77.2397 |

新增高并发配置在独立TP2实例分别重测，保留同机并行条件：

| 随机4K/1K客户端C | 服务／Graph上限 | 输出TPS（三轮均值） | 正式期间采样运行峰值 | 结果定位 |
|---|---|---:|---:|---|
| 64 | 64／64 | 398.9657 | 64 | 观测达到64，另组性能参考 |
| 128 | 128／128 | 378.3503 | 76 | 排队诊断，实际128并发未验证 |

运行峰值取三正式轮窗口内每5秒的 `sglang:num_running_reqs` 原标签系列，窗口两端各向内5秒，不跨rank求和。C128的KV预算为333,504 tokens；参数上限128不能证明实际运行128。两组的延迟、文本标记、591个原始源文件与配置见 [独立复测报告](v2/submit/docs/服务端64与128并发复测-20261017.md)。原服务32下的旧C64／128仅留排队诊断原件。

SLA按本地赛事阈值逐轮判定：4K P95 TTFT＜10秒、TPOT＜80ms；64K＜30秒、＜100ms。4K已测C1–8范围内C7通过、C8失败，不能据此称全局最大；64K C1跨阶段结果不一致，稳定性仍待解决。部分随机响应有U+FFFD标记，原始记录与判定保留，吞吐和请求完整性不代表精度或SLA通过。

## 从仓库开始复现

```bash
git clone https://github.com/gugugu-b/qwen3.8-k100-ai-int8.git
cd qwen3.8-k100-ai-int8

# 从仓库已有压缩包准备v2所需rocBLAS库；逐项校验162文件。
python3 -I v2/prepare_library.py
cd v2/submit
bash scripts/run_review01.sh check
```

随后按 **[v2提交README](v2/submit/README.md)** 顺序执行：

1. 环境准备：镜像拉取／载入、hycu驱动与设备、官方Channel-INT8权重及ShareGPT、资产SHA、端口和卡对锁检查、完整 `docker run`。
2. 依赖安装：使用固定厂商镜像，核对Torch、Triton、SGLang、Aiter及两张gfx928可见性。
3. 模型启动：默认物理卡0、1；私有guard副本仅改设备断言并记录Diff与SHA。
4. Benchmark：单点／完整矩阵／SLA复核，使用唯一RUN_ID。
5. 得到结果：读取逐轮原件、均值、文本标记与逐轮SLA，停止自有容器。

本机已按卡0、1步骤实际验证依赖、模型／数据SHA、模型health和真实API，两个rank的AR guard生效；自有容器已停止并释放锁。见 [运行验证](v2/submit/result/evidence/readme-review01-20261017/runtime-review.json)。

README采用用户指定的 `harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10` 拉取地址；本机验证时该地址因服务器DNS失败未完成在线身份核验。实测镜像来自同版本 `hcu/admin/base/sglang`，固定image ID为 `sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad`。提交README保留固定身份门禁与实测引用；不能仅按相同tag认定两个地址等价。

## v2改动与交付

v2沿用INT8 Multi-M LDS GEMV、权重转置缓存、Triton Long-KV、Aiter配置和rocBLAS库，增加 **18个INT8 Triton tile** 与 **BF16 native one-stage AllReduce资格筛选**。底层厂商native AR kernel保持原样；完整自定义源码、五份分项Diff、实际运行overlay、上游快照、构建入口和来源边界见 [补丁说明](v2/submit/patch/README.md)。卡0、1的设备断言Diff与完整guard也已归档。

```text
.
├── README.md                    # 当前v2总览与入口
├── PERFORMANCE.md               # v1历史性能记录
├── LICENSE
├── v1/                          # 历史交付与原运行库压缩包
└── v2/
    ├── README.md                # v2版本说明与资产准备
    ├── prepare_library.py       # 从已有压缩包准备并核验完整库
    ├── release.json             # 版本、源包身份与清单
    ├── tools/seal_manifest.py   # 材料有意变更后的封存工具
    └── submit/
        ├── README.md            # 环境→依赖→启动→Benchmark→结果
        ├── env_declare.yaml
        ├── scripts/             # 运行与CPU校验入口
        ├── patch/               # 完整源码、Diff、已测扩展与来源
        ├── config/              # 配置及完整库SHA清单
        ├── execution/           # 实际调度与精度脚本
        ├── logs/                # 服务日志、原始性能归档
        ├── result/              # 原始证据、逐请求与汇总
        ├── manifest/            # 包文件SHA
        └── docs/                # 性能、精度、Demo与材料状态
```

原提交包6192个封存文件逐字节保留，另保留运行历史。模型权重与ShareGPT大文件按README在包外准备；展开的rocBLAS库不重复纳入Git，由已有压缩包准备。完整原始日志和已测二进制随v2提供。

## 精度与当前限制

GSM8K本地冻结EvalScope 1.6.1四样例全量协议：**1288/1319＝97.6497%**，达到本地95%门槛，逐题响应与独立重评分通过，见 [GSM8K报告](v2/submit/docs/GSM8K精度验证-20261017.md)。最终官方协议等价性尚待核对；v2的LiveCodeBench v6、RealWorldQA、MathVision Without CI尚未运行。v1精度记录不能代替v2结果。

当前仍需解决64K SLA稳定性、其余精度、最终协议和组合验证，不宣称全部赛事准入通过。历史构建日志及供应商native AR匹配完整C++／HIP源码的交付边界见补丁说明。[Demo视频](v2/submit/docs/demo/demo.mp4)来自真实API事件时间流的终端回放。

项目代码许可见 [Apache 2.0](LICENSE)；随包上游源码与依赖的许可见对应目录。
