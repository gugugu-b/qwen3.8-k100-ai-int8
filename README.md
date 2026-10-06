# Qwen3.8-27B-Channel-INT8-w8a8 on Hygon K100-AI (TP=2) 深度调优加速方案

[![Hardware](https://img.shields.io/badge/Hardware-Hygon%20K100--AI%20(gfx928)-blue)](https://www.hygon.cn/)
[![Software](https://img.shields.io/badge/DTK-26.04-green)](https://sourcefind.cn/)
[![Framework](https://img.shields.io/badge/Serving-SGLang%200.5.12-orange)](https://github.com/sgl-project/sglang)
[![Model](https://img.shields.io/badge/Model-Qwen3.8--27B--Channel--INT8--w8a8-purple)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

面向 **海光 K100-AI（GPGPU / DCU，gfx928 架构，双卡 TP=2）** 的 **Qwen3.8-27B-Channel-INT8-w8a8** 深度推理性能优化与全流程工程加速交付方案。

---

## 🏆 核心性能突破与基准数据

在并发 $C=8$（16 请求/轮，每个请求严格生成 1024 完整 Tokens）的标准基准评测下，**两项核心场景均创下海光 K100-AI 历史最高性能纪录**，长上下文突破 30 tok/s 历史大关：

| 评测场景 | 基准状态 | 输出吞吐 (tok/s) | 总吞吐 (tok/s) | 单轮耗时 (s) | TTFT P50 | TPOT P50 | 相对原生提升 |
|---|---|---|---|---|---|---|---|
| **4k/1k C8** | **Tuned (本调优方案)** | **207.81** (均值 206.84) | **1034.21** | **78.04** | **2.70 s** | **31.83 ms** | **+37.11%** |
| | Stock (原生未调优) | 152.69 | 763.45 | 107.30 | 6.78 s | 45.77 ms | 基准对照 |
| **64k/1k C8** | **Tuned (本调优方案)** | **30.47** (均值 30.39) | **1975.47** | **537.78** | **40.75 s** | **167.64 ms** | **+29.26%** |
| | Stock (原生未调优) | 23.51 | 1528.15 | 697.10 | 133.64 s | 182.15 ms | 基准对照 |

### 核心亮点：
1. **64K 场景历史性突破 30 tok/s**：
   - 输出吞吐跨越至 **30.39 ~ 30.47 tok/s**（相比 Stock 提升 **+29.26%**）；
   - 首字延迟（TTFT P50）从 133.64 秒骤降至 **40.75 秒 (-69.5%)**；
   - 单轮 16 并发耗时从 697 秒缩短至 **537 秒 (-22.9%)**；
   - 跨两个独立净室容器（`tuning4` 与 `tuning5`）连续 6 轮实测，全部稳态落在 `30.26 ~ 30.48 tok/s`，轮间极差仅 0.65%，突破确定性成立。
2. **4K 场景极速解码**：
   - 输出吞吐峰值达到 **209.96 tok/s**，逐 Token 解码延迟压低至 **31.83 ms**。
3. **100% Zero-Defect 零缺陷质量认证**：
   - 全流程 96 条长短请求 100% 精确达到 1024 Token 长度；
   - 0 报错（`err=0`），0 乱码替换字符（`repl=0`），文本多样性与语义完整性完备。

---

## 🛠️ 六大系统级与算子级创新

```
                          ┌────────────────────────────────────────────────────────┐
                          │               Qwen3.8-27B 混合注意力推理管线           │
                          └────────────────────────────────────────────────────────┘
                                      │                                  │
                                      ▼ (长上下文 Prefill)                ▼ (自回归 Decode / Verify)
                        ┌───────────────────────────────┐  ┌───────────────────────────────────┐
                        │   16 层全注意力 (Full Attn)   │  │    Unified Multi-M LDS-x GEMV     │
                        │    Triton Long-KV GQA 算子    │  │ (qkv / gate_up / down / out 投影) │
                        │      BM128 / BN64 / w8        │  │     M in [1, 2, 3, 4] 算子加速    │
                        │    64K 单步提速 5.82x (84ms)  │  │   单算子较 Lightop 提速 1.55~3.62x │
                        └───────────────────────────────┘  └───────────────────────────────────┘
                                      │                                  │
                                      ▼                                  ▼
                        ┌───────────────────────────────┐  ┌───────────────────────────────────┐
                        │    48 层线性注意力 (GDN)      │  │        MTP 投机采样与调度选优     │
                        │ Aiter 补全真实维度 H=8, HV=24 │  │     steps=1, draft=2, chunk=8192  │
                        │    消除默认单 Wave 静默回退   │  │   消除显存带宽墙，有效吞吐跃升    │
                        └───────────────────────────────┘  └───────────────────────────────────┘
```

1. **Unified Multi-M ($M \in [1, 2, 3, 4]$) LDS-x 全投影 GEMV**：
   - 突破单 Token GEMV 限制，特化针对并发小批次及 MTP 验证阶段 $M \in [1, 2, 3, 4]$ 的多行向量并行；
   - 全面覆盖四大投影矩阵：`qkv_proj` ($K=5120 \to N=8192$)、`gate_up_proj` ($K=5120 \to N=17408$)、`down_proj` ($K=8704 \to N=5120$)、`out_proj` ($K=3072 \to N=5120$)；
   - 片上 LDS 激活向量高效复用，每个 Wavefront 读取一次权重即可多行复用，单算子提速 **1.55× ~ 3.62×**。
2. **权重转置持久零开销缓存**：
   - 权重在初次加载时转置一次并持久绑定在 Parameter 上，消除解码循环中高频重复的动态内存申请与转置拷贝。
3. **Triton Long-KV GQA 全注意力 Prefill 加速**：
   - 针对 16 个密集全注意力层（`layers 3, 7, 11, ..., 63`），在 64K 长 KV 下挂载优选 `BM128/BN64/w8` 算子并辅以 Ceil 边界保护；
   - 单步注意力耗时从 490.59 ms 骤降至 **84.23 ms（提速 5.82×）**，直接将 64K TTFT 缩短 92 秒。
4. **Aiter 线性注意力真实维度补齐**：
   - 为 48 层 GDN 线性注意力注入 TP=2 下真实维度（`H=8, HV=24`）配置，消除默认单 Wave 静默回退，释放 CDNA 计算潜能。
5. **MTP 投机采样与分块调度黄金组合**：
   - 锁定最优超参：`--speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`；
   - 调度策略锁定 `--chunked-prefill-size 8192`，最大化重叠 Prefill 与 Decode 吞吐。
6. **净室 Stock rocBLAS 高优 GEMM 选优库**：
   - 覆盖模型 46 类 Layer3 GEMM Shape（`M=48, K=5120` 等），Kernel 级几何平均加速比达 **2.725×**（SHA256: `c3e295b24494507e...`）。

---

## 📂 仓库结构

```
.
├── README.md                      # 本手册：总览、架构、性能与快速开始
├── PERFORMANCE.md                 # 官方基准数据与两轮独立复现明细表
├── LICENSE                        # Apache 2.0 开源协议
├── .gitignore                     # Git 忽略配置
└── v1/                            # Version 1 优化交付版本
    ├── README.md                  # v1 版本发布总览
    ├── submit1005/                # ⭐ 完整自包含的交付包 (开箱即用)
    │   ├── README.md              # 交付包详细手册
    │   ├── env_declare.yaml       # 软硬件环境与超参规范清单
    │   ├── requirements.txt       # Python 依赖清单
    │   ├── config/                # 调优配置目录
    │   │   ├── serving_config.yaml
    │   │   ├── library_gpu4_int8tuned_stock_clean.tar.gz # 净室 GEMM 选优库 (86MB 高度压缩包)
    │   │   └── aiter/             # Aiter Triton gfx928 优化配置 (H=8, HV=24)
    │   ├── patch/                 # 内核源码与运行时无侵入注入钩子
    │   │   ├── sitecustomize.py   # 运行时优化拦截注入
    │   │   ├── k100_int8_gemv_unified.so # 预编译 Unified GEMV 动态库
    │   │   ├── k100_int8_gemv_unified.hip # 内核源码
    │   │   ├── libtorch_c_dlpack_addon_torch210-rocm.so # tvm-ffi dlpack ROCm 预编缓存
    │   │   └── patch_dup_reg_guard.py # lightop 重复注册幂等保护补丁
    │   ├── scripts/               # 🚀 彻底解耦的自动化脚本
    │   │   ├── apply_patch.sh     # [步骤1] 一键打上全部补丁与环境自检
    │   │   ├── start_server.sh    # [步骤2] 独立拉起模型服务与健康轮询
    │   │   ├── benchmark.sh       # [步骤3] 独立基准评测 (all/4k/64k) 与质量报告
    │   │   └── run.sh             # [端到端] 一键串联运行全流程
    │   ├── logs/                  # 历史运行日志
    │   └── result/                # 完整原始测试数据
    └── verification/              # 净室容器 (tuning5) 独立复现验证凭证
        ├── VERIFICATION_REPORT.md # 独立复现深度分析报告
        ├── compare_verify.py      # 跨轮次对比脚本
        └── results/               # tuning5 独立复现完整原始数据
```

---

## 🚀 快速上手与跨机器复现

交付包已彻底解耦外部依赖，支持在任何海光 K100-AI 服务器上开箱即用：

### 1. 环境准备
使用官方标准基础镜像创建容器并进入：
```bash
docker run -itd \
    --shm-size 512g \
    --network=host \
    --name Qwen3.8-27B-Channel-INT8 \
    --privileged \
    --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd \
    --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -u root \
    -v /opt/hyhal/:/opt/hyhal/:ro \
    -v /data/:/data/ \
    harbor.sourcefind.cn:5443/hcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10 bash

docker exec -it qwen3.8-k100-ai-int8 bash
cd qwen3.8-k100-ai-int8/v1/submit1005
```

### 2. 模块化执行（推荐）

#### 第一步：一键打上全部补丁
```bash
bash scripts/apply_patch.sh
```
- **自动处理**：配置 `hipcc` 编译环境、部署 tvm-ffi 扩展缓存、打入 lightop 幂等保护补丁、同步 Aiter 真实维度、解压并校验 clean GEMM 选优库；
- **自检反馈**：脚本末尾自动进行 Python 动态导入自检，输出 `[SUCCESS] All patches successfully applied and verified!`。

#### 第二步：独立拉起模型服务
```bash
bash scripts/start_server.sh
```
- 自动清理历史占用端口、注入调优环境变量（Tensile clean 库、HIP_VISIBLE_DEVICES=4,5 等）；
- 后台启动并自动轮询健康检查接口（`http://127.0.0.1:31004/health`），直到返回 `200 OK` 汇报就绪。

#### 第三步：独立执行基准测试
```bash
# 全场景评测 (4k/1k 3轮 + 64k/1k 3轮)
bash scripts/benchmark.sh all

# 仅评测 4k/1k 场景
bash scripts/benchmark.sh 4k

# 仅评测 64k/1k 场景
bash scripts/benchmark.sh 64k
```
- 测试完成后自动执行 Zero-Defect 质量报告（校验报错数、乱码替换符、精确 1024 长度）并输出吞吐均值。

---

## 📄 许可证

本项目遵循 [Apache 2.0 开源协议](LICENSE)。
