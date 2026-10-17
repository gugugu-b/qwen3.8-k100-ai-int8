# 补丁、完整源码与构建说明

本目录交付实际候选使用的自定义源码、继承版本、分项 Diff、二进制身份及来源。默认启动使用已测二进制和 Python overlay；本次整理没有更改候选算法或替换 `.so`。另在固定镜像中以无 GPU 设备、无网络、只读容器导出供应商源文件和真实 Diff，没有编译或推理。

## 1. 先核验交付内容

在提交包根目录执行，仅需 Python 标准库，不会导入 Torch 或启动 GPU：

```bash
python3 -I scripts/verify_patch_bundle.py
```

预期 `status=PASS`。它核对 `bundle-manifest.json` 中的逐文件 SHA256、完整 v1 继承关系、18 个 tile、分项 Diff 与源码的一致性、native AR guard 的依赖身份。该 PASS 表示源码和交付材料一致；数值、Graph、模型精度与性能结果以 `result/` 原始证据和结果报告为准。

## 2. 修改范围与源码入口

| 修改项 | 完整自定义源码／配置 | 来源与边界 | 生效方式 |
|---|---|---|---|
| INT8 LDS 多 M GEMV | `k100_int8_gemv_unified.hip`、`build_unified.py`，对应已测 `.so` | 从 v1 原样继承；M=1/2/3/4，特定投影 shape；没有在 v2 改写 HIP kernel | `sitecustomize.py` 按 dtype、shape、布局筛选后调用 PyBind 扩展，其他输入回落原路径 |
| GEMV 调度、权重转置缓存 | `sitecustomize.py` 前 293 行；`baseline-v1/sitecustomize.py` 是完整继承版 | 从 v1 原样继承；源码包含 shape、scale、对齐检查和 fallback | `PYTHONPATH` 装载；不是对共享安装目录打补丁 |
| Long-KV Triton GQA | `sitecustomize.py` 的 `_raw` 变换、`_longkv_eligible`、`_longkv_try` | v1 继承；运行时读取镜像 FlashAttention 模块，改两个精确文本锚点，选择 BM128／BN64／8 warps；完整自定义变换、路由源码及固定镜像变换前后整文件已交付 | 生成进程私有 `/tmp/q38_tp2_longkv_<pid>.py` 并 JIT；未覆盖镜像源码 |
| INT8 Triton tile 选择 | `sitecustomize.py` 的 `_v2_make_matmul`；`triton_tiles.json` | v2 增量，18 个唯一 M/K/N，SPLIT_K=1；底层供应商 Triton kernel 未修改，分别覆盖 LMSlim 与 Lightop 私有 LMSlim 的别名 | Python wrapper 将匹配 shape 的 tile 配置传给原函数；JIT 由固定镜像 Triton 完成 |
| BF16 native one-stage AR 筛选 | `native_bf16_ar_guard.py` 与 sitecustomize 最后调用 | v2 增量，仅允许 BF16、TP2、安装版 deterministic copy-in 且输入小于 `max_size`；原生 AR kernel 属固定镜像，未修改；不更改拓扑探测 | 包装 `CustomAllreduce.should_custom_ar`，校验安装版 Python 和 kernel SHA；默认卡对 6,7；其他卡对由运行控制器生成仅变更设备断言的私有副本 |
| Lightop 重复注册修复 | `runtime/direct_register_custom_op.py` 为完整生效源码；`patch_dup_reg_guard.py` 为历史文本修复源 | v1 继承的兼容修复；遇已定义 op 保留既有实现 | `scripts/run.sh` 只读 bind 到单容器内指定文件。正常复现不运行会写共享路径的历史修复脚本 |
| TVM FFI Torch C DLPack 缓存 | `runtime/tvm-ffi/libtorch_c_dlpack_addon_torch210-rocm.so` | 既有上游缓存，200624 字节，从实测容器复制；固定镜像内完整 addon.cc、builder、loader、DLPack header 已归档。不是本项目自研算子，未改其源码或编译逻辑 | `TVM_FFI_CACHE_DIR=/run/qwen38-tvm-ffi` 加只读挂载；不关闭 TVM 功能 |
| Aiter／rocBLAS 配置资产 | `../config/aiter/` 五份 JSON；`../config/library-sha256.json` | 配置文件和只读库资产，不属于本目录新 kernel。库 162 文件需由 `prepare_assets.py` 完整 SHA 校验；本地小包不含完整库，服务器包或验证过的外部目录提供 | `scripts/run.sh` 只读挂载，路径在启动日志记录 |

`baseline-v1/` 提供回溯基准，不是额外的待安装依赖。当前候选与 v1 的完整 sitecustomize 差异也保留在 `v2-vs-v1.diff`。`native-ar-manifest.json` 是原始历史声明，保留其 `actual_model_graph_gate=REVIEW_REQUIRED`；本次源码核验不替代该审查。

## 3. 分项 Diff 的含义

| 文件 | 比较对象 | 使用方法 |
|---|---|---|
| `diffs/0001-lightop-duplicate-registration.patch` | 固定镜像 Lightop 原件 → 实际生效完整 overlay | 原件和运行态完整文件见 `source/fixed-image-export/materialized/`；与此前重建 preimage 和 Diff 已逐字节交叉核对一致。正常复现直接 bind overlay |
| `diffs/0002-v2-triton-tiles.patch` | 真实完整 v1 sitecustomize → tile 增量，同时新增完整 tile JSON | 供审核；其中中间 tile-only 文件不是默认候选 |
| `diffs/0003-v2-native-bf16-guard.patch` | tile 中间版 → 最终候选，同时新增完整 guard | 供审核；与上一项共同构成 v2 增量 |
| `diffs/0004-inherited-longkv-anchor-edits.diff` | 从实际 sitecustomize 提取的两个 old/new 字符串 | 仅供看清精确锚点内容，行号不是供应商整文件行号；整文件审阅使用下一项 |
| `diffs/0005-inherited-longkv-full-source.patch` | 固定镜像 FlashAttention 原件 → 实际 sitecustomize 变换后的运行态完整源 | 真实整文件行号，原件和运行态完整文件见 `source/fixed-image-export/materialized/`；自动核验仅两锚点改动 |

自定义源码已完整包含在本包，不要求评审从这些展示 Diff 重新组装运行目录。运行主流程按包根目录 README，通过只读 mount 加 `PYTHONPATH` 生效。

固定镜像对应的 LongKV 和 Lightop 原始／运行态完整源码、真实 Diff 和 SHA 已归档在 `source/fixed-image-export/materialized/`；环境命令见 `source/fixed-image-export/export-environment.json`。LongKV 原件 SHA 为 `f00a1286df85fb923dcf87be251d9ffd7b8e2967a9715937adec4dfbe8f8f307`，变换后为 `5a107b5d3e03260168820ceb13f0a20ae14527f9332e75760b07317dd89bcf3b`。若需独立再次核验，在固定镜像的新容器内执行（使用包根目录，`--output` 必须是未存在的新目录）：

```bash
python3 -I patch/source/materialize_vendor_overlay.py \
  --output /tmp/qwen38-vendor-source-review-unique
```

该工具不导入 Torch，也不写安装目录。它读取镜像原始模块，校验 LongKV 的已知锚点状态，输出供应商输入完整源码、运行时变换后完整源码、实际行号的 unified Diff 及逐文件 SHA。Triton 的固定 JIT 配置同时写入导出 manifest。若供应商文件不符合已知锚点，工具立即失败；不能用另一镜像静默替代。

## 4. 编译与加载

### 4.1 已测 HIP 扩展

`build_unified.py` 是从 v1 原样保留的历史构建配方：`torch.utils.cpp_extension.load`、模块名 `k100_int8_gemv_unified`、`CXX=hipcc`、`extra_cuda_cflags=['-O3','--offload-arch=gfx928']`。它硬编码 `/tmp/q38_build_unified_gemv` 与 `/data/wangzy/native_ext`，**正常复现直接使用随包已测 `.so`，不运行此历史脚本覆盖产物**。

已测源／二进制 SHA：

```text
k100_int8_gemv_unified.hip dd753458416463665a97c5aedc04cffaed7b992a58190a66c9188619d20c8ae2
k100_int8_gemv_unified.so  786865c76847c43aab8b389702cf4cbf522b90157575b67a3ca5d61a3f0ec97d
```

固定环境为 `env_declare.yaml` 的镜像 digest、Python 3.10、Torch 2.10 DTK2604、gfx928。若评审希望从源码另行重建，包内已有写入新目录的入口，在固定镜像内执行：

```bash
source /opt/dtk/env.sh
env -u PYTHONPATH CXX=/opt/dtk/bin/hipcc \
  python3 -I scripts/rebuild_kernel.py \
  --output /tmp/qwen38-gemv-rebuild-unique
```

该命令是按已交付构建配方提供的**重建入口，本次整理未执行**。本地归档未找到生成已测 `.so` 时的原始 Ninja／hipcc 全参数日志，因此没有伪造原始命令或宣称重建字节一致。重建结果不能直接冒用现有性能／精度成绩：先记录新 SHA，独立完成数值、Graph、模型精度与性能验证，再决定是否替换。入口不会覆盖已测 `.so`。

### 4.2 Python／Triton／AR

Python overlay 无额外编译步骤。Triton tile 复用镜像底层 kernel，LongKV 用镜像 Triton 按源码 JIT；本项目没有修改 Triton 编译器。BF16 AR 复用安装版 `sgl_kernel/common_ops...so`，没有自行编译原生 AR kernel。来源快照、原始路径、SHA 和限制见 `provenance/source-origin.json`、`source/vendor-reference/` 与 `bundle-manifest.json`。

## 5. 固定镜像上游源码与构建证据边界

`source/fixed-image-export/tvm-ffi-source/` 已归档固定镜像 Torch C DLPack builder、完整 `addon.cc`、loader、libinfo、lockfile 和 DLPack header。`addon.cc` 通过 AST 解析 builder 的 `cpp_source` 常量提取，不执行 Torch／TVM；verifier 会逐字节核对。上游源码保留版权声明，固定镜像实际提供的 `apache_tvm_ffi-0.1.9.dist-info` LICENSE／NOTICE 与 libbacktrace LICENSE 另归档在 `source/fixed-image-export/licenses/`；它们是上游依赖源码，没有标为自研。

TVM builder 的 Linux 配方在 `_run_build_on_linux_like`：C++17、PIC、O3、hidden visibility、shared、rpath，加对应 Torch／ROCm include 和库；具体 flags 由固定镜像 builder 与 Torch 推导。源码的 CLI 提供 `--build-with-rocm`、`--build-dir`、`--output-dir`、`--libname`，独立重建入口如下。输出目录必须另建，不能覆盖随包缓存；该命令**未在本次整理中执行**：

```bash
source /opt/dtk/env.sh
# 使用固定镜像原版 builder；清除候选导入路径，使用新缓存与新构建目录。
env -u PYTHONPATH CXX=/opt/dtk/bin/hipcc \
  TVM_FFI_CACHE_DIR=/tmp/qwen38-tvmffi-cache-review-unique \
  python3 -I /usr/local/lib/python3.10/dist-packages/tvm_ffi/utils/_build_optional_torch_c_dlpack.py \
  --build-with-rocm \
  --build-dir /tmp/qwen38-tvmffi-build-review-unique \
  --output-dir /tmp/qwen38-tvmffi-output-review-unique \
  --libname libtorch_c_dlpack_addon_torch210-rocm.so
```

TVM builder 自身为防构建递归在构建进程内设置 `TVM_FFI_DISABLE_TORCH_C_DLPACK=1`；这是上游 builder 的内部行为。提交的推理服务依旧加载只读缓存，没有关闭 TVM 功能。

仍须公开的证据边界：

1. **TVM cache 的原始构建记录**：缓存 SHA 为 `89a8fd252cf44cc4ae1120c75f938c54e796f79f94e083717a85df1cfe5ef7f6`，固定镜像源码和配方已提供；没有原缓存生成时的完整编译日志，未证明另行构建与已测缓存字节一致。不能将本次只读源码导出写成重编通过。
2. **native AR kernel**：已有匹配 guard SHA 的完整 Python 调用层快照及 `sgl_kernel` binary identity，未取得该安装版 `common_ops` 的匹配完整 C++／HIP 实现及构建配方。项目只交付自己改动的完整 Python guard，不把厂商 kernel 宣称为自研；如赛事要求供应商原生库也完整源码可构建，需向供应方补齐该部分。
3. **历史 HIP 全参数构建日志**：有完整 HIP、历史构建配方、已测二进制和 SHA，没有原始 Ninja 日志；重建入口如上，未假称本次重编通过。

源码交付范围是本项目实际改动的 HIP／Python／配置与变换逻辑。固定供应商依赖保留其来源边界，不以不匹配的公开源码替代对应版本证据。

## 6. 文件清单与回退

`bundle-manifest.json` 给出本目录全部交付文件和 verifier 的 size／SHA，`provenance/source-origin.json` 给出历史来源路径。此补丁 manifest 独立于总包 `../manifest/package-sha256.json`；总包封存时也应包含新补丁文件。

停止并删除本次创建的容器即可撤销只读 overlay 和运行时 monkey patch。不要将 `patch_dup_reg_guard.py`、`sitecustomize.py` 写回共享镜像；不要将本目录 `.so` 放入无版本隔离的系统库目录。
