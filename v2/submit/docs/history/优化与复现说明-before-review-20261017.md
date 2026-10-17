# 优化与复现说明

## 版本来源和源码

`submit1009`是开发工作包，其默认patch仍为v1。新目录选择10月10日交付包中的已测tile＋BF16 native AR候选，随后10月16日继续使用该site。候选site SHA：`da5b59bd28ea434356c7c96744c47f19493e540da7ba6f9bb890410e168c1610`；6,7 guard SHA：`55485f3994417b012fd76f3cf423b03b33fd90d52e2c73d98d0fc9315ab0b5aa`。

4,5测量的guard仅卡对断言不同，SHA为`836ea4d43b1a4cd6bcc56c3998e09ff1e92d9ebdb5105b4937f6fe7a1f522d12`；原件随64K/C1证据保存。本目录新增入口只支持6,7，以保留所选guard原SHA，不偷偷修改为通用卡对。两卡对结果只能按实际条件分别陈述。

- `patch/sitecustomize.py`：小M GEMV、Long-KV动态源变更、18-shape tile覆写与native guard安装。
- `patch/triton_tiles.json`：精确shape配置；`patch/native_bf16_ar_guard.py`：AR资格与厂商源码/二进制SHA断言。
- `patch/k100_int8_gemv_unified.hip`及so：自有HIP源码和实测二进制。
- `patch/baseline-v1/`：内部v1回滚原件；`patch/v2-vs-v1.diff`为site注入差异，新增guard/tile源码单独提供。
- `patch/runtime/direct_register_custom_op.py`：实测duplicate-registration防护完整源；以只读容器overlay加载。
- `patch/runtime/tvm-ffi/`：实测DLPack addon缓存，避免新镜像递归构建。
- `manifest/copied-source-map.json`：复制原件的工作区相对来源及SHA。**此文件描述整理时的原件，不描述后来新增/编辑的入口**；当前文件身份以package-sha256.json为准。

`patch/build_unified.py`是保留的历史构建原件，包含历史服务器输出路径。建议使用新增 `scripts/rebuild_kernel.py --output /tmp/qwen38-rebuild-唯一目录` 在声明镜像内重建，只输出到新目录，不覆盖实测so。保持目标gfx928及编译flags；重建后必须重新核数值、实际模型Graph和性能，不能沿用原二进制的验证结论。

## 环境和依赖准备

按照README先同步目录、准备模型、库和数据。SGLang镜像固定digest；不升级torch/triton。五项Aiter JSON逐文件只读挂入对应安装路径，overlay只作用于独立容器。rocBLAS完整库使用外部只读挂载或prepare_assets复制全部162项后逐SHA校验，不改共享DTK。

`env_declare.yaml`采用JSON语法（合法YAML）。其中TO_SUPPLY、NOT_RUN、REVIEW_REQUIRED是真实缺项；历史manifest/historical-v1只用于溯源，不能替代当前环境声明。

## 完整复跑

从README中的check→check-runtime→start开始。启动入口只创建带 `qwen38.v2.submit=20261016` 标签的专用容器 `qwen38-v2-submit`，拒绝同名覆盖；停止只针对该身份、image ID和物理卡位一致的容器。服务持有现有gpu67工作区锁；没有向0–3卡发任务。设备节点挂载遵循原厂容器惯例，进程通过HIP_VISIBLE_DEVICES=6,7约束计算。

固定长度：`RUN_ID=唯一值 bash scripts/run.sh benchmark 4k 8` 或 `64k 8`。新增并发参数仅改变客户端，服务运行数32/Graph32保持；高并发结果必须保留这两个配置。每档三轮，正式请求数max(16,2C)；C8与已有固定测试一致。扩展矩阵是诊断计划，若官方请求量不同须按官方命令重测，不能冒称其为官方最终工具。

ShareGPT：`RUN_ID=唯一值 bash scripts/run.sh sharegpt 32`，允许1/4/16/32/64，固定seed20260912、上下文67584、1000请求、完整预热1轮、正式3轮，每次64短预热。数据SHA与实测一致才执行。客户端日志包含benchmark实际参数。源码、配置与服务API随每轮数据一并归档。

新测试位于 `result/runs/<RUN_ID>/`，启动日志位于 `logs/<启动RUN_ID>/`；每个输出目录拒绝覆盖。请求失败/长度不符/缺轮则校验失败。若需重试，保留失败目录并换RUN_ID。Profiler/trace必须关闭后才采正式性能；单项优化分别A/B，不合并多个新变更。

## CPU核验与原始证据

`python3 scripts/review_summary.py`核对七场景的原始逐轮TPS、均值、历史参考变化与源SHA；`bash scripts/run.sh check`核对包内清单。新固定测试自动validate_results，新ShareGPT自动validate_sharegpt。PASS仅指该工具声明的完整性/算术范围，不代表SLA、精度或所有工程门禁。

result/evidence按4k-delivery、64k-validation、sharegpt-c32、sharegpt-c1、sharegpt-c4-c16-c64保存各自原始目录结构，包括命令、预热、正式原始输出、服务日志、API/配置、独立审核和原manifest。原manifest仍引用历史相对路径/远端路径，含未本地同步的大文件；请结合所在archive-inventory与README识别，不把原manifest全量可用当本地事实。当前包本地实际文件由package-sha256覆盖。

`result/summary/files/`按最终报告source_inventory拷入并逐SHA复核，避免汇总依赖包外工作区。各源文件未改写。component-gates中的tile-only Graph PASS仅限原tile；最终组合Graph仍REVIEW_REQUIRED。算子数值检查不能代替四项模型能力准入。

## 净室复现与回滚

本次新目录未启动GPU服务。历史10月10日新镜像启动检查已归档，仅能证明旧入口当时启动成功。应在库与权重资产固定后，用本目录独立新容器实际执行所有适用benchmark；保存镜像digest、CPU/驱动/内存/拓扑、完整依赖清单、配置/权重SHA、启动命令/API、数据、错误与资源释放日志，补上最终组合的真实模型Graph/compiler关联证据。

回滚：stop当前专用容器，再 `VARIANT=baseline-v1 bash scripts/run.sh start`。native自定义AR关闭，加载v1原site SHA `f1789532d58daa66187ba901d3d7513ff142cb33fafc7793ce44f4e821464495`；这不是官方stock baseline。需要证明单项因果收益时，应固定同卡对、同输入、同预热、同镜像，按原项分别做前置/候选/返回对照。最后stop并核实本任务进程与锁释放，保留记录。

性能与工程固定后，最后按官方数据/脚本运行四项精度准入并归档。失败后改变实现会使相关性能与清单失效，需要重新验证。


## 2026-10-16 后续执行更新

用户已授权8卡四个TP2实例。执行源码和分工见 `execution/community-v2-20261016-a/`；默认单实例run.sh仍固定6,7。四实例控制器持有每个卡对的宿主锁，并仅给自有容器传入外部锁已持有标记，serve.sh才跳过容器内重复加锁；0–7启动前都须检查空闲。算术不变，各卡对仅复制修改guard物理卡断言。正式数据保留四实例并行条件，不冒称隔离成绩。

另已收入晚间服务64/Graph64的C64失败归档 `result/evidence/sharegpt-c64-server64-failed/`：三轮473.6951tok/s、TTFT P99 6177.7480ms，但U+FFFD文本完整性检查失败，不能替换原七场景的合格数据或称修复通过。原始判定保留。本轮缺档诊断继续使用服务32/Graph32。
