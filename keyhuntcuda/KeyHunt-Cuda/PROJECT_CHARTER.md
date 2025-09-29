## Keyhunt-CUDA Master Implementation Charter

### 1. 引用复用与溯源
- 修改 `src/KeyhuntCore/ecc/`, `scan/`, `compare/` 任何文件前，必须运行 `tools/sync_reference_sources.sh --apply` 并将快照保存至 `src/reference_snapshots/`；快照需生成 SHA256 校验和并记录到 `docs/source_fusion_report.md`。
- 仅允许复用 VanitySearch、BitCrack、(lib)secp256k1 等可信仓库；禁止自造 ECC 原语，改动需在文件头撰写 Provenance Header 并登记 `docs/source_fusion_report.md`、`docs/source_fusion_patches/`、`docs/license_matrix.md`。
- CI 中新增 `license-check` 任务，校验文件 SPDX 标识与 `license_matrix.md` 一致，防止许可证污染。

### 2. GPU 编码与数据规范
- 设备端代码仅使用 POD+C 风格数组，禁止 STL/异常/复杂模板；所有私钥统一 `__align__(32) Scalar256 { uint64_t limbs[4]; }` 小端表示，确保对齐与合并访问。
- 内核 ABI 固定为 `__global__ void scan_kernel(const uint64_t *priv_limbs, size_t n);`，所有设备函数加 `__device__ __forceinline__`；常量表按大小分配在 `__constant__` 或主机侧加载，数据传输使用 pinned memory + 多 stream。
- 明确 warp-level 原语使用规范（如 `__shfl_sync`、`__ballot_sync`、`__syncwarp`），避免跨架构不兼容；所有 CUDA API 必须包裹 `CUDA_CHECK(...)`，静态分析审查未检测的错误路径。

### 3. 测试驱动与验证体系
- 坚持 TDD：先写测试再实现，覆盖 BigInt256/Scalar256 序列化、自增、截断保护、GPU roundtrip。
- 每个 PR 必须完成 CPU/GPU 一致性（libsecp256k1 对照）、性能基准与随机模糊测试；历史性能基准数据以 JSON/CSV 存档，新性能低于上一版本 95% 即 fail；`tests/unit/test_truncation_protection.cpp` 强制保护高位。
- 夜间构建需执行大规模随机 fuzz（≥2^20 随机 Scalar256），检测越界、NaN、未定义行为。

### 4. CI Gate（Merge 必须全部通过）
1. `sync-and-license`
2. `provenance-check`
3. `unit-tests`
4. `validation-smoke`
5. `performance-smoke`
6. `static-analysis`
7. `coverage-check`（≥80%）
8. `cuda-memcheck` smoke（检测越界/未初始化）
9. `license-check`

### 5. 提交与审核流程
- Commit 模板：`[FUSE] <模块>: 适配 <repo/path>@<sha> → <新路径>`，并在正文列出修改、测试、溯源。
- PR 至少三维审核：密码学、GPU 性能、安全性；提交体包含来源、许可证影响、新增测试；CI 全绿方可合并，并在 `docs/review_logs/PR_<id>.md` 存档审核要点。

### 6. 文档与紧急响应
- 维护 `source_fusion_report.md`、`source_fusion_patches/`、`license_matrix.md`、`endianness.md`、`abi_kernel.md`、`format_privatekey.md`，并提供 `scripts/trace_snapshot.sh` 用于回溯源码来源链路。
- 出现截断/数据损坏时走 hotfix→运行时检测→incident 报告→历史结果隔离→回归测试补齐的闭环；CI 在 incident 期间自动阻止新 PR 合并，直至修复验证完成。

### 7. AI 行为防御（ASP 永久生效）
- 仅允许增量 diff，禁止新文件/重写/占位；所有修改需复用开源实现并附单元测试。
- CI 检查 diff 规模，若单文件替换比例 >80% 触发人工复核；静态扫描禁用 `TODO`、`mock_`、`dummy_` 等占位标识；GPU Runner 的执行日志自动归档并推送给 Fixer Agent。
- 每次任务前通过代码索引/RAG 获取上下文，多 Agent（Developer → Reviewer → Executor → Fixer）协同执行“写→跑→修”闭环；CI 在每次合并前自动打印 ASP 协议，确保人员与 Agent 牢记规则。

### 8. GPU-only 实施阶段与时间表
| 阶段 | 目标与输出 | 防御要求 | 周期 |
|------|------------|----------|------|
|0|仓库清理、环境统一、GPU capability 工具|禁止新增冗余脚本，所有改动包含测试|1 周|
|1|CPU 调度与验证框架|保留主干结构，任务描述符具备测试|1.5 周|
|2|GPU 数学库/GLV/哈希基础|libsecp256k1 对照、自测核、截断保护|2 周|
|3|GPU 核心扫描器|流水线全覆盖、压缩/未压缩测试、自动化 CPU baseline 对比|2 周|
|4|Montgomery/Pippenger/GLV 批处理优化|优化/非优化一致性 + 基准回归 + CPU baseline 对比|3 周|
|5|PUZZLE71 专用支持|常量写入 constant memory、Puzzle64–70 验证|1.5 周|
|6|测试与 CI 体系|CI Gate 全流程（含 coverage、cuda-memcheck、license-check、性能回归记录）|2 周|
|7|文档与发布|完整溯源/测试/性能说明|1 周|

### 9. 夜间构建/验证例程
1. `build_all.sh` 生成多架构 fatbin。
2. `run_unit_tests.sh` 执行 GPU 自测 kernel。
3. `run_validation.sh`：2^24 范围扫描 + CPU 抽查，随机化 Scalar256 种子避免固定样本。
4. `bench_gpu.sh --sm-list` 输出性能曲线并追加至历史趋势；性能低于预设阈值时自动告警。
5. `package_release.sh` 打包二进制、文档、溯源与测试报告，存档当日性能/测试摘要。

### 10. 长周期防崩溃要点
- 上下文持久化：借助代码索引/embedding 获取历史实现，禁止脱离仓库重写。
- 增量提交：CI 拦截 `_new/_copy` 等文件命名；覆盖率 <80% 自动 fail；diff 超限触发人工复核。
- 执行闭环：所有提交在 GPU runner 上构建+测试，日志自动回传 Fixer；连续 3 次夜间构建失败将锁定主分支，直至事故处理完成。
- ASP 重申：每次会话/PR 起始明确“禁止重写/新文件/占位/跳测”，CI 合并前也自动输出提醒；Reviewer/Executor/Fixer 的审查日志需归档便于复盘。

> 本 Charter 必须在每次规划、评审、实施前复读确认，确保后续工作完全遵循以上规则、阶段进度及防御机制。
