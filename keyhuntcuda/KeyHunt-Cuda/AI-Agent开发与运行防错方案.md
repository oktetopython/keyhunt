## AI Agent 开发与运行防错方案（工业级版）

### 0. 适用范围
本方案用于 Keyhunt-CUDA 及其配套项目中所有 AI 协作开发活动，约束 Developer/Reviewer/Executor/Fixer 各类 Agent 以及人类开发者。

### 1. 术语与角色
- **Developer Agent**：负责编写/修改代码，必须遵守增量修改原则。  
- **Reviewer Agent**：审查代码质量、溯源、性能与安全性。  
- **Executor Agent**：执行构建、测试、基准、静态分析并归档日志。  
- **Fixer Agent**：根据 Executor 的失败日志进行修复。  
- **Baseline**：项目核心版本（含快照 SHA256）作为所有工作的锚点。  
- **Provenance Header**：标记源仓库、路径、commit、许可证等的头部注释。

### 2. 引用与溯源控制
1. 每次工作前必须运行 `tools/sync_reference_sources.sh --apply`；生成快照同时计算 SHA256 并登记 `docs/source_fusion_report.md`。  
2. 修改目标文件必须保留 Provenance Header；CI 的 `license-check` 确认 SPDX 与 `docs/license_matrix.md` 一致。  
3. `scripts/trace_snapshot.sh` 用于追溯某文件的所有上游来源。

### 3. 上下文锚定与增量编辑
1. **增量原则**：所有修改通过 diff/patch 完成；单个文件 diff 超过 80% 自动触发人工复核。  
2. 禁止新建未授权文件；CI 拦截 `_new`, `_copy`, `temp` 等命名。  
3. Prompt 强制包含：
   - 不允许新建文件/脚本  
   - 不允许输出占位/TODO/mock  
   - 必须在既有函数上做最小变更
4. 实际操作以 `ApplyPatch` 或 `git diff` 形式呈现，拒绝整段重写。

### 4. 代码质量与反退化机制
1. **禁止占位/虚拟实现**：CI 静态扫描 `TODO`, `FIXME`, `mock_`, `dummy_`, `pass` 等关键字。  
2. **CUDA 守护**：所有 CUDA 调用必须使用 `CUDA_CHECK` 包装；静态分析确认不漏检错误。  
3. **性能基准监控**：`bench_gpu.sh` 结果以 JSON/CSV 存档；当前分支性能低于上一版本 95% 自动 fail。  
4. **GPU baseline 对比**：阶段 2–4 的 GPU 算法必须与 libsecp256k1 CPU baseline 自动比对。

### 5. 测试与 CI Gate
CI 每次执行以下任务：  
1. `sync-and-license`  
2. `provenance-check`  
3. `unit-tests`  
4. `validation-smoke`（CPU/GPU 一致性）  
5. `performance-smoke`（记录并比较性能）  
6. `static-analysis`（包含 CUDA 错误检测）  
7. `coverage-check`（≥80% 必须达标）  
8. `cuda-memcheck`（随机 kernel 检测越界/未初始化）  
9. `license-check`  
10. **Mock/占位检测**（静态扫描禁词）  
11. **Diff 规模审查**（自动标注 diff>80% 文件）

### 6. 测试驱动与随机验证
1. 开发遵循 TDD；新增功能需先编写测试用例。  
2. 夜间 `run_validation.sh` 需随机生成 ≥2^20 个 Scalar256 fuzz input。  
3. `tests/unit/test_truncation_protection.cpp` 强制保护 256 位高位；新测试覆盖率低于 80% 拒绝合并。  
4. 性能趋势图每日更新，若检测退化立即报警。

### 7. 工作流程
**阶段流程**：
1. **准备**：锁定参考仓库，将 Baseline 版本号写入 `docs/source_fusion_report.md`。  
2. **开发**：Developer Agent 检索已有实现→diff 编写→提交 patch。  
3. **执行**：Executor Agent 运行 CI 全套任务，生成日志/基准报告。  
4. **修复**：如 CI 失败，Fixer Agent 根据日志修复并重新跑。  
5. **审查**：Reviewer Agent 从性能、密码学、安全性三个维度审查，并将结论写入 `docs/review_logs/PR_<id>.md`。  
6. **阶段总结**：每阶段输出：
   ```md
   ## 阶段总结
   - 已完成：
   - 修改文件：
   - 下一步：
   ```
   同时更新 Baseline 版本。

### 8. Nightly Routine & 熔断机制
1. 夜间流水线执行：`build_all.sh` → `run_unit_tests.sh` → `run_validation.sh`（随机种子）→ `bench_gpu.sh --sm-list`（追加趋势曲线）→ `package_release.sh`（存档性能/测试摘要）。  
2. GPU runner 日志自动归档并发送给 Fixer Agent；所有执行日志保留 90 天。  
3. 连续 3 次夜间构建失败：触发熔断，自动锁定主分支合并权限，直至事故处理完成并通过 CI。

### 9. Prompt 套件（可直接复用）
```text
禁止创建新文件、替换整份代码或输出占位实现；所有修改必须以最小 diff 形式呈现，并严格复用指定开源仓库的函数。请在提交前确认：
- 已在现有函数基础上修改
- 已编写/更新相关测试并确保通过
- 没有任何 TODO/mock 占位
- 性能不会低于上一版本 95%
```

### 10. 决策与审查日志
- Reviewer/Executor/Fixer 的关键结论必须写入 `docs/review_logs/PR_<id>.md`，包括：性能评估、密码学验证、安全检查、CI 结果。  
- 所有决定需要对应证据（日志、性能数据、测试截图），以备复盘。  
- CI 在每次合并前广播 ASP 协议，提醒开发者与 Agent。

### 11. 故障与应急
1. 发现截断/数据损坏等严重问题：创建 `hotfix/<issue>` 分支，立即开启运行时防护并生成 `docs/incident_<timestamp>.md`。  
2. CI 自动阻拦其他 PR 合并，直至问题修复并通过全部检查。  
3. 对历史结果运行 `detect_truncation.py`/性能回归检查，确认无扩散后解除封锁。

---

执行本方案后，AI Agent 在长周期内可保持高质量输出、严格溯源、性能安全可控，并具备可追溯、可异常熔断的工业级防错能力。
