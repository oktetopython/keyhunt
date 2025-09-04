# Project Status – KeyHunt ⇄ gECC Integration (Pause)

Updated: 2025-08-30
Owner: Augment Agent (with PM oversight)

## Executive Summary (中文)
- 抽象层不匹配：过程式 vs 泛型模板编程（gECC）
- 数据表示差异：同为256位但内部布局/含义不同（KeyHunt u64×5 vs gECC u32×8、坐标系/域表示差异）
- 算法范式差异：KeyHunt单点/流水方式 vs gECC批量优化/表驱动
- 优化上下文依赖：gECC优化在其类型系统/布局/批处理上下文中才成立
- 依赖复杂度：抽取单一函数往往牵动工厂、常量、宏与布局的整体体系

路线图（依据您的指导）

- 短期（1–2周）：
  1) 深入理解两个库（代码与文档，不进行集成）
  2) 性能剖析KeyHunt，找出真实瓶颈
  3) 算法研究：理解gECC优化原理（不照搬实现）

- 中期（2–4周）：
  1) 算法级集成：在KeyHunt现有表示上“按原理重写”
  2) 渐进式优化：每步优化配套正确性与性能验证
  3) 性能验证：每步都有量化收益

- 长期（1–2月）：
  1) 深度优化（充分理解后）
  2) 必要时调整KeyHunt架构以承载优化

## 1) Current Decision
- Project is PAUSED to prevent risky integration without sufficient evidence.
- Pivot to evidence-first analysis of gECC and KeyHunt compatibility.

## 2) Honest Baseline
- Correctness: KeyHunt works; our custom ECC attempts are unreliable. gECC is research-grade with different types/layouts.
- Risk: Data layout/ABI mismatch (KeyHunt uint64[5] vs gECC u32×8), CUDA memory alignment, coordinate-system differences, performance regression due to conversion overhead.
- Unknowns: Exact gECC template instantiation for secp256k1; verified per-config KeyHunt NB64BLOCK; net cost of conversions.

## 3) Evidence Log Plan (files and artifacts)
- docs/evidence/ASSUMPTIONS.md – Hypotheses, static_asserts, file/line citations.
- docs/evidence/GECC_MAP.md – gECC types/macros (DEFINE_*), constants, CUDA entry points with file/line refs.
- docs/evidence/KEYHUNT_LAYOUT.md – Int layout, NB64BLOCK/NB32BLOCK per-BISIZE; alignment; role of extra limb.
- docs/evidence/PERF_PLAN.md – Bench methodology: original vs. adapter vs. gECC path; conversion cost isolation.
- scripts/probe/ – Small probes to print sizes/alignments and run device asserts (no integration yet).

## 4) Today (Done/To‑do)
- [ ] Notify stakeholders: project paused (template below).
- [ ] Download/pin gECC full source (exact commit recorded) – no integration.
- [ ] Create evidence skeleton files and probe scripts.

## 5) This Week
- Analyze gECC architecture/API from source; capture DEFINE_* for secp256k1, field/order, EC types.
- Verify KeyHunt memory layout across configs (BISIZE=256/512), show code citations and static_asserts.
- Produce detailed evidence collection plan and acceptance criteria.

## 6) Early Next Week
- Choose path based on evidence: Full integration vs. partial vs. learn-and-port vs. defer.
- Draft new plan with milestones, exit criteria, and rollback.
- Define success metrics: correctness (test vectors/invariants), performance thresholds, and stop conditions.

## 7) Stakeholder Pause Notice (Template)
Subject: Pause – KeyHunt ⇄ gECC Integration (Evidence-First Pivot)

Hi all,

We are pausing the KeyHunt ⇄ gECC integration effective immediately to avoid high-risk changes without sufficient evidence. Root causes: architecture mismatch (types/layouts), potential CUDA ABI and coordinate-system differences, and unproven performance benefits once conversion overhead is included.

Next steps this week:
- Map gECC types/macros and CUDA APIs with citations
- Verify KeyHunt Int layout across build configs with static_asserts
- Produce an evidence-backed comparison and decision brief

We will return early next week with an evidence-based plan, success criteria, and rollback options.

– Owner

## 8) Concrete Commands (Do not run in CI; local only)
# Clone and pin gECC (confirm path with repo owner)
# Option A: third_party
#   mkdir -p third_party && cd third_party
#   git clone https://github.com/…/gECC.git gECC
#   cd gECC && git rev-parse HEAD > COMMIT_PIN.txt
# Option B: external (if policy prefers)
#   mkdir -p external && cd external
#   git clone …

## 9) Guardrails
- No integration builds this week. No dependency installs without approval.
- All claims must have code citations (file:line) and/or static_assert proof.
- Any CUDA device code change must come with a minimal, reproducible micro-benchmark.

## 10) Open Questions
- Exact gECC macros for secp256k1 (similar to SM2 in tests) and their limb/layout parameters
- Measurable conversion overhead device-side vs. host-side packing
- Whether partial adoption (algorithms-only) beats full template/type adoption

## 11) Appendix – Known Citations to Validate
- KeyHunt Int layout macros: keyhuntcuda/KeyHunt-Cuda/Int.h (NB64BLOCK, NB32BLOCK, BISIZE)
- KeyHunt use of bits64 across code paths (Int.cpp, IntMod.cpp)
- gECC macro usage examples (test/ecdsa_ec_fixed_pmul.cu – DEFINE_* lines)


