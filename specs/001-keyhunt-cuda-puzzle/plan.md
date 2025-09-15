# Implementation Plan: Keyhunt-CUDA Scientific Research System

**Branch**: `001-keyhunt-cuda-puzzle` | **Date**: 2025-09-13 | **Spec**: [spec.md](/mnt/d/mybitcoin/puzzlekeyhunt/keyhuntcuda/specs/001-keyhunt-cuda-puzzle/spec.md)
**Input**: Feature specification from `/specs/001-keyhunt-cuda-puzzle/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
4. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
5. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, or `GEMINI.md` for Gemini CLI).
6. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
7. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
8. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
GPU-accelerated Bitcoin private key scanning system for puzzle challenges requiring scientific accuracy validation, multi-GPU support, and comprehensive performance monitoring with checkpoint/resume capabilities.

## Technical Context
**Language/Version**: C++ 17, CUDA 11.0+  
**Primary Dependencies**: CUDA Toolkit, libsecp256k1, CMake 3.18+, GoogleTest framework  
**Storage**: Checkpoint files, configuration JSON, experimental result logs  
**Testing**: GoogleTest with TDD methodology, CPU/GPU consistency validation  
**Target Platform**: Linux with NVIDIA GPU (Turing/Hopper architectures)  
**Project Type**: single (scientific research system)  
**Performance Goals**: >1000M keys/s (Turing), >4000M keys/s (Hopper), <1e-10 scientific precision  
**Constraints**: GPU memory optimization, multi-GPU coordination, checkpoint recovery <10s  
**Scale/Scope**: Bitcoin puzzle ranges, multi-GPU scaling, scientific validation framework

**User-provided Technical Context**: T026 PrivateKeyRange model already implemented with 12 passing unit tests. Source code fusion architecture combining CudaBrainSecp and BitCrack with custom KeyhuntCore framework. TDD methodology with 26/92 tasks completed (28% progress). Scientific validation using libsecp256k1 CPU reference.

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:
- Projects: 1 (KeyhuntCore scientific research system) ✓
- Using framework directly? Yes (CUDA, GoogleTest, CMake) ✓
- Single data model? Yes (PrivateKeyRange, TargetAddress, etc.) ✓
- Avoiding patterns? Yes (direct GPU operations, no unnecessary abstractions) ✓

**Architecture**:
- EVERY feature as library? Yes (KeyhuntCore modules: ecc/, scan/, compare/, utils/) ✓
- Libraries listed: ECC operations (secp256k1), Scanning framework (range processing), Address comparison (Bitcoin validation), GPU coordination (multi-device), Utilities (logging, monitoring) ✓
- CLI per library: keyhunt CLI with --help/--version/--format support ✓
- Library docs: Scientific documentation in docs/ folder ✓

**Testing (NON-NEGOTIABLE)**:
- RED-GREEN-Refactor cycle enforced? Yes (TDD methodology) ✓
- Git commits show tests before implementation? Yes (contract tests created first) ✓
- Order: Contract→Integration→E2E→Unit strictly followed? Yes (T007-T025 tests, then implementation) ✓
- Real dependencies used? Yes (actual GPU, libsecp256k1, not mocks) ✓
- Integration tests for: new libraries, contract changes, shared schemas? Yes (scientific validation tests) ✓
- FORBIDDEN: Implementation before test, skipping RED phase ✓

**Observability**:
- Structured logging included? Yes (performance metrics, validation data) ✓
- Frontend logs → backend? N/A (single system) ✓
- Error context sufficient? Yes (scientific validation, error reporting) ✓

**Versioning**:
- Version number assigned? Yes (0.1.0) ✓
- BUILD increments on every change? Yes (CMake versioning) ✓
- Breaking changes handled? Yes (parallel tests, migration plan) ✓

## Project Structure

### Documentation (this feature)
```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure]
```

**Structure Decision**: Option 1 (Single project) - Scientific research system with CUDA/C++ modules

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `/scripts/bash/update-agent-context.sh claude` for your AI assistant
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each contract → contract test task [P]
- Each entity → model creation task [P] 
- Each user story → integration test task
- Implementation tasks to make tests pass

**Ordering Strategy**:
- TDD order: Tests before implementation 
- Dependency order: Models before services before UI
- Mark [P] for parallel execution (independent files)

**Estimated Output**: 25-30 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command) ✅ 2025-09-13
- [x] Phase 1: Design complete (/plan command) ✅ 2025-09-13
- [x] Phase 2: Task planning complete (/plan command - describe approach only) ✅ 2025-09-13
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS ✅
- [x] Post-Design Constitution Check: PASS ✅
- [x] All NEEDS CLARIFICATION resolved ✅
- [x] Complexity deviations documented ✅ (None required)

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*