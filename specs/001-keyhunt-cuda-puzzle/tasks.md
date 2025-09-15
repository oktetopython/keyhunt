# Tasks: Keyhunt-CUDA Scientific Research System

**Input**: Design documents from `/specs/001-keyhunt-cuda-puzzle/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → Extracted: C++17/CUDA tech stack, KeyhuntCore modular architecture
2. Load optional design documents:
   → data-model.md: 6 entities → model tasks (Private Key Range, Target Address, etc.)
   → contracts/: api-spec.yaml → 8 endpoint contract tests
   → research.md: CUDA ECC optimization → setup tasks
3. Generate tasks by category:
   → Setup: CMake project, CUDA dependencies, source fusion setup
   → Tests: contract tests, integration tests, scientific validation
   → Core: ECC kernels, scanning framework, address comparison
   → Integration: Multi-GPU coordination, checkpoint system
   → Polish: performance optimization, documentation
4. Apply task rules:
   → Different modules = mark [P] for parallel
   → Same CUDA files = sequential (no [P])
   → Tests before implementation (TDD + scientific validation)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph based on 8-phase user implementation plan
7. Create parallel execution examples
8. Validate scientific computing requirements
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions
- Scientific validation required for all ECC operations

## Path Conventions
- **Single project**: `src/KeyhuntCore/`, `tests/` at repository root
- Modular structure: `ecc/`, `scan/`, `compare/`, `utils/`
- External source fusion: `src/CudaBrainSecp/`, `src/BitCrack/`

## Phase 3.1: Project Setup & Dependencies
- [ ] T001 Create KeyhuntCore project structure per implementation plan at src/KeyhuntCore/
- [ ] T002 Initialize CMake project with CUDA 11.0+ and C++17 dependencies
- [ ] T003 [P] Configure CTest framework and scientific validation tools
- [ ] T004 [P] Set up libsecp256k1-dev integration for CPU reference validation
- [ ] T005 [P] Configure NCCL for multi-GPU communication support
- [ ] T006 [P] Set up source fusion directories for CudaBrainSecp and BitCrack analysis

## Phase 3.2: Contract Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [ ] T007 [P] Contract test POST /scan/configure in tests/contract/test_scan_configure.cpp
- [ ] T008 [P] Contract test POST /scan/start in tests/contract/test_scan_start.cpp  
- [ ] T009 [P] Contract test GET /scan/{scanId}/status in tests/contract/test_scan_status.cpp
- [ ] T010 [P] Contract test POST /scan/{scanId}/pause in tests/contract/test_scan_pause.cpp
- [ ] T011 [P] Contract test POST /scan/{scanId}/resume in tests/contract/test_scan_resume.cpp
- [ ] T012 [P] Contract test POST /targets/configure in tests/contract/test_targets_configure.cpp
- [ ] T013 [P] Contract test POST /validation/run in tests/contract/test_validation_run.cpp
- [ ] T014 [P] Contract test GET /results/matches in tests/contract/test_results_matches.cpp
- [ ] T015 [P] Contract test GET /experimental/report in tests/contract/test_experimental_report.cpp

## Phase 3.3: Scientific Validation Tests (MUST FAIL BEFORE IMPLEMENTATION)
- [ ] T016 [P] CPU/GPU consistency validation test in tests/validation/test_ecc_consistency.cpp
- [ ] T017 [P] ECC mathematical properties validation in tests/validation/test_ecc_properties.cpp  
- [ ] T018 [P] Address generation pipeline validation in tests/validation/test_address_pipeline.cpp
- [ ] T019 [P] Multi-GPU coordination validation in tests/validation/test_multi_gpu.cpp
- [ ] T020 [P] Checkpoint recovery validation in tests/validation/test_checkpoint_recovery.cpp

## Phase 3.4: Integration Tests (User Story Validation)
- [ ] T021 [P] Integration test basic private key range search in tests/integration/test_basic_range_search.cpp
- [ ] T022 [P] Integration test checkpoint and resume operations in tests/integration/test_checkpoint_resume.cpp
- [ ] T023 [P] Integration test multi-GPU configuration in tests/integration/test_multi_gpu_config.cpp
- [ ] T024 [P] Integration test real-time performance monitoring in tests/integration/test_performance_monitoring.cpp
- [ ] T025 [P] Integration test scientific validation and reporting in tests/integration/test_scientific_reporting.cpp

## Phase 3.5: Core Entity Models (Phase 1 Implementation - ONLY after tests are failing)
- [ ] T026 [P] Private Key Range model in src/KeyhuntCore/models/PrivateKeyRange.h/.cpp
- [ ] T027 [P] Target Address model in src/KeyhuntCore/models/TargetAddress.h/.cpp
- [ ] T028 [P] Checkpoint Data model in src/KeyhuntCore/models/CheckpointData.h/.cpp
- [ ] T029 [P] Experimental Results model in src/KeyhuntCore/models/ExperimentalResults.h/.cpp
- [ ] T030 [P] GPU Configuration model in src/KeyhuntCore/models/GPUConfiguration.h/.cpp
- [ ] T031 [P] Validation Report model in src/KeyhuntCore/models/ValidationReport.h/.cpp

## Phase 3.6: ECC Kernel Foundation (Phase 1.1-1.4 Implementation)
- [ ] T032 Analyze and extract CudaBrainSecp ECC kernel code from src/CudaBrainSecp/
- [ ] T033 Extract secp256k1 math operations to src/KeyhuntCore/ecc/secp256k1_math.cu
- [ ] T034 Extract point operations to src/KeyhuntCore/ecc/secp256k1_point.cu  
- [ ] T035 Create ECC interface header in src/KeyhuntCore/ecc/secp256k1.h
- [ ] T036 Fix compilation errors and missing dependencies in ECC modules
- [ ] T037 Implement CPU reference validation using libsecp256k1 in src/KeyhuntCore/ecc/secp256k1_cpu.cpp
- [ ] T038 Complete CPU/GPU consistency validation testing for ECC operations
- [ ] T039 Validate ECC mathematical properties and edge cases

## Phase 3.7: Scanning Framework Foundation (Phase 2.1-2.4 Implementation)  
- [ ] T040 Analyze BitCrack scanning framework architecture from src/BitCrack/
- [ ] T041 Design range scanning interface in src/KeyhuntCore/scan/scanner.h
- [ ] T042 Implement private key range scanning in src/KeyhuntCore/scan/scanner.cu
- [ ] T043 Build GPU kernel for parallel key scanning in src/KeyhuntCore/scan/scan_kernel.cu
- [ ] T044 Implement checkpoint system in src/KeyhuntCore/scan/checkpoint.cpp
- [ ] T045 Create scan recovery mechanisms in src/KeyhuntCore/scan/recovery.cpp

## Phase 3.8: Address Generation and Comparison (Phase 3.1-3.4 Implementation)
- [ ] T046 Analyze BitCrack address generation and comparison logic
- [ ] T047 Implement Bitcoin address generation pipeline in src/KeyhuntCore/compare/address_gen.cu
- [ ] T048 Create hash operations (SHA256/RIPEMD160) in src/KeyhuntCore/compare/hash.cu
- [ ] T049 Build single target address comparison in src/KeyhuntCore/compare/single_compare.cu
- [ ] T050 Create multi-target comparison with Bloom Filter in src/KeyhuntCore/compare/bloom_filter.cu
- [ ] T051 Implement Base58 encoding for address generation in src/KeyhuntCore/compare/base58.cu

## Phase 3.9: Performance Optimization (Phase 4.1-4.3 Implementation)
- [ ] T052 Optimize GPU kernel resource utilization in existing CUDA files
- [ ] T053 Implement batch processing optimizations for ECC operations
- [ ] T054 Profile and measure performance improvements using NVIDIA tools
- [ ] T055 Optimize memory coalescing patterns in GPU kernels
- [ ] T056 Implement register pressure optimization strategies

## Phase 3.10: Multi-GPU Support (Phase 5.1-5.3 Implementation)
- [ ] T057 Create multi-GPU coordination system in src/KeyhuntCore/gpu/coordinator.cpp
- [ ] T058 Implement dynamic load balancing using NCCL in src/KeyhuntCore/gpu/load_balancer.cu
- [ ] T059 Add result aggregation and synchronization in src/KeyhuntCore/gpu/aggregator.cpp
- [ ] T060 Implement CUDA streams for communication-computation overlap

## Phase 3.11: Multi-target and Batch Processing (Phase 6.1-6.3 Implementation)
- [ ] T061 Scale Bloom Filter for large target sets in src/KeyhuntCore/compare/bloom_filter.cu
- [ ] T062 Implement multi-range scanning capability in src/KeyhuntCore/scan/multi_range.cu
- [ ] T063 Create comprehensive logging in src/KeyhuntCore/utils/logger.cpp
- [ ] T064 Implement performance monitoring in src/KeyhuntCore/utils/monitor.cpp

## Phase 3.12: System Integration and API Implementation
- [ ] T065 Implement scan configuration endpoint logic
- [ ] T066 Implement scan start/pause/resume endpoint logic  
- [ ] T067 Implement target configuration endpoint logic
- [ ] T068 Implement validation run endpoint logic
- [ ] T069 Implement results retrieval endpoint logic
- [ ] T070 Implement experimental report generation endpoint logic
- [ ] T071 Connect models to services for data persistence
- [ ] T072 Add comprehensive error handling and logging

## Phase 3.13: Validation and Documentation (Phase 7.1-7.3 Implementation)
- [ ] T073 Generate performance analysis and benchmarks
- [ ] T074 Create scientific validation documentation
- [ ] T075 Document system architecture and usage  
- [ ] T076 Create CLI interface in src/KeyhuntCore/cli/keyhunt_cli.cpp
- [ ] T077 [P] Update CLAUDE.md with implementation details
- [ ] T078 [P] Create user manual documentation

## Phase 3.14: Integration Testing and Validation (Phase 8.1-8.4 Implementation)
- [ ] T079 Validate source code fusion effectiveness
- [ ] T080 Execute end-to-end pipeline validation
- [ ] T081 Test checkpoint and recovery mechanisms  
- [ ] T082 Perform extended stability and stress testing
- [ ] T083 Run complete quickstart guide validation
- [ ] T084 Verify all functional requirements (FR-001 through FR-012)

## Phase 3.15: Polish and Optimization
- [ ] T085 [P] Unit tests for utility functions in tests/unit/test_utils.cpp
- [ ] T086 [P] Unit tests for models in tests/unit/test_models.cpp
- [ ] T087 Performance optimization for >1000M keys/s target (Turing)
- [ ] T088 Performance optimization for >4000M keys/s target (Hopper)
- [ ] T089 Memory usage optimization for <90% GPU utilization
- [ ] T090 Code cleanup and duplicate removal
- [ ] T091 Final scientific validation with 1M+ samples
- [ ] T092 Create deployment and installation scripts

## Dependencies
**Critical Path Dependencies**:
- Setup (T001-T006) before all other phases
- Contract tests (T007-T015) before any implementation
- Scientific validation tests (T016-T020) before ECC implementation  
- Integration tests (T021-T025) before system integration
- Models (T026-T031) before services and APIs
- ECC foundation (T032-T039) blocks scanning framework (T040-T045)
- Scanning framework blocks address comparison (T046-T051)
- Core implementation before performance optimization (T052-T056)
- Single-GPU before multi-GPU (T057-T060)
- All core functionality before API implementation (T065-T072)
- Implementation before validation and documentation (T073-T084)

**Parallel Execution Blocks**:
- Contract tests (T007-T015) can run in parallel
- Scientific validation tests (T016-T020) can run in parallel
- Integration tests (T021-T025) can run in parallel  
- Entity models (T026-T031) can run in parallel
- Different module files can be developed in parallel within phases

## Parallel Execution Examples

### Phase 1 - Contract Tests (All Must Fail First)
```bash
# Launch T007-T015 together:
Task: "Contract test POST /scan/configure in tests/contract/test_scan_configure.cpp"
Task: "Contract test POST /scan/start in tests/contract/test_scan_start.cpp"
Task: "Contract test GET /scan/{scanId}/status in tests/contract/test_scan_status.cpp"
Task: "Contract test POST /targets/configure in tests/contract/test_targets_configure.cpp"
Task: "Contract test POST /validation/run in tests/contract/test_validation_run.cpp"
```

### Phase 2 - Scientific Validation Tests
```bash
# Launch T016-T020 together:
Task: "CPU/GPU consistency validation test in tests/validation/test_ecc_consistency.cpp"
Task: "ECC mathematical properties validation in tests/validation/test_ecc_properties.cpp"
Task: "Address generation pipeline validation in tests/validation/test_address_pipeline.cpp"
Task: "Multi-GPU coordination validation in tests/validation/test_multi_gpu.cpp"
```

### Phase 3 - Entity Models
```bash
# Launch T026-T031 together:
Task: "Private Key Range model in src/KeyhuntCore/models/PrivateKeyRange.h/.cpp"
Task: "Target Address model in src/KeyhuntCore/models/TargetAddress.h/.cpp"
Task: "Checkpoint Data model in src/KeyhuntCore/models/CheckpointData.h/.cpp"
Task: "GPU Configuration model in src/KeyhuntCore/models/GPUConfiguration.h/.cpp"
```

## Scientific Computing Requirements
- **Precision Validation**: All ECC operations must achieve <1e-10 relative error
- **CPU Reference**: libsecp256k1 used as authoritative reference for all GPU computations
- **Performance Targets**: >1000M keys/s (Turing), >4000M keys/s (Hopper)
- **GPU Utilization**: >90% target during scanning operations
- **Test-First Development**: All tests must fail before implementation begins
- **Source Code Fusion**: Systematic extraction and validation from CudaBrainSecp/BitCrack

## Validation Checklist
*GATE: Checked before execution*

- [x] All contracts have corresponding tests (T007-T015)
- [x] All entities have model tasks (T026-T031)
- [x] All tests come before implementation (Phases 3.2-3.4 before 3.5+)
- [x] Parallel tasks truly independent (different files/modules)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task
- [x] Scientific validation requirements integrated
- [x] 8-phase user implementation plan incorporated
- [x] Performance targets and constraints specified
- [x] Source code fusion approach documented

## Notes
- [P] tasks = different files/modules, no dependencies
- Verify tests fail before implementing (TDD + scientific validation)
- Commit after each task with clear phase identification
- GPU memory constraints must be considered for large batch sizes
- All CUDA kernels require corresponding CPU validation
- Multi-GPU coordination requires careful synchronization testing
- Performance optimization is iterative and measurement-driven
- Scientific accuracy takes precedence over performance optimization