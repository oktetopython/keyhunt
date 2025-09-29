## Baseline Environment Setup

1. **Install prerequisites**
   - CUDA Toolkit ≥ 11.4
   - Python 3.9+ with `pytest` (for tool smoke tests)
   - `git`, `sha256sum`, and `nvidia-smi` (when available)

2. **Synchronise reference sources**
   ```bash
   tools/sync_reference_sources.sh --apply
   ```
   The command creates a timestamped snapshot beneath `src/reference_snapshots/` and appends the digest/commit pair to `docs/source_fusion_report.md`.

3. **Detect GPU capability**
   ```bash
   tools/gpu_capability.sh
   ```
   Outputs a JSON report detailing installed GPUs and compute capability. If `nvidia-smi` is unavailable the script exits cleanly with an informational message.

4. **Run baseline smoke tests**
   ```bash
   python -m pytest tests/tools
   ```
   Ensures the helper scripts operate correctly before further development.
