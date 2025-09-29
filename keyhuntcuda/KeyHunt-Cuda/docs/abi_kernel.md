## GPU Kernel ABI Overview

- Primary entry point: `__global__ void scan_kernel(const uint64_t *priv_limbs, size_t n);`
  - `priv_limbs` points to contiguous little-endian `Scalar256` values (4 limbs per scalar).
  - Kernel implementations must accept any `n >= 0`; zero-length invocations perform a no-op.
- All device helper functions require the signature `__device__ __forceinline__` to minimise call overhead and enforce compile-time inlining.
- Warp-synchronous primitives must rely on cooperative groups (`__shfl_sync`, `__ballot_sync`, `__syncwarp`) with explicit masks to maintain forward compatibility across compute capabilities.
- Host code must upload constants via cudaMemcpyToSymbol or explicit buffers; `__constant__` memory is reserved for small shared tables (≤64 KiB).
