## Endianness Policy

- All scalar values stored in `Scalar256` use little-endian limb ordering (`uint64_t limbs[4]`).
- External interfaces (files, network, shared snapshots) serialize 256-bit integers as 32-byte big-endian sequences to remain compatible with Bitcoin standards.
- Tests must verify little↔big endian conversions, ensuring no truncation or order inversion when exchanging data with CPU-side reference implementations.
