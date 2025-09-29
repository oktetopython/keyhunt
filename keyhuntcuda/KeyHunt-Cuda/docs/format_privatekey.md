## Private Key Serialization Format

- Internal representation: `Scalar256` little-endian limbs (`limbs[0]` is least significant word).
- Persistent storage: 32-byte big-endian encoding (MSB first) with hex or binary output depending on context.
- When exporting to WIF or bech32 formats, conversions must pass through libsecp256k1 reference routines to guarantee correctness.
- Any new persistence layer must include round-trip unit tests covering min/max values and random samples.
