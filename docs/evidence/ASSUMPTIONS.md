# Assumptions and Proofs

Purpose: Replace guesses with evidence. Each assumption must include code citations and validation method.

## A1: KeyHunt Int layout
- Claim: BISIZE=256 ⇒ NB64BLOCK=5, NB32BLOCK=10; extra limb used for algorithms; 256-bit values occupy first 4×u64.
- Citations:
  - keyhuntcuda/KeyHunt-Cuda/Int.h: `#define NB64BLOCK 5`, `#define NB32BLOCK 10` (BISIZE==256)
  - keyhuntcuda/KeyHunt-Cuda/Int.cpp: loops over NB64BLOCK, use of bits64[4]
- Validation:
  - static_assert(sizeof(Int) == NB64BLOCK*8)
  - static_assert(alignof(Int) == alignof(uint64_t))
  - Probe: print GetSize64() across generated values

## A2: gECC type parameters for secp256k1
- Claim: We must discover exact DEFINE_* macro usage (as seen for SM2 in tests) for secp256k1.
- Citations:
  - gECC-main/test/ecdsa_ec_fixed_pmul.cu: DEFINE_SM2_FP/DEFINE_FP/DEFINE_EC
- Validation:
  - Locate or author secp256k1 DEFINE_* equivalents; log file/line
  - Confirm limb count and base type (u32 vs u64), layout, mont flag

## A3: Conversion overhead matters
- Claim: Device-side packing/unpacking between u64×4 (KeyHunt) and u32×8 (gECC) may dominate gains.
- Validation:
  - Microbench: loop N=1e6 pack/unpack pairs on device; measure μs
  - Compare to original compute_ec_point_add time per op

## A4: Correctness invariants
- Claims to verify:
  - Coordinate conversions preserve equality (affine/jacobian)
  - Mod operations produce same residues as KeyHunt for test vectors
- Validation:
  - Golden vectors; invariant checks before/after conversion


