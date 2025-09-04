# gECC Architecture Map (Work-in-Progress)

Targets:
- Identify macro instantiations for secp256k1 (similar to SM2 in test/ecdsa_ec_fixed_pmul.cu lines 59-62)
- Map FpT, ECPointJacobian, EC factories, and constants for secp256k1

Known example (SM2):
- DEFINE_SM2_FP(Fq_SM2_1, FqSM2, u32, 32, LayoutT<1>, 8, MONTFLAG::SOS, CURVEFLAG::SM2)
- DEFINE_FP(Fq_SM2_n, FqSM2_n, u32, 32, LayoutT<1>, 8)
- DEFINE_EC(G1_1, G1SM2, Fq_SM2_1, SM2_CURVE, 2)
- DEFINE_ECDSA(ECDSA_EC_PMUL_Solver, G1_1_G1SM2, Fq_SM2_1, Fq_SM2_n)

Actions:
- Search repo for DEFINE_* secp256k1 usage or generation scripts (scripts/constants*.py)
- If absent, determine how to declare secp256k1 equivalents; record exact parameters (base type u32 vs u64; LIMBS 8; LayoutT<1>; mont flag)

Open Questions:
- Does gECC provide prebuilt secp256k1 macros or must we define constants ourselves?
- Which CUDA kernels expose EC add/mul interfaces suitable for substituting into KeyHunt?

