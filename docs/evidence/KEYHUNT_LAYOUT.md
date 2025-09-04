# KeyHunt Int Layout Evidence

Citations (file:line ranges approximate; verify in editor):
- Int.h: BISIZE, NB64BLOCK, NB32BLOCK macros and union { uint32_t bits[NB32BLOCK]; uint64_t bits64[NB64BLOCK]; }
- Int.cpp: CLEAR/CLEARFF/Set use NB64BLOCK; arithmetic uses bits64[0..4]; ShiftL64Bit/ShiftR64Bit iterate NB64BLOCK; IsZero has NB64BLOCK>5 branches
- IntMod.cpp: Montgomery/Div paths copy NB64BLOCK-1 limbs; temporary arrays sized [NB64BLOCK]

Findings:
- For BISIZE==256: NB64BLOCK=5, NB32BLOCK=10. Fifth limb is an extra workspace limb (Knuth div, Montgomery, ModInv). 256-bit payload fits in first 4 limbs.
- Alignment: assumed to match uint64_t; verify with static_assert and sizeof(Int).

Probes to write:
- scripts/probe/sizeof_align.cpp: prints sizeof/alignof Int; checks NB64BLOCK.
- scripts/probe/layout_dump.cpp: initializes Int with random 256-bit and dumps bits[0..4] in hex.

Edge Cases to validate:
- Sign-sensitive shifts (ShiftR64Bit uses sign for fill); ensure unsigned semantics when used as modulus residues.
- Endianness of limb order when interop with gECC (confirm low limb index == least significant 64 bits).

