def to_ptx(base_bits, code):
    assert base_bits in (32, 64)
    if base_bits == 32:
        ptx_type = 'u32'
        ptx_modifier = 'r'
    elif base_bits == 64:
        ptx_type = 'u64'
        ptx_modifier = 'l'
    dsts = [
        dst
        for _, dst, _, _ in code
    ]
    srcs = []
    for _, _, src, _ in code:
        if type(src) is str:
            srcs.append(src)
    for _, _, _, src in code:
        if type(src) is str:
            srcs.append(src)
    ptx_code = []
    for op, dst, src0, src1 in code:
        dst_repr = '%{}'.format(dsts.index(dst))
        src0_repr = '%{}'.format(
            len(dsts) + srcs.index(src0)) if type(src0) is str else str(src0)
        src1_repr = '%{}'.format(
            len(dsts) + srcs.index(src1)) if type(src1) is str else str(src1)
        ptx_code.append(
            '{}.{} {}, {}, {};'.format(op, ptx_type, dst_repr, src0_repr, src1_repr))
    dsts_repr = [
        '"={}"({})'.format(ptx_modifier, dst)
        for dst in dsts
    ]
    srcs_repr = [
        '"{}"({})'.format(ptx_modifier, src)
        for src in srcs
    ]
    return 'asm("{}" : {} : {});'.format('\\n\\t""'.join(ptx_code), ','.join(dsts_repr), ','.join(srcs_repr))


def gen_add_cy(base_bits, limbs_per_lane):
    code = [('add.cc', 'c[0]', 'a[0]', 'b[0]')]
    for i in range(1, limbs_per_lane):
        code.append(('addc.cc', 'c[{}]'.format(
            i), 'a[{}]'.format(i), 'b[{}]'.format(i)))
    code.append(('addc', 'cy', 0, 0))
    ptx = to_ptx(base_bits, code)
    return '__device__ __forceinline__ static Base add_cy(Base c[{lpl}], const Base a[{lpl}], const Base b[{lpl}]) {{ Base cy; {ptx}; return cy; }}\n'.format(
        lpl=limbs_per_lane,
        ptx=ptx
    )

def gen_add_cy0(base_bits, limbs_per_lane):
    code = [('add.cc', 'c[0]', 'a[0]', 'b')]
    for i in range(1, limbs_per_lane):
        code.append(('addc.cc', 'c[{}]'.format(
            i), 'a[{}]'.format(i), 0))
    code.append(('addc', 'cy', 0, 0))
    ptx = to_ptx(base_bits, code)
    return '__device__ __forceinline__ static Base add_cy(Base c[{lpl}], const Base a[{lpl}], Base b) {{ Base cy; {ptx}; return cy; }}\n'.format(
        lpl=limbs_per_lane,
        ptx=ptx
    )

def gen_sub_br(base_bits, limbs_per_lane):
    code = [('sub.cc', 'c[0]', 'a[0]', 'b[0]')]
    for i in range(1, limbs_per_lane):
        code.append(('subc.cc', 'c[{}]'.format(
            i), 'a[{}]'.format(i), 'b[{}]'.format(i)))
    code.append(('subc', 'br', 0, 0))
    ptx = to_ptx(base_bits, code)
    return '__device__ __forceinline__ static Base sub_br(Base c[{lpl}], const Base a[{lpl}], const Base b[{lpl}]) {{ Base br; {ptx}; return br; }}\n'.format(
        lpl=limbs_per_lane,
        ptx=ptx
    )


def gen_sub_br0(base_bits, limbs_per_lane):
    code = [('sub.cc', 'c[0]', 'a[0]', 'b')]
    for i in range(1, limbs_per_lane):
        code.append(('subc.cc', 'c[{}]'.format(
            i), 'a[{}]'.format(i), 0))
    code.append(('subc', 'br', 0, 0))
    ptx = to_ptx(base_bits, code)
    return '__device__ __forceinline__ static Base sub_br(Base c[{lpl}], const Base a[{lpl}], Base b) {{ Base br; {ptx}; return br; }}\n'.format(
        lpl=limbs_per_lane,
        ptx=ptx
    )


def gen(base_bits, limbs_per_lane):
    members = ''
    members += gen_add_cy(base_bits, limbs_per_lane)
    members += gen_add_cy0(base_bits, limbs_per_lane)
    members += gen_sub_br(base_bits, limbs_per_lane)
    members += gen_sub_br0(base_bits, limbs_per_lane)
    return 'template <> struct CC<DigitT<u{bits}, {bits}>, {lpl}> {{ using Base = typename DigitT<u{bits}, {bits}>::Base; {members} }};\n'.format(
        bits=base_bits,
        lpl=limbs_per_lane,
        members=members
    )
