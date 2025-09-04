import field
import ec
import ec_ops
import ccgen

import argparse
import pathlib

import random

import math


class CRepr:
    width = 64
    def repr_list(self, lst):
        return '{' + ','.join(lst) + '}'

    def fp(self, n):
        b = []
        while n > 0:
            b.append(n & ((
                ~(
                    (0xFFFFFFFFFFFFFFFF << self.width) & 0xFFFFFFFFFFFFFFFF
                    )
            ) & 0xFFFFFFFFFFFFFFFF))
            n >>= self.width
        return self.repr_list(map(str, b))

    def fp2(self, x):
        a, b = x
        return self.repr_list([self.fp(a), self.fp(b)])

    def fp_array(self, ns):
        return self.repr_list(map(self.fp, ns))

    def fp2_array(self, ns):
        return self.repr_list(map(self.fp2, ns))

    def ec_component(self, c):
        if type(c) is not tuple:
            c = (c,)
        return self.repr_list(map(self.fp, c))

    def ec_point(self, p):
        return self.repr_list(map(self.ec_component, p))

    def ec_array(self, points):
        return self.repr_list(map(self.ec_point, points))

    def fp_constant(self, f):
        return '''static constexpr FpConstant {name} {{
            .bits = {bits},
            .rexp = {rexp},
            .pinv = {pinv}u,
            .p = {p},
            .p_minus_2 = {p_minus_2},
            .pp = {pp},
            .r = {r},
            .r2 = {r2},
            .adicity2 = {adicity2},
            .generator={generator},
            .inv2={inv2},
        }};'''.format(
            name=f.name,
            bits=f.bits,
            rexp=f.rexp,
            pinv=f.pinv(f.width),
            p=self.fp(f.p),
            p_minus_2=self.fp(f.p - 2),
            pp=self.fp(f.pp),
            r=self.fp(f.r),
            r2=self.fp(f.r2),
            adicity2=f.adicity2,
            generator=self.fp(f.generator),
            inv2=self.fp(f.inv2),
        )

    @staticmethod
    def fp2_constant(f):
        return '''static constexpr Fp2Constant {name} {{
            .alpha = {alpha},
        }};'''.format(
            name=f.name,
            alpha=f.alpha,
        )

    # @staticmethod
    def ec_constant(self, e):
        return '''static constexpr ECConstant {name} {{
            .a = {a},
            .a_mont = {a_mont},
        }};'''.format(
            name=e.name,
            a=e.a,
            a_mont=self.fp(e.field.to_mont(e.a)),
        )

    @staticmethod
    def ec2_constant(e):
        assert e.a[1] == 0
        return '''static constexpr ECConstant {name} {{
            .a = {a},
        }};'''.format(
            name=e.name,
            a=e.a[0],
        )

    # @staticmethod
    def ecdsa_constant(self, ec):
        k = math.ceil(math.log(ec.field.bits, 2))
        n = 1 << k
        p = ec.generator
        
        sig_affine = [None] * (n)
        for i in range(n):
            sig_affine[i] = p
            p = ec.affine_double(p)

        return '''static constexpr ECDSAConstant G1_1_{name} {{
            .K = {K},
            .SIG_AFF = {SIG_AFF},
        }};'''.format(
            name=ec.name,
            K=k,
            SIG_AFF=self.ec_array(map(ec.to_mont, sig_affine)),
        )

def generate_fp_test(out, name, f, k, width):
    n = 1 << k
    a = [random.randint(0, f.p - 1) for i in range(n)]
    b = [random.randint(0, f.p - 1) for i in range(n)]

    crepr = CRepr()
    crepr.width = width

    out.write('namespace {}_fp_test {{\n'.format(name))
    out.write('static const size_t N = {};\n'.format(n))
    out.write('static const uint64_t A[{}][MAX_LIMBS] = {};\n'.format(
        n, crepr.fp_array(a)))
    out.write('static const uint64_t B[{}][MAX_LIMBS] = {};\n'.format(
        n, crepr.fp_array(b)))
    out.write('static const uint64_t SUM[{}][MAX_LIMBS] = {};\n'.format(
        n, crepr.fp_array([(a[i] + b[i]) % f.p for i in range(n)])))
    out.write('static const uint64_t PROD[{}][MAX_LIMBS] = {};\n'.format(
        n, crepr.fp_array([a[i] * b[i] % f.p for i in range(n)])))
    # out.write('static const uint64_t PROD[{}][MAX_LIMBS] = {};\n'.format(
    #     n, crepr.fp_array([f.to_mont(a[i] * b[i] % f.p) for i in range(n)])))
    out.write('}}\n\n'.format(name))

    # pass test
    # print(f.to_bytes(f.to_mont(a[0])))
    # print(f.to_bytes(f.to_mont(b[0])))
    # f.mont_mul(f.to_bytes(f.to_mont(a[0])), f.to_bytes(f.to_mont(b[0])), f.to_bytes(f.p))

def generate_ecdsa_test(out, f, width):
    e = 0x10D51CB90C0C0522E94875A2BEA7AB72299EBE7192E64EFE0573B1C77110E5C9
    priv_key = 0x128B2FA8BD433C6C068C8D803DFF79792A519A55171B1B650C23661D15897263
    k = 0xE11F5909F947D5BE08C84A22CE9F7C338F7CF4A5B941B9268025495D7D433071
    s = 0xE11F5909F947D5BE08C84A22CE9F7C338F7CF4A5B941B9268025495D7D433071
    r = 0x23B20B796AAAFEAAA3F1592CB9B4A93D5A8D279843E1C57980E64E0ABC5F5B05
    # r = 0x23B20B796AAAFEAAA3F1592CB9B4A93D5A8D279843E1C57980E64E0ABC5F5B96
    key_x = 0xD5548C7825CBB56150A3506CD57464AF8A1AE0519DFAF3C58221DC810CAF28DD
    key_y = 0x921073768FE3D59CE54E79A49445CF73FED23086537027264D168946D479533E
    
    n = 3972
    random_r = [random.randint(0, f.p - 1)  for i in range(n)]
    random_s = [random.randint(0, f.p - 1)  for i in range(n)]
    random_e = [random.randint(0, f.p - 1)  for i in range(n)]
    random_priv_key = [random.randint(0, f.p - 1)  for i in range(n)]
    random_k = [random.randint(0, f.p - 1)  for i in range(n)]
    random_key_x = [random.randint(0, f.p - 1)  for i in range(n)]
    random_key_y = [random.randint(0, f.p - 1)  for i in range(n)]
    
    crepr = CRepr()
    crepr.width = width

    out.write('static const uint64_t E[MAX_LIMBS] = {};\n'.format(
        crepr.fp(e)))
    out.write('static const uint64_t PRIV_KEY[MAX_LIMBS] = {};\n'.format(
        crepr.fp(priv_key)))
    out.write('static const uint64_t K[MAX_LIMBS] = {};\n'.format(
        crepr.fp(k)))
    out.write('static const uint64_t S[MAX_LIMBS] = {};\n'.format(
        crepr.fp(s)))
    out.write('static const uint64_t R[MAX_LIMBS] = {};\n'.format(
        crepr.fp(r)))
    out.write('static const uint64_t KEY_X[MAX_LIMBS] = {};\n'.format(
        crepr.fp(key_x)))
    out.write('static const uint64_t KEY_Y[MAX_LIMBS] = {};\n'.format(
        crepr.fp(key_y)))
    out.write('static const uint64_t RANDOM_R[{}][MAX_LIMBS] = {};\n'.format(
        n, crepr.fp_array(random_r)))
    out.write('static const uint64_t RANDOM_S[{}][MAX_LIMBS] = {};\n'.format(
        n, crepr.fp_array(random_s)))
    out.write('static const uint64_t RANDOM_E[{}][MAX_LIMBS] = {};\n'.format(
        n, crepr.fp_array(random_e)))
    out.write('static const uint64_t RANDOM_PRIV_KEY[{}][MAX_LIMBS] = {};\n'.format(
        n, crepr.fp_array(random_priv_key)))
    out.write('static const uint64_t RANDOM_K[{}][MAX_LIMBS] = {};\n'.format(
        n, crepr.fp_array(random_k)))
    out.write('static const uint64_t RANDOM_KEY_X[{}][MAX_LIMBS] = {};\n'.format(
        n, crepr.fp_array(random_key_x)))
    out.write('static const uint64_t RANDOM_KEY_Y[{}][MAX_LIMBS] = {};\n'.format(
        n, crepr.fp_array(random_key_y)))


# def generate_ecdsa_test(out, f, ec, width):
#     e = 0x10D51CB90C0C0522E94875A2BEA7AB72299EBE7192E64EFE0573B1C77110E5C9
#     priv_key = 0x128B2FA8BD433C6C068C8D803DFF79792A519A55171B1B650C23661D15897263
#     k = 0xE11F5909F947D5BE08C84A22CE9F7C338F7CF4A5B941B9268025495D7D433071
#     s = 0xE11F5909F947D5BE08C84A22CE9F7C338F7CF4A5B941B9268025495D7D433071
#     r = 0x23B20B796AAAFEAAA3F1592CB9B4A93D5A8D279843E1C57980E64E0ABC5F5B05
#     key_x = 0xD5548C7825CBB56150A3506CD57464AF8A1AE0519DFAF3C58221DC810CAF28DD
#     key_y = 0x921073768FE3D59CE54E79A49445CF73FED23086537027264D168946D479533E
    
#     n = 1 << 10
#     random_r = [random.randint(0, f.p - 1)  for i in range(n)]
#     random_s = [random.randint(0, f.p - 1)  for i in range(n)]
#     random_e = [random.randint(0, f.p - 1)  for i in range(n)]
#     random_priv_key = [random.randint(0, f.p - 1)  for i in range(n)]
#     random_k = [random.randint(0, f.p - 1)  for i in range(n)]
#     distinct_bases = 1<<6
#     base_aff = [
#         ec.random_element()
#         for _ in range(distinct_bases)
#     ]
#     base_id = [random.randint(0, distinct_bases - 1) for i in range(n)]
#     random_key_x = []
#     random_key_y = []
#     for i in range(n):
#         x, y = base_aff[base_id[i]]
#         random_key_x.append(x)
#         random_key_y.append(y)
    
#     crepr = CRepr()
#     crepr.width = width

#     out.write('static const uint64_t E[MAX_LIMBS] = {};\n'.format(
#         crepr.fp(e)))
#     out.write('static const uint64_t PRIV_KEY[MAX_LIMBS] = {};\n'.format(
#         crepr.fp(priv_key)))
#     out.write('static const uint64_t K[MAX_LIMBS] = {};\n'.format(
#         crepr.fp(k)))
#     out.write('static const uint64_t S[MAX_LIMBS] = {};\n'.format(
#         crepr.fp(s)))
#     out.write('static const uint64_t R[MAX_LIMBS] = {};\n'.format(
#         crepr.fp(r)))
#     out.write('static const uint64_t KEY_X[MAX_LIMBS] = {};\n'.format(
#         crepr.fp(key_x)))
#     out.write('static const uint64_t KEY_Y[MAX_LIMBS] = {};\n'.format(
#         crepr.fp(key_y)))
#     out.write('static const uint64_t RANDOM_R[{}][MAX_LIMBS] = {};\n'.format(
#         n, crepr.fp_array(random_r)))
#     out.write('static const uint64_t RANDOM_S[{}][MAX_LIMBS] = {};\n'.format(
#         n, crepr.fp_array(random_s)))
#     out.write('static const uint64_t RANDOM_E[{}][MAX_LIMBS] = {};\n'.format(
#         n, crepr.fp_array(random_e)))
#     out.write('static const uint64_t RANDOM_PRIV_KEY[{}][MAX_LIMBS] = {};\n'.format(
#         n, crepr.fp_array(random_priv_key)))
#     out.write('static const uint64_t RANDOM_K[{}][MAX_LIMBS] = {};\n'.format(
#         n, crepr.fp_array(random_k)))
#     out.write('static const uint64_t RANDOM_KEY_X[{}][MAX_LIMBS] = {};\n'.format(
#         n, crepr.fp_array(random_key_x)))
#     out.write('static const uint64_t RANDOM_KEY_Y[{}][MAX_LIMBS] = {};\n'.format(
#         n, crepr.fp_array(random_key_y)))

random.seed(233)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()
    root = pathlib.Path(args.out)

    root.mkdir(exist_ok=True)

    with open(root / 'fp_constants.h', 'w') as f:
        crepr_64 = CRepr()
        crepr_64.width = 64
        f.write(crepr_64.fp_constant(field.Fq_SM2) + '\n')
        f.write(crepr_64.fp_constant(field.Fq_SM2_n) + '\n')

    with open(root / 'ec_constants.h', 'w') as f:
        crepr_64 = CRepr()
        crepr_64.width = 64
        f.write(crepr_64.ec_constant(ec.G1_SM2) + '\n')
        # f.write(crepr_64.ec_constant(ec.G1_ECDSA_VERIFY) + '\n')

    with open(root / 'ecdsa_constants.h', 'w') as f:
        crepr_64 = CRepr()
        crepr_64.width = 64
        f.write(crepr_64.ecdsa_constant(ec.G1_SM2) + '\n')
        # f.write(crepr_64.ecdsa_constant(ec.G1_ECDSA_VERIFY) + '\n')

    with open(root / 'fp_ops_cc_details.h', 'w') as f:
        for limbs_per_lane in range(1, 17):
            f.write(ccgen.gen(64, limbs_per_lane))
        for limbs_per_lane in range(1, 33):
            f.write(ccgen.gen(32, limbs_per_lane))

    with open(root / 'ec_ops_add_details.h', 'w') as f:
        f.write(ec_ops.EC_ADD.cucode() + '\n')
        
    with open(root / 'ec_ops_add_details_with_reduce.h', 'w') as f:
        f.write(ec_ops.EC_ADD.cucode_with_reduce() + '\n')

    with open(root / 'ec_ops_dbl_details.h', 'w') as f:
        f.write(ec_ops.EC_DBL.cucode() + '\n')
        
    with open(root / 'ec_ops_dbl_details_with_reduce.h', 'w') as f:
        f.write(ec_ops.EC_DBL.cucode_with_reduce() + '\n')

    with open(root / 'ec_ops_dbl_1_details.h', 'w') as f:
        f.write(ec_ops.EC_DBL_1.cucode() + '\n')

    with open(root / 'ec_ops_dbl_1_details_with_reduce.h', 'w') as f:
        f.write(ec_ops.EC_DBL_1.cucode_with_reduce() + '\n')

    with open(root / 'ec_ops_dbl_2_details.h', 'w') as f:
        f.write(ec_ops.EC_DBL_2.cucode() + '\n')

    with open(root / 'ec_ops_dbl_2_details_with_reduce.h', 'w') as f:
        f.write(ec_ops.EC_DBL_2.cucode_with_reduce() + '\n')

    with open(root / 'ec_ops_mixed_add_details.h', 'w') as f:
        f.write(ec_ops.EC_MIXED_ADD.cucode() + '\n')

    with open(root / 'ec_ops_mixed_add_details_with_reduce.h', 'w') as f:
        f.write(ec_ops.EC_MIXED_ADD.cucode_with_reduce() + '\n')

    with open(root / 'ec_ops_affine_dbl_details.h', 'w') as f:
        f.write(ec_ops.EC_AFF_DBL.cucode() + '\n')

    with open(root / 'ec_ops_affine_dbl_details_with_reduce.h', 'w') as f:
        f.write(ec_ops.EC_AFF_DBL.cucode_with_reduce() + '\n')

    with open(root / 'affine_ops_dbl_details.h', 'w') as f:
        f.write(ec_ops.AFF_DBL.cucode(0) + '\n')

    with open(root / 'affine_ops_dbl_details_with_reduce.h', 'w') as f:
        f.write(ec_ops.AFF_DBL.cucode_with_reduce(0) + '\n')

    with open(root / 'affine_ops_add_details.h', 'w') as f:
        f.write(ec_ops.AFF_ADD.cucode(0) + '\n')

    with open(root / 'affine_ops_add_details_with_reduce.h', 'w') as f:
        f.write(ec_ops.AFF_ADD.cucode_with_reduce(0) + '\n')

    # tests
    with open(root / 'fp_test_constants.h', 'w') as f:
        generate_fp_test(f, field.Fq_SM2.name, field.Fq_SM2, 6, field.Fq_SM2.width)
        generate_fp_test(f, field.Fq_SM2_n.name, field.Fq_SM2_n, 6, field.Fq_SM2_n.width)


    with open(root / 'ecdsa_test_constants.h', 'w') as f:
        generate_ecdsa_test(
            f, field.Fq_SM2_n, field.Fq_SM2_n.width)
        # generate_ecdsa_test(
        #     f, field.Fq_SM2, ec.G1_SM2, field.Fq_SM2.width)
