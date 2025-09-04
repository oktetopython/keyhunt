from mod_sqrt import modular_sqrt
from ec_ops import EC_ADD, EC_DBL, EC_MIXED_ADD, EC_AFF_DBL, EC_DBL_1, EC_DBL_2, AFF_DBL, AFF_DBL_1, AFF_ADD

import field
import constants

import random
import unittest


class EC:
    # y^2 = x^3 + a x + b

    def __init__(self, name, f, a, b, generator):
        self.name = name
        self.field = f
        self.a = a
        self.b = b
        self.generator = generator

    def random_element(self, difficulty=None):
        if type(self.field) is field.Fp and difficulty is None:
            p = self.field.p
            while True:
                x = random.randint(0, p - 1)
                y = modular_sqrt((x * x * x + self.a * x + self.b) %
                                 p, p)
                if y != -1:
                    return x, random.choice([y, (p - y) % p])
        r = self.zero_jacobian()
        g = self.to_jacobian(self.generator)
        for i in range(difficulty or 64):
            if random.randint(0, 1) == 1:
                r = self.add_jacobian(r, g)
            g = self.add_jacobian(g, g)
        return self.get_xy(r)

    def is_equal(self, p, q):
        (x1, y1, z1) = p
        (x2, y2, z2) = q
        z1z1 = self.field.mul(z1, z1)
        z1z1z1 = self.field.mul(z1z1, z1)
        z2z2 = self.field.mul(z2, z2)
        z2z2z2 = self.field.mul(z2z2, z2)
        return self.field.mul(x1, z2z2) == self.field.mul(x2, z1z1) \
            and self.field.mul(y1, z2z2z2) == self.field.mul(y2, z1z1z1)

    # Jac -> Aff -> Bool
    def is_equal_mixed(self, p, q):
        (x1, y1, z1) = p
        (x2, y2) = q
        z1z1 = self.field.mul(z1, z1)
        z1z1z1 = self.field.mul(z1z1, z1)
        return x1 == self.field.mul(x2, z1z1) \
            and y1 == self.field.mul(y2, z1z1z1)

    def double_jacobian(self, p):
        (X1, Y1, Z1) = p
        local_vars = locals()
        EC_DBL_2.pyexec(globals(), local_vars)
        return (local_vars['X3'], local_vars['Y3'], local_vars['Z3'])

    def double_affine(self, p):
        (X1, Y1) = p
        local_vars = locals()
        EC_AFF_DBL.pyexec(globals(), local_vars)
        return (local_vars['X3'], local_vars['Y3'], local_vars['Z3'])

    def affine_double(self, p):
        (X1, Y1) = p
        local_vars = locals()
        AFF_DBL.pyexec(globals(), local_vars)
        return (local_vars['X3'], local_vars['Y3'])

    def add_affine(self, p, q):
        if self.is_zero_affine(p):
            return q
        if self.is_zero_affine(q):
            return p
        (X1, Y1) = p
        (X2, Y2) = q
        if X1 == X2:
            if Y1 == Y2:
                return self.affine_double(p)
            elif (Y1 + Y2) % self.field.p == 0:
                return self.zero_affine()
        local_vars = locals()
        AFF_ADD.pyexec(globals(), local_vars)
        return (local_vars['X3'], local_vars['Y3'])

    def add_jacobian(self, p, q):
        if self.is_zero_jacobian(p):
            return q
        if self.is_zero_jacobian(q):
            return p
        if self.is_equal(p, q):
            return self.double_jacobian(p)
        (X1, Y1, Z1) = p
        (X2, Y2, Z2) = q
        local_vars = locals()
        EC_ADD.pyexec(globals(), local_vars)
        return (local_vars['X3'], local_vars['Y3'], local_vars['Z3'])

    # Jac -> Aff -> Jac
    def add_mixed(self, p, q):
        if self.is_zero_jacobian(p):
            return self.to_jacobian(q)
        if self.is_zero_affine(q):
            return p
        if self.is_equal_mixed(p, q):
            return self.double_affine(q)
        (X1, Y1, Z1) = p
        (X2, Y2) = q
        local_vars = locals()
        EC_MIXED_ADD.pyexec(globals(), local_vars)
        return (local_vars['X3'], local_vars['Y3'], local_vars['Z3'])

    def to_jacobian(self, p):
        if self.is_zero_affine(p):
            return self.zero_jacobian()
        (x, y) = p
        z = self.field.random_nonzero_element()
        z2 = self.field.mul(z, z)
        z3 = self.field.mul(z2, z)
        return (self.field.mul(x, z2), self.field.mul(y, z3), z)

    def negate_jacobian(self, p):
        (x, y, z) = p
        return (x, self.field.neg(y), z)

    def negate_affine(self, p):
        (x, y) = p
        return (x, self.field.neg(y))

    # = [n]p
    def multiply_jacobian(self, p, n):
        result = self.zero_jacobian()
        while n > 0:
            if n % 2 == 1:
                result = self.add_jacobian(result, p)
            p = self.double_jacobian(p)
            n >>= 1
        return result

    def get_xy(self, p):
        (x, y, z) = p
        if z == self.field.zero():
            return (self.field.zero(), self.field.zero())
        z_inv = self.field.inv(z)
        z2_inv = self.field.mul(z_inv, z_inv)
        z3_inv = self.field.mul(z2_inv, z_inv)
        return (self.field.mul(x, z2_inv), self.field.mul(y, z3_inv))

    def is_zero_affine(self, p):
        (x, y) = p
        # if self.name == 'G1MNT4753':
        return y == self.field.zero()
        # else:
        #     return x == self.field.zero()

    def is_zero_jacobian(self, p):
        (_, _, z) = p
        return z == self.field.zero()

    def zero_affine(self):
        return (self.field.one(), self.field.zero())

    def zero_jacobian(self):
        return (self.field.one(), self.field.one(), self.field.zero())

    def to_mont(self, p):
        return (type(p))(map(self.field.to_mont, p))

G1_SM2 = EC('G1SM2', field.Fq_SM2,
                constants.SM2_g1_a, constants.SM2_g1_b, generator=constants.SM2_g1_generator)
G1_ECDSA_VERIFY = EC('G1ECDSA_VERIFY', field.Fq_SM2, 
                constants.SM2_g1_a, constants.SM2_g1_b, generator=constants.ECDSA_Verify_g1_generator)

def test_ec(self, ec):
    affine_p = ec.random_element()
    self.assertIsNotNone(affine_p)
    p = ec.to_jacobian(affine_p)
    self.assertIsNotNone(ec.to_mont(p))
    zero = ec.zero_jacobian()
    self.assertTrue(ec.is_equal(ec.add_jacobian(p, zero), p))
    self.assertTrue(ec.is_equal(ec.add_jacobian(zero, p), p))
    self.assertTrue(ec.is_equal(ec.add_jacobian(
        p, ec.negate_jacobian(p)), zero))
    _2p = ec.double_jacobian(p)
    _3p = ec.add_jacobian(_2p, p)
    _4p = ec.double_jacobian(_2p)
    self.assertTrue(ec.is_equal(ec.add_jacobian(
        _4p, ec.negate_jacobian(p)), _3p))

class ECTest(unittest.TestCase):
    def test_g1(self):
        test_ec(self, G1_MNT4753)

    def test_g2(self):
        test_ec(self, G2_MNT4753)


if __name__ == '__main__':
    unittest.main()
