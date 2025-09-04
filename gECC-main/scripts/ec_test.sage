from constants import MNT4753_r, MNT4753_q, MNT4753_g1_a, MNT4753_g1_b
from ec import G1_MNT4753

import random
import unittest


def get_xy(p):
    try:
        return p.xy()
    except ZeroDivisionError:
        return (0, 0)


class G1Test(unittest.TestCase):
    def test_g1(self):
        F = Zmod(MNT4753_q)
        E = EllipticCurve(F, [MNT4753_g1_a, MNT4753_g1_b])
        g1 = G1_MNT4753

        def test_addition(p, q):
            s = get_xy(p + q)
            p = get_xy(p)
            q = get_xy(q)
            # r = g1.add_jacobian(g1.to_jacobian(p), g1.to_jacobian(q))
            # r = g1.get_xy(r)
            # self.assertEqual(r, s)
            r = g1.add_mixed(g1.to_jacobian(p), q)
            r = g1.get_xy(r)
            self.assertEqual(r, s)

        def test_multiplication(p, n):
            s = get_xy(p * n)
            p = get_xy(p)
            r = g1.multiply_jacobian(g1.to_jacobian(p), n)
            r = g1.get_xy(r)
            self.assertEqual(r, s)

        for _ in range(1024):
            p = E.random_point()
            z = p - p
            test_addition(z, z)
            test_addition(z, p)
            test_addition(p, z)
            test_addition(p, -p)
            test_addition(p, p)
            test_addition(p, p + p)
            test_addition(p + p, p)
        # for _ in range(32):
        #     p = E.random_point()
        #     n = random.randint(0, MNT4753_r - 1)
        #     test_multiplication(p, n)
        for i in range(1024):
            test_addition(E.random_point(), E.random_point())


if __name__ == '__main__':
    unittest.main()
