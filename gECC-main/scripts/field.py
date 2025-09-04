from constants import SM2_q, SM2_n

import random


class Fp:
    def __init__(self, name, p, width, rexp):
        self.degree = 1
        self.width = width
        self.rexp = rexp
        r = 2 ** (self.width * rexp)
        assert p < r
        self.name = name
        self.bits = 0
        while 2 ** self.bits <= p:
            self.bits += 1
        self.p = p
        self.rexp = rexp
        self.r = r % p
        self.r2 = r * r % p
        self.rinv = pow(r, p - 2, p)
        self.adicity2 = 0
        while (p - 1) % (2 ** (self.adicity2 + 1)) == 0:
            self.adicity2 += 1
        g = 2
        while pow(g, (p - 1) // 2, p) == 1:
            g += 1
        self.generator = pow(g, (p - 1) // (2 ** self.adicity2), p)
        assert(pow(self.generator, 2 ** (self.adicity2 - 1), p) != 1)
        assert(pow(self.generator, 2 ** self.adicity2, p) == 1)
        self.inv2 = (p + 1) // 2
        self.high_mask = (0xFFFFFFFFFFFFFFFF << self.width) & 0xFFFFFFFFFFFFFFFF
        self.low_mask = (~((0xFFFFFFFFFFFFFFFF << self.width) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF

    @property
    def pp(self):
        return self.p * 2

    @property
    def count(self):
        return 1

    def pinv(self, width):
        # assert width in (32, 64)
        return 2 ** width - pow(self.p, 2 ** (width - 1) - 1, 2 ** width)

    def to_mont(self, x):
        return x * self.r % self.p

    def from_mont(self, x):
        return x * pow(self.r, self.p - 2, self.p) % self.p

    # ops for ec

    def add(self, x, y):
        return (x + y) % self.p

    def neg(self, x):
        return (self.p - x) % self.p

    def sub(self, x, y):
        return (x + self.p - y) % self.p

    def mul(self, x, y):
        return (x * y) % self.p

    def inv(self, x):
        if x == self.zero():
            raise ZeroDivisionError()
        return pow(x, self.p - 2, self.p)

    def one(self):
        return 1

    def zero(self):
        return 0

    def random_element(self):
        return random.randint(0, self.p - 1)

    def random_nonzero_element(self):
        return random.randint(1, self.p - 1)

    def to_bytes(self, x):
        return [(x >> (i * self.width)) % (2 ** self.width) for i in range(self.rexp)]

    def mont_mul(self, a, b, p_list):
        t = [0 for i in range(2 * self.rexp)]
        for i in range(self.rexp):
            c = 0
            for j in range(self.rexp):
                t[i+j] = t[i+j] + a[i]*b[j] + c
                # print("a=" + str(a[i]))
                # print("b=" + str(b[j]))
                # print("t[i+j]=" + str(t[i+j]))
                c = (t[i+j] & self.high_mask) >> self.width
                t[i+j] = t[i+j] & self.low_mask
                # print("c=" + str(c) + " t[i+j]=" + str(t[i+j]))
            t[i+self.rexp] = c
        # out = 0
        # for i in range(self.rexp):
        #     out = out | (t[i + self.rexp - 1] << (i * self.width))
        # print(self.to_bytes(out))
        p_inv = self.pinv(self.width) & self.low_mask
        # print("p_inv=" + str(p_inv))
        for i in range(self.rexp):
            c = 0
            u = (t[i] * p_inv) & self.low_mask
            for j in range(self.rexp):
                t[i+j] = t[i+j] + u*p_list[j] + c
                # print("u=" + str(u))
                # print("p=" + str(p_list[j]))
                # print("t[i+j]=" + str(t[i+j]))
                c = (t[i+j] & self.high_mask) >> self.width
                t[i+j] = t[i+j] & self.low_mask
            t[i+self.rexp] = t[i+self.rexp] + c
        out = 0
        for i in range(self.rexp):
            out = out | (t[i + self.rexp] << (i * self.width))
        print(self.to_bytes(out % self.p))
        return out

class Fp2:
    # Fp[x] / (x^2 - alpha)
    def __init__(self, name, fp, alpha):
        self.degree = 2
        self.name = name
        self.fp = fp
        self.alpha = alpha

    @property
    def count(self):
        return 2

    def add(self, x0, x1):
        a0, b0 = x0
        a1, b1 = x1
        return (self.fp.add(a0, a1), self.fp.add(b0, b1))

    def neg(self, x):
        a, b = x
        return (self.fp.neg(a), self.fp.neg(b))

    def sub(self, x0, x1):
        a0, b0 = x0
        a1, b1 = x1
        return (self.fp.sub(a0, a1), self.fp.sub(b0, b1))

    def mul(self, x0, x1):
        a0, b0 = x0
        a1, b1 = x1
        b0b1 = self.fp.mul(b0, b1)
        a2 = self.fp.add(self.fp.mul(a0, a1), self.fp.mul(self.alpha, b0b1))
        b2 = self.fp.add(self.fp.mul(a0, b1), self.fp.mul(a1, b0))
        return (a2, b2)

    # helper
    def det(self, a, b):
        a2 = self.fp.mul(a, a)
        b2 = self.fp.mul(b, b)
        return self.fp.sub(a2, self.fp.mul(self.alpha, b2))

    def inv(self, x):
        a, b = x
        det = self.det(a, b)
        if det == self.fp.zero():
            raise ZeroDivisionError()
        inv_det = self.fp.inv(det)
        inv_a = self.fp.mul(a, inv_det)
        inv_b = self.fp.neg(self.fp.mul(b, inv_det))
        return (inv_a, inv_b)

    def one(self):
        return (self.fp.one(), self.fp.zero())

    def zero(self):
        return (self.fp.zero(), self.fp.zero())

    def random_nonzero_element(self):
        while True:
            a = self.fp.random_element()
            b = self.fp.random_element()
            if self.det(a, b) != self.zero():
                return (a, b)

    def to_mont(self, x):
        a, b = x
        return (self.fp.to_mont(a), self.fp.to_mont(b))

Fq_SM2 = Fp('FqSM2', SM2_q, 64, rexp=4)
Fq_SM2_n = Fp('FqSM2_n', SM2_n, 64, rexp=4)