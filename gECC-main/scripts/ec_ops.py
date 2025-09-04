class Op:
    def __init__(self, op, lhs, a, b):
        self.op = op
        self.lhs = lhs
        self.a = a
        self.b = b
        self.should_be_reduced = None


class ECOp:
    def __init__(self, code):
        self.ops = []
        tmp_vars = 0
        for line in code.strip().replace(' ', '').split('\n'):
            lhs, rhs = line.split('=')
            found_ops = 0
            for op in '+-*^/':
                if op in rhs:
                    found_ops += 1
                    a, b = rhs.split(op)
                    if a == 'a':
                        a = 'self.a'
                    if b == 'a':
                        b = 'self.a'
                    if op == '+':
                        self.ops.append(Op('+', lhs, a, b))
                    elif op == '-':
                        self.ops.append(Op('-', lhs, a, b))
                    elif op == '*':
                        try:
                            num_a = int(a)
                            if num_a == 2:
                                self.ops.append(Op('+', lhs, b, b))
                            elif num_a == 4:
                                tmp_vars += 1
                                tmp_var = 'gecc_optmp_{}'.format(tmp_vars)
                                self.ops.append(Op('+', tmp_var, b, b))
                                self.ops.append(Op('+', lhs, tmp_var, tmp_var))
                            elif num_a == 3:
                                tmp_vars += 1
                                tmp_var = 'gecc_optmp_{}'.format(tmp_vars)
                                self.ops.append(Op('+', tmp_var, b, b))
                                self.ops.append(Op('+', lhs, tmp_var, b))
                            elif num_a == 8:
                                tmp_vars += 2
                                tmp_var1 = 'gecc_optmp_{}'.format(tmp_vars - 1)
                                tmp_var2 = 'gecc_optmp_{}'.format(tmp_vars)
                                self.ops.append(Op('+', tmp_var1, b, b))
                                self.ops.append(
                                    Op('+', tmp_var2, tmp_var1, tmp_var1))
                                self.ops.append(
                                    Op('+', lhs, tmp_var2, tmp_var2))
                            else:
                                raise NotImplementedError()
                        except ValueError:
                            self.ops.append(Op('*', lhs, a, b))
                    elif op == '^':
                        try:
                            num_b = int(b)
                        except ValueError:
                            num_b = 0
                        assert num_b == 2
                        self.ops.append(Op('*', lhs, a, a))
                    elif op == '/':
                        try:
                            num_b = int(b)
                            if num_b == 2:
                                self.ops.append(Op('*', lhs, a, 'self.field.inv2'))
                            else:
                                raise NotImplementedError()
                        except ValueError:
                            self.ops.append(Op('/', lhs, a, b))

            assert found_ops <= 1
            if found_ops == 0:
                self.ops.append(Op('=', lhs, rhs, None))
            self.preprocess_should_be_reduced()

    def preprocess_should_be_reduced(self):
        used_operands = set()
        # fixed bug: there is a possibility of overflow, only when BLST_377, MNT4753 is right when use u64 u32
        for o in reversed(self.ops):
            if o.op in '+-':
            # if o.op in '+-' and (o.lhs in used_operands or o.lhs in ('X3', 'Y3', 'Z3')):
                o.should_be_reduced = True
            # else:
            #     o.should_be_reduced = False
            # if o.op == '+' or o.op == '-':
            #     used_operands.add(o.a)
            #     used_operands.add(o.b)

    def pycode(self):
        result = []
        for o in self.ops:
            if o.op == '=':
                result.append('{}={}'.format(o.lhs, o.a))
            elif o.op == '/':
                result.append('{}=self.field.{}({})'.format(
                    o.lhs, {'/': 'inv'}[o.op], o.b))
                result.append('{}=self.field.{}({}, {})'.format(
                    o.lhs, {'*': 'mul'}['*'], o.a, o.lhs))
            else:
                result.append('{}=self.field.{}({}, {})'.format(
                    o.lhs, {'+': 'add', '-': 'sub', '*': 'mul'}[o.op], o.a, o.b))
        return '\n'.join(result)

    def pyexec(self, global_vars, local_vars):
        exec(self.pycode(), global_vars, local_vars)

    @staticmethod
    def remap_var(v):
        if v in ('X1', 'Y1', 'Z1'):
            return v[0].lower()
        if v in ('X2', 'Y2', 'Z2'):
            return 'o.' + v[0].lower()
        return v

    # when without reduce, a+-*b in [0, 2p)
    def cucode(self, is_ec=1):
        result = []
        variables = list(sorted(list(set(op.lhs for op in self.ops))))
        result.append('BaseField {};'.format(', '.join(variables)))
        for o in self.ops:
            op = o.op
            lhs = o.lhs
            a = ECOp.remap_var(o.a)
            b = ECOp.remap_var(o.b)
            if op == '=':
                result.append('{} = {};'.format(lhs, a))
            else:
                if op == '+':
                    if o.should_be_reduced:
                        if o.b == 'self.a':
                            result.append(
                                'BaseField a_mont = BaseField::load_const(ECDCONST.a_mont);')
                            result.append(
                                '{} = ({} + a_mont).reduce_to_pp();'.format(lhs, a))
                        else:
                            result.append(
                                '{} = ({} + {}).reduce_to_pp();'.format(lhs, a, b))
                    else:
                        if o.b == 'self.a':
                            result.append(
                                'BaseField a_mont = BaseField::load_const(ECDCONST.a_mont);')
                            result.append(
                                '{} = ({} + a_mont);'.format(lhs, a))
                        else:
                            result.append(
                                '{} = {} + {};'.format(lhs, a, b))
                elif op == '-':
                    if o.should_be_reduced:
                        result.append(
                            '{} = ({} + BaseField::pp() - {}).reduce_to_pp();'.format(lhs, a, b))
                    else:
                        result.append(
                            '{} = {} + BaseField::pp() - {};'.format(lhs, a, b))
                elif op == '*':
                    if o.b == 'self.a':
                        result.append(
                            '{} = MultiplicativeChain<BaseField, HCONST.a>::multiply({});'.format(lhs, a))
                    elif o.b == 'self.field.inv2':
                        result.append(
                                'BaseField inv2 = BaseField::inv2();')
                        result.append(
                            '{} = {} * inv2;'.format(lhs, a))
                    elif a == b:
                        result.append(
                            '{} = {}.square();'.format(lhs, a))
                    else:
                        result.append(
                            '{} = {} * {};'.format(lhs, a, b))
                elif op == '/':
                    result.append(
                            '{} = {}.inverse();'.format(lhs, b))
                    result.append(
                            '{} = {} * {};'.format(lhs, a, lhs))
        if is_ec:
            result.append('ECPointJacobian result;')
        else:
            result.append('Affine result;')
        result.append('result.x = X3.reduce_to_p();')
        result.append('result.y = Y3.reduce_to_p();')
        if is_ec:
            result.append('result.z = Z3.reduce_to_p();')
        result.append('return result;')
        return '\n'.join(result)

    # when with reduce, a+-*b in [0, p), ignore should_be_reduced
    def cucode_with_reduce(self, is_ec=1):
        result = []
        variables = list(sorted(list(set(op.lhs for op in self.ops))))
        result.append('BaseField {};'.format(', '.join(variables)))
        for o in self.ops:
            op = o.op
            lhs = o.lhs
            a = ECOp.remap_var(o.a)
            b = ECOp.remap_var(o.b)
            if op == '=':
                result.append('{} = {};'.format(lhs, a))
            else:
                if op == '+':
                    if o.b == 'self.a':
                        result.append(
                            'BaseField a_mont = BaseField::load_const(ECDCONST.a_mont);')
                        result.append(
                            '{} = ({} + a_mont);'.format(lhs, a))
                    else:
                        result.append(
                            '{} = {} + {};'.format(lhs, a, b))
                elif op == '-':
                    result.append(
                        '{} = {} - {};'.format(lhs, a, b))
                elif op == '*':
                    if o.b == 'self.a':
                        result.append(
                            '{} = MultiplicativeChain<BaseField, HCONST.a>::multiply({});'.format(lhs, a))
                    elif o.b == 'self.field.inv2':
                        result.append(
                                'BaseField inv2 = BaseField::inv2();')
                        result.append(
                            '{} = {} * inv2;'.format(lhs, a))
                    elif a == b:
                        result.append(
                            '{} = {}.square();'.format(lhs, a))
                    else:
                        result.append(
                            '{} = {} * {};'.format(lhs, a, b))
                elif op == '/':
                    result.append(
                            '{} = {}.inverse();'.format(lhs, b))
                    result.append(
                            '{} = {} * {};'.format(lhs, a, lhs))
        if is_ec:
            result.append('ECPointJacobian result;')
        else:
            result.append('Affine result;')
        result.append('result.x = X3;')
        result.append('result.y = Y3;')
        if is_ec:
            result.append('result.z = Z3;')
        result.append('return result;')
        return '\n'.join(result)
    
    def graphviz(self):
        edges = []
        for op, lhs, a, b in self.ops:
            if op == '=':
                edges.append((lhs, a))
            else:
                edges.append((lhs, a))
                edges.append((lhs, b))
        return 'digraph { ' + ' '.join('{} -> {};'.format(a, b) for a, b in edges) + '}'


# Short Weierstrass curves: https://www.hyperelliptic.org/EFD/g1p/
# for a != 0 or a == 0
# https://www.hyperelliptic.org/EFD/g1p/auto-code/shortw/jacobian-0/addition/add-2007-bl.op3
EC_ADD = ECOp('''
Z1Z1 = Z1^2
Z2Z2 = Z2^2
U1 = X1*Z2Z2
U2 = X2*Z1Z1
t0 = Z2*Z2Z2
S1 = Y1*t0
t1 = Z1*Z1Z1
S2 = Y2*t1
H = U2-U1
t2 = 2*H
I = t2^2
J = H*I
t3 = S2-S1
r = 2*t3
V = U1*I
t4 = r^2
t5 = 2*V
t6 = t4-J
X3 = t6-t5
t7 = V-X3
t8 = S1*J
t9 = 2*t8
t10 = r*t7
Y3 = t10-t9
t11 = Z1+Z2
t12 = t11^2
t13 = t12-Z1Z1
t14 = t13-Z2Z2
Z3 = t14*H
''')

# for a != 0 or a == 0
# https://www.hyperelliptic.org/EFD/g1p/auto-code/shortw/jacobian/addition/madd-2007-bl.op3
EC_MIXED_ADD = ECOp('''
Z1Z1 = Z1^2
U2 = X2*Z1Z1
t0 = Z1*Z1Z1
S2 = Y2*t0
H = U2-X1
HH = H^2
I = 4*HH
J = H*I
t1 = S2-Y1
r = 2*t1
V = X1*I
t2 = r^2
t3 = 2*V
t4 = t2-J
X3 = t4-t3
t5 = V-X3
t6 = Y1*J
t7 = 2*t6
t8 = r*t5
Y3 = t8-t7
t9 = Z1+H
t10 = t9^2
t11 = t10-Z1Z1
Z3 = t11-HH
''')

# for a != 0
# https://www.hyperelliptic.org/EFD/g1p/auto-code/shortw/jacobian-0/doubling/dbl-2007-bl.op3
EC_DBL = ECOp('''
XX = X1^2
YY = Y1^2
YYYY = YY^2
ZZ = Z1^2
t0 = X1+YY
t1 = t0^2
t2 = t1-XX
t3 = t2-YYYY
S = 2*t3
t4 = ZZ^2
t5 = t4*a
t6 = 3*XX
M = t6+t5
t7 = M^2
t8 = 2*S
T = t7-t8
X3 = T
t9 = S-T
t10 = 8*YYYY
t11 = M*t9
Y3 = t11-t10
t12 = Y1+Z1
t13 = t12^2
t14 = t13-YY
Z3 = t14-ZZ
''')

# for a == 0
# https://www.hyperelliptic.org/EFD/g1p/auto-code/shortw/jacobian-0/doubling/dbl-2009-l.op3
EC_DBL_1 = ECOp('''
A = X1^2
B = Y1^2
C = B^2
t0 = X1+B
t1 = t0^2
t2 = t1-A
t3 = t2-C
D = 2*t3
E = 3*A
F = E^2
t4 = 2*D
X3 = F-t4
t5 = D-X3
t6 = 8*C
t7 = E*t5
Y3 = t7-t6
t8 = Y1*Z1
Z3 = 2*t8
''')

# for a == -3, M = 3x^2+az^4 = 3(x^2-z^4)
# https://www.hyperelliptic.org/EFD/g1p/auto-code/shortw/jacobian-3/addition/zadd-2007-m.op3
EC_DBL_2 = ECOp('''
XX = X1^2
YY = Y1^2
YYYY = YY^2
ZZ = Z1^2
t0 = X1+YY
t1 = t0^2
t2 = t1-XX
t3 = t2-YYYY
S = 2*t3
t4 = ZZ^2
t5 = XX-t4
M = 3*t5
t7 = M^2
t8 = 2*S
T = t7-t8
X3 = T
t9 = S-T
t10 = 8*YYYY
t11 = M*t9
Y3 = t11-t10
t12 = Y1+Z1
t13 = t12^2
t14 = t13-YY
Z3 = t14-ZZ
''')

# for a != 0 or a == 0
# https://www.hyperelliptic.org/EFD/g1p/auto-code/shortw/jacobian/doubling/mdbl-2007-bl.op3
EC_AFF_DBL = ECOp('''
XX = X1^2
YY = Y1^2
YYYY = YY^2
t0 = X1+YY
t1 = t0^2
t2 = t1-XX
t3 = t2-YYYY
S = 2*t3
t4 = 3*XX
M = t4+a
t5 = M^2
t6 = 2*S
T = t5-t6
X3 = T
t7 = S-T
t8 = 8*YYYY
t9 = M*t7
Y3 = t9-t8
Z3 = 2*Y1
''')

# a != 0
AFF_DBL = ECOp('''
XX = X1^2
t1 = 3*XX
t2 = t1+a
t3 = 2*Y1
M = t2/t3
MM = M^2
t4 = 2*X1
X3 = MM-t4
t5 = X1-X3
t6 = M*t5
Y3 = t6-Y1
''')

# a = 0
AFF_DBL_1 = ECOp('''
XX = X1^2
t1 = 3*XX
t3 = 2*Y1
M = t1/t3
MM = M^2
t4 = 2*X1
X3 = MM-t4
t5 = X1-X3
t6 = M*t5
Y3 = t6-Y1
''')

# for a != 0 or a == 0
AFF_ADD = ECOp('''
YD = Y2-Y1
XD = X2-X1
M = YD/XD
MM = M^2
t1 = MM-X1
X3 = t1-X2
t2 = X1-X3
t3 = M*t2
Y3 = t3-Y1
''')

if __name__ == '__main__':
    print(EC_ADD.pycode())
