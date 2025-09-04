#pragma once

// must need this
# ifdef __GNUC__
#  define asm __asm__ __volatile__
# else
#  define asm asm volatile
# endif

__device__ __forceinline__ static void mul_n(Base* acc, const Base* a, Base bi,
                             size_t n = REXP) {
  for (size_t j = 0; j < n; j += 2)
      asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
          : "=r"(acc[j]), "=r"(acc[j+1])
          : "r"(a[j]), "r"(bi));
}

__device__ __forceinline__ static void madc_n_rshift(Base* odd, const Base *a, Base bi, 
                                      size_t n = REXP) {
  for (size_t j = 0; j < n-2; j += 2)
      asm("madc.lo.cc.u32 %0, %2, %3, %4; madc.hi.cc.u32 %1, %2, %3, %5;"
          : "=r"(odd[j]), "=r"(odd[j+1])
          : "r"(a[j]), "r"(bi), "r"(odd[j+2]), "r"(odd[j+3]));
  asm("madc.lo.cc.u32 %0, %2, %3, %4; madc.hi.cc.u32 %1, %2, %3, 0;"
      : "=r"(odd[n-2]), "=r"(odd[n-1])
      : "r"(a[n-2]), "r"(bi), "r"(odd[n]));
}

__device__ __forceinline__ static void cmad_n(Base* acc, const Base* a, Base bi,
                              size_t n = REXP) {
  asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
      : "+r"(acc[0]), "+r"(acc[1])
      : "r"(a[0]), "r"(bi));
  for (size_t j = 2; j < n; j += 2)
      asm("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
          : "+r"(acc[j]), "+r"(acc[j+1])
          : "r"(a[j]), "r"(bi));
  // return carry flag
}

__device__ __forceinline__ static void cmad_n_2(Base* acc, const Base* a, Base bi,
                              size_t n = REXP) {
  // add carry flag
  for (size_t j = 0; j < n; j += 2)
      asm("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
          : "+r"(acc[j]), "+r"(acc[j+1])
          : "r"(a[j]), "r"(bi));
  // return carry flag
}

__device__ __forceinline__ static void mad_n_redc(Base *even, Base *odd,
                                  const Base *a, Base bi, 
                                  const Base *p, 
                                  bool first=false)
{
    if (first) {
        mul_n(odd, a+1, bi);
        mul_n(even, a,  bi);
    } else {
        asm("add.cc.u32 %0, %0, %1;" : "+r"(even[0]) : "r"(odd[1]));
        madc_n_rshift(odd, a+1, bi);
        asm("addc.u32 %0, 0, 0;" : "+r"(odd[REXP]));
        cmad_n(even, a, bi);
        asm("addc.u32 %0, %0, 0;" : "+r"(even[REXP]));
    }

    // Special note about pinv being declared as uint32_t& [as opposed to just
    // uint32_t]. It was noted that if pinv is 0xffffffff, CUDA compiler
    // generates suboptimal code for Montgomery reduction. The way to work
    // around the problem is to prevent compiler from viewing it as constant.
    // For this reason it's suggested to declare the parameter as following:
    //
    //    __device__ __constant__ /*const*/ pinv = <literal>;
    Base mi = even[0] * DCONST.pinv;

    cmad_n(odd, p+1, mi);
    asm("addc.u32 %0, %0, 0;" : "+r"(odd[REXP]));
    cmad_n(even, p,  mi);
    asm("addc.u32 %0, %0, 0;" : "+r"(even[REXP]));
}

__device__ __forceinline__ static void mad_n(Base *even, Base *odd,
                                  const Base *a, Base bi, 
                                  const Base *p, 
                                  bool first=false) {
    if (first) {
        mul_n(odd, a+1, bi);
        mul_n(even, a,  bi);
    } else {
        cmad_n(odd, a+1, bi);
        asm("addc.u32 %0, 0, 0;" : "+r"(odd[REXP]));
        cmad_n(even, a, bi);
        asm("addc.u32 %0, 0, 0;" : "+r"(even[REXP]));
    }
}


__device__ __forceinline__ static void redc(Base *even, Base *odd,
                                  const Base *a, Base &carry, bool first=false) {
    // Special note about pinv being declared as uint32_t& [as opposed to just
    // uint32_t]. It was noted that if pinv is 0xffffffff, CUDA compiler
    // generates suboptimal code for Montgomery reduction. The way to work
    // around the problem is to prevent compiler from viewing it as constant.
    // For this reason it's suggested to declare the parameter as following:
    //
    //    __device__ __constant__ /*const*/ pinv = <literal>;
    Base mi = even[0] * DCONST.pinv;

    // move carry to CC.CF
    asm("add.cc.u32 %0, %0, 0xFFFFFFFF;" : "+r"(carry));
    cmad_n_2(odd, a+1, mi);
    asm("addc.cc.u32 %0, %0, 0;" : "+r"(odd[REXP]));
    cmad_n(even, a, mi);
    asm("addc.u32 %0, %0, 0;" : "+r"(even[REXP]));
    asm("add.cc.u32 %0, %0, %1;" : "+r"(odd[0]) : "r"(even[1]));
    asm("addc.u32 %0, 0, 0;" : "=r"(carry));
}

__device__ __forceinline__ FpT redc_sm2(Base *even, Base *odd,
                                              const Base *a, Base &carry, bool first=false) const {
  size_t i;
  Base tmp[REXP];

  Base mi;
  // move carry to CC.CF
  asm("add.cc.u32 %0, %0, 0xFFFFFFFF;" : "+r"(carry));
  asm("addc.cc.u32 %0, %1, %2;" : "=r"(mi) : "r"(even[1]), "r"(odd[0]));
  asm("addc.u32 %0, 0, 0;" : "=r"(carry));
  tmp[0] = even[0];

  asm("sub.cc.u32 %0, %1, %2;" : "=r"(tmp[1]) : "r"(mi), "r"(even[0]));
  asm("subc.cc.u32 %0, %1, %2;" : "=r"(tmp[2]) : "r"(0), "r"(mi));
  asm("subc.cc.u32 %0, %1, %2;" : "=r"(tmp[3]) : "r"(0), "r"(0));
  asm("subc.cc.u32 %0, %1, %2;" : "=r"(tmp[4]) : "r"(0), "r"(0));
  asm("subc.cc.u32 %0, %1, %2;" : "=r"(tmp[5]) : "r"(0), "r"(even[0]));
  asm("subc.cc.u32 %0, %1, %2;" : "=r"(tmp[6]) : "r"(even[0]), "r"(mi));
  asm("subc.u32 %0, %1, %2;" : "=r"(tmp[7]) : "r"(mi), "r"(0));

  // move carry to CC.CF
  asm("add.cc.u32 %0, %0, 0xFFFFFFFF;" : "+r"(carry));

  for (i = 0; i < REXP; i++) {
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(even[2+i]) : "r"(even[2+i]), "r"(tmp[i]));
  }
  asm("addc.u32 %0, %1, %2;" : "=r"(even[10]) : "r"(even[10]), "r"(0)); 

  // update carry
  asm("add.cc.u32 %0, %0, %1;" : "+r"(odd[1]) : "r"(even[2]));
  asm("addc.u32 %0, 0, 0;" : "=r"(carry));
}

__device__ __forceinline__ FpT mont_multiply_cios(const Base *o,
                                                  const u32 stride = 1) const {
  static_assert(std::is_same<Base, u32>::value,
                "Base must be u32.");

  const FpT p = load_const(DCONST.p);

  u32 even[REXP + 1];
  u32 odd[REXP + 1];
  even[REXP] = 0;
  odd[REXP] = 0;
  
  size_t i;
  asm("{ .reg.pred %top;");

  #pragma unroll
  for (i = 0; i < REXP; i += 2) {
      mad_n_redc(even, odd, this->digits, o[i * stride], p.digits, i==0);
      mad_n_redc(odd, even, this->digits, o[(i + 1) * stride], p.digits);
  }
  
  FpT result;
  // merge |even| and |odd|
  asm("add.cc.u32 %0, %1, %2;" : "=r"(result.digits[0]) : "r"(even[0]) , "r"(odd[1]));
  for (i = 1; i < REXP; i++)
      asm("addc.cc.u32 %0, %1, %2;" : "=r"(result.digits[i]) : "r"(even[i]) , "r"(odd[i+1]));
  asm("addc.cc.u32 %0, %0, 0;" : "+r"(even[REXP]));

  // final subtraction
  asm("sub.cc.u32 %0, %1, %2;" : "=r"(even[0]) : "r"(result.digits[0]), "r"(p.digits[0]));
  for (i = 1; i < REXP; i++)
      asm("subc.cc.u32 %0, %1, %2;" : "=r"(even[i]) : "r"(result.digits[i]), "r"(p.digits[i]));
  asm("subc.u32 %0, %0, 0; setp.eq.u32 %top, %0, 0;" : "+r"(even[REXP]));

  for (i = 0; i < REXP; i++)
      asm("@%top mov.b32 %0, %1;" : "+r"(result.digits[i]) : "r"(even[i]));

  asm("}");

  return result;
}

__device__ __forceinline__ FpT mont_multiply_sos(const Base *o,
                                                  const u32 stride = 1) const {
  static_assert(std::is_same<Base, u32>::value,
                "Base must be u32.");

  const FpT p = load_const(DCONST.p);

  u32 even[REXP*2+1];
  u32 odd[REXP*2];
  Base carry = 0;
  size_t i;
  #pragma unroll
  for (i = REXP; i < REXP*2+1; i += 1) {
    even[i] = 0;
  }
  #pragma unroll
  for (i = REXP; i < REXP*2; i += 1) {
    odd[i] = 0;
  }

  #pragma unroll
  for (i = 0; i < REXP; i += 2) {
      mad_n(even+i, odd+i, this->digits, o[i * stride], p.digits, i==0);
      mad_n(odd+i, even+i+2, this->digits, o[(i + 1) * stride], p.digits);
  }

  #pragma unroll
  for (i = 0; i < REXP; i += 2) {
      redc(even+i, odd+i, p.digits, carry, i==0);
      redc(odd+i, even+i+2, p.digits, carry);
  }
  
  FpT result;
  // merge |even| and |odd|
  result.digits[0] = even[0+REXP];
  asm("add.cc.u32 %0, %0, 0xFFFFFFFF;" : "+r"(carry));
  for (i = 1; i < REXP; i++)
      asm("addc.cc.u32 %0, %1, %2;" : "=r"(result.digits[i]) : "r"(even[i+REXP]) , "r"(odd[i+REXP-1]));
  asm("addc.u32 %0, %0, 0;" : "+r"(even[2*REXP]));

  asm("{ .reg.pred %top;");
  // final subtraction
  asm("sub.cc.u32 %0, %1, %2;" : "=r"(even[0+REXP]) : "r"(result.digits[0]), "r"(p.digits[0]));
  for (i = 1; i < REXP; i++)
      asm("subc.cc.u32 %0, %1, %2;" : "=r"(even[i+REXP]) : "r"(result.digits[i]), "r"(p.digits[i]));
  asm("subc.u32 %0, %0, 0; setp.eq.u32 %top, %0, 0;" : "+r"(even[2*REXP]));

  for (i = 0; i < REXP; i++)
      asm("@%top mov.b32 %0, %1;" : "+r"(result.digits[i]) : "r"(even[i+REXP]));

  asm("}");

  return result;
}

__device__ __forceinline__ FpT mont_multiply_sos_sm2(const Base *o,
                                                            const u32 stride = 1) const {
  static_assert(std::is_same<Base, u32>::value,
                "Base must be u32.");

  const FpT p = load_const(DCONST.p);

  u32 even[REXP*2+1];
  u32 odd[REXP*2];
  Base carry = 0;
  size_t i;
  #pragma unroll
  for (i = REXP; i < REXP*2+1; i += 1) {
    even[i] = 0;
  }
  #pragma unroll
  for (i = REXP; i < REXP*2; i += 1) {
    odd[i] = 0;
  }

  #pragma unroll
  for (i = 0; i < REXP; i += 2) {
      mad_n(even+i, odd+i, this->digits, o[i * stride], p.digits, i==0);
      mad_n(odd+i, even+i+2, this->digits, o[(i + 1) * stride], p.digits);
  }

  #pragma unroll
  for (i = 0; i < REXP; i += 4) {
      redc_sm2(even+i, odd+i, p.digits, carry, i==0);
      redc_sm2(odd+i+1, even+i+3, p.digits, carry);
  }
  
  FpT result;
  // merge |even| and |odd|
  result.digits[0] = even[0+REXP];
  asm("add.cc.u32 %0, %0, 0xFFFFFFFF;" : "+r"(carry));
  for (i = 1; i < REXP; i++)
      asm("addc.cc.u32 %0, %1, %2;" : "=r"(result.digits[i]) : "r"(even[i+REXP]) , "r"(odd[i+REXP-1]));
  asm("addc.u32 %0, %0, 0;" : "+r"(even[2*REXP]));

  asm("{ .reg.pred %top;");
  // final subtraction
  asm("sub.cc.u32 %0, %1, %2;" : "=r"(even[0+REXP]) : "r"(result.digits[0]), "r"(p.digits[0]));
  for (i = 1; i < REXP; i++)
      asm("subc.cc.u32 %0, %1, %2;" : "=r"(even[i+REXP]) : "r"(result.digits[i]), "r"(p.digits[i]));
  asm("subc.u32 %0, %0, 0; setp.eq.u32 %top, %0, 0;" : "+r"(even[2*REXP]));

  for (i = 0; i < REXP; i++)
      asm("@%top mov.b32 %0, %1;" : "+r"(result.digits[i]) : "r"(even[i+REXP]));

  asm("}");

  return result;
}

__device__ __forceinline__ FpT operator*(const FpT &b) const {
  if (mont_flag == MONTFLAG::CIOS) {
    return mont_multiply_cios(b.digits);
  }
  else if (mont_flag == MONTFLAG::SOS) {
    if (curve_flag == CURVEFLAG::SM2)
      return mont_multiply_sos_sm2(b.digits);
    else
      return mont_multiply_sos(b.digits);
  }
}

__device__ __forceinline__ FpT mont_multiply(const Base *o,
                                              const u32 stride = 1) const {
  if (mont_flag == MONTFLAG::CIOS) {
    return mont_multiply_cios(o, stride);
  }
  else if (mont_flag == MONTFLAG::SOS) {
    if (curve_flag == CURVEFLAG::SM2)
      return mont_multiply_sos_sm2(o, stride);
    else
      return mont_multiply_sos(o, stride);
  }                       
}

# undef inline
# undef asm