# ifdef __GNUC__
#  define asm __asm__ __volatile__
# else
#  define asm asm volatile
# endif
__device__ __forceinline__ FpT operator+(const FpT& b) const {
  size_t i;
  u32 tmp[REXP+1];
  FpT result;
  asm("{ .reg.pred %top;");

  asm("add.cc.u32 %0, %1, %2;" : "=r"(result.digits[0]) : "r"(digits[0]), "r"(b.digits[0]));
  for (i = 1; i < REXP; i++)
      asm("addc.cc.u32 %0, %1, %2;" : "=r"(result.digits[i]) : "r"(digits[i]), "r"(b.digits[i]));
  asm("addc.u32 %0, 0, 0;" : "=r"(tmp[REXP]));

  asm("sub.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(result.digits[0]), "r"(DCONST.p[0]));
  for (i = 1; i < REXP; i++)
      asm("subc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(result.digits[i]), "r"(DCONST.p[i]));
  asm("subc.u32 %0, %0, 0; setp.eq.u32 %top, %0, 0;" : "+r"(tmp[REXP]));

  for (i = 0; i < REXP; i++)
      asm("@%top mov.b32 %0, %1;" : "+r"(result.digits[i]) : "r"(tmp[i]));

  asm("}");
  return result;
}

__device__ __forceinline__ FpT operator-(const FpT& b) const
{
  size_t i;
  u32 tmp[REXP], borrow;
  FpT result;

  asm("sub.cc.u32 %0, %1, %2;" : "=r"(result.digits[0]) : "r"(digits[0]), "r"(b.digits[0]));
  for (i = 1; i < REXP; i++)
      asm("subc.cc.u32 %0, %1, %2;" : "=r"(result.digits[i]) : "r"(digits[i]), "r"(b.digits[i]));
  asm("subc.u32 %0, 0, 0;" : "=r"(borrow));

  asm("add.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(result.digits[0]), "r"(DCONST.p[0]));
  for (i = 1; i < REXP-1; i++)
      asm("addc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(result.digits[i]), "r"(DCONST.p[i]));
  asm("addc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(result.digits[i]), "r"(DCONST.p[i]));

  asm("{ .reg.pred %top; setp.ne.u32 %top, %0, 0;" :: "r"(borrow));
  for (i = 0; i < REXP; i++)
      asm("@%top mov.b32 %0, %1;" : "+r"(result.digits[i]) : "r"(tmp[i]));
  asm("}");

  return result;
}
# undef inline
# undef asm