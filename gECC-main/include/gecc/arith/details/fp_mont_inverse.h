# ifdef __GNUC__
#  define asm __asm__ __volatile__
# else
#  define asm asm volatile
# endif

  __device__ __forceinline__ void add_p_unsafe(Base *digits) const {
    size_t i;
    asm("add.cc.u32 %0, %0, %1;" : "+r"(digits[0]) : "r"(DCONST.p[0]));
    for (i = 1; i < REXP; i++)
        asm("addc.cc.u32 %0, %0, %1;" : "+r"(digits[i]) : "r"(DCONST.p[i]));
    asm("addc.u32 %0, %0, 0;" : "+r"(digits[i]));
    return ;
  }

  __device__ __forceinline__ void sub_safe(Base *c, Base *a, const Base *b) const {
    u32 tmp[REXP + 1];
    size_t i;
    asm("sub.cc.u32 %0, %1, %2;" : "=r"(c[0]) : "r"(a[0]), "r"(b[0]));
    for (i = 1; i < REXP; i++)
      asm("subc.cc.u32 %0, %1, %2;" : "=r"(c[i]) : "r"(a[i]), "r"(b[i]));
    asm("subc.u32 %0, %1, %2;" : "=r"(c[i]) : "r"(a[i]), "r"(b[i]));
    
    asm("add.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(c[0]), "r"(DCONST.p[0]));
    for (i = 1; i < REXP; i++)
        asm("addc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(c[i]), "r"(DCONST.p[i]));
    tmp[i] = 0;

    asm("{ .reg.pred %top; setp.ne.u32 %top, %0, 0;" :: "r"(c[REXP]));
    for (i = 0; i < REXP + 1; i++)
        asm("@%top mov.b32 %0, %1;" : "+r"(c[i]) : "r"(tmp[i]));
    asm("}");
    return ;
  }

  __device__ __forceinline__ void sub(Base *u, Base *v, Base *b, Base *c) const {
    u32 tmp1[REXP + 1], tmp2[REXP + 1], tmp3[REXP + 1], tmp4[REXP + 1];
    size_t i;
    // u - v
    asm("sub.cc.u32 %0, %1, %2;" : "=r"(tmp1[0]) :"r"(u[0]), "r"(v[0]));
    for (i = 1; i < REXP; i++)
      asm("subc.cc.u32 %0, %1, %2;" : "=r"(tmp1[i]) : "r"(u[i]), "r"(v[i]));
    asm("subc.u32 %0, %1, %2;" : "=r"(tmp1[i]) : "r"(u[i]), "r"(v[i]));

    // v - u
    asm("sub.cc.u32 %0, %1, %2;" : "=r"(tmp2[0]) :"r"(v[0]), "r"(u[0]));
    for (i = 1; i < REXP; i++)
      asm("subc.cc.u32 %0, %1, %2;" : "=r"(tmp2[i]) : "r"(v[i]), "r"(u[i]));
    asm("subc.u32 %0, %1, %2;" : "=r"(tmp2[i]) : "r"(v[i]), "r"(u[i]));

    sub_safe(tmp3, b, c);
    sub_safe(tmp4, c, b);

    // if u > v, u = u - v, b = b - c
    asm("{ .reg.pred %top; setp.ne.u32 %top, %0, 0;" :: "r"(tmp2[REXP]));
    for (i = 0; i < REXP + 1; i++)
        asm("@%top mov.b32 %0, %1;" : "+r"(u[i]) : "r"(tmp1[i]));
    for (i = 0; i < REXP + 1; i++)
        asm("@%top mov.b32 %0, %1;" : "+r"(b[i]) : "r"(tmp3[i]));
    asm("}");


    // if u < v, v = v - u, c = c - b
    asm("{ .reg.pred %top; setp.ne.u32 %top, %0, 0;" :: "r"(tmp1[REXP]));
    for (i = 0; i < REXP + 1; i++)
        asm("@%top mov.b32 %0, %1;" : "+r"(v[i]) : "r"(tmp2[i]));
    for (i = 0; i < REXP + 1; i++)
        asm("@%top mov.b32 %0, %1;" : "+r"(c[i]) : "r"(tmp4[i]));
    asm("}");
  }
# undef inline
# undef asm

  // only for width = 1
  __device__ __forceinline__ FpT inverse() const {
    if (this->is_zero()) {
      printf("this is zero, cannot inverse\n");
      return *this;
    }

    Base u[REXP + 1];
    u[REXP] = 0;
#pragma unroll
    for (u32 i = 0; i < REXP; i++) {
      u[i] = digits[i];
    }

    Base v[REXP + 1];
    v[REXP] = 0;
#pragma unroll
    for (u32 i = 0; i < REXP; i++) {
      v[i] = DCONST.p[i];
    }

    Base b[REXP + 1];
    b[REXP] = 0;
#pragma unroll
    for (u32 i = 0; i < REXP; i++) {
      b[i] = DCONST.r2[i];
    }

    Base c[REXP + 1];
#pragma unroll
    for (u32 i = 0; i < REXP + 1; i++) {
      c[i] = 0;
    }
    
    Base one[REXP + 1];
    one[REXP] = 0;
#pragma unroll
    for (u32 i = 0; i < REXP; i++) {
      one[i] = DCONST.one[i];
    }

    while (true)
    {
      {
        bool equal = true;
        for (u32 i = 0; i < REXP+1; ++i) {
          equal &= u[i] == one[i];
        }
        if (equal) {
          break;
        }
      }
      {
        bool equal = true;
        for (u32 i = 0; i < REXP+1; ++i) {
          equal &= v[i] == one[i];
        }
        if (equal) {
          break;
        }
      }

      while (true) {
        if ((u[0] & 1)) {
          break;
        }
        div_by_2(u);
        
        if ((b[0] & 1)) {
          add_p_unsafe(b);
        }
        div_by_2(b);
      }

      while (true)
      {
        if ((v[0] & 1)) {
          break;
        }
        div_by_2(v);

        if ((c[0] & 1)) {
          add_p_unsafe(c);
        }
        div_by_2(c);
      }

      sub(u, v, b, c);
    }
    
    Fp result;
    bool equal = true;
#pragma unroll
    for (u32 i = 0; i < REXP; ++i) {
      equal &= u[i] == DCONST.one[i];
    }

    if (equal) {
#pragma unroll
      for (u32 i = 0; i < REXP; i++) {
        result.digits[i] = b[i];
      }
    }
    else {
#pragma unroll
      for (u32 i = 0; i < REXP; i++) {
        result.digits[i] = c[i];
      }
    }
    return result;
  }

  __device__ __forceinline__ void div_by_2(Base *digits) const {
    for (size_t i = 0; i < REXP; i++) {
      digits[i] = ((digits[i + 1] << (Digit::Digit_WIDTH - 1)) | (digits[i] >> 1)) & Digit::MAX;
    }
    digits[REXP] = digits[REXP] >> 1;
  }