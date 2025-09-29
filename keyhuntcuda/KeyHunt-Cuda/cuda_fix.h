#ifndef CUDA_FIX_H
#define CUDA_FIX_H

// Workaround for CUDA/GCC compatibility issues
#ifdef __CUDACC__
    // Disable problematic intrinsics when compiling CUDA code
    #define __AMX__ 0
    #define _AMXTILEINTRIN_H_INCLUDED
    
    // Float types are already defined in the system headers
#endif

#endif // CUDA_FIX_H