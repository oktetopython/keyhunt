/**
 * SharedMemLayout.cu
 * Implementation file for static member definitions
 */

#include "SharedMemLayout.cuh"

// Definition of static member - must be in device space
__device__ int DynamicSharedMemAllocator::offset = 0;