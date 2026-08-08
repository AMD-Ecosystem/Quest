#pragma once

// A ROCm PyTorch build puts __HIP_NO_HALF_CONVERSIONS__ on the command line, which strips
// the float conversion operators off __half. The kernels here convert with static_cast the
// way CUDA allows, so the shim asks hip_fp16.h for the full type.
#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__
#include <hip/hip_fp16.h>

using nv_half = __half;
using nv_half2 = __half2;
