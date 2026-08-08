# hip_compat

Shadow headers used only by the ROCm/HIP build (`-DUSE_HIP=ON`). This directory is
placed ahead of the CUDA include path so that sources including `<cuda_fp16.h>`,
`<cuda_runtime.h>`, `<cooperative_groups.h>` or `flashinfer/math.cuh` pick up a HIP
implementation without any edit to the kernel sources themselves. The CUDA build never
sees this directory, so its compilation is unchanged.
