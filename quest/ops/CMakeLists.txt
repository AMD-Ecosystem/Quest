cmake_minimum_required(VERSION 3.26.4)  # Specify your minimum CMake version

# Must be set before project() call
option(USE_HIP "Build with HIP for AMD GPUs" OFF)

if(NOT USE_HIP)
  set(CMAKE_C_COMPILER "/usr/bin/gcc-11")
  set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")
  # enable_language(CUDA) caches nvcc's own default the moment this is unset, so it has
  # to be decided here rather than next to the other CUDA settings below.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
endif()

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Choose languages based on backend
# We MUST have exactly one project() call. Use enable_language() for the GPU language.
project(_kernels LANGUAGES CXX)

if(USE_HIP)
  # Same ordering constraint as CUDA above: enable_language(HIP) caches whatever
  # rocm_agent_enumerator reports for the machine doing the build unless the target is
  # already known, so let PYTORCH_ROCM_ARCH be read first. An explicit
  # -DCMAKE_HIP_ARCHITECTURES is already in the cache here and wins over both.
  # It has to reach the cache: a normal variable would suppress the enumerator branch that
  # normally writes that entry, leaving the build directory with no architecture at all for
  # the next configure -- which ninja runs by itself whenever a listfile changes. FORCE
  # cannot clobber a -DCMAKE_HIP_ARCHITECTURES because the guard already excluded that case.
  if(NOT CMAKE_HIP_ARCHITECTURES AND DEFINED ENV{PYTORCH_ROCM_ARCH})
    string(REPLACE " " ";" _quest_hip_archs "$ENV{PYTORCH_ROCM_ARCH}")
    set(CMAKE_HIP_ARCHITECTURES "${_quest_hip_archs}" CACHE STRING "HIP architectures" FORCE)
  endif()
  enable_language(HIP)
  # find_package(Torch) below overwrites CMAKE_HIP_ARCHITECTURES with the architecture
  # list PyTorch itself was built for, using a normal variable that hides the cache entry.
  # Keep the target resolved here and give that to the module instead.
  set(QUEST_HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})
  list(REMOVE_DUPLICATES QUEST_HIP_ARCHITECTURES)
  set(CMAKE_HIP_STANDARD 20)
  set(CMAKE_CXX_STANDARD 20)
  # nvcc optimizes device code whatever the build type, but clang compiles HIP device code
  # at -O0 when none is given, which leaves the kernels many times slower.
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  endif()
else()
  # CUDA build path (original)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  # ------------- configure rapids-cmake --------------#
  include(${CMAKE_SOURCE_DIR}/cmake/fetch_rapids.cmake)
  include(rapids-cmake)
  include(rapids-cpm)
  include(rapids-cuda)
  include(rapids-export)
  include(rapids-find)
  rapids_cpm_init()
  include(${CMAKE_SOURCE_DIR}/cmake/get_raft.cmake)
endif()

# Check: https://stackoverflow.com/questions/68401650/how-can-i-make-a-pytorch-extension-with-cmake
# Fix linking error: https://github.com/pytorch/pytorch/issues/108041
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

# Try combine pybind
# Check: https://qiita.com/syoyo/items/c3e8e6e5c3e2d69c2325
add_subdirectory(${CMAKE_SOURCE_DIR}/../../kernels/3rdparty/pybind ${CMAKE_BINARY_DIR}/pybind11)

if(USE_HIP)
  # The prefill kernels are left out of the ROCm build; see csrc/bsk_ops.h.
  set(PYTORCH_SOURCES
    csrc/rms_norm.cu
    csrc/page.cu
    csrc/estimate.cu
    csrc/approx_attn.cu
    csrc/topk.cu
    csrc/bsk_ops.cu
  )
  set_source_files_properties(${PYTORCH_SOURCES} PROPERTIES LANGUAGE HIP)
else()
  file(GLOB PYTORCH_SOURCES "csrc/*.cu")
endif()

pybind11_add_module(_kernels MODULE ${PYTORCH_CPP_SOURCES} ${PYTORCH_SOURCES})

if(USE_HIP)
  # Not ${CMAKE_HIP_ARCHITECTURES}: find_package(Torch) has replaced it by now.
  set_target_properties(_kernels PROPERTIES HIP_ARCHITECTURES "${QUEST_HIP_ARCHITECTURES}")
endif()

target_compile_definitions(_kernels PRIVATE -DBSK_TORCH_CHECK) # Enable Torch Tensor Dimension Check
target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/../../kernels/include)
target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/../../kernels/3rdparty/flashinfer/include)
target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/../../kernels/3rdparty/pybind/include)

if(USE_HIP)
  target_compile_definitions(_kernels PRIVATE USE_HIP)
  # clang defaults to -ffp-contract=fast, which contracts across expressions and drifts
  # further from the reference than nvcc's expression-only contraction.
  target_compile_options(_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-compress -ffp-contract=on>
  )
  # Ahead of everything else so the CUDA toolkit headers the kernels include by name
  # resolve to the HIP shims in kernels/include/hip_compat.
  set(HIP_COMPAT_DIR ${CMAKE_SOURCE_DIR}/../../kernels/include/hip_compat)
  target_include_directories(_kernels BEFORE PRIVATE ${HIP_COMPAT_DIR})
  # These two flashinfer headers are pulled in by a sibling-relative include, which the
  # include path cannot shadow, so they are force-included ahead of it; each carries the
  # upstream include guard, which turns the CUDA original into a no-op. Editing either one
  # does not invalidate an object file, so wipe the build directory after touching them.
  target_compile_options(_kernels PRIVATE $<$<COMPILE_LANGUAGE:HIP>:
    -include${HIP_COMPAT_DIR}/flashinfer/math.cuh
    -include${HIP_COMPAT_DIR}/flashinfer/vec_dtypes.cuh>)
  target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/csrc)
else()
  target_compile_options(_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
endif()

if(USE_HIP)
  target_link_libraries(_kernels PRIVATE ${TORCH_LIBRARIES} Python::Python pybind11::module ${TORCH_PYTHON_LIBRARY})
else()
  target_link_libraries(_kernels PRIVATE ${TORCH_LIBRARIES} raft::raft Python::Python pybind11::module ${TORCH_PYTHON_LIBRARY})
endif()
