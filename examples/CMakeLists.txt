# Hill's vortex example, built for NVIDIA (CUDA) or AMD (HIP) GPUs depending
# on the backend Kokkos was built with. Override with -DMLC_GPU_BACKEND=CUDA|HIP.

##  FFTX_HOME must be defined in the environment or on the command line
if ( DEFINED ENV{FFTX_HOME} )
  message ( STATUS "FFTX_HOME = $ENV{FFTX_HOME}" )
  set ( FFTX_SOURCE_DIR $ENV{FFTX_HOME} )
else ()
  if ( "x${FFTX_HOME}" STREQUAL "x" )
    message ( FATAL_ERROR "FFTX_HOME environment variable undefined and not specified on command line" )
  endif ()
  set ( FFTX_SOURCE_DIR ${FFTX_HOME} )
endif ()

##  Include FFTX CMake functions
include ( "${FFTX_SOURCE_DIR}/CMakeIncludes/FFTXCmakeFunctions.cmake" )

# --- Pick the GPU backend from the Kokkos build ---
if ( NOT MLC_GPU_BACKEND )
  if ( Kokkos_ENABLE_HIP OR "HIP" IN_LIST Kokkos_DEVICES )
    set ( MLC_GPU_BACKEND HIP )
  elseif ( Kokkos_ENABLE_CUDA OR "CUDA" IN_LIST Kokkos_DEVICES )
    set ( MLC_GPU_BACKEND CUDA )
  else ()
    message ( FATAL_ERROR "Kokkos was built without CUDA or HIP; set -DMLC_GPU_BACKEND=CUDA or HIP" )
  endif ()
endif ()
message ( STATUS "MLC GPU backend = ${MLC_GPU_BACKEND}" )

# --- Hill's vortex example ---
add_executable( Hill Hills_Vortex.cpp )
target_include_directories( Hill PRIVATE ${FFTX_SOURCE_DIR}/examples/rconv )

if ( MLC_GPU_BACKEND STREQUAL "HIP" )

  find_package(hipfft CONFIG REQUIRED)   # provides hip::hipfft
  find_package(rocfft CONFIG REQUIRED)   # provides roc::rocfft

  # ROCm core install (default is the MI300 Faraday location)
  set(ROCM_CORE_ROOT "/opt/rocm/core-7.14" CACHE PATH "ROCm core root")
  # GPU architecture: gfx942 for MI300, gfx908 for MI100
  set(MLC_AMD_GPU_ARCH "gfx942" CACHE STRING "AMD GPU offload architecture")

  target_include_directories( Hill PRIVATE ${ROCM_CORE_ROOT}/include )
  target_link_libraries( Hill PRIVATE exampm dl hiprtc hip::hipfft roc::rocfft )
  FFTX_add_includes_libs_to_target ( Hill )
  target_compile_definitions( Hill PRIVATE FFTX_HIP )

  target_compile_options( Hill PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
    $<$<COMPILE_LANGUAGE:CXX>:--offload-arch=${MLC_AMD_GPU_ARCH}>
    $<$<COMPILE_LANGUAGE:CXX>:SHELL:-mllvm -unroll-threshold=2000>
    $<$<COMPILE_LANGUAGE:CXX>:SHELL:-mllvm -unroll-allow-partial>
  )

elseif ( MLC_GPU_BACKEND STREQUAL "CUDA" )

  target_link_libraries( Hill PRIVATE exampm dl nvrtc )
  FFTX_add_includes_libs_to_target( Hill LIB_DIR ${FFTX_LIB_DIR} )
  target_compile_definitions( Hill PRIVATE FFTX_CUDA )

else ()
  message ( FATAL_ERROR "Unknown MLC_GPU_BACKEND '${MLC_GPU_BACKEND}' (use CUDA or HIP)" )
endif ()

install(TARGETS Hill DESTINATION ${CMAKE_INSTALL_BINDIR})
