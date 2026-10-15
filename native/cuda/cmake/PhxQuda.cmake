include_guard(GLOBAL)
include(FetchContent)

# Eigen remains an unconditional dependency: eigen_helper.h and the eigensolver
# and generic BLAS sources instantiate it even with QUDA_USE_EIGEN=OFF. Keep the
# source pin explicit and redistribute its MPL-2.0 notice with the companion.
FetchContent_Declare(phx_quda_eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/e67c494cba7180066e73b9f6234d0b2129f1cdf5.tar.bz2
  URL_HASH SHA256=98d244932291506b75c4ae7459af29b1112ea3d2f04660686a925d9ef6634583
  DOWNLOAD_DIR "${PHX_CUDA_DOWNLOAD_DIR}"
  DOWNLOAD_NAME eigen-e67c494.tar.bz2
  SOURCE_SUBDIR phydrax-download-only)
FetchContent_MakeAvailable(phx_quda_eigen)
set(EIGEN_INCLUDE_DIR "${phx_quda_eigen_SOURCE_DIR}" CACHE PATH "Pinned QUDA Eigen headers" FORCE)

set(QUDA_BUILD_SHAREDLIB OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
foreach(option MPI QMP MULTIGRID BLOCKSOLVER BUILD_ALL_TESTS INSTALL_ALL_TESTS
    BUILD_NATIVE_FFT OPENMP USE_EIGEN BACKWARDS SSTEP DOWNLOAD_CCCL DOWNLOAD_EIGEN
    INTERFACE_MILC INTERFACE_CPS INTERFACE_QDPJIT INTERFACE_BQCD INTERFACE_TIFR
    INTERFACE_OPENQCD INTERFACE_ALL)
  set(QUDA_${option} OFF CACHE BOOL "" FORCE)
endforeach()
set(QUDA_INTERFACE_QDP ON CACHE BOOL "" FORCE)
# QDP is the device gauge/momentum pointer-array layout used by our handlers.
set(QUDA_DIRAC_DEFAULT_OFF ON CACHE BOOL "" FORCE)
set(QUDA_DIRAC_WILSON ON CACHE BOOL "" FORCE)
set(QUDA_DIRAC_CLOVER ON CACHE BOOL "" FORCE)
set(QUDA_PRECISION 12 CACHE STRING "Double and single precision only" FORCE)
set(QUDA_RECONSTRUCT 4 CACHE STRING "Unreconstructed SU(3) only" FORCE)
set(QUDA_FAST_COMPILE_REDUCE ON CACHE BOOL "" FORCE)
set(QUDA_MAX_MULTI_RHS 1 CACHE STRING "Sources are dispatched individually" FORCE)
# QUDA selects tuning defaults from one baseline, while its actual emitted
# architectures below come from the companion's canonical architecture setting.
set(_phx_quda_arches "${CMAKE_CUDA_ARCHITECTURES}")
if(PHX_CUDA_ARCHITECTURES STREQUAL "native")
  set(_phx_quda_baselines "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
else()
  set(_phx_quda_baselines "${PHX_CUDA_ARCHITECTURES}")
endif()
list(TRANSFORM _phx_quda_baselines REPLACE "-(real|virtual)$" "")
list(SORT _phx_quda_baselines COMPARE NATURAL ORDER ASCENDING)
list(GET _phx_quda_baselines 0 _phx_quda_baseline)
set(QUDA_GPU_ARCH "sm_${_phx_quda_baseline}" CACHE STRING "" FORCE)

FetchContent_Declare(phx_quda_source
  URL https://codeload.github.com/lattice/quda/tar.gz/ba501e4
  URL_HASH SHA256=3dabbec974f83a6daf8a2d55f38d45ff9561e636ddae4c1ad5c7c19424c70f0c
  DOWNLOAD_DIR "${PHX_CUDA_DOWNLOAD_DIR}"
  DOWNLOAD_NAME quda-ba501e4.tar.gz
  PATCH_COMMAND patch -p1 --forward -i "${CMAKE_CURRENT_LIST_DIR}/../patches/quda-device-gauge-force.patch"
    COMMAND patch -p1 --forward -i "${CMAKE_CURRENT_LIST_DIR}/../patches/quda-cmake-subproject.patch"
    COMMAND patch -p1 --forward -i "${CMAKE_CURRENT_LIST_DIR}/../patches/quda-portable-warp-election.patch"
  SOURCE_SUBDIR phydrax-download-only)
FetchContent_MakeAvailable(phx_quda_source)
# An archive must not identify the enclosing Phydrax git tree.
set(CMAKE_DISABLE_FIND_PACKAGE_Git TRUE)
set(GITVERSION ba501e4)
add_subdirectory("${phx_quda_source_SOURCE_DIR}" "${phx_quda_source_BINARY_DIR}" EXCLUDE_FROM_ALL)
set_target_properties(quda quda_cpp PROPERTIES POSITION_INDEPENDENT_CODE ON)
# CUDA 13 vectorized gauge access uses the bundled NVIDIA trove headers.
target_include_directories(quda SYSTEM PRIVATE "${phx_quda_source_SOURCE_DIR}/externals")
set_target_properties(quda PROPERTIES CUDA_ARCHITECTURES "${_phx_quda_arches}"
  CUDA_RUNTIME_LIBRARY Shared CUDA_STANDARD 20 CXX_STANDARD 20)
# Upstream propagates cudart_static even for a static QUDA archive. The
# companion must bind the same shared runtime as the selected JAX plugin.
get_target_property(_phx_quda_links quda INTERFACE_LINK_LIBRARIES)
list(REMOVE_ITEM _phx_quda_links CUDA::cudart_static)
set_property(TARGET quda PROPERTY INTERFACE_LINK_LIBRARIES "${_phx_quda_links}")
target_link_libraries(quda INTERFACE CUDA::cudart)
add_library(phx_quda ALIAS quda)
