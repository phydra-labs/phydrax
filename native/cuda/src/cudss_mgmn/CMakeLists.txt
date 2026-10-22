# Explicit site MPI ABI and GPU-aware communication shim; never part of core cudss.
if(NOT TARGET phydrax_cuda_collective)
  message(FATAL_ERROR "cudss_mgmn requires the separately site-built collective component")
endif()
set(PHX_CUDSS_COMM_LAYER "" CACHE FILEPATH "Exact GPU-aware cuDSS MPI communication layer")
set(PHX_CUDSS_COMM_SHA256 "" CACHE STRING "Expected SHA256 of PHX_CUDSS_COMM_LAYER")
set(PHX_CUDSS_COMM_BUILD_IDENTITY "" CACHE STRING "Qualified communication layer build/ABI identity")
if(NOT IS_ABSOLUTE "${PHX_CUDSS_COMM_LAYER}" OR NOT EXISTS "${PHX_CUDSS_COMM_LAYER}" OR
   NOT PHX_CUDSS_COMM_SHA256 MATCHES "^[0-9a-f]+$" OR NOT PHX_CUDSS_COMM_BUILD_IDENTITY)
  message(FATAL_ERROR "cudss_mgmn requires an exact communication layer path, SHA256, and site build identity")
endif()
get_filename_component(PHX_CUDSS_COMM_LAYER "${PHX_CUDSS_COMM_LAYER}" REALPATH)
file(SHA256 "${PHX_CUDSS_COMM_LAYER}" _phx_cudss_comm_sha)
if(NOT _phx_cudss_comm_sha STREQUAL PHX_CUDSS_COMM_SHA256)
  message(FATAL_ERROR "cuDSS communication layer does not match its supplied SHA256")
endif()
find_library(PHX_CUDSS_LIBRARY NAMES cudss
  PATHS "${CUDAToolkit_LIBRARY_DIR}" "${CUDAToolkit_ROOT}/lib"
  NO_DEFAULT_PATH REQUIRED)
phx_cuda_add_component(cudss_mgmn
  SOURCES solver.cpp sparse_action.cu
  LINK_LIBRARIES "${PHX_CUDSS_LIBRARY}" phydrax_cuda_collective
  HANDLERS
    phydrax_cudss_mgmn_analyze_f32 phydrax_cudss_mgmn_analyze_f64
    phydrax_cudss_mgmn_factorize_f32 phydrax_cudss_mgmn_factorize_f64
    phydrax_cudss_mgmn_refactorize_f32 phydrax_cudss_mgmn_refactorize_f64
    phydrax_cudss_mgmn_solve_f32 phydrax_cudss_mgmn_solve_f64
    phydrax_cudss_mgmn_norm_f32 phydrax_cudss_mgmn_norm_f64
    phydrax_cudss_mgmn_exchange_f32 phydrax_cudss_mgmn_exchange_f64
    phydrax_cudss_mgmn_reduce_f32 phydrax_cudss_mgmn_reduce_f64 phydrax_cudss_mgmn_reduce_i32
    phydrax_cudss_mgmn_release)
target_compile_definitions(phydrax_cuda_cudss_mgmn PRIVATE
  OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1
  PHX_CUDSS_COMM_LAYER="${PHX_CUDSS_COMM_LAYER}"
  PHX_CUDSS_COMM_SHA256="${PHX_CUDSS_COMM_SHA256}"
  PHX_CUDSS_COMM_BUILD_IDENTITY="${PHX_CUDSS_COMM_BUILD_IDENTITY}")
