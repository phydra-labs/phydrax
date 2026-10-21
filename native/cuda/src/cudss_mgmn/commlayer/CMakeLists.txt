cmake_minimum_required(VERSION 3.28)
# Build NVIDIA's unmodified cuDSS 0.8.0.10 OpenMPI communication source against
# the explicit site ABI. This is not a portable companion-wheel component.
project(phydrax_cudss_site_commlayer LANGUAGES CXX CUDA)

set(PHX_SITE_MPI_ROOT "" CACHE PATH "Exact GPU-aware site OpenMPI prefix")
set(PHX_CUDSS_INCLUDE_DIR "" CACHE PATH "Wheel-owned cuDSS 0.8 include directory")
set(PHX_CUDSS_COMM_SOURCE "" CACHE FILEPATH "NVIDIA cuDSS 0.8.0.10 src/cudss_commlayer_openmpi.cu")
# Official CUDA 13 x86-64 archive, indexed by redistrib_0.8.0.json:
# https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-0.8.0.10_cuda13-archive.tar.xz
# Archive SHA256: ba18f5fd80dcbbe905d158caac5b3061d848442bb5abd477b5f296b4257a4937
set(_phx_source_sha "327f641753624fcfd07fd67be0342b48b39ddccb323625c30a27b4a36aad12d9")
foreach(path PHX_SITE_MPI_ROOT PHX_CUDSS_INCLUDE_DIR PHX_CUDSS_COMM_SOURCE CUDAToolkit_ROOT)
  if(NOT IS_ABSOLUTE "${${path}}" OR NOT EXISTS "${${path}}")
    message(FATAL_ERROR "${path} must identify its explicit existing site/wheel-owned artifact")
  endif()
endforeach()
file(SHA256 "${PHX_CUDSS_COMM_SOURCE}" _phx_actual_source_sha)
if(NOT _phx_actual_source_sha STREQUAL _phx_source_sha)
  message(FATAL_ERROR "Communication source is not the pinned unmodified NVIDIA 0.8.0.10 source")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13 OR
   NOT CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 14)
  message(FATAL_ERROR "The site communication layer requires the wheel-owned CUDA 13 compiler")
endif()
find_package(CUDAToolkit 13 REQUIRED)
if(NOT CUDAToolkit_VERSION VERSION_LESS 14)
  message(FATAL_ERROR "The communication layer requires CUDA major 13")
endif()
find_path(PHX_COMM_MPI_INCLUDE mpi.h PATHS "${PHX_SITE_MPI_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(PHX_COMM_MPI_LIBRARY NAMES mpi PATHS "${PHX_SITE_MPI_ROOT}/lib" "${PHX_SITE_MPI_ROOT}/lib64" NO_DEFAULT_PATH REQUIRED)
get_filename_component(PHX_COMM_MPI_LIBRARY "${PHX_COMM_MPI_LIBRARY}" REALPATH)
get_filename_component(_phx_mpi_libdir "${PHX_COMM_MPI_LIBRARY}" DIRECTORY)
file(SHA256 "${PHX_COMM_MPI_LIBRARY}" _phx_mpi_sha)
file(SHA256 "${PHX_COMM_MPI_INCLUDE}/mpi.h" _phx_mpi_header_sha)
file(SHA256 "${PHX_CUDSS_INCLUDE_DIR}/cudss_distributed_interface.h" _phx_interface_sha)
string(SHA256 PHX_CUDSS_COMM_BUILD_IDENTITY
  "${_phx_source_sha}:${_phx_interface_sha}:${_phx_mpi_sha}:${_phx_mpi_header_sha}:${CMAKE_CUDA_COMPILER}:${CMAKE_CUDA_COMPILER_VERSION}:${CMAKE_CUDA_FLAGS}:${CMAKE_SHARED_LINKER_FLAGS}")

add_library(cudss_commlayer_openmpi SHARED "${PHX_CUDSS_COMM_SOURCE}")
target_include_directories(cudss_commlayer_openmpi PRIVATE "${PHX_CUDSS_INCLUDE_DIR}" "${PHX_COMM_MPI_INCLUDE}")
target_compile_definitions(cudss_commlayer_openmpi PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)
target_link_libraries(cudss_commlayer_openmpi PRIVATE "${PHX_COMM_MPI_LIBRARY}" CUDA::cudart)
set_target_properties(cudss_commlayer_openmpi PROPERTIES
  SOVERSION 0 VERSION 0.8.0 CUDA_RUNTIME_LIBRARY Shared
  BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "${_phx_mpi_libdir};${CUDAToolkit_LIBRARY_DIR}"
  INSTALL_RPATH_USE_LINK_PATH OFF)
target_link_options(cudss_commlayer_openmpi PRIVATE "LINKER:-z,defs" "LINKER:--enable-new-dtags")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/communication-build.json" CONTENT
"{\n  \"source_sha256\": \"${_phx_source_sha}\",\n  \"vendor_interface_sha256\": \"${_phx_interface_sha}\",\n  \"mpi_library\": \"${PHX_COMM_MPI_LIBRARY}\",\n  \"mpi_library_sha256\": \"${_phx_mpi_sha}\",\n  \"mpi_header_sha256\": \"${_phx_mpi_header_sha}\",\n  \"build_identity\": \"${PHX_CUDSS_COMM_BUILD_IDENTITY}\",\n  \"library_path\": \"$<TARGET_FILE:cudss_commlayer_openmpi>\",\n  \"communicators\": [\"host-mpi\", \"device-mpi\"]\n}\n")
message(STATUS "PHX_CUDSS_COMM_BUILD_IDENTITY=${PHX_CUDSS_COMM_BUILD_IDENTITY}")
