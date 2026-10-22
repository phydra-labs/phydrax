# Site-linked cuDecomp is optional and is never fetched or bundled here.
set(PHX_CUDECOMP_ROOT "" CACHE PATH "Explicit pinned cuDecomp 0.7.0 site installation")
if(NOT PHX_CUDECOMP_ROOT)
  message(FATAL_ERROR "cudecomp requires the explicit PHX_CUDECOMP_ROOT site prefix")
endif()
if(NOT TARGET phydrax_cuda_collective)
  message(FATAL_ERROR "cudecomp requires the shared site MPI collective component before this subdirectory")
endif()
find_path(PHX_CUDECOMP_MPI_EXT_INCLUDE mpi-ext.h
  PATHS "${PHX_SITE_MPI_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
foreach(_header mpi-ext.h openmpi/mpiext/mpiext_cuda_c.h)
  if(NOT EXISTS "${PHX_CUDECOMP_MPI_EXT_INCLUDE}/${_header}")
    message(FATAL_ERROR "cudecomp requires the qualified site Open MPI CUDA-awareness extension")
  endif()
  file(SHA256 "${PHX_CUDECOMP_MPI_EXT_INCLUDE}/${_header}" _header_hash)
  string(APPEND _phx_cudecomp_headers ":${_header}:${_header_hash}")
endforeach()
find_path(PHX_CUDECOMP_INCLUDE cudecomp.h
  PATHS "${PHX_CUDECOMP_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(PHX_CUDECOMP_LIBRARY NAMES cudecomp
  PATHS "${PHX_CUDECOMP_ROOT}/lib" "${PHX_CUDECOMP_ROOT}/lib64" NO_DEFAULT_PATH REQUIRED)
get_filename_component(PHX_CUDECOMP_LIBRARY "${PHX_CUDECOMP_LIBRARY}" REALPATH)
foreach(_header cudecomp.h cudecomp_config.h cudecomp_version.h)
  if(NOT EXISTS "${PHX_CUDECOMP_INCLUDE}/${_header}")
    message(FATAL_ERROR "PHX_CUDECOMP_ROOT must include the installed generated ${_header}")
  endif()
  file(SHA256 "${PHX_CUDECOMP_INCLUDE}/${_header}" _header_hash)
  string(APPEND _phx_cudecomp_headers ":${_header}:${_header_hash}")
endforeach()
file(SHA256 "${PHX_CUDECOMP_LIBRARY}" _phx_cudecomp_library_hash)
get_target_property(_phx_collective_definitions phydrax_cuda_collective COMPILE_DEFINITIONS)
string(SHA256 PHX_CUDECOMP_BUILD_IDENTITY
  "${PHX_CUDA_BUILD_HASH}:73121159fec42eb9144dfed9159e53fde6603fca:${_phx_cudecomp_library_hash}:${_phx_cudecomp_headers}:${_phx_collective_definitions}")
phx_cuda_add_component(cudecomp
  SOURCES transpose.cpp
  LINK_LIBRARIES "${PHX_CUDECOMP_LIBRARY}" phydrax_cuda_collective "${CMAKE_DL_LIBS}"
  HANDLERS phydrax_cudecomp_transpose_c64 phydrax_cudecomp_transpose_c128)
target_include_directories(phydrax_cuda_cudecomp PRIVATE
  "${PHX_CUDECOMP_INCLUDE}" "${PHX_CUDECOMP_MPI_EXT_INCLUDE}")
target_compile_definitions(phydrax_cuda_cudecomp PRIVATE
  PHX_CUDECOMP_LIBRARY="${PHX_CUDECOMP_LIBRARY}"
  PHX_CUDECOMP_BUILD_IDENTITY="${PHX_CUDECOMP_BUILD_IDENTITY}")
get_filename_component(_phx_cudecomp_libdir "${PHX_CUDECOMP_LIBRARY}" DIRECTORY)
set_target_properties(phydrax_cuda_cudecomp PROPERTIES
  BUILD_RPATH "${PHX_CUDA_RUNPATH};${_phx_cudecomp_libdir}"
  INSTALL_RPATH "${PHX_CUDA_RUNPATH};${_phx_cudecomp_libdir}")
