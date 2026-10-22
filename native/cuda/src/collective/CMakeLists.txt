# This component is site-built; never resolve or bundle an incidental MPI.
set(PHX_SITE_MPI_ROOT "" CACHE PATH "Explicit initialized site MPI installation")
if(NOT PHX_SITE_MPI_ROOT)
  message(FATAL_ERROR "collective requires PHX_SITE_MPI_ROOT and a site-initialized MPI runtime")
endif()
find_path(PHX_SITE_MPI_INCLUDE mpi.h PATHS "${PHX_SITE_MPI_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(PHX_SITE_MPI_LIBRARY NAMES mpi PATHS "${PHX_SITE_MPI_ROOT}/lib" "${PHX_SITE_MPI_ROOT}/lib64" NO_DEFAULT_PATH REQUIRED)
get_filename_component(PHX_SITE_MPI_LIBRARY "${PHX_SITE_MPI_LIBRARY}" REALPATH)
file(SHA256 "${PHX_SITE_MPI_LIBRARY}" PHX_SITE_MPI_SHA256)
file(SHA256 "${PHX_SITE_MPI_INCLUDE}/mpi.h" PHX_SITE_MPI_HEADER_SHA256)
string(SHA256 PHX_COLLECTIVE_BUILD_IDENTITY
  "${PHX_CUDA_BUILD_HASH}:${PHX_SITE_MPI_SHA256}:${PHX_SITE_MPI_HEADER_SHA256}:${CMAKE_CXX_COMPILER_ID}:${CMAKE_CXX_COMPILER_VERSION}:${CMAKE_CXX_FLAGS}")
find_package(Threads REQUIRED)
add_library(phydrax_cuda_collective SHARED
  "${CMAKE_CURRENT_LIST_DIR}/../common/collective_lease.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../common/identity.cpp")
target_compile_features(phydrax_cuda_collective PUBLIC cxx_std_17)
target_include_directories(phydrax_cuda_collective PUBLIC
  "${CMAKE_CURRENT_LIST_DIR}/../common" "${PHX_SITE_MPI_INCLUDE}"
  PRIVATE "${PHX_CUDA_SOURCE_DIR}/include")
target_link_libraries(phydrax_cuda_collective PUBLIC "${PHX_SITE_MPI_LIBRARY}" PRIVATE Threads::Threads "${CMAKE_DL_LIBS}")
target_compile_definitions(phydrax_cuda_collective PUBLIC OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)
target_compile_definitions(phydrax_cuda_collective PRIVATE
  PHX_CUDA_COMPONENT="collective" PHX_CUDA_VERSION="${PHX_CUDA_VERSION}"
  PHX_CUDA_BUILD_HASH="${PHX_CUDA_BUILD_HASH}" PHX_CUDA_MAJOR=13
  PHX_COLLECTIVE_BUILD_IDENTITY="${PHX_COLLECTIVE_BUILD_IDENTITY}"
  PHX_COLLECTIVE_MPI_LIBRARY="${PHX_SITE_MPI_LIBRARY}"
  PHX_COLLECTIVE_MPI_SHA256="${PHX_SITE_MPI_SHA256}"
  PHX_COLLECTIVE_MPI_HEADER_SHA256="${PHX_SITE_MPI_HEADER_SHA256}")
get_filename_component(_phx_site_mpi_libdir "${PHX_SITE_MPI_LIBRARY}" DIRECTORY)
set_target_properties(phydrax_cuda_collective PROPERTIES
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
  BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "${_phx_site_mpi_libdir}"
  INSTALL_RPATH_USE_LINK_PATH OFF)
target_link_options(phydrax_cuda_collective PRIVATE "LINKER:-z,defs" "LINKER:--enable-new-dtags")
# The MPI runtime remains at its site prefix; only our bridge is installed.
install(TARGETS phydrax_cuda_collective LIBRARY DESTINATION "phydrax_cuda13/lib")
