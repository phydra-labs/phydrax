include(${PROJECT_SOURCE_DIR}/cmake/PhxAmgX.cmake)
phx_cuda_add_component(amgx
  SOURCES amgx_ffi.cpp
  LINK_LIBRARIES phx_amgx_static
  HANDLERS phydrax_amgx_setup phydrax_amgx_refresh phydrax_amgx_solve phydrax_amgx_release)
