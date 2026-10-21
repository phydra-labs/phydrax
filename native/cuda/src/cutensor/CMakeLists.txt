phx_cuda_add_component(cutensor
  SOURCES cutensor_ffi.cpp
  LINK_LIBRARIES CUDA::cutensor
  HANDLERS phydrax_cutensor_contract)
