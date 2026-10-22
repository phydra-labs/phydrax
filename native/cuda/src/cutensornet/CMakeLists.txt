phx_cuda_add_component(cutensornet
  SOURCES cutensornet_ffi.cpp
  LINK_LIBRARIES CUDA::cutensornet
  HANDLERS phydrax_cutensornet_contract)
