include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/PhxQuda.cmake)
phx_cuda_add_component(quda SOURCES quda.cpp quda_kernels.cu LINK_LIBRARIES phx_quda
  HANDLERS phydrax_quda_gauge_action_c64 phydrax_quda_gauge_action_c128
    phydrax_quda_gauge_force_c64 phydrax_quda_gauge_force_c128
    phydrax_quda_wilson_dslash_c64 phydrax_quda_wilson_dslash_c128)
# These launches belong to the serialized legacy-default-stream boundary.
set_source_files_properties(quda_kernels.cu PROPERTIES
  COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--default-stream=legacy>")
