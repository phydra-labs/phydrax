find_library(PHX_CUDSS_LIBRARY NAMES cudss
  PATHS "${CUDAToolkit_LIBRARY_DIR}" "${CUDAToolkit_ROOT}/lib"
  NO_DEFAULT_PATH REQUIRED)
add_library(phx_vendor_cudss SHARED IMPORTED)
if(WIN32)
  set_target_properties(phx_vendor_cudss PROPERTIES
    IMPORTED_IMPLIB "${PHX_CUDSS_LIBRARY}"
    IMPORTED_LOCATION "${PHX_WINDOWS_CUDSS_DLL}")
else()
  set_target_properties(phx_vendor_cudss PROPERTIES IMPORTED_LOCATION "${PHX_CUDSS_LIBRARY}")
endif()
phx_cuda_add_component(cudss
  SOURCES solver.cpp
  LINK_LIBRARIES phx_vendor_cudss
  HANDLERS
    phydrax_cudss_analyze_f32 phydrax_cudss_analyze_f64
    phydrax_cudss_factorize_f32 phydrax_cudss_factorize_f64
    phydrax_cudss_refactorize_f32 phydrax_cudss_refactorize_f64
    phydrax_cudss_solve_f32 phydrax_cudss_solve_f64
    phydrax_cudss_diagonal_f32 phydrax_cudss_diagonal_f64
    phydrax_cudss_release)
