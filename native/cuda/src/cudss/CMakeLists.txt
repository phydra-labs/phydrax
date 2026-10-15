find_library(PHX_CUDSS_LIBRARY NAMES cudss
  PATHS "${CUDAToolkit_LIBRARY_DIR}" "${CUDAToolkit_ROOT}/lib"
  NO_DEFAULT_PATH REQUIRED)
phx_cuda_add_component(cudss
  SOURCES solver.cpp
  LINK_LIBRARIES "${PHX_CUDSS_LIBRARY}"
  HANDLERS
    phydrax_cudss_analyze_f32 phydrax_cudss_analyze_f64
    phydrax_cudss_factorize_f32 phydrax_cudss_factorize_f64
    phydrax_cudss_refactorize_f32 phydrax_cudss_refactorize_f64
    phydrax_cudss_solve_f32 phydrax_cudss_solve_f64
    phydrax_cudss_diagonal_f32 phydrax_cudss_diagonal_f64
    phydrax_cudss_release)
