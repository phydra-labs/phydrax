# The Python nvidia-cuest-cu13 wheel does not ship the C development interface.
# Use NVIDIA's official, immutable CUDA 13 C redistributable instead.
if(CUDAToolkit_VERSION VERSION_LESS "13.0.2")
  message(FATAL_ERROR "cuEST 0.2.2.2 requires CUDA Toolkit >= 13.0.2")
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "cuEST C redistributables require Linux")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  set(_phx_cuest_platform "linux-x86_64")
  set(_phx_cuest_sha "71641ba4aaa068dad037b9837f6fd0920edb7283f8297eccb9bedba6f405d228")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(_phx_cuest_platform "linux-sbsa")
  set(_phx_cuest_sha "413e9050d2c1e998a0a6eb803e1c468217fbeebe25e5e80068c22b5ce8e66c31")
else()
  message(FATAL_ERROR "Unsupported cuEST C redistributable architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()
if(NOT DEFINED PHX_CUDA_DOWNLOAD_DIR)
  set(PHX_CUDA_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/downloads")
endif()
# Share the optional runtime wheel's archive cache. A per-archive lock protects
# concurrent configurations, and cached downloads are never trusted without SHA.
function(_phx_cuest_archive platform sha)
  set(stem "libcuest-${platform}-0.2.2.2_cuda13-archive")
  set(archive "${PHX_CUDA_DOWNLOAD_DIR}/${stem}.tar.xz")
  file(MAKE_DIRECTORY "${PHX_CUDA_DOWNLOAD_DIR}")
  file(LOCK "${PHX_CUDA_DOWNLOAD_DIR}/${stem}.lock" GUARD FUNCTION TIMEOUT 600)
  if(NOT EXISTS "${archive}")
    file(DOWNLOAD
      "https://developer.download.nvidia.com/compute/cuest/redist/libcuest/${platform}/${stem}.tar.xz"
      "${archive}" EXPECTED_HASH "SHA256=${sha}" TLS_VERIFY ON)
  endif()
  file(SHA256 "${archive}" actual_sha)
  if(NOT actual_sha STREQUAL sha)
    message(FATAL_ERROR "cuEST C redistributable archive SHA256 mismatch")
  endif()
  set(extracted "${CMAKE_BINARY_DIR}/cuest-support/${stem}")
  if(NOT EXISTS "${extracted}")
    file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${CMAKE_BINARY_DIR}/cuest-support")
  endif()
  set(PHX_CUEST_ROOT "${extracted}" CACHE INTERNAL "Pinned cuEST C redistributable" FORCE)
endfunction()
_phx_cuest_archive("${_phx_cuest_platform}" "${_phx_cuest_sha}")
add_library(phx_cuest_vendor_shared SHARED IMPORTED)
set_target_properties(phx_cuest_vendor_shared PROPERTIES
  IMPORTED_LOCATION "${PHX_CUEST_ROOT}/lib/libcuest.so.0.2.2"
  IMPORTED_SONAME "libcuest.so.0"
  INTERFACE_INCLUDE_DIRECTORIES "${PHX_CUEST_ROOT}/include")
phx_cuda_add_component(cuest
  SOURCES cuest_ffi.cpp
  LINK_LIBRARIES phx_cuest_vendor_shared CUDA::cublas CUDA::cusolver
  HANDLERS phydrax_cuest_bind_geometry phydrax_cuest_one_electron phydrax_cuest_jk phydrax_cuest_gradient)
# The optional runtime wheel owns this installed path; absolute loader preloading
# is additional protection, not a replacement for a relocatable dependency edge.
set_property(TARGET phydrax_cuda_cuest APPEND PROPERTY
  INSTALL_RPATH "$ORIGIN/../../phydrax_cuest/lib")
# Shared packaging owns runtime dependency copying and RECORD generation.
# Keep NVIDIA's license, including third-party notices, beside the component.
install(FILES "${PHX_CUEST_ROOT}/LICENSE"
  DESTINATION "phydrax_cuda13/licenses" RENAME "cuEST-LICENSE.txt")
