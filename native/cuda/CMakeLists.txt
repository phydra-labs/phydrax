cmake_minimum_required(VERSION 3.28)
set(PHX_CUDA_COMPONENTS "cudss;amgx;quda;cutensor;cutensornet;cuest" CACHE STRING "Companion components")
foreach(component IN LISTS PHX_CUDA_COMPONENTS)
  if(NOT component MATCHES "^(cudss|amgx|quda|cutensor|cutensornet|cuest|collective|cudecomp|cudss_mgmn)$")
    message(FATAL_ERROR "Unknown CUDA component: ${component}")
  endif()
endforeach()
if(("cudecomp" IN_LIST PHX_CUDA_COMPONENTS OR "cudss_mgmn" IN_LIST PHX_CUDA_COMPONENTS)
   AND NOT "collective" IN_LIST PHX_CUDA_COMPONENTS)
  message(FATAL_ERROR "Site collective providers require the collective component in the same build")
endif()
set(PHX_CUDA_ARCHITECTURES "80-real;86-real;89-real;90-real;100-real;120-real;120-virtual"
  CACHE STRING "CUDA architectures; native for development")
set(CMAKE_CUDA_ARCHITECTURES "${PHX_CUDA_ARCHITECTURES}")
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(PHX_CUDA_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(PhxPipCudaToolkit)
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/../cuda13/pyproject.toml" _phx_version_line
  REGEX "^version = \"[0-9]+\\.[0-9]+\\.[0-9]+\"$" LIMIT_COUNT 1)
string(REGEX REPLACE "^version = \"([0-9.]+)\"$" "\\1" PHX_CUDA_VERSION "${_phx_version_line}")
if(NOT PHX_CUDA_VERSION)
  message(FATAL_ERROR "Companion pyproject.toml must declare a release version")
endif()
project(phydrax_cuda VERSION ${PHX_CUDA_VERSION} LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
find_package(CUDAToolkit REQUIRED GLOBAL)
find_package(Python REQUIRED COMPONENTS Interpreter)
if("cutensor" IN_LIST PHX_CUDA_COMPONENTS OR "cutensornet" IN_LIST PHX_CUDA_COMPONENTS
   OR "cudecomp" IN_LIST PHX_CUDA_COMPONENTS)
  find_path(PHX_CUTENSOR_INCLUDE_DIR cutensor.h
    PATHS "${PHX_CUDA_TOOLKIT}/include" NO_DEFAULT_PATH REQUIRED)
  find_library(PHX_CUTENSOR_LIBRARY NAMES cutensor
    PATHS "${PHX_CUDA_TOOLKIT}/lib" NO_DEFAULT_PATH REQUIRED)
  add_library(CUDA::cutensor SHARED IMPORTED GLOBAL)
  set_target_properties(CUDA::cutensor PROPERTIES
    IMPORTED_LOCATION "${PHX_CUTENSOR_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${PHX_CUTENSOR_INCLUDE_DIR}")
endif()
if("cutensornet" IN_LIST PHX_CUDA_COMPONENTS)
  find_path(PHX_CUTENSORNET_INCLUDE_DIR cutensornet.h
    PATHS "${PHX_CUDA_TOOLKIT}/include" NO_DEFAULT_PATH REQUIRED)
  find_library(PHX_CUTENSORNET_LIBRARY NAMES cutensornet
    PATHS "${PHX_CUDA_TOOLKIT}/lib" NO_DEFAULT_PATH REQUIRED)
  add_library(CUDA::cutensornet SHARED IMPORTED GLOBAL)
  set_target_properties(CUDA::cutensornet PROPERTIES
    IMPORTED_LOCATION "${PHX_CUTENSORNET_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${PHX_CUTENSORNET_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES CUDA::cutensor)
endif()
execute_process(COMMAND "${Python_EXECUTABLE}" -c "import jax.ffi; print(jax.ffi.include_dir())"
  OUTPUT_VARIABLE _phx_ffi_include OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
add_library(phx_cuda_ffi INTERFACE)
target_include_directories(phx_cuda_ffi INTERFACE "${_phx_ffi_include}")
add_library(phx_cuda_common OBJECT src/common/capi.cpp)
target_include_directories(phx_cuda_common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src/common")
target_link_libraries(phx_cuda_common PUBLIC phx_cuda_ffi CUDA::cudart)
set_target_properties(phx_cuda_common PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
file(GLOB_RECURSE _phx_sources CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu" "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh" "${CMAKE_CURRENT_SOURCE_DIR}/*.cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/*CMakeLists.txt" "${CMAKE_CURRENT_SOURCE_DIR}/*.patch" "${CMAKE_CURRENT_SOURCE_DIR}/*.map")
list(SORT _phx_sources)
set(_phx_hash_input "")
set(_phx_patch_json "{}")
foreach(path IN LISTS _phx_sources)
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${path}" hash)
  string(APPEND _phx_hash_input "${path}:${hash}\n")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${path}")
  if(path MATCHES "\\.patch$")
    string(JSON _phx_patch_json SET "${_phx_patch_json}" "${path}" "\"${hash}\"")
  endif()
endforeach()
string(SHA256 PHX_CUDA_BUILD_HASH "${_phx_hash_input}")
file(READ "${PHX_CUDA_TOOLKIT}/discovery.json" _phx_discovery)
string(JSON _phx_wheels GET "${_phx_discovery}" wheels)
set(_phx_upstream "${PHX_CUDA_TOOLCHAIN_PINS}")
if("amgx" IN_LIST PHX_CUDA_COMPONENTS)
  string(JSON _phx_upstream SET "${_phx_upstream}" amgx
    "{\"commit\":\"91a8413\",\"url\":\"https://codeload.github.com/NVIDIA/AMGX/tar.gz/91a8413\",\"sha256\":\"a5a9034b7dbfd7ae486436074609faafad0730c3c9acb07e90caecff6f216c0d\"}")
endif()
if("quda" IN_LIST PHX_CUDA_COMPONENTS)
  string(JSON _phx_upstream SET "${_phx_upstream}" quda
    "{\"commit\":\"ba501e4\",\"url\":\"https://codeload.github.com/lattice/quda/tar.gz/ba501e4\",\"sha256\":\"3dabbec974f83a6daf8a2d55f38d45ff9561e636ddae4c1ad5c7c19424c70f0c\"}")
  string(JSON _phx_upstream SET "${_phx_upstream}" eigen
    "{\"commit\":\"e67c494cba7180066e73b9f6234d0b2129f1cdf5\",\"url\":\"https://gitlab.com/libeigen/eigen/-/archive/e67c494cba7180066e73b9f6234d0b2129f1cdf5.tar.bz2\",\"sha256\":\"98d244932291506b75c4ae7459af29b1112ea3d2f04660686a925d9ef6634583\"}")
endif()
file(WRITE "${CMAKE_BINARY_DIR}/phx_cuda_build.json"
  "{\n\"build_hash\":\"${PHX_CUDA_BUILD_HASH}\",\n\"toolkit_wheels\":${_phx_wheels},\n\"patch_hashes\":${_phx_patch_json},\n\"upstream_pins\":${_phx_upstream}\n}\n")
include(PhxCudaComponent)
if("collective" IN_LIST PHX_CUDA_COMPONENTS)
  add_subdirectory("src/collective")
endif()
foreach(component IN LISTS PHX_CUDA_COMPONENTS)
  if(NOT component STREQUAL "collective")
    add_subdirectory("src/${component}")
  endif()
endforeach()
# License texts live at the repository root, not under a companion's project.
if(DEFINED SKBUILD_METADATA_DIR)
  get_filename_component(_phx_repo "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
  install(FILES "${_phx_repo}/LICENSE" "${_phx_repo}/NOTICE" DESTINATION "${SKBUILD_METADATA_DIR}/licenses")
  file(GLOB _phx_licenses "${_phx_repo}/LICENSES/SPINEAX-MIT.txt" "${_phx_repo}/LICENSES/QUDA-MIT.txt"
    "${_phx_repo}/LICENSES/AMGX-*.txt" "${_phx_repo}/LICENSES/EIGEN-MPL-2.0.txt")
  install(FILES ${_phx_licenses} DESTINATION "${SKBUILD_METADATA_DIR}/licenses/LICENSES")
endif()
