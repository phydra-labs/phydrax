cmake_minimum_required(VERSION 3.16)

project(phydrax_vorocrust_bridge LANGUAGES CXX)

include(GNUInstallDirs)

set(VOROCRUST_SOURCE_DIR "" CACHE PATH "VoroCrust source checkout")
set(VOROCRUST_BUILD_DIR "" CACHE PATH "Configured VoroCrust build directory")

set(_phydrax_vorocrust_source "${VOROCRUST_SOURCE_DIR}/src/Meshing")
set(_phydrax_vorocrust_binary "${VOROCRUST_BUILD_DIR}/src/Meshing")
set(_phydrax_vorocrust_cache "${VOROCRUST_BUILD_DIR}/CMakeCache.txt")
foreach(
  _required
  IN ITEMS
    "${_phydrax_vorocrust_source}/MeshingVoronoiMesher.h"
    "${_phydrax_vorocrust_binary}/Config.h"
    "${_phydrax_vorocrust_cache}"
)
  if(NOT EXISTS "${_required}")
    message(
      FATAL_ERROR
      "VOROCRUST_SOURCE_DIR and VOROCRUST_BUILD_DIR must identify one configured VoroCrust checkout/build; missing ${_required}."
    )
  endif()
endforeach()

get_filename_component(
  _phydrax_vorocrust_source_root
  "${VOROCRUST_SOURCE_DIR}"
  REALPATH
)
get_filename_component(
  _phydrax_vorocrust_build_root
  "${VOROCRUST_BUILD_DIR}"
  REALPATH
)
file(
  STRINGS "${_phydrax_vorocrust_cache}"
  _phydrax_vorocrust_cached_source
  REGEX "^CMAKE_HOME_DIRECTORY:INTERNAL="
)
if(NOT _phydrax_vorocrust_cached_source)
  message(FATAL_ERROR "VoroCrust CMakeCache.txt does not identify its source checkout.")
endif()
string(
  REGEX REPLACE
  "^CMAKE_HOME_DIRECTORY:INTERNAL="
  ""
  _phydrax_vorocrust_cached_source
  "${_phydrax_vorocrust_cached_source}"
)
get_filename_component(
  _phydrax_vorocrust_cached_source
  "${_phydrax_vorocrust_cached_source}"
  REALPATH
)
if(NOT "${_phydrax_vorocrust_cached_source}" STREQUAL "${_phydrax_vorocrust_source_root}")
  message(
    FATAL_ERROR
    "VOROCRUST_SOURCE_DIR and VOROCRUST_BUILD_DIR do not belong to the same configured checkout."
  )
endif()

set(
  PHYDRAX_VOROCRUST_REVISION
  ""
  CACHE STRING
  "Exact VoroCrust release or Git revision reported by the bridge"
)
if(
  NOT "${PHYDRAX_VOROCRUST_REVISION}"
      MATCHES "^[A-Za-z0-9][-A-Za-z0-9._+:]*$"
)
  message(
    FATAL_ERROR
    "Set PHYDRAX_VOROCRUST_REVISION to the exact non-whitespace VoroCrust release or Git revision."
  )
endif()

set(_phydrax_vorocrust_unsupported)
foreach(
  _option
  IN ITEMS
    VOROCRUST_ENABLE_EXODUS
    VOROCRUST_ENABLE_MPI
    VOROCRUST_ENABLE_TUCKERMPI
    VOROCRUST_TPL_BUILD_HDF5
    VOROCRUST_TPL_BUILD_NETCDF
    VOROCRUST_TPL_BUILD_OPENBLAS
    VOROCRUST_TPL_BUILD_SEACAS
    VOROCRUST_TPL_BUILD_ZLIB
    VOROCRUST_TPL_ENABLE_KOKKOS
    VOROCRUST_TPL_ENABLE_KOKKOSKERNELS
    VOROCRUST_TPL_USE_LAPACK
)
  file(
    STRINGS "${_phydrax_vorocrust_cache}"
    _enabled
    REGEX "^${_option}:BOOL=(ON|TRUE|YES|1)$"
  )
  if(_enabled)
    list(APPEND _phydrax_vorocrust_unsupported "${_option}")
  endif()
endforeach()
if(_phydrax_vorocrust_unsupported)
  list(JOIN _phydrax_vorocrust_unsupported ", " _unsupported_text)
  message(
    FATAL_ERROR
    "The standalone extraction bridge supports the serial CPU VoroCrust library with optional OpenMP only. Reconfigure VoroCrust without: ${_unsupported_text}."
  )
endif()

find_library(
  VOROCRUST_MESH_LIBRARY
  NAMES libVCMesh VCMesh
  HINTS "${VOROCRUST_BUILD_DIR}"
  PATH_SUFFIXES
    src/Meshing
    src/Meshing/Debug
    src/Meshing/Release
    lib
    lib/Debug
    lib/Release
  NO_DEFAULT_PATH
  DOC "VoroCrust libVCMesh library"
)
if(NOT VOROCRUST_MESH_LIBRARY)
  message(
    FATAL_ERROR
    "Could not find the built VoroCrust libVCMesh library below VOROCRUST_BUILD_DIR."
  )
endif()

get_filename_component(
  _phydrax_vorocrust_library
  "${VOROCRUST_MESH_LIBRARY}"
  REALPATH
)
string(
  FIND
  "${_phydrax_vorocrust_library}"
  "${_phydrax_vorocrust_build_root}/"
  _phydrax_vorocrust_library_prefix
)
if(NOT _phydrax_vorocrust_library_prefix EQUAL 0)
  message(FATAL_ERROR "VoroCrust library must reside below VOROCRUST_BUILD_DIR.")
endif()
file(
  SHA256
  "${_phydrax_vorocrust_source}/MeshingVoronoiMesher.h"
  PHYDRAX_VOROCRUST_SOURCE_SHA256
)
file(
  SHA256
  "${_phydrax_vorocrust_binary}/Config.h"
  PHYDRAX_VOROCRUST_CONFIG_SHA256
)
file(
  SHA256
  "${_phydrax_vorocrust_library}"
  PHYDRAX_VOROCRUST_LIBRARY_SHA256
)

file(
  STRINGS "${_phydrax_vorocrust_cache}"
  _phydrax_vorocrust_openmp
  REGEX "^VOROCRUST_ENABLE_OPENMP:BOOL=(ON|TRUE|YES|1)$"
)
if(_phydrax_vorocrust_openmp)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

add_executable(phydrax-vorocrust main.cpp)
target_compile_features(phydrax-vorocrust PRIVATE cxx_std_14)
set_target_properties(
  phydrax-vorocrust
  PROPERTIES CXX_EXTENSIONS OFF INSTALL_RPATH_USE_LINK_PATH ON
)
target_compile_definitions(
  phydrax-vorocrust
  PRIVATE
    "PHYDRAX_VOROCRUST_REVISION=\"${PHYDRAX_VOROCRUST_REVISION}\""
    "PHYDRAX_VOROCRUST_SOURCE_SHA256=\"${PHYDRAX_VOROCRUST_SOURCE_SHA256}\""
    "PHYDRAX_VOROCRUST_CONFIG_SHA256=\"${PHYDRAX_VOROCRUST_CONFIG_SHA256}\""
    "PHYDRAX_VOROCRUST_LIBRARY_SHA256=\"${PHYDRAX_VOROCRUST_LIBRARY_SHA256}\""
)
target_include_directories(
  phydrax-vorocrust
  PRIVATE "${_phydrax_vorocrust_source}" "${_phydrax_vorocrust_binary}"
)
target_link_libraries(
  phydrax-vorocrust
  PRIVATE "${VOROCRUST_MESH_LIBRARY}" "${CMAKE_DL_LIBS}"
)
if(_phydrax_vorocrust_openmp)
  target_link_libraries(phydrax-vorocrust PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS phydrax-vorocrust RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
