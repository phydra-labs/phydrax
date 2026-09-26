cmake_minimum_required(VERSION 3.20)

# The distribution version in pyproject.toml is the only version source; the
# Phydrax loader requires it to equal the installed phydrax release.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" PHX_MC_VERSION_LINE
  REGEX "^version = \"[0-9]+\\.[0-9]+\\.[0-9]+\"$" LIMIT_COUNT 1)
string(REGEX REPLACE "^version = \"([0-9.]+)\"$" "\\1" PHX_MC_PROJECT_VERSION
  "${PHX_MC_VERSION_LINE}")
if(NOT PHX_MC_PROJECT_VERSION)
  message(FATAL_ERROR "native/meshcore/pyproject.toml must declare version = \"X.Y.Z\".")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS pyproject.toml)

project(phydrax_meshcore VERSION ${PHX_MC_PROJECT_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(PHX_MC_TOP_LEVEL ON)
else()
  set(PHX_MC_TOP_LEVEL OFF)
endif()
option(PHX_MC_BUILD_TESTS "Build the meshcore C++ unit tests" ${PHX_MC_TOP_LEVEL})
option(PHX_MC_SANITIZE
  "Instrument the library and its tests with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

set(PHX_MC_SOURCES
  src/cdt2d.cpp
  src/clip2d.cpp
  src/clip3d.cpp
  src/core_capi.cpp
  src/delaunay2d.cpp
  src/delaunay3d.cpp
  src/predicates.cpp
)
set(PHX_MC_HEADERS
  include/phydrax_meshcore.h
  src/capi_guard.hpp
  src/clip_common.hpp
  src/expansion.hpp
  src/filtered.hpp
  src/mesh.hpp
  src/predicates.hpp
  src/spatial_sort.hpp
  src/triangulation2d.hpp
)

# Content identity of the library sources; CMake re-configures when any listed
# file changes, so the recorded hash always matches the compiled sources.
set(PHX_MC_HASH_INPUT "")
foreach(path IN LISTS PHX_MC_SOURCES PHX_MC_HEADERS)
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${path}" file_hash)
  string(APPEND PHX_MC_HASH_INPUT "${path}:${file_hash}\n")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${path}")
endforeach()
string(SHA256 PHX_MC_BUILD_HASH "${PHX_MC_HASH_INPUT}")

# Exact predicates require IEEE-754 binary64 semantics: no contraction of
# a * b + c into fused multiply-adds and no value-changing optimizations.
if(MSVC)
  set(PHX_MC_FP_FLAGS /fp:precise /fp:contract-)
else()
  set(PHX_MC_FP_FLAGS -ffp-contract=off -fno-fast-math -fno-finite-math-only)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i[3-6]86|x86)$")
    list(APPEND PHX_MC_FP_FLAGS -msse2 -mfpmath=sse)
  endif()
endif()

add_library(phydrax_meshcore_objects OBJECT ${PHX_MC_SOURCES})
set_target_properties(phydrax_meshcore_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(phydrax_meshcore_objects
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(phydrax_meshcore_objects PUBLIC
  PHX_MC_BUILDING
  PHX_MC_VERSION="${PROJECT_VERSION}"
  PHX_MC_BUILD_HASH="${PHX_MC_BUILD_HASH}")
target_compile_options(phydrax_meshcore_objects PUBLIC ${PHX_MC_FP_FLAGS}
  PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>)
if(PHX_MC_SANITIZE)
  if(MSVC)
    message(FATAL_ERROR "PHX_MC_SANITIZE requires a Clang or GCC toolchain.")
  endif()
  # Any report aborts the test process, so CTest fails on the first finding;
  # PHX_MC_SANITIZE disables the time bounds of the performance checks.
  set(PHX_MC_SANITIZER_FLAGS -fsanitize=address,undefined -fno-sanitize-recover=all
    -fno-omit-frame-pointer)
  target_compile_definitions(phydrax_meshcore_objects PUBLIC PHX_MC_SANITIZE)
  target_compile_options(phydrax_meshcore_objects PUBLIC ${PHX_MC_SANITIZER_FLAGS})
  target_link_options(phydrax_meshcore_objects PUBLIC ${PHX_MC_SANITIZER_FLAGS})
endif()

add_library(phydrax_meshcore SHARED)
target_link_libraries(phydrax_meshcore PRIVATE phydrax_meshcore_objects)

install(TARGETS phydrax_meshcore
  LIBRARY DESTINATION phydrax_meshcore
  RUNTIME DESTINATION phydrax_meshcore
  ARCHIVE DESTINATION phydrax_meshcore)

if(PHX_MC_BUILD_TESTS)
  enable_testing()
  function(phx_mc_add_test name)
    add_executable(${name} tests/${name}.cpp)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(${name} PRIVATE phydrax_meshcore_objects)
    add_test(NAME ${name} COMMAND ${name})
  endfunction()
  phx_mc_add_test(test_predicates)
  phx_mc_add_test(test_cdt2d)
  phx_mc_add_test(test_delaunay2d)
  phx_mc_add_test(test_delaunay3d)
  phx_mc_add_test(test_clip)
  phx_mc_add_test(test_capi_guard)
endif()
