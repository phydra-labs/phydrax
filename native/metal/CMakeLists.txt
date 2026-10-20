#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
# Native build of the phydrax-metal PJRT plugin. Driven only by
# tools/build_metal_wheels.py, which supplies every source tree from
# sources.lock.json (hash-verified and patched) and the locked LLVM/MLIR and MLX
# installs. The plugin links StableHLO and the Shardy sdy dialect at the XLA
# revision pinned by the qualified jaxlib, and MLX statically; only Apple system
# frameworks remain dynamic dependencies.

cmake_minimum_required(VERSION 3.25)
project(phydrax_metal LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

foreach(required
    PHYDRAX_METAL_JAX_MPS_SOURCE_DIR
    PHYDRAX_METAL_STABLEHLO_SOURCE_DIR
    PHYDRAX_METAL_SHARDY_SOURCE_DIR
    PHYDRAX_METAL_XLA_SOURCE_DIR
    PHYDRAX_METAL_VERSION
    PHYDRAX_METAL_SOURCE_ID)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "${required} must be provided by tools/build_metal_wheels.py")
  endif()
endforeach()
if(NOT PHYDRAX_METAL_SOURCE_ID MATCHES "^[0-9a-f]+$")
  message(FATAL_ERROR "PHYDRAX_METAL_SOURCE_ID must be a lowercase hex digest")
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
  message(FATAL_ERROR "phydrax-metal builds only for macOS arm64")
endif()

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
find_package(MLX REQUIRED CONFIG)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

# MLX's colocated-metallib lookup (jax-mps MLX patch 02) loads the library next
# to the binary that contains MLX. A plugin without it imports but cannot
# execute, so a missing metallib is a configuration failure.
find_file(PHYDRAX_METAL_METALLIB mlx.metallib
  PATHS "${MLX_DIR}/../../../lib" NO_DEFAULT_PATH REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# StableHLO as an embedded project: only the libraries the plugin links are
# built. Its directory scope keeps LLVM's compile options to its own targets.
set(STABLEHLO_BUILD_EMBEDDED ON CACHE BOOL "" FORCE)
set(STABLEHLO_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "" FORCE)
add_subdirectory("${PHYDRAX_METAL_STABLEHLO_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/stablehlo"
                 EXCLUDE_FROM_ALL)
set(PHYDRAX_METAL_STABLEHLO_INCLUDE_DIRS
    "${PHYDRAX_METAL_STABLEHLO_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}/stablehlo")

# The Shardy sdy dialect (Bazel-only upstream); generated headers land at
# sdy-gen/shardy/dialect/sdy/ir so upstream include paths resolve unchanged.
add_subdirectory(cmake/sdy "${CMAKE_BINARY_DIR}/sdy-gen/shardy/dialect/sdy/ir")

set(JAX_MPS "${PHYDRAX_METAL_JAX_MPS_SOURCE_DIR}")
set(PLUGIN_INCLUDE_DIRS
    "${JAX_MPS}/src"
    "${PHYDRAX_METAL_XLA_SOURCE_DIR}"
    ${PHYDRAX_METAL_STABLEHLO_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS})

# Inherits MLIR classes: compiled like MLIR itself (no RTTI, no exceptions).
add_library(phydrax_metal_mlir_interfaces STATIC
  "${JAX_MPS}/src/pjrt_plugin/passes/stablehlo_inliner_interface.cc")
set_target_properties(phydrax_metal_mlir_interfaces PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
target_compile_options(phydrax_metal_mlir_interfaces PRIVATE -fno-rtti -fno-exceptions)
target_include_directories(phydrax_metal_mlir_interfaces PRIVATE ${PLUGIN_INCLUDE_DIRS})
target_link_libraries(phydrax_metal_mlir_interfaces PRIVATE StablehloOps ChloOps MLIRTransformUtils)

add_library(pjrt_plugin_mps SHARED
  "${JAX_MPS}/src/pjrt_plugin/pjrt_api.cc"
  "${JAX_MPS}/src/pjrt_plugin/pjrt_client.cc"
  "${JAX_MPS}/src/pjrt_plugin/pjrt_device.cc"
  "${JAX_MPS}/src/pjrt_plugin/pjrt_memory.cc"
  "${JAX_MPS}/src/pjrt_plugin/pjrt_buffer.cc"
  "${JAX_MPS}/src/pjrt_plugin/pjrt_profiler.cc"
  "${JAX_MPS}/src/pjrt_plugin/pjrt_executable.cc"
  "${JAX_MPS}/src/pjrt_plugin/pjrt_event.cc"
  "${JAX_MPS}/src/pjrt_plugin/pjrt_topology.cc"
  "${JAX_MPS}/src/pjrt_plugin/stablehlo_parser.cc"
  "${JAX_MPS}/src/pjrt_plugin/type_utils.cc"
  "${JAX_MPS}/src/pjrt_plugin/mlx_client.cc"
  "${JAX_MPS}/src/pjrt_plugin/mlx_device.cc"
  "${JAX_MPS}/src/pjrt_plugin/mlx_buffer.cc"
  "${JAX_MPS}/src/pjrt_plugin/mlx_executable.cc"
  "${JAX_MPS}/src/pjrt_plugin/ops/arithmetic.cc"
  "${JAX_MPS}/src/pjrt_plugin/ops/shape.cc"
  "${JAX_MPS}/src/pjrt_plugin/ops/slice.cc"
  "${JAX_MPS}/src/pjrt_plugin/ops/gather_scatter.cc"
  "${JAX_MPS}/src/pjrt_plugin/ops/reduction.cc"
  "${JAX_MPS}/src/pjrt_plugin/ops/collectives.cc"
  "${JAX_MPS}/src/pjrt_plugin/ops/linalg.cc"
  "${JAX_MPS}/src/pjrt_plugin/ops/control_flow.cc"
  "${JAX_MPS}/src/pjrt_plugin/ops/device_regions.cc"
  "${JAX_MPS}/src/pjrt_plugin/ops/host_callback.cc"
  "${JAX_MPS}/src/pjrt_plugin/ops/sort_fft_complex.cc")
target_include_directories(pjrt_plugin_mps PRIVATE ${PLUGIN_INCLUDE_DIRS})
target_compile_definitions(pjrt_plugin_mps PRIVATE
  PHYDRAX_METAL_VERSION="${PHYDRAX_METAL_VERSION}"
  PHYDRAX_METAL_SOURCE_ID="${PHYDRAX_METAL_SOURCE_ID}")
target_link_libraries(pjrt_plugin_mps PRIVATE
  phydrax_metal_mlir_interfaces
  phydrax_metal_sdy
  StablehloSerialization
  StablehloOptimizationPasses
  StablehloPasses
  StablehloOps
  ChloOps
  VhloOps
  VhloTypes
  Version
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRTransforms
  MLIRPass
  MLIRParser
  MLIRBytecodeReader
  MLIRIR
  MLIRSupport
  LLVMSupport
  mlx
  "${METAL_FRAMEWORK}"
  "${MPS_FRAMEWORK}"
  "${FOUNDATION_FRAMEWORK}")
set_target_properties(pjrt_plugin_mps PROPERTIES
  OUTPUT_NAME "pjrt_plugin_mps"
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  INSTALL_NAME_DIR "@loader_path"
  BUILD_WITH_INSTALL_NAME_DIR ON)
# Only the PJRT entry point and the companion's host-callback registration are
# exported; statically linked LLVM/MLIR/StableHLO/MLX symbols stay private.
target_link_options(pjrt_plugin_mps PRIVATE
  "LINKER:-exported_symbol,_GetPjrtApi"
  "LINKER:-exported_symbol,_PhydraxMetal_SetHostCallbackTrampoline"
  "LINKER:-dead_strip")

install(TARGETS pjrt_plugin_mps LIBRARY DESTINATION lib)
install(FILES "${PHYDRAX_METAL_METALLIB}" DESTINATION lib)
