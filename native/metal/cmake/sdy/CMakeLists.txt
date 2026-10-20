#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
# CMake equivalent of Shardy's Bazel targets //shardy/dialect/sdy/ir:dialect and
# //shardy/common:logging at the locked revision. Configured with the binary
# directory <build>/sdy-gen/shardy/dialect/sdy/ir so generated *.inc files have
# the include paths the upstream sources expect.

set(SDY "${PHYDRAX_METAL_SHARDY_SOURCE_DIR}")
set(SDY_IR "${SDY}/shardy/dialect/sdy/ir")
set(SDY_GENERATED_ROOT "${CMAKE_BINARY_DIR}/sdy-gen")

include_directories("${SDY}" ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS}
                    ${PHYDRAX_METAL_STABLEHLO_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS "${SDY_IR}/op_interface.td")
mlir_tablegen(op_interface.h.inc -gen-op-interface-decls)
mlir_tablegen(op_interface.cc.inc -gen-op-interface-defs)
set(LLVM_TARGET_DEFINITIONS "${SDY_IR}/dialect.td")
mlir_tablegen(dialect.h.inc -gen-dialect-decls)
mlir_tablegen(dialect.cc.inc -gen-dialect-defs)
set(LLVM_TARGET_DEFINITIONS "${SDY_IR}/ops.td")
mlir_tablegen(ops.h.inc -gen-op-decls)
mlir_tablegen(ops.cc.inc -gen-op-defs)
set(LLVM_TARGET_DEFINITIONS "${SDY_IR}/attrs.td")
mlir_tablegen(attrs.h.inc -gen-attrdef-decls)
mlir_tablegen(attrs.cc.inc -gen-attrdef-defs)
set(LLVM_TARGET_DEFINITIONS "${SDY_IR}/enums.td")
mlir_tablegen(enums.h.inc -gen-enum-decls)
mlir_tablegen(enums.cc.inc -gen-enum-defs)
set(LLVM_TARGET_DEFINITIONS "${SDY_IR}/canonicalization.td")
mlir_tablegen(canonicalization.cc.inc -gen-rewriters)
set(LLVM_TARGET_DEFINITIONS "${SDY_IR}/bytecode.td")
mlir_tablegen(bytecode.cc.inc -gen-bytecode -bytecode-dialect=sdy)
add_public_tablegen_target(phydrax_metal_sdy_incgen)
add_dependencies(phydrax_metal_sdy_incgen StablehloOpsIncGen)

add_library(phydrax_metal_sdy STATIC
  "${SDY}/shardy/common/logging.cc"
  "${SDY_IR}/bytecode.cc"
  "${SDY_IR}/canonicalization.cc"
  "${SDY_IR}/compatibility.cc"
  "${SDY_IR}/dialect.cc"
  "${SDY_IR}/extensions/stablehlo_extensions.cc"
  "${SDY_IR}/parsers.cc"
  "${SDY_IR}/printers.cc"
  "${SDY_IR}/utils.cc"
  "${SDY_IR}/verifiers.cc")
add_dependencies(phydrax_metal_sdy phydrax_metal_sdy_incgen)
set_target_properties(phydrax_metal_sdy PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
target_compile_options(phydrax_metal_sdy PRIVATE -fno-rtti -fno-exceptions)
target_include_directories(phydrax_metal_sdy PUBLIC "${SDY}" "${SDY_GENERATED_ROOT}")
target_link_libraries(phydrax_metal_sdy PUBLIC
  StablehloAssemblyFormat
  StablehloOps
  StablehloTypeInference
  MLIRAnalysis
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRShapeDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTransformUtils
  LLVMSupport)
