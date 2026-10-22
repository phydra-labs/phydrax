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
option(PHX_MC_THREAD_SANITIZE
  "Instrument the library and its tests with ThreadSanitizer" OFF)
if(PHX_MC_SANITIZE AND PHX_MC_THREAD_SANITIZE)
  message(FATAL_ERROR "AddressSanitizer and ThreadSanitizer require separate builds.")
endif()
option(PHX_MC_LTO "Enable interprocedural optimization without changing floating-point semantics" OFF)
if(PHX_MC_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT PHX_MC_IPO_SUPPORTED OUTPUT PHX_MC_IPO_ERROR LANGUAGES CXX)
  if(NOT PHX_MC_IPO_SUPPORTED)
    message(FATAL_ERROR "PHX_MC_LTO requires supported interprocedural optimization: ${PHX_MC_IPO_ERROR}")
  endif()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

set(PHX_MC_SOURCES
  src/arrangement.cpp
  src/bounded_memory.cpp
  src/cdt2d.cpp
  src/clip2d.cpp
  src/clip3d.cpp
  src/core_capi.cpp
  src/delaunay2d.cpp
  src/delaunay3d.cpp
  src/feature_protection.cpp
  src/hex_topology.cpp
  src/graph_partition.cpp
  src/intersections.cpp
  src/periodic.cpp
  src/parallel.cpp
  src/parallel_capi.cpp
  src/mixed_refinement.cpp
  src/plc3d.cpp
  src/quad_topology.cpp
  src/restricted_delaunay.cpp
  src/power_diagram.cpp
  src/predicates.cpp
  src/improve3d.cpp
  src/refine3d.cpp
  src/surface_mesh.cpp
  src/surface_envelope.cpp
  src/level_set.cpp
  src/tet_mesh.cpp
  src/tet_mesh_state.cpp
  src/tet_mesh_source.cpp
  src/tet_mesh_insertion.cpp
  src/tet_mesh_collapse.cpp
  src/tet_mesh_refinement.cpp
  src/tet_mesh_improvement.cpp
  src/tet_mesh_export.cpp
)
set(PHX_MC_HEADERS
  include/phydrax_meshcore.h
  src/bounded_memory.hpp
  src/capi_guard.hpp
  src/boundary_policy.hpp
  src/counter_records.hpp
  src/cavity.hpp
  src/dyadic.hpp
  src/clip_common.hpp
  src/clip3d.hpp
  src/exact_line.hpp
  src/expansion.hpp
  src/facet_coverage.hpp
  src/filtered.hpp
  src/intersections.hpp
  src/geometry3.hpp
  src/exact_scalar.hpp
  src/topology_keys.hpp
  src/rounding.hpp
  src/mesh.hpp
  src/periodic.hpp
  src/parallel.hpp
  src/plc3d.hpp
  src/predicates.hpp
  src/reference_tetrahedron_clip.inc
  src/source_construction.hpp
  src/spatial_sort.hpp
  src/triangulation2d.hpp
  src/restricted_delaunay.hpp
  src/triangulation3d.hpp
  src/improve3d.hpp
  src/refine3d.hpp
  src/tet_mesh.hpp
)
list(SORT PHX_MC_SOURCES)
list(SORT PHX_MC_HEADERS)
set(PHX_MC_HASH_PATHS ${PHX_MC_SOURCES} ${PHX_MC_HEADERS} CMakeLists.txt pyproject.toml)
list(SORT PHX_MC_HASH_PATHS)

# Content identity of the library sources; CMake re-configures when any listed
# file changes, so the recorded hash always matches the compiled sources.
set(PHX_MC_HASH_INPUT "")
foreach(path IN LISTS PHX_MC_HASH_PATHS)
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${path}" file_hash)
  string(APPEND PHX_MC_HASH_INPUT "${path}:${file_hash}\n")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${path}")
endforeach()
string(SHA256 PHX_MC_BUILD_HASH "${PHX_MC_HASH_INPUT}")

# Canonical C ABI contract: the public header itself, line ends normalized,
# so the exported declarations are the only specification bindings verify.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/phydrax_meshcore.h" PHX_MC_PUBLIC_HEADER)
string(REPLACE "\r\n" "\n" PHX_MC_PUBLIC_HEADER "${PHX_MC_PUBLIC_HEADER}")
string(SHA256 PHX_MC_ABI_CONTRACT "${PHX_MC_PUBLIC_HEADER}")

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
string(JOIN "," PHX_MC_FP_IDENTITY ${PHX_MC_FP_FLAGS})
set(PHX_MC_CONFIGURATION
  "compiler=${CMAKE_CXX_COMPILER_ID}:${CMAKE_CXX_COMPILER_VERSION}|system=${CMAKE_SYSTEM_NAME}:${CMAKE_SYSTEM_PROCESSOR}|cxx=${CMAKE_CXX_STANDARD}|sanitize=${PHX_MC_SANITIZE}|thread_sanitize=${PHX_MC_THREAD_SANITIZE}|lto=${PHX_MC_LTO}|fp=${PHX_MC_FP_IDENTITY}|base=${CMAKE_CXX_FLAGS}|release=${CMAKE_CXX_FLAGS_RELEASE}|debug=${CMAKE_CXX_FLAGS_DEBUG}|relwithdebinfo=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string(REPLACE "\\" "\\\\" PHX_MC_CONFIGURATION "${PHX_MC_CONFIGURATION}")
string(REPLACE "\"" "\\\"" PHX_MC_CONFIGURATION "${PHX_MC_CONFIGURATION}")


add_library(phydrax_meshcore_objects OBJECT ${PHX_MC_SOURCES})
set_target_properties(phydrax_meshcore_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)
target_link_libraries(phydrax_meshcore_objects PUBLIC Threads::Threads)
target_include_directories(phydrax_meshcore_objects
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(phydrax_meshcore_objects PUBLIC
  PHX_MC_BUILDING
  PHX_MC_VERSION="${PROJECT_VERSION}"
  PHX_MC_BUILD_HASH="${PHX_MC_BUILD_HASH}"
  PHX_MC_ABI_CONTRACT="${PHX_MC_ABI_CONTRACT}"
  PHX_MC_BUILD_CONFIGURATION="${PHX_MC_CONFIGURATION}|execution_clocks=request-preparation+absolute-workflow|config=$<CONFIG>")
target_compile_options(phydrax_meshcore_objects PUBLIC ${PHX_MC_FP_FLAGS}
  PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(phydrax_meshcore_objects PUBLIC -fopenmp-simd)
endif()
if(PHX_MC_SANITIZE)
  if(MSVC)
    message(FATAL_ERROR "PHX_MC_SANITIZE requires a Clang or GCC toolchain.")
  endif()
  # Any report aborts the test process, so CTest fails on the first finding.
  set(PHX_MC_SANITIZER_FLAGS -fsanitize=address,undefined -fno-sanitize-recover=all
    -fno-omit-frame-pointer)
  target_compile_options(phydrax_meshcore_objects PUBLIC ${PHX_MC_SANITIZER_FLAGS})
  target_link_options(phydrax_meshcore_objects PUBLIC ${PHX_MC_SANITIZER_FLAGS})
endif()
if(PHX_MC_THREAD_SANITIZE)
  if(MSVC)
    message(FATAL_ERROR "PHX_MC_THREAD_SANITIZE requires a Clang or GCC toolchain.")
  endif()
  set(PHX_MC_THREAD_SANITIZER_FLAGS -fsanitize=thread -fno-omit-frame-pointer)
  target_compile_options(phydrax_meshcore_objects PUBLIC ${PHX_MC_THREAD_SANITIZER_FLAGS})
  target_link_options(phydrax_meshcore_objects PUBLIC ${PHX_MC_THREAD_SANITIZER_FLAGS})
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
  phx_mc_add_test(test_bounded_memory)
  phx_mc_add_test(test_cdt2d)
  phx_mc_add_test(test_delaunay2d)
  phx_mc_add_test(test_delaunay3d)
  phx_mc_add_test(test_clip)
  phx_mc_add_test(test_capi_guard)
  phx_mc_add_test(test_graph_partition)
  phx_mc_add_test(test_intersections)
  phx_mc_add_test(test_cavity)
  phx_mc_add_test(test_surface_mesh)
  phx_mc_add_test(test_level_set)
  phx_mc_add_test(test_periodic)
  phx_mc_add_test(test_plc3d)
  phx_mc_add_test(test_refine3d)
  phx_mc_add_test(test_improve3d)
  phx_mc_add_test(test_curved_collapse)
  phx_mc_add_test(test_power_diagram)
  phx_mc_add_test(test_mixed_refinement)
  add_executable(test_getters tests/test_getters.cpp)
  target_include_directories(test_getters PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_link_libraries(test_getters PRIVATE phydrax_meshcore_objects)
  foreach(case IN ITEMS
      mesh_null power_null arrangement_null periodic_null mesh_outputs
      creation_nonfinite creation_nan creation_range)
    add_test(NAME test_getters_${case} COMMAND test_getters ${case})
  endforeach()
  add_executable(test_execution_batch tests/test_execution_batch.cpp)
  target_include_directories(test_execution_batch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_link_libraries(test_execution_batch PRIVATE phydrax_meshcore_objects Threads::Threads)
  foreach(case IN ITEMS
      work_admission query_admission cavity_admission byte_admission unused_release
      nested_isolation nested_refusal native_work native_queries native_cavity native_bytes
      settlement_retry nested_overestimate_retry live_ticket_lifetime exception_settlement
      concurrent_admission nested_scope_reservation timeout_admission checkpoint_and_invalid_arguments
      refusal_consumption_evidence live_child_refusal live_child_refusal_limited_by_ancestor)
    add_test(NAME test_execution_batch_${case} COMMAND test_execution_batch ${case})
  endforeach()
  add_executable(test_parallel_budget tests/test_parallel_budget.cpp)
  target_include_directories(test_parallel_budget PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_link_libraries(test_parallel_budget PRIVATE phydrax_meshcore_objects Threads::Threads)
  foreach(case IN ITEMS
      work_admission query_admission cavity_admission byte_admission unused_release
      work_bound_defect query_bound_defect cavity_bound_defect byte_bound_defect
      nested_parent_ticket worker_isolation arena_recycling arena_lifetime exception_release
      staging_allocation_release staging_exception_release unwinding_release construction_restore
      timeout_evidence sum_overflow implicit_parent)
    add_test(NAME test_parallel_budget_${case} COMMAND test_parallel_budget ${case})
  endforeach()
  phx_mc_add_test(test_parallel_kernels)
  phx_mc_add_test(test_parallel)
  add_executable(test_parallel_policy tests/test_parallel_policy.cpp)
  target_include_directories(test_parallel_policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_link_libraries(test_parallel_policy PRIVATE phydrax_meshcore_objects Threads::Threads)
  foreach(case IN ITEMS invalid_arguments nested_restore creating_thread expired_restore
      preparation_clock_independence request_clock_still_debited preparation_exhausts_request
      preparation_cannot_renew_expired_workflow policy_entered_after_scope
      nested_scope_preparation_clock)
    add_test(NAME test_parallel_policy_${case} COMMAND test_parallel_policy ${case})
  endforeach()
  add_executable(test_execution_target tests/test_execution_target.cpp)
  target_include_directories(test_execution_target PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_link_libraries(test_execution_target PRIVATE phydrax_meshcore_objects Threads::Threads)
  foreach(case IN ITEMS
      real_mesh_creation reserved_native_charges host_native_interleaving native_refusal_evidence
      reserved_bytes zero_bytes host_native_bytes reserved_host_array unused_release
      escaped_memory child_escaped_memory nested_restore exception_restore worker_child
      child_scope child_scope_refusal borrowed_scope_batch invalid_lifetimes
      imported_evidence mapped_counter_reads retained_host_storage retained_host_refusal
      atomic_refusal_equivalence atomic_refusal_child atomic_refusal_estimate_defect
      atomic_refusal_worker admitting_parent_batch admitting_parent_native)
    add_test(NAME test_execution_target_${case} COMMAND test_execution_target ${case})
  endforeach()
  if(PHX_MC_SANITIZE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
      AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
      AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    # GCC's optimized UBSan TLS-address checks branch on ADD's flags, but GNU
    # ld's IE-to-LE relaxation replaces ADD with LEA without preserving them.
    # objdump of test_bounded_memory.cpp.o and its executable shows that rewrite.
    # The pre-wave source fails identically at RelWithDebInfo plus sanitizers;
    # its Debug sanitizer build passes. Select LE at compilation so the compiler
    # owns the address calculation and checks. Shared-library flags stay unchanged.
    get_property(PHX_MC_NATIVE_TARGETS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    get_target_property(PHX_MC_OBJECT_TYPE phydrax_meshcore_objects TYPE)
    foreach(target IN LISTS PHX_MC_NATIVE_TARGETS)
      get_target_property(PHX_MC_TARGET_TYPE ${target} TYPE)
      if(PHX_MC_TARGET_TYPE STREQUAL "EXECUTABLE")
        get_target_property(PHX_MC_TEST_LINKS ${target} LINK_LIBRARIES)
        if(NOT PHX_MC_OBJECT_TYPE STREQUAL "OBJECT_LIBRARY"
            OR NOT "phydrax_meshcore_objects" IN_LIST PHX_MC_TEST_LINKS)
          message(FATAL_ERROR
            "${target} local-exec TLS requires statically linked meshcore objects.")
        endif()
        target_compile_options(${target} PRIVATE -ftls-model=local-exec)
      endif()
    endforeach()
  endif()
endif()
