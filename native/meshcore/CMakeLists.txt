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
find_package(Threads REQUIRED)

set(PHX_MC_SOURCES
  src/arrangement.cpp
  src/cdt2d.cpp
  src/clip2d.cpp
  src/clip3d.cpp
  src/core_capi.cpp
  src/tet_mesh_capi.cpp
  src/delaunay2d.cpp
  src/delaunay3d.cpp
  src/verify_volume.cpp
  src/cell_map_validity.cpp
  src/polyhedral_certificates.cpp
  src/tet_entities.cpp
  src/box_tree.cpp
  src/deterministic_float.cpp
  src/deterministic_float_capi.cpp
  src/metric_float.cpp
  src/metric_float_capi.cpp
  src/deterministic_solve.cpp
  src/metric_goal.cpp
  src/metric_goal_capi.cpp
  src/metric_carrier.cpp
  src/metric_collapse.cpp
  src/metric_source_capi.cpp
  src/bvh_query.cpp
  src/bvh_query_capi.cpp
  src/exact_kernels.cpp
  src/exact_polynomial_kernels.cpp
  src/exact_polynomial_integral.cpp
  src/exact_basis_integral.cpp
  src/exact_polynomial_verdict.cpp
  src/feature_protection.cpp
  src/graph_partition.cpp
  src/hex_topology.cpp
  src/overset.cpp
  src/overset_capi.cpp
  src/prism_layers.cpp
  src/prism_layers_capi.cpp
  src/implicit_polygon.cpp
  src/implicit_face_proposals.cpp
  src/quad_topology.cpp
  src/interval_program.cpp
  src/interval_simplex_restrict.cpp
  src/interval_basis_bounds.cpp
  src/intersections.cpp
  src/level_set.cpp
  src/local_boundary_edit.cpp
  src/mixed_refinement.cpp
  src/parallel.cpp
  src/periodic.cpp
  src/plc3d.cpp
  src/conforming_voronoi.cpp
  src/power_diagram.cpp
  src/predicates.cpp
  src/quadratic_bounds.cpp
  src/restricted_delaunay.cpp
  src/surface_adapt.cpp
  src/mesh_source.cpp
  src/surface_certify.cpp
  src/surface_envelope.cpp
  src/surface_mesh.cpp
  src/tet_mesh.cpp
  src/tet_mesh_collapse.cpp
  src/tet_mesh_export.cpp
  src/tet_mesh_improvement.cpp
  src/tet_mesh_insertion.cpp
  src/tet_mesh_probe.cpp
  src/tet_mesh_refinement.cpp
  src/tet_mesh_relocation.cpp
  src/tet_mesh_relocation_capi.cpp
  src/tet_mesh_source.cpp
  src/tet_mesh_state.cpp
  src/refine3d.cpp
  src/improve3d.cpp
  src/zero_set_admission.cpp
  src/zero_set_tube.cpp
  src/elementary.cpp
  src/planar_partition.cpp
  src/angle.cpp
  src/angle_capi.cpp
  src/metric_relocation.cpp
  src/cad_geometry.cpp
  src/cad_refine2d.cpp
  src/cad_curves.cpp
  src/cad_curves_capi.cpp
  src/verify_chart_partition.cpp
  src/verify_surface_embedding.cpp
  src/verify_cad_fidelity.cpp
  src/cad_faces.cpp
  src/cad_mesh.cpp
  src/cad_capi.cpp
  src/cad_curving.cpp
)
set(PHX_MC_HEADERS
  include/phydrax_meshcore.h
  src/boundary_policy.hpp
  src/capi_guard.hpp
  src/cavity.hpp
  src/clip3d.hpp
  src/clip_common.hpp
  src/overset.hpp
  src/prism_layers.hpp
  src/dyadic.hpp
  src/exact_line.hpp
  src/exact_scalar.hpp
  src/expansion.hpp
  src/wide_integer.hpp
  src/exact_tetrahedra_arrangement.inc
  src/facet_coverage.hpp
  src/reference_tetrahedron_clip.inc
  src/filtered.hpp
  src/deterministic_float.hpp
  src/metric_float.hpp
  src/metric_goal.hpp
  src/metric_carrier.hpp
  src/metric_collapse.hpp
  src/floating_point_mode.hpp
  src/interval_arithmetic.hpp
  src/geometry3.hpp
  src/intersections.hpp
  src/box_tree.hpp
  src/key_map.hpp
  src/mesh.hpp
  src/parallel.hpp
  src/plc3d.hpp
  src/predicates.hpp
  src/restricted_delaunay.hpp
  src/rounding.hpp
  src/source_construction.hpp
  src/tet_mesh.hpp
  src/tet_mesh_handle.hpp
  src/spatial_sort.hpp
  src/refine3d.hpp
  src/improve3d.hpp
  src/local_incidence.hpp
  src/topology_keys.hpp
  src/triangulation2d.hpp
  src/triangulation3d.hpp
  src/zero_set_admission.hpp
  src/zero_set_tube.hpp
  src/surface_adapt.hpp
  src/surface_certify.hpp
  src/mesh_source.hpp
  src/elementary.hpp
  src/directed_norm.hpp
  src/directed_interval.hpp
  src/angle.hpp
  src/metric_relocation.hpp
  src/cad_geometry.hpp
  src/cad_geometry_programs.inc
  src/cad_jet.hpp
  src/cdt2d.hpp
  src/chart_surface.hpp
  src/cad_refine2d.hpp
  src/cad_curves.hpp
  src/cad_curve_discretization.hpp
  src/refine_queue.hpp
  src/verify_chart_partition.hpp
  src/verify_surface_embedding.hpp
  src/verify_cad_fidelity.hpp
  src/cad_faces.hpp
  src/cad_mesh.hpp
  src/cad_curving.hpp
  src/quadratic_validity_screen.hpp
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
if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # cl.exe has no flag that forbids contraction; from 19.30 on, /fp:precise
  # contracts only when /fp:contract is also given.
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.30)
    message(FATAL_ERROR "phydrax-meshcore requires MSVC 19.30 or newer: older /fp:precise may contract a * b + c.")
  endif()
  set(PHX_MC_FP_FLAGS /fp:precise)
elseif(MSVC)
  # clang-cl: /fp:precise permits contraction, so forbid it explicitly.
  set(PHX_MC_FP_FLAGS /fp:precise /clang:-ffp-contract=off)
else()
  set(PHX_MC_FP_FLAGS -ffp-contract=off -fno-fast-math -fno-finite-math-only)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i[3-6]86|x86)$")
    list(APPEND PHX_MC_FP_FLAGS -msse2 -mfpmath=sse)
  endif()
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  list(APPEND PHX_MC_FP_FLAGS -fopenmp-simd)
endif()

add_library(phydrax_meshcore_objects OBJECT ${PHX_MC_SOURCES})
set_target_properties(phydrax_meshcore_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(phydrax_meshcore_objects
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(phydrax_meshcore_objects PUBLIC Threads::Threads)
target_compile_definitions(phydrax_meshcore_objects PUBLIC
  PHX_MC_BUILDING
  PHX_MC_VERSION="${PROJECT_VERSION}"
  PHX_MC_BUILD_HASH="${PHX_MC_BUILD_HASH}")
target_compile_options(phydrax_meshcore_objects PUBLIC ${PHX_MC_FP_FLAGS}
  PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>)
# Every branch above forbids contraction; sources that depend on it check this.
target_compile_definitions(phydrax_meshcore_objects PUBLIC PHX_MC_FP_CONTRACT_OFF=1)
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
  phx_mc_add_test(test_verify_volume)
  phx_mc_add_test(test_cell_map_validity)
  phx_mc_add_test(test_polyhedral_certificates)
  phx_mc_add_test(test_exact_kernels)
  phx_mc_add_test(test_bvh_query)
  phx_mc_add_test(test_overset)
  phx_mc_add_test(test_prism_layers)
  phx_mc_add_test(test_conforming_voronoi)
  phx_mc_add_test(test_exact_polynomial_kernels)
  phx_mc_add_test(test_exact_polynomial_integral)
  phx_mc_add_test(test_exact_basis_integral)
  phx_mc_add_test(test_clip)
  phx_mc_add_test(test_capi_guard)
  phx_mc_add_test(test_cavity)
  phx_mc_add_test(test_facet_coverage)
  phx_mc_add_test(test_graph_partition)
  phx_mc_add_test(test_intersections)
  phx_mc_add_test(test_deterministic_float)
  phx_mc_add_test(test_metric_float)
  phx_mc_add_test(test_metric_goal)
  phx_mc_add_test(test_deterministic_solve)
  phx_mc_add_test(test_metric_carrier)
  phx_mc_add_test(test_metric_collapse)
  phx_mc_add_test(test_rounding)
  phx_mc_add_test(test_directed_interval)
  phx_mc_add_test(test_geometry_length)
  phx_mc_add_test(test_angle)
  phx_mc_add_test(test_metric_relocation)
  phx_mc_add_test(test_implicit_face_proposals)
  phx_mc_add_test(test_interval_program)
  phx_mc_add_test(test_interval_simplex_restrict)
  phx_mc_add_test(test_parallel)
  phx_mc_add_test(test_plc3d)
  phx_mc_add_test(test_quadratic_bounds)
  phx_mc_add_test(test_quadratic_validity)
  phx_mc_add_test(test_tetrahedra_arrangement)
  phx_mc_add_test(test_level_set)
  phx_mc_add_test(test_mixed_refinement)
  phx_mc_add_test(test_periodic)
  phx_mc_add_test(test_power_diagram)
  phx_mc_add_test(test_restricted_delaunay)
  phx_mc_add_test(test_box_tree)
  phx_mc_add_test(test_tet_mesh)
  phx_mc_add_test(test_tet_mesh_relocation)
  phx_mc_add_test(test_local_boundary_edits)
  phx_mc_add_test(test_improve3d)
  phx_mc_add_test(test_hex_topology)
  phx_mc_add_test(test_surface_envelope)
  phx_mc_add_test(test_surface_mesh)
  phx_mc_add_test(test_surface_certify)
  phx_mc_add_test(test_mesh_source)
  phx_mc_add_test(test_zero_set_admission)
  phx_mc_add_test(test_elementary)
  phx_mc_add_test(test_planar_partition)
  phx_mc_add_test(test_wide_integer)
  phx_mc_add_test(test_cad_geometry)
  phx_mc_add_test(test_cad_refine2d)
  phx_mc_add_test(test_cad_curves)
  phx_mc_add_test(test_verify_chart_partition)
  phx_mc_add_test(test_verify_surface_embedding)
  phx_mc_add_test(test_verify_cad_fidelity)
  phx_mc_add_test(test_cad_mesh)
  phx_mc_add_test(test_cad_curving)
  phx_mc_add_test(test_cad_volume)
endif()
