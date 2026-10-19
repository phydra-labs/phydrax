//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

void mesh_null() {
  PHX_CHECK(phx_mc_mesh_copy_points(nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_mesh_copy_cells(nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_mesh_copy_vertex_map(nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_mesh_copy_cell_constraints(nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_mesh_copy_cell_regions(nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
}

void power_null() {
  PHX_CHECK(phx_mc_power_cells_sizes(nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_power_cells_counters(nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_power_cells_failure(nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_power_cells_export(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                      nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_power_cells_export_vertex_carriers(nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_power_cells_export_vertex_sites(nullptr, nullptr, nullptr) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_power_cells_phase_seconds(nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
}

void arrangement_null() {
  PHX_CHECK(phx_mc_arrangement_sizes(nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_arrangement_counters(nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_arrangement_export(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                      nullptr, nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_arrangement_export_classification(nullptr, nullptr, nullptr) ==
            PHX_MC_INVALID_ARGUMENT);
}

void periodic_null() {
  PHX_CHECK(phx_mc_periodic_copy_cells(nullptr, nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
}

void mesh_outputs() {
  constexpr double points[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  phx_mc_mesh* mesh = nullptr;
  PHX_CHECK(phx_mc_delaunay_3d(4, points, 64, &mesh) == PHX_MC_OK);
  if (mesh == nullptr) return;
  PHX_CHECK(phx_mc_mesh_copy_points(mesh, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_mesh_copy_cells(mesh, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_mesh_copy_vertex_map(mesh, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_mesh_copy_cell_constraints(mesh, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_mesh_copy_cell_regions(mesh, nullptr) == PHX_MC_INVALID_ARGUMENT);
  std::array<double, 12> copied{};
  PHX_CHECK(phx_mc_mesh_copy_points(mesh, copied.data()) == PHX_MC_OK);
  for (std::size_t index = 0; index < copied.size(); ++index)
    PHX_CHECK(copied[index] == points[index]);
  phx_mc_mesh_free(mesh);
}

void creation_coordinate(double coordinate, int32_t expected) {
  double points[] = {coordinate, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  constexpr int32_t cells[] = {0, 1, 2, 3};
  constexpr int32_t regions[] = {0};
  phx_mc_tet_mesh* mesh = nullptr;
  phx_mc_tet_mesh_create_request request{};
  request.point_count = 4;
  request.points = points;
  request.tet_count = 1;
  request.tets = cells;
  request.tet_regions = regions;
  request.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_FIXED;
  request.max_vertices = 4;
  request.max_tetrahedra = 1;
  request.max_scratch_bytes = 0;
  request.mesh = &mesh;
  void* execution = nullptr;
  PHX_CHECK(phx_mc_execution_begin(1000, 1000, 1000, 0, INFINITY, nullptr, &execution) ==
            PHX_MC_OK);
  PHX_CHECK(phx_mc_tet_mesh_create(&request) == expected);
  PHX_CHECK(mesh == nullptr);
  uint64_t work[PHX_MC_EXECUTION_COUNTERS]{};
  uint64_t memory[PHX_MC_EXECUTION_MEMORY_VALUES]{};
  double seconds = 0;
  PHX_CHECK(phx_mc_execution_end(execution, work, memory, &seconds) == PHX_MC_OK);
  PHX_CHECK(memory[PHX_MC_EXECUTION_MEMORY_ALLOCATIONS] == 0);
  PHX_CHECK(memory[PHX_MC_EXECUTION_MEMORY_REFUSALS] == 0);
  PHX_CHECK(memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  const std::string_view scenario = argv[1];
  if (scenario == "mesh_null")
    mesh_null();
  else if (scenario == "power_null")
    power_null();
  else if (scenario == "arrangement_null")
    arrangement_null();
  else if (scenario == "periodic_null")
    periodic_null();
  else if (scenario == "mesh_outputs")
    mesh_outputs();
  else if (scenario == "creation_nonfinite")
    creation_coordinate(INFINITY, PHX_MC_NONFINITE_INPUT);
  else if (scenario == "creation_nan")
    creation_coordinate(NAN, PHX_MC_NONFINITE_INPUT);
  else if (scenario == "creation_range")
    creation_coordinate(0x1p121, PHX_MC_RANGE_ERROR);
  else
    return 1;
  return phx::mc::test::finish("test_getters");
}
