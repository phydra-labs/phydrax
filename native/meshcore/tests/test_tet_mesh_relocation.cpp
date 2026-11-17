//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include <array>
#include <cstdint>
#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

// A tetrahedron split at one interior vertex: four cells around vertex 14.
constexpr std::array<double, 15> kPoints{
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.25, 0.25, 0.25};
constexpr std::array<int32_t, 16> kTetrahedra{
    4, 1, 2, 3, 0, 4, 2, 3, 0, 1, 4, 3, 0, 1, 2, 4};
constexpr std::array<int32_t, 12> kFaces{1, 3, 2, 0, 2, 3, 0, 3, 1, 0, 1, 2};
constexpr std::array<int32_t, 4> kFaceSources{0, 1, 2, 3};
constexpr std::array<uint64_t, 5> kVertexIds{10, 11, 12, 13, 14};
constexpr std::array<uint64_t, 4> kCellIds{100, 101, 102, 103};
constexpr std::array<uint64_t, 4> kFacetIds{200, 201, 202, 203};

phx_mc_tet_mesh* create_mesh() {
  phx_mc_tet_mesh* mesh = nullptr;
  phx_mc_tet_mesh_create_request create{};
  create.point_count = 5;
  create.points = kPoints.data();
  create.vertex_ids = kVertexIds.data();
  create.tetrahedron_count = 4;
  create.tetrahedra = kTetrahedra.data();
  create.cell_ids = kCellIds.data();
  create.face_count = 4;
  create.faces = kFaces.data();
  create.face_sources = kFaceSources.data();
  create.facet_ids = kFacetIds.data();
  create.boundary_policy = PHX_MC_TET_MESH_BOUNDARY_CONFORMING;
  create.result = &mesh;
  PHX_CHECK(phx_mc_tet_mesh_create(&create) == PHX_MC_OK);
  return mesh;
}

uint64_t generation_of(const phx_mc_tet_mesh* mesh) {
  uint64_t generation = 99;
  int32_t bound = -1;
  int64_t lengths[4]{};
  PHX_CHECK(phx_mc_tet_mesh_identity_sizes(mesh, &generation, &bound, &lengths[0], &lengths[1],
                                           &lengths[2], &lengths[3]) == PHX_MC_OK);
  return generation;
}

// An explicitly cleared edge parameter for one moved vertex.
constexpr int8_t kEdgeWitnessAbsent = 0;
constexpr std::array<uint64_t, 2> kNoEdgeEndpoints{0, 0};
constexpr double kNoEdgeParameter = 0.0;

// The general relocation request the batched probe states for one group.
phx_mc_tet_mesh_edit_request relocation_request(uint64_t generation, const uint64_t* vertex,
                                                const double* point, int64_t cell_count,
                                                const uint64_t* cells) {
  phx_mc_tet_mesh_edit_request request{};
  request.expected_generation = generation;
  request.kind = PHX_MC_TET_EDIT_RELOCATE;
  request.changed_vertex_count = 1;
  request.changed_vertex_ids = vertex;
  request.changed_vertex_points = point;
  request.changed_edge_witness_present = &kEdgeWitnessAbsent;
  request.changed_edge_witness_endpoints = kNoEdgeEndpoints.data();
  request.changed_edge_parameters = &kNoEdgeParameter;
  request.removed_cell_count = cell_count;
  request.removed_cell_ids = cells;
  return request;
}

void batched_probe_matches_the_single_request_probe() {
  phx_mc_tet_mesh* mesh = create_mesh();
  PHX_CHECK(mesh != nullptr);
  if (mesh == nullptr) return;
  constexpr uint64_t moved = 14;
  // Admissible, inverted, incomplete star, coincident with vertex 10.
  constexpr std::array<double, 12> points{
      0.3, 0.25, 0.25, 2.0, 2.0, 2.0, 0.3, 0.25, 0.25, 0.0, 0.0, 0.0};
  constexpr std::array<uint64_t, 4> vertices{moved, moved, moved, moved};
  constexpr std::array<int64_t, 5> vertex_offsets{0, 1, 2, 3, 4};
  constexpr std::array<uint64_t, 15> cells{
      100, 101, 102, 103, 100, 101, 102, 103, 100, 101, 102, 100, 101, 102, 103};
  constexpr std::array<int64_t, 5> cell_offsets{0, 4, 8, 11, 15};
  constexpr std::array<int32_t, 4> expected_decided{0, 1, 1, 1};
  constexpr std::array<int32_t, 4> expected_status{
      PHX_MC_TET_EDIT_REJECTED_TOPOLOGY, PHX_MC_TET_EDIT_INVALID_GEOMETRY,
      PHX_MC_TET_EDIT_REJECTED_TOPOLOGY, PHX_MC_TET_EDIT_INVALID_GEOMETRY};
  std::array<int32_t, 4> decided{-1, -1, -1, -1};
  std::array<int32_t, 4> statuses{-1, -1, -1, -1};
  PHX_CHECK(phx_mc_tet_mesh_probe_relocations(
                mesh, 0, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, 4,
                vertex_offsets.data(), vertices.data(), points.data(), cell_offsets.data(),
                cells.data(), decided.data(), statuses.data()) == PHX_MC_OK);
  PHX_CHECK(decided == expected_decided);
  PHX_CHECK(statuses == expected_status);
  PHX_CHECK(generation_of(mesh) == 0);

  // Each group agrees with the request probe of that relocation alone.
  for (int64_t group = 0; group < 4; ++group) {
    const phx_mc_tet_mesh_edit_request request = relocation_request(
        0, &moved, points.data() + 3 * group, cell_offsets[group + 1] - cell_offsets[group],
        cells.data() + cell_offsets[group]);
    int32_t single_decided = -1;
    int32_t single_status = -1;
    PHX_CHECK(phx_mc_tet_mesh_probe_edit(mesh, &request, &single_decided, &single_status) ==
              PHX_MC_OK);
    PHX_CHECK(single_decided == decided[group]);
    PHX_CHECK(single_decided == 0 || single_status == statuses[group]);
  }

  // The undecided group applies; the probe then refuses the old generation.
  const phx_mc_tet_mesh_edit_request admitted =
      relocation_request(0, &moved, points.data(), 4, cells.data());
  int32_t edit_status = -1;
  PHX_CHECK(phx_mc_tet_mesh_apply_edit(mesh, &admitted, &edit_status) == PHX_MC_OK);
  PHX_CHECK(edit_status == PHX_MC_TET_EDIT_APPLIED);
  PHX_CHECK(generation_of(mesh) == 1);
  PHX_CHECK(phx_mc_tet_mesh_probe_relocations(
                mesh, 0, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, 4,
                vertex_offsets.data(), vertices.data(), points.data(), cell_offsets.data(),
                cells.data(), decided.data(), statuses.data()) == PHX_MC_STALE_GENERATION);
  phx_mc_tet_mesh_free(mesh);
}

void batched_probe_guards_its_arguments() {
  phx_mc_tet_mesh* mesh = create_mesh();
  PHX_CHECK(mesh != nullptr);
  if (mesh == nullptr) return;
  constexpr uint64_t moved = 14;
  constexpr std::array<double, 3> point{0.3, 0.25, 0.25};
  constexpr std::array<int64_t, 2> vertex_offsets{0, 1};
  constexpr std::array<int64_t, 2> cell_offsets{0, 4};
  int32_t decided = -1;
  int32_t status = -1;
  const auto probe = [&](uint64_t generation, int32_t bound, const char* text, int64_t length,
                         int64_t groups, const int64_t* vertex_starts,
                         const int64_t* cell_starts) {
    return phx_mc_tet_mesh_probe_relocations(
        mesh, generation, bound, text, length, text, length, text, length, text, length, groups,
        vertex_starts, &moved, point.data(), cell_starts, kCellIds.data(), &decided, &status);
  };
  PHX_CHECK(probe(0, 0, nullptr, 0, 0, nullptr, nullptr) == PHX_MC_OK);
  PHX_CHECK(probe(1, 0, nullptr, 0, 0, nullptr, nullptr) == PHX_MC_STALE_GENERATION);
  PHX_CHECK(probe(0, 0, nullptr, 0, 1, vertex_offsets.data(), cell_offsets.data()) == PHX_MC_OK);
  PHX_CHECK(decided == 0);
  // An unbound mesh carries no source identity.
  constexpr char named[] = "revision";
  PHX_CHECK(probe(0, 1, named, sizeof(named) - 1, 1, vertex_offsets.data(),
                  cell_offsets.data()) == PHX_MC_STALE_GENERATION);
  PHX_CHECK(probe(0, 0, named, sizeof(named) - 1, 1, vertex_offsets.data(),
                  cell_offsets.data()) == PHX_MC_INVALID_ARGUMENT);
  constexpr std::array<int64_t, 2> shifted{1, 1};
  constexpr std::array<int64_t, 2> decreasing{0, -1};
  PHX_CHECK(probe(0, 0, nullptr, 0, 1, shifted.data(), cell_offsets.data()) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(probe(0, 0, nullptr, 0, 1, vertex_offsets.data(), decreasing.data()) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(probe(0, 0, nullptr, 0, 1, nullptr, cell_offsets.data()) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(probe(0, 0, nullptr, 0, -1, vertex_offsets.data(), cell_offsets.data()) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_tet_mesh_probe_relocations(
                nullptr, 0, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, 0, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(generation_of(mesh) == 0);
  phx_mc_tet_mesh_free(mesh);
}

}  // namespace

int main() {
  batched_probe_matches_the_single_request_probe();
  batched_probe_guards_its_arguments();
  return phx::mc::test::finish("test_tet_mesh_relocation");
}
