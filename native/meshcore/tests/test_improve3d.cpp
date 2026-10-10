//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Tetrahedral improvement: sliver removal to a dihedral target after
// refinement with constrained faces and region volumes preserved exactly,
// and the operation-level API (2-3 / 3-2 flips, edge removal, relocation,
// vertex removal) with its refusals.
#include <cmath>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <limits>
#include <set>

#include "check.hpp"
#include "phydrax_meshcore.h"
#include "improve3d.hpp"
#include "predicates.hpp"
#include "tet_mesh.hpp"
#include "tet_mesh_fixtures.hpp"

namespace {

using phx::mc::test::Domain;
using phx::mc::test::Exported;

double smallest_dihedral(const Exported& mesh) {
  double worst = 180.0;
  for (std::size_t t = 0; t < mesh.regions.size(); ++t) {
    double ratio = 0.0;
    double dihedral = 0.0;
    phx::mc::test::cell_quality(mesh, t, ratio, dihedral);
    worst = std::min(worst, dihedral);
  }
  return worst;
}

// Constrained faces as vertex-free geometry: sorted per-source areas.
std::vector<double> source_areas(const Exported& mesh) {
  std::vector<double> areas;
  for (int32_t source = 0; source < 7; ++source) {
    areas.push_back(phx::mc::test::source_area(mesh, source));
  }
  return areas;
}

void test_improvement_removes_slivers_preserving_constraints() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(3, 60, 13), 0.0);
  phx_mc_tet_mesh* mesh = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  PHX_CHECK(mesh != nullptr);
  int64_t refine_counters[PHX_MC_TET_MESH_REFINE_COUNTERS];
  PHX_CHECK(phx_mc_tet_mesh_refine(mesh, nullptr, 2.0, 1 << 20, int64_t{1} << 40,
                                   refine_counters) == PHX_MC_OK);
  const Exported refined = phx::mc::test::export_mesh(mesh);
  int64_t counters[PHX_MC_TET_MESH_IMPROVE_COUNTERS];
  const int32_t status = phx_mc_tet_mesh_improve(mesh, 12.0, 0.0, 20, int64_t{1} << 40, counters);
  const Exported improved = phx::mc::test::export_mesh(mesh);
  std::printf("improve: dihedral %.3f -> %.3f, flips23 %lld, edge removals %lld, moves %lld, "
              "passes %lld, remaining %lld\n",
              smallest_dihedral(refined), smallest_dihedral(improved),
              static_cast<long long>(counters[0]), static_cast<long long>(counters[1]),
              static_cast<long long>(counters[2]), static_cast<long long>(counters[4]),
              static_cast<long long>(counters[6]));
  PHX_CHECK(status == PHX_MC_OK);
  PHX_CHECK(smallest_dihedral(improved) >= 12.0);
  PHX_CHECK(smallest_dihedral(improved) > smallest_dihedral(refined));
  PHX_CHECK(phx::mc::test::all_positive(improved));
  PHX_CHECK(phx::mc::test::faces_on_sides(improved, 0.0));
  PHX_CHECK(improved.segments == refined.segments);
  const std::vector<double> before = source_areas(refined);
  const std::vector<double> after = source_areas(improved);
  for (std::size_t s = 0; s < before.size(); ++s) {
    PHX_CHECK_NEAR(after[s], before[s], 1e-12);
  }
  PHX_CHECK_NEAR(phx::mc::test::region_volume(improved, 0), 1.0, 1e-12);
  PHX_CHECK(handle_audit(mesh));
  double values[PHX_MC_TET_MESH_QUALITY_VALUES];
  int64_t histogram[PHX_MC_TET_MESH_QUALITY_BINS];
  int64_t slivers = -1;
  PHX_CHECK(phx_mc_tet_mesh_quality(mesh, 12.0, values, histogram, &slivers) == PHX_MC_OK);
  PHX_CHECK(slivers == 0);
  PHX_CHECK_NEAR(values[0], smallest_dihedral(improved), 1e-9);
  PHX_CHECK_NEAR(values[5], 1.0, 1e-12);
  int64_t binned = 0;
  for (int64_t count : histogram) {
    binned += count;
  }
  PHX_CHECK(binned == static_cast<int64_t>(improved.regions.size()));
  phx_mc_tet_mesh_free(mesh);
}

void test_unattainable_target_reports_remaining_slivers() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(3, 20, 21), 0.0);
  phx_mc_tet_mesh* mesh = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_FIXED);
  int64_t counters[PHX_MC_TET_MESH_IMPROVE_COUNTERS];
  PHX_CHECK(phx_mc_tet_mesh_improve(mesh, 70.0, 0.0, 3, int64_t{1} << 40, counters) ==
            PHX_MC_REFINEMENT_LIMIT);
  int64_t counts[PHX_MC_TET_MESH_COUNTS];
  phx_mc_tet_mesh_counts(mesh, counts);
  PHX_CHECK(counts[4] == counters[6] && counts[4] > 0);
  std::vector<int32_t> tets(4 * static_cast<std::size_t>(counts[4]));
  std::vector<int32_t> codes(2 * static_cast<std::size_t>(counts[4]));
  std::vector<double> values(static_cast<std::size_t>(counts[4]));
  PHX_CHECK(phx_mc_tet_mesh_unmet(mesh, tets.data(), codes.data(), values.data()) == PHX_MC_OK);
  bool dihedral = true;
  for (std::size_t i = 0; i < values.size(); ++i) {
    dihedral = dihedral && codes[2 * i] == PHX_MC_TET_MESH_CRITERION_DIHEDRAL && values[i] < 70.0;
  }
  PHX_CHECK(dihedral);
  const Exported after = phx::mc::test::export_mesh(mesh);
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), 1.0, 1e-12);
  phx_mc_tet_mesh_free(mesh);
}

// Two tetrahedra sharing the unconstrained face (0, 1, 2).
Domain bipyramid() {
  Domain domain;
  domain.points = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.25, 0.25, 1.0, 0.25, 0.25, -1.0};
  domain.tets = {0, 1, 2, 3, 0, 2, 1, 4};
  domain.regions = {0, 0};
  domain.faces = {0, 1, 3, 1, 2, 3, 0, 3, 2, 0, 4, 1, 1, 4, 2, 0, 2, 4};
  domain.face_sources = {0, 1, 2, 3, 4, 5};
  domain.segments = {0, 1, 1, 2, 0, 2, 0, 3, 1, 3, 2, 3, 0, 4, 1, 4, 2, 4};
  domain.segment_sources = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  return domain;
}

void test_flip_and_edge_removal_round_trip() {
  const Domain domain = bipyramid();
  int32_t status = -1;
  phx_mc_tet_mesh* mesh =
      phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_FIXED, nullptr, 16, 16, &status);
  PHX_CHECK(status == PHX_MC_OK);
  const int32_t shared[3] = {0, 1, 2};
  int32_t applied = -1;
  PHX_CHECK(phx_mc_tet_mesh_flip_face(mesh, shared, 0, &applied) == PHX_MC_OK && applied == 1);
  Exported flipped = phx::mc::test::export_mesh(mesh);
  PHX_CHECK(flipped.regions.size() == 3);
  PHX_CHECK(phx::mc::test::all_positive(flipped));
  PHX_CHECK_NEAR(phx::mc::test::region_volume(flipped, 0), 2.0 / 6.0, 1e-15);
  // The new edge (3, 4) is removed again by the 3-2 flip.
  PHX_CHECK(phx_mc_tet_mesh_remove_edge(mesh, 3, 4, 0, &applied) == PHX_MC_OK && applied == 1);
  Exported restored = phx::mc::test::export_mesh(mesh);
  PHX_CHECK(restored.tets == phx::mc::test::export_mesh(mesh).tets);
  PHX_CHECK(restored.regions.size() == 2);
  PHX_CHECK_NEAR(phx::mc::test::region_volume(restored, 0), 2.0 / 6.0, 1e-15);
  // Constrained faces and segments refuse.
  const int32_t boundary[3] = {0, 1, 3};
  PHX_CHECK(phx_mc_tet_mesh_flip_face(mesh, boundary, 0, &applied) == PHX_MC_OK && applied == 0);
  PHX_CHECK(phx_mc_tet_mesh_remove_edge(mesh, 0, 1, 0, &applied) == PHX_MC_OK && applied == 0);
  PHX_CHECK(handle_audit(mesh));
  phx_mc_tet_mesh_free(mesh);
}

void test_relocation_and_vertex_removal() {
  // Unit cube split into the cone of its center over its 12 boundary faces.
  Domain domain;
  domain.points = {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
                   0.5, 0.5, 0.5};
  const int32_t sides[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4},
                               {2, 6, 7, 3}, {0, 4, 6, 2}, {1, 3, 7, 5}};
  int32_t next = 0;
  for (int s = 0; s < 6; ++s) {
    const int32_t* q = sides[s];
    // Quad q is counterclockwise seen from outside... split along q0-q2.
    const int32_t triangles[2][3] = {{q[0], q[1], q[2]}, {q[0], q[2], q[3]}};
    for (const auto& tri : triangles) {
      domain.faces.insert(domain.faces.end(), {tri[0], tri[1], tri[2]});
      domain.face_sources.push_back(s);
      int32_t cell[4] = {tri[0], tri[1], tri[2], 8};
      const double* p[4];
      for (int k = 0; k < 4; ++k) {
        p[k] = domain.points.data() + 3 * cell[k];
      }
      if (phx::mc::orient3d(p[0], p[1], p[2], p[3]) < 0) {
        std::swap(cell[0], cell[1]);
      }
      domain.tets.insert(domain.tets.end(), cell, cell + 4);
      domain.regions.push_back(0);
    }
  }
  const int32_t edges[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3},
                                {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  for (const auto& edge : edges) {
    domain.segments.insert(domain.segments.end(), {edge[0], edge[1]});
    domain.segment_sources.push_back(next++);
  }
  int32_t status = -1;
  phx_mc_tet_mesh* mesh =
      phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_FIXED, nullptr, 16, 64, &status);
  PHX_CHECK(status == PHX_MC_OK);
  int32_t applied = -1;
  const double inside[3] = {0.375, 0.5, 0.625};
  PHX_CHECK(phx_mc_tet_mesh_relocate(mesh, 8, inside, 0, &applied) == PHX_MC_OK && applied == 1);
  PHX_CHECK(phx::mc::test::export_mesh(mesh).points[24] == 0.375);
  const double outside[3] = {1.5, 0.5, 0.5};
  PHX_CHECK(phx_mc_tet_mesh_relocate(mesh, 8, outside, 0, &applied) == PHX_MC_OK && applied == 0);
  // The centered position improves the star; the off-center one does not.
  const double center[3] = {0.5, 0.5, 0.5};
  PHX_CHECK(phx_mc_tet_mesh_relocate(mesh, 8, center, 1, &applied) == PHX_MC_OK && applied == 1);
  PHX_CHECK(phx_mc_tet_mesh_relocate(mesh, 8, inside, 1, &applied) == PHX_MC_OK && applied == 0);
  // Corner vertices are not relocatable.
  PHX_CHECK(phx_mc_tet_mesh_relocate(mesh, 0, center, 0, &applied) == PHX_MC_OK && applied == 0);
  PHX_CHECK(phx_mc_tet_mesh_remove_vertex(mesh, 8, 0, &applied) == PHX_MC_OK && applied == 1);
  const Exported removed = phx::mc::test::export_mesh(mesh);
  PHX_CHECK(removed.dimension[8] == -1);
  PHX_CHECK(removed.regions.size() == 6);
  PHX_CHECK(phx::mc::test::all_positive(removed));
  PHX_CHECK_NEAR(phx::mc::test::region_volume(removed, 0), 1.0, 1e-15);
  PHX_CHECK(handle_audit(mesh));
  phx_mc_tet_mesh_free(mesh);
}

void test_directional_split_collapse_preserves_fixed_boundary() {
  const Domain domain = bipyramid();
  phx_mc_tet_mesh* handle =
      phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_FIXED, nullptr, 16, 64);
  phx::mc::TetMesh& mesh = *handle->mesh;
  int32_t t = -1;
  int slot = -1;
  PHX_CHECK(mesh.find_face(0, 1, 2, t, slot));
  PHX_CHECK(phx::mc::try_face_removal(mesh, t, slot, false));
  const Exported original = phx::mc::test::export_mesh(handle);
  const double midpoint[3] = {0.25, 0.25, 0.0};
  int32_t inserted = -1;
  PHX_CHECK(phx::mc::try_split_edge(mesh, 3, 4, midpoint, 0.0, inserted) ==
            phx::mc::Insertion::kOk);
  PHX_CHECK(inserted == 5);
  PHX_CHECK(mesh.dimension(inserted) == 3);
  PHX_CHECK(phx::mc::try_collapse_edge(mesh, inserted, 3, false));
  const Exported collapsed = phx::mc::test::export_mesh(handle);
  PHX_CHECK(collapsed.tets == original.tets);
  PHX_CHECK(collapsed.faces == original.faces);
  PHX_CHECK(collapsed.segments == original.segments);
  PHX_CHECK(collapsed.dimension[5] == -1);
  const double boundary[3] = {0.5, 0.0, 0.0};
  PHX_CHECK(phx::mc::try_split_edge(mesh, 0, 1, boundary, 0.0, inserted) ==
            phx::mc::Insertion::kRefused);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == collapsed);
  PHX_CHECK(!phx::mc::try_collapse_edge(mesh, 0, 1, false));
  PHX_CHECK(handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}

Domain right_tetrahedron() {
  Domain domain;
  domain.points = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  domain.tets = {0, 1, 2, 3};
  domain.regions = {0};
  domain.faces = {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3};
  domain.face_sources = {0, 1, 2, 3};
  domain.segments = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};
  domain.segment_sources = {0, 1, 2, 3, 4, 5};
  return domain;
}

void test_conforming_curve_split_collapse_preserves_source_and_ghost_closure() {
  const Domain domain = right_tetrahedron();
  phx_mc_tet_mesh* handle = phx::mc::test::create_mesh(
      domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING, nullptr, 32, 128);
  const Exported original = phx::mc::test::export_mesh(handle);
  const int32_t protected_vertices[4] = {0, 1, 2, 3};
  PHX_CHECK(phx_mc_tet_mesh_protect_vertices(handle, 4, protected_vertices) == PHX_MC_OK);
  const double midpoint[3] = {0.5, 0, 0};
  int32_t inserted = -1;
  PHX_CHECK(phx::mc::try_split_edge(*handle->mesh, 0, 1, midpoint, 0, inserted) ==
            phx::mc::Insertion::kOk);
  PHX_CHECK(handle->mesh->dimension(inserted) == 1);
  const Exported split = phx::mc::test::export_mesh(handle);
  PHX_CHECK(!phx::mc::try_collapse_edge(*handle->mesh, 0, inserted, false));
  PHX_CHECK(phx::mc::test::export_mesh(handle) == split);
  PHX_CHECK(phx::mc::try_collapse_edge(*handle->mesh, inserted, 0, false));
  const Exported restored = phx::mc::test::export_mesh(handle);
  PHX_CHECK(restored.dimension[inserted] == -1);
  PHX_CHECK(restored.regions.size() == 1);
  PHX_CHECK(std::equal(original.points.begin(), original.points.end(), restored.points.begin()));
  PHX_CHECK(source_areas(restored) == source_areas(original));
  PHX_CHECK(phx::mc::test::all_positive(restored));
  PHX_CHECK_NEAR(phx::mc::test::region_volume(restored, 0), 1.0 / 6.0, 1e-15);
  PHX_CHECK(handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}

void test_conforming_segment_relocation_preserves_exact_source_and_volume() {
  const Domain domain = right_tetrahedron();
  phx_mc_tet_mesh* handle = phx::mc::test::create_mesh(
      domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING, nullptr, 32, 128);
  PHX_CHECK(handle != nullptr);
  const Exported original = phx::mc::test::export_mesh(handle);
  const int32_t corners[4] = {0, 1, 2, 3};
  PHX_CHECK(phx_mc_tet_mesh_protect_vertices(handle, 4, corners) == PHX_MC_OK);
  const double midpoint[3] = {0.5, 0.0, 0.0};
  int32_t inserted = -1;
  PHX_CHECK(phx_mc_tet_mesh_split_edge(
                handle, 0, 1, midpoint, 0.0, 0.0, 1000000, &inserted) == PHX_MC_OK);
  PHX_CHECK(inserted >= 0 && handle->mesh->dimension(inserted) == 1);
  const Exported before = phx::mc::test::export_mesh(handle);
  const double position[3] = {0.375, 0.0, 0.0};
  int32_t applied = 0;
  PHX_CHECK(phx_mc_tet_mesh_relocate(handle, inserted, position, 0, &applied) == PHX_MC_OK);
  PHX_CHECK(applied == 1);
  const Exported after = phx::mc::test::export_mesh(handle);
  PHX_CHECK(std::equal(position, position + 3, after.points.data() + 3 * inserted));
  PHX_CHECK(std::equal(original.points.begin(), original.points.end(), after.points.begin()));
  PHX_CHECK(after.dimension[inserted] == 1);
  PHX_CHECK(after.segments == before.segments);
  PHX_CHECK(after.segment_sources == before.segment_sources);
  PHX_CHECK(after.regions == before.regions);
  PHX_CHECK(source_areas(after) == source_areas(original));
  PHX_CHECK(phx::mc::test::all_positive(after));
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), 1.0 / 6.0, 1e-15);
  const auto& witness = handle->mesh->witness(inserted);
  PHX_CHECK(witness.stratum == phx::mc::SourceStratum::kSegment);
  PHX_CHECK(witness.entity == 0 && witness.parameters[0] == 0.375);
  PHX_CHECK(witness.deviation == 0.0);
  PHX_CHECK(handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}
void test_scheduled_segment_relocation_preserves_source_and_volume() {
  const Domain domain = right_tetrahedron();
  phx_mc_tet_mesh* handle = phx::mc::test::create_mesh(
      domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING, nullptr, 32, 128);
  PHX_CHECK(handle != nullptr);
  const Exported original = phx::mc::test::export_mesh(handle);
  const int32_t corners[4] = {0, 1, 2, 3};
  PHX_CHECK(phx_mc_tet_mesh_protect_vertices(handle, 4, corners) == PHX_MC_OK);
  const double midpoint[3] = {0.5, 0.0, 0.0};
  int32_t inserted = -1;
  PHX_CHECK(phx_mc_tet_mesh_split_edge(
                handle, 0, 1, midpoint, 0.0, 0.0, 1000000, &inserted) == PHX_MC_OK);
  const Exported before = phx::mc::test::export_mesh(handle);
  int64_t counters[PHX_MC_TET_MESH_IMPROVE_COUNTERS] = {};
  PHX_CHECK(phx_mc_tet_mesh_improve(handle, 70.0, 0.0, 1, 0, counters) ==
            PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == before);
  PHX_CHECK(handle_audit(handle));
  {
    phx::mc::NativeExecutionScope budget(
        1000000, 1000000, 0, 1 << 20, std::numeric_limits<double>::infinity());
    PHX_CHECK(phx_mc_tet_mesh_improve(handle, 70.0, 0.0, 1, 1000000, counters) ==
              PHX_MC_CAPACITY_EXCEEDED);
  }
  PHX_CHECK(phx::mc::test::export_mesh(handle) == before);
  PHX_CHECK(handle_audit(handle));
  uint64_t memory[6] = {};
  PHX_CHECK(phx_mc_tet_mesh_memory_evidence(handle, memory) == PHX_MC_OK);
  PHX_CHECK(phx_mc_tet_mesh_set_memory_limit(handle, static_cast<int64_t>(memory[1])) ==
            PHX_MC_OK);
  PHX_CHECK(phx_mc_tet_mesh_improve(handle, 70.0, 0.0, 1, 1000000, counters) ==
            PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(phx_mc_tet_mesh_set_memory_limit(handle, 1 << 20) == PHX_MC_OK);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == before);
  PHX_CHECK(handle_audit(handle));
  PHX_CHECK(phx_mc_tet_mesh_improve(handle, 70.0, 0.0, 1, 1000000, counters) ==
            PHX_MC_REFINEMENT_LIMIT);
  const Exported after = phx::mc::test::export_mesh(handle);
  PHX_CHECK(counters[2] > 0);
  PHX_CHECK(after.points[3 * inserted] != midpoint[0]);
  PHX_CHECK(smallest_dihedral(after) > smallest_dihedral(before));
  PHX_CHECK(std::equal(original.points.begin(), original.points.end(), after.points.begin()));
  PHX_CHECK(after.dimension[inserted] == 1);
  PHX_CHECK(after.segments == before.segments);
  PHX_CHECK(after.segment_sources == before.segment_sources);
  PHX_CHECK(after.regions == before.regions);
  PHX_CHECK(source_areas(after) == source_areas(original));
  PHX_CHECK(phx::mc::test::all_positive(after));
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), 1.0 / 6.0, 1e-15);
  const auto& witness = handle->mesh->witness(inserted);
  PHX_CHECK(witness.stratum == phx::mc::SourceStratum::kSegment);
  PHX_CHECK(witness.entity == 0 && witness.deviation == 0.0);
  PHX_CHECK(witness.parameters[0] == after.points[3 * inserted]);
  PHX_CHECK(handle_audit(handle));
  PHX_CHECK(phx_mc_tet_mesh_protect_vertices(handle, 1, &inserted) == PHX_MC_OK);
  const Exported protected_state = phx::mc::test::export_mesh(handle);
  PHX_CHECK(phx_mc_tet_mesh_improve(handle, 70.0, 0.0, 1, 1000000, counters) ==
            PHX_MC_REFINEMENT_LIMIT);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == protected_state);
  PHX_CHECK(handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}



void test_conforming_curve_collapse_work_refusal_preserves_accepted_state() {
  const Domain domain = right_tetrahedron();
  phx_mc_tet_mesh* handle = phx::mc::test::create_mesh(
      domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING, nullptr, 32, 128);
  const double midpoint[3] = {0.5, 0, 0};
  int32_t inserted = -1;
  PHX_CHECK(phx::mc::try_split_edge(*handle->mesh, 0, 1, midpoint, 0, inserted) ==
            phx::mc::Insertion::kOk);
  const Exported accepted = phx::mc::test::export_mesh(handle);
  int32_t applied = -1;
  PHX_CHECK(phx_mc_tet_mesh_collapse_edge(handle, inserted, 0, 0, 0, &applied) ==
            PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(applied == 0);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == accepted);
  PHX_CHECK(handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}

void test_conforming_curve_collapse_allocation_refusal_preserves_accepted_state() {
  const Domain domain = right_tetrahedron();
  phx_mc_tet_mesh* handle = phx::mc::test::create_mesh(
      domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING, nullptr, 32, 128);
  const double midpoint[3] = {0.5, 0, 0};
  int32_t inserted = -1;
  PHX_CHECK(phx::mc::try_split_edge(*handle->mesh, 0, 1, midpoint, 0, inserted) ==
            phx::mc::Insertion::kOk);
  const Exported accepted = phx::mc::test::export_mesh(handle);
  uint64_t memory[6] = {};
  PHX_CHECK(phx_mc_tet_mesh_memory_evidence(handle, memory) == PHX_MC_OK);
  PHX_CHECK(phx_mc_tet_mesh_set_memory_limit(handle, static_cast<int64_t>(memory[1])) ==
            PHX_MC_OK);
  int32_t applied = -1;
  PHX_CHECK(phx_mc_tet_mesh_collapse_edge(handle, inserted, 0, 0, 100000, &applied) ==
            PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(applied == 0);
  // Export/audit may allocate their own scratch after the refused mutation.
  PHX_CHECK(phx_mc_tet_mesh_set_memory_limit(handle, 1 << 20) == PHX_MC_OK);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == accepted);
  PHX_CHECK(handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}

void test_split_budget_rolls_back_complete_star() {
  const Domain domain = bipyramid();
  phx_mc_tet_mesh* handle =
      phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING, nullptr, 16, 64);
  const Exported before = phx::mc::test::export_mesh(handle);
  handle->mesh->set_work_limit(0);
  const double midpoint[3] = {0.5, 0.0, 0.0};
  int32_t inserted = -1;
  PHX_CHECK(phx::mc::try_split_edge(*handle->mesh, 0, 1, midpoint, 0.0, inserted) ==
            phx::mc::Insertion::kCapacity);
  PHX_CHECK(inserted == -1);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == before);
  PHX_CHECK(handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}

void test_weighted_regular_reconnection_removes_an_interior_sliver() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(3, 20, 1), 0.0);
  phx_mc_tet_mesh* handle =
      phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  PHX_CHECK(handle != nullptr);
  phx::mc::TetMesh& mesh = *handle->mesh;
  const Exported original = phx::mc::test::export_mesh(handle);
  int32_t t = -1;
  int slot = -1;
  PHX_CHECK(mesh.find_face(1, 28, 46, t, slot));
  const int32_t other = mesh.tet(t).n[slot];
  const int32_t a = mesh.tet(t).v[slot];
  const int32_t b = mesh.tet(other).v[phx::mc::neighbor_slot(mesh.tet(other), t)];
  PHX_CHECK(phx::mc::try_face_removal(mesh, t, slot, false));
  PHX_CHECK(mesh.edge_tet(a, b) >= 0);
  phx::mc::ExudeOptions options;
  options.min_dihedral_degrees = 20.0;
  options.max_weight_fraction = 0.25;
  options.max_passes = 2;
  phx::mc::NativeVector<double> weights;
  int64_t counters[6];
  PHX_CHECK(phx::mc::exude_mesh(mesh, options, weights, counters) == PHX_MC_REFINEMENT_LIMIT);
  PHX_CHECK(mesh.edge_tet(a, b) == -1);
  const Exported improved = phx::mc::test::export_mesh(handle);
  PHX_CHECK(improved.tets == original.tets);
  PHX_CHECK(improved.points == original.points);
  PHX_CHECK(improved.faces == original.faces);
  PHX_CHECK(improved.segments == original.segments);
  for (int32_t v = 0; v < mesh.vertex_count(); ++v) {
    if (mesh.dimension(v) < 3 || mesh.protection(v) > 0.0) {
      PHX_CHECK(weights[static_cast<std::size_t>(v)] == 0.0);
    }
  }
  PHX_CHECK(handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}

// Weighted exudation shares the publishing canonical-chart floor: the same
// regular 3-2 restoration is refused when its restored cells would chart at or
// below the floor, and admitted below it. No accepted move adds a below-floor cell.
void test_weighted_exudation_never_creates_a_cell_below_the_floor() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(3, 20, 1), 0.0);
  double restored_relative = 1.0;
  for (const double factor : {2.0, 0.5}) {
    phx_mc_tet_mesh* handle =
        phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
    PHX_CHECK(handle != nullptr);
    if (handle == nullptr) return;
    phx::mc::TetMesh& mesh = *handle->mesh;
    int32_t t = -1;
    int slot = -1;
    PHX_CHECK(mesh.find_face(1, 28, 46, t, slot));
    const int32_t other = mesh.tet(t).n[slot];
    const int32_t a = mesh.tet(t).v[slot];
    const int32_t b = mesh.tet(other).v[phx::mc::neighbor_slot(mesh.tet(other), t)];
    if (factor == 2.0) {
      // Canonical relative determinants of the two cells exudation restores.
      for (const int32_t cell : {t, other}) {
        const double* corners[4];
        mesh.corners(mesh.tet(cell).v, corners);
        double score = 0.0;
        phx::mc::canonical_relative_floor(mesh.tet(cell).v, corners, 0.5, &score);
        restored_relative = std::min(restored_relative, 0.5 * std::sqrt(score));
      }
      PHX_CHECK(restored_relative > 0.0 && restored_relative < 0.4);
    }
    PHX_CHECK(phx::mc::try_face_removal(mesh, t, slot, false));
    const double floor = factor * restored_relative;
    const auto below = [&]() {
      int64_t count = 0;
      for (std::size_t s = 0; s < mesh.complex().tets.size(); ++s) {
        const auto cell = static_cast<int32_t>(s);
        if (!mesh.complex().live(cell) || !mesh.finite(cell)) continue;
        const double* corners[4];
        mesh.corners(mesh.tet(cell).v, corners);
        double score = 0.0;
        count += phx::mc::canonical_relative_floor(mesh.tet(cell).v, corners, floor, &score) <= 0;
      }
      return count;
    };
    const int64_t before = below();
    phx::mc::ExudeOptions options;
    options.min_dihedral_degrees = 20.0;
    options.max_weight_fraction = 0.25;
    options.max_passes = 2;
    options.minimum_relative_determinant = floor;
    phx::mc::NativeVector<double> weights;
    int64_t counters[6];
    const int32_t status = phx::mc::exude_mesh(mesh, options, weights, counters);
    PHX_CHECK(status == PHX_MC_OK || status == PHX_MC_REFINEMENT_LIMIT);
    PHX_CHECK(below() <= before);
    // Above the restored cells' floor the regular restoration is refused.
    PHX_CHECK((mesh.edge_tet(a, b) >= 0) == (factor == 2.0));
    for (int32_t v = 0; v < mesh.vertex_count(); ++v) {
      if (mesh.dimension(v) < 3 || mesh.protection(v) > 0.0) {
        PHX_CHECK(weights[static_cast<std::size_t>(v)] == 0.0);
      }
    }
    PHX_CHECK(handle_audit(handle));
    phx_mc_tet_mesh_free(handle);
  }
}

// A below-floor cell that exudation cannot repair stays an unmet VALIDITY
// record; boundary-only vertices receive no weight and nothing changes.
void test_exudation_reports_unrepaired_below_floor_cells() {
  const Domain domain = phx::mc::test::reconstruction_hinge_domain();
  phx_mc_tet_mesh* handle = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_FIXED);
  PHX_CHECK(handle != nullptr);
  if (handle == nullptr) return;
  const Exported before = phx::mc::test::export_mesh(handle);
  phx::mc::ExudeOptions options;
  options.min_dihedral_degrees = 0.0;
  options.minimum_relative_determinant = 1.0e-12;
  options.max_passes = 2;
  phx::mc::NativeVector<double> weights;
  int64_t counters[6];
  PHX_CHECK(phx::mc::exude_mesh(*handle->mesh, options, weights, counters) ==
            PHX_MC_REFINEMENT_LIMIT);
  PHX_CHECK(counters[1] == 0 && counters[2] == 0);
  int32_t cell[4] = {}, codes[2] = {};
  double margin = 0.0;
  PHX_CHECK(phx_mc_tet_mesh_unmet(handle, cell, codes, &margin) == PHX_MC_OK);
  PHX_CHECK(codes[0] == PHX_MC_TET_MESH_CRITERION_VALIDITY && margin <= 0.0);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == before && handle_audit(handle));
  options.minimum_relative_determinant = 1.0;
  PHX_CHECK(phx::mc::exude_mesh(*handle->mesh, options, weights, counters) ==
            PHX_MC_INVALID_ARGUMENT);
  phx_mc_tet_mesh_free(handle);
}

void test_protected_interior_vertex_cannot_move_or_retire() {
  phx_mc_tet_mesh* handle =
      phx::mc::test::create_mesh(bipyramid(), PHX_MC_TET_MESH_BOUNDARY_FIXED, nullptr, 16, 64);
  PHX_CHECK(handle != nullptr);
  phx::mc::TetMesh& mesh = *handle->mesh;
  int32_t t = -1;
  int slot = -1;
  PHX_CHECK(mesh.find_face(0, 1, 2, t, slot));
  PHX_CHECK(phx::mc::try_face_removal(mesh, t, slot, false));
  const double position[3] = {0.25, 0.25, 0.0};
  int32_t inserted = -1;
  PHX_CHECK(mesh.split_edge(3, 4, position, 0.0, inserted) == phx::mc::Insertion::kOk);
  const int32_t invalid[2] = {inserted, 1000};
  PHX_CHECK(!mesh.protect_vertices(2, invalid));
  PHX_CHECK(mesh.dimension(inserted) == 3);
  PHX_CHECK(mesh.protect_vertices(1, &inserted));
  const Exported protected_mesh = phx::mc::test::export_mesh(handle);
  const double moved[3] = {0.25, 0.25, 0.125};
  PHX_CHECK(!phx::mc::try_relocate(mesh, inserted, moved, false));
  PHX_CHECK(!phx::mc::try_collapse_edge(mesh, inserted, 3, false));
  PHX_CHECK(!phx::mc::try_remove_vertex(mesh, inserted, false, nullptr));
  PHX_CHECK(phx::mc::test::export_mesh(handle) == protected_mesh);
  phx_mc_tet_mesh_free(handle);
}

void test_protection_ball_center_rejects_otherwise_legal_relocation() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(3, 20, 1), 0.0);
  std::vector<double> radii(domain.points.size() / 3, 0.0);
  radii[27] = 1e-4;
  phx_mc_tet_mesh* handle =
      phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING, radii.data());
  PHX_CHECK(handle != nullptr);
  phx::mc::TetMesh& mesh = *handle->mesh;
  PHX_CHECK(mesh.dimension(27) == 3);
  const double position[3] = {mesh.point(27)[0] + 0x1p-25, mesh.point(27)[1], mesh.point(27)[2]};
  std::vector<int32_t> star;
  PHX_CHECK(mesh.vertex_star(27, star));
  for (int32_t t : star) {
    if (mesh.finite(t)) {
      PHX_CHECK(mesh.orient_with(t, phx::mc::vertex_slot(mesh.tet(t), 27), position) > 0);
    }
  }
  const Exported before = phx::mc::test::export_mesh(handle);
  PHX_CHECK(!phx::mc::try_relocate(mesh, 27, position, false));
  PHX_CHECK(phx::mc::test::export_mesh(handle) == before);
  phx_mc_tet_mesh_free(handle);
}

void test_expanded_insertion_inspection_is_atomic_and_epoch_bound() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(3, 20, 1), 0.0);
  phx_mc_tet_mesh* handle =
      phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  PHX_CHECK(handle != nullptr);
  const Exported original = phx::mc::test::export_mesh(handle);
  constexpr int64_t limit = 512;
  std::vector<int32_t> removed(4 * limit), proposed(4 * limit);
  int64_t counts[4] = {};
  std::set<std::array<int32_t, 2>> edges;
  for (std::size_t row = 0; row < original.regions.size(); ++row) {
    for (int a = 0; a < 4; ++a) {
      for (int b = a + 1; b < 4; ++b) {
        std::array<int32_t, 2> edge{original.tets[4 * row + a], original.tets[4 * row + b]};
        std::sort(edge.begin(), edge.end());
        edges.insert(edge);
      }
    }
  }
  std::array<int32_t, 2> seed{-1, -1};
  double position[3] = {};
  for (const auto& edge : edges) {
    for (int axis = 0; axis < 3; ++axis) {
      position[axis] = 0.5 * original.points[3 * edge[0] + axis] +
                       0.5 * original.points[3 * edge[1] + axis];
    }
    PHX_CHECK(phx_mc_tet_mesh_inspect_edge_insertion(
                  handle, edge[0], edge[1], position, position, 0.0, limit, 1000000,
                  removed.data(), proposed.data(), counts) == PHX_MC_OK);
    int64_t star = 0;
    for (std::size_t row = 0; row < original.regions.size(); ++row) {
      const int32_t* cell = original.tets.data() + 4 * row;
      star += std::find(cell, cell + 4, edge[0]) != cell + 4 &&
              std::find(cell, cell + 4, edge[1]) != cell + 4;
    }
    if (counts[2] == 0 && counts[0] > star) {
      seed = edge;
      break;
    }
  }
  PHX_CHECK(seed[0] >= 0);
  PHX_CHECK(counts[0] > 0 && counts[1] > 0 && counts[0] + counts[1] <= limit);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == original);
  const std::array<int64_t, 4> inspected{counts[0], counts[1], counts[2], counts[3]};
  std::vector<int32_t> mismatch = proposed;
  std::swap(mismatch[0], mismatch[1]);
  int32_t inserted = -1;
  PHX_CHECK(phx_mc_tet_mesh_commit_edge_insertion(
                handle, seed[0], seed[1], position, position, 0.0, limit,
                static_cast<int32_t>(inspected[2]), inspected[3], inspected[0], removed.data(),
                inspected[1], mismatch.data(), 0.0, 1000000, &inserted) == PHX_MC_OK);
  PHX_CHECK(inserted == -1 && phx::mc::test::export_mesh(handle) == original);
  PHX_CHECK(phx_mc_tet_mesh_inspect_edge_insertion(
                handle, seed[0], seed[1], position, position, 0.0, 1, 1000000,
                removed.data(), proposed.data(), counts) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == original);
  PHX_CHECK(phx_mc_tet_mesh_inspect_edge_insertion(
                handle, seed[0], seed[1], position, position, 0.0, limit, 0,
                removed.data(), proposed.data(), counts) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == original);
  PHX_CHECK(phx_mc_tet_mesh_inspect_edge_insertion(
                handle, seed[0], seed[1], position, position, 0.0, limit, 1000000,
                removed.data(), proposed.data(), counts) == PHX_MC_OK);
  const std::array<int64_t, 4> fresh{counts[0], counts[1], counts[2], counts[3]};
  PHX_CHECK(phx_mc_tet_mesh_commit_edge_insertion(
                handle, seed[0], seed[1], position, position, 0.0, limit,
                static_cast<int32_t>(fresh[2]), fresh[3], fresh[0], removed.data(),
                fresh[1], proposed.data(), 0.0, 0, &inserted) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(inserted == -1 && phx::mc::test::export_mesh(handle) == original);
  double moved[3];
  std::copy_n(handle->mesh->point(27), 3, moved);
  moved[0] += 0x1p-20;
  handle->mesh->set_work_limit(1000000);
  int32_t applied = 0;
  PHX_CHECK(phx_mc_tet_mesh_relocate(handle, 27, moved, 0, &applied) == PHX_MC_OK && applied == 1);
  const Exported relocated = phx::mc::test::export_mesh(handle);
  PHX_CHECK(phx_mc_tet_mesh_commit_edge_insertion(
                handle, seed[0], seed[1], position, position, 0.0, limit,
                static_cast<int32_t>(fresh[2]), fresh[3], fresh[0], removed.data(),
                fresh[1], proposed.data(), 0.0, 1000000, &inserted) == PHX_MC_OK);
  PHX_CHECK(inserted == -1 && phx::mc::test::export_mesh(handle) == relocated);
  handle->mesh->set_work_limit(1000000);
  PHX_CHECK(phx_mc_tet_mesh_relocate(
                handle, 27, original.points.data() + 3 * 27, 0, &applied) == PHX_MC_OK);
  PHX_CHECK(phx_mc_tet_mesh_inspect_edge_insertion(
                handle, seed[0], seed[1], position, position, 0.0, limit, 1000000,
                removed.data(), proposed.data(), counts) == PHX_MC_OK);
  PHX_CHECK(phx_mc_tet_mesh_commit_edge_insertion(
                handle, seed[0], seed[1], position, position, 0.0, limit,
                static_cast<int32_t>(counts[2]), counts[3], counts[0], removed.data(),
                counts[1], proposed.data(), 0.0, 1000000, &inserted) == PHX_MC_OK);
  PHX_CHECK(inserted == static_cast<int32_t>(original.points.size() / 3));
  const Exported accepted = phx::mc::test::export_mesh(handle);
  PHX_CHECK(std::equal(original.points.begin(), original.points.end(), accepted.points.begin()));
  PHX_CHECK(accepted.regions.size() == original.regions.size() - counts[0] + counts[1]);
  PHX_CHECK(accepted.faces == original.faces && accepted.segments == original.segments);
  PHX_CHECK(phx::mc::test::all_positive(accepted) && handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}

void test_shared_work_refusal_is_not_geometric_edit_refusal() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(3, 20, 1), 0.0);
  phx_mc_tet_mesh* handle =
      phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  PHX_CHECK(handle != nullptr);
  const Exported before = phx::mc::test::export_mesh(handle);
  int32_t applied = -1;
  const int32_t face[3] = {1, 28, 46};
  handle->mesh->set_work_limit(0);
  PHX_CHECK(phx_mc_tet_mesh_relocate(
                handle, 27, handle->mesh->point(27), 0, &applied) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(applied == 0 && phx::mc::test::export_mesh(handle) == before);
  handle->mesh->set_work_limit(0);
  PHX_CHECK(phx_mc_tet_mesh_flip_face(handle, face, 0, &applied) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(applied == 0 && phx::mc::test::export_mesh(handle) == before);
  handle->mesh->set_work_limit(0);
  PHX_CHECK(phx_mc_tet_mesh_remove_edge(handle, 27, 28, 0, &applied) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(applied == 0 && phx::mc::test::export_mesh(handle) == before);
  handle->mesh->set_work_limit(0);
  PHX_CHECK(phx_mc_tet_mesh_remove_vertex(handle, 27, 0, &applied) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(applied == 0 && phx::mc::test::export_mesh(handle) == before);
  handle->mesh->set_work_limit(1000000);
  PHX_CHECK(phx_mc_tet_mesh_relocate(
                handle, 27, handle->mesh->point(27), 0, &applied) == PHX_MC_OK && applied == 1);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == before && handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}

void test_exude_allocation_refusal_publishes_actual_weight_state() {
  phx_mc_tet_mesh* initial = phx::mc::test::create_mesh(
      right_tetrahedron(), PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  PHX_CHECK(initial != nullptr);
  const Exported unchanged = phx::mc::test::export_mesh(initial);
  const auto first_memory = initial->mesh->memory_evidence();
  PHX_CHECK(initial->mesh->set_memory_limit(first_memory[1]));
  double first_weights[4];
  std::fill_n(first_weights, 4, std::numeric_limits<double>::quiet_NaN());
  int64_t counters[6];
  PHX_CHECK(phx_mc_tet_mesh_exude(
                initial, 20.0, 0.25, 2.0, 0.0, 2, 1000000, first_weights, counters) ==
            PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(std::all_of(first_weights, first_weights + 4, [](double value) { return value == 0.0; }));
  PHX_CHECK(initial->mesh->set_memory_limit(1 << 20));
  PHX_CHECK(phx::mc::test::export_mesh(initial) == unchanged);
  phx_mc_tet_mesh_free(initial);

}

void test_zero_angle_goal_does_not_hide_uncertain_construction() {
  const Domain domain = phx::mc::test::reconstruction_sliver_domain();
  phx_mc_tet_mesh* handle = phx::mc::test::create_mesh(
      domain, PHX_MC_TET_MESH_BOUNDARY_FIXED);
  PHX_CHECK(handle != nullptr);
  if (handle == nullptr) return;
  const Exported before = phx::mc::test::export_mesh(handle);
  int64_t counters[PHX_MC_TET_MESH_IMPROVE_COUNTERS] = {};
  PHX_CHECK(phx_mc_tet_mesh_improve(handle, 0.0, 0.0, 1, 100000, counters) == PHX_MC_REFINEMENT_LIMIT);
  PHX_CHECK(counters[6] == 0);
  int64_t counts[PHX_MC_TET_MESH_COUNTS] = {};
  PHX_CHECK(phx_mc_tet_mesh_counts(handle, counts) == PHX_MC_OK && counts[4] == 1);
  int32_t cells[4] = {}, codes[2] = {};
  double value = 0.0;
  PHX_CHECK(phx_mc_tet_mesh_unmet(handle, cells, codes, &value) == PHX_MC_OK);
  PHX_CHECK(codes[0] == PHX_MC_TET_MESH_CRITERION_CONSTRUCTION);
  PHX_CHECK(codes[1] == PHX_MC_TET_MESH_REASON_NONREPRESENTABLE && value <= 0.0);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == before && handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}

void test_nonconvex_construction_ring_refuses_illegal_children() {
  const Domain domain = phx::mc::test::reconstruction_ring_domain();
  phx_mc_tet_mesh* handle = phx::mc::test::create_mesh(
      domain, PHX_MC_TET_MESH_BOUNDARY_FIXED);
  PHX_CHECK(handle != nullptr);
  if (handle == nullptr) return;
  const Exported before = phx::mc::test::export_mesh(handle);
  const auto consumer_measure = [](const Exported& data, std::size_t offset) {
    double cyclic[4][3];
    for (int corner = 0; corner < 4; ++corner) {
      const double* p = data.points.data() + 3 * data.tets[offset + corner];
      cyclic[corner][0] = p[1];
      cyclic[corner][1] = p[2];
      cyclic[corner][2] = p[0];
    }
    return (-phx::mc::orient3d_approx(
        cyclic[1], cyclic[2], cyclic[3], cyclic[0])).value;
  };
  bool negative_before = false;
  for (std::size_t offset = 0; offset < before.tets.size(); offset += 4) {
    negative_before = negative_before || consumer_measure(before, offset) <= 0.0;
  }
  PHX_CHECK(negative_before);
  // Both ring diagonals contain an exact inverted child. Disabling the
  // dihedral objective must not disable original-coordinate legality.
  int32_t applied = -1;
  PHX_CHECK(phx_mc_tet_mesh_remove_edge(handle, 1, 3, 0, &applied) == PHX_MC_OK);
  PHX_CHECK(applied == 0);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == before);
  PHX_CHECK(handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}

void test_construction_three_ring_repair_preserves_exact_source_boundary() {
  const Domain domain = phx::mc::test::reconstruction_repair_domain();
  phx_mc_tet_mesh* handle = phx::mc::test::create_mesh(
      domain, PHX_MC_TET_MESH_BOUNDARY_FIXED);
  PHX_CHECK(handle != nullptr);
  if (handle == nullptr) return;
  const Exported before = phx::mc::test::export_mesh(handle);
  int64_t counters[PHX_MC_TET_MESH_IMPROVE_COUNTERS] = {};
  PHX_CHECK(phx_mc_tet_mesh_improve(handle, 0.0, 1.0e-12, 1, 100000, counters) == PHX_MC_OK);
  const Exported after = phx::mc::test::export_mesh(handle);
  PHX_CHECK(after.points == before.points && after.faces == before.faces &&
            after.face_sources == before.face_sources && after.segments == before.segments &&
            after.segment_sources == before.segment_sources && after.dimension == before.dimension);
  for (std::size_t offset = 0; offset < after.tets.size(); offset += 4) {
    double cyclic[4][3];
    for (int corner = 0; corner < 4; ++corner) {
      const double* p = after.points.data() + 3 * after.tets[offset + corner];
      cyclic[corner][0] = p[1];
      cyclic[corner][1] = p[2];
      cyclic[corner][2] = p[0];
    }
    PHX_CHECK((-phx::mc::orient3d_approx(
        cyclic[1], cyclic[2], cyclic[3], cyclic[0])).value > 0.0);
  }
  PHX_CHECK(handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}

void test_determinant_floor_hinge_receives_certified_interior_vertex() {
  constexpr double kFloor = 1.0e-12;
  const Domain domain = phx::mc::test::reconstruction_hinge_domain();
  phx_mc_tet_mesh* handle = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_FIXED);
  PHX_CHECK(handle != nullptr);
  if (handle == nullptr) return;
  const Exported before = phx::mc::test::export_mesh(handle);
  const auto above_floor = [&](const Exported& data, std::size_t offset) {
    const double* p[4];
    for (int corner = 0; corner < 4; ++corner) {
      p[corner] = data.points.data() + 3 * data.tets[offset + corner];
    }
    double score = 0.0;
    return phx::mc::relative_orient3d_exact(p[0], p[1], p[2], p[3], kFloor, &score) == 1;
  };
  bool below_before = false;
  for (std::size_t offset = 0; offset < before.tets.size(); offset += 4) {
    below_before = below_before || !above_floor(before, offset);
  }
  PHX_CHECK(below_before);
  int64_t counters[PHX_MC_TET_MESH_IMPROVE_COUNTERS] = {};
  PHX_CHECK(phx_mc_tet_mesh_improve(handle, 0.0, kFloor, 1, 100000, counters) == PHX_MC_OK);
  PHX_CHECK(counters[8] == 1);
  const Exported after = phx::mc::test::export_mesh(handle);
  PHX_CHECK(std::equal(before.points.begin(), before.points.end(), after.points.begin()));
  PHX_CHECK(after.faces == before.faces && after.face_sources == before.face_sources &&
            after.segments == before.segments && after.segment_sources == before.segment_sources);
  PHX_CHECK(after.dimension.size() == before.dimension.size() + 1 && after.dimension.back() == 3);
  for (std::size_t offset = 0; offset < after.tets.size(); offset += 4) {
    PHX_CHECK(above_floor(after, offset));
  }
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), phx::mc::test::region_volume(before, 0),
                 1e-18);
  PHX_CHECK(handle_audit(handle));
  phx_mc_tet_mesh_free(handle);

  // Without vertex allowance the repair is a capacity refusal, not a hidden
  // geometric failure, and the accepted state is unchanged.
  int32_t status = -1;
  phx_mc_tet_mesh* capped = phx::mc::test::create_mesh(
      domain, PHX_MC_TET_MESH_BOUNDARY_FIXED, nullptr, 6, 1 << 10, &status);
  PHX_CHECK(status == PHX_MC_OK && capped != nullptr);
  if (capped == nullptr) return;
  PHX_CHECK(phx_mc_tet_mesh_improve(capped, 0.0, kFloor, 1, 100000, counters) ==
            PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(counters[8] == 0);
  PHX_CHECK(phx::mc::test::export_mesh(capped) == before && handle_audit(capped));
  phx_mc_tet_mesh_free(capped);
}

// One positive tetrahedron whose relative determinant is ~0.0705 from vertex
// (0, 0, 0) and ~3.9e-4 from vertex (10, 10, 1). `far_first` gives the far
// vertex identity 0, so the publishing canonical chart starts there.
Domain chart_domain(bool far_first) {
  Domain domain;
  const double near[4][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {10, 10, 1}};
  const int order[4] = {3, 0, 1, 2};
  for (int k = 0; k < 4; ++k) {
    const double* p = near[far_first ? order[k] : k];
    domain.points.insert(domain.points.end(), p, p + 3);
  }
  domain.tets = far_first ? std::vector<int32_t>{1, 2, 3, 0} : std::vector<int32_t>{0, 1, 2, 3};
  domain.regions = {0};
  domain.faces = {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
  domain.face_sources = {0, 1, 2, 3};
  domain.segments = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};
  domain.segment_sources = {0, 1, 2, 3, 4, 5};
  return domain;
}

void test_relative_floor_uses_the_publishing_canonical_chart_only() {
  constexpr double kFloor = 0.01;
  const Domain near = chart_domain(false);
  phx_mc_tet_mesh* handle = phx::mc::test::create_mesh(near, PHX_MC_TET_MESH_BOUNDARY_FIXED);
  PHX_CHECK(handle != nullptr);
  if (handle == nullptr) return;
  // Proposal assessment charts by explicit int64 identities, never local rows.
  const int32_t rows[4] = {0, 1, 2, 3};
  const int64_t near_ids[4] = {0, 1, 2, 3};
  const int64_t far_ids[4] = {1, 2, 3, 0};
  // Sparse, nonmonotone and beyond int32: the smallest identity (local row 1
  // here, row 3 below) alone selects the chart.
  const int64_t sparse_near_ids[4] = {900, 7, int64_t{1} << 40, 50};
  const int64_t sparse_far_ids[4] = {int64_t{9} << 50, (int64_t{1} << 40) + 5,
                                     (int64_t{1} << 40) + 7, 3};
  const int64_t repeated_ids[4] = {0, 1, 1, 2};
  const int64_t negative_ids[4] = {0, -1, 2, 3};
  double values[3] = {};
  const auto below = [&](const int64_t* ids) {
    values[2] = -1.0;
    PHX_CHECK(phx_mc_tet_mesh_proposal_shape(handle, 4, near.points.data(), ids, 1, rows,
                                             kFloor, 1000, values) == PHX_MC_OK);
    return values[2];
  };
  PHX_CHECK(below(near_ids) == 0.0);
  PHX_CHECK(below(far_ids) == 1.0);
  PHX_CHECK(below(sparse_near_ids) == 0.0);
  PHX_CHECK(below(sparse_far_ids) == 1.0);
  PHX_CHECK(phx_mc_tet_mesh_proposal_shape(handle, 4, near.points.data(), repeated_ids, 1, rows,
                                           kFloor, 1000, values) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(phx_mc_tet_mesh_proposal_shape(handle, 4, near.points.data(), negative_ids, 1, rows,
                                           kFloor, 1000, values) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(phx_mc_tet_mesh_proposal_shape(handle, 4, near.points.data(), near_ids, 1, rows,
                                           1.0, 1000, values) == PHX_MC_INVALID_ARGUMENT);
  // A stricter alternate chart must not reject what the policy admits.
  const Exported before = phx::mc::test::export_mesh(handle);
  int64_t counters[PHX_MC_TET_MESH_IMPROVE_COUNTERS] = {};
  PHX_CHECK(phx_mc_tet_mesh_improve(handle, 0.0, kFloor, 1, 100000, counters) == PHX_MC_OK);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == before && handle_audit(handle));
  phx_mc_tet_mesh_free(handle);

  phx_mc_tet_mesh* far = phx::mc::test::create_mesh(
      chart_domain(true), PHX_MC_TET_MESH_BOUNDARY_FIXED);
  PHX_CHECK(far != nullptr);
  if (far == nullptr) return;
  const Exported fixed = phx::mc::test::export_mesh(far);
  PHX_CHECK(phx_mc_tet_mesh_improve(far, 0.0, kFloor, 1, 100000, counters) ==
            PHX_MC_REFINEMENT_LIMIT);
  int32_t cell[4] = {}, codes[2] = {};
  double margin = 0.0;
  PHX_CHECK(phx_mc_tet_mesh_unmet(far, cell, codes, &margin) == PHX_MC_OK);
  PHX_CHECK(codes[0] == PHX_MC_TET_MESH_CRITERION_VALIDITY && margin <= 0.0);
  PHX_CHECK(codes[1] == PHX_MC_TET_MESH_REASON_NO_IMPROVEMENT);
  PHX_CHECK(phx::mc::test::export_mesh(far) == fixed && handle_audit(far));
  phx_mc_tet_mesh_free(far);
}

}  // namespace

int main() {
  test_determinant_floor_hinge_receives_certified_interior_vertex();
  test_relative_floor_uses_the_publishing_canonical_chart_only();
  test_nonconvex_construction_ring_refuses_illegal_children();
  test_construction_three_ring_repair_preserves_exact_source_boundary();
  test_zero_angle_goal_does_not_hide_uncertain_construction();
  test_improvement_removes_slivers_preserving_constraints();
  test_unattainable_target_reports_remaining_slivers();
  test_flip_and_edge_removal_round_trip();
  test_relocation_and_vertex_removal();
  test_directional_split_collapse_preserves_fixed_boundary();
  test_conforming_curve_split_collapse_preserves_source_and_ghost_closure();
  test_conforming_segment_relocation_preserves_exact_source_and_volume();
  test_scheduled_segment_relocation_preserves_source_and_volume();
  test_conforming_curve_collapse_work_refusal_preserves_accepted_state();
  test_conforming_curve_collapse_allocation_refusal_preserves_accepted_state();
  test_split_budget_rolls_back_complete_star();
  test_weighted_regular_reconnection_removes_an_interior_sliver();
  test_weighted_exudation_never_creates_a_cell_below_the_floor();
  test_exudation_reports_unrepaired_below_floor_cells();
  test_protected_interior_vertex_cannot_move_or_retire();
  test_protection_ball_center_rejects_otherwise_legal_relocation();
  test_expanded_insertion_inspection_is_atomic_and_epoch_bound();
  test_shared_work_refusal_is_not_geometric_edit_refusal();
  test_exude_allocation_refusal_publishes_actual_weight_state();
  return phx::mc::test::finish("test_improve3d");
}
