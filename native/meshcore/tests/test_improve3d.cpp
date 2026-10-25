//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Tetrahedral improvement: sliver removal to a dihedral target after
// refinement with constrained faces and region volumes preserved exactly,
// and the operation-level API (2-3 / 3-2 flips, edge removal, relocation,
// vertex removal) with its refusals. Weighted exudation with per-visit reuse
// is compared with a literal replay of the recomputing exudation.
#include <algorithm>
#include <cmath>
#include <array>
#include <cstdint>
#include <span>
#include <vector>
#include <limits>
#include <set>

#include "check.hpp"
#include "phydrax_meshcore.h"
#include "improve3d.hpp"
#include "geometry3.hpp"
#include "predicates.hpp"
#include "tet_mesh.hpp"
#include "tet_mesh_fixtures.hpp"

namespace {

using phx::mc::test::Domain;
using phx::mc::test::Exported;

// Historical empirical oracle allowances from this file's analytic fixtures.
// Unit-box measures use tet_mesh_fixtures.hpp::{region_volume,source_area};
// the small flip/collapse cavities sum only a few determinant/6 terms.
// Keep these acceptance values separate from the native quality/validity policy.
constexpr double kUnitBoxMeasureTolerance = 1e-12;
constexpr double kSmallCavityVolumeTolerance = 1e-15;
// cell_quality's independent acos oracle is compared with native atan2 angles.
constexpr double kDihedralOracleTolerance = 1e-9;
// reconstruction_hinge_domain has much smaller volume than the unit-box cases;
// retain its existing empirical determinant-sum allowance rather than scaling up.
constexpr double kHingeVolumeTolerance = 1e-18;
// The reconstruction-hinge tests request this canonical-chart validity policy;
// it is a tested admission threshold, not numerical slack on an assertion.
constexpr double kReconstructionFloor = 1.0e-12;

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
  // The coupled optimizer can expand through these boundary rings. Their
  // ghost closure must never become a physical optimization variable.
  PHX_CHECK(refined.faces.size() >= 3);
  phx::mc::NativeVector<int32_t> boundary_star, boundary_ring;
  PHX_CHECK(mesh->mesh->edge_ring(
      refined.faces[0], refined.faces[1], boundary_star, boundary_ring));
  PHX_CHECK(std::find(boundary_ring.begin(), boundary_ring.end(), phx::mc::kGhostVertex) !=
            boundary_ring.end());
  int64_t counters[PHX_MC_TET_MESH_IMPROVE_COUNTERS];
  const int32_t status = phx_mc_tet_mesh_improve(mesh, 12.0, 0.0, 20, int64_t{1} << 40, counters);
  const Exported improved = phx::mc::test::export_mesh(mesh);
  PHX_CHECK(status == PHX_MC_OK);
  PHX_CHECK(smallest_dihedral(improved) >= 12.0);
  PHX_CHECK(smallest_dihedral(improved) > smallest_dihedral(refined));
  PHX_CHECK(phx::mc::test::all_positive(improved));
  PHX_CHECK(phx::mc::test::faces_on_sides(improved, 0.0));
  PHX_CHECK(improved.segments == refined.segments);
  const std::vector<double> before = source_areas(refined);
  const std::vector<double> after = source_areas(improved);
  for (std::size_t s = 0; s < before.size(); ++s) {
    PHX_CHECK_NEAR(after[s], before[s], kUnitBoxMeasureTolerance);
  }
  PHX_CHECK_NEAR(phx::mc::test::region_volume(improved, 0), 1.0, kUnitBoxMeasureTolerance);
  PHX_CHECK(handle_audit(mesh));
  double values[PHX_MC_TET_MESH_QUALITY_VALUES];
  int64_t histogram[PHX_MC_TET_MESH_QUALITY_BINS];
  int64_t slivers = -1;
  PHX_CHECK(phx_mc_tet_mesh_quality(mesh, 12.0, values, histogram, &slivers) == PHX_MC_OK);
  PHX_CHECK(slivers == 0);
  PHX_CHECK_NEAR(values[0], smallest_dihedral(improved), kDihedralOracleTolerance);
  PHX_CHECK_NEAR(values[5], 1.0, kUnitBoxMeasureTolerance);
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
  PHX_CHECK(counts[PHX_MC_TET_MESH_COUNT_UNMET_RECORDS] == counters[PHX_MC_TET_MESH_IMPROVE_REMAINING_SLIVERS] && counts[PHX_MC_TET_MESH_COUNT_UNMET_RECORDS] > 0);
  std::vector<int32_t> tets(4 * static_cast<std::size_t>(counts[PHX_MC_TET_MESH_COUNT_UNMET_RECORDS]));
  std::vector<int32_t> codes(2 * static_cast<std::size_t>(counts[PHX_MC_TET_MESH_COUNT_UNMET_RECORDS]));
  std::vector<double> values(static_cast<std::size_t>(counts[PHX_MC_TET_MESH_COUNT_UNMET_RECORDS]));
  PHX_CHECK(phx_mc_tet_mesh_unmet(mesh, tets.data(), codes.data(), values.data()) == PHX_MC_OK);
  bool dihedral = true;
  for (std::size_t i = 0; i < values.size(); ++i) {
    dihedral = dihedral && codes[2 * i] == PHX_MC_TET_MESH_CRITERION_DIHEDRAL && values[i] < 70.0;
  }
  PHX_CHECK(dihedral);
  const Exported after = phx::mc::test::export_mesh(mesh);
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), 1.0, kUnitBoxMeasureTolerance);
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
  PHX_CHECK_NEAR(phx::mc::test::region_volume(flipped, 0), 2.0 / 6.0, kSmallCavityVolumeTolerance);
  // The new edge (3, 4) is removed again by the 3-2 flip.
  PHX_CHECK(phx_mc_tet_mesh_remove_edge(mesh, 3, 4, 0, &applied) == PHX_MC_OK && applied == 1);
  Exported restored = phx::mc::test::export_mesh(mesh);
  PHX_CHECK(restored.tets == phx::mc::test::export_mesh(mesh).tets);
  PHX_CHECK(restored.regions.size() == 2);
  PHX_CHECK_NEAR(phx::mc::test::region_volume(restored, 0), 2.0 / 6.0, kSmallCavityVolumeTolerance);
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
  PHX_CHECK_NEAR(phx::mc::test::region_volume(removed, 0), 1.0, kSmallCavityVolumeTolerance);
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
  PHX_CHECK_NEAR(phx::mc::test::region_volume(restored, 0), 1.0 / 6.0, kSmallCavityVolumeTolerance);
  PHX_CHECK(handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}

void test_curve_collapse_work_depends_only_on_incident_segments() {
  const Domain component = right_tetrahedron();
  int64_t local_work = -1;
  for (int components : {1, 33}) {
    Domain domain;
    for (int c = 0; c < components; ++c) {
      for (std::size_t i = 0; i < component.points.size(); ++i)
        domain.points.push_back(component.points[i] + (i % 3 == 0 ? 3.0 * c : 0.0));
      for (int32_t v : component.tets) domain.tets.push_back(v + 4 * c);
      domain.regions.push_back(c);
      for (int32_t v : component.faces) domain.faces.push_back(v + 4 * c);
      for (int32_t source : component.face_sources) domain.face_sources.push_back(source + 4 * c);
      for (int32_t v : component.segments) domain.segments.push_back(v + 4 * c);
      for (int32_t source : component.segment_sources)
        domain.segment_sources.push_back(source + 6 * c);
    }
    phx_mc_tet_mesh* handle = phx::mc::test::create_mesh(
        domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING, nullptr, 256, 2048);
    PHX_CHECK(handle != nullptr);
    if (handle == nullptr) continue;
    const Exported before = phx::mc::test::export_mesh(handle);
    const double midpoint[3] = {0.5, 0.0, 0.0};
    int32_t inserted = -1;
    PHX_CHECK(phx::mc::try_split_edge(*handle->mesh, 0, 1, midpoint, 0.0, inserted) ==
              phx::mc::Insertion::kOk);
    const int64_t start = handle->mesh->work();
    PHX_CHECK(phx::mc::try_collapse_edge(*handle->mesh, inserted, 0, false));
    const int64_t work = handle->mesh->work() - start;
    if (local_work < 0) local_work = work;
    PHX_CHECK(work == local_work);
    const Exported after = phx::mc::test::export_mesh(handle);
    PHX_CHECK(after.tets == before.tets && after.regions == before.regions);
    PHX_CHECK(after.faces == before.faces && after.face_sources == before.face_sources);
    PHX_CHECK(after.segments == before.segments && after.segment_sources == before.segment_sources);
    PHX_CHECK(std::equal(before.points.begin(), before.points.end(), after.points.begin()));
    PHX_CHECK(phx::mc::test::all_positive(after) && handle_audit(handle));
    phx_mc_tet_mesh_free(handle);
  }
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
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), 1.0 / 6.0, kSmallCavityVolumeTolerance);
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
  uint64_t memory[PHX_MC_MEMORY_VALUES] = {};
  PHX_CHECK(phx_mc_tet_mesh_memory_evidence(handle, memory) == PHX_MC_OK);
  PHX_CHECK(phx_mc_tet_mesh_set_memory_limit(handle, static_cast<int64_t>(memory[PHX_MC_MEMORY_LIVE_BYTES])) ==
            PHX_MC_OK);
  PHX_CHECK(phx_mc_tet_mesh_improve(handle, 70.0, 0.0, 1, 1000000, counters) ==
            PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(phx_mc_tet_mesh_set_memory_limit(handle, 1 << 20) == PHX_MC_OK);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == before);
  PHX_CHECK(handle_audit(handle));
  PHX_CHECK(phx_mc_tet_mesh_improve(handle, 70.0, 0.0, 1, 1000000, counters) ==
            PHX_MC_REFINEMENT_LIMIT);
  const Exported after = phx::mc::test::export_mesh(handle);
  PHX_CHECK(counters[PHX_MC_TET_MESH_IMPROVE_RELOCATIONS] > 0);
  PHX_CHECK(after.points[3 * inserted] != midpoint[0]);
  PHX_CHECK(smallest_dihedral(after) > smallest_dihedral(before));
  PHX_CHECK(std::equal(original.points.begin(), original.points.end(), after.points.begin()));
  PHX_CHECK(after.dimension[inserted] == 1);
  PHX_CHECK(after.segments == before.segments);
  PHX_CHECK(after.segment_sources == before.segment_sources);
  PHX_CHECK(after.regions == before.regions);
  PHX_CHECK(source_areas(after) == source_areas(original));
  PHX_CHECK(phx::mc::test::all_positive(after));
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), 1.0 / 6.0, kSmallCavityVolumeTolerance);
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
  uint64_t memory[PHX_MC_MEMORY_VALUES] = {};
  PHX_CHECK(phx_mc_tet_mesh_memory_evidence(handle, memory) == PHX_MC_OK);
  PHX_CHECK(phx_mc_tet_mesh_set_memory_limit(handle, static_cast<int64_t>(memory[PHX_MC_MEMORY_LIVE_BYTES])) ==
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
  phx::mc::TetMeshExudeCounters counters;
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
    phx::mc::TetMeshExudeCounters counters;
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
  options.minimum_relative_determinant = kReconstructionFloor;
  options.max_passes = 2;
  phx::mc::NativeVector<double> weights;
  phx::mc::TetMeshExudeCounters counters;
  PHX_CHECK(phx::mc::exude_mesh(*handle->mesh, options, weights, counters) ==
            PHX_MC_REFINEMENT_LIMIT);
  PHX_CHECK(counters.flips == 0 && counters.weighted_vertices == 0);
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
  radii[27] = 1e-4;  // Positive protection policy for this interior site, not oracle slack.
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

void test_coupled_relocation_cannot_enter_another_vertices_protecting_ball() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(3, 20, 1), 0.0);
  auto* unprotected =
      phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  PHX_CHECK(unprotected != nullptr);
  if (unprotected == nullptr) return;
  constexpr int32_t center = 13;
  int32_t moving = -1;
  double shortest = std::numeric_limits<double>::infinity();
  for (int32_t vertex = 27; vertex < 47; ++vertex) {
    const double distance = phx::mc::geometry::squared_distance(
        unprotected->mesh->point(center), unprotected->mesh->point(vertex));
    if (distance < shortest) {
      shortest = distance;
      moving = vertex;
    }
  }
  PHX_CHECK(moving >= 0);
  double position[3];
  for (int axis = 0; axis < 3; ++axis) {
    const double origin = unprotected->mesh->point(moving)[axis];
    position[axis] = origin + 0x1p-20 * (unprotected->mesh->point(center)[axis] - origin);
  }
  const double next_distance = std::sqrt(phx::mc::geometry::squared_distance(
      unprotected->mesh->point(center), position));
  std::vector<double> radii(domain.points.size() / 3, 0.0);
  radii[center] = 0.5 * (std::sqrt(shortest) + next_distance);
  auto* protected_mesh = phx::mc::test::create_mesh(
      domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING, radii.data());
  PHX_CHECK(protected_mesh != nullptr);
  if (protected_mesh == nullptr) {
    phx_mc_tet_mesh_free(unprotected);
    return;
  }
  const std::span<const int32_t> vertices(&moving, 1);
  PHX_CHECK(phx::mc::try_relocate_vertices(*unprotected->mesh, vertices, position, 0.0, 0.0));
  const Exported before = phx::mc::test::export_mesh(protected_mesh);
  const int64_t generation = protected_mesh->mesh->accepted_generation();
  PHX_CHECK(!phx::mc::try_relocate_vertices(*protected_mesh->mesh, vertices, position, 0.0, 0.0));
  PHX_CHECK(phx::mc::test::export_mesh(protected_mesh) == before);
  PHX_CHECK(protected_mesh->mesh->accepted_generation() == generation);
  PHX_CHECK(handle_audit(protected_mesh));
  phx_mc_tet_mesh_free(unprotected);
  phx_mc_tet_mesh_free(protected_mesh);
}

void test_expanded_insertion_inspection_is_atomic_and_epoch_bound() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(3, 20, 1), 0.0);
  phx_mc_tet_mesh* handle =
      phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  PHX_CHECK(handle != nullptr);
  const Exported original = phx::mc::test::export_mesh(handle);
  constexpr int64_t limit = 512;
  std::vector<int32_t> removed(4 * limit), proposed(4 * limit);
  int64_t counts[PHX_MC_EDGE_INSERTION_COUNTS] = {};
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
    if (counts[PHX_MC_EDGE_INSERTION_KIND] == 0 && counts[PHX_MC_EDGE_INSERTION_REMOVED] > star) {
      seed = edge;
      break;
    }
  }
  PHX_CHECK(seed[0] >= 0);
  PHX_CHECK(counts[PHX_MC_EDGE_INSERTION_REMOVED] > 0 && counts[PHX_MC_EDGE_INSERTION_PROPOSED] > 0 && counts[PHX_MC_EDGE_INSERTION_REMOVED] + counts[PHX_MC_EDGE_INSERTION_PROPOSED] <= limit);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == original);
  const std::array<int64_t, 4> inspected{counts[PHX_MC_EDGE_INSERTION_REMOVED], counts[PHX_MC_EDGE_INSERTION_PROPOSED], counts[PHX_MC_EDGE_INSERTION_KIND], counts[PHX_MC_EDGE_INSERTION_GENERATION]};
  std::vector<int32_t> mismatch = proposed;
  std::swap(mismatch[0], mismatch[1]);
  int32_t inserted = INT32_MAX;
  const phx_mc_tet_mesh_commit_edge_insertion_request mismatch_request{
      .mesh = handle,
      .first = seed[0],
      .second = seed[1],
      .split_position = position,
      .final_position = position,
      .source_fraction = 0.0,
      .maximum_cavity_cells = limit,
      .insertion_kind = static_cast<int32_t>(inspected[2]),
      .accepted_generation = inspected[3],
      .removed_count = inspected[0],
      .removed_tetrahedra = removed.data(),
      .proposed_count = inspected[1],
      .proposed_tetrahedra = mismatch.data(),
      .target_size = 0.0,
      .work_limit = 1000000,
      .inserted_vertex = &inserted,
  };
  PHX_CHECK(phx_mc_tet_mesh_commit_edge_insertion(&mismatch_request) == PHX_MC_OK);
  PHX_CHECK(inserted == -1 && phx::mc::test::export_mesh(handle) == original);
  PHX_CHECK(handle->mesh->accepted_generation() == inspected[3]);
  auto kind_mismatch_request = mismatch_request;
  kind_mismatch_request.proposed_tetrahedra = proposed.data();
  kind_mismatch_request.insertion_kind = (kind_mismatch_request.insertion_kind + 1) % 3;
  inserted = INT32_MAX;
  PHX_CHECK(phx_mc_tet_mesh_commit_edge_insertion(&kind_mismatch_request) == PHX_MC_OK);
  PHX_CHECK(inserted == -1 && phx::mc::test::export_mesh(handle) == original);
  PHX_CHECK(handle->mesh->accepted_generation() == inspected[3] && handle_audit(handle));
  PHX_CHECK(inspected[0] > 1);
  auto count_mismatch_request = kind_mismatch_request;
  count_mismatch_request.insertion_kind = static_cast<int32_t>(inspected[2]);
  --count_mismatch_request.removed_count;
  inserted = INT32_MAX;
  PHX_CHECK(phx_mc_tet_mesh_commit_edge_insertion(&count_mismatch_request) == PHX_MC_OK);
  PHX_CHECK(inserted == -1 && phx::mc::test::export_mesh(handle) == original);
  PHX_CHECK(handle->mesh->accepted_generation() == inspected[3] && handle_audit(handle));
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
  const std::array<int64_t, 4> fresh{counts[PHX_MC_EDGE_INSERTION_REMOVED], counts[PHX_MC_EDGE_INSERTION_PROPOSED], counts[PHX_MC_EDGE_INSERTION_KIND], counts[PHX_MC_EDGE_INSERTION_GENERATION]};
  const phx_mc_tet_mesh_commit_edge_insertion_request work_limit_request{
      .mesh = handle,
      .first = seed[0],
      .second = seed[1],
      .split_position = position,
      .final_position = position,
      .source_fraction = 0.0,
      .maximum_cavity_cells = limit,
      .insertion_kind = static_cast<int32_t>(fresh[2]),
      .accepted_generation = fresh[3],
      .removed_count = fresh[0],
      .removed_tetrahedra = removed.data(),
      .proposed_count = fresh[1],
      .proposed_tetrahedra = proposed.data(),
      .target_size = 0.0,
      .work_limit = 0,
      .inserted_vertex = &inserted,
  };
  inserted = INT32_MAX;
  PHX_CHECK(phx_mc_tet_mesh_commit_edge_insertion(&work_limit_request) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(inserted == -1 && phx::mc::test::export_mesh(handle) == original);
  double moved[3];
  std::copy_n(handle->mesh->point(27), 3, moved);
  moved[0] += 0x1p-20;
  handle->mesh->set_work_limit(1000000);
  int32_t applied = 0;
  PHX_CHECK(phx_mc_tet_mesh_relocate(handle, 27, moved, 0, &applied) == PHX_MC_OK && applied == 1);
  const Exported relocated = phx::mc::test::export_mesh(handle);
  const phx_mc_tet_mesh_commit_edge_insertion_request stale_request{
      .mesh = handle,
      .first = seed[0],
      .second = seed[1],
      .split_position = position,
      .final_position = position,
      .source_fraction = 0.0,
      .maximum_cavity_cells = limit,
      .insertion_kind = static_cast<int32_t>(fresh[2]),
      .accepted_generation = fresh[3],
      .removed_count = fresh[0],
      .removed_tetrahedra = removed.data(),
      .proposed_count = fresh[1],
      .proposed_tetrahedra = proposed.data(),
      .target_size = 0.0,
      .work_limit = 1000000,
      .inserted_vertex = &inserted,
  };
  inserted = INT32_MAX;
  PHX_CHECK(phx_mc_tet_mesh_commit_edge_insertion(&stale_request) == PHX_MC_OK);
  PHX_CHECK(inserted == -1 && phx::mc::test::export_mesh(handle) == relocated);
  const int64_t stale_work = handle->mesh->work();
  const int64_t relocated_generation = handle->mesh->accepted_generation();
  auto starved_stale_request = stale_request;
  starved_stale_request.work_limit = 0;
  inserted = INT32_MAX;
  PHX_CHECK(phx_mc_tet_mesh_commit_edge_insertion(&starved_stale_request) == PHX_MC_OK);
  PHX_CHECK(inserted == -1 && phx::mc::test::export_mesh(handle) == relocated);
  PHX_CHECK(handle->mesh->accepted_generation() == relocated_generation);
  PHX_CHECK(handle->mesh->work() == stale_work && handle_audit(handle));
  handle->mesh->set_work_limit(1000000);
  PHX_CHECK(phx_mc_tet_mesh_relocate(
                handle, 27, original.points.data() + 3 * 27, 0, &applied) == PHX_MC_OK);
  PHX_CHECK(phx_mc_tet_mesh_inspect_edge_insertion(
                handle, seed[0], seed[1], position, position, 0.0, limit, 1000000,
                removed.data(), proposed.data(), counts) == PHX_MC_OK);
  const phx_mc_tet_mesh_commit_edge_insertion_request commit_request{
      .mesh = handle,
      .first = seed[0],
      .second = seed[1],
      .split_position = position,
      .final_position = position,
      .source_fraction = 0.0,
      .maximum_cavity_cells = limit,
      .insertion_kind = static_cast<int32_t>(counts[PHX_MC_EDGE_INSERTION_KIND]),
      .accepted_generation = counts[PHX_MC_EDGE_INSERTION_GENERATION],
      .removed_count = counts[PHX_MC_EDGE_INSERTION_REMOVED],
      .removed_tetrahedra = removed.data(),
      .proposed_count = counts[PHX_MC_EDGE_INSERTION_PROPOSED],
      .proposed_tetrahedra = proposed.data(),
      .target_size = 0.0,
      .work_limit = 1000000,
      .inserted_vertex = &inserted,
  };
  PHX_CHECK(phx_mc_tet_mesh_commit_edge_insertion(&commit_request) == PHX_MC_OK);
  PHX_CHECK(inserted == static_cast<int32_t>(original.points.size() / 3));
  const Exported accepted = phx::mc::test::export_mesh(handle);
  PHX_CHECK(std::equal(original.points.begin(), original.points.end(), accepted.points.begin()));
  PHX_CHECK(accepted.regions.size() == original.regions.size() - counts[PHX_MC_EDGE_INSERTION_REMOVED] + counts[PHX_MC_EDGE_INSERTION_PROPOSED]);
  PHX_CHECK(accepted.faces == original.faces && accepted.segments == original.segments);
  PHX_CHECK(phx::mc::test::all_positive(accepted) && handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}

void test_coupled_relocation_is_atomic_at_the_exact_work_boundary() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(3, 2, 13), 0.0);
  auto* reference = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  auto* bounded = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  PHX_CHECK(reference != nullptr && bounded != nullptr);
  if (reference == nullptr || bounded == nullptr) {
    phx_mc_tet_mesh_free(reference);
    phx_mc_tet_mesh_free(bounded);
    return;
  }
  const Exported before = phx::mc::test::export_mesh(bounded);
  const int64_t generation = bounded->mesh->accepted_generation();
  const int32_t vertices[2] = {13, 27};
  double positions[6];
  for (int row = 0; row < 2; ++row) {
    std::copy_n(reference->mesh->point(vertices[row]), 3, positions + 3 * row);
    positions[3 * row] += (row == 0 ? 1.0 : -1.0) * 0x1p-20;
  }
  const int64_t start = reference->mesh->work();
  int32_t applied = -1;
  PHX_CHECK(phx_mc_tet_mesh_relocate_vertices(
      reference, 2, vertices, positions, 0.0, 0.0, 1000000, &applied) == PHX_MC_OK);
  PHX_CHECK(applied == 1);
  const int64_t required = reference->mesh->work() - start;
  PHX_CHECK(required > 0);
  PHX_CHECK(phx_mc_tet_mesh_relocate_vertices(
      bounded, 2, vertices, positions, 0.0, 0.0, required - 1, &applied) ==
      PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(applied == 0 && phx::mc::test::export_mesh(bounded) == before);
  PHX_CHECK(bounded->mesh->accepted_generation() == generation);
  PHX_CHECK(phx_mc_tet_mesh_relocate_vertices(
      bounded, 2, vertices, positions, 0.0, 0.0, required, &applied) == PHX_MC_OK);
  PHX_CHECK(applied == 1);
  const Exported after = phx::mc::test::export_mesh(bounded);
  PHX_CHECK(after == phx::mc::test::export_mesh(reference));
  PHX_CHECK(phx::mc::test::all_positive(after));
  PHX_CHECK(after.regions == before.regions && after.face_sources == before.face_sources);
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), 1.0, kUnitBoxMeasureTolerance);
  // Invalid joint trials cannot publish the earlier valid row.
  const int64_t accepted_generation = bounded->mesh->accepted_generation();
  positions[3] = std::numeric_limits<double>::quiet_NaN();
  PHX_CHECK(!phx::mc::try_relocate_vertices(*bounded->mesh, vertices, positions, 0.0, 0.0));
  PHX_CHECK(phx::mc::test::export_mesh(bounded) == after);
  PHX_CHECK(bounded->mesh->accepted_generation() == accepted_generation);
  PHX_CHECK(handle_audit(reference) && handle_audit(bounded));
  phx_mc_tet_mesh_free(reference);
  phx_mc_tet_mesh_free(bounded);
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
  int64_t counters[PHX_MC_TET_MESH_EXUDE_COUNTERS] = {};
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
  PHX_CHECK(counters[PHX_MC_TET_MESH_IMPROVE_REMAINING_SLIVERS] == 0);
  int64_t counts[PHX_MC_TET_MESH_COUNTS] = {};
  PHX_CHECK(phx_mc_tet_mesh_counts(handle, counts) == PHX_MC_OK && counts[PHX_MC_TET_MESH_COUNT_UNMET_RECORDS] == 1);
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
  PHX_CHECK(phx_mc_tet_mesh_improve(handle, 0.0, kReconstructionFloor, 1, 100000, counters) == PHX_MC_OK);
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

void test_determinant_floor_hinge_receives_certified_interior_vertex(const Domain& domain) {
  constexpr double kFloor = kReconstructionFloor;
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
  PHX_CHECK(counters[PHX_MC_TET_MESH_IMPROVE_VERTEX_INSERTIONS] == 1);
  const Exported after = phx::mc::test::export_mesh(handle);
  PHX_CHECK(std::equal(before.points.begin(), before.points.end(), after.points.begin()));
  PHX_CHECK(after.faces == before.faces && after.face_sources == before.face_sources &&
            after.segments == before.segments && after.segment_sources == before.segment_sources);
  PHX_CHECK(after.dimension.size() == before.dimension.size() + 1 && after.dimension.back() == 3);
  for (std::size_t offset = 0; offset < after.tets.size(); offset += 4) {
    PHX_CHECK(above_floor(after, offset));
  }
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), phx::mc::test::region_volume(before, 0), kHingeVolumeTolerance);
  PHX_CHECK(handle_audit(handle));
  phx_mc_tet_mesh_free(handle);

  // Without vertex allowance the repair is a capacity refusal, not a hidden
  // geometric failure, and the accepted state is unchanged.
  int32_t status = -1;
  phx_mc_tet_mesh* capped = phx::mc::test::create_mesh(
      domain, PHX_MC_TET_MESH_BOUNDARY_FIXED, nullptr,
      static_cast<int64_t>(domain.points.size() / 3), 1 << 10, &status);
  PHX_CHECK(status == PHX_MC_OK && capped != nullptr);
  if (capped == nullptr) return;
  PHX_CHECK(phx_mc_tet_mesh_improve(capped, 0.0, kFloor, 1, 100000, counters) ==
            PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(counters[PHX_MC_TET_MESH_IMPROVE_VERTEX_INSERTIONS] == 0);
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

void check_octahedron_cluster_rejections(phx::mc::TetMesh& mesh, int32_t cell) {
  std::vector<int32_t> cavity{cell};
  std::size_t begin = 0;
  for (int depth = 0; depth < 2; ++depth) {
    const std::size_t end = cavity.size();
    for (std::size_t cursor = begin; cursor < end; ++cursor) {
      for (int slot = 0; slot < 4; ++slot) {
        const int32_t other = mesh.tet(cavity[cursor]).n[slot];
        if (mesh.complex().constraint(cavity[cursor], slot) == phx::mc::kNoConstraint &&
            mesh.finite(other) && std::find(cavity.begin(), cavity.end(), other) == cavity.end()) {
          cavity.push_back(other);
        }
      }
    }
    begin = end;
  }
  // Literal uncached replay of the shared boundary/early losing-cone
  // calculation. Keep the exact edge-proof saving oracle below independent
  // of the now separately factorized multiface boundary work.
  PHX_CHECK(mesh.spend(static_cast<int64_t>(8 * cavity.size())));
  std::sort(cavity.begin(), cavity.end());
  std::vector<std::pair<std::array<int32_t, 4>, int>> boundary;
  double baseline = 180.0;
  for (int32_t entry : cavity) {
    const auto& tet = mesh.tet(entry);
    const double* corners[4];
    mesh.corners(tet.v, corners);
    baseline = std::min(baseline, phx::mc::geometry::minimum_dihedral(corners));
    for (int slot = 0; slot < 4; ++slot) {
      if (!std::binary_search(cavity.begin(), cavity.end(), tet.n[slot])) {
        boundary.push_back({{tet.v[0], tet.v[1], tet.v[2], tet.v[3]}, slot});
      }
    }
  }
  for (int32_t apex = 0; apex < 6; ++apex) {
    bool rejected = false;
    for (const auto& [vertices, slot] : boundary) {
      PHX_CHECK(mesh.spend(1));
      if (std::find(vertices.begin(), vertices.end(), apex) != vertices.end() &&
          vertices[slot] != apex) continue;
      auto cone = vertices;
      cone[slot] = apex;
      const double* corners[4];
      mesh.corners(cone.data(), corners);
      if (!mesh.positive(cone.data()) ||
          !(phx::mc::geometry::minimum_dihedral(corners) > baseline + 1e-9)) {
        rejected = true;
        break;
      }
    }
    PHX_CHECK(rejected);
  }
}

void test_unchanged_edge_rejections_reuse_geometric_proofs() {
  // The regular octahedron's three long diagonals give congruent cells.
  // No strictly improving reconnection exists; all exterior vertices are
  // fixed source corners. Four slivers therefore ask for the same failed
  // directed interior-edge proposal without any accepted-state change.
  Domain domain;
  domain.points = {0, 0, -1, 0, 0, 1, 1, 0, 0,
                   0, 1, 0, -1, 0, 0, 0, -1, 0};
  for (int32_t i = 0; i < 4; ++i) {
    const int32_t a = 2 + i, b = 2 + (i + 1) % 4;
    domain.tets.insert(domain.tets.end(), {0, 1, a, b});
    domain.regions.push_back(0);
    for (int32_t apex : {0, 1}) {
      domain.faces.insert(domain.faces.end(), {apex, a, b});
      domain.face_sources.push_back(static_cast<int32_t>(domain.face_sources.size()));
      domain.segments.insert(domain.segments.end(), {apex, a});
      domain.segment_sources.push_back(static_cast<int32_t>(domain.segment_sources.size()));
    }
    domain.segments.insert(domain.segments.end(), {a, b});
    domain.segment_sources.push_back(static_cast<int32_t>(domain.segment_sources.size()));
  }
  phx_mc_tet_mesh* reference = phx::mc::test::create_mesh(
      domain, PHX_MC_TET_MESH_BOUNDARY_FIXED);
  phx_mc_tet_mesh* scheduled = phx::mc::test::create_mesh(
      domain, PHX_MC_TET_MESH_BOUNDARY_FIXED);
  PHX_CHECK(reference != nullptr && scheduled != nullptr);
  if (reference == nullptr || scheduled == nullptr) {
    phx_mc_tet_mesh_free(reference);
    phx_mc_tet_mesh_free(scheduled);
    return;
  }
  const Exported before = phx::mc::test::export_mesh(reference);
  auto& mesh = *reference->mesh;
  const int64_t first = mesh.work();
  PHX_CHECK(!phx::mc::try_edge_removal(mesh, 0, 1, true));
  const int64_t edge_work = mesh.work() - first;
  PHX_CHECK(edge_work > 0);
  const int64_t reference_start = mesh.work();
  // Replay the uncached operation traversal. Ordinary vertex removal and
  // smoothing reject these fixed dimension-zero corners before any work.
  for (int32_t t = 0; t < 4; ++t) {
    PHX_CHECK(mesh.spend(1));  // Ordinary-sliver removal visit.
    for (int32_t vertex : mesh.tet(t).v) PHX_CHECK(mesh.dimension(vertex) == 0);
  }
  for (int32_t t = 0; t < 4; ++t) {
    PHX_CHECK(mesh.spend(1));  // Reconnection-frontier visit.
    for (int slot = 0; slot < 4; ++slot) {
      PHX_CHECK(!phx::mc::try_face_removal(mesh, t, slot, true));
    }
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        PHX_CHECK(!phx::mc::try_edge_removal(
            mesh, mesh.tet(t).v[i], mesh.tet(t).v[j], true));
      }
    }
    check_octahedron_cluster_rejections(mesh, t);
    for (int32_t vertex : mesh.tet(t).v) {
      PHX_CHECK(!phx::mc::try_remove_vertex(mesh, vertex, true, nullptr));
    }
  }
  const int64_t reference_work = mesh.work() - reference_start;
  PHX_CHECK(phx::mc::test::export_mesh(reference) == before);
  const int64_t generation = scheduled->mesh->accepted_generation();
  int64_t counters[PHX_MC_TET_MESH_IMPROVE_COUNTERS] = {};
  PHX_CHECK(phx_mc_tet_mesh_improve(scheduled, 70.0, 0.0, 1, 1000000, counters) ==
            PHX_MC_REFINEMENT_LIMIT);
  PHX_CHECK(scheduled->mesh->accepted_generation() == generation);
  PHX_CHECK(phx::mc::test::export_mesh(scheduled) == before);
  PHX_CHECK(counters[PHX_MC_TET_MESH_IMPROVE_REMAINING_SLIVERS] == 4);
  PHX_CHECK(reference_work - counters[PHX_MC_TET_MESH_IMPROVE_WORK_UNITS] == 3 * edge_work);
  PHX_CHECK(handle_audit(reference) && handle_audit(scheduled));
  phx_mc_tet_mesh_free(reference);
  phx_mc_tet_mesh_free(scheduled);
}

void test_completed_negative_vertex_searches_fit_reduced_work() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(3, 20, 21), 0.0);
  auto* reference = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_FIXED);
  auto* bounded = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_FIXED);
  PHX_CHECK(reference != nullptr && bounded != nullptr);
  if (reference == nullptr || bounded == nullptr) {
    phx_mc_tet_mesh_free(reference);
    phx_mc_tet_mesh_free(bounded);
    return;
  }
  // The fixed corpus charged 359643 units without vertex-proof reuse.
  // Its completed unchanged-state shortlists earn this smaller envelope.
  constexpr int64_t kCompletedSearchWork = 324569;
  int64_t reference_counters[PHX_MC_TET_MESH_IMPROVE_COUNTERS] = {};
  int64_t bounded_counters[PHX_MC_TET_MESH_IMPROVE_COUNTERS] = {};
  const int32_t reference_status = phx_mc_tet_mesh_improve(
      reference, 70.0, 0.0, 3, int64_t{1} << 40, reference_counters);
  const int32_t bounded_status = phx_mc_tet_mesh_improve(
      bounded, 70.0, 0.0, 3, kCompletedSearchWork, bounded_counters);
  PHX_CHECK(reference_status == PHX_MC_REFINEMENT_LIMIT);
  PHX_CHECK(bounded_status == reference_status);
  PHX_CHECK(phx::mc::test::export_mesh(bounded) == phx::mc::test::export_mesh(reference));
  PHX_CHECK(std::equal(reference_counters, reference_counters + PHX_MC_TET_MESH_IMPROVE_COUNTERS,
                      bounded_counters));
  PHX_CHECK(reference_counters[PHX_MC_TET_MESH_IMPROVE_WORK_UNITS] <= kCompletedSearchWork);
  PHX_CHECK(handle_audit(reference) && handle_audit(bounded));
  phx_mc_tet_mesh_free(reference);
  phx_mc_tet_mesh_free(bounded);
}

// ------------------------------------------------- recomputing exudation replay
// The weighted exudation before per-visit reuse, replayed literally through
// public operations with its charges: every trial weight re-ran each star
// cell's 2-3 trial (16 units, power test, size/floor admission, flip) and
// each neighbor's weighted edge removal (its own star walk, ring and ring
// triangulation), and every sliver vertex was revisited from scratch.

bool replay_size_targets(const phx::mc::TetMesh& mesh, const std::array<int32_t, 4>& cell,
                         double ratio_bound) {
  const double* p[4];
  mesh.corners(cell.data(), p);
  double center[3];
  if (!phx::mc::geometry::tetrahedron_circumcenter(p[0], p[1], p[2], p[3], center)) {
    return false;
  }
  const double radius2 = phx::mc::geometry::squared_distance(center, p[0]);
  double shortest2 = std::numeric_limits<double>::infinity();
  double target = 0.0;
  for (int i = 0; i < 4; ++i) {
    target += 0.25 * mesh.size(cell[static_cast<std::size_t>(i)]);
    for (int j = i + 1; j < 4; ++j) {
      shortest2 = std::min(shortest2, phx::mc::geometry::squared_distance(p[i], p[j]));
    }
  }
  return std::isfinite(radius2) && radius2 <= ratio_bound * ratio_bound * shortest2 &&
         (target == 0.0 || radius2 <= target * target);
}

bool replay_above_floor(const phx::mc::TetMesh& mesh, const std::array<int32_t, 4>& cell,
                        double floor) {
  if (!(floor > 0.0)) return true;
  const double* corners[4];
  mesh.corners(cell.data(), corners);
  double score = 0.0;
  return phx::mc::canonical_relative_floor(cell.data(), corners, floor, &score) > 0;
}

struct ReplaySliver {
  double quality;
  int32_t slot;
  std::array<int32_t, 4> v;
};

// Cells below the angle target, with an uncertain canonical-chart
// orientation, or at/below the floor; ordered as exudation schedules them.
std::vector<ReplaySliver> replay_slivers(const phx::mc::TetMesh& mesh, double target,
                                         double floor) {
  std::vector<ReplaySliver> slivers;
  const auto slots = static_cast<int32_t>(mesh.complex().tets.size());
  for (int32_t t = 0; t < slots; ++t) {
    if (!mesh.complex().live(t) || !mesh.finite(t)) continue;
    const int32_t* v = mesh.tet(t).v;
    const double* corners[4];
    mesh.corners(v, corners);
    const double q = phx::mc::geometry::minimum_dihedral(corners);
    int perm[4];
    phx::mc::canonical_permutation(v, 4, perm);
    double axes[4][3];
    for (int corner = 0; corner < 4; ++corner) {
      axes[corner][0] = corners[perm[corner]][1];
      axes[corner][1] = corners[perm[corner]][2];
      axes[corner][2] = corners[perm[corner]][0];
    }
    bool uncertain =
        (-phx::mc::orient3d_approx(axes[1], axes[2], axes[3], axes[0])).certified_sign() != 1;
    if (floor > 0.0) {
      double score = 0.0;
      uncertain = uncertain || phx::mc::canonical_relative_floor(v, corners, floor, &score) <= 0;
    }
    if (uncertain || !std::isfinite(q) || q < target) {
      ReplaySliver entry{q, t, {v[0], v[1], v[2], v[3]}};
      std::sort(entry.v.begin(), entry.v.end());
      slivers.push_back(entry);
    }
  }
  std::sort(slivers.begin(), slivers.end(), [](const ReplaySliver& a, const ReplaySliver& b) {
    const bool finite = std::isfinite(a.quality), other_finite = std::isfinite(b.quality);
    if (finite != other_finite) return !finite;
    if (finite && a.quality != b.quality) return a.quality < b.quality;
    return std::lexicographical_compare(a.v.begin(), a.v.end(), b.v.begin(), b.v.end());
  });
  return slivers;
}

bool replay_current(const phx::mc::TetMesh& mesh, const ReplaySliver& entry) {
  if (!mesh.complex().live(entry.slot) || !mesh.finite(entry.slot)) return false;
  std::array<int32_t, 4> v = {mesh.tet(entry.slot).v[0], mesh.tet(entry.slot).v[1],
                              mesh.tet(entry.slot).v[2], mesh.tet(entry.slot).v[3]};
  std::sort(v.begin(), v.end());
  return v == entry.v;
}

struct ExudeReplay {
  int64_t trials = 0;
  int64_t flips = 0;
  int64_t passes = 0;
};

bool replay_exude_vertex(phx::mc::TetMesh& mesh, int32_t vertex,
                         const phx::mc::ExudeOptions& options, std::vector<double>& weights,
                         ExudeReplay& replay) {
  if (mesh.dimension(vertex) != 3 || mesh.protection(vertex) > 0.0) return false;
  std::vector<int32_t> star;
  if (!mesh.vertex_star(vertex, star)) return false;
  const auto weight = [&](int32_t v) { return weights[static_cast<std::size_t>(v)]; };
  double shortest2 = std::numeric_limits<double>::infinity();
  for (int32_t t : star) {
    for (int32_t v : mesh.tet(t).v) {
      if (v != vertex && v >= 0) {
        shortest2 = std::min(
            shortest2, phx::mc::geometry::squared_distance(mesh.point(vertex), mesh.point(v)));
      }
    }
  }
  const double bound = options.max_weight_fraction * shortest2;
  for (int exponent = -12; exponent <= 0; ++exponent) {
    const double trial = std::ldexp(bound, exponent);
    if (!(trial > weight(vertex)) || !phx::mc::weight_in_domain(trial)) continue;
    for (int32_t t : star) {
      if (!mesh.spend(16)) return false;
      ++replay.trials;
      const phx::mc::Tetrahedron old = mesh.tet(t);
      const int slot = phx::mc::vertex_slot(old, vertex);
      const int32_t other = old.n[slot];
      if (!mesh.finite(other) || mesh.complex().constraint(t, slot) != phx::mc::kNoConstraint) {
        continue;
      }
      const phx::mc::Tetrahedron neighbor = mesh.tet(other);
      const int back = phx::mc::neighbor_slot(neighbor, t);
      const double* p[4];
      mesh.corners(other, p);
      if (phx::mc::power3d(p[0], p[1], p[2], p[3], mesh.point(vertex), weight(neighbor.v[0]),
                           weight(neighbor.v[1]), weight(neighbor.v[2]), weight(neighbor.v[3]),
                           trial) <= 0) {
        continue;
      }
      const double new_edge_bound =
          options.max_weight_fraction *
          phx::mc::geometry::squared_distance(mesh.point(vertex), mesh.point(neighbor.v[back]));
      bool size_valid = trial <= new_edge_bound && weight(neighbor.v[back]) <= new_edge_bound;
      for (int k = 0; k < 4; ++k) {
        if (k == back) continue;
        std::array<int32_t, 4> proposed = {neighbor.v[0], neighbor.v[1], neighbor.v[2],
                                           neighbor.v[3]};
        proposed[static_cast<std::size_t>(k)] = vertex;
        size_valid = size_valid && replay_size_targets(mesh, proposed, options.radius_edge_bound) &&
                     replay_above_floor(mesh, proposed, options.minimum_relative_determinant);
      }
      if (size_valid && phx::mc::try_face_removal(mesh, t, slot, true)) {
        weights[static_cast<std::size_t>(vertex)] = trial;
        ++replay.flips;
        return true;
      }
    }
    std::vector<int32_t> neighbors;
    for (int32_t t : star) {
      for (int32_t v : mesh.tet(t).v) {
        if (v != vertex && v >= 0) neighbors.push_back(v);
      }
    }
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    const double saved_weight = weight(vertex);
    weights[static_cast<std::size_t>(vertex)] = trial;
    for (int32_t other : neighbors) {
      ++replay.trials;
      if (phx::mc::try_edge_removal(mesh, vertex, other, true, std::span<const double>(weights),
                                    options.radius_edge_bound, options.max_weight_fraction, false,
                                    options.minimum_relative_determinant)) {
        ++replay.flips;
        return true;
      }
    }
    weights[static_cast<std::size_t>(vertex)] = saved_weight;
  }
  return false;
}

ExudeReplay replay_exude(phx::mc::TetMesh& mesh, const phx::mc::ExudeOptions& options,
                         std::vector<double>& weights) {
  ExudeReplay replay;
  weights.assign(static_cast<std::size_t>(mesh.vertex_count()), 0.0);
  for (int32_t pass = 0; pass < options.max_passes; ++pass) {
    const std::vector<ReplaySliver> slivers = replay_slivers(
        mesh, options.min_dihedral_degrees, options.minimum_relative_determinant);
    if (slivers.empty()) break;
    ++replay.passes;
    bool progress = false;
    for (const ReplaySliver& entry : slivers) {
      if (!replay_current(mesh, entry)) continue;
      const std::array<int32_t, 4> vertices = {mesh.tet(entry.slot).v[0], mesh.tet(entry.slot).v[1],
                                               mesh.tet(entry.slot).v[2], mesh.tet(entry.slot).v[3]};
      for (int32_t vertex : vertices) {
        if (replay_exude_vertex(mesh, vertex, options, weights, replay)) {
          progress = true;
          break;
        }
      }
      if (mesh.broken() || mesh.work_exhausted()) break;
    }
    if (!progress || mesh.broken() || mesh.work_exhausted()) break;
  }
  return replay;
}

// Exudation with per-visit reuse against the recomputing replay on an
// identical mesh: identical flips, weights, trials, passes and output mesh,
// with strictly fewer charged work units. Returns the accepted flips.
int64_t check_exudation_matches_replay(const Domain& domain, const phx::mc::ExudeOptions& options,
                                       bool force_sliver) {
  phx_mc_tet_mesh* replayed =
      phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  phx_mc_tet_mesh* reused =
      phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  PHX_CHECK(replayed != nullptr && reused != nullptr);
  if (replayed == nullptr || reused == nullptr) {
    phx_mc_tet_mesh_free(replayed);
    phx_mc_tet_mesh_free(reused);
    return -1;
  }
  if (force_sliver) {
    // The 2-3 flip of the interior face (1, 28, 46) creates the sliver that
    // a weighted 3-2 restoration removes again.
    for (phx_mc_tet_mesh* handle : {replayed, reused}) {
      int32_t t = -1;
      int slot = -1;
      PHX_CHECK(handle->mesh->find_face(1, 28, 46, t, slot));
      PHX_CHECK(phx::mc::try_face_removal(*handle->mesh, t, slot, false));
    }
  }
  PHX_CHECK(phx::mc::test::export_mesh(replayed) == phx::mc::test::export_mesh(reused));
  phx::mc::TetMesh& reference = *replayed->mesh;
  const int64_t replay_start = reference.work();
  std::vector<double> replay_weights;
  const ExudeReplay replay = replay_exude(reference, options, replay_weights);
  const int64_t replay_work = reference.work() - replay_start;
  PHX_CHECK(!reference.work_exhausted() && !reference.broken());
  phx::mc::NativeVector<double> weights;
  phx::mc::TetMeshExudeCounters counters;
  const int32_t status = phx::mc::exude_mesh(*reused->mesh, options, weights, counters);
  PHX_CHECK(status == PHX_MC_OK || status == PHX_MC_REFINEMENT_LIMIT);
  PHX_CHECK(phx::mc::test::export_mesh(reused) == phx::mc::test::export_mesh(replayed));
  PHX_CHECK(std::equal(weights.begin(), weights.end(), replay_weights.begin(),
                       replay_weights.end()));
  PHX_CHECK(counters.flips == replay.flips);
  PHX_CHECK(counters.trials == replay.trials);
  PHX_CHECK(counters.passes == replay.passes);
  PHX_CHECK(counters.trials > 0);
  PHX_CHECK(counters.work_units > 0 && counters.work_units < replay_work);
  PHX_CHECK(handle_audit(replayed) && handle_audit(reused));
  phx_mc_tet_mesh_free(replayed);
  phx_mc_tet_mesh_free(reused);
  return counters.flips;
}

void test_exudation_reuse_matches_recomputing_replay_on_forced_sliver() {
  phx::mc::ExudeOptions options;
  options.min_dihedral_degrees = 20.0;
  options.max_weight_fraction = 0.25;
  options.max_passes = 2;
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(3, 20, 1), 0.0);
  PHX_CHECK(check_exudation_matches_replay(domain, options, true) > 0);
}

void test_exudation_reuse_matches_recomputing_replay_on_sliver_rich_box() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(3, 60, 13), 0.0);
  for (const double floor : {0.0, 1e-2}) {
    phx::mc::ExudeOptions options;
    options.min_dihedral_degrees = 20.0;
    options.max_weight_fraction = 0.25;
    options.minimum_relative_determinant = floor;
    options.max_passes = 3;
    PHX_CHECK(check_exudation_matches_replay(domain, options, false) >= 0);
  }
}

}  // namespace

int main() {
  test_exudation_reuse_matches_recomputing_replay_on_forced_sliver();
  test_exudation_reuse_matches_recomputing_replay_on_sliver_rich_box();
  test_completed_negative_vertex_searches_fit_reduced_work();
  test_unchanged_edge_rejections_reuse_geometric_proofs();
  test_determinant_floor_hinge_receives_certified_interior_vertex(
      phx::mc::test::reconstruction_hinge_domain());
  test_determinant_floor_hinge_receives_certified_interior_vertex(
      phx::mc::test::reconstruction_narrow_hinge_domain());
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
  test_curve_collapse_work_depends_only_on_incident_segments();
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
  test_coupled_relocation_is_atomic_at_the_exact_work_boundary();
  test_coupled_relocation_cannot_enter_another_vertices_protecting_ball();
  test_exude_allocation_refusal_publishes_actual_weight_state();
  return phx::mc::test::finish("test_improve3d");
}
