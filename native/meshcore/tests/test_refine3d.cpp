//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Constrained Delaunay refinement: radius-edge and size targets on box
// domains, exact preservation of constrained faces and region volumes,
// fixed-boundary and protecting-ball evidence, budgets, determinism and
// input validation.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <utility>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"
#include "tet_mesh.hpp"
#include "tet_mesh_fixtures.hpp"

namespace {

using phx::mc::test::Domain;
using phx::mc::test::Exported;

struct Run {
  int32_t status = -1;
  int64_t counters[PHX_MC_TET_MESH_REFINE_COUNTERS] = {};
};

Run refine(phx_mc_tet_mesh* mesh, double bound, int64_t insertions = 1 << 20,
           const double* sizes = nullptr, int64_t work = int64_t{1} << 40) {
  Run run;
  run.status = phx_mc_tet_mesh_refine(mesh, sizes, bound, insertions, work, run.counters);
  return run;
}

double worst_ratio(const Exported& mesh) {
  double worst = 0.0;
  for (std::size_t t = 0; t < mesh.regions.size(); ++t) {
    double ratio = 0.0;
    double dihedral = 0.0;
    phx::mc::test::cell_quality(mesh, t, ratio, dihedral);
    worst = std::max(worst, ratio);
  }
  return worst;
}

std::vector<int32_t> unmet_reasons(const phx_mc_tet_mesh* mesh) {
  int64_t counts[PHX_MC_TET_MESH_COUNTS];
  phx_mc_tet_mesh_counts(mesh, counts);
  const auto n = static_cast<std::size_t>(counts[4]);
  std::vector<int32_t> tets(4 * n);
  std::vector<int32_t> codes(2 * n);
  std::vector<double> values(n);
  phx_mc_tet_mesh_unmet(mesh, tets.data(), codes.data(), values.data());
  std::vector<int32_t> reasons;
  for (std::size_t i = 0; i < n; ++i) {
    reasons.push_back(codes[2 * i + 1]);
  }
  return reasons;
}

void test_cube_point_set_reaches_radius_edge_bound() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(3, 40, 7), 0.0);
  phx_mc_tet_mesh* mesh = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  PHX_CHECK(mesh != nullptr);
  const Exported before = phx::mc::test::export_mesh(mesh);
  PHX_CHECK(worst_ratio(before) > 2.0);
  const Run run = refine(mesh, 2.0);
  PHX_CHECK(run.status == PHX_MC_OK);
  PHX_CHECK(run.counters[0] > 0);
  const Exported after = phx::mc::test::export_mesh(mesh);
  PHX_CHECK(worst_ratio(after) <= 2.0 + 1e-9);
  PHX_CHECK(phx::mc::test::all_positive(after));
  PHX_CHECK(phx::mc::test::faces_on_sides(after, 0.0));
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), 1.0, 1e-12);
  for (int32_t side = 0; side < 6; ++side) {
    PHX_CHECK_NEAR(phx::mc::test::source_area(after, side), 1.0, 1e-12);
  }
  PHX_CHECK(handle_audit(mesh));
  std::printf("cube: vertices %zu -> %zu, cells %zu, ratio %.3f -> %.3f\n",
              before.dimension.size(), after.dimension.size(), after.regions.size(),
              worst_ratio(before), worst_ratio(after));
  phx_mc_tet_mesh_free(mesh);
}

void test_refinement_is_deterministic() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(3, 30, 11), 0.0);
  phx_mc_tet_mesh* first = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  phx_mc_tet_mesh* second = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  const Run a = refine(first, 1.8);
  const Run b = refine(second, 1.8);
  PHX_CHECK(a.status == b.status);
  PHX_CHECK(phx::mc::test::export_mesh(first) == phx::mc::test::export_mesh(second));
  phx_mc_tet_mesh_free(first);
  phx_mc_tet_mesh_free(second);
}

void test_two_region_box_preserves_interface_and_volumes() {
  // A lattice keeps every Delaunay cell inside one lattice cube, so the
  // plane x = 0.5 is a union of faces.
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(5, 0, 3), 0.5);
  phx_mc_tet_mesh* mesh = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  PHX_CHECK(mesh != nullptr);
  const Run run = refine(mesh, 2.0);
  PHX_CHECK(run.status == PHX_MC_OK);
  const Exported after = phx::mc::test::export_mesh(mesh);
  PHX_CHECK(worst_ratio(after) <= 2.0 + 1e-9);
  PHX_CHECK(phx::mc::test::faces_on_sides(after, 0.5));
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), 0.5, 1e-12);
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 1), 0.5, 1e-12);
  PHX_CHECK_NEAR(phx::mc::test::source_area(after, 6), 1.0, 1e-12);
  // Every cell lies on its region's side of the interface.
  bool sided = true;
  for (std::size_t t = 0; t < after.regions.size(); ++t) {
    for (int k = 0; k < 4; ++k) {
      const double x = phx::mc::test::at(after, after.tets[4 * t + k])[0];
      sided = sided && (after.regions[t] == 0 ? x <= 0.5 : x >= 0.5);
    }
  }
  PHX_CHECK(sided);
  phx_mc_tet_mesh_free(mesh);
}

void test_size_field_bounds_circumradius() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(3, 0, 1), 0.0);
  phx_mc_tet_mesh* mesh = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  const std::vector<double> sizes(domain.points.size() / 3, 0.2);
  const Run run = refine(mesh, 2.0, 1 << 20, sizes.data());
  PHX_CHECK(run.status == PHX_MC_OK);
  const Exported after = phx::mc::test::export_mesh(mesh);
  double largest = 0.0;
  for (std::size_t t = 0; t < after.regions.size(); ++t) {
    double ratio = 0.0;
    double dihedral = 0.0;
    phx::mc::test::cell_quality(after, t, ratio, dihedral);
    const int32_t* v = after.tets.data() + 4 * t;
    double shortest = 1e300;
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        double d = 0.0;
        for (int k = 0; k < 3; ++k) {
          const double delta = phx::mc::test::at(after, v[i])[k] - phx::mc::test::at(after, v[j])[k];
          d += delta * delta;
        }
        shortest = std::min(shortest, std::sqrt(d));
      }
    }
    largest = std::max(largest, ratio * shortest);
  }
  PHX_CHECK(largest <= 0.2 + 1e-12);
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), 1.0, 1e-12);
  phx_mc_tet_mesh_free(mesh);
}

void test_fixed_boundary_keeps_faces_and_reports() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(3, 30, 5), 0.0);
  phx_mc_tet_mesh* mesh = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_FIXED);
  const Exported before = phx::mc::test::export_mesh(mesh);
  const Run run = refine(mesh, 1.2);
  const Exported after = phx::mc::test::export_mesh(mesh);
  PHX_CHECK(after.faces == before.faces);
  PHX_CHECK(after.segments == before.segments);
  PHX_CHECK(run.counters[1] == 0 && run.counters[2] == 0);
  PHX_CHECK(run.status == PHX_MC_REFINEMENT_LIMIT);
  bool fixed = false;
  for (int32_t reason : unmet_reasons(mesh)) {
    fixed = fixed || reason == PHX_MC_TET_MESH_REASON_FIXED_BOUNDARY;
  }
  PHX_CHECK(fixed);
  // New vertices are interior.
  bool interior = true;
  for (std::size_t v = before.dimension.size(); v < after.dimension.size(); ++v) {
    interior = interior && after.dimension[v] == 3;
  }
  PHX_CHECK(interior);
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), 1.0, 1e-12);
  phx_mc_tet_mesh_free(mesh);
}

void test_protecting_ball_is_respected_and_reported() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(3, 20, 9), 0.0);
  std::vector<double> protection(domain.points.size() / 3, 0.0);
  protection[0] = 0.3;  // the corner (0, 0, 0)
  phx_mc_tet_mesh* mesh = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING,
                                                     protection.data());
  const Exported before = phx::mc::test::export_mesh(mesh);
  const Run run = refine(mesh, 1.5);
  const Exported after = phx::mc::test::export_mesh(mesh);
  bool outside = true;
  for (std::size_t v = before.dimension.size(); v < after.dimension.size(); ++v) {
    const double* p = phx::mc::test::at(after, static_cast<int32_t>(v));
    outside = outside && p[0] * p[0] + p[1] * p[1] + p[2] * p[2] >= 0.09;
  }
  PHX_CHECK(outside);
  bool reported = false;
  for (int32_t reason : unmet_reasons(mesh)) {
    reported = reported || reason == PHX_MC_TET_MESH_REASON_PROTECTED;
  }
  PHX_CHECK(run.status == PHX_MC_REFINEMENT_LIMIT && reported);
  phx_mc_tet_mesh_free(mesh);
}

void test_budgets_refuse_cleanly() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(3, 40, 7), 0.0);
  phx_mc_tet_mesh* mesh = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  const Run limited = refine(mesh, 2.0, 5);
  PHX_CHECK(limited.status == PHX_MC_REFINEMENT_LIMIT);
  const Exported after = phx::mc::test::export_mesh(mesh);
  PHX_CHECK(after.dimension.size() == domain.points.size() / 3 + 5);
  bool budget = !unmet_reasons(mesh).empty();
  for (int32_t reason : unmet_reasons(mesh)) {
    budget = budget && reason == PHX_MC_TET_MESH_REASON_BUDGET;
  }
  PHX_CHECK(budget);
  const Run starved = refine(mesh, 2.0, 1 << 20, nullptr, 50);
  PHX_CHECK(starved.status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(starved.counters[7] <= 50);
  const Exported still = phx::mc::test::export_mesh(mesh);
  PHX_CHECK(phx::mc::test::all_positive(still));
  PHX_CHECK_NEAR(phx::mc::test::region_volume(still, 0), 1.0, 1e-12);
  PHX_CHECK(handle_audit(mesh));
  // The same mesh finishes once the budget allows it.
  PHX_CHECK(refine(mesh, 2.0).status == PHX_MC_OK);
  phx_mc_tet_mesh_free(mesh);

  int32_t status = -1;
  phx_mc_tet_mesh* capped = phx::mc::test::create_mesh(
      domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING, nullptr,
      static_cast<int64_t>(domain.points.size() / 3) + 3, 1 << 23, &status);
  PHX_CHECK(status == PHX_MC_OK);
  PHX_CHECK(refine(capped, 2.0).status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(phx::mc::test::export_mesh(capped).dimension.size() == domain.points.size() / 3 + 3);
  phx_mc_tet_mesh_free(capped);
}

void test_invalid_domains_are_refused() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(3, 5, 2), 0.0);
  int32_t status = -1;
  Domain inverted = domain;
  std::swap(inverted.tets[0], inverted.tets[1]);
  PHX_CHECK(phx::mc::test::create_mesh(inverted, 0, nullptr, 1 << 20, 1 << 23, &status) ==
            nullptr);
  PHX_CHECK(status == PHX_MC_INVALID_INPUT);
  Domain open = domain;
  open.faces.resize(open.faces.size() - 3);
  open.face_sources.pop_back();
  PHX_CHECK(phx::mc::test::create_mesh(open, 0, nullptr, 1 << 20, 1 << 23, &status) == nullptr);
  PHX_CHECK(status == PHX_MC_INVALID_INPUT);
  Domain unsegmented = domain;
  unsegmented.segments.clear();
  unsegmented.segment_sources.clear();
  PHX_CHECK(phx::mc::test::create_mesh(unsegmented, 0, nullptr, 1 << 20, 1 << 23, &status) ==
            nullptr);
  PHX_CHECK(status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(phx::mc::test::create_mesh(domain, 7, nullptr, 1 << 20, 1 << 23, &status) == nullptr);
  PHX_CHECK(status == PHX_MC_INVALID_ARGUMENT);
}

// ------------------------------------------- ancestry-backed bounded splits
// Tilted decimal tetrahedron. Rational arithmetic (fixture selection) showed
// that none of its edges has an exactly representable dyadic split and none
// of its faces an exact dyadic barycentric point; each test re-asserts the
// owner's exact search fails before exercising a bounded carrier.
Domain tilted_decimal_domain() {
  Domain domain;
  domain.points = {0.1, 0.2, 0.3, 1.7, 0.4, 0.9, 0.3, 1.9, 0.7, 0.6, 0.5, 2.3};
  domain.tets = {0, 1, 2, 3};
  if (phx::mc::orient3d(domain.points.data(), domain.points.data() + 3,
                        domain.points.data() + 6, domain.points.data() + 9) < 0) {
    std::swap(domain.tets[0], domain.tets[1]);
  }
  domain.regions = {0};
  domain.faces = {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
  domain.face_sources = {0, 1, 2, 3};
  domain.segments = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};
  domain.segment_sources = {0, 1, 2, 3, 4, 5};
  return domain;
}

struct Witnesses {
  std::vector<int8_t> strata;
  std::vector<int32_t> entities;
  std::vector<double> parameters;
};

// Rebuilds the handle over the domain's own rows as its declared source
// complex, with one deviation bound for every row and optional witnesses.
int32_t declare_source(phx_mc_tet_mesh* handle, const Domain& domain, double tolerance,
                       const Witnesses* witnesses = nullptr) {
  const std::vector<double> face_bounds(domain.face_sources.size(), tolerance);
  const std::vector<double> segment_bounds(domain.segment_sources.size(), tolerance);
  phx::mc::SourceComplex source;
  source.face_count = static_cast<int64_t>(domain.face_sources.size());
  source.faces = domain.faces.data();
  source.face_sources = domain.face_sources.data();
  source.face_tolerances = face_bounds.data();
  source.segment_count = static_cast<int64_t>(domain.segment_sources.size());
  source.segments = domain.segments.data();
  source.segment_sources = domain.segment_sources.data();
  source.segment_tolerances = segment_bounds.data();
  if (witnesses != nullptr) {
    source.witness_strata = witnesses->strata.data();
    source.witness_entities = witnesses->entities.data();
    source.witness_parameters = witnesses->parameters.data();
  }
  return handle->mesh->build(
      static_cast<int64_t>(domain.points.size() / 3), domain.points.data(),
      static_cast<int64_t>(domain.regions.size()), domain.tets.data(), domain.regions.data(),
      static_cast<int64_t>(domain.face_sources.size()), domain.faces.data(),
      domain.face_sources.data(), static_cast<int64_t>(domain.segment_sources.size()),
      domain.segments.data(), domain.segment_sources.data(), nullptr, &source);
}

phx_mc_tet_mesh* declared_mesh(const Domain& domain, double tolerance, int32_t policy) {
  phx_mc_tet_mesh* handle = phx::mc::test::create_mesh(domain, policy, nullptr, 256, 4096);
  PHX_CHECK(handle != nullptr);
  PHX_CHECK(declare_source(handle, domain, tolerance) == PHX_MC_OK);
  return handle;
}

// Independent exact oracles: the squared perpendicular distance of p to the
// line (a, b) or the plane (a, b, c) is at most bound^2, compared without
// division or square roots in expansion arithmetic.
bool within_line(const double* a, const double* b, const double* p, double bound) {
  using phx::mc::Expansion;
  Expansion u[3], w[3], cross, length;
  for (int k = 0; k < 3; ++k) {
    u[k] = Expansion::difference(b[k], a[k]);
    w[k] = Expansion::difference(p[k], a[k]);
  }
  for (int k = 0; k < 3; ++k) {
    const Expansion c = w[(k + 1) % 3] * u[(k + 2) % 3] - w[(k + 2) % 3] * u[(k + 1) % 3];
    cross = cross + c * c;
    length = length + u[k] * u[k];
  }
  return (cross - length * Expansion::product(bound, bound)).sign() <= 0;
}

bool within_plane(const double* a, const double* b, const double* c, const double* p,
                  double bound) {
  using phx::mc::Expansion;
  Expansion u[3], v[3], normal;
  for (int k = 0; k < 3; ++k) {
    u[k] = Expansion::difference(b[k], a[k]);
    v[k] = Expansion::difference(c[k], a[k]);
  }
  for (int k = 0; k < 3; ++k) {
    const Expansion n = u[(k + 1) % 3] * v[(k + 2) % 3] - u[(k + 2) % 3] * v[(k + 1) % 3];
    normal = normal + n * n;
  }
  const Expansion determinant = phx::mc::orient3d_exact(a, b, c, p);
  return (determinant * determinant - normal * Expansion::product(bound, bound)).sign() <= 0;
}

// Exact six-fold signed volume of all exported cells.
phx::mc::Expansion six_volume(const Exported& mesh) {
  phx::mc::Expansion total;
  for (std::size_t t = 0; t < mesh.regions.size(); ++t) {
    const int32_t* v = mesh.tets.data() + 4 * t;
    total = total + phx::mc::orient3d_exact(phx::mc::test::at(mesh, v[0]),
                                            phx::mc::test::at(mesh, v[1]),
                                            phx::mc::test::at(mesh, v[2]),
                                            phx::mc::test::at(mesh, v[3]));
  }
  return total;
}

// Every edge of the exported constrained faces bounds exactly two of them.
bool closed_shell(const Exported& mesh) {
  std::map<std::pair<int32_t, int32_t>, int> uses;
  for (std::size_t f = 0; f < mesh.face_sources.size(); ++f) {
    const int32_t* v = mesh.faces.data() + 3 * f;
    for (int k = 0; k < 3; ++k) {
      ++uses[std::minmax(v[k], v[(k + 1) % 3])];
    }
  }
  for (const auto& [edge, count] : uses) {
    if (count != 2) {
      return false;
    }
  }
  return !uses.empty();
}

// Every live boundary vertex is independently within its certified deviation
// of the source row named by its witness (zero-deviation vertices exactly on
// a row of each incident face), every face has an ancestor row, and the
// achieved bound is the largest certified deviation.
void check_source_fidelity(const phx_mc_tet_mesh* handle, const Domain& source,
                           double tolerance) {
  const phx::mc::TetMesh& mesh = *handle->mesh;
  const Exported exported = phx::mc::test::export_mesh(handle);
  double largest = 0.0;
  for (int32_t v = 0; v < mesh.vertex_count(); ++v) {
    if (!mesh.live_vertex(v)) {
      continue;
    }
    const phx::mc::SourceWitness& witness = mesh.witness(v);
    largest = std::max(largest, witness.deviation);
    PHX_CHECK(witness.deviation <= tolerance);
    if (witness.deviation == 0.0) {
      // Exact points name the source row that contains them exactly.
      if (witness.stratum != phx::mc::SourceStratum::kNone) {
        const int32_t* row = witness.stratum == phx::mc::SourceStratum::kSegment
                                 ? source.segments.data() + 2 * witness.entity
                                 : source.faces.data() + 3 * witness.entity;
        const double* p = source.points.data();
        const double* corners[3] = {p + 3 * row[0], p + 3 * row[1],
                                    witness.stratum == phx::mc::SourceStratum::kFacet
                                        ? p + 3 * row[2]
                                        : nullptr};
        PHX_CHECK(witness.entity >= 0 &&
                  phx::mc::on_source_entity(corners, witness.stratum, mesh.point(v)));
      }
      continue;
    }
    const int32_t* row = witness.stratum == phx::mc::SourceStratum::kSegment
                             ? source.segments.data() + 2 * witness.entity
                             : source.faces.data() + 3 * witness.entity;
    const double* p = source.points.data();
    PHX_CHECK(witness.stratum == phx::mc::SourceStratum::kSegment
                  ? within_line(p + 3 * row[0], p + 3 * row[1], mesh.point(v), witness.deviation)
                  : within_plane(p + 3 * row[0], p + 3 * row[1], p + 3 * row[2], mesh.point(v),
                                 witness.deviation));
  }
  PHX_CHECK(mesh.achieved_source_error_bound() == largest);
  for (std::size_t f = 0; f < exported.face_sources.size(); ++f) {
    PHX_CHECK(mesh.original_face_ancestor(exported.faces.data() + 3 * f,
                                          exported.face_sources[f]) ==
              exported.face_sources[f]);
  }
  for (std::size_t s = 0; s < exported.segment_sources.size(); ++s) {
    PHX_CHECK(mesh.original_segment_ancestor(exported.segments[2 * s],
                                             exported.segments[2 * s + 1],
                                             exported.segment_sources[s]) ==
              exported.segment_sources[s]);
  }
  PHX_CHECK(phx::mc::test::all_positive(exported) && closed_shell(exported) &&
            handle_audit(handle));
}

void test_rotated_decimal_constraint_split_has_exact_original_ancestry() {
  Domain domain;
  domain.points = {-1.6, 1.4, 0.9, 0.6, 0.5, -0.3, 0.2, -1.0, 1.0, 1.0, 2.0, 2.0};
  domain.tets = {0, 1, 2, 3};
  const double* p[4] = {domain.points.data(), domain.points.data() + 3,
                        domain.points.data() + 6, domain.points.data() + 9};
  if (phx::mc::orient3d(p[0], p[1], p[2], p[3]) < 0) {
    std::swap(domain.tets[0], domain.tets[1]);
  }
  domain.regions = {0};
  domain.faces = {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
  domain.face_sources = {0, 1, 2, 3};
  domain.segments = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};
  domain.segment_sources = {0, 1, 2, 3, 4, 5};
  phx_mc_tet_mesh* handle =
      phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING, nullptr, 16, 32);
  PHX_CHECK(handle != nullptr);
  phx::mc::TetMesh& mesh = *handle->mesh;
  PHX_CHECK(declare_source(handle, domain, 1e-14) == PHX_MC_OK);
  double midpoint[3];
  PHX_CHECK(mesh.construct_edge_point(0, 1, midpoint));
  const double rounded_midpoint[3] = {(p[0][0] + p[1][0]) * 0.5,
                                     (p[0][1] + p[1][1]) * 0.5,
                                     (p[0][2] + p[1][2]) * 0.5};
  PHX_CHECK(!phx::mc::collinear3d(p[0], p[1], rounded_midpoint));
  PHX_CHECK(phx::mc::collinear3d(p[0], p[1], midpoint));
  int32_t inserted = -1;
  PHX_CHECK(mesh.split_edge(0, 1, midpoint, 0.0, inserted) == phx::mc::Insertion::kOk);
  PHX_CHECK(mesh.original_segment_ancestor(0, inserted, 0) == 0);
  PHX_CHECK(mesh.original_segment_ancestor(inserted, 1, 0) == 0);
  const Exported after = phx::mc::test::export_mesh(handle);
  for (std::size_t i = 0; i < after.face_sources.size(); ++i) {
    const int32_t source = after.face_sources[i];
    const int32_t* face = after.faces.data() + 3 * i;
    PHX_CHECK(mesh.original_face_ancestor(face, source) == source);
    const int32_t* original = domain.faces.data() + 3 * source;
    for (int k = 0; k < 3; ++k) {
      PHX_CHECK(phx::mc::orient3d(domain.points.data() + 3 * original[0],
                                  domain.points.data() + 3 * original[1],
                                  domain.points.data() + 3 * original[2],
                                  mesh.point(face[k])) == 0);
    }
  }
  PHX_CHECK(mesh.achieved_source_error_bound() == 0.0);
  PHX_CHECK(mesh.requested_source_tolerance() == 1e-14);
  PHX_CHECK(handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}

void test_oblique_facet_refinement_preserves_shape_with_bounded_work() {
  // A near-boundary site encroaches an oblique source plane whose triangle
  // circumcenters need not be binary64 points. Arbitrary exact interior
  // star splits waste the work budget by creating smaller descendant angles.
  auto points = phx::mc::test::box_points(3, 0, 7);
  points.insert(points.end(), {0.25, 0.25, 0.03125});
  Domain domain = phx::mc::test::box_domain(points, 0.0);
  for (std::size_t i = 0; i < domain.points.size(); i += 3) {
    domain.points[i + 2] += 0.25 * domain.points[i] + 0.125 * domain.points[i + 1];
  }
  phx_mc_tet_mesh* mesh =
      phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  PHX_CHECK(mesh != nullptr);
  const Exported before = phx::mc::test::export_mesh(mesh);
  const Run starved = refine(mesh, 2.0, 1 << 20, nullptr, 50);
  PHX_CHECK(starved.status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(starved.counters[7] <= 50);
  PHX_CHECK(phx::mc::test::export_mesh(mesh) == before);
  PHX_CHECK(handle_audit(mesh));

  constexpr int64_t work_limit = 10000;
  const Run run = refine(mesh, 2.0, 1 << 20, nullptr, work_limit);
  PHX_CHECK(run.status == PHX_MC_OK);
  PHX_CHECK(run.counters[7] <= work_limit);
  const Exported after = phx::mc::test::export_mesh(mesh);
  PHX_CHECK(worst_ratio(after) <= 2.0 + 1e-9);
  PHX_CHECK(phx::mc::test::all_positive(after));
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), 1.0, 1e-12);
  PHX_CHECK(unmet_reasons(mesh).empty());
  // Pull back only the oracle coordinates, not the native source. The exact
  // shear preserves volume and maps every source side onto a unit-box side.
  Exported pulled_back = after;
  for (std::size_t i = 0; i < pulled_back.points.size(); i += 3) {
    pulled_back.points[i + 2] -=
        0.25 * pulled_back.points[i] + 0.125 * pulled_back.points[i + 1];
  }
  PHX_CHECK(phx::mc::test::faces_on_sides(pulled_back, 0.0));
  for (int32_t source = 0; source < 6; ++source) {
    PHX_CHECK_NEAR(phx::mc::test::source_area(pulled_back, source), 1.0, 1e-12);
  }
  for (std::size_t e = 0; e < after.segment_sources.size(); ++e) {
    PHX_CHECK(mesh->mesh->original_segment_ancestor(after.segments[2 * e],
                                                   after.segments[2 * e + 1],
                                                   after.segment_sources[e]) >= 0);
  }
  PHX_CHECK(handle_audit(mesh));
  phx_mc_tet_mesh_free(mesh);
}

void test_exact_acute_source_angle_reports_infeasible_shape_without_growth() {
  Domain domain;
  domain.points = {0, 0, 0, 1, 0, 0, 1, 0.125, 0, 0.25, 0.25, 1};
  domain.tets = {0, 1, 2, 3};
  if (phx::mc::orient3d(domain.points.data(), domain.points.data() + 3,
                        domain.points.data() + 6, domain.points.data() + 9) < 0) {
    std::swap(domain.tets[0], domain.tets[1]);
  }
  domain.regions = {0};
  domain.faces = {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
  domain.face_sources = {0, 1, 2, 3};
  domain.segments = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};
  domain.segment_sources = {0, 1, 2, 3, 4, 5};
  phx_mc_tet_mesh* mesh =
      phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING, nullptr, 1000, 5000);
  PHX_CHECK(mesh != nullptr);
  const Exported before = phx::mc::test::export_mesh(mesh);
  const Run run = refine(mesh, 2.0, 500, nullptr, 100000);
  PHX_CHECK(run.status == PHX_MC_REFINEMENT_LIMIT);
  PHX_CHECK(phx::mc::test::export_mesh(mesh) == before);
  int64_t counts[PHX_MC_TET_MESH_COUNTS];
  PHX_CHECK(phx_mc_tet_mesh_counts(mesh, counts) == PHX_MC_OK);
  PHX_CHECK(counts[4] == 1);
  const auto reasons = unmet_reasons(mesh);
  PHX_CHECK(reasons[0] == PHX_MC_TET_MESH_REASON_PROTECTED);
  PHX_CHECK(handle_audit(mesh));
  phx_mc_tet_mesh_free(mesh);
}

void test_exact_positive_reconstruction_sliver_is_not_silently_satisfied() {
  const Domain domain = phx::mc::test::reconstruction_sliver_domain();
  phx_mc_tet_mesh* mesh = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_FIXED);
  PHX_CHECK(mesh != nullptr);
  const Exported before = phx::mc::test::export_mesh(mesh);
  double values[PHX_MC_TET_MESH_QUALITY_VALUES] = {};
  int64_t histogram[PHX_MC_TET_MESH_QUALITY_BINS] = {}, slivers = 0;
  PHX_CHECK(phx_mc_tet_mesh_quality(mesh, 5.0, values, histogram, &slivers) == PHX_MC_OK);
  PHX_CHECK(std::isfinite(values[2]) && values[2] > 1e15);
  PHX_CHECK(values[4] > 0);
  PHX_CHECK_NEAR(values[4] / (8.987998638190837e-22 / 6.0), 1.0, 1e-14);
  PHX_CHECK(slivers == 1 && values[0] < 5.0);
  const Run run = refine(mesh, 2.0, 8, nullptr, 100000);
  PHX_CHECK(run.status == PHX_MC_REFINEMENT_LIMIT);
  PHX_CHECK(phx::mc::test::export_mesh(mesh) == before);
  int64_t counts[PHX_MC_TET_MESH_COUNTS] = {};
  PHX_CHECK(phx_mc_tet_mesh_counts(mesh, counts) == PHX_MC_OK && counts[4] == 1);
  PHX_CHECK(handle_audit(mesh));
  phx_mc_tet_mesh_free(mesh);
}

void test_independent_permuted_source_bank_uses_source_corner_topology() {
  const Domain domain = tilted_decimal_domain();
  // An unused source point and a permutation put every row in a bank wholly
  // independent of the four carrier slots.
  const int32_t permutation[4] = {4, 2, 1, 3};
  std::vector<double> source_points = {20, 30, 40};
  for (int32_t original : {2, 1, 3, 0}) {
    source_points.insert(source_points.end(), domain.points.begin() + 3 * original,
                         domain.points.begin() + 3 * original + 3);
  }
  std::vector<int32_t> source_faces = domain.faces;
  std::vector<int32_t> source_segments = domain.segments;
  for (int32_t& v : source_faces) v = permutation[v];
  for (int32_t& v : source_segments) v = permutation[v];
  const std::vector<double> bounds(6, 1e-12);
  phx::mc::SourceComplex source;
  source.point_count = 5;
  source.points = source_points.data();
  source.face_count = 4;
  source.faces = source_faces.data();
  source.face_sources = domain.face_sources.data();
  source.face_tolerances = bounds.data();
  source.segment_count = 6;
  source.segments = source_segments.data();
  source.segment_sources = domain.segment_sources.data();
  source.segment_tolerances = bounds.data();
  phx_mc_tet_mesh* handle =
      phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  PHX_CHECK(handle->mesh->build(4, domain.points.data(), 1, domain.tets.data(),
      domain.regions.data(), 4, domain.faces.data(), domain.face_sources.data(),
      6, domain.segments.data(), domain.segment_sources.data(), nullptr, &source) == PHX_MC_OK);
  const Run run = refine(handle, 2.0, 0, nullptr, 1 << 20);
  PHX_CHECK(run.status == PHX_MC_REFINEMENT_LIMIT || run.status == PHX_MC_OK);
  PHX_CHECK(handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}

void test_authored_subsegment_obeys_containing_zero_bound_facet() {
  Domain original = tilted_decimal_domain();
  original.points = {0, 0, 0, 1.7, .4, .9, 0, 2, 0, 0, 0, 2};
  if (phx::mc::orient3d(original.points.data(), original.points.data() + 3,
                        original.points.data() + 6, original.points.data() + 9) < 0) {
    std::swap(original.tets[0], original.tets[1]);
  }
  phx_mc_tet_mesh* handle = declared_mesh(
      original, 1e-12, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  handle->mesh->set_work_limit(1 << 20);
  double midpoint[3] = {.85, .2, .45};
  int32_t middle = -1;
  PHX_CHECK(handle->mesh->split_edge(0, 1, midpoint, 0.0, middle) ==
            phx::mc::Insertion::kOk);
  const Exported carrier = phx::mc::test::export_mesh(handle);
  std::vector<double> source_points = original.points;
  source_points.insert(source_points.end(), midpoint, midpoint + 3);
  const int32_t segments[4] = {0, 4, 4, 1};
  const int32_t segment_ids[2] = {0, 0};
  const double segment_bounds[2] = {1e-12, 1e-12};
  const double face_bounds[4] = {0, 1e-12, 1e-12, 1e-12};
  phx::mc::SourceComplex source;
  source.point_count = 5;
  source.points = source_points.data();
  source.face_count = 4;
  source.faces = original.faces.data();
  source.face_sources = original.face_sources.data();
  source.face_tolerances = face_bounds;
  // Include the remaining original edges as well as the authored subsegments.
  std::vector<int32_t> source_segments(segments, segments + 4);
  source_segments.insert(source_segments.end(), original.segments.begin() + 2, original.segments.end());
  std::vector<int32_t> ids(segment_ids, segment_ids + 2);
  ids.insert(ids.end(), original.segment_sources.begin() + 1, original.segment_sources.end());
  std::vector<double> bounds(ids.size(), segment_bounds[0]);
  source.segment_count = static_cast<int64_t>(ids.size());
  source.segments = source_segments.data();
  source.segment_sources = ids.data();
  source.segment_tolerances = bounds.data();
  PHX_CHECK(handle->mesh->build(5, carrier.points.data(),
      static_cast<int64_t>(carrier.regions.size()), carrier.tets.data(), carrier.regions.data(),
      static_cast<int64_t>(carrier.face_sources.size()), carrier.faces.data(), carrier.face_sources.data(),
      static_cast<int64_t>(carrier.segment_sources.size()), carrier.segments.data(),
      carrier.segment_sources.data(), nullptr, &source) == PHX_MC_OK);
  handle->mesh->set_work_limit(1 << 20);
  double position[3];
  phx::mc::SourceWitness witness;
  PHX_CHECK(handle->mesh->construct_source_split(middle, 1, .5, position, witness) ==
            phx::mc::Insertion::kDeviation);
  PHX_CHECK(!handle->mesh->source_refusals().empty());
  PHX_CHECK(handle->mesh->source_refusals().back().tolerance == 0.0);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == carrier);
  phx_mc_tet_mesh_free(handle);
}

void test_unsegmented_shared_diagonal_keeps_both_exact_facet_ancestors() {
  Domain domain;
  domain.points = {.85, .2, 0, 1.7, .4, 0, 1.7, .4, 1, .85, .2, 1, 0, 1, .5};
  domain.tets = {0, 1, 2, 4, 0, 2, 3, 4};
  for (int offset : {0, 4}) {
    const int32_t* v = domain.tets.data() + offset;
    if (phx::mc::orient3d(domain.points.data() + 3 * v[0], domain.points.data() + 3 * v[1],
                          domain.points.data() + 3 * v[2], domain.points.data() + 3 * v[3]) < 0) {
      std::swap(domain.tets[offset], domain.tets[offset + 1]);
    }
  }
  domain.regions = {0, 0};
  domain.faces = {0, 1, 2, 0, 2, 3, 0, 1, 4, 1, 2, 4, 2, 3, 4, 0, 3, 4};
  domain.face_sources = {0, 0, 2, 3, 4, 5};
  // The outer creases are source segments; the coplanar A-C diagonal is not.
  domain.segments = {0, 1, 1, 2, 2, 3, 0, 3, 0, 4, 1, 4, 2, 4, 3, 4};
  domain.segment_sources = {0, 1, 2, 3, 4, 5, 6, 7};
  phx_mc_tet_mesh* handle = declared_mesh(domain, 1e-12, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  handle->mesh->set_work_limit(1 << 20);
  double position[3];
  phx::mc::SourceWitness witness;
  PHX_CHECK(handle->mesh->construct_source_split(0, 2, .5, position, witness) ==
            phx::mc::Insertion::kOk);
  PHX_CHECK(witness.stratum == phx::mc::SourceStratum::kFacet && witness.deviation > 0);
  int32_t inserted = -1;
  PHX_CHECK(handle->mesh->split_edge_bounded(0, 2, position, witness, 0.0, inserted) ==
            phx::mc::Insertion::kOk);
  const Exported after = phx::mc::test::export_mesh(handle);
  for (std::size_t i = 0; i < after.face_sources.size(); ++i) {
    PHX_CHECK(handle->mesh->original_face_ancestor(
        after.faces.data() + 3 * i, after.face_sources[i]) >= 0);
  }
  PHX_CHECK(handle_audit(handle));
  double bounds[6] = {1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12};
  PHX_CHECK(witness.entity == 0 || witness.entity == 1);
  bounds[witness.entity == 0 ? 1 : 0] = 0.0;
  phx::mc::SourceComplex source;
  source.face_count = 6;
  source.faces = domain.faces.data();
  source.face_sources = domain.face_sources.data();
  source.face_tolerances = bounds;
  const double segment_bounds[8] = {1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12};
  source.segment_count = 8;
  source.segments = domain.segments.data();
  source.segment_sources = domain.segment_sources.data();
  source.segment_tolerances = segment_bounds;
  PHX_CHECK(handle->mesh->build(5, domain.points.data(), 2, domain.tets.data(),
      domain.regions.data(), 6, domain.faces.data(), domain.face_sources.data(),
      8, domain.segments.data(), domain.segment_sources.data(), nullptr, &source) == PHX_MC_OK);
  handle->mesh->set_work_limit(1 << 20);
  const Exported before_refusal = phx::mc::test::export_mesh(handle);
  PHX_CHECK(handle->mesh->construct_source_split(0, 2, .5, position, witness) ==
            phx::mc::Insertion::kDeviation);
  PHX_CHECK(handle->mesh->source_refusals().back().tolerance == 0);
  PHX_CHECK(phx::mc::test::export_mesh(handle) == before_refusal);
  phx_mc_tet_mesh_free(handle);
}

void test_tilted_decimal_bounded_segment_split_is_ancestry_backed() {
  const Domain domain = tilted_decimal_domain();
  phx_mc_tet_mesh* handle = declared_mesh(domain, 1e-12, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  phx::mc::TetMesh& mesh = *handle->mesh;
  const double* a = domain.points.data();
  const double* b = domain.points.data() + 3;
  double exact[3];
  PHX_CHECK(!mesh.construct_edge_point(0, 1, exact));
  double carrier[3];
  phx::mc::SourceWitness witness;
  PHX_CHECK(mesh.construct_source_split(0, 1, 0.5, carrier, witness) == phx::mc::Insertion::kOk);
  PHX_CHECK(witness.stratum == phx::mc::SourceStratum::kSegment && witness.entity == 0);
  PHX_CHECK(witness.parameters[0] == 0.5);
  PHX_CHECK(witness.deviation > 0.0 && witness.deviation <= 1e-12);
  PHX_CHECK(!phx::mc::collinear3d(a, b, carrier) && within_line(a, b, carrier, witness.deviation));
  // Each coordinate is the nearest binary64 value to the exact midpoint.
  for (int k = 0; k < 3; ++k) {
    const phx::mc::Expansion middle =
        (phx::mc::Expansion(a[k]) + phx::mc::Expansion(b[k])).scaled(0.5);
    for (const double neighbour : {std::nextafter(carrier[k], -1e300),
                                   std::nextafter(carrier[k], 1e300)}) {
      phx::mc::Expansion own = middle - phx::mc::Expansion(carrier[k]);
      phx::mc::Expansion other = middle - phx::mc::Expansion(neighbour);
      if (own.sign() < 0) own = -own;
      if (other.sign() < 0) other = -other;
      PHX_CHECK((own - other).sign() <= 0);
    }
  }
  const Exported before = phx::mc::test::export_mesh(handle);
  PHX_CHECK(mesh.achieved_source_error_bound() == 0.0);
  int32_t inserted = -1;
  PHX_CHECK(mesh.split_edge_bounded(0, 1, carrier, witness, 0.0, inserted) ==
            phx::mc::Insertion::kOk);
  const Exported after = phx::mc::test::export_mesh(handle);
  PHX_CHECK(after.dimension[static_cast<std::size_t>(inserted)] == 1);
  PHX_CHECK(after.face_sources.size() == 6 && after.segment_sources.size() == 7);
  PHX_CHECK(mesh.witness(inserted).deviation == witness.deviation);
  PHX_CHECK(mesh.achieved_source_error_bound() == witness.deviation);
  PHX_CHECK(mesh.original_segment_ancestor(0, inserted, 0) == 0);
  PHX_CHECK(mesh.original_segment_ancestor(inserted, 1, 0) == 0);
  check_source_fidelity(handle, domain, 1e-12);
  // The region volume moves by at most the slivers over the two source
  // faces through the split edge: |dV| <= (A(0,1,2) + A(0,1,3)) d / 3.
  const double changed =
      std::fabs((six_volume(after) - six_volume(before)).estimate()) / 6.0;
  const double bound =
      (phx::mc::test::source_area(before, 0) + phx::mc::test::source_area(before, 1)) *
      witness.deviation / 3.0;
  PHX_CHECK(changed <= bound * (1.0 + 1e-12));
  phx_mc_tet_mesh_free(handle);
}

void test_tilted_decimal_bound_below_certified_deviation_refuses_without_mutation() {
  const Domain domain = tilted_decimal_domain();
  phx_mc_tet_mesh* probe = declared_mesh(domain, 1e-12, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  double carrier[3];
  phx::mc::SourceWitness certified;
  PHX_CHECK(probe->mesh->construct_source_split(0, 1, 0.5, carrier, certified) ==
            phx::mc::Insertion::kOk);
  phx_mc_tet_mesh_free(probe);
  for (const double tolerance : {std::nextafter(certified.deviation, 0.0), 0.0}) {
    phx_mc_tet_mesh* handle = declared_mesh(domain, tolerance, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
    phx::mc::TetMesh& mesh = *handle->mesh;
    const Exported before = phx::mc::test::export_mesh(handle);
    phx::mc::SourceWitness refused;
    PHX_CHECK(mesh.construct_source_split(0, 1, 0.5, carrier, refused) ==
              phx::mc::Insertion::kDeviation);
    PHX_CHECK(mesh.source_refusals().size() == 1);
    PHX_CHECK(mesh.source_refusals()[0].deviation == certified.deviation);
    PHX_CHECK(mesh.source_refusals()[0].tolerance == tolerance);
    int32_t inserted = -1;
    PHX_CHECK(mesh.split_edge_bounded(0, 1, carrier, certified, 0.0, inserted) ==
              phx::mc::Insertion::kDeviation);
    PHX_CHECK(inserted == -1 && phx::mc::test::export_mesh(handle) == before);
    if (tolerance == 0.0) {
      // Zero tolerance: refinement of the oversized boundary leaves every
      // constrained face and segment exactly as declared, with evidence.
      std::vector<double> sizes(4, 0.5);
      const Run run = refine(handle, 2.0, 64, sizes.data(), 1 << 20);
      PHX_CHECK(run.status == PHX_MC_REFINEMENT_LIMIT);
      const Exported after = phx::mc::test::export_mesh(handle);
      PHX_CHECK(after.faces == before.faces && after.face_sources == before.face_sources &&
                after.segments == before.segments &&
                after.segment_sources == before.segment_sources);
      const auto reasons = unmet_reasons(handle);
      PHX_CHECK(std::find(reasons.begin(), reasons.end(),
                          PHX_MC_TET_MESH_REASON_NONREPRESENTABLE) != reasons.end());
      PHX_CHECK(!mesh.source_refusals().empty() && mesh.achieved_source_error_bound() == 0.0);
      check_source_fidelity(handle, domain, 0.0);
    }
    phx_mc_tet_mesh_free(handle);
  }
  phx_mc_tet_mesh* fixed = declared_mesh(domain, 1e-12, PHX_MC_TET_MESH_BOUNDARY_FIXED);
  const Exported before = phx::mc::test::export_mesh(fixed);
  PHX_CHECK(fixed->mesh->construct_source_split(0, 1, 0.5, carrier, certified) ==
            phx::mc::Insertion::kRefused);
  PHX_CHECK(phx::mc::test::export_mesh(fixed) == before);
  phx_mc_tet_mesh_free(fixed);
}

void test_tilted_decimal_refinement_uses_bounded_ancestry() {
  const Domain domain = tilted_decimal_domain();
  phx_mc_tet_mesh* handle = declared_mesh(domain, 1e-12, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  const Exported before = phx::mc::test::export_mesh(handle);
  double exact[3];
  PHX_CHECK(!handle->mesh->construct_facet_point(0, 1, 2, exact));
  std::vector<double> sizes(4, 0.6);
  const Run run = refine(handle, 2.0, 400, sizes.data(), int64_t{1} << 26);
  PHX_CHECK(run.status == PHX_MC_OK || run.status == PHX_MC_REFINEMENT_LIMIT);
  PHX_CHECK(run.counters[2] > 0 && run.counters[1] > 0);
  const phx::mc::TetMesh& mesh = *handle->mesh;
  PHX_CHECK(mesh.achieved_source_error_bound() > 0.0 &&
            mesh.achieved_source_error_bound() <= 1e-12);
  check_source_fidelity(handle, domain, 1e-12);
  // Total area of the constrained surface is the declared area up to the
  // certified fold, and the volume up to its slivers.
  const Exported after = phx::mc::test::export_mesh(handle);
  double declared_area = 0.0;
  for (std::size_t f = 0; f < before.face_sources.size(); ++f) {
    declared_area += phx::mc::test::face_area(before, f);
  }
  const double changed =
      std::fabs((six_volume(after) - six_volume(before)).estimate()) / 6.0;
  PHX_CHECK(changed <= declared_area * mesh.achieved_source_error_bound() * (1.0 + 1e-9));
  phx_mc_tet_mesh_free(handle);
}

// Witnesses survive publication: the refined carrier rebuilt with its
// exported witnesses over the same source rows recertifies every deviation;
// a tampered witness or one on an interior vertex is refused.
void test_bounded_witnesses_round_trip_and_tampering_is_refused() {
  const Domain source = tilted_decimal_domain();
  phx_mc_tet_mesh* handle = declared_mesh(source, 1e-12, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  std::vector<double> sizes(4, 0.8);
  const Run run = refine(handle, 2.0, 200, sizes.data(), int64_t{1} << 26);
  PHX_CHECK(run.status == PHX_MC_OK || run.status == PHX_MC_REFINEMENT_LIMIT);
  const phx::mc::TetMesh& mesh = *handle->mesh;
  const Exported carrier = phx::mc::test::export_mesh(handle);
  Domain rebuilt;
  rebuilt.points = carrier.points;
  rebuilt.tets = carrier.tets;
  rebuilt.regions = carrier.regions;
  rebuilt.faces = carrier.faces;
  rebuilt.face_sources = carrier.face_sources;
  rebuilt.segments = carrier.segments;
  rebuilt.segment_sources = carrier.segment_sources;
  Witnesses witnesses;
  int32_t bounded = -1;
  for (int32_t v = 0; v < mesh.vertex_count(); ++v) {
    // The canonical evidence is re-declared verbatim: exact witnesses keep
    // their named rows, retired vertices carry none.
    const phx::mc::SourceWitness& witness = mesh.witness(v);
    witnesses.strata.push_back(static_cast<int8_t>(witness.stratum));
    witnesses.entities.push_back(witness.entity);
    witnesses.parameters.push_back(witness.parameters[0]);
    witnesses.parameters.push_back(witness.parameters[1]);
    bounded = mesh.live_vertex(v) && witness.deviation > 0.0 ? v : bounded;
  }
  PHX_CHECK(bounded >= 0);
  phx_mc_tet_mesh* copy = phx::mc::test::create_mesh(
      rebuilt, PHX_MC_TET_MESH_BOUNDARY_CONFORMING, nullptr, 4096, 1 << 16);
  // The folded carrier is no longer its own exact complex.
  PHX_CHECK(copy == nullptr);
  copy = phx::mc::test::create_mesh(source, PHX_MC_TET_MESH_BOUNDARY_CONFORMING, nullptr, 4096,
                                    1 << 16);
  const auto rebuild = [&](const Witnesses& declared) {
    const std::vector<double> face_bounds(source.face_sources.size(), 1e-12);
    const std::vector<double> segment_bounds(source.segment_sources.size(), 1e-12);
    phx::mc::SourceComplex complex;
    complex.point_count = static_cast<int64_t>(source.points.size() / 3);
    complex.points = source.points.data();
    complex.face_count = static_cast<int64_t>(source.face_sources.size());
    complex.faces = source.faces.data();
    complex.face_sources = source.face_sources.data();
    complex.face_tolerances = face_bounds.data();
    complex.segment_count = static_cast<int64_t>(source.segment_sources.size());
    complex.segments = source.segments.data();
    complex.segment_sources = source.segment_sources.data();
    complex.segment_tolerances = segment_bounds.data();
    complex.witness_strata = declared.strata.data();
    complex.witness_entities = declared.entities.data();
    complex.witness_parameters = declared.parameters.data();
    return copy->mesh->build(
        static_cast<int64_t>(rebuilt.points.size() / 3), rebuilt.points.data(),
        static_cast<int64_t>(rebuilt.regions.size()), rebuilt.tets.data(), rebuilt.regions.data(),
        static_cast<int64_t>(rebuilt.face_sources.size()), rebuilt.faces.data(),
        rebuilt.face_sources.data(), static_cast<int64_t>(rebuilt.segment_sources.size()),
        rebuilt.segments.data(), rebuilt.segment_sources.data(), nullptr, &complex);
  };
  PHX_CHECK(rebuild(witnesses) == PHX_MC_OK);
  PHX_CHECK(copy->mesh->achieved_source_error_bound() == mesh.achieved_source_error_bound());
  check_source_fidelity(copy, source, 1e-12);
  Witnesses tampered = witnesses;
  tampered.parameters[2 * static_cast<std::size_t>(bounded)] *= 0.5;
  PHX_CHECK(rebuild(tampered) == PHX_MC_INVALID_INPUT);
  Witnesses interior = witnesses;
  int32_t inside = -1;
  for (int32_t v = 0; v < mesh.vertex_count(); ++v) {
    inside = mesh.live_vertex(v) && mesh.dimension(v) == 3 ? v : inside;
  }
  if (inside >= 0) {
    interior.strata[static_cast<std::size_t>(inside)] =
        static_cast<int8_t>(phx::mc::SourceStratum::kFacet);
    interior.entities[static_cast<std::size_t>(inside)] = 0;
    PHX_CHECK(rebuild(interior) == PHX_MC_INVALID_INPUT);
  }
  phx_mc_tet_mesh_free(copy);
  phx_mc_tet_mesh_free(handle);
}

void test_bounded_poor_shape_refines_even_when_size_is_already_met() {
  const Domain domain = tilted_decimal_domain();
  phx_mc_tet_mesh* handle = declared_mesh(
      domain, 1e-12, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  handle->mesh->set_work_limit(1 << 26);
  double position[3];
  phx::mc::SourceWitness witness;
  PHX_CHECK(handle->mesh->construct_source_split(0, 1, 1.0 / 3.0, position, witness) ==
            phx::mc::Insertion::kOk);
  PHX_CHECK(witness.deviation > 0.0);
  int32_t inserted = -1;
  PHX_CHECK(handle->mesh->split_edge_bounded(0, 1, position, witness, 10.0, inserted) ==
            phx::mc::Insertion::kOk);
  const Exported before = phx::mc::test::export_mesh(handle);
  PHX_CHECK(worst_ratio(before) > 2.0);
  const std::vector<double> sizes(before.dimension.size(), 10.0);
  const Run run = refine(handle, 2.0, 400, sizes.data(), int64_t{1} << 26);
  PHX_CHECK(run.status == PHX_MC_OK || run.status == PHX_MC_REFINEMENT_LIMIT);
  PHX_CHECK(run.counters[1] + run.counters[2] > 0);
  PHX_CHECK(handle->mesh->vertex_count() > static_cast<int64_t>(before.dimension.size()));
  PHX_CHECK(handle->mesh->source_refusals().empty());
  const auto reasons = unmet_reasons(handle);
  PHX_CHECK(std::find(reasons.begin(), reasons.end(),
                      PHX_MC_TET_MESH_REASON_NONREPRESENTABLE) == reasons.end());
  check_source_fidelity(handle, domain, 1e-12);
  PHX_CHECK(handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}
void test_exact_row_carrier_remains_canonical_authority_despite_parameter_rounding() {
  Domain domain = tilted_decimal_domain();
  domain.points = {0, 0, 0, 3, 1, 0, 0, 1, 0, 0, 0, 1};
  domain.tets = {0, 1, 2, 3};
  phx_mc_tet_mesh* handle = declared_mesh(domain, 1e-12, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  handle->mesh->set_work_limit(1 << 20);
  const double* corners[3];
  handle->mesh->row_corners(phx::mc::SourceStratum::kFacet, 0, corners);
  phx::mc::SourceWitness witness{phx::mc::SourceStratum::kFacet, 0, {1.0 / 3.0, .25}, 0.0};
  double position[3];
  PHX_CHECK(phx::mc::source_carrier(corners, witness.stratum, witness.parameters,
                                  position, witness.deviation));
  PHX_CHECK(witness.deviation == 0.0);
  phx::mc::Expansion authored[3], authority[3];
  PHX_CHECK(phx::mc::source_point(corners, witness.stratum, witness.parameters, authored));
  PHX_CHECK((authored[0] - phx::mc::Expansion(position[0])).sign() != 0);
  int32_t t = -1;
  int slot = -1;
  PHX_CHECK(handle->mesh->find_face(0, 1, 2, t, slot));
  PHX_CHECK(handle->mesh->prepare(position, t, -1, -1,
      std::numeric_limits<std::size_t>::max(), &witness) == phx::mc::Insertion::kOk);
  const int32_t inserted = handle->mesh->vertex_count();
  PHX_CHECK(handle->mesh->source_position(inserted, authority));
  for (int axis = 0; axis < 3; ++axis) {
    PHX_CHECK((authority[axis] - phx::mc::Expansion(position[axis])).sign() == 0);
  }
  PHX_CHECK(handle->mesh->commit(phx::mc::InsertKind::kSubfacet, 0.0) == phx::mc::Insertion::kOk);
  PHX_CHECK(handle->mesh->witness(inserted).deviation == 0.0);
  PHX_CHECK(handle->mesh->source_position(inserted, authority));
  const Exported after = phx::mc::test::export_mesh(handle);
  for (std::size_t row = 0; row < after.face_sources.size(); ++row) {
    PHX_CHECK(handle->mesh->original_face_ancestor(
        after.faces.data() + 3 * row, after.face_sources[row]) >= 0);
  }
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), .5, 1e-14);
  PHX_CHECK(handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}

}  // namespace


int main() {
  test_exact_positive_reconstruction_sliver_is_not_silently_satisfied();
  test_cube_point_set_reaches_radius_edge_bound();
  test_refinement_is_deterministic();
  test_two_region_box_preserves_interface_and_volumes();
  test_size_field_bounds_circumradius();
  test_fixed_boundary_keeps_faces_and_reports();
  test_protecting_ball_is_respected_and_reported();
  test_budgets_refuse_cleanly();
  test_invalid_domains_are_refused();
  test_rotated_decimal_constraint_split_has_exact_original_ancestry();
  test_exact_acute_source_angle_reports_infeasible_shape_without_growth();
  test_oblique_facet_refinement_preserves_shape_with_bounded_work();
  test_independent_permuted_source_bank_uses_source_corner_topology();
  test_authored_subsegment_obeys_containing_zero_bound_facet();
  test_unsegmented_shared_diagonal_keeps_both_exact_facet_ancestors();
  test_exact_row_carrier_remains_canonical_authority_despite_parameter_rounding();
  test_tilted_decimal_bounded_segment_split_is_ancestry_backed();
  test_tilted_decimal_bound_below_certified_deviation_refuses_without_mutation();
  test_tilted_decimal_refinement_uses_bounded_ancestry();
  test_bounded_poor_shape_refines_even_when_size_is_already_met();
  test_bounded_witnesses_round_trip_and_tampering_is_refused();
  return phx::mc::test::finish("test_refine3d");
}
