//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Constrained Delaunay refinement: radius-edge and size targets on box
// domains, exact preservation of constrained faces and region volumes,
// fixed-boundary and protecting-ball evidence, budgets, determinism and
// input validation; early-exit star lookups and the worklist cavity shrink
// against rebuilt former algorithms (same answers, no more work).
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "check.hpp"
#include "facet_coverage.hpp"
#include "geometry3.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "tet_mesh.hpp"
#include "tet_mesh_fixtures.hpp"

namespace {

using phx::mc::test::Domain;
using phx::mc::test::Exported;

// Historical empirical allowances of the independent unit-box fixture oracles
// in tet_mesh_fixtures.hpp::{cell_quality,region_volume,source_area}. These do
// not change native encroachment, size or radius-edge admission thresholds.
constexpr double kRadiusEdgeOracleTolerance = 1e-9;
constexpr double kUnitBoxMeasureTolerance = 1e-12;
constexpr double kCircumradiusOracleTolerance = 1e-12;
// Historical relative allowance for the expansion-volume comparison against
// reconstruction_sliver_domain's authored determinant/6 value.
constexpr double kExactVolumeRatioTolerance = 1e-14;
// Historical absolute allowance for the analytic 1/2-volume row-carrier case.
constexpr double kRowCarrierVolumeTolerance = 1e-14;
// Historical relative allowance on the independent two-face sliver-volume
// bound in test_tilted_decimal_bounded_segment_split_is_ancestry_backed.
constexpr double kSliverVolumeRelativeTolerance = 1e-12;
// Declared physical source-row policies for decimal and dyadic fixtures,
// respectively. These are inputs whose refusal/fidelity assertions stay strict.
constexpr double kDecimalSourceBound = 1e-12;
constexpr double kDyadicSourceBound = 1e-14;

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
  const auto n = static_cast<std::size_t>(counts[PHX_MC_TET_MESH_COUNT_UNMET_RECORDS]);
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
  PHX_CHECK(run.counters[PHX_MC_TET_MESH_REFINE_CIRCUMCENTER_INSERTIONS] > 0);
  const Exported after = phx::mc::test::export_mesh(mesh);
  PHX_CHECK(worst_ratio(after) <= 2.0 + kRadiusEdgeOracleTolerance);
  PHX_CHECK(phx::mc::test::all_positive(after));
  PHX_CHECK(phx::mc::test::faces_on_sides(after, 0.0));
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), 1.0, kUnitBoxMeasureTolerance);
  for (int32_t side = 0; side < 6; ++side) {
    PHX_CHECK_NEAR(phx::mc::test::source_area(after, side), 1.0, kUnitBoxMeasureTolerance);
  }
  PHX_CHECK(handle_audit(mesh));
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
  PHX_CHECK(worst_ratio(after) <= 2.0 + kRadiusEdgeOracleTolerance);
  PHX_CHECK(phx::mc::test::faces_on_sides(after, 0.5));
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), 0.5, kUnitBoxMeasureTolerance);
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 1), 0.5, kUnitBoxMeasureTolerance);
  PHX_CHECK_NEAR(phx::mc::test::source_area(after, 6), 1.0, kUnitBoxMeasureTolerance);
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
  PHX_CHECK(largest <= 0.2 + kCircumradiusOracleTolerance);
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), 1.0, kUnitBoxMeasureTolerance);
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
  PHX_CHECK(run.counters[PHX_MC_TET_MESH_REFINE_SUBFACET_SPLITS] == 0 && run.counters[PHX_MC_TET_MESH_REFINE_SUBSEGMENT_SPLITS] == 0);
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
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), 1.0, kUnitBoxMeasureTolerance);
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
  PHX_CHECK(starved.counters[PHX_MC_TET_MESH_REFINE_WORK_UNITS] <= 50);
  const Exported still = phx::mc::test::export_mesh(mesh);
  PHX_CHECK(phx::mc::test::all_positive(still));
  PHX_CHECK_NEAR(phx::mc::test::region_volume(still, 0), 1.0, kUnitBoxMeasureTolerance);
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

// Tilted decimal coordinates have neither representable dyadic edge splits
// nor exact dyadic face barycenters; the exact-search refusal assertions below
// distinguish their ancestry-backed bounded carriers from exact subdivision.
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

bool former_exact_edge_point(const double* a, const double* b, double* point,
                             double& parameter, double preferred) {
  if (!std::isfinite(preferred) || preferred < 0.0 || preferred >= 1.0) return false;
  const auto construct = [&](double fraction, bool general) {
    constexpr double minimum =
        (phx::mc::kCoordinateMinMagnitude / phx::mc::kCoordinateMaxMagnitude) * 0x1p-53;
    if (general && fraction < minimum) return false;
    bool exact = true;
    for (int axis = 0; axis < 3; ++axis) {
      const phx::mc::Expansion value =
          general ? phx::mc::Expansion(a[axis]) +
                        phx::mc::Expansion::difference(b[axis], a[axis]).scaled(fraction)
                  : phx::mc::Expansion::product(a[axis], 1.0 - fraction) +
                        phx::mc::Expansion::product(b[axis], fraction);
      point[axis] = value.estimate();
      exact = exact && (value - phx::mc::Expansion(point[axis])).is_zero();
    }
    if (!exact || !phx::mc::coordinate_in_domain(point[0]) ||
        !phx::mc::coordinate_in_domain(point[1]) || !phx::mc::coordinate_in_domain(point[2]) ||
        !phx::mc::geometry::collinear(a, b, point) ||
        std::equal(point, point + 3, a) || std::equal(point, point + 3, b)) {
      return false;
    }
    parameter = fraction;
    return true;
  };
  if (preferred != 0.0 && construct(preferred, true)) return true;
  struct Candidate { double fraction; int level; };
  std::array<Candidate, 129> candidates{};
  std::size_t count = 0;
  for (int level = 1; level <= 8; ++level) {
    for (int numerator = 1; numerator < (1 << level); numerator += 2) {
      const double fraction = std::ldexp(static_cast<double>(numerator), -level);
      if (fraction >= 0.25 && fraction <= 0.75) candidates[count++] = {fraction, level};
    }
  }
  PHX_CHECK(count == candidates.size());
  if (preferred != 0.0) {
    std::sort(candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
      const double first = std::abs(a.fraction - preferred);
      const double second = std::abs(b.fraction - preferred);
      if (first != second) return first < second;
      if (a.level != b.level) return a.level < b.level;
      return a.fraction < b.fraction;
    });
  }
  for (const Candidate& candidate : candidates) {
    if ((preferred == 0.0 || candidate.fraction != preferred) &&
        construct(candidate.fraction, false)) return true;
  }
  return false;
}

void test_dyadic_nonrepresentability_filter_preserves_exact_constructions() {
  for (const int power : {-115, 0, 118}) {
    for (const double sign : {-1.0, 1.0}) {
      Domain domain = tilted_decimal_domain();
      domain.tets = {0, 1, 2, 3};
      for (double& value : domain.points) value = sign * std::ldexp(value, power);
      if (phx::mc::orient3d(domain.points.data(), domain.points.data() + 3,
                            domain.points.data() + 6, domain.points.data() + 9) < 0) {
        std::swap(domain.tets[0], domain.tets[1]);
      }
      phx_mc_tet_mesh* handle = declared_mesh(
          domain, kDecimalSourceBound, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
      const Exported before = phx::mc::test::export_mesh(handle);
      for (const double preferred : {0.0, 0.375, 0.61,
                                     std::numeric_limits<double>::denorm_min()}) {
        double expected[3], actual[3];
        double expected_fraction = -1.0, fraction = -1.0;
        const bool former = former_exact_edge_point(
            domain.points.data(), domain.points.data() + 3,
            expected, expected_fraction, preferred);
        handle->mesh->set_work_limit(1 << 20);
        const bool constructed = handle->mesh->construct_edge_point(
            0, 1, actual, &fraction, preferred);
        PHX_CHECK(constructed == former && !handle->mesh->work_exhausted());
        if (constructed) {
          PHX_CHECK(fraction == expected_fraction);
          PHX_CHECK(std::equal(actual, actual + 3, expected));
        }
      }
      PHX_CHECK(phx::mc::test::export_mesh(handle) == before);
      phx_mc_tet_mesh_free(handle);
    }
  }
  // This existing rotated edge has an exact non-midpoint candidate. It must
  // not be discarded by an unsupported parity/exponent shortcut.
  Domain exact = tilted_decimal_domain();
  exact.tets = {0, 1, 2, 3};
  exact.points = {-1.6, 1.4, 0.9, 0.6, 0.5, -0.3, 0.2, -1.0, 1.0, 1.0, 2.0, 2.0};
  if (phx::mc::orient3d(exact.points.data(), exact.points.data() + 3,
                        exact.points.data() + 6, exact.points.data() + 9) < 0) {
    std::swap(exact.tets[0], exact.tets[1]);
  }
  phx_mc_tet_mesh* handle = declared_mesh(
      exact, kDecimalSourceBound, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  for (const double preferred : {0.0, 0.375, 0.61}) {
    double expected[3], actual[3], expected_fraction = -1.0, fraction = -1.0;
    PHX_CHECK(former_exact_edge_point(exact.points.data(), exact.points.data() + 3,
                                     expected, expected_fraction, preferred));
    PHX_CHECK(handle->mesh->construct_edge_point(0, 1, actual, &fraction, preferred));
    PHX_CHECK(fraction == expected_fraction && std::equal(actual, actual + 3, expected));
  }
  for (const double invalid : {std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN(), -0.5, 1.0}) {
    double unused[3];
    const int64_t before = handle->mesh->work();
    PHX_CHECK(!handle->mesh->construct_edge_point(0, 1, unused, nullptr, invalid));
    PHX_CHECK(handle->mesh->work() == before);
  }
  phx_mc_tet_mesh_free(handle);
}

void test_filtered_source_construction_keeps_witness_and_exact_work_boundary() {
  const Domain domain = tilted_decimal_domain();
  phx_mc_tet_mesh* probe = declared_mesh(
      domain, kDecimalSourceBound, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  double nonbinary[3];
  const int64_t first = probe->mesh->work();
  PHX_CHECK(!probe->mesh->construct_edge_point(0, 1, nonbinary));
  PHX_CHECK(probe->mesh->work() - first == 2 * phx::mc::TetWorkCost::kCoordinates);
  double position[3], parameter = 0.0, parameters[2], deviation = 0.0;
  int32_t constructed = 0, entity = -1;
  int8_t stratum = 0;
  const int64_t before = probe->mesh->work();
  PHX_CHECK(phx_mc_tet_mesh_construct_edge_split(
      probe, 0, 1, 0.0, 1 << 20, position, &parameter, &constructed,
      &stratum, &entity, parameters, &deviation) == PHX_MC_OK);
  PHX_CHECK(constructed == 1 && parameter == 0.5);
  const int64_t required = probe->mesh->work() - before;
  phx::mc::SourceWitness reference;
  double expected[3];
  PHX_CHECK(probe->mesh->construct_source_split(0, 1, .5, expected, reference) ==
            phx::mc::Insertion::kOk);
  PHX_CHECK(std::equal(position, position + 3, expected));
  PHX_CHECK(stratum == static_cast<int8_t>(reference.stratum) && entity == reference.entity);
  PHX_CHECK(parameters[0] == reference.parameters[0] &&
            parameters[1] == reference.parameters[1] && deviation == reference.deviation);
  phx_mc_tet_mesh_free(probe);
  for (const int64_t allowance : {required, required - 1}) {
    phx_mc_tet_mesh* handle = declared_mesh(
        domain, kDecimalSourceBound, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
    const Exported old = phx::mc::test::export_mesh(handle);
    const int64_t start = handle->mesh->work();
    constructed = 0;
    const int32_t status = phx_mc_tet_mesh_construct_edge_split(
        handle, 0, 1, 0.0, allowance, position, &parameter, &constructed,
        &stratum, &entity, parameters, &deviation);
    PHX_CHECK(status == (allowance == required ? PHX_MC_OK : PHX_MC_CAPACITY_EXCEEDED));
    PHX_CHECK(constructed == (allowance == required ? 1 : 0));
    PHX_CHECK(handle->mesh->work() - start <= allowance);
    PHX_CHECK(phx::mc::test::export_mesh(handle) == old);
    phx_mc_tet_mesh_free(handle);
  }
}

void test_edge_filter_unsupported_coordinate_inputs_stay_refused() {
  const Domain source = tilted_decimal_domain();
  for (const double value : {std::numeric_limits<double>::denorm_min(),
                             std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN(), 0x1p121, 0x1p-121}) {
    Domain invalid = source;
    invalid.points[0] = value;
    int32_t status = PHX_MC_OK;
    phx_mc_tet_mesh* handle = phx::mc::test::create_mesh(
        invalid, PHX_MC_TET_MESH_BOUNDARY_CONFORMING, nullptr, 256, 4096, &status);
    PHX_CHECK(handle == nullptr && status != PHX_MC_OK);
    phx_mc_tet_mesh_free(handle);
  }
}

void test_inspected_split_authenticates_the_existing_complete_native_cavity() {
  for (const bool exact : {false, true}) {
    Domain domain = tilted_decimal_domain();
    if (exact) {
      domain.points = {-1.6, 1.4, 0.9, 0.6, 0.5, -0.3, 0.2, -1.0, 1.0, 1.0, 2.0, 2.0};
      domain.tets = {0, 1, 2, 3};
      if (phx::mc::orient3d(domain.points.data(), domain.points.data() + 3,
                            domain.points.data() + 6, domain.points.data() + 9) < 0) {
        std::swap(domain.tets[0], domain.tets[1]);
      }
    }
    phx_mc_tet_mesh* direct = declared_mesh(
        domain, kDecimalSourceBound, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
    phx_mc_tet_mesh* inspected = declared_mesh(
        domain, kDecimalSourceBound, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
    const Exported old = phx::mc::test::export_mesh(inspected);
    double point[3], parameter = 0.0, parameters[2], deviation = 0.0;
    int32_t constructed = 0, entity = -1;
    int8_t stratum = 0;
    PHX_CHECK(phx_mc_tet_mesh_construct_edge_split(
        inspected, 0, 1, 0.0, 1 << 20, point, &parameter, &constructed,
        &stratum, &entity, parameters, &deviation) == PHX_MC_OK && constructed == 1);
    constexpr int64_t capacity = 512;
    std::array<int32_t, 4 * capacity> removed{}, proposed{};
    int64_t counts[4] = {};
    PHX_CHECK(phx_mc_tet_mesh_inspect_edge_insertion(
        inspected, 0, 1, point, point, parameter, capacity, 1 << 20,
        removed.data(), proposed.data(), counts) == PHX_MC_OK);
    PHX_CHECK(counts[0] == 1 && counts[1] == 2);
    PHX_CHECK(phx::mc::test::export_mesh(inspected) == old);
    int32_t inserted = -1;
    phx_mc_tet_mesh_commit_edge_insertion_request request{
        inspected, 0, 1, point, point, parameter, capacity,
        static_cast<int32_t>(counts[2]), counts[3], counts[0], removed.data(),
        counts[1], proposed.data(), 0.0, 1 << 20, &inserted};
    const int32_t correct = proposed[0];
    proposed[0] = correct == 0 ? 1 : 0;
    PHX_CHECK(phx_mc_tet_mesh_commit_edge_insertion(&request) == PHX_MC_OK);
    PHX_CHECK(inserted == -1 && phx::mc::test::export_mesh(inspected) == old);
    proposed[0] = correct;
    PHX_CHECK(phx_mc_tet_mesh_commit_edge_insertion(&request) == PHX_MC_OK);
    PHX_CHECK(inserted == 4);
    int32_t original_inserted = -1;
    PHX_CHECK(phx_mc_tet_mesh_split_edge(
        direct, 0, 1, point, 0.0, parameter, 1 << 20, &original_inserted) == PHX_MC_OK);
    PHX_CHECK(original_inserted == inserted);
    PHX_CHECK(phx::mc::test::export_mesh(inspected) == phx::mc::test::export_mesh(direct));
    PHX_CHECK(handle_audit(inspected));
    phx_mc_tet_mesh_free(direct);
    phx_mc_tet_mesh_free(inspected);
  }
}

void test_one_prepared_owner_is_movable_single_use_and_exception_safe() {
  const Domain domain = tilted_decimal_domain();
  auto* handle = declared_mesh(domain, kDecimalSourceBound,
                                PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  auto& mesh = *handle->mesh;
  const auto before = phx::mc::test::export_mesh(handle);
  const std::array<int32_t, 1> removed{0};
  const auto& old = mesh.tet(0);
  const std::array<std::array<int32_t, 4>, 1> proposed{{{old.v[0], old.v[1], old.v[2], old.v[3]}}};
  {
    auto canceled = mesh.prepare_replacement(removed, proposed);
    PHX_CHECK(canceled.status() == phx::mc::EditStatus::kOk);
  }
  PHX_CHECK(phx::mc::test::export_mesh(handle) == before);
  try {
    auto failed = mesh.prepare_replacement(removed, proposed);
    PHX_CHECK(failed.status() == phx::mc::EditStatus::kOk);
    throw 7;
  } catch (int value) {
    PHX_CHECK(value == 7);
  }
  PHX_CHECK(phx::mc::test::export_mesh(handle) == before && !mesh.broken());
  {
    auto stale = mesh.prepare_replacement(removed, proposed);
    auto current = mesh.prepare_replacement(removed, proposed);
    PHX_CHECK(stale.commit() == phx::mc::EditStatus::kBoundaryMismatch);
    auto moved = std::move(current);
    PHX_CHECK(current.commit() == phx::mc::EditStatus::kBoundaryMismatch);
    PHX_CHECK(moved.commit() == phx::mc::EditStatus::kOk);
    PHX_CHECK(moved.commit() == phx::mc::EditStatus::kBoundaryMismatch);
  }
  PHX_CHECK(phx::mc::test::export_mesh(handle) == before && !mesh.broken());
  phx_mc_tet_mesh_free(handle);
}

void test_retained_split_lifecycle_and_exact_resource_boundaries() {
  const Domain domain = tilted_decimal_domain();
  const auto source_readout = [](phx_mc_tet_mesh* handle) {
    const auto exported = phx::mc::test::export_mesh(handle);
    int64_t count = 0;
    PHX_CHECK(phx_mc_tet_mesh_source_evidence(handle, 0, &count, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == PHX_MC_OK);
    std::vector<int64_t> offsets(exported.face_sources.size() + 1);
    std::vector<int32_t> rows(static_cast<std::size_t>(count));
    std::vector<int32_t> segments(exported.segment_sources.size() + 1);
    std::vector<int8_t> strata(exported.dimension.size());
    std::vector<int32_t> entities(strata.size());
    std::vector<double> parameters(2 * strata.size()), deviations(strata.size());
    double bounds[2];
    PHX_CHECK(phx_mc_tet_mesh_source_evidence(handle, count, &count, bounds,
        offsets.data(), rows.data(), segments.data(), strata.data(), entities.data(),
        parameters.data(), deviations.data()) == PHX_MC_OK);
    std::vector<uint64_t> result;
    const auto append = [&](const auto& values) {
      for (const auto value : values) result.push_back(static_cast<uint64_t>(value));
    };
    append(offsets); append(rows); append(segments); append(strata); append(entities);
    for (double value : bounds) result.push_back(std::bit_cast<uint64_t>(value));
    for (double value : parameters) result.push_back(std::bit_cast<uint64_t>(value));
    for (double value : deviations) result.push_back(std::bit_cast<uint64_t>(value));
    for (const auto& face : handle->mesh->original_faces()) {
      for (int32_t value : face) result.push_back(static_cast<uint64_t>(value));
      for (int k = 0; k < 3; ++k) {
        for (int axis = 0; axis < 3; ++axis) {
          result.push_back(std::bit_cast<uint64_t>(handle->mesh->original_point(face[k])[axis]));
        }
      }
    }
    return result;
  };
  std::size_t scratch = 0;
  std::vector<uint64_t> expected_source;
  Exported expected;
  for (int mode = 0; mode < 8; ++mode) {
    auto* handle = declared_mesh(domain, kDecimalSourceBound,
                                  PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
    const auto before = phx::mc::test::export_mesh(handle);
    const auto old_source = source_readout(handle);
    double point[3], fraction, parameters[2], deviation;
    int32_t constructed = 0, entity;
    int8_t stratum;
    PHX_CHECK(phx_mc_tet_mesh_construct_edge_split(handle, 0, 1, 0.0, INT64_MAX,
        point, &fraction, &constructed, &stratum, &entity, parameters, &deviation) == PHX_MC_OK);
    PHX_CHECK(constructed == 1);
    constexpr int64_t capacity = 512;
    std::array<int32_t, 4 * capacity> removed{}, proposed{};
    int64_t counts[4]{};
    PHX_CHECK(phx_mc_tet_mesh_inspect_edge_insertion(handle, 0, 1, point, point,
        fraction, capacity, INT64_MAX, removed.data(), proposed.data(), counts) == PHX_MC_OK);
    PHX_CHECK(counts[0] > 0 && counts[1] > 0);
    PHX_CHECK(phx::mc::test::export_mesh(handle) == before);
    PHX_CHECK(source_readout(handle) == old_source);
    auto kind = phx::mc::InsertKind::kInterior;
    PHX_CHECK(handle->mesh->reuse_edge_insertion(0, 1, point, point, fraction, capacity, kind));
    PHX_CHECK(!handle->mesh->reuse_edge_insertion(1, 0, point, point, fraction, capacity, kind));
    PHX_CHECK(!handle->mesh->reuse_edge_insertion(0, 1, point, point,
        std::nextafter(fraction, 1.0), capacity, kind));
    PHX_CHECK(!handle->mesh->reuse_edge_insertion(0, 1, point, point, fraction, capacity - 1, kind));
    double changed[3] = {point[0], point[1], point[2]};
    changed[0] = std::nextafter(changed[0], 1.0);
    PHX_CHECK(!handle->mesh->reuse_edge_insertion(0, 1, point, changed, fraction, capacity, kind));
    PHX_CHECK(handle->mesh->reuse_edge_insertion(0, 1, point, point, fraction, capacity, kind));
    if (mode == 1) handle->mesh->abandon();
    if (mode == 2) {
      auto empty = handle->mesh->prepare_replacement({}, {});
      PHX_CHECK(empty.status() != phx::mc::EditStatus::kOk);
    }
    if (mode == 3) {
      PHX_CHECK(handle->mesh->prepare_edge_insertion(0, 1, point, point, fraction,
                                                     capacity, kind) == phx::mc::Insertion::kOk);
    }
    if (mode >= 1 && mode <= 3) {
      PHX_CHECK(!handle->mesh->reuse_edge_insertion(0, 1, point, point, fraction, capacity, kind));
    }
    const int64_t exact_work = 4 * (counts[0] + counts[1]);
    const int64_t work_before = handle->mesh->work();
    int32_t inserted = -1;
    phx_mc_tet_mesh_commit_edge_insertion_request request{
        handle, 0, 1, point, point, fraction, capacity, static_cast<int32_t>(counts[2]),
        counts[3], counts[0], removed.data(), counts[1], proposed.data(), 0.0,
        mode == 0 ? exact_work : mode == 4 ? exact_work - 1 : INT64_MAX, &inserted};
    const bool refused = mode == 4 || mode == 7;
    {
      phx::mc::MemoryBudgetWindow window(mode == 6 ? scratch : mode == 7 ? scratch - 1 : 1 << 24,
                                          handle->mesh->memory_owner());
      PHX_CHECK(phx_mc_tet_mesh_commit_edge_insertion(&request) ==
                   (refused ? PHX_MC_CAPACITY_EXCEEDED : PHX_MC_OK));
      if (mode == 5) scratch = static_cast<std::size_t>(window.evidence()[2]);
    }
    if (refused) {
      PHX_CHECK(inserted == -1 && phx::mc::test::export_mesh(handle) == before);
      PHX_CHECK(source_readout(handle) == old_source);
      PHX_CHECK(!handle->mesh->broken());
      request.work_limit = INT64_MAX;
      PHX_CHECK(phx_mc_tet_mesh_commit_edge_insertion(&request) == PHX_MC_OK);
    }
    PHX_CHECK(inserted == 4);
    if (mode == 0) {
      PHX_CHECK(handle->mesh->work() - work_before == exact_work);
      expected = phx::mc::test::export_mesh(handle);
      expected_source = source_readout(handle);
      auto* fresh = declared_mesh(domain, kDecimalSourceBound,
                                   PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
      request.mesh = fresh;
      request.work_limit = INT64_MAX;
      const int64_t fresh_work = fresh->mesh->work();
      PHX_CHECK(phx_mc_tet_mesh_commit_edge_insertion(&request) == PHX_MC_OK && inserted == 4);
      PHX_CHECK(fresh->mesh->work() - fresh_work > exact_work);
      PHX_CHECK(phx::mc::test::export_mesh(fresh) == expected);
      PHX_CHECK(source_readout(fresh) == expected_source);
      phx_mc_tet_mesh_free(fresh);
      request.mesh = handle;
    } else {
      PHX_CHECK(phx::mc::test::export_mesh(handle) == expected);
      PHX_CHECK(source_readout(handle) == expected_source);
      if (mode <= 3) PHX_CHECK(handle->mesh->work() - work_before > exact_work);
    }
    // A consumed or stale plan never applies twice.
    PHX_CHECK(phx_mc_tet_mesh_commit_edge_insertion(&request) == PHX_MC_OK && inserted == -1);
    PHX_CHECK(phx::mc::test::export_mesh(handle) == expected);
    phx_mc_tet_mesh_free(handle);
  }
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
    phx::mc::NativeVector<int32_t> contributors;
    PHX_CHECK(mesh.face_ancestry(exported.faces.data() + 3 * f,
                                  exported.face_sources[f], contributors));
    PHX_CHECK(contributors.size() == 1 && contributors.front() == exported.face_sources[f]);
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
  PHX_CHECK(declare_source(handle, domain, kDyadicSourceBound) == PHX_MC_OK);
  double midpoint[3];
  PHX_CHECK(mesh.construct_edge_point(0, 1, midpoint));
  const double rounded_midpoint[3] = {(p[0][0] + p[1][0]) * 0.5,
                                     (p[0][1] + p[1][1]) * 0.5,
                                     (p[0][2] + p[1][2]) * 0.5};
  PHX_CHECK(!phx::mc::geometry::collinear(p[0], p[1], rounded_midpoint));
  PHX_CHECK(phx::mc::geometry::collinear(p[0], p[1], midpoint));
  int32_t inserted = -1;
  PHX_CHECK(mesh.split_edge(0, 1, midpoint, 0.0, inserted) == phx::mc::Insertion::kOk);
  PHX_CHECK(mesh.original_segment_ancestor(0, inserted, 0) == 0);
  PHX_CHECK(mesh.original_segment_ancestor(inserted, 1, 0) == 0);
  const Exported after = phx::mc::test::export_mesh(handle);
  for (std::size_t i = 0; i < after.face_sources.size(); ++i) {
    const int32_t source = after.face_sources[i];
    const int32_t* face = after.faces.data() + 3 * i;
    phx::mc::NativeVector<int32_t> contributors;
    PHX_CHECK(mesh.face_ancestry(face, source, contributors));
    PHX_CHECK(contributors.size() == 1 && contributors.front() == source);
    const int32_t* original = domain.faces.data() + 3 * source;
    for (int k = 0; k < 3; ++k) {
      PHX_CHECK(phx::mc::orient3d(domain.points.data() + 3 * original[0],
                                  domain.points.data() + 3 * original[1],
                                  domain.points.data() + 3 * original[2],
                                  mesh.point(face[k])) == 0);
    }
  }
  PHX_CHECK(mesh.achieved_source_error_bound() == 0.0);
  PHX_CHECK(mesh.requested_source_tolerance() == kDyadicSourceBound);
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
  PHX_CHECK(starved.counters[PHX_MC_TET_MESH_REFINE_WORK_UNITS] <= 50);
  PHX_CHECK(phx::mc::test::export_mesh(mesh) == before);
  PHX_CHECK(handle_audit(mesh));

  constexpr int64_t work_limit = 10000;
  const Run run = refine(mesh, 2.0, 1 << 20, nullptr, work_limit);
  PHX_CHECK(run.status == PHX_MC_OK);
  PHX_CHECK(run.counters[PHX_MC_TET_MESH_REFINE_WORK_UNITS] <= work_limit);
  const Exported after = phx::mc::test::export_mesh(mesh);
  PHX_CHECK(worst_ratio(after) <= 2.0 + kRadiusEdgeOracleTolerance);
  PHX_CHECK(phx::mc::test::all_positive(after));
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), 1.0, kUnitBoxMeasureTolerance);
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
    PHX_CHECK_NEAR(phx::mc::test::source_area(pulled_back, source), 1.0, kUnitBoxMeasureTolerance);
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
  PHX_CHECK(counts[PHX_MC_TET_MESH_COUNT_UNMET_RECORDS] == 1);
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
  PHX_CHECK_NEAR(values[4] / (8.987998638190837e-22 / 6.0), 1.0, kExactVolumeRatioTolerance);
  PHX_CHECK(slivers == 1 && values[0] < 5.0);
  const Run run = refine(mesh, 2.0, 8, nullptr, 100000);
  PHX_CHECK(run.status == PHX_MC_REFINEMENT_LIMIT);
  PHX_CHECK(phx::mc::test::export_mesh(mesh) == before);
  int64_t counts[PHX_MC_TET_MESH_COUNTS] = {};
  PHX_CHECK(phx_mc_tet_mesh_counts(mesh, counts) == PHX_MC_OK && counts[PHX_MC_TET_MESH_COUNT_UNMET_RECORDS] == 1);
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
  const std::vector<double> bounds(6, kDecimalSourceBound);
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
      original, kDecimalSourceBound, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
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
  const double segment_bounds[2] = {kDecimalSourceBound, kDecimalSourceBound};
  const double face_bounds[4] = {0, kDecimalSourceBound, kDecimalSourceBound, kDecimalSourceBound};
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
  phx_mc_tet_mesh* handle = declared_mesh(domain, kDecimalSourceBound, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
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
    phx::mc::NativeVector<int32_t> contributors;
    PHX_CHECK(handle->mesh->face_ancestry(
        after.faces.data() + 3 * i, after.face_sources[i], contributors));
    PHX_CHECK(!contributors.empty());
  }
  PHX_CHECK(handle_audit(handle));
  double bounds[6] = {kDecimalSourceBound, kDecimalSourceBound, kDecimalSourceBound,
                     kDecimalSourceBound, kDecimalSourceBound, kDecimalSourceBound};
  PHX_CHECK(witness.entity == 0 || witness.entity == 1);
  bounds[witness.entity == 0 ? 1 : 0] = 0.0;
  phx::mc::SourceComplex source;
  source.face_count = 6;
  source.faces = domain.faces.data();
  source.face_sources = domain.face_sources.data();
  source.face_tolerances = bounds;
  const double segment_bounds[8] = {kDecimalSourceBound, kDecimalSourceBound, kDecimalSourceBound,
                                    kDecimalSourceBound, kDecimalSourceBound, kDecimalSourceBound,
                                    kDecimalSourceBound, kDecimalSourceBound};
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
  phx_mc_tet_mesh* handle = declared_mesh(domain, kDecimalSourceBound, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
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
  PHX_CHECK(witness.deviation > 0.0 && witness.deviation <= kDecimalSourceBound);
  PHX_CHECK(!phx::mc::geometry::collinear(a, b, carrier) && within_line(a, b, carrier, witness.deviation));
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
  check_source_fidelity(handle, domain, kDecimalSourceBound);
  // The region volume moves by at most the slivers over the two source
  // faces through the split edge: |dV| <= (A(0,1,2) + A(0,1,3)) d / 3.
  const double changed =
      std::fabs((six_volume(after) - six_volume(before)).estimate()) / 6.0;
  const double bound =
      (phx::mc::test::source_area(before, 0) + phx::mc::test::source_area(before, 1)) *
      witness.deviation / 3.0;
  PHX_CHECK(changed <= bound * (1.0 + kSliverVolumeRelativeTolerance));
  phx_mc_tet_mesh_free(handle);
}

void test_tilted_decimal_bound_below_certified_deviation_refuses_without_mutation() {
  const Domain domain = tilted_decimal_domain();
  phx_mc_tet_mesh* probe = declared_mesh(domain, kDecimalSourceBound, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
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
  phx_mc_tet_mesh* fixed = declared_mesh(domain, kDecimalSourceBound, PHX_MC_TET_MESH_BOUNDARY_FIXED);
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
  PHX_CHECK(run.counters[PHX_MC_TET_MESH_REFINE_SUBSEGMENT_SPLITS] > 0 && run.counters[PHX_MC_TET_MESH_REFINE_SUBFACET_SPLITS] > 0);
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
  PHX_CHECK(run.counters[PHX_MC_TET_MESH_REFINE_SUBFACET_SPLITS] + run.counters[PHX_MC_TET_MESH_REFINE_SUBSEGMENT_SPLITS] > 0);
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
    phx::mc::NativeVector<int32_t> contributors;
    PHX_CHECK(handle->mesh->face_ancestry(
        after.faces.data() + 3 * row, after.face_sources[row], contributors));
    PHX_CHECK(!contributors.empty());
  }
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), .5, kRowCarrierVolumeTolerance);
  PHX_CHECK(handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}

void test_logical_facet_coverage_retains_all_primitive_contributors() {
  const double points[] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
  const double crossing[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  const std::array<int32_t, 4> faces[] = {{0, 1, 2, 7}, {0, 2, 3, 7}};
  const int32_t rows[] = {0, 1};
  phx::mc::NativeVector<int32_t> contributors;
  PHX_CHECK(phx::mc::exact_facet_coverage(crossing, points, faces, rows, contributors));
  PHX_CHECK(contributors.size() == 2 && contributors[0] == 0 && contributors[1] == 1);
  const int32_t missing[] = {0};
  PHX_CHECK(!phx::mc::exact_facet_coverage(crossing, points, faces, missing, contributors));
  PHX_CHECK(contributors.empty());
  const std::array<int32_t, 4> overlapping[] =
      {{0, 1, 2, 7}, {0, 2, 3, 7}, {0, 1, 2, 7}};
  const int32_t overlap_rows[] = {0, 1, 2};
  PHX_CHECK(!phx::mc::exact_facet_coverage(
      crossing, points, overlapping, overlap_rows, contributors));
  const std::array<int32_t, 4> reversed[] = {{0, 1, 2, 7}, {3, 2, 0, 7}};
  PHX_CHECK(!phx::mc::exact_facet_coverage(crossing, points, reversed, rows, contributors));
  // All subject corners belong to the union of these three small triangles,
  // but the interior does not. Corner containment is not a union proof.
  const double gaps[] = {0,0,0, .25,0,0, 0,.25,0,
                        1,0,0, .75,0,0, 1,.25,0,
                        0,1,0, .25,.75,0, 0,.75,0};
  const std::array<int32_t, 4> islands[] = {{0,1,2,7}, {3,5,4,7}, {6,8,7,7}};
  const int32_t island_rows[] = {0, 1, 2};
  PHX_CHECK(!phx::mc::exact_facet_coverage(crossing, gaps, islands, island_rows, contributors));
}

// The single-row and partition predicate proofs must return exactly the
// arrangement result: subjects inside, across, touching and outside coplanar
// rows of a logical facet with noncoplanar, duplicated, reversed and
// degenerate rows, edge-conforming convex and nonconvex partitions, a
// T-junction overlap and a notch gap.
void test_single_row_facet_coverage_matches_arrangement() {
  const double points[] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 2, 0, 0, 2, 1, 0,
                           0, 0, 1, 1, 0, 1, .5, .5, 0};
  const std::array<int32_t, 4> faces[] = {
      {0, 1, 2, 7}, {0, 2, 3, 7}, {1, 4, 5, 7}, {1, 5, 2, 7},  // two coplanar quads
      {0, 1, 7, 7}, {0, 7, 6, 7},                              // noncoplanar wall
      {0, 1, 2, 7}, {3, 2, 0, 7}, {0, 8, 2, 7},                // duplicate, reversed, collinear
      {0, 4, 2, 7}, {2, 4, 5, 7}};                             // fan over the long edge
  const std::vector<std::vector<int32_t>> banks = {
      {0, 1, 2, 3}, {0, 1, 2, 3, 4, 5}, {0, 2, 4, 5}, {0, 1, 2, 3, 6}, {0, 7, 2, 3},
      {0, 1, 8}, {4, 5}, {0}, {9, 10, 1}, {9, 2, 3}, {0, 1, 2}, {9, 10}, {1, 9, 4}};
  int decided = 0;
  for (int a = 0; a <= 16; ++a) {
    for (int b = 0; b <= 8; ++b) {
      for (const auto& [du, dv, eu, ev] :
           {std::array<int, 4>{1, 0, 0, 1}, std::array<int, 4>{3, 1, 1, 2},
            std::array<int, 4>{0, 1, 1, 0}, std::array<int, 4>{8, 0, 0, 8},
            std::array<int, 4>{-2, 3, 1, 1}}) {
        const double subject[] = {a / 8.0,        b / 8.0,        0,
                                  (a + du) / 8.0, (b + dv) / 8.0, 0,
                                  (a + eu) / 8.0, (b + ev) / 8.0, 0};
        for (const auto& bank : banks) {
          phx::mc::NativeVector<int32_t> fast, reference;
          const bool fast_result =
              phx::mc::exact_facet_coverage(subject, points, faces, bank, fast);
          const bool reference_result = phx::mc::exact_facet_coverage_by_arrangement(
              subject, points, faces, bank, reference);
          PHX_CHECK(fast_result == reference_result);
          PHX_CHECK(std::equal(fast.begin(), fast.end(), reference.begin(), reference.end()));
          decided += fast_result ? 1 : 0;
        }
      }
    }
  }
  PHX_CHECK(decided > 0);
}

void test_logical_facet_retriangulation_split_relocation_and_collapse() {
  Domain carrier;
  carrier.points = {0,0,0, 1,0,0, 1,1,0, 0,1,0, .5,.5,1};
  carrier.tets = {0,1,3,4, 1,2,3,4};
  for (int offset : {0,4}) {
    const int32_t* v = carrier.tets.data() + offset;
    if (phx::mc::orient3d(carrier.points.data()+3*v[0], carrier.points.data()+3*v[1],
                          carrier.points.data()+3*v[2], carrier.points.data()+3*v[3]) < 0)
      std::swap(carrier.tets[offset], carrier.tets[offset+1]);
  }
  carrier.regions = {0,0};
  carrier.faces = {0,1,3, 1,2,3, 0,1,4, 1,2,4, 2,3,4, 0,3,4};
  carrier.face_sources = {0,0,2,3,4,5};
  carrier.segments = {0,1, 1,2, 2,3, 0,3, 0,4, 1,4, 2,4, 3,4};
  carrier.segment_sources = {0,1,2,3,4,5,6,7};
  auto* handle = phx::mc::test::create_mesh(
      carrier, PHX_MC_TET_MESH_BOUNDARY_CONFORMING, nullptr, 32, 128);
  PHX_CHECK(handle != nullptr);
  const int32_t original_faces[] = {0,1,2, 0,2,3, 0,1,4, 1,2,4, 2,3,4, 0,3,4};
  phx::mc::SourceComplex source;
  source.face_count = 6;
  source.faces = original_faces;
  source.face_sources = carrier.face_sources.data();
  source.segment_count = 8;
  source.segments = carrier.segments.data();
  source.segment_sources = carrier.segment_sources.data();
  PHX_CHECK(handle->mesh->build(5, carrier.points.data(), 2, carrier.tets.data(),
      carrier.regions.data(), 6, carrier.faces.data(), carrier.face_sources.data(),
      8, carrier.segments.data(), carrier.segment_sources.data(), nullptr, &source) == PHX_MC_OK);
  // Constructor verification already earned the complete canonical-export
  // ancestry bank, including both contributors of the crossed bottom faces.
  {
    const phx::mc::MemoryScope memory_scope(handle->mesh->memory_owner());
    phx::mc::NativeVector<std::array<int32_t, 4>> canonical;
    phx::mc::NativeVector<int64_t> constructor_offsets;
    phx::mc::NativeVector<int32_t> constructor_rows;
    handle->mesh->collect_faces(canonical);
    const bool seeded = handle->mesh->cached_face_ancestry(
        canonical, constructor_offsets, constructor_rows);
    PHX_CHECK(seeded);
    if (!seeded) {
      phx_mc_tet_mesh_free(handle);
      return;
    }
    PHX_CHECK(constructor_offsets.size() == canonical.size() + 1);
    PHX_CHECK(constructor_rows.size() == 8);
    for (std::size_t i = 0; i < canonical.size(); ++i) {
      phx::mc::NativeVector<int32_t> expected;
      PHX_CHECK(handle->mesh->face_ancestry(
          canonical[i].data(), canonical[i][3], expected));
      PHX_CHECK(static_cast<int64_t>(expected.size()) ==
                constructor_offsets[i + 1] - constructor_offsets[i]);
      PHX_CHECK(std::equal(expected.begin(), expected.end(),
                          constructor_rows.begin() + constructor_offsets[i]));
    }
  }
  void* constructor_execution = nullptr;
  PHX_CHECK(phx_mc_execution_begin(UINT64_MAX, 0, UINT64_MAX, 1 << 20,
      INFINITY, nullptr, &constructor_execution) == PHX_MC_OK);
  int64_t constructor_count = -1;
  PHX_CHECK(phx_mc_tet_mesh_source_evidence(handle, 0, &constructor_count, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == PHX_MC_OK);
  uint64_t constructor_work[PHX_MC_EXECUTION_COUNTERS];
  uint64_t constructor_memory[PHX_MC_EXECUTION_MEMORY_VALUES];
  double constructor_seconds;
  PHX_CHECK(phx_mc_execution_end(constructor_execution, constructor_work,
      constructor_memory, &constructor_seconds) == PHX_MC_OK);
  PHX_CHECK(constructor_count == 8);
  PHX_CHECK(constructor_work[PHX_MC_EXECUTION_SOURCE_QUERIES] == 0);
  const uint64_t required_work = constructor_work[PHX_MC_EXECUTION_CONSTRUCTION_WORK];
  PHX_CHECK(required_work > 0);
  for (uint64_t budget : {required_work, required_work - 1}) {
    PHX_CHECK(phx_mc_execution_begin(budget, 0, UINT64_MAX, 1 << 20,
        INFINITY, nullptr, &constructor_execution) == PHX_MC_OK);
    int64_t count = -1;
    const int32_t status = phx_mc_tet_mesh_source_evidence(handle, 0, &count,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    PHX_CHECK(status == (budget == required_work ? PHX_MC_OK : PHX_MC_CAPACITY_EXCEEDED));
    PHX_CHECK(count == (budget == required_work ? 8 : -1));
    PHX_CHECK(phx_mc_execution_end(constructor_execution, constructor_work,
        constructor_memory, &constructor_seconds) == status);
  }
  phx::mc::NativeVector<int32_t> contributors;
  PHX_CHECK(handle->mesh->face_ancestry(carrier.faces.data(), 0, contributors));
  PHX_CHECK(contributors.size() == 2 && contributors[0] == 0 && contributors[1] == 1);
  PHX_CHECK(!handle->mesh->face_ancestry(carrier.faces.data(), 2, contributors));
  int64_t required = -1;
  PHX_CHECK(phx_mc_tet_mesh_source_evidence(handle, 0, &required, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == PHX_MC_OK);
  PHX_CHECK(required == 8);
  double values[2] = {-44,-44}, parameters[10], deviations[5];
  int64_t offsets[7];
  std::fill_n(offsets, 7, -44);
  int32_t rows[8], segments[8], entities[5];
  int8_t strata[5];
  PHX_CHECK(phx_mc_tet_mesh_source_evidence(handle, required-1, &required, values,
      offsets, rows, segments, strata, entities, parameters, deviations) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(values[0] == -44 && values[1] == -44 && offsets[0] == -44 && offsets[6] == -44);
  PHX_CHECK(phx_mc_tet_mesh_source_evidence(handle, required, &required, values,
      offsets, rows, segments, strata, entities, parameters, deviations) == PHX_MC_OK);
  PHX_CHECK(offsets[0] == 0 && offsets[6] == required && values[0] == 0 && values[1] == 0);
  void* execution = nullptr;
  PHX_CHECK(phx_mc_execution_begin(UINT64_MAX, 0, UINT64_MAX, 1<<20,
      INFINITY, nullptr, &execution) == PHX_MC_OK);
  // The unchanged mesh reuses its exact ancestry memo: no source query is
  // issued, so even a zero query allowance returns the same count.
  int64_t reused_count = -44;
  PHX_CHECK(phx_mc_tet_mesh_source_evidence(handle, 0, &reused_count, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == PHX_MC_OK);
  PHX_CHECK(reused_count == required);
  uint64_t work[PHX_MC_EXECUTION_COUNTERS], memory[PHX_MC_EXECUTION_MEMORY_VALUES];
  double seconds;
  PHX_CHECK(phx_mc_execution_end(execution, work, memory, &seconds) == PHX_MC_OK);
  PHX_CHECK(work[PHX_MC_EXECUTION_SOURCE_QUERIES] == 0);
  handle->mesh->set_work_limit(1 << 20);
  double position[3];
  phx::mc::SourceWitness witness;
  PHX_CHECK(handle->mesh->construct_source_split(1,3,.5,position,witness) == phx::mc::Insertion::kOk);
  PHX_CHECK(witness.stratum == phx::mc::SourceStratum::kFacet && witness.deviation == 0);
  int32_t inserted = -1;
  PHX_CHECK(handle->mesh->split_edge_bounded(1,3,position,witness,0,inserted) == phx::mc::Insertion::kOk);
  // Changed faces, coordinates and witnesses retire the memo: the recomputed
  // coverage needs source queries again and refuses atomically without one.
  PHX_CHECK(phx_mc_execution_begin(UINT64_MAX, 0, UINT64_MAX, 1<<20,
      INFINITY, nullptr, &execution) == PHX_MC_OK);
  int64_t refused_count = -44;
  PHX_CHECK(phx_mc_tet_mesh_source_evidence(handle, 0, &refused_count, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(refused_count == -44);
  PHX_CHECK(phx_mc_execution_end(execution, work, memory, &seconds) == PHX_MC_CAPACITY_EXCEEDED);
  int64_t changed_count = -1;
  PHX_CHECK(phx_mc_tet_mesh_source_evidence(handle, 0, &changed_count, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == PHX_MC_OK);
  PHX_CHECK(changed_count > 0);
  int32_t applied = -1;
  const double moved[] = {.375,.5,0};
  PHX_CHECK(phx_mc_tet_mesh_relocate(handle,inserted,moved,0,&applied) == PHX_MC_OK && applied == 1);
  PHX_CHECK(phx_mc_tet_mesh_collapse_edge(handle,inserted,1,0,1<<20,&applied) == PHX_MC_OK && applied == 1);
  const auto after = phx::mc::test::export_mesh(handle);
  for (std::size_t row = 0; row < after.face_sources.size(); ++row)
    PHX_CHECK(handle->mesh->face_ancestry(
        after.faces.data()+3*row, after.face_sources[row], contributors));
  PHX_CHECK(handle_audit(handle));
  for (std::size_t row = 0; row < 6; ++row)
    PHX_CHECK(std::equal(original_faces+3*row, original_faces+3*row+3,
                         handle->mesh->original_faces()[row].begin()));
  phx_mc_tet_mesh_free(handle);
}

// A refined two-region box with size targets: many subfacet and subsegment
// splits on both the hull and the constrained interface.
phx_mc_tet_mesh* refined_interface_box() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(5, 0, 3), 0.5);
  phx_mc_tet_mesh* mesh = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  PHX_CHECK(mesh != nullptr);
  const std::vector<double> sizes(domain.points.size() / 3, 0.15);
  PHX_CHECK(refine(mesh, 2.0, 1 << 20, sizes.data()).status == PHX_MC_OK);
  mesh->mesh->set_work_limit(int64_t{1} << 40);
  return mesh;
}

// edge_tet/find_face stop their star walk at the first accepted cell. The
// former algorithm (complete vertex_star, then an ordered scan) is rebuilt
// here: both return the same cell and slot, and the early exit never visits
// more cells (strictly fewer in total).
void test_early_exit_star_lookup_matches_complete_star_scan() {
  phx_mc_tet_mesh* handle = refined_interface_box();
  phx::mc::TetMesh& mesh = *handle->mesh;
  const Exported after = phx::mc::test::export_mesh(handle);
  std::vector<int32_t> star;
  int64_t reference_work = 0;
  int64_t early_work = 0;
  for (std::size_t f = 0; f < after.face_sources.size(); ++f) {
    const int32_t a = after.faces[3 * f], b = after.faces[3 * f + 1], c = after.faces[3 * f + 2];
    const int64_t w0 = mesh.work();
    PHX_CHECK(mesh.vertex_star(a, star));
    int32_t expected_t = -1;
    int expected_slot = -1;
    for (int32_t s : star) {
      const phx::mc::Tetrahedron& cell = mesh.tet(s);
      if (phx::mc::is_ghost(cell) || phx::mc::vertex_slot(cell, b) < 0 ||
          phx::mc::vertex_slot(cell, c) < 0) {
        continue;
      }
      for (int k = 0; k < 4 && expected_slot < 0; ++k) {
        if (cell.v[k] != a && cell.v[k] != b && cell.v[k] != c) {
          expected_t = s;
          expected_slot = k;
        }
      }
      break;
    }
    const int64_t w1 = mesh.work();
    int32_t t = -1;
    int slot = -1;
    PHX_CHECK(mesh.find_face(a, b, c, t, slot));
    const int64_t w2 = mesh.work();
    PHX_CHECK(t == expected_t && slot == expected_slot);
    PHX_CHECK(w2 - w1 <= w1 - w0);
    reference_work += w1 - w0;
    early_work += w2 - w1;
  }
  for (std::size_t e = 0; e < after.segment_sources.size(); ++e) {
    const int32_t a = after.segments[2 * e], b = after.segments[2 * e + 1];
    const int64_t w0 = mesh.work();
    PHX_CHECK(mesh.vertex_star(a, star));
    int32_t expected = -1;
    for (int32_t s : star) {
      if (phx::mc::vertex_slot(mesh.tet(s), b) >= 0) {
        expected = s;
        break;
      }
    }
    const int64_t w1 = mesh.work();
    PHX_CHECK(mesh.edge_tet(a, b) == expected);
    const int64_t w2 = mesh.work();
    PHX_CHECK(expected >= 0 && w2 - w1 <= w1 - w0);
    reference_work += w1 - w0;
    early_work += w2 - w1;
  }
  PHX_CHECK(!after.face_sources.empty() && early_work < reference_work);
  PHX_CHECK(handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}

// The former shrink (full passes until one changes nothing, |C| units per
// pass) is rebuilt on an independently grown conflict region. prepare's
// worklist shrink must keep exactly the same cells in the same order, refuse
// exactly when the fixpoint refuses, and charge no more than 4 units per
// grown finite cell plus the former passes.
void test_worklist_shrink_matches_full_pass_fixpoint() {
  phx_mc_tet_mesh* handle = refined_interface_box();
  phx::mc::TetMesh& mesh = *handle->mesh;
  const auto slots = static_cast<int32_t>(mesh.complex().tets.size());
  int64_t compared = 0;
  for (int32_t t = 0; t < slots; ++t) {
    if (!mesh.complex().live(t) || !mesh.finite(t)) continue;
    const double* q[4];
    mesh.corners(t, q);
    double p[3];
    if (!phx::mc::geometry::tetrahedron_circumcenter(q[0], q[1], q[2], q[3], p) ||
        !phx::mc::geometry::point_in_domain(p)) {
      continue;
    }
    int32_t blocked_tet = -1;
    int blocked_slot = -1;
    const int32_t origin = mesh.locate(p, t, blocked_tet, blocked_slot);
    if (origin < 0) continue;
    // Reference grow: the conforming, unauthored, exact-source conflict walk.
    std::vector<char> in(static_cast<std::size_t>(slots), 0);
    std::vector<int32_t> cavity{origin};
    std::vector<int32_t> stack{origin};
    in[static_cast<std::size_t>(origin)] = 1;
    bool duplicate = false;
    for (int32_t v : mesh.tet(origin).v) {
      duplicate = duplicate || (v >= 0 && mesh.point(v)[0] == p[0] &&
                                mesh.point(v)[1] == p[1] && mesh.point(v)[2] == p[2]);
    }
    if (duplicate) continue;
    const auto conflicts = [&](int32_t u) {
      const int32_t* v = mesh.tet(u).v;
      return phx::mc::insphere(mesh.point(v[0]), mesh.point(v[1]), mesh.point(v[2]),
                               mesh.point(v[3]), p) > 0;
    };
    int64_t old_work = 0;
    while (!stack.empty()) {
      const int32_t s = stack.back();
      stack.pop_back();
      if (!mesh.finite(s)) continue;
      old_work += phx::mc::TetWorkCost::kFacet;
      for (int k = 0; k < 4; ++k) {
        const int32_t u = mesh.tet(s).n[k];
        if (in[static_cast<std::size_t>(u)] != 0) continue;
        const bool take = mesh.complex().constraint(s, k) != phx::mc::kNoConstraint
                              ? mesh.orient_with(s, k, p) == 0 && (!mesh.finite(u) || conflicts(u))
                              : mesh.finite(u) && conflicts(u);
        if (take) {
          in[static_cast<std::size_t>(u)] = 1;
          cavity.push_back(u);
          stack.push_back(u);
        }
      }
    }
    // Reference shrink: the former full-pass fixpoint.
    bool refused = false;
    for (bool changed = true; changed && !refused;) {
      changed = false;
      old_work += static_cast<int64_t>(cavity.size());
      for (int32_t s : cavity) {
        if (in[static_cast<std::size_t>(s)] == 0) continue;
        const phx::mc::Tetrahedron& cell = mesh.tet(s);
        bool drop = false;
        if (phx::mc::is_ghost(cell)) {
          drop = in[static_cast<std::size_t>(
                     cell.n[phx::mc::vertex_slot(cell, phx::mc::kGhostVertex)])] == 0;
        } else {
          for (int k = 0; k < 4 && !drop; ++k) {
            drop = in[static_cast<std::size_t>(cell.n[k])] == 0 && mesh.orient_with(s, k, p) <= 0;
          }
        }
        if (!drop) continue;
        bool inside = !phx::mc::is_ghost(cell);
        for (int k = 0; k < 4 && inside; ++k) inside = mesh.orient_with(s, k, p) >= 0;
        if (s == origin || inside) {
          refused = true;
          break;
        }
        in[static_cast<std::size_t>(s)] = 0;
        changed = true;
      }
    }
    std::vector<int32_t> kept;
    for (int32_t s : cavity) {
      if (in[static_cast<std::size_t>(s)] != 0) kept.push_back(s);
    }
    const int64_t w0 = mesh.work();
    const phx::mc::Insertion prepared = mesh.prepare(p, origin);
    const int64_t new_work = mesh.work() - w0;
    if (refused) {
      PHX_CHECK(prepared == phx::mc::Insertion::kRefused);
    } else if (prepared == phx::mc::Insertion::kOk) {
      const auto actual = mesh.cavity();
      PHX_CHECK(std::equal(actual.begin(), actual.end(), kept.begin(), kept.end()));
      PHX_CHECK(new_work <= old_work);
      ++compared;
    }
    mesh.abandon();
  }
  PHX_CHECK(compared > 0);
  PHX_CHECK(handle_audit(handle));
  phx_mc_tet_mesh_free(handle);
}

// Refinement output does not depend on the removed redundant work: fresh
// runs are identical, the measured work suffices exactly as a budget (with
// the identical mesh), and one unit less is a clean capacity refusal.
void test_refinement_work_is_exact_budget_with_identical_mesh() {
  const Domain domain = phx::mc::test::box_domain(phx::mc::test::box_points(5, 0, 3), 0.5);
  const std::vector<double> sizes(domain.points.size() / 3, 0.15);
  phx_mc_tet_mesh* first = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  const Run free_run = refine(first, 2.0, 1 << 20, sizes.data());
  PHX_CHECK(free_run.status == PHX_MC_OK);
  PHX_CHECK(free_run.counters[PHX_MC_TET_MESH_REFINE_SUBFACET_SPLITS] > 0);
  const int64_t work = free_run.counters[PHX_MC_TET_MESH_REFINE_WORK_UNITS];
  phx_mc_tet_mesh* exact = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  const Run exact_run = refine(exact, 2.0, 1 << 20, sizes.data(), work);
  PHX_CHECK(exact_run.status == PHX_MC_OK);
  PHX_CHECK(exact_run.counters[PHX_MC_TET_MESH_REFINE_WORK_UNITS] == work);
  PHX_CHECK(phx::mc::test::export_mesh(first) == phx::mc::test::export_mesh(exact));
  for (int c = 0; c < PHX_MC_TET_MESH_REFINE_COUNTERS; ++c) {
    PHX_CHECK(free_run.counters[c] == exact_run.counters[c]);
  }
  phx_mc_tet_mesh* short_mesh =
      phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  const Run short_run = refine(short_mesh, 2.0, 1 << 20, sizes.data(), work - 1);
  PHX_CHECK(short_run.status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(phx::mc::test::all_positive(phx::mc::test::export_mesh(short_mesh)));
  PHX_CHECK(handle_audit(short_mesh));
  phx_mc_tet_mesh_free(first);
  phx_mc_tet_mesh_free(exact);
  phx_mc_tet_mesh_free(short_mesh);
}

// Two wide chambers joined by a narrow channel. The coplanar walls beside
// each mouth have the same logical source but are disconnected in that plane.
// A volume conflict cavity must not cone one wall from a point on the other.
void test_refinement_preserves_disconnected_coplanar_patches() {
  Domain domain;
  constexpr double x[] = {0.0, 0.4, 0.6, 1.0};
  constexpr double y[] = {0.0, 0.19, 0.21, 0.4};
  const auto vertex = [](int i, int j, int k) { return 16 * k + 4 * j + i; };
  for (int k = 0; k < 2; ++k)
    for (double b : y)
      for (double a : x) domain.points.insert(domain.points.end(), {a, b, double(k)});
  std::map<std::array<int32_t, 3>, std::pair<int, std::array<int32_t, 3>>> facets;
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      if (i == 1 && j != 1) continue;
      std::array<int, 3> permutation{0, 1, 2};
      do {
        int q[3] = {i, j, 0};
        std::array<int32_t, 4> cell{vertex(q[0], q[1], q[2])};
        for (int r = 0; r < 3; ++r) {
          ++q[permutation[r]];
          cell[r + 1] = vertex(q[0], q[1], q[2]);
        }
        const auto at = [&](int32_t v) { return domain.points.data() + 3 * v; };
        if (phx::mc::orient3d(at(cell[0]), at(cell[1]), at(cell[2]), at(cell[3])) < 0)
          std::swap(cell[2], cell[3]);
        domain.tets.insert(domain.tets.end(), cell.begin(), cell.end());
        domain.regions.push_back(0);
        for (int r = 0; r < 4; ++r) {
          std::array<int32_t, 3> face;
          phx::mc::oriented_facet(cell.data(), r, face.data());
          std::swap(face[1], face[2]);
          auto key = face;
          std::sort(key.begin(), key.end());
          auto& entry = facets[key];
          ++entry.first;
          entry.second = face;
        }
      } while (std::next_permutation(permutation.begin(), permutation.end()));
    }
  }
  std::map<std::array<int32_t, 2>, std::vector<int>> edges;
  for (const auto& [key, entry] : facets) {
    if (entry.first != 1) continue;
    const auto& face = entry.second;
    domain.faces.insert(domain.faces.end(), face.begin(), face.end());
    domain.face_sources.push_back(0);
    int normal = -1;
    for (int axis = 0; axis < 3; ++axis) {
      if (domain.points[3 * face[0] + axis] == domain.points[3 * face[1] + axis] &&
          domain.points[3 * face[0] + axis] == domain.points[3 * face[2] + axis])
        normal = axis;
    }
    PHX_CHECK(normal >= 0);
    for (int r = 0; r < 3; ++r) {
      std::array<int32_t, 2> edge{face[r], face[(r + 1) % 3]};
      std::sort(edge.begin(), edge.end());
      edges[edge].push_back(normal);
    }
  }
  for (const auto& [edge, normals] : edges) {
    PHX_CHECK(normals.size() == 2);
    if (normals[0] == normals[1]) continue;
    domain.segments.insert(domain.segments.end(), edge.begin(), edge.end());
    domain.segment_sources.push_back(static_cast<int32_t>(domain.segment_sources.size()));
  }
  phx_mc_tet_mesh* mesh = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  PHX_CHECK(mesh != nullptr);
  const std::vector<double> sizes(domain.points.size() / 3, 0.3);
  const Run run = refine(mesh, 2.0, 2000, sizes.data(), 2000000);
  PHX_CHECK(run.status == PHX_MC_OK || run.status == PHX_MC_REFINEMENT_LIMIT);
  PHX_CHECK(run.counters[PHX_MC_TET_MESH_REFINE_SUBSEGMENT_SPLITS] > 0);
  int64_t ancestors = 0;
  PHX_CHECK(phx_mc_tet_mesh_source_evidence(mesh, 0, &ancestors, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == PHX_MC_OK);
  PHX_CHECK(ancestors > 0);
  // Compare the spatially selected proof against exhaustive original-row
  // coverage, including closed-box contacts on all coordinate planes.
  phx::mc::NativeVector<int32_t> all_rows, indexed, exhaustive;
  for (std::size_t row = 0; row < mesh->mesh->original_faces().size(); ++row)
    all_rows.push_back(static_cast<int32_t>(row));
  phx::mc::NativeVector<std::array<int32_t, 4>> exported_faces;
  mesh->mesh->collect_faces(exported_faces);
  const auto compare_ancestry = [&](const int32_t* face) {
    double subject[9];
    for (int k = 0; k < 3; ++k)
      std::copy_n(mesh->mesh->point(face[k]), 3, subject + 3 * k);
    const bool selected = mesh->mesh->face_ancestry(face, 0, indexed);
    const bool full = phx::mc::exact_facet_coverage(
        subject, domain.points.data(), mesh->mesh->original_faces(), all_rows, exhaustive);
    PHX_CHECK(selected == full && indexed == exhaustive);
    return selected;
  };
  for (const auto& face : exported_faces) PHX_CHECK(compare_ancestry(face.data()));
  // This triangle would close a channel mouth. Its box touches both walls,
  // but neither the hierarchy nor the exact oracle may claim gap coverage.
  const int32_t gap[3] = {vertex(2, 1, 0), vertex(2, 2, 0), vertex(2, 2, 1)};
  PHX_CHECK(!compare_ancestry(gap));
  PHX_CHECK(!mesh->mesh->face_ancestry(gap, 1, indexed) && indexed.empty());
  PHX_CHECK(phx::mc::test::all_positive(phx::mc::test::export_mesh(mesh)));
  PHX_CHECK(handle_audit(mesh));
  phx_mc_tet_mesh_free(mesh);
}

// A carrier that is not constrained Delaunay: the two cells across the
// unconstrained face abc have the apex e strictly inside the circumsphere of
// abcd and de crosses abc. Refinement restores the property before it
// assesses any cell (the 2-3 flip around de); a fixed boundary and an
// unreachable shape bound make no insertion, and the volume is unchanged.
void test_refinement_restores_constrained_delaunay_carrier() {
  Domain domain;
  domain.points = {1.0, 0.0, 0.0, -1.0, 1.0, 0.0, -1.0, -1.0, 0.0,
                   0.0, 0.0, 0.25, 0.0, 0.0, -0.25};
  const auto at = [&](int32_t v) { return domain.points.data() + 3 * v; };
  for (std::array<int32_t, 4> cell : {std::array<int32_t, 4>{0, 1, 2, 3},
                                      std::array<int32_t, 4>{0, 1, 2, 4}}) {
    if (phx::mc::orient3d(at(cell[0]), at(cell[1]), at(cell[2]), at(cell[3])) < 0) {
      std::swap(cell[2], cell[3]);
    }
    domain.tets.insert(domain.tets.end(), cell.begin(), cell.end());
    domain.regions.push_back(0);
  }
  domain.faces = {0, 1, 3, 1, 2, 3, 0, 2, 3, 0, 1, 4, 1, 2, 4, 0, 2, 4};
  domain.face_sources = {0, 1, 2, 3, 4, 5};
  domain.segments = {0, 1, 1, 2, 0, 2, 0, 3, 1, 3, 2, 3, 0, 4, 1, 4, 2, 4};
  domain.segment_sources = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  const double* p[5] = {at(0), at(1), at(2), at(3), at(4)};
  PHX_CHECK(phx::mc::insphere(p[0], p[1], p[2], p[3], p[4]) != 0);
  phx_mc_tet_mesh* mesh = phx::mc::test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_FIXED);
  PHX_CHECK(mesh != nullptr);
  const double volume = phx::mc::test::region_volume(phx::mc::test::export_mesh(mesh), 0);
  const Run run = refine(mesh, 1.0e6);
  PHX_CHECK(run.status == PHX_MC_OK);
  PHX_CHECK(run.counters[PHX_MC_TET_MESH_REFINE_VERTICES] == 5);
  const Exported after = phx::mc::test::export_mesh(mesh);
  PHX_CHECK(after.regions.size() == 3);
  for (std::size_t t = 0; t < after.regions.size(); ++t) {
    const int32_t* v = after.tets.data() + 4 * t;
    PHX_CHECK(std::count(v, v + 4, 3) == 1 && std::count(v, v + 4, 4) == 1);
  }
  PHX_CHECK(after.faces == phx::mc::test::export_mesh(mesh).faces);
  PHX_CHECK(phx::mc::test::all_positive(after));
  PHX_CHECK_NEAR(phx::mc::test::region_volume(after, 0), volume, kUnitBoxMeasureTolerance);
  PHX_CHECK(handle_audit(mesh));
  phx_mc_tet_mesh_free(mesh);
}

}  // namespace


int main() {
  test_refinement_preserves_disconnected_coplanar_patches();
  test_logical_facet_retriangulation_split_relocation_and_collapse();
  test_logical_facet_coverage_retains_all_primitive_contributors();
  test_single_row_facet_coverage_matches_arrangement();
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
  test_dyadic_nonrepresentability_filter_preserves_exact_constructions();
  test_filtered_source_construction_keeps_witness_and_exact_work_boundary();
  test_edge_filter_unsupported_coordinate_inputs_stay_refused();
  test_inspected_split_authenticates_the_existing_complete_native_cavity();
  test_retained_split_lifecycle_and_exact_resource_boundaries();
  test_one_prepared_owner_is_movable_single_use_and_exception_safe();
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
  test_early_exit_star_lookup_matches_complete_star_scan();
  test_worklist_shrink_matches_full_pass_fixpoint();
  test_refinement_work_is_exact_budget_with_identical_mesh();
  test_refinement_restores_constrained_delaunay_carrier();
  return phx::mc::test::finish("test_refine3d");
}
