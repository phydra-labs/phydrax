//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Periodic quotient topology, positive lifted orientation and deterministic export.
#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {
struct PeriodicResult {
  int32_t status = PHX_MC_INTERNAL_ERROR;
  std::vector<int32_t> vertices;
  std::vector<int32_t> shifts;
};

PeriodicResult triangulate() {
  constexpr double points[] = {0.23, 0.37};
  constexpr double fractional[] = {0.23, 0.37};
  constexpr double lattice[] = {1.0, 0.0, 0.0, 1.0};
  constexpr double inverse[] = {1.0, 0.0, 0.0, 1.0};
  phx_mc_periodic_triangulation* handle = nullptr;
  PeriodicResult result;
  result.status = phx_mc_periodic_delaunay(2, 1, points, fractional, lattice, inverse,
                                            1.0, &handle);
  if (result.status == PHX_MC_OK) {
    const int64_t cells = phx_mc_periodic_cell_count(handle);
    result.vertices.resize(static_cast<std::size_t>(3 * cells));
    result.shifts.resize(static_cast<std::size_t>(6 * cells));
    PHX_CHECK(phx_mc_periodic_copy_cells(handle, cells, result.vertices.data(),
                                         result.shifts.data()) == PHX_MC_OK);
  }
  phx_mc_periodic_free(handle);
  return result;
}

double lifted_area(const std::vector<int32_t>& shifts, int64_t cell) {
  std::array<std::array<double, 2>, 3> p{};
  constexpr double fractional[2] = {0.23, 0.37};
  for (int v = 0; v < 3; ++v) {
    p[v][0] = fractional[0] + shifts[6 * cell + 2 * v];
    p[v][1] = fractional[1] + shifts[6 * cell + 2 * v + 1];
  }
  return 0.5 * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) -
                (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]));
}

void one_site_torus() {
  const PeriodicResult first = triangulate();
  const PeriodicResult second = triangulate();
  PHX_CHECK(first.status == PHX_MC_OK);
  PHX_CHECK(second.status == PHX_MC_OK);
  PHX_CHECK(first.vertices.size() == 6);
  PHX_CHECK(first.shifts.size() == 12);
  PHX_CHECK(first.vertices == second.vertices);
  PHX_CHECK(first.shifts == second.shifts);
  double area = 0.0;
  for (int64_t cell = 0; cell < 2; ++cell) {
    PHX_CHECK(first.vertices[3 * cell] == 0);
    PHX_CHECK(first.vertices[3 * cell + 1] == 0);
    PHX_CHECK(first.vertices[3 * cell + 2] == 0);
    const double signed_area = lifted_area(first.shifts, cell);
    PHX_CHECK(signed_area > 0.0);
    area += signed_area;
  }
  PHX_CHECK(std::fabs(area - 1.0) < 1.0e-12);
}
void quotient_refinement_keeps_valid_unresolved_snapshot() {
  const PeriodicResult initial = triangulate();
  PHX_CHECK(initial.status == PHX_MC_OK);
  constexpr double points[] = {0.23, 0.37};
  constexpr double lattice[] = {1.0, 0.0, 0.0, 1.0};
  constexpr double inverse[] = {1.0, 0.0, 0.0, 1.0};
  const double absent = std::numeric_limits<double>::quiet_NaN();
  const double controls[] = {0.75, absent, absent, absent, 1.0};
  constexpr double quantiles[] = {0.5};
  phx_mc_periodic_refinement_input input{};
  input.point_count = 1;
  input.points = points;
  input.lattice = lattice;
  input.inverse = inverse;
  input.cell_count = 2;
  input.vertices = initial.vertices.data();
  input.shifts = initial.shifts.data();
  input.control_count = 1;
  input.controls = controls;
  input.statistic_count = 1;
  input.quantiles = quantiles;
  input.absolute_tolerance = 1.0e-12;
  input.relative_tolerance = 1.0e-12;
  input.protected_edge_count = 0;
  input.protected_edges = nullptr;
  phx_mc_periodic_refinement* handle = nullptr;
  const int32_t status = phx_mc_periodic_refine_2d(&input, &handle);
  PHX_CHECK(status == PHX_MC_OK || status == PHX_MC_UNRESOLVED);
  PHX_CHECK(handle != nullptr);
  if (handle == nullptr) return;
  std::array<int64_t, 2> sizes{};
  PHX_CHECK(phx_mc_periodic_refinement_sizes(handle, sizes.data()) == PHX_MC_OK);
  const int64_t point_count = sizes[0];
  const int64_t cell_count = sizes[1];
  PHX_CHECK(point_count >= 1);
  PHX_CHECK(cell_count == 2 * point_count);
  std::vector<double> refined_points(static_cast<std::size_t>(2 * point_count));
  std::vector<int32_t> vertices(static_cast<std::size_t>(3 * cell_count));
  std::vector<int32_t> shifts(static_cast<std::size_t>(6 * cell_count));
  PHX_CHECK(phx_mc_periodic_refinement_copy_points(handle, point_count,
                                                    refined_points.data()) == PHX_MC_OK);
  PHX_CHECK(phx_mc_periodic_refinement_copy_cells(handle, cell_count, vertices.data(),
                                                   shifts.data()) == PHX_MC_OK);
  double total_area = 0.0;
  for (int64_t cell = 0; cell < cell_count; ++cell) {
    std::array<std::array<double, 2>, 3> lifted_points{};
    for (int vertex = 0; vertex < 3; ++vertex) {
      const int32_t representative = vertices[3 * cell + vertex];
      lifted_points[vertex][0] =
          refined_points[2 * representative] + shifts[6 * cell + 2 * vertex];
      lifted_points[vertex][1] =
          refined_points[2 * representative + 1] + shifts[6 * cell + 2 * vertex + 1];
    }
    const auto& a = lifted_points[0];
    const auto& b = lifted_points[1];
    const auto& c = lifted_points[2];
    const double area = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) -
                               (b[1] - a[1]) * (c[0] - a[0]));
    PHX_CHECK(area > 0.0);
    total_area += area;
  }
  PHX_CHECK(std::fabs(total_area - 1.0) < 1.0e-11);
  phx_mc_periodic_refinement_free(handle);
}
void periodic_index_range_status() {
  constexpr double points[] = {0.23, 0.37};
  constexpr double lattice[] = {1.0, 0.0, 0.0, 1.0};
  constexpr double inverse[] = {1.0, 0.0, 0.0, 1.0};
  phx_mc_periodic_triangulation* triangulation = nullptr;
  const int64_t point_overflow = std::numeric_limits<int32_t>::max();
  PHX_CHECK(phx_mc_periodic_delaunay(
                2, point_overflow, points, points, lattice, inverse, 1.0,
                &triangulation) == PHX_MC_INDEX_RANGE_ERROR);
  PHX_CHECK(triangulation == nullptr);

  constexpr int32_t vertices[] = {0, 0, 0};
  constexpr int32_t shifts[] = {0, 0, 0, 0, 0, 0};
  const double absent = std::numeric_limits<double>::quiet_NaN();
  const double controls[] = {1.0, absent, absent, absent, 1.0};
  constexpr double quantiles[] = {0.5};
  phx_mc_periodic_refinement_input input{};
  input.point_count = 1;
  input.points = points;
  input.lattice = lattice;
  input.inverse = inverse;
  input.cell_count = std::numeric_limits<int32_t>::max();
  input.vertices = vertices;
  input.shifts = shifts;
  input.control_count = 1;
  input.controls = controls;
  input.statistic_count = 1;
  input.quantiles = quantiles;
  input.absolute_tolerance = 0.0;
  input.relative_tolerance = 0.0;
  phx_mc_periodic_refinement* refinement = nullptr;
  PHX_CHECK(phx_mc_periodic_refine_2d(&input, &refinement) == PHX_MC_INDEX_RANGE_ERROR);
  PHX_CHECK(refinement == nullptr);
}

struct LocalMeshResult {
  int32_t status = PHX_MC_INTERNAL_ERROR;
  double certified_margin = 0.0;
  std::vector<int32_t> vertices;
  std::vector<int32_t> shifts;
  std::vector<int64_t> cell_ids;
  std::vector<int64_t> cell_key_ids;
  std::vector<int32_t> cell_key_relative_shifts;
  std::vector<int32_t> cell_owner_ranks;
  std::vector<int8_t> support_cell_mask;
  std::vector<double> support_fractional_lower;
  std::vector<double> support_fractional_upper;
  int8_t has_owned_sites = 0;
  std::vector<double> owner_fractional_lower;
  std::vector<double> owner_fractional_upper;
};

LocalMeshResult local_delaunay(
    int32_t dimension, const std::vector<double>& points,
    const std::vector<double>& fractional, const std::vector<int64_t>& point_ids,
    const std::vector<int32_t>& point_owner_ranks, int32_t local_owner_rank,
    const double* lattice, const double* inverse) {
  phx_mc_periodic_local_delaunay_request request{};
  request.dimension = dimension;
  request.point_count = static_cast<int64_t>(point_ids.size());
  request.points = points.data();
  request.fractional = fractional.data();
  request.point_global_ids = point_ids.data();
  request.point_owner_ranks = point_owner_ranks.data();
  request.local_owner_rank = local_owner_rank;
  request.lattice = lattice;
  request.inverse = inverse;
  request.initial_margin = 1.0;
  phx_mc_periodic_local_delaunay* handle = nullptr;
  request.result = &handle;
  LocalMeshResult result;
  result.status = phx_mc_periodic_delaunay_local(&request);
  if (result.status != PHX_MC_OK) return result;
  int64_t cell_count = 0;
  if (phx_mc_periodic_local_delaunay_sizes(
          handle, &cell_count, &result.certified_margin) != PHX_MC_OK ||
      cell_count < 0) {
    result.status = PHX_MC_INTERNAL_ERROR;
    phx_mc_periodic_local_delaunay_free(handle);
    return result;
  }
  result.owner_fractional_lower.resize(static_cast<std::size_t>(dimension));
  result.owner_fractional_upper.resize(static_cast<std::size_t>(dimension));
  if (phx_mc_periodic_local_delaunay_owner_bounds(
          handle, &result.has_owned_sites, result.owner_fractional_lower.data(),
          result.owner_fractional_upper.data()) != PHX_MC_OK) {
    result.status = PHX_MC_INTERNAL_ERROR;
    phx_mc_periodic_local_delaunay_free(handle);
    return result;
  }
  const int arity = dimension + 1;
  result.vertices.resize(static_cast<std::size_t>(cell_count * arity));
  result.shifts.resize(static_cast<std::size_t>(cell_count * arity * dimension));
  result.cell_ids.resize(static_cast<std::size_t>(cell_count));
  result.cell_key_ids.resize(static_cast<std::size_t>(cell_count * arity));
  result.cell_key_relative_shifts.resize(
      static_cast<std::size_t>(cell_count * arity * dimension));
  result.cell_owner_ranks.resize(static_cast<std::size_t>(cell_count));
  result.support_fractional_lower.resize(
      static_cast<std::size_t>(cell_count * dimension));
  result.support_fractional_upper.resize(
      static_cast<std::size_t>(cell_count * dimension));
  result.support_cell_mask.resize(static_cast<std::size_t>(cell_count));
  result.status = phx_mc_periodic_local_delaunay_copy(
      handle, cell_count, result.vertices.data(), result.shifts.data(),
      result.cell_ids.data(), result.cell_key_ids.data(),
      result.cell_key_relative_shifts.data(), result.cell_owner_ranks.data(),
      result.support_cell_mask.data(), result.support_fractional_lower.data(),
      result.support_fractional_upper.data());
  phx_mc_periodic_local_delaunay_free(handle);
  return result;
}

struct SupportHits {
  int32_t status = PHX_MC_INTERNAL_ERROR;
  uint64_t candidate_revision = 0;
  std::vector<int64_t> cell_rows;
  std::vector<int64_t> point_ids;
  std::vector<int32_t> shifts;
  std::vector<int8_t> signs;
};

SupportHits owner_support_query(
    int32_t dimension, const std::vector<double>& cell_points,
    const std::vector<int64_t>& cell_point_ids,
    const std::vector<int32_t>& cell_shifts, const std::vector<double>& points,
    const std::vector<int64_t>& point_ids, const double* lattice,
    uint64_t revision) {
  phx_mc_periodic_support_query_request request{};
  request.dimension = dimension;
  request.cell_count =
      static_cast<int64_t>(cell_point_ids.size()) / (dimension + 1);
  request.cell_vertex_points = cell_points.data();
  request.cell_vertex_global_ids = cell_point_ids.data();
  request.cell_vertex_shifts = cell_shifts.data();
  request.point_count = static_cast<int64_t>(point_ids.size());
  request.points = points.data();
  request.point_global_ids = point_ids.data();
  request.lattice = lattice;
  request.candidate_revision = revision;
  phx_mc_periodic_support_query_result* handle = nullptr;
  request.result = &handle;
  SupportHits result;
  result.status = phx_mc_periodic_owner_support_query(&request);
  if (result.status != PHX_MC_OK || handle == nullptr) return result;
  int64_t hit_count = 0;
  if (phx_mc_periodic_support_query_sizes(
          handle, &hit_count, &result.candidate_revision) != PHX_MC_OK ||
      hit_count < 0) {
    result.status = PHX_MC_INTERNAL_ERROR;
    phx_mc_periodic_support_query_free(handle);
    return result;
  }
  result.cell_rows.resize(static_cast<std::size_t>(hit_count));
  result.point_ids.resize(static_cast<std::size_t>(hit_count));
  result.shifts.resize(static_cast<std::size_t>(hit_count * dimension));
  result.signs.resize(static_cast<std::size_t>(hit_count));
  result.status = phx_mc_periodic_support_query_copy_hits(
      handle, hit_count, result.cell_rows.data(), result.point_ids.data(),
      result.shifts.data(), result.signs.data());
  phx_mc_periodic_support_query_free(handle);
  return result;
}

std::vector<int64_t> periodic_simplex_key(
    int32_t dimension, const std::vector<int32_t>& cells,
    const std::vector<int32_t>& shifts, int64_t cell,
    const std::vector<int64_t>& point_ids) {
  const int arity = dimension + 1;
  struct KeyVertex {
    int64_t id = 0;
    std::array<int32_t, 3> shift{};
    std::array<int32_t, 3> relative{};
  };
  std::array<KeyVertex, 4> vertices{};
  int anchor = 0;
  for (int vertex = 0; vertex < arity; ++vertex) {
    const int64_t offset = cell * arity + vertex;
    KeyVertex& item = vertices[static_cast<std::size_t>(vertex)];
    item.id = point_ids[static_cast<std::size_t>(cells[static_cast<std::size_t>(offset)])];
    for (int axis = 0; axis < dimension; ++axis) {
      item.shift[static_cast<std::size_t>(axis)] =
          shifts[static_cast<std::size_t>(offset * dimension + axis)];
    }
    const KeyVertex& current = vertices[static_cast<std::size_t>(anchor)];
    if (item.id < current.id || (item.id == current.id && item.shift < current.shift)) {
      anchor = vertex;
    }
  }
  for (int vertex = 0; vertex < arity; ++vertex) {
    for (int axis = 0; axis < dimension; ++axis) {
      vertices[static_cast<std::size_t>(vertex)].relative[static_cast<std::size_t>(axis)] =
          vertices[static_cast<std::size_t>(vertex)].shift[static_cast<std::size_t>(axis)] -
          vertices[static_cast<std::size_t>(anchor)].shift[static_cast<std::size_t>(axis)];
    }
  }
  std::array<int, 4> order{0, 1, 2, 3};
  std::sort(order.begin(), order.begin() + arity,
            [&vertices](int left, int right) {
              const KeyVertex& a = vertices[static_cast<std::size_t>(left)];
              const KeyVertex& b = vertices[static_cast<std::size_t>(right)];
              if (a.id != b.id) return a.id < b.id;
              return a.relative < b.relative;
            });
  std::vector<int64_t> key;
  key.reserve(static_cast<std::size_t>(arity * (dimension + 1)));
  for (int position = 0; position < arity; ++position) {
    const KeyVertex& item = vertices[static_cast<std::size_t>(order[position])];
    key.push_back(item.id);
    for (int axis = 0; axis < dimension; ++axis) {
      key.push_back(item.relative[static_cast<std::size_t>(axis)]);
    }
  }
  return key;
}

void exact_support_query_reports_periodic_cospherical_image() {
  constexpr double lattice[] = {10.0, 0.0, 0.0, 10.0};
  constexpr double cell_points[] = {9.0, 9.0, 0.0, 9.0, 9.0, 0.0};
  constexpr int64_t cell_point_ids[] = {11, 23, 37};
  constexpr int32_t cell_shifts[] = {0, 0, 1, 0, 0, 1};
  constexpr double points[] = {0.0, 0.0};
  constexpr int64_t point_ids[] = {91};
  phx_mc_periodic_support_query_request request{};
  request.dimension = 2;
  request.cell_count = 1;
  request.cell_vertex_points = cell_points;
  request.cell_vertex_global_ids = cell_point_ids;
  request.cell_vertex_shifts = cell_shifts;
  request.point_count = 1;
  request.points = points;
  request.point_global_ids = point_ids;
  request.lattice = lattice;
  request.candidate_revision = 0xACEDULL;
  phx_mc_periodic_support_query_result* handle = nullptr;
  request.result = &handle;
  PHX_CHECK(phx_mc_periodic_owner_support_query(&request) == PHX_MC_OK);
  PHX_CHECK(handle != nullptr);
  if (handle == nullptr) return;
  int64_t hit_count = 0;
  uint64_t revision = 0;
  PHX_CHECK(phx_mc_periodic_support_query_sizes(handle, &hit_count, &revision) == PHX_MC_OK);
  PHX_CHECK(revision == request.candidate_revision);
  PHX_CHECK(hit_count == 1);
  std::vector<int64_t> cell_rows(static_cast<std::size_t>(hit_count));
  std::vector<int64_t> hit_ids(static_cast<std::size_t>(hit_count));
  std::vector<int32_t> image_shifts(static_cast<std::size_t>(2 * hit_count));
  std::vector<int8_t> signs(static_cast<std::size_t>(hit_count));
  PHX_CHECK(phx_mc_periodic_support_query_copy_hits(
                handle, hit_count, cell_rows.data(), hit_ids.data(), image_shifts.data(),
                signs.data()) == PHX_MC_OK);
  PHX_CHECK(cell_rows[0] == 0);
  PHX_CHECK(hit_ids[0] == point_ids[0]);
  PHX_CHECK(image_shifts[0] == 1 && image_shifts[1] == 1);
  PHX_CHECK(signs[0] == 0);
  phx_mc_periodic_support_query_free(handle);

  const double negative_cell_points[] = {9.0, 9.0, 9.0, 0.0, 0.0, 9.0};
  const int64_t negative_cell_ids[] = {11, 37, 23};
  const int32_t negative_cell_shifts[] = {0, 0, 0, 1, 1, 0};
  request.cell_vertex_points = negative_cell_points;
  request.cell_vertex_global_ids = negative_cell_ids;
  request.cell_vertex_shifts = negative_cell_shifts;
  handle = nullptr;
  request.result = &handle;
  PHX_CHECK(phx_mc_periodic_owner_support_query(&request) == PHX_MC_OK);
  PHX_CHECK(handle != nullptr);
  if (handle != nullptr) {
    int64_t normalized_hit_count = 0;
    uint64_t normalized_revision = 0;
    PHX_CHECK(phx_mc_periodic_support_query_sizes(
                  handle, &normalized_hit_count, &normalized_revision) == PHX_MC_OK);
    PHX_CHECK(normalized_hit_count == 1);
    std::array<int64_t, 1> normalized_cells{};
    std::array<int64_t, 1> normalized_ids{};
    std::array<int32_t, 2> normalized_shifts{};
    std::array<int8_t, 1> normalized_signs{};
    PHX_CHECK(phx_mc_periodic_support_query_copy_hits(
                  handle, normalized_hit_count, normalized_cells.data(),
                  normalized_ids.data(), normalized_shifts.data(),
                  normalized_signs.data()) == PHX_MC_OK);
    PHX_CHECK(normalized_signs[0] == 0);
    phx_mc_periodic_support_query_free(handle);
  }
  else {
    return;
  }

  const double degenerate_points[] = {0.0, 0.0, 1.0, 0.0, 2.0, 0.0};
  const int64_t degenerate_ids[] = {1, 2, 3};
  const int32_t degenerate_shifts[] = {0, 0, 0, 0, 0, 0};
  request.cell_vertex_points = degenerate_points;
  request.cell_vertex_global_ids = degenerate_ids;
  request.cell_vertex_shifts = degenerate_shifts;
  request.point_count = 0;
  request.points = nullptr;
  request.point_global_ids = nullptr;
  handle = nullptr;
  PHX_CHECK(phx_mc_periodic_owner_support_query(&request) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(handle == nullptr);
}

void non_dyadic_lattice_boundary_tie_uses_exact_transform() {
  constexpr double lattice[] = {3.0, 1.0, 1.0, 2.0};
  constexpr double inverse[] = {0.4, -0.2, -0.2, 0.6};
  const std::vector<double> source_points{100000005.0, 100000000.0};
  const std::vector<double> source_fractional{20000002.0, 39999999.0};
  const std::vector<int64_t> source_ids{91};
  const std::vector<int32_t> source_owners{0};
  const LocalMeshResult owner = local_delaunay(
      2, source_points, source_fractional, source_ids, source_owners, 0,
      lattice, inverse);
  PHX_CHECK(owner.status == PHX_MC_OK);
  PHX_CHECK(owner.has_owned_sites == 1);
  PHX_CHECK(owner.owner_fractional_lower[0] == 20000002.0);
  PHX_CHECK(owner.owner_fractional_upper[0] == 20000002.0);
  PHX_CHECK(owner.owner_fractional_lower[1] == 39999999.0);
  PHX_CHECK(owner.owner_fractional_upper[1] == 39999999.0);

  const std::vector<double> cell_points{
      100000004.0, 100000003.0, 100000002.0, 100000003.0,
      100000004.0, 99999999.0};
  const std::vector<int64_t> cell_point_ids{11, 23, 37};
  const std::vector<int32_t> cell_shifts(6, 0);
  const SupportHits hits = owner_support_query(
      2, cell_points, cell_point_ids, cell_shifts, source_points, source_ids,
      lattice, 0xB0A0DULL);
  PHX_CHECK(hits.status == PHX_MC_OK);
  PHX_CHECK(hits.candidate_revision == 0xB0A0DULL);
  const std::map<std::array<int32_t, 2>, int8_t> expected_hits{
      {{-2, 2}, 0}, {{-1, 0}, 0}, {{-1, 1}, 1}, {{-1, 2}, 0}, {{0, 0}, 0}};
  const bool aligned_hits =
      hits.cell_rows.size() == hits.point_ids.size() &&
      hits.signs.size() == hits.point_ids.size() &&
      hits.shifts.size() == 2 * hits.point_ids.size();
  PHX_CHECK(aligned_hits);
  std::map<std::array<int32_t, 2>, int8_t> actual_hits;
  if (aligned_hits) {
    for (std::size_t hit = 0; hit < hits.point_ids.size(); ++hit) {
      PHX_CHECK(hits.cell_rows[hit] == 0);
      PHX_CHECK(hits.point_ids[hit] == 91);
      const std::array<int32_t, 2> shift{
          hits.shifts[2 * hit], hits.shifts[2 * hit + 1]};
      const bool inserted = actual_hits.emplace(shift, hits.signs[hit]).second;
      PHX_CHECK(inserted);
    }
  }
  PHX_CHECK(actual_hits == expected_hits);
}

void non_dyadic_3d_lattice_boundary_tie_uses_exact_transform() {
  constexpr double lattice[] = {
      10.0, 2.0, 1.0, 1.0, 11.0, 3.0, 2.0, 1.0, 12.0};
  const std::vector<double> cell_points{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const std::vector<int64_t> cell_point_ids{11, 23, 37, 41};
  const std::vector<int32_t> cell_shifts(12, 0);
  const std::vector<double> source_points{1.0, 1.0, 1.0};
  const std::vector<int64_t> source_ids{91};
  const SupportHits hits = owner_support_query(
      3, cell_points, cell_point_ids, cell_shifts, source_points, source_ids,
      lattice, 0x3D0B0AULL);
  PHX_CHECK(hits.status == PHX_MC_OK);
  PHX_CHECK(hits.candidate_revision == 0x3D0B0AULL);
  PHX_CHECK(hits.cell_rows.size() == 1);
  PHX_CHECK(hits.point_ids.size() == 1);
  if (hits.point_ids.size() == 1) {
    PHX_CHECK(hits.cell_rows[0] == 0);
    PHX_CHECK(hits.point_ids[0] == 91);
    PHX_CHECK(hits.shifts[0] == 0 && hits.shifts[1] == 0 &&
              hits.shifts[2] == 0);
    PHX_CHECK(hits.signs[0] == 0);
  }
}


void owner_local_generation_expands_remote_support_and_matches_reference() {
  constexpr int32_t dimension = 2;
  constexpr int64_t point_count = 6;
  constexpr double lattice[] = {1.2, 0.1, 0.35, 1.05};
  constexpr double inverse[] = {
      0.8571428571428571, -0.0816326530612245,
      -0.2857142857142857, 0.9795918367346939};
  constexpr double fractional[] = {
      0.02, 0.03, 0.48, 0.07, 0.09, 0.51,
      0.55, 0.59, 0.95, 0.11, 0.91, 0.93};
  constexpr int64_t point_ids[] = {4, 17, 29, 43, 58, 71};
  constexpr int32_t point_owners[] = {0, 1, 0, 1, 0, 1};
  std::vector<double> points(static_cast<std::size_t>(2 * point_count));
  for (int64_t row = 0; row < point_count; ++row) {
    points[static_cast<std::size_t>(2 * row)] =
        fractional[2 * row] * lattice[0] + fractional[2 * row + 1] * lattice[2];
    points[static_cast<std::size_t>(2 * row + 1)] =
        fractional[2 * row] * lattice[1] + fractional[2 * row + 1] * lattice[3];
  }
  phx_mc_periodic_triangulation* full = nullptr;
  PHX_CHECK(phx_mc_periodic_delaunay(
                dimension, point_count, points.data(), fractional, lattice, inverse, 1.0,
                &full) == PHX_MC_OK);
  PHX_CHECK(full != nullptr);
  if (full == nullptr) return;
  const int64_t full_cells = phx_mc_periodic_cell_count(full);
  std::vector<int32_t> full_vertices(static_cast<std::size_t>(3 * full_cells));
  std::vector<int32_t> full_shifts(static_cast<std::size_t>(6 * full_cells));
  PHX_CHECK(phx_mc_periodic_copy_cells(full, full_cells, full_vertices.data(),
                                      full_shifts.data()) == PHX_MC_OK);
  phx_mc_periodic_free(full);
  std::set<std::vector<int64_t>> reference_keys;
  for (int64_t cell = 0; cell < full_cells; ++cell) {
    reference_keys.insert(periodic_simplex_key(
        dimension, full_vertices, full_shifts, cell,
        std::vector<int64_t>(point_ids, point_ids + point_count)));
  }

  std::array<LocalMeshResult, 2> owner_metadata;
  for (int32_t owner = 0; owner < 2; ++owner) {
    std::vector<double> owner_points;
    std::vector<double> owner_fractional;
    std::vector<int64_t> owner_ids;
    std::vector<int32_t> owner_ranks;
    for (int64_t row = 0; row < point_count; ++row) {
      if (point_owners[row] != owner) continue;
      owner_points.push_back(points[2 * row]);
      owner_points.push_back(points[2 * row + 1]);
      owner_fractional.push_back(fractional[2 * row]);
      owner_fractional.push_back(fractional[2 * row + 1]);
      owner_ids.push_back(point_ids[row]);
      owner_ranks.push_back(point_owners[row]);
    }
    owner_metadata[static_cast<std::size_t>(owner)] = local_delaunay(
        dimension, owner_points, owner_fractional, owner_ids, owner_ranks,
        owner, lattice, inverse);
    PHX_CHECK(owner_metadata[static_cast<std::size_t>(owner)].status == PHX_MC_OK);
    PHX_CHECK(owner_metadata[static_cast<std::size_t>(owner)].has_owned_sites == 1);
  }
  std::set<std::vector<int64_t>> owner_keys;
  std::map<int64_t, std::vector<int64_t>> ids_to_keys;
  std::array<int64_t, 2> owned_cell_counts{};
  bool support_expanded = false;
  uint64_t revision = 1;
  for (int32_t rank = 0; rank < 2; ++rank) {
    std::vector<int64_t> local_rows;
    for (int64_t row = 0; row < point_count; ++row) {
      if (point_owners[row] == rank) local_rows.push_back(row);
    }
    LocalMeshResult final_mesh;
    for (;;) {
      std::vector<double> local_points;
      std::vector<double> local_fractional;
      std::vector<int64_t> local_ids;
      std::vector<int32_t> local_owners;
      for (const int64_t source_row : local_rows) {
        local_points.push_back(points[2 * source_row]);
        local_points.push_back(points[2 * source_row + 1]);
        local_fractional.push_back(fractional[2 * source_row]);
        local_fractional.push_back(fractional[2 * source_row + 1]);
        local_ids.push_back(point_ids[source_row]);
        local_owners.push_back(point_owners[source_row]);
      }
      LocalMeshResult mesh = local_delaunay(
          dimension, local_points, local_fractional, local_ids, local_owners, rank,
          lattice, inverse);
      PHX_CHECK(mesh.status == PHX_MC_OK);
      if (mesh.status != PHX_MC_OK) return;
      PHX_CHECK(std::isfinite(mesh.certified_margin));
      std::vector<double> support_points;
      std::vector<int64_t> support_ids;
      std::vector<int32_t> support_shifts;
      std::vector<double> remote_points;
      std::vector<int64_t> remote_ids;
      const LocalMeshResult& remote_metadata =
          owner_metadata[static_cast<std::size_t>(1 - rank)];
      const auto& remote_lower = remote_metadata.owner_fractional_lower;
      const auto& remote_upper = remote_metadata.owner_fractional_upper;
      for (int64_t row = 0; row < point_count; ++row) {
        if (point_owners[row] == rank) continue;
        remote_points.push_back(points[2 * row]);
        remote_points.push_back(points[2 * row + 1]);
        remote_ids.push_back(point_ids[row]);
      }
      for (std::size_t cell = 0; cell < mesh.cell_owner_ranks.size(); ++cell) {
        if (!mesh.support_cell_mask[cell]) continue;
        bool intersects_owner_bounds = !remote_ids.empty();
        for (int axis = 0; axis < 2 && intersects_owner_bounds; ++axis) {
          const double lower = mesh.support_fractional_lower[2 * cell + axis];
          const double upper = mesh.support_fractional_upper[2 * cell + axis];
          const double first_shift = std::ceil(std::nextafter(
              lower - remote_upper[static_cast<std::size_t>(axis)],
              -std::numeric_limits<double>::infinity()));
          const double last_shift = std::floor(std::nextafter(
              upper - remote_lower[static_cast<std::size_t>(axis)],
              std::numeric_limits<double>::infinity()));
          intersects_owner_bounds = first_shift <= last_shift;
        }
        if (!intersects_owner_bounds) continue;
        for (int vertex = 0; vertex < 3; ++vertex) {
          const int32_t local_row = mesh.vertices[3 * cell + vertex];
          const int64_t source_row = local_rows[static_cast<std::size_t>(local_row)];
          support_points.push_back(points[2 * source_row]);
          support_points.push_back(points[2 * source_row + 1]);

          support_ids.push_back(point_ids[source_row]);
          support_shifts.push_back(mesh.shifts[6 * cell + 2 * vertex]);
          support_shifts.push_back(mesh.shifts[6 * cell + 2 * vertex + 1]);
        }
      }
      const SupportHits hits = owner_support_query(
          dimension, support_points, support_ids, support_shifts, remote_points,
          remote_ids, lattice, revision++);
      PHX_CHECK(hits.status == PHX_MC_OK);
      PHX_CHECK(hits.candidate_revision == revision - 1);
      PHX_CHECK(std::all_of(hits.signs.begin(), hits.signs.end(),
                            [](int8_t sign) { return sign >= 0; }));
      if (hits.status != PHX_MC_OK) return;
      std::set<int64_t> present(local_ids.begin(), local_ids.end());
      bool added = false;
      for (const int64_t source_id : hits.point_ids) {
        if (present.contains(source_id)) continue;
        const auto found = std::find(point_ids, point_ids + point_count, source_id);
        PHX_CHECK(found != point_ids + point_count);
        if (found == point_ids + point_count) continue;
        local_rows.push_back(static_cast<int64_t>(found - point_ids));
        present.insert(source_id);
        added = true;
        support_expanded = true;
      }
      if (!added) {
        final_mesh = std::move(mesh);
        break;
      }
      std::sort(local_rows.begin(), local_rows.end(), [&point_ids](int64_t left, int64_t right) {
        return point_ids[left] < point_ids[right];
      });
    }

    if (rank == 0) {
      std::vector<double> permuted_points;
      std::vector<double> permuted_fractional;
      std::vector<int64_t> permuted_ids;
      std::vector<int32_t> permuted_owners;
      for (auto row = local_rows.rbegin(); row != local_rows.rend(); ++row) {
        permuted_points.push_back(points[2 * *row]);
        permuted_points.push_back(points[2 * *row + 1]);
        permuted_fractional.push_back(fractional[2 * *row]);
        permuted_fractional.push_back(fractional[2 * *row + 1]);
        permuted_ids.push_back(point_ids[*row]);
        permuted_owners.push_back(point_owners[*row]);
      }
      const LocalMeshResult permuted = local_delaunay(
          dimension, permuted_points, permuted_fractional, permuted_ids,
          permuted_owners, rank, lattice, inverse);
      PHX_CHECK(permuted.status == PHX_MC_OK);
      PHX_CHECK(permuted.cell_ids == final_mesh.cell_ids);
      PHX_CHECK(permuted.cell_key_ids == final_mesh.cell_key_ids);
      PHX_CHECK(permuted.cell_key_relative_shifts ==
                final_mesh.cell_key_relative_shifts);
      PHX_CHECK(permuted.cell_owner_ranks == final_mesh.cell_owner_ranks);
      PHX_CHECK(permuted.support_cell_mask == final_mesh.support_cell_mask);
    }
    for (std::size_t cell = 0; cell < final_mesh.cell_ids.size(); ++cell) {
      PHX_CHECK(final_mesh.cell_owner_ranks[cell] >= 0);
      if (final_mesh.cell_owner_ranks[cell] != rank) continue;
      std::vector<int64_t> key(static_cast<std::size_t>(3 * 3));
      for (int vertex = 0; vertex < 3; ++vertex) {
        const std::size_t target = 3 * cell + vertex;
        key[3 * vertex] = final_mesh.cell_key_ids[target];
        key[3 * vertex + 1] = final_mesh.cell_key_relative_shifts[6 * cell + 2 * vertex];
        key[3 * vertex + 2] = final_mesh.cell_key_relative_shifts[6 * cell + 2 * vertex + 1];
      }
      PHX_CHECK(owner_keys.insert(key).second);
      const auto [position, inserted] = ids_to_keys.emplace(final_mesh.cell_ids[cell], key);
      PHX_CHECK(inserted || position->second == key);
      const auto owner_source =
          std::find(point_ids, point_ids + point_count, final_mesh.cell_key_ids[3 * cell]);
      PHX_CHECK(owner_source != point_ids + point_count);
      if (owner_source != point_ids + point_count) {
        PHX_CHECK(final_mesh.cell_owner_ranks[cell] ==
                  point_owners[owner_source - point_ids]);
      }
      ++owned_cell_counts[static_cast<std::size_t>(rank)];
    }
  }
  PHX_CHECK(support_expanded);
  PHX_CHECK(owned_cell_counts[0] > 0 && owned_cell_counts[1] > 0);
  PHX_CHECK(owner_keys == reference_keys);
}
void empty_owner_returns_empty_local_mesh() {
  constexpr double lattice[] = {1.0, 0.0, 0.0, 1.0};
  constexpr double inverse[] = {1.0, 0.0, 0.0, 1.0};
  const std::vector<double> no_points;
  const std::vector<int64_t> no_ids;
  const std::vector<int32_t> no_owners;
  const LocalMeshResult result = local_delaunay(
      2, no_points, no_points, no_ids, no_owners, 9, lattice, inverse);
  PHX_CHECK(result.status == PHX_MC_OK);
  PHX_CHECK(result.has_owned_sites == 0);
  PHX_CHECK(result.vertices.empty());
  PHX_CHECK(result.cell_ids.empty());
  PHX_CHECK(result.certified_margin == 0.0);
}
}  // namespace

int main() {
  one_site_torus();
  quotient_refinement_keeps_valid_unresolved_snapshot();
  periodic_index_range_status();
  exact_support_query_reports_periodic_cospherical_image();
  non_dyadic_lattice_boundary_tie_uses_exact_transform();
  non_dyadic_3d_lattice_boundary_tie_uses_exact_transform();
  owner_local_generation_expands_remote_support_and_matches_reference();
  empty_owner_returns_empty_local_mesh();
  return phx::mc::test::finish("test_periodic");
}
