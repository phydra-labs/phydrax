//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// C ABI: constrained Delaunay triangulation with Ruppert/Chew refinement.
//
// Pipeline: Delaunay triangulation of the input points (triangulation2d.hpp),
// segment recovery by exact walks and Anglada's pseudo-polygon
// retriangulation, optional hull constraints, carving by flood fill over
// unconstrained edges, then Delaunay refinement by Lawson insertion (splits
// plus flips that never cross constrained edges).
//
// Constraint ids stored on edges: input segment index (>= 0), kHullConstraint
// for convex hull boundary edges, -1 for free edges.  region[t] is 1 for
// triangles of the meshed domain and 0 elsewhere (carved or ghost); a domain
// boundary always runs along constrained edges, so flips and insertions that
// never cross constrained edges preserve the partition.
//
// A split point of a subsegment is a floating-point construction and need not
// lie exactly on the segment.  The split is performed with exact
// orientation tests: a point exactly on the open subsegment splits its two
// triangles into four; a point strictly on one side is inserted into the
// triangle on that side (or attached to the hull when that side is a ghost),
// the two half edges become the constrained subsegments and the old edge
// becomes a free edge whose triangles take the region of the far side.  The
// result is always a valid triangulation; configurations where no exact
// placement exists (degenerate neighborhoods) leave the subsegment unsplit and
// are reported through the final quality check.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "angle.hpp"
#include "cdt2d.hpp"
#include "capi_guard.hpp"
#include "expansion.hpp"
#include "filtered.hpp"
#include "floating_point_mode.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "triangulation2d.hpp"

namespace phx::mc {
namespace {

struct SeedEdge {
  std::array<double, 2> first;
  std::array<double, 2> last;
  bool boundary;
};

std::vector<SeedEdge> seed_edges(const Triangulation2D& coarse) {
  std::vector<SeedEdge> edges;
  for (std::size_t t = 0; t < coarse.triangles.size(); ++t) {
    if (coarse.live[t] == 0 || coarse.region[t] == 0) continue;
    const auto& cell = coarse.triangles[t];
    for (int k = 0; k < 3; ++k) {
      const auto neighbor = static_cast<std::size_t>(cell.n[k]);
      const bool boundary = coarse.region[neighbor] == 0;
      if (!boundary && (neighbor < t || coarse.constraint(static_cast<int32_t>(t), k) < 0))
        continue;
      const auto* a = coarse.point(cell.v[next3(k)]);
      const auto* b = coarse.point(cell.v[prev3(k)]);
      if (a[1] > b[1]) std::swap(a, b);
      edges.push_back({{a[0], a[1]}, {b[0], b[1]}, boundary});
    }
  }
  return edges;
}

struct SeedGrid {
  double x, y, step, rise;
  int32_t columns, rows;
  double clearance;
};

std::optional<SeedGrid> seed_grid(const cdt2d_detail::AreaEstimate& estimate, double max_area,
                                 int64_t input_count) {
  // A simple authored domain amortizes the preflight earlier. A heavily
  // sampled boundary needs more interior work before a second build pays.
  const double minimum_work = std::clamp(64.0 * input_count, 8192.0, 262144.0);
  if (estimate.elements < minimum_work || input_count > 16384) return std::nullopt;
  constexpr double equilateral_height = 0.86602540378443864676;
  const double step = 0.99 * std::sqrt((2.0 / equilateral_height) * max_area);
  const double rise = step * equilateral_height;
  const double cols = std::ceil((estimate.xmax - estimate.xmin) / step);
  const double rows = std::ceil((estimate.ymax - estimate.ymin) / rise);
  const double magnitude = std::max({std::abs(estimate.xmin), std::abs(estimate.xmax),
                                     std::abs(estimate.ymin), std::abs(estimate.ymax)});
  if (!(cols >= 1 && rows >= 1 && cols <= 4096 && rows <= 4096 &&
        cols * rows <= 4194304 && step > 128.0 * std::numeric_limits<double>::epsilon() * magnitude))
    return std::nullopt;
  return SeedGrid{estimate.xmin, estimate.ymin, step, rise,
                  static_cast<int32_t>(cols), static_cast<int32_t>(rows), 2.5 * step / 0.99};
}

struct ForbiddenSpan {
  double first;
  double second;
  bool operator<(const ForbiddenSpan& other) const noexcept {
    return first < other.first || (first == other.first && second < other.second);
  }
};

void seed_row(const std::vector<SeedEdge>& edges, double y, double clearance,
              std::vector<double>& crossings, std::vector<ForbiddenSpan>& forbidden) {
  crossings.clear();
  forbidden.clear();
  for (const SeedEdge& edge : edges) {
    if (edge.first[1] > y + clearance || edge.last[1] < y - clearance) continue;
    if (edge.first[1] == edge.last[1]) {
      forbidden.push_back({std::min(edge.first[0], edge.last[0]) - clearance,
                           std::max(edge.first[0], edge.last[0]) + clearance});
      continue;
    }
    const auto x = [&](double at) {
      const double fraction = (at - edge.first[1]) / (edge.last[1] - edge.first[1]);
      return edge.first[0] + fraction * (edge.last[0] - edge.first[0]);
    };
    if (edge.boundary && edge.first[1] <= y && y < edge.last[1]) crossings.push_back(x(y));
    const double a = x(std::max(y - clearance, edge.first[1]));
    const double b = x(std::min(y + clearance, edge.last[1]));
    // The clipped slab's x extent expanded by the same clearance encloses
    // every point on the row that could be within clearance of this edge.
    forbidden.push_back({std::min(a, b) - clearance, std::max(a, b) + clearance});
  }
  std::sort(crossings.begin(), crossings.end());
  std::sort(forbidden.begin(), forbidden.end());
}

bool admits_seed(Triangulation2D& coarse, int32_t& hint, const double* point) {
  if (!coordinate_in_domain(point[0]) || !coordinate_in_domain(point[1])) return false;
  hint = coarse.walk(hint, point);
  if (coarse.region[static_cast<std::size_t>(hint)] == 0) return false;
  const auto& t = coarse.triangles[static_cast<std::size_t>(hint)];
  int contacts = 0;
  for (int k = 0; k < 3; ++k) {
    if (orient2d_filtered(coarse.point(t.v[next3(k)]), coarse.point(t.v[prev3(k)]), point) != 0)
      continue;
    // The walk admits the closure of a cell. A grid site must not become
    // an authored vertex duplicate or split a protected source edge.
    if (coarse.constraint(hint, k) != -1 || ++contacts == 2) return false;
  }
  return true;
}

bool hole_seeds_clear(Triangulation2D& coarse, int64_t count, const double* holes) {
  int32_t hint = coarse.hint;
  for (int64_t i = 0; i < count; ++i) {
    const double* point = holes + 2 * i;
    hint = coarse.walk(hint, point);
    if (coarse.is_ghost(hint)) continue;
    const auto& cell = coarse.triangles[static_cast<std::size_t>(hint)];
    for (int k = 0; k < 3; ++k) {
      const auto* vertex = coarse.point(cell.v[k]);
      if (vertex[0] == point[0] && vertex[1] == point[1]) return false;
      if (coarse.constraint(hint, k) != -1 &&
          orient2d_filtered(coarse.point(cell.v[next3(k)]), coarse.point(cell.v[prev3(k)]),
                            point) == 0)
        return false;
    }
  }
  return true;
}

std::vector<double> interior_seed_bank(Triangulation2D& coarse, const SeedGrid& grid) {
  constexpr std::size_t max_sites = 1048576;
  const auto edges = seed_edges(coarse);
  if (edges.empty() || edges.size() * static_cast<std::size_t>(grid.rows) > 33554432)
    return {};
  const auto authored_size = coarse.coordinates.size();
  const auto grid_sites = static_cast<std::size_t>(grid.rows) * grid.columns;
  std::vector<double> sites;
  sites.reserve(authored_size + 2 * std::min(max_sites, grid_sites));
  sites.insert(sites.end(), coarse.coordinates.begin(), coarse.coordinates.end());
  std::vector<double> crossings;
  std::vector<ForbiddenSpan> forbidden;
  int32_t hint = coarse.hint;
  for (int32_t row = 0; row < grid.rows; ++row) {
    const double y = grid.y + (row + 0.5) * grid.rise;
    const double x = grid.x + (0.25 + 0.5 * (row & 1)) * grid.step;
    seed_row(edges, y, grid.clearance, crossings, forbidden);
    if (crossings.size() % 2 != 0) return {};
    std::size_t blocker = 0;
    for (std::size_t k = 0; k < crossings.size(); k += 2) {
      const int32_t begin = static_cast<int32_t>(std::ceil((crossings[k] - x) / grid.step));
      const int32_t end = static_cast<int32_t>(std::floor((crossings[k + 1] - x) / grid.step));
      for (int32_t col = begin; col <= end; ++col) {
        const double px = x + grid.step * col;
        while (blocker < forbidden.size() && forbidden[blocker].second < px) ++blocker;
        if (blocker < forbidden.size() && forbidden[blocker].first <= px) continue;
        const double point[2]{px, y};
        if (!admits_seed(coarse, hint, point)) continue;
        if ((sites.size() - authored_size) / 2 == max_sites) return {};
        sites.insert(sites.end(), point, point + 2);
      }
    }
  }
  return sites.size() == authored_size ? std::vector<double>{} : std::move(sites);
}

int32_t initialize_points(Triangulation2D& triangulation, const double* points, int64_t count,
                          int64_t segment_count, const int32_t* segments,
                          std::vector<int32_t>& map) {
  triangulation.reset(points, count, nullptr);
  const int32_t status = build_delaunay_2d(triangulation, points, count, nullptr, map);
  if (status != PHX_MC_OK) return status;
  for (int64_t s = 0; s < segment_count; ++s) {
    if (map[static_cast<std::size_t>(segments[2 * s])] ==
        map[static_cast<std::size_t>(segments[2 * s + 1])]) return PHX_MC_INVALID_INPUT;
  }
  return PHX_MC_OK;
}

void partition_domain(ConstrainedMesher& mesher, const int32_t* segments, int64_t segment_count,
                      const std::vector<int32_t>& map, int32_t keep_convex_hull,
                      int64_t hole_count, const double* holes) {
  for (int64_t s = 0; s < segment_count; ++s) {
    mesher.insert_segment(map[static_cast<std::size_t>(segments[2 * s])],
                          map[static_cast<std::size_t>(segments[2 * s + 1])],
                          static_cast<int32_t>(s));
  }
  if (keep_convex_hull != 0) mesher.constrain_hull();
  mesher.carve(keep_convex_hull != 0, hole_count, holes);
}

int32_t constrained_delaunay(int64_t point_count, const double* points, int64_t segment_count,
                             const int32_t* segments, int64_t hole_count, const double* holes,
                             int32_t keep_convex_hull, double min_angle_degrees, double max_area,
                             phx_mc_mesh** mesh) {
  if (mesh == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *mesh = nullptr;
  if (point_count < 0 || !addressable(point_count, 2, sizeof(double)) ||
      !addressable(point_count, 1, sizeof(int32_t)) ||
      !addressable(segment_count, 2, sizeof(int32_t)) ||
      !addressable(hole_count, 2, sizeof(double)) ||
      (point_count > 0 && points == nullptr) ||
      (segment_count > 0 && segments == nullptr) || (hole_count > 0 && holes == nullptr) ||
      (keep_convex_hull != 0 && keep_convex_hull != 1) ||
      !(min_angle_degrees >= 0.0 && min_angle_degrees < 60.0) || !(max_area > 0.0)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  // Vertex and constraint identifiers are stored as int32 indices. Keep only
  // their actual representability bounds; this is not a work/entity quota.
  constexpr int64_t kMaxConstraintCount =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
  if (point_count > kMaxMeshPoints || segment_count > kMaxConstraintCount) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  int32_t status = validate_points(points, point_count, 2, nullptr);
  if (status == PHX_MC_OK) {
    status = validate_points(holes, hole_count, 2, nullptr);
  }
  if (status != PHX_MC_OK) {
    return status;
  }
  for (int64_t k = 0; k < 2 * segment_count; ++k) {
    if (segments[k] < 0 || segments[k] >= point_count) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  if (point_count == 0) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  auto result = std::make_unique<phx_mc_mesh>();
  result->dimension = 2;
  result->input_point_count = point_count;
  Triangulation2D triangulation;
  status = initialize_points(triangulation, points, point_count, segment_count, segments,
                             result->vertex_map);
  if (status != PHX_MC_OK) return status;
  const auto authored = static_cast<int32_t>(point_count);
  const auto estimate = cdt2d_detail::estimate_elements(triangulation, authored, max_area);
  std::optional<ConstrainedMesher> mesher;
  try {
    // Keep the ordinary refinement ordering for higher angle requests.
    const auto grid =
        min_angle_degrees <= 30.0 ? seed_grid(estimate, max_area, point_count) : std::nullopt;
    if (grid) {
      {
        ConstrainedMesher coarse(triangulation, authored, min_angle_degrees, max_area,
                                cdt2d_detail::AreaEstimate{});
        partition_domain(coarse, segments, segment_count, result->vertex_map, keep_convex_hull,
                         hole_count, holes);
      }
      // A seed exactly on a constraint has an order-dependent carve side.
      // Do not change that caller's domain through a different initial mesh.
      const auto seeded = hole_seeds_clear(triangulation, hole_count, holes)
                              ? interior_seed_bank(triangulation, *grid)
                              : std::vector<double>{};
      const bool prepared = !seeded.empty();
      status = initialize_points(triangulation, prepared ? seeded.data() : points,
                                 prepared ? static_cast<int64_t>(seeded.size() / 2) : point_count,
                                 segment_count, segments, result->vertex_map);
      if (status != PHX_MC_OK) return status;
      result->vertex_map.resize(static_cast<std::size_t>(authored));
    }
    mesher.emplace(triangulation, authored, min_angle_degrees, max_area, estimate);
    partition_domain(*mesher, segments, segment_count, result->vertex_map, keep_convex_hull,
                     hole_count, holes);
    if (min_angle_degrees > 0.0 || std::isfinite(max_area)) status = mesher->refine();
  } catch (const StatusError& error) {
    return error.status;
  }
  // Cells and their segment rows in the order canonicalize_cells gives: each
  // kept triangle rotated to start at its smallest vertex (segment k stays
  // opposite vertex k), records sorted lexicographically by their vertices.
  // The vertex tuples are distinct, so that order is unique; it is built
  // here by a counting sort on the first vertex straight from the slots, then
  // each group ordered by rank (by radix when large), as sort_cell_records.
  const std::size_t vertex_total = static_cast<std::size_t>(triangulation.vertex_count());
  result->points = std::move(triangulation.coordinates);
  const std::size_t slots = triangulation.triangles.size();
  const auto kept = [&](std::size_t t) {
    return triangulation.live[t] != 0 && triangulation.region[t] != 0 &&
           !triangulation.is_ghost(static_cast<int32_t>(t));
  };
  const auto first_of = [](const Triangle2D& triangle) {
    int first = triangle.v[1] < triangle.v[0] ? 1 : 0;
    first = triangle.v[2] < triangle.v[first] ? 2 : first;
    return first;
  };
  std::vector<std::size_t> group_end(vertex_total + 1, 0);
  for (std::size_t t = 0; t < slots; ++t) {
    if (kept(t)) {
      const Triangle2D& triangle = triangulation.triangles[t];
      ++group_end[static_cast<std::size_t>(triangle.v[first_of(triangle)]) + 1];
    }
  }
  std::partial_sum(group_end.begin(), group_end.end(), group_end.begin());
  const std::size_t cell_count = group_end[vertex_total];
  const std::unique_ptr<int32_t[]> rows(new int32_t[6 * cell_count]);
  for (std::size_t t = 0; t < slots; ++t) {
    if (!kept(t)) continue;
    const Triangle2D& triangle = triangulation.triangles[t];
    const int first = first_of(triangle);
    int32_t* row = rows.get() + 6 * group_end[static_cast<std::size_t>(triangle.v[first])]++;
    for (int k = 0; k < 3; ++k) {
      const int r = (first + k) % 3;
      row[k] = triangle.v[r];
      const int32_t id = triangulation.constraints[3 * t + static_cast<std::size_t>(r)];
      row[3 + k] = id >= 0 ? id : -1;
    }
  }
  // group_end[v] is now the end of the group of first vertex v.
  int bits = 1;
  while ((std::uint64_t{1} << bits) <= static_cast<std::uint64_t>(vertex_total)) ++bits;
  std::unique_ptr<int32_t[]> scratch;
  std::vector<std::size_t> radix_offsets;
  for (std::size_t v = 0, begin = 0; v < vertex_total; begin = group_end[v++]) {
    const std::size_t size = group_end[v] - begin;
    if (size > 64) {
      if (!scratch) scratch.reset(new int32_t[6 * cell_count]);
      radix_sort_rows<6>(rows.get() + 6 * begin, scratch.get(), size, 1, 3, bits,
                         radix_offsets);
    } else if (size > 1) {
      rank_sort_group<6, 64>(rows.get() + 6 * begin, size, 3);
    }
  }
  result->cells.resize(3 * cell_count);
  result->cell_segments.resize(3 * cell_count);
  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    std::copy_n(rows.get() + 6 * cell, 3, result->cells.data() + 3 * cell);
    std::copy_n(rows.get() + 6 * cell + 3, 3, result->cell_segments.data() + 3 * cell);
  }
  mesher->publish_witnesses(*result);
  *mesh = result.release();
  return status;
}

}  // namespace
}  // namespace phx::mc

extern "C" {

int32_t phx_mc_constrained_delaunay_2d(int64_t point_count, const double* points,
                                       int64_t segment_count, const int32_t* segments,
                                       int64_t hole_count, const double* holes,
                                       int32_t keep_convex_hull, double min_angle_degrees,
                                       double max_area, phx_mc_mesh** mesh) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    return phx::mc::constrained_delaunay(point_count, points, segment_count, segments, hole_count,
                                         holes, keep_convex_hull, min_angle_degrees, max_area, mesh);
  });
}

}  // extern "C"
