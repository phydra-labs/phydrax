//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Source-aware Delaunay refinement by checked circumcenter cavity insertion.
// Any remaining criterion violation is reported per cell as UNRESOLVED.
#include "refine3d.hpp"
#include "improve3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cavity.hpp"
#include "geometry3.hpp"
#include "local_incidence.hpp"
#include "predicates.hpp"
#include "phydrax_meshcore.h"
#include "exact_scalar.hpp"

namespace phx::mc {
double cell_size_target(const std::array<double, 4>& corner_sizes) noexcept {
  double scale = 0.0;
  for (double size : corner_sizes) {
    if (!(size > 0.0)) return std::numeric_limits<double>::infinity();
    scale = std::max(scale, size);
  }
  // Normalized by the largest target so the sum cannot overflow.
  double normalized_sum = 0.0;
  for (double size : corner_sizes) normalized_sum += size / scale;
  return std::min(scale, scale * (0.25 * normalized_sum));
}

namespace {

template <class Exit>
struct ScopeExit {
  Exit exit;
  ~ScopeExit() { exit(); }
};
template <class Exit>
ScopeExit(Exit) -> ScopeExit<Exit>;

bool tetrahedron_geometry(const TetMesh& mesh, const TetSnapshotCell& cell,
                          std::array<std::int32_t, 4>& slots,
                          geometry::TetrahedronShape& shape) {
  const double* points[4];
  for (int corner = 0; corner < 4; ++corner) {
    slots[corner] = mesh.vertex_slot(cell.vertices[corner]);
    if (!mesh.live_vertex(slots[corner])) return false;
    points[corner] = mesh.point(slots[corner]);
  }
  return geometry::tetrahedron_shape(points, shape);
}

bool same_carrier(const double* first, const double* second) {
  return first[0] == second[0] && first[1] == second[1] && first[2] == second[2];
}

// (a + b) / 2 when it is a binary64 number.
bool exact_half_sum(double a, double b, double& half) {
  half = 0.5 * a + 0.5 * b;
  return (Dyadic(half) - (Dyadic(a) + Dyadic(b)) * Dyadic(0.5)).is_zero();
}

// The line parameter on source segment `row` (corners `corners`) of the
// exact position of `slot`: its own witness parameter, or a binary64 vertex
// that lies exactly on the closed segment.
bool segment_parameter(const TetMesh& mesh, std::int32_t slot, std::int32_t row,
                       const double* const* corners, double& parameter) {
  const SourceWitness& witness = mesh.witness(slot);
  if (witness.stratum == SourceStratum::kSegment) {
    parameter = witness.parameters[0];
    return witness.entity == row;
  }
  if (witness.stratum != SourceStratum::kNone) return false;
  const double* point = mesh.point(slot);
  for (int corner = 0; corner < 2; ++corner) {
    if (same_carrier(point, corners[corner])) {
      parameter = corner;
      return true;
    }
  }
  double parameters[2]{};
  source_locator(corners, SourceStratum::kSegment, point, parameters, false);
  double carrier[3];
  double deviation = 0.0;
  if (!source_carrier(corners, SourceStratum::kSegment, parameters, carrier, deviation) ||
      deviation != 0.0 || !same_carrier(carrier, point)) {
    return false;
  }
  parameter = parameters[0];
  return true;
}

// The exact barycentric weights (l1, l2) on source face `row` of the exact
// position of `slot`: its facet witness, a segment witness on an edge of the
// face, or a binary64 vertex at a face corner. False when a weight is not a
// binary64 number.
bool facet_weights(const TetMesh& mesh, std::int32_t slot, std::int32_t row,
                   const double* const* corners, double* weights) {
  const SourceWitness& witness = mesh.witness(slot);
  if (witness.stratum == SourceStratum::kFacet) {
    weights[0] = witness.parameters[0];
    weights[1] = witness.parameters[1];
    return witness.entity == row;
  }
  double barycentric[3]{};
  if (witness.stratum == SourceStratum::kSegment) {
    const double* ends[3]{};
    if (!mesh.source_corners(SourceStratum::kSegment, witness.entity, ends)) return false;
    const double* const* at_first = std::find(corners, corners + 3, ends[0]);
    const double* const* at_second = std::find(corners, corners + 3, ends[1]);
    if (at_first == corners + 3 || at_second == corners + 3) return false;
    const double parameter = witness.parameters[0];
    const double complement = 1.0 - parameter;
    if (!(Dyadic(complement) + Dyadic(parameter) - Dyadic(1.0)).is_zero()) return false;
    barycentric[at_first - corners] = complement;
    barycentric[at_second - corners] = parameter;
  } else if (witness.stratum == SourceStratum::kNone) {
    const double* point = mesh.point(slot);
    const double* const* corner = std::find_if(
        corners, corners + 3, [&](const double* candidate) { return same_carrier(point, candidate); });
    if (corner == corners + 3) return false;
    barycentric[corner - corners] = 1.0;
  } else {
    return false;
  }
  weights[0] = barycentric[1];
  weights[1] = barycentric[2];
  return true;
}

// Installs the witness (stratum, row, parameters) on `vertex` with its
// carrier, the correctly rounded exact source point, and the exact
// realization bound as its deviation.
bool install_witness(const double* const* corners, SourceStratum stratum, std::int32_t row,
                     const double* parameters, TetEditVertex& vertex) {
  if (!source_parameters_valid(stratum, parameters)) return false;
  double deviation = 0.0;
  if (!source_carrier(corners, stratum, parameters, vertex.carrier.data(), deviation)) {
    return false;
  }
  vertex.witness.stratum = stratum;
  vertex.witness.entity = row;
  vertex.witness.parameters[0] = parameters[0];
  vertex.witness.parameters[1] = parameters[1];
  vertex.witness.deviation = deviation;
  return true;
}

// The rows of source face group `rows` on whose plane the exact positions of
// all vertex slots `slots` lie (for a segment group: on whose line), so that
// a point located among them stays on the plane of the boundary piece it
// splits; a non-planar group never sends it to a far row.
void rows_through(const TetMesh& mesh, SourceStratum stratum, std::span<const std::int32_t> rows,
                  std::span<const std::int32_t> slots, std::vector<std::int32_t>& through) {
  through.clear();
  std::vector<std::array<Expansion, 3>> positions(slots.size());
  for (std::size_t vertex = 0; vertex < slots.size(); ++vertex) {
    if (!mesh.exact_position(slots[vertex], positions[vertex].data())) return;
  }
  for (std::int32_t row : rows) {
    const double* corners[3]{};
    if (!mesh.source_corners(stratum, row, corners)) continue;
    Expansion first[3];
    for (int axis = 0; axis < 3; ++axis) {
      first[axis] = Expansion::difference(corners[1][axis], corners[0][axis]);
    }
    const bool on = std::all_of(positions.begin(), positions.end(), [&](const auto& position) {
      Expansion offset[3];
      for (int axis = 0; axis < 3; ++axis) {
        offset[axis] = position[static_cast<std::size_t>(axis)] - Expansion(corners[0][axis]);
      }
      if (stratum == SourceStratum::kSegment) {
        return (first[1] * offset[2] - first[2] * offset[1]).is_zero() &&
               (first[2] * offset[0] - first[0] * offset[2]).is_zero() &&
               (first[0] * offset[1] - first[1] * offset[0]).is_zero();
      }
      Expansion second[3];
      for (int axis = 0; axis < 3; ++axis) {
        second[axis] = Expansion::difference(corners[2][axis], corners[0][axis]);
      }
      const Expansion distance = (first[1] * second[2] - first[2] * second[1]) * offset[0] +
                                 (first[2] * second[0] - first[0] * second[2]) * offset[1] +
                                 (first[0] * second[1] - first[1] * second[0]) * offset[2];
      return distance.is_zero();
    });
    if (on) through.push_back(row);
  }
}

// The source row among `rows` (segment or face rows of one source group)
// whose element holds `point` in its locator chart, with its parameters. A
// point that rounding leaves just outside every element is placed on the
// element whose parameters it violates least, at the clamped parameters, so
// the exact witness point is always a point of the group. False when no row
// yields finite parameters.
bool locate_in_rows(const TetMesh& mesh, SourceStratum stratum,
                    std::span<const std::int32_t> rows, const double* point,
                    std::int32_t& located, double* parameters) {
  located = -1;
  double least = std::numeric_limits<double>::infinity();
  for (std::int32_t row : rows) {
    const double* corners[3]{};
    if (!mesh.source_corners(stratum, row, corners)) continue;
    double candidate[2]{};
    source_locator(corners, stratum, point, candidate, false);
    if (!std::isfinite(candidate[0]) || !std::isfinite(candidate[1])) continue;
    const double violation =
        stratum == SourceStratum::kSegment
            ? std::max({-candidate[0], candidate[0] - 1.0, 0.0})
            : std::max({-candidate[0], -candidate[1], candidate[0] + candidate[1] - 1.0, 0.0});
    if (violation < least) {
      least = violation;
      located = row;
      parameters[0] = candidate[0];
      parameters[1] = candidate[1];
      if (violation == 0.0 && source_parameters_valid(stratum, parameters)) return true;
    }
  }
  if (located < 0) return false;
  clamp_source_parameters(stratum, parameters);
  return true;
}

bool mean_target_size(const std::array<std::uint64_t, 4>& vertices,
                      const std::unordered_map<std::uint64_t, double>& sizes,
                      double& target) {
  std::array<double, 4> values{};
  for (std::size_t corner = 0; corner < vertices.size(); ++corner) {
    const auto found = sizes.find(vertices[corner]);
    if (found == sizes.end()) return false;
    values[corner] = found->second;
  }
  target = cell_size_target(values);
  return std::isfinite(target);
}

bool violates(const TetMesh& mesh, const TetSnapshotCell& cell,
              double radius_edge_bound,
              const std::unordered_map<std::uint64_t, double>& sizes) {
  std::array<std::int32_t, 4> slots{};
  geometry::TetrahedronShape shape;
  if (!tetrahedron_geometry(mesh, cell, slots, shape)) return true;
  if (shape.radius_edge_ratio > radius_edge_bound) return true;
  double target = 0.0;
  return mean_target_size(cell.vertices, sizes, target) &&
         shape.circumradius > target;
}

// Why refinement left a cell: an unmet reason and, for INPUT_ANGLE, the apex
// vertex ID and angle in degrees of the acute input feature that protected the
// boundary split.
struct UnmetCause {
  std::int32_t reason = PHX_MC_TET_UNMET_NO_IMPROVEMENT;
  std::uint64_t apex_vertex = kNoUnmetApex;
  double apex_angle = std::numeric_limits<double>::quiet_NaN();
};

void collect_unmet(const TetMesh& mesh, const TetMeshSnapshot& snapshot,
                   double radius_edge_bound,
                   const std::unordered_map<std::uint64_t, double>& sizes,
                   const std::unordered_map<std::uint64_t, UnmetCause>& causes,
                   std::vector<TetUnmetRecord>& records,
                   std::int64_t& unmet_cells) {
  records.clear();
  if (snapshot.cells.size() <= records.max_size() / 2) {
    records.reserve(2 * snapshot.cells.size());
  }
  unmet_cells = 0;
  for (const TetSnapshotCell& cell : snapshot.cells) {
    std::array<std::int32_t, 4> slots{};
    geometry::TetrahedronShape shape;
    bool cell_unmet = false;
    if (!tetrahedron_geometry(mesh, cell, slots, shape)) {
      records.push_back({cell.id, PHX_MC_TET_UNMET_VALIDITY,
                         PHX_MC_TET_UNMET_NONREPRESENTABLE, 0.0});
      ++unmet_cells;
      continue;
    }
    const auto found = causes.find(cell.id);
    const UnmetCause cause = found == causes.end() ? UnmetCause{} : found->second;
    if (shape.radius_edge_ratio > radius_edge_bound) {
      records.push_back({cell.id, PHX_MC_TET_UNMET_RADIUS_EDGE, cause.reason,
                         shape.radius_edge_ratio, cause.apex_vertex, cause.apex_angle});
      cell_unmet = true;
    }
    double target = 0.0;
    if (mean_target_size(cell.vertices, sizes, target) &&
        shape.circumradius > target) {
      records.push_back({cell.id, PHX_MC_TET_UNMET_SIZE, cause.reason,
                         shape.circumradius / target, cause.apex_vertex, cause.apex_angle});
      cell_unmet = true;
    }
    if (cell_unmet) ++unmet_cells;
  }
}

struct GridKey {
  std::array<std::int64_t, 3> cell{};
  bool operator==(const GridKey&) const noexcept = default;
};
struct GridHash {
  std::size_t operator()(const GridKey& key) const noexcept {
    std::size_t seed = 0;
    for (std::int64_t value : key.cell) {
      seed ^= std::hash<std::int64_t>{}(value) + 0x9e3779b9U + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};
using HashGrid = std::unordered_map<GridKey, std::vector<std::int32_t>, GridHash>;

bool grid_cell(const double* point, double cell, GridKey& key) {
  for (int axis = 0; axis < 3; ++axis) {
    const double scaled = std::floor(point[axis] / cell);
    if (!(std::abs(scaled) < 0x1p62)) return false;
    key.cell[static_cast<std::size_t>(axis)] = static_cast<std::int64_t>(scaled);
  }
  return true;
}

// Calls `visit` on the items of every occupied cell meeting [lower, upper];
// a box spanning more cells than are occupied reads every occupied cell.
template <class Visit>
bool visit_box(const HashGrid& grid, double cell, const double* lower, const double* upper,
               Visit&& visit) {
  GridKey first;
  GridKey last;
  if (!grid_cell(lower, cell, first) || !grid_cell(upper, cell, last)) return false;
  double span = 1.0;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    span *= static_cast<double>(last.cell[axis] - first.cell[axis]) + 1.0;
  }
  if (span > static_cast<double>(grid.size())) {
    for (const auto& [key, items] : grid) {
      (void)key;
      for (std::int32_t item : items) visit(item);
    }
    return true;
  }
  for (std::int64_t x = first.cell[0]; x <= last.cell[0]; ++x) {
    for (std::int64_t y = first.cell[1]; y <= last.cell[1]; ++y) {
      for (std::int64_t z = first.cell[2]; z <= last.cell[2]; ++z) {
        const auto found = grid.find(GridKey{{x, y, z}});
        if (found == grid.end()) continue;
        for (std::int32_t item : found->second) visit(item);
      }
    }
  }
  return true;
}

bool register_box(HashGrid& grid, double cell, const double* lower, const double* upper,
                  std::int32_t item) {
  GridKey first;
  GridKey last;
  if (!grid_cell(lower, cell, first) || !grid_cell(upper, cell, last)) return false;
  for (std::int64_t x = first.cell[0]; x <= last.cell[0]; ++x) {
    for (std::int64_t y = first.cell[1]; y <= last.cell[1]; ++y) {
      for (std::int64_t z = first.cell[2]; z <= last.cell[2]; ++z) {
        grid[GridKey{{x, y, z}}].push_back(item);
      }
    }
  }
  return true;
}

// lambda(v) for every live vertex slot: the distance from v to the nearest
// feature not incident to v. Point features are the live vertices, line
// features the constrained segments and constrained-facet edges, plane
// features the constrained facets. Dead slots keep +inf. lambda is floored at
// the coordinate spacing at v. Fails only on non-finite geometry.
bool local_feature_sizes(const TetMesh& mesh, std::size_t slot_count,
                         std::span<const TetSnapshotFacet> facets,
                         std::span<const TetSnapshotSegment> segments,
                         std::vector<double>& sizes, double& smallest) {
  const double infinity = std::numeric_limits<double>::infinity();
  sizes.assign(slot_count, infinity);
  // The shortest incident edge bounds lambda from above and sets the query radius.
  std::vector<double> lengths;
  const TetrahedralComplex& complex = mesh.complex();
  for (std::size_t row = 0; row < complex.tets.size(); ++row) {
    const std::int32_t cell = static_cast<std::int32_t>(row);
    if (!complex.live(cell) || is_ghost(complex.tets[row])) continue;
    const Tetrahedron& tetrahedron = complex.tets[row];
    for (int first = 0; first < 4; ++first) {
      for (int second = first + 1; second < 4; ++second) {
        const std::int32_t a = tetrahedron.v[first];
        const std::int32_t b = tetrahedron.v[second];
        const double length = std::sqrt(geometry::squared_distance(mesh.point(a), mesh.point(b)));
        sizes[static_cast<std::size_t>(a)] = std::min(sizes[static_cast<std::size_t>(a)], length);
        sizes[static_cast<std::size_t>(b)] = std::min(sizes[static_cast<std::size_t>(b)], length);
        lengths.push_back(length);
      }
    }
  }
  if (lengths.empty()) return false;

  std::vector<std::array<std::int32_t, 2>> lines;
  std::vector<std::array<std::int32_t, 3>> planes;
  planes.reserve(facets.size());
  lines.reserve(segments.size() + 3 * facets.size());
  const auto ordered = [](std::int32_t a, std::int32_t b) {
    return a < b ? std::array<std::int32_t, 2>{a, b} : std::array<std::int32_t, 2>{b, a};
  };
  for (const TetSnapshotSegment& segment : segments) {
    const std::int32_t a = mesh.vertex_slot(segment.vertices[0]);
    const std::int32_t b = mesh.vertex_slot(segment.vertices[1]);
    if (!mesh.live_vertex(a) || !mesh.live_vertex(b)) return false;
    lines.push_back(ordered(a, b));
  }
  for (const TetSnapshotFacet& facet : facets) {
    std::array<std::int32_t, 3> slots{};
    for (std::size_t corner = 0; corner < 3; ++corner) {
      slots[corner] = mesh.vertex_slot(facet.vertices[corner]);
      if (!mesh.live_vertex(slots[corner])) return false;
    }
    planes.push_back(slots);
    for (std::size_t corner = 0; corner < 3; ++corner) {
      lines.push_back(ordered(slots[corner], slots[(corner + 1) % 3]));
    }
  }
  std::sort(lines.begin(), lines.end());
  lines.erase(std::unique(lines.begin(), lines.end()), lines.end());

  // Cells of the median edge length, coarsened so that no feature box spans
  // more than 17 cells per axis.
  std::nth_element(lengths.begin(), lengths.begin() + static_cast<std::ptrdiff_t>(lengths.size() / 2),
                   lengths.end());
  double cell = lengths[lengths.size() / 2];
  const auto feature_box = [&](std::span<const std::int32_t> corners, double* lower,
                               double* upper) {
    for (int axis = 0; axis < 3; ++axis) {
      lower[axis] = upper[axis] = mesh.point(corners[0])[axis];
      for (std::int32_t corner : corners.subspan(1)) {
        lower[axis] = std::min(lower[axis], mesh.point(corner)[axis]);
        upper[axis] = std::max(upper[axis], mesh.point(corner)[axis]);
      }
    }
  };
  double extent = 0.0;
  for (const auto& plane : planes) {
    double lower[3], upper[3];
    feature_box(plane, lower, upper);
    for (int axis = 0; axis < 3; ++axis) extent = std::max(extent, upper[axis] - lower[axis]);
  }
  cell = std::max(cell, extent / 16.0);
  if (!(cell > 0.0) || !std::isfinite(cell)) return false;

  HashGrid point_grid;
  HashGrid line_grid;
  HashGrid plane_grid;
  for (std::size_t slot = 0; slot < slot_count; ++slot) {
    const std::int32_t vertex = static_cast<std::int32_t>(slot);
    if (!mesh.live_vertex(vertex)) continue;
    if (!register_box(point_grid, cell, mesh.point(vertex), mesh.point(vertex), vertex)) {
      return false;
    }
  }
  for (std::size_t row = 0; row < lines.size(); ++row) {
    double lower[3], upper[3];
    feature_box(lines[row], lower, upper);
    if (!register_box(line_grid, cell, lower, upper, static_cast<std::int32_t>(row))) return false;
  }
  for (std::size_t row = 0; row < planes.size(); ++row) {
    double lower[3], upper[3];
    feature_box(planes[row], lower, upper);
    if (!register_box(plane_grid, cell, lower, upper, static_cast<std::int32_t>(row))) return false;
  }

  smallest = infinity;
  for (std::size_t slot = 0; slot < slot_count; ++slot) {
    const std::int32_t vertex = static_cast<std::int32_t>(slot);
    if (!mesh.live_vertex(vertex)) continue;
    const double* p = mesh.point(vertex);
    double lambda = sizes[slot];
    if (!std::isfinite(lambda)) return false;
    const double lower[3]{p[0] - lambda, p[1] - lambda, p[2] - lambda};
    const double upper[3]{p[0] + lambda, p[1] + lambda, p[2] + lambda};
    const bool visited =
        visit_box(point_grid, cell, lower, upper, [&](std::int32_t other) {
          if (other != vertex) {
            lambda = std::min(lambda, std::sqrt(geometry::squared_distance(p, mesh.point(other))));
          }
        }) &&
        visit_box(line_grid, cell, lower, upper, [&](std::int32_t row) {
          const auto& line = lines[static_cast<std::size_t>(row)];
          if (line[0] != vertex && line[1] != vertex) {
            lambda = std::min(lambda, geometry::point_segment_distance(
                                          p, mesh.point(line[0]), mesh.point(line[1])));
          }
        }) &&
        visit_box(plane_grid, cell, lower, upper, [&](std::int32_t row) {
          const auto& plane = planes[static_cast<std::size_t>(row)];
          if (plane[0] != vertex && plane[1] != vertex && plane[2] != vertex) {
            lambda = std::min(lambda, geometry::point_triangle_distance(
                                          p, mesh.point(plane[0]), mesh.point(plane[1]),
                                          mesh.point(plane[2])));
          }
        });
    if (!visited || !std::isfinite(lambda)) return false;
    // Features nearer than the coordinate spacing at v cannot be separated by
    // a representable point; lambda is floored there, which keeps it positive.
    double resolution = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double magnitude = std::abs(p[axis]);
      resolution = std::max(resolution, std::nextafter(magnitude, infinity) - magnitude);
    }
    lambda = std::max(lambda, resolution);
    sizes[slot] = lambda;
    smallest = std::min(smallest, lambda);
  }
  return std::isfinite(smallest);
}

// Spacing queries over hash grids whose cells double per level. A query of
// radius r reads the 27 cells around the point on the finest level whose
// cells are at least r; a level is built on its first query.
class SpacingIndex {
 public:
  SpacingIndex(const TetMesh& mesh, double base) : mesh_(mesh), base_(base) {}

  bool insert(std::int32_t vertex) {
    vertices_.push_back(vertex);
    for (std::size_t level = 0; level < levels_.size(); ++level) {
      if (built_[level] != 0 && !add(level, vertex)) return false;
    }
    return true;
  }

  // True when no indexed vertex lies closer than `radius` to `point`.
  bool clear(const double* point, double radius) {
    if (!(radius > 0.0) || !std::isfinite(radius)) return false;
    std::size_t level = 0;
    while (std::ldexp(base_, static_cast<int>(level)) < radius) ++level;
    if (!build(level)) return false;
    GridKey center;
    if (!grid_cell(point, std::ldexp(base_, static_cast<int>(level)), center)) return false;
    for (std::int64_t dx = -1; dx <= 1; ++dx) {
      for (std::int64_t dy = -1; dy <= 1; ++dy) {
        for (std::int64_t dz = -1; dz <= 1; ++dz) {
          const auto found = levels_[level].find(
              GridKey{{center.cell[0] + dx, center.cell[1] + dy, center.cell[2] + dz}});
          if (found == levels_[level].end()) continue;
          for (std::int32_t vertex : found->second) {
            if (std::sqrt(geometry::squared_distance(point, mesh_.point(vertex))) < radius) {
              return false;
            }
          }
        }
      }
    }
    return true;
  }

 private:
  bool build(std::size_t level) {
    if (level >= levels_.size()) {
      levels_.resize(level + 1);
      built_.resize(level + 1, 0);
    }
    if (built_[level] != 0) return true;
    for (std::int32_t vertex : vertices_) {
      if (!add(level, vertex)) return false;
    }
    built_[level] = 1;
    return true;
  }

  bool add(std::size_t level, std::int32_t vertex) {
    GridKey key;
    if (!grid_cell(mesh_.point(vertex), std::ldexp(base_, static_cast<int>(level)), key)) {
      return false;
    }
    levels_[level][key].push_back(vertex);
    return true;
  }

  const TetMesh& mesh_;
  double base_;
  std::vector<std::int32_t> vertices_;
  std::vector<HashGrid> levels_;
  std::vector<std::uint8_t> built_;
};

// Input angles whose cosine is at most this are right or obtuse: a rounded
// realization of a right angle is not an acute feature.
constexpr double kRightAngleCosine = 0x1p-20;

double dot3(const double* u, const double* v) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; }

void cross3(const double* u, const double* v, double* out) {
  out[0] = u[1] * v[2] - u[2] * v[1];
  out[1] = u[2] * v[0] - u[0] * v[2];
  out[2] = u[0] * v[1] - u[1] * v[0];
}

// Cosine of the angle between u and v; NaN when either is zero.
double cosine3(const double* u, const double* v) {
  return dot3(u, v) / (std::sqrt(dot3(u, u)) * std::sqrt(dot3(v, v)));
}

double degrees(double cosine) {
  return std::acos(std::clamp(cosine, -1.0, 1.0)) * (180.0 / std::numbers::pi);
}

// The smallest angle in degrees between the ray along `ray` and the planar
// sector spanned by `first` and `second` at the same apex.
double ray_sector_angle(const double* ray, const double* first, const double* second) {
  double normal[3];
  cross3(first, second, normal);
  const double normal_squared = dot3(normal, normal);
  if (normal_squared > 0.0) {
    const double height = dot3(ray, normal) / normal_squared;
    const double projection[3]{ray[0] - height * normal[0], ray[1] - height * normal[1],
                               ray[2] - height * normal[2]};
    double first_side[3];
    double second_side[3];
    cross3(first, projection, first_side);
    cross3(projection, second, second_side);
    if (dot3(first_side, normal) >= 0.0 && dot3(second_side, normal) >= 0.0) {
      const double cosine = cosine3(ray, projection);
      return std::isnan(cosine) ? 90.0 : degrees(cosine);
    }
  }
  return std::min(degrees(cosine3(ray, first)), degrees(cosine3(ray, second)));
}

// An acute input feature of the constrained complex entering refinement:
// constrained segments meeting at the vertex `apex` at an angle below 90
// degrees, or a segment meeting a facet (a constrained-facet group not
// containing it) there at such an angle; or, with `ridge` set, two facets of
// different groups meeting along the constrained edge (apex, ridge) at a
// dihedral angle below 90 degrees measured through the domain. `angle` is the
// smallest such angle in degrees. Its protected region holds the points
// closer than `radius` to the apex (to the ridge edge).
struct AcuteFeature {
  std::int32_t apex = -1;
  std::int32_t ridge = -1;
  double radius = 0.0;
  double angle = 0.0;
};

// The protected radius of an acute feature of angle `angle` (degrees) whose
// apex has local feature size `feature_size`: inside the wedge, a point at
// distance d from the apex lies within d * sin(angle) of the other feature,
// so the wedge is narrower than the apex's feature size exactly within
// feature_size / sin(angle) of it. Boundary splits there encroach each
// other ever closer to the apex.
double protected_radius(double feature_size, double angle) {
  return feature_size / std::sin(angle * (std::numbers::pi / 180.0));
}

// The acute features of the constrained complex (`facets`, `segments`) of
// `mesh`, apexes in slot order then ridges in slot-pair order; the radius is
// protected_radius of the local feature size `feature_sizes` at the apex (the
// smaller one of the ridge endpoints). False on a dead constrained vertex.
bool acute_input_features(const TetMesh& mesh, const detail::LocalIncidence& incidence,
                          std::span<const TetSnapshotFacet> facets,
                          std::span<const TetSnapshotSegment> segments,
                          std::span<const double> feature_sizes,
                          std::vector<AcuteFeature>& features) {
  features.clear();
  using Edge = std::pair<std::int32_t, std::int32_t>;
  const auto ordered = [](std::int32_t a, std::int32_t b) {
    return a < b ? Edge{a, b} : Edge{b, a};
  };
  struct Corner {
    std::int32_t first = -1;
    std::int32_t second = -1;
    std::int32_t group = kNoConstraint;
  };
  std::map<std::int32_t, std::vector<std::int32_t>> segment_ends;
  std::unordered_map<std::int32_t, std::vector<Corner>> facet_corners;
  std::set<std::pair<std::int32_t, Edge>> group_edges;
  std::map<Edge, std::vector<std::pair<std::int32_t, std::int32_t>>> edge_facets;
  for (const TetSnapshotSegment& segment : segments) {
    const std::int32_t a = mesh.vertex_slot(segment.vertices[0]);
    const std::int32_t b = mesh.vertex_slot(segment.vertices[1]);
    if (!mesh.live_vertex(a) || !mesh.live_vertex(b)) return false;
    segment_ends[a].push_back(b);
    segment_ends[b].push_back(a);
  }
  for (const TetSnapshotFacet& facet : facets) {
    std::int32_t slots[3];
    for (int corner = 0; corner < 3; ++corner) {
      slots[corner] = mesh.vertex_slot(facet.vertices[static_cast<std::size_t>(corner)]);
      if (!mesh.live_vertex(slots[corner])) return false;
    }
    for (int corner = 0; corner < 3; ++corner) {
      const std::int32_t next = slots[(corner + 1) % 3];
      const std::int32_t last = slots[(corner + 2) % 3];
      facet_corners[slots[corner]].push_back({next, last, facet.source});
      group_edges.emplace(facet.source, ordered(slots[corner], next));
      edge_facets[ordered(slots[corner], next)].emplace_back(last, facet.source);
    }
  }
  const auto direction = [&](std::int32_t from, std::int32_t to, double* out) {
    for (int axis = 0; axis < 3; ++axis) out[axis] = mesh.point(to)[axis] - mesh.point(from)[axis];
  };

  for (const auto& [apex, ends] : segment_ends) {
    double smallest = 90.0;
    for (std::size_t first = 0; first < ends.size(); ++first) {
      double ray[3];
      direction(apex, ends[first], ray);
      for (std::size_t second = first + 1; second < ends.size(); ++second) {
        double other[3];
        direction(apex, ends[second], other);
        const double cosine = cosine3(ray, other);
        if (cosine > kRightAngleCosine) smallest = std::min(smallest, degrees(cosine));
      }
      const auto corners = facet_corners.find(apex);
      if (corners == facet_corners.end()) continue;
      for (const Corner& corner : corners->second) {
        if (group_edges.contains({corner.group, ordered(apex, ends[first])})) continue;
        double sides[2][3];
        direction(apex, corner.first, sides[0]);
        direction(apex, corner.second, sides[1]);
        if (cosine3(ray, sides[0]) > kRightAngleCosine ||
            cosine3(ray, sides[1]) > kRightAngleCosine) {
          smallest = std::min(smallest, ray_sector_angle(ray, sides[0], sides[1]));
        }
      }
    }
    if (smallest < 90.0) {
      features.push_back({apex, -1,
                          protected_radius(feature_sizes[static_cast<std::size_t>(apex)], smallest),
                          smallest});
    }
  }

  // A wedge between the facets through (a, b, c) and (a, b, d) lies in the
  // domain when the finite cell across (a, b, c) on d's side exists.
  const auto domain_wedge = [&](std::int32_t a, std::int32_t b, std::int32_t c, std::int32_t d) {
    const int toward = orient3d(mesh.point(a), mesh.point(b), mesh.point(c), mesh.point(d));
    for (std::int32_t cell : incidence.edge_star(a, b)) {
      const Tetrahedron& tetrahedron = mesh.tet(cell);
      if (std::find(tetrahedron.v, tetrahedron.v + 4, c) == tetrahedron.v + 4) continue;
      for (std::int32_t x : tetrahedron.v) {
        if (x == a || x == b || x == c) continue;
        if (x == d || (toward != 0 && orient3d(mesh.point(a), mesh.point(b), mesh.point(c),
                                               mesh.point(x)) == toward)) {
          return true;
        }
      }
    }
    return false;
  };
  for (const auto& [edge, sides] : edge_facets) {
    const auto [a, b] = edge;
    double along[3];
    direction(a, b, along);
    const double length_squared = dot3(along, along);
    if (!(length_squared > 0.0)) continue;
    double smallest = 90.0;
    for (std::size_t first = 0; first < sides.size(); ++first) {
      for (std::size_t second = first + 1; second < sides.size(); ++second) {
        if (sides[first].second == sides[second].second) continue;
        double normal[2][3];
        for (const std::size_t side : {first, second}) {
          double offset[3];
          direction(a, sides[side].first, offset);
          const double height = dot3(offset, along) / length_squared;
          for (int axis = 0; axis < 3; ++axis) {
            normal[side == first ? 0 : 1][axis] = offset[axis] - height * along[axis];
          }
        }
        const double cosine = cosine3(normal[0], normal[1]);
        if (cosine > kRightAngleCosine &&
            (domain_wedge(a, b, sides[first].first, sides[second].first) ||
             domain_wedge(a, b, sides[second].first, sides[first].first))) {
          smallest = std::min(smallest, degrees(cosine));
        }
      }
    }
    if (smallest < 90.0) {
      features.push_back({a, b,
                          protected_radius(std::min(feature_sizes[static_cast<std::size_t>(a)],
                                                    feature_sizes[static_cast<std::size_t>(b)]),
                                           smallest),
                          smallest});
    }
  }
  return true;
}

// Point queries against the protected regions of acute features over one
// hash grid in which every feature is registered in the cells its region's
// bounding box meets.
class ProtectionIndex {
 public:
  ProtectionIndex(const TetMesh& mesh, std::span<const AcuteFeature> features)
      : mesh_(mesh), features_(features) {}

  bool build() {
    if (features_.empty()) return true;
    std::vector<double> radii;
    radii.reserve(features_.size());
    double extent = 0.0;
    for (const AcuteFeature& feature : features_) {
      double lower[3], upper[3];
      box(feature, lower, upper);
      for (int axis = 0; axis < 3; ++axis) extent = std::max(extent, upper[axis] - lower[axis]);
      radii.push_back(feature.radius);
    }
    std::nth_element(radii.begin(), radii.begin() + static_cast<std::ptrdiff_t>(radii.size() / 2),
                     radii.end());
    // Cells of the median radius, coarsened so that no region box spans more
    // than 17 cells per axis.
    cell_ = std::max(radii[radii.size() / 2], extent / 16.0);
    if (!(cell_ > 0.0) || !std::isfinite(cell_)) return false;
    for (std::size_t row = 0; row < features_.size(); ++row) {
      double lower[3], upper[3];
      box(features_[row], lower, upper);
      if (!register_box(grid_, cell_, lower, upper, static_cast<std::int32_t>(row))) return false;
    }
    return true;
  }

  // The lowest-indexed feature whose protected region holds `point`, else -1.
  std::int32_t find(const double* point) const {
    GridKey key;
    if (grid_.empty() || !grid_cell(point, cell_, key)) return -1;
    const auto found = grid_.find(key);
    if (found == grid_.end()) return -1;
    std::int32_t protecting = -1;
    for (std::int32_t row : found->second) {
      if (protecting >= 0 && row > protecting) continue;
      const AcuteFeature& feature = features_[static_cast<std::size_t>(row)];
      const double distance =
          feature.ridge < 0
              ? std::sqrt(geometry::squared_distance(point, mesh_.point(feature.apex)))
              : geometry::point_segment_distance(point, mesh_.point(feature.apex),
                                                 mesh_.point(feature.ridge));
      if (distance < feature.radius) protecting = row;
    }
    return protecting;
  }

 private:
  void box(const AcuteFeature& feature, double* lower, double* upper) const {
    const double* apex = mesh_.point(feature.apex);
    const double* end = feature.ridge < 0 ? apex : mesh_.point(feature.ridge);
    for (int axis = 0; axis < 3; ++axis) {
      lower[axis] = std::min(apex[axis], end[axis]) - feature.radius;
      upper[axis] = std::max(apex[axis], end[axis]) + feature.radius;
    }
  }

  const TetMesh& mesh_;
  std::span<const AcuteFeature> features_;
  HashGrid grid_;
  double cell_ = 0.0;
};

}  // namespace

std::int32_t refine_mesh(TetMesh& mesh, double radius_edge_bound,
                         std::span<const double> vertex_sizes,
                         std::int64_t& inserted_vertices,
                         std::int64_t& unmet_cells) {
  inserted_vertices = 0;
  unmet_cells = 0;
  if (!std::isfinite(radius_edge_bound) || radius_edge_bound <= 0.0) {
    return PHX_MC_INVALID_ARGUMENT;
  }

  TetMeshSnapshot initial;
  if (mesh.snapshot(initial) != TetEditStatus::kApplied) return PHX_MC_INTERNAL_ERROR;
  const std::int32_t installed = mesh.install_vertex_sizes(initial, vertex_sizes);
  if (installed != PHX_MC_OK) return installed;
  std::unordered_map<std::uint64_t, double>& sizes = mesh.vertex_sizes_;

  using EdgeKey = std::array<std::uint64_t, 2>;
  using FaceKey = std::array<std::uint64_t, 3>;
  const auto sorted_edge = [](std::uint64_t first, std::uint64_t second) {
    return first < second ? EdgeKey{first, second} : EdgeKey{second, first};
  };
  const auto sorted_face = [](FaceKey face) {
    std::sort(face.begin(), face.end());
    return face;
  };
  // Hard size fields are stored in regular-tetrahedron circumradius units.
  const double radius_to_edge = 4.0 / std::sqrt(6.0);
  const auto boundary_target = [&](const EdgeKey& edge, double& mean_size,
                                   double& edge_target) {
    const auto first = sizes.find(edge[0]);
    const auto second = sizes.find(edge[1]);
    if (first == sizes.end() || second == sizes.end() ||
        first->second <= 0.0 || second->second <= 0.0) {
      return false;
    }
    const double scale = std::max(first->second, second->second);
    mean_size =
        scale * 0.5 * (first->second / scale + second->second / scale);
    edge_target = mean_size * radius_to_edge;
    return edge_target > 0.0 && !std::isnan(edge_target);
  };
  // Local incidence and constrained-bank row indices: a boundary split or a
  // cone insertion touches only the cells, facets and segments around it.
  detail::LocalIncidence incidence;
  if (!incidence.initialize(mesh, initial)) return PHX_MC_INTERNAL_ERROR;
  initial = TetMeshSnapshot{};
  std::map<FaceKey, std::size_t> facet_rows_by_face;
  for (std::size_t row = 0; row < mesh.facets_.size(); ++row) {
    if (!facet_rows_by_face.emplace(sorted_face(mesh.facets_[row].vertices), row).second) {
      return PHX_MC_INTERNAL_ERROR;
    }
  }
  std::map<EdgeKey, std::size_t> segment_rows_by_edge;
  for (std::size_t row = 0; row < mesh.segments_.size(); ++row) {
    const TetSnapshotSegment& segment = mesh.segments_[row];
    if (!segment_rows_by_edge
             .emplace(sorted_edge(segment.vertices[0], segment.vertices[1]), row)
             .second) {
      return PHX_MC_INTERNAL_ERROR;
    }
  }
  // TetMesh::insert_local_boundary_point leaves the banks out of ID order.
  const auto restore_bank_order = [&]() noexcept {
    std::sort(mesh.facets_.begin(), mesh.facets_.end(),
              [](const TetSnapshotFacet& a, const TetSnapshotFacet& b) { return a.id < b.id; });
    std::sort(mesh.segments_.begin(), mesh.segments_.end(),
              [](const TetSnapshotSegment& a, const TetSnapshotSegment& b) {
                return a.id < b.id;
              });
  };
  const ScopeExit bank_order_guard{restore_bank_order};
  std::unordered_map<std::int32_t, std::vector<std::int32_t>> source_face_rows;
  for (std::size_t row = 0; row < mesh.source_face_groups_.size(); ++row) {
    source_face_rows[mesh.source_face_groups_[row]].push_back(static_cast<std::int32_t>(row));
  }
  std::unordered_map<std::int32_t, std::vector<std::int32_t>> source_segment_rows;
  for (std::size_t row = 0; row < mesh.source_segment_groups_.size(); ++row) {
    source_segment_rows[mesh.source_segment_groups_[row]].push_back(
        static_cast<std::int32_t>(row));
  }

  // Source witness of a boundary split point. When one source row of the
  // split segment's group (else of the split facet's group) carries both edge
  // endpoints, the point is the mean of their exact source parameters on it:
  // on a segment the mean rounded to binary64, which stays strictly between
  // the endpoint parameters, in a face the exact barycentric mean, so the
  // point lies exactly on the edge. An edge whose endpoints lie on different
  // rows of the group (it crosses a row border) is split at the midpoint of
  // its carriers, witnessed on the row that holds it (locate_in_rows). The
  // exact source point is authoritative; the carrier is its correctly
  // rounded realization.
  std::vector<std::int32_t> through_rows;
  const auto make_boundary_split = [&](std::int32_t first_slot, std::int32_t second_slot,
                                       std::span<const std::size_t> facet_rows,
                                       const TetSnapshotSegment* segment,
                                       TetEditVertex& vertex) {
    const double* corners[3]{};
    double midpoint[3];
    for (int axis = 0; axis < 3; ++axis) {
      midpoint[axis] = 0.5 * mesh.point(first_slot)[axis] + 0.5 * mesh.point(second_slot)[axis];
    }
    const auto witness_midpoint = [&](SourceStratum stratum, std::span<const std::int32_t> rows) {
      const std::int32_t ends[2]{first_slot, second_slot};
      rows_through(mesh, stratum, rows, ends, through_rows);
      std::int32_t row = -1;
      double parameters[2]{};
      return locate_in_rows(mesh, stratum, through_rows, midpoint, row, parameters) &&
             mesh.source_corners(stratum, row, corners) &&
             install_witness(corners, stratum, row, parameters, vertex) &&
             !same_carrier(vertex.carrier.data(), mesh.point(first_slot)) &&
             !same_carrier(vertex.carrier.data(), mesh.point(second_slot));
    };
    if (segment != nullptr) {
      const auto rows = source_segment_rows.find(segment->source);
      if (segment->protected_segment || rows == source_segment_rows.end()) return false;
      for (std::int32_t row : rows->second) {
        double first = 0.0;
        double second = 0.0;
        if (!mesh.source_corners(SourceStratum::kSegment, row, corners) ||
            !segment_parameter(mesh, first_slot, row, corners, first) ||
            !segment_parameter(mesh, second_slot, row, corners, second)) {
          continue;
        }
        const double parameters[2]{0.5 * first + 0.5 * second, 0.0};
        if (std::min(first, second) < parameters[0] && parameters[0] < std::max(first, second) &&
            install_witness(corners, SourceStratum::kSegment, row, parameters, vertex)) {
          return true;
        }
        break;
      }
      return witness_midpoint(SourceStratum::kSegment, rows->second);
    }
    std::set<std::int32_t> groups;
    for (std::size_t row : facet_rows) {
      if (mesh.facets_[row].source >= 0) groups.insert(mesh.facets_[row].source);
    }
    for (std::int32_t group : groups) {
      const auto rows = source_face_rows.find(group);
      if (rows == source_face_rows.end()) continue;
      for (std::int32_t row : rows->second) {
        double first[2]{};
        double second[2]{};
        double parameters[2]{};
        if (mesh.source_corners(SourceStratum::kFacet, row, corners) &&
            facet_weights(mesh, first_slot, row, corners, first) &&
            facet_weights(mesh, second_slot, row, corners, second)) {
          if (exact_half_sum(first[0], second[0], parameters[0]) &&
              exact_half_sum(first[1], second[1], parameters[1]) &&
              install_witness(corners, SourceStratum::kFacet, row, parameters, vertex)) {
            return true;
          }
          break;
        }
      }
    }
    if (groups.size() != 1) return false;
    const auto rows = source_face_rows.find(*groups.begin());
    return rows != source_face_rows.end() && witness_midpoint(SourceStratum::kFacet, rows->second);
  };

  // Interior Delaunay refinement. A violating cell (radius-edge ratio above the
  // bound, or circumradius above its size target) is removed by inserting its
  // circumcenter c: walk from the cell to c, grow the Bowyer-Watson cavity of
  // cells whose circumspheres strictly contain c without crossing constrained
  // faces, shrink it until it is star-shaped from c and keeps every vertex and
  // constrained segment on its boundary, then cone it to c through the
  // cavity-local TetMesh::insert_local_cone commit.
  // Encroachment follows Shewchuk's order: when c lies beyond a constrained
  // face, or inside the diametral ball of a constrained segment or the
  // equatorial ball of a constrained face bounding its cavity, a boundary
  // that may be split is split instead of inserting c: the encroached
  // subsegment at its midpoint, else the encroached subfacet at its
  // circumcenter (a subsegment that circumcenter encroaches, or that the walk
  // to it crosses, is split instead), each at an exact source parameter. A
  // fixed boundary keeps its faces: a c inside the domain is still inserted
  // (the cavity stops at the fixed faces and the cone is proved valid), and a
  // c beyond a fixed face leaves the cell unmet with reason FIXED_BOUNDARY.
  // Small input angles: Shewchuk's guarantee needs B > 2 and input angles of
  // at least 90 degrees; next to an acute input feature (acute_input_features)
  // boundary splits encroach each other ever closer to its apex. Following
  // Shewchuk's Terminator and TetGen's protecting balls, a boundary split
  // point inside the protected region of an acute feature (the ball of radius
  // lambda(apex) / sin(angle) around its apex, the capsule of that radius
  // around a ridge; protected_radius) is not inserted; splits of the size
  // pass are not encroachment chains and stay under the guard alone. A
  // boundary whose split is protected or refused is treated as fixed for
  // that cell: an in-domain c is inserted under the spacing guard; a cell
  // whose c lies beyond a protected boundary, or whose c the guard then
  // refuses, is left unmet with reason INPUT_ANGLE and the feature's apex and
  // angle as evidence.
  // Termination: a point x is inserted only at distance at least
  // g(x) = c * min(lfs(x), h(x)) from every vertex present, with c = 1/2 and
  // h(x) the circumradius size target interpolated at x (the target of the
  // cell or edge being split; g = c * lfs where no target applies). The local
  // feature size is measured on the constrained complex entering this pass:
  // lambda(v), for a vertex v present, is its distance to the nearest feature
  // not incident to v (another vertex, a constrained segment or constrained
  // facet edge not ending at v, or a constrained facet not containing v), and
  // lfs(x) = min_u (lambda(u) + |x - u|) over the natural neighbors u of x
  // (the vertices of x's insertion cavity, or the endpoints of a split edge).
  // An inserted vertex keeps lfs(x) as its own lambda. Next to a thin fixed
  // face lambda is that face's short edge or altitude, growing with the
  // distance to it, rather than the shortest edge anywhere in the mesh.
  // Packing: refinement never removes a vertex. By the triangle inequality
  // every stored lambda is at least lambda_min, the smallest lambda entering
  // the pass, which is positive: lambda(v) is floored at the coordinate
  // spacing at v, below which no representable point separates two features.
  // So any two vertices of which at least one was inserted are at least
  // g_min = c * min(lambda_min, h_min) > 0 apart, h_min the smallest positive
  // target. Balls of radius g_min / 2 around them are disjoint and lie in the
  // g_min / 2-neighborhood of the bounded domain, so at most
  // vol(domain + B(g_min / 2)) / vol(B(g_min / 2)) points are inserted. The
  // count is local as well: inserted points whose guards are at least g are
  // pairwise g apart, so a thin feature confines dense insertion to the region
  // where its lambda, grown by the distance to it, stays small. The fraction c
  // leaves room below the feature scale for the grading Delaunay refinement
  // needs next to a feature (its theory bounds insertion radii only by
  // lfs / C for a constant C > 1).
  // For a Delaunay cavity the circumcenter's nearest vertex lies at the
  // circumradius, so the guard refuses a cell only when its circumradius is
  // below g(x); in a cavity of a recovered, non-Delaunay neighborhood the
  // nearest vertex can be much closer. Such a cell is reported unmet with
  // reason REFUSED, never chased below the feature scale.
  // The protection rule keeps the packing bound (it only replaces a boundary
  // split by a guarded insertion of c, or by leaving the cell); it stops the
  // encroachment chain at an acute feature from running down to g_min, so
  // cells there end with INPUT_ANGLE evidence, not a cascade of refusals.
  std::unordered_map<std::uint64_t, UnmetCause> unmet_causes;
  // Delaunay refinement assumes a constrained Delaunay triangulation: a cell
  // whose circumsphere holds a vertex is not removed by inserting its
  // circumcenter, which then lies near that vertex. The input (a recovered
  // PLC) and every insertion are flipped toward it (flip_to_constrained_delaunay).
  const std::function<bool(std::uint64_t, std::uint64_t)> is_segment =
      [&](std::uint64_t first, std::uint64_t second) {
        return segment_rows_by_edge.contains(sorted_edge(first, second));
      };
  std::vector<std::uint64_t> flipped;
  {
    std::vector<std::uint64_t> all_cells;
    for (std::size_t row = 0; row < mesh.complex_.tets.size(); ++row) {
      const std::int32_t cell = static_cast<std::int32_t>(row);
      if (mesh.complex_.live(cell) && !is_ghost(mesh.tet(cell))) {
        all_cells.push_back(mesh.cell_id(cell));
      }
    }
    const std::int32_t status =
        flip_to_constrained_delaunay(mesh, incidence, all_cells, is_segment, flipped);
    if (status != PHX_MC_OK) return status;
  }
  constexpr double kSizeGuardFraction = 0.5;
  std::vector<double> feature_sizes;
  double smallest_feature = 0.0;
  if (!local_feature_sizes(mesh, mesh.vertex_ids_.size(), mesh.facets_, mesh.segments_,
                           feature_sizes, smallest_feature)) {
    return PHX_MC_INVALID_INPUT;
  }
  const auto feature_size_at = [&](const double* point, std::span<const std::int32_t> neighbors) {
    double size = std::numeric_limits<double>::infinity();
    for (std::int32_t vertex : neighbors) {
      size = std::min(size, feature_sizes[static_cast<std::size_t>(vertex)] +
                                std::sqrt(geometry::squared_distance(point, mesh.point(vertex))));
    }
    return size;
  };
  const auto spacing_guard = [&](double feature_size, double target_size) {
    const double guard = kSizeGuardFraction * feature_size;
    return target_size > 0.0 ? std::min(guard, kSizeGuardFraction * target_size) : guard;
  };
  // An inserted point x of natural radius r (the circumradius of the cell
  // whose circumcenter it is, half the split edge, the split subfacet's
  // circumradius) is equidistant from two to four vertices, which are
  // disjoint features, so lfs(x) <= r while the Lipschitz bound
  // lambda(u) + |x - u| exceeds r. The guard uses min(lfs(x), max(r,
  // lfs(x) / kGrading)): never above r, so a point r away from the vertices
  // it is placed between is admitted, and never below lfs(x) / kGrading,
  // which keeps the packing bound (Shewchuk's insertion radii are likewise
  // bounded below by lfs / C).
  constexpr double kGrading = 4.0;
  const auto insertion_guard = [&](double feature_size, double radius, double target_size) {
    return spacing_guard(std::min(feature_size, std::max(radius, feature_size / kGrading)),
                         target_size);
  };
  const auto record_feature_size = [&](std::int32_t vertex, double size) {
    if (feature_sizes.size() < mesh.vertex_ids_.size()) {
      feature_sizes.resize(mesh.vertex_ids_.size(), std::numeric_limits<double>::infinity());
    }
    feature_sizes[static_cast<std::size_t>(vertex)] = size;
  };
  SpacingIndex spacing(mesh, smallest_feature);
  for (std::size_t vertex = 0; vertex < mesh.vertex_ids_.size(); ++vertex) {
    if (mesh.vertex_live_[vertex] != 0 && !spacing.insert(static_cast<std::int32_t>(vertex))) {
      return PHX_MC_UNRESOLVED;
    }
  }
  std::vector<AcuteFeature> acute_features;
  if (!acute_input_features(mesh, incidence, mesh.facets_, mesh.segments_, feature_sizes,
                            acute_features)) {
    return PHX_MC_INVALID_INPUT;
  }
  ProtectionIndex protection(mesh, acute_features);
  if (!protection.build()) return PHX_MC_INVALID_INPUT;
  const auto input_angle = [&](std::int32_t feature) {
    const AcuteFeature& acute = acute_features[static_cast<std::size_t>(feature)];
    return UnmetCause{PHX_MC_TET_UNMET_INPUT_ANGLE, mesh.vertex_id(acute.apex), acute.angle};
  };

  // Largest circumradius first; cells are immutable, so a key stays valid
  // until its cell dies.
  std::set<std::pair<double, std::uint64_t>> queue;
  const auto snapshot_cell = [&](std::int32_t slot, TetSnapshotCell& cell) {
    cell.id = mesh.cell_id(slot);
    cell.region = mesh.complex_.region(slot);
    for (int corner = 0; corner < 4; ++corner) {
      cell.vertices[static_cast<std::size_t>(corner)] = mesh.vertex_id(mesh.tet(slot).v[corner]);
    }
  };
  const auto enqueue = [&](std::uint64_t id) {
    const std::int32_t slot = mesh.cell_slot(id);
    if (slot < 0 || !mesh.complex_.live(slot) || is_ghost(mesh.tet(slot))) return;
    TetSnapshotCell cell;
    snapshot_cell(slot, cell);
    if (!violates(mesh, cell, radius_edge_bound, sizes)) return;
    std::array<std::int32_t, 4> slots{};
    geometry::TetrahedronShape shape;
    const double radius =
        tetrahedron_geometry(mesh, cell, slots, shape) ? shape.circumradius : 0.0;
    queue.emplace(-radius, id);
  };
  // Flips the cells an insertion created toward the constrained Delaunay
  // triangulation, then enqueues them with the cells the flips created.
  const auto settle = [&](std::span<const std::uint64_t> created) -> std::int32_t {
    flipped.clear();
    const std::int32_t status =
        flip_to_constrained_delaunay(mesh, incidence, created, is_segment, flipped);
    for (std::uint64_t id : created) enqueue(id);
    for (std::uint64_t id : flipped) enqueue(id);
    return status;
  };
  for (std::size_t row = 0; row < mesh.complex_.tets.size(); ++row) {
    const std::int32_t cell = static_cast<std::int32_t>(row);
    if (mesh.complex_.live(cell) && !is_ghost(mesh.tet(cell))) enqueue(mesh.cell_id(cell));
  }

  const bool boundary_splittable = mesh.policy_ != BoundaryPolicy::kFixed;
  const auto edit_error = [](TetEditStatus status) -> std::int32_t {
    if (status == TetEditStatus::kAllocationFailure) return PHX_MC_ALLOCATION_FAILURE;
    if (status == TetEditStatus::kIndexOverflow) return PHX_MC_INDEX_RANGE_ERROR;
    if (status == TetEditStatus::kStaleGeneration) return PHX_MC_INTERNAL_ERROR;
    return PHX_MC_OK;
  };
  // Equatorial ball of a triangle: the smallest sphere through its corners.
  const auto circumcircle = [](const double* const corners[3], double* center) {
    double ab[3], ac[3];
    for (int axis = 0; axis < 3; ++axis) {
      ab[axis] = corners[1][axis] - corners[0][axis];
      ac[axis] = corners[2][axis] - corners[0][axis];
    }
    const double normal[3]{ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2],
                           ab[0] * ac[1] - ab[1] * ac[0]};
    const double normal_squared =
        normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
    if (!(normal_squared > 0.0) || !std::isfinite(normal_squared)) return -1.0;
    const double ab_squared = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    const double ac_squared = ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2];
    double radius_squared = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      const int next = (axis + 1) % 3;
      const int last = (axis + 2) % 3;
      const double offset = (ab_squared * (ac[next] * normal[last] - ac[last] * normal[next]) +
                             ac_squared * (normal[next] * ab[last] - normal[last] * ab[next])) /
                            (2.0 * normal_squared);
      radius_squared += offset * offset;
      center[axis] = corners[0][axis] + offset;
    }
    return radius_squared;
  };
  const auto in_equatorial_ball = [&](const double* const corners[3], const double* point) {
    double center[3];
    const double radius_squared = circumcircle(corners, center);
    return radius_squared > 0.0 && geometry::squared_distance(point, center) < radius_squared;
  };
  // Diametral ball of a segment.
  const auto in_diametral_ball = [](const double* a, const double* b, const double* point) {
    double dot = 0.0;
    for (int axis = 0; axis < 3; ++axis) dot += (point[axis] - a[axis]) * (point[axis] - b[axis]);
    return dot < 0.0;
  };

  // The constrained faces of `source` through the edge (first, second) other
  // than the one opposite `third`, each as (finite cell, face slot).
  const auto subfacets_across = [&](std::int32_t first, std::int32_t second, std::int32_t third,
                                    std::int32_t source,
                                    std::vector<std::pair<std::int32_t, int>>& found) {
    found.clear();
    for (std::int32_t cell : incidence.edge_star(first, second)) {
      const Tetrahedron& tetrahedron = mesh.tet(cell);
      for (int face = 0; face < 4; ++face) {
        const std::int32_t opposite = tetrahedron.v[face];
        if (opposite == first || opposite == second ||
            mesh.complex_.constraint(cell, face) != source) {
          continue;
        }
        std::int32_t corners[3];
        oriented_facet(tetrahedron.v, face, corners);
        if (std::find(corners, corners + 3, third) != corners + 3) continue;
        const FaceKey key = sorted_face({mesh.vertex_id(corners[0]), mesh.vertex_id(corners[1]),
                                         mesh.vertex_id(corners[2])});
        if (std::none_of(found.begin(), found.end(), [&](const std::pair<std::int32_t, int>& side) {
              std::int32_t other[3];
              oriented_facet(mesh.tet(side.first).v, side.second, other);
              return sorted_face({mesh.vertex_id(other[0]), mesh.vertex_id(other[1]),
                                  mesh.vertex_id(other[2])}) == key;
            })) {
          found.emplace_back(cell, face);
        }
      }
    }
  };
  const auto face_key_of = [&](std::int32_t cell, int face) {
    std::int32_t corners[3];
    oriented_facet(mesh.tet(cell).v, face, corners);
    return sorted_face(
        {mesh.vertex_id(corners[0]), mesh.vertex_id(corners[1]), mesh.vertex_id(corners[2])});
  };
  const auto finite_side = [&](std::int32_t cell, int face, std::pair<std::int32_t, int>& side) {
    if (!is_ghost(mesh.tet(cell))) {
      side = {cell, face};
      return true;
    }
    const std::int32_t other = mesh.tet(cell).n[face];
    if (!mesh.complex_.live(other) || is_ghost(mesh.tet(other))) return false;
    const int back = neighbor_slot(mesh.tet(other), cell);
    side = {other, back};
    return back >= 0;
  };

  // Inserts the boundary vertex `inserted` (witnessed on a facet, or on the
  // constrained segment `segment_edge`) with the Bowyer-Watson cavity of the
  // disks grown from `seeds` (finite sides of the constrained faces holding
  // it) across the non-segment edges of their facets into the subfacets whose
  // circumcircles contain it, else of the seeds alone, under the insertion
  // guard for the natural radius `radius` (none when 0) and size target
  // `target` (none when 0). A guarded vertex inside the protected region of
  // an acute input feature is not inserted unless `sizing_pass` (the size
  // field, not an encroachment chain, drives the split; the target and the
  // guard bound it); `protecting` is set to that feature. A facet vertex
  // inside the diametral ball of a subsegment of its cavity is not inserted;
  // `encroached_segment` is set to that subsegment. Fills `fan_rows` with the
  // facet rows it wrote. Returns OK and sets `split` when the vertex was
  // inserted.
  std::int32_t protecting = -1;
  bool sizing_pass = false;
  TetInsertionCavity cavity;
  std::vector<std::uint64_t> created_cells;
  std::vector<std::int32_t> cavity_vertices;
  std::vector<std::pair<std::int32_t, int>> across;
  std::vector<std::pair<std::int32_t, int>> disk_sides;
  std::vector<std::size_t> disk_rows;
  std::vector<std::size_t> split_segment_rows;
  std::vector<std::size_t> fan_rows;
  std::vector<std::int32_t> boundary_cavity;
  std::vector<FaceKey> removed_keys;
  std::optional<EdgeKey> encroached_segment;
  const auto insert_boundary_vertex =
      [&](TetEditVertex& inserted, std::span<const std::pair<std::int32_t, int>> seeds,
          const EdgeKey* segment_edge, double radius, double target,
          bool& split) -> std::int32_t {
    split = false;
    protecting = -1;
    encroached_segment.reset();
    fan_rows.clear();
    if (mesh.next_vertex_id_ == std::numeric_limits<std::uint64_t>::max() ||
        inserted_vertices == std::numeric_limits<std::int64_t>::max()) {
      return PHX_MC_INDEX_RANGE_ERROR;
    }
    inserted.id = mesh.next_vertex_id_;
    const double* point = inserted.carrier.data();
    if (!std::all_of(inserted.carrier.begin(), inserted.carrier.end(),
                     [](double value) { return coordinate_in_domain(value); })) {
      return PHX_MC_OK;
    }
    if (radius > 0.0 && !sizing_pass) {
      protecting = protection.find(point);
      if (protecting >= 0) return PHX_MC_OK;
    }
    split_segment_rows.clear();
    if (segment_edge != nullptr) {
      const auto row = segment_rows_by_edge.find(*segment_edge);
      if (row == segment_rows_by_edge.end()) return PHX_MC_INTERNAL_ERROR;
      split_segment_rows.push_back(row->second);
    }
    for (const bool whole_disk : {true, false}) {
      disk_sides.assign(seeds.begin(), seeds.end());
      for (std::size_t index = 0; whole_disk && index < disk_sides.size(); ++index) {
        const auto [side_cell, side_face] = disk_sides[index];
        const std::int32_t source = mesh.complex_.constraint(side_cell, side_face);
        std::int32_t corners[3];
        oriented_facet(mesh.tet(side_cell).v, side_face, corners);
        for (int corner = 0; corner < 3; ++corner) {
          const std::int32_t u = corners[corner];
          const std::int32_t v = corners[(corner + 1) % 3];
          if (segment_rows_by_edge.contains(sorted_edge(mesh.vertex_id(u), mesh.vertex_id(v)))) {
            continue;
          }
          subfacets_across(u, v, corners[(corner + 2) % 3], source, across);
          for (const auto& side : across) {
            const FaceKey key = face_key_of(side.first, side.second);
            if (std::any_of(disk_sides.begin(), disk_sides.end(), [&](const auto& member) {
                  return face_key_of(member.first, member.second) == key;
                })) {
              continue;
            }
            std::int32_t other[3];
            oriented_facet(mesh.tet(side.first).v, side.second, other);
            const double* points[3]{mesh.point(other[0]), mesh.point(other[1]),
                                    mesh.point(other[2])};
            if (in_equatorial_ball(points, point)) disk_sides.push_back(side);
          }
        }
      }
      disk_rows.clear();
      removed_keys.clear();
      for (const auto& [side_cell, side_face] : disk_sides) {
        const auto row = facet_rows_by_face.find(face_key_of(side_cell, side_face));
        if (row == facet_rows_by_face.end()) return PHX_MC_INTERNAL_ERROR;
        disk_rows.push_back(row->second);
        removed_keys.push_back(row->first);
      }
      const std::array<std::int32_t, 2> split_slots =
          segment_edge == nullptr
              ? std::array<std::int32_t, 2>{-1, -1}
              : std::array<std::int32_t, 2>{mesh.vertex_slot((*segment_edge)[0]),
                                            mesh.vertex_slot((*segment_edge)[1])};
      if (mesh.boundary_insertion_cavity(inserted, disk_sides, split_slots, boundary_cavity) !=
          TetCavityStatus::kFound) {
        continue;
      }
      cavity_vertices.clear();
      for (std::int32_t cavity_cell : boundary_cavity) {
        const Tetrahedron& tetrahedron = mesh.tet(cavity_cell);
        for (int corner = 0; corner < 4; ++corner) {
          const std::int32_t vertex = tetrahedron.v[corner];
          if (vertex < 0) continue;
          cavity_vertices.push_back(vertex);
          if (segment_edge != nullptr || encroached_segment.has_value()) continue;
          for (int other = corner + 1; other < 4; ++other) {
            const std::int32_t end = tetrahedron.v[other];
            if (end < 0) continue;
            const EdgeKey edge = sorted_edge(mesh.vertex_id(vertex), mesh.vertex_id(end));
            if (segment_rows_by_edge.contains(edge) &&
                in_diametral_ball(mesh.point(vertex), mesh.point(end), point)) {
              encroached_segment = edge;
              break;
            }
          }
        }
      }
      // Shewchuk: a subfacet circumcenter that encroaches a subsegment is not
      // inserted; the subsegment is split instead (split_encroached_face).
      if (encroached_segment.has_value()) {
        return PHX_MC_OK;
      }
      const double feature_size = feature_size_at(point, cavity_vertices);
      if (radius > 0.0 && !spacing.clear(point, insertion_guard(feature_size, radius, target))) {
        return PHX_MC_OK;
      }
      const std::size_t facet_count = mesh.facets_.size();
      const TetEditStatus status = incidence.insert_boundary_point(
          mesh, inserted, boundary_cavity, disk_rows, split_segment_rows, created_cells);
      const std::int32_t error = edit_error(status);
      if (error != PHX_MC_OK) return error;
      if (status != TetEditStatus::kApplied) continue;
      for (const FaceKey& key : removed_keys) facet_rows_by_face.erase(key);
      fan_rows.assign(disk_rows.begin(), disk_rows.end());
      for (std::size_t row = facet_count; row < mesh.facets_.size(); ++row) fan_rows.push_back(row);
      for (std::size_t row : fan_rows) {
        if (!facet_rows_by_face.emplace(sorted_face(mesh.facets_[row].vertices), row).second) {
          return PHX_MC_INTERNAL_ERROR;
        }
      }
      if (segment_edge != nullptr) {
        const std::size_t row = split_segment_rows.front();
        const std::size_t appended = mesh.segments_.size() - 1;
        segment_rows_by_edge.erase(*segment_edge);
        for (const std::size_t half : {row, appended}) {
          const auto& vertices = mesh.segments_[half].vertices;
          if (!segment_rows_by_edge.emplace(sorted_edge(vertices[0], vertices[1]), half).second) {
            return PHX_MC_INTERNAL_ERROR;
          }
        }
      }
      if (target > 0.0) sizes.emplace(inserted.id, target);
      const std::int32_t inserted_slot = mesh.vertex_slot(inserted.id);
      record_feature_size(inserted_slot, feature_size);
      if (!spacing.insert(inserted_slot)) return PHX_MC_UNRESOLVED;
      if (const std::int32_t settled = settle(created_cells); settled != PHX_MC_OK) return settled;
      ++inserted_vertices;
      split = true;
      return PHX_MC_OK;
    }
    return PHX_MC_OK;
  };

  // Splits the constrained edge `edge` at its exactly witnessed mean point
  // (make_boundary_split), its natural radius half the edge when `guarded`.
  std::vector<std::size_t> edge_facet_rows;
  std::vector<std::pair<std::int32_t, int>> edge_sides;
  const auto split_boundary_edge = [&](const EdgeKey& edge, double target, bool guarded,
                                       bool& split) -> std::int32_t {
    split = false;
    const std::int32_t first = mesh.vertex_slot(edge[0]);
    const std::int32_t second = mesh.vertex_slot(edge[1]);
    if (!mesh.live_vertex(first) || !mesh.live_vertex(second)) return PHX_MC_INTERNAL_ERROR;
    edge_sides.clear();
    edge_facet_rows.clear();
    for (std::int32_t cell : incidence.edge_star(first, second)) {
      const Tetrahedron& tetrahedron = mesh.tet(cell);
      for (int face = 0; face < 4; ++face) {
        if (tetrahedron.v[face] == first || tetrahedron.v[face] == second ||
            mesh.complex_.constraint(cell, face) == kNoConstraint) {
          continue;
        }
        const auto row = facet_rows_by_face.find(face_key_of(cell, face));
        if (row == facet_rows_by_face.end()) return PHX_MC_INTERNAL_ERROR;
        if (std::find(edge_facet_rows.begin(), edge_facet_rows.end(), row->second) ==
            edge_facet_rows.end()) {
          edge_facet_rows.push_back(row->second);
          edge_sides.emplace_back(cell, face);
        }
      }
    }
    if (edge_sides.empty()) return PHX_MC_OK;
    const auto segment = segment_rows_by_edge.find(edge);
    const bool on_segment = segment != segment_rows_by_edge.end();
    TetEditVertex inserted;
    if (!make_boundary_split(first, second, edge_facet_rows,
                             on_segment ? &mesh.segments_[segment->second] : nullptr, inserted)) {
      return PHX_MC_OK;
    }
    const double radius =
        guarded ? 0.5 * std::sqrt(geometry::squared_distance(mesh.point(first), mesh.point(second)))
                : 0.0;
    return insert_boundary_vertex(inserted, edge_sides, on_segment ? &edge : nullptr, radius,
                                  target, split);
  };

  // Shewchuk's encroached subfacet split. A constrained segment edge of the
  // subfacet whose diametral ball holds `encroacher` is split instead;
  // otherwise the subfacet's circumcenter is walked to across the subfacets
  // of its facet (a segment crossed on the way is encroached by it and split
  // instead), witnessed on the source face row of the facet's group that
  // holds it (locate_in_rows: oblique projection along the row's dominant
  // normal axis, binary64 barycentric parameters), whichever row that is, and
  // inserted (insert_boundary_vertex). Returns OK and sets `split` when a
  // vertex was inserted.
  const auto split_encroached_face = [&](std::int32_t cell, int face, const double* encroacher,
                                         bool& split) -> std::int32_t {
    split = false;
    std::pair<std::int32_t, int> current;
    if (!finite_side(cell, face, current)) return PHX_MC_OK;
    const std::int32_t source = mesh.complex_.constraint(current.first, current.second);
    std::int32_t facet[3];
    oriented_facet(mesh.tet(current.first).v, current.second, facet);
    const auto split_segment = [&](std::int32_t first, std::int32_t second) {
      const EdgeKey edge = sorted_edge(mesh.vertex_id(first), mesh.vertex_id(second));
      double mean_size = 0.0;
      double edge_target = 0.0;
      const bool has_size = boundary_target(edge, mean_size, edge_target);
      return split_boundary_edge(edge, has_size ? mean_size : 0.0, true, split);
    };
    for (int corner = 0; corner < 3; ++corner) {
      const std::int32_t first = facet[corner];
      const std::int32_t second = facet[(corner + 1) % 3];
      if (segment_rows_by_edge.contains(sorted_edge(mesh.vertex_id(first), mesh.vertex_id(second))) &&
          in_diametral_ball(mesh.point(first), mesh.point(second), encroacher)) {
        return split_segment(first, second);
      }
    }
    std::array<double, 3> center{};
    {
      const double* corners[3]{mesh.point(facet[0]), mesh.point(facet[1]), mesh.point(facet[2])};
      if (!(circumcircle(corners, center.data()) > 0.0)) return PHX_MC_OK;
    }
    const double radius = std::sqrt(geometry::squared_distance(center.data(), mesh.point(facet[0])));
    // Walk across the facet's subfacets to the one containing the center.
    for (std::size_t step = 0;; ++step) {
      if (step > mesh.facets_.size()) return PHX_MC_OK;
      oriented_facet(mesh.tet(current.first).v, current.second, facet);
      const double* corners[3]{mesh.point(facet[0]), mesh.point(facet[1]), mesh.point(facet[2])};
      double ab[3], ac[3];
      for (int k = 0; k < 3; ++k) {
        ab[k] = corners[1][k] - corners[0][k];
        ac[k] = corners[2][k] - corners[0][k];
      }
      const double normal[3]{ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2],
                             ab[0] * ac[1] - ab[1] * ac[0]};
      int axis = 0;
      for (int k = 1; k < 3; ++k) {
        if (std::abs(normal[k]) > std::abs(normal[axis])) axis = k;
      }
      const int x = (axis + 1) % 3;
      const int y = (axis + 2) % 3;
      const double target_point[2]{center[static_cast<std::size_t>(x)],
                                   center[static_cast<std::size_t>(y)]};
      int beyond = -1;
      for (int corner = 0; corner < 3 && beyond < 0; ++corner) {
        const double* u3 = corners[corner];
        const double* v3 = corners[(corner + 1) % 3];
        const double* w3 = corners[(corner + 2) % 3];
        const double u[2]{u3[x], u3[y]};
        const double v[2]{v3[x], v3[y]};
        const double w[2]{w3[x], w3[y]};
        const int inside = orient2d(u, v, w);
        const int side = orient2d(u, v, target_point);
        if (inside != 0 && side != 0 && side != inside) beyond = corner;
      }
      if (beyond < 0) break;
      const std::int32_t first = facet[beyond];
      const std::int32_t second = facet[(beyond + 1) % 3];
      if (segment_rows_by_edge.contains(sorted_edge(mesh.vertex_id(first), mesh.vertex_id(second)))) {
        return split_segment(first, second);
      }
      subfacets_across(first, second, facet[(beyond + 2) % 3], source, across);
      if (across.size() != 1) return PHX_MC_OK;
      current = across.front();
    }
    oriented_facet(mesh.tet(current.first).v, current.second, facet);
    const auto rows = source_face_rows.find(source);
    if (rows == source_face_rows.end()) return PHX_MC_OK;
    TetEditVertex inserted;
    {
      std::int32_t row = -1;
      double parameters[2]{};
      const double* corners[3]{};
      rows_through(mesh, SourceStratum::kFacet, rows->second, facet, through_rows);
      if (!locate_in_rows(mesh, SourceStratum::kFacet, through_rows, center.data(), row,
                          parameters) ||
          !mesh.source_corners(SourceStratum::kFacet, row, corners) ||
          !install_witness(corners, SourceStratum::kFacet, row, parameters, inserted)) {
        return PHX_MC_OK;
      }
    }
    const std::array<std::uint64_t, 4> corner_ids{mesh.vertex_id(facet[0]), mesh.vertex_id(facet[1]),
                                                  mesh.vertex_id(facet[2]), mesh.vertex_id(facet[2])};
    double point_size = 0.0;
    const bool has_size = mean_target_size(corner_ids, sizes, point_size);
    const std::pair<std::int32_t, int> seed[1]{current};
    const std::int32_t status = insert_boundary_vertex(
        inserted, seed, nullptr, radius, has_size ? point_size : 0.0, split);
    if (status != PHX_MC_OK || split || !encroached_segment.has_value()) return status;
    const EdgeKey edge = *encroached_segment;
    return split_segment(mesh.vertex_slot(edge[0]), mesh.vertex_slot(edge[1]));
  };

  // Boundary edges longer than their size targets are split first, longest
  // relative to its target first (ties: smallest key). Splitting never moves
  // a vertex, so a queued ratio stays exact until its edge is split; split
  // edges never reappear. A facet edge whose midpoint would encroach a
  // subsegment splits that subsegment first (under the insertion guard) and
  // is queued again: an unchecked midpoint next to a segment starts a chain
  // of ever thinner subfacets converging onto it.
  sizing_pass = true;
  std::set<EdgeKey> blocked_boundary_edges;
  if (!sizes.empty() && boundary_splittable) {
    std::set<std::pair<double, EdgeKey>> boundary_queue;
    const auto queue_boundary_edge = [&](std::uint64_t first_id, std::uint64_t second_id) {
      const EdgeKey edge = sorted_edge(first_id, second_id);
      double mean_size = 0.0;
      double edge_target = 0.0;
      if (!boundary_target(edge, mean_size, edge_target)) return true;
      const std::int32_t first_slot = mesh.vertex_slot(edge[0]);
      const std::int32_t second_slot = mesh.vertex_slot(edge[1]);
      if (!mesh.live_vertex(first_slot) || !mesh.live_vertex(second_slot)) return false;
      const double ratio =
          std::sqrt(geometry::squared_distance(mesh.point(first_slot), mesh.point(second_slot))) /
          edge_target;
      if (ratio > 1.0) boundary_queue.emplace(-ratio, edge);
      return true;
    };
    const auto queue_facet_edges = [&](const TetSnapshotFacet& facet) {
      return queue_boundary_edge(facet.vertices[0], facet.vertices[1]) &&
             queue_boundary_edge(facet.vertices[1], facet.vertices[2]) &&
             queue_boundary_edge(facet.vertices[2], facet.vertices[0]);
    };
    for (const TetSnapshotFacet& facet : mesh.facets_) {
      if (!queue_facet_edges(facet)) return PHX_MC_INTERNAL_ERROR;
    }
    while (!boundary_queue.empty()) {
      const EdgeKey selected_edge = boundary_queue.begin()->second;
      boundary_queue.erase(boundary_queue.begin());
      if (blocked_boundary_edges.contains(selected_edge)) continue;
      double selected_size = 0.0;
      double edge_target = 0.0;
      if (!boundary_target(selected_edge, selected_size, edge_target)) {
        return PHX_MC_INTERNAL_ERROR;
      }
      bool split = false;
      std::int32_t error = split_boundary_edge(selected_edge, selected_size, false, split);
      if (error != PHX_MC_OK) return error;
      if (!split && encroached_segment.has_value()) {
        // The midpoint encroaches a subsegment: split the subsegment first.
        const EdgeKey segment_edge = *encroached_segment;
        double segment_size = 0.0;
        if (boundary_target(segment_edge, segment_size, edge_target)) {
          error = split_boundary_edge(segment_edge, segment_size, true, split);
          if (error != PHX_MC_OK) return error;
        }
        if (split && !queue_boundary_edge(selected_edge[0], selected_edge[1])) {
          return PHX_MC_INTERNAL_ERROR;
        }
      }
      if (!split) {
        blocked_boundary_edges.insert(selected_edge);
        continue;
      }
      for (std::size_t row : fan_rows) {
        if (!queue_facet_edges(mesh.facets_[row])) return PHX_MC_INTERNAL_ERROR;
      }
    }
  }
  sizing_pass = false;

  while (!queue.empty()) {
    const std::uint64_t cell_id = queue.begin()->second;
    queue.erase(queue.begin());
    const std::int32_t root_slot = mesh.cell_slot(cell_id);
    if (root_slot < 0 || !mesh.complex_.live(root_slot)) continue;
    TetSnapshotCell parent;
    snapshot_cell(root_slot, parent);
    if (!violates(mesh, parent, radius_edge_bound, sizes)) continue;
    std::array<std::int32_t, 4> parent_slots{};
    geometry::TetrahedronShape shape;
    if (!tetrahedron_geometry(mesh, parent, parent_slots, shape)) {
      unmet_causes[cell_id] = {PHX_MC_TET_UNMET_NONREPRESENTABLE};
      continue;
    }
    const double* center = shape.circumcenter.data();
    if (!std::all_of(shape.circumcenter.begin(), shape.circumcenter.end(),
                     [](double value) { return coordinate_in_domain(value); })) {
      unmet_causes[cell_id] = {PHX_MC_TET_UNMET_NONREPRESENTABLE};
      continue;
    }

    const TetCavityStatus located = mesh.insertion_cavity(root_slot, center, true, cavity);
    if (located == TetCavityStatus::kRefused) {
      unmet_causes[cell_id] = {PHX_MC_TET_UNMET_REFUSED};
      continue;
    }
    if (located == TetCavityStatus::kBlocked) {
      bool split = false;
      protecting = -1;
      if (boundary_splittable) {
        const std::int32_t error =
            split_encroached_face(cavity.blocked_cell, cavity.blocked_face, center, split);
        if (error != PHX_MC_OK) return error;
      }
      if (split) {
        enqueue(cell_id);
      } else if (protecting >= 0) {
        unmet_causes[cell_id] = input_angle(protecting);
      } else {
        unmet_causes[cell_id] = {boundary_splittable ? PHX_MC_TET_UNMET_REFUSED
                                                     : PHX_MC_TET_UNMET_FIXED_BOUNDARY};
      }
      continue;
    }
    // Shewchuk: an encroached subsegment, then an encroached subfacet, of the
    // cavity's constrained boundary is split before the center is inserted;
    // a protected or refused split leaves that boundary fixed for this center.
    std::int32_t protected_by = -1;
    if (boundary_splittable) {
      std::int32_t encroached_cell = -1;
      int encroached_face = -1;
      bool segment_encroached = false;
      for (std::int32_t cell : cavity.cells) {
        for (int face = 0; face < 4 && !segment_encroached; ++face) {
          if (cavity.members.contains(mesh.tet(cell).n[face]) ||
              mesh.complex_.constraint(cell, face) == kNoConstraint) {
            continue;
          }
          std::int32_t facet[3];
          oriented_facet(mesh.tet(cell).v, face, facet);
          for (int corner = 0; corner < 3 && !segment_encroached; ++corner) {
            const std::int32_t a = facet[corner];
            const std::int32_t b = facet[(corner + 1) % 3];
            segment_encroached =
                segment_rows_by_edge.contains(sorted_edge(mesh.vertex_id(a), mesh.vertex_id(b))) &&
                in_diametral_ball(mesh.point(a), mesh.point(b), center);
          }
          const double* corners[3]{mesh.point(facet[0]), mesh.point(facet[1]),
                                   mesh.point(facet[2])};
          if (segment_encroached || (encroached_cell < 0 && in_equatorial_ball(corners, center))) {
            encroached_cell = cell;
            encroached_face = face;
          }
        }
        if (segment_encroached) break;
      }
      if (encroached_cell >= 0) {
        bool split = false;
        protecting = -1;
        const std::int32_t error =
            split_encroached_face(encroached_cell, encroached_face, center, split);
        if (error != PHX_MC_OK) return error;
        if (split) {
          enqueue(cell_id);
          continue;
        }
        protected_by = protecting;
      }
    }
    const UnmetCause refused =
        protected_by >= 0 ? input_angle(protected_by) : UnmetCause{PHX_MC_TET_UNMET_REFUSED};

    double center_size = 0.0;
    const bool has_center_size = mean_target_size(parent.vertices, sizes, center_size);
    cavity_vertices.clear();
    for (std::int32_t cell : cavity.cells) {
      for (std::int32_t vertex : mesh.tet(cell).v) {
        if (vertex >= 0) cavity_vertices.push_back(vertex);
      }
    }
    const double center_feature_size = feature_size_at(center, cavity_vertices);
    if (!spacing.clear(center, insertion_guard(center_feature_size, shape.circumradius,
                                               has_center_size ? center_size : 0.0))) {
      unmet_causes[cell_id] = refused;
      continue;
    }

    TetEditVertex inserted;
    inserted.id = mesh.next_vertex_id_;
    std::copy(shape.circumcenter.begin(), shape.circumcenter.end(), inserted.carrier.begin());
    const TetEditStatus status =
        incidence.insert_cone(mesh, inserted, cavity.cells, created_cells);
    const std::int32_t error = edit_error(status);
    if (error != PHX_MC_OK) return error;
    if (status != TetEditStatus::kApplied) {
      unmet_causes[cell_id] = refused;
      continue;
    }
    if (has_center_size) sizes.emplace(inserted.id, center_size);
    const std::int32_t inserted_slot = mesh.vertex_slot(inserted.id);
    record_feature_size(inserted_slot, center_feature_size);
    if (!spacing.insert(inserted_slot)) return PHX_MC_UNRESOLVED;
    ++inserted_vertices;
    if (const std::int32_t settled = settle(created_cells); settled != PHX_MC_OK) return settled;
  }

  restore_bank_order();
  TetMeshSnapshot final_state;
  if (mesh.snapshot(final_state) != TetEditStatus::kApplied) return PHX_MC_INTERNAL_ERROR;
  std::vector<TetUnmetRecord> records;
  collect_unmet(mesh, final_state, radius_edge_bound, sizes, unmet_causes, records,
                unmet_cells);
  std::map<std::uint64_t, double> boundary_unmet;
  for (const EdgeKey& edge : blocked_boundary_edges) {
    const std::int32_t first_slot = mesh.vertex_slot(edge[0]);
    const std::int32_t second_slot = mesh.vertex_slot(edge[1]);
    if (!mesh.live_vertex(first_slot) || !mesh.live_vertex(second_slot)) {
      return PHX_MC_INTERNAL_ERROR;
    }
    const std::span<const std::int32_t> star = incidence.edge_star(first_slot, second_slot);
    const bool remains_constrained =
        std::any_of(star.begin(), star.end(), [&](std::int32_t cell) {
          for (int face = 0; face < 4; ++face) {
            const std::int32_t opposite = mesh.tet(cell).v[face];
            if (opposite != first_slot && opposite != second_slot &&
                mesh.complex_.constraint(cell, face) != kNoConstraint) {
              return true;
            }
          }
          return false;
        });
    if (!remains_constrained) continue;
    double mean_size = 0.0;
    double edge_target = 0.0;
    if (!boundary_target(edge, mean_size, edge_target)) continue;
    const double* first = mesh.point(first_slot);
    const double* second = mesh.point(second_slot);
    const double length = std::hypot(
        std::hypot(second[0] - first[0], second[1] - first[1]),
        second[2] - first[2]);
    const double ratio = length / edge_target;
    if (!(ratio > 1.0)) continue;
    for (std::int32_t cell : star) {
      auto& value = boundary_unmet[mesh.cell_id(cell)];
      value = std::max(value, ratio);
    }
  }
  for (const auto& [cell_id, ratio] : boundary_unmet) {
    const auto existing = std::find_if(
        records.begin(), records.end(), [&](const TetUnmetRecord& record) {
          return record.cell_id == cell_id &&
                 record.criterion == PHX_MC_TET_UNMET_SIZE;
        });
    if (existing == records.end()) {
      records.push_back({cell_id, PHX_MC_TET_UNMET_SIZE,
                         PHX_MC_TET_UNMET_NO_IMPROVEMENT, ratio});
    } else {
      existing->value = std::max(existing->value, ratio);
    }
  }
  if (!boundary_unmet.empty()) {
    std::set<std::uint64_t> unmet_cell_ids;
    for (const TetUnmetRecord& record : records) {
      unmet_cell_ids.insert(record.cell_id);
    }
    unmet_cells = static_cast<std::int64_t>(unmet_cell_ids.size());
  }
  mesh.set_unmet_records(records);
  return unmet_cells == 0 ? PHX_MC_OK : PHX_MC_UNRESOLVED;
}

}  // namespace phx::mc
