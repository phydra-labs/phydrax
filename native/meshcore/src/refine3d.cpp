//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Source-aware Delaunay refinement by checked circumcenter cavity insertion.
// Any remaining criterion violation is reported per cell as UNRESOLVED.
#include "refine3d.hpp"
#include "improve3d.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cavity.hpp"
#include "angle.hpp"
#include "geometry3.hpp"
#include "local_incidence.hpp"
#include "predicates.hpp"
#include "phydrax_meshcore.h"
#include "exact_scalar.hpp"

namespace phx::mc {
namespace {

template <class Exit>
struct ScopeExit {
  Exit exit;
  ~ScopeExit() { exit(); }
};
template <class Exit>
ScopeExit(Exit) -> ScopeExit<Exit>;

std::array<double, 3> metric_point(const TetMesh& mesh, const double* p) {
  return mesh.metric_transform({p[0], p[1], p[2]});
}

double metric_distance(const TetMesh& mesh, const double* a, const double* b) {
  if (!mesh.metric_active()) return std::sqrt(geometry::squared_distance(a, b));
  const auto d = mesh.metric_transform({a[0] - b[0], a[1] - b[1], a[2] - b[2]});
  return geometry::length(d);
}

double metric_segment_distance(const TetMesh& mesh, const double* p, const double* a,
                               const double* b) {
  if (!mesh.metric_active()) return geometry::point_segment_distance(p, a, b);
  const auto wp = mesh.metric_transform({p[0] - a[0], p[1] - a[1], p[2] - a[2]});
  const std::array<double, 3> wa{};
  const auto wb = mesh.metric_transform({b[0] - a[0], b[1] - a[1], b[2] - a[2]});
  return geometry::point_segment_distance(wp.data(), wa.data(), wb.data());
}

double metric_triangle_distance(const TetMesh& mesh, const double* p, const double* a,
                                const double* b, const double* c) {
  if (!mesh.metric_active()) return geometry::point_triangle_distance(p, a, b, c);
  const auto wp = mesh.metric_transform({p[0] - a[0], p[1] - a[1], p[2] - a[2]});
  const std::array<double, 3> wa{};
  const auto wb = mesh.metric_transform({b[0] - a[0], b[1] - a[1], b[2] - a[2]});
  const auto wc = mesh.metric_transform({c[0] - a[0], c[1] - a[1], c[2] - a[2]});
  return geometry::point_triangle_distance(wp.data(), wa.data(), wb.data(), wc.data());
}

// The (metric) circumsphere and radius-edge ratio of a live cell.
bool cell_shape(const TetMesh& mesh, const Tetrahedron& tet, geometry::TetrahedronShape& shape) {
  const double* points[4];
  for (int corner = 0; corner < 4; ++corner) {
    points[corner] = mesh.point(tet.v[corner]);
    if (points[corner] == nullptr) return false;
  }
  if (!mesh.metric_active()) return geometry::tetrahedron_shape(points, shape);
  std::array<std::array<double, 3>, 4> world{};
  const double* transformed[4];
  for (int corner = 0; corner < 4; ++corner) {
    world[corner] = mesh.metric_transform(
        {points[corner][0] - points[0][0], points[corner][1] - points[0][1],
         points[corner][2] - points[0][2]});
    transformed[corner] = world[corner].data();
  }
  if (!geometry::tetrahedron_shape(transformed, shape)) return false;
  const auto center = mesh.metric_inverse_transform(shape.circumcenter);
  for (int axis = 0; axis < 3; ++axis) shape.circumcenter[axis] = points[0][axis] + center[axis];
  return true;
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
bool install_witness(const TetMesh& mesh, const double* const* corners, SourceStratum stratum,
                     std::int32_t row, const double* parameters, TetEditVertex& vertex) {
  if (!source_parameters_valid(stratum, parameters)) return false;
  double deviation = 0.0;
  if (!mesh.source_carrier_metric(corners, stratum, parameters, vertex.carrier.data(), deviation)) {
    return false;
  }
  vertex.witness.stratum = stratum;
  vertex.witness.entity = row;
  vertex.witness.parameters[0] = parameters[0];
  vertex.witness.parameters[1] = parameters[1];
  vertex.witness.deviation = deviation;
  return true;
}

// Installs the plane witness of source face row `row` with `parameters` on
// `vertex`: the carrier rounds the exact plane point, the deviation is its
// exact realization bound.
bool install_plane_parameters(const TetMesh& mesh, std::int32_t row, const double* parameters,
                              TetEditVertex& vertex) {
  double deviation = 0.0;
  if (!mesh.plane_carrier(row, parameters, vertex.carrier.data(), deviation)) return false;
  vertex.witness.stratum = SourceStratum::kPlane;
  vertex.witness.entity = row;
  vertex.witness.parameters[0] = parameters[0];
  vertex.witness.parameters[1] = parameters[1];
  vertex.witness.deviation = deviation;
  return true;
}

// Installs the plane witness of the declared plane of source face row `row`
// at the binary64 parameters of the (approximately computed) orthogonal
// projection of `point`: the exact plane point they address is the vertex,
// wherever it lands near `point`.
bool install_plane_witness(const TetMesh& mesh, std::int32_t row, const double* point,
                           TetEditVertex& vertex) {
  const double* plane = mesh.face_plane(row);
  if (plane == nullptr) return false;
  const double* u = plane + 3;
  const double* v = plane + 6;
  const double r[3]{point[0] - plane[0], point[1] - plane[1], point[2] - plane[2]};
  const auto dot = [](const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  };
  const double uu = dot(u, u);
  const double uv = dot(u, v);
  const double vv = dot(v, v);
  const double determinant = uu * vv - uv * uv;
  const double parameters[2]{(dot(u, r) * vv - dot(v, r) * uv) / determinant,
                             (dot(v, r) * uu - dot(u, r) * uv) / determinant};
  return install_plane_parameters(mesh, row, parameters, vertex);
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

// The circumradius size target of a cell from its corners' retained targets;
// false when a corner is unconstrained.
bool cell_target(const TetMesh& mesh, const std::int32_t* corners, double& target) {
  std::array<double, 4> values{};
  for (int corner = 0; corner < 4; ++corner) {
    values[static_cast<std::size_t>(corner)] = mesh.vertex_size(corners[corner]);
    if (!(values[static_cast<std::size_t>(corner)] > 0.0)) return false;
  }
  target = cell_size_target(values);
  return std::isfinite(target);
}

// How the cell (o, p, q, r) (identity metric) stands against the refinement
// criteria: circumradius R at most `bound` times its shortest edge (`bound` 0:
// none) and at most `target` (infinite: none). R^2 = |N|^2 / (4 D^2) for the
// edge vectors a, b, c from o, D = a . (b x c) and
// N = |a|^2 (b x c) + |b|^2 (c x a) + |c|^2 (a x b); the circumcenter is
// o + N / (2 D). The screen decides only cells with D^2 above
// 1e-8 |a|^2 |b|^2 |c|^2, where every binary64 term is within a relative 1e-9
// of its value, and only beyond a relative margin of 1e-6 from a limit; the
// rest are uncertain and cell_shape, which stays authoritative, decides them.
enum class CellScreen : std::uint8_t { kMeets, kViolates, kUncertain };
CellScreen screen_cell(const double* o, const double* p, const double* q, const double* r,
                       double bound, double target, double* center = nullptr) noexcept {
  const double a[3]{p[0] - o[0], p[1] - o[1], p[2] - o[2]};
  const double b[3]{q[0] - o[0], q[1] - o[1], q[2] - o[2]};
  const double c[3]{r[0] - o[0], r[1] - o[1], r[2] - o[2]};
  const double bc[3]{b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0]};
  const double ca[3]{c[1] * a[2] - c[2] * a[1], c[2] * a[0] - c[0] * a[2], c[0] * a[1] - c[1] * a[0]};
  const double ab[3]{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
  const double aa = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
  const double bb = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
  const double cc = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
  const double det = a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2];
  const double det_squared = det * det;
  if (!(det_squared > 1e-8 * (aa * bb * cc)) || !std::isfinite(det_squared)) {
    return CellScreen::kUncertain;
  }
  double components[3];
  double numerator = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    components[axis] = aa * bc[axis] + bb * ca[axis] + cc * ab[axis];
    numerator += components[axis] * components[axis];
  }
  if (!std::isfinite(numerator)) return CellScreen::kUncertain;
  constexpr double kMargin = (1.0 - 1e-6) * (1.0 - 1e-6);
  // R^2 against limit^2 as |N|^2 against 4 D^2 limit^2.
  const double scaled = 4.0 * det_squared;
  bool meets = true;
  bool violates = false;
  if (std::isfinite(target)) {
    const double limit = scaled * (target * target);
    meets = numerator <= limit * kMargin;
    violates = numerator * kMargin > limit;
  }
  if (bound > 0.0 && !violates) {
    const double d12 = (q[0] - p[0]) * (q[0] - p[0]) + (q[1] - p[1]) * (q[1] - p[1]) +
                       (q[2] - p[2]) * (q[2] - p[2]);
    const double d13 = (r[0] - p[0]) * (r[0] - p[0]) + (r[1] - p[1]) * (r[1] - p[1]) +
                       (r[2] - p[2]) * (r[2] - p[2]);
    const double d23 = (r[0] - q[0]) * (r[0] - q[0]) + (r[1] - q[1]) * (r[1] - q[1]) +
                       (r[2] - q[2]) * (r[2] - q[2]);
    const double limit = scaled * (bound * bound) * std::min({aa, bb, cc, d12, d13, d23});
    meets = meets && numerator <= limit * kMargin;
    violates = numerator * kMargin > limit;
  }
  if (violates) {
    if (center != nullptr) {
      for (int axis = 0; axis < 3; ++axis) center[axis] = o[axis] + components[axis] / (2.0 * det);
    }
    return CellScreen::kViolates;
  }
  return meets ? CellScreen::kMeets : CellScreen::kUncertain;
}

// Why refinement left a cell: an unmet reason and, for INPUT_ANGLE, the apex
// vertex ID and angle in degrees of the acute input feature that protected the
// boundary split.
struct UnmetCause {
  std::int32_t reason = PHX_MC_TET_UNMET_NO_IMPROVEMENT;
  std::uint64_t apex_vertex = kNoUnmetApex;
  double apex_angle = std::numeric_limits<double>::quiet_NaN();
};

// Unmet records of the live cells among `slots` (ascending: slot, i.e.
// snapshot, order); every live cell missing a criterion must be listed.
void collect_unmet(const TetMesh& mesh, double radius_edge_bound,
                   const std::unordered_map<std::uint64_t, UnmetCause>& causes,
                   std::span<const std::int32_t> slots,
                   std::vector<TetUnmetRecord>& records, std::int64_t& unmet_cells) {
  records.clear();
  unmet_cells = 0;
  for (std::int32_t slot : slots) {
    const Tetrahedron& tet = mesh.tet(slot);
    if (tet.v[0] == kDeadVertex || is_ghost(tet)) continue;
    const std::uint64_t id = mesh.cell_id(slot);
    geometry::TetrahedronShape shape;
    bool cell_unmet = false;
    if (!cell_shape(mesh, tet, shape)) {
      records.push_back({id, PHX_MC_TET_UNMET_VALIDITY, PHX_MC_TET_UNMET_NONREPRESENTABLE, 0.0});
      ++unmet_cells;
      continue;
    }
    const bool bad_ratio = radius_edge_bound > 0.0 && shape.radius_edge_ratio > radius_edge_bound;
    double target = 0.0;
    const bool oversized = cell_target(mesh, tet.v, target) && shape.circumradius > target;
    if (!bad_ratio && !oversized) continue;
    const auto found = causes.find(id);
    const UnmetCause cause = found == causes.end() ? UnmetCause{} : found->second;
    if (bad_ratio) {
      records.push_back({id, PHX_MC_TET_UNMET_RADIUS_EDGE, cause.reason,
                         shape.radius_edge_ratio, cause.apex_vertex, cause.apex_angle});
      cell_unmet = true;
    }
    if (oversized) {
      records.push_back({id, PHX_MC_TET_UNMET_SIZE, cause.reason,
                         shape.circumradius / target, cause.apex_vertex, cause.apex_angle});
      cell_unmet = true;
    }
    if (cell_unmet) ++unmet_cells;
  }
}

// A dense uniform grid of cubic cells over a box; coordinates outside it
// clamp to its boundary cells, so any box query reads a superset of the
// cells its part inside the grid meets.
struct DenseFrame {
  double origin[3]{};
  double cell = 0.0;
  std::size_t dims[3]{};

  // Cells of side at least `side` over [lower, upper], the side grown until
  // there are at most `cap` cells. False for a non-finite frame.
  bool fit(const double* lower, const double* upper, double side, double cap) {
    for (int attempt = 0; attempt < 64; ++attempt) {
      if (!(side > 0.0) || !std::isfinite(side)) return false;
      double count = 1.0;
      double spans[3];
      for (int axis = 0; axis < 3; ++axis) {
        spans[axis] = std::floor((upper[axis] - lower[axis]) / side) + 1.0;
        if (!(spans[axis] >= 1.0) || !std::isfinite(spans[axis])) return false;
        count *= spans[axis];
      }
      if (count <= cap) {
        for (int axis = 0; axis < 3; ++axis) {
          origin[axis] = lower[axis];
          dims[axis] = static_cast<std::size_t>(spans[axis]);
        }
        cell = side;
        return true;
      }
      side *= std::max(1.125, std::cbrt(count / cap));
    }
    return false;
  }
  std::size_t size() const noexcept { return dims[0] * dims[1] * dims[2]; }
  // The clamped cell coordinate of the finite or infinite `value` on `axis`.
  std::size_t coordinate(double value, std::size_t axis) const noexcept {
    const double scaled = std::floor((value - origin[axis]) / cell);
    if (!(scaled > 0.0)) return 0;
    const double top = static_cast<double>(dims[axis] - 1);
    return scaled < top ? static_cast<std::size_t>(scaled) : dims[axis] - 1;
  }
  std::size_t index(std::size_t x, std::size_t y, std::size_t z) const noexcept {
    return (z * dims[1] + y) * dims[0] + x;
  }
};

// Items of a DenseFrame grouped by cell in registration order (a counting
// sort); an item is registered in every cell of its cell range.
template <class Item>
class DenseBuckets {
 public:
  // `range(row, first, last)` sets the inclusive cell range of item `row` per
  // axis or returns false to skip it; `make(row)` returns its record.
  template <class Range, class Make>
  void build(const DenseFrame& frame, std::size_t item_count, Range&& range, Make&& make) {
    starts_.assign(frame.size() + 1, 0U);
    std::vector<std::size_t> cursor;
    for (int pass = 0; pass < 2; ++pass) {
      for (std::size_t row = 0; row < item_count; ++row) {
        std::size_t first[3];
        std::size_t last[3];
        if (!range(row, first, last)) continue;
        for (std::size_t z = first[2]; z <= last[2]; ++z) {
          for (std::size_t y = first[1]; y <= last[1]; ++y) {
            for (std::size_t x = first[0]; x <= last[0]; ++x) {
              const std::size_t at = frame.index(x, y, z);
              if (pass == 0) {
                ++starts_[at + 1];
              } else {
                items_[cursor[at]++] = make(row);
              }
            }
          }
        }
      }
      if (pass == 0) {
        for (std::size_t at = 1; at < starts_.size(); ++at) starts_[at] += starts_[at - 1];
        items_.resize(starts_.back());
        cursor.assign(starts_.begin(), starts_.end() - 1);
      }
    }
  }

  std::span<const Item> items(std::size_t at) const noexcept {
    return std::span<const Item>(items_).subspan(starts_[at], starts_[at + 1] - starts_[at]);
  }

 private:
  std::vector<std::size_t> starts_;
  std::vector<Item> items_;
};

// A point, line or plane feature of local_feature_sizes: its metric box, its
// corner slots (-1 past the last corner) and, for a feature registered in
// several cells, its row among those (else -1).
struct FeatureRecord {
  double lower[3];
  double upper[3];
  std::int32_t corners[3];
  std::int32_t row;
};

// The largest metric length of one binary64 coordinate step at a point.
class CoordinateResolution {
 public:
  explicit CoordinateResolution(const TetMesh& mesh) {
    // The step is a power of two, so the transform of the step along an
    // axis is the step times the transform of that axis.
    for (std::size_t axis = 0; axis < 3; ++axis) {
      std::array<double, 3> unit{};
      unit[axis] = 1.0;
      scales_[axis] = geometry::length(mesh.metric_transform(unit));
    }
  }
  double operator()(const double* p) const noexcept {
    double resolution = 0.0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const double magnitude = std::abs(p[axis]);
      // The next representable value above a nonnegative (or infinite or
      // NaN, whose step is NaN either way) magnitude.
      const double above =
          std::bit_cast<double>(std::bit_cast<std::uint64_t>(magnitude) + 1U);
      resolution = std::max(resolution, (above - magnitude) * scales_[axis]);
    }
    return resolution;
  }

 private:
  std::array<double, 3> scales_{};
};

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
  // The shortest incident edge bounds lambda from above and sets the query
  // radius. Squared lengths are compared: the square root (of metric_distance)
  // is correctly rounded and monotone, so it commutes with the minimum.
  const TetrahedralComplex& complex = mesh.complex();
  const bool metric = mesh.metric_active();
  for (std::size_t row = 0; row < complex.tets.size(); ++row) {
    const std::int32_t cell = static_cast<std::int32_t>(row);
    if (!complex.live(cell) || is_ghost(complex.tets[row])) continue;
    const Tetrahedron& tetrahedron = complex.tets[row];
    for (int first = 0; first < 4; ++first) {
      for (int second = first + 1; second < 4; ++second) {
        const std::int32_t a = tetrahedron.v[first];
        const std::int32_t b = tetrahedron.v[second];
        const double* p = mesh.point(a);
        const double* q = mesh.point(b);
        double squared = 0.0;
        if (metric) {
          const auto d = mesh.metric_transform({p[0] - q[0], p[1] - q[1], p[2] - q[2]});
          squared = geometry::dot(d.data(), d.data());
        } else {
          squared = geometry::squared_distance(p, q);
        }
        sizes[static_cast<std::size_t>(a)] = std::min(sizes[static_cast<std::size_t>(a)], squared);
        sizes[static_cast<std::size_t>(b)] = std::min(sizes[static_cast<std::size_t>(b)], squared);
      }
    }
  }
  for (double& size : sizes) size = std::sqrt(size);

  // Features by corner slots (-1 past the last corner): the live vertices,
  // the planes, the distinct lines.
  std::vector<std::array<std::int32_t, 3>> features;
  std::vector<std::uint64_t> lines;
  lines.reserve(segments.size() + 3 * facets.size());
  const auto line_key = [](std::int32_t a, std::int32_t b) {
    const auto low = static_cast<std::uint32_t>(std::min(a, b));
    const auto high = static_cast<std::uint32_t>(std::max(a, b));
    return (static_cast<std::uint64_t>(low) << 32) | high;
  };
  for (const TetSnapshotSegment& segment : segments) {
    const std::int32_t a = mesh.vertex_slot(segment.vertices[0]);
    const std::int32_t b = mesh.vertex_slot(segment.vertices[1]);
    if (!mesh.live_vertex(a) || !mesh.live_vertex(b)) return false;
    lines.push_back(line_key(a, b));
  }
  for (std::size_t slot = 0; slot < slot_count; ++slot) {
    if (mesh.live_vertex(static_cast<std::int32_t>(slot))) {
      features.push_back({static_cast<std::int32_t>(slot), -1, -1});
    }
  }
  const std::size_t plane_begin = features.size();
  for (const TetSnapshotFacet& facet : facets) {
    std::array<std::int32_t, 3> slots{};
    for (std::size_t corner = 0; corner < 3; ++corner) {
      slots[corner] = mesh.vertex_slot(facet.vertices[corner]);
      if (!mesh.live_vertex(slots[corner])) return false;
    }
    features.push_back(slots);
    for (std::size_t corner = 0; corner < 3; ++corner) {
      lines.push_back(line_key(slots[corner], slots[(corner + 1) % 3]));
    }
  }
  const std::size_t plane_end = features.size();
  std::sort(lines.begin(), lines.end());
  lines.erase(std::unique(lines.begin(), lines.end()), lines.end());
  for (std::uint64_t key : lines) {
    features.push_back({static_cast<std::int32_t>(key >> 32),
                        static_cast<std::int32_t>(key & 0xffffffffU), -1});
  }

  // Cells of 1.5 median shortest incident edges (measured best on the ledger
  // shapes: a query reads about 3 cells per axis).
  std::vector<double> radii;
  radii.reserve(slot_count);
  for (std::size_t slot = 0; slot < slot_count; ++slot) {
    if (std::isfinite(sizes[slot])) radii.push_back(sizes[slot]);
  }
  if (radii.empty()) return false;
  std::nth_element(radii.begin(), radii.begin() + static_cast<std::ptrdiff_t>(radii.size() / 2),
                   radii.end());
  double cell = 1.5 * radii[radii.size() / 2];
  // The edge of a plane opposite a corner is a line not incident to it,
  // measured first (as its line feature would be), so that the query starts
  // from a near feature.
  for (std::size_t row = plane_begin; row < plane_end; ++row) {
    const auto& corners = features[row];
    for (std::size_t corner = 0; corner < 3; ++corner) {
      const std::int32_t vertex = corners[corner];
      const std::int32_t a = corners[(corner + 1) % 3];
      const std::int32_t b = corners[(corner + 2) % 3];
      if (a == b || a == vertex || b == vertex) continue;
      double& size = sizes[static_cast<std::size_t>(vertex)];
      size = std::min(size, metric_segment_distance(mesh, mesh.point(vertex),
                                                    mesh.point(std::min(a, b)),
                                                    mesh.point(std::max(a, b))));
    }
  }
  // Metric images of the live vertices; the features' metric boxes over them.
  std::vector<std::array<double, 3>> world(slot_count);
  double frame_lower[3]{infinity, infinity, infinity};
  double frame_upper[3]{-infinity, -infinity, -infinity};
  for (std::size_t slot = 0; slot < slot_count; ++slot) {
    if (mesh.live_vertex(static_cast<std::int32_t>(slot))) {
      world[slot] = metric_point(mesh, mesh.point(static_cast<std::int32_t>(slot)));
      for (std::size_t axis = 0; axis < 3; ++axis) {
        frame_lower[axis] = std::min(frame_lower[axis], world[slot][axis]);
        frame_upper[axis] = std::max(frame_upper[axis], world[slot][axis]);
      }
    }
  }
  std::vector<FeatureRecord> records(features.size());
  double extent = 0.0;
  for (std::size_t row = 0; row < features.size(); ++row) {
    FeatureRecord& record = records[row];
    const auto& corners = features[row];
    const auto& first = world[static_cast<std::size_t>(corners[0])];
    std::copy(first.begin(), first.end(), record.lower);
    std::copy(first.begin(), first.end(), record.upper);
    for (std::size_t corner = 1; corner < 3 && corners[corner] >= 0; ++corner) {
      const auto& point = world[static_cast<std::size_t>(corners[corner])];
      for (std::size_t axis = 0; axis < 3; ++axis) {
        record.lower[axis] = std::min(record.lower[axis], point[axis]);
        record.upper[axis] = std::max(record.upper[axis], point[axis]);
      }
    }
    std::copy(corners.begin(), corners.end(), record.corners);
    if (row >= plane_begin && row < plane_end) {
      for (std::size_t axis = 0; axis < 3; ++axis) {
        extent = std::max(extent, record.upper[axis] - record.lower[axis]);
      }
    }
  }
  // No plane box spans more than 17 cells per axis.
  cell = std::max(cell, extent / 16.0);
  DenseFrame frame;
  if (!frame.fit(frame_lower, frame_upper, cell,
                 std::max(4096.0, 8.0 * static_cast<double>(slot_count)))) {
    return false;
  }
  // A small feature (box at most one cell wide per axis) is registered once,
  // in the cell of its box's lower corner, so a query of radius r reads the
  // cells of [x - r - cell, x + r]; a large one in every cell its box meets,
  // and a query reads it once (stamped with the query's vertex slot + 1).
  const auto small = [&](const FeatureRecord& record) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      if (!(record.upper[axis] - record.lower[axis] <= frame.cell)) return false;
    }
    return true;
  };
  std::int32_t large_count = 0;
  for (FeatureRecord& record : records) record.row = small(record) ? -1 : large_count++;
  DenseBuckets<FeatureRecord> small_grid;
  DenseBuckets<FeatureRecord> large_grid;
  small_grid.build(frame, records.size(), [&](std::size_t row, std::size_t* first,
                                              std::size_t* last) {
    const FeatureRecord& record = records[row];
    if (record.row >= 0) return false;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      first[axis] = last[axis] = frame.coordinate(record.lower[axis], axis);
    }
    return true;
  }, [&](std::size_t row) { return records[row]; });
  large_grid.build(frame, large_count > 0 ? records.size() : 0, [&](std::size_t row,
                                                                    std::size_t* first,
                                                                    std::size_t* last) {
    const FeatureRecord& record = records[row];
    if (record.row < 0) return false;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      first[axis] = frame.coordinate(record.lower[axis], axis);
      last[axis] = frame.coordinate(record.upper[axis], axis);
    }
    return true;
  }, [&](std::size_t row) { return records[row]; });
  std::vector<FeatureRecord>().swap(records);

  std::vector<std::uint32_t> seen(static_cast<std::size_t>(large_count), 0U);
  smallest = infinity;
  const CoordinateResolution coordinate_resolution(mesh);
  for (std::size_t slot = 0; slot < slot_count; ++slot) {
    const std::int32_t vertex = static_cast<std::int32_t>(slot);
    if (!mesh.live_vertex(vertex)) continue;
    const double* p = mesh.point(vertex);
    double lambda = sizes[slot];
    if (!std::isfinite(lambda)) return false;
    const auto& here = world[slot];
    // A feature whose box lies farther than lambda from v cannot lower
    // lambda: the box distance bounds the feature distance from below (a
    // relative slack of 1e-9 covers the rounding of either); the cell ranges
    // cover that slack.
    const auto measure = [&](const FeatureRecord& record) {
      double squared = 0.0;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        const double gap = std::max(
            {0.0, record.lower[axis] - here[axis], here[axis] - record.upper[axis]});
        squared += gap * gap;
      }
      if (squared > lambda * lambda * (1.0 + 1e-9)) return;
      const std::int32_t* corners = record.corners;
      if (corners[0] == vertex || corners[1] == vertex || corners[2] == vertex) return;
      const double distance =
          corners[1] < 0   ? metric_distance(mesh, p, mesh.point(corners[0]))
          : corners[2] < 0 ? metric_segment_distance(mesh, p, mesh.point(corners[0]),
                                                     mesh.point(corners[1]))
                           : metric_triangle_distance(mesh, p, mesh.point(corners[0]),
                                                      mesh.point(corners[1]),
                                                      mesh.point(corners[2]));
      lambda = std::min(lambda, distance);
    };
    const double reach = lambda * (1.0 + 1e-9);
    std::size_t first[3];
    std::size_t last[3];
    for (std::size_t axis = 0; axis < 3; ++axis) {
      first[axis] = frame.coordinate(here[axis] - reach - frame.cell, axis);
      last[axis] = frame.coordinate(here[axis] + reach, axis);
    }
    for (std::size_t z = first[2]; z <= last[2]; ++z) {
      for (std::size_t y = first[1]; y <= last[1]; ++y) {
        for (std::size_t x = first[0]; x <= last[0]; ++x) {
          for (const FeatureRecord& record : small_grid.items(frame.index(x, y, z))) {
            measure(record);
          }
        }
      }
    }
    if (large_count > 0) {
      const std::uint32_t stamp = static_cast<std::uint32_t>(slot) + 1U;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        first[axis] = frame.coordinate(here[axis] - reach, axis);
      }
      for (std::size_t z = first[2]; z <= last[2]; ++z) {
        for (std::size_t y = first[1]; y <= last[1]; ++y) {
          for (std::size_t x = first[0]; x <= last[0]; ++x) {
            for (const FeatureRecord& record : large_grid.items(frame.index(x, y, z))) {
              std::uint32_t& mark = seen[static_cast<std::size_t>(record.row)];
              if (mark == stamp) continue;
              mark = stamp;
              measure(record);
            }
          }
        }
      }
    }
    if (!std::isfinite(lambda)) return false;
    // Features nearer than the coordinate spacing at v cannot be separated by
    // a representable point; lambda is floored there, which keeps it positive.
    lambda = std::max(lambda, coordinate_resolution(p));
    sizes[slot] = lambda;
    smallest = std::min(smallest, lambda);
  }
  return std::isfinite(smallest);
}

// Bits per axis of the proposal order: a 4096^3 grid over the domain's box
// orders a round's proposals as finely as their cavities need (points in one
// grid cell keep their measurement order), at a little over half the work of
// the full 21 bits.
constexpr int kHilbertBits = 12;
constexpr std::uint32_t kHilbertMax = (1U << kHilbertBits) - 1U;

// Index of the 3D Hilbert curve through the grid point (x, y, z), by
// Skilling's transpose ("Programming the Hilbert curve", AIP Conf. Proc. 707,
// 2004): undo the excess work, Gray-encode, then interleave the bits. The
// per-bit choices are masks, not branches (the same bits).
std::uint64_t hilbert_key(std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
  std::uint32_t axes[3]{x, y, z};
  for (std::uint32_t q = 1U << (kHilbertBits - 1); q > 1; q >>= 1) {
    const std::uint32_t p = q - 1;
    for (int axis = 0; axis < 3; ++axis) {
      // Set bit: invert the low bits of axis 0; clear bit: exchange them.
      const std::uint32_t set = 0U - static_cast<std::uint32_t>((axes[axis] & q) != 0);
      const std::uint32_t swap = (axes[0] ^ axes[axis]) & p & ~set;
      axes[0] ^= (p & set) | swap;
      axes[axis] ^= swap;
    }
  }
  for (int axis = 1; axis < 3; ++axis) axes[axis] ^= axes[axis - 1];
  std::uint32_t flip = 0;
  for (std::uint32_t q = 1U << (kHilbertBits - 1); q > 1; q >>= 1) {
    flip ^= (q - 1) & (0U - static_cast<std::uint32_t>((axes[2] & q) != 0));
  }
  for (std::uint32_t& axis : axes) axis ^= flip;
  // Interleave: bit b of axis a goes to bit 3 b + (2 - a).
  const auto spread = [](std::uint64_t value) {
    value &= 0x1fffffULL;
    value = (value | value << 32) & 0x1f00000000ffffULL;
    value = (value | value << 16) & 0x1f0000ff0000ffULL;
    value = (value | value << 8) & 0x100f00f00f00f00fULL;
    value = (value | value << 4) & 0x10c30c30c30c30c3ULL;
    value = (value | value << 2) & 0x1249249249249249ULL;
    return value;
  };
  return spread(axes[0]) << 2 | spread(axes[1]) << 1 | spread(axes[2]);
}

// Hilbert keys of points in a box: each axis scaled by the largest extent.
struct HilbertFrame {
  double lower[3]{};
  double scale = 0.0;

  std::uint64_t key(const double* point) const noexcept {
    std::uint32_t grid[3];
    for (int axis = 0; axis < 3; ++axis) {
      const double scaled = (point[axis] - lower[axis]) * scale;
      grid[axis] = !(scaled > 0.0) ? 0U
                   : scaled >= static_cast<double>(kHilbertMax) ? kHilbertMax
                                                                 : static_cast<std::uint32_t>(scaled);
    }
    return hilbert_key(grid[0], grid[1], grid[2]);
  }
};

// A cell exactly positively oriented whose binary64 orientation determinant
// is positive as well (so it is measurable), from one floating evaluation:
// beyond the error bound its sign is both; within it, the exact predicate
// decides orientation and the floating value measurability.
bool measurably_positive(const double* a, const double* b, const double* c,
                         const double* d) noexcept {
  const OrientationStage stage = orientation_stage(a, b, c, d);
  if (-stage.det > stage.errbound) return true;
  if (!(-stage.det > 0.0)) return false;
  return orient3d(a, b, c, d) > 0;
}

// Advances a visit stamp over `marks`, clearing them when it wraps.
std::uint32_t next_stamp(std::vector<std::uint32_t>& marks, std::uint32_t& stamp) {
  if (++stamp == 0) {
    std::fill(marks.begin(), marks.end(), 0U);
    stamp = 1;
  }
  return stamp;
}

// The insertion cavity of an interior point over flat stamped scratch: the
// walk from `start` to the point (blocked at a constrained or hull face),
// the Bowyer-Watson cavity of the cells of the root's region whose
// circumspheres strictly contain it, grown across unconstrained faces and
// joined by the walk's cells, then shrunk (keeping the root's component)
// until every cone cell over its boundary is positive (exact orientation on
// carriers, and positive in binary64 so it is measurable), every vertex of a
// cavity cell lies on the boundary and no embedded segment is inside.
// It owns the visit stamps of the mesh's cells (Tetrahedron::stamp) while
// it lives, so a walk or cavity reads each cell's record and nothing else;
// a cell written by an edit has stamp 0, which no visit uses.
class InteriorCavity {
 public:
  enum class Status : std::uint8_t { kFound, kBlocked, kRefused };

  explicit InteriorCavity(TetrahedralComplex& complex) : complex_(complex) {}

  template <class IsEmbedded>
  Status build(const TetMesh& mesh, std::int32_t start, const double* point,
               bool has_embedded, IsEmbedded&& is_embedded) {
    std::vector<Tetrahedron>& tets = complex_.tets;
    // Every corner read below is a vertex of a live finite cell.
    const double* const carriers = mesh.carriers();
    const auto at = [carriers](std::int32_t vertex) {
      return carriers + 3 * static_cast<std::size_t>(vertex);
    };
    const bool metric = mesh.metric_active();
    cells.clear();
    boundary.clear();
    vertices.clear();
    root = -1;
    blocked_cell = -1;
    blocked_face = -1;
    const std::uint32_t walk = next_cell_stamp();
    walk_.clear();
    std::int32_t current = start;
    for (;;) {
      Tetrahedron& tet = tets[static_cast<std::size_t>(current)];
      if (tet.stamp == walk) return Status::kRefused;
      tet.stamp = walk;
      int exit = -1;
      for (int face = 0; face < 4 && exit < 0; ++face) {
        std::int32_t facet[3];
        oriented_facet(tet.v, face, facet);
        if (orient3d_filtered(at(facet[0]), at(facet[1]), at(facet[2]), point) < 0) {
          exit = face;
        }
      }
      if (exit < 0) break;
      const std::int32_t next = tet.n[exit];
      if (complex_.constraint(current, exit) != kNoConstraint || !complex_.live(next) ||
          is_ghost(tets[static_cast<std::size_t>(next)])) {
        blocked_cell = current;
        blocked_face = exit;
        return Status::kBlocked;
      }
      walk_.push_back(current);
      current = next;
    }
    root = current;
    const std::int32_t region = complex_.region(root);
    // Candidates found outside the circumsphere keep their own stamp, so a
    // cell next to several cavity cells is tested once.
    const std::uint32_t rejected = next_cell_stamp();
    std::uint32_t member = next_cell_stamp();
    cells.push_back(root);
    tets[static_cast<std::size_t>(root)].stamp = member;
    stack_.assign(1, root);
    for (std::int32_t cell : walk_) {
      Tetrahedron& tet = tets[static_cast<std::size_t>(cell)];
      if (tet.region != region || tet.stamp == member) continue;
      tet.stamp = member;
      cells.push_back(cell);
      stack_.push_back(cell);
    }
    while (!stack_.empty()) {
      const std::int32_t cell = stack_.back();
      stack_.pop_back();
      const Tetrahedron& tet = tets[static_cast<std::size_t>(cell)];
      for (int face = 0; face < 4; ++face) {
        const std::int32_t next = tet.n[face];
        if (((tet.constrained >> face) & 1U) != 0 || !complex_.live(next)) continue;
        Tetrahedron& candidate = tets[static_cast<std::size_t>(next)];
        if (candidate.stamp == member || candidate.stamp == rejected || is_ghost(candidate) ||
            candidate.region != region) {
          continue;
        }
        const double* corners[4]{at(candidate.v[0]), at(candidate.v[1]), at(candidate.v[2]),
                                 at(candidate.v[3])};
        const int side = metric ? mesh.metric_insphere(corners[0], corners[1], corners[2],
                                                       corners[3], point)
                                : insphere_filtered(corners[0], corners[1], corners[2], corners[3], point);
        if (side > 0) {
          candidate.stamp = member;
          cells.push_back(next);
          stack_.push_back(next);
          // Its neighbors are read next (the stack is last in, first out).
          for (std::int32_t beyond : candidate.n) {
            if (beyond >= 0) __builtin_prefetch(&tets[static_cast<std::size_t>(beyond)]);
          }
        } else {
          candidate.stamp = rejected;
        }
      }
    }

    // Shrinking: a cell with a boundary face whose cone is not positive stays
    // so as cells leave (its face stays on the boundary), as does a cell cut
    // off from the root, so every such cell is dropped at once and the result
    // is the one dropping them one at a time gives. A cavity cell with a
    // vertex off the boundary (or an embedded segment inside) can be cured by
    // a later drop, so those leave one at a time, the first in cavity order.
    for (;;) {
      boundary.clear();
      vertices.clear();
      dropped_.clear();
      if (marks_.size() < mesh.vertex_slot_count()) marks_.resize(mesh.vertex_slot_count(), 0U);
      const std::uint32_t mark = next_stamp(marks_, mark_stamp_);
      for (std::int32_t cell : cells) {
        const Tetrahedron& tet = tets[static_cast<std::size_t>(cell)];
        for (int face = 0; face < 4; ++face) {
          const std::int32_t outside = tet.n[face];
          const Tetrahedron& beyond = tets[static_cast<std::size_t>(outside)];
          if (beyond.stamp == member) continue;
          const double* cone[4];
          for (int corner = 0; corner < 4; ++corner) {
            cone[corner] = corner == face ? point : at(tet.v[corner]);
          }
          if (!measurably_positive(cone[0], cone[1], cone[2], cone[3])) {
            dropped_.push_back(cell);
            break;
          }
          boundary.push_back({cell, face, outside, neighbor_slot(beyond, cell)});
          for (int corner = 0; corner < 4; ++corner) {
            if (corner == face) continue;
            const std::int32_t vertex = tet.v[corner];
            if (marks_[static_cast<std::size_t>(vertex)] != mark) {
              marks_[static_cast<std::size_t>(vertex)] = mark;
              vertices.push_back(vertex);
            }
          }
        }
      }
      if (dropped_.empty()) {
        std::int32_t drop = -1;
        if (has_embedded) {
          kept_edges_.clear();
          for (const BoundaryFacet& face : boundary) {
            std::int32_t facet[3];
            oriented_facet(tets[static_cast<std::size_t>(face.tet)].v, face.slot, facet);
            for (int corner = 0; corner < 3; ++corner) {
              kept_edges_.push_back(slot_edge(facet[corner], facet[(corner + 1) % 3]));
            }
          }
          std::sort(kept_edges_.begin(), kept_edges_.end());
        }
        for (std::int32_t cell : cells) {
          if (cell == root) continue;
          const Tetrahedron& tet = tets[static_cast<std::size_t>(cell)];
          bool lost = false;
          for (std::int32_t vertex : tet.v) {
            lost = lost || marks_[static_cast<std::size_t>(vertex)] != mark;
          }
          for (int first = 0; has_embedded && !lost && first < 4; ++first) {
            for (int second = first + 1; !lost && second < 4; ++second) {
              lost = is_embedded(tet.v[first], tet.v[second]) &&
                     !std::binary_search(kept_edges_.begin(), kept_edges_.end(),
                                         slot_edge(tet.v[first], tet.v[second]));
            }
          }
          if (lost) {
            drop = cell;
            break;
          }
        }
        if (drop < 0) return Status::kFound;
        dropped_.push_back(drop);
      }
      if (dropped_.front() == root) {
        cells.clear();
        boundary.clear();
        vertices.clear();
        return Status::kRefused;
      }
      // Keep the root's component of the cavity without the dropped cells.
      for (std::int32_t cell : dropped_) tets[static_cast<std::size_t>(cell)].stamp = 0U;
      const std::uint32_t reached = next_cell_stamp();
      if (reached < member) {
        // The stamp wrapped and cleared every membership (once in 2^32
        // visits): refuse rather than rebuild.
        cells.clear();
        boundary.clear();
        vertices.clear();
        return Status::kRefused;
      }
      tets[static_cast<std::size_t>(root)].stamp = reached;
      stack_.assign(1, root);
      cells.assign(1, root);
      while (!stack_.empty()) {
        const std::int32_t cell = stack_.back();
        stack_.pop_back();
        for (std::int32_t next : tets[static_cast<std::size_t>(cell)].n) {
          if (next >= 0 && tets[static_cast<std::size_t>(next)].stamp == member) {
            tets[static_cast<std::size_t>(next)].stamp = reached;
            cells.push_back(next);
            stack_.push_back(next);
          }
        }
      }
      member = reached;
    }
  }

  std::vector<std::int32_t> cells;  // root first
  std::vector<BoundaryFacet> boundary;  // cavity cell, face, the cell beyond, its face
  std::vector<std::int32_t> vertices;  // distinct boundary vertices
  std::int32_t root = -1;
  std::int32_t blocked_cell = -1;
  int blocked_face = -1;

 private:
  static std::uint64_t slot_edge(std::int32_t a, std::int32_t b) noexcept {
    const std::uint32_t low = static_cast<std::uint32_t>(std::min(a, b));
    const std::uint32_t high = static_cast<std::uint32_t>(std::max(a, b));
    return (static_cast<std::uint64_t>(low) << 32) | high;
  }
  // The next cell visit stamp, clearing every cell's stamp when it wraps.
  std::uint32_t next_cell_stamp() {
    if (++cell_stamp_ == 0) {
      for (Tetrahedron& tet : complex_.tets) tet.stamp = 0U;
      cell_stamp_ = 1;
    }
    return cell_stamp_;
  }

  TetrahedralComplex& complex_;
  std::uint32_t cell_stamp_ = 0;
  std::vector<std::uint32_t> marks_;
  std::uint32_t mark_stamp_ = 0;
  std::vector<std::int32_t> walk_;
  std::vector<std::int32_t> stack_;
  std::vector<std::uint64_t> kept_edges_;
  std::vector<std::int32_t> dropped_;
};

// A point on the domain side of the constrained face `face` of the finite
// cell `cell` where the regular tetrahedron on the face's mean (metric) edge
// has its apex: the face centroid plus sqrt(2/3) mean edges along the unit
// normal toward the cell. False for a degenerate face.
bool face_apex(const TetMesh& mesh, std::int32_t cell, int face, double* apex) {
  const Tetrahedron& tet = mesh.tet(cell);
  std::int32_t facet[3];
  oriented_facet(tet.v, face, facet);
  const double* a = mesh.point(facet[0]);
  const double* b = mesh.point(facet[1]);
  const double* c = mesh.point(facet[2]);
  const double* inside = mesh.point(tet.v[face]);
  const auto u = mesh.metric_transform({b[0] - a[0], b[1] - a[1], b[2] - a[2]});
  const auto v = mesh.metric_transform({c[0] - a[0], c[1] - a[1], c[2] - a[2]});
  const auto w = mesh.metric_transform({inside[0] - a[0], inside[1] - a[1], inside[2] - a[2]});
  std::array<double, 3> normal{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                               u[0] * v[1] - u[1] * v[0]};
  const double norm = geometry::length(normal);
  if (!(norm > 0.0) || !std::isfinite(norm)) return false;
  const double side = normal[0] * w[0] + normal[1] * w[1] + normal[2] * w[2] < 0.0 ? -1.0 : 1.0;
  const std::array<double, 3> uv{v[0] - u[0], v[1] - u[1], v[2] - u[2]};
  const double mean_edge =
      (geometry::length(u) + geometry::length(v) + geometry::length(uv)) / 3.0;
  const double height = std::sqrt(2.0 / 3.0) * mean_edge * side / norm;
  for (double& value : normal) value *= height;
  const auto offset = mesh.metric_inverse_transform(normal);
  for (int axis = 0; axis < 3; ++axis) {
    apex[axis] = (a[axis] + b[axis] + c[axis]) / 3.0 + offset[static_cast<std::size_t>(axis)];
  }
  return std::isfinite(apex[0]) && std::isfinite(apex[1]) && std::isfinite(apex[2]);
}

// Input angles whose cosine is at most this are right or obtuse: a rounded
// realization of a right angle is not an acute feature.
constexpr double kRightAngleCosine = 0x1p-20;

double dot3(const double* u, const double* v) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; }

void cross3(const double* u, const double* v, double* out) {
  out[0] = u[1] * v[2] - u[2] * v[1];
  out[1] = u[2] * v[0] - u[0] * v[2];
  out[2] = u[0] * v[1] - u[1] * v[0];
}

// Screens the cone cells of a cavity's boundary faces to `point`: `meets`
// receives, per boundary row, whether the cone cell clearly meets the
// refinement criteria (screen_cell with `bound` and the size target of its
// corners' sizes, `point_size` the new vertex's, 0: none); no later round
// removes such a cell for itself, so it is persistent unless a neighbor's
// cavity takes it. Returns the smallest squared sine of a dihedral angle over
// the persistent cells (a dihedral near 0 or 180 degrees has a small sine):
// for the edge (i, j) of a cell with corners P0..P3, 6 V = D and face normals
// n_k opposite corner k, sin^2 theta_ij = D^2 |P_i - P_j|^2 / (|n_k|^2 |n_l|^2),
// {k, l} the other two corners. `toward` receives the unit normal of the
// worst cell's cavity face, pointing to `point`.
double screen_cone(const TetMesh& mesh, std::span<const BoundaryFacet> boundary,
                   const double* point, double bound, double point_size, double* toward,
                   std::vector<std::uint8_t>& meets) {
  meets.assign(boundary.size(), 0U);
  double worst = std::numeric_limits<double>::infinity();
  const double* const carriers = mesh.carriers();
  for (std::size_t row = 0; row < boundary.size(); ++row) {
    const std::int32_t cell = boundary[row].tet;
    const int face = boundary[row].slot;
    const Tetrahedron& tet = mesh.tet(cell);
    const double* corners[4];
    std::array<double, 4> sizes{};
    for (int corner = 0; corner < 4; ++corner) {
      corners[corner] =
          corner == face ? point : carriers + 3 * static_cast<std::size_t>(tet.v[corner]);
      sizes[static_cast<std::size_t>(corner)] =
          corner == face ? point_size : mesh.vertex_size(tet.v[corner]);
    }
    if (screen_cell(corners[0], corners[1], corners[2], corners[3], bound,
                    cell_size_target(sizes)) != CellScreen::kMeets) {
      continue;
    }
    meets[row] = 1U;
    double normal_squared[4];
    double normals[4][3];
    for (int opposite = 0; opposite < 4; ++opposite) {
      const double* m = corners[(opposite + 1) & 3];
      const double* l = corners[(opposite + 2) & 3];
      const double* o = corners[(opposite + 3) & 3];
      const double u[3]{l[0] - m[0], l[1] - m[1], l[2] - m[2]};
      const double v[3]{o[0] - m[0], o[1] - m[1], o[2] - m[2]};
      cross3(u, v, normals[opposite]);
      normal_squared[opposite] = dot3(normals[opposite], normals[opposite]);
    }
    const double edge[3]{corners[1][0] - corners[0][0], corners[1][1] - corners[0][1],
                         corners[1][2] - corners[0][2]};
    const double d = dot3(edge, normals[0]);
    double cell_worst = std::numeric_limits<double>::infinity();
    for (int first = 0; first < 4; ++first) {
      for (int second = first + 1; second < 4; ++second) {
        int others[2]{};
        for (int corner = 0, count = 0; corner < 4; ++corner) {
          if (corner != first && corner != second) others[count++] = corner;
        }
        const double e[3]{corners[first][0] - corners[second][0],
                          corners[first][1] - corners[second][1],
                          corners[first][2] - corners[second][2]};
        const double denominator = normal_squared[others[0]] * normal_squared[others[1]];
        const double sine = denominator > 0.0 ? d * d * dot3(e, e) / denominator : 0.0;
        cell_worst = std::min(cell_worst, sine);
      }
    }
    if (cell_worst < worst) {
      worst = cell_worst;
      const double* n = normals[face];
      const double length = std::sqrt(normal_squared[face]);
      const double* base = corners[(face + 1) & 3];
      const double height = (point[0] - base[0]) * n[0] + (point[1] - base[1]) * n[1] +
                            (point[2] - base[2]) * n[2];
      const double scale = length > 0.0 ? (height < 0.0 ? -1.0 : 1.0) / length : 0.0;
      for (int axis = 0; axis < 3; ++axis) toward[axis] = n[axis] * scale;
    }
  }
  return worst;
}

// Cosine of the angle between u and v; NaN when either is zero.
double cosine3(const double* u, const double* v) {
  return dot3(u, v) / (std::sqrt(dot3(u, u)) * std::sqrt(dot3(v, v)));
}

double degrees(double cosine) {
  return phx::mc::angle::arccosine(std::clamp(cosine, -1.0, 1.0)) *
         (180.0 / std::numbers::pi);
}

// The smallest angle in degrees between the ray along `ray` and the planar
// sector spanned by `first` and `second` at the same apex, measured in the
// sector's declared plane of normal `plane_normal` when it has one (null:
// the plane of the two sides).
double ray_sector_angle(const double* ray, const double* first, const double* second,
                        const double* plane_normal) {
  double normal[3];
  cross3(first, second, normal);
  if (plane_normal != nullptr) {
    const double sign = dot3(plane_normal, normal) < 0.0 ? -1.0 : 1.0;
    for (int axis = 0; axis < 3; ++axis) normal[axis] = sign * plane_normal[axis];
  }
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
  return feature_size / phx::mc::angle::sine(angle * (std::numbers::pi / 180.0));
}

// The acute features of the constrained complex (`facets`, `segments`) of
// `mesh`, apexes in slot order then ridges in slot-pair order; the radius is
// protected_radius of the local feature size `feature_sizes` at the apex (the
// smaller one of the ridge endpoints). Angles at a facet of a group with a
// declared plane are measured with the plane's normal (`plane_normals`, by
// group), not with its rounded subfacets. False on a dead constrained vertex.
bool acute_input_features(
    const TetMesh& mesh, const detail::LocalIncidence& incidence,
    std::span<const TetSnapshotFacet> facets, std::span<const TetSnapshotSegment> segments,
    std::span<const double> feature_sizes,
    const std::unordered_map<std::int32_t, std::array<double, 3>>& plane_normals,
    std::vector<AcuteFeature>& features) {
  const auto plane_normal = [&](std::int32_t group) -> const double* {
    const auto found = plane_normals.find(group);
    return found == plane_normals.end() ? nullptr : found->second.data();
  };
  features.clear();
  // Flat incidence rows grouped by a stable sort on their keys (each group in
  // input order): segment ends and facet corners by apex, facet sides by
  // edge.
  struct End {
    std::int32_t apex;
    std::int32_t end;
  };
  struct Corner {
    std::int32_t apex;
    std::int32_t first;
    std::int32_t second;
    std::int32_t group;
  };
  struct Side {
    std::int32_t a;
    std::int32_t b;
    std::int32_t last;
    std::int32_t group;
  };
  std::vector<End> ends;
  std::vector<Corner> corners;
  std::vector<Side> sides;
  ends.reserve(2 * segments.size());
  for (const TetSnapshotSegment& segment : segments) {
    const std::int32_t a = mesh.vertex_slot(segment.vertices[0]);
    const std::int32_t b = mesh.vertex_slot(segment.vertices[1]);
    if (!mesh.live_vertex(a) || !mesh.live_vertex(b)) return false;
    ends.push_back({a, b});
    ends.push_back({b, a});
  }
  // Corners are read only at segment apexes.
  const bool with_corners = !ends.empty();
  sides.reserve(3 * facets.size());
  if (with_corners) corners.reserve(3 * facets.size());
  for (const TetSnapshotFacet& facet : facets) {
    std::int32_t slots[3];
    for (int corner = 0; corner < 3; ++corner) {
      slots[corner] = mesh.vertex_slot(facet.vertices[static_cast<std::size_t>(corner)]);
      if (!mesh.live_vertex(slots[corner])) return false;
    }
    for (int corner = 0; corner < 3; ++corner) {
      const std::int32_t next = slots[(corner + 1) % 3];
      const std::int32_t last = slots[(corner + 2) % 3];
      const std::int32_t low = std::min(slots[corner], next);
      const std::int32_t high = std::max(slots[corner], next);
      sides.push_back({low, high, last, facet.source});
      if (with_corners) corners.push_back({slots[corner], next, last, facet.source});
    }
  }
  std::stable_sort(ends.begin(), ends.end(),
                   [](const End& x, const End& y) { return x.apex < y.apex; });
  std::stable_sort(corners.begin(), corners.end(),
                   [](const Corner& x, const Corner& y) { return x.apex < y.apex; });
  std::stable_sort(sides.begin(), sides.end(), [](const Side& x, const Side& y) {
    return x.a != y.a ? x.a < y.a : x.b < y.b;
  });
  const auto direction = [&](std::int32_t from, std::int32_t to, double* out) {
    const double* a = mesh.point(from);
    const double* b = mesh.point(to);
    const auto world = mesh.metric_transform({b[0] - a[0], b[1] - a[1], b[2] - a[2]});
    std::copy(world.begin(), world.end(), out);
  };

  // The groups of the facets at an apex that hold the segment (apex, end):
  // those facets' corners at the apex, which are the corners with `end` as a
  // side.
  std::vector<std::int32_t> edge_groups;
  for (std::size_t begin = 0, stop = 0; begin < ends.size(); begin = stop) {
    const std::int32_t apex = ends[begin].apex;
    for (stop = begin; stop < ends.size() && ends[stop].apex == apex; ++stop) {}
    const auto at_apex = std::equal_range(
        corners.begin(), corners.end(), Corner{apex, -1, -1, kNoConstraint},
        [](const Corner& x, const Corner& y) { return x.apex < y.apex; });
    double smallest = 90.0;
    for (std::size_t first = begin; first < stop; ++first) {
      double ray[3];
      direction(apex, ends[first].end, ray);
      for (std::size_t second = first + 1; second < stop; ++second) {
        double other[3];
        direction(apex, ends[second].end, other);
        const double cosine = cosine3(ray, other);
        if (cosine > kRightAngleCosine) smallest = std::min(smallest, degrees(cosine));
      }
      const std::int32_t end = ends[first].end;
      edge_groups.clear();
      for (auto corner = at_apex.first; corner != at_apex.second; ++corner) {
        if (corner->first == end || corner->second == end) edge_groups.push_back(corner->group);
      }
      for (auto corner = at_apex.first; corner != at_apex.second; ++corner) {
        if (std::find(edge_groups.begin(), edge_groups.end(), corner->group) !=
            edge_groups.end()) {
          continue;
        }
        double rays[2][3];
        direction(apex, corner->first, rays[0]);
        direction(apex, corner->second, rays[1]);
        if (cosine3(ray, rays[0]) > kRightAngleCosine ||
            cosine3(ray, rays[1]) > kRightAngleCosine) {
          smallest = std::min(smallest, ray_sector_angle(ray, rays[0], rays[1],
                                                         plane_normal(corner->group)));
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
  for (std::size_t begin = 0, stop = 0; begin < sides.size(); begin = stop) {
    const std::int32_t a = sides[begin].a;
    const std::int32_t b = sides[begin].b;
    for (stop = begin; stop < sides.size() && sides[stop].a == a && sides[stop].b == b; ++stop) {}
    // Only facets of different groups meet at a ridge.
    if (std::all_of(sides.begin() + static_cast<std::ptrdiff_t>(begin),
                    sides.begin() + static_cast<std::ptrdiff_t>(stop),
                    [&](const Side& side) { return side.group == sides[begin].group; })) {
      continue;
    }
    double along[3];
    direction(a, b, along);
    const double length_squared = dot3(along, along);
    if (!(length_squared > 0.0)) continue;
    double smallest = 90.0;
    for (std::size_t first = begin; first < stop; ++first) {
      for (std::size_t second = first + 1; second < stop; ++second) {
        if (sides[first].group == sides[second].group) continue;
        double normal[2][3];
        for (const std::size_t side : {first, second}) {
          double offset[3];
          direction(a, sides[side].last, offset);
          double* inward = normal[side == first ? 0 : 1];
          if (const double* plane = plane_normal(sides[side].group); plane != nullptr) {
            // The declared plane's direction perpendicular to the edge, toward the facet.
            cross3(plane, along, inward);
            if (dot3(inward, offset) < 0.0) {
              for (int axis = 0; axis < 3; ++axis) inward[axis] = -inward[axis];
            }
            continue;
          }
          const double height = dot3(offset, along) / length_squared;
          for (int axis = 0; axis < 3; ++axis) inward[axis] = offset[axis] - height * along[axis];
        }
        const double cosine = cosine3(normal[0], normal[1]);
        if (cosine > kRightAngleCosine &&
            (domain_wedge(a, b, sides[first].last, sides[second].last) ||
             domain_wedge(a, b, sides[second].last, sides[first].last))) {
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
// dense grid in which every feature is registered in the cells its region's
// bounding box meets.
class ProtectionIndex {
 public:
  ProtectionIndex(const TetMesh& mesh, std::span<const AcuteFeature> features)
      : mesh_(mesh), features_(features) {}

  bool build() {
    if (features_.empty()) return true;
    std::vector<double> radii;
    radii.reserve(features_.size());
    boxes_.resize(features_.size());
    double extent = 0.0;
    double frame_lower[3]{std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::infinity()};
    double frame_upper[3]{-frame_lower[0], -frame_lower[1], -frame_lower[2]};
    for (std::size_t row = 0; row < features_.size(); ++row) {
      const AcuteFeature& feature = features_[row];
      double* lower = boxes_[row].data();
      double* upper = lower + 3;
      box(feature, lower, upper);
      for (int axis = 0; axis < 3; ++axis) {
        extent = std::max(extent, upper[axis] - lower[axis]);
        frame_lower[axis] = std::min(frame_lower[axis], lower[axis]);
        frame_upper[axis] = std::max(frame_upper[axis], upper[axis]);
      }
      radii.push_back(feature.radius);
    }
    std::nth_element(radii.begin(), radii.begin() + static_cast<std::ptrdiff_t>(radii.size() / 2),
                     radii.end());
    // Cells of the median radius, coarsened so that no region box spans more
    // than 17 cells per axis.
    if (!frame_.fit(frame_lower, frame_upper, std::max(radii[radii.size() / 2], extent / 16.0),
                    std::max(4096.0, 8.0 * static_cast<double>(features_.size())))) {
      return false;
    }
    grid_.build(frame_, features_.size(), [&](std::size_t row, std::size_t* first,
                                              std::size_t* last) {
      for (std::size_t axis = 0; axis < 3; ++axis) {
        first[axis] = frame_.coordinate(boxes_[row][axis], axis);
        last[axis] = frame_.coordinate(boxes_[row][axis + 3], axis);
      }
      return true;
    }, [](std::size_t row) { return static_cast<std::int32_t>(row); });
    std::vector<std::array<double, 6>>().swap(boxes_);
    return true;
  }

  // The lowest-indexed feature whose protected region holds `point`, else -1.
  std::int32_t find(const double* point) const {
    if (features_.empty()) return -1;
    const auto world = metric_point(mesh_, point);
    const std::size_t at = frame_.index(frame_.coordinate(world[0], 0),
                                        frame_.coordinate(world[1], 1),
                                        frame_.coordinate(world[2], 2));
    std::int32_t protecting = -1;
    for (std::int32_t row : grid_.items(at)) {
      if (protecting >= 0 && row > protecting) continue;
      const AcuteFeature& feature = features_[static_cast<std::size_t>(row)];
      const double distance =
          feature.ridge < 0
              ? metric_distance(mesh_, point, mesh_.point(feature.apex))
              : metric_segment_distance(mesh_, point, mesh_.point(feature.apex),
                                        mesh_.point(feature.ridge));
      if (distance < feature.radius) protecting = row;
    }
    return protecting;
  }

 private:
  void box(const AcuteFeature& feature, double* lower, double* upper) const {
    const auto apex = metric_point(mesh_, mesh_.point(feature.apex));
    const auto end = feature.ridge < 0 ? apex : metric_point(mesh_, mesh_.point(feature.ridge));
    for (int axis = 0; axis < 3; ++axis) {
      lower[axis] = std::min(apex[axis], end[axis]) - feature.radius;
      upper[axis] = std::max(apex[axis], end[axis]) + feature.radius;
    }
  }

  const TetMesh& mesh_;
  std::span<const AcuteFeature> features_;
  std::vector<std::array<double, 6>> boxes_;
  DenseFrame frame_;
  DenseBuckets<std::int32_t> grid_;
};

}  // namespace

std::int32_t refine_mesh(TetMesh& mesh, double radius_edge_bound,
                         std::span<const double> vertex_sizes,
                         std::int64_t& inserted_vertices,
                         std::int64_t& unmet_cells) {
  inserted_vertices = 0;
  unmet_cells = 0;
  if (!std::isfinite(radius_edge_bound) || radius_edge_bound < 0.0) {
    return PHX_MC_INVALID_ARGUMENT;
  }

  const std::int32_t installed = mesh.install_vertex_sizes(vertex_sizes);
  if (installed != PHX_MC_OK) return installed;
  // Retained size target of a vertex by logical ID (0: none).
  const auto size_of = [&](std::uint64_t id) { return mesh.vertex_size(mesh.vertex_slot(id)); };

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
    const double first = size_of(edge[0]);
    const double second = size_of(edge[1]);
    if (!(first > 0.0) || !(second > 0.0)) return false;
    const double scale = std::max(first, second);
    mean_size = scale * 0.5 * (first / scale + second / scale);
    edge_target = mean_size * radius_to_edge;
    return edge_target > 0.0 && !std::isnan(edge_target);
  };
  // Local incidence and constrained-bank row indices: a boundary split or a
  // cone insertion touches only the cells, facets and segments around it.
  detail::LocalIncidence incidence;
  if (!incidence.initialize(mesh)) return PHX_MC_INTERNAL_ERROR;
  // Only boundary splits read the facet rows by face; a fixed boundary only
  // checks its faces distinct.
  std::map<FaceKey, std::size_t> facet_rows_by_face;
  if (mesh.policy_ != BoundaryPolicy::kFixed) {
    for (std::size_t row = 0; row < mesh.facets_.size(); ++row) {
      if (!facet_rows_by_face.emplace(sorted_face(mesh.facets_[row].vertices), row).second) {
        return PHX_MC_INTERNAL_ERROR;
      }
    }
  } else {
    std::vector<FaceKey> faces;
    faces.reserve(mesh.facets_.size());
    for (const TetSnapshotFacet& facet : mesh.facets_) faces.push_back(sorted_face(facet.vertices));
    std::sort(faces.begin(), faces.end());
    if (std::adjacent_find(faces.begin(), faces.end()) != faces.end()) {
      return PHX_MC_INTERNAL_ERROR;
    }
  }
  TetSegmentIndex segment_rows_by_edge;
  for (std::size_t row = 0; row < mesh.segments_.size(); ++row) {
    const TetSnapshotSegment& segment = mesh.segments_[row];
    if (!segment_rows_by_edge
             .emplace(sorted_edge(segment.vertices[0], segment.vertices[1]), row)
             .second) {
      return PHX_MC_INTERNAL_ERROR;
    }
  }
  incidence.bind_segment_index(segment_rows_by_edge);
  // Embedded segments: constrained segments on no constrained face (a curve
  // inside a region). No cavity boundary face carries one, so its
  // encroachment is tested on the cells around a center, its split cavity is
  // seeded with its edge star, and its halves stay embedded. The constrained
  // faces are the facet bank's rows, so a segment on a facet edge is on one;
  // only the others read their edge star.
  std::vector<EdgeKey> facet_edges;
  facet_edges.reserve(3 * mesh.facets_.size());
  for (const TetSnapshotFacet& facet : mesh.facets_) {
    for (std::size_t corner = 0; corner < 3; ++corner) {
      facet_edges.push_back(sorted_edge(facet.vertices[corner], facet.vertices[(corner + 1) % 3]));
    }
  }
  std::sort(facet_edges.begin(), facet_edges.end());
  std::set<EdgeKey> embedded_segments;
  for (const TetSnapshotSegment& segment : mesh.segments_) {
    const std::int32_t first = mesh.vertex_slot(segment.vertices[0]);
    const std::int32_t second = mesh.vertex_slot(segment.vertices[1]);
    if (!mesh.live_vertex(first) || !mesh.live_vertex(second)) return PHX_MC_INTERNAL_ERROR;
    if (std::binary_search(facet_edges.begin(), facet_edges.end(),
                           sorted_edge(segment.vertices[0], segment.vertices[1]))) {
      continue;
    }
    const std::span<const std::int32_t> star = incidence.edge_star(first, second);
    bool on_face = false;
    for (std::int32_t cell : star) {
      for (int face = 0; face < 4 && !on_face; ++face) {
        const std::int32_t opposite = mesh.tet(cell).v[face];
        on_face = opposite != first && opposite != second &&
                  mesh.complex_.constraint(cell, face) != kNoConstraint;
      }
      if (on_face) break;
    }
    if (!star.empty() && !on_face) {
      embedded_segments.insert(sorted_edge(segment.vertices[0], segment.vertices[1]));
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
             install_witness(mesh, corners, stratum, row, parameters, vertex) &&
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
            install_witness(mesh, corners, SourceStratum::kSegment, row, parameters, vertex)) {
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
    // A facet edge of a declared plane is split on the plane: at the exact
    // mean of the endpoints' plane parameters when both are witnessed on it
    // (the edge's midpoint), else at the projection of the carrier midpoint.
    if (groups.size() == 1) {
      const auto declared = mesh.declared_groups_.find(*groups.begin());
      if (declared != mesh.declared_groups_.end()) {
        const std::int32_t row = declared->second.rows.front();
        const SourceWitness& first = mesh.witness(first_slot);
        const SourceWitness& second = mesh.witness(second_slot);
        const auto on_plane = [&](const SourceWitness& witness) {
          return witness.stratum == SourceStratum::kPlane &&
                 mesh.source_face_groups_[static_cast<std::size_t>(witness.entity)] ==
                     *groups.begin();
        };
        double parameters[2]{};
        const bool installed =
            on_plane(first) && on_plane(second) &&
                    exact_half_sum(first.parameters[0], second.parameters[0], parameters[0]) &&
                    exact_half_sum(first.parameters[1], second.parameters[1], parameters[1])
                ? install_plane_parameters(mesh, row, parameters, vertex)
                : install_plane_witness(mesh, row, midpoint, vertex);
        return installed && !same_carrier(vertex.carrier.data(), mesh.point(first_slot)) &&
               !same_carrier(vertex.carrier.data(), mesh.point(second_slot));
      }
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
              install_witness(mesh, corners, SourceStratum::kFacet, row, parameters, vertex)) {
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
  // Encroachment follows Shewchuk's order: when c lies inside the diametral
  // ball of an embedded segment (on no constrained face) that is an edge of
  // c's conflict region, beyond a constrained face, or inside the diametral
  // ball of a constrained segment or the equatorial ball of a constrained
  // face bounding its cavity, a boundary that may be split is split instead
  // of inserting c: the encroached subsegment at its midpoint, else the
  // encroached subfacet at its circumcenter (a subsegment that circumcenter
  // encroaches, or that the walk to it crosses, is split instead), each at
  // an exact source parameter. A
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
  // lfs(x) estimate: the Lipschitz bound min_u (lambda(u) + |x - u|) from the
  // natural neighbors u of x, lambda precomputed per vertex on a grid
  // (local_feature_sizes), floored at the coordinate spacing. Every term is
  // an upper bound of the true lfs(x) (lfs is 1-Lipschitz), so the estimate
  // never falls below lfs_min > 0 and insertion_guard keeps the packing
  // argument (Ruppert, Shewchuk) bounding the vertex count.
  // feature_size_at keeps the metric distances from its point to its
  // neighbors (in neighbor order) for clear_of.
  std::vector<double> neighbor_distances;
  const CoordinateResolution coordinate_resolution(mesh);
  const auto feature_size_at = [&](const double* point, std::span<const std::int32_t> neighbors) {
    neighbor_distances.clear();
    double size = std::numeric_limits<double>::infinity();
    for (std::int32_t vertex : neighbors) {
      const double distance = metric_distance(mesh, point, mesh.point(vertex));
      neighbor_distances.push_back(distance);
      size = std::min(size, feature_sizes[static_cast<std::size_t>(vertex)] + distance);
    }
    return std::max(size, coordinate_resolution(point));
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
  // The spacing guard: no natural neighbor of a point (a vertex of its
  // cavity, which holds its nearest vertex in a Delaunay neighborhood) lies
  // closer than `radius`; it reads the neighbors feature_size_at measured
  // last.
  const auto clear_of = [&](double radius) {
    if (!(radius > 0.0) || !std::isfinite(radius)) return false;
    for (double distance : neighbor_distances) {
      if (distance < radius) return false;
    }
    return true;
  };
  // Acute features only protect boundary splits, and a fixed boundary is
  // never split, so it has none.
  const bool boundary_splittable = mesh.policy_ != BoundaryPolicy::kFixed;
  std::vector<AcuteFeature> acute_features;
  if (boundary_splittable) {
    std::unordered_map<std::int32_t, std::array<double, 3>> plane_normals;
    for (const auto& [group, declared] : mesh.declared_groups_) {
      const double* plane = mesh.face_plane(declared.rows.front());
      std::array<double, 3> normal{};
      const auto u = mesh.metric_transform({plane[3], plane[4], plane[5]});
      const auto v = mesh.metric_transform({plane[6], plane[7], plane[8]});
      cross3(u.data(), v.data(), normal.data());
      plane_normals.emplace(group, normal);
    }
    if (!acute_input_features(mesh, incidence, mesh.facets_, mesh.segments_, feature_sizes,
                              plane_normals, acute_features)) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  ProtectionIndex protection(mesh, acute_features);
  if (!protection.build()) return PHX_MC_INVALID_INPUT;
  const auto input_angle = [&](std::int32_t feature) {
    const AcuteFeature& acute = acute_features[static_cast<std::size_t>(feature)];
    return UnmetCause{PHX_MC_TET_UNMET_INPUT_ANGLE, mesh.vertex_id(acute.apex), acute.angle};
  };

  // Cells to evaluate in the next refinement round, deduplicated per round.
  std::vector<std::int32_t> pending;
  std::vector<std::uint32_t> pending_round;
  std::uint32_t round = 1;
  const auto enqueue_slot = [&](std::int32_t slot) {
    if (slot < 0 || !mesh.complex_.live(slot) || is_ghost(mesh.tet(slot))) return;
    if (pending_round.size() <= static_cast<std::size_t>(slot)) {
      pending_round.resize(mesh.complex_.tets.size() + 1, 0U);
    }
    if (pending_round[static_cast<std::size_t>(slot)] == round) return;
    pending_round[static_cast<std::size_t>(slot)] = round;
    pending.push_back(slot);
  };
  const auto enqueue = [&](std::uint64_t id) { enqueue_slot(mesh.cell_slot(id)); };
  // Flips the cells a boundary insertion created toward the constrained
  // Delaunay triangulation, then enqueues them with the cells the flips
  // created. Interior Bowyer-Watson insertions need no flips.
  const auto settle = [&](std::span<const std::uint64_t> created) -> std::int32_t {
    flipped.clear();
    const std::int32_t status =
        flip_to_constrained_delaunay(mesh, incidence, created, is_segment, flipped);
    for (std::uint64_t id : created) enqueue(id);
    for (std::uint64_t id : flipped) enqueue(id);
    return status;
  };
  for (std::size_t row = 0; row < mesh.complex_.tets.size(); ++row) {
    enqueue_slot(static_cast<std::int32_t>(row));
  }

  const auto edit_error = [](TetEditStatus status) -> std::int32_t {
    if (status == TetEditStatus::kAllocationFailure) return PHX_MC_ALLOCATION_FAILURE;
    if (status == TetEditStatus::kIndexOverflow) return PHX_MC_INDEX_RANGE_ERROR;
    if (status == TetEditStatus::kStaleGeneration) return PHX_MC_INTERNAL_ERROR;
    return PHX_MC_OK;
  };
  // Equatorial ball of a triangle: the smallest sphere through its corners.
  const auto circumcircle = [&](const double* const corners[3], double* center) {
    double ab[3], ac[3];
    const auto ab_world = mesh.metric_transform({corners[1][0] - corners[0][0],
                                                  corners[1][1] - corners[0][1],
                                                  corners[1][2] - corners[0][2]});
    const auto ac_world = mesh.metric_transform({corners[2][0] - corners[0][0],
                                                  corners[2][1] - corners[0][1],
                                                  corners[2][2] - corners[0][2]});
    std::copy(ab_world.begin(), ab_world.end(), ab);
    std::copy(ac_world.begin(), ac_world.end(), ac);
    const double normal[3]{ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2],
                           ab[0] * ac[1] - ab[1] * ac[0]};
    const double normal_squared =
        normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
    if (!(normal_squared > 0.0) || !std::isfinite(normal_squared)) return -1.0;
    const double ab_squared = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    const double ac_squared = ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2];
    double radius_squared = 0.0;
    std::array<double, 3> world_offset{};
    for (int axis = 0; axis < 3; ++axis) {
      const int next = (axis + 1) % 3;
      const int last = (axis + 2) % 3;
      const double offset = (ab_squared * (ac[next] * normal[last] - ac[last] * normal[next]) +
                             ac_squared * (normal[next] * ab[last] - normal[last] * ab[next])) /
                            (2.0 * normal_squared);
      radius_squared += offset * offset;
      world_offset[axis] = offset;
    }
    const auto index_offset = mesh.metric_inverse_transform(world_offset);
    for (int axis = 0; axis < 3; ++axis) center[axis] = corners[0][axis] + index_offset[axis];
    return radius_squared;
  };
  const auto in_equatorial_ball = [&](const double* const corners[3], const double* point) {
    double center[3];
    const double radius_squared = circumcircle(corners, center);
    // `circumcircle` leaves `center` unwritten when it returns the degenerate sentinel.
    if (!(radius_squared > 0.0)) return false;
    if (!mesh.metric_active()) {
      return geometry::squared_distance(point, center) < radius_squared;
    }
    const double distance = metric_distance(mesh, point, center);
    return distance * distance < radius_squared;
  };
  // Diametral ball of a segment.
  const auto in_diametral_ball = [&](const double* a, const double* b, const double* point) {
    const std::array<double, 3> first{point[0] - a[0], point[1] - a[1], point[2] - a[2]};
    const std::array<double, 3> second{point[0] - b[0], point[1] - b[1], point[2] - b[2]};
    return mesh.metric_dot(first, second) < 0.0;
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
  // circumcircles contain it, else of the seeds alone, or with no seeds (an
  // embedded segment) of the segment's edge star, under the insertion
  // guard for the natural radius `radius` (none when 0) and size target
  // `target` (none when 0). A guarded vertex inside the protected region of
  // an acute input feature is not inserted unless `sizing_pass` (the size
  // field, not an encroachment chain, drives the split; the target and the
  // guard bound it); `protecting` is set to that feature. A facet vertex
  // inside the diametral ball of a subsegment of its cavity is not inserted;
  // `encroached_segment` is set to that subsegment. Fills `fan_rows` with the
  // facet rows it wrote and `split_halves` with the halves of a split
  // embedded segment. Returns OK and sets `split` when the vertex was
  // inserted.
  std::int32_t protecting = -1;
  bool sizing_pass = false;
  bool selective_fixed = false;
  TetInsertionCavity cavity;
  std::vector<std::uint64_t> created_cells;
  std::vector<std::int32_t> cavity_vertices;
  std::vector<std::pair<std::int32_t, int>> across;
  std::vector<std::pair<std::int32_t, int>> disk_sides;
  std::vector<std::size_t> disk_rows;
  std::vector<std::size_t> split_segment_rows;
  std::vector<std::size_t> fan_rows;
  std::vector<std::int32_t> boundary_cavity;
  std::vector<std::int32_t> segment_star;
  std::vector<FaceKey> removed_keys;
  std::optional<EdgeKey> encroached_segment;
  // The halves of the embedded segment the last boundary insertion split.
  std::optional<std::array<EdgeKey, 2>> split_halves;
  const auto insert_boundary_vertex =
      [&](TetEditVertex& inserted, std::span<const std::pair<std::int32_t, int>> seeds,
          const EdgeKey* segment_edge, double radius, double target,
          bool& split) -> std::int32_t {
    split = false;
    protecting = -1;
    encroached_segment.reset();
    fan_rows.clear();
    split_halves.reset();
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
      segment_star.clear();
      if (seeds.empty()) {
        // An embedded segment: no disk, so the whole-disk pass is the only one.
        if (!whole_disk || segment_edge == nullptr) break;
        const std::span<const std::int32_t> star =
            incidence.edge_star(split_slots[0], split_slots[1]);
        segment_star.assign(star.begin(), star.end());
      }
      if (mesh.boundary_insertion_cavity(inserted, disk_sides, split_slots, segment_star,
                                         boundary_cavity, &segment_rows_by_edge) != TetCavityStatus::kFound) {
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
      if (radius > 0.0 && !clear_of(insertion_guard(feature_size, radius, target))) {
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
        const bool embedded = embedded_segments.erase(*segment_edge) != 0;
        std::array<EdgeKey, 2> halves{};
        for (int index = 0; index < 2; ++index) {
          const std::size_t half = index == 0 ? row : appended;
          const auto& vertices = mesh.segments_[half].vertices;
          halves[static_cast<std::size_t>(index)] = sorted_edge(vertices[0], vertices[1]);
          if (!segment_rows_by_edge.emplace(halves[static_cast<std::size_t>(index)], half)
                   .second) {
            return PHX_MC_INTERNAL_ERROR;
          }
        }
        if (embedded) {
          embedded_segments.insert(halves[0]);
          embedded_segments.insert(halves[1]);
          split_halves = halves;
        }
      }
      const std::int32_t inserted_slot = mesh.vertex_slot(inserted.id);
      if (target > 0.0) mesh.set_vertex_size(inserted_slot, target);
      record_feature_size(inserted_slot, feature_size);
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
        if (mesh.facets_[row->second].protected_facet) {
          selective_fixed = true;
          return PHX_MC_OK;
        }
        if (std::find(edge_facet_rows.begin(), edge_facet_rows.end(), row->second) ==
            edge_facet_rows.end()) {
          edge_facet_rows.push_back(row->second);
          edge_sides.emplace_back(cell, face);
        }
      }
    }
    // An edge on no constrained face is split only as an embedded segment,
    // its cavity seeded with its edge star.
    if (edge_sides.empty() && !embedded_segments.contains(edge)) return PHX_MC_OK;
    const auto segment = segment_rows_by_edge.find(edge);
    const bool on_segment = segment != segment_rows_by_edge.end();
    TetEditVertex inserted;
    if (!make_boundary_split(first, second, edge_facet_rows,
                             on_segment ? &mesh.segments_[segment->second] : nullptr, inserted)) {
      return PHX_MC_OK;
    }
    const double radius =
        guarded ? 0.5 * metric_distance(mesh, mesh.point(first), mesh.point(second))
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
    const auto protected_row = facet_rows_by_face.find(face_key_of(current.first, current.second));
    if (protected_row == facet_rows_by_face.end()) return PHX_MC_INTERNAL_ERROR;
    if (mesh.facets_[protected_row->second].protected_facet) {
      selective_fixed = true;
      return PHX_MC_OK;
    }
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
    const double radius = metric_distance(mesh, center.data(), mesh.point(facet[0]));
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
    if (const auto declared = mesh.declared_groups_.find(source);
        declared != mesh.declared_groups_.end()) {
      // On a declared plane the circumcenter is witnessed on the plane itself.
      if (!install_plane_witness(mesh, declared->second.rows.front(), center.data(), inserted)) {
        return PHX_MC_OK;
      }
    } else {
      std::int32_t row = -1;
      double parameters[2]{};
      const double* corners[3]{};
      rows_through(mesh, SourceStratum::kFacet, rows->second, facet, through_rows);
      if (!locate_in_rows(mesh, SourceStratum::kFacet, through_rows, center.data(), row,
                          parameters) ||
          !mesh.source_corners(SourceStratum::kFacet, row, corners) ||
          !install_witness(mesh, corners, SourceStratum::kFacet, row, parameters, inserted)) {
        return PHX_MC_OK;
      }
    }
    const std::int32_t corner_slots[4]{facet[0], facet[1], facet[2], facet[2]};
    double point_size = 0.0;
    const bool has_size = cell_target(mesh, corner_slots, point_size);
    const std::pair<std::int32_t, int> seed[1]{current};
    const std::int32_t status = insert_boundary_vertex(
        inserted, seed, nullptr, radius, has_size ? point_size : 0.0, split);
    if (status != PHX_MC_OK || split || !encroached_segment.has_value()) return status;
    const EdgeKey edge = *encroached_segment;
    return split_segment(mesh.vertex_slot(edge[0]), mesh.vertex_slot(edge[1]));
  };

  // The embedded subsegment whose diametral ball holds `center` among the
  // edges of the center's conflict region: the cells reached from `root`
  // (the cell whose circumcenter it is) across unconstrained faces into cells
  // whose circumspheres strictly hold it. The first found in that
  // deterministic order is set in `encroached`.
  std::vector<std::int32_t> conflict_stack;
  std::unordered_set<std::int32_t> conflict_cells;
  const auto encroached_embedded_segment = [&](std::int32_t root, const double* center,
                                               EdgeKey& encroached) {
    if (embedded_segments.empty()) return false;
    conflict_cells.clear();
    conflict_cells.insert(root);
    conflict_stack.assign(1, root);
    while (!conflict_stack.empty()) {
      const std::int32_t cell = conflict_stack.back();
      conflict_stack.pop_back();
      const Tetrahedron& tetrahedron = mesh.tet(cell);
      for (int corner = 0; corner < 4; ++corner) {
        for (int other = corner + 1; other < 4; ++other) {
          const std::int32_t a = tetrahedron.v[corner];
          const std::int32_t b = tetrahedron.v[other];
          const EdgeKey edge = sorted_edge(mesh.vertex_id(a), mesh.vertex_id(b));
          if (embedded_segments.contains(edge) &&
              in_diametral_ball(mesh.point(a), mesh.point(b), center)) {
            encroached = edge;
            return true;
          }
        }
      }
      for (int face = 0; face < 4; ++face) {
        const std::int32_t next = tetrahedron.n[face];
        if (mesh.complex_.constraint(cell, face) != kNoConstraint || !mesh.complex_.live(next) ||
            is_ghost(mesh.tet(next)) || conflict_cells.contains(next)) {
          continue;
        }
        const Tetrahedron& candidate = mesh.tet(next);
        if (mesh.metric_insphere(mesh.point(candidate.v[0]), mesh.point(candidate.v[1]),
                                 mesh.point(candidate.v[2]), mesh.point(candidate.v[3]),
                                 center) > 0) {
          conflict_cells.insert(next);
          conflict_stack.push_back(next);
        }
      }
    }
    return false;
  };

  // Boundary edges longer than their size targets are split first, longest
  // relative to its target first (ties: smallest key). Splitting never moves
  // a vertex, so a queued ratio stays exact until its edge is split; split
  // edges never reappear. A facet edge whose midpoint would encroach a
  // subsegment splits that subsegment first (under the insertion guard) and
  // is queued again: an unchecked midpoint next to a segment starts a chain
  // of ever thinner subfacets converging onto it. Embedded segments are
  // queued with the facet edges and their halves after each split.
  sizing_pass = true;
  std::set<EdgeKey> blocked_boundary_edges;
  const bool has_sizes = std::any_of(mesh.vertex_sizes_.begin(), mesh.vertex_sizes_.end(),
                                     [](double size) { return size > 0.0; });
  if (has_sizes && boundary_splittable) {
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
          metric_distance(mesh, mesh.point(first_slot), mesh.point(second_slot)) /
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
    for (const EdgeKey& edge : embedded_segments) {
      if (!queue_boundary_edge(edge[0], edge[1])) return PHX_MC_INTERNAL_ERROR;
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
      if (split_halves.has_value()) {
        for (const EdgeKey& half : *split_halves) {
          if (!queue_boundary_edge(half[0], half[1])) return PHX_MC_INTERNAL_ERROR;
        }
      }
    }
  }
  sizing_pass = false;

  // Interior refinement in rounds (the batched scheme of Gmsh's HXT): every
  // violating cell of a round proposes its circumcenter; proposals are taken
  // along the Hilbert curve, each walking from its own cell while that is
  // still live, so walks stay short and consecutive cavities share cache
  // lines. The first round is sorted by circumcenter; the cells an insertion
  // creates are the next round's candidates, already in curve order as their
  // insertions were. Bowyer-Watson insertion keeps the constrained Delaunay
  // property where the cavity is Delaunay, so interior insertions are not
  // followed by flips. A cell left unchanged keeps its cause and is not
  // proposed again; the round count is capped (no cap is reached on the
  // packing-bounded runs).
  HilbertFrame frame;
  {
    double lower[3]{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity()};
    double upper[3]{-lower[0], -lower[1], -lower[2]};
    for (std::size_t vertex = 0; vertex < mesh.vertex_ids_.size(); ++vertex) {
      if (mesh.vertex_live_[vertex] == 0) continue;
      for (int axis = 0; axis < 3; ++axis) {
        lower[axis] = std::min(lower[axis], mesh.points_[3 * vertex + static_cast<std::size_t>(axis)]);
        upper[axis] = std::max(upper[axis], mesh.points_[3 * vertex + static_cast<std::size_t>(axis)]);
      }
    }
    double extent = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      frame.lower[axis] = lower[axis];
      extent = std::max(extent, upper[axis] - lower[axis]);
    }
    frame.scale =
        extent > 0.0 && std::isfinite(extent) ? static_cast<double>(kHilbertMax) / extent : 0.0;
  }
  // A candidate cell of a round, keyed by its circumcenter in the first round.
  // Most candidates of a round die to earlier insertions of the same round
  // before their turn, so a later round's candidate is screened only when its
  // turn comes, and every proposal's shape is measured (cell_shape) then; its
  // target is a function of its corners, which holds while its ID is live.
  struct Proposal {
    std::uint64_t key;
    std::uint64_t cell_id;
    std::int32_t slot;
    bool has_target;
    double target;
  };
  std::vector<Proposal> proposals;
  // Every proposed cell ID: a live cell that misses a criterion when the
  // rounds end was proposed (each created cell is evaluated in the round
  // after its creation), so the unmet records measure only these.
  std::vector<std::uint64_t> proposed_ids;
  InteriorCavity interior(mesh.complex_);
  std::vector<std::int32_t> created_slots;
  std::vector<detail::LocalCell> removed_cells;
  std::unordered_set<std::uint64_t> apex_tried;
  const auto is_embedded = [&](std::int32_t a, std::int32_t b) {
    return embedded_segments.contains(sorted_edge(mesh.vertex_id(a), mesh.vertex_id(b)));
  };
  // Inserts the interior point `point` whose cavity `interior` holds when its
  // natural neighbors clear the guard of natural radius `radius` and size
  // `size` (0: none); `outcome` is 0 when inserted, 2 when refused. The
  // created cells are the next round's candidates, except those `meets`
  // (per cavity boundary row, from screen_cone; empty: none) marks as clearly
  // meeting the criteria, which the round would skip. Only boundary splits
  // and the blocked boundary edges read the local incidence after the initial
  // flips, so a fixed boundary leaves it stale.
  const auto commit_interior = [&](const double* point, double radius, double size,
                                   std::span<const std::uint8_t> meets,
                                   int& outcome) -> std::int32_t {
    outcome = 2;
    const double feature_size = feature_size_at(point, interior.vertices);
    if (!clear_of(insertion_guard(feature_size, radius, size))) {
      return PHX_MC_OK;
    }
    removed_cells.clear();
    if (boundary_splittable) {
      for (std::int32_t cell : interior.cells) {
        detail::LocalCell vertices{};
        std::copy_n(mesh.tet(cell).v, 4, vertices.begin());
        removed_cells.push_back(vertices);
      }
    }
    const TetEditStatus status =
        mesh.insert_interior_vertex(point, size, interior.cells, interior.boundary,
                                    interior.vertices, created_slots);
    const std::int32_t error = edit_error(status);
    if (error != PHX_MC_OK) return error;
    if (status != TetEditStatus::kApplied) return PHX_MC_OK;
    for (std::size_t row = 0; row < removed_cells.size(); ++row) {
      if (!incidence.erase(interior.cells[row], removed_cells[row])) return PHX_MC_INTERNAL_ERROR;
    }
    for (std::size_t row = 0; row < created_slots.size(); ++row) {
      const std::int32_t slot = created_slots[row];
      if (boundary_splittable) {
        detail::LocalCell vertices{};
        std::copy_n(mesh.tet(slot).v, 4, vertices.begin());
        if (!incidence.insert_logical(mesh, slot, vertices)) return PHX_MC_INTERNAL_ERROR;
      }
      if (meets.empty() || meets[row] == 0U) enqueue_slot(slot);
    }
    record_feature_size(static_cast<std::int32_t>(mesh.vertex_ids_.size()) - 1, feature_size);
    ++inserted_vertices;
    outcome = 0;
    return PHX_MC_OK;
  };
  // Builds the cavity of `point` reached from cell `start` and commits it;
  // `outcome` is 1 when the walk was blocked (interior.blocked_*).
  const auto insert_interior = [&](std::int32_t start, const double* point, double radius,
                                   double size, int& outcome) -> std::int32_t {
    outcome = 2;
    const InteriorCavity::Status found =
        interior.build(mesh, start, point, !embedded_segments.empty(), is_embedded);
    if (found == InteriorCavity::Status::kBlocked) {
      outcome = 1;
      return PHX_MC_OK;
    }
    if (found != InteriorCavity::Status::kFound) return PHX_MC_OK;
    return commit_interior(point, radius, size, {}, outcome);
  };
  // Sliver-aware placement in the picking region of Chew and of Li and Teng:
  // when a persistent cone cell of the circumcenter's cavity (screen_cone) has
  // a dihedral angle below kSliverDegrees (or above its supplement), points
  // displaced from the circumcenter by kPickFraction of the circumradius are
  // tried, up to kPickAttempts, along the running sum of the inward unit
  // normals of each attempt's worst cone face. The first attempt without such
  // a cell, else the best, is inserted. A candidate strictly inside the
  // proposal cell's circumball conflicts with it, so it is admitted only when
  // its cavity is found, holds the proposal cell, meets no constrained face
  // that could be split (encroachment was decided for the circumcenter) and
  // clears the spacing guard; the best attempt's cavity is kept while later
  // attempts build theirs. Natural radius and size stay the circumcenter's.
  // Slivers left by refinement are improvement's work; most of them start
  // here, as cone cells that meet the radius-edge and size criteria and so
  // are never refined. Six attempts: each sliver left costs improvement far
  // more than the attempts that avoid it (on the ledger L3 rows, three to six
  // attempts add about 0.02 s of refinement and save about 0.05 s of
  // improvement).
  constexpr double kSliverDegrees = 15.0;
  constexpr double kPickFraction = 0.3;
  constexpr int kPickAttempts = 6;
  const double sliver_sine = std::sin(kSliverDegrees * std::numbers::pi / 180.0);
  const double sliver_sine_squared = sliver_sine * sliver_sine;
  std::vector<std::uint8_t> cone_meets;
  std::vector<std::uint8_t> best_meets;
  // The best attempt's cavity while later attempts build theirs (the mesh is
  // not edited between them, so it stays current).
  std::vector<std::int32_t> best_cells;
  std::vector<BoundaryFacet> best_boundary;
  std::vector<std::int32_t> best_vertices;
  const auto commit_center = [&](std::int32_t start, const double* center, double radius,
                                 double size, int& outcome) -> std::int32_t {
    if (mesh.metric_active() || !embedded_segments.empty()) {
      return commit_interior(center, radius, size, {}, outcome);
    }
    double toward[3];
    double best = screen_cone(mesh, interior.boundary, center, radius_edge_bound, size, toward,
                              cone_meets);
    if (!(best < sliver_sine_squared)) {
      return commit_interior(center, radius, size, cone_meets, outcome);
    }
    double best_point[3]{center[0], center[1], center[2]};
    best_meets.swap(cone_meets);
    bool built_best = true;
    double offset[3]{};
    for (int attempt = 0; attempt < kPickAttempts; ++attempt) {
      for (int axis = 0; axis < 3; ++axis) offset[axis] += toward[axis];
      const double length = std::sqrt(dot3(offset, offset));
      if (!(length > 0.0)) break;
      double candidate[3];
      for (int axis = 0; axis < 3; ++axis) {
        candidate[axis] = center[axis] + kPickFraction * radius * offset[axis] / length;
      }
      if (!std::all_of(candidate, candidate + 3,
                       [](double value) { return coordinate_in_domain(value); })) {
        break;
      }
      if (built_best) {
        best_cells.swap(interior.cells);
        best_boundary.swap(interior.boundary);
        best_vertices.swap(interior.vertices);
      }
      built_best = false;
      if (interior.build(mesh, start, candidate, false, is_embedded) !=
          InteriorCavity::Status::kFound) {
        break;
      }
      if (std::find(interior.cells.begin(), interior.cells.end(), start) == interior.cells.end()) {
        break;
      }
      if (boundary_splittable &&
          std::any_of(interior.boundary.begin(), interior.boundary.end(), [&](const auto& entry) {
            return mesh.complex_.constraint(entry.tet, entry.slot) != kNoConstraint;
          })) {
        break;
      }
      const double feature_size = feature_size_at(candidate, interior.vertices);
      if (!clear_of(insertion_guard(feature_size, radius, size))) {
        break;
      }
      const double quality = screen_cone(mesh, interior.boundary, candidate, radius_edge_bound,
                                         size, toward, cone_meets);
      if (quality > best) {
        best = quality;
        std::copy_n(candidate, 3, best_point);
        best_meets.swap(cone_meets);
        built_best = true;
      }
      if (!(quality < sliver_sine_squared)) break;
    }
    if (!built_best) {
      interior.cells.swap(best_cells);
      interior.boundary.swap(best_boundary);
      interior.vertices.swap(best_vertices);
    }
    return commit_interior(best_point, radius, size, best_meets, outcome);
  };
  const auto refine_cell = [&](const Proposal& proposal) -> std::int32_t {
    selective_fixed = false;
    const std::int32_t root_slot = proposal.slot;
    const std::uint64_t cell_id = proposal.cell_id;
    geometry::TetrahedronShape shape;
    if (!cell_shape(mesh, mesh.tet(root_slot), shape)) {
      unmet_causes[cell_id] = {PHX_MC_TET_UNMET_NONREPRESENTABLE};
      return PHX_MC_OK;
    }
    const double center_size = proposal.target;
    const bool has_center_size = proposal.has_target;
    const double* center = shape.circumcenter.data();
    if (!std::all_of(shape.circumcenter.begin(), shape.circumcenter.end(),
                     [](double value) { return coordinate_in_domain(value); })) {
      unmet_causes[cell_id] = {PHX_MC_TET_UNMET_NONREPRESENTABLE};
      return PHX_MC_OK;
    }
    const double size = has_center_size ? center_size : 0.0;

    // Shewchuk: an encroached embedded subsegment is split before anything
    // else; no constrained face of the cavity carries it.
    std::int32_t embedded_protected = -1;
    if (boundary_splittable) {
      EdgeKey encroached{};
      if (encroached_embedded_segment(root_slot, center, encroached)) {
        bool split = false;
        protecting = -1;
        double mean_size = 0.0;
        double edge_target = 0.0;
        const bool has_size = boundary_target(encroached, mean_size, edge_target);
        const std::int32_t error =
            split_boundary_edge(encroached, has_size ? mean_size : 0.0, true, split);
        if (error != PHX_MC_OK) return error;
        if (split) {
          enqueue(cell_id);
          return PHX_MC_OK;
        }
        embedded_protected = protecting;
      }
    }

    const InteriorCavity::Status located =
        interior.build(mesh, root_slot, center, !embedded_segments.empty(), is_embedded);
    if (located == InteriorCavity::Status::kRefused) {
      unmet_causes[cell_id] = embedded_protected >= 0 ? input_angle(embedded_protected)
                                                      : UnmetCause{PHX_MC_TET_UNMET_REFUSED};
      return PHX_MC_OK;
    }
    if (located == InteriorCavity::Status::kBlocked) {
      bool split = false;
      protecting = -1;
      selective_fixed = false;
      const std::int32_t blocked_cell = interior.blocked_cell;
      const int blocked_face = interior.blocked_face;
      if (boundary_splittable) {
        const std::int32_t error = split_encroached_face(blocked_cell, blocked_face, center, split);
        if (error != PHX_MC_OK) return error;
      } else if (apex_tried.insert(cell_id).second) {
        // A fixed face the circumcenter lies beyond: insert the apex of the
        // regular tetrahedron on that face instead, once per cell.
        double apex[3];
        int outcome = 2;
        if (face_apex(mesh, blocked_cell, blocked_face, apex)) {
          const std::int32_t error =
              insert_interior(blocked_cell, apex, shape.circumradius, size, outcome);
          if (error != PHX_MC_OK) return error;
        }
        split = outcome == 0;
      }
      if (split) {
        enqueue(cell_id);
      } else if (protecting >= 0) {
        unmet_causes[cell_id] = input_angle(protecting);
      } else {
        unmet_causes[cell_id] = {!boundary_splittable || selective_fixed
                                     ? PHX_MC_TET_UNMET_FIXED_BOUNDARY
                                     : PHX_MC_TET_UNMET_REFUSED};
      }
      return PHX_MC_OK;
    }
    // Shewchuk: an encroached subsegment, then an encroached subfacet, of the
    // cavity's constrained boundary is split before the center is inserted;
    // a protected or refused split leaves that boundary fixed for this center.
    std::int32_t protected_by = embedded_protected;
    // A refused split attempt may have edited the mesh: the cavity is rebuilt.
    bool cavity_current = true;
    if (boundary_splittable) {
      std::int32_t encroached_cell = -1;
      int encroached_face = -1;
      bool segment_encroached = false;
      for (const BoundaryFacet& entry : interior.boundary) {
        const std::int32_t cell = entry.tet;
        const int face = entry.slot;
        if (mesh.complex_.constraint(cell, face) == kNoConstraint) continue;
        std::int32_t facet[3];
        oriented_facet(mesh.tet(cell).v, face, facet);
        for (int corner = 0; corner < 3 && !segment_encroached; ++corner) {
          const std::int32_t a = facet[corner];
          const std::int32_t b = facet[(corner + 1) % 3];
          segment_encroached =
              segment_rows_by_edge.contains(sorted_edge(mesh.vertex_id(a), mesh.vertex_id(b))) &&
              in_diametral_ball(mesh.point(a), mesh.point(b), center);
        }
        const double* face_corners[3]{mesh.point(facet[0]), mesh.point(facet[1]),
                                      mesh.point(facet[2])};
        if (segment_encroached ||
            (encroached_cell < 0 && in_equatorial_ball(face_corners, center))) {
          encroached_cell = cell;
          encroached_face = face;
        }
        if (segment_encroached) break;
      }
      if (encroached_cell >= 0) {
        bool split = false;
        protecting = -1;
        selective_fixed = false;
        cavity_current = false;
        const std::int32_t error =
            split_encroached_face(encroached_cell, encroached_face, center, split);
        if (error != PHX_MC_OK) return error;
        if (split) {
          enqueue(cell_id);
          return PHX_MC_OK;
        }
        protected_by = protecting;
      }
    }
    int outcome = 2;
    const std::int32_t error =
        cavity_current ? commit_center(root_slot, center, shape.circumradius, size, outcome)
                       : insert_interior(root_slot, center, shape.circumradius, size, outcome);
    if (error != PHX_MC_OK) return error;
    if (outcome != 0) {
      unmet_causes[cell_id] = protected_by >= 0 ? input_angle(protected_by)
                              : UnmetCause{selective_fixed ? PHX_MC_TET_UNMET_FIXED_BOUNDARY
                                                           : PHX_MC_TET_UNMET_REFUSED};
    }
    return PHX_MC_OK;
  };
  // Whether the live finite cell `slot` misses a criterion; fills the
  // proposal's target and, with `center`, its circumcenter.
  const auto violates = [&](std::int32_t slot, Proposal& proposal, double* center) {
    const Tetrahedron& tet = mesh.tet(slot);
    proposal.has_target = cell_target(mesh, tet.v, proposal.target);
    const CellScreen screen =
        mesh.metric_active()
            ? CellScreen::kUncertain
            : screen_cell(mesh.point(tet.v[0]), mesh.point(tet.v[1]), mesh.point(tet.v[2]),
                          mesh.point(tet.v[3]), radius_edge_bound,
                          proposal.has_target ? proposal.target
                                              : std::numeric_limits<double>::infinity(),
                          center);
    if (screen == CellScreen::kMeets) return false;
    if (screen == CellScreen::kUncertain) {
      geometry::TetrahedronShape shape;
      const bool measured = cell_shape(mesh, tet, shape);
      if (measured &&
          !(radius_edge_bound > 0.0 && shape.radius_edge_ratio > radius_edge_bound) &&
          !(proposal.has_target && shape.circumradius > proposal.target)) {
        return false;
      }
      if (center != nullptr) std::copy_n(shape.circumcenter.begin(), 3, center);
    }
    proposed_ids.push_back(proposal.cell_id);
    return true;
  };
  constexpr int kMaxRounds = 4096;
  for (int pass = 0; pass < kMaxRounds && !pending.empty(); ++pass) {
    proposals.clear();
    for (std::int32_t slot : pending) {
      Proposal proposal{};
      proposal.cell_id = mesh.cell_id(slot);
      proposal.slot = slot;
      // The first round is screened now and sorted by circumcenter (below);
      // a later round's cell is screened at its turn, if it is still live.
      if (pass == 0) {
        if (!mesh.complex_.live(slot) || is_ghost(mesh.tet(slot))) continue;
        double center[3];
        if (!violates(slot, proposal, center)) continue;
        proposal.key = frame.key(center);
      }
      proposals.push_back(proposal);
    }
    pending.clear();
    if (++round == 0) {
      std::fill(pending_round.begin(), pending_round.end(), 0U);
      round = 1;
    }
    // Later rounds keep the order in which the previous round's insertions,
    // taken along the curve, created their cells.
    if (pass == 0) {
      std::sort(proposals.begin(), proposals.end(), [](const Proposal& a, const Proposal& b) {
        return a.key != b.key ? a.key < b.key : a.cell_id < b.cell_id;
      });
    }
    for (Proposal& proposal : proposals) {
      if (!mesh.complex_.live(proposal.slot) || mesh.cell_id(proposal.slot) != proposal.cell_id) {
        continue;
      }
      if (pass > 0 && (is_ghost(mesh.tet(proposal.slot)) ||
                       !violates(proposal.slot, proposal, nullptr))) {
        continue;
      }
      if (const std::int32_t error = refine_cell(proposal); error != PHX_MC_OK) return error;
    }
  }

  restore_bank_order();
  // Cells left pending by the round cap were never proposed.
  for (std::int32_t slot : pending) {
    if (mesh.complex_.live(slot) && !is_ghost(mesh.tet(slot))) {
      proposed_ids.push_back(mesh.cell_id(slot));
    }
  }
  std::vector<std::int32_t> candidate_slots;
  for (std::uint64_t id : proposed_ids) {
    const std::int32_t slot = mesh.cell_slot(id);
    if (slot >= 0) candidate_slots.push_back(slot);
  }
  std::sort(candidate_slots.begin(), candidate_slots.end());
  candidate_slots.erase(std::unique(candidate_slots.begin(), candidate_slots.end()),
                        candidate_slots.end());
  std::vector<TetUnmetRecord> records;
  collect_unmet(mesh, radius_edge_bound, unmet_causes, candidate_slots, records, unmet_cells);
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
    const double length = mesh.metric_active()
                              ? metric_distance(mesh, first, second)
                              : std::sqrt(geometry::squared_distance(second, first));
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
