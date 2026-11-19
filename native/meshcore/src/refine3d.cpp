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

struct GridKey {
  std::array<std::int64_t, 3> cell{};
  bool operator==(const GridKey&) const noexcept = default;
};
struct GridHash {
  std::size_t operator()(const GridKey& key) const noexcept {
    std::uint64_t mixed = static_cast<std::uint64_t>(key.cell[0]) * 0x9e3779b97f4a7c15ULL ^
                          static_cast<std::uint64_t>(key.cell[1]) * 0xc2b2ae3d27d4eb4fULL ^
                          static_cast<std::uint64_t>(key.cell[2]) * 0x165667b19e3779f9ULL;
    mixed ^= mixed >> 29;
    return static_cast<std::size_t>(mixed);
  }
};

bool grid_cell(const double* point, double cell, GridKey& key) {
  for (int axis = 0; axis < 3; ++axis) {
    const double scaled = std::floor(point[axis] / cell);
    if (!(std::abs(scaled) < 0x1p62)) return false;
    key.cell[static_cast<std::size_t>(axis)] = static_cast<std::int64_t>(scaled);
  }
  return true;
}

// A uniform grid of items registered in the cells their boxes meet, built
// once and then queried: the (cell, item) pairs sorted by cell, the occupied
// cells in an open-addressing table over their ranges.
class CellGrid {
 public:
  bool add(double cell, const double* lower, const double* upper, std::int32_t item) {
    GridKey first;
    GridKey last;
    if (!grid_cell(lower, cell, first) || !grid_cell(upper, cell, last)) return false;
    for (std::int64_t x = first.cell[0]; x <= last.cell[0]; ++x) {
      for (std::int64_t y = first.cell[1]; y <= last.cell[1]; ++y) {
        for (std::int64_t z = first.cell[2]; z <= last.cell[2]; ++z) {
          entries_.push_back({GridKey{{x, y, z}}, item});
        }
      }
    }
    return true;
  }

  // Groups the registered pairs by cell (items in registration order).
  void finish() {
    std::stable_sort(entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) {
      return a.key.cell < b.key.cell;
    });
    keys_.clear();
    starts_.clear();
    items_.resize(entries_.size());
    for (std::size_t row = 0; row < entries_.size(); ++row) {
      if (row == 0 || !(entries_[row].key == entries_[row - 1].key)) {
        keys_.push_back(entries_[row].key);
        starts_.push_back(static_cast<std::uint32_t>(row));
      }
      items_[row] = entries_[row].item;
    }
    starts_.push_back(static_cast<std::uint32_t>(entries_.size()));
    std::vector<Entry>().swap(entries_);
    std::size_t size = 16;
    while (size < 2 * keys_.size()) size *= 2;
    table_.assign(size, -1);
    for (std::size_t row = 0; row < keys_.size(); ++row) {
      std::size_t probe = GridHash{}(keys_[row]) & (size - 1);
      while (table_[probe] >= 0) probe = (probe + 1) & (size - 1);
      table_[probe] = static_cast<std::int32_t>(row);
    }
  }

  bool empty() const noexcept { return keys_.empty(); }

  // The items of the cell `key` (empty when unoccupied).
  std::span<const std::int32_t> items(const GridKey& key) const noexcept {
    if (keys_.empty()) return {};
    const std::size_t mask = table_.size() - 1;
    for (std::size_t probe = GridHash{}(key) & mask; table_[probe] >= 0; probe = (probe + 1) & mask) {
      const std::size_t row = static_cast<std::size_t>(table_[probe]);
      if (keys_[row] == key) {
        return std::span<const std::int32_t>(items_).subspan(starts_[row], starts_[row + 1] - starts_[row]);
      }
    }
    return {};
  }

  // Calls `visit` on the items of every occupied cell meeting [lower, upper];
  // a box spanning more cells than are occupied reads every occupied cell.
  template <class Visit>
  bool visit_box(double cell, const double* lower, const double* upper, Visit&& visit) const {
    GridKey first;
    GridKey last;
    if (!grid_cell(lower, cell, first) || !grid_cell(upper, cell, last)) return false;
    double span = 1.0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      span *= static_cast<double>(last.cell[axis] - first.cell[axis]) + 1.0;
    }
    if (span > static_cast<double>(keys_.size())) {
      for (std::int32_t item : items_) visit(item);
      return true;
    }
    for (std::int64_t x = first.cell[0]; x <= last.cell[0]; ++x) {
      for (std::int64_t y = first.cell[1]; y <= last.cell[1]; ++y) {
        for (std::int64_t z = first.cell[2]; z <= last.cell[2]; ++z) {
          for (std::int32_t item : items(GridKey{{x, y, z}})) visit(item);
        }
      }
    }
    return true;
  }

 private:
  struct Entry {
    GridKey key;
    std::int32_t item;
  };
  std::vector<Entry> entries_;
  std::vector<GridKey> keys_;
  std::vector<std::uint32_t> starts_;
  std::vector<std::int32_t> items_;
  std::vector<std::int32_t> table_;
};

// The largest metric length of one binary64 coordinate step at `p`.
double coordinate_resolution(const TetMesh& mesh, const double* p) {
  double resolution = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    const double magnitude = std::abs(p[axis]);
    const double step = std::nextafter(magnitude, std::numeric_limits<double>::infinity()) -
                        magnitude;
    // The step is a power of two, so the transform of the step along an axis
    // is the step times the transform of that axis.
    std::array<double, 3> unit{};
    unit[static_cast<std::size_t>(axis)] = 1.0;
    resolution = std::max(resolution, step * geometry::length(mesh.metric_transform(unit)));
  }
  return resolution;
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
        const double length = metric_distance(mesh, mesh.point(a), mesh.point(b));
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
  // Metric images of the live vertices, feature boxes over them.
  std::vector<std::array<double, 3>> world(slot_count);
  for (std::size_t slot = 0; slot < slot_count; ++slot) {
    if (mesh.live_vertex(static_cast<std::int32_t>(slot))) {
      world[slot] = metric_point(mesh, mesh.point(static_cast<std::int32_t>(slot)));
    }
  }
  const auto feature_box = [&](std::span<const std::int32_t> corners, double* lower,
                               double* upper) {
    const auto& first = world[static_cast<std::size_t>(corners[0])];
    std::copy(first.begin(), first.end(), lower);
    std::copy(first.begin(), first.end(), upper);
    for (std::int32_t corner : corners.subspan(1)) {
      const auto& point = world[static_cast<std::size_t>(corner)];
      for (int axis = 0; axis < 3; ++axis) {
        lower[axis] = std::min(lower[axis], point[axis]);
        upper[axis] = std::max(upper[axis], point[axis]);
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

  CellGrid point_grid;
  CellGrid line_grid;
  CellGrid plane_grid;
  for (std::size_t slot = 0; slot < slot_count; ++slot) {
    const std::int32_t vertex = static_cast<std::int32_t>(slot);
    if (!mesh.live_vertex(vertex)) continue;
    if (!point_grid.add(cell, world[slot].data(), world[slot].data(), vertex)) return false;
  }
  // A feature whose box lies farther than lambda from v cannot lower lambda:
  // the box distance bounds the feature distance from below (a relative
  // slack of 1e-9 covers the rounding of either).
  std::vector<std::array<double, 6>> line_boxes(lines.size());
  std::vector<std::array<double, 6>> plane_boxes(planes.size());
  for (std::size_t row = 0; row < lines.size(); ++row) {
    double* box = line_boxes[row].data();
    feature_box(lines[row], box, box + 3);
    if (!line_grid.add(cell, box, box + 3, static_cast<std::int32_t>(row))) return false;
  }
  for (std::size_t row = 0; row < planes.size(); ++row) {
    double* box = plane_boxes[row].data();
    feature_box(planes[row], box, box + 3);
    if (!plane_grid.add(cell, box, box + 3, static_cast<std::int32_t>(row))) return false;
  }
  point_grid.finish();
  line_grid.finish();
  plane_grid.finish();
  const auto beyond = [](const std::array<double, 6>& box, const std::array<double, 3>& point,
                         double radius) {
    double squared = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double gap = std::max({0.0, box[static_cast<std::size_t>(axis)] - point[static_cast<std::size_t>(axis)],
                                   point[static_cast<std::size_t>(axis)] - box[static_cast<std::size_t>(axis) + 3]});
      squared += gap * gap;
    }
    return squared > radius * radius * (1.0 + 1e-9);
  };

  smallest = infinity;
  for (std::size_t slot = 0; slot < slot_count; ++slot) {
    const std::int32_t vertex = static_cast<std::int32_t>(slot);
    if (!mesh.live_vertex(vertex)) continue;
    const double* p = mesh.point(vertex);
    double lambda = sizes[slot];
    if (!std::isfinite(lambda)) return false;
    const auto& here = world[slot];
    const double lower[3]{here[0] - lambda, here[1] - lambda, here[2] - lambda};
    const double upper[3]{here[0] + lambda, here[1] + lambda, here[2] + lambda};
    const bool visited =
        point_grid.visit_box(cell, lower, upper, [&](std::int32_t other) {
          if (other != vertex) {
            lambda = std::min(lambda, metric_distance(mesh, p, mesh.point(other)));
          }
        }) &&
        line_grid.visit_box(cell, lower, upper, [&](std::int32_t row) {
          const auto& line = lines[static_cast<std::size_t>(row)];
          if (line[0] != vertex && line[1] != vertex &&
              !beyond(line_boxes[static_cast<std::size_t>(row)], here, lambda)) {
            lambda = std::min(lambda, metric_segment_distance(
                                          mesh, p, mesh.point(line[0]), mesh.point(line[1])));
          }
        }) &&
        plane_grid.visit_box(cell, lower, upper, [&](std::int32_t row) {
          const auto& plane = planes[static_cast<std::size_t>(row)];
          if (plane[0] != vertex && plane[1] != vertex && plane[2] != vertex &&
              !beyond(plane_boxes[static_cast<std::size_t>(row)], here, lambda)) {
            lambda = std::min(lambda, metric_triangle_distance(
                                          mesh, p, mesh.point(plane[0]), mesh.point(plane[1]),
                                          mesh.point(plane[2])));
          }
        });
    if (!visited || !std::isfinite(lambda)) return false;
    // Features nearer than the coordinate spacing at v cannot be separated by
    // a representable point; lambda is floored there, which keeps it positive.
    lambda = std::max(lambda, coordinate_resolution(mesh, p));
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

// The floating orientation determinant with the sign convention of
// orient3d (the negated Shewchuk determinant: positive for a positively
// oriented tetrahedron); no error bound (a positive value is necessary for a
// measurable cell, not sufficient for orientation).
double float_orientation(const double* a, const double* b, const double* c,
                         const double* d) noexcept {
  const double adx = a[0] - d[0], ady = a[1] - d[1], adz = a[2] - d[2];
  const double bdx = b[0] - d[0], bdy = b[1] - d[1], bdz = b[2] - d[2];
  const double cdx = c[0] - d[0], cdy = c[1] - d[1], cdz = c[2] - d[2];
  return -(adx * (bdy * cdz - bdz * cdy) + bdx * (cdy * adz - cdz * ady) +
           cdx * (ady * bdz - adz * bdy));
}

// orient3d through its first, floating stage inlined: the predicate's own
// expression and error bound (Shewchuk's stage A) decide the clear cases
// here, so only near-degenerate tests reach the exact predicate.
int orientation(const double* a, const double* b, const double* c, const double* d) noexcept {
  const double adx = a[0] - d[0], bdx = b[0] - d[0], cdx = c[0] - d[0];
  const double ady = a[1] - d[1], bdy = b[1] - d[1], cdy = c[1] - d[1];
  const double adz = a[2] - d[2], bdz = b[2] - d[2], cdz = c[2] - d[2];
  const double bdxcdy = bdx * cdy, cdxbdy = cdx * bdy;
  const double cdxady = cdx * ady, adxcdy = adx * cdy;
  const double adxbdy = adx * bdy, bdxady = bdx * ady;
  const double det = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady);
  const double permanent = (std::fabs(bdxcdy) + std::fabs(cdxbdy)) * std::fabs(adz) +
                           (std::fabs(cdxady) + std::fabs(adxcdy)) * std::fabs(bdz) +
                           (std::fabs(adxbdy) + std::fabs(bdxady)) * std::fabs(cdz);
  constexpr double kEpsilon = 0x1p-53;
  const double errbound = (7.0 + 56.0 * kEpsilon) * kEpsilon * permanent;
  // orient3d is the negated Shewchuk determinant.
  if (det > errbound) return -1;
  if (-det > errbound) return 1;
  return orient3d(a, b, c, d);
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
class InteriorCavity {
 public:
  enum class Status : std::uint8_t { kFound, kBlocked, kRefused };

  template <class IsEmbedded>
  Status build(const TetMesh& mesh, std::int32_t start, const double* point,
               bool has_embedded, IsEmbedded&& is_embedded) {
    const TetrahedralComplex& complex = mesh.complex();
    cells.clear();
    boundary.clear();
    vertices.clear();
    root = -1;
    blocked_cell = -1;
    blocked_face = -1;
    if (walked_.size() < complex.tets.size()) {
      walked_.resize(complex.tets.size(), 0U);
      in_.resize(complex.tets.size(), 0U);
    }
    const std::uint32_t walk = next_stamp(walked_, walk_stamp_);
    walk_.clear();
    std::int32_t current = start;
    for (;;) {
      if (walked_[static_cast<std::size_t>(current)] == walk) return Status::kRefused;
      walked_[static_cast<std::size_t>(current)] = walk;
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(current)];
      int exit = -1;
      for (int face = 0; face < 4 && exit < 0; ++face) {
        std::int32_t facet[3];
        oriented_facet(tet.v, face, facet);
        if (orientation(mesh.point(facet[0]), mesh.point(facet[1]), mesh.point(facet[2]), point) < 0) {
          exit = face;
        }
      }
      if (exit < 0) break;
      const std::int32_t next = tet.n[exit];
      if (complex.constraint(current, exit) != kNoConstraint || !complex.live(next) ||
          is_ghost(complex.tets[static_cast<std::size_t>(next)])) {
        blocked_cell = current;
        blocked_face = exit;
        return Status::kBlocked;
      }
      walk_.push_back(current);
      current = next;
    }
    root = current;
    const std::int32_t region = complex.region(root);
    std::uint32_t member = next_stamp(in_, in_stamp_);
    cells.push_back(root);
    in_[static_cast<std::size_t>(root)] = member;
    stack_.assign(1, root);
    for (std::int32_t cell : walk_) {
      if (complex.region(cell) != region || in_[static_cast<std::size_t>(cell)] == member) continue;
      in_[static_cast<std::size_t>(cell)] = member;
      cells.push_back(cell);
      stack_.push_back(cell);
    }
    while (!stack_.empty()) {
      const std::int32_t cell = stack_.back();
      stack_.pop_back();
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
      for (int face = 0; face < 4; ++face) {
        const std::int32_t next = tet.n[face];
        if (complex.constraint(cell, face) != kNoConstraint || !complex.live(next) ||
            in_[static_cast<std::size_t>(next)] == member) {
          continue;
        }
        const Tetrahedron& candidate = complex.tets[static_cast<std::size_t>(next)];
        if (is_ghost(candidate) || complex.region(next) != region) continue;
        if (mesh.metric_insphere(mesh.point(candidate.v[0]), mesh.point(candidate.v[1]),
                                 mesh.point(candidate.v[2]), mesh.point(candidate.v[3]),
                                 point) > 0) {
          in_[static_cast<std::size_t>(next)] = member;
          cells.push_back(next);
          stack_.push_back(next);
        }
      }
    }

    for (;;) {
      boundary.clear();
      vertices.clear();
      if (marks_.size() < mesh.vertex_slot_count()) marks_.resize(mesh.vertex_slot_count(), 0U);
      const std::uint32_t mark = next_stamp(marks_, mark_stamp_);
      std::int32_t drop = -1;
      for (std::int32_t cell : cells) {
        const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
        for (int face = 0; face < 4; ++face) {
          if (in_[static_cast<std::size_t>(tet.n[face])] == member) continue;
          const double* cone[4];
          for (int corner = 0; corner < 4; ++corner) {
            cone[corner] = corner == face ? point : mesh.point(tet.v[corner]);
          }
          if (!(float_orientation(cone[0], cone[1], cone[2], cone[3]) > 0.0) ||
              orientation(cone[0], cone[1], cone[2], cone[3]) <= 0) {
            drop = cell;
            break;
          }
          boundary.emplace_back(cell, face);
          for (int corner = 0; corner < 4; ++corner) {
            if (corner == face) continue;
            const std::int32_t vertex = tet.v[corner];
            if (marks_[static_cast<std::size_t>(vertex)] != mark) {
              marks_[static_cast<std::size_t>(vertex)] = mark;
              vertices.push_back(vertex);
            }
          }
        }
        if (drop >= 0) break;
      }
      if (drop < 0) {
        if (has_embedded) {
          kept_edges_.clear();
          for (const auto& [cell, face] : boundary) {
            std::int32_t facet[3];
            oriented_facet(complex.tets[static_cast<std::size_t>(cell)].v, face, facet);
            for (int corner = 0; corner < 3; ++corner) {
              kept_edges_.push_back(slot_edge(facet[corner], facet[(corner + 1) % 3]));
            }
          }
          std::sort(kept_edges_.begin(), kept_edges_.end());
        }
        for (std::int32_t cell : cells) {
          if (cell == root) continue;
          const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
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
      }
      if (drop < 0) return Status::kFound;
      if (drop == root) {
        cells.clear();
        boundary.clear();
        vertices.clear();
        return Status::kRefused;
      }
      // Keep the root's component of the cavity without the dropped cell.
      in_[static_cast<std::size_t>(drop)] = 0U;
      const std::uint32_t reached = next_stamp(in_, in_stamp_);
      if (reached < member) {
        // The stamp wrapped and cleared every membership (once in 2^32
        // cavities): refuse rather than rebuild.
        cells.clear();
        boundary.clear();
        vertices.clear();
        return Status::kRefused;
      }
      in_[static_cast<std::size_t>(root)] = reached;
      stack_.assign(1, root);
      cells.assign(1, root);
      while (!stack_.empty()) {
        const std::int32_t cell = stack_.back();
        stack_.pop_back();
        for (std::int32_t next : complex.tets[static_cast<std::size_t>(cell)].n) {
          if (next >= 0 && in_[static_cast<std::size_t>(next)] == member) {
            in_[static_cast<std::size_t>(next)] = reached;
            cells.push_back(next);
            stack_.push_back(next);
          }
        }
      }
      member = reached;
    }
  }

  std::vector<std::int32_t> cells;  // root first
  std::vector<std::pair<std::int32_t, int>> boundary;
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

  std::vector<std::uint32_t> walked_;
  std::uint32_t walk_stamp_ = 0;
  std::vector<std::uint32_t> in_;
  std::uint32_t in_stamp_ = 0;
  std::vector<std::uint32_t> marks_;
  std::uint32_t mark_stamp_ = 0;
  std::vector<std::int32_t> walk_;
  std::vector<std::int32_t> stack_;
  std::vector<std::uint64_t> kept_edges_;
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
double screen_cone(const TetMesh& mesh, std::span<const std::pair<std::int32_t, int>> boundary,
                   const double* point, double bound, double point_size, double* toward,
                   std::vector<std::uint8_t>& meets) {
  meets.assign(boundary.size(), 0U);
  double worst = std::numeric_limits<double>::infinity();
  for (std::size_t row = 0; row < boundary.size(); ++row) {
    const auto [cell, face] = boundary[row];
    const Tetrahedron& tet = mesh.tet(cell);
    const double* corners[4];
    std::array<double, 4> sizes{};
    for (int corner = 0; corner < 4; ++corner) {
      corners[corner] = corner == face ? point : mesh.point(tet.v[corner]);
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
    const double* a = mesh.point(from);
    const double* b = mesh.point(to);
    const auto world = mesh.metric_transform({b[0] - a[0], b[1] - a[1], b[2] - a[2]});
    std::copy(world.begin(), world.end(), out);
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
          smallest = std::min(smallest, ray_sector_angle(ray, sides[0], sides[1],
                                                         plane_normal(corner.group)));
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
          double* inward = normal[side == first ? 0 : 1];
          if (const double* plane = plane_normal(sides[side].second); plane != nullptr) {
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
      if (!grid_.add(cell_, lower, upper, static_cast<std::int32_t>(row))) return false;
    }
    grid_.finish();
    return true;
  }

  // The lowest-indexed feature whose protected region holds `point`, else -1.
  std::int32_t find(const double* point) const {
    GridKey key;
    const auto world = metric_point(mesh_, point);
    if (grid_.empty() || !grid_cell(world.data(), cell_, key)) return -1;
    std::int32_t protecting = -1;
    for (std::int32_t row : grid_.items(key)) {
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
  CellGrid grid_;
  double cell_ = 0.0;
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
  std::map<FaceKey, std::size_t> facet_rows_by_face;
  for (std::size_t row = 0; row < mesh.facets_.size(); ++row) {
    if (!facet_rows_by_face.emplace(sorted_face(mesh.facets_[row].vertices), row).second) {
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
  // seeded with its edge star, and its halves stay embedded.
  std::set<EdgeKey> embedded_segments;
  for (const TetSnapshotSegment& segment : mesh.segments_) {
    const std::int32_t first = mesh.vertex_slot(segment.vertices[0]);
    const std::int32_t second = mesh.vertex_slot(segment.vertices[1]);
    if (!mesh.live_vertex(first) || !mesh.live_vertex(second)) return PHX_MC_INTERNAL_ERROR;
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
  const auto feature_size_at = [&](const double* point, std::span<const std::int32_t> neighbors) {
    double size = std::numeric_limits<double>::infinity();
    for (std::int32_t vertex : neighbors) {
      size = std::min(size, feature_sizes[static_cast<std::size_t>(vertex)] +
                                metric_distance(mesh, point, mesh.point(vertex)));
    }
    return std::max(size, coordinate_resolution(mesh, point));
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
  // closer than `radius`.
  const auto clear_of = [&](const double* point, std::span<const std::int32_t> neighbors,
                            double radius) {
    if (!(radius > 0.0) || !std::isfinite(radius)) return false;
    for (std::int32_t vertex : neighbors) {
      if (metric_distance(mesh, point, mesh.point(vertex)) < radius) return false;
    }
    return true;
  };
  std::unordered_map<std::int32_t, std::array<double, 3>> plane_normals;
  for (const auto& [group, declared] : mesh.declared_groups_) {
    const double* plane = mesh.face_plane(declared.rows.front());
    std::array<double, 3> normal{};
    const auto u = mesh.metric_transform({plane[3], plane[4], plane[5]});
    const auto v = mesh.metric_transform({plane[6], plane[7], plane[8]});
    cross3(u.data(), v.data(), normal.data());
    plane_normals.emplace(group, normal);
  }
  std::vector<AcuteFeature> acute_features;
  if (!acute_input_features(mesh, incidence, mesh.facets_, mesh.segments_, feature_sizes,
                            plane_normals, acute_features)) {
    return PHX_MC_INVALID_INPUT;
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

  const bool boundary_splittable = mesh.policy_ != BoundaryPolicy::kFixed;
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
      if (radius > 0.0 &&
          !clear_of(point, cavity_vertices, insertion_guard(feature_size, radius, target))) {
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
  // A violating cell of a round, keyed by its circumcenter. Most proposals of
  // a round die to earlier insertions of the same round before their turn, so
  // a proposal holds only what orders it: its shape is measured (cell_shape)
  // when its turn comes, and its target is a function of its corners, which
  // holds while its ID is live.
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
  InteriorCavity interior;
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
  // meeting the criteria, which the round would skip.
  const auto commit_interior = [&](const double* point, double radius, double size,
                                   std::span<const std::uint8_t> meets,
                                   int& outcome) -> std::int32_t {
    outcome = 2;
    const double feature_size = feature_size_at(point, interior.vertices);
    if (!clear_of(point, interior.vertices, insertion_guard(feature_size, radius, size))) {
      return PHX_MC_OK;
    }
    removed_cells.clear();
    for (std::int32_t cell : interior.cells) {
      detail::LocalCell vertices{};
      std::copy_n(mesh.tet(cell).v, 4, vertices.begin());
      removed_cells.push_back(vertices);
    }
    const TetEditStatus status =
        mesh.insert_interior_vertex(point, size, interior.cells, interior.boundary, created_slots);
    const std::int32_t error = edit_error(status);
    if (error != PHX_MC_OK) return error;
    if (status != TetEditStatus::kApplied) return PHX_MC_OK;
    for (std::size_t row = 0; row < removed_cells.size(); ++row) {
      if (!incidence.erase(interior.cells[row], removed_cells[row])) return PHX_MC_INTERNAL_ERROR;
    }
    for (std::size_t row = 0; row < created_slots.size(); ++row) {
      const std::int32_t slot = created_slots[row];
      detail::LocalCell vertices{};
      std::copy_n(mesh.tet(slot).v, 4, vertices.begin());
      if (!incidence.insert_logical(mesh, slot, vertices)) return PHX_MC_INTERNAL_ERROR;
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
  // clears the spacing guard; the circumcenter's cavity is rebuilt when it
  // wins. Natural radius and size stay the circumcenter's. Slivers left by
  // refinement are improvement's work; most of them start here, as cone cells
  // that meet the radius-edge and size criteria and so are never refined.
  // Measured on the ledger shapes at their Gmsh HXT element counts: the
  // slivers below 15 degrees that refinement leaves drop by about two thirds,
  // for under 2% more vertices.
  constexpr double kSliverDegrees = 15.0;
  constexpr double kPickFraction = 0.3;
  constexpr int kPickAttempts = 3;
  const double sliver_sine = std::sin(kSliverDegrees * std::numbers::pi / 180.0);
  const double sliver_sine_squared = sliver_sine * sliver_sine;
  std::vector<std::uint8_t> cone_meets;
  std::vector<std::uint8_t> best_meets;
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
            return mesh.complex_.constraint(entry.first, entry.second) != kNoConstraint;
          })) {
        break;
      }
      const double feature_size = feature_size_at(candidate, interior.vertices);
      if (!clear_of(candidate, interior.vertices, insertion_guard(feature_size, radius, size))) {
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
    if (!built_best && interior.build(mesh, start, best_point, false, is_embedded) !=
                           InteriorCavity::Status::kFound) {
      return PHX_MC_INTERNAL_ERROR;
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
      for (const auto& [cell, face] : interior.boundary) {
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
  constexpr int kMaxRounds = 4096;
  for (int pass = 0; pass < kMaxRounds && !pending.empty(); ++pass) {
    proposals.clear();
    for (std::int32_t slot : pending) {
      if (!mesh.complex_.live(slot) || is_ghost(mesh.tet(slot))) continue;
      const Tetrahedron& tet = mesh.tet(slot);
      Proposal proposal{};
      proposal.has_target = cell_target(mesh, tet.v, proposal.target);
      // Only the first round is sorted (below), by circumcenter.
      double center[3];
      const CellScreen screen =
          mesh.metric_active()
              ? CellScreen::kUncertain
              : screen_cell(mesh.point(tet.v[0]), mesh.point(tet.v[1]), mesh.point(tet.v[2]),
                            mesh.point(tet.v[3]), radius_edge_bound,
                            proposal.has_target ? proposal.target
                                                : std::numeric_limits<double>::infinity(),
                            pass == 0 ? center : nullptr);
      if (screen == CellScreen::kMeets) continue;
      if (screen == CellScreen::kUncertain) {
        geometry::TetrahedronShape shape;
        const bool measured = cell_shape(mesh, tet, shape);
        if (measured &&
            !(radius_edge_bound > 0.0 && shape.radius_edge_ratio > radius_edge_bound) &&
            !(proposal.has_target && shape.circumradius > proposal.target)) {
          continue;
        }
        std::copy_n(shape.circumcenter.begin(), 3, center);
      }
      if (pass == 0) proposal.key = frame.key(center);
      proposal.cell_id = mesh.cell_id(slot);
      proposal.slot = slot;
      proposals.push_back(proposal);
      proposed_ids.push_back(proposal.cell_id);
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
    for (const Proposal& proposal : proposals) {
      if (!mesh.complex_.live(proposal.slot) || mesh.cell_id(proposal.slot) != proposal.cell_id) {
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
