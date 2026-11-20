//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Restricted Delaunay meshing of an implicit solid, and outward bounds of
// exact tetrahedron circumcenters.
//
// phx_mc_implicit_restricted_delaunay meshes {f <= 0} surface first, in one
// 3D Delaunay triangulation (Boissonnat and Oudot, "Provably good sampling
// and meshing of surfaces", 2005; Rineau and Yvinec, "Meshing 3D domains
// bounded by piecewise smooth surfaces", 2007):
//
//   0. seeds: the possible-zero box tree of f (box_tree.hpp) down to the
//      facet edge bound; every edge of a kept box whose corner values have
//      strictly opposite signs carries a root (binary64 corner values,
//      regula falsi on the edge). A component of the
//      zero set crossing a kept box's edges therefore has a root; roots
//      closer than a fraction of the edge bound to an earlier one are left
//      to the refinement;
//   1. the seeds and eight far corners are triangulated. A cell is inside
//      when f < 0 at its circumcenter; a facet between an inside and an
//      outside cell is restricted, and its dual Voronoi edge (circumcenter
//      to circumcenter, or a ray toward a ghost) crosses the zero set at the
//      center of its surface Delaunay ball (regula falsi on the edge);
//   2. restricted facets with a long edge, a far ball center or a small
//      angle are refined at their ball center, largest ball first; then
//      inside cells with a large radius-edge ratio or circumradius at their
//      circumcenter, unless its conflict region holds a restricted facet,
//      which is refined instead; no point is inserted closer to a vertex
//      than a separation bound, which bounds the vertex count by packing
//      (RestrictedMesher);
//   3. the restricted surface (the boundary of the inside cells) is checked
//      for edge-manifoldness, its topology and coverage of every box root;
//      failures insert box roots or ball centers there and refine again;
//   4. with the boundary vertices moved onto the zero set, slivers are
//      removed by flips and relocation; the circumcenters of those left are
//      inserted and steps 2-4 repeated, and cells still missing the
//      requested shapes go through the fixed-boundary quality improvement.
//
// Construction only: floating-point evaluation proposes; the volume, source
// fidelity and topology certificates run on the result afterwards.
#include "restricted_delaunay.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <numeric>
#include <queue>
#include <span>
#include <vector>

#include "box_tree.hpp"
#include "capi_guard.hpp"
#include "expansion.hpp"
#include "floating_point_mode.hpp"
#include "geometry3.hpp"
#include "improve3d.hpp"
#include "interval_program.hpp"
#include "key_map.hpp"
#include "mesh.hpp"
#include "predicates.hpp"
#include "refine3d.hpp"
#include "spatial_sort.hpp"
#include "tet_mesh.hpp"
#include "triangulation3d.hpp"

// The constructed restricted volume: points, positive tetrahedra, outward
// boundary triangles and the construction report.
struct phx_mc_restricted_volume {
  std::vector<double> points;
  std::vector<int32_t> cells, faces;
  std::array<int64_t, PHX_MC_RESTRICTED_REPORT_SIZE> report{};
};

namespace phx::mc {
namespace {
using namespace restricted_layout;
using ExactVector = std::array<Expansion, kDimensions>;

ExactVector exact_cross(const ExactVector& a, const ExactVector& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}

Expansion exact_dot(const ExactVector& a, const ExactVector& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<double, 2> expansion_bounds(const Expansion& value) {
  double lower = 0.0;
  double upper = 0.0;
  for (int index = 0; index < value.size(); ++index) {
    lower = geometry::next_down(lower + value.data()[index]);
    upper = geometry::next_up(upper + value.data()[index]);
  }
  return {lower, upper};
}

bool exact_center_bounds(const double* const* p, std::array<double, kCenterBounds>& bounds) {
  ExactVector u, v, w;
  for (int axis = 0; axis < kDimensions; ++axis) {
    u[axis] = Expansion::difference(p[1][axis], p[0][axis]);
    v[axis] = Expansion::difference(p[2][axis], p[0][axis]);
    w[axis] = Expansion::difference(p[3][axis], p[0][axis]);
  }
  const ExactVector vw = exact_cross(v, w);
  const ExactVector wu = exact_cross(w, u);
  const ExactVector uv = exact_cross(u, v);
  const Expansion denominator = exact_dot(u, vw).scaled(2.0);
  const auto d = expansion_bounds(denominator);
  if (!(d[0] > 0.0) || !std::isfinite(d[1])) return false;
  const Expansion uu = exact_dot(u, u);
  const Expansion vv = exact_dot(v, v);
  const Expansion ww = exact_dot(w, w);
  for (int axis = 0; axis < kDimensions; ++axis) {
    const Expansion numerator = uu * vw[axis] + vv * wu[axis] + ww * uv[axis] +
                                denominator.scaled(p[0][axis]);
    const auto n = expansion_bounds(numerator);
    const double quotients[4] = {n[0] / d[0], n[0] / d[1], n[1] / d[0], n[1] / d[1]};
    bounds[axis] = geometry::next_down(*std::min_element(quotients, quotients + 4));
    bounds[axis + kUpper] = geometry::next_up(*std::max_element(quotients, quotients + 4));
    if (!std::isfinite(bounds[axis]) || !std::isfinite(bounds[axis + kUpper])) return false;
  }
  return true;
}

int32_t validate_tets(int64_t point_count, const double* points, int64_t tet_count,
                      const int32_t* tets) {
  if (point_count > kMaxMeshPoints ||
      tet_count > std::numeric_limits<int32_t>::max()) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  if (point_count < kTetVertices || tet_count < 1 || points == nullptr || tets == nullptr ||
      !addressable(point_count, kDimensions, sizeof(double)) ||
      !addressable(tet_count, kTetVertices, sizeof(int32_t))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int32_t valid = validate_points(points, point_count, kDimensions, nullptr);
  if (valid != PHX_MC_OK) return valid;
  for (int64_t cell = 0; cell < tet_count; ++cell) {
    const int32_t* ids = tets + kTetVertices * cell;
    for (int vertex = 0; vertex < kTetVertices; ++vertex) {
      if (ids[vertex] < 0 || ids[vertex] >= point_count ||
          std::count(ids, ids + kTetVertices, ids[vertex]) != 1) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    if (orient3d(points + kDimensions * ids[0], points + kDimensions * ids[1],
                 points + kDimensions * ids[2], points + kDimensions * ids[3]) <= 0) {
      return PHX_MC_DEGENERATE_INPUT;
    }
  }
  return PHX_MC_OK;
}

int32_t construct_centers(int64_t point_count, const double* points, int64_t tet_count,
                          const int32_t* tets, double* bounds, int32_t* item_status) {
  if (item_status == nullptr || bounds == nullptr ||
      !addressable(tet_count, kCenterBounds, sizeof(double))) return PHX_MC_INVALID_ARGUMENT;
  const int32_t valid = validate_tets(point_count, points, tet_count, tets);
  if (valid != PHX_MC_OK) return valid;
  std::vector<double> staged(static_cast<std::size_t>(tet_count) * kCenterBounds);
  std::vector<int32_t> statuses(static_cast<std::size_t>(tet_count), PHX_MC_OK);
  for (int64_t cell = 0; cell < tet_count; ++cell) {
    const int32_t* ids = tets + kTetVertices * cell;
    const double* p[kTetVertices] = {
        points + kDimensions * ids[0], points + kDimensions * ids[1],
        points + kDimensions * ids[2], points + kDimensions * ids[3]};
    std::array<double, kCenterBounds> value{};
    if (!exact_center_bounds(p, value)) {
      statuses[static_cast<std::size_t>(cell)] = PHX_MC_DEGENERATE_INPUT;
      std::fill(value.begin(), value.begin() + kUpper, -std::numeric_limits<double>::infinity());
      std::fill(value.begin() + kUpper, value.end(), std::numeric_limits<double>::infinity());
    }
    std::copy(value.begin(), value.end(), staged.begin() + kCenterBounds * cell);
  }
  std::copy(staged.begin(), staged.end(), bounds);
  std::copy(statuses.begin(), statuses.end(), item_status);
  return PHX_MC_OK;
}

// ---------------------------------------------------------------- meshing

using Point = std::array<double, 3>;

constexpr int kCorners = 8;
constexpr int kRootIterations = 60;
// Regula falsi stops when its step or bracket is below these fractions of
// the segment.
constexpr double kRootStep = 0x1p-32;
constexpr double kRootWidth = 0x1p-44;
// Construction roots then move along their segment by an offset of at most
// this fraction of it (CGAL's implicit oracle stops at 1e-6). Without it,
// points on a zero set that is a sphere or a plane are cospherical or
// coplanar to rounding, so the Delaunay predicates on them fall through to
// exact arithmetic (three times the construction time on the ledger
// sphere). The offset is a hash of the bracket's endpoint bits, a function
// of the input alone (no addresses, clocks or random state), so equal
// inputs give equal meshes. The restricted surface's vertices are moved
// onto the zero set to float accuracy after the refinement.
constexpr double kRootOffset = 0x1p-24;
constexpr int kNewtonSteps = 8;
constexpr int kPolishSteps = 3;
constexpr int kRepairRounds = 4;
// Outward faces of a positive tetrahedron: face k omits vertex k.
constexpr int kOutwardFace[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
constexpr int kEdges[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

Point minus(const double* a, const double* b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double dot(const Point& a, const Point& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Point cross(const Point& a, const Point& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Point along(const Point& a, const Point& b, double t) {
  return {a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])};
}

// f at points in binary64: proposals that bound nothing.
class Field {
 public:
  explicit Field(const ProgramView& view) : view_(view) {}
  const ProgramView& view() const { return view_; }
  void values(std::span<const double> points, double* result) {
    const auto count = static_cast<int64_t>(points.size() / 3);
    if (count > 0) evaluate_point(view_, points.data(), count, result, workspace_);
  }
  // Value, gradient and Hessian (jet_components(3) entries per point).
  void jets(std::span<const double> points, std::vector<double>& result) {
    const auto count = static_cast<int64_t>(points.size() / 3);
    result.resize(static_cast<std::size_t>(count * jet_components(3)));
    if (count > 0) evaluate_jet(view_, points.data(), count, result.data(), workspace_);
  }

 private:
  ProgramView view_;
  PointWorkspace workspace_;
};

// Illinois regula falsi on every bracket at once, one batched evaluation per
// step, leaving each bracket's root where it stopped; a bracket leaves the
// batch when f vanishes at its trial point, its step is negligible or its
// parameter interval closes.
class RootSolver {
 public:
  void solve(Field& field, std::span<Bracket> brackets) {
    const std::size_t n = brackets.size();
    low_.assign(n, 0.0);
    high_.assign(n, 1.0);
    t_.assign(n, 0.5);
    previous_.assign(n, -1.0);
    side_.assign(n, 0);
    f_low_.resize(n);
    f_high_.resize(n);
    active_.resize(n);
    for (std::size_t c = 0; c < n; ++c) {
      f_low_[c] = brackets[c].f_negative;
      f_high_[c] = brackets[c].f_positive;
      active_[c] = c;
    }
    for (int iteration = 0; iteration < kRootIterations && !active_.empty(); ++iteration) {
      trial_.resize(3 * active_.size());
      for (std::size_t k = 0; k < active_.size(); ++k) {
        const std::size_t c = active_[k];
        double t = low_[c] - f_low_[c] * (high_[c] - low_[c]) / (f_high_[c] - f_low_[c]);
        if (!(t > low_[c] && t < high_[c])) t = 0.5 * (low_[c] + high_[c]);
        t_[c] = t;
        const Point p = along(brackets[c].negative, brackets[c].positive, t);
        std::copy(p.begin(), p.end(), trial_.begin() + static_cast<std::ptrdiff_t>(3 * k));
      }
      values_.resize(active_.size());
      field.values(trial_, values_.data());
      std::size_t kept = 0;
      for (std::size_t k = 0; k < active_.size(); ++k) {
        const std::size_t c = active_[k];
        const double value = values_[k];
        const double step = std::abs(t_[c] - previous_[c]);
        previous_[c] = t_[c];
        if (value == 0.0 || step <= kRootStep || high_[c] - low_[c] <= kRootWidth) continue;
        if (value < 0.0) {
          low_[c] = t_[c];
          f_low_[c] = value;
          if (side_[c] == -1) f_high_[c] *= 0.5;
          side_[c] = -1;
        } else {
          high_[c] = t_[c];
          f_high_[c] = value;
          if (side_[c] == 1) f_low_[c] *= 0.5;
          side_[c] = 1;
        }
        active_[kept++] = c;
      }
      active_.resize(kept);
    }
    for (std::size_t c = 0; c < n; ++c) {
      const Bracket& b = brackets[c];
      std::uint64_t hash = 0;
      for (std::size_t d = 0; d < 3; ++d)
        hash = splitmix64(hash ^ std::bit_cast<std::uint64_t>(b.negative[d]) ^
                          std::rotl(std::bit_cast<std::uint64_t>(b.positive[d]), 17));
      // A uniform offset in [-1, 1) times kRootOffset, hashed from the
      // bracket's endpoints only (kRootOffset).
      const double unit = static_cast<double>(hash >> 11) * 0x1p-52 - 1.0;
      const double t = std::clamp(t_[c] + kRootOffset * unit, 0.0, 1.0);
      brackets[c].root = along(b.negative, b.positive, t);
    }
  }

 private:
  std::vector<double> low_, high_, t_, previous_, f_low_, f_high_, trial_, values_;
  std::vector<int> side_;
  std::vector<std::size_t> active_;
};

// Points by uniform hash cells, for nearest-neighbor-within-radius queries.
class PointGrid {
 public:
  PointGrid(double cell, std::size_t expected) : cell_(cell), heads_(expected) {}
  void add(const Point& p) {
    int64_t& head = heads_.slot(key_of(cell_of(p)));
    next_.push_back(static_cast<int32_t>(head));
    head = static_cast<int64_t>(points_.size());
    points_.push_back(p);
  }
  // Whether some point lies within `radius` (at most the cell size) of p.
  bool near(const Point& p, double radius) const {
    const std::array<int64_t, 3> c = cell_of(p);
    const double r2 = radius * radius;
    for (int64_t i = -1; i <= 1; ++i)
      for (int64_t j = -1; j <= 1; ++j)
        for (int64_t k = -1; k <= 1; ++k)
          for (int64_t e = heads_.find(key_of({c[0] + i, c[1] + j, c[2] + k})); e >= 0;
               e = next_[static_cast<std::size_t>(e)]) {
            const Point& q = points_[static_cast<std::size_t>(e)];
            const Point d{q[0] - p[0], q[1] - p[1], q[2] - p[2]};
            if (dot(d, d) <= r2) return true;
          }
    return false;
  }

 private:
  std::array<int64_t, 3> cell_of(const Point& p) const {
    return {static_cast<int64_t>(std::floor(p[0] / cell_)), static_cast<int64_t>(std::floor(p[1] / cell_)),
            static_cast<int64_t>(std::floor(p[2] / cell_))};
  }
  // 21 bits per axis: the grid spans far fewer cells than 2^20 per axis.
  static std::uint64_t key_of(const std::array<int64_t, 3>& c) {
    constexpr int64_t kMask = (int64_t{1} << 21) - 1;
    return (static_cast<std::uint64_t>(c[0] & kMask) << 42) |
           (static_cast<std::uint64_t>(c[1] & kMask) << 21) | static_cast<std::uint64_t>(c[2] & kMask);
  }
  double cell_;
  KeyMap heads_;
  std::vector<int32_t> next_;
  std::vector<Point> points_;
};

struct Criteria {
  double edge2;        // longest restricted facet edge, squared
  double distance2;    // facet circumcenter to ball center, squared
  double angle_sine2;  // smallest facet angle's sine, squared
  double size2;        // inside cell circumradius, squared
  double ratio2;       // inside cell radius-edge ratio, squared
};

struct CellData {
  Point center{};
  double radius2 = 0.0;
  double value = 0.0;
  std::uint32_t generation = 0;
  bool inside = false;
};

struct FacetEntry {
  double key;
  int32_t cell, slot, other;
  std::uint32_t cell_generation, other_generation;
  Point center;
  bool operator<(const FacetEntry& other_entry) const { return key < other_entry.key; }
};

struct CellEntry {
  double key;
  int32_t cell;
  std::uint32_t generation;
  bool operator<(const CellEntry& other) const { return key < other.key; }
};

// The surface topology of outward triangles: whether every directed edge
// occurs once with its reverse, the edge-connected components, the Euler
// characteristic, and the vertices of non-manifold edges or pinched links.
struct SurfaceTopology {
  bool manifold = true;
  int64_t components = 0;
  int64_t euler = 0;
  std::vector<int32_t> singular;
};

int32_t find_root(std::vector<int32_t>& parent, int32_t x) {
  while (parent[static_cast<std::size_t>(x)] != x) {
    parent[static_cast<std::size_t>(x)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
    x = parent[static_cast<std::size_t>(x)];
  }
  return x;
}

void unite(std::vector<int32_t>& parent, int32_t a, int32_t b) {
  a = find_root(parent, a);
  b = find_root(parent, b);
  if (a != b) parent[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b);
}

SurfaceTopology surface_topology(std::span<const int32_t> faces) {
  SurfaceTopology result;
  const std::size_t face_count = faces.size() / 3;
  const auto head_of = [&](std::size_t corner) { return faces[corner - corner % 3 + (corner % 3 + 1) % 3]; };
  // The corners (directed edges by their tail corner) of each tail vertex,
  // in corner order.
  std::size_t vertex_count = 0;
  for (int32_t v : faces) vertex_count = std::max(vertex_count, static_cast<std::size_t>(v) + 1);
  std::vector<int32_t> offsets(vertex_count + 1, 0), cursor(vertex_count), tails(faces.size());
  for (int32_t v : faces) ++offsets[static_cast<std::size_t>(v) + 1];
  for (std::size_t v = 0; v < vertex_count; ++v) {
    offsets[v + 1] += offsets[v];
    cursor[v] = offsets[v];
  }
  for (std::size_t c = 0; c < faces.size(); ++c)
    tails[static_cast<std::size_t>(cursor[static_cast<std::size_t>(faces[c])]++)] = static_cast<int32_t>(c);
  std::vector<int32_t> face_parent(face_count), corner_parent(faces.size());
  std::iota(face_parent.begin(), face_parent.end(), 0);
  std::iota(corner_parent.begin(), corner_parent.end(), 0);
  const auto mark_singular = [&](int32_t tail, int32_t head) {
    result.manifold = false;
    result.singular.push_back(tail);
    result.singular.push_back(head);
  };
  for (std::size_t corner = 0; corner < faces.size(); ++corner) {
    const int32_t tail = faces[corner], head = head_of(corner);
    const auto first = static_cast<std::size_t>(offsets[static_cast<std::size_t>(tail)]);
    const auto last = static_cast<std::size_t>(offsets[static_cast<std::size_t>(tail) + 1]);
    bool repeated = false;
    for (std::size_t k = first; k < last && !repeated; ++k) {
      const auto other = static_cast<std::size_t>(tails[k]);
      repeated = other != corner && head_of(other) == head;
    }
    if (repeated) {
      mark_singular(tail, head);
      continue;
    }
    // The reverse edge's first corner; when the reverse repeats, its own
    // corners mark both vertices singular, so the choice changes nothing
    // the repair or the topology match reads.
    int32_t other = -1;
    const auto reverse_first = static_cast<std::size_t>(offsets[static_cast<std::size_t>(head)]);
    const auto reverse_last = static_cast<std::size_t>(offsets[static_cast<std::size_t>(head) + 1]);
    for (std::size_t k = reverse_first; k < reverse_last; ++k)
      if (head_of(static_cast<std::size_t>(tails[k])) == tail) {
        other = tails[k];
        break;
      }
    if (other < 0) {
      mark_singular(tail, head);
      continue;
    }
    // Around the tail a, the corner of this face at a follows the corner at
    // a of the reverse edge's face, where a is the reverse edge's head.
    unite(face_parent, static_cast<int32_t>(corner / 3), other / 3);
    const int32_t other_head = 3 * (other / 3) + (other % 3 + 1) % 3;
    unite(corner_parent, static_cast<int32_t>(corner), other_head);
  }
  for (std::size_t f = 0; f < face_count; ++f)
    result.components += find_root(face_parent, static_cast<int32_t>(f)) == static_cast<int32_t>(f);
  // Fans per vertex: one corner component each on a manifold surface.
  std::vector<int32_t> fans(vertex_count, 0);
  int64_t vertices = 0;
  for (std::size_t c = 0; c < faces.size(); ++c) {
    if (find_root(corner_parent, static_cast<int32_t>(c)) != static_cast<int32_t>(c)) continue;
    if (fans[static_cast<std::size_t>(faces[c])]++ == 0) {
      ++vertices;
    } else {
      result.manifold = false;
      result.singular.push_back(faces[c]);
    }
  }
  result.euler = vertices - static_cast<int64_t>(faces.size() / 2) + static_cast<int64_t>(face_count);
  std::sort(result.singular.begin(), result.singular.end());
  result.singular.erase(std::unique(result.singular.begin(), result.singular.end()),
                        result.singular.end());
  return result;
}

// One Delaunay triangulation of the seeds, the refinement points and eight
// far corners (vertices 0 to 7), with the circumcenter, its f value and the
// inside label of every cell slot.
//
// Termination is a packing argument (Ruppert; Shewchuk). The seeds are
// pairwise at least `separation` apart, and a point is inserted only inside
// the far corners' box and at least `separation` from every vertex present:
// in a Delaunay triangulation its nearest vertex is a vertex of its conflict
// zone, which the cell refinement walks anyway, and for a facet's surface
// Delaunay ball center it is a vertex of the facet, the ball being empty.
// Balls of radius separation / 2 around the inserted points are disjoint
// and lie in the far box grown by separation / 2, so their count is at
// most that volume over a ball's, whatever the field. A facet whose
// refinement point the guard refuses is recorded, and still violating the
// criteria at the end, makes the construction PHX_MC_UNRESOLVED; a cell is
// left to the quality stage. When the cavity of an admitted point, which
// holds its nearest vertex, shows a vertex closer than the bound (rounding
// in a binary64 circumcenter or ball center), the refinement stops at once
// with PHX_MC_UNRESOLVED. The guard never binds in a refinement that meets
// its criteria: a facet with an edge above the bound e or a ball center
// farther than d from its circumcenter has a ball radius above e / 2 or d,
// a small-angle facet one above its shortest edge, and a cell above its
// size bound or B times its shortest edge (radius-edge bound B), so
// insertion radii stay above min(separation, e / 2, d, size) (Chew;
// Boissonnat and Oudot). It binds where a facet keeps a vertex off the zero
// set, or binary64 circumcenters lose their meaning.
class RestrictedMesher {
 public:
  RestrictedMesher(Field& field, const double* lower, const double* upper, const Criteria& criteria,
                   double separation)
      : field_(field), criteria_(criteria), separation2_(separation * separation),
        triangulation_(nullptr, nullptr, 0) {
    double largest = 0.0;
    for (int d = 0; d < 3; ++d) largest = std::max(largest, upper[d] - lower[d]);
    for (int c = 0; c < kCorners; ++c)
      for (int d = 0; d < 3; ++d)
        points_.push_back((c >> d) & 1 ? upper[d] + largest : lower[d] - largest);
    for (std::size_t d = 0; d < 3; ++d) {
      far_lower_[d] = lower[d] - largest;
      far_upper_[d] = upper[d] + largest;
    }
    far_length_ = 8.0 * largest;
    triangulation_.complex().journal = &journal_;
  }
  RestrictedMesher(const RestrictedMesher&) = delete;
  RestrictedMesher& operator=(const RestrictedMesher&) = delete;

  Criteria& criteria() { return criteria_; }
  int64_t vertex_count() const { return static_cast<int64_t>(points_.size() / 3); }
  const double* point(int32_t v) const { return points_.data() + 3 * static_cast<std::size_t>(v); }

  int32_t start(std::span<const Point> seeds) {
    for (const Point& p : seeds) points_.insert(points_.end(), p.begin(), p.end());
    const auto count = static_cast<int32_t>(points_.size() / 3);
    alive_.assign(static_cast<std::size_t>(count), 1);
    on_surface_.assign(static_cast<std::size_t>(count), 1);
    std::fill_n(on_surface_.begin(), kCorners, 0);
    rebind();
    std::vector<int32_t> candidates(static_cast<std::size_t>(count - kCorners));
    std::iota(candidates.begin(), candidates.end(), kCorners);
    std::vector<int32_t> order(kCorners);
    std::iota(order.begin(), order.end(), 0);
    if (!candidates.empty()) {
      const std::vector<int32_t> sorted = brio_hilbert_order(points_.data(), 3, candidates);
      order.insert(order.end(), sorted.begin(), sorted.end());
    }
    const int32_t status = triangulation_.build(order, alive_);
    if (status != PHX_MC_OK) return status;
    journal_.clear();
    std::vector<int32_t> cells;
    for (std::size_t t = 0; t < tets().size(); ++t)
      if (complex().live(static_cast<int32_t>(t))) cells.push_back(static_cast<int32_t>(t));
    update(cells);
    return PHX_MC_OK;
  }

  // Inserts points of the zero set, except those within the separation
  // bound of a vertex (already covered at that distance).
  int32_t insert_all(std::span<const Point> points) {
    for (const Point& p : points) {
      if (!in_far_box(p)) continue;
      const int32_t start = triangulation_.locate(p.data());
      if (start < 0 || is_ghost(tets()[static_cast<std::size_t>(start)])) continue;
      if (conflict_zone(start, p) < separation2_) continue;
      const int32_t status = insert(p, true);
      if (status != PHX_MC_OK) return status;
    }
    return PHX_MC_OK;
  }

  // Inserts each point inside the solid, or the ball center of a restricted
  // facet its insertion would remove, unless the separation bound refuses
  // it.
  int32_t refine_at(std::span<const Point> points) {
    std::vector<double> coordinates;
    for (const Point& p : points) coordinates.insert(coordinates.end(), p.begin(), p.end());
    std::vector<double> values(points.size());
    field_.values(coordinates, values.data());
    for (std::size_t k = 0; k < points.size(); ++k) {
      const Point& p = points[k];
      if (!(values[k] < 0.0) || !in_far_box(p)) continue;
      const int32_t start = triangulation_.locate(p.data());
      if (start < 0 || is_ghost(tets()[static_cast<std::size_t>(start)])) continue;
      bool surface = false;
      double radius2 = 0.0;
      const Point chosen = target(start, p, surface, radius2);
      if (radius2 < separation2_ || !in_far_box(chosen)) continue;
      const int32_t status = insert(chosen, surface);
      if (status != PHX_MC_OK) return status;
    }
    return PHX_MC_OK;
  }

  // Every live cell's faces and shape against the (changed) criteria.
  void reexamine() {
    std::vector<int32_t> cells;
    for (std::size_t t = 0; t < tets().size(); ++t)
      if (complex().live(static_cast<int32_t>(t))) cells.push_back(static_cast<int32_t>(t));
    examine(cells);
  }

  // Refines until no facet or cell violates the criteria except those the
  // separation bound refused; PHX_MC_UNRESOLVED when an admitted point's
  // cavity breaks the bound.
  int32_t refine(int64_t& insertions) {
    for (;;) {
      Point target;
      bool surface = false;
      double radius2 = 0.0;
      // Largest circumball first, facet or cell: on a zero set that is a
      // sphere the seeds alone are cospherical, so every inside cell's
      // circumball is the sphere and each surface insertion would rebuild
      // all of them; the center insertions the large cells ask for first
      // keep the cavities local.
      const bool facet_first = !facets_.empty() &&
                               (cells_.empty() || facets_.top().key >= cells_.top().key);
      if (facet_first) {
        const FacetEntry entry = facets_.top();
        facets_.pop();
        if (!current(entry)) continue;
        if (entry.key < separation2_ || !in_far_box(entry.center)) {
          refused_.push_back(entry);
          continue;
        }
        target = entry.center;
        surface = true;
      } else if (!cells_.empty()) {
        const CellEntry entry = cells_.top();
        cells_.pop();
        if (!complex().live(entry.cell) || data(entry.cell).generation != entry.generation) continue;
        target = cell_target(entry.cell, surface, radius2);
        if (radius2 < separation2_ || !in_far_box(target)) continue;
      } else {
        return PHX_MC_OK;
      }
      const int32_t status = insert(target, surface);
      if (status != PHX_MC_OK) return status;
      ++insertions;
    }
  }

  // The refused facets still in the triangulation, each once.
  int64_t refused_facets() const {
    std::vector<std::array<int32_t, 2>> facets;
    for (const FacetEntry& entry : refused_)
      if (current(entry)) facets.push_back({entry.cell, entry.slot});
    std::sort(facets.begin(), facets.end());
    return std::unique(facets.begin(), facets.end()) - facets.begin();
  }

  // The inside cells and their outward boundary faces, with the inside cell
  // and slot of each face.
  void surface(std::vector<int32_t>& cells, std::vector<int32_t>& faces,
               std::vector<std::array<int32_t, 2>>& facets) const {
    cells.clear();
    faces.clear();
    facets.clear();
    for (std::size_t t = 0; t < tets().size(); ++t) {
      const auto id = static_cast<int32_t>(t);
      if (!complex().live(id) || !data(id).inside) continue;
      const Tetrahedron& cell = tets()[t];
      cells.insert(cells.end(), cell.v, cell.v + 4);
      for (int k = 0; k < 4; ++k) {
        if (data(cell.n[k]).inside) continue;
        for (int r = 0; r < 3; ++r) faces.push_back(cell.v[kOutwardFace[k][r]]);
        facets.push_back({id, k});
      }
    }
  }

  // Ball centers of the given restricted facets (inside cell, slot).
  void ball_centers(std::span<const std::array<int32_t, 2>> facets, std::vector<Point>& centers) {
    brackets_.clear();
    for (const auto& facet : facets) brackets_.push_back(dual_bracket(facet[0], facet[1]));
    solve_rays();
    solver_.solve(field_, brackets_);
    centers.clear();
    for (const Bracket& b : brackets_) centers.push_back(b.root);
  }

 private:
  std::vector<Tetrahedron>& tets() { return complex().tets; }
  const std::vector<Tetrahedron>& tets() const { return triangulation_.complex().tets; }
  TetrahedralComplex& complex() { return triangulation_.complex(); }
  const TetrahedralComplex& complex() const { return triangulation_.complex(); }
  const CellData& data(int32_t t) const { return data_[static_cast<std::size_t>(t)]; }

  void rebind() {
    const int64_t count = vertex_count();
    triangulation_.rebind(points_.data(), count);
    triangulation_.reserve_vertices(count);
  }

  // Inserts p, a point of the zero set when `surface`; PHX_MC_UNRESOLVED
  // when its cavity, which holds its nearest vertex, shows a vertex within
  // the separation bound.
  int32_t insert(const Point& p, bool surface) {
    points_.insert(points_.end(), p.begin(), p.end());
    alive_.push_back(1);
    on_surface_.push_back(surface ? 1 : 0);
    rebind();
    const auto id = static_cast<int32_t>(vertex_count() - 1);
    journal_.clear();
    const int32_t status = triangulation_.extend(std::span<const int32_t>(&id, 1), alive_);
    if (status != PHX_MC_OK) return status;
    created_.clear();
    double nearest = std::numeric_limits<double>::infinity();
    for (const CellEvent& event : journal_) {
      if (!event.created) continue;
      created_.push_back(event.slot);
      for (int32_t v : tets()[static_cast<std::size_t>(event.slot)].v) {
        if (v == id || v == kGhostVertex) continue;
        const Point gap = minus(point(v), p.data());
        nearest = std::min(nearest, dot(gap, gap));
      }
    }
    if (nearest < separation2_) return PHX_MC_UNRESOLVED;
    update(created_);
    return PHX_MC_OK;
  }

  // Circumcenters, values and labels of new cells, then their faces and shapes.
  void update(std::span<const int32_t> cells) {
    data_.resize(tets().size());
    mark_.resize(tets().size(), 0);
    centers_.clear();
    finite_.clear();
    for (int32_t t : cells) {
      CellData& cell = data_[static_cast<std::size_t>(t)];
      cell.generation = ++generation_;
      cell.inside = false;
      const Tetrahedron& tet = tets()[static_cast<std::size_t>(t)];
      if (is_ghost(tet)) continue;
      const double* p0 = point(tet.v[0]);
      const Point a = minus(point(tet.v[1]), p0), b = minus(point(tet.v[2]), p0),
                  c = minus(point(tet.v[3]), p0);
      const Point bc = cross(b, c), ca = cross(c, a), ab = cross(a, b);
      const double scale = 0.5 / dot(a, bc);
      const double aa = dot(a, a), bb = dot(b, b), cc = dot(c, c);
      Point offset{};
      for (std::size_t d = 0; d < 3; ++d) offset[d] = scale * (aa * bc[d] + bb * ca[d] + cc * ab[d]);
      cell.radius2 = dot(offset, offset);
      for (std::size_t d = 0; d < 3; ++d) cell.center[d] = p0[d] + offset[d];
      centers_.insert(centers_.end(), cell.center.begin(), cell.center.end());
      finite_.push_back(t);
    }
    values_.resize(finite_.size());
    field_.values(centers_, values_.data());
    for (std::size_t k = 0; k < finite_.size(); ++k) {
      CellData& cell = data_[static_cast<std::size_t>(finite_[k])];
      cell.value = values_[k];
      cell.inside = values_[k] < 0.0;
    }
    examine(cells);
  }

  // Queues the restricted facets and inside cells of `cells` that violate
  // the criteria.
  void examine(std::span<const int32_t> cells) {
    ++batch_;
    for (int32_t t : cells) mark_[static_cast<std::size_t>(t)] = batch_;
    pending_.clear();
    for (int32_t t : cells) {
      const Tetrahedron& tet = tets()[static_cast<std::size_t>(t)];
      const bool inside = data(t).inside;
      if (inside) check_cell(t);
      for (int k = 0; k < 4; ++k) {
        const int32_t n = tet.n[k];
        if (data(n).inside == inside) continue;
        if (inside) {
          pending_.push_back({t, k});
        } else if (mark_[static_cast<std::size_t>(n)] != batch_) {
          pending_.push_back({n, neighbor_slot(tets()[static_cast<std::size_t>(n)], t)});
        }
      }
    }
    if (pending_.empty()) return;
    brackets_.clear();
    for (const auto& facet : pending_) brackets_.push_back(dual_bracket(facet[0], facet[1]));
    solve_rays();
    solver_.solve(field_, brackets_);
    for (std::size_t k = 0; k < pending_.size(); ++k) {
      if (!(brackets_[k].f_positive >= 0.0)) continue;
      check_facet(pending_[k][0], pending_[k][1], brackets_[k].root);
    }
  }

  // The dual Voronoi edge of face `slot` of inside cell t: to the outside
  // neighbor's circumcenter, or along the outward normal past the far box
  // toward a ghost (its far value is filled by solve_rays).
  Bracket dual_bracket(int32_t t, int slot) {
    const Tetrahedron& tet = tets()[static_cast<std::size_t>(t)];
    const CellData& inner = data(t);
    const int32_t n = tet.n[slot];
    Bracket bracket;
    bracket.negative = inner.center;
    bracket.f_negative = inner.value;
    if (!is_ghost(tets()[static_cast<std::size_t>(n)])) {
      bracket.positive = data(n).center;
      bracket.f_positive = data(n).value;
      return bracket;
    }
    const int* face = kOutwardFace[slot];
    const double* p0 = point(tet.v[face[0]]);
    const Point normal = cross(minus(point(tet.v[face[1]]), p0), minus(point(tet.v[face[2]]), p0));
    const double length = std::sqrt(dot(normal, normal));
    for (std::size_t d = 0; d < 3; ++d)
      bracket.positive[d] = inner.center[d] + far_length_ * normal[d] / length;
    bracket.f_positive = std::numeric_limits<double>::quiet_NaN();
    return bracket;
  }

  void solve_rays() {
    rays_.clear();
    ray_points_.clear();
    for (std::size_t k = 0; k < brackets_.size(); ++k) {
      if (!std::isnan(brackets_[k].f_positive)) continue;
      rays_.push_back(k);
      ray_points_.insert(ray_points_.end(), brackets_[k].positive.begin(), brackets_[k].positive.end());
    }
    if (rays_.empty()) return;
    values_.resize(rays_.size());
    field_.values(ray_points_, values_.data());
    for (std::size_t r = 0; r < rays_.size(); ++r) brackets_[rays_[r]].f_positive = values_[r];
    // A ray that does not leave the solid has no crossing: its bracket is
    // degenerate at the inside circumcenter and its facet is left alone.
    for (std::size_t r = 0; r < rays_.size(); ++r) {
      Bracket& bracket = brackets_[rays_[r]];
      if (bracket.f_positive < 0.0) {
        bracket.positive = bracket.negative;
        bracket.f_positive = std::numeric_limits<double>::quiet_NaN();
      }
    }
  }

  void check_cell(int32_t t) {
    const Tetrahedron& tet = tets()[static_cast<std::size_t>(t)];
    const CellData& cell = data(t);
    double shortest = std::numeric_limits<double>::infinity();
    for (const auto& edge : kEdges) {
      const Point e = minus(point(tet.v[edge[0]]), point(tet.v[edge[1]]));
      shortest = std::min(shortest, dot(e, e));
    }
    if (cell.radius2 > criteria_.size2 || cell.radius2 > criteria_.ratio2 * shortest)
      cells_.push({cell.radius2, t, cell.generation});
  }

  void check_facet(int32_t t, int slot, const Point& center) {
    const Tetrahedron& tet = tets()[static_cast<std::size_t>(t)];
    const int32_t v[3] = {tet.v[kOutwardFace[slot][0]], tet.v[kOutwardFace[slot][1]],
                          tet.v[kOutwardFace[slot][2]]};
    const double* p = point(v[0]);
    const Point ball = minus(center.data(), p);
    const double key = dot(ball, ball);
    // A restricted facet must join points of the zero set.
    bool bad = !on_surface_[static_cast<std::size_t>(v[0])] ||
               !on_surface_[static_cast<std::size_t>(v[1])] ||
               !on_surface_[static_cast<std::size_t>(v[2])];
    if (!bad) {
      const Point a = minus(point(v[1]), p), b = minus(point(v[2]), p), c = minus(point(v[2]), point(v[1]));
      const double aa = dot(a, a), bb = dot(b, b), cc = dot(c, c);
      const Point normal = cross(a, b);
      const double nn = dot(normal, normal);
      const Point ra = cross(b, normal), rb = cross(normal, a);
      Point offset{};
      for (std::size_t d = 0; d < 3; ++d) offset[d] = (aa * ra[d] + bb * rb[d]) / (2.0 * nn);
      const double rho2 = dot(offset, offset);
      const Point gap{ball[0] - offset[0], ball[1] - offset[1], ball[2] - offset[2]};
      // The smallest angle is opposite the shortest edge: sin = shortest / (2 rho).
      bad = std::max({aa, bb, cc}) > criteria_.edge2 || dot(gap, gap) > criteria_.distance2 ||
            std::min({aa, bb, cc}) < 4.0 * criteria_.angle_sine2 * rho2 || !(nn > 0.0);
    }
    if (!bad) return;
    const int32_t n = tet.n[slot];
    facets_.push({key, t, slot, n, data(t).generation, data(n).generation, center});
  }

  bool current(const FacetEntry& entry) const {
    return complex().live(entry.cell) && data(entry.cell).generation == entry.cell_generation &&
           tets()[static_cast<std::size_t>(entry.cell)].n[entry.slot] == entry.other &&
           data(entry.other).generation == entry.other_generation;
  }

  // The circumcenter of inside cell t, or the ball center of a restricted
  // facet that the circumcenter's conflict region would remove.
  Point cell_target(int32_t t, bool& surface, double& radius2) {
    return target(t, data(t).center, surface, radius2);
  }

  bool in_far_box(const Point& p) const {
    for (std::size_t d = 0; d < 3; ++d)
      if (!(p[d] > far_lower_[d] && p[d] < far_upper_[d])) return false;
    return true;
  }

  // Grows zone_, the cells whose circumspheres hold p, from `start` (one of
  // them); returns the squared distance from p to the zone's nearest vertex,
  // its nearest vertex in the Delaunay triangulation.
  double conflict_zone(int32_t start, const Point& p) {
    ++batch_;
    zone_.assign(1, start);
    mark_[static_cast<std::size_t>(start)] = batch_;
    double nearest = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < zone_.size(); ++k) {
      const Tetrahedron& cell = tets()[static_cast<std::size_t>(zone_[k])];
      for (int32_t v : cell.v) {
        const Point gap = minus(point(v), p.data());
        nearest = std::min(nearest, dot(gap, gap));
      }
      for (int32_t n : cell.n) {
        if (mark_[static_cast<std::size_t>(n)] == batch_) continue;
        const Tetrahedron& next = tets()[static_cast<std::size_t>(n)];
        if (is_ghost(next) ||
            insphere(point(next.v[0]), point(next.v[1]), point(next.v[2]), point(next.v[3]),
                     p.data()) <= 0)
          continue;
        mark_[static_cast<std::size_t>(n)] = batch_;
        zone_.push_back(n);
      }
    }
    return nearest;
  }

  // `center`, or the ball center of a restricted facet whose surface
  // Delaunay ball holds it (the encroachment rule of Rineau and Yvinec),
  // with the squared distance to the chosen point's nearest vertex. A
  // facet's surface ball lies in the union of its two cells' circumballs
  // (the ends of the pencil of spheres through its vertices), so only the
  // restricted facets of the conflict region of `center` (grown from
  // `start`, a cell in conflict), removed or on its boundary, can hold it.
  // An interior point inside a surface ball would lie next to the zero
  // set, a vertex of restricted facets off it.
  Point target(int32_t start, Point center, bool& surface, double& radius2) {
    radius2 = conflict_zone(start, center);
    surface = false;
    pending_.clear();
    for (int32_t x : zone_) {
      const bool inside = data(x).inside;
      const Tetrahedron& cell = tets()[static_cast<std::size_t>(x)];
      for (int k = 0; k < 4; ++k) {
        const int32_t n = cell.n[k];
        if (data(n).inside == inside) continue;
        if (inside) {
          pending_.push_back({x, k});
        } else if (mark_[static_cast<std::size_t>(n)] != batch_) {
          pending_.push_back({n, neighbor_slot(tets()[static_cast<std::size_t>(n)], x)});
        }
      }
    }
    if (pending_.empty()) return center;
    brackets_.clear();
    for (const auto& facet : pending_) brackets_.push_back(dual_bracket(facet[0], facet[1]));
    solve_rays();
    solver_.solve(field_, brackets_);
    for (std::size_t k = 0; k < pending_.size(); ++k) {
      if (!(brackets_[k].f_positive >= 0.0)) continue;
      const Point& ball = brackets_[k].root;
      const Tetrahedron& cell = tets()[static_cast<std::size_t>(pending_[k][0])];
      // The surface Delaunay ball is empty: a facet vertex is nearest.
      const Point rim = minus(point(cell.v[kOutwardFace[pending_[k][1]][0]]), ball.data());
      const Point offset = minus(center.data(), ball.data());
      if (!(dot(offset, offset) < dot(rim, rim))) continue;
      radius2 = dot(rim, rim);
      surface = true;
      return ball;
    }
    return center;
  }

  Field& field_;
  Criteria criteria_;
  double separation2_;
  std::vector<double> points_;
  std::vector<char> alive_, on_surface_;
  Triangulation3D triangulation_;
  std::vector<CellEvent> journal_;
  std::vector<CellData> data_;
  std::vector<std::uint32_t> mark_;
  std::uint32_t batch_ = 0;
  std::uint32_t generation_ = 0;
  Point far_lower_{}, far_upper_{};
  double far_length_ = 0.0;
  std::priority_queue<FacetEntry> facets_;
  std::priority_queue<CellEntry> cells_;
  std::vector<FacetEntry> refused_;
  std::vector<int32_t> created_, finite_, zone_;
  std::vector<std::size_t> rays_;
  std::vector<double> centers_, values_, ray_points_;
  std::vector<std::array<int32_t, 2>> pending_;
  std::vector<Bracket> brackets_;
  RootSolver solver_;
};

// Moves points onto the zero set by `steps` Newton steps along the gradient.
void project(Field& field, std::vector<Point>& points, int steps) {
  std::vector<double> coordinates, jets;
  const auto width = static_cast<std::size_t>(jet_components(3));
  for (int step = 0; step < steps; ++step) {
    coordinates.clear();
    for (const Point& p : points) coordinates.insert(coordinates.end(), p.begin(), p.end());
    field.jets(coordinates, jets);
    for (std::size_t k = 0; k < points.size(); ++k) {
      const double* jet = jets.data() + width * k;
      const double g2 = jet[1] * jet[1] + jet[2] * jet[2] + jet[3] * jet[3];
      if (!(g2 > 0.0) || !std::isfinite(jet[0])) continue;
      for (std::size_t d = 0; d < 3; ++d) points[k][d] -= jet[0] * jet[1 + d] / g2;
    }
  }
}

// Moves the boundary vertices onto the zero set to float accuracy, keeping
// the construction position of any vertex of a cell the move would leave
// without positive orientation.
void polish_boundary(Field& field, std::vector<double>& points, std::span<const int32_t> cells,
                     std::span<const int32_t> faces) {
  std::vector<char> on_boundary(points.size() / 3, 0);
  for (int32_t v : faces) on_boundary[static_cast<std::size_t>(v)] = 1;
  std::vector<int32_t> boundary;
  for (std::size_t v = 0; v < on_boundary.size(); ++v)
    if (on_boundary[v]) boundary.push_back(static_cast<int32_t>(v));
  std::vector<Point> moved;
  moved.reserve(boundary.size());
  for (int32_t v : boundary) {
    const double* p = points.data() + 3 * static_cast<std::size_t>(v);
    moved.push_back({p[0], p[1], p[2]});
  }
  project(field, moved, kPolishSteps);
  const std::vector<double> original = points;
  for (std::size_t k = 0; k < boundary.size(); ++k)
    std::copy(moved[k].begin(), moved[k].end(), points.begin() + 3 * static_cast<std::ptrdiff_t>(boundary[k]));
  for (bool reverted = true; reverted;) {
    reverted = false;
    for (std::size_t c = 0; c < cells.size() / 4; ++c) {
      const int32_t* v = cells.data() + 4 * c;
      const auto at = [&](int k) { return points.data() + 3 * static_cast<std::size_t>(v[k]); };
      if (orient3d(at(0), at(1), at(2), at(3)) > 0) continue;
      for (int k = 0; k < 4; ++k) {
        const auto row = static_cast<std::ptrdiff_t>(3 * static_cast<std::size_t>(v[k]));
        if (!std::equal(original.begin() + row, original.begin() + row + 3, points.begin() + row)) {
          std::copy(original.begin() + row, original.begin() + row + 3, points.begin() + row);
          reverted = true;
        }
      }
    }
  }
}

// Removes cells whose smallest dihedral angle is below a target from a mesh
// with a fixed boundary: edge removal (the ring around an interior edge
// retriangulated by the dynamic program of Klingner and Shewchuk, "Aggressive
// tetrahedral mesh improvement", 2007), face removal (2-3 flips) and
// relocation of interior vertices. Every change strictly raises the
// smallest angle among the cells it replaces and keeps them positive (exact
// orientation) within the radius-edge ratio and relative determinant.
class SliverRemover {
 public:
  SliverRemover(std::vector<double>& points, std::vector<int32_t>& cells,
                std::span<const int32_t> faces, double dihedral, double ratio, double determinant)
      : points_(points), target_(std::cos(dihedral * (std::numbers::pi / 180.0))), ratio_(ratio),
        determinant_(determinant), fixed_(points.size() / 3, 0), stars_(points.size() / 3) {
    for (int32_t v : faces) fixed_[static_cast<std::size_t>(v)] = 1;
    // Each star with room for its cells and a few the flips add.
    std::vector<int32_t> degree(stars_.size(), 0);
    for (int32_t v : cells) ++degree[static_cast<std::size_t>(v)];
    for (std::size_t v = 0; v < stars_.size(); ++v) stars_[v].reserve(static_cast<std::size_t>(degree[v]) + 8);
    const std::size_t count = cells.size() / 4;
    cells_.reserve(2 * count);
    alive_.reserve(2 * count);
    cosine_.reserve(2 * count);
    for (std::size_t c = 0; c < count; ++c)
      add({cells[4 * c], cells[4 * c + 1], cells[4 * c + 2], cells[4 * c + 3]});
  }

  // Runs up to `passes` sweeps over the cells below the target; returns the
  // remaining cells (live, in order) in `cells`, the circumcenters of those
  // still below it in `leftover`, and their count.
  int64_t run(int passes, std::vector<int32_t>& cells, std::vector<Point>& leftover) {
    for (int pass = 0; pass < passes; ++pass) {
      bool changed = false, bad = false;
      const std::size_t count = cells_.size();
      for (std::size_t c = 0; c < count; ++c) {
        if (!alive_[c] || cosine_[c] <= target_) continue;
        bad = true;
        changed = remove_edge(c) || remove_face(c) || relocate(c) || changed;
      }
      if (!bad || !changed) break;
    }
    cells.clear();
    leftover.clear();
    for (std::size_t c = 0; c < cells_.size(); ++c) {
      if (!alive_[c]) continue;
      cells.insert(cells.end(), cells_[c].begin(), cells_[c].end());
      if (cosine_[c] <= target_) continue;
      const Cell& cell = cells_[c];
      const double* p0 = at(cell[0]);
      const Point a = minus(at(cell[1]), p0), b = minus(at(cell[2]), p0), d = minus(at(cell[3]), p0);
      const Point bd = cross(b, d), da = cross(d, a), ab = cross(a, b);
      const double scale = 0.5 / dot(a, bd);
      const double aa = dot(a, a), bb = dot(b, b), dd = dot(d, d);
      Point center{};
      for (std::size_t k = 0; k < 3; ++k) center[k] = p0[k] + scale * (aa * bd[k] + bb * da[k] + dd * ab[k]);
      if (std::isfinite(center[0]) && std::isfinite(center[1]) && std::isfinite(center[2]))
        leftover.push_back(center);
    }
    return static_cast<int64_t>(leftover.size());
  }

 private:
  using Cell = std::array<int32_t, 4>;
  const double* at(int32_t v) const { return points_.data() + 3 * static_cast<std::size_t>(v); }

  // The largest cosine of the cell's interior dihedral angles; +inf unless
  // it is positive, and, when it meets the target, +inf unless its
  // radius-edge ratio is within ratio_ (0: any) and its relative determinant
  // at least determinant_.
  double shape(const Cell& cell) const {
    const double* p[4] = {at(cell[0]), at(cell[1]), at(cell[2]), at(cell[3])};
    if (orient3d(p[0], p[1], p[2], p[3]) <= 0) return std::numeric_limits<double>::infinity();
    Point normals[4];
    double scale[4];
    for (int f = 0; f < 4; ++f) {
      const double* q0 = p[kOutwardFace[f][0]];
      normals[f] = cross(minus(p[kOutwardFace[f][1]], q0), minus(p[kOutwardFace[f][2]], q0));
      scale[f] = 1.0 / std::sqrt(dot(normals[f], normals[f]));
    }
    double cosine = -1.0;
    for (int i = 0; i < 4; ++i)
      for (int j = i + 1; j < 4; ++j)
        cosine = std::max(cosine, -dot(normals[i], normals[j]) * scale[i] * scale[j]);
    if (cosine > target_) return cosine;
    const Point a = minus(p[1], p[0]), b = minus(p[2], p[0]), c = minus(p[3], p[0]);
    const Point bc = cross(b, c);
    const double volume = dot(a, bc);
    double lengths[6], shortest = std::numeric_limits<double>::infinity();
    for (int e = 0; e < 6; ++e) {
      const Point d = minus(p[kEdges[e][0]], p[kEdges[e][1]]);
      lengths[e] = dot(d, d);
      shortest = std::min(shortest, lengths[e]);
    }
    if (ratio_ > 0.0) {
      const Point ca = cross(c, a), ab = cross(a, b);
      Point offset{};
      for (std::size_t d = 0; d < 3; ++d)
        offset[d] = (lengths[0] * bc[d] + lengths[1] * ca[d] + lengths[2] * ab[d]) / (2.0 * volume);
      if (dot(offset, offset) > ratio_ * ratio_ * shortest) return std::numeric_limits<double>::infinity();
    }
    if (determinant_ > 0.0) {
      // Products of the three squared edge lengths at each corner (kEdges
      // order: 01 02 03 12 13 23).
      const double products[4] = {lengths[0] * lengths[1] * lengths[2], lengths[0] * lengths[3] * lengths[4],
                                  lengths[1] * lengths[3] * lengths[5], lengths[2] * lengths[4] * lengths[5]};
      const double largest = std::max({products[0], products[1], products[2], products[3]});
      if (volume * volume < determinant_ * determinant_ * largest) return std::numeric_limits<double>::infinity();
    }
    return cosine;
  }

  void add(const Cell& cell) {
    const auto id = static_cast<int32_t>(cells_.size());
    cells_.push_back(cell);
    alive_.push_back(1);
    cosine_.push_back(shape(cell));
    for (int32_t v : cell) stars_[static_cast<std::size_t>(v)].push_back(id);
  }
  void kill(std::size_t c) {
    alive_[c] = 0;
    for (int32_t v : cells_[c]) {
      auto& star = stars_[static_cast<std::size_t>(v)];
      star.erase(std::find(star.begin(), star.end(), static_cast<int32_t>(c)));
    }
  }
  static bool has(const Cell& cell, int32_t v) {
    return cell[0] == v || cell[1] == v || cell[2] == v || cell[3] == v;
  }

  // The ring of cells around edge (a, b) and its vertices, ordered so that
  // (a, b, r[i], r[i + 1]) is positive; false unless the ring is closed.
  struct Ring {
    std::array<int32_t, 7> cells, vertices;
    std::size_t size = 0;
  };
  bool ring(int32_t a, int32_t b, Ring& result) const {
    std::array<int32_t, 7> around;
    std::size_t n = 0;
    for (int32_t c : stars_[static_cast<std::size_t>(a)]) {
      if (!has(cells_[static_cast<std::size_t>(c)], b)) continue;
      if (n == around.size()) return false;
      around[n++] = c;
    }
    if (n < 3) return false;
    std::array<std::array<int32_t, 2>, 7> links{};
    for (std::size_t k = 0; k < n; ++k) {
      const Cell& cell = cells_[static_cast<std::size_t>(around[k])];
      int32_t other[2], m = 0;
      for (int32_t v : cell)
        if (v != a && v != b) other[m++] = v;
      links[k] = orient3d(at(a), at(b), at(other[0]), at(other[1])) > 0
                     ? std::array<int32_t, 2>{other[0], other[1]}
                     : std::array<int32_t, 2>{other[1], other[0]};
    }
    result.vertices[0] = links[0][0];
    result.cells[0] = around[0];
    // Follow the directed link edges from the first one.
    std::array<char, 7> used{};
    used[0] = 1;
    int32_t head = links[0][1];
    for (std::size_t step = 1; step < n; ++step) {
      std::size_t next = n;
      for (std::size_t k = 0; k < n; ++k)
        if (!used[k] && links[k][0] == head) next = k;
      if (next == n) return false;
      used[next] = 1;
      result.vertices[step] = head;
      result.cells[step] = around[next];
      head = links[next][1];
    }
    if (head != result.vertices[0]) return false;
    result.size = n;
    return true;
  }

  double worst(std::span<const int32_t> cells) const {
    double value = -1.0;
    for (int32_t c : cells) value = std::max(value, cosine_[static_cast<std::size_t>(c)]);
    return value;
  }

  // Retriangulates the ring of the best of c's interior edges.
  bool remove_edge(std::size_t c) {
    const Cell cell = cells_[c];
    Ring around;
    for (const auto& e : kEdges) {
      const int32_t a = cell[static_cast<std::size_t>(e[0])], b = cell[static_cast<std::size_t>(e[1])];
      if (!ring(a, b, around)) continue;
      const std::span<const int32_t> vertices(around.vertices.data(), around.size);
      const double before = worst(std::span<const int32_t>(around.cells.data(), around.size));
      const int n = static_cast<int>(around.size);
      // best[i][j]: the smallest largest cosine over triangulations of the
      // polygon vertices i..j; split[i][j] its apex.
      double best[7][7];
      int split[7][7];
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) best[i][j] = -1.0;
      for (int length = 2; length < n; ++length)
        for (int i = 0; i + length < n; ++i) {
          const int j = i + length;
          best[i][j] = std::numeric_limits<double>::infinity();
          for (int k = i + 1; k < j; ++k) {
            const double sides = std::max(best[i][k], best[k][j]);
            if (sides >= best[i][j]) continue;
            const int32_t ri = vertices[static_cast<std::size_t>(i)], rk = vertices[static_cast<std::size_t>(k)],
                          rj = vertices[static_cast<std::size_t>(j)];
            const double value = std::max({sides, shape({a, ri, rk, rj}), shape({b, rj, rk, ri})});
            if (value < best[i][j]) {
              best[i][j] = value;
              split[i][j] = k;
            }
          }
        }
      if (!(best[0][n - 1] < before)) continue;
      for (std::size_t k = 0; k < around.size; ++k) kill(static_cast<std::size_t>(around.cells[k]));
      // Depth-first over the split tree, at most n intervals pending.
      std::array<std::array<int, 2>, 8> stack{};
      std::size_t pending = 0;
      stack[pending++] = {0, n - 1};
      while (pending > 0) {
        const auto [i, j] = stack[--pending];
        if (j - i < 2) continue;
        const int k = split[i][j];
        const int32_t ri = vertices[static_cast<std::size_t>(i)], rk = vertices[static_cast<std::size_t>(k)],
                      rj = vertices[static_cast<std::size_t>(j)];
        add({a, ri, rk, rj});
        add({b, rj, rk, ri});
        stack[pending++] = {i, k};
        stack[pending++] = {k, j};
      }
      return true;
    }
    return false;
  }

  // Replaces c and a face neighbor by three cells around their apexes' edge.
  bool remove_face(std::size_t c) {
    const Cell cell = cells_[c];
    for (int slot = 0; slot < 4; ++slot) {
      const int32_t x = cell[static_cast<std::size_t>(kOutwardFace[slot][0])],
                    y = cell[static_cast<std::size_t>(kOutwardFace[slot][1])],
                    z = cell[static_cast<std::size_t>(kOutwardFace[slot][2])];
      const int32_t a = cell[static_cast<std::size_t>(slot)];
      int32_t other = -1;
      for (int32_t d : stars_[static_cast<std::size_t>(x)])
        if (static_cast<std::size_t>(d) != c && has(cells_[static_cast<std::size_t>(d)], y) &&
            has(cells_[static_cast<std::size_t>(d)], z))
          other = d;
      if (other < 0) continue;
      int32_t b = -1;
      for (int32_t v : cells_[static_cast<std::size_t>(other)])
        if (v != x && v != y && v != z) b = v;
      const Cell created[3] = {{a, b, x, y}, {a, b, y, z}, {a, b, z, x}};
      const double before = std::max(cosine_[c], cosine_[static_cast<std::size_t>(other)]);
      double after = -1.0;
      for (const Cell& n : created) after = std::max(after, shape(n));
      if (!(after < before)) continue;
      kill(c);
      kill(static_cast<std::size_t>(other));
      for (const Cell& n : created) add(n);
      return true;
    }
    return false;
  }

  // Moves an interior vertex of c to raise the smallest angle of its star:
  // repeatedly away from the face opposite it in the star's worst cell
  // (which raises that cell's height), else toward the star's centroid.
  bool relocate(std::size_t c) {
    constexpr int kAscents = 6;
    constexpr double kSteps[] = {0.3, 0.15, 0.075};
    const Cell cell = cells_[c];
    bool moved = false;
    for (int k = 0; k < 4; ++k) {
      const int32_t v = cell[static_cast<std::size_t>(k)];
      if (fixed_[static_cast<std::size_t>(v)]) continue;
      const std::vector<int32_t>& star = stars_[static_cast<std::size_t>(v)];
      double* p = points_.data() + 3 * static_cast<std::size_t>(v);
      double best = worst(star);
      // The worst cosine over the star with v at q, abandoned at `best`.
      const auto star_cosine = [&](const Point& q) {
        std::copy(q.begin(), q.end(), p);
        double value = -1.0;
        for (int32_t s : star) {
          value = std::max(value, shape(cells_[static_cast<std::size_t>(s)]));
          if (value >= best) break;
        }
        return value;
      };
      for (int ascent = 0; ascent < kAscents; ++ascent) {
        const Point origin{p[0], p[1], p[2]};
        auto worst_cell = static_cast<std::size_t>(star.front());
        for (int32_t s : star)
          if (cosine_[static_cast<std::size_t>(s)] > cosine_[worst_cell]) worst_cell = static_cast<std::size_t>(s);
        const Cell& w = cells_[worst_cell];
        const auto slot = static_cast<int>(std::find(w.begin(), w.end(), v) - w.begin());
        const double* q0 = at(w[static_cast<std::size_t>(kOutwardFace[slot][0])]);
        const Point normal = cross(minus(at(w[static_cast<std::size_t>(kOutwardFace[slot][2])]), q0),
                                  minus(at(w[static_cast<std::size_t>(kOutwardFace[slot][1])]), q0));
        double edge = 0.0;
        for (const auto& e : kEdges) {
          const Point d = minus(at(w[static_cast<std::size_t>(e[0])]), at(w[static_cast<std::size_t>(e[1])]));
          edge += std::sqrt(dot(d, d)) / 6.0;
        }
        const double scale = edge / std::sqrt(dot(normal, normal));
        Point centroid{};
        for (int32_t s : star)
          for (int32_t u : cells_[static_cast<std::size_t>(s)])
            for (std::size_t d = 0; d < 3; ++d)
              centroid[d] += (u == v ? origin[d] : at(u)[d]) / (4.0 * static_cast<double>(star.size()));
        Point candidates[4];
        for (std::size_t j = 0; j < 3; ++j)
          for (std::size_t d = 0; d < 3; ++d) candidates[j][d] = origin[d] + kSteps[j] * scale * normal[d];
        candidates[3] = centroid;
        bool improved = false;
        for (const Point& q : candidates) {
          const double value = star_cosine(q);
          if (value < best) {
            best = value;
            improved = true;
            break;
          }
        }
        if (!improved) {
          std::copy(origin.begin(), origin.end(), p);
          break;
        }
        moved = true;
        for (int32_t s : star) cosine_[static_cast<std::size_t>(s)] = shape(cells_[static_cast<std::size_t>(s)]);
        if (cosine_[c] <= target_) return true;
      }
    }
    return moved;
  }

  std::vector<double>& points_;
  double target_, ratio_, determinant_;
  std::vector<char> fixed_;
  std::vector<std::vector<int32_t>> stars_;
  std::vector<Cell> cells_;
  std::vector<char> alive_;
  std::vector<double> cosine_;
};

// Whether binary64 shape estimates meet the quality targets (0 disables the
// radius-edge target).
bool meets_quality(std::span<const double> points, std::span<const int32_t> cells,
                   double radius_edge, double dihedral, double determinant) {
  for (std::size_t c = 0; c < cells.size() / 4; ++c) {
    const double* p[4];
    for (std::size_t k = 0; k < 4; ++k) p[k] = points.data() + 3 * static_cast<std::size_t>(cells[4 * c + k]);
    geometry::TetrahedronShape shape;
    if (!geometry::tetrahedron_shape(p, shape)) return false;
    if ((radius_edge > 0.0 && shape.radius_edge_ratio > radius_edge) ||
        geometry::minimum_dihedral_degrees(p) < dihedral ||
        geometry::relative_determinant(p) < determinant)
      return false;
  }
  return true;
}

// Outward faces of positive tetrahedra that no other cell shares.
void outward_boundary(std::span<const int32_t> cells, std::vector<int32_t>& faces) {
  struct Face {
    std::array<int32_t, 3> sorted, oriented;
  };
  std::vector<Face> all;
  all.reserve(cells.size());
  for (std::size_t c = 0; c < cells.size() / 4; ++c)
    for (const auto& face : kOutwardFace) {
      Face entry;
      for (std::size_t r = 0; r < 3; ++r) entry.oriented[r] = cells[4 * c + static_cast<std::size_t>(face[r])];
      entry.sorted = entry.oriented;
      std::sort(entry.sorted.begin(), entry.sorted.end());
      all.push_back(entry);
    }
  std::sort(all.begin(), all.end(), [](const Face& a, const Face& b) { return a.sorted < b.sorted; });
  faces.clear();
  for (std::size_t k = 0; k < all.size();) {
    std::size_t end = k + 1;
    while (end < all.size() && all[end].sorted == all[k].sorted) ++end;
    if (end == k + 1) faces.insert(faces.end(), all[k].oriented.begin(), all[k].oriented.end());
    k = end;
  }
}

// The fixed-boundary quality improvement of (points, cells, faces):
// improvement, then radius-edge refinement and improvement again when cells
// stay unresolved.
int32_t improve_quality(std::vector<double>& points, std::vector<int32_t>& cells,
                        std::vector<int32_t>& faces, const TetQualityRequest& request,
                        int64_t& unmet) {
  TetMesh mesh(BoundaryPolicy::kFixed);
  const std::size_t face_count = faces.size() / 3;
  std::vector<int32_t> regions(cells.size() / 4, 0), sources(face_count);
  std::iota(sources.begin(), sources.end(), 0);
  const std::vector<int8_t> orientations(face_count, 1);
  int32_t status = mesh.build(points, cells, regions, faces, sources, orientations, {}, {}, {}, nullptr);
  if (status != PHX_MC_OK) return status;
  status = mesh.assign_initial_ids({}, {}, {}, {});
  if (status != PHX_MC_OK) return status;
  int64_t inserted = 0, collapsed = 0, relocated = 0;
  status = improve_mesh(mesh, request, {}, inserted, collapsed, relocated, unmet);
  if (status == PHX_MC_UNRESOLVED) {
    status = refine_mesh(mesh, request.radius_edge_bound, {}, inserted, unmet);
    if (status != PHX_MC_OK && status != PHX_MC_UNRESOLVED) return status;
    status = improve_mesh(mesh, request, {}, inserted, collapsed, relocated, unmet);
  }
  if (status != PHX_MC_OK && status != PHX_MC_UNRESOLVED) return status;
  TetMeshSnapshot snapshot;
  if (mesh.snapshot(snapshot) != TetEditStatus::kApplied) return PHX_MC_INTERNAL_ERROR;
  std::vector<std::pair<std::uint64_t, int32_t>> rows;
  rows.reserve(snapshot.vertices.size());
  points.clear();
  for (const TetSnapshotVertex& vertex : snapshot.vertices) {
    rows.push_back({vertex.id, static_cast<int32_t>(rows.size())});
    points.insert(points.end(), vertex.carrier.begin(), vertex.carrier.end());
  }
  std::sort(rows.begin(), rows.end());
  const auto row_of = [&](std::uint64_t id) {
    return std::lower_bound(rows.begin(), rows.end(), std::pair<std::uint64_t, int32_t>{id, -1})->second;
  };
  cells.clear();
  for (const TetSnapshotCell& cell : snapshot.cells)
    for (std::uint64_t id : cell.vertices) cells.push_back(row_of(id));
  outward_boundary(cells, faces);
  return PHX_MC_OK;
}

struct MeshRequest {
  const double* lower;
  const double* upper;
  const double* criteria;
  const int64_t* topology;
  std::span<const double> extra_seeds;
};

// The seeds: the extra points moved onto the zero set, then the box roots
// farther than `spacing` from every earlier seed, polished on their edges;
// of these, the points farther than `separation` from every earlier one.
std::vector<Point> initial_seeds(Field& field, const PossibleZeroBoxes& roots, std::span<const double> extra,
                                 double spacing, double cell, double separation) {
  std::vector<Point> candidates;
  for (std::size_t k = 0; k + 2 < extra.size(); k += 3) candidates.push_back({extra[k], extra[k + 1], extra[k + 2]});
  project(field, candidates, kNewtonSteps);
  PointGrid seeded(std::max(spacing, cell), roots.roots.size() + candidates.size());
  for (const Point& p : candidates) seeded.add(p);
  std::vector<Bracket> chosen;
  for (const Bracket& root : roots.roots) {
    if (seeded.near(root.root, spacing)) continue;
    seeded.add(root.root);
    chosen.push_back(root);
  }
  RootSolver solver;
  solver.solve(field, chosen);
  for (const Bracket& root : chosen) candidates.push_back(root.root);
  PointGrid separated(separation, candidates.size());
  std::vector<Point> seeds;
  for (const Point& p : candidates) {
    if (separated.near(p, separation)) continue;
    separated.add(p);
    seeds.push_back(p);
  }
  return seeds;
}

// The refinement state and the inside cells' restricted surface.
struct SurfaceState {
  int64_t insertions = 0;
  int64_t rounds = 0;
  bool matches = false;
  int64_t uncovered = 0;
  std::vector<int32_t> cells, faces;
  std::vector<std::array<int32_t, 2>> facets;
};

// Refines, then checks the restricted surface for edge-manifoldness, the
// requested topology and coverage of every box root; repairs by failure
// kind (the roots of uncovered boxes, the ball centers of facets at
// singular vertices, or else a finer surface everywhere) for up to
// kRepairRounds rounds. PHX_MC_UNRESOLVED when an insertion broke the
// separation bound.
int32_t refine_surface(RestrictedMesher& mesher, Field& field, const PossibleZeroBoxes& roots,
                       const int64_t* topology, double width, SurfaceState& state) {
  RootSolver solver;
  for (int round = 0;; ++round) {
    int32_t status = mesher.refine(state.insertions);
    if (status != PHX_MC_OK) return status;
    mesher.surface(state.cells, state.faces, state.facets);
    const SurfaceTopology surface = surface_topology(state.faces);
    // The surface's vertices, each once (a closed surface has about one per
    // two faces).
    PointGrid vertices(width, state.faces.size() / 6);
    std::vector<char> added(static_cast<std::size_t>(mesher.vertex_count()), 0);
    for (int32_t v : state.faces) {
      if (added[static_cast<std::size_t>(v)]) continue;
      added[static_cast<std::size_t>(v)] = 1;
      vertices.add({mesher.point(v)[0], mesher.point(v)[1], mesher.point(v)[2]});
    }
    std::vector<char> box_missed(roots.leaf_offsets.size() - 1, 0);
    state.uncovered = 0;
    for (std::size_t r = 0; r < roots.roots.size(); ++r) {
      if (vertices.near(roots.roots[r].root, width)) continue;
      ++state.uncovered;
      box_missed[static_cast<std::size_t>(roots.root_leaf[r])] = 1;
    }
    state.matches = surface.manifold && surface.components == topology[0] && surface.euler == topology[1];
    if ((state.matches && state.uncovered == 0) || round == kRepairRounds) return PHX_MC_OK;
    ++state.rounds;
    std::vector<Bracket> repair;
    for (std::size_t b = 0; b + 1 < roots.leaf_offsets.size(); ++b) {
      if (!box_missed[b]) continue;
      for (int32_t k = roots.leaf_offsets[b]; k < roots.leaf_offsets[b + 1]; ++k)
        repair.push_back(roots.roots[static_cast<std::size_t>(roots.leaf_roots[static_cast<std::size_t>(k)])]);
    }
    solver.solve(field, repair);
    std::vector<Point> points;
    for (const Bracket& root : repair) points.push_back(root.root);
    std::vector<std::array<int32_t, 2>> singular_facets;
    for (std::size_t f = 0; f < state.facets.size(); ++f)
      for (std::size_t r = 0; r < 3; ++r)
        if (std::binary_search(surface.singular.begin(), surface.singular.end(), state.faces[3 * f + r])) {
          singular_facets.push_back(state.facets[f]);
          break;
        }
    std::vector<Point> centers;
    mesher.ball_centers(singular_facets, centers);
    points.insert(points.end(), centers.begin(), centers.end());
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    if (points.empty()) {
      mesher.criteria().edge2 *= 0.5;
      mesher.reexamine();
    }
    status = mesher.insert_all(points);
    if (status != PHX_MC_OK) return status;
  }
}

// The inside cells and their boundary faces with the vertices they use, in
// vertex order; false when a far corner is one of them.
bool compact(const RestrictedMesher& mesher, const SurfaceState& state, phx_mc_restricted_volume& result) {
  std::vector<int32_t> rows(static_cast<std::size_t>(mesher.vertex_count()), -1);
  for (int32_t v : state.cells) rows[static_cast<std::size_t>(v)] = 0;
  result.points.clear();
  int32_t next = 0;
  for (std::size_t v = 0; v < rows.size(); ++v) {
    if (rows[v] < 0) continue;
    rows[v] = next++;
    const double* p = mesher.point(static_cast<int32_t>(v));
    result.points.insert(result.points.end(), p, p + 3);
  }
  result.cells = state.cells;
  result.faces = state.faces;
  for (int32_t& v : result.cells) v = rows[static_cast<std::size_t>(v)];
  for (int32_t& v : result.faces) v = rows[static_cast<std::size_t>(v)];
  return std::none_of(rows.begin(), rows.begin() + kCorners, [](int32_t row) { return row >= 0; });
}

// PHX_MC_OK, or PHX_MC_UNRESOLVED with the proposal reached: an insertion
// broke the separation bound, an inside cell reached a far corner, or a
// facet the bound refused still violates the criteria.
int32_t mesh_restricted(const IntervalProgramView& interval, Field& field, const MeshRequest& request,
                        phx_mc_restricted_volume& result) {
  constexpr int kQualityRounds = 3;
  constexpr int kSliverPasses = 6;
  const double* criteria = request.criteria;
  const double width = criteria[0];
  // A quarter of the smallest insertion radius the criteria ask for, so
  // that the bound binds only where they cannot be met (RestrictedMesher).
  const double separation = 0.25 * std::min({0.5 * width, criteria[1], criteria[3]});
  // The anchored domain of the possible-zero cover: the box grown by 1/16 of
  // its largest extent on every side, which keeps the zero set (tangent to
  // the box's faces on a sphere) off the domain boundary, where no lattice
  // edge crosses it; its target-level cells are no wider than the edge bound
  // (diagonal at most sqrt(3) times it).
  double largest = 0.0;
  for (int d = 0; d < 3; ++d) largest = std::max(largest, request.upper[d] - request.lower[d]);
  double domain_lower[3], domain_upper[3];
  for (int d = 0; d < 3; ++d) {
    domain_lower[d] = request.lower[d] - 0.0625 * largest;
    domain_upper[d] = request.upper[d] + 0.0625 * largest;
  }
  PossibleZeroBoxes roots;
  int32_t status = possible_zero_boxes(interval, field.view(), domain_lower, domain_upper,
                                       dyadic_target_level(domain_lower, domain_upper, std::sqrt(3.0) * width),
                                       roots);
  if (status != PHX_MC_OK) return status;
  const std::vector<Point> seeds =
      initial_seeds(field, roots, request.extra_seeds, criteria[7] * width, width, separation);
  const double angle = criteria[2] * (std::numbers::pi / 180.0);
  const Criteria refinement{width * width, criteria[1] * criteria[1],
                            std::sin(angle) * std::sin(angle), criteria[3] * criteria[3],
                            criteria[4] > 0.0 ? criteria[4] * criteria[4]
                                              : std::numeric_limits<double>::infinity()};
  RestrictedMesher mesher(field, request.lower, request.upper, refinement, separation);
  status = mesher.start(seeds);
  if (status != PHX_MC_OK) return status;
  SurfaceState state;
  std::vector<Point> leftover;
  for (int round = 0;; ++round) {
    status = refine_surface(mesher, field, roots, request.topology, width, state);
    if (status != PHX_MC_OK && status != PHX_MC_UNRESOLVED) return status;
    if (!compact(mesher, state, result)) status = PHX_MC_UNRESOLVED;
    if (status == PHX_MC_UNRESOLVED) {
      result.report = {state.insertions, state.rounds, state.matches, state.uncovered, mesher.refused_facets(), 0};
      return status;
    }
    polish_boundary(field, result.points, result.cells, result.faces);
    if (!(criteria[5] > 0.0)) break;
    SliverRemover remover(result.points, result.cells, result.faces, criteria[5], criteria[4], criteria[6]);
    // Cells the flips and relocations leave below the targets are refined
    // at their circumcenters in the Delaunay triangulation, and the surface
    // and its interior constructed again.
    if (remover.run(kSliverPasses, result.cells, leftover) == 0 || round == kQualityRounds) break;
    status = mesher.refine_at(leftover);
    if (status == PHX_MC_UNRESOLVED) {
      // The arrays are the previous round's construction.
      result.report = {state.insertions, state.rounds, state.matches, state.uncovered, mesher.refused_facets(), 0};
      return status;
    }
    if (status != PHX_MC_OK) return status;
  }
  const int64_t refused = mesher.refused_facets();
  result.report = {state.insertions, state.rounds, state.matches, state.uncovered, refused, 0};
  const TetQualityRequest quality{criteria[4], criteria[5], criteria[6]};
  if (!result.cells.empty() &&
      !meets_quality(result.points, result.cells, quality.radius_edge_bound,
                     quality.minimum_dihedral_degrees, quality.minimum_relative_determinant)) {
    status = improve_quality(result.points, result.cells, result.faces, quality, result.report[5]);
    if (status != PHX_MC_OK) return status;
  }
  return refused > 0 ? PHX_MC_UNRESOLVED : PHX_MC_OK;
}
}  // namespace
}  // namespace phx::mc
extern "C" PHX_MC_API int32_t phx_mc_restricted_centers_3d(
    int64_t point_count, const double* points, int64_t tet_count,
    const int32_t* tets, double* center_bounds, int32_t* item_status) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    return phx::mc::construct_centers(point_count, points, tet_count, tets,
                                      center_bounds, item_status);
  });
}

extern "C" PHX_MC_API int32_t phx_mc_implicit_restricted_delaunay(
    const int64_t* instructions, int64_t instruction_count, const int64_t* indices,
    int64_t index_count, const double* parameters, int64_t parameter_count,
    const double* seed_lower, const double* seed_upper, int64_t workspace_size,
    int64_t output_entry, const double* lower, const double* upper, const double* criteria,
    const int64_t* topology, int64_t extra_seed_count, const double* extra_seeds,
    phx_mc_restricted_volume** volume) {
  return phx::mc::guarded([&]() -> int32_t {
    if (volume == nullptr) return PHX_MC_INVALID_ARGUMENT;
    *volume = nullptr;
    if (seed_lower == nullptr || seed_upper == nullptr || lower == nullptr ||
        upper == nullptr || criteria == nullptr || topology == nullptr || extra_seed_count < 0 ||
        (extra_seed_count > 0 && extra_seeds == nullptr) ||
        !phx::mc::addressable(extra_seed_count, 3, sizeof(double)))
      return PHX_MC_INVALID_ARGUMENT;
    for (int d = 0; d < 3; ++d)
      if (!std::isfinite(lower[d]) || !std::isfinite(upper[d]) || !(lower[d] < upper[d]))
        return PHX_MC_INVALID_ARGUMENT;
    // Edge bound, facet distance, facet angle (at most 30 degrees, the
    // Boissonnat-Oudot bound that keeps the refinement finite), cell size,
    // radius-edge ratio (0: none), dihedral angle, relative determinant and
    // the seed spacing per edge bound.
    if (!(criteria[0] > 0.0) || !std::isfinite(criteria[0]) || !(criteria[1] > 0.0) ||
        !(criteria[2] >= 0.0 && criteria[2] <= 30.0) || !(criteria[3] > 0.0) ||
        !(criteria[4] >= 0.0) || !(criteria[5] >= 0.0 && criteria[5] < 70.0) ||
        !(criteria[6] >= 0.0 && criteria[6] < 1.0) || !(criteria[7] >= 0.0 && criteria[7] <= 1.0))
      return PHX_MC_INVALID_ARGUMENT;
    for (int64_t k = 0; k < 3 * extra_seed_count; ++k)
      if (!std::isfinite(extra_seeds[k])) return PHX_MC_INVALID_ARGUMENT;
    if (output_entry < 0 || output_entry >= workspace_size) return PHX_MC_INVALID_ARGUMENT;
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    int32_t status = phx::mc::validate_point_program(
        instructions, instruction_count, indices, index_count, parameters, parameter_count,
        workspace_size, 3, &output_entry, 1);
    if (status != PHX_MC_OK) return status;
    // The point executor reads each seed's midpoint, as phx_mc_program_evaluate's callers pass.
    std::vector<double> point_seeds(static_cast<std::size_t>(workspace_size));
    for (std::size_t k = 0; k < point_seeds.size(); ++k)
      point_seeds[k] = 0.5 * (seed_lower[k] + seed_upper[k]);
    const phx::mc::ProgramView view{instructions,      instruction_count, indices, point_seeds.data(),
                                    workspace_size,    3,                 &output_entry, 1};
    phx::mc::Field field(view);
    const phx::mc::IntervalProgramView interval{instructions, instruction_count, indices,
                                                index_count,  parameters,        parameter_count,
                                                seed_lower,   seed_upper,        workspace_size,
                                                output_entry};
    auto result = std::make_unique<phx_mc_restricted_volume>();
    status = phx::mc::mesh_restricted(
        interval, field,
        {lower, upper, criteria, topology,
         std::span<const double>(extra_seeds, static_cast<std::size_t>(3 * extra_seed_count))},
        *result);
    if (status != PHX_MC_OK && status != PHX_MC_UNRESOLVED) return status;
    *volume = result.release();
    return status;
  });
}

extern "C" PHX_MC_API int32_t phx_mc_restricted_volume_sizes(
    const phx_mc_restricted_volume* volume, int64_t* sizes, int64_t* report) {
  if (volume == nullptr || sizes == nullptr || report == nullptr) return PHX_MC_INVALID_ARGUMENT;
  sizes[0] = static_cast<int64_t>(volume->points.size() / 3);
  sizes[1] = static_cast<int64_t>(volume->cells.size() / 4);
  sizes[2] = static_cast<int64_t>(volume->faces.size() / 3);
  std::copy(volume->report.begin(), volume->report.end(), report);
  return PHX_MC_OK;
}

extern "C" PHX_MC_API int32_t phx_mc_restricted_volume_copy(
    const phx_mc_restricted_volume* volume, double* points, int32_t* cells, int32_t* faces) {
  if (volume == nullptr || (!volume->points.empty() && points == nullptr) ||
      (!volume->cells.empty() && cells == nullptr) || (!volume->faces.empty() && faces == nullptr))
    return PHX_MC_INVALID_ARGUMENT;
  std::copy(volume->points.begin(), volume->points.end(), points);
  std::copy(volume->cells.begin(), volume->cells.end(), cells);
  std::copy(volume->faces.begin(), volume->faces.end(), faces);
  return PHX_MC_OK;
}

extern "C" PHX_MC_API void phx_mc_restricted_volume_free(phx_mc_restricted_volume* volume) {
  delete volume;
}
