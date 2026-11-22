//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// phx_mc_polyhedral_declared_contacts proves, for a pair of star-decomposed
// linear polyhedral cells, that every intersection of a tetrahedron of the
// first cell with one of the second lies in one convex piece of the pair's
// declared shared stratum (phydrax `_global_embedding._declared_stratum`).
// It reports only that statement; a pair it does not prove keeps the exact
// rational contact pass, which then decides it and states its findings.
//
// Pieces. The shared vertex images, the edges both cells' face loops declare
// within the shared vertices, and every face loop both declare within them
// whose rational pieces are modeled: a loop of more than three vertices
// proved convex exactly as `_convex_planar_face_union` (Fraction coordinates,
// nonzero area vector, coplanar, every turn and every fan triangle from the
// least vertex identifier nonnegative along it) is its hull; a triangle loop
// with a nonzero area vector is its own ear-clipped triangle; any other
// non-planar loop is its vertex-average fan when `polytope_face_routes`
// accepts it (planar non-convex loops are not modeled). Each is a piece of
// the rational stratum, which holds every point of one of its pieces, so a
// proved containment is the rational verdict.
//
// Contacts. Two closed tetrahedra on the closed opposite sides of a plane
// through a point of one of them meet in the plane, inside the hull of each
// one's corners on it. The planes tried are the face planes of both and, for
// every pair of their edges, the planes through either edge spanned with the
// other edge's direction: when the tetrahedra are disjoint one of these
// separates them (the separating axis theorem). When neither on-plane corner
// set lies in one piece, the same statement inside the plane with lines
// through their edges, and on a common line the overlap of the two segments
// (an interval whose endpoints are corners), narrow the contact to corners.
//
// Points arrive as homogeneous integer rows (X, W) of x = X / W, W > 0, with
// outward binary64 enclosures of x; corners of each cell also as integer
// rows over one common denominator of the cell (the scale of its face
// decisions). Every sign is decided in outward-rounded interval arithmetic
// when the enclosure excludes zero, else in exact big-integer arithmetic on
// differences X_b W_a - X_a W_b, which carry b - a with the positive factor
// W_a W_b. Equal point identities are equal exact points.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "capi_guard.hpp"
#include "directed_interval.hpp"
#include "dyadic.hpp"
#include "exact_scalar.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace {

using phx::mc::Dyadic;
using phx::mc::remesh::Interval;
using phx::mc::remesh::IntervalVector;
using Vec = std::array<Dyadic, 3>;
namespace iv = phx::mc::remesh;

constexpr int kUnknown = 2;
// Face m of a tetrahedron omits corner m.
constexpr int kFaceCorners[4][3] = {{1, 2, 3}, {2, 3, 0}, {3, 0, 1}, {0, 1, 2}};
constexpr int kEdgeCorners[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

Vec sub(const Vec& a, const Vec& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Dyadic dot(const Vec& a, const Vec& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec cross(const Vec& a, const Vec& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
bool nonzero(const Vec& a) { return a[0].sign() || a[1].sign() || a[2].sign(); }

int interval_sign(const Interval& value) {
  if (!(std::isfinite(value.lower) && std::isfinite(value.upper))) return kUnknown;
  if (value.lower > 0.0) return 1;
  if (value.upper < 0.0) return -1;
  return kUnknown;
}

IntervalVector box_sub(const IntervalVector& a, const IntervalVector& b) {
  return {iv::subtract(a[0], b[0]), iv::subtract(a[1], b[1]), iv::subtract(a[2], b[2])};
}

struct IntegerBank {
  const int64_t* offsets;
  const uint32_t* words;
  const int8_t* signs;

  Dyadic value(int64_t index) const {
    const int64_t start = offsets[index];
    return Dyadic::integer(words + start, static_cast<std::size_t>(offsets[index + 1] - start),
                           signs[index]);
  }
};

bool valid_bank(int64_t count, const int64_t* offsets, const uint32_t* words, const int8_t* signs) {
  if (count < 0 || offsets == nullptr || offsets[0] != 0 || (count > 0 && signs == nullptr)) {
    return false;
  }
  for (int64_t value = 0; value < count; ++value) {
    if (offsets[value + 1] < offsets[value]) return false;
  }
  return offsets[count] == 0 || words != nullptr;
}

struct HomogeneousPoint {
  Vec x;
  Dyadic w;
};

// Exact points loaded on first use, and their enclosures. Points the proof
// derives (fan apexes of shared faces) take the identity of an equal input
// point when one is offered, else a fresh identity after the input points.
class PointTable {
 public:
  PointTable(const IntegerBank& bank, int64_t count, const double* lower, const double* upper)
      : bank_(bank), count_(count), lower_(lower), upper_(upper) {}

  IntervalVector box(int64_t point) const {
    if (point >= count_) return derived_boxes_[static_cast<std::size_t>(point - count_)];
    const double* low = lower_ + 3 * point;
    const double* high = upper_ + 3 * point;
    return {Interval{low[0], high[0]}, Interval{low[1], high[1]}, Interval{low[2], high[2]}};
  }

  // The identity of an input point among `candidates` exactly equal to
  // `point`, else a fresh derived identity.
  int64_t derive(HomogeneousPoint point, const int64_t* candidates, int64_t count) {
    IntervalVector enclosure;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const auto bounds = phx::mc::exact::directed_quotient_bounds(point.x[axis], point.w);
      enclosure[axis] = {bounds.lower, bounds.upper};
    }
    for (int64_t index = 0; index < count; ++index) {
      const int64_t candidate = candidates[index];
      const IntervalVector other = box(candidate);
      bool overlaps = true;
      for (std::size_t axis = 0; axis < 3 && overlaps; ++axis) {
        overlaps = other[axis].lower <= enclosure[axis].upper &&
                   enclosure[axis].lower <= other[axis].upper;
      }
      if (!overlaps) continue;
      const HomogeneousPoint& known = exact(candidate);
      bool equal = true;
      for (std::size_t axis = 0; axis < 3 && equal; ++axis) {
        equal = (known.x[axis] * point.w - point.x[axis] * known.w).is_zero();
      }
      if (equal) return candidate;
    }
    const int64_t identity = count_ + static_cast<int64_t>(derived_boxes_.size());
    derived_boxes_.push_back(enclosure);
    cache_.emplace(identity, std::move(point));
    return identity;
  }

  // b - a up to the positive factor W_a W_b.
  Vec difference(int64_t a, int64_t b) {
    const HomogeneousPoint& first = exact(a);
    const HomogeneousPoint& second = exact(b);
    return {second.x[0] * first.w - first.x[0] * second.w,
            second.x[1] * first.w - first.x[1] * second.w,
            second.x[2] * first.w - first.x[2] * second.w};
  }

  IntervalVector box_difference(int64_t a, int64_t b) const { return box_sub(box(b), box(a)); }

  // An input point whose enclosure is a point is that binary64 point; inside
  // the predicate domain its orientations are decided by `orient3d`.
  bool binary64(int64_t point, std::array<double, 3>& value) const {
    if (point >= count_) return false;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const double low = lower_[3 * point + static_cast<int64_t>(axis)];
      if (low != upper_[3 * point + static_cast<int64_t>(axis)] ||
          !phx::mc::coordinate_in_domain(low)) {
        return false;
      }
      value[axis] = low;
    }
    return true;
  }

  // Strictly outside the outward box of `hull` on some axis.
  bool outside(int64_t point, const int64_t* hull, std::size_t count) const {
    const IntervalVector enclosure = box(point);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      double low = std::numeric_limits<double>::infinity();
      double high = -std::numeric_limits<double>::infinity();
      for (std::size_t index = 0; index < count; ++index) {
        const Interval bound = box(hull[index])[axis];
        low = std::min(low, bound.lower);
        high = std::max(high, bound.upper);
      }
      if (enclosure[axis].upper < low || enclosure[axis].lower > high) return true;
    }
    return false;
  }

 private:
  const HomogeneousPoint& exact(int64_t point) {
    auto found = cache_.find(point);
    if (found == cache_.end()) {
      HomogeneousPoint loaded{
          {bank_.value(4 * point), bank_.value(4 * point + 1), bank_.value(4 * point + 2)},
          bank_.value(4 * point + 3)};
      found = cache_.emplace(point, std::move(loaded)).first;
    }
    return found->second;
  }

  IntegerBank bank_;
  int64_t count_;
  const double* lower_;
  const double* upper_;
  std::vector<IntervalVector> derived_boxes_;
  // Node-based: references stay valid across insertions.
  std::unordered_map<int64_t, HomogeneousPoint> cache_;
};

// A plane through `anchor` with normal n, decided on the points it contains
// by construction (`on`) without arithmetic.
class Plane {
 public:
  enum class Kind { kFace, kEdges, kInPlane };

  static Plane face(int64_t a, int64_t b, int64_t c, const IntervalVector& normal) {
    Plane plane(Kind::kFace, a, {a, b, c, -1});
    plane.box_ = normal;
    return plane;
  }
  // Through the edge (a0, a1), spanned with the direction of (b0, b1).
  static Plane edges(PointTable& points, int64_t a0, int64_t a1, int64_t b0, int64_t b1) {
    Plane plane(Kind::kEdges, a0, {a0, a1, b0, b1});
    plane.box_ = iv::cross(points.box_difference(a0, a1), points.box_difference(b0, b1));
    return plane;
  }
  // The line through u and v inside `parent`, as the plane normal to it there.
  static Plane in_plane(PointTable& points, Plane& parent, int64_t u, int64_t v) {
    Plane plane(Kind::kInPlane, u, {u, v, -1, -1});
    plane.parent_ = &parent;
    plane.box_ = iv::cross(parent.box_, points.box_difference(u, v));
    return plane;
  }

  // The sign of n . (point - anchor): 0 on the generators, the enclosure's
  // sign when it excludes zero, else kUnknown.
  int enclosed_side(PointTable& points, int64_t point) const {
    if (point == anchor_ || point == generators_[1] ||
        (kind_ == Kind::kFace && point == generators_[2])) {
      return 0;
    }
    return interval_sign(iv::dot(box_, points.box_difference(anchor_, point)));
  }

  int exact_side(PointTable& points, int64_t point) {
    if (kind_ == Kind::kFace) {
      std::array<std::array<double, 3>, 4> corners{};
      if (points.binary64(generators_[0], corners[0]) &&
          points.binary64(generators_[1], corners[1]) &&
          points.binary64(generators_[2], corners[2]) && points.binary64(point, corners[3])) {
        return phx::mc::orient3d(corners[0].data(), corners[1].data(), corners[2].data(),
                                 corners[3].data());
      }
    }
    return dot(exact(points), points.difference(anchor_, point)).sign();
  }

 private:
  Plane(Kind kind, int64_t anchor, std::array<int64_t, 4> generators)
      : kind_(kind), anchor_(anchor), generators_(generators) {}

  const Vec& exact(PointTable& points) {
    if (!has_exact_) {
      switch (kind_) {
        case Kind::kFace:
          exact_ = cross(points.difference(generators_[0], generators_[1]),
                         points.difference(generators_[0], generators_[2]));
          break;
        case Kind::kEdges:
          exact_ = cross(points.difference(generators_[0], generators_[1]),
                         points.difference(generators_[2], generators_[3]));
          break;
        case Kind::kInPlane:
          exact_ = cross(parent_->exact(points), points.difference(generators_[0], generators_[1]));
          break;
      }
      has_exact_ = true;
    }
    return exact_;
  }

  Kind kind_;
  int64_t anchor_;
  std::array<int64_t, 4> generators_;
  Plane* parent_ = nullptr;
  IntervalVector box_{};
  bool has_exact_ = false;
  Vec exact_{};
};

// At most four corners on each side of a plane.
struct PointSet {
  std::array<int64_t, 4> ids{};
  int size = 0;
  void add(int64_t id) { ids[static_cast<std::size_t>(size++)] = id; }
};

struct Separation {
  bool separated = false;
  PointSet first;   // first points on the plane
  PointSet second;  // second points on the plane
};

// The first points on one closed side and the second on the other, some
// point strictly off the plane (a zero normal decides no point off it).
Separation separate(PointTable& points, Plane& plane, const PointSet& first,
                    const PointSet& second) {
  std::array<int, 8> signs{};
  const int count = first.size + second.size;
  auto id = [&](int index) {
    return index < first.size ? first.ids[static_cast<std::size_t>(index)]
                              : second.ids[static_cast<std::size_t>(index - first.size)];
  };
  // Whether first <= 0 <= second (below) or second <= 0 <= first (above)
  // remain possible after the signs decided so far.
  auto feasible = [&](int index, int value, bool& below, bool& above) {
    if (value == kUnknown || value == 0) return;
    const bool in_first = index < first.size;
    if ((in_first && value > 0) || (!in_first && value < 0)) below = false;
    if ((in_first && value < 0) || (!in_first && value > 0)) above = false;
  };
  bool below = true;
  bool above = true;
  // Enclosures reject most planes before any exact arithmetic.
  for (int index = 0; index < count && (below || above); ++index) {
    signs[static_cast<std::size_t>(index)] = plane.enclosed_side(points, id(index));
    feasible(index, signs[static_cast<std::size_t>(index)], below, above);
  }
  for (int index = 0; index < count && (below || above); ++index) {
    int& value = signs[static_cast<std::size_t>(index)];
    if (value != kUnknown) continue;
    value = plane.exact_side(points, id(index));
    feasible(index, value, below, above);
  }
  Separation result;
  if (!(below || above)) return result;
  bool strict = false;
  for (int index = 0; index < count; ++index) {
    const int value = signs[static_cast<std::size_t>(index)];
    if (value != 0) {
      strict = true;
      continue;
    }
    if (index < first.size) {
      result.first.add(id(index));
    } else {
      result.second.add(id(index));
    }
  }
  result.separated = strict;
  return result;
}

struct Polygon {
  std::vector<int64_t> points;  // loop order, consistent with `normal`
  Vec normal;
};

// The pieces of one cell pair's declared stratum this proof decides.
class Stratum {
 public:
  std::vector<int64_t> vertices;
  std::vector<std::array<int64_t, 2>> edges;
  std::vector<Polygon> polygons;

  bool contains(PointTable& points, const PointSet& set) {
    std::array<int64_t, 4> key{-1, -1, -1, -1};
    std::copy(set.ids.begin(), set.ids.begin() + set.size, key.begin());
    std::sort(key.begin(), key.begin() + set.size);
    for (const auto& [known, verdict] : cache_) {
      if (known == key) return verdict;
    }
    const bool verdict = decide(points, set);
    cache_.emplace_back(key, verdict);
    return verdict;
  }

 private:
  // Pieces holding every point by identity first; geometric membership only
  // when none does.
  bool decide(PointTable& points, const PointSet& set) {
    const auto begin = set.ids.begin();
    const auto end = set.ids.begin() + set.size;
    for (const int64_t vertex : vertices) {
      if (std::all_of(begin, end, [&](int64_t point) { return point == vertex; })) return true;
    }
    for (const auto& edge : edges) {
      if (std::all_of(begin, end,
                      [&](int64_t point) { return point == edge[0] || point == edge[1]; })) {
        return true;
      }
    }
    for (const auto& polygon : polygons) {
      const auto& loop = polygon.points;
      if (std::all_of(begin, end, [&](int64_t point) {
            return std::find(loop.begin(), loop.end(), point) != loop.end();
          })) {
        return true;
      }
    }
    for (const auto& edge : edges) {
      if (std::all_of(begin, end,
                      [&](int64_t point) { return on_segment(points, point, edge[0], edge[1]); })) {
        return true;
      }
    }
    for (const auto& polygon : polygons) {
      if (std::all_of(begin, end,
                      [&](int64_t point) { return in_polygon(points, point, polygon); })) {
        return true;
      }
    }
    return false;
  }

  static bool on_segment(PointTable& points, int64_t point, int64_t first, int64_t second) {
    if (point == first || point == second) return true;
    const std::array<int64_t, 2> ends{first, second};
    if (points.outside(point, ends.data(), ends.size())) return false;
    const IntervalVector enclosed =
        iv::cross(points.box_difference(first, second), points.box_difference(first, point));
    for (const Interval& component : enclosed) {
      const int sign = interval_sign(component);
      if (sign == 1 || sign == -1) return false;
    }
    const Vec direction = points.difference(first, second);
    // A zero-length edge holds no point but its own (`point_on_segment3`).
    if (!nonzero(direction)) return false;
    const Vec offset = points.difference(first, point);
    if (nonzero(cross(direction, offset))) return false;
    return dot(direction, offset).sign() >= 0 &&
           dot(direction, points.difference(point, second)).sign() >= 0;
  }

  static bool in_polygon(PointTable& points, int64_t point, const Polygon& polygon) {
    const auto& loop = polygon.points;
    if (std::find(loop.begin(), loop.end(), point) != loop.end()) return true;
    if (points.outside(point, loop.data(), loop.size())) return false;
    if (dot(polygon.normal, points.difference(loop[0], point)).sign() != 0) return false;
    for (std::size_t index = 0; index < loop.size(); ++index) {
      const int64_t start = loop[index];
      const int64_t stop = loop[(index + 1) % loop.size()];
      const Vec turn = cross(points.difference(start, stop), points.difference(start, point));
      if (dot(turn, polygon.normal).sign() < 0) return false;
    }
    return true;
  }

  std::vector<std::pair<std::array<int64_t, 4>, bool>> cache_;
};

// One cell's corners, faces and tetrahedra.
struct CellView {
  const int64_t* corner_points;
  const int64_t* corner_vertices;
  int64_t corner_first;  // first corner row in the cell row bank
  int64_t scale_row;     // the cell's common denominator in the cell row bank
  int64_t corner_count;
  const int64_t* face_offsets;  // face_count + 1 offsets into face_corners
  const int64_t* face_corners;
  int64_t face_count;
  const int64_t* tets;  // 4 point ids each
  int64_t tet_count;
};

// Area vector of a loop of cell rows (the rational `area_vector`).
Vec area_vector(const std::vector<Vec>& rows) {
  Vec total{};
  for (std::size_t index = 0; index < rows.size(); ++index) {
    const Vec term = cross(rows[index], rows[(index + 1) % rows.size()]);
    total = {total[0] + term[0], total[1] + term[1], total[2] + term[2]};
  }
  return total;
}

// `_convex_planar_face_union((loop,))` is True for a loop of more than three
// distinct vertex identifiers: its area vector N is nonzero, its points lie
// in one plane, and from its least identifier every turn and fan triangle has
// a nonnegative component along N.
bool convex_loop(const std::vector<int64_t>& identifiers, const std::vector<Vec>& rows,
                 Vec& normal) {
  normal = area_vector(rows);
  if (!nonzero(normal)) return false;
  for (const Vec& row : rows) {
    if (dot(sub(row, rows[0]), normal).sign() != 0) return false;
  }
  const std::size_t count = rows.size();
  const std::size_t start = static_cast<std::size_t>(
      std::min_element(identifiers.begin(), identifiers.end()) - identifiers.begin());
  auto at = [&](std::size_t index) -> const Vec& { return rows[(start + index) % count]; };
  for (std::size_t index = 0; index < count; ++index) {
    const Vec turn = cross(sub(at(index), at(index + count - 1)), sub(at(index + 1), at(index)));
    if (dot(turn, normal).sign() < 0) return false;
  }
  for (std::size_t index = 1; index + 1 < count; ++index) {
    const Vec fan = cross(sub(at(index), at(0)), sub(at(index + 1), at(0)));
    if (dot(fan, normal).sign() < 0) return false;
  }
  return true;
}

// The face loops of a cell within `shared`, keyed by their sorted vertex
// identifier sets (the first loop of each key, as `_shared_loops`).
struct SharedLoop {
  std::vector<int64_t> key;
  int64_t face;
};

std::vector<SharedLoop> shared_loops(const CellView& cell, const std::vector<int64_t>& shared) {
  std::vector<SharedLoop> loops;
  for (int64_t face = 0; face < cell.face_count; ++face) {
    std::vector<int64_t> key;
    bool inside = true;
    for (int64_t entry = cell.face_offsets[face]; entry < cell.face_offsets[face + 1]; ++entry) {
      const int64_t vertex = cell.corner_vertices[cell.face_corners[entry]];
      if (!std::binary_search(shared.begin(), shared.end(), vertex)) {
        inside = false;
        break;
      }
      key.push_back(vertex);
    }
    if (!inside) continue;
    std::sort(key.begin(), key.end());
    key.erase(std::unique(key.begin(), key.end()), key.end());
    if (key.size() < 3) continue;
    if (std::none_of(loops.begin(), loops.end(),
                     [&](const SharedLoop& loop) { return loop.key == key; })) {
      loops.push_back({std::move(key), face});
    }
  }
  return loops;
}

std::vector<std::array<int64_t, 2>> loop_edges(const CellView& cell,
                                               const std::vector<int64_t>& shared) {
  std::vector<std::array<int64_t, 2>> edges;
  for (int64_t face = 0; face < cell.face_count; ++face) {
    const int64_t first = cell.face_offsets[face];
    const int64_t size = cell.face_offsets[face + 1] - first;
    for (int64_t index = 0; index < size; ++index) {
      int64_t a = cell.corner_vertices[cell.face_corners[first + index]];
      int64_t b = cell.corner_vertices[cell.face_corners[first + (index + 1) % size]];
      if (a == b || !std::binary_search(shared.begin(), shared.end(), a) ||
          !std::binary_search(shared.begin(), shared.end(), b)) {
        continue;
      }
      if (b < a) std::swap(a, b);
      edges.push_back({a, b});
    }
  }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  return edges;
}

int64_t corner_of(const CellView& cell, int64_t vertex) {
  for (int64_t corner = 0; corner < cell.corner_count; ++corner) {
    if (cell.corner_vertices[corner] == vertex) return corner;
  }
  return -1;
}

// The triangles `polytope_face_triangles` assigns a non-planar loop: the fan
// (c, P_i, P_i+1) from its vertex average c, when every spoke turns forward
// about the loop's area vector and the spokes wind once (the refusals of
// `polytope_face_routes`). `rows` are the loop's cell rows over `scale`; the
// apex takes the identity of an equal point of `cell`'s tetrahedra.
void fan_triangles(PointTable& points, const CellView& cell, const std::vector<Vec>& rows,
                   const std::vector<int64_t>& loop, const Dyadic& scale, Stratum& stratum) {
  const std::size_t count = rows.size();
  Vec normal{};
  for (std::size_t index = 0; index < count; ++index) {
    const Vec term = cross(sub(rows[index], rows[0]), sub(rows[(index + 1) % count], rows[0]));
    normal = {normal[0] + term[0], normal[1] + term[1], normal[2] + term[2]};
  }
  if (!nonzero(normal)) return;
  if (std::all_of(rows.begin(), rows.end(),
                  [&](const Vec& row) { return dot(normal, sub(row, rows[0])).sign() == 0; })) {
    return;  // planar: the loop's ear-clipped triangles, not modeled here
  }
  const Dyadic size(static_cast<double>(count));
  Vec sum{};
  for (const Vec& row : rows) sum = {sum[0] + row[0], sum[1] + row[1], sum[2] + row[2]};
  // Spokes P_i - c scaled by the vertex count.
  std::vector<Vec> spokes;
  for (const Vec& row : rows) {
    spokes.push_back(sub({row[0] * size, row[1] * size, row[2] * size}, sum));
  }
  auto turn = [&](const Vec& u, const Vec& w) { return dot(normal, cross(u, w)).sign(); };
  for (std::size_t index = 0; index < count; ++index) {
    if (turn(spokes[index], spokes[(index + 1) % count]) <= 0) return;
  }
  const Vec& start = spokes[0];
  const Dyadic norm2 = dot(normal, normal);
  auto half = [&](const Vec& u) {
    const int side = turn(start, u);
    if (side != 0) return side > 0 ? 0 : 1;
    const int along = (dot(start, u) * norm2 - dot(start, normal) * dot(u, normal)).sign();
    return along > 0 ? 0 : 1;
  };
  // Every consecutive turn is forward, so `before(u_i, u_i+1)` of the
  // rational loop is half(u_i) <= half(u_i+1).
  std::vector<int> halves;
  for (const Vec& spoke : spokes) halves.push_back(half(spoke));
  std::size_t wraps = 0;
  for (std::size_t index = 0; index < count; ++index) {
    wraps += halves[index] > halves[(index + 1) % count] ? 1 : 0;
  }
  if (wraps != 1) return;
  const int64_t centre = points.derive({sum, size * scale}, cell.tets, 4 * cell.tet_count);
  for (std::size_t index = 0; index < count; ++index) {
    const std::size_t next = (index + 1) % count;
    stratum.polygons.push_back(
        {{centre, loop[index], loop[next]}, cross(spokes[index], spokes[next])});
  }
}

// Shared vertices, edges and decided face loops of the pair, at the first
// cell's corner images (the rational `point_by_id`).
Stratum declared_stratum(PointTable& points, const CellView& first, const CellView& second,
                         const IntegerBank& rows) {
  Stratum stratum;
  std::vector<int64_t> shared;
  for (int64_t corner = 0; corner < first.corner_count; ++corner) {
    if (corner_of(second, first.corner_vertices[corner]) >= 0) {
      shared.push_back(first.corner_vertices[corner]);
    }
  }
  std::sort(shared.begin(), shared.end());
  if (shared.empty()) return stratum;
  auto point = [&](int64_t vertex) { return first.corner_points[corner_of(first, vertex)]; };
  for (const int64_t vertex : shared) stratum.vertices.push_back(point(vertex));
  const auto first_edges = loop_edges(first, shared);
  const auto second_edges = loop_edges(second, shared);
  for (const auto& edge : first_edges) {
    if (std::binary_search(second_edges.begin(), second_edges.end(), edge)) {
      stratum.edges.push_back({point(edge[0]), point(edge[1])});
    }
  }
  const auto second_loops = shared_loops(second, shared);
  for (const auto& loop : shared_loops(first, shared)) {
    if (std::none_of(second_loops.begin(), second_loops.end(),
                     [&](const SharedLoop& other) { return other.key == loop.key; })) {
      continue;
    }
    const int64_t begin = first.face_offsets[loop.face];
    const int64_t size = first.face_offsets[loop.face + 1] - begin;
    if (size != static_cast<int64_t>(loop.key.size())) continue;  // repeated identifiers
    std::vector<int64_t> identifiers;
    std::vector<Vec> loop_rows;
    Polygon polygon;
    for (int64_t entry = begin; entry < begin + size; ++entry) {
      const int64_t corner = first.face_corners[entry];
      const int64_t row = first.corner_first + corner;
      identifiers.push_back(first.corner_vertices[corner]);
      loop_rows.push_back({rows.value(3 * row), rows.value(3 * row + 1), rows.value(3 * row + 2)});
      polygon.points.push_back(first.corner_points[corner]);
    }
    const bool decided =
        size > 3 ? convex_loop(identifiers, loop_rows, polygon.normal)
                 : nonzero(polygon.normal = area_vector(loop_rows));
    if (decided) {
      stratum.polygons.push_back(std::move(polygon));
    } else if (size > 3) {
      fan_triangles(points, first, loop_rows, polygon.points, rows.value(first.scale_row),
                    stratum);
    }
  }
  return stratum;
}

PointSet tet_points(const int64_t* tet) {
  PointSet set;
  for (int corner = 0; corner < 4; ++corner) set.add(tet[corner]);
  return set;
}

// Inside the plane of `plane`, lines through the edges of either on-plane
// set; on a common line the overlap of the two segments.
bool narrowed_contact(PointTable& points, Plane& plane, const Separation& contact,
                      Stratum& stratum) {
  std::vector<std::array<int64_t, 2>> lines;
  for (const PointSet* set : {&contact.first, &contact.second}) {
    for (int a = 0; a < set->size; ++a) {
      for (int b = a + 1; b < set->size; ++b) {
        lines.push_back(
            {set->ids[static_cast<std::size_t>(a)], set->ids[static_cast<std::size_t>(b)]});
      }
    }
  }
  for (const auto& line : lines) {
    Plane inner = Plane::in_plane(points, plane, line[0], line[1]);
    const Separation narrow = separate(points, inner, contact.first, contact.second);
    if (!narrow.separated) continue;
    if (narrow.first.size == 0 || narrow.second.size == 0) return true;
    if (stratum.contains(points, narrow.first) || stratum.contains(points, narrow.second)) {
      return true;
    }
    if (narrow.first.size < 2 || narrow.second.size < 2) continue;
    // Collinear segments meet in the hull of the corners within the other.
    const Vec direction = points.difference(narrow.first.ids[0], narrow.first.ids[1]);
    auto within = [&](int64_t point, const PointSet& set) {
      bool lower = false;
      bool upper = false;
      for (int index = 0; index < set.size; ++index) {
        const int value =
            dot(direction, points.difference(set.ids[static_cast<std::size_t>(index)], point))
                .sign();
        lower = lower || value >= 0;
        upper = upper || value <= 0;
      }
      return lower && upper;
    };
    PointSet overlap;
    for (const auto [own, other] : {std::pair{&narrow.first, &narrow.second},
                                    std::pair{&narrow.second, &narrow.first}}) {
      for (int index = 0; index < own->size; ++index) {
        const int64_t point = own->ids[static_cast<std::size_t>(index)];
        if (overlap.size < 4 && within(point, *other) &&
            std::find(overlap.ids.begin(), overlap.ids.begin() + overlap.size, point) ==
                overlap.ids.begin() + overlap.size) {
          overlap.add(point);
        }
      }
    }
    if (overlap.size > 0 && stratum.contains(points, overlap)) return true;
  }
  return false;
}

// A tetrahedron's corners, outward box, face normal enclosures, and the
// midpoint estimates that only order the planes the proof tries.
struct Tetrahedron {
  const int64_t* corners;
  std::array<double, 6> box;
  std::array<IntervalVector, 4> normals;
  std::array<std::array<double, 3>, 4> centers;
  // Unit estimates of the face normals pointing away from the omitted corner.
  std::array<std::array<double, 3>, 4> outward;
  // A nonzero exact orientation.
  bool nondegenerate;
};

Tetrahedron prepared_tetrahedron(PointTable& points, const int64_t* corners) {
  Tetrahedron tet{corners, {}, {}, {}, {}, false};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    tet.box[axis] = std::numeric_limits<double>::infinity();
    tet.box[axis + 3] = -std::numeric_limits<double>::infinity();
  }
  for (std::size_t corner = 0; corner < 4; ++corner) {
    const IntervalVector enclosure = points.box(corners[corner]);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      tet.box[axis] = std::min(tet.box[axis], enclosure[axis].lower);
      tet.box[axis + 3] = std::max(tet.box[axis + 3], enclosure[axis].upper);
      tet.centers[corner][axis] = 0.5 * enclosure[axis].lower + 0.5 * enclosure[axis].upper;
    }
  }
  for (std::size_t face = 0; face < 4; ++face) {
    const int* route = kFaceCorners[face];
    tet.normals[face] = iv::cross(points.box_difference(corners[route[0]], corners[route[1]]),
                                  points.box_difference(corners[route[0]], corners[route[2]]));
    std::array<double, 3> normal{};
    double apex = 0.0;
    double norm = 0.0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      normal[axis] = 0.5 * tet.normals[face][axis].lower + 0.5 * tet.normals[face][axis].upper;
      apex += normal[axis] * (tet.centers[face][axis] -
                              tet.centers[static_cast<std::size_t>(route[0])][axis]);
      norm += normal[axis] * normal[axis];
    }
    const double scale = (apex > 0.0 ? -1.0 : 1.0) / std::sqrt(norm);
    for (std::size_t axis = 0; axis < 3; ++axis) tet.outward[face][axis] = normal[axis] * scale;
  }
  Plane base = Plane::face(corners[1], corners[2], corners[3], tet.normals[0]);
  int orientation = base.enclosed_side(points, corners[0]);
  if (orientation == kUnknown) orientation = base.exact_side(points, corners[0]);
  tet.nondegenerate = orientation != 0;
  return tet;
}

// Least estimated clearance of `other`'s corners beyond face `face` of `tet`.
double clearance(const Tetrahedron& tet, std::size_t face, const Tetrahedron& other) {
  const auto& origin = tet.centers[static_cast<std::size_t>(kFaceCorners[face][0])];
  double least = std::numeric_limits<double>::infinity();
  for (const auto& corner : other.centers) {
    double value = 0.0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      value += tet.outward[face][axis] * (corner[axis] - origin[axis]);
    }
    least = std::min(least, value);
  }
  // A NaN estimate (an unbounded corner) orders last.
  return std::isnan(least) ? -std::numeric_limits<double>::infinity() : least;
}

// Strictly separated outward boxes, or a box no comparison certifies (NaN).
bool boxes_apart(const std::array<double, 6>& a, const std::array<double, 6>& b) {
  for (std::size_t axis = 0; axis < 3; ++axis) {
    if (a[axis + 3] < b[axis] || b[axis + 3] < a[axis]) return true;
  }
  return false;
}

// The closed contact of the two tetrahedra lies in one piece of `stratum`.
bool declared_tetrahedron_contact(PointTable& points, const Tetrahedron& first,
                                  const Tetrahedron& second, Stratum& stratum) {
  const PointSet left = tet_points(first.corners);
  const PointSet right = tet_points(second.corners);
  auto proves = [&](Plane& plane) {
    const Separation contact = separate(points, plane, left, right);
    if (!contact.separated) return false;
    if (contact.first.size == 0 || contact.second.size == 0) return true;
    return stratum.contains(points, contact.first) || stratum.contains(points, contact.second) ||
           narrowed_contact(points, plane, contact, stratum);
  };
  // Face planes in decreasing estimated clearance of the other tetrahedron.
  std::array<std::pair<double, std::size_t>, 8> order{};
  for (std::size_t face = 0; face < 4; ++face) {
    order[face] = {clearance(first, face, second), face};
    order[face + 4] = {clearance(second, face, first), face + 4};
  }
  for (std::size_t index = 1; index < order.size(); ++index) {
    for (std::size_t slot = index; slot > 0 && order[slot - 1].first < order[slot].first; --slot) {
      std::swap(order[slot - 1], order[slot]);
    }
  }
  for (const auto& [estimate, index] : order) {
    const Tetrahedron& tet = index < 4 ? first : second;
    const std::size_t face = index % 4;
    const int* route = kFaceCorners[face];
    Plane plane = Plane::face(tet.corners[route[0]], tet.corners[route[1]],
                              tet.corners[route[2]], tet.normals[face]);
    if (proves(plane)) return true;
  }
  const int64_t* a_corners = first.corners;
  const int64_t* b_corners = second.corners;
  for (const auto& a : kEdgeCorners) {
    for (const auto& b : kEdgeCorners) {
      Plane through_first =
          Plane::edges(points, a_corners[a[0]], a_corners[a[1]], b_corners[b[0]], b_corners[b[1]]);
      if (proves(through_first)) return true;
      Plane through_second =
          Plane::edges(points, b_corners[b[0]], b_corners[b[1]], a_corners[a[0]], a_corners[a[1]]);
      if (proves(through_second)) return true;
    }
  }
  return false;
}

// Prepared tetrahedra of the cells one call visits.
class TetrahedronCache {
 public:
  const std::vector<Tetrahedron>& of(PointTable& points, int64_t cell, const CellView& view) {
    auto found = cells_.find(cell);
    if (found == cells_.end()) {
      std::vector<Tetrahedron> tets;
      for (int64_t tet = 0; tet < view.tet_count; ++tet) {
        tets.push_back(prepared_tetrahedron(points, view.tets + 4 * tet));
      }
      found = cells_.emplace(cell, std::move(tets)).first;
    }
    return found->second;
  }

 private:
  std::unordered_map<int64_t, std::vector<Tetrahedron>> cells_;
};

bool declared_pair(PointTable& points, const IntegerBank& rows, const CellView& first,
                   const CellView& second, const std::vector<Tetrahedron>& first_tets,
                   const std::vector<Tetrahedron>& second_tets) {
  // The rational contact pass refuses a zero-volume piece of either cell.
  for (const auto* tets : {&first_tets, &second_tets}) {
    if (std::any_of(tets->begin(), tets->end(),
                    [](const Tetrahedron& tet) { return !tet.nondegenerate; })) {
      return false;
    }
  }
  Stratum stratum;
  bool prepared = false;
  for (const Tetrahedron& left : first_tets) {
    for (const Tetrahedron& right : second_tets) {
      if (boxes_apart(left.box, right.box)) continue;
      if (!prepared) {
        stratum = declared_stratum(points, first, second, rows);
        prepared = true;
      }
      if (!declared_tetrahedron_contact(points, left, right, stratum)) return false;
    }
  }
  return true;
}

}  // namespace

extern "C" int32_t phx_mc_polyhedral_declared_contacts(
    int64_t point_count, const int64_t* point_word_offsets, const uint32_t* point_words,
    const int8_t* point_signs, const double* point_lower, const double* point_upper,
    const int64_t* tet_points, int64_t cell_count, const int64_t* cell_tet_offsets,
    const int64_t* cell_corner_offsets, const int64_t* corner_points,
    const int64_t* corner_vertices, const int64_t* row_word_offsets, const uint32_t* row_words,
    const int8_t* row_signs, const int64_t* cell_face_offsets, const int64_t* face_offsets,
    const int64_t* face_corners, int64_t pair_count,
    const int64_t* pairs, int8_t* declared) {
  return phx::mc::guarded([&]() -> int32_t {
    if (point_count < 0 || cell_count < 0 || pair_count < 0 || cell_tet_offsets == nullptr ||
        cell_corner_offsets == nullptr || cell_face_offsets == nullptr ||
        face_offsets == nullptr ||
        (pair_count > 0 && (pairs == nullptr || declared == nullptr)) ||
        (point_count > 0 && (point_lower == nullptr || point_upper == nullptr)) ||
        !valid_bank(4 * point_count, point_word_offsets, point_words, point_signs) ||
        !valid_bank(3 * cell_corner_offsets[cell_count] + cell_count, row_word_offsets,
                    row_words, row_signs)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (cell_tet_offsets[0] != 0 || cell_corner_offsets[0] != 0 || cell_face_offsets[0] != 0 ||
        face_offsets[0] != 0) {
      return PHX_MC_INVALID_INPUT;
    }
    for (int64_t cell = 0; cell < cell_count; ++cell) {
      if (cell_tet_offsets[cell + 1] < cell_tet_offsets[cell] ||
          cell_corner_offsets[cell + 1] < cell_corner_offsets[cell] ||
          cell_face_offsets[cell + 1] < cell_face_offsets[cell]) {
        return PHX_MC_INVALID_INPUT;
      }
      const int64_t corners = cell_corner_offsets[cell + 1] - cell_corner_offsets[cell];
      for (int64_t face = cell_face_offsets[cell]; face < cell_face_offsets[cell + 1]; ++face) {
        if (face_offsets[face + 1] < face_offsets[face]) return PHX_MC_INVALID_INPUT;
        for (int64_t entry = face_offsets[face]; entry < face_offsets[face + 1]; ++entry) {
          if (face_corners[entry] < 0 || face_corners[entry] >= corners) {
            return PHX_MC_INVALID_INPUT;
          }
        }
      }
    }
    for (int64_t value = 0; value < 4 * cell_tet_offsets[cell_count]; ++value) {
      if (tet_points[value] < 0 || tet_points[value] >= point_count) return PHX_MC_INVALID_INPUT;
    }
    for (int64_t value = 0; value < cell_corner_offsets[cell_count]; ++value) {
      if (corner_points[value] < 0 || corner_points[value] >= point_count) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    for (int64_t value = 0; value < 2 * pair_count; ++value) {
      if (pairs[value] < 0 || pairs[value] >= cell_count) return PHX_MC_INVALID_INPUT;
    }
    const phx::mc::GradualNearestScope mode;
    PointTable points(IntegerBank{point_word_offsets, point_words, point_signs}, point_count,
                      point_lower, point_upper);
    const IntegerBank rows{row_word_offsets, row_words, row_signs};
    auto view = [&](int64_t cell) {
      const int64_t corner_first = cell_corner_offsets[cell];
      const int64_t face_first = cell_face_offsets[cell];
      return CellView{corner_points + corner_first,
                      corner_vertices + corner_first,
                      corner_first,
                      3 * cell_corner_offsets[cell_count] + cell,
                      cell_corner_offsets[cell + 1] - corner_first,
                      face_offsets + face_first,
                      face_corners,
                      cell_face_offsets[cell + 1] - face_first,
                      tet_points + 4 * cell_tet_offsets[cell],
                      cell_tet_offsets[cell + 1] - cell_tet_offsets[cell]};
    };
    TetrahedronCache tetrahedra;
    for (int64_t pair = 0; pair < pair_count; ++pair) {
      const CellView first = view(pairs[2 * pair]);
      const CellView second = view(pairs[2 * pair + 1]);
      declared[pair] =
          first.tet_count > 0 && second.tet_count > 0 &&
          declared_pair(points, rows, first, second,
                        tetrahedra.of(points, pairs[2 * pair], first),
                        tetrahedra.of(points, pairs[2 * pair + 1], second));
    }
    return PHX_MC_OK;
  });
}
