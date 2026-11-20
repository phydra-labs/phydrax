// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "surface_adapt.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "phydrax_meshcore.h"

namespace phx::mc::remesh {
namespace {

constexpr double kLong = 4.0 / 3.0;      // split above this multiple of the size
constexpr double kShort = 4.0 / 5.0;     // collapse below this multiple
constexpr double kNormalCosine = 0.5;    // created normals within 60 degrees of the source
// Sizes keep a well-shaped triangle's sagitta within this fraction of the
// tolerance, and the fidelity sweep splits triangles sampled above the
// second: the certificate after construction then decides most triangles
// without subdividing them.
constexpr double kSizeMargin = 0.7;
constexpr double kRepairMargin = 0.8;
// An input edge whose two triangles turn by more than 45 degrees is a crease
// of the source (such as a CSG edge). Every input triangle touching a crease
// vertex is kept exactly as the input has it: the first-order certificate
// cannot decide transversality next to a crease (the gradient enclosure
// spans both sides), so those triangles keep the carrier's own evidence.
constexpr double kCreaseCosine = 0.70710678118654752;
constexpr int kSizeSweeps = 5;
constexpr int kQualitySweeps = 8;
constexpr int kFidelitySweeps = 12;
// Certification rounds: the first, and up to three after local repairs.
constexpr int kRepairRounds = 4;
constexpr double kInfinity = std::numeric_limits<double>::infinity();

inline Point sub(const Point& a, const Point& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline Point add(const Point& a, const Point& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline Point scale(const Point& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
inline double dot(const Point& a, const Point& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline Point cross(const Point& a, const Point& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline double norm2(const Point& a) { return dot(a, a); }

// With corners on a surface of principal curvatures k1, k2, a triangle of
// side L deviates from it by about kappa L^2 / 8, kappa = max(|k|_max,
// (2/3)|k1 + k2|): the edge midpoints by the normal curvature along the edge
// times L^2 / 8, the centroid of an equilateral triangle by |k1 + k2| L^2 / 12
// (a sphere: k L^2 / 6). On a saddle the deviation has no interior extremum,
// so the edges bound it, and (2/3)|k1 + k2| < |k|_max there.
inline double effective_curvature(double first, double second) {
  return std::max(std::max(std::abs(first), std::abs(second)),
                  (2.0 / 3.0) * std::abs(first + second));
}

// Sine of the smallest angle of a triangle (0 for a degenerate one). Over a
// star, the smallest angle is the angle of the smallest sine (asin is
// monotone), so one asin serves the whole star.
double minimum_sine(const Point& a, const Point& b, const Point& c) {
  const double ab = norm2(sub(b, a)), bc = norm2(sub(c, b)), ca = norm2(sub(a, c));
  const double twice_area = std::sqrt(norm2(cross(sub(b, a), sub(c, a))));
  // The smallest angle is opposite the shortest side, between the other two.
  double l1 = 0, l2 = 0;
  if (ab <= bc && ab <= ca) {
    l1 = bc;
    l2 = ca;
  } else if (bc <= ab && bc <= ca) {
    l1 = ab;
    l2 = ca;
  } else {
    l1 = ab;
    l2 = bc;
  }
  const double denominator = std::sqrt(l1 * l2);
  if (!(denominator > 0) || !(twice_area > 0)) return 0;
  return std::min(1.0, twice_area / denominator);
}

// The angle in degrees of a smallest-angle sine. The angle between the two
// longest sides is at most 60 degrees, so asin is exact here.
double sine_degrees(double s) { return std::asin(s) * (180.0 / 3.141592653589793); }

// Smallest angle of a triangle in degrees (0 for a degenerate one).
double minimum_angle(const Point& a, const Point& b, const Point& c) {
  return sine_degrees(minimum_sine(a, b, c));
}

// Free vertices are projected, moved and removed; kept vertices (corners of
// kept triangles) stay bitwise where the input put them; crease vertices are
// kept, and their source normal belongs to one side of the crease only.
enum class Role : uint8_t { kFree = 0, kKept = 1, kCrease = 2 };

class Adapter {
 public:
  Adapter(const SurfaceOracle& oracle, const AdaptTargets& targets, AdaptCounters& counters)
      : oracle_(oracle), targets_(targets), counters_(counters) {}

  // `place` projects the free vertices onto the source; without it every
  // vertex keeps its position (a surface constructed before, to repair).
  // `twins` (closed_manifold's, for these triangles) skips that check.
  int32_t load(const std::vector<Point>& points, const std::vector<Triangle>& triangles,
               bool place, const std::vector<int64_t>* twins = nullptr);
  void run();
  // Splits the longest edge that is not kept of every flagged triangle (rows
  // as loaded), then restores angles by Delaunay flips; returns the splits.
  int64_t repair(const std::vector<uint8_t>& refused);
  // Relocates the corners of every triangle below the minimum angle while
  // that lifts their stars (as improve_quality, without collapses or
  // splits), with Delaunay flips; returns the relocations.
  int64_t relax();
  void store(std::vector<Point>& points, std::vector<Triangle>& triangles) const;

 private:
  // ---- topology
  static int32_t next(int32_t c) { return c - c % 3 + (c % 3 + 1) % 3; }
  static int32_t prev(int32_t c) { return c - c % 3 + (c % 3 + 2) % 3; }
  bool alive_triangle(int32_t t) const { return corners_[3 * static_cast<size_t>(t)] >= 0; }
  int32_t head(int32_t c) const { return corners_[static_cast<size_t>(next(c))]; }
  int32_t tail(int32_t c) const { return corners_[static_cast<size_t>(c)]; }
  int32_t twin(int32_t c) const { return opposite_[static_cast<size_t>(c)]; }
  void link(int32_t a, int32_t b) {
    opposite_[static_cast<size_t>(a)] = b;
    opposite_[static_cast<size_t>(b)] = a;
  }
  const Point& position(int32_t v) const { return positions_[static_cast<size_t>(v)]; }
  Role role(int32_t v) const { return roles_[static_cast<size_t>(v)]; }
  bool kept_edge(int32_t c) const { return kept_edges_[static_cast<size_t>(c)] != 0; }
  // Outgoing half-edges of v, counterclockwise.
  template <typename F>
  void for_ring(int32_t v, F&& visit) const {
    const int32_t first = vertex_corner_[static_cast<size_t>(v)];
    int32_t c = first;
    do {
      visit(c);
      c = twin(prev(c));
    } while (c != first);
  }
  int valence(int32_t v) const { return valences_[static_cast<size_t>(v)]; }
  int32_t new_vertex(const Point& p, const Point& normal, double size);
  int32_t new_triangle();
  // `kappa` is the effective curvature of a well-shaped triangle's sagitta
  // (see effective_curvature): its edges up to the split threshold keep it
  // within the size margin.
  double size_at(double kappa) const {
    const double limit =
        kappa > 0 ? std::sqrt(8.0 * kSizeMargin * targets_.tolerance / kappa) / kLong
                  : targets_.edge_length;
    return std::min(targets_.edge_length, limit);
  }
  double edge_size(int32_t a, int32_t b) const {
    return std::min(sizes_[static_cast<size_t>(a)], sizes_[static_cast<size_t>(b)]);
  }
  // A created triangle (p, q, r) is acceptable when its normal is within 60
  // degrees of the source normal at each corner that is not a crease vertex.
  bool oriented(const std::array<Point, 3>& p, const std::array<const Point*, 3>& normals) const {
    const Point n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
    const double length = std::sqrt(norm2(n));
    if (!(length > 0)) return false;
    for (const Point* normal : normals)
      if (normal != nullptr && !(dot(n, *normal) > kNormalCosine * length)) return false;
    return true;
  }
  const Point* judged_normal(int32_t v) const {
    return role(v) == Role::kCrease ? nullptr : &normals_[static_cast<size_t>(v)];
  }
  bool project(const Point& p, Point& on, Point& normal, double& kappa) const {
    SurfacePoint found;
    if (!oracle_.project(p, found)) return false;
    on = found.on;
    normal = found.normal;
    kappa = effective_curvature(found.curvature_first, found.curvature_second);
    return true;
  }

  // ---- edits
  bool split(int32_t h);
  bool collapse(int32_t h, bool require_quality_gain, bool at_endpoint = false);
  bool flip(int32_t h);
  bool relocate(int32_t v, const Point& goal, bool require_quality_gain);
  // Mean apex of the equilateral triangles on the star's opposite edges.
  Point equilateral_goal(int32_t v) const;
  bool flip_improves_valence(int32_t h) const;
  bool flip_is_delaunay_better(int32_t h) const;
  double star_minimum_angle(int32_t v) const;
  Point ring_centroid(int32_t v) const;

  // ---- passes
  void split_long();
  void collapse_short();
  void flip_valence();
  void flip_delaunay();
  void smooth();
  int64_t improve_quality();
  int64_t repair_fidelity();
  // Splits the longest edge of triangle t that is not kept; false when none
  // is or the split is refused.
  bool split_longest(int32_t t);
  // Splits the longest edge of every listed triangle still holding the
  // listed corners (an earlier split may reshape one); returns the splits.
  int64_t split_listed(const std::vector<std::pair<int32_t, std::array<int32_t, 3>>>& listed);
  // The largest sampled distance of each listed triangle from the surface.
  void deviations(const std::vector<int32_t>& triangles, std::vector<double>& out) const;
  std::vector<int32_t> alive_triangles() const;

  const SurfaceOracle& oracle_;
  const AdaptTargets& targets_;
  AdaptCounters& counters_;
  std::vector<Point> positions_, normals_;
  std::vector<double> sizes_;
  std::vector<Role> roles_;
  std::vector<int32_t> vertex_corner_;  // an outgoing half-edge, -1 when removed
  std::vector<int32_t> corners_;        // 3 per triangle, -1 when removed
  std::vector<int32_t> opposite_;       // twin half-edge of every corner
  std::vector<uint8_t> kept_edges_;  // edges of kept triangles, per half-edge, equal on twins
  std::vector<int32_t> free_triangles_;
  std::vector<int32_t> free_vertices_;
  // Outgoing half-edges of every vertex (its ring size), kept by every edit.
  std::vector<int32_t> valences_;
  // Per-vertex marks of the collapse link test; a vertex is marked when its
  // mark equals stamp_.
  std::vector<uint32_t> marks_;
  uint32_t stamp_ = 0;
};

int32_t Adapter::new_vertex(const Point& p, const Point& normal, double size) {
  if (!free_vertices_.empty()) {
    const int32_t v = free_vertices_.back();
    free_vertices_.pop_back();
    positions_[static_cast<size_t>(v)] = p;
    normals_[static_cast<size_t>(v)] = normal;
    sizes_[static_cast<size_t>(v)] = size;
    roles_[static_cast<size_t>(v)] = Role::kFree;
    return v;
  }
  positions_.push_back(p);
  normals_.push_back(normal);
  sizes_.push_back(size);
  roles_.push_back(Role::kFree);
  vertex_corner_.push_back(-1);
  valences_.push_back(0);
  return static_cast<int32_t>(positions_.size() - 1);
}

int32_t Adapter::new_triangle() {
  if (!free_triangles_.empty()) {
    const int32_t t = free_triangles_.back();
    free_triangles_.pop_back();
    return t;
  }
  corners_.insert(corners_.end(), 3, -1);
  opposite_.insert(opposite_.end(), 3, -1);
  kept_edges_.insert(kept_edges_.end(), 3, 0);
  return static_cast<int32_t>(corners_.size() / 3 - 1);
}

int32_t Adapter::load(const std::vector<Point>& points, const std::vector<Triangle>& triangles,
                      bool place, const std::vector<int64_t>* twins) {
  const size_t n = points.size(), m = triangles.size();
  if (n > static_cast<size_t>(INT32_MAX / 4) || m > static_cast<size_t>(INT32_MAX / 12))
    return PHX_MC_CAPACITY_EXCEEDED;
  std::vector<int64_t> computed;
  if (twins == nullptr) {
    if (!closed_manifold(triangles, n, &computed)) return PHX_MC_INVALID_INPUT;
    twins = &computed;
  } else if (twins->size() != 3 * m) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  corners_.resize(3 * m);
  opposite_.resize(3 * m);
  kept_edges_.assign(3 * m, 0);
  vertex_corner_.assign(n, -1);
  valences_.assign(n, 0);
  for (size_t c = 0; c < 3 * m; ++c) {
    corners_[c] = static_cast<int32_t>(triangles[c / 3][c % 3]);
    opposite_[c] = static_cast<int32_t>((*twins)[c]);
    vertex_corner_[static_cast<size_t>(corners_[c])] = static_cast<int32_t>(c);
    ++valences_[static_cast<size_t>(corners_[c])];
  }
  positions_ = points;
  normals_.assign(n, Point{0, 0, 0});
  sizes_.assign(n, targets_.edge_length);
  roles_.assign(n, Role::kFree);
  // Crease edges by the input's own turn.
  for (size_t c = 0; c < 3 * m && targets_.keep_creases; ++c) {
    const auto h = static_cast<int32_t>(c);
    const int32_t g = twin(h);
    if (g < h) continue;
    const Point first = cross(sub(points[static_cast<size_t>(corners_[3 * (c / 3) + 1])],
                                  points[static_cast<size_t>(corners_[3 * (c / 3)])]),
                              sub(points[static_cast<size_t>(corners_[3 * (c / 3) + 2])],
                                  points[static_cast<size_t>(corners_[3 * (c / 3)])]));
    const size_t base = 3 * static_cast<size_t>(g / 3);
    const Point second =
        cross(sub(points[static_cast<size_t>(corners_[base + 1])], points[static_cast<size_t>(corners_[base])]),
              sub(points[static_cast<size_t>(corners_[base + 2])], points[static_cast<size_t>(corners_[base])]));
    const double lengths = std::sqrt(norm2(first) * norm2(second));
    if (!(dot(first, second) < kCreaseCosine * lengths)) continue;
    roles_[static_cast<size_t>(tail(h))] = roles_[static_cast<size_t>(head(h))] = Role::kCrease;
  }
  // Triangles touching a crease vertex are kept: their corners and edges.
  for (size_t t = 0; t < m; ++t) {
    const size_t base = 3 * t;
    bool touches = false;
    for (size_t k = 0; k < 3; ++k)
      touches = touches || roles_[static_cast<size_t>(corners_[base + k])] == Role::kCrease;
    if (!touches) continue;
    for (size_t k = 0; k < 3; ++k) {
      auto& corner_role = roles_[static_cast<size_t>(corners_[base + k])];
      if (corner_role == Role::kFree) corner_role = Role::kKept;
      kept_edges_[base + k] = kept_edges_[static_cast<size_t>(opposite_[base + k])] = 1;
    }
  }
  for (size_t v = 0; v < n; ++v)
    if (vertex_corner_[v] < 0) free_vertices_.push_back(static_cast<int32_t>(v));
  // The used vertices in order: the points themselves when all are used.
  std::vector<Point> used;
  if (!free_vertices_.empty())
    for (size_t v = 0; v < n; ++v)
      if (vertex_corner_[v] >= 0) used.push_back(points[v]);
  std::vector<SurfacePoint> projected;
  std::vector<uint8_t> found;
  oracle_.project_all(free_vertices_.empty() ? points : used, projected, found);
  size_t at = 0;
  for (size_t v = 0; v < n; ++v) {
    if (vertex_corner_[v] < 0) continue;
    if (!found[at]) return PHX_MC_UNRESOLVED;
    const SurfacePoint& surface = projected[at++];
    if (place && roles_[v] == Role::kFree) positions_[v] = surface.on;
    normals_[v] = surface.normal;
    sizes_[v] = size_at(effective_curvature(surface.curvature_first, surface.curvature_second));
  }
  return PHX_MC_OK;
}

// Split half-edge h (a -> b, in t0 = (a, b, c)) at its projected midpoint;
// its twin g (b -> a) lies in t1 = (b, a, d). Result: t0 = (a, m, c),
// t2 = (m, b, c), t1 = (b, m, d), t3 = (m, a, d).
bool Adapter::split(int32_t h) {
  if (kept_edge(h)) return false;
  const int32_t g = twin(h);
  const int32_t a = tail(h), b = head(h);
  const int32_t c = corners_[static_cast<size_t>(prev(h))];
  const int32_t d = corners_[static_cast<size_t>(prev(g))];
  Point on{}, normal{};
  double kappa = 0;
  if (!project(scale(add(position(a), position(b)), 0.5), on, normal, kappa)) return false;
  const Point &pa = position(a), &pb = position(b), &pc = position(c), &pd = position(d);
  if (!oriented({pa, on, pc}, {judged_normal(a), &normal, judged_normal(c)}) ||
      !oriented({on, pb, pc}, {&normal, judged_normal(b), judged_normal(c)}) ||
      !oriented({pb, on, pd}, {judged_normal(b), &normal, judged_normal(d)}) ||
      !oriented({on, pa, pd}, {&normal, judged_normal(a), judged_normal(d)}))
    return false;
  const int32_t m = new_vertex(on, normal, size_at(kappa));
  const int32_t t2 = new_triangle(), t3 = new_triangle();
  const int32_t hn = next(h), gn = next(g);
  const int32_t o_hn = twin(hn), o_gn = twin(gn);
  const uint8_t f_hn = kept_edges_[static_cast<size_t>(hn)];
  const uint8_t f_gn = kept_edges_[static_cast<size_t>(gn)];
  // t0 = (a, m, c): corners h (a), hn (m), hp (c).
  corners_[static_cast<size_t>(hn)] = m;
  // t1 = (b, m, d): corners g (b), gn (m), gp (d).
  corners_[static_cast<size_t>(gn)] = m;
  // t2 = (m, b, c), t3 = (m, a, d).
  const int32_t e = 3 * t2, f = 3 * t3;
  corners_[static_cast<size_t>(e)] = m;
  corners_[static_cast<size_t>(e + 1)] = b;
  corners_[static_cast<size_t>(e + 2)] = c;
  corners_[static_cast<size_t>(f)] = m;
  corners_[static_cast<size_t>(f + 1)] = a;
  corners_[static_cast<size_t>(f + 2)] = d;
  link(e + 1, o_hn);  // b -> c
  link(e + 2, hn);    // c -> m with m -> c
  link(e, g);         // m -> b with b -> m
  link(f + 1, o_gn);  // a -> d
  link(f + 2, gn);    // d -> m with m -> d
  link(f, h);         // m -> a with a -> m
  // The outer edges b -> c and a -> d moved to t2 and t3 with their flags.
  kept_edges_[static_cast<size_t>(e + 1)] = f_hn;
  kept_edges_[static_cast<size_t>(f + 1)] = f_gn;
  for (const int32_t inner : {hn, gn, e, e + 2, f, f + 2}) kept_edges_[static_cast<size_t>(inner)] = 0;
  vertex_corner_[static_cast<size_t>(m)] = e;
  vertex_corner_[static_cast<size_t>(b)] = e + 1;
  vertex_corner_[static_cast<size_t>(a)] = h;
  vertex_corner_[static_cast<size_t>(c)] = e + 2;
  vertex_corner_[static_cast<size_t>(d)] = f + 2;
  valences_[static_cast<size_t>(m)] = 4;
  ++valences_[static_cast<size_t>(c)];
  ++valences_[static_cast<size_t>(d)];
  ++counters_.splits;
  return true;
}

double Adapter::star_minimum_angle(int32_t v) const {
  double worst = kInfinity;
  for_ring(v, [&](int32_t c) {
    const size_t base = 3 * static_cast<size_t>(c / 3);
    worst = std::min(worst, minimum_sine(position(corners_[base]), position(corners_[base + 1]),
                                         position(corners_[base + 2])));
  });
  return worst < kInfinity ? sine_degrees(worst) : 180;
}

// Collapse half-edge h (a -> b), keeping vertex a: at the projected midpoint
// when a is free and `at_endpoint` is not asked, else at a's position (no
// projection, and only b's triangles change). A kept vertex b never goes, so
// no kept triangle changes (b is in none, and neither is the edge a -> b).
bool Adapter::collapse(int32_t h, bool require_quality_gain, bool at_endpoint) {
  const int32_t g = twin(h);
  const int32_t a = tail(h), b = head(h);
  if (role(b) != Role::kFree) return false;
  const bool free = role(a) == Role::kFree && !at_endpoint;
  const int32_t c = corners_[static_cast<size_t>(prev(h))];
  const int32_t d = corners_[static_cast<size_t>(prev(g))];
  if (c == d) return false;
  // Link condition: the rings of a and b share exactly c and d (b's ring
  // marked once, a's ring tested against the marks).
  if (++stamp_ == 0) {
    std::fill(marks_.begin(), marks_.end(), 0);
    stamp_ = 1;
  }
  if (marks_.size() < positions_.size()) marks_.resize(positions_.size(), 0);
  int valence_b = 0;
  for_ring(b, [&](int32_t cb) {
    marks_[static_cast<size_t>(head(cb))] = stamp_;
    ++valence_b;
  });
  int shared = 0, valence_a = 0;
  bool bad = false;
  for_ring(a, [&](int32_t ca) {
    ++valence_a;
    const int32_t x = head(ca);
    if (x == b || marks_[static_cast<size_t>(x)] != stamp_) return;
    if (x == c || x == d)
      ++shared;
    else
      bad = true;
  });
  if (bad || shared != 2) return false;
  if (valence_a + valence_b - 4 < 3 || valence(c) <= 3 || valence(d) <= 3) return false;
  Point on{}, normal{};
  double kappa = 0;
  if (free) {
    if (!project(scale(add(position(a), position(b)), 0.5), on, normal, kappa)) return false;
  } else {
    on = position(a);
    normal = normals_[static_cast<size_t>(a)];
  }
  const double size = free ? size_at(kappa) : sizes_[static_cast<size_t>(a)];
  const double longest = kLong * size;
  // Only b's triangles change when a stays put (t0 and t1 are in both stars).
  const double before = !require_quality_gain ? 0
                        : free ? std::min(star_minimum_angle(a), star_minimum_angle(b))
                               : star_minimum_angle(b);
  double after_sine = kInfinity;
  // Every triangle around a or b except t0, t1 gets the merged vertex at `on`.
  const int32_t t0 = h / 3, t1 = g / 3;
  bool acceptable = true;
  const Point* merged_normal = judged_normal(a);
  if (free) merged_normal = &normal;
  const auto check = [&](int32_t v) {
    for_ring(v, [&](int32_t corner) {
      if (!acceptable) return;
      const int32_t t = corner / 3;
      if (t == t0 || t == t1) return;
      const size_t base = 3 * static_cast<size_t>(t);
      std::array<Point, 3> p{};
      std::array<const Point*, 3> n{};
      for (size_t k = 0; k < 3; ++k) {
        const int32_t w = corners_[base + k];
        const bool merged = w == a || w == b;
        p[k] = merged ? on : position(w);
        n[k] = merged ? merged_normal : judged_normal(w);
      }
      if (!oriented(p, n) || norm2(sub(position(head(corner)), on)) > longest * longest) {
        acceptable = false;
        return;
      }
      after_sine = std::min(after_sine, minimum_sine(p[0], p[1], p[2]));
    });
  };
  if (free) check(a);
  if (acceptable) check(b);
  if (!acceptable) return false;
  const double after = after_sine < kInfinity ? sine_degrees(after_sine) : 180;
  if (require_quality_gain ? !(after > before) : after < 0.5 * targets_.minimum_angle_degrees)
    return false;
  // Commit: b's corners become a; glue the outer edges of t0 and t1 (a glued
  // edge is kept if either of its halves was).
  for_ring(b, [&](int32_t corner) { corners_[static_cast<size_t>(corner)] = a; });
  const int32_t hn = next(h), hp = prev(h), gn = next(g), gp = prev(g);
  const int32_t o_hn = twin(hn), o_hp = twin(hp), o_gn = twin(gn), o_gp = twin(gp);
  const auto glue = [&](int32_t x, int32_t y) {
    link(x, y);
    const uint8_t flag = static_cast<uint8_t>(kept_edges_[static_cast<size_t>(x)] |
                                              kept_edges_[static_cast<size_t>(y)]);
    kept_edges_[static_cast<size_t>(x)] = kept_edges_[static_cast<size_t>(y)] = flag;
  };
  glue(o_hn, o_hp);
  glue(o_gn, o_gp);
  vertex_corner_[static_cast<size_t>(a)] = o_hp;
  vertex_corner_[static_cast<size_t>(c)] = o_hn;
  vertex_corner_[static_cast<size_t>(d)] = o_gn;
  vertex_corner_[static_cast<size_t>(b)] = -1;
  valences_[static_cast<size_t>(a)] = valence_a + valence_b - 4;
  --valences_[static_cast<size_t>(c)];
  --valences_[static_cast<size_t>(d)];
  valences_[static_cast<size_t>(b)] = 0;
  free_vertices_.push_back(b);
  for (const int32_t t : {t0, t1}) {
    for (size_t k = 0; k < 3; ++k) {
      corners_[3 * static_cast<size_t>(t) + k] = -1;
      kept_edges_[3 * static_cast<size_t>(t) + k] = 0;
    }
    free_triangles_.push_back(t);
  }
  if (free) {
    positions_[static_cast<size_t>(a)] = on;
    normals_[static_cast<size_t>(a)] = normal;
    sizes_[static_cast<size_t>(a)] = size;
  }
  ++counters_.collapses;
  return true;
}

// Flip half-edge h (a -> b in t0 = (a, b, c); twin in t1 = (b, a, d)) to the
// diagonal c - d: t0 = (c, a, d), t1 = (d, b, c).
bool Adapter::flip(int32_t h) {
  if (kept_edge(h)) return false;
  const int32_t g = twin(h);
  const int32_t a = tail(h), b = head(h);
  const int32_t c = corners_[static_cast<size_t>(prev(h))];
  const int32_t d = corners_[static_cast<size_t>(prev(g))];
  if (c == d) return false;
  bool exists = false;
  for_ring(c, [&](int32_t corner) { exists = exists || head(corner) == d; });
  if (exists) return false;
  if (!oriented({position(c), position(a), position(d)},
                {judged_normal(c), judged_normal(a), judged_normal(d)}) ||
      !oriented({position(d), position(b), position(c)},
                {judged_normal(d), judged_normal(b), judged_normal(c)}))
    return false;
  const int32_t hn = next(h), hp = prev(h), gn = next(g), gp = prev(g);
  const int32_t o_bc = twin(hn), o_ca = twin(hp), o_ad = twin(gn), o_db = twin(gp);
  const uint8_t f_bc = kept_edges_[static_cast<size_t>(hn)];
  const uint8_t f_ca = kept_edges_[static_cast<size_t>(hp)];
  const uint8_t f_ad = kept_edges_[static_cast<size_t>(gn)];
  const uint8_t f_db = kept_edges_[static_cast<size_t>(gp)];
  const int32_t s = 3 * (h / 3), r = 3 * (g / 3);
  // t0 = (c, a, d): corners s (c -> a), s+1 (a -> d), s+2 (d -> c).
  corners_[static_cast<size_t>(s)] = c;
  corners_[static_cast<size_t>(s + 1)] = a;
  corners_[static_cast<size_t>(s + 2)] = d;
  // t1 = (d, b, c): corners r (d -> b), r+1 (b -> c), r+2 (c -> d).
  corners_[static_cast<size_t>(r)] = d;
  corners_[static_cast<size_t>(r + 1)] = b;
  corners_[static_cast<size_t>(r + 2)] = c;
  link(s, o_ca);
  link(s + 1, o_ad);
  link(r, o_db);
  link(r + 1, o_bc);
  link(s + 2, r + 2);
  kept_edges_[static_cast<size_t>(s)] = f_ca;
  kept_edges_[static_cast<size_t>(s + 1)] = f_ad;
  kept_edges_[static_cast<size_t>(r)] = f_db;
  kept_edges_[static_cast<size_t>(r + 1)] = f_bc;
  kept_edges_[static_cast<size_t>(s + 2)] = kept_edges_[static_cast<size_t>(r + 2)] = 0;
  vertex_corner_[static_cast<size_t>(c)] = s;
  vertex_corner_[static_cast<size_t>(a)] = s + 1;
  vertex_corner_[static_cast<size_t>(d)] = r;
  vertex_corner_[static_cast<size_t>(b)] = r + 1;
  --valences_[static_cast<size_t>(a)];
  --valences_[static_cast<size_t>(b)];
  ++valences_[static_cast<size_t>(c)];
  ++valences_[static_cast<size_t>(d)];
  ++counters_.flips;
  return true;
}

bool Adapter::flip_improves_valence(int32_t h) const {
  const int32_t g = twin(h);
  const int32_t a = tail(h), b = head(h);
  const int32_t c = corners_[static_cast<size_t>(prev(h))];
  const int32_t d = corners_[static_cast<size_t>(prev(g))];
  const int va = valence(a), vb = valence(b), vc = valence(c), vd = valence(d);
  if (va <= 3 || vb <= 3) return false;
  const auto deviation = [](int x) { return (x - 6) * (x - 6); };
  const int before = deviation(va) + deviation(vb) + deviation(vc) + deviation(vd);
  const int after = deviation(va - 1) + deviation(vb - 1) + deviation(vc + 1) + deviation(vd + 1);
  return after < before;
}

// The flip raises the smaller minimum angle of the two triangles.
bool Adapter::flip_is_delaunay_better(int32_t h) const {
  const int32_t g = twin(h);
  const int32_t a = tail(h), b = head(h);
  const int32_t c = corners_[static_cast<size_t>(prev(h))];
  const int32_t d = corners_[static_cast<size_t>(prev(g))];
  if (valence(a) <= 3 || valence(b) <= 3) return false;
  const Point &pa = position(a), &pb = position(b), &pc = position(c), &pd = position(d);
  const double before = std::min(minimum_angle(pa, pb, pc), minimum_angle(pb, pa, pd));
  const double after = std::min(minimum_angle(pc, pa, pd), minimum_angle(pd, pb, pc));
  return after > before + 1e-9;
}

Point Adapter::ring_centroid(int32_t v) const {
  // Area-weighted centroid of the star's triangle centroids.
  Point sum{0, 0, 0};
  double total = 0;
  for_ring(v, [&](int32_t corner) {
    const size_t base = 3 * static_cast<size_t>(corner / 3);
    const Point& p0 = position(corners_[base]);
    const Point& p1 = position(corners_[base + 1]);
    const Point& p2 = position(corners_[base + 2]);
    const double area = std::sqrt(norm2(cross(sub(p1, p0), sub(p2, p0))));
    sum = add(sum, scale(add(add(p0, p1), p2), area / 3.0));
    total += area;
  });
  if (!(total > 0)) return position(v);
  return scale(sum, 1.0 / total);
}

Point Adapter::equilateral_goal(int32_t v) const {
  // Mean of the apexes, on v's side, of the equilateral triangles raised on
  // the star's opposite edges: each apex makes its triangle equilateral, so
  // their mean lifts the star's worst angle where the centroid does not.
  const Point& n = normals_[static_cast<size_t>(v)];
  Point sum{0, 0, 0};
  int count = 0;
  for_ring(v, [&](int32_t corner) {
    const Point& p = position(head(corner));
    const Point& q = position(corners_[static_cast<size_t>(prev(corner))]);
    const Point edge = sub(q, p);
    const Point inward = cross(n, edge);
    const double length = std::sqrt(norm2(inward));
    if (!(length > 0)) return;
    const double height = 0.8660254037844386 * std::sqrt(norm2(edge));
    sum = add(sum, add(scale(add(p, q), 0.5), scale(inward, height / length)));
    ++count;
  });
  return count > 0 ? scale(sum, 1.0 / count) : position(v);
}

// Tangential move of an ordinary vertex toward `goal`, then projection.
bool Adapter::relocate(int32_t v, const Point& goal, bool require_quality_gain) {
  if (role(v) != Role::kFree) return false;
  const Point& p = position(v);
  const Point& n = normals_[static_cast<size_t>(v)];
  const Point delta = sub(goal, p);
  const Point tangential = sub(delta, scale(n, dot(delta, n)));
  Point on{}, normal{};
  double kappa = 0;
  if (!project(add(p, tangential), on, normal, kappa)) return false;
  const double before = require_quality_gain ? star_minimum_angle(v) : 0;
  double after = 180;
  bool acceptable = true;
  for_ring(v, [&](int32_t corner) {
    if (!acceptable) return;
    const int32_t q = head(corner);
    const int32_t r = corners_[static_cast<size_t>(prev(corner))];
    if (!oriented({on, position(q), position(r)}, {&normal, judged_normal(q), judged_normal(r)})) {
      acceptable = false;
      return;
    }
    after = std::min(after, minimum_angle(on, position(q), position(r)));
  });
  if (!acceptable || (require_quality_gain && !(after > before))) return false;
  positions_[static_cast<size_t>(v)] = on;
  normals_[static_cast<size_t>(v)] = normal;
  sizes_[static_cast<size_t>(v)] = size_at(kappa);
  ++counters_.relocations;
  return true;
}

void Adapter::split_long() {
  const size_t count = corners_.size();
  for (size_t c = 0; c < count; ++c) {
    const auto h = static_cast<int32_t>(c);
    if (!alive_triangle(h / 3) || twin(h) < h) continue;
    const int32_t a = tail(h), b = head(h);
    const double limit = kLong * edge_size(a, b);
    if (norm2(sub(position(a), position(b))) > limit * limit) split(h);
  }
}

void Adapter::collapse_short() {
  const size_t count = corners_.size();
  for (size_t c = 0; c < count; ++c) {
    const auto h = static_cast<int32_t>(c);
    if (!alive_triangle(h / 3)) continue;
    const int32_t a = tail(h), b = head(h);
    const double limit = kShort * edge_size(a, b);
    const double length2 = norm2(sub(position(a), position(b)));
    if (!(length2 < limit * limit)) continue;
    // Edges far below the size (coarsening a fine input) merge into an
    // endpoint, without a projection; nearer the size, at the projected
    // midpoint, which spaces the survivors evenly.
    collapse(h, false, length2 < 0.25 * limit * limit);
  }
}

void Adapter::flip_valence() {
  const size_t count = corners_.size();
  for (size_t c = 0; c < count; ++c) {
    const auto h = static_cast<int32_t>(c);
    if (!alive_triangle(h / 3) || twin(h) < h) continue;
    if (flip_improves_valence(h)) flip(h);
  }
}

void Adapter::flip_delaunay() {
  for (size_t c = 0; c < corners_.size(); ++c) {
    const auto h = static_cast<int32_t>(c);
    if (!alive_triangle(h / 3) || twin(h) < h) continue;
    if (flip_is_delaunay_better(h)) flip(h);
  }
}

// Tangential relaxation toward the area-weighted ring centroid; a move below
// 1/20 of the vertex's size is not worth a projection.
void Adapter::smooth() {
  for (size_t v = 0; v < positions_.size(); ++v) {
    if (vertex_corner_[v] < 0 || roles_[v] != Role::kFree) continue;
    const auto vertex = static_cast<int32_t>(v);
    const Point goal = ring_centroid(vertex);
    const Point delta = sub(goal, positions_[v]);
    const Point tangential = sub(delta, scale(normals_[v], dot(delta, normals_[v])));
    const double small = 0.05 * sizes_[v];
    if (norm2(tangential) < small * small) continue;
    relocate(vertex, goal, false);
  }
}

// Sweeps over triangles below the minimum angle; returns the number still
// below it.
int64_t Adapter::improve_quality() {
  const double target = targets_.minimum_angle_degrees;
  int64_t below = 0;
  for (int sweep = 0; sweep < kQualitySweeps; ++sweep) {
    // Delaunay flips first: they raise the smaller angle of every flipped pair.
    flip_delaunay();
    below = 0;
    int64_t improved = 0;
    for (size_t t = 0; t < corners_.size() / 3; ++t) {
      if (!alive_triangle(static_cast<int32_t>(t))) continue;
      const size_t base = 3 * t;
      const auto angle = [&]() {
        return minimum_angle(position(corners_[base]), position(corners_[base + 1]),
                             position(corners_[base + 2]));
      };
      if (angle() >= target) continue;
      bool done = false;
      // Relocate each corner toward its ring centroid, else toward its
      // equilateral goal, while that lifts its star and the triangle stays
      // below the target.
      for (size_t k = 0; k < 3 && angle() < target; ++k) {
        const int32_t v = corners_[base + k];
        if (relocate(v, ring_centroid(v), true) || relocate(v, equilateral_goal(v), true))
          done = true;
      }
      if (angle() < target) {
        // Collapse the shortest edge (either way) if that lifts the stars,
        // else split the longest.
        int32_t shortest = static_cast<int32_t>(base), longest = shortest;
        double lo = INFINITY, hi = -1;
        for (size_t k = 0; k < 3; ++k) {
          const auto h = static_cast<int32_t>(base + k);
          const double length = norm2(sub(position(tail(h)), position(head(h))));
          if (length < lo) {
            lo = length;
            shortest = h;
          }
          if (length > hi) {
            hi = length;
            longest = h;
          }
        }
        bool changed = collapse(shortest, true) || collapse(twin(shortest), true);
        if (!changed && std::sqrt(hi) > edge_size(tail(longest), head(longest)))
          changed = split(longest);
        done = done || changed;
      }
      if (done) {
        ++improved;
        if (alive_triangle(static_cast<int32_t>(t)) && angle() < target) ++below;
      } else {
        ++below;
      }
    }
    if (improved == 0) break;
  }
  return below;
}

// The largest sampled distance (edge midpoints and centroid) of each listed
// triangle, from one block of oracle evaluations.
void Adapter::deviations(const std::vector<int32_t>& triangles, std::vector<double>& out) const {
  constexpr double third = 1.0 / 3.0;
  constexpr std::array<Point, 4> weights{
      {{0.5, 0.5, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {third, third, third}}};
  std::vector<Point> samples;
  samples.reserve(weights.size() * triangles.size());
  for (const int32_t t : triangles) {
    const size_t base = 3 * static_cast<size_t>(t);
    const Point& a = position(corners_[base]);
    const Point& b = position(corners_[base + 1]);
    const Point& c = position(corners_[base + 2]);
    for (const Point& w : weights)
      samples.push_back(add(add(scale(a, w[0]), scale(b, w[1])), scale(c, w[2])));
  }
  std::vector<double> distance;
  oracle_.distances(samples, distance);
  out.assign(triangles.size(), 0);
  for (size_t i = 0; i < triangles.size(); ++i)
    for (size_t k = 0; k < weights.size(); ++k)
      out[i] = std::max(out[i], distance[weights.size() * i + k]);
}

std::vector<int32_t> Adapter::alive_triangles() const {
  std::vector<int32_t> alive;
  for (size_t t = 0; t < corners_.size() / 3; ++t)
    if (alive_triangle(static_cast<int32_t>(t))) alive.push_back(static_cast<int32_t>(t));
  return alive;
}

// Splits the longest edge of every triangle sampled above the repair margin
// until none is, or no split helps. No vertex moves or goes here (splits add
// vertices, flips reconnect), so a triangle slot holding the same corners as
// when it was sampled keeps its deviation; only changed slots are sampled.
int64_t Adapter::repair_fidelity() {
  const double limit = kRepairMargin * targets_.tolerance;
  int64_t above = 0;
  std::vector<std::array<int32_t, 3>> sampled_corners;
  std::vector<double> sampled_deviation, deviation;
  std::vector<int32_t> stale;
  for (int sweep = 0; sweep < kFidelitySweeps; ++sweep) {
    const std::vector<int32_t> alive = alive_triangles();
    sampled_corners.resize(corners_.size() / 3, {-1, -1, -1});
    sampled_deviation.resize(corners_.size() / 3, 0);
    stale.clear();
    for (const int32_t t : alive) {
      const size_t base = 3 * static_cast<size_t>(t);
      const std::array<int32_t, 3> now{corners_[base], corners_[base + 1], corners_[base + 2]};
      if (sampled_corners[static_cast<size_t>(t)] == now) continue;
      sampled_corners[static_cast<size_t>(t)] = now;
      stale.push_back(t);
    }
    deviations(stale, deviation);
    for (size_t i = 0; i < stale.size(); ++i)
      sampled_deviation[static_cast<size_t>(stale[i])] = deviation[i];
    // Offenders with their corners as sampled: a split earlier in this sweep
    // may remove or reshape one, whose deviation is then unknown until the
    // next sweep.
    std::vector<std::pair<int32_t, std::array<int32_t, 3>>> offenders;
    for (const int32_t t : alive) {
      if (!(sampled_deviation[static_cast<size_t>(t)] > limit)) continue;
      offenders.push_back({t, sampled_corners[static_cast<size_t>(t)]});
    }
    above = static_cast<int64_t>(offenders.size());
    if (above == 0 || split_listed(offenders) == 0) break;
    // Splits leave long thin pairs; Delaunay flips restore the angles.
    flip_delaunay();
  }
  return above;
}

bool Adapter::split_longest(int32_t t) {
  const size_t base = 3 * static_cast<size_t>(t);
  int32_t longest = -1;
  double hi = -1;
  for (size_t k = 0; k < 3; ++k) {
    const auto h = static_cast<int32_t>(base + k);
    if (kept_edge(h)) continue;
    const double length = norm2(sub(position(tail(h)), position(head(h))));
    if (length > hi) {
      hi = length;
      longest = h;
    }
  }
  return longest >= 0 && split(longest);
}

int64_t Adapter::split_listed(
    const std::vector<std::pair<int32_t, std::array<int32_t, 3>>>& listed) {
  int64_t splits = 0;
  for (const auto& [t, sampled] : listed) {
    const size_t base = 3 * static_cast<size_t>(t);
    if (!alive_triangle(t) || corners_[base] != sampled[0] || corners_[base + 1] != sampled[1] ||
        corners_[base + 2] != sampled[2])
      continue;
    if (split_longest(t)) ++splits;
  }
  return splits;
}

int64_t Adapter::repair(const std::vector<uint8_t>& refused) {
  std::vector<std::pair<int32_t, std::array<int32_t, 3>>> listed;
  for (size_t t = 0; t < refused.size(); ++t) {
    if (!refused[t]) continue;
    const size_t base = 3 * t;
    listed.push_back(
        {static_cast<int32_t>(t), {corners_[base], corners_[base + 1], corners_[base + 2]}});
  }
  const int64_t splits = split_listed(listed);
  if (splits > 0) flip_delaunay();
  return splits;
}

int64_t Adapter::relax() {
  const double target = targets_.minimum_angle_degrees;
  const int64_t before = counters_.relocations;
  for (int sweep = 0; sweep < kQualitySweeps; ++sweep) {
    flip_delaunay();
    const int64_t moved = counters_.relocations;
    for (size_t t = 0; t < corners_.size() / 3; ++t) {
      if (!alive_triangle(static_cast<int32_t>(t))) continue;
      const size_t base = 3 * t;
      for (size_t k = 0; k < 3; ++k) {
        if (minimum_angle(position(corners_[base]), position(corners_[base + 1]),
                          position(corners_[base + 2])) >= target)
          break;
        const int32_t v = corners_[base + k];
        if (!relocate(v, ring_centroid(v), true)) relocate(v, equilateral_goal(v), true);
      }
    }
    if (counters_.relocations == moved) break;
  }
  return counters_.relocations - before;
}

void Adapter::run() {
  for (int sweep = 0; sweep < kSizeSweeps; ++sweep) {
    split_long();
    collapse_short();
    flip_valence();
    smooth();
  }
  improve_quality();
  repair_fidelity();
  counters_.below_angle = improve_quality();
  const std::vector<int32_t> alive = alive_triangles();
  std::vector<double> deviation;
  deviations(alive, deviation);
  counters_.above_tolerance = std::count_if(
      deviation.begin(), deviation.end(), [&](double d) { return d > targets_.tolerance; });
}

void Adapter::store(std::vector<Point>& points, std::vector<Triangle>& triangles) const {
  std::vector<int64_t> renumber(positions_.size(), -1);
  points.clear();
  triangles.clear();
  for (size_t t = 0; t < corners_.size() / 3; ++t) {
    if (!alive_triangle(static_cast<int32_t>(t))) continue;
    Triangle out{};
    for (size_t k = 0; k < 3; ++k) {
      const auto v = static_cast<size_t>(corners_[3 * t + k]);
      if (renumber[v] < 0) {
        renumber[v] = static_cast<int64_t>(points.size());
        points.push_back(positions_[v]);
      }
      out[k] = renumber[v];
    }
    triangles.push_back(out);
  }
}

}  // namespace

int32_t adapt_surface(std::vector<Point>& points, std::vector<Triangle>& triangles,
                      const SurfaceOracle& oracle, const AdaptTargets& targets,
                      const std::vector<int64_t>* twins, AdaptCounters& counters) {
  if (!(targets.edge_length > 0) || !(targets.tolerance > 0) ||
      !(targets.minimum_angle_degrees > 0) || !(targets.minimum_angle_degrees < 60))
    return PHX_MC_INVALID_ARGUMENT;
  Adapter adapter(oracle, targets, counters);
  const int32_t status = adapter.load(points, triangles, true, twins);
  if (status != PHX_MC_OK) {
    points.clear();
    triangles.clear();
    return status;
  }
  adapter.run();
  adapter.store(points, triangles);
  return PHX_MC_OK;
}

int32_t repair_surface(std::vector<Point>& points, std::vector<Triangle>& triangles,
                       const SurfaceOracle& oracle, const AdaptTargets& targets,
                       const std::vector<uint8_t>& refused, int64_t& splits) {
  splits = 0;
  AdaptCounters counters;
  Adapter adapter(oracle, targets, counters);
  const int32_t status = adapter.load(points, triangles, false);
  if (status != PHX_MC_OK) return status;
  splits = adapter.repair(refused);
  if (splits > 0) adapter.store(points, triangles);
  return PHX_MC_OK;
}

int32_t relax_surface(std::vector<Point>& points, std::vector<Triangle>& triangles,
                      const SurfaceOracle& oracle, const AdaptTargets& targets,
                      int64_t& relocations) {
  relocations = 0;
  AdaptCounters counters;
  Adapter adapter(oracle, targets, counters);
  const int32_t status = adapter.load(points, triangles, false);
  if (status != PHX_MC_OK) return status;
  relocations = adapter.relax();
  // Flips alone are kept only with relocations, so an unchanged surface
  // (relocations == 0) needs no new decision.
  if (relocations > 0) adapter.store(points, triangles);
  return PHX_MC_OK;
}

int32_t certify_surface(std::vector<Point>& points, std::vector<Triangle>& triangles,
                        const SurfaceOracle& oracle, const AdaptTargets& targets,
                        const TriangleCertifier& certifier, const CarrierEvidence& carrier,
                        const TriangleCertifier* repair_certifier, bool margin_from_start,
                        CertifyCounters& counters) {
  if (!(targets.edge_length > 0) || !(targets.tolerance > 0) ||
      !(targets.minimum_angle_degrees > 0) || !(targets.minimum_angle_degrees < 60))
    return PHX_MC_INVALID_ARGUMENT;
  counters = CertifyCounters{};
  if (!closed_manifold(triangles, points.size())) return PHX_MC_INVALID_INPUT;
  // What earlier rounds decided, by triangle key: a repair changes only the
  // stars it splits and the flips around them. A triangle refused before
  // (for either reason) has its contacts classified again.
  struct Decided {
    TriangleKey key;
    double bound;
    bool refused;
  };
  const auto by_key = [](const Decided& entry, const TriangleKey& key) { return entry.key < key; };
  std::vector<Decided> decided, next;
  std::vector<uint8_t> refused, above, fresh, changed, illegal;
  struct Standing {
    std::vector<Point> points;
    std::vector<Triangle> triangles;
    std::vector<uint8_t> carried;
    double distance;
    int64_t carrier;
  };
  std::optional<Standing> last_standing;
  for (int round = 0;; ++round) {
    const size_t m = triangles.size();
    refused.assign(m, 0);
    above.assign(m, 0);
    fresh.assign(m, 0);
    changed.assign(m, 0);
    counters.carried.assign(m, 0);
    next.clear();
    next.reserve(m);
    double maximum = 0;
    int64_t carried = 0;
    for (size_t t = 0; t < m; ++t) {
      const Point& a = points[static_cast<size_t>(triangles[t][0])];
      const Point& b = points[static_cast<size_t>(triangles[t][1])];
      const Point& c = points[static_cast<size_t>(triangles[t][2])];
      const TriangleKey key = triangle_key(a, b, c);
      const auto found = std::lower_bound(decided.begin(), decided.end(), key, by_key);
      const bool known = found != decided.end() && found->key == key;
      changed[t] = known && !found->refused ? 0 : 1;
      // The proof first (decided once per triangle); a refused triangle of
      // the carrier's own is accepted at the bound its producer proved.
      const double proved = known ? found->bound : certifier.bound(a, b, c);
      if (!known && !(proved < kInfinity)) ++counters.refused_fidelity;
      double bound = proved;
      if (!(proved < kInfinity)) {
        const auto entry = std::lower_bound(
            carrier.entries.begin(), carrier.entries.end(), key,
            [](const std::pair<TriangleKey, double>& x, const TriangleKey& k) { return x.first < k; });
        if (entry != carrier.entries.end() && entry->first == key) {
          bound = entry->second;
          counters.carried[t] = 1;
          ++carried;
        }
      }
      if (!(bound < kInfinity))
        refused[t] = 1;
      else
        maximum = std::max(maximum, bound);
      fresh[t] = static_cast<uint8_t>(!known);
      next.push_back({key, proved, false});
    }
    counters.refused_embedding += refuse_illegal_contacts(
        points, triangles, round == 0 ? nullptr : &changed, illegal);
    int64_t count = 0, margin = 0;
    for (size_t t = 0; t < m; ++t) {
      refused[t] = static_cast<uint8_t>(refused[t] | illegal[t]);
      next[t].refused = refused[t] != 0;
      count += refused[t];
    }
    // Once the surface needs a repair (or from the start when asked), every
    // triangle decided this round (all of them in the first) must also stand
    // in the tighter band: one proved only by `certifier` is repaired too,
    // not refused.
    if (repair_certifier != nullptr && (margin_from_start || round > 0 || count > 0))
      for (size_t t = 0; t < m; ++t) {
        if (!fresh[t] || refused[t] || counters.carried[t]) continue;
        const Point& a = points[static_cast<size_t>(triangles[t][0])];
        const Point& b = points[static_cast<size_t>(triangles[t][1])];
        const Point& c = points[static_cast<size_t>(triangles[t][2])];
        if (repair_certifier->bound(a, b, c) < kInfinity) continue;
        above[t] = 1;
        ++margin;
      }
    counters.rounds = round + 1;
    counters.carrier = carried;
    counters.distance = maximum;
    // A margin repair may break what already stood (a new contact): the
    // last standing surface is kept to return instead.
    if (count == 0 && margin > 0)
      last_standing = Standing{points, triangles, counters.carried, maximum, carried};
    const auto finish = [&]() {
      if (count == 0 || !last_standing) return count == 0 ? PHX_MC_OK : PHX_MC_UNRESOLVED;
      points = std::move(last_standing->points);
      triangles = std::move(last_standing->triangles);
      counters.carried = std::move(last_standing->carried);
      counters.distance = last_standing->distance;
      counters.carrier = last_standing->carrier;
      return PHX_MC_OK;
    };
    if (count + margin == 0 || round + 1 == kRepairRounds) return finish();
    std::sort(next.begin(), next.end(),
              [](const Decided& x, const Decided& y) { return x.key < y.key; });
    decided.swap(next);
    for (size_t t = 0; t < m; ++t) above[t] = static_cast<uint8_t>(above[t] | refused[t]);
    int64_t splits = 0;
    const int32_t status = repair_surface(points, triangles, oracle, targets, above, splits);
    if (status != PHX_MC_OK) return status;
    counters.repair_splits += splits;
    if (splits == 0) return finish();
  }
}

}  // namespace phx::mc::remesh

extern "C" {

PHX_MC_API int32_t phx_mc_surface_result_sizes(const void* result, int64_t* sizes) {
  if (result == nullptr || sizes == nullptr) return PHX_MC_INVALID_ARGUMENT;
  const auto& data = *static_cast<const phx::mc::remesh::SurfaceResult*>(result);
  sizes[0] = static_cast<int64_t>(data.points.size());
  sizes[1] = static_cast<int64_t>(data.triangles.size());
  return PHX_MC_OK;
}

PHX_MC_API int32_t phx_mc_surface_result_arrays(void* result, int64_t vertex_capacity,
                                                int64_t triangle_capacity, double* vertices,
                                                int64_t* triangles) {
  if (result == nullptr || vertices == nullptr || triangles == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  const auto& data = *static_cast<const phx::mc::remesh::SurfaceResult*>(result);
  if (vertex_capacity < static_cast<int64_t>(data.points.size()) ||
      triangle_capacity < static_cast<int64_t>(data.triangles.size()))
    return PHX_MC_CAPACITY_EXCEEDED;
  for (size_t v = 0; v < data.points.size(); ++v)
    for (size_t d = 0; d < 3; ++d) vertices[3 * v + d] = data.points[v][d];
  for (size_t t = 0; t < data.triangles.size(); ++t)
    for (size_t k = 0; k < 3; ++k) triangles[3 * t + k] = data.triangles[t][k];
  return PHX_MC_OK;
}

PHX_MC_API int32_t phx_mc_surface_result_carried(const void* result, int64_t triangle_capacity,
                                                 uint8_t* carried) {
  if (result == nullptr || carried == nullptr) return PHX_MC_INVALID_ARGUMENT;
  const auto& data = *static_cast<const phx::mc::remesh::SurfaceResult*>(result);
  if (triangle_capacity < static_cast<int64_t>(data.triangles.size()))
    return PHX_MC_CAPACITY_EXCEEDED;
  for (size_t t = 0; t < data.triangles.size(); ++t)
    carried[t] = data.carried.empty() ? 0 : data.carried[t];
  return PHX_MC_OK;
}

PHX_MC_API void phx_mc_surface_result_free(void* result) {
  delete static_cast<phx::mc::remesh::SurfaceResult*>(result);
}

}  // extern "C"
