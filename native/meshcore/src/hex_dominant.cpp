//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Hex-dominant meshing of a closed oriented triangle surface (hex_dominant.hpp).
//
// Construction (floating point, exact predicates for inside/outside):
//  1. Surface: closed consistently oriented manifold check, unit normals,
//     feature edges (normal angle above the feature angle or a facet change),
//     smooth patches between them, separating feature curves and corners
//     (vertices with a feature degree other than 0 and 2, or three patches),
//     transient bin indices of triangles and feature edges.
//  2. Inside/outside of lattice vertices: parity of +x ray crossings on their
//     lattice line, decided exactly (orient2d on the yz projection with the
//     symbolic perturbation (y + e, z + e^2), orient3d against the plane with
//     the vertex perturbed by +e x).
//  3. Octree from the core size h to the boundary size b = h / 2^K on one
//     lattice of spacing b: a level-l cell within c_band cells (L-infinity)
//     of a finest cell touching a triangle box is split; the split sets are
//     closed so neighbors (26) of leaves differ by at most one level.
//  4. Core cells: non-finest leaves are inside or outside as a whole; a
//     finest leaf near the surface is core when its eight corners are inside,
//     at least delta (half a boundary cell plus the layer stack) from the
//     surface and no triangle meets its box. Finest cells are removed at
//     non-manifold staircase edges and vertices until the staircase is a
//     2-manifold.
//  5. Faces: the smaller side of two leaves, carrying the midpoints of finer
//     neighbors' corners on its edges; a 2:1 coarse leaf becomes a polyhedron.
//  6. Projection layer: one hexahedron between every staircase face and the
//     closest-point images of its vertices. Staircase faces take the patch of
//     their centre's nearest triangle; the partition is cleaned (flips of faces
//     with three foreign neighbors, of faces whose curve vertices are not one
//     run along their loop and of faces at non-manifold chain vertices); chain
//     vertices snap to the feature curve between their two patches, corner
//     vertices to the nearest feature corner, the others onto their patch.
//  7. Smoothing (Jacobi, relaxed): fine-band and transition vertices move to
//     the mean of their cells' centroids, surface vertices to the mean of their
//     boundary faces' centres (curve vertices to the mean of their two chain
//     neighbors) and back onto their patch or curve by a local walk; a move is
//     undone when an incident cell loses a positive float corner or
//     decomposition volume.
//  8. Layers: each projection column is split at the cumulative thicknesses
//     from the wall, giving layer_count thin hexahedra and the remainder; the
//     final repair search moves a feature-curve column's layer points (with
//     those above them) deeper along its offset direction, never thinner.
// Certification (prove): every cell, published as a polyhedron with OpenFOAM
// face semantics (a non-planar face is the fan from its vertex average), by
// the exact orientation of its decomposition tetrahedra (cell centre, face
// vertex average, edge) and a closed genus-0 boundary.
#include "hex_dominant.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace phx::mc::hex_dominant {
namespace {

using Vec = std::array<double, 3>;
using Clock = std::chrono::steady_clock;

inline Vec load(const double* p) { return {p[0], p[1], p[2]}; }
inline Vec sub(const Vec& a, const Vec& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline Vec add(const Vec& a, const Vec& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline Vec mul(const Vec& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
inline double dot(const Vec& a, const Vec& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline Vec cross(const Vec& a, const Vec& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline double norm(const Vec& a) { return std::sqrt(dot(a, a)); }
inline double seconds(Clock::time_point from) {
  return std::chrono::duration<double>(Clock::now() - from).count();
}

// Lattice and cell keys: three 21-bit coordinates.
inline constexpr int64_t kKeyLimit = int64_t{1} << 21;
inline uint64_t pack(int64_t i, int64_t j, int64_t k) {
  return (static_cast<uint64_t>(i) << 42) | (static_cast<uint64_t>(j) << 21) |
         static_cast<uint64_t>(k);
}
inline std::array<int64_t, 3> unpack(uint64_t key) {
  const uint64_t mask = static_cast<uint64_t>(kKeyLimit - 1);
  return {static_cast<int64_t>(key >> 42), static_cast<int64_t>((key >> 21) & mask),
          static_cast<int64_t>(key & mask)};
}

// Open-addressing map from 63-bit keys to int32 values.
class KeyMap {
 public:
  explicit KeyMap(std::size_t expected = 16) { rehash(capacity_for(expected)); }
  int32_t find(uint64_t key) const {
    for (uint64_t slot = mix(key) & mask_;; slot = (slot + 1) & mask_) {
      if (keys_[slot] == key) return values_[slot];
      if (keys_[slot] == kEmpty) return -1;
    }
  }
  // The stored value of key, inserting value when absent.
  int32_t emplace(uint64_t key, int32_t value) {
    if (2 * (size_ + 1) > keys_.size()) rehash(2 * keys_.size());
    for (uint64_t slot = mix(key) & mask_;; slot = (slot + 1) & mask_) {
      if (keys_[slot] == key) return values_[slot];
      if (keys_[slot] == kEmpty) {
        keys_[slot] = key;
        values_[slot] = value;
        ++size_;
        return value;
      }
    }
  }
  void assign(uint64_t key, int32_t value) {
    if (2 * (size_ + 1) > keys_.size()) rehash(2 * keys_.size());
    for (uint64_t slot = mix(key) & mask_;; slot = (slot + 1) & mask_) {
      if (keys_[slot] == key) {
        values_[slot] = value;
        return;
      }
      if (keys_[slot] == kEmpty) {
        keys_[slot] = key;
        values_[slot] = value;
        ++size_;
        return;
      }
    }
  }
  std::size_t size() const { return size_; }

 private:
  static constexpr uint64_t kEmpty = ~uint64_t{0};
  static std::size_t capacity_for(std::size_t expected) {
    std::size_t capacity = 16;
    while (capacity < 2 * expected + 2) capacity *= 2;
    return capacity;
  }
  static uint64_t mix(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
  }
  void rehash(std::size_t capacity) {
    std::vector<uint64_t> keys(capacity, kEmpty);
    std::vector<int32_t> values(capacity, 0);
    keys.swap(keys_);
    values.swap(values_);
    mask_ = capacity - 1;
    size_ = 0;
    for (std::size_t slot = 0; slot < keys.size(); ++slot) {
      if (keys[slot] != kEmpty) assign(keys[slot], values[slot]);
    }
  }
  std::vector<uint64_t> keys_;
  std::vector<int32_t> values_;
  uint64_t mask_ = 0;
  std::size_t size_ = 0;
};

// ------------------------------------------------------------------ geometry

Vec closest_on_triangle(const Vec& p, const Vec& a, const Vec& b, const Vec& c) {
  const Vec ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
  const double d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) return a;
  const Vec bp = sub(p, b);
  const double d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) return b;
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return add(a, mul(ab, d1 / (d1 - d3)));
  const Vec cp = sub(p, c);
  const double d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) return c;
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return add(a, mul(ac, d2 / (d2 - d6)));
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    return add(b, mul(sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
  }
  const double denominator = 1.0 / (va + vb + vc);
  return add(a, add(mul(ab, vb * denominator), mul(ac, vc * denominator)));
}

Vec closest_on_segment(const Vec& p, const Vec& a, const Vec& b) {
  const Vec ab = sub(b, a);
  const double length = dot(ab, ab);
  if (length <= 0.0) return a;
  const double t = std::clamp(dot(sub(p, a), ab) / length, 0.0, 1.0);
  return add(a, mul(ab, t));
}

// Separating-axis triangle/box overlap (Akenine-Moller) on a box grown by
// `pad`; conservative: touching counts as overlap.
bool triangle_box_overlap(const Vec& centre, double half, const Vec& a0, const Vec& b0,
                          const Vec& c0) {
  const Vec v0 = sub(a0, centre), v1 = sub(b0, centre), v2 = sub(c0, centre);
  const Vec e[3] = {sub(v1, v0), sub(v2, v1), sub(v0, v2)};
  for (int axis = 0; axis < 3; ++axis) {
    const double lo = std::min({v0[axis], v1[axis], v2[axis]});
    const double hi = std::max({v0[axis], v1[axis], v2[axis]});
    if (lo > half || hi < -half) return false;
  }
  const Vec n = cross(e[0], e[1]);
  const double r = half * (std::fabs(n[0]) + std::fabs(n[1]) + std::fabs(n[2]));
  if (std::fabs(dot(n, v0)) > r) return false;
  const Vec vs[3] = {v0, v1, v2};
  for (int i = 0; i < 3; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      Vec unit{0.0, 0.0, 0.0};
      unit[axis] = 1.0;
      const Vec l = cross(e[i], unit);
      const double p0 = dot(l, vs[0]), p1 = dot(l, vs[1]), p2 = dot(l, vs[2]);
      const double radius = half * (std::fabs(l[0]) + std::fabs(l[1]) + std::fabs(l[2]));
      if (std::min({p0, p1, p2}) > radius || std::max({p0, p1, p2}) < -radius) return false;
    }
  }
  return true;
}

// ------------------------------------------------------------------ surface

// Uniform bins of items by bounding box (triangles or feature edges).
struct Bins {
  Vec lo{0.0, 0.0, 0.0};
  double g = 1.0;
  int64_t extent = 1;  // bins per axis bound for shell searches
  KeyMap slots;
  std::vector<int32_t> start;
  std::vector<int32_t> items;

  std::array<int64_t, 3> cell(const Vec& p) const {
    return {static_cast<int64_t>(std::floor((p[0] - lo[0]) / g)),
            static_cast<int64_t>(std::floor((p[1] - lo[1]) / g)),
            static_cast<int64_t>(std::floor((p[2] - lo[2]) / g))};
  }
  bool valid(int64_t i, int64_t j, int64_t k) const {
    return i >= 0 && j >= 0 && k >= 0 && i < kKeyLimit && j < kKeyLimit && k < kKeyLimit;
  }
  // Items of bin (i, j, k) as [first, last) into items.
  std::pair<int32_t, int32_t> range(int64_t i, int64_t j, int64_t k) const {
    if (!valid(i, j, k)) return {0, 0};
    const int32_t slot = slots.find(pack(i, j, k));
    if (slot < 0) return {0, 0};
    return {start[slot], start[slot + 1]};
  }
};

template <class BoxOf>
void build_bins(Bins& bins, const Vec& lo, double g, int64_t extent, int64_t count, BoxOf box_of) {
  bins.lo = lo;
  bins.g = g;
  bins.extent = extent;
  std::vector<std::pair<uint64_t, int32_t>> pairs;
  pairs.reserve(static_cast<std::size_t>(count) * 2);
  for (int64_t item = 0; item < count; ++item) {
    Vec mn, mx;
    box_of(item, mn, mx);
    const auto a = bins.cell(mn), b = bins.cell(mx);
    for (int64_t i = a[0]; i <= b[0]; ++i) {
      for (int64_t j = a[1]; j <= b[1]; ++j) {
        for (int64_t k = a[2]; k <= b[2]; ++k) {
          pairs.emplace_back(pack(i, j, k), static_cast<int32_t>(item));
        }
      }
    }
  }
  std::sort(pairs.begin(), pairs.end());
  bins.items.resize(pairs.size());
  bins.start.clear();
  bins.slots = KeyMap(pairs.size() / 2 + 16);
  for (std::size_t n = 0; n < pairs.size(); ++n) {
    if (n == 0 || pairs[n].first != pairs[n - 1].first) {
      bins.slots.assign(pairs[n].first, static_cast<int32_t>(bins.start.size()));
      bins.start.push_back(static_cast<int32_t>(n));
    }
    bins.items[n] = pairs[n].second;
  }
  bins.start.push_back(static_cast<int32_t>(pairs.size()));
}

struct Surface {
  int64_t n = 0, m = 0;
  const double* x = nullptr;
  const int32_t* tri = nullptr;
  const int32_t* facets = nullptr;
  std::vector<int32_t> opposite;  // half-edge 3t + e -> its opposite half-edge
  std::vector<Vec> normal;
  std::vector<int32_t> patch;
  std::vector<int32_t> patch_facet;  // the facet of every patch (patches stop at facet changes)
  int32_t patch_count = 0;
  // Separating feature edges (between two different patches).
  std::vector<std::array<int32_t, 2>> edge_vertices;
  std::vector<std::array<int32_t, 2>> edge_patches;
  std::vector<int32_t> edge_triangles;  // a triangle on each separating feature edge
  std::vector<int64_t> vertex_edge_offsets;  // CSR surface vertex -> feature edges
  std::vector<int32_t> vertex_edges;
  std::vector<int32_t> corners;  // corner surface vertices
  std::vector<std::vector<int32_t>> corner_patches;
  int64_t feature_edge_count = 0;  // every feature edge, separating or not
  Vec lo{0, 0, 0}, hi{0, 0, 0};
  Bins triangle_bins;
  Bins edge_bins;

  Vec point(int64_t v) const { return load(x + 3 * v); }
  Vec corner_point(int32_t t, int e) const { return point(tri[3 * t + e]); }
};

int32_t build_surface(const Input& in, double feature_angle, double bin_size, Surface& s,
                      std::vector<double>& stats) {
  s.n = in.vertex_count;
  s.m = in.triangle_count;
  s.x = in.vertices;
  s.tri = in.triangles;
  s.facets = in.triangle_facets;
  const int64_t m = s.m;
  s.normal.resize(m);
  double volume = 0.0;
  for (int64_t t = 0; t < m; ++t) {
    const Vec a = s.corner_point(static_cast<int32_t>(t), 0);
    const Vec b = s.corner_point(static_cast<int32_t>(t), 1);
    const Vec c = s.corner_point(static_cast<int32_t>(t), 2);
    const Vec n = cross(sub(b, a), sub(c, a));
    const double length = norm(n);
    if (!(length > 0.0)) return PHX_MC_DEGENERATE_INPUT;
    s.normal[t] = mul(n, 1.0 / length);
    volume += dot(a, cross(b, c));
  }
  if (!(volume > 0.0)) return PHX_MC_INVALID_INPUT;
  // Opposite half-edges: each undirected edge in exactly two triangles,
  // traversed in opposite directions.
  s.opposite.assign(3 * m, -1);
  {
    KeyMap first(static_cast<std::size_t>(3 * m / 2 + 16));
    for (int64_t h = 0; h < 3 * m; ++h) {
      const int32_t a = s.tri[h], b = s.tri[(h / 3) * 3 + (h % 3 + 1) % 3];
      if (a == b) return PHX_MC_INVALID_INPUT;
      const uint64_t key = static_cast<uint64_t>(std::min(a, b)) * static_cast<uint64_t>(s.n) +
                           static_cast<uint64_t>(std::max(a, b));
      const int32_t other = first.emplace(key, static_cast<int32_t>(h));
      if (other == static_cast<int32_t>(h)) continue;
      if (s.opposite[other] >= 0) return PHX_MC_INVALID_INPUT;
      const int32_t oa = s.tri[other], ob = s.tri[(other / 3) * 3 + (other % 3 + 1) % 3];
      if (oa != b || ob != a) return PHX_MC_INVALID_INPUT;
      s.opposite[other] = static_cast<int32_t>(h);
      s.opposite[h] = other;
    }
    for (int64_t h = 0; h < 3 * m; ++h) {
      if (s.opposite[h] < 0) return PHX_MC_INVALID_INPUT;
    }
  }
  // Feature edges and patches.
  const double feature_cos = std::cos(feature_angle * (3.14159265358979323846 / 180.0));
  std::vector<uint8_t> feature(3 * m, 0);
  for (int64_t h = 0; h < 3 * m; ++h) {
    const int64_t t = h / 3, u = s.opposite[h] / 3;
    const bool facet_change = s.facets[t] != s.facets[u];
    feature[h] = facet_change || dot(s.normal[t], s.normal[u]) < feature_cos;
  }
  s.patch.assign(m, -1);
  std::vector<int32_t> stack;
  for (int64_t seed = 0; seed < m; ++seed) {
    if (s.patch[seed] >= 0) continue;
    const int32_t id = s.patch_count++;
    s.patch[seed] = id;
    s.patch_facet.push_back(s.facets[seed]);
    stack.assign(1, static_cast<int32_t>(seed));
    while (!stack.empty()) {
      const int32_t t = stack.back();
      stack.pop_back();
      for (int e = 0; e < 3; ++e) {
        const int64_t h = 3 * static_cast<int64_t>(t) + e;
        if (feature[h]) continue;
        const int32_t u = s.opposite[h] / 3;
        if (s.patch[u] < 0) {
          s.patch[u] = id;
          stack.push_back(u);
        }
      }
    }
  }
  std::vector<int32_t> degree(s.n, 0);
  for (int64_t h = 0; h < 3 * m; ++h) {
    if (!feature[h]) continue;
    if (h < s.opposite[h]) ++s.feature_edge_count;
    const int64_t t = h / 3, u = s.opposite[h] / 3;
    if (h > s.opposite[h] || s.patch[t] == s.patch[u]) continue;
    const int32_t a = s.tri[h], b = s.tri[t * 3 + (h % 3 + 1) % 3];
    s.edge_vertices.push_back({a, b});
    s.edge_patches.push_back({std::min(s.patch[t], s.patch[u]), std::max(s.patch[t], s.patch[u])});
    s.edge_triangles.push_back(static_cast<int32_t>(t));
    ++degree[a];
    ++degree[b];
  }
  s.vertex_edge_offsets.assign(s.n + 1, 0);
  for (int64_t v = 0; v < s.n; ++v) s.vertex_edge_offsets[v + 1] = s.vertex_edge_offsets[v] + degree[v];
  s.vertex_edges.resize(s.vertex_edge_offsets[s.n]);
  {
    std::vector<int64_t> fill(s.vertex_edge_offsets.begin(), s.vertex_edge_offsets.end() - 1);
    for (std::size_t e = 0; e < s.edge_vertices.size(); ++e) {
      for (const int32_t v : s.edge_vertices[e]) s.vertex_edges[fill[v]++] = static_cast<int32_t>(e);
    }
  }
  // Corners: feature degree other than 0 and 2, or three or more patches.
  std::vector<std::vector<int32_t>> around;
  {
    std::vector<int32_t> count(s.n, 0);
    for (int64_t h = 0; h < 3 * m; ++h) ++count[s.tri[h]];
    std::vector<int64_t> offsets(s.n + 1, 0);
    for (int64_t v = 0; v < s.n; ++v) offsets[v + 1] = offsets[v] + count[v];
    std::vector<int32_t> values(offsets[s.n]);
    std::vector<int64_t> fill(offsets.begin(), offsets.end() - 1);
    for (int64_t h = 0; h < 3 * m; ++h) values[fill[s.tri[h]]++] = s.patch[h / 3];
    for (int64_t v = 0; v < s.n; ++v) {
      if (degree[v] == 0) continue;
      std::vector<int32_t> patches(values.begin() + offsets[v], values.begin() + offsets[v + 1]);
      std::sort(patches.begin(), patches.end());
      patches.erase(std::unique(patches.begin(), patches.end()), patches.end());
      if (degree[v] != 2 || patches.size() >= 3) {
        s.corners.push_back(static_cast<int32_t>(v));
        s.corner_patches.push_back(std::move(patches));
      }
    }
  }
  // Bounding box and bins.
  s.lo = s.hi = s.point(0);
  for (int64_t v = 0; v < s.n; ++v) {
    for (int a = 0; a < 3; ++a) {
      s.lo[a] = std::min(s.lo[a], s.x[3 * v + a]);
      s.hi[a] = std::max(s.hi[a], s.x[3 * v + a]);
    }
  }
  const Vec bin_lo = sub(s.lo, {2 * bin_size, 2 * bin_size, 2 * bin_size});
  int64_t extent = 1;
  for (int a = 0; a < 3; ++a) {
    extent = std::max(extent, static_cast<int64_t>((s.hi[a] - bin_lo[a]) / bin_size) + 3);
  }
  if (extent >= kKeyLimit) return PHX_MC_INDEX_RANGE_ERROR;
  build_bins(s.triangle_bins, bin_lo, bin_size, extent, m, [&](int64_t t, Vec& mn, Vec& mx) {
    mn = mx = s.corner_point(static_cast<int32_t>(t), 0);
    for (int e = 1; e < 3; ++e) {
      const Vec p = s.corner_point(static_cast<int32_t>(t), e);
      for (int a = 0; a < 3; ++a) {
        mn[a] = std::min(mn[a], p[a]);
        mx[a] = std::max(mx[a], p[a]);
      }
    }
  });
  build_bins(s.edge_bins, bin_lo, bin_size, extent, static_cast<int64_t>(s.edge_vertices.size()),
             [&](int64_t e, Vec& mn, Vec& mx) {
               const Vec p = s.point(s.edge_vertices[e][0]), q = s.point(s.edge_vertices[e][1]);
               for (int a = 0; a < 3; ++a) {
                 mn[a] = std::min(p[a], q[a]);
                 mx[a] = std::max(p[a], q[a]);
               }
             });
  stats[13] = static_cast<double>(s.feature_edge_count);
  stats[14] = static_cast<double>(s.patch_count);
  return PHX_MC_OK;
}

struct Hit {
  double d2 = std::numeric_limits<double>::infinity();
  int32_t item = -1;
  Vec point{0, 0, 0};
};

// Squared distance from p to the box [lo, hi].
inline double box_distance2(const Vec& p, const Vec& lo, const Vec& hi) {
  double d2 = 0.0;
  for (int a = 0; a < 3; ++a) {
    const double d = std::max({lo[a] - p[a], 0.0, p[a] - hi[a]});
    d2 += d * d;
  }
  return d2;
}

// Nearest triangle (of patch `patch`, any when -1) by shell search. A bin or
// triangle box farther than the best distance so far holds no nearer point
// (the nearest point of a triangle lies in its box and in a bin it is filed
// under).
Hit nearest_triangle(const Surface& s, const Vec& p, int32_t patch) {
  const Bins& bins = s.triangle_bins;
  const auto c = bins.cell(p);
  Hit best;
  for (int64_t r = 0; r <= bins.extent; ++r) {
    for (int64_t di = -r; di <= r; ++di) {
      for (int64_t dj = -r; dj <= r; ++dj) {
        const bool face = std::max(std::llabs(di), std::llabs(dj)) == r;
        for (int64_t dk = -r; dk <= r; dk += (face ? 1 : std::max<int64_t>(1, 2 * r))) {
          const auto [first, last] = bins.range(c[0] + di, c[1] + dj, c[2] + dk);
          if (first == last) continue;
          const Vec lo{bins.lo[0] + static_cast<double>(c[0] + di) * bins.g,
                       bins.lo[1] + static_cast<double>(c[1] + dj) * bins.g,
                       bins.lo[2] + static_cast<double>(c[2] + dk) * bins.g};
          if (box_distance2(p, lo, add(lo, {bins.g, bins.g, bins.g})) > best.d2) continue;
          for (int32_t slot = first; slot < last; ++slot) {
            const int32_t t = bins.items[slot];
            if (patch >= 0 && s.patch[t] != patch) continue;
            const Vec a = s.corner_point(t, 0), b = s.corner_point(t, 1), e = s.corner_point(t, 2);
            const Vec tlo{std::min({a[0], b[0], e[0]}), std::min({a[1], b[1], e[1]}), std::min({a[2], b[2], e[2]})};
            const Vec thi{std::max({a[0], b[0], e[0]}), std::max({a[1], b[1], e[1]}), std::max({a[2], b[2], e[2]})};
            if (box_distance2(p, tlo, thi) > best.d2) continue;
            const Vec q = closest_on_triangle(p, a, b, e);
            const Vec d = sub(q, p);
            const double d2 = dot(d, d);
            if (d2 < best.d2 || (d2 == best.d2 && t < best.item)) best = {d2, t, q};
          }
        }
      }
    }
    const double reach = static_cast<double>(r) * bins.g;
    if (best.item >= 0 && best.d2 <= reach * reach) break;
  }
  return best;
}

// Whether a triangle lies within distance `radius` (< inf) of p.
bool triangle_within(const Surface& s, const Vec& p, double radius) {
  const Bins& bins = s.triangle_bins;
  const auto a = bins.cell(sub(p, {radius, radius, radius}));
  const auto b = bins.cell(add(p, {radius, radius, radius}));
  const double r2 = radius * radius;
  for (int64_t i = a[0]; i <= b[0]; ++i) {
    for (int64_t j = a[1]; j <= b[1]; ++j) {
      for (int64_t k = a[2]; k <= b[2]; ++k) {
        const auto [first, last] = bins.range(i, j, k);
        for (int32_t slot = first; slot < last; ++slot) {
          const int32_t t = bins.items[slot];
          const Vec u = s.corner_point(t, 0), v = s.corner_point(t, 1), w = s.corner_point(t, 2);
          const Vec lo{std::min({u[0], v[0], w[0]}), std::min({u[1], v[1], w[1]}), std::min({u[2], v[2], w[2]})};
          const Vec hi{std::max({u[0], v[0], w[0]}), std::max({u[1], v[1], w[1]}), std::max({u[2], v[2], w[2]})};
          if (box_distance2(p, lo, hi) >= r2) continue;
          const Vec d = sub(closest_on_triangle(p, u, v, w), p);
          if (dot(d, d) < r2) return true;
        }
      }
    }
  }
  return false;
}

// Nearest separating feature edge between patches (p0, p1) (any when p0 < 0).
Hit nearest_edge(const Surface& s, const Vec& p, int32_t p0, int32_t p1) {
  const Bins& bins = s.edge_bins;
  Hit best;
  if (s.edge_vertices.empty()) return best;
  const auto c = bins.cell(p);
  for (int64_t r = 0; r <= bins.extent; ++r) {
    for (int64_t di = -r; di <= r; ++di) {
      for (int64_t dj = -r; dj <= r; ++dj) {
        const bool face = std::max(std::llabs(di), std::llabs(dj)) == r;
        for (int64_t dk = -r; dk <= r; dk += (face ? 1 : std::max<int64_t>(1, 2 * r))) {
          const auto [first, last] = bins.range(c[0] + di, c[1] + dj, c[2] + dk);
          for (int32_t slot = first; slot < last; ++slot) {
            const int32_t e = bins.items[slot];
            if (p0 >= 0 && (s.edge_patches[e][0] != p0 || s.edge_patches[e][1] != p1)) continue;
            const Vec q = closest_on_segment(p, s.point(s.edge_vertices[e][0]),
                                             s.point(s.edge_vertices[e][1]));
            const Vec d = sub(q, p);
            const double d2 = dot(d, d);
            if (d2 < best.d2 || (d2 == best.d2 && e < best.item)) best = {d2, e, q};
          }
        }
      }
    }
    const double reach = static_cast<double>(r) * bins.g;
    if (best.item >= 0 && best.d2 <= reach * reach) break;
  }
  return best;
}

// Local constrained projection: from triangle `t`, walk to triangles of the
// same patch around the current triangle's corners (its vertex fans, which
// hold its edge neighbors) while they are closer. Edge neighbors alone stall
// where the closest point is a vertex whose nearer fan triangles share only
// that vertex.
Hit walk_triangle(const Surface& s, const Vec& p, int32_t t) {
  Hit best;
  best.item = t;
  best.point = closest_on_triangle(p, s.corner_point(t, 0), s.corner_point(t, 1), s.corner_point(t, 2));
  best.d2 = dot(sub(best.point, p), sub(best.point, p));
  for (;;) {
    Hit candidate = best;
    const int32_t patch = s.patch[best.item];
    for (int e = 0; e < 3; ++e) {
      // Outgoing half-edges of corner e: the next one is the opposite of the
      // half-edge entering the corner in the current triangle.
      const int64_t first = 3 * static_cast<int64_t>(best.item) + e;
      for (int64_t h = s.opposite[(first / 3) * 3 + (first % 3 + 2) % 3]; h != first;
           h = s.opposite[(h / 3) * 3 + (h % 3 + 2) % 3]) {
        const int32_t u = static_cast<int32_t>(h / 3);
        if (s.patch[u] != patch) continue;
        const Vec q = closest_on_triangle(p, s.corner_point(u, 0), s.corner_point(u, 1), s.corner_point(u, 2));
        const double d2 = dot(sub(q, p), sub(q, p));
        if (d2 < candidate.d2) candidate = {d2, u, q};
      }
    }
    if (candidate.item == best.item) return best;
    best = candidate;
  }
}

// Local projection on the feature curve of edge e's patch pair.
Hit walk_edge(const Surface& s, const Vec& p, int32_t e) {
  Hit best;
  best.item = e;
  best.point = closest_on_segment(p, s.point(s.edge_vertices[e][0]), s.point(s.edge_vertices[e][1]));
  best.d2 = dot(sub(best.point, p), sub(best.point, p));
  for (;;) {
    int32_t next = -1;
    Hit candidate = best;
    for (const int32_t v : s.edge_vertices[best.item]) {
      for (int64_t slot = s.vertex_edge_offsets[v]; slot < s.vertex_edge_offsets[v + 1]; ++slot) {
        const int32_t f = s.vertex_edges[slot];
        if (f == best.item || s.edge_patches[f] != s.edge_patches[e]) continue;
        const Vec q = closest_on_segment(p, s.point(s.edge_vertices[f][0]), s.point(s.edge_vertices[f][1]));
        const double d2 = dot(sub(q, p), sub(q, p));
        if (d2 < candidate.d2) {
          candidate = {d2, f, q};
          next = f;
        }
      }
    }
    if (next < 0) return best;
    best = candidate;
  }
}

// ------------------------------------------------------------ inside oracle

// Parity of +x ray crossings, per lattice line (J, K) of the lattice
// x = lo + I b. Exact: orient2d on the yz projection with the symbolic
// perturbation (y + e, z + e^2); orient3d against the triangle plane with the
// query perturbed by +e x.
class InsideOracle {
 public:
  InsideOracle(const Surface& s, const Vec& lo, double b) : s_(s), lo_(lo), b_(b), lines_(1024) {
    std::vector<std::pair<uint64_t, int32_t>> pairs;
    for (int64_t t = 0; t < s.m; ++t) {
      double mn[3], mx[3];
      for (int a = 1; a < 3; ++a) {
        mn[a] = mx[a] = s.x[3 * s.tri[3 * t] + a];
        for (int e = 1; e < 3; ++e) {
          mn[a] = std::min(mn[a], s.x[3 * s.tri[3 * t + e] + a]);
          mx[a] = std::max(mx[a], s.x[3 * s.tri[3 * t + e] + a]);
        }
      }
      const int64_t j0 = static_cast<int64_t>(std::floor((mn[1] - lo[1]) / b)) - 1;
      const int64_t j1 = static_cast<int64_t>(std::ceil((mx[1] - lo[1]) / b)) + 1;
      const int64_t k0 = static_cast<int64_t>(std::floor((mn[2] - lo[2]) / b)) - 1;
      const int64_t k1 = static_cast<int64_t>(std::ceil((mx[2] - lo[2]) / b)) + 1;
      for (int64_t j = std::max<int64_t>(j0, 0); j <= j1; ++j) {
        const double y = coordinate(1, j);
        if (y < mn[1] || y > mx[1]) continue;
        for (int64_t k = std::max<int64_t>(k0, 0); k <= k1; ++k) {
          const double z = coordinate(2, k);
          if (z < mn[2] || z > mx[2]) continue;
          pairs.emplace_back(pack(0, j, k), static_cast<int32_t>(t));
        }
      }
    }
    std::sort(pairs.begin(), pairs.end());
    candidates_.resize(pairs.size());
    for (std::size_t n = 0; n < pairs.size(); ++n) {
      if (n == 0 || pairs[n].first != pairs[n - 1].first) {
        lines_.assign(pairs[n].first, static_cast<int32_t>(line_start_.size()));
        line_start_.push_back(static_cast<int32_t>(n));
      }
      candidates_[n] = pairs[n].second;
    }
    line_start_.push_back(static_cast<int32_t>(pairs.size()));
    crossings_.resize(line_start_.size() - 1);
    resolved_.assign(line_start_.size() - 1, 0);
  }

  double coordinate(int axis, int64_t index) const {
    return lo_[axis] + static_cast<double>(index) * b_;
  }

  bool inside(int64_t i, int64_t j, int64_t k) {
    const int32_t line = lines_.find(pack(0, j, k));
    if (line < 0) return false;
    if (!resolved_[line]) resolve(line, j, k);
    const double p[3] = {coordinate(0, i), coordinate(1, j), coordinate(2, k)};
    int count = 0;
    for (const Crossing& c : crossings_[line]) {
      if (p[0] < c.xmin) {
        ++count;
      } else if (p[0] <= c.xmax) {
        const int32_t t = c.triangle;
        const int o = orient3d(s_.x + 3 * s_.tri[3 * t], s_.x + 3 * s_.tri[3 * t + 1],
                               s_.x + 3 * s_.tri[3 * t + 2], p);
        if (o * c.nx < 0) ++count;
      }
    }
    return (count & 1) != 0;
  }

 private:
  struct Crossing {
    double xmin, xmax;
    int32_t triangle;
    int nx;
  };

  static int perturbed_orient2d(const double* a, const double* b, const double* p) {
    const int o = orient2d(a, b, p);
    if (o != 0) return o;
    if (a[1] != b[1]) return a[1] > b[1] ? 1 : -1;  // a.z - b.z
    if (a[0] != b[0]) return b[0] > a[0] ? 1 : -1;  // b.y - a.y
    return 0;
  }

  void resolve(int32_t line, int64_t j, int64_t k) {
    const double p[2] = {coordinate(1, j), coordinate(2, k)};
    std::vector<Crossing>& out = crossings_[line];
    for (int32_t slot = line_start_[line]; slot < line_start_[line + 1]; ++slot) {
      const int32_t t = candidates_[slot];
      double q[3][2];
      double xmin = std::numeric_limits<double>::infinity(), xmax = -xmin;
      for (int e = 0; e < 3; ++e) {
        const double* v = s_.x + 3 * s_.tri[3 * t + e];
        q[e][0] = v[1];
        q[e][1] = v[2];
        xmin = std::min(xmin, v[0]);
        xmax = std::max(xmax, v[0]);
      }
      const int orientation = orient2d(q[0], q[1], q[2]);
      if (orientation == 0) continue;
      bool contained = true;
      for (int e = 0; e < 3 && contained; ++e) {
        contained = perturbed_orient2d(q[e], q[(e + 1) % 3], p) == orientation;
      }
      if (contained) out.push_back({xmin, xmax, t, orientation});
    }
    resolved_[line] = 1;
  }

  const Surface& s_;
  Vec lo_;
  double b_;
  KeyMap lines_;
  std::vector<int32_t> line_start_;
  std::vector<int32_t> candidates_;
  std::vector<std::vector<Crossing>> crossings_;
  std::vector<uint8_t> resolved_;
};

// ------------------------------------------------------------------- octree

struct Leaf {
  int32_t o[3];  // origin, finest lattice units
  uint8_t level;
  uint8_t core;
  int32_t cell;  // core cell row, -1 otherwise
};

struct Octree {
  int levels = 0;  // K
  Vec lo{0, 0, 0};
  double b = 1.0;
  int64_t dims[3] = {0, 0, 0};  // finest cells per axis
  std::vector<Leaf> leaves;
  KeyMap index;  // origin -> leaf

  int64_t size(int level) const { return int64_t{1} << (levels - level); }
  // Leaf containing finest cell q, -1 outside the lattice.
  int32_t find(int64_t q0, int64_t q1, int64_t q2) const {
    if (q0 < 0 || q1 < 0 || q2 < 0 || q0 >= dims[0] || q1 >= dims[1] || q2 >= dims[2]) return -1;
    for (int l = levels; l >= 0; --l) {
      const int64_t s = size(l);
      const int32_t leaf = index.find(pack(q0 / s * s, q1 / s * s, q2 / s * s));
      if (leaf >= 0 && leaves[leaf].level == l) return leaf;
    }
    return -1;
  }
  bool occupied(int64_t q0, int64_t q1, int64_t q2) const {
    const int32_t leaf = find(q0, q1, q2);
    return leaf >= 0 && leaves[leaf].core;
  }
};

std::vector<uint64_t> unique_sorted(std::vector<uint64_t> keys) {
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  return keys;
}

std::vector<uint64_t> dilate(const std::vector<uint64_t>& keys, int64_t radius, int64_t limit) {
  std::vector<uint64_t> out;
  out.reserve(keys.size() * static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1) * (2 * radius + 1)));
  for (const uint64_t key : keys) {
    const auto c = unpack(key);
    for (int64_t di = -radius; di <= radius; ++di) {
      for (int64_t dj = -radius; dj <= radius; ++dj) {
        for (int64_t dk = -radius; dk <= radius; ++dk) {
          const int64_t i = c[0] + di, j = c[1] + dj, k = c[2] + dk;
          if (i < 0 || j < 0 || k < 0 || i >= limit || j >= limit || k >= limit) continue;
          out.push_back(pack(i, j, k));
        }
      }
    }
  }
  return unique_sorted(std::move(out));
}

std::vector<uint64_t> coarsen(const std::vector<uint64_t>& keys, int shift) {
  std::vector<uint64_t> out;
  out.reserve(keys.size());
  for (const uint64_t key : keys) {
    const auto c = unpack(key);
    out.push_back(pack(c[0] >> shift, c[1] >> shift, c[2] >> shift));
  }
  return unique_sorted(std::move(out));
}

// Band width (in cells of each level) of the refinement around the surface.
inline constexpr int64_t kBandCells = 1;

// ------------------------------------------------------------------ the mesh

struct Builder {
  const Input& in;
  Surface& s;
  Octree tree;
  Mesh& mesh;
  double b = 1.0;
  double delta = 0.0;  // core clearance from the surface
  double layer_total = 0.0;
  std::vector<double> stats = std::vector<double>(kStatCount, 0.0);

  // Lattice vertex ids (core part).
  KeyMap vertex_ids{1024};
  std::vector<std::array<int64_t, 3>> lattice;  // per core vertex
  // Core part sizes.
  int64_t core_points = 0, core_faces = 0, core_cells = 0;
  int64_t core_face_entries = 0, core_cell_entries = 0;
  // Staircase (boundary of the core).
  std::vector<int32_t> stair_faces;           // core face rows
  std::vector<std::array<int32_t, 4>> stair_loop;  // their vertices (outward)
  std::vector<std::array<int32_t, 4>> stair_adjacent;  // face across loop edge i
  std::vector<int32_t> stair_patch;
  std::vector<int32_t> stair_vertex;  // core vertex -> staircase vertex row, -1
  std::vector<int32_t> stair_vertices;  // staircase vertex row -> core vertex
  std::vector<int32_t> surface_vertex;  // staircase vertex row -> mesh point (wall)
  std::vector<uint8_t> surface_kind;    // per staircase vertex
  std::vector<int32_t> surface_item;    // triangle (patch/corner) or feature edge (curve)
  std::vector<std::array<int32_t, 2>> surface_pair;  // patch pair of curve vertices
  std::vector<std::array<int32_t, 2>> chain_neighbors;  // staircase vertex rows, -1
  std::vector<int64_t> stair_vertex_face_offsets;
  std::vector<int32_t> stair_vertex_faces;
  struct Piece {
    int32_t face;
    int32_t count;
    std::array<int32_t, 4> loop;
    int32_t patch;
    int32_t core_face;
  };
  std::vector<Piece> pieces;
  // Pieces around every staircase vertex (with repeats) and the wall face of
  // every piece; column points (level 0 the wall, `levels` the staircase).
  std::vector<std::vector<int32_t>> row_pieces;
  std::vector<int32_t> piece_wall_face;
  int32_t levels = 1;
  std::vector<int32_t> column;
  // Per column: the wall offset direction of its layer points and whether
  // its wall vertex lies on a feature curve (two patch normals).
  std::vector<Vec> layer_direction;
  std::vector<uint8_t> curve_column;

  Builder(const Input& input, Surface& surface, Mesh& output) : in(input), s(surface), mesh(output) {}

  Vec lattice_point(int64_t i, int64_t j, int64_t k) const {
    return {tree.lo[0] + static_cast<double>(i) * b, tree.lo[1] + static_cast<double>(j) * b,
            tree.lo[2] + static_cast<double>(k) * b};
  }
  Vec point(int64_t v) const { return load(mesh.points.data() + 3 * v); }
  void set_point(int64_t v, const Vec& p) {
    for (int a = 0; a < 3; ++a) mesh.points[3 * v + a] = p[a];
  }
  int32_t add_point(const Vec& p) {
    mesh.points.insert(mesh.points.end(), p.begin(), p.end());
    return static_cast<int32_t>(mesh.points.size() / 3 - 1);
  }
  int32_t add_face(const int32_t* loop, int count, int32_t owner, int32_t neighbor) {
    mesh.face_vertices.insert(mesh.face_vertices.end(), loop, loop + count);
    mesh.face_offsets.push_back(static_cast<int64_t>(mesh.face_vertices.size()));
    mesh.face_owner.push_back(owner);
    mesh.face_neighbor.push_back(neighbor);
    return static_cast<int32_t>(mesh.face_owner.size() - 1);
  }

  int32_t octree_and_core();
  void build_pieces();
  void split_core_faces();
  int32_t assemble_core();
  int32_t staircase();
  void partition();
  int32_t project();
  void emit_layers();
  void place_layers();
  void smooth();
  int32_t certify();
};

int32_t Builder::octree_and_core() {
  const int K = tree.levels;
  const double h = std::ldexp(b, K);
  int64_t coarse[3];
  for (int a = 0; a < 3; ++a) {
    tree.lo[a] = h * std::floor(s.lo[a] / h) - h;
    coarse[a] = static_cast<int64_t>(std::ceil((s.hi[a] - tree.lo[a]) / h)) + 1;
    tree.dims[a] = coarse[a] << K;
    if (tree.dims[a] + 1 >= kKeyLimit) return PHX_MC_INDEX_RANGE_ERROR;
  }
  const int64_t limit = std::max({tree.dims[0], tree.dims[1], tree.dims[2]}) + 1;
  // Finest cells touching a triangle box.
  std::vector<uint64_t> touched;
  touched.reserve(static_cast<std::size_t>(s.m) * 2);
  for (int64_t t = 0; t < s.m; ++t) {
    Vec mn = s.corner_point(static_cast<int32_t>(t), 0), mx = mn;
    for (int e = 1; e < 3; ++e) {
      const Vec p = s.corner_point(static_cast<int32_t>(t), e);
      for (int a = 0; a < 3; ++a) {
        mn[a] = std::min(mn[a], p[a]);
        mx[a] = std::max(mx[a], p[a]);
      }
    }
    int64_t r0[3], r1[3];
    for (int a = 0; a < 3; ++a) {
      r0[a] = static_cast<int64_t>(std::floor((mn[a] - tree.lo[a]) / b)) - 1;
      r1[a] = static_cast<int64_t>(std::floor((mx[a] - tree.lo[a]) / b)) + 1;
    }
    for (int64_t i = r0[0]; i <= r1[0]; ++i) {
      for (int64_t j = r0[1]; j <= r1[1]; ++j) {
        for (int64_t k = r0[2]; k <= r1[2]; ++k) {
          const Vec p = lattice_point(i, j, k), q = lattice_point(i + 1, j + 1, k + 1);
          if (p[0] > mx[0] || p[1] > mx[1] || p[2] > mx[2] || q[0] < mn[0] || q[1] < mn[1] ||
              q[2] < mn[2]) {
            continue;
          }
          touched.push_back(pack(i, j, k));
        }
      }
    }
  }
  touched = unique_sorted(std::move(touched));
  // Split sets per level: R_{K-1} = band(par(touched)); R_l adds the
  // neighbors of split cells one level finer (2:1 balance).
  std::vector<KeyMap> split(static_cast<std::size_t>(std::max(K, 1)));
  {
    std::vector<uint64_t> finer;
    for (int l = K - 1; l >= 0; --l) {
      std::vector<uint64_t> keys = dilate(coarsen(touched, K - l), kBandCells, limit);
      if (!finer.empty()) {
        std::vector<uint64_t> balance = dilate(coarsen(finer, 1), 1, limit);
        keys.insert(keys.end(), balance.begin(), balance.end());
        keys = unique_sorted(std::move(keys));
      }
      split[l] = KeyMap(keys.size());
      for (const uint64_t key : keys) split[l].assign(key, 1);
      finer = std::move(keys);
    }
  }
  // Finest cells needing the detailed core test.
  const int64_t clearance_cells = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(delta / b)));
  KeyMap near_finest(1024);
  for (const uint64_t key : dilate(touched, clearance_cells, limit)) near_finest.assign(key, 1);
  // Leaves.
  std::vector<std::pair<int, std::array<int64_t, 3>>> stack;
  for (int64_t i = 0; i < coarse[0]; ++i) {
    for (int64_t j = 0; j < coarse[1]; ++j) {
      for (int64_t k = 0; k < coarse[2]; ++k) {
        stack.push_back({0, {i << K, j << K, k << K}});
        while (!stack.empty()) {
          const auto [level, o] = stack.back();
          stack.pop_back();
          const int64_t size = tree.size(level);
          if (level < K && split[level].find(pack(o[0] / size, o[1] / size, o[2] / size)) >= 0) {
            const int64_t half = size / 2;
            for (int c = 7; c >= 0; --c) {
              stack.push_back({level + 1, {o[0] + (c & 1) * half, o[1] + ((c >> 1) & 1) * half,
                                           o[2] + ((c >> 2) & 1) * half}});
            }
            continue;
          }
          Leaf leaf{{static_cast<int32_t>(o[0]), static_cast<int32_t>(o[1]), static_cast<int32_t>(o[2])},
                    static_cast<uint8_t>(level), 0, -1};
          tree.index.assign(pack(o[0], o[1], o[2]), static_cast<int32_t>(tree.leaves.size()));
          tree.leaves.push_back(leaf);
        }
      }
    }
  }
  // Core classification.
  KeyMap inside_memo(1024);
  InsideOracle oracle(s, tree.lo, b);
  auto inside = [&](int64_t i, int64_t j, int64_t k) {
    const uint64_t key = pack(i, j, k);
    const int32_t known = inside_memo.find(key);
    if (known >= 0) return known == 1;
    const bool value = oracle.inside(i, j, k);
    inside_memo.assign(key, value ? 1 : 0);
    return value;
  };
  KeyMap clear_memo(1024);
  auto clear = [&](int64_t i, int64_t j, int64_t k) {
    const uint64_t key = pack(i, j, k);
    const int32_t known = clear_memo.find(key);
    if (known >= 0) return known == 1;
    const bool value = !triangle_within(s, lattice_point(i, j, k), delta);
    clear_memo.assign(key, value ? 1 : 0);
    return value;
  };
  for (Leaf& leaf : tree.leaves) {
    const int64_t i = leaf.o[0], j = leaf.o[1], k = leaf.o[2];
    if (leaf.level < K || near_finest.find(pack(i, j, k)) < 0) {
      leaf.core = inside(i, j, k);
      continue;
    }
    bool core = true;
    for (int c = 0; c < 8 && core; ++c) {
      core = inside(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
    }
    for (int c = 0; c < 8 && core; ++c) core = clear(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
    if (core) {
      const Vec centre = lattice_point(i, j, k);
      const Vec mid = add(centre, {0.5 * b, 0.5 * b, 0.5 * b});
      const Bins& bins = s.triangle_bins;
      const auto a0 = bins.cell(centre), a1 = bins.cell(lattice_point(i + 1, j + 1, k + 1));
      for (int64_t bi = a0[0]; bi <= a1[0] && core; ++bi) {
        for (int64_t bj = a0[1]; bj <= a1[1] && core; ++bj) {
          for (int64_t bk = a0[2]; bk <= a1[2] && core; ++bk) {
            const auto [first, last] = bins.range(bi, bj, bk);
            for (int32_t slot = first; slot < last && core; ++slot) {
              const int32_t t = bins.items[slot];
              core = !triangle_box_overlap(mid, 0.5 * b * (1.0 + 1e-9), s.corner_point(t, 0),
                                           s.corner_point(t, 1), s.corner_point(t, 2));
            }
          }
        }
      }
    }
    leaf.core = core;
  }
  // Manifold staircase: remove finest core cells at non-manifold edges and
  // vertices (the involved core cell whose centre is nearest the surface).
  auto occupied = [&](int64_t i, int64_t j, int64_t k) { return tree.occupied(i, j, k); };
  auto nearest_distance = [&](int64_t i, int64_t j, int64_t k) {
    return nearest_triangle(s, add(lattice_point(i, j, k), {0.5 * b, 0.5 * b, 0.5 * b}), -1).d2;
  };
  std::vector<int32_t> work;
  for (int32_t leaf = 0; leaf < static_cast<int32_t>(tree.leaves.size()); ++leaf) {
    if (tree.leaves[leaf].core && tree.leaves[leaf].level == K) work.push_back(leaf);
  }
  int64_t removed = 0;
  auto remove_one = [&](const std::vector<std::array<int64_t, 3>>& cells) -> int32_t {
    int32_t chosen = -1;
    double best = std::numeric_limits<double>::infinity();
    for (const auto& c : cells) {
      const int32_t leaf = tree.find(c[0], c[1], c[2]);
      if (leaf < 0 || !tree.leaves[leaf].core) continue;
      if (tree.leaves[leaf].level != K) return -2;
      const double d = nearest_distance(c[0], c[1], c[2]);
      if (d < best) {
        best = d;
        chosen = leaf;
      }
    }
    if (chosen >= 0) {
      tree.leaves[chosen].core = 0;
      ++removed;
      const Leaf& l = tree.leaves[chosen];
      for (int64_t di = -1; di <= 1; ++di) {
        for (int64_t dj = -1; dj <= 1; ++dj) {
          for (int64_t dk = -1; dk <= 1; ++dk) {
            const int32_t other = tree.find(l.o[0] + di, l.o[1] + dj, l.o[2] + dk);
            if (other >= 0 && tree.leaves[other].core && tree.leaves[other].level == K) work.push_back(other);
          }
        }
      }
    }
    return chosen;
  };
  while (!work.empty()) {
    const int32_t leaf = work.back();
    work.pop_back();
    if (!tree.leaves[leaf].core) continue;
    const int64_t i = tree.leaves[leaf].o[0], j = tree.leaves[leaf].o[1], k = tree.leaves[leaf].o[2];
    bool changed = false;
    // Edges: the four cells around each of the cell's twelve edges.
    for (int axis = 0; axis < 3 && !changed; ++axis) {
      const int a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
      for (int e = 0; e < 4 && !changed; ++e) {
        // Edge along `axis` at the cell corner offsets (e & 1, e >> 1) in (a1, a2).
        int64_t base[3] = {i, j, k};
        base[a1] += (e & 1);
        base[a2] += (e >> 1);
        std::array<std::array<int64_t, 3>, 4> around;
        bool occ[4];
        for (int q = 0; q < 4; ++q) {
          std::array<int64_t, 3> c = {base[0], base[1], base[2]};
          c[a1] -= 1 - (q & 1);
          c[a2] -= 1 - (q >> 1);
          around[q] = c;
          occ[q] = occupied(c[0], c[1], c[2]);
        }
        // Diagonal pairs: (0, 3) and (1, 2).
        if ((occ[0] && occ[3] && !occ[1] && !occ[2]) || (occ[1] && occ[2] && !occ[0] && !occ[3])) {
          const int32_t done = remove_one({around[0], around[1], around[2], around[3]});
          if (done == -2) return PHX_MC_INTERNAL_ERROR;
          changed = done >= 0;
        }
      }
    }
    // Vertices: occupied and empty octants each face-connected.
    for (int corner = 0; corner < 8 && !changed; ++corner) {
      const int64_t vi = i + (corner & 1), vj = j + ((corner >> 1) & 1), vk = k + ((corner >> 2) & 1);
      std::array<std::array<int64_t, 3>, 8> octant;
      int mask = 0;
      for (int q = 0; q < 8; ++q) {
        octant[q] = {vi - 1 + (q & 1), vj - 1 + ((q >> 1) & 1), vk - 1 + ((q >> 2) & 1)};
        if (occupied(octant[q][0], octant[q][1], octant[q][2])) mask |= 1 << q;
      }
      if (mask == 0 || mask == 255) continue;
      auto components = [](int set, int* label) {
        int count = 0;
        for (int q = 0; q < 8; ++q) label[q] = -1;
        for (int q = 0; q < 8; ++q) {
          if (!((set >> q) & 1) || label[q] >= 0) continue;
          int queue[8], head = 0, tail = 0;
          queue[tail++] = q;
          label[q] = count;
          while (head < tail) {
            const int u = queue[head++];
            for (int bit = 0; bit < 3; ++bit) {
              const int w = u ^ (1 << bit);
              if (((set >> w) & 1) && label[w] < 0) {
                label[w] = count;
                queue[tail++] = w;
              }
            }
          }
          ++count;
        }
        return count;
      };
      int full_label[8], empty_label[8];
      const int full = components(mask, full_label);
      const int empty = components(~mask & 255, empty_label);
      if (full <= 1 && empty <= 1) continue;
      std::vector<std::array<int64_t, 3>> candidates;
      if (full > 1) {
        // The occupied component holding the fewest octants.
        int sizes[8] = {0};
        for (int q = 0; q < 8; ++q) {
          if (full_label[q] >= 0) ++sizes[full_label[q]];
        }
        int smallest = 0;
        for (int c = 1; c < full; ++c) {
          if (sizes[c] < sizes[smallest]) smallest = c;
        }
        for (int q = 0; q < 8; ++q) {
          if (full_label[q] == smallest) candidates.push_back(octant[q]);
        }
      } else {
        // Occupied octants touching two empty components.
        for (int q = 0; q < 8; ++q) {
          if (!((mask >> q) & 1)) continue;
          int seen = -1;
          bool bridge = false;
          for (int bit = 0; bit < 3; ++bit) {
            const int w = q ^ (1 << bit);
            if (empty_label[w] < 0) continue;
            if (seen >= 0 && seen != empty_label[w]) bridge = true;
            seen = empty_label[w];
          }
          if (bridge) candidates.push_back(octant[q]);
        }
      }
      const int32_t done = remove_one(candidates);
      if (done == -2) return PHX_MC_INTERNAL_ERROR;
      changed = done >= 0;
    }
    if (changed && tree.leaves[leaf].core) work.push_back(leaf);
  }
  stats[17] = static_cast<double>(removed);
  return PHX_MC_OK;
}

// Faces of the core: the smaller side of every two leaves.
int32_t Builder::assemble_core() {
  const int K = tree.levels;
  int32_t cells = 0;
  for (Leaf& leaf : tree.leaves) leaf.cell = leaf.core ? cells++ : -1;
  struct FaceRec {
    int32_t owner, neighbor;
    int8_t axis, sign;
    int64_t plane, u, v, size;
  };
  std::vector<FaceRec> records;
  records.reserve(static_cast<std::size_t>(cells) * 3 + 16);
  for (int32_t index = 0; index < static_cast<int32_t>(tree.leaves.size()); ++index) {
    const Leaf& leaf = tree.leaves[index];
    if (!leaf.core) continue;
    const int64_t size = tree.size(leaf.level);
    for (int side = 0; side < 6; ++side) {
      const int axis = side / 2, a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
      const int sign = (side & 1) ? 1 : -1;
      int64_t q[3] = {leaf.o[0], leaf.o[1], leaf.o[2]};
      q[axis] = sign > 0 ? leaf.o[axis] + size : leaf.o[axis] - 1;
      const int64_t plane = sign > 0 ? leaf.o[axis] + size : leaf.o[axis];
      const int32_t other = tree.find(q[0], q[1], q[2]);
      auto emit = [&](int32_t neighbor, int64_t u, int64_t v, int64_t s2) {
        records.push_back({leaf.cell, neighbor, static_cast<int8_t>(axis), static_cast<int8_t>(sign),
                           plane, u, v, s2});
      };
      if (other < 0) {
        emit(-1, leaf.o[a1], leaf.o[a2], size);
        continue;
      }
      const Leaf& o = tree.leaves[other];
      if (o.level == leaf.level) {
        if (!o.core) {
          emit(-1, leaf.o[a1], leaf.o[a2], size);
        } else if (index < other) {
          emit(o.cell, leaf.o[a1], leaf.o[a2], size);
        }
      } else if (o.level < leaf.level) {
        emit(o.core ? o.cell : -1, leaf.o[a1], leaf.o[a2], size);
      } else {
        const int64_t half = size / 2;
        for (int c = 0; c < 4; ++c) {
          int64_t r[3] = {q[0], q[1], q[2]};
          r[a1] = leaf.o[a1] + (c & 1) * half;
          r[a2] = leaf.o[a2] + (c >> 1) * half;
          const int32_t finer = tree.find(r[0], r[1], r[2]);
          if (finer < 0 || tree.leaves[finer].level != leaf.level + 1) return PHX_MC_INTERNAL_ERROR;
          if (!tree.leaves[finer].core) emit(-1, r[a1], r[a2], half);
        }
      }
    }
  }
  // Vertices: the corners of every face.
  auto corner = [](const FaceRec& f, int c) {
    const int a1 = (f.axis + 1) % 3, a2 = (f.axis + 2) % 3;
    std::array<int64_t, 3> p{};
    p[f.axis] = f.plane;
    const int du[4] = {0, 1, 1, 0}, dv[4] = {0, 0, 1, 1};
    p[a1] = f.u + du[c] * f.size;
    p[a2] = f.v + dv[c] * f.size;
    return p;
  };
  for (const FaceRec& f : records) {
    for (int c = 0; c < 4; ++c) {
      const auto p = corner(f, c);
      const int32_t id = vertex_ids.emplace(pack(p[0], p[1], p[2]), static_cast<int32_t>(lattice.size()));
      if (id == static_cast<int32_t>(lattice.size())) {
        lattice.push_back(p);
        add_point(lattice_point(p[0], p[1], p[2]));
      }
    }
  }
  // Polygons with the midpoints of finer neighbors.
  std::vector<int32_t> loop;
  for (const FaceRec& f : records) {
    loop.clear();
    int order[4] = {0, 1, 2, 3};
    if (f.sign < 0) std::swap(order[1], order[3]);
    for (int c = 0; c < 4; ++c) {
      const auto p = corner(f, order[c]), q = corner(f, order[(c + 1) % 4]);
      loop.push_back(vertex_ids.find(pack(p[0], p[1], p[2])));
      if (f.size >= 2) {
        const int32_t mid = vertex_ids.find(pack((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2));
        if (mid >= 0) loop.push_back(mid);
      }
    }
    add_face(loop.data(), static_cast<int>(loop.size()), f.owner, f.neighbor);
  }
  // Cells: hexahedra when six quadrilateral faces, else polyhedra.
  std::vector<int32_t> faces_of(cells, 0), quads_of(cells, 0);
  for (std::size_t f = 0; f < records.size(); ++f) {
    const bool quad = mesh.face_offsets[f + 1] - mesh.face_offsets[f] == 4;
    for (const int32_t c : {mesh.face_owner[f], mesh.face_neighbor[f]}) {
      if (c < 0) continue;
      ++faces_of[c];
      quads_of[c] += quad;
    }
  }
  for (const Leaf& leaf : tree.leaves) {
    if (!leaf.core) continue;
    const bool hex = faces_of[leaf.cell] == 6 && quads_of[leaf.cell] == 6;
    mesh.cell_kinds.push_back(hex ? kHexahedron : kPolyhedron);
    mesh.cell_roles.push_back(kCore);
    if (hex) {
      const int64_t size = tree.size(leaf.level);
      const int64_t dx[8] = {0, 1, 1, 0, 0, 1, 1, 0}, dy[8] = {0, 0, 1, 1, 0, 0, 1, 1},
                    dz[8] = {0, 0, 0, 0, 1, 1, 1, 1};
      for (int c = 0; c < 8; ++c) {
        mesh.cell_vertices.push_back(vertex_ids.find(
            pack(leaf.o[0] + dx[c] * size, leaf.o[1] + dy[c] * size, leaf.o[2] + dz[c] * size)));
      }
    }
    mesh.cell_vertex_offsets.push_back(static_cast<int64_t>(mesh.cell_vertices.size()));
  }
  // Staircase faces: finest quadrilaterals.
  for (std::size_t f = 0; f < records.size(); ++f) {
    if (records[f].neighbor >= 0) continue;
    if (records[f].size != 1 || mesh.face_offsets[f + 1] - mesh.face_offsets[f] != 4) {
      return PHX_MC_INTERNAL_ERROR;
    }
    stair_faces.push_back(static_cast<int32_t>(f));
    std::array<int32_t, 4> loop4;
    for (int c = 0; c < 4; ++c) loop4[c] = mesh.face_vertices[mesh.face_offsets[f] + c];
    stair_loop.push_back(loop4);
  }
  (void)K;
  core_points = static_cast<int64_t>(mesh.points.size() / 3);
  core_faces = static_cast<int64_t>(mesh.face_owner.size());
  core_cells = cells;
  core_face_entries = static_cast<int64_t>(mesh.face_vertices.size());
  core_cell_entries = static_cast<int64_t>(mesh.cell_vertices.size());
  stats[12] = static_cast<double>(stair_faces.size());
  return PHX_MC_OK;
}

// Staircase topology: the face across each loop edge (a closed 2-manifold).
int32_t Builder::staircase() {
  const int64_t count = static_cast<int64_t>(stair_faces.size());
  stair_adjacent.assign(count, {-1, -1, -1, -1});
  stair_vertex.assign(core_points, -1);
  KeyMap edges(static_cast<std::size_t>(2 * count + 16));
  const uint64_t n = static_cast<uint64_t>(core_points);
  for (int64_t f = 0; f < count; ++f) {
    for (int e = 0; e < 4; ++e) {
      const int32_t a = stair_loop[f][e], c = stair_loop[f][(e + 1) % 4];
      const uint64_t key = static_cast<uint64_t>(std::min(a, c)) * n + static_cast<uint64_t>(std::max(a, c));
      const int32_t first = edges.emplace(key, static_cast<int32_t>(4 * f + e));
      if (first == static_cast<int32_t>(4 * f + e)) continue;
      const int32_t g = first / 4, ge = first % 4;
      if (stair_adjacent[g][ge] >= 0) return PHX_MC_INTERNAL_ERROR;
      stair_adjacent[g][ge] = static_cast<int32_t>(f);
      stair_adjacent[f][e] = g;
    }
    for (int e = 0; e < 4; ++e) {
      const int32_t v = stair_loop[f][e];
      if (stair_vertex[v] < 0) {
        stair_vertex[v] = static_cast<int32_t>(stair_vertices.size());
        stair_vertices.push_back(v);
      }
    }
  }
  for (const auto& a : stair_adjacent) {
    for (const int32_t g : a) {
      if (g < 0) return PHX_MC_INTERNAL_ERROR;
    }
  }
  const int64_t vertices = static_cast<int64_t>(stair_vertices.size());
  std::vector<int32_t> degree(vertices, 0);
  for (const auto& loop : stair_loop) {
    for (const int32_t v : loop) ++degree[stair_vertex[v]];
  }
  stair_vertex_face_offsets.assign(vertices + 1, 0);
  for (int64_t v = 0; v < vertices; ++v) stair_vertex_face_offsets[v + 1] = stair_vertex_face_offsets[v] + degree[v];
  stair_vertex_faces.resize(stair_vertex_face_offsets[vertices]);
  std::vector<int64_t> fill(stair_vertex_face_offsets.begin(), stair_vertex_face_offsets.end() - 1);
  for (int64_t f = 0; f < count; ++f) {
    for (const int32_t v : stair_loop[f]) stair_vertex_faces[fill[stair_vertex[v]]++] = static_cast<int32_t>(f);
  }
  return PHX_MC_OK;
}

// Patch partition of the staircase faces and its cleanup.
void Builder::partition() {
  const int64_t count = static_cast<int64_t>(stair_faces.size());
  stair_patch.assign(count, 0);
  if (s.patch_count <= 1) return;
  for (int64_t f = 0; f < count; ++f) {
    Vec c{0, 0, 0};
    for (const int32_t v : stair_loop[f]) c = add(c, point(v));
    stair_patch[f] = s.patch[nearest_triangle(s, mul(c, 0.25), -1).item];
  }
  // Chain edges at a staircase vertex (edges whose two faces differ).
  auto chain_degree = [&](int32_t row, int32_t& patches) {
    int32_t edges = 0;
    int32_t seen[8];
    patches = 0;
    for (int64_t slot = stair_vertex_face_offsets[row]; slot < stair_vertex_face_offsets[row + 1]; ++slot) {
      const int32_t f = stair_vertex_faces[slot];
      const int32_t v = stair_vertices[row];
      for (int e = 0; e < 4; ++e) {
        if (stair_loop[f][e] != v) continue;
        // The loop edge leaving v; its far face.
        if (stair_patch[stair_adjacent[f][e]] != stair_patch[f]) ++edges;
      }
      bool known = false;
      for (int32_t q = 0; q < patches && q < 8; ++q) known = known || seen[q] == stair_patch[f];
      if (!known && patches < 8) seen[patches++] = stair_patch[f];
    }
    return edges;
  };
  std::vector<uint8_t> flips(count, 0);
  std::vector<int32_t> work(count);
  std::iota(work.begin(), work.end(), 0);
  std::reverse(work.begin(), work.end());
  int64_t flipped = 0;
  auto flip = [&](int32_t f, int32_t patch) {
    stair_patch[f] = patch;
    ++flips[f];
    ++flipped;
    for (const int32_t v : stair_loop[f]) {
      const int32_t row = stair_vertex[v];
      for (int64_t slot = stair_vertex_face_offsets[row]; slot < stair_vertex_face_offsets[row + 1]; ++slot) {
        work.push_back(stair_vertex_faces[slot]);
      }
    }
  };
  constexpr uint8_t kFlipLimit = 4;  // termination guard; leftovers fail certification
  while (!work.empty()) {
    const int32_t f = work.back();
    work.pop_back();
    if (flips[f] >= kFlipLimit) continue;
    // Most common foreign patch among the edge neighbors.
    int32_t best = -1, best_count = 0;
    for (int e = 0; e < 4; ++e) {
      const int32_t p = stair_patch[stair_adjacent[f][e]];
      if (p == stair_patch[f]) continue;
      int32_t c = 0;
      for (int q = 0; q < 4; ++q) c += stair_patch[stair_adjacent[f][q]] == p;
      if (c > best_count) {
        best = p;
        best_count = c;
      }
    }
    if (best_count >= 3) {
      flip(f, best);
      continue;
    }
    // Curve vertices of f must form one run of at most three along its loop.
    int on[4], total = 0;
    bool crossing = false;
    for (int c = 0; c < 4; ++c) {
      int32_t patches = 0;
      const int32_t edges = chain_degree(stair_vertex[stair_loop[f][c]], patches);
      on[c] = edges > 0;
      total += on[c];
      crossing = crossing || (edges > 2 && patches == 2);
    }
    const bool diagonal = total == 2 && on[0] == on[2];
    if ((total == 4 || diagonal || crossing) && best >= 0) flip(f, best);
  }
  stats[18] = static_cast<double>(flipped);
}

// Pieces of the staircase: a face whose patch changes across two consecutive
// loop edges (a step of the curve chain) splits along the diagonal opposite
// its corner vertex; the corner triangle joins the neighbors' patch, so the
// chain runs along the diagonal and no wall face holds three curve vertices.
// Along a zigzag the step faces of both patches share chain edges; a face
// whose chain neighbor already splits stays whole (its steps vanish with the
// neighbor's split), so the chain becomes the run of diagonals. A split whose
// triangle would still hold three chain vertices (the corner or the opposite
// vertex stays on a chain through its other faces) flattens its prism onto
// the curve and is withdrawn; withdrawals repeat until none remains. The
// core face splits likewise (its cell becomes a polyhedron).
void Builder::build_pieces() {
  const int64_t count = static_cast<int64_t>(stair_faces.size());
  const int64_t vertices = static_cast<int64_t>(stair_vertices.size());
  std::vector<uint8_t> withdrawn(count, 0), splits(count, 0);
  std::vector<int32_t> first_patch(vertices), multi(vertices);
  int64_t split = 0;
  for (bool again = true; again;) {
    again = false;
    pieces.clear();
    pieces.reserve(static_cast<std::size_t>(count) + 64);
    std::fill(splits.begin(), splits.end(), 0);
    split = 0;
    for (int64_t f = 0; f < count; ++f) {
      const auto& loop = stair_loop[f];
      bool chain[4];
      for (int e = 0; e < 4; ++e) chain[e] = stair_patch[stair_adjacent[f][e]] != stair_patch[f];
      int corner = -1;
      for (int e = 0; e < 4 && !withdrawn[f]; ++e) {
        const int next = (e + 1) % 4;
        if (chain[e] && chain[next] && !chain[(e + 2) % 4] && !chain[(e + 3) % 4] &&
            stair_patch[stair_adjacent[f][e]] == stair_patch[stair_adjacent[f][next]]) {
          corner = next;  // loop[next] is shared by the two chain edges
        }
      }
      if (corner >= 0 && (splits[stair_adjacent[f][(corner + 3) % 4]] || splits[stair_adjacent[f][corner]])) {
        corner = -1;
      }
      if (corner < 0) {
        pieces.push_back({static_cast<int32_t>(f), 4, loop, stair_patch[f], -1});
        continue;
      }
      splits[f] = 1;
      ++split;
      const int32_t before = loop[(corner + 3) % 4], at = loop[corner], after = loop[(corner + 1) % 4],
                    opposite = loop[(corner + 2) % 4];
      pieces.push_back({static_cast<int32_t>(f), 3, {before, at, after, -1},
                        stair_patch[stair_adjacent[f][(corner + 3) % 4]], -1});
      pieces.push_back({static_cast<int32_t>(f), 3, {after, opposite, before, -1}, stair_patch[f], -1});
    }
    if (split == 0) break;
    std::fill(first_patch.begin(), first_patch.end(), -1);
    std::fill(multi.begin(), multi.end(), 0);
    for (const Piece& piece : pieces) {
      for (int c = 0; c < piece.count; ++c) {
        const int32_t row = stair_vertex[piece.loop[c]];
        if (first_patch[row] < 0) first_patch[row] = piece.patch;
        if (first_patch[row] != piece.patch) multi[row] = 1;
      }
    }
    for (const Piece& piece : pieces) {
      if (piece.count != 3 || withdrawn[piece.face]) continue;
      int on = 0;
      for (int c = 0; c < 3; ++c) on += multi[stair_vertex[piece.loop[c]]];
      if (on == 3) {
        withdrawn[piece.face] = 1;
        again = true;
      }
    }
  }
  stats[27] = static_cast<double>(split);
}

// Rebuild the core faces with every split staircase face as two triangles.
void Builder::split_core_faces() {
  std::vector<int32_t> first_piece(core_faces, -1);
  for (std::size_t p = 0; p < pieces.size(); ++p) {
    const int32_t face = stair_faces[pieces[p].face];
    if (first_piece[face] < 0) first_piece[face] = static_cast<int32_t>(p);
  }
  std::vector<int64_t> offsets{0};
  std::vector<int32_t> values, owners, neighbors;
  std::vector<uint8_t> changed(core_cells, 0);
  for (int64_t f = 0; f < core_faces; ++f) {
    const int32_t p = first_piece[f];
    const bool split = p >= 0 && pieces[p].count == 3;
    if (!split) {
      values.insert(values.end(), mesh.face_vertices.begin() + mesh.face_offsets[f],
                    mesh.face_vertices.begin() + mesh.face_offsets[f + 1]);
      offsets.push_back(static_cast<int64_t>(values.size()));
      owners.push_back(mesh.face_owner[f]);
      neighbors.push_back(mesh.face_neighbor[f]);
      if (p >= 0) pieces[p].core_face = static_cast<int32_t>(owners.size() - 1);
      continue;
    }
    changed[mesh.face_owner[f]] = 1;
    for (int32_t q = p; q < p + 2; ++q) {
      values.insert(values.end(), pieces[q].loop.begin(), pieces[q].loop.begin() + 3);
      offsets.push_back(static_cast<int64_t>(values.size()));
      owners.push_back(mesh.face_owner[f]);
      neighbors.push_back(-1);
      pieces[q].core_face = static_cast<int32_t>(owners.size() - 1);
    }
  }
  mesh.face_offsets.swap(offsets);
  mesh.face_vertices.swap(values);
  mesh.face_owner.swap(owners);
  mesh.face_neighbor.swap(neighbors);
  std::vector<int64_t> cell_offsets{0};
  std::vector<int32_t> cell_values;
  for (int64_t c = 0; c < core_cells; ++c) {
    if (changed[c]) mesh.cell_kinds[c] = kPolyhedron;
    if (mesh.cell_kinds[c] == kHexahedron) {
      cell_values.insert(cell_values.end(), mesh.cell_vertices.begin() + mesh.cell_vertex_offsets[c],
                         mesh.cell_vertices.begin() + mesh.cell_vertex_offsets[c + 1]);
    }
    cell_offsets.push_back(static_cast<int64_t>(cell_values.size()));
  }
  mesh.cell_vertex_offsets.swap(cell_offsets);
  mesh.cell_vertices.swap(cell_values);
  core_faces = static_cast<int64_t>(mesh.face_owner.size());
  core_face_entries = static_cast<int64_t>(mesh.face_vertices.size());
  core_cell_entries = static_cast<int64_t>(mesh.cell_vertices.size());
}

// Wall images of the staircase vertices, classified by the patches of the
// pieces around them: one patch, a feature curve between two, a corner.
int32_t Builder::project() {
  const int64_t vertices = static_cast<int64_t>(stair_vertices.size());
  surface_vertex.assign(vertices, -1);
  surface_kind.assign(vertices, kPatch);
  surface_item.assign(vertices, -1);
  surface_pair.assign(vertices, {-1, -1});
  chain_neighbors.assign(vertices, {-1, -1});
  // Piece edges: the two pieces of every undirected edge.
  KeyMap edge_slot(pieces.size() * 2 + 16);
  std::vector<std::array<int32_t, 2>> edge_pieces;
  std::vector<std::array<int32_t, 2>> edge_ends;
  const uint64_t n = static_cast<uint64_t>(core_points);
  row_pieces.assign(vertices, {});
  auto& around = row_pieces;
  for (std::size_t p = 0; p < pieces.size(); ++p) {
    const Piece& piece = pieces[p];
    for (int e = 0; e < piece.count; ++e) {
      const int32_t a = piece.loop[e], c = piece.loop[(e + 1) % piece.count];
      around[stair_vertex[a]].push_back(static_cast<int32_t>(p));
      const uint64_t key = static_cast<uint64_t>(std::min(a, c)) * n + static_cast<uint64_t>(std::max(a, c));
      const int32_t slot = edge_slot.emplace(key, static_cast<int32_t>(edge_pieces.size()));
      if (slot == static_cast<int32_t>(edge_pieces.size())) {
        edge_pieces.push_back({static_cast<int32_t>(p), -1});
        edge_ends.push_back({a, c});
      } else {
        edge_pieces[slot][1] = static_cast<int32_t>(p);
      }
    }
  }
  std::vector<std::vector<int32_t>> chain(vertices);
  for (std::size_t e = 0; e < edge_pieces.size(); ++e) {
    if (edge_pieces[e][1] < 0) return PHX_MC_INTERNAL_ERROR;
    if (pieces[edge_pieces[e][0]].patch == pieces[edge_pieces[e][1]].patch) continue;
    chain[stair_vertex[edge_ends[e][0]]].push_back(stair_vertex[edge_ends[e][1]]);
    chain[stair_vertex[edge_ends[e][1]]].push_back(stair_vertex[edge_ends[e][0]]);
  }
  int64_t curve = 0, corner = 0;
  for (int64_t row = 0; row < vertices; ++row) {
    const int32_t v = stair_vertices[row];
    const Vec p = point(v);
    std::vector<int32_t> patches;
    for (const int32_t q : around[row]) patches.push_back(pieces[q].patch);
    std::sort(patches.begin(), patches.end());
    patches.erase(std::unique(patches.begin(), patches.end()), patches.end());
    Hit hit;
    if (patches.size() == 1) {
      hit = nearest_triangle(s, p, patches[0]);
      surface_kind[row] = kPatch;
    } else if (patches.size() == 2) {
      hit = nearest_edge(s, p, patches[0], patches[1]);
      if (hit.item < 0) hit = nearest_edge(s, p, -1, -1);
      surface_kind[row] = kFeature;
      surface_pair[row] = {patches[0], patches[1]};
      if (chain[row].size() == 2) chain_neighbors[row] = {chain[row][0], chain[row][1]};
      ++curve;
    } else {
      // The nearest corner sharing the most patches.
      int32_t chosen = -1;
      double best = std::numeric_limits<double>::infinity();
      std::size_t best_shared = 0;
      for (std::size_t c = 0; c < s.corners.size(); ++c) {
        std::size_t shared = 0;
        for (const int32_t q : patches) {
          shared += std::binary_search(s.corner_patches[c].begin(), s.corner_patches[c].end(), q);
        }
        const Vec d = sub(s.point(s.corners[c]), p);
        const double d2 = dot(d, d);
        if (shared > best_shared || (shared == best_shared && d2 < best)) {
          best_shared = shared;
          best = d2;
          chosen = static_cast<int32_t>(c);
        }
      }
      surface_kind[row] = kCorner;
      if (chosen >= 0) {
        hit.item = s.corners[chosen];
        hit.point = s.point(s.corners[chosen]);
      } else {
        hit = nearest_edge(s, p, -1, -1);
      }
      ++corner;
    }
    if (hit.item < 0) {
      hit = nearest_triangle(s, p, -1);
      surface_kind[row] = kPatch;
    }
    surface_item[row] = hit.item;
    surface_vertex[row] = add_point(hit.point);
  }
  stats[19] = static_cast<double>(curve);
  stats[20] = static_cast<double>(corner);
  return PHX_MC_OK;
}

// Projection cells (levels == 1) or columns split into levels cells: a
// hexahedron per quadrilateral piece and level, a prism per triangle. Layer
// points are placed by place_layers.
void Builder::emit_layers() {
  mesh.points.resize(3 * (core_points + static_cast<int64_t>(stair_vertices.size())));
  mesh.face_offsets.resize(core_faces + 1);
  mesh.face_vertices.resize(core_face_entries);
  mesh.face_owner.resize(core_faces);
  mesh.face_neighbor.resize(core_faces);
  mesh.cell_kinds.resize(core_cells);
  mesh.cell_roles.resize(core_cells);
  mesh.cell_vertex_offsets.resize(core_cells + 1);
  mesh.cell_vertices.resize(core_cell_entries);
  const int64_t columns = static_cast<int64_t>(stair_vertices.size());
  column.assign(static_cast<std::size_t>(columns) * (levels + 1), -1);
  for (int64_t row = 0; row < columns; ++row) {
    column[row * (levels + 1)] = surface_vertex[row];
    column[row * (levels + 1) + levels] = stair_vertices[row];
    for (int32_t k = 1; k < levels; ++k) column[row * (levels + 1) + k] = add_point(point(surface_vertex[row]));
  }
  auto at = [&](int32_t v, int32_t level) { return column[stair_vertex[v] * (levels + 1) + level]; };
  const int64_t count = static_cast<int64_t>(pieces.size());
  const int32_t first_cell = static_cast<int32_t>(core_cells);
  // Cell of piece p at level k (k = 0 at the wall).
  auto cell_of = [&](int64_t p, int32_t k) { return first_cell + static_cast<int32_t>(p * levels + k); };
  for (int64_t p = 0; p < count; ++p) {
    const Piece& piece = pieces[p];
    mesh.face_neighbor[piece.core_face] = cell_of(p, levels - 1);
    for (int32_t k = 0; k < levels; ++k) {
      mesh.cell_kinds.push_back(piece.count == 4 ? kHexahedron : kPrism);
      mesh.cell_roles.push_back(levels > 1 && k < levels - 1 ? kLayer : kProjection);
      for (int c = 0; c < piece.count; ++c) mesh.cell_vertices.push_back(at(piece.loop[c], k + 1));
      for (int c = 0; c < piece.count; ++c) mesh.cell_vertices.push_back(at(piece.loop[c], k));
      mesh.cell_vertex_offsets.push_back(static_cast<int64_t>(mesh.cell_vertices.size()));
    }
  }
  int32_t loop[4];
  piece_wall_face.assign(count, -1);
  for (int64_t p = 0; p < count; ++p) {
    const Piece& piece = pieces[p];
    // The wall face (outward along the staircase normal) and the faces
    // between levels.
    for (int c = 0; c < piece.count; ++c) loop[c] = at(piece.loop[c], 0);
    piece_wall_face[p] = add_face(loop, piece.count, cell_of(p, 0), -1);
    for (int32_t k = 1; k < levels; ++k) {
      for (int c = 0; c < piece.count; ++c) loop[c] = at(piece.loop[c], k);
      add_face(loop, piece.count, cell_of(p, k), cell_of(p, k - 1));
    }
  }
  // Side faces, owned by the piece whose loop runs a -> b at the second
  // sighting of the edge (the first runs b -> a).
  KeyMap seen(static_cast<std::size_t>(2 * count + 16));
  const uint64_t n = static_cast<uint64_t>(core_points);
  for (int64_t p = 0; p < count; ++p) {
    const Piece& piece = pieces[p];
    for (int e = 0; e < piece.count; ++e) {
      const int32_t a = piece.loop[e], c = piece.loop[(e + 1) % piece.count];
      const uint64_t key = static_cast<uint64_t>(std::min(a, c)) * n + static_cast<uint64_t>(std::max(a, c));
      const int32_t other = seen.emplace(key, static_cast<int32_t>(p));
      if (other == static_cast<int32_t>(p)) continue;
      for (int32_t k = 0; k < levels; ++k) {
        loop[0] = at(a, k + 1);
        loop[1] = at(c, k + 1);
        loop[2] = at(c, k);
        loop[3] = at(a, k);
        add_face(loop, 4, cell_of(p, k), cell_of(other, k));
      }
    }
  }
}

// ---------------------------------------------------------- cell geometry

struct Topology {
  std::vector<int64_t> cell_face_offsets;
  std::vector<int32_t> cell_faces;  // signed: face + 1 (owner) or -(face + 1)
  std::vector<int64_t> vertex_cell_offsets;
  std::vector<int32_t> vertex_cells;
};

Topology topology(const Mesh& mesh) {
  Topology t;
  const int64_t cells = static_cast<int64_t>(mesh.cell_kinds.size());
  const int64_t faces = static_cast<int64_t>(mesh.face_owner.size());
  std::vector<int32_t> count(cells, 0);
  for (int64_t f = 0; f < faces; ++f) {
    ++count[mesh.face_owner[f]];
    if (mesh.face_neighbor[f] >= 0) ++count[mesh.face_neighbor[f]];
  }
  t.cell_face_offsets.assign(cells + 1, 0);
  for (int64_t c = 0; c < cells; ++c) t.cell_face_offsets[c + 1] = t.cell_face_offsets[c] + count[c];
  t.cell_faces.resize(t.cell_face_offsets[cells]);
  std::vector<int64_t> fill(t.cell_face_offsets.begin(), t.cell_face_offsets.end() - 1);
  for (int64_t f = 0; f < faces; ++f) {
    t.cell_faces[fill[mesh.face_owner[f]]++] = static_cast<int32_t>(f + 1);
    if (mesh.face_neighbor[f] >= 0) t.cell_faces[fill[mesh.face_neighbor[f]]++] = -static_cast<int32_t>(f + 1);
  }
  // Vertex -> cells (unique per cell).
  const int64_t points = static_cast<int64_t>(mesh.points.size() / 3);
  std::vector<std::pair<int32_t, int32_t>> pairs;
  pairs.reserve(t.cell_faces.size() * 4);
  for (int64_t c = 0; c < cells; ++c) {
    for (int64_t slot = t.cell_face_offsets[c]; slot < t.cell_face_offsets[c + 1]; ++slot) {
      const int64_t f = std::abs(t.cell_faces[slot]) - 1;
      for (int64_t q = mesh.face_offsets[f]; q < mesh.face_offsets[f + 1]; ++q) {
        pairs.emplace_back(mesh.face_vertices[q], static_cast<int32_t>(c));
      }
    }
  }
  std::sort(pairs.begin(), pairs.end());
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  t.vertex_cell_offsets.assign(points + 1, 0);
  for (const auto& p : pairs) ++t.vertex_cell_offsets[p.first + 1];
  for (int64_t v = 0; v < points; ++v) t.vertex_cell_offsets[v + 1] += t.vertex_cell_offsets[v];
  t.vertex_cells.resize(pairs.size());
  for (std::size_t n = 0; n < pairs.size(); ++n) t.vertex_cells[n] = pairs[n].second;
  return t;
}

// OpenFOAM face centre and area vector (triangles about the vertex average).
void face_geometry(const Mesh& mesh, int64_t f, Vec& centre, Vec& area, Vec& average) {
  const int64_t first = mesh.face_offsets[f], last = mesh.face_offsets[f + 1];
  const double count = static_cast<double>(last - first);
  average = {0, 0, 0};
  for (int64_t q = first; q < last; ++q) average = add(average, load(mesh.points.data() + 3 * mesh.face_vertices[q]));
  average = mul(average, 1.0 / count);
  area = {0, 0, 0};
  Vec weighted{0, 0, 0};
  double total = 0.0;
  for (int64_t q = first; q < last; ++q) {
    const int64_t r = q + 1 < last ? q + 1 : first;
    const Vec p = load(mesh.points.data() + 3 * mesh.face_vertices[q]);
    const Vec n = load(mesh.points.data() + 3 * mesh.face_vertices[r]);
    const Vec a = mul(cross(sub(p, average), sub(n, average)), 0.5);
    const double magnitude = norm(a);
    area = add(area, a);
    weighted = add(weighted, mul(add(add(p, n), average), magnitude / 3.0));
    total += magnitude;
  }
  centre = total > 0.0 ? mul(weighted, 1.0 / total) : average;
}

// A face's vertex-average fan does not fold: every fan triangle's area
// vector points along the loop's area vector N (the certifier's refusal of a
// folded fan), here with a relative margin of 1e-6 of N . N per triangle so
// that rounding cannot accept a folded fan.
bool fan_unfolded(const Mesh& mesh, int64_t f) {
  const int64_t first = mesh.face_offsets[f], last = mesh.face_offsets[f + 1];
  const double* x = mesh.points.data();
  Vec average{0, 0, 0}, normal{0, 0, 0};
  for (int64_t q = first; q < last; ++q) {
    const int64_t r = q + 1 < last ? q + 1 : first;
    average = add(average, load(x + 3 * mesh.face_vertices[q]));
    normal = add(normal, cross(load(x + 3 * mesh.face_vertices[q]), load(x + 3 * mesh.face_vertices[r])));
  }
  const double count = static_cast<double>(last - first);
  average = mul(average, 1.0 / count);
  const double floor = 1e-6 * dot(normal, normal) / count;
  for (int64_t q = first; q < last; ++q) {
    const int64_t r = q + 1 < last ? q + 1 : first;
    const Vec a = cross(sub(load(x + 3 * mesh.face_vertices[q]), average), sub(load(x + 3 * mesh.face_vertices[r]), average));
    if (!(dot(a, normal) > floor)) return false;
  }
  return true;
}

// OpenFOAM cell centroid (pyramids about the mean of the face centres).
Vec cell_centroid(const Mesh& mesh, const Topology& t, int64_t c) {
  Vec estimate{0, 0, 0};
  const int64_t first = t.cell_face_offsets[c], last = t.cell_face_offsets[c + 1];
  Vec centres[64], areas[64];
  const int64_t count = std::min<int64_t>(last - first, 64);
  for (int64_t slot = 0; slot < count; ++slot) {
    const int32_t signed_face = t.cell_faces[first + slot];
    Vec average;
    face_geometry(mesh, std::abs(signed_face) - 1, centres[slot], areas[slot], average);
    if (signed_face < 0) areas[slot] = mul(areas[slot], -1.0);
    estimate = add(estimate, centres[slot]);
  }
  estimate = mul(estimate, 1.0 / static_cast<double>(count));
  double volume = 0.0;
  Vec moment{0, 0, 0};
  for (int64_t slot = 0; slot < count; ++slot) {
    const double v = dot(areas[slot], sub(centres[slot], estimate)) / 3.0;
    volume += v;
    moment = add(moment, mul(add(mul(centres[slot], 0.75), mul(estimate, 0.25)), v));
  }
  return volume > 0.0 ? mul(moment, 1.0 / volume) : estimate;
}

// The eight corner Jacobians of hexahedron c, above a relative margin.
bool hexahedron_corners_positive(const Mesh& mesh, int64_t c) {
  const double* x = mesh.points.data();
  const int32_t* h = mesh.cell_vertices.data() + mesh.cell_vertex_offsets[c];
  Vec p[8];
  for (int i = 0; i < 8; ++i) p[i] = load(x + 3 * h[i]);
  // Corner neighbors in the reference order (x, y, z directions).
  static constexpr int kCorner[8][3] = {{1, 3, 4}, {2, 0, 5}, {3, 1, 6}, {0, 2, 7},
                                        {7, 5, 0}, {4, 6, 1}, {5, 7, 2}, {6, 4, 3}};
  for (int i = 0; i < 8; ++i) {
    const Vec a = sub(p[kCorner[i][0]], p[i]);
    const Vec b = sub(p[kCorner[i][1]], p[i]);
    const Vec d = sub(p[kCorner[i][2]], p[i]);
    if (!(dot(a, cross(b, d)) > 1e-6 * norm(a) * norm(b) * norm(d))) return false;
  }
  return true;
}

// Float screen of a cell: unfolded face fans, its decomposition tetrahedra
// (OpenFOAM centroid, face vertex average, edge; the certified
// decomposition) above a relative margin, and for hexahedra also the eight
// corner Jacobians. face(f, unfolded) yields face f's vertex average and sets
// whether its fan is unfolded, from whatever geometry the caller keeps.
template <class Face>
bool cell_screen(const Mesh& mesh, const Topology& t, int64_t c, const Vec& centre, Face face) {
  const double* x = mesh.points.data();
  if (mesh.cell_kinds[c] == kHexahedron && !hexahedron_corners_positive(mesh, c)) return false;
  for (int64_t slot = t.cell_face_offsets[c]; slot < t.cell_face_offsets[c + 1]; ++slot) {
    const int32_t signed_face = t.cell_faces[slot];
    const int64_t f = std::abs(signed_face) - 1;
    bool unfolded = false;
    const Vec average = face(f, unfolded);
    if (!unfolded) return false;
    const int64_t first = mesh.face_offsets[f], last = mesh.face_offsets[f + 1];
    const double reach = norm(sub(average, centre));
    for (int64_t q = first; q < last; ++q) {
      const int64_t r = q + 1 < last ? q + 1 : first;
      const Vec p = load(x + 3 * mesh.face_vertices[q]);
      const Vec n = load(x + 3 * mesh.face_vertices[r]);
      double v = dot(cross(sub(p, average), sub(n, average)), sub(average, centre));
      if (signed_face < 0) v = -v;
      if (!(v > 1e-9 * reach * reach * reach)) return false;
    }
  }
  return true;
}

// The screen on geometry computed afresh.
bool cell_valid_float(const Mesh& mesh, const Topology& t, int64_t c) {
  return cell_screen(mesh, t, c, cell_centroid(mesh, t, c), [&](int64_t f, bool& unfolded) {
    unfolded = fan_unfolded(mesh, f);
    const int64_t first = mesh.face_offsets[f], last = mesh.face_offsets[f + 1];
    Vec average{0, 0, 0};
    for (int64_t q = first; q < last; ++q) average = add(average, load(mesh.points.data() + 3 * mesh.face_vertices[q]));
    return mul(average, 1.0 / static_cast<double>(last - first));
  });
}

// Decomposition validity of a cell with a closed genus-0 boundary: unfolded
// face fans (with a margin) and exactly oriented decomposition tetrahedra.
bool polyhedron_certified(const Mesh& mesh, const Topology& t, int64_t c) {
  const Vec centre = cell_centroid(mesh, t, c);
  const double cp[3] = {centre[0], centre[1], centre[2]};
  std::vector<std::pair<int32_t, int32_t>> directed;
  std::vector<int32_t> vertices;
  int64_t faces = 0;
  for (int64_t slot = t.cell_face_offsets[c]; slot < t.cell_face_offsets[c + 1]; ++slot) {
    ++faces;
    const int32_t signed_face = t.cell_faces[slot];
    const int64_t f = std::abs(signed_face) - 1;
    if (!fan_unfolded(mesh, f)) return false;
    const int64_t first = mesh.face_offsets[f], last = mesh.face_offsets[f + 1];
    Vec average{0, 0, 0};
    for (int64_t q = first; q < last; ++q) average = add(average, load(mesh.points.data() + 3 * mesh.face_vertices[q]));
    average = mul(average, 1.0 / static_cast<double>(last - first));
    const double ap[3] = {average[0], average[1], average[2]};
    for (int64_t q = first; q < last; ++q) {
      const int64_t r = q + 1 < last ? q + 1 : first;
      int32_t u = mesh.face_vertices[q], w = mesh.face_vertices[r];
      if (signed_face < 0) std::swap(u, w);
      directed.emplace_back(u, w);
      vertices.push_back(u);
      // Outward triangle (average, u, w): the centre lies on its inner side.
      if (orient3d(ap, mesh.points.data() + 3 * u, mesh.points.data() + 3 * w, cp) >= 0) return false;
    }
  }
  std::sort(directed.begin(), directed.end());
  if (std::adjacent_find(directed.begin(), directed.end()) != directed.end()) return false;
  for (const auto& e : directed) {
    if (!std::binary_search(directed.begin(), directed.end(), std::make_pair(e.second, e.first))) return false;
  }
  std::sort(vertices.begin(), vertices.end());
  vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());
  const int64_t edges = static_cast<int64_t>(directed.size()) / 2;
  return static_cast<int64_t>(vertices.size()) - edges + faces == 2;
}

// Layer points along the wall offset direction m of every column: with n_i
// the unit outward normals of the patches around the wall vertex (area-
// weighted over its wall faces, patches meeting below the feature angle
// merged), m solves m . n_i = 1 (least squares beyond three), so the k-th
// layer point lies at the cumulative thickness from every adjacent patch
// plane: m = n on a patch, (n1 + n2) / (1 + n1 . n2) on a feature curve.
void Builder::place_layers() {
  if (levels == 1) return;
  const int64_t columns = static_cast<int64_t>(stair_vertices.size());
  layer_direction.resize(columns);
  curve_column.resize(columns);
  std::vector<Vec> area(pieces.size());
  for (std::size_t p = 0; p < pieces.size(); ++p) {
    Vec centre, average;
    face_geometry(mesh, piece_wall_face[p], centre, area[p], average);
  }
  const double feature_cos = std::cos(in.feature_angle_degrees * (3.14159265358979323846 / 180.0));
  std::vector<std::pair<int32_t, Vec>> by_patch;
  std::vector<Vec> normals, sums;
  for (int64_t row = 0; row < columns; ++row) {
    by_patch.clear();
    for (const int32_t p : row_pieces[row]) {
      const int32_t patch = pieces[p].patch;
      auto it = std::find_if(by_patch.begin(), by_patch.end(), [&](const auto& e) { return e.first == patch; });
      if (it == by_patch.end()) {
        by_patch.emplace_back(patch, area[p]);
      } else {
        it->second = add(it->second, area[p]);
      }
    }
    normals.clear();
    sums.clear();
    for (const auto& [patch, sum] : by_patch) {
      const Vec n = mul(sum, 1.0 / norm(sum));
      std::size_t g = 0;
      while (g < normals.size() && dot(normals[g], n) <= feature_cos) ++g;
      if (g == normals.size()) {
        normals.push_back(n);
        sums.push_back(sum);
      } else {
        sums[g] = add(sums[g], sum);
        normals[g] = mul(sums[g], 1.0 / norm(sums[g]));
      }
    }
    // The two most opposed normals: a feature curve, and the fallback of a
    // rank-deficient corner.
    auto pair = [&]() {
      std::size_t i = 0, j = 1;
      for (std::size_t a = 0; a < normals.size(); ++a) {
        for (std::size_t c = a + 1; c < normals.size(); ++c) {
          if (dot(normals[a], normals[c]) < dot(normals[i], normals[j])) i = a, j = c;
        }
      }
      return mul(add(normals[i], normals[j]), 1.0 / (1.0 + dot(normals[i], normals[j])));
    };
    Vec m = normals[0];
    if (normals.size() == 2) {
      m = pair();
    } else if (normals.size() > 2) {
      double a[3][3] = {}, r[3] = {};
      for (const Vec& n : normals) {
        for (int i = 0; i < 3; ++i) {
          r[i] += n[i];
          for (int j = 0; j < 3; ++j) a[i][j] += n[i] * n[j];
        }
      }
      auto det3 = [](const double (&q)[3][3]) {
        return q[0][0] * (q[1][1] * q[2][2] - q[1][2] * q[2][1]) - q[0][1] * (q[1][0] * q[2][2] - q[1][2] * q[2][0]) +
               q[0][2] * (q[1][0] * q[2][1] - q[1][1] * q[2][0]);
      };
      const double det = det3(a);
      // Unit normals: det measures how far they are from one plane.
      if (det > 1e-6) {
        for (int i = 0; i < 3; ++i) {
          double q[3][3];
          for (int x = 0; x < 3; ++x) {
            for (int y = 0; y < 3; ++y) q[x][y] = y == i ? r[x] : a[x][y];
          }
          m[i] = det3(q) / det;
        }
      } else {
        m = pair();
      }
    }
    layer_direction[row] = m;
    curve_column[row] = normals.size() == 2;
    const Vec wall = point(surface_vertex[row]);
    double height = 0.0;
    for (int32_t k = 1; k < levels; ++k) {
      height += in.layer_thicknesses[k - 1];
      set_point(column[row * (levels + 1) + k], sub(wall, mul(m, height)));
    }
  }
}

void Builder::smooth() {
  Topology t = topology(mesh);
  const int64_t points = static_cast<int64_t>(mesh.points.size() / 3);
  const int64_t cells = static_cast<int64_t>(mesh.cell_kinds.size());
  const int64_t faces = static_cast<int64_t>(mesh.face_owner.size());
  // Active interior vertices: those of polyhedra, of finest core cells and
  // the staircase.
  std::vector<uint8_t> role(points, 0);  // 1 interior active, 2 surface
  for (int64_t c = 0; c < core_cells; ++c) {
    bool active = mesh.cell_kinds[c] == kPolyhedron;
    if (!active) {
      const int32_t* h = mesh.cell_vertices.data() + mesh.cell_vertex_offsets[c];
      const Vec p = point(h[0]), q = point(h[6]);
      active = std::fabs(q[0] - p[0]) < 1.5 * b;
    }
    if (!active) continue;
    for (int64_t slot = t.cell_face_offsets[c]; slot < t.cell_face_offsets[c + 1]; ++slot) {
      const int64_t f = std::abs(t.cell_faces[slot]) - 1;
      for (int64_t q = mesh.face_offsets[f]; q < mesh.face_offsets[f + 1]; ++q) role[mesh.face_vertices[q]] = 1;
    }
  }
  const int64_t columns = static_cast<int64_t>(stair_vertices.size());
  std::vector<int32_t> column_of(points, -1);
  for (int64_t row = 0; row < columns; ++row) {
    role[stair_vertices[row]] = 1;
    if (surface_kind[row] != kCorner) role[surface_vertex[row]] = 2;
    column_of[surface_vertex[row]] = static_cast<int32_t>(row);
  }
  // Layer points (role 3) are placed from the wall, never moved on their own.
  const int64_t layer_first = core_points + columns;
  for (int64_t v = layer_first; v < points; ++v) role[v] = 3;
  // Boundary faces incident to every surface vertex.
  std::vector<std::vector<int32_t>> wall_faces(columns);
  for (int64_t f = core_faces; f < faces; ++f) {
    if (mesh.face_neighbor[f] >= 0) continue;
    for (int64_t q = mesh.face_offsets[f]; q < mesh.face_offsets[f + 1]; ++q) {
      wall_faces[column_of[mesh.face_vertices[q]]].push_back(static_cast<int32_t>(f));
    }
  }
  // Distinct vertices of every cell.
  std::vector<int64_t> cell_vertex_offsets(cells + 1, 0);
  for (int64_t v = 0; v < points; ++v) {
    for (int64_t slot = t.vertex_cell_offsets[v]; slot < t.vertex_cell_offsets[v + 1]; ++slot) {
      ++cell_vertex_offsets[t.vertex_cells[slot] + 1];
    }
  }
  for (int64_t c = 0; c < cells; ++c) cell_vertex_offsets[c + 1] += cell_vertex_offsets[c];
  std::vector<int32_t> cell_vertex_list(cell_vertex_offsets[cells]);
  {
    std::vector<int64_t> fill(cell_vertex_offsets.begin(), cell_vertex_offsets.end() - 1);
    for (int64_t v = 0; v < points; ++v) {
      for (int64_t slot = t.vertex_cell_offsets[v]; slot < t.vertex_cell_offsets[v + 1]; ++slot) {
        cell_vertex_list[fill[t.vertex_cells[slot]]++] = static_cast<int32_t>(v);
      }
    }
  }
  // Only cells with a movable vertex change shape: their faces and centroids
  // are recomputed each pass, the rest once.
  std::vector<int32_t> active_cells, active_faces;
  {
    std::vector<uint8_t> face_seen(faces, 0);
    for (int64_t c = 0; c < cells; ++c) {
      bool active = false;
      for (int64_t slot = cell_vertex_offsets[c]; slot < cell_vertex_offsets[c + 1] && !active; ++slot) {
        active = role[cell_vertex_list[slot]] != 0;
      }
      if (!active) continue;
      active_cells.push_back(static_cast<int32_t>(c));
      for (int64_t slot = t.cell_face_offsets[c]; slot < t.cell_face_offsets[c + 1]; ++slot) {
        const int32_t f = std::abs(t.cell_faces[slot]) - 1;
        if (!face_seen[f]) {
          face_seen[f] = 1;
          active_faces.push_back(f);
        }
      }
    }
    std::sort(active_faces.begin(), active_faces.end());
  }
  // Face centres, area vectors, vertex averages and unfolded fans, and cell
  // centroids: refreshed for the active faces and cells by geometry() and by
  // settle(), which leaves them current for the next pass.
  std::vector<Vec> centroid(cells), face_centre(faces), face_area(faces), face_average(faces);
  std::vector<uint8_t> face_unfolded(faces, 0);
  auto face_update = [&](int64_t f) {
    face_geometry(mesh, f, face_centre[f], face_area[f], face_average[f]);
    face_unfolded[f] = fan_unfolded(mesh, f);
  };
  auto cell_geometry = [&](int64_t c) {
    const int64_t first = t.cell_face_offsets[c], last = t.cell_face_offsets[c + 1];
    Vec estimate{0, 0, 0};
    for (int64_t slot = first; slot < last; ++slot) estimate = add(estimate, face_centre[std::abs(t.cell_faces[slot]) - 1]);
    estimate = mul(estimate, 1.0 / static_cast<double>(last - first));
    double volume = 0.0;
    Vec moment{0, 0, 0};
    for (int64_t slot = first; slot < last; ++slot) {
      const int32_t signed_face = t.cell_faces[slot];
      const int64_t f = std::abs(signed_face) - 1;
      const double v = (signed_face > 0 ? 1.0 : -1.0) * dot(face_area[f], sub(face_centre[f], estimate)) / 3.0;
      volume += v;
      moment = add(moment, mul(add(mul(face_centre[f], 0.75), mul(estimate, 0.25)), v));
    }
    centroid[c] = volume > 0.0 ? mul(moment, 1.0 / volume) : estimate;
  };
  for (int64_t f = 0; f < faces; ++f) face_update(f);
  for (int64_t c = 0; c < cells; ++c) cell_geometry(c);
  auto geometry = [&]() {
    for (const int32_t f : active_faces) face_update(f);
    for (const int32_t c : active_cells) cell_geometry(c);
  };
  // The validity screen on the current face and centroid data.
  auto cell_valid_cached = [&](int64_t c) {
    return cell_screen(mesh, t, c, centroid[c], [&](int64_t f, bool& unfolded) {
      unfolded = face_unfolded[f];
      return face_average[f];
    });
  };
  std::vector<double> previous(mesh.points);
  std::vector<uint8_t> moved(points, 0);
  int64_t reverted = 0;
  // Cells valid so far: smoothing never makes one of them invalid (only the
  // active cells change shape).
  std::vector<uint8_t> valid(cells, 1);
  for (const int32_t c : active_cells) valid[c] = cell_valid_float(mesh, t, c);
  // Surface vertex move: onto its patch or curve by a local walk.
  auto place_surface = [&](int64_t row, const Vec& target) {
    const int32_t v = surface_vertex[row];
    const Hit hit = surface_kind[row] == kFeature ? walk_edge(s, target, surface_item[row])
                                                  : walk_triangle(s, target, surface_item[row]);
    surface_item[row] = hit.item;
    set_point(v, hit.point);
  };
  // Columns keep room for the layer stack; cells keep positive float
  // volumes: undo the moves around any that does not. A wall vertex and its
  // layer points move and revert together.
  std::vector<int32_t> column_row(points, -1);
  for (int64_t row = 0; row < columns; ++row) {
    for (int32_t k = 0; k < levels; ++k) column_row[column[row * (levels + 1) + k]] = static_cast<int32_t>(row);
  }
  std::vector<int32_t> stamp(cells, -1);
  int32_t epoch = 0;
  std::vector<int32_t> check, next;
  auto revert = [&](int32_t v) {
    const int32_t row = column_row[v];
    const int32_t count = row >= 0 ? levels : 1;
    for (int32_t k = 0; k < count; ++k) {
      const int32_t u = row >= 0 ? column[row * (levels + 1) + k] : v;
      if (!moved[u]) continue;
      moved[u] = 0;
      ++reverted;
      for (int a = 0; a < 3; ++a) mesh.points[3 * u + a] = previous[3 * u + a];
      for (int64_t r = t.vertex_cell_offsets[u]; r < t.vertex_cell_offsets[u + 1]; ++r) {
        const int32_t c = t.vertex_cells[r];
        if (stamp[c] != epoch) {
          stamp[c] = epoch;
          next.push_back(c);
        }
      }
    }
  };
  auto settle = [&]() {
    check.clear();
    next.clear();
    ++epoch;
    for (int64_t row = 0; row < columns; ++row) {
      const double length = norm(sub(point(stair_vertices[row]), point(surface_vertex[row])));
      if (length < layer_total + 0.25 * b) {
        revert(stair_vertices[row]);
        revert(surface_vertex[row]);
      }
    }
    for (int64_t v = 0; v < points; ++v) {
      if (!moved[v]) continue;
      for (int64_t r = t.vertex_cell_offsets[v]; r < t.vertex_cell_offsets[v + 1]; ++r) {
        const int32_t c = t.vertex_cells[r];
        if (stamp[c] != epoch) {
          stamp[c] = epoch;
          next.push_back(c);
        }
      }
    }
    // The first pass reads the geometry of the moved positions; a later pass
    // refreshes the cells it checks (those around reverted vertices). Within
    // a pass a cell around a vertex reverted earlier in it is read afresh.
    geometry();
    for (bool first = true; !next.empty(); first = false) {
      check.swap(next);
      next.clear();
      ++epoch;
      if (!first) {
        for (const int32_t c : check) {
          for (int64_t slot = t.cell_face_offsets[c]; slot < t.cell_face_offsets[c + 1]; ++slot) {
            face_update(std::abs(t.cell_faces[slot]) - 1);
          }
        }
        for (const int32_t c : check) cell_geometry(c);
      }
      for (const int32_t c : check) {
        if (stamp[c] != epoch ? cell_valid_cached(c) : cell_valid_float(mesh, t, c)) {
          valid[c] = 1;
          continue;
        }
        // A cell invalid before smoothing may stay so while moves repair it.
        if (!valid[c]) continue;
        for (int64_t slot = cell_vertex_offsets[c]; slot < cell_vertex_offsets[c + 1]; ++slot) {
          const int32_t v = cell_vertex_list[slot];
          if (moved[v]) revert(v);
        }
      }
    }
  };
  // Smoothing constants, tuned on the F8 sphere and checked on the quadric
  // (ledger cad.md F8). Twelve centroidal sweeps at relaxation 0.6: a full
  // step oscillates at the staircase.
  constexpr int kSweeps = 12;
  constexpr double kRelax = 0.6;
  // Descent: 20 steps of 0.1 b^2 times the gradient, each vertex capped by a
  // trust radius of at most 0.05 b (halved when undone, regrown by 1.25 when
  // kept). The wall stays where the centroidal sweeps put it: moving it in
  // the descent trades surface distance for interior angles.
  constexpr int kIterations = 20;
  constexpr double kStep = 0.1;
  constexpr double kTrust = 0.05;
  // Skewness weight against the non-orthogonality term and the quartic
  // emphasis that concentrates the descent on the few most skewed faces
  // (the transition faces between octree levels and the projection layer).
  constexpr double kSkewWeight = 0.1;
  constexpr double kSkewEmphasis = 20.0;
  // 1. Centroidal smoothing.
  for (int sweep = 0; sweep < kSweeps; ++sweep) {
    // The geometry is current: computed above, then by every settle().
    previous = mesh.points;
    std::fill(moved.begin(), moved.end(), 0);
    for (int64_t v = 0; v < points; ++v) {
      if (role[v] != 1) continue;
      Vec target{0, 0, 0};
      const int64_t first = t.vertex_cell_offsets[v], last = t.vertex_cell_offsets[v + 1];
      for (int64_t slot = first; slot < last; ++slot) target = add(target, centroid[t.vertex_cells[slot]]);
      target = mul(target, 1.0 / static_cast<double>(last - first));
      const Vec p = point(v);
      set_point(v, add(p, mul(sub(target, p), kRelax)));
      moved[v] = 1;
    }
    for (int64_t row = 0; row < columns; ++row) {
      const int32_t v = surface_vertex[row];
      if (role[v] != 2) continue;
      const Vec p = load(previous.data() + 3 * v);
      Vec target{0, 0, 0};
      if (surface_kind[row] == kFeature) {
        if (chain_neighbors[row][0] < 0) continue;
        for (const int32_t other : chain_neighbors[row]) target = add(target, load(previous.data() + 3 * surface_vertex[other]));
        target = mul(target, 0.5);
      } else {
        double weight = 0.0;
        for (const int32_t f : wall_faces[row]) {
          const double area = norm(face_area[f]);
          target = add(target, mul(face_centre[f], area));
          weight += area;
        }
        target = mul(target, 1.0 / weight);
      }
      place_surface(row, add(p, mul(sub(target, p), kRelax)));
      moved[v] = 1;
    }
    // The layer stack follows the wall: its points move with their columns'
    // offset directions, and a revert restores them with the wall.
    place_layers();
    for (int64_t v = layer_first; v < points; ++v) moved[v] = 1;
    settle();
  }
  // 2. Descent on E = sum over interior faces of (1 - cos) of the angle
  // between the centroid join and the area vector, plus the skewness term;
  // cell centroids move with the mean of their vertices.
  std::vector<Vec> gradient(points), cell_gradient(cells);
  std::vector<double> trust(points, kTrust * b);
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    // The geometry is current (settle()).
    std::fill(gradient.begin(), gradient.end(), Vec{0, 0, 0});
    std::fill(cell_gradient.begin(), cell_gradient.end(), Vec{0, 0, 0});
    for (const int64_t f : active_faces) {
      const int32_t o = mesh.face_owner[f], n = mesh.face_neighbor[f];
      if (n < 0) continue;
      const Vec d = sub(centroid[n], centroid[o]);
      const Vec& area = face_area[f];
      const double ld = norm(d), ls = norm(area);
      if (!(ld > 0.0) || !(ls > 0.0)) continue;
      const Vec dh = mul(d, 1.0 / ld), sh = mul(area, 1.0 / ls);
      const double c = dot(dh, sh);
      const double weight = -1.0;  // dE/dc
      const Vec gd = mul(sub(sh, mul(dh, c)), weight / ld);
      const Vec gs = mul(sub(dh, mul(sh, c)), weight / ls);
      cell_gradient[n] = add(cell_gradient[n], gd);
      cell_gradient[o] = sub(cell_gradient[o], gd);
      const int64_t first = mesh.face_offsets[f], last = mesh.face_offsets[f + 1];
      for (int64_t q = first; q < last; ++q) {
        const int64_t before = q > first ? q - 1 : last - 1, after = q + 1 < last ? q + 1 : first;
        const Vec diff = sub(point(mesh.face_vertices[before]), point(mesh.face_vertices[after]));
        gradient[mesh.face_vertices[q]] = add(gradient[mesh.face_vertices[q]], mul(cross(gs, diff), 0.5));
      }
      // Skewness s = |sv| / fd, sv = Cf - (Co + alpha d) the offset of the
      // face centre from where the centroid join meets the face plane
      // (OpenFOAM's measure); E += w (s^2 + mu s^4) with the face normal
      // and fd frozen, the face centre moving with its vertex mean.
      {
        const Vec cpf = sub(face_centre[f], centroid[o]);
        const double nd = dot(sh, d);
        if (!(std::fabs(nd) > 0.0)) continue;
        const double alpha = dot(sh, cpf) / nd;
        const Vec sv = sub(cpf, mul(d, alpha));
        const double length = norm(sv);
        double fd = 0.2 * ld;
        if (length > 0.0) {
          const Vec unit = mul(sv, 1.0 / length);
          for (int64_t q = first; q < last; ++q) {
            fd = std::max(fd, std::fabs(dot(unit, sub(point(mesh.face_vertices[q]), face_centre[f]))));
          }
        }
        const double skew = length / fd;
        const double scale = kSkewWeight * (1.0 + 2.0 * kSkewEmphasis * skew * skew) * 2.0 / (fd * fd);
        const Vec w = mul(sub(sv, mul(sh, dot(sv, d) / nd)), scale);
        const Vec face_share = mul(w, 1.0 / static_cast<double>(last - first));
        for (int64_t q = first; q < last; ++q) {
          gradient[mesh.face_vertices[q]] = add(gradient[mesh.face_vertices[q]], face_share);
        }
        cell_gradient[n] = sub(cell_gradient[n], mul(w, alpha));
        cell_gradient[o] = sub(cell_gradient[o], mul(w, 1.0 - alpha));
      }
    }
    for (const int64_t c : active_cells) {
      const int64_t first = cell_vertex_offsets[c], last = cell_vertex_offsets[c + 1];
      const Vec share = mul(cell_gradient[c], 1.0 / static_cast<double>(last - first));
      for (int64_t slot = first; slot < last; ++slot) gradient[cell_vertex_list[slot]] = add(gradient[cell_vertex_list[slot]], share);
    }
    previous = mesh.points;
    std::fill(moved.begin(), moved.end(), 0);
    for (int64_t v = 0; v < points; ++v) {
      if (role[v] != 1) continue;
      Vec delta = mul(gradient[v], -kStep * b * b);
      const double length = norm(delta);
      if (length > trust[v]) delta = mul(delta, trust[v] / length);
      set_point(v, add(point(v), delta));
      moved[v] = 1;
    }
    std::vector<uint8_t> attempted(moved);
    settle();
    for (int64_t v = 0; v < points; ++v) {
      if (!attempted[v]) continue;
      trust[v] = moved[v] ? std::min(kTrust * b, 1.25 * trust[v]) : 0.5 * trust[v];
    }
  }
  // 3. Local repair: a pattern search per vertex of the faces still above
  // the goals, on the largest normalised badness max(skew / s0, angle / a0)
  // over the faces of its cells (OpenFOAM's measures, recomputed exactly;
  // a boundary face measures its skewness against the owner centroid's
  // projection along the face normal, as checkMesh does).
  // Goals: checkMesh skewness 0.46 and non-orthogonality 17.5 degrees, the
  // levels the F8 sphere reaches below cfMesh's (0.486, 18.7); three rounds.
  constexpr double kSkewGoal = 0.46;
  constexpr double kAngleGoal = 17.5;
  constexpr int kRepairRounds = 3;
  // Badness of face f; a non-orthogonality term at most `floor` (its cosine
  // at least floor_cosine = cos(floor a0)) is not evaluated, as callers only
  // compare the result with values of at least floor.
  auto face_badness = [&](int64_t f, const Vec& centre, const Vec& area, const Vec& co, const Vec* cn,
                          double floor_cosine) {
    const double ls = norm(area);
    if (!(ls > 0.0)) return std::numeric_limits<double>::infinity();
    const Vec cpf = sub(centre, co);
    const Vec d = cn != nullptr ? sub(*cn, co) : mul(area, dot(area, cpf) / (ls * ls));
    const double ld = norm(d);
    if (!(ld > 0.0)) return std::numeric_limits<double>::infinity();
    const double cosine = std::clamp(dot(d, area) / (ld * ls), -1.0, 1.0);
    const double angle = cn != nullptr && cosine < floor_cosine
                             ? std::acos(cosine) * (180.0 / 3.14159265358979323846)
                             : 0.0;
    const double nd = dot(area, d);
    if (!(std::fabs(nd) > 0.0)) return std::numeric_limits<double>::infinity();
    const Vec sv = sub(cpf, mul(d, dot(area, cpf) / nd));
    const double length = norm(sv);
    double fd = 0.2 * ld;
    if (length > 0.0) {
      const Vec unit = mul(sv, 1.0 / length);
      for (int64_t q = mesh.face_offsets[f]; q < mesh.face_offsets[f + 1]; ++q) {
        fd = std::max(fd, std::fabs(dot(unit, sub(point(mesh.face_vertices[q]), centre))));
      }
    }
    return std::max(length / fd / kSkewGoal, angle / kAngleGoal);
  };
  int64_t repaired = 0;
  // Per vertex (with the layer points it carries), the faces of its cells
  // with their geometry cached: a trial move recomputes only the faces
  // through the moving points and the centroids of their cells; validity is
  // checked only for an improving move.
  std::vector<int32_t> face_slot(faces, -1);
  std::vector<int32_t> local_cells, local_faces;
  std::vector<Vec> local_centroid, local_centre, local_area, local_average;
  // Per local face: moves with the movers; fan unfolded (2: not yet known).
  std::vector<uint8_t> local_moving, local_unfolded;
  std::vector<int32_t> movers;
  std::vector<Vec> starts;
  for (int round = 0; round < kRepairRounds; ++round) {
    geometry();
    std::vector<std::pair<double, int32_t>> bad;
    const double goal_cosine = std::cos(kAngleGoal * (3.14159265358979323846 / 180.0));
    for (const int64_t f : active_faces) {
      const int32_t o = mesh.face_owner[f], n = mesh.face_neighbor[f];
      const double value =
          face_badness(f, face_centre[f], face_area[f], centroid[o], n >= 0 ? &centroid[n] : nullptr, goal_cosine);
      if (value > 1.0) bad.emplace_back(-value, static_cast<int32_t>(f));
    }
    if (bad.empty()) break;
    std::sort(bad.begin(), bad.end());
    std::vector<uint8_t> done(points, 0);
    for (const auto& [value, f] : bad) {
      for (const int32_t c : {mesh.face_owner[f], mesh.face_neighbor[f]}) {
        if (c < 0) continue;
        for (int64_t slot = cell_vertex_offsets[c]; slot < cell_vertex_offsets[c + 1]; ++slot) {
          const int32_t v = cell_vertex_list[slot];
          // A layer point of a curve column moves along its column's offset
          // direction with the layer points above it, its layer never
          // thinner than requested: at a convex curve the parallel offset
          // leaves the face between the two layer cells across the curve
          // short against the distance of their centroids from the curve.
          const int32_t layer_row = role[v] == 3 ? column_row[v] : -1;
          const bool layer = layer_row >= 0 && curve_column[layer_row];
          if ((role[v] != 1 && !layer) || done[v]) continue;
          done[v] = 1;
          movers.assign(1, v);
          int32_t level = 0;
          if (layer) {
            const int32_t* stack = column.data() + layer_row * (levels + 1);
            while (stack[level] != v) ++level;
            for (int32_t k = level + 1; k < levels; ++k) movers.push_back(stack[k]);
          }
          local_cells.clear();
          ++epoch;
          for (const int32_t u : movers) {
            for (int64_t r = t.vertex_cell_offsets[u]; r < t.vertex_cell_offsets[u + 1]; ++r) {
              const int32_t k = t.vertex_cells[r];
              if (stamp[k] == epoch) continue;
              stamp[k] = epoch;
              local_cells.push_back(k);
            }
          }
          local_faces.clear();
          for (const int32_t k : local_cells) {
            for (int64_t s2 = t.cell_face_offsets[k]; s2 < t.cell_face_offsets[k + 1]; ++s2) {
              const int32_t g = std::abs(t.cell_faces[s2]) - 1;
              if (face_slot[g] < 0) {
                face_slot[g] = static_cast<int32_t>(local_faces.size());
                local_faces.push_back(g);
              }
            }
          }
          const std::size_t nf = local_faces.size(), nc = local_cells.size();
          local_centre.resize(nf);
          local_area.resize(nf);
          local_average.resize(nf);
          local_moving.assign(nf, 0);
          local_unfolded.assign(nf, 2);
          local_centroid.resize(nc);
          for (std::size_t q = 0; q < nf; ++q) {
            const int32_t g = local_faces[q];
            for (int64_t e = mesh.face_offsets[g]; e < mesh.face_offsets[g + 1] && !local_moving[q]; ++e) {
              local_moving[q] = std::find(movers.begin(), movers.end(), mesh.face_vertices[e]) != movers.end();
            }
            face_geometry(mesh, g, local_centre[q], local_area[q], local_average[q]);
          }
          auto centroid_of = [&](int32_t k) -> const Vec& {
            for (std::size_t q = 0; q < nc; ++q) {
              if (local_cells[q] == k) return local_centroid[q];
            }
            return centroid[k];
          };
          // The largest badness of the local faces, from the hardest face of
          // the last full evaluation on; it stops once it reaches bound (a
          // trial no better than the current position).
          std::size_t hardest = 0;
          auto evaluate = [&](double bound) {
            for (std::size_t q = 0; q < nf; ++q) {
              if (!local_moving[q]) continue;
              face_geometry(mesh, local_faces[q], local_centre[q], local_area[q], local_average[q]);
            }
            for (std::size_t q = 0; q < nc; ++q) {
              const int32_t k = local_cells[q];
              const int64_t first = t.cell_face_offsets[k], last = t.cell_face_offsets[k + 1];
              Vec estimate{0, 0, 0};
              for (int64_t s2 = first; s2 < last; ++s2) {
                estimate = add(estimate, local_centre[face_slot[std::abs(t.cell_faces[s2]) - 1]]);
              }
              estimate = mul(estimate, 1.0 / static_cast<double>(last - first));
              double volume = 0.0;
              Vec moment{0, 0, 0};
              for (int64_t s2 = first; s2 < last; ++s2) {
                const int32_t signed_face = t.cell_faces[s2];
                const int32_t slot_q = face_slot[std::abs(signed_face) - 1];
                const double w = (signed_face > 0 ? 1.0 : -1.0) *
                                 dot(local_area[slot_q], sub(local_centre[slot_q], estimate)) / 3.0;
                volume += w;
                moment = add(moment, mul(add(mul(local_centre[slot_q], 0.75), mul(estimate, 0.25)), w));
              }
              local_centroid[q] = volume > 0.0 ? mul(moment, 1.0 / volume) : estimate;
            }
            // The visiting order swaps the first face with the hardest face
            // of the last full evaluation (fixed for this evaluation).
            const std::size_t start = hardest;
            double worst = 0.0, worst_cosine = 1.0;
            for (std::size_t r = 0; r < nf && worst < bound; ++r) {
              const std::size_t q = r == 0 ? start : r == start ? 0 : r;
              const int32_t g = local_faces[q];
              const int32_t n = mesh.face_neighbor[g];
              const double value = face_badness(g, local_centre[q], local_area[q], centroid_of(mesh.face_owner[g]),
                                                n >= 0 ? &centroid_of(n) : nullptr, worst_cosine);
              if (value > worst) {
                worst = value;
                worst_cosine = std::cos(std::min(worst * kAngleGoal, 180.0) * (3.14159265358979323846 / 180.0));
                hardest = q;
              }
            }
            return worst;
          };
          double current = evaluate(std::numeric_limits<double>::infinity());
          const int32_t stair = v < core_points ? stair_vertex[v] : -1;
          Vec axis{0, 0, 0}, wall{0, 0, 0};
          double below = 0.0;  // height of the layer point below v
          if (layer) {
            const Vec& m = layer_direction[layer_row];
            axis = mul(m, 1.0 / norm(m));
            wall = point(surface_vertex[layer_row]);
            below = dot(sub(wall, point(column[layer_row * (levels + 1) + level - 1])), m) / dot(m, m);
          }
          for (const double scale : {0.08, 0.03, 0.01}) {
            for (bool improved = true; improved;) {
              improved = false;
              for (int direction = 0; direction < (layer ? 2 : 6); ++direction) {
                Vec delta{0, 0, 0};
                if (layer) {
                  delta = mul(axis, (direction & 1 ? -1.0 : 1.0) * scale * b);
                } else {
                  delta[direction / 2] = (direction & 1 ? -1.0 : 1.0) * scale * b;
                }
                starts.clear();
                for (const int32_t u : movers) {
                  starts.push_back(point(u));
                  set_point(u, add(point(u), delta));
                }
                const double value = evaluate(current);
                bool accept = value < current;
                // The layer keeps its thickness.
                if (accept && layer) {
                  const Vec& m = layer_direction[layer_row];
                  accept = dot(sub(wall, point(v)), m) / dot(m, m) - below >= in.layer_thicknesses[level - 1];
                }
                // The staircase vertex keeps room for the layer stack.
                if (accept && stair >= 0) {
                  accept = norm(sub(point(v), point(surface_vertex[stair]))) >= layer_total + 0.25 * b;
                }
                // The validity screen on the geometry evaluate() left: the
                // trial's faces and centroids.
                for (std::size_t q = 0; q < nc && accept; ++q) {
                  if (!valid[local_cells[q]]) continue;
                  accept = cell_screen(mesh, t, local_cells[q], local_centroid[q], [&](int64_t g, bool& unfolded) {
                    const int32_t slot_q = face_slot[g];
                    if (local_moving[slot_q]) {
                      unfolded = fan_unfolded(mesh, g);
                    } else {
                      if (local_unfolded[slot_q] == 2) local_unfolded[slot_q] = fan_unfolded(mesh, g);
                      unfolded = local_unfolded[slot_q] != 0;
                    }
                    return local_average[slot_q];
                  });
                }
                if (accept) {
                  current = value;
                  improved = true;
                  ++repaired;
                } else {
                  for (std::size_t q = 0; q < movers.size(); ++q) set_point(movers[q], starts[q]);
                }
              }
            }
          }
          evaluate(std::numeric_limits<double>::infinity());
          for (std::size_t q = 0; q < nc; ++q) centroid[local_cells[q]] = local_centroid[q];
          for (const int32_t g : local_faces) face_slot[g] = -1;
        }
      }
    }
  }
  stats[26] = static_cast<double>(repaired);
  stats[21] = static_cast<double>(reverted);
}

int32_t Builder::certify() {
  const Topology t = topology(mesh);
  const int64_t cells = static_cast<int64_t>(mesh.cell_kinds.size());
  int64_t failed = 0;
  for (int64_t c = 0; c < cells; ++c) failed += !polyhedron_certified(mesh, t, c);
  stats[10] = static_cast<double>(cells - failed);
  stats[23] = static_cast<double>(failed);
  stats[22] = failed == 0 ? 1.0 : 0.0;
  return failed == 0 ? PHX_MC_OK : PHX_MC_UNRESOLVED;
}

}  // namespace

int32_t build(const Input& in, Mesh& mesh) {
  const auto start = Clock::now();
  Surface surface;
  Builder builder(in, surface, mesh);
  builder.b = in.boundary_size;
  int levels = 0;
  while (std::ldexp(in.boundary_size, levels) < in.core_size) ++levels;
  builder.tree.levels = levels;
  builder.tree.b = in.boundary_size;
  for (int32_t k = 0; k < in.layer_count; ++k) builder.layer_total += in.layer_thicknesses[k];
  // Core clearance: half a boundary cell beyond the layer stack, so every
  // projection column starts at least b / 2 from the wall.
  builder.delta = 0.5 * in.boundary_size + builder.layer_total;
  auto& stats = builder.stats;
  int32_t status = build_surface(in, in.feature_angle_degrees, 0.5 * in.boundary_size, surface, stats);
  if (status != PHX_MC_OK) return status;
  stats[1] = seconds(start);
  auto stage = Clock::now();
  status = builder.octree_and_core();
  if (status != PHX_MC_OK) return status;
  stats[2] = seconds(stage);
  stage = Clock::now();
  status = builder.assemble_core();
  if (status == PHX_MC_OK) status = builder.staircase();
  if (status != PHX_MC_OK) return status;
  stats[3] = seconds(stage);
  stage = Clock::now();
  builder.partition();
  builder.build_pieces();
  builder.split_core_faces();
  status = builder.project();
  if (status != PHX_MC_OK) return status;
  builder.levels = in.layer_count + 1;
  builder.emit_layers();
  builder.place_layers();
  stats[4] = seconds(stage);
  stage = Clock::now();
  builder.smooth();
  stats[5] = seconds(stage);
  // Surface vertex ancestry and kinds: the triangle under a patch vertex and
  // a triangle on the feature edge under a curve vertex (walks from the item
  // of the last placement attempt, which a revert may have left behind), the
  // nearest triangle of a corner; the distance to that point.
  const int64_t points = static_cast<int64_t>(mesh.points.size() / 3);
  if (points >= std::numeric_limits<int32_t>::max()) return PHX_MC_INDEX_RANGE_ERROR;
  mesh.vertex_triangles.assign(points, -1);
  mesh.vertex_kinds.assign(points, kInterior);
  for (int64_t v = builder.core_points + static_cast<int64_t>(builder.stair_vertices.size()); v < points; ++v) {
    mesh.vertex_kinds[v] = kLayerVertex;
  }
  double vertex_distance = 0.0;
  for (std::size_t row = 0; row < builder.stair_vertices.size(); ++row) {
    const int32_t v = builder.surface_vertex[row];
    const int32_t item = builder.surface_item[row];
    const Vec p = builder.point(v);
    mesh.vertex_kinds[v] = builder.surface_kind[row];
    Hit hit;
    if (builder.surface_kind[row] == kPatch) {
      hit = walk_triangle(surface, p, item);
    } else if (builder.surface_kind[row] == kFeature) {
      hit = walk_edge(surface, p, item);
      hit.item = surface.edge_triangles[hit.item];
    } else {
      hit = nearest_triangle(surface, p, -1);
    }
    mesh.vertex_triangles[v] = hit.item;
    vertex_distance = std::max(vertex_distance, norm(sub(hit.point, p)));
  }
  // Boundary faces: the facet of their patch, read from a patch vertex's
  // triangle (a face without one: the nearest triangle of its centre). The
  // sampled distance of centres and edge midpoints is to the face's patch
  // (a walk from that triangle), an upper bound on the distance to the
  // surface.
  const int64_t faces = static_cast<int64_t>(mesh.face_owner.size());
  mesh.face_facets.assign(faces, -1);
  double sampled = 0.0;
  for (int64_t f = 0; f < faces; ++f) {
    if (mesh.face_neighbor[f] >= 0) continue;
    Vec centre, area, average;
    face_geometry(mesh, f, centre, area, average);
    int32_t seed = -1;
    for (int64_t q = mesh.face_offsets[f]; q < mesh.face_offsets[f + 1] && seed < 0; ++q) {
      const int32_t v = mesh.face_vertices[q];
      if (mesh.vertex_kinds[v] == kPatch) seed = mesh.vertex_triangles[v];
    }
    Hit hit = seed >= 0 ? walk_triangle(surface, average, seed) : nearest_triangle(surface, average, -1);
    mesh.face_facets[f] = surface.patch_facet[surface.patch[hit.item]];
    sampled = std::max(sampled, std::sqrt(hit.d2));
    for (int64_t q = mesh.face_offsets[f]; q < mesh.face_offsets[f + 1]; ++q) {
      const int64_t r = q + 1 < mesh.face_offsets[f + 1] ? q + 1 : mesh.face_offsets[f];
      const Vec mid = mul(add(builder.point(mesh.face_vertices[q]), builder.point(mesh.face_vertices[r])), 0.5);
      hit = walk_triangle(surface, mid, hit.item);
      sampled = std::max(sampled, std::sqrt(hit.d2));
    }
  }
  stats[15] = vertex_distance;
  stats[16] = sampled;
  stage = Clock::now();
  status = PHX_MC_OK;
  if (in.prove) {
    status = builder.certify();
  } else {
    stats[22] = -1.0;
  }
  stats[6] = seconds(stage);
  const int64_t cells = static_cast<int64_t>(mesh.cell_kinds.size());
  int64_t hexahedra = 0, layer_cells = 0;
  for (int64_t c = 0; c < cells; ++c) {
    hexahedra += mesh.cell_kinds[c] == kHexahedron;
    layer_cells += mesh.cell_roles[c] != kCore;
  }
  stats[7] = static_cast<double>(cells);
  stats[8] = static_cast<double>(hexahedra);
  stats[9] = static_cast<double>(cells - hexahedra);
  stats[11] = static_cast<double>(layer_cells);
  stats[25] = static_cast<double>(levels);
  stats[0] = seconds(start);
  std::copy(stats.begin(), stats.end(), mesh.stats);
  return status;
}

}  // namespace phx::mc::hex_dominant
