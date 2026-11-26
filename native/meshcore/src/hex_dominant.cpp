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
//     their center's nearest triangle; the partition is cleaned (flips of faces
//     with three foreign neighbors, of faces whose curve vertices are not one
//     run along their loop and of faces at non-manifold chain vertices); chain
//     vertices snap to the feature curve between their two patches, corner
//     vertices to the nearest feature corner, the others onto their patch.
//  7. Smoothing (Jacobi, relaxed): fine-band and transition vertices move to
//     the mean of their cells' centroids, surface vertices to the mean of their
//     boundary faces' centers (curve vertices to the mean of their two chain
//     neighbors) and back onto their patch or curve by a local walk; a move is
//     undone when an incident cell loses a positive float corner or
//     decomposition volume.
//  8. Layers: each projection column is split at the cumulative thicknesses
//     from the wall, giving layer_count thin hexahedra and the remainder; the
//     final repair search moves a feature-curve column's layer points (with
//     those above them) deeper along its offset direction, never thinner.
//     At a convex feature curve the layers do not wrap the curve: each
//     side's columns reattach to layer points slid along the other side's
//     patch, and a corner hexahedron column per curve segment fills the gap
//     (L x L for L layers), its two wall strips one per patch. With layers,
//     staircase steps split along the corner's own diagonal (the curve
//     keeps every staircase vertex); a curve segment whose chord deviates
//     from the curve more than any patch wall edge from its patch gains a
//     midpoint on the curve; patch columns next to the slid points smooth
//     to their wall-edge neighbours weighted by squared length, and the
//     repair slides them on their patch without raising the local
//     skewness or their wall faces' deviation from the surface. A skewed
//     face between layer cells of a smooth unsplit patch column tilts that
//     column's layer points rigidly about its wall vertex, toward and across
//     the face's skew direction, without raising the local skewness and
//     keeping every layer cell at least the requested fraction of its
//     thickness above the plane of its wall-side corners.
// Certification (prove): every cell, published as a polyhedron with OpenFOAM
// face semantics (a non-planar face is the fan from its vertex average), by
// the exact orientation of its decomposition tetrahedra (center, face vertex
// average, edge) about both the OpenFOAM centroid (OpenFOAM's decomposition)
// and the vertex average of its distinct vertices (the published local map),
// and a closed genus-0 boundary; smoothing screens both stars.
#include "hex_dominant.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include "floating_point_mode.hpp"
#include "parallel.hpp"
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
bool triangle_box_overlap(const Vec& center, double half, const Vec& a0, const Vec& b0,
                          const Vec& c0) {
  const Vec v0 = sub(a0, center), v1 = sub(b0, center), v2 = sub(c0, center);
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
    const int64_t base = 3 * static_cast<int64_t>(best.item);
    for (int e = 0; e < 3; ++e) {
      // Outgoing half-edges of corner e: the next one is the opposite of the
      // half-edge entering the corner in the current triangle. The last triangle
      // of fan e is the edge neighbor across (e, e + 1), which fan (e + 1) % 3
      // visits first (already checked for e == 2); skipping it preserves the
      // exact first-appearance order while avoiding three duplicate tests.
      const int64_t first = base + e;
      const int64_t prev_first = base + (e == 0 ? 2 : e - 1);
      for (int64_t h = s.opposite[prev_first]; h != first;) {
        const int32_t u = static_cast<int32_t>(h / 3);
        const int32_t rem = static_cast<int32_t>(h - 3 * static_cast<int64_t>(u));
        h = s.opposite[3 * static_cast<int64_t>(u) + (rem == 0 ? 2 : rem - 1)];
        if (h == first) break;
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
        const auto c = unpack(pairs[n].first);
        line_jk_.push_back({c[1], c[2]});
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

  int32_t resolve_all(uint32_t parallelism) {
    const std::size_t count = line_jk_.size();
    if (parallelism == 1 || count <= 128) {
      for (std::size_t line = 0; line < count; ++line) {
        if (!resolved_[line]) resolve(static_cast<int32_t>(line), line_jk_[line][0], line_jk_[line][1]);
      }
      return PHX_MC_OK;
    }
    return parallel_for(count, ParallelRequest{128, parallelism},
                        [&](std::size_t begin, std::size_t end, ChunkContext&) -> int32_t {
      const GradualNearestScope mode;
      if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
      for (std::size_t line = begin; line < end; ++line) {
        if (!resolved_[line]) resolve(static_cast<int32_t>(line), line_jk_[line][0], line_jk_[line][1]);
      }
      return PHX_MC_OK;
    });
  }

  bool inside_resolved(int64_t i, int64_t j, int64_t k) const {
    const int32_t line = lines_.find(pack(0, j, k));
    if (line < 0) return false;
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

  bool inside(int64_t i, int64_t j, int64_t k) {
    const int32_t line = lines_.find(pack(0, j, k));
    if (line < 0) return false;
    if (!resolved_[line]) resolve(line, j, k);
    return inside_resolved(i, j, k);
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
  std::vector<std::array<int64_t, 2>> line_jk_;
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

std::vector<uint8_t> dilate_grid(const std::vector<uint8_t>& src, int64_t sx, int64_t sy, int64_t sz,
                                 int64_t radius) {
  const std::size_t total = static_cast<std::size_t>(sx * sy * sz);
  std::vector<uint8_t> tmp(total, 0), dst(total, 0);
  for (int64_t i = 0; i < sx; ++i) {
    for (int64_t j = 0; j < sy; ++j) {
      const std::size_t base = static_cast<std::size_t>((i * sy + j) * sz);
      int64_t active = 0;
      for (int64_t k = 0; k < std::min(sz, radius); ++k) active += src[base + k];
      for (int64_t k = 0; k < sz; ++k) {
        if (k + radius < sz) active += src[base + k + radius];
        if (k - radius - 1 >= 0) active -= src[base + k - radius - 1];
        tmp[base + k] = active > 0 ? 1 : 0;
      }
    }
  }
  for (int64_t i = 0; i < sx; ++i) {
    const std::size_t ibase = static_cast<std::size_t>(i * sy * sz);
    for (int64_t k = 0; k < sz; ++k) {
      int64_t active = 0;
      for (int64_t j = 0; j < std::min(sy, radius); ++j) active += tmp[ibase + j * sz + k];
      for (int64_t j = 0; j < sy; ++j) {
        if (j + radius < sy) active += tmp[ibase + (j + radius) * sz + k];
        if (j - radius - 1 >= 0) active -= tmp[ibase + (j - radius - 1) * sz + k];
        dst[ibase + j * sz + k] = active > 0 ? 1 : 0;
      }
    }
  }
  std::fill(tmp.begin(), tmp.end(), 0);
  const std::size_t yz = static_cast<std::size_t>(sy * sz);
  for (std::size_t jk = 0; jk < yz; ++jk) {
    int64_t active = 0;
    for (int64_t i = 0; i < std::min(sx, radius); ++i) active += dst[i * yz + jk];
    for (int64_t i = 0; i < sx; ++i) {
      if (i + radius < sx) active += dst[(i + radius) * yz + jk];
      if (i - radius - 1 >= 0) active -= dst[(i - radius - 1) * yz + jk];
      tmp[i * yz + jk] = active > 0 ? 1 : 0;
    }
  }
  return tmp;
}

std::vector<uint8_t> coarsen_grid(const std::vector<uint8_t>& src, int64_t sx, int64_t sy, int64_t sz,
                                  int shift) {
  const int64_t dx = sx >> shift, dy = sy >> shift, dz = sz >> shift;
  std::vector<uint8_t> dst(static_cast<std::size_t>(dx * dy * dz), 0);
  for (int64_t i = 0; i < sx; ++i) {
    const int64_t ci = (i >> shift) * dy;
    for (int64_t j = 0; j < sy; ++j) {
      const std::size_t sbase = static_cast<std::size_t>((i * sy + j) * sz);
      const std::size_t dbase = static_cast<std::size_t>((ci + (j >> shift)) * dz);
      for (int64_t k = 0; k < sz; ++k) {
        if (src[sbase + k]) dst[dbase + (k >> shift)] = 1;
      }
    }
  }
  return dst;
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
  // Convex feature-curve columns split per side (cfMesh's extrusion edge):
  // the grid g(i, j) = grid[i (L + 1) + j] of L = levels - 1 layers, g(0, 0)
  // the curve vertex, g(i, 0) on the first patch of the pair and g(0, j) on
  // the second at the cumulative thicknesses from the curve, g(i, j) =
  // g(i, 0) + g(0, j) - g(0, 0). The first patch's pieces take the column
  // g(L, k), the second's g(k, L); a corner hex column per curve segment
  // fills the grid. item: the triangle under g(i, 0) (i - 1) and g(0, j)
  // (L + j - 1).
  struct Split {
    int32_t row;
    std::array<int32_t, 2> pair;
    std::vector<int32_t> grid;
    std::vector<int32_t> item;
  };
  std::vector<Split> splits;
  std::vector<int32_t> split_of;  // per staircase vertex row, -1
  // Curve segments between split columns whose chord deviates from the
  // feature curve more than any wall edge between patch vertices deviates
  // from its patch: split at the curve point nearest the chord midpoint, the
  // midpoint grid g(i, j) built like a split column's from the two end
  // grids, the corner column split in two and every face holding an end
  // grid pair gaining the midpoint. item: the triangles as Split::item.
  struct Midpoint {
    int32_t first_row, second_row;
    std::vector<int32_t> grid;
    int32_t curve_item;
    std::vector<int32_t> item;
  };
  std::vector<Midpoint> mids;
  KeyMap mid_rows{16};    // staircase row pair -> midpoint
  KeyMap mid_points{16};  // end grid point pair -> midpoint grid point
  static uint64_t pair_key(int32_t u, int32_t w) {
    return (static_cast<uint64_t>(std::min(u, w)) << 32) | static_cast<uint64_t>(std::max(u, w));
  }

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
  int32_t partition();
  int32_t project();
  void classify_splits();
  int32_t emit_layers();
  int32_t place_layers();
  void place_split(int64_t row, const std::vector<std::pair<int32_t, Vec>>& by_patch);
  void place_mids();
  // The points carried by a column's wall vertex (its layer stack, or a
  // split column's grid).
  std::pair<const int32_t*, int32_t> carried(int64_t row) const {
    if (split_of[row] >= 0) {
      const auto& grid = splits[split_of[row]].grid;
      return {grid.data(), static_cast<int32_t>(grid.size())};
    }
    return {column.data() + row * (levels + 1), levels};
  }
  // Least staircase-to-wall distance of a column: room for its layer stack
  // and a quarter boundary cell.
  double room(int64_t row) const {
    if (split_of[row] < 0) return layer_total + 0.25 * b;
    const auto& grid = splits[split_of[row]].grid;
    return norm(sub(point(grid.back()), point(grid[0]))) + 0.25 * b;
  }
  // A boundary-layer cell's thickness as the compliance report measures it:
  // the mean height of its upper corners (the first half of its vertex
  // list) over the plane of its wall-side corners, against the requested
  // fraction of its level's thickness. Other cells always pass.
  bool keeps_layer_floor(int32_t c) const {
    if (mesh.cell_roles[c] != kLayer) return true;
    const int64_t first = mesh.cell_vertex_offsets[c];
    const int64_t count = mesh.cell_vertex_offsets[c + 1] - first;
    if (count != 6 && count != 8) return true;
    const int64_t half = count / 2;
    auto corner = [&](int64_t i) { return point(mesh.cell_vertices[first + i]); };
    Vec normal{0, 0, 0};
    for (int64_t i = 0; i < half; ++i) normal = add(normal, cross(corner(half + i), corner(half + (i + 1) % half)));
    const double size = norm(normal);
    for (double& component : normal) component /= size;
    double height = 0.0;
    for (int64_t i = 0; i < half; ++i) height += dot(sub(corner(i), corner(half + i)), normal);
    const auto level = static_cast<std::size_t>((c - core_cells) % levels);
    // A degenerate wall-side plane gives NaN, which fails the comparison.
    return std::fabs(height) / static_cast<double>(half) >=
           in.layer_minimum_thickness_fraction * in.layer_thicknesses[level];
  }
  int32_t smooth();
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
  const int64_t gx = tree.dims[0], gy = tree.dims[1], gz = tree.dims[2];
  const int64_t clearance_cells = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(delta / b)));
  const bool use_dense =
      gx <= 512 && gy <= 512 && gz <= 512 && (gx + 1) * (gy + 1) * (gz + 1) <= 16'000'000;
  if (use_dense) {
    // Finest cells touching a triangle box on a dense grid.
    std::vector<uint8_t> touched_grid(static_cast<std::size_t>(gx * gy * gz), 0);
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
        r0[a] = std::max<int64_t>(0, static_cast<int64_t>(std::floor((mn[a] - tree.lo[a]) / b)) - 1);
        r1[a] = std::min<int64_t>(tree.dims[a] - 1, static_cast<int64_t>(std::floor((mx[a] - tree.lo[a]) / b)) + 1);
      }
      for (int64_t i = r0[0]; i <= r1[0]; ++i) {
        const double px = tree.lo[0] + static_cast<double>(i) * b;
        const double qx = tree.lo[0] + static_cast<double>(i + 1) * b;
        if (px > mx[0] || qx < mn[0]) continue;
        for (int64_t j = r0[1]; j <= r1[1]; ++j) {
          const double py = tree.lo[1] + static_cast<double>(j) * b;
          const double qy = tree.lo[1] + static_cast<double>(j + 1) * b;
          if (py > mx[1] || qy < mn[1]) continue;
          const std::size_t base = static_cast<std::size_t>((i * gy + j) * gz);
          for (int64_t k = r0[2]; k <= r1[2]; ++k) {
            const double pz = tree.lo[2] + static_cast<double>(k) * b;
            const double qz = tree.lo[2] + static_cast<double>(k + 1) * b;
            if (pz > mx[2] || qz < mn[2]) continue;
            touched_grid[base + k] = 1;
          }
        }
      }
    }
    std::vector<std::vector<uint8_t>> split_grid(static_cast<std::size_t>(std::max(K, 1)));
    for (int l = K - 1; l >= 0; --l) {
      const int64_t sx = coarse[0] << l, sy = coarse[1] << l, sz = coarse[2] << l;
      split_grid[l] = dilate_grid(coarsen_grid(touched_grid, gx, gy, gz, K - l), sx, sy, sz, kBandCells);
      if (l + 1 < K) {
        const std::vector<uint8_t> bal =
            dilate_grid(coarsen_grid(split_grid[l + 1], coarse[0] << (l + 1), coarse[1] << (l + 1),
                                     coarse[2] << (l + 1), 1),
                        sx, sy, sz, 1);
        for (std::size_t idx = 0; idx < bal.size(); ++idx) split_grid[l][idx] |= bal[idx];
      }
    }
    const std::vector<uint8_t> near_finest_grid = dilate_grid(touched_grid, gx, gy, gz, clearance_cells);
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
            const int64_t sy_l = coarse[1] << level, sz_l = coarse[2] << level;
            if (level < K &&
                split_grid[level][static_cast<std::size_t>(((o[0] / size) * sy_l + (o[1] / size)) * sz_l + (o[2] / size))] != 0) {
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
    // Core classification (deduped lattice corners evaluated in parallel).
    InsideOracle oracle(s, tree.lo, b);
    int32_t st = oracle.resolve_all(in.parallelism);
    if (st != PHX_MC_OK) return st;
    const int64_t nx = gx + 1, ny = gy + 1, nz = gz + 1;
    const std::size_t npts = static_cast<std::size_t>(nx * ny * nz);
    auto pt_idx = [ny, nz](int64_t i, int64_t j, int64_t k) -> std::size_t {
      return static_cast<std::size_t>((i * ny + j) * nz + k);
    };
    std::vector<uint8_t> inside_need(npts, 0), inside_val(npts, 0);
    std::vector<uint64_t> inside_pts;
    inside_pts.reserve(tree.leaves.size() * 2);
    std::vector<int32_t> near_leaves;
    near_leaves.reserve(tree.leaves.size() / 2);
    auto request_inside = [&](int64_t i, int64_t j, int64_t k) {
      const std::size_t idx = pt_idx(i, j, k);
      if (!inside_need[idx]) {
        inside_need[idx] = 1;
        inside_pts.push_back(pack(i, j, k));
      }
    };
    for (std::size_t li = 0; li < tree.leaves.size(); ++li) {
      const Leaf& leaf = tree.leaves[li];
      const int64_t i = leaf.o[0], j = leaf.o[1], k = leaf.o[2];
      if (leaf.level < K || !near_finest_grid[static_cast<std::size_t>((i * gy + j) * gz + k)]) {
        request_inside(i, j, k);
      } else {
        near_leaves.push_back(static_cast<int32_t>(li));
        for (int c = 0; c < 8; ++c) {
          request_inside(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
        }
      }
    }
    auto eval_inside = [&](std::size_t idx) {
      const auto p = unpack(inside_pts[idx]);
      inside_val[pt_idx(p[0], p[1], p[2])] = oracle.inside_resolved(p[0], p[1], p[2]) ? 1 : 0;
    };
    if (in.parallelism == 1 || inside_pts.size() <= 512) {
      for (std::size_t idx = 0; idx < inside_pts.size(); ++idx) eval_inside(idx);
    } else {
      st = parallel_for(inside_pts.size(), ParallelRequest{512, in.parallelism},
                        [&](std::size_t begin, std::size_t end, ChunkContext&) -> int32_t {
        const GradualNearestScope mode;
        if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
        for (std::size_t idx = begin; idx < end; ++idx) eval_inside(idx);
        return PHX_MC_OK;
      });
      if (st != PHX_MC_OK) return st;
    }
    std::vector<uint8_t> clear_need(npts, 0), clear_val(npts, 0);
    std::vector<uint64_t> clear_pts;
    clear_pts.reserve(near_leaves.size() * 2);
    std::vector<int32_t> near_inside_leaves;
    near_inside_leaves.reserve(near_leaves.size());
    for (const int32_t li : near_leaves) {
      const Leaf& leaf = tree.leaves[li];
      const int64_t i = leaf.o[0], j = leaf.o[1], k = leaf.o[2];
      bool all_in = true;
      for (int c = 0; c < 8 && all_in; ++c) {
        all_in = inside_val[pt_idx(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1))] != 0;
      }
      if (!all_in) continue;
      near_inside_leaves.push_back(li);
      for (int c = 0; c < 8; ++c) {
        const int64_t ci = i + (c & 1), cj = j + ((c >> 1) & 1), ck = k + ((c >> 2) & 1);
        const std::size_t idx = pt_idx(ci, cj, ck);
        if (!clear_need[idx]) {
          clear_need[idx] = 1;
          clear_pts.push_back(pack(ci, cj, ck));
        }
      }
    }
    auto eval_clear = [&](std::size_t idx) {
      const auto p = unpack(clear_pts[idx]);
      clear_val[pt_idx(p[0], p[1], p[2])] = !triangle_within(s, lattice_point(p[0], p[1], p[2]), delta) ? 1 : 0;
    };
    if (in.parallelism == 1 || clear_pts.size() <= 128) {
      for (std::size_t idx = 0; idx < clear_pts.size(); ++idx) eval_clear(idx);
    } else {
      st = parallel_for(clear_pts.size(), ParallelRequest{128, in.parallelism},
                        [&](std::size_t begin, std::size_t end, ChunkContext&) -> int32_t {
        const GradualNearestScope mode;
        if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
        for (std::size_t idx = begin; idx < end; ++idx) eval_clear(idx);
        return PHX_MC_OK;
      });
      if (st != PHX_MC_OK) return st;
    }
    std::vector<int32_t> overlap_leaves;
    overlap_leaves.reserve(near_inside_leaves.size());
    for (const int32_t li : near_inside_leaves) {
      const Leaf& leaf = tree.leaves[li];
      const int64_t i = leaf.o[0], j = leaf.o[1], k = leaf.o[2];
      bool all_clear = true;
      for (int c = 0; c < 8 && all_clear; ++c) {
        all_clear = clear_val[pt_idx(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1))] != 0;
      }
      if (all_clear) overlap_leaves.push_back(li);
    }
    for (Leaf& leaf : tree.leaves) {
      const int64_t i = leaf.o[0], j = leaf.o[1], k = leaf.o[2];
      if (leaf.level < K || !near_finest_grid[static_cast<std::size_t>((i * gy + j) * gz + k)]) {
        leaf.core = inside_val[pt_idx(i, j, k)];
      }
    }
    auto eval_overlap = [&](std::size_t idx) {
      Leaf& leaf = tree.leaves[overlap_leaves[idx]];
      const int64_t i = leaf.o[0], j = leaf.o[1], k = leaf.o[2];
      const Vec center = lattice_point(i, j, k);
      const Vec mid = add(center, {0.5 * b, 0.5 * b, 0.5 * b});
      const Bins& bins = s.triangle_bins;
      const auto a0 = bins.cell(center), a1 = bins.cell(lattice_point(i + 1, j + 1, k + 1));
      bool core = true;
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
      leaf.core = core ? 1 : 0;
    };
    if (in.parallelism == 1 || overlap_leaves.size() <= 256) {
      for (std::size_t idx = 0; idx < overlap_leaves.size(); ++idx) eval_overlap(idx);
    } else {
      st = parallel_for(overlap_leaves.size(), ParallelRequest{256, in.parallelism},
                        [&](std::size_t begin, std::size_t end, ChunkContext&) -> int32_t {
        const GradualNearestScope mode;
        if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
        for (std::size_t idx = begin; idx < end; ++idx) eval_overlap(idx);
        return PHX_MC_OK;
      });
      if (st != PHX_MC_OK) return st;
    }
  } else {
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
      const Vec center = lattice_point(i, j, k);
      const Vec mid = add(center, {0.5 * b, 0.5 * b, 0.5 * b});
      const Bins& bins = s.triangle_bins;
      const auto a0 = bins.cell(center), a1 = bins.cell(lattice_point(i + 1, j + 1, k + 1));
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
  }
  // Manifold staircase: remove finest core cells at non-manifold edges and
  // vertices (the involved core cell whose center is nearest the surface).
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
int32_t Builder::partition() {
  const int64_t count = static_cast<int64_t>(stair_faces.size());
  stair_patch.assign(count, 0);
  if (s.patch_count <= 1) return PHX_MC_OK;
  auto assign_patch = [&](std::size_t f) {
    Vec c{0, 0, 0};
    for (const int32_t v : stair_loop[f]) c = add(c, point(v));
    stair_patch[f] = s.patch[nearest_triangle(s, mul(c, 0.25), -1).item];
  };
  if (in.parallelism == 1 || count <= 256) {
    for (std::size_t f = 0; f < static_cast<std::size_t>(count); ++f) assign_patch(f);
  } else {
    const int32_t st = parallel_for(
        static_cast<std::size_t>(count), ParallelRequest{256, in.parallelism},
        [&](std::size_t begin, std::size_t end, ChunkContext&) -> int32_t {
          const GradualNearestScope mode;
          if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
          for (std::size_t f = begin; f < end; ++f) assign_patch(f);
          return PHX_MC_OK;
        });
    if (st != PHX_MC_OK) return st;
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
  return PHX_MC_OK;
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
      // With layers the step splits along the corner's own diagonal and
      // keeps its patch: the chain keeps every staircase vertex (the
      // layer split at convex curves needs no flattening guard).
      const bool keep = in.layer_count > 0;
      if (corner >= 0 && !keep && (splits[stair_adjacent[f][(corner + 3) % 4]] || splits[stair_adjacent[f][corner]])) {
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
      if (keep) {
        pieces.push_back({static_cast<int32_t>(f), 3, {before, at, opposite, -1}, stair_patch[f], -1});
        pieces.push_back({static_cast<int32_t>(f), 3, {at, after, opposite, -1}, stair_patch[f], -1});
        continue;
      }
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
  struct ProjectRow {
    Hit hit;
    uint8_t kind = kPatch;
    std::array<int32_t, 2> pair{-1, -1};
    std::array<int32_t, 2> chain_nbr{-1, -1};
    uint8_t is_curve = 0;
    uint8_t is_corner = 0;
  };
  std::vector<ProjectRow> projected(static_cast<std::size_t>(vertices));
  auto project_range = [&](std::size_t begin, std::size_t end) {
    std::vector<int32_t> patches;
    for (std::size_t r = begin; r < end; ++r) {
      const int64_t row = static_cast<int64_t>(r);
      const int32_t v = stair_vertices[row];
      const Vec p = point(v);
      patches.clear();
      for (const int32_t q : around[row]) patches.push_back(pieces[q].patch);
      std::sort(patches.begin(), patches.end());
      patches.erase(std::unique(patches.begin(), patches.end()), patches.end());
      ProjectRow out;
      if (patches.size() == 1) {
        out.hit = nearest_triangle(s, p, patches[0]);
        out.kind = kPatch;
      } else if (patches.size() == 2) {
        out.hit = nearest_edge(s, p, patches[0], patches[1]);
        if (out.hit.item < 0) out.hit = nearest_edge(s, p, -1, -1);
        out.kind = kFeature;
        out.pair = {patches[0], patches[1]};
        if (chain[row].size() == 2) out.chain_nbr = {chain[row][0], chain[row][1]};
        out.is_curve = 1;
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
        out.kind = kCorner;
        if (chosen >= 0) {
          out.hit.item = s.corners[chosen];
          out.hit.point = s.point(s.corners[chosen]);
        } else {
          out.hit = nearest_edge(s, p, -1, -1);
        }
        out.is_corner = 1;
      }
      if (out.hit.item < 0) {
        out.hit = nearest_triangle(s, p, -1);
        out.kind = kPatch;
      }
      projected[r] = out;
    }
  };
  if (in.parallelism == 1 || vertices <= 128) {
    project_range(0, static_cast<std::size_t>(vertices));
  } else {
    const int32_t st = parallel_for(
        static_cast<std::size_t>(vertices), ParallelRequest{128, in.parallelism},
        [&](std::size_t begin, std::size_t end, ChunkContext&) -> int32_t {
          const GradualNearestScope mode;
          if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
          project_range(begin, end);
          return PHX_MC_OK;
        });
    if (st != PHX_MC_OK) return st;
  }
  int64_t curve = 0, corner = 0;
  for (int64_t row = 0; row < vertices; ++row) {
    const ProjectRow& out = projected[static_cast<std::size_t>(row)];
    surface_kind[row] = out.kind;
    surface_pair[row] = out.pair;
    chain_neighbors[row] = out.chain_nbr;
    surface_item[row] = out.hit.item;
    surface_vertex[row] = add_point(out.hit.point);
    curve += out.is_curve;
    corner += out.is_corner;
  }
  stats[19] = static_cast<double>(curve);
  stats[20] = static_cast<double>(corner);
  return PHX_MC_OK;
}

// Convex feature-curve columns to split: a curve vertex with two chain
// neighbors between two patches whose normals (area-weighted over its wall
// pieces) differ by more than the feature angle, each side's pieces behind
// the other side's plane; and only while both chain neighbors split too, so
// every split vertex has two corner segments.
void Builder::classify_splits() {
  const int64_t rows = static_cast<int64_t>(stair_vertices.size());
  split_of.assign(rows, -1);
  splits.clear();
  if (levels == 1) return;
  const double feature_cos = std::cos(in.feature_angle_degrees * (3.14159265358979323846 / 180.0));
  std::vector<uint8_t> candidate(rows, 0);
  for (int64_t row = 0; row < rows; ++row) {
    if (surface_kind[row] != kFeature || chain_neighbors[row][0] < 0) continue;
    const auto& pair = surface_pair[row];
    const Vec p = point(surface_vertex[row]);
    Vec area[2] = {{0, 0, 0}, {0, 0, 0}}, center[2] = {{0, 0, 0}, {0, 0, 0}};
    for (const int32_t q : row_pieces[row]) {
      const Piece& piece = pieces[q];
      const int side = piece.patch == pair[0] ? 0 : 1;
      Vec mid{0, 0, 0};
      for (int c = 0; c < piece.count; ++c) {
        const Vec u = sub(point(surface_vertex[stair_vertex[piece.loop[c]]]), p);
        const Vec w = sub(point(surface_vertex[stair_vertex[piece.loop[(c + 1) % piece.count]]]), p);
        area[side] = add(area[side], cross(u, w));
        mid = add(mid, u);
      }
      center[side] = add(center[side], mul(mid, 1.0 / piece.count));
    }
    if (!(norm(area[0]) > 0.0) || !(norm(area[1]) > 0.0)) continue;
    const Vec n0 = mul(area[0], 1.0 / norm(area[0])), n1 = mul(area[1], 1.0 / norm(area[1]));
    candidate[row] = dot(n0, n1) <= feature_cos && dot(center[0], n1) < 0.0 && dot(center[1], n0) < 0.0;
  }
  for (bool again = true; again;) {
    again = false;
    for (int64_t row = 0; row < rows; ++row) {
      if (!candidate[row]) continue;
      for (const int32_t other : chain_neighbors[row]) {
        if (candidate[other]) continue;
        candidate[row] = 0;
        again = true;
        break;
      }
    }
  }
  for (int64_t row = 0; row < rows; ++row) {
    if (!candidate[row]) continue;
    split_of[row] = static_cast<int32_t>(splits.size());
    splits.push_back({static_cast<int32_t>(row), surface_pair[row], {}, {}});
  }
}

// Projection cells (levels == 1) or columns split into levels cells: a
// hexahedron per quadrilateral piece and level, a prism per triangle. At a
// split curve vertex each side's pieces take their own column of the grid,
// and every curve segment between split vertices gets L x L corner
// hexahedra. Layer points are placed by place_layers.
int32_t Builder::emit_layers() {
  mesh.points.resize(3 * (core_points + static_cast<int64_t>(stair_vertices.size())));
  mesh.face_offsets.resize(core_faces + 1);
  mesh.face_vertices.resize(core_face_entries);
  mesh.face_owner.resize(core_faces);
  mesh.face_neighbor.resize(core_faces);
  mesh.cell_kinds.resize(core_cells);
  mesh.cell_roles.resize(core_cells);
  mesh.cell_vertex_offsets.resize(core_cells + 1);
  mesh.cell_vertices.resize(core_cell_entries);
  classify_splits();
  const int32_t layers = levels - 1;
  const int64_t columns = static_cast<int64_t>(stair_vertices.size());
  column.assign(static_cast<std::size_t>(columns) * (levels + 1), -1);
  for (int64_t row = 0; row < columns; ++row) {
    column[row * (levels + 1)] = surface_vertex[row];
    column[row * (levels + 1) + levels] = stair_vertices[row];
    if (split_of[row] >= 0) {
      Split& split = splits[split_of[row]];
      split.grid.assign(static_cast<std::size_t>((layers + 1) * (layers + 1)), surface_vertex[row]);
      for (std::size_t q = 1; q < split.grid.size(); ++q) split.grid[q] = add_point(point(surface_vertex[row]));
      split.item.assign(static_cast<std::size_t>(2 * layers), -1);
      // The diagonal stands in for the column's layer points.
      for (int32_t k = 1; k < levels; ++k) column[row * (levels + 1) + k] = split.grid[k * (layers + 2)];
      continue;
    }
    for (int32_t k = 1; k < levels; ++k) column[row * (levels + 1) + k] = add_point(point(surface_vertex[row]));
  }
  // Column point of staircase vertex v at level for a piece of patch.
  auto at = [&](int32_t v, int32_t level, int32_t patch) {
    const int32_t row = stair_vertex[v];
    if (split_of[row] < 0 || level == levels) return column[row * (levels + 1) + level];
    const Split& split = splits[split_of[row]];
    return patch == split.pair[0] ? split.grid[layers * (layers + 1) + level] : split.grid[level * (layers + 1) + layers];
  };
  // A curve segment between split vertices.
  auto segment = [&](int32_t a, int32_t c) {
    const int32_t ra = stair_vertex[a], rc = stair_vertex[c];
    return split_of[ra] >= 0 && split_of[rc] >= 0 &&
           (chain_neighbors[ra][0] == rc || chain_neighbors[ra][1] == rc);
  };
  // Segment midpoints (see Midpoint): the deviation bound is the largest
  // chord deviation of the wall edges between patch vertices.
  mids.clear();
  mid_rows = KeyMap(16);
  mid_points = KeyMap(16);
  if (!splits.empty()) {
    double bound = 0.0;
    for (const Piece& piece : pieces) {
      for (int c = 0; c < piece.count; ++c) {
        const int32_t u = stair_vertex[piece.loop[c]], w = stair_vertex[piece.loop[(c + 1) % piece.count]];
        if (u > w || surface_kind[u] != kPatch || surface_kind[w] != kPatch) continue;
        const Vec mid = mul(add(point(surface_vertex[u]), point(surface_vertex[w])), 0.5);
        bound = std::max(bound, walk_triangle(s, mid, surface_item[u]).d2);
      }
    }
    for (const Split& split : splits) {
      for (const int32_t other : chain_neighbors[split.row]) {
        if (other < split.row) continue;
        const Vec mid = mul(add(point(surface_vertex[split.row]), point(surface_vertex[other])), 0.5);
        if (!(walk_edge(s, mid, surface_item[split.row]).d2 > bound)) continue;
        Midpoint m{split.row, other, {}, -1, std::vector<int32_t>(static_cast<std::size_t>(2 * layers), -1)};
        const auto& first = split.grid;
        const auto& second = splits[split_of[other]].grid;
        for (std::size_t q = 0; q < first.size(); ++q) {
          m.grid.push_back(add_point(mid));
          mid_points.assign(pair_key(first[q], second[q]), m.grid.back());
        }
        mid_rows.assign(pair_key(split.row, other), static_cast<int32_t>(mids.size()));
        mids.push_back(std::move(m));
      }
    }
  }
  auto mid_of = [&](int32_t a, int32_t c) {
    return segment(a, c) ? mid_rows.find(pair_key(stair_vertex[a], stair_vertex[c])) : -1;
  };
  // A face loop with the midpoints of its end grid pairs.
  std::vector<int32_t> expanded;
  auto add_expanded = [&](const int32_t* loop, int count, int32_t owner, int32_t neighbor) {
    if (mids.empty()) return add_face(loop, count, owner, neighbor);
    expanded.clear();
    for (int c = 0; c < count; ++c) {
      expanded.push_back(loop[c]);
      const int32_t m = mid_points.find(pair_key(loop[c], loop[(c + 1) % count]));
      if (m >= 0) expanded.push_back(m);
    }
    return add_face(expanded.data(), static_cast<int>(expanded.size()), owner, neighbor);
  };
  const int64_t count = static_cast<int64_t>(pieces.size());
  const int32_t first_cell = static_cast<int32_t>(core_cells);
  // Cell of piece p at level k (k = 0 at the wall).
  auto cell_of = [&](int64_t p, int32_t k) { return first_cell + static_cast<int32_t>(p * levels + k); };
  for (int64_t p = 0; p < count; ++p) {
    const Piece& piece = pieces[p];
    mesh.face_neighbor[piece.core_face] = cell_of(p, levels - 1);
    // A piece along a split segment holds the midpoint: polyhedra.
    bool split_edge = false;
    for (int c = 0; c < piece.count && !mids.empty(); ++c) {
      split_edge = split_edge || mid_of(piece.loop[c], piece.loop[(c + 1) % piece.count]) >= 0;
    }
    for (int32_t k = 0; k < levels; ++k) {
      mesh.cell_kinds.push_back(split_edge ? kPolyhedron : piece.count == 4 ? kHexahedron : kPrism);
      mesh.cell_roles.push_back(levels > 1 && k < levels - 1 ? kLayer : kProjection);
      if (!split_edge) {
        for (int c = 0; c < piece.count; ++c) mesh.cell_vertices.push_back(at(piece.loop[c], k + 1, piece.patch));
        for (int c = 0; c < piece.count; ++c) mesh.cell_vertices.push_back(at(piece.loop[c], k, piece.patch));
      }
      mesh.cell_vertex_offsets.push_back(static_cast<int64_t>(mesh.cell_vertices.size()));
    }
  }
  int32_t loop[4];
  piece_wall_face.assign(count, -1);
  for (int64_t p = 0; p < count; ++p) {
    const Piece& piece = pieces[p];
    // The wall face (outward along the staircase normal) and the faces
    // between levels.
    for (int c = 0; c < piece.count; ++c) loop[c] = at(piece.loop[c], 0, piece.patch);
    piece_wall_face[p] = add_expanded(loop, piece.count, cell_of(p, 0), -1);
    for (int32_t k = 1; k < levels; ++k) {
      for (int c = 0; c < piece.count; ++c) loop[c] = at(piece.loop[c], k, piece.patch);
      add_expanded(loop, piece.count, cell_of(p, k), cell_of(p, k - 1));
    }
  }
  // Side faces, owned by the piece whose loop runs a -> b at the second
  // sighting of the edge (the first runs b -> a); the layer levels of a
  // curve segment face the corner column instead.
  KeyMap seen(static_cast<std::size_t>(2 * count + 16));
  const uint64_t n = static_cast<uint64_t>(core_points);
  for (int64_t p = 0; p < count; ++p) {
    const Piece& piece = pieces[p];
    for (int e = 0; e < piece.count; ++e) {
      const int32_t a = piece.loop[e], c = piece.loop[(e + 1) % piece.count];
      const uint64_t key = static_cast<uint64_t>(std::min(a, c)) * n + static_cast<uint64_t>(std::max(a, c));
      const int32_t other = seen.emplace(key, static_cast<int32_t>(p));
      if (other == static_cast<int32_t>(p)) continue;
      for (int32_t k = segment(a, c) ? layers : 0; k < levels; ++k) {
        loop[0] = at(a, k + 1, piece.patch);
        loop[1] = at(c, k + 1, piece.patch);
        loop[2] = at(c, k, piece.patch);
        loop[3] = at(a, k, piece.patch);
        add_expanded(loop, 4, cell_of(p, k), cell_of(other, k));
      }
    }
  }
  if (splits.empty()) return PHX_MC_OK;
  // Corner columns and the faces around them: every face as seen from its
  // cell (outward loop), matched by vertex set; an unmatched face is a wall
  // strip of a corner cell.
  struct Sighting {
    std::array<int32_t, 4> key;
    int32_t cell;
    std::array<int32_t, 4> loop;
    bool wall;
  };
  std::vector<Sighting> sightings;
  auto sight = [&](int32_t cell, const std::array<int32_t, 4>& face, bool wall) {
    std::array<int32_t, 4> key = face;
    std::sort(key.begin(), key.end());
    sightings.push_back({key, cell, face, wall});
  };
  // Outward faces of a hexahedron in the reference order: bottom, top,
  // then the sides from edge 01 on.
  static constexpr int kFaces[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
  // The corner hexahedra between grids gb (first quadrilateral) and ga.
  auto corner_cells = [&](const std::vector<int32_t>& gb, const std::vector<int32_t>& ga) {
    for (int32_t i = 0; i < layers; ++i) {
      for (int32_t j = 0; j < layers; ++j) {
        const int32_t q[4] = {i * (layers + 1) + j, (i + 1) * (layers + 1) + j, (i + 1) * (layers + 1) + j + 1,
                              i * (layers + 1) + j + 1};
        const int32_t h[8] = {gb[q[0]], gb[q[1]], gb[q[2]], gb[q[3]], ga[q[0]], ga[q[1]], ga[q[2]], ga[q[3]]};
        const int32_t cell = static_cast<int32_t>(mesh.cell_kinds.size());
        mesh.cell_kinds.push_back(kHexahedron);
        mesh.cell_roles.push_back(kLayer);
        mesh.cell_vertices.insert(mesh.cell_vertices.end(), h, h + 8);
        mesh.cell_vertex_offsets.push_back(static_cast<int64_t>(mesh.cell_vertices.size()));
        for (int f = 0; f < 6; ++f) {
          const bool wall = (f == 2 && j == 0) || (f == 5 && i == 0);
          sight(cell, {h[kFaces[f][0]], h[kFaces[f][1]], h[kFaces[f][2]], h[kFaces[f][3]]}, wall);
        }
      }
    }
  };
  for (int64_t p = 0; p < count; ++p) {
    const Piece& piece = pieces[p];
    for (int e = 0; e < piece.count; ++e) {
      const int32_t a = piece.loop[e], c = piece.loop[(e + 1) % piece.count];
      if (!segment(a, c)) continue;
      const Split& sa = splits[split_of[stair_vertex[a]]];
      const int32_t mi = mid_of(a, c);
      for (int32_t k = 0; k < layers; ++k) {
        const int32_t a0 = at(a, k, piece.patch), a1 = at(a, k + 1, piece.patch);
        const int32_t c0 = at(c, k, piece.patch), c1 = at(c, k + 1, piece.patch);
        if (mi < 0) {
          sight(cell_of(p, k), {a1, c1, c0, a0}, false);
          continue;
        }
        const int32_t m0 = mid_points.find(pair_key(a0, c0)), m1 = mid_points.find(pair_key(a1, c1));
        sight(cell_of(p, k), {a1, m1, m0, a0}, false);
        sight(cell_of(p, k), {m1, c1, c0, m0}, false);
      }
      if (piece.patch != sa.pair[0]) continue;
      // The first patch's piece runs a -> c: with g(i + 1, j) - g(i, j)
      // into the first patch and g(i, j + 1) - g(i, j) into the second, the
      // corner hexahedron is positive with c's quadrilateral first.
      const auto& ga = sa.grid;
      const auto& gc = splits[split_of[stair_vertex[c]]].grid;
      if (mi < 0) {
        corner_cells(gc, ga);
      } else {
        corner_cells(mids[mi].grid, ga);
        corner_cells(gc, mids[mi].grid);
      }
    }
  }
  std::sort(sightings.begin(), sightings.end(), [](const Sighting& x, const Sighting& y) {
    return x.key != y.key ? x.key < y.key : x.cell < y.cell;
  });
  for (std::size_t q = 0; q < sightings.size();) {
    const Sighting& first = sightings[q];
    if (q + 1 < sightings.size() && sightings[q + 1].key == first.key) {
      if (first.wall || sightings[q + 1].wall || (q + 2 < sightings.size() && sightings[q + 2].key == first.key)) {
        return PHX_MC_INTERNAL_ERROR;
      }
      add_face(first.loop.data(), 4, first.cell, sightings[q + 1].cell);
      q += 2;
      continue;
    }
    if (!first.wall) return PHX_MC_INTERNAL_ERROR;
    add_face(first.loop.data(), 4, first.cell, -1);
    ++q;
  }
  return PHX_MC_OK;
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
  // Vertex -> cells (unique per cell, strictly ascending cell order).
  const int64_t points = static_cast<int64_t>(mesh.points.size() / 3);
  t.vertex_cell_offsets.assign(points + 1, 0);
  std::vector<int32_t> seen(points, -1);
  std::vector<int32_t> cell_verts;
  cell_verts.reserve(static_cast<std::size_t>(cells) * 9);
  std::vector<int64_t> cell_vert_offsets(cells + 1, 0);
  for (int64_t c = 0; c < cells; ++c) {
    const int32_t cid = static_cast<int32_t>(c);
    for (int64_t slot = t.cell_face_offsets[c]; slot < t.cell_face_offsets[c + 1]; ++slot) {
      const int64_t f = std::abs(t.cell_faces[slot]) - 1;
      for (int64_t q = mesh.face_offsets[f]; q < mesh.face_offsets[f + 1]; ++q) {
        const int32_t v = mesh.face_vertices[q];
        if (seen[v] != cid) {
          seen[v] = cid;
          ++t.vertex_cell_offsets[v + 1];
          cell_verts.push_back(v);
        }
      }
    }
    cell_vert_offsets[c + 1] = static_cast<int64_t>(cell_verts.size());
  }
  for (int64_t v = 0; v < points; ++v) t.vertex_cell_offsets[v + 1] += t.vertex_cell_offsets[v];
  t.vertex_cells.resize(t.vertex_cell_offsets[points]);
  std::vector<int64_t> vfill(t.vertex_cell_offsets.begin(), t.vertex_cell_offsets.end() - 1);
  for (int64_t c = 0; c < cells; ++c) {
    const int32_t cid = static_cast<int32_t>(c);
    for (int64_t q = cell_vert_offsets[c]; q < cell_vert_offsets[c + 1]; ++q) {
      t.vertex_cells[vfill[cell_verts[q]]++] = cid;
    }
  }
  return t;
}

// OpenFOAM face center and area vector (triangles about the vertex average).
void face_geometry(const Mesh& mesh, int64_t f, Vec& center, Vec& area, Vec& average) {
  const int64_t first = mesh.face_offsets[f], last = mesh.face_offsets[f + 1];
  const int64_t nverts = last - first;
  const double count = static_cast<double>(nverts);
  const double* x = mesh.points.data();
  const int32_t* fv = mesh.face_vertices.data() + first;
  average = {0, 0, 0};
  area = {0, 0, 0};
  Vec weighted{0, 0, 0};
  double total = 0.0;
  if (nverts <= 16) {
    Vec pts[16];
    for (int64_t i = 0; i < nverts; ++i) {
      pts[i] = load(x + 3 * fv[i]);
      average = add(average, pts[i]);
    }
    average = mul(average, 1.0 / count);
    for (int64_t i = 0; i < nverts; ++i) {
      const Vec& p = pts[i];
      const Vec& n = pts[i + 1 < nverts ? i + 1 : 0];
      const Vec a = mul(cross(sub(p, average), sub(n, average)), 0.5);
      const double magnitude = norm(a);
      area = add(area, a);
      weighted = add(weighted, mul(add(add(p, n), average), magnitude / 3.0));
      total += magnitude;
    }
  } else {
    for (int64_t i = 0; i < nverts; ++i) average = add(average, load(x + 3 * fv[i]));
    average = mul(average, 1.0 / count);
    for (int64_t i = 0; i < nverts; ++i) {
      const Vec p = load(x + 3 * fv[i]);
      const Vec n = load(x + 3 * fv[i + 1 < nverts ? i + 1 : 0]);
      const Vec a = mul(cross(sub(p, average), sub(n, average)), 0.5);
      const double magnitude = norm(a);
      area = add(area, a);
      weighted = add(weighted, mul(add(add(p, n), average), magnitude / 3.0));
      total += magnitude;
    }
  }
  center = total > 0.0 ? mul(weighted, 1.0 / total) : average;
}

// A face's vertex-average fan does not fold: every fan triangle's area
// vector points along the loop's area vector N (the certifier's refusal of a
// folded fan), here with a relative margin of 1e-6 of N . N per triangle so
// that rounding cannot accept a folded fan.
bool fan_unfolded(const Mesh& mesh, int64_t f) {
  const int64_t first = mesh.face_offsets[f], last = mesh.face_offsets[f + 1];
  const int64_t nverts = last - first;
  const double* x = mesh.points.data();
  const int32_t* fv = mesh.face_vertices.data() + first;
  const double count = static_cast<double>(nverts);
  Vec average{0, 0, 0}, normal{0, 0, 0};
  if (nverts <= 16) {
    Vec pts[16];
    for (int64_t i = 0; i < nverts; ++i) pts[i] = load(x + 3 * fv[i]);
    for (int64_t i = 0; i < nverts; ++i) {
      average = add(average, pts[i]);
      normal = add(normal, cross(pts[i], pts[i + 1 < nverts ? i + 1 : 0]));
    }
    average = mul(average, 1.0 / count);
    const double floor = 1e-6 * dot(normal, normal) / count;
    for (int64_t i = 0; i < nverts; ++i) {
      const Vec a = cross(sub(pts[i], average), sub(pts[i + 1 < nverts ? i + 1 : 0], average));
      if (!(dot(a, normal) > floor)) return false;
    }
    return true;
  }
  for (int64_t i = 0; i < nverts; ++i) {
    const Vec p = load(x + 3 * fv[i]);
    const Vec n = load(x + 3 * fv[i + 1 < nverts ? i + 1 : 0]);
    average = add(average, p);
    normal = add(normal, cross(p, n));
  }
  average = mul(average, 1.0 / count);
  const double floor = 1e-6 * dot(normal, normal) / count;
  for (int64_t i = 0; i < nverts; ++i) {
    const Vec p = load(x + 3 * fv[i]);
    const Vec n = load(x + 3 * fv[i + 1 < nverts ? i + 1 : 0]);
    const Vec a = cross(sub(p, average), sub(n, average));
    if (!(dot(a, normal) > floor)) return false;
  }
  return true;
}

// Combined face_geometry + fan_unfolded sharing vertex loads and fan cross products.
void face_geometry_and_unfolded(const Mesh& mesh, int64_t f, Vec& center, Vec& area, Vec& average,
                                uint8_t& unfolded) {
  const int64_t first = mesh.face_offsets[f], last = mesh.face_offsets[f + 1];
  const int64_t nverts = last - first;
  const double count = static_cast<double>(nverts);
  const double* x = mesh.points.data();
  const int32_t* fv = mesh.face_vertices.data() + first;
  if (nverts > 16) {
    face_geometry(mesh, f, center, area, average);
    unfolded = fan_unfolded(mesh, f);
    return;
  }
  Vec pts[16];
  for (int64_t i = 0; i < nverts; ++i) pts[i] = load(x + 3 * fv[i]);
  average = {0, 0, 0};
  Vec normal{0, 0, 0};
  for (int64_t i = 0; i < nverts; ++i) {
    average = add(average, pts[i]);
    normal = add(normal, cross(pts[i], pts[i + 1 < nverts ? i + 1 : 0]));
  }
  average = mul(average, 1.0 / count);
  const double floor = 1e-6 * dot(normal, normal) / count;
  area = {0, 0, 0};
  Vec weighted{0, 0, 0};
  double total = 0.0;
  bool ok = true;
  for (int64_t i = 0; i < nverts; ++i) {
    const Vec& p = pts[i];
    const Vec& n = pts[i + 1 < nverts ? i + 1 : 0];
    const Vec cr = cross(sub(p, average), sub(n, average));
    if (!(dot(cr, normal) > floor)) ok = false;
    const Vec a = mul(cr, 0.5);
    const double magnitude = norm(a);
    area = add(area, a);
    weighted = add(weighted, mul(add(add(p, n), average), magnitude / 3.0));
    total += magnitude;
  }
  center = total > 0.0 ? mul(weighted, 1.0 / total) : average;
  unfolded = ok ? 1 : 0;
}

// OpenFOAM cell centroid (pyramids about the mean of the face centers).
Vec cell_centroid(const Mesh& mesh, const Topology& t, int64_t c) {
  Vec estimate{0, 0, 0};
  const int64_t first = t.cell_face_offsets[c], last = t.cell_face_offsets[c + 1];
  Vec centers[64], areas[64];
  const int64_t count = std::min<int64_t>(last - first, 64);
  for (int64_t slot = 0; slot < count; ++slot) {
    const int32_t signed_face = t.cell_faces[first + slot];
    Vec average;
    face_geometry(mesh, std::abs(signed_face) - 1, centers[slot], areas[slot], average);
    if (signed_face < 0) areas[slot] = mul(areas[slot], -1.0);
    estimate = add(estimate, centers[slot]);
  }
  estimate = mul(estimate, 1.0 / static_cast<double>(count));
  double volume = 0.0;
  Vec moment{0, 0, 0};
  for (int64_t slot = 0; slot < count; ++slot) {
    const double v = dot(areas[slot], sub(centers[slot], estimate)) / 3.0;
    volume += v;
    moment = add(moment, mul(add(mul(centers[slot], 0.75), mul(estimate, 0.25)), v));
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

// The vertex average of a cell's distinct vertices: the center of the
// certified decomposition (the published cell's local map is the star from
// it to every face's fan triangles).
Vec cell_vertex_average(const Mesh& mesh, const Topology& t, int64_t c) {
  std::vector<int32_t> vertices;
  for (int64_t slot = t.cell_face_offsets[c]; slot < t.cell_face_offsets[c + 1]; ++slot) {
    const int64_t f = std::abs(t.cell_faces[slot]) - 1;
    vertices.insert(vertices.end(), mesh.face_vertices.begin() + mesh.face_offsets[f],
                    mesh.face_vertices.begin() + mesh.face_offsets[f + 1]);
  }
  std::sort(vertices.begin(), vertices.end());
  vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());
  Vec average{0, 0, 0};
  for (const int32_t v : vertices) average = add(average, load(mesh.points.data() + 3 * v));
  return mul(average, 1.0 / static_cast<double>(vertices.size()));
}

// Float screen of a cell: unfolded face fans, its decomposition tetrahedra
// (center, face vertex average, edge) about both centers above a relative
// margin: the OpenFOAM centroid (OpenFOAM's decomposition) and the vertex
// average of the distinct vertices (the published cell's local map, the
// star the certification checks); for hexahedra also the eight corner
// Jacobians. face(f, unfolded) yields face f's vertex average and sets
// whether its fan is unfolded, from whatever geometry the caller keeps.
template <class Face>
bool cell_screen(const Mesh& mesh, const Topology& t, int64_t c, const Vec& centroid, const Vec& vertex_average,
                 Face face) {
  const double* x = mesh.points.data();
  if (mesh.cell_kinds[c] == kHexahedron && !hexahedron_corners_positive(mesh, c)) return false;
  for (int64_t slot = t.cell_face_offsets[c]; slot < t.cell_face_offsets[c + 1]; ++slot) {
    const int32_t signed_face = t.cell_faces[slot];
    const int64_t f = std::abs(signed_face) - 1;
    bool unfolded = false;
    const Vec average = face(f, unfolded);
    if (!unfolded) return false;
    const int64_t first = mesh.face_offsets[f], last = mesh.face_offsets[f + 1];
    const Vec dc = sub(average, centroid);
    const double rc = norm(dc);
    const double tc = 1e-9 * rc * rc * rc;
    const Vec dv = sub(average, vertex_average);
    const double rv = norm(dv);
    const double tv = 1e-9 * rv * rv * rv;
    const int32_t* fv = mesh.face_vertices.data() + first;
    const int64_t nverts = last - first;
    Vec p = load(x + 3 * fv[0]);
    const Vec p0 = p;
    for (int64_t i = 0; i < nverts; ++i) {
      const Vec n = i + 1 < nverts ? load(x + 3 * fv[i + 1]) : p0;
      const Vec cr = cross(sub(p, average), sub(n, average));
      double vc = dot(cr, dc);
      double vv = dot(cr, dv);
      if (signed_face < 0) {
        vc = -vc;
        vv = -vv;
      }
      if (!(vc > tc) || !(vv > tv)) return false;
      p = n;
    }
  }
  return true;
}

// The screen on geometry computed afresh.
bool cell_valid_float(const Mesh& mesh, const Topology& t, int64_t c) {
  return cell_screen(mesh, t, c, cell_centroid(mesh, t, c), cell_vertex_average(mesh, t, c),
                     [&](int64_t f, bool& unfolded) {
                       unfolded = fan_unfolded(mesh, f);
                       const int64_t first = mesh.face_offsets[f], last = mesh.face_offsets[f + 1];
                       Vec average{0, 0, 0};
                       for (int64_t q = first; q < last; ++q) {
                         average = add(average, load(mesh.points.data() + 3 * mesh.face_vertices[q]));
                       }
                       return mul(average, 1.0 / static_cast<double>(last - first));
                     });
}

// Decomposition validity of a cell with a closed genus-0 boundary: unfolded
// face fans (with a margin) and exactly oriented decomposition tetrahedra
// about the OpenFOAM centroid and about the distinct-vertex average.
bool polyhedron_certified(const Mesh& mesh, const Topology& t, int64_t c) {
  const Vec centroid = cell_centroid(mesh, t, c), vertex_average = cell_vertex_average(mesh, t, c);
  const double cp[3] = {centroid[0], centroid[1], centroid[2]};
  const double vp[3] = {vertex_average[0], vertex_average[1], vertex_average[2]};
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
      // Outward triangle (average, u, w): the center lies on its inner side.
      if (orient3d(ap, mesh.points.data() + 3 * u, mesh.points.data() + 3 * w, cp) >= 0) return false;
      if (orient3d(ap, mesh.points.data() + 3 * u, mesh.points.data() + 3 * w, vp) >= 0) return false;
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

void Builder::place_split(int64_t row, const std::vector<std::pair<int32_t, Vec>>& by_patch) {
  Split& split = splits[split_of[row]];
  const int32_t layers = levels - 1;
  Vec n[2];
  for (int side = 0; side < 2; ++side) {
    const auto it = std::find_if(by_patch.begin(), by_patch.end(), [&](const auto& e) { return e.first == split.pair[side]; });
    n[side] = mul(it->second, 1.0 / norm(it->second));
  }
  // w[side]: the direction into that side's patch, away from the curve.
  Vec w[2];
  for (int side = 0; side < 2; ++side) {
    const Vec& other = n[1 - side];
    const Vec u = sub(other, mul(n[side], dot(other, n[side])));
    w[side] = mul(u, -1.0 / norm(u));
  }
  const Vec p = point(surface_vertex[row]);
  double extent = std::numeric_limits<double>::infinity();
  for (const int32_t q : row_pieces[row]) {
    const Piece& piece = pieces[q];
    const int side = piece.patch == split.pair[0] ? 0 : 1;
    double far = 0.0;
    for (int c = 0; c < piece.count; ++c) {
      const int32_t other = stair_vertex[piece.loop[c]];
      if (other == row) continue;
      far = std::max(far, dot(sub(point(surface_vertex[other]), p), w[side]));
    }
    extent = std::min(extent, far);
  }
  const double scale = std::min(1.0, 0.9 * extent / layer_total);
  double height = 0.0;
  for (int32_t k = 1; k <= layers; ++k) {
    height += scale * in.layer_thicknesses[k - 1];
    for (int side = 0; side < 2; ++side) {
      int32_t& item = split.item[side * layers + k - 1];
      const Vec target = add(p, mul(w[side], height));
      const Hit hit = item < 0 ? nearest_triangle(s, target, split.pair[side]) : walk_triangle(s, target, item);
      item = hit.item;
      set_point(split.grid[side == 0 ? k * (layers + 1) : k], hit.point);
    }
  }
  for (int32_t i = 1; i <= layers; ++i) {
    for (int32_t j = 1; j <= layers; ++j) {
      set_point(split.grid[i * (layers + 1) + j], sub(add(point(split.grid[i * (layers + 1)]), point(split.grid[j])), p));
    }
  }
}

// Midpoint grids: g(0, 0) on the feature curve nearest the chord midpoint,
// g(i, 0) and g(0, j) on their patches nearest the midpoints of the end
// grids' points, g(i, j) = g(i, 0) + g(0, j) - g(0, 0).
void Builder::place_mids() {
  const int32_t layers = levels - 1;
  for (Midpoint& m : mids) {
    const Split& first = splits[split_of[m.first_row]];
    const Split& second = splits[split_of[m.second_row]];
    auto middle = [&](int32_t q) { return mul(add(point(first.grid[q]), point(second.grid[q])), 0.5); };
    const Hit curve = walk_edge(s, middle(0), m.curve_item < 0 ? surface_item[m.first_row] : m.curve_item);
    m.curve_item = curve.item;
    set_point(m.grid[0], curve.point);
    for (int32_t k = 1; k <= layers; ++k) {
      for (int side = 0; side < 2; ++side) {
        const int32_t slot = side * layers + k - 1;
        const int32_t q = side == 0 ? k * (layers + 1) : k;
        const Hit hit = walk_triangle(s, middle(q), m.item[slot] < 0 ? first.item[slot] : m.item[slot]);
        m.item[slot] = hit.item;
        set_point(m.grid[q], hit.point);
      }
    }
    for (int32_t i = 1; i <= layers; ++i) {
      for (int32_t j = 1; j <= layers; ++j) {
        set_point(m.grid[i * (layers + 1) + j], sub(add(point(m.grid[i * (layers + 1)]), point(m.grid[j])), curve.point));
      }
    }
  }
}

// Layer points along the wall offset direction m of every column: with n_i
// the unit outward normals of the patches around the wall vertex (area-
// weighted over its wall faces, patches meeting below the feature angle
// merged), m solves m . n_i = 1 (least squares beyond three), so the k-th
// layer point lies at the cumulative thickness from every adjacent patch
// plane: m = n on a patch, (n1 + n2) / (1 + n1 . n2) on a feature curve.
// A split curve column instead slides its layer points along each patch:
// g(i, 0) = p - h_i w1 projected onto the first patch, w1 the second
// patch's normal without its first-normal part (unit), g(0, j) likewise,
// with h scaled to at most 0.9 of the least extent of the adjacent wall
// pieces away from the curve (cfMesh's limit).
int32_t Builder::place_layers() {
  if (levels == 1) return PHX_MC_OK;
  const int64_t columns = static_cast<int64_t>(stair_vertices.size());
  layer_direction.resize(columns);
  curve_column.resize(columns);
  std::vector<Vec> area(pieces.size());
  auto compute_piece_area = [&](std::size_t p) {
    Vec center, average;
    face_geometry(mesh, piece_wall_face[p], center, area[p], average);
  };
  if (in.parallelism == 1 || pieces.size() <= 512) {
    for (std::size_t p = 0; p < pieces.size(); ++p) compute_piece_area(p);
  } else {
    const int32_t st = parallel_for(
        pieces.size(), ParallelRequest{512, in.parallelism},
        [&](std::size_t begin, std::size_t end, ChunkContext&) -> int32_t {
          const GradualNearestScope mode;
          if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
          for (std::size_t p = begin; p < end; ++p) compute_piece_area(p);
          return PHX_MC_OK;
        });
    if (st != PHX_MC_OK) return st;
  }
  const double feature_cos = std::cos(in.feature_angle_degrees * (3.14159265358979323846 / 180.0));
  auto run_rows = [&](std::size_t begin, std::size_t end) {
    std::vector<std::pair<int32_t, Vec>> by_patch;
    std::vector<Vec> normals, sums;
    for (std::size_t r = begin; r < end; ++r) {
      const int64_t row = static_cast<int64_t>(r);
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
      if (split_of[row] >= 0) {
        place_split(row, by_patch);
        layer_direction[row] = {0, 0, 0};
        curve_column[row] = 0;
        continue;
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
        double a[3][3] = {}, rhs[3] = {};
        for (const Vec& n : normals) {
          for (int i = 0; i < 3; ++i) {
            rhs[i] += n[i];
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
              for (int y = 0; y < 3; ++y) q[x][y] = y == i ? rhs[x] : a[x][y];
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
  };
  const auto col_count = static_cast<std::size_t>(columns);
  if (in.parallelism == 1 || col_count <= 256) {
    run_rows(0, col_count);
  } else {
    const int32_t st = parallel_for(
        col_count, ParallelRequest{256, in.parallelism},
        [&](std::size_t begin, std::size_t end, ChunkContext&) -> int32_t {
          const GradualNearestScope mode;
          if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
          run_rows(begin, end);
          return PHX_MC_OK;
        });
    if (st != PHX_MC_OK) return st;
  }
  place_mids();
  return PHX_MC_OK;
}

// Small open-addressing int32 -> int32 slot map with O(1) generation clear.
struct SmallSlotMap {
  std::vector<int32_t> keys;
  std::vector<int32_t> vals;
  std::vector<uint32_t> stamps;
  uint32_t epoch = 1;
  uint32_t mask = 0;
  std::size_t size = 0;

  explicit SmallSlotMap(std::size_t initial_cap = 64) { reset_cap(initial_cap); }

  void clear() {
    ++epoch;
    if (epoch == 0) {
      std::fill(stamps.begin(), stamps.end(), 0);
      epoch = 1;
    }
    size = 0;
  }

  int32_t find(int32_t key) const {
    for (uint32_t slot = hash(key) & mask;; slot = (slot + 1) & mask) {
      if (stamps[slot] != epoch) return -1;
      if (keys[slot] == key) return vals[slot];
    }
  }

  int32_t insert(int32_t key, int32_t val) {
    if (2 * (size + 1) > keys.size()) grow();
    for (uint32_t slot = hash(key) & mask;; slot = (slot + 1) & mask) {
      if (stamps[slot] != epoch) {
        stamps[slot] = epoch;
        keys[slot] = key;
        vals[slot] = val;
        ++size;
        return val;
      }
      if (keys[slot] == key) return vals[slot];
    }
  }

 private:
  static uint32_t hash(int32_t key) { return static_cast<uint32_t>(key) * 2654435761u; }
  void reset_cap(std::size_t cap) {
    keys.assign(cap, 0);
    vals.assign(cap, -1);
    stamps.assign(cap, 0);
    mask = static_cast<uint32_t>(cap - 1);
    epoch = 1;
    size = 0;
  }
  void grow() {
    const std::size_t old_cap = keys.size();
    std::vector<int32_t> ok = std::move(keys);
    std::vector<int32_t> ov = std::move(vals);
    std::vector<uint32_t> os = std::move(stamps);
    const uint32_t oe = epoch;
    reset_cap(old_cap * 2);
    for (std::size_t i = 0; i < old_cap; ++i) {
      if (os[i] == oe) insert(ok[i], ov[i]);
    }
  }
};

struct RepairTask {
  int32_t v;
  int32_t f;
  int32_t layer_row;
  int32_t wall_row;
  int32_t level;
  int32_t movers_offset;
  int32_t movers_count;
  uint8_t layer;
  uint8_t tilt;
  uint8_t slide;
};

struct alignas(128) RepairScratch {
  SmallSlotMap cell_map{64};
  SmallSlotMap face_map{256};
  std::vector<int32_t> local_cells, local_faces, local_cell_faces;
  std::vector<std::size_t> local_cell_offsets;
  std::vector<Vec> local_centroid, local_center, local_area, local_average;
  std::vector<uint8_t> local_moving, local_unfolded;
  std::vector<uint8_t> local_face_done, local_cell_done;
  std::vector<uint32_t> moving_unfolded_trial;
  std::vector<uint8_t> moving_unfolded_val;
  std::vector<Vec> starts;
  std::vector<int32_t> sample_items;
  uint32_t trial_epoch = 0;
};

int32_t Builder::smooth() {
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
  std::vector<int32_t> interior_vertices;
  interior_vertices.reserve(static_cast<std::size_t>(points / 2));
  for (int64_t v = 0; v < points; ++v) {
    if (role[v] == 1) interior_vertices.push_back(static_cast<int32_t>(v));
  }
  // Boundary faces incident to every surface vertex.
  std::vector<std::vector<int32_t>> wall_faces(columns);
  for (int64_t f = core_faces; f < faces; ++f) {
    if (mesh.face_neighbor[f] >= 0) continue;
    for (int64_t q = mesh.face_offsets[f]; q < mesh.face_offsets[f + 1]; ++q) {
      // Split columns' slid wall points carry no column of their own.
      const int32_t row = column_of[mesh.face_vertices[q]];
      if (row >= 0) wall_faces[row].push_back(static_cast<int32_t>(f));
    }
  }
  // Patch vertices sharing a wall face with a split column's slid points.
  std::vector<uint8_t> near_split(columns, 0);
  for (int64_t row = 0; row < columns; ++row) {
    for (const int32_t f : wall_faces[row]) {
      for (int64_t q = mesh.face_offsets[f]; q < mesh.face_offsets[f + 1]; ++q) {
        near_split[row] = near_split[row] || column_of[mesh.face_vertices[q]] < 0;
      }
    }
  }
  // Patch columns the repair may slide: the ring around split columns.
  const std::vector<uint8_t>& slide_row = near_split;
  // Distinct vertices of every cell (strictly ascending vertex order).
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
  // Face centers, area vectors, vertex averages and unfolded fans, and cell
  // centroids: refreshed for the active faces and cells by geometry() and by
  // settle(), which leaves them current for the next pass.
  std::vector<Vec> centroid(cells), face_center(faces), face_area(faces), face_average(faces);
  std::vector<uint8_t> face_unfolded(faces, 0);
  auto face_update = [&](int64_t f) {
    face_geometry_and_unfolded(mesh, f, face_center[f], face_area[f], face_average[f], face_unfolded[f]);
  };
  auto cell_geometry = [&](int64_t c) {
    const int64_t first = t.cell_face_offsets[c], last = t.cell_face_offsets[c + 1];
    Vec estimate{0, 0, 0};
    for (int64_t slot = first; slot < last; ++slot) estimate = add(estimate, face_center[std::abs(t.cell_faces[slot]) - 1]);
    estimate = mul(estimate, 1.0 / static_cast<double>(last - first));
    double volume = 0.0;
    Vec moment{0, 0, 0};
    for (int64_t slot = first; slot < last; ++slot) {
      const int32_t signed_face = t.cell_faces[slot];
      const int64_t f = std::abs(signed_face) - 1;
      const double v = (signed_face > 0 ? 1.0 : -1.0) * dot(face_area[f], sub(face_center[f], estimate)) / 3.0;
      volume += v;
      moment = add(moment, mul(add(mul(face_center[f], 0.75), mul(estimate, 0.25)), v));
    }
    centroid[c] = volume > 0.0 ? mul(moment, 1.0 / volume) : estimate;
  };
  // All workers see the same coordinates. Complete the face bank before
  // updating cell centroids, with the original reduction order inside a row.
  const auto visit = [&](std::size_t count, std::size_t chunk, const auto& row) -> int32_t {
    if (in.parallelism == 1 || count <= chunk) {
      for (std::size_t i = 0; i < count; ++i) row(i);
      return PHX_MC_OK;
    }
    return parallel_for(count, ParallelRequest{chunk, in.parallelism},
                        [&](std::size_t begin, std::size_t end, ChunkContext&) -> int32_t {
      const GradualNearestScope mode;
      if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
      for (std::size_t i = begin; i < end; ++i) row(i);
      return PHX_MC_OK;
    });
  };
  int32_t status = visit(static_cast<std::size_t>(faces), 1024, face_update);
  if (status != PHX_MC_OK) return status;
  status = visit(static_cast<std::size_t>(cells), 1024, cell_geometry);
  if (status != PHX_MC_OK) return status;
  const auto refresh_geometry = [&](const std::vector<int32_t>& geom_faces,
                                    const std::vector<int32_t>& geom_cells) -> int32_t {
    const int32_t faces_status =
        visit(geom_faces.size(), 1024, [&](std::size_t i) { face_update(geom_faces[i]); });
    if (faces_status != PHX_MC_OK) return faces_status;
    return visit(geom_cells.size(), 1024, [&](std::size_t i) { cell_geometry(geom_cells[i]); });
  };
  // The vertex average of a cell's distinct vertices (the certified
  // decomposition's center) at the current positions.
  auto vertex_average = [&](int64_t c) {
    Vec average{0, 0, 0};
    for (int64_t slot = cell_vertex_offsets[c]; slot < cell_vertex_offsets[c + 1]; ++slot) {
      average = add(average, point(cell_vertex_list[slot]));
    }
    return mul(average, 1.0 / static_cast<double>(cell_vertex_offsets[c + 1] - cell_vertex_offsets[c]));
  };
  // The validity screen on the current face data.
  auto cell_valid_cached = [&](int64_t c) {
    return cell_screen(mesh, t, c, centroid[c], vertex_average(c), [&](int64_t f, bool& unfolded) {
      unfolded = face_unfolded[f] != 0;
      return face_average[f];
    });
  };
  // Fresh validity screen reusing cell_vertex_list and computing each incident
  // face's geometry and unfolded flag once per cell.
  auto cell_valid_fresh = [&](int64_t c) -> bool {
    const int64_t first = t.cell_face_offsets[c], last = t.cell_face_offsets[c + 1];
    if (last - first > 64) return cell_valid_float(mesh, t, c);
    if (mesh.cell_kinds[c] == kHexahedron && !hexahedron_corners_positive(mesh, c)) return false;
    const int64_t count = last - first;
    Vec centers[64], areas[64], averages[64];
    uint8_t unfolded[64];
    Vec estimate{0, 0, 0};
    for (int64_t slot = 0; slot < count; ++slot) {
      const int32_t signed_face = t.cell_faces[first + slot];
      face_geometry_and_unfolded(mesh, std::abs(signed_face) - 1, centers[slot], areas[slot],
                                 averages[slot], unfolded[slot]);
      if (signed_face < 0) areas[slot] = mul(areas[slot], -1.0);
      estimate = add(estimate, centers[slot]);
    }
    estimate = mul(estimate, 1.0 / static_cast<double>(count));
    double volume = 0.0;
    Vec moment{0, 0, 0};
    for (int64_t slot = 0; slot < count; ++slot) {
      const double v = dot(areas[slot], sub(centers[slot], estimate)) / 3.0;
      volume += v;
      moment = add(moment, mul(add(mul(centers[slot], 0.75), mul(estimate, 0.25)), v));
    }
    const Vec fresh_centroid = volume > 0.0 ? mul(moment, 1.0 / volume) : estimate;
    const Vec vavg = vertex_average(c);
    const double* x = mesh.points.data();
    for (int64_t slot = 0; slot < count; ++slot) {
      if (!unfolded[slot]) return false;
      const int32_t signed_face = t.cell_faces[first + slot];
      const int64_t f = std::abs(signed_face) - 1;
      const Vec& average = averages[slot];
      const Vec dc = sub(average, fresh_centroid);
      const double rc = norm(dc);
      const double tc = 1e-9 * rc * rc * rc;
      const Vec dv = sub(average, vavg);
      const double rv = norm(dv);
      const double tv = 1e-9 * rv * rv * rv;
      const int64_t ffirst = mesh.face_offsets[f], flast = mesh.face_offsets[f + 1];
      const int32_t* fv = mesh.face_vertices.data() + ffirst;
      const int64_t nverts = flast - ffirst;
      Vec p = load(x + 3 * fv[0]);
      const Vec p0 = p;
      for (int64_t i = 0; i < nverts; ++i) {
        const Vec n = i + 1 < nverts ? load(x + 3 * fv[i + 1]) : p0;
        const Vec cr = cross(sub(p, average), sub(n, average));
        double vc = dot(cr, dc);
        double vv = dot(cr, dv);
        if (signed_face < 0) {
          vc = -vc;
          vv = -vv;
        }
        if (!(vc > tc) || !(vv > tv)) return false;
        p = n;
      }
    }
    return true;
  };
  std::vector<double> previous(mesh.points);
  std::vector<uint8_t> moved(points, 0);
  int64_t reverted = 0;
  // Cells valid so far: smoothing never makes one of them invalid (only the
  // active cells change shape). The face bank and centroids were just computed
  // on the initial coordinates, so cached screening matches cell_valid_float.
  std::vector<uint8_t> valid(cells, 1);
  status = visit(active_cells.size(), 256, [&](std::size_t i) {
    const int32_t c = active_cells[i];
    valid[c] = (t.cell_face_offsets[c + 1] - t.cell_face_offsets[c] <= 64)
                   ? cell_valid_cached(c)
                   : cell_valid_float(mesh, t, c);
  });
  if (status != PHX_MC_OK) return status;
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
    const auto [stack, size] = carried(row);
    for (int32_t k = 0; k < size; ++k) column_row[stack[k]] = static_cast<int32_t>(row);
  }
  std::vector<int32_t> stamp(cells, -1);
  int32_t epoch = 0;
  std::vector<int32_t> face_stamp(faces, -1);
  int32_t face_epoch = 0;
  std::vector<int32_t> check, next, recheck_faces;
  // Precomputed initial check-cell lists when no column reverts during the
  // room check, preserving exact vertex-scan insertion order.
  std::vector<int32_t> s1_default_check, s2_default_check;
  {
    std::vector<uint8_t> s1_moves(points, 0);
    for (const int32_t v : interior_vertices) s1_moves[v] = 1;
    for (int64_t row = 0; row < columns; ++row) {
      const int32_t v = surface_vertex[row];
      if (role[v] != 2) continue;
      if (surface_kind[row] == kFeature && chain_neighbors[row][0] < 0) continue;
      s1_moves[v] = 1;
    }
    for (int64_t v = layer_first; v < points; ++v) s1_moves[v] = 1;
    ++epoch;
    for (int64_t v = 0; v < points; ++v) {
      if (!s1_moves[v]) continue;
      for (int64_t r = t.vertex_cell_offsets[v]; r < t.vertex_cell_offsets[v + 1]; ++r) {
        const int32_t c = t.vertex_cells[r];
        if (stamp[c] != epoch) {
          stamp[c] = epoch;
          s1_default_check.push_back(c);
        }
      }
    }
    ++epoch;
    for (const int32_t v : interior_vertices) {
      for (int64_t r = t.vertex_cell_offsets[v]; r < t.vertex_cell_offsets[v + 1]; ++r) {
        const int32_t c = t.vertex_cells[r];
        if (stamp[c] != epoch) {
          stamp[c] = epoch;
          s2_default_check.push_back(c);
        }
      }
    }
  }
  auto revert = [&](int32_t v) {
    const int32_t row = column_row[v];
    const auto [stack, size] = row >= 0 ? carried(row) : std::pair<const int32_t*, int32_t>{&v, 1};
    for (int32_t k = 0; k < size; ++k) {
      const int32_t u = stack[k];
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
  std::vector<uint8_t> cached_screens;
  const auto settle = [&](const std::vector<int32_t>& geom_faces,
                          const std::vector<int32_t>& geom_cells,
                          const std::vector<int32_t>& default_check) -> int32_t {
    check.clear();
    next.clear();
    ++epoch;
    for (int64_t row = 0; row < columns; ++row) {
      const double length = norm(sub(point(stair_vertices[row]), point(surface_vertex[row])));
      if (length < room(row)) {
        revert(stair_vertices[row]);
        revert(surface_vertex[row]);
      }
    }
    if (next.empty()) {
      next = default_check;
    } else {
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
    }
    // The first pass reads the geometry of the moved positions; a later pass
    // refreshes the cells it checks (those around reverted vertices). Within
    // a pass a cell around a vertex reverted earlier in it is read afresh.
    const int32_t geometry_status = refresh_geometry(geom_faces, geom_cells);
    if (geometry_status != PHX_MC_OK) return geometry_status;
    for (bool first = true; !next.empty(); first = false) {
      check.swap(next);
      next.clear();
      ++epoch;
      if (!first) {
        recheck_faces.clear();
        ++face_epoch;
        for (const int32_t c : check) {
          for (int64_t slot = t.cell_face_offsets[c]; slot < t.cell_face_offsets[c + 1]; ++slot) {
            const int32_t f = std::abs(t.cell_faces[slot]) - 1;
            if (face_stamp[f] != face_epoch) {
              face_stamp[f] = face_epoch;
              recheck_faces.push_back(f);
            }
          }
        }
        const int32_t rf_status =
            visit(recheck_faces.size(), 256, [&](std::size_t i) { face_update(recheck_faces[i]); });
        if (rf_status != PHX_MC_OK) return rf_status;
        const int32_t rc_status =
            visit(check.size(), 256, [&](std::size_t i) { cell_geometry(check[i]); });
        if (rc_status != PHX_MC_OK) return rc_status;
      }
      // Screen the immutable pass geometry before any rollback. If an
      // earlier rollback touches a cell, its epoch forces a fresh read
      // at exactly the same point in the ordered replay as before.
      const bool parallel_screen = in.parallelism != 1 && check.size() > 256;
      if (parallel_screen) {
        cached_screens.resize(check.size());
        const int32_t screened = visit(check.size(), 256, [&](std::size_t i) {
          cached_screens[i] = cell_valid_cached(check[i]);
        });
        if (screened != PHX_MC_OK) return screened;
      }
      for (std::size_t i = 0; i < check.size(); ++i) {
        const int32_t c = check[i];
        bool acceptable;
        if (stamp[c] == epoch) {
          acceptable = cell_valid_fresh(c);
        } else if (parallel_screen) {
          acceptable = cached_screens[i] != 0;
        } else {
          acceptable = cell_valid_cached(c);
        }
        if (acceptable) {
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
    return PHX_MC_OK;
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
    status = visit(interior_vertices.size(), 1024, [&](std::size_t idx) {
      const int32_t v = interior_vertices[idx];
      Vec target{0, 0, 0};
      const int64_t first = t.vertex_cell_offsets[v], last = t.vertex_cell_offsets[v + 1];
      for (int64_t slot = first; slot < last; ++slot) target = add(target, centroid[t.vertex_cells[slot]]);
      target = mul(target, 1.0 / static_cast<double>(last - first));
      const Vec p = load(previous.data() + 3 * v);
      set_point(v, add(p, mul(sub(target, p), kRelax)));
      moved[v] = 1;
    });
    if (status != PHX_MC_OK) return status;
    status = visit(static_cast<std::size_t>(columns), 256, [&](std::size_t r) {
      const int64_t row = static_cast<int64_t>(r);
      const int32_t v = surface_vertex[row];
      if (role[v] != 2) return;
      const Vec p = load(previous.data() + 3 * v);
      Vec target{0, 0, 0};
      if (surface_kind[row] == kFeature) {
        if (chain_neighbors[row][0] < 0) return;
        for (const int32_t other : chain_neighbors[row]) target = add(target, load(previous.data() + 3 * surface_vertex[other]));
        target = mul(target, 0.5);
      } else if (near_split[row]) {
        // Next to slid points: the wall-edge neighbours weighted by squared
        // edge length, pulling the vertex toward its long edges.
        double weight = 0.0;
        for (const int32_t f : wall_faces[row]) {
          const int64_t first = mesh.face_offsets[f], last = mesh.face_offsets[f + 1];
          for (int64_t q = first; q < last; ++q) {
            if (mesh.face_vertices[q] != v) continue;
            for (const int64_t rr : {q > first ? q - 1 : last - 1, q + 1 < last ? q + 1 : first}) {
              const Vec u = load(previous.data() + 3 * mesh.face_vertices[rr]);
              const Vec e = sub(u, p);
              const double w = dot(e, e);
              target = add(target, mul(u, w));
              weight += w;
            }
          }
        }
        target = mul(target, 1.0 / weight);
      } else {
        double weight = 0.0;
        for (const int32_t f : wall_faces[row]) {
          const double area = norm(face_area[f]);
          target = add(target, mul(face_center[f], area));
          weight += area;
        }
        target = mul(target, 1.0 / weight);
      }
      place_surface(row, add(p, mul(sub(target, p), kRelax)));
      moved[v] = 1;
    });
    if (status != PHX_MC_OK) return status;
    // The layer stack follows the wall: its points move with their columns'
    // offset directions, and a revert restores them with the wall.
    status = place_layers();
    if (status != PHX_MC_OK) return status;
    for (int64_t v = layer_first; v < points; ++v) moved[v] = 1;
    status = settle(active_faces, active_cells, s1_default_check);
    if (status != PHX_MC_OK) return status;
  }
  // 2. Descent on E = sum over interior faces of (1 - cos) of the angle
  // between the centroid join and the area vector, plus the skewness term;
  // cell centroids move with the mean of their vertices.
  // Only interior vertices move during descent. A face with no free vertex
  // in either incident cell cannot contribute to a used derivative, through
  // its own vertices or through those cells' centroids.
  std::vector<uint8_t> has_free_vertex(cells, 0);
  std::vector<int32_t> descent_cells;
  descent_cells.reserve(active_cells.size());
  for (const int32_t c : active_cells) {
    const auto first = cell_vertex_list.begin() + cell_vertex_offsets[c];
    const auto last = cell_vertex_list.begin() + cell_vertex_offsets[c + 1];
    if (std::any_of(first, last, [&](int32_t v) { return role[v] == 1; })) {
      has_free_vertex[c] = 1;
      descent_cells.push_back(c);
    }
  }
  std::vector<int32_t> descent_moving_faces, descent_eval_faces;
  descent_moving_faces.reserve(active_faces.size());
  descent_eval_faces.reserve(active_faces.size());
  std::vector<int32_t> descent_face_slot(faces, -1);
  for (const int32_t f : active_faces) {
    const int64_t first = mesh.face_offsets[f], last = mesh.face_offsets[f + 1];
    bool has_role1 = false;
    for (int64_t q = first; q < last && !has_role1; ++q) has_role1 = role[mesh.face_vertices[q]] == 1;
    if (has_role1) descent_moving_faces.push_back(f);
    const int32_t o = mesh.face_owner[f], n = mesh.face_neighbor[f];
    if (n >= 0 && (has_free_vertex[o] || has_free_vertex[n])) {
      descent_face_slot[f] = static_cast<int32_t>(descent_eval_faces.size());
      descent_eval_faces.push_back(f);
    }
  }
  // CSR mapping each interior vertex to its incident descent_eval_faces in
  // strictly increasing face order, enabling parallel bit-identical gathers.
  struct VertexFaceEntry {
    int32_t df_idx;
    int32_t before_v;
    int32_t after_v;
  };
  std::vector<int32_t> interior_index(points, -1);
  for (std::size_t idx = 0; idx < interior_vertices.size(); ++idx) {
    interior_index[interior_vertices[idx]] = static_cast<int32_t>(idx);
  }
  std::vector<int64_t> vdf_offsets(interior_vertices.size() + 1, 0);
  for (const int32_t f : descent_eval_faces) {
    for (int64_t q = mesh.face_offsets[f]; q < mesh.face_offsets[f + 1]; ++q) {
      const int32_t idx = interior_index[mesh.face_vertices[q]];
      if (idx >= 0) ++vdf_offsets[idx + 1];
    }
  }
  for (std::size_t i = 0; i < interior_vertices.size(); ++i) vdf_offsets[i + 1] += vdf_offsets[i];
  std::vector<VertexFaceEntry> vdf_entries(vdf_offsets.back());
  {
    std::vector<int64_t> fill(vdf_offsets.begin(), vdf_offsets.end() - 1);
    for (std::size_t df = 0; df < descent_eval_faces.size(); ++df) {
      const int32_t f = descent_eval_faces[df];
      const int64_t first = mesh.face_offsets[f], last = mesh.face_offsets[f + 1];
      for (int64_t q = first; q < last; ++q) {
        const int32_t idx = interior_index[mesh.face_vertices[q]];
        if (idx < 0) continue;
        const int64_t before = q > first ? q - 1 : last - 1;
        const int64_t after = q + 1 < last ? q + 1 : first;
        vdf_entries[fill[idx]++] = {static_cast<int32_t>(df), mesh.face_vertices[before],
                                    mesh.face_vertices[after]};
      }
    }
  }
  struct FaceGrad {
    Vec gd;
    Vec gs;
    Vec skew_n;
    Vec skew_o;
    Vec face_share;
    uint8_t valid;
    uint8_t has_skew;
  };
  std::vector<FaceGrad> face_grads(descent_eval_faces.size());
  std::vector<Vec> cell_share(cells);
  std::vector<double> trust(points, kTrust * b);
  previous = mesh.points;
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    // The geometry is current (settle()).
    // 2A. Per-face gradient contributions in parallel.
    status = visit(descent_eval_faces.size(), 512, [&](std::size_t df) {
      FaceGrad& fg = face_grads[df];
      const int64_t f = descent_eval_faces[df];
      const int32_t o = mesh.face_owner[f], n = mesh.face_neighbor[f];
      const Vec d = sub(centroid[n], centroid[o]);
      const Vec& area = face_area[f];
      const double ld = norm(d), ls = norm(area);
      if (!(ld > 0.0) || !(ls > 0.0)) {
        fg.valid = 0;
        fg.has_skew = 0;
        return;
      }
      const Vec dh = mul(d, 1.0 / ld), sh = mul(area, 1.0 / ls);
      const double c = dot(dh, sh);
      const double weight = -1.0;  // dE/dc
      fg.gd = mul(sub(sh, mul(dh, c)), weight / ld);
      fg.gs = mul(sub(dh, mul(sh, c)), weight / ls);
      fg.valid = 1;
      // Skewness s = |sv| / fd, sv = Cf - (Co + alpha d) the offset of the
      // face center from where the centroid join meets the face plane
      // (OpenFOAM's measure); E += w (s^2 + mu s^4) with the face normal
      // and fd frozen, the face center moving with its vertex mean.
      const Vec cpf = sub(face_center[f], centroid[o]);
      const double nd = dot(sh, d);
      if (!(std::fabs(nd) > 0.0)) {
        fg.has_skew = 0;
        return;
      }
      const int64_t first = mesh.face_offsets[f], last = mesh.face_offsets[f + 1];
      const double alpha = dot(sh, cpf) / nd;
      const Vec sv = sub(cpf, mul(d, alpha));
      const double length = norm(sv);
      double fd = 0.2 * ld;
      if (length > 0.0) {
        const Vec unit = mul(sv, 1.0 / length);
        for (int64_t q = first; q < last; ++q) {
          fd = std::max(fd, std::fabs(dot(unit, sub(point(mesh.face_vertices[q]), face_center[f]))));
        }
      }
      const double skew = length / fd;
      const double scale = kSkewWeight * (1.0 + 2.0 * kSkewEmphasis * skew * skew) * 2.0 / (fd * fd);
      const Vec w = mul(sub(sv, mul(sh, dot(sv, d) / nd)), scale);
      fg.face_share = mul(w, 1.0 / static_cast<double>(last - first));
      fg.skew_n = mul(w, alpha);
      fg.skew_o = mul(w, 1.0 - alpha);
      fg.has_skew = 1;
    });
    if (status != PHX_MC_OK) return status;
    // 2B. Gather cell gradients in exact ascending face order (t.cell_faces).
    status = visit(descent_cells.size(), 512, [&](std::size_t idx) {
      const int32_t c = descent_cells[idx];
      Vec cg{0, 0, 0};
      for (int64_t slot = t.cell_face_offsets[c]; slot < t.cell_face_offsets[c + 1]; ++slot) {
        const int32_t signed_face = t.cell_faces[slot];
        const int32_t df = descent_face_slot[std::abs(signed_face) - 1];
        if (df < 0) continue;
        const FaceGrad& fg = face_grads[df];
        if (!fg.valid) continue;
        if (signed_face > 0) {
          cg = sub(cg, fg.gd);
          if (fg.has_skew) cg = sub(cg, fg.skew_o);
        } else {
          cg = add(cg, fg.gd);
          if (fg.has_skew) cg = sub(cg, fg.skew_n);
        }
      }
      const int64_t count = cell_vertex_offsets[c + 1] - cell_vertex_offsets[c];
      cell_share[c] = mul(cg, 1.0 / static_cast<double>(count));
    });
    if (status != PHX_MC_OK) return status;
    // 2C. Gather vertex gradients in exact ascending face then cell order and
    // apply the trust-capped descent step in parallel.
    if (iteration > 0) {
      status = visit(interior_vertices.size(), 1024, [&](std::size_t idx) {
        const int32_t v = interior_vertices[idx];
        for (int a = 0; a < 3; ++a) previous[3 * v + a] = mesh.points[3 * v + a];
      });
      if (status != PHX_MC_OK) return status;
    }
    std::fill(moved.begin(), moved.end(), 0);
    status = visit(interior_vertices.size(), 512, [&](std::size_t idx) {
      const int32_t v = interior_vertices[idx];
      Vec g{0, 0, 0};
      for (int64_t slot = vdf_offsets[idx]; slot < vdf_offsets[idx + 1]; ++slot) {
        const VertexFaceEntry& e = vdf_entries[slot];
        const FaceGrad& fg = face_grads[e.df_idx];
        if (!fg.valid) continue;
        const Vec diff = sub(load(previous.data() + 3 * e.before_v),
                             load(previous.data() + 3 * e.after_v));
        g = add(g, mul(cross(fg.gs, diff), 0.5));
        if (fg.has_skew) g = add(g, fg.face_share);
      }
      for (int64_t slot = t.vertex_cell_offsets[v]; slot < t.vertex_cell_offsets[v + 1]; ++slot) {
        const int32_t c = t.vertex_cells[slot];
        if (has_free_vertex[c]) g = add(g, cell_share[c]);
      }
      Vec delta = mul(g, -kStep * b * b);
      const double length = norm(delta);
      if (length > trust[v]) delta = mul(delta, trust[v] / length);
      set_point(v, add(load(previous.data() + 3 * v), delta));
      moved[v] = 1;
    });
    if (status != PHX_MC_OK) return status;
    status = settle(descent_moving_faces, descent_cells, s2_default_check);
    if (status != PHX_MC_OK) return status;
    for (const int32_t v : interior_vertices) {
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
  auto face_badness = [&](int64_t f, const Vec& center, const Vec& area, const Vec& co, const Vec* cn,
                          double floor_cosine, double* scaled_skewness = nullptr) {
    const double ls = norm(area);
    if (!(ls > 0.0)) return std::numeric_limits<double>::infinity();
    const Vec cpf = sub(center, co);
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
        fd = std::max(fd, std::fabs(dot(unit, sub(point(mesh.face_vertices[q]), center))));
      }
    }
    const double skewness = length / fd / kSkewGoal;
    if (scaled_skewness != nullptr) *scaled_skewness = skewness;
    return std::max(skewness, angle / kAngleGoal);
  };
  int64_t repaired = 0;
  std::vector<double> bad_values(active_faces.size());
  std::vector<uint8_t> bad_skewed(active_faces.size());
  std::vector<RepairTask> tasks;
  std::vector<int32_t> task_movers;
  std::vector<int64_t> task_repaired;
  // Exact order-preserving wave-scheduler tables (reused across rounds via wave_base).
  std::vector<int32_t> last_w_pt, last_r_pt, last_w_cell, last_r_cell, last_w_face, last_r_face;
  std::vector<int32_t> sched_cell_stamp, sched_face_stamp, sched_pt_stamp;
  int32_t wave_base = 0;
  int32_t sched_epoch = 0;
  std::vector<int32_t> sched_w_cells, sched_r_cells, sched_r_faces, sched_w_faces, sched_r_pts;
  std::vector<int32_t> task_waves;
  std::vector<std::vector<int32_t>> wave_buckets;
  if (in.parallelism > 1) {
    last_w_pt.assign(points, -1);
    last_r_pt.assign(points, -1);
    last_w_cell.assign(cells, -1);
    last_r_cell.assign(cells, -1);
    last_w_face.assign(faces, -1);
    last_r_face.assign(faces, -1);
    sched_cell_stamp.assign(cells, 0);
    sched_face_stamp.assign(faces, 0);
    sched_pt_stamp.assign(points, 0);
  }
  const auto run_repair_task = [&](const RepairTask& task, RepairScratch& sc) -> int64_t {
    const int32_t v = task.v;
    const int32_t f = task.f;
    const int32_t layer_row = task.layer_row;
    const int32_t wall_row = task.wall_row;
    const int32_t level = task.level;
    const bool layer = task.layer != 0;
    const bool tilt = task.tilt != 0;
    const bool slide = task.slide != 0;
    const int32_t* movers_ptr = task_movers.data() + task.movers_offset;
    const std::size_t movers_count = static_cast<std::size_t>(task.movers_count);
    const int32_t neighbor = mesh.face_neighbor[f];
    // Largest distance from the patch of the sliding vertex's wall
    // faces, sampled at their edge midpoints and vertex averages (the
    // fan triangles' far points); each sample's walk starts from its
    // last foot.
    sc.sample_items.clear();
    auto wall_deviation = [&]() {
      double deviation = 0.0;
      std::size_t next_s = 0;
      auto sample = [&](const Vec& x) {
        if (next_s == sc.sample_items.size()) sc.sample_items.push_back(surface_item[wall_row]);
        const Hit hit = walk_triangle(s, x, sc.sample_items[next_s]);
        sc.sample_items[next_s++] = hit.item;
        deviation = std::max(deviation, hit.d2);
      };
      for (const int32_t g : wall_faces[wall_row]) {
        const int64_t first = mesh.face_offsets[g], last = mesh.face_offsets[g + 1];
        Vec average{0, 0, 0};
        for (int64_t q = first; q < last; ++q) average = add(average, point(mesh.face_vertices[q]));
        average = mul(average, 1.0 / static_cast<double>(last - first));
        sample(average);
        for (int64_t q = first; q < last; ++q) {
          const Vec a = point(mesh.face_vertices[q]), c = point(mesh.face_vertices[q + 1 < last ? q + 1 : first]);
          sample(mul(add(a, c), 0.5));
        }
      }
      return deviation;
    };
    const double deviation_bound = slide ? wall_deviation() : 0.0;
    sc.cell_map.clear();
    sc.face_map.clear();
    sc.local_cells.clear();
    for (std::size_t m = 0; m < movers_count; ++m) {
      const int32_t u = movers_ptr[m];
      for (int64_t r = t.vertex_cell_offsets[u]; r < t.vertex_cell_offsets[u + 1]; ++r) {
        const int32_t k = t.vertex_cells[r];
        if (sc.cell_map.find(k) >= 0) continue;
        sc.cell_map.insert(k, static_cast<int32_t>(sc.local_cells.size()));
        sc.local_cells.push_back(k);
      }
    }
    sc.local_faces.clear();
    sc.local_cell_faces.clear();
    sc.local_cell_offsets.assign(1, 0);
    for (const int32_t k : sc.local_cells) {
      for (int64_t s2 = t.cell_face_offsets[k]; s2 < t.cell_face_offsets[k + 1]; ++s2) {
        const int32_t signed_face = t.cell_faces[s2];
        const int32_t g = std::abs(signed_face) - 1;
        int32_t fslot = sc.face_map.find(g);
        if (fslot < 0) {
          fslot = static_cast<int32_t>(sc.local_faces.size());
          sc.face_map.insert(g, fslot);
          sc.local_faces.push_back(g);
        }
        const int32_t local = fslot + 1;
        sc.local_cell_faces.push_back(signed_face > 0 ? local : -local);
      }
      sc.local_cell_offsets.push_back(sc.local_cell_faces.size());
    }
    const std::size_t nf = sc.local_faces.size(), nc = sc.local_cells.size();
    sc.local_center.resize(nf);
    sc.local_area.resize(nf);
    sc.local_average.resize(nf);
    sc.local_moving.assign(nf, 0);
    sc.local_unfolded.assign(nf, 2);
    sc.local_face_done.resize(nf);
    sc.local_cell_done.resize(nc);
    sc.local_centroid.resize(nc);
    if (sc.moving_unfolded_trial.size() < nf) {
      sc.moving_unfolded_trial.assign(nf, 0);
      sc.moving_unfolded_val.assign(nf, 0);
    }
    for (std::size_t q = 0; q < nf; ++q) {
      const int32_t g = sc.local_faces[q];
      for (int64_t e = mesh.face_offsets[g]; e < mesh.face_offsets[g + 1] && !sc.local_moving[q]; ++e) {
        const int32_t fv = mesh.face_vertices[e];
        for (std::size_t m = 0; m < movers_count; ++m) {
          if (movers_ptr[m] == fv) {
            sc.local_moving[q] = 1;
            break;
          }
        }
      }
      if (!sc.local_moving[q]) {
        sc.local_center[q] = face_center[g];
        sc.local_area[q] = face_area[g];
        sc.local_average[q] = face_average[g];
      }
    }
    auto ensure_face = [&](std::size_t q) {
      if (sc.local_moving[q] && !sc.local_face_done[q]) {
        face_geometry(mesh, sc.local_faces[q], sc.local_center[q], sc.local_area[q], sc.local_average[q]);
        sc.local_face_done[q] = 1;
      }
    };
    auto ensure_cell = [&](std::size_t slot_c) {
      if (sc.local_cell_done[slot_c]) return;
      const std::size_t first = sc.local_cell_offsets[slot_c], last = sc.local_cell_offsets[slot_c + 1];
      Vec estimate{0, 0, 0};
      for (std::size_t s2 = first; s2 < last; ++s2) {
        const std::size_t fq = static_cast<std::size_t>(std::abs(sc.local_cell_faces[s2]) - 1);
        ensure_face(fq);
        estimate = add(estimate, sc.local_center[fq]);
      }
      estimate = mul(estimate, 1.0 / static_cast<double>(last - first));
      double volume = 0.0;
      Vec moment{0, 0, 0};
      for (std::size_t s2 = first; s2 < last; ++s2) {
        const int32_t signed_face = sc.local_cell_faces[s2];
        const std::size_t fq = static_cast<std::size_t>(std::abs(signed_face) - 1);
        const double w = (signed_face > 0 ? 1.0 : -1.0) *
                         dot(sc.local_area[fq], sub(sc.local_center[fq], estimate)) / 3.0;
        volume += w;
        moment = add(moment, mul(add(mul(sc.local_center[fq], 0.75), mul(estimate, 0.25)), w));
      }
      sc.local_centroid[slot_c] = volume > 0.0 ? mul(moment, 1.0 / volume) : estimate;
      sc.local_cell_done[slot_c] = 1;
    };
    auto ensure_all = [&]() {
      for (std::size_t q = 0; q < nf; ++q) ensure_face(q);
      for (std::size_t q = 0; q < nc; ++q) ensure_cell(q);
    };
    auto centroid_of = [&](int32_t k) -> const Vec& {
      const int32_t slot_q = sc.cell_map.find(k);
      return slot_q >= 0 ? sc.local_centroid[slot_q] : centroid[k];
    };
    // The largest badness of the local faces, from the hardest face of
    // the last full evaluation on; it stops once it reaches bound (a
    // trial no better than the current position). Local face geometries and
    // centroids are computed lazily on demand and skipped on early exit.
    std::size_t hardest = 0;
    auto evaluate = [&](double bound) {
      std::fill(sc.local_face_done.begin(), sc.local_face_done.end(), 0);
      std::fill(sc.local_cell_done.begin(), sc.local_cell_done.end(), 0);
      const std::size_t start = hardest;
      double worst = 0.0, worst_cosine = 1.0;
      for (std::size_t r = 0; r < nf && worst < bound; ++r) {
        const std::size_t q = r == 0 ? start : r == start ? 0 : r;
        const int32_t g = sc.local_faces[q];
        ensure_face(q);
        const int32_t ow = mesh.face_owner[g];
        const int32_t ow_slot = sc.cell_map.find(ow);
        if (ow_slot >= 0) ensure_cell(static_cast<std::size_t>(ow_slot));
        const Vec& co = ow_slot >= 0 ? sc.local_centroid[ow_slot] : centroid[ow];
        const int32_t nb = mesh.face_neighbor[g];
        const Vec* cn = nullptr;
        if (nb >= 0) {
          const int32_t nb_slot = sc.cell_map.find(nb);
          if (nb_slot >= 0) ensure_cell(static_cast<std::size_t>(nb_slot));
          cn = nb_slot >= 0 ? &sc.local_centroid[nb_slot] : &centroid[nb];
        }
        const double value = face_badness(g, sc.local_center[q], sc.local_area[q], co, cn, worst_cosine);
        if (value > worst) {
          worst = value;
          worst_cosine = std::cos(std::min(worst * kAngleGoal, 180.0) * (3.14159265358979323846 / 180.0));
          hardest = q;
        }
      }
      return worst;
    };
    double current = evaluate(std::numeric_limits<double>::infinity());
    ensure_all();
    bool geom_valid = true;
    // A slide never raises the largest skewness among its local faces
    // (the skewness part of the badness, on the geometry evaluate()
    // left): it trades only non-orthogonality.
    auto local_skew = [&]() {
      double worst = 0.0;
      for (std::size_t q = 0; q < nf; ++q) {
        const int32_t g = sc.local_faces[q];
        const int32_t n = mesh.face_neighbor[g];
        worst = std::max(worst, face_badness(g, sc.local_center[q], sc.local_area[q],
                                             centroid_of(mesh.face_owner[g]),
                                             n >= 0 ? &centroid_of(n) : nullptr, -1.0));
      }
      return worst;
    };
    double skew_bound = slide || tilt ? local_skew() : 0.0, trial_skew = 0.0;
    const int32_t stair = v < core_points ? stair_vertex[v] : -1;
    Vec axis{0, 0, 0}, wall{0, 0, 0};
    double below = 0.0;  // height of the layer point below v
    if (layer) {
      const Vec& m = layer_direction[layer_row];
      axis = mul(m, 1.0 / norm(m));
      wall = point(surface_vertex[layer_row]);
      below = dot(sub(wall, point(column[layer_row * (levels + 1) + level - 1])), m) / dot(m, m);
    }
    std::array<Vec, 2> tilt_directions{};
    if (tilt) {
      wall = point(surface_vertex[layer_row]);
      const Vec column_axis = sub(point(v), wall);
      const double depth = norm(column_axis);
      const auto q = static_cast<std::size_t>(sc.face_map.find(f));
      const Vec& co = centroid_of(mesh.face_owner[f]);
      const Vec join = sub(centroid_of(neighbor), co);
      const Vec offset = sub(sc.local_center[q], co);
      const Vec& area = sc.local_area[q];
      const double normal_join = dot(area, join);
      if (depth > 0.0 && normal_join != 0.0) {
        const Vec unit_axis = mul(column_axis, 1.0 / depth);
        const Vec skew = sub(offset, mul(join, dot(area, offset) / normal_join));
        const Vec tangent = sub(skew, mul(unit_axis, dot(skew, unit_axis)));
        if (const double size = norm(tangent); size > 0.0) {
          tilt_directions[0] = mul(tangent, 1.0 / size);
          tilt_directions[1] = cross(unit_axis, tilt_directions[0]);
        }
      }
    }
    int64_t local_repaired = 0;
    for (const double scale : {0.08, 0.03, 0.01}) {
      for (bool improved = true; improved && (!tilt || current > 1.0);) {
        improved = false;
        const int directions = layer ? 2 : tilt ? 4 : 6;
        for (int direction = 0; direction < directions && (!tilt || current > 1.0); ++direction) {
          Vec delta{0, 0, 0};
          if (layer) {
            delta = mul(axis, (direction & 1 ? -1.0 : 1.0) * scale * b);
          } else if (tilt) {
            delta = mul(tilt_directions[direction / 2], (direction & 1 ? -1.0 : 1.0) * scale * b);
          } else {
            delta[direction / 2] = (direction & 1 ? -1.0 : 1.0) * scale * b;
          }
          sc.starts.clear();
          if (tilt) {
            const Vec column_axis = sub(point(v), wall);
            const double depth = norm(column_axis);
            if (!(depth > 0.0)) continue;
            const Vec longitudinal = mul(column_axis, 1.0 / depth);
            const Vec lateral = sub(delta, mul(longitudinal, dot(delta, longitudinal)));
            const double lateral_size = norm(lateral);
            if (!(lateral_size > 0.0)) continue;
            const Vec tangent = mul(lateral, 1.0 / lateral_size);
            const double slope = lateral_size / norm(sub(point(movers_ptr[movers_count - 1]), wall));
            const double cosine = 1.0 / std::sqrt(1.0 + slope * slope);
            const double sine = slope * cosine;
            for (std::size_t m = 0; m < movers_count; ++m) {
              const int32_t u = movers_ptr[m];
              sc.starts.push_back(point(u));
              const Vec from_wall = sub(sc.starts.back(), wall);
              const double x = dot(from_wall, longitudinal);
              const double y = dot(from_wall, tangent);
              // Apply one proper rotation to the entire stack.
              const Vec change = add(mul(longitudinal, (cosine - 1.0) * x - sine * y),
                                     mul(tangent, sine * x + (cosine - 1.0) * y));
              set_point(u, add(sc.starts.back(), change));
            }
          } else {
            for (std::size_t m = 0; m < movers_count; ++m) {
              const int32_t u = movers_ptr[m];
              sc.starts.push_back(point(u));
              set_point(u, add(point(u), delta));
            }
          }
          Hit slid;
          if (slide) {
            const Vec w = point(surface_vertex[wall_row]);
            slid = walk_triangle(s, w, surface_item[wall_row]);
            const Vec back = sub(slid.point, w);
            for (std::size_t m = 0; m < movers_count; ++m) {
              const int32_t u = movers_ptr[m];
              set_point(u, add(point(u), back));
            }
          }
          const double value = evaluate(current);
          bool accept = value < current;
          if (accept && slide) {
            accept = norm(sub(point(stair_vertices[wall_row]), point(surface_vertex[wall_row]))) >= room(wall_row);
            if (accept) {
              trial_skew = local_skew();
              accept = trial_skew <= skew_bound;
            }
          }
          if (accept && tilt) {
            trial_skew = local_skew();
            accept = trial_skew <= skew_bound;
            // A tilted stack keeps every layer cell above its floor.
            for (std::size_t q = 0; q < nc && accept; ++q) accept = keeps_layer_floor(sc.local_cells[q]);
          }
          // The layer keeps its thickness.
          if (accept && layer) {
            const Vec& m = layer_direction[layer_row];
            accept = dot(sub(wall, point(v)), m) / dot(m, m) - below >= in.layer_thicknesses[level - 1];
          }
          // The staircase vertex keeps room for the layer stack.
          if (accept && stair >= 0) {
            accept = norm(sub(point(v), point(surface_vertex[stair]))) >= room(stair);
          }
          // The validity screen on the geometry evaluate() left: the
          // trial's faces and centroids.
          if (accept) {
            ++sc.trial_epoch;
            if (sc.trial_epoch == 0) {
              std::fill(sc.moving_unfolded_trial.begin(), sc.moving_unfolded_trial.end(), 0);
              sc.trial_epoch = 1;
            }
            for (std::size_t q = 0; q < nc && accept; ++q) {
              if (!valid[sc.local_cells[q]]) continue;
              accept = cell_screen(mesh, t, sc.local_cells[q], sc.local_centroid[q],
                                   vertex_average(sc.local_cells[q]),
                                   [&](int64_t g, bool& unfolded) {
                const int32_t slot_q = sc.face_map.find(static_cast<int32_t>(g));
                if (sc.local_moving[slot_q]) {
                  if (sc.moving_unfolded_trial[slot_q] != sc.trial_epoch) {
                    sc.moving_unfolded_trial[slot_q] = sc.trial_epoch;
                    sc.moving_unfolded_val[slot_q] = fan_unfolded(mesh, g);
                  }
                  unfolded = sc.moving_unfolded_val[slot_q] != 0;
                } else {
                  if (sc.local_unfolded[slot_q] == 2) sc.local_unfolded[slot_q] = fan_unfolded(mesh, g);
                  unfolded = sc.local_unfolded[slot_q] != 0;
                }
                return sc.local_average[slot_q];
              });
            }
          }
          if (accept && slide) accept = wall_deviation() <= deviation_bound;
          if (accept) {
            current = value;
            improved = true;
            ++local_repaired;
            geom_valid = true;
            if (tilt) skew_bound = trial_skew;
            if (slide) {
              surface_item[wall_row] = slid.item;
              skew_bound = trial_skew;
            }
          } else {
            for (std::size_t q = 0; q < movers_count; ++q) set_point(movers_ptr[q], sc.starts[q]);
            geom_valid = false;
          }
        }
      }
    }
    if (!geom_valid) {
      std::fill(sc.local_face_done.begin(), sc.local_face_done.end(), 0);
      std::fill(sc.local_cell_done.begin(), sc.local_cell_done.end(), 0);
      ensure_all();
    }
    for (std::size_t q = 0; q < nc; ++q) {
      centroid[sc.local_cells[q]] = sc.local_centroid[q];
    }
    // Publish only the restored or accepted coordinates' face data.
    for (std::size_t q = 0; q < nf; ++q) {
      if (sc.local_moving[q]) {
        const int32_t g = sc.local_faces[q];
        face_center[g] = sc.local_center[q];
        face_area[g] = sc.local_area[q];
        face_average[g] = sc.local_average[q];
      }
    }
    return local_repaired;
  };
  RepairScratch single_scratch;
  const std::size_t max_wave_chunks = 4 * static_cast<std::size_t>(in.parallelism);
  std::vector<RepairScratch> wave_scratches(in.parallelism > 1 ? max_wave_chunks : 0);
  for (int round = 0; round < kRepairRounds; ++round) {
    status = refresh_geometry(active_faces, active_cells);
    if (status != PHX_MC_OK) return status;
    std::vector<std::tuple<double, int32_t, bool>> bad;
    const double goal_cosine = std::cos(kAngleGoal * (3.14159265358979323846 / 180.0));
    status = visit(active_faces.size(), 512, [&](std::size_t i) {
      const int64_t f = active_faces[i];
      const int32_t o = mesh.face_owner[f], n = mesh.face_neighbor[f];
      double skewness = 0.0;
      bad_values[i] = face_badness(f, face_center[f], face_area[f], centroid[o],
                                   n >= 0 ? &centroid[n] : nullptr, goal_cosine, &skewness);
      bad_skewed[i] = skewness > 1.0 ? 1 : 0;
    });
    if (status != PHX_MC_OK) return status;
    for (std::size_t i = 0; i < active_faces.size(); ++i) {
      if (bad_values[i] > 1.0) bad.emplace_back(-bad_values[i], active_faces[i], bad_skewed[i] != 0);
    }
    if (bad.empty()) break;
    std::sort(bad.begin(), bad.end());
    std::vector<uint8_t> done(points, 0);
    tasks.clear();
    task_movers.clear();
    for (const auto& [value, f, skewed] : bad) {
      (void)value;
      const auto face_begin = mesh.face_vertices.begin() + mesh.face_offsets[f];
      const auto face_end = mesh.face_vertices.begin() + mesh.face_offsets[f + 1];
      const bool feature_wall = in.layer_count == 0 && skewed && mesh.face_neighbor[f] < 0 &&
                                std::any_of(face_begin, face_end, [&](int32_t vertex) {
        const int32_t row = column_of[vertex];
        return row >= 0 && surface_kind[row] != kPatch;
      });
      const int32_t neighbor = mesh.face_neighbor[f];
      const bool skewed_layer_side = skewed && neighbor >= 0 &&
          mesh.cell_roles[mesh.face_owner[f]] == kLayer && mesh.cell_roles[neighbor] == kLayer;
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
          const bool tilt = skewed_layer_side && layer_row >= 0 && !layer &&
              surface_kind[layer_row] == kPatch && split_of[layer_row] < 0 &&
              !slide_row[layer_row] &&
              column[layer_row * (levels + 1) + 1] == v &&
              std::find(face_begin, face_end, v) != face_end &&
              std::find(face_begin, face_end, surface_vertex[layer_row]) != face_end;
          // A patch vertex slides with its stack near a split column.
          // On an unlayered wall it can also repair a skewed face next
          // to a fixed feature without moving the feature itself.
          const int32_t wall_row = role[v] >= 2 ? column_row[v] : -1;
          const bool slide = wall_row >= 0 && surface_kind[wall_row] == kPatch &&
                             (slide_row[wall_row] ||
                              (feature_wall && role[v] == 2 &&
                               std::find(face_begin, face_end, v) != face_end));
          if ((role[v] != 1 && !layer && !slide && !tilt) || done[v]) continue;
          done[v] = 1;
          int32_t level = 0;
          if (layer) {
            const int32_t* stack = column.data() + layer_row * (levels + 1);
            while (stack[level] != v) ++level;
          }
          const int32_t movers_offset = static_cast<int32_t>(task_movers.size());
          if (slide) {
            if (tilt) {
              const auto [tstack, tsize] = carried(layer_row);
              for (int32_t u = 1; u < tsize; ++u) done[tstack[u]] = 1;
            }
            const auto [stack, size] = carried(wall_row);
            task_movers.insert(task_movers.end(), stack, stack + size);
            for (int32_t u = 0; u < size; ++u) done[stack[u]] = 1;
          } else if (tilt) {
            const auto [stack, size] = carried(layer_row);
            task_movers.insert(task_movers.end(), stack + 1, stack + size);
            for (int32_t u = 1; u < size; ++u) done[stack[u]] = 1;
          } else if (layer) {
            task_movers.push_back(v);
            const int32_t* stack = column.data() + layer_row * (levels + 1);
            for (int32_t k = level + 1; k < levels; ++k) task_movers.push_back(stack[k]);
          } else {
            task_movers.push_back(v);
          }
          const int32_t movers_count = static_cast<int32_t>(task_movers.size()) - movers_offset;
          tasks.push_back({v, f, layer_row, wall_row, level, movers_offset, movers_count,
                           static_cast<uint8_t>(layer), static_cast<uint8_t>(tilt),
                           static_cast<uint8_t>(slide)});
        }
      }
    }
    if (in.parallelism == 1 || tasks.size() <= 1) {
      for (const RepairTask& task : tasks) repaired += run_repair_task(task, single_scratch);
      continue;
    }
    // Build an exact dependency-preserving wave schedule over tasks: every read
    // observes the exact write produced by sequential execution order.
    task_waves.resize(tasks.size());
    int32_t max_wave = 0;
    for (std::size_t ti = 0; ti < tasks.size(); ++ti) {
      const RepairTask& task = tasks[ti];
      const int32_t* movers_ptr = task_movers.data() + task.movers_offset;
      const int32_t movers_count = task.movers_count;
      ++sched_epoch;
      const int32_t ep = sched_epoch;
      sched_w_cells.clear();
      sched_r_cells.clear();
      sched_r_faces.clear();
      sched_w_faces.clear();
      sched_r_pts.clear();
      auto add_r_pt = [&](int32_t u) {
        if (sched_pt_stamp[u] != ep) {
          sched_pt_stamp[u] = ep;
          sched_r_pts.push_back(u);
        }
      };
      for (int32_t m = 0; m < movers_count; ++m) {
        const int32_t u = movers_ptr[m];
        add_r_pt(u);
        for (int64_t r = t.vertex_cell_offsets[u]; r < t.vertex_cell_offsets[u + 1]; ++r) {
          const int32_t k = t.vertex_cells[r];
          if (sched_cell_stamp[k] != ep) {
            sched_cell_stamp[k] = ep;
            sched_w_cells.push_back(k);
            sched_r_cells.push_back(k);
          }
        }
      }
      for (const int32_t k : sched_w_cells) {
        for (int64_t slot = cell_vertex_offsets[k]; slot < cell_vertex_offsets[k + 1]; ++slot) {
          add_r_pt(cell_vertex_list[slot]);
        }
        for (int64_t s2 = t.cell_face_offsets[k]; s2 < t.cell_face_offsets[k + 1]; ++s2) {
          const int32_t g = std::abs(t.cell_faces[s2]) - 1;
          if (sched_face_stamp[g] == ep) continue;
          sched_face_stamp[g] = ep;
          sched_r_faces.push_back(g);
          const int32_t ow = mesh.face_owner[g], nb = mesh.face_neighbor[g];
          if (sched_cell_stamp[ow] != ep) {
            sched_cell_stamp[ow] = ep;
            sched_r_cells.push_back(ow);
          }
          if (nb >= 0 && sched_cell_stamp[nb] != ep) {
            sched_cell_stamp[nb] = ep;
            sched_r_cells.push_back(nb);
          }
          bool moving = false;
          for (int64_t e = mesh.face_offsets[g]; e < mesh.face_offsets[g + 1] && !moving; ++e) {
            const int32_t fv = mesh.face_vertices[e];
            for (int32_t m = 0; m < movers_count; ++m) {
              if (movers_ptr[m] == fv) {
                moving = true;
                break;
              }
            }
          }
          if (moving) sched_w_faces.push_back(g);
        }
      }
      if (task.slide) {
        add_r_pt(stair_vertices[task.wall_row]);
        add_r_pt(surface_vertex[task.wall_row]);
        if (split_of[task.wall_row] >= 0) {
          const auto& grid = splits[split_of[task.wall_row]].grid;
          add_r_pt(grid.front());
          add_r_pt(grid.back());
        }
        for (const int32_t g : wall_faces[task.wall_row]) {
          for (int64_t q = mesh.face_offsets[g]; q < mesh.face_offsets[g + 1]; ++q) {
            add_r_pt(mesh.face_vertices[q]);
          }
        }
      }
      if (task.layer) {
        add_r_pt(surface_vertex[task.layer_row]);
        add_r_pt(column[task.layer_row * (levels + 1) + task.level - 1]);
      }
      if (task.tilt) {
        add_r_pt(surface_vertex[task.layer_row]);
      }
      if (task.v < core_points && stair_vertex[task.v] >= 0) {
        const int32_t st_row = stair_vertex[task.v];
        add_r_pt(surface_vertex[st_row]);
        if (split_of[st_row] >= 0) {
          const auto& grid = splits[split_of[st_row]].grid;
          add_r_pt(grid.front());
          add_r_pt(grid.back());
        }
      }
      int32_t dep = wave_base - 1;
      for (const int32_t u : sched_r_pts) dep = std::max(dep, last_w_pt[u]);
      for (const int32_t k : sched_r_cells) dep = std::max(dep, last_w_cell[k]);
      for (const int32_t g : sched_r_faces) dep = std::max(dep, last_w_face[g]);
      for (int32_t m = 0; m < movers_count; ++m) {
        const int32_t u = movers_ptr[m];
        dep = std::max(dep, std::max(last_r_pt[u], last_w_pt[u]));
      }
      for (const int32_t k : sched_w_cells) dep = std::max(dep, std::max(last_r_cell[k], last_w_cell[k]));
      for (const int32_t g : sched_w_faces) dep = std::max(dep, std::max(last_r_face[g], last_w_face[g]));
      const int32_t w = dep + 1;
      for (const int32_t u : sched_r_pts) last_r_pt[u] = std::max(last_r_pt[u], w);
      for (const int32_t k : sched_r_cells) last_r_cell[k] = std::max(last_r_cell[k], w);
      for (const int32_t g : sched_r_faces) last_r_face[g] = std::max(last_r_face[g], w);
      for (int32_t m = 0; m < movers_count; ++m) last_w_pt[movers_ptr[m]] = w;
      for (const int32_t k : sched_w_cells) last_w_cell[k] = w;
      for (const int32_t g : sched_w_faces) last_w_face[g] = w;
      const int32_t rel = w - wave_base;
      task_waves[ti] = rel;
      if (rel > max_wave) max_wave = rel;
    }
    const int32_t num_waves = max_wave + 1;
    wave_base += num_waves;
    wave_buckets.resize(static_cast<std::size_t>(num_waves));
    for (int32_t w = 0; w < num_waves; ++w) wave_buckets[w].clear();
    for (std::size_t ti = 0; ti < tasks.size(); ++ti) {
      wave_buckets[task_waves[ti]].push_back(static_cast<int32_t>(ti));
    }
    task_repaired.assign(tasks.size(), 0);
    const std::size_t min_parallel_wave = std::max<std::size_t>(8, 2 * static_cast<std::size_t>(in.parallelism));
    for (int32_t w = 0; w < num_waves; ++w) {
      const auto& bucket = wave_buckets[w];
      if (bucket.empty()) continue;
      if (bucket.size() < min_parallel_wave) {
        for (const int32_t ti : bucket) task_repaired[ti] = run_repair_task(tasks[ti], single_scratch);
        continue;
      }
      const std::size_t chunk = (bucket.size() + max_wave_chunks - 1) / max_wave_chunks;
      status = parallel_for(
          bucket.size(), ParallelRequest{chunk, in.parallelism},
          [&](std::size_t begin, std::size_t end, ChunkContext& ctx) -> int32_t {
            const GradualNearestScope mode;
            if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
            RepairScratch& sc = wave_scratches[ctx.chunk_index];
            for (std::size_t j = begin; j < end; ++j) {
              const int32_t ti = bucket[j];
              task_repaired[ti] = run_repair_task(tasks[ti], sc);
            }
            return PHX_MC_OK;
          });
      if (status != PHX_MC_OK) return status;
    }
    for (const int64_t r : task_repaired) repaired += r;
  }
  stats[26] = static_cast<double>(repaired);
  stats[21] = static_cast<double>(reverted);
  return PHX_MC_OK;
}

int32_t Builder::certify() {
  const int64_t cells = static_cast<int64_t>(mesh.cell_kinds.size());
  const int64_t failed = uncertified_cells(mesh);
  stats[10] = static_cast<double>(cells - failed);
  stats[23] = static_cast<double>(failed);
  stats[22] = failed == 0 ? 1.0 : 0.0;
  return failed == 0 ? PHX_MC_OK : PHX_MC_UNRESOLVED;
}

}  // namespace

int64_t uncertified_cells(const Mesh& mesh) {
  const Topology t = topology(mesh);
  const int64_t cells = static_cast<int64_t>(mesh.cell_kinds.size());
  int64_t failed = 0;
  for (int64_t c = 0; c < cells; ++c) failed += !polyhedron_certified(mesh, t, c);
  return failed;
}

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
  status = builder.partition();
  if (status != PHX_MC_OK) return status;
  builder.build_pieces();
  builder.split_core_faces();
  status = builder.project();
  if (status != PHX_MC_OK) return status;
  builder.levels = in.layer_count + 1;
  status = builder.emit_layers();
  if (status != PHX_MC_OK) return status;
  status = builder.place_layers();
  if (status != PHX_MC_OK) return status;
  stats[4] = seconds(stage);
  stage = Clock::now();
  status = builder.smooth();
  if (status != PHX_MC_OK) return status;
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
  {
    const std::size_t n_stair = builder.stair_vertices.size();
    std::vector<double> row_dist(n_stair, 0.0);
    auto eval_stair = [&](std::size_t row) {
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
      row_dist[row] = norm(sub(hit.point, p));
    };
    if (in.parallelism == 1 || n_stair <= 256) {
      for (std::size_t row = 0; row < n_stair; ++row) eval_stair(row);
    } else {
      status = parallel_for(n_stair, ParallelRequest{256, in.parallelism},
                            [&](std::size_t begin, std::size_t end, ChunkContext&) -> int32_t {
        const GradualNearestScope mode;
        if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
        for (std::size_t row = begin; row < end; ++row) eval_stair(row);
        return PHX_MC_OK;
      });
      if (status != PHX_MC_OK) return status;
    }
    for (const double d : row_dist) vertex_distance = std::max(vertex_distance, d);
  }
  // Slid wall points of split curve columns: on their patch.
  const int32_t layers = builder.levels - 1;
  for (const auto& split : builder.splits) {
    for (int32_t q = 0; q < 2 * layers; ++q) {
      const int32_t k = q % layers + 1;
      const int32_t v = split.grid[q < layers ? k * (layers + 1) : k];
      const Vec p = builder.point(v);
      const Hit hit = walk_triangle(surface, p, split.item[q]);
      mesh.vertex_kinds[v] = kPatch;
      mesh.vertex_triangles[v] = hit.item;
      vertex_distance = std::max(vertex_distance, norm(sub(hit.point, p)));
    }
  }
  // Segment midpoints: on the curve and on their patches.
  for (const auto& m : builder.mids) {
    {
      const Vec p = builder.point(m.grid[0]);
      Hit hit = walk_edge(surface, p, m.curve_item);
      mesh.vertex_kinds[m.grid[0]] = kFeature;
      mesh.vertex_triangles[m.grid[0]] = surface.edge_triangles[hit.item];
      vertex_distance = std::max(vertex_distance, norm(sub(hit.point, p)));
    }
    for (int32_t q = 0; q < 2 * layers; ++q) {
      const int32_t k = q % layers + 1;
      const int32_t v = m.grid[q < layers ? k * (layers + 1) : k];
      const Vec p = builder.point(v);
      const Hit hit = walk_triangle(surface, p, m.item[q]);
      mesh.vertex_kinds[v] = kPatch;
      mesh.vertex_triangles[v] = hit.item;
      vertex_distance = std::max(vertex_distance, norm(sub(hit.point, p)));
    }
  }
  // Boundary faces: the facet of their patch, read from a patch vertex's
  // triangle (a face without one: the nearest triangle of its center). The
  // sampled distance of centers and edge midpoints is to the face's patch
  // (a walk from that triangle), an upper bound on the distance to the
  // surface.
  const int64_t faces = static_cast<int64_t>(mesh.face_owner.size());
  mesh.face_facets.assign(faces, -1);
  double sampled = 0.0;
  {
    std::vector<int32_t> boundary_faces;
    boundary_faces.reserve(builder.stair_vertices.size() * 2);
    for (int64_t f = 0; f < faces; ++f) {
      if (mesh.face_neighbor[f] < 0) boundary_faces.push_back(static_cast<int32_t>(f));
    }
    std::vector<double> face_sampled(boundary_faces.size(), 0.0);
    auto eval_bface = [&](std::size_t i) {
      const int64_t f = boundary_faces[i];
      Vec center, area, average;
      face_geometry(mesh, f, center, area, average);
      int32_t seed = -1;
      for (int64_t q = mesh.face_offsets[f]; q < mesh.face_offsets[f + 1] && seed < 0; ++q) {
        const int32_t v = mesh.face_vertices[q];
        if (mesh.vertex_kinds[v] == kPatch) seed = mesh.vertex_triangles[v];
      }
      Hit hit = seed >= 0 ? walk_triangle(surface, average, seed) : nearest_triangle(surface, average, -1);
      mesh.face_facets[f] = surface.patch_facet[surface.patch[hit.item]];
      double local_sampled = std::sqrt(hit.d2);
      for (int64_t q = mesh.face_offsets[f]; q < mesh.face_offsets[f + 1]; ++q) {
        const int64_t r = q + 1 < mesh.face_offsets[f + 1] ? q + 1 : mesh.face_offsets[f];
        const Vec mid = mul(add(builder.point(mesh.face_vertices[q]), builder.point(mesh.face_vertices[r])), 0.5);
        hit = walk_triangle(surface, mid, hit.item);
        local_sampled = std::max(local_sampled, std::sqrt(hit.d2));
      }
      face_sampled[i] = local_sampled;
    };
    if (in.parallelism == 1 || boundary_faces.size() <= 256) {
      for (std::size_t i = 0; i < boundary_faces.size(); ++i) eval_bface(i);
    } else {
      status = parallel_for(boundary_faces.size(), ParallelRequest{256, in.parallelism},
                            [&](std::size_t begin, std::size_t end, ChunkContext&) -> int32_t {
        const GradualNearestScope mode;
        if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
        for (std::size_t i = begin; i < end; ++i) eval_bface(i);
        return PHX_MC_OK;
      });
      if (status != PHX_MC_OK) return status;
    }
    for (const double d : face_sampled) sampled = std::max(sampled, d);
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
