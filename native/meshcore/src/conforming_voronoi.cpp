//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Conforming Voronoi cells of a triangulated solid (phx_mc_conforming_voronoi).
//
// Seeds. Automatic seeds follow the size h: sharp feature corners, feature
// edges sampled at spacing h, then a maximal Poisson-disk sample of the
// surface at radius h (candidates in Morton order); every surface sample p
// with inward offset direction m (n at smooth points, the least-squares
// solution of n_k . m = 1 over the incident normals at features) gives a seed
// p - h/2 m on each solid side of its triangle, kept when it lies on that side
// (vertical ray parity with exact orient2d) and at least h/4 from the surface.
// A body-centred cubic lattice of cell volume h^3 fills each region beyond
// 3h/2 of the surface. Every automatic seed is displaced by a deterministic
// hash of relative size 2^-10: symmetric seed patterns (feature samples about
// a corner) are exactly cospherical, and the displacement splits them into
// Voronoi features near 2^-10 h rather than at rounding scale; eight box
// seeds far outside close the diagram.
//
// Cells. The exact Delaunay triangulation (delaunay3d) gives the Voronoi
// diagram. The restricted Voronoi diagram of the surface is built triangle by
// triangle: a triangle is clipped (Sutherland-Hodgman) by the bisectors of the
// Delaunay neighbours of each seed met in a breadth-first walk. A clipped
// vertex is symbolic: a triangle corner, a triangle edge on the bisector of two
// seeds, or the point of the triangle on the bisectors of three seeds (where a
// Voronoi edge pierces it). In barycentric coordinates every constraint is a
// linear form (a unit vector, or the row of g_ix = |x - s_i|^2 - |x - s_x|^2 at
// the corners), a vertex is the cross product of its two forms, and the side
// of a vertex against bisector (i, l) is sign(r_l . (c1 x c2)) sign(1 . (c1 x
// c2)): a polynomial in the input coordinates, evaluated with a running error
// bound and exactly (expansions) when the bound does not decide. Symbolic keys
// weld vertices without tolerances. Voronoi vertices (tet circumcentres) get
// their region by a walk over the triangulation from a box seed's tet: along a
// Voronoi edge the region changes at each piercing point, from the back to the
// front region of the pierced triangle when the edge runs along its normal
// (exact sign of n_t . N); every second arrival must agree.
//
// A cell is a connected component of (Voronoi cell of a seed) intersected
// with one region. Cells whose seed meets no surface piece are Voronoi cells
// taken directly from the dual; others get their surface pieces (merged per
// source facet) and, for every Delaunay neighbour j, the loops of the bisector
// face: its Voronoi edge portions split at piercing points plus the surface
// curve on the bisector, chained per region.
//
// Certification (prove): every cell closed (each directed edge met once by
// its reverse), every surface triangle covered by its pieces, every region's
// cell volumes summing to its volume, and exact convexity: a clipped cell is
// convex exactly when no reflex surface edge (exact orient3d) meets its
// surface faces (Voronoi cells are convex). A degenerate exact predicate, an
// inconsistent region walk or a failed certificate repairs automatic seeds by
// a new displacement and rebuilds, up to the displacement bound h 2^-6.
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "capi_guard.hpp"
#include "expansion.hpp"
#include "floating_point_mode.hpp"
#include "geometry3.hpp"
#include "key_map.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

struct phx_mc_voronoi_cells {
  std::vector<double> vertices;
  std::vector<int8_t> vertex_kinds;
  std::vector<int32_t> vertex_sources;  // two per vertex
  std::vector<int64_t> face_offsets{0};
  std::vector<int32_t> face_vertices;
  std::vector<int32_t> face_cells;  // two per face
  std::vector<int32_t> face_facets;
  std::vector<int32_t> cell_seeds;
  std::vector<int32_t> cell_regions;
  std::vector<double> cell_volumes;
  std::vector<double> cell_centroids;
  std::vector<uint8_t> cell_convex;
  std::vector<double> seeds;
  // Exact coordinates of every vertex: x_k = N_k / D, the expansions
  // N_0, N_1, N_2, D of vertex v at exact_offsets[4 v + 0 .. 4 v + 4].
  // `vertices` holds their correctly rounded (nearest-even) values.
  std::vector<int64_t> exact_offsets{0};
  std::vector<double> exact_values;
  std::vector<double> stats = std::vector<double>(PHX_MC_CONFORMING_VORONOI_STATS, 0.0);
};

namespace phx::mc::conforming_voronoi {
namespace {

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

constexpr double kUnit = 0x1p-53;

// A binary64 value with a bound on its distance to the exact value.
struct Bounded {
  double v;
  double e;
};
inline Bounded operator+(Bounded a, Bounded b) {
  const double v = a.v + b.v;
  return {v, a.e + b.e + kUnit * std::fabs(v)};
}
inline Bounded operator-(Bounded a, Bounded b) {
  const double v = a.v - b.v;
  return {v, a.e + b.e + kUnit * std::fabs(v)};
}
inline Bounded operator*(Bounded a, Bounded b) {
  const double v = a.v * b.v;
  return {v, std::fabs(a.v) * b.e + std::fabs(b.v) * a.e + a.e * b.e + kUnit * std::fabs(v)};
}
// 1 / -1 when the bound decides the sign, 2 otherwise. The bound itself was
// rounded a few dozen times; the factor covers that relative error.
inline int decided_sign(Bounded a) {
  const double bound = a.e * (1.0 + 0x1p-40) + 0x1p-1000;
  if (a.v > bound) return 1;
  if (a.v < -bound) return -1;
  return 2;
}

Bounded bounded_difference(double a, double b) {
  const double v = a - b;
  return {v, kUnit * std::fabs(v)};
}
Bounded bounded_squared_distance(const double* a, const double* s) {
  Bounded total{0.0, 0.0};
  for (int k = 0; k < 3; ++k) {
    const Bounded d = bounded_difference(a[k], s[k]);
    total = total + d * d;
  }
  return total;
}
Expansion exact_squared_distance(const double* a, const double* s) {
  Expansion total;
  for (int k = 0; k < 3; ++k) {
    const Expansion d = Expansion::difference(a[k], s[k]);
    total = total + d * d;
  }
  return total;
}

template <class T>
std::array<T, 3> cross3(const std::array<T, 3>& a, const std::array<T, 3>& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
template <class T>
T dot3(const std::array<T, 3>& a, const std::array<T, 3>& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Shewchuk's compress: the same value with few, nonadjacent components.
Expansion compressed(const Expansion& e) {
  const int m = e.size();
  if (m <= 1) return e;
  const double* x = e.data();
  std::vector<double> g(static_cast<std::size_t>(m));
  int bottom = m - 1;
  double q = x[m - 1];
  for (int k = m - 2; k >= 0; --k) {
    double sum, small;
    fast_two_sum(q, x[k], sum, small);
    if (small != 0.0) {
      g[bottom--] = sum;
      q = small;
    } else {
      q = sum;
    }
  }
  g[bottom] = q;
  std::vector<double> h;
  h.reserve(static_cast<std::size_t>(m - bottom));
  for (int k = bottom + 1; k < m; ++k) {
    double sum, small;
    fast_two_sum(g[k], q, sum, small);
    if (small != 0.0) h.push_back(small);
    q = sum;
  }
  h.push_back(q);
  return Expansion::from_components(h.data(), static_cast<int>(h.size()));
}

// sign(n - (q + half) d) for d > 0, with q + half an exact two-term value.
int compare_quotient(const Expansion& n, const Expansion& d, double q, double half) {
  return (n - (d.scaled(q) + d.scaled(half))).sign();
}

// The binary64 value nearest n / d (ties to even); d is nonzero.
double rounded_quotient(Expansion n, Expansion d) {
  if (d.sign() < 0) {
    n = -n;
    d = -d;
  }
  double q = n.estimate() / d.estimate();
  const auto odd = [](double x) { return (std::bit_cast<uint64_t>(x) & 1u) != 0; };
  while (true) {
    const double above = std::nextafter(q, HUGE_VAL);
    const int up = compare_quotient(n, d, q, 0.5 * (above - q));
    if (up > 0 || (up == 0 && odd(q))) {
      q = above;
      continue;
    }
    const double below = std::nextafter(q, -HUGE_VAL);
    const int down = compare_quotient(n, d, q, 0.5 * (below - q));
    if (down < 0 || (down == 0 && odd(q))) {
      q = below;
      continue;
    }
    return q;
  }
}

// 128-bit vertex identity: corner, edge on a bisector, or triangle on two.
struct Key {
  uint64_t hi;
  uint64_t lo;
  bool operator==(const Key& o) const { return hi == o.hi && lo == o.lo; }
  bool operator<(const Key& o) const { return hi != o.hi ? hi < o.hi : lo < o.lo; }
};
struct KeyHash {
  std::size_t operator()(const Key& k) const noexcept {
    uint64_t h = k.hi * 0x9E3779B97F4A7C15ULL ^ (k.lo + 0x632BE59BD9B4E019ULL + (k.hi << 6));
    h ^= h >> 31;
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 29;
    return static_cast<std::size_t>(h);
  }
};
Key corner_key(int32_t vertex) { return {0, static_cast<uint64_t>(vertex)}; }
Key edge_key(int32_t edge, int32_t a, int32_t b) {
  if (a > b) std::swap(a, b);
  return {(uint64_t{1} << 62) | static_cast<uint64_t>(edge),
          (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b)};
}
std::array<int32_t, 3> sorted3(int32_t a, int32_t b, int32_t c) {
  std::array<int32_t, 3> s{a, b, c};
  std::sort(s.begin(), s.end());
  return s;
}
Key triple_key(const std::array<int32_t, 3>& s) {
  return {(static_cast<uint64_t>(s[0])), (static_cast<uint64_t>(s[1]) << 32) |
                                              static_cast<uint64_t>(s[2])};
}
Key face_key(int32_t triangle, const std::array<int32_t, 3>& s) {
  return {(uint64_t{2} << 62) | (static_cast<uint64_t>(triangle) << 31) |
              static_cast<uint64_t>(s[0]),
          (static_cast<uint64_t>(s[1]) << 32) | static_cast<uint64_t>(s[2])};
}

enum VertexKind : int8_t { kVoronoi = 0, kSurfaceVertex = 1, kSurfaceEdge = 2, kSurfaceFace = 3 };

struct Input {
  int64_t vertex_count;
  const double* vertices;
  int64_t triangle_count;
  const int32_t* triangles;
  const int32_t* regions;  // (front, back)
  const int32_t* facets;
  int64_t seed_count;
  const double* seeds;
  double size;
  double feature_angle;
  bool prove;
};

struct Failure : std::exception {
  int32_t status;
  explicit Failure(int32_t s) : status(s) {}
};

// ------------------------------------------------------------------ surface
struct Surface {
  const Input* in = nullptr;
  int32_t region_count = 0;
  std::vector<int32_t> edges;            // two vertices per undirected edge
  std::vector<int32_t> triangle_edges;   // edge opposite corner k: 3 t + k
  std::vector<int32_t> edge_triangles;   // up to two incident triangles (-1 when more/less)
  std::vector<uint8_t> edge_reflex;      // exact: some incident pair reflex
  std::vector<uint8_t> edge_feature;
  // Classes of edge-adjacent triangles of one facet and region pair that are
  // exactly coplanar (orient3d) with equal orientation: pieces merge within one.
  std::vector<int32_t> plane_class;
  std::vector<double> normals;           // unit, 3 per triangle
  std::vector<double> areas;
  double lower[3]{}, upper[3]{};
  double diameter = 0.0;
  int32_t feature_edges = 0;
  int32_t corners = 0;

  const double* point(int32_t v) const { return in->vertices + 3 * static_cast<int64_t>(v); }
  const double* corner(int32_t t, int k) const {
    return point(in->triangles[3 * static_cast<int64_t>(t) + k]);
  }
  int32_t front(int32_t t) const { return in->regions[2 * static_cast<int64_t>(t)]; }
  int32_t back(int32_t t) const { return in->regions[2 * static_cast<int64_t>(t) + 1]; }
};

void build_surface(const Input& in, Surface& s) {
  s.in = &in;
  const int64_t m = in.triangle_count;
  for (int k = 0; k < 3; ++k) {
    s.lower[k] = in.vertices[k];
    s.upper[k] = in.vertices[k];
  }
  for (int64_t v = 0; v < in.vertex_count; ++v) {
    for (int k = 0; k < 3; ++k) {
      s.lower[k] = std::min(s.lower[k], in.vertices[3 * v + k]);
      s.upper[k] = std::max(s.upper[k], in.vertices[3 * v + k]);
    }
  }
  s.diameter = std::sqrt(geometry::squared_distance(s.lower, s.upper));
  s.normals.resize(3 * m);
  s.areas.resize(m);
  for (int64_t t = 0; t < m; ++t) {
    const int32_t front = in.regions[2 * t], back = in.regions[2 * t + 1];
    s.region_count = std::max({s.region_count, front + 1, back + 1});
    double ab[3], ac[3], n[3];
    geometry::difference(s.corner(t, 1), s.corner(t, 0), ab);
    geometry::difference(s.corner(t, 2), s.corner(t, 0), ac);
    geometry::cross(ab, ac, n);
    const double length = geometry::length(n);
    if (!(length > 0.0)) throw Failure(PHX_MC_DEGENERATE_INPUT);
    for (int k = 0; k < 3; ++k) s.normals[3 * t + k] = n[k] / length;
    s.areas[t] = 0.5 * length;
  }
  // Undirected edges by sorting (a, b, triangle, corner).
  std::vector<std::array<int32_t, 4>> rows;
  rows.reserve(3 * m);
  for (int64_t t = 0; t < m; ++t) {
    for (int k = 0; k < 3; ++k) {
      int32_t a = in.triangles[3 * t + (k + 1) % 3], b = in.triangles[3 * t + (k + 2) % 3];
      if (a > b) std::swap(a, b);
      rows.push_back({a, b, static_cast<int32_t>(t), k});
    }
  }
  std::sort(rows.begin(), rows.end());
  s.triangle_edges.assign(3 * m, -1);
  s.plane_class.resize(m);
  std::iota(s.plane_class.begin(), s.plane_class.end(), 0);
  for (std::size_t r = 0; r < rows.size();) {
    std::size_t e = r;
    while (e < rows.size() && rows[e][0] == rows[r][0] && rows[e][1] == rows[r][1]) ++e;
    const int32_t edge = static_cast<int32_t>(s.edges.size() / 2);
    s.edges.push_back(rows[r][0]);
    s.edges.push_back(rows[r][1]);
    const bool pair = e - r == 2;
    s.edge_triangles.push_back(pair ? rows[r][2] : -1);
    s.edge_triangles.push_back(pair ? rows[r + 1][2] : -1);
    // Bit 1: reflex seen from the back region; bit 2: from the front region.
    // Edges without exactly two triangles count as reflex on both sides.
    uint8_t reflex = pair ? 0 : 3, feature = pair ? 0 : 1;
    for (std::size_t x = r; x < e; ++x) {
      s.triangle_edges[3 * rows[x][2] + rows[x][3]] = edge;
      for (std::size_t y = x + 1; y < e; ++y) {
        const int32_t t = rows[x][2], u = rows[y][2];
        const int32_t opposite = in.triangles[3 * static_cast<int64_t>(u) + rows[y][3]];
        // The other triangle's far corner in front of t bends the surface
        // towards the front: reflex for the back region, convex for the front.
        const int bend =
            orient3d(s.corner(t, 0), s.corner(t, 1), s.corner(t, 2), s.point(opposite));
        if (bend > 0) reflex |= 1;
        if (bend < 0) reflex |= 2;
        if (pair && bend == 0 && in.facets[t] == in.facets[u] && s.front(t) == s.front(u) &&
            s.back(t) == s.back(u) &&
            geometry::dot(&s.normals[3 * t], &s.normals[3 * u]) > 0.0) {
          int32_t a = t, b = u;
          while (s.plane_class[a] != a) a = s.plane_class[a] = s.plane_class[s.plane_class[a]];
          while (s.plane_class[b] != b) b = s.plane_class[b] = s.plane_class[s.plane_class[b]];
          if (a != b) s.plane_class[std::max(a, b)] = std::min(a, b);
        }
        const double c = geometry::dot(&s.normals[3 * t], &s.normals[3 * u]);
        if (c < std::cos(in.feature_angle) || s.front(t) != s.front(u) ||
            s.back(t) != s.back(u)) {
          feature = 1;
        }
      }
    }
    if (e - r == 1) feature = 1;
    s.edge_reflex.push_back(reflex);
    s.edge_feature.push_back(feature);
    s.feature_edges += feature;
    r = e;
  }
  for (int64_t t = 0; t < m; ++t) {
    int32_t root = static_cast<int32_t>(t);
    while (s.plane_class[root] != root) root = s.plane_class[root];
    s.plane_class[t] = root;
  }
}

// Uniform grid of triangles by bounding box.
struct TriangleGrid {
  double origin[3]{};
  double cell = 1.0;
  int64_t dims[3]{1, 1, 1};
  std::vector<int64_t> offsets;
  std::vector<int32_t> items;

  int64_t clamp_index(double x, int axis) const {
    const double q = std::floor((x - origin[axis]) / cell);
    if (!(q > 0.0)) return 0;
    return std::min<int64_t>(dims[axis] - 1, static_cast<int64_t>(q));
  }
  int64_t flat(int64_t i, int64_t j, int64_t k) const { return (k * dims[1] + j) * dims[0] + i; }

  void build(const Surface& s, double requested) {
    double extent = 0.0;
    for (int k = 0; k < 3; ++k) extent = std::max(extent, s.upper[k] - s.lower[k]);
    cell = std::max(requested, extent / 256.0);
    for (int k = 0; k < 3; ++k) {
      origin[k] = s.lower[k];
      dims[k] = std::max<int64_t>(1, static_cast<int64_t>((s.upper[k] - s.lower[k]) / cell) + 1);
    }
    const int64_t total = dims[0] * dims[1] * dims[2];
    std::vector<int64_t> counts(total + 1, 0);
    const int64_t m = s.in->triangle_count;
    auto range = [&](int64_t t, int64_t lo[3], int64_t hi[3]) {
      for (int a = 0; a < 3; ++a) {
        double l = s.corner(t, 0)[a], u = l;
        for (int k = 1; k < 3; ++k) {
          l = std::min(l, s.corner(t, k)[a]);
          u = std::max(u, s.corner(t, k)[a]);
        }
        lo[a] = clamp_index(l, a);
        hi[a] = clamp_index(u, a);
      }
    };
    for (int pass = 0; pass < 2; ++pass) {
      for (int64_t t = 0; t < m; ++t) {
        int64_t lo[3], hi[3];
        range(t, lo, hi);
        for (int64_t k = lo[2]; k <= hi[2]; ++k)
          for (int64_t j = lo[1]; j <= hi[1]; ++j)
            for (int64_t i = lo[0]; i <= hi[0]; ++i) {
              const int64_t c = flat(i, j, k);
              if (pass == 0) {
                ++counts[c + 1];
              } else {
                items[offsets[c] + counts[c]++] = static_cast<int32_t>(t);
              }
            }
      }
      if (pass == 0) {
        offsets.assign(total + 1, 0);
        for (int64_t c = 0; c < total; ++c) offsets[c + 1] = offsets[c] + counts[c + 1];
        items.assign(offsets[total], 0);
        std::fill(counts.begin(), counts.end(), 0);
      }
    }
  }

  // Whether any triangle lies within `radius` of p.
  bool any_within(const Surface& s, const double* p, double radius) const {
    int64_t lo[3], hi[3];
    for (int a = 0; a < 3; ++a) {
      lo[a] = clamp_index(p[a] - radius, a);
      hi[a] = clamp_index(p[a] + radius, a);
    }
    for (int64_t k = lo[2]; k <= hi[2]; ++k)
      for (int64_t j = lo[1]; j <= hi[1]; ++j)
        for (int64_t i = lo[0]; i <= hi[0]; ++i) {
          const int64_t c = flat(i, j, k);
          for (int64_t e = offsets[c]; e < offsets[c + 1]; ++e) {
            const int32_t t = items[e];
            if (geometry::point_triangle_distance(p, s.corner(t, 0), s.corner(t, 1),
                                                  s.corner(t, 2)) < radius) {
              return true;
            }
          }
        }
    return false;
  }
};

// Vertical columns: triangles by their xy bounding boxes.
struct ColumnGrid {
  double origin[2]{};
  double cell = 1.0;
  int64_t dims[2]{1, 1};
  std::vector<int64_t> offsets;
  std::vector<int32_t> items;

  int64_t index(double x, int axis) const {
    const double q = std::floor((x - origin[axis]) / cell);
    if (!(q > 0.0)) return 0;
    return std::min<int64_t>(dims[axis] - 1, static_cast<int64_t>(q));
  }

  void build(const Surface& s, double requested) {
    const double extent = std::max(s.upper[0] - s.lower[0], s.upper[1] - s.lower[1]);
    cell = std::max(requested, extent / 1024.0);
    for (int a = 0; a < 2; ++a) {
      origin[a] = s.lower[a];
      dims[a] = std::max<int64_t>(1, static_cast<int64_t>((s.upper[a] - s.lower[a]) / cell) + 1);
    }
    const int64_t total = dims[0] * dims[1];
    std::vector<int64_t> counts(total + 1, 0);
    const int64_t m = s.in->triangle_count;
    for (int pass = 0; pass < 2; ++pass) {
      for (int64_t t = 0; t < m; ++t) {
        int64_t lo[2], hi[2];
        for (int a = 0; a < 2; ++a) {
          double l = s.corner(t, 0)[a], u = l;
          for (int k = 1; k < 3; ++k) {
            l = std::min(l, s.corner(t, k)[a]);
            u = std::max(u, s.corner(t, k)[a]);
          }
          lo[a] = index(l, a);
          hi[a] = index(u, a);
        }
        for (int64_t j = lo[1]; j <= hi[1]; ++j)
          for (int64_t i = lo[0]; i <= hi[0]; ++i) {
            const int64_t c = j * dims[0] + i;
            if (pass == 0) {
              ++counts[c + 1];
            } else {
              items[offsets[c] + counts[c]++] = static_cast<int32_t>(t);
            }
          }
      }
      if (pass == 0) {
        offsets.assign(total + 1, 0);
        for (int64_t c = 0; c < total; ++c) offsets[c + 1] = offsets[c] + counts[c + 1];
        items.assign(offsets[total], 0);
        std::fill(counts.begin(), counts.end(), 0);
      }
    }
  }
};

// Sign of orient2d(a, b, q) for q displaced by (eps, eps^2).
int perturbed_orient2d(const double* a, const double* b, const double* q) {
  const double a2[2] = {a[0], a[1]}, b2[2] = {b[0], b[1]}, q2[2] = {q[0], q[1]};
  const int sign = orient2d(a2, b2, q2);
  if (sign != 0) return sign;
  if (b[1] != a[1]) return b[1] > a[1] ? -1 : 1;
  return b[0] > a[0] ? 1 : (b[0] < a[0] ? -1 : 0);
}

// Crossings of the upward vertical line through (x, y) with the surface:
// (height, region below, region above), sorted by height.
struct Crossing {
  double z;
  int32_t below;
  int32_t above;
};
void column_crossings(const Surface& s, const ColumnGrid& grid, const double* q,
                      std::vector<Crossing>& out) {
  out.clear();
  const int64_t c = grid.index(q[1], 1) * grid.dims[0] + grid.index(q[0], 0);
  for (int64_t e = grid.offsets[c]; e < grid.offsets[c + 1]; ++e) {
    const int32_t t = grid.items[e];
    const double* a = s.corner(t, 0);
    const double* b = s.corner(t, 1);
    const double* d = s.corner(t, 2);
    const double a2[2] = {a[0], a[1]}, b2[2] = {b[0], b[1]}, d2[2] = {d[0], d[1]};
    const int orientation = orient2d(a2, b2, d2);
    if (orientation == 0) continue;  // vertical triangle: no perturbed point inside
    if (perturbed_orient2d(a, b, q) != orientation || perturbed_orient2d(b, d, q) != orientation ||
        perturbed_orient2d(d, a, q) != orientation) {
      continue;
    }
    const double* n = &s.normals[3 * static_cast<int64_t>(t)];
    // Height of the triangle plane over (x, y).
    const double z = a[2] - (n[0] * (q[0] - a[0]) + n[1] * (q[1] - a[1])) / n[2];
    // orientation > 0: the normal points up, so the front lies above.
    if (orientation > 0) {
      out.push_back({z, s.back(t), s.front(t)});
    } else {
      out.push_back({z, s.front(t), s.back(t)});
    }
  }
  std::sort(out.begin(), out.end(), [](const Crossing& x, const Crossing& y) {
    return x.z < y.z;
  });
}

// Region of a point from the crossings above it (the top is exterior).
int32_t region_at(const std::vector<Crossing>& crossings, double z) {
  int32_t region = -1;
  for (std::size_t k = crossings.size(); k-- > 0;) {
    if (crossings[k].z <= z) break;
    region = crossings[k].below;
  }
  return region;
}

// ------------------------------------------------------------------ seeds
uint64_t mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
double unit_hash(uint64_t x) { return static_cast<double>(mix(x) >> 11) * 0x1p-53 * 2.0 - 1.0; }

uint64_t morton(const double* p, const double* lower, double scale) {
  uint64_t key = 0;
  uint32_t q[3];
  for (int a = 0; a < 3; ++a) {
    const double x = (p[a] - lower[a]) * scale;
    q[a] = x <= 0.0 ? 0u : (x >= 2097151.0 ? 2097151u : static_cast<uint32_t>(x));
  }
  for (int bit = 20; bit >= 0; --bit) {
    for (int a = 0; a < 3; ++a) key = (key << 1) | ((q[a] >> bit) & 1u);
  }
  return key;
}

struct Sample {
  double p[3];
  double m[3];       // inward offset direction (towards the back side)
  int32_t triangle;  // a triangle carrying the sample
};

struct HashGrid {
  double cell;
  std::unordered_map<uint64_t, std::vector<int32_t>> bins;
  explicit HashGrid(double c) : cell(c) {}
  static uint64_t key(int64_t i, int64_t j, int64_t k) {
    return (static_cast<uint64_t>(i + (1 << 20)) << 42) |
           (static_cast<uint64_t>(j + (1 << 20)) << 21) | static_cast<uint64_t>(k + (1 << 20));
  }
  int64_t index(double x) const { return static_cast<int64_t>(std::floor(x / cell)); }
  bool near(const std::vector<Sample>& samples, const double* p, double radius) const {
    const int64_t i = index(p[0]), j = index(p[1]), k = index(p[2]);
    for (int64_t dk = -1; dk <= 1; ++dk)
      for (int64_t dj = -1; dj <= 1; ++dj)
        for (int64_t di = -1; di <= 1; ++di) {
          const auto found = bins.find(key(i + di, j + dj, k + dk));
          if (found == bins.end()) continue;
          for (const int32_t s : found->second) {
            if (geometry::squared_distance(samples[s].p, p) < radius * radius) return true;
          }
        }
    return false;
  }
  void add(const std::vector<Sample>& samples, int32_t s) {
    bins[key(index(samples[s].p[0]), index(samples[s].p[1]), index(samples[s].p[2]))].push_back(s);
  }
};

// Least-squares direction m with n_k . m = 1 over distinct incident normals.
void feature_direction(const std::vector<const double*>& normals, double* m) {
  double a[9] = {0}, b[3] = {0};
  for (const double* n : normals) {
    for (int r = 0; r < 3; ++r) {
      b[r] += n[r];
      for (int c = 0; c < 3; ++c) a[3 * r + c] += n[r] * n[c];
    }
  }
  const double reg = 1e-3 * static_cast<double>(normals.size());
  for (int r = 0; r < 3; ++r) a[4 * r] += reg;
  // Cramer's rule on the regularized symmetric positive definite system.
  const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
                     a[2] * (a[3] * a[7] - a[4] * a[6]);
  for (int c = 0; c < 3; ++c) {
    double t[9];
    std::copy(a, a + 9, t);
    for (int r = 0; r < 3; ++r) t[3 * r + c] = b[r];
    m[c] = (t[0] * (t[4] * t[8] - t[5] * t[7]) - t[1] * (t[3] * t[8] - t[5] * t[6]) +
            t[2] * (t[3] * t[7] - t[4] * t[6])) /
           det;
  }
  const double length = geometry::length(m);
  if (length > 4.0) {
    for (int k = 0; k < 3; ++k) m[k] *= 4.0 / length;
  }
}

struct SeedSet {
  std::vector<double> points;
  std::vector<int32_t> regions;  // known region of automatic seeds, -2 unknown
  int32_t surface_seeds = 0;
  int32_t lattice_seeds = 0;
};

void automatic_seeds(const Surface& s, double h, SeedSet& out) {
  const Input& in = *s.in;
  const int64_t m = in.triangle_count;
  // Vertex feature degree and incident triangles.
  std::vector<int32_t> degree(in.vertex_count, 0);
  for (std::size_t e = 0; e < s.edge_feature.size(); ++e) {
    if (s.edge_feature[e] == 0) continue;
    ++degree[s.edges[2 * e]];
    ++degree[s.edges[2 * e + 1]];
  }
  std::vector<int64_t> star_offsets(in.vertex_count + 1, 0);
  for (int64_t k = 0; k < 3 * m; ++k) ++star_offsets[in.triangles[k] + 1];
  for (int64_t v = 0; v < in.vertex_count; ++v) star_offsets[v + 1] += star_offsets[v];
  std::vector<int32_t> star(3 * m);
  {
    std::vector<int64_t> cursor(star_offsets.begin(), star_offsets.end() - 1);
    for (int64_t t = 0; t < m; ++t)
      for (int k = 0; k < 3; ++k) star[cursor[in.triangles[3 * t + k]]++] = static_cast<int32_t>(t);
  }
  auto vertex_normals = [&](int32_t v, std::vector<const double*>& normals) {
    normals.clear();
    for (int64_t e = star_offsets[v]; e < star_offsets[v + 1]; ++e) {
      const double* n = &s.normals[3 * static_cast<int64_t>(star[e])];
      bool distinct = true;
      for (const double* o : normals) distinct = distinct && geometry::dot(n, o) < 0.999;
      if (distinct) normals.push_back(n);
    }
  };
  std::vector<Sample> samples;
  HashGrid grid(h);
  std::vector<const double*> normals;
  int32_t corner_count = 0;
  // Corners: feature degree other than 0 and 2.
  for (int32_t v = 0; v < in.vertex_count; ++v) {
    if (degree[v] == 0 || degree[v] == 2) continue;
    Sample sample{};
    std::copy_n(s.point(v), 3, sample.p);
    vertex_normals(v, normals);
    feature_direction(normals, sample.m);
    sample.triangle = star[star_offsets[v]];
    samples.push_back(sample);
    grid.add(samples, static_cast<int32_t>(samples.size() - 1));
    ++corner_count;
  }
  // Feature chains at spacing about h.
  {
    std::vector<int64_t> adjacency_offsets(in.vertex_count + 1, 0);
    for (std::size_t e = 0; e < s.edge_feature.size(); ++e) {
      if (!s.edge_feature[e]) continue;
      ++adjacency_offsets[s.edges[2 * e] + 1];
      ++adjacency_offsets[s.edges[2 * e + 1] + 1];
    }
    for (int64_t v = 0; v < in.vertex_count; ++v) adjacency_offsets[v + 1] += adjacency_offsets[v];
    std::vector<int32_t> adjacency(adjacency_offsets[in.vertex_count]);
    std::vector<int64_t> cursor(adjacency_offsets.begin(), adjacency_offsets.end() - 1);
    for (std::size_t e = 0; e < s.edge_feature.size(); ++e) {
      if (!s.edge_feature[e]) continue;
      adjacency[cursor[s.edges[2 * e]]++] = static_cast<int32_t>(e);
      adjacency[cursor[s.edges[2 * e + 1]]++] = static_cast<int32_t>(e);
    }
    std::vector<uint8_t> used(s.edge_feature.size(), 0);
    auto walk = [&](int32_t start, int32_t first_edge) {
      std::vector<int32_t> chain{start};
      int32_t v = start, e = first_edge;
      while (true) {
        used[e] = 1;
        const int32_t w = s.edges[2 * e] == v ? s.edges[2 * e + 1] : s.edges[2 * e];
        chain.push_back(w);
        v = w;
        if (degree[v] != 2) break;
        int32_t next = -1;
        for (int64_t k = adjacency_offsets[v]; k < adjacency_offsets[v + 1]; ++k)
          if (!used[adjacency[k]]) next = adjacency[k];
        if (next < 0) break;
        e = next;
      }
      double length = 0.0;
      for (std::size_t k = 1; k < chain.size(); ++k)
        length += std::sqrt(geometry::squared_distance(s.point(chain[k - 1]), s.point(chain[k])));
      const int64_t pieces = std::max<int64_t>(1, static_cast<int64_t>(std::llround(length / h)));
      const double step = length / static_cast<double>(pieces);
      double travelled = 0.0;
      std::size_t segment = 1;
      for (int64_t q = 1; q < pieces || (q == pieces && degree[chain.front()] == 2 && chain.front() == chain.back() && false); ++q) {
        const double target = step * static_cast<double>(q);
        while (segment < chain.size()) {
          const double piece =
              std::sqrt(geometry::squared_distance(s.point(chain[segment - 1]), s.point(chain[segment])));
          if (travelled + piece >= target) break;
          travelled += piece;
          ++segment;
        }
        if (segment >= chain.size()) break;
        const double* a = s.point(chain[segment - 1]);
        const double* b = s.point(chain[segment]);
        const double piece = std::sqrt(geometry::squared_distance(a, b));
        const double u = piece > 0.0 ? (target - travelled) / piece : 0.0;
        Sample sample{};
        for (int k = 0; k < 3; ++k) sample.p[k] = a[k] + u * (b[k] - a[k]);
        // Normals of the triangles on this feature edge.
        int32_t lo = chain[segment - 1], hi = chain[segment];
        if (lo > hi) std::swap(lo, hi);
        normals.clear();
        sample.triangle = -1;
        for (int64_t k = star_offsets[lo]; k < star_offsets[lo + 1]; ++k) {
          const int32_t t = star[k];
          const int32_t* tri = in.triangles + 3 * static_cast<int64_t>(t);
          if (tri[0] != hi && tri[1] != hi && tri[2] != hi) continue;
          const double* n = &s.normals[3 * static_cast<int64_t>(t)];
          bool distinct = true;
          for (const double* o : normals) distinct = distinct && geometry::dot(n, o) < 0.999;
          if (distinct) normals.push_back(n);
          if (sample.triangle < 0) sample.triangle = t;
        }
        feature_direction(normals, sample.m);
        if (!grid.near(samples, sample.p, 0.5 * h)) {
          samples.push_back(sample);
          grid.add(samples, static_cast<int32_t>(samples.size() - 1));
        }
      }
    };
    for (int32_t v = 0; v < in.vertex_count; ++v) {
      if (degree[v] == 0 || degree[v] == 2) continue;
      for (int64_t k = adjacency_offsets[v]; k < adjacency_offsets[v + 1]; ++k)
        if (!used[adjacency[k]]) walk(v, adjacency[k]);
    }
    // Closed loops without corners start at their smallest vertex.
    for (std::size_t e = 0; e < s.edge_feature.size(); ++e) {
      if (!s.edge_feature[e] || used[e]) continue;
      const int32_t start = s.edges[2 * e];
      Sample sample{};
      std::copy_n(s.point(start), 3, sample.p);
      vertex_normals(start, normals);
      feature_direction(normals, sample.m);
      sample.triangle = star[star_offsets[start]];
      samples.push_back(sample);
      grid.add(samples, static_cast<int32_t>(samples.size() - 1));
      walk(start, static_cast<int32_t>(e));
    }
  }
  // Smooth candidates: vertices off features and per-triangle barycentric
  // points at spacing at most h / 2, in Morton order.
  struct Candidate {
    uint64_t key;
    int64_t index;  // vertex v as -1 - v, else triangle point 3 * (t * K) ...
    double p[3];
    int32_t triangle;
    int32_t vertex;
  };
  std::vector<Candidate> candidates;
  const double scale = 2097151.0 / std::max(s.diameter, 1e-300);
  for (int32_t v = 0; v < in.vertex_count; ++v) {
    if (degree[v] != 0 || star_offsets[v + 1] == star_offsets[v]) continue;
    Candidate c{};
    std::copy_n(s.point(v), 3, c.p);
    c.key = morton(c.p, s.lower, scale);
    c.index = static_cast<int64_t>(candidates.size());
    c.triangle = star[star_offsets[v]];
    c.vertex = v;
    candidates.push_back(c);
  }
  for (int64_t t = 0; t < m; ++t) {
    double longest = 0.0;
    for (int k = 0; k < 3; ++k)
      longest = std::max(longest, geometry::squared_distance(s.corner(t, k), s.corner(t, (k + 1) % 3)));
    const int64_t K = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(std::sqrt(longest) / (0.5 * h))));
    for (int64_t i = 0; i <= K; ++i)
      for (int64_t j = 0; i + j <= K; ++j) {
        const int64_t k = K - i - j;
        if (K > 1 && (i == K || j == K || k == K)) continue;
        if (K == 1 && !(i == 0 && j == 0)) continue;
        double w[3] = {static_cast<double>(i) / K, static_cast<double>(j) / K,
                       static_cast<double>(k) / K};
        if (K == 1) w[0] = w[1] = w[2] = 1.0 / 3.0;
        Candidate c{};
        for (int a = 0; a < 3; ++a)
          c.p[a] = w[0] * s.corner(t, 0)[a] + w[1] * s.corner(t, 1)[a] + w[2] * s.corner(t, 2)[a];
        c.key = morton(c.p, s.lower, scale);
        c.index = static_cast<int64_t>(candidates.size());
        c.triangle = static_cast<int32_t>(t);
        c.vertex = -1;
        candidates.push_back(c);
      }
  }
  std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
    return a.key != b.key ? a.key < b.key : a.index < b.index;
  });
  for (const Candidate& c : candidates) {
    if (grid.near(samples, c.p, h)) continue;
    Sample sample{};
    std::copy_n(c.p, 3, sample.p);
    sample.triangle = c.triangle;
    if (c.vertex >= 0) {
      double n[3] = {0, 0, 0};
      for (int64_t e = star_offsets[c.vertex]; e < star_offsets[c.vertex + 1]; ++e)
        for (int k = 0; k < 3; ++k) n[k] += s.areas[star[e]] * s.normals[3 * static_cast<int64_t>(star[e]) + k];
      const double length = geometry::length(n);
      for (int k = 0; k < 3; ++k) sample.m[k] = n[k] / length;
    } else {
      std::copy_n(&s.normals[3 * static_cast<int64_t>(c.triangle)], 3, sample.m);
    }
    samples.push_back(sample);
    grid.add(samples, static_cast<int32_t>(samples.size() - 1));
  }

  TriangleGrid triangles;
  triangles.build(s, 1.5 * h);
  ColumnGrid columns;
  columns.build(s, h);
  std::vector<Crossing> crossings;
  // Inward-offset seeds on each solid side.
  const double depth = 0.5 * h;
  for (const Sample& sample : samples) {
    for (int side = 0; side < 2; ++side) {
      const int32_t region = side == 0 ? s.back(sample.triangle) : s.front(sample.triangle);
      if (region < 0) continue;
      const double sign = side == 0 ? -1.0 : 1.0;
      double q[3];
      for (int k = 0; k < 3; ++k) q[k] = sample.p[k] + sign * depth * sample.m[k];
      if (triangles.any_within(s, q, 0.25 * h)) continue;
      column_crossings(s, columns, q, crossings);
      if (region_at(crossings, q[2]) != region) continue;
      out.points.insert(out.points.end(), q, q + 3);
      out.regions.push_back(region);
      ++out.surface_seeds;
    }
  }
  // Body-centred cubic lattice of cell volume h^3, beyond 3h/2 of the surface.
  const double a = std::cbrt(2.0) * h;
  const double offset[3] = {0.1234567 * a, 0.2345678 * a, 0.3456789 * a};
  int64_t counts[3];
  for (int k = 0; k < 3; ++k)
    counts[k] = static_cast<int64_t>(std::ceil((s.upper[k] - s.lower[k]) / a)) + 1;
  for (int shift = 0; shift < 2; ++shift) {
    for (int64_t j = 0; j < counts[1]; ++j)
      for (int64_t i = 0; i < counts[0]; ++i) {
        double q[3] = {s.lower[0] + offset[0] + (static_cast<double>(i) + 0.5 * shift) * a,
                       s.lower[1] + offset[1] + (static_cast<double>(j) + 0.5 * shift) * a, 0.0};
        column_crossings(s, columns, q, crossings);
        if (crossings.empty()) continue;
        for (int64_t k = 0; k < counts[2]; ++k) {
          q[2] = s.lower[2] + offset[2] + (static_cast<double>(k) + 0.5 * shift) * a;
          const int32_t region = region_at(crossings, q[2]);
          if (region < 0) continue;
          if (triangles.any_within(s, q, 1.5 * h)) continue;
          out.points.insert(out.points.end(), q, q + 3);
          out.regions.push_back(region);
          ++out.lattice_seeds;
        }
      }
  }
  // Store counts for evidence.
  (void)corner_count;
}

// ------------------------------------------------------------------ builder
struct Piece {
  int32_t triangle;
  int32_t seed;
  int32_t begin;
  int32_t size;
};
struct CurveEdge {  // piece edge on a bisector
  int32_t seed;
  int32_t other;
  int32_t from;
  int32_t to;
  int32_t triangle;
};
struct CrossingRecord {
  Key triple;
  int32_t vertex;
  int32_t triangle;
};
struct FaceRecord {
  int32_t begin;
  int32_t size;
  int32_t left_seed, left_region;
  int32_t right_seed, right_region;  // right_seed -1: boundary
  int32_t facet;
};

class Builder {
 public:
  Builder(const Surface& surface, std::vector<double> seeds, std::vector<int32_t> seed_regions,
          bool automatic, double h)
      : s_(surface),
        seeds_(std::move(seeds)),
        seed_regions_(std::move(seed_regions)),
        automatic_(automatic),
        h_(h) {}

  int32_t run(phx_mc_voronoi_cells& out, double* timings) {
    auto clock = Clock::now();
    add_box();
    delaunay();
    timings[0] = seconds_since(clock);
    clock = Clock::now();
    restricted_surface();
    timings[1] = seconds_since(clock);
    clock = Clock::now();
    regions();
    timings[2] = seconds_since(clock);
    clock = Clock::now();
    assemble();
    timings[3] = seconds_since(clock);
    clock = Clock::now();
    publish(out);
    timings[4] = seconds_since(clock);
    return PHX_MC_OK;
  }

  // Seeds touched by degenerate predicates or inconsistencies.
  const std::vector<int32_t>& suspects() const { return suspects_; }
  int64_t degenerate_events() const { return degenerate_; }
  int32_t interior_count() const { return interior_; }
  int32_t boundary_cells() const { return boundary_cells_; }

  // Certificates (prove).
  void certify(phx_mc_voronoi_cells& out);
  bool certified() const { return certified_; }
  int64_t open_edges() const { return open_edges_; }
  double coverage_residual() const { return coverage_residual_; }
  double volume_residual() const { return volume_residual_; }
  double planarity_residual() const { return planarity_residual_; }

 private:
  const Surface& s_;
  std::vector<double> seeds_;
  std::vector<int32_t> seed_regions_;
  bool automatic_;
  double h_;
  int32_t interior_ = 0;  // seeds [0, interior_) are cells; the rest are box seeds
  std::vector<int32_t> tets_;
  std::vector<int32_t> tet_neighbors_;
  std::vector<int64_t> neighbor_offsets_;
  std::vector<int32_t> neighbors_;
  std::vector<int64_t> star_offsets_;
  std::vector<int32_t> star_;
  // Restricted surface.
  std::unordered_map<Key, int32_t, KeyHash> keys_;
  std::vector<double> coords_;
  std::vector<int8_t> kinds_;
  std::vector<int32_t> sources_;
  // Seeds defining each vertex: (i, j) of an edge point, the sorted triple
  // of a face point, the representative tet of a Voronoi vertex.
  std::vector<std::array<int32_t, 3>> defs_;
  std::vector<Piece> pieces_;
  std::vector<int32_t> piece_vertices_;
  std::vector<int32_t> piece_supports_;
  std::vector<CurveEdge> curve_;
  std::vector<CrossingRecord> crossings_;
  std::vector<uint8_t> has_piece_;
  std::vector<int64_t> seed_piece_offsets_;
  std::vector<int32_t> seed_pieces_;
  std::vector<int32_t> tet_vertex_;
  std::vector<int32_t> tet_region_;
  std::vector<int32_t> tet_root_;
  // Faces.
  std::vector<int32_t> loop_vertices_;
  std::vector<FaceRecord> faces_;
  std::vector<int32_t> suspects_;
  int64_t degenerate_ = 0;
  int32_t boundary_cells_ = 0;
  bool certified_ = false;
  int64_t open_edges_ = 0;
  double coverage_residual_ = 0.0;
  double volume_residual_ = 0.0;
  double planarity_residual_ = 0.0;
  std::vector<int32_t> piece_face_;
  std::vector<int8_t> inside_;
  // Published cells.
  std::vector<std::array<int32_t, 2>> cell_keys_;  // seed, region per published cell

  const double* seed(int32_t i) const { return seeds_.data() + 3 * static_cast<int64_t>(i); }

  void suspect(int32_t a, int32_t b = -1, int32_t c = -1) {
    ++degenerate_;
    for (const int32_t x : {a, b, c})
      if (x >= 0 && x < interior_) suspects_.push_back(x);
  }

  void add_box() {
    interior_ = static_cast<int32_t>(seeds_.size() / 3);
    double lower[3], upper[3];
    for (int k = 0; k < 3; ++k) {
      lower[k] = s_.lower[k];
      upper[k] = s_.upper[k];
    }
    for (int32_t i = 0; i < interior_; ++i)
      for (int k = 0; k < 3; ++k) {
        lower[k] = std::min(lower[k], seed(i)[k]);
        upper[k] = std::max(upper[k], seed(i)[k]);
      }
    double extent = 0.0;
    for (int k = 0; k < 3; ++k) extent = std::max(extent, upper[k] - lower[k]);
    const double margin = 4.0 * extent + 1.0;
    for (int corner = 0; corner < 8; ++corner) {
      for (int k = 0; k < 3; ++k) {
        const double base = (corner >> k) & 1 ? upper[k] + margin : lower[k] - margin;
        seeds_.push_back(base + 1e-3 * margin * unit_hash(97 * corner + k));
      }
    }
  }

  void delaunay() {
    const int64_t n = static_cast<int64_t>(seeds_.size() / 3);
    phx_mc_mesh* raw = nullptr;
    const int32_t status = triangulate_3d(n, seeds_.data(), nullptr, false, false, &raw);
    std::unique_ptr<phx_mc_mesh> mesh(raw);
    if (status != PHX_MC_OK) throw Failure(status);
    for (int64_t v = 0; v < n; ++v)
      if (mesh->vertex_map[v] != v) throw Failure(PHX_MC_DEGENERATE_INPUT);  // duplicate seed
    tets_ = std::move(mesh->cells);
    const int64_t count = static_cast<int64_t>(tets_.size() / 4);
    // Face adjacency by sorting.
    std::vector<std::array<int32_t, 4>> rows;
    rows.reserve(4 * count);
    for (int64_t t = 0; t < count; ++t)
      for (int k = 0; k < 4; ++k) {
        std::array<int32_t, 3> f{};
        int c = 0;
        for (int v = 0; v < 4; ++v)
          if (v != k) f[c++] = tets_[4 * t + v];
        std::sort(f.begin(), f.end());
        rows.push_back({f[0], f[1], f[2], static_cast<int32_t>(4 * t + k)});
      }
    std::sort(rows.begin(), rows.end());
    tet_neighbors_.assign(4 * count, -1);
    for (std::size_t r = 0; r + 1 < rows.size(); ++r) {
      if (rows[r][0] == rows[r + 1][0] && rows[r][1] == rows[r + 1][1] &&
          rows[r][2] == rows[r + 1][2]) {
        tet_neighbors_[rows[r][3]] = rows[r + 1][3] / 4;
        tet_neighbors_[rows[r + 1][3]] = rows[r][3] / 4;
        ++r;
      }
    }
    // Seed adjacency and stars.
    std::vector<std::pair<int32_t, int32_t>> pairs;
    pairs.reserve(12 * count);
    star_offsets_.assign(n + 1, 0);
    for (int64_t t = 0; t < count; ++t)
      for (int a = 0; a < 4; ++a) {
        ++star_offsets_[tets_[4 * t + a] + 1];
        for (int b = 0; b < 4; ++b)
          if (a != b) pairs.emplace_back(tets_[4 * t + a], tets_[4 * t + b]);
      }
    for (int64_t v = 0; v < n; ++v) star_offsets_[v + 1] += star_offsets_[v];
    star_.resize(star_offsets_[n]);
    std::vector<int64_t> cursor(star_offsets_.begin(), star_offsets_.end() - 1);
    for (int64_t t = 0; t < count; ++t)
      for (int a = 0; a < 4; ++a) star_[cursor[tets_[4 * t + a]]++] = static_cast<int32_t>(t);
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    neighbor_offsets_.assign(n + 1, 0);
    neighbors_.resize(pairs.size());
    for (std::size_t k = 0; k < pairs.size(); ++k) {
      ++neighbor_offsets_[pairs[k].first + 1];
      neighbors_[k] = pairs[k].second;
    }
    for (int64_t v = 0; v < n; ++v) neighbor_offsets_[v + 1] += neighbor_offsets_[v];
  }

  // ------------------------------------------------ predicates on a triangle
  struct TriangleFrame {
    const double* c[3];
    int32_t t;
  };
  using Row = std::array<Bounded, 3>;
  Row row(const TriangleFrame& f, int32_t i, int32_t x) const {
    Row r;
    for (int k = 0; k < 3; ++k)
      r[k] = bounded_squared_distance(f.c[k], seed(i)) - bounded_squared_distance(f.c[k], seed(x));
    return r;
  }
  std::array<Expansion, 3> exact_row(const TriangleFrame& f, int32_t i, int32_t x) const {
    std::array<Expansion, 3> r;
    for (int k = 0; k < 3; ++k)
      r[k] = exact_squared_distance(f.c[k], seed(i)) - exact_squared_distance(f.c[k], seed(x));
    return r;
  }
  // Constraint c < 0: barycentric coordinate (-1 - c) vanishes; c >= 0: bisector (i, c).
  Row form(const TriangleFrame& f, int32_t i, int32_t c) const {
    if (c < 0) {
      Row r{Bounded{0, 0}, Bounded{0, 0}, Bounded{0, 0}};
      r[-1 - c] = Bounded{1.0, 0.0};
      return r;
    }
    return row(f, i, c);
  }
  std::array<Expansion, 3> exact_form(const TriangleFrame& f, int32_t i, int32_t c) const {
    if (c < 0) {
      std::array<Expansion, 3> r;
      r[-1 - c] = Expansion(1.0);
      return r;
    }
    return exact_row(f, i, c);
  }
  // Sign of g_il at the vertex (c1, c2) of triangle f in cell i.
  int side(const TriangleFrame& f, int32_t i, int32_t c1, int32_t c2, int32_t l, const Row& rl) {
    if (c1 < 0 && c2 < 0) {
      const int corner = 3 - (-1 - c1) - (-1 - c2);
      const int sign = decided_sign(rl[corner]);
      if (sign != 2) return sign;
      const Expansion exact = exact_squared_distance(f.c[corner], seed(i)) -
                              exact_squared_distance(f.c[corner], seed(l));
      return exact.sign();
    }
    const Row a = form(f, i, c1), b = form(f, i, c2);
    const Row x = cross3(a, b);
    const Bounded num = dot3(rl, x);
    const Bounded den = x[0] + x[1] + x[2];
    const int sn = decided_sign(num), sd = decided_sign(den);
    if (sn != 2 && sd != 2) return sn * sd;
    const auto ea = exact_form(f, i, c1), eb = exact_form(f, i, c2), el = exact_row(f, i, l);
    const auto ex = cross3(ea, eb);
    const Expansion en = dot3(el, ex);
    const Expansion ed = ex[0] + ex[1] + ex[2];
    return en.sign() * ed.sign();
  }

  // Global vertex of the symbolic vertex (c1, c2) of triangle t in cell i.
  int32_t vertex_of(const TriangleFrame& f, int32_t i, int32_t c1, int32_t c2) {
    Key key;
    int8_t kind;
    int32_t source0, source1 = -1;
    const int32_t* tri = s_.in->triangles + 3 * static_cast<int64_t>(f.t);
    if (c1 < 0 && c2 < 0) {
      const int corner = 3 - (-1 - c1) - (-1 - c2);
      key = corner_key(tri[corner]);
      kind = kSurfaceVertex;
      source0 = tri[corner];
    } else if (c1 < 0 || c2 < 0) {
      const int32_t e = c1 < 0 ? c1 : c2, x = c1 < 0 ? c2 : c1;
      const int32_t edge = s_.triangle_edges[3 * static_cast<int64_t>(f.t) + (-1 - e)];
      key = edge_key(edge, i, x);
      kind = kSurfaceEdge;
      source0 = s_.edges[2 * edge];
      source1 = s_.edges[2 * edge + 1];
    } else {
      key = face_key(f.t, sorted3(i, c1, c2));
      kind = kSurfaceFace;
      source0 = f.t;
    }
    const auto [found, inserted] = keys_.try_emplace(key, static_cast<int32_t>(kinds_.size()));
    if (!inserted) return found->second;
    // Barycentric coordinates from the cross product of the two forms.
    const Row a = form(f, i, c1), b = form(f, i, c2);
    const Row x = cross3(a, b);
    double w[3] = {x[0].v, x[1].v, x[2].v};
    const double sum = w[0] + w[1] + w[2];
    double p[3] = {0, 0, 0};
    for (int k = 0; k < 3; ++k) {
      const double weight = sum != 0.0 ? w[k] / sum : 1.0 / 3.0;
      for (int axis = 0; axis < 3; ++axis) p[axis] += weight * f.c[k][axis];
    }
    if (kind == kSurfaceVertex) std::copy_n(s_.point(source0), 3, p);
    coords_.insert(coords_.end(), p, p + 3);
    kinds_.push_back(kind);
    sources_.push_back(source0);
    sources_.push_back(source1);
    if (kind == kSurfaceFace) {
      const auto triple = sorted3(i, c1, c2);
      defs_.push_back({triple[0], triple[1], triple[2]});
    } else {
      defs_.push_back({i, c1 < 0 ? c2 : c1, -1});
    }
    if (kind == kSurfaceFace) {
      crossings_.push_back({triple_key(sorted3(i, c1, c2)), found->second, f.t});
    }
    return found->second;
  }

  // Exact nearest seed of corner p, walking from `start`; ties go to the
  // smaller index, the rule every clip applies.
  int32_t nearest(const double* p, int32_t start) {
    int32_t current = start;
    while (true) {
      int32_t next = -1;
      for (int64_t e = neighbor_offsets_[current]; e < neighbor_offsets_[current + 1]; ++e) {
        const int32_t y = neighbors_[e];
        const int32_t best = next < 0 ? current : next;
        const Bounded g = bounded_squared_distance(p, seed(best)) - bounded_squared_distance(p, seed(y));
        int sign = decided_sign(g);
        if (sign == 2) {
          sign = (exact_squared_distance(p, seed(best)) - exact_squared_distance(p, seed(y))).sign();
        }
        if (sign > 0 || (sign == 0 && y < best)) next = y;
      }
      if (next < 0) return current;
      current = next;
    }
  }

  struct PolyVertex {
    int32_t c1, c2;
    int32_t support;  // constraint of the edge leaving this vertex
  };

  // Clips triangle f by the cell of seed i; false when empty.
  bool clip(const TriangleFrame& f, int32_t i, std::vector<PolyVertex>& poly,
            std::vector<PolyVertex>& scratch) {
    poly.clear();
    poly.push_back({-2, -3, -3});  // corner 0 (b0 = 1): next edge opposite corner 2
    poly.push_back({-1, -3, -1});  // corner 1
    poly.push_back({-1, -2, -2});  // corner 2
    for (int64_t e = neighbor_offsets_[i]; e < neighbor_offsets_[i + 1]; ++e) {
      const int32_t l = neighbors_[e];
      const Row rl = row(f, i, l);
      // The polygon lies in the triangle: corners decide when they agree.
      const int s0 = decided_sign(rl[0]), s1 = decided_sign(rl[1]), s2 = decided_sign(rl[2]);
      if (s0 == -1 && s1 == -1 && s2 == -1) continue;
      std::vector<int8_t>& inside = inside_;
      inside.resize(poly.size());
      bool any_in = false, any_out = false;
      for (std::size_t k = 0; k < poly.size(); ++k) {
        int sign = side(f, i, poly[k].c1, poly[k].c2, l, rl);
        if (sign == 0) {
          suspect(i, l);
          sign = i < l ? -1 : 1;
        }
        inside[k] = sign < 0 ? 1 : 0;
        any_in = any_in || inside[k];
        any_out = any_out || !inside[k];
      }
      if (!any_out) continue;
      if (!any_in) return false;
      scratch.clear();
      const std::size_t n = poly.size();
      for (std::size_t k = 0; k < n; ++k) {
        const PolyVertex& cur = poly[k];
        const bool next_in = inside[(k + 1) % n] != 0;
        if (inside[k]) {
          scratch.push_back(cur);
          if (!next_in) scratch.push_back({cur.support, l, l});
        } else if (next_in) {
          scratch.push_back({cur.support, l, cur.support});
        }
      }
      poly.swap(scratch);
    }
    return poly.size() >= 3;
  }

  void restricted_surface() {
    const int64_t m = s_.in->triangle_count;
    has_piece_.assign(interior_, 0);
    keys_.reserve(static_cast<std::size_t>(4 * m));
    std::vector<PolyVertex> poly, scratch;
    std::vector<int32_t> queue, visited;
    int32_t hint = 0;
    for (int64_t t = 0; t < m; ++t) {
      TriangleFrame f{{s_.corner(t, 0), s_.corner(t, 1), s_.corner(t, 2)}, static_cast<int32_t>(t)};
      const int32_t owner = nearest(f.c[0], hint);
      hint = owner;
      queue.assign(1, owner);
      visited.assign(1, owner);
      for (std::size_t q = 0; q < queue.size(); ++q) {
        const int32_t i = queue[q];
        if (!clip(f, i, poly, scratch)) continue;
        if (i >= interior_) throw Failure(PHX_MC_UNRESOLVED);  // a box seed reaches the solid
        Piece piece{static_cast<int32_t>(t), i, static_cast<int32_t>(piece_vertices_.size()),
                    static_cast<int32_t>(poly.size())};
        for (const PolyVertex& v : poly) {
          piece_vertices_.push_back(vertex_of(f, i, v.c1, v.c2));
          piece_supports_.push_back(v.support);
        }
        for (int32_t k = 0; k < piece.size; ++k) {
          const int32_t support = piece_supports_[piece.begin + k];
          if (support < 0) continue;
          curve_.push_back({i, support, piece_vertices_[piece.begin + k],
                            piece_vertices_[piece.begin + (k + 1) % piece.size],
                            static_cast<int32_t>(t)});
          if (std::find(visited.begin(), visited.end(), support) == visited.end()) {
            visited.push_back(support);
            queue.push_back(support);
          }
        }
        has_piece_[i] = 1;
        pieces_.push_back(piece);
      }
    }
    // Pieces by seed.
    seed_piece_offsets_.assign(interior_ + 1, 0);
    for (const Piece& p : pieces_) ++seed_piece_offsets_[p.seed + 1];
    for (int32_t i = 0; i < interior_; ++i) seed_piece_offsets_[i + 1] += seed_piece_offsets_[i];
    seed_pieces_.resize(pieces_.size());
    std::vector<int64_t> cursor(seed_piece_offsets_.begin(), seed_piece_offsets_.end() - 1);
    for (std::size_t p = 0; p < pieces_.size(); ++p)
      seed_pieces_[cursor[pieces_[p].seed]++] = static_cast<int32_t>(p);
    std::sort(curve_.begin(), curve_.end(), [](const CurveEdge& a, const CurveEdge& b) {
      return a.seed != b.seed ? a.seed < b.seed : a.other < b.other;
    });
    std::sort(crossings_.begin(), crossings_.end(),
              [](const CrossingRecord& a, const CrossingRecord& b) { return a.triple < b.triple; });
  }

  // ------------------------------------------------ Voronoi vertices
  void circumcenter(int32_t t, double* out) const {
    const double* p0 = seed(tets_[4 * t]);
    double a[3], b[3], c[3];
    geometry::difference(seed(tets_[4 * t + 1]), p0, a);
    geometry::difference(seed(tets_[4 * t + 2]), p0, b);
    geometry::difference(seed(tets_[4 * t + 3]), p0, c);
    double bc[3], ca[3], ab[3];
    geometry::cross(b, c, bc);
    geometry::cross(c, a, ca);
    geometry::cross(a, b, ab);
    const double det = 2.0 * geometry::dot(a, bc);
    const double la = geometry::dot(a, a), lb = geometry::dot(b, b), lc = geometry::dot(c, c);
    for (int k = 0; k < 3; ++k) out[k] = p0[k] + (la * bc[k] + lb * ca[k] + lc * ab[k]) / det;
  }
  // One vertex per class of exactly cospherical adjacent tets (their
  // circumcentres coincide), so given symmetric seeds leave no zero edges.
  int32_t tet_vertex(int32_t t) {
    while (tet_root_[t] != t) t = tet_root_[t] = tet_root_[tet_root_[t]];
    if (tet_vertex_[t] >= 0) return tet_vertex_[t];
    double p[3];
    circumcenter(t, p);
    tet_vertex_[t] = static_cast<int32_t>(kinds_.size());
    coords_.insert(coords_.end(), p, p + 3);
    kinds_.push_back(kVoronoi);
    defs_.push_back({t, -1, -1});
    sources_.push_back(t);
    sources_.push_back(-1);
    return tet_vertex_[t];
  }
  // Exact coordinates N_k / D of vertex v from its defining seeds and surface.
  void exact_vertex(int32_t v, std::array<Expansion, 3>& n, Expansion& d) const {
    const auto& def = defs_[v];
    switch (kinds_[v]) {
      case kSurfaceVertex: {
        const double* p = s_.point(sources_[2 * static_cast<int64_t>(v)]);
        for (int k = 0; k < 3; ++k) n[k] = Expansion(p[k]);
        d = Expansion(1.0);
        return;
      }
      case kSurfaceEdge: {
        // The edge point a + g(a) / (g(a) - g(b)) (b - a) on bisector (i, j).
        const double* a = s_.point(sources_[2 * static_cast<int64_t>(v)]);
        const double* b = s_.point(sources_[2 * static_cast<int64_t>(v) + 1]);
        const Expansion ga = exact_squared_distance(a, seed(def[0])) -
                             exact_squared_distance(a, seed(def[1]));
        const Expansion gb = exact_squared_distance(b, seed(def[0])) -
                             exact_squared_distance(b, seed(def[1]));
        d = compressed(ga - gb);
        for (int k = 0; k < 3; ++k) {
          n[k] = compressed(d.scaled(a[k]) + ga * Expansion::difference(b[k], a[k]));
        }
        return;
      }
      case kSurfaceFace: {
        // Barycentric weights cross(r_ij, r_ik) of the triangle corners.
        const int32_t t = sources_[2 * static_cast<int64_t>(v)];
        const TriangleFrame f{{s_.corner(t, 0), s_.corner(t, 1), s_.corner(t, 2)}, t};
        auto rj = exact_row(f, def[0], def[1]);
        auto rk = exact_row(f, def[0], def[2]);
        for (int k = 0; k < 3; ++k) {
          rj[k] = compressed(rj[k]);
          rk[k] = compressed(rk[k]);
        }
        auto weights = cross3(rj, rk);
        for (auto& w : weights) w = compressed(w);
        d = compressed(weights[0] + weights[1] + weights[2]);
        for (int k = 0; k < 3; ++k) {
          n[k] = compressed(weights[0].scaled(f.c[0][k]) + weights[1].scaled(f.c[1][k]) +
                            weights[2].scaled(f.c[2][k]));
        }
        return;
      }
      default: {
        // Circumcentre p0 + (|a|^2 b x c + |b|^2 c x a + |c|^2 a x b) / (2 a . b x c).
        const int32_t* tet = tets_.data() + 4 * static_cast<int64_t>(def[0]);
        const double* p0 = seed(tet[0]);
        std::array<std::array<Expansion, 3>, 3> e;
        for (int r = 0; r < 3; ++r)
          for (int k = 0; k < 3; ++k) e[r][k] = Expansion::difference(seed(tet[r + 1])[k], p0[k]);
        auto bc = cross3(e[1], e[2]), ca = cross3(e[2], e[0]), ab = cross3(e[0], e[1]);
        for (int k = 0; k < 3; ++k) {
          bc[k] = compressed(bc[k]);
          ca[k] = compressed(ca[k]);
          ab[k] = compressed(ab[k]);
        }
        const Expansion la = compressed(dot3(e[0], e[0]));
        const Expansion lb = compressed(dot3(e[1], e[1]));
        const Expansion lc = compressed(dot3(e[2], e[2]));
        d = compressed(dot3(e[0], bc).scaled(2.0));
        for (int k = 0; k < 3; ++k) {
          n[k] = compressed(d.scaled(p0[k]) + la * bc[k] + lb * ca[k] + lc * ab[k]);
        }
        return;
      }
    }
  }
  void cospherical_classes() {
    const int64_t count = static_cast<int64_t>(tets_.size() / 4);
    tet_root_.resize(count);
    std::iota(tet_root_.begin(), tet_root_.end(), 0);
    for (int64_t t = 0; t < count; ++t) {
      const int32_t* v = tets_.data() + 4 * t;
      for (int k = 0; k < 4; ++k) {
        const int32_t u = tet_neighbors_[4 * t + k];
        if (u < t) continue;
        int32_t apex = -1;
        for (int w = 0; w < 4; ++w) {
          const int32_t x = tets_[4 * static_cast<int64_t>(u) + w];
          if (x != v[0] && x != v[1] && x != v[2] && x != v[3]) apex = x;
        }
        if (insphere(seed(v[0]), seed(v[1]), seed(v[2]), seed(v[3]), seed(apex)) != 0) continue;
        int32_t a = static_cast<int32_t>(t), b = u;
        while (tet_root_[a] != a) a = tet_root_[a];
        while (tet_root_[b] != b) b = tet_root_[b];
        if (a != b) tet_root_[std::max(a, b)] = std::min(a, b);
      }
    }
  }

  // Sign of n_t . ((s_j - s_i) x (s_k - s_i)).
  int piercing_sign(int32_t t, const std::array<int32_t, 3>& tri) const {
    const double* q0 = s_.corner(t, 0);
    const double* q1 = s_.corner(t, 1);
    const double* q2 = s_.corner(t, 2);
    const double* a = seed(tri[0]);
    const double* b = seed(tri[1]);
    const double* c = seed(tri[2]);
    std::array<Bounded, 3> A, B, C, D;
    for (int k = 0; k < 3; ++k) {
      A[k] = bounded_difference(q1[k], q0[k]);
      B[k] = bounded_difference(q2[k], q0[k]);
      C[k] = bounded_difference(b[k], a[k]);
      D[k] = bounded_difference(c[k], a[k]);
    }
    const Bounded value = dot3(A, C) * dot3(B, D) - dot3(A, D) * dot3(B, C);
    const int sign = decided_sign(value);
    if (sign != 2) return sign;
    std::array<Expansion, 3> EA, EB, EC, ED;
    for (int k = 0; k < 3; ++k) {
      EA[k] = Expansion::difference(q1[k], q0[k]);
      EB[k] = Expansion::difference(q2[k], q0[k]);
      EC[k] = Expansion::difference(b[k], a[k]);
      ED[k] = Expansion::difference(c[k], a[k]);
    }
    return (dot3(EA, EC) * dot3(EB, ED) - dot3(EA, ED) * dot3(EB, EC)).sign();
  }

  struct Pierce {
    double along;
    int32_t vertex;
    int32_t triangle;
    int direction;  // +1: from back to front of the triangle
  };
  // Piercing points of the Voronoi edge of Delaunay triangle `tri` from the
  // circumcentre of tet `from` towards that of the tet whose apex is `apex`.
  void pierces(const std::array<int32_t, 3>& tri, int32_t from, int32_t apex,
               std::vector<Pierce>& out) {
    out.clear();
    const Key key = triple_key(tri);
    auto lo = std::lower_bound(crossings_.begin(), crossings_.end(), key,
                               [](const CrossingRecord& r, const Key& k) { return r.triple < k; });
    if (lo == crossings_.end() || !(lo->triple == key)) return;
    // Orientation of the edge: towards the apex side of the triangle plane.
    const int towards = orient3d(seed(tri[0]), seed(tri[1]), seed(tri[2]), seed(apex));
    if (towards == 0) throw Failure(PHX_MC_INTERNAL_ERROR);
    double origin[3], n[3], u[3], v[3];
    circumcenter(from, origin);
    geometry::difference(seed(tri[1]), seed(tri[0]), u);
    geometry::difference(seed(tri[2]), seed(tri[0]), v);
    geometry::cross(u, v, n);
    for (auto it = lo; it != crossings_.end() && it->triple == key; ++it) {
      const double* x = coords_.data() + 3 * static_cast<int64_t>(it->vertex);
      double d[3];
      geometry::difference(x, origin, d);
      int sign = piercing_sign(it->triangle, tri);
      if (sign == 0) {
        suspect(tri[0], tri[1], tri[2]);
        sign = 1;
      }
      out.push_back({towards * geometry::dot(d, n), it->vertex, it->triangle, sign * towards});
    }
    std::sort(out.begin(), out.end(), [](const Pierce& a, const Pierce& b) {
      return a.along != b.along ? a.along < b.along : a.vertex < b.vertex;
    });
  }
  int32_t transition(int32_t region, const Pierce& p, int32_t a, int32_t b, int32_t c) {
    const int32_t from = p.direction > 0 ? s_.back(p.triangle) : s_.front(p.triangle);
    const int32_t to = p.direction > 0 ? s_.front(p.triangle) : s_.back(p.triangle);
    if (from != region) suspect(a, b, c);
    return to;
  }

  void regions() {
    const int64_t count = static_cast<int64_t>(tets_.size() / 4);
    tet_vertex_.assign(count, -1);
    cospherical_classes();
    tet_region_.assign(count, -2);
    // A box seed's cell misses the solid: its tets are exterior.
    const int32_t anchor = star_[star_offsets_[interior_]];
    tet_region_[anchor] = -1;
    std::vector<int32_t> queue{anchor};
    std::vector<Pierce> list;
    for (std::size_t q = 0; q < queue.size(); ++q) {
      const int32_t t = queue[q];
      for (int k = 0; k < 4; ++k) {
        const int32_t u = tet_neighbors_[4 * static_cast<int64_t>(t) + k];
        if (u < 0) continue;
        std::array<int32_t, 3> tri{};
        int c = 0;
        for (int v = 0; v < 4; ++v)
          if (v != k) tri[c++] = tets_[4 * static_cast<int64_t>(t) + v];
        std::sort(tri.begin(), tri.end());
        int32_t apex = -1;
        for (int v = 0; v < 4; ++v) {
          const int32_t w = tets_[4 * static_cast<int64_t>(u) + v];
          if (w != tri[0] && w != tri[1] && w != tri[2]) apex = w;
        }
        int32_t region = tet_region_[t];
        pierces(tri, t, apex, list);
        for (const Pierce& p : list) region = transition(region, p, tri[0], tri[1], tri[2]);
        if (tet_region_[u] == -2) {
          tet_region_[u] = region;
          queue.push_back(u);
        } else if (tet_region_[u] != region) {
          suspect(tri[0], tri[1], tri[2]);
        }
      }
    }
  }

  // ------------------------------------------------ faces
  // Ring of tets around Delaunay edge (i, j), counterclockwise about s_j - s_i;
  // apex[q] is shared by ring[q - 1] and ring[q].
  void ring(int32_t i, int32_t j, int32_t start, std::vector<int32_t>& tets,
            std::vector<int32_t>& apexes) {
    tets.clear();
    apexes.clear();
    int32_t t = start;
    int32_t a = -1, b = -1;
    for (int v = 0; v < 4; ++v) {
      const int32_t w = tets_[4 * static_cast<int64_t>(t) + v];
      if (w == i || w == j) continue;
      (a < 0 ? a : b) = w;
    }
    if (orient3d(seed(i), seed(j), seed(a), seed(b)) < 0) std::swap(a, b);
    // tets[q] holds apexes (apexes[q], apexes[q + 1]).
    apexes.push_back(a);
    while (true) {
      tets.push_back(t);
      apexes.push_back(b);
      // Next tet across the face opposite a (it keeps i, j, b).
      int32_t slot = -1;
      for (int v = 0; v < 4; ++v)
        if (tets_[4 * static_cast<int64_t>(t) + v] == a) slot = v;
      const int32_t u = tet_neighbors_[4 * static_cast<int64_t>(t) + slot];
      if (u < 0) throw Failure(PHX_MC_INTERNAL_ERROR);
      if (u == start) break;
      int32_t c = -1;
      for (int v = 0; v < 4; ++v) {
        const int32_t w = tets_[4 * static_cast<int64_t>(u) + v];
        if (w != i && w != j && w != b) c = w;
      }
      t = u;
      a = b;
      b = c;
    }
    apexes.pop_back();  // the last apex repeats the first
  }

  void emit_face(const std::vector<int32_t>& loop, int32_t left_seed, int32_t left_region,
                 int32_t right_seed, int32_t right_region, int32_t facet) {
    FaceRecord face{static_cast<int32_t>(loop_vertices_.size()), static_cast<int32_t>(loop.size()),
                    left_seed, left_region, right_seed, right_region, facet};
    loop_vertices_.insert(loop_vertices_.end(), loop.begin(), loop.end());
    faces_.push_back(face);
  }

  // Simple loops of a directed edge set, false at a pinch or open chain.
  static bool chain(std::vector<std::pair<int32_t, int32_t>>& edges,
                    std::vector<std::vector<int32_t>>& loops) {
    loops.clear();
    std::sort(edges.begin(), edges.end());
    for (std::size_t k = 1; k < edges.size(); ++k)
      if (edges[k].first == edges[k - 1].first) return false;
    std::vector<uint8_t> used(edges.size(), 0);
    for (std::size_t k = 0; k < edges.size(); ++k) {
      if (used[k]) continue;
      std::vector<int32_t> loop;
      std::size_t current = k;
      while (!used[current]) {
        used[current] = 1;
        loop.push_back(edges[current].first);
        const int32_t next = edges[current].second;
        const auto found = std::lower_bound(edges.begin(), edges.end(),
                                            std::make_pair(next, INT32_MIN));
        if (found == edges.end() || found->first != next) return false;
        current = static_cast<std::size_t>(found - edges.begin());
      }
      if (current != k || loop.size() < 3) return false;
      loops.push_back(std::move(loop));
    }
    return true;
  }

  void newell(const std::vector<int32_t>& loop, double* n) const {
    n[0] = n[1] = n[2] = 0.0;
    for (std::size_t k = 0; k < loop.size(); ++k) {
      const double* a = coords_.data() + 3 * static_cast<int64_t>(loop[k]);
      const double* b = coords_.data() + 3 * static_cast<int64_t>(loop[(k + 1) % loop.size()]);
      n[0] += (a[1] - b[1]) * (a[2] + b[2]);
      n[1] += (a[2] - b[2]) * (a[0] + b[0]);
      n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
  }

  void bisector_faces() {
    const int64_t count = static_cast<int64_t>(tets_.size() / 4);
    // Unique Delaunay edges with a representative tet, in (i, j) order.
    std::vector<std::array<int32_t, 3>> edges;
    edges.reserve(7 * count);
    for (int64_t t = 0; t < count; ++t)
      for (int a = 0; a < 4; ++a)
        for (int b = a + 1; b < 4; ++b) {
          int32_t i = tets_[4 * t + a], j = tets_[4 * t + b];
          if (i > j) std::swap(i, j);
          if (i >= interior_) continue;
          edges.push_back({i, j, static_cast<int32_t>(t)});
        }
    std::sort(edges.begin(), edges.end());
    std::vector<int32_t> tets, apexes, loop;
    std::vector<Pierce> list;
    std::vector<std::array<int32_t, 3>> directed;  // region, from, to
    std::vector<std::pair<int32_t, int32_t>> region_edges;
    std::vector<std::vector<int32_t>> loops;
    for (std::size_t k = 0; k < edges.size(); ++k) {
      if (k > 0 && edges[k][0] == edges[k - 1][0] && edges[k][1] == edges[k - 1][1]) continue;
      const int32_t i = edges[k][0], j = edges[k][1];
      ring(i, j, edges[k][2], tets, apexes);
      const bool box = j >= interior_;
      const bool clipped = has_piece_[i] || (!box && has_piece_[j]);
      if (!clipped) {
        const int32_t region = tet_region_[tets[0]];
        if (region < 0) continue;
        if (box) {
          suspect(i);
          continue;
        }
        loop.clear();
        for (const int32_t t : tets) {
          const int32_t vertex = tet_vertex(t);
          if (loop.empty() || loop.back() != vertex) loop.push_back(vertex);
        }
        while (loop.size() > 1 && loop.front() == loop.back()) loop.pop_back();
        if (loop.size() >= 3) emit_face(loop, i, region, j, region, -1);
        continue;
      }
      directed.clear();
      const std::size_t m = tets.size();
      for (std::size_t q = 0; q < m; ++q) {
        const int32_t from = tets[q], to = tets[(q + 1) % m];
        const int32_t apex_to = apexes[(q + 2) % m == 0 ? 0 : (q + 2) % m];
        std::array<int32_t, 3> tri = sorted3(i, j, apexes[(q + 1) % m]);
        pierces(tri, from, apex_to, list);
        int32_t region = tet_region_[from];
        int32_t current = -1;
        for (const Pierce& p : list) {
          if (region >= 0) {
            if (current < 0) current = tet_vertex(from);
            directed.push_back({region, current, p.vertex});
          }
          region = transition(region, p, tri[0], tri[1], tri[2]);
          current = p.vertex;
        }
        if (region != tet_region_[to]) suspect(tri[0], tri[1], tri[2]);
        if (region >= 0) {
          if (current < 0) current = tet_vertex(from);
          if (current != tet_vertex(to)) directed.push_back({region, current, tet_vertex(to)});
        }
      }
      // The surface curve on the bisector, from i's pieces.
      CurveEdge probe{i, j, 0, 0, 0};
      auto lo = std::lower_bound(curve_.begin(), curve_.end(), probe,
                                 [](const CurveEdge& a, const CurveEdge& b) {
                                   return a.seed != b.seed ? a.seed < b.seed : a.other < b.other;
                                 });
      for (auto it = lo; it != curve_.end() && it->seed == i && it->other == j; ++it) {
        directed.push_back({s_.back(it->triangle), it->to, it->from});
        if (s_.front(it->triangle) >= 0) directed.push_back({s_.front(it->triangle), it->from, it->to});
      }
      std::sort(directed.begin(), directed.end());
      for (std::size_t a = 0; a < directed.size();) {
        std::size_t b = a;
        while (b < directed.size() && directed[b][0] == directed[a][0]) ++b;
        const int32_t region = directed[a][0];
        region_edges.clear();
        for (std::size_t c = a; c < b; ++c) region_edges.emplace_back(directed[c][1], directed[c][2]);
        a = b;
        if (region < 0) continue;
        if (box) {
          suspect(i);
          continue;
        }
        if (!chain(region_edges, loops)) {
          suspect(i, j);
          continue;
        }
        double axis[3];
        geometry::difference(seed(j), seed(i), axis);
        for (const auto& l : loops) {
          double n[3];
          newell(l, n);
          if (geometry::dot(n, axis) <= 0.0) {
            // A hole or an inverted loop: not a simple face.
            suspect(i, j);
            continue;
          }
          emit_face(l, i, region, j, region, -1);
        }
      }
    }
  }

  void surface_faces() {
    std::vector<std::pair<int32_t, int32_t>> edges;
    std::vector<std::vector<int32_t>> loops;
    std::vector<int32_t> loop;
    for (int32_t i = 0; i < interior_; ++i) {
      const int64_t begin = seed_piece_offsets_[i], end = seed_piece_offsets_[i + 1];
      if (begin == end) continue;
      std::vector<int32_t> order(seed_pieces_.begin() + begin, seed_pieces_.begin() + end);
      auto facet_key = [&](int32_t p) {
        const int32_t t = pieces_[p].triangle;
        return std::array<int32_t, 4>{s_.plane_class[t], s_.front(t), s_.back(t), t};
      };
      std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return facet_key(a) < facet_key(b);
      });
      for (std::size_t a = 0; a < order.size();) {
        std::size_t b = a;
        const auto ka = facet_key(order[a]);
        while (b < order.size()) {
          const auto kb = facet_key(order[b]);
          if (kb[0] != ka[0] || kb[1] != ka[1] || kb[2] != ka[2]) break;
          ++b;
        }
        const int32_t t = pieces_[order[a]].triangle;
        const int32_t back = s_.back(t), front = s_.front(t);
        const int32_t left_region = back >= 0 ? back : front;
        const bool reversed = back < 0;
        const int32_t right_region = back >= 0 ? front : -1;
        auto emit = [&](std::vector<int32_t> l, std::size_t first, std::size_t last) {
          if (reversed) std::reverse(l.begin(), l.end());
          for (std::size_t c = first; c < last; ++c)
            piece_face_[order[c]] = static_cast<int32_t>(faces_.size());
          emit_face(l, i, left_region, right_region >= 0 ? i : -1, right_region,
                    s_.in->facets[t]);
        };
        bool merged = false;
        if (b - a > 1) {
          // Cancel opposite edges among the pieces in one plane.
          edges.clear();
          for (std::size_t c = a; c < b; ++c) {
            const Piece& p = pieces_[order[c]];
            for (int32_t q = 0; q < p.size; ++q)
              edges.emplace_back(piece_vertices_[p.begin + q],
                                 piece_vertices_[p.begin + (q + 1) % p.size]);
          }
          std::sort(edges.begin(), edges.end());
          std::vector<std::pair<int32_t, int32_t>> boundary;
          for (const auto& e : edges) {
            if (!std::binary_search(edges.begin(), edges.end(), std::make_pair(e.second, e.first)))
              boundary.push_back(e);
          }
          if (chain(boundary, loops)) {
            bool positive = true;
            for (const auto& l : loops) {
              double n[3];
              newell(l, n);
              positive = positive && geometry::dot(n, &s_.normals[3 * static_cast<int64_t>(t)]) > 0.0;
            }
            if (positive && loops.size() == 1) {
              emit(loops[0], a, b);
              merged = true;
            }
          }
        }
        if (!merged) {
          for (std::size_t c = a; c < b; ++c) {
            const Piece& p = pieces_[order[c]];
            loop.assign(piece_vertices_.begin() + p.begin, piece_vertices_.begin() + p.begin + p.size);
            emit(loop, c, c + 1);
          }
        }
        a = b;
      }
    }
  }

  void assemble() {
    bisector_faces();
    piece_face_.assign(pieces_.size(), -1);
    surface_faces();
  }

  // ------------------------------------------------ publication
  void publish(phx_mc_voronoi_cells& out);
};

void Builder::publish(phx_mc_voronoi_cells& out) {
  // Cells: components of (seed, region) through shared edges.
  struct Side {
    int32_t seed, region, face;
    bool reversed;
  };
  std::vector<Side> sides;
  sides.reserve(2 * faces_.size());
  for (std::size_t f = 0; f < faces_.size(); ++f) {
    const FaceRecord& r = faces_[f];
    sides.push_back({r.left_seed, r.left_region, static_cast<int32_t>(f), false});
    if (r.right_seed >= 0) sides.push_back({r.right_seed, r.right_region, static_cast<int32_t>(f), true});
  }
  std::sort(sides.begin(), sides.end(), [](const Side& a, const Side& b) {
    if (a.seed != b.seed) return a.seed < b.seed;
    if (a.region != b.region) return a.region < b.region;
    return a.face < b.face;
  });
  std::vector<int32_t> face_left(faces_.size(), -1), face_right(faces_.size(), -1);
  std::vector<int32_t> parent;
  std::vector<std::pair<uint64_t, int32_t>> edge_owner;
  cell_keys_.clear();
  std::vector<int32_t> used_seed(interior_, -1);
  boundary_cells_ = 0;
  for (std::size_t a = 0; a < sides.size();) {
    std::size_t b = a;
    while (b < sides.size() && sides[b].seed == sides[a].seed && sides[b].region == sides[a].region) ++b;
    const std::size_t n = b - a;
    parent.resize(n);
    std::iota(parent.begin(), parent.end(), 0);
    auto root = [&](int32_t x) {
      while (parent[x] != x) x = parent[x] = parent[parent[x]];
      return x;
    };
    edge_owner.clear();
    for (std::size_t c = 0; c < n; ++c) {
      const FaceRecord& r = faces_[sides[a + c].face];
      for (int32_t q = 0; q < r.size; ++q) {
        uint32_t u = static_cast<uint32_t>(loop_vertices_[r.begin + q]);
        uint32_t v = static_cast<uint32_t>(loop_vertices_[r.begin + (q + 1) % r.size]);
        if (u > v) std::swap(u, v);
        edge_owner.emplace_back((static_cast<uint64_t>(u) << 32) | v, static_cast<int32_t>(c));
      }
    }
    std::sort(edge_owner.begin(), edge_owner.end());
    for (std::size_t e = 1; e < edge_owner.size(); ++e) {
      if (edge_owner[e].first != edge_owner[e - 1].first) continue;
      const int32_t x = root(edge_owner[e].second), y = root(edge_owner[e - 1].second);
      if (x != y) parent[std::max(x, y)] = std::min(x, y);
    }
    // Components in order of their first face.
    std::vector<int32_t> component(n, -1);
    for (std::size_t c = 0; c < n; ++c) {
      const int32_t r = root(static_cast<int32_t>(c));
      if (component[r] < 0) {
        component[r] = static_cast<int32_t>(cell_keys_.size());
        cell_keys_.push_back({sides[a].seed, sides[a].region});
        if (has_piece_[sides[a].seed]) ++boundary_cells_;
      }
      const Side& side = sides[a + c];
      (side.reversed ? face_right : face_left)[side.face] = component[r];
    }
    a = b;
  }
  // Vertices in use, in first-use order of the faces.
  std::vector<int32_t> remap(kinds_.size(), -1);
  int32_t vertex_count = 0;
  for (const int32_t v : loop_vertices_)
    if (remap[v] < 0) remap[v] = vertex_count++;
  out.vertices.assign(3 * static_cast<std::size_t>(vertex_count), 0.0);
  out.vertex_kinds.assign(vertex_count, 0);
  out.vertex_sources.assign(2 * static_cast<std::size_t>(vertex_count), -1);
  std::vector<int32_t> order(static_cast<std::size_t>(vertex_count));
  for (std::size_t v = 0; v < remap.size(); ++v)
    if (remap[v] >= 0) order[remap[v]] = static_cast<int32_t>(v);
  out.exact_offsets.assign(1, 0);
  out.exact_values.clear();
  std::array<Expansion, 3> numerator;
  Expansion denominator;
  for (int32_t w = 0; w < vertex_count; ++w) {
    const int32_t v = order[w];
    exact_vertex(v, numerator, denominator);
    for (int k = 0; k < 3; ++k) {
      out.vertices[3 * static_cast<int64_t>(w) + k] = rounded_quotient(numerator[k], denominator);
    }
    for (const Expansion* e : {&numerator[0], &numerator[1], &numerator[2], &denominator}) {
      out.exact_values.insert(out.exact_values.end(), e->data(), e->data() + e->size());
      out.exact_offsets.push_back(static_cast<int64_t>(out.exact_values.size()));
    }
    out.vertex_kinds[w] = kinds_[v];
    out.vertex_sources[2 * w] = sources_[2 * v];
    out.vertex_sources[2 * w + 1] = sources_[2 * v + 1];
  }
  out.face_offsets.assign(1, 0);
  out.face_vertices.clear();
  out.face_cells.clear();
  out.face_facets.clear();
  for (std::size_t f = 0; f < faces_.size(); ++f) {
    const FaceRecord& r = faces_[f];
    for (int32_t q = 0; q < r.size; ++q) out.face_vertices.push_back(remap[loop_vertices_[r.begin + q]]);
    out.face_offsets.push_back(static_cast<int64_t>(out.face_vertices.size()));
    out.face_cells.push_back(face_left[f]);
    out.face_cells.push_back(face_right[f]);
    out.face_facets.push_back(r.facet);
  }
  const std::size_t cells = cell_keys_.size();
  out.cell_seeds.resize(cells);
  out.cell_regions.resize(cells);
  out.cell_volumes.assign(cells, 0.0);
  out.cell_centroids.assign(3 * cells, 0.0);
  out.cell_convex.assign(cells, 1);
  for (std::size_t c = 0; c < cells; ++c) {
    out.cell_seeds[c] = cell_keys_[c][0];
    out.cell_regions[c] = cell_keys_[c][1];
  }
  // Volumes and centroids by fans from each cell's seed.
  for (std::size_t f = 0; f < faces_.size(); ++f) {
    const int64_t begin = out.face_offsets[f], end = out.face_offsets[f + 1];
    for (int side = 0; side < 2; ++side) {
      const int32_t cell = side == 0 ? face_left[f] : face_right[f];
      if (cell < 0) continue;
      const double* o = seed(cell_keys_[cell][0]);
      const double sign = side == 0 ? 1.0 : -1.0;
      const double* x0 = out.vertices.data() + 3 * static_cast<int64_t>(out.face_vertices[begin]);
      for (int64_t q = begin + 1; q + 1 < end; ++q) {
        const double* x1 = out.vertices.data() + 3 * static_cast<int64_t>(out.face_vertices[q]);
        const double* x2 = out.vertices.data() + 3 * static_cast<int64_t>(out.face_vertices[q + 1]);
        double a[3], b[3], c[3], bc[3];
        geometry::difference(x0, o, a);
        geometry::difference(x1, o, b);
        geometry::difference(x2, o, c);
        geometry::cross(b, c, bc);
        const double volume = sign * geometry::dot(a, bc) / 6.0;
        out.cell_volumes[cell] += volume;
        for (int k = 0; k < 3; ++k)
          out.cell_centroids[3 * cell + k] += volume * (a[k] + b[k] + c[k]) / 4.0;
      }
    }
  }
  for (std::size_t c = 0; c < cells; ++c) {
    const double* o = seed(cell_keys_[c][0]);
    for (int k = 0; k < 3; ++k) {
      out.cell_centroids[3 * c + k] =
          o[k] + (out.cell_volumes[c] != 0.0 ? out.cell_centroids[3 * c + k] / out.cell_volumes[c] : 0.0);
    }
  }
  // Exact convexity: Voronoi cells are convex, and a clipped cell is convex
  // exactly when no surface edge reflex towards its region (exact orient3d)
  // carries a side of one of its pieces.
  for (std::size_t p = 0; p < pieces_.size(); ++p) {
    const Piece& piece = pieces_[p];
    const int32_t face = piece_face_[p];
    for (int32_t q = 0; q < piece.size; ++q) {
      const int32_t support = piece_supports_[piece.begin + q];
      if (support >= 0) continue;
      const int32_t edge = s_.triangle_edges[3 * static_cast<int64_t>(piece.triangle) + (-1 - support)];
      const uint8_t reflex = s_.edge_reflex[edge];
      if ((reflex & 1) && s_.back(piece.triangle) >= 0) {
        const int32_t cell = s_.back(piece.triangle) == faces_[face].left_region ? face_left[face]
                                                                                  : face_right[face];
        if (cell >= 0) out.cell_convex[cell] = 0;
      }
      if ((reflex & 2) && s_.front(piece.triangle) >= 0) {
        const int32_t cell = s_.front(piece.triangle) == faces_[face].left_region ? face_left[face]
                                                                                   : face_right[face];
        if (cell >= 0) out.cell_convex[cell] = 0;
      }
    }
  }
  out.seeds.assign(seeds_.begin(), seeds_.begin() + 3 * static_cast<int64_t>(interior_));
  // Planarity residual of bisector faces: distance of each vertex to the
  // exact bisector plane of its two seeds, relative to h.
  double residual = 0.0;
  for (std::size_t f = 0; f < faces_.size(); ++f) {
    const FaceRecord& r = faces_[f];
    if (r.facet >= 0) continue;
    const double* a = seed(r.left_seed);
    const double* b = seed(r.right_seed);
    double axis[3], middle[3];
    geometry::difference(b, a, axis);
    for (int k = 0; k < 3; ++k) middle[k] = 0.5 * (a[k] + b[k]);
    const double length = geometry::length(axis);
    for (int64_t q = out.face_offsets[f]; q < out.face_offsets[f + 1]; ++q) {
      double d[3];
      geometry::difference(out.vertices.data() + 3 * static_cast<int64_t>(out.face_vertices[q]), middle, d);
      residual = std::max(residual, std::fabs(geometry::dot(d, axis)) / length);
    }
  }
  planarity_residual_ = residual / h_;
}

void Builder::certify(phx_mc_voronoi_cells& out) {
  // Closure: every directed edge of a cell meets its reverse exactly once.
  std::vector<std::array<int32_t, 3>> directed;
  for (std::size_t f = 0; f + 1 < out.face_offsets.size(); ++f) {
    const int64_t begin = out.face_offsets[f], end = out.face_offsets[f + 1];
    for (int side = 0; side < 2; ++side) {
      const int32_t cell = out.face_cells[2 * f + side];
      if (cell < 0) continue;
      for (int64_t q = begin; q < end; ++q) {
        int32_t u = out.face_vertices[q];
        int32_t v = out.face_vertices[q + 1 < end ? q + 1 : begin];
        if (side == 1) std::swap(u, v);
        directed.push_back({cell, u, v});
      }
    }
  }
  std::sort(directed.begin(), directed.end());
  open_edges_ = 0;
  for (std::size_t k = 0; k < directed.size(); ++k) {
    const auto& e = directed[k];
    const bool duplicate = k > 0 && directed[k - 1] == e;
    const std::array<int32_t, 3> reverse{e[0], e[2], e[1]};
    const auto range = std::equal_range(directed.begin(), directed.end(), reverse);
    if (duplicate || range.second - range.first != 1) {
      ++open_edges_;
      suspect(out.cell_seeds[e[0]]);
    }
  }
  // Coverage: the pieces of every surface triangle sum to its area.
  const int64_t m = s_.in->triangle_count;
  std::vector<double> covered(m, 0.0);
  for (const Piece& p : pieces_) {
    double n[3] = {0, 0, 0};
    for (int32_t q = 0; q < p.size; ++q) {
      const double* a = coords_.data() + 3 * static_cast<int64_t>(piece_vertices_[p.begin + q]);
      const double* b =
          coords_.data() + 3 * static_cast<int64_t>(piece_vertices_[p.begin + (q + 1) % p.size]);
      n[0] += (a[1] - b[1]) * (a[2] + b[2]);
      n[1] += (a[2] - b[2]) * (a[0] + b[0]);
      n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    covered[p.triangle] += 0.5 * geometry::dot(n, &s_.normals[3 * static_cast<int64_t>(p.triangle)]);
  }
  coverage_residual_ = 0.0;
  for (int64_t t = 0; t < m; ++t)
    coverage_residual_ = std::max(coverage_residual_, std::fabs(covered[t] - s_.areas[t]) / s_.areas[t]);
  // Region volumes by the divergence theorem against the cell volumes.
  std::vector<double> expected(s_.region_count, 0.0), achieved(s_.region_count, 0.0);
  const double* origin = s_.lower;
  for (int64_t t = 0; t < m; ++t) {
    double a[3], b[3], c[3], bc[3];
    geometry::difference(s_.corner(t, 0), origin, a);
    geometry::difference(s_.corner(t, 1), origin, b);
    geometry::difference(s_.corner(t, 2), origin, c);
    geometry::cross(b, c, bc);
    const double volume = geometry::dot(a, bc) / 6.0;
    if (s_.back(t) >= 0) expected[s_.back(t)] += volume;
    if (s_.front(t) >= 0) expected[s_.front(t)] -= volume;
  }
  for (std::size_t c = 0; c < out.cell_volumes.size(); ++c) achieved[out.cell_regions[c]] += out.cell_volumes[c];
  double total = 0.0;
  volume_residual_ = 0.0;
  for (int32_t r = 0; r < s_.region_count; ++r) total += std::fabs(expected[r]);
  for (int32_t r = 0; r < s_.region_count; ++r)
    volume_residual_ = std::max(volume_residual_, std::fabs(achieved[r] - expected[r]) / total);
  certified_ = open_edges_ == 0 && coverage_residual_ <= 1e-8 && volume_residual_ <= 1e-9 &&
               degenerate_ == 0;
}

// Deterministic displacement of automatic seeds; suspects get a new salt and
// a twice larger amplitude per repair round.
void displace(const std::vector<double>& base, const std::vector<int32_t>& rounds, double h,
              std::vector<double>& out) {
  out.resize(base.size());
  for (std::size_t i = 0; i < rounds.size(); ++i) {
    const double amplitude = std::ldexp(h, -10 + rounds[i]);
    for (int k = 0; k < 3; ++k) {
      const uint64_t salt = (static_cast<uint64_t>(i) << 8) ^ (static_cast<uint64_t>(rounds[i]) << 2) ^ k;
      out[3 * i + k] = base[3 * i + k] + amplitude * unit_hash(salt);
    }
  }
}

int32_t build(const Input& in, phx_mc_voronoi_cells& out) {
  const auto start = Clock::now();
  Surface surface;
  build_surface(in, surface);
  const bool automatic = in.seed_count == 0;
  SeedSet seeds;
  auto clock = Clock::now();
  if (automatic) {
    automatic_seeds(surface, in.size, seeds);
  } else {
    seeds.points.assign(in.seeds, in.seeds + 3 * in.seed_count);
    seeds.regions.assign(in.seed_count, -2);
  }
  const double seeding = seconds_since(clock);
  const int64_t count = static_cast<int64_t>(seeds.regions.size());
  if (count == 0) return PHX_MC_UNRESOLVED;
  const double h = automatic ? in.size : surface.diameter / std::cbrt(static_cast<double>(count));
  std::vector<int32_t> rounds(count, 0);
  std::vector<double> displaced;
  double timings[5] = {0, 0, 0, 0, 0};
  double certification = 0.0;
  int32_t repairs = 0;
  int64_t events = 0;
  while (true) {
    if (automatic) {
      displace(seeds.points, rounds, h, displaced);
    } else {
      displaced = seeds.points;
    }
    Builder builder(surface, displaced, seeds.regions, automatic, h);
    out = phx_mc_voronoi_cells{};
    builder.run(out, timings);
    events += builder.degenerate_events();
    clock = Clock::now();
    if (in.prove && builder.degenerate_events() == 0) builder.certify(out);
    certification += seconds_since(clock);
    const bool settled = builder.degenerate_events() == 0 && (!in.prove || builder.certified());
    // Local repair: new displacements for the seeds the failure involved,
    // while every one stays within the displacement bound h 2^-6.
    std::vector<int32_t> involved = builder.suspects();
    if (involved.empty()) {
      involved.resize(count);
      std::iota(involved.begin(), involved.end(), 0);
    }
    std::sort(involved.begin(), involved.end());
    involved.erase(std::unique(involved.begin(), involved.end()), involved.end());
    const bool repairable =
        automatic && std::all_of(involved.begin(), involved.end(),
                                 [&](int32_t i) { return rounds[i] < 4; });
    if (settled || !repairable) {
      double* stats = out.stats.data();
      stats[0] = seeding;
      for (int k = 0; k < 5; ++k) stats[1 + k] = timings[k];
      stats[6] = certification;
      stats[7] = seconds_since(start);
      stats[8] = static_cast<double>(count);
      stats[9] = static_cast<double>(seeds.surface_seeds);
      stats[10] = static_cast<double>(seeds.lattice_seeds);
      stats[11] = static_cast<double>(out.cell_seeds.size());
      stats[12] = static_cast<double>(builder.boundary_cells());
      stats[13] = static_cast<double>(repairs);
      stats[14] = static_cast<double>(events);
      stats[15] = in.prove ? (builder.certified() ? 1.0 : 0.0) : -1.0;
      stats[16] = in.prove ? static_cast<double>(builder.open_edges()) : -1.0;
      stats[17] = in.prove ? builder.coverage_residual() : -1.0;
      stats[18] = in.prove ? builder.volume_residual() : -1.0;
      int64_t nonconvex = 0;
      for (const uint8_t c : out.cell_convex) nonconvex += c == 0;
      stats[19] = static_cast<double>(nonconvex);
      stats[20] = builder.planarity_residual();
      stats[21] = static_cast<double>(surface.feature_edges);
      stats[22] = static_cast<double>(out.face_cells.size() / 2);
      stats[23] = h;
      if (settled) return PHX_MC_OK;
      return !automatic && builder.degenerate_events() > 0 ? PHX_MC_DEGENERATE_INPUT
                                                            : PHX_MC_UNRESOLVED;
    }
    for (const int32_t i : involved) ++rounds[i];
    ++repairs;
  }
}

}  // namespace
}  // namespace phx::mc::conforming_voronoi

namespace {

int32_t validate(const phx_mc_conforming_voronoi_request* in) {
  if (in == nullptr || in->result == nullptr || in->vertex_count < 4 || in->triangle_count < 4 ||
      in->vertices == nullptr || in->triangles == nullptr || in->triangle_regions == nullptr ||
      in->triangle_facets == nullptr || in->seed_count < 0 ||
      (in->seed_count > 0 && in->seeds == nullptr) || (in->prove != 0 && in->prove != 1) ||
      !(in->feature_angle_degrees > 0.0 && in->feature_angle_degrees < 180.0) ||
      (in->seed_count == 0 && !(in->size > 0.0 && std::isfinite(in->size)))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (in->vertex_count >= INT32_MAX || in->triangle_count >= (int64_t{1} << 31) / 3 ||
      in->seed_count >= INT32_MAX - 8 || !phx::mc::addressable(in->vertex_count, 3, sizeof(double)) ||
      !phx::mc::addressable(in->triangle_count, 3, sizeof(int32_t)) ||
      !phx::mc::addressable(in->seed_count, 3, sizeof(double))) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  for (int64_t k = 0; k < 3 * in->vertex_count; ++k) {
    if (!std::isfinite(in->vertices[k])) return PHX_MC_NONFINITE_INPUT;
    if (!phx::mc::coordinate_in_domain(in->vertices[k])) return PHX_MC_RANGE_ERROR;
  }
  for (int64_t k = 0; k < 3 * in->seed_count; ++k) {
    if (!std::isfinite(in->seeds[k])) return PHX_MC_NONFINITE_INPUT;
    if (!phx::mc::coordinate_in_domain(in->seeds[k])) return PHX_MC_RANGE_ERROR;
  }
  for (int64_t t = 0; t < in->triangle_count; ++t) {
    for (int k = 0; k < 3; ++k) {
      const int32_t v = in->triangles[3 * t + k];
      if (v < 0 || v >= in->vertex_count) return PHX_MC_INVALID_INPUT;
    }
    const int32_t front = in->triangle_regions[2 * t], back = in->triangle_regions[2 * t + 1];
    if (front < -1 || back < -1 || front == back || in->triangle_facets[t] < 0) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  return PHX_MC_OK;
}

}  // namespace

extern "C" {

int32_t phx_mc_conforming_voronoi(const phx_mc_conforming_voronoi_request* request) {
  if (request != nullptr && request->result != nullptr) *request->result = nullptr;
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    const int32_t status = validate(request);
    if (status != PHX_MC_OK) return status;
    const phx::mc::conforming_voronoi::Input input{
        request->vertex_count,    request->vertices,        request->triangle_count,
        request->triangles,       request->triangle_regions, request->triangle_facets,
        request->seed_count,      request->seeds,           request->size,
        request->feature_angle_degrees * (3.14159265358979323846 / 180.0),
        request->prove != 0};
    auto result = std::make_unique<phx_mc_voronoi_cells>();
    int32_t built;
    try {
      built = phx::mc::conforming_voronoi::build(input, *result);
    } catch (const phx::mc::conforming_voronoi::Failure& failure) {
      return failure.status;
    }
    if (built != PHX_MC_OK && built != PHX_MC_UNRESOLVED) return built;
    if (built == PHX_MC_UNRESOLVED && result->cell_seeds.empty()) return built;
    *request->result = result.release();
    return built;
  });
}

int32_t phx_mc_conforming_voronoi_sizes(const phx_mc_voronoi_cells* result, int64_t* sizes) {
  if (result == nullptr || sizes == nullptr) return PHX_MC_INVALID_ARGUMENT;
  sizes[0] = static_cast<int64_t>(result->vertex_kinds.size());
  sizes[1] = static_cast<int64_t>(result->face_facets.size());
  sizes[2] = static_cast<int64_t>(result->face_vertices.size());
  sizes[3] = static_cast<int64_t>(result->cell_seeds.size());
  sizes[4] = static_cast<int64_t>(result->seeds.size() / 3);
  sizes[5] = static_cast<int64_t>(result->exact_values.size());
  return PHX_MC_OK;
}

int32_t phx_mc_conforming_voronoi_export(const phx_mc_voronoi_cells* result,
                                         const phx_mc_conforming_voronoi_arrays* arrays) {
  if (result == nullptr || arrays == nullptr || arrays->vertices == nullptr ||
      arrays->vertex_kinds == nullptr || arrays->vertex_sources == nullptr ||
      arrays->face_offsets == nullptr || arrays->face_vertices == nullptr ||
      arrays->face_cells == nullptr || arrays->face_facets == nullptr ||
      arrays->cell_seeds == nullptr || arrays->cell_regions == nullptr ||
      arrays->cell_volumes == nullptr || arrays->cell_centroids == nullptr ||
      arrays->cell_convex == nullptr || arrays->seeds == nullptr || arrays->stats == nullptr ||
      arrays->exact_offsets == nullptr ||
      (!result->exact_values.empty() && arrays->exact_values == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  std::copy(result->vertices.begin(), result->vertices.end(), arrays->vertices);
  std::copy(result->vertex_kinds.begin(), result->vertex_kinds.end(), arrays->vertex_kinds);
  std::copy(result->vertex_sources.begin(), result->vertex_sources.end(), arrays->vertex_sources);
  std::copy(result->face_offsets.begin(), result->face_offsets.end(), arrays->face_offsets);
  std::copy(result->face_vertices.begin(), result->face_vertices.end(), arrays->face_vertices);
  std::copy(result->face_cells.begin(), result->face_cells.end(), arrays->face_cells);
  std::copy(result->face_facets.begin(), result->face_facets.end(), arrays->face_facets);
  std::copy(result->cell_seeds.begin(), result->cell_seeds.end(), arrays->cell_seeds);
  std::copy(result->cell_regions.begin(), result->cell_regions.end(), arrays->cell_regions);
  std::copy(result->cell_volumes.begin(), result->cell_volumes.end(), arrays->cell_volumes);
  std::copy(result->cell_centroids.begin(), result->cell_centroids.end(), arrays->cell_centroids);
  std::copy(result->cell_convex.begin(), result->cell_convex.end(), arrays->cell_convex);
  std::copy(result->seeds.begin(), result->seeds.end(), arrays->seeds);
  std::copy(result->stats.begin(), result->stats.end(), arrays->stats);
  std::copy(result->exact_offsets.begin(), result->exact_offsets.end(), arrays->exact_offsets);
  std::copy(result->exact_values.begin(), result->exact_values.end(), arrays->exact_values);
  return PHX_MC_OK;
}

void phx_mc_conforming_voronoi_free(phx_mc_voronoi_cells* result) { delete result; }

}  // extern "C"
