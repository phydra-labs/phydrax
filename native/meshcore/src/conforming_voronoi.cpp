//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Conforming Voronoi cells of a triangulated solid (phx_mc_conforming_voronoi).
//
// The output is the Voronoi diagram of interior seeds and mirrored exterior
// ghost seeds, restricted to the interior seeds: every cell is the exact,
// closed, convex Voronoi cell of its seed, and the boundary is formed by the
// bisector faces between interior seeds and ghosts (VoroCrust's construction).
//
// Seeds, for the cell size h, surface spacing r = 0.75 h and depth d = 0.3 r:
// - Surface samples: corners (feature degree other than 0 and 2), feature
//   chains at spacing about r, then a maximal Poisson-disk sample of radius r
//   (barycentric candidates in Morton order).
// - Smooth sample p on triangle t: a seed p + c_t n - d n on every solid side
//   of t, where the curvature shift c_t (0.15 times the mean offset of the
//   same-patch neighbours from the tangent plane, exactly 0 on a plane) moves
//   the mirror plane inside convex surfaces so that the bisector surface
//   straddles the input.
// - Feature edge sample: around the edge the incident triangles bound angular
//   sectors; every solid sector gets seeds at radius d / sin(a) at angular
//   offset a from its bounding planes (a = min(smallest sector / 2, 45 deg)),
//   and more seeds inside so that they are at most 90 deg apart.
// - Corner: the sector seeds of every incident feature edge, placed at
//   distance d from the corner along that edge.
// - Body-centered cubic lattice of cell volume h^3 beyond 1.2 r of the surface.
// - Ghosts: every interior seed s within 1.2 r of the surface is reflected
//   across the shifted tangent plane of each of its feet: a foot is the
//   nearest point of the surface among triangles whose normals lie within the
//   feature angle of each other, and a local minimum of the distance. A ghost
//   is kept when the foot's triangle separates s's region from the exterior
//   and the ghost lies outside every region (exact vertical-ray parity).
// - Every seed is rounded to the grid 2^(e - 40), e the exponent of the
//   largest coordinate: reflections across axis-aligned planes are then
//   exact, so the bisectors lie exactly on those planes.
// Seeding is a function of the input alone (no random state): deterministic.
//
// Cells. The exact Delaunay triangulation (delaunay3d, symbolic perturbation)
// of all seeds plus eight far box seeds gives the diagram. Adjacent tets with
// exactly cospherical vertices (exact insphere = 0) share one Voronoi vertex;
// the face of cell i towards seed j is the ring of Voronoi vertices around
// Delaunay edge (i, j) with repeats removed, dropped when fewer than three
// remain. Vertices are exact circumcenters (expansions); the published
// coordinates are their nearest binary64 values. The boundary is exact when
// every vertex of a boundary or interface face lies exactly (exact
// orientation) on the plane of its nearest triangle.
//
// Resolution: nearly cospherical seeds (mirror pairs about planes that are
// not axis-aligned, after the dyadic snap) give adjacent circumcenters within
// 2^-14 h, hence edges and faces too small for a validity certificate (star
// determinants scale as (edge / R)^2 / 6 and must clear 2^-35 with margin, so
// edges under ~3e-5 R fail; detection at 2^-14 h covers them and a 2^-10 h move
// clears it). Each such apex pair is resolved by rigid moves first, one per
// rebuild while the pair persists, over the seeds of its two tets (apexes
// first, unmirrored seeds before mirrored ones): an unmirrored seed moves
// alone, a seed whose mirror pairs form a tree moves 2^-10 h outward along the
// ray from a tet's circumcenter and the tree is reflected again outward from
// it, so every facet between mirrors stays put. Mirror pairs across one plane
// are always cocircular (isosceles trapezoids), so every facet vertex is
// degenerate by construction:
// exact reflections (axis-aligned planes) make it exactly cospherical and it
// merges, but snapped reflections across oblique planes split it into a
// snap-level cluster that rigid moves keep, and curved surfaces do the same
// across neighbouring planes. A pair that persists after its rigid moves is
// resolved by a lone move of one apex seed (a ghost when there is one), which
// then leaves its mirror pairs and tilts their facets by up to 2^-10 h. Every
// pair passes each stage at most once, so the rebuilds terminate.
//
// Repair: an interior cell meeting a box seed, or a boundary vertex farther
// than 1.2 r from the surface, gets a ghost: its seed reflected across the
// nearest triangle to the offending vertex. Rebuilt until no cell leaks; every
// (seed, triangle) pair is used at most once, so repair terminates.
//
// Certification (prove): every cell closed (each directed edge met once by its
// reverse), the triangulation locally Delaunay at every face of an interior
// seed's star (exact insphere <= 0: the cells are exactly Voronoi cells, hence
// convex), and an upper bound on the distance of every boundary face to the
// surface: the distance to one triangle is convex, so its maximum over a
// piece of a face is attained at the piece's corners; pieces are split until
// the bound meets the vertex maximum. The volume error against the input
// solid, with a bound of its rounding, is reported with or without prove.
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
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

struct phx_mc_voronoi_cells {
  std::vector<double> vertices;
  std::vector<int32_t> vertex_triangles;  // nearest input triangle of a boundary vertex, else -1
  std::vector<int64_t> face_offsets{0};
  std::vector<int32_t> face_vertices;
  std::vector<int32_t> face_cells;  // two per face
  std::vector<int32_t> face_facets;
  std::vector<int32_t> cell_seeds;
  std::vector<int32_t> cell_regions;
  std::vector<double> cell_volumes;
  std::vector<double> cell_centroids;
  std::vector<double> seeds;
  std::vector<int32_t> seed_regions;
  // Exact coordinates of every vertex: x_k = N_k / D, the expansions
  // N_0, N_1, N_2, D of vertex v at exact_offsets[4 v + 0 .. 4 v + 4].
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

constexpr double kSpacing = 0.75;  // surface sample spacing r / h
constexpr double kDepth = 0.3;     // seed depth d / r
constexpr double kReach = 1.2;     // lattice clearance and reflection reach / r
constexpr double kShift = 0.15;    // curvature shift of the mirror plane
constexpr double kResolution = 0x1p-14;  // nearly cospherical circumcenters / h
constexpr double kNudge = 0x1p-10;       // seed move off a circumsphere / h
constexpr double kPi = 3.14159265358979323846;

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

struct Input {
  int64_t vertex_count;
  const double* vertices;
  int64_t triangle_count;
  const int32_t* triangles;
  const int32_t* regions;  // (front, back)
  const int32_t* facets;
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
  std::vector<int32_t> edges;           // two vertices per undirected edge
  std::vector<int64_t> edge_offsets;    // incident triangles per edge (CSR)
  std::vector<int32_t> edge_items;
  std::vector<uint8_t> edge_feature;
  // Classes of edge-adjacent, exactly coplanar triangles of one facet and
  // region pair; patches: classes of triangles joined by non-feature edges.
  std::vector<int32_t> plane_class;
  std::vector<int32_t> patch;
  std::vector<double> normals;  // unit, 3 per triangle
  std::vector<double> areas;
  std::vector<int64_t> star_offsets;  // triangles per vertex (CSR)
  std::vector<int32_t> star;
  double lower[3]{}, upper[3]{};
  double diameter = 0.0;
  int32_t feature_edges = 0;

  const double* point(int32_t v) const { return in->vertices + 3 * static_cast<int64_t>(v); }
  const double* corner(int32_t t, int k) const {
    return point(in->triangles[3 * static_cast<int64_t>(t) + k]);
  }
  const double* normal(int32_t t) const { return normals.data() + 3 * static_cast<int64_t>(t); }
  int32_t front(int32_t t) const { return in->regions[2 * static_cast<int64_t>(t)]; }
  int32_t back(int32_t t) const { return in->regions[2 * static_cast<int64_t>(t) + 1]; }
};

int32_t find_root(std::vector<int32_t>& parent, int32_t a) {
  while (parent[a] != a) a = parent[a] = parent[parent[a]];
  return a;
}
void unite(std::vector<int32_t>& parent, int32_t a, int32_t b) {
  a = find_root(parent, a);
  b = find_root(parent, b);
  if (a != b) parent[std::max(a, b)] = std::min(a, b);
}

void build_surface(const Input& in, Surface& s) {
  s.in = &in;
  const int64_t m = in.triangle_count;
  for (int k = 0; k < 3; ++k) s.lower[k] = s.upper[k] = in.vertices[k];
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
    s.region_count = std::max({s.region_count, in.regions[2 * t] + 1, in.regions[2 * t + 1] + 1});
    double ab[3], ac[3], n[3];
    geometry::difference(s.corner(t, 1), s.corner(t, 0), ab);
    geometry::difference(s.corner(t, 2), s.corner(t, 0), ac);
    geometry::cross(ab, ac, n);
    const double length = geometry::length(n);
    if (!(length > 0.0)) throw Failure(PHX_MC_DEGENERATE_INPUT);
    for (int k = 0; k < 3; ++k) s.normals[3 * t + k] = n[k] / length;
    s.areas[t] = 0.5 * length;
  }
  s.star_offsets.assign(in.vertex_count + 1, 0);
  for (int64_t k = 0; k < 3 * m; ++k) ++s.star_offsets[in.triangles[k] + 1];
  for (int64_t v = 0; v < in.vertex_count; ++v) s.star_offsets[v + 1] += s.star_offsets[v];
  s.star.resize(3 * m);
  {
    std::vector<int64_t> cursor(s.star_offsets.begin(), s.star_offsets.end() - 1);
    for (int64_t t = 0; t < m; ++t)
      for (int k = 0; k < 3; ++k) s.star[cursor[in.triangles[3 * t + k]]++] = static_cast<int32_t>(t);
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
  s.plane_class.resize(m);
  std::iota(s.plane_class.begin(), s.plane_class.end(), 0);
  s.patch.resize(m);
  std::iota(s.patch.begin(), s.patch.end(), 0);
  s.edge_offsets.assign(1, 0);
  const double cosine = std::cos(in.feature_angle);
  for (std::size_t r = 0; r < rows.size();) {
    std::size_t e = r;
    while (e < rows.size() && rows[e][0] == rows[r][0] && rows[e][1] == rows[r][1]) ++e;
    s.edges.push_back(rows[r][0]);
    s.edges.push_back(rows[r][1]);
    for (std::size_t x = r; x < e; ++x) s.edge_items.push_back(rows[x][2]);
    s.edge_offsets.push_back(static_cast<int64_t>(s.edge_items.size()));
    uint8_t feature = e - r == 2 ? 0 : 1;
    if (e - r == 2) {
      const int32_t t = rows[r][2], u = rows[r + 1][2];
      const double c = geometry::dot(s.normal(t), s.normal(u));
      if (c < cosine || s.front(t) != s.front(u) || s.back(t) != s.back(u)) feature = 1;
      const int32_t opposite = in.triangles[3 * static_cast<int64_t>(u) + rows[r + 1][3]];
      if (in.facets[t] == in.facets[u] && s.front(t) == s.front(u) && s.back(t) == s.back(u) &&
          c > 0.0 &&
          orient3d(s.corner(t, 0), s.corner(t, 1), s.corner(t, 2), s.point(opposite)) == 0) {
        unite(s.plane_class, t, u);
      }
      if (!feature) unite(s.patch, t, u);
    }
    s.edge_feature.push_back(feature);
    s.feature_edges += feature;
    r = e;
  }
  for (int64_t t = 0; t < m; ++t) {
    s.plane_class[t] = find_root(s.plane_class, static_cast<int32_t>(t));
    s.patch[t] = find_root(s.patch, static_cast<int32_t>(t));
  }
}

// Closest point q of triangle (a, b, c) to p (Ericson).
void closest_point(const double* p, const double* a, const double* b, const double* c, double* q) {
  const double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const double ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const double ap[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
  const auto set = [&](const double* x) { std::copy_n(x, 3, q); };
  const auto along = [&](const double* x, const double* d, double t) {
    for (int k = 0; k < 3; ++k) q[k] = x[k] + t * d[k];
  };
  const double d1 = geometry::dot(ab, ap), d2 = geometry::dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) return set(a);
  const double bp[3] = {p[0] - b[0], p[1] - b[1], p[2] - b[2]};
  const double d3 = geometry::dot(ab, bp), d4 = geometry::dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) return set(b);
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return along(a, ab, d1 / (d1 - d3));
  const double cp[3] = {p[0] - c[0], p[1] - c[1], p[2] - c[2]};
  const double d5 = geometry::dot(ab, cp), d6 = geometry::dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) return set(c);
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return along(a, ac, d2 / (d2 - d6));
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    const double bc[3] = {c[0] - b[0], c[1] - b[1], c[2] - b[2]};
    return along(b, bc, (d4 - d3) / ((d4 - d3) + (d5 - d6)));
  }
  const double denominator = va + vb + vc;
  const double v = vb / denominator, w = vc / denominator;
  for (int k = 0; k < 3; ++k) q[k] = a[k] + ab[k] * v + ac[k] * w;
}

// Uniform grid of triangles by bounding box.
struct TriangleGrid {
  const Surface* s = nullptr;
  double origin[3]{};
  double cell = 1.0;
  int64_t dims[3]{1, 1, 1};
  std::vector<int64_t> offsets;
  std::vector<int32_t> items;
  mutable std::vector<uint32_t> stamp;
  mutable uint32_t clock = 0;

  int64_t clamp_index(double x, int axis) const {
    const double q = std::floor((x - origin[axis]) / cell);
    if (!(q > 0.0)) return 0;
    return std::min<int64_t>(dims[axis] - 1, static_cast<int64_t>(q));
  }
  int64_t flat(int64_t i, int64_t j, int64_t k) const { return (k * dims[1] + j) * dims[0] + i; }

  void build(const Surface& surface, double requested) {
    s = &surface;
    double extent = 0.0;
    for (int k = 0; k < 3; ++k) extent = std::max(extent, s->upper[k] - s->lower[k]);
    cell = std::max(requested, extent / 256.0);
    for (int k = 0; k < 3; ++k) {
      origin[k] = s->lower[k];
      dims[k] = std::max<int64_t>(1, static_cast<int64_t>((s->upper[k] - s->lower[k]) / cell) + 1);
    }
    const int64_t total = dims[0] * dims[1] * dims[2];
    std::vector<int64_t> counts(total + 1, 0);
    const int64_t m = s->in->triangle_count;
    auto range = [&](int64_t t, int64_t lo[3], int64_t hi[3]) {
      for (int a = 0; a < 3; ++a) {
        double l = s->corner(t, 0)[a], u = l;
        for (int k = 1; k < 3; ++k) {
          l = std::min(l, s->corner(t, k)[a]);
          u = std::max(u, s->corner(t, k)[a]);
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
    stamp.assign(m, 0);
  }

  // Calls f(t) once for every triangle whose box meets the box of radius
  // `radius` about p.
  template <class F>
  void visit(const double* p, double radius, F&& f) const {
    if (++clock == 0) {
      std::fill(stamp.begin(), stamp.end(), 0);
      clock = 1;
    }
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
            if (stamp[t] == clock) continue;
            stamp[t] = clock;
            f(t);
          }
        }
  }

  bool any_within(const double* p, double radius) const {
    bool found = false;
    visit(p, radius, [&](int32_t t) {
      if (!found &&
          geometry::point_triangle_distance(p, s->corner(t, 0), s->corner(t, 1), s->corner(t, 2)) <
              radius) {
        found = true;
      }
    });
    return found;
  }

  // Nearest triangle to p (and its closest point), searching outwards from
  // `radius`; -1 only for an empty surface.
  int32_t nearest(const double* p, double radius, double* q, double* distance) const {
    double limit = radius;
    while (true) {
      int32_t best = -1;
      double best_distance = HUGE_VAL;
      visit(p, limit, [&](int32_t t) {
        double x[3];
        closest_point(p, s->corner(t, 0), s->corner(t, 1), s->corner(t, 2), x);
        const double d = std::sqrt(geometry::squared_distance(p, x));
        if (d < best_distance || (d == best_distance && t < best)) {
          best_distance = d;
          best = t;
          std::copy_n(x, 3, q);
        }
      });
      if (best >= 0 && best_distance <= limit) {
        *distance = best_distance;
        return best;
      }
      if (limit > 4.0 * s->diameter + 1.0 && best >= 0) {
        *distance = best_distance;
        return best;
      }
      if (limit > 4.0 * s->diameter + 1.0) return -1;
      limit *= 2.0;
    }
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

// Region of p by the crossings of the upward vertical ray through it: the
// region below the lowest crossing above p (exterior when none). The ray
// position is perturbed symbolically (exact orient2d); heights are binary64.
int32_t region_at(const Surface& s, const ColumnGrid& grid, const double* q) {
  if (q[0] < s.lower[0] || q[0] > s.upper[0] || q[1] < s.lower[1] || q[1] > s.upper[1] ||
      q[2] > s.upper[2]) {
    return -1;
  }
  const int64_t c = grid.index(q[1], 1) * grid.dims[0] + grid.index(q[0], 0);
  double best = HUGE_VAL;
  int32_t region = -1;
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
    const double* n = s.normal(t);
    const double z = a[2] - (n[0] * (q[0] - a[0]) + n[1] * (q[1] - a[1])) / n[2];
    if (z <= q[2] || z >= best) continue;
    best = z;
    // orientation > 0: the normal points up, so the back lies below.
    region = orientation > 0 ? s.back(t) : s.front(t);
  }
  return region;
}

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

// Points in a hash grid of cell `cell`.
struct PointGrid {
  double cell;
  std::unordered_map<uint64_t, std::vector<int32_t>> bins;
  explicit PointGrid(double c) : cell(c) {}
  static uint64_t key(int64_t i, int64_t j, int64_t k) {
    return (static_cast<uint64_t>(i + (1 << 20)) << 42) |
           (static_cast<uint64_t>(j + (1 << 20)) << 21) | static_cast<uint64_t>(k + (1 << 20));
  }
  int64_t index(double x) const { return static_cast<int64_t>(std::floor(x / cell)); }
  void add(const double* p, int32_t id) {
    bins[key(index(p[0]), index(p[1]), index(p[2]))].push_back(id);
  }
  // Calls f(id) for the points of the 27 cells about p.
  template <class F>
  void visit(const double* p, F&& f) const {
    const int64_t i = index(p[0]), j = index(p[1]), k = index(p[2]);
    for (int64_t dk = -1; dk <= 1; ++dk)
      for (int64_t dj = -1; dj <= 1; ++dj)
        for (int64_t di = -1; di <= 1; ++di) {
          const auto found = bins.find(key(i + di, j + dj, k + dk));
          if (found == bins.end()) continue;
          for (const int32_t id : found->second) f(id);
        }
  }
};

// ------------------------------------------------------------------ seeds
struct SeedSet {
  std::vector<double> points;
  std::vector<int32_t> regions;  // -1: ghost
  // Mirror pairs (a, b, t): seeds reflected into each other across the
  // (curvature-shifted) plane of triangle t.
  std::vector<std::array<int32_t, 3>> mirrors;
  int32_t interior = 0;
  int32_t lattice = 0;
};

class Seeder {
 public:
  Seeder(const Surface& s, double h)
      : s_(s),
        h_(h),
        r_(kSpacing * h),
        depth_(kDepth * kSpacing * h),
        reach_(kReach * kSpacing * h),
        cluster_(std::cos(s.in->feature_angle)) {
    double largest = 0.0;
    for (int k = 0; k < 3; ++k)
      largest = std::max({largest, std::fabs(s.lower[k]), std::fabs(s.upper[k])});
    largest += 4.0 * s.diameter + 1.0;
    grid_ = std::ldexp(1.0, std::ilogb(largest) - 40);
    triangles_.build(s, std::max(reach_, 0.5 * h));
    columns_.build(s, 0.5 * h);
  }

  double spacing() const { return r_; }
  double reach() const { return reach_; }
  const TriangleGrid& triangles() const { return triangles_; }
  const ColumnGrid& columns() const { return columns_; }
  const std::vector<double>& shifts() const { return shift_; }

  void snap(double* p) const {
    for (int k = 0; k < 3; ++k) p[k] = std::nearbyint(p[k] / grid_) * grid_;
  }

  void interior(SeedSet& out) {
    samples();
    curvature_shifts();
    std::vector<double> points;
    std::vector<int32_t> regions;
    std::vector<std::array<int32_t, 3>> pairs;  // mirrored across triangle t
    const auto add = [&](double* q, int32_t region) -> int64_t {
      snap(q);
      if (region < 0 || region_at(s_, columns_, q) != region) return -1;
      if (triangles_.any_within(q, 0.5 * depth_)) return -1;
      points.insert(points.end(), q, q + 3);
      regions.push_back(region);
      return static_cast<int64_t>(regions.size()) - 1;
    };
    for (const Sample& sample : samples_) {
      switch (sample.kind) {
        case kSmooth: {
          const int32_t t = sample.element;
          const double* n = s_.normal(t);
          int64_t sides[2];
          for (int side = 0; side < 2; ++side) {
            const double sign = side == 0 ? -1.0 : 1.0;
            double q[3];
            for (int k = 0; k < 3; ++k) q[k] = sample.p[k] + (shift_[t] + sign * depth_) * n[k];
            sides[side] = add(q, side == 0 ? s_.back(t) : s_.front(t));
          }
          if (sides[0] >= 0 && sides[1] >= 0)
            pairs.push_back({static_cast<int32_t>(sides[0]), static_cast<int32_t>(sides[1]), t});
          break;
        }
        case kEdge:
          edge_seeds(sample, add);
          break;
        case kCorner:
          corner_seeds(sample, add);
          break;
      }
    }
    // Center the body-centered cubic lattice in the source box. Both
    // sublattices then meet opposing wall seed bands at the same phase,
    // while retaining cell volume h^3 and the same wall clearance.
    const double a = std::cbrt(2.0) * h_;
    double center[3];
    int64_t half[3];
    for (int axis = 0; axis < 3; ++axis) {
      center[axis] = std::midpoint(s_.lower[axis], s_.upper[axis]);
      half[axis] = static_cast<int64_t>(std::ceil((s_.upper[axis] - s_.lower[axis]) / (2.0 * a)));
    }
    for (int shift = 0; shift < 2; ++shift)
      for (int64_t k = -half[2]; k <= half[2] - shift; ++k)
        for (int64_t j = -half[1]; j <= half[1] - shift; ++j)
          for (int64_t i = -half[0]; i <= half[0] - shift; ++i) {
            double q[3] = {center[0] + (static_cast<double>(i) + 0.5 * shift) * a,
                           center[1] + (static_cast<double>(j) + 0.5 * shift) * a,
                           center[2] + (static_cast<double>(k) + 0.5 * shift) * a};
            snap(q);
            const int32_t region = region_at(s_, columns_, q);
            if (region < 0 || triangles_.any_within(q, reach_)) continue;
            points.insert(points.end(), q, q + 3);
            regions.push_back(region);
            ++out.lattice;
          }
    // Exact duplicates keep their first occurrence.
    const int64_t n = static_cast<int64_t>(regions.size());
    std::vector<int64_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int64_t x, int64_t y) {
      for (int k = 0; k < 3; ++k)
        if (points[3 * x + k] != points[3 * y + k]) return points[3 * x + k] < points[3 * y + k];
      return x < y;
    });
    std::vector<uint8_t> keep(n, 1);
    std::vector<int64_t> first(n);  // the kept copy of each seed
    std::iota(first.begin(), first.end(), 0);
    for (int64_t k = 1; k < n; ++k) {
      const int64_t x = order[k - 1], y = order[k];
      if (points[3 * x] == points[3 * y] && points[3 * x + 1] == points[3 * y + 1] &&
          points[3 * x + 2] == points[3 * y + 2]) {
        keep[y] = 0;
        first[y] = first[x];
      }
    }
    std::vector<int32_t> index(n, -1);
    for (int64_t k = 0; k < n; ++k) {
      if (!keep[k]) continue;
      index[k] = static_cast<int32_t>(out.regions.size());
      out.points.insert(out.points.end(), &points[3 * k], &points[3 * k] + 3);
      out.regions.push_back(regions[k]);
    }
    out.interior = static_cast<int32_t>(out.regions.size());
    for (const auto& [a, b, t] : pairs)
      out.mirrors.push_back({index[first[a]], index[first[b]], t});
  }

  // Ghost of seed i reflected across triangle t at the foot q (when valid).
  bool reflect(const SeedSet& seeds, int32_t i, int32_t t, const double* q, double* g) const {
    const double* p = seeds.points.data() + 3 * static_cast<int64_t>(i);
    const double* n = s_.normal(t);
    double d[3];
    geometry::difference(p, q, d);
    const double height = geometry::dot(d, n) - shift_[t];
    const int32_t region = seeds.regions[i];
    if (height > 0.0 ? s_.front(t) != region || s_.back(t) != -1
                     : height < 0.0 ? s_.back(t) != region || s_.front(t) != -1 : true) {
      return false;
    }
    for (int k = 0; k < 3; ++k) g[k] = p[k] - 2.0 * height * n[k];
    snap(g);
    if (region_at(s_, columns_, g) != -1) return false;
    // A mirror is as far from the surface as its seed; near a reflex edge a
    // reflection can land close to another facet (under half its mirror
    // distance), where it would claim interior points: such a ghost is dropped.
    double foot[3], distance = 0.0;
    triangles_.nearest(g, reach_, foot, &distance);
    return distance >= 0.5 * std::fabs(height);
  }

  // The mirror image g of p across the (shifted) plane of triangle t.
  void mirror(const double* p, int32_t t, double* g) const {
    const double* n = s_.normal(t);
    double d[3];
    geometry::difference(p, s_.corner(t, 0), d);
    const double height = geometry::dot(d, n) - shift_[t];
    for (int k = 0; k < 3; ++k) g[k] = p[k] - 2.0 * height * n[k];
    snap(g);
  }

  // Ghosts of every interior seed near the surface, across each of its feet.
  void ghosts(SeedSet& seeds, PointGrid& ghost_grid) {
    struct Foot {
      double d;
      int32_t t;
      double q[3];
    };
    std::vector<Foot> feet;
    std::vector<int32_t> accepted;
    double g[3];
    for (int32_t i = 0; i < seeds.interior; ++i) {
      const double* p = seeds.points.data() + 3 * static_cast<int64_t>(i);
      feet.clear();
      triangles_.visit(p, reach_, [&](int32_t t) {
        Foot f{};
        f.t = t;
        closest_point(p, s_.corner(t, 0), s_.corner(t, 1), s_.corner(t, 2), f.q);
        f.d = std::sqrt(geometry::squared_distance(p, f.q));
        if (f.d <= reach_) feet.push_back(f);
      });
      std::sort(feet.begin(), feet.end(), [](const Foot& a, const Foot& b) {
        return a.d != b.d ? a.d < b.d : a.t < b.t;
      });
      accepted.clear();
      for (std::size_t k = 0; k < feet.size(); ++k) {
        const double* n = s_.normal(feet[k].t);
        bool clustered = false;
        for (const int32_t a : accepted)
          clustered = clustered || geometry::dot(n, s_.normal(feet[a].t)) >= cluster_;
        if (clustered) continue;
        // A local minimum: no nearer foot of a similar normal close by.
        bool dominated = false;
        for (std::size_t j = 0; j < k && !dominated; ++j) {
          dominated = feet[j].d < feet[k].d &&
                      geometry::dot(n, s_.normal(feet[j].t)) >= cluster_ &&
                      geometry::squared_distance(feet[j].q, feet[k].q) < 0.09 * r_ * r_;
        }
        if (dominated) continue;
        accepted.push_back(static_cast<int32_t>(k));
        if (reflect(seeds, i, feet[k].t, feet[k].q, g))
          add_ghost(seeds, ghost_grid, i, feet[k].t, g);
      }
    }
  }

  // Adds ghost g of seed i across triangle t unless it repeats an existing
  // ghost exactly (which then mirrors i as well).
  bool add_ghost(SeedSet& seeds, PointGrid& ghost_grid, int32_t i, int32_t t,
                 const double* g) const {
    int32_t repeated = -1;
    ghost_grid.visit(g, [&](int32_t id) {
      const double* x = seeds.points.data() + 3 * static_cast<int64_t>(id);
      if (x[0] == g[0] && x[1] == g[1] && x[2] == g[2]) repeated = id;
    });
    if (repeated >= 0) {
      seeds.mirrors.push_back({i, repeated, t});
      return false;
    }
    const int32_t id = static_cast<int32_t>(seeds.regions.size());
    seeds.points.insert(seeds.points.end(), g, g + 3);
    seeds.regions.push_back(-1);
    seeds.mirrors.push_back({i, id, t});
    ghost_grid.add(g, id);
    return true;
  }

 private:
  enum Kind : int8_t { kCorner = 0, kEdge = 1, kSmooth = 2 };
  struct Sample {
    double p[3];
    Kind kind;
    int32_t element;  // vertex (corner), input edge (edge) or triangle (smooth)
  };

  const Surface& s_;
  double h_, r_, depth_, reach_, cluster_;
  double grid_ = 1.0;
  TriangleGrid triangles_;
  ColumnGrid columns_;
  std::vector<Sample> samples_;
  std::vector<double> shift_;  // per triangle
  std::vector<int64_t> feature_offsets_;  // feature edges per vertex (CSR)
  std::vector<int32_t> feature_edges_;

  void samples() {
    const Input& in = *s_.in;
    const int64_t m = in.triangle_count;
    std::vector<int32_t> degree(in.vertex_count, 0);
    std::vector<int64_t> adjacency_offsets(in.vertex_count + 1, 0);
    const int32_t edge_count = static_cast<int32_t>(s_.edge_feature.size());
    for (int32_t e = 0; e < edge_count; ++e) {
      if (!s_.edge_feature[e]) continue;
      ++degree[s_.edges[2 * e]];
      ++degree[s_.edges[2 * e + 1]];
      ++adjacency_offsets[s_.edges[2 * e] + 1];
      ++adjacency_offsets[s_.edges[2 * e + 1] + 1];
    }
    for (int64_t v = 0; v < in.vertex_count; ++v) adjacency_offsets[v + 1] += adjacency_offsets[v];
    std::vector<int32_t> adjacency(adjacency_offsets[in.vertex_count]);
    {
      std::vector<int64_t> cursor(adjacency_offsets.begin(), adjacency_offsets.end() - 1);
      for (int32_t e = 0; e < edge_count; ++e) {
        if (!s_.edge_feature[e]) continue;
        adjacency[cursor[s_.edges[2 * e]]++] = e;
        adjacency[cursor[s_.edges[2 * e + 1]]++] = e;
      }
    }
    feature_offsets_ = adjacency_offsets;
    feature_edges_ = adjacency;
    PointGrid grid(r_);
    const auto push = [&](const double* p, Kind kind, int32_t element) {
      Sample sample{};
      std::copy_n(p, 3, sample.p);
      sample.kind = kind;
      sample.element = element;
      grid.add(p, static_cast<int32_t>(samples_.size()));
      samples_.push_back(sample);
    };
    const auto near = [&](const double* p, double radius) {
      bool found = false;
      grid.visit(p, [&](int32_t id) {
        found = found || geometry::squared_distance(samples_[id].p, p) < radius * radius;
      });
      return found;
    };
    for (int32_t v = 0; v < in.vertex_count; ++v)
      if (degree[v] != 0 && degree[v] != 2) push(s_.point(v), kCorner, v);
    // Feature chains at spacing about r.
    std::vector<uint8_t> used(edge_count, 0);
    const auto walk = [&](int32_t start, int32_t first) {
      std::vector<int32_t> chain{start}, chain_edges;
      int32_t v = start, e = first;
      while (true) {
        used[e] = 1;
        chain_edges.push_back(e);
        const int32_t w = s_.edges[2 * e] == v ? s_.edges[2 * e + 1] : s_.edges[2 * e];
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
        length += std::sqrt(geometry::squared_distance(s_.point(chain[k - 1]), s_.point(chain[k])));
      const bool closed = chain.front() == chain.back() && degree[chain.front()] == 2;
      const int64_t pieces = std::max<int64_t>(1, std::llround(length / r_));
      const double step = length / static_cast<double>(pieces);
      double travelled = 0.0;
      std::size_t segment = 1;
      for (int64_t q = closed ? 0 : 1; q < pieces; ++q) {
        const double target = step * static_cast<double>(q);
        while (segment < chain.size()) {
          const double piece = std::sqrt(
              geometry::squared_distance(s_.point(chain[segment - 1]), s_.point(chain[segment])));
          if (travelled + piece >= target) break;
          travelled += piece;
          ++segment;
        }
        if (segment >= chain.size()) break;
        const double* a = s_.point(chain[segment - 1]);
        const double* b = s_.point(chain[segment]);
        const double piece = std::sqrt(geometry::squared_distance(a, b));
        const double u = piece > 0.0 ? std::clamp((target - travelled) / piece, 0.0, 1.0) : 0.0;
        double p[3];
        for (int k = 0; k < 3; ++k) p[k] = a[k] + u * (b[k] - a[k]);
        if (!near(p, 0.5 * r_)) push(p, kEdge, chain_edges[segment - 1]);
      }
    };
    for (int32_t v = 0; v < in.vertex_count; ++v) {
      if (degree[v] == 0 || degree[v] == 2) continue;
      for (int64_t k = adjacency_offsets[v]; k < adjacency_offsets[v + 1]; ++k)
        if (!used[adjacency[k]]) walk(v, adjacency[k]);
    }
    for (int32_t e = 0; e < edge_count; ++e)
      if (s_.edge_feature[e] && !used[e]) walk(s_.edges[2 * e], e);
    // Smooth candidates: barycentric points at spacing at most r / 3.
    struct Candidate {
      uint64_t key;
      int64_t index;
      double p[3];
      int32_t triangle;
    };
    std::vector<Candidate> candidates;
    const double scale = 2097151.0 / std::max(s_.diameter, 1e-300);
    for (int64_t t = 0; t < m; ++t) {
      double longest = 0.0;
      for (int k = 0; k < 3; ++k)
        longest =
            std::max(longest, geometry::squared_distance(s_.corner(t, k), s_.corner(t, (k + 1) % 3)));
      // Interior barycentric points; a triangle of at most twice the
      // candidate spacing offers its centroid.
      const int64_t K =
          std::max<int64_t>(1, static_cast<int64_t>(std::ceil(std::sqrt(longest) / (r_ / 3.0))));
      const auto offer = [&](const double* w) {
        Candidate c{};
        for (int a = 0; a < 3; ++a)
          c.p[a] = w[0] * s_.corner(t, 0)[a] + w[1] * s_.corner(t, 1)[a] + w[2] * s_.corner(t, 2)[a];
        c.key = morton(c.p, s_.lower, scale);
        c.index = static_cast<int64_t>(candidates.size());
        c.triangle = static_cast<int32_t>(t);
        candidates.push_back(c);
      };
      if (K <= 2) {
        const double w[3] = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
        offer(w);
        continue;
      }
      for (int64_t i = 1; i < K; ++i)
        for (int64_t j = 1; i + j < K; ++j) {
          const double w[3] = {static_cast<double>(i) / K, static_cast<double>(j) / K,
                               static_cast<double>(K - i - j) / K};
          offer(w);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
      return a.key != b.key ? a.key < b.key : a.index < b.index;
    });
    for (const Candidate& c : candidates)
      if (!near(c.p, r_)) push(c.p, kSmooth, c.triangle);
    // Holes of the maximal sample (empty disks up to radius r) bound the
    // deviation of the bisector surface; a second pass at 3r/4 fills them.
    for (const Candidate& c : candidates)
      if (!near(c.p, 0.75 * r_)) push(c.p, kSmooth, c.triangle);
  }

  void curvature_shifts() {
    const int64_t m = s_.in->triangle_count;
    std::vector<double> sample_shift(samples_.size(), 0.0);
    PointGrid smooth(1.5 * r_);
    for (std::size_t k = 0; k < samples_.size(); ++k)
      if (samples_[k].kind == kSmooth) smooth.add(samples_[k].p, static_cast<int32_t>(k));
    for (std::size_t k = 0; k < samples_.size(); ++k) {
      const Sample& a = samples_[k];
      if (a.kind != kSmooth) continue;
      const int32_t t = a.element;
      double total = 0.0;
      int32_t count = 0;
      smooth.visit(a.p, [&](int32_t id) {
        const Sample& b = samples_[id];
        if (static_cast<std::size_t>(id) == k || s_.patch[b.element] != s_.patch[t] ||
            geometry::squared_distance(a.p, b.p) >= 2.25 * r_ * r_) {
          return;
        }
        ++count;
        if (s_.plane_class[b.element] == s_.plane_class[t]) return;
        double d[3];
        geometry::difference(b.p, a.p, d);
        total += geometry::dot(d, s_.normal(t));
      });
      if (count) sample_shift[k] = kShift * total / count;
    }
    // Each triangle takes the shift of the nearest smooth sample of its patch.
    shift_.assign(m, 0.0);
    for (int64_t t = 0; t < m; ++t) {
      double c[3];
      for (int k = 0; k < 3; ++k)
        c[k] = (s_.corner(t, 0)[k] + s_.corner(t, 1)[k] + s_.corner(t, 2)[k]) / 3.0;
      double best = HUGE_VAL;
      int32_t chosen = -1;
      smooth.visit(c, [&](int32_t id) {
        if (s_.patch[samples_[id].element] != s_.patch[t]) return;
        const double d = geometry::squared_distance(samples_[id].p, c);
        if (d < best || (d == best && id < chosen)) {
          best = d;
          chosen = id;
        }
      });
      if (chosen >= 0) shift_[t] = sample_shift[chosen];
    }
  }

  template <class Add>
  void edge_seeds(const Sample& sample, Add& add) {
    const int32_t e = sample.element;
    const double* a = s_.point(s_.edges[2 * e]);
    const double* b = s_.point(s_.edges[2 * e + 1]);
    double tangent[3];
    geometry::difference(b, a, tangent);
    const double length = geometry::length(tangent);
    for (double& x : tangent) x /= length;
    // An orthonormal frame (b1, b2) of the normal plane, b1 x b2 = tangent.
    double b1[3] = {0, 0, 0}, b2[3];
    int axis = 0;
    for (int k = 1; k < 3; ++k)
      if (std::fabs(tangent[k]) < std::fabs(tangent[axis])) axis = k;
    b1[axis] = 1.0;
    const double along = geometry::dot(b1, tangent);
    for (int k = 0; k < 3; ++k) b1[k] -= along * tangent[k];
    const double b1_length = geometry::length(b1);
    for (double& x : b1) x /= b1_length;
    geometry::cross(tangent, b1, b2);
    struct Wing {
      double angle;
      int32_t region;  // counterclockwise of the wing
    };
    std::vector<Wing> wings;
    for (int64_t k = s_.edge_offsets[e]; k < s_.edge_offsets[e + 1]; ++k) {
      const int32_t t = s_.edge_items[k];
      int32_t third = -1;
      for (int c = 0; c < 3; ++c) {
        const int32_t v = s_.in->triangles[3 * static_cast<int64_t>(t) + c];
        if (v != s_.edges[2 * e] && v != s_.edges[2 * e + 1]) third = v;
      }
      double d[3];
      geometry::difference(s_.point(third), a, d);
      const double x = geometry::dot(d, b1), y = geometry::dot(d, b2);
      // Rotating the wing direction by +90 degrees about the tangent.
      double rotated[3];
      for (int c = 0; c < 3; ++c) rotated[c] = -y * b1[c] + x * b2[c];
      const bool front = geometry::dot(s_.normal(t), rotated) > 0.0;
      wings.push_back({std::atan2(y, x), front ? s_.front(t) : s_.back(t)});
    }
    std::sort(wings.begin(), wings.end(), [](const Wing& x, const Wing& y) {
      return x.angle < y.angle;
    });
    const std::size_t count = wings.size();
    double smallest = 2.0 * kPi;
    for (std::size_t k = 0; k < count; ++k) {
      const double next = k + 1 < count ? wings[k + 1].angle : wings[0].angle + 2.0 * kPi;
      smallest = std::min(smallest, next - wings[k].angle);
    }
    const double offset = std::min(0.5 * smallest, 0.25 * kPi);
    if (!(offset > 1e-6)) return;
    const double radius = std::min(depth_ / std::sin(offset), r_);
    for (std::size_t k = 0; k < count; ++k) {
      if (wings[k].region < 0) continue;
      const double start = wings[k].angle;
      const double end = k + 1 < count ? wings[k + 1].angle : wings[0].angle + 2.0 * kPi;
      const double inner = end - start - 2.0 * offset;
      const int64_t gaps =
          inner > 1e-9 ? static_cast<int64_t>(std::ceil(inner / (0.5 * kPi))) : 0;
      for (int64_t g = 0; g <= gaps; ++g) {
        const double angle =
            start + offset + (gaps ? inner * static_cast<double>(g) / static_cast<double>(gaps) : 0.0);
        double q[3];
        for (int c = 0; c < 3; ++c)
          q[c] = sample.p[c] + radius * (std::cos(angle) * b1[c] + std::sin(angle) * b2[c]);
        add(q, wings[k].region);
      }
    }
  }

  // Corner: the sector seeds of every incident feature edge at distance d
  // from the corner along that edge (on a cube corner the three coincide).
  template <class Add>
  void corner_seeds(const Sample& sample, Add& add) {
    const int32_t v = sample.element;
    for (int64_t k = feature_offsets_[v]; k < feature_offsets_[v + 1]; ++k) {
      const int32_t e = feature_edges_[k];
      const int32_t w = s_.edges[2 * e] == v ? s_.edges[2 * e + 1] : s_.edges[2 * e];
      double u[3];
      geometry::difference(s_.point(w), s_.point(v), u);
      const double length = geometry::length(u);
      Sample near{};
      for (int c = 0; c < 3; ++c) near.p[c] = sample.p[c] + std::min(depth_, 0.5 * length) * u[c] / length;
      near.kind = kEdge;
      near.element = e;
      edge_seeds(near, add);
    }
  }
};

// ------------------------------------------------------------------ builder
// A seed to move by `shift`, then its mirror images re-reflected in order
// (child, parent, triangle): the child becomes the parent's mirror across the
// plane of the triangle. For the nearly cospherical apex pair (a, b), applied
// at its stage: the rigid moves (every mirror pair kept) of the seeds of the
// two tets in turn while the pair persists, then the lone move (`lone`).
struct Perturbation {
  int32_t seed;
  std::array<double, 3> shift;
  std::vector<std::array<int32_t, 3>> mirrors;
  int32_t a, b;
  int32_t stage;
  bool lone;
};

class Builder {
 public:
  Builder(const Surface& surface, const Seeder& seeder, const SeedSet& seeds, double h)
      : s_(surface), seeder_(seeder), seeds_(seeds.points), regions_(seeds.regions), h_(h) {
    // Mirror adjacency: (seed, triangle) partners of each seed, repeats dropped.
    const std::size_t n = regions_.size();
    std::vector<std::vector<std::pair<int32_t, int32_t>>> lists(n);
    for (const auto& [a, b, t] : seeds.mirrors) {
      if (a == b) continue;
      for (const auto& [x, y] : {std::pair{a, b}, std::pair{b, a}}) {
        auto& list = lists[x];
        if (std::none_of(list.begin(), list.end(), [&](const auto& e) { return e.first == y; }))
          list.emplace_back(y, t);
      }
    }
    mirror_offsets_.assign(n + 1, 0);
    for (std::size_t i = 0; i < n; ++i) {
      mirror_offsets_[i + 1] = mirror_offsets_[i] + static_cast<int32_t>(lists[i].size());
      mirror_items_.insert(mirror_items_.end(), lists[i].begin(), lists[i].end());
    }
  }

  void run(phx_mc_voronoi_cells& out, double* timings) {
    auto clock = Clock::now();
    add_box();
    delaunay();
    timings[0] += seconds_since(clock);
    clock = Clock::now();
    cells();
    timings[1] += seconds_since(clock);
    clock = Clock::now();
    publish(out);
    timings[2] += seconds_since(clock);
  }

  // Interior seeds whose cells leak, with an offending point each.
  const std::vector<std::pair<int32_t, std::array<double, 3>>>& leaks() const { return leaks_; }
  // Nearly cospherical configurations and the seed moves that resolve them.
  const std::vector<Perturbation>& near() const { return near_; }
  void certify(const phx_mc_voronoi_cells& out);
  bool certified() const { return certified_; }
  int64_t open_edges() const { return open_edges_; }
  int64_t violations() const { return violations_; }
  double distance_bound() const { return bound_; }
  double vertex_distance() const { return vertex_distance_; }
  int32_t boundary_cells() const { return boundary_cells_; }
  int32_t boundary_faces() const { return boundary_faces_; }
  bool boundary_exact() const { return boundary_exact_; }
  double volume_magnitude() const { return volume_magnitude_; }

 private:
  // The mirror pairs reachable from seed s as (child, parent, triangle) in
  // breadth-first order; false when they close a cycle.
  bool mirror_tree(int32_t s, std::vector<std::array<int32_t, 3>>& tree) const {
    tree.clear();
    std::vector<int32_t> nodes{s};
    int64_t entries = 0;
    for (std::size_t k = 0; k < nodes.size(); ++k) {
      const int32_t parent = nodes[k];
      for (int32_t e = mirror_offsets_[parent]; e < mirror_offsets_[parent + 1]; ++e) {
        ++entries;
        const auto [child, t] = mirror_items_[e];
        if (std::find(nodes.begin(), nodes.end(), child) != nodes.end()) continue;
        nodes.push_back(child);
        tree.push_back({child, parent, t});
      }
    }
    return entries == 2 * static_cast<int64_t>(tree.size());
  }

  const Surface& s_;
  const Seeder& seeder_;
  std::vector<double> seeds_;
  std::vector<int32_t> regions_;  // -1 ghost, -2 box
  std::vector<int32_t> mirror_offsets_;  // seeds before the box
  std::vector<std::pair<int32_t, int32_t>> mirror_items_;  // (mirror seed, triangle)
  double h_;
  bool boundary_exact_ = true;
  double volume_magnitude_ = 0.0;  // sum of |fan volume| over the cells
  std::vector<Perturbation> near_;
  int32_t count_ = 0;  // seeds before the box
  std::vector<int32_t> tets_;
  std::vector<int32_t> tet_neighbors_;
  std::vector<int64_t> star_offsets_;
  std::vector<int32_t> star_;
  std::vector<int32_t> tet_root_;
  std::vector<int32_t> tet_vertex_;
  std::vector<int32_t> vertex_tets_;  // representative tet per vertex
  std::vector<int32_t> cell_of_seed_;
  std::vector<int32_t> face_owner_seed_;
  std::vector<std::pair<int32_t, std::array<double, 3>>> leaks_;
  bool certified_ = false;
  int64_t open_edges_ = 0;
  int64_t violations_ = 0;
  double bound_ = -1.0;
  double vertex_distance_ = 0.0;
  int32_t boundary_cells_ = 0;
  int32_t boundary_faces_ = 0;

  const double* seed(int32_t i) const { return seeds_.data() + 3 * static_cast<int64_t>(i); }
  bool interior(int32_t i) const { return i < count_ && regions_[i] >= 0; }

  void add_box() {
    count_ = static_cast<int32_t>(regions_.size());
    double lower[3], upper[3];
    for (int k = 0; k < 3; ++k) {
      lower[k] = s_.lower[k];
      upper[k] = s_.upper[k];
    }
    for (int32_t i = 0; i < count_; ++i)
      for (int k = 0; k < 3; ++k) {
        lower[k] = std::min(lower[k], seed(i)[k]);
        upper[k] = std::max(upper[k], seed(i)[k]);
      }
    double extent = 0.0;
    for (int k = 0; k < 3; ++k) extent = std::max(extent, upper[k] - lower[k]);
    const double margin = 4.0 * extent + 1.0;
    for (int corner = 0; corner < 8; ++corner) {
      double p[3];
      for (int k = 0; k < 3; ++k)
        p[k] = ((corner >> k) & 1 ? upper[k] + margin : lower[k] - margin) +
               (0.001 * (k + 1) + 0.0007 * corner) * margin;
      seeder_.snap(p);
      seeds_.insert(seeds_.end(), p, p + 3);
      regions_.push_back(-2);
    }
  }

  void delaunay() {
    const int64_t n = static_cast<int64_t>(regions_.size());
    phx_mc_mesh* raw = nullptr;
    const int32_t status = triangulate_3d(n, seeds_.data(), nullptr, false, false, &raw, 1);
    std::unique_ptr<phx_mc_mesh> mesh(raw);
    if (status != PHX_MC_OK) throw Failure(status);
    for (int64_t v = 0; v < n; ++v)
      if (mesh->vertex_map[v] != v) throw Failure(PHX_MC_INTERNAL_ERROR);  // duplicate seed
    tets_ = std::move(mesh->cells);
    const int64_t count = static_cast<int64_t>(tets_.size() / 4);
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
    star_offsets_.assign(n + 1, 0);
    for (int64_t t = 0; t < count; ++t)
      for (int a = 0; a < 4; ++a) ++star_offsets_[tets_[4 * t + a] + 1];
    for (int64_t v = 0; v < n; ++v) star_offsets_[v + 1] += star_offsets_[v];
    star_.resize(star_offsets_[n]);
    std::vector<int64_t> cursor(star_offsets_.begin(), star_offsets_.end() - 1);
    for (int64_t t = 0; t < count; ++t)
      for (int a = 0; a < 4; ++a) star_[cursor[tets_[4 * t + a]]++] = static_cast<int32_t>(t);
  }

  bool relevant(int32_t t) const {
    for (int v = 0; v < 4; ++v)
      if (interior(tets_[4 * static_cast<int64_t>(t) + v])) return true;
    return false;
  }

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

  // Exact circumcenter p0 + (|a|^2 b x c + |b|^2 c x a + |c|^2 a x b) / (2 a . b x c).
  void exact_circumcenter(int32_t t, std::array<Expansion, 3>& n, Expansion& d) const {
    const int32_t* tet = tets_.data() + 4 * static_cast<int64_t>(t);
    const double* p0 = seed(tet[0]);
    std::array<std::array<Expansion, 3>, 3> e;
    for (int r = 0; r < 3; ++r)
      for (int k = 0; k < 3; ++k) e[r][k] = Expansion::difference(seed(tet[r + 1])[k], p0[k]);
    const auto cross = [](const std::array<Expansion, 3>& x, const std::array<Expansion, 3>& y) {
      return std::array<Expansion, 3>{x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2],
                                      x[0] * y[1] - x[1] * y[0]};
    };
    const auto dot = [](const std::array<Expansion, 3>& x, const std::array<Expansion, 3>& y) {
      return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    };
    auto bc = cross(e[1], e[2]), ca = cross(e[2], e[0]), ab = cross(e[0], e[1]);
    for (int k = 0; k < 3; ++k) {
      bc[k] = compressed(bc[k]);
      ca[k] = compressed(ca[k]);
      ab[k] = compressed(ab[k]);
    }
    const Expansion la = compressed(dot(e[0], e[0]));
    const Expansion lb = compressed(dot(e[1], e[1]));
    const Expansion lc = compressed(dot(e[2], e[2]));
    d = compressed(dot(e[0], bc).scaled(2.0));
    for (int k = 0; k < 3; ++k) n[k] = compressed(d.scaled(p0[k]) + la * bc[k] + lb * ca[k] + lc * ab[k]);
  }

  void rounded_circumcenter(int32_t t, double* out) const {
    std::array<Expansion, 3> n;
    Expansion d;
    exact_circumcenter(t, n, d);
    for (int k = 0; k < 3; ++k) out[k] = rounded_quotient(n[k], d);
  }

  // Whether the exact vertex v lies on the plane of input triangle t.
  bool on_plane(int32_t v, int32_t t) const {
    std::array<Expansion, 3> n;
    Expansion d;
    exact_circumcenter(vertex_tets_[v], n, d);
    const double* a = s_.corner(t, 0);
    std::array<Expansion, 3> u, w, x;
    for (int k = 0; k < 3; ++k) {
      u[k] = Expansion::difference(s_.corner(t, 1)[k], a[k]);
      w[k] = Expansion::difference(s_.corner(t, 2)[k], a[k]);
      x[k] = n[k] - d.scaled(a[k]);
    }
    const Expansion volume = (u[1] * w[2] - u[2] * w[1]) * x[0] +
                             (u[2] * w[0] - u[0] * w[2]) * x[1] +
                             (u[0] * w[1] - u[1] * w[0]) * x[2];
    return volume.sign() == 0;
  }

  int32_t apex(int32_t t, int slot) const {
    const int32_t u = tet_neighbors_[4 * static_cast<int64_t>(t) + slot];
    const int32_t* f = tets_.data() + 4 * static_cast<int64_t>(t);
    const int32_t* g = tets_.data() + 4 * static_cast<int64_t>(u);
    for (int v = 0; v < 4; ++v)
      if (g[v] != f[0] && g[v] != f[1] && g[v] != f[2] && g[v] != f[3]) return g[v];
    throw Failure(PHX_MC_INTERNAL_ERROR);
  }

  int insphere_of(int32_t t, int32_t e) const {
    const int32_t* f = tets_.data() + 4 * static_cast<int64_t>(t);
    return insphere(seed(f[0]), seed(f[1]), seed(f[2]), seed(f[3]), seed(e));
  }

  // Ring of tets around Delaunay edge (i, j), counterclockwise about s_j - s_i.
  void ring(int32_t i, int32_t j, int32_t start, std::vector<int32_t>& tets) const {
    tets.clear();
    int32_t t = start;
    int32_t a = -1, b = -1;
    for (int v = 0; v < 4; ++v) {
      const int32_t w = tets_[4 * static_cast<int64_t>(t) + v];
      if (w == i || w == j) continue;
      (a < 0 ? a : b) = w;
    }
    if (orient3d(seed(i), seed(j), seed(a), seed(b)) < 0) std::swap(a, b);
    while (true) {
      tets.push_back(t);
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
  }

  // Published cells and faces (loops over tet-class vertices).
  std::vector<int32_t> loops_;
  std::vector<int64_t> loop_offsets_{0};
  std::vector<int32_t> loop_cells_;   // owner cell, neighbour cell (-1 exterior)
  std::vector<int32_t> loop_seeds_;   // owner seed, neighbour seed
  std::vector<int32_t> cell_seed_list_;

  int32_t vertex_of(int32_t t) {
    const int32_t root = find_root(tet_root_, t);
    if (tet_vertex_[root] < 0) {
      tet_vertex_[root] = static_cast<int32_t>(vertex_tets_.size());
      vertex_tets_.push_back(root);
    }
    return tet_vertex_[root];
  }

  void cells() {
    const int64_t count = static_cast<int64_t>(tets_.size() / 4);
    // Vertices: classes of adjacent relevant tets that are exactly
    // cospherical or whose circumcenters round to within the merge radius.
    tet_root_.resize(count);
    std::iota(tet_root_.begin(), tet_root_.end(), 0);
    tet_vertex_.assign(count, -1);
    std::vector<uint8_t> relevant_tet(count, 0);
    for (int64_t t = 0; t < count; ++t) relevant_tet[t] = relevant(static_cast<int32_t>(t));
    for (int64_t t = 0; t < count; ++t) {
      if (!relevant_tet[t]) continue;
      for (int k = 0; k < 4; ++k) {
        const int32_t u = tet_neighbors_[4 * t + k];
        if (u <= t || !relevant_tet[u]) continue;
        if (insphere_of(static_cast<int32_t>(t), apex(static_cast<int32_t>(t), k)) == 0)
          unite(tet_root_, static_cast<int32_t>(t), u);
      }
    }
    // Nearly cospherical seeds (mirror pairs about planes that are not
    // axis-aligned, after the dyadic snap) put adjacent circumcenters within
    // the resolution: such a pair is recorded so that build() moves an apex
    // seed off the other tet's circumsphere and rebuilds. Thresholds: the
    // validity certificate's star determinants scale as (edge / R)^2 / 6 and
    // must clear 2^-35 with margin, so edges under ~3e-5 R are unresolved;
    // detection at 2^-14 h covers them and a 2^-10 h move clears it.
    const double resolution = kResolution * h_;
    std::vector<std::array<double, 3>> center(count);
    for (int64_t t = 0; t < count; ++t)
      if (relevant_tet[t]) circumcenter(static_cast<int32_t>(t), center[t].data());
    near_.clear();
    for (int64_t t = 0; t < count; ++t) {
      if (!relevant_tet[t]) continue;
      for (int k = 0; k < 4; ++k) {
        const int32_t u = tet_neighbors_[4 * t + k];
        if (u <= t || !relevant_tet[u]) continue;
        if (find_root(tet_root_, static_cast<int32_t>(t)) == find_root(tet_root_, u)) continue;
        // The binary64 circumcenters screen; the rounded exact ones decide.
        if (!(geometry::squared_distance(center[t].data(), center[u].data()) <=
              16.0 * resolution * resolution))
          continue;
        double a[3], b[3];
        rounded_circumcenter(static_cast<int32_t>(t), a);
        rounded_circumcenter(u, b);
        if (geometry::squared_distance(a, b) > resolution * resolution) continue;
        const int32_t l = tets_[4 * t + k], m = apex(static_cast<int32_t>(t), k);
        // Rigid and lone moves, see the header (Resolution).
        const auto shift_of = [&](int32_t s, std::array<double, 3>& shift) {
          // kNudge h outward along the ray from the other tet's circumcenter.
          const double* c = center[s == l ? u : t].data();
          double ray[3];
          geometry::difference(seed(s), c, ray);
          const double length = geometry::length(ray);
          for (int k3 = 0; k3 < 3; ++k3) shift[k3] = kNudge * h_ * ray[k3] / length;
          return length > 0.0;
        };
        const bool l_first = regions_[l] == -1 && regions_[m] != -1 ? true
                             : regions_[m] == -1                     ? false
                                                                     : l > m;
        // The apex seeds (preferred order), then the seeds of the shared face.
        int32_t candidates[5] = {l_first ? l : m, l_first ? m : l, -1, -1, -1};
        for (int j = 0, n = 2; j < 4; ++j)
          if (j != k) candidates[n++] = tets_[4 * t + j];
        const int32_t lo = std::min(l, m), hi = std::max(l, m);
        int32_t stage = 0;
        std::vector<std::array<int32_t, 3>> tree;
        for (int pass = 0; pass < 2; ++pass)
          for (const int32_t s : candidates) {
            if (regions_[s] == -2) continue;
            const bool mirrored = mirror_offsets_[s + 1] > mirror_offsets_[s];
            if ((pass == 0) == mirrored || (mirrored && !mirror_tree(s, tree))) continue;
            Perturbation p{s, {}, pass == 0 ? decltype(tree){} : tree, lo, hi, stage, false};
            if (!shift_of(s, p.shift)) continue;
            near_.push_back(std::move(p));
            ++stage;
          }
        if (regions_[candidates[0]] == -2) continue;
        Perturbation p{candidates[0], {}, {}, lo, hi, stage, true};
        if (shift_of(candidates[0], p.shift)) near_.push_back(std::move(p));
      }
    }
    cell_of_seed_.assign(count_, -1);
    for (int32_t i = 0; i < count_; ++i) {
      if (!interior(i)) continue;
      cell_of_seed_[i] = static_cast<int32_t>(cell_seed_list_.size());
      cell_seed_list_.push_back(i);
    }
    std::vector<int32_t> ring_tets, loop;
    std::vector<int32_t> neighbor_tet;
    for (const int32_t i : cell_seed_list_) {
      // Delaunay neighbours of i with a representative tet each.
      std::vector<std::pair<int32_t, int32_t>> around;
      bool leak = false;
      for (int64_t k = star_offsets_[i]; k < star_offsets_[i + 1]; ++k) {
        const int32_t t = star_[k];
        for (int v = 0; v < 4; ++v) {
          const int32_t j = tets_[4 * static_cast<int64_t>(t) + v];
          if (j == i) continue;
          around.emplace_back(j, t);
          if (regions_[j] == -2) leak = true;
        }
      }
      if (leak) {
        // The farthest circumcenter of the star shows where the cell escapes.
        std::array<double, 3> far{};
        double best = -1.0;
        for (int64_t k = star_offsets_[i]; k < star_offsets_[i + 1]; ++k) {
          double p[3];
          circumcenter(star_[k], p);
          const double d = geometry::squared_distance(p, seed(i));
          if (std::isfinite(d) && d > best) {
            best = d;
            std::copy_n(p, 3, far.data());
          }
        }
        leaks_.emplace_back(i, far);
        continue;
      }
      std::sort(around.begin(), around.end());
      for (std::size_t k = 0; k < around.size(); ++k) {
        if (k > 0 && around[k].first == around[k - 1].first) continue;
        const int32_t j = around[k].first;
        if (interior(j) && j < i) continue;  // emitted from j
        ring(i, j, around[k].second, ring_tets);
        loop.clear();
        for (const int32_t t : ring_tets) {
          const int32_t v = vertex_of(t);
          if (loop.empty() || loop.back() != v) loop.push_back(v);
        }
        while (loop.size() > 1 && loop.front() == loop.back()) loop.pop_back();
        if (loop.size() < 3) continue;
        loops_.insert(loops_.end(), loop.begin(), loop.end());
        loop_offsets_.push_back(static_cast<int64_t>(loops_.size()));
        loop_cells_.push_back(cell_of_seed_[i]);
        loop_cells_.push_back(interior(j) ? cell_of_seed_[j] : -1);
        loop_seeds_.push_back(i);
        loop_seeds_.push_back(j);
      }
    }
  }

  void publish(phx_mc_voronoi_cells& out);
  mutable double lower_ = 0.0;  // largest sampled boundary distance (certify)
  double piece_bound(const double* a, const double* b, const double* c) const;
};

void Builder::publish(phx_mc_voronoi_cells& out) {
  const int32_t vertex_count = static_cast<int32_t>(vertex_tets_.size());
  out.vertices.assign(3 * static_cast<std::size_t>(vertex_count), 0.0);
  out.exact_offsets.assign(1, 0);
  out.exact_values.clear();
  std::array<Expansion, 3> numerator;
  Expansion denominator;
  for (int32_t v = 0; v < vertex_count; ++v) {
    exact_circumcenter(vertex_tets_[v], numerator, denominator);
    for (int k = 0; k < 3; ++k)
      out.vertices[3 * static_cast<int64_t>(v) + k] = rounded_quotient(numerator[k], denominator);
    for (const Expansion* e : {&numerator[0], &numerator[1], &numerator[2], &denominator}) {
      out.exact_values.insert(out.exact_values.end(), e->data(), e->data() + e->size());
      out.exact_offsets.push_back(static_cast<int64_t>(out.exact_values.size()));
    }
  }
  const int64_t faces = static_cast<int64_t>(loop_offsets_.size()) - 1;
  out.face_offsets = loop_offsets_;
  out.face_vertices = loops_;
  out.face_cells = loop_cells_;
  out.face_facets.assign(faces, -1);
  // Boundary vertices: nearest triangle; faces between regions or towards a
  // ghost take the facet of the triangle nearest their vertex centroid.
  out.vertex_triangles.assign(vertex_count, -1);
  std::vector<double> vertex_distance(vertex_count, 0.0);
  const TriangleGrid& grid = seeder_.triangles();
  const double reach = seeder_.reach();
  std::vector<uint8_t> leaking(count_, 0);
  boundary_faces_ = 0;
  vertex_distance_ = 0.0;
  std::vector<uint8_t> boundary_cell(cell_seed_list_.size(), 0);
  for (int64_t f = 0; f < faces; ++f) {
    const int32_t owner = loop_cells_[2 * f], neighbor = loop_cells_[2 * f + 1];
    const bool exterior = neighbor < 0;
    if (!exterior && regions_[loop_seeds_[2 * f]] == regions_[loop_seeds_[2 * f + 1]]) continue;
    if (exterior) {
      ++boundary_faces_;
      boundary_cell[owner] = 1;
    }
    double centroid[3] = {0, 0, 0};
    const int64_t begin = loop_offsets_[f], end = loop_offsets_[f + 1];
    for (int64_t q = begin; q < end; ++q) {
      const int32_t v = loops_[q];
      const double* x = out.vertices.data() + 3 * static_cast<int64_t>(v);
      for (int k = 0; k < 3; ++k) centroid[k] += x[k] / static_cast<double>(end - begin);
      if (out.vertex_triangles[v] < 0) {
        double foot[3], d = 0.0;
        out.vertex_triangles[v] = grid.nearest(x, reach, foot, &d);
        vertex_distance[v] = d;
      }
      if (exterior) {
        vertex_distance_ = std::max(vertex_distance_, vertex_distance[v]);
        const int32_t i = loop_seeds_[2 * f];
        if (vertex_distance[v] > reach && !leaking[i]) {
          leaking[i] = 1;
          leaks_.emplace_back(i, std::array<double, 3>{x[0], x[1], x[2]});
        }
      }
    }
    double foot[3], d = 0.0;
    const int32_t t = grid.nearest(centroid, reach, foot, &d);
    out.face_facets[f] = t >= 0 ? s_.in->facets[t] : -1;
  }
  boundary_exact_ = true;
  for (int32_t v = 0; v < vertex_count && boundary_exact_; ++v)
    if (out.vertex_triangles[v] >= 0 && !on_plane(v, out.vertex_triangles[v])) boundary_exact_ = false;
  boundary_cells_ = 0;
  for (const uint8_t b : boundary_cell) boundary_cells_ += b;
  const std::size_t cells = cell_seed_list_.size();
  out.cell_seeds = cell_seed_list_;
  out.cell_regions.resize(cells);
  out.cell_volumes.assign(cells, 0.0);
  out.cell_centroids.assign(3 * cells, 0.0);
  for (std::size_t c = 0; c < cells; ++c) out.cell_regions[c] = regions_[cell_seed_list_[c]];
  // Volumes and centroids by fans from each cell's seed.
  for (int64_t f = 0; f < faces; ++f) {
    const int64_t begin = loop_offsets_[f], end = loop_offsets_[f + 1];
    for (int side = 0; side < 2; ++side) {
      const int32_t cell = loop_cells_[2 * f + side];
      if (cell < 0) continue;
      const double* o = seed(cell_seed_list_[cell]);
      const double sign = side == 0 ? 1.0 : -1.0;
      const double* x0 = out.vertices.data() + 3 * static_cast<int64_t>(loops_[begin]);
      for (int64_t q = begin + 1; q + 1 < end; ++q) {
        const double* x1 = out.vertices.data() + 3 * static_cast<int64_t>(loops_[q]);
        const double* x2 = out.vertices.data() + 3 * static_cast<int64_t>(loops_[q + 1]);
        double a[3], b[3], c[3], bc[3];
        geometry::difference(x0, o, a);
        geometry::difference(x1, o, b);
        geometry::difference(x2, o, c);
        geometry::cross(b, c, bc);
        const double volume = sign * geometry::dot(a, bc) / 6.0;
        out.cell_volumes[cell] += volume;
        volume_magnitude_ += std::fabs(volume);
        for (int k = 0; k < 3; ++k) out.cell_centroids[3 * cell + k] += volume * (a[k] + b[k] + c[k]) / 4.0;
      }
    }
  }
  for (std::size_t c = 0; c < cells; ++c) {
    const double* o = seed(cell_seed_list_[c]);
    for (int k = 0; k < 3; ++k) {
      out.cell_centroids[3 * c + k] =
          o[k] + (out.cell_volumes[c] != 0.0 ? out.cell_centroids[3 * c + k] / out.cell_volumes[c] : 0.0);
    }
  }
  out.seeds.assign(seeds_.begin(), seeds_.begin() + 3 * static_cast<int64_t>(count_));
  out.seed_regions.assign(regions_.begin(), regions_.begin() + count_);
}

// Upper bound of the distance to the surface over triangle (a, b, c): the
// distance to one triangle is convex, so its maximum over the piece is at a
// corner. Branch and bound: the distance at every piece centroid raises the
// lower bound `lower_`; a piece splits in four while its bound exceeds the
// lower bound by more than 1/16 (plus a rounding allowance) and the piece is
// larger than r / 8. Every returned value is an upper bound.
double Builder::piece_bound(const double* a, const double* b, const double* c) const {
  const TriangleGrid& grid = seeder_.triangles();
  double centroid[3], foot[3], d = 0.0;
  for (int k = 0; k < 3; ++k) centroid[k] = (a[k] + b[k] + c[k]) / 3.0;
  const int32_t t = grid.nearest(centroid, seeder_.reach(), foot, &d);
  if (t < 0) return HUGE_VAL;
  lower_ = std::max(lower_, d);
  double bound = 0.0;
  for (const double* x : {a, b, c})
    bound = std::max(bound, geometry::point_triangle_distance(x, s_.corner(t, 0), s_.corner(t, 1),
                                                              s_.corner(t, 2)));
  // Rounding of the computed distances (a few ulps of the coordinates).
  const double rounding = 0x1p-40 * (s_.diameter + h_);
  bound += rounding;
  const double size = std::sqrt(std::max({geometry::squared_distance(a, b),
                                          geometry::squared_distance(b, c),
                                          geometry::squared_distance(c, a)}));
  if (bound <= lower_ * (1.0 + 1.0 / 16.0) + 2.0 * rounding || size <= 0.125 * seeder_.spacing()) {
    return bound;
  }
  double ab[3], bc[3], ca[3];
  for (int k = 0; k < 3; ++k) {
    ab[k] = 0.5 * (a[k] + b[k]);
    bc[k] = 0.5 * (b[k] + c[k]);
    ca[k] = 0.5 * (c[k] + a[k]);
  }
  return std::max({piece_bound(a, ab, ca), piece_bound(ab, b, bc), piece_bound(ca, bc, c),
                   piece_bound(ab, bc, ca)});
}

void Builder::certify(const phx_mc_voronoi_cells& out) {
  // Closure: every directed edge of a cell meets its reverse exactly once.
  std::vector<std::array<int32_t, 3>> directed;
  const int64_t faces = static_cast<int64_t>(out.face_offsets.size()) - 1;
  for (int64_t f = 0; f < faces; ++f) {
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
    if (duplicate || range.second - range.first != 1) ++open_edges_;
  }
  // Local Delaunay at every face of an interior seed's star.
  violations_ = 0;
  const int64_t count = static_cast<int64_t>(tets_.size() / 4);
  for (int64_t t = 0; t < count; ++t) {
    if (!relevant(static_cast<int32_t>(t))) continue;
    for (int k = 0; k < 4; ++k) {
      const int32_t u = tet_neighbors_[4 * t + k];
      if (u < 0 || u < t) continue;
      if (insphere_of(static_cast<int32_t>(t), apex(static_cast<int32_t>(t), k)) > 0) ++violations_;
    }
  }
  // Distance bound of the boundary faces.
  bound_ = 0.0;
  lower_ = vertex_distance_;
  for (int64_t f = 0; f < faces; ++f) {
    if (out.face_cells[2 * f + 1] >= 0) continue;
    const int64_t begin = out.face_offsets[f], end = out.face_offsets[f + 1];
    const double* x0 = out.vertices.data() + 3 * static_cast<int64_t>(out.face_vertices[begin]);
    for (int64_t q = begin + 1; q + 1 < end; ++q) {
      const double* x1 = out.vertices.data() + 3 * static_cast<int64_t>(out.face_vertices[q]);
      const double* x2 = out.vertices.data() + 3 * static_cast<int64_t>(out.face_vertices[q + 1]);
      bound_ = std::max(bound_, piece_bound(x0, x1, x2));
    }
  }
  bound_ = std::max(bound_, lower_);
  certified_ = open_edges_ == 0 && violations_ == 0 && leaks_.empty();
}

int32_t build(const Input& in, phx_mc_voronoi_cells& out) {
  const auto start = Clock::now();
  Surface surface;
  build_surface(in, surface);
  const double h = in.size;
  auto clock = Clock::now();
  Seeder seeder(surface, h);
  SeedSet seeds;
  seeder.interior(seeds);
  if (seeds.interior == 0) return PHX_MC_UNRESOLVED;
  PointGrid ghost_grid(seeder.spacing());
  seeder.ghosts(seeds, ghost_grid);
  double timings[3] = {0, 0, 0};
  const double seeding = seconds_since(clock);
  double certification = 0.0;
  int32_t repairs = 0;
  std::vector<std::pair<int32_t, int32_t>> used;      // (seed, triangle) repairs
  // Nearly cospherical apex pairs and the stage of their next move.
  std::vector<std::pair<std::pair<int32_t, int32_t>, int>> resolved;
  while (true) {
    Builder builder(surface, seeder, seeds, h);
    out = phx_mc_voronoi_cells{};
    builder.run(out, timings);
    // Repair: reflect each leaking seed across the triangle nearest its
    // offending point.
    int32_t added = 0;
    for (const auto& [i, point] : builder.leaks()) {
      double foot[3], d = 0.0, g[3];
      const int32_t t = seeder.triangles().nearest(point.data(), seeder.reach(), foot, &d);
      if (t < 0) continue;
      const std::pair<int32_t, int32_t> key{i, t};
      if (std::find(used.begin(), used.end(), key) != used.end()) continue;
      used.push_back(key);
      double q[3];
      closest_point(seeds.points.data() + 3 * static_cast<int64_t>(i), surface.corner(t, 0),
                    surface.corner(t, 1), surface.corner(t, 2), q);
      if (seeder.reflect(seeds, i, t, q, g) && seeder.add_ghost(seeds, ghost_grid, i, t, g))
        ++added;
    }
    // Nearly cospherical apex pairs: one staged move per pair and round (each
    // pair passes every stage at most once, so the rebuilds terminate).
    if (added == 0) {
      std::vector<int32_t> moved;
      std::vector<std::pair<int32_t, int32_t>> round;  // pairs moved this round
      const auto taken = [&](int32_t s) {
        return std::find(moved.begin(), moved.end(), s) != moved.end();
      };
      for (const Perturbation& p : builder.near()) {
        const std::pair<int32_t, int32_t> key{p.a, p.b};
        if (std::find(round.begin(), round.end(), key) != round.end()) continue;
        const auto found = std::find_if(resolved.begin(), resolved.end(),
                                        [&](const auto& r) { return r.first == key; });
        const int stage = found == resolved.end() ? 0 : found->second;
        if (p.stage != stage) continue;
        if (taken(p.seed) || std::any_of(p.mirrors.begin(), p.mirrors.end(),
                                         [&](const auto& e) { return taken(e[0]); }))
          continue;
        const int next = stage + 1;
        if (found == resolved.end())
          resolved.emplace_back(key, next);
        else
          found->second = next;
        round.push_back(key);
        double* x = seeds.points.data() + 3 * static_cast<int64_t>(p.seed);
        for (int a = 0; a < 3; ++a) x[a] += p.shift[a];
        seeder.snap(x);
        moved.push_back(p.seed);
        for (const auto& [child, parent, t] : p.mirrors) {
          seeder.mirror(seeds.points.data() + 3 * static_cast<int64_t>(parent), t,
                        seeds.points.data() + 3 * static_cast<int64_t>(child));
          moved.push_back(child);
        }
        // A seed moved alone is no mirror any more: later rigid moves must not
        // reflect it back onto the symmetry it was moved off.
        if (p.lone)
          std::erase_if(seeds.mirrors,
                        [&](const auto& e) { return e[0] == p.seed || e[1] == p.seed; });
        ++added;
      }
    }
    if (added > 0) {
      ++repairs;
      continue;
    }
    clock = Clock::now();
    if (in.prove) builder.certify(out);
    certification = seconds_since(clock);
    // Signed volume error against the regions of the input, and its bound:
    // the error of each rounded fan volume is far below 2^-40 of its magnitude.
    std::vector<double> expected(surface.region_count, 0.0);
    double magnitude = builder.volume_magnitude();
    for (int64_t t = 0; t < in.triangle_count; ++t) {
      double a[3], b[3], c[3], bc[3];
      geometry::difference(surface.corner(t, 0), surface.lower, a);
      geometry::difference(surface.corner(t, 1), surface.lower, b);
      geometry::difference(surface.corner(t, 2), surface.lower, c);
      geometry::cross(b, c, bc);
      const double volume = geometry::dot(a, bc) / 6.0;
      if (surface.back(t) >= 0) expected[surface.back(t)] += volume;
      if (surface.front(t) >= 0) expected[surface.front(t)] -= volume;
      magnitude += std::fabs(volume);
    }
    double total = 0.0, achieved = 0.0;
    for (const double v : expected) total += v;
    for (const double v : out.cell_volumes) achieved += v;
    double* stats = out.stats.data();
    stats[0] = seeding;
    stats[1] = timings[0];
    stats[2] = timings[1];
    stats[3] = timings[2];
    stats[4] = certification;
    stats[5] = seconds_since(start);
    stats[6] = static_cast<double>(seeds.regions.size());
    stats[7] = static_cast<double>(seeds.interior);
    stats[8] = static_cast<double>(seeds.regions.size()) - seeds.interior;
    stats[9] = static_cast<double>(seeds.lattice);
    stats[10] = static_cast<double>(out.cell_seeds.size());
    stats[11] = static_cast<double>(builder.boundary_cells());
    stats[12] = static_cast<double>(builder.boundary_faces());
    stats[13] = static_cast<double>(out.face_facets.size());
    stats[14] = static_cast<double>(repairs);
    stats[15] = in.prove ? (builder.certified() ? 1.0 : 0.0) : -1.0;
    stats[16] = in.prove ? static_cast<double>(builder.open_edges()) : -1.0;
    stats[17] = in.prove ? static_cast<double>(builder.violations()) : -1.0;
    stats[18] = total != 0.0 ? achieved / total - 1.0 : 0.0;
    stats[19] = builder.vertex_distance() / h;
    stats[20] = in.prove ? builder.distance_bound() / h : -1.0;
    stats[21] = static_cast<double>(surface.feature_edges);
    stats[22] = h;
    stats[23] = seeder.spacing();
    stats[24] = builder.boundary_exact() ? 1.0 : 0.0;
    stats[25] = total != 0.0 ? (std::fabs(achieved - total) + 0x1p-40 * magnitude) / std::fabs(total)
                             : HUGE_VAL;
    if (!builder.leaks().empty()) return PHX_MC_UNRESOLVED;
    return !in.prove || builder.certified() ? PHX_MC_OK : PHX_MC_UNRESOLVED;
  }
}

}  // namespace
}  // namespace phx::mc::conforming_voronoi

namespace {

int32_t validate(const phx_mc_conforming_voronoi_request* in) {
  if (in == nullptr || in->result == nullptr || in->vertex_count < 4 || in->triangle_count < 4 ||
      in->vertices == nullptr || in->triangles == nullptr || in->triangle_regions == nullptr ||
      in->triangle_facets == nullptr || (in->prove != 0 && in->prove != 1) ||
      !(in->feature_angle_degrees > 0.0 && in->feature_angle_degrees < 180.0) ||
      !(in->size > 0.0 && std::isfinite(in->size))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (in->vertex_count >= INT32_MAX || in->triangle_count >= (int64_t{1} << 31) / 3 ||
      !phx::mc::addressable(in->vertex_count, 3, sizeof(double)) ||
      !phx::mc::addressable(in->triangle_count, 3, sizeof(int32_t))) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  for (int64_t k = 0; k < 3 * in->vertex_count; ++k) {
    if (!std::isfinite(in->vertices[k])) return PHX_MC_NONFINITE_INPUT;
    if (!phx::mc::coordinate_in_domain(in->vertices[k])) return PHX_MC_RANGE_ERROR;
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
        request->vertex_count,     request->vertices,        request->triangle_count,
        request->triangles,        request->triangle_regions, request->triangle_facets,
        request->size,
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
  sizes[0] = static_cast<int64_t>(result->vertex_triangles.size());
  sizes[1] = static_cast<int64_t>(result->face_facets.size());
  sizes[2] = static_cast<int64_t>(result->face_vertices.size());
  sizes[3] = static_cast<int64_t>(result->cell_seeds.size());
  sizes[4] = static_cast<int64_t>(result->seed_regions.size());
  sizes[5] = static_cast<int64_t>(result->exact_values.size());
  return PHX_MC_OK;
}

int32_t phx_mc_conforming_voronoi_export(const phx_mc_voronoi_cells* result,
                                         const phx_mc_conforming_voronoi_arrays* arrays) {
  if (result == nullptr || arrays == nullptr || arrays->vertices == nullptr ||
      arrays->vertex_triangles == nullptr || arrays->face_offsets == nullptr ||
      arrays->face_vertices == nullptr || arrays->face_cells == nullptr ||
      arrays->face_facets == nullptr || arrays->cell_seeds == nullptr ||
      arrays->cell_regions == nullptr || arrays->cell_volumes == nullptr ||
      arrays->cell_centroids == nullptr || arrays->seeds == nullptr ||
      arrays->seed_regions == nullptr || arrays->stats == nullptr ||
      arrays->exact_offsets == nullptr ||
      (!result->exact_values.empty() && arrays->exact_values == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  std::copy(result->vertices.begin(), result->vertices.end(), arrays->vertices);
  std::copy(result->vertex_triangles.begin(), result->vertex_triangles.end(), arrays->vertex_triangles);
  std::copy(result->face_offsets.begin(), result->face_offsets.end(), arrays->face_offsets);
  std::copy(result->face_vertices.begin(), result->face_vertices.end(), arrays->face_vertices);
  std::copy(result->face_cells.begin(), result->face_cells.end(), arrays->face_cells);
  std::copy(result->face_facets.begin(), result->face_facets.end(), arrays->face_facets);
  std::copy(result->cell_seeds.begin(), result->cell_seeds.end(), arrays->cell_seeds);
  std::copy(result->cell_regions.begin(), result->cell_regions.end(), arrays->cell_regions);
  std::copy(result->cell_volumes.begin(), result->cell_volumes.end(), arrays->cell_volumes);
  std::copy(result->cell_centroids.begin(), result->cell_centroids.end(), arrays->cell_centroids);
  std::copy(result->seeds.begin(), result->seeds.end(), arrays->seeds);
  std::copy(result->seed_regions.begin(), result->seed_regions.end(), arrays->seed_regions);
  std::copy(result->stats.begin(), result->stats.end(), arrays->stats);
  std::copy(result->exact_offsets.begin(), result->exact_offsets.end(), arrays->exact_offsets);
  std::copy(result->exact_values.begin(), result->exact_values.end(), arrays->exact_values);
  return PHX_MC_OK;
}

void phx_mc_conforming_voronoi_free(phx_mc_voronoi_cells* result) { delete result; }

}  // extern "C"
