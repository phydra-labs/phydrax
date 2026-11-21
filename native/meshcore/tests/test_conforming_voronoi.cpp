//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Conforming Voronoi cells against independent references: two seeds in a
// cube split it exactly at their bisector; random seeds in a cube give the
// volumes of a brute-force clip of the cube by every bisector half-space;
// automatic seeds on a convex icosphere, a notched (reflex) block and two
// stacked material cubes give closed cells whose volumes sum to each region,
// exact convexity verdicts and identical output on a second run.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <utility>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

struct Surface {
  std::vector<double> x;
  std::vector<int32_t> f;
  std::vector<int32_t> regions;  // (front, back) per triangle
  std::vector<int32_t> facets;
};

void add_box(Surface& s, const double lo[3], const double hi[3], int32_t front, int32_t back) {
  const int32_t base = static_cast<int32_t>(s.x.size() / 3);
  for (int c = 0; c < 8; ++c) {
    s.x.push_back(c & 1 ? hi[0] : lo[0]);
    s.x.push_back(c & 2 ? hi[1] : lo[1]);
    s.x.push_back(c & 4 ? hi[2] : lo[2]);
  }
  // Outward quads (counterclockwise seen from outside), two triangles each.
  const int quads[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4},
                           {2, 6, 7, 3}, {0, 4, 6, 2}, {1, 3, 7, 5}};
  for (int q = 0; q < 6; ++q) {
    const int32_t facet = static_cast<int32_t>(s.facets.size() / 2);
    for (const auto& tri : {std::array<int, 3>{0, 1, 2}, std::array<int, 3>{0, 2, 3}}) {
      for (int k : tri) s.f.push_back(base + quads[q][k]);
      s.regions.push_back(front);
      s.regions.push_back(back);
      s.facets.push_back(facet);
    }
  }
}

struct Cells {
  int32_t status = -1;
  std::vector<double> vertices, cell_volumes, cell_centroids, seeds, stats, exact_values;
  std::vector<int8_t> kinds;
  std::vector<int32_t> sources, face_vertices, face_cells, face_facets, cell_seeds,
      cell_regions;
  std::vector<int64_t> face_offsets, exact_offsets;
  std::vector<uint8_t> convex;
};

Cells run(const Surface& s, const std::vector<double>& seeds, double size, int32_t prove) {
  Cells out;
  phx_mc_voronoi_cells* handle = nullptr;
  const phx_mc_conforming_voronoi_request request{
      static_cast<int64_t>(s.x.size() / 3), s.x.data(), static_cast<int64_t>(s.f.size() / 3),
      s.f.data(), s.regions.data(), s.facets.data(), static_cast<int64_t>(seeds.size() / 3),
      seeds.empty() ? nullptr : seeds.data(), size, 30.0, prove, &handle};
  out.status = phx_mc_conforming_voronoi(&request);
  if (handle == nullptr) return out;
  int64_t sizes[PHX_MC_CONFORMING_VORONOI_SIZES];
  phx_mc_conforming_voronoi_sizes(handle, sizes);
  out.vertices.resize(3 * sizes[0]);
  out.kinds.resize(sizes[0]);
  out.sources.resize(2 * sizes[0]);
  out.face_offsets.resize(sizes[1] + 1);
  out.face_vertices.resize(sizes[2]);
  out.face_cells.resize(2 * sizes[1]);
  out.face_facets.resize(sizes[1]);
  out.cell_seeds.resize(sizes[3]);
  out.cell_regions.resize(sizes[3]);
  out.cell_volumes.resize(sizes[3]);
  out.cell_centroids.resize(3 * sizes[3]);
  out.convex.resize(sizes[3]);
  out.seeds.resize(3 * sizes[4]);
  out.stats.resize(PHX_MC_CONFORMING_VORONOI_STATS);
  out.exact_offsets.resize(4 * sizes[0] + 1);
  out.exact_values.resize(sizes[5]);
  const phx_mc_conforming_voronoi_arrays arrays{
      out.vertices.data(),     out.kinds.data(),          out.sources.data(),
      out.face_offsets.data(), out.face_vertices.data(),  out.face_cells.data(),
      out.face_facets.data(),  out.cell_seeds.data(),     out.cell_regions.data(),
      out.cell_volumes.data(), out.cell_centroids.data(), out.convex.data(),
      out.seeds.data(),        out.stats.data(),          out.exact_offsets.data(),
      out.exact_values.data()};
  PHX_CHECK(phx_mc_conforming_voronoi_export(handle, &arrays) == PHX_MC_OK);
  phx_mc_conforming_voronoi_free(handle);
  return out;
}

double region_volume(const Surface& s, int32_t region) {
  double total = 0.0;
  for (std::size_t t = 0; t < s.f.size() / 3; ++t) {
    const double* a = &s.x[3 * s.f[3 * t]];
    const double* b = &s.x[3 * s.f[3 * t + 1]];
    const double* c = &s.x[3 * s.f[3 * t + 2]];
    const double v = (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
                      a[2] * (b[0] * c[1] - b[1] * c[0])) /
                     6.0;
    if (s.regions[2 * t + 1] == region) total += v;
    if (s.regions[2 * t] == region) total -= v;
  }
  return total;
}

// Every cell closed: each directed edge of its faces met once by its reverse.
bool closed(const Cells& c) {
  std::map<std::array<int32_t, 3>, int> edges;
  for (std::size_t f = 0; f + 1 < c.face_offsets.size(); ++f) {
    const int64_t b = c.face_offsets[f], e = c.face_offsets[f + 1];
    for (int side = 0; side < 2; ++side) {
      const int32_t cell = c.face_cells[2 * f + side];
      if (cell < 0) continue;
      for (int64_t q = b; q < e; ++q) {
        int32_t u = c.face_vertices[q], v = c.face_vertices[q + 1 < e ? q + 1 : b];
        if (side == 1) std::swap(u, v);
        ++edges[{cell, u, v}];
      }
    }
  }
  for (const auto& [key, count] : edges) {
    const auto reverse = edges.find({key[0], key[2], key[1]});
    if (count != 1 || reverse == edges.end() || reverse->second != 1) return false;
  }
  return true;
}

// Brute-force reference: clip a convex polytope (faces as loops) by a plane
// n . x <= d, keeping the inner side.
using Polygon = std::vector<std::array<double, 3>>;
std::vector<Polygon> clip(const std::vector<Polygon>& faces, const double* n, double d) {
  std::vector<Polygon> result;
  std::vector<std::array<double, 3>> cap;
  for (const Polygon& face : faces) {
    Polygon kept;
    for (std::size_t k = 0; k < face.size(); ++k) {
      const auto& a = face[k];
      const auto& b = face[(k + 1) % face.size()];
      const double sa = n[0] * a[0] + n[1] * a[1] + n[2] * a[2] - d;
      const double sb = n[0] * b[0] + n[1] * b[1] + n[2] * b[2] - d;
      if (sa <= 0.0) kept.push_back(a);
      if ((sa < 0.0) != (sb < 0.0) && sa != sb) {
        const double t = sa / (sa - sb);
        const std::array<double, 3> p{a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]),
                                      a[2] + t * (b[2] - a[2])};
        kept.push_back(p);
        cap.push_back(p);
      }
    }
    if (kept.size() >= 3) result.push_back(kept);
  }
  if (cap.size() >= 3) {
    std::array<double, 3> c{0, 0, 0};
    for (const auto& p : cap)
      for (int k = 0; k < 3; ++k) c[k] += p[k] / static_cast<double>(cap.size());
    const double u0[3] = {std::fabs(n[0]) < 0.9 ? 1.0 : 0.0, std::fabs(n[0]) < 0.9 ? 0.0 : 1.0,
                          0.0};
    double u[3] = {n[1] * u0[2] - n[2] * u0[1], n[2] * u0[0] - n[0] * u0[2],
                   n[0] * u0[1] - n[1] * u0[0]};
    double w[3] = {n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2],
                   n[0] * u[1] - n[1] * u[0]};
    std::sort(cap.begin(), cap.end(), [&](const auto& p, const auto& q) {
      const double pa = std::atan2((p[0] - c[0]) * w[0] + (p[1] - c[1]) * w[1] + (p[2] - c[2]) * w[2],
                                   (p[0] - c[0]) * u[0] + (p[1] - c[1]) * u[1] + (p[2] - c[2]) * u[2]);
      const double qa = std::atan2((q[0] - c[0]) * w[0] + (q[1] - c[1]) * w[1] + (q[2] - c[2]) * w[2],
                                   (q[0] - c[0]) * u[0] + (q[1] - c[1]) * u[1] + (q[2] - c[2]) * u[2]);
      return pa < qa;
    });
    result.push_back(cap);
  }
  return result;
}

double polytope_volume(const std::vector<Polygon>& faces) {
  // Divergence theorem with an interior reference point; orientation free.
  std::array<double, 3> o{0, 0, 0};
  int count = 0;
  for (const auto& f : faces)
    for (const auto& p : f) {
      for (int k = 0; k < 3; ++k) o[k] += p[k];
      ++count;
    }
  for (int k = 0; k < 3; ++k) o[k] /= count;
  double total = 0.0;
  for (const auto& f : faces)
    for (std::size_t k = 1; k + 1 < f.size(); ++k) {
      double a[3], b[3], c[3];
      for (int x = 0; x < 3; ++x) {
        a[x] = f[0][x] - o[x];
        b[x] = f[k][x] - o[x];
        c[x] = f[k + 1][x] - o[x];
      }
      total += std::fabs(a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
                         a[2] * (b[0] * c[1] - b[1] * c[0])) /
               6.0;
    }
  return total;
}

void test_two_seeds_split_the_cube_exactly() {
  Surface s;
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  add_box(s, lo, hi, -1, 0);
  const std::vector<double> seeds = {0.25, 0.5, 0.5, 0.75, 0.5, 0.5};
  const Cells c = run(s, seeds, 0.0, 1);
  PHX_CHECK(c.status == PHX_MC_OK);
  PHX_CHECK(c.cell_seeds.size() == 2);
  PHX_CHECK(c.stats[15] == 1.0);
  for (const double v : c.cell_volumes) PHX_CHECK(v == 0.5);
  // Every surface-edge point lies exactly on the bisector x = 0.5: the four
  // cube-edge midpoints plus crossings of the face diagonals.
  int midpoints = 0;
  for (std::size_t v = 0; v < c.kinds.size(); ++v) {
    if (c.kinds[v] != 2) continue;
    ++midpoints;
    PHX_CHECK(c.vertices[3 * v] == 0.5);
  }
  PHX_CHECK(midpoints >= 4);
  // One bisector face and five merged cube faces per cell.
  PHX_CHECK(c.face_facets.size() == 11);
  PHX_CHECK(closed(c));
  for (const uint8_t convex : c.convex) PHX_CHECK(convex == 1);
}

void test_random_seeds_match_brute_force_clipping() {
  Surface s;
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  add_box(s, lo, hi, -1, 0);
  std::mt19937_64 random(20261120);
  std::uniform_real_distribution<double> unit(0.05, 0.95);
  std::vector<double> seeds(3 * 40);
  for (double& x : seeds) x = unit(random);
  const Cells c = run(s, seeds, 0.0, 1);
  PHX_CHECK(c.status == PHX_MC_OK);
  PHX_CHECK(c.cell_seeds.size() == 40);
  PHX_CHECK(closed(c));
  double total = 0.0;
  for (std::size_t cell = 0; cell < c.cell_seeds.size(); ++cell) {
    const int32_t i = c.cell_seeds[cell];
    std::vector<Polygon> poly = {
        {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}}, {{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}},
        {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}}, {{0, 1, 0}, {0, 1, 1}, {1, 1, 1}, {1, 1, 0}},
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}}, {{1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {1, 0, 1}}};
    for (int j = 0; j < 40; ++j) {
      if (j == i) continue;
      double n[3], d = 0.0;
      for (int k = 0; k < 3; ++k) {
        n[k] = seeds[3 * j + k] - seeds[3 * i + k];
        d += 0.5 * n[k] * (seeds[3 * j + k] + seeds[3 * i + k]);
      }
      poly = clip(poly, n, d);
    }
    const double reference = polytope_volume(poly);
    PHX_CHECK_NEAR(c.cell_volumes[cell], reference, 1e-12);
    PHX_CHECK(c.convex[cell] == 1);
    total += c.cell_volumes[cell];
  }
  PHX_CHECK_NEAR(total, 1.0, 1e-12);
}

Surface icosphere(int levels) {
  const double t = (1.0 + std::sqrt(5.0)) / 2.0;
  Surface s;
  const double base[12][3] = {{-1, t, 0}, {1, t, 0},  {-1, -t, 0}, {1, -t, 0},
                              {0, -1, t}, {0, 1, t},  {0, -1, -t}, {0, 1, -t},
                              {t, 0, -1}, {t, 0, 1},  {-t, 0, -1}, {-t, 0, 1}};
  for (const auto& p : base) {
    const double l = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    s.x.insert(s.x.end(), {p[0] / l, p[1] / l, p[2] / l});
  }
  s.f = {0, 11, 5, 0, 5,  1, 0, 1, 7, 0, 7,  10, 0, 10, 11, 1, 5, 9, 5, 11,
         4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3, 9,  4,  3,  4,  2, 3, 2, 6, 3,
         6, 8,  3,  8, 9,  4, 9, 5, 2, 4, 11, 6, 2,  10, 8,  6, 7, 9, 8, 1};
  for (int level = 0; level < levels; ++level) {
    std::map<std::pair<int32_t, int32_t>, int32_t> mid;
    auto middle = [&](int32_t a, int32_t b) {
      const auto key = std::make_pair(std::min(a, b), std::max(a, b));
      const auto it = mid.find(key);
      if (it != mid.end()) return it->second;
      double p[3];
      for (int k = 0; k < 3; ++k) p[k] = s.x[3 * a + k] + s.x[3 * b + k];
      const double l = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
      const int32_t id = static_cast<int32_t>(s.x.size() / 3);
      s.x.insert(s.x.end(), {p[0] / l, p[1] / l, p[2] / l});
      mid.emplace(key, id);
      return id;
    };
    std::vector<int32_t> next;
    for (std::size_t q = 0; q < s.f.size(); q += 3) {
      const int32_t a = s.f[q], b = s.f[q + 1], c = s.f[q + 2];
      const int32_t ab = middle(a, b), bc = middle(b, c), ca = middle(c, a);
      next.insert(next.end(), {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca});
    }
    s.f = next;
  }
  const std::size_t m = s.f.size() / 3;
  s.regions.assign(2 * m, 0);
  for (std::size_t t2 = 0; t2 < m; ++t2) s.regions[2 * t2] = -1;
  s.facets.assign(m, 0);
  return s;
}

void test_automatic_sphere_cells_are_closed_convex_and_deterministic() {
  const Surface s = icosphere(3);
  const Cells c = run(s, {}, 0.3, 1);
  PHX_CHECK(c.status == PHX_MC_OK);
  PHX_CHECK(c.stats[15] == 1.0);
  PHX_CHECK(c.cell_seeds.size() > 100);
  PHX_CHECK(closed(c));
  double total = 0.0;
  for (const double v : c.cell_volumes) {
    PHX_CHECK(v > 0.0);
    total += v;
  }
  PHX_CHECK_NEAR(total, region_volume(s, 0), 1e-12);
  // The icosphere is convex: no reflex edge, hence every cell convex.
  for (const uint8_t convex : c.convex) PHX_CHECK(convex == 1);
  const Cells again = run(s, {}, 0.3, 1);
  PHX_CHECK(again.vertices == c.vertices);
  PHX_CHECK(again.face_vertices == c.face_vertices);
  PHX_CHECK(again.face_cells == c.face_cells);
  PHX_CHECK(again.seeds == c.seeds);
}

void test_stacked_regions_share_their_interface() {
  Surface t;
  // Stacked regions: [0,1]^2 x [0,1] (region 0) and [0,1]^2 x [1,2] (region 1).
  const double base[12][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
                              {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1},
                              {0, 0, 2}, {1, 0, 2}, {0, 1, 2}, {1, 1, 2}};
  for (const auto& p : base) t.x.insert(t.x.end(), {p[0], p[1], p[2]});
  auto quad = [&](int p, int q, int r, int w, int32_t front, int32_t back, int32_t facet) {
    for (const auto& tri : {std::array<int, 3>{p, q, r}, std::array<int, 3>{p, r, w}}) {
      for (int k : tri) t.f.push_back(k);
      t.regions.push_back(front);
      t.regions.push_back(back);
      t.facets.push_back(facet);
    }
  };
  quad(0, 2, 3, 1, -1, 0, 0);   // z = 0 bottom
  quad(4, 5, 7, 6, 1, 0, 1);    // z = 1 interface, normal +z: region 1 in front
  quad(8, 9, 11, 10, -1, 1, 2); // z = 2 top
  quad(0, 1, 5, 4, -1, 0, 3);
  quad(2, 6, 7, 3, -1, 0, 4);
  quad(0, 4, 6, 2, -1, 0, 5);
  quad(1, 3, 7, 5, -1, 0, 6);
  quad(4, 5, 9, 8, -1, 1, 7);
  quad(6, 10, 11, 7, -1, 1, 8);
  quad(4, 8, 10, 6, -1, 1, 9);
  quad(5, 7, 11, 9, -1, 1, 10);
  const Cells c = run(t, {}, 0.25, 1);
  PHX_CHECK(c.status == PHX_MC_OK);
  PHX_CHECK(c.stats[15] == 1.0);
  PHX_CHECK(closed(c));
  double volume[2] = {0, 0};
  for (std::size_t cell = 0; cell < c.cell_regions.size(); ++cell)
    volume[c.cell_regions[cell]] += c.cell_volumes[cell];
  PHX_CHECK_NEAR(volume[0], 1.0, 1e-12);
  PHX_CHECK_NEAR(volume[1], 1.0, 1e-12);
  int interface = 0;
  for (std::size_t f = 0; f < c.face_facets.size(); ++f) {
    if (c.face_facets[f] != 1) continue;
    ++interface;
    PHX_CHECK(c.face_cells[2 * f + 1] >= 0);
    PHX_CHECK(c.cell_regions[c.face_cells[2 * f]] == 0);
    PHX_CHECK(c.cell_regions[c.face_cells[2 * f + 1]] == 1);
  }
  PHX_CHECK(interface > 0);
}

void test_reflex_edge_cells_are_nonconvex() {
  // A notched block: the union of [0,2]x[0,2]x[0,1] minus [1,2]x[1,2]x[0,1].
  Surface s;
  const double xy[6][2] = {{0, 0}, {2, 0}, {2, 1}, {1, 1}, {1, 2}, {0, 2}};
  for (int z = 0; z < 2; ++z)
    for (const auto& p : xy) s.x.insert(s.x.end(), {p[0], p[1], static_cast<double>(z)});
  // Bottom and top as fans about the reflex corner 3 (the L is star-shaped from it).
  const int fan[4][2] = {{4, 5}, {5, 0}, {0, 1}, {1, 2}};
  for (const auto& e : fan) {
    s.f.insert(s.f.end(), {3, e[1], e[0]});  // bottom, normal -z
    s.f.insert(s.f.end(), {9, 6 + e[0], 6 + e[1]});  // top, normal +z
    for (int k = 0; k < 2; ++k) {
      s.regions.insert(s.regions.end(), {-1, 0});
      s.facets.push_back(k);
    }
  }
  for (int k = 0; k < 6; ++k) {
    const int a = k, b = (k + 1) % 6;
    s.f.insert(s.f.end(), {a, b, 6 + b, a, 6 + b, 6 + a});
    for (int r = 0; r < 2; ++r) {
      s.regions.insert(s.regions.end(), {-1, 0});
      s.facets.push_back(2 + k);
    }
  }
  const Cells c = run(s, {}, 0.3, 1);
  PHX_CHECK(c.status == PHX_MC_OK);
  PHX_CHECK(c.stats[15] == 1.0);
  PHX_CHECK(closed(c));
  double total = 0.0;
  int nonconvex = 0;
  for (std::size_t cell = 0; cell < c.cell_volumes.size(); ++cell) {
    total += c.cell_volumes[cell];
    nonconvex += c.convex[cell] == 0;
  }
  PHX_CHECK_NEAR(total, 3.0, 1e-12);
  PHX_CHECK(nonconvex > 0);
  PHX_CHECK(c.stats[19] == static_cast<double>(nonconvex));
}

void test_invalid_requests_are_refused() {
  Surface s;
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  add_box(s, lo, hi, -1, 0);
  Surface same = s;
  same.regions[0] = same.regions[1];
  PHX_CHECK(run(same, {}, 0.2, 0).status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(run(s, {}, -1.0, 0).status == PHX_MC_INVALID_ARGUMENT);
  Surface flat = s;
  flat.f[2] = flat.f[1];
  PHX_CHECK(run(flat, {}, 0.2, 0).status == PHX_MC_DEGENERATE_INPUT);
  const std::vector<double> duplicate = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
  PHX_CHECK(run(s, duplicate, 0.0, 0).status == PHX_MC_DEGENERATE_INPUT);
}

}  // namespace

int main() {
  test_two_seeds_split_the_cube_exactly();
  test_random_seeds_match_brute_force_clipping();
  test_automatic_sphere_cells_are_closed_convex_and_deterministic();
  test_stacked_regions_share_their_interface();
  test_reflex_edge_cells_are_nonconvex();
  test_invalid_requests_are_refused();
  return phx::mc::test::finish("test_conforming_voronoi");
}
