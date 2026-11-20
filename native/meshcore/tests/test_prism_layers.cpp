//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Prism layers against closed-form references: on a sphere every level lies at
// its height by true distance; on a box (ridges and corners) every column top
// lies at the total height from each face it leaves, exactly as the inward
// offset of the box; a slab thinner than two stacks is thinned until its top
// level is embedded; corner Jacobians are recomputed independently; invalid
// walls are refused.
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

struct Surface {
  std::vector<double> x;
  std::vector<int32_t> f;
};

// Icosphere of radius 1 with `levels` midpoint subdivisions, outward.
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
    for (size_t i = 0; i < s.f.size(); i += 3) {
      const int32_t a = s.f[i], b = s.f[i + 1], c = s.f[i + 2];
      const int32_t ab = middle(a, b), bc = middle(b, c), ca = middle(c, a);
      next.insert(next.end(), {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca});
    }
    s.f = next;
  }
  return s;
}

// Box [0, lx] x [0, ly] x [0, lz] with `n` x `n` quads per face, outward.
Surface box(double lx, double ly, double lz, int n) {
  Surface s;
  std::map<std::array<int, 3>, int32_t> ids;
  const double size[3] = {lx, ly, lz};
  auto vertex = [&](std::array<int, 3> g) {
    const auto it = ids.find(g);
    if (it != ids.end()) return it->second;
    const int32_t id = static_cast<int32_t>(s.x.size() / 3);
    for (int a = 0; a < 3; ++a) s.x.push_back(size[a] * g[a] / n);
    ids.emplace(g, id);
    return id;
  };
  for (int axis = 0; axis < 3; ++axis) {
    const int u = (axis + 1) % 3, v = (axis + 2) % 3;
    for (int side = 0; side <= n; side += n) {
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          std::array<int32_t, 4> q;
          const int di[4] = {0, 1, 1, 0}, dj[4] = {0, 0, 1, 1};
          for (int c = 0; c < 4; ++c) {
            std::array<int, 3> g{};
            g[axis] = side;
            g[u] = i + di[c];
            g[v] = j + dj[c];
            q[c] = vertex(g);
          }
          // (u, v, axis) is right-handed: counterclockwise in (u, v) points +axis.
          if (side == n) {
            s.f.insert(s.f.end(), {q[0], q[1], q[2], q[0], q[2], q[3]});
          } else {
            s.f.insert(s.f.end(), {q[0], q[2], q[1], q[0], q[3], q[2]});
          }
        }
      }
    }
  }
  return s;
}

void flip(Surface& s) {
  for (size_t i = 0; i < s.f.size(); i += 3) std::swap(s.f[i + 1], s.f[i + 2]);
}

struct Layers {
  int32_t status = -1;
  std::vector<double> points, displacement, scale, stats;
  std::vector<int32_t> prisms;
  std::vector<uint8_t> kind;
};

Layers grow(const Surface& s, const std::vector<double>& heights, int prove,
            double minimum_scale = 0.05) {
  Layers r;
  const int64_t n = static_cast<int64_t>(s.x.size() / 3), m = static_cast<int64_t>(s.f.size() / 3);
  const int32_t layers = static_cast<int32_t>(heights.size());
  r.points.assign(static_cast<size_t>(3 * n * (layers + 1)), 0.0);
  r.prisms.assign(static_cast<size_t>(6 * m * layers), -1);
  r.displacement.assign(static_cast<size_t>(3 * n), 0.0);
  r.scale.assign(static_cast<size_t>(n), 0.0);
  r.kind.assign(static_cast<size_t>(n), 9);
  r.stats.assign(21, 0.0);
  const phx_mc_prism_layers_request request{n,       s.x.data(), m,  s.f.data(), layers,
                                            heights.data(), 30.0,       minimum_scale, prove};
  const phx_mc_prism_layers_result result{r.points.data(), r.prisms.data(), r.displacement.data(),
                                          r.scale.data(),  r.kind.data(),   r.stats.data()};
  r.status = phx_mc_prism_layers(&request, &result);
  return r;
}

// Corner Jacobians recomputed from the output rows (long double).
long double lowest_corner(const Layers& r) {
  long double lowest = 1.0L;
  for (size_t p = 0; p < r.prisms.size(); p += 6) {
    const double* q[6];
    for (int c = 0; c < 6; ++c) q[c] = &r.points[3 * static_cast<size_t>(r.prisms[p + c])];
    for (int c = 0; c < 3; ++c) {
      const int a = c, b = (c + 1) % 3, d = (c + 2) % 3;
      for (int top = 0; top < 2; ++top) {
        const int o = 3 * top;
        long double e1[3], e2[3], up[3];
        for (int k = 0; k < 3; ++k) {
          e1[k] = static_cast<long double>(q[b + o][k]) - q[a + o][k];
          e2[k] = static_cast<long double>(q[d + o][k]) - q[a + o][k];
          up[k] = static_cast<long double>(q[a + 3][k]) - q[a][k];
        }
        const long double det = up[0] * (e1[1] * e2[2] - e1[2] * e2[1]) -
                                up[1] * (e1[0] * e2[2] - e1[2] * e2[0]) +
                                up[2] * (e1[0] * e2[1] - e1[1] * e2[0]);
        if (det < lowest) lowest = det;
      }
    }
  }
  return lowest;
}

const std::vector<double> kHeights = {0.01, 0.022, 0.0364, 0.05368, 0.074416};

void sphere_levels_at_true_distance() {
  Surface s = icosphere(3);
  flip(s);  // grow inward
  const Layers r = grow(s, kHeights, 1);
  PHX_CHECK(r.status == PHX_MC_OK);
  PHX_CHECK(r.stats[16] == 1.0);   // certified
  PHX_CHECK(r.stats[14] == 0.0);   // invalid prisms
  PHX_CHECK(r.stats[15] == 0.0);   // top contacts
  PHX_CHECK(r.stats[7] == 0.0 && r.stats[8] == 0.0);
  PHX_CHECK(r.stats[10] == 0.0);   // nothing thinned
  PHX_CHECK(lowest_corner(r) > 0.0L);
  const size_t n = s.x.size() / 3;
  double worst = 0.0;
  for (size_t k = 0; k <= kHeights.size(); ++k) {
    const double h = k == 0 ? 0.0 : kHeights[k - 1];
    for (size_t v = 0; v < n; ++v) {
      const double* p = &r.points[3 * (k * n + v)];
      worst = std::max(worst, std::fabs(1.0 - std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - h));
    }
  }
  PHX_CHECK(worst < 1e-3 * kHeights.back());
  for (size_t i = 0; i < 3 * n; ++i) PHX_CHECK(r.points[i] == s.x[i]);  // level 0 bitwise
  // prove = false constructs the same levels without the certificate.
  const Layers u = grow(s, kHeights, 0);
  PHX_CHECK(u.status == PHX_MC_OK);
  PHX_CHECK(u.stats[16] == 0.0 && u.stats[15] == -1.0);
  PHX_CHECK(u.points == r.points);
}

void box_tops_at_offset() {
  Surface s = box(1.0, 0.8, 0.6, 4);
  flip(s);
  const Layers r = grow(s, kHeights, 1);
  PHX_CHECK(r.status == PHX_MC_OK);
  PHX_CHECK(r.stats[16] == 1.0);
  PHX_CHECK(r.stats[10] == 0.0);
  PHX_CHECK(r.stats[8] == 8.0);                    // corners
  PHX_CHECK(r.stats[7] == 12.0 * 3.0);             // ridge vertices (3 interior per edge)
  PHX_CHECK(lowest_corner(r) > 0.0L);
  // The inward offset of a box is the inner box: every top is at distance D
  // from each face plane it leaves and at least D from the others.
  const size_t n = s.x.size() / 3, top = kHeights.size() * n;
  const double d = kHeights.back(), size[3] = {1.0, 0.8, 0.6};
  double worst = 0.0;
  for (size_t v = 0; v < n; ++v) {
    const double* p = &s.x[3 * v];
    const double* q = &r.points[3 * (top + v)];
    for (int a = 0; a < 3; ++a) {
      const bool on_low = p[a] == 0.0, on_high = p[a] == size[a];
      if (on_low) worst = std::max(worst, std::fabs(q[a] - d));
      if (on_high) worst = std::max(worst, std::fabs(size[a] - q[a] - d));
      if (!on_low && !on_high) PHX_CHECK(q[a] >= d - 1e-12 && size[a] - q[a] >= d - 1e-12);
    }
  }
  PHX_CHECK(worst < 1e-12);
}

void slab_is_thinned_until_embedded() {
  // Two stacks of 0.0744 do not fit in 0.1: the top level would cross itself.
  Surface s = box(1.0, 1.0, 0.1, 6);
  flip(s);
  const Layers r = grow(s, kHeights, 1);
  PHX_CHECK(r.status == PHX_MC_OK);
  PHX_CHECK(r.stats[16] == 1.0);
  PHX_CHECK(r.stats[10] > 0.0);       // thinned columns
  PHX_CHECK(r.stats[12] < 0.68);      // 2 D s < 0.1 needs s < 0.672
  PHX_CHECK(lowest_corner(r) > 0.0L);
  // With no room to thin, the failure is reported.
  const Layers stuck = grow(s, kHeights, 1, 1.0);
  PHX_CHECK(stuck.status == PHX_MC_UNRESOLVED);
  PHX_CHECK(stuck.stats[16] == 0.0 && (stuck.stats[14] > 0.0 || stuck.stats[15] > 0.0));
}

void invalid_walls_are_refused() {
  Surface s = icosphere(1);
  Surface open = s;
  open.f.resize(open.f.size() - 3);
  PHX_CHECK(grow(open, kHeights, 1).status == PHX_MC_INVALID_INPUT);
  Surface twisted = s;
  std::swap(twisted.f[1], twisted.f[2]);
  PHX_CHECK(grow(twisted, kHeights, 1).status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(grow(s, {0.02, 0.01}, 1).status == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(grow(s, kHeights, 1, 0.0).status == PHX_MC_INVALID_ARGUMENT);
}

}  // namespace

int main() {
  sphere_levels_at_true_distance();
  box_tops_at_offset();
  slab_is_thinned_until_embedded();
  invalid_walls_are_refused();
  return phx::mc::test::finish("test_prism_layers");
}
