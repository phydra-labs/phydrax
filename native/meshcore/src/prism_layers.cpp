//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Prism layers from a closed oriented triangle wall (prism_layers.hpp).
//
// Construction (floating point):
//  1. Topology: opposite half-edges, the ordered face fan of every vertex
//     (closed, consistently oriented 2-manifold required); feature edges are
//     those whose face normals differ by more than the feature angle. The
//     fan of a vertex splits at its feature edges into smooth sectors (one
//     sector unless two or more feature edges meet there).
//  2. Jets: per vertex and sector, a weighted least-squares quadratic height
//     fit over the sector's faces and their smooth neighbors gives the
//     surface normal at the vertex and its second fundamental form, so a
//     sector seen from one side of a ridge is not biased by its one-sided
//     fan.
//  3. Directions: a smooth vertex grows along its normal by the total height
//     D. A ridge or corner vertex grows by the minimum-norm w with
//     w . g_i = a_i for every sector normal g_i (the medial direction of its
//     sectors), where a_i corrects D for the sector's curvature along the
//     tangential part of w, so the column top lies at distance D from every
//     sector (the inward offset of a convex ridge is exact).
//  4. Relaxation: near features the tangential offsets of ridge columns are
//     carried into the neighboring smooth columns by a screened Laplacian
//     (decay length four times the largest ridge offset), solved by a
//     Gauss-Seidel worklist to 1e-4 D; this keeps the column tops from
//     squeezing the triangles beside a ridge. Each relaxed column's normal
//     height is corrected for curvature along its tangential offset.
//  5. Levels: level k of vertex v is p + (h_k / D) scale_v d_v; level 0 is the
//     wall bitwise. Prisms with a nonpositive float Bernstein Jacobian control
//     have their columns thinned (scale x 0.8) until they have none.
// Certification (prove): every prism valid by the exact signs of its nine
// Bernstein Jacobian controls (the six corner Jacobians by orient3d, the
// three mixed controls by a filtered expansion), and the top level embedded
// (exact triangle contacts: neighbors meet only in their shared vertices or
// edge) and disjoint from the wall; a failure thins the columns involved and
// the certificate is rerun. With a positive Jacobian on every prism and an
// embedded boundary (wall and top), the layer map is injective.
#include "prism_layers.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <utility>
#include <vector>

#include "intersections.hpp"
#include "phydrax_meshcore.h"
#include "expansion.hpp"
#include "predicates.hpp"

namespace phx::mc::prism_layers {
namespace {

using Vec = std::array<double, 3>;
using Clock = std::chrono::steady_clock;

constexpr double kThinning = 0.8;
constexpr double kRelaxationTolerance = 1e-4;  // of the total height
constexpr double kDecayFactor = 4.0;           // decay length / largest ridge offset

inline Vec sub(const double* a, const double* b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline Vec sub(const Vec& a, const Vec& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline Vec add(const Vec& a, const Vec& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline Vec mul(const Vec& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
inline double dot(const Vec& a, const Vec& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline Vec cross(const Vec& a, const Vec& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline double norm(const Vec& a) { return std::sqrt(dot(a, a)); }
inline Vec unit(const Vec& a) {
  const double l = norm(a);
  return l > 0.0 ? mul(a, 1.0 / l) : Vec{0.0, 0.0, 0.0};
}
inline Vec tangential(const Vec& a, const Vec& g) { return sub(a, mul(g, dot(a, g))); }
inline double seconds(Clock::time_point from) {
  return std::chrono::duration<double>(Clock::now() - from).count();
}

// A quadratic height fit z = c1 x + c2 y + (a x^2 + 2 b x y + c y^2) / 2 of a
// smooth sector in the frame (t1, t2, g0); `g` is its normal at the vertex.
struct Jet {
  Vec g{}, t1{}, t2{};
  double a = 0.0, b = 0.0, c = 0.0;
  double curvature(const Vec& s) const {
    const double x = dot(s, t1), y = dot(s, t2), r = x * x + y * y;
    return r > 0.0 ? (a * x * x + 2.0 * b * x * y + c * y * y) / r : 0.0;
  }
};

// Normal height a placing p + s + a g at distance D from a circle of
// curvature kappa through p tangent to the wall (exact on a sphere or a
// cylinder section); D itself where the radius is below D.
double corrected_height(double depth, double kappa, double offset) {
  const double reach = 1.0 - kappa * depth;
  const double disc = reach * reach - kappa * kappa * offset * offset;
  if (reach <= 0.0 || disc <= 0.0) return depth;
  return (2.0 * depth - kappa * depth * depth + kappa * offset * offset) / (1.0 + std::sqrt(disc));
}

// Solves the symmetric positive definite k x k system in place (Cholesky);
// false when a pivot is not positive relative to the diagonal scale.
bool cholesky_solve(std::vector<double>& m, std::vector<double>& rhs, int k) {
  double scale = 0.0;
  for (int i = 0; i < k; ++i) scale = std::max(scale, m[i * k + i]);
  if (!(scale > 0.0)) return false;
  for (int j = 0; j < k; ++j) {
    double d = m[j * k + j];
    for (int p = 0; p < j; ++p) d -= m[j * k + p] * m[j * k + p];
    if (!(d > 1e-13 * scale)) return false;
    d = std::sqrt(d);
    m[j * k + j] = d;
    for (int i = j + 1; i < k; ++i) {
      double s = m[i * k + j];
      for (int p = 0; p < j; ++p) s -= m[i * k + p] * m[j * k + p];
      m[i * k + j] = s / d;
    }
  }
  for (int i = 0; i < k; ++i) {
    double s = rhs[i];
    for (int p = 0; p < i; ++p) s -= m[i * k + p] * rhs[p];
    rhs[i] = s / m[i * k + i];
  }
  for (int i = k - 1; i >= 0; --i) {
    double s = rhs[i];
    for (int p = i + 1; p < k; ++p) s -= m[p * k + i] * rhs[p];
    rhs[i] = s / m[i * k + i];
  }
  return true;
}

struct Wall {
  int64_t n = 0, m = 0;
  const double* x = nullptr;
  const int32_t* f = nullptr;
  std::vector<int64_t> opposite;       // half-edge 3 f + k (F[k] -> F[k+1]) -> opposite half-edge
  std::vector<uint8_t> feature;        // per half-edge
  std::vector<Vec> normal;             // unit, growth side
  std::vector<std::array<double, 3>> angle;
  std::vector<int64_t> fan_offsets;    // ordered fan of each vertex
  std::vector<int64_t> fan;            // half-edge 3 f + i with F[i] = v, in rotation order
  std::vector<int64_t> ring_offsets;   // neighbors of each vertex
  std::vector<int32_t> ring;
};

int32_t build_wall(Wall& w, double feature_cos) {
  const int64_t n = w.n, m = w.m;
  const int32_t* f = w.f;
  // Opposite half-edges: sort by undirected key.
  std::vector<std::pair<uint64_t, int64_t>> keys(static_cast<size_t>(3 * m));
  for (int64_t h = 0; h < 3 * m; ++h) {
    const int64_t t = h / 3, k = h % 3;
    const uint32_t a = static_cast<uint32_t>(f[3 * t + k]);
    const uint32_t b = static_cast<uint32_t>(f[3 * t + (k + 1) % 3]);
    if (a == b) return PHX_MC_INVALID_INPUT;
    keys[static_cast<size_t>(h)] = {(static_cast<uint64_t>(std::min(a, b)) << 32) | std::max(a, b), h};
  }
  std::sort(keys.begin(), keys.end());
  w.opposite.assign(static_cast<size_t>(3 * m), -1);
  for (size_t i = 0; i < keys.size(); i += 2) {
    if (i + 1 >= keys.size() || keys[i].first != keys[i + 1].first ||
        (i + 2 < keys.size() && keys[i + 2].first == keys[i].first)) {
      return PHX_MC_INVALID_INPUT;  // boundary or non-manifold edge
    }
    const int64_t h0 = keys[i].second, h1 = keys[i + 1].second;
    if (f[3 * (h0 / 3) + h0 % 3] != f[3 * (h1 / 3) + (h1 % 3 + 1) % 3]) {
      return PHX_MC_INVALID_INPUT;  // inconsistent orientation
    }
    w.opposite[static_cast<size_t>(h0)] = h1;
    w.opposite[static_cast<size_t>(h1)] = h0;
  }
  // Normals and corner angles.
  w.normal.resize(static_cast<size_t>(m));
  w.angle.resize(static_cast<size_t>(m));
  for (int64_t t = 0; t < m; ++t) {
    const double* p[3] = {w.x + 3 * f[3 * t], w.x + 3 * f[3 * t + 1], w.x + 3 * f[3 * t + 2]};
    const Vec nrm = cross(sub(p[1], p[0]), sub(p[2], p[0]));
    if (!(norm(nrm) > 0.0)) return PHX_MC_DEGENERATE_INPUT;
    w.normal[static_cast<size_t>(t)] = unit(nrm);
    for (int k = 0; k < 3; ++k) {
      const Vec e1 = sub(p[(k + 1) % 3], p[k]), e2 = sub(p[(k + 2) % 3], p[k]);
      w.angle[static_cast<size_t>(t)][k] = std::atan2(norm(cross(e1, e2)), dot(e1, e2));
    }
  }
  w.feature.assign(static_cast<size_t>(3 * m), 0);
  for (int64_t h = 0; h < 3 * m; ++h) {
    const int64_t o = w.opposite[static_cast<size_t>(h)];
    w.feature[static_cast<size_t>(h)] =
        dot(w.normal[static_cast<size_t>(h / 3)], w.normal[static_cast<size_t>(o / 3)]) <
        feature_cos;
  }
  // Ordered fans: rotate across the outgoing half-edge v -> F[i+1].
  std::vector<int64_t> count(static_cast<size_t>(n) + 1, 0), first(static_cast<size_t>(n), -1);
  for (int64_t h = 0; h < 3 * m; ++h) {
    const int32_t v = f[h];
    ++count[static_cast<size_t>(v) + 1];
    if (first[static_cast<size_t>(v)] < 0) first[static_cast<size_t>(v)] = h;
  }
  w.fan_offsets.assign(static_cast<size_t>(n) + 1, 0);
  for (int64_t v = 0; v < n; ++v) {
    if (count[static_cast<size_t>(v) + 1] == 0) return PHX_MC_INVALID_INPUT;  // unused vertex
    w.fan_offsets[static_cast<size_t>(v) + 1] =
        w.fan_offsets[static_cast<size_t>(v)] + count[static_cast<size_t>(v) + 1];
  }
  w.fan.resize(static_cast<size_t>(3 * m));
  for (int64_t v = 0; v < n; ++v) {
    const int64_t begin = w.fan_offsets[static_cast<size_t>(v)];
    const int64_t size = w.fan_offsets[static_cast<size_t>(v) + 1] - begin;
    int64_t h = first[static_cast<size_t>(v)];
    for (int64_t j = 0; j < size; ++j) {
      if (j > 0 && h == first[static_cast<size_t>(v)]) return PHX_MC_INVALID_INPUT;
      w.fan[static_cast<size_t>(begin + j)] = h;
      // The opposite of v -> F[i+1] is F[i+1] -> v in the next face; v is its head.
      const int64_t o = w.opposite[static_cast<size_t>(h)];
      h = 3 * (o / 3) + (o % 3 + 1) % 3;
    }
    if (h != first[static_cast<size_t>(v)]) return PHX_MC_INVALID_INPUT;  // non-manifold vertex
  }
  // Neighbor rings: the head of each outgoing half-edge, in fan order.
  w.ring_offsets = w.fan_offsets;
  w.ring.resize(static_cast<size_t>(3 * m));
  for (int64_t j = 0; j < 3 * m; ++j) {
    const int64_t h = w.fan[static_cast<size_t>(j)];
    w.ring[static_cast<size_t>(j)] = f[3 * (h / 3) + (h % 3 + 1) % 3];
  }
  return PHX_MC_OK;
}

// Sectors of a vertex fan: [begin, end) runs of fan positions (cyclic,
// end may exceed the fan size) between feature crossings; one run covering
// the fan when fewer than two feature edges meet the vertex.
struct Sector {
  int64_t begin, end;
};

void vertex_sectors(const Wall& w, int64_t v, std::vector<Sector>& sectors) {
  sectors.clear();
  const int64_t b = w.fan_offsets[static_cast<size_t>(v)];
  const int64_t size = w.fan_offsets[static_cast<size_t>(v) + 1] - b;
  // Fan position j is crossed into from j - 1 across the incoming edge of
  // its face, which is the outgoing half-edge of position j - 1.
  std::vector<int64_t> cuts;
  for (int64_t j = 0; j < size; ++j) {
    const int64_t h = w.fan[static_cast<size_t>(b + j)];
    if (w.feature[static_cast<size_t>(h)]) cuts.push_back(j + 1);  // cut before position j + 1
  }
  if (cuts.size() < 2) {
    sectors.push_back({0, size});
    return;
  }
  for (size_t i = 0; i < cuts.size(); ++i) {
    const int64_t start = cuts[i] % size;
    int64_t stop = cuts[(i + 1) % cuts.size()] % size;
    if (stop <= start) stop += size;
    sectors.push_back({start, stop});
  }
}

Jet fit_jet(const Wall& w, int64_t v, const Sector& sector, std::vector<int32_t>& scratch) {
  const int64_t b = w.fan_offsets[static_cast<size_t>(v)];
  const int64_t size = w.fan_offsets[static_cast<size_t>(v) + 1] - b;
  const double* p = w.x + 3 * v;
  Jet jet;
  Vec g0{0.0, 0.0, 0.0};
  scratch.clear();
  for (int64_t j = sector.begin; j < sector.end; ++j) {
    const int64_t h = w.fan[static_cast<size_t>(b + j % size)];
    const int64_t t = h / 3;
    g0 = add(g0, mul(w.normal[static_cast<size_t>(t)], w.angle[static_cast<size_t>(t)][h % 3]));
    for (int k = 0; k < 3; ++k) {
      scratch.push_back(w.f[3 * t + k]);
      const int64_t e = 3 * t + k;
      if (w.feature[static_cast<size_t>(e)]) continue;
      const int64_t o = w.opposite[static_cast<size_t>(e)] / 3;
      for (int q = 0; q < 3; ++q) scratch.push_back(w.f[3 * o + q]);
    }
  }
  std::sort(scratch.begin(), scratch.end());
  scratch.erase(std::unique(scratch.begin(), scratch.end()), scratch.end());
  g0 = unit(g0);
  jet.g = g0;
  const Vec helper = std::fabs(g0[0]) < 0.6 ? Vec{1.0, 0.0, 0.0} : Vec{0.0, 1.0, 0.0};
  jet.t1 = unit(tangential(helper, g0));
  jet.t2 = cross(g0, jet.t1);
  // Weighted normal equations of the five-term fit, coordinates scaled by
  // the mean neighbor distance; weight 1 / |d|^2 (scaled).
  double reach = 0.0;
  int used = 0;
  for (const int32_t q : scratch) {
    if (q == v) continue;
    reach += norm(sub(w.x + 3 * q, p));
    ++used;
  }
  if (used < 2 || !(reach > 0.0)) return jet;
  reach /= used;
  std::vector<double> m5(25, 0.0), r5(5, 0.0), m2(4, 0.0), r2(2, 0.0);
  for (const int32_t q : scratch) {
    if (q == v) continue;
    const Vec d = mul(sub(w.x + 3 * q, p), 1.0 / reach);
    const double x = dot(d, jet.t1), y = dot(d, jet.t2), z = dot(d, g0);
    const double weight = 1.0 / std::max(dot(d, d), 1e-300);
    const double row[5] = {x, y, 0.5 * x * x, x * y, 0.5 * y * y};
    for (int i = 0; i < 5; ++i) {
      r5[i] += weight * row[i] * z;
      for (int k = 0; k < 5; ++k) m5[i * 5 + k] += weight * row[i] * row[k];
    }
    for (int i = 0; i < 2; ++i) {
      r2[i] += weight * row[i] * z;
      for (int k = 0; k < 2; ++k) m2[i * 2 + k] += weight * row[i] * row[k];
    }
  }
  double c1 = 0.0, c2 = 0.0;
  if (used >= 5 && cholesky_solve(m5, r5, 5)) {
    c1 = r5[0];
    c2 = r5[1];
    jet.a = r5[2] / reach;
    jet.b = r5[3] / reach;
    jet.c = r5[4] / reach;
  } else if (cholesky_solve(m2, r2, 2)) {
    c1 = r2[0];
    c2 = r2[1];
  } else {
    return jet;
  }
  const Vec g = unit(sub(g0, add(mul(jet.t1, c1), mul(jet.t2, c2))));
  // Keep the fitted normal only where it stays visible from the sector.
  if (dot(g, g0) > 0.5) jet.g = g;
  return jet;
}

// Minimum-norm w with w . g_i = a_i (Tikhonov-stabilized for dependent
// normals); returns false on a singular system.
bool medial(const std::vector<Jet>& jets, const std::vector<double>& heights, Vec& w) {
  const int k = static_cast<int>(jets.size());
  std::vector<double> m(static_cast<size_t>(k * k)), r(heights);
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) m[static_cast<size_t>(i * k + j)] = dot(jets[i].g, jets[j].g);
    m[static_cast<size_t>(i * k + i)] += 1e-12;
  }
  if (!cholesky_solve(m, r, k)) return false;
  w = {0.0, 0.0, 0.0};
  for (int i = 0; i < k; ++i) w = add(w, mul(jets[i].g, r[static_cast<size_t>(i)]));
  return true;
}

// The Jacobian of a linear prism (bottom q0 q1 q2, tops q3 q4 q5) is
// sum_i N_i det(A(z), B(z), u_i) with A(z) = (1 - z) (q1 - q0) + z (q4 - q3),
// B(z) likewise from q2, and u_i = q(i+3) - q(i): degree one on the triangle
// and two in z. Its Bernstein controls are the six corner Jacobians (z = 0, 1)
// and, per vertex i, the mixed control (det(A0, B1, u_i) + det(A1, B0, u_i)) / 2;
// all nine positive prove the Jacobian positive on the whole prism.
double mixed_control(const double* const q[6], int i) {
  const Vec a0 = sub(q[1], q[0]), b0 = sub(q[2], q[0]);
  const Vec a1 = sub(q[4], q[3]), b1 = sub(q[5], q[3]);
  const Vec u = sub(q[3 + i], q[i]);
  return dot(a0, cross(b1, u)) + dot(a1, cross(b0, u));
}

// Exact sign of the mixed control: a float evaluation decided by a forward
// error bound, else expansion arithmetic.
int mixed_control_sign(const double* const q[6], int i) {
  // (A, B) of each determinant as (head, base) point indices.
  constexpr int kTerms[2][4] = {{1, 0, 5, 3}, {4, 3, 2, 0}};
  double value = 0.0, magnitude = 0.0;
  for (const auto& t : kTerms) {
    const Vec a = sub(q[t[0]], q[t[1]]), b = sub(q[t[2]], q[t[3]]), u = sub(q[3 + i], q[i]);
    value += dot(a, cross(b, u));
    for (int k = 0; k < 3; ++k) {
      const int j = (k + 1) % 3, l = (k + 2) % 3;
      magnitude += std::fabs(a[k]) * (std::fabs(b[j] * u[l]) + std::fabs(b[l] * u[j]));
    }
  }
  // Three rounded differences, two products and up to twelve additions stay
  // below 32 unit roundoffs of the absolute sum (plus a subnormal floor).
  const double bound = 32.0 * std::numeric_limits<double>::epsilon() * magnitude +
                       64.0 * std::numeric_limits<double>::denorm_min();
  if (value > bound) return 1;
  if (value < -bound) return -1;
  Expansion total;
  for (const auto& t : kTerms) {
    Expansion a[3], b[3], u[3];
    for (int k = 0; k < 3; ++k) {
      a[k] = Expansion::difference(q[t[0]][k], q[t[1]][k]);
      b[k] = Expansion::difference(q[t[2]][k], q[t[3]][k]);
      u[k] = Expansion::difference(q[3 + i][k], q[i][k]);
    }
    for (int k = 0; k < 3; ++k) {
      const int j = (k + 1) % 3, l = (k + 2) % 3;
      total = total + a[k] * (b[j] * u[l] - b[l] * u[j]);
    }
  }
  return total.sign();
}

struct Grid {
  double lo[3] = {0, 0, 0};
  double cell = 1.0;
  int64_t key(const double* x) const {
    int64_t c[3];
    for (int a = 0; a < 3; ++a) c[a] = static_cast<int64_t>(std::floor((x[a] - lo[a]) / cell));
    return (c[0] << 42) | (c[1] << 21) | c[2];
  }
};

}  // namespace

int32_t grow(const Input& in, const Output& out) {
  const Clock::time_point start = Clock::now();
  double* stats = out.stats;
  std::fill(stats, stats + kStatCount, 0.0);
  Wall w;
  w.n = in.vertex_count;
  w.m = in.triangle_count;
  w.x = in.vertices;
  w.f = in.triangles;
  const double kPi = 3.14159265358979323846;
  int32_t status = build_wall(w, std::cos(in.feature_angle_degrees * kPi / 180.0));
  if (status != PHX_MC_OK) return status;
  const int64_t n = w.n, m = w.m;
  const int32_t layers = in.layer_count;
  const double depth = in.heights[layers - 1];
  int64_t feature_edges = 0;
  for (const uint8_t e : w.feature) feature_edges += e;
  stats[18] = static_cast<double>(feature_edges / 2);
  stats[0] = seconds(start);

  // Jets and directions.
  Clock::time_point stage = Clock::now();
  std::vector<Jet> smooth_jet(static_cast<size_t>(n));
  std::vector<Vec> disp(static_cast<size_t>(n));
  std::vector<uint8_t> kind(static_cast<size_t>(n), 0);
  std::vector<std::vector<Jet>> sector_jets(static_cast<size_t>(n));
  std::vector<Sector> sectors;
  std::vector<int32_t> scratch;
  for (int64_t v = 0; v < n; ++v) {
    vertex_sectors(w, v, sectors);
    if (sectors.size() == 1) {
      smooth_jet[static_cast<size_t>(v)] = fit_jet(w, v, sectors[0], scratch);
      continue;
    }
    kind[static_cast<size_t>(v)] = sectors.size() == 2 ? 1 : 2;
    auto& jets = sector_jets[static_cast<size_t>(v)];
    for (const Sector& s : sectors) jets.push_back(fit_jet(w, v, s, scratch));
  }
  stats[1] = seconds(stage);
  stage = Clock::now();
  double largest_offset = 0.0;
  for (int64_t v = 0; v < n; ++v) {
    if (kind[static_cast<size_t>(v)] == 0) {
      disp[static_cast<size_t>(v)] = mul(smooth_jet[static_cast<size_t>(v)].g, depth);
      continue;
    }
    const auto& jets = sector_jets[static_cast<size_t>(v)];
    std::vector<double> heights(jets.size(), depth);
    Vec wv{0.0, 0.0, 0.0};
    Vec mean{0.0, 0.0, 0.0};
    for (const Jet& j : jets) mean = add(mean, j.g);
    if (!medial(jets, heights, wv)) wv = mul(unit(mean), depth);
    // Curvature correction: a fixed point that contracts while the column
    // offset stays below the curvature radius; stop when it no longer moves.
    double previous = std::numeric_limits<double>::infinity();
    while (true) {
      double change = 0.0;
      for (size_t i = 0; i < jets.size(); ++i) {
        const Vec s = tangential(wv, jets[i].g);
        const double a = corrected_height(depth, jets[i].curvature(s), norm(s));
        change = std::max(change, std::fabs(a - heights[i]));
        heights[i] = a;
      }
      Vec next{};
      if (!medial(jets, heights, next)) break;
      wv = next;
      if (!(change < previous) || change <= 1e-14 * depth) break;
      previous = change;
    }
    disp[static_cast<size_t>(v)] = wv;
    for (const Jet& j : jets) largest_offset = std::max(largest_offset, norm(tangential(wv, j.g)));
    stats[kind[static_cast<size_t>(v)] == 1 ? 7 : 8] += 1.0;
  }
  stats[2] = seconds(stage);

  // Relaxation of the smooth columns' tangential offsets near features.
  stage = Clock::now();
  if (largest_offset > 0.0) {
    const double decay = kDecayFactor * largest_offset;
    const double tolerance = kRelaxationTolerance * depth;
    std::vector<Vec> offset(static_cast<size_t>(n), Vec{0.0, 0.0, 0.0});
    std::vector<double> screen(static_cast<size_t>(n), 0.0);
    std::vector<uint8_t> queued(static_cast<size_t>(n), 0);
    std::deque<int64_t> work;
    for (int64_t v = 0; v < n; ++v) {
      if (kind[static_cast<size_t>(v)] != 0) continue;
      double length = 0.0;
      bool near = false;
      const int64_t b = w.ring_offsets[static_cast<size_t>(v)];
      const int64_t e = w.ring_offsets[static_cast<size_t>(v) + 1];
      for (int64_t j = b; j < e; ++j) {
        const int32_t u = w.ring[static_cast<size_t>(j)];
        length += norm(sub(w.x + 3 * u, w.x + 3 * v));
        near = near || kind[static_cast<size_t>(u)] != 0;
      }
      length /= static_cast<double>(e - b);
      screen[static_cast<size_t>(v)] = 1.0 + (length / decay) * (length / decay);
      if (near) {
        work.push_back(v);
        queued[static_cast<size_t>(v)] = 1;
      }
    }
    int64_t updates = 0;
    while (!work.empty()) {
      const int64_t v = work.front();
      work.pop_front();
      queued[static_cast<size_t>(v)] = 0;
      ++updates;
      const Vec& g = smooth_jet[static_cast<size_t>(v)].g;
      Vec sum{0.0, 0.0, 0.0};
      const int64_t b = w.ring_offsets[static_cast<size_t>(v)];
      const int64_t e = w.ring_offsets[static_cast<size_t>(v) + 1];
      for (int64_t j = b; j < e; ++j) {
        const int32_t u = w.ring[static_cast<size_t>(j)];
        sum = add(sum, kind[static_cast<size_t>(u)] != 0 ? disp[static_cast<size_t>(u)]
                                                         : offset[static_cast<size_t>(u)]);
      }
      const Vec next = mul(tangential(sum, g),
                           1.0 / (static_cast<double>(e - b) * screen[static_cast<size_t>(v)]));
      const double change = norm(sub(next, offset[static_cast<size_t>(v)]));
      offset[static_cast<size_t>(v)] = next;
      if (change <= tolerance) continue;
      for (int64_t j = b; j < e; ++j) {
        const int32_t u = w.ring[static_cast<size_t>(j)];
        if (kind[static_cast<size_t>(u)] == 0 && !queued[static_cast<size_t>(u)]) {
          queued[static_cast<size_t>(u)] = 1;
          work.push_back(u);
        }
      }
    }
    for (int64_t v = 0; v < n; ++v) {
      if (kind[static_cast<size_t>(v)] != 0) continue;
      const Vec& s = offset[static_cast<size_t>(v)];
      const double length = norm(s);
      if (length <= tolerance) continue;
      const Jet& jet = smooth_jet[static_cast<size_t>(v)];
      disp[static_cast<size_t>(v)] =
          add(s, mul(jet.g, corrected_height(depth, jet.curvature(s), length)));
      stats[9] += 1.0;
    }
    stats[19] = static_cast<double>(updates);
  }
  stats[3] = seconds(stage);

  // Levels and float repair.
  stage = Clock::now();
  std::vector<double> scale(static_cast<size_t>(n), 1.0);
  const int64_t level_rows = n;
  auto place = [&](int64_t v) {
    const double* p = w.x + 3 * v;
    for (int a = 0; a < 3; ++a) out.points[3 * v + a] = p[a];
    for (int32_t k = 1; k <= layers; ++k) {
      const double t = in.heights[k - 1] / depth * scale[static_cast<size_t>(v)];
      double* q = out.points + 3 * (k * level_rows + v);
      for (int a = 0; a < 3; ++a) q[a] = p[a] + t * disp[static_cast<size_t>(v)][a];
    }
  };
  for (int64_t v = 0; v < n; ++v) place(v);
  for (int32_t k = 0; k < layers; ++k) {
    for (int64_t t = 0; t < m; ++t) {
      int32_t* row = out.prisms + 6 * (k * m + t);
      for (int c = 0; c < 3; ++c) {
        row[c] = static_cast<int32_t>(w.f[3 * t + c] + k * level_rows);
        row[3 + c] = static_cast<int32_t>(w.f[3 * t + c] + (k + 1) * level_rows);
      }
    }
  }
  // Float Bernstein controls of one prism: the least corner Jacobian, the
  // largest corner magnitude and the least mixed control.
  auto corners = [&](int64_t prism, double& lowest, double& largest, double& mixed) {
    const int32_t* r = out.prisms + 6 * prism;
    const double* q[6];
    for (int c = 0; c < 6; ++c) q[c] = out.points + 3 * static_cast<int64_t>(r[c]);
    lowest = std::numeric_limits<double>::infinity();
    mixed = lowest;
    largest = 0.0;
    for (int c = 0; c < 3; ++c) {
      const int a = c, b = (c + 1) % 3, d = (c + 2) % 3;
      const Vec up = sub(q[a + 3], q[a]);
      const double low = dot(cross(sub(q[b], q[a]), sub(q[d], q[a])), up);
      const double high = dot(cross(sub(q[b + 3], q[a + 3]), sub(q[d + 3], q[a + 3])), up);
      lowest = std::min({lowest, low, high});
      largest = std::max({largest, std::fabs(low), std::fabs(high)});
      mixed = std::min(mixed, mixed_control(q, c));
    }
  };
  std::vector<uint8_t> marked(static_cast<size_t>(n), 0);
  int64_t rounds = 0;
  // Thins the marked columns; false when every marked column is at the minimum.
  auto thin = [&]() {
    bool any = false;
    for (int64_t v = 0; v < n; ++v) {
      if (!marked[static_cast<size_t>(v)]) continue;
      marked[static_cast<size_t>(v)] = 0;
      double& s = scale[static_cast<size_t>(v)];
      if (s <= in.minimum_scale) continue;
      s = std::max(in.minimum_scale, s * kThinning);
      place(v);
      any = true;
    }
    return any;
  };
  int64_t float_invalid = 0;
  int64_t detected = 0;  // failures over every construction and certification round
  double minimum_ratio = std::numeric_limits<double>::infinity();
  while (true) {
    float_invalid = 0;
    minimum_ratio = std::numeric_limits<double>::infinity();
    for (int64_t prism = 0; prism < layers * m; ++prism) {
      double lowest = 0.0, largest = 0.0, mixed = 0.0;
      corners(prism, lowest, largest, mixed);
      minimum_ratio = std::min(minimum_ratio, largest > 0.0 ? lowest / largest : -1.0);
      if (lowest > 0.0 && mixed > 0.0) continue;
      ++float_invalid;
      const int64_t t = prism % m;
      for (int c = 0; c < 3; ++c) marked[static_cast<size_t>(w.f[3 * t + c])] = 1;
    }
    detected += float_invalid;
    if (float_invalid == 0 || !thin()) break;
    ++rounds;
  }
  const int64_t float_rounds = rounds;
  stats[4] = seconds(stage);

  // Certification.
  stage = Clock::now();
  int64_t invalid = float_invalid, crossings = -1;
  if (in.prove && float_invalid == 0) {
    while (true) {
      invalid = 0;
      for (int64_t prism = 0; prism < layers * m; ++prism) {
        const int32_t* r = out.prisms + 6 * prism;
        const double* q[6];
        for (int c = 0; c < 6; ++c) q[c] = out.points + 3 * static_cast<int64_t>(r[c]);
        bool ok = true;
        for (int c = 0; c < 3 && ok; ++c) {
          ok = orient3d_filtered(q[0], q[1], q[2], q[3 + c]) > 0 &&
               orient3d_filtered(q[3], q[4], q[5], q[c]) < 0 && mixed_control_sign(q, c) > 0;
        }
        if (ok) continue;
        ++invalid;
        const int64_t t = prism % m;
        for (int c = 0; c < 3; ++c) marked[static_cast<size_t>(w.f[3 * t + c])] = 1;
      }
      // Top level against itself and the wall, on a uniform grid of the
      // triangles' boxes; a pair is decided in the cell holding the lower
      // corner of its box overlap.
      crossings = 0;
      const int64_t top = static_cast<int64_t>(layers) * level_rows;
      struct Box {
        double lo[3], hi[3];
      };
      std::vector<Box> box(static_cast<size_t>(2 * m));
      double edge = 0.0;
      Grid grid;
      for (int a = 0; a < 3; ++a) grid.lo[a] = std::numeric_limits<double>::infinity();
      double hi[3] = {-grid.lo[0], -grid.lo[1], -grid.lo[2]};
      auto vertex = [&](int64_t tri, int c) {
        const int64_t v = w.f[3 * (tri % m) + c];
        return out.points + 3 * (tri < m ? top + v : v);
      };
      for (int64_t tri = 0; tri < 2 * m; ++tri) {
        Box& bx = box[static_cast<size_t>(tri)];
        for (int a = 0; a < 3; ++a) {
          bx.lo[a] = std::numeric_limits<double>::infinity();
          bx.hi[a] = -bx.lo[a];
        }
        for (int c = 0; c < 3; ++c) {
          const double* p = vertex(tri, c);
          for (int a = 0; a < 3; ++a) {
            bx.lo[a] = std::min(bx.lo[a], p[a]);
            bx.hi[a] = std::max(bx.hi[a], p[a]);
          }
        }
        for (int a = 0; a < 3; ++a) {
          grid.lo[a] = std::min(grid.lo[a], bx.lo[a]);
          hi[a] = std::max(hi[a], bx.hi[a]);
        }
        if (tri < m) edge += norm(sub(vertex(tri, 1), vertex(tri, 0)));
      }
      double extent = 0.0;
      for (int a = 0; a < 3; ++a) extent = std::max(extent, hi[a] - grid.lo[a]);
      grid.cell = std::max(edge / static_cast<double>(m), extent / static_cast<double>(1 << 20));
      std::vector<std::pair<int64_t, int64_t>> entries;
      entries.reserve(static_cast<size_t>(16 * m));
      for (int64_t tri = 0; tri < 2 * m; ++tri) {
        const Box& bx = box[static_cast<size_t>(tri)];
        int64_t lo[3], up[3];
        for (int a = 0; a < 3; ++a) {
          lo[a] = static_cast<int64_t>(std::floor((bx.lo[a] - grid.lo[a]) / grid.cell));
          up[a] = static_cast<int64_t>(std::floor((bx.hi[a] - grid.lo[a]) / grid.cell));
        }
        for (int64_t i = lo[0]; i <= up[0]; ++i) {
          for (int64_t j = lo[1]; j <= up[1]; ++j) {
            for (int64_t k = lo[2]; k <= up[2]; ++k) {
              entries.emplace_back((i << 42) | (j << 21) | k, tri);
            }
          }
        }
      }
      std::sort(entries.begin(), entries.end());
      for (size_t s = 0; s < entries.size();) {
        size_t e = s;
        while (e < entries.size() && entries[e].first == entries[s].first) ++e;
        for (size_t i = s; i < e; ++i) {
          const int64_t ta = entries[i].second;
          for (size_t j = i + 1; j < e; ++j) {
            const int64_t tb = entries[j].second;
            if (ta >= m && tb >= m) continue;  // wall pairs are the input
            const Box& ba = box[static_cast<size_t>(ta)];
            const Box& bb = box[static_cast<size_t>(tb)];
            double corner[3];
            bool overlap = true;
            for (int a = 0; a < 3; ++a) {
              overlap = overlap && ba.lo[a] <= bb.hi[a] && bb.lo[a] <= ba.hi[a];
              corner[a] = std::max(ba.lo[a], bb.lo[a]);
            }
            if (!overlap || grid.key(corner) != entries[s].first) continue;
            const double* first[3] = {vertex(ta, 0), vertex(ta, 1), vertex(ta, 2)};
            const double* second[3] = {vertex(tb, 0), vertex(tb, 1), vertex(tb, 2)};
            int64_t ida[3], idb[3];
            for (int c = 0; c < 3; ++c) {
              ida[c] = w.f[3 * (ta % m) + c] + (ta < m ? top : 0);
              idb[c] = w.f[3 * (tb % m) + c] + (tb < m ? top : 0);
            }
            Contact contact;
            const int32_t result = intersect_triangles(first, second, ida, idb, contact);
            if (result == PHX_MC_OK &&
                (contact.kind == PHX_MC_DISJOINT || contact.kind == PHX_MC_SHARED_VERTEX ||
                 contact.kind == PHX_MC_SHARED_EDGE)) {
              continue;
            }
            ++crossings;
            for (const int64_t tri : {ta, tb}) {
              if (tri >= m) continue;
              for (int c = 0; c < 3; ++c) marked[static_cast<size_t>(w.f[3 * tri + c])] = 1;
            }
          }
        }
        s = e;
      }
      detected += invalid + crossings;
      if ((invalid == 0 && crossings == 0) || !thin()) break;
      ++rounds;
      // Thinning may invert a prism in float again; the exact pass decides.
    }
  }
  stats[5] = seconds(stage);

  // Outputs and evidence.
  double minimum_scale = 1.0;
  for (int64_t v = 0; v < n; ++v) {
    const double s = scale[static_cast<size_t>(v)];
    out.scale[v] = s;
    out.vertex_kind[v] = kind[static_cast<size_t>(v)];
    for (int a = 0; a < 3; ++a) out.displacement[3 * v + a] = disp[static_cast<size_t>(v)][a];
    minimum_scale = std::min(minimum_scale, s);
    if (s < 1.0) {
      stats[10] += 1.0;
      if (kind[static_cast<size_t>(v)] != 0) stats[11] += 1.0;
    }
  }
  if (rounds > float_rounds) {
    // Certification thinned columns: the float ratio of the final levels.
    minimum_ratio = std::numeric_limits<double>::infinity();
    for (int64_t prism = 0; prism < layers * m; ++prism) {
      double lowest = 0.0, largest = 0.0, mixed = 0.0;
      corners(prism, lowest, largest, mixed);
      minimum_ratio = std::min(minimum_ratio, largest > 0.0 ? lowest / largest : -1.0);
    }
  }
  const bool certified = in.prove && invalid == 0 && crossings == 0;
  stats[12] = minimum_scale;
  stats[13] = static_cast<double>(rounds);
  stats[14] = static_cast<double>(invalid);
  stats[15] = static_cast<double>(crossings);
  stats[16] = certified ? 1.0 : 0.0;
  stats[17] = minimum_ratio;
  stats[20] = static_cast<double>(detected);
  stats[6] = seconds(start);
  if (invalid != 0 || crossings > 0) return PHX_MC_UNRESOLVED;
  return PHX_MC_OK;
}

}  // namespace phx::mc::prism_layers
