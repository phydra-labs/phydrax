//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Overset assembly against independent brute-force references: holes by the
// float solid-angle winding number away from the wall, fringe by the cell
// rule, every donor by brute-force float location over all cells, and the
// finest-donor choice among every containing active cell.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <tuple>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Part {
  std::vector<double> x;    // (n, 3)
  std::vector<int32_t> c;   // (m, 8), local
  std::vector<int32_t> f;   // (k, 4), local
  std::vector<uint8_t> role;
};

// Background [-h, h]^3 with N^3 cells: hexahedra, or six Kuhn tetrahedra each.
Part background(int n, double h, bool tets) {
  Part p;
  const int n1 = n + 1;
  for (int k = 0; k < n1; ++k) {
    for (int j = 0; j < n1; ++j) {
      for (int i = 0; i < n1; ++i) {
        p.x.insert(p.x.end(), {-h + 2 * h * i / n, -h + 2 * h * j / n, -h + 2 * h * k / n});
      }
    }
  }
  auto id = [&](int i, int j, int k) { return (k * n1 + j) * n1 + i; };
  static constexpr int kKuhn[6][4] = {{0, 1, 2, 6}, {0, 2, 3, 6}, {0, 3, 7, 6},
                                      {0, 7, 4, 6}, {0, 4, 5, 6}, {0, 5, 1, 6}};
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const int v[8] = {id(i, j, k),         id(i + 1, j, k),     id(i + 1, j + 1, k),
                          id(i, j + 1, k),     id(i, j, k + 1),     id(i + 1, j, k + 1),
                          id(i + 1, j + 1, k + 1), id(i, j + 1, k + 1)};
        if (!tets) {
          p.c.insert(p.c.end(), v, v + 8);
          continue;
        }
        for (const auto& t : kKuhn) {
          p.c.insert(p.c.end(), {v[t[0]], v[t[1]], v[t[2]], v[t[3]], -1, -1, -1, -1});
        }
      }
    }
  }
  return p;
}

// Cubed-sphere hexahedral shell r in [r0, r1]; wall faces at r0, receptor
// boundary at r1. With tets, each hexahedron splits into six tetrahedra on
// its 0-6 diagonal and each face into two triangles.
Part shell(int n, int layers, double r0, double r1, bool tets) {
  Part p;
  std::map<std::tuple<long, long, long>, int> key;
  auto node = [&](const double* q, int layer) {
    const double s = 1.0 / std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
    const double r = r0 + (r1 - r0) * layer / layers;
    const double X = q[0] * s * r, Y = q[1] * s * r, Z = q[2] * s * r;
    const auto k = std::make_tuple(std::lround(X * 1e9), std::lround(Y * 1e9), std::lround(Z * 1e9));
    const auto it = key.find(k);
    if (it != key.end()) return it->second;
    const int id = static_cast<int>(p.x.size() / 3);
    key.emplace(k, id);
    p.x.insert(p.x.end(), {X, Y, Z});
    return id;
  };
  auto facept = [](int f, double a, double b, double* q) {
    const double rows[6][3] = {{1, a, b}, {-1, -a, b}, {-a, 1, b}, {a, -1, b}, {-b, a, 1}, {b, a, -1}};
    for (int d = 0; d < 3; ++d) q[d] = rows[f][d];
  };
  std::vector<int> ids((n + 1) * (n + 1) * (layers + 1));
  for (int f = 0; f < 6; ++f) {
    for (int l = 0; l <= layers; ++l) {
      for (int j = 0; j <= n; ++j) {
        for (int i = 0; i <= n; ++i) {
          double q[3];
          facept(f, std::tan(kPi / 4 * (-1 + 2.0 * i / n)), std::tan(kPi / 4 * (-1 + 2.0 * j / n)), q);
          ids[(l * (n + 1) + j) * (n + 1) + i] = node(q, l);
        }
      }
    }
    auto I = [&](int i, int j, int l) { return ids[(l * (n + 1) + j) * (n + 1) + i]; };
    for (int l = 0; l < layers; ++l) {
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
          int v[8] = {I(i, j, l),     I(i + 1, j, l),     I(i + 1, j + 1, l),     I(i, j + 1, l),
                      I(i, j, l + 1), I(i + 1, j, l + 1), I(i + 1, j + 1, l + 1), I(i, j + 1, l + 1)};
          auto X = [&](int q, int d) { return p.x[3 * v[q] + d]; };
          double e1[3], e3[3], e4[3];
          for (int d = 0; d < 3; ++d) {
            e1[d] = X(1, d) - X(0, d);
            e3[d] = X(3, d) - X(0, d);
            e4[d] = X(4, d) - X(0, d);
          }
          const double t = (e1[1] * e3[2] - e1[2] * e3[1]) * e4[0] +
                           (e1[2] * e3[0] - e1[0] * e3[2]) * e4[1] +
                           (e1[0] * e3[1] - e1[1] * e3[0]) * e4[2];
          if (t < 0) {
            std::swap(v[1], v[3]);
            std::swap(v[5], v[7]);
          }
          if (tets) {
            static constexpr int kSplit[6][4] = {{0, 1, 2, 6}, {0, 2, 3, 6}, {0, 3, 7, 6},
                                                 {0, 7, 4, 6}, {0, 4, 5, 6}, {0, 5, 1, 6}};
            for (const auto& s : kSplit) {
              p.c.insert(p.c.end(), {v[s[0]], v[s[1]], v[s[2]], v[s[3]], -1, -1, -1, -1});
            }
          } else {
            p.c.insert(p.c.end(), v, v + 8);
          }
          if (l == 0 || l == layers - 1) {
            const int* q = l == 0 ? v : v + 4;
            const uint8_t role = l == 0 ? 1 : 2;
            if (tets) {
              p.f.insert(p.f.end(), {q[0], q[1], q[2], -1, q[0], q[2], q[3], -1});
              p.role.insert(p.role.end(), {role, role});
            } else {
              p.f.insert(p.f.end(), {q[0], q[1], q[2], q[3]});
              p.role.push_back(role);
            }
          }
        }
      }
    }
  }
  return p;
}

struct Result {
  std::vector<int8_t> vertex_status, cell_status;
  std::vector<int32_t> fringe_layer, donor_status, candidate_parts;
  std::vector<uint8_t> wall_contact, conflict, ambiguous;
  std::vector<int64_t> donor_cell;
  std::vector<double> reference, residual, measure, stats;
};

struct Assembly {
  std::vector<int64_t> voff{0}, coff{0}, foff{0};
  std::vector<double> x;
  std::vector<int32_t> c, f;
  std::vector<int64_t> ids;
  std::vector<uint8_t> excluded, role;
  std::vector<int32_t> priorities, ranks;

  void add(const Part& p) {
    const int32_t base = static_cast<int32_t>(x.size() / 3);
    x.insert(x.end(), p.x.begin(), p.x.end());
    for (int32_t v : p.c) c.push_back(v < 0 ? -1 : v + base);
    for (int32_t v : p.f) f.push_back(v < 0 ? -1 : v + base);
    role.insert(role.end(), p.role.begin(), p.role.end());
    for (size_t k = 0; k < p.c.size() / 8; ++k) ids.push_back(static_cast<int64_t>(k));
    excluded.resize(c.size() / 8, 0);
    voff.push_back(static_cast<int64_t>(x.size() / 3));
    coff.push_back(static_cast<int64_t>(c.size() / 8));
    foff.push_back(static_cast<int64_t>(f.size() / 4));
    priorities.push_back(0);
    ranks.push_back(static_cast<int32_t>(ranks.size()));
  }

  int32_t run(int32_t prove, Result& r, int32_t layers = 1, int32_t fringe_donors = 0) const {
    const int64_t nv = voff.back(), nc = coff.back();
    r.vertex_status.assign(nv, 0);
    r.fringe_layer.assign(nv, 0);
    r.wall_contact.assign(nv, 0);
    r.conflict.assign(nv, 0);
    r.cell_status.assign(nc, 0);
    r.ambiguous.assign(nc, 0);
    r.donor_status.assign(nv, 0);
    r.donor_cell.assign(nv, 0);
    r.reference.assign(3 * nv, 0);
    r.residual.assign(nv, 0);
    r.measure.assign(nv, 0);
    r.candidate_parts.assign(nv, 0);
    r.stats.assign(16, 0);
    const phx_mc_overset_parts parts{static_cast<int32_t>(priorities.size()),
                                     voff.data(), x.data(), coff.data(), c.data(), ids.data(),
                                     excluded.data(), foff.data(), f.data(), role.data(),
                                     priorities.data(), ranks.data(), layers, 0, fringe_donors, prove};
    const phx_mc_overset_result result{
        r.vertex_status.data(), r.fringe_layer.data(), r.wall_contact.data(), r.conflict.data(),
        r.cell_status.data(),   r.ambiguous.data(),    r.donor_status.data(), r.donor_cell.data(),
        r.reference.data(),     r.residual.data(),     r.measure.data(),      r.candidate_parts.data(),
        r.stats.data()};
    return phx_mc_overset_assemble(&parts, &result);
  }

  const double* p(int64_t v) const { return x.data() + 3 * v; }
  bool hex(int64_t cell) const { return c[8 * cell + 4] >= 0; }
  int32_t part_of_cell(int64_t cell) const {
    return static_cast<int32_t>(std::upper_bound(coff.begin(), coff.end(), cell) - coff.begin() - 1);
  }
  int32_t part_of_vertex(int64_t v) const {
    return static_cast<int32_t>(std::upper_bound(voff.begin(), voff.end(), v) - voff.begin() - 1);
  }
};

// Float location: the reference coordinates' distance outside the cell.
double outside_by(const Assembly& a, int64_t cell, const double* q, double* ref) {
  const int32_t* r = a.c.data() + 8 * cell;
  if (!a.hex(cell)) {
    const double *v0 = a.p(r[0]), *v1 = a.p(r[1]), *v2 = a.p(r[2]), *v3 = a.p(r[3]);
    double m[3][3], b[3];
    for (int d = 0; d < 3; ++d) {
      m[d][0] = v1[d] - v0[d];
      m[d][1] = v2[d] - v0[d];
      m[d][2] = v3[d] - v0[d];
      b[d] = q[d] - v0[d];
    }
    const double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                       m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                       m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    for (int k = 0; k < 3; ++k) {
      double mk[3][3];
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) mk[i][j] = j == k ? b[i] : m[i][j];
      }
      ref[k] = (mk[0][0] * (mk[1][1] * mk[2][2] - mk[1][2] * mk[2][1]) -
                mk[0][1] * (mk[1][0] * mk[2][2] - mk[1][2] * mk[2][0]) +
                mk[0][2] * (mk[1][0] * mk[2][1] - mk[1][1] * mk[2][0])) /
               det;
    }
    return std::max({-ref[0], -ref[1], -ref[2], ref[0] + ref[1] + ref[2] - 1.0});
  }
  static constexpr int kBit[3][8] = {{0, 1, 1, 0, 0, 1, 1, 0}, {0, 0, 1, 1, 0, 0, 1, 1},
                                     {0, 0, 0, 0, 1, 1, 1, 1}};
  ref[0] = ref[1] = ref[2] = 0.5;
  for (int it = 0; it < 60; ++it) {
    double X[3] = {0, 0, 0}, J[3][3] = {};
    for (int k = 0; k < 8; ++k) {
      double f[3], g[3];
      for (int a2 = 0; a2 < 3; ++a2) {
        f[a2] = kBit[a2][k] ? ref[a2] : 1 - ref[a2];
        g[a2] = kBit[a2][k] ? 1 : -1;
      }
      for (int d = 0; d < 3; ++d) {
        const double v = a.p(r[k])[d];
        X[d] += f[0] * f[1] * f[2] * v;
        J[d][0] += g[0] * f[1] * f[2] * v;
        J[d][1] += f[0] * g[1] * f[2] * v;
        J[d][2] += f[0] * f[1] * g[2] * v;
      }
    }
    const double det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
                       J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                       J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
    double res[3] = {X[0] - q[0], X[1] - q[1], X[2] - q[2]};
    double delta[3];
    for (int k = 0; k < 3; ++k) {
      double mk[3][3];
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) mk[i][j] = j == k ? res[i] : J[i][j];
      }
      delta[k] = (mk[0][0] * (mk[1][1] * mk[2][2] - mk[1][2] * mk[2][1]) -
                  mk[0][1] * (mk[1][0] * mk[2][2] - mk[1][2] * mk[2][0]) +
                  mk[0][2] * (mk[1][0] * mk[2][1] - mk[1][1] * mk[2][0])) /
                 det;
    }
    for (int k = 0; k < 3; ++k) ref[k] -= delta[k];
    if (std::fabs(ref[0]) + std::fabs(ref[1]) + std::fabs(ref[2]) > 50) return 1e300;
  }
  return std::max({-ref[0], -ref[1], -ref[2], ref[0] - 1, ref[1] - 1, ref[2] - 1});
}

double cell_volume(const Assembly& a, int64_t cell) {
  // Volume by 4^3 midpoint sampling of the Jacobian is not needed: compare by
  // the kernel's measure only through the order of two containing cells.
  const int32_t* r = a.c.data() + 8 * cell;
  if (!a.hex(cell)) {
    const double *v0 = a.p(r[0]), *v1 = a.p(r[1]), *v2 = a.p(r[2]), *v3 = a.p(r[3]);
    double e[3][3];
    for (int d = 0; d < 3; ++d) {
      e[0][d] = v1[d] - v0[d];
      e[1][d] = v2[d] - v0[d];
      e[2][d] = v3[d] - v0[d];
    }
    return std::fabs(e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                     e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                     e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0])) /
           6.0;
  }
  return -1.0;
}

// Solid-angle winding number of the wall faces (role 1) of `part` at q.
double winding(const Assembly& a, int32_t part, const double* q) {
  double total = 0.0;
  for (int64_t face = a.foff[part]; face < a.foff[part + 1]; ++face) {
    if (a.role[face] != 1) continue;
    const int32_t* r = a.f.data() + 4 * face;
    const int count = r[3] < 0 ? 3 : 4;
    for (int t = 1; t + 1 < count; ++t) {
      const double* corners[3] = {a.p(r[0]), a.p(r[t]), a.p(r[t + 1])};
      double v[3][3], l[3];
      for (int k = 0; k < 3; ++k) {
        for (int d = 0; d < 3; ++d) v[k][d] = corners[k][d] - q[d];
        l[k] = std::sqrt(v[k][0] * v[k][0] + v[k][1] * v[k][1] + v[k][2] * v[k][2]);
      }
      const double det = v[0][0] * (v[1][1] * v[2][2] - v[1][2] * v[2][1]) -
                         v[0][1] * (v[1][0] * v[2][2] - v[1][2] * v[2][0]) +
                         v[0][2] * (v[1][0] * v[2][1] - v[1][1] * v[2][0]);
      auto dot = [&](int i, int j) { return v[i][0] * v[j][0] + v[i][1] * v[j][1] + v[i][2] * v[j][2]; };
      const double den = l[0] * l[1] * l[2] + dot(0, 1) * l[2] + dot(1, 2) * l[0] + dot(2, 0) * l[1];
      total += 2.0 * std::atan2(det, den);
    }
  }
  return total / (4.0 * kPi);
}

void check_assembly(const Assembly& a, const Result& r, double wall_radius, double band,
                    bool fringe_donors) {
  const int64_t nv = a.voff.back(), nc = a.coff.back();
  const int32_t parts = static_cast<int32_t>(a.priorities.size());
  // Holes: the winding number of each other part's wall away from the wall.
  for (int64_t v = 0; v < nv; ++v) {
    const int32_t part = a.part_of_vertex(v);
    const double radius = std::sqrt(a.p(v)[0] * a.p(v)[0] + a.p(v)[1] * a.p(v)[1] + a.p(v)[2] * a.p(v)[2]);
    if (r.wall_contact[v]) {
      // Contact: on a wall triangle or in a wall quad's corner tetrahedron.
      PHX_CHECK(std::fabs(radius - wall_radius) < 0.05 && r.vertex_status[v] == 2);
      continue;
    }
    if (std::fabs(radius - wall_radius) < band) continue;
    bool inside = false;
    for (int32_t other = 0; other < parts; ++other) {
      if (other != part && std::fabs(winding(a, other, a.p(v))) > 0.5) inside = true;
    }
    PHX_CHECK((r.vertex_status[v] == 2) == inside);
  }
  // Fringe: non-hole vertices of hole cells are receptors or wall vertices.
  std::vector<uint8_t> wall(nv, 0);
  for (size_t face = 0; face < a.role.size(); ++face) {
    if (a.role[face] != 1) continue;
    for (int k = 0; k < 4; ++k) {
      if (a.f[4 * face + k] >= 0) wall[a.f[4 * face + k]] = 1;
    }
  }
  for (int64_t cell = 0; cell < nc; ++cell) {
    const int n = a.hex(cell) ? 8 : 4;
    bool hole_vertex = false;
    for (int k = 0; k < n; ++k) hole_vertex = hole_vertex || r.vertex_status[a.c[8 * cell + k]] == 2;
    if (hole_vertex) PHX_CHECK(r.cell_status[cell] == 2);
    if (r.cell_status[cell] != 2) continue;
    for (int k = 0; k < n; ++k) {
      const int32_t v = a.c[8 * cell + k];
      PHX_CHECK(r.vertex_status[v] == 2 || r.vertex_status[v] == 1 || wall[v]);
    }
  }
  // Donors: brute force over every cell of every other part.
  int64_t receptors = 0;
  for (int64_t v = 0; v < nv; ++v) {
    if (r.vertex_status[v] != 1) {
      PHX_CHECK(r.donor_status[v] == -1);
      continue;
    }
    ++receptors;
    const int32_t part = a.part_of_vertex(v);
    double best = std::numeric_limits<double>::infinity();
    bool any_active = false, any_fringe = false;
    for (int64_t cell = 0; cell < nc; ++cell) {
      if (a.part_of_cell(cell) == part || r.cell_status[cell] == 2 || r.ambiguous[cell]) continue;
      double ref[3];
      if (outside_by(a, cell, a.p(v), ref) <= -1e-9) {
        if (r.cell_status[cell] == 1) {
          any_fringe = true;
          continue;
        }
        any_active = true;
        const double volume = cell_volume(a, cell);
        if (volume > 0) best = std::min(best, volume);
      }
    }
    if (any_active || (fringe_donors && any_fringe)) PHX_CHECK(r.donor_status[v] == 0);
    if (r.donor_status[v] != 0) continue;
    const int64_t cell = r.donor_cell[v];
    PHX_CHECK(a.part_of_cell(cell) != part);
    PHX_CHECK(!r.ambiguous[cell]);
    PHX_CHECK(r.cell_status[cell] == 0 || (fringe_donors && !any_active && r.cell_status[cell] == 1));
    double ref[3];
    PHX_CHECK(outside_by(a, cell, a.p(v), ref) <= 1e-12);
    for (int k = 0; k < 3; ++k) PHX_CHECK_NEAR(ref[k], r.reference[3 * v + k], 1e-9);
    PHX_CHECK(r.residual[v] <= 1e-12);
    if (!a.hex(cell) && r.cell_status[cell] == 0 && std::isfinite(best)) {
      PHX_CHECK(r.measure[v] <= best * (1 + 1e-12));
    }
  }
  PHX_CHECK(r.stats[9] == static_cast<double>(receptors));
}

void test_pair(bool background_tets, bool shell_tets) {
  Assembly a;
  a.add(background(9, 2.0, background_tets));
  a.add(shell(5, 3, 0.5, 1.2, shell_tets));
  // Active donors only: every receptor without one is reported, never silent.
  Result strict;
  PHX_CHECK(a.run(1, strict) == PHX_MC_OK);
  check_assembly(a, strict, 0.5, 1e-6, false);
  PHX_CHECK(strict.stats[14] == 0.0);
  for (size_t v = 0; v < strict.donor_status.size(); ++v) {
    PHX_CHECK(strict.vertex_status[v] != 1 || strict.donor_status[v] == 0 ||
              strict.donor_status[v] == 2);
  }
  // Fringe donors second: no orphans.
  Result r;
  PHX_CHECK(a.run(1, r, 1, 1) == PHX_MC_OK);
  check_assembly(a, r, 0.5, 1e-6, true);
  PHX_CHECK(r.stats[10] == 0.0);  // no orphans
  PHX_CHECK(r.stats[13] == 0.0);  // no unproved ambiguous cells
  PHX_CHECK(r.stats[8] > 0.0);
  PHX_CHECK(r.vertex_status == strict.vertex_status);
  // prove = 0 constructs the same blanking and finds a donor for each receptor.
  Result unproved;
  PHX_CHECK(a.run(0, unproved, 1, 1) == PHX_MC_OK);
  PHX_CHECK(unproved.vertex_status == r.vertex_status);
  for (size_t v = 0; v < r.donor_status.size(); ++v) {
    PHX_CHECK(unproved.donor_status[v] == r.donor_status[v]);
  }
  // Two fringe layers grow the receptor set.
  Result deeper;
  PHX_CHECK(a.run(1, deeper, 2, 1) == PHX_MC_OK);
  check_assembly(a, deeper, 0.5, 1e-6, true);
  PHX_CHECK(deeper.stats[9] > r.stats[9]);
}

void test_orphans_are_reported() {
  // A shell that leaves the background: its outer receptors have no donor.
  Assembly a;
  a.add(background(4, 1.0, true));
  a.add(shell(3, 2, 0.5, 1.6, false));
  Result r;
  PHX_CHECK(a.run(1, r) == PHX_MC_OK);
  int64_t outside = 0;
  for (size_t v = 0; v < r.donor_status.size(); ++v) outside += r.donor_status[v] == 1;
  PHX_CHECK(outside > 0);
  PHX_CHECK(r.stats[10] >= static_cast<double>(outside));
}

void test_invalid_arguments() {
  Assembly a;
  a.add(background(2, 1.0, true));
  a.add(shell(2, 1, 0.3, 0.6, true));
  Result r;
  Assembly bad_role = a;
  bad_role.role[0] = 3;
  PHX_CHECK(bad_role.run(1, r) == PHX_MC_INVALID_INPUT);
  Assembly bad_index = a;
  bad_index.c[0] = static_cast<int32_t>(a.voff[1]);  // a vertex of the other part
  PHX_CHECK(bad_index.run(1, r) == PHX_MC_INVALID_INPUT);
  Assembly nonfinite = a;
  nonfinite.x[4] = std::numeric_limits<double>::quiet_NaN();
  PHX_CHECK(nonfinite.run(1, r) == PHX_MC_NONFINITE_INPUT);
  PHX_CHECK(phx_mc_overset_assemble(nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
}

}  // namespace

int main() {
  test_pair(true, false);
  test_pair(false, false);
  test_pair(true, true);
  test_pair(false, true);
  test_orphans_are_reported();
  test_invalid_arguments();
  return phx::mc::test::finish("test_overset");
}
