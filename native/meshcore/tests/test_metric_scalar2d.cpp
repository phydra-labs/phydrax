// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "metric_scalar2d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "dyadic.hpp"
#include "exact_scalar.hpp"
#include "metric_float.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "topology_keys.hpp"

namespace {

using phx::mc::Dyadic;
using phx::mc::orient2d;
using phx::mc::metric::Adapt2DBoxInput;
using phx::mc::metric::Adapt2DBoxOutcome;
using phx::mc::metric::Adapt2DBoxOutput;
using phx::mc::metric::adapt_metric_2d_box;

struct GridMesh2D {
  std::vector<double> points;
  std::vector<double> metrics;
  std::vector<std::uint64_t> vertex_ids;
  std::vector<std::int8_t> fixed;
  std::vector<std::int64_t> cells;
  std::vector<std::uint64_t> cell_ids;
  std::vector<std::int64_t> edges;
  std::vector<std::int64_t> edge_classes;
  std::vector<std::int8_t> protected_edges;
};

GridMesh2D make_square_grid(int n, auto metric_fn) {
  GridMesh2D g;
  const int side = n + 1;
  const std::size_t vcount = static_cast<std::size_t>(side * side);
  g.points.resize(2 * vcount);
  g.metrics.resize(vcount);
  g.vertex_ids.resize(vcount);
  g.fixed.assign(vcount, 0);
  for (int j = 0; j <= n; ++j) {
    for (int i = 0; i <= n; ++i) {
      const std::size_t row = static_cast<std::size_t>(j * side + i);
      const double x = static_cast<double>(i) / static_cast<double>(n);
      const double y = static_cast<double>(j) / static_cast<double>(n);
      g.points[2 * row] = x;
      g.points[2 * row + 1] = y;
      g.metrics[row] = metric_fn(x, y);
      g.vertex_ids[row] = 10 + row;
    }
  }
  const std::size_t ccount = static_cast<std::size_t>(2 * n * n);
  g.cells.reserve(3 * ccount);
  g.cell_ids.reserve(ccount);
  std::vector<std::uint64_t> edge_keys;
  edge_keys.reserve(3 * ccount);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const std::int64_t v00 = j * side + i;
      const std::int64_t v10 = v00 + 1;
      const std::int64_t v01 = v00 + side;
      const std::int64_t v11 = v01 + 1;
      const std::array<std::array<std::int64_t, 3>, 2> tris{{
          {v00, v10, v11},
          {v00, v11, v01},
      }};
      for (const auto& t : tris) {
        g.cell_ids.push_back(1000 + g.cell_ids.size());
        g.cells.insert(g.cells.end(), t.begin(), t.end());
        for (int k = 0; k < 3; ++k) {
          edge_keys.push_back(phx::mc::topology::undirected_edge(
              static_cast<std::int32_t>(t[k]), static_cast<std::int32_t>(t[(k + 1) % 3])));
        }
      }
    }
  }
  std::sort(edge_keys.begin(), edge_keys.end());
  for (std::size_t idx = 0; idx < edge_keys.size();) {
    std::size_t end = idx + 1;
    while (end < edge_keys.size() && edge_keys[end] == edge_keys[idx]) ++end;
    const std::int64_t a = static_cast<std::int64_t>(edge_keys[idx] >> 32);
    const std::int64_t b = static_cast<std::int64_t>(static_cast<std::uint32_t>(edge_keys[idx]));
    g.edges.push_back(a);
    g.edges.push_back(b);
    g.edge_classes.push_back((end - idx) == 1 ? 1 : 0);
    g.protected_edges.push_back(0);
    idx = end;
  }
  return g;
}

Adapt2DBoxInput as_input(const GridMesh2D& g) {
  return {
      g.points,
      g.metrics,
      g.vertex_ids,
      g.fixed,
      g.cells,
      g.cell_ids,
      g.edges,
      g.edge_classes,
      g.protected_edges,
      100'000,
      200'000,
      phx::mc::metric::kLowerLength,
  };
}

// Independent brute-force exact 2D triangle-triangle positive-area check (rounded + lifted).
bool brute_overlap(const GridMesh2D& g, const Adapt2DBoxOutput& out, std::size_t t, std::size_t s) {
  const auto tv0 = static_cast<std::size_t>(out.cells[3 * t]);
  const auto tv1 = static_cast<std::size_t>(out.cells[3 * t + 1]);
  const auto tv2 = static_cast<std::size_t>(out.cells[3 * t + 2]);
  const double* tp[3] = {out.points.data() + 2 * tv0, out.points.data() + 2 * tv1,
                         out.points.data() + 2 * tv2};
  const double* sp[3] = {
      g.points.data() + 2 * static_cast<std::size_t>(g.cells[3 * s]),
      g.points.data() + 2 * static_cast<std::size_t>(g.cells[3 * s + 1]),
      g.points.data() + 2 * static_cast<std::size_t>(g.cells[3 * s + 2]),
  };
  bool rounded = true;
  for (int e = 0; e < 3 && rounded; ++e) {
    if (orient2d(tp[e], tp[(e + 1) % 3], sp[0]) <= 0 &&
        orient2d(tp[e], tp[(e + 1) % 3], sp[1]) <= 0 &&
        orient2d(tp[e], tp[(e + 1) % 3], sp[2]) <= 0) {
      rounded = false;
    }
  }
  for (int e = 0; e < 3 && rounded; ++e) {
    if (orient2d(sp[e], sp[(e + 1) % 3], tp[0]) <= 0 &&
        orient2d(sp[e], sp[(e + 1) % 3], tp[1]) <= 0 &&
        orient2d(sp[e], sp[(e + 1) % 3], tp[2]) <= 0) {
      rounded = false;
    }
  }
  if (rounded) return true;

  auto lift = [&](std::size_t v) {
    Dyadic x, y;
    for (std::int64_t k = out.stencil_offsets[v]; k < out.stencil_offsets[v + 1]; ++k) {
      const auto sr = static_cast<std::size_t>(out.stencil_rows[k]);
      const Dyadic w(out.stencil_weights[k]);
      x = x + w * Dyadic(g.points[2 * sr]);
      y = y + w * Dyadic(g.points[2 * sr + 1]);
    }
    return std::array<Dyadic, 2>{x, y};
  };
  const std::array<std::array<Dyadic, 2>, 3> lt{lift(tv0), lift(tv1), lift(tv2)};
  const std::array<std::array<Dyadic, 2>, 3> ls{{
      {Dyadic(sp[0][0]), Dyadic(sp[0][1])},
      {Dyadic(sp[1][0]), Dyadic(sp[1][1])},
      {Dyadic(sp[2][0]), Dyadic(sp[2][1])},
  }};
  auto o_exact = [](const std::array<Dyadic, 2>& a, const std::array<Dyadic, 2>& b,
                    const std::array<Dyadic, 2>& c) {
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])).sign();
  };
  for (int e = 0; e < 3; ++e) {
    if (o_exact(lt[e], lt[(e + 1) % 3], ls[0]) <= 0 &&
        o_exact(lt[e], lt[(e + 1) % 3], ls[1]) <= 0 &&
        o_exact(lt[e], lt[(e + 1) % 3], ls[2]) <= 0) {
      return false;
    }
  }
  for (int e = 0; e < 3; ++e) {
    if (o_exact(ls[e], ls[(e + 1) % 3], lt[0]) <= 0 &&
        o_exact(ls[e], ls[(e + 1) % 3], lt[1]) <= 0 &&
        o_exact(ls[e], ls[(e + 1) % 3], lt[2]) <= 0) {
      return false;
    }
  }
  return true;
}

bool require(bool cond, const char* msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  // 1. Graded circle refinement on [0, 1]^2 (exact carriers, brute-force ancestry parity, determinism).
  auto circle_metric = [](double x, double y) {
    const double r = std::hypot(x - 0.5, y - 0.5);
    const double h = std::min(0.05 + 0.3 * std::abs(r - 0.3), 0.16);
    return 1.0 / (h * h);
  };
  const GridMesh2D graded = make_square_grid(8, circle_metric);
  Adapt2DBoxOutput out1, out2;
  const auto res1 = adapt_metric_2d_box(as_input(graded), out1);
  const auto res2 = adapt_metric_2d_box(as_input(graded), out2);
  if (!require(res1 == Adapt2DBoxOutcome::kAdapted, "graded circle must adapt")) return 1;
  if (!require(res2 == Adapt2DBoxOutcome::kAdapted, "second run must adapt")) return 1;
  if (!require(out1.points == out2.points && out1.cells == out2.cells &&
                   out1.stencil_weights == out2.stencil_weights &&
                   out1.ancestry_ids == out2.ancestry_ids,
               "graded 2D box adaptation must be deterministic")) {
    return 1;
  }
  if (!require(out1.measurements[0] >= phx::mc::metric::kLowerLength &&
                   out1.measurements[1] <= phx::mc::metric::kUpperLength &&
                   out1.measurements[2] == 1.0 && out1.measurements[3] >= 0.70,
               "graded measurements must meet strict band and quality")) {
    return 1;
  }

  // Verify every stencil reproduces its exact rounded binary64 point and sums to exact 1.
  const std::size_t vcount = out1.vertex_ids.size();
  for (std::size_t v = 0; v < vcount; ++v) {
    Dyadic wsum, lx, ly;
    for (std::int64_t k = out1.stencil_offsets[v]; k < out1.stencil_offsets[v + 1]; ++k) {
      const auto sr = static_cast<std::size_t>(out1.stencil_rows[k]);
      const Dyadic w(out1.stencil_weights[k]);
      wsum = wsum + w;
      lx = lx + w * Dyadic(graded.points[2 * sr]);
      ly = ly + w * Dyadic(graded.points[2 * sr + 1]);
    }
    if (!require((wsum - Dyadic(1.0)).sign() == 0, "stencil weights must sum to exact 1")) return 1;
    const double rx = phx::mc::exact::rounded_quotient(lx, Dyadic(1.0));
    const double ry = phx::mc::exact::rounded_quotient(ly, Dyadic(1.0));
    if (!require(rx == out1.points[2 * v] && ry == out1.points[2 * v + 1],
                 "target point must be exact canonical rounding of stencil")) {
      return 1;
    }
  }

  // Verify BFS cell ancestry against brute-force exact SAT for every target triangle.
  const std::size_t tcount = out1.cell_ids.size();
  const std::size_t scount = graded.cell_ids.size();
  for (std::size_t t = 0; t < tcount; ++t) {
    std::vector<std::uint64_t> expected;
    for (std::size_t s = 0; s < scount; ++s) {
      if (brute_overlap(graded, out1, t, s)) expected.push_back(graded.cell_ids[s]);
    }
    std::sort(expected.begin(), expected.end());
    std::vector<std::uint64_t> actual(
        out1.ancestry_ids.begin() + out1.ancestry_offsets[t],
        out1.ancestry_ids.begin() + out1.ancestry_offsets[t + 1]);
    if (!require(actual == expected, "BFS cell ancestry must match brute-force exact SAT")) {
      return 1;
    }
  }

  // 2. Coarsening test.
  const GridMesh2D fine = make_square_grid(10, [](double, double) { return 1.0 / (0.35 * 0.35); });
  Adapt2DBoxOutput coarse_out;
  if (!require(adapt_metric_2d_box(as_input(fine), coarse_out) == Adapt2DBoxOutcome::kAdapted,
               "coarsening must adapt")) {
    return 1;
  }
  if (!require(coarse_out.cell_ids.size() < fine.cell_ids.size(), "coarsening must reduce cells")) {
    return 1;
  }

  // 3. Already-unit no-op test.
  const GridMesh2D unit = make_square_grid(4, [](double, double) { return 1.0 / (0.28 * 0.28); });
  Adapt2DBoxOutput noop_out;
  if (!require(adapt_metric_2d_box(as_input(unit), noop_out) == Adapt2DBoxOutcome::kAlreadyUnit,
               "already-unit square must return kAlreadyUnit")) {
    return 1;
  }

  // 4. Refusal tests: protected edge, fixed interior vertex, non-box perturbed boundary.
  GridMesh2D prot = graded;
  prot.protected_edges[0] = 1;
  Adapt2DBoxOutput declined;
  if (!require(adapt_metric_2d_box(as_input(prot), declined) == Adapt2DBoxOutcome::kDeclined,
               "protected edge must decline")) {
    return 1;
  }
  GridMesh2D fixed_mid = graded;
  fixed_mid.fixed[40] = 1;
  if (!require(adapt_metric_2d_box(as_input(fixed_mid), declined) == Adapt2DBoxOutcome::kDeclined,
               "fixed interior vertex must decline")) {
    return 1;
  }
  GridMesh2D nonbox = graded;
  nonbox.points[2 * 4 + 1] = -0.05;  // bow bottom wall outward
  if (!require(adapt_metric_2d_box(as_input(nonbox), declined) == Adapt2DBoxOutcome::kDeclined,
               "non-box boundary must decline")) {
    return 1;
  }

  // 5. C ABI smoke test.
  phx_mc_metric_adapt2d_box_request req{};
  req.vertex_count = static_cast<int64_t>(graded.vertex_ids.size());
  req.points = graded.points.data();
  req.scalar_metrics = graded.metrics.data();
  req.vertex_ids = graded.vertex_ids.data();
  req.fixed_vertices = graded.fixed.data();
  req.cell_count = static_cast<int64_t>(graded.cell_ids.size());
  req.cells = graded.cells.data();
  req.cell_ids = graded.cell_ids.data();
  req.edge_count = static_cast<int64_t>(graded.edge_classes.size());
  req.edges = graded.edges.data();
  req.edge_classes = graded.edge_classes.data();
  req.protected_edges = graded.protected_edges.data();
  req.next_vertex_id = 100'000;
  req.next_cell_id = 200'000;
  req.lower = phx::mc::metric::kLowerLength;
  int32_t capi_outcome = -1;
  phx_mc_metric_adapt2d_box* handle = nullptr;
  if (!require(phx_mc_metric_adapt_2d_box(&req, &capi_outcome, &handle) == PHX_MC_OK &&
                   capi_outcome == 1 && handle != nullptr,
               "C ABI adapt_2d_box must succeed")) {
    return 1;
  }
  int64_t sizes[PHX_MC_METRIC_ADAPT2D_SIZES]{};
  if (!require(phx_mc_metric_adapt_2d_sizes(handle, sizes) == PHX_MC_OK &&
                   sizes[0] == static_cast<int64_t>(out1.vertex_ids.size()) &&
                   sizes[2] == static_cast<int64_t>(out1.cell_ids.size()),
               "C ABI sizes must match C++ output")) {
    phx_mc_metric_adapt_2d_free(handle);
    return 1;
  }
  phx_mc_metric_adapt_2d_free(handle);
  return 0;
}
