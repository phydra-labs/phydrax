// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "verify_chart_partition.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "capi_guard.hpp"
#include "expansion.hpp"
#include "floating_point_mode.hpp"
#include "predicates.hpp"
#include "verify_surface_embedding.hpp"

namespace {
using phx::mc::Expansion;
using Point = std::array<double, 2>;
struct Edge {
  int64_t patch;
  Point lo, hi;
  int64_t row;
  bool forward, trim;
};

Point point(const double* p) { return {p[0], p[1]}; }
bool bits_equal(double a, double b) {
  return std::bit_cast<uint64_t>(a) == std::bit_cast<uint64_t>(b);
}
bool point_less(const Point& a, const Point& b) {
  for (int k = 0; k < 2; ++k) {
    if (a[k] < b[k]) return true;
    if (a[k] > b[k]) return false;
    if (!bits_equal(a[k], b[k]))
      return std::bit_cast<uint64_t>(a[k]) < std::bit_cast<uint64_t>(b[k]);
  }
  return false;
}
bool point_equal(const Point& a, const Point& b) {
  return bits_equal(a[0], b[0]) && bits_equal(a[1], b[1]);
}
Edge edge(int64_t patch, Point a, Point b, int64_t row, bool trim) {
  bool forward = point_less(a, b);
  return {patch, forward ? a : b, forward ? b : a, row, forward, trim};
}
bool same_key(const Edge& a, const Edge& b) {
  return a.patch == b.patch && point_equal(a.lo, b.lo) && point_equal(a.hi, b.hi);
}
bool valid_uv(const double* uv, int64_t count) {
  for (int64_t i = 0; i < count; ++i)
    if (!std::isfinite(uv[i]) || !phx::mc::coordinate_in_domain(uv[i])) return false;
  return true;
}

void check_edges(std::vector<Edge>& edges, std::vector<phx_mc_chart_failure>& failures) {
  std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
    if (a.patch != b.patch) return a.patch < b.patch;
    if (!point_equal(a.lo, b.lo)) return point_less(a.lo, b.lo);
    if (!point_equal(a.hi, b.hi)) return point_less(a.hi, b.hi);
    if (a.trim != b.trim) return a.trim < b.trim;
    return a.row < b.row;
  });
  for (size_t i = 0; i < edges.size();) {
    size_t j = i + 1;
    while (j < edges.size() && same_key(edges[i], edges[j])) ++j;
    const Edge& a = edges[i];
    bool good = j - i == 2 && !a.trim &&
                (edges[i + 1].trim ? a.forward == edges[i + 1].forward
                                   : a.forward != edges[i + 1].forward);
    if (!good) failures.push_back({PHX_MC_CHART_EDGE, a.patch, a.trim ? -1 : a.row,
                                   j - i > 1 ? edges[i + 1].row : -1});
    i = j;
  }
}

}  // namespace

namespace phx::mc::chart {
void check_uses(int64_t use_count, const int64_t* use_patch, const double* use_range,
                int64_t chord_count, const int64_t* chord_use, const int64_t* chord_loop,
                const double* chord_uv, const double* chord_range,
                std::vector<phx_mc_chart_failure>& failures) {
  std::vector<int64_t> order(static_cast<size_t>(chord_count));
  std::iota(order.begin(), order.end(), int64_t{0});
  std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    if (chord_use[a] != chord_use[b]) return chord_use[a] < chord_use[b];
    const int64_t u = chord_use[a];
    const bool ascending = use_range[2*u] < use_range[2*u+1];
    if (chord_range[2*a] != chord_range[2*b])
      return ascending ? chord_range[2*a] < chord_range[2*b] : chord_range[2*a] > chord_range[2*b];
    return a < b;
  });
  size_t pos = 0;
  for (int64_t u = 0; u < use_count; ++u) {
    const bool ascending = use_range[2*u] < use_range[2*u+1];
    double next = use_range[2*u];
    int64_t bad = -1, previous = -1;
    bool any = false;
    while (pos < order.size() && chord_use[order[pos]] == u) {
      const int64_t c = order[pos++];
      double a = chord_range[2*c], b = chord_range[2*c+1];
      if (!bits_equal(a, next) || (ascending ? !(a < b) : !(a > b))) bad = c;
      // Consecutive chords of one use join bitwise in UV and stay in one loop.
      if (previous != -1 && (!point_equal(point(chord_uv + 4*previous + 2), point(chord_uv + 4*c)) ||
                             chord_loop[previous] != chord_loop[c])) bad = c;
      previous = c;
      next = b;
      any = true;
    }
    if (!any || bad != -1 || !bits_equal(next, use_range[2*u+1]))
      failures.push_back({PHX_MC_CHART_USE_COVERAGE, use_patch[u], u, bad});
  }
}
void check_loops(int64_t loop_count, int64_t chord_count, const int64_t* chord_loop,
                 const int64_t* chord_use, const int64_t* use_patch, const double* uv,
                 std::vector<phx_mc_chart_failure>& failures) {
  std::vector<int64_t> order(static_cast<size_t>(chord_count));
  std::iota(order.begin(), order.end(), int64_t{0});
  std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    if (chord_loop[a] != chord_loop[b]) return chord_loop[a] < chord_loop[b];
    Point pa = point(uv + 4*a), pb = point(uv + 4*b);
    if (!point_equal(pa, pb)) return point_less(pa, pb);
    return a < b;
  });
  std::vector<bool> visited(static_cast<size_t>(chord_count), false);
  size_t begin = 0;
  for (int64_t loop = 0; loop < loop_count; ++loop) {
    size_t end = begin;
    while (end < order.size() && chord_loop[order[end]] == loop) ++end;
    bool good = end > begin;
    int64_t patch = good ? use_patch[chord_use[order[begin]]] : -1;
    size_t current = begin;
    for (size_t step = begin; good && step < end; ++step) {
      int64_t c = order[current];
      if (visited[current] || use_patch[chord_use[c]] != patch) { good = false; break; }
      visited[current] = true;
      Point target = point(uv + 4*c + 2);
      auto it = std::lower_bound(order.begin() + begin, order.begin() + end, target,
          [&](int64_t row, const Point& p) { return point_less(point(uv + 4*row), p); });
      if (it == order.begin() + end || !point_equal(point(uv + 4*(*it)), target)) {
        good = false; break;
      }
      current = static_cast<size_t>(it - order.begin());
    }
    if (good && current != begin) good = false;
    if (!good) failures.push_back({PHX_MC_CHART_LOOP, patch, loop, -1});
    begin = end;
  }
}
}  // namespace phx::mc::chart

namespace {
using phx::mc::chart::check_loops;
using phx::mc::chart::check_uses;

// Orientation agreement plus boundary = chord chain fixes the covering multiplicity
// to the winding number of the chord cycles; it does not exclude multiplicity 2.
// Pairwise interiors must also be disjoint: the patch's correctly oriented triangles are
// lifted to (u, v, 0) with one id per bitwise-distinct UV point and classified by the
// surface embedding verifier's exact contact enumeration (one owner of the overlap test).
int32_t check_overlaps(int64_t triangle_count, const int32_t* triangle_patch,
                       const double* triangle_uv, const std::vector<uint8_t>& oriented,
                       std::vector<phx_mc_chart_failure>& failures) {
  struct Corner {
    int32_t patch;
    Point uv;
    int64_t slot;
  };
  std::vector<Corner> corners;
  for (int64_t t = 0; t < triangle_count; ++t)
    if (oriented[t])
      for (int k = 0; k < 3; ++k)
        corners.push_back({triangle_patch[t], point(triangle_uv + 6*t + 2*k), 3*t + k});
  std::sort(corners.begin(), corners.end(), [](const Corner& a, const Corner& b) {
    if (a.patch != b.patch) return a.patch < b.patch;
    if (!point_equal(a.uv, b.uv)) return point_less(a.uv, b.uv);
    return a.slot < b.slot;
  });
  std::vector<int64_t> by_patch;
  for (int64_t t = 0; t < triangle_count; ++t) if (oriented[t]) by_patch.push_back(t);
  std::stable_sort(by_patch.begin(), by_patch.end(),
                   [&](int64_t a, int64_t b) { return triangle_patch[a] < triangle_patch[b]; });
  std::vector<int64_t> local_id(static_cast<size_t>(3*triangle_count), -1);
  size_t next_triangle = 0;
  for (size_t begin = 0; begin < corners.size();) {
    const int32_t patch = corners[begin].patch;
    size_t end = begin;
    while (end < corners.size() && corners[end].patch == patch) ++end;
    std::vector<double> points;
    for (size_t i = begin; i < end; ++i) {
      if (i == begin || !point_equal(corners[i].uv, corners[i-1].uv)) {
        points.push_back(corners[i].uv[0]);
        points.push_back(corners[i].uv[1]);
        points.push_back(0.0);
      }
      local_id[corners[i].slot] = static_cast<int64_t>(points.size() / 3) - 1;
    }
    std::vector<int64_t> global, triangles;
    for (; next_triangle < by_patch.size() && triangle_patch[by_patch[next_triangle]] == patch;
         ++next_triangle) {
      const int64_t t = by_patch[next_triangle];
      global.push_back(t);
      for (int k = 0; k < 3; ++k) triangles.push_back(local_id[3*t + k]);
    }
    std::vector<phx::mc::SurfaceContact> contacts;
    const int32_t status = phx::mc::surface_embedding_contacts(
        points.data(), static_cast<int64_t>(global.size()), triangles.data(), contacts);
    if (status != PHX_MC_OK) return status;
    for (const phx::mc::SurfaceContact& c : contacts)
      failures.push_back({PHX_MC_CHART_OVERLAP, patch, global[c.first], global[c.second]});
    begin = end;
  }
  return PHX_MC_OK;
}

}  // namespace

extern "C" int32_t phx_mc_verify_chart_partition(
    int64_t patch_count, int64_t triangle_count, const int32_t* triangle_patch,
    const double* triangle_uv, int64_t use_count, const int64_t* use_patch,
    const double* use_range, int64_t loop_count, int64_t chord_count,
    const int64_t* chord_use, const int64_t* chord_loop, const double* chord_uv,
    const double* chord_range, int64_t evidence_capacity,
    phx_mc_chart_failure* evidence, int64_t* evidence_count) {
  return phx::mc::guarded([&]() -> int32_t {
    using phx::mc::addressable;
    if (patch_count <= 0 || triangle_count < 0 || use_count < 0 || loop_count < 0 || chord_count < 0 ||
        evidence_capacity < 0 || !evidence_count || (evidence_capacity && !evidence) ||
        (triangle_count && (!triangle_patch || !triangle_uv)) ||
        (use_count && (!use_patch || !use_range)) ||
        (chord_count && (!chord_use || !chord_loop || !chord_uv || !chord_range)) ||
        !addressable(patch_count, 1, sizeof(Expansion)) ||
        !addressable(triangle_count, 6, sizeof(double)) ||
        !addressable(use_count, 2, sizeof(double)) ||
        !addressable(chord_count, 4, sizeof(double)) ||
        !addressable(evidence_capacity, 1, sizeof(phx_mc_chart_failure)) ||
        !addressable(triangle_count, 3, sizeof(Edge)) ||
        !addressable(chord_count, 1, sizeof(Edge))) return PHX_MC_INVALID_ARGUMENT;
    if (!valid_uv(triangle_uv, triangle_count*6) || !valid_uv(chord_uv, chord_count*4))
      return PHX_MC_INVALID_INPUT;
    for (int64_t i = 0; i < triangle_count; ++i)
      if (triangle_patch[i] < 0 || triangle_patch[i] >= patch_count) return PHX_MC_INVALID_INPUT;
    for (int64_t i = 0; i < use_count; ++i)
      if (use_patch[i] < 0 || use_patch[i] >= patch_count ||
          !std::isfinite(use_range[2*i]) || !std::isfinite(use_range[2*i+1]) ||
          use_range[2*i] == use_range[2*i+1]) return PHX_MC_INVALID_INPUT;
    for (int64_t i = 0; i < chord_count; ++i)
      if (chord_use[i] < 0 || chord_use[i] >= use_count ||
          chord_loop[i] < 0 || chord_loop[i] >= loop_count ||
          !std::isfinite(chord_range[2*i]) || !std::isfinite(chord_range[2*i+1]))
        return PHX_MC_INVALID_INPUT;
    phx::mc::GradualNearestScope scope;
    if (!scope.established()) return PHX_MC_INTERNAL_ERROR;
    std::vector<Expansion> area(static_cast<size_t>(patch_count));
    std::vector<Edge> edges;
    std::vector<phx_mc_chart_failure> failures;
    for (int64_t i = 0; i < chord_count; ++i) {
      Point a = point(chord_uv + 4*i), b = point(chord_uv + 4*i + 2);
      int64_t patch = use_patch[chord_use[i]];
      area[patch] = area[patch] + Expansion::product(a[0], b[1]) - Expansion::product(a[1], b[0]);
      edges.push_back(edge(patch, a, b, i, true));
    }
    for (int64_t p = 0; p < patch_count; ++p)
      if (area[p].sign() == 0) failures.push_back({PHX_MC_CHART_ORIENTATION, p, -1, -1});
    std::vector<uint8_t> oriented(static_cast<size_t>(triangle_count), 0);
    for (int64_t i = 0; i < triangle_count; ++i) {
      const double* uv = triangle_uv + 6*i;
      int64_t patch = triangle_patch[i];
      if (phx::mc::orient2d(uv, uv+2, uv+4) != area[patch].sign() || area[patch].sign() == 0)
        failures.push_back({PHX_MC_CHART_ORIENTATION, patch, i, -1});
      else
        oriented[i] = 1;
      for (int k = 0; k < 3; ++k)
        edges.push_back(edge(patch, point(uv+2*k), point(uv+2*((k+1)%3)), i, false));
    }
    check_edges(edges, failures);
    check_uses(use_count, use_patch, use_range, chord_count, chord_use, chord_loop, chord_uv,
               chord_range, failures);
    check_loops(loop_count, chord_count, chord_loop, chord_use, use_patch, chord_uv, failures);
    const int32_t overlap = check_overlaps(triangle_count, triangle_patch, triangle_uv,
                                           oriented, failures);
    if (overlap != PHX_MC_OK) return overlap;
    *evidence_count = static_cast<int64_t>(failures.size());
    if (failures.size() > static_cast<size_t>(evidence_capacity)) return PHX_MC_CAPACITY_EXCEEDED;
    if (!failures.empty()) std::copy(failures.begin(), failures.end(), evidence);
    return PHX_MC_OK;
  });
}
