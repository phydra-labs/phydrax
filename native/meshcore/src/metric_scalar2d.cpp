// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "metric_scalar2d.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include "capi_guard.hpp"
#include "deterministic_float.hpp"
#include "dyadic.hpp"
#include "exact_scalar.hpp"
#include "expansion.hpp"
#include "floating_point_mode.hpp"
#include "metric_float.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "topology_keys.hpp"

namespace phx::mc::metric {
namespace {

using Id = std::uint64_t;
using Row = std::int32_t;
using EdgeKey = std::uint64_t;

constexpr Id kLastId = std::numeric_limits<std::int64_t>::max();
constexpr auto kSwappedFrom = static_cast<std::int32_t>(AdaptLineage::kSwappedFrom);
constexpr int kCarrierBits = 52;
constexpr std::uint64_t kCarrierScale = std::uint64_t{1} << kCarrierBits;
constexpr double kCarrierScaleDbl = static_cast<double>(kCarrierScale);
constexpr double kCarrierInvScale = 0x1p-52;
constexpr double kSqrt3 = 1.732050807568877293527;
constexpr double kBandSafetyMargin = 1.e-11;

bool fits_ids(Id first, std::size_t count) {
  return count == 0 || (first <= kLastId && count - 1 <= kLastId - first);
}

// Fast exact binary64 orient2d using an inline Shewchuk error filter,
// falling back to phx::mc::orient2d only when near collinear.
inline int orient2d_fast(const double* a, const double* b, const double* c) {
  const double detleft = (a[0] - c[0]) * (b[1] - c[1]);
  const double detright = (a[1] - c[1]) * (b[0] - c[0]);
  const double det = detleft - detright;
  const double detsum = std::abs(detleft) + std::abs(detright);
  const double errbound = 3.3306690738754716e-16 * detsum;
  if (det > errbound) return 1;
  if (det < -errbound) return -1;
  return orient2d(a, b, c);
}

struct PointRow2D {
  Point2 point;
  std::size_t row;
};

struct WallSegment {
  double low;
  double high;
  std::size_t a;  // source/target vertex row at `low`
  std::size_t b;  // source/target vertex row at `high`
  std::array<Id, 2> key;
};

struct SourceBox2D {
  Point2 low{};
  Point2 high{};
  std::vector<PointRow2D> point_rows;
  std::array<std::vector<WallSegment>, 4> walls;
  std::vector<std::int32_t> vertex_cell;  // one incident source cell per source vertex
  std::vector<std::int32_t> neighbors;    // 3 * cell_count; slot k opposite vertex k

  const PointRow2D* find_point(const Point2& p) const {
    const auto it = std::lower_bound(
        point_rows.begin(), point_rows.end(), p,
        [](const PointRow2D& entry, const Point2& target) { return entry.point < target; });
    return it != point_rows.end() && it->point == p ? &*it : nullptr;
  }

  int wall_of_edge(const Point2& a, const Point2& b) const {
    if (a[0] == low[0] && b[0] == low[0]) return 0;
    if (a[0] == high[0] && b[0] == high[0]) return 1;
    if (a[1] == low[1] && b[1] == low[1]) return 2;
    if (a[1] == high[1] && b[1] == high[1]) return 3;
    return -1;
  }

  // Exact CCW inward boundary orientation for a positively oriented triangle edge a -> b.
  bool ccw_wall_edge(int wall, const Point2& a, const Point2& b) const {
    switch (wall) {
      case 0: return a[1] > b[1];  // left wall x == low[0], interior right -> goes down
      case 1: return a[1] < b[1];  // right wall x == high[0], interior left -> goes up
      case 2: return a[0] < b[0];  // bottom wall y == low[1], interior above -> goes right
      case 3: return a[0] > b[0];  // top wall y == high[1], interior below -> goes left
      default: return false;
    }
  }

  bool is_corner(const Point2& p) const {
    return (p[0] == low[0] || p[0] == high[0]) && (p[1] == low[1] || p[1] == high[1]);
  }
};

// Exact 2D log-mean metric length matching phydrax.meshing._metric.metric_edge_lengths.
double metric_edge_length_2d(const Point2& a, double ma, const Point2& b, double mb) {
  const double dx = b[0] - a[0], dy = b[1] - a[1];
  const double d2 = dx * dx + dy * dy;
  const double first = std::sqrt(d2 * ma);
  const double second = std::sqrt(d2 * mb);
  if (!(first > 0.0) || !(second > 0.0) || !std::isfinite(first) || !std::isfinite(second)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double ratio = second / first;
  if (std::abs(ratio - 1.0) <= 1.0e-6) return 0.5 * (first + second);
  return (first - second) / deterministic::log(first / second);
}

// Heron metric quality matching phydrax.meshing._local_metric._quality.
double heron_quality(double l0, double l1, double l2) {
  std::array<double, 3> s{l0, l1, l2};
  std::sort(s.begin(), s.end());
  const double a = s[0], b = s[1], c = s[2];
  const double prod = (a + b + c) * (b + c - a) * (a + c - b) * (a + b - c);
  const double denom = a * a + b * b + c * c;
  if (!(denom > 0.0) || !std::isfinite(denom)) return 0.0;
  return kSqrt3 * std::sqrt(std::max(prod, 0.0)) / denom;
}

// Validate that the source triangle mesh is an unprotected, non-fixed-interior,
// single-region Cartesian rectangle with positive orient2d and exact CCW wall coverage.
// By Green's theorem, strictly positive triangles with canceling opposite interior
// half-edges and a contiguous 1-fold CCW boundary partition of all 4 walls prove
// exact coverage of the rectangle without overlaps or holes.
bool bind_source_box_2d(const Adapt2DBoxInput& input, SourceBox2D& box,
                        std::vector<double>& log_rates) {
  const std::size_t vcount = input.vertex_ids.size();
  const std::size_t ccount = input.cell_ids.size();
  const std::size_t ecount = input.edge_classes.size();
  if (vcount < 4 || ccount < 2 || ccount > kMaxScalar2DSourceCells || ecount < 4 ||
      vcount >= static_cast<std::size_t>(std::numeric_limits<Row>::max()) ||
      input.points.size() != 2 * vcount || input.scalar_metrics.size() != vcount ||
      input.fixed_vertices.size() != vcount || input.cells.size() != 3 * ccount ||
      input.edges.size() != 2 * ecount || input.protected_edges.size() != ecount) {
    return false;
  }
  for (std::int8_t prot : input.protected_edges) {
    if (prot != 0) return false;
  }

  Id max_vid = 0;
  box.low = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
  box.high = {-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
  box.point_rows.reserve(vcount);
  log_rates.resize(vcount);
  for (std::size_t v = 0; v < vcount; ++v) {
    const double x = input.points[2 * v], y = input.points[2 * v + 1];
    const double m = input.scalar_metrics[v];
    if (!coordinate_in_domain(x) || !coordinate_in_domain(y) || !(m > 0.0) || !std::isfinite(m)) {
      return false;
    }
    const double lm = deterministic::log(m);
    if (!std::isfinite(lm)) return false;
    log_rates[v] = 0.5 * lm;
    box.low[0] = std::min(box.low[0], x);
    box.low[1] = std::min(box.low[1], y);
    box.high[0] = std::max(box.high[0], x);
    box.high[1] = std::max(box.high[1], y);
    box.point_rows.push_back({{x, y}, v});
    max_vid = std::max(max_vid, input.vertex_ids[v]);
  }
  if (!(box.low[0] < box.high[0] && box.low[1] < box.high[1])) return false;
  if (input.next_vertex_id <= max_vid) return false;
  Id max_cid = 0;
  for (Id cid : input.cell_ids) max_cid = std::max(max_cid, cid);
  if (input.next_cell_id <= max_cid) return false;

  for (std::size_t v = 0; v < vcount; ++v) {
    if (input.fixed_vertices[v] != 0 &&
        !box.is_corner({input.points[2 * v], input.points[2 * v + 1]})) {
      return false;
    }
  }
  std::sort(box.point_rows.begin(), box.point_rows.end(),
            [](const PointRow2D& a, const PointRow2D& b) { return a.point < b.point; });
  if (std::adjacent_find(box.point_rows.begin(), box.point_rows.end(),
                         [](const PointRow2D& a, const PointRow2D& b) {
                           return a.point == b.point;
                         }) != box.point_rows.end()) {
    return false;
  }

  // O(V + C) linear vertex-bucketed half-edge incidence + exact orient2d check.
  struct HalfEdge {
    Row hi;
    Row cell;
    std::uint8_t slot;
    std::int8_t dir;
  };
  std::vector<std::uint32_t> degree(vcount + 1, 0);
  box.vertex_cell.assign(vcount, -1);
  for (std::size_t c = 0; c < ccount; ++c) {
    const std::int64_t r0 = input.cells[3 * c];
    const std::int64_t r1 = input.cells[3 * c + 1];
    const std::int64_t r2 = input.cells[3 * c + 2];
    if (r0 < 0 || r1 < 0 || r2 < 0 || static_cast<std::size_t>(r0) >= vcount ||
        static_cast<std::size_t>(r1) >= vcount || static_cast<std::size_t>(r2) >= vcount ||
        r0 == r1 || r1 == r2 || r2 == r0) {
      return false;
    }
    const double* p0 = input.points.data() + 2 * static_cast<std::size_t>(r0);
    const double* p1 = input.points.data() + 2 * static_cast<std::size_t>(r1);
    const double* p2 = input.points.data() + 2 * static_cast<std::size_t>(r2);
    if (orient2d_fast(p0, p1, p2) <= 0) return false;
    if (box.vertex_cell[r0] < 0) box.vertex_cell[r0] = static_cast<Row>(c);
    if (box.vertex_cell[r1] < 0) box.vertex_cell[r1] = static_cast<Row>(c);
    if (box.vertex_cell[r2] < 0) box.vertex_cell[r2] = static_cast<Row>(c);
    ++degree[static_cast<std::size_t>(std::min(r1, r2)) + 1];
    ++degree[static_cast<std::size_t>(std::min(r2, r0)) + 1];
    ++degree[static_cast<std::size_t>(std::min(r0, r1)) + 1];
  }
  for (std::size_t v = 1; v <= vcount; ++v) degree[v] += degree[v - 1];
  std::vector<HalfEdge> half_edges(degree[vcount]);
  std::vector<std::uint32_t> cursor(degree.begin(), degree.end() - 1);
  for (std::size_t c = 0; c < ccount; ++c) {
    const Row rv[3] = {static_cast<Row>(input.cells[3 * c]),
                       static_cast<Row>(input.cells[3 * c + 1]),
                       static_cast<Row>(input.cells[3 * c + 2])};
    for (int slot = 0; slot < 3; ++slot) {
      const Row a = rv[(slot + 1) % 3], b = rv[(slot + 2) % 3];
      const Row lo = std::min(a, b), hi = std::max(a, b);
      half_edges[cursor[static_cast<std::size_t>(lo)]++] = {
          hi, static_cast<Row>(c), static_cast<std::uint8_t>(slot),
          static_cast<std::int8_t>(a < b ? 1 : -1)};
    }
  }

  // Bucket input edge_classes by min vertex row in O(V + E).
  struct EdgeClassEntry {
    Row hi;
    std::int64_t cls;
  };
  std::vector<std::uint32_t> edeg(vcount + 1, 0);
  for (std::size_t e = 0; e < ecount; ++e) {
    const std::int64_t a = input.edges[2 * e], b = input.edges[2 * e + 1];
    if (a < 0 || b < 0 || static_cast<std::size_t>(a) >= vcount ||
        static_cast<std::size_t>(b) >= vcount || a == b) {
      return false;
    }
    ++edeg[static_cast<std::size_t>(std::min(a, b)) + 1];
  }
  for (std::size_t v = 1; v <= vcount; ++v) edeg[v] += edeg[v - 1];
  std::vector<EdgeClassEntry> eclasses(edeg[vcount]);
  std::vector<std::uint32_t> ecur(edeg.begin(), edeg.end() - 1);
  for (std::size_t e = 0; e < ecount; ++e) {
    const Row a = static_cast<Row>(input.edges[2 * e]);
    const Row b = static_cast<Row>(input.edges[2 * e + 1]);
    const Row lo = std::min(a, b), hi = std::max(a, b);
    eclasses[ecur[static_cast<std::size_t>(lo)]++] = {hi, input.edge_classes[e]};
  }

  box.neighbors.assign(3 * ccount, -1);
  std::array<std::int64_t, 4> wall_classes{-1, -1, -1, -1};
  std::size_t seen_edges = 0;
  for (std::size_t lo = 0; lo < vcount; ++lo) {
    const std::uint32_t begin = degree[lo];
    const std::uint32_t end = degree[lo + 1];
    if (begin == end) continue;
    std::sort(half_edges.begin() + begin, half_edges.begin() + end,
              [](const HalfEdge& a, const HalfEdge& b) { return a.hi < b.hi; });
    const std::uint32_t ebegin = edeg[lo], eend = edeg[lo + 1];
    auto lookup_class = [&](Row hi) -> std::int64_t {
      for (std::uint32_t k = ebegin; k < eend; ++k) {
        if (eclasses[k].hi == hi) return eclasses[k].cls;
      }
      return -1;
    };
    std::uint32_t i = begin;
    while (i < end) {
      std::uint32_t j = i + 1;
      while (j < end && half_edges[j].hi == half_edges[i].hi) ++j;
      const std::uint32_t mult = j - i;
      const Row hi = half_edges[i].hi;
      const std::int64_t cls = lookup_class(hi);
      if (cls < 0) return false;
      ++seen_edges;
      if (mult == 2) {
        if (cls != 0 || half_edges[i].dir == half_edges[i + 1].dir) return false;
        box.neighbors[3 * static_cast<std::size_t>(half_edges[i].cell) + half_edges[i].slot] =
            half_edges[i + 1].cell;
        box.neighbors[3 * static_cast<std::size_t>(half_edges[i + 1].cell) +
                      half_edges[i + 1].slot] = half_edges[i].cell;
      } else if (mult == 1) {
        if (cls <= 0) return false;
        const std::size_t a = (half_edges[i].dir > 0) ? lo : static_cast<std::size_t>(hi);
        const std::size_t b = (half_edges[i].dir > 0) ? static_cast<std::size_t>(hi) : lo;
        const Point2 pa{input.points[2 * a], input.points[2 * a + 1]};
        const Point2 pb{input.points[2 * b], input.points[2 * b + 1]};
        const int w = box.wall_of_edge(pa, pb);
        if (w < 0 || !box.ccw_wall_edge(w, pa, pb)) return false;
        if (wall_classes[w] < 0) wall_classes[w] = cls;
        else if (wall_classes[w] != cls) return false;
        const int free_axis = 1 - (w / 2);
        std::size_t lo_v = a, hi_v = b;
        double lo_c = pa[free_axis], hi_c = pb[free_axis];
        if (lo_c > hi_c) {
          std::swap(lo_v, hi_v);
          std::swap(lo_c, hi_c);
        }
        if (!(lo_c < hi_c)) return false;
        Id id_a = input.vertex_ids[lo_v], id_b = input.vertex_ids[hi_v];
        if (id_a > id_b) std::swap(id_a, id_b);
        box.walls[w].push_back({lo_c, hi_c, lo_v, hi_v, {id_a, id_b}});
      } else {
        return false;
      }
      i = j;
    }
  }
  if (seen_edges != ecount) return false;

  // Verify exact 1-fold contiguous partition of all 4 rectangle walls.
  for (int w = 0; w < 4; ++w) {
    const int free_axis = 1 - (w / 2);
    if (box.walls[w].empty()) return false;
    std::sort(box.walls[w].begin(), box.walls[w].end(),
              [](const WallSegment& a, const WallSegment& b) { return a.low < b.low; });
    if (box.walls[w].front().low != box.low[free_axis] ||
        box.walls[w].back().high != box.high[free_axis]) {
      return false;
    }
    for (std::size_t k = 1; k < box.walls[w].size(); ++k) {
      if (box.walls[w][k - 1].high != box.walls[w][k].low) return false;
    }
  }
  return true;
}

bool check_already_unit(const Adapt2DBoxInput& input, double lower, Adapt2DBoxOutput& output) {
  const double low_safe = lower + kBandSafetyMargin;
  const double high_safe = kUpperLength - kBandSafetyMargin;
  const std::size_t ecount = input.edges.size() / 2;
  double min_len = std::numeric_limits<double>::infinity();
  double max_len = 0.0;
  for (std::size_t e = 0; e < ecount; ++e) {
    const auto a = static_cast<std::size_t>(input.edges[2 * e]);
    const auto b = static_cast<std::size_t>(input.edges[2 * e + 1]);
    const double len = metric_edge_length_2d(
        {input.points[2 * a], input.points[2 * a + 1]}, input.scalar_metrics[a],
        {input.points[2 * b], input.points[2 * b + 1]}, input.scalar_metrics[b]);
    if (!(len >= low_safe && len <= high_safe)) return false;
    min_len = std::min(min_len, len);
    max_len = std::max(max_len, len);
  }
  const std::size_t ccount = input.cells.size() / 3;
  double min_q = std::numeric_limits<double>::infinity();
  for (std::size_t c = 0; c < ccount; ++c) {
    const auto r0 = static_cast<std::size_t>(input.cells[3 * c]);
    const auto r1 = static_cast<std::size_t>(input.cells[3 * c + 1]);
    const auto r2 = static_cast<std::size_t>(input.cells[3 * c + 2]);
    const Point2 p0{input.points[2 * r0], input.points[2 * r0 + 1]};
    const Point2 p1{input.points[2 * r1], input.points[2 * r1 + 1]};
    const Point2 p2{input.points[2 * r2], input.points[2 * r2 + 1]};
    const double l01 = metric_edge_length_2d(p0, input.scalar_metrics[r0], p1, input.scalar_metrics[r1]);
    const double l12 = metric_edge_length_2d(p1, input.scalar_metrics[r1], p2, input.scalar_metrics[r2]);
    const double l20 = metric_edge_length_2d(p2, input.scalar_metrics[r2], p0, input.scalar_metrics[r0]);
    const double q = heron_quality(l01, l12, l20);
    if (!std::isfinite(q)) return false;
    min_q = std::min(min_q, q);
  }
  output = {};
  output.measured_edges = static_cast<std::int64_t>(ecount);
  output.measurements = {min_len, max_len, 1.0, min_q};
  return true;
}

// Exact IEEE-754 round-to-nearest-even of sum_{i=0..k-1} w[i] * x[i] (k <= 3)
// using stack-only Shewchuk expansion arithmetic; falls back to Dyadic only on
// the rare exact half-ULP tie or subnormal case.
double exact_rounded_dot(const double* w, const double* x, int k) {
  double e[8];
  int elen = 0;
  for (int i = 0; i < k; ++i) {
    double hi, lo;
    two_product(w[i], x[i], hi, lo);
    double term[2];
    int tlen = 0;
    if (lo != 0.0) term[tlen++] = lo;
    if (hi != 0.0 || tlen == 0) term[tlen++] = hi;
    if (elen == 0) {
      for (int j = 0; j < tlen; ++j) e[j] = term[j];
      elen = tlen;
    } else {
      double tmp[8];
      elen = fast_expansion_sum_zeroelim(elen, e, tlen, term, tmp);
      for (int j = 0; j < elen; ++j) e[j] = tmp[j];
    }
  }
  if (elen == 0) return 0.0;
  if (elen == 1) return e[0];

  // Compress expansion on stack using unconditional two_sum so h[top-1] is top, h[top-2] is next.
  double q = e[elen - 1];
  double h[8];
  int bottom = elen - 1;
  for (int i = elen - 2; i >= 0; --i) {
    double qnew, hh;
    two_sum(q, e[i], qnew, hh);
    if (hh != 0.0) {
      h[bottom--] = qnew;
      q = hh;
    } else {
      q = qnew;
    }
  }
  int top = 0;
  for (int i = bottom + 1; i < elen; ++i) {
    double qnew, hh;
    two_sum(h[i], q, qnew, hh);
    if (hh != 0.0) h[top++] = hh;
    q = qnew;
  }
  h[top++] = q;
  if (top == 1) return h[0];

  const double hi = h[top - 1];
  const double lo = h[top - 2];
  double tail_bound = 0.0;
  for (int i = 0; i < top - 2; ++i) tail_bound += std::abs(h[i]);
  double res, err;
  two_sum(hi, lo, res, err);
  const double abs_res = std::abs(res);
  if (std::isfinite(res) && abs_res >= 0x1p-900) {
    const double ulp_up = std::nextafter(abs_res, std::numeric_limits<double>::infinity()) - abs_res;
    // Distance to nearest midpoint is at least 0.25 * ulp_up (even at powers of 2).
    if (std::abs(err) + tail_bound < 0.24 * ulp_up) {
      return res;
    }
  }
  // Exact tie or extreme range: exact Dyadic fallback.
  Dyadic exact_sum;
  for (int i = 0; i < k; ++i) exact_sum = exact_sum + Dyadic(w[i]) * Dyadic(x[i]);
  return exact::nearest_quotient(exact_sum, Dyadic(1.0), std::isfinite(res) ? res : 0.0);
}

struct BoundVertex2D {
  Point2 point;
  Id target_id = 0;
  double scalar_metric = 0.0;
  std::array<Row, 3> rows{-1, -1, -1};
  std::array<double, 3> weights{0.0, 0.0, 0.0};
  int count = 0;
  Row owner_cell = -1;
};

bool finalize_stencil_vertex(const Adapt2DBoxInput& input, const std::vector<double>& log_rates,
                             std::array<Row, 3> rows, std::array<std::uint64_t, 3> iweights,
                             int nterms, int fixed_axis, double fixed_val, Row owner_cell,
                             BoundVertex2D& out) {
  std::array<std::pair<Row, std::uint64_t>, 3> terms;
  int k = 0;
  std::uint64_t sum = 0;
  for (int i = 0; i < nterms; ++i) {
    if (iweights[i] == 0) continue;
    terms[k++] = {rows[i], iweights[i]};
    sum += iweights[i];
  }
  if (k == 0 || sum != kCarrierScale) return false;
  if (k == 2 && input.vertex_ids[terms[0].first] > input.vertex_ids[terms[1].first]) {
    std::swap(terms[0], terms[1]);
  } else if (k == 3) {
    if (input.vertex_ids[terms[0].first] > input.vertex_ids[terms[1].first]) {
      std::swap(terms[0], terms[1]);
    }
    if (input.vertex_ids[terms[1].first] > input.vertex_ids[terms[2].first]) {
      std::swap(terms[1], terms[2]);
    }
    if (input.vertex_ids[terms[0].first] > input.vertex_ids[terms[1].first]) {
      std::swap(terms[0], terms[1]);
    }
  }

  double w_arr[3], x_arr[3], y_arr[3];
  const double m0 = input.scalar_metrics[terms[0].first];
  bool constant_m = true;
  double min_m = m0, max_m = m0;
  double log_m = 0.0;
  out.count = k;
  out.owner_cell = owner_cell;
  for (int i = 0; i < k; ++i) {
    const Row r = terms[i].first;
    const double w = static_cast<double>(terms[i].second) * kCarrierInvScale;
    out.rows[i] = r;
    out.weights[i] = w;
    w_arr[i] = w;
    x_arr[i] = input.points[2 * static_cast<std::size_t>(r)];
    y_arr[i] = input.points[2 * static_cast<std::size_t>(r) + 1];
    const double mr = input.scalar_metrics[r];
    if (mr != m0) constant_m = false;
    min_m = std::min(min_m, mr);
    max_m = std::max(max_m, mr);
    log_m += w * (2.0 * log_rates[r]);
  }
  out.point[0] = (fixed_axis == 0) ? fixed_val : exact_rounded_dot(w_arr, x_arr, k);
  out.point[1] = (fixed_axis == 1) ? fixed_val : exact_rounded_dot(w_arr, y_arr, k);
  if (!coordinate_in_domain(out.point[0]) || !coordinate_in_domain(out.point[1])) return false;
  out.scalar_metric = constant_m ? m0 : std::clamp(deterministic::exp(log_m), min_m, max_m);
  return out.scalar_metric > 0.0 && std::isfinite(out.scalar_metric);
}

bool bind_box_point_2d(const Adapt2DBoxInput& input, const SourceBox2D& box,
                       const ScalarField2D& field, const std::vector<double>& log_rates,
                       const Point2& goal, Row hint_cell, BoundVertex2D& out) {
  if (const auto* match = box.find_point(goal)) {
    const auto r = static_cast<Row>(match->row);
    out.point = goal;
    out.target_id = input.vertex_ids[r];
    out.scalar_metric = input.scalar_metrics[r];
    out.rows = {r, -1, -1};
    out.weights = {1.0, 0.0, 0.0};
    out.count = 1;
    out.owner_cell = box.vertex_cell[r];
    return out.owner_cell >= 0;
  }

  int wall = -1;
  if (goal[0] == box.low[0]) wall = 0;
  else if (goal[0] == box.high[0]) wall = 1;
  else if (goal[1] == box.low[1]) wall = 2;
  else if (goal[1] == box.high[1]) wall = 3;
  if (wall >= 0) {
    const int fixed_axis = wall / 2;
    const int free_axis = 1 - fixed_axis;
    const double fixed_val = (wall % 2 == 0) ? box.low[fixed_axis] : box.high[fixed_axis];
    const double g = goal[free_axis];
    const auto& segs = box.walls[wall];
    const auto it = std::upper_bound(
        segs.begin(), segs.end(), g,
        [](double val, const WallSegment& seg) { return val < seg.high; });
    const auto& seg = (it == segs.end()) ? segs.back() : *it;
    if (!(g >= seg.low && g <= seg.high)) return false;
    const double t = std::clamp((g - seg.low) / (seg.high - seg.low), 0.0, 1.0);
    const auto w1 = static_cast<std::uint64_t>(std::clamp(std::rint(t * kCarrierScaleDbl), 0.0, kCarrierScaleDbl));
    const std::uint64_t w0 = kCarrierScale - w1;
    if (!finalize_stencil_vertex(input, log_rates,
                                 {static_cast<Row>(seg.a), static_cast<Row>(seg.b), -1},
                                 {w0, w1, 0}, 2, fixed_axis, fixed_val, box.vertex_cell[seg.a],
                                 out)) {
      return false;
    }
    return out.point[fixed_axis] == fixed_val && out.point[free_axis] > box.low[free_axis] &&
           out.point[free_axis] < box.high[free_axis];
  }

  const Row cell = field.locate_exact(goal, hint_cell);
  if (cell < 0) return false;
  const Row r0 = static_cast<Row>(input.cells[3 * static_cast<std::size_t>(cell)]);
  const Row r1 = static_cast<Row>(input.cells[3 * static_cast<std::size_t>(cell) + 1]);
  const Row r2 = static_cast<Row>(input.cells[3 * static_cast<std::size_t>(cell) + 2]);
  const double* p0 = input.points.data() + 2 * static_cast<std::size_t>(r0);
  const double* p1 = input.points.data() + 2 * static_cast<std::size_t>(r1);
  const double* p2 = input.points.data() + 2 * static_cast<std::size_t>(r2);

  const double den = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
  if (!(den > 0.0) || !std::isfinite(den)) return false;
  const double u0 =
      std::clamp(((p1[0] - goal[0]) * (p2[1] - goal[1]) - (p1[1] - goal[1]) * (p2[0] - goal[0])) /
                     den,
                 0.0, 1.0);
  const double u1 =
      std::clamp(((p2[0] - goal[0]) * (p0[1] - goal[1]) - (p2[1] - goal[1]) * (p0[0] - goal[0])) /
                     den,
                 0.0, 1.0 - u0);
  const auto c1 = static_cast<std::uint64_t>(std::clamp(std::rint(u0 * kCarrierScaleDbl), 0.0, kCarrierScaleDbl));
  const auto c2 = static_cast<std::uint64_t>(
      std::clamp(std::rint((u0 + u1) * kCarrierScaleDbl), static_cast<double>(c1), kCarrierScaleDbl));
  const std::uint64_t w0 = c1;
  const std::uint64_t w1 = c2 - c1;
  const std::uint64_t w2 = kCarrierScale - c2;

  if (!finalize_stencil_vertex(input, log_rates, {r0, r1, r2}, {w0, w1, w2}, 3, -1, 0.0, cell,
                               out)) {
    return false;
  }
  return out.point[0] > box.low[0] && out.point[0] < box.high[0] &&
         out.point[1] > box.low[1] && out.point[1] < box.high[1];
}

// Lazy exact lifted coordinate cache for exact orient2d checks on near-collinear pairs.
struct LiftedCache {
  const Adapt2DBoxInput& input;
  const std::vector<BoundVertex2D>& verts;
  std::vector<std::array<Dyadic, 2>> lifted;
  std::vector<std::uint8_t> ready;

  LiftedCache(const Adapt2DBoxInput& in, const std::vector<BoundVertex2D>& v)
      : input(in), verts(v), lifted(v.size()), ready(v.size(), 0) {}

  const std::array<Dyadic, 2>& at(std::size_t row) {
    if (!ready[row]) {
      const auto& bv = verts[row];
      Dyadic x, y;
      for (int i = 0; i < bv.count; ++i) {
        const auto sr = static_cast<std::size_t>(bv.rows[i]);
        const Dyadic w(bv.weights[i]);
        x = x + w * Dyadic(input.points[2 * sr]);
        y = y + w * Dyadic(input.points[2 * sr + 1]);
      }
      lifted[row] = {std::move(x), std::move(y)};
      ready[row] = 1;
    }
    return lifted[row];
  }
};

inline double lifted_orient_bound(const double* a, const double* b, const double* c, double& det) {
  const double acx = a[0] - c[0], bcx = b[0] - c[0];
  const double acy = a[1] - c[1], bcy = b[1] - c[1];
  det = acx * bcy - acy * bcx;
  const double mag_x = std::abs(a[0]) + std::abs(b[0]) + std::abs(c[0]);
  const double mag_y = std::abs(a[1]) + std::abs(b[1]) + std::abs(c[1]);
  return 64.0 * std::numeric_limits<double>::epsilon() *
         (std::abs(acx * bcy) + std::abs(acy * bcx) + mag_x * (std::abs(acy) + std::abs(bcy)) +
          mag_y * (std::abs(acx) + std::abs(bcx)) +
          std::numeric_limits<double>::epsilon() * mag_x * mag_y);
}

int orient2d_lifted_sign(const double* a_dbl, const double* b_dbl, const double* c_dbl,
                         auto&& get_a, auto&& get_b, auto&& get_c) {
  double det = 0.0;
  const double bound = lifted_orient_bound(a_dbl, b_dbl, c_dbl, det);
  if (det > bound) return 1;
  if (det < -bound) return -1;
  const auto& a = get_a();
  const auto& b = get_b();
  const auto& c = get_c();
  return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])).sign();
}

// Verify positive rounded + lifted orientations, exact CCW 1-fold wall coverage,
// strict metric band on every distinct edge (evaluated once per edge), and Heron quality.
bool verify_target_mesh_and_walls(const SourceBox2D& box, const std::vector<BoundVertex2D>& verts,
                                  const std::vector<Tri2>& cells, LiftedCache& lifted, double lower,
                                  std::array<std::vector<WallSegment>, 4>& target_walls,
                                  std::int64_t& edge_count, std::array<double, 4>& measurements) {
  const double low_safe = std::max(lower, kLowerLength) + kBandSafetyMargin;
  const double high_safe = kUpperLength - kBandSafetyMargin;
  const std::size_t vcount = verts.size();
  const std::size_t ccount = cells.size();

  // O(V + C) bucketed half-edges + positive orientation check (both rounded & lifted).
  struct TargetHalfEdge {
    Row hi;
    Row cell;
    std::uint8_t slot;
    std::int8_t dir;
  };
  std::vector<std::uint32_t> degree(vcount + 1, 0);
  for (std::size_t c = 0; c < ccount; ++c) {
    const auto [r0, r1, r2] = cells[c];
    const double* p0 = verts[r0].point.data();
    const double* p1 = verts[r1].point.data();
    const double* p2 = verts[r2].point.data();
    if (orient2d_fast(p0, p1, p2) <= 0) return false;
    if (orient2d_lifted_sign(
            p0, p1, p2, [&]() -> const auto& { return lifted.at(r0); },
            [&]() -> const auto& { return lifted.at(r1); },
            [&]() -> const auto& { return lifted.at(r2); }) <= 0) {
      return false;
    }
    ++degree[static_cast<std::size_t>(std::min(r1, r2)) + 1];
    ++degree[static_cast<std::size_t>(std::min(r2, r0)) + 1];
    ++degree[static_cast<std::size_t>(std::min(r0, r1)) + 1];
  }
  for (std::size_t v = 1; v <= vcount; ++v) degree[v] += degree[v - 1];
  std::vector<TargetHalfEdge> half_edges(degree[vcount]);
  std::vector<std::uint32_t> cursor(degree.begin(), degree.end() - 1);
  for (std::size_t c = 0; c < ccount; ++c) {
    const Row rv[3] = {cells[c][0], cells[c][1], cells[c][2]};
    for (int slot = 0; slot < 3; ++slot) {
      const Row a = rv[(slot + 1) % 3], b = rv[(slot + 2) % 3];
      const Row lo = std::min(a, b), hi = std::max(a, b);
      half_edges[cursor[static_cast<std::size_t>(lo)]++] = {
          hi, static_cast<Row>(c), static_cast<std::uint8_t>(slot),
          static_cast<std::int8_t>(a < b ? 1 : -1)};
    }
  }

  // Evaluate each distinct edge length ONCE, scatter to incident triangles for Heron quality.
  std::vector<double> cell_edge_lengths(3 * ccount, 0.0);
  double min_len = std::numeric_limits<double>::infinity();
  double max_len = 0.0;
  std::int64_t distinct_edges = 0;
  for (std::size_t lo = 0; lo < vcount; ++lo) {
    const std::uint32_t begin = degree[lo];
    const std::uint32_t end = degree[lo + 1];
    if (begin == end) continue;
    std::sort(half_edges.begin() + begin, half_edges.begin() + end,
              [](const TargetHalfEdge& x, const TargetHalfEdge& y) { return x.hi < y.hi; });
    std::uint32_t i = begin;
    while (i < end) {
      std::uint32_t j = i + 1;
      while (j < end && half_edges[j].hi == half_edges[i].hi) ++j;
      const std::uint32_t mult = j - i;
      const auto hi = static_cast<std::size_t>(half_edges[i].hi);
      const double len = metric_edge_length_2d(verts[lo].point, verts[lo].scalar_metric,
                                               verts[hi].point, verts[hi].scalar_metric);
      if (!(len >= low_safe && len <= high_safe)) return false;
      min_len = std::min(min_len, len);
      max_len = std::max(max_len, len);
      ++distinct_edges;

      if (mult == 2) {
        if (half_edges[i].dir == half_edges[i + 1].dir) return false;
        cell_edge_lengths[3 * static_cast<std::size_t>(half_edges[i].cell) + half_edges[i].slot] =
            len;
        cell_edge_lengths[3 * static_cast<std::size_t>(half_edges[i + 1].cell) +
                          half_edges[i + 1].slot] = len;
      } else if (mult == 1) {
        cell_edge_lengths[3 * static_cast<std::size_t>(half_edges[i].cell) + half_edges[i].slot] =
            len;
        const std::size_t a = (half_edges[i].dir > 0) ? lo : hi;
        const std::size_t b = (half_edges[i].dir > 0) ? hi : lo;
        const int w = box.wall_of_edge(verts[a].point, verts[b].point);
        if (w < 0 || !box.ccw_wall_edge(w, verts[a].point, verts[b].point)) return false;
        const int free_axis = 1 - (w / 2);
        std::size_t lo_v = a, hi_v = b;
        double lo_c = verts[a].point[free_axis], hi_c = verts[b].point[free_axis];
        if (lo_c > hi_c) {
          std::swap(lo_v, hi_v);
          std::swap(lo_c, hi_c);
        }
        if (!(lo_c < hi_c)) return false;
        Id id_a = verts[lo_v].target_id, id_b = verts[hi_v].target_id;
        if (id_a > id_b) std::swap(id_a, id_b);
        target_walls[w].push_back({lo_c, hi_c, lo_v, hi_v, {id_a, id_b}});
      } else {
        return false;
      }
      i = j;
    }
  }
  for (int w = 0; w < 4; ++w) {
    const int free_axis = 1 - (w / 2);
    if (target_walls[w].empty()) return false;
    std::sort(target_walls[w].begin(), target_walls[w].end(),
              [](const WallSegment& x, const WallSegment& y) { return x.low < y.low; });
    if (target_walls[w].front().low != box.low[free_axis] ||
        target_walls[w].back().high != box.high[free_axis]) {
      return false;
    }
    for (std::size_t k = 1; k < target_walls[w].size(); ++k) {
      if (target_walls[w][k - 1].high != target_walls[w][k].low) return false;
    }
  }

  double min_q = std::numeric_limits<double>::infinity();
  for (std::size_t c = 0; c < ccount; ++c) {
    const double q = heron_quality(cell_edge_lengths[3 * c], cell_edge_lengths[3 * c + 1],
                                   cell_edge_lengths[3 * c + 2]);
    if (!(q >= 0.65) || !std::isfinite(q)) return false;
    min_q = std::min(min_q, q);
  }
  edge_count = distinct_edges;
  measurements = {min_len, max_len, 1.0, min_q};
  return true;
}

void write_wall_edge_relations(const SourceBox2D& box,
                               const std::array<std::vector<WallSegment>, 4>& target_walls,
                               LiftedCache& lifted, Adapt2DBoxOutput& out) {
  struct Rel {
    std::array<Id, 2> src;
    std::array<Id, 2> tgt;
    std::int32_t kind;
  };
  std::vector<Rel> rels;
  for (int w = 0; w < 4; ++w) {
    const int free_axis = 1 - (w / 2);
    const auto& src = box.walls[w];
    const auto& tgt = target_walls[w];
    std::size_t si = 0;
    for (const auto& t : tgt) {
      const double t_lo_pad = std::nextafter(
          std::nextafter(t.low, -std::numeric_limits<double>::infinity()),
          -std::numeric_limits<double>::infinity());
      while (si < src.size() && src[si].high <= t_lo_pad) ++si;
      for (std::size_t sj = si; sj < src.size(); ++sj) {
        const double t_hi_pad = std::nextafter(
            std::nextafter(t.high, std::numeric_limits<double>::infinity()),
            std::numeric_limits<double>::infinity());
        if (src[sj].low >= t_hi_pad) break;
        bool overlaps = std::max(src[sj].low, t.low) < std::min(src[sj].high, t.high);
        if (!overlaps) {
          const Dyadic s0(src[sj].low), s1(src[sj].high);
          const Dyadic& t0 = lifted.at(t.a)[free_axis];
          const Dyadic& t1 = lifted.at(t.b)[free_axis];
          const Dyadic lo = (s0 - t0).sign() >= 0 ? s0 : t0;
          const Dyadic hi = (s1 - t1).sign() <= 0 ? s1 : t1;
          overlaps = (hi - lo).sign() > 0;
        }
        if (overlaps && src[sj].key != t.key) {
          rels.push_back({src[sj].key, t.key, kSwappedFrom});
        }
      }
    }
  }
  std::sort(rels.begin(), rels.end(), [](const Rel& a, const Rel& b) {
    if (a.src != b.src) return a.src < b.src;
    if (a.tgt != b.tgt) return a.tgt < b.tgt;
    return a.kind < b.kind;
  });
  out.edge_relation_sources.reserve(2 * rels.size());
  out.edge_relation_targets.reserve(2 * rels.size());
  out.edge_relation_kinds.reserve(rels.size());
  for (const auto& r : rels) {
    out.edge_relation_sources.push_back(r.src[0]);
    out.edge_relation_sources.push_back(r.src[1]);
    out.edge_relation_targets.push_back(r.tgt[0]);
    out.edge_relation_targets.push_back(r.tgt[1]);
    out.edge_relation_kinds.push_back(r.kind);
  }
}

// Exact 2D triangle-triangle positive-area SAT over all 6 CCW edges (both rounded
// and exact lifted), with strict-separation early exit so non-borderline pairs
// never construct Dyadic lifted coordinates.
bool positive_area_overlap_2d(const double* t0, const double* t1, const double* t2,
                              Row tv0, Row tv1, Row tv2, const double* s0, const double* s1,
                              const double* s2, LiftedCache& lifted,
                              std::array<bool, 3>& cross_source_edge) {
  const double* tp[3] = {t0, t1, t2};
  const double* sp[3] = {s0, s1, s2};
  const Row tv[3] = {tv0, tv1, tv2};

  // First check the 3 target edges: if any target edge strictly separates all 3
  // source vertices beyond the lifted-shift margin, neither rounded nor lifted can overlap!
  bool rounded_overlap = true;
  bool borderline_sep = false;
  for (int e = 0; e < 3; ++e) {
    const double* a = tp[e];
    const double* b = tp[(e + 1) % 3];
    bool any_pos = false;
    bool any_near = false;
    for (int j = 0; j < 3; ++j) {
      double det = 0.0;
      const double bound = lifted_orient_bound(a, b, sp[j], det);
      if (det > bound) {
        any_pos = true;
        break;
      }
      if (det >= -bound) {
        any_near = true;
        if (orient2d(a, b, sp[j]) > 0) {
          any_pos = true;
          break;
        }
      }
    }
    if (!any_pos) {
      if (!any_near) return false;  // Strictly separated in both rounded and lifted!
      rounded_overlap = false;
      borderline_sep = true;
    }
  }

  cross_source_edge = {false, false, false};
  for (int slot = 0; slot < 3; ++slot) {
    const double* a = sp[(slot + 1) % 3];
    const double* b = sp[(slot + 2) % 3];
    bool any_pos = false;
    bool any_near = false;
    for (int i = 0; i < 3; ++i) {
      double det = 0.0;
      const double bound = lifted_orient_bound(a, b, tp[i], det);
      if (det > bound) {
        any_pos = true;
      } else if (det < -bound) {
        cross_source_edge[slot] = true;
      } else {
        any_near = true;
        const int s = orient2d(a, b, tp[i]);
        if (s > 0) any_pos = true;
        if (s <= 0) cross_source_edge[slot] = true;
      }
    }
    if (!any_pos) {
      if (!any_near) return false;  // Strictly separated in both rounded and lifted!
      rounded_overlap = false;
      borderline_sep = true;
    }
  }
  if (rounded_overlap) return true;
  if (!borderline_sep) return false;

  // Rare borderline case: decide exact lifted overlap in Dyadic.
  auto src_dyadic = [](const double* p) { return std::array<Dyadic, 2>{Dyadic(p[0]), Dyadic(p[1])}; };
  for (int e = 0; e < 3; ++e) {
    const int e1 = (e + 1) % 3;
    bool any_pos = false;
    for (int j = 0; j < 3; ++j) {
      if (orient2d_lifted_sign(
              tp[e], tp[e1], sp[j], [&]() -> const auto& { return lifted.at(tv[e]); },
              [&]() -> const auto& { return lifted.at(tv[e1]); },
              [&]() { return src_dyadic(sp[j]); }) > 0) {
        any_pos = true;
        break;
      }
    }
    if (!any_pos) return false;
  }
  for (int slot = 0; slot < 3; ++slot) {
    const double* a = sp[(slot + 1) % 3];
    const double* b = sp[(slot + 2) % 3];
    bool any_pos = false;
    for (int i = 0; i < 3; ++i) {
      const int s = orient2d_lifted_sign(
          a, b, tp[i], [&]() { return src_dyadic(a); }, [&]() { return src_dyadic(b); },
          [&]() -> const auto& { return lifted.at(tv[i]); });
      if (s > 0) any_pos = true;
      if (s < 0) cross_source_edge[slot] = true;
    }
    if (!any_pos) return false;
  }
  return true;
}

bool write_cell_ancestry_2d(const Adapt2DBoxInput& input, const SourceBox2D& box,
                            const ScalarField2D& field, const std::vector<BoundVertex2D>& verts,
                            const std::vector<Tri2>& target_cells, LiftedCache& lifted,
                            Adapt2DBoxOutput& out) {
  const std::size_t scount = input.cell_ids.size();
  const std::size_t tcount = target_cells.size();

  struct Box2 {
    double lx, ly, hx, hy;
  };
  std::vector<Box2> sboxes(scount);
  for (std::size_t s = 0; s < scount; ++s) {
    const double* p0 = input.points.data() + 2 * static_cast<std::size_t>(input.cells[3 * s]);
    const double* p1 = input.points.data() + 2 * static_cast<std::size_t>(input.cells[3 * s + 1]);
    const double* p2 = input.points.data() + 2 * static_cast<std::size_t>(input.cells[3 * s + 2]);
    sboxes[s] = {
        std::min({p0[0], p1[0], p2[0]}), std::min({p0[1], p1[1], p2[1]}),
        std::max({p0[0], p1[0], p2[0]}), std::max({p0[1], p1[1], p2[1]}),
    };
  }

  out.ancestry_offsets.clear();
  out.ancestry_offsets.reserve(tcount + 1);
  out.ancestry_offsets.push_back(0);
  out.ancestry_ids.clear();
  out.ancestry_ids.reserve(6 * tcount);

  std::vector<std::uint32_t> seen(scount, 0);
  std::uint32_t stamp = 0;
  std::vector<Row> queue;
  queue.reserve(64);
  std::vector<Id> ancestors;
  ancestors.reserve(32);

  for (std::size_t t = 0; t < tcount; ++t) {
    const auto [tv0, tv1, tv2] = target_cells[t];
    const double* t0 = verts[tv0].point.data();
    const double* t1 = verts[tv1].point.data();
    const double* t2 = verts[tv2].point.data();
    const double raw_lx = std::min({t0[0], t1[0], t2[0]});
    const double raw_ly = std::min({t0[1], t1[1], t2[1]});
    const double raw_hx = std::max({t0[0], t1[0], t2[0]});
    const double raw_hy = std::max({t0[1], t1[1], t2[1]});
    const double pad_lx = std::nextafter(
        std::nextafter(raw_lx, -std::numeric_limits<double>::infinity()),
        -std::numeric_limits<double>::infinity());
    const double pad_ly = std::nextafter(
        std::nextafter(raw_ly, -std::numeric_limits<double>::infinity()),
        -std::numeric_limits<double>::infinity());
    const double pad_hx = std::nextafter(
        std::nextafter(raw_hx, std::numeric_limits<double>::infinity()),
        std::numeric_limits<double>::infinity());
    const double pad_hy = std::nextafter(
        std::nextafter(raw_hy, std::numeric_limits<double>::infinity()),
        std::numeric_limits<double>::infinity());

    if (++stamp == 0) {
      std::fill(seen.begin(), seen.end(), 0);
      stamp = 1;
    }
    queue.clear();
    ancestors.clear();
    auto push_seed = [&](Row s) {
      if (s >= 0 && seen[s] != stamp) {
        seen[s] = stamp;
        queue.push_back(s);
      }
    };
    push_seed(verts[tv0].owner_cell);
    push_seed(verts[tv1].owner_cell);
    push_seed(verts[tv2].owner_cell);

    auto run_bfs = [&]() {
      std::size_t head = 0;
      while (head < queue.size()) {
        const Row s = queue[head++];
        const auto& sb = sboxes[s];
        if (sb.hx < pad_lx || sb.lx > pad_hx || sb.hy < pad_ly || sb.ly > pad_hy) continue;
        const double* s0 = input.points.data() + 2 * static_cast<std::size_t>(input.cells[3 * s]);
        const double* s1 =
            input.points.data() + 2 * static_cast<std::size_t>(input.cells[3 * s + 1]);
        const double* s2 =
            input.points.data() + 2 * static_cast<std::size_t>(input.cells[3 * s + 2]);
        std::array<bool, 3> cross_edge{};
        const bool hit = positive_area_overlap_2d(t0, t1, t2, tv0, tv1, tv2, s0, s1, s2, lifted,
                                                  cross_edge);
        if (hit) {
          ancestors.push_back(input.cell_ids[s]);
          for (int slot = 0; slot < 3; ++slot) {
            if (!cross_edge[slot]) continue;
            const Row nb = box.neighbors[3 * static_cast<std::size_t>(s) + slot];
            if (nb >= 0 && seen[nb] != stamp) {
              const auto& nbox = sboxes[nb];
              if (!(nbox.hx < pad_lx || nbox.lx > pad_hx || nbox.hy < pad_ly ||
                    nbox.ly > pad_hy)) {
                seen[nb] = stamp;
                queue.push_back(nb);
              }
            }
          }
        } else if (ancestors.empty()) {
          for (int slot = 0; slot < 3; ++slot) {
            const Row nb = box.neighbors[3 * static_cast<std::size_t>(s) + slot];
            if (nb >= 0 && seen[nb] != stamp) {
              const auto& nbox = sboxes[nb];
              if (!(nbox.hx < pad_lx || nbox.lx > pad_hx || nbox.hy < pad_ly ||
                    nbox.ly > pad_hy)) {
                seen[nb] = stamp;
                queue.push_back(nb);
              }
            }
          }
        }
      }
    };
    run_bfs();
    if (ancestors.empty()) {
      const Point2 centroid{(t0[0] + t1[0] + t2[0]) / 3.0, (t0[1] + t1[1] + t2[1]) / 3.0};
      push_seed(field.locate_exact(centroid, verts[tv0].owner_cell));
      run_bfs();
    }
    if (ancestors.empty()) return false;
    std::sort(ancestors.begin(), ancestors.end());
    ancestors.erase(std::unique(ancestors.begin(), ancestors.end()), ancestors.end());
    out.ancestry_ids.insert(out.ancestry_ids.end(), ancestors.begin(), ancestors.end());
    out.ancestry_offsets.push_back(static_cast<std::int64_t>(out.ancestry_ids.size()));
  }
  return true;
}

}  // namespace

Adapt2DBoxOutcome adapt_metric_2d_box(const Adapt2DBoxInput& input, Adapt2DBoxOutput& output) {
  const double lower = std::max(input.lower, kLowerLength);
  if (!(lower < kUpperLength)) return Adapt2DBoxOutcome::kDeclined;

  SourceBox2D box;
  std::vector<double> log_rates;
  if (!bind_source_box_2d(input, box, log_rates)) return Adapt2DBoxOutcome::kDeclined;
  if (check_already_unit(input, lower, output)) return Adapt2DBoxOutcome::kAlreadyUnit;

  ScalarField2D field;
  if (!field.prepare(input.points, input.cells, log_rates, box.low, box.high)) {
    return Adapt2DBoxOutcome::kDeclined;
  }
  ScalarSites2D sites;
  if (!propose_scalar_box_2d(field, lower, sites)) return Adapt2DBoxOutcome::kDeclined;

  const std::size_t vcount = sites.points.size();
  const std::size_t ccount = sites.cells.size();
  std::vector<BoundVertex2D> bound(vcount);
  Id next_vid = input.next_vertex_id;
  for (std::size_t v = 0; v < vcount; ++v) {
    if (!bind_box_point_2d(input, box, field, log_rates, sites.points[v], sites.samples[v].cell,
                           bound[v])) {
      return Adapt2DBoxOutcome::kDeclined;
    }
    if (bound[v].target_id == 0 && !(bound[v].count == 1 &&
                                     input.vertex_ids[bound[v].rows[0]] == 0 &&
                                     bound[v].point[0] == input.points[2 * bound[v].rows[0]] &&
                                     bound[v].point[1] == input.points[2 * bound[v].rows[0] + 1])) {
      if (!fits_ids(next_vid, 1)) return Adapt2DBoxOutcome::kDeclined;
      bound[v].target_id = next_vid++;
    }
  }
  if (!fits_ids(input.next_cell_id, ccount)) return Adapt2DBoxOutcome::kDeclined;

  std::vector<std::size_t> order(vcount);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) { return bound[a].target_id < bound[b].target_id; });
  std::vector<Row> remap(vcount, -1);
  std::vector<BoundVertex2D> ordered_verts(vcount);
  for (std::size_t i = 0; i < vcount; ++i) {
    remap[order[i]] = static_cast<Row>(i);
    ordered_verts[i] = bound[order[i]];
    if (i > 0 && ordered_verts[i].target_id == ordered_verts[i - 1].target_id) {
      return Adapt2DBoxOutcome::kDeclined;
    }
  }
  std::vector<Tri2> ordered_cells(ccount);
  for (std::size_t c = 0; c < ccount; ++c) {
    ordered_cells[c] = {remap[sites.cells[c][0]], remap[sites.cells[c][1]],
                        remap[sites.cells[c][2]]};
  }

  LiftedCache lifted(input, ordered_verts);
  std::array<std::vector<WallSegment>, 4> target_walls;
  std::int64_t edge_count = 0;
  std::array<double, 4> measurements{};
  if (!verify_target_mesh_and_walls(box, ordered_verts, ordered_cells, lifted, lower, target_walls,
                                    edge_count, measurements)) {
    return Adapt2DBoxOutcome::kDeclined;
  }

  Adapt2DBoxOutput staged;
  staged.measured_edges = edge_count;
  staged.measurements = measurements;
  staged.points.resize(2 * vcount);
  staged.vertex_ids.resize(vcount);
  staged.scalar_metrics.resize(vcount);
  staged.stencil_offsets.reserve(vcount + 1);
  staged.stencil_offsets.push_back(0);
  staged.stencil_rows.reserve(3 * vcount);
  staged.stencil_weights.reserve(3 * vcount);
  for (std::size_t v = 0; v < vcount; ++v) {
    staged.points[2 * v] = ordered_verts[v].point[0];
    staged.points[2 * v + 1] = ordered_verts[v].point[1];
    staged.vertex_ids[v] = ordered_verts[v].target_id;
    staged.scalar_metrics[v] = ordered_verts[v].scalar_metric;
    for (int k = 0; k < ordered_verts[v].count; ++k) {
      staged.stencil_rows.push_back(ordered_verts[v].rows[k]);
      staged.stencil_weights.push_back(ordered_verts[v].weights[k]);
    }
    staged.stencil_offsets.push_back(static_cast<std::int64_t>(staged.stencil_rows.size()));
  }
  staged.cells.resize(3 * ccount);
  staged.cell_ids.resize(ccount);
  for (std::size_t c = 0; c < ccount; ++c) {
    staged.cells[3 * c] = ordered_cells[c][0];
    staged.cells[3 * c + 1] = ordered_cells[c][1];
    staged.cells[3 * c + 2] = ordered_cells[c][2];
    staged.cell_ids[c] = input.next_cell_id + c;
  }
  write_wall_edge_relations(box, target_walls, lifted, staged);
  if (!write_cell_ancestry_2d(input, box, field, ordered_verts, ordered_cells, lifted, staged)) {
    return Adapt2DBoxOutcome::kDeclined;
  }

  output = std::move(staged);
  return Adapt2DBoxOutcome::kAdapted;
}

}  // namespace phx::mc::metric

struct phx_mc_metric_adapt2d_box {
  phx::mc::metric::Adapt2DBoxOutput output;
};

extern "C" {

int32_t phx_mc_metric_adapt_2d_box(const phx_mc_metric_adapt2d_box_request* request,
                                   int32_t* outcome, phx_mc_metric_adapt2d_box** result) {
  if (request == nullptr || outcome == nullptr || result == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *result = nullptr;
  *outcome = static_cast<int32_t>(phx::mc::metric::Adapt2DBoxOutcome::kDeclined);
  if (request->vertex_count < 0 || request->cell_count < 0 || request->edge_count < 0 ||
      !(request->lower > 0.0) || !(request->lower <= 1.0)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (!phx::mc::addressable(request->vertex_count, 2, sizeof(double)) ||
      !phx::mc::addressable(request->cell_count, 3, sizeof(int64_t)) ||
      !phx::mc::addressable(request->edge_count, 2, sizeof(int64_t))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if ((request->vertex_count > 0 &&
       (request->points == nullptr || request->scalar_metrics == nullptr ||
        request->vertex_ids == nullptr || request->fixed_vertices == nullptr)) ||
      (request->cell_count > 0 && (request->cells == nullptr || request->cell_ids == nullptr)) ||
      (request->edge_count > 0 &&
       (request->edges == nullptr || request->edge_classes == nullptr ||
        request->protected_edges == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    const auto v = static_cast<std::size_t>(request->vertex_count);
    const auto c = static_cast<std::size_t>(request->cell_count);
    const auto e = static_cast<std::size_t>(request->edge_count);
    phx::mc::metric::Adapt2DBoxInput input{
        {request->points, 2 * v},
        {request->scalar_metrics, v},
        {request->vertex_ids, v},
        {request->fixed_vertices, v},
        {request->cells, 3 * c},
        {request->cell_ids, c},
        {request->edges, 2 * e},
        {request->edge_classes, e},
        {request->protected_edges, e},
        request->next_vertex_id,
        request->next_cell_id,
        request->lower,
    };
    auto owned = std::make_unique<phx_mc_metric_adapt2d_box>();
    const auto res = phx::mc::metric::adapt_metric_2d_box(input, owned->output);
    *outcome = static_cast<int32_t>(res);
    if (res != phx::mc::metric::Adapt2DBoxOutcome::kDeclined) {
      *result = owned.release();
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_metric_adapt_2d_sizes(const phx_mc_metric_adapt2d_box* result, int64_t* sizes) {
  if (result == nullptr || sizes == nullptr) return PHX_MC_INVALID_ARGUMENT;
  const auto& out = result->output;
  sizes[0] = static_cast<int64_t>(out.vertex_ids.size());
  sizes[1] = static_cast<int64_t>(out.stencil_rows.size());
  sizes[2] = static_cast<int64_t>(out.cell_ids.size());
  sizes[3] = static_cast<int64_t>(out.ancestry_ids.size());
  sizes[4] = static_cast<int64_t>(out.edge_relation_kinds.size());
  sizes[5] = out.measured_edges;
  return PHX_MC_OK;
}

int32_t phx_mc_metric_adapt_2d_export(const phx_mc_metric_adapt2d_box* result,
                                      const phx_mc_metric_adapt2d_output* banks) {
  if (result == nullptr || banks == nullptr || banks->measurements == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const auto& out = result->output;
  if (!phx::mc::output_bank_valid(out.points, banks->points) ||
      !phx::mc::output_bank_valid(out.vertex_ids, banks->vertex_ids) ||
      !phx::mc::output_bank_valid(out.scalar_metrics, banks->scalar_metrics) ||
      !phx::mc::output_bank_valid(out.stencil_offsets, banks->stencil_offsets) ||
      !phx::mc::output_bank_valid(out.stencil_rows, banks->stencil_rows) ||
      !phx::mc::output_bank_valid(out.stencil_weights, banks->stencil_weights) ||
      !phx::mc::output_bank_valid(out.cells, banks->cells) ||
      !phx::mc::output_bank_valid(out.cell_ids, banks->cell_ids) ||
      !phx::mc::output_bank_valid(out.ancestry_offsets, banks->ancestry_offsets) ||
      !phx::mc::output_bank_valid(out.ancestry_ids, banks->ancestry_ids) ||
      !phx::mc::output_bank_valid(out.edge_relation_sources, banks->edge_relation_sources) ||
      !phx::mc::output_bank_valid(out.edge_relation_targets, banks->edge_relation_targets) ||
      !phx::mc::output_bank_valid(out.edge_relation_kinds, banks->edge_relation_kinds)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  phx::mc::copy_output_bank(out.points, banks->points);
  phx::mc::copy_output_bank(out.vertex_ids, banks->vertex_ids);
  phx::mc::copy_output_bank(out.scalar_metrics, banks->scalar_metrics);
  phx::mc::copy_output_bank(out.stencil_offsets, banks->stencil_offsets);
  phx::mc::copy_output_bank(out.stencil_rows, banks->stencil_rows);
  phx::mc::copy_output_bank(out.stencil_weights, banks->stencil_weights);
  phx::mc::copy_output_bank(out.cells, banks->cells);
  phx::mc::copy_output_bank(out.cell_ids, banks->cell_ids);
  phx::mc::copy_output_bank(out.ancestry_offsets, banks->ancestry_offsets);
  phx::mc::copy_output_bank(out.ancestry_ids, banks->ancestry_ids);
  phx::mc::copy_output_bank(out.edge_relation_sources, banks->edge_relation_sources);
  phx::mc::copy_output_bank(out.edge_relation_targets, banks->edge_relation_targets);
  phx::mc::copy_output_bank(out.edge_relation_kinds, banks->edge_relation_kinds);
  std::copy(out.measurements.begin(), out.measurements.end(), banks->measurements);
  return PHX_MC_OK;
}

void phx_mc_metric_adapt_2d_free(phx_mc_metric_adapt2d_box* result) { delete result; }

}  // extern "C"
