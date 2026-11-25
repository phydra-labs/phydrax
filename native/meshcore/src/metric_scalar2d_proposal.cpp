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
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "bounded_lbfgs.hpp"
#include "deterministic_float.hpp"
#include "floating_point_mode.hpp"
#include "key_map.hpp"
#include "mesh.hpp"
#include "metric_box_rate.hpp"
#include "metric_float.hpp"
#include "predicates.hpp"
#include "topology_keys.hpp"

namespace phx::mc::metric {
namespace {

using Row = std::int32_t;
using Edge = std::array<Row, 2>;
using EdgeKey = std::uint64_t;

constexpr double kSeedSeparation = 0.74;
constexpr double kDensifySeparation = 0.62;
constexpr double kShapeFloor2D = 0.72;
constexpr double kShapeTarget2D = 0.74;
constexpr double kSqrt3x2 = 3.46410161513775438635;  // 2 * sqrt(3)
constexpr double kNearEqual = 0.002;
constexpr double kPenaltyGrowth = 2.5;
constexpr int kMaxStages = 7;
constexpr int kStepsPerStage = 60;
constexpr int kSplitRounds = 32;
constexpr double kInvalidMerit = std::numeric_limits<double>::quiet_NaN();
constexpr std::size_t kWorkBuffers = 5 + 2 * detail::BoundedLbfgsHistory::kCapacity;

struct Band {
  double low;
  double high;
};

struct LogRate {
  double log;
  double rate;
};

bool positive_finite(double x) { return x > 0.0 && std::isfinite(x); }

LogRate make_log_rate(double lr) { return {lr, deterministic::exp(lr)}; }

double mean_rate(double log_a, double rate_a, double log_b, double rate_b) {
  const double gap = log_b - log_a;
  if (std::abs(gap) < kNearEqual) {
    const double w2 = (0.25 * gap) * gap;
    return (0.5 * rate_a + 0.5 * rate_b) * (1.0 + w2 * (-1.0 / 3.0 + w2 * (2.0 / 15.0)));
  }
  return (rate_b - rate_a) / gap;
}

std::array<double, 2> mean_rate_gradient(double log_a, double rate_a, double log_b, double rate_b,
                                         double mean) {
  const double gap = log_b - log_a;
  if (std::abs(gap) < kNearEqual) {
    const double w = 0.5 * gap;
    const double odd = (0.5 * rate_a + 0.5 * rate_b) * (w * (1.0 / 3.0 - w * w * (2.0 / 15.0)));
    return {0.5 * (mean - odd), 0.5 * (mean + odd)};
  }
  return {(mean - rate_a) / gap, (rate_b - mean) / gap};
}

double metric_fraction(double delta, double mass_fraction) {
  if (delta == 0.0 || mass_fraction == 0.0 || mass_fraction == 1.0) return mass_fraction;
  if (std::abs(delta) < 1e-8) {
    return mass_fraction + 0.5 * delta * mass_fraction * (1.0 - mass_fraction);
  }
  if (delta < 700.0) {
    return deterministic::log1p(mass_fraction * (delta * integrated_exponential(delta))) / delta;
  }
  const double x = delta + deterministic::log(mass_fraction);
  return (x > 0.0 ? x + deterministic::log1p((1.0 - mass_fraction) * deterministic::exp(-x))
                  : deterministic::log1p(deterministic::exp(x) - mass_fraction)) /
         delta;
}

std::size_t round_even(double x) {
  const auto floor_val = static_cast<std::size_t>(x);
  const double frac = x - static_cast<double>(floor_val);
  return floor_val + (frac > 0.5 || (frac == 0.5 && (floor_val & 1U) != 0));
}

double dist2(const Point2& a, const Point2& b) {
  const double dx = b[0] - a[0], dy = b[1] - a[1];
  return dx * dx + dy * dy;
}

double dist(const Point2& a, const Point2& b) {
  const double sq = dist2(a, b);
  if (sq >= std::numeric_limits<double>::min() && std::isfinite(sq)) return std::sqrt(sq);
  const double dx = b[0] - a[0], dy = b[1] - a[1];
  const double scale = std::max(std::abs(dx), std::abs(dy));
  if (scale == 0.0) return 0.0;
  const double ux = dx / scale, uy = dy / scale;
  return scale * std::sqrt(ux * ux + uy * uy);
}

bool topology_accepted_2d(std::int32_t status) {
  if (status == PHX_MC_ALLOCATION_FAILURE) throw std::bad_alloc();
  if (status == PHX_MC_INDEX_RANGE_ERROR) {
    throw std::length_error("Scalar 2D box site index overflow.");
  }
  if (status == PHX_MC_INTERNAL_ERROR) {
    throw std::logic_error("Scalar 2D box triangulation failed.");
  }
  return status == PHX_MC_OK;
}

unsigned wall_mask(const ScalarField2D& field, const Point2& p) {
  unsigned mask = 0;
  if (p[0] == field.low[0]) mask |= 1u;
  if (p[0] == field.high[0]) mask |= 2u;
  if (p[1] == field.low[1]) mask |= 4u;
  if (p[1] == field.high[1]) mask |= 8u;
  return mask;
}

// O(V + T) linear-time distinct undirected edge collector using vertex buckets
// by min endpoint (average ~3 outgoing edges per vertex -> no O(T log T) sort).
bool collect_edges_linear(std::size_t vertices, std::span<const Tri2> cells,
                          std::vector<Edge>& edges) {
  edges.clear();
  if (cells.empty() || vertices == 0) return false;
  std::vector<std::uint32_t> degree(vertices + 1, 0);
  for (const auto& [a, b, c] : cells) {
    if (a < 0 || b < 0 || c < 0 || static_cast<std::size_t>(a) >= vertices ||
        static_cast<std::size_t>(b) >= vertices || static_cast<std::size_t>(c) >= vertices) {
      return false;
    }
    ++degree[static_cast<std::size_t>(std::min(a, b)) + 1];
    ++degree[static_cast<std::size_t>(std::min(b, c)) + 1];
    ++degree[static_cast<std::size_t>(std::min(c, a)) + 1];
  }
  for (std::size_t v = 1; v <= vertices; ++v) degree[v] += degree[v - 1];
  std::vector<Row> targets(degree[vertices]);
  std::vector<std::uint32_t> cursor(degree.begin(), degree.end() - 1);
  for (const auto& [a, b, c] : cells) {
    const Row e[3][2] = {{a, b}, {b, c}, {c, a}};
    for (int k = 0; k < 3; ++k) {
      const Row lo = std::min(e[k][0], e[k][1]);
      const Row hi = std::max(e[k][0], e[k][1]);
      targets[cursor[static_cast<std::size_t>(lo)]++] = hi;
    }
  }
  edges.reserve(3 * cells.size() / 2 + vertices);
  for (std::size_t v = 0; v < vertices; ++v) {
    const std::uint32_t begin = degree[v];
    const std::uint32_t end = degree[v + 1];
    if (begin == end) continue;
    if (end - begin <= 16) {
      for (std::uint32_t i = begin + 1; i < end; ++i) {
        const Row val = targets[i];
        std::uint32_t j = i;
        while (j > begin && targets[j - 1] > val) {
          targets[j] = targets[j - 1];
          --j;
        }
        targets[j] = val;
      }
    } else {
      std::sort(targets.begin() + begin, targets.begin() + end);
    }
    Row prev = -1;
    for (std::uint32_t i = begin; i < end; ++i) {
      if (targets[i] != prev) {
        edges.push_back({static_cast<Row>(v), targets[i]});
        prev = targets[i];
      }
    }
  }
  return !edges.empty();
}

// Sparse 2D bucket grid for separation queries during seeding and densification.
class SeparationGrid2D {
 public:
  SeparationGrid2D(Point2 origin, double width, std::size_t count)
      : origin_(origin), width_(width), heads_(count) {
    links_.reserve(count);
  }

  void insert(const Point2& p, Row row) {
    const int ix = static_cast<int>((p[0] - origin_[0]) / width_);
    const int iy = static_cast<int>((p[1] - origin_[1]) / width_);
    auto& head = heads_.slot(key(ix, iy));
    links_.push_back({row, static_cast<Row>(head)});
    head = static_cast<Row>(links_.size() - 1);
  }

  template <class Conflicts>
  bool any(const Point2& p, double radius, Conflicts&& conflicts) const {
    const double cx = (p[0] - origin_[0]) / width_;
    const double cy = (p[1] - origin_[1]) / width_;
    const double pad_x = 8 * std::numeric_limits<double>::epsilon() * (1.0 + cx);
    const double pad_y = 8 * std::numeric_limits<double>::epsilon() * (1.0 + cy);
    const double rx = radius / width_ + pad_x;
    const double ry = radius / width_ + pad_y;
    const int x0 = static_cast<int>(std::max(0.0, cx - rx));
    const int x1 = static_cast<int>(cx + rx);
    const int y0 = static_cast<int>(std::max(0.0, cy - ry));
    const int y1 = static_cast<int>(cy + ry);
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        auto link = heads_.find(key(x, y));
        while (link >= 0) {
          if (conflicts(links_[link].row)) return true;
          link = links_[link].next;
        }
      }
    }
    return false;
  }

 private:
  struct Link {
    Row row;
    Row next;
  };
  static std::uint64_t key(int x, int y) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) << 32) |
           static_cast<std::uint32_t>(y);
  }
  Point2 origin_;
  double width_;
  KeyMap heads_;
  std::vector<Link> links_;
};

double separation_width_2d(const ScalarField2D& field, double separation) {
  const double max_r = separation / deterministic::exp(field.minimum_log_rate);
  const double min_r = separation / deterministic::exp(field.maximum_log_rate);
  if (!std::isfinite(max_r)) return 0.0;
  const double extent = std::max(field.high[0] - field.low[0], field.high[1] - field.low[1]);
  return std::max({std::sqrt(max_r) * std::sqrt(min_r), 0.25 * max_r, std::ldexp(extent, -20)});
}

struct SeparationProbe2D {
  Point2 point;
  LogRate rate;
  double separation;
  double radius;

  SeparationProbe2D(const Point2& p, LogRate r, LogRate min_r, double sep)
      : point(p),
        rate(r),
        separation(sep),
        radius(sep / mean_rate(min_r.log, min_r.rate, r.log, r.rate)) {}

  bool conflicts(const Point2& other, LogRate other_r) const {
    return dist(point, other) * mean_rate(rate.log, rate.rate, other_r.log, other_r.rate) <
           separation;
  }
};

// Integrate exp(log_rate) along one rectangle wall using adaptive bisection on
// sub-segments until each piece has metric length <= 0.25, then place N equal-
// metric-length interior wall sites.
bool build_wall_sites(const ScalarField2D& field, int wall_id, std::vector<Point2>& out_points,
                      std::vector<ScalarSample2D>& out_samples) {
  const int fixed_axis = wall_id / 2;
  const int free_axis = 1 - fixed_axis;
  const double fixed_val = (wall_id % 2 == 0) ? field.low[fixed_axis] : field.high[fixed_axis];
  const double a = field.low[free_axis];
  const double b = field.high[free_axis];

  auto point_at = [&](double t) {
    Point2 p;
    p[fixed_axis] = fixed_val;
    p[free_axis] = t == 0.0 ? a : (t == 1.0 ? b : std::lerp(a, b, t));
    return p;
  };

  struct Seg {
    double s0, s1;
    double lr0, lr1;
    std::int32_t cell;
    double mass;
  };
  ScalarSample2D s_start{}, s_end{};
  if (!field.sample(point_at(0.0), s_start) || !field.sample(point_at(1.0), s_end)) return false;

  std::vector<Seg> segments;
  segments.reserve(256);
  std::vector<std::array<double, 4>> stack;
  std::vector<std::int32_t> cell_stack;
  stack.push_back({0.0, 1.0, s_start.log_rate, s_end.log_rate});
  cell_stack.push_back(s_start.cell);
  const double span = b - a;
  constexpr std::size_t kMaxWallSubsegs = 16'384;
  while (!stack.empty()) {
    const auto cur = stack.back();
    const std::int32_t hint = cell_stack.back();
    stack.pop_back();
    cell_stack.pop_back();
    const double s0 = cur[0], s1 = cur[1], lr0 = cur[2], lr1 = cur[3];
    const double mid_s = 0.5 * (s0 + s1);
    ScalarSample2D s_mid{};
    s_mid.cell = hint;
    if (!field.sample(point_at(mid_s), s_mid)) return false;
    const double expected = 0.5 * (lr0 + lr1);
    const double ds = (s1 - s0) * span;
    const double r0 = deterministic::exp(lr0);
    const double r1 = deterministic::exp(lr1);
    const double mass = ds * mean_rate(lr0, r0, lr1, r1);
    const bool flat = std::abs(s_mid.log_rate - expected) <= 0.015;
    if ((mass <= 0.25 && flat) || (s1 - s0) <= 0x1p-18 ||
        segments.size() + stack.size() >= kMaxWallSubsegs) {
      segments.push_back({s0, s1, lr0, lr1, s_mid.cell, mass});
    } else {
      stack.push_back({mid_s, s1, s_mid.log_rate, lr1});
      cell_stack.push_back(s_mid.cell);
      stack.push_back({s0, mid_s, lr0, s_mid.log_rate});
      cell_stack.push_back(hint);
    }
  }
  double total_mass = 0.0;
  for (const auto& seg : segments) total_mass += seg.mass;
  if (!positive_finite(total_mass)) return false;
  std::size_t n = std::max<std::size_t>(1, round_even(total_mass));
  if (total_mass / static_cast<double>(n) > 1.35) ++n;
  if (n > 65'536) return false;
  if (n <= 1) return true;

  const double step = total_mass / static_cast<double>(n);
  std::size_t seg_idx = 0;
  double prefix = 0.0;
  for (std::size_t k = 1; k < n; ++k) {
    const double target = step * static_cast<double>(k);
    while (seg_idx + 1 < segments.size() && prefix + segments[seg_idx].mass < target) {
      prefix += segments[seg_idx].mass;
      ++seg_idx;
    }
    const auto& seg = segments[seg_idx];
    const double frac = seg.mass > 0.0 ? std::clamp((target - prefix) / seg.mass, 0.0, 1.0) : 0.5;
    const double local_t = metric_fraction(seg.lr1 - seg.lr0, frac);
    const double s = std::lerp(seg.s0, seg.s1, local_t);
    Point2 p = point_at(s);
    if (p[free_axis] <= a || p[free_axis] >= b) continue;
    ScalarSample2D sample{};
    sample.cell = seg.cell;
    if (!field.sample(p, sample)) return false;
    out_points.push_back(p);
    out_samples.push_back(sample);
  }
  return true;
}

bool seed_scalar_box_2d(const ScalarField2D& field, ScalarSites2D& sites) {
  sites.points.clear();
  sites.samples.clear();
  sites.cells.clear();
  const double lx = field.high[0] - field.low[0];
  const double ly = field.high[1] - field.low[1];
  if (!positive_finite(lx) || !positive_finite(ly)) return false;

  const double min_rate = deterministic::exp(field.minimum_log_rate);
  if (!positive_finite(min_rate)) return false;
  std::size_t nx = std::clamp<std::size_t>(round_even(lx * min_rate), 1, 256);
  std::size_t ny = std::clamp<std::size_t>(round_even(ly * min_rate), 1, 256);
  while (nx * ny > 65'536) {
    if (nx >= ny && nx > 1) nx = (nx + 1) / 2;
    else if (ny > 1) ny = (ny + 1) / 2;
    else break;
  }

  const std::size_t v_init = (nx + 1) * (ny + 1);
  if (v_init > kMaxScalar2DSites) return false;
  sites.points.resize(v_init);
  sites.samples.resize(v_init);
  std::vector<LogRate> rates(v_init);
  for (std::size_t j = 0; j <= ny; ++j) {
    const double ty = static_cast<double>(j) / static_cast<double>(ny);
    const double y = j == 0 ? field.low[1] : (j == ny ? field.high[1] : std::lerp(field.low[1], field.high[1], ty));
    for (std::size_t i = 0; i <= nx; ++i) {
      const double tx = static_cast<double>(i) / static_cast<double>(nx);
      const double x = i == 0 ? field.low[0] : (i == nx ? field.high[0] : std::lerp(field.low[0], field.high[0], tx));
      const std::size_t row = j * (nx + 1) + i;
      sites.points[row] = {x, y};
      if (i > 0) sites.samples[row].cell = sites.samples[row - 1].cell;
      else if (j > 0) sites.samples[row].cell = sites.samples[row - (nx + 1)].cell;
      if (!field.sample(sites.points[row], sites.samples[row])) return false;
      rates[row] = make_log_rate(sites.samples[row].log_rate);
      if (!positive_finite(rates[row].rate)) return false;
    }
  }
  sites.cells.reserve(2 * nx * ny);
  for (std::size_t j = 0; j < ny; ++j) {
    for (std::size_t i = 0; i < nx; ++i) {
      const auto v00 = static_cast<Row>(j * (nx + 1) + i);
      const auto v10 = static_cast<Row>(v00 + 1);
      const auto v01 = static_cast<Row>(v00 + static_cast<Row>(nx + 1));
      const auto v11 = static_cast<Row>(v01 + 1);
      if (((i + j) & 1U) == 0) {
        sites.cells.push_back({v00, v10, v11});
        sites.cells.push_back({v00, v11, v01});
      } else {
        sites.cells.push_back({v00, v10, v01});
        sites.cells.push_back({v10, v11, v01});
      }
    }
  }

  // Recursive separated edge-midpoint splitting scanning only active (newly created)
  // triangles in each round, avoiding O(T) re-scans of finished background regions.
  const LogRate minimum = make_log_rate(field.minimum_log_rate);
  const double width = separation_width_2d(field, kSeedSeparation);
  if (!positive_finite(width)) return false;
  SeparationGrid2D grid(field.low, width, std::max<std::size_t>(v_init, 1024));
  for (std::size_t v = 0; v < v_init; ++v) grid.insert(sites.points[v], static_cast<Row>(v));

  struct LongEdge {
    EdgeKey key;
    double length;
  };
  struct Accepted {
    EdgeKey key;
    Point2 point;
    ScalarSample2D sample;
    LogRate rate;
  };
  std::vector<LongEdge> proposals;
  std::vector<Accepted> accepted;
  std::vector<Tri2> next_cells;
  std::vector<std::uint8_t> active(sites.cells.size(), 1);
  std::vector<std::uint8_t> next_active;

  for (int round = 0; round < kSplitRounds; ++round) {
    // Collect long edges only from active triangles, deduplicating with a small KeyMap.
    proposals.clear();
    std::size_t active_count = 0;
    for (std::uint8_t a : active) active_count += a;
    if (active_count == 0) break;
    KeyMap seen_long(std::max<std::size_t>(active_count, 64));
    for (std::size_t c = 0; c < sites.cells.size(); ++c) {
      if (!active[c]) continue;
      const auto& tri = sites.cells[c];
      const Row e[3][2] = {{tri[0], tri[1]}, {tri[1], tri[2]}, {tri[2], tri[0]}};
      for (int k = 0; k < 3; ++k) {
        const Row a = e[k][0], b = e[k][1];
        const double len =
            dist(sites.points[a], sites.points[b]) *
            mean_rate(rates[a].log, rates[a].rate, rates[b].log, rates[b].rate);
        if (!positive_finite(len)) return false;
        if (len <= kUpperLength) continue;
        const EdgeKey key = topology::undirected_edge(a, b);
        auto& slot = seen_long.slot(key);
        if (slot < 0) {
          slot = 1;
          proposals.push_back({key, len});
        }
      }
    }
    if (proposals.empty()) break;
    std::sort(proposals.begin(), proposals.end(), [](const LongEdge& a, const LongEdge& b) {
      return a.length == b.length ? a.key < b.key : a.length > b.length;
    });
    accepted.clear();
    const auto base = static_cast<Row>(sites.points.size());
    for (const auto& prop : proposals) {
      const auto a = static_cast<Row>(prop.key >> 32);
      const auto b = static_cast<Row>(static_cast<std::uint32_t>(prop.key));
      const double t =
          static_cast<double>(round_even(1024.0 * metric_fraction(rates[b].log - rates[a].log, 0.5))) /
          1024.0;
      if (t <= 0.0 || t >= 1.0) continue;
      Point2 p{std::lerp(sites.points[a][0], sites.points[b][0], t),
               std::lerp(sites.points[a][1], sites.points[b][1], t)};
      for (int axis = 0; axis < 2; ++axis) {
        if (sites.points[a][axis] == sites.points[b][axis]) p[axis] = sites.points[a][axis];
      }
      ScalarSample2D sample{};
      sample.cell = sites.samples[a].cell;
      if (!field.sample(p, sample)) return false;
      const LogRate lr = make_log_rate(sample.log_rate);
      if (!positive_finite(lr.rate)) return false;
      const SeparationProbe2D probe(p, lr, minimum, kSeedSeparation);
      if (grid.any(p, probe.radius, [&](Row row) {
            if (row < base) return probe.conflicts(sites.points[row], rates[row]);
            const auto& other = accepted[row - base];
            return probe.conflicts(other.point, other.rate);
          })) {
        continue;
      }
      if (sites.points.size() + accepted.size() >= kMaxScalar2DSites) return false;
      const Row idx = base + static_cast<Row>(accepted.size());
      grid.insert(p, idx);
      accepted.push_back({prop.key, p, sample, lr});
    }
    if (accepted.empty()) break;
    KeyMap split_mid(accepted.size());
    for (const auto& item : accepted) {
      const auto row = static_cast<Row>(sites.points.size());
      sites.points.push_back(item.point);
      sites.samples.push_back(item.sample);
      rates.push_back(item.rate);
      split_mid.slot(item.key) = row;
    }
    next_cells.clear();
    next_active.clear();
    next_cells.reserve(sites.cells.size() + 3 * accepted.size());
    next_active.reserve(sites.cells.size() + 3 * accepted.size());
    for (std::size_t c = 0; c < sites.cells.size(); ++c) {
      const auto& tri = sites.cells[c];
      const Row m0 = static_cast<Row>(split_mid.find(topology::undirected_edge(tri[1], tri[2])));
      const Row m1 = static_cast<Row>(split_mid.find(topology::undirected_edge(tri[2], tri[0])));
      const Row m2 = static_cast<Row>(split_mid.find(topology::undirected_edge(tri[0], tri[1])));
      const int mask = (m0 >= 0 ? 1 : 0) | (m1 >= 0 ? 2 : 0) | (m2 >= 0 ? 4 : 0);
      if (mask == 0) {
        next_cells.push_back(tri);
        next_active.push_back(0);
      } else if (mask == 7) {
        next_cells.push_back({tri[0], m2, m1});
        next_cells.push_back({tri[1], m0, m2});
        next_cells.push_back({tri[2], m1, m0});
        next_cells.push_back({m0, m1, m2});
        next_active.insert(next_active.end(), 4, 1);
      } else if (std::popcount(static_cast<unsigned>(mask)) == 1) {
        const int rot = (mask == 1) ? 0 : (mask == 2 ? 1 : 2);
        const Row v0 = tri[rot], v1 = tri[(rot + 1) % 3], v2 = tri[(rot + 2) % 3];
        const Row m = (mask == 1) ? m0 : (mask == 2 ? m1 : m2);
        next_cells.push_back({v0, v1, m});
        next_cells.push_back({v0, m, v2});
        next_active.insert(next_active.end(), 2, 1);
      } else {
        const int rot = (mask == 6) ? 0 : (mask == 5 ? 1 : 2);
        const Row v0 = tri[rot], v1 = tri[(rot + 1) % 3], v2 = tri[(rot + 2) % 3];
        const Row ma = (rot == 0) ? m2 : (rot == 1 ? m0 : m1);
        const Row mb = (rot == 0) ? m1 : (rot == 1 ? m2 : m0);
        next_cells.push_back({v0, ma, mb});
        if (dist2(sites.points[ma], sites.points[v2]) <=
            dist2(sites.points[mb], sites.points[v1])) {
          next_cells.push_back({ma, v1, v2});
          next_cells.push_back({ma, v2, mb});
        } else {
          next_cells.push_back({ma, v1, mb});
          next_cells.push_back({mb, v1, v2});
        }
        next_active.insert(next_active.end(), 3, 1);
      }
      if (next_cells.size() > kMaxScalar2DCells) return false;
    }
    sites.cells.swap(next_cells);
    active.swap(next_active);
  }

  // Rebuild the 4 rectangle walls in exact integrated metric length, keeping
  // the 4 corners first and filtering interior sites too close to wall sites.
  std::vector<Point2> clean_points;
  std::vector<ScalarSample2D> clean_samples;
  clean_points.reserve(sites.points.size() + 256);
  clean_samples.reserve(sites.points.size() + 256);
  const std::array<Point2, 4> corners{{
      {field.low[0], field.low[1]},
      {field.high[0], field.low[1]},
      {field.high[0], field.high[1]},
      {field.low[0], field.high[1]},
  }};
  for (const auto& c : corners) {
    ScalarSample2D s{};
    if (!field.sample(c, s)) return false;
    clean_points.push_back(c);
    clean_samples.push_back(s);
  }
  for (int w = 0; w < 4; ++w) {
    if (!build_wall_sites(field, w, clean_points, clean_samples)) return false;
  }
  const std::size_t wall_count = clean_points.size();
  std::vector<LogRate> wall_rates(wall_count);
  for (std::size_t i = 0; i < wall_count; ++i) {
    wall_rates[i] = make_log_rate(clean_samples[i].log_rate);
  }
  constexpr double kWallClearance = 0.55;
  const double clear_width = separation_width_2d(field, kWallClearance);
  SeparationGrid2D wall_grid(field.low, clear_width, std::max<std::size_t>(wall_count, 64));
  for (std::size_t i = 0; i < wall_count; ++i) {
    wall_grid.insert(clean_points[i], static_cast<Row>(i));
  }
  for (std::size_t i = 0; i < sites.points.size(); ++i) {
    if (wall_mask(field, sites.points[i]) != 0) continue;
    const SeparationProbe2D probe(sites.points[i], rates[i], minimum, kWallClearance);
    if (wall_grid.any(sites.points[i], probe.radius, [&](Row w) {
          return probe.conflicts(clean_points[w], wall_rates[w]);
        })) {
      continue;
    }
    clean_points.push_back(sites.points[i]);
    clean_samples.push_back(sites.samples[i]);
  }
  sites.points.swap(clean_points);
  sites.samples.swap(clean_samples);
  sites.cells.clear();
  return sites.points.size() >= 4 && sites.points.size() <= kMaxScalar2DSites;
}

bool triangulate_sites_2d(ScalarSites2D& sites, bool canonical) {
  const std::size_t count = sites.points.size();
  if (count < 4 || count > kMaxScalar2DSites) return false;
  std::vector<double> coords(2 * count);
  for (std::size_t v = 0; v < count; ++v) {
    coords[2 * v] = sites.points[v][0];
    coords[2 * v + 1] = sites.points[v][1];
  }
  phx_mc_mesh* raw = nullptr;
  const auto status = triangulate_2d(static_cast<std::int64_t>(count), coords.data(), nullptr,
                                     false, canonical, &raw);
  const std::unique_ptr<phx_mc_mesh> mesh(raw);
  if (!topology_accepted_2d(status)) return false;
  const auto cell_count = static_cast<std::size_t>(mesh->cell_count());
  if (cell_count == 0 || cell_count > kMaxScalar2DCells) return false;
  for (std::size_t v = 0; v < count; ++v) {
    if (mesh->vertex_map[v] != static_cast<std::int32_t>(v)) return false;
  }
  sites.cells.resize(cell_count);
  for (std::size_t c = 0; c < cell_count; ++c) {
    sites.cells[c] = {mesh->cells[3 * c], mesh->cells[3 * c + 1], mesh->cells[3 * c + 2]};
  }
  return true;
}

// Run `sweeps` Delaunay CVT moves followed by one final Delaunay triangulation
// of the moved sites (sweeps + 1 triangulations total; canonical only on last).
bool relax_scalar_box_2d(const ScalarField2D& field, ScalarSites2D& sites, int sweeps,
                         bool canonical_end) {
  const std::size_t count = sites.points.size();
  if (sweeps <= 0 || count < 4 || count > kMaxScalar2DSites) return false;
  const GradualNearestScope mode;
  if (!mode.established()) return false;

  Point2 lower, upper;
  for (int axis = 0; axis < 2; ++axis) {
    const double lo = field.low[axis], hi = field.high[axis];
    if (!coordinate_in_domain(lo) || !coordinate_in_domain(hi) || !(lo < hi)) return false;
    const double margin = 1.e-8 * (hi - lo);
    lower[axis] = std::max(lo + margin, std::nextafter(lo, hi));
    upper[axis] = std::min(hi - margin, std::nextafter(hi, lo));
  }

  std::vector<unsigned> walls(count, 0);
  sites.samples.resize(count);
  for (std::size_t v = 0; v < count; ++v) {
    const auto& p = sites.points[v];
    if (!(p[0] >= field.low[0] && p[0] <= field.high[0] &&
          p[1] >= field.low[1] && p[1] <= field.high[1])) {
      return false;
    }
    walls[v] = wall_mask(field, p);
    if (sites.samples[v].cell < 0 && !field.sample(p, sites.samples[v])) return false;
  }

  struct Moment {
    double weight = 0.0;
    Point2 sum{};
  };
  std::vector<Moment> moments(count);
  std::vector<double> coords(2 * count);
  constexpr int kTriEdges[3][2] = {{0, 1}, {1, 2}, {2, 0}};

  for (int sweep = 0; sweep <= sweeps; ++sweep) {
    const bool final_triangulation = (sweep == sweeps);
    for (std::size_t v = 0; v < count; ++v) {
      coords[2 * v] = sites.points[v][0];
      coords[2 * v + 1] = sites.points[v][1];
    }
    phx_mc_mesh* raw = nullptr;
    const auto status = triangulate_2d(static_cast<std::int64_t>(count), coords.data(), nullptr,
                                       false, final_triangulation && canonical_end, &raw);
    const std::unique_ptr<phx_mc_mesh> mesh(raw);
    if (!topology_accepted_2d(status)) return false;
    const auto cell_count = static_cast<std::size_t>(mesh->cell_count());
    if (cell_count == 0 || cell_count > kMaxScalar2DCells) return false;
    for (std::size_t v = 0; v < count; ++v) {
      if (mesh->vertex_map[v] != static_cast<std::int32_t>(v)) return false;
    }
    if (final_triangulation) {
      sites.cells.resize(cell_count);
      for (std::size_t c = 0; c < cell_count; ++c) {
        sites.cells[c] = {mesh->cells[3 * c], mesh->cells[3 * c + 1], mesh->cells[3 * c + 2]};
      }
      return true;
    }

    std::fill(moments.begin(), moments.end(), Moment{});
    for (std::size_t c = 0; c < cell_count; ++c) {
      const Row* ids = mesh->cells.data() + 3 * c;
      const Row a = ids[0], b = ids[1], d = ids[2];
      const unsigned wa = walls[a], wb = walls[b], wd = walls[d];

      if (wa == 0 || wb == 0 || wd == 0) {
        const Point2& p0 = sites.points[a];
        Point2 u{sites.points[b][0] - p0[0], sites.points[b][1] - p0[1]};
        Point2 v{sites.points[d][0] - p0[0], sites.points[d][1] - p0[1]};
        const double extent =
            std::max({std::abs(u[0]), std::abs(u[1]), std::abs(v[0]), std::abs(v[1])});
        if (!positive_finite(extent)) return false;
        int exp_val = 0;
        std::frexp(extent, &exp_val);
        const double inv = std::ldexp(1.0, -exp_val);
        const double scale = std::ldexp(1.0, exp_val);
        u[0] *= inv;
        u[1] *= inv;
        v[0] *= inv;
        v[1] *= inv;
        const double det = u[0] * v[1] - u[1] * v[0];
        if (!(det > 0.0)) return false;
        const double mean_lr =
            (sites.samples[a].log_rate + sites.samples[b].log_rate + sites.samples[d].log_rate) /
            3.0;
        const double h = deterministic::exp(-mean_lr);
        if (!positive_finite(h)) return false;
        const double rate = 1.0 / h;
        const double m_scale = scale * rate;
        const double w = (((det * m_scale) * m_scale) * rate) * rate;
        if (!positive_finite(w)) return false;
        const double uu = 0.5 * (u[0] * u[0] + u[1] * u[1]);
        const double vv = 0.5 * (v[0] * v[0] + v[1] * v[1]);
        Point2 center{(uu * v[1] - u[1] * vv) / det, (u[0] * vv - uu * v[0]) / det};
        if (!std::isfinite(center[0]) || !std::isfinite(center[1])) return false;
        const double limit = h / scale;
        const Point2 centroid{(u[0] + v[0]) / 3.0, (u[1] + v[1]) / 3.0};
        center[0] = centroid[0] + std::clamp(center[0] - centroid[0], -limit, limit);
        center[1] = centroid[1] + std::clamp(center[1] - centroid[1], -limit, limit);
        const std::array<Point2, 3> offsets{{{0.0, 0.0}, u, v}};
        for (int k = 0; k < 3; ++k) {
          if (walls[ids[k]] == 0) {
            moments[ids[k]].weight += w;
            moments[ids[k]].sum[0] += ((center[0] - offsets[k][0]) * scale) * w;
            moments[ids[k]].sum[1] += ((center[1] - offsets[k][1]) * scale) * w;
          }
        }
      }

      for (const auto& e : kTriEdges) {
        const Row p = ids[e[0]], q = ids[e[1]];
        const unsigned common = walls[p] & walls[q];
        if (common == 0) continue;
        const int free_axis = (common & 3u) != 0 ? 1 : 0;
        const double delta = sites.points[q][free_axis] - sites.points[p][free_axis];
        const double mean_lr = 0.5 * (sites.samples[p].log_rate + sites.samples[q].log_rate);
        const double rate = deterministic::exp(mean_lr);
        const double w = ((std::abs(delta) * rate) * rate) * rate;
        if (!positive_finite(w)) return false;
        const double shift = (0.5 * delta) * w;
        if (walls[p] == common) {
          moments[p].weight += w;
          moments[p].sum[free_axis] += shift;
        }
        if (walls[q] == common) {
          moments[q].weight += w;
          moments[q].sum[free_axis] -= shift;
        }
      }
    }

    for (std::size_t v = 0; v < count; ++v) {
      if (std::popcount(walls[v]) >= 2 || !(moments[v].weight > 0.0)) continue;
      Point2 moved = sites.points[v];
      for (int axis = 0; axis < 2; ++axis) {
        if ((walls[v] & (3u << (2 * axis))) != 0) continue;
        const double target = moved[axis] + moments[v].sum[axis] / moments[v].weight;
        if (!std::isfinite(target)) return false;
        moved[axis] = std::clamp(target, lower[axis], upper[axis]);
      }
      ScalarSample2D sample = sites.samples[v];
      if (!field.sample(moved, sample)) return false;
      sites.points[v] = moved;
      sites.samples[v] = sample;
    }
  }
  return true;
}

std::int64_t split_scalar_edges_2d(const ScalarField2D& field, ScalarSites2D& sites,
                                   double separation, double threshold) {
  const std::size_t count = sites.points.size();
  if (count >= kMaxScalar2DSites) return -1;
  std::vector<Edge> edges;
  if (!collect_edges_linear(count, sites.cells, edges)) return -1;
  std::vector<LogRate> rates(count);
  for (std::size_t v = 0; v < count; ++v) {
    rates[v] = make_log_rate(sites.samples[v].log_rate);
    if (!positive_finite(rates[v].rate)) return -1;
  }
  struct Proposal {
    EdgeKey key;
    double length;
  };
  std::vector<Proposal> proposals;
  for (const auto [a, b] : edges) {
    const double len =
        dist(sites.points[a], sites.points[b]) *
        mean_rate(rates[a].log, rates[a].rate, rates[b].log, rates[b].rate);
    if (!positive_finite(len)) return -1;
    if (len > threshold) proposals.push_back({topology::undirected_edge(a, b), len});
  }
  if (proposals.empty()) return 0;
  std::sort(proposals.begin(), proposals.end(), [](const Proposal& a, const Proposal& b) {
    return a.length == b.length ? a.key < b.key : a.length > b.length;
  });
  const LogRate minimum = make_log_rate(field.minimum_log_rate);
  const double width = separation_width_2d(field, separation);
  if (!positive_finite(width)) return -1;
  SeparationGrid2D grid(field.low, width, count);
  for (std::size_t v = 0; v < count; ++v) grid.insert(sites.points[v], static_cast<Row>(v));

  struct Accepted {
    EdgeKey key;
    Point2 point;
    ScalarSample2D sample;
    LogRate rate;
  };
  std::vector<Accepted> accepted;
  const auto first = static_cast<Row>(count);
  for (const auto& prop : proposals) {
    const auto a = static_cast<Row>(prop.key >> 32);
    const auto b = static_cast<Row>(static_cast<std::uint32_t>(prop.key));
    const double t =
        static_cast<double>(round_even(1024.0 * metric_fraction(rates[b].log - rates[a].log, 0.5))) /
        1024.0;
    if (t <= 0.0 || t >= 1.0) continue;
    Point2 p{std::lerp(sites.points[a][0], sites.points[b][0], t),
             std::lerp(sites.points[a][1], sites.points[b][1], t)};
    for (int axis = 0; axis < 2; ++axis) {
      if (sites.points[a][axis] == sites.points[b][axis]) p[axis] = sites.points[a][axis];
    }
    ScalarSample2D sample{};
    sample.cell = sites.samples[a].cell;
    if (!field.sample(p, sample)) return -1;
    const LogRate lr = make_log_rate(sample.log_rate);
    if (!positive_finite(lr.rate)) return -1;
    const SeparationProbe2D probe(p, lr, minimum, separation);
    if (grid.any(p, probe.radius, [&](Row row) {
          if (row < first) return probe.conflicts(sites.points[row], rates[row]);
          const auto& other = accepted[row - first];
          return probe.conflicts(other.point, other.rate);
        })) {
      continue;
    }
    if (count + accepted.size() >= kMaxScalar2DSites) return -1;
    grid.insert(p, first + static_cast<Row>(accepted.size()));
    accepted.push_back({prop.key, p, sample, lr});
  }
  std::sort(accepted.begin(), accepted.end(), [](const Accepted& a, const Accepted& b) {
    return a.key < b.key;
  });
  for (const auto& item : accepted) {
    sites.points.push_back(item.point);
    sites.samples.push_back(item.sample);
  }
  return static_cast<std::int64_t>(accepted.size());
}

std::int64_t thin_short_edges_2d(const ScalarField2D& field, ScalarSites2D& sites,
                                 double threshold) {
  const std::size_t count = sites.points.size();
  std::vector<Edge> edges;
  if (!collect_edges_linear(count, sites.cells, edges)) return -1;
  std::vector<LogRate> rates(count);
  for (std::size_t v = 0; v < count; ++v) rates[v] = make_log_rate(sites.samples[v].log_rate);
  struct ShortEdge {
    EdgeKey key;
    double length;
  };
  std::vector<ShortEdge> short_edges;
  for (const auto [a, b] : edges) {
    const double len =
        dist(sites.points[a], sites.points[b]) *
        mean_rate(rates[a].log, rates[a].rate, rates[b].log, rates[b].rate);
    if (len < threshold) short_edges.push_back({topology::undirected_edge(a, b), len});
  }
  if (short_edges.empty()) return 0;
  std::vector<unsigned> walls(count);
  for (std::size_t v = 0; v < count; ++v) walls[v] = wall_mask(field, sites.points[v]);
  std::sort(short_edges.begin(), short_edges.end(), [](const ShortEdge& a, const ShortEdge& b) {
    return a.length == b.length ? a.key < b.key : a.length < b.length;
  });
  std::vector<std::uint8_t> drop(count, 0);
  std::vector<std::uint8_t> locked(count, 0);
  std::int64_t dropped = 0;
  for (const auto& se : short_edges) {
    const auto a = static_cast<Row>(se.key >> 32);
    const auto b = static_cast<Row>(static_cast<std::uint32_t>(se.key));
    if (drop[a] || drop[b] || locked[a] || locked[b]) continue;
    const int wa = std::popcount(walls[a]);
    const int wb = std::popcount(walls[b]);
    Row victim = -1;
    Row survivor = -1;
    if (wa == 0 && wb == 0) {
      victim = b; survivor = a;
    } else if (wa == 0) {
      victim = a; survivor = b;
    } else if (wb == 0) {
      victim = b; survivor = a;
    } else if (wa == 1 && wb == 1 && walls[a] == walls[b]) {
      victim = b; survivor = a;
    } else if (wa == 1 && wb >= 2) {
      victim = a; survivor = b;
    } else if (wb == 1 && wa >= 2) {
      victim = b; survivor = a;
    }
    if (victim < 0) continue;
    drop[victim] = 1;
    locked[survivor] = 1;
    ++dropped;
  }
  if (dropped == 0) return 0;
  std::vector<Point2> next_points;
  std::vector<ScalarSample2D> next_samples;
  next_points.reserve(count - static_cast<std::size_t>(dropped));
  next_samples.reserve(count - static_cast<std::size_t>(dropped));
  for (std::size_t v = 0; v < count; ++v) {
    if (drop[v]) continue;
    next_points.push_back(sites.points[v]);
    next_samples.push_back(sites.samples[v]);
  }
  sites.points.swap(next_points);
  sites.samples.swap(next_samples);
  return dropped;
}

double long_edge_fraction_2d(const ScalarSites2D& sites, double threshold) {
  std::vector<Edge> edges;
  if (!collect_edges_linear(sites.points.size(), sites.cells, edges)) return -1.0;
  std::vector<LogRate> rates(sites.points.size());
  for (std::size_t v = 0; v < sites.points.size(); ++v) {
    rates[v] = make_log_rate(sites.samples[v].log_rate);
    if (!positive_finite(rates[v].rate)) return -1.0;
  }
  std::size_t longer = 0;
  for (const auto [a, b] : edges) {
    const double len =
        dist(sites.points[a], sites.points[b]) *
        mean_rate(rates[a].log, rates[a].rate, rates[b].log, rates[b].rate);
    if (!positive_finite(len)) return -1.0;
    if (len > threshold) ++longer;
  }
  return static_cast<double>(longer) / static_cast<double>(edges.size());
}

// Local-first augmented-Lagrangian L-BFGS repair enforcing strict 2D metric
// edge band and Euclidean triangle mean-ratio quality floor >= kShapeFloor2D.
class ScalarRepair2D {
 public:
  ScalarRepair2D(const ScalarField2D& field, const ScalarSites2D& sites, Band hard)
      : field_(field),
        sites_(sites),
        hard_(hard),
        target_{hard.low + std::min(0.001, (hard.high - hard.low) / 8.0),
                hard.high - std::min(0.0012, (hard.high - hard.low) / 8.0)} {}

  bool prepare(bool local);
  double evaluate(std::span<const double> x, std::span<double> gradient);
  void increase_penalty();
  void publish(ScalarSites2D& sites) const;
  std::span<const Band> bounds() const { return bounds_; }
  bool sparse() const { return sparse_; }
  bool feasible() const { return feasible_; }
  bool interior() const { return interior_; }

 private:
  struct EdgeData {
    Edge rows;
    double low_shift = 0.0;
    double high_shift = 0.0;
    double distance = 0.0;
    double mean_rate_val = 0.0;
    double length = 0.0;
  };
  struct ShapeData {
    double shift = 0.0;
    double quality = 0.0;
    double inv_squares = 0.0;
  };

  bool select_dofs(std::span<const std::uint8_t> selected);
  bool update_edge(const Point2& pa, double lra, double ra, const Point2& pb, double lrb,
                   double rb, EdgeData& edge) const;
  bool update_shape(const Point2& a, const Point2& b, const Point2& c, double limit,
                    ShapeData& shape) const;

  const ScalarField2D& field_;
  const ScalarSites2D& sites_;
  Band hard_;
  Band target_;
  std::vector<Point2> points_;
  std::vector<double> log_rates_;
  std::vector<double> rates_;
  std::vector<std::uint8_t> changed_;
  std::vector<ScalarSample2D> samples_;
  std::vector<std::array<Row, 2>> dofs_;
  std::vector<double> sizes_;
  std::vector<Row> active_nodes_;
  std::vector<Row> active_edges_;
  std::vector<Row> active_cells_;
  std::vector<Band> bounds_;
  std::vector<EdgeData> edges_;
  std::vector<ShapeData> shapes_;
  double edge_penalty_ = 16.0;
  double shape_penalty_ = 32.0;
  bool sparse_ = false;
  bool feasible_ = false;
  bool interior_ = false;
};

bool ScalarRepair2D::update_edge(const Point2& pa, double lra, double ra, const Point2& pb,
                                 double lrb, double rb, EdgeData& edge) const {
  edge.distance = dist(pa, pb);
  edge.mean_rate_val = mean_rate(lra, ra, lrb, rb);
  edge.length = edge.distance * edge.mean_rate_val;
  return positive_finite(edge.length);
}

bool ScalarRepair2D::update_shape(const Point2& a, const Point2& b, const Point2& c, double limit,
                                  ShapeData& shape) const {
  const double ux = b[0] - a[0], uy = b[1] - a[1];
  const double vx = c[0] - a[0], vy = c[1] - a[1];
  const double wx = c[0] - b[0], wy = c[1] - b[1];
  const double det = ux * vy - uy * vx;
  const double squares = (ux * ux + uy * uy) + (vx * vx + vy * vy) + (wx * wx + wy * wy);
  if (!positive_finite(squares) || !std::isfinite(det)) return false;
  const double q = kSqrt3x2 * det / squares;
  if (!std::isfinite(q)) return false;
  shape.quality = q > limit * (1.0 + 0x1p-40) ? limit : q;
  shape.inv_squares = 1.0 / squares;
  return true;
}

bool ScalarRepair2D::select_dofs(std::span<const std::uint8_t> selected) {
  bounds_.clear();
  for (std::size_t v = 0; v < points_.size(); ++v) {
    dofs_[v] = {-1, -1};
    if (!selected[v]) continue;
    const auto& p = sites_.points[v];
    const double h = sizes_[v];
    const double shift = 0.5 * h;
    for (int axis = 0; axis < 2; ++axis) {
      const double lo = field_.low[axis], hi = field_.high[axis];
      if (p[axis] == lo || p[axis] == hi) continue;
      const double margin = 1.e-8 * (hi - lo);
      const double min_x = std::max(lo + margin, std::nextafter(lo, hi));
      const double max_x = std::min(hi - margin, std::nextafter(hi, lo));
      const double low_s = (std::max(min_x, p[axis] - shift) - p[axis]) / h;
      const double high_s = (std::min(max_x, p[axis] + shift) - p[axis]) / h;
      if (!(low_s <= 0.0 && high_s >= 0.0)) return false;
      dofs_[v][axis] = static_cast<Row>(bounds_.size());
      bounds_.push_back({low_s, high_s});
    }
  }
  return true;
}

bool ScalarRepair2D::prepare(bool local) {
  const std::size_t n = sites_.points.size();
  const std::size_t m = sites_.cells.size();
  if (n < 4 || m == 0 || sites_.samples.size() != n) return false;

  points_ = sites_.points;
  samples_ = sites_.samples;
  log_rates_.resize(n);
  rates_.resize(n);
  changed_.assign(n, 0);
  dofs_.assign(n, {-1, -1});
  sizes_.resize(n);
  for (std::size_t v = 0; v < n; ++v) {
    log_rates_[v] = samples_[v].log_rate;
    rates_[v] = deterministic::exp(log_rates_[v]);
    sizes_[v] = 1.0 / rates_[v];
    if (!positive_finite(rates_[v]) || !positive_finite(sizes_[v])) return false;
  }

  std::vector<Edge> raw_edges;
  if (!collect_edges_linear(n, sites_.cells, raw_edges)) return false;
  edges_.resize(raw_edges.size());
  for (std::size_t i = 0; i < raw_edges.size(); ++i) {
    edges_[i] = {raw_edges[i]};
    const auto [a, b] = raw_edges[i];
    if (!update_edge(points_[a], log_rates_[a], rates_[a], points_[b], log_rates_[b], rates_[b],
                     edges_[i])) {
      return false;
    }
  }
  shapes_.assign(m, {});
  for (std::size_t c = 0; c < m; ++c) {
    const auto [a, b, d] = sites_.cells[c];
    if (!update_shape(points_[a], points_[b], points_[d], kShapeTarget2D, shapes_[c])) return false;
  }

  std::vector<std::uint8_t> selected(n, local ? 0 : 1);
  if (local) {
    std::vector<std::uint8_t> seed(n, 0);
    for (const auto& e : edges_) {
      if (e.length < target_.low || e.length > target_.high) {
        seed[e.rows[0]] = 1;
        seed[e.rows[1]] = 1;
      }
    }
    for (std::size_t c = 0; c < m; ++c) {
      if (shapes_[c].quality < kShapeTarget2D) {
        for (const Row v : sites_.cells[c]) seed[v] = 1;
      }
    }
    selected = seed;
    for (int ring = 0; ring < 2; ++ring) {
      auto next = selected;
      for (const auto& e : edges_) {
        if (selected[e.rows[0]] || selected[e.rows[1]]) {
          next[e.rows[0]] = 1;
          next[e.rows[1]] = 1;
        }
      }
      selected.swap(next);
    }
    std::size_t chosen = 0;
    for (std::uint8_t s : selected) chosen += s;
    if (chosen * 4 >= 3 * n) {
      std::fill(selected.begin(), selected.end(), 1);
      local = false;
    }
  }
  sparse_ = local;
  if (!select_dofs(selected)) return false;

  active_nodes_.clear();
  active_edges_.clear();
  active_cells_.clear();
  if (sparse_) {
    for (std::size_t v = 0; v < n; ++v) {
      if (dofs_[v][0] >= 0 || dofs_[v][1] >= 0) active_nodes_.push_back(static_cast<Row>(v));
    }
    for (std::size_t i = 0; i < edges_.size(); ++i) {
      const auto [a, b] = edges_[i].rows;
      if (selected[a] || selected[b]) active_edges_.push_back(static_cast<Row>(i));
    }
    for (std::size_t c = 0; c < m; ++c) {
      const auto [a, b, d] = sites_.cells[c];
      if (selected[a] || selected[b] || selected[d]) active_cells_.push_back(static_cast<Row>(c));
    }
  }
  return true;
}

double ScalarRepair2D::evaluate(std::span<const double> x, std::span<double> gradient) {
  std::fill(gradient.begin(), gradient.end(), 0.0);
  feasible_ = true;
  interior_ = true;

  auto update_node = [&](std::size_t v) -> bool {
    const auto [dx, dy] = dofs_[v];
    if (dx < 0 && dy < 0) return true;
    Point2 p = sites_.points[v];
    const double h = sizes_[v];
    if (dx >= 0) p[0] += h * x[dx];
    if (dy >= 0) p[1] += h * x[dy];
    if (p == points_[v]) {
      changed_[v] = 0;
      return true;
    }
    ScalarSample2D sample = samples_[v];
    if (!field_.sample(p, sample)) return false;
    const double rate = deterministic::exp(sample.log_rate);
    if (!positive_finite(rate)) return false;
    points_[v] = p;
    samples_[v] = sample;
    log_rates_[v] = sample.log_rate;
    rates_[v] = rate;
    changed_[v] = 1;
    return true;
  };
  if (sparse_) {
    for (const Row v : active_nodes_) {
      if (!update_node(static_cast<std::size_t>(v))) return kInvalidMerit;
    }
  } else {
    for (std::size_t v = 0; v < points_.size(); ++v) {
      if (!update_node(v)) return kInvalidMerit;
    }
  }

  double merit = 0.0;
  auto add_grad = [&](Row v, double gx, double gy) {
    const double h = sizes_[v];
    const auto [dx, dy] = dofs_[v];
    if (dx >= 0) gradient[dx] += h * gx;
    if (dy >= 0) gradient[dy] += h * gy;
  };

  auto visit_edge = [&](std::size_t idx) -> bool {
    auto& e = edges_[idx];
    const auto [a, b] = e.rows;
    if (changed_[a] || changed_[b]) {
      if (!update_edge(points_[a], log_rates_[a], rates_[a], points_[b], log_rates_[b], rates_[b],
                       e)) {
        return false;
      }
    }
    if (e.length < hard_.low || e.length > hard_.high) feasible_ = false;
    if (e.length < target_.low || e.length > target_.high) interior_ = false;
    const double low_act = std::max(0.0, (target_.low - e.length) + e.low_shift);
    const double high_act = std::max(0.0, (e.length - target_.high) + e.high_shift);
    const double factor = edge_penalty_ * (high_act - low_act);
    if (low_act > 0.0) merit += 0.5 * edge_penalty_ * low_act * low_act;
    if (high_act > 0.0) merit += 0.5 * edge_penalty_ * high_act * high_act;
    if (factor != 0.0) {
      const double ux = (points_[b][0] - points_[a][0]) / e.distance;
      const double uy = (points_[b][1] - points_[a][1]) / e.distance;
      const double geom = factor * e.mean_rate_val;
      const auto [da, db] =
          mean_rate_gradient(log_rates_[a], rates_[a], log_rates_[b], rates_[b], e.mean_rate_val);
      const double fa = factor * e.distance * da;
      const double fb = factor * e.distance * db;
      add_grad(a, -geom * ux + fa * samples_[a].gradient[0],
               -geom * uy + fa * samples_[a].gradient[1]);
      add_grad(b, geom * ux + fb * samples_[b].gradient[0],
               geom * uy + fb * samples_[b].gradient[1]);
    }
    return true;
  };
  if (sparse_) {
    for (const Row idx : active_edges_) {
      if (!visit_edge(static_cast<std::size_t>(idx))) return kInvalidMerit;
    }
  } else {
    for (std::size_t idx = 0; idx < edges_.size(); ++idx) {
      if (!visit_edge(idx)) return kInvalidMerit;
    }
  }

  auto visit_cell = [&](std::size_t c) -> bool {
    const auto [a, b, d] = sites_.cells[c];
    auto& shape = shapes_[c];
    const double limit = kShapeTarget2D + shape.shift;
    if (changed_[a] || changed_[b] || changed_[d]) {
      if (!update_shape(points_[a], points_[b], points_[d], limit, shape)) return false;
    }
    if (!(shape.quality > 0.0)) return false;
    if (shape.quality < kShapeFloor2D) feasible_ = false;
    if (shape.quality < kShapeTarget2D) interior_ = false;
    const double act = limit - shape.quality;
    if (act > 0.0) {
      merit += 0.5 * shape_penalty_ * act * act;
      const double factor = -shape_penalty_ * act;
      const double ux = points_[b][0] - points_[a][0], uy = points_[b][1] - points_[a][1];
      const double vx = points_[d][0] - points_[a][0], vy = points_[d][1] - points_[a][1];
      const double wx = points_[d][0] - points_[b][0], wy = points_[d][1] - points_[b][1];
      const double inv_s = shape.inv_squares;
      const double c_det = kSqrt3x2 * inv_s;
      const double c_sq = 2.0 * shape.quality * inv_s;
      const double gbx = factor * (c_det * vy - c_sq * (ux - wx));
      const double gby = factor * (-c_det * vx - c_sq * (uy - wy));
      const double gdx = factor * (-c_det * uy - c_sq * (vx + wx));
      const double gdy = factor * (c_det * ux - c_sq * (vy + wy));
      add_grad(a, -(gbx + gdx), -(gby + gdy));
      add_grad(b, gbx, gby);
      add_grad(d, gdx, gdy);
    }
    return true;
  };
  if (sparse_) {
    for (const Row c : active_cells_) {
      if (!visit_cell(static_cast<std::size_t>(c))) return kInvalidMerit;
    }
  } else {
    for (std::size_t c = 0; c < sites_.cells.size(); ++c) {
      if (!visit_cell(c)) return kInvalidMerit;
    }
  }
  return std::isfinite(merit) ? merit : kInvalidMerit;
}

void ScalarRepair2D::increase_penalty() {
  for (auto& e : edges_) {
    e.low_shift = std::max(0.0, e.low_shift + (target_.low - e.length)) / kPenaltyGrowth;
    e.high_shift = std::max(0.0, e.high_shift + (e.length - target_.high)) / kPenaltyGrowth;
  }
  for (auto& s : shapes_) {
    s.shift = std::max(0.0, s.shift + (kShapeTarget2D - s.quality)) / kPenaltyGrowth;
  }
  edge_penalty_ *= kPenaltyGrowth;
  shape_penalty_ *= kPenaltyGrowth;
}

void ScalarRepair2D::publish(ScalarSites2D& sites) const {
  sites.points = points_;
  sites.samples = samples_;
}

bool correct_scalar_box_2d(const ScalarField2D& field, ScalarSites2D& sites, double lower) {
  const Band hard{std::max(lower, kLowerLength) + 2.e-6, kUpperLength - 2.e-6};
  if (!(hard.low < hard.high)) return false;

  for (int attempt = 0; attempt < 2; ++attempt) {
    const bool local = attempt == 0;
    ScalarRepair2D repair(field, sites, hard);
    if (!repair.prepare(local)) return false;
    const std::size_t dof_count = repair.bounds().size();
    std::vector<double> x(dof_count, 0.0);
    std::vector<double> grad(dof_count, 0.0);
    const double initial = repair.evaluate(x, grad);
    if (!std::isfinite(initial)) {
      if (local) continue;
      return false;
    }
    if (repair.feasible() && repair.interior()) {
      repair.publish(sites);
      return true;
    }
    if (dof_count == 0) return repair.feasible();
    std::vector<double> work(kWorkBuffers * dof_count);
    for (int stage = 0; stage < kMaxStages; ++stage) {
      const auto result = detail::projected_bounded_lbfgs(
          x, repair.bounds(), 0.25, kStepsPerStage, 1.e-10, work,
          [&](std::span<const double> trial, std::span<double> g) {
            return repair.evaluate(trial, g);
          },
          [&]() { return repair.interior(); });
      if (result.status == detail::BoundedLbfgsStatus::kInvalidObjective) break;
      if (repair.feasible()) {
        repair.publish(sites);
        return true;
      }
      repair.increase_penalty();
    }
    if (!repair.sparse()) return false;
  }
  return false;
}

}  // namespace

bool ScalarField2D::prepare(std::span<const double> points, std::span<const std::int64_t> cells,
                            std::span<const double> log_rates, const Point2& minimum,
                            const Point2& maximum) {
  const std::size_t vertices = points.size() / 2;
  const std::size_t tri_count = cells.size() / 3;
  if (vertices < 3 || tri_count == 0 || tri_count > kMaxScalar2DSourceCells ||
      2 * vertices != points.size() || 3 * tri_count != cells.size() ||
      log_rates.size() != vertices) {
    return false;
  }
  low = minimum;
  high = maximum;
  if (!(low[0] < high[0] && low[1] < high[1])) return false;

  minimum_log_rate = std::numeric_limits<double>::infinity();
  maximum_log_rate = -std::numeric_limits<double>::infinity();
  std::vector<double> vertex_m(vertices);
  for (std::size_t v = 0; v < vertices; ++v) {
    if (!std::isfinite(log_rates[v])) return false;
    minimum_log_rate = std::min(minimum_log_rate, log_rates[v]);
    maximum_log_rate = std::max(maximum_log_rate, log_rates[v]);
    vertex_m[v] = deterministic::exp(2.0 * log_rates[v]);
  }

  std::vector<double> box_low(3 * tri_count, 0.0);
  std::vector<double> box_high(3 * tri_count, 0.0);
  double estimated_work = 0.0;
  constexpr double kDetSqFloor = (64.0 * std::numeric_limits<double>::epsilon()) *
                                 (64.0 * std::numeric_limits<double>::epsilon());
  for (std::size_t t = 0; t < tri_count; ++t) {
    const auto i0 = static_cast<std::size_t>(cells[3 * t]);
    const auto i1 = static_cast<std::size_t>(cells[3 * t + 1]);
    const auto i2 = static_cast<std::size_t>(cells[3 * t + 2]);
    if (i0 >= vertices || i1 >= vertices || i2 >= vertices) return false;
    const double x0 = points[2 * i0], y0 = points[2 * i0 + 1];
    const double x1 = points[2 * i1], y1 = points[2 * i1 + 1];
    const double x2 = points[2 * i2], y2 = points[2 * i2 + 1];
    const double ux = x1 - x0, uy = y1 - y0;
    const double vx = x2 - x0, vy = y2 - y0;
    const double det = ux * vy - uy * vx;
    const double u2v2 = (ux * ux + uy * uy) * (vx * vx + vy * vy);
    if (!(det > 0.0) || !(det * det > kDetSqFloor * u2v2)) return false;
    const double lx = std::min({x0, x1, x2}), hx = std::max({x0, x1, x2});
    const double ly = std::min({y0, y1, y2}), hy = std::max({y0, y1, y2});
    if (lx < low[0] || hx > high[0] || ly < low[1] || hy > high[1]) return false;
    box_low[3 * t] = lx;
    box_low[3 * t + 1] = ly;
    box_high[3 * t] = hx;
    box_high[3 * t + 1] = hy;
    estimated_work += (0.5 / 3.0) * det * (vertex_m[i0] + vertex_m[i1] + vertex_m[i2]);
  }
  if (!(estimated_work > 0.0 && 4.0 * estimated_work <= static_cast<double>(kMaxScalar2DCells))) {
    return false;
  }
  points_ = points;
  cells_ = cells;
  log_rates_ = log_rates;
  tris_.resize(tri_count);
  ready_.assign(tri_count, 0);
  boxes_.prepare(box_low, box_high, 0b011, 64 * tri_count);
  return boxes_.indexed();
}

bool ScalarField2D::ensure_element(std::size_t id) const {
  if (ready_[id] != 0) return true;
  Point2 p[3];
  double lr[3];
  for (int k = 0; k < 3; ++k) {
    const auto idx = static_cast<std::size_t>(cells_[3 * id + k]);
    p[k] = {points_[2 * idx], points_[2 * idx + 1]};
    lr[k] = log_rates_[idx];
  }
  const double ux = p[1][0] - p[0][0], uy = p[1][1] - p[0][1];
  const double vx = p[2][0] - p[0][0], vy = p[2][1] - p[0][1];
  const double det = ux * vy - uy * vx;
  if (!(det > 0.0) || !std::isfinite(det)) return false;
  auto& tri = tris_[id];
  tri.origin = p[0];
  tri.log_rate = lr[0];
  tri.gradients[1] = {vy / det, -vx / det};
  tri.gradients[2] = {-uy / det, ux / det};
  tri.gradients[0] = {-tri.gradients[1][0] - tri.gradients[2][0],
                      -tri.gradients[1][1] - tri.gradients[2][1]};
  tri.log_gradient = {
      (lr[1] - lr[0]) * tri.gradients[1][0] + (lr[2] - lr[0]) * tri.gradients[2][0],
      (lr[1] - lr[0]) * tri.gradients[1][1] + (lr[2] - lr[0]) * tri.gradients[2][1],
  };
  ready_[id] = 1;
  return true;
}

std::span<const std::int64_t> ScalarField2D::box_candidates(const Point2& x) const noexcept {
  const double target[3] = {x[0], x[1], 0.0};
  return boxes_.candidates(target);
}

bool ScalarField2D::sample(const Point2& x, ScalarSample2D& out) const {
  if (!coordinate_in_domain(x[0]) || !coordinate_in_domain(x[1])) return false;
  auto try_cell = [&](std::size_t id) -> bool {
    if (!ensure_element(id)) return false;
    const auto& tri = tris_[id];
    const double dx = x[0] - tri.origin[0], dy = x[1] - tri.origin[1];
    for (int k = 0; k < 3; ++k) {
      const double b =
          (k == 0 ? 1.0 : 0.0) + tri.gradients[k][0] * dx + tri.gradients[k][1] * dy;
      if (b < -1.e-10) return false;
    }
    const double val = tri.log_rate + tri.log_gradient[0] * dx + tri.log_gradient[1] * dy;
    if (!std::isfinite(val)) return false;
    out = {static_cast<std::int32_t>(id), val, tri.log_gradient};
    return true;
  };
  if (out.cell >= 0 && static_cast<std::size_t>(out.cell) < tris_.size() &&
      try_cell(static_cast<std::size_t>(out.cell))) {
    return true;
  }
  for (const auto cell : box_candidates(x)) {
    if (try_cell(static_cast<std::size_t>(cell))) return true;
  }
  return false;
}

std::int32_t ScalarField2D::locate_exact(const Point2& x, std::int32_t hint) const {
  auto contains = [&](std::size_t id) -> bool {
    const auto r0 = static_cast<std::size_t>(cells_[3 * id]);
    const auto r1 = static_cast<std::size_t>(cells_[3 * id + 1]);
    const auto r2 = static_cast<std::size_t>(cells_[3 * id + 2]);
    const double* p0 = points_.data() + 2 * r0;
    const double* p1 = points_.data() + 2 * r1;
    const double* p2 = points_.data() + 2 * r2;
    return orient2d(p1, p2, x.data()) >= 0 && orient2d(p2, p0, x.data()) >= 0 &&
           orient2d(p0, p1, x.data()) >= 0;
  };
  if (hint >= 0 && static_cast<std::size_t>(hint) < tris_.size() &&
      contains(static_cast<std::size_t>(hint))) {
    return hint;
  }
  for (const auto cell : box_candidates(x)) {
    if (contains(static_cast<std::size_t>(cell))) return static_cast<std::int32_t>(cell);
  }
  return -1;
}

bool propose_scalar_box_2d(const ScalarField2D& field, double lower, ScalarSites2D& sites) {
  constexpr int kSmoothingSteps = 3;
  if (!seed_scalar_box_2d(field, sites) ||
      !relax_scalar_box_2d(field, sites, kSmoothingSteps, true)) {
    return false;
  }
  auto thin = [&]() -> bool {
    const auto removed = thin_short_edges_2d(field, sites, 0.56);
    if (removed < 0) return false;
    if (removed == 0) return true;
    return relax_scalar_box_2d(field, sites, 1, true);
  };
  auto densify = [&]() -> bool {
    for (int round = 0; round < 2; ++round) {
      const auto added = split_scalar_edges_2d(field, sites, kDensifySeparation, kUpperLength);
      if (added < 0) return false;
      if (added == 0) break;
      if (!relax_scalar_box_2d(field, sites, 1, true)) return false;
    }
    return thin();
  };

  if (!thin()) return false;
  const double tail = long_edge_fraction_2d(sites, 1.55);
  if (tail < 0.0) return false;
  const bool densified = tail > 0.005;
  if (densified && !densify()) return false;
  if (correct_scalar_box_2d(field, sites, lower)) return true;
  if (!densified) {
    if (!densify()) return false;
    if (correct_scalar_box_2d(field, sites, lower)) return true;
  }
  if (!triangulate_sites_2d(sites, true) || !thin()) return false;
  return correct_scalar_box_2d(field, sites, lower);
}

}  // namespace phx::mc::metric
