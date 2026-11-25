// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "metric_scalar_box.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "key_map.hpp"
#include "metric_box_rate.hpp"
#include "metric_float.hpp"
#include "mesh.hpp"
#include "topology_keys.hpp"

namespace phx::mc::metric {
namespace {

constexpr double kSeparation = 0.7072;  // Just inside the strict unit band.
constexpr int kSplitRounds = 16;
constexpr std::array<std::array<int, 2>, 6> kEdges{{
    {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3},
}};
constexpr std::array<std::array<int, 2>, 3> kFixedAxes{{{1, 2}, {0, 2}, {0, 1}}};

struct LogRate {
  double log;
  double rate;
};

LogRate log_rate(double log) { return {log, deterministic::exp(log)}; }

// Average of exp(lerp(a.log, b.log, t)). Factoring out the higher
// rate keeps expm1 finite even for a large log difference.
double mean_rate(LogRate a, LogRate b) {
  if (a.log < b.log) std::swap(a, b);
  const double delta = b.log - a.log;
  return std::abs(delta) < 1.e-4 ? a.rate * integrated_exponential(delta) : (b.rate - a.rate) / delta;
}

double metric_fraction(double delta, double mass_fraction) {
  if (delta == 0.0 || mass_fraction == 0.0 || mass_fraction == 1.0) return mass_fraction;
  // The continuous limit also avoids dividing rounded subnormal masses.
  if (std::abs(delta) < 1e-8) {
    return mass_fraction + 0.5 * delta * mass_fraction * (1.0 - mass_fraction);
  }
  if (delta < 700.0) return deterministic::log1p(mass_fraction * (delta * integrated_exponential(delta))) / delta;
  const double x = delta + deterministic::log(mass_fraction);
  return (x > 0.0 ? x + deterministic::log1p((1.0 - mass_fraction) * deterministic::exp(-x))
                  : deterministic::log1p(deterministic::exp(x) - mass_fraction)) / delta;
}

// Nonnegative, bounded arguments only; ties go to even independently of the
// ambient rounding mode, as in the dyadic midpoint proposal.
std::size_t round_even(double x) {
  const auto floor = static_cast<std::size_t>(x);
  const double fraction = x - static_cast<double>(floor);
  return floor + (fraction > 0.5 || (fraction == 0.5 && (floor & 1U) != 0));
}

double distance_squared(const BoxPoint& a, const BoxPoint& b) {
  const double x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
  return x * x + y * y + z * z;
}

bool normal_squared_distance(double squared) {
  return squared >= std::numeric_limits<double>::min() && std::isfinite(squared);
}

double distance(const BoxPoint& a, const BoxPoint& b) {
  const double squared = distance_squared(a, b);
  if (normal_squared_distance(squared)) return std::sqrt(squared);
  const auto d = geometry::difference(a, b);
  const double scale = std::max({std::abs(d[0]), std::abs(d[1]), std::abs(d[2])});
  if (scale == 0.0) return 0.0;
  const BoxPoint u{d[0] / scale, d[1] / scale, d[2] / scale};
  return scale * std::sqrt(geometry::dot(u.data(), u.data()));
}

using EdgeKey = std::uint64_t;

EdgeKey edge_key(const BoxCell& cell, int edge) {
  const auto [a, b] = kEdges[edge];
  return topology::undirected_edge(cell[a], cell[b]);
}

std::array<std::int32_t, 2> edge_rows(EdgeKey key) {
  return {static_cast<std::int32_t>(key >> 32),
          static_cast<std::int32_t>(static_cast<std::uint32_t>(key))};
}

struct SplitTemplate {
  std::array<std::array<std::uint8_t, 4>, 8> cells;
  std::size_t count = 0;

  bool build(unsigned mask) {
    if (count != 0) return true;
    std::array<double, 30> xyz{1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1};
    std::array<std::uint8_t, 10> slot{0, 1, 2, 3};
    std::size_t next = 4;
    for (unsigned i = 0; i < kEdges.size(); ++i) {
      if ((mask & (1U << i)) == 0) continue;
      const auto [a, b] = kEdges[i];
      for (int j = 0; j < 3; ++j) xyz[3 * next + j] = 0.5 * (xyz[3 * a + j] + xyz[3 * b + j]);
      slot[next++] = static_cast<std::uint8_t>(4 + i);
    }
    // The regular face geometry and relative perturbation identities agree
    // across parents: each uses ascending vertex rows, followed by split
    // sites in ascending endpoint-key order.
    phx_mc_mesh* raw = nullptr;
    const auto status = triangulate_3d(static_cast<std::int64_t>(next), xyz.data(), nullptr,
                                       false, true, &raw, 1);
    const std::unique_ptr<phx_mc_mesh> mesh(raw);
    if (!scalar_topology_accepted(status) || mesh->cells.empty() || mesh->cells.size() > 4 * cells.size()) {
      return false;
    }
    count = mesh->cells.size() / 4;
    for (std::size_t row = 0; row < count; ++row) {
      for (int j = 0; j < 4; ++j) cells[row][j] = slot[mesh->cells[4 * row + j]];
    }
    return true;
  }
};

// Sparse occupied buckets, with one link per retained site. Accepted trials
// enter the same grid immediately; their rows are relabeled after the batch.
class SeparationGrid {
 public:
  SeparationGrid(BoxPoint origin, double width, std::size_t count)
      : origin_(origin), width_(width), heads_(count) {
    links_.reserve(count);
  }

  void insert(const BoxPoint& point, std::int32_t row) {
    std::array<int, 3> bucket;
    for (int a = 0; a < 3; ++a) bucket[a] = static_cast<int>(offset(point, a));
    auto& head = heads_.slot(key(bucket[0], bucket[1], bucket[2]));
    links_.push_back({row, static_cast<std::int32_t>(head)});
    head = static_cast<std::int32_t>(links_.size() - 1);
  }

  void relabel(std::int32_t link, std::int32_t row) { links_[link].row = row; }

  template <class Conflicts>
  bool any(const BoxPoint& point, double radius, Conflicts conflicts) const {
    std::array<int, 3> low, high;
    for (int a = 0; a < 3; ++a) {
      const double center = offset(point, a);
      const double pad = 8 * std::numeric_limits<double>::epsilon() * (1.0 + center);
      const double range = radius / width_ + pad;
      low[a] = static_cast<int>(std::max(0.0, center - range));
      high[a] = static_cast<int>(center + range);
    }
    for (int z = low[2]; z <= high[2]; ++z) {
      for (int y = low[1]; y <= high[1]; ++y) {
        for (int x = low[0]; x <= high[0]; ++x) {
          auto link = heads_.find(key(x, y, z));
          while (link >= 0) {
            if (conflicts(links_[link].row)) return true;
            link = links_[link].next;
          }
        }
      }
    }
    return false;
  }

 private:
  struct Link { std::int32_t row, next; };

  // Bucket width is at least 2^-20 times the box extent.
  static std::uint64_t key(int x, int y, int z) {
    return (static_cast<std::uint64_t>(x) << 42) |
           (static_cast<std::uint64_t>(y) << 21) | static_cast<std::uint64_t>(z);
  }

  double offset(const BoxPoint& point, int axis) const {
    return (point[axis] - origin_[axis]) / width_;
  }

  BoxPoint origin_;
  double width_;
  KeyMap heads_;
  std::vector<Link> links_;
};

struct LongEdge {
  EdgeKey key;
  double length;
};

struct AcceptedSplit {
  EdgeKey key;
  BoxPoint point;
  LogRate rate;
  std::int32_t link;
};

bool split_sites(const AdaptInput& input, const ScalarBoxField& field,
                 std::vector<BoxPoint>& points, std::vector<LogRate>& rates) {
  std::vector<BoxCell> cells(input.source_cells.size() / 4), next_cells;
  for (std::size_t row = 0; row < cells.size(); ++row) {
    for (int j = 0; j < 4; ++j) {
      const auto source = input.source_cells[4 * row + j];
      if (source < 0 || static_cast<std::size_t>(source) >= points.size()) return false;
      cells[row][j] = static_cast<std::int32_t>(source);
    }
    std::sort(cells[row].begin(), cells[row].end());
  }

  const auto [minimum_row, maximum_row] = std::minmax_element(rates.begin(), rates.end(),
      [](LogRate a, LogRate b) { return a.log < b.log; });
  const LogRate minimum = *minimum_row;
  const double max_radius = kSeparation / minimum.rate;
  if (!std::isfinite(max_radius)) return false;
  const double min_radius = kSeparation / maximum_row->rate;
  double extent = 0.0;
  for (int a = 0; a < 3; ++a) extent = std::max(extent, field.high[a] - field.low[a]);
  // Finer buckets for the dense end of the field. Keep even the widest
  // physical query within four bucket widths without clipping its radius.
  const double width = std::max({std::sqrt(max_radius) * std::sqrt(min_radius),
                                 0.25 * max_radius, std::ldexp(extent, -20)});
  if (!(width > 0.0) || !std::isfinite(width)) return false;
  SeparationGrid grid(field.low, width, points.size());
  for (std::size_t row = 0; row < points.size(); ++row) {
    grid.insert(points[row], static_cast<std::int32_t>(row));
  }

  std::array<SplitTemplate, 64> templates;
  std::vector<LongEdge> proposals;
  std::vector<AcceptedSplit> accepted;
  std::vector<std::uint8_t> masks;
  std::int32_t first_new_row = 0;
  std::size_t new_cell_count = cells.size();
  for (int round = 0; round < kSplitRounds; ++round) {
    // A conforming tetrahedral ball has at most C + 2V distinct edges;
    // only rows containing a site added in the last round can add edges.
    KeyMap edges(std::min(6 * new_cell_count, cells.size() + 2 * points.size()));
    proposals.clear();
    accepted.clear();
    for (const auto& cell : cells) {
      if (cell.back() < first_new_row) continue;
      for (int edge = 0; edge < 6; ++edge) {
        // Old corner pairs already formed edges in the preceding round.
        // An unchanged midpoint rejected then is still rejected: its logs
        // and coordinates are fixed, and the conflicting site is retained.
        if (cell[kEdges[edge][1]] < first_new_row) continue;
        const auto key = edge_key(cell, edge);
        auto& split = edges.slot(key);
        if (split != -1) continue;
        split = -2;
        const auto [a, b] = edge_rows(key);
        const double length = distance(points[a], points[b]) * mean_rate(rates[a], rates[b]);
        if (!std::isfinite(length)) return false;
        if (length > kUpperLength) {
          if (proposals.size() == kMaxScalarCells) return false;
          proposals.push_back({key, length});
        }
      }
    }
    std::sort(proposals.begin(), proposals.end(), [](const LongEdge& a, const LongEdge& b) {
      return a.length == b.length ? a.key < b.key : a.length > b.length;
    });
    const auto first = static_cast<std::int32_t>(points.size());
    for (const auto& proposal : proposals) {
      const auto [a, b] = edge_rows(proposal.key);
      const double delta = rates[b].log - rates[a].log;
      const double t = static_cast<double>(round_even(1024.0 * metric_fraction(delta, 0.5))) / 1024.0;
      if (t <= 0.0 || t >= 1.0) continue;
      BoxPoint p;
      for (int j = 0; j < 3; ++j) p[j] = std::lerp(points[a][j], points[b][j], t);
      // Each template edge stays in an original P1 simplex (or on its
      // boundary), so the endpoint logs determine its midpoint log.
      const LogRate rate = log_rate(std::lerp(rates[a].log, rates[b].log, t));
      // The mean rate increases in either endpoint log, so the global
      // minimum bounds the physical query radius for this trial.
      const double radius = kSeparation / mean_rate(minimum, rate);
      const double padded_radius = radius + radius * (8 * std::numeric_limits<double>::epsilon());
      constexpr double kBroadPad = 1.0 + 32 * std::numeric_limits<double>::epsilon();
      constexpr double kSeparationSquared = kSeparation * kSeparation * kBroadPad;
      const double radius_squared = padded_radius * padded_radius * kBroadPad;
      const auto conflicts = [&](const BoxPoint& q, LogRate other) {
        const double squared = distance_squared(p, q);
        double d;
        if (normal_squared_distance(squared)) {
          // The log-linear mean is at least the geometric mean. With a
          // roundoff margin this saves expm1 for nonconflicting neighbors.
          const auto [lo, hi] = std::minmax(rate.rate, other.rate);
          if (squared > radius_squared || (squared * lo) * hi > kSeparationSquared) return false;
          d = std::sqrt(squared);
        } else {
          d = distance(p, q);
        }
        return d <= padded_radius && d * mean_rate(rate, other) < kSeparation;
      };
      if (grid.any(p, radius, [&](std::int32_t row) {
        if (row < first) return conflicts(points[row], rates[row]);
        const auto& other = accepted[row - first];
        return conflicts(other.point, other.rate);
      })) continue;
      if (points.size() + accepted.size() == kMaxScalarSites) return false;
      const auto link = first + static_cast<std::int32_t>(accepted.size());
      grid.insert(p, link);
      accepted.push_back({proposal.key, p, rate, link});
    }
    if (accepted.empty()) break;

    // Greedy priority is length order; candidate rows are assigned in key order.
    std::sort(accepted.begin(), accepted.end(), [](const AcceptedSplit& a, const AcceptedSplit& b) {
      return a.key < b.key;
    });
    for (const auto& split : accepted) {
      const auto row = static_cast<std::int32_t>(points.size());
      edges.slot(split.key) = row;
      grid.relabel(split.link, row);
      points.push_back(split.point);
      rates.push_back(split.rate);
    }

    masks.resize(cells.size());
    std::size_t count = 0;
    new_cell_count = 0;
    for (std::size_t row = 0; row < cells.size(); ++row) {
      unsigned mask = 0;
      for (int edge = 0; edge < 6; ++edge) {
        if (edges.find(edge_key(cells[row], edge)) >= 0) mask |= 1U << edge;
      }
      masks[row] = static_cast<std::uint8_t>(mask);
      if (mask != 0 && !templates[mask].build(mask)) return false;
      if (mask != 0) new_cell_count += templates[mask].count;
      count += mask == 0 ? 1 : templates[mask].count;
      if (count > kMaxScalarCells) return false;
    }
    next_cells.clear();
    next_cells.reserve(count);
    for (std::size_t row = 0; row < cells.size(); ++row) {
      const unsigned mask = masks[row];
      if (mask == 0) {
        next_cells.push_back(cells[row]);
        continue;
      }
      std::array<std::int32_t, 10> local;
      std::copy(cells[row].begin(), cells[row].end(), local.begin());
      for (int edge = 0; edge < 6; ++edge) {
        if ((mask & (1U << edge)) != 0) {
          local[4 + edge] = static_cast<std::int32_t>(edges.find(edge_key(cells[row], edge)));
        }
      }
      for (std::size_t i = 0; i < templates[mask].count; ++i) {
        const auto& v = templates[mask].cells[i];
        BoxCell child{local[v[0]], local[v[1]], local[v[2]], local[v[3]]};
        std::sort(child.begin(), child.end());
        next_cells.push_back(child);
      }
    }
    cells.swap(next_cells);
    first_new_row = first;
  }
  return true;
}

int wall_rank(const BoxPoint& p, const ScalarBoxField& field) {
  int rank = 0;
  for (int a = 0; a < 3; ++a) rank += p[a] == field.low[a] || p[a] == field.high[a];
  return rank;
}

bool redistribute_seams(const ScalarBoxField& field, std::size_t source_count,
                        const std::vector<LogRate>& rates, std::vector<BoxPoint>& points) {
  struct Knot { double coordinate; LogRate rate; };
  std::array<std::vector<Knot>, 12> knots;
  for (std::size_t row = 0; row < source_count; ++row) {
    const auto& p = points[row];
    for (int axis = 0; axis < 3; ++axis) {
      const auto [u, v] = kFixedAxes[axis];
      if ((p[u] != field.low[u] && p[u] != field.high[u]) ||
          (p[v] != field.low[v] && p[v] != field.high[v])) continue;
      const int seam = 4 * axis + 2 * (p[u] == field.high[u]) + (p[v] == field.high[v]);
      knots[seam].push_back({p[axis], rates[row]});
    }
  }
  points.erase(std::remove_if(points.begin(), points.end(), [&](const BoxPoint& p) {
    return wall_rank(p, field) == 2;
  }), points.end());

  std::vector<double> prefix;
  for (int seam = 0; seam < 12; ++seam) {
    const int axis = seam / 4;
    const auto [u, v] = kFixedAxes[axis];
    auto& line = knots[seam];
    std::sort(line.begin(), line.end(), [](const Knot& a, const Knot& b) {
      return a.coordinate < b.coordinate;
    });
    if (line.size() < 2 || line.front().coordinate != field.low[axis] ||
        line.back().coordinate != field.high[axis]) return false;
    prefix.resize(line.size());
    prefix[0] = 0.0;
    for (std::size_t i = 1; i < line.size(); ++i) {
      prefix[i] = prefix[i - 1] + (line[i].coordinate - line[i - 1].coordinate) *
                                       mean_rate(line[i - 1].rate, line[i].rate);
      if (!(prefix[i] > prefix[i - 1])) return false;
    }
    const double mass = prefix.back();
    if (!std::isfinite(mass) || mass > static_cast<double>(kMaxScalarSites + 1)) return false;
    const auto intervals = std::max(std::size_t{1}, round_even(mass));
    if (intervals - 1 > kMaxScalarSites - points.size()) return false;
    BoxPoint p = field.low;
    if ((seam & 2) != 0) p[u] = field.high[u];
    if ((seam & 1) != 0) p[v] = field.high[v];
    double previous = field.low[axis];
    std::size_t i = 1;
    for (std::size_t j = 1; j < intervals; ++j) {
      const double target = mass * (static_cast<double>(j) / static_cast<double>(intervals));
      while (target > prefix[i]) ++i;
      const auto& a = line[i - 1];
      const auto& b = line[i];
      const double fraction = (target - prefix[i - 1]) / (prefix[i] - prefix[i - 1]);
      p[axis] = std::lerp(a.coordinate, b.coordinate, metric_fraction(b.rate.log - a.rate.log, fraction));
      if (!(p[axis] > previous && p[axis] < field.high[axis])) return false;
      points.push_back(p);
      previous = p[axis];
    }
  }
  return true;
}

constexpr int kWallGapRounds = 3;
constexpr std::size_t kMaxWallAdds = 8192;
constexpr std::size_t kMaxAddsPerWall = 4096;

struct WallGap {
  BoxPoint point;
  ScalarBoxSample sample;
  LogRate rate;
  double radius;
  int face;
  std::array<std::int32_t, 3> support;
};

bool wall_circumcenter(const ScalarBoxField& field, int face, const BoxPoint& a,
                       const BoxPoint& b, const BoxPoint& c, BoxPoint& center) {
  const int axis = face / 2;
  const auto [u, v] = kFixedAxes[axis];
  double bu = b[u] - a[u], bv = b[v] - a[v];
  double cu = c[u] - a[u], cv = c[v] - a[v];
  const double scale = std::max({std::abs(bu), std::abs(bv), std::abs(cu), std::abs(cv)});
  if (!(scale > 0.0) || !std::isfinite(scale)) return false;
  bu /= scale;
  bv /= scale;
  cu /= scale;
  cv /= scale;
  const double determinant = 2.0 * (bu * cv - bv * cu);
  if (determinant == 0.0) return false;
  const double bb = bu * bu + bv * bv, cc = cu * cu + cv * cv;
  center[axis] = a[axis];
  center[u] = a[u] + scale * ((bb * cv - cc * bv) / determinant);
  center[v] = a[v] + scale * ((bu * cc - cu * bb) / determinant);
  return center[u] > field.low[u] && center[u] < field.high[u] &&
         center[v] > field.low[v] && center[v] < field.high[v];
}

template <class PointAt>
bool collect_wall_gaps(const ScalarBoxField& field, int face, PointAt point_at,
                       const std::vector<LogRate>& rates, const std::vector<std::int32_t>& wall,
                       std::vector<double>& xy, std::vector<WallGap>& gaps) {
  const auto [u, v] = kFixedAxes[face / 2];
  xy.clear();
  xy.reserve(2 * wall.size());
  for (auto row : wall) {
    const auto& p = point_at(row);
    xy.push_back(p[u]);
    xy.push_back(p[v]);
  }
  phx_mc_mesh* raw = nullptr;
  const auto status = triangulate_2d(static_cast<std::int64_t>(wall.size()), xy.data(),
                                     nullptr, false, true, &raw);
  const std::unique_ptr<phx_mc_mesh> mesh(raw);
  if (!scalar_topology_accepted(status)) return false;
  for (std::size_t row = 0; row < mesh->cells.size(); row += 3) {
    const auto a = wall[mesh->cells[row]], b = wall[mesh->cells[row + 1]],
               c = wall[mesh->cells[row + 2]];
    const auto& p = point_at(a);
    const auto& q = point_at(b);
    const auto& r = point_at(c);
    if (distance(p, q) * mean_rate(rates[a], rates[b]) <= kUpperLength &&
        distance(q, r) * mean_rate(rates[b], rates[c]) <= kUpperLength &&
        distance(r, p) * mean_rate(rates[c], rates[a]) <= kUpperLength) continue;
    WallGap gap{};
    if (!wall_circumcenter(field, face, p, q, r, gap.point) ||
        !field.sample(gap.point, gap.sample)) continue;
    gap.rate = log_rate(gap.sample.log_rate);
    if (!(gap.rate.rate > 0.0) || !std::isfinite(gap.rate.rate)) continue;
    // The planar triangulation gives a physically empty circumdisk. Rank
    // its size by the smallest transported radius to a supporting site.
    gap.radius = std::min({distance(gap.point, p) * mean_rate(gap.rate, rates[a]),
                           distance(gap.point, q) * mean_rate(gap.rate, rates[b]),
                           distance(gap.point, r) * mean_rate(gap.rate, rates[c])});
    if (!(gap.radius >= kSeparation) || !std::isfinite(gap.radius)) continue;
    gap.face = face;
    gap.support = {a, b, c};
    std::sort(gap.support.begin(), gap.support.end());
    if (gaps.size() == kMaxScalarCells) return false;
    gaps.push_back(gap);
  }
  return true;
}

}  // namespace

bool seed_scalar_box(const AdaptInput& input, const ScalarBoxField& field, ScalarBoxSites& sites) {
  const std::size_t count = input.source_points.size() / 3;
  if (count < 8 || count > kMaxScalarSites || input.source_points.size() != 3 * count ||
      input.source_logarithms.size() != 9 * count || input.source_cells.empty() ||
      input.source_cells.size() % 4 != 0 || input.source_cells.size() / 4 > kMaxScalarCells) {
    return false;
  }
  ScalarBoxSites candidate;
  candidate.points.reserve(count);
  std::vector<LogRate> rates;
  rates.reserve(count);
  for (std::size_t row = 0; row < count; ++row) {
    BoxPoint p;
    for (int a = 0; a < 3; ++a) {
      p[a] = input.source_points[3 * row + a];
      if (!(p[a] >= field.low[a] && p[a] <= field.high[a])) return false;
    }
    const auto rate = log_rate(0.5 * input.source_logarithms[9 * row]);
    if (!(rate.rate > 0.0) || !std::isfinite(rate.rate)) return false;
    candidate.points.push_back(p);
    rates.push_back(rate);
  }
  if (!split_sites(input, field, candidate.points, rates) ||
      !redistribute_seams(field, count, rates, candidate.points)) return false;
  candidate.samples.resize(candidate.points.size());
  for (std::size_t row = 0; row < candidate.points.size(); ++row) {
    if (!field.sample(candidate.points[row], candidate.samples[row])) return false;
  }
  sites = std::move(candidate);
  return true;
}

bool fill_scalar_wall_gaps(const ScalarBoxField& field, ScalarBoxSites& sites) {
  const std::size_t initial = sites.points.size();
  if (initial < 8 || initial >= kMaxScalarSites || sites.samples.size() != initial) return false;
  const auto budget = std::min(kMaxWallAdds, kMaxScalarSites - initial);
  double extent = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    extent = std::max(extent, field.high[axis] - field.low[axis]);
  }
  if (!(extent > 0.0) || !std::isfinite(extent)) return false;

  std::array<std::vector<std::int32_t>, 6> walls;
  std::vector<std::int32_t> boundary;
  std::vector<LogRate> rates(initial);
  LogRate minimum{};
  minimum.log = std::numeric_limits<double>::infinity();
  for (std::size_t row = 0; row < initial; ++row) {
    const auto& p = sites.points[row];
    const auto id = static_cast<std::int32_t>(row);
    bool on_wall = false;
    for (int axis = 0; axis < 3; ++axis) {
      if (!(p[axis] >= field.low[axis] && p[axis] <= field.high[axis])) return false;
      if (p[axis] == field.low[axis]) {
        walls[2 * axis].push_back(id);
        on_wall = true;
      } else if (p[axis] == field.high[axis]) {
        walls[2 * axis + 1].push_back(id);
        on_wall = true;
      }
    }
    if (!on_wall) continue;
    rates[row] = log_rate(sites.samples[row].log_rate);
    if (!(rates[row].rate > 0.0) || !std::isfinite(rates[row].rate)) return false;
    if (rates[row].log < minimum.log) minimum = rates[row];
    boundary.push_back(id);
  }
  for (const auto& wall : walls) if (wall.size() < 4) return false;

  std::vector<WallGap> added, gaps;
  const auto point_at = [&](std::size_t row) -> const BoxPoint& {
    return row < initial ? sites.points[row] : added[row - initial].point;
  };
  std::array<std::size_t, 6> wall_adds{};
  std::vector<double> xy;
  for (int round = 0; round < kWallGapRounds && added.size() < budget; ++round) {
    gaps.clear();
    for (int face = 0; face < 6; ++face) {
      if (wall_adds[face] == kMaxAddsPerWall) continue;
      if (!collect_wall_gaps(field, face, point_at, rates, walls[face], xy, gaps)) return false;
    }
    if (gaps.empty()) break;
    std::sort(gaps.begin(), gaps.end(), [](const WallGap& a, const WallGap& b) {
      if (a.radius != b.radius) return a.radius > b.radius;
      if (a.face != b.face) return a.face < b.face;
      return a.support < b.support;
    });

    // Include this batch in the minimum so the sparse grid also sees every
    // nearby point accepted earlier in this round, including adjacent walls.
    LogRate batch_min = minimum;
    for (const auto& gap : gaps) if (gap.rate.log < batch_min.log) batch_min = gap.rate;
    const double width = std::max(kSeparation / batch_min.rate, std::ldexp(extent, -20));
    if (!(width > 0.0) || !std::isfinite(width)) return false;
    SeparationGrid grid(field.low, width, boundary.size());
    for (auto row : boundary) grid.insert(point_at(row), row);
    const std::size_t before = added.size();
    for (const auto& gap : gaps) {
      if (added.size() == budget) break;
      if (wall_adds[gap.face] == kMaxAddsPerWall) continue;
      const double radius = kSeparation / mean_rate(batch_min, gap.rate);
      if (grid.any(gap.point, radius, [&](std::int32_t row) {
        const double d = distance(gap.point, point_at(row));
        return d * mean_rate(gap.rate, rates[row]) < kSeparation;
      })) continue;
      const auto row = static_cast<std::int32_t>(initial + added.size());
      added.push_back(gap);
      rates.push_back(gap.rate);
      if (gap.rate.log < minimum.log) minimum = gap.rate;
      boundary.push_back(row);
      walls[gap.face].push_back(row);
      ++wall_adds[gap.face];
      grid.insert(gap.point, row);
    }
    if (added.size() == before) break;
  }
  if (added.empty()) return false;
  sites.points.reserve(initial + added.size());
  sites.samples.reserve(initial + added.size());
  for (const auto& gap : added) {
    sites.points.push_back(gap.point);
    sites.samples.push_back(gap.sample);
  }
  sites.cells.clear();
  return true;
}

}  // namespace phx::mc::metric
