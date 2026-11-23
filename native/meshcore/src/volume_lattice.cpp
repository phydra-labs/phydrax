//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include "volume_lattice.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <utility>

#include "geometry3.hpp"

namespace phx::mc {

BccRotation bcc_lattice_rotation() {
  double q[4]{0.9, 0.3, 0.2, 0.25};
  const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  for (double& value : q) value /= norm;
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  return {{{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
           {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
           {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}}};
}

double bcc_lattice_spacing(double circumradius) {
  return 4.0 * kBccLatticeMargin * circumradius / std::sqrt(5.0);
}

namespace {

using Point = std::array<double, 3>;
constexpr auto kShift = kBccLatticeShift;
constexpr std::size_t kDenseBytes = 64U * 1024 * 1024;
constexpr std::size_t kScratchBytes = 128U * 1024 * 1024;
// Bound the non-linear work of overlapping facet boxes and detached roots.
constexpr std::size_t kFacetVisits = 1U << 26;
constexpr std::int32_t kOutsideBox = -1;
constexpr std::int32_t kSurfaceBand = -2;

struct Frame {
  BccRotation rotation;
  Point low, high, middle;
  double spacing, roundoff;
  int reach, side;
  std::size_t plane, cube;

  Point site(int shift, int i, int j, int k) const {
    Point point;
    bcc_lattice_point(middle.data(), rotation, spacing, shift, i, j, k, point.data());
    return point;
  }

  Point inverse(const double* point) const {
    const Point delta{point[0] - middle[0], point[1] - middle[1], point[2] - middle[2]};
    Point local;
    for (int axis = 0; axis < 3; ++axis) {
      local[axis] = (rotation[0][axis] * delta[0] + rotation[1][axis] * delta[1] +
                     rotation[2][axis] * delta[2]) / spacing;
    }
    return local;
  }

  std::size_t slot(int shift, int i, int j, int k) const {
    return (static_cast<std::size_t>(shift * side + k + reach) * side + (j + reach)) * side +
           (i + reach);
  }
};

bool prepare_frame(std::span<const double> points, double circumradius, Frame& frame) {
  frame.low = frame.high = {points[0], points[1], points[2]};
  for (std::size_t vertex = 0; vertex < points.size(); vertex += 3) {
    if (!geometry::point_in_domain(points.data() + vertex)) return false;
    for (int axis = 0; axis < 3; ++axis) {
      frame.low[axis] = std::min(frame.low[axis], points[vertex + axis]);
      frame.high[axis] = std::max(frame.high[axis], points[vertex + axis]);
    }
  }
  double half_diagonal_squared = 0.0;
  double world_scale = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    const double width = frame.high[axis] - frame.low[axis];
    if (!(width > 0.0)) return false;
    frame.middle[axis] = 0.5 * (frame.low[axis] + frame.high[axis]);
    half_diagonal_squared += 0.25 * width * width;
    world_scale = std::max({world_scale, std::abs(frame.low[axis]), std::abs(frame.high[axis])});
  }
  frame.spacing = bcc_lattice_spacing(circumradius);
  if (!(frame.spacing > 0.0 && frame.spacing <= 2.0 * kCoordinateMaxMagnitude)) return false;
  const double extent = std::sqrt(half_diagonal_squared) / frame.spacing;
  constexpr std::size_t kMaxSlots = kDenseBytes / sizeof(std::int32_t);
  if (!(extent < static_cast<double>(kMaxSlots))) return false;

  // The rotated box is inside the half-diagonal sphere. Allow for the larger
  // phase offset (0.97 at most), and keep one full index halo for its shifted
  // neighbors. World rounding must be well below the 0.03 remaining slack.
  frame.roundoff = 32.0 * std::numeric_limits<double>::epsilon() *
                   (world_scale / frame.spacing + extent + 2.0);
  if (!(frame.roundoff < 0x1p-12)) return false;
  frame.reach = static_cast<int>(extent) + 2;
  frame.side = 2 * frame.reach + 1;
  const auto side = static_cast<std::size_t>(frame.side);
  if (side > kMaxSlots / 2 / side / side) return false;
  frame.plane = side * side;
  frame.cube = frame.plane * side;
  frame.rotation = bcc_lattice_rotation();
  return true;
}

struct IntegerRange {
  int first = 0;
  int last = -1;

  std::size_t size() const {
    return first <= last ? static_cast<std::size_t>(last - first + 1) : 0;
  }
};

IntegerRange integer_range(double low, double high, int reach) {
  low = std::max(low, static_cast<double>(-reach));
  high = std::min(high, static_cast<double>(reach));
  if (low > high) return {};
  return {static_cast<int>(std::ceil(low)), static_cast<int>(std::floor(high))};
}

bool spend(std::size_t visits, std::size_t& remaining) {
  if (visits > remaining) return false;
  remaining -= visits;
  return true;
}

bool mark_facet_band(const Frame& frame, const Point& u, const Point& v, const Point& w,
                     const double* a, const double* b, const double* c,
                     std::span<std::int32_t> bank, std::size_t& remaining) {
  Point low, high;
  for (int axis = 0; axis < 3; ++axis) {
    low[axis] = std::min({u[axis], v[axis], w[axis]});
    high[axis] = std::max({u[axis], v[axis], w[axis]});
    const double magnitude = std::max(std::abs(low[axis]), std::abs(high[axis]));
    const double pad = 0.5 + frame.roundoff + 1e-9 * (1.0 + magnitude);
    low[axis] -= pad;
    high[axis] += pad;
  }
  // The distance's Voronoi-region tests subtract products to resolve the
  // triangle normal. When the area is too small at this scale, conservatively
  // mask the expanded box instead of trusting those tests near the wall.
  double ab[3], ac[3], n[3];
  geometry::difference(b, a, ab);
  geometry::difference(c, a, ac);
  geometry::cross(ab, ac, n);
  const double scale = std::max({frame.spacing * frame.spacing, geometry::dot(ab, ab),
                                 geometry::dot(ac, ac), geometry::squared_distance(b, c)});
  const bool box_only = geometry::dot(n, n) <= 0x1p-22 * scale * scale;
  const double band = 0.5 * frame.spacing;
  for (int shift = 0; shift < 2; ++shift) {
    const double half = 0.5 * static_cast<double>(shift);
    std::array<IntegerRange, 3> range;
    for (int axis = 0; axis < 3; ++axis) {
      range[axis] = integer_range(low[axis] - kShift[axis] - half,
                                  high[axis] - kShift[axis] - half, frame.reach);
    }
    const auto visits = range[0].size() * range[1].size() * range[2].size();
    if (visits == 0) continue;
    if (!spend(visits, remaining)) return false;
    for (int k = range[2].first; k <= range[2].last; ++k) {
      for (int j = range[1].first; j <= range[1].last; ++j) {
        auto slot = frame.slot(shift, range[0].first, j, k);
        for (int i = range[0].first; i <= range[0].last; ++i, ++slot) {
          if (bank[slot] == kSurfaceBand) continue;
          if (box_only || geometry::point_triangle_distance(frame.site(shift, i, j, k).data(),
                                                            a, b, c) < band) {
            bank[slot] = kSurfaceBand;
          }
        }
      }
    }
  }
  return true;
}

bool mark_surface_band(const Frame& frame, std::span<const double> points,
                       std::span<const std::int32_t> triangles, std::span<std::int32_t> bank,
                       std::size_t& remaining) {
  const auto vertices = points.size() / 3;
  if (!spend(triangles.size() / 3, remaining)) return false;
  std::vector<Point> local;
  local.reserve(vertices);
  for (std::size_t i = 0; i < points.size(); i += 3) {
    local.push_back(frame.inverse(points.data() + i));
  }
  for (std::size_t i = 0; i < triangles.size(); i += 3) {
    const auto a = static_cast<std::size_t>(triangles[i]);
    const auto b = static_cast<std::size_t>(triangles[i + 1]);
    const auto c = static_cast<std::size_t>(triangles[i + 2]);
    if (a >= vertices || b >= vertices || c >= vertices || a == b || b == c || c == a) {
      return false;
    }
    if (!mark_facet_band(frame, local[a], local[b], local[c], points.data() + 3 * a,
                         points.data() + 3 * b, points.data() + 3 * c, bank, remaining)) {
      return false;
    }
  }
  return true;
}

struct BoxRow {
  int shift, j, k;
  IntegerRange i;
};

struct Site {
  Point point;
  std::size_t slot;
};

bool clip_box_rows(const Frame& frame, std::size_t candidate_limit, std::vector<BoxRow>& rows,
                   std::size_t& candidate_bound) {
  const int halo = frame.reach - 1;
  const Point steps{frame.rotation[0][0] * frame.spacing, frame.rotation[1][0] * frame.spacing,
                    frame.rotation[2][0] * frame.spacing};
  for (int shift = 0; shift < 2; ++shift) {
    for (int k = -halo; k <= halo; ++k) {
      for (int j = -halo; j <= halo; ++j) {
        const Point anchor = frame.site(shift, 0, j, k);
        double start = -halo, stop = halo;
        // Clip each row to the world box, then leave a full site beyond the
        // rounded interval at both ends for the strict world-coordinate test.
        for (int axis = 0; axis < 3; ++axis) {
          const double a = (frame.low[axis] - anchor[axis]) / steps[axis];
          const double b = (frame.high[axis] - anchor[axis]) / steps[axis];
          start = std::max(start, std::min(a, b) - 1.0);
          stop = std::min(stop, std::max(a, b) + 1.0);
        }
        const auto i = integer_range(start, stop, halo);
        const auto width = i.size();
        if (width == 0) continue;
        if (!spend(width, candidate_limit)) return false;
        candidate_bound += width;
        rows.push_back({shift, j, k, i});
      }
    }
  }
  return true;
}

bool enumerate_sites(const Frame& frame, std::span<const BoxRow> rows,
                     std::span<std::int32_t> bank, std::vector<Site>& candidates) {
  for (const BoxRow& row : rows) {
    auto slot = frame.slot(row.shift, row.i.first, row.j, row.k);
    for (int i = row.i.first; i <= row.i.last; ++i, ++slot) {
      if (bank[slot] == kSurfaceBand) continue;
      const Point point = frame.site(row.shift, i, row.j, row.k);
      if (!(point[0] > frame.low[0] && point[0] < frame.high[0] &&
            point[1] > frame.low[1] && point[1] < frame.high[1] &&
            point[2] > frame.low[2] && point[2] < frame.high[2])) {
        continue;
      }
      if (!geometry::point_in_domain(point)) return false;
      bank[slot] = static_cast<std::int32_t>(candidates.size());
      candidates.push_back({point, slot});
    }
  }
  return true;
}

std::int32_t root(std::span<std::int32_t> parents, std::int32_t site) {
  while (parents[site] != site) {
    parents[site] = parents[parents[site]];
    site = parents[site];
  }
  return site;
}

void connect_clear_sites(const Frame& frame, std::span<const std::int32_t> bank,
                         std::span<const Site> candidates, std::span<std::int32_t> parents) {
  const auto exterior = static_cast<std::int32_t>(candidates.size());
  const auto join = [&](std::int32_t a, std::int32_t b) {
    a = root(parents, a);
    b = root(parents, b);
    if (a != b) parents[std::max(a, b)] = std::min(a, b);
  };
  // A nearest-neighbor BCC edge is sqrt(3)/2 spacings long. A crossing
  // needs at least one endpoint within sqrt(3)/4 spacings of the wall;
  // the half-spacing band leaves slack for distance and frame roundoff.
  // The band also marks endpoints outside the strict world box.
  const auto side = static_cast<std::size_t>(frame.side);
  for (std::int32_t v = 0; v < exterior; ++v) {
    const auto slot = candidates[v].slot;
    const bool shifted = slot >= frame.cube;
    const auto base = shifted ? slot - frame.cube : slot + frame.cube - frame.plane - side - 1;
    for (std::size_t dz : {std::size_t{0}, frame.plane}) {
      for (std::size_t dy : {std::size_t{0}, side}) {
        for (std::size_t dx : {std::size_t{0}, std::size_t{1}}) {
          const auto other = bank[base + dz + dy + dx];
          if (other == kOutsideBox) join(v, exterior);
          if (other >= 0 && !shifted) join(v, other);
        }
      }
    }
  }
}

bool wall_winding(std::span<const double> points, std::span<const std::int32_t> triangles,
                  const Point& site, std::int64_t& winding) {
  for (std::size_t i = 0; i < triangles.size(); i += 3) {
    const double* a = points.data() + 3 * static_cast<std::size_t>(triangles[i]);
    const double* b = points.data() + 3 * static_cast<std::size_t>(triangles[i + 1]);
    const double* c = points.data() + 3 * static_cast<std::size_t>(triangles[i + 2]);
    if (!add_volume_ray_crossing(a, b, c, site.data(), winding)) return false;
  }
  return true;
}

bool collect_owned_sites(std::span<const double> points, std::span<const std::int32_t> triangles,
                         std::span<const Site> candidates, std::span<std::int32_t> parents,
                         std::size_t remaining, std::vector<double>& sites) {
  const auto exterior = static_cast<std::int32_t>(candidates.size());
  std::vector<std::int8_t> verdict(parents.size(), -1);
  verdict[root(parents, exterior)] = 0;
  std::size_t owned = 0;
  for (std::int32_t v = 0; v < exterior; ++v) {
    const auto representative = root(parents, v);
    parents[v] = representative;
    auto& label = verdict[representative];
    if (label < 0) {
      std::int64_t winding = 0;
      if (!spend(triangles.size() / 3, remaining) ||
          !wall_winding(points, triangles, candidates[v].point, winding)) {
        return false;
      }
      label = winding != 0;
    }
    owned += label;
  }
  std::vector<double> proposals;
  proposals.reserve(3 * owned);
  for (std::int32_t v = 0; v < exterior; ++v) {
    if (verdict[parents[v]]) {
      const Point& point = candidates[v].point;
      proposals.insert(proposals.end(), point.begin(), point.end());
    }
  }
  sites = std::move(proposals);
  return true;
}

}  // namespace

bool prepare_owned_volume_lattice(std::span<const double> points,
                                 std::span<const std::int32_t> triangles,
                                 double circumradius, std::vector<double>& sites) {
  if (points.size() < 12 || points.size() % 3 != 0 ||
      triangles.size() < 12 || triangles.size() % 3 != 0 ||
      points.size() / 3 > kScratchBytes / sizeof(Point) || triangles.size() / 3 > kFacetVisits) {
    return false;
  }
  Frame frame;
  if (!prepare_frame(points, circumradius, frame)) return false;
  const auto bank_bytes = 2 * frame.cube * sizeof(std::int32_t);
  if (points.size() / 3 > (kScratchBytes - bank_bytes) / sizeof(Point)) return false;

  std::size_t remaining = kFacetVisits;
  std::vector<std::int32_t> bank(2 * frame.cube, kOutsideBox);
  if (!mark_surface_band(frame, points, triangles, bank, remaining)) return false;

  std::vector<Site> candidates;
  {
    const auto row_side = static_cast<std::size_t>(frame.side - 2);
    const auto max_rows = 2 * row_side * row_side;
    constexpr auto kLabelBytes = sizeof(std::int32_t) + sizeof(std::int8_t);
    constexpr auto kSiteBytes = sizeof(Site) + kLabelBytes + sizeof(Point);
    // Charge the reserved upper bound, DSU, and packed output in addition to
    // the dense index, even though row clipping and output do not overlap.
    const auto candidate_limit =
        (kScratchBytes - bank_bytes - max_rows * sizeof(BoxRow) - kLabelBytes) / kSiteBytes;
    std::vector<BoxRow> rows;
    rows.reserve(max_rows);
    std::size_t candidate_bound = 0;
    if (!clip_box_rows(frame, candidate_limit, rows, candidate_bound)) return false;
    candidates.reserve(candidate_bound);
    if (!enumerate_sites(frame, rows, bank, candidates)) return false;
  }
  if (candidates.empty()) {
    sites.clear();
    return true;
  }
  std::vector<std::int32_t> parents(candidates.size() + 1);
  std::iota(parents.begin(), parents.end(), std::int32_t{0});
  connect_clear_sites(frame, bank, candidates, parents);
  return collect_owned_sites(points, triangles, candidates, parents, remaining, sites);
}

}  // namespace phx::mc
