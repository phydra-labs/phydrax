//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Numerical feature protection and exact source-ancestry construction shared
// by constrained recovery and tetrahedral refinement. Protection radii only
// schedule insertions; exact predicates, source witnesses, and cavity
// transactions decide geometric validity.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <vector>

#include "expansion.hpp"
#include "geometry3.hpp"
#include "intersections.hpp"
#include "predicates.hpp"
#include "rounding.hpp"
#include "source_construction.hpp"

namespace phx::mc {
namespace {

double distance(const double* a, const double* b) {
  return std::sqrt(geometry::squared_distance(a, b));
}

using geometry::dot;
using geometry::point_segment_distance;
using geometry::point_triangle_distance;

// This index encloses the computed closest-point kernels, not only ideal
// segments: a + t*(b-a) may end at a rounded value distinct from b. Rows
// are points, segments (`arity` 2) or triangles (`arity` 3) of `cells`.
// Triangle boxes are widened by 2^-20 of their extent plus 2^-40 of their
// coordinates, enclosing the rounded interior closest points of the
// Voronoi-region kernel; triangles with a nearly degenerate corner angle
// have unbounded boxes.
//
// The rows with finite boxes are registered in every cell their box meets
// of a dense uniform grid (cells of side `side`, coarsened to at most eight
// cells per row) in row order, the others are listed apart. The cell
// coordinate is monotone in the coordinate, so a box that meets the query
// cube meets one of its cells.
class ProtectionIndex {
  struct Box {
    double lower[3];
    double upper[3];
  };

 public:
  void build(const double* points, std::size_t count, std::span<const int32_t> cells,
             std::size_t arity, double side) {
    const std::size_t size = arity > 1 ? cells.size() / arity : count;
    boxes_.assign(size, Box{});
    stamps_.assign(size, 0U);
    if (size == 0) return;
    std::vector<char> bounded(size, 0);
    double low[3], high[3];
    std::fill_n(low, 3, std::numeric_limits<double>::infinity());
    std::fill_n(high, 3, -std::numeric_limits<double>::infinity());
    for (std::size_t row = 0; row < size; ++row) {
      Box& box = boxes_[row];
      const double* a = points + 3 * (arity > 1 ? static_cast<std::size_t>(cells[arity * row]) : row);
      bool unbounded = false;
      if (arity == 3) {
        const double* b = points + 3 * static_cast<std::size_t>(cells[3 * row + 1]);
        const double* c = points + 3 * static_cast<std::size_t>(cells[3 * row + 2]);
        const double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const double ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const double normal[3] = {ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2],
                                  ab[0] * ac[1] - ab[1] * ac[0]};
        // Corner angles with sin^2 below 2^-30 make the interior closest
        // point ill-conditioned; such triangles meet every query.
        unbounded = !(dot(normal, normal) >= 0x1p-30 * dot(ab, ab) * dot(ac, ac));
      }
      for (int axis = 0; axis < 3; ++axis) {
        box.lower[axis] = box.upper[axis] = a[axis];
        if (arity == 2) {
          const double b = points[3 * static_cast<std::size_t>(cells[2 * row + 1]) + axis];
          const double reconstructed = a[axis] + (b - a[axis]);
          if (std::isfinite(reconstructed)) {
            box.lower[axis] = std::min({a[axis], b, reconstructed});
            box.upper[axis] = std::max({a[axis], b, reconstructed});
          } else {
            box.lower[axis] = -std::numeric_limits<double>::infinity();
            box.upper[axis] = std::numeric_limits<double>::infinity();
          }
        } else if (arity == 3) {
          const double b = points[3 * static_cast<std::size_t>(cells[3 * row + 1]) + axis];
          const double c = points[3 * static_cast<std::size_t>(cells[3 * row + 2]) + axis];
          const double lower = std::min({a[axis], b, c});
          const double upper = std::max({a[axis], b, c});
          const double margin = 0x1p-20 * (upper - lower) +
                                0x1p-40 * std::max(std::fabs(lower), std::fabs(upper));
          const bool finite = !unbounded && std::isfinite(margin);
          box.lower[axis] = finite ? lower - margin : -std::numeric_limits<double>::infinity();
          box.upper[axis] = finite ? upper + margin : std::numeric_limits<double>::infinity();
        }
      }
      bool finite = true;
      for (int axis = 0; axis < 3; ++axis) {
        finite = finite && std::isfinite(box.lower[axis]) && std::isfinite(box.upper[axis]);
      }
      if (!finite) {
        unbounded_.push_back(static_cast<uint32_t>(row));
        continue;
      }
      bounded[row] = 1;
      for (int axis = 0; axis < 3; ++axis) {
        low[axis] = std::min(low[axis], box.lower[axis]);
        high[axis] = std::max(high[axis], box.upper[axis]);
      }
    }
    if (unbounded_.size() == size) return;
    // Cells of side `side` (the extent when it is not positive), coarsened
    // until there are at most eight cells and sixteen registrations per row.
    double extent = 0.0;
    for (int axis = 0; axis < 3; ++axis) extent = std::max(extent, high[axis] - low[axis]);
    cell_ = side > 0.0 && std::isfinite(side) ? side : extent;
    if (!(cell_ > 0.0)) cell_ = 1.0;
    const double cap = 8.0 * static_cast<double>(size) + 64.0;
    const auto range = [&](std::size_t row, std::size_t* first, std::size_t* last) {
      std::size_t cells = 1;
      for (int axis = 0; axis < 3; ++axis) {
        first[axis] = coordinate(boxes_[row].lower[axis], axis);
        last[axis] = coordinate(boxes_[row].upper[axis], axis);
        cells *= last[axis] - first[axis] + 1;
      }
      return cells;
    };
    for (;;) {
      double total = 1.0;
      for (int axis = 0; axis < 3; ++axis) {
        origin_[axis] = low[axis];
        const double span = std::floor((high[axis] - low[axis]) / cell_) + 1.0;
        dims_[axis] = static_cast<std::size_t>(std::min(span, 0x1p40));
        total *= span;
      }
      if (total <= cap) {
        double registrations = 0.0;
        for (std::size_t row = 0; row < size; ++row) {
          std::size_t first[3], last[3];
          if (bounded[row] != 0) registrations += static_cast<double>(range(row, first, last));
        }
        if (registrations <= 2.0 * cap || total == 1.0) break;
      }
      cell_ *= std::max(1.125, std::cbrt(total / cap));
    }
    starts_.assign(dims_[0] * dims_[1] * dims_[2] + 1, 0U);
    std::vector<uint32_t> cursor;
    for (int pass = 0; pass < 2; ++pass) {
      for (std::size_t row = 0; row < size; ++row) {
        if (bounded[row] == 0) continue;
        std::size_t first[3], last[3];
        range(row, first, last);
        for (std::size_t z = first[2]; z <= last[2]; ++z) {
          for (std::size_t y = first[1]; y <= last[1]; ++y) {
            for (std::size_t x = first[0]; x <= last[0]; ++x) {
              const std::size_t at = (z * dims_[1] + y) * dims_[0] + x;
              if (pass == 0) {
                ++starts_[at + 1];
              } else {
                rows_[cursor[at]++] = static_cast<uint32_t>(row);
              }
            }
          }
        }
      }
      if (pass == 0) {
        for (std::size_t at = 1; at < starts_.size(); ++at) starts_[at] += starts_[at - 1];
        rows_.resize(starts_.back());
        cursor.assign(starts_.begin(), starts_.end() - 1);
      }
    }
  }

  // Calls visit(row) for the rows whose boxes may lie closer than 2 `reach`
  // to `point`, in no particular order, each once: every row a distance
  // below 2 `reach` can come from.
  template <class Visit>
  void candidates(const double* point, double reach, Visit&& visit) const {
    // Subtraction, squaring, summation, and square root are monotone in each
    // absolute coordinate difference, so the clamped-box distance is also a
    // lower bound for the actual represented closest-point calculation. A
    // squared box distance of at least (2 reach)^2 (1 + 2^-40), rounded
    // down, bounds every such distance by at least 2 reach after rounding.
    const double threshold = 4.0 * reach * reach * (1.0 + 0x1p-40);
    const auto take = [&](uint32_t row) {
      const Box& box = boxes_[row];
      double squared = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        const double gap = point[axis] - std::clamp(point[axis], box.lower[axis], box.upper[axis]);
        squared += gap * gap;
      }
      if (!(squared >= threshold)) visit(static_cast<std::size_t>(row));
    };
    for (const uint32_t row : unbounded_) take(row);
    if (starts_.empty()) return;
    // A box that is not far meets the cube of half side 2 reach (1 + 2^-20)
    // around the point on every axis; its bounds, widened by a relative
    // 2^-50 of their magnitude, round outward.
    const double half = 2.0 * reach * (1.0 + 0x1p-20);
    std::size_t first[3], last[3];
    double cells = 1.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double low = point[axis] - half;
      const double high = point[axis] + half;
      first[axis] = coordinate(low - std::fabs(low) * 0x1p-50, axis);
      last[axis] = coordinate(high + std::fabs(high) * 0x1p-50, axis);
      cells *= static_cast<double>(last[axis] - first[axis] + 1);
    }
    if (!(cells <= static_cast<double>(rows_.size()))) {
      // A query wider than the registrations reads every row once.
      for (std::size_t row = 0; row < boxes_.size(); ++row) {
        if (std::isfinite(boxes_[row].lower[0]) && std::isfinite(boxes_[row].upper[0]) &&
            std::isfinite(boxes_[row].lower[1]) && std::isfinite(boxes_[row].upper[1]) &&
            std::isfinite(boxes_[row].lower[2]) && std::isfinite(boxes_[row].upper[2])) {
          take(static_cast<uint32_t>(row));
        }
      }
      return;
    }
    if (++stamp_ == 0) {
      std::fill(stamps_.begin(), stamps_.end(), 0U);
      stamp_ = 1;
    }
    for (std::size_t z = first[2]; z <= last[2]; ++z) {
      for (std::size_t y = first[1]; y <= last[1]; ++y) {
        const std::size_t base = (z * dims_[1] + y) * dims_[0];
        for (uint32_t index = starts_[base + first[0]]; index < starts_[base + last[0] + 1];
             ++index) {
          const uint32_t row = rows_[index];
          if (stamps_[row] == stamp_) continue;
          stamps_[row] = stamp_;
          take(row);
        }
      }
    }
  }

  // The rows with finite boxes cell by cell (each once for points, whose
  // boxes are single cells), then the others.
  std::vector<uint32_t> order() const {
    std::vector<uint32_t> rows = rows_;
    rows.insert(rows.end(), unbounded_.begin(), unbounded_.end());
    return rows;
  }

 private:
  // The clamped cell coordinate of `value` on `axis`.
  std::size_t coordinate(double value, int axis) const noexcept {
    const double scaled = std::floor((value - origin_[axis]) / cell_);
    if (!(scaled > 0.0)) return 0;
    const double top = static_cast<double>(dims_[axis] - 1);
    return scaled < top ? static_cast<std::size_t>(scaled) : dims_[axis] - 1;
  }

  std::vector<Box> boxes_;
  std::vector<uint32_t> unbounded_;
  double origin_[3]{};
  double cell_ = 1.0;
  std::size_t dims_[3]{};
  std::vector<uint32_t> starts_;
  std::vector<uint32_t> rows_;
  mutable std::vector<uint32_t> stamps_;
  mutable uint32_t stamp_ = 0;
};

constexpr double kMinimumParameter = 0x1p-500;

bool parameter_admissible(double value) {
  return std::isfinite(value) && value >= 0.0 && value <= 1.0 &&
         (value == 0.0 || value >= kMinimumParameter);
}

int closer(const Expansion& value, double first, double second) {
  Expansion a = value - Expansion(first);
  Expansion b = value - Expansion(second);
  if (a.sign() < 0) a = -a;
  if (b.sign() < 0) b = -b;
  return (a - b).sign();
}

bool even_significand(double x) {
  return (std::bit_cast<std::uint64_t>(x) & 1U) == 0U;
}

void clamp_simplex(double* parameters) {
  double& first = parameters[0];
  double& second = parameters[1];
  first = std::clamp(std::isfinite(first) ? first : 0.0, 0.0, 1.0);
  second = std::clamp(std::isfinite(second) ? second : 0.0, 0.0, 1.0);
  if (first > 0.0 && first < kMinimumParameter) first = 0.0;
  if (second > 0.0 && second < kMinimumParameter) second = 0.0;
  const double total = first + second;
  if (total > 1.0) {
    first /= total;
    second = 1.0 - first;
  }
  while ((Expansion::sum(first, second) - Expansion(1.0)).sign() > 0) {
    second = std::nextafter(second, 0.0);
  }
  if (second > 0.0 && second < kMinimumParameter) second = 0.0;
}

}  // namespace

std::vector<double> protection_radii(const double* points, int64_t point_count,
                                     std::span<const int32_t> edges,
                                     std::span<const int32_t> triangles,
                                     const double* affine_metric) {
  const std::size_t count = static_cast<std::size_t>(point_count);
  // Translation is irrelevant to protection geometry; centering avoids
  // losing small distances to a large coordinate origin.
  std::vector<double> world_points;
  if (affine_metric != nullptr && count != 0) {
    world_points.resize(3 * count);
    for (std::size_t v = 0; v < count; ++v) {
      const double delta[3] = {points[3 * v] - points[0],
                               points[3 * v + 1] - points[1],
                               points[3 * v + 2] - points[2]};
      for (int axis = 0; axis < 3; ++axis) {
        world_points[3 * v + axis] = affine_metric[3 * axis] * delta[0] +
                                     affine_metric[3 * axis + 1] * delta[1] +
                                     affine_metric[3 * axis + 2] * delta[2];
      }
    }
    points = world_points.data();
  }
  // Incident vertices per vertex, in edge order.
  std::vector<std::size_t> offsets(count + 1, 0);
  for (std::size_t e = 0; e + 1 < edges.size(); e += 2) {
    ++offsets[static_cast<std::size_t>(edges[e]) + 1];
    ++offsets[static_cast<std::size_t>(edges[e + 1]) + 1];
  }
  std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
  std::vector<int32_t> incident(offsets.back());
  {
    std::vector<std::size_t> next(offsets.begin(), offsets.end() - 1);
    for (std::size_t e = 0; e + 1 < edges.size(); e += 2) {
      incident[next[static_cast<std::size_t>(edges[e])]++] = edges[e + 1];
      incident[next[static_cast<std::size_t>(edges[e + 1])]++] = edges[e];
    }
  }
  std::vector<double> radii(count, 0.0);
  std::vector<char> acute(count, 0);
  for (std::size_t v = 0; v < count; ++v) {
    const double* p = points + 3 * v;
    const std::span<const int32_t> around(incident.data() + offsets[v],
                                          offsets[v + 1] - offsets[v]);
    for (std::size_t i = 0; i < around.size() && acute[v] == 0; ++i) {
      const double* a = points + 3 * static_cast<std::size_t>(around[i]);
      const double u[3] = {a[0] - p[0], a[1] - p[1], a[2] - p[2]};
      for (std::size_t j = i + 1; j < around.size() && acute[v] == 0; ++j) {
        const double* b = points + 3 * static_cast<std::size_t>(around[j]);
        const double w[3] = {b[0] - p[0], b[1] - p[1], b[2] - p[2]};
        acute[v] = dot(u, w) > 0.0 ? 1 : 0;
      }
    }
  }
  if (std::find(acute.begin(), acute.end(), 1) == acute.end()) return radii;
  // The initial reach of an acute vertex: a third of its shortest edge.
  std::vector<double> reaches(count, std::numeric_limits<double>::infinity());
  std::vector<double> sample;
  for (std::size_t v = 0; v < count; ++v) {
    if (acute[v] == 0) continue;
    const double* p = points + 3 * v;
    for (std::size_t at = offsets[v]; at < offsets[v + 1]; ++at) {
      reaches[v] = std::min(
          reaches[v], distance(p, points + 3 * static_cast<std::size_t>(incident[at])) / 3.0);
    }
    if (std::isfinite(reaches[v])) sample.push_back(reaches[v]);
  }
  // Grid cells as wide as the typical query (2 reach around the vertex).
  double side = 0.0;
  if (!sample.empty()) {
    const auto middle = sample.begin() + static_cast<std::ptrdiff_t>(sample.size() / 2);
    std::nth_element(sample.begin(), middle, sample.end());
    side = 4.0 * *middle;
  }
  // A triangle's distance from v is at most that of each of its corners and
  // edges, so a point or edge on a triangle not incident to v never lowers
  // v's reach: an edge of a triangle is dropped (the triangles on it that
  // contain v hold it as v's opposite edge, measured first), and so is a
  // vertex whose triangles share no vertex but itself (for every v some
  // triangle at it excludes v).
  std::vector<std::array<int32_t, 2>> shared(count, {-2, -2});
  std::vector<std::uint64_t> triangle_edges;
  triangle_edges.reserve(triangles.size());
  const auto edge_key = [](int32_t a, int32_t b) {
    return (static_cast<std::uint64_t>(static_cast<uint32_t>(std::min(a, b))) << 32) |
           static_cast<uint32_t>(std::max(a, b));
  };
  for (std::size_t row = 0; row + 2 < triangles.size(); row += 3) {
    for (std::size_t corner = 0; corner < 3; ++corner) {
      const int32_t vertex = triangles[row + corner];
      const int32_t first = triangles[row + (corner + 1) % 3];
      const int32_t second = triangles[row + (corner + 2) % 3];
      triangle_edges.push_back(edge_key(first, second));
      auto& common = shared[static_cast<std::size_t>(vertex)];
      if (common[0] == -2) {
        common = {first, second};
      } else {
        for (int32_t& other : common) {
          if (other != first && other != second) other = -1;
        }
      }
      if (acute[static_cast<std::size_t>(vertex)] != 0 && first != second && first != vertex &&
          second != vertex) {
        double& reach = reaches[static_cast<std::size_t>(vertex)];
        reach = std::min(reach, 0.5 * point_segment_distance(
                                    points + 3 * static_cast<std::size_t>(vertex),
                                    points + 3 * static_cast<std::size_t>(std::min(first, second)),
                                    points + 3 * static_cast<std::size_t>(std::max(first, second))));
      }
    }
  }
  std::sort(triangle_edges.begin(), triangle_edges.end());
  std::vector<int32_t> lone_points;
  std::vector<double> lone_coordinates;
  for (std::size_t v = 0; v < count; ++v) {
    if (shared[v][0] == -2 || shared[v][0] >= 0 || shared[v][1] >= 0) {
      lone_points.push_back(static_cast<int32_t>(v));
      lone_coordinates.insert(lone_coordinates.end(), points + 3 * v, points + 3 * v + 3);
    }
  }
  std::vector<int32_t> lone_edges;
  for (std::size_t e = 0; e + 1 < edges.size(); e += 2) {
    if (!std::binary_search(triangle_edges.begin(), triangle_edges.end(),
                            edge_key(edges[e], edges[e + 1]))) {
      lone_edges.push_back(edges[e]);
      lone_edges.push_back(edges[e + 1]);
    }
  }
  ProtectionIndex point_index, segment_index, triangle_index, order_index;
  order_index.build(points, count, {}, 1, side);
  point_index.build(lone_coordinates.data(), lone_points.size(), {}, 1, side);
  if (!lone_edges.empty()) segment_index.build(points, 0, lone_edges, 2, side);
  triangle_index.build(points, 0, triangles, 3, side);
  // Each radius depends only on its vertex; the vertices go cell by cell,
  // so consecutive queries read nearby cells.
  for (const uint32_t vertex : order_index.order()) {
    const std::size_t v = vertex;
    if (acute[v] == 0) continue;
    const double* p = points + 3 * v;
    double reach = reaches[v];
    point_index.candidates(p, reach, [&](std::size_t row) {
      const std::size_t u = static_cast<std::size_t>(lone_points[row]);
      if (u != v) reach = std::min(reach, 0.5 * distance(p, points + 3 * u));
    });
    segment_index.candidates(p, reach, [&](std::size_t row) {
      const std::size_t e = 2 * row;
      if (static_cast<std::size_t>(lone_edges[e]) != v &&
          static_cast<std::size_t>(lone_edges[e + 1]) != v) {
        reach = std::min(reach, 0.5 * point_segment_distance(
                                    p, points + 3 * static_cast<std::size_t>(lone_edges[e]),
                                    points + 3 * static_cast<std::size_t>(lone_edges[e + 1])));
      }
    });
    // The minimum is independent of row order, so rows whose boxes are
    // farther than 2 reach are skipped.
    triangle_index.candidates(p, reach, [&](std::size_t row) {
      const std::size_t a = static_cast<std::size_t>(triangles[3 * row]);
      const std::size_t b = static_cast<std::size_t>(triangles[3 * row + 1]);
      const std::size_t c = static_cast<std::size_t>(triangles[3 * row + 2]);
      if (a != v && b != v && c != v) {
        reach = std::min(reach, 0.5 * point_triangle_distance(
                                    p, points + 3 * a, points + 3 * b, points + 3 * c));
      }
    });
    radii[v] = reach;
  }
  return radii;
}

bool source_parameters_valid(SourceStratum stratum, const double* parameters) {
  switch (stratum) {
    case SourceStratum::kSegment:
      return parameter_admissible(parameters[0]);
    case SourceStratum::kFacet:
      return parameter_admissible(parameters[0]) && parameter_admissible(parameters[1]) &&
             (Expansion::sum(parameters[0], parameters[1]) - Expansion(1.0)).sign() <= 0;
    // Declared planes have unrestricted chart coordinates and are realized
    // through TetMesh::plane_carrier, not this bounded simplex interface.
    case SourceStratum::kPlane:
      return false;
    case SourceStratum::kNone:
      return false;
  }
  return false;
}

bool source_point(const double* const* corners, SourceStratum stratum,
                  const double* parameters, Expansion* point) {
  if (!source_parameters_valid(stratum, parameters)) return false;
  for (int k = 0; k < 3; ++k) {
    point[k] = Expansion(corners[0][k]) +
               Expansion::difference(corners[1][k], corners[0][k]).scaled(parameters[0]);
    if (stratum == SourceStratum::kFacet) {
      point[k] = point[k] +
                 Expansion::difference(corners[2][k], corners[0][k]).scaled(parameters[1]);
    }
  }
  return true;
}

double nearest_double(const Expansion& value) {
  double x = value.estimate();
  for (;;) {
    const double next = x + (value - Expansion(x)).estimate();
    if (next == x || closer(value, next, x) >= 0) break;
    x = next;
  }
  for (;;) {
    const double down = geometry::next_down(x);
    const double up = geometry::next_up(x);
    const int below = closer(value, down, x);
    const int above = closer(value, up, x);
    if (below < 0 || (below == 0 && even_significand(down))) {
      x = down;
    } else if (above < 0 || (above == 0 && even_significand(up))) {
      x = up;
    } else {
      return x;
    }
  }
}

double outward_magnitude(const Expansion& value) {
  const Expansion magnitude = value.sign() < 0 ? -value : value;
  if (magnitude.is_zero()) return 0.0;
  double bound = std::fabs(magnitude.estimate());
  while ((Expansion(bound) - magnitude).sign() < 0) bound = geometry::next_up(bound);
  return bound;
}

double outward_distance(const double* position, const Expansion* point) {
  double components[3];
  double scale = 0.0;
  for (int k = 0; k < 3; ++k) {
    components[k] = outward_magnitude(Expansion(position[k]) - point[k]);
    scale = std::max(scale, components[k]);
  }
  if (scale == 0.0) return 0.0;
  double sum = 0.0;
  for (double component : components) {
    const double ratio = component / scale;
    sum += ratio * ratio;
  }
  const double norm = scale * std::sqrt(sum) * (1.0 + 0x1p-49);
  return geometry::next_up(norm);
}

bool on_source_entity(const double* const* corners, SourceStratum stratum,
                      const double* position) {
  switch (stratum) {
    case SourceStratum::kSegment:
      if (!geometry::collinear(corners[0], corners[1], position)) return false;
      for (int k = 0; k < 3; ++k) {
        if (position[k] < std::min(corners[0][k], corners[1][k]) ||
            position[k] > std::max(corners[0][k], corners[1][k])) return false;
      }
      return true;
    case SourceStratum::kFacet: {
      const double* const triangle[3] = {corners[0], corners[1], corners[2]};
      int side = 0;
      return locate_on_triangle(position, triangle, side) != kFeatureNone;
    }
    // A declared plane cannot be established from simplex corners: its
    // authority and admission tolerance belong to the owning TetMesh.
    case SourceStratum::kPlane:
      return false;
    case SourceStratum::kNone:
      return false;
  }
  return false;
}

double source_deviation(const double* const* corners, SourceStratum stratum,
                        const double* parameters, const double* position) {
  Expansion point[3];
  if (!source_point(corners, stratum, parameters, point)) return -1.0;
  return on_source_entity(corners, stratum, position) ? 0.0 : outward_distance(position, point);
}

bool source_carrier(const double* const* corners, SourceStratum stratum,
                    const double* parameters, double* position, double& deviation) {
  Expansion point[3];
  if (!source_point(corners, stratum, parameters, point)) return false;
  for (int k = 0; k < 3; ++k) position[k] = nearest_double(point[k]);
  if (!geometry::point_in_domain(position)) return false;
  deviation = on_source_entity(corners, stratum, position) ? 0.0 : outward_distance(position, point);
  return true;
}

void source_locator(const double* const* corners, SourceStratum stratum,
                    const double* position, double* parameters, bool clamp) {
  const double* a = corners[0];
  const double* b = corners[1];
  if (stratum == SourceStratum::kSegment) {
    const double d[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const double r[3] = {position[0] - a[0], position[1] - a[1], position[2] - a[2]};
    parameters[0] = dot(r, d) / dot(d, d);
    if (clamp) clamp_source_parameters(stratum, parameters);
    return;
  }
  const double* c = corners[2];
  const double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const double v[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const double n[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                       u[0] * v[1] - u[1] * v[0]};
  int axis = 0;
  for (int k = 1; k < 3; ++k) if (std::fabs(n[k]) > std::fabs(n[axis])) axis = k;
  const int x = (axis + 1) % 3, y = (axis + 2) % 3;
  const double r[2] = {position[x] - a[x], position[y] - a[y]};
  const double area = u[x] * v[y] - u[y] * v[x];
  parameters[0] = (r[0] * v[y] - r[1] * v[x]) / area;
  parameters[1] = (u[x] * r[1] - u[y] * r[0]) / area;
  if (clamp) clamp_source_parameters(stratum, parameters);
}

void clamp_source_parameters(SourceStratum stratum, double* parameters) {
  if (stratum == SourceStratum::kFacet) {
    clamp_simplex(parameters);
    return;
  }
  double& t = parameters[0];
  t = std::clamp(std::isfinite(t) ? t : 0.0, 0.0, 1.0);
  if (t > 0.0 && t < kMinimumParameter) t = 0.0;
  parameters[1] = 0.0;
}

namespace {

void exact_point(const ExactPointRef& point, Expansion* coordinates) {
  for (int axis = 0; axis < 3; ++axis) coordinates[axis] = coordinate(point, axis);
}

// Upper bounds of |exact - carrier|, one per coordinate (0 for exact carriers).
void carrier_deviation(const ExactPointRef& point, double* deviation) {
  for (int axis = 0; axis < 3; ++axis) {
    deviation[axis] =
        point.exact == nullptr     ? 0.0
        : point.deviation != nullptr ? point.deviation[axis]
                                     : outward_magnitude(point.exact[axis] - Expansion(point.carrier[axis]));
  }
}

// Bound on |det(exact) - fl(det(carrier differences))| for a determinant of
// difference rows u (binary64, |u - exact| <= e per entry): `perturbed` and
// `plain` are the permanents of |u| + e and |u| evaluated in binary64. The
// permanent difference bounds the perturbation (every term of the
// multilinear expansion with a perturbed factor is at most its absolute
// value), 2^-48 plain bounds the evaluation error, and the widenings by
// 2^-40 absorb the roundings of the bound itself; 2^-1000 covers underflow.
double determinant_error(double perturbed, double plain) {
  const double widen = 1.0 + 0x1p-40;
  return (perturbed * widen - plain * (1.0 - 0x1p-40) + 0x1p-48 * plain) * widen + 0x1p-1000;
}

// Sign of orient3d(a, b, c, d) on the exact points (the sign of
// -det[a - d, b - d, c - d], as orient3d_sign) when binary64 evaluation on
// the carriers decides it, else 0.
int filtered_orient3d(const ExactPointRef& a, const ExactPointRef& b, const ExactPointRef& c,
                      const ExactPointRef& d) {
  const ExactPointRef* rows[3] = {&a, &b, &c};
  double base[3];
  carrier_deviation(d, base);
  double u[3][3], magnitude[3][3], widened[3][3];
  for (int row = 0; row < 3; ++row) {
    double deviation[3];
    carrier_deviation(*rows[row], deviation);
    for (int axis = 0; axis < 3; ++axis) {
      u[row][axis] = rows[row]->carrier[axis] - d.carrier[axis];
      magnitude[row][axis] = std::fabs(u[row][axis]);
      widened[row][axis] =
          magnitude[row][axis] + (0x1p-52 * magnitude[row][axis] + deviation[axis] + base[axis]);
    }
  }
  const auto permanent = [](const double (*m)[3]) {
    return m[0][0] * (m[1][1] * m[2][2] + m[1][2] * m[2][1]) +
           m[0][1] * (m[1][0] * m[2][2] + m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] + m[1][1] * m[2][0]);
  };
  const double determinant = u[0][0] * (u[1][1] * u[2][2] - u[1][2] * u[2][1]) -
                             u[0][1] * (u[1][0] * u[2][2] - u[1][2] * u[2][0]) +
                             u[0][2] * (u[1][0] * u[2][1] - u[1][1] * u[2][0]);
  const double bound = determinant_error(permanent(widened), permanent(magnitude));
  if (!std::isfinite(determinant) || !std::isfinite(bound) || !(std::fabs(determinant) > bound)) {
    return 0;
  }
  return determinant > 0.0 ? -1 : 1;
}

// Sign of insphere(a, b, c, d, e) on the exact points (the sign of minus
// the lifted determinant of the rows p - e, |p - e|^2, as insphere_sign)
// when binary64 evaluation on the carriers decides it, else 0. A lift
// entry is bounded by the squared widened differences: |L - fl(lift)| is
// at most their sum minus fl(lift), up to the 2^-40 widening.
int filtered_insphere(const ExactPointRef& a, const ExactPointRef& b, const ExactPointRef& c,
                      const ExactPointRef& d, const ExactPointRef& e) {
  const ExactPointRef* rows[4] = {&a, &b, &c, &d};
  double base[3];
  carrier_deviation(e, base);
  // Columns x, y, z, lift of the values, their magnitudes and widened bounds.
  double value[4][4], magnitude[4][4], widened[4][4];
  for (int row = 0; row < 4; ++row) {
    double deviation[3];
    carrier_deviation(*rows[row], deviation);
    double lift = 0.0, lift_bound = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double difference = rows[row]->carrier[axis] - e.carrier[axis];
      value[axis][row] = difference;
      magnitude[axis][row] = std::fabs(difference);
      widened[axis][row] = magnitude[axis][row] + (0x1p-52 * magnitude[axis][row] +
                                                   deviation[axis] + base[axis]);
      lift += difference * difference;
      lift_bound += widened[axis][row] * widened[axis][row];
    }
    value[3][row] = lift;
    magnitude[3][row] = lift;
    widened[3][row] = lift_bound * (1.0 + 0x1p-40);
  }
  const auto permanent = [](const double (*m)[4]) {
    const double* x = m[0];
    const double* y = m[1];
    const double* z = m[2];
    const double* l = m[3];
    const double ab = x[0] * y[1] + x[1] * y[0], bc = x[1] * y[2] + x[2] * y[1];
    const double cd = x[2] * y[3] + x[3] * y[2], da = x[3] * y[0] + x[0] * y[3];
    const double ac = x[0] * y[2] + x[2] * y[0], bd = x[1] * y[3] + x[3] * y[1];
    const double abc = z[0] * bc + z[1] * ac + z[2] * ab;
    const double bcd = z[1] * cd + z[2] * bd + z[3] * bc;
    const double cda = z[2] * da + z[3] * ac + z[0] * cd;
    const double dab = z[3] * ab + z[0] * bd + z[1] * da;
    return l[3] * abc + l[2] * dab + l[1] * cda + l[0] * bcd;
  };
  const double* x = value[0];
  const double* y = value[1];
  const double* z = value[2];
  const double* l = value[3];
  const double ab = x[0] * y[1] - x[1] * y[0], bc = x[1] * y[2] - x[2] * y[1];
  const double cd = x[2] * y[3] - x[3] * y[2], da = x[3] * y[0] - x[0] * y[3];
  const double ac = x[0] * y[2] - x[2] * y[0], bd = x[1] * y[3] - x[3] * y[1];
  const double abc = z[0] * bc - z[1] * ac + z[2] * ab;
  const double bcd = z[1] * cd - z[2] * bd + z[3] * bc;
  const double cda = z[2] * da + z[3] * ac + z[0] * cd;
  const double dab = z[3] * ab + z[0] * bd + z[1] * da;
  const double determinant = (l[3] * abc - l[2] * dab) + (l[1] * cda - l[0] * bcd);
  // The 4x4 evaluation error is below 2^-46 of the permanent.
  const double plain = permanent(magnitude);
  const double bound = determinant_error(permanent(widened), plain) + 0x1p-46 * plain;
  if (!std::isfinite(determinant) || !std::isfinite(bound) || !(std::fabs(determinant) > bound)) {
    return 0;
  }
  return determinant > 0.0 ? -1 : 1;
}

// Exact stages of the ExactPointRef predicates, out of line: their
// expansion coordinates would otherwise enlarge the stack frame of every
// call, including the binary64 fast path.
[[gnu::noinline, gnu::cold]] int exact_orient3d(const ExactPointRef& a, const ExactPointRef& b,
                                                const ExactPointRef& c, const ExactPointRef& d,
                                                const int64_t* ids) {
  Expansion pa[3], pb[3], pc[3], pd[3];
  exact_point(a, pa);
  exact_point(b, pb);
  exact_point(c, pc);
  exact_point(d, pd);
  return ids == nullptr ? orient3d_sign(pa, pb, pc, pd)
                        : orient3d_sos_sign(pa, pb, pc, pd, ids[0], ids[1], ids[2], ids[3]);
}

[[gnu::noinline, gnu::cold]] int exact_insphere_sos(
    const ExactPointRef& a, const ExactPointRef& b, const ExactPointRef& c,
    const ExactPointRef& d, const ExactPointRef& e, const int64_t* ids) {
  Expansion pa[3], pb[3], pc[3], pd[3], pe[3];
  exact_point(a, pa);
  exact_point(b, pb);
  exact_point(c, pc);
  exact_point(d, pd);
  exact_point(e, pe);
  return insphere_sos_sign(pa, pb, pc, pd, pe, ids[0], ids[1], ids[2], ids[3], ids[4]);
}

}  // namespace

Expansion coordinate(const ExactPointRef& point, int axis) {
  return point.exact != nullptr ? point.exact[axis] : Expansion(point.carrier[axis]);
}

int orient3d(const ExactPointRef& a, const ExactPointRef& b, const ExactPointRef& c,
             const ExactPointRef& d) {
  if (a.exact == nullptr && b.exact == nullptr && c.exact == nullptr && d.exact == nullptr) {
    return orient3d(a.carrier, b.carrier, c.carrier, d.carrier);
  }
  if (const int sign = filtered_orient3d(a, b, c, d); sign != 0) return sign;
  return exact_orient3d(a, b, c, d, nullptr);
}

int orient3d_sos(const ExactPointRef& a, const ExactPointRef& b, const ExactPointRef& c,
                 const ExactPointRef& d, int64_t ia, int64_t ib, int64_t ic, int64_t id) {
  if (a.exact == nullptr && b.exact == nullptr && c.exact == nullptr && d.exact == nullptr) {
    return orient3d_sos(a.carrier, b.carrier, c.carrier, d.carrier, ia, ib, ic, id);
  }
  if (const int sign = filtered_orient3d(a, b, c, d); sign != 0) return sign;
  const int64_t ids[4] = {ia, ib, ic, id};
  return exact_orient3d(a, b, c, d, ids);
}

int insphere_sos(const ExactPointRef& a, const ExactPointRef& b, const ExactPointRef& c,
                 const ExactPointRef& d, const ExactPointRef& e, int64_t ia, int64_t ib,
                 int64_t ic, int64_t id, int64_t ie) {
  if (a.exact == nullptr && b.exact == nullptr && c.exact == nullptr &&
      d.exact == nullptr && e.exact == nullptr) {
    return insphere_sos(a.carrier, b.carrier, c.carrier, d.carrier, e.carrier, ia, ib, ic, id,
                        ie);
  }
  if (const int sign = filtered_insphere(a, b, c, d, e); sign != 0) return sign;
  const int64_t ids[5] = {ia, ib, ic, id, ie};
  return exact_insphere_sos(a, b, c, d, e, ids);
}

int affine_insphere_sos(const double* linear, int orientation,
                        const ExactPointRef& a, const ExactPointRef& b,
                        const ExactPointRef& c, const ExactPointRef& d,
                        const ExactPointRef& e, int64_t ia, int64_t ib,
                        int64_t ic, int64_t id, int64_t ie) {
  const ExactPointRef* points[4] = {&a, &b, &c, &d};
  Expansion transformed[5][3];
  Expansion origin[3];
  for (int axis = 0; axis < 3; ++axis) origin[axis] = coordinate(e, axis);
  for (int vertex = 0; vertex < 4; ++vertex) {
    Expansion delta[3];
    for (int axis = 0; axis < 3; ++axis) {
      delta[axis] = coordinate(*points[vertex], axis) - origin[axis];
    }
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        transformed[vertex][row] = transformed[vertex][row] +
            Expansion(linear[3 * row + col]) * delta[col];
      }
    }
  }
  return orientation * insphere_sos_sign(
      transformed[0], transformed[1], transformed[2], transformed[3],
      transformed[4], ia, ib, ic, id, ie);
}

int orient2d_projected(const ExactPointRef& a, const ExactPointRef& b, const ExactPointRef& c,
                       int axis) {
  const int x = (axis + 1) % 3;
  const int y = (axis + 2) % 3;
  if (a.exact == nullptr && b.exact == nullptr && c.exact == nullptr) {
    const double pa[2] = {a.carrier[x], a.carrier[y]};
    const double pb[2] = {b.carrier[x], b.carrier[y]};
    const double pc[2] = {c.carrier[x], c.carrier[y]};
    return orient2d(pa, pb, pc);
  }
  // Binary64 evaluation on the carriers decides when it clears the bound
  // of determinant_error; otherwise the exact positions do.
  double deviation[3][3];
  carrier_deviation(a, deviation[0]);
  carrier_deviation(b, deviation[1]);
  carrier_deviation(c, deviation[2]);
  // Rows a - c and b - c of the projected determinant.
  double u[2][2], magnitude[2][2], widened[2][2];
  const ExactPointRef* rows[2] = {&a, &b};
  for (int row = 0; row < 2; ++row) {
    for (int column = 0; column < 2; ++column) {
      const int coordinate_axis = column == 0 ? x : y;
      u[row][column] = rows[row]->carrier[coordinate_axis] - c.carrier[coordinate_axis];
      magnitude[row][column] = std::fabs(u[row][column]);
      widened[row][column] =
          magnitude[row][column] + (0x1p-52 * magnitude[row][column] +
                                    deviation[row][coordinate_axis] + deviation[2][coordinate_axis]);
    }
  }
  const double determinant = u[0][0] * u[1][1] - u[0][1] * u[1][0];
  const double bound =
      determinant_error(widened[0][0] * widened[1][1] + widened[0][1] * widened[1][0],
                        magnitude[0][0] * magnitude[1][1] + magnitude[0][1] * magnitude[1][0]);
  if (std::isfinite(determinant) && std::isfinite(bound) && std::fabs(determinant) > bound) {
    return determinant > 0.0 ? 1 : -1;
  }
  const Expansion pa[2] = {coordinate(a, x), coordinate(a, y)};
  const Expansion pb[2] = {coordinate(b, x), coordinate(b, y)};
  const Expansion pc[2] = {coordinate(c, x), coordinate(c, y)};
  return orient2d_sign(pa, pb, pc);
}

bool same_point(const ExactPointRef& a, const ExactPointRef& b) {
  if (a.exact == nullptr && b.exact == nullptr) {
    return geometry::same_point(a.carrier, b.carrier);
  }
  for (int axis = 0; axis < 3; ++axis) {
    if ((coordinate(a, axis) - coordinate(b, axis)).sign() != 0) return false;
  }
  return true;
}

bool collinear(const ExactPointRef& a, const ExactPointRef& b, const ExactPointRef& c) {
  if (a.exact == nullptr && b.exact == nullptr && c.exact == nullptr) {
    return geometry::collinear(a.carrier, b.carrier, c.carrier);
  }
  for (int axis = 0; axis < 3; ++axis) {
    if (orient2d_projected(a, b, c, axis) != 0) return false;
  }
  return true;
}

int8_t locate_on_triangle(const ExactPointRef& x, const ExactPointRef* triangle, int& side) {
  if (x.exact == nullptr && triangle[0].exact == nullptr && triangle[1].exact == nullptr &&
      triangle[2].exact == nullptr) {
    const double* const corners[3] = {triangle[0].carrier, triangle[1].carrier,
                                      triangle[2].carrier};
    return locate_on_triangle(x.carrier, corners, side);
  }
  side = orient3d(triangle[0], triangle[1], triangle[2], x);
  if (side != 0) return kFeatureNone;
  for (int axis = 0; axis < 3; ++axis) {
    const int orientation = orient2d_projected(triangle[0], triangle[1], triangle[2], axis);
    if (orientation == 0) continue;
    int signs[3];
    int zeros = 0;
    for (int k = 0; k < 3; ++k) {
      signs[k] = orientation *
                 orient2d_projected(triangle[(k + 1) % 3], triangle[(k + 2) % 3], x, axis);
      if (signs[k] < 0) return kFeatureNone;
      zeros += signs[k] == 0 ? 1 : 0;
    }
    if (zeros == 0) return kFeatureInterior;
    for (int k = 0; k < 3; ++k) {
      if (zeros == 1 && signs[k] == 0) return static_cast<int8_t>(3 + k);
      if (zeros == 2 && signs[k] != 0) return static_cast<int8_t>(k);
    }
    return kFeatureNone;
  }
  return kFeatureNone;
}

void segment_witness_position(const double* first, const double* second, double parameter,
                              Expansion* position) {
  for (int axis = 0; axis < 3; ++axis) {
    position[axis] = Expansion(first[axis]) +
                     Expansion::difference(second[axis], first[axis]).scaled(parameter);
  }
}

}  // namespace phx::mc
