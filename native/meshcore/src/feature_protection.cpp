//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Numerical feature protection and exact source-ancestry construction shared
// by constrained recovery and tetrahedral refinement. Protection radii only
// schedule insertions; exact predicates, source witnesses, and cavity
// transactions decide geometric validity.
#include <algorithm>
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

double point_segment_distance(const double* p, const double* a, const double* b) {
  const double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const double ap[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
  const double length = dot(ab, ab);
  const double t = length > 0.0 ? std::clamp(dot(ap, ab) / length, 0.0, 1.0) : 0.0;
  const double q[3] = {a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]};
  return distance(p, q);
}

// Closest point on a triangle by Voronoi-region classification (Ericson).
double point_triangle_distance(const double* p, const double* a, const double* b,
                               const double* c) {
  const double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const double ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const double ap[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
  const double d1 = dot(ab, ap);
  const double d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) return distance(p, a);
  const double bp[3] = {p[0] - b[0], p[1] - b[1], p[2] - b[2]};
  const double d3 = dot(ab, bp);
  const double d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) return distance(p, b);
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return point_segment_distance(p, a, b);
  const double cp[3] = {p[0] - c[0], p[1] - c[1], p[2] - c[2]};
  const double d5 = dot(ab, cp);
  const double d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) return distance(p, c);
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return point_segment_distance(p, a, c);
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    return point_segment_distance(p, b, c);
  }
  const double denominator = va + vb + vc;
  if (!(denominator > 0.0)) {
    return std::min({point_segment_distance(p, a, b), point_segment_distance(p, b, c),
                     point_segment_distance(p, c, a)});
  }
  const double v = vb / denominator;
  const double w = vc / denominator;
  const double q[3] = {a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w,
                       a[2] + ab[2] * v + ac[2] * w};
  return distance(p, q);
}

// This index encloses the computed closest-point kernels, not only ideal
// segments: a + t*(b-a) may end at a rounded value distinct from b.
class ProtectionIndex {
  struct Node {
    double lower[3]{};
    double upper[3]{};
    std::size_t left = 0;
    std::size_t right = 0;
    std::size_t row = std::numeric_limits<std::size_t>::max();
  };

 public:
  void build(const double* points, std::size_t count, std::span<const int32_t> edges) {
    const bool segments = !edges.empty();
    const std::size_t size = segments ? edges.size() / 2 : count;
    if (size == 0) return;
    nodes_.reserve(2 * size - 1);
    std::vector<std::size_t> order(size);
    std::iota(order.begin(), order.end(), std::size_t{0});
    for (std::size_t row = 0; row < size; ++row) {
      Node node;
      node.row = row;
      const double* a = points + 3 * (segments ? edges[2 * row] : row);
      for (int axis = 0; axis < 3; ++axis) {
        node.lower[axis] = node.upper[axis] = a[axis];
        if (segments) {
          const double b = points[3 * static_cast<std::size_t>(edges[2 * row + 1]) + axis];
          const double reconstructed = a[axis] + (b - a[axis]);
          if (std::isfinite(reconstructed)) {
            node.lower[axis] = std::min({a[axis], b, reconstructed});
            node.upper[axis] = std::max({a[axis], b, reconstructed});
          } else {
            node.lower[axis] = -std::numeric_limits<double>::infinity();
            node.upper[axis] = std::numeric_limits<double>::infinity();
          }
        }
      }
      nodes_.push_back(node);
    }
    const auto build_range = [&](auto&& self, std::size_t first, std::size_t end)
        -> std::size_t {
      if (end - first == 1) return order[first];
      Node node = nodes_[order[first]];
      node.row = std::numeric_limits<std::size_t>::max();
      for (std::size_t i = first + 1; i < end; ++i) {
        for (int axis = 0; axis < 3; ++axis) {
          node.lower[axis] = std::min(node.lower[axis], nodes_[order[i]].lower[axis]);
          node.upper[axis] = std::max(node.upper[axis], nodes_[order[i]].upper[axis]);
        }
      }
      int axis = 0;
      for (int k = 1; k < 3; ++k) {
        if (node.upper[k] - node.lower[k] > node.upper[axis] - node.lower[axis]) axis = k;
      }
      const auto center = [&](std::size_t row) {
        const double value = std::midpoint(nodes_[row].lower[axis], nodes_[row].upper[axis]);
        return std::isfinite(value) ? value : 0.0;
      };
      const std::size_t middle = first + (end - first) / 2;
      std::nth_element(order.begin() + first, order.begin() + middle, order.begin() + end,
                       [&](std::size_t a, std::size_t b) {
                         const double x = center(a), y = center(b);
                         return x < y || (x == y && a < b);
                       });
      node.left = self(self, first, middle);
      node.right = self(self, middle, end);
      const std::size_t index = nodes_.size();
      nodes_.push_back(node);
      return index;
    };
    root_ = build_range(build_range, 0, size);
  }

  void candidates(const double* point, double reach, std::vector<std::size_t>& rows) const {
    rows.clear();
    if (nodes_.empty()) return;
    const auto query = [&](auto&& self, std::size_t index) -> void {
      const Node& node = nodes_[index];
      const double nearest[3] = {
          std::clamp(point[0], node.lower[0], node.upper[0]),
          std::clamp(point[1], node.lower[1], node.upper[1]),
          std::clamp(point[2], node.lower[2], node.upper[2])};
      // Subtraction, squaring, summation, and square root are monotone in each
      // absolute coordinate difference, so the clamped-box distance is also a
      // lower bound for the actual represented closest-point calculation.
      if (0.5 * distance(point, nearest) >= reach) return;
      if (node.row != std::numeric_limits<std::size_t>::max()) {
        rows.push_back(node.row);
      } else {
        self(self, node.left);
        self(self, node.right);
      }
    };
    query(query, root_);
    std::sort(rows.begin(), rows.end());
  }

 private:
  std::vector<Node> nodes_;
  std::size_t root_ = 0;
};

bool strictly_between(const double* a, const double* b, const double* p) {
  for (int axis = 0; axis < 3; ++axis) {
    if (a[axis] != b[axis]) {
      return a[axis] < b[axis] ? (a[axis] < p[axis] && p[axis] < b[axis])
                               : (b[axis] < p[axis] && p[axis] < a[axis]);
    }
  }
  return false;
}

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
                                     std::span<const int32_t> triangles) {
  const std::size_t count = static_cast<std::size_t>(point_count);
  std::vector<std::vector<int32_t>> incident(count);
  for (std::size_t e = 0; e + 1 < edges.size(); e += 2) {
    incident[static_cast<std::size_t>(edges[e])].push_back(edges[e + 1]);
    incident[static_cast<std::size_t>(edges[e + 1])].push_back(edges[e]);
  }
  std::vector<double> radii(count, 0.0);
  ProtectionIndex point_index, segment_index;
  std::vector<std::size_t> rows;
  bool indexed = false;
  for (std::size_t v = 0; v < count; ++v) {
    const double* p = points + 3 * v;
    const std::vector<int32_t>& around = incident[v];
    bool acute = false;
    for (std::size_t i = 0; i < around.size() && !acute; ++i) {
      const double* a = points + 3 * static_cast<std::size_t>(around[i]);
      const double u[3] = {a[0] - p[0], a[1] - p[1], a[2] - p[2]};
      for (std::size_t j = i + 1; j < around.size() && !acute; ++j) {
        const double* b = points + 3 * static_cast<std::size_t>(around[j]);
        const double w[3] = {b[0] - p[0], b[1] - p[1], b[2] - p[2]};
        acute = dot(u, w) > 0.0;
      }
    }
    if (!acute) continue;
    if (!indexed) {
      point_index.build(points, count, {});
      if (!edges.empty()) segment_index.build(points, 0, edges);
      rows.reserve(std::max(count, edges.size() / 2));
      indexed = true;
    }
    double reach = std::numeric_limits<double>::infinity();
    for (int32_t other : around) {
      reach = std::min(reach, distance(p, points + 3 * static_cast<std::size_t>(other)) / 3.0);
    }
    point_index.candidates(p, reach, rows);
    for (std::size_t u : rows) {
      if (u != v) reach = std::min(reach, 0.5 * distance(p, points + 3 * u));
    }
    segment_index.candidates(p, reach, rows);
    for (std::size_t row : rows) {
      const std::size_t e = 2 * row;
      if (static_cast<std::size_t>(edges[e]) != v &&
          static_cast<std::size_t>(edges[e + 1]) != v) {
        reach = std::min(reach, 0.5 * point_segment_distance(
                                    p, points + 3 * static_cast<std::size_t>(edges[e]),
                                    points + 3 * static_cast<std::size_t>(edges[e + 1])));
      }
    }
    // The existing Voronoi kernel has no cheap conservative triangle bounds;
    // retain its original exhaustive, deterministic row order.
    for (std::size_t t = 0; t + 2 < triangles.size(); t += 3) {
      const std::size_t a = static_cast<std::size_t>(triangles[t]);
      const std::size_t b = static_cast<std::size_t>(triangles[t + 1]);
      const std::size_t c = static_cast<std::size_t>(triangles[t + 2]);
      if (a != v && b != v && c != v) {
        reach = std::min(reach, 0.5 * point_triangle_distance(
                                    p, points + 3 * a, points + 3 * b, points + 3 * c));
      }
    }
    radii[v] = reach;
  }
  return radii;
}

double split_target(const double* a, const double* b, double apex_a, double apex_b,
                    double& tolerance) {
  const double length = distance(a, b);
  constexpr double shell_tolerance = 0x1p-8;
  const auto shell = [&](double radius, bool from_a) {
    double distance_from_apex = radius;
    if (length <= 4.0 * radius / 3.0) {
      const double k = std::round(std::log2(length / (2.0 * radius)));
      distance_from_apex = std::clamp(radius * std::exp2(k), length / 3.0, 2.0 * length / 3.0);
    }
    const double t = distance_from_apex / length;
    tolerance = shell_tolerance * std::min(t, 1.0 - t);
    return from_a ? t : 1.0 - t;
  };
  if (apex_a > 0.0 && length > 4.0 * apex_a / 3.0) return shell(apex_a, true);
  if (apex_b > 0.0 && length > 4.0 * apex_b / 3.0) return shell(apex_b, false);
  if (apex_a > 0.0) return shell(apex_a, true);
  if (apex_b > 0.0) return shell(apex_b, false);
  tolerance = 0.125;
  return 0.5;
}

bool exact_segment_point(const double* a, const double* b, double target, double tolerance,
                         double* point) {
  const double direction[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  double previous = -1.0;
  for (int resolution = 1; resolution <= 52; ++resolution) {
    const double scale = std::exp2(resolution);
    const double t = std::round(target * scale) / scale;
    if (t == previous || !(t > 0.0 && t < 1.0) || std::fabs(t - target) > tolerance) continue;
    previous = t;
    for (int axis = 0; axis < 3; ++axis) point[axis] = a[axis] + t * direction[axis];
    if (geometry::point_in_domain(point) && geometry::collinear(a, b, point) &&
        strictly_between(a, b, point)) return true;
  }
  return false;
}

bool exact_triangle_point(const double* a, const double* b, const double* c, double* point) {
  const double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const double ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const double* const triangle[3] = {a, b, c};
  for (int resolution = 2; resolution <= 52; ++resolution) {
    const double scale = std::exp2(resolution);
    const double t = std::round(scale / 3.0) / scale;
    for (int axis = 0; axis < 3; ++axis) point[axis] = a[axis] + t * ab[axis] + t * ac[axis];
    int side = 0;
    if (geometry::point_in_domain(point) &&
        locate_on_triangle(point, triangle, side) == kFeatureInterior) return true;
  }
  return false;
}

bool source_parameters_valid(SourceStratum stratum, const double* parameters) {
  switch (stratum) {
    case SourceStratum::kSegment:
      return parameter_admissible(parameters[0]);
    case SourceStratum::kFacet:
      return parameter_admissible(parameters[0]) && parameter_admissible(parameters[1]) &&
             (Expansion::sum(parameters[0], parameters[1]) - Expansion(1.0)).sign() <= 0;
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

}  // namespace

Expansion coordinate(const ExactPointRef& point, int axis) {
  return point.exact != nullptr ? point.exact[axis] : Expansion(point.carrier[axis]);
}

int orient3d(const ExactPointRef& a, const ExactPointRef& b, const ExactPointRef& c,
             const ExactPointRef& d) {
  if (a.exact == nullptr && b.exact == nullptr && c.exact == nullptr && d.exact == nullptr) {
    return orient3d(a.carrier, b.carrier, c.carrier, d.carrier);
  }
  Expansion pa[3], pb[3], pc[3], pd[3];
  exact_point(a, pa);
  exact_point(b, pb);
  exact_point(c, pc);
  exact_point(d, pd);
  return orient3d_sign(pa, pb, pc, pd);
}

int orient3d_sos(const ExactPointRef& a, const ExactPointRef& b, const ExactPointRef& c,
                 const ExactPointRef& d, int64_t ia, int64_t ib, int64_t ic, int64_t id) {
  if (a.exact == nullptr && b.exact == nullptr && c.exact == nullptr && d.exact == nullptr) {
    return orient3d_sos(a.carrier, b.carrier, c.carrier, d.carrier, ia, ib, ic, id);
  }
  Expansion pa[3], pb[3], pc[3], pd[3];
  exact_point(a, pa);
  exact_point(b, pb);
  exact_point(c, pc);
  exact_point(d, pd);
  return orient3d_sos_sign(pa, pb, pc, pd, ia, ib, ic, id);
}

int insphere_sos(const ExactPointRef& a, const ExactPointRef& b, const ExactPointRef& c,
                 const ExactPointRef& d, const ExactPointRef& e, int64_t ia, int64_t ib,
                 int64_t ic, int64_t id, int64_t ie) {
  if (a.exact == nullptr && b.exact == nullptr && c.exact == nullptr &&
      d.exact == nullptr && e.exact == nullptr) {
    return insphere_sos(a.carrier, b.carrier, c.carrier, d.carrier, e.carrier, ia, ib, ic, id,
                        ie);
  }
  Expansion pa[3], pb[3], pc[3], pd[3], pe[3];
  exact_point(a, pa);
  exact_point(b, pb);
  exact_point(c, pc);
  exact_point(d, pd);
  exact_point(e, pe);
  return insphere_sos_sign(pa, pb, pc, pd, pe, ia, ib, ic, id, ie);
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
