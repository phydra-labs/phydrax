//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Corner and curve protection and source-ancestry construction shared by
// constrained recovery and refinement.
//
// Numerical protection radii and concentric-shell insertion schedules reduce
// encroachment near acute input vertices. Distances and shell locations here
// are floating-point estimates, not certified feature-distance bounds or a
// termination theorem for arbitrarily small angles. Scientific validity is
// decided independently: every exact split must be exactly collinear/coplanar
// on the represented source, every bounded split carries an exact source
// witness with an outward deviation bound, and every cavity must preserve its
// constraints.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "intersections.hpp"
#include "plc3d.hpp"
#include "predicates.hpp"
#include "source_construction.hpp"

namespace phx::mc {
namespace {

double squared(double x) { return x * x; }

double distance(const double* a, const double* b) {
  return std::sqrt(squared(a[0] - b[0]) + squared(a[1] - b[1]) + squared(a[2] - b[2]));
}

double dot(const double* u, const double* v) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; }

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
  if (d1 <= 0.0 && d2 <= 0.0) {
    return distance(p, a);
  }
  const double bp[3] = {p[0] - b[0], p[1] - b[1], p[2] - b[2]};
  const double d3 = dot(ab, bp);
  const double d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) {
    return distance(p, b);
  }
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    return point_segment_distance(p, a, b);
  }
  const double cp[3] = {p[0] - c[0], p[1] - c[1], p[2] - c[2]};
  const double d5 = dot(ab, cp);
  const double d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) {
    return distance(p, c);
  }
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    return point_segment_distance(p, a, c);
  }
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

bool in_domain(const double* p) {
  return coordinate_in_domain(p[0]) && coordinate_in_domain(p[1]) && coordinate_in_domain(p[2]);
}

// Exact strict betweenness of a point already exactly collinear with (a, b).
bool strictly_between(const double* a, const double* b, const double* p) {
  for (int axis = 0; axis < 3; ++axis) {
    if (a[axis] != b[axis]) {
      return a[axis] < b[axis] ? (a[axis] < p[axis] && p[axis] < b[axis])
                               : (b[axis] < p[axis] && p[axis] < a[axis]);
    }
  }
  return false;
}

// Parameters below this magnitude are refused: exact expansion products of
// domain coordinate differences stay far above binary64 underflow.
constexpr double kMinimumParameter = 0x1p-500;

bool parameter_admissible(double value) {
  return std::isfinite(value) && value >= 0.0 && value <= 1.0 &&
         (value == 0.0 || value >= kMinimumParameter);
}

// Sign of |value - first| - |value - second| (exact).
int closer(const Expansion& value, double first, double second) {
  Expansion a = value - Expansion(first);
  Expansion b = value - Expansion(second);
  if (a.sign() < 0) a = -a;
  if (b.sign() < 0) b = -b;
  return (a - b).sign();
}

bool even_significand(double x) { return (std::bit_cast<std::uint64_t>(x) & 1U) == 0U; }

// Closed simplex of two barycentric weights; the complementary weight is
// checked exactly, so a rounded 1 - l1 never leaves the triangle.
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

NativeVector<double> protection_radii(const double* points, int64_t point_count,
                                     std::span<const int32_t> edges,
                                     std::span<const int32_t> triangles, int64_t& work,
                                     int64_t work_limit) {
  const std::size_t count = static_cast<std::size_t>(point_count);
  const auto spend = [&]() {
    if (work >= work_limit) {
      return false;
    }
    ++work;
    return true;
  };
  NativeVector<NativeVector<int32_t>> incident(count);
  for (std::size_t e = 0; e + 1 < edges.size(); e += 2) {
    if (!spend()) {
      return {};
    }
    incident[static_cast<std::size_t>(edges[e])].push_back(edges[e + 1]);
    incident[static_cast<std::size_t>(edges[e + 1])].push_back(edges[e]);
  }
  NativeVector<double> radii(count, 0.0);
  for (std::size_t v = 0; v < count; ++v) {
    const double* p = points + 3 * v;
    const NativeVector<int32_t>& around = incident[v];
    bool acute = false;
    for (std::size_t i = 0; i < around.size() && !acute; ++i) {
      const double* a = points + 3 * static_cast<std::size_t>(around[i]);
      const double u[3] = {a[0] - p[0], a[1] - p[1], a[2] - p[2]};
      for (std::size_t j = i + 1; j < around.size() && !acute; ++j) {
        if (!spend()) {
          return {};
        }
        const double* b = points + 3 * static_cast<std::size_t>(around[j]);
        const double w[3] = {b[0] - p[0], b[1] - p[1], b[2] - p[2]};
        acute = dot(u, w) > 0.0;
      }
    }
    if (!acute) {
      continue;
    }
    double reach = std::numeric_limits<double>::infinity();
    for (int32_t other : around) {
      if (!spend()) {
        return {};
      }
      reach = std::min(reach, distance(p, points + 3 * static_cast<std::size_t>(other)) / 3.0);
    }
    for (std::size_t u = 0; u < count; ++u) {
      if (u != v) {
        if (!spend()) {
          return {};
        }
        reach = std::min(reach, 0.5 * distance(p, points + 3 * u));
      }
    }
    for (std::size_t e = 0; e + 1 < edges.size(); e += 2) {
      if (static_cast<std::size_t>(edges[e]) != v && static_cast<std::size_t>(edges[e + 1]) != v) {
        if (!spend()) {
          return {};
        }
        reach = std::min(reach, 0.5 * point_segment_distance(
                                          p, points + 3 * static_cast<std::size_t>(edges[e]),
                                          points + 3 * static_cast<std::size_t>(edges[e + 1])));
      }
    }
    for (std::size_t t = 0; t + 2 < triangles.size(); t += 3) {
      const std::size_t a = static_cast<std::size_t>(triangles[t]);
      const std::size_t b = static_cast<std::size_t>(triangles[t + 1]);
      const std::size_t c = static_cast<std::size_t>(triangles[t + 2]);
      if (a != v && b != v && c != v) {
        if (!spend()) {
          return {};
        }
        reach = std::min(reach, 0.5 * point_triangle_distance(p, points + 3 * a, points + 3 * b,
                                                              points + 3 * c));
      }
    }
    radii[v] = reach;
  }
  return radii;
}

double split_target(const double* a, const double* b, double apex_a, double apex_b,
                    double& tolerance) {
  const double length = distance(a, b);
  // Shell splits must hit their radius closely: an endpoint at relative
  // radius error e leaves concentric shells non-encroaching for input angles
  // above about sqrt(2 e); 2^-8 keeps that bound near five degrees.
  constexpr double kShellTolerance = 0x1p-8;
  const auto shell = [&](double radius, bool from_a) {
    double distance_from_apex = radius;
    if (length <= 4.0 * radius / 3.0) {
      // The power of two times the radius closest to half the length; it lies
      // in [length / 3, 2 length / 3] because that interval spans a factor 2.
      const double k = std::round(std::log2(length / (2.0 * radius)));
      distance_from_apex = std::clamp(radius * std::exp2(k), length / 3.0, 2.0 * length / 3.0);
    }
    const double t = distance_from_apex / length;
    tolerance = kShellTolerance * std::min(t, 1.0 - t);
    return from_a ? t : 1.0 - t;
  };
  if (apex_a > 0.0 && length > 4.0 * apex_a / 3.0) {
    return shell(apex_a, true);
  }
  if (apex_b > 0.0 && length > 4.0 * apex_b / 3.0) {
    return shell(apex_b, false);
  }
  if (apex_a > 0.0) {
    return shell(apex_a, true);
  }
  if (apex_b > 0.0) {
    return shell(apex_b, false);
  }
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
    if (t == previous || !(t > 0.0 && t < 1.0) || std::fabs(t - target) > tolerance) {
      continue;
    }
    previous = t;
    for (int axis = 0; axis < 3; ++axis) {
      point[axis] = a[axis] + t * direction[axis];
    }
    if (in_domain(point) && collinear3d(a, b, point) && strictly_between(a, b, point)) {
      return true;
    }
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
    for (int axis = 0; axis < 3; ++axis) {
      point[axis] = a[axis] + t * ab[axis] + t * ac[axis];
    }
    int side = 0;
    if (in_domain(point) && locate_on_triangle(point, triangle, side) == kFeatureInterior) {
      return true;
    }
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
  if (!source_parameters_valid(stratum, parameters)) {
    return false;
  }
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
  // The residual correction converges monotonically to within one unit;
  // the exact neighbour comparison then certifies the nearest value.
  double x = value.estimate();
  for (;;) {
    const double next = x + (value - Expansion(x)).estimate();
    if (next == x || closer(value, next, x) >= 0) {
      break;
    }
    x = next;
  }
  for (;;) {
    const double down = std::nextafter(x, -std::numeric_limits<double>::infinity());
    const double up = std::nextafter(x, std::numeric_limits<double>::infinity());
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
  if (magnitude.is_zero()) {
    return 0.0;
  }
  double bound = std::fabs(magnitude.estimate());
  while ((Expansion(bound) - magnitude).sign() < 0) {
    bound = std::nextafter(bound, std::numeric_limits<double>::infinity());
  }
  return bound;
}

double outward_distance(const double* position, const Expansion* point) {
  double components[3];
  double scale = 0.0;
  for (int k = 0; k < 3; ++k) {
    components[k] = outward_magnitude(Expansion(position[k]) - point[k]);
    scale = std::max(scale, components[k]);
  }
  if (scale == 0.0) {
    return 0.0;
  }
  // Scaled evaluation: at most 4.5 relative roundings below the exact norm
  // of the outward components; the 2^-49 inflation and the final upward step
  // dominate them and the rounding of the inflation itself.
  double sum = 0.0;
  for (double component : components) {
    const double ratio = component / scale;
    sum += ratio * ratio;
  }
  const double norm = scale * std::sqrt(sum) * (1.0 + 0x1p-49);
  return std::nextafter(norm, std::numeric_limits<double>::infinity());
}

bool on_source_entity(const double* const* corners, SourceStratum stratum,
                      const double* position) {
  switch (stratum) {
    case SourceStratum::kSegment: {
      if (!collinear3d(corners[0], corners[1], position)) {
        return false;
      }
      for (int k = 0; k < 3; ++k) {
        if (position[k] < std::min(corners[0][k], corners[1][k]) ||
            position[k] > std::max(corners[0][k], corners[1][k])) {
          return false;
        }
      }
      return true;
    }
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
  if (!source_point(corners, stratum, parameters, point)) {
    return -1.0;
  }
  return on_source_entity(corners, stratum, position) ? 0.0 : outward_distance(position, point);
}

bool source_carrier(const double* const* corners, SourceStratum stratum,
                    const double* parameters, double* position, double& deviation) {
  Expansion point[3];
  if (!source_point(corners, stratum, parameters, point)) {
    return false;
  }
  for (int k = 0; k < 3; ++k) {
    position[k] = nearest_double(point[k]);
  }
  if (!in_domain(position)) {
    return false;
  }
  deviation = on_source_entity(corners, stratum, position) ? 0.0
                                                           : outward_distance(position, point);
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
  for (int k = 1; k < 3; ++k) {
    if (std::fabs(n[k]) > std::fabs(n[axis])) axis = k;
  }
  const int x = (axis + 1) % 3;
  const int y = (axis + 2) % 3;
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

}  // namespace phx::mc
