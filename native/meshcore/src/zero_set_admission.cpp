// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// kZeroSetBand admission and the zero-set surface remesh C entry; contract in
// zero_set_admission.hpp and phydrax_meshcore.h.
#include "zero_set_admission.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <tuple>
#include <utility>

#include "capi_guard.hpp"
#include "phydrax_meshcore.h"
#include "rounding.hpp"

namespace phx::mc::remesh {
namespace {
using phx::mc::geometry::next_down;
using phx::mc::geometry::next_up;

constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr int kNewtonSteps = 16;

bool finite(const Interval& value) {
  return std::isfinite(value.lower) && std::isfinite(value.upper) && value.lower <= value.upper;
}
Interval point(double value) { return {value, value}; }
Interval add(const Interval& a, const Interval& b) {
  return {next_down(a.lower + b.lower), next_up(a.upper + b.upper)};
}
Interval subtract(const Interval& a, const Interval& b) {
  return {next_down(a.lower - b.upper), next_up(a.upper - b.lower)};
}
Interval multiply(const Interval& a, const Interval& b) {
  const double p[4] = {a.lower * b.lower, a.lower * b.upper, a.upper * b.lower,
                       a.upper * b.upper};
  return {next_down(std::min({p[0], p[1], p[2], p[3]})),
          next_up(std::max({p[0], p[1], p[2], p[3]}))};
}
// Smallest magnitude in the interval (0 when it contains zero).
double mignitude(const Interval& value) {
  if (value.lower > 0) return value.lower;
  if (value.upper < 0) return -value.upper;
  return 0;
}
double magnitude(const Interval& value) {
  return std::max(std::fabs(value.lower), std::fabs(value.upper));
}
// Certified lower bound of |g| over the gradient enclosure.
double norm_lower(const std::array<Interval, 3>& gradient) {
  double sum = 0;
  for (const Interval& component : gradient) {
    const double m = mignitude(component);
    sum = next_down(sum + next_down(m * m));
  }
  return sum > 0 ? next_down(std::sqrt(sum)) : 0.0;
}
void bounding_box(const Point& a, const Point& b, const Point& c, Point& lower, Point& upper) {
  for (int d = 0; d < 3; ++d) {
    lower[d] = std::min({a[d], b[d], c[d]});
    upper[d] = std::max({a[d], b[d], c[d]});
  }
}
bool gradient_over(const IntervalBoxProgram& program, const Point& lower, const Point& upper,
                   std::array<Interval, 3>& gradient) {
  if (!program.evaluate(lower, upper, gradient.data())) return false;
  return finite(gradient[0]) && finite(gradient[1]) && finite(gradient[2]);
}
}  // namespace

int32_t IntervalBoxProgram::bind(const int64_t* instructions, int64_t instruction_count,
                                 const int64_t* indices, int64_t index_count,
                                 const double* parameters, int64_t parameter_count,
                                 const double* workspace_lower, const double* workspace_upper,
                                 int64_t workspace_size, const int64_t* outputs,
                                 int64_t output_count, const int64_t* guards,
                                 int64_t guard_count) {
  if (instruction_count < 0 || index_count < 0 || parameter_count < 0 || workspace_size < 3 ||
      output_count < 1 || guard_count < 0 || (instruction_count > 0 && instructions == nullptr) ||
      (index_count > 0 && indices == nullptr) || (parameter_count > 0 && parameters == nullptr) ||
      (guard_count > 0 && guards == nullptr) || workspace_lower == nullptr ||
      workspace_upper == nullptr || outputs == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  if (!phx::mc::addressable(instruction_count, 8, sizeof(int64_t)) ||
      !phx::mc::addressable(workspace_size, 1, sizeof(double)))
    return PHX_MC_INVALID_ARGUMENT;
  for (int64_t k = 0; k < output_count; ++k)
    if (outputs[k] < 3 || outputs[k] >= workspace_size) return PHX_MC_INVALID_ARGUMENT;
  for (int64_t k = 0; k < guard_count; ++k)
    if (guards[k] < 0 || guards[k] >= workspace_size) return PHX_MC_INVALID_ARGUMENT;
  instructions_.assign(instructions, instructions + 8 * instruction_count);
  indices_.assign(indices, indices + index_count);
  parameters_.assign(parameters, parameters + parameter_count);
  template_lower_.assign(workspace_lower, workspace_lower + workspace_size);
  template_upper_.assign(workspace_upper, workspace_upper + workspace_size);
  outputs_.assign(outputs, outputs + output_count);
  guards_.assign(guards, guards + guard_count);
  lower_.resize(static_cast<size_t>(workspace_size));
  upper_.resize(static_cast<size_t>(workspace_size));
  // One dry execution over the origin validates the whole program once.
  lower_ = template_lower_;
  upper_ = template_upper_;
  std::fill(lower_.begin(), lower_.begin() + 3, 0.0);
  std::fill(upper_.begin(), upper_.begin() + 3, 0.0);
  return phx_mc_interval_program_execute(
      instructions_.data(), instruction_count, indices_.data(), index_count,
      parameters_.data(), parameter_count, lower_.data(), upper_.data(), workspace_size);
}

bool IntervalBoxProgram::evaluate(const Point& lower, const Point& upper,
                                  Interval* result) const {
  for (int d = 0; d < 3; ++d)
    if (!(lower[d] <= upper[d]) || !std::isfinite(lower[d]) || !std::isfinite(upper[d]))
      return false;
  std::copy(template_lower_.begin(), template_lower_.end(), lower_.begin());
  std::copy(template_upper_.begin(), template_upper_.end(), upper_.begin());
  for (int d = 0; d < 3; ++d) {
    lower_[static_cast<size_t>(d)] = lower[d];
    upper_[static_cast<size_t>(d)] = upper[d];
  }
  const int32_t status = phx_mc_interval_program_execute(
      instructions_.data(), static_cast<int64_t>(instructions_.size() / 8), indices_.data(),
      static_cast<int64_t>(indices_.size()), parameters_.data(),
      static_cast<int64_t>(parameters_.size()), lower_.data(), upper_.data(),
      static_cast<int64_t>(lower_.size()));
  if (status != PHX_MC_OK) return false;
  // A real-power base outside its nonnegative domain leaves f undefined there.
  for (int64_t guard : guards_)
    if (lower_[static_cast<size_t>(guard)] < 0.0) return false;
  for (size_t k = 0; k < outputs_.size(); ++k) {
    const size_t at = static_cast<size_t>(outputs_[k]);
    result[k] = {lower_[at], upper_[at]};
    if (std::isnan(result[k].lower) || std::isnan(result[k].upper)) return false;
  }
  return true;
}

VertexEvidence ZeroSetBandAdmission::evaluate(const Point& position) const {
  Interval value{};
  if (!value_.evaluate(position, position, &value) || !finite(value))
    return {-kInfinity, kInfinity};
  return {value.lower, value.upper};
}

double ZeroSetBandAdmission::piece_distance(
    const Point& a, const Point& b, const Point& c,
    const std::array<VertexEvidence, 3>& evidence) const {
  Point lower{}, upper{};
  bounding_box(a, b, c, lower, upper);
  std::array<Interval, 3> gradient{};
  if (!gradient_over(gradient_, lower, upper, gradient)) return kInfinity;
  Interval field{};
  if (!value_.evaluate(lower, upper, &field) || !finite(field)) return kInfinity;
  // Mean-value forms anchored at each corner p with edges e1, e2 to the other
  // corners: x = p + s e1 + t e2 (s, t >= 0, s + t <= 1) and f(x) - f(p) =
  // s (g . e1) + t (g . e2) for some g = grad f(xi), xi on [p, x] in bbox(T).
  // Over the simplex this lies in the hull of 0, G . e1 and G . e2; tangent
  // edges make it second order in the triangle size.
  const std::array<const Point*, 3> corners{&a, &b, &c};
  for (size_t k = 0; k < 3; ++k) {
    Interval form{evidence[k].lower, evidence[k].upper};
    if (!finite(form)) continue;
    const Point& p = *corners[k];
    const Point& q = *corners[(k + 1) % 3];
    const Point& r = *corners[(k + 2) % 3];
    Interval along_q{0, 0}, along_r{0, 0};
    for (int d = 0; d < 3; ++d) {
      const Interval& g = gradient[static_cast<size_t>(d)];
      along_q = add(along_q, multiply(g, subtract(point(q[d]), point(p[d]))));
      along_r = add(along_r, multiply(g, subtract(point(r[d]), point(p[d]))));
    }
    form = add(form, {std::min({0.0, along_q.lower, along_r.lower}),
                      std::max({0.0, along_q.upper, along_r.upper})});
    if (!finite(form)) continue;
    field = {std::max(field.lower, form.lower), std::min(field.upper, form.upper)};
    if (field.lower > field.upper) return kInfinity;  // inconsistent enclosures
  }
  const double residual = magnitude(field);
  if (residual == 0) return 0;
  const double bbox_norm = norm_lower(gradient);
  if (!(bbox_norm > 0)) return kInfinity;
  const double estimate = next_up(residual / bbox_norm);
  double radii[2] = {std::min(tolerance_, next_up(2 * estimate)), tolerance_};
  for (const double radius : radii) {
    if (!(radius > 0) || !std::isfinite(radius)) continue;
    Point wide_lower{}, wide_upper{};
    for (int d = 0; d < 3; ++d) {
      wide_lower[d] = next_down(lower[d] - radius);
      wide_upper[d] = next_up(upper[d] + radius);
    }
    std::array<Interval, 3> wide{};
    if (!gradient_over(gradient_, wide_lower, wide_upper, wide)) continue;
    const double norm = norm_lower(wide);
    if (!(norm > 0)) continue;
    const double bound = next_up(residual / norm);
    if (bound <= radius) return bound;
  }
  return kInfinity;
}

// Pieces whose bound misses the tolerance split into four at edge midpoints
// until each piece is decided: certified within the tolerance, refuted (no
// point of it lies within the tolerance), or at binary64 resolution (a rounded
// midpoint coincides with an endpoint), which refuses. Rounded midpoints leave
// every point of T within one coordinate ulp per level of the union of the
// pieces, so a piece at depth k adds k such slacks.
double ZeroSetBandAdmission::distance_bound(
    const Point& a, const Point& b, const Point& c,
    const std::array<VertexEvidence, 3>& evidence) const {
  double scale = 0;
  for (const Point* p : {&a, &b, &c})
    for (double coordinate : *p) scale = std::max(scale, std::fabs(coordinate));
  const double slack = next_up(2 * std::numeric_limits<double>::epsilon() * scale);
  return split_distance(a, b, c, evidence, 0, slack);
}

// No point of the piece lies within the tolerance of Z: otherwise a zero of f
// lies in the tolerance-widened box B within the tolerance of that point, so
// |f| <= max |grad f| over B times the tolerance there.
bool ZeroSetBandAdmission::piece_refuted(const Point& a, const Point& b,
                                         const Point& c) const {
  Point lower{}, upper{};
  bounding_box(a, b, c, lower, upper);
  Interval field{};
  if (!value_.evaluate(lower, upper, &field) || !finite(field)) return false;
  Point wide_lower{}, wide_upper{};
  for (int d = 0; d < 3; ++d) {
    wide_lower[d] = next_down(lower[d] - tolerance_);
    wide_upper[d] = next_up(upper[d] + tolerance_);
  }
  std::array<Interval, 3> gradient{};
  if (!gradient_over(gradient_, wide_lower, wide_upper, gradient)) return false;
  double sum = 0;
  for (const Interval& component : gradient) {
    const double m = magnitude(component);
    sum = next_up(sum + next_up(m * m));
  }
  const double lipschitz = next_up(std::sqrt(sum));
  return mignitude(field) > next_up(lipschitz * tolerance_);
}

double ZeroSetBandAdmission::split_distance(const Point& a, const Point& b, const Point& c,
                                            const std::array<VertexEvidence, 3>& evidence,
                                            int depth, double slack) const {
  const double piece = piece_distance(a, b, c, evidence);
  const double bound =
      std::isfinite(piece) ? next_up(piece + next_up(depth * slack)) : kInfinity;
  if (bound <= tolerance_ || piece_refuted(a, b, c)) return bound;
  const auto middle = [](const Point& p, const Point& q) {
    return Point{0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]), 0.5 * (p[2] + q[2])};
  };
  const Point ab = middle(a, b), bc = middle(b, c), ca = middle(c, a);
  if (ab == a || ab == b || bc == b || bc == c || ca == c || ca == a) return kInfinity;
  const VertexEvidence eab = evaluate(ab), ebc = evaluate(bc), eca = evaluate(ca);
  double worst = 0;
  for (const auto& [p, q, r, ep, eq, er] :
       {std::tuple{a, ab, ca, evidence[0], eab, eca}, std::tuple{ab, b, bc, eab, evidence[1], ebc},
        std::tuple{ca, bc, c, eca, ebc, evidence[2]}, std::tuple{bc, ca, ab, ebc, eca, eab}}) {
    worst = std::max(worst, split_distance(p, q, r, {ep, eq, er}, depth + 1, slack));
    if (!(worst <= tolerance_)) return kInfinity;
  }
  return worst;
}

int ZeroSetBandAdmission::transversal_sign(const Point& a, const Point& b,
                                           const Point& c) const {
  Point lower{}, upper{};
  bounding_box(a, b, c, lower, upper);
  std::array<Interval, 3> gradient{};
  if (!gradient_over(gradient_, lower, upper, gradient)) return 0;
  std::array<Interval, 3> first{}, second{};
  for (int d = 0; d < 3; ++d) {
    first[static_cast<size_t>(d)] = subtract(point(b[d]), point(a[d]));
    second[static_cast<size_t>(d)] = subtract(point(c[d]), point(a[d]));
  }
  Interval product{0, 0};
  for (int d = 0; d < 3; ++d) {
    const size_t i = static_cast<size_t>((d + 1) % 3), j = static_cast<size_t>((d + 2) % 3);
    const Interval normal =
        subtract(multiply(first[i], second[j]), multiply(first[j], second[i]));
    product = add(product, multiply(normal, gradient[static_cast<size_t>(d)]));
  }
  if (!finite(product)) return 0;
  if (product.lower > 0) return 1;
  if (product.upper < 0) return -1;
  return 0;
}

bool ZeroSetBandAdmission::admits(const Point& a, const Point& b, const Point& c,
                                  const std::array<VertexEvidence, 3>& evidence) const {
  return transversal_sign(a, b, c) == orientation_ &&
         distance_bound(a, b, c, evidence) <= tolerance_;
}

Point ZeroSetBandAdmission::project(const Point& position) const {
  std::array<Interval, 3> gradient{};
  Interval value{};
  if (!gradient_over(gradient_, position, position, gradient) ||
      !value_.evaluate(position, position, &value) || !finite(value))
    return position;
  Point direction{};
  double norm = 0;
  for (int d = 0; d < 3; ++d) {
    direction[d] = 0.5 * (gradient[static_cast<size_t>(d)].lower +
                          gradient[static_cast<size_t>(d)].upper);
    norm += direction[d] * direction[d];
  }
  norm = std::sqrt(norm);
  if (!(norm > 0) || !std::isfinite(norm)) return position;
  for (double& component : direction) component /= norm;
  const double start = 0.5 * (value.lower + value.upper);
  if (start == 0) return position;
  // phi(t) = f(position - t u); its root lies near t = f / |grad f|.
  const double reach = 2 * std::fabs(start) / norm;
  Interval step{-reach, reach};
  for (int iteration = 0; iteration < kNewtonSteps; ++iteration) {
    const double middle = 0.5 * (step.lower + step.upper);
    Point at{}, lower{}, upper{};
    for (int d = 0; d < 3; ++d) {
      at[d] = position[d] - middle * direction[d];
      const double first = position[d] - step.lower * direction[d];
      const double second = position[d] - step.upper * direction[d];
      lower[d] = next_down(std::min(first, second));
      upper[d] = next_up(std::max(first, second));
    }
    Interval at_value{};
    std::array<Interval, 3> slope_gradient{};
    if (!value_.evaluate(at, at, &at_value) || !finite(at_value) ||
        !gradient_over(gradient_, lower, upper, slope_gradient))
      return position;
    Interval slope{0, 0};
    for (int d = 0; d < 3; ++d)
      slope = subtract(slope, multiply(slope_gradient[static_cast<size_t>(d)],
                                       point(direction[d])));
    if (!(slope.lower > 0) && !(slope.upper < 0)) break;
    const double q[4] = {at_value.lower / slope.lower, at_value.lower / slope.upper,
                         at_value.upper / slope.lower, at_value.upper / slope.upper};
    const Interval newton{next_down(middle - next_up(std::max({q[0], q[1], q[2], q[3]}))),
                          next_up(middle - next_down(std::min({q[0], q[1], q[2], q[3]})))};
    const Interval narrowed{std::max(step.lower, newton.lower),
                            std::min(step.upper, newton.upper)};
    if (narrowed.lower > narrowed.upper) return position;  // no root on this segment
    const bool settled = narrowed.lower == step.lower && narrowed.upper == step.upper;
    step = narrowed;
    if (settled) break;
  }
  const double t = 0.5 * (step.lower + step.upper);
  Point projected{};
  for (int d = 0; d < 3; ++d) projected[d] = position[d] - t * direction[d];
  return projected;
}

namespace {
struct ZeroSetRemeshResult {
  std::vector<Point> points;
  std::vector<Triangle> triangles;
};

using SortedCorners = std::array<Point, 3>;
SortedCorners sorted_corners(const Point& a, const Point& b, const Point& c) {
  SortedCorners corners{a, b, c};
  std::sort(corners.begin(), corners.end());
  return corners;
}

int32_t remesh_zero_set(const phx_mc_zero_set_remesh_request& request,
                        ZeroSetRemeshResult& result, int64_t* counters, double* distance) {
  if (request.vertex_count < 3 || request.triangle_count < 1 ||
      !phx::mc::addressable(request.vertex_count, 3, sizeof(double)) ||
      !phx::mc::addressable(request.triangle_count, 3, sizeof(int64_t)) ||
      request.vertices == nullptr || request.triangles == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  if (!(request.tolerance > 0) || !std::isfinite(request.tolerance) ||
      !(request.carrier_distance >= 0) || !std::isfinite(request.carrier_distance))
    return PHX_MC_INVALID_ARGUMENT;
  IntervalBoxProgram value, gradient;
  const auto bind = [](IntervalBoxProgram& program, const phx_mc_interval_box_program& source,
                       int64_t outputs) -> int32_t {
    if (source.output_count != outputs) return PHX_MC_INVALID_ARGUMENT;
    return program.bind(source.instructions, source.instruction_count, source.indices,
                        source.index_count, source.parameters, source.parameter_count,
                        source.workspace_lower, source.workspace_upper, source.workspace_size,
                        source.outputs, source.output_count, source.guards,
                        source.guard_count);
  };
  int32_t status = bind(value, request.value, 1);
  if (status != PHX_MC_OK) return status;
  status = bind(gradient, request.gradient, 3);
  if (status != PHX_MC_OK) return status;
  std::vector<Point> points(static_cast<size_t>(request.vertex_count));
  for (size_t v = 0; v < points.size(); ++v)
    for (int d = 0; d < 3; ++d) {
      points[v][static_cast<size_t>(d)] = request.vertices[3 * v + static_cast<size_t>(d)];
      if (!std::isfinite(points[v][static_cast<size_t>(d)])) return PHX_MC_NONFINITE_INPUT;
    }
  std::vector<Triangle> triangles(static_cast<size_t>(request.triangle_count));
  for (size_t t = 0; t < triangles.size(); ++t)
    for (int k = 0; k < 3; ++k) {
      const int64_t v = request.triangles[3 * t + static_cast<size_t>(k)];
      if (v < 0 || v >= request.vertex_count) return PHX_MC_INVALID_INPUT;
      triangles[t][static_cast<size_t>(k)] = v;
    }
  // The carrier declares its orientation relative to grad f: every input
  // triangle whose transversal sign is certified must agree on it.
  const ZeroSetBandAdmission probe(value, gradient, request.tolerance, 1);
  int orientation = 0;
  std::map<SortedCorners, int> input_triangles;
  for (const Triangle& t : triangles) {
    const Point& a = points[static_cast<size_t>(t[0])];
    const Point& b = points[static_cast<size_t>(t[1])];
    const Point& c = points[static_cast<size_t>(t[2])];
    input_triangles.emplace(sorted_corners(a, b, c), 0);
    const int sign = probe.transversal_sign(a, b, c);
    if (sign == 0) continue;
    if (orientation != 0 && sign != orientation) return PHX_MC_INVALID_INPUT;
    orientation = sign;
  }
  if (orientation == 0) return PHX_MC_UNRESOLVED;
  const ZeroSetBandAdmission admission(value, gradient, request.tolerance, orientation);
  RemeshCounters remesh{};
  status = remesh_surface(points, triangles, admission,
                          {request.edge_length, request.minimum_angle_degrees}, remesh);
  if (status != PHX_MC_OK) return status;
  // Every surviving triangle is either admitted (its certified distance) or an
  // untouched input triangle (the carrier's declared distance).
  double maximum = 0;
  for (const Triangle& t : triangles) {
    const Point& a = points[static_cast<size_t>(t[0])];
    const Point& b = points[static_cast<size_t>(t[1])];
    const Point& c = points[static_cast<size_t>(t[2])];
    const std::array<VertexEvidence, 3> evidence{admission.evaluate(a), admission.evaluate(b),
                                                 admission.evaluate(c)};
    double bound = kInfinity;
    if (admission.admits(a, b, c, evidence)) bound = admission.distance_bound(a, b, c, evidence);
    if (!(bound <= request.tolerance)) {
      if (input_triangles.count(sorted_corners(a, b, c)) == 0) return PHX_MC_INTERNAL_ERROR;
      bound = request.carrier_distance;
    }
    maximum = std::max(maximum, bound);
  }
  counters[0] = remesh.splits;
  counters[1] = remesh.collapses;
  counters[2] = remesh.flips;
  counters[3] = remesh.relocations;
  counters[4] = remesh.unmet;
  counters[5] = orientation;
  *distance = maximum;
  result.points = std::move(points);
  result.triangles = std::move(triangles);
  return PHX_MC_OK;
}
}  // namespace
}  // namespace phx::mc::remesh

extern "C" {

PHX_MC_API int32_t phx_mc_zero_set_remesh(const phx_mc_zero_set_remesh_request* request) {
  if (request == nullptr || request->result == nullptr || request->counters == nullptr ||
      request->distance == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  *request->result = nullptr;
  return phx::mc::guarded([&]() -> int32_t {
    auto owned = std::make_unique<phx::mc::remesh::ZeroSetRemeshResult>();
    const int32_t status =
        phx::mc::remesh::remesh_zero_set(*request, *owned, request->counters, request->distance);
    if (status != PHX_MC_OK) return status;
    *request->result = owned.release();
    return PHX_MC_OK;
  });
}

PHX_MC_API int32_t phx_mc_zero_set_remesh_sizes(const void* result, int64_t* sizes) {
  if (result == nullptr || sizes == nullptr) return PHX_MC_INVALID_ARGUMENT;
  const auto& data = *static_cast<const phx::mc::remesh::ZeroSetRemeshResult*>(result);
  sizes[0] = static_cast<int64_t>(data.points.size());
  sizes[1] = static_cast<int64_t>(data.triangles.size());
  return PHX_MC_OK;
}

PHX_MC_API int32_t phx_mc_zero_set_remesh_arrays(void* result, int64_t vertex_capacity,
                                                 int64_t triangle_capacity, double* vertices,
                                                 int64_t* triangles) {
  if (result == nullptr || vertices == nullptr || triangles == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  const auto& data = *static_cast<const phx::mc::remesh::ZeroSetRemeshResult*>(result);
  if (vertex_capacity < static_cast<int64_t>(data.points.size()) ||
      triangle_capacity < static_cast<int64_t>(data.triangles.size()))
    return PHX_MC_CAPACITY_EXCEEDED;
  for (size_t v = 0; v < data.points.size(); ++v)
    for (size_t d = 0; d < 3; ++d) vertices[3 * v + d] = data.points[v][d];
  for (size_t t = 0; t < data.triangles.size(); ++t)
    for (size_t k = 0; k < 3; ++k) triangles[3 * t + k] = data.triangles[t][k];
  return PHX_MC_OK;
}

PHX_MC_API void phx_mc_zero_set_remesh_free(void* result) {
  delete static_cast<phx::mc::remesh::ZeroSetRemeshResult*>(result);
}

}  // extern "C"
