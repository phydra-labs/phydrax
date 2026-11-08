// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// kZeroSetBand admission and the zero-set surface remesh C entry; contract in
// zero_set_admission.hpp and phydrax_meshcore.h.
#include "zero_set_admission.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <system_error>
#include <utility>
#include <vector>

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
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
  program_.reset();
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
  const int32_t status = program_.bind(instructions, instruction_count, indices, index_count,
                                       parameters, parameter_count, workspace_size);
  if (status != PHX_MC_OK) return status;
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
  return program_.execute(lower_.data(), upper_.data(), workspace_size);
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
  const int32_t status = program_.execute(lower_.data(), upper_.data(),
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
// point of it lies within the tolerance), or at binary64 resolution, which
// refuses. Binary64 resolution is relative to the triangle: a piece whose
// corners all lie within one slack (two coordinate ulps at the triangle's
// scale) of each other cannot be split into representable distinct pieces at
// that scale. An absolute test alone (a rounded midpoint equal to an endpoint)
// would run to subnormal spacing near a zero coordinate, about 1074 levels.
// Each level halves the piece's extent, so the depth is at most about 53, and
// the pieces are kept on an explicit stack in any case. Rounded midpoints leave
// every point of T within one coordinate ulp per level of the union of the
// pieces, so a piece at depth k adds k such slacks.
double ZeroSetBandAdmission::distance_bound(
    const Point& a, const Point& b, const Point& c,
    const std::array<VertexEvidence, 3>& evidence) const {
  double scale = 0;
  for (const Point* p : {&a, &b, &c})
    for (double coordinate : *p) scale = std::max(scale, std::fabs(coordinate));
  const double slack = next_up(2 * std::numeric_limits<double>::epsilon() * scale);
  return split_distance(a, b, c, evidence, slack);
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

// Depth-first over the pieces, in the order of the four sub-pieces; the first
// piece that is neither certified nor splittable refuses the whole triangle.
// An undecided root reports its own bound, a refused sub-piece +inf.
double ZeroSetBandAdmission::split_distance(const Point& a, const Point& b, const Point& c,
                                            const std::array<VertexEvidence, 3>& evidence,
                                            double slack) const {
  struct Piece {
    Point a, b, c;
    std::array<VertexEvidence, 3> evidence;
    int depth;
  };
  const auto middle = [](const Point& p, const Point& q) {
    return Point{0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]), 0.5 * (p[2] + q[2])};
  };
  std::vector<Piece> stack{{a, b, c, evidence, 0}};
  double worst = 0;
  while (!stack.empty()) {
    const Piece piece = stack.back();
    stack.pop_back();
    const double own = piece_distance(piece.a, piece.b, piece.c, piece.evidence);
    const double bound =
        std::isfinite(own) ? next_up(own + next_up(piece.depth * slack)) : kInfinity;
    if (bound <= tolerance_) {
      worst = std::max(worst, bound);
      continue;
    }
    if (piece_refuted(piece.a, piece.b, piece.c)) return piece.depth == 0 ? bound : kInfinity;
    const Point ab = middle(piece.a, piece.b), bc = middle(piece.b, piece.c),
                ca = middle(piece.c, piece.a);
    if (ab == piece.a || ab == piece.b || bc == piece.b || bc == piece.c || ca == piece.c ||
        ca == piece.a)
      return kInfinity;
    double extent = 0;
    for (int d = 0; d < 3; ++d) {
      extent = std::max({extent, std::fabs(piece.a[d] - piece.b[d]),
                         std::fabs(piece.b[d] - piece.c[d]), std::fabs(piece.c[d] - piece.a[d])});
    }
    if (extent <= slack) return kInfinity;
    const VertexEvidence eab = evaluate(ab), ebc = evaluate(bc), eca = evaluate(ca);
    const auto& e = piece.evidence;
    const int depth = piece.depth + 1;
    // Pushed in reverse so the first sub-piece is decided first.
    stack.push_back({bc, ca, ab, {ebc, eca, eab}, depth});
    stack.push_back({ca, bc, piece.c, {eca, ebc, e[2]}, depth});
    stack.push_back({ab, piece.b, bc, {eab, e[1], ebc}, depth});
    stack.push_back({piece.a, ab, ca, {e[0], eab, eca}, depth});
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
  if (transversal_sign(a, b, c) != orientation_) {
    ++refusals_.transversality;
    return false;
  }
  if (!(distance_bound(a, b, c, evidence) <= tolerance_)) {
    ++refusals_.distance;
    return false;
  }
  return true;
}

// The source normals near the triangle are orientation * grad f / |grad f| at
// zero-set points within the tolerance of it, so over bbox(T) widened by the
// tolerance. With the gradient enclosure's midpoint g and radius vector r
// (|r| < |g|), every enclosed direction lies within asin(|r| / |g|) of g. An
// admitted triangle's points lie within the tolerance of Z, so across its
// smallest altitude it tilts from the local tangent plane by up to
// asin(2 tolerance / altitude), which widens the cone. An undecided or
// zero-straddling enclosure constrains nothing (a crease enclosure holds
// both sides' gradients, so the crease is kept).
NormalCone ZeroSetBandAdmission::normal_cone(const Point& a, const Point& b, const Point& c,
                                             const std::array<VertexEvidence, 3>& evidence) const {
  NormalCone cone;
  const double tilt = normal_cone_floor(a, b, c, evidence);
  if (!(tilt < kOpenCone)) return cone;
  Point lower{}, upper{};
  bounding_box(a, b, c, lower, upper);
  for (int d = 0; d < 3; ++d) {
    lower[d] = next_down(lower[d] - tolerance_);
    upper[d] = next_up(upper[d] + tolerance_);
  }
  std::array<Interval, 3> gradient{};
  if (!gradient_over(gradient_, lower, upper, gradient)) return cone;
  Point middle{}, spread{};
  for (int d = 0; d < 3; ++d) {
    const Interval& g = gradient[static_cast<size_t>(d)];
    middle[d] = g.lower + 0.5 * (g.upper - g.lower);
    spread[d] = next_up(std::max(g.upper - middle[d], middle[d] - g.lower));
  }
  const double middle_norm = std::hypot(middle[0], middle[1], middle[2]);
  const double spread_norm = next_up(std::hypot(spread[0], spread[1], spread[2]));
  if (!(spread_norm < middle_norm) || !std::isfinite(middle_norm)) return cone;
  const double half = std::asin(spread_norm / middle_norm) + tilt;
  if (!(half < cone.half_angle)) return cone;
  for (int d = 0; d < 3; ++d) cone.axis[d] = orientation_ * middle[d] / middle_norm;
  cone.half_angle = half;
  return cone;
}

double ZeroSetBandAdmission::normal_cone_floor(const Point& a, const Point& b, const Point& c,
                                               const std::array<VertexEvidence, 3>&) const {
  Point u{}, w{};
  for (int d = 0; d < 3; ++d) {
    u[d] = b[d] - a[d];
    w[d] = c[d] - a[d];
  }
  const double twice_area = std::hypot(u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2],
                                       u[0] * w[1] - u[1] * w[0]);
  const double longest = std::max({std::hypot(u[0], u[1], u[2]), std::hypot(w[0], w[1], w[2]),
                                   std::hypot(c[0] - b[0], c[1] - b[1], c[2] - b[2])});
  if (!(twice_area > 0) || !(longest > 0)) return kOpenCone;
  const double tilt = 2 * tolerance_ * longest / twice_area;  // 2 tolerance / altitude
  return tilt < 1 ? std::asin(tilt) : kOpenCone;
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
// With distinct corners, strictly increasing valid face rows for every
// vertex plus exactly 3F memberships proves complete incidence, without a
// second CSR or any mutation of the caller's banks.
int32_t validate_compact_incidence(const phx_mc_zero_set_remesh_request& request,
                                   const int32_t* offsets, int64_t offset_count,
                                   const int32_t* rows, int64_t row_count) {
  if (offsets == nullptr && rows == nullptr && offset_count == 0 && row_count == 0)
    return PHX_MC_OK;
  if (offsets == nullptr || rows == nullptr || request.vertex_count < 3 ||
      request.triangle_count < 1 || request.vertices == nullptr || request.triangles == nullptr ||
      !phx::mc::addressable(request.vertex_count, 3, sizeof(double)) ||
      !phx::mc::addressable(request.triangle_count, 3, sizeof(int32_t)) ||
      request.triangle_count > std::numeric_limits<int32_t>::max() / 3 ||
      offset_count != request.vertex_count + 1 || row_count != 3 * request.triangle_count ||
      !phx::mc::addressable(offset_count, 1, sizeof(int32_t)) ||
      !phx::mc::addressable(row_count, 1, sizeof(int32_t)))
    return PHX_MC_INVALID_ARGUMENT;
  const auto* faces = reinterpret_cast<const int32_t*>(request.triangles);
  for (int64_t face = 0; face < request.triangle_count; ++face) {
    const size_t at = 3 * static_cast<size_t>(face);
    const int32_t a = faces[at], b = faces[at + 1], c = faces[at + 2];
    if (a < 0 || b < 0 || c < 0 || a >= request.vertex_count ||
        b >= request.vertex_count || c >= request.vertex_count ||
        a == b || b == c || c == a) return PHX_MC_INVALID_INPUT;
  }
  if (offsets[0] != 0 || offsets[static_cast<size_t>(request.vertex_count)] != row_count)
    return PHX_MC_INVALID_INPUT;
  for (int64_t vertex = 0; vertex < request.vertex_count; ++vertex) {
    const int32_t begin = offsets[static_cast<size_t>(vertex)];
    const int32_t end = offsets[static_cast<size_t>(vertex) + 1];
    if (begin < 0 || end < begin || end > row_count) return PHX_MC_INVALID_INPUT;
    int32_t previous = -1;
    for (int32_t at = begin; at < end; ++at) {
      const int32_t face = rows[static_cast<size_t>(at)];
      if (face <= previous || face < 0 || face >= request.triangle_count)
        return PHX_MC_INVALID_INPUT;
      const size_t index = 3 * static_cast<size_t>(face);
      if (faces[index] != vertex && faces[index + 1] != vertex && faces[index + 2] != vertex)
        return PHX_MC_INVALID_INPUT;
      previous = face;
    }
  }
  return PHX_MC_OK;
}


int32_t remesh_zero_set(const phx_mc_zero_set_remesh_request& request,
                        ZeroSetRemeshResult& result, int64_t* counters, double* distance,
                        bool indices_i32, const int32_t* offsets, const int32_t* face_rows) {
  if (request.vertex_count < 3 || request.triangle_count < 1 ||
      !phx::mc::addressable(request.vertex_count, 3, sizeof(double)) ||
      !phx::mc::addressable(request.triangle_count, 3, indices_i32 ? sizeof(int32_t) : sizeof(int64_t)) ||
      request.vertices == nullptr || request.triangles == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  if (!(request.tolerance > 0) || !std::isfinite(request.tolerance) ||
      !(request.carrier_distance >= 0) || !std::isfinite(request.carrier_distance) ||
      (request.admission_limited_coarsening != 0 && request.admission_limited_coarsening != 1))
    return PHX_MC_INVALID_ARGUMENT;
  const RemeshTargets targets{request.edge_length, request.minimum_angle_degrees,
                              request.admission_limited_coarsening == 1};
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
  const auto input_index = [&](size_t offset) -> int64_t {
    if (indices_i32) return reinterpret_cast<const int32_t*>(request.triangles)[offset];
    return request.triangles[offset];
  };
  std::vector<Point> points;
  if (!indices_i32) points.resize(static_cast<size_t>(request.vertex_count));
  for (size_t v = 0; v < static_cast<size_t>(request.vertex_count); ++v)
    for (int d = 0; d < 3; ++d) {
      const double coordinate = request.vertices[3 * v + static_cast<size_t>(d)];
      if (!std::isfinite(coordinate)) return PHX_MC_NONFINITE_INPUT;
      if (!indices_i32) points[v][static_cast<size_t>(d)] = coordinate;
    }
  const auto source_point = [&](size_t v) {
    return Point{request.vertices[3 * v], request.vertices[3 * v + 1],
                 request.vertices[3 * v + 2]};
  };
  std::vector<Triangle> triangles;
  if (!indices_i32) triangles.resize(static_cast<size_t>(request.triangle_count));
  for (size_t t = 0; t < static_cast<size_t>(request.triangle_count); ++t)
    for (int k = 0; k < 3; ++k) {
      const int64_t v = input_index(3 * t + static_cast<size_t>(k));
      if (v < 0 || v >= request.vertex_count) return PHX_MC_INVALID_INPUT;
      if (!indices_i32) triangles[t][static_cast<size_t>(k)] = v;
    }
  // The carrier declares its orientation relative to grad f: every input
  // triangle whose transversal sign is certified must agree on it.
  const ZeroSetBandAdmission probe(value, gradient, request.tolerance, 1);
  int orientation = 0;
  for (size_t row = 0; row < static_cast<size_t>(request.triangle_count); ++row) {
    const Triangle t{input_index(3 * row), input_index(3 * row + 1), input_index(3 * row + 2)};
    const Point a = source_point(static_cast<size_t>(t[0]));
    const Point b = source_point(static_cast<size_t>(t[1]));
    const Point c = source_point(static_cast<size_t>(t[2]));
    const int sign = probe.transversal_sign(a, b, c);
    if (sign == 0) continue;
    if (orientation != 0 && sign != orientation) return PHX_MC_INVALID_INPUT;
    orientation = sign;
  }
  if (orientation == 0) return PHX_MC_UNRESOLVED;
  const ZeroSetBandAdmission admission(value, gradient, request.tolerance, orientation);
  RemeshCounters remesh{};
  if (indices_i32)
    status = remesh_surface_borrowed_i32(
        points, request.vertices, static_cast<size_t>(request.vertex_count),
        reinterpret_cast<const int32_t*>(request.triangles),
        static_cast<size_t>(request.triangle_count), offsets, face_rows, triangles, admission,
        targets, remesh);
  else
    status = remesh_surface(points, triangles, admission, targets, remesh);
  if (status != PHX_MC_OK) return status;
  const ZeroSetBandAdmission::Refusals refusals = admission.refusals();
  // Every surviving triangle is either admitted (its certified distance) or an
  // untouched input triangle (the carrier's declared distance).
  double maximum = 0;
  // The input remains borrowed for the call.  Index its rows only if a
  // surviving face needs the untouched-carrier fallback; storing a tree node
  // and nine copied coordinates for every input face is unnecessary.
  std::vector<int64_t> input_order;
  const auto input_corners = [&](int64_t row) {
    std::array<Point, 3> corners{};
    for (int k = 0; k < 3; ++k) {
      const int64_t v = input_index(3 * static_cast<size_t>(row) + k);
      for (int d = 0; d < 3; ++d)
        corners[k][d] = request.vertices[3 * static_cast<size_t>(v) + d];
    }
    std::sort(corners.begin(), corners.end());
    return corners;
  };
  for (const Triangle& t : triangles) {
    const Point& a = points[static_cast<size_t>(t[0])];
    const Point& b = points[static_cast<size_t>(t[1])];
    const Point& c = points[static_cast<size_t>(t[2])];
    const std::array<VertexEvidence, 3> evidence{admission.evaluate(a), admission.evaluate(b),
                                                 admission.evaluate(c)};
    double bound = kInfinity;
    if (admission.admits(a, b, c, evidence)) bound = admission.distance_bound(a, b, c, evidence);
    if (!(bound <= request.tolerance)) {
      if (input_order.empty()) {
        input_order.resize(static_cast<size_t>(request.triangle_count));
        std::iota(input_order.begin(), input_order.end(), int64_t{0});
        std::sort(input_order.begin(), input_order.end(), [&](int64_t a, int64_t b) {
          return input_corners(a) < input_corners(b);
        });
      }
      const auto key = sorted_corners(a, b, c);
      const auto found = std::lower_bound(
          input_order.begin(), input_order.end(), key,
          [&](int64_t row, const SortedCorners& value) { return input_corners(row) < value; });
      if (found == input_order.end() || input_corners(*found) != key) return PHX_MC_INTERNAL_ERROR;
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
  counters[6] = refusals.transversality;
  counters[7] = refusals.distance;
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
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    auto owned = std::make_unique<phx::mc::remesh::ZeroSetRemeshResult>();
    const int32_t status =
        phx::mc::remesh::remesh_zero_set(*request, *owned, request->counters, request->distance,
                                         false, nullptr, nullptr);
    if (status != PHX_MC_OK) return status;
    *request->result = owned.release();
    return PHX_MC_OK;
  });
}

PHX_MC_API int32_t phx_mc_zero_set_remesh_i32(
    const phx_mc_zero_set_remesh_request* request, const int32_t* vertex_face_offsets,
    int64_t vertex_face_offset_count, const int32_t* vertex_faces, int64_t vertex_face_count) {
  if (request == nullptr || request->result == nullptr || request->counters == nullptr ||
      request->distance == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  *request->result = nullptr;
  const int32_t incidence_status = phx::mc::remesh::validate_compact_incidence(
      *request, vertex_face_offsets, vertex_face_offset_count, vertex_faces, vertex_face_count);
  if (incidence_status != PHX_MC_OK) return incidence_status;
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    try {
      auto owned = std::make_unique<phx::mc::remesh::ZeroSetRemeshResult>();
      const int32_t status = phx::mc::remesh::remesh_zero_set(
          *request, *owned, request->counters, request->distance, true,
          vertex_face_offsets, vertex_faces);
      if (status != PHX_MC_OK) return status;
      *request->result = owned.release();
      return PHX_MC_OK;
    } catch (const std::system_error&) {
      return PHX_MC_IO_FAILURE;
    }
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
