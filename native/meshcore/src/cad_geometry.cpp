// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "cad_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string_view>

#include "capi_guard.hpp"
#include "directed_interval.hpp"
#include "floating_point_mode.hpp"

namespace phx::mc::cad {

#include "cad_geometry_programs.inc"

namespace {

// Highest spline degree: the basis-derivative tables live on the stack.
constexpr int kMaxDegree = 31;
// Reals per intersection chart: start, end, lower, upper (4 each), P (3, 3).
constexpr int64_t kChartReals = 25;
// The source's chart iteration: fixed preconditioned corrections from the
// chart's interpolation. It converges linearly at the preconditioner's
// contraction rate, so evaluation runs kChartSteps of it, then Newton steps
// with the residual Jacobian (the same root, quadratically) until a
// correction is at rounding level (kSettled ulp of the coordinate). A
// Jacobian that cannot be solved or a root that does not settle within
// kNewtonIterations falls back to the source's kChartIterations.
constexpr int kChartIterations = 64;
constexpr int kChartSteps = 2;
constexpr int kNewtonIterations = 8;
constexpr double kSettled = 16.0;

ProgramView view(const ProgramData& data, const std::vector<double>& seeds,
                 int64_t input_dimension) noexcept {
  return {data.instructions, data.instruction_count, data.indices,    seeds.data(),
          data.workspace_size, input_dimension,      data.outputs,    data.output_count};
}

bool templates_valid() {
  for (int32_t id = 0; id < kTemplateCount; ++id) {
    const TemplateDefinition& definition = kTemplates[id];
    for (const ProgramData* data :
         {&definition.value, &definition.tangent, &definition.hessian}) {
      if (validate_point_program(data->instructions, data->instruction_count, data->indices,
                                 data->index_count, data->parameters, data->parameter_count,
                                 data->workspace_size, definition.input_dimension,
                                 data->outputs, data->output_count) != PHX_MC_OK)
        return false;
      for (int32_t k = 0; k < definition.coefficient_count; ++k)
        if (data->coefficients[k] < definition.input_dimension ||
            data->coefficients[k] >= data->workspace_size)
          return false;
    }
    const int32_t k = definition.input_dimension, d = definition.output_dimension;
    if (definition.value.output_count != d || definition.tangent.output_count != d * (1 + k) ||
        definition.hessian.output_count != d * (1 + k + k * (k + 1) / 2))
      return false;
  }
  return true;
}

bool finite(const double* values, int64_t count) noexcept {
  for (int64_t k = 0; k < count; ++k)
    if (!std::isfinite(values[k])) return false;
  return true;
}

bool nondecreasing(const double* values, int64_t count) noexcept {
  for (int64_t k = 1; k < count; ++k)
    if (!(values[k - 1] <= values[k])) return false;
  return true;
}

double factorial(int n) noexcept {
  double value = 1.0;
  for (int k = 2; k <= n; ++k) value *= k;
  return value;
}

// Nonzero basis functions and their derivatives through `order` at u in knot
// span `span` (The NURBS Book, A2.3): ders[k][j] = N^(k)_(span - p + j)(u).
void basis_derivatives(const double* knots, int span, int degree, double u, int order,
                       double ders[kMaxOrder + 1][kMaxDegree + 1]) noexcept {
  double ndu[kMaxDegree + 1][kMaxDegree + 1];
  double left[kMaxDegree + 1], right[kMaxDegree + 1];
  ndu[0][0] = 1.0;
  for (int j = 1; j <= degree; ++j) {
    left[j] = u - knots[span + 1 - j];
    right[j] = knots[span + j] - u;
    double saved = 0.0;
    for (int r = 0; r < j; ++r) {
      ndu[j][r] = right[r + 1] + left[j - r];
      const double temp = ndu[r][j - 1] / ndu[j][r];
      ndu[r][j] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    ndu[j][j] = saved;
  }
  for (int j = 0; j <= degree; ++j) ders[0][j] = ndu[j][degree];
  const int top = std::min(order, degree);
  double a[2][kMaxDegree + 1];
  for (int r = 0; r <= degree; ++r) {
    int s1 = 0, s2 = 1;
    a[0][0] = 1.0;
    for (int k = 1; k <= top; ++k) {
      double d = 0.0;
      const int rk = r - k, pk = degree - k;
      if (r >= k) {
        a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
        d = a[s2][0] * ndu[rk][pk];
      }
      const int j1 = rk >= -1 ? 1 : -rk;
      const int j2 = r - 1 <= pk ? k - 1 : degree - r;
      for (int j = j1; j <= j2; ++j) {
        a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
        d += a[s2][j] * ndu[rk + j][pk];
      }
      if (r <= pk) {
        a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
        d += a[s2][k] * ndu[r][pk];
      }
      ders[k][r] = d;
      std::swap(s1, s2);
    }
  }
  double factor = degree;
  for (int k = 1; k <= top; ++k) {
    for (int j = 0; j <= degree; ++j) ders[k][j] *= factor;
    factor *= degree - k;
  }
  for (int k = top + 1; k <= order; ++k)
    for (int j = 0; j <= degree; ++j) ders[k][j] = 0.0;
}

// Knot span of u as searchsorted(knots, u, "right") - 1, clipped to the
// active spans [degree, count - 1].
int knot_span(const double* knots, int degree, int count, double u) noexcept {
  const double* end = knots + count + degree + 1;
  const int span = static_cast<int>(std::upper_bound(knots, end, u) - knots) - 1;
  return std::clamp(span, degree, count - 1);
}

using SplineInterval = phx::mc::remesh::Interval;

// All denominators here are strictly positive knot differences or weights.
bool spline_divide(const SplineInterval& numerator, const SplineInterval& denominator,
                   SplineInterval& result) {
  using namespace phx::mc::remesh;
  if (!(denominator.lower > 0.0) || !finite(denominator)) return false;
  const SplineInterval inverse{geometry::next_down(1.0 / denominator.upper),
                               geometry::next_up(1.0 / denominator.lower)};
  result = multiply(numerator, inverse);
  return finite(result);
}

// Cox--de Boor recursion on the closure of one nonempty span.  The arrays
// contain derivatives, not Taylor coefficients. Only active terms are
// evaluated, so a repeated knot never causes division by zero.
bool spline_basis(const double* knot_lower, const double* knot_upper, int span, int degree,
                  const SplineInterval& parameter, SplineInterval basis[3][kMaxDegree + 1]) {
  using namespace phx::mc::remesh;
  const auto knot = [&](int i) { return SplineInterval{knot_lower[i], knot_upper[i]}; };
  SplineInterval previous[3][kMaxDegree + 1]{};
  previous[0][degree] = point(1.0);
  for (int d = 1; d <= degree; ++d) {
    SplineInterval next[3][kMaxDegree + 1]{};
    for (int j = degree - d; j <= degree; ++j) {
      const int i = span - degree + j;
      for (int side = 0; side < 2; ++side) {
        const int old = j + side;
        if (old < degree - d + 1 || old > degree) continue;
        const SplineInterval denominator = side == 0
            ? subtract(knot(i + d), knot(i))
            : subtract(knot(i + d + 1), knot(i + 1));
        SplineInterval factor, slope;
        const SplineInterval numerator = side == 0
            ? subtract(parameter, knot(i)) : subtract(knot(i + d + 1), parameter);
        if (!spline_divide(numerator, denominator, factor) ||
            !spline_divide(point(side == 0 ? 1.0 : -1.0), denominator, slope))
          return false;
        next[0][j] = add(next[0][j], multiply(factor, previous[0][old]));
        next[1][j] = add(next[1][j], add(multiply(factor, previous[1][old]),
                                           multiply(slope, previous[0][old])));
        next[2][j] = add(next[2][j], add(multiply(factor, previous[2][old]),
            multiply(point(2.0), multiply(slope, previous[1][old]))));
      }
    }
    for (int n = 0; n < 3; ++n)
      for (int j = 0; j <= degree; ++j) previous[n][j] = next[n][j];
  }
  for (int n = 0; n < 3; ++n)
    for (int j = 0; j <= degree; ++j) basis[n][j] = previous[n][j];
  return true;
}

int32_t spline_interval_jets(bool surface, const int64_t* ints, const double* lower,
                             const double* upper, const double* input_lower,
                             const double* input_upper, int order, double* output_lower,
                             double* output_upper) {
  using namespace phx::mc::remesh;
  const int p = static_cast<int>(ints[0]);
  const int q = surface ? static_cast<int>(ints[1]) : 0;
  const int nu = static_cast<int>(ints[surface ? 2 : 1]);
  const int nv = surface ? static_cast<int>(ints[3]) : 1;
  const int dim = surface ? 3 : static_cast<int>(ints[2]);
  const int uk = nu + p + 1, vk = surface ? nv + q + 1 : 0;
  const int64_t controls = static_cast<int64_t>(nu) * nv;
  const int64_t points = uk + vk, weights = points + controls * dim;
  const int64_t length = weights + controls;
  for (int64_t i = 0; i < length; ++i)
    if (!phx::mc::remesh::finite({lower[i], upper[i]})) return PHX_MC_INVALID_ARGUMENT;
  if (!surface && (input_lower[0] < upper[p] || input_upper[0] > lower[nu]))
    return PHX_MC_UNRESOLVED;
  double box_lower[2] = {input_lower[0], surface ? input_lower[1] : 0.0};
  double box_upper[2] = {input_upper[0], surface ? input_upper[1] : 0.0};
  bool clamped[2] = {false, false};
  if (surface) {
    const int first[2] = {p, uk + q}, last[2] = {nu, uk + nv};
    for (int axis = 0; axis < 2; ++axis) {
      clamped[axis] = input_lower[axis] < upper[first[axis]] ||
                      input_upper[axis] > lower[last[axis]];
      box_lower[axis] = std::max(lower[first[axis]],
                                 std::min(input_lower[axis], lower[last[axis]]));
      box_upper[axis] = std::min(upper[last[axis]],
                                 std::max(input_upper[axis], upper[first[axis]]));
    }
  }
  const int terms = surface ? (order + 1) * (order + 2) / 2 : order + 1;
  const int du[6] = {0, 1, surface ? 0 : 2, 2, 1, 0};
  const int dv[6] = {0, 0, surface ? 1 : 0, 0, 1, 2};
  bool found = false;
  for (int us = p; us < nu; ++us) {
    if (box_upper[0] < lower[us] || box_lower[0] > upper[us + 1]) continue;
    if (lower[us] == upper[us] && lower[us + 1] == upper[us + 1] &&
        lower[us] == lower[us + 1]) continue;
    const SplineInterval u{std::max(box_lower[0], lower[us]),
                           std::min(box_upper[0], upper[us + 1])};
    SplineInterval ub[3][kMaxDegree + 1];
    if (!spline_basis(lower, upper, us, p, u, ub)) return PHX_MC_UNRESOLVED;
    for (int vs = q; vs < (surface ? nv : 1); ++vs) {
      if (surface && (box_upper[1] < lower[uk + vs] ||
                      box_lower[1] > upper[uk + vs + 1])) continue;
      if (surface && lower[uk + vs] == upper[uk + vs] &&
          lower[uk + vs + 1] == upper[uk + vs + 1] &&
          lower[uk + vs] == lower[uk + vs + 1]) continue;
      SplineInterval vb[3][kMaxDegree + 1]{};
      if (surface) {
        const SplineInterval v{std::max(box_lower[1], lower[uk + vs]),
                               std::min(box_upper[1], upper[uk + vs + 1])};
        if (!spline_basis(lower + uk, upper + uk, vs, q, v, vb))
          return PHX_MC_UNRESOLVED;
      } else {
        vb[0][0] = point(1.0);
      }
      SplineInterval numerator[6][3]{}, denominator[6]{};
      double min_weight = std::numeric_limits<double>::infinity(), max_weight = 0.0;
      for (int a = 0; a <= p; ++a)
        for (int b = 0; b <= q; ++b) {
          const int64_t row = static_cast<int64_t>(us - p + a) * nv + vs - q + b;
          const SplineInterval weight{lower[weights + row], upper[weights + row]};
          if (!(weight.lower > 0.0)) return PHX_MC_UNRESOLVED;
          min_weight = std::min(min_weight, weight.lower);
          max_weight = std::max(max_weight, weight.upper);
          for (int t = 0; t < terms; ++t) {
            const SplineInterval basis = multiply(ub[du[t]][a], vb[dv[t]][b]);
            const SplineInterval weighted = multiply(basis, weight);
            denominator[t] = add(denominator[t], weighted);
            for (int c = 0; c < dim; ++c)
              numerator[t][c] = add(numerator[t][c], multiply(weighted,
                  {lower[points + row * dim + c], upper[points + row * dim + c]}));
          }
        }
      // Positive B-spline basis functions form a partition of unity.
      denominator[0].lower = std::max(denominator[0].lower, min_weight);
      denominator[0].upper = std::min(denominator[0].upper, max_weight);
      SplineInterval result[6][3]{};
      for (int c = 0; c < dim; ++c) {
        if (!spline_divide(numerator[0][c], denominator[0], result[0][c]))
          return PHX_MC_UNRESOLVED;
        for (int t = 1; t < terms; ++t) {
          SplineInterval rhs = subtract(numerator[t][c], multiply(denominator[t], result[0][c]));
          if (du[t] == 2)
            rhs = subtract(rhs, multiply(point(2.0), multiply(denominator[1], result[1][c])));
          if (dv[t] == 2)
            rhs = subtract(rhs, multiply(point(2.0), multiply(denominator[2], result[2][c])));
          if (du[t] == 1 && dv[t] == 1) {
            rhs = subtract(rhs, multiply(denominator[1], result[2][c]));
            rhs = subtract(rhs, multiply(denominator[2], result[1][c]));
          }
          if (!spline_divide(rhs, denominator[0], result[t][c])) return PHX_MC_UNRESOLVED;
        }
      }
      for (int t = 0; t < terms; ++t)
        for (int c = 0; c < dim; ++c) {
          if ((du[t] > 0 && clamped[0]) || (dv[t] > 0 && clamped[1])) {
            result[t][c].lower = std::min(result[t][c].lower, 0.0);
            result[t][c].upper = std::max(result[t][c].upper, 0.0);
          }
          const int slot = t * dim + c;
          if (!found) {
            output_lower[slot] = result[t][c].lower;
            output_upper[slot] = result[t][c].upper;
          } else {
            output_lower[slot] = std::min(output_lower[slot], result[t][c].lower);
            output_upper[slot] = std::max(output_upper[slot], result[t][c].upper);
          }
        }
      found = true;
    }
  }
  return found ? PHX_MC_OK : PHX_MC_UNRESOLVED;
}

// Enclosure of a univariate Bernstein polynomial's coefficients restricted to
// [a, b] within [0, 1]: coefficient k is the blossom at (a^(n-k), b^k),
// evaluated by de Casteljau steps in interval arithmetic.
void restrict_bernstein(const SplineInterval* in, int degree, double a, double b,
                        int stride, SplineInterval* out) {
  using namespace phx::mc::remesh;
  SplineInterval source[4];
  for (int i = 0; i <= degree; ++i) source[i] = in[i * stride];
  for (int k = 0; k <= degree; ++k) {
    SplineInterval work[4];
    for (int i = 0; i <= degree; ++i) work[i] = source[i];
    for (int step = 0; step < degree; ++step) {
      const double t = step < degree - k ? a : b;
      const SplineInterval weight = point(t), complement = subtract(point(1.0), point(t));
      for (int i = 0; i < degree - step; ++i)
        work[i] = add(multiply(complement, work[i]), multiply(weight, work[i + 1]));
    }
    out[k * stride] = work[0];
  }
}

// Rational cubic T-spline over the pieces meeting the box. Each piece is the
// exact Bernstein extraction on its exact parameter rectangle. Per piece the
// homogeneous Bernstein nets of the requested derivatives are restricted to
// the box's local rectangle and their hulls bound the polynomial derivatives
// (convex hull property). The local rectangle is clamped to [0, 1]: box points
// outside a piece belong to a neighbour, which is visited as well. The
// source's passing ASTS certificate (required to build the surface) proves C2
// trace continuity, so the hull of per-piece Hessians bounds the whole box.
int32_t tspline_interval_jets(const int64_t* ints, const double* lower, const double* upper,
                              const double* input_lower, const double* input_upper, int order,
                              double* output_lower, double* output_upper) {
  using namespace phx::mc::remesh;
  const int64_t pieces = ints[0], count = ints[1], width = ints[2];
  const int64_t* rows = ints + 3;
  const int64_t operators = 4 * pieces, points = operators + 16 * pieces * width;
  const int64_t weights = points + 3 * count, length = weights + count;
  for (int64_t i = 0; i < length; ++i)
    if (!finite(SplineInterval{lower[i], upper[i]})) return PHX_MC_INVALID_ARGUMENT;
  const auto at = [&](int64_t i) { return SplineInterval{lower[i], upper[i]}; };
  const int terms = (order + 1) * (order + 2) / 2;
  const int du[6] = {0, 1, 0, 2, 1, 0}, dv[6] = {0, 0, 1, 0, 1, 2};
  bool found = false;
  for (int64_t piece = 0; piece < pieces; ++piece) {
    const int64_t b = 4 * piece;
    if (input_upper[0] < lower[b] || input_lower[0] > upper[b + 1] ||
        input_upper[1] < lower[b + 2] || input_lower[1] > upper[b + 3])
      continue;
    SplineInterval local[2], scale[2];
    bool empty = false;
    for (int axis = 0; axis < 2; ++axis) {
      const SplineInterval first = at(b + 2 * axis), last = at(b + 2 * axis + 1);
      const SplineInterval extent = subtract(last, first);
      if (!spline_divide(subtract(SplineInterval{input_lower[axis], input_upper[axis]}, first),
                         extent, local[axis]) ||
          !spline_divide(point(1.0), extent, scale[axis]))
        return PHX_MC_UNRESOLVED;
      local[axis] = {std::max(local[axis].lower, 0.0), std::min(local[axis].upper, 1.0)};
      empty = empty || local[axis].lower > local[axis].upper;
    }
    if (empty) continue;
    // Homogeneous Bernstein net (w x, w y, w z, w) of this piece.
    SplineInterval net[4][4][4]{};
    double min_weight = std::numeric_limits<double>::infinity(), max_weight = 0.0;
    for (int64_t l = 0; l < width; ++l) {
      const int64_t row = rows[piece * width + l];
      if (row < 0) continue;
      const SplineInterval weight = at(weights + row);
      if (!(weight.lower > 0.0)) return PHX_MC_UNRESOLVED;
      min_weight = std::min(min_weight, weight.lower);
      max_weight = std::max(max_weight, weight.upper);
      SplineInterval homogeneous[4];
      for (int c = 0; c < 3; ++c) homogeneous[c] = multiply(weight, at(points + 3 * row + c));
      homogeneous[3] = weight;
      const int64_t op = operators + 16 * (piece * width + l);
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
          const SplineInterval entry = at(op + 4 * i + j);
          if (entry.lower == 0.0 && entry.upper == 0.0) continue;
          for (int c = 0; c < 4; ++c)
            net[c][i][j] = add(net[c][i][j], multiply(entry, homogeneous[c]));
        }
    }
    SplineInterval numerator[6][3]{}, denominator[6]{};
    for (int t = 0; t < terms; ++t) {
      const int nu = 3 - du[t], nv = 3 - dv[t];
      // Derivative nets: forward differences times n!/(n-d)!, times the
      // chain-rule factor of the piece's parameter scaling.
      SplineInterval factor = point(du[t] == 0 ? 1.0 : du[t] == 1 ? 3.0 : 6.0);
      factor = multiply(factor, point(dv[t] == 0 ? 1.0 : dv[t] == 1 ? 3.0 : 6.0));
      for (int k = 0; k < du[t]; ++k) factor = multiply(factor, scale[0]);
      for (int k = 0; k < dv[t]; ++k) factor = multiply(factor, scale[1]);
      for (int c = 0; c < 4; ++c) {
        SplineInterval d[4][4];
        for (int i = 0; i <= nu; ++i)
          for (int j = 0; j <= nv; ++j) {
            SplineInterval value = point(0.0);
            for (int p = 0; p <= du[t]; ++p)
              for (int q = 0; q <= dv[t]; ++q) {
                const double sign = ((du[t] - p + dv[t] - q) % 2 == 0 ? 1.0 : -1.0) *
                                    (du[t] == 2 && p == 1 ? 2.0 : 1.0) *
                                    (dv[t] == 2 && q == 1 ? 2.0 : 1.0);
                value = add(value, multiply(point(sign), net[c][i + p][j + q]));
              }
            d[i][j] = value;
          }
        SplineInterval restricted[4][4];
        for (int i = 0; i <= nu; ++i)
          restrict_bernstein(&d[i][0], nv, local[1].lower, local[1].upper, 1, &restricted[i][0]);
        for (int j = 0; j <= nv; ++j)
          restrict_bernstein(&restricted[0][j], nu, local[0].lower, local[0].upper, 4,
                             &restricted[0][j]);
        SplineInterval hull = restricted[0][0];
        for (int i = 0; i <= nu; ++i)
          for (int j = 0; j <= nv; ++j)
            hull = {std::min(hull.lower, restricted[i][j].lower),
                    std::max(hull.upper, restricted[i][j].upper)};
        hull = multiply(hull, factor);
        if (c == 3) denominator[t] = hull;
        else numerator[t][c] = hull;
      }
    }
    // The extracted basis is a nonnegative partition of unity on the piece.
    denominator[0].lower = std::max(denominator[0].lower, min_weight);
    denominator[0].upper = std::min(denominator[0].upper, max_weight);
    SplineInterval result[6][3]{};
    for (int c = 0; c < 3; ++c) {
      if (!spline_divide(numerator[0][c], denominator[0], result[0][c]))
        return PHX_MC_UNRESOLVED;
      for (int t = 1; t < terms; ++t) {
        SplineInterval rhs = subtract(numerator[t][c], multiply(denominator[t], result[0][c]));
        if (du[t] == 2)
          rhs = subtract(rhs, multiply(point(2.0), multiply(denominator[1], result[1][c])));
        if (dv[t] == 2)
          rhs = subtract(rhs, multiply(point(2.0), multiply(denominator[2], result[2][c])));
        if (du[t] == 1 && dv[t] == 1) {
          rhs = subtract(rhs, multiply(denominator[1], result[2][c]));
          rhs = subtract(rhs, multiply(denominator[2], result[1][c]));
        }
        if (!spline_divide(rhs, denominator[0], result[t][c])) return PHX_MC_UNRESOLVED;
      }
    }
    for (int t = 0; t < terms; ++t)
      for (int c = 0; c < 3; ++c) {
        const int slot = t * 3 + c;
        output_lower[slot] = found ? std::min(output_lower[slot], result[t][c].lower)
                                   : result[t][c].lower;
        output_upper[slot] = found ? std::max(output_upper[slot], result[t][c].upper)
                                   : result[t][c].upper;
      }
    found = true;
  }
  return found ? PHX_MC_OK : PHX_MC_UNRESOLVED;
}

// Normalized interval Taylor coefficients through total order three. This
// evaluates the smooth value program directly, avoiding branch flags in the
// generated derivative programs when an offset needs a certified normal.
struct IntervalTaylor {
  SplineInterval c[10]{};
};

IntervalTaylor interval_taylor_multiply(const IntervalTaylor& a, const IntervalTaylor& b,
                                        const Truncation& truncation) {
  using namespace phx::mc::remesh;
  IntervalTaylor result;
  for (int n = 0; n <= truncation.order; ++n)
    for (int j = 0; j <= truncation.v_limit(n); ++j) {
      const int i = n - j;
      for (int u = 0; u <= i; ++u)
        for (int v = 0; v <= j; ++v)
          result.c[term(i, j)] = add(result.c[term(i, j)],
              multiply(a.c[term(u, v)], b.c[term(i - u, j - v)]));
    }
  return result;
}

IntervalTaylor interval_taylor_compose(const IntervalTaylor& input,
                                       const SplineInterval* derivatives,
                                       const Truncation& truncation) {
  using namespace phx::mc::remesh;
  IntervalTaylor delta = input, result;
  delta.c[0] = point(0);
  result.c[0] = derivatives[truncation.order];
  for (int n = truncation.order - 1; n >= 0; --n) {
    result = interval_taylor_multiply(result, delta, truncation);
    result.c[0] = add(result.c[0], derivatives[n]);
  }
  return result;
}

bool interval_taylor_inverse(const IntervalTaylor& input, const Truncation& truncation,
                             IntervalTaylor& result) {
  using namespace phx::mc::remesh;
  const SplineInterval x = input.c[0];
  if (!finite(x) || (x.lower <= 0 && x.upper >= 0)) return false;
  const SplineInterval inverse{geometry::next_down(1.0 / x.upper),
                               geometry::next_up(1.0 / x.lower)};
  SplineInterval derivatives[4];
  derivatives[0] = inverse;
  for (int n = 1; n <= truncation.order; ++n)
    derivatives[n] = subtract(point(0), multiply(derivatives[n - 1], inverse));
  result = interval_taylor_compose(input, derivatives, truncation);
  return true;
}

bool interval_taylor_unary(int64_t opcode, const IntervalTaylor& input,
                           const Truncation& truncation, IntervalTaylor& result) {
  using namespace phx::mc::remesh;
  const SplineInterval x = input.c[0];
  if (opcode == 12) {
    result = interval_taylor_multiply(input, input, truncation);
    return true;
  }
  if (opcode == 28) {
    IntervalTaylor sine, cosine, inverse;
    if (!interval_taylor_unary(8, input, truncation, sine) ||
        !interval_taylor_unary(9, input, truncation, cosine) ||
        !interval_taylor_inverse(cosine, truncation, inverse)) return false;
    result = interval_taylor_multiply(sine, inverse, truncation);
    return true;
  }
  SplineInterval sine{}, cosine{}, exponential{};
  if (opcode == 17) {
    const auto e = elementary::exp(x.lower, x.upper);
    exponential = {e.lower, e.upper};
  } else if (opcode == 8 || opcode == 9) {
    const auto s = elementary::sin(x.lower, x.upper);
    const auto c = elementary::cos(x.lower, x.upper);
    sine = {s.lower, s.upper};
    cosine = {c.lower, c.upper};
  } else {
    return false;
  }
  SplineInterval derivatives[4];
  for (int n = 0; n <= truncation.order; ++n) {
    SplineInterval value;
    if (opcode == 17) value = exponential;
    else if (opcode == 8)
      value = n % 4 == 0 ? sine : n % 4 == 1 ? cosine
              : n % 4 == 2 ? subtract(point(0), sine) : subtract(point(0), cosine);
    else if (opcode == 9)
      value = n % 4 == 0 ? cosine : n % 4 == 1 ? subtract(point(0), sine)
              : n % 4 == 2 ? subtract(point(0), cosine) : sine;
    else return false;
    if (!spline_divide(value, point(factorial(n)), derivatives[n])) return false;
  }
  result = interval_taylor_compose(input, derivatives, truncation);
  return true;
}

int32_t interval_template_jets(const CadGeometry& geometry, int32_t index,
                               const double* input_lower, const double* input_upper, int order,
                               double* output_lower, double* output_upper,
                               IntervalEvaluationWorkspace& workspace) {
  using namespace phx::mc::remesh;
  const Node& node = geometry.node(index);
  const ProgramData& program = node.definition->value;
  const Truncation truncation{node.inputs, order};
  const int64_t stride = 10, size = program.workspace_size * stride;
  workspace.taylor_lower.assign(static_cast<std::size_t>(size), 0.0);
  workspace.taylor_upper.assign(static_cast<std::size_t>(size), 0.0);
  const auto store = [&](int64_t slot, const IntervalTaylor& value) {
    for (int k = 0; k < stride; ++k) {
      workspace.taylor_lower[static_cast<std::size_t>(slot * stride + k)] = value.c[k].lower;
      workspace.taylor_upper[static_cast<std::size_t>(slot * stride + k)] = value.c[k].upper;
    }
  };
  const auto load = [&](int64_t slot) {
    IntervalTaylor value;
    for (int k = 0; k < stride; ++k)
      value.c[k] = {workspace.taylor_lower[static_cast<std::size_t>(slot * stride + k)],
                    workspace.taylor_upper[static_cast<std::size_t>(slot * stride + k)]};
    return value;
  };
  for (int64_t i = 0; i < program.workspace_size; ++i) {
    IntervalTaylor seed;
    seed.c[0] = point(program.seeds[i]);
    store(i, seed);
  }
  for (int axis = 0; axis < node.inputs; ++axis) {
    IntervalTaylor input;
    input.c[0] = {input_lower[axis], input_upper[axis]};
    if (order > 0) input.c[axis == 0 ? term(1, 0) : term(0, 1)] = point(1);
    store(axis, input);
  }
  const int64_t offset = geometry.model().node_real_offsets[index];
  for (int i = 0; i < node.definition->coefficient_count; ++i) {
    IntervalTaylor coefficient;
    coefficient.c[0] = {workspace.coefficient_lower[offset + i],
                        workspace.coefficient_upper[offset + i]};
    if (!phx::mc::remesh::finite(coefficient.c[0])) return PHX_MC_INVALID_ARGUMENT;
    store(program.coefficients[i], coefficient);
  }
  for (int64_t row = 0; row < program.instruction_count; ++row) {
    const int64_t* op = program.instructions + 8 * row;
    for (int64_t lane = 0; lane < op[2]; ++lane) {
      const auto operand = [&](int64_t base, int64_t indices) {
        return load(base + (indices < 0 ? lane : program.indices[indices + lane]));
      };
      const IntervalTaylor a = operand(op[3], op[4]);
      IntervalTaylor value;
      if (op[0] == 2 || op[0] == 3) {
        const IntervalTaylor b = operand(op[5], op[6]);
        for (int k = 0; k < stride; ++k)
          value.c[k] = op[0] == 2 ? add(a.c[k], b.c[k]) : subtract(a.c[k], b.c[k]);
      } else if (op[0] == 5) {
        value = interval_taylor_multiply(a, operand(op[5], op[6]), truncation);
      } else if (op[0] == 6) {
        for (int k = 0; k < stride; ++k) value.c[k] = subtract(point(0), a.c[k]);
      } else if (op[0] == 25) {
        IntervalTaylor inverse;
        if (!interval_taylor_inverse(operand(op[5], op[6]), truncation, inverse))
          return PHX_MC_UNRESOLVED;
        value = interval_taylor_multiply(a, inverse, truncation);
      } else if (!interval_taylor_unary(op[0], a, truncation, value)) {
        return PHX_MC_UNRESOLVED;
      }
      store(op[1] + lane, value);
    }
  }
  int slot = 0;
  for (int n = 0; n <= order; ++n)
    for (int j = 0; j <= truncation.v_limit(n); ++j)
      for (int component = 0; component < node.outputs; ++component) {
        const IntervalTaylor value = load(program.outputs[component]);
        const SplineInterval derivative = multiply(value.c[term(n - j, j)],
                                                    point(factorial(n - j) * factorial(j)));
        if (!phx::mc::remesh::finite(derivative)) return PHX_MC_UNRESOLVED;
        output_lower[slot] = derivative.lower;
        output_upper[slot++] = derivative.upper;
      }
  return PHX_MC_OK;
}

int32_t offset_interval_jets(const CadGeometry& geometry, const Node& node,
                             const SplineInterval* coefficients, const double* input_lower,
                             const double* input_upper, int order, double* output_lower,
                             double* output_upper, IntervalEvaluationWorkspace& workspace) {
  using namespace phx::mc::remesh;
  const bool surface = node.kind == PHX_MC_CAD_NODE_OFFSET_SURFACE;
  const int dim = node.outputs, axes = surface ? 2 : 1;
  for (int64_t i = 0; i < node.real_count; ++i)
    if (!phx::mc::remesh::finite(coefficients[i])) return PHX_MC_INVALID_ARGUMENT;
  double base_lower[30], base_upper[30];
  const int32_t child = static_cast<int32_t>(node.integers[0]);
  const bool previous_direct_template = workspace.direct_template;
  workspace.direct_template = true;
  const int32_t status = geometry.node(child).kind == PHX_MC_CAD_NODE_TEMPLATE
      ? interval_template_jets(geometry, child, input_lower, input_upper, order + 1,
                               base_lower, base_upper, workspace)
      : order == 2 ? PHX_MC_UNRESOLVED
      : geometry.interval_jets(child, input_lower, input_upper, order + 1,
                               base_lower, base_upper, workspace);
  workspace.direct_template = previous_direct_template;
  if (status != PHX_MC_OK) return status;
  const auto base = [&](int term, int coordinate) {
    return SplineInterval{base_lower[term * dim + coordinate], base_upper[term * dim + coordinate]};
  };
  SplineInterval normal[3]{}, derivative[2][3]{}, second[3][3]{};
  if (surface) {
    for (int c = 0; c < 3; ++c) {
      const int a = (c + 1) % 3, b = (c + 2) % 3;
      normal[c] = subtract(multiply(base(1, a), base(2, b)),
                           multiply(base(1, b), base(2, a)));
      if (order >= 1) {
        derivative[0][c] = add(subtract(multiply(base(3, a), base(2, b)),
                                         multiply(base(3, b), base(2, a))),
                                subtract(multiply(base(1, a), base(4, b)),
                                         multiply(base(1, b), base(4, a))));
        derivative[1][c] = add(subtract(multiply(base(4, a), base(2, b)),
                                         multiply(base(4, b), base(2, a))),
                                subtract(multiply(base(1, a), base(5, b)),
                                         multiply(base(1, b), base(5, a))));
      }
      if (order == 2) {
        const auto cross = [&](int first, int last) {
          return subtract(multiply(base(first, a), base(last, b)),
                          multiply(base(first, b), base(last, a)));
        };
        second[0][c] = add(add(cross(6, 2), multiply(point(2), cross(3, 4))), cross(1, 7));
        second[1][c] = add(add(cross(7, 2), cross(3, 5)), cross(1, 8));
        second[2][c] = add(add(cross(8, 2), multiply(point(2), cross(4, 5))), cross(1, 9));
      }
    }
  } else if (dim == 2) {
    normal[0] = base(1, 1);
    normal[1] = subtract(point(0), base(1, 0));
    if (order >= 1) {
      derivative[0][0] = base(2, 1);
      derivative[0][1] = subtract(point(0), base(2, 0));
    }
    if (order == 2) {
      second[0][0] = base(3, 1);
      second[0][1] = subtract(point(0), base(3, 0));
    }
  } else {
    for (int c = 0; c < 3; ++c) {
      const int a = (c + 1) % 3, b = (c + 2) % 3;
      normal[c] = subtract(multiply(base(1, a), coefficients[1 + b]),
                           multiply(base(1, b), coefficients[1 + a]));
      if (order >= 1)
        derivative[0][c] = subtract(multiply(base(2, a), coefficients[1 + b]),
                                    multiply(base(2, b), coefficients[1 + a]));
      if (order == 2)
        second[0][c] = subtract(multiply(base(3, a), coefficients[1 + b]),
                                multiply(base(3, b), coefficients[1 + a]));
    }
  }
  double least[3]{}, most[3]{};
  for (int c = 0; c < dim; ++c) {
    least[c] = mignitude(normal[c]);
    most[c] = magnitude(normal[c]);
  }
  const SplineInterval length{geometry::norm_lower(least), geometry::norm_upper(most)};
  SplineInterval unit[3];
  for (int c = 0; c < dim; ++c)
    if (!spline_divide(normal[c], length, unit[c])) return PHX_MC_UNRESOLVED;
  SplineInterval unit_derivative[2][3]{}, length_derivative[2]{};
  if (order >= 1) {
    for (int axis = 0; axis < axes; ++axis) {
      for (int c = 0; c < dim; ++c)
        length_derivative[axis] = add(length_derivative[axis],
                                       multiply(unit[c], derivative[axis][c]));
      for (int c = 0; c < dim; ++c) {
        const SplineInterval tangent = subtract(derivative[axis][c],
                                               multiply(unit[c], length_derivative[axis]));
        if (!spline_divide(tangent, length, unit_derivative[axis][c])) return PHX_MC_UNRESOLVED;
      }
    }
  }
  for (int term = 0; term < 1 + (order >= 1 ? axes : 0); ++term)
    for (int c = 0; c < dim; ++c) {
      const SplineInterval direction = term == 0 ? unit[c] : unit_derivative[term - 1][c];
      const SplineInterval value = add(base(term, c), multiply(coefficients[0], direction));
      if (!phx::mc::remesh::finite(value)) return PHX_MC_UNRESOLVED;
      output_lower[term * dim + c] = value.lower;
      output_upper[term * dim + c] = value.upper;
    }
  if (order == 2) {
    const int second_count = surface ? 3 : 1;
    const int first_axis[3] = {0, 0, 1}, second_axis[3] = {0, 1, 1};
    for (int k = 0; k < second_count; ++k) {
      const int i = first_axis[k], j = second_axis[k];
      SplineInterval length_second = point(0);
      for (int c = 0; c < dim; ++c)
        length_second = add(length_second,
            add(multiply(unit_derivative[j][c], derivative[i][c]),
                multiply(unit[c], second[k][c])));
      for (int c = 0; c < dim; ++c) {
        SplineInterval numerator = subtract(second[k][c],
            multiply(unit_derivative[j][c], length_derivative[i]));
        numerator = subtract(numerator, multiply(unit_derivative[i][c], length_derivative[j]));
        numerator = subtract(numerator, multiply(unit[c], length_second));
        SplineInterval direction;
        if (!spline_divide(numerator, length, direction)) return PHX_MC_UNRESOLVED;
        const int term = 1 + axes + k;
        const SplineInterval value = add(base(term, c), multiply(coefficients[0], direction));
        if (!phx::mc::remesh::finite(value)) return PHX_MC_UNRESOLVED;
        output_lower[term * dim + c] = value.lower;
        output_upper[term * dim + c] = value.upper;
      }
    }
  }
  return PHX_MC_OK;
}

// Bernstein cubic and its derivatives at t, as the T-spline realization.
void bernstein_cubic(double t, int derivative, double* out) noexcept {
  const double s = 1.0 - t;
  switch (derivative) {
    case 0:
      out[0] = s * s * s, out[1] = 3.0 * t * (s * s), out[2] = 3.0 * (t * t) * s,
      out[3] = t * t * t;
      return;
    case 1:
      out[0] = -3.0 * (s * s), out[1] = 3.0 * s * (1.0 - 3.0 * t),
      out[2] = 3.0 * t * (2.0 - 3.0 * t), out[3] = 3.0 * (t * t);
      return;
    case 2:
      out[0] = 6.0 * s, out[1] = 18.0 * t - 12.0, out[2] = 6.0 - 18.0 * t, out[3] = 6.0 * t;
      return;
    case 3:
      out[0] = -6.0, out[1] = 18.0, out[2] = -18.0, out[3] = 6.0;
      return;
    default:
      out[0] = out[1] = out[2] = out[3] = 0.0;
  }
}

// x = A^-1 b for a 3x3 row-major A by Gaussian elimination with partial
// pivoting: the implicit-function derivatives of an intersection chart.
bool solve3(const double* matrix, const double* rhs, double* solution) noexcept {
  double a[3][4];
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) a[r][c] = matrix[3 * r + c];
    a[r][3] = rhs[r];
  }
  for (int column = 0; column < 3; ++column) {
    int pivot = column;
    for (int r = column + 1; r < 3; ++r)
      if (std::fabs(a[r][column]) > std::fabs(a[pivot][column])) pivot = r;
    if (!(std::fabs(a[pivot][column]) > 0.0)) return false;
    if (pivot != column)
      for (int c = 0; c < 4; ++c) std::swap(a[pivot][c], a[column][c]);
    for (int r = column + 1; r < 3; ++r) {
      const double factor = a[r][column] / a[column][column];
      for (int c = column; c < 4; ++c) a[r][c] -= factor * a[column][c];
    }
  }
  for (int r = 2; r >= 0; --r) {
    double value = a[r][3];
    for (int c = r + 1; c < 3; ++c) value -= a[r][c] * solution[c];
    solution[r] = value / a[r][r];
  }
  return std::isfinite(solution[0]) && std::isfinite(solution[1]) && std::isfinite(solution[2]);
}

// The constant-term-only identity substitution needs no composition.
bool identity_inputs(const Taylor* inputs, int count, const Truncation& t) noexcept {
  if (t.variables != count) return false;
  for (int axis = 0; axis < count; ++axis)
    for (int n = 1; n <= t.order; ++n)
      for (int j = 0; j <= t.v_limit(n); ++j) {
        const double expected = n == 1 && j == axis ? 1.0 : 0.0;
        if (inputs[axis].c[term(n - j, j)] != expected) return false;
      }
  return true;
}

// Reverses a univariate expansion's parameter: f(-t) about its point.
void reverse(Taylor* values, int count, int order) noexcept {
  for (int k = 0; k < count; ++k)
    for (int n = 1; n <= order; n += 2) values[k].c[term(n, 0)] = -values[k].c[term(n, 0)];
}

}  // namespace

int32_t CadGeometry::interval_jets(int32_t index, const double* input_lower,
                                   const double* input_upper, int order, double* output_lower,
                                   double* output_upper,
                                   IntervalEvaluationWorkspace& workspace) const {
  if (index < 0 || index >= model_.node_count || order < 0 || order > 2 ||
      input_lower == nullptr || input_upper == nullptr || output_lower == nullptr ||
      output_upper == nullptr || workspace.coefficient_count < model_.node_real_offsets[model_.node_count] ||
      (workspace.coefficient_count > 0 &&
       (workspace.coefficient_lower == nullptr || workspace.coefficient_upper == nullptr)))
    return PHX_MC_INVALID_ARGUMENT;
  const Node& current = nodes_[static_cast<std::size_t>(index)];
  for (int i = 0; i < current.inputs; ++i)
    if (!std::isfinite(input_lower[i]) || !std::isfinite(input_upper[i]) ||
        input_lower[i] > input_upper[i])
      return PHX_MC_INVALID_ARGUMENT;
  if (current.kind == PHX_MC_CAD_NODE_OFFSET_SURFACE ||
      current.kind == PHX_MC_CAD_NODE_OFFSET_CURVE) {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    const int64_t offset = model_.node_real_offsets[index];
    SplineInterval coefficients[4];
    for (int64_t i = 0; i < current.real_count; ++i)
      coefficients[i] = {workspace.coefficient_lower[offset + i],
                         workspace.coefficient_upper[offset + i]};
    return offset_interval_jets(*this, current, coefficients, input_lower, input_upper, order,
                                output_lower, output_upper, workspace);
  }
  if (current.kind == PHX_MC_CAD_NODE_BSPLINE_CURVE ||
      current.kind == PHX_MC_CAD_NODE_BSPLINE_SURFACE) {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    const int64_t offset = model_.node_real_offsets[index];
    return spline_interval_jets(current.kind == PHX_MC_CAD_NODE_BSPLINE_SURFACE,
                                current.integers, workspace.coefficient_lower + offset,
                                workspace.coefficient_upper + offset, input_lower, input_upper,
                                order, output_lower, output_upper);
  }
  if (current.kind == PHX_MC_CAD_NODE_TSPLINE_SURFACE) {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    const int64_t offset = model_.node_real_offsets[index];
    return tspline_interval_jets(current.integers, workspace.coefficient_lower + offset,
                                 workspace.coefficient_upper + offset, input_lower, input_upper,
                                 order, output_lower, output_upper);
  }
  if (current.kind == PHX_MC_CAD_NODE_REVOLUTION) {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    namespace interval = phx::mc::remesh;
    double curve_lower[9], curve_upper[9];
    const int32_t status = interval_jets(static_cast<int32_t>(current.integers[0]),
                                         input_lower + 1, input_upper + 1, order, curve_lower,
                                         curve_upper, workspace);
    if (status != PHX_MC_OK) return status;
    const int64_t offset = model_.node_real_offsets[index];
    interval::Interval origin[3], axis[3];
    for (int i = 0; i < 3; ++i) {
      origin[i] = {workspace.coefficient_lower[offset + i], workspace.coefficient_upper[offset + i]};
      axis[i] = {workspace.coefficient_lower[offset + 3 + i],
                 workspace.coefficient_upper[offset + 3 + i]};
      if (!interval::finite(origin[i]) || !interval::finite(axis[i])) return PHX_MC_INVALID_ARGUMENT;
    }
    const auto sin_enclosure = phx::mc::elementary::sin(input_lower[0], input_upper[0]);
    const auto cos_enclosure = phx::mc::elementary::cos(input_lower[0], input_upper[0]);
    const interval::Interval sine{sin_enclosure.lower, sin_enclosure.upper};
    const interval::Interval cosine{cos_enclosure.lower, cos_enclosure.upper};
    const interval::Interval negative_sine{-sine.upper, -sine.lower};
    const interval::Interval negative_cosine{-cosine.upper, -cosine.lower};
    double result_lower[18]{}, result_upper[18]{};
    for (int derivative = 0; derivative <= order; ++derivative) {
      interval::Interval relative[3], radial[3], axial[3], cross[3];
      auto dot = interval::point(0.0);
      for (int i = 0; i < 3; ++i) {
        const int k = derivative * 3 + i;
        relative[i] = {curve_lower[k], curve_upper[k]};
        if (derivative == 0) relative[i] = interval::subtract(relative[i], origin[i]);
        dot = interval::add(dot, interval::multiply(axis[i], relative[i]));
      }
      for (int i = 0; i < 3; ++i) {
        axial[i] = interval::multiply(axis[i], dot);
        radial[i] = interval::subtract(relative[i], axial[i]);
      }
      for (int i = 0; i < 3; ++i)
        cross[i] = interval::subtract(interval::multiply(axis[(i + 1) % 3], radial[(i + 2) % 3]),
                                      interval::multiply(axis[(i + 2) % 3], radial[(i + 1) % 3]));
      for (int i = 0; i < 3; ++i) {
        auto value = interval::add(axial[i], interval::multiply(cosine, radial[i]));
        value = interval::add(value, interval::multiply(sine, cross[i]));
        if (derivative == 0) value = interval::add(value, origin[i]);
        if (!interval::finite(value)) return PHX_MC_UNRESOLVED;
        const int row = derivative == 0 ? 0 : derivative == 1 ? 6 : 15;
        result_lower[row + i] = value.lower;
        result_upper[row + i] = value.upper;
        if (derivative < order) {
          value = interval::add(interval::multiply(negative_sine, radial[i]),
                                interval::multiply(cosine, cross[i]));
          if (!interval::finite(value)) return PHX_MC_UNRESOLVED;
          const int angle_row = derivative == 0 ? 3 : 12;
          result_lower[angle_row + i] = value.lower;
          result_upper[angle_row + i] = value.upper;
        }
        if (derivative == 0 && order >= 2) {
          value = interval::add(interval::multiply(negative_cosine, radial[i]),
                                interval::multiply(negative_sine, cross[i]));
          if (!interval::finite(value)) return PHX_MC_UNRESOLVED;
          result_lower[9 + i] = value.lower;
          result_upper[9 + i] = value.upper;
        }
      }
    }
    const int terms = (order + 1) * (order + 2) / 2;
    for (int i = 0; i < terms * 3; ++i) {
      output_lower[i] = result_lower[i];
      output_upper[i] = result_upper[i];
    }
    return PHX_MC_OK;
  }
  if (current.kind == PHX_MC_CAD_NODE_RULED) {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    namespace interval = phx::mc::remesh;
    double a_lower[9], a_upper[9], b_lower[9], b_upper[9];
    int32_t status = interval_jets(static_cast<int32_t>(current.integers[0]),
                                    input_lower, input_upper, order, a_lower, a_upper, workspace);
    if (status != PHX_MC_OK) return status;
    status = interval_jets(static_cast<int32_t>(current.integers[1]),
                           input_lower, input_upper, order, b_lower, b_upper, workspace);
    if (status != PHX_MC_OK) return status;
    const interval::Interval v{input_lower[1], input_upper[1]};
    const auto complement = interval::subtract(interval::point(1.0), v);
    double result_lower[18]{}, result_upper[18]{};
    for (int component = 0; component < 3; ++component) {
      for (int derivative = 0; derivative <= order; ++derivative) {
        const int k = derivative * 3 + component;
        const auto value = interval::add(
            interval::multiply(complement, {a_lower[k], a_upper[k]}),
            interval::multiply(v, {b_lower[k], b_upper[k]}));
        if (!interval::finite(value)) return PHX_MC_UNRESOLVED;
        const int out = (derivative == 0 ? 0 : derivative == 1 ? 3 : 9) + component;
        result_lower[out] = value.lower;
        result_upper[out] = value.upper;
      }
      if (order >= 1) {
        const auto value = interval::subtract({b_lower[component], b_upper[component]},
                                              {a_lower[component], a_upper[component]});
        if (!interval::finite(value)) return PHX_MC_UNRESOLVED;
        result_lower[6 + component] = value.lower;
        result_upper[6 + component] = value.upper;
      }
      if (order >= 2) {
        const auto value = interval::subtract({b_lower[3 + component], b_upper[3 + component]},
                                              {a_lower[3 + component], a_upper[3 + component]});
        if (!interval::finite(value)) return PHX_MC_UNRESOLVED;
        result_lower[12 + component] = value.lower;
        result_upper[12 + component] = value.upper;
      }
    }
    const int terms = (order + 1) * (order + 2) / 2;
    for (int i = 0; i < terms * 3; ++i) {
      output_lower[i] = result_lower[i];
      output_upper[i] = result_upper[i];
    }
    return PHX_MC_OK;
  }
  if (current.kind == PHX_MC_CAD_NODE_EXTRUSION) {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    namespace interval = phx::mc::remesh;
    double curve_lower[9], curve_upper[9], result_lower[18]{}, result_upper[18]{};
    const int32_t status = interval_jets(static_cast<int32_t>(current.integers[0]),
                                         input_lower, input_upper, order, curve_lower,
                                         curve_upper, workspace);
    if (status != PHX_MC_OK) return status;
    const int64_t offset = model_.node_real_offsets[index];
    const interval::Interval v{input_lower[1], input_upper[1]};
    for (int component = 0; component < 3; ++component) {
      const interval::Interval direction{workspace.coefficient_lower[offset + component],
                                          workspace.coefficient_upper[offset + component]};
      if (!interval::finite(direction)) return PHX_MC_INVALID_ARGUMENT;
      const auto value = interval::add({curve_lower[component], curve_upper[component]},
                                       interval::multiply(v, direction));
      if (!interval::finite(value)) return PHX_MC_UNRESOLVED;
      result_lower[component] = value.lower;
      result_upper[component] = value.upper;
      if (order >= 1) {
        result_lower[3 + component] = curve_lower[3 + component];
        result_upper[3 + component] = curve_upper[3 + component];
        result_lower[6 + component] = direction.lower;
        result_upper[6 + component] = direction.upper;
      }
      if (order >= 2) {
        result_lower[9 + component] = curve_lower[6 + component];
        result_upper[9 + component] = curve_upper[6 + component];
      }
    }
    const int terms = (order + 1) * (order + 2) / 2;
    for (int i = 0; i < terms * 3; ++i) {
      output_lower[i] = result_lower[i];
      output_upper[i] = result_upper[i];
    }
    return PHX_MC_OK;
  }
  if (current.kind == PHX_MC_CAD_NODE_SURFACE_MAPPED) {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    namespace interval = phx::mc::remesh;
    double chart_lower[6], chart_upper[6], surface_lower[18], surface_upper[18];
    int32_t status = interval_jets(static_cast<int32_t>(current.integers[1]), input_lower,
                                    input_upper, order, chart_lower, chart_upper, workspace);
    if (status != PHX_MC_OK) return status;
    status = interval_jets(static_cast<int32_t>(current.integers[0]), chart_lower, chart_upper,
                           order, surface_lower, surface_upper, workspace);
    if (status != PHX_MC_OK) return status;
    double result_lower[9], result_upper[9];
    for (int component = 0; component < 3; ++component) {
      auto surface = [&](int term) {
        const int k = term * 3 + component;
        return interval::Interval{surface_lower[k], surface_upper[k]};
      };
      auto chart = [&](int term, int axis) {
        const int k = term * 2 + axis;
        return interval::Interval{chart_lower[k], chart_upper[k]};
      };
      auto value = surface(0);
      result_lower[component] = value.lower;
      result_upper[component] = value.upper;
      if (order >= 1) {
        value = interval::add(interval::multiply(surface(1), chart(1, 0)),
                              interval::multiply(surface(2), chart(1, 1)));
        if (!interval::finite(value)) return PHX_MC_UNRESOLVED;
        result_lower[3 + component] = value.lower;
        result_upper[3 + component] = value.upper;
      }
      if (order >= 2) {
        const auto u = chart(1, 0), v = chart(1, 1);
        auto second = interval::multiply(surface(3), interval::multiply(u, u));
        auto mixed = interval::multiply(surface(4), interval::multiply(u, v));
        second = interval::add(second, interval::add(mixed, mixed));
        second = interval::add(second, interval::multiply(surface(5), interval::multiply(v, v)));
        second = interval::add(second, interval::multiply(surface(1), chart(2, 0)));
        second = interval::add(second, interval::multiply(surface(2), chart(2, 1)));
        if (!interval::finite(second)) return PHX_MC_UNRESOLVED;
        result_lower[6 + component] = second.lower;
        result_upper[6 + component] = second.upper;
      }
    }
    for (int i = 0; i < (order + 1) * 3; ++i) {
      output_lower[i] = result_lower[i];
      output_upper[i] = result_upper[i];
    }
    return PHX_MC_OK;
  }
  if (current.kind == PHX_MC_CAD_NODE_ISOPARAMETRIC) {
    const int fixed = static_cast<int>(current.integers[1]);
    const int64_t offset = model_.node_real_offsets[index];
    double lo[2], hi[2];
    lo[fixed] = workspace.coefficient_lower[offset];
    hi[fixed] = workspace.coefficient_upper[offset];
    lo[1 - fixed] = input_lower[0];
    hi[1 - fixed] = input_upper[0];
    double child_lower[18], child_upper[18];
    const int32_t status = interval_jets(static_cast<int32_t>(current.integers[0]), lo, hi,
                                         order, child_lower, child_upper, workspace);
    if (status != PHX_MC_OK) return status;
    const int positions[] = {0, fixed == 0 ? 2 : 1, fixed == 0 ? 5 : 3};
    for (int term = 0; term <= order; ++term)
      for (int component = 0; component < current.outputs; ++component) {
        const int out = term * current.outputs + component;
        const int in = positions[term] * current.outputs + component;
        output_lower[out] = child_lower[in];
        output_upper[out] = child_upper[in];
      }
    return PHX_MC_OK;
  }
  if (current.kind == PHX_MC_CAD_NODE_REPARAMETERIZED) {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    namespace interval = phx::mc::remesh;
    const int64_t offset = model_.node_real_offsets[index];
    const interval::Interval first{workspace.coefficient_lower[offset],
                                   workspace.coefficient_upper[offset]};
    const interval::Interval last{workspace.coefficient_lower[offset + 1],
                                  workspace.coefficient_upper[offset + 1]};
    if (!interval::finite(first) || !interval::finite(last)) return PHX_MC_INVALID_ARGUMENT;
    const auto delta = interval::subtract(last, first);
    const bool reversed = current.integers[1] != 0;
    const interval::Interval t{input_lower[0], input_upper[0]};
    const auto fraction = reversed ? interval::subtract(interval::point(1.0), t) : t;
    const auto parameter = interval::add(first, interval::multiply(fraction, delta));
    if (!interval::finite(parameter)) return PHX_MC_UNRESOLVED;
    double child_lower[9], child_upper[9], result_lower[9], result_upper[9];
    const int32_t status = interval_jets(static_cast<int32_t>(current.integers[0]),
                                         &parameter.lower, &parameter.upper, order, child_lower,
                                         child_upper, workspace);
    if (status != PHX_MC_OK) return status;
    const interval::Interval slope = reversed ? interval::Interval{-delta.upper, -delta.lower}
                                               : delta;
    const auto squared = interval::multiply(slope, slope);
    for (int term = 0; term <= order; ++term) {
      const auto factor = term == 0 ? interval::point(1.0) : term == 1 ? slope : squared;
      for (int component = 0; component < current.outputs; ++component) {
        const int k = term * current.outputs + component;
        const auto value = interval::multiply({child_lower[k], child_upper[k]}, factor);
        if (!interval::finite(value)) return PHX_MC_UNRESOLVED;
        result_lower[k] = value.lower;
        result_upper[k] = value.upper;
      }
    }
    for (int i = 0; i < (order + 1) * current.outputs; ++i) {
      output_lower[i] = result_lower[i];
      output_upper[i] = result_upper[i];
    }
    return PHX_MC_OK;
  }
  if (current.kind == PHX_MC_CAD_NODE_AFFINE) {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    double child_lower[18], child_upper[18], result_lower[18], result_upper[18];
    const int32_t status = interval_jets(static_cast<int32_t>(current.integers[0]), input_lower,
                                         input_upper, order, child_lower, child_upper, workspace);
    if (status != PHX_MC_OK) return status;
    const int dimensions = current.outputs;
    const int terms = current.inputs == 2 ? (order + 1) * (order + 2) / 2 : order + 1;
    const int64_t offset = model_.node_real_offsets[index];
    for (int64_t i = 0; i < current.real_count; ++i)
      if (!std::isfinite(workspace.coefficient_lower[offset + i]) ||
          !std::isfinite(workspace.coefficient_upper[offset + i]) ||
          workspace.coefficient_lower[offset + i] > workspace.coefficient_upper[offset + i])
        return PHX_MC_INVALID_ARGUMENT;
    for (int term = 0; term < terms; ++term) {
      for (int row = 0; row < dimensions; ++row) {
        using namespace phx::mc::remesh;
        Interval value = point(0.0);
        if (term == 0) {
          const int64_t k = offset + dimensions * dimensions + row;
          value = {workspace.coefficient_lower[k], workspace.coefficient_upper[k]};
        }
        for (int col = 0; col < dimensions; ++col) {
          const int64_t k = offset + row * dimensions + col;
          value = phx::mc::remesh::add(
              value, phx::mc::remesh::multiply(
                         {workspace.coefficient_lower[k], workspace.coefficient_upper[k]},
                         {child_lower[term * dimensions + col],
                          child_upper[term * dimensions + col]}));
        }
        if (!finite(value)) return PHX_MC_UNRESOLVED;
        result_lower[term * dimensions + row] = value.lower;
        result_upper[term * dimensions + row] = value.upper;
      }
    }
    for (int i = 0; i < terms * dimensions; ++i) {
      output_lower[i] = result_lower[i];
      output_upper[i] = result_upper[i];
    }
    return PHX_MC_OK;
  }
  if (current.kind != PHX_MC_CAD_NODE_TEMPLATE) return PHX_MC_UNRESOLVED;
  if (workspace.direct_template)
    return interval_template_jets(*this, index, input_lower, input_upper, order,
                                  output_lower, output_upper, workspace);
  const TemplateDefinition& definition = *current.definition;
  const ProgramData& data = order == 0 ? definition.value
                            : order == 1 ? definition.tangent : definition.hessian;
  if (workspace.bound_program[order] != &data) {
    workspace.bound_program[order] = nullptr;
    const int32_t status = workspace.program[order].bind(
        data.instructions, data.instruction_count, data.indices, data.index_count,
        data.parameters, data.parameter_count, data.workspace_size);
    if (status != PHX_MC_OK) return status;
    workspace.bound_program[order] = &data;
  }
  workspace.lower.assign(data.seeds, data.seeds + data.workspace_size);
  workspace.upper.assign(data.seeds, data.seeds + data.workspace_size);
  for (int i = 0; i < current.inputs; ++i) {
    workspace.lower[static_cast<std::size_t>(i)] = input_lower[i];
    workspace.upper[static_cast<std::size_t>(i)] = input_upper[i];
  }
  const int64_t offset = model_.node_real_offsets[index];
  for (int i = 0; i < definition.coefficient_count; ++i) {
    const double lo = workspace.coefficient_lower[offset + i];
    const double hi = workspace.coefficient_upper[offset + i];
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi) return PHX_MC_INVALID_ARGUMENT;
    const auto slot = static_cast<std::size_t>(data.coefficients[i]);
    workspace.lower[slot] = lo;
    workspace.upper[slot] = hi;
  }
  const int32_t status = workspace.program[order].execute(workspace.lower.data(), workspace.upper.data(),
                                                    data.workspace_size);
  if (status != PHX_MC_OK) return status;
  for (int64_t i = 0; i < data.output_count; ++i) {
    const auto slot = static_cast<std::size_t>(data.outputs[i]);
    if (!std::isfinite(workspace.lower[slot]) || !std::isfinite(workspace.upper[slot]) ||
        workspace.lower[slot] > workspace.upper[slot])
      return PHX_MC_UNRESOLVED;
  }
  for (int64_t i = 0; i < data.output_count; ++i) {
    const auto slot = static_cast<std::size_t>(data.outputs[i]);
    output_lower[i] = workspace.lower[slot];
    output_upper[i] = workspace.upper[slot];
  }
  return PHX_MC_OK;
}

int32_t CadGeometry::validate_node(int32_t index) {
  const phx_mc_cad_model& m = model_;
  Node& node = nodes_[static_cast<std::size_t>(index)];
  const int64_t integer_begin = m.node_integer_offsets[index];
  const int64_t real_begin = m.node_real_offsets[index];
  node.kind = m.node_kinds[index];
  node.integers = m.node_integers + integer_begin;
  node.integer_count = m.node_integer_offsets[index + 1] - integer_begin;
  node.reals = m.node_reals + real_begin;
  node.real_count = m.node_real_offsets[index + 1] - real_begin;
  if (!finite(node.reals, node.real_count)) return PHX_MC_INVALID_INPUT;
  const int64_t* ints = node.integers;
  const int64_t ni = node.integer_count, nr = node.real_count;
  const auto child = [&](int64_t value, int inputs, int outputs) -> const Node* {
    if (value < 0 || value >= index) return nullptr;
    const Node& c = nodes_[static_cast<std::size_t>(value)];
    if ((inputs > 0 && c.inputs != inputs) || (outputs > 0 && c.outputs != outputs))
      return nullptr;
    return &c;
  };
  switch (node.kind) {
    case PHX_MC_CAD_NODE_TEMPLATE: {
      if (ni != 1 || ints[0] < 0 || ints[0] >= kTemplateCount) return PHX_MC_INVALID_INPUT;
      const TemplateDefinition& definition = kTemplates[ints[0]];
      if (nr != definition.coefficient_count) return PHX_MC_INVALID_INPUT;
      node.definition = &definition;
      node.inputs = definition.input_dimension;
      node.outputs = definition.output_dimension;
      node.value_seeds.assign(definition.value.seeds,
                              definition.value.seeds + definition.value.workspace_size);
      node.tangent_seeds.assign(definition.tangent.seeds,
                                definition.tangent.seeds + definition.tangent.workspace_size);
      node.hessian_seeds.assign(definition.hessian.seeds,
                                definition.hessian.seeds + definition.hessian.workspace_size);
      for (int32_t k = 0; k < definition.coefficient_count; ++k) {
        node.value_seeds[static_cast<std::size_t>(definition.value.coefficients[k])] =
            node.reals[k];
        node.tangent_seeds[static_cast<std::size_t>(definition.tangent.coefficients[k])] =
            node.reals[k];
        node.hessian_seeds[static_cast<std::size_t>(definition.hessian.coefficients[k])] =
            node.reals[k];
      }
      return PHX_MC_OK;
    }
    case PHX_MC_CAD_NODE_BSPLINE_CURVE: {
      if (ni != 3) return PHX_MC_INVALID_INPUT;
      const int64_t degree = ints[0], count = ints[1], dimension = ints[2];
      if (degree < 0 || degree > kMaxDegree || count <= degree ||
          count > std::numeric_limits<int32_t>::max() / 4 || (dimension != 2 && dimension != 3) ||
          nr != (count + degree + 1) + count * dimension + count)
        return PHX_MC_INVALID_INPUT;
      const double* knots = node.reals;
      const double* weights = node.reals + (count + degree + 1) + count * dimension;
      if (!nondecreasing(knots, count + degree + 1) || !(knots[count] > knots[degree]))
        return PHX_MC_INVALID_INPUT;
      for (int64_t k = 0; k < count; ++k)
        if (!(weights[k] > 0.0)) return PHX_MC_INVALID_INPUT;
      node.inputs = 1;
      node.outputs = static_cast<int32_t>(dimension);
      return PHX_MC_OK;
    }
    case PHX_MC_CAD_NODE_BSPLINE_SURFACE: {
      if (ni != 4) return PHX_MC_INVALID_INPUT;
      const int64_t p = ints[0], q = ints[1], nu = ints[2], nv = ints[3];
      if (p < 0 || q < 0 || p > kMaxDegree || q > kMaxDegree || nu <= p || nv <= q ||
          nu > (int64_t{1} << 20) || nv > (int64_t{1} << 20) ||
          nr != (nu + p + 1) + (nv + q + 1) + 4 * nu * nv)
        return PHX_MC_INVALID_INPUT;
      const double* u_knots = node.reals;
      const double* v_knots = u_knots + nu + p + 1;
      const double* weights = v_knots + nv + q + 1 + 3 * nu * nv;
      if (!nondecreasing(u_knots, nu + p + 1) || !nondecreasing(v_knots, nv + q + 1) ||
          !(u_knots[nu] > u_knots[p]) || !(v_knots[nv] > v_knots[q]))
        return PHX_MC_INVALID_INPUT;
      for (int64_t k = 0; k < nu * nv; ++k)
        if (!(weights[k] > 0.0)) return PHX_MC_INVALID_INPUT;
      node.inputs = 2;
      node.outputs = 3;
      return PHX_MC_OK;
    }
    case PHX_MC_CAD_NODE_TSPLINE_SURFACE: {
      if (ni < 3) return PHX_MC_INVALID_INPUT;
      const int64_t pieces = ints[0], count = ints[1], width = ints[2];
      if (pieces < 1 || count < 1 || width < 1 || pieces > (int64_t{1} << 24) ||
          count > (int64_t{1} << 30) || width > 4096 || ni != 3 + pieces * width ||
          nr != 4 * pieces + 16 * pieces * width + 4 * count)
        return PHX_MC_INVALID_INPUT;
      for (int64_t k = 0; k < pieces * width; ++k)
        if (ints[3 + k] < -1 || ints[3 + k] >= count) return PHX_MC_INVALID_INPUT;
      const double* bounds = node.reals;
      const double* weights = node.reals + 4 * pieces + 16 * pieces * width + 3 * count;
      for (int64_t k = 0; k < count; ++k)
        if (!(weights[k] > 0.0)) return PHX_MC_INVALID_INPUT;
      for (int64_t piece = 0; piece < pieces; ++piece) {
        const double* b = bounds + 4 * piece;
        if (!(b[0] < b[1]) || !(b[2] < b[3])) return PHX_MC_INVALID_INPUT;
        node.u_breaks.push_back(b[0]), node.u_breaks.push_back(b[1]);
        node.v_breaks.push_back(b[2]), node.v_breaks.push_back(b[3]);
      }
      for (std::vector<double>* breaks : {&node.u_breaks, &node.v_breaks}) {
        std::sort(breaks->begin(), breaks->end());
        breaks->erase(std::unique(breaks->begin(), breaks->end()), breaks->end());
      }
      const std::size_t cells_u = node.u_breaks.size() - 1, cells_v = node.v_breaks.size() - 1;
      node.cell_pieces.assign(cells_u * cells_v, -1);
      for (int64_t piece = 0; piece < pieces; ++piece) {
        const double* b = bounds + 4 * piece;
        const auto index = [](const std::vector<double>& breaks, double value) {
          return static_cast<std::size_t>(
              std::lower_bound(breaks.begin(), breaks.end(), value) - breaks.begin());
        };
        for (std::size_t i = index(node.u_breaks, b[0]); i < index(node.u_breaks, b[1]); ++i)
          for (std::size_t j = index(node.v_breaks, b[2]); j < index(node.v_breaks, b[3]); ++j)
            if (node.cell_pieces[i * cells_v + j] < 0)
              node.cell_pieces[i * cells_v + j] = static_cast<int32_t>(piece);
      }
      node.inputs = 2;
      node.outputs = 3;
      return PHX_MC_OK;
    }
    case PHX_MC_CAD_NODE_AFFINE: {
      if (ni != 2) return PHX_MC_INVALID_INPUT;
      const Node* source = child(ints[0], 0, static_cast<int>(ints[1]));
      if (source == nullptr || (ints[1] != 2 && ints[1] != 3) || nr != ints[1] * ints[1] + ints[1])
        return PHX_MC_INVALID_INPUT;
      node.inputs = source->inputs;
      node.outputs = source->outputs;
      return PHX_MC_OK;
    }
    case PHX_MC_CAD_NODE_REPARAMETERIZED: {
      const Node* source = ni == 2 ? child(ints[0], 1, 0) : nullptr;
      if (source == nullptr || (ints[1] != 0 && ints[1] != 1) || nr != 2)
        return PHX_MC_INVALID_INPUT;
      node.inputs = 1;
      node.outputs = source->outputs;
      return PHX_MC_OK;
    }
    case PHX_MC_CAD_NODE_OFFSET_SURFACE:
      if (ni != 1 || child(ints[0], 2, 3) == nullptr || nr != 1) return PHX_MC_INVALID_INPUT;
      node.inputs = 2;
      node.outputs = 3;
      return PHX_MC_OK;
    case PHX_MC_CAD_NODE_OFFSET_CURVE: {
      const Node* source = ni == 2 ? child(ints[0], 1, 0) : nullptr;
      if (source == nullptr || (ints[1] != 0 && ints[1] != 1) ||
          (ints[1] == 0) != (source->outputs == 2) || nr != (ints[1] ? 4 : 1))
        return PHX_MC_INVALID_INPUT;
      node.inputs = 1;
      node.outputs = source->outputs;
      return PHX_MC_OK;
    }
    case PHX_MC_CAD_NODE_RULED:
      if (ni != 2 || child(ints[0], 1, 3) == nullptr || child(ints[1], 1, 3) == nullptr ||
          nr != 0)
        return PHX_MC_INVALID_INPUT;
      node.inputs = 2;
      node.outputs = 3;
      return PHX_MC_OK;
    case PHX_MC_CAD_NODE_EXTRUSION:
    case PHX_MC_CAD_NODE_REVOLUTION:
      if (ni != 1 || child(ints[0], 1, 3) == nullptr ||
          nr != (node.kind == PHX_MC_CAD_NODE_EXTRUSION ? 3 : 6))
        return PHX_MC_INVALID_INPUT;
      node.inputs = 2;
      node.outputs = 3;
      return PHX_MC_OK;
    case PHX_MC_CAD_NODE_SURFACE_MAPPED:
      if (ni != 2 || child(ints[0], 2, 3) == nullptr || child(ints[1], 1, 2) == nullptr ||
          nr != 0)
        return PHX_MC_INVALID_INPUT;
      node.inputs = 1;
      node.outputs = 3;
      return PHX_MC_OK;
    case PHX_MC_CAD_NODE_ISOPARAMETRIC:
      if (ni != 2 || child(ints[0], 2, 3) == nullptr || (ints[1] != 0 && ints[1] != 1) ||
          nr != 1)
        return PHX_MC_INVALID_INPUT;
      node.inputs = 1;
      node.outputs = 3;
      return PHX_MC_OK;
    case PHX_MC_CAD_NODE_INTERSECTION: {
      if (ni < 7) return PHX_MC_INVALID_INPUT;
      const int64_t charts = ints[2], output = ints[3], fixed = ints[4], terminal = ints[5];
      if (child(ints[0], 2, 3) == nullptr || child(ints[1], 2, 3) == nullptr || charts < 1 ||
          charts > (int64_t{1} << 24) || ni != 7 + charts || nr != 2 + kChartReals * charts ||
          output < 0 || output > 2 || fixed < -1 || fixed >= charts || terminal < -1 ||
          terminal >= charts || (ints[6] != 0 && ints[6] != 1))
        return PHX_MC_INVALID_INPUT;
      for (int64_t chart = 0; chart < charts; ++chart)
        if (ints[7 + chart] < 0 || ints[7 + chart] > 3) return PHX_MC_INVALID_INPUT;
      node.inputs = 1;
      node.outputs = output == 0 ? 3 : 2;
      return PHX_MC_OK;
    }
    default:
      return PHX_MC_INVALID_INPUT;
  }
}

int32_t CadGeometry::bind(const phx_mc_cad_model& m, Evaluators evaluators) {
  static const bool kTemplatesValid = templates_valid();
  if (!kTemplatesValid) return PHX_MC_INTERNAL_ERROR;
  model_ = m;
  nodes_.clear();
  const auto present = [](int64_t count, const void* pointer) {
    return count == 0 || pointer != nullptr;
  };
  if (m.node_count < 0 || m.node_count > std::numeric_limits<int32_t>::max() ||
      m.corner_count < 0 || m.surface_count < 0 || m.curve_count < 0 || m.loop_count < 0 ||
      m.use_count < 0 || m.region_count < 0 || m.node_integer_offsets == nullptr ||
      m.node_real_offsets == nullptr || !present(m.node_count, m.node_kinds) ||
      !present(m.corner_count, m.corner_points) ||
      !present(m.surface_count, m.surface_nodes) ||
      !present(m.surface_count, m.surface_orientations) ||
      !present(m.surface_count, m.surface_periods) ||
      !present(m.surface_count, m.surface_parameter_domains) ||
      !present(m.curve_count, m.curve_nodes) || !present(m.curve_count, m.curve_corners) ||
      !present(m.curve_count, m.curve_ranges) || m.loop_use_offsets == nullptr ||
      !present(m.loop_count, m.loop_surfaces) || !present(m.use_count, m.use_kinds) ||
      !present(m.use_count, m.use_entities) || !present(m.use_count, m.use_senses) ||
      !present(m.use_count, m.use_ranges) || !present(m.use_count, m.use_pcurve_nodes) ||
      m.region_surface_offsets == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  if (m.node_integer_offsets[0] != 0 || m.node_real_offsets[0] != 0)
    return PHX_MC_INVALID_INPUT;
  for (int64_t k = 0; k < m.node_count; ++k)
    if (m.node_integer_offsets[k + 1] < m.node_integer_offsets[k] ||
        m.node_real_offsets[k + 1] < m.node_real_offsets[k])
      return PHX_MC_INVALID_INPUT;
  if ((m.node_integer_offsets[m.node_count] > 0 && m.node_integers == nullptr) ||
      (m.node_real_offsets[m.node_count] > 0 && m.node_reals == nullptr))
    return PHX_MC_INVALID_ARGUMENT;
  nodes_.resize(static_cast<std::size_t>(m.node_count));
  for (int64_t k = 0; k < m.node_count; ++k) {
    const int32_t status = validate_node(static_cast<int32_t>(k));
    if (status != PHX_MC_OK) return status;
    if (evaluators == Evaluators::kClosedForms)
      bind_closed_form(nodes_[static_cast<std::size_t>(k)]);
  }
  const auto node_with = [&](int32_t index, int inputs, int outputs) {
    return index >= 0 && index < m.node_count && node(index).inputs == inputs &&
           node(index).outputs == outputs;
  };
  if (!finite(m.corner_points, 3 * m.corner_count)) return PHX_MC_INVALID_INPUT;
  for (int64_t f = 0; f < m.surface_count; ++f)
    if (!node_with(m.surface_nodes[f], 2, 3) ||
        (m.surface_orientations[f] != 1 && m.surface_orientations[f] != -1) ||
        !finite(m.surface_periods + 2 * f, 2))
      return PHX_MC_INVALID_INPUT;
  for (int64_t e = 0; e < m.curve_count; ++e)
    if (!node_with(m.curve_nodes[e], 1, 3) || m.curve_corners[2 * e] < 0 ||
        m.curve_corners[2 * e] >= m.corner_count || m.curve_corners[2 * e + 1] < 0 ||
        m.curve_corners[2 * e + 1] >= m.corner_count || !finite(m.curve_ranges + 2 * e, 2))
      return PHX_MC_INVALID_INPUT;
  if (m.loop_use_offsets[0] != 0 || m.loop_use_offsets[m.loop_count] != m.use_count)
    return PHX_MC_INVALID_INPUT;
  for (int64_t l = 0; l < m.loop_count; ++l)
    if (m.loop_use_offsets[l + 1] <= m.loop_use_offsets[l] || m.loop_surfaces[l] < 0 ||
        m.loop_surfaces[l] >= m.surface_count)
      return PHX_MC_INVALID_INPUT;
  for (int64_t u = 0; u < m.use_count; ++u) {
    const int64_t limit = m.use_kinds[u] == 0 ? m.curve_count : m.corner_count;
    if ((m.use_kinds[u] != 0 && m.use_kinds[u] != 1) || m.use_entities[u] < 0 ||
        m.use_entities[u] >= limit || (m.use_senses[u] != 1 && m.use_senses[u] != -1) ||
        !finite(m.use_ranges + 2 * u, 2) || !node_with(m.use_pcurve_nodes[u], 1, 2))
      return PHX_MC_INVALID_INPUT;
  }
  if (m.region_surface_offsets[0] != 0) return PHX_MC_INVALID_INPUT;
  for (int64_t r = 0; r < m.region_count; ++r)
    if (m.region_surface_offsets[r + 1] <= m.region_surface_offsets[r])
      return PHX_MC_INVALID_INPUT;
  const int64_t region_rows = m.region_surface_offsets[m.region_count];
  if (region_rows > 0 && (m.region_surfaces == nullptr || m.region_orientations == nullptr))
    return PHX_MC_INVALID_ARGUMENT;
  for (int64_t k = 0; k < region_rows; ++k)
    if (m.region_surfaces[k] < 0 || m.region_surfaces[k] >= m.surface_count ||
        (m.region_orientations[k] != 1 && m.region_orientations[k] != -1))
      return PHX_MC_INVALID_INPUT;
  return PHX_MC_OK;
}

void CadGeometry::bind_closed_form(Node& node) const {
  static const int32_t kPlane = template_index("plane"), kCylinder = template_index("cylinder"),
                       kCone = template_index("cone"), kSphere = template_index("sphere"),
                       kTorus = template_index("torus"), kLine = template_index("line3"),
                       kExactLine = template_index("exact_line3");
  double* data = node.closed_data;
  // Every closed form is affine in its frame: a point (data 0..2) and
  // `vectors` linear vectors after it, then the kind's scalars.
  const auto vectors = [](ClosedForm kind) { return kind == ClosedForm::kLineRevolution ? 5 : 3; };
  if (node.kind == PHX_MC_CAD_NODE_TEMPLATE) {
    const int64_t id = node.integers[0];
    // Template coefficients: origin or center, e1, e2, then axis and the
    // kind's scalars (cylinder and sphere R; cone r0, semi-angle; torus R, r).
    std::copy(node.reals, node.reals + node.real_count, data);
    if (id == kPlane) {
      node.closed = ClosedForm::kPlane;
    } else if (id == kCylinder) {
      node.closed = ClosedForm::kCylinder;
    } else if (id == kCone) {
      node.closed = ClosedForm::kCone;
      data[13] = std::tan(node.reals[13]);
    } else if (id == kSphere) {
      node.closed = ClosedForm::kSphere;
    } else if (id == kTorus) {
      node.closed = ClosedForm::kTorus;
    }
    return;
  }
  if (node.kind == PHX_MC_CAD_NODE_AFFINE && node.integers[1] == 3) {
    // A placement of a closed form is the closed form of the placed frame.
    const Node& child = this->node(static_cast<int32_t>(node.integers[0]));
    if (child.closed == ClosedForm::kNone) return;
    const double* matrix = node.reals;
    const double* offset = node.reals + 9;
    std::copy(child.closed_data, child.closed_data + 24, data);
    for (int slot = 0; slot <= vectors(child.closed); ++slot) {
      const double* x = child.closed_data + 3 * slot;
      for (int r = 0; r < 3; ++r)
        data[3 * slot + r] = matrix[3 * r] * x[0] + matrix[3 * r + 1] * x[1] +
                             matrix[3 * r + 2] * x[2] + (slot == 0 ? offset[r] : 0.0);
    }
    node.closed = child.closed;
    return;
  }
  if (node.kind != PHX_MC_CAD_NODE_REVOLUTION) return;
  const Node& profile = this->node(static_cast<int32_t>(node.integers[0]));
  if (profile.kind != PHX_MC_CAD_NODE_TEMPLATE ||
      (profile.integers[0] != kLine && profile.integers[0] != kExactLine))
    return;
  // Axis origin O, unit axis a, profile p(v) = p0 + v d (line; exact line
  // (1 - v) first + v last: p0 = first, d = last - first). The revolution
  // O + a (a.(p - O)) + r cos u + (a x r) sin u with r = (p - O) - a (a.(p - O))
  // is affine in v: P0 + v P1 + (R0 + v R1) cos u + (C0 + v C1) sin u.
  const double *origin = node.reals, *a = node.reals + 3, *p = profile.reals;
  double d[3], relative[3];
  for (int i = 0; i < 3; ++i) {
    d[i] = profile.integers[0] == kExactLine ? p[3 + i] - p[i] : p[3 + i];
    relative[i] = p[i] - origin[i];
  }
  const double dot = relative[0] * a[0] + relative[1] * a[1] + relative[2] * a[2];
  const double slope = d[0] * a[0] + d[1] * a[1] + d[2] * a[2];
  double *p0 = data, *p1 = data + 3, *r0 = data + 6, *r1 = data + 9;
  for (int i = 0; i < 3; ++i) {
    p0[i] = origin[i] + a[i] * dot;
    p1[i] = a[i] * slope;
    r0[i] = relative[i] - a[i] * dot;
    r1[i] = d[i] - a[i] * slope;
  }
  for (const double* r : {r0, r1}) {
    double* c = data + (r == r0 ? 12 : 15);
    c[0] = a[1] * r[2] - a[2] * r[1];
    c[1] = a[2] * r[0] - a[0] * r[2];
    c[2] = a[0] * r[1] - a[1] * r[0];
  }
  node.closed = ClosedForm::kLineRevolution;
}

namespace {

// Normalized derivatives cos^(n)(x) / n! and sin^(n)(x) / n!, n <= order.
void trigonometric_series(double x, int order, double* cosine, double* sine) noexcept {
  const double c = std::cos(x), s = std::sin(x);
  const double cosine_cycle[4] = {c, -s, -c, s}, sine_cycle[4] = {s, c, -s, -c};
  double factorial = 1.0;
  for (int n = 0; n <= order; ++n) {
    if (n > 1) factorial *= n;
    cosine[n] = cosine_cycle[n % 4] / factorial;
    sine[n] = sine_cycle[n % 4] / factorial;
  }
}

}  // namespace

// The source formulas of PlanePatch, CylinderPatch, ConePatch, SpherePatch,
// TorusPatch and RevolutionSurface over a line, in the source's operation
// order for the value; every Taylor coefficient is a product of a frame
// vector and normalized trigonometric derivatives.
bool CadGeometry::expand_closed(const Node& node, const double* point, int order,
                                Taylor* out) const {
  if (order > kMaxOrder) return false;
  const Truncation local{2, order};
  for (int i = 0; i < 3; ++i) clear(out[i], local);
  const double* data = node.closed_data;
  const double *origin = data, *e1 = data + 3, *e2 = data + 6, *axis = data + 9;
  const double u = point[0], v = point[1];
  double cu[kMaxOrder + 1], su[kMaxOrder + 1], cv[kMaxOrder + 1], sv[kMaxOrder + 1];
  switch (node.closed) {
    case ClosedForm::kPlane:
      for (int i = 0; i < 3; ++i) {
        out[i].c[0] = origin[i] + u * e1[i] + v * e2[i];
        if (order >= 1) out[i].c[term(1, 0)] = e1[i], out[i].c[term(0, 1)] = e2[i];
      }
      return true;
    case ClosedForm::kCylinder: {
      const double radius = data[12];
      trigonometric_series(u, order, cu, su);
      for (int i = 0; i < 3; ++i) {
        out[i].c[0] = origin[i] + radius * (cu[0] * e1[i] + su[0] * e2[i]) + v * axis[i];
        for (int n = 1; n <= order; ++n)
          out[i].c[term(n, 0)] = radius * (cu[n] * e1[i] + su[n] * e2[i]);
        if (order >= 1) out[i].c[term(0, 1)] = axis[i];
      }
      return true;
    }
    case ClosedForm::kCone: {
      const double slope = data[13], radius = data[12] + v * slope;
      trigonometric_series(u, order, cu, su);
      for (int i = 0; i < 3; ++i) {
        const double radial = cu[0] * e1[i] + su[0] * e2[i];
        out[i].c[0] = origin[i] + v * axis[i] + radius * radial;
        for (int n = 1; n <= order; ++n) {
          const double derivative = cu[n] * e1[i] + su[n] * e2[i];
          out[i].c[term(n, 0)] = radius * derivative;
          if (n + 1 <= order) out[i].c[term(n, 1)] = slope * derivative;
        }
        if (order >= 1) out[i].c[term(0, 1)] = axis[i] + slope * radial;
      }
      return true;
    }
    case ClosedForm::kSphere: {
      const double radius = data[12];
      trigonometric_series(u, order, cu, su);
      trigonometric_series(v, order, cv, sv);
      for (int i = 0; i < 3; ++i) {
        const double equatorial = cu[0] * e1[i] + su[0] * e2[i];
        out[i].c[0] = origin[i] + radius * (cv[0] * equatorial + sv[0] * axis[i]);
        for (int n = 1; n <= order; ++n)
          for (int j = 0; j <= n; ++j) {
            const double turned = cu[n - j] * e1[i] + su[n - j] * e2[i];
            out[i].c[term(n - j, j)] =
                radius * (n == j ? cv[j] * turned + sv[j] * axis[i] : cv[j] * turned);
          }
      }
      return true;
    }
    case ClosedForm::kTorus: {
      const double major = data[12], minor = data[13];
      trigonometric_series(u, order, cu, su);
      trigonometric_series(v, order, cv, sv);
      const double ring = major + minor * cv[0];
      for (int i = 0; i < 3; ++i) {
        out[i].c[0] = origin[i] + ring * (cu[0] * e1[i] + su[0] * e2[i]) +
                      minor * sv[0] * axis[i];
        for (int n = 1; n <= order; ++n)
          for (int j = 0; j <= n; ++j) {
            const double radial = cu[n - j] * e1[i] + su[n - j] * e2[i];
            double value = (j == 0 ? ring : minor * cv[j]) * radial;
            if (n == j) value += minor * sv[j] * axis[i];
            out[i].c[term(n - j, j)] = value;
          }
      }
      return true;
    }
    case ClosedForm::kLineRevolution: {
      const double *p1 = data + 3, *r0 = data + 6, *r1 = data + 9, *c0 = data + 12,
                   *c1 = data + 15;
      trigonometric_series(u, order, cu, su);
      for (int i = 0; i < 3; ++i) {
        const double radial = r0[i] + v * r1[i], cross = c0[i] + v * c1[i];
        out[i].c[0] = origin[i] + v * p1[i] + radial * cu[0] + cross * su[0];
        for (int n = 1; n <= order; ++n) {
          out[i].c[term(n, 0)] = radial * cu[n] + cross * su[n];
          if (n + 1 <= order) out[i].c[term(n, 1)] = r1[i] * cu[n] + c1[i] * su[n];
        }
        if (order >= 1) out[i].c[term(0, 1)] = p1[i] + r1[i] * cu[0] + c1[i] * su[0];
      }
      return true;
    }
    case ClosedForm::kNone:
      return false;
  }
  return false;
}

bool CadGeometry::expand_template(const Node& node, const double* point, int order, Taylor* out,
                                  EvaluationWorkspace& workspace) const {
  const TemplateDefinition& definition = *node.definition;
  const int k = definition.input_dimension, d = definition.output_dimension;
  const Truncation local{k, order};
  for (int i = 0; i < d; ++i) clear(out[i], local);
  if (order == 0) {
    double values[3];
    evaluate_point(view(definition.value, node.value_seeds, k), point, 1, values,
                   workspace.program);
    for (int i = 0; i < d; ++i) out[i].c[0] = values[i];
    return true;
  }
  if (order > kMaxOrder) return false;
  const int64_t components = jet_components(k);
  // Order 3 needs the jets of the first partials, order 4 those of the second.
  const ProgramData& data =
      order == 4 ? definition.hessian : order == 3 ? definition.tangent : definition.value;
  const std::vector<double>& seeds =
      order == 4 ? node.hessian_seeds : order == 3 ? node.tangent_seeds : node.value_seeds;
  workspace.jets.resize(static_cast<std::size_t>(data.output_count * components));
  double* jet = workspace.jets.data();
  evaluate_jet(view(data, seeds, k), point, 1, jet, workspace.program);
  // Output rows: value d, first partials k d, second partials (pairs) d.
  const auto row = [&](int block, int i) { return jet + (block * d + i) * components; };
  for (int i = 0; i < d; ++i) {
    const double* f = row(0, i);
    out[i].c[0] = f[0];
    out[i].c[term(1, 0)] = f[1];
    if (k == 2) out[i].c[term(0, 1)] = f[2];
    if (order >= 2) {
      if (k == 1) {
        out[i].c[term(2, 0)] = 0.5 * f[2];
      } else {
        out[i].c[term(2, 0)] = 0.5 * f[3];
        out[i].c[term(1, 1)] = f[4];
        out[i].c[term(0, 2)] = 0.5 * f[5];
      }
    }
    if (order == 3) {
      // Hessians of the first partials.
      const double* fu = row(1, i);
      if (k == 1) {
        out[i].c[term(3, 0)] = fu[2] / 6.0;
      } else {
        const double* fv = row(2, i);
        out[i].c[term(3, 0)] = fu[3] / 6.0;
        out[i].c[term(2, 1)] = 0.5 * fu[4];
        out[i].c[term(1, 2)] = 0.5 * fu[5];
        out[i].c[term(0, 3)] = fv[5] / 6.0;
      }
    } else if (order == 4) {
      // Gradients and Hessians of the second partials.
      if (k == 1) {
        const double* fuu = row(2, i);
        out[i].c[term(3, 0)] = fuu[1] / 6.0;
        out[i].c[term(4, 0)] = fuu[2] / 24.0;
      } else {
        const double *fuu = row(3, i), *fuv = row(4, i), *fvv = row(5, i);
        out[i].c[term(3, 0)] = fuu[1] / 6.0;
        out[i].c[term(2, 1)] = 0.5 * fuu[2];
        out[i].c[term(1, 2)] = 0.5 * fvv[1];
        out[i].c[term(0, 3)] = fvv[2] / 6.0;
        out[i].c[term(4, 0)] = fuu[3] / 24.0;
        out[i].c[term(3, 1)] = fuu[4] / 6.0;
        out[i].c[term(2, 2)] = 0.25 * fuu[5];
        out[i].c[term(1, 3)] = fuv[5] / 6.0;
        out[i].c[term(0, 4)] = fvv[5] / 24.0;
      }
    }
  }
  return true;
}

bool CadGeometry::expand_bspline_curve(const Node& node, double t, int order,
                                       int span_override, Taylor* out) const {
  const int degree = static_cast<int>(node.integers[0]);
  const int count = static_cast<int>(node.integers[1]);
  const int dimension = static_cast<int>(node.integers[2]);
  const double* knots = node.reals;
  const double* points = knots + count + degree + 1;
  const double* weights = points + count * dimension;
  if (!(t >= knots[degree] && t <= knots[count])) return false;
  const int span = span_override >= 0 ? span_override : knot_span(knots, degree, count, t);
  if (span_override >= 0 && !(t >= knots[span] && t <= knots[span + 1])) return false;
  double ders[kMaxOrder + 1][kMaxDegree + 1];
  basis_derivatives(knots, span, degree, t, order, ders);
  const Truncation local{1, order};
  Taylor numerator[3], denominator = constant(0.0, local);
  for (int i = 0; i < dimension; ++i) clear(numerator[i], local);
  for (int n = 0; n <= order; ++n) {
    const double scale = 1.0 / factorial(n);
    for (int j = 0; j <= degree; ++j) {
      const int row = span - degree + j;
      const double weighted = ders[n][j] * weights[row] * scale;
      denominator.c[term(n, 0)] += weighted;
      for (int i = 0; i < dimension; ++i)
        numerator[i].c[term(n, 0)] += weighted * points[row * dimension + i];
    }
  }
  Taylor inverse;
  reciprocal(denominator, inverse, local);
  for (int i = 0; i < dimension; ++i) multiply(numerator[i], inverse, out[i], local);
  return true;
}

bool CadGeometry::expand_bspline_surface(const Node& node, const double* point, int order,
                                         Taylor* out) const {
  const int p = static_cast<int>(node.integers[0]), q = static_cast<int>(node.integers[1]);
  const int nu = static_cast<int>(node.integers[2]), nv = static_cast<int>(node.integers[3]);
  const double* u_knots = node.reals;
  const double* v_knots = u_knots + nu + p + 1;
  const double* points = v_knots + nv + q + 1;
  const double* weights = points + 3 * nu * nv;
  // Clamped to the active domain, as the source's clipped evaluation: the
  // clamped axis carries no derivative.
  const double u = std::clamp(point[0], u_knots[p], u_knots[nu]);
  const double v = std::clamp(point[1], v_knots[q], v_knots[nv]);
  const bool u_clamped = u != point[0], v_clamped = v != point[1];
  const int u_span = knot_span(u_knots, p, nu, u), v_span = knot_span(v_knots, q, nv, v);
  double u_ders[kMaxOrder + 1][kMaxDegree + 1], v_ders[kMaxOrder + 1][kMaxDegree + 1];
  basis_derivatives(u_knots, u_span, p, u, order, u_ders);
  basis_derivatives(v_knots, v_span, q, v, order, v_ders);
  const Truncation local{2, order};
  Taylor numerator[3], denominator = constant(0.0, local);
  for (int i = 0; i < 3; ++i) clear(numerator[i], local);
  for (int n = 0; n <= order; ++n)
    for (int dv = 0; dv <= n; ++dv) {
      const int du = n - dv;
      if ((du > 0 && u_clamped) || (dv > 0 && v_clamped)) continue;
      const double scale = 1.0 / (factorial(du) * factorial(dv));
      const int index = term(du, dv);
      for (int a = 0; a <= p; ++a) {
        const int row = u_span - p + a;
        for (int b = 0; b <= q; ++b) {
          const int column = v_span - q + b;
          const int64_t control = static_cast<int64_t>(row) * nv + column;
          const double weighted = u_ders[du][a] * v_ders[dv][b] * weights[control] * scale;
          denominator.c[index] += weighted;
          for (int i = 0; i < 3; ++i) numerator[i].c[index] += weighted * points[3 * control + i];
        }
      }
    }
  Taylor inverse;
  reciprocal(denominator, inverse, local);
  for (int i = 0; i < 3; ++i) multiply(numerator[i], inverse, out[i], local);
  return true;
}

bool CadGeometry::expand_tspline_surface(const Node& node, const double* point, int order,
                                         Taylor* out) const {
  const int64_t pieces = node.integers[0], count = node.integers[1], width = node.integers[2];
  const int64_t* rows = node.integers + 3;
  const double* bounds = node.reals;
  const double* operators = bounds + 4 * pieces;
  const double* points = operators + 16 * pieces * width;
  const double* weights = points + 3 * count;
  // Candidate grid cells whose closure holds the point; the first covering piece wins.
  const auto candidates = [](const std::vector<double>& breaks, double value, std::size_t* out) {
    if (!(value >= breaks.front() && value <= breaks.back())) return 0;
    const std::size_t upper = static_cast<std::size_t>(
        std::upper_bound(breaks.begin(), breaks.end(), value) - breaks.begin());
    int found = 0;
    if (upper < breaks.size()) out[found++] = upper - 1;  // breaks[upper - 1] <= value
    if (breaks[upper - 1] == value && upper >= 2) out[found++] = upper - 2;
    return found;
  };
  std::size_t u_cells[2], v_cells[2];
  const int u_found = candidates(node.u_breaks, point[0], u_cells);
  const int v_found = candidates(node.v_breaks, point[1], v_cells);
  int64_t piece = -1;
  const std::size_t cells_v = node.v_breaks.size() - 1;
  for (int a = 0; a < u_found; ++a)
    for (int b = 0; b < v_found; ++b) {
      const int32_t owner = node.cell_pieces[u_cells[a] * cells_v + v_cells[b]];
      if (owner >= 0 && (piece < 0 || owner < piece)) piece = owner;
    }
  if (piece < 0) return false;
  const double* b = bounds + 4 * piece;
  const double u_extent = b[1] - b[0], v_extent = b[3] - b[2];
  const double s = (point[0] - b[0]) / u_extent, t = (point[1] - b[2]) / v_extent;
  const Truncation local{2, order};
  Taylor numerator[3], denominator = constant(0.0, local);
  for (int i = 0; i < 3; ++i) clear(numerator[i], local);
  const int64_t* piece_rows = rows + piece * width;
  const double* piece_operator = operators + 16 * piece * width;
  for (int n = 0; n <= order; ++n)
    for (int dv = 0; dv <= n; ++dv) {
      const int du = n - dv;
      if (du > 3 || dv > 3) continue;
      double bu[4], bv[4];
      bernstein_cubic(s, du, bu);
      bernstein_cubic(t, dv, bv);
      const double width_scale = std::pow(u_extent, -du) * std::pow(v_extent, -dv);
      const double scale = 1.0 / (factorial(du) * factorial(dv));
      const int index = term(du, dv);
      for (int64_t l = 0; l < width; ++l) {
        const int64_t row = piece_rows[l];
        if (row < 0) continue;
        double basis = 0.0;
        for (int a = 0; a < 4; ++a)
          for (int c = 0; c < 4; ++c) basis += bu[a] * bv[c] * piece_operator[16 * l + 4 * a + c];
        basis = basis * width_scale * scale;
        denominator.c[index] += basis * weights[row];
        for (int i = 0; i < 3; ++i)
          numerator[i].c[index] += basis * (points[3 * row + i] * weights[row]);
      }
    }
  Taylor inverse;
  reciprocal(denominator, inverse, local);
  for (int i = 0; i < 3; ++i) multiply(numerator[i], inverse, out[i], local);
  return true;
}

bool CadGeometry::expand_offset_surface(const Node& node, const double* point, int order,
                                        Taylor* out, EvaluationWorkspace& workspace) const {
  if (order + 1 > kMaxOrder) return false;
  Taylor base[3];
  if (!expand(static_cast<int32_t>(node.integers[0]), point, order + 1, base, workspace))
    return false;
  const Truncation local{2, order}, wide{2, order + 1};
  Taylor su[3], sv[3];
  for (int i = 0; i < 3; ++i) {
    differentiate(base[i], 0, su[i], wide);
    differentiate(base[i], 1, sv[i], wide);
  }
  Taylor normal[3], first, second, squared = constant(0.0, local);
  for (int i = 0; i < 3; ++i) {
    multiply(su[(i + 1) % 3], sv[(i + 2) % 3], first, local);
    multiply(su[(i + 2) % 3], sv[(i + 1) % 3], second, local);
    axpy(first, -1.0, second, normal[i], local);
    multiply(normal[i], normal[i], first, local);
    axpy(squared, 1.0, first, squared, local);
  }
  if (!(squared.c[0] > 0.0)) return false;
  Taylor length, inverse, unit;
  square_root(squared, length, local);
  reciprocal(length, inverse, local);
  for (int i = 0; i < 3; ++i) {
    multiply(normal[i], inverse, unit, local);
    axpy(base[i], node.reals[0], unit, out[i], local);
  }
  return true;
}

bool CadGeometry::expand_offset_curve(const Node& node, double t, int order, Taylor* out,
                                      EvaluationWorkspace& workspace) const {
  if (order + 1 > kMaxOrder) return false;
  const Node& base_node = this->node(static_cast<int32_t>(node.integers[0]));
  const int dimension = base_node.outputs;
  Taylor base[3];
  if (!expand(static_cast<int32_t>(node.integers[0]), &t, order + 1, base, workspace))
    return false;
  const Truncation local{1, order}, wide{1, order + 1};
  Taylor tangent[3]{}, normal[3];
  for (int i = 0; i < dimension; ++i) differentiate(base[i], 0, tangent[i], wide);
  if (dimension == 2) {
    normal[0] = tangent[1];
    scale(tangent[0], -1.0, normal[1], local);
  } else {
    const double* direction = node.reals + 1;
    for (int i = 0; i < 3; ++i) {
      Taylor term_one, term_two;
      scale(tangent[(i + 1) % 3], direction[(i + 2) % 3], term_one, local);
      scale(tangent[(i + 2) % 3], direction[(i + 1) % 3], term_two, local);
      axpy(term_one, -1.0, term_two, normal[i], local);
    }
  }
  Taylor squared = constant(0.0, local), product;
  for (int i = 0; i < dimension; ++i) {
    multiply(normal[i], normal[i], product, local);
    axpy(squared, 1.0, product, squared, local);
  }
  if (!(squared.c[0] > 0.0)) return false;
  Taylor length, inverse, unit;
  square_root(squared, length, local);
  reciprocal(length, inverse, local);
  for (int i = 0; i < dimension; ++i) {
    multiply(normal[i], inverse, unit, local);
    axpy(base[i], node.reals[0], unit, out[i], local);
  }
  return true;
}

bool CadGeometry::expand_intersection(const Node& node, double t, int order, Taylor* out,
                                      EvaluationWorkspace& workspace) const {
  const int32_t first = static_cast<int32_t>(node.integers[0]);
  const int32_t second = static_cast<int32_t>(node.integers[1]);
  const int64_t charts = node.integers[2], output = node.integers[3];
  const int64_t fixed = node.integers[4], terminal = node.integers[5];
  const bool reversed = node.integers[6] != 0;
  const double carrier = reversed ? node.reals[0] - t : t;
  int64_t chart = fixed;
  if (chart < 0) {
    if (!(carrier >= 0.0 && carrier <= static_cast<double>(charts))) return false;
    chart = std::min(static_cast<int64_t>(std::floor(carrier)), charts - 1);
    if (terminal >= 0 && carrier >= node.reals[1]) chart = terminal;
  }
  if (workspace.span_node == &node) chart = workspace.span_chart;
  const int axis = static_cast<int>(node.integers[7 + chart]);
  const double* data = node.reals + 2 + kChartReals * chart;
  const double *start = data, *end = data + 4, *lower = data + 8, *upper = data + 12;
  const double* preconditioner = data + 16;
  int free[3];
  for (int k = 0, f = 0; k < 4; ++k)
    if (k != axis) free[f++] = k;
  const double fraction = carrier - static_cast<double>(chart);
  double q[4];
  const auto interpolate = [&]() {
    for (int k = 0; k < 4; ++k) q[k] = start[k] + fraction * (end[k] - start[k]);
  };
  Taylor value[3], value_two[3];
  const auto chart_step = [&]() {
    if (!expand(first, q, 0, value, workspace) || !expand(second, q + 2, 0, value_two, workspace))
      return false;
    double residual[3];
    for (int i = 0; i < 3; ++i) residual[i] = value[i].c[0] - value_two[i].c[0];
    for (int f = 0; f < 3; ++f) {
      const double* row = preconditioner + 3 * f;
      const double correction = row[0] * residual[0] + row[1] * residual[1] + row[2] * residual[2];
      q[free[f]] = std::clamp(q[free[f]] - correction, lower[free[f]], upper[free[f]]);
    }
    return true;
  };
  // Free-column Jacobian of S1(q01) - S2(q23) from first-order expansions.
  const auto free_jacobian = [&](const Taylor* one, const Taylor* two, double* jacobian) {
    for (int i = 0; i < 3; ++i)
      for (int f = 0; f < 3; ++f) {
        const int k = free[f];
        const Taylor& source = k < 2 ? one[i] : two[i];
        const double derivative = source.c[(k % 2) == 0 ? term(1, 0) : term(0, 1)];
        jacobian[3 * i + f] = k < 2 ? derivative : -derivative;
      }
  };
  interpolate();
  for (int step = 0; step < kChartSteps; ++step)
    if (!chart_step()) return false;
  bool settled = false;
  for (int iteration = 0; iteration < kNewtonIterations && !settled; ++iteration) {
    if (!expand(first, q, 1, value, workspace) || !expand(second, q + 2, 1, value_two, workspace))
      return false;
    double residual[3], correction[3], jacobian[9];
    for (int i = 0; i < 3; ++i) residual[i] = value[i].c[0] - value_two[i].c[0];
    free_jacobian(value, value_two, jacobian);
    if (!solve3(jacobian, residual, correction)) break;
    settled = true;
    for (int f = 0; f < 3; ++f) {
      const int k = free[f];
      const double scale = std::max(1.0, std::fabs(q[k]));
      if (!(std::fabs(correction[f]) <= kSettled * std::numeric_limits<double>::epsilon() * scale))
        settled = false;
      q[k] = std::clamp(q[k] - correction[f], lower[k], upper[k]);
    }
  }
  if (!settled) {
    interpolate();
    for (int step = 0; step < kChartIterations; ++step)
      if (!chart_step()) return false;
  }
  const Truncation line{1, order};
  // q(carrier): the chart coordinate moves linearly; the free ones follow
  // the root by Newton steps on the expansion, one order per step.
  Taylor coordinates[4];
  for (int k = 0; k < 4; ++k) coordinates[k] = constant(q[k], line);
  Taylor local_first[3], local_second[3];
  if (order >= 1) {
    coordinates[axis].c[term(1, 0)] = end[axis] - start[axis];
    if (!expand(first, q, order, local_first, workspace) ||
        !expand(second, q + 2, order, local_second, workspace))
      return false;
    double jacobian[9];
    free_jacobian(local_first, local_second, jacobian);
    for (int step = 0; step < order; ++step) {
      compose(local_first, 3, 2, coordinates, value, line);
      compose(local_second, 3, 2, coordinates + 2, value_two, line);
      for (int n = 1; n <= order; ++n) {
        double residual[3], correction[3];
        for (int i = 0; i < 3; ++i)
          residual[i] = value[i].c[term(n, 0)] - value_two[i].c[term(n, 0)];
        if (!solve3(jacobian, residual, correction)) return false;
        for (int f = 0; f < 3; ++f) coordinates[free[f]].c[term(n, 0)] -= correction[f];
      }
    }
  }
  if (output == 0) {
    if (order >= 1) {
      compose(local_first, 3, 2, coordinates, out, line);
    } else if (!expand(first, q, 0, out, workspace)) {
      return false;
    }
  } else {
    const int base = output == 1 ? 0 : 2;
    out[0] = coordinates[base];
    out[1] = coordinates[base + 1];
  }
  if (reversed) reverse(out, node.outputs, order);
  return true;
}

bool CadGeometry::expand(int32_t index, const double* point, int order, Taylor* out,
                         EvaluationWorkspace& workspace) const {
  const Node& node = this->node(index);
  for (int k = 0; k < node.inputs; ++k)
    if (!std::isfinite(point[k])) return false;
  const Truncation local{node.inputs, order};
  if (node.closed != ClosedForm::kNone) return expand_closed(node, point, order, out);
  switch (node.kind) {
    case PHX_MC_CAD_NODE_TEMPLATE:
      return expand_template(node, point, order, out, workspace);
    case PHX_MC_CAD_NODE_BSPLINE_CURVE:
      return expand_bspline_curve(node, point[0], order,
                                  workspace.span_node == &node
                                      ? static_cast<int>(workspace.span_chart) : -1, out);
    case PHX_MC_CAD_NODE_BSPLINE_SURFACE:
      return expand_bspline_surface(node, point, order, out);
    case PHX_MC_CAD_NODE_TSPLINE_SURFACE:
      return expand_tspline_surface(node, point, order, out);
    case PHX_MC_CAD_NODE_AFFINE: {
      Taylor source[3];
      if (!expand(static_cast<int32_t>(node.integers[0]), point, order, source, workspace))
        return false;
      const int d = node.outputs;
      const double* matrix = node.reals;
      const double* offset = node.reals + d * d;
      for (int r = 0; r < d; ++r) {
        out[r] = constant(0.0, local);
        for (int c = 0; c < d; ++c) axpy(out[r], matrix[d * r + c], source[c], out[r], local);
        out[r].c[0] += offset[r];
      }
      return true;
    }
    case PHX_MC_CAD_NODE_REPARAMETERIZED: {
      const double first = node.reals[0], last = node.reals[1];
      const bool reversed = node.integers[1] != 0;
      const double fraction = reversed ? 1.0 - point[0] : point[0];
      Taylor parameter = constant(first + fraction * (last - first), local);
      if (order >= 1) parameter.c[term(1, 0)] = reversed ? -(last - first) : last - first;
      return substitute(static_cast<int32_t>(node.integers[0]), &parameter, local, out,
                        workspace);
    }
    case PHX_MC_CAD_NODE_OFFSET_SURFACE:
      return expand_offset_surface(node, point, order, out, workspace);
    case PHX_MC_CAD_NODE_OFFSET_CURVE:
      return expand_offset_curve(node, point[0], order, out, workspace);
    case PHX_MC_CAD_NODE_RULED: {
      const Taylor u = variable(point[0], 0, local), v = variable(point[1], 1, local);
      Taylor first[3], second[3];
      if (!substitute(static_cast<int32_t>(node.integers[0]), &u, local, first, workspace) ||
          !substitute(static_cast<int32_t>(node.integers[1]), &u, local, second, workspace))
        return false;
      Taylor complement, left, right;
      axpy(constant(1.0, local), -1.0, v, complement, local);
      for (int i = 0; i < 3; ++i) {
        multiply(complement, first[i], left, local);
        multiply(v, second[i], right, local);
        axpy(left, 1.0, right, out[i], local);
      }
      return true;
    }
    case PHX_MC_CAD_NODE_EXTRUSION: {
      const Taylor u = variable(point[0], 0, local), v = variable(point[1], 1, local);
      Taylor curve[3];
      if (!substitute(static_cast<int32_t>(node.integers[0]), &u, local, curve, workspace))
        return false;
      for (int i = 0; i < 3; ++i) axpy(curve[i], node.reals[i], v, out[i], local);
      return true;
    }
    case PHX_MC_CAD_NODE_REVOLUTION: {
      const Taylor angle = variable(point[0], 0, local), v = variable(point[1], 1, local);
      Taylor profile[3];
      if (!substitute(static_cast<int32_t>(node.integers[0]), &v, local, profile, workspace))
        return false;
      const double* origin = node.reals;
      const double* axis = node.reals + 3;
      Taylor relative[3], dot = constant(0.0, local), radial[3], sine, cosine;
      for (int i = 0; i < 3; ++i) {
        relative[i] = profile[i];
        relative[i].c[0] -= origin[i];
        axpy(dot, axis[i], relative[i], dot, local);
      }
      for (int i = 0; i < 3; ++i) axpy(relative[i], -axis[i], dot, radial[i], local);
      sine_cosine(angle, sine, cosine, local);
      for (int i = 0; i < 3; ++i) {
        Taylor axial, cross, turned, product;
        scale(dot, axis[i], axial, local);
        scale(radial[(i + 2) % 3], axis[(i + 1) % 3], cross, local);
        axpy(cross, -axis[(i + 2) % 3], radial[(i + 1) % 3], cross, local);
        out[i] = axial;
        out[i].c[0] += origin[i];
        multiply(radial[i], cosine, turned, local);
        axpy(out[i], 1.0, turned, out[i], local);
        multiply(cross, sine, product, local);
        axpy(out[i], 1.0, product, out[i], local);
      }
      return true;
    }
    case PHX_MC_CAD_NODE_SURFACE_MAPPED: {
      Taylor chart[2];
      if (!expand(static_cast<int32_t>(node.integers[1]), point, order, chart, workspace))
        return false;
      return substitute(static_cast<int32_t>(node.integers[0]), chart, local, out, workspace);
    }
    case PHX_MC_CAD_NODE_ISOPARAMETRIC: {
      const int fixed_axis = static_cast<int>(node.integers[1]);
      Taylor chart[2];
      chart[fixed_axis] = constant(node.reals[0], local);
      chart[1 - fixed_axis] = variable(point[0], 0, local);
      return substitute(static_cast<int32_t>(node.integers[0]), chart, local, out, workspace);
    }
    case PHX_MC_CAD_NODE_INTERSECTION:
      return expand_intersection(node, point[0], order, out, workspace);
    default:
      return false;
  }
}

bool CadGeometry::substitute(int32_t index, const Taylor* inputs, const Truncation& truncation,
                             Taylor* out, EvaluationWorkspace& workspace) const {
  const Node& node = this->node(index);
  const double point[2] = {inputs[0].c[0], node.inputs == 2 ? inputs[1].c[0] : 0.0};
  if (identity_inputs(inputs, node.inputs, truncation))
    return expand(index, point, truncation.order, out, workspace);
  Taylor local[3];
  if (!expand(index, point, truncation.order, local, workspace)) return false;
  compose(local, node.outputs, node.inputs, inputs, out, truncation);
  return true;
}

void CadGeometry::curve_breakpoints(int32_t index, double first, double last,
                                    std::vector<double>& points, int& degree) const {
  const Node& current = node(index);
  const double scale = last - first;
  auto add = [&](double parameter) {
    if (scale == 0) return;
    const double s = (parameter - first) / scale;
    if (s > 0 && s < 1 && std::isfinite(s)) points.push_back(s);
  };
  switch (current.kind) {
    case PHX_MC_CAD_NODE_BSPLINE_CURVE: {
      degree = std::max(degree, static_cast<int>(current.integers[0]));
      const int64_t knot_count = current.integers[0] + current.integers[1] + 1;
      for (int64_t i = 0; i < knot_count; ++i) add(current.reals[i]);
      break;
    }
    case PHX_MC_CAD_NODE_INTERSECTION:
      if (current.integers[4] < 0) {
        for (int64_t chart = 1; chart < current.integers[2]; ++chart)
          add(current.integers[6] ? current.reals[0] - chart : static_cast<double>(chart));
        if (current.integers[5] >= 0)
          add(current.integers[6] ? current.reals[0] - current.reals[1]
                                  : current.reals[1]);
      }
      break;
    case PHX_MC_CAD_NODE_REPARAMETERIZED: {
      const double delta = current.reals[1] - current.reals[0];
      const bool reversed = current.integers[1] != 0;
      const auto map = [&](double value) {
        return current.reals[0] + (reversed ? 1 - value : value) * delta;
      };
      curve_breakpoints(static_cast<int32_t>(current.integers[0]), map(first), map(last),
                        points, degree);
      break;
    }
    case PHX_MC_CAD_NODE_AFFINE:
    case PHX_MC_CAD_NODE_OFFSET_CURVE:
      curve_breakpoints(static_cast<int32_t>(current.integers[0]), first, last, points, degree);
      break;
    case PHX_MC_CAD_NODE_SURFACE_MAPPED:
      curve_breakpoints(static_cast<int32_t>(current.integers[1]), first, last, points, degree);
      break;
    default:
      break;
  }
}

bool CadGeometry::jets_in_span(int32_t index, double parameter, double span_interior,
                               int order, double* result, EvaluationWorkspace& workspace) const {
  if (!std::isfinite(parameter) || !std::isfinite(span_interior)) return false;
  int32_t current=index;
  double interior=span_interior;
  const Node* selected=nullptr;
  int64_t chart=-1;
  for (;;) {
    const Node& n=node(current);
    if (n.kind==PHX_MC_CAD_NODE_INTERSECTION) {
      const double carrier=n.integers[6] ? n.reals[0]-interior : interior;
      const int64_t charts=n.integers[2];
      if (!(carrier>0 && carrier<static_cast<double>(charts)) ||
          carrier==std::floor(carrier) ||
          (n.integers[5]>=0 && carrier==n.reals[1])) return false;
      chart=n.integers[4];
      if (chart>=0 && !(carrier>chart && carrier<chart+1)) return false;
      if (chart<0) {
        chart=std::min(static_cast<int64_t>(std::floor(carrier)),charts-1);
        if (n.integers[5]>=0 && carrier>=n.reals[1]) chart=n.integers[5];
      }
      selected=&n;
      break;
    }
    if (n.kind==PHX_MC_CAD_NODE_BSPLINE_CURVE) {
      if (!(interior>n.reals[n.integers[0]] && interior<n.reals[n.integers[1]]))
        return false;
      const int64_t knot_count=n.integers[0]+n.integers[1]+1;
      for (int64_t i=0;i<knot_count;++i) if (interior==n.reals[i]) return false;
      selected=&n;
      chart=knot_span(n.reals,static_cast<int>(n.integers[0]),
                      static_cast<int>(n.integers[1]),interior);
      break;
    }
    if (n.kind==PHX_MC_CAD_NODE_REPARAMETERIZED) {
      interior=n.reals[0]+(n.integers[1] ? 1-interior : interior)*(n.reals[1]-n.reals[0]);
      current=static_cast<int32_t>(n.integers[0]);
    } else if (n.kind==PHX_MC_CAD_NODE_AFFINE || n.kind==PHX_MC_CAD_NODE_OFFSET_CURVE) {
      current=static_cast<int32_t>(n.integers[0]);
    } else if (n.kind==PHX_MC_CAD_NODE_SURFACE_MAPPED) {
      current=static_cast<int32_t>(n.integers[1]);
    } else {
      break;
    }
  }
  const Node* previous_node=workspace.span_node;
  const int64_t previous_chart=workspace.span_chart;
  workspace.span_node=selected;
  workspace.span_chart=chart;
  const bool ok=jets(index,&parameter,order,result,workspace);
  workspace.span_node=previous_node;
  workspace.span_chart=previous_chart;
  return ok;
}

bool CadGeometry::jets(int32_t index, const double* point, int order, double* result,
                       EvaluationWorkspace& workspace) const {
  const Node& node = this->node(index);
  Taylor local[3];
  if (!expand(index, point, order, local, workspace)) return false;
  int64_t slot = 0;
  for (int n = 0; n <= order; ++n)
    for (int j = 0; j <= (node.inputs == 2 ? n : 0); ++j) {
      const double scale = factorial(n - j) * factorial(j);
      for (int i = 0; i < node.outputs; ++i)
        result[slot++] = scale * local[i].c[term(n - j, j)];
    }
  return true;
}

int32_t template_index(const char* name) noexcept {
  for (int32_t id = 0; id < kTemplateCount; ++id)
    if (std::string_view(kTemplates[id].name) == name) return id;
  return -1;
}

bool surface_differential(const double* jet, SurfaceDifferential& out) noexcept {
  const double *su = jet + 3, *sv = jet + 6, *suu = jet + 9, *suv = jet + 12, *svv = jet + 15;
  const auto dot = [](const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  };
  const double e = dot(su, su), f = dot(su, sv), g = dot(sv, sv);
  double normal[3] = {su[1] * sv[2] - su[2] * sv[1], su[2] * sv[0] - su[0] * sv[2],
                      su[0] * sv[1] - su[1] * sv[0]};
  const double length = std::sqrt(dot(normal, normal));
  if (!(length > 0.0) || !std::isfinite(length)) return false;
  for (double& component : normal) component /= length;
  const double l = dot(suu, normal), m = dot(suv, normal), n = dot(svv, normal);
  const double determinant = e * g - f * f;
  if (!(determinant > 0.0)) return false;
  // Shape operator W = I^-1 II = [a b; c d]; its eigenvalues are
  // (a + d) / 2 +- sqrt(((a - d) / 2)^2 + b c), which stays accurate at umbilics
  // where the mean^2 - Gaussian form cancels.
  const double a = (g * l - f * m) / determinant, b = (g * m - f * n) / determinant;
  const double c = (e * m - f * l) / determinant, d = (e * n - f * m) / determinant;
  const double mean = 0.5 * (a + d), half_difference = 0.5 * (a - d);
  const double spread = std::sqrt(std::max(half_difference * half_difference + b * c, 0.0));
  out.first_form[0] = e, out.first_form[1] = f, out.first_form[2] = g;
  for (int i = 0; i < 3; ++i) out.normal[i] = normal[i];
  out.curvatures[0] = mean - spread;
  out.curvatures[1] = mean + spread;
  return true;
}

bool curve_differential(const double* jet, double* speed, double* curvature) noexcept {
  const double *d1 = jet + 3, *d2 = jet + 6;
  const double cross[3] = {d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2],
                           d1[0] * d2[1] - d1[1] * d2[0]};
  const double length = std::sqrt(d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]);
  if (!(length > 0.0) || !std::isfinite(length)) return false;
  *speed = length;
  *curvature =
      std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]) /
      (length * length * length);
  return true;
}

}  // namespace phx::mc::cad

extern "C" int32_t phx_mc_cad_geometry_evaluate(const phx_mc_cad_model* model,
                                                int32_t entity_kind, int64_t entity,
                                                int64_t count, const double* parameters,
                                                int32_t quantity, double* result) {
  using namespace phx::mc::cad;
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (model == nullptr || count < 0 || quantity < PHX_MC_CAD_VALUE ||
        quantity > PHX_MC_CAD_DIFFERENTIAL)
      return PHX_MC_INVALID_ARGUMENT;
    CadGeometry geometry;
    const int32_t status = geometry.bind(*model);
    if (status != PHX_MC_OK) return status;
    int32_t node = -1;
    int inputs = 1, outputs = 0;
    switch (entity_kind) {
      case PHX_MC_CAD_SURFACE:
        if (entity < 0 || entity >= geometry.surface_count()) return PHX_MC_INVALID_ARGUMENT;
        node = geometry.surface_node(entity), inputs = 2, outputs = 3;
        break;
      case PHX_MC_CAD_CURVE:
        if (entity < 0 || entity >= geometry.curve_count()) return PHX_MC_INVALID_ARGUMENT;
        node = geometry.curve_node(entity), outputs = 3;
        break;
      case PHX_MC_CAD_PCURVE:
        if (entity < 0 || entity >= geometry.use_count() || quantity == PHX_MC_CAD_DIFFERENTIAL)
          return PHX_MC_INVALID_ARGUMENT;
        node = geometry.pcurve_node(entity), outputs = 2;
        break;
      default:
        return PHX_MC_INVALID_ARGUMENT;
    }
    const int order = quantity == PHX_MC_CAD_DIFFERENTIAL ? 2 : quantity;
    const int64_t jet_width = outputs * (inputs == 2 ? (order + 1) * (order + 2) / 2 : order + 1);
    const int64_t width =
        quantity == PHX_MC_CAD_DIFFERENTIAL ? (inputs == 2 ? 8 : 2) : jet_width;
    if (!phx::mc::addressable(count, inputs, sizeof(double)) ||
        !phx::mc::addressable(count, width, sizeof(double)) ||
        (count > 0 && (parameters == nullptr || result == nullptr)))
      return PHX_MC_INVALID_ARGUMENT;
    EvaluationWorkspace workspace;
    double jet[18];
    for (int64_t k = 0; k < count; ++k) {
      const double* point = parameters + inputs * k;
      double* row = result + width * k;
      if (quantity != PHX_MC_CAD_DIFFERENTIAL) {
        if (!geometry.jets(node, point, order, row, workspace)) return PHX_MC_INVALID_INPUT;
        continue;
      }
      if (!geometry.jets(node, point, order, jet, workspace)) return PHX_MC_INVALID_INPUT;
      if (inputs == 2) {
        SurfaceDifferential differential;
        if (!surface_differential(jet, differential)) return PHX_MC_INVALID_INPUT;
        for (int i = 0; i < 3; ++i) row[i] = differential.first_form[i];
        for (int i = 0; i < 3; ++i) row[3 + i] = differential.normal[i];
        row[6] = differential.curvatures[0];
        row[7] = differential.curvatures[1];
      } else if (!curve_differential(jet, row, row + 1)) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    return PHX_MC_OK;
  });
}
