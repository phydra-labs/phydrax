// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Outward enclosures of simplex quadratic Bernstein coefficients and exact
// continuous validity proofs for quadratic tetrahedral maps.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "capi_guard.hpp"
#include "dyadic.hpp"
#include "expansion.hpp"
#include "exact_scalar.hpp"
#include "floating_point_mode.hpp"
#include "geometry3.hpp"
#include "phydrax_meshcore.h"
#include "quadratic_validity_screen.hpp"
#include "rounding.hpp"

namespace {

using phx::mc::Dyadic;
using phx::mc::geometry::next_down;
using phx::mc::geometry::next_up;

constexpr int kMaximumVertices = 4;
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr int kNodeCount = 10;
constexpr int kDerivativePowers = 4;
constexpr std::int32_t kValidityValid = 0;
constexpr std::int32_t kValidityInvalid = 1;
constexpr std::int32_t kValidityUnresolved = 2;

struct ValidityResult final {
  std::int32_t decision;
  double lower;
  double upper;
};

struct Power final {
  int x;
  int y;
  int z;
  int degree;
};

// Complete power basis through degree three in deterministic graded order.
constexpr std::array<Power, 20> kPowers{{
    {0, 0, 0, 0},
    {1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1},
    {2, 0, 0, 2}, {1, 1, 0, 2}, {1, 0, 1, 2},
    {0, 2, 0, 2}, {0, 1, 1, 2}, {0, 0, 2, 2},
    {3, 0, 0, 3}, {2, 1, 0, 3}, {2, 0, 1, 3},
    {1, 2, 0, 3}, {1, 1, 1, 3}, {1, 0, 2, 3},
    {0, 3, 0, 3}, {0, 2, 1, 3}, {0, 1, 2, 3}, {0, 0, 3, 3},
}};

int power_index(int x, int y, int z) noexcept {
  const int degree = x + y + z;
  if (degree == 0) return 0;
  if (degree == 1) return x != 0 ? 1 : y != 0 ? 2 : 3;
  if (degree == 2) {
    if (x == 2) return 4;
    if (x == 1) return y != 0 ? 5 : 6;
    if (y == 2) return 7;
    return y != 0 ? 8 : 9;
  }
  if (x == 3) return 10;
  if (x == 2) return y != 0 ? 11 : 12;
  if (x == 1) return y == 2 ? 13 : y == 1 ? 14 : 15;
  if (y == 3) return 16;
  return y == 2 ? 17 : y == 1 ? 18 : 19;
}

int falling(int value, int order) noexcept {
  if (order > value) return 0;
  int result = 1;
  for (int factor = 0; factor < order; ++factor) result *= value - factor;
  return result;
}

int bernstein_count(int degree) noexcept {
  return (degree + 1) * (degree + 2) * (degree + 3) / 6;
}

int compare_absolute(const Dyadic& first, const Dyadic& second) {
  const int first_sign = first.sign();
  const int second_sign = second.sign();
  if (first_sign == 0) return second_sign == 0 ? 0 : -1;
  if (second_sign == 0) return 1;
  if (first_sign == second_sign) {
    return first_sign > 0 ? (first - second).sign() : (second - first).sign();
  }
  const int sum_sign = (first + second).sign();
  return first_sign < 0 ? -sum_sign : sum_sign;
}

using Jacobian = std::array<Dyadic, 9 * kDerivativePowers>;
using JacobianCorners = std::array<std::array<Dyadic, 9>, 4>;
using ParameterVertex = std::array<Dyadic, 3>;
using ParameterVertices = std::array<ParameterVertex, 4>;
using Controls = std::array<Dyadic, 20>;

struct ValidityCell final {
  Controls controls;
  ParameterVertices vertices;
};

struct VertexBlend final {
  int first;
  int second;
};

Dyadic determinant(const Jacobian& jacobian, int first, int second, int third) {
  return phx::mc::exact::det3(
      jacobian[first], jacobian[4 + second], jacobian[8 + third],
      jacobian[12 + first], jacobian[16 + second], jacobian[20 + third],
      jacobian[24 + first], jacobian[28 + second], jacobian[32 + third]);
}

std::array<Dyadic, 20> determinant_power_coefficients(const Jacobian& jacobian) {
  std::array<std::array<bool, 4>, 3> active{};
  for (int column = 0; column < 3; ++column) {
    for (int power = 0; power < 4; ++power) {
      active[column][power] = jacobian[column * 4 + power].sign() != 0 ||
                              jacobian[(3 + column) * 4 + power].sign() != 0 ||
                              jacobian[(6 + column) * 4 + power].sign() != 0;
    }
  }
  std::array<Dyadic, 20> coefficients;
  for (int first = 0; first < 4; ++first) {
    if (!active[0][first]) continue;
    for (int second = 0; second < 4; ++second) {
      if (!active[1][second]) continue;
      for (int third = 0; third < 4; ++third) {
        if (!active[2][third]) continue;
        Dyadic value = determinant(jacobian, first, second, third);
        if (value.sign() == 0) continue;
        const int x = kPowers[first].x + kPowers[second].x + kPowers[third].x;
        const int y = kPowers[first].y + kPowers[second].y + kPowers[third].y;
        const int z = kPowers[first].z + kPowers[second].z + kPowers[third].z;
        Dyadic& coefficient = coefficients[power_index(x, y, z)];
        coefficient = coefficient.sign() == 0 ? std::move(value) : coefficient + value;
      }
    }
  }
  return coefficients;
}

int actual_degree(const std::array<Dyadic, 20>& coefficients) noexcept {
  int degree = 0;
  for (std::size_t index = 0; index < coefficients.size(); ++index) {
    if (coefficients[index].sign() != 0) degree = std::max(degree, kPowers[index].degree);
  }
  return degree;
}

const std::array<Dyadic, 7>& small_integer_dyadics() {
  static const std::array<Dyadic, 7> values{{Dyadic(0.0), Dyadic(1.0), Dyadic(2.0),
                                              Dyadic(3.0), Dyadic(4.0), Dyadic(5.0),
                                              Dyadic(6.0)}};
  return values;
}

Controls determinant_bernstein_controls(const std::array<Dyadic, 20>& coefficients, int degree) {
  Controls controls;
  const int count = bernstein_count(degree);
  std::array<Power, 20> alphas{};
  int alpha_count = 0;
  for (int x = 0; x <= degree; ++x) {
    for (int y = 0; y <= degree - x; ++y) {
      for (int z = 0; z <= degree - x - y; ++z) {
        alphas[alpha_count++] = {x, y, z, x + y + z};
      }
    }
  }
  for (int alpha = 0; alpha < alpha_count; ++alpha) {
    Dyadic control;
    bool has_term = false;
    for (int beta = 0; beta < count; ++beta) {
      const Power& exponent = kPowers[beta];
      const int denominator = falling(degree, exponent.degree);
      const int multiplier =
          (6 / denominator) * falling(alphas[alpha].x, exponent.x) *
          falling(alphas[alpha].y, exponent.y) * falling(alphas[alpha].z, exponent.z);
      const Dyadic& coefficient = coefficients[beta];
      if (multiplier == 0 || coefficient.sign() == 0) continue;
      const Dyadic value = coefficient * small_integer_dyadics()[multiplier];
      if (has_term) {
        control = control + value;
      } else {
        control = value;
        has_term = true;
      }
    }
    controls[power_index(alphas[alpha].x, alphas[alpha].y, alphas[alpha].z)] =
        std::move(control);
  }
  return controls;
}

JacobianCorners jacobian_corners(const Jacobian& jacobian) {
  JacobianCorners corners;
  for (int vertex = 0; vertex < 4; ++vertex) {
    const int active_power = vertex;
    for (int entry = 0; entry < 9; ++entry) {
      const Dyadic& constant = jacobian[entry * 4];
      corners[vertex][entry] = vertex == 0 ? constant : constant + jacobian[entry * 4 + active_power];
    }
  }
  return corners;
}

Dyadic squared_hadamard_scale(const JacobianCorners& corners) {
  std::array<Dyadic, 9> squared_maxima;
  for (int entry = 0; entry < 9; ++entry) {
    const Dyadic* maximum = &corners[0][entry];
    for (int vertex = 1; vertex < 4; ++vertex) {
      if (compare_absolute(corners[vertex][entry], *maximum) > 0) {
        maximum = &corners[vertex][entry];
      }
    }
    squared_maxima[entry] = *maximum * *maximum;
  }
  std::array<Dyadic, 3> column_norms;
  for (int column = 0; column < 3; ++column) {
    column_norms[column] = squared_maxima[column] + squared_maxima[3 + column] +
                           squared_maxima[6 + column];
  }
  return (column_norms[0] * column_norms[1]) * column_norms[2];
}

Dyadic relative_threshold(double relative_floor, const JacobianCorners& corners) {
  const Dyadic floor(relative_floor);
  static const Dyadic thirty_six(36.0);
  return ((floor * floor) * thirty_six) * squared_hadamard_scale(corners);
}

int corner_control_index(int vertex, int degree) noexcept {
  if (degree == 0 || vertex == 0) return 0;
  if (vertex == 1) return power_index(degree, 0, 0);
  if (vertex == 2) return power_index(0, degree, 0);
  return power_index(0, 0, degree);
}

Dyadic minimum_control(const Controls& controls, int count, int& index) {
  index = 0;
  for (int control = 1; control < count; ++control) {
    if ((controls[control] - controls[index]).sign() < 0) index = control;
  }
  return controls[index];
}

Dyadic maximum_control(const Controls& controls, int count, int& index) {
  index = 0;
  for (int control = 1; control < count; ++control) {
    if ((controls[control] - controls[index]).sign() > 0) index = control;
  }
  return controls[index];
}

bool has_representable_interior(const Dyadic& lower, const Dyadic& upper) {
  if ((upper - lower).sign() <= 0) return false;
  static const Dyadic one(1.0);
  const auto inside = [&](double candidate) {
    const Dyadic value(candidate);
    return (value - lower).sign() > 0 && (upper - value).sign() > 0;
  };
  const double lower_carrier = phx::mc::exact::rounded_quotient(lower, one);
  if (inside(lower_carrier)) return true;
  if (inside(next_up(lower_carrier))) return true;
  const double upper_carrier = phx::mc::exact::rounded_quotient(upper, one);
  if (inside(next_down(upper_carrier))) return true;
  return inside(upper_carrier);
}

bool has_parameter_resolution(const ParameterVertices& vertices,
                              const std::array<bool, 3>& active_axes) {
  for (int axis = 0; axis < 3; ++axis) {
    if (!active_axes[axis]) continue;
    const Dyadic* minimum = &vertices[0][axis];
    const Dyadic* maximum = &vertices[0][axis];
    for (int vertex = 1; vertex < 4; ++vertex) {
      if ((vertices[vertex][axis] - *minimum).sign() < 0) minimum = &vertices[vertex][axis];
      if ((vertices[vertex][axis] - *maximum).sign() > 0) maximum = &vertices[vertex][axis];
    }
    if (has_representable_interior(*minimum, *maximum)) return true;
  }
  return false;
}

std::array<Dyadic, 20> subdivide_controls(const Controls& controls, int degree,
                                         const std::array<VertexBlend, 4>& child_vertices) {
  std::array<Dyadic, 20> output;
  const int output_count = bernstein_count(degree);
  for (int output_index = 0; output_index < output_count; ++output_index) {
    const Power& alpha = kPowers[output_index];
    const std::array<int, 4> multiplicities{{degree - alpha.degree, alpha.x, alpha.y, alpha.z}};
    Controls current = controls;
    Controls next;
    int current_degree = degree;
    for (int vertex = 0; vertex < 4; ++vertex) {
      for (int repetition = 0; repetition < multiplicities[vertex]; ++repetition) {
        const int next_count = bernstein_count(current_degree - 1);
        const VertexBlend blend = child_vertices[vertex];
        for (int beta_index = 0; beta_index < next_count; ++beta_index) {
          const Power& beta = kPowers[beta_index];
          const int first = power_index(beta.x + (blend.first == 1),
                                        beta.y + (blend.first == 2),
                                        beta.z + (blend.first == 3));
          if (blend.first == blend.second) {
            next[beta_index] = current[first];
          } else {
            const int second = power_index(beta.x + (blend.second == 1),
                                           beta.y + (blend.second == 2),
                                           beta.z + (blend.second == 3));
            next[beta_index] = (current[first] + current[second]).scaled_power(-1);
          }
        }
        current.swap(next);
        --current_degree;
      }
    }
    output[output_index] = std::move(current[0]);
  }
  return output;
}

ParameterVertex midpoint(const ParameterVertex& first, const ParameterVertex& second) {
  ParameterVertex result;
  for (int axis = 0; axis < 3; ++axis) result[axis] = (first[axis] + second[axis]).scaled_power(-1);
  return result;
}

struct Split final {
  int first = -1;
  int second = -1;
};

Split longest_splittable_edge(const ParameterVertices& vertices,
                              const std::array<bool, 3>& active_axes) {
  static const Dyadic one(1.0);
  double longest = 0.0;
  Split split;
  for (int first = 0; first < 4; ++first) {
    for (int second = first + 1; second < 4; ++second) {
      double extent = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        if (!active_axes[axis]) continue;
        const double a = phx::mc::exact::rounded_quotient(vertices[first][axis], one);
        const double b = phx::mc::exact::rounded_quotient(vertices[second][axis], one);
        extent = std::max(extent, std::fabs(a - b));
      }
      if (extent > longest) {
        longest = extent;
        split = {first, second};
      }
    }
  }
  return split;
}

std::array<ValidityCell, 2> bisect_cell(const ValidityCell& parent, int degree, Split split) {
  const ParameterVertex center = midpoint(parent.vertices[split.first], parent.vertices[split.second]);
  std::array<VertexBlend, 4> first_blends{};
  std::array<VertexBlend, 4> second_blends{};
  ParameterVertices first_vertices = parent.vertices;
  ParameterVertices second_vertices = parent.vertices;
  first_vertices[split.second] = center;
  second_vertices[split.first] = center;
  for (int vertex = 0; vertex < 4; ++vertex) {
    first_blends[vertex] = {vertex, vertex};
    second_blends[vertex] = {vertex, vertex};
  }
  first_blends[split.second] = {split.first, split.second};
  second_blends[split.first] = {split.first, split.second};

  std::array<ValidityCell, 2> children;
  children[0].controls = subdivide_controls(parent.controls, degree, first_blends);
  children[0].vertices = std::move(first_vertices);
  children[1].controls = subdivide_controls(parent.controls, degree, second_blends);
  children[1].vertices = std::move(second_vertices);
  return children;
}

struct UnivariateCell final {
  std::array<Dyadic, 4> controls;
  Dyadic lower;
  Dyadic upper;
};

std::int32_t classify_univariate_validity(const Controls& root_controls, int degree,
                                          const Dyadic& threshold, int axis) {
  UnivariateCell root;
  for (int power = 0; power <= degree; ++power) {
    const int x = axis == 0 ? power : 0;
    const int y = axis == 1 ? power : 0;
    const int z = axis == 2 ? power : 0;
    root.controls[power] = root_controls[power_index(x, y, z)];
  }
  root.lower = Dyadic(0.0);
  root.upper = Dyadic(1.0);
  std::vector<UnivariateCell> pending;
  pending.push_back(std::move(root));
  bool unresolved = false;

  while (!pending.empty()) {
    UnivariateCell cell = std::move(pending.back());
    pending.pop_back();
    bool terminal_unresolved = false;
    for (int endpoint : std::array<int, 2>{{0, degree}}) {
      const Dyadic& value = cell.controls[endpoint];
      if (value.sign() <= 0) return kValidityInvalid;
      const Dyadic margin = value * value - threshold;
      if (margin.sign() < 0) return kValidityInvalid;
      if (margin.sign() == 0) terminal_unresolved = true;
    }
    int minimum_index = 0;
    int maximum_index = 0;
    for (int index = 1; index <= degree; ++index) {
      if ((cell.controls[index] - cell.controls[minimum_index]).sign() < 0) minimum_index = index;
      if ((cell.controls[index] - cell.controls[maximum_index]).sign() > 0) maximum_index = index;
    }
    const Dyadic& minimum = cell.controls[minimum_index];
    const Dyadic& maximum = cell.controls[maximum_index];
    if (maximum.sign() <= 0 || (maximum * maximum - threshold).sign() < 0) {
      return kValidityInvalid;
    }
    if (!terminal_unresolved && minimum.sign() > 0 &&
        (minimum * minimum - threshold).sign() > 0) {
      continue;
    }
    if (terminal_unresolved || !has_representable_interior(cell.lower, cell.upper)) {
      unresolved = true;
      continue;
    }

    std::array<Dyadic, 4> work = cell.controls;
    UnivariateCell left;
    UnivariateCell right;
    left.lower = cell.lower;
    right.upper = cell.upper;
    left.controls[0] = cell.controls[0];
    right.controls[degree] = cell.controls[degree];
    for (int level = 1; level <= degree; ++level) {
      const int active = degree - level + 1;
      for (int index = 0; index < active; ++index) {
        work[index] = (work[index] + work[index + 1]).scaled_power(-1);
      }
      left.controls[level] = work[0];
      right.controls[degree - level] = work[active - 1];
    }
    const Dyadic center = (cell.lower + cell.upper).scaled_power(-1);
    left.upper = center;
    right.lower = center;
    pending.push_back(std::move(right));
    pending.push_back(std::move(left));
  }
  return unresolved ? kValidityUnresolved : kValidityValid;
}

std::int32_t classify_continuous_validity(const Controls& root_controls, int degree,
                                          const Dyadic& threshold,
                                          const std::array<bool, 3>& active_axes) {
  int active_axis = -1;
  for (int axis = 0; axis < 3; ++axis) {
    if (!active_axes[axis]) continue;
    if (active_axis >= 0) {
      active_axis = -2;
      break;
    }
    active_axis = axis;
  }
  if (active_axis >= 0) {
    return classify_univariate_validity(root_controls, degree, threshold, active_axis);
  }
  ValidityCell root;
  root.controls = root_controls;
  root.vertices[0] = {Dyadic(0.0), Dyadic(0.0), Dyadic(0.0)};
  root.vertices[1] = {Dyadic(1.0), Dyadic(0.0), Dyadic(0.0)};
  root.vertices[2] = {Dyadic(0.0), Dyadic(1.0), Dyadic(0.0)};
  root.vertices[3] = {Dyadic(0.0), Dyadic(0.0), Dyadic(1.0)};
  std::vector<ValidityCell> pending;
  pending.push_back(std::move(root));
  bool unresolved = false;
  const int count = bernstein_count(degree);

  while (!pending.empty()) {
    ValidityCell cell = std::move(pending.back());
    pending.pop_back();
    bool terminal_unresolved = false;
    for (int vertex = 0; vertex < 4; ++vertex) {
      const Dyadic& value = cell.controls[corner_control_index(vertex, degree)];
      if (value.sign() <= 0) return kValidityInvalid;
      const Dyadic margin = value * value - threshold;
      if (margin.sign() < 0) return kValidityInvalid;
      if (margin.sign() == 0) terminal_unresolved = true;
    }
    int minimum_index = 0;
    int maximum_index = 0;
    const Dyadic minimum = minimum_control(cell.controls, count, minimum_index);
    const Dyadic maximum = maximum_control(cell.controls, count, maximum_index);
    if (maximum.sign() <= 0 || (maximum * maximum - threshold).sign() < 0) {
      return kValidityInvalid;
    }
    if (!terminal_unresolved && minimum.sign() > 0 &&
        (minimum * minimum - threshold).sign() > 0) {
      continue;
    }
    if (terminal_unresolved || !has_parameter_resolution(cell.vertices, active_axes)) {
      unresolved = true;
      continue;
    }
    const Split split = longest_splittable_edge(cell.vertices, active_axes);
    if (split.first < 0) {
      unresolved = true;
      continue;
    }
    std::array<ValidityCell, 2> children = bisect_cell(cell, degree, split);
    // LIFO order visits the first child before the second, independent of allocation history.
    pending.push_back(std::move(children[1]));
    pending.push_back(std::move(children[0]));
  }
  return unresolved ? kValidityUnresolved : kValidityValid;
}

ValidityResult evaluate_validity_cell(const double* nodes, const std::int32_t* derivative_weights,
                                      double relative_floor) {
  Jacobian jacobian;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      for (int power = 0; power < 4; ++power) {
        Dyadic sum;
        for (int node = 0; node < kNodeCount; ++node) {
          const std::int32_t weight = derivative_weights[(column * 4 + power) * kNodeCount + node];
          if (weight == 0) continue;
          Dyadic term(nodes[node * 3 + row]);
          if (weight == -1) {
            term.negate();
          } else if (weight != 1) {
            term = term * Dyadic(static_cast<double>(weight));
          }
          sum = sum + term;
        }
        jacobian[(row * 3 + column) * 4 + power] = std::move(sum);
      }
    }
  }

  const auto coefficients = determinant_power_coefficients(jacobian);
  const int degree = actual_degree(coefficients);
  const Controls controls = determinant_bernstein_controls(coefficients, degree);
  const int count = bernstein_count(degree);
  int minimum_index = 0;
  int maximum_index = 0;
  const Dyadic minimum = minimum_control(controls, count, minimum_index);
  const Dyadic maximum = maximum_control(controls, count, maximum_index);
  const Dyadic& six = small_integer_dyadics()[6];
  const auto bounds = phx::mc::exact::directed_quotient_bounds(minimum, six);
  const auto maximum_bounds = maximum_index == minimum_index
                                  ? bounds
                                  : phx::mc::exact::directed_quotient_bounds(maximum, six);
  const Dyadic threshold = relative_threshold(relative_floor, jacobian_corners(jacobian));
  std::array<bool, 3> active_axes{};
  for (std::size_t index = 0; index < coefficients.size(); ++index) {
    if (coefficients[index].sign() == 0) continue;
    active_axes[0] = active_axes[0] || kPowers[index].x != 0;
    active_axes[1] = active_axes[1] || kPowers[index].y != 0;
    active_axes[2] = active_axes[2] || kPowers[index].z != 0;
  }
  const std::int32_t decision =
      classify_continuous_validity(controls, degree, threshold, active_axes);
  return {decision, bounds.lower, maximum_bounds.upper};
}

int32_t validate_validity_inputs(std::int64_t count, const double* nodes,
                                 const std::int32_t* derivative_weights, double relative_floor,
                                 std::int32_t* decisions, double* lower, double* upper) noexcept {
  if (count < 0 || !std::isfinite(relative_floor) || relative_floor < 0.0 ||
      relative_floor >= 1.0 || !phx::mc::addressable(count, 30, sizeof(double)) ||
      !phx::mc::addressable(count, 1, sizeof(ValidityResult)) ||
      !phx::mc::addressable(count, 1, sizeof(std::int32_t)) ||
      !phx::mc::addressable(count, 1, sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (count > 0 && (nodes == nullptr || derivative_weights == nullptr || decisions == nullptr ||
                    (lower == nullptr) != (upper == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  for (std::int64_t scalar = 0; scalar < count * 30; ++scalar) {
    if (!std::isfinite(nodes[scalar])) return PHX_MC_NONFINITE_INPUT;
  }
  return PHX_MC_OK;
}

void sum_barycentric_minus_one(const double* values, int count, int& sign) {
  phx::mc::Expansion total(values[0]);
  for (int index = 1; index < count; ++index) total = total + phx::mc::Expansion(values[index]);
  sign = (total - phx::mc::Expansion(1.0)).sign();
}

// Outward endpoint arithmetic for x in [0,1] and interval y. An exact zero
// scale/addend remains exact; all other rounded products and sums step outward.
double scale_down(double x, double y) noexcept {
  if (x == 0.0 || y == 0.0) return 0.0;
  if (x == 1.0) return y;
  return next_down(x * y);
}

double scale_up(double x, double y) noexcept {
  if (x == 0.0 || y == 0.0) return 0.0;
  if (x == 1.0) return y;
  return next_up(x * y);
}

double sum_down(double a, double b) noexcept {
  if (a == 0.0) return b;
  if (b == 0.0) return a;
  const double sum = a + b;
  return sum == 0.0 ? 0.0 : next_down(sum);
}

double sum_up(double a, double b) noexcept {
  if (a == 0.0) return b;
  if (b == 0.0) return a;
  const double sum = a + b;
  return sum == 0.0 ? 0.0 : next_up(sum);
}

double product_lower(double x_lower, double x_upper, double y_lower) noexcept {
  return scale_down(y_lower >= 0.0 ? x_lower : x_upper, y_lower);
}

double product_upper(double x_lower, double x_upper, double y_upper) noexcept {
  return scale_up(y_upper >= 0.0 ? x_upper : x_lower, y_upper);
}

void enclose_item(int m, int components, const double* q_lower, const double* q_upper,
                  const double* l_lower, const double* l_upper, double* lower,
                  double* upper) noexcept {
  double t_lower[kMaximumVertices][kMaximumVertices];
  double t_upper[kMaximumVertices][kMaximumVertices];
  for (int component = 0; component < components; ++component) {
    const double* ql = q_lower + component * m * m;
    const double* qu = q_upper + component * m * m;
    // T = L Q.
    for (int k = 0; k < m; ++k) {
      const double* row_lower = l_lower + k * m;
      const double* row_upper = l_upper + k * m;
      for (int j = 0; j < m; ++j) {
        double low = product_lower(row_lower[0], row_upper[0], ql[j]);
        double high = product_upper(row_lower[0], row_upper[0], qu[j]);
        for (int i = 1; i < m; ++i) {
          low = sum_down(low, product_lower(row_lower[i], row_upper[i], ql[i * m + j]));
          high = sum_up(high, product_upper(row_lower[i], row_upper[i], qu[i * m + j]));
        }
        t_lower[k][j] = low;
        t_upper[k][j] = high;
      }
    }
    // Q'kl = sum_j T_kj L_lj for k <= l.
    double minimum = kInfinity;
    double maximum = -kInfinity;
    for (int k = 0; k < m; ++k) {
      for (int l = k; l < m; ++l) {
        const double* row_lower = l_lower + l * m;
        const double* row_upper = l_upper + l * m;
        double low = product_lower(row_lower[0], row_upper[0], t_lower[k][0]);
        double high = product_upper(row_lower[0], row_upper[0], t_upper[k][0]);
        for (int j = 1; j < m; ++j) {
          low = sum_down(low, product_lower(row_lower[j], row_upper[j], t_lower[k][j]));
          high = sum_up(high, product_upper(row_lower[j], row_upper[j], t_upper[k][j]));
        }
        minimum = std::min(minimum, low);
        maximum = std::max(maximum, high);
      }
    }
    lower[component] = minimum;
    upper[component] = maximum;
  }
}

int32_t validate_quadratic_bounds(int64_t count, int m, int components,
                                  const double* q_lower, const double* q_upper,
                                  const double* l_lower, const double* l_upper) {
  const int64_t q_width = static_cast<int64_t>(components) * m * m;
  const int64_t l_width = static_cast<int64_t>(m) * m;
  bool invalid = false;
  bool nonfinite = false;
  for (int64_t item = 0; item < count; ++item) {
    const double* ql = q_lower + item * q_width;
    const double* qu = q_upper + item * q_width;
    for (int64_t index = 0; index < q_width; ++index) {
      nonfinite = nonfinite || std::isnan(ql[index]) || ql[index] == kInfinity ||
                  std::isnan(qu[index]) || qu[index] == -kInfinity;
      invalid = invalid || !(ql[index] <= qu[index]);
    }
    for (int component = 0; component < components; ++component) {
      const double* cl = ql + component * m * m;
      const double* cu = qu + component * m * m;
      for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < m; ++j) {
          invalid = invalid || cl[i * m + j] != cl[j * m + i] ||
                    cu[i * m + j] != cu[j * m + i];
        }
      }
    }
    const double* ll = l_lower + item * l_width;
    const double* lu = l_upper + item * l_width;
    for (int k = 0; k < m; ++k) {
      bool row_ordered = true;
      for (int i = 0; i < m; ++i) {
        const double low = ll[k * m + i];
        const double high = lu[k * m + i];
        nonfinite = nonfinite || !std::isfinite(low) || !std::isfinite(high);
        row_ordered = row_ordered && 0.0 <= low && low <= high && high <= 1.0;
      }
      if (!row_ordered) {
        invalid = true;
        continue;
      }
      int lower_sum_sign = 0;
      int upper_sum_sign = 0;
      sum_barycentric_minus_one(ll + k * m, m, lower_sum_sign);
      sum_barycentric_minus_one(lu + k * m, m, upper_sum_sign);
      invalid = invalid || lower_sum_sign > 0 || upper_sum_sign < 0;
    }
  }
  if (nonfinite) return PHX_MC_NONFINITE_INPUT;
  return invalid ? PHX_MC_INVALID_INPUT : PHX_MC_OK;
}

}  // namespace

namespace phx::mc {
namespace {

// Error analysis of the binary64 screen (round to nearest, u = 2^-53).
// Every quantity is a polynomial in the node scalars x evaluated by the tree
//   J      = sum over <= 10 nodes of fl(w * x)          (1 product, <= 9 sums)
//   det    = a0 (b1 c2 - b2 c1) - a1 (...) + a2 (...)   (3 J factors + 5 ops)
//   coef_b = sum of <= 6 det terms                       (<= 5 sums)
//   ctrl_a = sum over <= 20 b of fl(m_ab * coef_b)       (1 product, <= 19 sums)
// so the computed control expands to sum_t mono_t * prod (1 + d_i) with at
// most k = 3 * 10 + 5 + 5 + 1 + 19 = 60 factors |d_i| <= u per monomial (the
// first sum of a running total adds to zero and is exact, so it is not counted).
// Hence |fl(ctrl) - ctrl| <= gamma_60 * A with A = sum_t |mono_t|, the same
// tree on |w x| with every subtraction replaced by an addition. Computing A in
// binary64 only multiplies each nonnegative monomial by >= (1 - u)^60, so
// A <= fl(A) / (1 - u)^60, and fl(2^-47 fl(A)) loses one more (1 - u):
// 2^-47 = 64 u >= gamma_60 / (1 - u)^61 = 60 u (1 + O(120 u)). The power
// coefficients use the same constant (k = 40 there).
// Underflow would break the relative model, so every computed Jacobian entry
// and its magnitude sum must be 0 or lie in [2^-300, 2^300]: w is a nonzero
// integer, so fl(w x) is exact when subnormal; products of entries are normal
// (>= 2^-600); a nonzero difference of such products is a multiple of
// 2^-652, times a factor >= 2^-300 is >= 2^-952, and every later nonzero sum
// of multiples of 2^-1004 stays normal; magnitudes stay below 2^906, so no
// overflow. Sums of binary64 numbers are exact when subnormal.
// The relative threshold is bounded above with next_up after every
// nonnegative operation; a Jacobian corner J0 + Jv is a 10-node sum plus one
// addition (k = 11), so |corner| <= |fl(J0 + Jv)| + 2^-47 fl(A0 + Av).
// A control c is certified with c >= fl(c) - B > 0 and
// c^2 >= next_down(L * L) for L = next_down(fl(c) - B). The exact path uses
// the controls of the actual degree d; d is only known to lie between the
// highest certified-nonzero and the highest nonzero-magnitude coefficient
// degree, so the controls of every such degree must pass. Passing makes the
// exact root cell VALID (corner and edge controls are among the controls).
constexpr double kScreenRelative = 0x1p-47;
constexpr double kScreenTiny = 0x1p-300;
constexpr double kScreenHuge = 0x1p300;

// Upper bound of |J0 + Jv| from the computed sum and its magnitude sum.
double screen_corner_bound(double value, double magnitude) noexcept {
  return next_up(std::fabs(value) + next_up(kScreenRelative * next_up(magnitude)));
}

bool screen_range(double value) noexcept {
  const double magnitude = std::fabs(value);
  return magnitude == 0.0 || (magnitude >= kScreenTiny && magnitude <= kScreenHuge);
}

struct ScreenTerm final {
  int beta;
  double multiplier;
};

// Bernstein controls (times 6) of degrees 0..3 as terms over the power
// coefficients, the formula of determinant_bernstein_controls: the controls of
// degree d are indexed by the alpha with |alpha| <= d (alpha_0 = d - |alpha|).
struct ScreenTable final {
  std::array<int, 5> control_offsets{};  // degree d: controls [offsets[d], offsets[d + 1])
  std::array<int, 36> term_offsets{};
  std::array<ScreenTerm, 35 * 20> terms{};
};

const ScreenTable& screen_table() {
  static const ScreenTable table = [] {
    ScreenTable built;
    int control = 0;
    int term = 0;
    for (int degree = 0; degree <= 3; ++degree) {
      built.control_offsets[degree] = control;
      for (int alpha = 0; alpha < bernstein_count(degree); ++alpha) {
        built.term_offsets[control++] = term;
        for (int beta = 0; beta < bernstein_count(degree); ++beta) {
          const Power& exponent = kPowers[beta];
          const int multiplier = (6 / falling(degree, exponent.degree)) *
                                 falling(kPowers[alpha].x, exponent.x) *
                                 falling(kPowers[alpha].y, exponent.y) *
                                 falling(kPowers[alpha].z, exponent.z);
          if (multiplier != 0) built.terms[term++] = {beta, static_cast<double>(multiplier)};
        }
      }
    }
    built.control_offsets[4] = control;
    built.term_offsets[control] = term;
    return built;
  }();
  return table;
}

}  // namespace

bool quadratic_affine_neighborhood_screen(const double* nodes, double relative_floor) noexcept {
  double magnitude = 0.0;
  for (int i = 0; i < 30; ++i) magnitude = std::max(magnitude, std::abs(nodes[i]));
  if (!(magnitude >= 0x1p-100 && magnitude <= 0x1p100)) return false;

  // J = A + E(x), with A the binary64 corner-edge columns. For an edge
  // correction delta the derivative of 4*l_i*l_j has coefficient l1 norm
  // at most 20 in any column: l_0 has norm 4 and the others have norm 1.
  // Thus E has coefficient-vector l1 norm <= 20 times the sum of edge
  // residual l1 norms, plus the error in A's subtractions. The factors of
  // two cover accumulation; the absolute pads dominate the <100 binary64
  // operations' rounding (including gradual underflow) at the admitted scale.
  constexpr int edges[6][2]{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  double residual = 0.0;
  for (int e = 0; e < 6; ++e) {
    for (int axis = 0; axis < 3; ++axis) {
      const double middle = 0.5 * (nodes[3 * edges[e][0] + axis] +
                                   nodes[3 * edges[e][1] + axis]);
      residual += std::abs(nodes[12 + 3 * e + axis] - middle);
    }
  }
  const double e = geometry::next_up(40.0 * residual + 0x1p-38 * magnitude);
  double columns[3][3], norms[3];
  for (int column = 0; column < 3; ++column) {
    geometry::difference(nodes + 3 * (column + 1), nodes, columns[column]);
    const double* a = columns[column];
    norms[column] = 2 * (std::abs(a[0]) + std::abs(a[1]) + std::abs(a[2]));
  }

  // Multilinearity and Hadamard bound the l1 sum of the determinant's
  // perturbation coefficients. Conversion to a Bernstein control of any
  // admissible actual degree multiplies each monomial by a number in [0,1].
  // The same error therefore covers *all* root controls, not just pointwise
  // positivity of the exact map. The scale pad also bounds error in det(A).
  const auto [a, b, c] = norms;
  const double error = 2 * e * (a * b + a * c + b * c + e * (a + b + c + e)) +
                       0x1p-40 * magnitude * magnitude * magnitude;
  double cross[3];
  geometry::cross(columns[1], columns[2], cross);
  const double lower = geometry::next_down(geometry::dot(columns[0], cross) - error);
  if (!(lower > 0.0)) return false;
  const double bound = geometry::next_up(
      geometry::next_up(geometry::next_up(a + e) * geometry::next_up(b + e)) *
      geometry::next_up(c + e));
  return lower > geometry::next_up(relative_floor * bound);
}

bool quadratic_validity_screen(const double* nodes, const QuadraticDerivativeStencil& stencil,
                               double relative_floor) noexcept {
  double value[9][4];
  double magnitude[9][4];
  for (int column = 0; column < 3; ++column) {
    for (int power = 0; power < 4; ++power) {
      double sum[3] = {};
      double absolute[3] = {};
      for (const auto& term : stencil.terms(column, power)) {
        for (int row = 0; row < 3; ++row) {
          const double product = term.weight * nodes[term.coordinate_offset + row];
          sum[row] += product;
          absolute[row] += std::fabs(product);
        }
      }
      for (int row = 0; row < 3; ++row) {
        if (!screen_range(sum[row]) || !screen_range(absolute[row])) return false;
        value[row * 3 + column][power] = sum[row];
        magnitude[row * 3 + column][power] = absolute[row];
      }
    }
  }

  double coefficient[20] = {};
  double coefficient_magnitude[20] = {};
  for (int first = 0; first < 4; ++first) {
    for (int second = 0; second < 4; ++second) {
      for (int third = 0; third < 4; ++third) {
        const double a0 = value[0][first], a1 = value[1][second], a2 = value[2][third];
        const double b0 = value[3][first], b1 = value[4][second], b2 = value[5][third];
        const double c0 = value[6][first], c1 = value[7][second], c2 = value[8][third];
        const double p0 = magnitude[0][first], p1 = magnitude[1][second], p2 = magnitude[2][third];
        const double q0 = magnitude[3][first], q1 = magnitude[4][second], q2 = magnitude[5][third];
        const double r0 = magnitude[6][first], r1 = magnitude[7][second], r2 = magnitude[8][third];
        const int index = power_index(kPowers[first].x + kPowers[second].x + kPowers[third].x,
                                      kPowers[first].y + kPowers[second].y + kPowers[third].y,
                                      kPowers[first].z + kPowers[second].z + kPowers[third].z);
        coefficient[index] +=
            a0 * (b1 * c2 - b2 * c1) - a1 * (b0 * c2 - b2 * c0) + a2 * (b0 * c1 - b1 * c0);
        coefficient_magnitude[index] +=
            p0 * (q1 * r2 + q2 * r1) + p1 * (q0 * r2 + q2 * r0) + p2 * (q0 * r1 + q1 * r0);
      }
    }
  }

  int lowest = 0;
  int highest = 0;
  for (int index = 0; index < 20; ++index) {
    if (coefficient_magnitude[index] != 0.0) highest = std::max(highest, kPowers[index].degree);
    if (std::fabs(coefficient[index]) > kScreenRelative * coefficient_magnitude[index]) {
      lowest = std::max(lowest, kPowers[index].degree);
    }
  }

  double threshold = 0.0;
  if (relative_floor > 0.0) {
    double hadamard = 1.0;
    for (int column = 0; column < 3; ++column) {
      double norm = 0.0;
      for (int row = 0; row < 3; ++row) {
        const double* entry = value[row * 3 + column];
        const double* entry_magnitude = magnitude[row * 3 + column];
        double largest = screen_corner_bound(entry[0], entry_magnitude[0]);
        for (int vertex = 1; vertex < 4; ++vertex) {
          largest = std::max(largest, screen_corner_bound(entry[0] + entry[vertex],
                                                          entry_magnitude[0] +
                                                              entry_magnitude[vertex]));
        }
        norm = next_up(norm + next_up(largest * largest));
      }
      hadamard = next_up(hadamard * norm);
    }
    threshold = next_up(next_up(next_up(relative_floor * relative_floor) * 36.0) * hadamard);
  }

  const ScreenTable& table = screen_table();
  for (int degree = lowest; degree <= highest; ++degree) {
    for (int control = table.control_offsets[degree];
         control < table.control_offsets[degree + 1]; ++control) {
      double sum = 0.0;
      double absolute = 0.0;
      for (int term = table.term_offsets[control]; term < table.term_offsets[control + 1];
           ++term) {
        const ScreenTerm& entry = table.terms[term];
        sum += entry.multiplier * coefficient[entry.beta];
        absolute += entry.multiplier * coefficient_magnitude[entry.beta];
      }
      const double error = kScreenRelative * absolute;
      if (!(sum > error)) return false;
      if (threshold > 0.0) {
        const double lower = next_down(sum - error);
        if (!(next_down(lower * lower) > threshold)) return false;
      }
    }
  }
  return true;
}

}  // namespace phx::mc

extern "C" {

int32_t phx_mc_simplex_quadratic_bounds(int64_t count, int32_t dimension, int32_t components,
                                        const double* q_lower, const double* q_upper,
                                        const double* barycentric_lower,
                                        const double* barycentric_upper, double* lower,
                                        double* upper) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (dimension < 1 || dimension > 3 || components < 1 || components > 3 || count < 0) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const int m = dimension + 1;
    const int64_t q_width = static_cast<int64_t>(components) * m * m;
    const int64_t l_width = static_cast<int64_t>(m) * m;
    if (!phx::mc::addressable(count, q_width, sizeof(double)) ||
        !phx::mc::addressable(count, components, sizeof(double)) ||
        !phx::mc::addressable(count, l_width, sizeof(double))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (count > 0 && (q_lower == nullptr || q_upper == nullptr || barycentric_lower == nullptr ||
                      barycentric_upper == nullptr || lower == nullptr || upper == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const int32_t validation = validate_quadratic_bounds(
        count, m, components, q_lower, q_upper, barycentric_lower, barycentric_upper);
    if (validation != PHX_MC_OK) return validation;
    for (int64_t item = 0; item < count; ++item) {
      enclose_item(m, components, q_lower + item * q_width, q_upper + item * q_width,
                   barycentric_lower + item * l_width, barycentric_upper + item * l_width,
                   lower + item * components, upper + item * components);
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_quadratic_tetrahedron_validity(int64_t count, const double* nodes,
                                              const int32_t* derivative_weights,
                                              double relative_floor, int32_t* decisions,
                                              double* lower, double* upper) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    const int32_t validation = validate_validity_inputs(
        count, nodes, derivative_weights, relative_floor, decisions, lower, upper);
    if (validation != PHX_MC_OK) return validation;
    // Without requested bounds only verdicts are produced, so the binary64
    // screen may decide VALID; every other cell takes the exact path.
    const bool bounds = lower != nullptr;
    std::optional<phx::mc::QuadraticDerivativeStencil> stencil;
    if (!bounds && count > 0) stencil.emplace(derivative_weights);
    std::vector<ValidityResult> staged;
    staged.reserve(static_cast<std::size_t>(count));
    for (int64_t cell = 0; cell < count; ++cell) {
      const double* cell_nodes = nodes + 30 * cell;
      if (stencil && phx::mc::quadratic_validity_screen(cell_nodes, *stencil, relative_floor)) {
        staged.push_back({kValidityValid, 0.0, 0.0});
      } else {
        staged.push_back(evaluate_validity_cell(cell_nodes, derivative_weights, relative_floor));
      }
    }
    for (int64_t cell = 0; cell < count; ++cell) {
      const ValidityResult& result = staged[static_cast<std::size_t>(cell)];
      decisions[cell] = result.decision;
      if (!bounds) continue;
      lower[cell] = result.lower;
      upper[cell] = result.upper;
    }
    return PHX_MC_OK;
  });
}

}  // extern "C"
