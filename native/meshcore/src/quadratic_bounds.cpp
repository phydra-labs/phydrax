//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// C ABI: outward interval enclosures of degree-two Bernstein forms on child
// simplices. A root form f(lambda) = lambda^T Q lambda restricted to the child
// whose vertex k has root barycentric row L_k has child Bernstein coefficients
// Q' = L Q L^T, and the child image lies in [min Q'_kl, max Q'_kl] (k <= l).
// Outward rounding and the exact row-sum expansions require round-to-nearest
// with gradual underflow, so validation and every chunk run in a
// GradualNearestScope on their own thread and refuse with INTERNAL_ERROR when
// the probe shows the mode was not established; the caller's mode returns.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <utility>

#include "capi_guard.hpp"
#include "dyadic.hpp"
#include "expansion.hpp"
#include "exact_scalar.hpp"
#include "floating_point_mode.hpp"
#include "parallel.hpp"
#include "phydrax_meshcore.h"
#include "rounding.hpp"

namespace {

using phx::mc::Dyadic;
using phx::mc::NativeVector;

struct ValidityResult final {
  std::int32_t decision = 0;
  double lower = 0.0;
  double upper = 0.0;
};

struct ValidityEvidence final {
  std::int64_t work = 0;
  std::int64_t queries = 0;
  std::int64_t peak_bytes = 0;
  std::int64_t completed = 0;
  std::int64_t failing_cell = -1;
  std::int64_t failure_category = 1;
};

// Work is charged before each exact scalar operation or bounded polynomial
// visit. A multi-unit charge is used only for a fixed group that is executed
// immediately after its admission.
class ValidityWork final {
 public:
  ValidityWork(std::int64_t maximum, std::int64_t& work) noexcept
      : maximum_(maximum), work_(work) {}

  void charge(std::int64_t amount = 1) {
    if (amount < 0 || amount > maximum_ - work_) {
      throw phx::mc::ExecutionRefusal{
          phx::mc::native_execution_status(PHX_MC_CAPACITY_EXCEEDED)};
    }
    phx::mc::native_execution_charge(amount);
    work_ += amount;
  }

  Dyadic add(const Dyadic& first, const Dyadic& second) {
    charge();
    return first + second;
  }
  Dyadic subtract(const Dyadic& first, const Dyadic& second) {
    charge();
    return first - second;
  }
  Dyadic multiply(const Dyadic& first, const Dyadic& second) {
    charge();
    return first * second;
  }
  Dyadic decode(double value) {
    charge();
    return Dyadic(value);
  }

 private:
  std::int64_t maximum_;
  std::int64_t& work_;
};

struct Power final {
  int x;
  int y;
  int z;
  int degree;
};

// The complete power basis through degree three, in deterministic graded order.
constexpr std::array<Power, 20> kPowers{{
    {0, 0, 0, 0},
    {1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1},
    {2, 0, 0, 2}, {1, 1, 0, 2}, {1, 0, 1, 2},
    {0, 2, 0, 2}, {0, 1, 1, 2}, {0, 0, 2, 2},
    {3, 0, 0, 3}, {2, 1, 0, 3}, {2, 0, 1, 3},
    {1, 2, 0, 3}, {1, 1, 1, 3}, {1, 0, 2, 3},
    {0, 3, 0, 3}, {0, 2, 1, 3}, {0, 1, 2, 3}, {0, 0, 3, 3},
}};

int power_index(int x, int y, int z) {
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

int falling(int value, int order) {
  if (order > value) return 0;
  int result = 1;
  for (int factor = 0; factor < order; ++factor) result *= value - factor;
  return result;
}

int compare_absolute(ValidityWork& work, const Dyadic& first, const Dyadic& second) {
  const int first_sign = first.sign();
  const int second_sign = second.sign();
  if (first_sign == 0) return second_sign == 0 ? 0 : -1;
  if (second_sign == 0) return 1;
  if (first_sign == second_sign) {
    return first_sign > 0 ? work.subtract(first, second).sign()
                          : work.subtract(second, first).sign();
  }
  const int sum_sign = work.add(first, second).sign();
  return first_sign < 0 ? -sum_sign : sum_sign;
}


ValidityResult evaluate_validity_cell(const double* nodes, const std::int32_t* derivative_weights,
                                      double relative_floor, std::int64_t maximum_bernstein_nodes,
                                      std::int64_t maximum_subdivision_depth,
                                      ValidityWork& work) {
  // J[row,column,power], with powers 1,x,y,z. Every template entry is visited;
  // exact-zero derivative weights are skipped only after their visit charge.
  std::array<Dyadic, 9 * 4> jacobian;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      for (int power = 0; power < 4; ++power) {
        Dyadic sum;
        work.charge(10);
        for (int node = 0; node < 10; ++node) {
          const std::int32_t weight = derivative_weights[(column * 4 + power) * 10 + node];
          if (weight == 0) continue;
          work.charge();
          Dyadic term(nodes[node * 3 + row]);
          if (weight == -1) {
            work.charge();
            term.negate();
          } else if (weight != 1) {
            const Dyadic exact_weight = work.decode(static_cast<double>(weight));
            term = work.multiply(term, exact_weight);
          }
          if (term.sign() == 0) continue;
          if (sum.sign() == 0) {
            sum = std::move(term);
          } else {
            sum = work.add(sum, term);
          }
        }
        jacobian[(row * 3 + column) * 4 + power] = std::move(sum);
      }
    }
  }

  // Multilinearity in the three columns shares the canonical exact 3x3
  // determinant instead of reimplementing its six signed products.
  std::array<std::array<bool, 4>, 3> active{};
  work.charge(36);
  for (int column = 0; column < 3; ++column) {
    for (int power = 0; power < 4; ++power) {
      active[column][power] = jacobian[column * 4 + power].sign() != 0 ||
                             jacobian[(3 + column) * 4 + power].sign() != 0 ||
                             jacobian[(6 + column) * 4 + power].sign() != 0;
    }
  }
  std::array<Dyadic, 20> power_coefficients;
  for (int first = 0; first < 4; ++first) {
    work.charge();
    if (!active[0][first]) continue;
    for (int second = 0; second < 4; ++second) {
      work.charge();
      if (!active[1][second]) continue;
      for (int third = 0; third < 4; ++third) {
        work.charge();
        if (!active[2][third]) continue;
        work.charge(14);
        Dyadic value = phx::mc::exact::det3(
            jacobian[first], jacobian[4 + second], jacobian[8 + third],
            jacobian[12 + first], jacobian[16 + second], jacobian[20 + third],
            jacobian[24 + first], jacobian[28 + second], jacobian[32 + third]);
        if (value.sign() == 0) continue;
        const int x = kPowers[first].x + kPowers[second].x + kPowers[third].x;
        const int y = kPowers[first].y + kPowers[second].y + kPowers[third].y;
        const int z = kPowers[first].z + kPowers[second].z + kPowers[third].z;
        Dyadic& coefficient = power_coefficients[power_index(x, y, z)];
        coefficient = coefficient.sign() == 0 ? std::move(value) : work.add(coefficient, value);
      }
    }
  }

  work.charge(static_cast<std::int64_t>(power_coefficients.size()));
  int actual_degree = 0;
  for (std::size_t index = 0; index < power_coefficients.size(); ++index) {
    if (power_coefficients[index].sign() != 0) {
      actual_degree = std::max(actual_degree, kPowers[index].degree);
    }
  }

  work.charge(9 * 4);
  int maximum_entry_nodes = 1;
  for (int entry = 0; entry < 9; ++entry) {
    bool nonconstant = false;
    for (int power = 1; power < 4; ++power) {
      nonconstant = nonconstant || jacobian[entry * 4 + power].sign() != 0;
    }
    if (nonconstant) maximum_entry_nodes = 4;
  }
  constexpr std::array<int, 4> kBernsteinNodeCounts{{1, 4, 10, 20}};
  const int determinant_nodes = kBernsteinNodeCounts[actual_degree];
  if (std::max(maximum_entry_nodes, determinant_nodes) > maximum_bernstein_nodes) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {3, nan, nan};
  }

  const int control_count = determinant_nodes;
  std::array<Power, 20> alphas{};
  int alpha_count = 0;
  for (int x = 0; x <= actual_degree; ++x) {
    for (int y = 0; y <= actual_degree - x; ++y) {
      for (int z = 0; z <= actual_degree - x - y; ++z) {
        alphas[alpha_count++] = {x, y, z, x + y + z};
      }
    }
  }

  std::array<Dyadic, 20> controls;
  for (int alpha = 0; alpha < alpha_count; ++alpha) {
    work.charge(control_count);
    Dyadic control;
    bool has_term = false;
    for (int beta = 0; beta < control_count; ++beta) {
      const Power& exponent = kPowers[beta];
      const int denominator_factor = falling(actual_degree, exponent.degree);
      const int multiplier =
          (6 / denominator_factor) * falling(alphas[alpha].x, exponent.x) *
          falling(alphas[alpha].y, exponent.y) * falling(alphas[alpha].z, exponent.z);
      const Dyadic& coefficient = power_coefficients[beta];
      if (multiplier == 0 || coefficient.sign() == 0) continue;
      Dyadic value;
      if (multiplier == 1) {
        work.charge();
        value = coefficient;
      } else {
        const Dyadic exact_multiplier =
            work.decode(static_cast<double>(multiplier));
        value = work.multiply(coefficient, exact_multiplier);
      }
      if (has_term) {
        control = work.add(control, value);
      } else {
        control = std::move(value);
        has_term = true;
      }
    }
    controls[alpha] = std::move(control);
  }

  work.charge(control_count);
  int minimum_index = 0, maximum_index = 0;
  for (int index = 1; index < control_count; ++index) {
    if (work.subtract(controls[index], controls[minimum_index]).sign() < 0) {
      minimum_index = index;
    }
    if (work.subtract(controls[index], controls[maximum_index]).sign() > 0) {
      maximum_index = index;
    }
  }

  // Exact corner magnitudes of the affine Jacobian entries feed the squared
  // Hadamard scale; a corner already equal to its constant term reuses it.
  std::array<Dyadic, 9> squared_maxima;
  for (int entry = 0; entry < 9; ++entry) {
    const Dyadic& constant = jacobian[entry * 4];
    std::array<Dyadic, 3> sums;
    const Dyadic* corners[4] = {&constant, nullptr, nullptr, nullptr};
    for (int power = 1; power < 4; ++power) {
      work.charge();
      const Dyadic& coefficient = jacobian[entry * 4 + power];
      if (coefficient.sign() == 0) {
        corners[power] = &constant;
      } else if (constant.sign() == 0) {
        corners[power] = &coefficient;
      } else {
        sums[power - 1] = work.add(constant, coefficient);
        corners[power] = &sums[power - 1];
      }
    }
    const Dyadic* maximum = corners[0];
    for (int corner = 1; corner < 4; ++corner) {
      work.charge();
      if (corners[corner] != maximum &&
          compare_absolute(work, *corners[corner], *maximum) > 0) {
        maximum = corners[corner];
      }
    }
    squared_maxima[entry] = work.multiply(*maximum, *maximum);
  }

  std::array<Dyadic, 3> column_norms;
  for (int column = 0; column < 3; ++column) {
    const Dyadic first = std::move(squared_maxima[column]);
    const Dyadic second = std::move(squared_maxima[3 + column]);
    const Dyadic third = std::move(squared_maxima[6 + column]);
    column_norms[column] = work.add(work.add(first, second), third);
  }
  const Dyadic first_two_norms = work.multiply(column_norms[0], column_norms[1]);
  const Dyadic squared_scale = work.multiply(first_two_norms, column_norms[2]);

  const Dyadic floor = work.decode(relative_floor);
  const Dyadic floor_squared = work.multiply(floor, floor);
  const Dyadic thirty_six = work.decode(36.0);
  const Dyadic scaled_floor = work.multiply(floor_squared, thirty_six);
  const Dyadic threshold36 = work.multiply(scaled_floor, squared_scale);

  const int n = actual_degree;
  const int first_corner = 0;
  int x_corner = -1, y_corner = -1, z_corner = -1;
  work.charge(3 * control_count);
  for (int index = 0; index < control_count; ++index) {
    const Power& alpha = alphas[index];
    if (alpha.x == n && alpha.y == 0 && alpha.z == 0) x_corner = index;
    if (alpha.x == 0 && alpha.y == n && alpha.z == 0) y_corner = index;
    if (alpha.x == 0 && alpha.y == 0 && alpha.z == n) z_corner = index;
  }
  const int corner_indices[4] = {first_corner, x_corner, y_corner, z_corner};
  bool invalid = false;
  const int corner_count = n == 0 ? 1 : 4;
  for (int corner_index = 0; corner_index < corner_count; ++corner_index) {
    const int corner = corner_indices[corner_index];
    const Dyadic& value = controls[corner];
    work.charge();
    if (value.sign() <= 0) {
      invalid = true;
      break;
    }
    const Dyadic square = work.multiply(value, value);
    if (work.subtract(square, threshold36).sign() < 0) {
      invalid = true;
      break;
    }
  }

  std::int32_t decision = 0;
  if (invalid) {
    decision = 2;
  } else if (controls[minimum_index].sign() > 0) {
    const Dyadic minimum_squared =
        work.multiply(controls[minimum_index], controls[minimum_index]);
    if (work.subtract(minimum_squared, threshold36).sign() > 0) {
      decision = 1;
    } else {
      decision = maximum_subdivision_depth == 0 ? 4 : 0;
    }
  } else {
    decision = maximum_subdivision_depth == 0 ? 4 : 0;
  }

  const Dyadic six = work.decode(6.0);
  const phx::mc::exact::QuotientRoundingWork rounding{
      nullptr, &work, [](void* context, std::int64_t operations) {
        static_cast<ValidityWork*>(context)->charge(operations);
      }};
  const auto lower_bounds = phx::mc::exact::directed_quotient_bounds(
      controls[minimum_index], six, rounding);
  const auto upper_bounds = minimum_index == maximum_index
                                ? lower_bounds
                                : phx::mc::exact::directed_quotient_bounds(
                                      controls[maximum_index], six, rounding);
  return {decision, lower_bounds.lower, upper_bounds.upper};
}

int32_t validate_validity_inputs(std::int64_t count, const double* nodes,
                                 const std::int32_t* derivative_weights, double relative_floor,
                                 std::int64_t maximum_bernstein_nodes,
                                 std::int64_t maximum_subdivision_depth,
                                 std::int64_t maximum_work_units,
                                 std::int64_t maximum_scratch_bytes, std::int32_t* decisions,
                                 double* lower, double* upper, std::int64_t* failing_cell,
                                 ValidityWork& work) {
  if (count < 0 || maximum_bernstein_nodes <= 0 || maximum_subdivision_depth < 0 ||
      maximum_work_units < 0 || maximum_scratch_bytes < 0 ||
      static_cast<std::uint64_t>(maximum_scratch_bytes) >
          std::numeric_limits<std::size_t>::max() ||
      !std::isfinite(relative_floor) || relative_floor < 0.0 || relative_floor >= 1.0 ||
      !phx::mc::addressable(count, 30, sizeof(double)) ||
      !phx::mc::addressable(count, 1, sizeof(ValidityResult)) ||
      !phx::mc::addressable(count, 1, sizeof(std::int32_t)) ||
      !phx::mc::addressable(count, 1, sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (count > 0 &&
      (nodes == nullptr || derivative_weights == nullptr || decisions == nullptr ||
       lower == nullptr || upper == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }

  constexpr std::int64_t kValidationChunkScalars = 1024;
  const std::int64_t scalar_count = count * 30;
  for (std::int64_t begin = 0; begin < scalar_count; begin += kValidationChunkScalars) {
    phx::mc::native_execution_checkpoint();
    const std::int64_t end = std::min(scalar_count, begin + kValidationChunkScalars);
    *failing_cell = begin / 30;
    work.charge(end - begin);
    std::int64_t first_nonfinite = -1;
    for (std::int64_t scalar = begin; scalar < end; ++scalar) {
      const bool finite = std::isfinite(nodes[scalar]);
      if (!finite && first_nonfinite < 0) first_nonfinite = scalar / 30;
    }
    phx::mc::native_execution_checkpoint();
    if (first_nonfinite >= 0) {
      *failing_cell = first_nonfinite;
      return PHX_MC_NONFINITE_INPUT;
    }
    *failing_cell = -1;
  }
  if (scalar_count > 0) phx::mc::native_execution_checkpoint();
  return PHX_MC_OK;
}

void publish_validity_evidence(const ValidityEvidence& state, std::int64_t* evidence) noexcept {
  evidence[0] = state.work;
  evidence[1] = state.queries;
  evidence[2] = state.peak_bytes;
  evidence[3] = state.completed;
  evidence[4] = state.failing_cell;
  evidence[5] = state.failure_category;
}

}  // namespace
namespace {

using phx::mc::geometry::next_down;
using phx::mc::geometry::next_up;

constexpr int kMaximumVertices = 4;  // m = dimension + 1 with dimension <= 3.
constexpr double kInfinity = std::numeric_limits<double>::infinity();

// Validation polls the execution scope once per fixed chunk of items (the
// parallel region's chunk) before the chunk is read and again after the scan,
// like coordinate validation. The ledger is never locked per item.
constexpr int64_t kValidationChunkItems =
    static_cast<int64_t>(phx::mc::ParallelRequest{}.chunk_size);

// Logical work of one item: one unit per interval product, m^3 for T = L Q
// and m for each of the m (m + 1) / 2 upper-triangular coefficients of T L^T,
// for every component. Fixed by (dimension, components) alone.
uint64_t item_work(int m, int components) {
  const auto vertices = static_cast<uint64_t>(m);
  return static_cast<uint64_t>(components) *
         (vertices * vertices * vertices + vertices * vertices * (vertices + 1) / 2);
}

// Outward endpoint arithmetic in round-to-nearest binary64. One adjacent value
// outward encloses every rounded product or sum, including gradual underflow
// and overflow to infinity. The scale x is a barycentric endpoint in [0, 1]:
// x == 0 is exact even against an infinite endpoint (endpoints bound finite
// reals), x == 1 is exact, and |x y| <= |y| never overflows. A zero addend or
// a zero sum is exact. Lower endpoints are never +inf and upper endpoints never
// -inf, so no operation forms inf - inf or 0 * inf and no NaN is produced.
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

// Bounds of x y over x in [x_lower, x_upper] with 0 <= x_lower and y in
// [y_lower, y_upper]: the extreme y endpoint, scaled by the x endpoint that
// matches its sign.
double product_lower(double x_lower, double x_upper, double y_lower) noexcept {
  return scale_down(y_lower >= 0.0 ? x_lower : x_upper, y_lower);
}

double product_upper(double x_lower, double x_upper, double y_upper) noexcept {
  return scale_up(y_upper >= 0.0 ? x_upper : x_lower, y_upper);
}

// One item: q_* (components, m, m), l_* (m, m), outputs (components,).
// Every row and term is visited in increasing index order.
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
    // Q'_kl = sum_j T_kj L_lj for k <= l.
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

// Exact sign of (sum of m values in [0, 1]) - 1; at most five components, so
// the expansions stay in inline storage.
int sum_minus_one_sign(const double* values, int m) {
  phx::mc::Expansion total(values[0]);
  for (int index = 1; index < m; ++index) total = total + phx::mc::Expansion(values[index]);
  return (total - phx::mc::Expansion(1.0)).sign();
}

// A coefficient lower endpoint may be -inf and an upper endpoint +inf (an
// unbounded side); NaN or a reversed infinity is nonfinite input. Barycentric
// endpoints must be finite. Every entry of every item is checked; a nonfinite
// value anywhere outranks an invalid one, which is reported only once the
// whole batch is known finite.
int32_t validate(int64_t count, int m, int components, const double* q_lower,
                 const double* q_upper, const double* l_lower, const double* l_upper) {
  const int64_t q_width = static_cast<int64_t>(components) * m * m;
  const int64_t l_width = static_cast<int64_t>(m) * m;
  bool invalid = false;
  for (int64_t begin = 0; begin < count; begin += kValidationChunkItems) {
    phx::mc::native_execution_checkpoint();
    const int64_t end = std::min(count, begin + kValidationChunkItems);
    bool nonfinite = false;
    bool rejected = false;
    for (int64_t item = begin; item < end; ++item) {
      const double* ql = q_lower + item * q_width;
      const double* qu = q_upper + item * q_width;
      for (int64_t index = 0; index < q_width; ++index) {
        nonfinite = nonfinite || std::isnan(ql[index]) || ql[index] == kInfinity ||
                    std::isnan(qu[index]) || qu[index] == -kInfinity;
        rejected = rejected || !(ql[index] <= qu[index]);
      }
      for (int component = 0; component < components; ++component) {
        const double* cl = ql + component * m * m;
        const double* cu = qu + component * m * m;
        for (int i = 0; i < m; ++i) {
          for (int j = i + 1; j < m; ++j) {
            rejected = rejected || cl[i * m + j] != cl[j * m + i] || cu[i * m + j] != cu[j * m + i];
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
        rejected = rejected || !row_ordered ||
                   sum_minus_one_sign(ll + k * m, m) > 0 || sum_minus_one_sign(lu + k * m, m) < 0;
      }
    }
    if (nonfinite) {
      phx::mc::native_execution_checkpoint();
      return PHX_MC_NONFINITE_INPUT;
    }
    invalid = invalid || rejected;
  }
  if (count > 0) phx::mc::native_execution_checkpoint();
  return invalid ? PHX_MC_INVALID_INPUT : PHX_MC_OK;
}


int32_t quadratic_tetrahedron_validity(
    int64_t count, const double* nodes, const int32_t* derivative_weights,
    double relative_floor, int64_t maximum_bernstein_nodes,
    int64_t maximum_subdivision_depth, int64_t maximum_work_units,
    int64_t maximum_scratch_bytes, int32_t* decisions, double* lower, double* upper,
    int64_t* evidence) {
  if (evidence == nullptr) return PHX_MC_INVALID_ARGUMENT;

  ValidityEvidence state;
  int32_t status = PHX_MC_OK;
  int64_t current_cell = -1;
  std::optional<phx::mc::MemoryBudgetWindow> scratch;
  bool publishing = false;
  try {
    ValidityWork work(maximum_work_units, state.work);
    const int32_t validation = validate_validity_inputs(
        count, nodes, derivative_weights, relative_floor, maximum_bernstein_nodes,
        maximum_subdivision_depth, maximum_work_units, maximum_scratch_bytes, decisions, lower,
        upper, &state.failing_cell, work);
    if (validation != PHX_MC_OK) {
      status = validation;
    } else {
      const phx::mc::GradualNearestScope mode;
      if (!mode.established()) {
        status = PHX_MC_INTERNAL_ERROR;
        state.failure_category = 2;
      } else {
        phx::mc::native_execution_checkpoint();
        if (count > 0) {
          scratch.emplace(static_cast<std::size_t>(maximum_scratch_bytes));
          {
            phx::mc::MemoryScope memory_scope(scratch->owner());
            NativeVector<ValidityResult> staged;
            staged.reserve(static_cast<std::size_t>(count));
            for (int64_t cell = 0; cell < count; ++cell) {
              current_cell = cell;
              phx::mc::native_execution_primitive_query();
              ++state.queries;
              staged.push_back(evaluate_validity_cell(
                  nodes + 30 * cell, derivative_weights, relative_floor,
                  maximum_bernstein_nodes, maximum_subdivision_depth, work));
              ++state.completed;
              current_cell = -1;
            }
            const auto memory = scratch->evidence();
            state.peak_bytes = static_cast<std::int64_t>(memory[2]);
            publishing = true;
            phx::mc::native_execution_checkpoint();
            work.charge(3 * count);
            for (int64_t cell = 0; cell < count; ++cell) {
              const ValidityResult& result = staged[static_cast<std::size_t>(cell)];
              decisions[cell] = result.decision;
              lower[cell] = result.lower;
              upper[cell] = result.upper;
            }
            publishing = false;
          }
        }
      }
    }
  } catch (const phx::mc::ExecutionRefusal& refusal) {
    status = refusal.status;
    state.failure_category = publishing ? 3 : 2;
    if (current_cell >= 0) state.failing_cell = current_cell;
  } catch (const std::bad_alloc&) {
    status = phx::mc::native_execution_status(PHX_MC_CAPACITY_EXCEEDED);
    state.failure_category = publishing ? 3 : 2;
    if (current_cell >= 0) state.failing_cell = current_cell;
  } catch (...) {
    status = PHX_MC_INTERNAL_ERROR;
    state.failure_category = publishing ? 3 : 2;
    if (current_cell >= 0) state.failing_cell = current_cell;
  }

  if (scratch.has_value() && status != PHX_MC_OK) {
    try {
      const auto memory = scratch->evidence();
      state.peak_bytes = static_cast<std::int64_t>(memory[2]);
    } catch (...) {
      // Keep the original native refusal status; the bounded peak was not
      // observable through the allocator's evidence lock.
    }
  }
  if (status == PHX_MC_OK) {
    state.failure_category = 0;
    state.failing_cell = -1;
  } else if (state.failure_category == 1 && state.failing_cell == -1 &&
             current_cell >= 0) {
    state.failing_cell = current_cell;
  }
  publish_validity_evidence(state, evidence);
  return status;
}

}  // namespace

extern "C" {

int32_t phx_mc_simplex_quadratic_bounds(int64_t count, int32_t dimension, int32_t components,
                                        const double* q_lower, const double* q_upper,
                                        const double* barycentric_lower,
                                        const double* barycentric_upper, double* lower,
                                        double* upper) {
  return phx::mc::guarded([&]() -> int32_t {
    if (dimension < 1 || dimension > 3 || components < 1 || components > 3) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const int m = dimension + 1;
    const int64_t q_width = static_cast<int64_t>(components) * m * m;
    // The barycentric rows (m * m) are never wider than the coefficient rows.
    if (count < 0 || lower == nullptr || upper == nullptr ||
        !phx::mc::addressable(count, q_width, sizeof(double))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (count > 0 && (q_lower == nullptr || q_upper == nullptr ||
                      barycentric_lower == nullptr || barycentric_upper == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    int32_t status = PHX_MC_OK;
    {
      const phx::mc::GradualNearestScope mode;
      if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
      status = validate(count, m, components, q_lower, q_upper, barycentric_lower,
                        barycentric_upper);
    }
    if (status != PHX_MC_OK) return status;
    // Every intermediate is a fixed-size stack array: the region admits its
    // logical work and requests no arena payload (its managed bytes are only
    // the region's arena bookkeeping). Each chunk charges its whole admitted
    // work before evaluating any item.
    const uint64_t work = item_work(m, components);
    phx::mc::ParallelRequest request;
    request.parallelism = phx::mc::current_parallelism();
    request.lease_seconds = phx::mc::current_parallel_lease_seconds();
    request.per_item.work = work;
    const int64_t l_width = static_cast<int64_t>(m) * m;
    return phx::mc::parallel_for(static_cast<std::size_t>(count), request,
        [&](std::size_t begin, std::size_t end, phx::mc::ChunkContext&) -> int32_t {
          const phx::mc::GradualNearestScope mode;
          if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
          phx::mc::native_execution_charge(static_cast<int64_t>(work * (end - begin)));
          for (std::size_t item = begin; item < end; ++item) {
            const auto row = static_cast<int64_t>(item);
            enclose_item(m, components, q_lower + row * q_width, q_upper + row * q_width,
                         barycentric_lower + row * l_width, barycentric_upper + row * l_width,
                         lower + row * components, upper + row * components);
          }
          return PHX_MC_OK;
        });
  });
}

int32_t phx_mc_quadratic_tetrahedron_validity(
    int64_t count, const double* nodes, const int32_t* derivative_weights,
    double relative_floor, int64_t maximum_bernstein_nodes,
    int64_t maximum_subdivision_depth, int64_t maximum_work_units,
    int64_t maximum_scratch_bytes, int32_t* decisions, double* lower, double* upper,
    int64_t* evidence) {
  return quadratic_tetrahedron_validity(
      count, nodes, derivative_weights, relative_floor, maximum_bernstein_nodes,
      maximum_subdivision_depth, maximum_work_units, maximum_scratch_bytes, decisions, lower,
      upper, evidence);
}

}  // extern "C"
