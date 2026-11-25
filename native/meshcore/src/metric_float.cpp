//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Riemannian metric measurements on the deterministic binary64 kernels; see
// metric_float.hpp. Every sum below is written in the order it is evaluated.
#include "metric_float.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "deterministic_float.hpp"
#include "geometry3.hpp"

namespace phx::mc::metric {
namespace {

constexpr std::uint64_t kNaNBits = 0x7ff8000000000000ULL;
// Lengths at the two ends of an edge count as equal within this ratio.
constexpr double kNearlyEqual = 0x1.0c6f7a0b5ed8dp-20;
// Symmetry defect and smallest eigenvalue of a field tensor are judged
// against this multiple (64 * 2^-52) of its largest eigenvalue magnitude.
constexpr double kPropertyTolerance = 0x1p-46;

double nan() noexcept { return std::bit_cast<double>(kNaNBits); }
double canonical(double value) noexcept { return value == value ? value : nan(); }
double magnitude(double value) noexcept { return value < 0.0 ? -value : value; }
bool finite(double value) noexcept {
  return magnitude(value) <= std::numeric_limits<double>::max();
}

// sqrt of a quadratic form; a negative or NaN form has no length.
double root(double form) noexcept { return form >= 0.0 ? std::sqrt(form) : nan(); }

// The smallest integer not below `value`, for |value| < 2^52; larger
// magnitudes are integers already and are returned with infinities unchanged.
double ceiling(double value) noexcept {
  if (!(magnitude(value) < 0x1p+52)) return value;
  const double truncated = static_cast<double>(static_cast<std::int64_t>(value));
  return truncated < value ? truncated + 1.0 : truncated;
}

double distance(const double* p, const double* q) noexcept {
  const double x = p[0] - q[0];
  const double y = p[1] - q[1];
  const double z = p[2] - q[2];
  return std::sqrt((x * x + y * y) + z * z);
}

}  // namespace

// det(p1 - p0, p2 - p0, p3 - p0) by cofactors of the first row.
double corner_determinant(const double* a, const double* b, const double* c,
                         const double* d) noexcept {
  const double* corners[3] = {b, c, d};
  double rows[9];
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      rows[3 * row + axis] = corners[row][axis] - a[axis];
    }
  }
  return deterministic::determinant3(rows);
}

bool logarithm(const double* metric, double* logarithm) noexcept {
  deterministic::SymmetricEigen3 eigen;
  if (!deterministic::symmetric_eigen3(metric, eigen)) return false;
  std::array<double, 3> mapped{};
  for (std::size_t index = 0; index < 3; ++index) {
    if (!(eigen.values[index] > 0.0)) return false;
    mapped[index] = deterministic::log(eigen.values[index]);
  }
  const std::array<double, 9> result = deterministic::symmetric_function3(eigen, mapped);
  for (std::size_t index = 0; index < 9; ++index) logarithm[index] = result[index];
  return true;
}

bool exponential(const double* logarithm, double* metric) noexcept {
  deterministic::SymmetricEigen3 eigen;
  if (!deterministic::symmetric_eigen3(logarithm, eigen)) return false;
  std::array<double, 3> mapped{};
  for (std::size_t index = 0; index < 3; ++index) {
    mapped[index] = deterministic::exp(eigen.values[index]);
  }
  const std::array<double, 9> result = deterministic::symmetric_function3(eigen, mapped);
  for (const double value : result) {
    if (!finite(value)) return false;
  }
  for (std::size_t index = 0; index < 9; ++index) metric[index] = result[index];
  return true;
}

bool interpolate(std::int64_t count, const double* weights, const double* metrics,
                 const double* logarithms, double* metric) noexcept {
  bool constant = true;
  for (std::int64_t k = 1; k < count && constant; ++k) {
    for (std::size_t index = 0; index < 9; ++index) {
      if (metrics[9 * k + static_cast<std::int64_t>(index)] != metrics[index]) {
        constant = false;
        break;
      }
    }
  }
  if (constant) {
    // A constant field is a fixed point of the mean: its tensor is returned
    // unchanged, once the kernel finds it a metric as it would for the mean.
    double unused[9];
    if (!logarithm(metrics, unused)) return false;
    for (std::size_t index = 0; index < 9; ++index) metric[index] = metrics[index];
    return true;
  }
  // sum_k w_k log M_k, the support in order; the lower triangle mirrors the upper.
  double mean[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (std::int64_t k = 0; k < count; ++k) {
    double evaluated[9];
    const double* term = nullptr;
    if (logarithms != nullptr) {
      term = logarithms + 9 * k;
    } else {
      if (!logarithm(metrics + 9 * k, evaluated)) return false;
      term = evaluated;
    }
    for (std::size_t row = 0; row < 3; ++row) {
      for (std::size_t column = row; column < 3; ++column) {
        const double product = weights[k] * term[3 * row + column];
        mean[3 * row + column] = k == 0 ? product : mean[3 * row + column] + product;
      }
    }
  }
  return exponential(mean, metric);
}

double length(const double* p, const double* q, const double* metric_p,
              const double* metric_q) noexcept {
  const double delta[3] = {q[0] - p[0], q[1] - p[1], q[2] - p[2]};
  const double first = root(deterministic::quadratic_form3(delta, metric_p));
  const double second = root(deterministic::quadratic_form3(delta, metric_q));
  const double ratio = second / (first > 0.0 ? first : 1.0);
  if (magnitude(ratio - 1.0) <= kNearlyEqual) return canonical(0.5 * (first + second));
  return canonical((first - second) / deterministic::log(1.0 / ratio));
}

void band_quanta(double length, double& above, double& below) noexcept {
  // A length that is not a number was not measured: it is neither inside the
  // band nor a violation of it.
  if (length != length) {
    above = nan();
    below = nan();
    return;
  }
  above = 0.0;
  below = 0.0;
  if (length > kUpperLength) {
    const double quanta =
        ceiling(deterministic::log(length / kUpperLength) * kViolationQuanta);
    above = quanta > 1.0 ? quanta : 1.0;
  } else if (length < kLowerLength) {
    const double quanta =
        ceiling(deterministic::log(kLowerLength / length) * kViolationQuanta);
    below = quanta > 1.0 ? quanta : 1.0;
  }
}

// Error of the evaluated determinant, to first order in u = 2^-53. Let S be
// the sum of the six absolute triple products of the expansion and R the
// product of the three row lengths; every triple product is at most R, so
// S <= 6 R.
//   rows:    each entry p_k - p_0 is rounded once, a relative error u per
//            factor: 3 u S.
//   minors:  two rounded products and a rounded difference: 2 u times the
//            sum of the two absolute products.
//   terms:   the rounded product with the first-row entry adds u more:
//            3 u S over the three terms.
//   sums:    two rounded additions of the three terms: 2 u S.
// Together 8 u S <= 48 u R. The rows are the edges at corner 0, so R is the
// product of the edge lengths there, at most the largest corner product; the
// stated 50 u covers the rounding of that product and the higher orders.
void relative_determinant(const double* corners, double& determinant,
                          double& relative) noexcept {
  determinant = canonical(corner_determinant(corners, corners + 3, corners + 6, corners + 9));
  double largest = 0.0;
  bool products_finite = true;
  for (std::size_t corner = 0; corner < 4; ++corner) {
    double product = 1.0;
    for (std::size_t other = 0; other < 4; ++other) {
      if (other != corner) {
        product = product * distance(corners + 3 * corner, corners + 3 * other);
      }
    }
    if (!finite(product)) products_finite = false;
    if (product > largest) largest = product;
  }
  const double size = magnitude(determinant);
  relative = finite(size) && products_finite && largest > 0.0 ? size / largest : 0.0;
}

double geometric_shape(const double* corners, double determinant) noexcept {
  double squares = 0.0;
  for (std::size_t a = 0; a < 4; ++a) {
    for (std::size_t b = a + 1; b < 4; ++b) {
      squares += geometry::squared_distance(corners + 3 * a, corners + 3 * b);
    }
  }
  const double volume_scale = deterministic::cube_root(0.5 * determinant);
  return 12.0 * volume_scale * volume_scale / squares;
}

double log_determinant(const double* metric) noexcept {
  return deterministic::log(deterministic::determinant3(metric));
}

double tetra_quality(const double* corners, const double* metrics) noexcept {
  double logs[4];
  for (std::size_t corner = 0; corner < 4; ++corner) {
    logs[corner] = log_determinant(metrics + 9 * corner);
  }
  return tetra_quality(corners, metrics, logs);
}

namespace {

double mean_log_determinant(const double* log_determinants) noexcept {
  double value = log_determinants[0];
  for (std::size_t corner = 1; corner < 4; ++corner) {
    value = value + log_determinants[corner];
  }
  return value * 0.25;
}

double finish_tetra_quality(double determinant, double squares, double mean_log) noexcept {
  const double volume = magnitude(determinant) / 6.0;
  const double metric_volume = volume * deterministic::exp(0.5 * mean_log);
  if (!(metric_volume > 0.0 && squares > 0.0)) return 0.0;
  return canonical(12.0 * deterministic::pow_two_thirds(3.0 * metric_volume) / squares);
}

}  // namespace

double tetra_quality_from_measurements(double determinant, double squares,
                                       const double* log_determinants) noexcept {
  return finish_tetra_quality(determinant, squares, mean_log_determinant(log_determinants));
}

double tetra_quality(const double* corners, const double* metrics,
                     const double* log_determinants) noexcept {
  const double average_log = mean_log_determinant(log_determinants);
  double squares = 0.0;
  bool first_edge = true;
  for (std::size_t first = 0; first < 4; ++first) {
    for (std::size_t second = first + 1; second < 4; ++second) {
      const double edge = length(corners + 3 * first, corners + 3 * second,
                                 metrics + 9 * first, metrics + 9 * second);
      const double square = edge * edge;
      squares = first_edge ? square : squares + square;
      first_edge = false;
    }
  }
  return finish_tetra_quality(
      corner_determinant(corners, corners + 3, corners + 6, corners + 9), squares, average_log);
}

void tensor_properties(const double* tensor, bool& symmetric,
                       bool& positive_definite) noexcept {
  symmetric = false;
  positive_definite = false;
  double defect = 0.0;
  double part[9];
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      const double entry = tensor[3 * row + column];
      const double mirror = tensor[3 * column + row];
      if (!finite(entry)) return;
      const double difference = magnitude(entry - mirror);
      if (difference > defect) defect = difference;
      part[3 * row + column] = 0.5 * (entry + mirror);
    }
  }
  deterministic::SymmetricEigen3 eigen;
  if (!deterministic::symmetric_eigen3(part, eigen)) return;
  const double largest = magnitude(eigen.values[0]) > magnitude(eigen.values[2])
                             ? magnitude(eigen.values[0])
                             : magnitude(eigen.values[2]);
  const double tolerance = kPropertyTolerance * largest;
  symmetric = defect <= tolerance;
  positive_definite = symmetric && eigen.values[0] > tolerance;
}

void congruence_defect(const double* linear, const double* metric, const double* reference,
                       double& error, double& scale) noexcept {
  // B = M A by rows, then C = A^T B by rows: every entry is a fixed-order sum
  // of three products.
  double product[9];
  double image[9];
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      product[3 * row + column] =
          (metric[3 * row] * linear[column] + metric[3 * row + 1] * linear[3 + column]) +
          metric[3 * row + 2] * linear[6 + column];
    }
  }
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      image[3 * row + column] =
          (linear[row] * product[column] + linear[3 + row] * product[3 + column]) +
          linear[6 + row] * product[6 + column];
    }
  }
  double largest_error = 0.0;
  double largest_image = 0.0;
  double largest_reference = 0.0;
  bool defined = true;
  for (std::size_t row = 0; row < 3; ++row) {
    double error_sum = 0.0;
    double image_sum = 0.0;
    double reference_sum = 0.0;
    for (std::size_t column = 0; column < 3; ++column) {
      const double difference = magnitude(image[3 * row + column] - reference[3 * row + column]);
      const double image_size = magnitude(image[3 * row + column]);
      const double reference_size = magnitude(reference[3 * row + column]);
      error_sum = column == 0 ? difference : error_sum + difference;
      image_sum = column == 0 ? image_size : image_sum + image_size;
      reference_sum = column == 0 ? reference_size : reference_sum + reference_size;
    }
    if (!(error_sum == error_sum && image_sum == image_sum && reference_sum == reference_sum)) {
      defined = false;
    }
    if (error_sum > largest_error) largest_error = error_sum;
    if (image_sum > largest_image) largest_image = image_sum;
    if (reference_sum > largest_reference) largest_reference = reference_sum;
  }
  error = defined ? largest_error : nan();
  scale = defined ? (largest_image > largest_reference ? largest_image : largest_reference)
                  : nan();
}

}  // namespace phx::mc::metric
