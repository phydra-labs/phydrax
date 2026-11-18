//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include "check.hpp"
#include "phydrax_meshcore.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

#include <limits>

namespace {

// Independent scalar transcription of the NumPy operation order, without the
// native kernel's eligibility checks. Compare binary64 representations.
std::array<double, 2> reference(const double* c, const double* point) {
  double x = 1.0 / 3.0, y = 1.0 / 3.0;
  for (int iteration = 0; iteration < 8; ++iteration) {
    const double values[3][10] = {
        {1.0, x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y},
        {0.0, 1.0, 0.0, 2.0*x, y, 0.0, 3.0*x*x, 2.0*x*y, y*y, 0.0},
        {0.0, 0.0, 1.0, 0.0, x, 2.0*y, 0.0, x*x, 2.0*x*y, 3.0*y*y}};
    double eval[3][3];
    for (int kind = 0; kind < 3; ++kind) {
      for (int axis = 0; axis < 3; ++axis) {
        double sum = c[axis*10] * values[kind][0];
        for (int k = 1; k < 10; ++k) sum = sum + c[axis*10+k] * values[kind][k];
        eval[kind][axis] = sum;
      }
    }
    double a = 0.0, b = 0.0, d = 0.0, r = 0.0, s = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double residual = eval[0][axis] - point[axis];
      a = a + eval[1][axis] * eval[1][axis];
      b = b + eval[1][axis] * eval[2][axis];
      d = d + eval[2][axis] * eval[2][axis];
      r = r + eval[1][axis] * residual;
      s = s + eval[2][axis] * residual;
    }
    r = -r;
    s = -s;
    const double determinant = a*d - b*b;
    const bool valid = std::isfinite(determinant) && determinant > 0.0;
    const double safe = valid ? determinant : 1.0;
    const double dx = (r*d - s*b) / safe;
    const double dy = (a*s - b*r) / safe;
    x = std::clamp(x + (valid ? dx : 0.0), 0.0, 1.0);
    y = std::clamp(y + (valid ? dy : 0.0), 0.0, 1.0);
    const double total = x + y;
    const bool over = total > 1.0;
    const double divisor = over ? total : 1.0;
    const double divided_x = x / divisor, divided_y = y / divisor;
    x = std::clamp(over ? divided_x : x, 0.0, 1.0);
    y = over ? divided_y : y;
    const double cap = std::nextafter(std::nextafter(1.0 - x, 0.0), 0.0);
    y = std::clamp(std::min(y, cap), 0.0, 1.0);
  }
  return {x, y};
}

bool same(double a, double b) {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

void comparison(const std::array<double, 30>& coefficients, const std::array<double, 3>& point) {
  const int64_t face = 0;
  double x = -11.0, y = -12.0;
  int32_t handled = -1;
  const auto expected = reference(coefficients.data(), point.data());
  PHX_CHECK(phx_mc_implicit_face_proposals(coefficients.data(), 1, &face, point.data(), 1,
                                           &x, &y, &handled) == PHX_MC_OK);
  PHX_CHECK(handled == 1);
  PHX_CHECK(same(x, expected[0]));
  PHX_CHECK(same(y, expected[1]));
}

void ordinary_and_zero_coefficients() {
  std::array<double, 30> linear{};
  linear[1] = 1.0;
  linear[12] = 1.0;
  const std::array<double, 3> point{0.25, 0.5, 0.75};
  const int64_t face = 0;
  double x = -1.0, y = -1.0;
  int32_t handled = -1;
  PHX_CHECK(phx_mc_implicit_face_proposals(linear.data(), 1, &face, point.data(), 1,
                                           &x, &y, &handled) == PHX_MC_OK);
  PHX_CHECK(handled == 1 && same(x, 0.25) && same(y, 0.5));
  comparison(linear, point);
  comparison({}, point);
  std::array<double, 30> curved{};
  for (int k = 0; k < 30; ++k) curved[k] = (k % 7 - 3) * 0.03125;
  curved[1] = 1.0;
  curved[12] = 1.0;
  for (int k = 0; k < 30; ++k) {
    auto one_zero = curved;
    one_zero[k] = 0.0;
    comparison(one_zero, point);
  }
  // Dependent derivatives exercise the safe determinant and both numerator paths.
  std::array<double, 30> rank_deficient{};
  rank_deficient[1] = rank_deficient[2] = 1.0;
  comparison(rank_deficient, point);
  comparison(linear, {1.0, 1.0, 0.0});
}

void refusal() {
  std::array<double, 30> coefficients{};
  coefficients[1] = coefficients[12] = 0x1p-400;
  std::array<double, 3> point{0.0, 0.0, 0.0};
  const int64_t face = 0;
  double x = 123.0, y = 456.0;
  int32_t handled = 9;
  PHX_CHECK(phx_mc_implicit_face_proposals(coefficients.data(), 1, &face, point.data(), 1,
                                           &x, &y, &handled) == PHX_MC_OK);
  PHX_CHECK(handled == 0 && x == 123.0 && y == 456.0);
  coefficients[1] = std::numeric_limits<double>::infinity();
  handled = 9;
  PHX_CHECK(phx_mc_implicit_face_proposals(coefficients.data(), 1, &face, point.data(), 1,
                                           &x, &y, &handled) == PHX_MC_OK);
  PHX_CHECK(handled == 0 && x == 123.0 && y == 456.0);

  std::array<double, 60> faces{};
  faces[1] = faces[12] = 1.0;
  faces[31] = faces[42] = 0x1p-400;
  const int64_t indices[2] = {0, 1};
  const double points[6] = {0.25, 0.5, 0.0, 0.0, 0.0, 0.0};
  double first[2] = {123.0, 123.0}, second[2] = {456.0, 456.0};
  handled = 9;
  PHX_CHECK(phx_mc_implicit_face_proposals(faces.data(), 2, indices, points, 2,
                                           first, second, &handled) == PHX_MC_OK);
  PHX_CHECK(handled == 0 && first[0] == 0.25 && second[0] == 0.5);
  PHX_CHECK(first[1] == 123.0 && second[1] == 456.0);
}

void validation() {
  std::array<double, 30> coefficients{};
  std::array<double, 3> point{};
  int64_t index = 0;
  double first = 123.0, second = 456.0;
  int32_t handled = 9;
  auto call = [&](const double* c, int64_t n, const int64_t* rows, const double* p,
                  int64_t count, double* a, double* b, int32_t* h) {
    return phx_mc_implicit_face_proposals(c, n, rows, p, count, a, b, h);
  };
  PHX_CHECK(call(nullptr, 1, &index, point.data(), 1, &first, &second, &handled) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(coefficients.data(), -1, &index, point.data(), 1, &first, &second, &handled) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(coefficients.data(), 1, &index, point.data(), -1, &first, &second, &handled) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(coefficients.data(), std::numeric_limits<int64_t>::max(), &index, point.data(), 1, &first, &second, &handled) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(coefficients.data(), 1, &index, point.data(), std::numeric_limits<int64_t>::max(), &first, &second, &handled) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(coefficients.data(), 1, nullptr, point.data(), 1, &first, &second, &handled) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(coefficients.data(), 1, &index, nullptr, 1, &first, &second, &handled) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(coefficients.data(), 1, &index, point.data(), 1, nullptr, &second, &handled) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(coefficients.data(), 1, &index, point.data(), 1, &first, nullptr, &handled) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(coefficients.data(), 1, &index, point.data(), 1, &first, &second, nullptr) == PHX_MC_INVALID_ARGUMENT);
  index = -1;
  PHX_CHECK(call(coefficients.data(), 1, &index, point.data(), 1, &first, &second, &handled) == PHX_MC_INVALID_INPUT);
  index = 1;
  PHX_CHECK(call(coefficients.data(), 1, &index, point.data(), 1, &first, &second, &handled) == PHX_MC_INVALID_INPUT);
  index = 0;
  // Every output is checked against every input, including partial overlaps.
  for (double* output : {coefficients.data() + 1, point.data() + 1,
                         reinterpret_cast<double*>(&index)}) {
    PHX_CHECK(call(coefficients.data(), 1, &index, point.data(), 1, output, &second, &handled) == PHX_MC_INVALID_ARGUMENT);
    PHX_CHECK(call(coefficients.data(), 1, &index, point.data(), 1, &first, output, &handled) == PHX_MC_INVALID_ARGUMENT);
  }
  for (int32_t* output : {reinterpret_cast<int32_t*>(coefficients.data() + 1),
                          reinterpret_cast<int32_t*>(point.data() + 1),
                          reinterpret_cast<int32_t*>(&index),
                          reinterpret_cast<int32_t*>(&first),
                          reinterpret_cast<int32_t*>(&second)}) {
    PHX_CHECK(call(coefficients.data(), 1, &index, point.data(), 1, &first, &second, output) == PHX_MC_INVALID_ARGUMENT);
  }
  const int64_t rows[2] = {0, 1};
  const double points[6] = {};
  double first_rows[2] = {123.0, 123.0}, second_rows[2] = {456.0, 456.0};
  PHX_CHECK(call(coefficients.data(), 1, rows, points, 2, first_rows, second_rows, &handled) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(first_rows[0] == 123.0 && first_rows[1] == 123.0 &&
            second_rows[0] == 456.0 && second_rows[1] == 456.0);
  PHX_CHECK(call(coefficients.data(), 1, &index, point.data(), 1, &first, &first, &handled) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(coefficients.data(), 1, &index, point.data(), 1, coefficients.data(), &second, &handled) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(coefficients.data(), 1, &index, point.data(), 1, &first, point.data(), &handled) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(coefficients.data(), 1, &index, point.data(), 1, &first, &second, reinterpret_cast<int32_t*>(&index)) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(coefficients.data(), 1, &index, point.data(), 1, &first, &second, reinterpret_cast<int32_t*>(&first)) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(first == 123.0 && second == 456.0 && handled == 9);
  PHX_CHECK(call(nullptr, 0, nullptr, nullptr, 0, nullptr, nullptr, &handled) == PHX_MC_OK);
  PHX_CHECK(handled == 1);
}

void rounding_modes() {
  const int original = std::fegetround();
  std::array<double, 30> c{};
  const double point[3] = {};
  const int64_t index = 0;
  for (const int mode : {FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    if (std::fesetround(mode) != 0) continue;
    double first = 123.0, second = 456.0;
    int32_t handled = 9;
    PHX_CHECK(phx_mc_implicit_face_proposals(c.data(), 1, &index, point, 1,
                                             &first, &second, &handled) == PHX_MC_OK);
    PHX_CHECK(handled == 0 && first == 123.0 && second == 456.0);
    PHX_CHECK(std::fegetround() == mode);
  }
  PHX_CHECK(std::fesetround(original) == 0);
}

void gradual_underflow_mode() {
  std::array<double, 30> c{};
  const double point[3] = {};
  const int64_t index = 0;
  double first = 123.0, second = 456.0;
  int32_t handled = 9;
#if defined(__SSE__)
  const unsigned int original = _mm_getcsr();
  _mm_setcsr(original | (1u << 15));
  PHX_CHECK(phx_mc_implicit_face_proposals(c.data(), 1, &index, point, 1,
                                           &first, &second, &handled) == PHX_MC_OK);
  PHX_CHECK(handled == 0 && first == 123.0 && second == 456.0);
  _mm_setcsr(original | (1u << 6));
  handled = 9;
  PHX_CHECK(phx_mc_implicit_face_proposals(c.data(), 1, &index, point, 1,
                                           &first, &second, &handled) == PHX_MC_OK);
  PHX_CHECK(handled == 0 && first == 123.0 && second == 456.0);
  _mm_setcsr(original);
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
  std::uint64_t original;
  __asm__ volatile("mrs %0, fpcr" : "=r"(original));
  const std::uint64_t flush = original | (std::uint64_t{1} << 24);
  __asm__ volatile("msr fpcr, %0" : : "r"(flush));
  PHX_CHECK(phx_mc_implicit_face_proposals(c.data(), 1, &index, point, 1,
                                           &first, &second, &handled) == PHX_MC_OK);
  PHX_CHECK(handled == 0 && first == 123.0 && second == 456.0);
  __asm__ volatile("msr fpcr, %0" : : "r"(original));
#else
  (void)c;
  (void)point;
  (void)index;
  (void)first;
  (void)second;
  (void)handled;
#endif
}

}  // namespace

int main() {
  ordinary_and_zero_coefficients();
  refusal();
  validation();
  rounding_modes();
  gradual_underflow_mode();
  return phx::mc::test::finish("test_implicit_face_proposals");
}
