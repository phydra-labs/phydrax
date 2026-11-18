//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include "capi_guard.hpp"
#include "phydrax_meshcore.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if !defined(PHX_MC_FP_CONTRACT_OFF)
#error "Implicit face proposals require separately rounded operations"
#endif

namespace {

struct Span {
  std::uintptr_t begin;
  std::uintptr_t end;
};

bool span(const void* pointer, std::size_t size, Span& result) noexcept {
  const auto start = reinterpret_cast<std::uintptr_t>(pointer);
  if (size != 0 && (pointer == nullptr || size > UINTPTR_MAX - start)) return false;
  result = {start, start + size};
  return true;
}

bool overlaps(Span a, Span b) noexcept {
  return a.begin < a.end && b.begin < b.end && a.begin < b.end && b.begin < a.end;
}

// Volatile inputs prevent compile-time evaluation and expose the active mode,
// including flush-to-zero and denormals-are-zero, without modifying it.
bool supported_mode() noexcept {
  volatile double normal = std::numeric_limits<double>::min();
  volatile double half = normal * 0.5;
  volatile double restored = half * 2.0;
  volatile double one = 1.0;
  volatile double half_ulp = 0x1p-53;
  volatile double three_half_ulp = 0x1.8p-52;
  volatile double tie_even = one + half_ulp;
  volatile double tie_odd = one + three_half_ulp;
  return restored == normal && tie_even == 1.0 && tie_odd == 0x1.0000000000002p+0;
}

struct Arithmetic {
  bool eligible = true;

  double checked(double x) noexcept {
    if (!std::isfinite(x)) eligible = false;
    return x;
  }
  double add(double a, double b) noexcept { return checked(a + b); }
  double sub(double a, double b) noexcept { return checked(a - b); }
  double mul(double a, double b) noexcept {
    const double result = checked(a * b);
    if (a != 0.0 && b != 0.0 && std::fabs(result) < std::numeric_limits<double>::min())
      eligible = false;
    return result;
  }
  double div(double a, double b) noexcept {
    const double result = checked(a / b);
    if (b == 0.0 || (a != 0.0 && std::fabs(result) < std::numeric_limits<double>::min()))
      eligible = false;
    return result;
  }
};

double dot3(Arithmetic& op, const double* a, const double* b) noexcept {
  double result = 0.0;
  for (int k = 0; k < 3; ++k) result = op.add(result, op.mul(a[k], b[k]));
  return result;
}

void evaluate(Arithmetic& op, const double* coefficients, const double* values,
              double* result) noexcept {
  for (int axis = 0; axis < 3; ++axis) {
    const double* c = coefficients + 10 * axis;
    double total = op.mul(c[0], values[0]);
    for (int k = 1; k < 10; ++k) total = op.add(total, op.mul(c[k], values[k]));
    result[axis] = total;
  }
}

double clip(double value) noexcept {
  if (value < 0.0) return 0.0;
  if (value > 1.0) return 1.0;
  return value;
}

void proposal(Arithmetic& op, const double* c, const double* p,
              double& first, double& second) noexcept {
  for (int k = 0; k < 30; ++k) op.checked(c[k]);
  for (int k = 0; k < 3; ++k) op.checked(p[k]);
  first = 1.0 / 3.0;
  second = 1.0 / 3.0;
  for (int iter = 0; iter < 8; ++iter) {
    const double m[10] = {
        1.0, first, second, op.mul(first, first), op.mul(first, second),
        op.mul(second, second), op.mul(op.mul(first, first), first),
        op.mul(op.mul(first, first), second), op.mul(op.mul(first, second), second),
        op.mul(op.mul(second, second), second)};
    const double u[10] = {
        0.0, 1.0, 0.0, op.mul(2.0, first), second, 0.0,
        op.mul(op.mul(3.0, first), first), op.mul(op.mul(2.0, first), second),
        op.mul(second, second), 0.0};
    const double v[10] = {
        0.0, 0.0, 1.0, 0.0, first, op.mul(2.0, second), 0.0,
        op.mul(first, first), op.mul(op.mul(2.0, first), second),
        op.mul(op.mul(3.0, second), second)};
    double residual[3], du[3], dv[3];
    evaluate(op, c, m, residual);
    for (int k = 0; k < 3; ++k) residual[k] = op.sub(residual[k], p[k]);
    evaluate(op, c, u, du);
    evaluate(op, c, v, dv);
    const double a11 = dot3(op, du, du), a12 = dot3(op, du, dv), a22 = dot3(op, dv, dv);
    const double b1 = -dot3(op, du, residual), b2 = -dot3(op, dv, residual);
    const double determinant = op.sub(op.mul(a11, a22), op.mul(a12, a12));
    const bool valid = std::isfinite(determinant) && determinant > 0.0;
    const double safe = valid ? determinant : 1.0;
    const double step_first = op.div(op.sub(op.mul(b1, a22), op.mul(b2, a12)), safe);
    const double step_second = op.div(op.sub(op.mul(a11, b2), op.mul(a12, b1)), safe);
    first = clip(op.add(first, valid ? step_first : 0.0));
    second = clip(op.add(second, valid ? step_second : 0.0));
    const double total = op.add(first, second);
    const bool over = total > 1.0;
    const double divisor = over ? total : 1.0;
    const double divided_first = op.div(first, divisor);
    const double divided_second = op.div(second, divisor);
    first = clip(over ? divided_first : first);
    second = over ? divided_second : second;
    const double cap = std::nextafter(std::nextafter(op.sub(1.0, first), 0.0), 0.0);
    op.checked(cap);
    second = clip(second > cap ? cap : second);
  }
}

}  // namespace

extern "C" int32_t phx_mc_implicit_face_proposals(
    const double* coefficients, int64_t face_count, const int64_t* face_rows,
    const double* points, int64_t count, double* first, double* second, int32_t* handled) {
  using phx::mc::addressable;
  if (!addressable(face_count, 30, sizeof(double)) ||
      !addressable(count, 3, sizeof(double)) ||
      !addressable(count, 1, sizeof(int64_t))) return PHX_MC_INVALID_ARGUMENT;
  Span inputs[3], outputs[3];
  if (!span(coefficients, static_cast<std::size_t>(face_count) * 30 * sizeof(double), inputs[0]) ||
      !span(face_rows, static_cast<std::size_t>(count) * sizeof(int64_t), inputs[1]) ||
      !span(points, static_cast<std::size_t>(count) * 3 * sizeof(double), inputs[2]) ||
      !span(first, static_cast<std::size_t>(count) * sizeof(double), outputs[0]) ||
      !span(second, static_cast<std::size_t>(count) * sizeof(double), outputs[1]) ||
      !span(handled, sizeof(int32_t), outputs[2])) return PHX_MC_INVALID_ARGUMENT;
  for (int i = 0; i < 3; ++i) {
    for (const Span input : inputs)
      if (overlaps(outputs[i], input)) return PHX_MC_INVALID_ARGUMENT;
    for (int j = 0; j < i; ++j)
      if (overlaps(outputs[i], outputs[j])) return PHX_MC_INVALID_ARGUMENT;
  }
  for (int64_t row = 0; row < count; ++row)
    if (face_rows[row] < 0 || face_rows[row] >= face_count) return PHX_MC_INVALID_INPUT;
  *handled = 0;
  if (!supported_mode()) return PHX_MC_OK;
  Arithmetic op;
  for (int64_t row = 0; row < count; ++row) {
    double a, b;
    proposal(op, coefficients + 30 * face_rows[row], points + 3 * row, a, b);
    if (!op.eligible) return PHX_MC_OK;
    first[row] = a;
    second[row] = b;
  }
  *handled = 1;
  return PHX_MC_OK;
}
