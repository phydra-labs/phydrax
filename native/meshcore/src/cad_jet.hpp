// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Truncated Taylor polynomials in one or two variables: the derivative carrier
// of the CAD geometry nodes. A node evaluates its own local Taylor expansion at
// a parameter point; composition with its input expansions (chain rule of any
// order) is a polynomial substitution truncated at the requested order.
#pragma once

#include <cmath>

namespace phx::mc::cad {

// Highest total derivative order carried. Output jets are order 2; a generic
// offset asks its base for one order more, so two nested offsets fit.
constexpr int kMaxOrder = 4;
constexpr int kTerms = (kMaxOrder + 1) * (kMaxOrder + 2) / 2;

// Normalized coefficient (i, j) = d^(i+j) f / (du^i dv^j) / (i! j!), stored in
// graded order: total degree n occupies [n (n + 1) / 2, (n + 1) (n + 2) / 2).
constexpr int term(int i, int j) noexcept { return (i + j) * (i + j + 1) / 2 + j; }

// Variables (1 or 2) and total order (0 to kMaxOrder) of one evaluation.
struct Truncation {
  int variables;
  int order;
  // Highest power of v kept at total degree n.
  int v_limit(int n) const noexcept { return variables == 2 ? n : 0; }
};

struct Taylor {
  double c[kTerms];
};

inline void clear(Taylor& a, const Truncation& t) noexcept {
  for (int n = 0; n <= t.order; ++n)
    for (int j = 0; j <= t.v_limit(n); ++j) a.c[term(n - j, j)] = 0.0;
}

inline Taylor constant(double value, const Truncation& t) noexcept {
  Taylor a;
  clear(a, t);
  a.c[0] = value;
  return a;
}

// The coordinate function of variable `axis` (0 = u, 1 = v) at `value`.
inline Taylor variable(double value, int axis, const Truncation& t) noexcept {
  Taylor a = constant(value, t);
  if (t.order >= 1) a.c[axis == 0 ? term(1, 0) : term(0, 1)] = 1.0;
  return a;
}

// out = x + alpha * y (out may alias x or y).
inline void axpy(const Taylor& x, double alpha, const Taylor& y, Taylor& out,
                 const Truncation& t) noexcept {
  for (int n = 0; n <= t.order; ++n)
    for (int j = 0; j <= t.v_limit(n); ++j) {
      const int k = term(n - j, j);
      out.c[k] = x.c[k] + alpha * y.c[k];
    }
}

inline void scale(const Taylor& x, double alpha, Taylor& out, const Truncation& t) noexcept {
  for (int n = 0; n <= t.order; ++n)
    for (int j = 0; j <= t.v_limit(n); ++j) out.c[term(n - j, j)] = alpha * x.c[term(n - j, j)];
}

// Truncated Cauchy product; out must not alias x or y.
inline void multiply(const Taylor& x, const Taylor& y, Taylor& out, const Truncation& t) noexcept {
  for (int n = 0; n <= t.order; ++n)
    for (int j = 0; j <= t.v_limit(n); ++j) {
      const int i = n - j;
      double sum = 0.0;
      for (int a = 0; a <= i; ++a)
        for (int b = 0; b <= j; ++b) sum += x.c[term(a, b)] * y.c[term(i - a, j - b)];
      out.c[term(i, j)] = sum;
    }
}

// sum_n f[n] (x - x0)^n by Horner: the normalized derivatives f[n] = f^(n)(x0) / n!
// of a scalar function at the constant term x0 of x. out must not alias x.
inline void compose_unary(const double* f, const Taylor& x, Taylor& out,
                          const Truncation& t) noexcept {
  Taylor delta = x;
  delta.c[0] = 0.0;
  out = constant(f[t.order], t);
  Taylor product;
  for (int n = t.order - 1; n >= 0; --n) {
    multiply(out, delta, product, t);
    out = product;
    out.c[0] += f[n];
  }
}

inline void sine_cosine(const Taylor& x, Taylor& sine, Taylor& cosine,
                        const Truncation& t) noexcept {
  const double s = std::sin(x.c[0]);
  const double c = std::cos(x.c[0]);
  // d^n sin = sin(x + n pi / 2): the cycle s, c, -s, -c.
  const double sine_cycle[4] = {s, c, -s, -c};
  const double cosine_cycle[4] = {c, -s, -c, s};
  double fs[kMaxOrder + 1], fc[kMaxOrder + 1];
  double factorial = 1.0;
  for (int n = 0; n <= t.order; ++n) {
    if (n > 0) factorial *= n;
    fs[n] = sine_cycle[n % 4] / factorial;
    fc[n] = cosine_cycle[n % 4] / factorial;
  }
  compose_unary(fs, x, sine, t);
  compose_unary(fc, x, cosine, t);
}

inline void square_root(const Taylor& x, Taylor& out, const Truncation& t) noexcept {
  // binom(1/2, n) x0^(1/2 - n).
  const double x0 = x.c[0];
  double f[kMaxOrder + 1];
  f[0] = std::sqrt(x0);
  double binomial = 1.0;
  for (int n = 1; n <= t.order; ++n) {
    binomial *= (0.5 - (n - 1)) / n;
    f[n] = binomial * f[0] / std::pow(x0, n);
  }
  compose_unary(f, x, out, t);
}

inline void reciprocal(const Taylor& x, Taylor& out, const Truncation& t) noexcept {
  // (-1)^n x0^(-n-1).
  const double inverse = 1.0 / x.c[0];
  double f[kMaxOrder + 1];
  f[0] = inverse;
  for (int n = 1; n <= t.order; ++n) f[n] = -f[n - 1] * inverse;
  compose_unary(f, x, out, t);
}

// Partial derivative along `axis`: one order lower (callers truncate at order - 1).
inline void differentiate(const Taylor& x, int axis, Taylor& out, const Truncation& t) noexcept {
  for (int n = 0; n < t.order; ++n)
    for (int j = 0; j <= t.v_limit(n); ++j) {
      const int i = n - j;
      out.c[term(i, j)] =
          axis == 0 ? (i + 1) * x.c[term(i + 1, j)] : (j + 1) * x.c[term(i, j + 1)];
    }
}

// Substitutes the local expansion F (in `local` variables, components
// `count`) at inputs whose constant terms are the expansion point:
// out[k] = sum_(a,b) F[k](a,b) (x0 - x0(0))^a (x1 - x1(0))^b.
inline void compose(const Taylor* local, int count, int local_variables, const Taylor* inputs,
                    Taylor* out, const Truncation& t) noexcept {
  Taylor powers[2][kMaxOrder + 1];
  for (int axis = 0; axis < local_variables; ++axis) {
    Taylor delta = inputs[axis];
    delta.c[0] = 0.0;
    powers[axis][0] = constant(1.0, t);
    for (int a = 1; a <= t.order; ++a) multiply(powers[axis][a - 1], delta, powers[axis][a], t);
  }
  for (int k = 0; k < count; ++k) clear(out[k], t);
  for (int n = 0; n <= t.order; ++n)
    for (int b = 0; b <= (local_variables == 2 ? n : 0); ++b) {
      const int a = n - b;
      Taylor monomial;
      if (local_variables == 2)
        multiply(powers[0][a], powers[1][b], monomial, t);
      else
        monomial = powers[0][a];
      for (int k = 0; k < count; ++k) axpy(out[k], local[k].c[term(a, b)], monomial, out[k], t);
    }
}

}  // namespace phx::mc::cad
