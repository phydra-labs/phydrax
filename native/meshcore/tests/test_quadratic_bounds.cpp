// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Quadratic simplex bounds against independent exact dyadic oracles: the child
// Bernstein transform (L Q L^T)_kl and rational evaluations of
// f(L^T mu) = (L^T mu)^T Q (L^T mu) at dyadic child barycentric points mu,
// including callers left in flush-to-zero or directed-rounding modes.
#include <algorithm>
#include <array>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <vector>

#include "check.hpp"
#include "dyadic.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"

namespace {

using phx::mc::Dyadic;

constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kLargest = std::numeric_limits<double>::max();
constexpr double kSmallest = std::numeric_limits<double>::denorm_min();
constexpr double kSentinel = 12345.0;
// A deadline that a scope meets on entry and misses after the sleep.
constexpr double kDeadlineSeconds = 0.05;
constexpr auto kPastDeadline = std::chrono::milliseconds(80);

uint64_t item_work(int dimension, int components) {
  const auto m = static_cast<uint64_t>(dimension + 1);
  return static_cast<uint64_t>(components) * (m * m * m + m * m * (m + 1) / 2);
}

struct Batch {
  int dimension;
  int components;
  int64_t count;
  std::vector<double> q_lower;
  std::vector<double> q_upper;
  std::vector<double> l_lower;
  std::vector<double> l_upper;

  Batch(int dimension_, int components_, int64_t count_)
      : dimension(dimension_),
        components(components_),
        count(count_),
        q_lower(static_cast<std::size_t>(count_ * components_ * (dimension_ + 1) *
                                         (dimension_ + 1))),
        q_upper(q_lower.size()),
        l_lower(static_cast<std::size_t>(count_ * (dimension_ + 1) * (dimension_ + 1))),
        l_upper(l_lower.size()) {}

  int m() const { return dimension + 1; }
  std::size_t q_offset(int64_t item, int component) const {
    return static_cast<std::size_t>((item * components + component) * m() * m());
  }
  std::size_t l_offset(int64_t item) const { return static_cast<std::size_t>(item * m() * m()); }
  // Symmetric coefficient interval Q_ij = Q_ji.
  void coefficient(int64_t item, int component, int i, int j, double low, double high) {
    const std::size_t base = q_offset(item, component);
    q_lower[base + i * m() + j] = low;
    q_upper[base + i * m() + j] = high;
    q_lower[base + j * m() + i] = low;
    q_upper[base + j * m() + i] = high;
  }
  void vertex(int64_t item, int k, int i, double low, double high) {
    l_lower[l_offset(item) + k * m() + i] = low;
    l_upper[l_offset(item) + k * m() + i] = high;
  }
};

struct Bounds {
  std::vector<double> lower;
  std::vector<double> upper;
};

int32_t evaluate(const Batch& batch, Bounds& out) {
  const auto size = static_cast<std::size_t>(batch.count * batch.components);
  out.lower.assign(size, kSentinel);
  out.upper.assign(size, kSentinel);
  return phx_mc_simplex_quadratic_bounds(batch.count, batch.dimension, batch.components,
                                         batch.q_lower.data(), batch.q_upper.data(),
                                         batch.l_lower.data(), batch.l_upper.data(),
                                         out.lower.data(), out.upper.data());
}

bool untouched(const Bounds& out) {
  return std::all_of(out.lower.begin(), out.lower.end(), [](double x) { return x == kSentinel; }) &&
         std::all_of(out.upper.begin(), out.upper.end(), [](double x) { return x == kSentinel; });
}

bool identical(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

// ------------------------------------------------------------- exact oracle

int order(const Dyadic& a, const Dyadic& b) { return (a - b).sign(); }

// Exact (L Q L^T)_kr of one finite point choice (q, l), both (m, m).
Dyadic child_coefficient(int m, const double* q, const double* l, int k, int r) {
  Dyadic total;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      total = total + Dyadic(l[k * m + i]) * Dyadic(q[i * m + j]) * Dyadic(l[r * m + j]);
    }
  }
  return total;
}

// Exact f(L^T mu) for a dyadic child barycentric point mu.
Dyadic image_value(int m, const double* q, const double* l, const double* mu) {
  std::array<Dyadic, 4> lambda{};
  for (int j = 0; j < m; ++j) {
    for (int k = 0; k < m; ++k) lambda[j] = lambda[j] + Dyadic(mu[k]) * Dyadic(l[k * m + j]);
  }
  Dyadic total;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) total = total + lambda[i] * Dyadic(q[i * m + j]) * lambda[j];
  }
  return total;
}

// Every child point with coordinates in {0, 1/4, ..., 1} summing to one,
// including the child vertices.
std::vector<std::array<double, 4>> child_points(int m) {
  std::vector<std::array<double, 4>> points;
  std::array<int, 4> parts{};
  const auto visit = [&](auto&& self, int index, int remaining) -> void {
    if (index == m - 1) {
      parts[index] = remaining;
      std::array<double, 4> mu{};
      for (int i = 0; i < m; ++i) mu[i] = parts[i] / 4.0;
      points.push_back(mu);
      return;
    }
    for (int value = 0; value <= remaining; ++value) {
      parts[index] = value;
      self(self, index + 1, remaining - value);
    }
  };
  visit(visit, 0, 4);
  return points;
}

// lower <= value <= upper; an infinite endpoint is permitted only on its own
// side, and neither endpoint is NaN.
void check_contains(const Dyadic& value, double lower, double upper) {
  PHX_CHECK(!std::isnan(lower) && !std::isnan(upper));
  PHX_CHECK(lower != kInfinity && upper != -kInfinity);
  if (std::isfinite(lower)) PHX_CHECK(order(Dyadic(lower), value) <= 0);
  if (std::isfinite(upper)) PHX_CHECK(order(value, Dyadic(upper)) <= 0);
}

void check_choice(int m, const double* q, const double* l, double lower, double upper,
                  const std::vector<std::array<double, 4>>& points) {
  for (int k = 0; k < m; ++k) {
    for (int r = k; r < m; ++r) check_contains(child_coefficient(m, q, l, k, r), lower, upper);
  }
  for (const auto& mu : points) check_contains(image_value(m, q, l, mu.data()), lower, upper);
}

// Outward rounding loses at most a few units in the last place of the
// coefficient scale (plus gradual underflow) on point inputs.
double slack(int m, const double* q) {
  double scale = 0.0;
  for (int index = 0; index < m * m; ++index) scale = std::max(scale, std::fabs(q[index]));
  return std::ldexp(scale, -44) + 0x1p-1060;
}

void check_tight(int m, const double* q, const double* l, double lower, double upper) {
  Dyadic minimum = child_coefficient(m, q, l, 0, 0);
  Dyadic maximum = minimum;
  for (int k = 0; k < m; ++k) {
    for (int r = k; r < m; ++r) {
      const Dyadic value = child_coefficient(m, q, l, k, r);
      if (order(value, minimum) < 0) minimum = value;
      if (order(value, maximum) > 0) maximum = value;
    }
  }
  PHX_CHECK(std::isfinite(lower) && std::isfinite(upper));
  const Dyadic tolerance(slack(m, q));
  PHX_CHECK(order(minimum - Dyadic(lower), tolerance) <= 0);
  PHX_CHECK(order(Dyadic(upper) - maximum, tolerance) <= 0);
}

// ------------------------------------------------------------ input builders

double dyadic_value(std::mt19937_64& random) {
  const auto numerator = static_cast<int>(random() % 2049) - 1024;
  return std::ldexp(static_cast<double>(numerator), -static_cast<int>(random() % 10));
}

// Exact dyadic barycentric row: m parts of 64 summing to 64.
void random_row(std::mt19937_64& random, int m, double* row) {
  std::array<int, 5> cuts{};
  cuts[static_cast<std::size_t>(m)] = 64;
  for (int i = 1; i < m; ++i) cuts[static_cast<std::size_t>(i)] = static_cast<int>(random() % 65);
  std::sort(cuts.begin() + 1, cuts.begin() + m);
  for (int i = 0; i < m; ++i) row[i] = (cuts[i + 1] - cuts[i]) / 64.0;
}

enum class Map { Identity, Corner, Midpoint, Random, CollapsedPoint, CollapsedEdge };
constexpr std::array<Map, 6> kMaps{Map::Identity, Map::Corner,         Map::Midpoint,
                                   Map::Random,   Map::CollapsedPoint, Map::CollapsedEdge};

// Point child map: row k holds the root barycentric coordinates of vertex k.
void fill_map(Batch& batch, int64_t item, Map map, std::mt19937_64& random) {
  const int m = batch.m();
  std::array<std::array<double, 4>, 4> rows{};
  switch (map) {
    case Map::Identity:
      for (int k = 0; k < m; ++k) rows[k][k] = 1.0;
      break;
    case Map::Corner:  // Regular-subdivision child at root vertex 0.
      for (int k = 0; k < m; ++k) {
        rows[k][0] += 0.5;
        rows[k][k] += 0.5;
      }
      break;
    case Map::Midpoint:  // Child spanned by cyclic edge midpoints.
      for (int k = 0; k < m; ++k) {
        rows[k][k] += 0.5;
        rows[k][(k + 1) % m] += 0.5;
      }
      break;
    case Map::Random:
      for (int k = 0; k < m; ++k) random_row(random, m, rows[k].data());
      break;
    case Map::CollapsedPoint:
      random_row(random, m, rows[0].data());
      for (int k = 1; k < m; ++k) rows[k] = rows[0];
      break;
    case Map::CollapsedEdge:
      random_row(random, m, rows[0].data());
      random_row(random, m, rows[1].data());
      for (int k = 2; k < m; ++k) rows[k] = rows[k % 2];
      break;
  }
  for (int k = 0; k < m; ++k) {
    for (int i = 0; i < m; ++i) batch.vertex(item, k, i, rows[k][i], rows[k][i]);
  }
}

// Point degree-two Bernstein matrix of a signed polynomial of total degree
// 0 (a (sum lambda)^2), 1 (sum c_i lambda_i) or 2.
void fill_polynomial(Batch& batch, int64_t item, int component, int degree,
                     std::mt19937_64& random) {
  const int m = batch.m();
  std::array<double, 4> linear{};
  const double constant = dyadic_value(random);
  for (int i = 0; i < m; ++i) linear[i] = dyadic_value(random);
  for (int i = 0; i < m; ++i) {
    for (int j = i; j < m; ++j) {
      const double value = degree == 0   ? constant
                           : degree == 1 ? (linear[i] + linear[j]) / 2.0
                                         : dyadic_value(random);
      batch.coefficient(item, component, i, j, value, value);
    }
  }
}

void fill_point_batch(Batch& batch, std::mt19937_64& random) {
  for (int64_t item = 0; item < batch.count; ++item) {
    fill_map(batch, item, kMaps[static_cast<std::size_t>(item) % kMaps.size()], random);
    for (int component = 0; component < batch.components; ++component) {
      fill_polynomial(batch, item, component, static_cast<int>((item + component) % 3), random);
    }
  }
}

// Outward widening of a point batch: barycentric endpoints by 2^-20 within
// [0, 1] (rows then contain sum one), coefficients by a relative 2^-10.
void widen(Batch& batch) {
  for (std::size_t index = 0; index < batch.l_lower.size(); ++index) {
    batch.l_lower[index] = std::max(0.0, batch.l_lower[index] - 0x1p-20);
    batch.l_upper[index] = std::min(1.0, batch.l_upper[index] + 0x1p-20);
  }
  for (std::size_t index = 0; index < batch.q_lower.size(); ++index) {
    const double width = std::ldexp(std::fabs(batch.q_lower[index]), -10) + 0x1p-30;
    batch.q_lower[index] -= width;
    batch.q_upper[index] += width;
  }
}

// Endpoint selections of an interval item: pattern 0 all lower, 1 all upper,
// then mixed corners. Coefficient choices stay symmetric.
bool select(int pattern, int entry) {
  switch (pattern) {
    case 0:
      return false;
    case 1:
      return true;
    case 2:
      return entry % 2 == 1;
    case 3:
      return (entry / 2) % 2 == 1;
    default:
      return (entry * 7 + pattern) % 5 < 2;
  }
}

void check_interval_item(const Batch& batch, const Bounds& out, int64_t item,
                         const std::vector<std::array<double, 4>>& points) {
  const int m = batch.m();
  for (int component = 0; component < batch.components; ++component) {
    const auto slot = static_cast<std::size_t>(item * batch.components + component);
    for (int pattern = 0; pattern < 6; ++pattern) {
      std::array<double, 16> q{};
      std::array<double, 16> l{};
      const std::size_t q_base = batch.q_offset(item, component);
      const std::size_t l_base = batch.l_offset(item);
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
          const int pair = std::min(i, j) * m + std::max(i, j);
          q[i * m + j] = select(pattern, pair) ? batch.q_upper[q_base + i * m + j]
                                               : batch.q_lower[q_base + i * m + j];
          l[i * m + j] = select(pattern + 1, i * m + j) ? batch.l_upper[l_base + i * m + j]
                                                        : batch.l_lower[l_base + i * m + j];
        }
      }
      check_choice(m, q.data(), l.data(), out.lower[slot], out.upper[slot], points);
    }
  }
}

class Execution final {
 public:
  Execution(uint64_t work, uint64_t bytes, double seconds = INFINITY) {
    PHX_CHECK(phx_mc_execution_begin(work, 0, 0, bytes, seconds, nullptr, &scope_) == PHX_MC_OK);
  }
  Execution(const Execution&) = delete;
  Execution& operator=(const Execution&) = delete;
  ~Execution() {
    if (scope_ != nullptr) finish(PHX_MC_OK);
  }
  void* scope() const { return scope_; }
  std::array<uint64_t, PHX_MC_EXECUTION_COUNTERS> finish(
      int32_t status, std::array<uint64_t, PHX_MC_EXECUTION_MEMORY_VALUES>* memory_out = nullptr) {
    std::array<uint64_t, PHX_MC_EXECUTION_COUNTERS> counters{};
    std::array<uint64_t, PHX_MC_EXECUTION_MEMORY_VALUES> memory{};
    double seconds = 0.0;
    PHX_CHECK(phx_mc_execution_end(scope_, counters.data(), memory.data(), &seconds) == status);
    scope_ = nullptr;
    if (memory_out != nullptr) *memory_out = memory;
    return counters;
  }

 private:
  void* scope_ = nullptr;
};

class Policy final {
 public:
  explicit Policy(uint32_t threads) {
    PHX_CHECK(phx_mc_parallel_begin(static_cast<int32_t>(threads), INFINITY, &policy_) ==
              PHX_MC_OK);
  }
  ~Policy() { PHX_CHECK(phx_mc_parallel_end(policy_) == PHX_MC_OK); }
  Policy(const Policy&) = delete;
  Policy& operator=(const Policy&) = delete;

 private:
  void* policy_ = nullptr;
};

// ------------------------------------------------------------------- tests

// Degree 0/1/2 signed polynomials on identity, corner, midpoint, random and
// collapsed children in every dimension and component count: the bounds
// contain every exact child coefficient and dyadic image sample and are tight
// to the exact coefficient range. Identity children are exact.
void polynomial_degrees() {
  std::mt19937_64 random(20261022);
  for (int dimension = 1; dimension <= 3; ++dimension) {
    const auto points = child_points(dimension + 1);
    for (int components = 1; components <= 3; ++components) {
      Batch batch(dimension, components, 3 * static_cast<int64_t>(kMaps.size()) * 2);
      fill_point_batch(batch, random);
      Bounds out;
      PHX_CHECK(evaluate(batch, out) == PHX_MC_OK);
      const int m = batch.m();
      for (int64_t item = 0; item < batch.count; ++item) {
        const double* l = batch.l_lower.data() + batch.l_offset(item);
        for (int component = 0; component < components; ++component) {
          const double* q = batch.q_lower.data() + batch.q_offset(item, component);
          const auto slot = static_cast<std::size_t>(item * components + component);
          check_choice(m, q, l, out.lower[slot], out.upper[slot], points);
          check_tight(m, q, l, out.lower[slot], out.upper[slot]);
          if (kMaps[static_cast<std::size_t>(item) % kMaps.size()] == Map::Identity) {
            double minimum = q[0], maximum = q[0];
            for (int i = 0; i < m; ++i) {
              for (int j = i; j < m; ++j) {
                minimum = std::min(minimum, q[i * m + j]);
                maximum = std::max(maximum, q[i * m + j]);
              }
            }
            PHX_CHECK(out.lower[slot] == minimum && out.upper[slot] == maximum);
          }
        }
      }
    }
  }
}

// Interval coefficients and interval child maps: every sampled endpoint
// corner, including rows that do not sum to one, lies in the bounds.
void interval_inputs() {
  std::mt19937_64 random(7);
  for (int dimension = 1; dimension <= 3; ++dimension) {
    const auto points = child_points(dimension + 1);
    for (int components = 1; components <= 3; ++components) {
      Batch batch(dimension, components, 24);
      fill_point_batch(batch, random);
      widen(batch);
      Bounds out;
      PHX_CHECK(evaluate(batch, out) == PHX_MC_OK);
      for (int64_t item = 0; item < batch.count; ++item) {
        check_interval_item(batch, out, item, points);
      }
    }
  }
}

// Gradual underflow in products and endpoints, overflow to an unbounded
// endpoint, and unbounded coefficient sides. No NaN is produced.
void extreme_magnitudes() {
  std::mt19937_64 random(11);
  for (int dimension = 1; dimension <= 3; ++dimension) {
    const int m = dimension + 1;
    const auto points = child_points(m);
    // Subnormal coefficients under fractional children underflow every product.
    {
      Batch batch(dimension, 2, 4);
      const std::array<Map, 4> maps{Map::Corner, Map::Midpoint, Map::Random, Map::CollapsedEdge};
      for (int64_t item = 0; item < batch.count; ++item) {
        fill_map(batch, item, maps[static_cast<std::size_t>(item)], random);
        for (int i = 0; i < m; ++i) {
          for (int j = i; j < m; ++j) {
            const double sign = (i + j + item) % 2 == 0 ? 1.0 : -1.0;
            const double tiny = sign * (i + 2 * j + 1) * kSmallest;
            const double small = sign * (j - i + 3) * 0x1p-1060;
            batch.coefficient(item, 0, i, j, tiny, tiny);
            batch.coefficient(item, 1, i, j, small, small);
          }
        }
      }
      Bounds out;
      PHX_CHECK(evaluate(batch, out) == PHX_MC_OK);
      for (int64_t item = 0; item < batch.count; ++item) {
        const double* l = batch.l_lower.data() + batch.l_offset(item);
        for (int component = 0; component < 2; ++component) {
          const double* q = batch.q_lower.data() + batch.q_offset(item, component);
          const auto slot = static_cast<std::size_t>(item * 2 + component);
          check_choice(m, q, l, out.lower[slot], out.upper[slot], points);
          check_tight(m, q, l, out.lower[slot], out.upper[slot]);
        }
      }
    }
    // Subnormal barycentric endpoints.
    {
      Batch batch(dimension, 1, 1);
      for (int k = 0; k < m; ++k) {
        for (int i = 0; i < m; ++i) {
          const bool tiny = i == k;
          batch.vertex(0, k, i, tiny ? kSmallest : 0.25, tiny ? 2 * kSmallest : 1.0);
        }
      }
      for (int i = 0; i < m; ++i) {
        for (int j = i; j < m; ++j) {
          const double value = dyadic_value(random);
          batch.coefficient(0, 0, i, j, value, value + 0.5);
        }
      }
      Bounds out;
      PHX_CHECK(evaluate(batch, out) == PHX_MC_OK);
      check_interval_item(batch, out, 0, points);
    }
    // Largest finite coefficients: a constant kLargest overflows the upper
    // midpoint sums to +inf while the lower endpoint stays finite; signed
    // and interval largest coefficients remain NaN-free enclosures.
    {
      Batch batch(dimension, 3, 3);
      const std::array<Map, 3> maps{Map::Midpoint, Map::Random, Map::Corner};
      for (int64_t item = 0; item < batch.count; ++item) {
        fill_map(batch, item, maps[static_cast<std::size_t>(item)], random);
        for (int i = 0; i < m; ++i) {
          for (int j = i; j < m; ++j) {
            batch.coefficient(item, 0, i, j, kLargest, kLargest);
            const double signed_value = i == j ? kLargest : -kLargest;
            batch.coefficient(item, 1, i, j, signed_value, signed_value);
            batch.coefficient(item, 2, i, j, -kLargest, kLargest);
          }
        }
      }
      Bounds out;
      PHX_CHECK(evaluate(batch, out) == PHX_MC_OK);
      PHX_CHECK(out.upper[0] == kInfinity && std::isfinite(out.lower[0]) &&
                out.lower[0] <= kLargest);
      for (int64_t item = 0; item < batch.count; ++item) {
        const double* l = batch.l_lower.data() + batch.l_offset(item);
        for (int component = 0; component < 2; ++component) {
          const double* q = batch.q_lower.data() + batch.q_offset(item, component);
          const auto slot = static_cast<std::size_t>(item * 3 + component);
          check_choice(m, q, l, out.lower[slot], out.upper[slot], points);
        }
        check_interval_item(batch, out, item, points);
      }
    }
  }
  // Unbounded coefficient sides on the triangle: a child whose rows ignore
  // root vertex 0 stays finite (0 * unbounded endpoint = 0); a child touching
  // it inherits the unbounded side only.
  {
    const int m = 3;
    const auto points = child_points(m);
    Batch batch(2, 1, 3);
    for (int64_t item = 0; item < batch.count; ++item) {
      for (int i = 0; i < m; ++i) {
        for (int j = i; j < m; ++j) batch.coefficient(item, 0, i, j, 1.0, 1.0);
      }
    }
    batch.coefficient(0, 0, 0, 0, -kInfinity, 1.0);
    batch.coefficient(1, 0, 0, 0, -kInfinity, 1.0);
    batch.coefficient(2, 0, 0, 1, 0.0, kInfinity);
    const std::array<std::array<double, 3>, 3> face{{{0.0, 1.0, 0.0}, {0.0, 0.5, 0.5},
                                                     {0.0, 0.0, 1.0}}};
    for (int k = 0; k < m; ++k) {
      for (int i = 0; i < m; ++i) batch.vertex(0, k, i, face[k][i], face[k][i]);
    }
    fill_map(batch, 1, Map::Corner, random);
    fill_map(batch, 2, Map::Corner, random);
    Bounds out;
    PHX_CHECK(evaluate(batch, out) == PHX_MC_OK);
    PHX_CHECK(std::isfinite(out.lower[0]) && std::isfinite(out.upper[0]));
    PHX_CHECK(out.lower[1] == -kInfinity && std::isfinite(out.upper[1]));
    PHX_CHECK(std::isfinite(out.lower[2]) && out.upper[2] == kInfinity);
    for (int64_t item = 0; item < batch.count; ++item) {
      const double* l = batch.l_lower.data() + batch.l_offset(item);
      for (double value : {-kLargest, -1.0, 0.0, 1.0, kLargest}) {
        std::array<double, 9> q{};
        std::fill(q.begin(), q.end(), 1.0);
        if (item < 2 && value <= 1.0) {
          q[0] = value;
        } else if (item == 2 && value >= 0.0) {
          q[1] = q[3] = value;
        } else {
          continue;
        }
        check_choice(m, q.data(), l, out.lower[static_cast<std::size_t>(item)],
                     out.upper[static_cast<std::size_t>(item)], points);
      }
    }
  }
}

// Every rejected input writes no output; a nonfinite value anywhere outranks
// an invalid one; rows summing to one only to rounding are rejected exactly.
void invalid_inputs() {
  std::mt19937_64 random(3);
  Batch base(2, 2, 150);  // Two full validation chunks and one partial chunk.
  fill_point_batch(base, random);
  const int m = base.m();
  const auto expect = [](const Batch& batch, int32_t status) {
    Bounds out;
    PHX_CHECK(evaluate(batch, out) == status);
    if (status != PHX_MC_OK) PHX_CHECK(untouched(out));
  };
  expect(base, PHX_MC_OK);
  const auto mutated = [&](auto&& change, int32_t status) {
    Batch batch = base;
    change(batch);
    expect(batch, status);
  };
  const auto set_row = [&](Batch& batch, int64_t item, int k, std::array<double, 3> low,
                           std::array<double, 3> high) {
    for (int i = 0; i < m; ++i) batch.vertex(item, k, i, low[i], high[i]);
  };
  // Invalid intervals and maps.
  mutated([&](Batch& b) { b.q_lower[b.q_offset(5, 1) + 0 * m + 1] -= 1.0; }, PHX_MC_INVALID_INPUT);
  mutated([&](Batch& b) { b.q_upper[b.q_offset(7, 0) + 2 * m + 0] += 1.0; }, PHX_MC_INVALID_INPUT);
  mutated([&](Batch& b) {
    const double high = b.q_upper[b.q_offset(9, 1) + 1 * m + 2];
    b.coefficient(9, 1, 1, 2, high + 1.0, high);
  }, PHX_MC_INVALID_INPUT);
  mutated([&](Batch& b) { set_row(b, 64, 0, {-0.25, 0.5, 0.75}, {0.0, 0.5, 0.75}); },
          PHX_MC_INVALID_INPUT);
  mutated([&](Batch& b) { set_row(b, 65, 1, {0.0, 0.5, 0.5}, {0.0, 0.5, 1.25}); },
          PHX_MC_INVALID_INPUT);
  mutated([&](Batch& b) { set_row(b, 66, 2, {0.75, 0.25, 0.0}, {0.5, 0.5, 0.0}); },
          PHX_MC_INVALID_INPUT);
  mutated([&](Batch& b) { set_row(b, 67, 0, {0.25, 0.25, 0.25}, {0.25, 0.25, 0.25}); },
          PHX_MC_INVALID_INPUT);
  // 1 + 2^-53 and 1 - 2^-54 both round to one; the exact row sums do not.
  const double above = 0.5 + 0x1p-53;
  const double below = 0.5 - 0x1p-54;
  PHX_CHECK(0.5 + above == 1.0 && 0.5 + below == 1.0);
  mutated([&](Batch& b) { set_row(b, 149, 0, {0.5, above, 0.0}, {0.5, above, 0.0}); },
          PHX_MC_INVALID_INPUT);
  mutated([&](Batch& b) { set_row(b, 149, 1, {0.5, below, 0.0}, {0.5, below, 0.0}); },
          PHX_MC_INVALID_INPUT);
  // Accepted extremes: exact interval sums and unbounded coefficient sides.
  mutated([&](Batch& b) { set_row(b, 3, 0, {0.5, below, 0.0}, {0.5, above, 0.0}); }, PHX_MC_OK);
  mutated([&](Batch& b) {
    b.coefficient(4, 0, 0, 0, -kInfinity, b.q_upper[b.q_offset(4, 0)]);
    b.coefficient(4, 1, 1, 2, b.q_lower[b.q_offset(4, 1) + 1 * m + 2], kInfinity);
  }, PHX_MC_OK);
  // Nonfinite input, including reversed infinities and barycentric infinities.
  for (double value : {kNaN, kInfinity}) {
    mutated([&](Batch& b) { b.q_lower[b.q_offset(70, 1) + 4] = value; }, PHX_MC_NONFINITE_INPUT);
  }
  for (double value : {kNaN, -kInfinity}) {
    mutated([&](Batch& b) { b.q_upper[b.q_offset(71, 0) + 8] = value; }, PHX_MC_NONFINITE_INPUT);
  }
  for (double value : {kNaN, kInfinity, -kInfinity}) {
    mutated([&](Batch& b) { b.l_lower[b.l_offset(128) + 1] = value; }, PHX_MC_NONFINITE_INPUT);
    mutated([&](Batch& b) { b.l_upper[b.l_offset(129) + 5] = value; }, PHX_MC_NONFINITE_INPUT);
  }
  // An earlier invalid item never masks a later nonfinite one.
  mutated([&](Batch& b) {
    b.q_lower[b.q_offset(0, 0) + 1] -= 1.0;
    b.l_upper[b.l_offset(149) + 8] = kNaN;
  }, PHX_MC_NONFINITE_INPUT);
  // Arguments.
  Bounds out;
  out.lower.assign(static_cast<std::size_t>(base.count * base.components), kSentinel);
  out.upper = out.lower;
  const auto call = [&](int64_t count, int32_t dimension, int32_t components, const double* ql,
                        const double* qu, const double* ll, const double* lu, double* lower,
                        double* upper) {
    return phx_mc_simplex_quadratic_bounds(count, dimension, components, ql, qu, ll, lu, lower,
                                           upper);
  };
  const double* ql = base.q_lower.data();
  const double* qu = base.q_upper.data();
  const double* ll = base.l_lower.data();
  const double* lu = base.l_upper.data();
  double* lower = out.lower.data();
  double* upper = out.upper.data();
  for (int32_t dimension : {0, 4, -1}) {
    PHX_CHECK(call(1, dimension, 1, ql, qu, ll, lu, lower, upper) == PHX_MC_INVALID_ARGUMENT);
  }
  for (int32_t components : {0, 4, -1}) {
    PHX_CHECK(call(1, 2, components, ql, qu, ll, lu, lower, upper) == PHX_MC_INVALID_ARGUMENT);
  }
  PHX_CHECK(call(-1, 2, 2, ql, qu, ll, lu, lower, upper) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(INT64_MAX, 2, 2, ql, qu, ll, lu, lower, upper) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(1, 2, 2, nullptr, qu, ll, lu, lower, upper) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(1, 2, 2, ql, nullptr, ll, lu, lower, upper) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(1, 2, 2, ql, qu, nullptr, lu, lower, upper) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(1, 2, 2, ql, qu, ll, nullptr, lower, upper) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(1, 2, 2, ql, qu, ll, lu, nullptr, upper) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(1, 2, 2, ql, qu, ll, lu, lower, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(0, 2, 2, nullptr, nullptr, nullptr, nullptr, nullptr, upper) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(untouched(out));
  PHX_CHECK(call(0, 2, 2, nullptr, nullptr, nullptr, nullptr, lower, upper) == PHX_MC_OK);
  PHX_CHECK(untouched(out));
}

// The exact logical work R = count * components (m^3 + m^2 (m + 1) / 2) is
// admitted and charged at every parallelism with adequate scratch; R - 1 is
// refused before any output is written. This kernel's own scratch peak S,
// measured once per policy, is admitted exactly as a cap on the same input
// and policy, and S - 1 is refused with outputs untouched. A refused or
// expired scope outranks input errors.
void resource_admission() {
  std::mt19937_64 random(5);
  Batch batch(3, 3, 200);  // Three full chunks and one partial chunk.
  fill_point_batch(batch, random);
  widen(batch);
  Bounds reference;
  PHX_CHECK(evaluate(batch, reference) == PHX_MC_OK);
  const uint64_t required = static_cast<uint64_t>(batch.count) * item_work(3, 3);
  PHX_CHECK(required == 200u * 312u);
  for (uint32_t threads : {1u, 4u, 7u}) {
    Policy policy(threads);
    uint64_t scratch = 0;
    {
      Execution execution(required, UINT64_MAX);
      Bounds out;
      const int32_t status = evaluate(batch, out);
      std::array<uint64_t, PHX_MC_EXECUTION_MEMORY_VALUES> memory{};
      const auto counters = execution.finish(PHX_MC_OK, &memory);
      PHX_CHECK(status == PHX_MC_OK);
      PHX_CHECK(identical(out.lower, reference.lower) && identical(out.upper, reference.upper));
      PHX_CHECK(counters[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == required);
      PHX_CHECK(counters[PHX_MC_EXECUTION_WORK_REFUSALS] == 0);
      scratch = memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES];
    }
    {
      Execution execution(required - 1, UINT64_MAX);
      Bounds out;
      const int32_t status = evaluate(batch, out);
      const auto counters = execution.finish(PHX_MC_CAPACITY_EXCEEDED);
      PHX_CHECK(status == PHX_MC_CAPACITY_EXCEEDED);
      PHX_CHECK(untouched(out));
      PHX_CHECK(counters[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 0);
      PHX_CHECK(counters[PHX_MC_EXECUTION_WORK_REFUSALS] == 1);
    }
    PHX_CHECK(scratch > 0);
    {
      Execution execution(UINT64_MAX, scratch);
      Bounds out;
      const int32_t status = evaluate(batch, out);
      const auto counters = execution.finish(PHX_MC_OK);
      PHX_CHECK(status == PHX_MC_OK);
      PHX_CHECK(identical(out.lower, reference.lower) && identical(out.upper, reference.upper));
      PHX_CHECK(counters[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == required);
    }
    if (scratch > 0) {
      Execution execution(UINT64_MAX, scratch - 1);
      Bounds out;
      const int32_t status = evaluate(batch, out);
      const auto counters = execution.finish(PHX_MC_CAPACITY_EXCEEDED);
      PHX_CHECK(status == PHX_MC_CAPACITY_EXCEEDED);
      PHX_CHECK(untouched(out));
      PHX_CHECK(counters[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 0);
      PHX_CHECK(counters[PHX_MC_EXECUTION_WORK_REFUSALS] == 0);
    }
  }
  Batch invalid = batch;
  invalid.q_lower.front() = kNaN;
  for (const Batch* input : {&batch, &invalid}) {
    {
      Execution execution(0, UINT64_MAX);
      PHX_CHECK(phx_mc_execution_charge(execution.scope(), 1, 0) == PHX_MC_CAPACITY_EXCEEDED);
      Bounds out;
      PHX_CHECK(evaluate(*input, out) == PHX_MC_CAPACITY_EXCEEDED);
      PHX_CHECK(untouched(out));
      execution.finish(PHX_MC_CAPACITY_EXCEEDED);
    }
    {
      Execution execution(UINT64_MAX, UINT64_MAX, kDeadlineSeconds);
      PHX_CHECK(phx_mc_execution_charge(execution.scope(), 0, 0) == PHX_MC_OK);
      std::this_thread::sleep_for(kPastDeadline);
      Bounds out;
      PHX_CHECK(evaluate(*input, out) == PHX_MC_TIMEOUT);
      PHX_CHECK(untouched(out));
      const auto counters = execution.finish(PHX_MC_TIMEOUT);
      PHX_CHECK(counters[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 0);
    }
  }
}

// Items are independent: outputs are bitwise identical at every parallelism,
// under item reversal, and one item at a time.
void deterministic_ordering() {
  std::mt19937_64 random(13);
  Batch batch(3, 3, 300);
  fill_point_batch(batch, random);
  widen(batch);
  Bounds reference;
  PHX_CHECK(evaluate(batch, reference) == PHX_MC_OK);
  for (uint32_t threads : {2u, 4u, 7u, 64u}) {
    Policy policy(threads);
    Bounds out;
    PHX_CHECK(evaluate(batch, out) == PHX_MC_OK);
    PHX_CHECK(identical(out.lower, reference.lower) && identical(out.upper, reference.upper));
  }
  Batch reversed(3, 3, batch.count);
  const std::size_t q_item = static_cast<std::size_t>(3 * 16);
  const std::size_t l_item = 16;
  for (int64_t item = 0; item < batch.count; ++item) {
    const auto from = static_cast<std::size_t>(item);
    const auto to = static_cast<std::size_t>(batch.count - 1 - item);
    std::copy_n(batch.q_lower.begin() + from * q_item, q_item, reversed.q_lower.begin() + to * q_item);
    std::copy_n(batch.q_upper.begin() + from * q_item, q_item, reversed.q_upper.begin() + to * q_item);
    std::copy_n(batch.l_lower.begin() + from * l_item, l_item, reversed.l_lower.begin() + to * l_item);
    std::copy_n(batch.l_upper.begin() + from * l_item, l_item, reversed.l_upper.begin() + to * l_item);
  }
  Bounds backward;
  PHX_CHECK(evaluate(reversed, backward) == PHX_MC_OK);
  bool mirrored = true;
  for (int64_t item = 0; item < batch.count; ++item) {
    for (int component = 0; component < 3; ++component) {
      const auto a = static_cast<std::size_t>(item * 3 + component);
      const auto b = static_cast<std::size_t>((batch.count - 1 - item) * 3 + component);
      mirrored = mirrored && std::memcmp(&reference.lower[a], &backward.lower[b], sizeof(double)) == 0 &&
                 std::memcmp(&reference.upper[a], &backward.upper[b], sizeof(double)) == 0;
    }
  }
  PHX_CHECK(mirrored);
  bool single = true;
  for (int64_t item = 0; item < batch.count; item += 37) {
    std::array<double, 3> lower{}, upper{};
    PHX_CHECK(phx_mc_simplex_quadratic_bounds(
                  1, 3, 3, batch.q_lower.data() + batch.q_offset(item, 0),
                  batch.q_upper.data() + batch.q_offset(item, 0),
                  batch.l_lower.data() + batch.l_offset(item),
                  batch.l_upper.data() + batch.l_offset(item), lower.data(), upper.data()) ==
              PHX_MC_OK);
    const auto slot = static_cast<std::size_t>(item * 3);
    single = single &&
             std::memcmp(lower.data(), &reference.lower[slot], 3 * sizeof(double)) == 0 &&
             std::memcmp(upper.data(), &reference.upper[slot], 3 * sizeof(double)) == 0;
  }
  PHX_CHECK(single);
}

// A calling thread left with flush-to-zero/flushed subnormal operands or a
// directed rounding mode: validation and every chunk establish their own
// round-to-nearest gradual-underflow mode, so subnormal-sensitive outputs are
// bitwise those of the default mode, exact row sums still reject 1 + 2^-53,
// and the caller's mode fields are restored after every call, whatever the
// parallelism. Runs first, so lazily created pool workers may inherit a
// hazardous mode where thread creation inherits it. Only integer comparisons
// and memcmp decide checks while a hazard is installed.
void caller_floating_point_modes() {
  PHX_CHECK(phx::mc::gradual_nearest_arithmetic());
  std::mt19937_64 random(17);
  Batch batch(3, 3, 130);
  fill_point_batch(batch, random);
  // Exact scaling of short dyadics puts coefficients around and below the
  // least normal binary64 (2^-1034 .. 2^-1015); odd items carry interval maps.
  for (std::size_t index = 0; index < batch.q_lower.size(); ++index) {
    batch.q_lower[index] = std::ldexp(batch.q_lower[index], -1025);
    batch.q_upper[index] = std::ldexp(batch.q_upper[index], -1025);
  }
  for (int64_t item = 1; item < batch.count; item += 2) {
    const std::size_t base = batch.l_offset(item);
    for (std::size_t entry = 0; entry < 16; ++entry) {
      batch.l_lower[base + entry] = std::max(0.0, batch.l_lower[base + entry] - 0x1p-20);
      batch.l_upper[base + entry] = std::min(1.0, batch.l_upper[base + entry] + 0x1p-20);
    }
  }
  Bounds reference;
  PHX_CHECK(evaluate(batch, reference) == PHX_MC_OK);
  const auto points = child_points(4);
  for (int64_t item = 0; item < batch.count; ++item) {
    if (item % 2 == 1) {
      check_interval_item(batch, reference, item, points);
      continue;
    }
    const double* l = batch.l_lower.data() + batch.l_offset(item);
    for (int component = 0; component < 3; ++component) {
      const double* q = batch.q_lower.data() + batch.q_offset(item, component);
      const auto slot = static_cast<std::size_t>(item * 3 + component);
      check_choice(4, q, l, reference.lower[slot], reference.upper[slot], points);
      check_tight(4, q, l, reference.lower[slot], reference.upper[slot]);
    }
  }
  Batch rejected = batch;
  const std::array<double, 4> row{0.5, 0.5 + 0x1p-53, 0.0, 0.0};
  for (int i = 0; i < 4; ++i) rejected.vertex(129, 2, i, row[i], row[i]);

  struct Hazard {
    bool flushing;
    int rounding;
  };
  const std::array<Hazard, 5> hazards{{{true, FE_TONEAREST},
                                       {false, FE_UPWARD},
                                       {false, FE_DOWNWARD},
                                       {false, FE_TOWARDZERO},
                                       {true, FE_DOWNWARD}}};
  const phx::mc::FloatingPointMode initial_mode = phx::mc::floating_point_mode();
  const int initial_rounding = std::fegetround();
  for (const Hazard& hazard : hazards) {
    if (hazard.flushing && !phx::mc::kFlushControl) continue;
    phx::mc::set_floating_point_mode(hazard.flushing ? phx::mc::kFlushingMode
                                                     : phx::mc::kGradualNearestMode);
    PHX_CHECK(std::fesetround(hazard.rounding) == 0);
    const phx::mc::FloatingPointMode installed = phx::mc::floating_point_mode();
    PHX_CHECK(!phx::mc::gradual_nearest_arithmetic());
    {
      const phx::mc::GradualNearestScope outer;
      PHX_CHECK(outer.established());
      {
        const phx::mc::GradualNearestScope inner;
        PHX_CHECK(inner.established());
      }
      PHX_CHECK(phx::mc::gradual_nearest_arithmetic());
    }
    PHX_CHECK(phx::mc::floating_point_mode() == installed);
    for (uint32_t threads : {1u, 4u}) {
      Policy policy(threads);
      Bounds out;
      PHX_CHECK(evaluate(batch, out) == PHX_MC_OK);
      PHX_CHECK(identical(out.lower, reference.lower) && identical(out.upper, reference.upper));
      PHX_CHECK(phx::mc::floating_point_mode() == installed &&
                std::fegetround() == hazard.rounding);
      Bounds refused;
      PHX_CHECK(evaluate(rejected, refused) == PHX_MC_INVALID_INPUT);
      PHX_CHECK(untouched(refused));
      PHX_CHECK(phx::mc::floating_point_mode() == installed &&
                std::fegetround() == hazard.rounding);
    }
    PHX_CHECK(!phx::mc::gradual_nearest_arithmetic());
  }
  phx::mc::set_floating_point_mode(initial_mode);
  PHX_CHECK(std::fesetround(initial_rounding) == 0);
  PHX_CHECK(phx::mc::floating_point_mode() == initial_mode);
  PHX_CHECK(phx::mc::gradual_nearest_arithmetic());
}

}  // namespace

int main() {
  caller_floating_point_modes();
  polynomial_degrees();
  interval_inputs();
  extreme_magnitudes();
  invalid_inputs();
  resource_admission();
  deterministic_ordering();
  return phx::mc::test::finish("test_quadratic_bounds");
}
