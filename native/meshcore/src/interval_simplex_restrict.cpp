// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "interval_arithmetic.hpp"
#include "parallel.hpp"
#include "phydrax_meshcore.h"

namespace {
using Index = std::array<int32_t, 4>;
constexpr double inf = std::numeric_limits<double>::infinity();

// Here k <= 3. Cancel before multiplying so addressability checks cannot wrap.
uint64_t choose(uint64_t n, unsigned k) {
  if (n < k) return 0;
  std::array<uint64_t, 3> factors{1, 1, 1};
  for (unsigned i = 0; i < k; ++i) factors[i] = n - i;
  for (unsigned divisor = 2; divisor <= k; ++divisor) {
    uint64_t remaining = divisor;
    for (unsigned i = 0; i < k; ++i) {
      const uint64_t g = std::gcd(factors[i], remaining);
      factors[i] /= g;
      remaining /= g;
    }
  }
  uint64_t value = 1;
  for (unsigned i = 0; i < k; ++i) {
    if (factors[i] && value > uint64_t{INT64_MAX} / factors[i]) return UINT64_MAX;
    value *= factors[i];
  }
  return value;
}

void compositions(int32_t remaining, int position, int parts, Index& index,
                  std::vector<Index>& result) {
  if (position == parts - 1) {
    index[position] = remaining;
    result.push_back(index);
    return;
  }
  for (int64_t first = remaining; first >= 0; --first) {
    index[position] = static_cast<int32_t>(first);
    compositions(remaining - static_cast<int32_t>(first), position + 1, parts, index, result);
  }
}

std::size_t rank(const Index& index, int32_t degree, int parts) {
  int64_t remaining = degree;
  uint64_t result = 0;
  for (int i = 0; i < parts - 1; ++i) {
    const int rest_parts = parts - i - 1;
    result += choose(static_cast<uint64_t>(remaining - index[i] + rest_parts - 1),
                     static_cast<unsigned>(rest_parts));
    remaining -= index[i];
  }
  return static_cast<std::size_t>(result);
}

struct Level {
  std::vector<Index> indices;
  std::vector<std::array<std::size_t, 4>> children;
};

struct Workspace {
  explicit Workspace(std::size_t size)
      : lo(size), hi(size), next_lo(size), next_hi(size) {}
  std::vector<double> lo, hi, next_lo, next_hi;
};

bool finite_interval(double lower, double upper) {
  return std::isfinite(lower) && std::isfinite(upper) && lower <= upper;
}

// A Bernstein control of the restricted polynomial is the symmetric blossom
// evaluated at the corresponding multiset of new vertices. Each de Casteljau
// step is its affine, multi-affine recurrence in barycentric coordinates.
// Interval sums and products enclose every real operation, including for signed
// barycentric coordinates, so induction encloses every blossom value. The
// convex-hull property of the new simplex bounds the polynomial by the extrema
// of these controls. Dependency only widens the enclosure.
int32_t restrict_item(int dimension, int degree, const std::vector<Level>& levels,
                      const double* control_lo, const double* control_hi,
                      const double* corner_lo, const double* corner_hi,
                      double& out_lo, double& out_hi, Workspace& work) {
  const int parts = dimension + 1;
  const std::size_t controls = levels[degree].indices.size();
  out_lo = -inf;
  out_hi = inf;
  for (std::size_t i = 0; i < controls; ++i) {
    if (!finite_interval(control_lo[i], control_hi[i])) return PHX_MC_OK;
  }
  std::array<std::array<double, 4>, 4> bary_lo{}, bary_hi{};
  for (int vertex = 0; vertex < parts; ++vertex) {
    bary_lo[vertex][0] = bary_hi[vertex][0] = 1.0;
    for (int axis = 0; axis < dimension; ++axis) {
      const int i = vertex * dimension + axis;
      if (!finite_interval(corner_lo[i], corner_hi[i])) return PHX_MC_OK;
      bary_lo[vertex][axis + 1] = corner_lo[i];
      bary_hi[vertex][axis + 1] = corner_hi[i];
      double lo, hi;
      phx::mc::interval::interval_sum(bary_lo[vertex][0], bary_hi[vertex][0],
                                      corner_lo[i], corner_hi[i], true, lo, hi);
      if (!finite_interval(lo, hi)) return PHX_MC_OK;
      bary_lo[vertex][0] = lo;
      bary_hi[vertex][0] = hi;
    }
  }
  auto& [lo, hi, next_lo, next_hi] = work;
  double minimum = inf, maximum = -inf;
  for (const Index& target : levels[degree].indices) {
    std::copy_n(control_lo, controls, lo.begin());
    std::copy_n(control_hi, controls, hi.begin());
    int remaining = degree;
    for (int vertex = 0; vertex < parts; ++vertex) {
      for (int repeat = 0; repeat < target[vertex]; ++repeat) {
        const auto& level = levels[remaining - 1];
        const std::size_t size = level.indices.size();
        for (std::size_t i = 0; i < size; ++i) {
          double sum_lo = 0.0, sum_hi = 0.0;
          for (int axis = 0; axis < parts; ++axis) {
            const auto child = level.children[i][axis];
            double product_lo, product_hi, new_lo, new_hi;
            phx::mc::interval::interval_product(lo[child], hi[child], bary_lo[vertex][axis],
                                                bary_hi[vertex][axis], product_lo, product_hi);
            phx::mc::interval::interval_sum(sum_lo, sum_hi, product_lo, product_hi, false,
                                            new_lo, new_hi);
            sum_lo = new_lo;
            sum_hi = new_hi;
          }
          if (!finite_interval(sum_lo, sum_hi)) return PHX_MC_OK;
          next_lo[i] = sum_lo;
          next_hi[i] = sum_hi;
        }
        lo.swap(next_lo);
        hi.swap(next_hi);
        --remaining;
      }
    }
    minimum = std::min(minimum, lo[0]);
    maximum = std::max(maximum, hi[0]);
  }
  out_lo = minimum;
  out_hi = maximum;
  return PHX_MC_OK;
}
}  // namespace

extern "C" int32_t phx_mc_interval_simplex_restrict(
    int32_t dimension, int32_t degree, int64_t count, const double* control_lower,
    const double* control_upper, const double* corners_lower, const double* corners_upper,
    double* result_lower, double* result_upper, int32_t workers) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (dimension < 1 || dimension > 3 || degree < 0 || count < 0 || workers < 1)
      return PHX_MC_INVALID_ARGUMENT;
    const uint64_t controls = choose(static_cast<uint64_t>(degree) + dimension, dimension);
    if (controls > uint64_t{INT64_MAX} ||
        !phx::mc::addressable(1, static_cast<int64_t>(controls), sizeof(double)) ||
        !phx::mc::addressable(count, dimension * (dimension + 1), sizeof(double)) ||
        !phx::mc::addressable(count, 1, sizeof(double))) return PHX_MC_INVALID_ARGUMENT;
    if (count == 0) return PHX_MC_OK;
    if (!control_lower || !control_upper || !corners_lower || !corners_upper ||
        !result_lower || !result_upper) return PHX_MC_INVALID_ARGUMENT;
    std::vector<Level> levels(static_cast<std::size_t>(degree) + 1);
    for (int64_t n = 0; n <= degree; ++n) {
      auto& level = levels[static_cast<std::size_t>(n)];
      Index index{};
      compositions(static_cast<int32_t>(n), 0, dimension + 1, index, level.indices);
      if (n < degree) {
        level.children.resize(level.indices.size());
        for (std::size_t i = 0; i < level.indices.size(); ++i) {
          for (int axis = 0; axis <= dimension; ++axis) {
            Index child = level.indices[i];
            ++child[axis];
            level.children[i][axis] = rank(child, static_cast<int32_t>(n + 1), dimension + 1);
          }
        }
      }
    }
    const std::size_t stride = static_cast<std::size_t>(dimension * (dimension + 1));
    return phx::mc::parallel_for(static_cast<std::size_t>(count),
        phx::mc::ParallelRequest{16, static_cast<uint32_t>(workers)},
        [&](std::size_t begin, std::size_t end, phx::mc::ChunkContext&) -> int32_t {
          const phx::mc::GradualNearestScope worker_mode;
          if (!worker_mode.established()) return PHX_MC_INTERNAL_ERROR;
          Workspace work(static_cast<std::size_t>(controls));
          for (std::size_t i = begin; i < end; ++i) {
            const int32_t status = restrict_item(dimension, degree, levels, control_lower, control_upper,
                corners_lower + i * stride, corners_upper + i * stride, result_lower[i], result_upper[i], work);
            if (status != PHX_MC_OK) return status;
          }
          return PHX_MC_OK;
        });
  });
}
