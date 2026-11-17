// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {
using Index = std::array<int, 4>;
void indices(int left, int position, int parts, Index& index, std::vector<Index>& out) {
  if (position == parts - 1) {
    index[position] = left;
    out.push_back(index);
    return;
  }
  for (int i = left; i >= 0; --i) {
    index[position] = i;
    indices(left - i, position + 1, parts, index, out);
  }
}

double polynomial(const std::array<double, 3>& x, int dimension) {
  return 1.0 + x[0] * x[0] + (dimension > 1 ? 2.0 * x[1] : 0.0) -
         (dimension > 2 ? x[2] : 0.0);
}

void exercise(int dimension) {
  constexpr int count = 73;
  const int stride = dimension * (dimension + 1);
  std::vector<Index> net;
  Index index{};
  indices(2, 0, dimension + 1, index, net);
  std::vector<double> controls;
  for (const auto& alpha : net) {
    controls.push_back(1.0 + alpha[1] * (alpha[1] - 1) / 2.0 +
        (dimension > 1 ? alpha[2] : 0.0) - (dimension > 2 ? alpha[3] / 2.0 : 0.0));
  }
  std::vector<double> corners(count * stride), lower(count), upper(count),
      parallel_lower(count), parallel_upper(count), split_lower(count), split_upper(count);
  for (int item = 0; item < count; ++item) {
    for (int vertex = 0; vertex <= dimension; ++vertex) {
      for (int axis = 0; axis < dimension; ++axis) {
        corners[item * stride + vertex * dimension + axis] =
            ((item % 7) - 3) * 0.125 + (vertex == axis + 1 ? 0.5 : -0.25) + axis * 0.125;
      }
    }
  }
  auto call = [&](int begin, int size, int workers, std::vector<double>& lo, std::vector<double>& hi) {
    return phx_mc_interval_simplex_restrict(dimension, 2, size, controls.data(), controls.data(),
        corners.data() + begin * stride, corners.data() + begin * stride,
        lo.data() + begin, hi.data() + begin, workers);
  };
  PHX_CHECK(call(0, count, 1, lower, upper) == PHX_MC_OK);
  PHX_CHECK(call(0, count, 4, parallel_lower, parallel_upper) == PHX_MC_OK);
  for (int begin = 0; begin < count;) {
    const int size = std::min(11, count - begin);
    PHX_CHECK(call(begin, size, 4, split_lower, split_upper) == PHX_MC_OK);
    begin += size;
  }
  PHX_CHECK(std::memcmp(lower.data(), parallel_lower.data(), count * sizeof(double)) == 0);
  PHX_CHECK(std::memcmp(upper.data(), parallel_upper.data(), count * sizeof(double)) == 0);
  PHX_CHECK(std::memcmp(lower.data(), split_lower.data(), count * sizeof(double)) == 0);
  PHX_CHECK(std::memcmp(upper.data(), split_upper.data(), count * sizeof(double)) == 0);
  for (int item = 0; item < count; ++item) {
    PHX_CHECK(std::isfinite(lower[item]) && std::isfinite(upper[item]));
    for (int a = 0; a <= dimension; ++a) {
      for (int b = 0; b <= dimension; ++b) {
        std::array<double, 3> x{};
        for (int axis = 0; axis < dimension; ++axis) {
          x[axis] = 0.25 * corners[item * stride + a * dimension + axis] +
                    0.75 * corners[item * stride + b * dimension + axis];
        }
        const double value = polynomial(x, dimension);
        PHX_CHECK(lower[item] <= value && value <= upper[item]);
      }
    }
  }
}
}  // namespace

int main() {
  for (int dimension = 1; dimension <= 3; ++dimension) exercise(dimension);
  const double one = 1.0;
  const double corners[] = {-2.0, 3.0};
  double lo = 0.0, hi = 0.0;
  PHX_CHECK(phx_mc_interval_simplex_restrict(1, 0, 1, &one, &one, corners, corners,
                                             &lo, &hi, 1) == PHX_MC_OK);
  PHX_CHECK(lo == 1.0 && hi == 1.0);
  const double invalid[] = {std::numeric_limits<double>::quiet_NaN(), 1.0};
  PHX_CHECK(phx_mc_interval_simplex_restrict(1, 0, 1, &one, &one, invalid, invalid,
                                             &lo, &hi, 1) == PHX_MC_OK);
  PHX_CHECK(lo == -INFINITY && hi == INFINITY);
  const double quadratic_lo[] = {0.0, 0.0, 0.9};
  const double quadratic_hi[] = {0.0, 0.0, 1.1};
  const double uncertain_lo[] = {-1.25, 1.5};
  const double uncertain_hi[] = {-0.75, 2.0};
  PHX_CHECK(phx_mc_interval_simplex_restrict(1, 2, 1, quadratic_lo, quadratic_hi,
      uncertain_lo, uncertain_hi, &lo, &hi, 1) == PHX_MC_OK);
  for (double coefficient : {0.9, 1.1}) {
    for (double first : {-1.25, -0.75}) {
      for (double last : {1.5, 2.0}) {
        for (int t = 0; t <= 4; ++t) {
          const double x = (first * (4 - t) + last * t) / 4.0;
          const double value = coefficient * x * x;
          PHX_CHECK(lo <= value && value <= hi);
        }
      }
    }
  }
  PHX_CHECK(phx_mc_interval_simplex_restrict(0, 0, 0, nullptr, nullptr, nullptr, nullptr,
                                             nullptr, nullptr, 1) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_interval_simplex_restrict(1, 0, 0, nullptr, nullptr, nullptr, nullptr,
                                             nullptr, nullptr, 1) == PHX_MC_OK);
  const int original_round = std::fegetround();
  PHX_CHECK(std::fesetround(FE_UPWARD) == 0);
  PHX_CHECK(phx_mc_interval_simplex_restrict(1, 0, 1, &one, &one, corners, corners,
                                             &lo, &hi, 1) == PHX_MC_OK);
  PHX_CHECK(std::fegetround() == FE_UPWARD);
  PHX_CHECK(std::fesetround(original_round) == 0);
}
