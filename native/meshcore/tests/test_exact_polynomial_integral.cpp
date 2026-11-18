// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "check.hpp"
#include "phydrax_meshcore.h"
#include "wide_integer.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

int main() {
  using W = phx::mc::Int256;
  constexpr int64_t n = 80;
  std::vector<int64_t> offsets(n * 4 + 1), denominators(n), counts(n);
  std::vector<int32_t> exponents(n * 8, 0), statuses(n);
  std::vector<uint8_t> input(n * 4 * W::kBytes), sums(n * W::kBytes);
  for (int64_t i = 0; i <= n * 4; ++i) offsets[i] = i;
  for (int64_t i = 0; i < n; ++i) {
    W::from_int64(i % 3 == 0 ? -2 : i % 3 == 1 ? 0 : 2).store(input.data() + (i * 4) * W::kBytes);
    W::from_int64(0).store(input.data() + (i * 4 + 1) * W::kBytes);
    W::from_int64(0).store(input.data() + (i * 4 + 2) * W::kBytes);
    W::from_int64(3).store(input.data() + (i * 4 + 3) * W::kBytes);
  }
  auto call = [&](int workers) {
    return phx_mc_polynomial_jacobian_integrals(n, 2, PHX_MC_BERNSTEIN_SIMPLEX,
        offsets.data(), exponents.data(), input.data(), sums.data(), denominators.data(),
        counts.data(), statuses.data(), 4, workers);
  };
  PHX_CHECK(call(1) == PHX_MC_OK);
  for (int64_t i = 0; i < n; ++i) {
    PHX_CHECK(statuses[i] == PHX_MC_OK);
    PHX_CHECK(denominators[i] == 1 && counts[i] == 1);
    auto expected = W::from_int64(i % 3 == 0 ? -6 : i % 3 == 1 ? 0 : 6);
    PHX_CHECK(W::load(sums.data() + i * W::kBytes).limb == expected.limb);
  }
  auto serial = sums;
  PHX_CHECK(call(4) == PHX_MC_OK && sums == serial);
  for (int64_t start = 0; start < n; start += 13) {
    const int64_t size = std::min<int64_t>(13, n - start);
    std::vector<int64_t> local_offsets(size * 4 + 1);
    for (int64_t i = 0; i <= size * 4; ++i) local_offsets[i] = i;
    PHX_CHECK(phx_mc_polynomial_jacobian_integrals(size, 2, PHX_MC_BERNSTEIN_SIMPLEX,
        local_offsets.data(), exponents.data() + start * 8, input.data() + start * 4 * W::kBytes,
        sums.data() + start * W::kBytes, denominators.data() + start, counts.data() + start,
        statuses.data() + start, 4, 3) == PHX_MC_OK);
  }
  PHX_CHECK(sums == serial);
  W huge; huge.limb[3] = uint64_t{1} << 62;
  huge.store(input.data());
  PHX_CHECK(call(4) == PHX_MC_OK);
  PHX_CHECK(statuses[0] == PHX_MC_RANGE_ERROR && statuses[1] == PHX_MC_OK);
  return phx::mc::test::finish("test_exact_polynomial_integral");
}
