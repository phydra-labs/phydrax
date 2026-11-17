// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "check.hpp"
#include "phydrax_meshcore.h"
#include "wide_integer.hpp"
#include <array>
#include <cstdint>
#include <vector>

int main() {
  using W = phx::mc::Int256;
  // Basis (1-x-y, x, y), routed to a triangle with area 3.
  const int64_t offsets[] = {0, 3, 4, 5};
  const int32_t powers[] = {0,0, 1,0, 0,1, 1,0, 0,1};
  std::vector<uint8_t> basis(5 * W::kBytes);
  const int64_t coefficients[] = {1,-1,-1,1,1};
  for (int i = 0; i < 5; ++i) W::from_int64(coefficients[i]).store(basis.data() + i * W::kBytes);
  const int64_t sig[] = {0,0, 1,0, 0,3};
  const int16_t exp[] = {0,0, 1,0, 0,0};
  const int64_t sparse_index[] = {0}, sparse_rows[] = {0,0,0,0};
  const uint8_t empty[] = {0};
  const int64_t routes[] = {0,1,2, 0,2,1, 0,3,1};
  std::array<uint8_t, 3 * W::kBytes> sums{}, scales{};
  int64_t denominators[3]{}, counts[3]{};
  int32_t statuses[3]{};
  auto call = [&](int workers) {
    return phx_mc_basis_jacobian_integrals(3, 2, PHX_MC_BERNSTEIN_SIMPLEX, 3, 3,
      routes, sig, exp, sparse_index, sparse_rows, empty, empty, offsets, powers, basis.data(),
      sums.data(), scales.data(), denominators, counts, statuses, 4, workers);
  };
  PHX_CHECK(call(1) == PHX_MC_OK);
  PHX_CHECK(statuses[0] == PHX_MC_OK && statuses[1] == PHX_MC_OK && statuses[2] == PHX_MC_RANGE_ERROR);
  PHX_CHECK(W::load(sums.data()).limb == W::from_int64(6).limb);
  PHX_CHECK(W::load(sums.data() + W::kBytes).limb == W::from_int64(-6).limb);
  PHX_CHECK(W::load(scales.data()).limb == W::from_int64(1).limb);
  PHX_CHECK(denominators[0] == 1 && counts[0] == 1);
  const auto serial = sums;
  PHX_CHECK(call(4) == PHX_MC_OK && sums == serial);
  // One sparse coordinate is 1/3; the common coordinate scale is exactly 3.
  const int64_t sparse_sig[] = {0,0, 0,0, 0,3};
  const int16_t sparse_exp[] = {0,0, INT16_MIN,0, 0,0};
  const int64_t sparse_indices[] = {2}, sparse_starts[] = {0,0,1,1};
  std::array<uint8_t, W::kBytes> numerator{}, denominator{};
  W::from_int64(1).store(numerator.data());
  W::from_int64(3).store(denominator.data());
  PHX_CHECK(phx_mc_basis_jacobian_integrals(3, 2, PHX_MC_BERNSTEIN_SIMPLEX, 3, 3,
      routes, sparse_sig, sparse_exp, sparse_indices, sparse_starts, numerator.data(),
      denominator.data(), offsets, powers, basis.data(), sums.data(), scales.data(),
      denominators, counts, statuses, 4, 2) == PHX_MC_OK);
  PHX_CHECK(statuses[0] == PHX_MC_OK && statuses[2] == PHX_MC_RANGE_ERROR);
  PHX_CHECK(W::load(scales.data()).limb == W::from_int64(3).limb);
  PHX_CHECK(W::load(sums.data()).limb == W::from_int64(9).limb);
  return 0;
}
