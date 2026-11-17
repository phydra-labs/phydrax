// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "check.hpp"
#include "phydrax_meshcore.h"
#include "wide_integer.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

int main() {
  constexpr int64_t n = 80;
  using W = phx::mc::Int256;
  std::vector<int64_t> offsets(n * 4 + 1), output_offsets(n * 2 + 1), counts(n * 2);
  std::vector<int32_t> exponents(n * 8, 0), out_exponents(n * 4), status(n);
  std::vector<uint8_t> input(n * 4 * W::kBytes), output(n * 2 * W::kBytes);
  for (int64_t i = 0; i <= n * 4; ++i) offsets[i] = i;
  for (int64_t i = 0; i <= n * 2; ++i) output_offsets[i] = i;
  for (int64_t i = 0; i < n; ++i) {
    const std::array<int64_t, 4> matrix{2, 1, 1, 3};
    for (int k = 0; k < 4; ++k) W::from_int64(matrix[k]).store(input.data() + (4*i+k)*W::kBytes);
  }
  auto call = [&](int workers) {
    return phx_mc_symmetric_polynomial_minors(n, 2, offsets.data(), exponents.data(), input.data(),
      output_offsets.data(), out_exponents.data(), output.data(), counts.data(), status.data(), 4, workers);
  };
  PHX_CHECK(call(1) == PHX_MC_OK);
  for (int64_t i = 0; i < n; ++i) {
    PHX_CHECK(status[i] == PHX_MC_OK);
    PHX_CHECK(counts[2*i] == 1 && counts[2*i+1] == 1);
    PHX_CHECK(W::load(output.data() + 2*i*W::kBytes).limb[0] == 4);
    PHX_CHECK(W::load(output.data() + (2*i+1)*W::kBytes).limb[0] == 20);
  }
  const auto serial = output;
  PHX_CHECK(call(4) == PHX_MC_OK);
  PHX_CHECK(output == serial);
  for (int64_t begin = 0; begin < n; begin += 13) {
    const int64_t size = std::min<int64_t>(13, n - begin);
    std::vector<int64_t> local_offsets(size * 4 + 1), local_output_offsets(size * 2 + 1);
    std::vector<int64_t> local_counts(size * 2);
    std::vector<int32_t> local_exponents(size * 4), local_status(size);
    std::vector<uint8_t> local_output(size * 2 * W::kBytes);
    for (int64_t i = 0; i <= size * 4; ++i) local_offsets[i] = i;
    for (int64_t i = 0; i <= size * 2; ++i) local_output_offsets[i] = i;
    PHX_CHECK(phx_mc_symmetric_polynomial_minors(
        size, 2, local_offsets.data(), exponents.data() + begin * 8,
        input.data() + begin * 4 * W::kBytes, local_output_offsets.data(),
        local_exponents.data(), local_output.data(), local_counts.data(),
        local_status.data(), 4, 4) == PHX_MC_OK);
    PHX_CHECK(std::equal(local_output.begin(), local_output.end(),
                         serial.begin() + begin * 2 * W::kBytes));
    PHX_CHECK(std::equal(local_exponents.begin(), local_exponents.end(),
                         out_exponents.begin() + begin * 4));
    PHX_CHECK(std::equal(local_counts.begin(), local_counts.end(), counts.begin() + begin * 2));
    PHX_CHECK(std::equal(local_status.begin(), local_status.end(), status.begin() + begin));
  }
  std::vector<uint8_t> scales(n * W::kBytes), hashes(n * 64);
  std::vector<int32_t> verdicts(n * 2), verdict_status(n);
  for (int64_t i = 0; i < n; ++i) W::from_int64(1).store(scales.data() + i * W::kBytes);
  auto verdict_call = [&](int workers) {
    return phx_mc_symmetric_polynomial_verdicts(
        n, 2, PHX_MC_BERNSTEIN_BOX, offsets.data(), exponents.data(), input.data(),
        scales.data(), hashes.data(), verdicts.data(), verdict_status.data(), 4, workers);
  };
  PHX_CHECK(verdict_call(1) == PHX_MC_OK);
  for (int64_t i = 0; i < n; ++i) {
    PHX_CHECK(verdict_status[i] == PHX_MC_OK);
    PHX_CHECK(verdicts[2 * i] == 1 && verdicts[2 * i + 1] == 1);
  }
  const auto serial_hashes = hashes;
  PHX_CHECK(verdict_call(4) == PHX_MC_OK);
  PHX_CHECK(hashes == serial_hashes);
  for (int64_t begin = 0; begin < n; begin += 13) {
    const int64_t size = std::min<int64_t>(13, n - begin);
    std::vector<int64_t> local_offsets(size * 4 + 1);
    std::vector<uint8_t> local_hashes(size * 64);
    std::vector<int32_t> local_verdicts(size * 2), local_status(size);
    for (int64_t i = 0; i <= size * 4; ++i) local_offsets[i] = i;
    PHX_CHECK(phx_mc_symmetric_polynomial_verdicts(
        size, 2, PHX_MC_BERNSTEIN_BOX, local_offsets.data(),
        exponents.data() + begin * 8, input.data() + begin * 4 * W::kBytes,
        scales.data() + begin * W::kBytes, local_hashes.data(), local_verdicts.data(),
        local_status.data(), 4, 4) == PHX_MC_OK);
    PHX_CHECK(std::equal(local_hashes.begin(), local_hashes.end(),
                         serial_hashes.begin() + begin * 64));
    PHX_CHECK(std::equal(local_verdicts.begin(), local_verdicts.end(), verdicts.begin() + 2 * begin));
    PHX_CHECK(std::equal(local_status.begin(), local_status.end(), verdict_status.begin() + begin));
  }
  W huge; huge.limb[3] = uint64_t{1} << 62;
  huge.store(input.data());
  PHX_CHECK(call(4) == PHX_MC_OK);
  PHX_CHECK(status[0] == PHX_MC_RANGE_ERROR);
  PHX_CHECK(status[1] == PHX_MC_OK);
  PHX_CHECK(verdict_call(4) == PHX_MC_OK);
  PHX_CHECK(verdict_status[0] == PHX_MC_RANGE_ERROR);
  PHX_CHECK(verdict_status[1] == PHX_MC_OK && verdicts[2] == 1 && verdicts[3] == 1);
  W::from_int64(1).store(input.data());
  W::from_int64(2).store(input.data() + W::kBytes);
  W::from_int64(2).store(input.data() + 2 * W::kBytes);
  PHX_CHECK(verdict_call(1) == PHX_MC_OK);
  PHX_CHECK(verdict_status[0] == PHX_MC_OK && verdicts[0] == 1 && verdicts[1] == 0);
  return 0;
}
