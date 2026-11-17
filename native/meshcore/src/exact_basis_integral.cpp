// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Assemble each Jacobian on a common exact integer coordinate scale.
#include "phydrax_meshcore.h"
#include "capi_guard.hpp"
#include "parallel.hpp"
#include "wide_integer.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace {
template <size_t N> using Word = phx::mc::WideInt<N>;

template <size_t N>
int32_t item(int64_t row, int d, int domain, int64_t nodes, int64_t bank_rows,
             const int64_t* routes, const int64_t* sig, const int16_t* exp,
             const int64_t* sparse_index, const int64_t* sparse_rows,
             const uint8_t* sparse_num, const uint8_t* sparse_den,
             const int64_t* offsets, const int32_t* powers, const uint8_t* basis,
             uint8_t* sums, uint8_t* scales, int64_t* denominators, int64_t* counts) {
  using W = Word<N>;
  struct Coordinate { W numerator; W denominator; int exponent; bool sparse; };
  std::vector<Coordinate> coords;
  coords.reserve(static_cast<size_t>(nodes) * d);
  int shift = 0;
  for (int64_t n = 0; n < nodes; ++n) {
    const int64_t r = routes[row * nodes + n];
    if (r < 0 || r >= bank_rows) return PHX_MC_RANGE_ERROR;
    int64_t s = sparse_rows[r];
    for (int a = 0; a < d; ++a) {
      const int64_t index = r * d + a;
      if (exp[index] == INT16_MIN) {
        if (s >= sparse_rows[r + 1] || sparse_index[s] != index) return PHX_MC_RANGE_ERROR;
        W numerator = W::load(sparse_num + s * W::kBytes);
        W denominator = W::load(sparse_den + s * W::kBytes);
        if (denominator.zero() || denominator.negative()) return PHX_MC_RANGE_ERROR;
        coords.push_back({numerator, denominator, 0, true});
        ++s;
      } else {
        coords.push_back({W::from_int64(sig[index]), W::from_int64(1), exp[index], false});
        if (sig[index] != 0) shift = std::max(shift, -static_cast<int>(exp[index]));
      }
    }
  }
  W scale = W::from_int64(1), next;
  if (!phx::mc::checked_shift_left(scale, shift, next)) return PHX_MC_RANGE_ERROR;
  scale = next;
  for (const auto& c : coords) if (c.sparse) {
    if (!phx::mc::checked_multiply(scale, c.denominator, next)) return PHX_MC_RANGE_ERROR;
    scale = next;
  }
  std::vector<W> scaled(coords.size());
  for (size_t i = 0; i < coords.size(); ++i) {
    W value = coords[i].numerator;
    const int power = coords[i].sparse ? shift : shift + coords[i].exponent;
    if (value.zero()) { scaled[i] = value; continue; }
    if (!phx::mc::checked_shift_left(value, power, next)) return PHX_MC_RANGE_ERROR;
    value = next;
    for (size_t j = 0; j < coords.size(); ++j) if (coords[j].sparse && i != j) {
      if (!phx::mc::checked_multiply(value, coords[j].denominator, next)) return PHX_MC_RANGE_ERROR;
      value = next;
    }
    scaled[i] = value;
  }
  std::vector<int64_t> jac_offsets(1, 0);
  std::vector<int32_t> jac_powers;
  std::vector<uint8_t> jac_coefficients;
  for (int a = 0; a < d; ++a) for (int axis = 0; axis < d; ++axis) {
    for (int64_t n = 0; n < nodes; ++n) {
      for (int64_t t = offsets[n]; t < offsets[n + 1]; ++t) {
        const int32_t degree = powers[t * d + axis];
        if (degree == 0) continue;
        W product, coefficient = W::load(basis + t * W::kBytes);
        if (!phx::mc::checked_multiply(coefficient, scaled[n * d + a], product) ||
            !phx::mc::checked_multiply(product, W::from_int64(degree), next)) return PHX_MC_RANGE_ERROR;
        for (int k = 0; k < d; ++k) jac_powers.push_back(powers[t * d + k] - (k == axis));
        const size_t pos = jac_coefficients.size();
        jac_coefficients.resize(pos + W::kBytes);
        next.store(jac_coefficients.data() + pos);
      }
    }
    jac_offsets.push_back(static_cast<int64_t>(jac_coefficients.size() / W::kBytes));
  }
  int32_t status = PHX_MC_OK;
  const int32_t empty_power = 0;
  const uint8_t empty_word = 0;
  const int32_t call = phx_mc_polynomial_jacobian_integrals(
      1, d, domain, jac_offsets.data(), jac_powers.empty() ? &empty_power : jac_powers.data(),
      jac_coefficients.empty() ? &empty_word : jac_coefficients.data(), sums + row * W::kBytes,
      denominators + row, counts + row, &status, static_cast<int32_t>(N), 1);
  if (call != PHX_MC_OK) return call;
  if (status == PHX_MC_OK) scale.store(scales + row * W::kBytes);
  return status;
}

template <size_t N>
int32_t run(int64_t count, int d, int domain, int64_t nodes, int64_t bank_rows,
            const int64_t* routes, const int64_t* sig, const int16_t* exp,
            const int64_t* sparse_index, const int64_t* sparse_rows,
            const uint8_t* sparse_num, const uint8_t* sparse_den,
            const int64_t* offsets, const int32_t* powers, const uint8_t* basis,
            uint8_t* sums, uint8_t* scales, int64_t* denominators, int64_t* counts,
            int32_t* statuses, int32_t workers) {
  return phx::mc::parallel_for(static_cast<size_t>(count), {16, static_cast<uint32_t>(workers)},
    [&](size_t first, size_t last, phx::mc::ChunkContext&) {
      for (size_t i = first; i < last; ++i)
        statuses[i] = item<N>(static_cast<int64_t>(i), d, domain, nodes, bank_rows, routes,
          sig, exp, sparse_index, sparse_rows, sparse_num, sparse_den, offsets, powers, basis,
          sums, scales, denominators, counts);
    });
}
}  // namespace

extern "C" PHX_MC_API int32_t phx_mc_basis_jacobian_integrals(
    int64_t count, int32_t d, int32_t domain, int64_t nodes, int64_t bank_rows,
    const int64_t* routes, const int64_t* sig, const int16_t* exp,
    const int64_t* sparse_index, const int64_t* sparse_rows,
    const uint8_t* sparse_num, const uint8_t* sparse_den,
    const int64_t* offsets, const int32_t* powers, const uint8_t* basis,
    uint8_t* sums, uint8_t* scales, int64_t* denominators, int64_t* counts,
    int32_t* statuses, int32_t limbs, int32_t workers) {
  return phx::mc::guarded([&]() -> int32_t {
    if (count < 0 || nodes < 0 || bank_rows < 0 || d < 1 || d > 3 || domain < 0 || domain > 2 ||
        (domain == PHX_MC_BERNSTEIN_PRISM && d != 3) || workers < 1 ||
        (limbs != 4 && limbs != 8 && limbs != 16 && limbs != 32 && limbs != 64) ||
        !phx::mc::addressable(count, nodes, sizeof(int64_t)) ||
        !phx::mc::addressable(bank_rows, d, sizeof(int64_t)) ||
        !phx::mc::addressable(count, 8 * limbs, 1)) return PHX_MC_INVALID_ARGUMENT;
    if (count == 0) return PHX_MC_OK;
    if (!routes || !sig || !exp || !sparse_index || !sparse_rows || !sparse_num || !sparse_den ||
        !offsets || !powers || !basis || !sums || !scales || !denominators || !counts || !statuses ||
        offsets[0] != 0 || sparse_rows[0] != 0) return PHX_MC_INVALID_ARGUMENT;
    for (int64_t n = 0; n < nodes; ++n)
      if (offsets[n] < 0 || offsets[n] > offsets[n + 1]) return PHX_MC_INVALID_ARGUMENT;
    for (int64_t r = 0; r < bank_rows; ++r)
      if (sparse_rows[r] < 0 || sparse_rows[r] > sparse_rows[r + 1]) return PHX_MC_INVALID_ARGUMENT;
    if (!phx::mc::addressable(offsets[nodes], d, sizeof(int32_t)) ||
        !phx::mc::addressable(offsets[nodes], 8 * limbs, 1) ||
        !phx::mc::addressable(sparse_rows[bank_rows], 8 * limbs, 1)) return PHX_MC_INVALID_ARGUMENT;
    for (int64_t i = 0; i < offsets[nodes] * d; ++i)
      if (powers[i] < 0 || powers[i] > 715827882) return PHX_MC_INVALID_ARGUMENT;
#define PHX_RUN(N) run<N>(count,d,domain,nodes,bank_rows,routes,sig,exp,sparse_index,sparse_rows,sparse_num,sparse_den,offsets,powers,basis,sums,scales,denominators,counts,statuses,workers)
    switch (limbs) { case 4: return PHX_RUN(4); case 8: return PHX_RUN(8); case 16: return PHX_RUN(16); case 32: return PHX_RUN(32); default: return PHX_RUN(64); }
#undef PHX_RUN
  });
}
