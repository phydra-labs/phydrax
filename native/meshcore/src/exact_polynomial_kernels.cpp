// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Checked integer leading principal minors of a symmetric polynomial matrix.
#include "phydrax_meshcore.h"
#include "capi_guard.hpp"
#include "parallel.hpp"
#include "wide_integer.hpp"
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

namespace {
using Key = std::array<int32_t, 3>;
template <size_t N> using Poly = std::map<Key, phx::mc::WideInt<N>>;
template <size_t N> bool add(Poly<N>& out, const Poly<N>& a, int sign) {
  for (const auto& [key, value] : a) {
    auto v = value;
    if (sign < 0) {
      v = phx::mc::wrapping_negate(v);
      if (v.negative() == value.negative() && !value.zero()) return false;
    }
    phx::mc::WideInt<N> sum;
    if (!phx::mc::checked_add(out[key], v, sum)) return false;
    if (sum.zero()) out.erase(key); else out[key] = sum;
  }
  return true;
}
template <size_t N> bool mul(const Poly<N>& a, const Poly<N>& b, Poly<N>& out) {
  for (const auto& [x, u] : a) for (const auto& [y, v] : b) {
    Key key{};
    for (int k = 0; k < 3; ++k) {
      if (x[k] > std::numeric_limits<int32_t>::max() - y[k]) return false;
      key[k] = x[k] + y[k];
    }
    phx::mc::WideInt<N> product, sum;
    if (!phx::mc::checked_multiply(u, v, product) ||
        !phx::mc::checked_add(out[key], product, sum)) return false;
    if (sum.zero()) out.erase(key); else out[key] = sum;
  }
  return true;
}
template <size_t N> bool product_add(Poly<N>& out, const Poly<N>& a, const Poly<N>& b, int sign) {
  Poly<N> p;
  return mul(a, b, p) && add(out, p, sign);
}
template <size_t N> int32_t item(int64_t row, int d, const int64_t* offsets,
    const int32_t* exponents, const uint8_t* coefficients, const int64_t* output_offsets,
    int32_t* out_exponents, uint8_t* out_coefficients, int64_t* counts) {
  using W = phx::mc::WideInt<N>;
  std::array<Poly<N>, 9> s;
  for (int i = 0; i < d; ++i) for (int j = 0; j < d; ++j) {
    const auto index = row * d * d + i * d + j;
    for (int64_t t = offsets[index]; t < offsets[index + 1]; ++t) {
      Key key{};
      for (int k = 0; k < d; ++k) key[k] = exponents[t * d + k];
      Poly<N> term;
      term.emplace(key, W::load(coefficients + t * W::kBytes));
      if (!add(s[i * d + j], term, 1) || !add(s[j * d + i], term, 1))
        return PHX_MC_RANGE_ERROR;
    }
  }
  std::array<Poly<N>, 3> minors;
  minors[0] = s[0];
  if (d >= 2 && (!product_add(minors[1], s[0], s[d+1], 1) ||
                 !product_add(minors[1], s[1], s[d], -1))) return PHX_MC_RANGE_ERROR;
  if (d == 3) {
    Poly<N> a, b, c;
    if (!product_add(a, s[4], s[8], 1) || !product_add(a, s[5], s[7], -1) ||
        !product_add(b, s[3], s[8], 1) || !product_add(b, s[5], s[6], -1) ||
        !product_add(c, s[3], s[7], 1) || !product_add(c, s[4], s[6], -1) ||
        !product_add(minors[2], s[0], a, 1) || !product_add(minors[2], s[1], b, -1) ||
        !product_add(minors[2], s[2], c, 1)) return PHX_MC_RANGE_ERROR;
  }
  for (int m = 0; m < d; ++m) {
    const int64_t slot = row * d + m;
    if (minors[m].size() > static_cast<size_t>(output_offsets[slot + 1] - output_offsets[slot]))
      return PHX_MC_RANGE_ERROR;
    counts[slot] = static_cast<int64_t>(minors[m].size());
    int64_t t = output_offsets[slot];
    for (const auto& [key, value] : minors[m]) {
      for (int k = 0; k < d; ++k) out_exponents[t * d + k] = key[k];
      value.store(out_coefficients + t * W::kBytes);
      ++t;
    }
  }
  return PHX_MC_OK;
}
template <size_t N> int32_t run(int64_t count, int d, const int64_t* offsets,
    const int32_t* exponents, const uint8_t* coefficients, const int64_t* output_offsets,
    int32_t* out_exponents, uint8_t* out_coefficients, int64_t* counts, int32_t* status,
    int32_t workers) {
  return phx::mc::parallel_for(static_cast<size_t>(count), {16, static_cast<uint32_t>(workers)},
    [&](size_t first, size_t last, phx::mc::ChunkContext&) {
      for (size_t i = first; i < last; ++i)
        status[i] = item<N>(static_cast<int64_t>(i), d, offsets, exponents, coefficients,
                            output_offsets, out_exponents, out_coefficients, counts);
    });
}
}
extern "C" PHX_MC_API int32_t phx_mc_symmetric_polynomial_minors(
    int64_t count, int32_t dimension, const int64_t* offsets, const int32_t* exponents,
    const uint8_t* coefficients, const int64_t* output_offsets, int32_t* out_exponents,
    uint8_t* out_coefficients, int64_t* counts, int32_t* status, int32_t limbs, int32_t workers) {
  return phx::mc::guarded([&]() -> int32_t {
    if (count < 0 || count > (INT64_MAX - 1) / 9 || dimension < 1 || dimension > 3 ||
        workers < 1 || (limbs != 4 && limbs != 8 && limbs != 16 && limbs != 32 && limbs != 64))
      return PHX_MC_INVALID_ARGUMENT;
    if (count == 0) return PHX_MC_OK;
    if (!phx::mc::addressable(count * dimension * dimension + 1, 1, sizeof(int64_t)) ||
        !phx::mc::addressable(count * dimension + 1, 1, sizeof(int64_t)) ||
        !offsets || !exponents || !coefficients || !output_offsets || !out_exponents ||
        !out_coefficients || !counts || !status || offsets[0] != 0 || output_offsets[0] != 0)
      return PHX_MC_INVALID_ARGUMENT;
    const int64_t entries = count * dimension * dimension, slots = count * dimension;
    for (int64_t i = 0; i < entries; ++i) if (offsets[i] < 0 || offsets[i] > offsets[i+1]) return PHX_MC_INVALID_ARGUMENT;
    for (int64_t i = 0; i < slots; ++i) if (output_offsets[i] < 0 || output_offsets[i] > output_offsets[i+1]) return PHX_MC_INVALID_ARGUMENT;
    if (!phx::mc::addressable(offsets[entries], dimension, sizeof(int32_t)) ||
        !phx::mc::addressable(output_offsets[slots], dimension, sizeof(int32_t)) ||
        !phx::mc::addressable(offsets[entries], 8 * limbs, 1) ||
        !phx::mc::addressable(output_offsets[slots], 8 * limbs, 1)) return PHX_MC_INVALID_ARGUMENT;
    for (int64_t i = 0; i < offsets[entries] * dimension; ++i) if (exponents[i] < 0) return PHX_MC_INVALID_ARGUMENT;
#define PHX_RUN(N) run<N>(count, dimension, offsets, exponents, coefficients, output_offsets, out_exponents, out_coefficients, counts, status, workers)
    switch (limbs) { case 4: return PHX_RUN(4); case 8: return PHX_RUN(8); case 16: return PHX_RUN(16); case 32: return PHX_RUN(32); default: return PHX_RUN(64); }
#undef PHX_RUN
  });
}
