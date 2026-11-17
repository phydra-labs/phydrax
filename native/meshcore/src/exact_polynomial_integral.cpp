// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Checked ordinary polynomial determinants and exact Bernstein integration.
#include "phydrax_meshcore.h"
#include "capi_guard.hpp"
#include "parallel.hpp"
#include "wide_integer.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

namespace {
using Key = std::array<int32_t, 3>;
template <size_t N> using Word = phx::mc::WideInt<N>;
template <size_t N> using Poly = std::map<Key, Word<N>>;

template <size_t N> bool accumulate(Poly<N>& out, const Key& key, Word<N> value, int sign) {
  if (sign < 0 && !value.zero()) {
    const bool was_negative = value.negative();
    value = phx::mc::wrapping_negate(value);
    if (value.negative() == was_negative) return false;
  }
  Word<N> sum;
  if (!phx::mc::checked_add(out[key], value, sum)) return false;
  if (sum.zero()) {
    out.erase(key);
  } else {
    out[key] = sum;
  }
  return true;
}

template <size_t N> bool multiply(const Poly<N>& a, const Poly<N>& b, Poly<N>& out) {
  for (const auto& [x, u] : a) {
    for (const auto& [y, v] : b) {
      Key key{};
      for (int k = 0; k < 3; ++k) {
        if (x[k] > INT32_MAX - y[k]) return false;
        key[k] = x[k] + y[k];
      }
      Word<N> product;
      if (!phx::mc::checked_multiply(u, v, product) ||
          !accumulate(out, key, product, 1)) {
        return false;
      }
    }
  }
  return true;
}

template <size_t N>
bool determinant(const std::array<Poly<N>, 9>& entries, int d, Poly<N>& result) {
  std::array<int, 3> permutation{0, 1, 2};
  do {
    Poly<N> product = entries[permutation[0]];
    for (int row = 1; row < d; ++row) {
      Poly<N> next;
      if (!multiply(product, entries[row * d + permutation[row]], next)) return false;
      product = std::move(next);
    }
    int inversions = 0;
    for (int i = 0; i < d; ++i) {
      for (int j = i + 1; j < d; ++j) {
        inversions += permutation[i] > permutation[j];
      }
    }
    for (const auto& [key, value] : product) {
      if (!accumulate(result, key, value, inversions % 2 ? -1 : 1)) return false;
    }
  } while (std::next_permutation(permutation.begin(), permutation.begin() + d));
  return true;
}

bool shape(int d, int domain, const std::vector<int32_t>& exponents,
           std::array<int32_t, 3>& degree, int64_t& count) {
  for (size_t t = 0; t < exponents.size(); t += d) {
    if (domain == PHX_MC_BERNSTEIN_BOX) {
      for (int k = 0; k < d; ++k) {
        degree[k] = std::max(degree[k], exponents[t + k]);
      }
    } else {
      const int parts = domain == PHX_MC_BERNSTEIN_SIMPLEX ? d : 2;
      int64_t total = 0;
      for (int k = 0; k < parts; ++k) {
        total += exponents[t + k];
      }
      if (total > INT32_MAX) return false;
      degree[0] = std::max(degree[0], static_cast<int32_t>(total));
      if (domain == PHX_MC_BERNSTEIN_PRISM) {
        degree[1] = std::max(degree[1], exponents[t + 2]);
      }
    }
  }
  count = 1;
  if (domain == PHX_MC_BERNSTEIN_BOX) {
    for (int k = 0; k < d; ++k) {
      int64_t factor = int64_t(degree[k]) + 1;
      if (factor > 100000 / count) return false;
      count *= factor;
    }
  } else {
    const int parts = domain == PHX_MC_BERNSTEIN_SIMPLEX ? d : 2;
    for (int k = 1; k <= parts; ++k) {
      const int64_t factor = int64_t(degree[0]) + k;
      if (count > 100000 * k / factor) return false;
      count = count * factor / k;
    }
    if (domain == PHX_MC_BERNSTEIN_PRISM) {
      int64_t factor = int64_t(degree[1]) + 1;
      if (factor > 100000 / count) return false;
      count *= factor;
    }
  }
  return true;
}

template <size_t N>
int32_t item(int64_t row, int d, int domain, const int64_t* offsets,
             const int32_t* exponents, const uint8_t* coefficients, uint8_t* sums,
             int64_t* denominators, int64_t* counts) {
  std::array<Poly<N>, 9> entries;
  for (int e = 0; e < d * d; ++e) {
    int64_t slot = row * d * d + e;
    for (int64_t t = offsets[slot]; t < offsets[slot + 1]; ++t) {
      Key key{};
      for (int k = 0; k < d; ++k) {
        key[k] = exponents[t * d + k];
      }
      if (!accumulate(entries[e], key, Word<N>::load(coefficients + t * Word<N>::kBytes), 1)) {
        return PHX_MC_RANGE_ERROR;
      }
    }
  }
  Poly<N> det;
  if (!determinant(entries, d, det)) return PHX_MC_RANGE_ERROR;
  std::vector<int32_t> powers;
  powers.reserve(det.size() * static_cast<size_t>(d));
  std::vector<uint8_t> words(det.size() * Word<N>::kBytes);
  size_t index = 0;
  for (const auto& [key, value] : det) {
    for (int k = 0; k < d; ++k) {
      powers.push_back(key[k]);
    }
    value.store(words.data() + index++ * Word<N>::kBytes);
  }
  std::array<int32_t, 3> degree{};
  int64_t count;
  if (!shape(d, domain, powers, degree, count)) return PHX_MC_RANGE_ERROR;
  const int64_t term_offsets[2] = {0, static_cast<int64_t>(det.size())};
  const int64_t control_offsets[2] = {0, count};
  std::vector<uint8_t> controls(count * Word<N>::kBytes);
  int64_t denominator = 0, extremes[2]{};
  int32_t status = PHX_MC_OK;
  const int32_t empty_power = 0;
  const uint8_t empty_word = 0;
  const int32_t call = phx_mc_bernstein_controls(
      1, domain, d, degree.data(), term_offsets,
      powers.empty() ? &empty_power : powers.data(), words.empty() ? &empty_word : words.data(),
      control_offsets, controls.data(), &denominator, extremes, &status, static_cast<int32_t>(N),
      1);
  if (call != PHX_MC_OK) return call;
  if (status != PHX_MC_OK) return status;
  Word<N> sum;
  for (int64_t i = 0; i < count; ++i) {
    Word<N> next;
    if (!phx::mc::checked_add(sum, Word<N>::load(controls.data() + i * Word<N>::kBytes),
                              next)) {
      return PHX_MC_RANGE_ERROR;
    }
    sum = next;
  }
  sum.store(sums + row * Word<N>::kBytes);
  denominators[row] = denominator;
  counts[row] = count;
  return PHX_MC_OK;
}

template <size_t N>
int32_t run(int64_t count, int d, int domain, const int64_t* offsets,
            const int32_t* exponents, const uint8_t* coefficients, uint8_t* sums,
            int64_t* denominators, int64_t* counts, int32_t* statuses, int32_t workers) {
  return phx::mc::parallel_for(
      static_cast<size_t>(count), {16, static_cast<uint32_t>(workers)},
      [&](size_t first, size_t last, phx::mc::ChunkContext&) {
        for (size_t i = first; i < last; ++i) {
          statuses[i] = item<N>(static_cast<int64_t>(i), d, domain, offsets, exponents,
                                coefficients, sums, denominators, counts);
        }
      });
}
}  // namespace

extern "C" PHX_MC_API int32_t phx_mc_polynomial_jacobian_integrals(
    int64_t count, int32_t dimension, int32_t domain, const int64_t* offsets,
    const int32_t* exponents, const uint8_t* coefficients, uint8_t* sums,
    int64_t* denominators, int64_t* counts, int32_t* statuses, int32_t limbs, int32_t workers) {
  return phx::mc::guarded([&]() -> int32_t {
    if (count < 0 || count > (INT64_MAX - 1) / 9 || dimension < 1 || dimension > 3 ||
        domain < PHX_MC_BERNSTEIN_SIMPLEX || domain > PHX_MC_BERNSTEIN_PRISM ||
        (domain == PHX_MC_BERNSTEIN_PRISM && dimension != 3) || workers < 1 ||
        (limbs != 4 && limbs != 8 && limbs != 16 && limbs != 32 && limbs != 64)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (count == 0) return PHX_MC_OK;
    const int64_t entries = count * dimension * dimension;
    if (!phx::mc::addressable(entries + 1, 1, sizeof(int64_t)) ||
        !phx::mc::addressable(count, 8 * limbs, 1) || !offsets || !exponents ||
        !coefficients || !sums || !denominators || !counts || !statuses || offsets[0] != 0) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    for (int64_t i = 0; i < entries; ++i) {
      if (offsets[i] < 0 || offsets[i] > offsets[i + 1]) return PHX_MC_INVALID_ARGUMENT;
    }
    if (!phx::mc::addressable(offsets[entries], dimension, sizeof(int32_t)) ||
        !phx::mc::addressable(offsets[entries], 8 * limbs, 1)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    for (int64_t i = 0; i < offsets[entries] * dimension; ++i) {
      if (exponents[i] < 0) return PHX_MC_INVALID_ARGUMENT;
    }
#define PHX_RUN(N) \
    run<N>(count, dimension, domain, offsets, exponents, coefficients, sums, denominators, \
           counts, statuses, workers)
    switch (limbs) {
      case 4: return PHX_RUN(4);
      case 8: return PHX_RUN(8);
      case 16: return PHX_RUN(16);
      case 32: return PHX_RUN(32);
      default: return PHX_RUN(64);
    }
#undef PHX_RUN
  });
}
