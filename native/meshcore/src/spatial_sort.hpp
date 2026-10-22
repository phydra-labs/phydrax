//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Deterministic biased randomized insertion order (BRIO, Amenta-Choi-Rote)
// with Hilbert-curve ordering inside each round.  Round membership uses a
// fixed-seed splitmix64 hash of the point index, so the order depends only on
// the input.  Ties in the Hilbert key are ordered by point index.
#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

#include "mesh.hpp"

namespace phx::mc {

inline std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

// Places bit i of a 32-bit value at bit 2i (five word-parallel stages).
template <class Charge>
inline std::uint64_t spread_bits_2(std::uint64_t value, const Charge& charge) {
  constexpr std::uint64_t kMasks[5] = {0x0000FFFF0000FFFFULL, 0x00FF00FF00FF00FFULL,
                                       0x0F0F0F0F0F0F0F0FULL, 0x3333333333333333ULL,
                                       0x5555555555555555ULL};
  constexpr int kShifts[5] = {16, 8, 4, 2, 1};
  value &= 0xFFFFFFFFULL;
  for (int stage = 0; stage < 5; ++stage) {
    charge();
    value = (value | (value << kShifts[stage])) & kMasks[stage];
  }
  return value;
}

// Places bit i of a 21-bit value at bit 3i (five word-parallel stages).
template <class Charge>
inline std::uint64_t spread_bits_3(std::uint64_t value, const Charge& charge) {
  constexpr std::uint64_t kMasks[5] = {0x001F00000000FFFFULL, 0x001F0000FF0000FFULL,
                                       0x100F00F00F00F00FULL, 0x10C30C30C30C30C3ULL,
                                       0x1249249249249249ULL};
  constexpr int kShifts[5] = {32, 16, 8, 4, 2};
  value &= 0x1FFFFFULL;
  for (int stage = 0; stage < 5; ++stage) {
    charge();
    value = (value | (value << kShifts[stage])) & kMasks[stage];
  }
  return value;
}

// Hilbert index of 2D integer coordinates with `bits` <= 32 bits per axis by
// a word-parallel prefix scan over the curve's orientation states (the
// loop-free formulation of the transpose algorithm below): one charge per
// executed word stage instead of one per bit.  Bit-identical to the
// transpose algorithm for every input.
template <class Charge>
inline std::uint64_t hilbert_key_2d(std::uint32_t x, std::uint32_t y, int bits,
                                    const Charge& charge) {
  constexpr std::uint32_t kAll = 0xFFFFFFFFU;
  charge();
  x <<= 32 - bits;
  y <<= 32 - bits;
  std::uint32_t a = x ^ y;
  std::uint32_t b = kAll ^ a;
  std::uint32_t c = kAll ^ (x | y);
  std::uint32_t d = x & (y ^ kAll);
  std::uint32_t big_a = a | (b >> 1);
  std::uint32_t big_b = (a >> 1) ^ a;
  std::uint32_t big_c = ((c >> 1) ^ (b & (d >> 1))) ^ c;
  std::uint32_t big_d = ((a & (c >> 1)) ^ (d >> 1)) ^ d;
  for (int shift = 2; shift <= 16; shift <<= 1) {
    charge();
    a = big_a;
    b = big_b;
    c = big_c;
    d = big_d;
    big_a = (a & (a >> shift)) ^ (b & (b >> shift));
    big_b = (a & (b >> shift)) ^ (b & ((a ^ b) >> shift));
    big_c ^= (a & (c >> shift)) ^ (b & (d >> shift));
    big_d ^= (b & (c >> shift)) ^ ((a ^ b) & (d >> shift));
  }
  charge();
  const std::uint32_t undo_c = big_c ^ (big_c >> 1);
  const std::uint32_t undo_d = big_d ^ (big_d >> 1);
  const std::uint32_t low = x ^ y;
  const std::uint32_t high = undo_d | (kAll ^ (low | undo_c));
  const std::uint64_t key = (spread_bits_2(high, charge) << 1) | spread_bits_2(low, charge);
  charge();
  return key >> (64 - 2 * bits);
}

// Hilbert index of integer coordinates with `bits` bits per axis (Skilling's
// transpose algorithm), for dimension 2 or 3.  Dimension 2 uses the
// equivalent word-parallel scan above; the transposed 3D representation is
// interleaved by word-parallel bit spreading.
template <class Charge = NativePreparationCharge>
inline std::uint64_t hilbert_key(std::uint32_t* x, int dimension, int bits,
                                 const Charge& charge = {}) {
  if (dimension == 2) {
    return hilbert_key_2d(x[0], x[1], bits, charge);
  }
  const std::uint32_t top = std::uint32_t{1} << (bits - 1);
  // Inverse undo excess work.
  for (std::uint32_t q = top; q > 1; q >>= 1) {
    const std::uint32_t p = q - 1;
    for (int i = 0; i < dimension; ++i) {
      charge();
      if (x[i] & q) {
        x[0] ^= p;
      } else {
        const std::uint32_t t = (x[0] ^ x[i]) & p;
        x[0] ^= t;
        x[i] ^= t;
      }
    }
  }
  // Gray encode.
  for (int i = 1; i < dimension; ++i) {
    charge();
    x[i] ^= x[i - 1];
  }
  std::uint32_t t = 0;
  for (std::uint32_t q = top; q > 1; q >>= 1) {
    charge();
    if (x[dimension - 1] & q) {
      t ^= q - 1;
    }
  }
  for (int i = 0; i < dimension; ++i) {
    charge();
    x[i] ^= t;
  }
  // Interleave the transposed representation, most significant bit first:
  // bit b of axis i lands at key bit dimension * b + (dimension - 1 - i).
  charge();
  return (spread_bits_3(x[0], charge) << 2) | (spread_bits_3(x[1], charge) << 1) |
         spread_bits_3(x[2], charge);
}

// Returns `candidates` reordered for incremental insertion.
template <class Charge = NativePreparationCharge>
inline NativeVector<int32_t> brio_hilbert_order(
    const double* points, int dimension, std::span<const int32_t> candidates,
    const Charge& charge = {}) {
  const std::size_t count = candidates.size();
  if (count == 0) {
    return {};
  }
  double lower[3] = {0.0, 0.0, 0.0};
  double upper[3] = {0.0, 0.0, 0.0};
  for (int axis = 0; axis < dimension; ++axis) {
    charge();
    lower[axis] = upper[axis] = points[static_cast<int64_t>(candidates[0]) * dimension + axis];
  }
  for (int32_t index : candidates) {
    charge();
    for (int axis = 0; axis < dimension; ++axis) {
      charge();
      const double value = points[static_cast<int64_t>(index) * dimension + axis];
      lower[axis] = std::min(lower[axis], value);
      upper[axis] = std::max(upper[axis], value);
    }
  }
  const int bits = dimension == 2 ? 31 : 21;
  const double cells = static_cast<double>((std::uint64_t{1} << bits) - 1);
  int rounds = 1;
  while ((std::size_t{1} << rounds) < count && rounds < 62) {
    charge();
    ++rounds;
  }
  struct Entry {
    int round;
    std::uint64_t key;
    int32_t index;
  };
  NativeVector<Entry> entries;
  entries.reserve(count);
  for (int32_t index : candidates) {
    charge();
    std::uint32_t quantized[3] = {0, 0, 0};
    for (int axis = 0; axis < dimension; ++axis) {
      charge();
      const double extent = upper[axis] - lower[axis];
      const double value = points[static_cast<int64_t>(index) * dimension + axis];
      const double scaled = extent > 0.0 ? (value - lower[axis]) / extent * cells : 0.0;
      quantized[axis] = static_cast<std::uint32_t>(std::clamp(scaled, 0.0, cells));
    }
    const std::uint64_t hash = splitmix64(static_cast<std::uint64_t>(index) ^ 0x5DEECE66DULL);
    int level = 0;
    while (level < rounds - 1 && ((hash >> level) & 1U) == 0U) {
      charge();
      ++level;
    }
    entries.push_back(
        {rounds - 1 - level, hilbert_key(quantized, dimension, bits, charge), index});
  }
  std::sort(entries.begin(), entries.end(), [&](const Entry& left, const Entry& right) {
    charge();
    if (left.round != right.round) {
      return left.round < right.round;
    }
    if (left.key != right.key) {
      return left.key < right.key;
    }
    return left.index < right.index;
  });
  NativeVector<int32_t> order;
  order.reserve(count);
  for (const Entry& entry : entries) {
    charge();
    order.push_back(entry.index);
  }
  return order;
}

}  // namespace phx::mc
