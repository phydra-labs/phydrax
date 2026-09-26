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
#include <vector>

namespace phx::mc {

inline std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

// Hilbert index of integer coordinates with `bits` bits per axis (Skilling's
// transpose algorithm), for dimension 2 or 3.
inline std::uint64_t hilbert_key(std::uint32_t* x, int dimension, int bits) {
  const std::uint32_t top = std::uint32_t{1} << (bits - 1);
  // Inverse undo excess work.
  for (std::uint32_t q = top; q > 1; q >>= 1) {
    const std::uint32_t p = q - 1;
    for (int i = 0; i < dimension; ++i) {
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
    x[i] ^= x[i - 1];
  }
  std::uint32_t t = 0;
  for (std::uint32_t q = top; q > 1; q >>= 1) {
    if (x[dimension - 1] & q) {
      t ^= q - 1;
    }
  }
  for (int i = 0; i < dimension; ++i) {
    x[i] ^= t;
  }
  // Interleave the transposed representation, most significant bit first.
  std::uint64_t key = 0;
  for (int bit = bits - 1; bit >= 0; --bit) {
    for (int i = 0; i < dimension; ++i) {
      key = (key << 1) | ((x[i] >> bit) & 1U);
    }
  }
  return key;
}

// Returns `candidates` reordered for incremental insertion.
inline std::vector<int32_t> brio_hilbert_order(const double* points, int dimension,
                                               const std::vector<int32_t>& candidates) {
  const std::size_t count = candidates.size();
  if (count == 0) {
    return {};
  }
  double lower[3] = {0.0, 0.0, 0.0};
  double upper[3] = {0.0, 0.0, 0.0};
  for (int axis = 0; axis < dimension; ++axis) {
    lower[axis] = upper[axis] = points[static_cast<int64_t>(candidates[0]) * dimension + axis];
  }
  for (int32_t index : candidates) {
    for (int axis = 0; axis < dimension; ++axis) {
      const double value = points[static_cast<int64_t>(index) * dimension + axis];
      lower[axis] = std::min(lower[axis], value);
      upper[axis] = std::max(upper[axis], value);
    }
  }
  const int bits = dimension == 2 ? 31 : 21;
  const double cells = static_cast<double>((std::uint64_t{1} << bits) - 1);
  int rounds = 1;
  while ((std::size_t{1} << rounds) < count && rounds < 62) {
    ++rounds;
  }
  struct Entry {
    int round;
    std::uint64_t key;
    int32_t index;
  };
  std::vector<Entry> entries;
  entries.reserve(count);
  for (int32_t index : candidates) {
    std::uint32_t quantized[3] = {0, 0, 0};
    for (int axis = 0; axis < dimension; ++axis) {
      const double extent = upper[axis] - lower[axis];
      const double value = points[static_cast<int64_t>(index) * dimension + axis];
      const double scaled = extent > 0.0 ? (value - lower[axis]) / extent * cells : 0.0;
      quantized[axis] = static_cast<std::uint32_t>(std::clamp(scaled, 0.0, cells));
    }
    const std::uint64_t hash = splitmix64(static_cast<std::uint64_t>(index) ^ 0x5DEECE66DULL);
    int level = 0;
    while (level < rounds - 1 && ((hash >> level) & 1U) == 0U) {
      ++level;
    }
    entries.push_back({rounds - 1 - level, hilbert_key(quantized, dimension, bits), index});
  }
  std::sort(entries.begin(), entries.end(), [](const Entry& left, const Entry& right) {
    if (left.round != right.round) {
      return left.round < right.round;
    }
    if (left.key != right.key) {
      return left.key < right.key;
    }
    return left.index < right.index;
  });
  std::vector<int32_t> order;
  order.reserve(count);
  for (const Entry& entry : entries) {
    order.push_back(entry.index);
  }
  return order;
}

}  // namespace phx::mc
