//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Deterministic biased randomized insertion order (BRIO, Amenta-Choi-Rote)
// with Hilbert-curve ordering inside each round.  Round membership uses a
// fixed-seed splitmix64 hash of the point index, so the order depends only on
// the input.  Ties in the Hilbert key keep the candidates' order (ascending
// point index for every caller).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace phx::mc {

inline std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

// Hint that `*address` is read soon: insertion walks and cavities prefetch
// the cells they are about to visit.
template <class T>
inline void prefetch_read([[maybe_unused]] const T* address) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(address);
#endif
}

// Bit b of v moved to bit 2b (v below 2^32) or 3b (v below 2^21).
inline std::uint64_t spread_bits_2(std::uint64_t v) {
  v = (v | (v << 16)) & 0x0000FFFF0000FFFFULL;
  v = (v | (v << 8)) & 0x00FF00FF00FF00FFULL;
  v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0FULL;
  v = (v | (v << 2)) & 0x3333333333333333ULL;
  return (v | (v << 1)) & 0x5555555555555555ULL;
}
inline std::uint64_t spread_bits_3(std::uint64_t v) {
  v &= 0x1FFFFFULL;
  v = (v | (v << 32)) & 0x001F00000000FFFFULL;
  v = (v | (v << 16)) & 0x001F0000FF0000FFULL;
  v = (v | (v << 8)) & 0x100F00F00F00F00FULL;
  v = (v | (v << 4)) & 0x10C30C30C30C30C3ULL;
  return (v | (v << 2)) & 0x1249249249249249ULL;
}

// Skilling's transpose algorithm on `Lanes` points at once (independent
// lanes, so the serial chain of each point overlaps the others'): x[i][lane]
// holds axis i of a point with `Bits` bits per axis; on return, its Hilbert
// transpose.
template <int Dimension, int Bits, int Lanes>
void hilbert_transpose(std::uint32_t (&x)[Dimension][Lanes]) {
  // Inverse undo excess work: for each bit s from the top, invert the low
  // bits of x[0] when bit s of x[i] is set, else exchange them with x[i].
  for (int s = Bits - 1; s > 0; --s) {
    const std::uint32_t low = (std::uint32_t{1} << s) - 1U;
    for (int lane = 0; lane < Lanes; ++lane) {
      x[0][lane] ^= low & (0U - ((x[0][lane] >> s) & 1U));
    }
    for (int i = 1; i < Dimension; ++i) {
      for (int lane = 0; lane < Lanes; ++lane) {
        const std::uint32_t set = 0U - ((x[i][lane] >> s) & 1U);
        const std::uint32_t exchange = (x[0][lane] ^ x[i][lane]) & low & ~set;
        x[0][lane] ^= (low & set) | exchange;
        x[i][lane] ^= exchange;
      }
    }
  }
  // Gray encode; t has bit j set iff the bits of x[Dimension - 1] above j
  // have odd parity (the suffix XOR shifted by one).
  for (int i = 1; i < Dimension; ++i) {
    for (int lane = 0; lane < Lanes; ++lane) x[i][lane] ^= x[i - 1][lane];
  }
  for (int lane = 0; lane < Lanes; ++lane) {
    std::uint32_t t = x[Dimension - 1][lane];
    t ^= t >> 1;
    t ^= t >> 2;
    t ^= t >> 4;
    t ^= t >> 8;
    t ^= t >> 16;
    t >>= 1;
    for (int i = 0; i < Dimension; ++i) x[i][lane] ^= t;
  }
}

// Hilbert index from the transpose, most significant bit first (axis 0
// carries the highest bit of every group).
inline std::uint64_t interleave_transpose(const std::uint32_t* x, int dimension) {
  if (dimension == 2) return (spread_bits_2(x[0]) << 1) | spread_bits_2(x[1]);
  return (spread_bits_3(x[0]) << 2) | (spread_bits_3(x[1]) << 1) | spread_bits_3(x[2]);
}

// One point of a spatial sort: Hilbert key, point index, BRIO round.
struct SpatialKey {
  std::uint64_t key;
  int32_t index;
  std::uint32_t round;
};

// Keys of the candidates in [begin, end), at most Lanes of them, quantized
// to Bits bits per axis in the box (lower, extent).
template <int Dimension, int Bits, int Lanes>
void hilbert_key_block(const double* points, std::span<const int32_t> candidates,
                       std::size_t begin, std::size_t end, const double* lower,
                       const double* extent, std::vector<SpatialKey>& keys) {
  constexpr double kCells = static_cast<double>((std::uint64_t{1} << Bits) - 1);
  std::uint32_t x[Dimension][Lanes] = {};
  for (std::size_t at = begin; at < end; ++at) {
    const int32_t index = candidates[at];
    for (int axis = 0; axis < Dimension; ++axis) {
      const double value = points[static_cast<int64_t>(index) * Dimension + axis];
      const double scaled =
          extent[axis] > 0.0 ? (value - lower[axis]) / extent[axis] * kCells : 0.0;
      x[axis][at - begin] = static_cast<std::uint32_t>(std::clamp(scaled, 0.0, kCells));
    }
  }
  hilbert_transpose<Dimension, Bits, Lanes>(x);
  for (std::size_t at = begin; at < end; ++at) {
    std::uint32_t transposed[3] = {0, 0, 0};
    for (int axis = 0; axis < Dimension; ++axis) transposed[axis] = x[axis][at - begin];
    keys.push_back({interleave_transpose(transposed, Dimension), candidates[at], 0U});
  }
}

// Hilbert keys (31 bits per axis in 2D, 21 in 3D) of `candidates` in their
// bounding box, in candidate order; rounds are left 0.
inline void hilbert_keys(const double* points, int dimension, std::span<const int32_t> candidates,
                         std::vector<SpatialKey>& keys) {
  keys.clear();
  if (candidates.empty()) return;
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
  const double extent[3] = {upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2]};
  constexpr std::size_t kLanes = 8;
  keys.reserve(candidates.size());
  for (std::size_t begin = 0; begin < candidates.size(); begin += kLanes) {
    const std::size_t end = std::min(candidates.size(), begin + kLanes);
    if (dimension == 2) {
      hilbert_key_block<2, 31, kLanes>(points, candidates, begin, end, lower, extent, keys);
    } else {
      hilbert_key_block<3, 21, kLanes>(points, candidates, begin, end, lower, extent, keys);
    }
  }
}

// BRIO rounds for `count` points and the round of point `index`: round r
// holds about 2^(r - rounds) of the points, the last round half of them.
inline int brio_round_count(std::size_t count) {
  int rounds = 1;
  while ((std::size_t{1} << rounds) < count && rounds < 62) ++rounds;
  return rounds;
}
inline std::uint32_t brio_round(int32_t index, int rounds) {
  const std::uint64_t hash = splitmix64(static_cast<std::uint64_t>(index) ^ 0x5DEECE66DULL);
  int level = 0;
  while (level < rounds - 1 && ((hash >> level) & 1U) == 0U) ++level;
  return static_cast<std::uint32_t>(rounds - 1 - level);
}

// Stable least-significant-digit radix sort by the key (`by_key`) and then
// by the round (`by_round`, most significant). Six 11-bit digits cover the
// key (at most 63 bits); rounds are below 62. Digits constant over all
// entries are skipped. `scratch` is working storage.
inline void radix_sort_keys(std::vector<SpatialKey>& keys, std::vector<SpatialKey>& scratch,
                            bool by_key, bool by_round) {
  constexpr int kDigitBits = 11;
  constexpr std::size_t kBuckets = std::size_t{1} << kDigitBits;
  constexpr int kKeyDigits = 6;
  const std::size_t count = keys.size();
  if (count < 2) return;
  const int first = by_key ? 0 : kKeyDigits;
  const int last = by_round ? kKeyDigits : kKeyDigits - 1;
  if (first > last) return;
  const auto digit = [](const SpatialKey& entry, int d) -> std::size_t {
    return d == kKeyDigits ? entry.round
                           : static_cast<std::size_t>(entry.key >> (kDigitBits * d)) &
                                 (kBuckets - 1);
  };
  std::vector<std::size_t> histogram(static_cast<std::size_t>(last - first + 1) * kBuckets, 0);
  for (const SpatialKey& entry : keys) {
    for (int d = first; d <= last; ++d) {
      ++histogram[static_cast<std::size_t>(d - first) * kBuckets + digit(entry, d)];
    }
  }
  scratch.resize(count);
  for (int d = first; d <= last; ++d) {
    std::size_t* offsets = histogram.data() + static_cast<std::size_t>(d - first) * kBuckets;
    if (offsets[digit(keys.front(), d)] == count) continue;
    std::size_t total = 0;
    for (std::size_t bucket = 0; bucket < kBuckets; ++bucket) {
      const std::size_t size = offsets[bucket];
      offsets[bucket] = total;
      total += size;
    }
    for (const SpatialKey& entry : keys) scratch[offsets[digit(entry, d)]++] = entry;
    keys.swap(scratch);
  }
}

// Returns `candidates` reordered for incremental insertion: by round, then
// Hilbert key, then candidate order.
inline std::vector<int32_t> brio_hilbert_order(const double* points, int dimension,
                                               const std::vector<int32_t>& candidates) {
  std::vector<SpatialKey> keys, scratch;
  hilbert_keys(points, dimension, candidates, keys);
  const int rounds = brio_round_count(keys.size());
  for (SpatialKey& entry : keys) entry.round = brio_round(entry.index, rounds);
  radix_sort_keys(keys, scratch, true, true);
  std::vector<int32_t> order;
  order.reserve(keys.size());
  for (const SpatialKey& entry : keys) order.push_back(entry.index);
  return order;
}

}  // namespace phx::mc
