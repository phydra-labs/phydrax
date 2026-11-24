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
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "floating_point_mode.hpp"
#include "parallel.hpp"

namespace phx::mc {

inline std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

// Bit b of v moved to bit 3b (v below 2^21).
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

// Three-dimensional Hilbert index from the transpose, axis 0 first.
inline std::uint64_t interleave_transpose(const std::uint32_t* x) {
  return (spread_bits_3(x[0]) << 2) | (spread_bits_3(x[1]) << 1) | spread_bits_3(x[2]);
}

// In two dimensions, the lower-coordinate frame only swaps axes and
// complements both axes. Four successive transitions fit one lookup.
inline constexpr auto kPlanarHilbertDigits = [] {
  std::array<std::uint16_t, 1024> table{};
  for (unsigned orientation = 0; orientation < 4; ++orientation) {
    for (unsigned x = 0; x < 16; ++x) {
      for (unsigned y = 0; y < 16; ++y) {
        unsigned state = orientation;
        unsigned digits = 0;
        for (int bit = 3; bit >= 0; --bit) {
          const bool swapped = (state & 1U) != 0;
          const unsigned complement = state >> 1U;
          const unsigned a = (((swapped ? y : x) >> bit) & 1U) ^ complement;
          const unsigned b = (((swapped ? x : y) >> bit) & 1U) ^ complement;
          digits = (digits << 2U) | ((3U * a) ^ b);
          if (b == 0) state ^= 1U | (a << 1U);
        }
        table[(orientation << 8U) | (x << 4U) | y] =
            static_cast<std::uint16_t>((digits << 2U) | state);
      }
    }
  }
  return table;
}();

inline std::uint64_t planar_hilbert_key(std::uint32_t x, std::uint32_t y) {
  std::uint64_t key = 0;
  unsigned state = 0;
  for (int bit = 27; bit >= 3; bit -= 4) {
    const unsigned index = (state << 8U) | (((x >> bit) & 15U) << 4U) |
                           ((y >> bit) & 15U);
    const unsigned entry = kPlanarHilbertDigits[index];
    key = (key << 8U) | (entry >> 2U);
    state = entry & 3U;
  }
  // Pad the last three coordinate bits on the right and discard the added
  // digit. Higher digits never depend on any lower coordinate bits.
  const unsigned index = (state << 8U) | ((x & 7U) << 5U) | ((y & 7U) << 1U);
  return (key << 6U) | (kPlanarHilbertDigits[index] >> 4U);
}

// One point of a spatial sort: Hilbert key, point index, BRIO round.
struct SpatialKey {
  std::uint64_t key;
  int32_t index;
  std::uint32_t round;
};

// Keys of the candidates in [begin, end), at most Lanes of them, quantized
// to Bits bits per axis in the box (lower, extent), into keys[begin, end).
template <int Dimension, int Bits, int Lanes>
void hilbert_key_block(const double* points, std::span<const int32_t> candidates,
                       std::size_t begin, std::size_t end, const double* lower,
                       const double* extent, SpatialKey* keys) {
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
  if constexpr (Dimension == 2) {
    static_assert(Bits == 31);
    for (std::size_t at = begin; at < end; ++at) {
      keys[at] = {planar_hilbert_key(x[0][at - begin], x[1][at - begin]),
                  candidates[at], 0U};
    }
  } else {
    hilbert_transpose<Dimension, Bits, Lanes>(x);
    for (std::size_t at = begin; at < end; ++at) {
      const std::uint32_t transposed[3] = {x[0][at - begin], x[1][at - begin],
                                           x[2][at - begin]};
      keys[at] = {interleave_transpose(transposed), candidates[at], 0U};
    }
  }
}

// Hilbert keys (31 bits per axis in 2D, 21 in 3D) of `candidates` in their
// bounding box, in candidate order; rounds are left 0.
inline int32_t hilbert_keys(const double* points, int dimension,
                            std::span<const int32_t> candidates, std::vector<SpatialKey>& keys,
                            std::uint32_t parallelism = 1) {
  keys.clear();
  if (candidates.empty()) return PHX_MC_OK;
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
  // Lanes are independent dependency chains through the transpose: enough
  // of them keep the vector units busy.
  constexpr std::size_t kLanes = 32;
  keys.resize(candidates.size());
  const auto blocks = (candidates.size() + kLanes - 1) / kLanes;
  const auto fill = [&](std::size_t first, std::size_t last) {
    for (std::size_t block = first; block < last; ++block) {
      const std::size_t begin = block * kLanes;
      const auto end = std::min(candidates.size(), begin + kLanes);
      if (dimension == 2) {
        hilbert_key_block<2, 31, kLanes>(points, candidates, begin, end, lower, extent, keys.data());
      } else {
        hilbert_key_block<3, 21, kLanes>(points, candidates, begin, end, lower, extent, keys.data());
      }
    }
  };
  if (parallelism == 1 || candidates.size() < 65536) {
    fill(0, blocks);
    return PHX_MC_OK;
  }
  return parallel_for(blocks, ParallelRequest{128, parallelism},
                      [&](std::size_t first, std::size_t last, ChunkContext&) -> int32_t {
    const GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    fill(first, last);
    return PHX_MC_OK;
  });
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
  // The number of trailing zero bits, at most rounds - 1.
  const int level = std::min(std::countr_zero(hash), rounds - 1);
  return static_cast<std::uint32_t>(rounds - 1 - level);
}

// Stable sort by the key (`by_key`) and then by the round (`by_round`, most
// significant). The key is radix sorted (least significant digit first, 11
// bits per digit) by its top bits only, enough to separate about 2^12 times
// as many keys as there are entries; runs that tie on those bits are then
// sorted stably by the whole key, which is the order of a full stable key
// sort. The round (below 62) is one more radix pass. Digits constant over
// all entries are skipped. `scratch` is working storage.
// Few entries are sorted by std::stable_sort on the same fields instead (the
// same stable order): the radix passes would clear and scan a 2048-bucket
// histogram per digit.
inline void radix_sort_keys(std::vector<SpatialKey>& keys, std::vector<SpatialKey>& scratch,
                            bool by_key, bool by_round) {
  constexpr int kDigitBits = 11;
  constexpr std::size_t kBuckets = std::size_t{1} << kDigitBits;
  constexpr std::size_t kRoundBuckets = 64;
  const std::size_t count = keys.size();
  if (count < 2) return;
  if (count <= 256) {
    std::stable_sort(keys.begin(), keys.end(),
                     [by_key, by_round](const SpatialKey& left, const SpatialKey& right) {
                       if (by_round && left.round != right.round) {
                         return left.round < right.round;
                       }
                       return by_key && left.key < right.key;
                     });
    return;
  }
  std::uint64_t used = 0;
  if (by_key) {
    for (const SpatialKey& entry : keys) used |= entry.key;
  }
  const int width = std::bit_width(used);
  const int separated = std::min(width, std::bit_width(count) + 12);
  const int digits = (separated + kDigitBits - 1) / kDigitBits;
  const int base = std::max(0, width - kDigitBits * digits);
  const auto key_digit = [base](const SpatialKey& entry, int d) -> std::size_t {
    return static_cast<std::size_t>(entry.key >> (base + kDigitBits * d)) & (kBuckets - 1);
  };
  const auto key_rows = static_cast<std::size_t>(digits);
  std::vector<std::size_t> histogram(key_rows * kBuckets + (by_round ? kRoundBuckets : 0), 0);
  std::size_t* round_offsets = histogram.data() + key_rows * kBuckets;
  for (const SpatialKey& entry : keys) {
    for (int d = 0; d < digits; ++d) {
      ++histogram[static_cast<std::size_t>(d) * kBuckets + key_digit(entry, d)];
    }
    if (by_round) ++round_offsets[entry.round];
  }
  scratch.resize(count);
  const auto pass = [&](std::size_t* offsets, std::size_t buckets, auto digit) {
    if (offsets[digit(keys.front())] == count) return;
    std::size_t total = 0;
    for (std::size_t bucket = 0; bucket < buckets; ++bucket) {
      const std::size_t size = offsets[bucket];
      offsets[bucket] = total;
      total += size;
    }
    for (const SpatialKey& entry : keys) scratch[offsets[digit(entry)]++] = entry;
    keys.swap(scratch);
  };
  for (int d = 0; d < digits; ++d) {
    pass(histogram.data() + static_cast<std::size_t>(d) * kBuckets, kBuckets,
         [&](const SpatialKey& entry) { return key_digit(entry, d); });
  }
  if (base > 0) {
    const auto by_whole_key = [](const SpatialKey& left, const SpatialKey& right) {
      return left.key < right.key;
    };
    std::size_t begin = 0;
    while (begin < count) {
      std::size_t end = begin + 1;
      while (end < count && (keys[end].key >> base) == (keys[begin].key >> base)) ++end;
      if (end - begin > 1) {
        std::stable_sort(keys.begin() + static_cast<std::ptrdiff_t>(begin),
                         keys.begin() + static_cast<std::ptrdiff_t>(end), by_whole_key);
      }
      begin = end;
    }
  }
  if (by_round) {
    pass(round_offsets, kRoundBuckets, [](const SpatialKey& entry) {
      return static_cast<std::size_t>(entry.round);
    });
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
