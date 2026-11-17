//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Recorded bit vectors of deterministic binary64 kernels: rows of 64-bit
// fields (a value is its IEEE 754 bit pattern; counts and flags are
// integers), each under the name of its kind. A test produces its rows from
// inputs it generates in integers, and either prints them (`--emit`) or
// requires them to equal the rows of its `.inc` file, where a row is written
// `NAME(field, ...)` and read through one macro per kind.
#pragma once

#include <bit>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

#include "check.hpp"

namespace phx::mc::test {

struct BitRow {
  const char* kind;
  std::vector<std::uint64_t> fields;
};

inline std::uint64_t bits_of(double value) { return std::bit_cast<std::uint64_t>(value); }
inline double value_of(std::uint64_t bits) { return std::bit_cast<double>(bits); }

inline constexpr std::uint64_t kNaNBits = 0x7ff8000000000000ULL;
inline constexpr std::uint64_t kInfinityBits = 0x7ff0000000000000ULL;

// Knuth's 64-bit linear congruential generator: integer arithmetic only.
struct IntegerGenerator {
  std::uint64_t state;
  std::uint64_t next() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state >> 33;
  }
  // An integer in [low, high].
  std::int64_t integer(std::int64_t low, std::int64_t high) {
    return low + static_cast<std::int64_t>(next() % static_cast<std::uint64_t>(high - low + 1));
  }
  // A binary64 value with the given unbiased exponent, a pseudo-random
  // significand and, when `sign` is set, a pseudo-random sign.
  double with_exponent(int exponent, bool sign) {
    const std::uint64_t high = next();
    const std::uint64_t low = next();
    std::uint64_t bits = (((high << 31) ^ low) & 0x000fffffffffffffULL) |
                         (static_cast<std::uint64_t>(exponent + 1023) << 52);
    if (sign && (next() & 1U) != 0) bits |= 0x8000000000000000ULL;
    return value_of(bits);
  }
};

// 2^exponent for exponent in [-1022, 1023], exactly.
inline double power_of_two(int exponent) {
  return value_of(static_cast<std::uint64_t>(exponent + 1023) << 52);
}

inline void append_bits(BitRow& row, const double* values, std::size_t count) {
  for (std::size_t index = 0; index < count; ++index) {
    row.fields.push_back(bits_of(values[index]));
  }
}

inline void emit_rows(const std::vector<BitRow>& rows) {
  for (const BitRow& row : rows) {
    std::printf("%s(", row.kind);
    for (std::size_t field = 0; field < row.fields.size(); ++field) {
      std::printf("%s0x%016" PRIx64, field == 0 ? "" : ", ", row.fields[field]);
    }
    std::printf(")\n");
  }
}

// Requires the produced rows to be the recorded ones, in order, and prints
// the differing fields of the first rows that are not.
inline void compare_rows(const std::vector<BitRow>& produced,
                         const std::vector<BitRow>& recorded) {
  PHX_CHECK(produced.size() == recorded.size());
  int reported = 0;
  for (std::size_t row = 0; row < produced.size() && row < recorded.size(); ++row) {
    const BitRow& got = produced[row];
    const BitRow& expected = recorded[row];
    const bool same = std::string_view(got.kind) == std::string_view(expected.kind) &&
                      got.fields == expected.fields;
    PHX_CHECK(same);
    if (same || reported++ >= 8) continue;
    std::fprintf(stderr, "  row %zu %s (recorded %s):", row, got.kind, expected.kind);
    for (std::size_t field = 0; field < got.fields.size() && field < expected.fields.size();
         ++field) {
      if (got.fields[field] != expected.fields[field]) {
        std::fprintf(stderr, " field %zu 0x%016" PRIx64 " (recorded 0x%016" PRIx64 ")", field,
                     got.fields[field], expected.fields[field]);
      }
    }
    std::fprintf(stderr, "\n");
  }
}

}  // namespace phx::mc::test
