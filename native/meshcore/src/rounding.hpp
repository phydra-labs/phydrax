// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Adjacent binary64 values directed toward the two infinities.
//
// The neighbor is taken on the bit pattern: the values are those of
// std::nextafter toward an infinity, without a library call, without a
// floating-point comparison (so independent of the denormal mode) and
// without raising an exception flag. A NaN is returned unchanged.
#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace phx::mc::geometry {

inline double next_up(double value) noexcept {
  constexpr std::uint64_t kSign = std::uint64_t{1} << 63;
  constexpr std::uint64_t kInfinity = std::uint64_t{0x7ff} << 52;
  const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
  const std::uint64_t magnitude = bits & ~kSign;
  if (magnitude > kInfinity || bits == kInfinity) return value;  // NaN, +infinity
  if (magnitude == 0) return std::bit_cast<double>(std::uint64_t{1});  // either zero
  return std::bit_cast<double>((bits & kSign) != 0 ? bits - 1 : bits + 1);
}

inline double next_down(double value) noexcept {
  constexpr std::uint64_t kSign = std::uint64_t{1} << 63;
  constexpr std::uint64_t kInfinity = std::uint64_t{0x7ff} << 52;
  const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
  const std::uint64_t magnitude = bits & ~kSign;
  if (magnitude > kInfinity || bits == (kSign | kInfinity)) return value;  // NaN, -infinity
  if (magnitude == 0) return std::bit_cast<double>(kSign | 1);  // either zero
  return std::bit_cast<double>((bits & kSign) != 0 ? bits + 1 : bits - 1);
}

}  // namespace phx::mc::geometry
