//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// The directed neighbors of rounding.hpp are the values of std::nextafter
// toward the infinities, bit for bit: on every class of binary64 value (both
// zeros, the smallest and largest subnormals and normals, the powers of two
// and their neighbors, the infinities) and on a fixed stream of bit patterns.
// A NaN is returned unchanged.
#include <bit>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include "check.hpp"
#include "rounding.hpp"

namespace {

using phx::mc::geometry::next_down;
using phx::mc::geometry::next_up;

std::uint64_t bits_of(double value) { return std::bit_cast<std::uint64_t>(value); }

void same_as_library(double value) {
  constexpr double infinity = std::numeric_limits<double>::infinity();
  PHX_CHECK(bits_of(next_up(value)) == bits_of(std::nextafter(value, infinity)));
  PHX_CHECK(bits_of(next_down(value)) == bits_of(std::nextafter(value, -infinity)));
}

void classes() {
  constexpr double infinity = std::numeric_limits<double>::infinity();
  constexpr double smallest = std::numeric_limits<double>::denorm_min();
  constexpr double normal = std::numeric_limits<double>::min();
  constexpr double largest = std::numeric_limits<double>::max();
  for (const double value : {0.0, smallest, normal, 1.0, 2.0, 0.1, largest, infinity}) {
    for (const double signed_value : {value, -value}) {
      same_as_library(signed_value);
      // Its two neighbors as well: the last subnormal, the first normal, the
      // values on both sides of a power of two.
      if (signed_value != infinity) same_as_library(std::nextafter(signed_value, infinity));
      if (signed_value != -infinity) same_as_library(std::nextafter(signed_value, -infinity));
    }
  }
  // The stated values at the ends of the range.
  PHX_CHECK(bits_of(next_up(0.0)) == 1 && bits_of(next_up(-0.0)) == 1);
  PHX_CHECK(bits_of(next_down(0.0)) == ((std::uint64_t{1} << 63) | 1));
  PHX_CHECK(bits_of(next_down(-0.0)) == ((std::uint64_t{1} << 63) | 1));
  PHX_CHECK(bits_of(next_up(-smallest)) == bits_of(-0.0));
  PHX_CHECK(bits_of(next_down(smallest)) == bits_of(0.0));
  PHX_CHECK(next_up(largest) == infinity && next_down(-largest) == -infinity);
  PHX_CHECK(next_up(infinity) == infinity && next_down(-infinity) == -infinity);
  PHX_CHECK(next_down(infinity) == largest && next_up(-infinity) == -largest);
  const double not_a_number = std::numeric_limits<double>::quiet_NaN();
  PHX_CHECK(bits_of(next_up(not_a_number)) == bits_of(not_a_number));
  PHX_CHECK(bits_of(next_down(not_a_number)) == bits_of(not_a_number));
}

void stream() {
  // A fixed 64-bit generator; patterns that are NaNs are skipped.
  std::uint64_t state = 0x9e3779b97f4a7c15ULL;
  for (int index = 0; index < 2000000; ++index) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    std::uint64_t pattern = state ^ (state >> 29);
    // Every fourth pattern is squeezed into the subnormal range.
    if (index % 4 == 0) pattern &= (std::uint64_t{1} << 63) | ((std::uint64_t{1} << 52) - 1);
    const double value = std::bit_cast<double>(pattern);
    if (value != value) continue;
    same_as_library(value);
  }
}

}  // namespace

int main() {
  classes();
  stream();
  return phx::mc::test::finish("test_rounding");
}
