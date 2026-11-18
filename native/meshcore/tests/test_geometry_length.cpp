//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// geometry::length is the square root of the left-to-right sum of squares.
// Exact cases, inputs on which another summation order or a correctly
// rounded hypot gives the neighboring double (so the stated order is the one
// computed), and the stated behavior outside the range of the squares.
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include "check.hpp"
#include "geometry3.hpp"

namespace {

using phx::mc::geometry::length;

std::uint64_t bits(double value) { return std::bit_cast<std::uint64_t>(value); }

void exact_lengths() {
  PHX_CHECK(length(3.0, 4.0) == 5.0);
  PHX_CHECK(length(-5.0, 12.0) == 13.0);
  PHX_CHECK(length(std::array<double, 3>{2.0, 3.0, 6.0}) == 7.0);
  PHX_CHECK(length(std::array<double, 3>{-1.0, 4.0, -8.0}) == 9.0);
  // Scaling by a power of two is exact through products, sums and the root.
  const double scaled[3] = {0x1p-40 * 2.0, 0x1p-40 * 3.0, 0x1p-40 * 6.0};
  PHX_CHECK(length(scaled) == 0x1p-40 * 7.0);
  PHX_CHECK(length(0.0, 0.0) == 0.0);
  PHX_CHECK(length(std::array<double, 3>{0.0, -0.0, 0.0}) == 0.0);
}

void stated_order() {
  // (x, y, z), the length, and its neighbor, which x^2 + (y^2 + z^2),
  // (x^2 + z^2) + y^2 and the correctly rounded length all give.
  const double triples[3][5] = {
      {-0x1.5272b0fdbf6f8p+0, 0x1.3f2f8fc65d974p+0, -0x1.d578fb07018ccp+0, 0x1.4a774c43bf407p+1,
       0x1.4a774c43bf408p+1},
      {-0x1.49d4b729c96a0p-2, 0x1.3a6700fb09a60p-3, 0x1.72cc90a579648p-1, 0x1.9d5c5b01b8e7ap-1,
       0x1.9d5c5b01b8e7bp-1},
      {0x1.887f3e3e90db6p+0, 0x1.a5591e75f2aacp+0, -0x1.1270d5b6daba0p-4, 0x1.200bd1a01d57ep+1,
       0x1.200bd1a01d57fp+1}};
  for (const auto& row : triples) {
    const double x = row[0], y = row[1], z = row[2];
    PHX_CHECK(bits(length(row)) == bits(row[3]));
    PHX_CHECK(bits(length(std::array<double, 3>{x, y, z})) == bits(row[3]));
    PHX_CHECK(bits(std::sqrt(x * x + (y * y + z * z))) == bits(row[4]));
    PHX_CHECK(bits(std::sqrt((x * x + z * z) + y * y)) == bits(row[4]));
  }
  // (x, y), the length, and the correctly rounded length beside it.
  const double pairs[2][4] = {
      {0x1.8144b08a76f34p-1, 0x1.0b49bfee71fbcp-1, 0x1.d4e86aacc051dp-1, 0x1.d4e86aacc051ep-1},
      {-0x1.dc3046682e3b4p+0, -0x1.30bfcac020a60p-2, 0x1.e23eac42447cap+0, 0x1.e23eac42447cbp+0}};
  for (const auto& row : pairs) {
    PHX_CHECK(bits(length(row[0], row[1])) == bits(row[2]));
    PHX_CHECK(std::nextafter(row[2], std::numeric_limits<double>::infinity()) == row[3]);
  }
}

void outside_the_range() {
  // Nothing is rescaled: squares beyond the range overflow or vanish.
  PHX_CHECK(length(1.0e200, 1.0e200) == std::numeric_limits<double>::infinity());
  PHX_CHECK(length(std::array<double, 3>{1.0e-200, 1.0e-200, 0.0}) == 0.0);
  PHX_CHECK(std::isnan(length(std::numeric_limits<double>::quiet_NaN(), 1.0)));
}

}  // namespace

int main() {
  exact_lengths();
  stated_order();
  outside_the_range();
  return phx::mc::test::finish("test_geometry_length");
}
