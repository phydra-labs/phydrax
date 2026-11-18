//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// The portable angles of angle.hpp: stated values, the quadrants of
// `between`, the distance of every kernel from the interval kernel's
// enclosure of the true value on a generated stream, the width of the sine
// enclosure, agreement with the platform functions to a few values, and bit
// vectors that hold every platform to the same doubles. `test_angle --emit`
// prints the vector table from the generator below.
#include <algorithm>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>

#include "angle.hpp"
#include "check.hpp"
#include "floating_point_mode.hpp"
#include "rounding.hpp"

namespace {

namespace angle = phx::mc::angle;
namespace elementary = phx::mc::elementary;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInfinity = std::numeric_limits<double>::infinity();

std::uint64_t bits(double value) { return std::bit_cast<std::uint64_t>(value); }

// Position of a finite value among the binary64 values in increasing order.
std::int64_t ordinal(double value) {
  const std::uint64_t pattern = bits(value);
  const std::int64_t magnitude = static_cast<std::int64_t>(pattern & ~(std::uint64_t{1} << 63));
  return (pattern >> 63) != 0 ? -magnitude : magnitude;
}
std::int64_t steps(const elementary::Enclosure& enclosure) {
  return ordinal(enclosure.upper) - ordinal(enclosure.lower);
}
bool within(double value, double reference, std::int64_t count) {
  const std::int64_t distance = ordinal(value) - ordinal(reference);
  return distance >= -count && distance <= count;
}

struct Generator {
  std::uint64_t state;
  // Uniform in [0, 1) with 53 bits.
  double unit() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state >> 11) * 0x1p-53;
  }
};

void stated_values() {
  PHX_CHECK(bits(angle::sine(0.0)) == bits(0.0));
  PHX_CHECK(bits(angle::arcsine(0.0)) == bits(0.0));
  PHX_CHECK(bits(angle::arcsine(-0.0)) == bits(-0.0));
  PHX_CHECK(bits(angle::arccosine(1.0)) == bits(0.0));
  // asin(1/2) = pi/6 and acos(1/2) = pi/3: the nearest doubles.
  PHX_CHECK(angle::arcsine(0.5) == 0x1.0c152382d7366p-1);
  PHX_CHECK(angle::arcsine(-0.5) == -0x1.0c152382d7366p-1);
  PHX_CHECK(angle::arccosine(0.5) == 0x1.0c152382d7366p+0);
  PHX_CHECK(angle::arcsine(1.0) == angle::kHalfPi);
  PHX_CHECK(angle::arccosine(0.0) == angle::kHalfPi);
  PHX_CHECK(angle::arccosine(-1.0) == angle::kPi);
  // Outside the domain, as the platform functions.
  for (const double outside : {1.0000000000000002, -1.0000000000000002, 2.0, kInfinity, kNaN}) {
    PHX_CHECK(std::isnan(angle::arcsine(outside)));
    PHX_CHECK(std::isnan(angle::arccosine(outside)));
  }
  PHX_CHECK(std::isnan(angle::sine(kInfinity)) && std::isnan(angle::sine(kNaN)));
}

void quadrants() {
  PHX_CHECK(bits(angle::between(0.0, 0.0)) == bits(0.0));
  PHX_CHECK(bits(angle::between(0.0, 3.0)) == bits(0.0));
  PHX_CHECK(angle::between(2.0, 0.0) == angle::kHalfPi);
  PHX_CHECK(angle::between(2.0, -0.0) == angle::kHalfPi);
  // pi / 4, 3 pi / 4 and pi: the nearest doubles.
  PHX_CHECK(angle::between(1.0, 1.0) == 0x1.921fb54442d18p-1);
  PHX_CHECK(angle::between(1.0, -1.0) == 0x1.2d97c7f3321d2p+1);
  PHX_CHECK(angle::between(0.0, -1.0) == angle::kPi);
  // A direction beyond the range of the quotient is a right angle.
  PHX_CHECK(angle::between(1.0e300, 1.0e-300) == angle::kHalfPi);
  // From the obtuse side pi - pi/2 is formed from rounded parts: one value off.
  PHX_CHECK(within(angle::between(1.0e300, -1.0e-300), angle::kHalfPi, 1));
  for (const double invalid : {-1.0, kNaN, kInfinity}) {
    PHX_CHECK(std::isnan(angle::between(invalid, 1.0)));
  }
  PHX_CHECK(std::isnan(angle::between(1.0, kNaN)) && std::isnan(angle::between(1.0, kInfinity)));
}

// The distance of a value from an enclosure, in binary64 values.
std::int64_t outside(double value, const elementary::Enclosure& enclosure) {
  if (value < enclosure.lower) return ordinal(enclosure.lower) - ordinal(value);
  if (value > enclosure.upper) return ordinal(value) - ordinal(enclosure.upper);
  return 0;
}

void accuracy() {
  using phx::mc::geometry::next_down;
  using phx::mc::geometry::next_up;
  Generator generator{20261118};
  std::int64_t widest_sine = 0, farthest = 0;
  for (int index = 0; index < 200000; ++index) {
    const double first = generator.unit();
    const double second = generator.unit();
    const double third = generator.unit();
    const double fourth = generator.unit();
    // Arguments of the meshers: cosines and sines in [-1, 1], angles of a
    // few radians, ratios over forty binades; every tenth near 1, every
    // tenth near the seam of the kernels at 1/2.
    double unit = 2.0 * first - 1.0;
    if (index % 10 == 0) unit = (unit < 0.0 ? -1.0 : 1.0) * (1.0 - 0x1p-30 * second);
    if (index % 10 == 1) unit = (unit < 0.0 ? -1.0 : 1.0) * (0.5 + 0x1p-30 * (second - 0.5));
    const double radians = 16.0 * second - 8.0;
    const double across = std::ldexp(third, static_cast<int>(40.0 * fourth) - 20);
    const double along = 2.0 * fourth - 1.0;

    // The kernels against the interval kernel's enclosure of the true value.
    farthest = std::max(farthest, outside(angle::arcsine(unit), elementary::asin(unit, unit)));
    farthest = std::max(farthest, outside(angle::arccosine(unit), elementary::acos(unit, unit)));
    const double ratio = across / std::fabs(along);
    const elementary::Enclosure acute = elementary::atan(ratio, ratio);
    if (along > 0.0) {
      farthest = std::max(farthest, outside(angle::between(across, along), acute));
    } else if (along < 0.0) {
      const elementary::Enclosure obtuse = {next_down(angle::kPi - acute.upper),
                                            next_up(next_up(angle::kPi) - acute.lower)};
      farthest = std::max(farthest, outside(angle::between(across, along), obtuse));
    }
    widest_sine = std::max(widest_sine, steps(elementary::sin(radians, radians)));

    // The platform functions are within a value or two of the true ones.
    PHX_CHECK(within(angle::arcsine(unit), std::asin(unit), 6));
    PHX_CHECK(within(angle::arccosine(unit), std::acos(unit), 6));
    PHX_CHECK(within(angle::sine(radians), std::sin(radians), 6) ||
              std::fabs(angle::sine(radians) - std::sin(radians)) <= 0x1p-52);
    PHX_CHECK(within(angle::between(across, along), std::atan2(across, along), 6));
  }
  std::printf("angle kernels: at most %lld values from the enclosure, sine enclosure %lld wide\n",
              static_cast<long long>(farthest), static_cast<long long>(widest_sine));
  PHX_CHECK(farthest <= 3);
  PHX_CHECK(widest_sine <= 4);
}

// function: 0 sine, 1 arcsine, 2 arccosine, 3 between(first, second).
struct Vector {
  int function;
  std::uint64_t first, second, result;
};

double evaluate(int function, double first, double second) {
  switch (function) {
    case 0:
      return angle::sine(first);
    case 1:
      return angle::arcsine(first);
    case 2:
      return angle::arccosine(first);
    default:
      return angle::between(first, second);
  }
}

constexpr Vector kVectors[] = {
#include "angle_vectors.inc"
};

// The inputs of the vector table: the minimum angles, dihedral cosines and
// cone directions the meshers pass, then a generated stream.
template <class Visit>
void for_each_input(Visit visit) {
  for (const double degrees : {5.0, 12.0, 18.0, 20.7, 25.0, 30.0, 33.8, 45.0, 60.0, 90.0}) {
    visit(0, degrees * (3.14159265358979323846 / 180.0), 0.0);
  }
  for (const double value : {-1.0, -0.999, -0.5, -0x1p-30, 0.0, 0x1p-30, 0.25, 0.5, 0.7071067811865476,
                             0.999, 0.9999999999999999, 1.0}) {
    visit(1, value, 0.0);
    visit(2, value, 0.0);
  }
  for (const double across : {0.0, 0x1p-40, 0.5, 1.0, 3.0, 0x1p40}) {
    for (const double along : {-0x1p40, -1.0, -0x1p-40, 0.0, 0x1p-40, 1.0, 0x1p40}) {
      visit(3, across, along);
    }
  }
  Generator generator{20261119};
  for (int index = 0; index < 160; ++index) {
    const double first = generator.unit();
    const double second = generator.unit();
    switch (index % 4) {
      case 0:
        visit(0, 16.0 * first - 8.0, 0.0);
        break;
      case 1:
        visit(1, 2.0 * first - 1.0, 0.0);
        break;
      case 2:
        visit(2, 2.0 * first - 1.0, 0.0);
        break;
      default:
        visit(3, 4.0 * first, 4.0 * second - 2.0);
        break;
    }
  }
}

void emit() {
  std::printf("// Bit vectors of angle.hpp, written by `test_angle --emit`: the function\n"
              "// (0 sine, 1 arcsine, 2 arccosine, 3 between), its arguments and its value\n"
              "// as IEEE 754 binary64 bit patterns.\n");
  for_each_input([](int function, double first, double second) {
    std::printf("{%d, 0x%016" PRIx64 ", 0x%016" PRIx64 ", 0x%016" PRIx64 "},\n", function,
                bits(first), bits(second), bits(evaluate(function, first, second)));
  });
}

void vectors() {
  std::size_t row = 0;
  int reported = 0;
  for_each_input([&](int function, double first, double second) {
    const std::uint64_t result = bits(evaluate(function, first, second));
    const bool known = row < std::size(kVectors);
    const bool same = known && kVectors[row].function == function &&
                      kVectors[row].first == bits(first) &&
                      kVectors[row].second == bits(second) && kVectors[row].result == result;
    PHX_CHECK(same);
    if (!same && known && reported++ < 8) {
      std::fprintf(stderr, "  row %zu function %d: 0x%016" PRIx64 " (recorded 0x%016" PRIx64 ")\n",
                   row, function, result, kVectors[row].result);
    }
    ++row;
  });
  PHX_CHECK(row == std::size(kVectors));
}

}  // namespace

int main(int argc, char** argv) {
  const phx::mc::GradualNearestScope mode;
  if (argc == 2 && std::strcmp(argv[1], "--emit") == 0) {
    emit();
    return 0;
  }
  stated_values();
  quadrants();
  accuracy();
  vectors();
  return phx::mc::test::finish("test_angle");
}
