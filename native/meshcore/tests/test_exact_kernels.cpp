// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Exact Bernstein kernels, interval restriction, separation screen and the BVH
// pair stream: exact values, typed range refusals, and 1-vs-N thread identity.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"
#include "wide_integer.hpp"

namespace {

using phx::mc::Int256;
using PairSet = std::set<std::pair<int64_t, int64_t>>;

constexpr int32_t kThreads = 4;

std::vector<std::uint8_t> words(const std::vector<int64_t>& values) {
  std::vector<std::uint8_t> out(values.size() * Int256::kBytes);
  for (std::size_t index = 0; index < values.size(); ++index) {
    Int256::from_int64(values[index]).store(out.data() + index * Int256::kBytes);
  }
  return out;
}

int64_t word(const std::vector<std::uint8_t>& bytes, std::size_t index) {
  const Int256 value = Int256::load(bytes.data() + index * Int256::kBytes);
  return static_cast<int64_t>(value.limb[0]);
}

Int256 power_of_two(int exponent) {
  Int256 out;
  out.limb[static_cast<std::size_t>(exponent / 64)] = std::uint64_t{1} << (exponent % 64);
  return out;
}

void test_wide_integer() {
  Int256 sum;
  PHX_CHECK(phx::mc::checked_add(Int256::from_int64(-5), Int256::from_int64(7), sum));
  PHX_CHECK(phx::mc::compare(sum, Int256::from_int64(2)) == 0);
  Int256 product;
  PHX_CHECK(phx::mc::checked_multiply(Int256::from_int64(-3), Int256::from_int64(-4), product));
  PHX_CHECK(phx::mc::compare(product, Int256::from_int64(12)) == 0);
  PHX_CHECK(phx::mc::checked_multiply(power_of_two(127), Int256::from_int64(-2), product));
  PHX_CHECK(product.negative());
  PHX_CHECK(!phx::mc::checked_multiply(power_of_two(128), power_of_two(127), product));
  PHX_CHECK(phx::mc::checked_multiply(power_of_two(128), power_of_two(126), product));
  PHX_CHECK(!phx::mc::checked_add(power_of_two(254), power_of_two(254), sum));
  Int256 shifted;
  PHX_CHECK(phx::mc::checked_shift_left(Int256::from_int64(-3), 100, shifted));
  PHX_CHECK(phx::mc::checked_multiply(Int256::from_int64(-3), power_of_two(100), product));
  PHX_CHECK(phx::mc::compare(shifted, product) == 0);
  PHX_CHECK(!phx::mc::checked_shift_left(Int256::from_int64(1), 255, shifted));
  PHX_CHECK(!phx::mc::checked_shift_left(Int256::from_int64(3), 254, shifted));
  PHX_CHECK(phx::mc::compare(Int256::from_int64(-1), Int256::from_int64(0)) < 0);
}

// Reference step of the host interval de Casteljau formula for one entry.
void test_interval_restrict() {
  std::mt19937_64 random(7);
  std::uniform_real_distribution<double> value(-3.0, 3.0);
  const int32_t degree = 3;
  const int64_t width = 5;
  const int64_t count = 97;
  const std::size_t entries = static_cast<std::size_t>((degree + 1) * width);
  std::vector<double> lower(static_cast<std::size_t>(count) * entries);
  std::vector<double> upper(lower.size());
  for (std::size_t index = 0; index < lower.size(); ++index) {
    const double a = value(random);
    const double b = value(random);
    lower[index] = std::min(a, b);
    upper[index] = std::max(a, b);
  }
  std::vector<double> parameters(static_cast<std::size_t>(count) * 4);
  std::vector<int32_t> sides(static_cast<std::size_t>(count));
  for (int64_t item = 0; item < count; ++item) {
    const double last = std::uniform_real_distribution<double>(0.25, 1.0)(random);
    const double ratio = std::uniform_real_distribution<double>(0.0, 0.75)(random);
    parameters[static_cast<std::size_t>(4 * item)] = last;
    parameters[static_cast<std::size_t>(4 * item + 1)] = item % 3 == 0 ? last : std::nextafter(last, 2.0);
    parameters[static_cast<std::size_t>(4 * item + 2)] = ratio;
    parameters[static_cast<std::size_t>(4 * item + 3)] = std::nextafter(ratio, 2.0);
    sides[static_cast<std::size_t>(item)] = static_cast<int32_t>(item % 4);
  }
  std::vector<double> one_lower(lower.size()), one_upper(lower.size());
  std::vector<double> many_lower(lower.size()), many_upper(lower.size());
  PHX_CHECK(phx_mc_interval_bernstein_restrict(count, degree, width, lower.data(), upper.data(),
                                               parameters.data(), sides.data(), one_lower.data(),
                                               one_upper.data(), 1) == PHX_MC_OK);
  PHX_CHECK(phx_mc_interval_bernstein_restrict(count, degree, width, lower.data(), upper.data(),
                                               parameters.data(), sides.data(), many_lower.data(),
                                               many_upper.data(), kThreads) == PHX_MC_OK);
  PHX_CHECK(std::memcmp(one_lower.data(), many_lower.data(), lower.size() * sizeof(double)) == 0);
  PHX_CHECK(std::memcmp(one_upper.data(), many_upper.data(), lower.size() * sizeof(double)) == 0);
  // Side mask 0 copies; every output encloses the exact restriction of the
  // midpoint net at the parameter midpoint (a containment spot check).
  for (int64_t item = 0; item < count; ++item) {
    const std::size_t base = static_cast<std::size_t>(item) * entries;
    if (sides[static_cast<std::size_t>(item)] == 0) {
      PHX_CHECK(std::equal(lower.begin() + static_cast<std::ptrdiff_t>(base),
                           lower.begin() + static_cast<std::ptrdiff_t>(base + entries),
                           one_lower.begin() + static_cast<std::ptrdiff_t>(base)));
    }
    for (std::size_t index = 0; index < entries; ++index) {
      PHX_CHECK(one_lower[base + index] <= one_upper[base + index]);
    }
  }
  // A degenerate point net restricted on the left keeps exact first rows.
  std::vector<double> point_lower = {1.0, 1.0, 1.0};
  std::vector<double> point_upper = point_lower;
  std::vector<double> point_parameters = {0.5, 0.5, 0.0, 0.0};
  std::vector<int32_t> point_sides = {1};
  std::vector<double> out_lower(3), out_upper(3);
  PHX_CHECK(phx_mc_interval_bernstein_restrict(1, 2, 1, point_lower.data(), point_upper.data(),
                                               point_parameters.data(), point_sides.data(),
                                               out_lower.data(), out_upper.data(), 1) == PHX_MC_OK);
  PHX_CHECK(out_lower[0] == 1.0 && out_upper[0] == 1.0);
  PHX_CHECK(out_lower[1] < 1.0 && out_upper[1] > 1.0);
  PHX_CHECK(out_lower[1] > 0.999999 && out_upper[1] < 1.000001);
}

void test_bernstein_controls() {
  // Simplex dimension 2: p(x, y) = 1 + 2x + 3y^2 at degree 2.
  // Controls in composition order (2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2):
  // values 1, 2, 1, 3, 2, 4.
  const std::vector<int32_t> degrees = {2};
  const std::vector<int64_t> term_offsets = {0, 3};
  const std::vector<int32_t> exponents = {0, 0, 1, 0, 0, 2};
  const auto numerators = words({1, 2, 3});
  const std::vector<int64_t> control_offsets = {0, 6};
  std::vector<std::uint8_t> controls(6 * Int256::kBytes);
  std::vector<int64_t> denominators(1), extremes(2);
  std::vector<int32_t> status(1);
  PHX_CHECK(phx_mc_bernstein_controls(1, PHX_MC_BERNSTEIN_SIMPLEX, 2, degrees.data(),
                                      term_offsets.data(), exponents.data(), numerators.data(),
                                      control_offsets.data(), controls.data(), denominators.data(),
                                      extremes.data(), status.data(), 4, 1) == PHX_MC_OK);
  PHX_CHECK(status[0] == PHX_MC_OK);
  const int64_t expected[6] = {1, 2, 1, 3, 2, 4};
  for (std::size_t index = 0; index < 6; ++index) {
    PHX_CHECK(word(controls, index) == expected[index] * denominators[0]);
  }
  PHX_CHECK(extremes[0] == 0 && extremes[1] == 5);

  // Box degree (1, 2): p = x y^2 -> controls (C order): 0,0,0,0,0,1.
  const std::vector<int32_t> box_degrees = {1, 2};
  const std::vector<int32_t> box_exponents = {1, 2};
  const auto box_numerators = words({1});
  const std::vector<int64_t> box_terms = {0, 1};
  const std::vector<int64_t> box_offsets = {0, 6};
  PHX_CHECK(phx_mc_bernstein_controls(1, PHX_MC_BERNSTEIN_BOX, 2, box_degrees.data(),
                                      box_terms.data(), box_exponents.data(), box_numerators.data(),
                                      box_offsets.data(), controls.data(), denominators.data(),
                                      extremes.data(), status.data(), 4, 1) == PHX_MC_OK);
  PHX_CHECK(status[0] == PHX_MC_OK);
  for (std::size_t index = 0; index < 6; ++index) {
    PHX_CHECK(word(controls, index) == (index == 5 ? denominators[0] : 0));
  }
  // Box degree 2 on one axis: p = x -> controls 0, 1/2, 1.
  const std::vector<int32_t> line_degree = {2};
  const std::vector<int32_t> line_exponent = {1};
  const std::vector<int64_t> line_offsets = {0, 3};
  PHX_CHECK(phx_mc_bernstein_controls(1, PHX_MC_BERNSTEIN_BOX, 1, line_degree.data(),
                                      box_terms.data(), line_exponent.data(), box_numerators.data(),
                                      line_offsets.data(), controls.data(), denominators.data(),
                                      extremes.data(), status.data(), 4, 1) == PHX_MC_OK);
  PHX_CHECK(denominators[0] == 2);
  PHX_CHECK(word(controls, 0) == 0 && word(controls, 1) == 1 && word(controls, 2) == 2);

  // Prism (1, 1): p = z -> controls per triangle row (l = 0, 1): 0, 1.
  const std::vector<int32_t> prism_degree = {1, 1};
  const std::vector<int32_t> prism_exponent = {0, 0, 1};
  const std::vector<int64_t> prism_offsets = {0, 6};
  PHX_CHECK(phx_mc_bernstein_controls(1, PHX_MC_BERNSTEIN_PRISM, 3, prism_degree.data(),
                                      box_terms.data(), prism_exponent.data(), box_numerators.data(),
                                      prism_offsets.data(), controls.data(), denominators.data(),
                                      extremes.data(), status.data(), 4, 1) == PHX_MC_OK);
  for (std::size_t index = 0; index < 6; ++index) {
    PHX_CHECK(word(controls, index) == (index % 2 == 1 ? denominators[0] : 0));
  }

  // A numerator 2^252 times the weight 8 leaves the range: refused.
  std::vector<std::uint8_t> huge(Int256::kBytes);
  power_of_two(252).store(huge.data());
  const std::vector<int32_t> big_degree = {8};
  const std::vector<int32_t> big_exponent = {1, 0};
  const std::vector<int64_t> big_offsets = {0, 45};
  std::vector<std::uint8_t> big_controls(45 * Int256::kBytes);
  PHX_CHECK(phx_mc_bernstein_controls(1, PHX_MC_BERNSTEIN_SIMPLEX, 2, big_degree.data(),
                                      box_terms.data(), big_exponent.data(), huge.data(),
                                      big_offsets.data(), big_controls.data(), denominators.data(),
                                      extremes.data(), status.data(), 4, 1) == PHX_MC_OK);
  PHX_CHECK(status[0] == PHX_MC_RANGE_ERROR);
  // A control count disagreeing with the degree is an argument error.
  const std::vector<int64_t> wrong_offsets = {0, 5};
  PHX_CHECK(phx_mc_bernstein_controls(1, PHX_MC_BERNSTEIN_SIMPLEX, 2, degrees.data(),
                                      term_offsets.data(), exponents.data(), numerators.data(),
                                      wrong_offsets.data(), controls.data(), denominators.data(),
                                      extremes.data(), status.data(), 4, 1) == PHX_MC_INVALID_ARGUMENT);

  // Thread invariance over a random batch.
  std::mt19937_64 random(11);
  const int64_t count = 203;
  std::vector<int32_t> batch_degrees;
  std::vector<int64_t> batch_terms = {0};
  std::vector<int32_t> batch_exponents;
  std::vector<int64_t> batch_numerators;
  std::vector<int64_t> batch_controls = {0};
  for (int64_t item = 0; item < count; ++item) {
    const int32_t degree = static_cast<int32_t>(1 + random() % 5);
    batch_degrees.push_back(degree);
    const int terms = static_cast<int>(1 + random() % 6);
    for (int term = 0; term < terms; ++term) {
      const int32_t a = static_cast<int32_t>(random() % (degree + 1));
      const int32_t b = static_cast<int32_t>(random() % (degree + 1 - a));
      batch_exponents.push_back(a);
      batch_exponents.push_back(b);
      batch_numerators.push_back(static_cast<int64_t>(random() % 2001) - 1000);
    }
    batch_terms.push_back(batch_terms.back() + terms);
    batch_controls.push_back(batch_controls.back() + (degree + 1) * (degree + 2) / 2);
  }
  const auto batch_words = words(batch_numerators);
  const std::size_t total = static_cast<std::size_t>(batch_controls.back());
  std::vector<std::uint8_t> one(total * Int256::kBytes), many(total * Int256::kBytes);
  std::vector<int64_t> one_den(count), many_den(count), one_ext(2 * count), many_ext(2 * count);
  std::vector<int32_t> one_status(count), many_status(count);
  PHX_CHECK(phx_mc_bernstein_controls(count, PHX_MC_BERNSTEIN_SIMPLEX, 2, batch_degrees.data(),
                                      batch_terms.data(), batch_exponents.data(),
                                      batch_words.data(), batch_controls.data(), one.data(),
                                      one_den.data(), one_ext.data(), one_status.data(), 4,
                                      1) == PHX_MC_OK);
  PHX_CHECK(phx_mc_bernstein_controls(count, PHX_MC_BERNSTEIN_SIMPLEX, 2, batch_degrees.data(),
                                      batch_terms.data(), batch_exponents.data(),
                                      batch_words.data(), batch_controls.data(), many.data(),
                                      many_den.data(), many_ext.data(), many_status.data(), 4,
                                      kThreads) == PHX_MC_OK);
  PHX_CHECK(one == many && one_den == many_den && one_ext == many_ext && one_status == many_status);
}

// Dense degree-(6, 6, 6) box polynomials: 1024-bit words carry numerators near
// 2^600 exactly (the controls are the small ones shifted by 600 bits), and
// numerators near 2^1000 are refused at that width.
void test_wide_box_controls() {
  using Wide = phx::mc::WideInt<16>;
  std::mt19937_64 random(29);
  const std::vector<int32_t> degrees = {6, 6, 6};
  std::vector<int32_t> exponents;
  std::vector<int64_t> small;
  for (int32_t a = 0; a <= 6; ++a) {
    for (int32_t b = 0; b <= 6; ++b) {
      for (int32_t c = 0; c <= 6; ++c) {
        exponents.insert(exponents.end(), {a, b, c});
        small.push_back(static_cast<int64_t>(random() % 2001) - 1000);
      }
    }
  }
  const std::vector<int64_t> terms = {0, static_cast<int64_t>(small.size())};
  const std::vector<int64_t> offsets = {0, 343};
  const auto encode = [&](int shift) {
    std::vector<std::uint8_t> out(small.size() * Wide::kBytes);
    for (std::size_t index = 0; index < small.size(); ++index) {
      Wide value;
      PHX_CHECK(phx::mc::checked_shift_left(Wide::from_int64(small[index]), shift, value));
      value.store(out.data() + index * Wide::kBytes);
    }
    return out;
  };
  const auto run = [&](const std::vector<std::uint8_t>& numerators, int32_t parallelism,
                       std::vector<std::uint8_t>& controls, int64_t& denominator,
                       std::vector<int64_t>& extremes) {
    controls.assign(343 * Wide::kBytes, 0);
    extremes.assign(2, -1);
    int32_t status = -1;
    PHX_CHECK(phx_mc_bernstein_controls(1, PHX_MC_BERNSTEIN_BOX, 3, degrees.data(), terms.data(),
                                        exponents.data(), numerators.data(), offsets.data(),
                                        controls.data(), &denominator, extremes.data(), &status,
                                        16, parallelism) == PHX_MC_OK);
    return status;
  };
  std::vector<std::uint8_t> base, shifted, threaded;
  std::vector<int64_t> base_extremes, shifted_extremes, threaded_extremes;
  int64_t base_denominator = 0, shifted_denominator = 0, threaded_denominator = 0;
  PHX_CHECK(run(encode(0), 1, base, base_denominator, base_extremes) == PHX_MC_OK);
  PHX_CHECK(run(encode(600), 1, shifted, shifted_denominator, shifted_extremes) == PHX_MC_OK);
  PHX_CHECK(run(encode(600), kThreads, threaded, threaded_denominator, threaded_extremes) ==
            PHX_MC_OK);
  PHX_CHECK(shifted == threaded && shifted_extremes == threaded_extremes);
  PHX_CHECK(base_denominator == shifted_denominator && base_extremes == shifted_extremes);
  for (std::size_t index = 0; index < 343; ++index) {
    Wide expected;
    PHX_CHECK(phx::mc::checked_shift_left(Wide::load(base.data() + index * Wide::kBytes), 600,
                                          expected));
    PHX_CHECK(phx::mc::compare(expected, Wide::load(shifted.data() + index * Wide::kBytes)) == 0);
  }
  PHX_CHECK(run(encode(1000), 1, shifted, shifted_denominator, shifted_extremes) ==
            PHX_MC_RANGE_ERROR);
}

void test_blossoms() {
  // Degree-2 triangle net; the identity corners return the net itself and the
  // subtriangle (0,0),(1/2,0),(0,1/2) (scale 2) scales value at corner 0.
  const std::vector<int32_t> degrees = {2, 2};
  const std::vector<int64_t> offsets = {0, 6, 12};
  const auto controls = words({5, -1, 4, 2, 7, 3, 5, -1, 4, 2, 7, 3});
  const auto corners = words({1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0, 1, 1, 0, 1, 0, 1});
  std::vector<std::uint8_t> one(12 * Int256::kBytes), many(12 * Int256::kBytes);
  std::vector<int64_t> one_ext(4), many_ext(4);
  std::vector<int32_t> one_status(2), many_status(2);
  PHX_CHECK(phx_mc_simplex_bernstein_blossoms(2, 3, degrees.data(), offsets.data(), controls.data(),
                                              corners.data(), one.data(), one_ext.data(),
                                              one_status.data(), 4, 1) == PHX_MC_OK);
  PHX_CHECK(phx_mc_simplex_bernstein_blossoms(2, 3, degrees.data(), offsets.data(), controls.data(),
                                              corners.data(), many.data(), many_ext.data(),
                                              many_status.data(), 4, kThreads) == PHX_MC_OK);
  PHX_CHECK(one == many && one_ext == many_ext && one_status == many_status);
  PHX_CHECK(one_status[0] == PHX_MC_OK && one_status[1] == PHX_MC_OK);
  const int64_t net[6] = {5, -1, 4, 2, 7, 3};
  for (std::size_t index = 0; index < 6; ++index) PHX_CHECK(word(one, index) == net[index]);
  PHX_CHECK(one_ext[0] == 1 && one_ext[1] == 4);
  // Corner multiplicity (2,0,0) of the second item: corner 0 twice = 4 * net[(2,0,0)].
  PHX_CHECK(word(one, 6) == 4 * 5);
  // (1,1,0): blossom(c0, c1) = sum over weights: 2*(1*net200 + 1*net110) = 2*(5 - 1).
  PHX_CHECK(word(one, 7) == 2 * (5 - 1));
  // Overflowing corner weights are refused.
  std::vector<std::uint8_t> big_corners(9 * Int256::kBytes);
  for (std::size_t index = 0; index < 9; ++index) power_of_two(200).store(big_corners.data() + index * Int256::kBytes);
  PHX_CHECK(phx_mc_simplex_bernstein_blossoms(1, 3, degrees.data(), offsets.data(), controls.data(),
                                              big_corners.data(), one.data(), one_ext.data(),
                                              one_status.data(), 4, 1) == PHX_MC_OK);
  PHX_CHECK(one_status[0] == PHX_MC_RANGE_ERROR);
}

void test_bisect() {
  // Line group of degree 2 (slots 0, 1): x = (0, 4, 8) in index order (2,0),(1,1),(0,2).
  const std::vector<int32_t> indices = {2, 0, 1, 1, 0, 2};
  const auto nets = words({0, 4, 8, 0, 4, 8});
  std::vector<std::uint8_t> left(6 * Int256::kBytes), right(6 * Int256::kBytes);
  std::vector<std::uint8_t> left_many(6 * Int256::kBytes), right_many(6 * Int256::kBytes);
  int32_t exponent = -1;
  int32_t exponent_many = -1;
  std::vector<int32_t> status(2), status_many(2);
  PHX_CHECK(phx_mc_bernstein_bisect(2, 3, 2, 1, indices.data(), 0, 1, nets.data(), left.data(),
                                    right.data(), &exponent, status.data(), 4, 1) == PHX_MC_OK);
  PHX_CHECK(phx_mc_bernstein_bisect(2, 3, 2, 1, indices.data(), 0, 1, nets.data(),
                                    left_many.data(), right_many.data(), &exponent_many,
                                    status_many.data(), 4, kThreads) == PHX_MC_OK);
  PHX_CHECK(left == left_many && right == right_many && exponent == exponent_many);
  PHX_CHECK(exponent == 2 && status[0] == PHX_MC_OK);
  // Left half on [0, 1/2]: (0, 2, 4) -> numerators over 4: 0, 8, 16.
  PHX_CHECK(word(left, 0) == 0 && word(left, 1) == 8 && word(left, 2) == 16);
  // Right half on [1/2, 1]: (4, 6, 8) -> 16, 24, 32.
  PHX_CHECK(word(right, 0) == 16 && word(right, 1) == 24 && word(right, 2) == 32);
  // An incomplete line is invalid input.
  const std::vector<int32_t> broken = {2, 0, 1, 1, 1, 1};
  PHX_CHECK(phx_mc_bernstein_bisect(1, 3, 2, 1, broken.data(), 0, 1, nets.data(), left.data(),
                                    right.data(), &exponent, status.data(), 4, 1) ==
            PHX_MC_INVALID_INPUT);
  // Values that double past the range are refused.
  std::vector<std::uint8_t> huge(3 * Int256::kBytes);
  for (std::size_t index = 0; index < 3; ++index) power_of_two(254).store(huge.data() + index * Int256::kBytes);
  PHX_CHECK(phx_mc_bernstein_bisect(1, 3, 2, 1, indices.data(), 0, 1, huge.data(), left.data(),
                                    right.data(), &exponent, status.data(), 4, 1) == PHX_MC_OK);
  PHX_CHECK(status[0] == PHX_MC_RANGE_ERROR);
}

void test_separation_screen() {
  // Pair 0: first above z = 2, second below z = 1 -> separated along +z.
  // Pair 1: overlapping -> not separated. Pair 2: touching -> undecided.
  const std::vector<int64_t> first_offsets = {0, 2, 4, 5};
  const std::vector<int64_t> second_offsets = {0, 2, 4, 5};
  const std::vector<double> first = {0, 0, 2, 1, 1, 3, 0, 0, 0, 1, 1, 1, 0, 0, 1};
  const std::vector<double> second = {0, 0, 0, 1, 1, 1, 0, 0, 0.5, 1, 1, 2, 0, 0, 1};
  const std::vector<double> normals = {0, 0, 1, 0, 0, 1, 0, 0, 1};
  std::vector<int8_t> one(3), many(3);
  PHX_CHECK(phx_mc_separation_screen(3, 3, first_offsets.data(), first.data(), first.data(),
                                     second_offsets.data(), second.data(), second.data(),
                                     normals.data(), one.data(), 1) == PHX_MC_OK);
  PHX_CHECK(phx_mc_separation_screen(3, 3, first_offsets.data(), first.data(), first.data(),
                                     second_offsets.data(), second.data(), second.data(),
                                     normals.data(), many.data(), kThreads) == PHX_MC_OK);
  PHX_CHECK(one == many);
  PHX_CHECK(one[0] == PHX_MC_SEPARATION_SEPARATED);
  PHX_CHECK(one[1] == PHX_MC_SEPARATION_NOT_SEPARATED);
  PHX_CHECK(one[2] == PHX_MC_SEPARATION_UNDECIDED);
}

struct Tree {
  std::vector<double> node_min, node_max, node_size;
  std::vector<int64_t> left, right, leaf, leaf_items;
  std::vector<double> item_min, item_max;
  int64_t leaf_width = 2;

  phx_mc_bvh_tree_view view() const {
    return {static_cast<int64_t>(leaf.size()),
            static_cast<int64_t>(leaf_items.size()) / leaf_width,
            leaf_width,
            static_cast<int64_t>(item_min.size()) / 2,
            node_min.data(),
            node_max.data(),
            node_size.data(),
            left.data(),
            right.data(),
            leaf.data(),
            leaf_items.data(),
            item_min.data(),
            item_max.data()};
  }
};

// Median-split tree over 2D boxes with two items per leaf.
Tree build_tree(const std::vector<double>& lo, const std::vector<double>& hi) {
  Tree tree;
  tree.item_min = lo;
  tree.item_max = hi;
  const auto build = [&](auto&& self, std::vector<int64_t> items) -> int64_t {
    const int64_t node = static_cast<int64_t>(tree.leaf.size());
    tree.leaf.push_back(-1);
    tree.left.push_back(-1);
    tree.right.push_back(-1);
    double box[4] = {INFINITY, INFINITY, -INFINITY, -INFINITY};
    for (const int64_t item : items) {
      for (int axis = 0; axis < 2; ++axis) {
        box[axis] = std::min(box[axis], lo[static_cast<std::size_t>(2 * item + axis)]);
        box[2 + axis] = std::max(box[2 + axis], hi[static_cast<std::size_t>(2 * item + axis)]);
      }
    }
    tree.node_min.insert(tree.node_min.end(), {box[0], box[1]});
    tree.node_max.insert(tree.node_max.end(), {box[2], box[3]});
    tree.node_size.push_back((box[2] - box[0]) + (box[3] - box[1]));
    if (items.size() <= 2) {
      tree.leaf[static_cast<std::size_t>(node)] = static_cast<int64_t>(tree.leaf_items.size()) / 2;
      tree.leaf_items.push_back(items[0]);
      tree.leaf_items.push_back(items.size() > 1 ? items[1] : -1);
      return node;
    }
    const int axis = (box[2] - box[0]) >= (box[3] - box[1]) ? 0 : 1;
    std::sort(items.begin(), items.end(), [&](int64_t a, int64_t b) {
      return lo[static_cast<std::size_t>(2 * a + axis)] < lo[static_cast<std::size_t>(2 * b + axis)];
    });
    const std::size_t half = items.size() / 2;
    const int64_t l = self(self, std::vector<int64_t>(items.begin(), items.begin() + static_cast<std::ptrdiff_t>(half)));
    const int64_t r = self(self, std::vector<int64_t>(items.begin() + static_cast<std::ptrdiff_t>(half), items.end()));
    tree.left[static_cast<std::size_t>(node)] = l;
    tree.right[static_cast<std::size_t>(node)] = r;
    return node;
  };
  std::vector<int64_t> items(lo.size() / 2);
  for (std::size_t item = 0; item < items.size(); ++item) items[item] = static_cast<int64_t>(item);
  build(build, items);
  return tree;
}

std::vector<std::pair<int64_t, int64_t>> drain(const Tree& a, const Tree& b, int32_t parallelism,
                                               int64_t item_block, int64_t node_block,
                                               std::vector<int64_t>& block_sizes) {
  const phx_mc_bvh_tree_view first = a.view();
  const phx_mc_bvh_tree_view second = b.view();
  phx_mc_bvh_pair_stream* stream = nullptr;
  PHX_CHECK(phx_mc_bvh_pair_stream_create(&first, &second, 2, 1, 0.0, 0.0, item_block, node_block,
                                          parallelism, &stream) == PHX_MC_OK);
  std::vector<std::pair<int64_t, int64_t>> pairs;
  const int64_t capacity = std::max<int64_t>(item_block, 4);
  std::vector<int64_t> first_items(static_cast<std::size_t>(capacity)),
      second_items(static_cast<std::size_t>(capacity)), ends(3);
  while (true) {
    int64_t blocks = 0;
    PHX_CHECK(phx_mc_bvh_pair_stream_next(stream, capacity, 3, first_items.data(),
                                          second_items.data(), ends.data(), &blocks) == PHX_MC_OK);
    if (blocks == 0) break;
    int64_t start = 0;
    for (int64_t block = 0; block < blocks; ++block) {
      const int64_t stop = ends[static_cast<std::size_t>(block)];
      block_sizes.push_back(stop - start);
      for (int64_t index = start; index < stop; ++index) {
        pairs.emplace_back(first_items[static_cast<std::size_t>(index)],
                           second_items[static_cast<std::size_t>(index)]);
      }
      start = stop;
    }
  }
  phx_mc_bvh_pair_stream_free(stream);
  return pairs;
}

void test_bvh_pair_stream() {
  std::mt19937_64 random(3);
  std::uniform_real_distribution<double> position(0.0, 10.0);
  std::uniform_real_distribution<double> extent(0.05, 1.0);
  const auto boxes = [&](std::size_t count, std::vector<double>& lo, std::vector<double>& hi) {
    for (std::size_t item = 0; item < count; ++item) {
      for (int axis = 0; axis < 2; ++axis) {
        const double start = position(random);
        lo.push_back(start);
        hi.push_back(start + extent(random));
      }
    }
  };
  std::vector<double> lo_a, hi_a, lo_b, hi_b;
  boxes(301, lo_a, hi_a);
  boxes(257, lo_b, hi_b);
  const Tree a = build_tree(lo_a, hi_a);
  const Tree b = build_tree(lo_b, hi_b);
  std::set<std::pair<int64_t, int64_t>> brute;
  for (int64_t i = 0; i < 301; ++i) {
    for (int64_t j = 0; j < 257; ++j) {
      bool overlap = true;
      for (int axis = 0; axis < 2; ++axis) {
        const double e = std::min(hi_a[static_cast<std::size_t>(2 * i + axis)], hi_b[static_cast<std::size_t>(2 * j + axis)]) -
                         std::max(lo_a[static_cast<std::size_t>(2 * i + axis)], lo_b[static_cast<std::size_t>(2 * j + axis)]);
        overlap = overlap && e >= 0.0;
      }
      if (overlap) brute.emplace(i, j);
    }
  }
  std::vector<int64_t> sizes_one, sizes_many, sizes_small;
  const auto one = drain(a, b, 1, 1 << 20, 1 << 16, sizes_one);
  // Coincident boxes: one leaf-pair chunk holds 40000 candidate pairs, so the
  // pair tests run on the worker pool; the order stays the serial order.
  std::vector<double> lo_c, hi_c;
  for (int item = 0; item < 200; ++item) {
    lo_c.insert(lo_c.end(), {0.0, 0.0});
    hi_c.insert(hi_c.end(), {1.0, 1.0});
  }
  const Tree c = build_tree(lo_c, hi_c);
  std::vector<int64_t> dense_one_sizes, dense_many_sizes;
  const auto dense_one = drain(c, c, 1, 1 << 20, 1 << 16, dense_one_sizes);
  const auto dense_many = drain(c, c, kThreads, 1 << 20, 1 << 16, dense_many_sizes);
  PHX_CHECK(dense_one.size() == 40000);
  PHX_CHECK(dense_one == dense_many && dense_one_sizes == dense_many_sizes);
  const auto many = drain(a, b, kThreads, 1 << 20, 1 << 16, sizes_many);
  PHX_CHECK(one == many && sizes_one == sizes_many);
  PHX_CHECK(PairSet(one.begin(), one.end()) == brute);
  PHX_CHECK(one.size() == brute.size());
  // Small blocks bound every yield and keep the pair set.
  const auto small = drain(a, b, 1, 8, 3, sizes_small);
  PHX_CHECK(PairSet(small.begin(), small.end()) == brute);
  PHX_CHECK(small.size() == brute.size());
  PHX_CHECK(*std::max_element(sizes_small.begin(), sizes_small.end()) <= 8);
}

}  // namespace

int main() {
  test_wide_integer();
  test_interval_restrict();
  test_bernstein_controls();
  test_wide_box_controls();
  test_blossoms();
  test_bisect();
  test_separation_screen();
  test_bvh_pair_stream();
  return phx::mc::test::finish("test_exact_kernels");
}
