// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Independent base-2^32 schoolbook references for the wide integer helpers.
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include "check.hpp"
#include "wide_integer.hpp"

namespace {
using phx::mc::Int256;
using Words = std::array<std::uint32_t, 8>;
using Product = std::array<std::uint32_t, 16>;

std::uint64_t random_word(std::uint64_t& state) {
  state = state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
  return state;
}

template <std::size_t N, std::size_t M>
std::array<std::uint32_t, N + M> schoolbook(
    const std::array<std::uint32_t, N>& a, const std::array<std::uint32_t, M>& b) {
  std::array<std::uint32_t, N + M> result{};
  for (std::size_t i = 0; i < N; ++i) {
    std::uint64_t carry = 0;
    for (std::size_t j = 0; j < M; ++j) {
      const std::uint64_t column = static_cast<std::uint64_t>(a[i]) * b[j] +
                                   result[i + j] + carry;
      result[i + j] = static_cast<std::uint32_t>(column);
      carry = column >> 32;
    }
    for (std::size_t k = i + M; carry != 0 && k < result.size(); ++k) {
      const std::uint64_t column = static_cast<std::uint64_t>(result[k]) + carry;
      result[k] = static_cast<std::uint32_t>(column);
      carry = column >> 32;
    }
    PHX_CHECK(carry == 0);
  }
  return result;
}

std::array<std::uint32_t, 2> halves(std::uint64_t value) {
  return {static_cast<std::uint32_t>(value), static_cast<std::uint32_t>(value >> 32)};
}

std::uint64_t pair(const std::uint32_t low, const std::uint32_t high) {
  return static_cast<std::uint64_t>(low) | (static_cast<std::uint64_t>(high) << 32);
}

void add_to_product(std::array<std::uint32_t, 4>& value, std::uint64_t addend) {
  const auto part = halves(addend);
  std::uint64_t carry = 0;
  for (std::size_t i = 0; i < value.size(); ++i) {
    const std::uint64_t total = static_cast<std::uint64_t>(value[i]) +
                                (i < part.size() ? part[i] : 0) + carry;
    value[i] = static_cast<std::uint32_t>(total);
    carry = total >> 32;
  }
  PHX_CHECK(carry == 0);
}

void check_64(std::uint64_t a, std::uint64_t b, std::uint64_t addend,
              std::uint64_t incoming) {
  auto expected = schoolbook(halves(a), halves(b));
  std::uint64_t high = 0;
  const std::uint64_t low = phx::mc::multiply_wide(a, b, high);
  PHX_CHECK(low == pair(expected[0], expected[1]));
  PHX_CHECK(high == pair(expected[2], expected[3]));
  std::uint64_t product = 0;
  const bool fits = phx::mc::checked_multiply(a, b, product);
  PHX_CHECK(fits == (expected[2] == 0 && expected[3] == 0));
  if (fits) PHX_CHECK(product == pair(expected[0], expected[1]));
  add_to_product(expected, addend);
  add_to_product(expected, incoming);
  const std::uint64_t accumulated = phx::mc::multiply_accumulate(a, b, addend, incoming, high);
  PHX_CHECK(accumulated == pair(expected[0], expected[1]));
  PHX_CHECK(high == pair(expected[2], expected[3]));
}

void check_signed_64(std::int64_t a, std::int64_t b) {
  const bool negative = (a < 0) != (b < 0);
  const std::uint64_t ua = static_cast<std::uint64_t>(a);
  const std::uint64_t ub = static_cast<std::uint64_t>(b);
  const auto expected = schoolbook(halves(a < 0 ? 0 - ua : ua),
                                   halves(b < 0 ? 0 - ub : ub));
  const std::uint64_t magnitude = pair(expected[0], expected[1]);
  const std::uint64_t limit = (UINT64_C(1) << 63) - (negative ? 0 : 1);
  const bool fits = expected[2] == 0 && expected[3] == 0 && magnitude <= limit;
  std::int64_t result = 0;
  PHX_CHECK(phx::mc::checked_multiply(a, b, result) == fits);
  if (fits) {
    const std::uint64_t bits = negative ? 0 - magnitude : magnitude;
    PHX_CHECK(result == std::bit_cast<std::int64_t>(bits));
  }
}

void check_add(std::uint64_t a, std::uint64_t b, std::uint64_t incoming) {
  const auto x = halves(a);
  const auto y = halves(b);
  const std::uint64_t first = static_cast<std::uint64_t>(x[0]) + y[0] + incoming;
  const std::uint64_t second = static_cast<std::uint64_t>(x[1]) + y[1] + (first >> 32);
  const std::uint64_t expected = pair(static_cast<std::uint32_t>(first),
                                      static_cast<std::uint32_t>(second));
  std::uint64_t carry = incoming;
  const std::uint64_t actual = phx::mc::add_with_carry(a, b, carry);
  PHX_CHECK(actual == expected);
  PHX_CHECK(carry == (second >> 32));
}

Words twos_complement(Words value) {
  std::uint64_t carry = 1;
  for (std::uint32_t& word : value) {
    const std::uint64_t sum =
        static_cast<std::uint64_t>(static_cast<std::uint32_t>(~word)) + carry;
    word = static_cast<std::uint32_t>(sum);
    carry = sum >> 32;
  }
  return value;
}

Int256 wide(const Words& magnitude, bool negative) {
  const Words words = negative ? twos_complement(magnitude) : magnitude;
  Int256 result;
  for (std::size_t i = 0; i < result.limb.size(); ++i)
    result.limb[i] = pair(words[2 * i], words[2 * i + 1]);
  return result;
}

void check_256(const Words& a, bool a_negative, const Words& b, bool b_negative) {
  const Product product = schoolbook(a, b);
  bool fits = (product[7] & UINT32_C(0x80000000)) == 0;
  for (std::size_t i = 8; i < product.size(); ++i) fits = fits && product[i] == 0;
  Words expected{};
  for (std::size_t i = 0; i < expected.size(); ++i) expected[i] = product[i];
  bool zero = true;
  for (std::uint32_t word : expected) zero = zero && word == 0;
  if ((a_negative != b_negative) && !zero) expected = twos_complement(expected);
  Int256 result;
  const bool actual = phx::mc::checked_multiply(wide(a, a_negative), wide(b, b_negative), result);
  PHX_CHECK(actual == fits);
  if (actual) {
    for (std::size_t i = 0; i < result.limb.size(); ++i)
      PHX_CHECK(result.limb[i] == pair(expected[2 * i], expected[2 * i + 1]));
  }
}

void test_64_bit_helpers() {
  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  const std::array<std::array<std::uint64_t, 2>, 8> cases{{
      {maximum, maximum}, {UINT64_C(1) << 63, 2}, {maximum, 1},
      {UINT64_C(1) << 32, UINT64_C(1) << 32}, {maximum, 2},
      {UINT64_C(0x8000000000000001), 3}, {UINT64_C(0xffffffff00000001), 2}, {0, maximum},
  }};
  for (const auto& values : cases) {
    check_64(values[0], values[1], 0, 0);
    check_64(values[0], values[1], maximum, maximum);
    check_add(values[0], values[1], 0);
    check_add(values[0], values[1], 1);
    check_signed_64(std::bit_cast<std::int64_t>(values[0]),
                    std::bit_cast<std::int64_t>(values[1]));
  }
  check_signed_64(std::numeric_limits<std::int64_t>::min(), 1);
  check_signed_64(std::numeric_limits<std::int64_t>::min(), -1);
  check_signed_64(std::numeric_limits<std::int64_t>::max(), 1);
  check_signed_64(std::numeric_limits<std::int64_t>::max(), 2);
  std::uint64_t state = UINT64_C(0x4d595df4d0f33173);
  for (int i = 0; i < 768; ++i) {
    const std::uint64_t a = random_word(state), b = random_word(state);
    const std::uint64_t addend = random_word(state);
    const std::uint64_t incoming = random_word(state);
    check_64(a, b, addend, incoming);
    check_add(a, b, static_cast<std::uint64_t>(i & 1));
    check_signed_64(std::bit_cast<std::int64_t>(a), std::bit_cast<std::int64_t>(b));
  }
  std::uint64_t carry = 1;
  for (int i = 0; i < 4; ++i) {
    PHX_CHECK(phx::mc::add_with_carry(maximum, 0, carry) == 0);
    PHX_CHECK(carry == 1);
  }
  Int256 all_ones;
  all_ones.limb.fill(maximum);
  Int256 result;
  PHX_CHECK(phx::mc::checked_add(all_ones, Int256::from_int64(1), result));
  PHX_CHECK(result.zero());
}

void test_256_bit_products() {
  Words one{}; one[0] = 1;
  Words two{}; two[0] = 2;
  Words maximum{}; maximum.fill(UINT32_MAX); maximum[7] = UINT32_C(0x7fffffff);
  Words half{}; half[7] = UINT32_C(0x40000000);
  Words below_half = half; below_half[7] -= 1;
  for (std::size_t i = 0; i < 7; ++i) below_half[i] = UINT32_MAX;
  for (bool negative : {false, true}) {
    check_256(maximum, negative, one, false);
    check_256(maximum, negative, two, false);
    check_256(half, negative, two, false);
    check_256(below_half, negative, two, false);
  }
  Int256 minimum{};
  minimum.limb[3] = UINT64_C(1) << 63;
  Int256 out;
  PHX_CHECK(!phx::mc::checked_multiply(minimum, Int256::from_int64(1), out));
  PHX_CHECK(!phx::mc::checked_multiply(minimum, Int256::from_int64(-1), out));
  std::uint64_t state = UINT64_C(0x8f3f73b5cf1c9ade);
  for (int trial = 0; trial < 400; ++trial) {
    Words a{}, b{};
    for (std::size_t i = 0; i < a.size(); ++i) {
      a[i] = static_cast<std::uint32_t>(random_word(state) >> 32);
      b[i] = static_cast<std::uint32_t>(random_word(state) >> 32);
    }
    // Mix fitting products with overflowing products, with deterministic signs.
    for (std::size_t i = static_cast<std::size_t>(1 + trial % 7); i < a.size(); ++i) a[i] = 0;
    for (std::size_t i = static_cast<std::size_t>(1 + (trial * 3) % 7); i < b.size(); ++i) b[i] = 0;
    check_256(a, (trial & 1) != 0, b, (trial & 2) != 0);
  }
}
}  // namespace

int main() {
  test_64_bit_helpers();
  test_256_bit_products();
  return phx::mc::test::finish("test_wide_integer");
}
