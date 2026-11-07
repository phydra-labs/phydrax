// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Fixed-width two's-complement integers of N 64-bit limbs with checked
// arithmetic. Every operation reports overflow instead of wrapping, so a kernel
// can refuse an item whose exact value leaves the representable range. Limb
// arithmetic is exact on every compiler: the 64 x 64 -> 128-bit product uses
// the compiler's exact wide multiply where one exists and a portable
// 32-bit partial-product split otherwise; all routes give identical bits.
#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(_MSC_VER) && !defined(__SIZEOF_INT128__) && \
    (defined(_M_X64) || defined(_M_ARM64))
#include <intrin.h>
#endif

namespace phx::mc {

static_assert(std::endian::native == std::endian::little,
              "wide integer words are little-endian byte images");

// a + b + carry as one limb; `carry` (0 or 1) receives the outgoing carry.
inline std::uint64_t add_with_carry(std::uint64_t a, std::uint64_t b,
                                    std::uint64_t& carry) noexcept {
  const std::uint64_t partial = a + b;
  const std::uint64_t sum = partial + carry;
  carry = static_cast<std::uint64_t>(partial < a) + static_cast<std::uint64_t>(sum < partial);
  return sum;
}

// Exact 128-bit product a * b: returns the low limb and stores the high limb.
inline std::uint64_t multiply_wide(std::uint64_t a, std::uint64_t b,
                                   std::uint64_t& high) noexcept {
#if defined(__SIZEOF_INT128__)
  __extension__ using Product = unsigned __int128;
  const Product product = static_cast<Product>(a) * b;
  high = static_cast<std::uint64_t>(product >> 64);
  return static_cast<std::uint64_t>(product);
#elif defined(_MSC_VER) && defined(_M_X64)
  return _umul128(a, b, &high);
#elif defined(_MSC_VER) && defined(_M_ARM64)
  high = __umulh(a, b);
  return a * b;
#else
  constexpr std::uint64_t kLow32 = 0xFFFFFFFFu;
  const std::uint64_t low_low = (a & kLow32) * (b & kLow32);
  const std::uint64_t high_low = (a >> 32) * (b & kLow32);
  const std::uint64_t low_high = (a & kLow32) * (b >> 32);
  const std::uint64_t high_high = (a >> 32) * (b >> 32);
  // At most (2^32 - 1)^2 + 2 (2^32 - 1) = 2^64 - 1, so the column sum is exact.
  const std::uint64_t middle = (low_low >> 32) + (high_low & kLow32) + low_high;
  high = high_high + (high_low >> 32) + (middle >> 32);
  return (middle << 32) | (low_low & kLow32);
#endif
}

// out = a * b; false when the exact product is not representable.
inline bool checked_multiply(std::uint64_t a, std::uint64_t b, std::uint64_t& out) noexcept {
  std::uint64_t high = 0;
  const std::uint64_t low = multiply_wide(a, b, high);
  if (high != 0) return false;
  out = low;
  return true;
}

// out = a * b; false when the exact product is not representable.
inline bool checked_multiply(std::int64_t a, std::int64_t b, std::int64_t& out) noexcept {
  const bool negative = (a < 0) != (b < 0);
  const std::uint64_t a_magnitude =
      a < 0 ? std::uint64_t{0} - static_cast<std::uint64_t>(a) : static_cast<std::uint64_t>(a);
  const std::uint64_t b_magnitude =
      b < 0 ? std::uint64_t{0} - static_cast<std::uint64_t>(b) : static_cast<std::uint64_t>(b);
  std::uint64_t high = 0;
  const std::uint64_t magnitude = multiply_wide(a_magnitude, b_magnitude, high);
  const std::uint64_t limit = (std::uint64_t{1} << 63) - (negative ? 0 : 1);
  if (high != 0 || magnitude > limit) return false;
  out = static_cast<std::int64_t>(negative ? std::uint64_t{0} - magnitude : magnitude);
  return true;
}

template <std::size_t N>
struct WideInt {
  static_assert(N >= 2, "wide integers hold at least two limbs");
  std::array<std::uint64_t, N> limb{};  // little-endian limbs, two's complement

  static constexpr std::size_t kLimbs = N;
  static constexpr std::size_t kBytes = 8 * N;

  static WideInt from_int64(std::int64_t value) noexcept {
    WideInt out;
    const std::uint64_t fill = value < 0 ? ~std::uint64_t{0} : 0;
    out.limb.fill(fill);
    out.limb[0] = static_cast<std::uint64_t>(value);
    return out;
  }
  static WideInt from_uint64(std::uint64_t value) noexcept {
    WideInt out;
    out.limb[0] = value;
    return out;
  }
  static WideInt load(const std::uint8_t* bytes) noexcept {
    WideInt out;
    std::memcpy(out.limb.data(), bytes, kBytes);
    return out;
  }
  void store(std::uint8_t* bytes) const noexcept { std::memcpy(bytes, limb.data(), kBytes); }

  bool negative() const noexcept { return (limb[N - 1] >> 63) != 0; }
  bool zero() const noexcept {
    for (const std::uint64_t value : limb) {
      if (value != 0) return false;
    }
    return true;
  }
};

using Int256 = WideInt<4>;

template <std::size_t N>
WideInt<N> wrapping_negate(const WideInt<N>& value) noexcept {
  WideInt<N> out;
  std::uint64_t carry = 1;
  for (std::size_t index = 0; index < N; ++index) {
    out.limb[index] = add_with_carry(~value.limb[index], 0, carry);
  }
  return out;
}

// out = a + b; false on signed overflow.
template <std::size_t N>
bool checked_add(const WideInt<N>& a, const WideInt<N>& b, WideInt<N>& out) noexcept {
  std::uint64_t carry = 0;
  WideInt<N> sum;
  for (std::size_t index = 0; index < N; ++index) {
    sum.limb[index] = add_with_carry(a.limb[index], b.limb[index], carry);
  }
  if (a.negative() == b.negative() && sum.negative() != a.negative()) return false;
  out = sum;
  return true;
}

// Number of limbs up to the most significant nonzero one.
template <std::size_t N>
std::size_t used_limbs(const WideInt<N>& value) noexcept {
  std::size_t count = N;
  while (count > 0 && value.limb[count - 1] == 0) --count;
  return count;
}

// out = a * b; false when the exact product is not representable. The single
// value -2^(64N-1) is refused as an operand magnitude, which only narrows the range.
template <std::size_t N>
bool checked_multiply(const WideInt<N>& a, const WideInt<N>& b, WideInt<N>& out) noexcept {
  const bool negative = a.negative() != b.negative();
  const WideInt<N> x = a.negative() ? wrapping_negate(a) : a;
  const WideInt<N> y = b.negative() ? wrapping_negate(b) : b;
  if (x.negative() || y.negative()) return false;
  const std::size_t x_used = used_limbs(x);
  const std::size_t y_used = used_limbs(y);
  if (x_used == 0 || y_used == 0) {
    out = WideInt<N>{};
    return true;
  }
  if (x_used + y_used > N + 1) return false;
  std::array<std::uint64_t, N + 1> product{};
  for (std::size_t i = 0; i < x_used; ++i) {
    std::uint64_t carry = 0;
    for (std::size_t j = 0; j < y_used; ++j) {
      // x_i y_j + product[i + j] + carry <= 2^128 - 1, so the high limb never wraps.
      std::uint64_t high = 0;
      const std::uint64_t low = multiply_wide(x.limb[i], y.limb[j], high);
      std::uint64_t first = 0;
      std::uint64_t second = 0;
      product[i + j] = add_with_carry(add_with_carry(low, product[i + j], first), carry, second);
      carry = high + first + second;
    }
    std::size_t k = i + y_used;
    while (carry != 0) {
      if (k > N) return false;
      std::uint64_t next = 0;
      product[k] = add_with_carry(product[k], carry, next);
      carry = next;
      ++k;
    }
  }
  if (product[N] != 0 || (product[N - 1] >> 63) != 0) return false;
  WideInt<N> magnitude;
  std::memcpy(magnitude.limb.data(), product.data(), 8 * N);
  out = negative ? wrapping_negate(magnitude) : magnitude;
  return true;
}

// out = value * 2^shift; false when not representable.
template <std::size_t N>
bool checked_shift_left(const WideInt<N>& value, int shift, WideInt<N>& out) noexcept {
  if (shift < 0) return false;
  if (shift == 0 || value.zero()) {
    out = value;
    return true;
  }
  if (static_cast<std::size_t>(shift) >= 64 * N - 1) return false;
  const bool negative = value.negative();
  const WideInt<N> magnitude = negative ? wrapping_negate(value) : value;
  if (magnitude.negative()) return false;
  const std::size_t words = static_cast<std::size_t>(shift) / 64;
  const int bits = shift % 64;
  std::array<std::uint64_t, N + 1> shifted{};
  for (std::size_t index = 0; index < N; ++index) {
    if (magnitude.limb[index] == 0) continue;
    const std::size_t target = index + words;
    if (target >= N) return false;
    shifted[target] |= bits == 0 ? magnitude.limb[index] : magnitude.limb[index] << bits;
    if (bits != 0) shifted[target + 1] |= magnitude.limb[index] >> (64 - bits);
  }
  if (shifted[N] != 0 || (shifted[N - 1] >> 63) != 0) return false;
  WideInt<N> result;
  std::memcpy(result.limb.data(), shifted.data(), 8 * N);
  out = negative ? wrapping_negate(result) : result;
  return true;
}

// Signed three-way comparison.
template <std::size_t N>
int compare(const WideInt<N>& a, const WideInt<N>& b) noexcept {
  if (a.negative() != b.negative()) return a.negative() ? -1 : 1;
  for (std::size_t index = N; index-- > 0;) {
    if (a.limb[index] != b.limb[index]) return a.limb[index] < b.limb[index] ? -1 : 1;
  }
  return 0;
}

}  // namespace phx::mc
