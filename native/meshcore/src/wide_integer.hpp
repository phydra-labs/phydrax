// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Fixed-width two's-complement integers of N 64-bit limbs with checked
// arithmetic. Every operation reports overflow instead of wrapping, so a kernel
// can refuse an item whose exact value leaves the representable range.
#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace phx::mc {

static_assert(std::endian::native == std::endian::little,
              "wide integer words are little-endian byte images");

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
  unsigned __int128 carry = 1;
  for (std::size_t index = 0; index < N; ++index) {
    const unsigned __int128 sum = static_cast<unsigned __int128>(~value.limb[index]) + carry;
    out.limb[index] = static_cast<std::uint64_t>(sum);
    carry = sum >> 64;
  }
  return out;
}

// out = a + b; false on signed overflow.
template <std::size_t N>
bool checked_add(const WideInt<N>& a, const WideInt<N>& b, WideInt<N>& out) noexcept {
  unsigned __int128 carry = 0;
  WideInt<N> sum;
  for (std::size_t index = 0; index < N; ++index) {
    const unsigned __int128 total =
        static_cast<unsigned __int128>(a.limb[index]) + b.limb[index] + carry;
    sum.limb[index] = static_cast<std::uint64_t>(total);
    carry = total >> 64;
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
    unsigned __int128 carry = 0;
    for (std::size_t j = 0; j < y_used; ++j) {
      const unsigned __int128 term = static_cast<unsigned __int128>(x.limb[i]) * y.limb[j] +
                                     product[i + j] + carry;
      product[i + j] = static_cast<std::uint64_t>(term);
      carry = term >> 64;
    }
    std::size_t k = i + y_used;
    while (carry != 0) {
      if (k > N) return false;
      const unsigned __int128 term = static_cast<unsigned __int128>(product[k]) + carry;
      product[k] = static_cast<std::uint64_t>(term);
      carry = term >> 64;
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
