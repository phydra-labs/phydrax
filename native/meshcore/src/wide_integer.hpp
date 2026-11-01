// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Fixed 256-bit two's-complement integers with checked arithmetic. Every
// operation reports overflow instead of wrapping, so a kernel can refuse an
// item whose exact value leaves the representable range.
#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>

namespace phx::mc {

static_assert(std::endian::native == std::endian::little,
              "wide integer words are little-endian byte images");

struct Int256 {
  std::array<std::uint64_t, 4> limb{};  // little-endian limbs, two's complement

  static constexpr std::size_t kBytes = 32;

  static Int256 from_int64(std::int64_t value) noexcept {
    Int256 out;
    const std::uint64_t fill = value < 0 ? ~std::uint64_t{0} : 0;
    out.limb = {static_cast<std::uint64_t>(value), fill, fill, fill};
    return out;
  }
  static Int256 load(const std::uint8_t* bytes) noexcept {
    Int256 out;
    std::memcpy(out.limb.data(), bytes, kBytes);
    return out;
  }
  void store(std::uint8_t* bytes) const noexcept { std::memcpy(bytes, limb.data(), kBytes); }

  bool negative() const noexcept { return (limb[3] >> 63) != 0; }
  bool zero() const noexcept { return (limb[0] | limb[1] | limb[2] | limb[3]) == 0; }
};

inline Int256 wrapping_negate(const Int256& value) noexcept {
  Int256 out;
  unsigned __int128 carry = 1;
  for (int index = 0; index < 4; ++index) {
    const unsigned __int128 sum = static_cast<unsigned __int128>(~value.limb[index]) + carry;
    out.limb[index] = static_cast<std::uint64_t>(sum);
    carry = sum >> 64;
  }
  return out;
}

// out = a + b; false on signed overflow.
inline bool checked_add(const Int256& a, const Int256& b, Int256& out) noexcept {
  unsigned __int128 carry = 0;
  Int256 sum;
  for (int index = 0; index < 4; ++index) {
    const unsigned __int128 total =
        static_cast<unsigned __int128>(a.limb[index]) + b.limb[index] + carry;
    sum.limb[index] = static_cast<std::uint64_t>(total);
    carry = total >> 64;
  }
  if (a.negative() == b.negative() && sum.negative() != a.negative()) return false;
  out = sum;
  return true;
}

// out = a * b; false when the exact product is not representable. The single
// value -2^255 is refused as an operand magnitude, which only narrows the range.
inline bool checked_multiply(const Int256& a, const Int256& b, Int256& out) noexcept {
  const bool negative = a.negative() != b.negative();
  const Int256 x = a.negative() ? wrapping_negate(a) : a;
  const Int256 y = b.negative() ? wrapping_negate(b) : b;
  if (x.negative() || y.negative()) return false;
  std::array<std::uint64_t, 8> product{};
  for (int i = 0; i < 4; ++i) {
    if (x.limb[i] == 0) continue;
    unsigned __int128 carry = 0;
    for (int j = 0; j < 4; ++j) {
      const unsigned __int128 term = static_cast<unsigned __int128>(x.limb[i]) * y.limb[j] +
                                     product[i + j] + carry;
      product[i + j] = static_cast<std::uint64_t>(term);
      carry = term >> 64;
    }
    for (int k = i + 4; carry != 0 && k < 8; ++k) {
      const unsigned __int128 term = static_cast<unsigned __int128>(product[k]) + carry;
      product[k] = static_cast<std::uint64_t>(term);
      carry = term >> 64;
    }
  }
  if ((product[4] | product[5] | product[6] | product[7]) != 0 || (product[3] >> 63) != 0) {
    return false;
  }
  Int256 magnitude;
  magnitude.limb = {product[0], product[1], product[2], product[3]};
  out = negative ? wrapping_negate(magnitude) : magnitude;
  return true;
}

// out = value * 2^shift; false when not representable.
inline bool checked_shift_left(const Int256& value, int shift, Int256& out) noexcept {
  if (shift < 0) return false;
  if (shift == 0 || value.zero()) {
    out = value;
    return true;
  }
  if (shift >= 255) return false;
  const bool negative = value.negative();
  const Int256 magnitude = negative ? wrapping_negate(value) : value;
  if (magnitude.negative()) return false;
  const int words = shift / 64;
  const int bits = shift % 64;
  std::array<std::uint64_t, 5> shifted{};
  for (int index = 0; index < 4; ++index) {
    const int target = index + words;
    if (target < 5) shifted[target] |= bits == 0 ? magnitude.limb[index]
                                                  : magnitude.limb[index] << bits;
    if (bits != 0 && target + 1 < 5) shifted[target + 1] |= magnitude.limb[index] >> (64 - bits);
    if (magnitude.limb[index] != 0 && target >= 4) return false;
  }
  if (shifted[4] != 0 || (shifted[3] >> 63) != 0) return false;
  Int256 result;
  result.limb = {shifted[0], shifted[1], shifted[2], shifted[3]};
  out = negative ? wrapping_negate(result) : result;
  return true;
}

// Signed three-way comparison.
inline int compare(const Int256& a, const Int256& b) noexcept {
  if (a.negative() != b.negative()) return a.negative() ? -1 : 1;
  for (int index = 3; index >= 0; --index) {
    if (a.limb[index] != b.limb[index]) return a.limb[index] < b.limb[index] ? -1 : 1;
  }
  return 0;
}

}  // namespace phx::mc
