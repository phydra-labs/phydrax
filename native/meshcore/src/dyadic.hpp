// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Exact dyadic rationals over unbounded binary integers and exponents. Every
// finite binary64 value is one; sums, differences and products are exact, so
// rational constructions of finite inputs never depend on an exponent window.
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <utility>

#include "bounded_memory.hpp"

namespace phx::mc {

// ------------------------------------------------------------ dyadic integers

using Limbs = NativeVector<std::uint32_t>;

inline void trim(Limbs& limbs) {
  while (!limbs.empty() && limbs.back() == 0) {
    limbs.pop_back();
  }
}

inline int compare_magnitude(const Limbs& a, const Limbs& b) {
  if (a.size() != b.size()) {
    return a.size() < b.size() ? -1 : 1;
  }
  for (std::size_t index = a.size(); index-- > 0;) {
    if (a[index] != b[index]) {
      return a[index] < b[index] ? -1 : 1;
    }
  }
  return 0;
}

inline Limbs add_magnitude(const Limbs& a, const Limbs& b) {
  const Limbs& longer = a.size() >= b.size() ? a : b;
  const Limbs& shorter = a.size() >= b.size() ? b : a;
  Limbs out(longer.size() + 1, 0);
  std::uint64_t carry = 0;
  for (std::size_t index = 0; index < longer.size(); ++index) {
    const std::uint64_t sum = static_cast<std::uint64_t>(longer[index]) +
                              (index < shorter.size() ? shorter[index] : 0u) + carry;
    out[index] = static_cast<std::uint32_t>(sum);
    carry = sum >> 32;
  }
  out[longer.size()] = static_cast<std::uint32_t>(carry);
  trim(out);
  return out;
}

// a - b for a >= b.
inline Limbs subtract_magnitude(const Limbs& a, const Limbs& b) {
  Limbs out(a.size(), 0);
  std::int64_t borrow = 0;
  for (std::size_t index = 0; index < a.size(); ++index) {
    std::int64_t difference = static_cast<std::int64_t>(a[index]) -
                              (index < b.size() ? static_cast<std::int64_t>(b[index]) : 0) -
                              borrow;
    borrow = difference < 0 ? 1 : 0;
    if (difference < 0) {
      difference += std::int64_t{1} << 32;
    }
    out[index] = static_cast<std::uint32_t>(difference);
  }
  trim(out);
  return out;
}

inline Limbs multiply_magnitude(const Limbs& a, const Limbs& b) {
  Limbs out(a.size() + b.size(), 0);
  for (std::size_t row = 0; row < a.size(); ++row) {
    std::uint64_t carry = 0;
    for (std::size_t column = 0; column < b.size(); ++column) {
      const std::uint64_t current = static_cast<std::uint64_t>(out[row + column]) +
                                    static_cast<std::uint64_t>(a[row]) * b[column] + carry;
      out[row + column] = static_cast<std::uint32_t>(current);
      carry = current >> 32;
    }
    out[row + b.size()] = static_cast<std::uint32_t>(carry);
  }
  trim(out);
  return out;
}

inline Limbs shift_left(const Limbs& a, std::int64_t bits) {
  if (a.empty() || bits == 0) {
    return a;
  }
  const std::size_t words = static_cast<std::size_t>(bits / 32);
  const int rest = static_cast<int>(bits % 32);
  Limbs out(a.size() + words + 1, 0);
  for (std::size_t index = 0; index < a.size(); ++index) {
    const std::uint64_t shifted = static_cast<std::uint64_t>(a[index]) << rest;
    out[index + words] |= static_cast<std::uint32_t>(shifted);
    out[index + words + 1] |= static_cast<std::uint32_t>(shifted >> 32);
  }
  trim(out);
  return out;
}

inline std::int64_t bit_length(const Limbs& a) {
  return a.empty() ? 0
                   : static_cast<std::int64_t>(a.size() - 1) * 32 +
                         (32 - std::countl_zero(a.back()));
}

// Bits [start, start + 64) of a, truncating.
inline std::uint64_t bits_from(const Limbs& a, std::int64_t start) {
  std::uint64_t value = 0;
  for (int offset = 0; offset < 64; ++offset) {
    const std::int64_t bit = start + offset;
    const std::size_t word = static_cast<std::size_t>(bit / 32);
    if (word < a.size() && ((a[word] >> (bit % 32)) & 1u)) {
      value |= std::uint64_t{1} << offset;
    }
  }
  return value;
}

// Exact dyadic rational (-1)^negative * magnitude * 2^exponent with an odd (or
// zero) magnitude.  Every finite binary64 value is one, and sums, differences
// and products of dyadic rationals are computed exactly.
class Dyadic {
 public:
  Dyadic() = default;

  explicit Dyadic(double value) {
    if (value == 0.0) {
      return;
    }
    int exponent = 0;
    const double fraction = std::frexp(value, &exponent);
    negative_ = fraction < 0.0;
    std::uint64_t mantissa = static_cast<std::uint64_t>(std::ldexp(std::fabs(fraction), 53));
    const int zeros = std::countr_zero(mantissa);
    mantissa >>= zeros;
    exponent_ = static_cast<std::int64_t>(exponent) - 53 + zeros;
    magnitude_ = {static_cast<std::uint32_t>(mantissa), static_cast<std::uint32_t>(mantissa >> 32)};
    trim(magnitude_);
  }
  static Dyadic integer(const std::uint32_t* words, std::size_t count, int sign) {
    Dyadic value;
    if (count == 0) return value;
    value.magnitude_.assign(words, words + count);
    trim(value.magnitude_);
    value.negative_ = sign < 0;
    value.normalize();
    return value;
  }

  void negate() noexcept {
    if (!magnitude_.empty()) {
      negative_ = !negative_;
    }
  }


  int sign() const {
    if (magnitude_.empty()) {
      return 0;
    }
    return negative_ ? -1 : 1;
  }

  Dyadic operator-() const {
    Dyadic result = *this;
    if (!result.magnitude_.empty()) {
      result.negative_ = !result.negative_;
    }
    return result;
  }

  friend Dyadic operator+(const Dyadic& a, const Dyadic& b) { return combine(a, b, false); }
  friend Dyadic operator-(const Dyadic& a, const Dyadic& b) { return combine(a, b, true); }

  friend Dyadic operator*(const Dyadic& a, const Dyadic& b) {
    Dyadic result;
    if (a.magnitude_.empty() || b.magnitude_.empty()) {
      return result;
    }
    // Products of odd magnitudes are odd: the result stays normalized.
    result.magnitude_ = multiply_magnitude(a.magnitude_, b.magnitude_);
    result.negative_ = a.negative_ != b.negative_;
    result.exponent_ = a.exponent_ + b.exponent_;
    return result;
  }

  // value = fraction * 2^exponent up to a relative error below 2^-52.
  double fraction(std::int64_t& exponent) const {
    if (magnitude_.empty()) {
      exponent = 0;
      return 0.0;
    }
    const std::int64_t length = bit_length(magnitude_);
    const std::int64_t start = std::max<std::int64_t>(0, length - 64);
    exponent = exponent_ + start;
    const double top = static_cast<double>(bits_from(magnitude_, start));
    return negative_ ? -top : top;
  }

  const Limbs& magnitude() const { return magnitude_; }
  std::int64_t exponent() const { return exponent_; }
  Dyadic scaled_power(std::int64_t shift) const {
    Dyadic result = *this;
    result.exponent_ += shift;
    return result;
  }

 private:
  static Dyadic combine(const Dyadic& a, const Dyadic& b, bool negate_b) {
    if (b.magnitude_.empty()) {
      return a;
    }
    const bool b_negative = b.negative_ != negate_b;
    if (a.magnitude_.empty()) {
      Dyadic result = b;
      result.negative_ = b_negative;
      return result;
    }
    const std::int64_t exponent = std::min(a.exponent_, b.exponent_);
    const Limbs left = shift_left(a.magnitude_, a.exponent_ - exponent);
    const Limbs right = shift_left(b.magnitude_, b.exponent_ - exponent);
    Dyadic result;
    result.exponent_ = exponent;
    if (a.negative_ == b_negative) {
      result.magnitude_ = add_magnitude(left, right);
      result.negative_ = a.negative_;
    } else {
      const int order = compare_magnitude(left, right);
      if (order == 0) {
        return Dyadic();
      }
      result.magnitude_ = order > 0 ? subtract_magnitude(left, right)
                                    : subtract_magnitude(right, left);
      result.negative_ = order > 0 ? a.negative_ : b_negative;
    }
    result.normalize();
    return result;
  }

  void normalize() {
    std::size_t words = 0;
    while (words < magnitude_.size() && magnitude_[words] == 0) {
      ++words;
    }
    if (words == magnitude_.size()) {
      magnitude_.clear();
      exponent_ = 0;
      return;
    }
    const int bits = std::countr_zero(magnitude_[words]);
    if (words == 0 && bits == 0) {
      return;
    }
    Limbs out(magnitude_.size() - words, 0);
    for (std::size_t index = words; index < magnitude_.size(); ++index) {
      const std::uint64_t pair =
          static_cast<std::uint64_t>(magnitude_[index]) |
          (index + 1 < magnitude_.size()
               ? static_cast<std::uint64_t>(magnitude_[index + 1]) << 32
               : 0u);
      out[index - words] = static_cast<std::uint32_t>(pair >> bits);
    }
    trim(out);
    magnitude_ = std::move(out);
    exponent_ += static_cast<std::int64_t>(words) * 32 + bits;
  }

  bool negative_ = false;
  Limbs magnitude_;
  std::int64_t exponent_ = 0;
};

}  // namespace phx::mc
