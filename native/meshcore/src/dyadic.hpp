// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Exact dyadic rationals over unbounded binary integers and exponents. Every
// finite binary64 value is represented exactly; sums, differences and products
// are exact without an exponent window or an operation-accounting layer.
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace phx::mc {

using Limbs = std::vector<std::uint32_t>;

inline void trim(Limbs& limbs) {
  while (!limbs.empty() && limbs.back() == 0) limbs.pop_back();
}

inline int compare_magnitude(const Limbs& a, const Limbs& b) {
  if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
  for (std::size_t index = a.size(); index-- > 0;) {
    if (a[index] != b[index]) return a[index] < b[index] ? -1 : 1;
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
    if (difference < 0) difference += std::int64_t{1} << 32;
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
  if (a.empty() || bits == 0) return a;
  if (bits < 0) throw std::invalid_argument("A dyadic magnitude shift must be nonnegative.");
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
  const std::size_t first_word = static_cast<std::size_t>(start / 32);
  const unsigned offset = static_cast<unsigned>(start % 32);
  for (std::size_t part = 0; part < 2; ++part) {
    const std::size_t word = first_word + part;
    if (word >= a.size()) break;
    std::uint64_t fragment = static_cast<std::uint64_t>(a[word]) >> offset;
    if (offset != 0 && word + 1 < a.size()) {
      fragment |= static_cast<std::uint64_t>(a[word + 1]) << (32 - offset);
    }
    value |= fragment << (part * 32);
  }
  return value;
}

// Exact dyadic rational (-1)^negative * magnitude * 2^exponent with an odd
// (or zero) magnitude. Arithmetic is exact for all finite binary64 operands.
class Dyadic {
 public:
  Dyadic() = default;

  explicit Dyadic(double value) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("A dyadic scalar must be finite.");
    }
    if (value == 0.0) return;
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
    if (words == nullptr) throw std::invalid_argument("A nonempty dyadic integer needs words.");
    value.magnitude_.assign(words, words + count);
    trim(value.magnitude_);
    value.negative_ = sign < 0;
    value.normalize();
    return value;
  }

  void negate() noexcept {
    if (!magnitude_.empty()) negative_ = !negative_;
  }

  bool is_zero() const noexcept { return magnitude_.empty(); }
  bool is_positive_one() const noexcept {
    return !negative_ && exponent_ == 0 && magnitude_.size() == 1 && magnitude_[0] == 1;
  }
  int sign() const noexcept {
    if (magnitude_.empty()) return 0;
    return negative_ ? -1 : 1;
  }

  Dyadic operator-() const {
    Dyadic result = *this;
    result.negate();
    return result;
  }

  static Dyadic signed_sum(const Dyadic& a, bool negate_a, const Dyadic& b, bool negate_b) {
    return combine_signed(a, negate_a, b, negate_b);
  }

  friend Dyadic operator+(const Dyadic& a, const Dyadic& b) { return combine(a, b, false); }
  friend Dyadic operator-(const Dyadic& a, const Dyadic& b) { return combine(a, b, true); }
  friend Dyadic operator*(const Dyadic& a, const Dyadic& b) {
    Dyadic result;
    if (a.magnitude_.empty() || b.magnitude_.empty()) return result;
    result.magnitude_ = multiply_magnitude(a.magnitude_, b.magnitude_);
    result.negative_ = a.negative_ != b.negative_;
    result.exponent_ = a.exponent_ + b.exponent_;
    return result;
  }

  // value = fraction * 2^exponent to relative precision of the leading 64 bits.
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

  const Limbs& magnitude() const noexcept { return magnitude_; }
  std::int64_t exponent() const noexcept { return exponent_; }

  Dyadic scaled_power(std::int64_t shift) const {
    Dyadic result = *this;
    result.exponent_ += shift;
    return result;
  }

 private:
  static Dyadic combine(const Dyadic& a, const Dyadic& b, bool negate_b) {
    return combine_signed(a, false, b, negate_b);
  }

  static Dyadic combine_signed(const Dyadic& a, bool negate_a, const Dyadic& b, bool negate_b) {
    const bool a_negative = a.negative_ != negate_a;
    const bool b_negative = b.negative_ != negate_b;
    if (b.magnitude_.empty()) {
      if (!negate_a || a.magnitude_.empty()) return a;
      Dyadic result = a;
      result.negate();
      return result;
    }
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
    if (a_negative == b_negative) {
      result.magnitude_ = add_magnitude(left, right);
      result.negative_ = a_negative;
    } else {
      const int order = compare_magnitude(left, right);
      if (order == 0) return Dyadic();
      result.magnitude_ = order > 0 ? subtract_magnitude(left, right) : subtract_magnitude(right, left);
      result.negative_ = order > 0 ? a_negative : b_negative;
    }
    result.normalize();
    return result;
  }

  void normalize() {
    std::size_t words = 0;
    while (words < magnitude_.size() && magnitude_[words] == 0) ++words;
    if (words == magnitude_.size()) {
      magnitude_.clear();
      exponent_ = 0;
      negative_ = false;
      return;
    }
    const int bits = std::countr_zero(magnitude_[words]);
    if (words == 0 && bits == 0) return;
    Limbs out(magnitude_.size() - words, 0);
    for (std::size_t index = words; index < magnitude_.size(); ++index) {
      const std::uint64_t pair = static_cast<std::uint64_t>(magnitude_[index]) |
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
