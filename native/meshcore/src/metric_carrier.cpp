//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact carriers of a goal in a tetrahedral source; see metric_carrier.hpp.
//
// Everything is integer arithmetic over one common denominator D of the
// source points. A binary64 target is m 2^e exactly; with s the largest
// -e of its coordinates (at least 0) the offset of the target from a source
// point N / D is (m 2^(e + s) D - N 2^s) / (D 2^s), an integer numerator.
// Barycentric weights follow by Cramer's rule as integer numerators over an
// integer denominator, so no exact division is needed: the only quotients
// are floors and correctly rounded binary64 quotients of integers.
//
// The search is written once over an integer type and run on inline integers
// of fixed capacity; a query whose values leave that capacity is repeated on
// unbounded integers. Both evaluate the same exact quantities, so the result
// does not depend on which one ran.
#include "metric_carrier.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "exact_scalar.hpp"
#include "wide_integer.hpp"

namespace phx::mc::metric {
namespace {

// A value left the capacity of the inline integers.
struct CapacityExceeded {};

// Sign and magnitude integer of at most kLimbs 64-bit limbs held inline. An
// operation whose result does not fit throws CapacityExceeded; no operation
// rounds or wraps.
class Small {
 public:
  static constexpr std::size_t kLimbs = 16;

  Small() noexcept {}
  Small(const Small& other) noexcept : size_(other.size_), negative_(other.negative_) {
    for (std::size_t index = 0; index < size_; ++index) limb_[index] = other.limb_[index];
  }
  Small& operator=(const Small& other) noexcept {
    size_ = other.size_;
    negative_ = other.negative_;
    for (std::size_t index = 0; index < size_; ++index) limb_[index] = other.limb_[index];
    return *this;
  }

  static Small from_uint64(std::uint64_t value) noexcept {
    Small out;
    if (value != 0) {
      out.limb_[0] = value;
      out.size_ = 1;
    }
    return out;
  }
  // The magnitude `limbs` (little-endian, `count` of them) with the sign.
  static Small from_limbs(const std::uint64_t* limbs, std::size_t count, bool negative) {
    while (count > 0 && limbs[count - 1] == 0) --count;
    if (count > kLimbs) throw CapacityExceeded{};
    Small out;
    for (std::size_t index = 0; index < count; ++index) out.limb_[index] = limbs[index];
    out.size_ = count;
    out.negative_ = negative && count != 0;
    return out;
  }

  int sign() const noexcept { return size_ == 0 ? 0 : (negative_ ? -1 : 1); }
  Small negated() const noexcept {
    Small out = *this;
    if (size_ != 0) out.negative_ = !negative_;
    return out;
  }
  std::int64_t bits() const noexcept {
    if (size_ == 0) return 0;
    return static_cast<std::int64_t>(64 * size_) - std::countl_zero(limb_[size_ - 1]);
  }
  // |value| = leading * 2^exponent to the relative precision of its leading
  // 64 bits.
  double leading(std::int64_t& exponent) const noexcept {
    const std::int64_t length = bits();
    if (length <= 64) {
      exponent = 0;
      return size_ == 0 ? 0.0 : static_cast<double>(limb_[0]);
    }
    const std::int64_t start = length - 64;
    const std::size_t word = static_cast<std::size_t>(start / 64);
    const int offset = static_cast<int>(start % 64);
    std::uint64_t top = limb_[word] >> offset;
    if (offset != 0) top |= limb_[word + 1] << (64 - offset);
    exponent = start;
    return static_cast<double>(top);
  }
  // value * 2^count for count >= 0.
  Small shifted(std::int64_t count) const {
    if (size_ == 0 || count == 0) return *this;
    const std::size_t words = static_cast<std::size_t>(count / 64);
    const int rest = static_cast<int>(count % 64);
    if (words + size_ > kLimbs) throw CapacityExceeded{};
    Small out;
    for (std::size_t index = 0; index < words; ++index) out.limb_[index] = 0;
    std::uint64_t carry = 0;
    for (std::size_t index = 0; index < size_; ++index) {
      out.limb_[index + words] = rest == 0 ? limb_[index] : (limb_[index] << rest) | carry;
      carry = rest == 0 ? 0 : limb_[index] >> (64 - rest);
    }
    out.size_ = size_ + words;
    if (carry != 0) {
      if (out.size_ == kLimbs) throw CapacityExceeded{};
      out.limb_[out.size_] = carry;
      ++out.size_;
    }
    out.negative_ = negative_;
    return out;
  }

  friend Small operator+(const Small& a, const Small& b) { return combine(a, b, b.negative_); }
  friend Small operator-(const Small& a, const Small& b) { return combine(a, b, !b.negative_); }
  friend Small operator*(const Small& a, const Small& b) {
    Small out;
    if (a.size_ == 0 || b.size_ == 0) return out;
    if (a.size_ + b.size_ > kLimbs) throw CapacityExceeded{};
    const std::size_t size = a.size_ + b.size_;
    for (std::size_t index = 0; index < size; ++index) out.limb_[index] = 0;
    for (std::size_t row = 0; row < a.size_; ++row) {
      std::uint64_t carry = 0;
      for (std::size_t column = 0; column < b.size_; ++column) {
        out.limb_[row + column] = multiply_accumulate(a.limb_[row], b.limb_[column],
                                                      out.limb_[row + column], carry, carry);
      }
      out.limb_[row + b.size_] = carry;
    }
    out.size_ = out.limb_[size - 1] == 0 ? size - 1 : size;
    out.negative_ = a.negative_ != b.negative_;
    return out;
  }
  friend int compare(const Small& a, const Small& b) noexcept {
    if (a.sign() != b.sign()) return a.sign() < b.sign() ? -1 : 1;
    const int order = compare_magnitude(a, b);
    return a.negative_ ? -order : order;
  }

 private:
  static int compare_magnitude(const Small& a, const Small& b) noexcept {
    if (a.size_ != b.size_) return a.size_ < b.size_ ? -1 : 1;
    for (std::size_t index = a.size_; index-- > 0;) {
      if (a.limb_[index] != b.limb_[index]) return a.limb_[index] < b.limb_[index] ? -1 : 1;
    }
    return 0;
  }
  // a + (b with the sign `b_negative`).
  static Small combine(const Small& a, const Small& b, bool b_negative) {
    if (b.size_ == 0) return a;
    if (a.size_ == 0) {
      Small out = b;
      out.negative_ = b_negative;
      return out;
    }
    Small out;
    if (a.negative_ == b_negative) {
      const Small& longer = a.size_ >= b.size_ ? a : b;
      const Small& shorter = a.size_ >= b.size_ ? b : a;
      std::uint64_t carry = 0;
      for (std::size_t index = 0; index < longer.size_; ++index) {
        out.limb_[index] = add_with_carry(
            longer.limb_[index], index < shorter.size_ ? shorter.limb_[index] : 0, carry);
      }
      out.size_ = longer.size_;
      if (carry != 0) {
        if (out.size_ == kLimbs) throw CapacityExceeded{};
        out.limb_[out.size_] = carry;
        ++out.size_;
      }
      out.negative_ = a.negative_;
      return out;
    }
    const int order = compare_magnitude(a, b);
    if (order == 0) return out;
    const Small& larger = order > 0 ? a : b;
    const Small& smaller = order > 0 ? b : a;
    std::uint64_t borrow = 0;
    for (std::size_t index = 0; index < larger.size_; ++index) {
      const std::uint64_t subtrahend = index < smaller.size_ ? smaller.limb_[index] : 0;
      const std::uint64_t difference = larger.limb_[index] - subtrahend - borrow;
      borrow = larger.limb_[index] < subtrahend ||
                       (larger.limb_[index] == subtrahend && borrow != 0)
                   ? 1
                   : 0;
      out.limb_[index] = difference;
    }
    out.size_ = larger.size_;
    while (out.size_ > 0 && out.limb_[out.size_ - 1] == 0) --out.size_;
    out.negative_ = order > 0 ? a.negative_ : b_negative;
    return out;
  }

  // Limbs at and beyond size_ are never read.
  std::uint64_t limb_[kLimbs];
  std::size_t size_ = 0;
  bool negative_ = false;
};

// The operations the search needs of its integer type.
template <class Z>
struct Integer;

template <>
struct Integer<Small> {
  static Small of(std::uint64_t value) { return Small::from_uint64(value); }
  static Small shifted(const Small& value, std::int64_t count) { return value.shifted(count); }
  static Small negated(const Small& value) { return value.negated(); }
  static std::int64_t bits(const Small& value) { return value.bits(); }
  static double leading(const Small& value, std::int64_t& exponent) {
    return value.leading(exponent);
  }
  static int order(const Small& a, const Small& b) { return compare(a, b); }
  // A quotient outside the normal binary64 range is left to the unbounded route.
  [[noreturn]] static double quotient(const Small&, const Small&, std::int64_t, bool&) {
    throw CapacityExceeded{};
  }
};

template <>
struct Integer<Dyadic> {
  static Dyadic of(std::uint64_t value) {
    const std::uint32_t words[2] = {static_cast<std::uint32_t>(value),
                                    static_cast<std::uint32_t>(value >> 32)};
    return Dyadic::integer(words, 2, 1);
  }
  static Dyadic shifted(const Dyadic& value, std::int64_t count) {
    return value.scaled_power(count);
  }
  static Dyadic negated(const Dyadic& value) { return -value; }
  static std::int64_t bits(const Dyadic& value) {
    return value.is_zero() ? 0 : bit_length(value.magnitude()) + value.exponent();
  }
  static double leading(const Dyadic& value, std::int64_t& exponent) {
    return std::fabs(value.fraction(exponent));
  }
  static int order(const Dyadic& a, const Dyadic& b) { return (a - b).sign(); }
  // value / (denominator 2^extra) correctly rounded, whatever its range.
  static double quotient(const Dyadic& value, const Dyadic& denominator, std::int64_t extra,
                         bool& exact) {
    const Dyadic divisor = denominator.scaled_power(extra);
    const double rounded = exact::rounded_quotient(value, divisor);
    exact = (value - Dyadic(rounded) * divisor).sign() == 0;
    return rounded;
  }
};

// Correction steps a quotient may take. The estimate from the leading bits is
// within 2^-50 of a quotient below 2^62, so within about 2^12; the first step,
// estimated the same way from the remainder, leaves less than 2, and unit
// steps finish: four at most, and never more than two on the carriers of the
// tests. Twice the bound is allowed. More means the integer arithmetic under
// the search is wrong, which must end the query and not spin.
constexpr int kQuotientCorrections = 8;

// floor(numerator 2^shift / denominator) for numerator >= 0, denominator > 0
// and a quotient below 2^62; `exact` reports a zero remainder. The quotient is
// estimated from the leading bits and corrected with exact remainders; a
// quotient that does not settle in kQuotientCorrections steps throws.
template <class Z>
std::uint64_t floor_quotient(const Z& numerator, std::int64_t shift, const Z& denominator,
                             bool& exact) {
  if (numerator.sign() == 0) {
    exact = true;
    return 0;
  }
  const Z scaled = shift > 0 ? Integer<Z>::shifted(numerator, shift) : numerator;
  const Z divisor = shift < 0 ? Integer<Z>::shifted(denominator, -shift) : denominator;
  std::int64_t divisor_exponent = 0;
  const double divisor_leading = Integer<Z>::leading(divisor, divisor_exponent);
  const auto ratio = [&](const Z& value) {
    std::int64_t exponent = 0;
    const double leading = Integer<Z>::leading(value, exponent);
    const std::int64_t scale = exponent - divisor_exponent;
    if (scale > 80) return std::numeric_limits<double>::infinity();
    return std::ldexp(leading / divisor_leading, static_cast<int>(scale < -200 ? -200 : scale));
  };
  const double estimate = ratio(scaled);
  if (!(estimate < 0x1p62)) throw std::logic_error("A carrier quotient is out of range.");
  std::uint64_t quotient = static_cast<std::uint64_t>(estimate);
  Z remainder = scaled - Integer<Z>::of(quotient) * divisor;
  for (int corrections = 0;;) {
    const bool below = remainder.sign() < 0;
    if (!below && Integer<Z>::order(remainder, divisor) < 0) break;
    if (++corrections > kQuotientCorrections) {
      throw std::logic_error("A carrier quotient did not converge.");
    }
    if (below) {
      const double behind = ratio(remainder);
      std::uint64_t step = behind >= 1.0 ? static_cast<std::uint64_t>(behind) : 1;
      if (quotient == 0) throw std::logic_error("A carrier quotient is negative.");
      if (step > quotient) step = quotient;
      quotient -= step;
      remainder = remainder + Integer<Z>::of(step) * divisor;
    } else {
      const double ahead = ratio(remainder);
      const std::uint64_t step = ahead >= 1.0 && ahead < 0x1p62 ? static_cast<std::uint64_t>(ahead) : 1;
      quotient += step;
      remainder = remainder - Integer<Z>::of(step) * divisor;
    }
  }
  exact = remainder.sign() == 0;
  return quotient;
}

// The binary64 value nearest value / (denominator 2^extra), ties to even, for
// denominator > 0 and extra >= 0; `exact` reports that it is the quotient.
template <class Z>
double rounded_quotient(const Z& value, const Z& denominator, std::int64_t extra, bool& exact) {
  const int sign = value.sign();
  if (sign == 0) {
    exact = true;
    return 0.0;
  }
  const Z magnitude = sign < 0 ? Integer<Z>::negated(value) : value;
  // magnitude / denominator lies in (2^(d - 1), 2^(d + 1)) for the difference
  // d of their bit lengths: the floor below has 55 or 56 bits.
  const std::int64_t shift = 55 - (Integer<Z>::bits(magnitude) - Integer<Z>::bits(denominator));
  bool whole = false;
  const std::uint64_t quotient = floor_quotient(magnitude, shift, denominator, whole);
  const int drop = (64 - std::countl_zero(quotient)) - 53;
  const std::int64_t exponent = drop - shift - extra;
  // Rounding to 53 bits is the binary64 rounding only in the normal range.
  if (drop < 1 || exponent < -1074 || exponent > 970) {
    return Integer<Z>::quotient(value, denominator, extra, exact);
  }
  std::uint64_t mantissa = quotient >> drop;
  const std::uint64_t rest = quotient & ((std::uint64_t{1} << drop) - 1);
  const std::uint64_t half = std::uint64_t{1} << (drop - 1);
  if (rest > half || (rest == half && (!whole || (mantissa & 1) != 0))) ++mantissa;
  exact = whole && rest == 0;
  const double rounded = std::ldexp(static_cast<double>(mantissa), static_cast<int>(exponent));
  return sign < 0 ? -rounded : rounded;
}

template <class Z>
using Vector = std::array<Z, 3>;

template <class Z>
Z dot(const Vector<Z>& a, const Vector<Z>& b) {
  return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2];
}

template <class Z>
Vector<Z> cross(const Vector<Z>& a, const Vector<Z>& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

// value = mantissa 2^exponent with an odd mantissa; both 0 for zero.
void decompose(double value, std::int64_t& mantissa, std::int64_t& exponent) noexcept {
  mantissa = 0;
  exponent = 0;
  if (value == 0.0) return;
  int binary = 0;
  const double fraction = std::frexp(value, &binary);
  const std::int64_t scaled = static_cast<std::int64_t>(std::ldexp(fraction, 53));
  const std::uint64_t size = static_cast<std::uint64_t>(scaled < 0 ? -scaled : scaled);
  const int zeros = std::countr_zero(size);
  mantissa = static_cast<std::int64_t>(size >> zeros);
  if (scaled < 0) mantissa = -mantissa;
  exponent = static_cast<std::int64_t>(binary) - 53 + zeros;
}

// The exact source as seen by the search: numerators over one denominator.
struct ExactView {
  using Value = Dyadic;
  const std::vector<Dyadic>* numerators;
  const Dyadic* common;
  const Dyadic& numerator(std::int64_t row, int axis) const {
    return (*numerators)[static_cast<std::size_t>(3 * row + axis)];
  }
  const Dyadic& denominator() const { return *common; }
};

// The same numbers as inline integers, read from their limb bank.
struct InlineView {
  using Value = Small;
  const std::vector<std::uint64_t>* limbs;
  const std::vector<std::uint32_t>* offsets;
  const std::vector<std::uint8_t>* negative;
  Small common;
  Small at(std::size_t index) const {
    const std::uint32_t begin = (*offsets)[index];
    return Small::from_limbs(limbs->data() + begin, (*offsets)[index + 1] - begin,
                             (*negative)[index] != 0);
  }
  Small numerator(std::int64_t row, int axis) const {
    return at(static_cast<std::size_t>(3 * row + axis));
  }
  const Small& denominator() const { return common; }
};

// A located goal: weights[i] / (denominator 2^shift) at point rows[i].
template <class Z>
struct Located {
  std::array<std::int64_t, 4> rows{};
  std::array<Z, 4> weights;
  Z denominator;
  std::int64_t shift = 0;
  int size = 0;
  bool declared = false;
};

struct Geometry {
  const std::vector<double>& points;
  const std::vector<std::int64_t>& cells;
  const std::vector<double>& lower;
  const std::vector<double>& upper;
  const std::array<double, 3>& grid_origin;
  const std::array<double, 3>& grid_scale;
  const std::array<std::int64_t, 3>& grid_size;
  const std::vector<std::int64_t>& grid_offsets;
  const std::vector<std::int64_t>& grid_cells;
};

// The grid bucket coordinate of `value` on `axis`: monotone in value, so a
// box [lower, upper] meets every bucket between those of its corners.
std::int64_t bucket_coordinate(const std::array<double, 3>& origin,
                               const std::array<double, 3>& scale,
                               const std::array<std::int64_t, 3>& size, int axis,
                               double value) {
  const auto at = static_cast<std::size_t>(axis);
  const double position = std::floor((value - origin[at]) * scale[at]);
  if (!(position > 0.0)) return 0;
  const auto last = static_cast<double>(size[at] - 1);
  return position >= last ? size[at] - 1 : static_cast<std::int64_t>(position);
}

// The target as integer numerators over (denominator 2^shift).
template <class View>
struct Target {
  using Z = typename View::Value;
  Vector<Z> numerators;
  std::int64_t shift = 0;

  Target(const View& view, const double* target) {
    std::int64_t mantissa[3];
    std::int64_t exponent[3];
    for (int axis = 0; axis < 3; ++axis) {
      decompose(target[axis], mantissa[axis], exponent[axis]);
      if (mantissa[axis] != 0 && -exponent[axis] > shift) shift = -exponent[axis];
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const std::int64_t value = mantissa[axis];
      Z scaled = Integer<Z>::of(static_cast<std::uint64_t>(value < 0 ? -value : value));
      if (value != 0) scaled = Integer<Z>::shifted(scaled, exponent[axis] + shift);
      if (value < 0) scaled = Integer<Z>::negated(scaled);
      numerators[axis] = scaled * view.denominator();
    }
  }
};

template <class View>
bool locate_cell(const View& view, const Geometry& geometry, const double* target,
                 const Target<View>& exact_target, std::int64_t only,
                 Located<typename View::Value>& located) {
  using Z = typename View::Value;
  // The cells to try, ascending: `only`, or the target's grid bucket (every
  // cell whose box can hold the target is listed there, so the first holding
  // cell is the one a scan of all cells would find).
  const std::int64_t* candidates = nullptr;
  std::size_t candidate_count = 0;
  const std::int64_t single = only;
  if (only >= 0) {
    candidates = &single;
    candidate_count = 1;
  } else {
    for (int axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(target[axis])) return false;
    }
    std::int64_t bucket = 0;
    for (int axis = 2; axis >= 0; --axis) {
      bucket = bucket * geometry.grid_size[static_cast<std::size_t>(axis)] +
               bucket_coordinate(geometry.grid_origin, geometry.grid_scale, geometry.grid_size,
                                 axis, target[axis]);
    }
    const auto row = static_cast<std::size_t>(bucket);
    const auto begin = static_cast<std::size_t>(geometry.grid_offsets[row]);
    const auto end = static_cast<std::size_t>(geometry.grid_offsets[row + 1]);
    candidates = geometry.grid_cells.data() + begin;
    candidate_count = end - begin;
  }
  for (std::size_t index = 0; index < candidate_count; ++index) {
    const auto cell = static_cast<std::size_t>(candidates[index]);
    // Rounding is monotone and the box corners are binary64 source points, so
    // a cell that holds the exact target passes this comparison.
    bool inside = true;
    for (std::size_t axis = 0; axis < 3 && inside; ++axis) {
      inside = geometry.lower[3 * cell + axis] <= target[axis] &&
               target[axis] <= geometry.upper[3 * cell + axis];
    }
    if (!inside) continue;
    const std::int64_t* rows = geometry.cells.data() + 4 * cell;
    Vector<Z> edges[3];
    Vector<Z> offset;
    for (int axis = 0; axis < 3; ++axis) {
      const std::size_t slot = static_cast<std::size_t>(axis);
      const Z origin = view.numerator(rows[0], axis);
      for (int corner = 0; corner < 3; ++corner) {
        edges[corner][slot] = view.numerator(rows[corner + 1], axis) - origin;
      }
      offset[slot] = exact_target.numerators[slot] - Integer<Z>::shifted(origin, exact_target.shift);
    }
    const Vector<Z> normal = cross(edges[1], edges[2]);
    Z volume = dot(edges[0], normal);
    if (volume.sign() == 0) continue;
    Z weights[4];
    weights[1] = dot(offset, normal);
    weights[2] = dot(offset, cross(edges[2], edges[0]));
    weights[3] = dot(offset, cross(edges[0], edges[1]));
    if (volume.sign() < 0) {
      volume = Integer<Z>::negated(volume);
      for (int corner = 1; corner < 4; ++corner) weights[corner] = Integer<Z>::negated(weights[corner]);
    }
    weights[0] = ((Integer<Z>::shifted(volume, exact_target.shift) - weights[1]) - weights[2]) -
                 weights[3];
    bool holds = true;
    for (const Z& weight : weights) holds = holds && weight.sign() >= 0;
    if (!holds) continue;
    located.size = 4;
    located.declared = false;
    located.shift = exact_target.shift;
    located.denominator = volume;
    for (std::size_t corner = 0; corner < 4; ++corner) {
      located.rows[corner] = rows[corner];
      located.weights[corner] = weights[corner];
    }
    return true;
  }
  return false;
}

template <class View>
bool locate_simplex(const View& view, const Target<View>& exact_target,
                    const CarrierSimplex& simplex, Located<typename View::Value>& located) {
  using Z = typename View::Value;
  Vector<Z> edges[2];
  Vector<Z> offset;
  for (int axis = 0; axis < 3; ++axis) {
    const std::size_t slot = static_cast<std::size_t>(axis);
    const Z origin = view.numerator(simplex.rows[0], axis);
    for (int corner = 1; corner < simplex.size; ++corner) {
      edges[corner - 1][slot] =
          view.numerator(simplex.rows[static_cast<std::size_t>(corner)], axis) - origin;
    }
    offset[slot] = exact_target.numerators[slot] - Integer<Z>::shifted(origin, exact_target.shift);
  }
  Z weights[3];
  Z total;
  if (simplex.size == 2) {
    total = dot(edges[0], edges[0]);
    if (total.sign() == 0) return false;
    const Z whole = Integer<Z>::shifted(total, exact_target.shift);
    weights[1] = dot(edges[0], offset);
    if (weights[1].sign() < 0 || Integer<Z>::order(weights[1], whole) > 0) return false;
    weights[0] = whole - weights[1];
  } else {
    const Z g00 = dot(edges[0], edges[0]);
    const Z g01 = dot(edges[0], edges[1]);
    const Z g11 = dot(edges[1], edges[1]);
    total = g00 * g11 - g01 * g01;
    if (total.sign() == 0) return false;
    const Z whole = Integer<Z>::shifted(total, exact_target.shift);
    const Z r0 = dot(edges[0], offset);
    const Z r1 = dot(edges[1], offset);
    weights[1] = r0 * g11 - r1 * g01;
    if (weights[1].sign() < 0 || Integer<Z>::order(weights[1], whole) > 0) return false;
    const Z remaining = whole - weights[1];
    weights[2] = r1 * g00 - r0 * g01;
    if (weights[2].sign() < 0 || Integer<Z>::order(weights[2], remaining) > 0) return false;
    weights[0] = remaining - weights[2];
  }
  located.size = simplex.size;
  located.declared = simplex.declared;
  located.shift = exact_target.shift;
  located.denominator = total;
  for (int corner = 0; corner < simplex.size; ++corner) {
    const std::size_t slot = static_cast<std::size_t>(corner);
    located.rows[slot] = simplex.rows[slot];
    located.weights[slot] = weights[corner];
  }
  return true;
}

bool within_projection_box(const Geometry& geometry, const CarrierSimplex& simplex,
                           const double* target) {
  for (int axis = 0; axis < 3; ++axis) {
    if ((simplex.projected_axes & (1U << axis)) == 0) continue;
    const double first = geometry.points[3 * simplex.rows[0] + axis];
    const double second = geometry.points[3 * simplex.rows[1] + axis];
    const double third = simplex.size == 3 ? geometry.points[3 * simplex.rows[2] + axis] : first;
    // The projected coordinate is unchanged. Monotone rounding implies
    // that a binary64 target in the exact box is also in the rounded box.
    if (target[axis] < std::min({first, second, third}) ||
        target[axis] > std::max({first, second, third})) return false;
  }
  return true;
}

template <class View>
std::vector<Carrier> search(const View& view, const Geometry& geometry, const double* target,
                            const double* current, const std::vector<CarrierSimplex>& simplices,
                            std::int64_t only, CarrierOrder order, std::int64_t limit) {
  using Z = typename View::Value;
  std::vector<Carrier> found;
  if (limit <= 0) return found;
  const Target<View> exact_target(view, target);
  Located<Z> located;
  const bool on_simplex = !simplices.empty();
  bool has_location = false;
  if (on_simplex) {
    for (const CarrierSimplex& simplex : simplices) {
      if (simplex.size != 2 && simplex.size != 3) continue;
      if (simplex.projected_axes != 0 && !within_projection_box(geometry, simplex, target)) {
        continue;
      }
      if (locate_simplex(view, exact_target, simplex, located)) {
        has_location = true;
        break;
      }
    }
  } else {
    has_location = locate_cell(view, geometry, target, exact_target, only, located);
  }
  if (!has_location) return found;

  // The distinct support points in located order, and the support slot of
  // every located vertex (a vertex named twice shares its slot).
  std::array<std::int64_t, 4> support{};
  std::array<int, 4> column{};
  int support_size = 0;
  for (int vertex = 0; vertex < located.size; ++vertex) {
    int slot = 0;
    while (slot < support_size && support[static_cast<std::size_t>(slot)] !=
                                      located.rows[static_cast<std::size_t>(vertex)]) {
      ++slot;
    }
    if (slot == support_size) {
      support[static_cast<std::size_t>(support_size)] = located.rows[static_cast<std::size_t>(vertex)];
      ++support_size;
    }
    column[static_cast<std::size_t>(vertex)] = slot;
  }
  // The numerators of the support points, read once.
  Z numerators[4][3];
  for (int slot = 0; slot < support_size; ++slot) {
    for (int axis = 0; axis < 3; ++axis) {
      numerators[slot][axis] = view.numerator(support[static_cast<std::size_t>(slot)], axis);
    }
  }

  // The goal: the target in a cell, the correctly rounded projection on a simplex.
  double goal[3] = {target[0], target[1], target[2]};
  if (on_simplex) {
    const Z image_denominator = located.denominator * view.denominator();
    for (int axis = 0; axis < 3; ++axis) {
      Z sum;
      for (int vertex = 0; vertex < located.size; ++vertex) {
        const std::size_t slot = static_cast<std::size_t>(vertex);
        sum = sum + located.weights[slot] * numerators[column[slot]][axis];
      }
      bool unused = false;
      goal[axis] = rounded_quotient(sum, image_denominator, located.shift, unused);
    }
  }
  const double moved[3] = {goal[0] - current[0], goal[1] - current[1], goal[2] - current[2]};
  const double distance = (moved[0] * moved[0] + moved[1] * moved[1]) + moved[2] * moved[2];
  double span = 0.0;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    double low = 0.0;
    double high = 0.0;
    for (int slot = 0; slot < support_size; ++slot) {
      const double value = geometry.points[3 * static_cast<std::size_t>(
                                                   support[static_cast<std::size_t>(slot)]) +
                                           axis];
      if (slot == 0 || value < low) low = value;
      if (slot == 0 || value > high) high = value;
    }
    if (high - low > span) span = high - low;
  }
  if (!(distance > 0.0) || !(span > 0.0)) return found;

  // A grid coarser than the move cannot resolve it: start at the binary
  // exponent of span / |move|, which is floor(log2) exactly.
  const double ratio = span / std::sqrt(distance);
  int exponent = 0;
  if (ratio <= std::numeric_limits<double>::max()) {
    std::frexp(ratio, &exponent);
  }
  const int first = exponent - 1 < 1 ? 1 : (exponent - 1 > 53 ? 53 : exponent - 1);

  // The partial sums of the weights in located order, all but the last, as
  // floor(sum 2^54) with whether the floor is the value: every coarser
  // rounding follows from these 55 bits and that flag.
  std::array<std::uint64_t, 3> partial{};
  std::array<bool, 3> partial_exact{};
  {
    Z running;
    for (int vertex = 0; vertex + 1 < located.size; ++vertex) {
      const std::size_t slot = static_cast<std::size_t>(vertex);
      running = vertex == 0 ? located.weights[0] : running + located.weights[slot];
      bool whole = false;
      partial[slot] = floor_quotient(running, 54 - located.shift, located.denominator, whole);
      partial_exact[slot] = whole;
    }
  }

  const auto try_resolution = [&](int bits) {
    const std::int64_t scale = std::int64_t{1} << bits;
    std::array<std::int64_t, 4> merged{};
    std::int64_t previous = 0;
    const int drop = 54 - bits;
    for (int vertex = 0; vertex + 1 < located.size; ++vertex) {
      const std::size_t slot = static_cast<std::size_t>(vertex);
      // The partial sum times 2^bits, rounded half to even.
      std::int64_t quotient = static_cast<std::int64_t>(partial[slot] >> drop);
      const std::uint64_t rest = partial[slot] & ((std::uint64_t{1} << drop) - 1);
      const std::uint64_t half = std::uint64_t{1} << (drop - 1);
      if (rest > half || (rest == half && (!partial_exact[slot] || (quotient & 1) != 0))) {
        ++quotient;
      }
      if (quotient < previous) quotient = previous;
      if (quotient > scale) quotient = scale;
      merged[static_cast<std::size_t>(column[slot])] += quotient - previous;
      previous = quotient;
    }
    merged[static_cast<std::size_t>(column[static_cast<std::size_t>(located.size - 1)])] +=
        scale - previous;
    Carrier carrier;
    carrier.bits = bits;
    carrier.exact = true;
    for (int axis = 0; axis < 3; ++axis) {
      Z value;
      for (int slot = 0; slot < support_size; ++slot) {
        const std::size_t index = static_cast<std::size_t>(slot);
        if (merged[index] == 0) continue;
        value = value + Integer<Z>::of(static_cast<std::uint64_t>(merged[index])) *
                            numerators[slot][axis];
      }
      bool whole = false;
      carrier.point[static_cast<std::size_t>(axis)] =
          rounded_quotient(value, view.denominator(), bits, whole);
      if (!whole) carrier.exact = false;
    }
    const double dx = carrier.point[0] - goal[0];
    const double dy = carrier.point[1] - goal[1];
    const double dz = carrier.point[2] - goal[2];
    if (!((dx * dx + dy * dy) + dz * dz < distance)) return false;
    if (on_simplex && !carrier.exact && !located.declared) return false;
    // The support in ascending row order: one stencil has one spelling.
    for (int slot = 0; slot < support_size; ++slot) {
      const std::size_t index = static_cast<std::size_t>(slot);
      if (merged[index] == 0) continue;
      std::size_t place = static_cast<std::size_t>(carrier.size);
      while (place > 0 && carrier.rows[place - 1] > support[index]) {
        carrier.rows[place] = carrier.rows[place - 1];
        carrier.weights[place] = carrier.weights[place - 1];
        --place;
      }
      carrier.rows[place] = support[index];
      carrier.weights[place] = merged[index];
      ++carrier.size;
    }
    found.push_back(carrier);
    return true;
  };

  const auto full = [&]() { return static_cast<std::int64_t>(found.size()) >= limit; };
  if (order == CarrierOrder::kFineFirst) {
    for (int bits = 53; bits >= first && !full(); --bits) try_resolution(bits);
    return found;
  }
  for (int bits = first; bits <= 53 && !full(); ++bits) {
    if (try_resolution(bits) && order == CarrierOrder::kCoarseThenFine) {
      for (int fine = 53; fine >= first && !full(); --fine) try_resolution(fine);
      return found;
    }
  }
  return found;
}

}  // namespace

CarrierSource::CarrierSource(std::vector<double> points, std::vector<Dyadic> numerators,
                             Dyadic denominator, std::vector<std::int64_t> cells)
    : points_(std::move(points)),
      numerators_(std::move(numerators)),
      denominator_(std::move(denominator)),
      cells_(std::move(cells)) {
  if (points_.size() % 3 != 0 || numerators_.size() != points_.size() ||
      cells_.size() % 4 != 0 || denominator_.sign() <= 0) {
    throw std::invalid_argument("An exact carrier source needs aligned points and cells.");
  }
  const std::int64_t count = point_count();
  const std::size_t cell_count = cells_.size() / 4;
  lower_.resize(3 * cell_count);
  upper_.resize(3 * cell_count);
  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    for (std::size_t corner = 0; corner < 4; ++corner) {
      const std::int64_t row = cells_[4 * cell + corner];
      if (row < 0 || row >= count) {
        throw std::invalid_argument("An exact carrier source cell names an absent point.");
      }
      for (std::size_t axis = 0; axis < 3; ++axis) {
        const double value = points_[3 * static_cast<std::size_t>(row) + axis];
        if (corner == 0 || value < lower_[3 * cell + axis]) lower_[3 * cell + axis] = value;
        if (corner == 0 || value > upper_[3 * cell + axis]) upper_[3 * cell + axis] = value;
      }
    }
  }

  // The bucket grid: about one cell per bucket over the boxes' hull.
  std::array<double, 3> low{};
  std::array<double, 3> high{};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    low[axis] = std::numeric_limits<double>::infinity();
    high[axis] = -std::numeric_limits<double>::infinity();
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
      low[axis] = std::min(low[axis], lower_[3 * cell + axis]);
      high[axis] = std::max(high[axis], upper_[3 * cell + axis]);
    }
  }
  const auto side = static_cast<std::int64_t>(
      std::ceil(std::cbrt(static_cast<double>(std::max<std::size_t>(cell_count, 1)))));
  std::int64_t buckets = 1;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const double extent = high[axis] - low[axis];
    grid_size_[axis] = cell_count == 0 || !(extent > 0.0) ? 1 : side;
    grid_origin_[axis] = cell_count == 0 ? 0.0 : low[axis];
    grid_scale_[axis] = grid_size_[axis] == 1 ? 0.0 : static_cast<double>(side) / extent;
    buckets *= grid_size_[axis];
  }
  grid_offsets_.assign(static_cast<std::size_t>(buckets) + 1, 0);
  const auto for_each_bucket = [&](std::size_t cell, auto&& visit) {
    std::array<std::int64_t, 3> first{};
    std::array<std::int64_t, 3> last{};
    for (int axis = 0; axis < 3; ++axis) {
      const auto at = static_cast<std::size_t>(axis);
      first[at] = bucket_coordinate(grid_origin_, grid_scale_, grid_size_, axis,
                                    lower_[3 * cell + at]);
      last[at] = bucket_coordinate(grid_origin_, grid_scale_, grid_size_, axis,
                                   upper_[3 * cell + at]);
    }
    for (std::int64_t z = first[2]; z <= last[2]; ++z) {
      for (std::int64_t y = first[1]; y <= last[1]; ++y) {
        for (std::int64_t x = first[0]; x <= last[0]; ++x) {
          visit(static_cast<std::size_t>((z * grid_size_[1] + y) * grid_size_[0] + x));
        }
      }
    }
  };
  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    for_each_bucket(cell, [&](std::size_t bucket) { ++grid_offsets_[bucket + 1]; });
  }
  for (std::size_t bucket = 0; bucket < static_cast<std::size_t>(buckets); ++bucket) {
    grid_offsets_[bucket + 1] += grid_offsets_[bucket];
  }
  grid_cells_.resize(static_cast<std::size_t>(grid_offsets_.back()));
  std::vector<std::int64_t> cursor(grid_offsets_.begin(), grid_offsets_.end() - 1);
  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    for_each_bucket(cell, [&](std::size_t bucket) {
      grid_cells_[static_cast<std::size_t>(cursor[bucket]++)] = static_cast<std::int64_t>(cell);
    });
  }

  // The same integers as 64-bit limbs for the inline route, when every one
  // of them and their offsets fit.
  inline_ = true;
  inline_offsets_.reserve(numerators_.size() + 2);
  inline_offsets_.push_back(0);
  const auto append = [&](const Dyadic& value) {
    // An integer is magnitude 2^exponent with exponent >= 0.
    const Limbs& words = value.magnitude();
    const std::int64_t length =
        value.is_zero() ? 0 : bit_length(words) + value.exponent();
    const std::size_t limbs = static_cast<std::size_t>((length + 63) / 64);
    if (limbs > Small::kLimbs || value.exponent() < 0) {
      inline_ = false;
      return;
    }
    const std::size_t begin = inline_limbs_.size();
    inline_limbs_.resize(begin + limbs, 0);
    for (std::size_t index = 0; index < words.size(); ++index) {
      const std::int64_t bit = value.exponent() + static_cast<std::int64_t>(32 * index);
      const std::size_t limb = static_cast<std::size_t>(bit / 64);
      const int offset = static_cast<int>(bit % 64);
      inline_limbs_[begin + limb] |= static_cast<std::uint64_t>(words[index]) << offset;
      // The part of a word that crosses a limb is zero only at the top.
      const std::uint64_t high =
          offset > 32 ? static_cast<std::uint64_t>(words[index]) >> (64 - offset) : 0;
      if (high != 0) inline_limbs_[begin + limb + 1] |= high;
    }
    inline_negative_.push_back(value.sign() < 0 ? 1 : 0);
    if (inline_limbs_.size() > std::numeric_limits<std::uint32_t>::max()) {
      inline_ = false;
      return;
    }
    inline_offsets_.push_back(static_cast<std::uint32_t>(inline_limbs_.size()));
  };
  for (const Dyadic& value : numerators_) {
    if (inline_) append(value);
  }
  if (inline_) append(denominator_);
  if (!inline_) {
    inline_limbs_.clear();
    inline_offsets_.clear();
    inline_negative_.clear();
  }
}

CarrierSimplex CarrierSource::prepare_simplex(CarrierSimplex simplex) const {
  simplex.projected_axes = 0;
  if (simplex.size != 2 && simplex.size != 3) return simplex;
  for (int corner = 0; corner < simplex.size; ++corner) {
    const auto row = simplex.rows[static_cast<std::size_t>(corner)];
    if (row < 0 || row >= point_count()) {
      throw std::invalid_argument("A carrier simplex names an absent source point.");
    }
  }
  unsigned axes = 0;
  for (int axis = 0; axis < 3; ++axis) {
    const Dyadic& origin = numerators_[3 * simplex.rows[0] + axis];
    for (int corner = 1; corner < simplex.size; ++corner) {
      const auto row = simplex.rows[static_cast<std::size_t>(corner)];
      if (!(numerators_[3 * row + axis] - origin).is_zero()) {
        axes |= 1U << axis;
        break;
      }
    }
  }
  // An independent simplex spanning exactly this many coordinate directions
  // fills its coordinate subspace. Degenerate candidates still fail location.
  if (std::popcount(axes) == simplex.size - 1) {
    simplex.projected_axes = static_cast<std::uint8_t>(axes);
  }
  return simplex;
}

std::vector<Carrier> CarrierSource::carriers(const double* target, const double* current,
                                             const std::vector<CarrierSimplex>& simplices,
                                             std::int64_t cell, CarrierOrder order,
                                             std::int64_t limit) const {
  const Geometry geometry{points_,      cells_,     lower_,        upper_,     grid_origin_,
                          grid_scale_,  grid_size_, grid_offsets_, grid_cells_};
  if (inline_) {
    try {
      InlineView view{&inline_limbs_, &inline_offsets_, &inline_negative_, Small()};
      view.common = view.at(numerators_.size());
      return search(view, geometry, target, current, simplices, cell, order, limit);
    } catch (const CapacityExceeded&) {
      // The unbounded integers below repeat the query.
    }
  }
  const ExactView view{&numerators_, &denominator_};
  return search(view, geometry, target, current, simplices, cell, order, limit);
}

}  // namespace phx::mc::metric
