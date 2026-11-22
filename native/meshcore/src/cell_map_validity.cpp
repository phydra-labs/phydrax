// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// phx_mc_cell_map_validity: the exact validity proof of dof-linear polynomial
// cell maps (affine simplices, multilinear boxes and P1 prisms in three dimensions),
// reproducing bit for bit the decisions, directed determinant endpoints and
// exact-map digests of the rational per-cell proof
// (phydrax.discretization._cell_geometry_validity._certify_source_block).
//
// A cell's map is x_a(xi) = sum_dof c_dof,a phi_dof(xi) with an integer basis
// phi_dof = sum_terms (numerator / 2^dlog) xi^exponent and dyadic coordinates
// c = s 2^e. Scaling every coordinate of a cell by 2^-emin makes every
// coefficient an integer T 2^(emin - dlog); all decisions are exact integer
// arithmetic on T. The decided proof is
//   affine maps: d = det J, valid iff d > 0 and d^2 > floor^2 prod_k |J e_k|^2,
//     endpoints the directed binary64 roundings of d, depth 0;
//   multilinear box maps: the least-degree tensor Bernstein controls of
//     det J(xi) on [0,1]^3; a corner value <= 0 is invalid, a positive least
//     control is valid when the relative Hadamard margin follows from the
//     directed corner bounds of every Jacobian entry; endpoints the directed
//     roundings of the least and greatest control, depth 0;
//   P1 prism maps (triangle-linear times interval-linear): the same root
//     proof on det J's support projection (prism, triangle or box over the
//     active axes, least degree), the entry bounds being each entry's values
//     at the six prism vertices.
// Everything else (subdivision, the expanded Hadamard margin, values outside
// the fixed-width integer range or binary64) is left undecided for the
// rational proof. The digest is BLAKE2b-128 of the ASCII Python repr of the
// rational proof's digest record, built from the exact coefficients in the
// insertion order of its linear combination (zero totals are removed and
// re-inserted at the end, as a Python dict does).
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "parallel.hpp"
#include "phydrax_meshcore.h"
#include "wide_integer.hpp"

namespace {

// ---------------------------------------------------------------- integers

constexpr int kLimbs = 24;  // 1536 bits

struct Undecided {};  // a value left the fixed-width or binary64 range

struct Big {
  bool negative = false;
  int size = 0;  // used limbs; size == 0 is zero
  std::array<std::uint64_t, kLimbs> limb;  // limbs at and above `size` are unspecified

  Big() noexcept = default;
  // Copies move only the used limbs (a few, so no library call).
  Big(const Big& other) noexcept : negative(other.negative), size(other.size) {
    for (int index = 0; index < size; ++index) limb[index] = other.limb[index];
  }
  Big& operator=(const Big& other) noexcept {
    negative = other.negative;
    size = other.size;
    for (int index = 0; index < size; ++index) limb[index] = other.limb[index];
    return *this;
  }

  bool zero() const noexcept { return size == 0; }
  int sign() const noexcept { return size == 0 ? 0 : (negative ? -1 : 1); }
  void trim() noexcept {
    while (size > 0 && limb[size - 1] == 0) --size;
    if (size == 0) negative = false;
  }
};

Big from_int64(std::int64_t value) {
  Big out;
  if (value == 0) return out;
  out.negative = value < 0;
  out.limb[0] = value < 0 ? std::uint64_t{0} - static_cast<std::uint64_t>(value)
                          : static_cast<std::uint64_t>(value);
  out.size = 1;
  return out;
}

int compare_magnitude(const Big& a, const Big& b) noexcept {
  if (a.size != b.size) return a.size < b.size ? -1 : 1;
  for (int index = a.size - 1; index >= 0; --index) {
    if (a.limb[index] != b.limb[index]) return a.limb[index] < b.limb[index] ? -1 : 1;
  }
  return 0;
}

// |a| + |b| into out (out may alias neither).
void add_magnitude(const Big& a, const Big& b, Big& out) {
  const Big& longer = a.size >= b.size ? a : b;
  const Big& shorter = a.size >= b.size ? b : a;
  std::uint64_t carry = 0;
  for (int index = 0; index < longer.size; ++index) {
    out.limb[index] = phx::mc::add_with_carry(
        longer.limb[index], index < shorter.size ? shorter.limb[index] : 0, carry);
  }
  out.size = longer.size;
  if (carry != 0) {
    if (out.size == kLimbs) throw Undecided{};
    out.limb[out.size++] = carry;
  }
}

// |a| - |b| into out for |a| >= |b|.
void subtract_magnitude(const Big& a, const Big& b, Big& out) {
  std::uint64_t borrow = 0;
  for (int index = 0; index < a.size; ++index) {
    const std::uint64_t right = index < b.size ? b.limb[index] : 0;
    const std::uint64_t left = a.limb[index];
    const std::uint64_t difference = left - right - borrow;
    borrow = (left < right || (left == right && borrow != 0)) ? 1 : 0;
    out.limb[index] = difference;
  }
  out.size = a.size;
  out.trim();
}

Big signed_sum(const Big& a, const Big& b, bool negate_b) {
  Big out;
  const bool b_negative = b.negative != negate_b;
  if (b.zero()) return a;
  if (a.zero()) {
    out = b;
    out.negative = b_negative;
    return out;
  }
  if (a.negative == b_negative) {
    add_magnitude(a, b, out);
    out.negative = a.negative;
    return out;
  }
  const int order = compare_magnitude(a, b);
  if (order == 0) return out;
  if (order > 0) {
    subtract_magnitude(a, b, out);
    out.negative = a.negative;
  } else {
    subtract_magnitude(b, a, out);
    out.negative = b_negative;
  }
  out.trim();
  return out;
}

Big operator+(const Big& a, const Big& b) { return signed_sum(a, b, false); }
Big operator-(const Big& a, const Big& b) { return signed_sum(a, b, true); }

Big operator*(const Big& a, const Big& b) {
  Big out;
  if (a.zero() || b.zero()) return out;
  if (a.size + b.size > kLimbs) throw Undecided{};
  const int width = a.size + b.size;
  for (int index = 0; index < width; ++index) out.limb[index] = 0;
  for (int row = 0; row < a.size; ++row) {
    std::uint64_t carry = 0;
    for (int column = 0; column < b.size; ++column) {
      const int at = row + column;
      out.limb[at] = phx::mc::multiply_accumulate(
          a.limb[row], b.limb[column], out.limb[at], carry, carry);
    }
    out.limb[row + b.size] = carry;
  }
  out.size = width;
  out.negative = a.negative != b.negative;
  out.trim();
  return out;
}

Big shift_left(const Big& a, std::int64_t bits) {
  if (a.zero() || bits == 0) return a;
  const std::int64_t length =
      (a.size - 1) * 64 + (64 - std::countl_zero(a.limb[a.size - 1])) + bits;
  if (length > 64 * kLimbs) throw Undecided{};
  const std::int64_t words = bits / 64;
  const int rest = static_cast<int>(bits % 64);
  Big out;
  out.negative = a.negative;
  out.size = static_cast<int>((length + 63) / 64);
  std::fill(out.limb.begin(), out.limb.begin() + out.size, 0);
  for (int index = 0; index < a.size; ++index) {
    const std::int64_t at = index + words;
    if (at < out.size) out.limb[at] |= a.limb[index] << rest;
    if (rest != 0 && at + 1 < out.size) out.limb[at + 1] |= a.limb[index] >> (64 - rest);
  }
  out.trim();
  return out;
}

std::int64_t bit_length(const Big& a) noexcept {
  return a.zero() ? 0 : (a.size - 1) * 64 + (64 - std::countl_zero(a.limb[a.size - 1]));
}

std::int64_t trailing_zeros(const Big& a) noexcept {
  for (int index = 0; index < a.size; ++index) {
    if (a.limb[index] != 0) return index * 64 + std::countr_zero(a.limb[index]);
  }
  return 0;
}

bool bit(const Big& a, std::int64_t position) noexcept {
  if (position < 0) return false;
  const std::int64_t word = position / 64;
  return word < a.size && ((a.limb[word] >> (position % 64)) & 1u) != 0;
}

// Bits [start, start + count) of |a| for 0 <= start and count <= 64.
std::uint64_t bits_from(const Big& a, std::int64_t start, int count) noexcept {
  if (count == 0) return 0;
  const std::int64_t word = start / 64;
  const int offset = static_cast<int>(start % 64);
  std::uint64_t value = word < a.size ? a.limb[word] >> offset : 0;
  if (offset != 0 && word + 1 < a.size) value |= a.limb[word + 1] << (64 - offset);
  return count == 64 ? value : value & ((std::uint64_t{1} << count) - 1);
}

bool any_below(const Big& a, std::int64_t position) noexcept {
  return !a.zero() && position > 0 && trailing_zeros(a) < position;
}

Big shift_right_magnitude(const Big& a, std::int64_t bits) {
  Big out;
  if (bits >= bit_length(a)) return out;
  const std::int64_t words = bits / 64;
  const int rest = static_cast<int>(bits % 64);
  out.size = static_cast<int>(a.size - words);
  for (int index = 0; index < out.size; ++index) {
    const std::uint64_t low = a.limb[index + words] >> rest;
    const std::uint64_t high = (rest != 0 && index + words + 1 < a.size)
                                   ? a.limb[index + words + 1] << (64 - rest)
                                   : 0;
    out.limb[index] = low | high;
  }
  out.negative = a.negative;
  out.trim();
  return out;
}

// Decimal text writers advance a cursor into storage the caller sized.
void put(const char* text, std::size_t length, char*& out) noexcept {
  std::memcpy(out, text, length);
  out += length;
}

template <std::size_t N>
void put(const char (&text)[N], char*& out) noexcept {
  put(text, N - 1, out);
}

void append_unsigned(std::uint64_t value, char*& out) noexcept {
  out = std::to_chars(out, out + 24, value).ptr;
}

// Longest decimal text of a magnitude below 2^(64 kLimbs), sign included.
constexpr std::size_t kDecimalCapacity = 20 * kLimbs + 2;

// Python's int repr of a.
void append_decimal(const Big& a, char*& out) {
  if (a.negative) *out++ = '-';
  if (a.size <= 1) {
    append_unsigned(a.zero() ? 0 : a.limb[0], out);
    return;
  }
  std::array<std::uint64_t, kLimbs> work{};
  std::copy_n(a.limb.begin(), a.size, work.begin());
  int size = a.size;
  std::array<std::uint64_t, 2 * kLimbs> chunks{};  // base 10^19, least significant first
  int count = 0;
  constexpr std::uint64_t kBase = UINT64_C(10000000000000000000);
  while (size > 0) {
    std::uint64_t remainder = 0;
    for (int index = size - 1; index >= 0; --index) {
      work[index] = phx::mc::divide_wide(work[index], kBase, remainder);
    }
    while (size > 0 && work[size - 1] == 0) --size;
    chunks[count++] = remainder;
  }
  append_unsigned(chunks[count - 1], out);
  for (int index = count - 1; index-- > 0;) {
    char buffer[24];
    const auto end = std::to_chars(buffer, buffer + sizeof(buffer), chunks[index]).ptr;
    const std::size_t digits = static_cast<std::size_t>(end - buffer);
    std::memset(out, '0', 19 - digits);
    out += 19 - digits;
    put(buffer, digits, out);
  }
}

// --------------------------------------------------------------- rounding

// Round-to-nearest-even binary64 value of a 2^exponent, as Python's
// float(Fraction) computes it (a negative value rounding to zero gives -0.0),
// and direction = sign(rounded - exact).
double round_nearest(const Big& a, std::int64_t exponent, int& direction) {
  direction = 0;
  if (a.zero()) return 0.0;
  const std::int64_t length = bit_length(a);
  const std::int64_t top = length - 1 + exponent;  // floor(log2 |value|)
  if (top > 1023) throw Undecided{};
  // Lowest kept bit position (as a power of two) is max(top - 52, -1074).
  const std::int64_t lowest = std::max<std::int64_t>(top - 52, -1074);
  const std::int64_t drop = lowest - exponent;  // low bits of |a| to discard
  std::uint64_t kept = 0;
  bool up = false;
  bool inexact = false;
  if (drop <= 0) {
    kept = bits_from(a, 0, static_cast<int>(length));
    kept <<= -drop;
  } else {
    kept = drop >= length ? 0 : bits_from(a, drop, static_cast<int>(length - drop));
    const bool half = bit(a, drop - 1);
    const bool sticky = any_below(a, drop - 1);
    inexact = half || sticky;
    if (half && (sticky || (kept & 1u) != 0)) {
      ++kept;
      up = true;
    }
  }
  std::int64_t scale = lowest;
  if (kept == (std::uint64_t{1} << 53)) {
    kept >>= 1;
    ++scale;
  }
  if (kept != 0 && scale + (63 - std::countl_zero(kept)) > 1023) throw Undecided{};
  double magnitude = std::ldexp(static_cast<double>(kept), static_cast<int>(scale));
  if (!std::isfinite(magnitude)) throw Undecided{};
  if (inexact) direction = up ? 1 : -1;  // in magnitude
  if (a.negative) {
    direction = -direction;
    magnitude = -magnitude;
  }
  return magnitude;
}

double round_down(const Big& a, std::int64_t exponent) {
  int direction = 0;
  const double nearest = round_nearest(a, exponent, direction);
  return direction > 0 ? std::nextafter(nearest, -std::numeric_limits<double>::infinity())
                       : nearest;
}

double round_up(const Big& a, std::int64_t exponent) {
  int direction = 0;
  const double nearest = round_nearest(a, exponent, direction);
  return direction < 0 ? std::nextafter(nearest, std::numeric_limits<double>::infinity())
                       : nearest;
}

// Exact binary64 value as integer and exponent (value = out 2^exponent).
Big from_double(double value, std::int64_t& exponent) {
  exponent = 0;
  if (value == 0.0) return Big{};
  int binary = 0;
  const double fraction = std::frexp(std::fabs(value), &binary);
  const auto mantissa = static_cast<std::int64_t>(std::ldexp(fraction, 53));
  exponent = static_cast<std::int64_t>(binary) - 53;
  Big out = from_int64(mantissa);
  out.negative = value < 0.0;
  return out;
}

// a 2^ea compared with b 2^eb.
int compare_scaled(const Big& a, std::int64_t ea, const Big& b, std::int64_t eb) {
  const std::int64_t base = std::min(ea, eb);
  const Big left = shift_left(a, ea - base);
  const Big right = shift_left(b, eb - base);
  return (left - right).sign();
}

// ---------------------------------------------------------------- BLAKE2b

class Blake2b128 {
 public:
  Blake2b128() {
    state_ = kIv;
    state_[0] ^= 0x01010000u ^ 16u;  // digest length 16, no key, fanout/depth 1
  }

  void update(const char* next, std::size_t left) {
    while (left > 0) {
      if (filled_ == 128) {
        counter_ += 128;
        compress(block_.data(), false);
        filled_ = 0;
      }
      if (filled_ == 0 && left > 128) {
        // A full block followed by more input is not the last: hash it in place.
        counter_ += 128;
        compress(reinterpret_cast<const std::uint8_t*>(next), false);
        next += 128;
        left -= 128;
        continue;
      }
      const std::size_t take = std::min<std::size_t>(left, 128 - filled_);
      std::memcpy(block_.data() + filled_, next, take);
      filled_ += static_cast<int>(take);
      next += take;
      left -= take;
    }
  }

  void finish(std::uint8_t* out) {
    counter_ += filled_;
    std::fill(block_.begin() + filled_, block_.end(), 0);
    compress(block_.data(), true);
    for (int index = 0; index < 16; ++index) {
      out[index] = static_cast<std::uint8_t>(state_[index / 8] >> (8 * (index % 8)));
    }
  }

 private:
  static constexpr std::array<std::uint64_t, 8> kIv{
      UINT64_C(0x6a09e667f3bcc908), UINT64_C(0xbb67ae8584caa73b),
      UINT64_C(0x3c6ef372fe94f82b), UINT64_C(0xa54ff53a5f1d36f1),
      UINT64_C(0x510e527fade682d1), UINT64_C(0x9b05688c2b3e6c1f),
      UINT64_C(0x1f83d9abfb41bd6b), UINT64_C(0x5be0cd19137e2179)};
  static constexpr std::array<std::array<std::uint8_t, 16>, 12> kSigma{{
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
      {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
      {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
      {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
      {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
      {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
      {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
      {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
      {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
      {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
      {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
  }};

  static void mix(std::array<std::uint64_t, 16>& v, int a, int b, int c, int d,
                  std::uint64_t x, std::uint64_t y) noexcept {
    v[a] = v[a] + v[b] + x;
    v[d] = std::rotr(v[d] ^ v[a], 32);
    v[c] = v[c] + v[d];
    v[b] = std::rotr(v[b] ^ v[c], 24);
    v[a] = v[a] + v[b] + y;
    v[d] = std::rotr(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = std::rotr(v[b] ^ v[c], 63);
  }

  void compress(const std::uint8_t* block, bool last) noexcept {
    std::array<std::uint64_t, 16> message;
    for (int word = 0; word < 16; ++word) {
      std::uint64_t value = 0;
      for (int byte = 0; byte < 8; ++byte) {
        value |= static_cast<std::uint64_t>(block[8 * word + byte]) << (8 * byte);
      }
      message[word] = value;
    }
    std::array<std::uint64_t, 16> v;
    for (int index = 0; index < 8; ++index) {
      v[index] = state_[index];
      v[index + 8] = kIv[index];
    }
    v[12] ^= counter_;
    if (last) v[14] = ~v[14];
#if defined(__clang__)
#pragma clang loop unroll(full)
#endif
    for (const auto& s : kSigma) {
      mix(v, 0, 4, 8, 12, message[s[0]], message[s[1]]);
      mix(v, 1, 5, 9, 13, message[s[2]], message[s[3]]);
      mix(v, 2, 6, 10, 14, message[s[4]], message[s[5]]);
      mix(v, 3, 7, 11, 15, message[s[6]], message[s[7]]);
      mix(v, 0, 5, 10, 15, message[s[8]], message[s[9]]);
      mix(v, 1, 6, 11, 12, message[s[10]], message[s[11]]);
      mix(v, 2, 7, 8, 13, message[s[12]], message[s[13]]);
      mix(v, 3, 4, 9, 14, message[s[14]], message[s[15]]);
    }
    for (int index = 0; index < 8; ++index) state_[index] ^= v[index] ^ v[index + 8];
  }

  std::array<std::uint64_t, 8> state_{};
  std::array<std::uint8_t, 128> block_{};
  int filled_ = 0;
  std::uint64_t counter_ = 0;  // messages here are far below 2^64 bytes
};

// ------------------------------------------------------------- cell proofs

struct Term {
  int key;  // exponent (a, b, c) encoded 64 a + 8 b + c
  std::array<int, 3> exponent;
  Big total;
};

using Expression = std::vector<Term>;  // insertion order

struct Basis {
  std::vector<std::int64_t> starts;
  std::vector<std::array<int, 3>> exponents;
  std::vector<std::int64_t> numerators;
};

// Python's PreparedLinearCombinationBasis.combine for one axis, scaled.
void combine(const Basis& basis, const std::vector<Big>& values, Expression& out) {
  out.clear();
  const std::size_t dofs = values.size();
  for (std::size_t dof = 0; dof < dofs; ++dof) {
    const Big& multiplier = values[dof];
    if (multiplier.zero()) continue;
    for (std::int64_t term = basis.starts[dof]; term < basis.starts[dof + 1]; ++term) {
      const auto& exponent = basis.exponents[static_cast<std::size_t>(term)];
      const int key = 64 * exponent[0] + 8 * exponent[1] + exponent[2];
      const std::int64_t numerator = basis.numerators[static_cast<std::size_t>(term)];
      Big product = multiplier;
      if (numerator == -1) {
        product.negative = !product.negative;
      } else if (numerator != 1) {
        product = multiplier * from_int64(numerator);
      }
      auto found = std::find_if(out.begin(), out.end(),
                                [key](const Term& entry) { return entry.key == key; });
      if (found == out.end()) {
        if (!product.zero()) out.push_back(Term{key, exponent, product});
      } else {
        Big total = found->total + product;
        if (total.zero()) {
          out.erase(found);
        } else {
          found->total = total;
        }
      }
    }
  }
}

// Exact rational T 2^shift in lowest terms, as Python's Fraction repr parts.
void append_fraction(const Big& total, std::int64_t shift, char*& out) {
  const std::int64_t zeros = trailing_zeros(total);
  const std::int64_t power = shift + zeros;
  if (total.size == 1 && power < 0 && power > -64) {
    // One-limb odd numerator over a one-limb power of two.
    if (total.negative) *out++ = '-';
    append_unsigned(total.limb[0] >> zeros, out);
    put(", ", out);
    append_unsigned(std::uint64_t{1} << -power, out);
    return;
  }
  const Big odd = shift_right_magnitude(total, zeros);
  if (power >= 0) {
    append_decimal(shift_left(odd, power), out);
    put(", 1", out);
  } else {
    append_decimal(odd, out);
    put(", ", out);
    append_decimal(shift_left(from_int64(1), -power), out);
  }
}

// `record` is caller scratch storage reused across cells.
void map_digest(const std::string& kind, const std::array<Expression, 3>& axes,
                std::int64_t shift, std::string& record, std::uint8_t* digest) {
  // Every term writes three one-digit exponents and two fraction parts.
  std::size_t capacity = kind.size() + 8;
  for (const Expression& expression : axes) {
    capacity += 40 + expression.size() * (24 + 2 * kDecimalCapacity);
  }
  if (record.size() < capacity) record.resize(capacity);
  char* const start = record.data();
  char* out = start;
  put("['", out);
  put(kind.data(), kind.size(), out);
  put("', 3", out);
  for (const Expression& expression : axes) {
    put(", 'polynomial', ", out);
    append_unsigned(expression.size(), out);
    for (const Term& term : expression) {
      put(", ((", out);
      *out++ = static_cast<char>('0' + term.exponent[0]);
      put(", ", out);
      *out++ = static_cast<char>('0' + term.exponent[1]);
      put(", ", out);
      *out++ = static_cast<char>('0' + term.exponent[2]);
      put("), ", out);
      append_fraction(term.total, shift, out);
      *out++ = ')';
    }
  }
  *out++ = ']';
  Blake2b128 hash;
  hash.update(start, static_cast<std::size_t>(out - start));
  hash.finish(digest);
}

Big determinant3(const std::array<std::array<Big, 3>, 3>& m) {
  return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
         m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
         m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

// Dense polynomial in three variables with degree <= 2 per axis.
struct Dense {
  std::array<Big, 27> c;  // index 9a + 3b + c
};

// Indices of the nonzero coefficients.
struct Slots {
  int count = 0;
  std::array<std::uint8_t, 27> index{};
};

Slots nonzero(const Dense& p) noexcept {
  Slots out;
  for (int i = 0; i < 27; ++i) {
    if (!p.c[i].zero()) out.index[out.count++] = static_cast<std::uint8_t>(i);
  }
  return out;
}

// p q accumulated into out with sign (+1 or -1).
void accumulate_product(const Dense& p, const Dense& q, int sign, Dense& out) {
  const Slots left = nonzero(p);
  const Slots right = nonzero(q);
  for (int x = 0; x < left.count; ++x) {
    const int i = left.index[x];
    for (int y = 0; y < right.count; ++y) {
      const int j = right.index[y];
      const int a = i / 9 + j / 9, b = (i / 3) % 3 + (j / 3) % 3, c = i % 3 + j % 3;
      if (a > 2 || b > 2 || c > 2) throw Undecided{};
      Big& slot = out.c[9 * a + 3 * b + c];
      slot = signed_sum(slot, p.c[i] * q.c[j], sign < 0);
    }
  }
}

Dense product(const Dense& p, const Dense& q) {
  Dense out;
  accumulate_product(p, q, 1, out);
  return out;
}

Dense difference(const Dense& p, const Dense& q) {
  Dense out = p;
  for (int i = 0; i < 27; ++i) {
    if (!q.c[i].zero()) out.c[i] = out.c[i] - q.c[i];
  }
  return out;
}

int compare(const Big& a, const Big& b) noexcept {
  if (a.sign() != b.sign()) return a.sign() < b.sign() ? -1 : 1;
  const int magnitude = compare_magnitude(a, b);
  return a.negative ? -magnitude : magnitude;
}

// In place along `axis`, over the index box [0, degree_k] of every axis: the
// monomial coefficients of degree `degree[axis]` become tensor Bernstein
// controls. Scaled (`doubled`) controls are 2 C(i, j) / C(d, j) weighted sums,
// integers for d <= 2 (twice the controls); unscaled is only for d <= 1.
void bernstein_axis(Dense& p, const std::array<int, 3>& degree, int axis, bool doubled) {
  static constexpr std::array<int, 3> kStride{9, 3, 1};
  const int first = axis == 0 ? 1 : 0;
  const int second = axis == 2 ? 1 : 2;
  for (int u = 0; u <= degree[first]; ++u) {
    for (int v = 0; v <= degree[second]; ++v) {
      const int base = kStride[first] * u + kStride[second] * v;
      Big& a0 = p.c[base];
      Big& a1 = p.c[base + kStride[axis]];
      Big& a2 = p.c[base + 2 * kStride[axis]];
      switch (degree[axis]) {
        case 0:
          if (doubled) a0 = shift_left(a0, 1);
          break;
        case 1:
          a1 = a0 + a1;
          if (doubled) {
            a0 = shift_left(a0, 1);
            a1 = shift_left(a1, 1);
          }
          break;
        default: {
          // 2 b0 = 2 a0, 2 b1 = 2 a0 + a1, 2 b2 = 2 (a0 + a1 + a2).
          const Big twice = shift_left(a0, 1);
          a2 = shift_left(a0 + a1 + a2, 1);
          a1 = twice + a1;
          a0 = twice;
          break;
        }
      }
    }
  }
}

struct Proof {
  int status = 0;
  double lower = 0.0;
  double upper = 0.0;
};

// Exact d^2 > floor^2 prod_k sum_a J[a][k]^2 with J and d on one integer scale.
bool affine_margin(const Big& determinant, const std::array<std::array<Big, 3>, 3>& jacobian,
                   double floor) {
  Big product = from_int64(1);
  for (int column = 0; column < 3; ++column) {
    Big norm;
    for (int row = 0; row < 3; ++row) norm = norm + jacobian[row][column] * jacobian[row][column];
    product = product * norm;
  }
  std::int64_t floor_exponent = 0;
  const Big floor_integer = from_double(floor, floor_exponent);
  const Big right = floor_integer * floor_integer * product;
  return compare_scaled(determinant * determinant, 0, right, 2 * floor_exponent) > 0;
}

bool prove_affine(const std::array<Expression, 3>& axes, std::int64_t scale, double floor,
                  Proof& proof) {
  std::array<std::array<Big, 3>, 3> jacobian;
  for (int row = 0; row < 3; ++row) {
    for (const Term& term : axes[row]) {
      const int degree = term.exponent[0] + term.exponent[1] + term.exponent[2];
      if (degree > 1) return false;
      if (degree == 1) {
        const int column = term.exponent[0] == 1 ? 0 : (term.exponent[1] == 1 ? 1 : 2);
        jacobian[row][column] = term.total;
      }
    }
  }
  const Big determinant = determinant3(jacobian);
  const bool valid = determinant.sign() > 0 && affine_margin(determinant, jacobian, floor);
  proof.status = valid ? 0 : 1;
  proof.lower = round_down(determinant, 3 * scale);
  proof.upper = round_up(determinant, 3 * scale);
  return true;
}

// Least and greatest control of one Jacobian entry (on the 2^scale grid).
struct Range {
  Big low;
  Big high;
  void include(const Big& value, bool first) {
    if (first || compare(value, low) < 0) low = value;
    if (first || compare(value, high) > 0) high = value;
  }
};
using EntryBounds = std::array<std::array<Range, 3>, 3>;  // [row][column]

// The relative Hadamard margin of _hadamard_margin_from_bounds: lower^2 >
// floor^2 prod_k sum_a max(|low|, |high|)^2 over the directed binary64
// endpoints of every entry's controls; false leaves the expanded margin.
bool bounded_margin(double determinant_lower, const EntryBounds& bounds, std::int64_t scale,
                    double floor) {
  if (!(determinant_lower > 0.0)) return false;
  std::int64_t lower_exponent = 0;
  const Big lower = from_double(determinant_lower, lower_exponent);
  Big norms_product = from_int64(1);
  std::int64_t product_exponent = 0;
  for (int column = 0; column < 3; ++column) {
    Big norm;
    std::int64_t norm_exponent = 0;
    bool started = false;
    for (int row = 0; row < 3; ++row) {
      const Range& range = bounds[row][column];
      const double bound = std::max(std::fabs(round_down(range.low, scale)),
                                    std::fabs(round_up(range.high, scale)));
      std::int64_t exponent = 0;
      const Big magnitude = from_double(bound, exponent);
      const Big square = magnitude * magnitude;
      if (square.zero()) continue;
      const std::int64_t square_exponent = 2 * exponent;
      if (!started) {
        norm = square;
        norm_exponent = square_exponent;
        started = true;
      } else {
        const std::int64_t base = std::min(norm_exponent, square_exponent);
        norm = shift_left(norm, norm_exponent - base) +
               shift_left(square, square_exponent - base);
        norm_exponent = base;
      }
    }
    norms_product = norms_product * norm;
    product_exponent += norm_exponent;
  }
  std::int64_t floor_exponent = 0;
  const Big floor_integer = from_double(floor, floor_exponent);
  const Big right = floor_integer * floor_integer * norms_product;
  return compare_scaled(lower * lower, 2 * lower_exponent, right,
                        2 * floor_exponent + product_exponent) > 0;
}

// Root Bernstein proof of a multilinear box map; false leaves it undecided.
bool prove_multilinear(const std::array<Expression, 3>& axes, std::int64_t scale,
                       double floor, Proof& proof) {
  // Jacobian entries J[a][k] = d x_a / d xi_k, dense, degree <= 1 per axis.
  std::array<std::array<Dense, 3>, 3> jacobian;
  for (int row = 0; row < 3; ++row) {
    for (const Term& term : axes[row]) {
      for (int axis = 0; axis < 3; ++axis) {
        if (term.exponent[axis] > 1) return false;
      }
      for (int column = 0; column < 3; ++column) {
        if (term.exponent[column] != 1) continue;
        std::array<int, 3> reduced = term.exponent;
        reduced[column] = 0;
        jacobian[row][column].c[9 * reduced[0] + 3 * reduced[1] + reduced[2]] = term.total;
      }
    }
  }
  // Cofactor expansion along the first row; every product keeps degree <= 2.
  Dense determinant;
  const auto minor = [&](int left, int right) {
    return difference(product(jacobian[1][left], jacobian[2][right]),
                      product(jacobian[1][right], jacobian[2][left]));
  };
  accumulate_product(jacobian[0][0], minor(1, 2), 1, determinant);
  accumulate_product(jacobian[0][1], minor(0, 2), -1, determinant);
  accumulate_product(jacobian[0][2], minor(0, 1), 1, determinant);
  // Least tensor degree from the nonzero coefficients.
  std::array<int, 3> degree{0, 0, 0};
  for (int i = 0; i < 27; ++i) {
    if (determinant.c[i].zero()) continue;
    degree[0] = std::max(degree[0], i / 9);
    degree[1] = std::max(degree[1], (i / 3) % 3);
    degree[2] = std::max(degree[2], i % 3);
  }
  // Eight times the least-degree controls; the corner controls are eight
  // times the corner values.
  Dense& controls = determinant;
  for (int axis = 0; axis < 3; ++axis) bernstein_axis(controls, degree, axis, true);
  Big least;
  Big greatest;
  bool first = true;
  bool corner_nonpositive = false;
  for (int i0 = 0; i0 <= degree[0]; ++i0) {
    for (int i1 = 0; i1 <= degree[1]; ++i1) {
      for (int i2 = 0; i2 <= degree[2]; ++i2) {
        const Big& control = controls.c[9 * i0 + 3 * i1 + i2];
        if (first || compare(control, least) < 0) least = control;
        if (first || compare(control, greatest) > 0) greatest = control;
        first = false;
        if ((i0 == 0 || i0 == degree[0]) && (i1 == 0 || i1 == degree[1]) &&
            (i2 == 0 || i2 == degree[2]) && control.sign() <= 0) {
          corner_nonpositive = true;
        }
      }
    }
  }
  proof.lower = round_down(least, 3 * scale - 3);
  proof.upper = round_up(greatest, 3 * scale - 3);
  if (corner_nonpositive) {
    proof.status = 1;
    return true;
  }
  if (least.sign() <= 0) return false;  // subdivision
  // Relative Hadamard margin from directed corner bounds of every entry.
  static constexpr std::array<int, 3> kMultilinear{1, 1, 1};
  EntryBounds bounds;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      // Corner values of the multilinear entry (its degree-one controls).
      Dense& entry = jacobian[row][column];
      for (int axis = 0; axis < 3; ++axis) bernstein_axis(entry, kMultilinear, axis, false);
      for (int corner = 0; corner < 8; ++corner) {
        const Big& value = entry.c[9 * (corner & 1) + 3 * ((corner >> 1) & 1) + (corner >> 2)];
        bounds[row][column].include(value, corner == 0);
      }
    }
  }
  if (!bounded_margin(proof.lower, bounds, scale, floor)) return false;
  proof.status = 0;
  return true;
}

using Vector = std::array<Big, 3>;

Vector cross(const Vector& a, const Vector& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

Big dot(const Vector& a, const Vector& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// Root Bernstein proof of a P1 prism (wedge) map, every term of degree <= 1
// in (xi, eta) and in zeta: J = [A0 + zeta A1, B0 + zeta B1, C0 + xi C1 +
// eta C2], so det J = sum_k zeta^k (X_k . V(xi, eta)) with X_0 = A0 x B0,
// X_1 = A1 x B0 + A0 x B1, X_2 = A1 x B1 and V affine. The rational proof
// projects det J to its active axes (prism, the (xi, eta) triangle, or a box
// over the active axes) at least degree; on every projection the controls
// are, as a set, the zeta controls of least degree l = max_k {X_k . V != 0}
// at the three triangle vertices (V linear there; an inactive axis repeats
// values), and its corners are their zeta ends. The Hadamard entry bounds
// are each entry's least-degree prism controls: its vertex values.
bool prove_prism(const std::array<Expression, 3>& axes, std::int64_t scale, double floor,
                 Proof& proof) {
  Vector a0, a1, b0, b1, c0, c1, c2;
  for (int row = 0; row < 3; ++row) {
    for (const Term& term : axes[row]) {
      const auto& e = term.exponent;
      if (e[0] + e[1] > 1 || e[2] > 1) return false;
      if (e[0] == 1) (e[2] == 0 ? a0 : a1)[row] = term.total;
      if (e[1] == 1) (e[2] == 0 ? b0 : b1)[row] = term.total;
      if (e[2] == 1) (e[0] == 1 ? c1 : e[1] == 1 ? c2 : c0)[row] = term.total;
    }
  }
  const std::array<Vector, 3> vertex{
      c0, Vector{c0[0] + c1[0], c0[1] + c1[1], c0[2] + c1[2]},
      Vector{c0[0] + c2[0], c0[1] + c2[1], c0[2] + c2[2]}};
  const Vector x0 = cross(a0, b0);
  const Vector x2 = cross(a1, b1);
  const Vector p = cross(a1, b0);
  const Vector q = cross(a0, b1);
  const Vector x1{p[0] + q[0], p[1] + q[1], p[2] + q[2]};
  std::array<std::array<Big, 3>, 3> power;  // [vertex][k]: zeta^k coefficient
  int line = 0;
  for (int v = 0; v < 3; ++v) {
    power[v] = {dot(x0, vertex[v]), dot(x1, vertex[v]), dot(x2, vertex[v])};
    if (!power[v][2].zero()) line = 2;
    if (line == 0 && !power[v][1].zero()) line = 1;
  }
  // Twice the controls: 2 b0 = 2 c0; degree 1: 2 b1 = 2 (c0 + c1); degree 2:
  // 2 b1 = 2 c0 + c1, 2 b2 = 2 (c0 + c1 + c2).
  Big least;
  Big greatest;
  bool corner_nonpositive = false;
  for (int v = 0; v < 3; ++v) {
    const auto& c = power[v];
    std::array<Big, 3> controls;
    controls[0] = shift_left(c[0], 1);
    if (line == 1) controls[1] = shift_left(c[0] + c[1], 1);
    if (line == 2) {
      controls[1] = controls[0] + c[1];
      controls[2] = shift_left(c[0] + c[1] + c[2], 1);
    }
    for (int k = 0; k <= line; ++k) {
      if ((v == 0 && k == 0) || compare(controls[k], least) < 0) least = controls[k];
      if ((v == 0 && k == 0) || compare(controls[k], greatest) > 0) greatest = controls[k];
      if ((k == 0 || k == line) && controls[k].sign() <= 0) corner_nonpositive = true;
    }
  }
  proof.lower = round_down(least, 3 * scale - 1);
  proof.upper = round_up(greatest, 3 * scale - 1);
  if (corner_nonpositive) {
    proof.status = 1;
    return true;
  }
  if (least.sign() <= 0) return false;  // subdivision
  EntryBounds bounds;
  for (int row = 0; row < 3; ++row) {
    bounds[row][0].include(a0[row], true);
    bounds[row][0].include(a0[row] + a1[row], false);
    bounds[row][1].include(b0[row], true);
    bounds[row][1].include(b0[row] + b1[row], false);
    for (int v = 0; v < 3; ++v) bounds[row][2].include(vertex[v][row], v == 0);
  }
  if (!bounded_margin(proof.lower, bounds, scale, floor)) return false;
  proof.status = 0;
  return true;
}

constexpr std::size_t kCellChunk = 16;

}  // namespace

extern "C" int32_t phx_mc_cell_map_validity(
    int32_t domain, int64_t dof_count, const int64_t* term_starts,
    const int32_t* term_exponents, const int64_t* term_numerators,
    int32_t basis_denominator_log2, const char* kind, int64_t kind_length, int64_t row_count,
    const int64_t* significands, const int16_t* exponents, int16_t sparse_exponent,
    int64_t cell_count, const int64_t* routes, double floor, int8_t* decided,
    int32_t* status, double* lower, double* upper, uint8_t* digests, int32_t parallelism) {
  return phx::mc::guarded([&]() -> int32_t {
    if (domain < 0 || domain > 2 || dof_count < 1 || term_starts == nullptr ||
        kind == nullptr || kind_length < 1 || kind_length > 64 || row_count < 0 ||
        cell_count < 0 || basis_denominator_log2 < 0 || basis_denominator_log2 > 64 ||
        !(floor >= 0.0) || !(floor < 1.0) ||
        !phx::mc::addressable(row_count, 3, sizeof(int64_t)) ||
        !phx::mc::addressable(cell_count, dof_count, sizeof(int64_t)) ||
        !phx::mc::addressable(cell_count, 16, 1) ||
        (row_count > 0 && (significands == nullptr || exponents == nullptr)) ||
        (cell_count > 0 && (routes == nullptr || decided == nullptr || status == nullptr ||
                            lower == nullptr || upper == nullptr || digests == nullptr)) ||
        parallelism < 1) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const std::int64_t term_count = term_starts[dof_count];
    if (term_starts[0] != 0 || term_count < 0 ||
        (term_count > 0 && (term_exponents == nullptr || term_numerators == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    Basis basis;
    basis.starts.assign(term_starts, term_starts + dof_count + 1);
    for (std::int64_t dof = 0; dof < dof_count; ++dof) {
      if (basis.starts[dof + 1] < basis.starts[dof]) return PHX_MC_INVALID_ARGUMENT;
    }
    for (std::int64_t term = 0; term < term_count; ++term) {
      std::array<int, 3> exponent{};
      for (int axis = 0; axis < 3; ++axis) {
        const int32_t value = term_exponents[3 * term + axis];
        if (value < 0 || value > 7) return PHX_MC_INVALID_ARGUMENT;
        exponent[axis] = value;
      }
      basis.exponents.push_back(exponent);
      basis.numerators.push_back(term_numerators[term]);
    }
    const std::string kind_name(kind, static_cast<std::size_t>(kind_length));
    for (const char character : kind_name) {
      if (character == '\'' || character == '\\' || character < 0x20 || character > 0x7e) {
        return PHX_MC_INVALID_ARGUMENT;
      }
    }
    for (int64_t index = 0; index < cell_count * dof_count; ++index) {
      if (routes[index] < 0 || routes[index] >= row_count) return PHX_MC_INVALID_INPUT;
    }
    // Cells are independent: each chunk of kCellChunk cells (fixed, so results
    // do not depend on parallelism) writes only its own cells' outputs.
    return phx::mc::parallel_for(
        static_cast<std::size_t>(cell_count),
        phx::mc::ParallelRequest{kCellChunk, static_cast<uint32_t>(parallelism)},
        [&](std::size_t begin, std::size_t end, phx::mc::ChunkContext&) -> int32_t {
          const phx::mc::GradualNearestScope mode;
          std::vector<Big> values(static_cast<std::size_t>(dof_count));
          std::array<Expression, 3> axes;
          std::string record;
          for (int64_t cell = static_cast<int64_t>(begin); cell < static_cast<int64_t>(end);
               ++cell) {
            decided[cell] = 0;
            const int64_t* route = routes + cell * dof_count;
            bool dense = true;
            std::int64_t minimum = std::numeric_limits<std::int64_t>::max();
            for (int64_t dof = 0; dof < dof_count && dense; ++dof) {
              for (int axis = 0; axis < 3; ++axis) {
                const int64_t flat = 3 * route[dof] + axis;
                if (exponents[flat] == sparse_exponent) {
                  dense = false;
                  break;
                }
                if (significands[flat] != 0)
                  minimum = std::min<std::int64_t>(minimum, exponents[flat]);
              }
            }
            if (!dense) continue;
            if (minimum == std::numeric_limits<std::int64_t>::max()) minimum = 0;
            try {
              for (int axis = 0; axis < 3; ++axis) {
                for (int64_t dof = 0; dof < dof_count; ++dof) {
                  const int64_t flat = 3 * route[dof] + axis;
                  values[static_cast<std::size_t>(dof)] =
                      shift_left(from_int64(significands[flat]), exponents[flat] - minimum);
                }
                combine(basis, values, axes[axis]);
              }
              const std::int64_t scale = minimum - basis_denominator_log2;
              map_digest(kind_name, axes, scale, record, digests + 16 * cell);
              Proof proof;
              const bool affine =
                  std::all_of(axes.begin(), axes.end(), [](const Expression& value) {
                    return std::all_of(value.begin(), value.end(), [](const Term& term) {
                      return term.exponent[0] + term.exponent[1] + term.exponent[2] <= 1;
                    });
                  });
              bool done = false;
              if (affine) {
                done = prove_affine(axes, scale, floor, proof);
              } else if (domain == 1) {
                done = prove_multilinear(axes, scale, floor, proof);
              } else if (domain == 2) {
                done = prove_prism(axes, scale, floor, proof);
              }
              if (!done) continue;
              status[cell] = proof.status;
              lower[cell] = proof.lower;
              upper[cell] = proof.upper;
              decided[cell] = 1;
            } catch (const Undecided&) {
              decided[cell] = 0;
            }
          }
          return PHX_MC_OK;
        });
  });
}
