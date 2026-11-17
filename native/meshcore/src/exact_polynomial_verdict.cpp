// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Fused exact Bernstein verdict and canonical SHA-256 control identity.
#include "phydrax_meshcore.h"
#include "capi_guard.hpp"
#include "parallel.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {
// Unsigned, little-endian base 2^32. Scale arithmetic may exceed the checked
// polynomial word width, so it is deliberately growable.
using Big = std::vector<uint32_t>;

void trim(Big& a) {
  while (!a.empty() && a.back() == 0) a.pop_back();
}

Big integer(const uint8_t* p, int bytes, bool& negative) {
  negative = (p[bytes - 1] & 128) != 0;
  Big a(bytes / 4);
  uint64_t carry = negative ? 1 : 0;
  for (int i = 0; i < bytes / 4; ++i) {
    uint32_t v = uint32_t(p[4 * i]) | (uint32_t(p[4 * i + 1]) << 8) |
                 (uint32_t(p[4 * i + 2]) << 16) | (uint32_t(p[4 * i + 3]) << 24);
    uint64_t w = uint64_t(negative ? ~v : v) + carry;
    a[i] = uint32_t(w);
    carry = w >> 32;
  }
  trim(a);
  return a;
}

int cmp(const Big& a, const Big& b) {
  if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
  for (size_t i = a.size(); i-- > 0;) {
    if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
  }
  return 0;
}

void subtract(Big& a, const Big& b) {
  uint64_t borrow = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const uint64_t sub = (i < b.size() ? b[i] : 0) + borrow;
    const uint64_t old = a[i];
    a[i] = uint32_t(old - sub);
    borrow = old < sub;
  }
  trim(a);
}

void shift(Big& a, uint32_t bit) {
  uint64_t carry = bit;
  for (auto& v : a) {
    const uint64_t x = (uint64_t(v) << 1) | carry;
    v = uint32_t(x);
    carry = x >> 32;
  }
  if (carry) a.push_back(uint32_t(carry));
}

size_t trailing_bits(const Big& a) {
  size_t i = 0;
  while (a[i] == 0) ++i;
  return i * 32 + std::countr_zero(a[i]);
}

void shift_right(Big& a, size_t bits) {
  const size_t words = bits / 32;
  const unsigned rem = bits % 32;
  for (size_t i = 0; i + words < a.size(); ++i) {
    uint64_t v = a[i + words];
    if (rem && i + words + 1 < a.size()) v |= uint64_t(a[i + words + 1]) << 32;
    a[i] = uint32_t(v >> rem);
  }
  a.resize(a.size() - words);
  trim(a);
}

void shift_left(Big& a, size_t bits) {
  if (a.empty()) return;
  const size_t words = bits / 32;
  const unsigned rem = bits % 32;
  if (words) a.insert(a.begin(), words, 0);
  if (rem) {
    uint64_t carry = 0;
    for (auto& word : a) {
      uint64_t v = (uint64_t(word) << rem) | carry;
      word = uint32_t(v);
      carry = v >> 32;
    }
    if (carry) a.push_back(uint32_t(carry));
  }
}

Big divide(const Big& a, const Big& b, Big& remainder) {
  if (b.size() == 1 && std::has_single_bit(b[0])) {
    const unsigned bits = std::countr_zero(b[0]);
    remainder.clear();
    if (!a.empty()) {
      const uint32_t low = a[0] & (b[0] - 1);
      if (low) remainder.push_back(low);
    }
    Big q = a;
    if (!q.empty()) shift_right(q, bits);
    return q;
  }
  if (b.size() == 1) {
    Big q(a.size());
    uint64_t r = 0;
    for (size_t i = a.size(); i-- > 0;) {
      uint64_t v = (r << 32) | a[i];
      q[i] = uint32_t(v / b[0]);
      r = v % b[0];
    }
    trim(q);
    remainder.clear();
    if (r) remainder.push_back(uint32_t(r));
    return q;
  }
  const size_t bits = trailing_bits(b);
  if (bits == (b.size() - 1) * 32 + std::countr_zero(b.back()) &&
      std::has_single_bit(b.back())) {
    Big q = a;
    remainder.assign(a.begin(), a.begin() + std::min(a.size(), bits / 32));
    if (bits % 32 && bits / 32 < a.size())
      remainder.push_back(a[bits / 32] & ((uint32_t{1} << (bits % 32)) - 1));
    trim(remainder);
    if (bits / 32 >= q.size()) q.clear(); else shift_right(q, bits);
    return q;
  }
  Big q(a.size(), 0);
  remainder.clear();
  for (size_t bit = a.size() * 32; bit-- > 0;) {
    shift(remainder, (a[bit / 32] >> (bit % 32)) & 1);
    if (cmp(remainder, b) >= 0) {
      subtract(remainder, b);
      q[bit / 32] |= uint32_t{1} << (bit % 32);
    }
  }
  trim(q);
  return q;
}

Big gcd(Big a, Big b) {
  if (a.empty()) return b;
  if (b.empty()) return a;
  const size_t common = std::min(trailing_bits(a), trailing_bits(b));
  if (a.size() == 1 && std::has_single_bit(a[0]))
    return Big{uint32_t{1} << common};
  if (b.size() == 1 && std::has_single_bit(b[0]))
    return Big{uint32_t{1} << common};
  shift_right(a, trailing_bits(a));
  do {
    shift_right(b, trailing_bits(b));
    if (cmp(a, b) > 0) std::swap(a, b);
    subtract(b, a);
  } while (!b.empty());
  shift_left(a, common);
  return a;
}

Big multiply(const Big& a, const Big& b) {
  Big out(a.size() + b.size(), 0);
  for (size_t i = 0; i < a.size(); ++i) {
    uint64_t carry = 0;
    for (size_t j = 0; j < b.size(); ++j) {
      uint64_t value = uint64_t(a[i]) * b[j] + out[i + j] + carry;
      out[i + j] = uint32_t(value);
      carry = value >> 32;
    }
    out[i + b.size()] = uint32_t(carry);
  }
  trim(out);
  return out;
}

// FIPS 180-4 SHA-256, with input assembled in the public canonical encoding.
std::array<uint8_t, 32> sha256(const std::string& text) {
  static constexpr uint32_t k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  std::array<uint32_t, 8> h{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::vector<uint8_t> msg(text.begin(), text.end());
  const uint64_t bits = uint64_t(msg.size()) * 8;
  msg.push_back(128);
  while (msg.size() % 64 != 56) msg.push_back(0);
  for (int i = 7; i >= 0; --i) msg.push_back(uint8_t(bits >> (8 * i)));
  for (size_t off = 0; off < msg.size(); off += 64) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (uint32_t(msg[off + 4 * i]) << 24) |
             (uint32_t(msg[off + 4 * i + 1]) << 16) |
             (uint32_t(msg[off + 4 * i + 2]) << 8) | msg[off + 4 * i + 3];
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t a = std::rotr(w[i - 15], 7) ^ std::rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t b = std::rotr(w[i - 2], 17) ^ std::rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + a + w[i - 7] + b;
    }
    auto [a, b, c, d, e, f, g, hh] = h;
    for (int i = 0; i < 64; ++i) {
      const uint32_t s1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = hh + s1 + ch + k[i] + w[i];
      const uint32_t s0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + maj;
      hh = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
  }
  std::array<uint8_t, 32> out{};
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 4; ++j) out[4 * i + j] = uint8_t(h[i] >> (24 - 8 * j));
  }
  return out;
}

// Find the least domain degree and refuse excessively large control banks.
bool control_shape(int dimension, int domain, const int32_t* indices, int64_t count,
                   std::array<int32_t, 3>& degree, int64_t& controls) {
  for (int64_t j = 0; j < count; ++j) {
    const int32_t* e = indices + j * dimension;
    if (domain == PHX_MC_BERNSTEIN_SIMPLEX) {
      int64_t sum = 0;
      for (int k = 0; k < dimension; ++k) sum += e[k];
      if (sum > INT32_MAX) return false;
      degree[0] = std::max(degree[0], int32_t(sum));
    } else if (domain == PHX_MC_BERNSTEIN_BOX) {
      for (int k = 0; k < dimension; ++k) degree[k] = std::max(degree[k], e[k]);
    } else {
      const int64_t sum = int64_t(e[0]) + e[1];
      if (sum > INT32_MAX) return false;
      degree[0] = std::max(degree[0], int32_t(sum));
      degree[1] = std::max(degree[1], e[2]);
    }
  }
  controls = 1;
  if (domain == PHX_MC_BERNSTEIN_BOX) {
    for (int k = 0; k < dimension; ++k) {
      const int64_t factor = int64_t(degree[k]) + 1;
      if (factor > 100000 / controls) return false;
      controls *= factor;
    }
  } else {
    const int64_t n = degree[0];
    const int parts = domain == PHX_MC_BERNSTEIN_SIMPLEX ? dimension : 2;
    for (int k = 1; k <= parts; ++k) {
      if (controls > 100000 * k / (n + k)) return false;
      controls = controls * (n + k) / k;
    }
    if (domain == PHX_MC_BERNSTEIN_PRISM) {
      const int64_t factor = int64_t(degree[1]) + 1;
      if (factor > 100000 / controls) return false;
      controls *= factor;
    }
  }
  return true;
}

void append_u64(std::string& text, uint64_t value) {
  for (int i = 7; i >= 0; --i) text.push_back(char(value >> (8 * i)));
}

void append_magnitude(std::string& text, const Big& value) {
  size_t bytes = value.size() * 4;
  if (bytes) bytes -= std::countl_zero(value.back()) / 8;
  append_u64(text, bytes);
  for (size_t i = bytes; i-- > 0;)
    text.push_back(char(value[i / 4] >> ((i % 4) * 8)));
}

std::array<uint8_t, 32> control_hash(const std::vector<uint8_t>& controls, int width,
                                     int64_t count, const Big& scale) {
  Big divisor = scale;
  for (int64_t j = 0; j < count && !(divisor.size() == 1 && divisor[0] == 1); ++j) {
    bool negative = false;
    Big numerator = integer(controls.data() + j * width, width, negative);
    if (!numerator.empty()) divisor = gcd(std::move(divisor), std::move(numerator));
  }
  const bool reduced = divisor.size() == 1 && divisor[0] == 1;
  Big remainder;
  const Big denominator = reduced ? scale : divide(scale, divisor, remainder);
  std::string text("phydrax-exact-controls\0R", 24);
  append_u64(text, uint64_t(count));
  append_magnitude(text, denominator);
  for (int64_t j = 0; j < count; ++j) {
    bool negative = false;
    Big numerator = integer(controls.data() + j * width, width, negative);
    if (!reduced && !numerator.empty()) numerator = divide(numerator, divisor, remainder);
    text.push_back(negative && !numerator.empty() ? '\1' : '\0');
    append_magnitude(text, numerator);
  }
  return sha256(text);
}

int32_t verdict_item(int64_t row, int d, int domain, const int64_t* offsets,
                     const int32_t* exponents, const uint8_t* coefficients,
                     const uint8_t* scales, uint8_t* digests, int32_t* verdict, int limbs) {
  const int width = limbs * 8, entries = d * d;
  std::vector<int64_t> local(entries + 1);
  const int64_t first = offsets[row * entries];
  for (int i = 0; i <= entries; ++i) local[i] = offsets[row * entries + i] - first;
  std::array<int32_t, 3> maxima{};
  for (int64_t i = first; i < offsets[(row + 1) * entries]; ++i) {
    for (int k = 0; k < d; ++k) maxima[k] = std::max(maxima[k], exponents[i * d + k]);
  }
  std::vector<int64_t> caps(d + 1, 0);
  for (int m = 0; m < d; ++m) {
    int64_t cap = 1;
    for (int k = 0; k < d; ++k) {
      const int64_t n = int64_t(maxima[k]) * (m + 1) + 1;
      if (n > 100000 / cap) return PHX_MC_RANGE_ERROR;
      cap *= n;
    }
    if (caps[m] + cap > 100000) return PHX_MC_RANGE_ERROR;
    caps[m + 1] = caps[m] + cap;
  }
  std::vector<int32_t> indices(caps.back() * d);
  std::vector<uint8_t> values(caps.back() * width);
  std::vector<int64_t> counts(d);
  int32_t status = 0;
  int32_t call = phx_mc_symmetric_polynomial_minors(
      1, d, local.data(), exponents + first * d, coefficients + first * width, caps.data(),
      indices.data(), values.data(), counts.data(), &status, limbs, 1);
  if (call != PHX_MC_OK) return call;
  if (status != PHX_MC_OK) return status;
  bool negative = false;
  Big q = integer(scales + row * width, width, negative);
  if (negative || q.empty()) return PHX_MC_INVALID_ARGUMENT;
  Big base = multiply(q, Big{2}), scale{1};
  std::array<std::array<uint8_t, 32>, 3> hashes{};
  std::array<int32_t, 3> positivity{};
  for (int m = 0; m < d; ++m) {
    scale = multiply(scale, base);
    std::array<int32_t, 3> degree{};
    int64_t nc = 0;
    if (!control_shape(d, domain, indices.data() + caps[m] * d, counts[m], degree, nc))
      return PHX_MC_RANGE_ERROR;
    std::vector<uint8_t> controls(nc * width);
    int64_t term_offsets[2] = {0, counts[m]}, control_offsets[2] = {0, nc};
    int64_t den = 0, extremes[2]{};
    call = phx_mc_bernstein_controls(
        1, domain, d, degree.data(), term_offsets, indices.data() + caps[m] * d,
        values.data() + caps[m] * width, control_offsets, controls.data(), &den, extremes,
        &status, limbs, 1);
    if (call != PHX_MC_OK) return call;
    if (status != PHX_MC_OK) return status;
    bool least_negative = false;
    Big least = integer(controls.data() + extremes[0] * width, width, least_negative);
    positivity[m] = !least_negative && !least.empty() ? 1 : 0;
    Big full_scale = multiply(scale, Big{uint32_t(den), uint32_t(uint64_t(den) >> 32)});
    trim(full_scale);
    hashes[m] = control_hash(controls, width, nc, full_scale);
  }
  for (int m = 0; m < d; ++m) {
    std::memcpy(digests + (row * d + m) * 32, hashes[m].data(), 32);
    verdict[m] = positivity[m];
  }
  return PHX_MC_OK;
}
}  // namespace

extern "C" PHX_MC_API int32_t phx_mc_symmetric_polynomial_verdicts(
    int64_t count, int32_t dimension, int32_t domain, const int64_t* offsets,
    const int32_t* exponents, const uint8_t* coefficients, const uint8_t* scales,
    uint8_t* digests, int32_t* verdicts, int32_t* statuses, int32_t limbs, int32_t workers) {
  return phx::mc::guarded([&]() -> int32_t {
    if (count < 0 || count > (INT64_MAX - 1) / 9 || dimension < 1 || dimension > 3 ||
        domain < 0 || domain > 2 || (domain == 2 && dimension != 3) || workers < 1 ||
        (limbs != 4 && limbs != 8 && limbs != 16 && limbs != 32 && limbs != 64))
      return PHX_MC_INVALID_ARGUMENT;
    if (count == 0) return PHX_MC_OK;
    if (!offsets || !exponents || !coefficients || !scales || !digests || !verdicts ||
        !statuses || offsets[0] != 0 ||
        !phx::mc::addressable(count * dimension * dimension + 1, 1, sizeof(int64_t)) ||
        !phx::mc::addressable(count, limbs * 8, 1) ||
        !phx::mc::addressable(count, dimension * 32, 1))
      return PHX_MC_INVALID_ARGUMENT;
    const int64_t entries = count * dimension * dimension;
    for (int64_t i = 0; i < entries; ++i) {
      if (offsets[i] < 0 || offsets[i] > offsets[i + 1]) return PHX_MC_INVALID_ARGUMENT;
    }
    if (!phx::mc::addressable(offsets[entries], dimension, sizeof(int32_t)) ||
        !phx::mc::addressable(offsets[entries], limbs * 8, 1))
      return PHX_MC_INVALID_ARGUMENT;
    for (int64_t i = 0; i < offsets[entries] * dimension; ++i)
      if (exponents[i] < 0) return PHX_MC_INVALID_ARGUMENT;
    return phx::mc::parallel_for(
        size_t(count), {16, uint32_t(workers)},
        [&](size_t first, size_t last, phx::mc::ChunkContext&) {
          for (size_t i = first; i < last; ++i)
            statuses[i] = verdict_item(int64_t(i), dimension, domain, offsets, exponents,
                                       coefficients, scales, digests, &verdicts[i * dimension], limbs);
        });
  });
}
