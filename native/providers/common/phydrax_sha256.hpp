// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// FIPS 180-4 SHA-256 for exchange payload checksums (no external dependency).
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace phydrax::sha256 {

class Hasher {
 public:
  void update(void const* data, std::size_t size) {
    auto const* bytes = static_cast<unsigned char const*>(data);
    total_ += size;
    while (size > 0) {
      std::size_t const take = std::min<std::size_t>(size, 64 - filled_);
      std::memcpy(block_.data() + filled_, bytes, take);
      filled_ += take;
      bytes += take;
      size -= take;
      if (filled_ == 64) {
        compress(block_.data());
        filled_ = 0;
      }
    }
  }

  std::string hexdigest() {
    std::uint64_t const bits = total_ * 8;
    unsigned char const one = 0x80;
    update(&one, 1);
    unsigned char const zero = 0;
    while (filled_ != 56) update(&zero, 1);
    unsigned char length[8];
    for (int index = 0; index < 8; ++index)
      length[index] = static_cast<unsigned char>(bits >> (56 - 8 * index));
    update(length, 8);
    static char const digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (std::uint32_t word : state_)
      for (int shift = 28; shift >= 0; shift -= 4)
        out.push_back(digits[(word >> shift) & 0xf]);
    return out;
  }

 private:
  static std::uint32_t rotate(std::uint32_t value, int count) {
    return (value >> count) | (value << (32 - count));
  }

  void compress(unsigned char const* chunk) {
    static constexpr std::uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    std::uint32_t w[64];
    for (int index = 0; index < 16; ++index)
      w[index] = (static_cast<std::uint32_t>(chunk[4 * index]) << 24) |
                 (static_cast<std::uint32_t>(chunk[4 * index + 1]) << 16) |
                 (static_cast<std::uint32_t>(chunk[4 * index + 2]) << 8) |
                 static_cast<std::uint32_t>(chunk[4 * index + 3]);
    for (int index = 16; index < 64; ++index) {
      std::uint32_t const s0 = rotate(w[index - 15], 7) ^ rotate(w[index - 15], 18) ^
                               (w[index - 15] >> 3);
      std::uint32_t const s1 = rotate(w[index - 2], 17) ^ rotate(w[index - 2], 19) ^
                               (w[index - 2] >> 10);
      w[index] = w[index - 16] + s0 + w[index - 7] + s1;
    }
    std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (int index = 0; index < 64; ++index) {
      std::uint32_t const s1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
      std::uint32_t const choice = (e & f) ^ (~e & g);
      std::uint32_t const first = h + s1 + choice + k[index] + w[index];
      std::uint32_t const s0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
      std::uint32_t const majority = (a & b) ^ (a & c) ^ (b & c);
      std::uint32_t const second = s0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + first;
      d = c;
      c = b;
      b = a;
      a = first + second;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                         0xa54ff53a, 0x510e527f, 0x9b05688c,
                                         0x1f83d9ab, 0x5be0cd19};
  std::array<unsigned char, 64> block_{};
  std::size_t filled_ = 0;
  std::uint64_t total_ = 0;
};

inline std::string hexdigest(void const* data, std::size_t size) {
  Hasher hasher;
  hasher.update(data, size);
  return hasher.hexdigest();
}

}  // namespace phydrax::sha256
