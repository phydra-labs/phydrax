// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Rigorous binary64 interval enclosures of exp, log, sin, cos, tan, atan, tanh,
// asin and acos without a platform libm: binary64 basic operations and sqrt in
// round-to-nearest without contraction, correctly rounded std::fma, and
// double-word (DW)
// arithmetic with published error bounds. Every function requires the
// GradualNearestScope of floating_point_mode.hpp (RN, gradual underflow); the
// error proofs live beside each kernel in elementary.cpp.
#pragma once

#include <cstdint>

namespace phx::mc::elementary {

// Selector values equal the PHX_MC_ELEMENTARY_* codes of the C API.
enum class Function : std::int32_t {
  kExp = 0,
  kLog = 1,
  kSin = 2,
  kCos = 3,
  kTan = 4,
  kAtan = 5,
  kTanh = 6,
  kAsin = 7,
  kAcos = 8,
};

struct Enclosure {
  double lower;
  double upper;
};

// Enclosure of f over [lower, upper] under the conventions documented at
// phx_mc_interval_elementary in phydrax_meshcore.h.
Enclosure exp(double lower, double upper) noexcept;
Enclosure log(double lower, double upper) noexcept;
Enclosure sin(double lower, double upper) noexcept;
Enclosure cos(double lower, double upper) noexcept;
Enclosure tan(double lower, double upper) noexcept;
Enclosure atan(double lower, double upper) noexcept;
Enclosure tanh(double lower, double upper) noexcept;
Enclosure asin(double lower, double upper) noexcept;
Enclosure acos(double lower, double upper) noexcept;
Enclosure evaluate(Function function, double lower, double upper) noexcept;

// Unevaluated sum high + low with high = RN(high + low).
struct DoubleWord {
  double high;
  double low;
};

// Double-word building blocks (Joldes, Muller, Popescu, ACM TOMS 44(2), 2017,
// "JMP"), exposed for the constant-consistency tests. u = 2^-53.
DoubleWord two_sum(double a, double b) noexcept;            // exact (Knuth)
DoubleWord two_product(double a, double b) noexcept;        // exact (fma), no underflow
DoubleWord add(DoubleWord x, double y) noexcept;            // JMP Alg. 4, <= 2u^2
DoubleWord add(DoubleWord x, DoubleWord y) noexcept;        // JMP Alg. 6, <= 3u^2 + 13u^3
DoubleWord multiply(DoubleWord x, double y) noexcept;       // JMP Alg. 9, <= 2u^2
DoubleWord multiply(DoubleWord x, DoubleWord y) noexcept;   // DWTimesDW (fma), < 5u^2
DoubleWord divide(DoubleWord n, DoubleWord d) noexcept;     // one correction, <= 11.01u^2

// Constants generated with mpmath at 1600 bits (generator recorded in
// elementary.cpp). Each DW pair is (RN(c), RN(c - RN(c))), so its relative
// representation error is at most 2^-106 |c| unless stated otherwise.
namespace constants {
// ln 2 = kLn2[0] + kLn2[1] + kLn2[2] + e with |e| < 2^-157.3; kLn2[0] has 42
// significant bits, so k * kLn2[0] is exact for |k| < 2^11 (Cody-Waite).
inline constexpr double kLn2[3] = {0x1.62e42fefa3800p-1, 0x1.ef35793c76730p-45,
                                   0x1.f97b57a079a19p-103};
// pi / 2 = high + low + e with |e| < 2^-109.04.
inline constexpr DoubleWord kHalfPi = {0x1.921fb54442d18p+0, 0x1.1a62633145c07p-54};
// 2 / pi = sum_i kTwoOverPi[i] 2^(-64 (i + 1)) + e with 0 <= e < 2^-1280.
inline constexpr std::uint64_t kTwoOverPi[20] = {
    0xA2F9836E4E441529u, 0xFC2757D1F534DDC0u, 0xDB6295993C439041u, 0xFE5163ABDEBBC561u,
    0xB7246E3A424DD2E0u, 0x06492EEA09D1921Cu, 0xFE1DEB1CB129A73Eu, 0xE88235F52EBB4484u,
    0xE99C7026B45F7E41u, 0x3991D639835339F4u, 0x9C845F8BBDF9283Bu, 0x1FF897FFDE05980Fu,
    0xEF2F118B5A0A6D1Fu, 0x6D367ECF27CB09B7u, 0x4F463F669E5FEA2Du, 0x7527BAC7EBE5F17Bu,
    0x3D0739F78A5292EAu, 0x6BFB5FB11F8D5D08u, 0x56033046FC7B6BABu, 0xF0CFBC209AF4361Du,
};
// 1 / n! for n = 0..29.
inline constexpr DoubleWord kInverseFactorial[30] = {
    {0x1.0000000000000p+0, 0x0.0p+0},
    {0x1.0000000000000p+0, 0x0.0p+0},
    {0x1.0000000000000p-1, 0x0.0p+0},
    {0x1.5555555555555p-3, 0x1.5555555555555p-57},
    {0x1.5555555555555p-5, 0x1.5555555555555p-59},
    {0x1.1111111111111p-7, 0x1.1111111111111p-63},
    {0x1.6c16c16c16c17p-10, -0x1.f49f49f49f49fp-65},
    {0x1.a01a01a01a01ap-13, 0x1.a01a01a01a01ap-73},
    {0x1.a01a01a01a01ap-16, 0x1.a01a01a01a01ap-76},
    {0x1.71de3a556c734p-19, -0x1.c154f8ddc6c00p-73},
    {0x1.27e4fb7789f5cp-22, 0x1.cbbc05b4fa99ap-76},
    {0x1.ae64567f544e4p-26, -0x1.c062e06d1f209p-80},
    {0x1.1eed8eff8d898p-29, -0x1.2aec959e14c06p-83},
    {0x1.6124613a86d09p-33, 0x1.f28e0cc748ebep-87},
    {0x1.93974a8c07c9dp-37, 0x1.05d6f8a2efd1fp-92},
    {0x1.ae7f3e733b81fp-41, 0x1.1d8656b0ee8cbp-97},
    {0x1.ae7f3e733b81fp-45, 0x1.1d8656b0ee8cbp-101},
    {0x1.952c77030ad4ap-49, 0x1.ac981465ddc6cp-103},
    {0x1.6827863b97d97p-53, 0x1.eec01221a8b0bp-107},
    {0x1.2f49b46814157p-57, 0x1.2650f61dbdcb4p-112},
    {0x1.e542ba4020225p-62, 0x1.ea72b4afe3c2fp-120},
    {0x1.71b8ef6dcf572p-66, -0x1.d043ae40c4647p-120},
    {0x1.0ce396db7f853p-70, -0x1.aebcdbd20331cp-124},
    {0x1.761b41316381ap-75, -0x1.3423c7d91404fp-130},
    {0x1.f2cf01972f578p-80, -0x1.9ada5fcc1ab14p-135},
    {0x1.3f3ccdd165fa9p-84, -0x1.58ddadf344487p-139},
    {0x1.88e85fc6a4e5ap-89, -0x1.71c37ebd16540p-143},
    {0x1.d1ab1c2dccea3p-94, 0x1.054d0c78aea14p-149},
    {0x1.0a18a2635085dp-98, 0x1.b9e2e28e1aa54p-153},
    {0x1.259f98b4358adp-103, 0x1.eaf8c39dd9bc5p-157},
};
// 1 / (2n + 1) for n = 0..19.
inline constexpr DoubleWord kInverseOdd[20] = {
    {0x1.0000000000000p+0, 0x0.0p+0},
    {0x1.5555555555555p-2, 0x1.5555555555555p-56},
    {0x1.999999999999ap-3, -0x1.999999999999ap-57},
    {0x1.2492492492492p-3, 0x1.2492492492492p-57},
    {0x1.c71c71c71c71cp-4, 0x1.c71c71c71c71cp-58},
    {0x1.745d1745d1746p-4, -0x1.745d1745d1746p-59},
    {0x1.3b13b13b13b14p-4, -0x1.3b13b13b13b14p-58},
    {0x1.1111111111111p-4, 0x1.1111111111111p-60},
    {0x1.e1e1e1e1e1e1ep-5, 0x1.e1e1e1e1e1e1ep-61},
    {0x1.af286bca1af28p-5, 0x1.af286bca1af28p-59},
    {0x1.8618618618618p-5, 0x1.8618618618618p-59},
    {0x1.642c8590b2164p-5, 0x1.642c8590b2164p-60},
    {0x1.47ae147ae147bp-5, -0x1.eb851eb851eb8p-61},
    {0x1.2f684bda12f68p-5, 0x1.2f684bda12f68p-59},
    {0x1.1a7b9611a7b96p-5, 0x1.1a7b9611a7b96p-61},
    {0x1.0842108421084p-5, 0x1.0842108421084p-60},
    {0x1.f07c1f07c1f08p-6, -0x1.f07c1f07c1f08p-61},
    {0x1.d41d41d41d41dp-6, 0x1.0750750750750p-60},
    {0x1.bacf914c1bad0p-6, -0x1.bacf914c1bad0p-60},
    {0x1.a41a41a41a41ap-6, 0x1.0690690690690p-60},
};
// atan(j / 16) for j = 0..16.
inline constexpr DoubleWord kArctangent[17] = {
    {0x0.0p+0, 0x0.0p+0},
    {0x1.ff55bb72cfdeap-5, -0x1.c934d86d23f1dp-60},
    {0x1.fd5ba9aac2f6ep-4, -0x1.cd37686760c17p-59},
    {0x1.7b97b4bce5b02p-3, 0x1.347b0b4f881cap-58},
    {0x1.f5b75f92c80ddp-3, 0x1.8ab6e3cf7afbdp-57},
    {0x1.362773707ebccp-2, -0x1.963a544b672d8p-57},
    {0x1.6f61941e4def1p-2, -0x1.c63aae6f6e918p-56},
    {0x1.a64eec3cc23fdp-2, -0x1.24dec1b50b7ffp-56},
    {0x1.dac670561bb4fp-2, 0x1.a2b7f222f65e2p-56},
    {0x1.0657e94db30d0p-1, -0x1.d5b495f6349e6p-56},
    {0x1.1e00babdefeb4p-1, -0x1.928df287a668fp-58},
    {0x1.345f01cce37bbp-1, 0x1.1021137c71102p-55},
    {0x1.4978fa3269ee1p-1, 0x1.2419a87f2a458p-56},
    {0x1.5d58987169b18p-1, 0x1.0028e4bc5e7cap-57},
    {0x1.700a7c5784634p-1, -0x1.8c34d25aadef6p-56},
    {0x1.819d0b7158a4dp-1, -0x1.bf76229d3b917p-56},
    {0x1.921fb54442d18p-1, 0x1.1a62633145c07p-55},
};
}  // namespace constants

}  // namespace phx::mc::elementary
