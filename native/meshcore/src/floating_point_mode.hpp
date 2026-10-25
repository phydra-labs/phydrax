//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Thread-local binary64 arithmetic mode for kernels whose outward rounding or
// error-free transformations require IEEE round-to-nearest with gradual
// underflow. A calling thread may run with flush-to-zero, denormals-are-zero
// (flushed operands) or directed rounding, left by a compiler runtime or a
// fast-math library. A scope saves only those mode fields, selects
// round-to-nearest with gradual underflow, and restores the saved fields on
// destruction, including unwinding; exception masks and flags are untouched.
// The mode is per thread, so every thread that evaluates kernel arithmetic
// needs its own scope. gradual_nearest_arithmetic() probes the arithmetic the
// thread actually performs: a kernel refuses instead of computing whenever the
// mode cannot be established (for example, where no flush control exists).
#pragma once

#include <atomic>
#include <bit>
#include <cstdint>

#if defined(_MSC_VER)
#include <float.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#elif !defined(__aarch64__)
#include <cfenv>
#endif

namespace phx::mc {

#if defined(_MSC_VER)
using FloatingPointMode = unsigned int;
// CRT denormal control (operands and results) and rounding control.
inline constexpr FloatingPointMode kFloatingPointModeFields = _MCW_DN | _MCW_RC;
inline constexpr FloatingPointMode kGradualNearestMode = _DN_SAVE | _RC_NEAR;
inline constexpr FloatingPointMode kFlushingMode = _DN_FLUSH | _RC_NEAR;
inline constexpr bool kFlushControl = true;
namespace floating_point_detail {
inline FloatingPointMode read() noexcept {
  unsigned int control = 0;
  _controlfp_s(&control, 0, 0);
  return control & kFloatingPointModeFields;
}
inline void write(FloatingPointMode mode) noexcept {
  unsigned int control = 0;
  _controlfp_s(&control, mode, kFloatingPointModeFields);
}
}  // namespace floating_point_detail
#elif defined(__x86_64__) || defined(__i386__)
using FloatingPointMode = unsigned int;
// MXCSR denormals-are-zero (bit 6), rounding control (bits 13-14) and
// flush-to-zero (bit 15).
inline constexpr FloatingPointMode kFloatingPointModeFields = (1u << 6) | (3u << 13) | (1u << 15);
inline constexpr FloatingPointMode kGradualNearestMode = 0;
inline constexpr FloatingPointMode kFlushingMode = (1u << 6) | (1u << 15);
inline constexpr bool kFlushControl = true;
namespace floating_point_detail {
inline FloatingPointMode read() noexcept { return _mm_getcsr() & kFloatingPointModeFields; }
inline void write(FloatingPointMode mode) noexcept {
  _mm_setcsr((_mm_getcsr() & ~kFloatingPointModeFields) | (mode & kFloatingPointModeFields));
}
}  // namespace floating_point_detail
#elif defined(__aarch64__)
using FloatingPointMode = std::uint64_t;
// FPCR flushed inputs FIZ (bit 0, FEAT_AFP; RES0 otherwise), rounding RMode
// (bits 22-23) and flush-to-zero FZ (bit 24).
inline constexpr FloatingPointMode kFloatingPointModeFields =
    (std::uint64_t{1} << 0) | (std::uint64_t{3} << 22) | (std::uint64_t{1} << 24);
inline constexpr FloatingPointMode kGradualNearestMode = 0;
inline constexpr FloatingPointMode kFlushingMode = std::uint64_t{1} << 24;
inline constexpr bool kFlushControl = true;
namespace floating_point_detail {
inline FloatingPointMode read() noexcept {
  std::uint64_t control = 0;
  __asm__ __volatile__("mrs %0, fpcr" : "=r"(control) : : "memory");
  return control & kFloatingPointModeFields;
}
inline void write(FloatingPointMode mode) noexcept {
  std::uint64_t control = 0;
  __asm__ __volatile__("mrs %0, fpcr" : "=r"(control) : : "memory");
  control = (control & ~kFloatingPointModeFields) | (mode & kFloatingPointModeFields);
  __asm__ __volatile__("msr fpcr, %0" : : "r"(control) : "memory");
}
}  // namespace floating_point_detail
#else
// Only the rounding direction is portable; the probe detects any flushing.
using FloatingPointMode = int;
inline constexpr FloatingPointMode kGradualNearestMode = FE_TONEAREST;
inline constexpr FloatingPointMode kFlushingMode = FE_TONEAREST;
inline constexpr bool kFlushControl = false;
namespace floating_point_detail {
inline FloatingPointMode read() noexcept { return std::fegetround(); }
inline void write(FloatingPointMode mode) noexcept { std::fesetround(mode); }
}  // namespace floating_point_detail
#endif

// The calling thread's mode fields.
inline FloatingPointMode floating_point_mode() noexcept { return floating_point_detail::read(); }

// Replaces the calling thread's mode fields. Compiler fences keep memory
// accesses, hence the arithmetic on loaded operands, on their side of the
// switch.
inline void set_floating_point_mode(FloatingPointMode mode) noexcept {
  std::atomic_signal_fence(std::memory_order_seq_cst);
  floating_point_detail::write(mode);
  std::atomic_signal_fence(std::memory_order_seq_cst);
}

// Whether this thread performs round-to-nearest in both directions (0.75 ulp
// above one rounds away from one for either sign), keeps a subnormal product
// (no flush-to-zero) and adds subnormal operands (no denormals-are-zero).
// Bit patterns are compared, so flushed comparison operands cannot mask it.
inline bool gradual_nearest_arithmetic() noexcept {
  volatile double one = 1.0;
  volatile double three_quarters_ulp = 0x1.8p-53;
  volatile double least_normal = 0x1p-1022;
  volatile double half = 0.5;
  volatile double least_subnormal = 0x1p-1074;
  const double above = one + three_quarters_ulp;
  const double below = -one - three_quarters_ulp;
  const double product = least_normal * half;
  const double sum = least_subnormal + least_subnormal;
  return std::bit_cast<std::uint64_t>(above) == std::bit_cast<std::uint64_t>(0x1.0000000000001p0) &&
         std::bit_cast<std::uint64_t>(below) == std::bit_cast<std::uint64_t>(-0x1.0000000000001p0) &&
         std::bit_cast<std::uint64_t>(product) == std::bit_cast<std::uint64_t>(0x1p-1023) &&
         std::bit_cast<std::uint64_t>(sum) == std::bit_cast<std::uint64_t>(0x1p-1073);
}

// Round-to-nearest with gradual underflow on the calling thread for the
// lifetime of the scope; the saved mode fields are restored on destruction.
class GradualNearestScope final {
 public:
  GradualNearestScope() noexcept : saved_(floating_point_mode()) {
    if (saved_ != kGradualNearestMode) set_floating_point_mode(kGradualNearestMode);
  }
  ~GradualNearestScope() {
    if (saved_ != kGradualNearestMode) set_floating_point_mode(saved_);
  }
  GradualNearestScope(const GradualNearestScope&) = delete;
  GradualNearestScope& operator=(const GradualNearestScope&) = delete;

  // Whether this thread now performs the arithmetic the scope selects.
  bool established() const noexcept { return gradual_nearest_arithmetic(); }

 private:
  FloatingPointMode saved_;
};

}  // namespace phx::mc
