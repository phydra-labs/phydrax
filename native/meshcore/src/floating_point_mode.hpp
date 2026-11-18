// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Thread-local binary64 mode for kernels requiring round-to-nearest with
// gradual underflow. The scope restores the caller's mode on every exit path.
// A scope opened inside a scope of the same thread that has verified the
// arithmetic, with no mode write in between, relies on that verification.
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
// MXCSR denormals-are-zero (bit 6), rounding control (bits 13-14), FTZ (bit 15).
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
// FPCR flushed inputs FIZ (bit 0) and alternate handling AH (bit 1), both RES0 without
// FEAT_AFP; rounding RMode (bits 22-23); FZ (bit 24).
inline constexpr FloatingPointMode kFloatingPointModeFields =
    (std::uint64_t{1} << 0) | (std::uint64_t{1} << 1) | (std::uint64_t{3} << 22) |
    (std::uint64_t{1} << 24);
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
using FloatingPointMode = int;
inline constexpr FloatingPointMode kGradualNearestMode = FE_TONEAREST;
inline constexpr FloatingPointMode kFlushingMode = FE_TONEAREST;
inline constexpr bool kFlushControl = false;
namespace floating_point_detail {
inline FloatingPointMode read() noexcept { return std::fegetround(); }
inline void write(FloatingPointMode mode) noexcept { std::fesetround(mode); }
}  // namespace floating_point_detail
#endif

inline FloatingPointMode floating_point_mode() noexcept { return floating_point_detail::read(); }

namespace floating_point_detail {
// Mode writes of this thread through set_floating_point_mode, and the count
// at which the innermost live scope verified its arithmetic (0: none has).
inline thread_local std::uint64_t mode_writes = 1;
inline thread_local std::uint64_t verified_at = 0;
}  // namespace floating_point_detail

inline void set_floating_point_mode(FloatingPointMode mode) noexcept {
  std::atomic_signal_fence(std::memory_order_seq_cst);
  floating_point_detail::write(mode);
  ++floating_point_detail::mode_writes;
  std::atomic_signal_fence(std::memory_order_seq_cst);
}

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

// Establishes round-to-nearest with gradual underflow for its lifetime, and
// `established` verifies it by arithmetic. The verification holds until the
// thread's next mode write: the kernels write the mode only through
// set_floating_point_mode and run no foreign code inside a scope (the
// parallel callback entry holds none). A scope opened under a verified one
// with no write since is nested: it neither reads the control register nor
// repeats the check, whose subnormal operations cost a microcode assist each
// on common processors, once for every interval program a kernel executes.
class GradualNearestScope final {
 public:
  GradualNearestScope() noexcept
      : nested_(floating_point_detail::verified_at == floating_point_detail::mode_writes),
        saved_(nested_ ? kGradualNearestMode : floating_point_mode()) {
    if (saved_ != kGradualNearestMode) set_floating_point_mode(kGradualNearestMode);
  }
  ~GradualNearestScope() {
    if (verified_) floating_point_detail::verified_at = enclosing_;
    if (saved_ != kGradualNearestMode) set_floating_point_mode(saved_);
  }
  GradualNearestScope(const GradualNearestScope&) = delete;
  GradualNearestScope& operator=(const GradualNearestScope&) = delete;

  bool established() const noexcept {
    if (nested_ || verified_) return true;
    if (!gradual_nearest_arithmetic()) return false;
    enclosing_ = floating_point_detail::verified_at;
    floating_point_detail::verified_at = floating_point_detail::mode_writes;
    verified_ = true;
    return true;
  }

 private:
  bool nested_;
  FloatingPointMode saved_;
  mutable bool verified_ = false;
  mutable std::uint64_t enclosing_ = 0;
};

}  // namespace phx::mc
