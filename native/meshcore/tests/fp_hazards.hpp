//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Hazardous caller floating-point modes for C API tests: every certified entry
// point must give its round-to-nearest, gradual-underflow result under them and
// leave the caller's mode untouched.
#pragma once

#include <array>
#include <cfenv>

#include "check.hpp"
#include "floating_point_mode.hpp"

namespace phx::mc::test {

struct FloatingPointHazard {
  bool flush;
  int rounding;
};

inline constexpr std::array<FloatingPointHazard, 5> kFloatingPointHazards{
    {{true, FE_TONEAREST},
     {false, FE_UPWARD},
     {false, FE_DOWNWARD},
     {false, FE_TOWARDZERO},
     {true, FE_DOWNWARD}}};

// Calls `call(hazard)` once per hazard with that mode installed on the calling
// thread, checks that the call preserves the installed mode, then restores the
// initial mode.
template <class Call>
void under_floating_point_hazards(Call&& call) {
  const FloatingPointMode initial_mode = floating_point_mode();
  const int initial_rounding = std::fegetround();
  for (const FloatingPointHazard& hazard : kFloatingPointHazards) {
    if (hazard.flush && !kFlushControl) continue;
    set_floating_point_mode(hazard.flush ? kFlushingMode : kGradualNearestMode);
    PHX_CHECK(std::fesetround(hazard.rounding) == 0);
    const FloatingPointMode installed = floating_point_mode();
    call(hazard);
    PHX_CHECK(floating_point_mode() == installed);
    PHX_CHECK(std::fegetround() == hazard.rounding);
  }
  set_floating_point_mode(initial_mode);
  PHX_CHECK(std::fesetround(initial_rounding) == 0);
  PHX_CHECK(floating_point_mode() == initial_mode);
}

}  // namespace phx::mc::test
