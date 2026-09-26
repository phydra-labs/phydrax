//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Minimal assertion helpers for the meshcore ctest executables.
#pragma once

#include <cstdio>
#include <cstdlib>

namespace phx::mc::test {

inline int& failure_count() {
  static int failures = 0;
  return failures;
}

inline void record_failure(const char* file, int line, const char* expression) {
  ++failure_count();
  std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, expression);
}

inline int finish(const char* name) {
  if (failure_count() != 0) {
    std::fprintf(stderr, "%s: %d check(s) failed\n", name, failure_count());
    return EXIT_FAILURE;
  }
  std::printf("%s: all checks passed\n", name);
  return EXIT_SUCCESS;
}

}  // namespace phx::mc::test

#define PHX_CHECK(expression)                                                \
  do {                                                                       \
    if (!(expression)) {                                                     \
      ::phx::mc::test::record_failure(__FILE__, __LINE__, #expression);      \
    }                                                                        \
  } while (false)

#define PHX_CHECK_NEAR(actual, expected, tolerance)                          \
  do {                                                                       \
    const double phx_actual_ = (actual);                                     \
    const double phx_expected_ = (expected);                                 \
    const double phx_delta_ = phx_actual_ - phx_expected_;                   \
    if (!(phx_delta_ <= (tolerance) && -phx_delta_ <= (tolerance))) {        \
      std::fprintf(stderr, "  actual=%.17g expected=%.17g\n", phx_actual_,   \
                   phx_expected_);                                           \
      ::phx::mc::test::record_failure(__FILE__, __LINE__,                    \
                                      #actual " ~= " #expected);             \
    }                                                                        \
  } while (false)
