// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// A cache prefetch hint on every supported compiler; it never changes results.
#pragma once

#if defined(_MSC_VER) && !defined(__clang__) && (defined(_M_X64) || defined(_M_IX86))
#include <xmmintrin.h>
#endif

namespace phx::mc {

inline void prefetch(const void* address) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(address);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  _mm_prefetch(static_cast<const char*>(address), _MM_HINT_T0);
#else
  static_cast<void>(address);
#endif
}

}  // namespace phx::mc
