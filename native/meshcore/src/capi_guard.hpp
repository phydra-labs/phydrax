//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Boundary invariants of the C ABI: no C++ exception crosses an extern "C"
// entry point, and every row offset of a caller array is representable.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>

#include "phydrax_meshcore.h"
#include "bounded_memory.hpp"

namespace phx::mc {

// Runs one entry-point body and maps an escaping exception to its call status:
// allocation failure is a work-limit refusal, a rejected argument is invalid,
// and anything else is an internal failure.
template <class Body>
int32_t guarded(Body&& body) noexcept {
  try {
    return native_execution_status(body());
  } catch (const ExecutionRefusal& refusal) {
    return refusal.status;
  } catch (const std::bad_alloc&) {
    return native_execution_status(PHX_MC_CAPACITY_EXCEEDED);
  } catch (const std::invalid_argument&) {
    return PHX_MC_INVALID_ARGUMENT;
  } catch (...) {
    return PHX_MC_INTERNAL_ERROR;
  }
}

// Whether `count` rows of `width` items of `item_bytes` bytes each can form one
// addressable array, so that every `row * width` offset is exact.
inline bool addressable(int64_t count, int64_t width, std::size_t item_bytes) noexcept {
  if (count < 0 || width < 0) {
    return false;
  }
  if (count == 0 || width == 0) {
    return true;
  }
  const int64_t items =
      static_cast<int64_t>(static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
                           item_bytes);
  return count <= items / width;
}

}  // namespace phx::mc
