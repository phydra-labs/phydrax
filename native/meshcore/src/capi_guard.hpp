//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Boundary invariants of the C ABI: no C++ exception crosses an extern "C"
// entry point, and every row offset of a caller array is representable.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>
#include "phydrax_meshcore.h"

namespace phx::mc {

// Runs one entry-point body without allowing a C++ exception to cross the C ABI.
// Allocation failure is distinct from insufficient caller output storage.
template <class Body>
int32_t guarded(Body&& body) noexcept {
  try {
    return body();
  } catch (const std::bad_alloc&) {
    return PHX_MC_ALLOCATION_FAILURE;
  } catch (const std::invalid_argument&) {
    return PHX_MC_INVALID_ARGUMENT;
  } catch (const std::length_error&) {
    return PHX_MC_INDEX_RANGE_ERROR;
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

template <class T>
bool output_bank_valid(const std::vector<T>& bank, const T* output) noexcept {
  return bank.empty() || output != nullptr;
}

template <class T>
void copy_output_bank(const std::vector<T>& bank, T* output) {
  if (!bank.empty()) std::copy(bank.begin(), bank.end(), output);
}

}  // namespace phx::mc
