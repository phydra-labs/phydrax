// Copyright © 2026 Phydra, Inc. All rights reserved.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>

#include "phydrax_meshcore.h"

#if defined(__clang__)
#define PHX_MC_SIMD _Pragma("clang loop vectorize(enable)")
#elif defined(__GNUC__)
#define PHX_MC_SIMD _Pragma("omp simd")
#else
#define PHX_MC_SIMD
#endif

namespace phx::mc {

struct ParallelRequest {
  std::size_t chunk_size = 64;
  uint32_t parallelism = 1;
};

struct ChunkContext {
  std::size_t chunk_index;
};

namespace parallel_detail {
using BorrowedBody = int32_t (*)(const void*, std::size_t, std::size_t, ChunkContext&);
int32_t parallel_for_erased(std::size_t count, ParallelRequest request, const void* context,
                            BorrowedBody body);
}  // namespace parallel_detail

// Each chunk reads immutable inputs and writes disjoint indexed output slots.
// Reduction/merge code must fold chunk partials in increasing chunk_index order.
// All chunks complete even when one reports an expected non-OK status. Unexpected
// exceptions are rethrown by the lowest chunk index after every participant joins.
// Top-level regions share one process-wide lease, including single-participant calls;
// concurrent top-level regions wait, and nested regions do not reacquire the lease.
// Worker threads are created lazily, reused across regions, and joined at teardown.
// A forked child refuses an inherited pool. Nested regions run on their calling thread.
template <class Body>
int32_t parallel_for(std::size_t count, ParallelRequest request, Body&& body) {
  using ImmutableBody = const std::remove_reference_t<Body>&;
  static_assert(std::is_invocable_v<ImmutableBody, std::size_t, std::size_t, ChunkContext&>,
                "parallel_for.body must be callable through an immutable reference");
  const auto borrowed = [&body](std::size_t begin, std::size_t end,
                                ChunkContext& chunk) -> int32_t {
    using Result = std::invoke_result_t<ImmutableBody, std::size_t, std::size_t, ChunkContext&>;
    if constexpr (std::is_void_v<Result>) {
      std::invoke(static_cast<ImmutableBody>(body), begin, end, chunk);
      return PHX_MC_OK;
    } else {
      static_assert(std::is_convertible_v<Result, int32_t>,
                    "parallel_for.body must return void or an int32_t status");
      return std::invoke(static_cast<ImmutableBody>(body), begin, end, chunk);
    }
  };
  return parallel_detail::parallel_for_erased(
      count, request, std::addressof(borrowed),
      [](const void* context, std::size_t begin, std::size_t end, ChunkContext& chunk) -> int32_t {
        return (*static_cast<const decltype(borrowed)*>(context))(begin, end, chunk);
      });
}

}  // namespace phx::mc
