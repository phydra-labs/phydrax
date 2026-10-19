//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>

#include "bounded_memory.hpp"

// These pragmas apply only to independent elementwise loops, never reductions.
// GCC requires -fopenmp-simd, which does not introduce an OpenMP runtime.
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
  NativeChunkAllowance per_item{};
  // bytes in per_item must be zero: scratch is one proved chunk-lifetime
  // bound, including allocator headers/alignment, not an item-count multiple.
  uint64_t arena_bytes_per_chunk = 0;
  uint32_t parallelism = 1;
  // Bounds waiting for the pool; a scoped policy and parent execution also
  // impose their absolute remaining deadline on the region.
  double lease_seconds = INFINITY;
};

struct ChunkContext {
  std::size_t chunk_index;
  NativeExecutionScope& execution;
  std::pmr::memory_resource& arena;
};

// Scoped policy owns an absolute deadline; nested policies cannot extend it.
// Workers copy the thread count but suppress deadline inheritance for a chunk:
// only the region's copied deadline is polled, at fixed logical boundaries.
class NativeParallelPolicy final {
 public:
  explicit NativeParallelPolicy(uint32_t parallelism, double lease_seconds = INFINITY);
  ~NativeParallelPolicy();
  NativeParallelPolicy(const NativeParallelPolicy&) = delete;
  NativeParallelPolicy& operator=(const NativeParallelPolicy&) = delete;

 private:
  uint32_t previous_parallelism_;
  std::chrono::steady_clock::time_point previous_deadline_;
};

uint32_t current_parallelism() noexcept;
double current_parallel_lease_seconds() noexcept;

namespace parallel_detail {
using BorrowedBody = int32_t (*)(const void*, std::size_t, std::size_t, ChunkContext&);
int32_t parallel_for_erased(std::size_t count, ParallelRequest request, const void* context,
                            BorrowedBody body);
}  // namespace parallel_detail

// Boundaries and child allowances depend only on count and chunk_size. A body
// reads immutable inputs and writes disjoint indexed output slots. The caller
// owns any reduction and folds chunk partials in increasing chunk_index order.
// Expected statuses do not cancel sibling chunks. Unexpected exceptions are
// rethrown by lowest chunk index only after every admitted chunk has joined.
// A deadline is the sole source of cancellation and is checked at boundaries.
// No output from an unexpected failure may be accepted as scientific evidence.
// Every positive top-level region takes the single process-wide pool lease,
// including serial requests. Nested regions bypass the lease and run serially.
// Threads are created lazily and reused. Fork-after-pool calls throw before
// touching inherited locks, including otherwise-empty and serial calls.
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
