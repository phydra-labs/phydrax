//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory_resource>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "check.hpp"
#include "parallel.hpp"

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
using namespace phx::mc;
using Work = std::array<uint64_t, PHX_MC_EXECUTION_COUNTERS>;
using Memory = std::array<uint64_t, PHX_MC_EXECUTION_MEMORY_VALUES>;
constexpr std::size_t memory_limit = 32 * 1024 * 1024;

void notify(std::atomic<bool>& flag) {
  flag.store(true, std::memory_order_release);
  flag.notify_all();
}

void await(const std::atomic<bool>& flag) {
  while (!flag.load(std::memory_order_acquire)) flag.wait(false, std::memory_order_acquire);
}

struct IndexedFailure {
  std::size_t index;
};

#if !defined(_WIN32)
std::atomic<std::size_t> shutdown_completed{0};
pid_t test_parent_process = 0;
bool inherited_pool_teardown = false;
int inherited_pool_exit_status = EXIT_FAILURE;

void finish_inherited_pool_teardown() {
  if (getpid() != test_parent_process && inherited_pool_teardown) {
    // Registered before pool creation, so its static destructor has already
    // returned. LSan cannot census vanished parent threads or their live
    // reservations in this fork clone. Parent and independent-pool children
    // retain ordinary exit and full leak checking.
    std::_Exit(inherited_pool_exit_status);
  }
}
#endif

void test_static_shutdown_drains_active_callers() {
#if !defined(_WIN32)
  // The parent is still lazy; this child owns an independent pool.
  const pid_t child = fork();
  PHX_CHECK(child >= 0);
  if (child == 0) {
    alarm(5);
    // Registered before pool creation: this observer runs after pool teardown.
    if (std::atexit([] {
          if (shutdown_completed.load(std::memory_order_acquire) != 2) std::_Exit(EXIT_FAILURE);
        }) != 0)
      std::_Exit(EXIT_FAILURE);
    std::atomic<bool> entered{false};
    ParallelRequest request;
    request.parallelism = 2;
    request.chunk_size = 1;
    std::thread caller([&] {
      try {
        parallel_for(2, request, [&](std::size_t, std::size_t, ChunkContext&) {
          notify(entered);
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
          shutdown_completed.fetch_add(1, std::memory_order_release);
        });
      } catch (...) {
        std::_Exit(EXIT_FAILURE);
      }
    });
    caller.detach();
    await(entered);
    // Teardown must wait for both chunks and the admitted caller to release.
    std::exit(EXIT_SUCCESS);
  }
  if (child > 0) {
    int status = 0;
    PHX_CHECK(waitpid(child, &status, 0) == child);
    PHX_CHECK(WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS);
  }
#endif
}

void test_zero_count_and_lazy_fork() {
  bool called = false;
  ParallelRequest request;
  request.parallelism = 64;
  auto owner = std::make_shared<BoundedMemoryResource>(0);
  {
    NativeExecutionScope execution(0, 0, 0, 0, INFINITY, owner);
    PHX_CHECK(parallel_for(0, request, [&](std::size_t, std::size_t, ChunkContext&) {
                called = true;
              }) == PHX_MC_OK);
    PHX_CHECK(!called);
    PHX_CHECK(execution.evidence() == Work{});
    PHX_CHECK(!execution.has_live_batches());
    PHX_CHECK(owner->allocations() == 0);
  }
#if !defined(_WIN32)
  // An empty call must not prevent a child from creating its own pool.
  const pid_t child = fork();
  PHX_CHECK(child >= 0);
  if (child == 0) {
    std::array<int, 2> output{};
    request.chunk_size = 1;
    request.parallelism = 2;
    int result = EXIT_FAILURE;
    try {
      const int32_t status =
          parallel_for(2, request, [&](std::size_t begin, std::size_t end, ChunkContext&) {
            for (std::size_t i = begin; i < end; ++i) output[i] = static_cast<int>(i + 1);
          });
      if (status == PHX_MC_OK && output == std::array<int, 2>{1, 2}) result = EXIT_SUCCESS;
    } catch (...) {
      result = EXIT_FAILURE;
    }
    // This also exercises shutdown of the child's independently created pool.
    std::exit(result);
  }
  if (child > 0) {
    int status = 0;
    PHX_CHECK(waitpid(child, &status, 0) == child);
    PHX_CHECK(WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS);
  }
#endif
}

void test_policy_nesting_and_thread_locality() {
  PHX_CHECK(current_parallelism() == 1);
  PHX_CHECK(std::isinf(current_parallel_lease_seconds()));
  {
    NativeParallelPolicy outer(7, 0.5);
    const double outer_remaining = current_parallel_lease_seconds();
    PHX_CHECK(current_parallelism() == 7 && outer_remaining >= 0.0 && outer_remaining <= 0.5);
    {
      NativeParallelPolicy inner(2, 0.25);
      const double inner_remaining = current_parallel_lease_seconds();
      PHX_CHECK(current_parallelism() == 2 && inner_remaining >= 0.0 &&
                inner_remaining <= std::min(0.25, outer_remaining));
    }
    PHX_CHECK(current_parallelism() == 7 && current_parallel_lease_seconds() <= outer_remaining);
    {
      const double before = current_parallel_lease_seconds();
      NativeParallelPolicy longer(2, 5.0);
      PHX_CHECK(current_parallel_lease_seconds() <= before);
    }
    uint32_t thread_policy = 0;
    std::jthread thread([&] { thread_policy = current_parallelism(); });
    thread.join();
    PHX_CHECK(thread_policy == 1);
  }
  PHX_CHECK(current_parallelism() == 1);
  PHX_CHECK(std::isinf(current_parallel_lease_seconds()));
  bool rejected = false;
  try {
    NativeParallelPolicy invalid(0);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  PHX_CHECK(rejected && current_parallelism() == 1);
}

void test_detached_chunks_suppress_scoped_deadline_inheritance() {
  NativeParallelPolicy policy(2, 10.0);
  ParallelRequest request;
  request.parallelism = current_parallelism();
  request.lease_seconds = current_parallel_lease_seconds();
  request.chunk_size = 1;
  std::array<uint8_t, 2> detached{};
  PHX_CHECK(parallel_for(2, request, [&](std::size_t begin, std::size_t, ChunkContext&) {
              detached[begin] =
                  current_parallelism() == 2 && std::isinf(current_parallel_lease_seconds());
            }) == PHX_MC_OK);
  PHX_CHECK((detached == std::array<uint8_t, 2>{1, 1}));
  PHX_CHECK(!std::isinf(current_parallel_lease_seconds()) &&
            current_parallel_lease_seconds() <= request.lease_seconds);
}

void test_expired_policy_does_not_renew_for_each_region() {
  NativeParallelPolicy policy(2, 0.0);
  ParallelRequest request;
  request.parallelism = 2;
  request.chunk_size = 1;
  request.lease_seconds = current_parallel_lease_seconds();
  bool called = false;
  const auto body = [&](std::size_t, std::size_t, ChunkContext&) { called = true; };
  PHX_CHECK(parallel_for(2, request, body) == PHX_MC_TIMEOUT);
  PHX_CHECK(!called && current_parallel_lease_seconds() == 0.0);
  PHX_CHECK(parallel_for(2, request, body) == PHX_MC_TIMEOUT);
  PHX_CHECK(!called && current_parallel_lease_seconds() == 0.0);
}

void test_invalid_request_before_launch() {
  ParallelRequest request;
  auto body = [](std::size_t, std::size_t, ChunkContext&) { throw IndexedFailure{0}; };
  request.chunk_size = 0;
  PHX_CHECK(parallel_for(1, request, body) == PHX_MC_INVALID_ARGUMENT);
  request.chunk_size = 1;
  request.parallelism = 0;
  PHX_CHECK(parallel_for(1, request, body) == PHX_MC_INVALID_ARGUMENT);
  request.parallelism = 1;
  request.lease_seconds = -1.0;
  PHX_CHECK(parallel_for(1, request, body) == PHX_MC_INVALID_ARGUMENT);
  request.lease_seconds = std::numeric_limits<double>::quiet_NaN();
  PHX_CHECK(parallel_for(1, request, body) == PHX_MC_INVALID_ARGUMENT);
  request.lease_seconds = INFINITY;
  request.per_item.bytes = 1;
  PHX_CHECK(parallel_for(1, request, body) == PHX_MC_INVALID_ARGUMENT);
  request.per_item.bytes = 0;
  request.chunk_size = 2;
  request.per_item.work = std::numeric_limits<uint64_t>::max();
  PHX_CHECK(parallel_for(2, request, body) == PHX_MC_CAPACITY_EXCEEDED);
}

struct BatchResult {
  std::vector<uint64_t> output;
  std::vector<int32_t> item_status;
  std::vector<uint64_t> chunk_partials;
  uint64_t reduction = 0;
  Work counters{};
  Memory memory{};
  int32_t status = PHX_MC_OK;
};

BatchResult independent_batch(uint32_t threads) {
  constexpr std::size_t count = 773;
  constexpr std::size_t chunk_size = 11;
  constexpr std::size_t chunks = (count + chunk_size - 1) / chunk_size;
  BatchResult result;
  result.output.resize(count);
  result.item_status.resize(count);
  result.chunk_partials.resize(chunks);
  auto owner = std::make_shared<BoundedMemoryResource>(memory_limit);
  NativeExecutionScope execution(count * 10, count * 5, count * 3, memory_limit, INFINITY, owner);
  ParallelRequest request;
  request.chunk_size = chunk_size;
  request.parallelism = threads;
  request.per_item = {7, 4, 2, 0};
  request.arena_bytes_per_chunk =
      native_parallel_allocation_bytes(chunk_size * sizeof(double), alignof(double));
  result.status = parallel_for(
      count, request, [&](std::size_t begin, std::size_t end, ChunkContext& chunk) -> int32_t {
        std::pmr::vector<double> scratch(&chunk.arena);
        scratch.resize(end - begin);
        PHX_MC_SIMD
        for (std::size_t k = 0; k < end - begin; ++k) {
          scratch[k] = static_cast<double>(begin + k) * 0.25 - 7.0;
        }
        double partial = 0.0;
        int32_t chunk_status = PHX_MC_OK;
        for (std::size_t i = begin; i < end; ++i) {
          native_execution_charge(3, 1);
          native_execution_primitive_query();
          if (!chunk.execution.charge_external(2, 1) || !chunk.execution.admit_cavity(2)) {
            throw ExecutionRefusal{chunk.execution.status()};
          }
          const double value = i % 3 == 0 ? 1.0e16 : (i % 3 == 1 ? scratch[i - begin] : -1.0e16);
          result.output[i] = std::bit_cast<uint64_t>(value);
          partial += value;
          const int32_t status = i % 19 == 3 ? PHX_MC_NONFINITE_INPUT
                                             : (i % 17 == 2 ? PHX_MC_DEGENERATE_INPUT : PHX_MC_OK);
          result.item_status[i] = status;
          if (chunk_status == PHX_MC_OK) chunk_status = status;
        }
        result.chunk_partials[chunk.chunk_index] = std::bit_cast<uint64_t>(partial);
        return chunk_status;
      });
  double reduction = 0.0;
  for (uint64_t partial : result.chunk_partials) reduction += std::bit_cast<double>(partial);
  result.reduction = std::bit_cast<uint64_t>(reduction);
  result.counters = execution.evidence();
  result.memory = execution.memory_evidence();
  PHX_CHECK(!execution.has_live_batches());
  PHX_CHECK(owner->live_bytes() == 0);
  return result;
}

void test_outputs_statuses_counters_and_ordered_reduction() {
  const BatchResult reference = independent_batch(1);
  PHX_CHECK(reference.status == PHX_MC_DEGENERATE_INPUT);
  PHX_CHECK(reference.counters[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 773 * 5);
  PHX_CHECK(reference.counters[PHX_MC_EXECUTION_SOURCE_QUERIES] == 773 * 2);
  PHX_CHECK(reference.counters[PHX_MC_EXECUTION_EXTERNAL_WORK] == 773 * 2);
  PHX_CHECK(reference.counters[PHX_MC_EXECUTION_EXTERNAL_QUERIES] == 773);
  PHX_CHECK(reference.counters[PHX_MC_EXECUTION_PRIMITIVE_QUERIES] == 773);
  PHX_CHECK(reference.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0);
  PHX_CHECK(reference.memory[PHX_MC_EXECUTION_MEMORY_REFUSALS] == 0);
  for (std::size_t i = 0; i < reference.output.size(); ++i) {
    const double value =
        i % 3 == 0 ? 1.0e16 : (i % 3 == 1 ? static_cast<double>(i) * 0.25 - 7.0 : -1.0e16);
    const int32_t status =
        i % 19 == 3 ? PHX_MC_NONFINITE_INPUT : (i % 17 == 2 ? PHX_MC_DEGENERATE_INPUT : PHX_MC_OK);
    PHX_CHECK(reference.output[i] == std::bit_cast<uint64_t>(value));
    PHX_CHECK(reference.item_status[i] == status);
  }
  double ordered = 0.0;
  for (std::size_t begin = 0; begin < reference.output.size(); begin += 11) {
    double partial = 0.0;
    for (std::size_t i = begin; i < std::min(begin + 11, reference.output.size()); ++i) {
      partial += std::bit_cast<double>(reference.output[i]);
    }
    PHX_CHECK(reference.chunk_partials[begin / 11] == std::bit_cast<uint64_t>(partial));
    ordered += partial;
  }
  PHX_CHECK(reference.reduction == std::bit_cast<uint64_t>(ordered));
  for (uint32_t threads : {2u, 7u, 64u}) {
    const BatchResult result = independent_batch(threads);
    PHX_CHECK(result.output == reference.output);
    PHX_CHECK(result.item_status == reference.item_status);
    PHX_CHECK(result.chunk_partials == reference.chunk_partials);
    PHX_CHECK(result.reduction == reference.reduction);
    PHX_CHECK(result.counters == reference.counters);
    PHX_CHECK(result.memory == reference.memory);
    PHX_CHECK(result.status == reference.status);
  }
}

void test_workers_are_reused_across_regions() {
  ParallelRequest request;
  request.parallelism = 2;
  request.chunk_size = 1;
  std::barrier barrier(2);
  std::array<uint32_t, 2> visits{};
  const auto body = [&](std::size_t begin, std::size_t, ChunkContext&) {
    static thread_local uint32_t thread_visits = 0;
    visits[begin] = ++thread_visits;
    // Force the caller and a worker to participate in each region.
    barrier.arrive_and_wait();
  };
  PHX_CHECK(parallel_for(2, request, body) == PHX_MC_OK);
  PHX_CHECK((visits == std::array<uint32_t, 2>{1, 1}));
  PHX_CHECK(parallel_for(2, request, body) == PHX_MC_OK);
  PHX_CHECK((visits == std::array<uint32_t, 2>{2, 2}));
}

void test_recycled_row_scratch_has_one_bound_per_chunk() {
  constexpr std::size_t count = 193;
  constexpr std::size_t bytes = 64 * 1024;
  for (uint32_t threads : {1u, 2u, 7u, 64u}) {
    auto owner = std::make_shared<BoundedMemoryResource>(bytes);
    NativeExecutionScope execution(0, 0, 0, bytes, INFINITY, owner);
    ParallelRequest request;
    request.parallelism = threads;
    request.chunk_size = 64;
    request.arena_bytes_per_chunk =
        native_parallel_allocation_bytes(128 * sizeof(double), alignof(double));
    std::array<uint64_t, count> output{};
    PHX_CHECK(
        parallel_for(count, request, [&](std::size_t begin, std::size_t end, ChunkContext& chunk) {
          for (std::size_t i = begin; i < end; ++i) {
            std::pmr::vector<double> row(128, static_cast<double>(i), &chunk.arena);
            output[i] = std::bit_cast<uint64_t>(row[127]);
          }
        }) == PHX_MC_OK);
    for (std::size_t i = 0; i < count; ++i) {
      PHX_CHECK(output[i] == std::bit_cast<uint64_t>(static_cast<double>(i)));
    }
    PHX_CHECK(owner->peak_bytes() <= bytes);
    PHX_CHECK(owner->live_bytes() == 0 && !execution.has_live_batches());
  }
}

// Scratch arenas recycle per participant slot: many chunks admit a bounded
// number of arenas, independent of the requested parallelism, with identical
// outputs and execution evidence.
void test_recycled_arenas_scale_with_slots_not_chunks() {
  constexpr std::size_t count = 64 * 200;
  const uint64_t arena = native_parallel_allocation_bytes(1024 * sizeof(double), alignof(double));
  // Room for 64 slots, far below one arena per chunk (200 chunks).
  const std::size_t limit = static_cast<std::size_t>(80 * (arena + 4096));
  std::array<uint64_t, PHX_MC_EXECUTION_MEMORY_VALUES> reference_memory{};
  for (uint32_t threads : {1u, 8u, 32u}) {
    auto owner = std::make_shared<BoundedMemoryResource>(limit);
    NativeExecutionScope execution(0, 0, 0, limit, INFINITY, owner);
    ParallelRequest request;
    request.parallelism = threads;
    request.chunk_size = 64;
    request.arena_bytes_per_chunk = arena;
    std::vector<uint64_t> output(count);
    PHX_CHECK(
        parallel_for(count, request, [&](std::size_t begin, std::size_t end, ChunkContext& chunk) {
          std::pmr::vector<double> scratch(1024, 0.5, &chunk.arena);
          for (std::size_t i = begin; i < end; ++i) {
            output[i] = std::bit_cast<uint64_t>(scratch[i % 1024] * static_cast<double>(i));
          }
        }) == PHX_MC_OK);
    for (std::size_t i = 0; i < count; ++i) {
      PHX_CHECK(output[i] == std::bit_cast<uint64_t>(0.5 * static_cast<double>(i)));
    }
    const auto memory = execution.memory_evidence();
    if (threads == 1) reference_memory = memory;
    PHX_CHECK(memory == reference_memory);
    PHX_CHECK(owner->peak_bytes() <= limit);
    PHX_CHECK(owner->live_bytes() == 0 && !execution.has_live_batches());
  }
}

// A nested region runs serially and recycles one arena: more than 64 inner
// chunks fit an outer chunk arena that holds a single inner arena.
void test_nested_region_recycles_one_arena() {
  constexpr std::size_t outer_count = 3;
  constexpr std::size_t inner_count = 100;
  const uint64_t inner_arena =
      native_parallel_allocation_bytes(256 * sizeof(double), alignof(double));
  Work reference_work{};
  std::array<uint64_t, PHX_MC_EXECUTION_MEMORY_VALUES> reference_memory{};
  for (uint32_t threads : {1u, 3u}) {
    auto owner = std::make_shared<BoundedMemoryResource>(memory_limit);
    NativeExecutionScope execution(0, 0, 0, memory_limit, INFINITY, owner);
    ParallelRequest outer;
    outer.parallelism = threads;
    outer.chunk_size = 1;
    // Room for one inner arena and its storage, far below 64 of them.
    outer.arena_bytes_per_chunk = 2 * inner_arena + 4096;
    std::array<std::array<uint64_t, inner_count>, outer_count> output{};
    std::array<int32_t, outer_count> statuses{};
    PHX_CHECK(parallel_for(outer_count, outer, [&](std::size_t begin, std::size_t, ChunkContext&) {
                ParallelRequest inner;
                inner.parallelism = 64;
                inner.chunk_size = 1;
                inner.arena_bytes_per_chunk = inner_arena;
                statuses[begin] = parallel_for(
                    inner_count, inner, [&](std::size_t row, std::size_t, ChunkContext& nested) {
                      std::pmr::vector<double> scratch(256, 0.25, &nested.arena);
                      output[begin][row] =
                          std::bit_cast<uint64_t>(scratch[255] * static_cast<double>(row + begin));
                    });
              }) == PHX_MC_OK);
    for (std::size_t chunk = 0; chunk < outer_count; ++chunk) {
      PHX_CHECK(statuses[chunk] == PHX_MC_OK);
      for (std::size_t row = 0; row < inner_count; ++row) {
        PHX_CHECK(output[chunk][row] ==
                  std::bit_cast<uint64_t>(0.25 * static_cast<double>(row + chunk)));
      }
    }
    const Work work = execution.evidence();
    const auto memory = execution.memory_evidence();
    if (threads == 1) {
      reference_work = work;
      reference_memory = memory;
    }
    PHX_CHECK(work == reference_work && memory == reference_memory);
    PHX_CHECK(owner->live_bytes() == 0 && !execution.has_live_batches());
  }
}

void test_resource_admission_before_launch() {
  Work reference{};
  for (uint32_t threads : {1u, 2u, 7u, 64u}) {
    auto owner = std::make_shared<BoundedMemoryResource>(memory_limit);
    NativeExecutionScope execution(14, 100, 100, memory_limit, INFINITY, owner);
    ParallelRequest request;
    request.chunk_size = 2;
    request.parallelism = threads;
    request.per_item.work = 3;
    std::atomic<std::size_t> calls{0};
    PHX_CHECK(parallel_for(5, request, [&](std::size_t, std::size_t, ChunkContext&) {
                calls.fetch_add(1, std::memory_order_relaxed);
              }) == PHX_MC_CAPACITY_EXCEEDED);
    PHX_CHECK(calls.load() == 0);
    PHX_CHECK(!execution.has_live_batches());
    PHX_CHECK(owner->live_bytes() == 0);
    const Work counters = execution.evidence();
    PHX_CHECK(counters[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 0);
    PHX_CHECK(counters[PHX_MC_EXECUTION_WORK_REFUSALS] == 1);
    if (threads == 1)
      reference = counters;
    else
      PHX_CHECK(counters == reference);
  }
}

void test_unused_reservation_release_after_expected_failure() {
  auto owner = std::make_shared<BoundedMemoryResource>(memory_limit);
  NativeExecutionScope execution(20, 20, 20, memory_limit, INFINITY, owner);
  ParallelRequest request;
  request.chunk_size = 1;
  request.parallelism = 7;
  request.per_item = {5, 5, 5, 0};
  PHX_CHECK(parallel_for(4, request, [](std::size_t begin, std::size_t, ChunkContext&) -> int32_t {
              native_execution_charge(1, 1);
              return begin == 0 ? PHX_MC_DEGENERATE_INPUT : PHX_MC_OK;
            }) == PHX_MC_DEGENERATE_INPUT);
  const auto remaining = execution.remaining();
  PHX_CHECK(remaining[0] == 16 && remaining[1] == 16 && remaining[2] == 20);
  PHX_CHECK(remaining[3] == memory_limit);
  PHX_CHECK(owner->live_bytes() == 0 && !execution.has_live_batches());
  request.per_item = {4, 4, 5, 0};
  PHX_CHECK(parallel_for(4, request, [](std::size_t, std::size_t, ChunkContext&) {
              native_execution_charge(2, 2);
            }) == PHX_MC_OK);
  PHX_CHECK(execution.evidence()[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 12);
  PHX_CHECK(execution.evidence()[PHX_MC_EXECUTION_SOURCE_QUERIES] == 12);
  PHX_CHECK(owner->live_bytes() == 0 && !execution.has_live_batches());
}

void test_lowest_exception_after_all_chunks_join() {
  constexpr std::size_t count = 31;
  std::array<std::size_t, count> delays{};
  for (std::size_t i = 0; i < count; ++i) delays[i] = i % 5;
  std::mt19937 random(0x51A7u);
  std::shuffle(delays.begin(), delays.end(), random);
  for (uint32_t threads : {1u, 2u, 7u, 64u}) {
    auto owner = std::make_shared<BoundedMemoryResource>(memory_limit);
    {
      NativeExecutionScope execution(2 * count, 0, 0, memory_limit, INFINITY, owner);
      ParallelRequest request;
      request.chunk_size = 1;
      request.parallelism = threads;
      request.per_item.work = 2;
      std::array<uint8_t, count> visited{};
      std::atomic<bool> later_failure{false};
      std::size_t caught = count;
      try {
        parallel_for(count, request, [&](std::size_t begin, std::size_t, ChunkContext&) {
          native_execution_charge(1);
          visited[begin] = 1;
          std::this_thread::sleep_for(std::chrono::milliseconds(delays[begin]));
          // Force a later exception to finish first, independent of randomized
          // chunk completion, without deadlocking the serial reference.
          if (begin == 3 && threads != 1) await(later_failure);
          if (begin == 13) notify(later_failure);
          if (begin == 3 || begin == 13 || begin == 27) throw IndexedFailure{begin};
        });
      } catch (const IndexedFailure& failure) {
        caught = failure.index;
      }
      PHX_CHECK(caught == 3);
      PHX_CHECK(
          std::all_of(visited.begin(), visited.end(), [](uint8_t value) { return value == 1; }));
      PHX_CHECK(execution.evidence()[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == count);
      PHX_CHECK(!execution.has_live_batches() && owner->live_bytes() == 0);
    }
    // Unexpected failure must release the process lease for a fresh caller.
    ParallelRequest next;
    next.parallelism = threads;
    next.chunk_size = 1;
    PHX_CHECK(parallel_for(2, next, [](std::size_t, std::size_t, ChunkContext&) {}) == PHX_MC_OK);
  }
}

void test_local_execution_refusal_is_bound_defect() {
  for (int defect : {0, 1, 2, 3}) {
    auto owner = std::make_shared<BoundedMemoryResource>(memory_limit);
    NativeExecutionScope execution(10, 0, 0, memory_limit, INFINITY, owner);
    ParallelRequest request;
    request.parallelism = 2;
    request.chunk_size = 1;
    if (defect == 3) request.arena_bytes_per_chunk = 64;
    bool rejected = false;
    try {
      parallel_for(2, request, [&](std::size_t, std::size_t, ChunkContext& chunk) -> int32_t {
        if (defect == 0)
          chunk.execution.charge(1);
        else if (defect == 1)
          native_execution_charge(1);
        else if (defect == 2)
          chunk.execution.admit_cavity(1);
        else {
          std::pmr::vector<double> exceeds_arena(128, 0.0, &chunk.arena);
          (void)exceeds_arena;
        }
        return PHX_MC_CAPACITY_EXCEEDED;
      });
    } catch (const std::logic_error&) {
      rejected = true;
    }
    PHX_CHECK(rejected);
    PHX_CHECK(!execution.has_live_batches() && owner->live_bytes() == 0);
  }
}

void test_nested_calls_are_serial_and_restore_detached_state() {
  constexpr std::size_t count = 17;
  auto owner = std::make_shared<BoundedMemoryResource>(memory_limit);
  NativeExecutionScope execution(count * 10, count * 10, 0, memory_limit, INFINITY, owner);
  ParallelRequest outer;
  outer.parallelism = 7;
  outer.chunk_size = 1;
  outer.per_item = {10, 10, 0, 0};
  outer.arena_bytes_per_chunk = 1024 * 1024;
  std::array<uint8_t, count> errors{};
  PHX_CHECK(parallel_for(count, outer, [&](std::size_t begin, std::size_t, ChunkContext& chunk) {
              const std::thread::id thread = std::this_thread::get_id();
              if (current_parallelism() != 7 || active_execution_scope != &chunk.execution ||
                  active_execution_target != nullptr || scratch_memory_resource() != &chunk.arena)
                errors[begin] |= 1;
              native_execution_charge(1);
              ParallelRequest inner;
              inner.parallelism = 64;
              inner.lease_seconds = 0.0;
              inner.chunk_size = 1;
              inner.per_item = {2, 1, 0, 0};
              const int32_t status =
                  parallel_for(3, inner, [&](std::size_t, std::size_t, ChunkContext& nested) {
                    if (std::this_thread::get_id() != thread ||
                        active_execution_scope != &nested.execution ||
                        active_execution_target != nullptr || current_parallelism() != 64)
                      errors[begin] |= 2;
                    native_execution_charge(1, 1);
                    native_execution_primitive_query();
                  });
              if (status != PHX_MC_OK || active_execution_scope != &chunk.execution ||
                  scratch_memory_resource() != &chunk.arena || current_parallelism() != 7)
                errors[begin] |= 4;
            }) == PHX_MC_OK);
  PHX_CHECK(std::all_of(errors.begin(), errors.end(), [](uint8_t error) { return error == 0; }));
  const Work evidence = execution.evidence();
  PHX_CHECK(evidence[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == count * 4);
  PHX_CHECK(evidence[PHX_MC_EXECUTION_SOURCE_QUERIES] == count * 3);
  PHX_CHECK(evidence[PHX_MC_EXECUTION_PRIMITIVE_QUERIES] == count * 3);
  PHX_CHECK(active_execution_scope == &execution && current_parallelism() == 1);
  PHX_CHECK(!execution.has_live_batches() && owner->live_bytes() == 0);
}

void test_single_lease_timeout_concurrent_callers_and_release() {
  ParallelRequest request;
  request.parallelism = 7;
  request.chunk_size = 1;
  std::atomic<bool> first_entered{false}, release_first{false};
  std::atomic<bool> second_queued{false}, second_entered{false};
  int32_t first_status = PHX_MC_INTERNAL_ERROR;
  int32_t second_status = PHX_MC_INTERNAL_ERROR;
  std::jthread first([&] {
    first_status = parallel_for(8, request, [&](std::size_t begin, std::size_t, ChunkContext&) {
      if (begin == 0) {
        notify(first_entered);
        await(release_first);
      }
    });
  });
  await(first_entered);
  std::jthread second([&] {
    notify(second_queued);
    second_status = parallel_for(
        8, request, [&](std::size_t, std::size_t, ChunkContext&) { notify(second_entered); });
  });
  await(second_queued);
  // A serial request must wait for the same lease, not overlap the pool region.
  ParallelRequest immediate;
  immediate.parallelism = 1;
  immediate.lease_seconds = 0.0;
  bool called = false;
  PHX_CHECK(parallel_for(1, immediate, [&](std::size_t, std::size_t, ChunkContext&) {
              called = true;
            }) == PHX_MC_TIMEOUT);
  PHX_CHECK(!called && !second_entered.load(std::memory_order_acquire));
  {
    auto owner = std::make_shared<BoundedMemoryResource>(memory_limit);
    NativeExecutionScope execution(10, 0, 0, memory_limit, INFINITY, owner);
    ParallelRequest bounded;
    bounded.parallelism = 2;
    bounded.chunk_size = 1;
    bounded.per_item.work = 1;
    bounded.lease_seconds = 0.01;
    PHX_CHECK(parallel_for(3, bounded, [&](std::size_t, std::size_t, ChunkContext&) {
                called = true;
              }) == PHX_MC_TIMEOUT);
    PHX_CHECK(!called && execution.evidence()[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 0);
    PHX_CHECK(execution.remaining()[0] == 10 && execution.remaining()[3] == memory_limit);
    PHX_CHECK(owner->live_bytes() == 0 && !execution.has_live_batches());
  }
  notify(release_first);
  first.join();
  second.join();
  PHX_CHECK(first_status == PHX_MC_OK && second_status == PHX_MC_OK);
  PHX_CHECK(second_entered.load(std::memory_order_acquire));
  PHX_CHECK(parallel_for(1, immediate, [&](std::size_t, std::size_t, ChunkContext&) {
              called = true;
            }) == PHX_MC_OK);
  PHX_CHECK(called);
}

void test_deadline_only_cancels_at_chunk_boundaries() {
  auto owner = std::make_shared<BoundedMemoryResource>(memory_limit);
  NativeExecutionScope execution(12, 3, 0, memory_limit, 0.1, owner);
  ParallelRequest request;
  request.chunk_size = 1;
  request.per_item = {4, 1, 0, 0};
  std::array<uint8_t, 3> completed{};
  PHX_CHECK(parallel_for(3, request, [&](std::size_t begin, std::size_t, ChunkContext&) {
              native_execution_charge(1);
              std::this_thread::sleep_for(std::chrono::milliseconds(150));
              // The detached scope has no internal wall-clock deadline. Its second
              // charge still completes, and only the next fixed boundary cancels.
              native_execution_charge(1, 1);
              completed[begin] = 1;
            }) == PHX_MC_TIMEOUT);
  PHX_CHECK((completed == std::array<uint8_t, 3>{1, 0, 0}));
  const Work counters = execution.evidence();
  PHX_CHECK(counters[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 2);
  PHX_CHECK(counters[PHX_MC_EXECUTION_SOURCE_QUERIES] == 1);
  PHX_CHECK(!execution.has_live_batches() && owner->live_bytes() == 0);
}

void test_fork_after_pool_refuses_before_inherited_lease() {
#if !defined(_WIN32)
  ParallelRequest request;
  request.parallelism = 2;
  request.chunk_size = 1;
  std::atomic<bool> entered{false}, release{false};
  int32_t holder_status = PHX_MC_INTERNAL_ERROR;
  std::jthread holder([&] {
    holder_status = parallel_for(2, request, [&](std::size_t begin, std::size_t, ChunkContext&) {
      if (begin == 0) {
        notify(entered);
        await(release);
      }
    });
  });
  await(entered);
  const pid_t child = fork();
  PHX_CHECK(child >= 0);
  if (child == 0) {
    // A hung inherited lease must fail the child instead of hanging CTest.
    alarm(5);
    std::size_t rejected = 0;
    for (std::size_t count : {std::size_t{0}, std::size_t{1}, std::size_t{2}}) {
      try {
        parallel_for(count, request, [](std::size_t, std::size_t, ChunkContext&) {});
      } catch (const std::logic_error&) {
        ++rejected;
      }
    }
    // std::exit still runs the inherited pool's static destructor. The early
    // observer exits only after that teardown returns, preserving this check
    // without LSan's unsupported census of the vanished parent's threads.
    inherited_pool_teardown = true;
    inherited_pool_exit_status = rejected == 3 ? EXIT_SUCCESS : EXIT_FAILURE;
    std::exit(inherited_pool_exit_status);
  }
  if (child > 0) {
    int status = 0;
    PHX_CHECK(waitpid(child, &status, 0) == child);
    PHX_CHECK(WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS);
  }
  notify(release);
  holder.join();
  PHX_CHECK(holder_status == PHX_MC_OK);
#endif
}
}  // namespace

int main() {
#if !defined(_WIN32)
  test_parent_process = getpid();
  // This must precede every pool initialization, including in fork children.
  const int observer_registration = std::atexit(finish_inherited_pool_teardown);
  PHX_CHECK(observer_registration == 0);
  if (observer_registration != 0) return phx::mc::test::finish("test_parallel");
#endif
  test_zero_count_and_lazy_fork();
  test_static_shutdown_drains_active_callers();
  test_policy_nesting_and_thread_locality();
  test_detached_chunks_suppress_scoped_deadline_inheritance();
  test_expired_policy_does_not_renew_for_each_region();
  test_invalid_request_before_launch();
  test_outputs_statuses_counters_and_ordered_reduction();
  test_workers_are_reused_across_regions();
  test_recycled_row_scratch_has_one_bound_per_chunk();
  test_recycled_arenas_scale_with_slots_not_chunks();
  test_nested_region_recycles_one_arena();
  test_resource_admission_before_launch();
  test_unused_reservation_release_after_expected_failure();
  test_lowest_exception_after_all_chunks_join();
  test_local_execution_refusal_is_bound_defect();
  test_nested_calls_are_serial_and_restore_detached_state();
  test_single_lease_timeout_concurrent_callers_and_release();
  test_deadline_only_cancels_at_chunk_boundaries();
  test_fork_after_pool_refuses_before_inherited_lease();
  return phx::mc::test::finish("test_parallel");
}
