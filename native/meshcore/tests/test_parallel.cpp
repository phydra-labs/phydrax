// Copyright © 2026 Phydra, Inc. All rights reserved.
// Worker-pool lifecycle, deterministic indexed output, nested calls, and failures.
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
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

void notify(std::atomic<bool>& flag) {
  flag.store(true, std::memory_order_release);
  flag.notify_all();
}

void await(const std::atomic<bool>& flag) {
  while (!flag.load(std::memory_order_acquire)) flag.wait(false, std::memory_order_acquire);
}

#if !defined(_WIN32)
std::atomic<int> shutdown_completed{0};

void test_zero_count_and_fork_before_pool_initialization() {
  ParallelRequest request;
  request.parallelism = 2;
  bool called = false;
  PHX_CHECK(parallel_for(0, request, [&](std::size_t, std::size_t, ChunkContext&) {
              called = true;
            }) == PHX_MC_OK);
  PHX_CHECK(!called);

  const pid_t child = fork();
  PHX_CHECK(child >= 0);
  if (child == 0) {
    std::array<int, 2> output{};
    request.chunk_size = 1;
    const int32_t status =
        parallel_for(2, request, [&](std::size_t begin, std::size_t, ChunkContext&) {
          output[begin] = static_cast<int>(begin + 1);
        });
    std::exit(status == PHX_MC_OK && output == std::array<int, 2>{1, 2} ? EXIT_SUCCESS
                                                                      : EXIT_FAILURE);
  }
  if (child > 0) {
    int status = 0;
    PHX_CHECK(waitpid(child, &status, 0) == child);
    PHX_CHECK(WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS);
  }
}

void test_static_shutdown_drains_active_region() {
  const pid_t child = fork();
  PHX_CHECK(child >= 0);
  if (child == 0) {
    if (std::atexit([] {
          if (shutdown_completed.load(std::memory_order_acquire) != 2) std::_Exit(EXIT_FAILURE);
        }) != 0) {
      std::_Exit(EXIT_FAILURE);
    }
    std::atomic<bool> entered{false};
    std::atomic<bool> release{false};
    ParallelRequest request;
    request.parallelism = 2;
    request.chunk_size = 1;
    std::thread caller([&] {
      try {
        parallel_for(2, request, [&](std::size_t, std::size_t, ChunkContext&) {
          notify(entered);
          await(release);
          shutdown_completed.fetch_add(1, std::memory_order_release);
        });
      } catch (...) {
        std::_Exit(EXIT_FAILURE);
      }
    });
    caller.detach();
    std::thread releaser([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      notify(release);
    });
    releaser.detach();
    await(entered);
    std::exit(EXIT_SUCCESS);
  }
  if (child > 0) {
    int status = 0;
    PHX_CHECK(waitpid(child, &status, 0) == child);
    PHX_CHECK(WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS);
  }
}
#endif

struct ParallelResult {
  std::vector<uint64_t> output;
  std::vector<int32_t> item_status;
  std::vector<uint64_t> chunk_partials;
  uint64_t reduction = 0;
  int32_t status = PHX_MC_OK;
};

ParallelResult run_chunk_merge(uint32_t parallelism) {
  constexpr std::size_t count = 773;
  constexpr std::size_t chunk_size = 11;
  constexpr std::size_t chunks = (count + chunk_size - 1) / chunk_size;
  ParallelResult result;
  result.output.resize(count);
  result.item_status.resize(count);
  result.chunk_partials.resize(chunks);
  ParallelRequest request;
  request.chunk_size = chunk_size;
  request.parallelism = parallelism;
  result.status = parallel_for(
      count, request, [&](std::size_t begin, std::size_t end, ChunkContext& chunk) -> int32_t {
        double partial = 0.0;
        int32_t chunk_status = PHX_MC_OK;
        for (std::size_t i = begin; i < end; ++i) {
          const double value = i % 3 == 0 ? 1.0e16
                              : i % 3 == 1 ? static_cast<double>(i) * 0.25 - 7.0
                                           : -1.0e16;
          result.output[i] = std::bit_cast<uint64_t>(value);
          partial += value;
          const int32_t item_status =
              i % 19 == 3 ? PHX_MC_NONFINITE_INPUT
                          : (i % 17 == 2 ? PHX_MC_DEGENERATE_INPUT : PHX_MC_OK);
          result.item_status[i] = item_status;
          if (chunk_status == PHX_MC_OK) chunk_status = item_status;
        }
        result.chunk_partials[chunk.chunk_index] = std::bit_cast<uint64_t>(partial);
        return chunk_status;
      });
  double reduction = 0.0;
  for (const uint64_t partial : result.chunk_partials) reduction += std::bit_cast<double>(partial);
  result.reduction = std::bit_cast<uint64_t>(reduction);
  return result;
}

void test_indexed_outputs_statuses_and_ordered_merge() {
  const ParallelResult reference = run_chunk_merge(1);
  PHX_CHECK(reference.status == PHX_MC_DEGENERATE_INPUT);
  for (std::size_t i = 0; i < reference.output.size(); ++i) {
    const double value = i % 3 == 0 ? 1.0e16
                        : i % 3 == 1 ? static_cast<double>(i) * 0.25 - 7.0
                                     : -1.0e16;
    const int32_t status = i % 19 == 3 ? PHX_MC_NONFINITE_INPUT
                          : i % 17 == 2 ? PHX_MC_DEGENERATE_INPUT
                                        : PHX_MC_OK;
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
  for (const uint32_t parallelism : {2u, 4u}) {
    const ParallelResult result = run_chunk_merge(parallelism);
    PHX_CHECK(result.output == reference.output);
    PHX_CHECK(result.item_status == reference.item_status);
    PHX_CHECK(result.chunk_partials == reference.chunk_partials);
    PHX_CHECK(result.reduction == reference.reduction);
    PHX_CHECK(result.status == reference.status);
  }
}

void test_workers_are_reused_across_regions() {
  ParallelRequest request;
  request.parallelism = 2;
  request.chunk_size = 1;
  std::barrier rendezvous(2);
  std::array<uint32_t, 2> visits{};
  const auto body = [&](std::size_t begin, std::size_t, ChunkContext&) {
    static thread_local uint32_t thread_visits = 0;
    visits[begin] = ++thread_visits;
    rendezvous.arrive_and_wait();
  };
  PHX_CHECK(parallel_for(2, request, body) == PHX_MC_OK);
  PHX_CHECK((visits == std::array<uint32_t, 2>{1, 1}));
  PHX_CHECK(parallel_for(2, request, body) == PHX_MC_OK);
  PHX_CHECK((visits == std::array<uint32_t, 2>{2, 2}));
}

void test_nested_regions_run_serially_on_the_calling_thread() {
  constexpr std::size_t outer_count = 4;
  constexpr std::size_t inner_count = 5;
  std::array<std::array<uint64_t, inner_count>, outer_count> output{};
  std::array<int32_t, outer_count> inner_status{};
  std::array<uint8_t, outer_count> wrong_thread{};
  ParallelRequest outer;
  outer.parallelism = 3;
  outer.chunk_size = 1;
  PHX_CHECK(parallel_for(outer_count, outer, [&](std::size_t begin, std::size_t,
                                                 ChunkContext&) {
              const std::thread::id caller = std::this_thread::get_id();
              ParallelRequest inner;
              inner.parallelism = 8;
              inner.chunk_size = 1;
              inner_status[begin] = parallel_for(
                  inner_count, inner, [&](std::size_t row, std::size_t,
                                            ChunkContext&) {
                    if (std::this_thread::get_id() != caller) wrong_thread[begin] = 1;
                    output[begin][row] = begin + row;
                  });
            }) == PHX_MC_OK);
  for (std::size_t outer_index = 0; outer_index < outer_count; ++outer_index) {
    PHX_CHECK(inner_status[outer_index] == PHX_MC_OK);
    PHX_CHECK(wrong_thread[outer_index] == 0);
    for (std::size_t inner_index = 0; inner_index < inner_count; ++inner_index) {
      PHX_CHECK(output[outer_index][inner_index] == outer_index + inner_index);
    }
  }
}

struct IndexedFailure {
  std::size_t index;
};

void test_lowest_exception_is_rethrown_after_all_chunks_join() {
  constexpr std::size_t count = 31;
  std::array<std::atomic<uint8_t>, count> visited{};
  for (const uint32_t parallelism : {1u, 2u, 4u}) {
    ParallelRequest request;
    request.parallelism = parallelism;
    request.chunk_size = 1;
    std::size_t caught = count;
    try {
      parallel_for(count, request, [&](std::size_t begin, std::size_t,
                                       ChunkContext&) {
        visited[begin].store(1, std::memory_order_relaxed);
        if (begin == 3) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (begin == 13) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (begin == 3 || begin == 13 || begin == 27) throw IndexedFailure{begin};
      });
    } catch (const IndexedFailure& failure) {
      caught = failure.index;
    }
    PHX_CHECK(caught == 3);
    for (const auto& flag : visited) {
      PHX_CHECK(flag.load(std::memory_order_relaxed) == 1);
    }
    for (auto& flag : visited) flag.store(0, std::memory_order_relaxed);
  }
}

void test_top_level_callers_do_not_overlap_pool_regions() {
  ParallelRequest request;
  request.parallelism = 2;
  request.chunk_size = 1;
  ParallelRequest serial = request;
  serial.parallelism = 1;
  std::atomic<bool> first_entered{false};
  std::atomic<bool> release_first{false};
  std::atomic<bool> second_started{false};
  std::atomic<bool> second_entered{false};
  int32_t first_status = PHX_MC_INTERNAL_ERROR;
  int32_t second_status = PHX_MC_INTERNAL_ERROR;
  std::jthread first([&] {
    first_status = parallel_for(8, request, [&](std::size_t begin, std::size_t,
                                                ChunkContext&) {
      if (begin == 0) {
        notify(first_entered);
        await(release_first);
      }
    });
  });
  await(first_entered);
  std::jthread second([&] {
    notify(second_started);
    second_status = parallel_for(2, serial, [&](std::size_t, std::size_t,
                                                 ChunkContext&) {
      notify(second_entered);
    });
  });
  await(second_started);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  const bool remained_blocked = !second_entered.load(std::memory_order_acquire);
  notify(release_first);
  first.join();
  second.join();
  PHX_CHECK(remained_blocked);
  PHX_CHECK(first_status == PHX_MC_OK && second_status == PHX_MC_OK);
  PHX_CHECK(second_entered.load(std::memory_order_acquire));
}

void test_invalid_requests_do_not_launch_bodies() {
  ParallelRequest request;
  bool called = false;
  const auto body = [&](std::size_t, std::size_t, ChunkContext&) { called = true; };
  request.chunk_size = 0;
  PHX_CHECK(parallel_for(1, request, body) == PHX_MC_INVALID_ARGUMENT);
  request.chunk_size = 1;
  request.parallelism = 0;
  PHX_CHECK(parallel_for(1, request, body) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(!called);
}

#if !defined(_WIN32)
void test_fork_after_pool_initialization_is_refused() {
  const pid_t child = fork();
  PHX_CHECK(child >= 0);
  if (child == 0) {
    ParallelRequest request;
    request.parallelism = 2;
    std::size_t rejected = 0;
    for (const std::size_t count : {std::size_t{0}, std::size_t{1}}) {
      try {
        parallel_for(count, request, [](std::size_t, std::size_t, ChunkContext&) {});
      } catch (const std::logic_error&) {
        ++rejected;
      }
    }
    std::exit(rejected == 2 ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (child > 0) {
    int status = 0;
    PHX_CHECK(waitpid(child, &status, 0) == child);
    PHX_CHECK(WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS);
  }
}
#endif

}  // namespace

int main() {
#if !defined(_WIN32)
  test_zero_count_and_fork_before_pool_initialization();
  test_static_shutdown_drains_active_region();
#endif
  test_indexed_outputs_statuses_and_ordered_merge();
  test_workers_are_reused_across_regions();
  test_nested_regions_run_serially_on_the_calling_thread();
  test_lowest_exception_is_rethrown_after_all_chunks_join();
  test_top_level_callers_do_not_overlap_pool_regions();
  test_invalid_requests_do_not_launch_bodies();
#if !defined(_WIN32)
  test_fork_after_pool_initialization_is_refused();
#endif
  return phx::mc::test::finish("test_parallel");
}
