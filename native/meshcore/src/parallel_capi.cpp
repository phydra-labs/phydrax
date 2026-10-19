//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include <chrono>
#include <cmath>
#include <memory>
#include <thread>

#include "capi_guard.hpp"
#include "parallel.hpp"
#include "phydrax_meshcore.h"

namespace {

using Clock = std::chrono::steady_clock;
struct ParallelPolicyHandle;
thread_local ParallelPolicyHandle* active_parallel_policy = nullptr;

struct ParallelPolicyHandle {
  ParallelPolicyHandle(uint32_t parallelism, double seconds)
      : previous(active_parallel_policy),
        thread(std::this_thread::get_id()),
        started(Clock::now()),
        maximum_seconds(seconds),
        policy(parallelism, seconds) {}

  ParallelPolicyHandle* previous;
  std::thread::id thread;
  Clock::time_point started;
  double maximum_seconds;
  phx::mc::NativeParallelPolicy policy;
};

}  // namespace

extern "C" int32_t phx_mc_parallel_begin(int32_t parallelism, double maximum_seconds,
                                         void** out_policy) {
  if (out_policy == nullptr) return PHX_MC_INVALID_ARGUMENT;
  *out_policy = nullptr;
  if (parallelism < 1 || std::isnan(maximum_seconds) || maximum_seconds < 0.0) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  return phx::mc::guarded([&]() -> int32_t {
    auto handle =
        std::make_unique<ParallelPolicyHandle>(static_cast<uint32_t>(parallelism), maximum_seconds);
    active_parallel_policy = handle.get();
    *out_policy = handle.release();
    return PHX_MC_OK;
  });
}

extern "C" int32_t phx_mc_parallel_end(void* opaque) {
  auto* handle = static_cast<ParallelPolicyHandle*>(opaque);
  if (handle == nullptr || handle != active_parallel_policy ||
      handle->thread != std::this_thread::get_id()) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const bool timed_out = std::chrono::duration<double>(Clock::now() - handle->started).count() >=
                         handle->maximum_seconds;
  active_parallel_policy = handle->previous;
  delete handle;
  return timed_out ? PHX_MC_TIMEOUT : PHX_MC_OK;
}
