//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include "parallel.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <semaphore>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace phx::mc {
namespace {
using Clock = std::chrono::steady_clock;
thread_local uint32_t policy_parallelism = 1;
thread_local Clock::time_point policy_deadline = Clock::time_point::max();
thread_local std::size_t region_depth = 0;

// These atomics precede singleton initialization. A forked child must refuse
// before a C++ static initialization guard or any inherited mutex is touched.
static_assert(std::atomic<int>::is_always_lock_free);
std::atomic<int> pool_process{0};
std::atomic<bool> shutting_down{false};
std::atomic<std::size_t> active_calls{0};

int process_id() noexcept {
#if defined(_WIN32)
  return _getpid();
#else
  return static_cast<int>(getpid());
#endif
}

void check_process() {
  const int owner = pool_process.load(std::memory_order_acquire);
  if (owner != 0 && owner != process_id()) {
    throw std::logic_error(
        "parallel_for.process must not fork after pool initialization; got a forked child");
  }
}

class CallLifetime final {
 public:
  CallLifetime() {
    check_process();
    if (shutting_down.load(std::memory_order_seq_cst)) refuse_shutdown();
    active_calls.fetch_add(1, std::memory_order_seq_cst);
    if (shutting_down.load(std::memory_order_seq_cst)) {
      release();
      refuse_shutdown();
    }
  }
  ~CallLifetime() { release(); }
  CallLifetime(const CallLifetime&) = delete;
  CallLifetime& operator=(const CallLifetime&) = delete;

 private:
  [[noreturn]] static void refuse_shutdown() {
    throw std::logic_error("parallel_for.pool must remain open; got static shutdown");
  }
  static void release() noexcept {
    active_calls.fetch_sub(1, std::memory_order_seq_cst);
    active_calls.notify_all();
  }
};

Clock::time_point deadline_after(double seconds, Clock::time_point now) noexcept {
  const double available = std::chrono::duration<double>(Clock::time_point::max() - now).count();
  if (std::isinf(seconds) || seconds >= available) return Clock::time_point::max();
  if (seconds <= 0.0) return now;
  return now + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
}

class ChunkLifetime final {
 public:
  explicit ChunkLifetime(ParallelRequest request)
      : previous_parallelism_(policy_parallelism), previous_deadline_(policy_deadline) {
    policy_parallelism = request.parallelism;
    policy_deadline = Clock::time_point::max();
    ++region_depth;
  }
  ~ChunkLifetime() {
    --region_depth;
    policy_parallelism = previous_parallelism_;
    policy_deadline = previous_deadline_;
  }

 private:
  uint32_t previous_parallelism_;
  Clock::time_point previous_deadline_;
};

// Worker-visible state has no batch, scope-chain or target pointer. Each
// indexed slot has one writer and is read by the caller only after joining.
struct Job final {
  std::size_t count;
  ParallelRequest request;
  const void* body_context;
  parallel_detail::BorrowedBody body;
  std::span<const NativeChunkAllowance> allowances;
  std::span<const MemoryOwner> owners;
  std::span<NativeChunkEvidence> evidence;
  std::span<std::exception_ptr> failures;
  Clock::time_point deadline;
  std::atomic<std::size_t> next{0};
  std::atomic<bool> cancelled{false};

  bool boundary_timeout() noexcept {
    if (cancelled.load(std::memory_order_acquire)) return true;
    if (deadline != Clock::time_point::max() && Clock::now() >= deadline) {
      cancelled.store(true, std::memory_order_release);
      return true;
    }
    return false;
  }

  void evaluate(std::size_t index) noexcept {
    try {
      ChunkLifetime lifetime(request);
      NativeDetachedExecution detached(allowances[index], owners[index]);
      try {
        const std::size_t begin = index * request.chunk_size;
        const std::size_t end = begin + std::min(request.chunk_size, count - begin);
        ChunkContext chunk{index, detached.execution(), *owners[index]};
        const int32_t status = body(body_context, begin, end, chunk);
        if (detached.execution().status() != PHX_MC_OK) {
          throw std::logic_error(
              "parallel_for.chunk allowance must cover local execution; got a bound defect");
        }
        evidence[index] = detached.evidence(status, true);
      } catch (const ExecutionRefusal&) {
        evidence[index] = detached.evidence(PHX_MC_INTERNAL_ERROR, false);
        failures[index] = std::make_exception_ptr(std::logic_error(
            "parallel_for.chunk allowance must cover local execution; got ExecutionRefusal"));
      } catch (const std::bad_alloc&) {
        evidence[index] = detached.evidence(PHX_MC_INTERNAL_ERROR, false);
        if (owners[index]->refusals() != 0) {
          failures[index] = std::make_exception_ptr(std::logic_error(
              "parallel_for.chunk arena must cover local allocations; got a bound defect"));
        } else {
          failures[index] = std::current_exception();
        }
      } catch (...) {
        evidence[index] = detached.evidence(PHX_MC_INTERNAL_ERROR, false);
        failures[index] = std::current_exception();
      }
    } catch (...) {
      evidence[index].status = PHX_MC_INTERNAL_ERROR;
      evidence[index].completed = false;
      failures[index] = std::current_exception();
    }
  }

  void run() noexcept {
    for (;;) {
      // Cancellation never interrupts an item or a logical chunk.
      if (boundary_timeout()) return;
      const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
      if (index >= allowances.size()) return;
      if (boundary_timeout()) return;
      evaluate(index);
      if (boundary_timeout()) return;
    }
  }
};

class WorkerPool final {
 public:
  ~WorkerPool() { shutdown(); }

  bool acquire(Clock::time_point deadline) {
    std::unique_lock lock(lease_mutex_);
    const auto available = [this] { return closing_ || !leased_; };
    if (deadline == Clock::time_point::max()) {
      lease_changed_.wait(lock, available);
    } else if (!lease_changed_.wait_until(lock, deadline, available)) {
      return false;
    }
    if (closing_) return false;
    leased_ = true;
    return true;
  }

  void release() noexcept {
    {
      const std::lock_guard lock(lease_mutex_);
      leased_ = false;
    }
    lease_changed_.notify_all();
  }

  void run(Job& job, std::size_t worker_count) {
    if (worker_count == 0) {
      job.run();
      return;
    }
    {
      const std::lock_guard lock(work_mutex_);
      // All creation precedes dispatch. If creation fails, existing workers
      // stay parked and the lease/reservation unwind without a dangling job.
      while (workers_.size() < worker_count) {
        workers_.emplace_back();
        Worker* const state = &workers_.back();
        try {
          state->thread = std::jthread([this, state] { worker(*state); });
        } catch (...) {
          workers_.pop_back();
          throw;
        }
      }
      job_ = &job;
      pending_ = worker_count;
    }
    // Only participating workers wake; a historical larger request does not
    // make small batches contend with every parked thread.
    for (std::size_t index = 0; index < worker_count; ++index) workers_[index].ready.release();
    job.run();
    std::unique_lock lock(work_mutex_);
    joined_.wait(lock, [this] { return pending_ == 0; });
    job_ = nullptr;
  }

  void shutdown() noexcept {
    {
      std::unique_lock lock(lease_mutex_);
      if (closed_) return;
      closing_ = true;
      lease_changed_.notify_all();
      lease_changed_.wait(lock, [this] { return !leased_; });
      closed_ = true;
    }
    {
      const std::lock_guard lock(work_mutex_);
      stopping_ = true;
    }
    for (Worker& state : workers_) state.ready.release();
    // Never join while holding the mutex workers need to leave their wait.
    workers_.clear();
  }

 private:
  struct Worker {
    std::binary_semaphore ready{0};
    std::jthread thread;
  };
  void worker(Worker& state) noexcept {
    for (;;) {
      state.ready.acquire();
      std::unique_lock lock(work_mutex_);
      if (stopping_) return;
      Job* const job = job_;
      lock.unlock();
      job->run();
      lock.lock();
      --pending_;
      if (pending_ == 0) joined_.notify_one();
    }
  }

  std::mutex lease_mutex_;
  std::condition_variable lease_changed_;
  bool leased_ = false;
  bool closing_ = false;
  bool closed_ = false;
  std::mutex work_mutex_;
  std::condition_variable joined_;
  std::deque<Worker> workers_;
  Job* job_ = nullptr;
  std::size_t pending_ = 0;
  bool stopping_ = false;
};

struct PoolLifetime final {
  WorkerPool* pool = new WorkerPool;
  ~PoolLifetime() {
    // In a forked child even destroying an inherited jthread is unsafe.
    if (pool_process.load(std::memory_order_acquire) != process_id()) return;
    shutting_down.store(true, std::memory_order_seq_cst);
    pool->shutdown();
    for (std::size_t count = active_calls.load(std::memory_order_seq_cst); count != 0;
         count = active_calls.load(std::memory_order_seq_cst)) {
      active_calls.wait(count, std::memory_order_seq_cst);
    }
    delete pool;
  }
};

WorkerPool& process_pool() {
  check_process();
  int empty = 0;
  pool_process.compare_exchange_strong(empty, process_id(), std::memory_order_acq_rel);
  check_process();
  static PoolLifetime lifetime;
  return *lifetime.pool;
}

class PoolLease final {
 public:
  PoolLease(WorkerPool& pool, Clock::time_point deadline)
      : pool_(pool.acquire(deadline) ? &pool : nullptr) {}
  ~PoolLease() {
    if (pool_ != nullptr) pool_->release();
  }
  explicit operator bool() const noexcept { return pool_ != nullptr; }
  PoolLease(const PoolLease&) = delete;
  PoolLease& operator=(const PoolLease&) = delete;

 private:
  WorkerPool* pool_;
};

bool valid_request(ParallelRequest request) noexcept {
  return request.chunk_size != 0 && request.parallelism != 0 && request.per_item.bytes == 0 &&
         !std::isnan(request.lease_seconds) && request.lease_seconds >= 0.0;
}

bool multiply_allowance(NativeChunkAllowance per_item, std::size_t items,
                        NativeChunkAllowance& result) noexcept {
  const uint64_t n = static_cast<uint64_t>(items);
  const uint64_t maximum = std::numeric_limits<uint64_t>::max();
  if ((per_item.work != 0 && n > maximum / per_item.work) ||
      (per_item.queries != 0 && n > maximum / per_item.queries) ||
      (per_item.cavity != 0 && n > maximum / per_item.cavity))
    return false;
  result = {per_item.work * n, per_item.queries * n, per_item.cavity * n, 0};
  return true;
}

int32_t joined_status(const Job& job, int32_t settlement) {
  for (const std::exception_ptr& failure : job.failures) {
    if (failure) std::rethrow_exception(failure);
  }
  if (job.cancelled.load(std::memory_order_acquire)) return PHX_MC_TIMEOUT;
  if (settlement != PHX_MC_OK) return settlement;
  for (const NativeChunkEvidence& chunk : job.evidence) {
    if (chunk.status != PHX_MC_OK) return chunk.status;
  }
  return PHX_MC_OK;
}
}  // namespace

NativeParallelPolicy::NativeParallelPolicy(uint32_t parallelism, double lease_seconds)
    : previous_parallelism_(policy_parallelism), previous_deadline_(policy_deadline) {
  if (parallelism == 0 || std::isnan(lease_seconds) || lease_seconds < 0.0) {
    throw std::invalid_argument(
        "NativeParallelPolicy.parallelism/lease_seconds must be positive/nonnegative; got an "
        "invalid policy");
  }
  policy_parallelism = parallelism;
  policy_deadline = std::min(previous_deadline_, deadline_after(lease_seconds, Clock::now()));
}

NativeParallelPolicy::~NativeParallelPolicy() {
  policy_parallelism = previous_parallelism_;
  policy_deadline = previous_deadline_;
}

uint32_t current_parallelism() noexcept { return policy_parallelism; }
double current_parallel_lease_seconds() noexcept {
  if (policy_deadline == Clock::time_point::max()) return INFINITY;
  return std::max(0.0, std::chrono::duration<double>(policy_deadline - Clock::now()).count());
}

int32_t parallel_detail::parallel_for_erased(std::size_t count, ParallelRequest request,
                                             const void* context, BorrowedBody body) {
  const CallLifetime call;
  if (!valid_request(request)) return PHX_MC_INVALID_ARGUMENT;
  if (count == 0) return PHX_MC_OK;
  const std::size_t chunks = count / request.chunk_size + (count % request.chunk_size != 0);
  std::vector<NativeChunkAllowance> allowances(chunks);
  for (std::size_t index = 0; index < chunks; ++index) {
    const std::size_t items = std::min(request.chunk_size, count - index * request.chunk_size);
    if (!multiply_allowance(request.per_item, items, allowances[index]))
      return PHX_MC_CAPACITY_EXCEEDED;
    allowances[index].bytes = request.arena_bytes_per_chunk;
  }
  NativeParallelBatch batch(allowances);
  if (batch.status() != PHX_MC_OK) return batch.status();
  std::vector<MemoryOwner> owners;
  owners.reserve(chunks);
  for (std::size_t index = 0; index < chunks; ++index) owners.push_back(batch.chunk_owner(index));
  std::vector<NativeChunkEvidence> evidence(chunks);
  for (NativeChunkEvidence& chunk : evidence) chunk.status = PHX_MC_TIMEOUT;
  std::vector<std::exception_ptr> failures(chunks);
  const Clock::time_point now = Clock::now();
  const Clock::time_point deadline =
      std::min(policy_deadline, deadline_after(batch.remaining_wall_seconds(), now));
  Job job{count, request, context, body, allowances, owners, evidence, failures, deadline};
  if (region_depth != 0) {
    job.run();
  } else {
    WorkerPool& pool = process_pool();
    const Clock::time_point lease_deadline =
        std::min(deadline, deadline_after(request.lease_seconds, now));
    PoolLease lease(pool, lease_deadline);
    if (lease) {
      const std::size_t threads = std::min<std::size_t>(request.parallelism, chunks);
      pool.run(job, threads - 1);
    } else {
      job.cancelled.store(true, std::memory_order_release);
    }
  }
  // Scheduling handles do not keep scratch backing alive through settlement.
  // Escaped NativeAllocator buffers retain their own owner independently.
  owners.clear();
  const int32_t settlement = batch.settle(evidence);
  return joined_status(job, settlement);
}

}  // namespace phx::mc
