// Copyright © 2026 Phydra, Inc. All rights reserved.
#include "parallel.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
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

thread_local std::size_t region_depth = 0;
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
    throw std::logic_error("parallel_for cannot use an inherited worker pool after fork");
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
    throw std::logic_error("parallel_for cannot start during worker-pool shutdown");
  }
  static void release() noexcept {
    active_calls.fetch_sub(1, std::memory_order_seq_cst);
    active_calls.notify_all();
  }
};

class RegionLifetime final {
 public:
  RegionLifetime() noexcept { ++region_depth; }
  ~RegionLifetime() { --region_depth; }
  RegionLifetime(const RegionLifetime&) = delete;
  RegionLifetime& operator=(const RegionLifetime&) = delete;
};

struct Job final {
  Job(std::size_t count_, ParallelRequest request_, const void* context_,
      parallel_detail::BorrowedBody body_, std::size_t chunk_count)
      : count(count_),
        request(request_),
        context(context_),
        body(body_),
        statuses(chunk_count, PHX_MC_OK),
        failures(chunk_count) {}

  void run() noexcept {
    const RegionLifetime region;
    for (;;) {
      const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
      if (index >= statuses.size()) return;
      const std::size_t begin = index * request.chunk_size;
      const std::size_t end = begin + std::min(request.chunk_size, count - begin);
      ChunkContext chunk{index};
      try {
        statuses[index] = body(context, begin, end, chunk);
      } catch (...) {
        failures[index] = std::current_exception();
      }
    }
  }

  int32_t result() const {
    for (const std::exception_ptr& failure : failures) {
      if (failure) std::rethrow_exception(failure);
    }
    for (const int32_t status : statuses) {
      if (status != PHX_MC_OK) return status;
    }
    return PHX_MC_OK;
  }

  std::size_t count;
  ParallelRequest request;
  const void* context;
  parallel_detail::BorrowedBody body;
  std::vector<int32_t> statuses;
  std::vector<std::exception_ptr> failures;
  std::atomic<std::size_t> next{0};
};

class WorkerPool final {
 public:
  ~WorkerPool() { shutdown(); }
  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;
  WorkerPool() = default;

  bool acquire() {
    std::unique_lock lock(lease_mutex_);
    lease_changed_.wait(lock, [this] { return closing_ || !leased_; });
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
    ensure_workers(worker_count);
    {
      const std::lock_guard lock(work_mutex_);
      job_ = &job;
      active_workers_ = worker_count;
      pending_workers_ = worker_count;
      ++generation_;
    }
    work_changed_.notify_all();
    job.run();
    std::unique_lock lock(work_mutex_);
    workers_joined_.wait(lock, [this] { return pending_workers_ == 0; });
    job_ = nullptr;
    active_workers_ = 0;
    ++generation_;
    lock.unlock();
    work_changed_.notify_all();
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
      ++generation_;
    }
    work_changed_.notify_all();
    // Workers must leave their condition-variable wait before jthread joins.
    workers_.clear();
  }

 private:
  void ensure_workers(std::size_t count) {
    std::lock_guard lock(work_mutex_);
    workers_.reserve(std::max(workers_.size(), count));
    while (workers_.size() < count) {
      const std::size_t index = workers_.size();
      workers_.emplace_back([this, index] { worker(index); });
    }
  }

  void worker(std::size_t index) noexcept {
    std::unique_lock lock(work_mutex_);
    std::size_t seen_generation = 0;
    for (;;) {
      work_changed_.wait(lock, [this, &seen_generation] {
        return stopping_ || generation_ != seen_generation;
      });
      if (stopping_) return;
      seen_generation = generation_;
      if (index >= active_workers_) continue;
      Job* const job = job_;
      lock.unlock();
      job->run();
      lock.lock();
      --pending_workers_;
      if (pending_workers_ == 0) workers_joined_.notify_one();
    }
  }

  std::mutex lease_mutex_;
  std::condition_variable lease_changed_;
  bool leased_ = false;
  bool closing_ = false;
  bool closed_ = false;

  std::mutex work_mutex_;
  std::condition_variable work_changed_;
  std::condition_variable workers_joined_;
  std::vector<std::jthread> workers_;
  Job* job_ = nullptr;
  std::size_t active_workers_ = 0;
  std::size_t pending_workers_ = 0;
  std::size_t generation_ = 0;
  bool stopping_ = false;
};

struct PoolLifetime final {
  std::unique_ptr<WorkerPool> pool = std::make_unique<WorkerPool>();

  ~PoolLifetime() {
    // Destroying inherited jthreads or synchronization objects in a forked
    // child is unsafe; the child refuses every call before reaching this pool.
    if (pool_process.load(std::memory_order_acquire) != process_id()) {
      pool.release();
      return;
    }
    shutting_down.store(true, std::memory_order_seq_cst);
    pool->shutdown();
    for (std::size_t count = active_calls.load(std::memory_order_seq_cst); count != 0;
         count = active_calls.load(std::memory_order_seq_cst)) {
      active_calls.wait(count, std::memory_order_seq_cst);
    }
    pool.reset();
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
  explicit PoolLease(WorkerPool& pool) : pool_(pool.acquire() ? &pool : nullptr) {
    if (pool_ == nullptr) {
      throw std::logic_error("parallel_for worker pool is shutting down");
    }
  }
  ~PoolLease() {
    if (pool_ != nullptr) pool_->release();
  }
  PoolLease(const PoolLease&) = delete;
  PoolLease& operator=(const PoolLease&) = delete;

 private:
  WorkerPool* pool_;
};

}  // namespace

int32_t parallel_detail::parallel_for_erased(std::size_t count, ParallelRequest request,
                                             const void* context, BorrowedBody body) {
  const CallLifetime call;
  if (request.chunk_size == 0 || request.parallelism == 0 || body == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (count == 0) return PHX_MC_OK;

  const std::size_t chunks = count / request.chunk_size + (count % request.chunk_size != 0);
  Job job(count, request, context, body, chunks);
  if (region_depth != 0) {
    job.run();
    return job.result();
  }

  WorkerPool& pool = process_pool();
  const PoolLease lease(pool);
  const std::size_t participants =
      std::min(chunks, static_cast<std::size_t>(request.parallelism));
  pool.run(job, participants - 1);
  return job.result();
}

}  // namespace phx::mc
