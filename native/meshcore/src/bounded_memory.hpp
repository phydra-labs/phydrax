//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Native numerical buffer allocation accounting. Limits apply to the exact
// byte requests presented to the resource, including simultaneous old/new
// buffers during growth, not estimates of container lengths or process RSS.
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <memory_resource>
#include <new>
#include <mutex>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "phydrax_meshcore.h"
namespace phx::mc {

inline bool native_execution_spend(int64_t work, uint64_t queries) noexcept;

class BoundedMemoryResource final : public std::pmr::memory_resource {
 public:
  explicit BoundedMemoryResource(
      std::size_t limit = std::numeric_limits<std::size_t>::max(),
      std::pmr::memory_resource* upstream = std::pmr::new_delete_resource())
      : limit_(limit), upstream_(upstream) {}

  bool set_limit(std::size_t limit) noexcept {
    const std::lock_guard lock(mutex_);
    if (limit < live_ || limit - live_ < external_upper_) {
      return false;
    }
    limit_ = limit;
    return true;
  }
  std::size_t live_bytes() const noexcept {
    const std::lock_guard lock(mutex_);
    return live_;
  }
  std::size_t peak_bytes() const noexcept {
    const std::lock_guard lock(mutex_);
    return peak_;
  }
  // Last allocator request, including a refused request.
  std::size_t requested_bytes() const noexcept {
    const std::lock_guard lock(mutex_);
    return requested_;
  }
  std::size_t refusals() const noexcept {
    const std::lock_guard lock(mutex_);
    return refusals_;
  }
  std::size_t allocations() const noexcept {
    const std::lock_guard lock(mutex_);
    return allocations_;
  }
  std::size_t limit_bytes() const noexcept {
    const std::lock_guard lock(mutex_);
    return limit_;
  }
  std::size_t available_bytes() const noexcept {
    const std::lock_guard lock(mutex_);
    return limit_ - live_ - external_upper_;
  }
  // Conservative live Python/object storage is not native allocated memory.
  // It consumes the same cap without allocating a dummy payload.
  bool resize_external_storage(std::size_t previous, std::size_t requested) noexcept {
    const std::lock_guard lock(mutex_);
    if (requested > previous &&
        requested - previous > limit_ - live_ - external_upper_) return false;
    external_upper_ = external_upper_ - previous + requested;
    for (Observation* observer = observation_; observer != nullptr; observer = observer->previous) {
      observer->external_peak_upper = std::max(observer->external_peak_upper, external_upper_);
    }
    return true;
  }
  void release_external_storage(std::size_t bytes) noexcept {
    const std::lock_guard lock(mutex_);
    external_upper_ -= bytes;
  }
  std::array<std::uint64_t, 6> evidence() const noexcept {
    const std::lock_guard lock(mutex_);
    return {static_cast<std::uint64_t>(limit_), static_cast<std::uint64_t>(live_),
            static_cast<std::uint64_t>(peak_), static_cast<std::uint64_t>(requested_),
            static_cast<std::uint64_t>(allocations_), static_cast<std::uint64_t>(refusals_)};
  }

 private:
  friend class MemoryBudgetWindow;
  struct Observation {
    Observation* previous = nullptr;
    std::size_t peak = 0;
    std::size_t last_request = 0;
    std::size_t external_peak_upper = 0;
  };
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    const std::lock_guard lock(mutex_);
    requested_ = bytes;
    for (Observation* observer = observation_; observer != nullptr; observer = observer->previous) {
      observer->last_request = bytes;
    }
    if (!native_execution_spend(0, 0) || bytes > limit_ - live_ - external_upper_ ||
        allocations_ == std::numeric_limits<std::size_t>::max()) {
      if (refusals_ != std::numeric_limits<std::size_t>::max()) {
        ++refusals_;
      }
      throw std::bad_alloc();
    }
    void* buffer;
    try {
      buffer = upstream_->allocate(bytes, alignment);
    } catch (const std::bad_alloc&) {
      if (refusals_ != std::numeric_limits<std::size_t>::max()) {
        ++refusals_;
      }
      throw;
    }
    live_ += bytes;
    ++allocations_;
    if (live_ > peak_) {
      peak_ = live_;
    }
    for (Observation* observer = observation_; observer != nullptr; observer = observer->previous) {
      observer->peak = std::max(observer->peak, live_);
    }
    return buffer;
  }
  void do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) override {
    const std::lock_guard lock(mutex_);
    upstream_->deallocate(pointer, bytes, alignment);
    live_ -= bytes;
  }
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

  // Native operations remain serialized by their owning thread. Escaped
  // buffers may release on another thread, so allocation, observations,
  // evidence, limits, and upstream release share this resource lock.
  mutable std::mutex mutex_;
  std::size_t limit_;
  std::pmr::memory_resource* upstream_;
  std::size_t live_ = 0;
  std::size_t external_upper_ = 0;
  std::size_t peak_ = 0;
  std::size_t requested_ = 0;
  std::size_t allocations_ = 0;
  std::size_t refusals_ = 0;
  Observation* observation_ = nullptr;
};

using MemoryOwner = std::shared_ptr<BoundedMemoryResource>;
extern thread_local MemoryOwner active_memory_owner;

inline MemoryOwner scratch_memory_owner() noexcept { return active_memory_owner; }
inline std::pmr::memory_resource* scratch_memory_resource() noexcept {
  return active_memory_owner ? active_memory_owner.get() : std::pmr::new_delete_resource();
}

// Thread-local and nestable; no global default-resource mutation. Allocators
// and spilled buffers retain the owner independently of this lexical scope.
class MemoryScope final {
 public:
  explicit MemoryScope(MemoryOwner owner) noexcept
      : previous_(std::move(active_memory_owner)) {
    active_memory_owner = std::move(owner);
  }
  ~MemoryScope() { active_memory_owner = std::move(previous_); }
  MemoryScope(const MemoryScope&) = delete;
  MemoryScope& operator=(const MemoryScope&) = delete;
 private:
  MemoryOwner previous_;
};

// A nested operation borrows the existing pool rather than escaping its byte
// ledger. The requested quota bounds additional managed allocation above the
// real live-byte baseline; the parent total limit remains in force. Observed
// per-operation peak is sampled on every allocation, not guessed from sizes.
// Native operations are serialized; escaped buffers may release concurrently.
class MemoryBudgetWindow final {
 public:
  explicit MemoryBudgetWindow(std::size_t additional_limit,
                              MemoryOwner owner = scratch_memory_owner())
      : owner_(owner ? std::move(owner)
                     : std::make_shared<BoundedMemoryResource>(additional_limit)) {
    const std::lock_guard lock(owner_->mutex_);
    baseline_ = owner_->live_;
    previous_limit_ = owner_->limit_;
    allocation_start_ = owner_->allocations_;
    refusal_start_ = owner_->refusals_;
    limit_ = std::min(additional_limit, previous_limit_ - baseline_ - owner_->external_upper_);
    owner_->limit_ = baseline_ + owner_->external_upper_ + limit_;
    observation_.previous = owner_->observation_;
    observation_.peak = baseline_;
    observation_.external_peak_upper = owner_->external_upper_;
    owner_->observation_ = &observation_;
  }
  ~MemoryBudgetWindow() {
    const std::lock_guard lock(owner_->mutex_);
    owner_->observation_ = observation_.previous;
    owner_->limit_ = previous_limit_;
  }
  MemoryBudgetWindow(const MemoryBudgetWindow&) = delete;
  MemoryBudgetWindow& operator=(const MemoryBudgetWindow&) = delete;
  const MemoryOwner& owner() const noexcept { return owner_; }
  std::array<std::uint64_t, 6> evidence() const noexcept {
    const std::lock_guard lock(owner_->mutex_);
    const std::size_t current = owner_->live_;
    return {static_cast<std::uint64_t>(limit_),
            static_cast<std::uint64_t>(current > baseline_ ? current - baseline_ : 0),
            static_cast<std::uint64_t>(observation_.peak - baseline_),
            static_cast<std::uint64_t>(observation_.last_request),
            static_cast<std::uint64_t>(owner_->allocations_ - allocation_start_),
            static_cast<std::uint64_t>(owner_->refusals_ - refusal_start_)};
  }
  std::array<std::uint64_t, 2> external_evidence() const noexcept {
    const std::lock_guard lock(owner_->mutex_);
    return {static_cast<std::uint64_t>(owner_->external_upper_),
            static_cast<std::uint64_t>(observation_.external_peak_upper)};
  }
 private:
  MemoryOwner owner_;
  std::size_t baseline_;
  std::size_t previous_limit_;
  std::size_t allocation_start_;
  std::size_t refusal_start_;
  std::size_t limit_ = 0;
  BoundedMemoryResource::Observation observation_;
};

// One bounded native phase, shared by nested owners. Work is charged before
// initialization/search/staging, queries before geometric evaluation, and
// cavity capacity before growing the tentative edit. Accepted edits never
// acquire a new allowance.
struct ExecutionRefusal {
  int32_t status;
};

class NativeExecutionScope;
extern thread_local NativeExecutionScope* active_execution_scope;

class NativeExecutionScope final {
 public:
  NativeExecutionScope(uint64_t work, uint64_t queries, uint64_t cavity,
                       std::size_t scratch, double seconds,
                       MemoryOwner owner = scratch_memory_owner())
      : previous_(active_execution_scope), work_limit_(work),
        query_limit_(queries), cavity_limit_(cavity),
        start_(Clock::now()),
        deadline_(seconds == std::numeric_limits<double>::infinity()
                      ? Clock::time_point::max()
                      : start_ + std::chrono::duration_cast<Clock::duration>(
                                     std::chrono::duration<double>(seconds))),
        memory_(scratch, std::move(owner)), memory_scope_(memory_.owner()) {
    active_execution_scope = this;
  }
  ~NativeExecutionScope() { active_execution_scope = previous_; }
  NativeExecutionScope(const NativeExecutionScope&) = delete;
  NativeExecutionScope& operator=(const NativeExecutionScope&) = delete;

  bool admit(uint64_t work, uint64_t queries = 0) noexcept {
    const auto now = Clock::now();
    for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
      if (scope->status_ != PHX_MC_OK) return false;
      if (now >= scope->deadline_) {
        scope->status_ = PHX_MC_TIMEOUT;
        return false;
      }
      if (work > scope->work_limit_ - scope->work_) {
        for (auto* owner = this; owner != nullptr; owner = owner->previous_) {
          ++owner->work_refusals_;
          owner->status_ = PHX_MC_CAPACITY_EXCEEDED;
        }
        return false;
      }
      if (queries > scope->query_limit_ - scope->queries_) {
        for (auto* owner = this; owner != nullptr; owner = owner->previous_) {
          ++owner->query_refusals_;
          owner->status_ = PHX_MC_CAPACITY_EXCEEDED;
        }
        return false;
      }
    }
    return true;
  }
  bool charge(uint64_t work, uint64_t queries = 0) noexcept {
    if (!admit(work, queries)) return false;
    for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
      scope->work_ += work;
      scope->queries_ += queries;
    }
    return true;
  }
  bool charge_external(uint64_t work, uint64_t queries) noexcept {
    if (!charge(work, queries)) return false;
    for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
      scope->external_work_ += work;
      scope->external_queries_ += queries;
    }
    return true;
  }
  bool import_preparation(uint64_t work, uint64_t queries, double seconds) noexcept {
    if (!admit(work, queries)) return false;
    const auto now = Clock::now();
    for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
      if (scope->deadline_ != Clock::time_point::max() &&
          seconds > std::chrono::duration<double>(scope->deadline_ - now).count()) {
        scope->status_ = PHX_MC_TIMEOUT;
        return false;
      }
      if (!std::isfinite(scope->prior_seconds_ + seconds)) return false;
    }
    if (!charge_external(work, queries)) return false;
    const auto debit = std::chrono::ceil<Clock::duration>(
        std::chrono::duration<double>(seconds));
    for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
      scope->prior_seconds_ += seconds;
      if (scope->deadline_ != Clock::time_point::max()) scope->deadline_ -= debit;
    }
    return true;
  }
  double prior_seconds() const noexcept { return prior_seconds_; }
  bool admit_cavity(std::size_t cells) noexcept {
    if (!charge(0)) return false;
    for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
      if (cells > scope->cavity_limit_) {
        ++scope->cavity_refusals_;
        scope->status_ = PHX_MC_CAPACITY_EXCEEDED;
        return false;
      }
    }
    for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
      scope->peak_cavity_ = std::max(scope->peak_cavity_, static_cast<uint64_t>(cells));
    }
    return true;
  }
  void refuse_external_storage() noexcept {
    for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
      scope->status_ = PHX_MC_CAPACITY_EXCEEDED;
    }
  }
  int32_t status() const noexcept {
    for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
      if (scope->status_ != PHX_MC_OK) return scope->status_;
    }
    return PHX_MC_OK;
  }
  bool primitive_query() noexcept {
    if (!charge(0)) return false;
    for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
      if (scope->primitive_queries_ != std::numeric_limits<uint64_t>::max()) {
        ++scope->primitive_queries_;
      }
    }
    return true;
  }
  std::array<uint64_t, 9> evidence() const noexcept {
    return {work_, queries_, peak_cavity_, work_refusals_, query_refusals_, cavity_refusals_,
            external_work_, external_queries_, primitive_queries_};
  }
  std::array<uint64_t, PHX_MC_EXECUTION_MEMORY_VALUES> memory_evidence() const noexcept {
    const auto allocated = memory_.evidence();
    const auto external = memory_.external_evidence();
    return {allocated[0], allocated[1], allocated[2], allocated[3],
            allocated[4], allocated[5], external[0], external[1]};
  }
  double seconds() const noexcept {
    return std::chrono::duration<double>(Clock::now() - start_).count();
  }
  std::array<uint64_t, 4> remaining() const noexcept {
    std::array<uint64_t, 4> result;
    result.fill(std::numeric_limits<uint64_t>::max());
    for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
      result[0] = std::min(result[0], scope->work_limit_ - scope->work_);
      result[1] = std::min(result[1], scope->query_limit_ - scope->queries_);
      result[2] = std::min(result[2], scope->cavity_limit_);
      const auto& pool = scope->memory_.owner();
      result[3] = std::min(result[3], static_cast<uint64_t>(pool->available_bytes()));
    }
    return result;
  }
  double remaining_wall_seconds() const noexcept {
    auto deadline = Clock::time_point::max();
    for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
      deadline = std::min(deadline, scope->deadline_);
    }
    if (deadline == Clock::time_point::max()) return std::numeric_limits<double>::infinity();
    return std::max(0.0, std::chrono::duration<double>(deadline - Clock::now()).count());
  }
 private:
  using Clock = std::chrono::steady_clock;
  NativeExecutionScope* previous_;
  uint64_t work_limit_, query_limit_, cavity_limit_;
  uint64_t work_ = 0, queries_ = 0, peak_cavity_ = 0;
  uint64_t work_refusals_ = 0, query_refusals_ = 0, cavity_refusals_ = 0;
  uint64_t external_work_ = 0, external_queries_ = 0;
  uint64_t primitive_queries_ = 0;
  double prior_seconds_ = 0.0;
  int32_t status_ = PHX_MC_OK;
  Clock::time_point start_, deadline_;
  MemoryBudgetWindow memory_;
  MemoryScope memory_scope_;
};

inline bool native_execution_spend(int64_t work, uint64_t queries = 0) noexcept {
  return work >= 0 && (active_execution_scope == nullptr ||
                      active_execution_scope->charge(static_cast<uint64_t>(work), queries));
}
inline void native_execution_charge(int64_t work, uint64_t queries = 0) {
  if (!native_execution_spend(work, queries)) {
    throw ExecutionRefusal{active_execution_scope == nullptr
                               ? PHX_MC_CAPACITY_EXCEEDED : active_execution_scope->status()};
  }
}
inline void native_execution_primitive_query() {
  if (active_execution_scope != nullptr && !active_execution_scope->primitive_query()) {
    throw ExecutionRefusal{active_execution_scope->status()};
  }
}
inline void charge_preparation_visit(void (*charge)(void*), void* context) {
  if (charge != nullptr) charge(context);
  else native_execution_charge(0);
}
inline bool native_execution_cavity(std::size_t cells) noexcept {
  return active_execution_scope == nullptr || active_execution_scope->admit_cavity(cells);
}
inline int32_t native_execution_status(int32_t status) noexcept {
  if (active_execution_scope != nullptr &&
      status == PHX_MC_CAPACITY_EXCEEDED) {
    const int32_t refusal = active_execution_scope->status();
    if (refusal != PHX_MC_OK) return refusal;
  }
  return status;
}

// A PMR-backed allocator with scoped default selection and retained resource
// lifetime. Standard PMR allocators borrow a raw pointer and use a process-wide
// default; neither is sufficient for buffers moved beyond a bounded scope.
template <class T>
class NativeAllocator {
 public:
  using value_type = T;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;
  using is_always_equal = std::false_type;

  NativeAllocator() noexcept : owner_(scratch_memory_owner()) {}
  explicit NativeAllocator(MemoryOwner owner) noexcept : owner_(std::move(owner)) {}
  template <class U>
  NativeAllocator(const NativeAllocator<U>& other) noexcept : owner_(other.owner()) {}

  T* allocate(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(resource()->allocate(count * sizeof(T), alignof(T)));
  }
  void deallocate(T* pointer, std::size_t count) noexcept {
    resource()->deallocate(pointer, count * sizeof(T), alignof(T));
  }
  const MemoryOwner& owner() const noexcept { return owner_; }
  std::pmr::memory_resource* resource() const noexcept {
    return owner_ ? owner_.get() : std::pmr::new_delete_resource();
  }
  template <class U>
  bool operator==(const NativeAllocator<U>& other) const noexcept {
    return resource() == other.resource();
  }
 private:
  MemoryOwner owner_;
};

template <class T> using NativeVector = std::vector<T, NativeAllocator<T>>;
template <class T> using NativeDeque = std::deque<T, NativeAllocator<T>>;
template <class K, class Compare = std::less<K>>
using NativeSet = std::set<K, Compare, NativeAllocator<K>>;
template <class K, class V, class Compare = std::less<K>>
using NativeMap = std::map<K, V, Compare, NativeAllocator<std::pair<const K, V>>>;
template <class K, class V, class Hash = std::hash<K>, class Equal = std::equal_to<K>>
using NativeUnorderedMap =
    std::unordered_map<K, V, Hash, Equal, NativeAllocator<std::pair<const K, V>>>;
template <class K, class Hash = std::hash<K>, class Equal = std::equal_to<K>>
using NativeUnorderedSet = std::unordered_set<K, Hash, Equal, NativeAllocator<K>>;

// Opaque objects released across the C boundary retain allocation ownership
// separately from their field buffers. Ordinary external new/delete objects
// have no resource owner; native factories bind one only after construction.
class NativeAllocatedObject {
 public:
  const MemoryOwner& native_object_owner() const noexcept { return native_object_owner_; }
  void bind_native_object_owner(MemoryOwner owner) noexcept {
    native_object_owner_ = std::move(owner);
  }
 private:
  MemoryOwner native_object_owner_;
};

template <class T>
struct NativeObjectDeleter {
  MemoryOwner owner;
  void operator()(T* pointer) const noexcept {
    if (pointer != nullptr) {
      NativeAllocator<T> allocator(owner);
      std::destroy_at(pointer);
      allocator.deallocate(pointer, 1);
    }
  }
};
template <class T>
using NativeUniquePtr = std::unique_ptr<T, NativeObjectDeleter<T>>;

template <class T, class... Args>
NativeUniquePtr<T> make_native_unique(Args&&... args) {
  NativeAllocator<T> allocator;
  T* pointer = allocator.allocate(1);
  try {
    std::construct_at(pointer, std::forward<Args>(args)...);
  } catch (...) {
    allocator.deallocate(pointer, 1);
    throw;
  }
  if constexpr (std::is_base_of_v<NativeAllocatedObject, T>) {
    pointer->bind_native_object_owner(allocator.owner());
  }
  return NativeUniquePtr<T>(pointer, NativeObjectDeleter<T>{allocator.owner()});
}

template <class T>
void destroy_native_object(T* pointer) noexcept {
  if (pointer != nullptr) {
    MemoryOwner owner;
    if constexpr (std::is_base_of_v<NativeAllocatedObject, T>) {
      owner = pointer->native_object_owner();
    }
    NativeObjectDeleter<T>{std::move(owner)}(pointer);
  }
}

// An opaque host array's owner is allocated through the same owned-object
// substrate as native C handles. Its exact numerical request (including zero)
// and owner metadata are managed requests, not a foreign reservation ledger.
class NativeHostBuffer final : public NativeAllocatedObject {
 public:
  NativeHostBuffer(std::size_t bytes, std::size_t alignment)
      : bytes_(bytes), alignment_(alignment),
        data_(scratch_memory_resource()->allocate(bytes, alignment)) {}
  ~NativeHostBuffer() {
    const auto& owner = native_object_owner();
    auto* resource = owner ? owner.get() : std::pmr::new_delete_resource();
    resource->deallocate(data_, bytes_, alignment_);
  }
  NativeHostBuffer(const NativeHostBuffer&) = delete;
  NativeHostBuffer& operator=(const NativeHostBuffer&) = delete;
  void* data() const noexcept { return data_; }
 private:
  std::size_t bytes_;
  std::size_t alignment_;
  void* data_;
};

// Ownership for expansion heap spills. Inline expansions do not allocate and
// do not acquire a resource owner; every spill remembers exact deallocation
// size/alignment and remains valid after its creating MemoryScope ends.
struct DoubleBufferDeleter {
  MemoryOwner owner;
  std::size_t count = 0;
  void operator()(double* pointer) const noexcept {
    if (pointer != nullptr) {
      auto* resource = owner ? owner.get() : std::pmr::new_delete_resource();
      resource->deallocate(pointer, count * sizeof(double), alignof(double));
    }
  }
};
using DoubleBuffer = std::unique_ptr<double[], DoubleBufferDeleter>;
inline DoubleBuffer allocate_double_buffer(std::size_t count) {
  NativeAllocator<double> allocator;
  double* buffer = allocator.allocate(count);
  return DoubleBuffer(buffer, DoubleBufferDeleter{allocator.owner(), count});
}

}  // namespace phx::mc
