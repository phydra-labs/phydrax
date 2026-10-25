//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Native numerical buffer allocation accounting. Limits apply to the exact
// byte requests presented to the resource, including simultaneous old/new
// buffers during growth, not estimates of container lengths or process RSS.
#pragma once
#include "counter_records.hpp"

#include <algorithm>
#include <atomic>
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
#include <span>
#include <type_traits>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "phydrax_meshcore.h"
namespace phx::mc {

inline bool native_execution_spend(int64_t work, uint64_t queries) noexcept;
struct NativeBatchStorage;
class NativeExecutionTarget;
extern thread_local NativeExecutionTarget* active_execution_target;

class BoundedMemoryResource final : public std::pmr::memory_resource {
 public:
  explicit BoundedMemoryResource(
      std::size_t limit = std::numeric_limits<std::size_t>::max(),
      std::pmr::memory_resource* upstream = std::pmr::new_delete_resource())
      : limit_(limit), upstream_(upstream) {}

  bool set_limit(std::size_t limit) noexcept {
    const std::lock_guard ledger(*ledger_mutex_);
    const std::lock_guard lock(mutex_);
    if (limit < live_ || limit - live_ < external_upper_ + batch_upper_) {
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
    return effective_limit_locked();
  }
  std::size_t available_bytes() const noexcept;
  const std::shared_ptr<std::mutex>& ledger_mutex() const noexcept { return ledger_mutex_; }
  // Allocator headers are separate from exact numerical payload requests.
  std::size_t bookkeeping_bytes() const noexcept {
    const std::lock_guard lock(mutex_);
    return bookkeeping_;
  }
  // Conservative live Python/object storage is not native allocated memory.
  // It consumes the same cap without allocating a dummy payload.
  bool resize_external_storage(std::size_t previous, std::size_t requested,
                               const std::shared_ptr<NativeBatchStorage>& storage = {}) noexcept;
  void release_external_storage(std::size_t bytes,
                                const std::shared_ptr<NativeBatchStorage>& storage = {}) noexcept;
  bool reserve_batch_storage(std::size_t bytes) noexcept {
    const std::lock_guard lock(mutex_);
    if (bytes > available_locked()) return false;
    batch_upper_ += bytes;
    return true;
  }
  void settle_batch_storage(std::size_t upper, std::size_t peak) noexcept {
    const std::lock_guard lock(mutex_);
    batch_upper_ -= upper;
    if (peak == 0) return;
    for (Observation* observer = observation_; observer != nullptr; observer = observer->previous) {
      observer->external_peak_upper = std::max(
          observer->external_peak_upper, external_upper_ + batch_upper_ + peak);
    }
  }
  std::array<std::uint64_t, 6> evidence() const noexcept {
    const std::lock_guard lock(mutex_);
    return {static_cast<std::uint64_t>(effective_limit_locked()), static_cast<std::uint64_t>(live_),
            static_cast<std::uint64_t>(peak_), static_cast<std::uint64_t>(requested_),
            static_cast<std::uint64_t>(allocations_), static_cast<std::uint64_t>(refusals_)};
  }

 private:
  friend class MemoryBudgetWindow;
  friend class NativeExecutionScope;
  friend class NativeExecutionBatch;
  friend class NativeExecutionTarget;
  struct Observation {
    Observation* previous = nullptr;
    std::size_t peak = 0;
    std::size_t limit = 0;
    std::size_t managed_limit = 0;
    std::size_t external_baseline = 0;
    std::size_t last_request = 0;
    std::thread::id thread;
    bool aggregate = false;
    std::size_t allocation_start = 0;
    std::size_t allocations = 0, refusals = 0, live = 0;
    std::size_t external_peak_upper = 0;
  };
  std::size_t effective_limit_locked() const noexcept;
  std::size_t available_locked() const noexcept;
  void change_storage_locked(const std::shared_ptr<NativeBatchStorage>& storage,
                             std::size_t bytes, bool increase) noexcept;
  void* do_allocate(std::size_t bytes, std::size_t alignment) override;
  void do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) override;
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

  // Native operations remain serialized by their owning thread. Escaped
  // buffers may release on another thread, so allocation, observations,
  // evidence, limits, and upstream release share this resource lock.
  mutable std::mutex mutex_;
  std::shared_ptr<std::mutex> ledger_mutex_ = std::make_shared<std::mutex>();
  std::size_t limit_;
  std::pmr::memory_resource* upstream_;
  std::size_t live_ = 0;
  std::size_t external_upper_ = 0;
  std::size_t batch_upper_ = 0;
  std::size_t peak_ = 0;
  std::size_t requested_ = 0;
  std::size_t allocations_ = 0;
  std::size_t refusals_ = 0;
  std::size_t bookkeeping_ = 0;
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
                              MemoryOwner owner = scratch_memory_owner());
  ~MemoryBudgetWindow();
  MemoryBudgetWindow(const MemoryBudgetWindow&) = delete;
  MemoryBudgetWindow& operator=(const MemoryBudgetWindow&) = delete;
  const MemoryOwner& owner() const noexcept { return owner_; }
  std::array<std::uint64_t, 6> evidence() const noexcept {
    const std::lock_guard lock(owner_->mutex_);
    return {static_cast<std::uint64_t>(limit_),
            static_cast<std::uint64_t>(observation_.live),
            static_cast<std::uint64_t>(observation_.peak),
            static_cast<std::uint64_t>(observation_.last_request),
            static_cast<std::uint64_t>(observation_.allocations),
            static_cast<std::uint64_t>(observation_.refusals)};
  }
  std::array<std::uint64_t, 2> external_evidence() const noexcept {
    // Reservations are not host payload peaks.
    const std::lock_guard lock(owner_->mutex_);
    return {static_cast<std::uint64_t>(owner_->external_upper_ + owner_->batch_upper_),
            static_cast<std::uint64_t>(observation_.external_peak_upper)};
  }
 private:
  MemoryOwner owner_;
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
class NativeExecutionBatch;
extern thread_local NativeExecutionScope* active_execution_scope;

class NativeExecutionScope final {
 public:
  NativeExecutionScope(uint64_t work, uint64_t queries, uint64_t cavity,
                       std::size_t scratch, double seconds,
                       MemoryOwner owner = scratch_memory_owner());
  ~NativeExecutionScope();
  NativeExecutionScope(const NativeExecutionScope&) = delete;
  NativeExecutionScope& operator=(const NativeExecutionScope&) = delete;

  static bool is_active_or_ancestor(const NativeExecutionScope* candidate) noexcept {
    for (auto* scope = active_execution_scope; scope != nullptr; scope = scope->previous_) {
      if (scope == candidate) return true;
    }
    return false;
  }

  bool admit(uint64_t work, uint64_t queries = 0) noexcept;
  bool charge(uint64_t work, uint64_t queries = 0) noexcept;
  bool charge_external(uint64_t work, uint64_t queries) noexcept;
  bool import_preparation(uint64_t work, uint64_t queries, double seconds) noexcept;
  double prior_seconds() const noexcept;
  bool admit_cavity(std::size_t cells) noexcept;
  void refuse_external_storage(uint64_t bytes = 0) noexcept;
  int32_t status() const noexcept;
  bool primitive_query() noexcept;
  bool checkpoint() noexcept;
  // charge(0), with the same status, target refusal and admission checks,
  // except that it reads the deadline every time, not on the logical cadence.
  bool checkpoint_charge() noexcept;
  bool has_live_batches() const noexcept;
  int32_t batch_refusal(uint64_t* values) const noexcept;
  int32_t refusal_request(uint64_t* values) const noexcept;
  std::array<uint64_t, PHX_MC_EXECUTION_COUNTERS> evidence() const noexcept;
  std::array<uint64_t, PHX_MC_EXECUTION_MEMORY_VALUES> memory_evidence() const noexcept;
  uint64_t allocator_bookkeeping() const noexcept;
  double seconds() const noexcept;
  std::array<uint64_t, 4> remaining() const noexcept;
  double remaining_wall_seconds() const noexcept;
 private:
  using Clock = std::chrono::steady_clock;
  friend class NativeExecutionBatch;
  friend class NativeParallelBatch;
  bool checkpoint_locked(Clock::time_point now) noexcept;
  bool logical_checkpoint_locked(bool read_deadline = false) noexcept;
  bool admit_locked(uint64_t work, uint64_t queries) noexcept;
  void commit_locked(uint64_t work, uint64_t queries, bool external) noexcept;
  void refuse_locked(std::size_t resource) noexcept;
  int32_t status_locked() const noexcept;
  std::array<uint64_t, 4> remaining_locked() const noexcept;
  NativeExecutionScope* previous_;
  NativeExecutionTarget* borrowed_target_;
  MemoryBudgetWindow memory_;
  std::shared_ptr<std::mutex> ledger_mutex_;
  std::thread::id thread_ = std::this_thread::get_id();
  uint64_t work_limit_, query_limit_, cavity_limit_;
  uint64_t work_ = 0, queries_ = 0, peak_cavity_ = 0;
  uint64_t borrowed_cavity_ = 0;
  uint64_t reserved_work_ = 0, reserved_queries_ = 0, reserved_cavity_ = 0;
  uint64_t reserved_bytes_ = 0;
  uint64_t live_batches_ = 0;
  uint64_t live_scopes_ = 0, live_targets_ = 0;
  uint16_t checkpoint_charges_ = 0;
  uint64_t work_refusals_ = 0, query_refusals_ = 0, cavity_refusals_ = 0;
  uint64_t external_work_ = 0, external_queries_ = 0;
  uint64_t primitive_queries_ = 0;
  double prior_seconds_ = 0.0;
  int32_t status_ = PHX_MC_OK;
  ExecutionBatchRefusal batch_refusal_;
  bool has_batch_refusal_ = false;
  std::array<uint64_t, 4> refusal_request_{};
  bool has_refusal_request_ = false;
  Clock::time_point start_, deadline_;
  // Captured absolute policy expiry is never shifted by imported preparation.
  Clock::time_point workflow_deadline_;
  friend class NativeExecutionTarget;
  friend class BoundedMemoryResource;
  MemoryScope memory_scope_;
};

// Shared allocation state outlives ticket handles and lexical scopes. It never
// contains borrowed scope/mutex pointers; escaped payloads retain this state.
struct NativeBatchStorage {
  explicit NativeBatchStorage(std::shared_ptr<NativeBatchStorage> parent_state)
      : parent(std::move(parent_state)) {}
  std::shared_ptr<NativeBatchStorage> parent;
  phx_mc_execution_usage usage{};
  uint64_t inclusive_live = 0;
  bool reservation_open = true;
};

class NativeExecutionBatch final {
 public:
  static int32_t admit(NativeExecutionScope& scope, NativeExecutionBatch* parent,
                       std::array<uint64_t, 4> upper, void** out);
  int32_t settle(std::array<uint64_t, 4> actual) noexcept;
  // A proven refusal settles and must then be destroyed (`settled`); with live
  // children it records the refusal and keeps the ticket for a later settle.
  int32_t refuse(std::array<uint64_t, 4> actual, std::array<uint64_t, 4> requested,
                 uint64_t* out, bool& settled) noexcept;
  int32_t checkpoint() noexcept;
  int32_t usage(phx_mc_execution_usage** out) noexcept;
 private:
  friend class NativeExecutionTarget;
  friend class NativeExecutionScope;
  friend class BoundedMemoryResource;
  friend class NativeHostStorage;
  friend class NativeParallelBatch;
  static void capture_refusal_locked(
      NativeExecutionScope& scope, NativeExecutionBatch* parent,
      std::array<uint64_t, 4> requested,
      NativeExecutionScope* admitting_scope = nullptr) noexcept;
  std::array<uint64_t, 4> remaining_locked() const noexcept;
  ExecutionBatchRefusal snapshot_locked() const noexcept;
  int32_t validate_actual_locked(std::array<uint64_t, 4>& actual,
                                 bool live_children_allowed = false) const noexcept;
  void settle_locked(std::array<uint64_t, 4> actual,
                     std::array<uint64_t, 4> host) noexcept;
  NativeExecutionBatch(NativeExecutionScope& scope, NativeExecutionBatch* parent,
                       std::array<uint64_t, 4> upper);
  NativeExecutionScope& scope_;
  NativeExecutionBatch* parent_;
  std::array<uint64_t, 4> upper_;
  std::shared_ptr<NativeBatchStorage> storage_;
  std::array<uint64_t, 2> external_consumed_{};
  uint64_t child_host_peak_bytes_ = 0;
  uint64_t live_children_ = 0;
  bool target_active_ = false;
  bool has_usage_ = false;
};

class NativeExecutionTarget final {
 public:
  explicit NativeExecutionTarget(NativeExecutionBatch& ticket);
  ~NativeExecutionTarget();
  NativeExecutionTarget(const NativeExecutionTarget&) = delete;
  NativeExecutionTarget& operator=(const NativeExecutionTarget&) = delete;
  int32_t close() noexcept;
  phx_mc_execution_usage* usage() noexcept { return &ticket_.storage_->usage; }
 private:
  friend class NativeExecutionScope;
  friend class NativeExecutionBatch;
  friend class BoundedMemoryResource;
  friend class MemoryBudgetWindow;
  friend class NativeHostStorage;
  friend class NativeParallelBatch;
  bool admit_locked(uint64_t work, uint64_t queries) noexcept;
  void commit_locked(uint64_t work, uint64_t queries, bool external) noexcept;
  bool cavity_locked(uint64_t cells) noexcept;
  void refuse_locked(std::size_t resource, std::array<uint64_t, 4> requested = {}) noexcept;
  std::array<uint64_t, 4> remaining_locked() const noexcept;
  NativeExecutionBatch& ticket_;
  NativeExecutionTarget* previous_target_;
  NativeExecutionScope* previous_scope_;
  MemoryOwner previous_memory_;
  std::thread::id thread_;
  bool closed_ = false;
};

struct NativeChunkAllowance {
  uint64_t work = 0;
  uint64_t queries = 0;
  uint64_t cavity = 0;
  // Fixed arena capacity, including native allocation headers and alignment.
  uint64_t bytes = 0;
};

struct NativeChunkEvidence {
  ExecutionCounters counters;
  ManagedMemory memory;
  int32_t status = PHX_MC_OK;
  bool completed = false;
};

// Bounds one allocation in a recyclable chunk arena. Sum cumulative requests
// unless lifetimes prove a tighter fragmentation bound. Overflow is a caller
// bound defect, detected before launching any chunk.
uint64_t native_parallel_allocation_bytes(
    uint64_t payload, std::size_t alignment = alignof(std::max_align_t));

// Admits all chunk allowances as one ticket. With arena_slots == 0 every chunk
// owns one arena (chunk outputs may retain it). Otherwise arena_slots
// recyclable arenas, each the largest chunk byte bound, serve the chunks: a
// participant reuses its slot only after its chunk released every allocation,
// so each chunk starts from an identical empty arena and the admitted bytes
// scale with participants, not with the batch's chunk count.
class NativeParallelBatch final {
 public:
  explicit NativeParallelBatch(std::span<const NativeChunkAllowance> allowances,
                               std::size_t arena_slots = 0);
  ~NativeParallelBatch();
  NativeParallelBatch(const NativeParallelBatch&) = delete;
  NativeParallelBatch& operator=(const NativeParallelBatch&) = delete;
  int32_t status() const noexcept { return status_; }
  std::size_t arena_count() const noexcept { return owners_.size(); }
  const MemoryOwner& arena_owner(std::size_t index) const noexcept { return owners_[index]; }
  const NativeChunkAllowance& chunk_allowance(std::size_t index) const noexcept {
    return allowances_[index];
  }
  double remaining_wall_seconds() const noexcept;
  int32_t settle(std::span<const NativeChunkEvidence> evidence) noexcept;
 private:
  void release() noexcept;
  void fail_internal() noexcept;
  std::unique_ptr<NativeExecutionScope> implicit_scope_;
  NativeExecutionScope* scope_ = nullptr;
  std::unique_ptr<NativeExecutionBatch> ticket_;
  std::vector<NativeChunkAllowance> allowances_;
  std::vector<MemoryOwner> owners_;
  std::thread::id thread_ = std::this_thread::get_id();
  int32_t status_ = PHX_MC_OK;
};

// Clearing TLS precedes local scope construction; restoration follows scope
// destruction, including construction failures and exception unwinding.
class NativeDetachedExecution final {
 public:
  NativeDetachedExecution(NativeChunkAllowance allowance, MemoryOwner owner);
  NativeDetachedExecution(const NativeDetachedExecution&) = delete;
  NativeDetachedExecution& operator=(const NativeDetachedExecution&) = delete;
  NativeExecutionScope& execution() noexcept { return execution_; }
  NativeChunkEvidence evidence(int32_t status = PHX_MC_OK,
                               bool completed = true) const noexcept;
 private:
  struct Affinity final {
    Affinity() noexcept;
    ~Affinity();
    NativeExecutionScope* scope;
    NativeExecutionTarget* target;
    MemoryOwner memory;
  };
  Affinity affinity_;
  NativeExecutionScope execution_;
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
// Reads the deadline at every call; fixed-chunk validation polls with it.
inline void native_execution_checkpoint() {
  if (active_execution_scope != nullptr && !active_execution_scope->checkpoint_charge()) {
    throw ExecutionRefusal{active_execution_scope->status()};
  }
}
inline void native_execution_primitive_query() {
  if (active_execution_scope != nullptr && !active_execution_scope->primitive_query()) {
    throw ExecutionRefusal{active_execution_scope->status()};
  }
}
inline bool native_execution_cavity(std::size_t cells) noexcept {
  return active_execution_scope == nullptr || active_execution_scope->admit_cavity(cells);
}
// Refusal recorded on the active scope chain; PHX_MC_OK without a scope.
inline int32_t native_execution_refusal() noexcept {
  return active_execution_scope == nullptr ? PHX_MC_OK : active_execution_scope->status();
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
