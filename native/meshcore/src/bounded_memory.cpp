//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include "bounded_memory.hpp"
#include "counter_records.hpp"
#include "tet_mesh.hpp"

namespace phx::mc {
thread_local MemoryOwner active_memory_owner;
thread_local NativeExecutionScope* active_execution_scope = nullptr;
thread_local NativeExecutionTarget* active_execution_target = nullptr;

namespace {
static_assert(std::atomic_ref<uint64_t>::is_always_lock_free);
static_assert(alignof(phx_mc_execution_usage) >= std::atomic_ref<uint64_t>::required_alignment);
uint64_t load_shared(const uint64_t& value) noexcept {
  return std::atomic_ref<uint64_t>(const_cast<uint64_t&>(value)).load(std::memory_order_acquire);
}
void store_shared(uint64_t& value, uint64_t amount) noexcept {
  std::atomic_ref<uint64_t>(value).store(amount, std::memory_order_release);
}
void add_shared(uint64_t& value, uint64_t amount) noexcept {
  std::atomic_ref<uint64_t>(value).fetch_add(amount, std::memory_order_release);
}
void subtract_shared(uint64_t& value, uint64_t amount) noexcept {
  std::atomic_ref<uint64_t>(value).fetch_sub(amount, std::memory_order_release);
}
void peak_shared(uint64_t& value, uint64_t peak) noexcept {
  std::atomic_ref<uint64_t> counter(value);
  auto old = counter.load(std::memory_order_relaxed);
  while (old < peak && !counter.compare_exchange_weak(old, peak, std::memory_order_release, std::memory_order_relaxed)) {}
}
uint64_t available(uint64_t upper, uint64_t first, uint64_t second = 0,
                   uint64_t third = 0, uint64_t fourth = 0) noexcept {
  for (const auto used : {first, second, third, fourth}) {
    if (used > upper) return 0;
    upper -= used;
  }
  return upper;
}
struct AllocationHeader {
  void* raw;
  std::size_t total;
  std::size_t alignment;
  std::thread::id thread;
  std::size_t allocation;
  std::shared_ptr<NativeBatchStorage> storage;
};
}

std::size_t BoundedMemoryResource::effective_limit_locked() const noexcept {
  auto result = limit_;
  for (auto* observer = observation_; observer != nullptr; observer = observer->previous) {
    if (observer->aggregate || observer->thread == std::this_thread::get_id()) {
      result = std::min(result, observer->managed_limit);
    }
  }
  return result;
}
std::size_t BoundedMemoryResource::available_locked() const noexcept {
  std::size_t result = effective_limit_locked() - live_ - external_upper_ - batch_upper_;
  for (auto* observer = observation_; observer != nullptr; observer = observer->previous) {
    if (!observer->aggregate && observer->thread != std::this_thread::get_id()) continue;
    const auto external = external_upper_ > observer->external_baseline ? external_upper_ - observer->external_baseline : 0;
    result = std::min(result, static_cast<std::size_t>(available(observer->limit, observer->live, external)));
  }
  return result;
}
std::size_t BoundedMemoryResource::available_bytes() const noexcept {
  const std::lock_guard lock(mutex_);
  return available_locked();
}
void BoundedMemoryResource::change_storage_locked(
    const std::shared_ptr<NativeBatchStorage>& storage, std::size_t bytes, bool increase) noexcept {
  auto* state = storage.get();
  bool settled_branch = false;
  while (state) {
    if (increase) state->inclusive_live += bytes; else state->inclusive_live -= bytes;
    if (settled_branch) {
      if (increase) add_shared(state->usage.children_live_bytes, bytes);
      else subtract_shared(state->usage.children_live_bytes, bytes);
    }
    settled_branch = !state->reservation_open;
    if (!state->parent && state->reservation_open) {
      if (increase) batch_upper_ -= bytes; else batch_upper_ += bytes;
    }
    state = state->parent.get();
  }
}
bool BoundedMemoryResource::resize_external_storage(
    std::size_t previous, std::size_t requested,
    const std::shared_ptr<NativeBatchStorage>& storage) noexcept {
  const std::lock_guard ledger(*ledger_mutex_);
  std::unique_lock pool(mutex_);
  if (requested > previous) {
    const auto increase = requested - previous;
    auto* target = active_execution_target;
    const auto refuse = [&]() noexcept {
      pool.unlock();
      if (target != nullptr) target->refuse_locked(3, {0, 0, 0, static_cast<uint64_t>(requested)});
      return false;
    };
    if (target != nullptr && (!storage || storage != target->ticket_.storage_)) return refuse();
    if (storage && !storage->reservation_open) {
      auto* root = storage.get();
      while (root->parent) root = root->parent.get();
      if (root->reservation_open) return refuse();
    }
    if (storage && storage->reservation_open) {
      if (target == nullptr || increase > target->remaining_locked()[3]) return refuse();
      for (auto* observer = observation_; observer != nullptr; observer = observer->previous) {
        if (!observer->aggregate && observer->thread != std::this_thread::get_id()) continue;
        const auto external = external_upper_ > observer->external_baseline ? external_upper_ - observer->external_baseline : 0;
        if (increase > available(observer->limit, observer->live, external)) return refuse();
      }
    } else if (increase > available_locked()) return refuse();
    if (storage) {
      add_shared(storage->usage.retained_host_bytes, increase);
      change_storage_locked(storage, increase, true);
    }
  } else if (storage) {
    subtract_shared(storage->usage.retained_host_bytes, previous - requested);
    change_storage_locked(storage, previous - requested, false);
  }
  external_upper_ = external_upper_ - previous + requested;
  if (storage) {
    auto& u = storage->usage;
    peak_shared(u.combined_peak_bytes,
        load_shared(u.host_bytes) + load_shared(u.native_bytes) + load_shared(u.retained_host_bytes) +
        load_shared(u.children_live_bytes) + load_shared(u.reserved_bytes));
  }
  for (auto* observer = observation_; observer != nullptr; observer = observer->previous) {
    if (observer->aggregate || observer->thread == std::this_thread::get_id()) {
      observer->external_peak_upper = std::max(observer->external_peak_upper, external_upper_);
    }
  }
  return true;
}
void BoundedMemoryResource::release_external_storage(
    std::size_t bytes, const std::shared_ptr<NativeBatchStorage>& storage) noexcept {
  const std::lock_guard ledger(*ledger_mutex_);
  const std::lock_guard pool(mutex_);
  if (storage) {
    subtract_shared(storage->usage.retained_host_bytes, bytes);
    change_storage_locked(storage, bytes, false);
  }
  external_upper_ -= bytes;
}

MemoryBudgetWindow::MemoryBudgetWindow(std::size_t additional_limit, MemoryOwner owner)
    : owner_(owner ? std::move(owner)
                   : std::make_shared<BoundedMemoryResource>(additional_limit)) {
  const std::lock_guard ledger(*owner_->ledger_mutex_);
  const std::lock_guard pool(owner_->mutex_);
  std::size_t allowance = owner_->available_locked();
  if (active_execution_target != nullptr && scratch_memory_owner() == owner_) {
    allowance = static_cast<std::size_t>(active_execution_target->remaining_locked()[3]);
    // The reservation is already inside the pool cap; a borrowed window must
    // not subtract it a second time. Existing phase windows still bound live.
    for (auto* observer = owner_->observation_; observer != nullptr; observer = observer->previous) {
      if (!observer->aggregate && observer->thread != std::this_thread::get_id()) continue;
      const auto external = owner_->external_upper_ > observer->external_baseline ? owner_->external_upper_ - observer->external_baseline : 0;
      allowance = std::min(allowance, static_cast<std::size_t>(available(observer->limit, observer->live, external)));
    }
  }
  limit_ = std::min(additional_limit, allowance);
  observation_.previous = owner_->observation_;
  observation_.limit = limit_;
  observation_.external_baseline = owner_->external_upper_;
  const auto base = owner_->live_ + owner_->external_upper_ + owner_->batch_upper_;
  observation_.managed_limit = std::min(owner_->effective_limit_locked(), base + std::min(limit_, SIZE_MAX - base));
  observation_.thread = std::this_thread::get_id();
  observation_.aggregate = active_execution_scope == nullptr;
  observation_.allocation_start = owner_->allocations_;
  observation_.external_peak_upper = owner_->external_upper_;
  owner_->observation_ = &observation_;
}
MemoryBudgetWindow::~MemoryBudgetWindow() {
  const std::lock_guard ledger(*owner_->ledger_mutex_);
  const std::lock_guard pool(owner_->mutex_);
  // Worker phases need not finish in the order their observations were added.
  auto** link = &owner_->observation_;
  while (*link != &observation_) link = &(*link)->previous;
  *link = observation_.previous;
}

void* BoundedMemoryResource::do_allocate(std::size_t bytes, std::size_t alignment) {
  if (active_execution_target != nullptr &&
      active_execution_target->ticket_.scope_.memory_.owner().get() != this) {
    // Retained buffers must use a target bound to their pool.
    active_execution_scope->refuse_external_storage(bytes);
    const std::lock_guard ledger(*ledger_mutex_);
    const std::lock_guard pool(mutex_);
    requested_ = bytes;
    if (refusals_ != SIZE_MAX) ++refusals_;
    for (auto* observer = observation_; observer != nullptr; observer = observer->previous) {
      if (!observer->aggregate && observer->thread != std::this_thread::get_id()) continue;
      observer->last_request = bytes;
      if (observer->refusals != SIZE_MAX) ++observer->refusals;
    }
    throw std::bad_alloc();
  }
  const bool foreign_scope = active_execution_scope != nullptr &&
      active_execution_scope->memory_.owner().get() != this;
  const bool foreign_allowed = !foreign_scope || native_execution_spend(0, 0);
  const std::lock_guard ledger(*ledger_mutex_);
  auto* scope = active_execution_scope;
  auto* target = active_execution_target;
  const bool execution_allowed = scope == nullptr || (foreign_scope ? foreign_allowed : scope->logical_checkpoint_locked());
  const bool ticket_allowed = target == nullptr || bytes <= target->remaining_locked()[3];
  std::unique_lock pool(mutex_);
  requested_ = bytes;
  for (auto* observer = observation_; observer != nullptr; observer = observer->previous) {
    if (observer->aggregate || observer->thread == std::this_thread::get_id()) observer->last_request = bytes;
  }
  bool window_allowed = true;
  for (auto* observer = observation_; observer != nullptr; observer = observer->previous) {
    if (!observer->aggregate && observer->thread != std::this_thread::get_id()) continue;
    const auto external = external_upper_ > observer->external_baseline ? external_upper_ - observer->external_baseline : 0;
    if (bytes > available(observer->limit, observer->live, external)) window_allowed = false;
  }
  const auto upstream_alignment = std::max(alignment, alignof(AllocationHeader));
  const bool overflow = sizeof(AllocationHeader) > SIZE_MAX - (upstream_alignment - 1) ||
      bytes > SIZE_MAX - sizeof(AllocationHeader) - (upstream_alignment - 1);
  const auto prefix = overflow ? 0 : (sizeof(AllocationHeader) + upstream_alignment - 1) & ~(upstream_alignment - 1);
  if (!execution_allowed || !ticket_allowed || !window_allowed || overflow ||
      (target == nullptr && bytes > effective_limit_locked() - live_ - external_upper_ - batch_upper_) ||
      allocations_ == SIZE_MAX || prefix > SIZE_MAX - bookkeeping_) {
    if (refusals_ != SIZE_MAX) ++refusals_;
    for (auto* observer = observation_; observer != nullptr; observer = observer->previous) {
      if ((observer->aggregate || observer->thread == std::this_thread::get_id()) && observer->refusals != SIZE_MAX) ++observer->refusals;
    }
    pool.unlock();
    if (target != nullptr) target->refuse_locked(3, {0, 0, 0, bytes});
    throw std::bad_alloc();
  }
  const auto total = prefix + bytes;
  void* raw;
  try {
    raw = upstream_->allocate(total, upstream_alignment);
  } catch (const std::bad_alloc&) {
    if (refusals_ != SIZE_MAX) ++refusals_;
    for (auto* observer = observation_; observer != nullptr; observer = observer->previous) {
      if ((observer->aggregate || observer->thread == std::this_thread::get_id()) && observer->refusals != SIZE_MAX) ++observer->refusals;
    }
    pool.unlock();
    if (target != nullptr) target->refuse_locked(3, {0, 0, 0, bytes});
    throw;
  }
  auto* buffer = static_cast<std::byte*>(raw) + prefix;
  const std::shared_ptr<NativeBatchStorage> untargeted;
  const auto& storage = target == nullptr ? untargeted : target->ticket_.storage_;
  std::construct_at(reinterpret_cast<AllocationHeader*>(buffer - sizeof(AllocationHeader)),
                    AllocationHeader{raw, total, upstream_alignment, std::this_thread::get_id(), allocations_ + 1, storage});
  if (storage) {
    add_shared(storage->usage.native_bytes, bytes);
    peak_shared(storage->usage.native_peak_bytes, load_shared(storage->usage.native_bytes));
    change_storage_locked(storage, bytes, true);
    auto& usage = storage->usage;
    peak_shared(usage.combined_peak_bytes,
        load_shared(usage.host_bytes) + load_shared(usage.native_bytes) + load_shared(usage.retained_host_bytes) +
        load_shared(usage.reserved_bytes) + load_shared(usage.children_live_bytes));
    // Host usage is conservative storage, not a native allocation request.
    for (auto* observer = observation_; observer != nullptr; observer = observer->previous) {
      if (observer->aggregate || observer->thread == std::this_thread::get_id()) {
        observer->external_peak_upper = std::max(observer->external_peak_upper, load_shared(usage.host_bytes));
      }
    }
  }
  bookkeeping_ += prefix;
  live_ += bytes;
  ++allocations_;
  peak_ = std::max(peak_, live_);
  for (auto* observer = observation_; observer != nullptr; observer = observer->previous) {
    if (!observer->aggregate && observer->thread != std::this_thread::get_id()) continue;
    observer->live += bytes;
    ++observer->allocations;
    observer->peak = std::max(observer->peak, observer->live);
  }
  return buffer;
}
void BoundedMemoryResource::do_deallocate(void* pointer, std::size_t bytes, std::size_t) {
  auto* header = reinterpret_cast<AllocationHeader*>(static_cast<std::byte*>(pointer) - sizeof(AllocationHeader));
  const auto& storage = header->storage;
  const std::lock_guard ledger(*ledger_mutex_);
  const std::lock_guard pool(mutex_);
  if (storage) {
    subtract_shared(storage->usage.native_bytes, bytes);
    change_storage_locked(storage, bytes, false);
  }
  const auto raw = header->raw;
  const auto total = header->total;
  const auto alignment = header->alignment;
  for (auto* observer = observation_; observer != nullptr; observer = observer->previous) {
    if (header->allocation > observer->allocation_start && (observer->aggregate || observer->thread == header->thread)) observer->live -= bytes;
  }
  bookkeeping_ -= total - bytes;
  std::destroy_at(header);
  upstream_->deallocate(raw, total, alignment);
  live_ -= bytes;
}

NativeExecutionScope::NativeExecutionScope(
    uint64_t work, uint64_t queries, uint64_t cavity, std::size_t scratch,
    double seconds, MemoryOwner owner)
    : previous_(active_execution_scope), borrowed_target_(active_execution_target),
      memory_(scratch, std::move(owner)), ledger_mutex_(memory_.owner()->ledger_mutex()),
      work_limit_(work), query_limit_(queries), cavity_limit_(cavity), start_(Clock::now()),
      deadline_(seconds == std::numeric_limits<double>::infinity() ? Clock::time_point::max()
          : start_ + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds))),
      memory_scope_(memory_.owner()) {
  const std::lock_guard lock(*ledger_mutex_);
  if (previous_ != nullptr && previous_->ledger_mutex_ != ledger_mutex_) {
    throw std::invalid_argument("native.execution.scope.memory_owner must match its parent pool; got a different pool");
  }
  if (borrowed_target_ != nullptr) borrowed_cavity_ = load_shared(borrowed_target_->usage()->native_cavity);
  for (auto* parent = previous_; parent != nullptr; parent = parent->previous_) ++parent->live_scopes_;
  active_execution_scope = this;
}
NativeExecutionScope::~NativeExecutionScope() {
  {
    const std::lock_guard lock(*ledger_mutex_);
    if (active_execution_scope != this || live_batches_ != 0 || live_targets_ != 0 || live_scopes_ != 0) std::terminate();
    if (borrowed_target_ != nullptr) store_shared(borrowed_target_->usage()->native_cavity, borrowed_cavity_);
    for (auto* parent = previous_; parent != nullptr; parent = parent->previous_) --parent->live_scopes_;
    active_execution_scope = previous_;
  }
}
int32_t NativeExecutionScope::status_locked() const noexcept {
  for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
    if (scope->status_ != PHX_MC_OK) return scope->status_;
  }
  return PHX_MC_OK;
}
bool NativeExecutionScope::checkpoint_locked(Clock::time_point now) noexcept {
  for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
    if (scope->status_ != PHX_MC_OK) return false;
    if (now >= scope->deadline_) { scope->status_ = PHX_MC_TIMEOUT; return false; }
  }
  return true;
}
bool NativeExecutionScope::logical_checkpoint_locked() noexcept {
  if (status_locked() != PHX_MC_OK) return false;
  const bool sample = checkpoint_charges_ == 0;
  checkpoint_charges_ = static_cast<uint16_t>((checkpoint_charges_ + 1) % 1024);
  const bool allowed = !sample || checkpoint_locked(Clock::now());
  if (!allowed && active_execution_target != nullptr) active_execution_target->refuse_locked(0);
  return allowed;
}
void NativeExecutionScope::refuse_locked(std::size_t resource) noexcept {
  for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
    uint64_t* counter = resource == 0 ? &scope->work_refusals_
                       : resource == 1 ? &scope->query_refusals_
                       : resource == 2 ? &scope->cavity_refusals_ : nullptr;
    if (counter != nullptr && *counter != UINT64_MAX) ++*counter;
    scope->status_ = PHX_MC_CAPACITY_EXCEEDED;
  }
}
bool NativeExecutionScope::admit_locked(uint64_t work, uint64_t queries) noexcept {
  auto* target = active_execution_target != nullptr ? active_execution_target : borrowed_target_;
  if (target != nullptr && !target->admit_locked(work, queries)) return false;
  for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
    if (target != nullptr && scope == &target->ticket_.scope_) break;
    if (work > available(scope->work_limit_, scope->work_, scope->reserved_work_)) {
      if (target != nullptr) target->refuse_locked(0, {work, queries, 0, 0});
      else {
        NativeExecutionBatch::capture_refusal_locked(*this, nullptr, {work, queries, 0, 0}, scope);
        refuse_locked(0);
      }
      return false;
    }
    if (queries > available(scope->query_limit_, scope->queries_, scope->reserved_queries_)) {
      if (target != nullptr) target->refuse_locked(1, {work, queries, 0, 0});
      else {
        NativeExecutionBatch::capture_refusal_locked(*this, nullptr, {work, queries, 0, 0}, scope);
        refuse_locked(1);
      }
      return false;
    }
  }
  return true;
}
void NativeExecutionScope::commit_locked(uint64_t work, uint64_t queries, bool external) noexcept {
  for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
    if (borrowed_target_ != nullptr && scope == &borrowed_target_->ticket_.scope_) break;
    scope->work_ += work;
    scope->queries_ += queries;
    if (external) { scope->external_work_ += work; scope->external_queries_ += queries; }
  }
}
bool NativeExecutionScope::admit(uint64_t work, uint64_t queries) noexcept {
  const std::lock_guard lock(*ledger_mutex_);
  return checkpoint_locked(Clock::now()) && admit_locked(work, queries);
}
bool NativeExecutionScope::charge(uint64_t work, uint64_t queries) noexcept {
  const std::lock_guard lock(*ledger_mutex_);
  if (!logical_checkpoint_locked() || !admit_locked(work, queries)) return false;
  auto* target = active_execution_target != nullptr ? active_execution_target : borrowed_target_;
  if (target != nullptr) target->commit_locked(work, queries, false);
  else commit_locked(work, queries, false);
  return true;
}
bool NativeExecutionScope::charge_external(uint64_t work, uint64_t queries) noexcept {
  const std::lock_guard lock(*ledger_mutex_);
  if (!logical_checkpoint_locked() || !admit_locked(work, queries)) return false;
  auto* target = active_execution_target != nullptr ? active_execution_target : borrowed_target_;
  if (target != nullptr) target->commit_locked(work, queries, true);
  else commit_locked(work, queries, true);
  return true;
}
bool NativeExecutionScope::import_preparation(uint64_t work, uint64_t queries, double seconds) noexcept {
  const std::lock_guard lock(*ledger_mutex_);
  const auto now = Clock::now();
  if (!checkpoint_locked(now) || !admit_locked(work, queries)) return false;
  for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
    if (scope->deadline_ != Clock::time_point::max() &&
        seconds > std::chrono::duration<double>(scope->deadline_ - now).count()) {
      scope->status_ = PHX_MC_TIMEOUT; return false;
    }
    if (!std::isfinite(scope->prior_seconds_ + seconds)) return false;
  }
  auto* target = active_execution_target != nullptr ? active_execution_target : borrowed_target_;
  if (target != nullptr) target->commit_locked(work, queries, true);
  else commit_locked(work, queries, true);
  const auto debit = std::chrono::ceil<Clock::duration>(std::chrono::duration<double>(seconds));
  for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
    scope->prior_seconds_ += seconds;
    if (scope->deadline_ != Clock::time_point::max()) scope->deadline_ -= debit;
  }
  return true;
}
double NativeExecutionScope::prior_seconds() const noexcept {
  const std::lock_guard lock(*ledger_mutex_); return prior_seconds_;
}
bool NativeExecutionScope::admit_cavity(std::size_t cells) noexcept {
  const std::lock_guard lock(*ledger_mutex_);
  if (!logical_checkpoint_locked()) return false;
  auto* target = active_execution_target != nullptr ? active_execution_target : borrowed_target_;
  for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
    if (target != nullptr && scope == &target->ticket_.scope_) break;
    if (cells > available(scope->cavity_limit_, scope->reserved_cavity_)) {
      if (target != nullptr) target->refuse_locked(2, {0, 0, static_cast<uint64_t>(cells), 0}); else refuse_locked(2);
      return false;
    }
  }
  if (target != nullptr && !target->cavity_locked(cells)) return false;
  for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
    if (target != nullptr && scope == &target->ticket_.scope_) break;
    scope->peak_cavity_ = std::max(scope->peak_cavity_, scope->reserved_cavity_ + static_cast<uint64_t>(cells));
  }
  return true;
}
void NativeExecutionScope::refuse_external_storage(uint64_t bytes) noexcept {
  const std::lock_guard lock(*ledger_mutex_);
  if (active_execution_target != nullptr) active_execution_target->refuse_locked(3, {0, 0, 0, bytes});
  else refuse_locked(3);
}
int32_t NativeExecutionScope::status() const noexcept {
  const std::lock_guard lock(*ledger_mutex_); return status_locked();
}
bool NativeExecutionScope::primitive_query() noexcept {
  const std::lock_guard lock(*ledger_mutex_);
  if (!logical_checkpoint_locked()) return false;
  // Primitive telemetry is not work/source-query settlement. It is visible in
  // each phase and its enclosing receipt immediately, including under tickets.
  for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
    if (scope->primitive_queries_ != UINT64_MAX) ++scope->primitive_queries_;
  }
  return true;
}
bool NativeExecutionScope::checkpoint() noexcept {
  const std::lock_guard lock(*ledger_mutex_); return checkpoint_locked(Clock::now());
}
bool NativeExecutionScope::has_live_batches() const noexcept {
  const std::lock_guard lock(*ledger_mutex_);
  return live_batches_ != 0 || live_targets_ != 0 || live_scopes_ != 0;
}
int32_t NativeExecutionScope::batch_refusal(uint64_t* values) const noexcept {
  const std::lock_guard lock(*ledger_mutex_);
  if (!has_batch_refusal_) return PHX_MC_INVALID_ARGUMENT;
  const auto& state = batch_refusal_;
  CounterPublisher publisher{state, values};
  return PHX_MC_OK;
}
int32_t NativeExecutionScope::refusal_request(uint64_t* values) const noexcept {
  const std::lock_guard lock(*ledger_mutex_);
  if (!has_refusal_request_) return PHX_MC_INVALID_ARGUMENT;
  const auto state = ExecutionResources::from_slots(refusal_request_.data());
  CounterPublisher publisher{state, values};
  return PHX_MC_OK;
}
std::array<uint64_t, PHX_MC_EXECUTION_COUNTERS> NativeExecutionScope::evidence() const noexcept {
  const std::lock_guard lock(*ledger_mutex_);
  return {work_, queries_, peak_cavity_, work_refusals_, query_refusals_, cavity_refusals_,
          external_work_, external_queries_, primitive_queries_};
}
std::array<uint64_t, PHX_MC_EXECUTION_MEMORY_VALUES> NativeExecutionScope::memory_evidence() const noexcept {
  const std::lock_guard lock(*ledger_mutex_);
  const auto allocated = memory_.evidence();
  const auto external = memory_.external_evidence();
  return {allocated[0], allocated[1], allocated[2], allocated[3], allocated[4], allocated[5],
          external[0], std::max(external[0], external[1])};
}
uint64_t NativeExecutionScope::allocator_bookkeeping() const noexcept {
  const std::lock_guard lock(*ledger_mutex_);
  return static_cast<uint64_t>(memory_.owner()->bookkeeping_bytes());
}
double NativeExecutionScope::seconds() const noexcept {
  return std::chrono::duration<double>(Clock::now() - start_).count();
}
std::array<uint64_t, 4> NativeExecutionScope::remaining() const noexcept {
  const std::lock_guard lock(*ledger_mutex_); return remaining_locked();
}
std::array<uint64_t, 4> NativeExecutionScope::remaining_locked() const noexcept {
  std::array<uint64_t, 4> result;
  result.fill(UINT64_MAX);
  if (borrowed_target_ != nullptr) result = borrowed_target_->remaining_locked();
  for (auto* scope = this; scope != nullptr; scope = scope->previous_) {
    if (borrowed_target_ != nullptr && scope == &borrowed_target_->ticket_.scope_) break;
    result[0] = std::min(result[0], available(scope->work_limit_, scope->work_, scope->reserved_work_));
    result[1] = std::min(result[1], available(scope->query_limit_, scope->queries_, scope->reserved_queries_));
    result[2] = std::min(result[2], available(scope->cavity_limit_, scope->reserved_cavity_));
  }
  if (borrowed_target_ == nullptr) result[3] = memory_.owner()->available_bytes();
  else {
    const auto memory = memory_.evidence();
    result[3] = std::min(result[3], available(memory[0], memory[1], reserved_bytes_));
  }
  return result;
}
double NativeExecutionScope::remaining_wall_seconds() const noexcept {
  const std::lock_guard lock(*ledger_mutex_);
  auto deadline = Clock::time_point::max();
  for (auto* scope = this; scope != nullptr; scope = scope->previous_) deadline = std::min(deadline, scope->deadline_);
  if (deadline == Clock::time_point::max()) return std::numeric_limits<double>::infinity();
  return std::max(0.0, std::chrono::duration<double>(deadline - Clock::now()).count());
}

NativeExecutionBatch::NativeExecutionBatch(NativeExecutionScope& scope, NativeExecutionBatch* parent,
                                         std::array<uint64_t, 4> upper)
    : scope_(scope), parent_(parent), upper_(upper),
      storage_(std::make_shared<NativeBatchStorage>(parent == nullptr ? nullptr : parent->storage_)) {}
std::array<uint64_t, 4> NativeExecutionBatch::remaining_locked() const noexcept {
  const auto& u = storage_->usage;
  return {available(available(upper_[0], load_shared(u.imported_work)), load_shared(u.host_work),
                    load_shared(u.native_work), load_shared(u.children_work), load_shared(u.reserved_work)),
          available(available(upper_[1], load_shared(u.imported_queries)), load_shared(u.host_queries),
                    load_shared(u.native_queries), load_shared(u.children_queries), load_shared(u.reserved_queries)),
          available(upper_[2], load_shared(u.host_cavity), load_shared(u.native_cavity),
                    load_shared(u.reserved_cavity)),
          available(available(upper_[3], load_shared(u.retained_host_bytes)), load_shared(u.host_bytes),
                    load_shared(u.native_bytes), load_shared(u.reserved_bytes), load_shared(u.children_live_bytes))};
}
ExecutionBatchRefusal NativeExecutionBatch::snapshot_locked() const noexcept {
  const auto remaining = remaining_locked();
  auto* root = this;
  while (root->parent_ != nullptr) root = root->parent_;
  uint64_t work = root->scope_.work_, queries = root->scope_.queries_, cavity = root->scope_.peak_cavity_;
  uint64_t bytes = root->scope_.memory_.external_evidence()[1];
  for (auto* ticket = this; ticket != nullptr; ticket = ticket->parent_) {
    const auto& u = ticket->storage_->usage;
    work += load_shared(u.host_work) + load_shared(u.native_work) + load_shared(u.imported_work) + load_shared(u.children_work);
    queries += load_shared(u.host_queries) + load_shared(u.native_queries) + load_shared(u.imported_queries) + load_shared(u.children_queries);
    cavity = std::max(cavity, std::max({load_shared(u.host_peak_cavity), load_shared(u.native_peak_cavity),
                                     load_shared(u.combined_peak_cavity), load_shared(u.children_peak_cavity)}));
    bytes = std::max(bytes, std::max({load_shared(u.host_peak_bytes), load_shared(u.native_peak_bytes),
                                   load_shared(u.combined_peak_bytes), load_shared(u.children_peak_bytes)}));
  }
  return {remaining[0], remaining[1], remaining[2], remaining[3], work, queries, cavity, bytes};
}
void NativeExecutionBatch::capture_refusal_locked(
    NativeExecutionScope& scope, NativeExecutionBatch* parent,
    std::array<uint64_t, 4> requested, NativeExecutionScope* admitting_scope) noexcept {
  if (scope.has_batch_refusal_) return;
  std::array<uint64_t, 4> remaining;
  remaining.fill(UINT64_MAX);
  std::array<NativeExecutionScope*, 4> owners{};
  if (parent != nullptr) {
    remaining = parent->remaining_locked();
    scope.batch_refusal_ = parent->snapshot_locked();
  }
  if (parent == nullptr || &scope != &parent->scope_) {
    for (auto* candidate = &scope; candidate != nullptr; candidate = candidate->previous_) {
      if (scope.borrowed_target_ != nullptr &&
          candidate == &scope.borrowed_target_->ticket_.scope_) break;
      const std::array<uint64_t, 3> local{
          available(candidate->work_limit_, candidate->work_, candidate->reserved_work_),
          available(candidate->query_limit_, candidate->queries_, candidate->reserved_queries_),
          available(candidate->cavity_limit_, candidate->reserved_cavity_)};
      for (std::size_t resource = 0; resource < local.size(); ++resource) {
        if (local[resource] < remaining[resource]) {
          remaining[resource] = local[resource];
          owners[resource] = candidate;
        }
      }
    }
    const auto local = scope.remaining_locked();
    if (local[3] < remaining[3]) {
      remaining[3] = local[3];
      owners[3] = &scope;
    }
  }
  if (admitting_scope == nullptr) {
    for (std::size_t resource = 0; resource < remaining.size(); ++resource) {
      if (requested[resource] > remaining[resource]) {
        admitting_scope = owners[resource];
        break;
      }
    }
  }
  // Remaining bounds can come from several ancestors; consumed evidence belongs
  // to the scope or ticket whose bound actually refused this admission.
  if (admitting_scope == nullptr && parent == nullptr) admitting_scope = &scope;
  if (admitting_scope != nullptr) {
    const auto host = admitting_scope->memory_.external_evidence();
    scope.batch_refusal_.consumed_work = admitting_scope->work_;
    scope.batch_refusal_.consumed_queries = admitting_scope->queries_;
    scope.batch_refusal_.peak_cavity = admitting_scope->peak_cavity_;
    scope.batch_refusal_.peak_bytes = host[1];
  }
  scope.batch_refusal_.remaining_work = remaining[0];
  scope.batch_refusal_.remaining_queries = remaining[1];
  scope.batch_refusal_.remaining_cavity = remaining[2];
  scope.batch_refusal_.remaining_bytes = remaining[3];
  scope.has_batch_refusal_ = true;
  scope.refusal_request_ = requested;
  scope.has_refusal_request_ = true;
  if (scope.borrowed_target_ != nullptr) {
    for (auto* owner = scope.previous_; owner != nullptr; owner = owner->previous_) {
      if (owner->has_batch_refusal_) continue;
      owner->batch_refusal_ = scope.batch_refusal_;
      owner->has_batch_refusal_ = true;
      owner->refusal_request_ = requested;
      owner->has_refusal_request_ = true;
    }
  }
}
int32_t NativeExecutionBatch::admit(NativeExecutionScope& scope, NativeExecutionBatch* parent,
                                  std::array<uint64_t, 4> upper, void** out) {
  const std::lock_guard lock(*scope.ledger_mutex_);
  if (parent == nullptr && scope.borrowed_target_ != nullptr) {
    parent = &(active_execution_target != nullptr ? active_execution_target : scope.borrowed_target_)->ticket_;
  }
  if (parent != nullptr && parent->scope_.ledger_mutex_ != scope.ledger_mutex_) return PHX_MC_INVALID_ARGUMENT;
  if (parent != nullptr && &parent->scope_ != &scope &&
      (scope.borrowed_target_ == nullptr || &scope.borrowed_target_->ticket_ != parent)) return PHX_MC_INVALID_ARGUMENT;
  // The affine host writer must stop while child reservations are created.
  if (parent != nullptr && parent->target_active_ &&
      (active_execution_target == nullptr || &active_execution_target->ticket_ != parent)) return PHX_MC_INVALID_ARGUMENT;
  if (!scope.checkpoint_locked(NativeExecutionScope::Clock::now())) {
    capture_refusal_locked(scope, parent, upper); return scope.status_locked();
  }
  if (parent != nullptr) {
    auto remaining = parent->remaining_locked();
    if (&parent->scope_ != &scope) {
      const auto local = scope.remaining_locked();
      for (std::size_t r = 0; r < 4; ++r) remaining[r] = std::min(remaining[r], local[r]);
    }
    if (parent->live_children_ == UINT64_MAX) return PHX_MC_INTERNAL_ERROR;
    for (std::size_t r = 0; r < upper.size(); ++r) {
      if (upper[r] > remaining[r]) {
        capture_refusal_locked(scope, parent, upper); scope.refuse_locked(r); return PHX_MC_CAPACITY_EXCEEDED;
      }
    }
  } else {
    const auto remaining = scope.remaining_locked();
    for (std::size_t r = 0; r < upper.size(); ++r) {
      if (upper[r] > remaining[r]) {
        capture_refusal_locked(scope, nullptr, upper); scope.refuse_locked(r); return PHX_MC_CAPACITY_EXCEEDED;
      }
    }
  }
  for (auto* owner = &scope; owner != nullptr && (parent == nullptr || owner != &parent->scope_); owner = owner->previous_) {
    if (owner->live_batches_ == UINT64_MAX) return PHX_MC_INTERNAL_ERROR;
  }
  std::unique_ptr<NativeExecutionBatch> ticket;
  try { ticket.reset(new NativeExecutionBatch(scope, parent, upper)); }
  catch (const std::bad_alloc&) {
    capture_refusal_locked(scope, parent, upper); scope.refuse_locked(3); return PHX_MC_CAPACITY_EXCEEDED;
  }
  if (parent != nullptr) {
    auto& u = parent->storage_->usage;
    add_shared(u.reserved_work, upper[0]); add_shared(u.reserved_queries, upper[1]);
    add_shared(u.reserved_cavity, upper[2]); add_shared(u.reserved_bytes, upper[3]);
    ++parent->live_children_;
    for (auto* owner = &scope; owner != nullptr && owner != &parent->scope_; owner = owner->previous_) {
      owner->reserved_work_ += upper[0]; owner->reserved_queries_ += upper[1];
      owner->reserved_cavity_ += upper[2]; owner->reserved_bytes_ += upper[3];
    }
  } else {
    if (!scope.memory_.owner()->reserve_batch_storage(static_cast<std::size_t>(upper[3]))) {
      capture_refusal_locked(scope, nullptr, upper); scope.refuse_locked(3); return PHX_MC_CAPACITY_EXCEEDED;
    }
    for (auto* owner = &scope; owner != nullptr; owner = owner->previous_) {
      owner->reserved_work_ += upper[0]; owner->reserved_queries_ += upper[1]; owner->reserved_cavity_ += upper[2];
    }
  }
  for (auto* owner = &scope; owner != nullptr && (parent == nullptr || owner != &parent->scope_); owner = owner->previous_) ++owner->live_batches_;
  *out = ticket.release();
  return PHX_MC_OK;
}
int32_t NativeExecutionBatch::validate_actual_locked(std::array<uint64_t, 4>& actual) const noexcept {
  if (live_children_ != 0 || target_active_) return PHX_MC_INVALID_ARGUMENT;
  const auto& u = storage_->usage;
  if (has_usage_ && (actual[0] != load_shared(u.host_work) || actual[1] != load_shared(u.host_queries) ||
      actual[2] < load_shared(u.host_peak_cavity) || actual[3] < load_shared(u.host_peak_bytes))) return PHX_MC_INTERNAL_ERROR;
  const std::array<uint64_t, 2> measured{
      load_shared(u.native_work) + load_shared(u.imported_work) + load_shared(u.children_work),
      load_shared(u.native_queries) + load_shared(u.imported_queries) + load_shared(u.children_queries)};
  for (std::size_t r = 0; r < 2; ++r) {
    if (measured[r] > upper_[r] || actual[r] > upper_[r] - measured[r]) return PHX_MC_INTERNAL_ERROR;
    actual[r] += measured[r];
  }
  // Individual peaks need not coexist; combined peaks record simultaneous
  // host/native capacity at native calls. Foreign local writes enforce the same
  // bound against the mapped live counters before performing their work.
  actual[2] = std::max({actual[2], load_shared(u.native_peak_cavity), load_shared(u.combined_peak_cavity), load_shared(u.children_peak_cavity)});
  actual[3] = std::max({actual[3], load_shared(u.native_peak_bytes), load_shared(u.combined_peak_bytes), load_shared(u.children_peak_bytes)});
  if (actual[2] > upper_[2] || actual[3] > upper_[3]) return PHX_MC_INTERNAL_ERROR;
  return PHX_MC_OK;
}
void NativeExecutionBatch::settle_locked(std::array<uint64_t, 4> actual, std::array<uint64_t, 4> host) noexcept {
  const uint64_t external_work = host[0] + external_consumed_[0];
  const uint64_t external_queries = host[1] + external_consumed_[1];
  const uint64_t host_peak = std::max(host[3], child_host_peak_bytes_);
  storage_->reservation_open = false;
  if (parent_ != nullptr) {
    auto& u = parent_->storage_->usage;
    add_shared(u.children_work, actual[0]); add_shared(u.children_queries, actual[1]);
    const auto& own = storage_->usage;
    add_shared(u.children_native_work, load_shared(own.native_work) + load_shared(own.children_native_work));
    add_shared(u.children_native_queries, load_shared(own.native_queries) + load_shared(own.children_native_queries));
    if (actual[2] != 0) peak_shared(u.children_peak_cavity,
        load_shared(u.host_cavity) + load_shared(u.native_cavity) +
        load_shared(u.reserved_cavity) - upper_[2] + actual[2]);
    if (actual[3] != 0) peak_shared(u.children_peak_bytes,
        load_shared(u.host_bytes) + load_shared(u.native_bytes) +
        load_shared(u.retained_host_bytes) + load_shared(u.children_live_bytes) +
        load_shared(u.reserved_bytes) - upper_[3] + actual[3]);
    add_shared(u.children_live_bytes, storage_->inclusive_live);
    subtract_shared(u.reserved_work, upper_[0]); subtract_shared(u.reserved_queries, upper_[1]);
    subtract_shared(u.reserved_cavity, upper_[2]); subtract_shared(u.reserved_bytes, upper_[3]);
    parent_->external_consumed_[0] += external_work; parent_->external_consumed_[1] += external_queries;
    if (host_peak != 0) parent_->child_host_peak_bytes_ = std::max(
        parent_->child_host_peak_bytes_, load_shared(u.host_bytes) +
        load_shared(u.reserved_bytes) + host_peak);
    --parent_->live_children_;
    // A ticket created by a borrowed phase also belongs in that phase receipt.
    for (auto* owner = &scope_; owner != nullptr && owner != &parent_->scope_; owner = owner->previous_) {
      owner->reserved_work_ -= upper_[0]; owner->reserved_queries_ -= upper_[1];
      owner->reserved_cavity_ -= upper_[2]; owner->reserved_bytes_ -= upper_[3];
      owner->work_ += actual[0]; owner->queries_ += actual[1];
      owner->external_work_ += external_work; owner->external_queries_ += external_queries;
      owner->peak_cavity_ = std::max(owner->peak_cavity_, owner->reserved_cavity_ + actual[2]);
    }
  } else {
    for (auto* owner = &scope_; owner != nullptr; owner = owner->previous_) {
      owner->reserved_work_ -= upper_[0]; owner->reserved_queries_ -= upper_[1]; owner->reserved_cavity_ -= upper_[2];
      owner->work_ += actual[0]; owner->queries_ += actual[1];
      owner->external_work_ += external_work; owner->external_queries_ += external_queries;
      if (actual[2] != 0) owner->peak_cavity_ = std::max(owner->peak_cavity_, owner->reserved_cavity_ + actual[2]);
    }
    scope_.memory_.owner()->settle_batch_storage(static_cast<std::size_t>(upper_[3] - storage_->inclusive_live),
                                               static_cast<std::size_t>(host_peak));
  }
  for (auto* owner = &scope_; owner != nullptr && (parent_ == nullptr || owner != &parent_->scope_); owner = owner->previous_) --owner->live_batches_;
}
int32_t NativeExecutionBatch::settle(std::array<uint64_t, 4> actual) noexcept {
  const std::lock_guard lock(*scope_.ledger_mutex_);
  const auto host = actual;
  const auto valid = validate_actual_locked(actual);
  if (valid != PHX_MC_OK) return valid;
  settle_locked(actual, host);
  return PHX_MC_OK;
}
int32_t NativeExecutionBatch::usage(phx_mc_execution_usage** out) noexcept {
  const std::lock_guard lock(*scope_.ledger_mutex_);
  if (out == nullptr || target_active_) return PHX_MC_INVALID_ARGUMENT;
  has_usage_ = true;
  *out = &storage_->usage;
  return PHX_MC_OK;
}
int32_t NativeExecutionBatch::refuse(std::array<uint64_t, 4> actual,
                                  std::array<uint64_t, 4> requested, uint64_t* out) noexcept {
  const std::lock_guard lock(*scope_.ledger_mutex_);
  const auto host = actual;
  const auto valid = validate_actual_locked(actual);
  if (valid != PHX_MC_OK) return valid;
  std::array<uint64_t, 4> after;
  if (parent_ != nullptr) {
    after = parent_->remaining_locked();
    after[0] += upper_[0] - actual[0]; after[1] += upper_[1] - actual[1];
    after[2] += upper_[2]; after[3] += upper_[3] - storage_->inclusive_live;
    for (auto* owner = &scope_; owner != nullptr && owner != &parent_->scope_; owner = owner->previous_) {
      after[0] = std::min(after[0], available(owner->work_limit_, owner->work_, owner->reserved_work_) + upper_[0] - actual[0]);
      after[1] = std::min(after[1], available(owner->query_limit_, owner->queries_, owner->reserved_queries_) + upper_[1] - actual[1]);
      after[2] = std::min(after[2], available(owner->cavity_limit_, owner->reserved_cavity_) + upper_[2]);
      const auto memory = owner->memory_.evidence();
      after[3] = std::min(after[3], available(memory[0], memory[1], owner->reserved_bytes_ - upper_[3]));
    }
  } else {
    after.fill(UINT64_MAX);
    for (auto* owner = &scope_; owner != nullptr; owner = owner->previous_) {
      after[0] = std::min(after[0], available(owner->work_limit_, owner->work_, owner->reserved_work_) + upper_[0] - actual[0]);
      after[1] = std::min(after[1], available(owner->query_limit_, owner->queries_, owner->reserved_queries_) + upper_[1] - actual[1]);
      after[2] = std::min(after[2], available(owner->cavity_limit_, owner->reserved_cavity_) + upper_[2]);
    }
    const auto& pool = scope_.memory_.owner();
    const std::lock_guard memory(pool->mutex_);
    after[3] = pool->effective_limit_locked() - pool->live_ - pool->external_upper_ - pool->batch_upper_ + upper_[3] - storage_->inclusive_live;
    for (auto* observer = pool->observation_; observer != nullptr; observer = observer->previous) {
      if (!observer->aggregate && observer->thread != std::this_thread::get_id()) continue;
      const auto external = pool->external_upper_ > observer->external_baseline ? pool->external_upper_ - observer->external_baseline : 0;
      after[3] = std::min(after[3], available(observer->limit, observer->live, external));
    }
  }
  std::size_t resource = 4;
  for (std::size_t r = 0; r < 4; ++r) if (requested[r] > after[r]) { resource = r; break; }
  const bool timed_out = !scope_.checkpoint_locked(NativeExecutionScope::Clock::now()) && scope_.status_locked() == PHX_MC_TIMEOUT;
  if (resource == 4 && !timed_out) return PHX_MC_INTERNAL_ERROR;
  settle_locked(actual, host);
  // Capture precisely the admission snapshot after settlement, under the same
  // lock, before recording refusal. An earlier snapshot remains immutable.
  capture_refusal_locked(scope_, parent_, requested);
  if (!timed_out && scope_.status_locked() == PHX_MC_OK) scope_.refuse_locked(resource);
  const auto& state = scope_.batch_refusal_;
  CounterPublisher publisher{state, out};
  return timed_out ? PHX_MC_TIMEOUT : PHX_MC_CAPACITY_EXCEEDED;
}
int32_t NativeExecutionBatch::checkpoint() noexcept {
  return scope_.checkpoint() ? PHX_MC_OK : scope_.status();
}

NativeExecutionTarget::NativeExecutionTarget(NativeExecutionBatch& ticket)
    : ticket_(ticket), previous_target_(active_execution_target), previous_scope_(active_execution_scope),
      thread_(std::this_thread::get_id()) {
  const std::lock_guard lock(*ticket_.scope_.ledger_mutex_);
  if (ticket_.target_active_) throw std::invalid_argument("native.execution.target.ticket must have no live target; got an active target");
  if (ticket_.parent_ == nullptr && ticket_.scope_.thread_ != thread_) {
    throw std::invalid_argument("native.execution.target.ticket must be an explicit child on a worker; got a root ticket");
  }
  if (!ticket_.scope_.checkpoint_locked(NativeExecutionScope::Clock::now())) {
    if (!ticket_.scope_.has_batch_refusal_) {
      ticket_.scope_.batch_refusal_ = ticket_.snapshot_locked();
      ticket_.scope_.has_batch_refusal_ = true;
    }
    throw ExecutionRefusal{ticket_.scope_.status_locked()};
  }
  ticket_.target_active_ = true;
  ticket_.has_usage_ = true;
  ++ticket_.scope_.live_targets_;
  previous_memory_ = std::move(active_memory_owner);
  active_execution_scope = &ticket_.scope_;
  active_memory_owner = ticket_.scope_.memory_.owner();
  active_execution_target = this;
}
NativeExecutionTarget::~NativeExecutionTarget() {
  if (!closed_ && close() == PHX_MC_INVALID_ARGUMENT) std::terminate();
}
int32_t NativeExecutionTarget::close() noexcept {
  if (closed_ || thread_ != std::this_thread::get_id() || active_execution_target != this ||
      active_execution_scope != &ticket_.scope_) return PHX_MC_INVALID_ARGUMENT;
  const std::lock_guard lock(*ticket_.scope_.ledger_mutex_);
  ticket_.scope_.checkpoint_locked(NativeExecutionScope::Clock::now());
  admit_locked(0, 0);
  store_shared(ticket_.storage_->usage.native_cavity, 0);
  ticket_.target_active_ = false;
  --ticket_.scope_.live_targets_;
  active_execution_scope = previous_scope_;
  active_memory_owner = std::move(previous_memory_);
  active_execution_target = previous_target_;
  closed_ = true;
  return ticket_.scope_.status_locked();
}
std::array<uint64_t, 4> NativeExecutionTarget::remaining_locked() const noexcept {
  return ticket_.remaining_locked();
}
bool NativeExecutionTarget::admit_locked(uint64_t work, uint64_t queries) noexcept {
  auto& u = ticket_.storage_->usage;
  peak_shared(u.host_peak_cavity, load_shared(u.host_cavity));
  peak_shared(u.host_peak_bytes, load_shared(u.host_bytes));
  const auto cavity = load_shared(u.host_cavity) + load_shared(u.native_cavity);
  const auto bytes = load_shared(u.host_bytes) + load_shared(u.native_bytes) +
      load_shared(u.retained_host_bytes) + load_shared(u.children_live_bytes);
  if (cavity != 0) peak_shared(u.combined_peak_cavity,
      cavity + load_shared(u.reserved_cavity));
  if (bytes != 0) peak_shared(u.combined_peak_bytes,
      bytes + load_shared(u.reserved_bytes));
  const auto remaining = remaining_locked();
  if (work > remaining[0]) { refuse_locked(0, {work, queries, 0, 0}); return false; }
  if (queries > remaining[1]) { refuse_locked(1, {work, queries, 0, 0}); return false; }
  if (load_shared(u.host_work) > ticket_.upper_[0] || load_shared(u.host_queries) > ticket_.upper_[1] ||
      load_shared(u.host_cavity) > ticket_.upper_[2] || load_shared(u.host_bytes) > ticket_.upper_[3]) {
    refuse_locked(3); return false;
  }
  return true;
}
void NativeExecutionTarget::commit_locked(uint64_t work, uint64_t queries, bool external) noexcept {
  auto& u = ticket_.storage_->usage;
  if (external) {
    add_shared(u.imported_work, work); add_shared(u.imported_queries, queries);
    ticket_.external_consumed_[0] += work; ticket_.external_consumed_[1] += queries;
  } else {
    add_shared(u.native_work, work); add_shared(u.native_queries, queries);
  }
  for (auto* scope = active_execution_scope; scope != nullptr && scope != &ticket_.scope_; scope = scope->previous_) {
    scope->work_ += work; scope->queries_ += queries;
    if (external) { scope->external_work_ += work; scope->external_queries_ += queries; }
  }
}
bool NativeExecutionTarget::cavity_locked(uint64_t cells) noexcept {
  auto& u = ticket_.storage_->usage;
  peak_shared(u.host_peak_cavity, load_shared(u.host_cavity));
  if (cells > remaining_locked()[2] + load_shared(u.native_cavity)) {
    refuse_locked(2, {0, 0, cells, 0}); return false;
  }
  store_shared(u.native_cavity, cells);
  peak_shared(u.native_peak_cavity, cells);
  if (load_shared(u.host_cavity) + cells != 0) {
    peak_shared(u.combined_peak_cavity,
        load_shared(u.host_cavity) + cells + load_shared(u.reserved_cavity));
  }
  return true;
}
void NativeExecutionTarget::refuse_locked(std::size_t resource, std::array<uint64_t, 4> requested) noexcept {
  auto* scope = active_execution_scope == nullptr ? &ticket_.scope_ : active_execution_scope;
  if (!scope->has_batch_refusal_) {
    auto snapshot = ticket_.snapshot_locked();
    const bool cavity_replacement = resource == 2 && requested[2] != 0;
    if (cavity_replacement) {
      snapshot.remaining_cavity += load_shared(ticket_.storage_->usage.native_cavity);
    }
    if (scope != &ticket_.scope_) {
      const auto local = scope->remaining_locked();
      snapshot.remaining_work = std::min(snapshot.remaining_work, local[0]);
      snapshot.remaining_queries = std::min(snapshot.remaining_queries, local[1]);
      if (!cavity_replacement) {
        snapshot.remaining_cavity = std::min(snapshot.remaining_cavity, local[2]);
      }
      snapshot.remaining_bytes = std::min(snapshot.remaining_bytes, local[3]);
      if (cavity_replacement) {
        for (auto* owner = scope; owner != nullptr && owner != &ticket_.scope_;
             owner = owner->previous_) {
          snapshot.remaining_cavity = std::min(snapshot.remaining_cavity,
              available(owner->cavity_limit_, owner->reserved_cavity_));
        }
      }
    }
    for (auto* owner = scope; owner != nullptr; owner = owner->previous_) {
      if (!owner->has_batch_refusal_) {
        owner->batch_refusal_ = snapshot;
        owner->has_batch_refusal_ = true;
        if (scope->status_locked() == PHX_MC_OK) {
          owner->refusal_request_ = requested;
          owner->has_refusal_request_ = true;
        }
      }
    }
  }
  if (scope->status_locked() == PHX_MC_OK) scope->refuse_locked(resource);
}

namespace {
struct ArenaBlock {
  std::size_t bytes;
  ArenaBlock* next;
};
struct ArenaAllocation {
  void* begin;
  std::size_t bytes;
};

// The aliasing MemoryOwner points at resource_, retaining both this upstream
// and its parent-owned backing. Only caller-side construction/destruction
// accesses the parent resource; chunk allocation and recycling remain local.
class NativeChunkStorage final : public std::pmr::memory_resource,
                                 public NativeAllocatedObject {
 public:
  NativeChunkStorage(std::size_t bytes, MemoryOwner parent)
      : parent_(std::move(parent)), bytes_(bytes), resource_(bytes, this) {
    if (bytes_ != 0) {
      buffer_ = parent_->allocate(bytes_, alignof(std::max_align_t));
      if (bytes_ >= sizeof(ArenaBlock)) {
        free_ = std::construct_at(static_cast<ArenaBlock*>(buffer_),
                                 ArenaBlock{bytes_, nullptr});
      }
    }
  }
  ~NativeChunkStorage() {
    if (resource_.live_bytes() != 0) std::terminate();
    if (buffer_ != nullptr) {
      parent_->deallocate(buffer_, bytes_, alignof(std::max_align_t));
    }
  }
  BoundedMemoryResource& resource() noexcept { return resource_; }
 private:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    for (auto** link = &free_; *link != nullptr; link = &(*link)->next) {
      auto* block = *link;
      const auto begin = reinterpret_cast<std::uintptr_t>(block);
      const auto first = begin + sizeof(ArenaAllocation);
      if (alignment - 1 > UINTPTR_MAX - first) continue;
      const auto aligned = (first + alignment - 1) & ~(alignment - 1);
      const auto prefix = static_cast<std::size_t>(aligned - begin);
      if (prefix > block->bytes || bytes > block->bytes - prefix) continue;
      const auto used = prefix + bytes;
      const auto padding = (alignof(ArenaBlock) - used % alignof(ArenaBlock)) %
                           alignof(ArenaBlock);
      if (padding > block->bytes - used) continue;
      auto total = used + padding;
      const auto remaining = block->bytes - total;
      auto* next = block->next;
      if (remaining >= sizeof(ArenaBlock)) {
        auto* split = reinterpret_cast<ArenaBlock*>(begin + total);
        *link = std::construct_at(split, ArenaBlock{remaining, next});
      } else {
        total = block->bytes;
        *link = next;
      }
      auto* header = reinterpret_cast<ArenaAllocation*>(aligned - sizeof(ArenaAllocation));
      std::construct_at(header, ArenaAllocation{block, total});
      return reinterpret_cast<void*>(aligned);
    }
    throw std::bad_alloc();
  }
  void do_deallocate(void* pointer, std::size_t, std::size_t) override {
    auto* header = reinterpret_cast<ArenaAllocation*>(
        static_cast<std::byte*>(pointer) - sizeof(ArenaAllocation));
    auto* begin = static_cast<ArenaBlock*>(header->begin);
    const auto bytes = header->bytes;
    std::destroy_at(header);
    auto** link = &free_;
    ArenaBlock* previous = nullptr;
    while (*link != nullptr &&
           reinterpret_cast<std::uintptr_t>(*link) < reinterpret_cast<std::uintptr_t>(begin)) {
      previous = *link;
      link = &(*link)->next;
    }
    auto* block = std::construct_at(begin, ArenaBlock{bytes, *link});
    *link = block;
    if (block->next != nullptr &&
        reinterpret_cast<std::byte*>(block) + block->bytes ==
            reinterpret_cast<std::byte*>(block->next)) {
      block->bytes += block->next->bytes;
      block->next = block->next->next;
    }
    if (previous != nullptr &&
        reinterpret_cast<std::byte*>(previous) + previous->bytes ==
            reinterpret_cast<std::byte*>(block)) {
      previous->bytes += block->bytes;
      previous->next = block->next;
    }
  }
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }
  MemoryOwner parent_;
  std::size_t bytes_;
  BoundedMemoryResource resource_;
  void* buffer_ = nullptr;
  ArenaBlock* free_ = nullptr;
};

bool add_bound(uint64_t& total, uint64_t value) noexcept {
  if (value > UINT64_MAX - total) return false;
  total += value;
  return true;
}
bool parallel_upper(std::span<const NativeChunkAllowance> allowances,
                    std::array<uint64_t, 4>& upper) noexcept {
  for (const auto& allowance : allowances) {
    if (allowance.bytes > SIZE_MAX || !add_bound(upper[0], allowance.work) ||
        !add_bound(upper[1], allowance.queries) ||
        !add_bound(upper[2], allowance.cavity) ||
        !add_bound(upper[3], allowance.bytes) ||
        !add_bound(upper[3], sizeof(NativeChunkStorage))) return false;
  }
  return upper[3] <= SIZE_MAX;
}
MemoryOwner make_chunk_owner(std::size_t bytes, const MemoryOwner& parent) {
  auto storage = make_native_unique<NativeChunkStorage>(bytes, parent);
  // Weak arena observers retain the control block, not the parent allocator.
  // Consume the deleter's parent owner when destroying the last strong arena
  // reference so an expired arena cannot keep its backing pool alive.
  std::shared_ptr<NativeChunkStorage> shared(
      storage.release(), [owner = parent](NativeChunkStorage* pointer) mutable noexcept {
        NativeObjectDeleter<NativeChunkStorage>{std::move(owner)}(pointer);
      });
  return MemoryOwner(shared, &shared->resource());
}
}

uint64_t native_parallel_allocation_bytes(uint64_t payload, std::size_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    throw std::invalid_argument("native.parallel.allocation.alignment must be a power of two; got an invalid alignment");
  }
  const auto upstream_alignment = std::max(alignment, alignof(AllocationHeader));
  uint64_t result = payload;
  // Native prefix rounds up to upstream alignment; the arena independently
  // aligns that upstream request and rounds its reusable block tail.
  for (const auto amount : {sizeof(AllocationHeader), upstream_alignment - 1,
                            sizeof(ArenaAllocation), upstream_alignment - 1,
                            sizeof(ArenaBlock) - 1}) {
    if (!add_bound(result, amount)) {
      throw std::overflow_error("native.parallel.allocation.bytes must fit uint64_t; got an overflowing bound");
    }
  }
  return result;
}

NativeParallelBatch::NativeParallelBatch(std::span<const NativeChunkAllowance> allowances)
    : scope_(active_execution_scope) {
  if (scope_ == nullptr) {
    implicit_scope_ = std::make_unique<NativeExecutionScope>(
        UINT64_MAX, UINT64_MAX, UINT64_MAX, SIZE_MAX, INFINITY);
    scope_ = implicit_scope_.get();
  }
  std::array<uint64_t, 4> upper{};
  if (!parallel_upper(allowances, upper)) {
    fail_internal();
    return;
  }
  auto* target = active_execution_target != nullptr
      ? active_execution_target : scope_->borrowed_target_;
  void* admitted = nullptr;
  status_ = NativeExecutionBatch::admit(
      *scope_, target == nullptr ? nullptr : &target->ticket_, upper, &admitted);
  if (status_ != PHX_MC_OK) return;
  ticket_.reset(static_cast<NativeExecutionBatch*>(admitted));
  try {
    allowances_.assign(allowances.begin(), allowances.end());
    owners_.reserve(allowances.size());
    NativeExecutionTarget allocation(*ticket_);
    const auto parent = scratch_memory_owner();
    for (const auto& allowance : allowances_) {
      owners_.push_back(make_chunk_owner(static_cast<std::size_t>(allowance.bytes), parent));
    }
  } catch (const std::bad_alloc&) {
    scope_->refuse_external_storage(upper[3]);
    status_ = PHX_MC_CAPACITY_EXCEEDED;
    release();
  } catch (...) {
    fail_internal();
    release();
    throw;
  }
}

NativeParallelBatch::~NativeParallelBatch() { release(); }
void NativeParallelBatch::fail_internal() noexcept {
  status_ = PHX_MC_INTERNAL_ERROR;
  const std::lock_guard lock(*scope_->ledger_mutex_);
  for (auto* owner = scope_; owner != nullptr; owner = owner->previous_) {
    owner->status_ = PHX_MC_INTERNAL_ERROR;
  }
}
void NativeParallelBatch::release() noexcept {
  if (thread_ != std::this_thread::get_id()) std::terminate();
  owners_.clear();
  if (ticket_ != nullptr) {
    // Worker work must be supplied through settle; allocation usage already
    // resides in the same ticket even during staging failures.
    if (ticket_->settle({0, 0, 0, 0}) != PHX_MC_OK) std::terminate();
    ticket_.reset();
  }
}
double NativeParallelBatch::remaining_wall_seconds() const noexcept {
  return scope_->remaining_wall_seconds();
}

namespace {
bool valid_chunk(const NativeChunkEvidence& evidence,
                 const NativeChunkAllowance& allowance) noexcept {
  const auto& counters = evidence.counters;
  return counters.construction_work <= allowance.work &&
      counters.source_queries <= allowance.queries &&
      counters.peak_cavity <= allowance.cavity &&
      counters.external_work <= counters.construction_work &&
      counters.external_queries <= counters.source_queries &&
      evidence.memory.live_bytes <= evidence.memory.peak_bytes &&
      evidence.memory.peak_bytes <= allowance.bytes;
}
int32_t chunk_status(const NativeChunkEvidence& evidence) noexcept {
  if (evidence.counters.work_refusals != 0 || evidence.counters.query_refusals != 0 ||
      evidence.counters.cavity_refusals != 0 || evidence.memory.refusals != 0 ||
      evidence.status == PHX_MC_CAPACITY_EXCEEDED ||
      (!evidence.completed && evidence.status != PHX_MC_TIMEOUT)) return PHX_MC_INTERNAL_ERROR;
  return evidence.status;
}
MemoryOwner detached_owner(NativeChunkAllowance allowance, MemoryOwner owner) {
  if (owner == nullptr) {
    throw std::invalid_argument("native.parallel.chunk.memory_owner must be present; got no owner");
  }
  if (allowance.bytes > SIZE_MAX) {
    throw std::overflow_error("native.parallel.chunk.bytes must fit size_t; got an overflowing bound");
  }
  return owner;
}
}

int32_t NativeParallelBatch::settle(std::span<const NativeChunkEvidence> evidence) noexcept {
  if (thread_ != std::this_thread::get_id()) return PHX_MC_INVALID_ARGUMENT;
  if (ticket_ == nullptr) return status_ == PHX_MC_OK ? PHX_MC_INVALID_ARGUMENT : status_;
  ExecutionCounters total;
  int32_t result = PHX_MC_OK;
  if (evidence.size() != allowances_.size()) result = PHX_MC_INTERNAL_ERROR;
  const auto count = std::min(evidence.size(), allowances_.size());
  for (std::size_t index = 0; index < count; ++index) {
    const auto& child = evidence[index];
    if (!valid_chunk(child, allowances_[index])) {
      result = PHX_MC_INTERNAL_ERROR;
      continue;
    }
    const auto status = chunk_status(child);
    if (status == PHX_MC_INTERNAL_ERROR ||
        (status == PHX_MC_TIMEOUT && result != PHX_MC_INTERNAL_ERROR) ||
        result == PHX_MC_OK) result = status;
    // Admission proves these work/query/cavity sums fit. Primitive telemetry
    // has no allowance, so its overflow is checked separately.
    total.construction_work += child.counters.construction_work;
    total.source_queries += child.counters.source_queries;
    total.peak_cavity += child.counters.peak_cavity;
    total.external_work += child.counters.external_work;
    total.external_queries += child.counters.external_queries;
    if (!add_bound(total.primitive_queries, child.counters.primitive_queries)) {
      total.primitive_queries = UINT64_MAX;
      result = PHX_MC_INTERNAL_ERROR;
    }
  }
  {
    const std::lock_guard lock(*scope_->ledger_mutex_);
    auto& usage = ticket_->storage_->usage;
    store_shared(usage.native_work, total.construction_work - total.external_work);
    store_shared(usage.native_queries, total.source_queries - total.external_queries);
    store_shared(usage.imported_work, total.external_work);
    store_shared(usage.imported_queries, total.external_queries);
    ticket_->external_consumed_ = {total.external_work, total.external_queries};
    for (auto* owner = scope_; owner != nullptr; owner = owner->previous_) {
      if (!add_bound(owner->primitive_queries_, total.primitive_queries)) {
        owner->primitive_queries_ = UINT64_MAX;
        result = PHX_MC_INTERNAL_ERROR;
      }
    }
    for (auto* owner = scope_; owner != nullptr; owner = owner->previous_) {
      if (result == PHX_MC_INTERNAL_ERROR) owner->status_ = PHX_MC_INTERNAL_ERROR;
      else if (result == PHX_MC_TIMEOUT && owner->status_ == PHX_MC_OK) owner->status_ = PHX_MC_TIMEOUT;
    }
  }
  owners_.clear();
  const auto settled = ticket_->settle({0, 0, total.peak_cavity, 0});
  if (settled != PHX_MC_OK) std::terminate();
  ticket_.reset();
  status_ = result;
  return status_;
}

NativeDetachedExecution::Affinity::Affinity() noexcept
    : scope(active_execution_scope), target(active_execution_target),
      memory(std::move(active_memory_owner)) {
  active_execution_scope = nullptr;
  active_execution_target = nullptr;
}
NativeDetachedExecution::Affinity::~Affinity() {
  active_execution_scope = scope;
  active_execution_target = target;
  active_memory_owner = std::move(memory);
}
NativeDetachedExecution::NativeDetachedExecution(NativeChunkAllowance allowance, MemoryOwner owner)
    : execution_(allowance.work, allowance.queries, allowance.cavity,
                 static_cast<std::size_t>(allowance.bytes), INFINITY,
                 detached_owner(allowance, std::move(owner))) {}
NativeChunkEvidence NativeDetachedExecution::evidence(int32_t status, bool completed) const noexcept {
  const auto counters = execution_.evidence();
  const auto memory = execution_.memory_evidence();
  NativeChunkEvidence result{ExecutionCounters::from_slots(counters.data()),
      ManagedMemory::from_slots(memory.data()), status, completed};
  const auto local_status = execution_.status();
  if (local_status != PHX_MC_OK) result.status = local_status;
  result.status = chunk_status(result);
  return result;
}

class NativeHostStorage final : public NativeAllocatedObject {
 public:
  NativeHostStorage()
      : storage_(active_execution_target == nullptr ? nullptr : active_execution_target->ticket_.storage_) {}
  ~NativeHostStorage() { native_object_owner()->release_external_storage(bound_, storage_); }
  NativeHostStorage(const NativeHostStorage&) = delete;
  NativeHostStorage& operator=(const NativeHostStorage&) = delete;
  bool resize(std::size_t requested) noexcept {
    if (!native_object_owner()->resize_external_storage(bound_, requested, storage_)) return false;
    bound_ = requested;
    return true;
  }
  const MemoryOwner& owner() const noexcept { return native_object_owner(); }
  std::size_t bound() const noexcept { return bound_; }
 private:
  std::size_t bound_ = 0;
  std::shared_ptr<NativeBatchStorage> storage_;
};
}  // namespace phx::mc

extern "C" int32_t phx_mc_execution_begin(
    uint64_t work, uint64_t queries, uint64_t cavity, uint64_t scratch,
    double seconds, phx_mc_tet_mesh* memory_mesh, void** out) {
  if (out == nullptr) return PHX_MC_INVALID_ARGUMENT;
  *out = nullptr;
  using Clock = std::chrono::steady_clock;
  const double available = std::chrono::duration<double>(Clock::time_point::max() - Clock::now()).count();
  if (scratch > std::numeric_limits<std::size_t>::max() || std::isnan(seconds) || seconds < 0.0 ||
      (std::isfinite(seconds) && seconds >= available)) return PHX_MC_INVALID_ARGUMENT;
  if (memory_mesh != nullptr && memory_mesh->mesh == nullptr) return PHX_MC_INVALID_ARGUMENT;
  if (memory_mesh != nullptr && phx::mc::active_execution_scope != nullptr &&
      memory_mesh->mesh->memory_owner() != phx::mc::scratch_memory_owner()) return PHX_MC_INVALID_ARGUMENT;
  try {
    *out = new phx::mc::NativeExecutionScope(work, queries, cavity, static_cast<std::size_t>(scratch), seconds,
        memory_mesh == nullptr ? phx::mc::scratch_memory_owner() : memory_mesh->mesh->memory_owner());
    return PHX_MC_OK;
  } catch (const std::bad_alloc&) { return PHX_MC_CAPACITY_EXCEEDED; }
  catch (...) { return PHX_MC_INTERNAL_ERROR; }
}
extern "C" int32_t phx_mc_execution_end(void* opaque, uint64_t* counters, uint64_t* memory, double* seconds) {
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  if (scope == nullptr || scope != phx::mc::active_execution_scope || counters == nullptr ||
      memory == nullptr || seconds == nullptr || scope->has_live_batches()) return PHX_MC_INVALID_ARGUMENT;
  scope->checkpoint();
  const auto work = scope->evidence();
  const auto bytes = scope->memory_evidence();
  const auto work_state = phx::mc::ExecutionCounters::from_slots(work.data());
  
  const auto memory_state = phx::mc::ExecutionMemory::from_slots(bytes.data());
  phx::mc::CounterPublisher memory_publisher{memory_state, memory};
  phx::mc::CounterPublisher work_publisher{work_state, counters};
  *seconds = scope->seconds();
  const int32_t status = scope->status() == PHX_MC_OK && bytes[5] != 0 ? PHX_MC_CAPACITY_EXCEEDED : scope->status();
  delete scope;
  return status;
}
extern "C" int32_t phx_mc_execution_allocator_bookkeeping(void* opaque, uint64_t* live_bytes) {
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  if (scope == nullptr || live_bytes == nullptr) return PHX_MC_INVALID_ARGUMENT;
  *live_bytes = scope->allocator_bookkeeping();
  return PHX_MC_OK;
}
extern "C" int32_t phx_mc_execution_charge(void* opaque, uint64_t work, uint64_t queries) {
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  if (scope == nullptr || scope != phx::mc::active_execution_scope) return PHX_MC_INVALID_ARGUMENT;
  return scope->charge_external(work, queries) ? PHX_MC_OK : scope->status();
}
extern "C" int32_t phx_mc_execution_batch_admit(void* opaque, void* parent_ticket, uint64_t work,
    uint64_t queries, uint64_t cavity, uint64_t bytes, void** out_ticket) {
  if (out_ticket == nullptr) return PHX_MC_INVALID_ARGUMENT;
  *out_ticket = nullptr;
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  if (scope == nullptr || bytes > std::numeric_limits<std::size_t>::max()) return PHX_MC_INVALID_ARGUMENT;
  try {
    return phx::mc::NativeExecutionBatch::admit(*scope, static_cast<phx::mc::NativeExecutionBatch*>(parent_ticket),
                                               {work, queries, cavity, bytes}, out_ticket);
  } catch (const std::bad_alloc&) { scope->refuse_external_storage(); return PHX_MC_CAPACITY_EXCEEDED; }
  catch (...) { return PHX_MC_INTERNAL_ERROR; }
}
extern "C" int32_t phx_mc_execution_batch_settle(void* opaque, uint64_t work, uint64_t queries,
                                                 uint64_t cavity, uint64_t bytes) {
  auto* ticket = static_cast<phx::mc::NativeExecutionBatch*>(opaque);
  if (ticket == nullptr) return PHX_MC_INVALID_ARGUMENT;
  const int32_t status = ticket->settle({work, queries, cavity, bytes});
  if (status == PHX_MC_OK) delete ticket;
  return status;
}
extern "C" int32_t phx_mc_execution_batch_refuse(void* opaque, const uint64_t* actual,
                                                 const uint64_t* requested, uint64_t* out) {
  auto* ticket = static_cast<phx::mc::NativeExecutionBatch*>(opaque);
  if (ticket == nullptr || actual == nullptr || requested == nullptr || out == nullptr ||
      requested[3] > std::numeric_limits<std::size_t>::max()) return PHX_MC_INVALID_ARGUMENT;
  const auto status = ticket->refuse({actual[0], actual[1], actual[2], actual[3]},
                                    {requested[0], requested[1], requested[2], requested[3]}, out);
  if (status == PHX_MC_CAPACITY_EXCEEDED || status == PHX_MC_TIMEOUT) delete ticket;
  return status;
}
extern "C" int32_t phx_mc_execution_batch_usage(
    void* opaque, phx_mc_execution_usage** out) {
  if (out != nullptr) *out = nullptr;
  auto* ticket = static_cast<phx::mc::NativeExecutionBatch*>(opaque);
  return ticket == nullptr || out == nullptr ? PHX_MC_INVALID_ARGUMENT : ticket->usage(out);
}
extern "C" int32_t phx_mc_execution_target_begin(void* opaque, void** out,
                                                  phx_mc_execution_usage** usage) {
  if (out != nullptr) *out = nullptr;
  if (usage != nullptr) *usage = nullptr;
  auto* ticket = static_cast<phx::mc::NativeExecutionBatch*>(opaque);
  if (ticket == nullptr || out == nullptr || usage == nullptr ||
      static_cast<void*>(out) == static_cast<void*>(usage)) return PHX_MC_INVALID_ARGUMENT;
  try {
    auto target = std::make_unique<phx::mc::NativeExecutionTarget>(*ticket);
    *usage = target->usage(); *out = target.release(); return PHX_MC_OK;
  } catch (const phx::mc::ExecutionRefusal& refusal) { return refusal.status; }
  catch (const std::invalid_argument&) { return PHX_MC_INVALID_ARGUMENT; }
  catch (const std::bad_alloc&) { return PHX_MC_CAPACITY_EXCEEDED; }
  catch (...) { return PHX_MC_INTERNAL_ERROR; }
}
extern "C" int32_t phx_mc_execution_target_end(void* opaque) {
  auto* target = static_cast<phx::mc::NativeExecutionTarget*>(opaque);
  if (target == nullptr) return PHX_MC_INVALID_ARGUMENT;
  const auto status = target->close();
  if (status != PHX_MC_INVALID_ARGUMENT) delete target;
  return status;
}
extern "C" int32_t phx_mc_execution_batch_checkpoint(void* opaque) {
  auto* ticket = static_cast<phx::mc::NativeExecutionBatch*>(opaque);
  return ticket == nullptr ? PHX_MC_INVALID_ARGUMENT : ticket->checkpoint();
}
extern "C" int32_t phx_mc_execution_batch_refusal(void* opaque, uint64_t* values) {
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  if (scope == nullptr || values == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return scope->batch_refusal(values);
}
extern "C" int32_t phx_mc_execution_refusal_request(void* opaque, uint64_t* values) {
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  if (scope == nullptr || values == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return scope->refusal_request(values);
}
extern "C" int32_t phx_mc_execution_import_preparation(void* opaque, uint64_t work,
                                                       uint64_t queries, double seconds) {
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  const double duration_max = std::chrono::duration<double>(std::chrono::steady_clock::duration::max()).count();
  if (scope == nullptr || scope != phx::mc::active_execution_scope || !std::isfinite(seconds) ||
      seconds < 0.0 || seconds >= duration_max) return PHX_MC_INVALID_ARGUMENT;
  return scope->import_preparation(work, queries, seconds) ? PHX_MC_OK :
      (scope->status() == PHX_MC_OK ? PHX_MC_INVALID_ARGUMENT : scope->status());
}
extern "C" int32_t phx_mc_execution_prior_seconds(void* opaque, double* seconds) {
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  if (scope == nullptr || scope != phx::mc::active_execution_scope || seconds == nullptr) return PHX_MC_INVALID_ARGUMENT;
  *seconds = scope->prior_seconds(); return PHX_MC_OK;
}
extern "C" int32_t phx_mc_execution_admit_work_bound(void* opaque, uint64_t maximum_work) {
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  if (scope == nullptr || scope != phx::mc::active_execution_scope) return PHX_MC_INVALID_ARGUMENT;
  return scope->admit(maximum_work) ? PHX_MC_OK : scope->status();
}
extern "C" int32_t phx_mc_execution_admit_cavity(void* opaque, uint64_t cells) {
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  if (scope == nullptr || scope != phx::mc::active_execution_scope || cells > std::numeric_limits<std::size_t>::max()) return PHX_MC_INVALID_ARGUMENT;
  return scope->admit_cavity(static_cast<std::size_t>(cells)) ? PHX_MC_OK : scope->status();
}
extern "C" int32_t phx_mc_execution_remaining(void* opaque, uint64_t* remaining, double* wall_seconds) {
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  if (scope == nullptr || scope != phx::mc::active_execution_scope || remaining == nullptr || wall_seconds == nullptr) return PHX_MC_INVALID_ARGUMENT;
  scope->checkpoint(); const auto allowance = scope->remaining();
  const auto state = phx::mc::ExecutionResources::from_slots(allowance.data());
  phx::mc::CounterPublisher publisher{state, remaining};
  *wall_seconds = scope->remaining_wall_seconds(); return scope->status();
}
extern "C" int32_t phx_mc_execution_allocate_host_array(void* opaque, uint64_t bytes, uint64_t alignment,
                                                       void** out_owner, void** out_data) {
  if (out_owner != nullptr) *out_owner = nullptr;
  if (out_data != nullptr) *out_data = nullptr;
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  if (scope == nullptr || scope != phx::mc::active_execution_scope || out_owner == nullptr ||
      out_data == nullptr || out_owner == out_data || bytes > static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
      alignment == 0 || alignment > std::numeric_limits<std::size_t>::max() || (alignment & (alignment - 1)) != 0) return PHX_MC_INVALID_ARGUMENT;
  if (!scope->charge(0)) return scope->status();
  try {
    auto owner = phx::mc::make_native_unique<phx::mc::NativeHostBuffer>(static_cast<std::size_t>(bytes), static_cast<std::size_t>(alignment));
    *out_data = owner->data(); *out_owner = owner.release(); return PHX_MC_OK;
  } catch (const std::bad_alloc&) { return phx::mc::native_execution_status(PHX_MC_CAPACITY_EXCEEDED); }
  catch (...) { return PHX_MC_INTERNAL_ERROR; }
}
extern "C" void phx_mc_execution_free_host_array(void* owner) {
  phx::mc::destroy_native_object(static_cast<phx::mc::NativeHostBuffer*>(owner));
}
extern "C" int32_t phx_mc_execution_reserve_host_storage(void* opaque, uint64_t bytes_upper, void** reservation) {
  auto* scope = static_cast<phx::mc::NativeExecutionScope*>(opaque);
  if (scope == nullptr || scope != phx::mc::active_execution_scope || reservation == nullptr ||
      bytes_upper > std::numeric_limits<std::size_t>::max()) return PHX_MC_INVALID_ARGUMENT;
  auto* existing = static_cast<phx::mc::NativeHostStorage*>(*reservation);
  if (existing != nullptr && existing->owner() != phx::mc::scratch_memory_owner()) return PHX_MC_INVALID_ARGUMENT;
  if (existing != nullptr && bytes_upper <= existing->bound()) {
    existing->resize(static_cast<std::size_t>(bytes_upper)); return PHX_MC_OK;
  }
  if (!scope->charge(0)) return scope->status();
  try {
    if (existing != nullptr) {
      if (existing->resize(static_cast<std::size_t>(bytes_upper))) return PHX_MC_OK;
    } else {
      auto owner = phx::mc::make_native_unique<phx::mc::NativeHostStorage>();
      if (owner->resize(static_cast<std::size_t>(bytes_upper))) {
        *reservation = owner.release(); return PHX_MC_OK;
      }
    }
    scope->refuse_external_storage(bytes_upper); return PHX_MC_CAPACITY_EXCEEDED;
  } catch (const std::bad_alloc&) { scope->refuse_external_storage(bytes_upper); return PHX_MC_CAPACITY_EXCEEDED; }
  catch (...) { return PHX_MC_INTERNAL_ERROR; }
}
extern "C" void phx_mc_execution_release_host_storage(void* reservation) {
  phx::mc::destroy_native_object(static_cast<phx::mc::NativeHostStorage*>(reservation));
}
