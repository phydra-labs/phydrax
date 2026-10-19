//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

#include "bounded_memory.hpp"
#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {
uint64_t mapped_counter(uint64_t& value) {
  std::atomic_ref<uint64_t> counter(value);
  PHX_CHECK(counter.is_lock_free());
  return counter.load(std::memory_order_acquire);
}
using Counts = std::array<uint64_t, 4>;
using Refusal = std::array<uint64_t, PHX_MC_EXECUTION_BATCH_REFUSAL_VALUES>;

struct Receipt {
  std::array<uint64_t, PHX_MC_EXECUTION_COUNTERS> work{};
  std::array<uint64_t, PHX_MC_EXECUTION_MEMORY_VALUES> memory{};
};

class Execution final {
 public:
  explicit Execution(Counts limits) {
    PHX_CHECK(phx_mc_execution_begin(limits[0], limits[1], limits[2], limits[3],
                                     INFINITY, nullptr, &scope) == PHX_MC_OK);
  }
  ~Execution() {
    if (scope != nullptr) finish(expected_status);
  }
  Execution(const Execution&) = delete;
  Execution& operator=(const Execution&) = delete;

  Counts remaining(int32_t expected = PHX_MC_OK) const {
    Counts result{};
    double seconds = -1;
    PHX_CHECK(phx_mc_execution_remaining(scope, result.data(), &seconds) == expected);
    return result;
  }
  Refusal refusal() const {
    Refusal result{};
    PHX_CHECK(phx_mc_execution_batch_refusal(scope, result.data()) == PHX_MC_OK);
    return result;
  }
  Counts refused_request() const {
    Counts result{};
    PHX_CHECK(phx_mc_execution_refusal_request(scope, result.data()) == PHX_MC_OK);
    return result;
  }
  Receipt finish(int32_t expected = PHX_MC_OK) {
    Receipt result;
    double seconds = -1;
    PHX_CHECK(phx_mc_execution_end(scope, result.work.data(), result.memory.data(),
                                   &seconds) == expected);
    scope = nullptr;
    return result;
  }
  void* scope = nullptr;
  int32_t expected_status = PHX_MC_OK;
};

class Batch final {
 public:
  Batch(void* scope, Counts upper, void* parent = nullptr) {
    PHX_CHECK(phx_mc_execution_batch_admit(scope, parent, upper[0], upper[1],
                                          upper[2], upper[3], &ticket) == PHX_MC_OK);
  }
  ~Batch() {
    if (ticket != nullptr) settle();
  }
  Batch(const Batch&) = delete;
  Batch& operator=(const Batch&) = delete;

  void settle() {
    PHX_CHECK(phx_mc_execution_batch_settle(ticket, actual[0], actual[1],
                                           actual[2], actual[3]) == PHX_MC_OK);
    ticket = nullptr;
  }
  void* ticket = nullptr;
  Counts actual{};
};

class Target final {
 public:
  explicit Target(void* ticket) {
    PHX_CHECK(phx_mc_execution_target_begin(ticket, &target, &usage) == PHX_MC_OK);
    PHX_CHECK(target != nullptr && usage != nullptr);
  }
  ~Target() {
    if (target != nullptr) finish(phx::mc::native_execution_refusal());
  }
  Target(const Target&) = delete;
  Target& operator=(const Target&) = delete;

  void finish(int32_t expected = PHX_MC_OK) {
    PHX_CHECK(phx_mc_execution_target_end(target) == expected);
    target = nullptr;
  }
  void* target = nullptr;
  phx_mc_execution_usage* usage = nullptr;
};

void check_work(const Receipt& receipt, uint64_t work, uint64_t queries,
                uint64_t host_work, uint64_t host_queries) {
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == work && receipt.work[PHX_MC_EXECUTION_SOURCE_QUERIES] == queries);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_EXTERNAL_WORK] == host_work && receipt.work[PHX_MC_EXECUTION_EXTERNAL_QUERIES] == host_queries);
}

void test_reserved_native_charges() {
  Execution execution({8, 6, 5, 0});
  Batch batch(execution.scope, {8, 6, 5, 0});
  PHX_CHECK((execution.remaining() == Counts{0, 0, 0, 0}));
  {
    Target target(batch.ticket);
    PHX_CHECK(phx::mc::native_execution_spend(8, 6));
    PHX_CHECK(phx::mc::native_execution_cavity(5));
    PHX_CHECK(target.usage->native_work == 8 && target.usage->native_queries == 6);
    PHX_CHECK(target.usage->native_peak_cavity == 5);
    PHX_CHECK(target.usage->host_work == 0 && target.usage->host_queries == 0);
  }
  batch.settle();
  PHX_CHECK((execution.remaining() == Counts{0, 0, 5, 0}));
  const Receipt receipt = execution.finish();
  check_work(receipt, 8, 6, 0, 0);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_PEAK_CAVITY] == 5);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_WORK_REFUSALS] == 0 && receipt.work[PHX_MC_EXECUTION_QUERY_REFUSALS] == 0 && receipt.work[PHX_MC_EXECUTION_CAVITY_REFUSALS] == 0);
}

void test_host_native_interleaving() {
  Execution execution({8, 6, 5, 0});
  Batch batch(execution.scope, {8, 6, 5, 0});
  {
    Target target(batch.ticket);
    PHX_CHECK(phx::mc::native_execution_spend(2, 1));
    target.usage->host_work += 3;
    target.usage->host_queries += 2;
    PHX_CHECK(phx::mc::native_execution_spend(3, 3));
    PHX_CHECK(target.usage->native_work == 5 && target.usage->native_queries == 4);
    PHX_CHECK(target.usage->host_work == 3 && target.usage->host_queries == 2);
    batch.actual = {3, 2, 0, 0};
  }
  batch.settle();
  const Receipt receipt = execution.finish();
  check_work(receipt, 8, 6, 3, 2);
}

void native_refusal(std::size_t resource) {
  Execution execution({6, 6, 6, 0});
  Batch batch(execution.scope, {6, 6, 6, 0});
  {
    Target target(batch.ticket);
    PHX_CHECK(phx::mc::native_execution_spend(2, 2));
    target.usage->host_work = 3;
    target.usage->host_queries = 3;
    target.usage->host_cavity = 3;
    target.usage->host_peak_cavity = 3;
    if (resource < 2) {
      PHX_CHECK(!phx::mc::native_execution_spend(resource == 0 ? 2 : 0,
                                               resource == 1 ? 2 : 0));
    } else {
      PHX_CHECK(phx::mc::native_execution_cavity(3));
      PHX_CHECK(!phx::mc::native_execution_cavity(4));
    }
    PHX_CHECK(phx::mc::native_execution_refusal() == PHX_MC_CAPACITY_EXCEEDED);
    PHX_CHECK(target.usage->native_work == 2 && target.usage->native_queries == 2);
    const Refusal refusal = execution.refusal();
    Counts requested{};
    requested[resource] = resource == 2 ? 4 : 2;
    PHX_CHECK(execution.refused_request() == requested);
    PHX_CHECK(refusal[PHX_MC_BATCH_CONSUMED_WORK] == 5);
    PHX_CHECK(refusal[PHX_MC_BATCH_CONSUMED_QUERIES] == 5);
    PHX_CHECK(refusal[resource] == (resource == 2 ? 3 : 1));
    batch.actual = {3, 3, 3, 0};
  }
  batch.settle();
  const Receipt receipt = execution.finish(PHX_MC_CAPACITY_EXCEEDED);
  check_work(receipt, 5, 5, 3, 3);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_WORK_REFUSALS + resource] == 1);
}

void test_native_refusal_evidence() {
  for (std::size_t resource = 0; resource < 3; ++resource) native_refusal(resource);
}

void test_reserved_bytes() {
  Execution execution({0, 0, 0, 32});
  const auto owner = phx::mc::scratch_memory_owner();
  Batch batch(execution.scope, {0, 0, 0, 32});
  PHX_CHECK((execution.remaining() == Counts{0, 0, 0, 0}));
  uint64_t bookkeeping = UINT64_MAX;
  PHX_CHECK(phx_mc_execution_allocator_bookkeeping(execution.scope, &bookkeeping) == PHX_MC_OK);
  PHX_CHECK(bookkeeping == 0);
  {
    Target target(batch.ticket);
    phx::mc::NativeVector<uint8_t> first(16, 7);
    phx::mc::NativeVector<uint8_t> second(16, 9);
    PHX_CHECK(owner->live_bytes() == 32 && owner->available_bytes() == 0);
    PHX_CHECK(target.usage->native_bytes == 32 && target.usage->native_peak_bytes == 32);
    PHX_CHECK(target.usage->host_bytes == 0 && target.usage->host_peak_bytes == 0);
    PHX_CHECK(first[15] == 7 && second[15] == 9);
    PHX_CHECK(phx_mc_execution_allocator_bookkeeping(execution.scope, &bookkeeping) == PHX_MC_OK);
    PHX_CHECK(bookkeeping > 0);
  }
  PHX_CHECK(owner->live_bytes() == 0);
  PHX_CHECK(phx_mc_execution_allocator_bookkeeping(execution.scope, &bookkeeping) == PHX_MC_OK);
  PHX_CHECK(bookkeeping == 0);
  batch.settle();
  PHX_CHECK((execution.remaining() == Counts{0, 0, 0, 32}));
  const Receipt receipt = execution.finish();
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] == 32);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_ALLOCATIONS] == 2 && receipt.memory[PHX_MC_EXECUTION_MEMORY_REFUSALS] == 0);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_PEAK_UPPER_BYTES] == 0);
}

void test_zero_bytes() {
  Execution execution({0, 0, 0, 64});
  const auto owner = phx::mc::scratch_memory_owner();
  Batch batch(execution.scope, {0, 0, 0, 0});
  {
    Target target(batch.ticket);
    bool refused = false;
    try {
      phx::mc::NativeVector<uint8_t> buffer(1, 0);
    } catch (const std::bad_alloc&) {
      refused = true;
    }
    PHX_CHECK(refused);
    PHX_CHECK(owner->live_bytes() == 0 && owner->allocations() == 0);
    PHX_CHECK(target.usage->native_bytes == 0 && target.usage->native_peak_bytes == 0);
    PHX_CHECK(phx::mc::native_execution_refusal() == PHX_MC_CAPACITY_EXCEEDED);
    const Refusal snapshot = execution.refusal();
    PHX_CHECK(snapshot[PHX_MC_BATCH_REMAINING_BYTES] == 0);
    PHX_CHECK((execution.refused_request() == Counts{0, 0, 0, 1}));
  }
  batch.settle();
  const Receipt receipt = execution.finish(PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_REFUSALS] == 1);
}

void test_host_native_bytes() {
  Execution execution({0, 0, 0, 64});
  const auto owner = phx::mc::scratch_memory_owner();
  Batch batch(execution.scope, {0, 0, 0, 32});
  {
    Target target(batch.ticket);
    target.usage->host_bytes = 16;
    target.usage->host_peak_bytes = 16;
    phx::mc::NativeVector<uint8_t> buffer(16, 7);
    PHX_CHECK(owner->live_bytes() == 16);
    PHX_CHECK(target.usage->native_bytes == 16 && target.usage->host_bytes == 16);
    PHX_CHECK(mapped_counter(target.usage->combined_peak_bytes) == 32);
    bool refused = false;
    try {
      phx::mc::NativeVector<uint8_t> extra(1, 0);
    } catch (const std::bad_alloc&) {
      refused = true;
    }
    PHX_CHECK(refused);
    PHX_CHECK(owner->live_bytes() == 16 && owner->allocations() == 1);
    const Refusal snapshot = execution.refusal();
    PHX_CHECK(snapshot[PHX_MC_BATCH_REMAINING_BYTES] == 0);
    PHX_CHECK((execution.refused_request() == Counts{0, 0, 0, 1}));
    batch.actual = {0, 0, 0, 16};
  }
  batch.settle();
  PHX_CHECK((execution.remaining(PHX_MC_CAPACITY_EXCEEDED) == Counts{0, 0, 0, 64}));
  const Receipt receipt = execution.finish(PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] == 16);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_ALLOCATIONS] == 1 && receipt.memory[PHX_MC_EXECUTION_MEMORY_REFUSALS] == 1);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_PEAK_UPPER_BYTES] == 16);
}

void test_reserved_host_array() {
  constexpr uint64_t bytes = sizeof(phx::mc::NativeHostBuffer) + 32;
  Execution execution({0, 0, 0, bytes});
  Batch batch(execution.scope, {0, 0, 0, bytes});
  void* array = nullptr;
  void* data = nullptr;
  {
    Target target(batch.ticket);
    PHX_CHECK(phx_mc_execution_allocate_host_array(
                  execution.scope, 32, 8, &array, &data) == PHX_MC_OK);
    PHX_CHECK(array != nullptr && data != nullptr);
    static_cast<uint64_t*>(data)[0] = 37;
    PHX_CHECK(target.usage->native_bytes == bytes);
    PHX_CHECK(target.usage->native_peak_bytes == bytes);
  }
  batch.settle();
  PHX_CHECK((execution.remaining() == Counts{0, 0, 0, 0}));
  const Receipt receipt = execution.finish();
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == bytes && receipt.memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] == bytes);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_ALLOCATIONS] == 2 && receipt.memory[PHX_MC_EXECUTION_MEMORY_REFUSALS] == 0);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_PEAK_UPPER_BYTES] == 0);
  bool valid = false;
  std::thread finalizer([&] {
    valid = static_cast<uint64_t*>(data)[0] == 37;
    phx_mc_execution_free_host_array(array);
  });
  finalizer.join();
  PHX_CHECK(valid);
}

void test_unused_release() {
  Execution execution({10, 10, 10, 64});
  Batch batch(execution.scope, {10, 10, 10, 64});
  {
    Target target(batch.ticket);
    target.usage->host_work = 2;
    target.usage->host_queries = 1;
    PHX_CHECK(phx::mc::native_execution_spend(3, 2));
    PHX_CHECK(phx::mc::native_execution_cavity(4));
    phx::mc::NativeVector<uint8_t> buffer(16, 5);
    batch.actual = {2, 1, 0, 0};
  }
  batch.settle();
  PHX_CHECK((execution.remaining() == Counts{5, 7, 10, 64}));
  Batch next(execution.scope, {5, 7, 10, 64});
  next.settle();
  const Receipt receipt = execution.finish();
  check_work(receipt, 5, 3, 2, 1);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_PEAK_CAVITY] == 4 && receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] == 16);
}

void test_escaped_memory() {
  Execution execution({0, 0, 0, 64});
  std::weak_ptr<phx::mc::BoundedMemoryResource> lifetime(phx::mc::scratch_memory_owner());
  phx::mc::NativeVector<uint8_t> escaped;
  Batch batch(execution.scope, {0, 0, 0, 64});
  {
    Target target(batch.ticket);
    escaped = phx::mc::NativeVector<uint8_t>(32, 19);
  }
  batch.settle();
  PHX_CHECK((execution.remaining() == Counts{0, 0, 0, 32}));
  Batch next(execution.scope, {0, 0, 0, 32});
  {
    Target target(next.ticket);
    phx::mc::NativeVector<uint8_t> second(32, 23);
    PHX_CHECK(phx::mc::scratch_memory_owner()->live_bytes() == 64);
  }
  next.settle();
  const Receipt receipt = execution.finish();
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 32 && receipt.memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] == 64);
  PHX_CHECK(!lifetime.expired() && !phx::mc::scratch_memory_owner());
  PHX_CHECK(escaped.front() == 19 && escaped.back() == 19);
  bool payload_valid = false;
  std::thread finalizer([buffer = std::move(escaped), &payload_valid]() mutable {
    payload_valid = buffer.front() == 19 && buffer.back() == 19;
    buffer.back() = 29;
    phx::mc::NativeVector<uint8_t> empty;
    buffer.swap(empty);
  });
  finalizer.join();
  PHX_CHECK(payload_valid && lifetime.expired());
}

void test_nested_restore() {
  Execution execution({10, 10, 10, 64});
  const auto owner = phx::mc::scratch_memory_owner();
  Batch outer(execution.scope, {6, 6, 6, 32});
  Batch inner(execution.scope, {4, 4, 4, 32});
  {
    Target first(outer.ticket);
    PHX_CHECK(phx::mc::native_execution_spend(1, 1));
    {
      Target second(inner.ticket);
      PHX_CHECK(phx::mc::scratch_memory_owner() == owner);
      PHX_CHECK(phx::mc::native_execution_spend(2, 2));
      phx::mc::NativeVector<uint8_t> buffer(16, 3);
      PHX_CHECK(second.usage->native_bytes == 16);
    }
    PHX_CHECK(phx::mc::scratch_memory_owner() == owner);
    PHX_CHECK(phx::mc::native_execution_spend(3, 2));
    PHX_CHECK(first.usage->native_work == 4 && first.usage->native_queries == 3);
  }
  inner.settle();
  outer.settle();
  PHX_CHECK(phx::mc::native_execution_spend(1, 1));
  const Receipt receipt = execution.finish();
  check_work(receipt, 7, 6, 0, 0);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] == 16);
}

void test_exception_restore() {
  Execution execution({10, 10, 10, 64});
  Batch outer(execution.scope, {6, 6, 6, 32});
  Batch inner(execution.scope, {4, 4, 4, 32});
  {
    Target target(outer.ticket);
    bool escaped = false;
    try {
      phx::mc::NativeExecutionTarget nested(
          *static_cast<phx::mc::NativeExecutionBatch*>(inner.ticket));
      PHX_CHECK(phx::mc::native_execution_spend(2, 1));
      phx::mc::NativeVector<uint8_t> buffer(16, 3);
      throw std::runtime_error("operation failed after measured native work");
    } catch (const std::runtime_error&) {
      escaped = true;
    }
    PHX_CHECK(escaped);
    PHX_CHECK(phx::mc::native_execution_spend(3, 2));
    PHX_CHECK(target.usage->native_work == 3 && target.usage->native_queries == 2);
    phx::mc::NativeVector<uint8_t> restored_buffer(16, 5);
    PHX_CHECK(target.usage->native_bytes == 16);
  }
  inner.settle();
  outer.settle();
  PHX_CHECK(phx::mc::native_execution_spend(1, 1));
  const Receipt receipt = execution.finish();
  check_work(receipt, 6, 4, 0, 0);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] == 16);
  PHX_CHECK(phx::mc::active_execution_scope == nullptr && !phx::mc::scratch_memory_owner());
}

void test_worker_child() {
  Execution execution({8, 6, 6, 64});
  Batch parent(execution.scope, {8, 6, 6, 64});
  Batch child(execution.scope, {5, 4, 4, 32}, parent.ticket);
  const auto owner = phx::mc::scratch_memory_owner();
  std::barrier gate(2);
  bool worker_valid = false;
  bool worker_restored = false;
  std::thread worker([&] {
    const bool initially_empty = phx::mc::active_execution_scope == nullptr &&
                                 !phx::mc::scratch_memory_owner();
    {
      Target target(child.ticket);
      worker_valid = initially_empty && phx::mc::scratch_memory_owner() == owner;
      worker_valid = worker_valid && phx::mc::native_execution_spend(4, 3);
      phx::mc::NativeVector<uint8_t> buffer(24, 7);
      gate.arrive_and_wait();
      gate.arrive_and_wait();
    }
    worker_restored = phx::mc::active_execution_scope == nullptr &&
                      !phx::mc::scratch_memory_owner();
    child.settle();
  });
  gate.arrive_and_wait();
  void* refused = nullptr;
  PHX_CHECK(phx_mc_execution_batch_admit(execution.scope, parent.ticket,
                                        4, 0, 0, 0, &refused) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(refused == nullptr);
  const Refusal snapshot = execution.refusal();
  PHX_CHECK(snapshot[PHX_MC_BATCH_REMAINING_WORK] == 3);
  PHX_CHECK((execution.refused_request() == Counts{4, 0, 0, 0}));
  gate.arrive_and_wait();
  worker.join();
  PHX_CHECK(worker_valid && worker_restored);
  parent.settle();
  PHX_CHECK((execution.remaining(PHX_MC_CAPACITY_EXCEEDED) == Counts{4, 3, 6, 64}));
  const Receipt receipt = execution.finish(PHX_MC_CAPACITY_EXCEEDED);
  check_work(receipt, 4, 3, 0, 0);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] == 24);
}

void test_child_scope() {
  Execution execution({8, 6, 5, 64});
  Batch batch(execution.scope, {8, 6, 5, 64});
  {
    Target target(batch.ticket);
    target.usage->host_work = 2;
    target.usage->host_queries = 1;
    Execution child({4, 3, 4, 32});
    PHX_CHECK((child.remaining() == Counts{4, 3, 4, 32}));
    PHX_CHECK(phx::mc::native_execution_spend(3, 2));
    PHX_CHECK(phx::mc::native_execution_cavity(3));
    {
      phx::mc::NativeVector<uint8_t> buffer(16, 11);
    }
    const Receipt child_receipt = child.finish();
    check_work(child_receipt, 3, 2, 0, 0);
    PHX_CHECK(child_receipt.work[PHX_MC_EXECUTION_PEAK_CAVITY] == 3);
    PHX_CHECK(child_receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0 && child_receipt.memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] == 16);
    PHX_CHECK(child_receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_PEAK_UPPER_BYTES] >= child_receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES]);
    PHX_CHECK(target.usage->native_work == 3 && target.usage->native_queries == 2);
    batch.actual = {2, 1, 0, 0};
  }
  batch.settle();
  const Receipt receipt = execution.finish();
  check_work(receipt, 5, 3, 2, 1);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_PEAK_CAVITY] == 3 && receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] == 16);
}

void test_child_scope_refusal() {
  Execution execution({8, 6, 5, 64});
  Batch batch(execution.scope, {8, 6, 5, 64});
  Refusal snapshot{};
  {
    Target target(batch.ticket);
    Execution child({2, 3, 4, 32});
    PHX_CHECK(phx::mc::native_execution_spend(1, 1));
    PHX_CHECK(!phx::mc::native_execution_spend(2, 0));
    snapshot = child.refusal();
    PHX_CHECK(snapshot[PHX_MC_BATCH_REMAINING_WORK] == 1);
    PHX_CHECK(snapshot[PHX_MC_BATCH_CONSUMED_WORK] == 1);
    PHX_CHECK(snapshot[PHX_MC_BATCH_CONSUMED_QUERIES] == 1);
    PHX_CHECK((child.refused_request() == Counts{2, 0, 0, 0}));
    const Receipt child_receipt = child.finish(PHX_MC_CAPACITY_EXCEEDED);
    check_work(child_receipt, 1, 1, 0, 0);
    PHX_CHECK(child_receipt.work[PHX_MC_EXECUTION_WORK_REFUSALS] == 1);
  }
  batch.settle();
  PHX_CHECK(execution.refusal() == snapshot);
  PHX_CHECK((execution.refused_request() == Counts{2, 0, 0, 0}));
  const Receipt receipt = execution.finish(PHX_MC_CAPACITY_EXCEEDED);
  check_work(receipt, 1, 1, 0, 0);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_WORK_REFUSALS] == 1 && receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0);
}

void test_invalid_lifetimes() {
  Execution execution({10, 10, 10, 64});
  Batch outer(execution.scope, {6, 6, 6, 32});
  Batch inner(execution.scope, {4, 4, 4, 32});
  Target first(outer.ticket);
  Receipt untouched;
  untouched.work.fill(91);
  untouched.memory.fill(92);
  double seconds = 93;
  PHX_CHECK(phx_mc_execution_end(execution.scope, untouched.work.data(),
                                 untouched.memory.data(), &seconds) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(untouched.work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 91 && untouched.memory[PHX_MC_EXECUTION_MEMORY_LIMIT_BYTES] == 92 && seconds == 93);
  PHX_CHECK(phx_mc_execution_batch_settle(outer.ticket, 0, 0, 0, 0) == PHX_MC_INVALID_ARGUMENT);
  const Counts actual{};
  const Counts requested{11, 0, 0, 0};
  Refusal refused{};
  refused.fill(91);
  PHX_CHECK(phx_mc_execution_batch_refuse(outer.ticket, actual.data(), requested.data(),
                                         refused.data()) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(std::all_of(refused.begin(), refused.end(),
                        [](uint64_t value) { return value == 91; }));
  void* duplicate = first.target;
  phx_mc_execution_usage* duplicate_usage = first.usage;
  PHX_CHECK(phx_mc_execution_target_begin(outer.ticket, &duplicate, &duplicate_usage) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(duplicate == nullptr && duplicate_usage == nullptr);
  int32_t wrong_end = PHX_MC_OK;
  int32_t wrong_begin = PHX_MC_OK;
  bool outputs_cleared = false;
  std::thread other([&] {
    wrong_end = phx_mc_execution_target_end(first.target);
    void* handle = first.target;
    phx_mc_execution_usage* usage = first.usage;
    wrong_begin = phx_mc_execution_target_begin(outer.ticket, &handle, &usage);
    outputs_cleared = handle == nullptr && usage == nullptr;
  });
  other.join();
  PHX_CHECK(wrong_end == PHX_MC_INVALID_ARGUMENT && wrong_begin == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(outputs_cleared);
  {
    Target second(inner.ticket);
    PHX_CHECK(phx_mc_execution_target_end(first.target) == PHX_MC_INVALID_ARGUMENT);
    PHX_CHECK(phx::mc::native_execution_spend(1, 1));
  }
  {
    Execution child({2, 2, 2, 16});
    PHX_CHECK(phx_mc_execution_target_end(first.target) == PHX_MC_INVALID_ARGUMENT);
    PHX_CHECK(phx::mc::native_execution_spend(1, 1));
  }
  PHX_CHECK(phx::mc::native_execution_spend(1, 1));
  first.finish();
  PHX_CHECK(phx_mc_execution_target_end(nullptr) == PHX_MC_INVALID_ARGUMENT);
  void* invalid = nullptr;
  phx_mc_execution_usage* usage = nullptr;
  PHX_CHECK(phx_mc_execution_target_begin(nullptr, &invalid, &usage) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_execution_target_begin(outer.ticket, nullptr, &usage) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_execution_target_begin(outer.ticket, &invalid, nullptr) == PHX_MC_INVALID_ARGUMENT);
  inner.settle();
  outer.settle();
  const Receipt receipt = execution.finish();
  check_work(receipt, 3, 3, 0, 0);
}

void test_imported_evidence() {
  Execution predecessor({3, 2, 0, 0});
  PHX_CHECK(phx::mc::native_execution_spend(3, 2));
  const Receipt prior = predecessor.finish();
  check_work(prior, 3, 2, 0, 0);
  Execution execution({10, 8, 0, 0});
  PHX_CHECK(phx_mc_execution_import_preparation(execution.scope, prior.work[PHX_MC_EXECUTION_CONSTRUCTION_WORK],
                                               prior.work[PHX_MC_EXECUTION_SOURCE_QUERIES], 0.125) == PHX_MC_OK);
  Batch batch(execution.scope, {7, 6, 0, 0});
  {
    Target target(batch.ticket);
    target.usage->host_work = 2;
    target.usage->host_queries = 1;
    PHX_CHECK(phx::mc::native_execution_spend(4, 3));
    batch.actual = {2, 1, 0, 0};
  }
  batch.settle();
  PHX_CHECK((execution.remaining() == Counts{1, 2, 0, 0}));
  double prior_seconds = -1;
  PHX_CHECK(phx_mc_execution_prior_seconds(execution.scope, &prior_seconds) == PHX_MC_OK);
  PHX_CHECK(prior_seconds == 0.125);
  const Receipt receipt = execution.finish();
  check_work(receipt, 9, 6, 5, 3);
}


void test_mapped_counter_reads() {
  Execution execution({12, 10, 0, 0});
  Batch parent(execution.scope, {12, 10, 0, 0});
  phx_mc_execution_usage* inactive_usage = nullptr;
  PHX_CHECK(phx_mc_execution_batch_usage(parent.ticket, &inactive_usage) == PHX_MC_OK);
  PHX_CHECK(inactive_usage != nullptr);
  PHX_CHECK(mapped_counter(inactive_usage->native_work) == 0);
  {
    Target target(parent.ticket);
    PHX_CHECK(target.usage == inactive_usage);
    phx_mc_execution_usage* rejected_view = inactive_usage;
    PHX_CHECK(phx_mc_execution_batch_usage(parent.ticket, &rejected_view) == PHX_MC_INVALID_ARGUMENT);
    PHX_CHECK(rejected_view == nullptr);
    PHX_CHECK(mapped_counter(target.usage->native_work) == 0);
    target.usage->host_work = 2;
    target.usage->host_queries = 1;
    PHX_CHECK(phx::mc::native_execution_spend(3, 2));
    PHX_CHECK(mapped_counter(target.usage->native_work) == 3);
    PHX_CHECK(mapped_counter(target.usage->native_queries) == 2);
    PHX_CHECK(phx_mc_execution_import_preparation(execution.scope, 1, 1, 0.25) == PHX_MC_OK);
    PHX_CHECK(mapped_counter(target.usage->native_work) == 3);
    PHX_CHECK(mapped_counter(target.usage->native_queries) == 2);
    PHX_CHECK(mapped_counter(target.usage->imported_work) == 1);
    PHX_CHECK(mapped_counter(target.usage->imported_queries) == 1);
    Batch child(execution.scope, {4, 4, 0, 0}, parent.ticket);
    PHX_CHECK(mapped_counter(target.usage->reserved_work) == 4);
    PHX_CHECK(mapped_counter(target.usage->reserved_queries) == 4);
    std::barrier gate(2);
    std::atomic<phx_mc_execution_usage*> mapped_child = nullptr;
    std::thread worker([&] {
      {
        Target child_target(child.ticket);
        child_target.usage->host_work = 1;
        child_target.usage->host_queries = 1;
        mapped_child.store(child_target.usage, std::memory_order_release);
        for (uint64_t step = 1; step <= 2; ++step) {
          PHX_CHECK(phx::mc::native_execution_spend(1, 1));
          if (step == 2) {
            PHX_CHECK(phx_mc_execution_import_preparation(
                          execution.scope, 1, 1, 0.125) == PHX_MC_OK);
          }
          gate.arrive_and_wait();
          gate.arrive_and_wait();
        }
        child.actual = {1, 1, 0, 0};
      }
      child.settle();
    });
    uint64_t prior_native = 0;
    for (uint64_t step = 1; step <= 2; ++step) {
      gate.arrive_and_wait();
      auto* usage = mapped_child.load(std::memory_order_acquire);
      PHX_CHECK(usage != nullptr);
      const uint64_t native = mapped_counter(usage->native_work);
      PHX_CHECK(native >= prior_native && native == step);
      PHX_CHECK(mapped_counter(usage->native_queries) == step);
      PHX_CHECK(mapped_counter(usage->imported_work) == step - 1);
      PHX_CHECK(mapped_counter(usage->imported_queries) == step - 1);
      PHX_CHECK(mapped_counter(target.usage->native_work) == 3);
      PHX_CHECK(mapped_counter(target.usage->children_work) == 0);
      prior_native = native;
      gate.arrive_and_wait();
    }
    worker.join();
    PHX_CHECK(phx_mc_execution_batch_checkpoint(parent.ticket) == PHX_MC_OK);
    PHX_CHECK(mapped_counter(target.usage->reserved_work) == 0);
    PHX_CHECK(mapped_counter(target.usage->reserved_queries) == 0);
    PHX_CHECK(mapped_counter(target.usage->children_work) == 4);
    PHX_CHECK(mapped_counter(target.usage->children_queries) == 4);
    PHX_CHECK(mapped_counter(target.usage->children_native_work) == 2);
    PHX_CHECK(mapped_counter(target.usage->children_native_queries) == 2);
    PHX_CHECK(mapped_counter(target.usage->native_work) +
                  mapped_counter(target.usage->children_native_work) == 5);
    PHX_CHECK(target.usage->host_work + mapped_counter(target.usage->native_work) +
                  mapped_counter(target.usage->imported_work) +
                  mapped_counter(target.usage->children_work) == 10);
    PHX_CHECK(target.usage->host_queries + mapped_counter(target.usage->native_queries) +
                  mapped_counter(target.usage->imported_queries) +
                  mapped_counter(target.usage->children_queries) == 8);
    parent.actual = {2, 1, 0, 0};
  }
  parent.settle();
  PHX_CHECK((execution.remaining() == Counts{2, 2, 0, 0}));
  double prior_seconds = -1;
  PHX_CHECK(phx_mc_execution_prior_seconds(execution.scope, &prior_seconds) == PHX_MC_OK);
  PHX_CHECK(prior_seconds == 0.375);
  const Receipt receipt = execution.finish();
  check_work(receipt, 10, 8, 5, 4);
}

void test_child_escaped_memory() {
  phx::mc::NativeVector<uint8_t> escaped;
  Execution execution({0, 0, 0, 64});
  const auto owner = phx::mc::scratch_memory_owner();
  Batch parent(execution.scope, {0, 0, 0, 64});
  {
    Target target(parent.ticket);
    Batch child(execution.scope, {0, 0, 0, 32}, parent.ticket);
    {
      Target child_target(child.ticket);
      escaped = phx::mc::NativeVector<uint8_t>(32, 17);
      PHX_CHECK(mapped_counter(child_target.usage->native_bytes) == 32);
    }
    child.settle();
    PHX_CHECK(mapped_counter(target.usage->reserved_bytes) == 0);
    PHX_CHECK(mapped_counter(target.usage->children_live_bytes) == 32);
    PHX_CHECK(owner->live_bytes() == 32 && owner->available_bytes() == 0);
    std::barrier gate(2);
    bool payload_valid = false;
    std::thread finalizer([buffer = std::move(escaped), &gate, &payload_valid]() mutable {
      payload_valid = buffer.front() == 17 && buffer.back() == 17;
      gate.arrive_and_wait();
      phx::mc::NativeVector<uint8_t> empty;
      buffer.swap(empty);
    });
    gate.arrive_and_wait();
    phx::mc::NativeVector<uint8_t> first(16, 23);
    finalizer.join();
    PHX_CHECK(payload_valid);
    PHX_CHECK(mapped_counter(target.usage->children_live_bytes) == 0);
    PHX_CHECK(mapped_counter(target.usage->native_bytes) == 16);
    PHX_CHECK(owner->live_bytes() == 16 && owner->available_bytes() == 0);
    phx::mc::NativeVector<uint8_t> second(48, 29);
    PHX_CHECK(owner->live_bytes() == 64 && owner->peak_bytes() == 64);
    PHX_CHECK(mapped_counter(target.usage->native_bytes) == 64);
    PHX_CHECK(first.back() == 23 && second.back() == 29);
  }
  parent.settle();
  PHX_CHECK((execution.remaining() == Counts{0, 0, 0, 64}));
  const Receipt receipt = execution.finish();
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] == 64);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_ALLOCATIONS] == 3 && receipt.memory[PHX_MC_EXECUTION_MEMORY_REFUSALS] == 0);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_PEAK_UPPER_BYTES] == 0);
}

void test_borrowed_scope_batch() {
  Execution execution({8, 6, 4, 32});
  Batch parent(execution.scope, {8, 6, 4, 32});
  {
    Target target(parent.ticket);
    Execution child_scope({4, 4, 3, 16});
    Batch child(child_scope.scope, {3, 3, 3, 16});
    {
      Target child_target(child.ticket);
      child_target.usage->host_work = 1;
      child_target.usage->host_queries = 1;
      PHX_CHECK(phx::mc::native_execution_spend(2, 1));
      PHX_CHECK(phx::mc::native_execution_cavity(2));
      phx::mc::NativeVector<uint8_t> buffer(8, 19);
      child.actual = {1, 1, 0, 0};
    }
    child.settle();
    PHX_CHECK((child_scope.remaining() == Counts{1, 2, 3, 16}));
    PHX_CHECK(mapped_counter(target.usage->children_work) == 3);
    PHX_CHECK(mapped_counter(target.usage->children_queries) == 2);
    PHX_CHECK(mapped_counter(target.usage->children_native_work) == 2);
    void* refused = nullptr;
    PHX_CHECK(phx_mc_execution_batch_admit(child_scope.scope, nullptr,
                                          2, 0, 0, 0, &refused) == PHX_MC_CAPACITY_EXCEEDED);
    PHX_CHECK(refused == nullptr);
    const Refusal snapshot = child_scope.refusal();
    PHX_CHECK(snapshot[PHX_MC_BATCH_REMAINING_WORK] == 1);
    PHX_CHECK(snapshot[PHX_MC_BATCH_CONSUMED_WORK] == 3);
    PHX_CHECK(snapshot[PHX_MC_BATCH_CONSUMED_QUERIES] == 2);
    PHX_CHECK((child_scope.refused_request() == Counts{2, 0, 0, 0}));
    const Receipt child_receipt = child_scope.finish(PHX_MC_CAPACITY_EXCEEDED);
    check_work(child_receipt, 3, 2, 1, 1);
    PHX_CHECK(child_receipt.work[PHX_MC_EXECUTION_PEAK_CAVITY] == 2 && child_receipt.work[PHX_MC_EXECUTION_WORK_REFUSALS] == 1);
    PHX_CHECK(child_receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0 && child_receipt.memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] == 8);
  }
  parent.settle();
  const Receipt receipt = execution.finish(PHX_MC_CAPACITY_EXCEEDED);
  check_work(receipt, 3, 2, 1, 1);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_PEAK_CAVITY] == 2 && receipt.work[PHX_MC_EXECUTION_WORK_REFUSALS] == 1);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] == 8);
}

void test_retained_host_storage() {
  constexpr uint64_t upper = 1024;
  Execution execution({0, 0, 0, upper});
  const auto owner = phx::mc::scratch_memory_owner();
  Batch batch(execution.scope, {0, 0, 0, upper});
  void* reservation = nullptr;
  uint64_t metadata = 0;
  {
    Target target(batch.ticket);
    PHX_CHECK(phx_mc_execution_reserve_host_storage(
                  execution.scope, 64, &reservation) == PHX_MC_OK);
    PHX_CHECK(reservation != nullptr);
    metadata = owner->live_bytes();
    PHX_CHECK(metadata > 0 && metadata <= upper - 64);
    PHX_CHECK(mapped_counter(target.usage->native_bytes) == metadata);
    PHX_CHECK(mapped_counter(target.usage->retained_host_bytes) == 64);
    PHX_CHECK(target.usage->host_bytes == 0);
    PHX_CHECK(phx_mc_execution_reserve_host_storage(
                  execution.scope, 32, &reservation) == PHX_MC_OK);
    PHX_CHECK(mapped_counter(target.usage->retained_host_bytes) == 32);
    PHX_CHECK(phx_mc_execution_reserve_host_storage(
                  execution.scope, 0, &reservation) == PHX_MC_OK);
    PHX_CHECK(mapped_counter(target.usage->retained_host_bytes) == 0);
    {
      phx::mc::NativeVector<uint8_t> buffer(upper - metadata, 23);
      PHX_CHECK(owner->live_bytes() == upper && owner->available_bytes() == 0);
      PHX_CHECK(mapped_counter(target.usage->native_bytes) == upper);
    }
    PHX_CHECK(owner->live_bytes() == metadata);
    PHX_CHECK(phx_mc_execution_reserve_host_storage(
                  execution.scope, 64, &reservation) == PHX_MC_OK);
    PHX_CHECK(mapped_counter(target.usage->retained_host_bytes) == 64);
  }
  batch.settle();
  PHX_CHECK((execution.remaining() == Counts{0, 0, 0, upper - metadata - 64}));
  const Receipt receipt = execution.finish();
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == metadata && receipt.memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] == upper);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_ALLOCATIONS] == 2 && receipt.memory[PHX_MC_EXECUTION_MEMORY_REFUSALS] == 0);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 64 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_PEAK_UPPER_BYTES] == 64);
  std::thread finalizer([reservation] {
    phx_mc_execution_release_host_storage(reservation);
  });
  finalizer.join();
  PHX_CHECK(owner->live_bytes() == 0 && owner->available_bytes() == upper);
}

void test_retained_host_refusal() {
  Execution execution({0, 0, 0, 1024});
  const auto owner = phx::mc::scratch_memory_owner();
  Batch batch(execution.scope, {0, 0, 0, 64});
  {
    Target target(batch.ticket);
    void* reservation = nullptr;
    PHX_CHECK(phx_mc_execution_reserve_host_storage(
                  execution.scope, 128, &reservation) == PHX_MC_CAPACITY_EXCEEDED);
    PHX_CHECK(reservation == nullptr);
    PHX_CHECK(mapped_counter(target.usage->native_bytes) == 0);
    PHX_CHECK(mapped_counter(target.usage->retained_host_bytes) == 0);
    PHX_CHECK(owner->live_bytes() == 0 && owner->available_bytes() == 960);
    PHX_CHECK(phx::mc::native_execution_refusal() == PHX_MC_CAPACITY_EXCEEDED);
  }
  batch.settle();
  PHX_CHECK((execution.remaining(PHX_MC_CAPACITY_EXCEEDED) == Counts{0, 0, 0, 1024}));
  const Receipt receipt = execution.finish(PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0);
  PHX_CHECK(owner->live_bytes() == 0 && owner->available_bytes() == 1024);
}

struct RefusedOperation {
  Refusal snapshot{};
  Receipt receipt{};
  Counts remaining{};
  Counts requested{};
};

RefusedOperation root_refusal(std::size_t resource, bool atomic) {
  Execution execution({10, 10, 10, 64});
  const auto owner = phx::mc::scratch_memory_owner();
  Batch batch(execution.scope, {10, 10, 10, 64});
  {
    Target target(batch.ticket);
    target.usage->host_work = 3;
    target.usage->host_queries = 2;
    target.usage->host_cavity = 4;
    target.usage->host_peak_cavity = 4;
    target.usage->host_bytes = 5;
    target.usage->host_peak_bytes = 5;
    PHX_CHECK(phx::mc::native_execution_spend(2, 1));
    batch.actual = {3, 2, 4, 5};
  }
  Counts requested{1, 1, 1, 1};
  requested[resource] = Counts{6, 8, 11, 65}[resource];
  RefusedOperation result;
  if (atomic) {
    PHX_CHECK(phx_mc_execution_batch_refuse(batch.ticket, batch.actual.data(),
                                           requested.data(), result.snapshot.data()) ==
              PHX_MC_CAPACITY_EXCEEDED);
    batch.ticket = nullptr;
  } else {
    batch.settle();
    void* refused = nullptr;
    PHX_CHECK(phx_mc_execution_batch_admit(
                  execution.scope, nullptr, requested[0], requested[1],
                  requested[2], requested[3], &refused) == PHX_MC_CAPACITY_EXCEEDED);
    PHX_CHECK(refused == nullptr);
    result.snapshot = execution.refusal();
  }
  PHX_CHECK(execution.refusal() == result.snapshot);
  result.requested = execution.refused_request();
  PHX_CHECK(result.requested == requested);
  PHX_CHECK(result.snapshot[PHX_MC_BATCH_CONSUMED_WORK] == 5);
  PHX_CHECK(result.snapshot[PHX_MC_BATCH_CONSUMED_QUERIES] == 3);
  result.remaining = execution.remaining(PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK((result.remaining == Counts{5, 7, 10, 64}));
  PHX_CHECK(owner->live_bytes() == 0 && owner->available_bytes() == 64);
  result.receipt = execution.finish(PHX_MC_CAPACITY_EXCEEDED);
  check_work(result.receipt, 5, 3, 3, 2);
  PHX_CHECK(result.receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0 && result.receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_PEAK_UPPER_BYTES] == 5);
  return result;
}

void check_equivalent(const RefusedOperation& direct, const RefusedOperation& atomic) {
  PHX_CHECK(direct.snapshot == atomic.snapshot);
  PHX_CHECK(direct.remaining == atomic.remaining);
  PHX_CHECK(direct.requested == atomic.requested);
  PHX_CHECK(direct.receipt.work == atomic.receipt.work);
  PHX_CHECK(direct.receipt.memory == atomic.receipt.memory);
}

void test_atomic_refusal_equivalence() {
  for (std::size_t resource = 0; resource < 4; ++resource) {
    const auto direct = root_refusal(resource, false);
    const auto atomic = root_refusal(resource, true);
    check_equivalent(direct, atomic);
  }
}

RefusedOperation child_refusal(bool atomic, bool worker) {
  Execution execution({10, 10, 10, 64});
  const auto owner = phx::mc::scratch_memory_owner();
  Batch parent(execution.scope, {10, 10, 10, 64});
  Batch child(execution.scope, {6, 6, 6, 32}, parent.ticket);
  Batch sibling(execution.scope, {4, 4, 4, 32}, parent.ticket);
  RefusedOperation result;
  const Counts requested{4, 1, 1, 1};
  auto operation = [&] {
    {
      Target target(child.ticket);
      target.usage->host_work = 1;
      target.usage->host_queries = 1;
      target.usage->host_cavity = 4;
      target.usage->host_peak_cavity = 4;
      target.usage->host_bytes = 5;
      target.usage->host_peak_bytes = 5;
      PHX_CHECK(phx::mc::native_execution_spend(2, 1));
      child.actual = {1, 1, 4, 5};
    }
    if (atomic) {
      PHX_CHECK(phx_mc_execution_batch_refuse(child.ticket, child.actual.data(),
                                             requested.data(), result.snapshot.data()) ==
                PHX_MC_CAPACITY_EXCEEDED);
      child.ticket = nullptr;
    } else {
      child.settle();
      void* refused = nullptr;
      PHX_CHECK(phx_mc_execution_batch_admit(
                    execution.scope, parent.ticket, requested[0], requested[1],
                    requested[2], requested[3], &refused) == PHX_MC_CAPACITY_EXCEEDED);
      PHX_CHECK(refused == nullptr);
      result.snapshot = execution.refusal();
    }
  };
  if (worker) {
    std::thread thread(operation);
    thread.join();
  } else {
    operation();
  }
  PHX_CHECK(execution.refusal() == result.snapshot);
  result.requested = execution.refused_request();
  PHX_CHECK(result.requested == requested);
  PHX_CHECK(result.snapshot[PHX_MC_BATCH_REMAINING_WORK] == 3);
  PHX_CHECK(result.snapshot[PHX_MC_BATCH_CONSUMED_WORK] == 3);
  PHX_CHECK(result.snapshot[PHX_MC_BATCH_CONSUMED_QUERIES] == 2);
  sibling.settle();
  parent.actual = {1, 1, 0, 0};
  parent.settle();
  result.remaining = execution.remaining(PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK((result.remaining == Counts{6, 7, 10, 64}));
  PHX_CHECK(owner->live_bytes() == 0 && owner->available_bytes() == 64);
  result.receipt = execution.finish(PHX_MC_CAPACITY_EXCEEDED);
  check_work(result.receipt, 4, 3, 2, 2);
  return result;
}

void test_atomic_refusal_child() {
  const auto direct = child_refusal(false, false);
  const auto atomic = child_refusal(true, false);
  check_equivalent(direct, atomic);
}

void test_atomic_refusal_worker() {
  const auto direct = child_refusal(false, false);
  const auto atomic = child_refusal(true, true);
  check_equivalent(direct, atomic);
}

void test_atomic_refusal_estimate_defect() {
  Execution execution({10, 10, 10, 64});
  const auto owner = phx::mc::scratch_memory_owner();
  Batch batch(execution.scope, {10, 10, 10, 64});
  const Counts actual{3, 2, 4, 5};
  const Counts fits{7, 8, 10, 64};
  Refusal untouched{};
  untouched.fill(91);
  PHX_CHECK(phx_mc_execution_batch_refuse(batch.ticket, actual.data(), fits.data(),
                                         untouched.data()) == PHX_MC_INTERNAL_ERROR);
  PHX_CHECK(std::all_of(untouched.begin(), untouched.end(),
                        [](uint64_t value) { return value == 91; }));
  PHX_CHECK((execution.remaining() == Counts{0, 0, 0, 0}));
  PHX_CHECK(phx_mc_execution_batch_refusal(execution.scope, untouched.data()) ==
            PHX_MC_INVALID_ARGUMENT);
  Counts no_request{};
  no_request.fill(91);
  PHX_CHECK(phx_mc_execution_refusal_request(execution.scope, no_request.data()) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK((no_request == Counts{91, 91, 91, 91}));
  PHX_CHECK(owner->live_bytes() == 0 && owner->available_bytes() == 0);
  const Counts defect{11, 2, 4, 5};
  const Counts exceeds{8, 0, 0, 0};
  PHX_CHECK(phx_mc_execution_batch_refuse(nullptr, actual.data(), exceeds.data(),
                                         untouched.data()) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_execution_batch_refuse(batch.ticket, nullptr, exceeds.data(),
                                         untouched.data()) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_execution_batch_refuse(batch.ticket, actual.data(), nullptr,
                                         untouched.data()) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_execution_batch_refuse(batch.ticket, actual.data(), exceeds.data(),
                                         nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_execution_batch_refuse(batch.ticket, defect.data(), exceeds.data(),
                                         untouched.data()) == PHX_MC_INTERNAL_ERROR);
  PHX_CHECK((execution.remaining() == Counts{0, 0, 0, 0}));
  PHX_CHECK(phx_mc_execution_batch_refuse(batch.ticket, actual.data(), exceeds.data(),
                                         untouched.data()) == PHX_MC_CAPACITY_EXCEEDED);
  batch.ticket = nullptr;
  PHX_CHECK((untouched == Refusal{7, 8, 10, 64, 3, 2, 4, 5}));
  PHX_CHECK(execution.refusal() == untouched);
  PHX_CHECK(execution.refused_request() == exceeds);
  PHX_CHECK((execution.remaining(PHX_MC_CAPACITY_EXCEEDED) == Counts{7, 8, 10, 64}));
  PHX_CHECK(owner->available_bytes() == 64);
  const Receipt receipt = execution.finish(PHX_MC_CAPACITY_EXCEEDED);
  check_work(receipt, 3, 2, 3, 2);
  PHX_CHECK(receipt.work[PHX_MC_EXECUTION_WORK_REFUSALS] == 1 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_PEAK_UPPER_BYTES] == 5);
}

void test_real_mesh_creation() {
  const std::array<double, 12> points = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  const std::array<int32_t, 4> tets = {0, 1, 2, 3};
  const std::array<int32_t, 1> regions = {0};
  const std::array<int32_t, 12> faces = {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
  const std::array<int32_t, 4> face_sources = {0, 1, 2, 3};
  const std::array<int32_t, 12> segments = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};
  const std::array<int32_t, 6> segment_sources = {0, 1, 2, 3, 4, 5};
  Execution execution({9000, 80, 64, 900000});
  const auto owner = phx::mc::scratch_memory_owner();
  Batch batch(execution.scope, {9000, 80, 64, 900000});
  Counts native{};
  {
    Target target(batch.ticket);
    PHX_CHECK(owner->limit_bytes() == 900000);
    PHX_CHECK(owner->limit_bytes() != (1 << 20));
    phx_mc_tet_mesh* mesh = nullptr;
    const phx_mc_tet_mesh_create_request request{
        .point_count = 4,
        .points = points.data(),
        .tet_count = 1,
        .tets = tets.data(),
        .tet_regions = regions.data(),
        .face_count = 4,
        .faces = faces.data(),
        .face_sources = face_sources.data(),
        .segment_count = 6,
        .segments = segments.data(),
        .segment_sources = segment_sources.data(),
        .protection_radii = nullptr,
        .source_point_count = 0,
        .source_points = nullptr,
        .source_face_count = 0,
        .source_faces = nullptr,
        .source_face_group_rows = nullptr,
        .source_face_tolerances = nullptr,
        .source_segment_count = 0,
        .source_segments = nullptr,
        .source_segment_group_rows = nullptr,
        .source_segment_tolerances = nullptr,
        .witness_strata = nullptr,
        .witness_entities = nullptr,
        .witness_parameters = nullptr,
        .boundary_policy = PHX_MC_TET_MESH_BOUNDARY_FIXED,
        .max_vertices = 16,
        .max_tetrahedra = 64,
        .max_scratch_bytes = 1 << 20,
        .mesh = &mesh,
    };
    const int32_t create_status = phx_mc_tet_mesh_create(&request);
    PHX_CHECK(create_status == PHX_MC_OK);
    PHX_CHECK(mesh != nullptr);
    native[0] = mapped_counter(target.usage->native_work);
    native[1] = mapped_counter(target.usage->native_queries);
    PHX_CHECK(native[0] > 0 && native[0] <= 9000 && native[1] <= 80);
    PHX_CHECK(mapped_counter(target.usage->native_bytes) > 0);
    PHX_CHECK(mapped_counter(target.usage->native_bytes) == owner->live_bytes());
    PHX_CHECK(mapped_counter(target.usage->native_peak_bytes) <= 900000);
    PHX_CHECK(target.usage->host_work == 0 && target.usage->host_queries == 0);
    phx_mc_tet_mesh_free(mesh);
    PHX_CHECK(owner->live_bytes() == 0);
    PHX_CHECK(mapped_counter(target.usage->native_bytes) == 0);
    uint64_t bookkeeping = UINT64_MAX;
    PHX_CHECK(phx_mc_execution_allocator_bookkeeping(execution.scope, &bookkeeping) == PHX_MC_OK);
    PHX_CHECK(bookkeeping == 0);
  }
  batch.settle();
  const Receipt receipt = execution.finish();
  check_work(receipt, native[0], native[1], 0, 0);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] > 0);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] <= 900000 && receipt.memory[PHX_MC_EXECUTION_MEMORY_REFUSALS] == 0);
  PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0 && receipt.memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_PEAK_UPPER_BYTES] == 0);
}

void admitting_parent_refusal(bool batch_admission) {
  for (std::size_t resource = 0; resource < 2; ++resource) {
    Counts limits{10, 10, 3, 1024};
    limits[resource] = 1;
    Counts requested{};
    requested[resource] = 1;
    Execution parent(limits);
    {
      Batch spent(parent.scope, requested);
      spent.actual = requested;
    }
    Execution child({100, 100, 3, 1024});
    if (batch_admission) {
      void* denied = child.scope;
      PHX_CHECK(phx_mc_execution_batch_admit(
          child.scope, nullptr, requested[0], requested[1], 0, 0, &denied) ==
          PHX_MC_CAPACITY_EXCEEDED);
      PHX_CHECK(denied == nullptr);
    } else {
      PHX_CHECK(!static_cast<phx::mc::NativeExecutionScope*>(child.scope)->admit(
          requested[0], requested[1]));
    }
    const Refusal refusal = child.refusal();
    PHX_CHECK(child.refused_request() == requested);
    PHX_CHECK(refusal[resource] == 0);
    PHX_CHECK(refusal[PHX_MC_BATCH_CONSUMED_WORK + resource] == 1);
    PHX_CHECK(refusal[PHX_MC_BATCH_CONSUMED_WORK + (1 - resource)] == 0);
    const Receipt child_receipt = child.finish(PHX_MC_CAPACITY_EXCEEDED);
    check_work(child_receipt, 0, 0, 0, 0);
    PHX_CHECK(child_receipt.work[PHX_MC_EXECUTION_WORK_REFUSALS + resource] == 1);
    limits[resource] = 0;
    PHX_CHECK(parent.remaining(PHX_MC_CAPACITY_EXCEEDED) == limits);
    const Receipt parent_receipt = parent.finish(PHX_MC_CAPACITY_EXCEEDED);
    check_work(parent_receipt, requested[0], requested[1], requested[0], requested[1]);
    PHX_CHECK(parent_receipt.work[PHX_MC_EXECUTION_WORK_REFUSALS + resource] == 1);
  }
}

void test_admitting_parent_batch() { admitting_parent_refusal(true); }
void test_admitting_parent_native() { admitting_parent_refusal(false); }

struct Case {
  std::string_view name;
  void (*run)();
};
constexpr std::array cases = {
    Case{"reserved_native_charges", test_reserved_native_charges},
    Case{"host_native_interleaving", test_host_native_interleaving},
    Case{"native_refusal_evidence", test_native_refusal_evidence},
    Case{"reserved_bytes", test_reserved_bytes},
    Case{"host_native_bytes", test_host_native_bytes},
    Case{"reserved_host_array", test_reserved_host_array},
    Case{"zero_bytes", test_zero_bytes},
    Case{"unused_release", test_unused_release},
    Case{"escaped_memory", test_escaped_memory},
    Case{"child_escaped_memory", test_child_escaped_memory},
    Case{"nested_restore", test_nested_restore},
    Case{"exception_restore", test_exception_restore},
    Case{"worker_child", test_worker_child},
    Case{"child_scope", test_child_scope},
    Case{"child_scope_refusal", test_child_scope_refusal},
    Case{"borrowed_scope_batch", test_borrowed_scope_batch},
    Case{"retained_host_storage", test_retained_host_storage},
    Case{"retained_host_refusal", test_retained_host_refusal},
    Case{"invalid_lifetimes", test_invalid_lifetimes},
    Case{"imported_evidence", test_imported_evidence},
    Case{"mapped_counter_reads", test_mapped_counter_reads},
    Case{"atomic_refusal_equivalence", test_atomic_refusal_equivalence},
    Case{"atomic_refusal_child", test_atomic_refusal_child},
    Case{"atomic_refusal_estimate_defect", test_atomic_refusal_estimate_defect},
    Case{"atomic_refusal_worker", test_atomic_refusal_worker},
    Case{"real_mesh_creation", test_real_mesh_creation},
    Case{"admitting_parent_batch", test_admitting_parent_batch},
    Case{"admitting_parent_native", test_admitting_parent_native},
};
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  for (const Case& test : cases) {
    if (test.name == argv[1]) {
      test.run();
      return phx::mc::test::finish(argv[1]);
    }
  }
  return 1;
}
