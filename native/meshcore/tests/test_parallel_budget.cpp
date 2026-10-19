//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include <array>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <thread>

#include "bounded_memory.hpp"
#include "check.hpp"

namespace {
using namespace phx::mc;
constexpr std::size_t storage_limit = 1U << 20;
using Resources = std::array<uint64_t, 4>;

class FailingArenaUpstream final : public std::pmr::memory_resource {
 public:
  explicit FailingArenaUpstream(bool unexpected) : unexpected_(unexpected) {}
  std::size_t requests = 0;
  std::size_t releases = 0;

 private:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    if (++requests == 2) {
      if (unexpected_) throw std::runtime_error("deliberate arena staging failure");
      throw std::bad_alloc();
    }
    return std::pmr::new_delete_resource()->allocate(bytes, alignment);
  }
  void do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) override {
    ++releases;
    std::pmr::new_delete_resource()->deallocate(pointer, bytes, alignment);
  }
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }
  bool unexpected_;
};

void staging_failure_releases_reservation(bool unexpected) {
  FailingArenaUpstream upstream(unexpected);
  auto owner = std::make_shared<BoundedMemoryResource>(storage_limit, &upstream);
  NativeExecutionScope parent(100, 100, 100, storage_limit, INFINITY, owner);
  const auto before = parent.remaining();
  const std::array allowances{NativeChunkAllowance{4, 3, 2, 128}};
  bool threw = false;
  try {
    NativeParallelBatch batch(allowances);
    PHX_CHECK(!unexpected && batch.status() == PHX_MC_CAPACITY_EXCEEDED);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  PHX_CHECK(threw == unexpected);
  PHX_CHECK(upstream.requests == 2);
  PHX_CHECK(upstream.releases == 1);
  PHX_CHECK(parent.remaining() == before);
  PHX_CHECK(!parent.has_live_batches());
  PHX_CHECK(parent.status() == (unexpected ? PHX_MC_INTERNAL_ERROR : PHX_MC_CAPACITY_EXCEEDED));
  PHX_CHECK(parent.evidence()[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 0);
  PHX_CHECK(parent.memory_evidence()[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0);
}
void staging_allocation_release() { staging_failure_releases_reservation(false); }
void staging_exception_release() { staging_failure_releases_reservation(true); }

void admission_refusal(std::size_t resource) {
  NativeExecutionScope parent(10, 10, 10, resource == 3 ? 0 : storage_limit, INFINITY);
  const auto before = parent.remaining();
  NativeChunkAllowance allowance{1, 1, 1, 0};
  if (resource == 0) allowance.work = 11;
  if (resource == 1) allowance.queries = 11;
  if (resource == 2) allowance.cavity = 11;
  const std::array allowances{allowance};
  {
    NativeParallelBatch batch(allowances);
    PHX_CHECK(batch.status() == PHX_MC_CAPACITY_EXCEEDED);
    PHX_CHECK(parent.remaining() == before);
    PHX_CHECK(!parent.has_live_batches());
  }
  const auto counters = parent.evidence();
  const auto memory = parent.memory_evidence();
  PHX_CHECK(counters[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 0);
  PHX_CHECK(counters[PHX_MC_EXECUTION_SOURCE_QUERIES] == 0);
  PHX_CHECK(memory[PHX_MC_EXECUTION_MEMORY_ALLOCATIONS] == 0);
  PHX_CHECK(memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0);
}
void work_admission() { admission_refusal(0); }
void query_admission() { admission_refusal(1); }
void cavity_admission() { admission_refusal(2); }
void byte_admission() { admission_refusal(3); }

void settlement_releases_unused_resources() {
  NativeExecutionScope parent(100, 100, 100, storage_limit, INFINITY);
  const auto before = parent.remaining();
  const auto bytes = native_parallel_allocation_bytes(32, alignof(uint64_t));
  const std::array allowances{NativeChunkAllowance{9, 7, 5, bytes},
                              NativeChunkAllowance{9, 7, 5, bytes}};
  NativeParallelBatch batch(allowances);
  PHX_CHECK(batch.status() == PHX_MC_OK);
  const auto reserved = parent.remaining();
  PHX_CHECK(reserved[0] == before[0] - 18);
  PHX_CHECK(reserved[1] == before[1] - 14);
  PHX_CHECK(reserved[2] == before[2] - 10);
  PHX_CHECK(reserved[3] < before[3] - 2 * bytes);
  std::array<NativeChunkEvidence, 2> evidence;
  {
    NativeDetachedExecution child(batch.chunk_allowance(0), batch.chunk_owner(0));
    native_execution_charge(3, 2);
    PHX_CHECK(child.execution().charge_external(2, 1));
    PHX_CHECK(native_execution_cavity(2));
    native_execution_primitive_query();
    {
      NativeVector<uint64_t> buffer(4, 17);
      PHX_CHECK(buffer[3] == 17);
    }
    evidence[0] = child.evidence();
  }
  {
    NativeDetachedExecution child(batch.chunk_allowance(1), batch.chunk_owner(1));
    native_execution_charge(1, 1);
    PHX_CHECK(native_execution_cavity(3));
    native_execution_primitive_query();
    evidence[1] = child.evidence(PHX_MC_CONSTRAINT_INTERSECTION);
  }
  PHX_CHECK(parent.evidence()[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 0);
  PHX_CHECK(parent.evidence()[PHX_MC_EXECUTION_PRIMITIVE_QUERIES] == 0);
  PHX_CHECK(batch.settle(evidence) == PHX_MC_CONSTRAINT_INTERSECTION);
  PHX_CHECK((parent.remaining() == Resources{94, 96, 100, storage_limit}));
  const auto counters = parent.evidence();
  PHX_CHECK(counters[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 6);
  PHX_CHECK(counters[PHX_MC_EXECUTION_SOURCE_QUERIES] == 4);
  PHX_CHECK(counters[PHX_MC_EXECUTION_PEAK_CAVITY] == 5);
  PHX_CHECK(counters[PHX_MC_EXECUTION_EXTERNAL_WORK] == 2);
  PHX_CHECK(counters[PHX_MC_EXECUTION_EXTERNAL_QUERIES] == 1);
  PHX_CHECK(counters[PHX_MC_EXECUTION_PRIMITIVE_QUERIES] == 2);
  PHX_CHECK(parent.status() == PHX_MC_OK);
  PHX_CHECK(!parent.has_live_batches());
  PHX_CHECK(parent.memory_evidence()[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0);
}

void bound_defect(std::size_t resource) {
  NativeExecutionScope parent(100, 100, 100, storage_limit, INFINITY);
  const auto bytes = native_parallel_allocation_bytes(64);
  const std::array allowances{NativeChunkAllowance{4, 3, 2, bytes}};
  NativeParallelBatch batch(allowances);
  PHX_CHECK(batch.status() == PHX_MC_OK);
  std::array<NativeChunkEvidence, 1> evidence;
  {
    NativeDetachedExecution child(batch.chunk_allowance(0), batch.chunk_owner(0));
    native_execution_charge(1, 1);
    bool refused = false;
    if (resource == 0 || resource == 1) {
      try {
        native_execution_charge(resource == 0 ? 4 : 0, resource == 1 ? 3 : 0);
      } catch (const ExecutionRefusal&) {
        refused = true;
      }
    } else if (resource == 2) {
      refused = !native_execution_cavity(3);
    } else {
      // The numerical request fits the local payload limit, but not the fixed
      // arena after its allocator headers. Missing header bounds are defects.
      try {
        NativeVector<std::byte> buffer(static_cast<std::size_t>(bytes));
      } catch (const std::bad_alloc&) {
        refused = true;
      }
    }
    PHX_CHECK(refused);
    evidence[0] = child.evidence();
    PHX_CHECK(evidence[0].status == PHX_MC_INTERNAL_ERROR);
  }
  PHX_CHECK(batch.settle(evidence) == PHX_MC_INTERNAL_ERROR);
  PHX_CHECK(parent.status() == PHX_MC_INTERNAL_ERROR);
  PHX_CHECK((parent.remaining() == Resources{99, 99, 100, storage_limit}));
  PHX_CHECK(!parent.has_live_batches());
  PHX_CHECK(parent.evidence()[PHX_MC_EXECUTION_WORK_REFUSALS] == 0);
  PHX_CHECK(parent.evidence()[PHX_MC_EXECUTION_QUERY_REFUSALS] == 0);
  PHX_CHECK(parent.evidence()[PHX_MC_EXECUTION_CAVITY_REFUSALS] == 0);
}
void work_bound_defect() { bound_defect(0); }
void query_bound_defect() { bound_defect(1); }
void cavity_bound_defect() { bound_defect(2); }
void byte_bound_defect() { bound_defect(3); }

void nested_parent_ticket_routing() {
  NativeExecutionScope parent(100, 100, 100, 4 * storage_limit, INFINITY);
  void* admitted = nullptr;
  PHX_CHECK(NativeExecutionBatch::admit(parent, nullptr, {50, 50, 50, 2 * storage_limit},
                                        &admitted) == PHX_MC_OK);
  std::unique_ptr<NativeExecutionBatch> ticket(static_cast<NativeExecutionBatch*>(admitted));
  {
    NativeExecutionTarget target(*ticket);
    {
      NativeExecutionScope borrowed(30, 30, 30, storage_limit, INFINITY);
      const auto bytes = native_parallel_allocation_bytes(32);
      const std::array allowances{NativeChunkAllowance{4, 3, 2, bytes},
                                  NativeChunkAllowance{4, 3, 2, bytes}};
      NativeParallelBatch batch(allowances);
      PHX_CHECK(batch.status() == PHX_MC_OK);
      std::array<NativeChunkEvidence, 2> evidence;
      for (std::size_t index = 0; index < evidence.size(); ++index) {
        NativeDetachedExecution child(batch.chunk_allowance(index), batch.chunk_owner(index));
        native_execution_charge(2, 1);
        PHX_CHECK(child.execution().charge_external(1, 1));
        PHX_CHECK(native_execution_cavity(1));
        native_execution_primitive_query();
        evidence[index] = child.evidence();
      }
      PHX_CHECK(parent.evidence()[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 0);
      PHX_CHECK(borrowed.evidence()[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 0);
      PHX_CHECK(batch.settle(evidence) == PHX_MC_OK);
      PHX_CHECK(borrowed.evidence()[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 6);
      PHX_CHECK(borrowed.evidence()[PHX_MC_EXECUTION_EXTERNAL_WORK] == 2);
      PHX_CHECK(parent.evidence()[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 0);
      PHX_CHECK(active_execution_scope == &borrowed);
      PHX_CHECK(active_execution_target == &target);
    }
    PHX_CHECK(active_execution_scope == &parent);
    PHX_CHECK(active_execution_target == &target);
    native_execution_charge(1, 1);
  }
  PHX_CHECK(ticket->settle({0, 0, 0, 0}) == PHX_MC_OK);
  ticket.reset();
  const auto counters = parent.evidence();
  PHX_CHECK(counters[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 7);
  PHX_CHECK(counters[PHX_MC_EXECUTION_SOURCE_QUERIES] == 5);
  PHX_CHECK(counters[PHX_MC_EXECUTION_EXTERNAL_WORK] == 2);
  PHX_CHECK(counters[PHX_MC_EXECUTION_EXTERNAL_QUERIES] == 2);
  PHX_CHECK(counters[PHX_MC_EXECUTION_PRIMITIVE_QUERIES] == 2);
  PHX_CHECK((parent.remaining() == Resources{93, 95, 100, 4 * storage_limit}));
  PHX_CHECK(!parent.has_live_batches());
}

void workers_leave_parent_unchanged() {
  NativeExecutionScope parent(100, 100, 100, storage_limit, INFINITY);
  const auto bytes = native_parallel_allocation_bytes(32);
  const std::array allowances{NativeChunkAllowance{4, 3, 2, bytes},
                              NativeChunkAllowance{4, 3, 2, bytes}};
  NativeParallelBatch batch(allowances);
  PHX_CHECK(batch.status() == PHX_MC_OK);
  std::array owners{batch.chunk_owner(0), batch.chunk_owner(1)};
  const auto counters = parent.evidence();
  const auto memory = parent.memory_evidence();
  std::array<NativeChunkEvidence, 2> evidence;
  std::array<bool, 2> detached{};
  std::barrier charged(3);
  std::barrier release(3);
  const auto body = [&](std::size_t index) {
    NativeDetachedExecution child(allowances[index], owners[index]);
    native_execution_charge(2, 1);
    native_execution_primitive_query();
    NativeVector<uint64_t> buffer(4, 9);
    detached[index] = active_execution_scope == &child.execution() &&
                      active_execution_target == nullptr && scratch_memory_owner() == owners[index];
    charged.arrive_and_wait();
    release.arrive_and_wait();
    evidence[index] = child.evidence();
  };
  std::jthread first(body, 0);
  std::jthread second(body, 1);
  charged.arrive_and_wait();
  PHX_CHECK(parent.evidence() == counters);
  PHX_CHECK(parent.memory_evidence() == memory);
  release.arrive_and_wait();
  first.join();
  second.join();
  PHX_CHECK(detached[0] && detached[1]);
  owners = {};
  PHX_CHECK(batch.settle(evidence) == PHX_MC_OK);
  PHX_CHECK(parent.evidence()[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 4);
  PHX_CHECK(parent.evidence()[PHX_MC_EXECUTION_PRIMITIVE_QUERIES] == 2);
  PHX_CHECK(parent.memory_evidence()[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0);
}

void arena_recycling_and_alignment() {
  NativeExecutionScope parent(100, 100, 100, storage_limit, INFINITY);
  const auto bytes = 2 * native_parallel_allocation_bytes(32, 64);
  const std::array allowances{NativeChunkAllowance{0, 0, 0, bytes}};
  NativeParallelBatch batch(allowances);
  PHX_CHECK(batch.status() == PHX_MC_OK);
  std::array<NativeChunkEvidence, 1> evidence;
  {
    NativeDetachedExecution child(batch.chunk_allowance(0), batch.chunk_owner(0));
    auto* resource = scratch_memory_resource();
    for (std::size_t iteration = 0; iteration < 100; ++iteration) {
      void* first = resource->allocate(32, 64);
      void* second = resource->allocate(32, 64);
      PHX_CHECK(reinterpret_cast<std::uintptr_t>(first) % 64 == 0);
      PHX_CHECK(reinterpret_cast<std::uintptr_t>(second) % 64 == 0);
      resource->deallocate(first, 32, 64);
      resource->deallocate(second, 32, 64);
    }
    evidence[0] = child.evidence();
    PHX_CHECK(evidence[0].memory.allocations == 200);
    PHX_CHECK(evidence[0].memory.live_bytes == 0);
    PHX_CHECK(evidence[0].memory.peak_bytes == 64);
    PHX_CHECK(evidence[0].memory.refusals == 0);
  }
  PHX_CHECK(batch.settle(evidence) == PHX_MC_OK);
  PHX_CHECK(parent.memory_evidence()[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0);
}

void arena_outlives_batch_and_parent_scope() {
  std::optional<NativeVector<uint64_t>> escaped;
  std::weak_ptr<BoundedMemoryResource> arena_lifetime;
  std::weak_ptr<BoundedMemoryResource> parent_lifetime;
  {
    NativeExecutionScope parent(100, 100, 100, storage_limit, INFINITY);
    parent_lifetime = scratch_memory_owner();
    const auto bytes = native_parallel_allocation_bytes(32);
    const std::array allowances{NativeChunkAllowance{0, 0, 0, bytes}};
    NativeParallelBatch batch(allowances);
    PHX_CHECK(batch.status() == PHX_MC_OK);
    arena_lifetime = batch.chunk_owner(0);
    std::array<NativeChunkEvidence, 1> evidence;
    {
      NativeDetachedExecution child(batch.chunk_allowance(0), batch.chunk_owner(0));
      escaped.emplace(4, 37);
      evidence[0] = child.evidence();
    }
    PHX_CHECK(batch.settle(evidence) == PHX_MC_OK);
    PHX_CHECK(!parent.has_live_batches());
    PHX_CHECK(parent.memory_evidence()[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] > 32);
  }
  PHX_CHECK(!arena_lifetime.expired());
  PHX_CHECK(!parent_lifetime.expired());
  PHX_CHECK((*escaped)[3] == 37);
  escaped.reset();
  PHX_CHECK(arena_lifetime.expired());
  PHX_CHECK(parent_lifetime.expired());
}

void exception_evidence_releases_reservation() {
  NativeExecutionScope parent(100, 100, 100, storage_limit, INFINITY);
  const auto bytes = native_parallel_allocation_bytes(32);
  const std::array allowances{NativeChunkAllowance{4, 3, 2, bytes}};
  NativeParallelBatch batch(allowances);
  PHX_CHECK(batch.status() == PHX_MC_OK);
  const auto owner = scratch_memory_owner();
  std::array<NativeChunkEvidence, 1> evidence;
  {
    NativeDetachedExecution child(batch.chunk_allowance(0), batch.chunk_owner(0));
    try {
      NativeVector<uint64_t> temporary(4, 1);
      native_execution_charge(2, 1);
      PHX_CHECK(child.execution().charge_external(1, 1));
      throw std::runtime_error("deliberate chunk failure");
    } catch (const std::runtime_error&) {
      evidence[0] = child.evidence(PHX_MC_INTERNAL_ERROR, false);
    }
  }
  PHX_CHECK(active_execution_scope == &parent);
  PHX_CHECK(active_execution_target == nullptr);
  PHX_CHECK(scratch_memory_owner() == owner);
  PHX_CHECK(batch.settle(evidence) == PHX_MC_INTERNAL_ERROR);
  PHX_CHECK(parent.status() == PHX_MC_INTERNAL_ERROR);
  PHX_CHECK((parent.remaining() == Resources{97, 98, 100, storage_limit}));
  PHX_CHECK(parent.evidence()[PHX_MC_EXECUTION_EXTERNAL_WORK] == 1);
  PHX_CHECK(!parent.has_live_batches());
}

void unwinding_releases_unlaunched_batch() {
  NativeExecutionScope parent(100, 100, 100, storage_limit, INFINITY);
  const auto before = parent.remaining();
  const std::array allowances{NativeChunkAllowance{4, 3, 2, 128}};
  try {
    NativeParallelBatch batch(allowances);
    PHX_CHECK(batch.status() == PHX_MC_OK);
    throw std::runtime_error("deliberate caller failure");
  } catch (const std::runtime_error&) {
  }
  PHX_CHECK(parent.remaining() == before);
  PHX_CHECK(parent.status() == PHX_MC_OK);
  PHX_CHECK(!parent.has_live_batches());
}

void detached_construction_failure_restores_affinity() {
  NativeExecutionScope parent(100, 100, 100, storage_limit, INFINITY);
  void* admitted = nullptr;
  PHX_CHECK(NativeExecutionBatch::admit(parent, nullptr, {10, 10, 10, 1024}, &admitted) ==
            PHX_MC_OK);
  std::unique_ptr<NativeExecutionBatch> ticket(static_cast<NativeExecutionBatch*>(admitted));
  {
    NativeExecutionTarget target(*ticket);
    const auto owner = scratch_memory_owner();
    bool failed = false;
    try {
      NativeDetachedExecution child(NativeChunkAllowance{1, 1, 1, 0}, {});
    } catch (const std::invalid_argument&) {
      failed = true;
    }
    PHX_CHECK(failed);
    PHX_CHECK(active_execution_scope == &parent);
    PHX_CHECK(active_execution_target == &target);
    PHX_CHECK(scratch_memory_owner() == owner);
    native_execution_charge(1, 1);
  }
  PHX_CHECK(ticket->settle({0, 0, 0, 0}) == PHX_MC_OK);
  ticket.reset();
  PHX_CHECK(parent.evidence()[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 1);
}

void timeout_preserves_completed_chunk_evidence() {
  NativeExecutionScope parent(100, 100, 100, storage_limit, INFINITY);
  const std::array allowances{NativeChunkAllowance{4, 3, 2, 0}, NativeChunkAllowance{4, 3, 2, 0}};
  NativeParallelBatch batch(allowances);
  PHX_CHECK(batch.status() == PHX_MC_OK);
  std::array<NativeChunkEvidence, 2> evidence;
  {
    NativeDetachedExecution child(batch.chunk_allowance(0), batch.chunk_owner(0));
    native_execution_charge(2, 1);
    evidence[0] = child.evidence();
  }
  evidence[1].status = PHX_MC_TIMEOUT;
  PHX_CHECK(batch.settle(evidence) == PHX_MC_TIMEOUT);
  PHX_CHECK(parent.status() == PHX_MC_TIMEOUT);
  PHX_CHECK((parent.remaining() == Resources{98, 99, 100, storage_limit}));
  PHX_CHECK(!parent.has_live_batches());
}

void allowance_sum_overflow_is_internal() {
  NativeExecutionScope parent(UINT64_MAX, UINT64_MAX, UINT64_MAX, storage_limit, INFINITY);
  const auto before = parent.remaining();
  const std::array allowances{NativeChunkAllowance{UINT64_MAX, 0, 0, 0},
                              NativeChunkAllowance{1, 0, 0, 0}};
  NativeParallelBatch batch(allowances);
  PHX_CHECK(batch.status() == PHX_MC_INTERNAL_ERROR);
  PHX_CHECK(parent.status() == PHX_MC_INTERNAL_ERROR);
  PHX_CHECK(parent.remaining() == before);
  PHX_CHECK(!parent.has_live_batches());
  PHX_CHECK(parent.memory_evidence()[PHX_MC_EXECUTION_MEMORY_ALLOCATIONS] == 0);
  bool failed = false;
  try {
    native_parallel_allocation_bytes(UINT64_MAX);
  } catch (const std::overflow_error&) {
    failed = true;
  }
  PHX_CHECK(failed);
}

void implicit_parent_restores_affinity() {
  PHX_CHECK(active_execution_scope == nullptr);
  PHX_CHECK(active_execution_target == nullptr);
  const auto prior_owner = scratch_memory_owner();
  const std::array allowances{NativeChunkAllowance{4, 3, 2, 0}};
  {
    NativeParallelBatch batch(allowances);
    PHX_CHECK(batch.status() == PHX_MC_OK);
    std::array<NativeChunkEvidence, 1> evidence;
    {
      NativeDetachedExecution child(batch.chunk_allowance(0), batch.chunk_owner(0));
      native_execution_charge(2, 1);
      evidence[0] = child.evidence();
    }
    PHX_CHECK(batch.settle(evidence) == PHX_MC_OK);
  }
  PHX_CHECK(active_execution_scope == nullptr);
  PHX_CHECK(active_execution_target == nullptr);
  PHX_CHECK(scratch_memory_owner() == prior_owner);
}

struct Case {
  std::string_view name;
  void (*run)();
};
constexpr std::array cases = {
    Case{"work_admission", work_admission},
    Case{"query_admission", query_admission},
    Case{"cavity_admission", cavity_admission},
    Case{"byte_admission", byte_admission},
    Case{"unused_release", settlement_releases_unused_resources},
    Case{"work_bound_defect", work_bound_defect},
    Case{"query_bound_defect", query_bound_defect},
    Case{"cavity_bound_defect", cavity_bound_defect},
    Case{"byte_bound_defect", byte_bound_defect},
    Case{"nested_parent_ticket", nested_parent_ticket_routing},
    Case{"worker_isolation", workers_leave_parent_unchanged},
    Case{"arena_recycling", arena_recycling_and_alignment},
    Case{"arena_lifetime", arena_outlives_batch_and_parent_scope},
    Case{"exception_release", exception_evidence_releases_reservation},
    Case{"staging_allocation_release", staging_allocation_release},
    Case{"staging_exception_release", staging_exception_release},
    Case{"unwinding_release", unwinding_releases_unlaunched_batch},
    Case{"construction_restore", detached_construction_failure_restores_affinity},
    Case{"timeout_evidence", timeout_preserves_completed_chunk_evidence},
    Case{"sum_overflow", allowance_sum_overflow_is_internal},
    Case{"implicit_parent", implicit_parent_restores_affinity},
};
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  for (const auto& test : cases) {
    if (test.name == argv[1]) {
      test.run();
      return phx::mc::test::finish(argv[1]);
    }
  }
  return 1;
}
