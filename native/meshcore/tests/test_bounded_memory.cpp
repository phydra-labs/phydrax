//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <new>
#include <thread>
#include <utility>

#include "bounded_memory.hpp"
#include "check.hpp"
#include "expansion.hpp"
#include "improve3d.hpp"
#include "tet_mesh_fixtures.hpp"

namespace {
using namespace phx::mc;

class RefusingResource final : public std::pmr::memory_resource {
 public:
  bool refuse = false;
  std::size_t requests = 0;
  std::size_t refuse_at = 0;
  std::size_t releases = 0;
 private:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    ++requests;
    if (refuse || (refuse_at != 0 && requests == refuse_at)) throw std::bad_alloc();
    return std::pmr::new_delete_resource()->allocate(bytes, alignment);
  }
  void do_deallocate(void* buffer, std::size_t bytes, std::size_t alignment) override {
    ++releases;
    std::pmr::new_delete_resource()->deallocate(buffer, bytes, alignment);
  }
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }
};

void test_growth_counts_simultaneous_buffers_and_preserves_values() {
  auto owner = std::make_shared<BoundedMemoryResource>();
  MemoryScope scope(owner);
  NativeVector<std::uint64_t> values{1, 2, 3, 4};
  const std::size_t before = owner->live_bytes();
  PHX_CHECK(owner->set_limit(before));
  bool refused = false;
  try { values.reserve(16); } catch (const std::bad_alloc&) { refused = true; }
  PHX_CHECK(refused);
  const std::array<std::uint64_t, 4> expected = {1, 2, 3, 4};
  PHX_CHECK(values.size() == expected.size());
  PHX_CHECK(std::equal(values.begin(), values.end(), expected.begin()));
  PHX_CHECK(owner->live_bytes() == before);
  PHX_CHECK(owner->requested_bytes() == 16 * sizeof(std::uint64_t));
  PHX_CHECK(owner->refusals() == 1);
  PHX_CHECK(owner->set_limit(before + owner->requested_bytes()));
  values.reserve(16);
  PHX_CHECK(owner->peak_bytes() == before + 16 * sizeof(std::uint64_t));
  PHX_CHECK(values[0] == 1 && values[3] == 4);
}

void test_scopes_restore_and_spills_keep_allocator_lifetime() {
  auto first = std::make_shared<BoundedMemoryResource>();
  auto second = std::make_shared<BoundedMemoryResource>();
  std::weak_ptr<BoundedMemoryResource> lifetime(first);
  Expansion escaped;
  std::array<double, 32> components{};
  for (std::size_t i = 0; i < components.size(); ++i) {
    components[i] = std::ldexp(1.0, -1000 + 53 * static_cast<int>(i));
  }
  {
    MemoryScope scope(first);
    {
      MemoryScope nested(second);
      NativeVector<int32_t> data(8, 7);
      PHX_CHECK(second->live_bytes() == 8 * sizeof(int32_t));
      PHX_CHECK(first->live_bytes() == 0);
    }
    PHX_CHECK(scratch_memory_owner() == first);
    escaped = Expansion::from_components(components.data(), components.size());
    PHX_CHECK(first->live_bytes() == components.size() * sizeof(double));
  }
  PHX_CHECK(!scratch_memory_owner());
  first.reset();
  PHX_CHECK(!lifetime.expired());
  PHX_CHECK(escaped.sign() > 0);
  escaped = Expansion();
  PHX_CHECK(lifetime.expired());
}

void test_inline_expansion_needs_no_allocation_under_zero_limit() {
  auto owner = std::make_shared<BoundedMemoryResource>(0);
  MemoryScope scope(owner);
  const Expansion result = Expansion::product(1.25, 2.5) - Expansion(3.0);
  PHX_CHECK(result.sign() > 0);
  PHX_CHECK(owner->allocations() == 0 && owner->refusals() == 0);
}

void test_upstream_refusal_preserves_expansion_assignment() {
  RefusingResource upstream;
  auto owner = std::make_shared<BoundedMemoryResource>(SIZE_MAX, &upstream);
  std::array<double, 32> components{};
  for (std::size_t i = 0; i < components.size(); ++i) {
    components[i] = std::ldexp(1.0, -1000 + 53 * static_cast<int>(i));
  }
  Expansion source = Expansion::from_components(components.data(), components.size());
  Expansion destination(1.0);
  {
    MemoryScope scope(owner);
    upstream.refuse = true;
    bool refused = false;
    try { destination = source; } catch (const std::bad_alloc&) { refused = true; }
    PHX_CHECK(refused);
  }
  PHX_CHECK(destination.estimate() == 1.0);
  PHX_CHECK(owner->live_bytes() == 0 && owner->refusals() == 1);
}

void test_nested_window_shares_ledger_and_restores_parent_quota() {
  auto owner = std::make_shared<BoundedMemoryResource>(1024);
  MemoryScope scope(owner);
  NativeVector<int32_t> resident(8, 1);
  const std::size_t baseline = owner->live_bytes();
  {
    MemoryBudgetWindow window(64);
    MemoryScope nested(window.owner());
    NativeVector<int32_t> temporary(16, 2);
    PHX_CHECK(owner->live_bytes() == baseline + 64);
    bool refused = false;
    try { temporary.reserve(32); } catch (const std::bad_alloc&) { refused = true; }
    PHX_CHECK(refused);
    const auto evidence = window.evidence();
    PHX_CHECK(evidence[0] == 64 && evidence[2] == 64 && evidence[5] == 1);
  }
  PHX_CHECK(owner->limit_bytes() == 1024 && owner->live_bytes() == baseline);
}

void test_host_array_and_native_buffers_share_actual_cap() {
  constexpr std::size_t metadata = sizeof(NativeHostBuffer);
  auto owner = std::make_shared<BoundedMemoryResource>();
  MemoryScope memory_scope(owner);
  void* execution = nullptr;
  PHX_CHECK(phx_mc_execution_begin(UINT64_MAX, UINT64_MAX, UINT64_MAX,
                                   metadata + 96, INFINITY, nullptr, &execution) == PHX_MC_OK);
  NativeVector<uint64_t> resident(4, 7);
  void *host = nullptr, *data = nullptr;
  PHX_CHECK(phx_mc_execution_allocate_host_array(execution, 64, 64, &host, &data) == PHX_MC_OK);
  PHX_CHECK(reinterpret_cast<std::uintptr_t>(data) % 64 == 0);
  static_cast<uint64_t*>(data)[0] = 19;
  PHX_CHECK((owner->evidence() ==
             std::array<uint64_t, 6>{metadata + 96, metadata + 96, metadata + 96, 64, 3, 0}));
  uint64_t remaining[4];
  double seconds;
  PHX_CHECK(phx_mc_execution_remaining(execution, remaining, &seconds) == PHX_MC_OK);
  PHX_CHECK(remaining[3] == 0);
  void *refused_owner = host, *refused_data = data;
  PHX_CHECK(phx_mc_execution_allocate_host_array(execution, 1, 1, &refused_owner,
                                                &refused_data) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(refused_owner == nullptr && refused_data == nullptr);
  PHX_CHECK(owner->live_bytes() == metadata + 96 && owner->peak_bytes() == metadata + 96);
  PHX_CHECK(owner->allocations() == 3 && owner->refusals() == 1);
  PHX_CHECK(resident[0] == 7 && static_cast<uint64_t*>(data)[0] == 19);
  phx_mc_execution_free_host_array(host);
  PHX_CHECK(phx_mc_execution_remaining(execution, remaining, &seconds) == PHX_MC_OK);
  PHX_CHECK(remaining[3] == metadata + 64 && owner->live_bytes() == 32);
  uint64_t counters[PHX_MC_EXECUTION_COUNTERS], bytes[PHX_MC_EXECUTION_MEMORY_VALUES];
  PHX_CHECK(phx_mc_execution_end(execution, counters, bytes, &seconds) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(bytes[1] == 32 && bytes[2] == metadata + 96 && bytes[4] == 3 && bytes[5] == 1);
}

void test_host_payload_refusal_counts_only_successful_metadata() {
  constexpr std::size_t metadata = sizeof(NativeHostBuffer);
  RefusingResource upstream;
  auto owner = std::make_shared<BoundedMemoryResource>(SIZE_MAX, &upstream);
  MemoryScope memory_scope(owner);
  void* execution = nullptr;
  PHX_CHECK(phx_mc_execution_begin(0, 0, 0, metadata + 8, INFINITY, nullptr,
                                   &execution) == PHX_MC_OK);
  void *host = nullptr, *data = nullptr;
  PHX_CHECK(phx_mc_execution_allocate_host_array(execution, 16, 8, &host,
                                                &data) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(host == nullptr && data == nullptr);
  PHX_CHECK((owner->evidence() ==
             std::array<uint64_t, 6>{metadata + 8, 0, metadata, 16, 1, 1}));
  PHX_CHECK(upstream.requests == 1 && upstream.releases == 1);
  uint64_t remaining[4], counters[PHX_MC_EXECUTION_COUNTERS], bytes[PHX_MC_EXECUTION_MEMORY_VALUES];
  double seconds;
  PHX_CHECK(phx_mc_execution_remaining(execution, remaining, &seconds) == PHX_MC_OK);
  PHX_CHECK(remaining[3] == metadata + 8);
  PHX_CHECK(phx_mc_execution_end(execution, counters, bytes, &seconds) == PHX_MC_CAPACITY_EXCEEDED);
}

void test_host_upstream_oom_does_not_commit_failed_request() {
  constexpr std::size_t metadata = sizeof(NativeHostBuffer);
  for (std::size_t failure = 1; failure <= 2; ++failure) {
    RefusingResource upstream;
    upstream.refuse_at = failure;
    auto owner = std::make_shared<BoundedMemoryResource>(SIZE_MAX, &upstream);
    MemoryScope memory_scope(owner);
    void* execution = nullptr;
    PHX_CHECK(phx_mc_execution_begin(0, 0, 0, metadata + 64, INFINITY, nullptr,
                                     &execution) == PHX_MC_OK);
    void *host = nullptr, *data = nullptr;
    PHX_CHECK(phx_mc_execution_allocate_host_array(execution, 64, 8, &host,
                                                  &data) == PHX_MC_CAPACITY_EXCEEDED);
    PHX_CHECK(host == nullptr && data == nullptr && owner->live_bytes() == 0);
    PHX_CHECK(owner->peak_bytes() == (failure == 1 ? 0 : metadata));
    PHX_CHECK(owner->allocations() == failure - 1 && owner->refusals() == 1);
    PHX_CHECK(upstream.requests == failure && upstream.releases == failure - 1);
    uint64_t counters[PHX_MC_EXECUTION_COUNTERS], bytes[PHX_MC_EXECUTION_MEMORY_VALUES];
    double seconds;
    PHX_CHECK(phx_mc_execution_end(execution, counters, bytes, &seconds) == PHX_MC_CAPACITY_EXCEEDED);
  }
}

void test_host_array_zero_and_invalid_arguments_preserve_accounting() {
  constexpr std::size_t metadata = sizeof(NativeHostBuffer);
  void* execution = nullptr;
  PHX_CHECK(phx_mc_execution_begin(0, 0, 0, metadata, INFINITY, nullptr,
                                   &execution) == PHX_MC_OK);
  const auto owner = scratch_memory_owner();
  void *host = nullptr, *data = nullptr;
  PHX_CHECK(phx_mc_execution_allocate_host_array(execution, 0, 64, &host, &data) == PHX_MC_OK);
  PHX_CHECK(host != nullptr && data != nullptr && reinterpret_cast<std::uintptr_t>(data) % 64 == 0);
  const auto before = owner->evidence();
  PHX_CHECK((before == std::array<uint64_t, 6>{metadata, metadata, metadata, 0, 2, 0}));
  struct InvalidRequest { uint64_t bytes, alignment; };
  for (const auto request : std::array<InvalidRequest, 4>{{{0, 0}, {1, 3},
                                                          {UINT64_MAX, 8}, {1, UINT64_MAX}}}) {
    void *invalid_owner = host, *invalid_data = data;
    PHX_CHECK(phx_mc_execution_allocate_host_array(execution, request.bytes, request.alignment,
                                                  &invalid_owner, &invalid_data) == PHX_MC_INVALID_ARGUMENT);
    PHX_CHECK(invalid_owner == nullptr && invalid_data == nullptr);
  }
  void* ignored = nullptr;
  PHX_CHECK(phx_mc_execution_allocate_host_array(execution, 0, 8, nullptr,
                                                &ignored) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_execution_allocate_host_array(execution, 0, 8, &ignored,
                                                &ignored) == PHX_MC_INVALID_ARGUMENT);
  std::atomic<int32_t> other_thread_status{PHX_MC_OK};
  std::thread other_thread([&] {
    void *wrong_owner = nullptr, *wrong_data = nullptr;
    other_thread_status = phx_mc_execution_allocate_host_array(
        execution, 0, 8, &wrong_owner, &wrong_data);
  });
  other_thread.join();
  PHX_CHECK(other_thread_status == PHX_MC_INVALID_ARGUMENT && owner->evidence() == before);
  phx_mc_execution_free_host_array(host);
  phx_mc_execution_free_host_array(nullptr);
  uint64_t counters[PHX_MC_EXECUTION_COUNTERS], bytes[PHX_MC_EXECUTION_MEMORY_VALUES];
  double seconds;
  PHX_CHECK(phx_mc_execution_end(execution, counters, bytes, &seconds) == PHX_MC_OK);
  PHX_CHECK(bytes[1] == 0 && bytes[4] == 2 && bytes[5] == 0);
  PHX_CHECK(phx_mc_execution_allocate_host_array(execution, 0, 8, &ignored,
                                                &data) == PHX_MC_INVALID_ARGUMENT);
}

void test_host_array_nested_scopes_keep_root_minimum() {
  constexpr std::size_t metadata = sizeof(NativeHostBuffer);
  void *outer = nullptr, *inner = nullptr;
  PHX_CHECK(phx_mc_execution_begin(0, 0, 0, 256, INFINITY, nullptr, &outer) == PHX_MC_OK);
  void *outer_host = nullptr, *outer_data = nullptr;
  PHX_CHECK(phx_mc_execution_allocate_host_array(outer, 16, 8, &outer_host,
                                                &outer_data) == PHX_MC_OK);
  PHX_CHECK(phx_mc_execution_begin(0, 0, 0, 1024, INFINITY, nullptr, &inner) == PHX_MC_OK);
  void *inner_host = nullptr, *inner_data = nullptr;
  PHX_CHECK(phx_mc_execution_allocate_host_array(inner, 32, 8, &inner_host,
                                                &inner_data) == PHX_MC_OK);
  uint64_t remaining[4], counters[PHX_MC_EXECUTION_COUNTERS], bytes[PHX_MC_EXECUTION_MEMORY_VALUES];
  double seconds;
  PHX_CHECK(phx_mc_execution_remaining(inner, remaining, &seconds) == PHX_MC_OK);
  PHX_CHECK(remaining[3] == 256 - 2 * metadata - 48);
  void *refused_owner = nullptr, *refused_data = nullptr;
  PHX_CHECK(phx_mc_execution_allocate_host_array(inner, 256, 8, &refused_owner,
                                                &refused_data) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(phx_mc_execution_end(inner, counters, bytes, &seconds) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(bytes[0] == 256 - metadata - 16 && bytes[1] == metadata + 32);
  PHX_CHECK(bytes[2] == 2 * metadata + 32 && bytes[3] == 256 && bytes[4] == 3 && bytes[5] == 1);
  phx_mc_execution_free_host_array(inner_host);
  PHX_CHECK(phx_mc_execution_remaining(outer, remaining, &seconds) == PHX_MC_OK);
  PHX_CHECK(remaining[3] == 256 - metadata - 16);
  PHX_CHECK(phx_mc_execution_end(outer, counters, bytes, &seconds) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(bytes[0] == 256 && bytes[1] == metadata + 16 && bytes[4] == 5);
  phx_mc_execution_free_host_array(outer_host);
}

void test_host_allocation_preserves_original_execution_refusal() {
  for (const bool timed_out : {false, true}) {
    void* execution = nullptr;
    PHX_CHECK(phx_mc_execution_begin(0, 0, 0, 1024, timed_out ? 0 : INFINITY,
                                     nullptr, &execution) == PHX_MC_OK);
    const int32_t expected = timed_out ? PHX_MC_TIMEOUT : PHX_MC_CAPACITY_EXCEEDED;
    if (!timed_out) PHX_CHECK(phx_mc_execution_charge(execution, 1, 0) == expected);
    void *host = nullptr, *data = nullptr;
    PHX_CHECK(phx_mc_execution_allocate_host_array(execution, 64, 8, &host, &data) == expected);
    PHX_CHECK(host == nullptr && data == nullptr);
    uint64_t counters[PHX_MC_EXECUTION_COUNTERS], bytes[PHX_MC_EXECUTION_MEMORY_VALUES];
    double seconds;
    PHX_CHECK(phx_mc_execution_end(execution, counters, bytes, &seconds) == expected);
    PHX_CHECK(bytes[1] == 0 && bytes[2] == 0 && bytes[4] == 0 && bytes[5] == 0);
  }
}

void test_host_owner_outlives_scope_and_releases_on_another_thread() {
  void* execution = nullptr;
  PHX_CHECK(phx_mc_execution_begin(0, 0, 0, 1024, INFINITY, nullptr, &execution) == PHX_MC_OK);
  std::weak_ptr<BoundedMemoryResource> lifetime(scratch_memory_owner());
  void *host = nullptr, *data = nullptr;
  PHX_CHECK(phx_mc_execution_allocate_host_array(execution, 64, 8, &host, &data) == PHX_MC_OK);
  static_cast<uint64_t*>(data)[0] = 0x12345678;
  uint64_t counters[PHX_MC_EXECUTION_COUNTERS], bytes[PHX_MC_EXECUTION_MEMORY_VALUES];
  double seconds;
  PHX_CHECK(phx_mc_execution_end(execution, counters, bytes, &seconds) == PHX_MC_OK);
  PHX_CHECK(!lifetime.expired() && !scratch_memory_owner());
  PHX_CHECK(static_cast<uint64_t*>(data)[0] == 0x12345678);
  static_cast<uint64_t*>(data)[7] = 0x87654321;
  std::thread finalizer([host] { phx_mc_execution_free_host_array(host); });
  finalizer.join();
  PHX_CHECK(lifetime.expired());
}

void test_host_concurrent_release_serializes_pool_operations_and_scope_exit() {
  RefusingResource upstream;
  auto owner = std::make_shared<BoundedMemoryResource>(SIZE_MAX, &upstream);
  MemoryScope memory_scope(owner);
  void* execution = nullptr;
  PHX_CHECK(phx_mc_execution_begin(0, 0, 0, 1 << 20, INFINITY, nullptr,
                                   &execution) == PHX_MC_OK);
  std::array<void*, 256> buffers{};
  for (auto& buffer : buffers) {
    void* data = nullptr;
    PHX_CHECK(phx_mc_execution_allocate_host_array(execution, 64, 8, &buffer, &data) == PHX_MC_OK);
  }
  std::atomic<bool> start{false};
  std::thread finalizer([&] {
    while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
    for (void* buffer : buffers) {
      phx_mc_execution_free_host_array(buffer);
      std::this_thread::yield();
    }
  });
  start.store(true, std::memory_order_release);
  for (int iteration = 0; iteration < 512; ++iteration) {
    MemoryBudgetWindow window(1024);
    NativeVector<uint64_t> values(8, 3);
    values.reserve(16);
    const auto evidence = window.evidence();
    PHX_CHECK(evidence[1] <= evidence[0] && evidence[2] <= evidence[0] && evidence[5] == 0);
  }
  uint64_t counters[PHX_MC_EXECUTION_COUNTERS], bytes[PHX_MC_EXECUTION_MEMORY_VALUES];
  double seconds;
  PHX_CHECK(phx_mc_execution_end(execution, counters, bytes, &seconds) == PHX_MC_OK);
  finalizer.join();
  PHX_CHECK(owner->live_bytes() == 0 && owner->allocations() == 1536 && owner->refusals() == 0);
  PHX_CHECK(upstream.requests == 1536 && upstream.releases == 1536);
}

void test_external_host_storage_shares_native_cap_and_releases_after_refusal() {
  auto owner = std::make_shared<BoundedMemoryResource>(1024);
  MemoryScope memory_scope(owner);
  void* execution = nullptr;
  PHX_CHECK(phx_mc_execution_begin(UINT64_MAX, UINT64_MAX, UINT64_MAX,
      1024, INFINITY, nullptr, &execution) == PHX_MC_OK);
  void* reservation = nullptr;
  PHX_CHECK(phx_mc_execution_reserve_host_storage(execution, 700, &reservation) == PHX_MC_OK);
  PHX_CHECK(reservation != nullptr);
  const std::size_t metadata = owner->live_bytes();
  // Only token metadata is measured allocation; the live host upper bound
  // nevertheless consumes the exact same original allowance.
  PHX_CHECK(metadata > 0 && metadata < 324);
  PHX_CHECK(owner->available_bytes() == 1024 - metadata - 700);
  void* original_token = reservation;
  const std::size_t bound = 1024 - metadata - 8;
  PHX_CHECK(phx_mc_execution_reserve_host_storage(execution, bound, &reservation) == PHX_MC_OK);
  PHX_CHECK(reservation == original_token && owner->available_bytes() == 8);
  void *array = nullptr, *data = nullptr;
  PHX_CHECK(phx_mc_execution_allocate_host_array(execution, 16, 8, &array, &data) ==
            PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(array == nullptr && data == nullptr && owner->live_bytes() == metadata);
  PHX_CHECK(phx_mc_execution_reserve_host_storage(execution, 0, &reservation) == PHX_MC_OK);
  PHX_CHECK(owner->available_bytes() == 1024 - metadata);
  uint64_t work[PHX_MC_EXECUTION_COUNTERS], memory[PHX_MC_EXECUTION_MEMORY_VALUES];
  double seconds;
  PHX_CHECK(phx_mc_execution_end(execution, work, memory, &seconds) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(memory[1] == metadata && memory[6] == 0 && memory[7] == bound);
  std::thread release([&]() { phx_mc_execution_release_host_storage(reservation); });
  release.join();
  PHX_CHECK(owner->live_bytes() == 0 && owner->available_bytes() == 1024);
  phx_mc_execution_release_host_storage(nullptr);
}

void test_external_host_reservation_resize_refusal_preserves_original_token() {
  auto owner = std::make_shared<BoundedMemoryResource>(1024);
  MemoryScope memory_scope(owner);
  void* execution = nullptr;
  PHX_CHECK(phx_mc_execution_begin(UINT64_MAX, UINT64_MAX, UINT64_MAX,
      1024, INFINITY, nullptr, &execution) == PHX_MC_OK);
  void* reservation = nullptr;
  PHX_CHECK(phx_mc_execution_reserve_host_storage(execution, 700, &reservation) == PHX_MC_OK);
  void* token = reservation;
  const auto available = owner->available_bytes();
  const auto allocated = owner->evidence();
  PHX_CHECK(phx_mc_execution_reserve_host_storage(execution, 1024, &reservation) ==
            PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(reservation == token && owner->available_bytes() == available);
  PHX_CHECK(owner->evidence() == allocated);
  PHX_CHECK(phx_mc_execution_reserve_host_storage(execution, 600, &reservation) == PHX_MC_OK);
  uint64_t work[PHX_MC_EXECUTION_COUNTERS], memory[PHX_MC_EXECUTION_MEMORY_VALUES];
  double seconds;
  PHX_CHECK(phx_mc_execution_end(execution, work, memory, &seconds) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(memory[6] == 600 && memory[7] == 700);
  phx_mc_execution_release_host_storage(reservation);
  PHX_CHECK(owner->available_bytes() == 1024 && owner->live_bytes() == 0);
}

void test_split_allocation_refusal_restores_cells_constraints_and_coordinates() {
  const auto domain = test::box_domain(test::box_points(3, 2, 11), 0.0);
  auto* handle = test::create_mesh(domain, PHX_MC_TET_MESH_BOUNDARY_CONFORMING);
  PHX_CHECK(handle != nullptr);
  const auto before = test::export_mesh(handle);
  auto& mesh = *handle->mesh;
  int32_t a = domain.segments[0], b = domain.segments[1];
  double point[3], parameter = 0.0;
  PHX_CHECK(mesh.construct_edge_point(a, b, point, &parameter));
  PHX_CHECK(mesh.memory_owner()->set_limit(mesh.memory_owner()->live_bytes()));
  int32_t inserted = -1;
  const auto status = mesh.split_edge(a, b, point, 0.0, inserted);
  PHX_CHECK(status == Insertion::kCapacity && inserted == -1);
  PHX_CHECK(!mesh.broken() && mesh.complex().audit());
  PHX_CHECK(mesh.memory_owner()->set_limit(SIZE_MAX));
  PHX_CHECK(test::export_mesh(handle) == before);
  PHX_CHECK(mesh.split_edge(a, b, point, 0.0, inserted) == Insertion::kOk);
  PHX_CHECK(mesh.complex().audit());
  phx_mc_tet_mesh_free(handle);
}

test::Domain single_tetrahedron() {
  test::Domain domain;
  domain.points = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  domain.tets = {0, 1, 2, 3};
  if (orient3d(domain.points.data(), domain.points.data() + 3,
                domain.points.data() + 6, domain.points.data() + 9) < 0) {
    std::swap(domain.tets[0], domain.tets[1]);
  }
  domain.regions = {0};
  domain.faces = {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
  domain.face_sources = {0, 1, 2, 3};
  domain.segments = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};
  domain.segment_sources = {0, 1, 2, 3, 4, 5};
  return domain;
}

NativeUniquePtr<phx_mc_tet_mesh> build_fault_fixture(const test::Domain& domain) {
  auto handle = make_native_unique<phx_mc_tet_mesh>();
  handle->mesh = make_native_unique<TetMesh>(BoundaryPolicy::kConforming, 16, 64);
  PHX_CHECK(handle->mesh->build(
                domain.points.size() / 3, domain.points.data(), domain.regions.size(),
                domain.tets.data(), domain.regions.data(), domain.face_sources.size(),
                domain.faces.data(), domain.face_sources.data(), domain.segment_sources.size(),
                domain.segments.data(), domain.segment_sources.data(), nullptr) == PHX_MC_OK);
  return handle;
}

void test_every_split_allocation_failure_rolls_back_retired_source_marks() {
  const auto domain = single_tetrahedron();
  const double position[3] = {0.5, 0, 0};
  std::size_t operation_allocations;
  {
    auto owner = std::make_shared<BoundedMemoryResource>();
    MemoryScope scope(owner);
    auto handle = build_fault_fixture(domain);
    const std::size_t start = owner->allocations();
    int32_t inserted = -1;
    PHX_CHECK(handle->mesh->split_edge(0, 1, position, 0, inserted) == Insertion::kOk);
    operation_allocations = owner->allocations() - start;
  }
  // Fault every real allocation boundary, including validation after original
  // facet marks have been retired. No guessed byte sizes or phase percentages.
  for (std::size_t fault = 1; fault <= operation_allocations; ++fault) {
    RefusingResource upstream;
    auto owner = std::make_shared<BoundedMemoryResource>(SIZE_MAX, &upstream);
    MemoryScope scope(owner);
    auto handle = build_fault_fixture(domain);
    const auto before = test::export_mesh(handle.get());
    upstream.refuse_at = upstream.requests + fault;
    int32_t inserted = -1;
    PHX_CHECK(handle->mesh->split_edge(0, 1, position, 0, inserted) == Insertion::kCapacity);
    upstream.refuse_at = 0;
    PHX_CHECK(inserted == -1 && !handle->mesh->broken());
    PHX_CHECK(handle->mesh->complex().audit());
    PHX_CHECK(test::export_mesh(handle.get()) == before);
    PHX_CHECK(owner->refusals() == 1);
    PHX_CHECK(handle->mesh->split_edge(0, 1, position, 0, inserted) == Insertion::kOk);
    PHX_CHECK(handle->mesh->complex().audit());
  }
}
void test_envelope_refusals_share_original_work_queries_and_memory() {
  const double points[] = {0.,0.,0., .2,0.,0., 0.,.2,0.};
  const int64_t faces[] = {0,1,2};
  const std::array<std::array<uint64_t,3>,3> limits = {{
      {0,UINT64_MAX,UINT64_MAX},
      {UINT64_MAX,0,UINT64_MAX},
      {UINT64_MAX,UINT64_MAX,1},
  }};
  for (const auto& limit : limits) {
    auto owner = std::make_shared<BoundedMemoryResource>();
    MemoryScope memory_scope(owner);
    void* execution = nullptr;
    PHX_CHECK(phx_mc_execution_begin(limit[0],limit[1],UINT64_MAX,
        limit[2],INFINITY,nullptr,&execution) == PHX_MC_OK);
    void* envelope = nullptr;
    int64_t counts[7];
    double bounds[4];
    PHX_CHECK(phx_mc_surface_envelope_create(points,3,faces,1,
        .263,.11,.05,100000,10000000,10000,20000,20000,100000,
        &envelope,counts,bounds) == PHX_MC_CAPACITY_EXCEEDED);
    PHX_CHECK(envelope == nullptr);
    // Failed construction releases every partially built source/carrier owner.
    PHX_CHECK(owner->live_bytes() == 0);
    uint64_t work[PHX_MC_EXECUTION_COUNTERS];
    uint64_t memory[PHX_MC_EXECUTION_MEMORY_VALUES];
    double seconds;
    PHX_CHECK(phx_mc_execution_end(execution,work,memory,&seconds) ==
              PHX_MC_CAPACITY_EXCEEDED);
    if (limit[0] == 0) PHX_CHECK(work[3] > 0 && work[0] == 0);
    if (limit[1] == 0) PHX_CHECK(work[4] > 0 && work[1] == 0 && work[0] > 0);
    if (limit[2] == 1) PHX_CHECK(memory[5] > 0 && memory[1] == 0);
  }
}

void test_actual_preparation_import_preserves_raw_clock_and_parent_allowance() {
  void* source = nullptr;
  PHX_CHECK(phx_mc_execution_begin(3,2,10,1000000,10.,nullptr,&source) == PHX_MC_OK);
  PHX_CHECK(phx_mc_execution_charge(source,3,2) == PHX_MC_OK);
  uint64_t source_work[PHX_MC_EXECUTION_COUNTERS], source_memory[PHX_MC_EXECUTION_MEMORY_VALUES];
  double source_seconds = 0.;
  PHX_CHECK(phx_mc_execution_end(source,source_work,source_memory,&source_seconds) == PHX_MC_OK);
  PHX_CHECK(source_seconds > 0.);
  void* parent = nullptr;
  PHX_CHECK(phx_mc_execution_begin(5,3,10,1000000,10.,nullptr,&parent) == PHX_MC_OK);
  PHX_CHECK(phx_mc_execution_import_preparation(parent,source_work[0],source_work[1],
                                               source_seconds) == PHX_MC_OK);
  void* child = nullptr;
  PHX_CHECK(phx_mc_execution_begin(2,1,10,1000000,10.-source_seconds,nullptr,&child) == PHX_MC_OK);
  PHX_CHECK(phx_mc_execution_charge(child,2,1) == PHX_MC_OK);
  double prior = -1.;
  PHX_CHECK(phx_mc_execution_prior_seconds(child,&prior) == PHX_MC_OK && prior == 0.);
  uint64_t work[PHX_MC_EXECUTION_COUNTERS], memory[PHX_MC_EXECUTION_MEMORY_VALUES];
  double actual_seconds = 0.;
  PHX_CHECK(phx_mc_execution_end(child,work,memory,&actual_seconds) == PHX_MC_OK);
  PHX_CHECK(work[0] == 2 && work[1] == 1);
  PHX_CHECK(phx_mc_execution_prior_seconds(parent,&prior) == PHX_MC_OK && prior == source_seconds);
  PHX_CHECK(phx_mc_execution_end(parent,work,memory,&actual_seconds) == PHX_MC_OK);
  PHX_CHECK(work[0] == 5 && work[1] == 3 && work[6] == 5 && work[7] == 3);
  PHX_CHECK(actual_seconds >= 0.); // Raw wall duration was not rewritten as prior time.
}

void test_preparation_import_invalid_and_depleted_allowances_are_atomic() {
  void* scope = nullptr;
  PHX_CHECK(phx_mc_execution_begin(5,3,10,1000000,10.,nullptr,&scope) == PHX_MC_OK);
  for (double invalid : std::array<double,3>{-1., INFINITY, NAN}) {
    PHX_CHECK(phx_mc_execution_import_preparation(scope,3,2,invalid) == PHX_MC_INVALID_ARGUMENT);
  }
  double prior = -1.;
  PHX_CHECK(phx_mc_execution_prior_seconds(scope,&prior) == PHX_MC_OK && prior == 0.);
  PHX_CHECK(phx_mc_execution_charge(scope,5,3) == PHX_MC_OK);
  PHX_CHECK(phx_mc_execution_import_preparation(scope,1,0,.01) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(phx_mc_execution_prior_seconds(scope,&prior) == PHX_MC_OK && prior == 0.);
  uint64_t work[PHX_MC_EXECUTION_COUNTERS], memory[PHX_MC_EXECUTION_MEMORY_VALUES];
  double seconds = 0.;
  PHX_CHECK(phx_mc_execution_end(scope,work,memory,&seconds) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(work[0] == 5 && work[1] == 3);

  void* source = nullptr;
  PHX_CHECK(phx_mc_execution_begin(1,1,10,1000000,10.,nullptr,&source) == PHX_MC_OK);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  PHX_CHECK(phx_mc_execution_end(source,work,memory,&seconds) == PHX_MC_OK);
  PHX_CHECK(seconds > 0.);
  PHX_CHECK(phx_mc_execution_begin(5,3,10,1000000,seconds*.5,nullptr,&scope) == PHX_MC_OK);
  PHX_CHECK(phx_mc_execution_import_preparation(scope,0,0,seconds) == PHX_MC_TIMEOUT);
  PHX_CHECK(phx_mc_execution_prior_seconds(scope,&prior) == PHX_MC_OK && prior == 0.);
  PHX_CHECK(phx_mc_execution_end(scope,work,memory,&seconds) == PHX_MC_TIMEOUT);
  PHX_CHECK(work[0] == 0 && work[1] == 0);
}
}  // namespace

int main() {
  test_growth_counts_simultaneous_buffers_and_preserves_values();
  test_scopes_restore_and_spills_keep_allocator_lifetime();
  test_inline_expansion_needs_no_allocation_under_zero_limit();
  test_upstream_refusal_preserves_expansion_assignment();
  test_nested_window_shares_ledger_and_restores_parent_quota();
  test_host_array_and_native_buffers_share_actual_cap();
  test_host_payload_refusal_counts_only_successful_metadata();
  test_host_upstream_oom_does_not_commit_failed_request();
  test_host_array_zero_and_invalid_arguments_preserve_accounting();
  test_host_array_nested_scopes_keep_root_minimum();
  test_host_allocation_preserves_original_execution_refusal();
  test_host_owner_outlives_scope_and_releases_on_another_thread();
  test_host_concurrent_release_serializes_pool_operations_and_scope_exit();
  test_external_host_storage_shares_native_cap_and_releases_after_refusal();
  test_external_host_reservation_resize_refusal_preserves_original_token();
  test_split_allocation_refusal_restores_cells_constraints_and_coordinates();
  test_every_split_allocation_failure_rolls_back_retired_source_marks();
  test_envelope_refusals_share_original_work_queries_and_memory();
  test_actual_preparation_import_preserves_raw_clock_and_parent_allowance();
  test_preparation_import_invalid_and_depleted_allowances_are_atomic();
  return phx::mc::test::finish("test_bounded_memory");
}
