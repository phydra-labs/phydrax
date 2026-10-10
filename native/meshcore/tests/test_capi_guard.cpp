//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Failure boundary of the C ABI: refused allocations, escaped exceptions, and
// unaddressable counts become call statuses and never cross extern "C".
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>

#include "capi_guard.hpp"
#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

// While set, every global allocation of this process is refused.
std::atomic<bool> refuse_allocations{false};
std::atomic<long> refused_allocations{0};

void* allocate(std::size_t size) {
  if (refuse_allocations.load(std::memory_order_relaxed)) {
    refused_allocations.fetch_add(1, std::memory_order_relaxed);
    throw std::bad_alloc();
  }
  if (void* memory = std::malloc(size == 0 ? 1 : size)) {
    return memory;
  }
  throw std::bad_alloc();
}

template <class Call>
int32_t refused(Call&& call) {
  refused_allocations.store(0);
  refuse_allocations.store(true);
  const int32_t status = call();
  refuse_allocations.store(false);
  PHX_CHECK(refused_allocations.load() > 0);
  return status;
}

void test_guard_maps_escaped_exceptions() {
  using phx::mc::guarded;
  static_assert(noexcept(guarded([]() -> int32_t { return PHX_MC_OK; })));
  PHX_CHECK(guarded([]() -> int32_t { return PHX_MC_RANGE_ERROR; }) == PHX_MC_RANGE_ERROR);
  PHX_CHECK(guarded([]() -> int32_t { throw std::bad_alloc(); }) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(guarded([]() -> int32_t { throw std::bad_array_new_length(); }) ==
            PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(guarded([]() -> int32_t { throw std::invalid_argument("rejected"); }) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(guarded([]() -> int32_t { throw std::logic_error("broken"); }) ==
            PHX_MC_INTERNAL_ERROR);
  PHX_CHECK(guarded([]() -> int32_t { throw 7; }) == PHX_MC_INTERNAL_ERROR);
}

void test_addressable_extents() {
  using phx::mc::addressable;
  const int64_t limit = std::numeric_limits<int64_t>::max();
  PHX_CHECK(addressable(0, 0, sizeof(double)));
  PHX_CHECK(addressable(limit, 0, sizeof(double)));
  PHX_CHECK(addressable(1000, 3, sizeof(double)));
  PHX_CHECK(!addressable(-1, 3, sizeof(double)));
  PHX_CHECK(!addressable(1, -1, sizeof(double)));
  // count * width fits int64 but its byte size does not.
  PHX_CHECK(!addressable(int64_t{1} << 60, 3, sizeof(double)));
  PHX_CHECK(!addressable(limit / 2, 2, sizeof(double)));
}

// Counts whose row offsets overflow are refused before any array is read; the
// arrays below hold a single row.
void test_unaddressable_counts_are_invalid_arguments() {
  const int64_t huge = std::numeric_limits<int64_t>::max() / 2;
  const double p[3] = {0.0, 0.0, 0.0};
  const int64_t ids[5] = {0, 1, 2, 3, 4};
  int8_t signs[1] = {9};
  PHX_CHECK(phx_mc_orient2d(huge, p, p, p, signs) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_orient3d(int64_t{1} << 60, p, p, p, p, signs) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_incircle(huge, p, p, p, p, signs) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_insphere(huge, p, p, p, p, p, signs) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_orient2d_sos(huge, p, p, p, ids, signs) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_insphere_sos(huge, p, p, p, p, p, ids, signs) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(signs[0] == 9);

  const double tetrahedron[12] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  double measure = 0.0;
  double moment[3] = {0.0, 0.0, 0.0};
  int32_t status = PHX_MC_OK;
  PHX_CHECK(phx_mc_tetrahedron_intersection_moments(int64_t{1} << 60, tetrahedron, tetrahedron,
                                                    64, &measure, moment,
                                                    &status) == PHX_MC_INVALID_ARGUMENT);
  const double square[8] = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
  const int32_t corners = 4;
  PHX_CHECK(phx_mc_polygon_intersection_moments(huge, 4, square, &corners, 4, square, &corners,
                                                &measure, moment,
                                                &status) == PHX_MC_INVALID_ARGUMENT);
  const int32_t segment[2] = {0, 1};
  phx_mc_mesh* mesh = nullptr;
  PHX_CHECK(phx_mc_constrained_delaunay_2d(3, square, huge, segment, 0, nullptr, 1, 0.0,
                                           std::numeric_limits<double>::infinity(), 0, 100,
                                           INT64_MAX, INT64_MAX, INT64_MAX, nullptr, nullptr,
                                           &mesh) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(mesh == nullptr);
}

// Refused allocations surface as CAPACITY_EXCEEDED from every kernel family and
// leave the library usable once allocation succeeds again.
void test_refused_allocations_are_capacity_statuses() {
  const double points[8] = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
  phx_mc_mesh* mesh = nullptr;
  PHX_CHECK(refused([&] {
              return phx_mc_delaunay_2d(4, points, 100, INT64_MAX, INT64_MAX, INT64_MAX,
                                         nullptr, nullptr, &mesh);
            }) ==
            PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(mesh == nullptr);
  PHX_CHECK(phx_mc_delaunay_2d(4, points, 100, INT64_MAX, INT64_MAX, INT64_MAX,
                               nullptr, nullptr, &mesh) == PHX_MC_OK);
  PHX_CHECK(mesh != nullptr && phx_mc_mesh_cell_count(mesh) == 2);
  phx_mc_mesh_free(mesh);

  const double shifted[8] = {0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5};
  const int32_t corners = 4;
  double area = -1.0;
  double moment[2] = {0.0, 0.0};
  int32_t item = -1;
  PHX_CHECK(refused([&] {
              return phx_mc_polygon_intersection_moments(1, 4, points, &corners, 4, shifted,
                                                         &corners, &area, moment, &item);
            }) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(phx_mc_polygon_intersection_moments(1, 4, points, &corners, 4, shifted, &corners,
                                                &area, moment, &item) == PHX_MC_OK);
  PHX_CHECK(item == PHX_MC_OK);
  PHX_CHECK_NEAR(area, 0.25, 1e-15);
}

void test_native_phase_limits_refuse_before_publication() {
  const double points[15] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.125, 0.125, 0.125};
  const uint64_t unlimited = std::numeric_limits<uint64_t>::max();
  struct Limits {
    uint64_t work, queries, cavity, scratch;
    double seconds;
    int32_t status;
    int counter;
  };
  const Limits limits[] = {
      {0, unlimited, unlimited, unlimited, 10, PHX_MC_CAPACITY_EXCEEDED, 3},
      {unlimited, unlimited, 0, unlimited, 10, PHX_MC_CAPACITY_EXCEEDED, 5},
      {unlimited, unlimited, unlimited, 0, 10, PHX_MC_CAPACITY_EXCEEDED, -1},
      {unlimited, unlimited, unlimited, unlimited, 0, PHX_MC_TIMEOUT, -2},
  };
  for (const auto& limit : limits) {
    void* scope = nullptr;
    PHX_CHECK(phx_mc_execution_begin(limit.work, limit.queries, limit.cavity, limit.scratch,
                                     limit.seconds, nullptr, &scope) == PHX_MC_OK);
    phx_mc_mesh* mesh = nullptr;
    PHX_CHECK(phx_mc_delaunay_3d(5, points, 100, &mesh) == limit.status);
    PHX_CHECK(mesh == nullptr);
    uint64_t counters[PHX_MC_EXECUTION_COUNTERS] = {}, memory[PHX_MC_EXECUTION_MEMORY_VALUES] = {};
    double seconds = -1;
    PHX_CHECK(phx_mc_execution_end(scope, counters, memory, &seconds) == limit.status);
    PHX_CHECK(seconds >= 0);
    PHX_CHECK(memory[1] == 0);
    if (limit.counter >= 0) PHX_CHECK(counters[limit.counter] > 0);
    if (limit.counter == 3 || limit.counter == -2) PHX_CHECK(memory[2] == 0);
    if (limit.counter == -1) {
      PHX_CHECK(memory[2] == 0 && memory[3] > 0 && memory[5] > 0);
    }
  }
  phx_mc_mesh* mesh = nullptr;
  PHX_CHECK(phx_mc_delaunay_3d(5, points, 100, &mesh) == PHX_MC_OK);
  PHX_CHECK(mesh != nullptr && phx_mc_mesh_cell_count(mesh) == 4);
  phx_mc_mesh_free(mesh);
}

void test_nested_phase_counters_share_original_allowance() {
  const uint64_t unlimited = std::numeric_limits<uint64_t>::max();
  void *outer = nullptr, *inner = nullptr;
  PHX_CHECK(phx_mc_execution_begin(unlimited, 1, unlimited, unlimited, 10,
                                   nullptr, &outer) == PHX_MC_OK);
  PHX_CHECK(phx_mc_execution_begin(unlimited, unlimited, unlimited, unlimited, 10,
                                   nullptr, &inner) == PHX_MC_OK);
  const double a[3] = {0, 0, 0}, b[3] = {1, 0, 0}, c[3] = {0, 1, 0}, d[3] = {0, 0, 1};
  int8_t sign = 0;
  PHX_CHECK(phx_mc_execution_charge(inner, 0, 1) == PHX_MC_OK);
  PHX_CHECK(phx_mc_orient3d(1, a, b, c, d, &sign) == PHX_MC_OK && sign == 1);
  PHX_CHECK(phx_mc_execution_charge(inner, 0, 1) == PHX_MC_CAPACITY_EXCEEDED);
  uint64_t work[PHX_MC_EXECUTION_COUNTERS] = {}, memory[PHX_MC_EXECUTION_MEMORY_VALUES] = {};
  double seconds = 0;
  PHX_CHECK(phx_mc_execution_end(inner, work, memory, &seconds) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(work[1] == 1);
  PHX_CHECK(phx_mc_execution_end(outer, work, memory, &seconds) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(work[1] == 1 && work[4] == 1 && work[7] == 1 && work[8] == 1);
}

void test_pure_batch_bound_is_not_reported_as_consumed_work() {
  const uint64_t unlimited = std::numeric_limits<uint64_t>::max();
  void* scope = nullptr;
  PHX_CHECK(phx_mc_execution_begin(3, unlimited, unlimited, unlimited, 10,
                                   nullptr, &scope) == PHX_MC_OK);
  PHX_CHECK(phx_mc_execution_admit_work_bound(scope, 3) == PHX_MC_OK);
  PHX_CHECK(phx_mc_execution_charge(scope, 2, 0) == PHX_MC_OK);
  PHX_CHECK(phx_mc_execution_admit_work_bound(scope, 2) == PHX_MC_CAPACITY_EXCEEDED);
  uint64_t work[PHX_MC_EXECUTION_COUNTERS] = {}, memory[PHX_MC_EXECUTION_MEMORY_VALUES] = {};
  double seconds = 0;
  PHX_CHECK(phx_mc_execution_end(scope, work, memory, &seconds) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(work[0] == 2 && work[6] == 2 && work[3] == 1);
}

}  // namespace

void* operator new(std::size_t size) { return allocate(size); }
void* operator new[](std::size_t size) { return allocate(size); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

int main() {
  test_guard_maps_escaped_exceptions();
  test_addressable_extents();
  test_unaddressable_counts_are_invalid_arguments();
  test_refused_allocations_are_capacity_statuses();
  test_native_phase_limits_refuse_before_publication();
  test_nested_phase_counters_share_original_allowance();
  test_pure_batch_bound_is_not_reported_as_consumed_work();
  return phx::mc::test::finish("test_capi_guard");
}
