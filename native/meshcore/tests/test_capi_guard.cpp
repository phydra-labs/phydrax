//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Failure boundary of the C ABI: refused allocations, escaped exceptions, and
// unaddressable counts become call statuses and never cross extern "C".
#include <atomic>
#include <cmath>
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

void test_null_requests_are_invalid_arguments() {
  PHX_CHECK(phx_mc_clip_reference_tetrahedron_exact(nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_clip_box_halfspaces(nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_constrained_delaunay_2d(nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_surface_reconnect(nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_tet_mesh_create(nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_plc3d_recover(nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_restricted_power_cells(nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_restricted_power_cells_exact(nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_polyhedron_corner_hexes(nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_level_set_split_3d(nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_surface_envelope_create(nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_tet_mesh_commit_edge_insertion(nullptr) == PHX_MC_INVALID_ARGUMENT);
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
  const phx_mc_constrained_delaunay_2d_request request{
      .point_count = 3,
      .points = square,
      .segment_count = huge,
      .segments = segment,
      .hole_count = 0,
      .holes = nullptr,
      .keep_convex_hull = 1,
      .min_angle_degrees = 0.0,
      .max_area = std::numeric_limits<double>::infinity(),
      .max_steiner = 0,
      .max_triangles = 100,
      .max_cavity_cells = INT64_MAX,
      .max_work = INT64_MAX,
      .max_scratch_bytes = INT64_MAX,
      .work_evidence = nullptr,
      .memory_evidence = nullptr,
      .witness_capacity = 0,
      .witness_parents = nullptr,
      .witness_segments = nullptr,
      .witness_ratios = nullptr,
      .mesh = &mesh,
  };
  PHX_CHECK(phx_mc_constrained_delaunay_2d(&request) ==
            PHX_MC_INVALID_ARGUMENT);
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
    PHX_CHECK(memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0);
    if (limit.counter >= 0) PHX_CHECK(counters[limit.counter] > 0);
    if (limit.counter == 3 || limit.counter == -2) PHX_CHECK(memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] == 0);
    if (limit.counter == -1) {
      PHX_CHECK(memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] == 0 && memory[PHX_MC_EXECUTION_MEMORY_REQUESTED_BYTES] > 0 && memory[PHX_MC_EXECUTION_MEMORY_REFUSALS] > 0);
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
  PHX_CHECK(work[PHX_MC_EXECUTION_SOURCE_QUERIES] == 1);
  PHX_CHECK(phx_mc_execution_end(outer, work, memory, &seconds) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(work[PHX_MC_EXECUTION_SOURCE_QUERIES] == 1 && work[PHX_MC_EXECUTION_QUERY_REFUSALS] == 1 && work[PHX_MC_EXECUTION_EXTERNAL_QUERIES] == 1 && work[PHX_MC_EXECUTION_PRIMITIVE_QUERIES] == 1);
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
  PHX_CHECK(work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 2 && work[PHX_MC_EXECUTION_EXTERNAL_WORK] == 2 && work[PHX_MC_EXECUTION_WORK_REFUSALS] == 1);
}

void test_batch_handle_allocation_refusal_leaks_no_reservations() {
  void* scope = nullptr;
  PHX_CHECK(phx_mc_execution_begin(10, 10, 10, 10, INFINITY, nullptr, &scope) == PHX_MC_OK);
  void* admitted = nullptr;
  PHX_CHECK(phx_mc_execution_batch_admit(scope, nullptr, 6, 6, 6, 6, &admitted) == PHX_MC_OK);
  void* ticket = admitted;
  PHX_CHECK(refused([&] {
              return phx_mc_execution_batch_admit(scope, nullptr, 1, 1, 1, 1, &ticket);
            }) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(ticket == nullptr);
  uint64_t remaining[4] = {};
  double seconds = 0;
  PHX_CHECK(phx_mc_execution_remaining(scope, remaining, &seconds) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(remaining[0] == 4 && remaining[1] == 4 && remaining[2] == 4 && remaining[3] == 4);
  uint64_t refusal[PHX_MC_EXECUTION_BATCH_REFUSAL_VALUES] = {};
  PHX_CHECK(phx_mc_execution_batch_refusal(scope, refusal) == PHX_MC_OK);
  PHX_CHECK(refusal[PHX_MC_BATCH_REMAINING_WORK] == 4 &&
            refusal[PHX_MC_BATCH_REMAINING_BYTES] == 4);
  PHX_CHECK(phx_mc_execution_batch_settle(admitted, 2, 1, 3, 4) == PHX_MC_OK);
  uint64_t work[PHX_MC_EXECUTION_COUNTERS] = {}, memory[PHX_MC_EXECUTION_MEMORY_VALUES] = {};
  PHX_CHECK(phx_mc_execution_end(scope, work, memory, &seconds) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(work[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 2 && work[PHX_MC_EXECUTION_SOURCE_QUERIES] == 1 && work[PHX_MC_EXECUTION_EXTERNAL_WORK] == 2 && work[PHX_MC_EXECUTION_EXTERNAL_QUERIES] == 1);
  PHX_CHECK(memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0 && memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] == 0 && memory[PHX_MC_EXECUTION_MEMORY_EXTERNAL_PEAK_UPPER_BYTES] == 4);
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
  test_null_requests_are_invalid_arguments();
  test_addressable_extents();
  test_unaddressable_counts_are_invalid_arguments();
  test_refused_allocations_are_capacity_statuses();
  test_native_phase_limits_refuse_before_publication();
  test_nested_phase_counters_share_original_allowance();
  test_pure_batch_bound_is_not_reported_as_consumed_work();
  test_batch_handle_allocation_refusal_leaks_no_reservations();
  return phx::mc::test::finish("test_capi_guard");
}
