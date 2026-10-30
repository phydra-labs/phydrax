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
  PHX_CHECK(guarded([]() -> int32_t { throw std::bad_alloc(); }) == PHX_MC_ALLOCATION_FAILURE);
  PHX_CHECK(guarded([]() -> int32_t { throw std::bad_array_new_length(); }) ==
            PHX_MC_ALLOCATION_FAILURE);
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
  PHX_CHECK(phx_mc_tetrahedron_intersection_moments(
                int64_t{1} << 60, tetrahedron, tetrahedron, &measure, moment, &status) ==
            PHX_MC_INVALID_ARGUMENT);
  const double square[8] = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
  const int32_t corners = 4;
  PHX_CHECK(phx_mc_polygon_intersection_moments(huge, 4, square, &corners, 4, square, &corners,
                                                &measure, moment,
                                                &status) == PHX_MC_INVALID_ARGUMENT);
  const int32_t segment[2] = {0, 1};
  phx_mc_mesh* mesh = nullptr;
  PHX_CHECK(phx_mc_constrained_delaunay_2d(3, square, huge, segment, 0, nullptr, 1, 0.0,
                                           std::numeric_limits<double>::infinity(), &mesh) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(mesh == nullptr);
}

// Real allocation failures have their own status and leave the library usable
// once allocation succeeds again.
void test_allocation_failure_statuses() {
  // Collinear points take the symbolic-perturbation path, whose monomial
  // tables are allocated on first use.
  const double a[2] = {0.0, 0.0};
  const double b[2] = {1.0, 1.0};
  const double c[2] = {2.0, 2.0};
  const int64_t ids[3] = {0, 1, 2};
  int8_t sign = 9;
  PHX_CHECK(refused([&] { return phx_mc_orient2d_sos(1, a, b, c, ids, &sign); }) ==
            PHX_MC_ALLOCATION_FAILURE);
  PHX_CHECK(phx_mc_orient2d_sos(1, a, b, c, ids, &sign) == PHX_MC_OK);
  PHX_CHECK(sign == 1 || sign == -1);

  const double points[8] = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
  phx_mc_mesh* mesh = nullptr;
  PHX_CHECK(refused([&] { return phx_mc_delaunay_2d(4, points, &mesh); }) ==
            PHX_MC_ALLOCATION_FAILURE);
  PHX_CHECK(mesh == nullptr);
  PHX_CHECK(phx_mc_delaunay_2d(4, points, &mesh) == PHX_MC_OK);
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
            }) == PHX_MC_ALLOCATION_FAILURE);
  PHX_CHECK(phx_mc_polygon_intersection_moments(1, 4, points, &corners, 4, shifted, &corners,
                                                &area, moment, &item) == PHX_MC_OK);
  PHX_CHECK(item == PHX_MC_OK);
  PHX_CHECK_NEAR(area, 0.25, 1e-15);
}

void test_plc_source_prepare_c_api() {
  const double points[24] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0,
                             0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1};
  const int64_t polygon_offsets[7] = {0, 4, 8, 12, 16, 20, 24};
  const int32_t polygon_vertices[24] = {0, 2, 3, 1, 4, 5, 7, 6, 0, 1, 5, 4,
                                        2, 6, 7, 3, 0, 4, 6, 2, 1, 3, 7, 5};
  const int32_t polygon_facets[6] = {0, 1, 2, 3, 4, 5};
  const int32_t facet_regions[12] = {-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0};
  phx_mc_plc3d_result* result = nullptr;
  phx_mc_plc3d_prepare_source_request request{
      .point_count = 8,
      .points = points,
      .polygon_count = 6,
      .polygon_vertex_count = 24,
      .polygon_offsets = polygon_offsets,
      .polygon_vertices = polygon_vertices,
      .polygon_facets = polygon_facets,
      .facet_count = 6,
      .facet_regions = facet_regions,
      .segment_count = 0,
      .segments = nullptr,
      .seed_count = 0,
      .seeds = nullptr,
      .seed_regions = nullptr,
      .boundary_policy = PHX_MC_TET_MESH_BOUNDARY_CONFORMING,
      .facet_tolerances = nullptr,
      .segment_tolerances = nullptr,
      .result = &result,
  };
  PHX_CHECK(phx_mc_plc3d_prepare_source(&request) == PHX_MC_OK);
  PHX_CHECK(result != nullptr);
  if (result == nullptr) return;

  int64_t sizes[PHX_MC_PLC3D_SIZES] = {-1, -1, -1, -1, -1, -1};
  PHX_CHECK(phx_mc_plc3d_sizes(result, sizes) == PHX_MC_OK);
  PHX_CHECK(sizes[0] == 8 && sizes[1] == 0 && sizes[2] == 0 && sizes[3] == 0 &&
            sizes[4] == 12 && sizes[5] == 12);
  double output_points[24] = {};
  double protection[8] = {};
  int8_t dimensions[8] = {};
  int32_t plc_edges[24] = {};
  int32_t input_triangles[36] = {};
  int32_t input_polygons[12] = {};
  PHX_CHECK(phx_mc_plc3d_export(
                result, 8, 0, 0, 0, 12, 12, output_points, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, protection, dimensions, plc_edges,
                input_triangles, input_polygons) == PHX_MC_OK);
  PHX_CHECK(output_points[0] == 0.0 && output_points[23] == 1.0);
  PHX_CHECK(dimensions[0] == PHX_MC_SOURCE_STRATUM_NONE && protection[0] == 0.0);
  int8_t strata[8] = {};
  int32_t entities[8] = {};
  double parameters[16] = {};
  double deviations[8] = {};
  PHX_CHECK(phx_mc_plc3d_source_witnesses(result, 8, strata, entities, parameters,
                                          deviations) == PHX_MC_OK);
  for (int vertex = 0; vertex < 8; ++vertex) {
    PHX_CHECK(strata[vertex] == PHX_MC_SOURCE_STRATUM_NONE);
    PHX_CHECK(entities[vertex] == -1 && deviations[vertex] == 0.0);
  }
  int32_t reason = -1, first_kind = -1, second_kind = -1;
  int64_t first = -2, second = -2;
  double certified = -1.0, bound = -1.0;
  PHX_CHECK(phx_mc_plc3d_failure(result, &reason, &first_kind, &first, &second_kind,
                                 &second, &certified, &bound) == PHX_MC_OK);
  PHX_CHECK(reason == PHX_MC_PLC_NO_FAILURE && first_kind == PHX_MC_PLC_ENTITY_NONE &&
            first == -1 && second_kind == PHX_MC_PLC_ENTITY_NONE && second == -1 &&
            certified == 0.0 && bound == 0.0);
  phx_mc_plc3d_free(result);
  phx_mc_plc3d_result* recovered = nullptr;
  request.result = &recovered;
  PHX_CHECK(phx_mc_plc3d_recover(&request) == PHX_MC_OK);
  PHX_CHECK(recovered != nullptr);
  if (recovered != nullptr) {
    PHX_CHECK(phx_mc_plc3d_sizes(recovered, sizes) == PHX_MC_OK);
    PHX_CHECK(sizes[0] == 8 && sizes[1] > 0 && sizes[2] > 0);
    phx_mc_plc3d_free(recovered);
  }
}

struct ParallelCapiState {
  int32_t values[9]{};
  std::atomic<int32_t> calls{0};
};

int32_t fill_parallel_chunk(void* context, int64_t begin, int64_t end, int64_t) {
  auto& state = *static_cast<ParallelCapiState*>(context);
  state.calls.fetch_add(1, std::memory_order_relaxed);
  for (int64_t row = begin; row < end; ++row) {
    state.values[row] = static_cast<int32_t>(row + 1);
  }
  return PHX_MC_OK;
}

int32_t status_parallel_chunk(void* context, int64_t, int64_t, int64_t chunk_index) {
  auto& calls = *static_cast<std::atomic<int32_t>*>(context);
  calls.fetch_add(1, std::memory_order_relaxed);
  return chunk_index == 0 ? PHX_MC_RANGE_ERROR : PHX_MC_UNRESOLVED;
}

void test_parallel_c_api() {
  ParallelCapiState state;
  PHX_CHECK(phx_mc_parallel_for(9, 2, 3, fill_parallel_chunk, &state) == PHX_MC_OK);
  PHX_CHECK(state.calls.load() == 5);
  for (int32_t row = 0; row < 9; ++row) PHX_CHECK(state.values[row] == row + 1);
  state.calls.store(0);
  PHX_CHECK(phx_mc_parallel_for(0, 1, 2, fill_parallel_chunk, &state) == PHX_MC_OK);
  PHX_CHECK(state.calls.load() == 0);

  std::atomic<int32_t> status_calls{0};
  PHX_CHECK(phx_mc_parallel_for(4, 1, 2, status_parallel_chunk, &status_calls) ==
            PHX_MC_RANGE_ERROR);
  PHX_CHECK(status_calls.load() == 4);
  PHX_CHECK(phx_mc_parallel_for(-1, 1, 1, fill_parallel_chunk, &state) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_parallel_for(1, 0, 1, fill_parallel_chunk, &state) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_parallel_for(1, 1, 1, nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
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
  test_allocation_failure_statuses();
  test_plc_source_prepare_c_api();
  test_parallel_c_api();
  return phx::mc::test::finish("test_capi_guard");
}
