//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// C ABI: library identity, batched predicates, and mesh handle accessors.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>

#include "capi_guard.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "plc3d.hpp"
#include "parallel.hpp"

struct phx_mc_plc3d_result {
  phx::mc::PlcRecovery recovery;
};
#ifndef PHX_MC_VERSION
#error "PHX_MC_VERSION must be defined by the build."
#endif
#ifndef PHX_MC_BUILD_HASH
#error "PHX_MC_BUILD_HASH must be defined by the build."
#endif

namespace {

using phx::mc::addressable;
using phx::mc::coordinate_in_domain;
using phx::mc::guarded;

int32_t validate_coordinates(int64_t count, int width, const double* const* arrays, int arity) {
  for (int k = 0; k < arity; ++k) {
    for (int64_t index = 0; index < count * width; ++index) {
      if (!std::isfinite(arrays[k][index])) {
        return PHX_MC_NONFINITE_INPUT;
      }
    }
  }
  for (int k = 0; k < arity; ++k) {
    for (int64_t index = 0; index < count * width; ++index) {
      if (!coordinate_in_domain(arrays[k][index])) {
        return PHX_MC_RANGE_ERROR;
      }
    }
  }
  return PHX_MC_OK;
}

// Rejects a negative or unaddressable count before any `count * width` offset.
int32_t validate_batch(int64_t count, int width, const double* const* arrays, int arity,
                       const void* output) {
  if (count < 0 || output == nullptr || !addressable(count, width, sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  for (int k = 0; k < arity; ++k) {
    if (count > 0 && arrays[k] == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
  }
  return validate_coordinates(count, width, arrays, arity);
}

int32_t validate_ids(int64_t count, int arity, const int64_t* ids) {
  if ((count > 0 && ids == nullptr) || !addressable(count, arity, sizeof(int64_t))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  for (int64_t row = 0; row < count; ++row) {
    const int64_t* r = ids + row * arity;
    for (int i = 0; i < arity; ++i) {
      for (int j = i + 1; j < arity; ++j) {
        if (r[i] == r[j]) {
          return PHX_MC_INVALID_INPUT;
        }
      }
    }
  }
  return PHX_MC_OK;
}

int32_t validate_plc3d_prepare_request(const phx_mc_plc3d_prepare_source_request& request) {
  if (request.result == nullptr || request.point_count < 0 || request.polygon_count < 0 ||
      request.polygon_vertex_count < 0 || request.facet_count < 0 || request.segment_count < 0 ||
      request.seed_count < 0) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int64_t index_limit = std::numeric_limits<int32_t>::max();
  if (request.point_count >= index_limit || request.polygon_count > index_limit ||
      request.facet_count > index_limit || request.segment_count > index_limit) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  if (request.polygon_count == std::numeric_limits<int64_t>::max() ||
      !addressable(request.point_count, 3, sizeof(double)) ||
      !addressable(request.polygon_count + 1, 1, sizeof(int64_t)) ||
      !addressable(request.polygon_vertex_count, 1, sizeof(int32_t)) ||
      !addressable(request.polygon_count, 1, sizeof(int32_t)) ||
      !addressable(request.facet_count, 2, sizeof(int32_t)) ||
      !addressable(request.segment_count, 2, sizeof(int32_t)) ||
      !addressable(request.seed_count, 3, sizeof(double)) ||
      !addressable(request.seed_count, 1, sizeof(int32_t)) ||
      !addressable(request.facet_count, 1, sizeof(double)) ||
      !addressable(request.segment_count, 1, sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (request.points == nullptr || request.polygon_offsets == nullptr ||
      request.polygon_facets == nullptr || request.facet_regions == nullptr ||
      (request.polygon_vertex_count > 0 && request.polygon_vertices == nullptr) ||
      (request.segment_count > 0 && request.segments == nullptr) ||
      (request.seed_count > 0 &&
       (request.seeds == nullptr || request.seed_regions == nullptr)) ||
      (request.boundary_policy != PHX_MC_TET_MESH_BOUNDARY_FIXED &&
       request.boundary_policy != PHX_MC_TET_MESH_BOUNDARY_CONFORMING)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (request.polygon_offsets[0] != 0 ||
      request.polygon_offsets[request.polygon_count] != request.polygon_vertex_count) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  for (int64_t polygon = 0; polygon < request.polygon_count; ++polygon) {
    const int64_t offset = request.polygon_offsets[polygon + 1];
    if (offset < 0 || offset > request.polygon_vertex_count) return PHX_MC_INVALID_ARGUMENT;
  }
  return PHX_MC_OK;
}

using PlcBuilder = int32_t (*)(const phx::mc::PlcInput&, phx::mc::PlcRecovery&);

int32_t run_plc3d_request(const phx_mc_plc3d_prepare_source_request* request,
                          PlcBuilder builder) {
  if (request == nullptr) return PHX_MC_INVALID_ARGUMENT;
  if (request->result != nullptr) *request->result = nullptr;
  return guarded([&]() -> int32_t {
    const int32_t valid = validate_plc3d_prepare_request(*request);
    if (valid != PHX_MC_OK) return valid;
    phx::mc::PlcInput input;
    input.point_count = request->point_count;
    input.points = request->points;
    input.polygon_count = request->polygon_count;
    input.polygon_offsets = request->polygon_offsets;
    input.polygon_vertices = request->polygon_vertices;
    input.polygon_facets = request->polygon_facets;
    input.facet_count = request->facet_count;
    input.facet_regions = request->facet_regions;
    input.segment_count = request->segment_count;
    input.segments = request->segments;
    input.seed_count = request->seed_count;
    input.seeds = request->seeds;
    input.seed_regions = request->seed_regions;
    input.policy = static_cast<phx::mc::BoundaryPolicy>(request->boundary_policy);
    input.facet_tolerances = request->facet_tolerances;
    input.segment_tolerances = request->segment_tolerances;
    auto result = std::make_unique<phx_mc_plc3d_result>();
    const int32_t status = builder(input, result->recovery);
    if (status == PHX_MC_OK ||
        result->recovery.failure.reason != phx::mc::kPlcNoFailure) {
      *request->result = result.release();
    }
    return status;
  });
}

template <class T>
bool plc_output_bank_valid(const std::vector<T>& bank, const T* output) {
  return bank.empty() || output != nullptr;
}

template <class T>
void copy_plc_output_bank(const std::vector<T>& bank, T* output) {
  if (!bank.empty()) std::copy(bank.begin(), bank.end(), output);
}

}  // namespace

extern "C" {

const char* phx_mc_version(void) { return PHX_MC_VERSION; }

const char* phx_mc_build_hash(void) { return PHX_MC_BUILD_HASH; }

void phx_mc_exact_domain(int32_t* min_exponent, int32_t* max_exponent) {
  if (min_exponent != nullptr) {
    *min_exponent = phx::mc::kCoordinateMinExponent;
  }
  if (max_exponent != nullptr) {
    *max_exponent = phx::mc::kCoordinateMaxExponent;
  }
}
int32_t phx_mc_parallel_for(int64_t count, int64_t chunk_size, int32_t parallelism,
                            phx_mc_parallel_chunk_callback callback, void* context) {
  return guarded([&]() -> int32_t {
    if (count < 0 || chunk_size <= 0 || parallelism <= 0 || callback == nullptr ||
        static_cast<uint64_t>(count) > std::numeric_limits<std::size_t>::max() ||
        static_cast<uint64_t>(chunk_size) > std::numeric_limits<std::size_t>::max()) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const phx::mc::ParallelRequest request{static_cast<std::size_t>(chunk_size),
                                           static_cast<uint32_t>(parallelism)};
    return phx::mc::parallel_for(
        static_cast<std::size_t>(count), request,
        [callback, context](std::size_t begin, std::size_t end,
                            phx::mc::ChunkContext& chunk) -> int32_t {
          return callback(context, static_cast<int64_t>(begin), static_cast<int64_t>(end),
                          static_cast<int64_t>(chunk.chunk_index));
        });
  });
}

int32_t phx_mc_orient2d(int64_t count, const double* a, const double* b, const double* c,
                        int8_t* signs) {
  return guarded([&]() -> int32_t {
    const double* arrays[3] = {a, b, c};
    const int32_t status = validate_batch(count, 2, arrays, 3, signs);
    if (status != PHX_MC_OK) {
      return status;
    }
    for (int64_t i = 0; i < count; ++i) {
      signs[i] = static_cast<int8_t>(phx::mc::orient2d(a + 2 * i, b + 2 * i, c + 2 * i));
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_orient3d(int64_t count, const double* a, const double* b, const double* c,
                        const double* d, int8_t* signs) {
  return guarded([&]() -> int32_t {
    const double* arrays[4] = {a, b, c, d};
    const int32_t status = validate_batch(count, 3, arrays, 4, signs);
    if (status != PHX_MC_OK) {
      return status;
    }
    for (int64_t i = 0; i < count; ++i) {
      signs[i] = static_cast<int8_t>(
          phx::mc::orient3d(a + 3 * i, b + 3 * i, c + 3 * i, d + 3 * i));
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_incircle(int64_t count, const double* a, const double* b, const double* c,
                        const double* d, int8_t* signs) {
  return guarded([&]() -> int32_t {
    const double* arrays[4] = {a, b, c, d};
    const int32_t status = validate_batch(count, 2, arrays, 4, signs);
    if (status != PHX_MC_OK) {
      return status;
    }
    for (int64_t i = 0; i < count; ++i) {
      signs[i] = static_cast<int8_t>(
          phx::mc::incircle(a + 2 * i, b + 2 * i, c + 2 * i, d + 2 * i));
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_insphere(int64_t count, const double* a, const double* b, const double* c,
                        const double* d, const double* e, int8_t* signs) {
  return guarded([&]() -> int32_t {
    const double* arrays[5] = {a, b, c, d, e};
    const int32_t status = validate_batch(count, 3, arrays, 5, signs);
    if (status != PHX_MC_OK) {
      return status;
    }
    for (int64_t i = 0; i < count; ++i) {
      signs[i] = static_cast<int8_t>(
          phx::mc::insphere(a + 3 * i, b + 3 * i, c + 3 * i, d + 3 * i, e + 3 * i));
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_orient2d_sos(int64_t count, const double* a, const double* b, const double* c,
                            const int64_t* ids, int8_t* signs) {
  return guarded([&]() -> int32_t {
    const double* arrays[3] = {a, b, c};
    int32_t status = validate_batch(count, 2, arrays, 3, signs);
    if (status == PHX_MC_OK) {
      status = validate_ids(count, 3, ids);
    }
    if (status != PHX_MC_OK) {
      return status;
    }
    for (int64_t i = 0; i < count; ++i) {
      const int64_t* r = ids + 3 * i;
      signs[i] = static_cast<int8_t>(
          phx::mc::orient2d_sos(a + 2 * i, b + 2 * i, c + 2 * i, r[0], r[1], r[2]));
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_orient3d_sos(int64_t count, const double* a, const double* b, const double* c,
                            const double* d, const int64_t* ids, int8_t* signs) {
  return guarded([&]() -> int32_t {
    const double* arrays[4] = {a, b, c, d};
    int32_t status = validate_batch(count, 3, arrays, 4, signs);
    if (status == PHX_MC_OK) {
      status = validate_ids(count, 4, ids);
    }
    if (status != PHX_MC_OK) {
      return status;
    }
    for (int64_t i = 0; i < count; ++i) {
      const int64_t* r = ids + 4 * i;
      signs[i] = static_cast<int8_t>(phx::mc::orient3d_sos(
          a + 3 * i, b + 3 * i, c + 3 * i, d + 3 * i, r[0], r[1], r[2], r[3]));
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_incircle_sos(int64_t count, const double* a, const double* b, const double* c,
                            const double* d, const int64_t* ids, int8_t* signs) {
  return guarded([&]() -> int32_t {
    const double* arrays[4] = {a, b, c, d};
    int32_t status = validate_batch(count, 2, arrays, 4, signs);
    if (status == PHX_MC_OK) {
      status = validate_ids(count, 4, ids);
    }
    if (status != PHX_MC_OK) {
      return status;
    }
    for (int64_t i = 0; i < count; ++i) {
      const int64_t* r = ids + 4 * i;
      signs[i] = static_cast<int8_t>(phx::mc::incircle_sos(
          a + 2 * i, b + 2 * i, c + 2 * i, d + 2 * i, r[0], r[1], r[2], r[3]));
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_insphere_sos(int64_t count, const double* a, const double* b, const double* c,
                            const double* d, const double* e, const int64_t* ids,
                            int8_t* signs) {
  return guarded([&]() -> int32_t {
    const double* arrays[5] = {a, b, c, d, e};
    int32_t status = validate_batch(count, 3, arrays, 5, signs);
    if (status == PHX_MC_OK) {
      status = validate_ids(count, 5, ids);
    }
    if (status != PHX_MC_OK) {
      return status;
    }
    for (int64_t i = 0; i < count; ++i) {
      const int64_t* r = ids + 5 * i;
      signs[i] = static_cast<int8_t>(phx::mc::insphere_sos(a + 3 * i, b + 3 * i, c + 3 * i,
                                                           d + 3 * i, e + 3 * i, r[0], r[1],
                                                           r[2], r[3], r[4]));
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_mesh_dimension(const phx_mc_mesh* mesh) {
  return mesh == nullptr ? -1 : mesh->dimension;
}

int64_t phx_mc_mesh_point_count(const phx_mc_mesh* mesh) {
  return mesh == nullptr ? -1 : mesh->point_count();
}

int64_t phx_mc_mesh_cell_count(const phx_mc_mesh* mesh) {
  return mesh == nullptr ? -1 : mesh->cell_count();
}

int64_t phx_mc_mesh_input_point_count(const phx_mc_mesh* mesh) {
  return mesh == nullptr ? -1 : mesh->input_point_count;
}

int32_t phx_mc_mesh_copy_points(const phx_mc_mesh* mesh, int64_t point_capacity,
                                double* points) {
  return guarded([&]() -> int32_t {
    if (mesh == nullptr || point_capacity < 0 ||
        !addressable(point_capacity, mesh->dimension, sizeof(double))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (point_capacity < mesh->point_count()) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    if (!mesh->points.empty() && points == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    std::copy(mesh->points.begin(), mesh->points.end(), points);
    return PHX_MC_OK;
  });
}

int32_t phx_mc_mesh_copy_cells(const phx_mc_mesh* mesh, int64_t cell_capacity, int32_t* cells) {
  return guarded([&]() -> int32_t {
    if (mesh == nullptr || cell_capacity < 0 ||
        !addressable(cell_capacity, mesh->dimension + 1, sizeof(int32_t))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (cell_capacity < mesh->cell_count()) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    if (!mesh->cells.empty() && cells == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    std::copy(mesh->cells.begin(), mesh->cells.end(), cells);
    return PHX_MC_OK;
  });
}

int32_t phx_mc_mesh_copy_vertex_map(const phx_mc_mesh* mesh, int64_t point_capacity,
                                   int32_t* vertex_map) {
  return guarded([&]() -> int32_t {
    if (mesh == nullptr || point_capacity < 0 ||
        !addressable(point_capacity, 1, sizeof(int32_t))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (point_capacity < mesh->input_point_count) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    if (!mesh->vertex_map.empty() && vertex_map == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    std::copy(mesh->vertex_map.begin(), mesh->vertex_map.end(), vertex_map);
    return PHX_MC_OK;
  });
}

int32_t phx_mc_mesh_copy_cell_segments(const phx_mc_mesh* mesh, int64_t cell_capacity,
                                       int32_t* cell_segments) {
  return guarded([&]() -> int32_t {
    if (mesh == nullptr || cell_capacity < 0 ||
        !addressable(cell_capacity, 3, sizeof(int32_t))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (cell_capacity < mesh->cell_count()) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    if (mesh->cell_count() > 0 && cell_segments == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (mesh->cell_segments.empty()) {
      std::fill_n(cell_segments, 3 * mesh->cell_count(), -1);
    } else {
      std::copy(mesh->cell_segments.begin(), mesh->cell_segments.end(), cell_segments);
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_mesh_copy_constraint_sources(const phx_mc_mesh* mesh, int64_t point_capacity,
                                            int32_t* sources) {
  return guarded([&]() -> int32_t {
    if (mesh == nullptr || point_capacity < 0 ||
        !addressable(point_capacity, 1, sizeof(int32_t))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (point_capacity < mesh->point_count()) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    if (mesh->point_count() > 0 && sources == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (mesh->constraint_sources.empty()) {
      std::fill_n(sources, mesh->point_count(), -1);
    } else {
      std::copy(mesh->constraint_sources.begin(), mesh->constraint_sources.end(), sources);
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_mesh_copy_constraint_parents(const phx_mc_mesh* mesh, int64_t point_capacity,
                                            int32_t* parents) {
  return guarded([&]() -> int32_t {
    if (mesh == nullptr || point_capacity < 0 ||
        !addressable(point_capacity, 2, sizeof(int32_t))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (point_capacity < mesh->point_count()) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    if (mesh->point_count() > 0 && parents == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (mesh->constraint_parents.empty()) {
      std::fill_n(parents, 2 * mesh->point_count(), -1);
    } else {
      std::copy(mesh->constraint_parents.begin(), mesh->constraint_parents.end(), parents);
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_mesh_copy_constraint_ratios(const phx_mc_mesh* mesh, int64_t point_capacity,
                                           double* ratios) {
  return guarded([&]() -> int32_t {
    if (mesh == nullptr || point_capacity < 0 ||
        !addressable(point_capacity, 1, sizeof(double))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (point_capacity < mesh->point_count()) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    if (mesh->point_count() > 0 && ratios == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (mesh->constraint_ratios.empty()) {
      std::fill_n(ratios, mesh->point_count(), 0.0);
    } else {
      std::copy(mesh->constraint_ratios.begin(), mesh->constraint_ratios.end(), ratios);
    }
    return PHX_MC_OK;
  });
}

void phx_mc_mesh_free(phx_mc_mesh* mesh) { delete mesh; }

int32_t phx_mc_plc3d_prepare_source(
    const phx_mc_plc3d_prepare_source_request* request) {
  return run_plc3d_request(request, phx::mc::prepare_plc_source);
}

int32_t phx_mc_plc3d_recover(
    const phx_mc_plc3d_prepare_source_request* request) {
  return run_plc3d_request(request, phx::mc::recover_plc);
}

int32_t phx_mc_plc3d_sizes(const phx_mc_plc3d_result* result, int64_t* sizes) {
  return guarded([&]() -> int32_t {
    if (result == nullptr || sizes == nullptr) return PHX_MC_INVALID_ARGUMENT;
    const auto& recovery = result->recovery;
    if (recovery.points.size() % 3 != 0 || recovery.tets.size() % 4 != 0 ||
        recovery.faces.size() % 3 != 0 || recovery.segments.size() % 2 != 0 ||
        recovery.plc_edges.size() % 2 != 0 || recovery.input_triangles.size() % 3 != 0 ||
        recovery.input_polygons.size() != recovery.input_triangles.size() / 3 ||
        recovery.tet_regions.size() != recovery.tets.size() / 4 ||
        recovery.face_sources.size() != recovery.faces.size() / 3 ||
        recovery.face_orientations.size() != recovery.faces.size() / 3 ||
        recovery.segment_sources.size() != recovery.segments.size() / 2 ||
        recovery.vertex_dimension.size() != recovery.points.size() / 3 ||
        recovery.protection.size() != recovery.points.size() / 3 ||
        recovery.witnesses.size() != recovery.points.size() / 3) {
      return PHX_MC_INTERNAL_ERROR;
    }
    const std::size_t maximum =
        static_cast<std::size_t>(std::numeric_limits<int64_t>::max());
    if (recovery.points.size() > maximum || recovery.tets.size() > maximum ||
        recovery.faces.size() > maximum || recovery.segments.size() > maximum ||
        recovery.plc_edges.size() > maximum || recovery.input_triangles.size() > maximum) {
      return PHX_MC_INDEX_RANGE_ERROR;
    }
    sizes[0] = static_cast<int64_t>(recovery.points.size() / 3);
    sizes[1] = static_cast<int64_t>(recovery.tets.size() / 4);
    sizes[2] = static_cast<int64_t>(recovery.faces.size() / 3);
    sizes[3] = static_cast<int64_t>(recovery.segments.size() / 2);
    sizes[4] = static_cast<int64_t>(recovery.plc_edges.size() / 2);
    sizes[5] = static_cast<int64_t>(recovery.input_triangles.size() / 3);
    return PHX_MC_OK;
  });
}

int32_t phx_mc_plc3d_export(
    const phx_mc_plc3d_result* result, int64_t point_capacity,
    int64_t tetrahedron_capacity, int64_t face_capacity, int64_t segment_capacity,
    int64_t edge_capacity, int64_t input_triangle_capacity, double* points,
    int32_t* tetrahedra, int32_t* tetrahedron_regions, int32_t* faces,
    int32_t* face_sources, int8_t* face_orientations, int32_t* segments,
    int32_t* segment_sources, double* protection, int8_t* vertex_dimensions,
    int32_t* plc_edges, int32_t* input_triangles, int32_t* input_polygons) {
  return guarded([&]() -> int32_t {
    if (result == nullptr || point_capacity < 0 || tetrahedron_capacity < 0 ||
        face_capacity < 0 || segment_capacity < 0 || edge_capacity < 0 ||
        input_triangle_capacity < 0 ||
        !addressable(point_capacity, 3, sizeof(double)) ||
        !addressable(point_capacity, 1, sizeof(double)) ||
        !addressable(point_capacity, 1, sizeof(int8_t)) ||
        !addressable(tetrahedron_capacity, 4, sizeof(int32_t)) ||
        !addressable(tetrahedron_capacity, 1, sizeof(int32_t)) ||
        !addressable(face_capacity, 3, sizeof(int32_t)) ||
        !addressable(face_capacity, 1, sizeof(int32_t)) ||
        !addressable(face_capacity, 1, sizeof(int8_t)) ||
        !addressable(segment_capacity, 2, sizeof(int32_t)) ||
        !addressable(segment_capacity, 1, sizeof(int32_t)) ||
        !addressable(edge_capacity, 2, sizeof(int32_t)) ||
        !addressable(input_triangle_capacity, 3, sizeof(int32_t)) ||
        !addressable(input_triangle_capacity, 1, sizeof(int32_t))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const auto& recovery = result->recovery;
    if (recovery.points.size() % 3 != 0 || recovery.tets.size() % 4 != 0 ||
        recovery.faces.size() % 3 != 0 || recovery.segments.size() % 2 != 0 ||
        recovery.plc_edges.size() % 2 != 0 || recovery.input_triangles.size() % 3 != 0 ||
        recovery.input_polygons.size() != recovery.input_triangles.size() / 3 ||
        recovery.tet_regions.size() != recovery.tets.size() / 4 ||
        recovery.face_sources.size() != recovery.faces.size() / 3 ||
        recovery.face_orientations.size() != recovery.faces.size() / 3 ||
        recovery.segment_sources.size() != recovery.segments.size() / 2 ||
        recovery.vertex_dimension.size() != recovery.points.size() / 3 ||
        recovery.protection.size() != recovery.points.size() / 3 ||
        recovery.witnesses.size() != recovery.points.size() / 3) {
      return PHX_MC_INTERNAL_ERROR;
    }
    const int64_t point_count = static_cast<int64_t>(recovery.points.size() / 3);
    const int64_t tet_count = static_cast<int64_t>(recovery.tets.size() / 4);
    const int64_t face_count = static_cast<int64_t>(recovery.faces.size() / 3);
    const int64_t segment_count = static_cast<int64_t>(recovery.segments.size() / 2);
    const int64_t edge_count = static_cast<int64_t>(recovery.plc_edges.size() / 2);
    const int64_t input_triangle_count =
        static_cast<int64_t>(recovery.input_triangles.size() / 3);
    if (point_capacity < point_count || tetrahedron_capacity < tet_count ||
        face_capacity < face_count || segment_capacity < segment_count ||
        edge_capacity < edge_count || input_triangle_capacity < input_triangle_count) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    if (!plc_output_bank_valid(recovery.points, points) ||
        !plc_output_bank_valid(recovery.tets, tetrahedra) ||
        !plc_output_bank_valid(recovery.tet_regions, tetrahedron_regions) ||
        !plc_output_bank_valid(recovery.faces, faces) ||
        !plc_output_bank_valid(recovery.face_sources, face_sources) ||
        !plc_output_bank_valid(recovery.face_orientations, face_orientations) ||
        !plc_output_bank_valid(recovery.segments, segments) ||
        !plc_output_bank_valid(recovery.segment_sources, segment_sources) ||
        !plc_output_bank_valid(recovery.protection, protection) ||
        !plc_output_bank_valid(recovery.vertex_dimension, vertex_dimensions) ||
        !plc_output_bank_valid(recovery.plc_edges, plc_edges) ||
        !plc_output_bank_valid(recovery.input_triangles, input_triangles) ||
        !plc_output_bank_valid(recovery.input_polygons, input_polygons)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    copy_plc_output_bank(recovery.points, points);
    copy_plc_output_bank(recovery.tets, tetrahedra);
    copy_plc_output_bank(recovery.tet_regions, tetrahedron_regions);
    copy_plc_output_bank(recovery.faces, faces);
    copy_plc_output_bank(recovery.face_sources, face_sources);
    copy_plc_output_bank(recovery.face_orientations, face_orientations);
    copy_plc_output_bank(recovery.segments, segments);
    copy_plc_output_bank(recovery.segment_sources, segment_sources);
    copy_plc_output_bank(recovery.protection, protection);
    copy_plc_output_bank(recovery.vertex_dimension, vertex_dimensions);
    copy_plc_output_bank(recovery.plc_edges, plc_edges);
    copy_plc_output_bank(recovery.input_triangles, input_triangles);
    copy_plc_output_bank(recovery.input_polygons, input_polygons);
    return PHX_MC_OK;
  });
}

int32_t phx_mc_plc3d_source_witnesses(
    const phx_mc_plc3d_result* result, int64_t point_capacity, int8_t* strata,
    int32_t* entities, double* parameters, double* deviations) {
  return guarded([&]() -> int32_t {
    if (result == nullptr || point_capacity < 0 ||
        !addressable(point_capacity, 1, sizeof(int8_t)) ||
        !addressable(point_capacity, 1, sizeof(int32_t)) ||
        !addressable(point_capacity, 2, sizeof(double)) ||
        !addressable(point_capacity, 1, sizeof(double))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const auto& recovery = result->recovery;
    if (recovery.witnesses.size() != recovery.points.size() / 3)
      return PHX_MC_INTERNAL_ERROR;
    if (point_capacity < static_cast<int64_t>(recovery.witnesses.size()))
      return PHX_MC_CAPACITY_EXCEEDED;
    if (!recovery.witnesses.empty() &&
        (strata == nullptr || entities == nullptr || parameters == nullptr ||
         deviations == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    for (std::size_t vertex = 0; vertex < recovery.witnesses.size(); ++vertex) {
      const auto& witness = recovery.witnesses[vertex];
      strata[vertex] = static_cast<int8_t>(witness.stratum);
      entities[vertex] = witness.entity;
      parameters[2 * vertex] = witness.parameters[0];
      parameters[2 * vertex + 1] = witness.parameters[1];
      deviations[vertex] = witness.deviation;
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_plc3d_failure(
    const phx_mc_plc3d_result* result, int32_t* reason, int32_t* first_kind,
    int64_t* first, int32_t* second_kind, int64_t* second, double* certified_deviation,
    double* declared_bound) {
  if (result == nullptr || reason == nullptr || first_kind == nullptr || first == nullptr ||
      second_kind == nullptr || second == nullptr || certified_deviation == nullptr ||
      declared_bound == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const auto& failure = result->recovery.failure;
  *reason = failure.reason;
  *first_kind = failure.first_kind;
  *first = failure.first;
  *second_kind = failure.second_kind;
  *second = failure.second;
  *certified_deviation = result->recovery.source_refusal.certified_deviation;
  *declared_bound = result->recovery.source_refusal.declared_bound;
  return PHX_MC_OK;
}

int32_t phx_mc_plc3d_suppression_search(const phx_mc_plc3d_result* result, int64_t* values) {
  if (result == nullptr || values == nullptr) return PHX_MC_INVALID_ARGUMENT;
  const auto& search = result->recovery.suppression_search;
  values[0] = search.searches;
  values[1] = search.fills;
  values[2] = search.nodes;
  values[3] = search.max_nodes;
  values[4] = search.max_wedge_cells;
  values[5] = search.merges;
  values[6] = search.max_merge_layers;
  values[7] = search.exhausted;
  values[8] = search.max_exhausted_cells;
  return PHX_MC_OK;
}

void phx_mc_plc3d_free(phx_mc_plc3d_result* result) { delete result; }
 

}  // extern "C"
