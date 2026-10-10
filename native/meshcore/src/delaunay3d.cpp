//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// 3D Delaunay and regular triangulation entry points: point-set construction
// and the prepared incremental triangulation handle (see triangulation3d.hpp
// for the construction invariants).
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "capi_guard.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "spatial_sort.hpp"
#include "triangulation3d.hpp"

namespace phx::mc {
namespace {

// Deduplicated BRIO insertion of a validated point set: vertex_map receives
// the deduplication map and alive the surviving vertices.
int32_t build_point_set(Triangulation3D& triangulation, const double* points,
                        int64_t point_count, const double* weights,
                        NativeVector<int32_t>& vertex_map, NativeVector<char>& alive) {
  MemoryScope memory(triangulation.memory_owner());
  const auto representatives =
      deduplicate_points(points, point_count, 3, weights, vertex_map);
  const auto order = brio_hilbert_order(points, 3, representatives);
  alive.assign(static_cast<std::size_t>(point_count), 0);
  for (int32_t vertex : representatives) {
    alive[static_cast<std::size_t>(vertex)] = 1;
  }
  return triangulation.build(order, alive);
}

// Canonical mesh of the finite cells, with facet constraints and regions when
// the complex carries labels.
NativeUniquePtr<phx_mc_mesh> mesh_of(const Triangulation3D& triangulation,
                                    const double* points, int64_t point_count,
                                    NativeVector<int32_t> vertex_map) {
  MemoryScope memory(triangulation.memory_owner());
  auto result = make_native_unique<phx_mc_mesh>();
  const TetrahedralComplex& complex = triangulation.complex();
  triangulation.collect_cells(result->cells);
  if (complex.labeled()) {
    result->cell_constraints.reserve(result->cells.size());
    result->cell_regions.reserve(result->cells.size() / 4);
    for (std::size_t t = 0; t < complex.tets.size(); ++t) {
      const Tetrahedron& tet = complex.tets[t];
      if (tet.v[0] == kDeadVertex || is_ghost(tet)) {
        continue;
      }
      for (int k = 0; k < 4; ++k) {
        result->cell_constraints.push_back(complex.constraint(static_cast<int32_t>(t), k));
      }
      result->cell_regions.push_back(complex.region(static_cast<int32_t>(t)));
    }
  }
  result->dimension = 3;
  result->input_point_count = point_count;
  result->points.assign(points, points + 3 * point_count);
  result->vertex_map = std::move(vertex_map);
  canonicalize_cells(*result);
  return result;
}

int32_t triangulate_3d(int64_t point_count, const double* points, const double* weights,
                       bool weighted, int64_t max_tetrahedra, phx_mc_mesh** mesh) {
  if (mesh == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *mesh = nullptr;
  if (point_count < 0 || point_count > kMaxMeshPoints || max_tetrahedra < 0) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (point_count > 0 && (points == nullptr || (weighted && weights == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (point_count >= 4 && active_execution_scope != nullptr &&
      !active_execution_scope->admit(5)) return active_execution_scope->status();
  MemoryOwner memory_owner = scratch_memory_owner();
  if (!memory_owner) memory_owner = std::make_shared<BoundedMemoryResource>();
  MemoryScope memory(memory_owner);
  const double* active_weights = weighted ? weights : nullptr;
  int32_t status = validate_points(points, point_count, 3, active_weights);
  if (status != PHX_MC_OK) {
    return status;
  }
  InsertionLimits limits;
  limits.max_tetrahedra = max_tetrahedra;
  Triangulation3D triangulation(points, active_weights, point_count, limits);
  NativeVector<int32_t> vertex_map;
  NativeVector<char> alive;
  status = build_point_set(triangulation, points, point_count, active_weights, vertex_map, alive);
  if (status != PHX_MC_OK) {
    return status;
  }
  for (int32_t& target : vertex_map) {
    if (target >= 0 && alive[static_cast<std::size_t>(target)] == 0) {
      target = -1;
    }
  }
  *mesh = mesh_of(triangulation, points, point_count, std::move(vertex_map)).release();
  return PHX_MC_OK;
}

}  // namespace
}  // namespace phx::mc

// Prepared incremental Delaunay triangulation: every submitted point receives
// the next vertex id; vertex_map[id] is id for a vertex, the id of an earlier
// identical point, or -1 when the insertion was refused.
struct phx_mc_triangulation_3d : phx::mc::NativeAllocatedObject {
  explicit phx_mc_triangulation_3d(phx::mc::MemoryOwner owner = phx::mc::scratch_memory_owner())
      : memory_owner(std::move(owner)),
        points(phx::mc::NativeAllocator<double>(memory_owner)),
        vertex_map(phx::mc::NativeAllocator<int32_t>(memory_owner)),
        alive(phx::mc::NativeAllocator<char>(memory_owner)) {}
  phx::mc::MemoryOwner memory_owner;
  phx::mc::NativeVector<double> points;
  phx::mc::NativeVector<int32_t> vertex_map;
  phx::mc::NativeVector<char> alive;
  phx::mc::NativeUniquePtr<phx::mc::Triangulation3D> triangulation;
  int64_t max_vertices = 0;
  int64_t committed = 0;
  int64_t refused = 0;
  std::size_t peak_bytes = 0;
  // Set when a commit contradicted the star-shapedness premise after
  // writing; every later call reports PHX_MC_INTERNAL_ERROR.
  bool broken = false;

  int64_t vertex_count() const { return static_cast<int64_t>(vertex_map.size()); }

  std::size_t retained_bytes() const { return memory_owner->live_bytes(); }

  void record_peak() { peak_bytes = memory_owner->peak_bytes(); }

  bool live_vertex(int64_t id) const {
    return id >= 0 && id < vertex_count() && vertex_map[static_cast<std::size_t>(id)] == id;
  }
};

namespace phx::mc {
namespace {

using Handle = phx_mc_triangulation_3d;

int32_t create(int64_t point_count, const double* points, int64_t max_vertices,
               int64_t max_tetrahedra, int64_t max_cavity, Handle** out) {
  if (out == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *out = nullptr;
  if (point_count < 0 || max_vertices < point_count || max_vertices > kMaxMeshPoints ||
      max_tetrahedra < 0 || max_cavity < 1 || (point_count > 0 && points == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (point_count >= 4 && active_execution_scope != nullptr &&
      !active_execution_scope->admit(5)) return active_execution_scope->status();
  MemoryOwner memory_owner = scratch_memory_owner();
  if (!memory_owner) memory_owner = std::make_shared<BoundedMemoryResource>();
  MemoryScope memory(memory_owner);
  int32_t status = validate_points(points, point_count, 3, nullptr);
  if (status != PHX_MC_OK) {
    return status;
  }
  auto handle = make_native_unique<Handle>(memory_owner);
  handle->max_vertices = max_vertices;
  handle->points.assign(points, points + 3 * point_count);
  // The initial point set is built like phx_mc_delaunay_3d; the cavity bound
  // applies to later insertions.
  InsertionLimits limits;
  limits.max_tetrahedra = max_tetrahedra;
  handle->triangulation =
      make_native_unique<Triangulation3D>(handle->points.data(), nullptr, point_count, limits);
  status = build_point_set(*handle->triangulation, handle->points.data(), point_count, nullptr,
                           handle->vertex_map, handle->alive);
  if (status != PHX_MC_OK) {
    return status;
  }
  handle->triangulation->limits().max_cavity = static_cast<std::size_t>(max_cavity);
  handle->record_peak();
  *out = handle.release();
  return PHX_MC_OK;
}

int32_t check_handle(const Handle* handle) {
  if (handle == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  return handle->broken ? PHX_MC_INTERNAL_ERROR : PHX_MC_OK;
}

// Readiness of a handle about to change or walk: a batch work limit left by an
// allocation failure escaping mid-batch no longer applies.
int32_t ready(Handle* handle) {
  const int32_t status = check_handle(handle);
  if (status == PHX_MC_OK) {
    handle->triangulation->limits().work_limit = std::numeric_limits<int64_t>::max();
  }
  return status;
}

// Inserts one batch: validated as a whole, ids assigned in submission order,
// exact duplicates within the batch share their first occurrence, insertion
// in BRIO order (the result does not depend on it) under one work budget.
// Outputs are written as items complete, so an allocation failure escaping
// mid-batch leaves them describing exactly the committed items.
int32_t insert(Handle* handle, int64_t count, const double* points, int64_t work_limit,
               int32_t* vertices, int32_t* item_status) {
  int32_t status = ready(handle);
  if (status != PHX_MC_OK) {
    return status;
  }
  MemoryScope memory(handle->memory_owner);
  if (count < 0 || work_limit < 0 || vertices == nullptr || item_status == nullptr ||
      !addressable(count, 3, sizeof(double)) || (count > 0 && points == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  status = validate_points(points, count, 3, nullptr);
  if (status != PHX_MC_OK) {
    return status;
  }
  const int64_t base = handle->vertex_count();
  if (count > handle->max_vertices - base) {
    return PHX_MC_CAPACITY_EXCEEDED;
  }
  std::fill_n(vertices, count, -1);
  std::fill_n(item_status, count, PHX_MC_CAPACITY_EXCEEDED);
  NativeVector<int32_t> local_map;
  const auto representatives = deduplicate_points(points, count, 3, nullptr, local_map);
  const auto order = brio_hilbert_order(points, 3, representatives);
  // Growth is reserved before any state changes.
  const std::size_t total = static_cast<std::size_t>(base + count);
  Triangulation3D& triangulation = *handle->triangulation;
  reserve_for(handle->vertex_map, total);
  reserve_for(handle->alive, total);
  triangulation.reserve_vertices(static_cast<int64_t>(total));
  reserve_for(handle->points, 3 * total);
  // Coordinate relocation is immediately followed by a no-allocation bind.
  // No later reservation can leave the accepted complex borrowing freed data.
  triangulation.rebind(handle->points.data(), base);
  handle->points.insert(handle->points.end(), points, points + 3 * count);
  triangulation.rebind(handle->points.data(), static_cast<int64_t>(total));
  handle->vertex_map.resize(total, -1);
  handle->alive.resize(total, 0);
  InsertionLimits& limits = triangulation.limits();
  limits.work_limit = triangulation.work() > std::numeric_limits<int64_t>::max() - work_limit
                          ? std::numeric_limits<int64_t>::max()
                          : triangulation.work() + work_limit;
  bool exhausted = false;
  for (int32_t local : order) {
    if (exhausted) {
      break;
    }
    const int32_t id = static_cast<int32_t>(base + local);
    int32_t duplicate = -1;
    const int32_t result = triangulation.insert(id, handle->alive, &duplicate);
    if (result == PHX_MC_INTERNAL_ERROR) {
      handle->broken = true;
      return PHX_MC_INTERNAL_ERROR;
    }
    item_status[local] = result;
    if (result == PHX_MC_OK) {
      const int32_t target = duplicate >= 0 ? duplicate : id;
      handle->vertex_map[static_cast<std::size_t>(id)] = target;
      handle->alive[static_cast<std::size_t>(id)] = duplicate >= 0 ? 0 : 1;
      handle->committed += duplicate >= 0 ? 0 : 1;
      vertices[local] = target;
    } else {
      ++handle->refused;
      exhausted = triangulation.refusal() == Refusal::kWork;
    }
  }
  limits.work_limit = std::numeric_limits<int64_t>::max();
  for (int64_t i = 0; i < count; ++i) {
    const int32_t representative = local_map[static_cast<std::size_t>(i)];
    const int32_t target =
        handle->vertex_map[static_cast<std::size_t>(base + representative)];
    handle->vertex_map[static_cast<std::size_t>(base + i)] = target;
    vertices[i] = target;
    item_status[i] = item_status[representative];
  }
  handle->record_peak();
  return PHX_MC_OK;
}

int32_t constrain_facets(Handle* handle, int64_t count, const int32_t* facets,
                         const int32_t* ids, int32_t* item_status) {
  int32_t status = ready(handle);
  if (status != PHX_MC_OK) {
    return status;
  }
  MemoryScope memory(handle->memory_owner);
  if (count < 0 || item_status == nullptr || !addressable(count, 3, sizeof(int32_t)) ||
      (count > 0 && (facets == nullptr || ids == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  Triangulation3D& triangulation = *handle->triangulation;
  TetrahedralComplex& complex = triangulation.complex();
  complex.enable_labels();
  for (int64_t i = 0; i < count; ++i) {
    const int32_t* f = facets + 3 * i;
    int32_t t = -1;
    int slot = -1;
    const bool valid = ids[i] >= 0 && handle->live_vertex(f[0]) && handle->live_vertex(f[1]) &&
                       handle->live_vertex(f[2]) && f[0] != f[1] && f[0] != f[2] &&
                       f[1] != f[2] && triangulation.find_facet(f[0], f[1], f[2], t, slot);
    int32_t result = PHX_MC_INVALID_INPUT;
    if (valid) {
      const int32_t existing = complex.constraint(t, slot);
      if (existing == kNoConstraint) {
        complex.set_facet_constraint(t, slot, ids[i]);
        triangulation.count_constrained_facet();
        result = PHX_MC_OK;
      } else if (existing == ids[i]) {
        result = PHX_MC_OK;
      }
    }
    item_status[i] = result;
  }
  handle->record_peak();
  return PHX_MC_OK;
}

int32_t label_regions(Handle* handle, int64_t count, const double* seeds, const int32_t* labels,
                      int32_t* item_status) {
  int32_t status = ready(handle);
  if (status != PHX_MC_OK) {
    return status;
  }
  MemoryScope memory(handle->memory_owner);
  if (count < 0 || item_status == nullptr || !addressable(count, 3, sizeof(double)) ||
      (count > 0 && (seeds == nullptr || labels == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  status = validate_points(seeds, count, 3, nullptr);
  if (status != PHX_MC_OK) {
    return status;
  }
  Triangulation3D& triangulation = *handle->triangulation;
  TetrahedralComplex& complex = triangulation.complex();
  complex.enable_labels();
  NativeVector<int32_t> proposed(complex.tets.size(), kNoRegion);
  NativeVector<int32_t> proposed_status(static_cast<std::size_t>(count), PHX_MC_INVALID_INPUT);
  std::fill_n(item_status, count, PHX_MC_CAPACITY_EXCEEDED);
  for (int64_t i = 0; i < count; ++i) {
    const double* seed = seeds + 3 * i;
    const int32_t t = triangulation.locate(seed);
    int32_t result = PHX_MC_INVALID_INPUT;
    if (labels[i] >= 0 && t >= 0 && triangulation.location(t, seed) == 0) {
      const int32_t existing = proposed[static_cast<std::size_t>(t)];
      if (existing == kNoRegion) {
        triangulation.flood_region(t, labels[i], proposed);
        result = PHX_MC_OK;
      } else if (existing == labels[i]) {
        result = PHX_MC_OK;
      }
    }
    proposed_status[static_cast<std::size_t>(i)] = result;
  }
  for (std::size_t t = 0; t < proposed.size(); ++t) {
    complex.set_region(static_cast<int32_t>(t), proposed[t]);
  }
  std::copy(proposed_status.begin(), proposed_status.end(), item_status);
  handle->record_peak();
  return PHX_MC_OK;
}

int32_t locate(Handle* handle, int64_t count, const double* points, int32_t* cells,
               int8_t* locations, int32_t* item_status) {
  int32_t status = ready(handle);
  if (status != PHX_MC_OK) {
    return status;
  }
  MemoryScope memory(handle->memory_owner);
  if (count < 0 || cells == nullptr || locations == nullptr || item_status == nullptr ||
      !addressable(count, 4, sizeof(double)) || (count > 0 && points == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  Triangulation3D& triangulation = *handle->triangulation;
  for (int64_t i = 0; i < count; ++i) {
    const double* p = points + 3 * i;
    const int32_t result = validate_points(p, 1, 3, nullptr);
    int32_t* cell = cells + 4 * i;
    std::fill_n(cell, 4, -1);
    locations[i] = -1;
    if (result == PHX_MC_OK) {
      // The walk is unbudgeted outside insertion batches.
      const int32_t t = triangulation.locate(p);
      std::copy_n(triangulation.tet(t).v, 4, cell);
      canonicalize_cell(cell, nullptr, 4);
      locations[i] = static_cast<int8_t>(triangulation.location(t, p));
    }
    item_status[i] = result;
  }
  return PHX_MC_OK;
}

int32_t finalize(const Handle* handle, phx_mc_mesh** mesh) {
  if (mesh == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *mesh = nullptr;
  const int32_t status = check_handle(handle);
  if (status != PHX_MC_OK) {
    return status;
  }
  MemoryScope memory(handle->memory_owner);
  *mesh = mesh_of(*handle->triangulation, handle->points.data(), handle->vertex_count(),
                  handle->vertex_map)
              .release();
  return PHX_MC_OK;
}

}  // namespace
}  // namespace phx::mc

extern "C" {

int32_t phx_mc_delaunay_3d(int64_t point_count, const double* points, int64_t max_tetrahedra,
                           phx_mc_mesh** mesh) {
  return phx::mc::guarded([&] {
    return phx::mc::triangulate_3d(point_count, points, nullptr, false, max_tetrahedra, mesh);
  });
}

int32_t phx_mc_regular_3d(int64_t point_count, const double* points, const double* weights,
                          int64_t max_tetrahedra, phx_mc_mesh** mesh) {
  return phx::mc::guarded([&] {
    return phx::mc::triangulate_3d(point_count, points, weights, true, max_tetrahedra, mesh);
  });
}

int32_t phx_mc_triangulation_3d_create(int64_t point_count, const double* points,
                                       int64_t max_vertices, int64_t max_tetrahedra,
                                       int64_t max_cavity,
                                       phx_mc_triangulation_3d** triangulation) {
  return phx::mc::guarded([&] {
    return phx::mc::create(point_count, points, max_vertices, max_tetrahedra, max_cavity,
                           triangulation);
  });
}

int32_t phx_mc_triangulation_3d_insert(phx_mc_triangulation_3d* triangulation, int64_t count,
                                       const double* points, int64_t work_limit,
                                       int32_t* vertices, int32_t* item_status) {
  return phx::mc::guarded([&] {
    return phx::mc::insert(triangulation, count, points, work_limit, vertices, item_status);
  });
}

int32_t phx_mc_triangulation_3d_constrain_facets(phx_mc_triangulation_3d* triangulation,
                                                 int64_t count, const int32_t* facets,
                                                 const int32_t* constraint_ids,
                                                 int32_t* item_status) {
  return phx::mc::guarded([&] {
    return phx::mc::constrain_facets(triangulation, count, facets, constraint_ids, item_status);
  });
}

int32_t phx_mc_triangulation_3d_label_regions(phx_mc_triangulation_3d* triangulation,
                                              int64_t seed_count, const double* seeds,
                                              const int32_t* labels, int32_t* item_status) {
  return phx::mc::guarded([&] {
    return phx::mc::label_regions(triangulation, seed_count, seeds, labels, item_status);
  });
}

int32_t phx_mc_triangulation_3d_locate(phx_mc_triangulation_3d* triangulation, int64_t count,
                                       const double* points, int32_t* cells, int8_t* locations,
                                       int32_t* item_status) {
  return phx::mc::guarded([&] {
    return phx::mc::locate(triangulation, count, points, cells, locations, item_status);
  });
}

int32_t phx_mc_triangulation_3d_statistics(const phx_mc_triangulation_3d* triangulation,
                                           int64_t* values) {
  return phx::mc::guarded([&]() -> int32_t {
    if (triangulation == nullptr || values == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    phx::mc::MemoryScope memory(triangulation->memory_owner);
    const phx::mc::Triangulation3D& state = *triangulation->triangulation;
    const phx::mc::TetrahedralComplex& complex = state.complex();
    int64_t live = 0;
    for (int64_t id = 0; id < triangulation->vertex_count(); ++id) {
      live += triangulation->live_vertex(id) ? 1 : 0;
    }
    values[PHX_MC_TRIANGULATION_3D_VERTEX_IDS] = triangulation->vertex_count();
    values[PHX_MC_TRIANGULATION_3D_LIVE_VERTICES] = live;
    values[PHX_MC_TRIANGULATION_3D_FINITE_CELLS] = complex.finite_count;
    values[PHX_MC_TRIANGULATION_3D_GHOST_CELLS] = complex.live_count() - complex.finite_count;
    values[PHX_MC_TRIANGULATION_3D_CELL_SLOTS] = static_cast<int64_t>(complex.tets.size());
    values[PHX_MC_TRIANGULATION_3D_FREE_SLOTS] = static_cast<int64_t>(complex.free_slots.size());
    values[PHX_MC_TRIANGULATION_3D_CONSTRAINED_FACETS] = state.constrained_facets();
    values[PHX_MC_TRIANGULATION_3D_WORK] = state.work();
    values[PHX_MC_TRIANGULATION_3D_COMMITTED] = triangulation->committed;
    values[PHX_MC_TRIANGULATION_3D_REFUSED] = triangulation->refused;
    values[PHX_MC_TRIANGULATION_3D_LARGEST_CAVITY] =
        static_cast<int64_t>(state.largest_cavity());
    values[PHX_MC_TRIANGULATION_3D_RETAINED_BYTES] =
        static_cast<int64_t>(triangulation->retained_bytes());
    values[PHX_MC_TRIANGULATION_3D_PEAK_BYTES] = static_cast<int64_t>(triangulation->peak_bytes);
    return triangulation->broken ? PHX_MC_INTERNAL_ERROR : PHX_MC_OK;
  });
}

int32_t phx_mc_triangulation_3d_finalize(const phx_mc_triangulation_3d* triangulation,
                                         phx_mc_mesh** mesh) {
  return phx::mc::guarded([&] { return phx::mc::finalize(triangulation, mesh); });
}

void phx_mc_triangulation_3d_free(phx_mc_triangulation_3d* triangulation) {
  phx::mc::destroy_native_object(triangulation);
}

}  // extern "C"
