//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// C ABI: 2D Delaunay and regular triangulations (incremental Bowyer-Watson
// with ghost triangles and symbolic perturbation, see triangulation2d.hpp).
#include <algorithm>
#include <cstdint>

#include "capi_guard.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "triangulation2d.hpp"

namespace {

struct EvidenceWriter {
  const phx::mc::PlanarBudget& budget;
  const phx::mc::MemoryBudgetWindow& window;
  uint64_t* work;
  uint64_t* memory;
  ~EvidenceWriter() {
    if (work != nullptr) {
      const auto values = budget.evidence();
      std::copy(values.begin(), values.end(), work);
    }
    if (memory != nullptr) {
      const auto values = window.evidence();
      std::copy(values.begin(), values.end(), memory);
    }
  }
};

int32_t triangulate_2d(int64_t point_count, const double* points, const double* weights,
                       bool weighted, int64_t max_triangles, int64_t max_cavity_cells,
                       int64_t max_work, int64_t max_scratch_bytes,
                       uint64_t* work_evidence, uint64_t* memory_evidence, phx_mc_mesh** mesh) {
  if (work_evidence != nullptr) {
    std::fill_n(work_evidence, 9, uint64_t{0});
  }
  if (memory_evidence != nullptr) {
    std::fill_n(memory_evidence, 6, uint64_t{0});
  }
  if (mesh == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *mesh = nullptr;
  if (point_count < 0 || point_count > phx::mc::kMaxMeshPoints ||
      (point_count > 0 && points == nullptr) ||
      (weighted && point_count > 0 && weights == nullptr) || max_triangles < 0 ||
      max_cavity_cells < 0 || max_work < 0 || max_scratch_bytes < 0) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  phx::mc::MemoryBudgetWindow window(static_cast<std::size_t>(max_scratch_bytes));
  phx::mc::MemoryScope scope(window.owner());
  phx::mc::PlanarBudget budget(max_cavity_cells, max_work, window.owner());
  EvidenceWriter evidence{budget, window, work_evidence, memory_evidence};
  int32_t status = phx::mc::validate_points(points, point_count, 2, weights,
                                            phx::mc::charge_planar_work, &budget);
  if (status != PHX_MC_OK) {
    return status;
  }
  auto result = phx::mc::make_native_unique<phx_mc_mesh>();
  result->dimension = 2;
  result->input_point_count = point_count;
  if (point_count == 0) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  phx::mc::Triangulation2D triangulation(budget);
  triangulation.reset(points, point_count, weights);
  status = phx::mc::build_delaunay_2d(triangulation, points, point_count, weights, max_triangles,
                                      result->vertex_map);
  if (status != PHX_MC_OK) {
    return status;
  }
  for (int32_t& target : result->vertex_map) {
    budget.charge();
    if (target >= 0 && triangulation.vertex_triangle[static_cast<std::size_t>(target)] < 0) {
      target = -1;
    }
  }
  budget.charge(static_cast<std::uint64_t>(point_count));
  result->points.assign(points, points + 2 * point_count);
  triangulation.finite_cells(result->cells);
  phx::mc::canonicalize_cells(*result, phx::mc::charge_planar_work, &budget);
  *mesh = result.release();
  return PHX_MC_OK;
}

}  // namespace

extern "C" {

int32_t phx_mc_delaunay_2d(int64_t point_count, const double* points, int64_t max_triangles,
                           int64_t max_cavity_cells, int64_t max_work, int64_t max_scratch_bytes,
                           uint64_t* work_evidence, uint64_t* memory_evidence, phx_mc_mesh** mesh) {
  return phx::mc::guarded([&] {
    return triangulate_2d(point_count, points, nullptr, false, max_triangles,
                          max_cavity_cells, max_work, max_scratch_bytes,
                          work_evidence, memory_evidence, mesh);
  });
}

int32_t phx_mc_regular_2d(int64_t point_count, const double* points, const double* weights,
                          int64_t max_triangles, int64_t max_cavity_cells, int64_t max_work,
                          int64_t max_scratch_bytes, uint64_t* work_evidence,
                          uint64_t* memory_evidence, phx_mc_mesh** mesh) {
  return phx::mc::guarded([&] {
    return triangulate_2d(point_count, points, weights, true, max_triangles,
                          max_cavity_cells, max_work, max_scratch_bytes,
                          work_evidence, memory_evidence, mesh);
  });
}

}  // extern "C"
