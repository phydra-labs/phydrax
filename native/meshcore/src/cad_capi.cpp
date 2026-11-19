// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "cad_mesh.hpp"
#include "cad_curves.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"

struct phx_mc_cad_surface_result {
  phx::mc::CadSurfaceMesh mesh;
  phx::mc::cad::CurveDiscretization curves;
};

namespace {
template <class T>
void copy_rows(const std::vector<T>& source, T* destination) {
  if (!source.empty()) std::copy(source.begin(), source.end(), destination);
}
template <class T, std::size_t N>
void copy_flat(const std::vector<std::array<T, N>>& source, T* destination) {
  for (std::size_t i = 0; i < source.size(); ++i)
    std::copy(source[i].begin(), source[i].end(), destination + N * i);
}
bool present(const void* p, std::size_t n) { return n == 0 || p != nullptr; }
}  // namespace

extern "C" int32_t phx_mc_cad_mesh(const phx_mc_cad_model* model,
                                     const phx_mc_cad_surface_controls* controls,
                                     phx_mc_cad_surface_result** result) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (!model || !controls || !result) return PHX_MC_INVALID_ARGUMENT;
    *result = nullptr;
    phx::mc::cad::CadGeometry geometry;
    const int32_t bound = geometry.bind(*model);
    if (bound != PHX_MC_OK) return bound;
    auto candidate = std::make_unique<phx_mc_cad_surface_result>();
    const phx::mc::cad::CurveControls curve_controls{
        controls->face_sizes, controls->face_tolerances, controls->curve_tolerances,
        controls->tolerance_fraction, controls->grading, controls->minimum_size};
    const int32_t curve_status = phx::mc::cad::discretize_curves(geometry, curve_controls,
                                                                 candidate->curves);
    if (curve_status != PHX_MC_OK && curve_status != PHX_MC_UNRESOLVED) return curve_status;
    if (candidate->curves.offsets.size() != static_cast<std::size_t>(model->curve_count) + 1)
      return curve_status == PHX_MC_UNRESOLVED ? curve_status : PHX_MC_INTERNAL_ERROR;
    const int32_t surface_status = phx::mc::cad_mesh_surface(
        geometry, candidate->curves.view(), controls->face_sizes, controls->face_tolerances,
        controls->tolerance_fraction, candidate->mesh);
    if (surface_status != PHX_MC_OK && surface_status != PHX_MC_UNRESOLVED) return surface_status;
    *result = candidate.release();
    return curve_status == PHX_MC_UNRESOLVED ? curve_status : surface_status;
  });
}

extern "C" int32_t phx_mc_cad_surface_result_sizes(const phx_mc_cad_surface_result* result,
                                                     int64_t sizes[9]) {
  if (!result || !sizes) return PHX_MC_INVALID_ARGUMENT;
  const auto& m = result->mesh;
  sizes[0] = static_cast<int64_t>(m.points.size());
  sizes[1] = static_cast<int64_t>(m.triangles.size());
  sizes[2] = static_cast<int64_t>(m.sliver_uv.size());
  sizes[3] = static_cast<int64_t>(m.curve_edges.size());
  sizes[4] = static_cast<int64_t>(m.face_stats.size());
  sizes[5] = static_cast<int64_t>(result->curves.offsets.size() - 1);
  sizes[6] = static_cast<int64_t>(result->curves.parameters.size());
  sizes[7] = static_cast<int64_t>(result->curves.statistics.unresolved.size());
  sizes[8] = static_cast<int64_t>(m.face_failures.size());
  return PHX_MC_OK;
}

extern "C" int32_t phx_mc_cad_surface_result_export(const phx_mc_cad_surface_result* result,
                                                      const phx_mc_cad_surface_arrays* a) {
  if (!result || !a) return PHX_MC_INVALID_ARGUMENT;
  const auto& m = result->mesh;
  const auto& curves = result->curves;
  if (a->vertex_capacity < 0 || a->triangle_capacity < 0 || a->sliver_capacity < 0 ||
      a->curve_edge_capacity < 0 || a->face_capacity < 0 || a->curve_capacity < 0 ||
      a->parameter_capacity < 0 || a->failure_capacity < 0 || a->face_failure_capacity < 0)
    return PHX_MC_INVALID_ARGUMENT;
  if (static_cast<uint64_t>(a->vertex_capacity) < m.points.size() ||
      static_cast<uint64_t>(a->triangle_capacity) < m.triangles.size() ||
      static_cast<uint64_t>(a->sliver_capacity) < m.sliver_uv.size() ||
      static_cast<uint64_t>(a->curve_edge_capacity) < m.curve_edges.size() ||
      static_cast<uint64_t>(a->face_capacity) < m.face_stats.size() ||
      static_cast<uint64_t>(a->curve_capacity) < curves.offsets.size() - 1 ||
      static_cast<uint64_t>(a->parameter_capacity) < curves.parameters.size() ||
      static_cast<uint64_t>(a->failure_capacity) < curves.statistics.unresolved.size() ||
      static_cast<uint64_t>(a->face_failure_capacity) < m.face_failures.size())
    return PHX_MC_CAPACITY_EXCEEDED;
  if (!present(a->points,m.points.size()) || !present(a->vertex_dimensions,m.points.size()) ||
      !present(a->vertex_entities,m.points.size()) || !present(a->vertex_parameters,m.points.size()) ||
      !present(a->triangles,m.triangles.size()) || !present(a->triangle_patches,m.triangles.size()) ||
      !present(a->triangle_uv,m.triangles.size()) || !present(a->triangle_edge_uses,m.triangles.size()) ||
      !present(a->facet_regions,m.triangles.size()) || !present(a->sliver_uv,m.sliver_uv.size()) ||
      !present(a->sliver_patches,m.sliver_uv.size()) || !present(a->curve_edges,m.curve_edges.size()) ||
      !present(a->curve_edge_curves,m.curve_edges.size()) ||
      !present(a->curve_edge_parameters,m.curve_edges.size()) || !present(a->face_stats,m.face_stats.size()) ||
      !a->curve_offsets || !present(a->curve_parameters,curves.parameters.size()) ||
      !present(a->curve_sag_splits,curves.statistics.sag_splits_per_curve.size()) ||
      !present(a->curve_failures,curves.statistics.unresolved.size()) || !a->curve_statistics ||
      !present(a->face_failures,m.face_failures.size()))
    return PHX_MC_INVALID_ARGUMENT;
  copy_flat(m.points,a->points); copy_rows(m.vertex_dimensions,a->vertex_dimensions);
  copy_rows(m.vertex_entities,a->vertex_entities); copy_flat(m.vertex_parameters,a->vertex_parameters);
  copy_flat(m.triangles,a->triangles); copy_rows(m.triangle_patches,a->triangle_patches);
  copy_flat(m.triangle_uv,a->triangle_uv); copy_flat(m.triangle_edge_uses,a->triangle_edge_uses);
  copy_flat(m.facet_regions,a->facet_regions); copy_flat(m.sliver_uv,a->sliver_uv);
  copy_rows(m.sliver_patches,a->sliver_patches); copy_flat(m.curve_edges,a->curve_edges);
  copy_rows(m.curve_edge_curves,a->curve_edge_curves); copy_flat(m.curve_edge_parameters,a->curve_edge_parameters);
  for (std::size_t i = 0; i < m.face_stats.size(); ++i) {
    const auto& s = m.face_stats[i];
    a->face_stats[i] = {s.metric_evaluations,s.insertions,s.unresolved_size,s.unresolved_angle,
                        s.encroachment_refusals,s.nonstar_refusals,s.outside_refusals,
                        s.proximity_refusals,s.unresolved_triangles,s.excluded_refusals,
                        s.fixed_triangles_skipped,m.omitted_pole_points[i],
                        m.periodic_corner_gaps[i],m.max_periodic_corner_gap_ulps[i]};
  }
  copy_rows(curves.offsets,a->curve_offsets);
  copy_rows(curves.parameters,a->curve_parameters);
  copy_rows(curves.statistics.sag_splits_per_curve,a->curve_sag_splits);
  for (std::size_t i = 0; i < curves.statistics.unresolved.size(); ++i) {
    const auto& failure = curves.statistics.unresolved[i];
    a->curve_failures[i] = {failure.curve,failure.first,failure.last};
  }
  const auto& s = curves.statistics;
  *a->curve_statistics = {s.first_order_evaluations,s.second_order_evaluations,s.sag_evaluations,
                          s.small_feature_resamples,s.sag_splits,
                          static_cast<int64_t>(curves.parameters.size()),
                          static_cast<int64_t>(s.unresolved.size())};
  copy_rows(m.face_failures,a->face_failures);
  return PHX_MC_OK;
}

extern "C" void phx_mc_cad_surface_result_free(phx_mc_cad_surface_result* result) { delete result; }
