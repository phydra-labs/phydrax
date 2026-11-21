// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "cad_mesh.hpp"
#include "cad_curves.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <vector>

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"

struct phx_mc_cad_surface_result {
  phx::mc::CadSurfaceMesh mesh;
  phx::mc::cad::CurveDiscretization curves;
  phx::mc::CadVolumeMesh volume;
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

// Curve and face stages into `candidate` (faces at the request's size h,
// refined where its tolerance needs it), with their stage times. A solid
// passes its minimum dihedral target as rim_dihedral_degrees (CurveControls;
// 0 for a surface), which also grades its pole rings (cad_mesh_face).
// Returns OK/UNRESOLVED with a candidate, else the failure.
int32_t mesh_surface(const phx::mc::cad::CadGeometry& geometry, const phx_mc_cad_model& model,
                     const phx_mc_cad_surface_controls& controls, double rim_dihedral_degrees,
                     phx_mc_cad_surface_result& candidate) {
  const double* face_sizes = controls.face_sizes;
  phx::mc::cad::CurveControls curve_controls{
      face_sizes, controls.face_tolerances, controls.curve_tolerances,
      controls.tolerance_fraction, controls.grading, controls.minimum_size};
  curve_controls.rim_dihedral_degrees = rim_dihedral_degrees;
  const auto start = std::chrono::steady_clock::now();
  const int32_t curve_status = phx::mc::cad::discretize_curves(geometry, curve_controls,
                                                               candidate.curves);
  candidate.volume.stats.curve_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - start).count();
  if (curve_status != PHX_MC_OK && curve_status != PHX_MC_UNRESOLVED) return curve_status;
  if (candidate.curves.offsets.size() != static_cast<std::size_t>(model.curve_count) + 1)
    return curve_status == PHX_MC_UNRESOLVED ? curve_status : PHX_MC_INTERNAL_ERROR;
  const int32_t surface_status = phx::mc::cad_mesh_surface(
      geometry, candidate.curves.view(), face_sizes, controls.face_tolerances,
      controls.tolerance_fraction, candidate.mesh, rim_dihedral_degrees > 0.0);
  candidate.volume.stats.face_ns = candidate.mesh.face_ns;
  candidate.volume.stats.assembly_ns = candidate.mesh.assembly_ns;
  if (surface_status != PHX_MC_OK && surface_status != PHX_MC_UNRESOLVED) return surface_status;
  return curve_status == PHX_MC_UNRESOLVED ? curve_status : surface_status;
}
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
    const int32_t status = mesh_surface(geometry, *model, *controls, 0.0, *candidate);
    if (status != PHX_MC_OK && status != PHX_MC_UNRESOLVED) return status;
    *result = candidate.release();
    return status;
  });
}

extern "C" int32_t phx_mc_cad_mesh_solid(const phx_mc_cad_model* model,
                                           const phx_mc_cad_surface_controls* controls,
                                           const phx_mc_cad_volume_controls* volume,
                                           phx_mc_cad_surface_result** result) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (!model || !controls || !volume || !result) return PHX_MC_INVALID_ARGUMENT;
    *result = nullptr;
    if (volume->geometry_order != 1 && volume->geometry_order != 2)
      return PHX_MC_INVALID_ARGUMENT;
    phx::mc::cad::CadGeometry geometry;
    const int32_t bound = geometry.bind(*model);
    if (bound != PHX_MC_OK) return bound;
    auto candidate = std::make_unique<phx_mc_cad_surface_result>();
    const int32_t status =
        mesh_surface(geometry, *model, *controls, volume->min_dihedral_degrees, *candidate);
    if (status != PHX_MC_OK && status != PHX_MC_UNRESOLVED) return status;
    if (status == PHX_MC_OK) {
      const phx::mc::CadVolumeControls volume_controls{
          volume->radius_edge_bound, volume->min_dihedral_degrees,
          volume->min_relative_determinant, volume->geometry_order};
      const phx_mc_cad_volume_stats timings = candidate->volume.stats;
      const int32_t solid = phx::mc::cad_mesh_volume(geometry, candidate->curves, candidate->mesh,
                                                     controls->face_sizes, volume_controls,
                                                     candidate->volume);
      auto& stats = candidate->volume.stats;
      stats.curve_ns = timings.curve_ns;
      stats.face_ns = timings.face_ns;
      stats.assembly_ns = timings.assembly_ns;
      if (solid != PHX_MC_OK && solid != PHX_MC_UNRESOLVED) return solid;
      *result = candidate.release();
      return solid;
    }
    *result = candidate.release();
    return status;
  });
}

extern "C" int32_t phx_mc_cad_volume_sizes(const phx_mc_cad_surface_result* result,
                                             int64_t sizes[6]) {
  if (!result || !sizes) return PHX_MC_INVALID_ARGUMENT;
  const auto& v = result->volume;
  sizes[0] = static_cast<int64_t>(v.points.size());
  sizes[1] = static_cast<int64_t>(v.tetrahedra.size());
  sizes[2] = static_cast<int64_t>(v.boundary_faces.size());
  sizes[3] = static_cast<int64_t>(v.quadratic.edges.size());
  sizes[4] = static_cast<int64_t>(v.quadratic.straightened.size());
  sizes[5] = static_cast<int64_t>(v.splits.size());
  return PHX_MC_OK;
}

extern "C" int32_t phx_mc_cad_volume_export(const phx_mc_cad_surface_result* result,
                                              const phx_mc_cad_volume_arrays* a) {
  if (!result || !a) return PHX_MC_INVALID_ARGUMENT;
  const auto& v = result->volume;
  const auto& q = v.quadratic;
  const std::size_t np = v.points.size(), nt = v.tetrahedra.size(), nb = v.boundary_faces.size(),
                    ne = q.edges.size(), nr = q.straightened.size(), ns = v.splits.size();
  const std::size_t nc = ne ? nt : 0;
  if (a->point_capacity < 0 || a->tetrahedron_capacity < 0 || a->triangle_capacity < 0 ||
      a->edge_capacity < 0 || a->record_capacity < 0 || a->split_capacity < 0)
    return PHX_MC_INVALID_ARGUMENT;
  if (static_cast<uint64_t>(a->point_capacity) < np ||
      static_cast<uint64_t>(a->tetrahedron_capacity) < nt ||
      static_cast<uint64_t>(a->triangle_capacity) < nb ||
      static_cast<uint64_t>(a->edge_capacity) < ne ||
      static_cast<uint64_t>(a->record_capacity) < nr ||
      static_cast<uint64_t>(a->split_capacity) < ns)
    return PHX_MC_CAPACITY_EXCEEDED;
  if (!present(a->points, np) || !present(a->tetrahedra, nt) ||
      !present(a->tetrahedron_regions, nt) || !present(a->boundary_faces, nb) ||
      !present(a->edges, ne) || !present(a->nodes, ne) || !present(a->edge_kinds, ne) ||
      !present(a->edge_entities, ne) || !present(a->cell_edges, nc) ||
      !present(a->cell_decisions, nc) || !present(a->records, nr) || !present(a->splits, ns) ||
      !a->stats)
    return PHX_MC_INVALID_ARGUMENT;
  copy_flat(v.points, a->points); copy_flat(v.tetrahedra, a->tetrahedra);
  copy_rows(v.regions, a->tetrahedron_regions); copy_flat(v.boundary_faces, a->boundary_faces);
  copy_flat(q.edges, a->edges); copy_flat(q.nodes, a->nodes);
  copy_rows(q.edge_kinds, a->edge_kinds); copy_rows(q.edge_entities, a->edge_entities);
  if (nc) {
    copy_flat(q.cell_edges, a->cell_edges);
    copy_rows(q.cell_decisions, a->cell_decisions);
  }
  copy_flat(q.straightened, a->records); copy_rows(v.splits, a->splits);
  *a->stats = v.stats;
  return PHX_MC_OK;
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

extern "C" int32_t phx_mc_cad_surface_pole_sectors(
    const phx_mc_cad_surface_result* result, int64_t capacity,
    int64_t* rows, int64_t* count) {
  if (!result || !count || capacity < 0) return PHX_MC_INVALID_ARGUMENT;
  const auto& sectors = result->mesh.pole_sectors;
  const int64_t required = static_cast<int64_t>(sectors.size());
  if (capacity == 0 && !rows) {
    *count = required;
    return PHX_MC_OK;
  }
  if (capacity < required || !present(rows, sectors.size())) return PHX_MC_INVALID_ARGUMENT;
  copy_flat(sectors, rows);
  *count = required;
  return PHX_MC_OK;
}

extern "C" int32_t phx_mc_cad_chart_sizes(const phx_mc_cad_surface_result* result,
                                            int64_t sizes[3]) {
  if (!result || !sizes) return PHX_MC_INVALID_ARGUMENT;
  const auto& chart = result->mesh.chart;
  sizes[0] = static_cast<int64_t>(chart.triangles.size());
  sizes[1] = static_cast<int64_t>(chart.use_patch.size());
  sizes[2] = static_cast<int64_t>(chart.chord_use.size());
  return PHX_MC_OK;
}

extern "C" int32_t phx_mc_cad_chart_export(const phx_mc_cad_surface_result* result,
                                             const phx_mc_cad_chart_arrays* a) {
  if (!result || !a || a->triangle_capacity < 0 || a->use_capacity < 0 ||
      a->chord_capacity < 0) return PHX_MC_INVALID_ARGUMENT;
  const auto& c = result->mesh.chart;
  const auto nt = c.triangles.size(), nu = c.use_patch.size(), nc = c.chord_use.size();
  if (static_cast<uint64_t>(a->triangle_capacity) < nt ||
      static_cast<uint64_t>(a->use_capacity) < nu ||
      static_cast<uint64_t>(a->chord_capacity) < nc) return PHX_MC_CAPACITY_EXCEEDED;
  if (!present(a->triangles, nt) || !present(a->patches, nt) || !present(a->uv, nt) ||
      !present(a->chart_to_mesh, nt) || !present(a->edge_chord, nt) ||
      !present(a->use_patch, nu) || !present(a->use_range, nu) ||
      !present(a->chord_use, nc) || !present(a->chord_loop, nc) ||
      !present(a->chord_uv, nc) || !present(a->chord_range, nc))
    return PHX_MC_INVALID_ARGUMENT;
  copy_flat(c.triangles, a->triangles); copy_rows(c.patches, a->patches);
  copy_flat(c.uv, a->uv); copy_rows(c.chart_to_mesh, a->chart_to_mesh);
  copy_flat(c.edge_chord, a->edge_chord); copy_rows(c.use_patch, a->use_patch);
  copy_flat(c.use_range, a->use_range); copy_rows(c.chord_use, a->chord_use);
  copy_rows(c.chord_loop, a->chord_loop); copy_flat(c.chord_uv, a->chord_uv);
  copy_flat(c.chord_range, a->chord_range);
  return PHX_MC_OK;
}
