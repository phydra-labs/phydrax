// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "cad_curves.hpp"
#include "cad_curving.hpp"
#include "cad_faces.hpp"

namespace phx::mc {

// Verifier input in chart-triangle order. Triangle vertex IDs refer to points
// in the assembled mesh, including repeated pole IDs in collapsed slivers.
struct CadChartEvidence {
  std::vector<std::array<int64_t, 3>> triangles;
  std::vector<int32_t> patches;
  std::vector<std::array<double, 6>> uv;
  std::vector<int64_t> chart_to_mesh;
  std::vector<std::array<int64_t, 3>> edge_chord;
  std::vector<int64_t> use_patch;
  std::vector<std::array<double, 2>> use_range;
  std::vector<int64_t> chord_use, chord_loop;
  std::vector<std::array<double, 4>> chord_uv;
  std::vector<std::array<double, 2>> chord_range;
};


struct CadSurfaceMesh {
  std::vector<std::array<double, 3>> points;
  std::vector<int8_t> vertex_dimensions;
  std::vector<int32_t> vertex_entities;
  std::vector<std::array<double, 2>> vertex_parameters;
  std::vector<std::array<int32_t, 3>> triangles;
  std::vector<int32_t> triangle_patches;
  std::vector<std::array<double, 6>> triangle_uv;
  std::vector<std::array<int32_t, 3>> triangle_edge_uses;
  std::vector<std::array<int32_t, 2>> facet_regions;
  std::vector<std::array<double, 6>> sliver_uv;
  std::vector<int32_t> sliver_patches;
  std::vector<std::array<int32_t, 2>> curve_edges;
  std::vector<int32_t> curve_edge_curves;
  std::vector<std::array<double, 2>> curve_edge_parameters;
  std::vector<CadRefine2DStats> face_stats;
  std::vector<int64_t> omitted_pole_points;
  // Rows are (face, pole use, protected sectors, free sectors).
  std::vector<std::array<int64_t, 4>> pole_sectors;
  std::vector<int64_t> periodic_corner_gaps;
  std::vector<uint64_t> max_periodic_corner_gap_ulps;
  CadChartEvidence chart;
  std::vector<phx_mc_cad_face_failure> face_failures;
  // Time spent in the per-face chart meshing (cad_mesh_face) and in the rest
  // of the assembly, nanoseconds.
  int64_t face_ns = 0, assembly_ns = 0;
  // Set when a retryable pass stopped after Phase-1 chart refinement because
  // at least one face reported a curve-segment encroachment refusal.
  bool deferred_encroachment = false;
};

// Volume stage controls: affine and sampled curved shape targets, geometry
// order, and the optional proximity field that bounds each surface size.
struct CadVolumeControls {
  double radius_edge_bound = 2.0;
  double minimum_dihedral_degrees = 15.5;
  double minimum_relative_determinant = 0.0;
  int32_t geometry_order = 1;
  const cad::ProximityField* proximity = nullptr;
  std::uint32_t parallelism = 1;
  double sampled_minimum_scaled_jacobian = 0.0;
  double sampled_minimum_determinant_ratio = 0.0;
  double maximum_dihedral_degrees = 180.0;
};

// The fixed-boundary tetrahedralization of the assembled surface. Points
// begin with the surface points, unchanged and in the same rows. Cells are
// right-handed: det[b - a, c - a, d - a] > 0 (orient3d > 0), as S3 keeps them.
// boundary_faces holds, per surface triangle, the cell and the local face
// (opposite that local vertex) equal to it.
struct CadVolumeMesh {
  std::vector<std::array<double, 3>> points;
  std::vector<std::array<int32_t, 4>> tetrahedra;
  std::vector<int32_t> regions;
  std::vector<std::array<int32_t, 2>> boundary_faces;
  CadQuadraticMesh quadratic;
  CadCubicMesh cubic;
  std::vector<phx_mc_cad_boundary_split> splits;
  phx_mc_cad_volume_stats stats{};
  // Per model curve (empty when the volume stage stopped before quality):
  // the smallest true wedge in degrees between its two faces, filled only
  // when some unmet cell is not forced (count_forced_cells), else NaN.
  std::vector<double> curve_rim_wedges;
};

// Test hook: refuse the first recovery as if it needed a boundary Steiner
// point on the surface edge (a, b), so the split-and-retry path runs.
struct CadForcedRefusal {
  int32_t a = -1, b = -1;
};

// Splits the surface edge (a, b) on the source in both incident triangles:
// a curve edge at the curve parameter midpoint (the p-curve gives each face
// its UV), a face edge at the UV midpoint of its triangle records.
int32_t cad_split_surface_edge(const cad::CadGeometry& geometry, cad::CurveDiscretization& curves,
                               CadSurfaceMesh& mesh, int32_t a, int32_t b,
                               phx_mc_cad_boundary_split& record);

// Volume stage (design B.7) and, for geometry order 2, P2 nodes (B.8).
// `sizes` are the requested face edge lengths h. OK: tetrahedra meeting
// every target; UNRESOLVED: a complete mesh with unmet quality targets or
// P2 cells, or no mesh when recovery refused: a surface triangle flat to
// rounding (S3's INVALID_FACET, the triangle row in failure_entity and its
// smallest-angle sine in failure_measure) or a refusal persisting after the
// one boundary-split retry (stats.recovery_status, recovery_reason,
// failure_kind/entity).
int32_t cad_mesh_volume(const cad::CadGeometry& geometry, cad::CurveDiscretization& curves,
                        CadSurfaceMesh& surface, const double* sizes,
                        const CadVolumeControls& controls, CadVolumeMesh& result,
                        const CadForcedRefusal* forced = nullptr);

// Retained chart-local results for one native surface call and its retries.
// The caller keeps geometry and face controls fixed across the passes.
// An entry holds either a completed `mesh` or a Phase-1 `stage` (when a prior
// retryable pass deferred smoothing on curve encroachment).
struct CadFaceRetryWindow {
  static constexpr std::size_t capacity = 8;
  struct Entry {
    CadFaceMesh mesh;
    CadFaceStagePtr stage;
    std::size_t retained_bytes = 0;
  };
  std::vector<Entry> faces;
  std::size_t retained_bytes = 0;
};

// graded_pole_rings and proximity (null: none): solid route (cad_mesh_face).
// When `defer_on_encroachment` is true, Phase-1 2D chart refinement runs on
// all faces first; if any face reports a curve-segment encroachment refusal,
// 3D flip sweeps and smoothing are skipped, resolved Phase-1 stages are
// retained in `retry`, `result.face_failures` receives the Phase-1 failures,
// and `result.deferred_encroachment` is set to true.
int32_t cad_mesh_surface(const cad::CadGeometry& geometry, const CadCurveDiscretization& curves,
                         const double* sizes, const double* tolerances, double theta,
                         CadSurfaceMesh& result, bool graded_pole_rings = false,
                         const cad::ProximityField* proximity = nullptr,
                         int32_t parallelism = 1, CadFaceRetryWindow* retry = nullptr,
                         const CadCurveDiscretization* previous_curves = nullptr,
                         bool defer_on_encroachment = false);

}  // namespace phx::mc
