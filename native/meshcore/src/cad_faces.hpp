// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "cad_curve_discretization.hpp"
#include "cad_geometry.hpp"
#include "cad_refine2d.hpp"

namespace phx::mc {

struct CadFaceVertex {
  std::array<double, 2> uv;
  int32_t dimension;
  int32_t entity;
  int32_t curve_index;
};

struct CadFaceEdge {
  std::array<int32_t, 2> vertices;
  int32_t use;
  int32_t curve;
  std::array<double, 2> parameters;
};
// Snapping uses a pre-existing adjacent curve sample at every use through the
// pole only when their exact offsets agree, within [spacing/2, spacing] and
// within the computed pole-zone width. The lower limit is the existing
// half-metric-unit boundary clearance; the upper preserves the original spacing.
// A ring edge or fan triangle may meet a trim/seam chord only at shared chart
// vertex IDs, or (for a triangle) along the identical chord edge. All other
// contacts, including non-shared touches and partial overlaps, remain free.
struct CadPoleSectors {
  int32_t use;
  int64_t fixed;
  int64_t free;
};


struct CadFaceMesh {
  std::vector<CadFaceVertex> vertices;
  std::vector<std::array<int32_t, 3>> triangles;
  std::vector<std::array<int32_t, 3>> slivers;
  std::vector<CadFaceEdge> boundary_edges;
  CadRefine2DStats stats;
  int64_t omitted_pole_points = 0;
  std::vector<CadPoleSectors> pole_sectors;
  std::vector<std::array<int32_t, 2>> protected_pole_edges;
  int64_t periodic_corner_gaps = 0;
  uint64_t max_periodic_corner_gap_ulps = 0;
  phx_mc_cad_face_failure failure{};
  std::vector<phx_mc_cad_face_failure> refinement_failures;
};

// Width of a pole's fixed-ring zone for ring spacing `spacing`: the face
// stage seeds rings at least this far from a pole (further while the chart
// stays anisotropic, the whole face when it is narrower) and the chart
// kernel inserts no point there. The curve stage reverts a curve's
// curvature-shrunk size to the request within it (no kernel triangle there
// needs the curvature size).
inline double pole_ring_width(double spacing) { return 3.0 * spacing; }

// Share alpha of a face's theta * tol taken by a curve chord's lens (the
// chart arc vs the chart chord, mapped); the face triangles that touch a
// curve (one layer) keep their own interpolation error within the rest,
// (1 - alpha) theta tol, so a fidelity check that adds the two per point
// (V3) holds there; the rest of the face keeps theta tol. Derivation: per
// unit curve length the boundary layer holds n = 1/L_c + 1/e vertices, with
// chords L_c = sqrt(8 alpha theta tol / kappa_l) (lens ~ kappa_l L^2 / 8,
// kappa_l the curvature of the mapped chart arc off its chord) and layer
// edges e = sqrt(3) R, R = sqrt(2 (1 - alpha) theta tol / kappa) (own error
// ~ kappa R^2 / 2). n ~ a / sqrt(alpha) + b / sqrt(1 - alpha) with
// a / b = sqrt(3 kappa_l / (4 kappa)) is smallest at alpha / (1 - alpha) =
// (a / b)^(2/3), and n(1/2) / n_min = sqrt(2) (a + b) / (a^(2/3) +
// b^(2/3))^(3/2) <= 1.03 for a / b in [0.41, 2.4]. Measured on the quadric
// fixture's chords: kappa_l / kappa_max in [1.4, 2.1] on the radius-1
// cylinder's faces and in [0, 0.75] (median 0.22) on the radius-0.6 one,
// i.e. a / b up to 1.26 (median 0.41 on the latter; below it the lens
// is small and its share idle): the minimum is flat, so alpha = 1/2.
inline constexpr double kLensShare = 0.5;

// Exact chart segment contact under the shared-vertex/chord rule above.
bool cad_face_segments_conflict(int32_t a, int32_t b, const std::array<double, 2>& auv,
                                const std::array<double, 2>& buv, int32_t c, int32_t d,
                                const std::array<double, 2>& cuv, const std::array<double, 2>& duv);

// Validates the shared curve discretization once before any face consumes it.
int32_t validate_cad_curve_discretization(const phx_mc_cad_model& model,
                                          const CadCurveDiscretization& curves);

// Meshes one native chart, preserving curve samples and collapsed-pole records.
// The chart kernel's angle pass receives the face position hook, which yields
// the classified corner/curve point at an input boundary chart and the face's
// order-0 evaluation elsewhere; every final face vertex position comes from it.
// position_calls, when given, records each hook result as {u, v, x, y, z}.
int32_t cad_mesh_face(const cad::CadGeometry& geometry,
                      const CadCurveDiscretization& curves, int32_t face,
                      double size, double tolerance, double theta,
                      CadFaceMesh& result,
                      std::vector<std::array<double, 2>>* metric_points = nullptr,
                      std::vector<std::array<double, 5>>* position_calls = nullptr);

}  // namespace phx::mc
