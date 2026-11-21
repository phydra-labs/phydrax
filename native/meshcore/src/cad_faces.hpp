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

// Largest share alpha of a face's theta * tol taken by a curve chord's lens
// (the chart arc vs the chart chord, mapped). The share is per chord: the
// face triangles next to a chord (one layer) keep their own interpolation
// error within theta tol minus that chord's lens, so a fidelity check that
// adds the two per point (V3) holds there; the rest of the face keeps theta
// tol. The curve stage sizes a chord so that layer edges of its own length
// fit: kappa L^2 / 4 + kappa_l L^2 / 8 <= theta tol (kappa_l the curvature
// of the mapped chart arc off its chord), i.e. the chord's share is
// kappa_l / (2 kappa + kappa_l): none on a chart-straight seam, 1/2 at
// kappa_l = 2 kappa. Beyond that (a nearly flat face under a curved trim)
// the lens is capped at alpha theta tol, L <= sqrt(8 alpha theta tol /
// kappa_l), so the layer never drops below (1 - alpha) theta tol.
// Measured on the quadric fixture's chords: kappa_l / kappa_max in
// [1.4, 2.1] on the radius-1 cylinder's faces and in [0, 0.75] (median
// 0.22) on the radius-0.6 one.
inline constexpr double kLensShare = 0.5;

// Exact chart segment contact under the shared-vertex/chord rule above.
bool cad_face_segments_conflict(int32_t a, int32_t b, const std::array<double, 2>& auv,
                                const std::array<double, 2>& buv, int32_t c, int32_t d,
                                const std::array<double, 2>& cuv, const std::array<double, 2>& duv);

// Validates the shared curve discretization once before any face consumes it.
int32_t validate_cad_curve_discretization(const phx_mc_cad_model& model,
                                          const CadCurveDiscretization& curves);

// |k|max of `face` at the middle of its chart box (the end points of its
// boundary uses' p-curves): the curvature with which the face stage spaces
// its pole rings and the curve stage sizes curves inside a pole zone, so
// both take the same theta * tol budget there. False on a failed
// evaluation or a degenerate jet.
bool pole_zone_curvature(const cad::CadGeometry& geometry, int32_t face,
                         cad::EvaluationWorkspace& workspace, double& kappa);

// Meshes one native chart, preserving curve samples and collapsed-pole records.
// The chart kernel's angle pass receives the face position hook, which yields
// the classified corner/curve point at an input boundary chart and the face's
// order-0 evaluation elsewhere; every final face vertex position comes from it.
// position_calls, when given, records each hook result as {u, v, x, y, z}.
// graded_pole_rings (solid route) starts pole rings at the graded base size
// and grows them ring by ring (seed_poles).
int32_t cad_mesh_face(const cad::CadGeometry& geometry,
                      const CadCurveDiscretization& curves, int32_t face,
                      double size, double tolerance, double theta,
                      CadFaceMesh& result,
                      std::vector<std::array<double, 2>>* metric_points = nullptr,
                      std::vector<std::array<double, 5>>* position_calls = nullptr,
                      bool graded_pole_rings = false);

}  // namespace phx::mc
