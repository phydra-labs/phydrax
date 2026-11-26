// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "cad_curve_discretization.hpp"
#include "cad_geometry.hpp"
#include "cad_proximity.hpp"
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

// Polar trim chart (V3 class 2). A sphere face whose pole is trimmed away by a
// latitude circle has no pole chart in V3: its rows beside the trim are
// certified in the uv chart, by the chart-linear deviation |S - I_T|, which
// there far exceeds the flat triangle's own (the chart is as anisotropic as
// beside a pole). The curve stage sizes the trim's chords and the face stage
// the band before its first ring by that deviation, in that chart.
//
// The face's surface in its chart is S = o + r (cos v (cos u e1 + sin u e2) +
// sin v a): a sphere template, a placement of one, or the revolution of a
// circle3 whose center and second axis lie on the revolution axis exactly
// (V3's pole_sphere_frame families with a circle profile, in binary64).
// equator = max_u |cos u e1 + sin u e2| (the Gram matrix's largest root),
// pole = |a|, frame = max |x e1 + y e2 + z a| over unit (x, y, z), bounded
// by the square root of the Gram matrix of e1, e2, a's largest absolute row
// sum (Gershgorin; 1 for an orthonormal frame).
struct SphereChart {
  double radius = 0.0;
  double equator = 0.0;
  double pole = 0.0;
  double frame = 0.0;
};
bool sphere_chart(const cad::CadGeometry& geometry, int32_t face, SphereChart& chart);

// Bounds of |S_uu|, |S_uv|, |S_vv| over every u and every v between v0 and
// v1 (either order): r e max|cos v|, r e max|sin v|, r (e max|cos v| +
// |a| max|sin v|), the maxima exact over the interval (S_uu = -r cos v R,
// S_uv = -r sin v R', S_vv = -r (cos v R + sin v a), |R|, |R'| <= e).
void sphere_chart_hessian(const SphereChart& chart, double v0, double v1, double bound[3]);

// The band bound. For x in a triangle with chart corners x_i, Taylor at x
// gives S(x_i) = S(x) + DS (x_i - x) + R_i and sum_i lambda_i (x_i - x) = 0,
// so |S(x) - I_T(x)| <= sum_i lambda_i |R_i| <= sum_i lambda_i q(x_i - x) / 2
// with q(d) = A d_u^2 + 2 B |d_u d_v| + C d_v^2 (A, B, C the bounds above over
// the triangle). Per axis sum_i lambda_i (x_i - x)_k^2 is a variance over the
// triangle's extent D_k, at most D_k^2 / 4, and the mixed sum is at most
// D_u D_v / 4 (Cauchy-Schwarz): the deviation is at most (A D_u^2 + 2 B D_u
// D_v + C D_v^2) / 8. A band triangle spans one trim chord on u (D_u, the
// band ring is aligned to the trim's vertices) and the band on v (D_v).
//
// polar_trim_band: the largest band D_v from the trim line v = line toward
// `direction` (the face side) with that bound within `budget` / 8 for
// chords of span `span`: first with the bounds at the line alone (D_v = b0),
// then over [line, line + direction b0] (D_v = b <= b0, so the bounds over
// [0, b] are within those used). Infinity when none fits (A span^2 >=
// budget).
double polar_trim_band(const SphereChart& chart, double line, double direction, double span,
                       double budget);

// polar_row_span: the largest free-axis extent D_u of a row between
// latitudes v0 and v1 (D_v = |v1 - v0|) with that bound, the bounds over
// [v0, v1], within `budget` / 8; 0 when none fits (C D_v^2 >= budget).
double polar_row_span(const SphereChart& chart, double v0, double v1, double budget);

// polar_trim_span: the trim chord span the curve stage takes. The trim's
// own term A D_u^2 and the band's 2 B D_u D_v + C D_v^2 share one budget;
// the split takes the band triangle isotropic in 3D, its band height equal
// to its trim chord: D_v |S_v| = D_u |S_u|, i.e. D_v = aspect D_u with aspect
// = |S_u| / |S_v| at the line (a larger band shortens the trim's chords
// below the band's height, a smaller one leaves a band sliver). The span is
// D_u = sqrt(budget / (A + 2 B aspect + C aspect^2)), first with the bounds
// at the line alone, then over [line, line + direction aspect D_u0]: the
// second span is at most the first, so its band lies in that interval.
// Infinity when A or aspect is 0.
double polar_trim_span(const SphereChart& chart, double line, double direction, double aspect,
                       double budget);

// A use of `face` on a polar trim line (as the face stage's anchors): the
// face is a sphere chart (sphere_chart), its chart box (its uses' p-curve
// end points) spans one whole turn on u and has the use's constant v = line
// as a side, there |S_u|^2 < |S_v|^2 / 2, and the face lies on the
// equator's side of the line (direction * line < 0), so the pole beyond the
// line is trimmed away (a pole on the face's side keeps V3's pole chart
// over the trim's rows). direction is the face's side of the line. The
// p-curve must be constant in v.
bool polar_trim_line(const cad::CadGeometry& geometry, int32_t face, double line,
                     cad::EvaluationWorkspace& workspace, SphereChart& chart,
                     double& direction);

// Meshes one native chart, preserving curve samples and collapsed-pole records.
// The chart kernel's angle pass receives the face position hook, which yields
// the classified corner/curve point at an input boundary chart and the face's
// order-0 evaluation elsewhere; every final face vertex position comes from it.
// position_calls, when given, records each hook result as {u, v, x, y, z}.
// graded_pole_rings (solid route) starts pole rings at the graded base size
// and grows them ring by ring (seed_poles). proximity (solid route, null:
// none) bounds the base size at each face point (cad_proximity.hpp).
int32_t cad_mesh_face(const cad::CadGeometry& geometry,
                      const CadCurveDiscretization& curves, int32_t face,
                      double size, double tolerance, double theta,
                      CadFaceMesh& result,
                      std::vector<std::array<double, 2>>* metric_points = nullptr,
                      std::vector<std::array<double, 5>>* position_calls = nullptr,
                      bool graded_pole_rings = false,
                      const cad::ProximityField* proximity = nullptr);

// Opaque Phase-1 face refinement state (boundary, metric, pole seeds, and 2D
// chart refinement, prior to 3D flip sweep and smoothing) so retryable surface
// passes can skip smoothing when any face triggers a shared-curve retry.
struct CadFaceStage;
struct CadFaceStageDeleter {
  void operator()(CadFaceStage* stage) const noexcept;
};
using CadFaceStagePtr = std::unique_ptr<CadFaceStage, CadFaceStageDeleter>;

int32_t cad_refine_face_stage(const cad::CadGeometry& geometry,
                              const CadCurveDiscretization& curves, int32_t face,
                              double size, double tolerance, double theta,
                              CadFaceStagePtr& stage, CadFaceMesh& early_result,
                              bool& has_curve_encroachment,
                              bool graded_pole_rings = false,
                              const cad::ProximityField* proximity = nullptr);

std::size_t cad_face_stage_bytes(const CadFaceStage& stage) noexcept;
const std::vector<phx_mc_cad_face_failure>& cad_face_stage_failures(
    const CadFaceStage& stage) noexcept;

int32_t cad_finish_face_stage(CadFaceStage& stage, const CadCurveDiscretization& curves,
                              CadFaceMesh& result);

}  // namespace phx::mc
