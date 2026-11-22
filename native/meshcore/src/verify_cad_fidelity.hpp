// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <cstdint>

#include "cad_geometry.hpp"

namespace phx::mc::cad {

struct FidelityTriangleResult {
  double upper;
  int32_t status;
  int32_t reason;
  int32_t subdivisions;
};

FidelityTriangleResult fidelity_triangle(const CadGeometry& geometry, int32_t face,
                                         const double* vertices, const double* uv,
                                         double tolerance, int32_t max_depth,
                                         IntervalEvaluationWorkspace& workspace);

// The row's bound in the regular pole chart (x, y) = rho (cos u, sin u) of
// a closed-form sphere face (verify_cad_pole_chart.hpp): pole_side +1 north,
// -1 south; bit j of pole_mask marks corner j as the pole vertex (exact
// topology, the caller's), placed at the origin. Same result contract as
// fidelity_triangle, without lens charges; PHX_MC_UNRESOLVED with
// PHX_MC_FIDELITY_INTERVAL_REFUSAL for any other surface family.
FidelityTriangleResult fidelity_pole_triangle(const CadGeometry& geometry, int32_t face,
                                              const double* vertices, const double* uv,
                                              int32_t pole_side, int32_t pole_mask,
                                              double tolerance, int32_t max_depth,
                                              IntervalEvaluationWorkspace& workspace);
// The pole-chart image enclosure of a chart point: image = {x lower, x upper,
// y lower, y upper}.
int32_t fidelity_pole_image(const CadGeometry& geometry, int32_t face,
                            IntervalEvaluationWorkspace& workspace, int32_t pole_side,
                            const double* uv, double* image);
// Pole-chart jets over a box: S, S_xx, S_xy, S_yy (12 values).
int32_t fidelity_pole_jets(const CadGeometry& geometry, int32_t face,
                           IntervalEvaluationWorkspace& workspace, int32_t pole_side,
                           const double* lower, const double* upper, double* jet_lower,
                           double* jet_upper);
// The seam lens cap of one pole-region boundary edge from chart corner a
// to b, with pole-chart corners c and mesh endpoints x. If midpoint_node
// is given the physical trace Q is quadratic, otherwise it is affine.
// out = {cap, sup |S_uv(gamma) - Q|, sup |S_xy(Phi) - Q|}.
int32_t fidelity_seam_lens(const CadGeometry& geometry, int32_t face,
                           IntervalEvaluationWorkspace& workspace, int32_t pole_side,
                           const double* uv_a, const double* uv_b, const double* c_a,
                           const double* c_b, const double* x_a, const double* x_b,
                           double* out, const double* midpoint_node = nullptr);

// The per-face kernels fidelity_triangle uses, for tests.
// The anchored cos/sin enclosure over [lower, upper] (fresh anchors):
// out = {cos lower, cos upper, sin lower, sin upper}.
int32_t fidelity_anchored_sincos(double lower, double upper, double* out);
// Closed-form jets (terms 0..5 of the point-jet layout, 18 values) over a
// parameter box; PHX_MC_UNRESOLVED when the face is not a closed form.
int32_t fidelity_closed_jets(const CadGeometry& geometry, int32_t face,
                             IntervalEvaluationWorkspace& workspace, const double* lower,
                             const double* upper, double* jet_lower, double* jet_upper);
// Restricted numerator-net enclosure of S_uu, S_uv, S_vv (9 values) of a
// B-spline face; PHX_MC_UNRESOLVED when no nets apply to the box.
int32_t fidelity_net_hessian(const CadGeometry& geometry, int32_t face,
                             IntervalEvaluationWorkspace& workspace, const double* lower,
                             const double* upper, double* hessian_lower, double* hessian_upper);
// The trim lens cap of one chord of `use` on `face` and its padded UV box
// (u lower, v lower, u upper, v upper).
int32_t fidelity_trim_lens(const CadGeometry& geometry, int32_t face, int64_t use,
                           const double* chord_uv, const double* parameters,
                           IntervalEvaluationWorkspace& workspace, double* cap, double* padded);
// The largest cap of a chord of `face` whose padded box (boxes: u lower,
// v lower, u upper, v upper per chord) meets `box` (the lens charge index).
double fidelity_lens_charge(int64_t surface_count, int64_t chord_count, const int32_t* faces,
                            const double* boxes, const double* caps, int32_t face,
                            const double* box);

}  // namespace phx::mc::cad
