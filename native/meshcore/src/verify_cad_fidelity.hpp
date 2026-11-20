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

}  // namespace phx::mc::cad
