// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// The tube of a certified zero-set surface: the zero set -> mesh half of the
// two-way certificate.
//
// M is a closed, consistently oriented triangle mesh of Z = {f = 0}. Every
// vertex i gets a vector w_i (its angle-weighted normal, oriented along
// grad f, of length L_i), and on each triangle T = (p0, p1, p2) the vectors
// interpolate barycentrically: w(y) = sum lambda_k w_k, continuous over M.
// F(y, t) = y + t w(y), t in [-1, 1], sweeps the tube U = F(M x [-1, 1]).
// Per triangle, over pieces of T x [-1, 1] (midpoint sub-triangles times
// halves of t, refined where a bound is weak) with G an enclosure of grad f
// over a piece's outward swept box:
//  * monotonicity: g_T = min over pieces and their corners' w of
//    lower(G . w) > 0, so along every segment d/dt f(F(y, t)) =
//    grad f . w(y) >= g_T (w(y) is a convex combination of a piece's
//    corners' w);
//  * residual: R_T >= |f| over T (per sub-triangle, the natural enclosure
//    over its box intersected with corner-anchored mean-value forms);
//  * caps: g_T > R_T, so f(F(y, +-1)) has the sign of +-1 and every segment
//    holds exactly one zero, at |t| <= tau_T = R_T / g_T; with g0_T >= g_T
//    the monotonicity bound over T x [-tau_T, tau_T] (a thinner swept box),
//    the zero lies at |t| <= R_T / g0_T: every zero of f in U lies within
//    reverse = max_T (R_T / g0_T) max_k |w_k| of M;
//  * orientation: det[dF/ds, dF/dr, dF/dt] has the orientation's sign on
//    T x [-1, 1] (s, r the barycentric parameters along p1 - p0 and p2 - p0;
//    per piece, intervals over its t and over the box hull of its corners'
//    w), so every point of U off the caps that is a regular value has
//    topological degree of absolute value its preimage count.
// rho: with r = max_k |w_k| and q = r / 2, a cap point c = F(y, +-1) lies
// in its cap box (the outward box of the three p_k +- w_k, the cap being
// the triangle of those points) and has |f(c)| >= g_T - R_T. A point m of M
// with |c - m| < q lies in that cap box widened by q, on a triangle whose
// box meets it, so |f(m)| <= R'_T, the largest R over those triangles, and
// [c, m] lies in the widened box, where |grad f| <= Lip_T: |c - m| >=
// (g_T - R_T - R'_T) / Lip_T. Hence every cap lies at least
// rho = min_T min(q, (g_T - R_T - R'_T) / Lip_T) from M (refused unless
// g_T - R_T - R'_T > 0).
// For x within rho of M, the segment to its nearest point y0 of M stays
// within rho of M, so it meets no cap: the degree of x equals that of points
// near y0, which is nonzero ((y0, 0) is a preimage). So x
// lies in U, and a zero at x lies within `reverse` of M.
// A triangle failing a condition refuses it; the lengths of its corners
// double (caps not clear of the zero) or halve (a sign, orientation or
// enclosure failing; halving wins at shared corners), all L_i starting
// at `length`, for at most kTubeRounds rounds.
#pragma once

#include <cstdint>
#include <vector>

#include "surface_certify.hpp"
#include "mesh_source.hpp"
#include "zero_set_admission.hpp"

namespace phx::mc::remesh {

struct TubeCertificate {
  // Every point within rho of M lies in the tube.
  double rho = 0;
  // Every zero of f in the tube lies within this of M.
  double reverse = 0;
  // Triangles refused in the last round (0 on success).
  int64_t refused = 0;
  // Rounds run (lengths doubled or halved between rounds).
  int64_t rounds = 0;
};

// PHX_MC_OK with `certificate` filled, PHX_MC_UNRESOLVED when a triangle is
// still refused after the last round (certificate.refused counts them),
// PHX_MC_INVALID_INPUT for a mesh that is not a closed oriented 2-manifold.
// `value` and `gradient` enclose f and grad f over boxes. `orientation` is
// +1 when grad f points along (p1 - p0) x (p2 - p0), -1 when against it;
// `length` (> 0) is the initial |w_i|; `tolerance` (> 0) is the reverse
// bound the residual refinement aims for (the certificate holds whatever
// bound it proves).
int32_t certify_tube(const std::vector<Point>& points, const std::vector<Triangle>& triangles,
                     const IntervalBoxProgram& value, const IntervalBoxProgram& gradient,
                     int orientation, double length, double tolerance,
                     TubeCertificate& certificate);

// An upper bound of the distance from every point of the closed box to the
// mesh bound to `mesh` (+inf when it cannot be formed). Derivation: with c
// the computed box centre, every box point x has |x_d - c_d| <= h_d =
// max(c_d - lower_d, upper_d - c_d) (each difference enclosed outward), so
// |x - c| <= |h|. The mesh query names a triangle (p0, p1, p2) near c with
// barycentric weights; s = w1 / W and t = w2 / W (W the sum of the weights
// clamped at 0) are nonnegative doubles, and t is lowered, if needed, to at
// most the downward rounding of 1 - s, so s + t <= 1 exactly and P = p0 +
// s (p1 - p0) + t (p2 - p0) is a point of that triangle, hence of M. |c - P|
// is enclosed by interval arithmetic over P's coordinates. Then d(x, M) <=
// |x - P| <= |h| + |c - P|, both terms upper norms of outward enclosures and
// their sum rounded up. The query's choice of triangle affects only the
// tightness, never the soundness.
double box_reach(const MeshSource& mesh, const Point& lower, const Point& upper);

}  // namespace phx::mc::remesh
