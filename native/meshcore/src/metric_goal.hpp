//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Floating-point goals of metric mesh edits on the deterministic binary64
// kernels (deterministic_float.hpp): one result per input on every platform.
//
// A goal is a point an edit would like to reach: where to split an edge, or
// where to put the survivor of a collapse. It proposes only. The caller rounds
// it to an exact source point and decides the edit with exact predicates and
// the measured objective.
#pragma once

#include <cstdint>

namespace phx::mc::metric {

// Parameter t of the point p + t (q - p) at half the metric length of the
// segment when the size varies geometrically along it:
// log1p((r - 1) / 2) / log1p(r - 1) with r = sqrt(d^T M_q d / d^T M_p d),
// d = q - p; 1/2 when r is 1. Also 1/2 when a form is not positive and
// finite, their sum is not finite, or r is zero or not finite: such a segment
// has no metric midpoint and the arithmetic one stands in.
double midpoint_parameter(const double* p, const double* q, const double* metric_p,
                          const double* metric_q) noexcept;

// Center of the smallest ball, in the metric M, that holds `count` >= 1
// points (three values each). The points are mapped by the Cholesky factor of
// M, visited in a fixed pseudo-random order (an integer generator, the same
// for every platform and every call with this count), and Welzl's recursion
// is run with an explicit stack: a point outside the current ball by more
// than 32 * 2^-52 times the squared largest mapped coordinate joins the
// boundary set, of at most four points. The ball of a boundary set is the
// smallest, over its subsets, of the circumscribed ball of the subset that
// holds the set. False, with `center` untouched, when there is no point, M
// has no Cholesky factor, or a mapped point, the square of the largest mapped
// coordinate or a center is not finite.
// Allocates its working arrays and may throw std::bad_alloc.
bool enclosing_center(std::int64_t count, const double* points, const double* metric,
                      double* center);

// Orthogonal projection of `goal` onto the affine hull of two or three
// corners (three values each): corner 0 plus the least-squares combination of
// the edges from corner 0, of minimum norm when they are dependent. False,
// with `projected` untouched, when a value is not finite.
bool hull_projection(int corner_count, const double* corners, const double* goal,
                     double* projected) noexcept;

// The three goals of the survivor of a collapse, in the order they are tried.
// `points` holds the survivor, the removed vertex and then the `count` - 2
// vertices of the merged link; `metrics` the tensors of the first two. With
// t the midpoint parameter of the collapsed edge, the goals are the points of
// that edge at t and at t / 2, and the enclosing center of the link in the
// log-Euclidean mean of the two tensors, which minimizes the longest spoke.
// With `corner_count` 2 or 3 every goal is projected onto the hull of those
// corners, the survivor's carrier. valid[g] is false, with goal g zero, when
// a step of that goal fails. May throw std::bad_alloc.
void survivor_goals(std::int64_t count, const double* points, const double* metrics,
                    int corner_count, const double* corners, double* goals, bool* valid);

}  // namespace phx::mc::metric
