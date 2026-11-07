# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Fixed-branch homogeneous Hermite bridges between rational spline boundaries.

The bridge's first parameter runs along the seam on [0, 1], and its second
runs from the left source to the right source on [0, 1].  An along map gives
source parameters at bridge parameters zero and one.  Collar lengths measure
inward source-parameter distance per unit bridge transverse parameter: the
left bridge jet points outward, the right bridge jet points inward.

Only full along-domain affine maps with exactly matching pulled-back expanded
knots and degrees are supported.  No knot refinement or approximation occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial
from typing import cast, Literal

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer

from ..._interpolation import bspline_jet_stencil
from ..._interpolation._bspline_extraction import (
    _bernstein_derivative,
    _univariate_span_extraction,
)
from ...discretization._exact_algebraic_geometry import ExactScalar
from ...ein import contract
from ...linalg._exact import (
    ExactLinearFactorization,
    ExactLinearSolveEvidence,
    factor_exact,
)
from ...typing import parse
from ._patches import _require_exact_numeric_shadow, BSplineSurfacePatch


def _hermite_factor(continuity: int) -> ExactLinearFactorization[Fraction]:
    degree = 2 * continuity + 1
    rows: list[list[Fraction]] = []
    for high in (False, True):
        for order in range(continuity + 1):
            row = [Fraction(0)] * (degree + 1)
            scale = factorial(degree) // factorial(degree - order)
            for i in range(order + 1):
                row[(degree - order if high else 0) + i] = Fraction(
                    scale * (-1) ** (order - i) * comb(order, i)
                )
            rows.append(row)
    return factor_exact(rows)


Side = Literal["u0", "u1", "v0", "v1"]


def _fraction(value: object, name: str) -> Fraction:
    if not isinstance(value, Fraction):
        raise TypeError(f"{name} must be an exact Fraction.")
    return value


def _endpoint_extraction(
    knots: tuple[Fraction, ...], degree: int, count: int, high: bool, order: int
) -> tuple[int, list[list[ExactScalar]]]:
    """Use the native exact span extraction and Bernstein derivative owner."""
    boundary = knots[count] if high else knots[degree]
    span = (
        next(i for i in range(count - 1, degree - 1, -1) if knots[i] < boundary)
        if high
        else next(i for i in range(degree, count) if knots[i + 1] > boundary)
    )
    extraction = _univariate_span_extraction(knots, degree, span)
    jets: list[list[ExactScalar]] = [
        [Fraction(0) for _ in range(count)] for _ in range(order + 1)
    ]
    for local, polynomial in enumerate(extraction):
        for k in range(order + 1):
            jets[k][span - degree + local] = polynomial[-1 if high else 0]
            polynomial = _bernstein_derivative(polynomial, (knots[span], knots[span + 1]))
    return span, jets


def _source(
    patch: BSplineSurfacePatch,
    side: Side,
    along: tuple[Fraction, Fraction],
    collar: Fraction,
    order: int,
) -> tuple[int, tuple[Fraction, ...], list[list[list[ExactScalar]]], list[Array], Array]:
    if not isinstance(patch, BSplineSurfacePatch):
        raise TypeError("Bridge sources must be BSplineSurfacePatch instances.")
    side = parse(side, Side, "source side")
    if not isinstance(along, tuple) or len(along) != 2:
        raise TypeError("An along map must be a pair of Fractions.")
    a, b = (_fraction(x, "along map") for x in along)
    collar = _fraction(collar, "collar")
    if collar <= 0 or a == b:
        raise ValueError("Collars must be positive and along maps nonconstant.")
    cp, w = patch.exact_control_point_bank, patch.exact_weight_bank
    uk, vk = patch.exact_u_knot_bank, patch.exact_v_knot_bank
    if cp is None or w is None or uk is None or vk is None:
        raise ValueError("Bridge construction requires exact source banks.")
    if any(
        not isinstance(x, Fraction) for row in cp for point in row for x in point
    ) or any(not isinstance(x, Fraction) for row in w for x in row):
        raise TypeError("Bridge construction requires rational source coefficients.")
    transverse_u = side[0] == "u"
    ak, tk = (vk, uk) if transverse_u else (uk, vk)
    ad, td = (
        (patch.v_degree, patch.u_degree)
        if transverse_u
        else (patch.u_degree, patch.v_degree)
    )
    na, nt = (
        (patch.control_points.shape[1], patch.control_points.shape[0])
        if transverse_u
        else patch.control_points.shape[:2]
    )
    if (min(a, b), max(a, b)) != (ak[ad], ak[na]):
        raise ValueError("Only full along-domain affine maps are supported.")
    reverse = b < a
    pulled = tuple((x - a) / (b - a) for x in (reversed(ak) if reverse else ak))
    # Exact banks in along-major, transverse-minor order.
    exact_h = [
        [
            tuple(cp[j][i][k] * w[j][i] for k in range(3)) + (w[j][i],)
            if transverse_u
            else tuple(cp[i][j][k] * w[i][j] for k in range(3)) + (w[i][j],)
            for j in range(nt)
        ]
        for i in range(na)
    ]
    if reverse:
        exact_h.reverse()
    for numeric, exact, role in (
        (patch.control_points, cp, "control points"),
        (patch.weights, w, "weights"),
        (patch.u_knots, uk, "u knots"),
        (patch.v_knots, vk, "v knots"),
    ):
        _require_exact_numeric_shadow(numeric, exact, role)
    high = side[1] == "1"
    span, ej = _endpoint_extraction(tk, td, nt, high, order)
    inward = -collar if high else collar
    exact_jets = [
        [
            [
                sum((ej[k][j] * exact_h[i][j][c] for j in range(nt)), Fraction(0))
                * inward**k
                for c in range(4)
            ]
            for i in range(na)
        ]
        for k in range(order + 1)
    ]
    numeric_h = jnp.concatenate(
        (patch.control_points * patch.weights[..., None], patch.weights[..., None]),
        axis=-1,
    )
    if transverse_u:
        numeric_h = jnp.swapaxes(numeric_h, 0, 1)
    if reverse:
        numeric_h = numeric_h[::-1]
    numeric_knots = patch.u_knots if transverse_u else patch.v_knots
    stencil = bspline_jet_stencil(
        numeric_knots,
        numeric_knots[nt if high else td],
        degree=td,
        maximum_order=order,
        spans=jnp.asarray(span, dtype=jnp.int64),
        bounds="clip",
    )
    numeric_jets = [
        contract("j,ijc->ic", stencil.jets[k], numeric_h[:, stencil.indices, :])
        * float(inward) ** k
        for k in range(order + 1)
    ]
    numeric_along = patch.v_knots if transverse_u else patch.u_knots
    if reverse:
        numeric_along = numeric_along[::-1]
    numeric_along = (numeric_along - float(a)) / float(b - a)
    return ad, pulled, exact_jets, numeric_jets, numeric_along


def build_hermite_bridge(
    left: BSplineSurfacePatch,
    right: BSplineSurfacePatch,
    *,
    left_side: Side,
    right_side: Side,
    left_along: tuple[Fraction, Fraction],
    right_along: tuple[Fraction, Fraction],
    left_collar: Fraction,
    right_collar: Fraction,
    continuity: int = 1,
) -> BSplineSurfacePatch:
    """Construct an exact homogeneous C1 or C2 endpoint-jet bridge.

    Positivity is certified by strictly positive exact Bernstein weights.  This
    sufficient condition is required by BSplineSurfacePatch; inputs whose
    bridge fails it are rejected rather than modifying the prescribed jets.
    Source coefficient and knot banks must be rational.  For traced source
    leaves the returned coefficient banks are intentionally absent: the JAX
    computation remains differentiable without asserting stale exact evidence.
    """
    if isinstance(continuity, bool) or continuity not in (1, 2):
        raise ValueError("continuity must be 1 or 2.")
    ld, lk, le, ln, along_knots = _source(
        left, left_side, left_along, left_collar, continuity
    )
    rd, rk, re, rn, _ = _source(right, right_side, right_along, right_collar, continuity)
    if ld != rd or lk != rk:
        raise ValueError(
            "Along-seam degrees and pulled-back knot vectors must match exactly; refinement is unsupported."
        )
    if any(lk.count(knot) > ld - continuity for knot in set(lk) if 0 < knot < 1):
        raise ValueError(
            "Along-seam knot continuity is insufficient for the requested order."
        )
    degree = 2 * continuity + 1

    # The left transverse direction is opposite the source inward direction;
    # the right transverse direction agrees with its source inward direction.
    def controls(l: list[Array], r: list[Array]) -> list[Array]:
        first = [l[0], l[0] - l[1] / degree]
        last = [r[0] - r[1] / degree, r[0]]
        if continuity == 2:
            first.append(l[0] - 2 * l[1] / degree + l[2] / (degree * (degree - 1)))
            last.insert(0, r[0] - 2 * r[1] / degree + r[2] / (degree * (degree - 1)))
        return first + last

    # Solve all along-seam homogeneous coefficients in one exact factorization.
    rhs = [
        [cast(Fraction, value) * (-1) ** k for point in le[k] for value in point]
        for k in range(continuity + 1)
    ]
    rhs.extend(
        [
            [cast(Fraction, value) for point in re[k] for value in point]
            for k in range(continuity + 1)
        ]
    )
    solved = _hermite_factor(continuity).solve(rhs)
    exact_rows = [
        [solved[j][4 * i : 4 * i + 4] for j in range(degree + 1)]
        for i in range(len(le[0]))
    ]
    exact_w = tuple(tuple(point[3] for point in row) for row in exact_rows)
    if any(weight <= 0 for row in exact_w for weight in row):
        raise ValueError("Hermite bridge cannot certify positive Bernstein weights.")
    exact_cp = tuple(
        tuple(tuple(point[k] / point[3] for k in range(3)) for point in row)
        for row in exact_rows
    )
    homogeneous = jnp.stack(controls(ln, rn), axis=1)
    weights = homogeneous[..., 3]
    points = homogeneous[..., :3] / weights[..., None]
    traced = any(
        isinstance(x, Tracer)
        for x in (
            left.control_points,
            left.weights,
            left.u_knots,
            left.v_knots,
            right.control_points,
            right.weights,
            right.u_knots,
            right.v_knots,
        )
    )
    if not traced:
        # Ensure exact-bank shadows are correctly rounded, independent of the
        # order in which floating point jet arithmetic happens to be evaluated.
        points = jnp.asarray(np.asarray(exact_cp, dtype=np.float64))
        weights = jnp.asarray(np.asarray(exact_w, dtype=np.float64))
        along_knots = jnp.asarray([float(x) for x in lk])
    transverse_knots = (Fraction(0),) * (degree + 1) + (Fraction(1),) * (degree + 1)
    return BSplineSurfacePatch(
        points,
        weights,
        along_knots,
        jnp.asarray([float(x) for x in transverse_knots]),
        ld,
        degree,
        exact_control_points=None if traced else exact_cp,
        exact_weights=None if traced else exact_w,
        exact_u_knots=None if traced else lk,
        exact_v_knots=transverse_knots,
    )


@dataclass(frozen=True, slots=True)
class HermiteBridgeEvidence:
    """Exact solve evidence bound to the two source patches and bridge result."""

    left: BSplineSurfacePatch
    right: BSplineSurfacePatch
    bridge: BSplineSurfacePatch
    left_side: Side
    right_side: Side
    left_along: tuple[Fraction, Fraction]
    right_along: tuple[Fraction, Fraction]
    left_collar: Fraction
    right_collar: Fraction
    continuity: int
    solve: ExactLinearSolveEvidence


def build_hermite_bridge_certified(
    left: BSplineSurfacePatch,
    right: BSplineSurfacePatch,
    *,
    left_side: Side,
    right_side: Side,
    left_along: tuple[Fraction, Fraction],
    right_along: tuple[Fraction, Fraction],
    left_collar: Fraction,
    right_collar: Fraction,
    continuity: int = 1,
) -> tuple[BSplineSurfacePatch, HermiteBridgeEvidence]:
    """Construct a concrete bridge and retain its source-bound exact rank proof."""
    if any(
        isinstance(value, Tracer)
        for patch in (left, right)
        for value in (patch.control_points, patch.weights, patch.u_knots, patch.v_knots)
    ):
        raise ValueError("Exact bridge evidence requires concrete source leaves.")
    bridge = build_hermite_bridge(
        left,
        right,
        left_side=left_side,
        right_side=right_side,
        left_along=left_along,
        right_along=right_along,
        left_collar=left_collar,
        right_collar=right_collar,
        continuity=continuity,
    )
    evidence = HermiteBridgeEvidence(
        left,
        right,
        bridge,
        left_side,
        right_side,
        left_along,
        right_along,
        left_collar,
        right_collar,
        continuity,
        _hermite_factor(continuity).evidence,
    )
    return bridge, evidence
