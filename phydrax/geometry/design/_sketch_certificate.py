#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from fractions import Fraction
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg._policies import RankPolicy
from ...linalg._rank import host_numerical_rank_data


class RootDOFEvidence(StrictModule):
    """Numerical root and local degree-of-freedom evidence for a design solve.

    The rank is that of the residual Jacobian at the returned state, resolved
    using the native linalg default numerical-rank policy. This is a local
    numerical diagnostic; it is not a global root count or interval uniqueness
    proof.
    """

    problem_id: str = eqx.field(static=True)
    variable_count: int = eqx.field(static=True)
    equation_count: int = eqx.field(static=True)
    jacobian_rank: int = eqx.field(static=True)
    degrees_of_freedom: int = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    residual_norm: Array
    singular_values: Array
    numerically_finite: bool = eqx.field(static=True)
    root_converged: bool = eqx.field(static=True)
    full_column_rank: bool = eqx.field(static=True)
    regular_root_indicated: bool = eqx.field(static=True)
    scope: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem_id: str,
        coordinates: ArrayLike,
        residual: ArrayLike,
        jacobian: ArrayLike,
        /,
        *,
        residual_tolerance: float,
    ) -> None:
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError("problem_id must be a non-empty string.")
        tolerance = float(residual_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("residual_tolerance must be finite and positive.")
        coordinates_ = np.asarray(coordinates, dtype=np.float64).reshape((-1,))
        residual_ = np.asarray(residual, dtype=np.float64).reshape((-1,))
        jacobian_ = np.asarray(jacobian, dtype=np.float64)
        if jacobian_.shape != (residual_.size, coordinates_.size):
            raise ValueError(
                "The residual Jacobian must align with equations and variables."
            )
        numerically_finite = bool(
            np.all(np.isfinite(coordinates_))
            and np.all(np.isfinite(residual_))
            and np.all(np.isfinite(jacobian_))
        )
        if numerically_finite:
            singular_values = np.linalg.svd(jacobian_, compute_uv=False)
            if singular_values.size:
                rank_data = host_numerical_rank_data(
                    singular_values, jacobian_.shape[0], jacobian_.shape[1], RankPolicy()
                )
                rank_tolerance = float(rank_data.cutoff)
                rank = int(rank_data.rank)
            else:
                rank_tolerance = 0.0
                rank = 0
            residual_norm = float(np.linalg.norm(residual_))
        else:
            singular_values = np.empty((0,), dtype=np.float64)
            rank_tolerance = 0.0
            rank = 0
            residual_norm = float(np.linalg.norm(residual_))
        root_converged = numerically_finite and residual_norm <= tolerance
        full_column_rank = numerically_finite and rank == coordinates_.size
        regular_root_indicated = root_converged and full_column_rank
        scope = "pointwise-residual-jacobian-rank"
        evidence_id = canonical_fingerprint(
            {
                "kind": "design-root-dof-evidence",
                "problem": problem_id,
                "coordinates": coordinates_.tolist() if numerically_finite else None,
                "residual": residual_.tolist() if numerically_finite else None,
                "jacobian": jacobian_.tolist() if numerically_finite else None,
                "numerically_finite": numerically_finite,
                "rank_tolerance": rank_tolerance,
                "residual_tolerance": tolerance,
                "scope": scope,
            }
        )
        self.numerically_finite = numerically_finite
        self.problem_id = problem_id
        self.variable_count = coordinates_.size
        self.equation_count = residual_.size
        self.jacobian_rank = rank
        self.degrees_of_freedom = coordinates_.size - rank
        self.rank_tolerance = rank_tolerance
        self.residual_tolerance = tolerance
        self.residual_norm = jnp.asarray(residual_norm, dtype=jnp.float64)
        self.singular_values = jnp.asarray(singular_values, dtype=jnp.float64)
        self.root_converged = root_converged
        self.full_column_rank = full_column_rank
        self.regular_root_indicated = regular_root_indicated
        self.scope = scope
        self.evidence_id = evidence_id


class TwoCircleConstruction(NamedTuple):
    """Exact rational data for one transversal, reference-selected branch."""

    first: tuple[Fraction, Fraction]
    second: tuple[Fraction, Fraction]
    first_radius: Fraction
    second_radius: Fraction
    squared_height_numerator: Fraction
    branch_sign: int


class LineCircleConstruction(NamedTuple):
    """Exact rational data for a transversal axis-aligned line-circle branch."""

    center: tuple[Fraction, Fraction]
    radius: Fraction
    fixed_axis: int
    line_coordinate: Fraction
    discriminant: Fraction
    branch_sign: int


def certify_line_circle_construction(
    center: ArrayLike,
    radius: ArrayLike,
    fixed_axis: int,
    line_coordinate: ArrayLike,
    reference: ArrayLike,
    /,
    *,
    reference_center: ArrayLike | None = None,
) -> LineCircleConstruction | None:
    """Prove a transversal branch from represented inputs, not a rounded root."""
    c = np.asarray(center, dtype=np.float64).reshape((-1,))
    before = np.asarray(reference, dtype=np.float64).reshape((-1,))
    old_center = (
        c
        if reference_center is None
        else np.asarray(reference_center, dtype=np.float64).reshape((-1,))
    )
    r = np.asarray(radius, dtype=np.float64)
    line = np.asarray(line_coordinate, dtype=np.float64)
    if (
        fixed_axis not in (0, 1)
        or c.shape != (2,)
        or before.shape != (2,)
        or old_center.shape != (2,)
        or r.shape != ()
        or line.shape != ()
        or not all(
            np.all(np.isfinite(value)) for value in (c, before, old_center, r, line)
        )
        or float(r) <= 0.0
    ):
        return None
    cq = tuple(Fraction.from_float(float(value)) for value in c)
    rq = Fraction.from_float(float(r))
    lineq = Fraction.from_float(float(line))
    difference = lineq - cq[fixed_axis]
    discriminant = rq * rq - difference * difference
    free_axis = 1 - fixed_axis
    side = Fraction.from_float(float(before[free_axis])) - Fraction.from_float(
        float(old_center[free_axis])
    )
    if discriminant <= 0 or side == 0:
        return None
    return LineCircleConstruction(
        (cq[0], cq[1]),
        rq,
        fixed_axis,
        lineq,
        discriminant,
        1 if side > 0 else -1,
    )


def certify_two_circle_construction(
    first: ArrayLike,
    second: ArrayLike,
    first_radius: ArrayLike,
    second_radius: ArrayLike,
    reference: ArrayLike,
    /,
    *,
    reference_centers: tuple[ArrayLike, ArrayLike] | None = None,
) -> TwoCircleConstruction | None:
    """Prove a transversal branch over the original represented inputs.

    The algebraic point is ``a + (k*d + s*sqrt(H)*perp(d))/q``.
    No rounded numerical point is used as a proof premise.
    """
    a, b, before = (
        np.asarray(value, dtype=np.float64).reshape((-1,))
        for value in (first, second, reference)
    )
    old_a, old_b = (
        (a, b)
        if reference_centers is None
        else tuple(
            np.asarray(value, dtype=np.float64).reshape((-1,))
            for value in reference_centers
        )
    )
    radii = np.asarray([first_radius, second_radius], dtype=np.float64).reshape((-1,))
    if (
        a.shape != (2,)
        or b.shape != (2,)
        or before.shape != (2,)
        or old_a.shape != (2,)
        or old_b.shape != (2,)
        or radii.shape != (2,)
        or not all(
            np.all(np.isfinite(value)) for value in (a, b, before, old_a, old_b, radii)
        )
        or np.any(radii <= 0.0)
    ):
        return None
    aq = tuple(Fraction.from_float(float(value)) for value in a)
    bq = tuple(Fraction.from_float(float(value)) for value in b)
    rq = tuple(Fraction.from_float(float(value)) for value in radii)
    refq = tuple(Fraction.from_float(float(value)) for value in before)
    old_aq = tuple(Fraction.from_float(float(value)) for value in old_a)
    old_bq = tuple(Fraction.from_float(float(value)) for value in old_b)
    dx, dy = bq[0] - aq[0], bq[1] - aq[1]
    q = dx * dx + dy * dy
    if q == 0:
        return None
    k = (q + rq[0] * rq[0] - rq[1] * rq[1]) / 2
    height = rq[0] * rq[0] * q - k * k
    old_dx, old_dy = old_bq[0] - old_aq[0], old_bq[1] - old_aq[1]
    side = old_dx * (refq[1] - old_aq[1]) - old_dy * (refq[0] - old_aq[0])
    if height <= 0 or side == 0:
        return None
    return TwoCircleConstruction(
        (aq[0], aq[1]),
        (bq[0], bq[1]),
        rq[0],
        rq[1],
        height,
        1 if side > 0 else -1,
    )


__all__ = ["RootDOFEvidence"]
