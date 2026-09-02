#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._estimates import DiscoveredBreakpoints
from ._plans import BreakpointDiscoveryPlan
from ._status import IntegrationStatus


_BREAKPOINT_JUMP = 1
_BREAKPOINT_CUSP = 2
_BREAKPOINT_NONFINITE = 3


def _row_values(values: Array, /) -> tuple[Array, Array]:
    flattened = values.reshape((values.shape[0], -1))
    if not jnp.issubdtype(flattened.dtype, jnp.inexact):
        flattened = flattened.astype(float)
    finite = jnp.all(jnp.isfinite(flattened), axis=-1)
    return jnp.where(jnp.isfinite(flattened), flattened, 0.0), finite


def _row_difference(left: Array, right: Array, /) -> Array:
    return jnp.max(jnp.abs(right - left), axis=-1)


def discover_breakpoints(
    integrand: Callable[[Array], Array],
    bounds: Array,
    plan: BreakpointDiscoveryPlan,
    /,
    *,
    explicit: tuple[float, ...] = (),
) -> tuple[DiscoveredBreakpoints, Array]:
    """Return bounded numerical feature candidates and pilot evaluation count."""
    if not isinstance(plan, BreakpointDiscoveryPlan):
        raise TypeError("plan must be a BreakpointDiscoveryPlan.")
    if plan.max_candidates > plan.pilot_count - 1:
        raise ValueError("max_candidates cannot exceed the number of pilot intervals.")
    bounds_ = jnp.asarray(bounds, dtype=float)
    if bounds_.shape != (2,):
        raise ValueError("Breakpoint discovery bounds must have shape (2,).")
    parameter = jnp.linspace(0.0, 1.0, plan.pilot_count)
    nested = 0.5 * (1.0 - jnp.cos(jnp.pi * parameter))
    points = bounds_[0] + (bounds_[1] - bounds_[0]) * nested
    values = jnp.asarray(integrand(points))
    if values.ndim == 0 or values.shape[0] != plan.pilot_count:
        raise ValueError("Breakpoint pilot integrands must preserve the point axis.")
    rows, finite = _row_values(values)
    first = _row_difference(rows[:-1], rows[1:])
    first_scale = jnp.maximum(
        jnp.median(first),
        jnp.finfo(first.dtype).tiny,
    )
    jump_ratio = first / first_scale
    second = jnp.max(
        jnp.abs(rows[2:] - 2.0 * rows[1:-1] + rows[:-2]),
        axis=-1,
    )
    curvature = jnp.zeros_like(first)
    curvature = curvature.at[:-1].max(second)
    curvature = curvature.at[1:].max(second)
    second_scale = jnp.maximum(
        jnp.median(second),
        jnp.finfo(second.dtype).tiny,
    )
    defect_ratio = curvature / second_scale
    nonfinite_transition = ~(finite[:-1] & finite[1:])
    evidence = jnp.maximum(
        jump_ratio / plan.jump_threshold,
        defect_ratio / plan.defect_threshold,
    )
    evidence = jnp.where(nonfinite_transition, jnp.asarray(jnp.inf), evidence)
    eligible = (evidence >= 1.0) | nonfinite_transition
    total_candidates = jnp.sum(eligible, dtype=jnp.int32)
    ranked = jnp.where(eligible, evidence, -jnp.inf)
    scores, indices = jax.lax.top_k(ranked, plan.max_candidates)
    active = jnp.isfinite(scores) | (scores == jnp.inf)
    left = points[indices]
    right = points[indices + 1]
    left_values = rows[indices]
    right_values = rows[indices + 1]
    left_finite = finite[indices]
    right_finite = finite[indices + 1]

    def refine(carry, _):
        lower, upper, lower_value, upper_value, lower_finite, upper_finite = carry
        midpoint = 0.5 * (lower + upper)
        mid_values = jnp.asarray(integrand(midpoint))
        mid_rows, mid_finite = _row_values(mid_values)
        left_defect = jnp.where(
            lower_finite & mid_finite,
            _row_difference(lower_value, mid_rows),
            jnp.inf,
        )
        right_defect = jnp.where(
            mid_finite & upper_finite,
            _row_difference(mid_rows, upper_value),
            jnp.inf,
        )
        choose_left = left_defect >= right_defect
        return (
            jnp.where(choose_left, lower, midpoint),
            jnp.where(choose_left, midpoint, upper),
            jnp.where(choose_left[:, None], lower_value, mid_rows),
            jnp.where(choose_left[:, None], mid_rows, upper_value),
            jnp.where(choose_left, lower_finite, mid_finite),
            jnp.where(choose_left, mid_finite, upper_finite),
        ), None

    refined, _ = jax.lax.scan(
        refine,
        (left, right, left_values, right_values, left_finite, right_finite),
        xs=None,
        length=plan.refinement_rounds,
    )
    left, right, _, _, left_finite, right_finite = refined
    candidates = jax.lax.stop_gradient(0.5 * (left + right))
    kinds = jnp.where(
        ~(left_finite & right_finite),
        _BREAKPOINT_NONFINITE,
        jnp.where(
            jump_ratio[indices] >= plan.jump_threshold,
            _BREAKPOINT_JUMP,
            _BREAKPOINT_CUSP,
        ),
    ).astype(jnp.int32)

    order = jnp.argsort(jnp.where(active, candidates, bounds_[1]))
    candidates = candidates[order]
    scores = jax.lax.stop_gradient(scores[order])
    kinds = kinds[order]
    active = active[order]
    separated = jnp.concatenate(
        (
            jnp.asarray([True]),
            jnp.diff(candidates) > plan.minimum_separation,
        )
    )
    if explicit:
        explicit_points = jnp.asarray(explicit, dtype=candidates.dtype)
        separated &= jnp.all(
            jnp.abs(candidates[:, None] - explicit_points[None, :])
            > plan.minimum_separation,
            axis=-1,
        )
    active &= separated
    unresolved = active & (kinds == _BREAKPOINT_NONFINITE)
    overflow = total_candidates > plan.max_candidates
    status = jnp.where(
        overflow,
        int(IntegrationStatus.BREAKPOINT_CANDIDATE_OVERFLOW),
        jnp.where(
            jnp.any(unresolved),
            int(IntegrationStatus.UNRESOLVED_NONFINITE),
            int(IntegrationStatus.CONVERGED),
        ),
    ).astype(jnp.int32)
    discovery = DiscoveredBreakpoints(
        points=candidates,
        active=active,
        scores=scores,
        kinds=kinds,
        status=status,
    )
    evaluations = plan.pilot_count + plan.refinement_rounds * plan.max_candidates
    return discovery, jnp.asarray(evaluations, dtype=jnp.int32)


__all__ = ["discover_breakpoints"]
