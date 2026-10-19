# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Explicit priority admission; optional constraints are never silently lost."""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked
from ._sketch import Sketch, SketchSolution
from ._sketch_certificate import certify_sketch_solution, SketchCertifiedSolution


@final
class SketchPrioritySolution(StrictModule):
    solution: SketchSolution
    sketch: Sketch
    lowest_satisfied_tier: int = eqx.field(static=True)
    unsatisfied_constraint_ids: tuple[str, ...] = eqx.field(static=True)
    rejected_constraint_ids: tuple[str, ...] = eqx.field(static=True)
    constraint_residual_norms: Array
    tier_certificates: tuple[SketchCertifiedSolution, ...]


def _constraint_report(
    sketch: Sketch,
    solution: SketchSolution,
    admitted: tuple[int, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], Array]:
    """Report every authored constraint, including incidental satisfaction of rejected tiers."""
    norms = jnp.asarray(
        [
            jnp.max(
                jnp.abs(
                    constraint.residual(
                        solution.points,
                        sketch.lines,
                        sketch.circle_centers,
                        solution.circle_radii,
                    )
                )
            )
            for constraint in sketch.constraints
        ],
        dtype=jnp.float64,
    )
    host, residual_bound = jax.device_get((norms, solution.residual_bound))
    unsatisfied = tuple(
        identity
        for identity, norm in zip(sketch.constraint_ids, np.asarray(host), strict=True)
        if not np.isfinite(norm) or norm > residual_bound
    )
    rejected = tuple(
        identity
        for index, identity in enumerate(sketch.constraint_ids)
        if index not in admitted
    )
    return unsatisfied, rejected, norms


@checked
def solve_priority_sketch(
    sketch: Sketch,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> SketchPrioritySolution:
    """Admit cumulative tiers in ascending order, preserving every higher tier.

    Zero denotes mandatory constraints. A tier is admitted only as a whole with
    strict Krawczyk inclusion; underconstrained tiers remain provisional until a
    later cumulative tier proves a root. No certified tier gives lowest
    tier -1. Every attempted tier publishes its certificate, including refusals.
    """
    tiers = tuple(sorted(set((0, *sketch.constraint_priorities))))
    count = sketch.points.size + sketch.circle_radii.size
    budget.reserve(len(tiers) * 32 * 12 * max(1, count) ** 3, 512 * max(1, count) ** 2)
    admitted: tuple[int, ...] = ()
    lowest = -1
    solution: SketchSolution | None = None
    certificates: list[SketchCertifiedSolution] = []
    certified_solution: SketchSolution | None = None
    for tier in tiers:
        selected = tuple(
            index
            for index, priority in enumerate(sketch.constraint_priorities)
            if priority <= tier
        )
        trial_sketch = Sketch(
            sketch.points if solution is None else solution.points,
            lines=sketch.lines,
            circle_centers=sketch.circle_centers,
            circle_radii=sketch.circle_radii
            if solution is None
            else solution.circle_radii,
            arcs=sketch.arcs,
            constraints=tuple(sketch.constraints[index] for index in selected),
            constraint_ids=tuple(sketch.constraint_ids[index] for index in selected),
            constraint_priorities=tuple(
                sketch.constraint_priorities[index] for index in selected
            ),
            feature_id=sketch.feature_id,
        )
        trial = trial_sketch.solve()
        certificate = certify_sketch_solution(trial, budget=budget)
        certificates.append(certificate)
        if solution is None:
            solution = trial
        if not bool(jax.device_get(trial.converged)):
            break
        solution = trial
        if certificate.status == "certified":
            admitted, certified_solution, lowest = selected, trial, tier
        elif certificate.status != "not_fully_constrained":
            break
    if solution is None:
        raise RuntimeError(
            "SketchPrioritySolution.solution must include a mandatory solve result; got no result"
        )
    if certified_solution is not None:
        solution = certified_solution
    unsatisfied, rejected, norms = _constraint_report(sketch, solution, admitted)
    return SketchPrioritySolution(
        solution, sketch, lowest, unsatisfied, rejected, norms, tuple(certificates)
    )


__all__ = ["SketchPrioritySolution", "solve_priority_sketch"]
