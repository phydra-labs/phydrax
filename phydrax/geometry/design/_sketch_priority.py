#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit cumulative sketch priority solving without hidden resource admission."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ._sketch import Sketch, SketchSolution
from ._sketch_certificate import RootDOFEvidence


class SketchPrioritySolution(StrictModule):
    """Priority-tier attempts with explicit rejected constraints and local rank scope."""

    solution: SketchSolution
    sketch: Sketch
    solved_sketch: Sketch
    lowest_satisfied_tier: int = eqx.field(static=True)
    admitted_constraint_ids: tuple[str, ...] = eqx.field(static=True)
    unsatisfied_constraint_ids: tuple[str, ...] = eqx.field(static=True)
    rejected_constraint_ids: tuple[str, ...] = eqx.field(static=True)
    constraint_residual_norms: Array
    tier_evidence: tuple[RootDOFEvidence, ...]
    scope: str = eqx.field(static=True)

    def __init__(
        self,
        solution: SketchSolution,
        sketch: Sketch,
        solved_sketch: Sketch,
        lowest_satisfied_tier: int,
        admitted_constraint_ids: tuple[str, ...],
        unsatisfied_constraint_ids: tuple[str, ...],
        rejected_constraint_ids: tuple[str, ...],
        constraint_residual_norms: Array,
        tier_evidence: tuple[RootDOFEvidence, ...],
        /,
    ) -> None:
        if not isinstance(solution, SketchSolution):
            raise TypeError("solution must be a SketchSolution.")
        if not isinstance(sketch, Sketch):
            raise TypeError("sketch must be a Sketch.")
        if any(not isinstance(value, RootDOFEvidence) for value in tier_evidence):
            raise TypeError("tier_evidence must contain RootDOFEvidence values.")
        self.solution = solution
        if not isinstance(solved_sketch, Sketch):
            raise TypeError("solved_sketch must be a Sketch.")
        self.sketch = sketch
        self.solved_sketch = solved_sketch
        self.lowest_satisfied_tier = int(lowest_satisfied_tier)
        self.admitted_constraint_ids = tuple(admitted_constraint_ids)
        self.unsatisfied_constraint_ids = tuple(unsatisfied_constraint_ids)
        self.rejected_constraint_ids = tuple(rejected_constraint_ids)
        self.constraint_residual_norms = jnp.asarray(
            constraint_residual_norms, dtype=jnp.float64
        )
        self.tier_evidence = tuple(tier_evidence)
        self.scope = "numerical-local-rank-indicator-not-interval-uniqueness-proof"


def _constraint_report(
    sketch: Sketch,
    solution: SketchSolution,
    admitted: tuple[int, ...],
    /,
) -> tuple[tuple[str, ...], tuple[str, ...], Array]:
    components = sketch._residual_components(
        solution.points,
        solution.circle_radii,
        dimensions=sketch.dimensions,
    )
    norms = jnp.asarray(
        [jnp.linalg.norm(value) for value in components], dtype=jnp.float64
    )
    host_norms = np.asarray(norms, dtype=np.float64)
    tolerance = solution.root_evidence.residual_tolerance
    unsatisfied = tuple(
        identifier
        for identifier, norm in zip(sketch.constraint_ids, host_norms, strict=True)
        if not np.isfinite(norm) or norm > tolerance
    )
    admitted_set = frozenset(admitted)
    rejected = tuple(
        identifier
        for index, identifier in enumerate(sketch.constraint_ids)
        if index not in admitted_set
    )
    return unsatisfied, rejected, norms


def solve_priority_sketch(
    sketch: Sketch,
    /,
    *,
    max_iterations: int = 32,
    tolerance: float = 1e-9,
    damping: float = 1e-8,
) -> SketchPrioritySolution:
    """Solve cumulative priority tiers and admit only numerically regular roots.

    Full-column-rank evidence is local and numerical. It is never reported as an
    interval uniqueness certificate; every tier's evidence remains available.
    """
    if not isinstance(sketch, Sketch):
        raise TypeError("sketch must be a Sketch.")
    if not sketch.constraints:
        raise ValueError("Priority solving requires at least one sketch constraint.")
    tiers = tuple(sorted(set(sketch.constraint_priorities)))
    evidence: list[RootDOFEvidence] = []
    admitted: tuple[int, ...] = ()
    admitted_tier = -1
    regular_solution: SketchSolution | None = None
    regular_sketch: Sketch | None = None
    provisional_sketch: Sketch | None = None
    provisional_solution: SketchSolution | None = None
    for tier in tiers:
        selected = tuple(
            index
            for index, priority in enumerate(sketch.constraint_priorities)
            if priority <= tier
        )
        trial_sketch = Sketch(
            sketch.points
            if provisional_solution is None
            else provisional_solution.points,
            lines=sketch.lines,
            circle_centers=sketch.circle_centers,
            arc_endpoints=sketch.arc_endpoints,
            arc_centers=sketch.arc_centers,
            arc_counterclockwise=sketch.arc_counterclockwise,
            circle_radii=(
                sketch.circle_radii
                if provisional_solution is None
                else provisional_solution.circle_radii
            ),
            constraints=tuple(sketch.constraints[index] for index in selected),
            feature_id=sketch.feature_id,
            constraint_ids=tuple(sketch.constraint_ids[index] for index in selected),
            constraint_priorities=tuple(
                sketch.constraint_priorities[index] for index in selected
            ),
        )
        trial = trial_sketch.solve(
            max_iterations=max_iterations,
            tolerance=tolerance,
            damping=damping,
        )
        evidence.append(trial.root_evidence)
        if provisional_solution is None or bool(np.asarray(trial.converged)):
            provisional_sketch = trial_sketch
        if provisional_solution is None or bool(np.asarray(trial.converged)):
            provisional_solution = trial
        if not bool(np.asarray(trial.converged)):
            break
        if trial.root_evidence.regular_root_indicated:
            regular_solution = trial
            admitted = selected
            admitted_tier = tier
            regular_sketch = trial_sketch

    if regular_solution is not None:
        selected_solution = regular_solution
        selected_sketch = regular_sketch
    else:
        selected_solution = provisional_solution
        selected_sketch = provisional_sketch
    if selected_solution is None or selected_sketch is None:
        raise RuntimeError("Priority solving completed without a sketch result.")
    unsatisfied, rejected, norms = _constraint_report(
        sketch,
        selected_solution,
        admitted,
    )
    return SketchPrioritySolution(
        selected_solution,
        sketch,
        selected_sketch,
        admitted_tier,
        tuple(sketch.constraint_ids[index] for index in admitted),
        unsatisfied,
        rejected,
        norms,
        tuple(evidence),
    )


__all__ = ["SketchPrioritySolution", "solve_priority_sketch"]
