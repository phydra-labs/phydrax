#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..operators.quantum import BathCorrelationExpansion
from ._heom import HEOMHierarchy, HEOMProblem, HEOMSolution, solve_heom
from ._heom_implicit import solve_heom_bdf


class HEOMContinuationStage(StrictModule):
    depth: int
    auxiliary_count: int
    root_difference: Array
    maximum_top_tier_norm: Array
    valid: Array

    def __init__(
        self,
        depth: int,
        auxiliary_count: int,
        root_difference: ArrayLike,
        maximum_top_tier_norm: ArrayLike,
        valid: ArrayLike,
        /,
    ):
        self.depth = int(depth)
        self.auxiliary_count = int(auxiliary_count)
        self.root_difference = jnp.asarray(root_difference)
        self.maximum_top_tier_norm = jnp.asarray(maximum_top_tier_norm)
        self.valid = jnp.asarray(valid, dtype=bool)


class HEOMContinuationResult(StrictModule):
    solutions: tuple[HEOMSolution, ...]
    stages: tuple[HEOMContinuationStage, ...]
    converged: Array

    def __init__(
        self,
        solutions: Sequence[HEOMSolution],
        stages: Sequence[HEOMContinuationStage],
        /,
        *,
        tolerance: float,
    ):
        self.solutions = tuple(solutions)
        self.stages = tuple(stages)
        self.converged = (
            self.stages[-1].valid
            & (self.stages[-1].root_difference <= tolerance)
            & (self.stages[-1].maximum_top_tier_norm <= tolerance)
        )


def solve_heom_continuation(
    base_problem: HEOMProblem,
    depths: Sequence[int],
    /,
    *,
    step_size: ArrayLike,
    steps: int,
    tolerance: float = 1e-5,
) -> HEOMContinuationResult:
    depth_sequence = tuple(int(depth) for depth in depths)
    if not depth_sequence or any(
        right <= left
        for left, right in zip(depth_sequence[:-1], depth_sequence[1:], strict=True)
    ):
        raise ValueError("HEOM continuation depths must be strictly increasing.")
    solutions = []
    stages = []
    initial_root = base_problem.initial_state[0]
    previous_final = None
    for depth in depth_sequence:
        hierarchy = HEOMHierarchy(base_problem.expansion.rank, depth)
        problem = HEOMProblem(
            base_problem.hamiltonian,
            base_problem.coupling_operator,
            base_problem.expansion,
            hierarchy,
            initial_root,
            geometry_precision=base_problem.geometry_precision,
            hermitian_precision=base_problem.hermitian_precision,
            problem_id=f"{base_problem.problem_id}:depth-{depth}",
        )
        solution = solve_heom(problem, step_size=step_size, steps=steps)
        root = solution.root_states[-1]
        difference = (
            jnp.asarray(jnp.inf)
            if previous_final is None
            else jnp.linalg.norm(root - previous_final)
        )
        top_norm = solution.maximum_auxiliary_norm_by_level[-1]
        stages.append(
            HEOMContinuationStage(
                depth,
                hierarchy.auxiliary_count,
                difference,
                top_norm,
                solution.valid,
            )
        )
        solutions.append(solution)
        previous_final = root
    return HEOMContinuationResult(solutions, stages, tolerance=tolerance)


class HEOMGridContinuationResult(StrictModule):
    final_roots: Array
    depth_differences: Array
    bath_differences: Array
    valid: Array

    def __init__(
        self,
        final_roots: ArrayLike,
        depth_differences: ArrayLike,
        bath_differences: ArrayLike,
        valid: ArrayLike,
        /,
    ):
        self.final_roots = jnp.asarray(final_roots)
        self.depth_differences = jnp.asarray(depth_differences)
        self.bath_differences = jnp.asarray(bath_differences)
        self.valid = jnp.asarray(valid, dtype=bool)


def solve_heom_continuation_grid(
    base_problem: HEOMProblem,
    expansions: Sequence[BathCorrelationExpansion],
    depths: Sequence[int],
    /,
    *,
    step_size: ArrayLike,
    steps: int,
    maximum_order: int = 2,
) -> HEOMGridContinuationResult:
    expansions_ = tuple(expansions)
    depths_ = tuple(int(depth) for depth in depths)
    if not expansions_ or not depths_:
        raise ValueError("HEOM continuation grid axes must be non-empty.")
    roots = []
    valid = []
    initial = base_problem.initial_state[0]
    for expansion in expansions_:
        row = []
        row_valid = []
        for depth in depths_:
            problem = HEOMProblem(
                base_problem.hamiltonian,
                base_problem.coupling_operator,
                expansion,
                HEOMHierarchy(expansion.rank, depth),
                initial,
                geometry_precision=base_problem.geometry_precision,
                hermitian_precision=base_problem.hermitian_precision,
                problem_id=(
                    f"{base_problem.problem_id}:bath-{expansion.expansion_id}:"
                    f"depth-{depth}"
                ),
            )
            result = solve_heom_bdf(
                problem,
                step_size=step_size,
                steps=steps,
                maximum_order=maximum_order,
            )
            row.append(result.solution.root_states[-1])
            row_valid.append(result.valid)
        roots.append(jnp.stack(row))
        valid.append(jnp.stack(row_valid))
    values = jnp.stack(roots)
    depth_differences = (
        jnp.linalg.norm(values[:, 1:] - values[:, :-1], axis=(-2, -1))
        if len(depths_) > 1
        else jnp.zeros((len(expansions_), 0))
    )
    bath_differences = (
        jnp.linalg.norm(values[1:] - values[:-1], axis=(-2, -1))
        if len(expansions_) > 1
        else jnp.zeros((0, len(depths_)))
    )
    return HEOMGridContinuationResult(
        values,
        depth_differences,
        bath_differences,
        jnp.all(jnp.stack(valid)),
    )


class PreparedHEOMRefinementPlan(StrictModule):
    """Finite bath/depth/time epochs with fixed topology inside each solve."""

    expansions: tuple[BathCorrelationExpansion, ...]
    depths: tuple[int, ...]
    time_steps: tuple[int, ...]
    tolerance: float
    future_contraction: float | None
    plan_id: str

    def __init__(
        self,
        expansions: Sequence[BathCorrelationExpansion],
        depths: Sequence[int],
        time_steps: Sequence[int],
        /,
        *,
        tolerance: float,
        future_contraction: float | None = None,
        plan_id: str,
    ):
        expansions_ = tuple(expansions)
        depths_ = tuple(int(value) for value in depths)
        steps_ = tuple(int(value) for value in time_steps)
        if (
            len(expansions_) < 1
            or len(depths_) < 2
            or len(steps_) < 2
            or any(
                right <= left
                for left, right in zip(depths_[:-1], depths_[1:], strict=True)
            )
            or any(
                right <= left for left, right in zip(steps_[:-1], steps_[1:], strict=True)
            )
            or tolerance < 0.0
            or not isinstance(plan_id, str)
            or not plan_id
        ):
            raise ValueError("Prepared HEOM refinement axes/tolerance/ID are invalid.")
        if future_contraction is not None and not 0.0 <= future_contraction < 1.0:
            raise ValueError("future_contraction must lie in [0,1) when certified.")
        self.expansions = expansions_
        self.depths = depths_
        self.time_steps = steps_
        self.tolerance = float(tolerance)
        self.future_contraction = (
            None if future_contraction is None else float(future_contraction)
        )
        self.plan_id = plan_id


class HEOMRefinementCertificate(StrictModule):
    final_roots: Array
    depth_differences: Array
    bath_differences: Array
    time_differences: Array
    top_tier_norms: Array
    remainder: Array
    stabilized: Array
    valid: Array
    estimate_kind: str
    plan_id: str
    claim: str


def solve_prepared_heom_refinement(
    base_problem: HEOMProblem,
    plan: PreparedHEOMRefinementPlan,
    /,
    *,
    final_time: float,
    maximum_order: int = 2,
) -> HEOMRefinementCertificate:
    """Execute finite refinement epochs; a tail bound requires a supplied hypothesis."""
    if not isinstance(base_problem, HEOMProblem):
        raise TypeError("base_problem must be HEOMProblem.")
    if not isinstance(plan, PreparedHEOMRefinementPlan):
        raise TypeError("plan must be PreparedHEOMRefinementPlan.")
    roots = []
    top_tiers = []
    validity = []
    initial = base_problem.initial_state[0]
    for expansion in plan.expansions:
        depth_rows = []
        tier_rows = []
        valid_rows = []
        for depth in plan.depths:
            time_values = []
            tier_values = []
            time_valid = []
            for steps in plan.time_steps:
                problem = HEOMProblem(
                    base_problem.hamiltonian,
                    base_problem.coupling_operator,
                    expansion,
                    HEOMHierarchy(expansion.rank, depth),
                    initial,
                    geometry_precision=base_problem.geometry_precision,
                    hermitian_precision=base_problem.hermitian_precision,
                    problem_id=(
                        f"{base_problem.problem_id}:prepared:"
                        f"{expansion.expansion_id}:{depth}:{steps}"
                    ),
                )
                result = solve_heom_bdf(
                    problem,
                    step_size=final_time / steps,
                    steps=steps,
                    maximum_order=maximum_order,
                )
                time_values.append(result.solution.root_states[-1])
                tier_values.append(result.solution.maximum_auxiliary_norm_by_level[-1])
                time_valid.append(result.valid)
            depth_rows.append(jnp.stack(time_values))
            tier_rows.append(jnp.stack(tier_values))
            valid_rows.append(jnp.stack(time_valid))
        roots.append(jnp.stack(depth_rows))
        top_tiers.append(jnp.stack(tier_rows))
        validity.append(jnp.stack(valid_rows))
    values = jnp.stack(roots)
    tiers = jnp.stack(top_tiers)
    depth_difference = jnp.sqrt(
        jnp.sum(jnp.abs(values[:, 1:] - values[:, :-1]) ** 2, axis=(-2, -1))
    )
    bath_difference = (
        jnp.sqrt(jnp.sum(jnp.abs(values[1:] - values[:-1]) ** 2, axis=(-2, -1)))
        if len(plan.expansions) > 1
        else jnp.zeros((0, len(plan.depths), len(plan.time_steps)))
    )
    time_difference = jnp.sqrt(
        jnp.sum(jnp.abs(values[:, :, 1:] - values[:, :, :-1]) ** 2, axis=(-2, -1))
    )
    last_depth = jnp.max(depth_difference[:, -1, -1])
    last_time = jnp.max(time_difference[:, -1, -1])
    last_bath = (
        jnp.max(bath_difference[:, -1, -1])
        if len(plan.expansions) > 1
        else jnp.asarray(0.0)
    )
    observed = jnp.maximum(last_depth, jnp.maximum(last_time, last_bath))
    if plan.future_contraction is None:
        remainder = observed
        kind = "difference"
    else:
        remainder = observed * plan.future_contraction / (1.0 - plan.future_contraction)
        kind = "bound"
    stabilized = (observed <= plan.tolerance) & (
        jnp.max(tiers[:, -1, -1]) <= plan.tolerance
    )
    return HEOMRefinementCertificate(
        final_roots=values,
        depth_differences=depth_difference,
        bath_differences=bath_difference,
        time_differences=time_difference,
        top_tier_norms=tiers,
        remainder=remainder,
        stabilized=stabilized,
        valid=jnp.all(jnp.stack(validity)) & jnp.isfinite(remainder),
        estimate_kind=kind,
        plan_id=plan.plan_id,
        claim="finite-heom-refinement-bound-only-under-certified-contraction",
    )


__all__ = [
    "HEOMRefinementCertificate",
    "PreparedHEOMRefinementPlan",
    "HEOMContinuationResult",
    "HEOMContinuationStage",
    "HEOMGridContinuationResult",
    "solve_heom_continuation",
    "solve_heom_continuation_grid",
    "solve_prepared_heom_refinement",
]
