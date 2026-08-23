#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._heom import HEOMHierarchy, HEOMProblem, HEOMSolution, solve_heom


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


__all__ = [
    "HEOMContinuationResult",
    "HEOMContinuationStage",
    "solve_heom_continuation",
]
