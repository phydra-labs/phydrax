#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..stochastic._rough import AbstractRoughControl
from ._rough import (
    AbstractRoughSolver,
    Davie,
    RoughDifferentialProblem,
    RoughEuler,
    solve_rough_differential,
)


class RoughEvolutionPolicy(StrictModule):
    """Static capability requirements for one finite-depth geometric rough solve."""

    candidate_solvers: tuple[AbstractRoughSolver, ...]
    order: int = eqx.field(static=True)
    p_variation_upper: float = eqx.field(static=True)
    vector_field_regularity: float = eqx.field(static=True)
    maximum_depth: int = eqx.field(static=True)
    differentiation: Literal["fixed-route"] = eqx.field(static=True)

    def __init__(
        self,
        *,
        order: int = 2,
        p_variation_upper: float = 3.0,
        vector_field_regularity: float = 3.0,
        candidate_solvers: tuple[AbstractRoughSolver, ...] = (Davie(), RoughEuler()),
        maximum_depth: int = 4,
        differentiation: Literal["fixed-route"] = "fixed-route",
    ):
        selected = tuple(candidate_solvers)
        if not selected or any(
            not isinstance(item, AbstractRoughSolver) for item in selected
        ):
            raise TypeError("candidate_solvers must contain AbstractRoughSolver values.")
        requested_order = int(order)
        depth = int(maximum_depth)
        roughness = float(p_variation_upper)
        regularity = float(vector_field_regularity)
        if requested_order <= 0 or depth <= 0:
            raise ValueError("order and maximum_depth must be positive.")
        if not all(isfinite(value) and value >= 1.0 for value in (roughness, regularity)):
            raise ValueError(
                "roughness and regularity bounds must be finite and at least one."
            )
        if differentiation != "fixed-route":
            raise ValueError("Only fixed-route rough differentiation is supported.")
        self.candidate_solvers = selected
        self.order = requested_order
        self.p_variation_upper = roughness
        self.vector_field_regularity = regularity
        self.maximum_depth = depth
        self.differentiation = differentiation


class PreparedRoughEvolution(StrictModule):
    """One capability-checked rough solver route; selection is frozen."""

    problem: RoughDifferentialProblem
    control: AbstractRoughControl
    solver: AbstractRoughSolver
    policy: RoughEvolutionPolicy
    prepared_id: str = eqx.field(static=True)
    capability_evidence: tuple[str, ...] = eqx.field(static=True)


def prepare_rough_evolution(
    problem: RoughDifferentialProblem,
    control: AbstractRoughControl,
    policy: RoughEvolutionPolicy,
    /,
) -> PreparedRoughEvolution:
    """Select an existing solver solely from declared finite capabilities."""
    if not isinstance(problem, RoughDifferentialProblem):
        raise TypeError("problem must be a RoughDifferentialProblem.")
    if not isinstance(control, AbstractRoughControl):
        raise TypeError("control must be an AbstractRoughControl.")
    if not isinstance(policy, RoughEvolutionPolicy):
        raise TypeError("policy must be a RoughEvolutionPolicy.")
    if control.depth > policy.maximum_depth:
        raise ValueError("control depth exceeds the declared preparation cap.")
    if control.dimension != problem.driver_dimension:
        raise ValueError("control and rough problem driver dimensions differ.")
    minimum_depth = max(1, min(policy.order, policy.maximum_depth))
    selected = None
    reasons = []
    for candidate in policy.candidate_solvers:
        if candidate.required_depth > control.depth:
            reasons.append(f"{candidate.solver_name}:insufficient-control-depth")
            continue
        if candidate.required_depth < minimum_depth:
            reasons.append(f"{candidate.solver_name}:insufficient-method-order")
            continue
        if policy.vector_field_regularity <= policy.p_variation_upper:
            reasons.append(f"{candidate.solver_name}:insufficient-declared-regularity")
            continue
        selected = candidate
        break
    if selected is None:
        raise ValueError(
            "No rough solver satisfies the declared capabilities: " + ", ".join(reasons)
        )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-rough-evolution-v1",
            "problem": problem.problem_id,
            "control": control.control_id,
            "solver": selected.solver_id,
            "order": policy.order,
            "p_variation_upper": policy.p_variation_upper,
            "vector_field_regularity": policy.vector_field_regularity,
            "maximum_depth": policy.maximum_depth,
        }
    )
    evidence = (
        f"geometric-control:{control.control_id}",
        f"control-depth:{control.depth}",
        f"required-depth:{selected.required_depth}",
        f"p-variation-upper:{policy.p_variation_upper}",
        f"vector-field-regularity:{policy.vector_field_regularity}",
        "selection-derivative:none",
    )
    return PreparedRoughEvolution(
        problem=problem,
        control=control,
        solver=selected,
        policy=policy,
        prepared_id=prepared_id,
        capability_evidence=evidence,
    )


def solve_prepared_rough(
    prepared: PreparedRoughEvolution,
    /,
    *,
    save_times=None,
):
    """Execute one frozen rough route with no fallback or reselection."""
    if not isinstance(prepared, PreparedRoughEvolution):
        raise TypeError("prepared must be a PreparedRoughEvolution.")
    return solve_rough_differential(
        prepared.problem,
        prepared.control,
        save_times=save_times,
        solver=prepared.solver,
    )


__all__ = [
    "PreparedRoughEvolution",
    "RoughEvolutionPolicy",
    "prepare_rough_evolution",
    "solve_prepared_rough",
]
