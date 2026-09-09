#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.random as jr
import optax

from ..._strict import StrictModule
from ...terms import AugmentedInterfaceEvidence, AugmentedValueConstraint
from .._functional_solver import FunctionalSolver


class AugmentedInterfacePlan(StrictModule):
    outer_iterations: int = eqx.field(static=True)
    inner_iterations: int = eqx.field(static=True)
    primal_tolerance: float = eqx.field(static=True)
    dual_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        outer_iterations: int,
        inner_iterations: int,
        /,
        *,
        primal_tolerance: float = 1.0e-6,
        dual_tolerance: float = 1.0e-6,
    ):
        outer = int(outer_iterations)
        inner = int(inner_iterations)
        primal = float(primal_tolerance)
        dual = float(dual_tolerance)
        if outer <= 0 or inner <= 0:
            raise ValueError("Augmented interface work counts must be positive.")
        if (
            not math.isfinite(primal)
            or not math.isfinite(dual)
            or primal < 0.0
            or dual < 0.0
        ):
            raise ValueError(
                "Augmented interface tolerances must be finite and non-negative."
            )
        self.outer_iterations = outer
        self.inner_iterations = inner
        self.primal_tolerance = primal
        self.dual_tolerance = dual


class AugmentedInterfaceResult(StrictModule):
    solver: FunctionalSolver
    constraint: AugmentedValueConstraint
    history: tuple[AugmentedInterfaceEvidence, ...]
    converged: bool = eqx.field(static=True)

    def __init__(
        self,
        solver: FunctionalSolver,
        constraint: AugmentedValueConstraint,
        history: tuple[AugmentedInterfaceEvidence, ...],
        /,
        *,
        converged: bool,
    ):
        self.solver = solver
        self.constraint = constraint
        self.history = tuple(history)
        self.converged = bool(converged)


def solve_augmented_interface(
    solver: FunctionalSolver,
    constraint_index: int,
    plan: AugmentedInterfacePlan,
    optimizer: optax.GradientTransformation | optax.GradientTransformationExtraArgs,
    /,
    *,
    seed: int = 0,
    jit: bool = True,
) -> AugmentedInterfaceResult:
    """Alternate functional minimization and accepted interface multiplier updates."""
    if not isinstance(solver, FunctionalSolver):
        raise TypeError("solver must be a FunctionalSolver.")
    if not isinstance(plan, AugmentedInterfacePlan):
        raise TypeError("plan must be an AugmentedInterfacePlan.")
    index = int(constraint_index)
    if not 0 <= index < len(solver.terms):
        raise IndexError("constraint_index is outside the solver term collection.")
    constraint = solver.terms[index]
    if not isinstance(constraint, AugmentedValueConstraint):
        raise TypeError("constraint_index must select an AugmentedValueConstraint.")
    current = solver
    history = []
    converged = False
    for outer in range(plan.outer_iterations):
        current = current.solve(
            num_iter=plan.inner_iterations,
            optim=optimizer,
            seed=seed + outer,
            jit=jit,
            keep_best=False,
            log_every=0,
        )
        constraint, evidence = constraint.update(current.functions)
        history.append(evidence)
        terms = list(current.terms)
        terms[index] = constraint
        current = FunctionalSolver(
            functions=current.functions,
            terms=tuple(terms),
            evaluation_terms=current.evaluation_terms,
            enforcement=current.enforcement,
            collocation_key=jr.key(seed + outer + 1),
        )
        if (
            float(evidence.primal_residual) <= plan.primal_tolerance
            and float(evidence.dual_residual) <= plan.dual_tolerance
        ):
            converged = True
            break
    return AugmentedInterfaceResult(
        current,
        constraint,
        tuple(history),
        converged=converged,
    )


__all__ = [
    "AugmentedInterfacePlan",
    "AugmentedInterfaceResult",
    "solve_augmented_interface",
]
