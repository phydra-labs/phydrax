#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._costs import PrecomputedCost
from ._measure import lower_transport_measure
from ._problem import DiscreteTransportProblem
from ._results import AbstractBalancedTransportSolver
from ._sinkhorn import Sinkhorn


class GromovWassersteinProblem(StrictModule):
    """Two finite measures with explicit within-space relational costs."""

    source: Any
    target: Any
    source_relation: Array
    target_relation: Array
    feature_cost: Array | None
    alpha: Array


class GromovWassersteinResult(StrictModule):
    """Finite nonconvex local solve with fixed outer-iteration provenance."""

    problem: GromovWassersteinProblem
    coupling: Array
    objective: Array
    objective_history: Array
    coupling_change: Array
    marginal_residual: Array
    stationarity_surrogate: Array
    inner_converged: Array
    valid: Array
    status: Array
    approximation_kind: str = eqx.field(static=True)
    bounded_non_claim: str = eqx.field(static=True)


def gromov_wasserstein_problem(
    source: Any,
    target: Any,
    /,
    *,
    source_relation: ArrayLike,
    target_relation: ArrayLike,
    feature_cost: ArrayLike | None = None,
    alpha: float = 1.0,
    encoders: tuple[Any, Any] | None = None,
) -> GromovWassersteinProblem:
    """Lower two existing finite measures into one represented GW problem."""
    source_encoder, target_encoder = (None, None) if encoders is None else encoders
    source_measure = lower_transport_measure(
        source, encoder=source_encoder, name="source"
    )
    target_measure = lower_transport_measure(
        target, encoder=target_encoder, name="target"
    )
    left = jnp.asarray(source_relation, dtype=float)
    right = jnp.asarray(target_relation, dtype=float)
    if left.shape != (source_measure.num_atoms, source_measure.num_atoms):
        raise ValueError("source_relation must be square on source atoms.")
    if right.shape != (target_measure.num_atoms, target_measure.num_atoms):
        raise ValueError("target_relation must be square on target atoms.")
    if not bool(jnp.all(jnp.isfinite(left))) or not bool(jnp.all(jnp.isfinite(right))):
        raise ValueError("relation costs must be finite.")
    if not bool(jnp.allclose(left, left.T)) or not bool(jnp.allclose(right, right.T)):
        raise ValueError("GW relational costs must be symmetric.")
    feature = None if feature_cost is None else jnp.asarray(feature_cost, dtype=float)
    if feature is not None and feature.shape != (
        source_measure.num_atoms,
        target_measure.num_atoms,
    ):
        raise ValueError("feature_cost must align source and target atoms.")
    if feature is not None and (
        not bool(jnp.all(jnp.isfinite(feature))) or bool(jnp.any(feature < 0.0))
    ):
        raise ValueError("feature_cost must be finite and nonnegative.")
    mixing = float(alpha)
    if not isfinite(mixing) or not 0.0 <= mixing <= 1.0:
        raise ValueError("alpha must lie in [0, 1].")
    if mixing < 1.0 and feature is None:
        raise ValueError("fused GW with alpha < 1 requires feature_cost.")
    return GromovWassersteinProblem(
        source=source_measure,
        target=target_measure,
        source_relation=left,
        target_relation=right,
        feature_cost=feature,
        alpha=jnp.asarray(mixing),
    )


def _relational_squared_loss(
    problem: GromovWassersteinProblem,
    coupling: Array,
    /,
) -> Array:
    source_probabilities = problem.source.probabilities
    target_probabilities = problem.target.probabilities
    left_square = problem.source_relation**2 @ source_probabilities
    right_square = problem.target_relation**2 @ target_probabilities
    cross = oe.contract(
        "ik,kl,jl->ij",
        problem.source_relation,
        coupling,
        problem.target_relation,
    )
    return left_square[:, None] + right_square[None, :] - 2.0 * cross


def _linearized_cost(
    problem: GromovWassersteinProblem,
    coupling: Array,
    /,
) -> Array:
    relational = _relational_squared_loss(problem, coupling)
    if problem.feature_cost is None:
        return 2.0 * relational
    return 2.0 * problem.alpha * relational + (1.0 - problem.alpha) * problem.feature_cost


def _gromov_objective(
    problem: GromovWassersteinProblem,
    coupling: Array,
    /,
) -> Array:
    relational = oe.contract(
        "ij,ij->",
        coupling,
        _relational_squared_loss(problem, coupling),
    )
    if problem.feature_cost is None:
        return relational
    feature = oe.contract("ij,ij->", coupling, problem.feature_cost)
    return problem.alpha * relational + (1.0 - problem.alpha) * feature


def _quantile_coupling(
    source_probabilities: Array,
    target_probabilities: Array,
    /,
) -> Array:
    source_upper = jnp.cumsum(source_probabilities)
    target_upper = jnp.cumsum(target_probabilities)
    source_lower = source_upper - source_probabilities
    target_lower = target_upper - target_probabilities
    return jnp.maximum(
        jnp.minimum(source_upper[:, None], target_upper[None, :])
        - jnp.maximum(source_lower[:, None], target_lower[None, :]),
        0.0,
    )


class GromovWasserstein(StrictModule):
    """Fixed-iteration entropic GW/fused-GW local solver."""

    inner_solver: AbstractBalancedTransportSolver
    max_outer_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    differentiation: str = eqx.field(static=True)

    def __init__(
        self,
        epsilon: float,
        /,
        *,
        max_outer_iterations: int = 50,
        tolerance: float = 1.0e-7,
        inner_solver: AbstractBalancedTransportSolver | None = None,
        block_size: int | None = None,
        differentiation: str = "fixed-iterations",
    ):
        iterations = int(max_outer_iterations)
        threshold = float(tolerance)
        if iterations <= 0 or not isfinite(threshold) or threshold < 0.0:
            raise ValueError(
                "outer iterations must be positive and tolerance nonnegative."
            )
        if differentiation != "fixed-iterations":
            raise ValueError("GW supports fixed-iterations differentiation only.")
        solver = (
            Sinkhorn(epsilon, block_size=block_size)
            if inner_solver is None
            else inner_solver
        )
        if not isinstance(solver, AbstractBalancedTransportSolver):
            raise TypeError("inner_solver must be a balanced finite transport solver.")
        self.inner_solver = solver
        self.max_outer_iterations = iterations
        self.tolerance = threshold
        self.differentiation = differentiation

    def __call__(self, problem: GromovWassersteinProblem, /) -> GromovWassersteinResult:
        if not isinstance(problem, GromovWassersteinProblem):
            raise TypeError("problem must be a GromovWassersteinProblem.")
        coupling = _quantile_coupling(
            problem.source.probabilities,
            problem.target.probabilities,
        )
        objectives = []
        changes = []
        convergences = []
        for outer in range(self.max_outer_iterations):
            linear_cost = _linearized_cost(problem, coupling)
            finite_nonnegative = jnp.all(jnp.isfinite(linear_cost)) & jnp.all(
                linear_cost >= 0.0
            )
            safe_cost = jnp.where(
                finite_nonnegative, linear_cost, jnp.full_like(linear_cost, jnp.nan)
            )
            linear_problem = DiscreteTransportProblem(
                problem.source,
                problem.target,
                PrecomputedCost(safe_cost, cost_id=f"gw-linearization:{outer}"),
            )
            inner = self.inner_solver(linear_problem)
            physical_coupling = inner.dense_plan()
            updated = physical_coupling / jnp.sum(physical_coupling)
            change = jnp.sqrt(jnp.sum((updated - coupling) ** 2))
            coupling = updated
            objective = _gromov_objective(problem, coupling)
            objectives.append(objective)
            changes.append(change)
            convergences.append(inner.converged)
        objective_history = jnp.stack(objectives)
        coupling_change = jnp.stack(changes)
        inner_converged = jnp.stack(convergences)
        source_residual = jnp.max(
            jnp.abs(jnp.sum(coupling, axis=-1) - problem.source.probabilities)
        )
        target_residual = jnp.max(
            jnp.abs(jnp.sum(coupling, axis=-2) - problem.target.probabilities)
        )
        marginal_residual = jnp.maximum(source_residual, target_residual)
        stationarity = coupling_change[-1]
        valid = (
            jnp.all(inner_converged)
            & jnp.all(jnp.isfinite(objective_history))
            & jnp.isfinite(marginal_residual)
            & (marginal_residual <= self.tolerance)
        )
        status = jnp.where(valid, 0, jnp.where(~jnp.all(inner_converged), 1, 2)).astype(
            jnp.int32
        )
        return GromovWassersteinResult(
            problem=problem,
            coupling=coupling,
            objective=objective_history[-1],
            objective_history=objective_history,
            coupling_change=coupling_change,
            marginal_residual=marginal_residual,
            stationarity_surrogate=stationarity,
            inner_converged=inner_converged,
            valid=valid,
            status=status,
            approximation_kind="finite-entropic-gw-local-solve",
            bounded_non_claim=(
                "GW and fused GW are nonconvex finite local solves; convergence does "
                "not certify a global minimizer."
            ),
        )


__all__ = [
    "GromovWasserstein",
    "GromovWassersteinProblem",
    "GromovWassersteinResult",
    "gromov_wasserstein_problem",
]
