#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

import phydrax.ein as ein

from .._strict import StrictModule
from ..integration._targets import weighted
from ..ml.mixture import GaussianMixtureModel
from ._costs import PrecomputedCost
from ._problem import discrete_problem
from ._results import AbstractBalancedTransportSolver


class GaussianMixtureTransportProblem(StrictModule):
    source: GaussianMixtureModel
    target: GaussianMixtureModel
    component_costs: Array
    component_valid: Array
    mass_tolerance: float = eqx.field(static=True)


class GaussianMixtureTransportResult(StrictModule):
    problem: GaussianMixtureTransportProblem
    component_plan: Any
    coupling: Array
    objective: Array
    valid: Array
    status: Array
    approximation_kind: str = eqx.field(static=True)
    bounded_non_claim: str = eqx.field(static=True)


def _psd_square_root(matrix: Array, tolerance: float, /) -> tuple[Array, Array, Array]:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = jnp.linalg.eigh(symmetric)
    valid = jnp.all(jnp.isfinite(symmetric)) & jnp.all(eigenvalues >= -tolerance)
    safe_values = jnp.where(
        valid, jnp.maximum(eigenvalues, 0.0), jnp.ones_like(eigenvalues)
    )
    root = ein.contract("ik,k,jk->ij", eigenvectors, jnp.sqrt(safe_values), eigenvectors)
    rank = jnp.sum(eigenvalues > tolerance).astype(jnp.int32)
    return root, rank, valid


def _gaussian_w2_cost(
    left_mean: Array,
    left_covariance: Array,
    right_mean: Array,
    right_covariance: Array,
    tolerance: float,
    /,
) -> tuple[Array, Array]:
    left_root, _, left_valid = _psd_square_root(left_covariance, tolerance)
    _, _, right_valid = _psd_square_root(right_covariance, tolerance)
    middle = left_root @ right_covariance @ left_root
    middle_root, _, middle_valid = _psd_square_root(middle, tolerance)
    difference = left_mean - right_mean
    value = ein.contract("i,i->", difference, difference) + jnp.trace(
        left_covariance + right_covariance - 2.0 * middle_root
    )
    valid = (
        left_valid
        & right_valid
        & middle_valid
        & jnp.isfinite(value)
        & (value >= -tolerance)
    )
    return jnp.where(valid, jnp.maximum(value, 0.0), jnp.nan), valid


def gaussian_mixture_transport_problem(
    source: GaussianMixtureModel,
    target: GaussianMixtureModel,
    /,
    *,
    mass_tolerance: float = 1.0e-8,
) -> GaussianMixtureTransportProblem:
    """Build finite Gaussian-component W2 costs for two existing GMMs."""
    if not isinstance(source, GaussianMixtureModel) or not isinstance(
        target, GaussianMixtureModel
    ):
        raise TypeError("source and target must be GaussianMixtureModel values.")
    if source.case_shape or target.case_shape:
        raise ValueError("Gaussian mixture transport requires one case per problem.")
    if source.in_size != target.in_size:
        raise ValueError("Gaussian mixture event dimensions differ.")
    if source.covariance.ndim != 3 or target.covariance.ndim != 3:
        raise ValueError("Gaussian mixture transport requires explicit full covariances.")
    tolerance = float(mass_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("mass_tolerance must be finite and nonnegative.")

    def one_left(left_mean, left_covariance):
        return jax.vmap(
            lambda right_mean, right_covariance: _gaussian_w2_cost(
                left_mean,
                left_covariance,
                right_mean,
                right_covariance,
                tolerance,
            )
        )(target.means, target.covariance)

    costs, valid = jax.vmap(one_left)(source.means, source.covariance)
    return GaussianMixtureTransportProblem(
        source=source,
        target=target,
        component_costs=costs,
        component_valid=valid,
        mass_tolerance=tolerance,
    )


def solve_gaussian_mixture_transport(
    problem: GaussianMixtureTransportProblem,
    solver: AbstractBalancedTransportSolver,
    /,
) -> GaussianMixtureTransportResult:
    """Transport component masses and report the induced mixture-W2 upper bound."""
    if not isinstance(problem, GaussianMixtureTransportProblem):
        raise TypeError("problem must be a GaussianMixtureTransportProblem.")
    if not isinstance(solver, AbstractBalancedTransportSolver):
        raise TypeError("solver must be a balanced finite transport solver.")
    if not bool(jnp.all(problem.component_valid)):
        raise ValueError("At least one Gaussian component covariance is not PSD.")
    source_target = weighted(
        problem.source.means,
        jnp.log(problem.source.mixing_weights),
        normalized=True,
        independent=False,
        sample_axes=0,
        provenance="gaussian-mixture-source-components",
    )
    target_target = weighted(
        problem.target.means,
        jnp.log(problem.target.mixing_weights),
        normalized=True,
        independent=False,
        sample_axes=0,
        provenance="gaussian-mixture-target-components",
    )
    represented = discrete_problem(
        source_target,
        target_target,
        cost=PrecomputedCost(problem.component_costs, cost_id="gaussian-component-w2"),
        mass_tolerance=problem.mass_tolerance,
    )
    component_plan = solver(represented)
    coupling = component_plan.dense_plan()
    objective = ein.contract("ij,ij->", coupling, problem.component_costs)
    single = problem.source.out_size == 1 and problem.target.out_size == 1
    valid = component_plan.converged & jnp.isfinite(objective)
    return GaussianMixtureTransportResult(
        problem=problem,
        component_plan=component_plan,
        coupling=coupling,
        objective=objective,
        valid=valid,
        status=jnp.where(valid, 0, 1).astype(jnp.int32),
        approximation_kind=(
            "exact-single-gaussian-w2"
            if single
            else "gaussian-component-coupling-upper-bound"
        ),
        bounded_non_claim=(
            "For multiple components this is the component-coupling Gaussian-W2 "
            "upper bound, not the exact Wasserstein distance between mixture densities."
        ),
    )


__all__ = [
    "GaussianMixtureTransportProblem",
    "GaussianMixtureTransportResult",
    "gaussian_mixture_transport_problem",
    "solve_gaussian_mixture_transport",
]
