#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certified Cayley TDVP for a declared finite linear variational subspace."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    HermitianSpectrum,
    PreparedFactorization,
)


class FiniteVariationalSubspaceTDVPProblem(StrictModule):
    overlap: Array
    hamiltonian: Array
    initial_coefficients: Array
    overlap_spectrum: HermitianSpectrum
    overlap_hermiticity_residual: Array
    hamiltonian_hermiticity_residual: Array
    valid: Array
    dimension: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)

    def __init__(
        self,
        overlap: ArrayLike,
        hamiltonian: ArrayLike,
        initial_coefficients: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
        problem_id: str,
    ):
        metric, operator, coefficients = map(
            jnp.asarray, (overlap, hamiltonian, initial_coefficients)
        )
        if (
            metric.ndim != 2
            or metric.shape[0] != metric.shape[1]
            or operator.shape != metric.shape
        ):
            raise ValueError(
                "overlap and hamiltonian must be matching nonempty square matrices."
            )
        dimension = int(metric.shape[0])
        if coefficients.shape != (dimension,):
            raise ValueError(
                "initial_coefficients dimension must match the finite subspace."
            )
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError("problem_id must be nonempty.")
        overlap_residual = jnp.max(jnp.abs(metric - jnp.conj(metric.T)))
        hamiltonian_residual = jnp.max(jnp.abs(operator - jnp.conj(operator.T)))
        spectrum = HermitianSpectrum(
            0.5 * (metric + jnp.conj(metric.T)), tolerance=tolerance
        )
        valid = (
            spectrum.valid
            & (spectrum.minimum_eigenvalue > tolerance)
            & (overlap_residual <= tolerance)
            & (hamiltonian_residual <= tolerance)
            & jnp.all(jnp.isfinite(coefficients))
        )
        self.overlap = metric
        self.hamiltonian = operator
        self.initial_coefficients = coefficients
        self.overlap_spectrum = spectrum
        self.overlap_hermiticity_residual = overlap_residual
        self.hamiltonian_hermiticity_residual = hamiltonian_residual
        self.valid = valid
        self.dimension = dimension
        self.problem_id = problem_id
        self.claim = "finite-linear-time-independent-hermitian-subspace"


class FiniteSubspaceTDVPPlan(StrictModule):
    left_factorization: PreparedFactorization
    left_matrix: Array
    right_matrix: Array
    step_size: float = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)


class FiniteSubspaceTDVPResult(StrictModule):
    times: Array
    coefficients: Array
    metric_norms: Array
    energies: Array
    solve_residuals: Array
    norm_drifts: Array
    energy_drifts: Array
    reversibility_residual: Array
    valid_steps: Array
    valid: Array
    problem_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def prepare_finite_subspace_tdvp(
    problem: FiniteVariationalSubspaceTDVPProblem,
    /,
    *,
    step_size: float,
    num_steps: int,
    tolerance: float = 1e-9,
) -> FiniteSubspaceTDVPPlan:
    if not isinstance(problem, FiniteVariationalSubspaceTDVPProblem):
        raise TypeError("problem must be FiniteVariationalSubspaceTDVPProblem.")
    step, steps, threshold = float(step_size), int(num_steps), float(tolerance)
    if not np.isfinite(step) or step == 0.0 or steps < 0 or threshold < 0.0:
        raise ValueError("step_size/num_steps/tolerance are invalid.")
    left = problem.overlap + 0.5j * step * problem.hamiltonian
    right = problem.overlap - 0.5j * step * problem.hamiltonian
    precision_floor = (
        np.finfo(np.dtype(left.real.dtype)).eps * problem.dimension * 2 * max(steps, 1)
    )
    threshold = max(threshold, float(precision_floor))
    prepared = factorize(DenseLinearOperator(left), FactorizationPolicy("lu"))
    return FiniteSubspaceTDVPPlan(
        left_factorization=prepared,
        left_matrix=left,
        right_matrix=right,
        step_size=step,
        num_steps=steps,
        tolerance=threshold,
        problem_id=problem.problem_id,
    )


def _metric_norm(problem, coefficients):
    return jnp.real(jnp.vdot(coefficients, problem.overlap @ coefficients))


def _energy(problem, coefficients):
    return jnp.real(jnp.vdot(coefficients, problem.hamiltonian @ coefficients))


def solve_finite_subspace_tdvp(
    problem: FiniteVariationalSubspaceTDVPProblem,
    plan: FiniteSubspaceTDVPPlan,
    /,
) -> FiniteSubspaceTDVPResult:
    """Apply fixed Cayley solves and retain algebraic invariant residuals."""
    if not isinstance(problem, FiniteVariationalSubspaceTDVPProblem) or not isinstance(
        plan, FiniteSubspaceTDVPPlan
    ):
        raise TypeError("problem/plan types are invalid.")
    if plan.problem_id != problem.problem_id:
        raise ValueError("plan was prepared for a different finite-subspace problem.")
    coefficients = problem.initial_coefficients
    trajectory = [coefficients]
    residuals = []
    step_validity = []
    for _ in range(plan.num_steps):
        solve = plan.left_factorization.solve(plan.right_matrix @ coefficients)
        proposed = jnp.asarray(solve.value)
        valid = problem.valid & solve.successful & jnp.all(jnp.isfinite(proposed))
        coefficients = jnp.where(valid, proposed, jnp.full_like(proposed, jnp.nan))
        trajectory.append(coefficients)
        residuals.append(solve.diagnostics.residual_norm)
        step_validity.append(valid)
    history = jnp.stack(trajectory)
    norms = jnp.stack([_metric_norm(problem, value) for value in trajectory])
    energies = jnp.stack([_energy(problem, value) for value in trajectory])
    # The inverse Cayley map is obtained by swapping the two prepared matrices.
    reverse_factor = factorize(
        DenseLinearOperator(plan.right_matrix), FactorizationPolicy("lu")
    )
    back = coefficients
    reverse_validity = []
    for _ in range(plan.num_steps):
        reverse_solve = reverse_factor.solve(plan.left_matrix @ back)
        proposed_back = jnp.asarray(reverse_solve.value)
        reverse_valid = reverse_solve.successful & jnp.all(jnp.isfinite(proposed_back))
        back = jnp.where(
            reverse_valid, proposed_back, jnp.full_like(proposed_back, jnp.nan)
        )
        reverse_validity.append(reverse_valid)
    reversibility = jnp.max(jnp.abs(back - problem.initial_coefficients))
    valid_steps = (
        jnp.stack(step_validity) if step_validity else jnp.empty((0,), dtype=bool)
    )
    residual_array = jnp.stack(residuals) if residuals else jnp.empty((0,))
    reverse_valid = (
        jnp.all(jnp.stack(reverse_validity)) if reverse_validity else jnp.asarray(True)
    )
    residuals_valid = jnp.all(jnp.isfinite(residual_array)) & jnp.all(
        residual_array <= plan.tolerance
    )
    valid = (
        problem.valid
        & jnp.all(valid_steps)
        & reverse_valid
        & residuals_valid
        & (reversibility <= plan.tolerance)
    )
    return FiniteSubspaceTDVPResult(
        times=plan.step_size * jnp.arange(plan.num_steps + 1),
        coefficients=history,
        metric_norms=norms,
        energies=energies,
        solve_residuals=residual_array,
        norm_drifts=norms - norms[0],
        energy_drifts=energies - energies[0],
        reversibility_residual=reversibility,
        valid_steps=valid_steps,
        valid=valid,
        problem_id=problem.problem_id,
        claim="cayley-norm-energy-preserving-up-to-retained-solve-residual",
    )


__all__ = [
    "FiniteSubspaceTDVPPlan",
    "FiniteSubspaceTDVPResult",
    "FiniteVariationalSubspaceTDVPProblem",
    "prepare_finite_subspace_tdvp",
    "solve_finite_subspace_tdvp",
]
