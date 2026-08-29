#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from .._hermitian_spectral import HermitianSpectrum
from .._operators import DenseLinearOperator
from .._policies import (
    DenseCholesky,
    DifferentiationPolicy,
    FailurePolicy,
    LinearSolvePolicy,
)
from .._problems import LinearSystem
from .._properties import OperatorProperties
from .._runtime import solve as solve_linear
from .._spaces import RHSLayout
from ._policies import DenseEigh, EigenSolvePolicy
from ._problems import Eigenproblem, EigenproblemLike, GeneralizedEigenproblem
from ._results import EigenSolveResult
from ._runtime import eigensolve


def _adjoint(value: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(value), -1, -2)


def _square_pair(
    stiffness: ArrayLike,
    mass: ArrayLike,
    /,
) -> tuple[Array, Array]:
    stiffness_ = jnp.asarray(stiffness)
    mass_ = jnp.asarray(mass)
    if stiffness_.ndim != 2 or stiffness_.shape[0] != stiffness_.shape[1]:
        raise ValueError("stiffness must be one square matrix.")
    if mass_.shape != stiffness_.shape:
        raise ValueError("mass must have the same square shape as stiffness.")
    if not jnp.issubdtype(stiffness_.dtype, jnp.inexact) or not jnp.issubdtype(
        mass_.dtype, jnp.inexact
    ):
        raise TypeError("Ritz matrices must use real or complex inexact dtypes.")
    dtype = jnp.result_type(stiffness_, mass_)
    return stiffness_.astype(dtype), mass_.astype(dtype)


def _hermiticity_residual(value: Array, /) -> Array:
    scale = jnp.maximum(jnp.max(jnp.abs(value)), 1.0)
    return jnp.max(jnp.abs(value - _adjoint(value))) / scale


def _gram_properties(*, positive_definite: bool) -> OperatorProperties:
    evidence = {"self_adjoint": "asserted"}
    if positive_definite:
        evidence["positive_definite"] = "asserted"
    return OperatorProperties(
        self_adjoint=True,
        positive_definite=positive_definite,
        evidence=evidence,
    )


class BlockRayleighEvaluation(StrictModule):
    """Block trace quotient plus rank, conditioning, and structure evidence."""

    stiffness: Array
    mass: Array
    solved_stiffness: Array
    objective: Array
    imaginary_objective: Array
    stiffness_hermiticity_residual: Array
    mass_hermiticity_residual: Array
    mass_minimum_eigenvalue: Array
    mass_condition_number: Array
    mass_numerical_rank: Array
    valid: Array
    tolerance: float = eqx.field(static=True)

    @property
    def dimension(self) -> int:
        return int(self.mass.shape[0])


class ReducedRitzResult(StrictModule):
    """A reduced generalized eigensolve tied to its block quotient evidence."""

    evaluation: BlockRayleighEvaluation
    solve: EigenSolveResult

    @property
    def eigenvalues(self) -> Array:
        return self.solve.eigenvalues

    @property
    def coefficients(self) -> Array:
        return jnp.asarray(self.solve.eigenvectors)

    @property
    def successful(self) -> Array:
        return self.evaluation.valid & self.solve.successful


class TrialSubspaceRitzResult(StrictModule):
    """Ritz pairs lifted into the original space with full residual evidence."""

    reduced: ReducedRitzResult
    trial_basis: Array
    operator_basis: Array
    metric_basis: Array
    eigenvectors: Array
    residuals: Array
    residual_norms: Array
    relative_residuals: Array
    metric_orthogonality_error: Array

    @property
    def eigenvalues(self) -> Array:
        return self.reduced.eigenvalues

    @property
    def mode_mask(self) -> Array:
        return self.reduced.solve.mode_mask

    @property
    def valid(self) -> Array:
        return self.reduced.successful & jnp.all(jnp.isfinite(self.relative_residuals))


class WarmStartedEigenResult(StrictModule):
    """Certified trial-space extraction followed by the authoritative eigensolve."""

    trial: TrialSubspaceRitzResult
    solve: EigenSolveResult

    @property
    def successful(self) -> Array:
        return self.trial.valid & self.solve.successful


def block_rayleigh_trace(
    stiffness: ArrayLike,
    mass: ArrayLike,
    /,
    *,
    tolerance: float = 1e-10,
) -> BlockRayleighEvaluation:
    """Evaluate ``Re trace(M⁻¹ K)`` without differentiating eigenvectors."""
    tolerance_ = float(tolerance)
    if tolerance_ < 0.0:
        raise ValueError("tolerance must be non-negative.")
    stiffness_, mass_ = _square_pair(stiffness, mass)
    stiffness_defect = _hermiticity_residual(stiffness_)
    mass_defect = _hermiticity_residual(mass_)
    finite = jnp.all(jnp.isfinite(stiffness_)) & jnp.all(jnp.isfinite(mass_))
    guarded_stiffness = eqx.error_if(
        stiffness_,
        stiffness_defect > tolerance_,
        "Block Rayleigh stiffness matrix is not Hermitian within tolerance.",
    )
    guarded_mass = eqx.error_if(
        mass_,
        mass_defect > tolerance_,
        "Block Rayleigh mass matrix is not Hermitian within tolerance.",
    )
    hermitian_stiffness = 0.5 * (guarded_stiffness + _adjoint(guarded_stiffness))
    hermitian_mass = 0.5 * (guarded_mass + _adjoint(guarded_mass))
    operator = DenseLinearOperator(
        hermitian_mass,
        properties=_gram_properties(positive_definite=True),
        operator_id="block-rayleigh-mass",
    )
    linear = solve_linear(
        LinearSystem(operator, problem_id="block-rayleigh-trace"),
        hermitian_stiffness,
        policy=LinearSolvePolicy(
            DenseCholesky(),
            differentiation=DifferentiationPolicy("mathematical"),
            failure=FailurePolicy("status"),
        ),
        rhs_layout=RHSLayout((int(hermitian_stiffness.shape[1]),)),
    )
    solved = jnp.asarray(linear.value)
    trace = jnp.trace(solved)
    objective = jnp.real(trace)
    imaginary = jnp.abs(jnp.imag(trace))
    spectrum = HermitianSpectrum(
        jax.lax.stop_gradient(hermitian_mass),
        tolerance=tolerance_,
    )
    positive = spectrum.minimum_eigenvalue > tolerance_
    full_rank = spectrum.numerical_rank == int(hermitian_mass.shape[0])
    valid = (
        finite
        & (stiffness_defect <= tolerance_)
        & (mass_defect <= tolerance_)
        & jnp.all(linear.successful)
        & spectrum.valid
        & positive
        & full_rank
        & jnp.isfinite(objective)
        & jnp.isfinite(imaginary)
    )
    return BlockRayleighEvaluation(
        stiffness=hermitian_stiffness,
        mass=hermitian_mass,
        solved_stiffness=solved,
        objective=objective,
        imaginary_objective=imaginary,
        stiffness_hermiticity_residual=stiffness_defect,
        mass_hermiticity_residual=mass_defect,
        mass_minimum_eigenvalue=spectrum.minimum_eigenvalue,
        mass_condition_number=spectrum.condition_number,
        mass_numerical_rank=spectrum.numerical_rank,
        valid=valid,
        tolerance=tolerance_,
    )


def solve_reduced_ritz(
    stiffness: ArrayLike,
    mass: ArrayLike,
    /,
    *,
    count: int | None = None,
    which: str = "smallest-algebraic",
    tolerance: float = 1e-10,
) -> ReducedRitzResult:
    """Solve one dense Hermitian Ritz pencil through the native eigen runtime."""
    evaluation = block_rayleigh_trace(stiffness, mass, tolerance=tolerance)
    guarded_stiffness = eqx.error_if(
        evaluation.stiffness,
        ~evaluation.valid,
        "Reduced Ritz matrices failed structure or positive-definiteness checks.",
    )
    dimension = evaluation.dimension
    count_ = dimension if count is None else int(count)
    if count_ < 1 or count_ > dimension:
        raise ValueError("count must lie between one and the Ritz dimension.")
    operator = DenseLinearOperator(
        guarded_stiffness,
        properties=_gram_properties(positive_definite=False),
        operator_id="reduced-ritz-stiffness",
    )
    metric = DenseLinearOperator(
        evaluation.mass,
        source=operator.source,
        target=operator.target,
        properties=_gram_properties(positive_definite=True),
        operator_id="reduced-ritz-mass",
    )
    problem = GeneralizedEigenproblem(
        operator,
        metric,
        problem_id="reduced-rayleigh-ritz",
    )
    solve = eigensolve(
        problem,
        policy=EigenSolvePolicy(
            DenseEigh(),
            count=count_,
            which=which,
            differentiation="none",
            failure=FailurePolicy("status"),
        ),
    )
    return ReducedRitzResult(evaluation=evaluation, solve=solve)


def _operator_columns(operator: Any, space: Any, basis: Array, /) -> Array:
    def apply(column):
        return space.flatten(operator.mv(space.unflatten(column)))

    return jax.vmap(apply, in_axes=1, out_axes=1)(basis)


def _pairing_matrix(space: Any, left: Array, right: Array, /) -> Array:
    def row(left_column):
        left_vector = space.unflatten(left_column)
        return jax.vmap(
            lambda right_column: space.inner(
                left_vector,
                space.unflatten(right_column),
            ),
            in_axes=1,
        )(right)

    return jax.vmap(row, in_axes=1)(left)


def _column_norms(space: Any, values: Array, /) -> Array:
    def norm(column):
        vector = space.unflatten(column)
        return jnp.sqrt(jnp.maximum(jnp.real(space.inner(vector, vector)), 0.0))

    return jax.vmap(norm, in_axes=1)(values)


def rayleigh_ritz(
    problem: EigenproblemLike,
    trial_basis: ArrayLike,
    /,
    *,
    count: int | None = None,
    which: str = "smallest-algebraic",
    tolerance: float = 1e-10,
) -> TrialSubspaceRitzResult:
    """Extract and certify Ritz pairs from any coordinate trial basis."""
    if not isinstance(problem, (Eigenproblem, GeneralizedEigenproblem)):
        raise TypeError("problem must be an Eigenproblem or GeneralizedEigenproblem.")
    if problem.batch_shape:
        raise ValueError("rayleigh_ritz does not accept batched eigenproblems.")
    space = problem.operator.source
    basis = jnp.asarray(trial_basis)
    if basis.ndim != 2 or basis.shape[0] != space.size or basis.shape[1] < 1:
        raise ValueError(
            "trial_basis must have shape (problem.dimension, positive trial count)."
        )
    if not jnp.issubdtype(basis.dtype, jnp.inexact):
        raise TypeError("trial_basis must use a real or complex inexact dtype.")
    space.unflatten(basis[:, 0])
    if problem.constraints is not None:
        basis = jax.vmap(
            lambda column: space.flatten(
                problem.constraints.orthogonal_component(space.unflatten(column))
            ),
            in_axes=1,
            out_axes=1,
        )(basis)
    operator_basis = _operator_columns(problem.operator, space, basis)
    if isinstance(problem, GeneralizedEigenproblem):
        metric_basis = _operator_columns(problem.metric_operator, space, basis)
    else:
        metric_basis = basis
    stiffness = _pairing_matrix(space, basis, operator_basis)
    mass = _pairing_matrix(space, basis, metric_basis)
    reduced = solve_reduced_ritz(
        stiffness,
        mass,
        count=count,
        which=which,
        tolerance=tolerance,
    )
    coefficients = reduced.coefficients
    eigenvectors = basis @ coefficients
    lifted_operator = operator_basis @ coefficients
    lifted_metric = metric_basis @ coefficients
    residuals = lifted_operator - lifted_metric * reduced.eigenvalues[None, :]
    residual_norms = _column_norms(space, residuals)
    operator_norms = _column_norms(space, lifted_operator)
    metric_norms = _column_norms(space, lifted_metric)
    tiny = jnp.finfo(residual_norms.dtype).tiny
    relative = residual_norms / jnp.maximum(
        operator_norms + jnp.abs(reduced.eigenvalues) * metric_norms,
        tiny,
    )
    gram = _pairing_matrix(space, eigenvectors, lifted_metric)
    identity = jnp.eye(gram.shape[0], dtype=gram.dtype)
    orthogonality = jnp.max(jnp.abs(gram - identity))
    return TrialSubspaceRitzResult(
        reduced=reduced,
        trial_basis=basis,
        operator_basis=operator_basis,
        metric_basis=metric_basis,
        eigenvectors=eigenvectors,
        residuals=residuals,
        residual_norms=residual_norms,
        relative_residuals=relative,
        metric_orthogonality_error=orthogonality,
    )


def with_initial_basis(
    policy: EigenSolvePolicy,
    initial_basis: ArrayLike,
    /,
) -> EigenSolvePolicy:
    """Return one otherwise-identical eigensolve policy with a new trial basis."""
    if not isinstance(policy, EigenSolvePolicy):
        raise TypeError("policy must be an EigenSolvePolicy.")
    return EigenSolvePolicy(
        policy.method,
        count=policy.count,
        which=policy.which,
        max_steps=policy.max_steps,
        tolerance=policy.tolerance,
        resources=policy.resources,
        materialization=policy.materialization,
        initial_basis=initial_basis,
        key=policy.key,
        preconditioning=policy.preconditioning,
        differentiation=policy.differentiation,
        failure=policy.failure,
    )


def warm_started_eigensolve(
    problem: EigenproblemLike,
    trial_basis: ArrayLike,
    /,
    *,
    policy: EigenSolvePolicy | None = None,
    tolerance: float = 1e-10,
) -> WarmStartedEigenResult:
    """Certify a learned trial space and use its Ritz vectors to start refinement."""
    basis = jnp.asarray(trial_basis)
    selected = (
        EigenSolvePolicy(count=int(basis.shape[1])) if policy is None else policy
    )
    if not isinstance(selected, EigenSolvePolicy):
        raise TypeError("policy must be an EigenSolvePolicy or None.")
    trial = rayleigh_ritz(
        problem,
        basis,
        count=selected.count,
        which=selected.which,
        tolerance=tolerance,
    )
    guarded_basis = eqx.error_if(
        trial.eigenvectors,
        ~trial.valid,
        "Warm-start trial space failed Ritz validation.",
    )
    solve = eigensolve(
        problem,
        policy=with_initial_basis(selected, guarded_basis),
    )
    return WarmStartedEigenResult(trial=trial, solve=solve)


__all__ = [
    "BlockRayleighEvaluation",
    "ReducedRitzResult",
    "TrialSubspaceRitzResult",
    "WarmStartedEigenResult",
    "block_rayleigh_trace",
    "rayleigh_ritz",
    "solve_reduced_ritz",
    "warm_started_eigensolve",
    "with_initial_basis",
]
