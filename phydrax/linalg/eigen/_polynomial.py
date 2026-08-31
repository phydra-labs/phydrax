#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._operators import (
    AbstractLinearOperator,
    BlockLinearOperator,
    IdentityLinearOperator,
)
from .._spaces import _coordinate_dtype, AbstractVectorSpace, BlockSpace
from ._general import (
    _operator_coordinate_action,
    _unflatten_complex_columns,
    general_eigensolve,
    GeneralEigenproblem,
    GeneralEigenSolvePlan,
    GeneralEigenSolvePolicy,
    GeneralEigenSolveResult,
    plan_general_eigensolve,
    prepare_general_eigensolve,
    PreparedGeneralEigenSolve,
    refresh_general_eigensolve,
)


class PolynomialEigenSolveStatus(IntEnum):
    SUCCESS = 0
    LINEARIZED_FAILURE = 1
    ORIGINAL_RESIDUAL_TOLERANCE_NOT_MET = 2
    NONFINITE_OUTPUT = 3
    PHYSICAL_VECTOR_EXTRACTION_FAILURE = 4


class PolynomialEigenproblem(StrictModule):
    """Operator polynomial ``A₀ + λ A₁ + ... + λᵈ A_d``."""

    coefficients: tuple[AbstractLinearOperator, ...]
    degree: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        coefficients: Sequence[AbstractLinearOperator],
        /,
        *,
        problem_id: str | None = None,
    ):
        values = tuple(coefficients)
        if len(values) < 2 or not all(
            isinstance(value, AbstractLinearOperator) for value in values
        ):
            raise TypeError("coefficients must contain at least two linear operators.")
        source = values[0].source
        target = values[0].target
        if values[0].batch_shape or not source.compatible(target):
            raise ValueError("Polynomial eigenproblems require unbatched endomorphisms.")
        for operator in values[1:]:
            if (
                operator.batch_shape
                or not operator.source.compatible(source)
                or not operator.target.compatible(target)
            ):
                raise ValueError(
                    "Polynomial coefficients must share one endomorphism space."
                )
        if values[-1].properties.rank == 0:
            raise ValueError("The declared leading polynomial coefficient is zero.")
        degree = len(values) - 1
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "polynomial-eigenproblem",
                    "coefficients": [operator.operator_id for operator in values],
                    "degree": degree,
                    "source": source.space_id,
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be nonempty.")
        self.coefficients = values
        self.degree = degree
        self.problem_id = identifier

    @property
    def dimension(self) -> int:
        return self.coefficients[0].source.size


class PolynomialEigenSolvePolicy(StrictModule, NonTrainableState):
    general: GeneralEigenSolvePolicy
    eigenvalue_scale: float = eqx.field(static=True)
    relative_residual_tolerance: float = eqx.field(static=True)
    absolute_residual_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        general: GeneralEigenSolvePolicy | None = None,
        eigenvalue_scale: float = 1.0,
        relative_residual_tolerance: float = 1e-8,
        absolute_residual_tolerance: float = 1e-10,
    ):
        general_ = GeneralEigenSolvePolicy() if general is None else general
        scale = float(eigenvalue_scale)
        relative = float(relative_residual_tolerance)
        absolute = float(absolute_residual_tolerance)
        if not isinstance(general_, GeneralEigenSolvePolicy):
            raise TypeError("general must be a GeneralEigenSolvePolicy or None.")
        if (
            not math.isfinite(scale)
            or scale <= 0.0
            or not math.isfinite(relative)
            or relative < 0.0
            or not math.isfinite(absolute)
            or absolute < 0.0
        ):
            raise ValueError("Polynomial eigen policy values are invalid.")
        self.general = general_
        self.eigenvalue_scale = scale
        self.relative_residual_tolerance = relative
        self.absolute_residual_tolerance = absolute
        self.policy_id = canonical_fingerprint(
            {
                "kind": "polynomial-eigen-solve-policy",
                "general_method": general_.method.name,
                "general_transform": general_.transform.name,
                "general_selection": general_.selection.selection_id,
                "eigenvalue_scale": scale,
                "relative_residual_tolerance": relative,
                "absolute_residual_tolerance": absolute,
            }
        )


class PolynomialEigenSolvePlan(StrictModule, NonTrainableState):
    policy: PolynomialEigenSolvePolicy
    general_plan: GeneralEigenSolvePlan
    degree: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    coefficient_operator_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedPolynomialEigenSolve(StrictModule):
    problem: PolynomialEigenproblem
    linearized_problem: GeneralEigenproblem
    linearized: PreparedGeneralEigenSolve
    plan: PolynomialEigenSolvePlan = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: Array


class PolynomialEigenSolveDiagnostics(StrictModule):
    original_residual_norms: Array
    original_relative_residuals: Array
    original_residual_scales: Array
    finite_mask: Array
    infinite_mask: Array
    indeterminate_mask: Array
    converged_mask: Array
    linearized_status: Array
    right_extraction_blocks: Array
    right_extraction_norms: Array
    eigenvalue_scale: Array


class PolynomialEigenSolveProvenance(StrictModule):
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    coefficient_operator_ids: tuple[str, ...] = eqx.field(static=True)
    linearization: str = eqx.field(static=True)
    numeric_version: Array


class PolynomialEigenSolveResult(StrictModule):
    eigenvalues: Array
    alpha: Array
    beta: Array
    right_eigenvectors: PyTree[Array]
    left_eigenvectors: PyTree[Array]
    status: Array
    diagnostics: PolynomialEigenSolveDiagnostics
    provenance: PolynomialEigenSolveProvenance
    linearized_result: GeneralEigenSolveResult

    @property
    def successful(self) -> Array:
        return self.status == int(PolynomialEigenSolveStatus.SUCCESS)


def plan_polynomial_eigensolve(
    problem: PolynomialEigenproblem,
    policy: PolynomialEigenSolvePolicy | None = None,
    /,
) -> PolynomialEigenSolvePlan:
    if not isinstance(problem, PolynomialEigenproblem):
        raise TypeError("problem must be a PolynomialEigenproblem.")
    policy_ = PolynomialEigenSolvePolicy() if policy is None else policy
    if not isinstance(policy_, PolynomialEigenSolvePolicy):
        raise TypeError("policy must be a PolynomialEigenSolvePolicy or None.")
    linearized = _linearized_problem(problem, policy_)
    general_plan = plan_general_eigensolve(linearized, policy_.general)
    return PolynomialEigenSolvePlan(
        policy=policy_,
        general_plan=general_plan,
        degree=problem.degree,
        dimension=problem.dimension,
        problem_id=problem.problem_id,
        source_space_id=problem.coefficients[0].source.space_id,
        coefficient_operator_ids=tuple(
            operator.operator_id for operator in problem.coefficients
        ),
        plan_id=canonical_fingerprint(
            {
                "kind": "polynomial-eigen-solve-plan",
                "problem": problem.problem_id,
                "policy": policy_.policy_id,
                "general_plan": general_plan.plan_id,
                "degree": problem.degree,
                "dimension": problem.dimension,
            }
        ),
    )


def prepare_polynomial_eigensolve(
    problem: PolynomialEigenproblem,
    policy: PolynomialEigenSolvePolicy | PolynomialEigenSolvePlan | None = None,
    /,
) -> PreparedPolynomialEigenSolve:
    plan = (
        policy
        if isinstance(policy, PolynomialEigenSolvePlan)
        else plan_polynomial_eigensolve(problem, policy)
    )
    return _prepare_polynomial(problem, plan, numeric_version=0)


def refresh_polynomial_eigensolve(
    prepared: PreparedPolynomialEigenSolve,
    problem: PolynomialEigenproblem,
    /,
) -> PreparedPolynomialEigenSolve:
    if not isinstance(prepared, PreparedPolynomialEigenSolve):
        raise TypeError("prepared must be a PreparedPolynomialEigenSolve.")
    _validate_polynomial_plan(problem, prepared.plan)
    linearized_problem = _linearized_problem(problem, prepared.plan.policy)
    linearized = refresh_general_eigensolve(prepared.linearized, linearized_problem)
    return PreparedPolynomialEigenSolve(
        problem=problem,
        linearized_problem=linearized_problem,
        linearized=linearized,
        plan=prepared.plan,
        prepared_id=prepared.prepared_id,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
    )


def polynomial_eigensolve(
    problem_or_prepared: PolynomialEigenproblem | PreparedPolynomialEigenSolve,
    /,
    *,
    policy: PolynomialEigenSolvePolicy | PolynomialEigenSolvePlan | None = None,
) -> PolynomialEigenSolveResult:
    if isinstance(problem_or_prepared, PreparedPolynomialEigenSolve):
        if policy is not None:
            raise ValueError("policy must be omitted for a prepared polynomial solve.")
        prepared = problem_or_prepared
    elif isinstance(problem_or_prepared, PolynomialEigenproblem):
        prepared = prepare_polynomial_eigensolve(problem_or_prepared, policy)
    else:
        raise TypeError("Expected a polynomial eigenproblem or prepared solve.")
    general = general_eigensolve(prepared.linearized)
    scale = jnp.asarray(
        prepared.plan.policy.eigenvalue_scale,
        dtype=general.alpha.real.dtype,
    )
    alpha = scale.astype(general.alpha.dtype) * general.alpha
    beta = general.beta
    finite = general.finite_mask
    eigenvalues = jnp.where(finite, alpha / beta, jnp.inf + 0j)
    right_coordinates, extraction_blocks, extraction_norms = (
        _extract_physical_right_coordinates(prepared.problem, general)
    )
    source = prepared.problem.coefficients[0].source
    right = _unflatten_complex_columns(source, right_coordinates)
    left_coordinates = general.left_eigenvector_coordinates[-prepared.problem.dimension :]
    left = _unflatten_complex_columns(source, left_coordinates)
    residuals, relative, residual_scales = _original_residuals(
        prepared.problem,
        alpha,
        beta,
        right_coordinates,
    )
    finite_output = (
        jnp.all(jnp.isfinite(alpha))
        & jnp.all(jnp.isfinite(beta))
        & jnp.all(jnp.isfinite(residuals))
        & jnp.all(jnp.isfinite(relative))
        & jnp.all(jnp.isfinite(residual_scales))
        & jnp.all(jnp.isfinite(extraction_norms))
    )
    physical_vectors_valid = jnp.all(extraction_norms > 0.0)
    mode_residual_ok = residuals <= (
        prepared.plan.policy.absolute_residual_tolerance
        + prepared.plan.policy.relative_residual_tolerance * residual_scales
    )
    residual_ok = jnp.all(general.indeterminate_mask | mode_residual_ok)
    status = jnp.where(
        ~general.successful,
        int(PolynomialEigenSolveStatus.LINEARIZED_FAILURE),
        jnp.where(
            ~finite_output,
            int(PolynomialEigenSolveStatus.NONFINITE_OUTPUT),
            jnp.where(
                ~physical_vectors_valid,
                int(PolynomialEigenSolveStatus.PHYSICAL_VECTOR_EXTRACTION_FAILURE),
                jnp.where(
                    ~residual_ok,
                    int(PolynomialEigenSolveStatus.ORIGINAL_RESIDUAL_TOLERANCE_NOT_MET),
                    int(PolynomialEigenSolveStatus.SUCCESS),
                ),
            ),
        ),
    ).astype(jnp.int32)
    converged = (
        general.diagnostics.converged_mask & mode_residual_ok & (extraction_norms > 0.0)
    )
    return PolynomialEigenSolveResult(
        eigenvalues=eigenvalues,
        alpha=alpha,
        beta=beta,
        right_eigenvectors=right,
        left_eigenvectors=left,
        status=status,
        diagnostics=PolynomialEigenSolveDiagnostics(
            original_residual_norms=residuals,
            original_relative_residuals=relative,
            original_residual_scales=residual_scales,
            finite_mask=finite,
            infinite_mask=general.infinite_mask,
            indeterminate_mask=general.indeterminate_mask,
            converged_mask=converged,
            linearized_status=general.status,
            right_extraction_blocks=extraction_blocks,
            right_extraction_norms=extraction_norms,
            eigenvalue_scale=scale,
        ),
        provenance=PolynomialEigenSolveProvenance(
            problem_id=prepared.problem.problem_id,
            plan_id=prepared.plan.plan_id,
            prepared_id=prepared.prepared_id,
            source_space_id=prepared.problem.coefficients[0].source.space_id,
            coefficient_operator_ids=tuple(
                operator.operator_id for operator in prepared.problem.coefficients
            ),
            linearization=(
                "homogeneous first Frobenius companion; physical right vector "
                "selected from the largest companion block"
            ),
            numeric_version=prepared.numeric_version,
        ),
        linearized_result=general,
    )


def _prepare_polynomial(
    problem: PolynomialEigenproblem,
    plan: PolynomialEigenSolvePlan,
    /,
    *,
    numeric_version: int,
) -> PreparedPolynomialEigenSolve:
    _validate_polynomial_plan(problem, plan)
    linearized_problem = _linearized_problem(problem, plan.policy)
    linearized = prepare_general_eigensolve(linearized_problem, plan.general_plan)
    return PreparedPolynomialEigenSolve(
        problem=problem,
        linearized_problem=linearized_problem,
        linearized=linearized,
        plan=plan,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-polynomial-eigen-solve",
                "plan": plan.plan_id,
                "coefficients": [
                    operator.operator_id for operator in problem.coefficients
                ],
                "numeric_version": numeric_version,
            }
        ),
        numeric_version=jnp.asarray(numeric_version, dtype=jnp.int32),
    )


def _validate_polynomial_plan(
    problem: PolynomialEigenproblem,
    plan: PolynomialEigenSolvePlan,
    /,
) -> None:
    if not isinstance(problem, PolynomialEigenproblem) or not isinstance(
        plan, PolynomialEigenSolvePlan
    ):
        raise TypeError("Polynomial preparation requires a problem and plan.")
    coefficient_ids = tuple(operator.operator_id for operator in problem.coefficients)
    if (
        problem.problem_id != plan.problem_id
        or problem.degree != plan.degree
        or problem.dimension != plan.dimension
        or problem.coefficients[0].source.space_id != plan.source_space_id
        or coefficient_ids != plan.coefficient_operator_ids
    ):
        raise ValueError("Polynomial eigenproblem is incompatible with its plan.")


def _linearized_problem(
    problem: PolynomialEigenproblem,
    policy: PolynomialEigenSolvePolicy,
    /,
) -> GeneralEigenproblem:
    degree = problem.degree
    source_space = problem.coefficients[0].source
    block_space = BlockSpace(
        (source_space,) * degree,
        names=tuple(f"power{power}" for power in range(degree)),
    )
    identity = IdentityLinearOperator(source_space)
    matrix_blocks: list[list[AbstractLinearOperator | None]] = [
        [None for _ in range(degree)] for _ in range(degree)
    ]
    mass_blocks: list[list[AbstractLinearOperator | None]] = [
        [None for _ in range(degree)] for _ in range(degree)
    ]
    for row in range(degree - 1):
        matrix_blocks[row][row + 1] = identity
        mass_blocks[row][row] = identity
    scale = policy.eigenvalue_scale
    for column in range(degree):
        matrix_blocks[-1][column] = -(scale**column) * problem.coefficients[column]
    mass_blocks[-1][-1] = (scale**degree) * problem.coefficients[-1]
    matrix = BlockLinearOperator(
        matrix_blocks,
        source=block_space,
        target=block_space,
        operator_id=canonical_fingerprint(
            {
                "kind": "polynomial-first-companion-matrix",
                "problem": problem.problem_id,
                "scale": scale,
            }
        ),
    )
    mass = BlockLinearOperator(
        mass_blocks,
        source=block_space,
        target=block_space,
        operator_id=canonical_fingerprint(
            {
                "kind": "polynomial-first-companion-mass",
                "problem": problem.problem_id,
                "scale": scale,
            }
        ),
    )
    return GeneralEigenproblem(
        matrix,
        mass,
        problem_id=canonical_fingerprint(
            {
                "kind": "linearized-polynomial-eigenproblem",
                "problem": problem.problem_id,
                "scale": scale,
            }
        ),
    )


def _extract_physical_right_coordinates(
    problem: PolynomialEigenproblem,
    general: GeneralEigenSolveResult,
    /,
) -> tuple[Array, Array, Array]:
    coordinates = jnp.asarray(general.right_eigenvector_coordinates).reshape(
        (problem.degree, problem.dimension, -1)
    )
    vectors = []
    blocks = []
    norms = []
    for index in range(int(general.alpha.size)):
        candidates = coordinates[:, :, index]
        candidate_norms = jnp.stack(
            tuple(
                _coordinate_norm(problem.coefficients[0].source, candidates[block])
                for block in range(problem.degree)
            )
        )
        block = jnp.argmax(candidate_norms)
        vector = jnp.take(candidates, block, axis=0)
        norm = jnp.take(candidate_norms, block)
        vectors.append(vector / jnp.where(norm > 0.0, norm, 1.0))
        blocks.append(block)
        norms.append(norm)
    return jnp.stack(vectors, axis=1), jnp.stack(blocks), jnp.stack(norms)


def _original_residuals(
    problem: PolynomialEigenproblem,
    alpha: Array,
    beta: Array,
    right_coordinates: Array,
    /,
) -> tuple[Array, Array, Array]:
    count = int(alpha.size)
    residuals = []
    relatives = []
    scales = []
    for index in range(count):
        vector = right_coordinates[:, index]
        residual = jnp.zeros(
            (problem.coefficients[0].target.size,),
            dtype=jnp.result_type(vector.dtype, alpha.dtype),
        )
        residual_scale = jnp.asarray(0.0, dtype=alpha.real.dtype)
        for power, operator in enumerate(problem.coefficients):
            action = _operator_coordinate_action(
                operator,
                vector,
                adjoint_action=False,
            )
            coefficient = alpha[index] ** power * beta[index] ** (problem.degree - power)
            residual = residual + coefficient * action
            residual_scale = residual_scale + jnp.abs(coefficient) * _coordinate_norm(
                operator.target,
                action,
            )
        residual_norm = _coordinate_norm(
            problem.coefficients[0].target,
            residual,
        )
        residuals.append(residual_norm)
        scales.append(residual_scale)
        relatives.append(
            residual_norm
            / jnp.maximum(residual_scale, jnp.finfo(residual_scale.dtype).tiny)
        )
    return jnp.stack(residuals), jnp.stack(relatives), jnp.stack(scales)


def _coordinate_norm(space: AbstractVectorSpace, coordinates: Array, /) -> Array:
    dtype = np.dtype(_coordinate_dtype(space))
    if np.issubdtype(dtype, np.complexfloating):
        vector = space.unflatten(coordinates.astype(dtype))
        return jnp.sqrt(jnp.maximum(jnp.real(space.pairing.inner(vector, vector)), 0.0))
    real = space.unflatten(jnp.real(coordinates).astype(dtype))
    imaginary = space.unflatten(jnp.imag(coordinates).astype(dtype))
    square = jnp.real(space.pairing.inner(real, real)) + jnp.real(
        space.pairing.inner(imaginary, imaginary)
    )
    return jnp.sqrt(jnp.maximum(square, 0.0))


__all__ = [
    "PolynomialEigenSolveDiagnostics",
    "PolynomialEigenSolvePlan",
    "PolynomialEigenSolvePolicy",
    "PolynomialEigenSolveProvenance",
    "PolynomialEigenSolveResult",
    "PolynomialEigenSolveStatus",
    "PolynomialEigenproblem",
    "PreparedPolynomialEigenSolve",
    "plan_polynomial_eigensolve",
    "polynomial_eigensolve",
    "prepare_polynomial_eigensolve",
    "refresh_polynomial_eigensolve",
]
