#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import combinations, combinations_with_replacement
from math import comb
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    LinearSystem,
    RankPolicy,
    solve,
)
from ..linalg.eigen import (
    DenseSchurQZ,
    general_eigensolve,
    GeneralEigenproblem,
    GeneralEigenSolvePolicy,
    GeneralEigenTolerancePolicy,
)
from ..optim import least_squares
from ._contracts import (
    SymmetricWaringCostEstimate,
    SymmetricWaringEvidence,
    SymmetricWaringPlan,
    SymmetricWaringProblem,
    SymmetricWaringRankPolicy,
    SymmetricWaringRefinement,
    SymmetricWaringResourcePolicy,
    SymmetricWaringResult,
    SymmetricWaringStatus,
)


class _ChartCandidate(NamedTuple):
    weights: Array
    factors: Array
    reconstruction: Array
    residual: Array
    relative_residual: float
    chart_axis: int
    observed_rank: int
    basis_indices: tuple[int, ...]
    hankel_condition: float
    commutator_defect: float
    joint_diagonalization_defect: float
    joint_spectrum_separation: float


def prepare_symmetric_waring(
    problem: SymmetricWaringProblem,
    /,
    *,
    rank_policy: SymmetricWaringRankPolicy | None = None,
    resources: SymmetricWaringResourcePolicy | None = None,
    refinement: SymmetricWaringRefinement | None = None,
    chart_axis: int | None = None,
) -> SymmetricWaringPlan:
    """Plan finite affine Hankel work without claiming global identifiability."""
    if not isinstance(problem, SymmetricWaringProblem):
        raise TypeError("problem must be a SymmetricWaringProblem.")
    rank_policy_ = SymmetricWaringRankPolicy() if rank_policy is None else rank_policy
    resources_ = SymmetricWaringResourcePolicy() if resources is None else resources
    refinement_ = SymmetricWaringRefinement() if refinement is None else refinement
    if not isinstance(rank_policy_, SymmetricWaringRankPolicy):
        raise TypeError("rank_policy must be SymmetricWaringRankPolicy or None.")
    if not isinstance(resources_, SymmetricWaringResourcePolicy):
        raise TypeError("resources must be SymmetricWaringResourcePolicy or None.")
    if not isinstance(refinement_, SymmetricWaringRefinement):
        raise TypeError("refinement must be SymmetricWaringRefinement or None.")

    if chart_axis is None:
        chart_axes = tuple(range(problem.dimension))
    else:
        axis = int(chart_axis)
        if axis < 0 or axis >= problem.dimension:
            raise ValueError("chart_axis is outside the tensor dimension.")
        chart_axes = (axis,)

    affine_dimension = problem.dimension - 1
    quotient_degree = _quotient_degree(
        affine_dimension,
        problem.order,
        problem.rank,
    )
    algebraically_admissible = quotient_degree >= 0
    quotient_exponents = (
        _monomial_exponents(affine_dimension, quotient_degree)
        if algebraically_admissible
        else ()
    )
    basis_size = len(quotient_exponents)
    basis_subsets = comb(basis_size, problem.rank) if basis_size >= problem.rank else 0
    hankel_entries = (
        basis_size * basis_size + affine_dimension * problem.rank * problem.rank
        if algebraically_admissible
        else 0
    )
    vandermonde_entries = problem.independent_entry_count * problem.rank
    complex_multiplier = (
        2 if jnp.issubdtype(problem.tensor.dtype, jnp.complexfloating) else 1
    )
    refinement_parameters = (
        complex_multiplier * problem.rank * (problem.dimension + 1)
        if refinement_.enabled
        else 0
    )
    cost = SymmetricWaringCostEstimate(
        tensor_entries=int(problem.tensor.size),
        quotient_basis_size=basis_size,
        hankel_entries=hankel_entries,
        vandermonde_entries=vandermonde_entries,
        basis_subsets=basis_subsets,
        refinement_parameters=refinement_parameters,
    )
    rejection = _resource_rejection(problem, cost, resources_)
    return SymmetricWaringPlan(
        problem,
        rank_policy_,
        resources_,
        refinement_,
        chart_axes,
        quotient_degree,
        quotient_exponents,
        algebraically_admissible,
        rejection is None,
        rejection,
        cost,
    )


def solve_symmetric_waring(
    problem_or_plan: SymmetricWaringProblem | SymmetricWaringPlan,
    /,
    *,
    rank_policy: SymmetricWaringRankPolicy | None = None,
    resources: SymmetricWaringResourcePolicy | None = None,
    refinement: SymmetricWaringRefinement | None = None,
    chart_axis: int | None = None,
) -> SymmetricWaringResult:
    """Recover an isolated finite Waring decomposition when the Hankel path admits it.

    Preparation and numerical recovery are deliberately separate from physical
    acceptance. A successful status means the selected affine chart had the
    requested finite Hankel rank and isolated joint spectrum, and that the final
    dense reconstruction met the declared relative tolerance. It is not a claim
    that every tensor of the same shape or rank is identifiable.
    """
    if isinstance(problem_or_plan, SymmetricWaringPlan):
        if any(
            value is not None
            for value in (rank_policy, resources, refinement, chart_axis)
        ):
            raise ValueError(
                "Policy and chart overrides must be omitted when solving a prepared plan."
            )
        plan = problem_or_plan
    elif isinstance(problem_or_plan, SymmetricWaringProblem):
        plan = prepare_symmetric_waring(
            problem_or_plan,
            rank_policy=rank_policy,
            resources=resources,
            refinement=refinement,
            chart_axis=chart_axis,
        )
    else:
        raise TypeError("Expected SymmetricWaringProblem or SymmetricWaringPlan.")

    problem = plan.problem
    if not plan.resource_admitted:
        evidence = _failure_evidence(
            plan,
            detail=f"resource policy rejected plan: {plan.resource_rejection}",
        )
        return _failure_result(plan, SymmetricWaringStatus.RESOURCE_REJECTED, evidence)
    if problem.rank * problem.dimension > problem.independent_entry_count:
        evidence = _failure_evidence(
            plan,
            detail=(
                "the requested component parameter count exceeds the symmetric "
                "tensor coordinate count, so the generic fiber is positive-dimensional"
            ),
        )
        return _failure_result(
            plan,
            SymmetricWaringStatus.NONGENERIC_OR_MULTIPLE_DECOMPOSITION,
            evidence,
        )
    if not plan.algebraically_admissible:
        evidence = _failure_evidence(
            plan,
            detail=(
                "tensor order does not provide the 2s+1 moments required by a "
                "rank-sized affine quotient basis"
            ),
        )
        return _failure_result(
            plan,
            SymmetricWaringStatus.ALGEBRAIC_INITIALIZATION_NOT_ADMISSIBLE,
            evidence,
        )

    candidates: list[_ChartCandidate] = []
    observed_ranks: list[int] = []
    exact_rank_chart_seen = False
    algebraic_failure_seen = False
    for axis in plan.chart_axes:
        candidate, observed_rank, failure_kind = _recover_chart(plan, axis)
        observed_ranks.append(observed_rank)
        exact_rank_chart_seen = exact_rank_chart_seen or observed_rank == problem.rank
        algebraic_failure_seen = algebraic_failure_seen or failure_kind == "provider"
        if candidate is not None:
            candidates.append(candidate)

    observed = max(observed_ranks, default=0)
    if not candidates:
        if observed < problem.rank:
            status = SymmetricWaringStatus.OVER_REQUESTED_RANK_OR_REPEATED_FACTOR
            detail = (
                "every admitted affine Hankel matrix has rank below the requested "
                "rank; the request is over-ranked or its factors coalesce"
            )
        elif observed > problem.rank:
            status = SymmetricWaringStatus.REQUESTED_RANK_MISMATCH
            detail = "an admitted affine Hankel matrix exceeds the requested rank"
        elif algebraic_failure_seen and exact_rank_chart_seen:
            status = SymmetricWaringStatus.ALGEBRAIC_INITIALIZATION_FAILED
            detail = "a Phydrax linear or general-eigen provider rejected the path"
        else:
            status = SymmetricWaringStatus.NONGENERIC_OR_MULTIPLE_DECOMPOSITION
            detail = (
                "rank-sized Hankel data did not yield a conditioned commuting "
                "multiplication family with an isolated joint spectrum"
            )
        evidence = _failure_evidence(
            plan,
            observed_hankel_rank=observed,
            detail=detail,
            provider=(
                "phydrax.linalg(DenseLU,DenseSVD)/phydrax.linalg.eigen(DenseSchurQZ)"
                if exact_rank_chart_seen
                else "host-catalecticant-rank"
            ),
        )
        return _failure_result(plan, status, evidence)

    algebraic = min(candidates, key=lambda candidate: candidate.relative_residual)
    final = algebraic
    refinement_attempted = plan.refinement.enabled
    refinement_accepted = False
    refinement_status = -1
    if refinement_attempted:
        final, refinement_accepted, refinement_status = _refine_candidate(
            plan,
            algebraic,
        )

    physical_accepted = bool(
        np.isfinite(final.relative_residual)
        and final.relative_residual <= plan.rank_policy.relative_reconstruction_tolerance
    )
    status = (
        SymmetricWaringStatus.SUCCESS
        if physical_accepted
        else SymmetricWaringStatus.RECONSTRUCTION_TOLERANCE_NOT_MET
    )
    residual_norm = jnp.linalg.norm(final.residual.reshape((-1,)))
    evidence = SymmetricWaringEvidence(
        requested_rank=problem.rank,
        observed_hankel_rank=final.observed_rank,
        chart_axis=final.chart_axis,
        quotient_degree=plan.quotient_degree,
        basis_indices=final.basis_indices,
        hankel_condition=final.hankel_condition,
        commutator_defect=final.commutator_defect,
        joint_diagonalization_defect=final.joint_diagonalization_defect,
        joint_spectrum_separation=final.joint_spectrum_separation,
        algebraic_relative_residual=algebraic.relative_residual,
        final_relative_residual=final.relative_residual,
        path_accepted=True,
        physical_accepted=physical_accepted,
        refinement_attempted=refinement_attempted,
        refinement_accepted=refinement_accepted,
        refinement_status=refinement_status,
        provider=(
            "phydrax.linalg(DenseLU,DenseSVD)/"
            "phydrax.linalg.eigen(DenseSchurQZ)"
            + (
                f";phydrax.optim({plan.refinement.method.method_id})"
                if refinement_attempted
                else ""
            )
        ),
        detail=(
            "isolated joint multiplication spectrum and accepted reconstruction"
            if physical_accepted
            else "algebraic path was isolated but reconstruction tolerance was not met"
        ),
    )
    return SymmetricWaringResult(
        weights=final.weights,
        factors=final.factors,
        reconstruction=final.reconstruction,
        residual=final.residual,
        residual_norm=residual_norm,
        relative_residual=final.relative_residual,
        status=status,
        evidence=evidence,
        plan_id=plan.plan_id,
    )


def reconstruct_symmetric_tensor(
    weights: ArrayLike,
    factors: ArrayLike,
    order: int,
    /,
) -> Array:
    """Materialize ``sum_s weight_s factor_s**order`` through ``phydrax.ein``."""
    weights_ = jnp.asarray(weights)
    factors_ = jnp.asarray(factors)
    order_ = int(order)
    if order_ < 1:
        raise ValueError("order must be positive.")
    if weights_.ndim != 1 or factors_.ndim != 2:
        raise ValueError("weights and factors must have shapes (rank,) and (rank,n).")
    if factors_.shape[0] != weights_.shape[0]:
        raise ValueError("weights and factors must have one shared component axis.")
    dtype = jnp.result_type(weights_, factors_)
    weights_ = weights_.astype(dtype)
    factors_ = factors_.astype(dtype)
    arguments: list[Any] = [weights_, [0]]
    for axis in range(order_):
        arguments.extend((factors_, [0, axis + 1]))
    arguments.append(list(range(1, order_ + 1)))
    return jnp.asarray(contract(*arguments))


def normalize_waring_components(
    weights: ArrayLike,
    factors: ArrayLike,
    order: int,
    /,
) -> tuple[Array, Array]:
    """Choose deterministic representatives of Waring scale/phase/permutation orbits."""
    weights_ = np.asarray(weights)
    factors_ = np.asarray(factors)
    order_ = int(order)
    if order_ < 1:
        raise ValueError("order must be positive.")
    if weights_.ndim != 1 or factors_.ndim != 2:
        raise ValueError("weights and factors must have shapes (rank,) and (rank,n).")
    if factors_.shape[0] != weights_.shape[0]:
        raise ValueError("weights and factors must contain the same rank.")
    if not np.all(np.isfinite(weights_)) or not np.all(np.isfinite(factors_)):
        raise ValueError("Waring components must be finite.")
    dtype = np.result_type(weights_.dtype, factors_.dtype, np.float32)
    normalized = factors_.astype(dtype, copy=True)
    normalized_weights = weights_.astype(dtype, copy=True)
    norms = np.linalg.norm(normalized, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("Waring factors must be nonzero.")
    normalized /= norms[:, None]
    normalized_weights *= norms**order_
    pivots = np.argmax(np.abs(normalized), axis=1)
    pivot_values = normalized[np.arange(normalized.shape[0]), pivots]
    phases = pivot_values / np.abs(pivot_values)
    normalized /= phases[:, None]
    normalized_weights *= phases**order_
    normalized[np.arange(normalized.shape[0]), pivots] = np.abs(
        normalized[np.arange(normalized.shape[0]), pivots]
    )
    key = np.concatenate(
        (
            np.real(normalized),
            np.imag(normalized),
            np.real(normalized_weights)[:, None],
            np.imag(normalized_weights)[:, None],
        ),
        axis=1,
    )
    ordering = np.lexsort(
        tuple(key[:, index] for index in range(key.shape[1] - 1, -1, -1))
    )
    normalized = normalized[ordering]
    normalized_weights = normalized_weights[ordering]
    if not (
        np.issubdtype(weights_.dtype, np.complexfloating)
        or np.issubdtype(factors_.dtype, np.complexfloating)
    ):
        normalized = np.real(normalized)
        normalized_weights = np.real(normalized_weights)
    return jnp.asarray(normalized_weights), jnp.asarray(normalized)


def _recover_chart(
    plan: SymmetricWaringPlan,
    chart_axis: int,
    /,
) -> tuple[_ChartCandidate | None, int, str | None]:
    problem = plan.problem
    policy = plan.rank_policy
    rank = problem.rank
    exponents = plan.quotient_exponents
    tensor = problem.tensor
    hankel = _hankel_matrix(tensor, chart_axis, exponents, exponents)
    hankel_host = np.asarray(hankel)
    singular_values = np.linalg.svd(hankel_host, compute_uv=False)
    observed_rank = _numerical_rank(singular_values, policy.relative_rank_tolerance)
    if observed_rank != rank:
        return None, observed_rank, "rank"

    basis_indices, condition = _select_quotient_basis(
        hankel_host,
        rank,
        policy.relative_rank_tolerance,
    )
    if basis_indices is None or condition > policy.maximum_hankel_condition:
        return None, observed_rank, "nongeneric"
    basis = tuple(exponents[index] for index in basis_indices)
    base = hankel[jnp.ix_(jnp.asarray(basis_indices), jnp.asarray(basis_indices))]

    affine_axes = tuple(axis for axis in range(problem.dimension) if axis != chart_axis)
    multiplication: list[Array] = []
    linear_policy = LinearSolvePolicy(DenseLU())
    for affine_index in range(problem.dimension - 1):
        shifted = tuple(
            tuple(
                exponent_index + (1 if index == affine_index else 0)
                for index, exponent_index in enumerate(exponent)
            )
            for exponent in basis
        )
        shifted_hankel = _hankel_matrix(tensor, chart_axis, basis, shifted)
        solved = solve(
            LinearSystem(DenseLinearOperator(base)),
            shifted_hankel,
            policy=linear_policy,
        )
        if not bool(np.all(np.asarray(solved.successful))):
            return None, observed_rank, "provider"
        multiplication.append(jnp.asarray(solved.value))

    commutator_defect = _commutator_defect(multiplication)
    if commutator_defect > policy.relative_commutator_tolerance:
        return None, observed_rank, "nongeneric"
    joint = sum(
        np.sqrt(float(index + 2)) * matrix for index, matrix in enumerate(multiplication)
    )
    eigensolve = general_eigensolve(
        GeneralEigenproblem(DenseLinearOperator(joint)),
        policy=GeneralEigenSolvePolicy(
            DenseSchurQZ(),
            tolerance=GeneralEigenTolerancePolicy(
                relative=max(policy.relative_rank_tolerance, 1.0e-6),
                absolute=1.0e-8,
                biorthogonality=max(
                    policy.relative_joint_diagonalization_tolerance,
                    1.0e-6,
                ),
                cluster_relative=policy.relative_joint_spectrum_separation,
            ),
        ),
    )
    if not bool(np.all(np.asarray(eigensolve.successful))):
        return None, observed_rank, "provider"
    eigenvalues = np.asarray(eigensolve.eigenvalues)
    eigenvectors = jnp.asarray(eigensolve.right_eigenvector_coordinates)
    separation = _relative_separation(eigenvalues)
    if separation < policy.relative_joint_spectrum_separation:
        return None, observed_rank, "nongeneric"
    eigenvector_condition = float(np.linalg.cond(np.asarray(eigenvectors)))
    if not np.isfinite(eigenvector_condition) or (
        eigenvector_condition > policy.maximum_hankel_condition
    ):
        return None, observed_rank, "nongeneric"

    coordinates: list[Array] = []
    diagonalization_defect = 0.0
    eigenvector_policy = LinearSolvePolicy(DenseLU())
    for matrix in multiplication:
        transformed = solve(
            LinearSystem(DenseLinearOperator(eigenvectors)),
            matrix @ eigenvectors,
            policy=eigenvector_policy,
        )
        if not bool(np.all(np.asarray(transformed.successful))):
            return None, observed_rank, "provider"
        diagonalized = jnp.asarray(transformed.value)
        diagonal = jnp.diag(diagonalized)
        off_diagonal = diagonalized - jnp.diag(diagonal)
        scale = max(float(np.linalg.norm(np.asarray(diagonalized))), np.finfo(float).tiny)
        diagonalization_defect = max(
            diagonalization_defect,
            float(np.linalg.norm(np.asarray(off_diagonal)) / scale),
        )
        coordinates.append(diagonal)
    if diagonalization_defect > policy.relative_joint_diagonalization_tolerance:
        return None, observed_rank, "nongeneric"

    affine_points = jnp.stack(coordinates, axis=1)
    if not jnp.issubdtype(tensor.dtype, jnp.complexfloating):
        imaginary_scale = float(
            np.max(np.abs(np.imag(np.asarray(affine_points))), initial=0.0)
        )
        real_scale = max(
            float(np.max(np.abs(np.real(np.asarray(affine_points))), initial=0.0)),
            1.0,
        )
        if imaginary_scale > policy.relative_rank_tolerance * real_scale:
            return None, observed_rank, "nongeneric"
        affine_points = jnp.real(affine_points).astype(tensor.dtype)

    charted_factors = jnp.zeros(
        (rank, problem.dimension),
        dtype=jnp.result_type(tensor, affine_points),
    )
    charted_factors = charted_factors.at[:, chart_axis].set(1.0)
    for affine_index, physical_axis in enumerate(affine_axes):
        charted_factors = charted_factors.at[:, physical_axis].set(
            affine_points[:, affine_index]
        )

    moment_exponents = _monomial_exponents(problem.dimension - 1, problem.order)
    vandermonde = _evaluate_monomials(affine_points, moment_exponents)
    moments = jnp.stack(
        tuple(
            _affine_moment(tensor, chart_axis, exponent) for exponent in moment_exponents
        )
    )
    moments = moments.astype(vandermonde.dtype)
    weight_solve = solve(
        LeastSquaresProblem(DenseLinearOperator(vandermonde)),
        moments,
        policy=LinearSolvePolicy(
            DenseSVD(),
            rank=RankPolicy(relative_cutoff=policy.relative_rank_tolerance),
        ),
    )
    if not bool(np.all(np.asarray(weight_solve.successful))):
        return None, observed_rank, "provider"
    weights, factors = normalize_waring_components(
        weight_solve.value,
        charted_factors,
        problem.order,
    )
    reconstruction = reconstruct_symmetric_tensor(weights, factors, problem.order)
    if not jnp.issubdtype(tensor.dtype, jnp.complexfloating):
        reconstruction = jnp.real(reconstruction).astype(tensor.dtype)
        weights = jnp.real(weights).astype(tensor.dtype)
        factors = jnp.real(factors).astype(tensor.dtype)
    residual = reconstruction - tensor
    relative_residual = _relative_residual(residual, tensor)
    return (
        _ChartCandidate(
            weights=weights,
            factors=factors,
            reconstruction=reconstruction,
            residual=residual,
            relative_residual=relative_residual,
            chart_axis=chart_axis,
            observed_rank=observed_rank,
            basis_indices=basis_indices,
            hankel_condition=condition,
            commutator_defect=commutator_defect,
            joint_diagonalization_defect=diagonalization_defect,
            joint_spectrum_separation=separation,
        ),
        observed_rank,
        None,
    )


def _refine_candidate(
    plan: SymmetricWaringPlan,
    candidate: _ChartCandidate,
    /,
) -> tuple[_ChartCandidate, bool, int]:
    problem = plan.problem
    complex_parameters = jnp.issubdtype(problem.tensor.dtype, jnp.complexfloating)
    initial = _pack_components(
        candidate.weights,
        candidate.factors,
        complex_parameters,
    )

    def residual(parameters, _arguments):
        weights, factors = _unpack_components(
            parameters,
            problem.rank,
            problem.dimension,
            complex_parameters,
        )
        difference = (
            reconstruct_symmetric_tensor(weights, factors, problem.order) - problem.tensor
        ).reshape((-1,))
        if complex_parameters:
            return jnp.concatenate((jnp.real(difference), jnp.imag(difference)))
        return jnp.real(difference)

    optimization = least_squares(
        residual,
        initial,
        method=plan.refinement.method,
        termination=plan.refinement.termination,
    )
    status = int(np.asarray(optimization.status))
    weights, factors = _unpack_components(
        optimization.parameters,
        problem.rank,
        problem.dimension,
        complex_parameters,
    )
    weights, factors = normalize_waring_components(weights, factors, problem.order)
    reconstruction = reconstruct_symmetric_tensor(weights, factors, problem.order)
    if not complex_parameters:
        weights = jnp.real(weights).astype(problem.tensor.dtype)
        factors = jnp.real(factors).astype(problem.tensor.dtype)
        reconstruction = jnp.real(reconstruction).astype(problem.tensor.dtype)
    difference = reconstruction - problem.tensor
    relative = _relative_residual(difference, problem.tensor)
    accepted = bool(np.isfinite(relative) and relative <= candidate.relative_residual)
    if not accepted:
        return candidate, False, status
    return (
        _ChartCandidate(
            weights=weights,
            factors=factors,
            reconstruction=reconstruction,
            residual=difference,
            relative_residual=relative,
            chart_axis=candidate.chart_axis,
            observed_rank=candidate.observed_rank,
            basis_indices=candidate.basis_indices,
            hankel_condition=candidate.hankel_condition,
            commutator_defect=candidate.commutator_defect,
            joint_diagonalization_defect=candidate.joint_diagonalization_defect,
            joint_spectrum_separation=candidate.joint_spectrum_separation,
        ),
        True,
        status,
    )


def _pack_components(weights: Array, factors: Array, complex_values: bool, /) -> Array:
    values = jnp.concatenate((weights.reshape((-1,)), factors.reshape((-1,))))
    if complex_values:
        return jnp.concatenate((jnp.real(values), jnp.imag(values)))
    return jnp.real(values)


def _unpack_components(
    parameters: Array,
    rank: int,
    dimension: int,
    complex_values: bool,
    /,
) -> tuple[Array, Array]:
    values = jnp.asarray(parameters)
    component_size = rank * (dimension + 1)
    if complex_values:
        values = values[:component_size] + 1j * values[component_size:]
    weights = values[:rank]
    factors = values[rank:].reshape((rank, dimension))
    return weights, factors


def _resource_rejection(
    problem: SymmetricWaringProblem,
    cost: SymmetricWaringCostEstimate,
    policy: SymmetricWaringResourcePolicy,
    /,
) -> str | None:
    comparisons = (
        (problem.rank, policy.maximum_rank, "rank"),
        (cost.tensor_entries, policy.maximum_tensor_entries, "tensor entries"),
        (cost.quotient_basis_size, policy.maximum_basis_size, "quotient basis size"),
        (cost.hankel_entries, policy.maximum_hankel_entries, "Hankel entries"),
        (
            cost.vandermonde_entries,
            policy.maximum_vandermonde_entries,
            "Vandermonde entries",
        ),
        (cost.basis_subsets, policy.maximum_basis_subsets, "basis subsets"),
        (
            cost.refinement_parameters,
            policy.maximum_refinement_parameters,
            "refinement parameters",
        ),
    )
    for requested, maximum, label in comparisons:
        if requested > maximum:
            return f"{label} require {requested}, exceeding limit {maximum}"
    return None


def _quotient_degree(affine_dimension: int, order: int, rank: int, /) -> int:
    for degree in range((order - 1) // 2 + 1):
        if comb(affine_dimension + degree, degree) >= rank:
            return degree
    return -1


def _monomial_exponents(dimension: int, degree: int, /) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    for total_degree in range(degree + 1):
        for indices in combinations_with_replacement(range(dimension), total_degree):
            exponent = [0] * dimension
            for index in indices:
                exponent[index] += 1
            rows.append(tuple(exponent))
    return tuple(rows)


def _affine_moment(tensor: Array, chart_axis: int, exponent: tuple[int, ...], /) -> Array:
    affine_axes = tuple(axis for axis in range(tensor.shape[0]) if axis != chart_axis)
    degree = sum(exponent)
    if degree > tensor.ndim:
        raise ValueError("An affine moment exponent exceeds the tensor order.")
    indices: list[int] = [chart_axis] * (tensor.ndim - degree)
    for axis, power in zip(affine_axes, exponent, strict=True):
        indices.extend([axis] * power)
    return tensor[tuple(indices)]


def _hankel_matrix(
    tensor: Array,
    chart_axis: int,
    row_exponents: tuple[tuple[int, ...], ...],
    column_exponents: tuple[tuple[int, ...], ...],
    /,
) -> Array:
    rows = tuple(
        jnp.stack(
            tuple(
                _affine_moment(
                    tensor,
                    chart_axis,
                    tuple(left + right for left, right in zip(row, column, strict=True)),
                )
                for column in column_exponents
            )
        )
        for row in row_exponents
    )
    return jnp.stack(rows)


def _numerical_rank(singular_values: np.ndarray, relative_tolerance: float, /) -> int:
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return 0
    return int(
        np.count_nonzero(singular_values > relative_tolerance * singular_values[0])
    )


def _select_quotient_basis(
    hankel: np.ndarray,
    rank: int,
    relative_tolerance: float,
    /,
) -> tuple[tuple[int, ...] | None, float]:
    best: tuple[int, ...] | None = None
    best_condition = np.inf
    for indices in combinations(range(hankel.shape[0]), rank):
        submatrix = hankel[np.ix_(indices, indices)]
        singular_values = np.linalg.svd(submatrix, compute_uv=False)
        if _numerical_rank(singular_values, relative_tolerance) != rank:
            continue
        condition = float(singular_values[0] / singular_values[-1])
        if condition < best_condition:
            best = tuple(int(index) for index in indices)
            best_condition = condition
    return best, best_condition


def _commutator_defect(matrices: list[Array], /) -> float:
    defect = 0.0
    tiny = np.finfo(float).tiny
    for left_index in range(len(matrices)):
        for right_index in range(left_index + 1, len(matrices)):
            left = np.asarray(matrices[left_index])
            right = np.asarray(matrices[right_index])
            scale = max(float(np.linalg.norm(left) * np.linalg.norm(right)), tiny)
            defect = max(
                defect,
                float(np.linalg.norm(left @ right - right @ left) / scale),
            )
    return defect


def _relative_separation(values: np.ndarray, /) -> float:
    if values.size <= 1:
        return np.inf
    scale = max(float(np.max(np.abs(values))), 1.0)
    separation = np.inf
    for left in range(values.size):
        for right in range(left + 1, values.size):
            separation = min(separation, float(abs(values[left] - values[right]) / scale))
    return separation


def _evaluate_monomials(
    points: Array,
    exponents: tuple[tuple[int, ...], ...],
    /,
) -> Array:
    powers = jnp.asarray(exponents, dtype=jnp.int32)
    return jnp.prod(points[None, :, :] ** powers[:, None, :], axis=-1)


def _relative_residual(residual: Array, reference: Array, /) -> float:
    residual_norm = float(np.linalg.norm(np.asarray(residual).reshape(-1)))
    reference_norm = float(np.linalg.norm(np.asarray(reference).reshape(-1)))
    tiny = float(np.finfo(np.asarray(reference).real.dtype).tiny)
    return residual_norm / max(reference_norm, tiny)


def _failure_evidence(
    plan: SymmetricWaringPlan,
    /,
    *,
    observed_hankel_rank: int = 0,
    provider: str = "not-invoked",
    detail: str,
) -> SymmetricWaringEvidence:
    return SymmetricWaringEvidence(
        requested_rank=plan.problem.rank,
        observed_hankel_rank=observed_hankel_rank,
        chart_axis=-1,
        quotient_degree=plan.quotient_degree,
        path_accepted=False,
        physical_accepted=False,
        refinement_attempted=False,
        detail=detail,
        provider=provider,
    )


def _failure_result(
    plan: SymmetricWaringPlan,
    status: SymmetricWaringStatus,
    evidence: SymmetricWaringEvidence,
    /,
) -> SymmetricWaringResult:
    tensor = plan.problem.tensor
    reconstruction = jnp.zeros_like(tensor)
    residual = reconstruction - tensor
    residual_norm = jnp.linalg.norm(residual.reshape((-1,)))
    relative = _relative_residual(residual, tensor)
    return SymmetricWaringResult(
        weights=jnp.zeros((0,), dtype=tensor.dtype),
        factors=jnp.zeros((0, plan.problem.dimension), dtype=tensor.dtype),
        reconstruction=reconstruction,
        residual=residual,
        residual_norm=residual_norm,
        relative_residual=relative,
        status=status,
        evidence=evidence,
        plan_id=plan.plan_id,
    )


__all__ = [
    "normalize_waring_components",
    "prepare_symmetric_waring",
    "reconstruct_symmetric_tensor",
    "solve_symmetric_waring",
]
