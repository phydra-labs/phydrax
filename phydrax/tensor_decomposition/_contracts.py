#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from math import comb, isfinite
from numbers import Integral
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..optim import (
    AbstractLeastSquaresMethod,
    LevenbergMarquardt,
    OptimizationTermination,
)


class SymmetricWaringStatus(IntEnum):
    """Outcome of one admitted algebraic symmetric-tensor decomposition."""

    SUCCESS = 0
    RESOURCE_REJECTED = 1
    ALGEBRAIC_INITIALIZATION_NOT_ADMISSIBLE = 2
    REQUESTED_RANK_MISMATCH = 3
    OVER_REQUESTED_RANK_OR_REPEATED_FACTOR = 4
    NONGENERIC_OR_MULTIPLE_DECOMPOSITION = 5
    ALGEBRAIC_INITIALIZATION_FAILED = 6
    RECONSTRUCTION_TOLERANCE_NOT_MET = 7


class SymmetricWaringRankPolicy(StrictModule):
    """Numerical rank, isolation, and physical reconstruction acceptance policy."""

    relative_rank_tolerance: float = eqx.field(static=True)
    maximum_hankel_condition: float = eqx.field(static=True)
    relative_joint_spectrum_separation: float = eqx.field(static=True)
    relative_commutator_tolerance: float = eqx.field(static=True)
    relative_joint_diagonalization_tolerance: float = eqx.field(static=True)
    relative_reconstruction_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        relative_rank_tolerance: float = 1.0e-7,
        maximum_hankel_condition: float = 1.0e9,
        relative_joint_spectrum_separation: float = 1.0e-7,
        relative_commutator_tolerance: float = 1.0e-6,
        relative_joint_diagonalization_tolerance: float = 1.0e-6,
        relative_reconstruction_tolerance: float = 2.0e-6,
    ):
        values = (
            float(relative_rank_tolerance),
            float(maximum_hankel_condition),
            float(relative_joint_spectrum_separation),
            float(relative_commutator_tolerance),
            float(relative_joint_diagonalization_tolerance),
            float(relative_reconstruction_tolerance),
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Symmetric Waring rank tolerances must be positive finite.")
        if values[1] <= 1.0:
            raise ValueError("maximum_hankel_condition must exceed one.")
        (
            self.relative_rank_tolerance,
            self.maximum_hankel_condition,
            self.relative_joint_spectrum_separation,
            self.relative_commutator_tolerance,
            self.relative_joint_diagonalization_tolerance,
            self.relative_reconstruction_tolerance,
        ) = values


class SymmetricWaringResourcePolicy(StrictModule):
    """Hard limits for dense tensor, Hankel, basis-search, and refinement work."""

    maximum_rank: int = eqx.field(static=True)
    maximum_tensor_entries: int = eqx.field(static=True)
    maximum_basis_size: int = eqx.field(static=True)
    maximum_hankel_entries: int = eqx.field(static=True)
    maximum_vandermonde_entries: int = eqx.field(static=True)
    maximum_basis_subsets: int = eqx.field(static=True)
    maximum_refinement_parameters: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_rank: int = 64,
        maximum_tensor_entries: int = 1_000_000,
        maximum_basis_size: int = 256,
        maximum_hankel_entries: int = 1_000_000,
        maximum_vandermonde_entries: int = 1_000_000,
        maximum_basis_subsets: int = 16_384,
        maximum_refinement_parameters: int = 100_000,
    ):
        values = (
            _positive_integer(maximum_rank, "maximum_rank"),
            _positive_integer(maximum_tensor_entries, "maximum_tensor_entries"),
            _positive_integer(maximum_basis_size, "maximum_basis_size"),
            _positive_integer(maximum_hankel_entries, "maximum_hankel_entries"),
            _positive_integer(maximum_vandermonde_entries, "maximum_vandermonde_entries"),
            _positive_integer(maximum_basis_subsets, "maximum_basis_subsets"),
            _positive_integer(
                maximum_refinement_parameters, "maximum_refinement_parameters"
            ),
        )
        (
            self.maximum_rank,
            self.maximum_tensor_entries,
            self.maximum_basis_size,
            self.maximum_hankel_entries,
            self.maximum_vandermonde_entries,
            self.maximum_basis_subsets,
            self.maximum_refinement_parameters,
        ) = values


class SymmetricWaringRefinement(StrictModule):
    """Optional nonlinear least-squares refinement through ``phydrax.optim``."""

    method: AbstractLeastSquaresMethod
    termination: OptimizationTermination
    enabled: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        enabled: bool = True,
        method: AbstractLeastSquaresMethod | None = None,
        termination: OptimizationTermination | None = None,
    ):
        method_ = LevenbergMarquardt() if method is None else method
        termination_ = (
            OptimizationTermination(maximum_steps=32)
            if termination is None
            else termination
        )
        if not isinstance(method_, AbstractLeastSquaresMethod):
            raise TypeError("method must be an AbstractLeastSquaresMethod or None.")
        if not isinstance(termination_, OptimizationTermination):
            raise TypeError("termination must be an OptimizationTermination or None.")
        self.method = method_
        self.termination = termination_
        self.enabled = bool(enabled)


class SymmetricWaringProblem(StrictModule):
    """One dense symmetric tensor and a requested symmetric CP/Waring rank.

    The represented convention is ``T = sum_s weights[s] * factors[s]**order``:
    powers are algebraic, without conjugating complex factors.
    """

    tensor: Array
    rank: int = eqx.field(static=True)
    order: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    independent_entry_count: int = eqx.field(static=True)
    symmetry_error: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        tensor: ArrayLike,
        rank: int,
        /,
        *,
        symmetry_tolerance: float = 1.0e-6,
        problem_id: str | None = None,
    ):
        values = jnp.asarray(tensor)
        if values.ndim < 3:
            raise ValueError("A symmetric Waring tensor must have order at least three.")
        if any(size != values.shape[0] for size in values.shape):
            raise ValueError("Every symmetric tensor mode must have one dimension.")
        dimension = values.shape[0]
        if dimension < 2:
            raise ValueError("A symmetric Waring tensor dimension must be at least two.")
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype("float64")
        host = np.asarray(values)
        if not np.all(np.isfinite(host)):
            raise ValueError("Symmetric Waring tensor entries must be finite.")
        tolerance = float(symmetry_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("symmetry_tolerance must be positive finite.")
        norm = float(np.linalg.norm(host.reshape(-1)))
        scale = max(norm, float(np.finfo(host.real.dtype).tiny))
        symmetry_error = 0.0
        for axis in range(values.ndim - 1):
            swapped = np.swapaxes(host, axis, axis + 1)
            symmetry_error = max(
                symmetry_error,
                float(np.linalg.norm((host - swapped).reshape(-1)) / scale),
            )
        if symmetry_error > tolerance:
            raise ValueError(
                "tensor is not symmetric within symmetry_tolerance; "
                f"relative adjacent-swap defect is {symmetry_error:.3e}."
            )
        rank_ = _positive_integer(rank, "rank")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "symmetric-waring-problem",
                    "rank": rank_,
                    "tensor": array_tree_fingerprint(host),
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.tensor = values
        self.rank = rank_
        self.order = values.ndim
        self.dimension = dimension
        self.independent_entry_count = comb(dimension + values.ndim - 1, values.ndim)
        self.symmetry_error = symmetry_error
        self.problem_id = identifier


class SymmetricWaringCostEstimate(StrictModule):
    """Exact structural sizes used for resource admission."""

    tensor_entries: int = eqx.field(static=True)
    quotient_basis_size: int = eqx.field(static=True)
    hankel_entries: int = eqx.field(static=True)
    vandermonde_entries: int = eqx.field(static=True)
    basis_subsets: int = eqx.field(static=True)
    refinement_parameters: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        tensor_entries: int,
        quotient_basis_size: int,
        hankel_entries: int,
        vandermonde_entries: int,
        basis_subsets: int,
        refinement_parameters: int,
    ):
        values = tuple(
            _nonnegative_integer(value, name)
            for value, name in (
                (tensor_entries, "tensor_entries"),
                (quotient_basis_size, "quotient_basis_size"),
                (hankel_entries, "hankel_entries"),
                (vandermonde_entries, "vandermonde_entries"),
                (basis_subsets, "basis_subsets"),
                (refinement_parameters, "refinement_parameters"),
            )
        )
        (
            self.tensor_entries,
            self.quotient_basis_size,
            self.hankel_entries,
            self.vandermonde_entries,
            self.basis_subsets,
            self.refinement_parameters,
        ) = values


class SymmetricWaringPlan(StrictModule):
    """Prepared affine-chart and finite Hankel work for one fixed problem."""

    problem: SymmetricWaringProblem
    rank_policy: SymmetricWaringRankPolicy
    resources: SymmetricWaringResourcePolicy
    refinement: SymmetricWaringRefinement
    chart_axes: tuple[int, ...] = eqx.field(static=True)
    quotient_degree: int = eqx.field(static=True)
    quotient_exponents: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    algebraically_admissible: bool = eqx.field(static=True)
    resource_admitted: bool = eqx.field(static=True)
    resource_rejection: str | None = eqx.field(static=True)
    cost: SymmetricWaringCostEstimate
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: SymmetricWaringProblem,
        rank_policy: SymmetricWaringRankPolicy,
        resources: SymmetricWaringResourcePolicy,
        refinement: SymmetricWaringRefinement,
        chart_axes: Sequence[int],
        quotient_degree: int,
        quotient_exponents: Sequence[Sequence[int]],
        algebraically_admissible: bool,
        resource_admitted: bool,
        resource_rejection: str | None,
        cost: SymmetricWaringCostEstimate,
        /,
    ):
        if not isinstance(problem, SymmetricWaringProblem):
            raise TypeError("problem must be a SymmetricWaringProblem.")
        if not isinstance(rank_policy, SymmetricWaringRankPolicy):
            raise TypeError("rank_policy must be a SymmetricWaringRankPolicy.")
        if not isinstance(resources, SymmetricWaringResourcePolicy):
            raise TypeError("resources must be a SymmetricWaringResourcePolicy.")
        if not isinstance(refinement, SymmetricWaringRefinement):
            raise TypeError("refinement must be a SymmetricWaringRefinement.")
        if not isinstance(cost, SymmetricWaringCostEstimate):
            raise TypeError("cost must be a SymmetricWaringCostEstimate.")
        axes = tuple(chart_axes)
        if not axes or len(set(axes)) != len(axes):
            raise ValueError("chart_axes must be nonempty and unique.")
        if any(axis < 0 or axis >= problem.dimension for axis in axes):
            raise ValueError("chart_axes contains an out-of-range tensor axis.")
        exponents = tuple(tuple(row) for row in quotient_exponents)
        if any(
            len(row) != problem.dimension - 1 or any(value < 0 for value in row)
            for row in exponents
        ):
            raise ValueError("quotient_exponents contains an invalid affine monomial.")
        rejection = None if resource_rejection is None else str(resource_rejection)
        if bool(resource_admitted) == (rejection is not None):
            raise ValueError("resource_admitted and resource_rejection disagree.")
        self.problem = problem
        self.rank_policy = rank_policy
        self.resources = resources
        self.refinement = refinement
        self.chart_axes = axes
        self.quotient_degree = int(quotient_degree)
        self.quotient_exponents = exponents
        self.algebraically_admissible = bool(algebraically_admissible)
        self.resource_admitted = bool(resource_admitted)
        self.resource_rejection = rejection
        self.cost = cost
        self.plan_id = canonical_fingerprint(
            {
                "kind": "symmetric-waring-plan",
                "problem": problem.problem_id,
                "charts": list(axes),
                "quotient_degree": self.quotient_degree,
                "quotient_exponents": [list(row) for row in exponents],
                "rank_policy": _rank_policy_payload(rank_policy),
                "resources": _resource_policy_payload(resources),
                "refinement": {
                    "enabled": refinement.enabled,
                    "method": refinement.method.method_id,
                    "termination": _termination_payload(refinement.termination),
                },
            }
        )


class SymmetricWaringEvidence(StrictModule):
    """Path-level algebraic/provider evidence and separate physical acceptance."""

    requested_rank: int = eqx.field(static=True)
    observed_hankel_rank: int = eqx.field(static=True)
    chart_axis: int = eqx.field(static=True)
    quotient_degree: int = eqx.field(static=True)
    basis_indices: tuple[int, ...] = eqx.field(static=True)
    hankel_condition: Array
    commutator_defect: Array
    joint_diagonalization_defect: Array
    joint_spectrum_separation: Array
    algebraic_relative_residual: Array
    final_relative_residual: Array
    path_accepted: Array
    physical_accepted: Array
    refinement_attempted: bool = eqx.field(static=True)
    refinement_accepted: Array
    refinement_status: Array
    provider: str = eqx.field(static=True)
    detail: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        requested_rank: int,
        observed_hankel_rank: int,
        chart_axis: int,
        quotient_degree: int,
        basis_indices: Sequence[int] = (),
        hankel_condition: Any = jnp.inf,
        commutator_defect: Any = jnp.inf,
        joint_diagonalization_defect: Any = jnp.inf,
        joint_spectrum_separation: Any = 0.0,
        algebraic_relative_residual: Any = jnp.inf,
        final_relative_residual: Any = jnp.inf,
        path_accepted: Any = False,
        physical_accepted: Any = False,
        refinement_attempted: bool = False,
        refinement_accepted: Any = False,
        refinement_status: Any = -1,
        provider: str = "not-invoked",
        detail: str = "",
    ):
        self.requested_rank = int(requested_rank)
        self.observed_hankel_rank = int(observed_hankel_rank)
        self.chart_axis = int(chart_axis)
        self.quotient_degree = int(quotient_degree)
        self.basis_indices = tuple(basis_indices)
        self.hankel_condition = jnp.asarray(hankel_condition)
        self.commutator_defect = jnp.asarray(commutator_defect)
        self.joint_diagonalization_defect = jnp.asarray(joint_diagonalization_defect)
        self.joint_spectrum_separation = jnp.asarray(joint_spectrum_separation)
        self.algebraic_relative_residual = jnp.asarray(algebraic_relative_residual)
        self.final_relative_residual = jnp.asarray(final_relative_residual)
        self.path_accepted = jnp.asarray(path_accepted, dtype=jnp.bool_)
        self.physical_accepted = jnp.asarray(physical_accepted, dtype=jnp.bool_)
        self.refinement_attempted = bool(refinement_attempted)
        self.refinement_accepted = jnp.asarray(refinement_accepted, dtype=jnp.bool_)
        self.refinement_status = jnp.asarray(refinement_status, dtype=jnp.int32)
        self.provider = str(provider)
        self.detail = str(detail)


class SymmetricWaringResult(StrictModule):
    """Normalized components in deterministic permutation/sign/phase convention.

    Every factor has Euclidean norm one. Its largest-magnitude coordinate is
    nonnegative real; the inverse sign or phase power is absorbed into the
    corresponding weight. Components are then ordered lexicographically. These
    conventions select one representative only and do not assert global
    uniqueness beyond the reported isolated joint-spectrum path.
    """

    weights: Array
    factors: Array
    reconstruction: Array
    residual: Array
    residual_norm: Array
    relative_residual: Array
    status: Array
    evidence: SymmetricWaringEvidence
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        weights: ArrayLike,
        factors: ArrayLike,
        reconstruction: ArrayLike,
        residual: ArrayLike,
        residual_norm: Any,
        relative_residual: Any,
        status: int | SymmetricWaringStatus | ArrayLike,
        evidence: SymmetricWaringEvidence,
        plan_id: str,
    ):
        weights_ = jnp.asarray(weights)
        factors_ = jnp.asarray(factors)
        reconstruction_ = jnp.asarray(reconstruction)
        residual_ = jnp.asarray(residual)
        if weights_.ndim != 1 or factors_.ndim != 2:
            raise ValueError("weights and factors must have shapes (rank,) and (rank,n).")
        if factors_.shape[0] != weights_.shape[0]:
            raise ValueError("weights and factors must contain the same rank.")
        if reconstruction_.shape != residual_.shape:
            raise ValueError("reconstruction and residual shapes must agree.")
        if not isinstance(evidence, SymmetricWaringEvidence):
            raise TypeError("evidence must be SymmetricWaringEvidence.")
        identifier = str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.weights = weights_
        self.factors = factors_
        self.reconstruction = reconstruction_
        self.residual = residual_
        self.residual_norm = jnp.asarray(residual_norm)
        self.relative_residual = jnp.asarray(relative_residual)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.evidence = evidence
        self.plan_id = identifier

    @property
    def successful(self) -> Array:
        return self.status == int(SymmetricWaringStatus.SUCCESS)

    @property
    def rank(self) -> int:
        return self.weights.shape[0]


def _positive_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _rank_policy_payload(policy: SymmetricWaringRankPolicy, /) -> dict[str, float]:
    return {
        "relative_rank_tolerance": policy.relative_rank_tolerance,
        "maximum_hankel_condition": policy.maximum_hankel_condition,
        "relative_joint_spectrum_separation": policy.relative_joint_spectrum_separation,
        "relative_commutator_tolerance": policy.relative_commutator_tolerance,
        "relative_joint_diagonalization_tolerance": (
            policy.relative_joint_diagonalization_tolerance
        ),
        "relative_reconstruction_tolerance": (policy.relative_reconstruction_tolerance),
    }


def _resource_policy_payload(policy: SymmetricWaringResourcePolicy, /) -> dict[str, int]:
    return {
        "maximum_rank": policy.maximum_rank,
        "maximum_tensor_entries": policy.maximum_tensor_entries,
        "maximum_basis_size": policy.maximum_basis_size,
        "maximum_hankel_entries": policy.maximum_hankel_entries,
        "maximum_vandermonde_entries": policy.maximum_vandermonde_entries,
        "maximum_basis_subsets": policy.maximum_basis_subsets,
        "maximum_refinement_parameters": policy.maximum_refinement_parameters,
    }


def _termination_payload(termination: OptimizationTermination, /) -> dict[str, Any]:
    return {
        "absolute_optimality": termination.absolute_optimality,
        "relative_optimality": termination.relative_optimality,
        "absolute_step": termination.absolute_step,
        "relative_step": termination.relative_step,
        "maximum_steps": termination.maximum_steps,
        "maximum_evaluations": termination.maximum_evaluations,
    }


__all__ = [
    "SymmetricWaringCostEstimate",
    "SymmetricWaringEvidence",
    "SymmetricWaringPlan",
    "SymmetricWaringProblem",
    "SymmetricWaringRankPolicy",
    "SymmetricWaringRefinement",
    "SymmetricWaringResourcePolicy",
    "SymmetricWaringResult",
    "SymmetricWaringStatus",
]
