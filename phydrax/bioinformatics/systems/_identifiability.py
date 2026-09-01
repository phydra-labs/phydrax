#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...linalg import DenseLinearOperator
from ...linalg.svd import svd, SVDProblem, SVDSolvePolicy, SVDSolveResult
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


class LocalIdentifiabilityStatus(IntEnum):
    """Terminal status for local sensitivity-rank diagnostics."""

    LOCALLY_IDENTIFIABLE = 0
    RANK_DEFICIENT = 1
    UNDERDETERMINED = 2
    NONFINITE_SENSITIVITY = 3
    SVD_FAILURE = 4


class GlobalIdentifiabilityStatus(IntEnum):
    """Claim status for finite candidate-set global diagnostics."""

    IDENTIFIABLE_ON_EXHAUSTIVE_SET = 0
    NOT_IDENTIFIABLE = 1
    INCONCLUSIVE_NONEXHAUSTIVE = 2
    NONFINITE_OUTPUT = 3


class IdentifiabilityCapacityError(ValueError):
    """Raised before pairwise materialization when a declared capacity is exceeded."""


class LocalSensitivityEvidence(StrictModule):
    """Native SVD, Fisher information, cutoff, and local-rank evidence."""

    svd_result: SVDSolveResult
    fisher_information: Array
    singular_values: Array
    numerical_rank: Array
    cutoff: Array
    condition_number: Array
    output_dimension: int = eqx.field(static=True)
    parameter_dimension: int = eqx.field(static=True)
    rank_claim: str = eqx.field(static=True)


class LocalIdentifiabilityResult(StrictModule):
    """Local differential identifiability; never promoted to a global claim."""

    valid: Array
    status: Array
    sensitivity: Array
    locally_identifiable: Array
    evidence: LocalSensitivityEvidence
    method_contract: BioinformaticsMethodContract

    @property
    def successful(self) -> Array:
        return self.valid & (
            self.status == int(LocalIdentifiabilityStatus.LOCALLY_IDENTIFIABLE)
        )


class GlobalIdentifiabilityEvidence(StrictModule):
    """Pairwise parameter/output separation and explicit completeness evidence."""

    pair_indices: Array
    parameter_distances: Array
    output_distances: Array
    collision_mask: Array
    collision_count: Array
    minimum_output_separation: Array
    exhaustive: bool = eqx.field(static=True)
    claim_scope: str = eqx.field(static=True)


class GlobalIdentifiabilityResult(StrictModule):
    """Global candidate-set result with conclusive and inconclusive claims separated."""

    valid: Array
    status: Array
    globally_identifiable: Array
    conclusive: Array
    evidence: GlobalIdentifiabilityEvidence
    method_contract: BioinformaticsMethodContract


def _local_contract(relative_tolerance: float, /) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "local-sensitivity-identifiability",
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.EXACT_AD,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "Local rank is thresholded relative to the largest sensitivity singular value; "
            "near-symmetries are tolerance dependent."
        ),
        truncation_statement="All output-by-parameter sensitivity directions are included.",
        capacity_semantics="Native dense SVD preflights the complete sensitivity matrix.",
        assumptions=(
            "The supplied observation map is differentiable at the parameter point.",
            "Full local rank is evidence only for local, not global, identifiability.",
        ),
        nondifferentiable_outputs=(
            "numerical_rank",
            "status",
            "valid",
            "locally_identifiable",
        ),
        relative_tolerance=relative_tolerance,
    )


def _global_contract(
    tolerance: float, exhaustive: bool, /
) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "finite-candidate-global-identifiability",
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.NONE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "Candidate outputs separated by at most the absolute tolerance are treated as "
            "observationally indistinguishable."
        ),
        truncation_statement=(
            "The supplied candidate set is declared exhaustive."
            if exhaustive
            else "The supplied candidate set is nonexhaustive; absence of collisions is inconclusive."
        ),
        capacity_semantics="All unique candidate pairs are preflighted and compared.",
        assumptions=(
            "Candidate parameter rows represent distinct scientific parameterizations.",
        ),
        nondifferentiable_outputs=(
            "collision_mask",
            "status",
            "conclusive",
            "globally_identifiable",
        ),
        absolute_tolerance=tolerance,
    )


def local_identifiability(
    observation_map: Callable[[Array], Any],
    parameters: ArrayLike,
    /,
    *,
    relative_tolerance: float = 1.0e-8,
) -> LocalIdentifiabilityResult:
    """Differentiate an observation map and diagnose full local parameter rank."""

    if not callable(observation_map):
        raise TypeError("observation_map must be callable.")
    parameters_ = jnp.asarray(parameters)
    if parameters_.ndim != 1 or parameters_.shape[0] < 1:
        raise ValueError("parameters must be a non-empty one-dimensional array.")
    parameters_ = (
        parameters_
        if jnp.issubdtype(parameters_.dtype, jnp.inexact)
        else parameters_.astype(float)
    )
    tolerance = float(relative_tolerance)
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive.")

    def flattened(value: Array) -> Array:
        output = jnp.asarray(observation_map(value))
        if not jnp.issubdtype(output.dtype, jnp.inexact):
            output = output.astype(float)
        return output.reshape((-1,))

    sensitivity = jax.jacrev(flattened)(parameters_)
    if sensitivity.ndim != 2 or sensitivity.shape[0] < 1:
        raise ValueError(
            "observation_map must return at least one numeric scalar output."
        )
    rows, columns = sensitivity.shape
    count = min(rows, columns)
    svd_result = svd(
        SVDProblem(
            DenseLinearOperator(sensitivity),
            problem_id="bioinformatics-local-identifiability",
        ),
        policy=SVDSolvePolicy(count=count, which="largest"),
    )
    singular_values = jnp.asarray(svd_result.singular_values)
    largest = jnp.max(singular_values, initial=0.0)
    cutoff = relative_tolerance * jnp.maximum(largest, 1.0)
    rank = jnp.sum(singular_values > cutoff, dtype=jnp.int32)
    finite = jnp.all(jnp.isfinite(sensitivity)) & jnp.all(jnp.isfinite(singular_values))
    svd_valid = jnp.asarray(svd_result.successful)
    underdetermined = rows < columns
    locally_identifiable = (
        finite & svd_valid & jnp.asarray(not underdetermined) & (rank == columns)
    )
    status = jnp.where(
        ~finite,
        int(LocalIdentifiabilityStatus.NONFINITE_SENSITIVITY),
        jnp.where(
            ~svd_valid,
            int(LocalIdentifiabilityStatus.SVD_FAILURE),
            jnp.where(
                underdetermined,
                int(LocalIdentifiabilityStatus.UNDERDETERMINED),
                jnp.where(
                    rank < columns,
                    int(LocalIdentifiabilityStatus.RANK_DEFICIENT),
                    int(LocalIdentifiabilityStatus.LOCALLY_IDENTIFIABLE),
                ),
            ),
        ),
    ).astype(jnp.int32)
    smallest = jnp.min(singular_values, initial=jnp.inf)
    condition = jnp.where(smallest > cutoff, largest / smallest, jnp.inf)
    fisher = oe.contract("op,oq->pq", sensitivity, sensitivity)
    evidence = LocalSensitivityEvidence(
        svd_result=svd_result,
        fisher_information=fisher,
        singular_values=singular_values,
        numerical_rank=rank,
        cutoff=cutoff,
        condition_number=condition,
        output_dimension=rows,
        parameter_dimension=columns,
        rank_claim="local differential rank only; no global injectivity claim",
    )
    return LocalIdentifiabilityResult(
        valid=finite & svd_valid,
        status=status,
        sensitivity=sensitivity,
        locally_identifiable=locally_identifiable,
        evidence=evidence,
        method_contract=_local_contract(tolerance),
    )


def global_candidate_identifiability(
    parameter_candidates: ArrayLike,
    predicted_outputs: ArrayLike,
    /,
    *,
    exhaustive: bool = False,
    tolerance: float = 1.0e-8,
    max_pairs: int = 1_000_000,
) -> GlobalIdentifiabilityResult:
    """Find global nonidentifiability witnesses or certify an exhaustive finite set."""

    parameters = jnp.asarray(parameter_candidates)
    outputs = jnp.asarray(predicted_outputs)
    if parameters.ndim != 2 or parameters.shape[0] < 1:
        raise ValueError("parameter_candidates must have shape (candidate, parameter).")
    if outputs.ndim < 2 or outputs.shape[0] != parameters.shape[0]:
        raise ValueError("predicted_outputs must share the candidate leading axis.")
    if parameters.shape[1] < 1 or np.prod(outputs.shape[1:]) < 1:
        raise ValueError(
            "Candidates and outputs must have non-empty trailing dimensions."
        )
    tolerance_ = float(tolerance)
    capacity = int(max_pairs)
    if not isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")
    candidate_count = parameters.shape[0]
    pair_count = candidate_count * (candidate_count - 1) // 2
    if capacity < pair_count:
        raise IdentifiabilityCapacityError(
            f"Global diagnostic requires {pair_count} candidate pairs; capacity is {capacity}."
        )
    left, right = np.triu_indices(candidate_count, k=1)
    pair_indices = jnp.asarray(np.stack((left, right), axis=1), dtype=jnp.int32)
    flat_outputs = outputs.reshape((candidate_count, -1))
    parameter_delta = parameters[pair_indices[:, 0]] - parameters[pair_indices[:, 1]]
    output_delta = flat_outputs[pair_indices[:, 0]] - flat_outputs[pair_indices[:, 1]]
    parameter_distance = jnp.max(jnp.abs(parameter_delta), axis=1)
    output_distance = jnp.max(jnp.abs(output_delta), axis=1)
    distinct_parameters = parameter_distance > tolerance_
    collisions = distinct_parameters & (output_distance <= tolerance_)
    finite = jnp.all(jnp.isfinite(parameters)) & jnp.all(jnp.isfinite(outputs))
    collision_count = jnp.sum(collisions, dtype=jnp.int32)
    has_collision = collision_count > 0
    exhaustive_ = bool(exhaustive)
    conclusive = finite & (has_collision | exhaustive_)
    globally_identifiable = finite & ~has_collision & exhaustive_
    status = jnp.where(
        ~finite,
        int(GlobalIdentifiabilityStatus.NONFINITE_OUTPUT),
        jnp.where(
            has_collision,
            int(GlobalIdentifiabilityStatus.NOT_IDENTIFIABLE),
            (
                int(GlobalIdentifiabilityStatus.IDENTIFIABLE_ON_EXHAUSTIVE_SET)
                if exhaustive_
                else int(GlobalIdentifiabilityStatus.INCONCLUSIVE_NONEXHAUSTIVE)
            ),
        ),
    ).astype(jnp.int32)
    minimum_separation = jnp.min(
        jnp.where(distinct_parameters, output_distance, jnp.inf), initial=jnp.inf
    )
    evidence = GlobalIdentifiabilityEvidence(
        pair_indices=pair_indices,
        parameter_distances=parameter_distance,
        output_distances=output_distance,
        collision_mask=collisions,
        collision_count=collision_count,
        minimum_output_separation=minimum_separation,
        exhaustive=exhaustive_,
        claim_scope=(
            "global over caller-declared exhaustive finite candidate set"
            if exhaustive_
            else "nonexhaustive finite candidate set; only collisions are globally conclusive"
        ),
    )
    return GlobalIdentifiabilityResult(
        valid=finite,
        status=status,
        globally_identifiable=globally_identifiable,
        conclusive=conclusive,
        evidence=evidence,
        method_contract=_global_contract(tolerance_, exhaustive_),
    )


__all__ = [
    "global_candidate_identifiability",
    "local_identifiability",
    "GlobalIdentifiabilityEvidence",
    "GlobalIdentifiabilityResult",
    "GlobalIdentifiabilityStatus",
    "IdentifiabilityCapacityError",
    "LocalIdentifiabilityResult",
    "LocalIdentifiabilityStatus",
    "LocalSensitivityEvidence",
]
