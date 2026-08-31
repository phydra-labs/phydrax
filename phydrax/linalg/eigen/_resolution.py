#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from scipy.optimize import linear_sum_assignment

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._general import GeneralEigenSolveResult


class GeneralEigenMatchStatus(IntEnum):
    """Per-coarse-mode cross-resolution evidence status."""

    TRUSTED = 0
    UNRESOLVED = 1
    NONCONVERGED = 2
    ILL_CONDITIONED = 3
    AMBIGUOUS_CLUSTER = 4
    UNMATCHED = 5
    INDETERMINATE = 6


class GeneralEigenResolutionPolicy(StrictModule, NonTrainableState):
    """Chordal drift, residual, clustering, and conditioning thresholds."""

    chordal_tolerance: float = eqx.field(static=True)
    normalized_drift_tolerance: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    cluster_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        chordal_tolerance: float = 1e-6,
        normalized_drift_tolerance: float = 0.1,
        residual_tolerance: float = 1e-7,
        condition_limit: float = 1e10,
        cluster_tolerance: float = 1e-8,
    ):
        values = tuple(
            float(value)
            for value in (
                chordal_tolerance,
                normalized_drift_tolerance,
                residual_tolerance,
                condition_limit,
                cluster_tolerance,
            )
        )
        if (
            any(not math.isfinite(value) or value < 0.0 for value in values)
            or values[3] <= 0.0
        ):
            raise ValueError("Eigen resolution policy values must be finite and valid.")
        (
            self.chordal_tolerance,
            self.normalized_drift_tolerance,
            self.residual_tolerance,
            self.condition_limit,
            self.cluster_tolerance,
        ) = values
        self.policy_id = canonical_fingerprint(
            {
                "kind": "general-eigen-resolution-policy",
                "chordal_tolerance": values[0],
                "normalized_drift_tolerance": values[1],
                "residual_tolerance": values[2],
                "condition_limit": values[3],
                "cluster_tolerance": values[4],
            }
        )


class GeneralEigenResolutionReport(StrictModule, NonTrainableState):
    """One-to-one homogeneous matching and per-mode trust evidence."""

    fine_indices: Array
    homogeneous_classes: Array
    chordal_distances: Array
    local_separations: Array
    normalized_drifts: Array
    maximum_relative_residuals: Array
    maximum_condition_estimates: Array
    coarse_cluster_ids: Array
    fine_cluster_ids: Array
    matched_fine_cluster_ids: Array
    statuses: Array
    trusted_mask: Array
    matched_mask: Array
    matched_count: Array
    trusted_count: Array
    report_id: str = eqx.field(static=True)
    coarse_problem_id: str = eqx.field(static=True)
    fine_problem_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    host_only: bool = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.all(self.matched_mask | (self.homogeneous_classes == 2))


def compare_general_eigen_resolutions(
    coarse: GeneralEigenSolveResult,
    fine: GeneralEigenSolveResult,
    /,
    *,
    policy: GeneralEigenResolutionPolicy | None = None,
) -> GeneralEigenResolutionReport:
    """Match two selected homogeneous spectra without ordinal assumptions."""
    if not isinstance(coarse, GeneralEigenSolveResult) or not isinstance(
        fine, GeneralEigenSolveResult
    ):
        raise TypeError("coarse and fine must be GeneralEigenSolveResult values.")
    policy_ = GeneralEigenResolutionPolicy() if policy is None else policy
    if not isinstance(policy_, GeneralEigenResolutionPolicy):
        raise TypeError("policy must be a GeneralEigenResolutionPolicy or None.")

    coarse_alpha = np.asarray(coarse.alpha, dtype=complex).reshape((-1,))
    coarse_beta = np.asarray(coarse.beta, dtype=complex).reshape((-1,))
    fine_alpha = np.asarray(fine.alpha, dtype=complex).reshape((-1,))
    fine_beta = np.asarray(fine.beta, dtype=complex).reshape((-1,))
    coarse_class = _homogeneous_classes(coarse)
    fine_class = _homogeneous_classes(fine)
    coarse_count = coarse_alpha.size
    fine_indices = np.full((coarse_count,), -1, dtype=np.int32)
    distances = np.full((coarse_count,), np.inf, dtype=float)
    matched = np.zeros((coarse_count,), dtype=bool)

    for class_value in (0, 1):
        coarse_candidates = np.flatnonzero(coarse_class == class_value)
        fine_candidates = np.flatnonzero(fine_class == class_value)
        if coarse_candidates.size == 0 or fine_candidates.size == 0:
            continue
        costs = _chordal_matrix(
            coarse_alpha[coarse_candidates],
            coarse_beta[coarse_candidates],
            fine_alpha[fine_candidates],
            fine_beta[fine_candidates],
        )
        deterministic = np.finfo(float).eps * np.arange(costs.size).reshape(costs.shape)
        rows, columns = linear_sum_assignment(costs + deterministic)
        selected_coarse = coarse_candidates[rows]
        selected_fine = fine_candidates[columns]
        fine_indices[selected_coarse] = selected_fine
        distances[selected_coarse] = costs[rows, columns]
        matched[selected_coarse] = True

    coarse_separation = _local_separations(coarse_alpha, coarse_beta, coarse_class)
    fine_separation = _local_separations(fine_alpha, fine_beta, fine_class)
    local = np.full((coarse_count,), np.inf, dtype=float)
    normalized = np.full((coarse_count,), np.inf, dtype=float)
    valid_match = np.flatnonzero(matched)
    if valid_match.size:
        target = fine_indices[valid_match]
        local[valid_match] = np.minimum(
            coarse_separation[valid_match],
            fine_separation[target],
        )
        normalized[valid_match] = distances[valid_match] / np.maximum(
            local[valid_match],
            np.finfo(float).eps,
        )

    coarse_clusters = _cluster_ids(
        coarse_alpha,
        coarse_beta,
        coarse_class,
        policy_.cluster_tolerance,
    )
    fine_clusters = _cluster_ids(
        fine_alpha,
        fine_beta,
        fine_class,
        policy_.cluster_tolerance,
    )
    matched_fine_clusters = np.full((coarse_count,), -1, dtype=np.int32)
    matched_fine_clusters[valid_match] = fine_clusters[fine_indices[valid_match]]

    coarse_residual = np.asarray(
        coarse.diagnostics.right_relative_residuals,
        dtype=float,
    )
    fine_residual = np.asarray(
        fine.diagnostics.right_relative_residuals,
        dtype=float,
    )
    residual = np.full((coarse_count,), np.inf, dtype=float)
    residual[valid_match] = np.maximum(
        coarse_residual[valid_match],
        fine_residual[fine_indices[valid_match]],
    )
    coarse_condition = np.asarray(
        coarse.diagnostics.eigenvalue_condition_estimates,
        dtype=float,
    )
    fine_condition = np.asarray(
        fine.diagnostics.eigenvalue_condition_estimates,
        dtype=float,
    )
    condition = np.full((coarse_count,), np.inf, dtype=float)
    condition[valid_match] = np.maximum(
        coarse_condition[valid_match],
        fine_condition[fine_indices[valid_match]],
    )
    coarse_converged = np.asarray(coarse.diagnostics.converged_mask, dtype=bool)
    fine_converged = np.asarray(fine.diagnostics.converged_mask, dtype=bool)

    statuses = np.full(
        (coarse_count,),
        int(GeneralEigenMatchStatus.UNMATCHED),
        dtype=np.int32,
    )
    statuses[coarse_class == 2] = int(GeneralEigenMatchStatus.INDETERMINATE)
    for index in valid_match:
        fine_index = fine_indices[index]
        coarse_cluster_size = np.count_nonzero(coarse_clusters == coarse_clusters[index])
        fine_cluster_size = np.count_nonzero(fine_clusters == fine_clusters[fine_index])
        if not coarse_converged[index] or not fine_converged[fine_index]:
            status = GeneralEigenMatchStatus.NONCONVERGED
        elif coarse_cluster_size > 1 or fine_cluster_size > 1:
            status = GeneralEigenMatchStatus.AMBIGUOUS_CLUSTER
        elif (
            not np.isfinite(condition[index])
            or condition[index] > policy_.condition_limit
        ):
            status = GeneralEigenMatchStatus.ILL_CONDITIONED
        elif (
            not np.isfinite(residual[index])
            or residual[index] > policy_.residual_tolerance
            or distances[index] > policy_.chordal_tolerance
            or normalized[index] > policy_.normalized_drift_tolerance
        ):
            status = GeneralEigenMatchStatus.UNRESOLVED
        else:
            status = GeneralEigenMatchStatus.TRUSTED
        statuses[index] = int(status)

    trusted = statuses == int(GeneralEigenMatchStatus.TRUSTED)
    report_id = canonical_fingerprint(
        {
            "kind": "general-eigen-resolution-report",
            "coarse_problem": coarse.provenance.problem_id,
            "fine_problem": fine.provenance.problem_id,
            "coarse_prepared": coarse.provenance.prepared_id,
            "fine_prepared": fine.provenance.prepared_id,
            "policy": policy_.policy_id,
            "fine_indices": fine_indices.tolist(),
            "statuses": statuses.tolist(),
        }
    )
    return GeneralEigenResolutionReport(
        fine_indices=jnp.asarray(fine_indices),
        homogeneous_classes=jnp.asarray(coarse_class),
        chordal_distances=jnp.asarray(distances),
        local_separations=jnp.asarray(local),
        normalized_drifts=jnp.asarray(normalized),
        maximum_relative_residuals=jnp.asarray(residual),
        maximum_condition_estimates=jnp.asarray(condition),
        coarse_cluster_ids=jnp.asarray(coarse_clusters),
        fine_cluster_ids=jnp.asarray(fine_clusters),
        matched_fine_cluster_ids=jnp.asarray(matched_fine_clusters),
        statuses=jnp.asarray(statuses),
        trusted_mask=jnp.asarray(trusted),
        matched_mask=jnp.asarray(matched),
        matched_count=jnp.asarray(np.count_nonzero(matched), dtype=jnp.int32),
        trusted_count=jnp.asarray(np.count_nonzero(trusted), dtype=jnp.int32),
        report_id=report_id,
        coarse_problem_id=coarse.provenance.problem_id,
        fine_problem_id=fine.provenance.problem_id,
        policy_id=policy_.policy_id,
        host_only=True,
    )


def _homogeneous_classes(result: GeneralEigenSolveResult, /) -> np.ndarray:
    finite = np.asarray(result.finite_mask, dtype=bool)
    infinite = np.asarray(result.infinite_mask, dtype=bool)
    indeterminate = np.asarray(result.indeterminate_mask, dtype=bool)
    classes = np.full(finite.shape, 2, dtype=np.int32)
    classes[finite] = 0
    classes[infinite] = 1
    classes[indeterminate] = 2
    return classes


def _chordal_matrix(
    left_alpha: np.ndarray,
    left_beta: np.ndarray,
    right_alpha: np.ndarray,
    right_beta: np.ndarray,
    /,
) -> np.ndarray:
    numerator = np.abs(
        left_alpha[:, None] * right_beta[None, :]
        - left_beta[:, None] * right_alpha[None, :]
    )
    left_norm = np.sqrt(np.abs(left_alpha) ** 2 + np.abs(left_beta) ** 2)
    right_norm = np.sqrt(np.abs(right_alpha) ** 2 + np.abs(right_beta) ** 2)
    denominator = left_norm[:, None] * right_norm[None, :]
    return numerator / np.maximum(denominator, np.finfo(float).tiny)


def _local_separations(
    alpha: np.ndarray,
    beta: np.ndarray,
    classes: np.ndarray,
    /,
) -> np.ndarray:
    output = np.ones(alpha.shape, dtype=float)
    for class_value in (0, 1):
        indices = np.flatnonzero(classes == class_value)
        if indices.size <= 1:
            continue
        distances = _chordal_matrix(
            alpha[indices],
            beta[indices],
            alpha[indices],
            beta[indices],
        )
        np.fill_diagonal(distances, np.inf)
        output[indices] = np.min(distances, axis=1)
    output[classes == 2] = 0.0
    return output


def _cluster_ids(
    alpha: np.ndarray,
    beta: np.ndarray,
    classes: np.ndarray,
    tolerance: float,
    /,
) -> np.ndarray:
    groups = np.full(alpha.shape, -1, dtype=np.int32)
    next_group = 0
    for class_value in (0, 1):
        indices = np.flatnonzero(classes == class_value)
        if indices.size == 0:
            continue
        distances = _chordal_matrix(
            alpha[indices],
            beta[indices],
            alpha[indices],
            beta[indices],
        )
        remaining = set(range(indices.size))
        while remaining:
            seed = min(remaining)
            component = {seed}
            frontier = [seed]
            remaining.remove(seed)
            while frontier:
                current = frontier.pop()
                neighbors = tuple(
                    candidate
                    for candidate in sorted(remaining)
                    if distances[current, candidate] <= tolerance
                )
                for candidate in neighbors:
                    remaining.remove(candidate)
                    component.add(candidate)
                    frontier.append(candidate)
            groups[indices[sorted(component)]] = next_group
            next_group += 1
    return groups


__all__ = [
    "GeneralEigenMatchStatus",
    "GeneralEigenResolutionPolicy",
    "GeneralEigenResolutionReport",
    "compare_general_eigen_resolutions",
]
