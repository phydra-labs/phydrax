#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DenseLinearOperator, FactorizationPolicy, factorize
from ...linalg.eigen import GeneralEigenSolveResult
from ...linalg.eigen._resolution import (
    compare_general_eigen_resolutions,
    GeneralEigenMatchStatus,
    GeneralEigenResolutionPolicy,
    GeneralEigenResolutionReport,
)
from ._space import TensorSpectralDiscretization
from ._transfer import PreparedSpectralModalTransfer


class SpectralEigenResolutionPolicy(StrictModule, NonTrainableState):
    """General homogeneous and physical eigenspace comparison tolerances."""

    general: GeneralEigenResolutionPolicy
    subspace_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        general: GeneralEigenResolutionPolicy | None = None,
        /,
        *,
        subspace_tolerance: float = 1e-5,
    ):
        general_ = GeneralEigenResolutionPolicy() if general is None else general
        tolerance = float(subspace_tolerance)
        if not isinstance(general_, GeneralEigenResolutionPolicy):
            raise TypeError("general must be a GeneralEigenResolutionPolicy or None.")
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("subspace_tolerance must be finite and non-negative.")
        self.general = general_
        self.subspace_tolerance = tolerance
        self.policy_id = canonical_fingerprint(
            {
                "kind": "spectral-eigen-resolution-policy",
                "general": general_.policy_id,
                "subspace_tolerance": tolerance,
            }
        )


class SpectralEigenResolutionReport(StrictModule, NonTrainableState):
    """Homogeneous matches strengthened by transferred physical subspace evidence."""

    general: GeneralEigenResolutionReport
    subspace_errors: Array
    statuses: Array
    trusted_mask: Array
    trusted_count: Array
    report_id: str = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)


def compare_spectral_eigen_resolutions(
    coarse_result: GeneralEigenSolveResult,
    fine_result: GeneralEigenSolveResult,
    coarse: TensorSpectralDiscretization,
    fine: TensorSpectralDiscretization,
    transfer: PreparedSpectralModalTransfer,
    /,
    *,
    policy: SpectralEigenResolutionPolicy | None = None,
) -> SpectralEigenResolutionReport:
    """Compare modal eigenspaces after exact coarse-to-fine transfer."""
    if not isinstance(coarse_result, GeneralEigenSolveResult) or not isinstance(
        fine_result, GeneralEigenSolveResult
    ):
        raise TypeError("coarse_result and fine_result must be general eigen results.")
    if not isinstance(coarse, TensorSpectralDiscretization) or not isinstance(
        fine, TensorSpectralDiscretization
    ):
        raise TypeError("coarse and fine must be tensor spectral discretizations.")
    if not isinstance(transfer, PreparedSpectralModalTransfer):
        raise TypeError("transfer must be a PreparedSpectralModalTransfer.")
    if (
        transfer.plan.source.prepared_id != coarse.prepared_id
        or transfer.plan.target.prepared_id != fine.prepared_id
    ):
        raise ValueError("Spectral eigen transfer does not bind the supplied spaces.")
    if not transfer.report.lossless:
        raise ValueError("Spectral eigenspace evidence requires a lossless transfer.")
    if (
        coarse_result.provenance.source_space_id
        != coarse.modal_space.vector_space.space_id
    ):
        raise ValueError("Coarse eigen result does not use the coarse modal space.")
    if fine_result.provenance.source_space_id != fine.modal_space.vector_space.space_id:
        raise ValueError("Fine eigen result does not use the fine modal space.")
    policy_ = SpectralEigenResolutionPolicy() if policy is None else policy
    if not isinstance(policy_, SpectralEigenResolutionPolicy):
        raise TypeError("policy must be a SpectralEigenResolutionPolicy or None.")

    general = compare_general_eigen_resolutions(
        coarse_result,
        fine_result,
        policy=policy_.general,
    )
    coarse_coordinates = jnp.asarray(coarse_result.right_eigenvector_coordinates)
    fine_coordinates = jnp.asarray(fine_result.right_eigenvector_coordinates)
    transferred_columns = tuple(
        transfer(coarse_coordinates[:, index].reshape(coarse.modal_shape)).reshape((-1,))
        for index in range(coarse_coordinates.shape[1])
    )
    transferred = jnp.stack(transferred_columns, axis=1)
    coarse_physical = fine.reconstruct(
        transferred.reshape(fine.modal_shape + (transferred.shape[1],)),
        real_output=False,
    ).reshape((fine.num_points, transferred.shape[1]))
    fine_physical = fine.reconstruct(
        fine_coordinates.reshape(fine.modal_shape + (fine_coordinates.shape[1],)),
        real_output=False,
    ).reshape((fine.num_points, fine_coordinates.shape[1]))
    square_root_weights = jnp.sqrt(fine.quadrature_weights.reshape((-1, 1)))
    coarse_weighted = np.asarray(square_root_weights * coarse_physical)
    fine_weighted = np.asarray(square_root_weights * fine_physical)

    coarse_clusters = np.asarray(general.coarse_cluster_ids, dtype=np.int32)
    fine_cluster_labels = np.asarray(general.fine_cluster_ids, dtype=np.int32)
    matched_fine_clusters = np.asarray(
        general.matched_fine_cluster_ids,
        dtype=np.int32,
    )
    fine_indices = np.asarray(general.fine_indices, dtype=np.int32)
    matched = np.asarray(general.matched_mask, dtype=bool)
    errors = np.full((coarse_coordinates.shape[1],), np.inf, dtype=float)
    for cluster in sorted(set(coarse_clusters[coarse_clusters >= 0].tolist())):
        coarse_group = np.flatnonzero((coarse_clusters == cluster) & matched)
        if coarse_group.size == 0:
            continue
        target_clusters = sorted(set(matched_fine_clusters[coarse_group].tolist()))
        target_clusters = [value for value in target_clusters if value >= 0]
        if len(target_clusters) != 1:
            continue
        fine_group = np.flatnonzero(fine_cluster_labels == target_clusters[0])
        error = _bidirectional_subspace_error(
            coarse_weighted[:, coarse_group],
            fine_weighted[:, fine_group],
        )
        errors[coarse_group] = error

    statuses = np.asarray(general.statuses, dtype=np.int32).copy()
    for index in np.flatnonzero(matched):
        status = GeneralEigenMatchStatus(int(statuses[index]))
        if status in (
            GeneralEigenMatchStatus.TRUSTED,
            GeneralEigenMatchStatus.AMBIGUOUS_CLUSTER,
        ):
            generally_resolved = (
                float(general.chordal_distances[index])
                <= policy_.general.chordal_tolerance
                and float(general.normalized_drifts[index])
                <= policy_.general.normalized_drift_tolerance
                and float(general.maximum_relative_residuals[index])
                <= policy_.general.residual_tolerance
                and float(general.maximum_condition_estimates[index])
                <= policy_.general.condition_limit
            )
            if (
                generally_resolved
                and np.isfinite(errors[index])
                and errors[index] <= policy_.subspace_tolerance
            ):
                statuses[index] = int(GeneralEigenMatchStatus.TRUSTED)
            else:
                statuses[index] = int(GeneralEigenMatchStatus.UNRESOLVED)
    trusted = statuses == int(GeneralEigenMatchStatus.TRUSTED)
    report_id = canonical_fingerprint(
        {
            "kind": "spectral-eigen-resolution-report",
            "general": general.report_id,
            "coarse": coarse.prepared_id,
            "fine": fine.prepared_id,
            "transfer": transfer.prepared_id,
            "policy": policy_.policy_id,
            "statuses": statuses.tolist(),
        }
    )
    return SpectralEigenResolutionReport(
        general=general,
        subspace_errors=jnp.asarray(errors),
        statuses=jnp.asarray(statuses),
        trusted_mask=jnp.asarray(trusted),
        trusted_count=jnp.asarray(np.count_nonzero(trusted), dtype=jnp.int32),
        report_id=report_id,
        transfer_id=transfer.prepared_id,
        policy_id=policy_.policy_id,
    )


def _bidirectional_subspace_error(
    left: np.ndarray,
    right: np.ndarray,
    /,
) -> float:
    if left.shape[1] == 0 or right.shape[1] == 0:
        return float("inf")
    return max(
        _relative_projection_error(left, right), _relative_projection_error(right, left)
    )


def _relative_projection_error(source: np.ndarray, target: np.ndarray, /) -> float:
    factorization = factorize(
        DenseLinearOperator(jnp.asarray(target)),
        FactorizationPolicy("svd"),
    )
    errors = []
    for column in range(source.shape[1]):
        vector = jnp.asarray(source[:, column])
        result = factorization.solve(vector)
        if not bool(result.successful):
            return float("inf")
        residual = vector - jnp.asarray(target) @ result.value
        numerator = float(np.sqrt(np.sum(np.abs(np.asarray(residual)) ** 2)))
        denominator = float(np.sqrt(np.sum(np.abs(source[:, column]) ** 2)))
        errors.append(numerator / max(denominator, np.finfo(float).tiny))
    return max(errors, default=0.0)


__all__ = [
    "SpectralEigenResolutionPolicy",
    "SpectralEigenResolutionReport",
    "compare_spectral_eigen_resolutions",
]
