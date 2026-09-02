#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from numbers import Integral

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._assignment_core import hungarian_assignment_one
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule


class HermitianEigenspaceTrackingStatus(IntEnum):
    """Outcome of one Hermitian eigenspace correspondence calculation."""

    SUCCESS = 0
    NONFINITE = 1
    ASSIGNMENT_FAILED = 2
    AMBIGUOUS = 3
    CLUSTER_MISMATCH = 4
    ORTHONORMALITY_FAILED = 5


class HermitianEigenspaceTrackingPolicy(StrictModule):
    """Fixed tolerances for one-to-one Hermitian eigenspace tracking."""

    degeneracy_absolute: float = eqx.field(static=True)
    degeneracy_relative: float = eqx.field(static=True)
    minimum_overlap: float = eqx.field(static=True)
    minimum_assignment_margin: float = eqx.field(static=True)
    orthogonality_tolerance: float = eqx.field(static=True)
    maximum_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        degeneracy_absolute: float = 1e-10,
        degeneracy_relative: float = 1e-8,
        minimum_overlap: float = 0.5,
        minimum_assignment_margin: float = 1e-6,
        orthogonality_tolerance: float = 1e-8,
        maximum_dimension: int = 4096,
    ):
        values = {
            "degeneracy_absolute": degeneracy_absolute,
            "degeneracy_relative": degeneracy_relative,
            "minimum_overlap": minimum_overlap,
            "minimum_assignment_margin": minimum_assignment_margin,
            "orthogonality_tolerance": orthogonality_tolerance,
        }
        converted = {name: float(value) for name, value in values.items()}
        if any(not isfinite(value) or value < 0.0 for value in converted.values()):
            raise ValueError("Tracking tolerances must be finite and non-negative.")
        if converted["minimum_overlap"] > 1.0:
            raise ValueError("minimum_overlap must not exceed one.")
        if isinstance(maximum_dimension, bool) or not isinstance(
            maximum_dimension, Integral
        ):
            raise TypeError("maximum_dimension must be a positive integer.")
        if int(maximum_dimension) <= 0:
            raise ValueError("maximum_dimension must be positive.")
        self.degeneracy_absolute = converted["degeneracy_absolute"]
        self.degeneracy_relative = converted["degeneracy_relative"]
        self.minimum_overlap = converted["minimum_overlap"]
        self.minimum_assignment_margin = converted["minimum_assignment_margin"]
        self.orthogonality_tolerance = converted["orthogonality_tolerance"]
        self.maximum_dimension = int(maximum_dimension)

    def degeneracy_threshold(self, values: ArrayLike, /) -> Array:
        values_ = jnp.asarray(values)
        scale = jnp.maximum(jnp.max(jnp.abs(values_)), 1.0)
        return self.degeneracy_absolute + self.degeneracy_relative * scale


class HermitianEigenspaceTrackingPlan(StrictModule):
    """Reference clustering and identity for repeated numerical tracking."""

    reference_values: Array
    policy: HermitianEigenspaceTrackingPolicy
    clusters: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class HermitianEigenspaceTrackingDiagnostics(StrictModule):
    """Independent overlap, cluster, and orthonormality evidence."""

    overlap_matrix: Array
    selected_overlaps: Array
    cluster_minimum_overlaps: Array
    assignment_margin: Array
    reference_orthogonality_residual: Array
    candidate_orthogonality_residual: Array
    aligned_orthogonality_residual: Array
    cluster_consistent: Array
    assignment_valid: Array
    finite: Array
    valid: Array


class HermitianEigenspaceTrackingResult(StrictModule):
    """Candidate eigensystem aligned into one reference ordering and gauge."""

    values: Array
    vectors: Array
    assignment: Array
    status: Array
    diagnostics: HermitianEigenspaceTrackingDiagnostics
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(HermitianEigenspaceTrackingStatus.SUCCESS)


def _reference_clusters(
    values: np.ndarray,
    policy: HermitianEigenspaceTrackingPolicy,
    /,
) -> tuple[tuple[int, ...], ...]:
    scale = max(float(np.max(np.abs(values), initial=0.0)), 1.0)
    threshold = policy.degeneracy_absolute + policy.degeneracy_relative * scale
    clusters: list[tuple[int, ...]] = []
    start = 0
    for index in range(1, values.size):
        if abs(float(values[index] - values[index - 1])) > threshold:
            clusters.append(tuple(range(start, index)))
            start = index
    clusters.append(tuple(range(start, values.size)))
    return tuple(clusters)


def plan_hermitian_eigenspace_tracking(
    reference_values: ArrayLike,
    /,
    *,
    policy: HermitianEigenspaceTrackingPolicy | None = None,
) -> HermitianEigenspaceTrackingPlan:
    """Plan static degeneracy clusters from one concrete reference spectrum."""

    values = jnp.asarray(reference_values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("reference_values must be one nonempty vector.")
    if jnp.issubdtype(values.dtype, jnp.complexfloating):
        raise TypeError("reference_values must be real.")
    if isinstance(values, jax_core.Tracer):
        raise TypeError("Eigenspace tracking must be planned from concrete values.")
    selected = HermitianEigenspaceTrackingPolicy() if policy is None else policy
    if not isinstance(selected, HermitianEigenspaceTrackingPolicy):
        raise TypeError("policy must be a HermitianEigenspaceTrackingPolicy or None.")
    if int(values.size) > selected.maximum_dimension:
        raise ValueError("Reference dimension exceeds maximum_dimension.")
    host_values = np.asarray(values)
    if not np.all(np.isfinite(host_values)):
        raise ValueError("reference_values must be finite.")
    clusters = _reference_clusters(host_values, selected)
    plan_id = canonical_fingerprint(
        {
            "kind": "hermitian-eigenspace-tracking",
            "reference_values": array_tree_fingerprint(values),
            "clusters": [list(cluster) for cluster in clusters],
            "policy": {
                "degeneracy_absolute": selected.degeneracy_absolute,
                "degeneracy_relative": selected.degeneracy_relative,
                "minimum_overlap": selected.minimum_overlap,
                "minimum_assignment_margin": selected.minimum_assignment_margin,
                "orthogonality_tolerance": selected.orthogonality_tolerance,
                "maximum_dimension": selected.maximum_dimension,
            },
        }
    )
    return HermitianEigenspaceTrackingPlan(
        values,
        selected,
        clusters,
        int(values.size),
        plan_id,
    )


def _orthogonality_residual(vectors: Array, /) -> Array:
    dimension = vectors.shape[1]
    gram = ein.contract("ai,aj->ij", jnp.conj(vectors), vectors)
    return jnp.max(jnp.abs(gram - jnp.eye(dimension, dtype=gram.dtype)))


def _cluster_consistency(
    values: Array,
    assignment: Array,
    clusters: tuple[tuple[int, ...], ...],
    threshold: Array,
    /,
) -> Array:
    consistent = jnp.asarray(True)
    count = assignment.shape[0]
    for cluster in clusters:
        selected = values[assignment[jnp.asarray(cluster, dtype=jnp.int32)]]
        if len(cluster) > 1:
            consistent = consistent & (
                (jnp.max(selected) - jnp.min(selected)) <= threshold
            )
        else:
            index = cluster[0]
            distances = jnp.abs(values - selected[0])
            distances = distances.at[assignment[index]].set(jnp.inf)
            consistent = consistent & (jnp.min(distances) > threshold)
    return consistent & (count <= values.size)


def track_hermitian_eigenspaces(
    plan: HermitianEigenspaceTrackingPlan,
    reference_vectors: ArrayLike,
    candidate_values: ArrayLike,
    candidate_vectors: ArrayLike,
    /,
) -> HermitianEigenspaceTrackingResult:
    """Align candidate Hermitian eigenvectors to a planned reference eigensystem."""

    if not isinstance(plan, HermitianEigenspaceTrackingPlan):
        raise TypeError("plan must be a HermitianEigenspaceTrackingPlan.")
    reference = jnp.asarray(reference_vectors)
    values = jnp.asarray(candidate_values)
    candidate = jnp.asarray(candidate_vectors)
    if reference.ndim != 2 or candidate.ndim != 2 or values.ndim != 1:
        raise ValueError("Eigenvector inputs must be matrices and values one vector.")
    if reference.shape[1] != plan.dimension:
        raise ValueError("reference_vectors column count must match the plan dimension.")
    if candidate.shape[0] != reference.shape[0] or candidate.shape[1] != values.size:
        raise ValueError("Candidate values and vectors have incompatible shapes.")
    if candidate.shape[1] < plan.dimension:
        raise ValueError("Candidate eigensystem has too few vectors.")
    if candidate.shape[1] > plan.policy.maximum_dimension:
        raise ValueError("Candidate dimension exceeds maximum_dimension.")
    if jnp.issubdtype(values.dtype, jnp.complexfloating):
        raise TypeError("candidate_values must be real.")

    overlap = ein.contract("ai,aj->ij", jnp.conj(reference), candidate)
    overlap_weights = jnp.real(overlap * jnp.conj(overlap))
    assignment, _, _, assignment_solved, _ = hungarian_assignment_one(
        -jax.lax.stop_gradient(overlap_weights),
        jnp.ones_like(overlap_weights, dtype=bool),
    )
    safe_assignment = jnp.clip(assignment, 0, candidate.shape[1] - 1)
    selected_values = values[safe_assignment]
    selected_vectors = candidate[:, safe_assignment]
    selected_overlaps = overlap_weights[
        jnp.arange(plan.dimension, dtype=jnp.int32), safe_assignment
    ]

    aligned = jnp.zeros_like(selected_vectors)
    cluster_overlaps: list[Array] = []
    cluster_margins: list[Array] = []
    for cluster in plan.clusters:
        indices = jnp.asarray(cluster, dtype=jnp.int32)
        reference_cluster = reference[:, indices]
        candidate_cluster = selected_vectors[:, indices]
        cross = ein.contract("ai,aj->ij", jnp.conj(reference_cluster), candidate_cluster)
        left, singular_values, right_adjoint = jnp.linalg.svd(
            cross,
            full_matrices=False,
        )
        rotation = jnp.conj(right_adjoint.T) @ jnp.conj(left.T)
        aligned_cluster = candidate_cluster @ rotation
        aligned = aligned.at[:, indices].set(aligned_cluster)
        cluster_overlaps.append(jnp.min(singular_values) ** 2)
        candidate_indices = safe_assignment[indices]
        candidate_index = jnp.arange(candidate.shape[1], dtype=jnp.int32)
        in_cluster = jnp.any(
            candidate_index[None, :] == candidate_indices[:, None],
            axis=0,
        )
        reference_weights = overlap_weights[indices]
        selected_strength = jnp.sum(
            jnp.where(in_cluster[None, :], reference_weights, 0.0),
            axis=1,
        )
        outside_maximum = jnp.where(
            jnp.any(~in_cluster),
            jnp.max(
                jnp.where(~in_cluster[None, :], reference_weights, -jnp.inf),
                axis=1,
            ),
            jnp.zeros((len(cluster),), dtype=overlap_weights.dtype),
        )
        cluster_margins.append(jnp.min(selected_strength - outside_maximum))

    cluster_minimum_overlaps = jnp.stack(cluster_overlaps)
    assignment_margin = jnp.min(jnp.stack(cluster_margins))
    reference_residual = _orthogonality_residual(reference)
    candidate_residual = _orthogonality_residual(candidate)
    aligned_residual = _orthogonality_residual(aligned)
    threshold = plan.policy.degeneracy_threshold(values)
    cluster_consistent = _cluster_consistency(
        values,
        safe_assignment,
        plan.clusters,
        threshold,
    )
    finite = (
        jnp.all(jnp.isfinite(reference))
        & jnp.all(jnp.isfinite(values))
        & jnp.all(jnp.isfinite(candidate))
        & jnp.all(jnp.isfinite(aligned))
    )
    orthonormal = (reference_residual <= plan.policy.orthogonality_tolerance) & (
        candidate_residual <= plan.policy.orthogonality_tolerance
    )
    assignment_valid = assignment_solved & jnp.all(assignment >= 0)
    sufficient_overlap = jnp.min(cluster_minimum_overlaps) >= plan.policy.minimum_overlap
    separated = assignment_margin >= plan.policy.minimum_assignment_margin
    valid = (
        finite
        & orthonormal
        & assignment_valid
        & cluster_consistent
        & sufficient_overlap
        & separated
    )
    status = jnp.where(
        ~finite,
        int(HermitianEigenspaceTrackingStatus.NONFINITE),
        jnp.where(
            ~assignment_valid,
            int(HermitianEigenspaceTrackingStatus.ASSIGNMENT_FAILED),
            jnp.where(
                ~orthonormal,
                int(HermitianEigenspaceTrackingStatus.ORTHONORMALITY_FAILED),
                jnp.where(
                    ~cluster_consistent,
                    int(HermitianEigenspaceTrackingStatus.CLUSTER_MISMATCH),
                    jnp.where(
                        ~(sufficient_overlap & separated),
                        int(HermitianEigenspaceTrackingStatus.AMBIGUOUS),
                        int(HermitianEigenspaceTrackingStatus.SUCCESS),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    diagnostics = HermitianEigenspaceTrackingDiagnostics(
        overlap_weights,
        selected_overlaps,
        cluster_minimum_overlaps,
        assignment_margin,
        reference_residual,
        candidate_residual,
        aligned_residual,
        cluster_consistent,
        assignment_valid,
        finite,
        valid,
    )
    return HermitianEigenspaceTrackingResult(
        selected_values,
        aligned,
        assignment,
        status,
        diagnostics,
        plan.plan_id,
    )


__all__ = [
    "HermitianEigenspaceTrackingDiagnostics",
    "HermitianEigenspaceTrackingPlan",
    "HermitianEigenspaceTrackingPolicy",
    "HermitianEigenspaceTrackingResult",
    "HermitianEigenspaceTrackingStatus",
    "plan_hermitian_eigenspace_tracking",
    "track_hermitian_eigenspaces",
]
