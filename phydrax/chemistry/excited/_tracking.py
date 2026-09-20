#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Overlap-based root assignment and degenerate-subspace state tracking."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from scipy.optimize import linear_sum_assignment

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._manifold import ElectronicManifoldResult
from ._representation import (
    BiorthogonalStateRepresentation,
    CIStateRepresentation,
    RPAStateRepresentation,
    TDAStateRepresentation,
)


class StateTrackingResult(StrictModule, NonTrainableState):
    overlap_matrix: Array
    permutation: Array
    alignment: Array
    assigned_overlaps: Array
    subspace_singular_values: Array
    unitarity_residual: Array
    minimum_overlap: Array
    successful: Array
    previous_manifold_id: str = eqx.field(static=True)
    current_manifold_id: str = eqx.field(static=True)
    tracking_id: str = eqx.field(static=True)

    def __init__(
        self,
        overlap_matrix,
        permutation,
        alignment,
        assigned_overlaps,
        subspace_singular_values,
        successful,
        previous_manifold_id,
        current_manifold_id,
        /,
    ):
        overlap = jnp.asarray(overlap_matrix)
        permutation_ = jnp.asarray(permutation, dtype=jnp.int32)
        alignment_ = jnp.asarray(alignment, dtype=overlap.dtype)
        assigned = jnp.asarray(assigned_overlaps, dtype=overlap.real.dtype)
        singular = jnp.asarray(subspace_singular_values, dtype=overlap.real.dtype)
        roots = overlap.shape[0] if overlap.ndim == 2 else 0
        if (
            overlap.shape != (roots, roots)
            or permutation_.shape != (roots,)
            or alignment_.shape != (roots, roots)
            or assigned.shape != (roots,)
            or singular.shape != (roots,)
        ):
            raise ValueError("State-tracking arrays must align on the same root space.")
        unitary_residual = jnp.max(
            jnp.abs(jnp.conj(alignment_.T) @ alignment_ - jnp.eye(roots)),
            initial=0.0,
        )
        minimum = jnp.min(assigned, initial=jnp.inf)
        self.overlap_matrix = overlap
        self.permutation = permutation_
        self.alignment = alignment_
        self.assigned_overlaps = assigned
        self.subspace_singular_values = singular
        self.unitarity_residual = unitary_residual
        self.minimum_overlap = minimum
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(()) & (
            unitary_residual <= 1.0e-8
        )
        self.previous_manifold_id = str(previous_manifold_id)
        self.current_manifold_id = str(current_manifold_id)
        self.tracking_id = canonical_fingerprint(
            {
                "kind": "excited-state-tracking-result",
                "previous_manifold": self.previous_manifold_id,
                "current_manifold": self.current_manifold_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "overlap": np.asarray(overlap),
                        "permutation": np.asarray(permutation_),
                        "alignment": np.asarray(alignment_),
                        "assigned_overlaps": np.asarray(assigned),
                        "subspace_singular_values": np.asarray(singular),
                    }
                ),
            }
        )


def _left_right(representation):
    if isinstance(representation, TDAStateRepresentation):
        return representation.amplitudes, representation.amplitudes
    if isinstance(representation, CIStateRepresentation):
        return representation.coefficients, representation.coefficients
    if isinstance(representation, BiorthogonalStateRepresentation):
        return representation.left_amplitudes, representation.right_amplitudes
    if isinstance(representation, RPAStateRepresentation):
        right = jnp.concatenate(
            (representation.x_amplitudes, representation.y_amplitudes), axis=0
        )
        return representation.left_amplitudes, right
    raise TypeError("Unknown excited-state representation.")


def track_excited_states(
    previous: ElectronicManifoldResult,
    current: ElectronicManifoldResult,
    /,
    *,
    minimum_overlap: float = 0.25,
) -> StateTrackingResult:
    """Assign roots globally, then align each declared degenerate subspace."""
    if not isinstance(previous, ElectronicManifoldResult) or not isinstance(
        current, ElectronicManifoldResult
    ):
        raise TypeError("State tracking requires two electronic manifolds.")
    if type(previous.representation) is not type(current.representation):
        raise ValueError("State tracking cannot mix excited-state representations.")
    if previous.spin_sector != current.spin_sector:
        raise ValueError("State tracking cannot mix spin sectors.")
    left, _ = _left_right(previous.representation)
    _, right = _left_right(current.representation)
    if left.shape[0] != right.shape[0] or left.shape[1] != right.shape[1]:
        raise ValueError("State tracking requires equal root and representation spaces.")
    overlap = np.asarray(jnp.conj(left.T) @ right)
    roots = overlap.shape[0]
    rows, columns = linear_sum_assignment(-np.abs(overlap))
    permutation = columns[np.argsort(rows)]
    permutation_matrix = np.zeros((roots, roots), dtype=overlap.dtype)
    permutation_matrix[permutation, np.arange(roots)] = 1.0
    alignment = permutation_matrix.copy()
    singular_values = np.empty((roots,), dtype=np.asarray(overlap.real).dtype)
    for cluster in previous.clusters:
        indices = np.asarray(cluster, dtype=np.int64)
        assigned = permutation[indices]
        block = overlap[np.ix_(indices, assigned)]
        left_vectors, singular, right_vectors_h = np.linalg.svd(block)
        rotation = right_vectors_h.conj().T @ left_vectors.conj().T
        alignment[:, indices] = permutation_matrix[:, indices] @ rotation
        singular_values[indices] = singular
    aligned = overlap @ alignment
    assigned_overlaps = np.abs(np.diag(aligned))
    success = (
        bool(previous.successful)
        and bool(current.successful)
        and np.all(np.isfinite(overlap))
        and np.min(assigned_overlaps, initial=np.inf) >= float(minimum_overlap)
    )
    return StateTrackingResult(
        overlap,
        permutation,
        alignment,
        assigned_overlaps,
        singular_values,
        success,
        previous.result_id,
        current.result_id,
    )


__all__ = ["StateTrackingResult", "track_excited_states"]
