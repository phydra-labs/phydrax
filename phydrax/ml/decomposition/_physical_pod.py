#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import AbstractVectorSpace, LinearSubspace


class PhysicalPODResult(StrictModule, NonTrainableState):
    subspace: LinearSubspace
    offset: Array
    singular_values: Array
    retained_energy: Array
    tail_energy: Array
    orthogonality_defect: Array
    requested_energy: float = eqx.field(static=True)
    achieved_rank: int = eqx.field(static=True)
    target_met: bool = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class PhysicalPODPlan(StrictModule, NonTrainableState):
    """Method-of-snapshots POD in an arbitrary declared vector-space pairing."""

    maximum_rank: int = eqx.field(static=True)
    retained_energy: float = eqx.field(static=True)
    centered: bool = eqx.field(static=True)
    minimum_singular_value: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_rank: int,
        /,
        *,
        retained_energy: float = 1.0,
        centered: bool = True,
        minimum_singular_value: float = 0.0,
    ):
        rank = int(maximum_rank)
        energy = float(retained_energy)
        threshold = float(minimum_singular_value)
        if rank <= 0:
            raise ValueError("maximum_rank must be positive.")
        if not 0.0 < energy <= 1.0:
            raise ValueError("retained_energy must lie in (0, 1].")
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("minimum_singular_value must be finite and nonnegative.")
        self.maximum_rank = rank
        self.retained_energy = energy
        self.centered = bool(centered)
        self.minimum_singular_value = threshold
        self.plan_id = canonical_fingerprint(
            {
                "kind": "physical-pod-plan",
                "maximum_rank": rank,
                "retained_energy": energy,
                "centered": self.centered,
                "minimum_singular_value": threshold,
            }
        )

    def fit(
        self,
        space: AbstractVectorSpace,
        snapshots: ArrayLike,
        /,
        *,
        sample_weights: ArrayLike | None = None,
        source_artifact_ids: Sequence[str],
    ) -> PhysicalPODResult:
        if not isinstance(space, AbstractVectorSpace):
            raise TypeError("space must be an AbstractVectorSpace.")
        values = jnp.asarray(snapshots)
        if values.ndim != 2 or values.shape[1] != space.size:
            raise ValueError("snapshots must have shape (samples, space.size).")
        sample_count = int(values.shape[0])
        weights = (
            jnp.ones((sample_count,), dtype=values.real.dtype)
            if sample_weights is None
            else jnp.asarray(sample_weights, dtype=values.real.dtype)
        )
        if weights.shape != (sample_count,):
            raise ValueError("sample_weights must match the snapshot count.")
        host_weights = np.asarray(weights)
        if (
            np.any(~np.isfinite(host_weights))
            or np.any(host_weights < 0.0)
            or not np.any(host_weights > 0.0)
        ):
            raise ValueError("sample_weights must be finite, nonnegative, and nonzero.")
        source_ids = tuple(str(value) for value in source_artifact_ids)
        if not source_ids or any(not value for value in source_ids):
            raise ValueError("source_artifact_ids must be non-empty.")
        normalized = weights / jnp.sum(weights)
        offset = (
            jnp.sum(normalized[:, None] * values, axis=0)
            if self.centered
            else jnp.zeros((space.size,), dtype=values.dtype)
        )
        centered = values - offset
        vectors = jax.vmap(space.unflatten)(centered)
        root = jnp.sqrt(normalized)
        weighted = jax.tree.map(
            lambda leaf: root.reshape((sample_count,) + (1,) * (leaf.ndim - 1)) * leaf,
            vectors,
        )
        gram = jax.vmap(
            lambda left: jax.vmap(lambda right: space.inner(left, right))(weighted)
        )(weighted)
        gram = 0.5 * (gram + jnp.conj(gram.T))
        eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
        order = jnp.argsort(eigenvalues)[::-1]
        eigenvalues = jnp.maximum(jnp.real(eigenvalues[order]), 0.0)
        eigenvectors = eigenvectors[:, order]
        singular_values = jnp.sqrt(eigenvalues)
        energy = eigenvalues / jnp.maximum(
            jnp.sum(eigenvalues), jnp.finfo(eigenvalues.dtype).tiny
        )
        cumulative = jnp.cumsum(energy)
        energy_rank = int(
            np.searchsorted(np.asarray(cumulative), self.retained_energy) + 1
        )
        numerical_rank = int(
            np.sum(np.asarray(singular_values) > self.minimum_singular_value)
        )
        rank = min(self.maximum_rank, sample_count, max(numerical_rank, 1), energy_rank)
        selected_values = singular_values[:rank]
        selected_vectors = eigenvectors[:, :rank]
        coefficients = root[:, None] * selected_vectors / selected_values[None, :]
        basis = jnp.conj(coefficients.T) @ centered
        basis = jnp.conj(basis.T)
        subspace = LinearSubspace(
            space,
            basis,
            orthonormal=True,
            subspace_id=canonical_fingerprint(
                {
                    "kind": "physical-pod-subspace",
                    "space": space.space_id,
                    "plan": self.plan_id,
                    "sources": list(source_ids),
                    "content": array_tree_fingerprint(basis)["sha256"],
                }
            ),
        )
        gram_basis = jax.vmap(
            lambda left: jax.vmap(
                lambda right: space.inner(space.unflatten(left), space.unflatten(right)),
                in_axes=1,
            )(basis),
            in_axes=1,
        )(basis)
        defect = jnp.max(jnp.abs(gram_basis - jnp.eye(rank, dtype=gram_basis.dtype)))
        retained = jnp.sum(eigenvalues[:rank]) / jnp.maximum(
            jnp.sum(eigenvalues), jnp.finfo(eigenvalues.dtype).tiny
        )
        tail = jnp.maximum(1.0 - retained, 0.0)
        target_met = bool(
            np.asarray(retained) + 32.0 * np.finfo(host_weights.dtype).eps
            >= self.retained_energy
        )
        result_id = canonical_fingerprint(
            {
                "kind": "physical-pod-result",
                "subspace": subspace.subspace_id,
                "plan": self.plan_id,
                "sources": list(source_ids),
                "rank": rank,
                "target_met": target_met,
            }
        )
        return PhysicalPODResult(
            subspace,
            offset,
            singular_values,
            retained,
            tail,
            defect,
            self.retained_energy,
            rank,
            target_met,
            result_id,
        )


__all__ = ["PhysicalPODPlan", "PhysicalPODResult"]
