#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

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
    r"""Method-of-snapshots POD in an arbitrary declared vector-space pairing.

    `fit` forms the weighted snapshot Gram matrix $G_{ij} = \langle w_i, w_j\rangle$
    of the $N$ (centered, weighted) snapshots in a space of $m$ coordinates and
    takes its eigenpairs, so the singular values are $\sigma_i = \sqrt{\lambda_i}$.
    The computed eigenvalues carry an absolute error of about
    $\tau\lambda_1$ with $\tau = (N + \sqrt{m})\,\varepsilon$ ($\varepsilon$ the
    machine epsilon of the snapshot dtype): $N\varepsilon\lVert G\rVert$ from
    the backward-stable symmetric eigensolver and $\sqrt{m}\,\varepsilon$ from
    the rounding of length-$m$ inner products (the probabilistic bound of
    Higham and Mary). Squaring the singular values therefore resolves them only
    to about $\sqrt{\tau}\,\sigma_1$, not $\varepsilon\sigma_1$. A direction with
    $\lambda_i \le \tau\lambda_1$, i.e. $\sigma_i \le \sqrt{\tau}\,\sigma_1$, is
    indistinguishable from roundoff: its eigenvector is arbitrary and its
    $1/\sigma_i$ scaling amplifies noise, so it never enters the basis. Data
    whose trailing singular values matter below that floor needs an SVD of the
    snapshots in orthonormal coordinates instead.

    The returned rank is the smallest of:

    - ``maximum_rank``;
    - the resolved directions with $\sigma_i >$ ``minimum_singular_value`` (an
      absolute threshold declared by the caller, applied on top of the relative
      resolution floor and never below it; at least one direction is kept);
    - the energy rank: the smallest $k$ whose resolved tail
      $\sum_{k \le i < r} \lambda_i / \sum_i \lambda_i$ is at most
      ``1 - retained_energy``, where $r$ is the number of resolved directions.
      With the default ``retained_energy=1.0`` this is exactly $r$: all
      resolved directions and no roundoff direction, a decision that does not
      depend on the last ulp of the cumulative energy.

    The result's ``retained_energy``/``tail_energy`` are fractions of the total
    Gram energy (the tail includes energy below the resolution floor), and
    ``target_met`` states that the returned rank reaches the energy rank.
    """

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
    ) -> None:
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
        sample_count = values.shape[0]
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
        host_eigenvalues = np.asarray(eigenvalues)
        total = max(
            float(np.sum(host_eigenvalues)), np.finfo(host_eigenvalues.dtype).tiny
        )
        # Gram eigenvalues are accurate to (N + sqrt(m)) eps lambda_1 absolutely;
        # below that floor a direction is roundoff (see the class docstring).
        tolerance = (sample_count + np.sqrt(space.size)) * np.finfo(
            host_eigenvalues.dtype
        ).eps
        resolved = int(np.sum(host_eigenvalues > tolerance * host_eigenvalues[0]))
        # Energy left out by keeping the leading k resolved directions, summed
        # from the smallest so an all-resolved tail is exactly zero.
        resolved_tail = (
            np.concatenate(
                (np.cumsum(host_eigenvalues[:resolved][::-1])[::-1], np.zeros((1,)))
            )
            / total
        )
        energy_rank = max(int(np.argmax(resolved_tail <= 1.0 - self.retained_energy)), 1)
        numerical_rank = int(
            np.sum(np.asarray(singular_values)[:resolved] > self.minimum_singular_value)
        )
        rank = min(self.maximum_rank, max(numerical_rank, 1), energy_rank)
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
        retained = jnp.sum(eigenvalues[:rank]) / total
        tail = jnp.sum(eigenvalues[rank:]) / total
        target_met = rank >= energy_rank
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
