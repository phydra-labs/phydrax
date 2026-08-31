#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._diagram import PackedPersistenceDiagram
from ._persistence import FrozenPersistenceEvaluation


class PersistenceFeaturePolicy(StrictModule, NonTrainableState):
    """Degree, interval, and coordinate policy for JAX persistence features."""

    degree: int = eqx.field(static=True)
    include_essential: bool = eqx.field(static=True)
    coordinates: Literal["birth-death", "birth-persistence"] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        degree: int,
        /,
        *,
        include_essential: bool = False,
        coordinates: Literal["birth-death", "birth-persistence"] = "birth-persistence",
    ):
        if int(degree) < 0:
            raise ValueError("Persistence feature degree must be non-negative.")
        if coordinates not in ("birth-death", "birth-persistence"):
            raise ValueError("Unknown persistence feature coordinates.")
        self.degree = int(degree)
        self.include_essential = bool(include_essential)
        self.coordinates = coordinates
        self.policy_id = canonical_fingerprint(
            {
                "kind": "persistence-feature-policy",
                "degree": int(degree),
                "include_essential": bool(include_essential),
                "coordinates": coordinates,
            }
        )


def _packed_values(diagram: PackedPersistenceDiagram, policy: PersistenceFeaturePolicy):
    active = diagram.active_mask & (diagram.degrees == policy.degree)
    if not policy.include_essential:
        active = active & diagram.has_finite_death
    birth = diagram.birth_values
    death = diagram.death_values
    persistence = jnp.where(diagram.has_finite_death, jnp.abs(death - birth), 0)
    return active, birth, death, persistence


def total_persistence(
    diagram: PackedPersistenceDiagram,
    policy: PersistenceFeaturePolicy,
    /,
    *,
    exponent: float = 1.0,
) -> Array:
    exponent_ = float(exponent)
    if exponent_ <= 0.0:
        raise ValueError("Total-persistence exponent must be positive.")
    active, _, _, persistence = _packed_values(diagram, policy)
    return jnp.sum(jnp.where(active, persistence**exponent_, 0), axis=-1)


def betti_curve(
    diagram: PackedPersistenceDiagram,
    policy: PersistenceFeaturePolicy,
    thresholds: ArrayLike,
    /,
) -> Array:
    threshold = jnp.asarray(thresholds)
    if threshold.ndim != 1:
        raise ValueError("Betti-curve thresholds must be rank-1.")
    active, birth, death, _ = _packed_values(diagram, policy)
    alive = (birth[..., :, None] <= threshold) & (
        ~diagram.has_finite_death[..., :, None] | (death[..., :, None] > threshold)
    )
    return jnp.sum((active[..., :, None] & alive).astype(jnp.int32), axis=-2)


def persistence_image(
    diagram: PackedPersistenceDiagram,
    policy: PersistenceFeaturePolicy,
    birth_grid: ArrayLike,
    persistence_grid: ArrayLike,
    /,
    *,
    bandwidth: float,
) -> Array:
    births_axis = jnp.asarray(birth_grid)
    persistence_axis = jnp.asarray(persistence_grid)
    bandwidth_ = float(bandwidth)
    if births_axis.ndim != 1 or persistence_axis.ndim != 1:
        raise ValueError("Persistence-image grids must be rank-1.")
    if bandwidth_ <= 0.0:
        raise ValueError("Persistence-image bandwidth must be positive.")
    active, birth, death, persistence = _packed_values(diagram, policy)
    second = death if policy.coordinates == "birth-death" else persistence
    birth_delta = birth[..., :, None, None] - births_axis[None, :, None]
    second_delta = second[..., :, None, None] - persistence_axis[None, None, :]
    kernel = jnp.exp(-0.5 * (birth_delta**2 + second_delta**2) / bandwidth_**2)
    weights = jnp.where(active, persistence, 0)
    return jnp.sum(weights[..., :, None, None] * kernel, axis=-3)


def frozen_total_persistence(
    evaluation: FrozenPersistenceEvaluation,
    /,
    *,
    degree: int,
    exponent: float = 1.0,
) -> tuple[Array, Array]:
    exponent_ = float(exponent)
    if exponent_ <= 0.0:
        raise ValueError("Total-persistence exponent must be positive.")
    selected = (evaluation.degrees == int(degree)) & evaluation.has_finite_death
    persistence = jnp.abs(evaluation.death_values - evaluation.birth_values)
    value = jnp.sum(jnp.where(selected, persistence**exponent_, 0), axis=-1)
    return value, evaluation.ordering_valid


class PersistenceFeatureEvidence(StrictModule, NonTrainableState):
    """Identity and derivative-status evidence for one persistence feature."""

    value: Array
    ordering_valid: Array
    source_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        value: ArrayLike,
        ordering_valid: ArrayLike,
        /,
        *,
        source_id: str,
        policy_id: str,
    ):
        value_ = jnp.asarray(value)
        valid = jnp.asarray(ordering_valid, dtype=bool)
        self.value = value_
        self.ordering_valid = valid
        self.source_id = str(source_id)
        self.policy_id = str(policy_id)
        self.feature_id = canonical_fingerprint(
            {
                "kind": "persistence-feature-evidence",
                "source": self.source_id,
                "policy": self.policy_id,
                "value": array_tree_fingerprint(value_),
            }
        )


__all__ = [
    "PersistenceFeatureEvidence",
    "PersistenceFeaturePolicy",
    "betti_curve",
    "frozen_total_persistence",
    "persistence_image",
    "total_persistence",
]
