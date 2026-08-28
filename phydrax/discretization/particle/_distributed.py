#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._pairwise import ParticleBox
from ._precision import ParticlePrecisionPolicy


ParticleBackendKind: TypeAlias = Literal["pure-jax", "pallas", "triton"]
ParticleDeterminism: TypeAlias = Literal["fast", "deterministic", "compensated"]


class ParticleBackendPolicy(StrictModule, NonTrainableState):
    backend: ParticleBackendKind = eqx.field(static=True)
    determinism: ParticleDeterminism = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        backend: ParticleBackendKind = "pure-jax",
        /,
        *,
        determinism: ParticleDeterminism = "deterministic",
    ):
        if backend not in ("pure-jax", "pallas", "triton"):
            raise ValueError("Unknown particle backend.")
        if determinism not in ("fast", "deterministic", "compensated"):
            raise ValueError("Unknown particle determinism mode.")
        self.backend = backend
        self.determinism = determinism
        self.policy_id = canonical_fingerprint(
            {
                "kind": "particle-backend-policy",
                "backend": backend,
                "determinism": determinism,
            }
        )


class ParticleKernelRequestPlan(StrictModule, NonTrainableState):
    distance: bool = eqx.field(static=True)
    direction: bool = eqx.field(static=True)
    kernel_value: bool = eqx.field(static=True)
    kernel_gradient: bool = eqx.field(static=True)
    smoothing_derivative: bool = eqx.field(static=True)
    materialize: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        distance: bool = True,
        direction: bool = False,
        kernel_value: bool = False,
        kernel_gradient: bool = True,
        smoothing_derivative: bool = False,
        materialize: bool = False,
    ):
        self.distance = bool(distance)
        self.direction = bool(direction)
        self.kernel_value = bool(kernel_value)
        self.kernel_gradient = bool(kernel_gradient)
        self.smoothing_derivative = bool(smoothing_derivative)
        self.materialize = bool(materialize)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "particle-kernel-request",
                "distance": distance,
                "direction": direction,
                "kernel_value": kernel_value,
                "kernel_gradient": kernel_gradient,
                "smoothing_derivative": smoothing_derivative,
                "materialize": materialize,
            }
        )


class MixedPrecisionCertification(StrictModule):
    finite: Array
    relative_error: Array
    tolerance: Array
    successful: Array
    evidence_id: str = eqx.field(static=True)


def certify_particle_precision(
    reference: ArrayLike,
    candidate: ArrayLike,
    precision: ParticlePrecisionPolicy,
    /,
    *,
    tolerance: float,
) -> MixedPrecisionCertification:
    reference_ = precision.certification(reference)
    candidate_ = precision.certification(candidate)
    scale = jnp.maximum(jnp.max(jnp.abs(reference_)), jnp.finfo(reference_.dtype).tiny)
    error = jnp.max(jnp.abs(candidate_ - reference_)) / scale
    finite = jnp.all(jnp.isfinite(candidate_))
    successful = finite & (error <= tolerance)
    evidence_id = canonical_fingerprint(
        {
            "kind": "particle-mixed-precision-certification",
            "precision": precision.policy_id,
            "tolerance": tolerance,
            "shape": list(reference_.shape),
        }
    )
    return MixedPrecisionCertification(
        finite, error, jnp.asarray(tolerance, reference_.dtype), successful, evidence_id
    )


class ParticleDomainDecompositionPlan(StrictModule, NonTrainableState):
    partitions: int = eqx.field(static=True)
    halo_radius: float = eqx.field(static=True)
    box: ParticleBox
    plan_id: str = eqx.field(static=True)

    def __init__(self, partitions: int, halo_radius: float, box: ParticleBox, /):
        if partitions <= 0 or halo_radius <= 0.0:
            raise ValueError("Particle decomposition parameters are invalid.")
        self.partitions = int(partitions)
        self.halo_radius = float(halo_radius)
        self.box = box
        self.plan_id = canonical_fingerprint(
            {
                "kind": "particle-domain-decomposition",
                "partitions": partitions,
                "halo_radius": halo_radius,
                "box": box.box_id,
            }
        )


class ParticleHaloState(StrictModule, NonTrainableState):
    owner: Array
    owned_mask: Array
    halo_mask: Array
    local_mask: Array
    migration_count: Array
    halo_count: Array
    successful: Array


def prepare_particle_halos(
    plan: ParticleDomainDecompositionPlan,
    position: ArrayLike,
    active_mask: ArrayLike,
    /,
) -> ParticleHaloState:
    position_ = jnp.asarray(position)
    active = jnp.asarray(active_mask, bool)
    relative = (position_[:, 0] - plan.box.lower[0]) / plan.box.lengths[0]
    owner = jnp.clip(
        jnp.floor(relative * plan.partitions).astype(jnp.int32), 0, plan.partitions - 1
    )
    owned = jnp.arange(plan.partitions)[:, None] == owner[None, :]
    edges = (
        plan.box.lower[0]
        + plan.box.lengths[0] * jnp.arange(plan.partitions + 1) / plan.partitions
    )
    distance_to_left = jnp.abs(position_[:, 0][None, :] - edges[:-1, None])
    distance_to_right = jnp.abs(position_[:, 0][None, :] - edges[1:, None])
    halo = (
        (distance_to_left <= plan.halo_radius) | (distance_to_right <= plan.halo_radius)
    ) & ~owned
    halo = halo & active[None, :]
    owned = owned & active[None, :]
    local = owned | halo
    return ParticleHaloState(
        owner,
        owned,
        halo,
        local,
        jnp.zeros((), jnp.int32),
        jnp.sum(halo, dtype=jnp.int32),
        jnp.all(jnp.isfinite(position_)),
    )


def halo_update(values: ArrayLike, halo: ParticleHaloState, /) -> Array:
    value = jnp.asarray(values)
    return jnp.where(
        halo.local_mask.reshape(halo.local_mask.shape + (1,) * (value.ndim - 1)),
        value[None, ...],
        0.0,
    )


def halo_sum(local_values: ArrayLike, halo: ParticleHaloState, /) -> Array:
    local = jnp.asarray(local_values)
    if local.shape[:2] != halo.local_mask.shape:
        raise ValueError("Halo-local values must begin with (partition, particle).")
    return jnp.sum(local, axis=0)


def migrate_particle_halos(
    plan: ParticleDomainDecompositionPlan,
    previous: ParticleHaloState,
    position: ArrayLike,
    active_mask: ArrayLike,
    /,
) -> ParticleHaloState:
    current = prepare_particle_halos(plan, position, active_mask)
    migration = jnp.sum(
        (current.owner != previous.owner) & jnp.asarray(active_mask, bool)
    )
    return ParticleHaloState(
        current.owner,
        current.owned_mask,
        current.halo_mask,
        current.local_mask,
        migration,
        current.halo_count,
        current.successful,
    )


class ParticleLoadBalanceReport(StrictModule):
    owned_particles: Array
    halo_particles: Array
    weighted_work: Array
    imbalance: Array


def particle_load_balance_report(
    halo: ParticleHaloState,
    pair_counts: ArrayLike,
    iteration_counts: ArrayLike,
    /,
) -> ParticleLoadBalanceReport:
    owned = jnp.sum(halo.owned_mask, axis=1)
    halos = jnp.sum(halo.halo_mask, axis=1)
    work = owned + halos + jnp.asarray(pair_counts) + jnp.asarray(iteration_counts)
    imbalance = jnp.max(work) / jnp.maximum(jnp.mean(work), 1.0)
    return ParticleLoadBalanceReport(owned, halos, work, imbalance)


__all__ = [
    "MixedPrecisionCertification",
    "ParticleBackendPolicy",
    "ParticleDomainDecompositionPlan",
    "ParticleHaloState",
    "ParticleKernelRequestPlan",
    "ParticleLoadBalanceReport",
    "certify_particle_precision",
    "halo_sum",
    "halo_update",
    "migrate_particle_halos",
    "particle_load_balance_report",
    "prepare_particle_halos",
]
