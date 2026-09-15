#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class FlowObservablePlan(StrictModule, NonTrainableState):
    harmonics: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, harmonics: Sequence[int], /):
        harmonics_ = tuple(int(value) for value in harmonics)
        if (
            not harmonics_
            or any(value < 1 for value in harmonics_)
            or len(set(harmonics_)) != len(harmonics_)
        ):
            raise ValueError("Flow harmonics must be distinct positive integers.")
        self.harmonics = harmonics_
        self.plan_id = canonical_fingerprint(
            {"kind": "heavy-ion-flow-observable-plan", "harmonics": list(harmonics_)}
        )


class FlowObservables(StrictModule, NonTrainableState):
    q_vectors: Array
    event_plane_angles: Array
    flow_magnitudes: Array
    two_particle_cumulants: Array
    multiplicity: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


def compute_flow_observables(
    plan: FlowObservablePlan,
    phi: ArrayLike,
    weights: ArrayLike,
    active: ArrayLike,
    /,
) -> FlowObservables:
    if not isinstance(plan, FlowObservablePlan):
        raise TypeError("plan must be FlowObservablePlan.")
    phi_ = jnp.asarray(phi)
    weights_ = jnp.asarray(weights, dtype=phi_.dtype)
    active_ = jnp.asarray(active, dtype=bool)
    if phi_.ndim != 2 or weights_.shape != phi_.shape or active_.shape != phi_.shape:
        raise ValueError(
            "Flow angles, weights, and activity must have shape (event, particle)."
        )
    effective_weights = jnp.where(active_, weights_, 0.0)
    sum_weights = jnp.sum(effective_weights, axis=1)
    sum_squared_weights = jnp.sum(effective_weights * effective_weights, axis=1)
    q_vectors = jnp.stack(
        tuple(
            jnp.sum(effective_weights * jnp.exp(1j * harmonic * phi_), axis=1)
            for harmonic in plan.harmonics
        ),
        axis=1,
    )
    harmonics = jnp.asarray(plan.harmonics, dtype=phi_.dtype)
    event_plane = jnp.angle(q_vectors) / harmonics[None, :]
    magnitude = jnp.abs(q_vectors) / jnp.maximum(
        sum_weights[:, None], jnp.finfo(phi_.dtype).tiny
    )
    denominator = sum_weights * sum_weights - sum_squared_weights
    cumulants = (jnp.abs(q_vectors) ** 2 - sum_squared_weights[:, None]) / jnp.maximum(
        denominator[:, None], jnp.finfo(phi_.dtype).tiny
    )
    multiplicity = jnp.sum(active_, axis=1, dtype=jnp.int32)
    valid = (
        (multiplicity >= 2)
        & jnp.all(jnp.isfinite(magnitude), axis=1)
        & jnp.all(jnp.isfinite(cumulants), axis=1)
    )
    return FlowObservables(
        q_vectors, event_plane, magnitude, cumulants, multiplicity, valid, plan.plan_id
    )


__all__ = ["FlowObservablePlan", "FlowObservables", "compute_flow_observables"]
