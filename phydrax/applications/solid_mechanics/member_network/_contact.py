#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ...._trainable import NonTrainableState


def _fischer_burmeister(first: Array, second: Array, /) -> Array:
    return jnp.hypot(first, second) - first - second


class NodePlaneContact(StrictModule, NonTrainableState):
    node_indices: Array
    origins: Array
    normals: Array
    friction_coefficient: Array
    contact_id: str = eqx.field(static=True)

    def __init__(
        self,
        node_indices: ArrayLike,
        origins: ArrayLike,
        normals: ArrayLike,
        /,
        *,
        friction_coefficient: ArrayLike | None = None,
        contact_id: str = "node-plane-contact",
    ):
        indices = jnp.asarray(node_indices, dtype=jnp.int32)
        origins_ = jnp.asarray(origins)
        normals_ = jnp.asarray(normals, dtype=origins_.dtype)
        if origins_.shape != normals_.shape or origins_.shape[0] != indices.size:
            raise ValueError("Contact origins and normals must align with node indices.")
        norm = jnp.sqrt(jnp.sum(normals_ * normals_, axis=-1))
        if bool(jnp.any(norm <= 0.0)):
            raise ValueError("Contact normals must be nonzero.")
        friction = (
            jnp.zeros((indices.size,), dtype=origins_.dtype)
            if friction_coefficient is None
            else jnp.asarray(friction_coefficient, dtype=origins_.dtype)
        )
        if friction.shape != (indices.size,) or bool(jnp.any(friction < 0.0)):
            raise ValueError("Friction coefficients must be nonnegative per contact.")
        self.node_indices = indices
        self.origins = origins_
        self.normals = normals_ / norm[:, None]
        self.friction_coefficient = friction
        self.contact_id = str(contact_id)


class ContactState(StrictModule):
    gap: Array
    normal_traction: Array
    tangential_traction: Array
    active: Array
    sticking: Array
    normal_complementarity: Array
    friction_cone_residual: Array
    ambiguous: Array


class CableSaddleContact(StrictModule, NonTrainableState):
    incoming_member: int = eqx.field(static=True)
    outgoing_member: int = eqx.field(static=True)
    saddle_node: int = eqx.field(static=True)
    friction_coefficient: float = eqx.field(static=True)
    contact_id: str = eqx.field(static=True)

    def __init__(
        self,
        incoming_member: int,
        outgoing_member: int,
        saddle_node: int,
        /,
        *,
        friction_coefficient: float = 0.0,
        contact_id: str = "cable-saddle-contact",
    ):
        if float(friction_coefficient) < 0.0:
            raise ValueError("Saddle friction must be nonnegative.")
        self.incoming_member = int(incoming_member)
        self.outgoing_member = int(outgoing_member)
        self.saddle_node = int(saddle_node)
        self.friction_coefficient = float(friction_coefficient)
        self.contact_id = str(contact_id)


class CableSaddleState(StrictModule):
    wrap_angle: Array
    incoming_tension: Array
    outgoing_tension: Array
    capstan_limit: Array
    sliding: Array
    friction_residual: Array
    force_balance: Array


def evaluate_node_plane_contact(
    contact: NodePlaneContact,
    positions: ArrayLike,
    normal_traction: ArrayLike,
    tangential_trial: ArrayLike,
    /,
    *,
    ambiguity_tolerance: float = 1.0e-7,
) -> ContactState:
    xyz = jnp.asarray(positions)
    selected = xyz[contact.node_indices]
    traction = jnp.asarray(normal_traction, dtype=xyz.dtype)
    tangential = jnp.asarray(tangential_trial, dtype=xyz.dtype)
    gap = jnp.sum((selected - contact.origins) * contact.normals, axis=-1)
    if traction.shape != gap.shape or tangential.shape != selected.shape:
        raise ValueError("Contact traction arrays have incompatible shapes.")
    normal_residual = _fischer_burmeister(gap, traction)
    tangential_norm = jnp.sqrt(jnp.sum(tangential * tangential, axis=-1))
    limit = contact.friction_coefficient * jnp.maximum(traction, 0.0)
    scale = jnp.minimum(
        1.0,
        limit / jnp.maximum(tangential_norm, jnp.finfo(xyz.dtype).tiny),
    )
    projected = scale[:, None] * tangential
    cone = jnp.maximum(tangential_norm - limit, 0.0)
    active = traction > ambiguity_tolerance
    ambiguous = (gap <= ambiguity_tolerance) & (traction <= ambiguity_tolerance)
    return ContactState(
        gap,
        traction,
        projected,
        active,
        tangential_norm <= limit,
        normal_residual,
        cone,
        ambiguous,
    )


def evaluate_cable_saddle(
    contact: CableSaddleContact,
    positions: ArrayLike,
    incoming_anchor: int,
    outgoing_anchor: int,
    incoming_tension: ArrayLike,
    outgoing_tension: ArrayLike,
    /,
) -> CableSaddleState:
    xyz = jnp.asarray(positions)
    saddle = xyz[contact.saddle_node]
    incoming = xyz[int(incoming_anchor)] - saddle
    outgoing = xyz[int(outgoing_anchor)] - saddle
    incoming = incoming / jnp.sqrt(jnp.dot(incoming, incoming))
    outgoing = outgoing / jnp.sqrt(jnp.dot(outgoing, outgoing))
    angle = jnp.arccos(jnp.clip(jnp.dot(-incoming, outgoing), -1.0, 1.0))
    first = jnp.asarray(incoming_tension, dtype=xyz.dtype)
    second = jnp.asarray(outgoing_tension, dtype=xyz.dtype)
    limit = jnp.exp(contact.friction_coefficient * angle)
    ratio = jnp.maximum(first, second) / jnp.maximum(
        jnp.minimum(first, second), jnp.finfo(xyz.dtype).tiny
    )
    sliding = ratio >= limit
    residual = jnp.maximum(ratio - limit, 0.0)
    balance = first * incoming + second * outgoing
    return CableSaddleState(
        angle,
        first,
        second,
        limit,
        sliding,
        residual,
        balance,
    )


__all__ = [
    "CableSaddleContact",
    "CableSaddleState",
    "ContactState",
    "NodePlaneContact",
    "evaluate_cable_saddle",
    "evaluate_node_plane_contact",
]
