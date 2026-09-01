#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._distance import contact_tangent_basis


class ContactInterfacePlan(StrictModule, NonTrainableState):
    """Fixed contact quadrature connecting two nonmatching nodal traces."""

    plus_indices: Array
    plus_weights: Array
    minus_indices: Array
    minus_weights: Array
    reference_normal: Array
    quadrature_weight: Array
    route_keys: Array
    valid: Array
    plus_node_count: int = eqx.field(static=True)
    minus_node_count: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    interface_id: str = eqx.field(static=True)

    def __init__(
        self,
        plus_indices: ArrayLike,
        plus_weights: ArrayLike,
        minus_indices: ArrayLike,
        minus_weights: ArrayLike,
        reference_normal: ArrayLike,
        quadrature_weight: ArrayLike,
        /,
        *,
        plus_node_count: int,
        minus_node_count: int,
        route_keys: ArrayLike | None = None,
        valid: ArrayLike | None = None,
    ):
        plus_index = np.asarray(plus_indices)
        minus_index = np.asarray(minus_indices)
        plus_weight = np.asarray(plus_weights, dtype=float)
        minus_weight = np.asarray(minus_weights, dtype=float)
        normal = np.asarray(reference_normal, dtype=float)
        measure = np.asarray(quadrature_weight, dtype=float)
        plus_count = int(plus_node_count)
        minus_count = int(minus_node_count)
        if (
            plus_index.ndim != 2
            or minus_index.ndim != 2
            or not np.issubdtype(plus_index.dtype, np.integer)
            or not np.issubdtype(minus_index.dtype, np.integer)
        ):
            raise TypeError("Interface trace indices must be integer matrices.")
        capacity = plus_index.shape[0]
        if minus_index.shape[0] != capacity:
            raise ValueError("Interface trace capacities disagree.")
        if (
            plus_weight.shape != plus_index.shape
            or minus_weight.shape != minus_index.shape
        ):
            raise ValueError("Interface trace weights must match trace indices.")
        if (
            normal.ndim != 2
            or normal.shape[0] != capacity
            or normal.shape[1] not in (2, 3)
        ):
            raise ValueError("Interface normals require capacity by dimension shape.")
        if measure.shape != (capacity,):
            raise ValueError("Interface quadrature weights require capacity shape.")
        active = (
            np.ones((capacity,), dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        )
        if active.shape != (capacity,):
            raise ValueError("Interface valid mask requires capacity shape.")
        if plus_count <= 0 or minus_count <= 0:
            raise ValueError("Interface nodal trace sizes must be positive.")
        if np.any(
            active[:, None] & ((plus_index < 0) | (plus_index >= plus_count))
        ) or np.any(active[:, None] & ((minus_index < 0) | (minus_index >= minus_count))):
            raise ValueError("Active interface trace index is out of bounds.")
        if (
            np.any(~np.isfinite(plus_weight))
            or np.any(~np.isfinite(minus_weight))
            or np.any(~np.isfinite(normal))
            or np.any(~np.isfinite(measure))
            or np.any(measure < 0.0)
        ):
            raise ValueError("Interface quadrature data must be finite/nonnegative.")
        if not np.allclose(plus_weight[active].sum(axis=1), 1.0) or not np.allclose(
            minus_weight[active].sum(axis=1), 1.0
        ):
            raise ValueError(
                "Active interface trace weights must form affine coordinates."
            )
        normal_norm = np.sqrt(np.sum(normal * normal, axis=-1))
        if np.any(active & (normal_norm <= 0.0)):
            raise ValueError("Active interface normals must be nonzero.")
        normal = normal / np.where(normal_norm > 0.0, normal_norm, 1.0)[:, None]
        keys = (
            np.arange(capacity, dtype=np.int64)
            if route_keys is None
            else np.asarray(route_keys)
        )
        if keys.shape != (capacity,) or not np.issubdtype(keys.dtype, np.integer):
            raise TypeError("Interface route keys must be one integer vector.")
        keys = keys.astype(np.int64, copy=False)
        if np.unique(keys[active]).size != int(np.count_nonzero(active)):
            raise ValueError("Active interface route keys must be unique.")
        self.plus_indices = jnp.asarray(plus_index, dtype=jnp.int32)
        self.plus_weights = jnp.asarray(plus_weight)
        self.minus_indices = jnp.asarray(minus_index, dtype=jnp.int32)
        self.minus_weights = jnp.asarray(minus_weight)
        self.reference_normal = jnp.asarray(normal)
        self.quadrature_weight = jnp.asarray(measure)
        self.route_keys = jnp.asarray(keys, dtype=jnp.int64)
        self.valid = jnp.asarray(active)
        self.plus_node_count = plus_count
        self.minus_node_count = minus_count
        self.ambient_dimension = int(normal.shape[1])
        self.interface_id = canonical_fingerprint(
            {
                "kind": "contact-interface-plan",
                "plus_indices": array_tree_fingerprint(plus_index),
                "plus_weights": array_tree_fingerprint(plus_weight),
                "minus_indices": array_tree_fingerprint(minus_index),
                "minus_weights": array_tree_fingerprint(minus_weight),
                "normal": array_tree_fingerprint(normal),
                "measure": array_tree_fingerprint(measure),
                "keys": array_tree_fingerprint(keys),
                "valid": array_tree_fingerprint(active),
            }
        )

    @property
    def capacity(self) -> int:
        return int(self.valid.size)


class ContactInterfaceKinematics(StrictModule):
    plus_point: Array
    minus_point: Array
    relative_displacement: Array
    normal: Array
    tangent_basis: Array
    gap: Array
    tangential_jump: Array
    quadrature_weight: Array
    route_keys: Array
    valid: Array
    finite: Array
    successful: Array
    interface_id: str = eqx.field(static=True)


class ContactInterfaceResidual(StrictModule):
    plus_residual: Array
    minus_residual: Array
    action_reaction_residual: Array
    finite: Array
    successful: Array
    interface_id: str = eqx.field(static=True)


def evaluate_contact_interface(
    plan: ContactInterfacePlan,
    plus_positions: ArrayLike,
    minus_positions: ArrayLike,
    /,
) -> ContactInterfaceKinematics:
    if not isinstance(plan, ContactInterfacePlan):
        raise TypeError("plan must be ContactInterfacePlan.")
    plus = jnp.asarray(plus_positions)
    minus = jnp.asarray(minus_positions, dtype=plus.dtype)
    if plus.shape != (plan.plus_node_count, plan.ambient_dimension) or minus.shape != (
        plan.minus_node_count,
        plan.ambient_dimension,
    ):
        raise ValueError("Interface nodal position shapes are invalid.")
    safe_plus = jnp.clip(plan.plus_indices, 0, plan.plus_node_count - 1)
    safe_minus = jnp.clip(plan.minus_indices, 0, plan.minus_node_count - 1)
    plus_point = jnp.sum(
        plan.plus_weights[..., None].astype(plus.dtype) * plus[safe_plus], axis=1
    )
    minus_point = jnp.sum(
        plan.minus_weights[..., None].astype(plus.dtype) * minus[safe_minus], axis=1
    )
    relative = plus_point - minus_point
    normal = plan.reference_normal.astype(plus.dtype)
    tangent = contact_tangent_basis(normal)
    gap = jnp.sum(relative * normal, axis=-1)
    tangential = jnp.sum(tangent * relative[..., :, None], axis=-2)
    finite = (
        jnp.all(jnp.isfinite(plus_point))
        & jnp.all(jnp.isfinite(minus_point))
        & jnp.all(jnp.isfinite(gap))
        & jnp.all(jnp.isfinite(tangential))
    )
    return ContactInterfaceKinematics(
        plus_point,
        minus_point,
        relative,
        normal,
        tangent,
        gap,
        tangential,
        plan.quadrature_weight.astype(plus.dtype),
        plan.route_keys,
        plan.valid,
        finite,
        finite,
        plan.interface_id,
    )


def assemble_contact_interface_traction(
    plan: ContactInterfacePlan,
    traction: ArrayLike,
    /,
) -> ContactInterfaceResidual:
    if not isinstance(plan, ContactInterfacePlan):
        raise TypeError("plan must be ContactInterfacePlan.")
    traction_ = jnp.asarray(traction)
    if traction_.shape != (plan.capacity, plan.ambient_dimension):
        raise ValueError("Interface traction has invalid shape.")
    weighted = traction_ * plan.quadrature_weight[:, None].astype(traction_.dtype)
    weighted = jnp.where(plan.valid[:, None], weighted, 0.0)
    plus_local = (
        plan.plus_weights[..., None].astype(traction_.dtype) * weighted[:, None, :]
    )
    minus_local = (
        -plan.minus_weights[..., None].astype(traction_.dtype) * weighted[:, None, :]
    )
    safe_plus = jnp.clip(plan.plus_indices, 0, plan.plus_node_count - 1)
    safe_minus = jnp.clip(plan.minus_indices, 0, plan.minus_node_count - 1)
    plus_residual = (
        jnp.zeros((plan.plus_node_count, plan.ambient_dimension), dtype=traction_.dtype)
        .at[safe_plus.reshape((-1,))]
        .add(plus_local.reshape((-1, plan.ambient_dimension)))
    )
    minus_residual = (
        jnp.zeros((plan.minus_node_count, plan.ambient_dimension), dtype=traction_.dtype)
        .at[safe_minus.reshape((-1,))]
        .add(minus_local.reshape((-1, plan.ambient_dimension)))
    )
    balance = jnp.sum(plus_residual, axis=0) + jnp.sum(minus_residual, axis=0)
    finite = (
        jnp.all(jnp.isfinite(plus_residual))
        & jnp.all(jnp.isfinite(minus_residual))
        & jnp.all(jnp.isfinite(balance))
    )
    return ContactInterfaceResidual(
        plus_residual,
        minus_residual,
        balance,
        finite,
        finite,
        plan.interface_id,
    )


__all__ = [
    "ContactInterfaceKinematics",
    "ContactInterfacePlan",
    "ContactInterfaceResidual",
    "assemble_contact_interface_traction",
    "evaluate_contact_interface",
]
