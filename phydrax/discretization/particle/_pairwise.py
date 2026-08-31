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
from ..._numerics._compensated import two_sum
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import EdgeRelation
from ._periodic_cell import ParticleCell
from ._precision import ParticleAccumulation


class ParticleBox(StrictModule, NonTrainableState):
    """Axis-aligned particle domain with explicit periodic axes."""

    lower: Array
    upper: Array
    lengths: Array
    periodic_mask: Array
    periodic_axes: tuple[bool, ...] = eqx.field(static=True)
    box_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        /,
        *,
        periodic_axes: Sequence[bool] | None = None,
    ):
        lower_host = np.asarray(lower)
        upper_host = np.asarray(upper)
        if lower_host.ndim != 1 or upper_host.shape != lower_host.shape:
            raise ValueError("ParticleBox bounds must be matching rank-1 arrays.")
        if lower_host.size == 0:
            raise ValueError("ParticleBox requires at least one axis.")
        if np.any(~np.isfinite(lower_host)) or np.any(~np.isfinite(upper_host)):
            raise ValueError("ParticleBox bounds must be finite.")
        if np.any(upper_host <= lower_host):
            raise ValueError("ParticleBox upper bounds must exceed lower bounds.")
        axes = (
            (True,) * int(lower_host.size)
            if periodic_axes is None
            else tuple(bool(value) for value in periodic_axes)
        )
        if len(axes) != int(lower_host.size):
            raise ValueError("periodic_axes must align with ParticleBox bounds.")
        dtype = np.result_type(lower_host.dtype, upper_host.dtype, np.float32)
        lower_host = lower_host.astype(dtype, copy=False)
        upper_host = upper_host.astype(dtype, copy=False)
        self.lower = jnp.asarray(lower_host)
        self.upper = jnp.asarray(upper_host)
        self.lengths = jnp.asarray(upper_host - lower_host)
        self.periodic_mask = jnp.asarray(axes, dtype=bool)
        self.periodic_axes = axes
        self.box_id = canonical_fingerprint(
            {
                "kind": "particle-box",
                "bounds": array_tree_fingerprint(
                    {"lower": lower_host, "upper": upper_host}
                ),
                "periodic_axes": list(axes),
            }
        )

    @property
    def ambient_dimension(self) -> int:
        return int(self.lower.shape[0])

    def minimum_image(self, displacement: ArrayLike, /) -> Array:
        value = jnp.asarray(displacement)
        if not value.shape or value.shape[-1] != self.ambient_dimension:
            raise ValueError("Particle displacement must end in the box dimension.")
        lengths = self.lengths.astype(value.dtype)
        shift_count = jax.lax.stop_gradient(jnp.round(value / lengths))
        return value - jnp.where(
            self.periodic_mask,
            shift_count * lengths,
            jnp.zeros((), dtype=value.dtype),
        )


class ParticlePairRelation(StrictModule, NonTrainableState):
    """Canonical particle pairs with stable physical endpoint identities."""

    relation: EdgeRelation
    left_particle_ids: Array
    right_particle_ids: Array
    source_support_id: str = eqx.field(static=True)
    target_support_id: str = eqx.field(static=True)
    same_set: bool = eqx.field(static=True)
    unordered: bool = eqx.field(static=True)
    relation_schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        relation: EdgeRelation,
        left_particle_ids: ArrayLike,
        right_particle_ids: ArrayLike,
        /,
        *,
        source_support_id: str,
        target_support_id: str,
        same_set: bool,
        unordered: bool,
        relation_schema_id: str | None = None,
    ):
        if not isinstance(relation, EdgeRelation):
            raise TypeError("relation must be an EdgeRelation.")
        left_ids = jnp.asarray(left_particle_ids)
        right_ids = jnp.asarray(right_particle_ids)
        if (
            left_ids.shape != relation.route_shape
            or right_ids.shape != relation.route_shape
        ):
            raise ValueError("Stable particle IDs must have the relation route shape.")
        if not jnp.issubdtype(left_ids.dtype, jnp.integer) or not jnp.issubdtype(
            right_ids.dtype, jnp.integer
        ):
            raise TypeError("Stable particle pair IDs must be integers.")
        source_id = str(source_support_id)
        target_id = str(target_support_id)
        if not source_id or not target_id:
            raise ValueError("Particle pair support IDs must be non-empty.")
        if relation_schema_id is None:
            left_host = np.asarray(left_ids)
            right_host = np.asarray(right_ids)
            valid_host = np.asarray(relation.valid, dtype=bool)
            if (
                same_set
                and unordered
                and np.any(left_host[valid_host] >= right_host[valid_host])
            ):
                raise ValueError(
                    "Unordered same-set pairs require increasing stable particle IDs."
                )
            schema_id = canonical_fingerprint(
                {
                    "kind": "particle-pair-relation-schema",
                    "source_support": source_id,
                    "target_support": target_id,
                    "same_set": bool(same_set),
                    "unordered": bool(unordered),
                    "capacity": relation.capacity,
                    "pairs": array_tree_fingerprint(
                        {
                            "left_ids": left_host,
                            "right_ids": right_host,
                            "valid": valid_host,
                        }
                    ),
                }
            )
        else:
            schema_id = str(relation_schema_id)
            if not schema_id:
                raise ValueError("relation_schema_id must be non-empty.")
            if same_set and unordered:
                left_ids = eqx.error_if(
                    left_ids,
                    jnp.any(relation.valid & (left_ids >= right_ids)),
                    "Unordered same-set pairs require increasing stable particle IDs.",
                )
        self.relation = relation
        self.left_particle_ids = left_ids.astype(jnp.int64)
        self.right_particle_ids = right_ids.astype(jnp.int64)
        self.source_support_id = source_id
        self.target_support_id = target_id
        self.same_set = bool(same_set)
        self.unordered = bool(unordered)
        self.relation_schema_id = schema_id

    @property
    def left_indices(self) -> Array:
        return self.relation.source_indices

    @property
    def right_indices(self) -> Array:
        return self.relation.target_indices

    @property
    def valid(self) -> Array:
        return self.relation.valid

    @property
    def capacity(self) -> int:
        return self.relation.capacity


class ParticlePairGeometry(StrictModule):
    """Numeric pair geometry for one particle relation realization."""

    displacement: Array
    distance: Array
    direction: Array
    valid: Array
    relation_schema_id: str = eqx.field(static=True)
    box_id: str | None = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        displacement: ArrayLike,
        distance: ArrayLike,
        direction: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        relation_schema_id: str,
        box_id: str | None,
    ):
        displacement_ = jnp.asarray(displacement)
        distance_ = jnp.asarray(distance)
        direction_ = jnp.asarray(direction)
        valid_ = jnp.asarray(valid, dtype=bool)
        if displacement_.ndim != 2:
            raise ValueError("Pair displacement must have shape (pairs, dimension).")
        if direction_.shape != displacement_.shape:
            raise ValueError("Pair directions must match pair displacements.")
        if distance_.shape != displacement_.shape[:1] or valid_.shape != distance_.shape:
            raise ValueError("Pair distances and validity must have shape (pairs,).")
        self.displacement = displacement_
        self.distance = distance_
        self.direction = direction_
        self.valid = valid_
        self.relation_schema_id = str(relation_schema_id)
        self.box_id = None if box_id is None else str(box_id)
        self.schema_id = canonical_fingerprint(
            {
                "kind": "particle-pair-geometry-schema",
                "relation_schema": self.relation_schema_id,
                "box": self.box_id,
                "shape": list(displacement_.shape),
                "dtype": str(displacement_.dtype),
            }
        )


def particle_pair_geometry(
    positions: ArrayLike,
    pairs: ParticlePairRelation,
    /,
    *,
    box: ParticleBox | ParticleCell | None = None,
) -> ParticlePairGeometry:
    """Evaluate finite, zero-safe geometry on canonical particle pairs."""

    value = jnp.asarray(positions)
    if value.ndim != 2:
        raise ValueError("Particle positions must have shape (particles, dimension).")
    if value.shape[0] != pairs.relation.source_size:
        raise ValueError("Particle positions do not match the pair relation size.")
    if box is not None and box.ambient_dimension != value.shape[1]:
        raise ValueError("ParticleBox dimension does not match particle positions.")
    left = value[pairs.left_indices]
    right = value[pairs.right_indices]
    displacement = left - right
    if box is not None:
        displacement = box.minimum_image(displacement)
    valid = pairs.valid
    safe_displacement = jnp.where(valid[:, None], displacement, 0.0)
    distance_squared = jnp.sum(safe_displacement * safe_displacement, axis=-1)
    distance = jnp.sqrt(distance_squared)
    positive = valid & (distance > 0.0)
    safe_distance = jnp.where(positive, distance, 1.0)
    direction = jnp.where(
        positive[:, None],
        safe_displacement / safe_distance[:, None],
        0.0,
    )
    return ParticlePairGeometry(
        safe_displacement,
        distance,
        direction,
        valid,
        relation_schema_id=pairs.relation_schema_id,
        box_id=None if box is None else box.box_id,
    )


def _masked_pair_values(values: ArrayLike, valid: ArrayLike, /) -> Array:
    array = jnp.asarray(values)
    valid_ = jnp.asarray(valid, dtype=bool)
    if array.ndim < 1 or array.shape[0] != valid_.shape[0]:
        raise ValueError("Pair values must begin with the pair-validity dimension.")
    mask = valid_.reshape(valid_.shape + (1,) * (array.ndim - 1))
    return jnp.where(mask, array, 0.0)


def _scatter_deterministic(
    left_indices: Array,
    right_indices: Array,
    left_values: Array,
    right_values: Array,
    size: int,
    /,
) -> Array:
    output_shape = (size,) + left_values.shape[1:]

    def step(index, total):
        total = total.at[left_indices[index]].add(left_values[index])
        return total.at[right_indices[index]].add(right_values[index])

    return jax.lax.fori_loop(
        0,
        int(left_indices.shape[0]),
        step,
        jnp.zeros(output_shape, dtype=left_values.dtype),
    )


def _scatter_compensated(
    left_indices: Array,
    right_indices: Array,
    left_values: Array,
    right_values: Array,
    size: int,
    /,
) -> Array:
    output_shape = (size,) + left_values.shape[1:]
    zeros = jnp.zeros(output_shape, dtype=left_values.dtype)

    def add_one(total, correction, index, value):
        current = total[index]
        next_total, error = two_sum(current, value)
        total = total.at[index].set(next_total)
        correction = correction.at[index].add(error)
        return total, correction

    def step(index, carry):
        total, correction = carry
        total, correction = add_one(
            total,
            correction,
            left_indices[index],
            left_values[index],
        )
        return add_one(
            total,
            correction,
            right_indices[index],
            right_values[index],
        )

    total, correction = jax.lax.fori_loop(
        0,
        int(left_indices.shape[0]),
        step,
        (zeros, zeros),
    )
    return total + correction


def scatter_pair_sum(
    pairs: ParticlePairRelation,
    left_values: ArrayLike,
    right_values: ArrayLike,
    /,
    *,
    size: int,
    accumulation: ParticleAccumulation,
    valid: ArrayLike | None = None,
) -> Array:
    """Reduce independent left/right pair payloads to logical particles."""

    route_valid = pairs.valid if valid is None else pairs.valid & jnp.asarray(valid, bool)
    left = _masked_pair_values(left_values, route_valid)
    right = _masked_pair_values(right_values, route_valid)
    if left.shape != right.shape:
        raise ValueError("Left and right pair values must have matching shapes.")
    if left.shape[0] == 0:
        return jnp.zeros((int(size),) + left.shape[1:], dtype=left.dtype)
    if accumulation == "fast":
        output_shape = (int(size),) + left.shape[1:]
        result = jnp.zeros(output_shape, dtype=left.dtype)
        result = result.at[pairs.left_indices].add(left)
        return result.at[pairs.right_indices].add(right)
    if accumulation == "deterministic":
        return _scatter_deterministic(
            pairs.left_indices,
            pairs.right_indices,
            left,
            right,
            int(size),
        )
    if accumulation == "compensated":
        return _scatter_compensated(
            pairs.left_indices,
            pairs.right_indices,
            left,
            right,
            int(size),
        )
    raise ValueError("Unknown particle accumulation policy.")


def scatter_pair_exchange(
    pairs: ParticlePairRelation,
    pair_values: ArrayLike,
    /,
    *,
    size: int,
    accumulation: ParticleAccumulation,
    valid: ArrayLike | None = None,
) -> Array:
    """Scatter one pair exchange with equal and opposite endpoint signs."""

    values = jnp.asarray(pair_values)
    return scatter_pair_sum(
        pairs,
        values,
        -values,
        size=size,
        accumulation=accumulation,
        valid=valid,
    )


__all__ = [
    "ParticleBox",
    "ParticlePairGeometry",
    "ParticlePairRelation",
    "particle_pair_geometry",
    "scatter_pair_exchange",
    "scatter_pair_sum",
]
