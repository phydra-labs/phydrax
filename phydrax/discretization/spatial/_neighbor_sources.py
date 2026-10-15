#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from phydrax._fingerprint import canonical_fingerprint
from phydrax._jax_config import x64_enabled
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState

from ._morton import canonical_morton_order, MortonAddressPlan, MortonEncoding


@final
class PreparedMortonSources(NonTrainableState, StrictModule):
    """Live point geometry and its reusable, canonical Morton storage order."""

    source_plan: MortonSourcePlan = eqx.field(static=True)
    points: Array
    sorted_coordinates: Array
    active: Array
    stable_ids: Array
    finite: Array
    in_domain: Array
    storage_to_logical: Array
    logical_to_storage: Array
    sorted_codes: Array
    sorted_stable_ids: Array
    sorted_active: Array
    active_count: Array
    invalid_points: Array
    stable_ids_unique: Array
    stable_ids_declared: bool = eqx.field(static=True)

    @property
    def point_capacity(self) -> int:
        return self.source_plan.point_capacity


@final
class MortonSourceTransition(NonTrainableState, StrictModule):
    """A refreshed candidate and atomic acceptance of valid source geometry."""

    candidate: PreparedMortonSources
    accepted: PreparedMortonSources
    successful: Array
    reused_order: Array


def _floating_source_points(points: ArrayLike, capacity: int, dimension: int) -> Array:
    values = jnp.asarray(points)
    if values.shape != (capacity, dimension):
        raise ValueError(f"points must have shape {(capacity, dimension)}.")
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError("points must have floating dtype.")
    return values


def _source_mask(mask: ArrayLike | None, capacity: int) -> Array:
    if mask is None:
        return jnp.ones((capacity,), dtype=jnp.bool_)
    active = jnp.asarray(mask, dtype=jnp.bool_)
    if active.shape != (capacity,):
        raise ValueError("active_mask must match point_capacity.")
    return active


def _source_ids(stable_ids: ArrayLike | None, capacity: int) -> Array:
    if stable_ids is None:
        return jnp.arange(capacity, dtype=jnp.int64)
    identifiers = jnp.asarray(stable_ids)
    if identifiers.shape != (capacity,):
        raise ValueError("stable_ids must match point_capacity.")
    if not jnp.issubdtype(identifiers.dtype, jnp.integer):
        raise TypeError("stable_ids must have integer dtype.")
    return identifiers


def _unique_active_ids(identifiers: Array, active: Array) -> Array:
    order = jnp.lexsort((identifiers, (~active).astype(jnp.int32)))
    sorted_ids = identifiers[order]
    sorted_active = active[order]
    duplicate = (
        sorted_active[1:] & sorted_active[:-1] & (sorted_ids[1:] == sorted_ids[:-1])
    )
    return ~jnp.any(duplicate)


def _ordered_sources(
    plan: MortonSourcePlan,
    points: Array,
    active: Array,
    identifiers: Array,
    encoding: MortonEncoding,
    order: Array,
    inverse: Array,
    unique: Array,
    declared: bool,
) -> PreparedMortonSources:
    valid = active & encoding.finite & encoding.in_domain
    return PreparedMortonSources(
        source_plan=plan,
        points=points,
        sorted_coordinates=encoding.coordinates[order],
        active=active,
        stable_ids=identifiers,
        finite=encoding.finite,
        in_domain=encoding.in_domain,
        storage_to_logical=order,
        logical_to_storage=inverse,
        sorted_codes=encoding.codes[order],
        sorted_stable_ids=identifiers[order],
        sorted_active=valid[order],
        active_count=jnp.sum(valid, dtype=jnp.int32),
        invalid_points=jnp.sum(active & ~encoding.in_domain, dtype=jnp.int32),
        stable_ids_unique=unique,
        stable_ids_declared=declared,
    )


@final
class MortonSourcePlan(StrictModule):
    """Bind an address and fixed source capacity independently of query batches."""

    address_plan: MortonAddressPlan
    point_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, address_plan: MortonAddressPlan, point_capacity: int) -> None:
        if not isinstance(address_plan, MortonAddressPlan):
            raise TypeError("address_plan must be a MortonAddressPlan.")
        if point_capacity < 1:
            raise ValueError("point_capacity must be positive.")
        if not x64_enabled():
            raise ValueError(
                "Morton sources require jax_enable_x64 for 64-bit Morton codes."
            )
        self.address_plan = address_plan
        self.point_capacity = point_capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "morton-source-plan",
                "address_plan_id": address_plan.plan_id,
                "point_capacity": point_capacity,
            }
        )

    def prepare(
        self,
        points: ArrayLike,
        *,
        active_mask: ArrayLike | None = None,
        stable_ids: ArrayLike | None = None,
    ) -> PreparedMortonSources:
        positions = _floating_source_points(
            points, self.point_capacity, self.address_plan.dimension
        )
        active = _source_mask(active_mask, self.point_capacity)
        identifiers = _source_ids(stable_ids, self.point_capacity)
        encoding = self.address_plan.encode(positions)
        order = canonical_morton_order(
            encoding.codes, identifiers, active & encoding.in_domain
        )
        inverse = (
            jnp.zeros_like(order)
            .at[order]
            .set(jnp.arange(self.point_capacity, dtype=jnp.int32))
        )
        return _ordered_sources(
            self,
            positions,
            active,
            identifiers,
            encoding,
            order,
            inverse,
            _unique_active_ids(identifiers, active),
            stable_ids is not None,
        )

    def refresh(
        self,
        previous: PreparedMortonSources,
        points: ArrayLike,
        *,
        active_mask: ArrayLike | None = None,
        stable_ids: ArrayLike | None = None,
    ) -> MortonSourceTransition:
        """Refresh geometry and ID values without changing preparation-time dtypes or declaration lineage."""
        _validate_sources(self, previous)
        positions = _floating_source_points(
            points, self.point_capacity, self.address_plan.dimension
        )
        active = (
            previous.active
            if active_mask is None
            else _source_mask(active_mask, self.point_capacity)
        )
        identifiers = (
            previous.stable_ids
            if stable_ids is None
            else _source_ids(stable_ids, self.point_capacity)
        )
        if positions.dtype != previous.points.dtype:
            raise ValueError("refresh must preserve the prepared coordinate dtype.")
        if identifiers.dtype != previous.stable_ids.dtype:
            raise ValueError("refresh must preserve the prepared stable-ID dtype.")
        encoding = self.address_plan.encode(positions)
        valid = active & encoding.in_domain
        order = previous.storage_to_logical
        sorted_valid = valid[order]
        codes = encoding.codes[order]
        ids = identifiers[order]
        # Complete canonical (~valid, code, stable_id) order, including inactive slots.
        activity_ordered = sorted_valid[:-1] | ~sorted_valid[1:]
        same_activity = sorted_valid[:-1] == sorted_valid[1:]
        keys_ordered = (codes[:-1] < codes[1:]) | (
            (codes[:-1] == codes[1:]) & (ids[:-1] <= ids[1:])
        )
        reuse = (
            (previous.invalid_points == 0)
            & previous.stable_ids_unique
            & jnp.all(previous.active == active)
            & jnp.all(previous.stable_ids == identifiers)
            & jnp.all(~active | encoding.in_domain)
            & jnp.all(activity_ordered & (~same_activity | keys_ordered))
        )

        def reused(_: None) -> PreparedMortonSources:
            return _ordered_sources(
                self,
                positions,
                active,
                identifiers,
                encoding,
                order,
                previous.logical_to_storage,
                previous.stable_ids_unique,
                previous.stable_ids_declared,
            )

        def rebuilt(_: None) -> PreparedMortonSources:
            fresh_order = canonical_morton_order(encoding.codes, identifiers, valid)
            inverse = (
                jnp.zeros_like(fresh_order)
                .at[fresh_order]
                .set(jnp.arange(self.point_capacity, dtype=jnp.int32))
            )
            return _ordered_sources(
                self,
                positions,
                active,
                identifiers,
                encoding,
                fresh_order,
                inverse,
                _unique_active_ids(identifiers, active),
                previous.stable_ids_declared,
            )

        candidate = jax.lax.cond(reuse, reused, rebuilt, None)
        successful = (candidate.invalid_points == 0) & candidate.stable_ids_unique
        accepted = jax.lax.cond(successful, lambda _: candidate, lambda _: previous, None)
        return MortonSourceTransition(
            candidate=candidate,
            accepted=accepted,
            successful=successful,
            reused_order=successful & reuse,
        )


def _validate_sources(plan: MortonSourcePlan, sources: PreparedMortonSources) -> None:
    if not isinstance(sources, PreparedMortonSources):
        raise TypeError("sources must be PreparedMortonSources.")
    if sources.source_plan.plan_id != plan.plan_id:
        raise ValueError("sources must belong to the query's MortonSourcePlan.")


__all__ = ["MortonSourcePlan", "PreparedMortonSources", "MortonSourceTransition"]
