#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._incompressible import FaceVelocity
from ._mac_marker_transfer import (
    MACMarkerRelation,
    MACMarkerTransferDiagnostics,
    PreparedMACMarkerTransfer,
)


DistributedMarkerExchange = Callable[[str, object], object]


class DistributedMarkerOwnershipPlan(StrictModule, NonTrainableState):
    """Deterministic single-owner and support-rank schedule for stable marker IDs."""

    marker_ids: Array
    owner_rank: Array
    support_rank: Array
    support_valid: Array
    rank_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        marker_ids: ArrayLike,
        owner_rank: ArrayLike,
        support_rank: ArrayLike,
        support_valid: ArrayLike,
        /,
        *,
        rank_count: int,
    ):
        ids = np.asarray(marker_ids)
        owner = np.asarray(owner_rank)
        support = np.asarray(support_rank)
        valid = np.asarray(support_valid, dtype=bool)
        ranks = int(rank_count)
        if ids.ndim != 1 or np.unique(ids).size != ids.size:
            raise ValueError("Distributed marker IDs must be unique and rank one.")
        if (
            owner.shape != ids.shape
            or support.shape != valid.shape
            or support.shape[0] != ids.size
        ):
            raise ValueError("Distributed marker ownership arrays are incompatible.")
        if ranks <= 0 or np.any(owner < 0) or np.any(owner >= ranks):
            raise ValueError("Every marker must have one valid owner rank.")
        if np.any(valid & ((support < 0) | (support >= ranks))):
            raise ValueError("A marker support route references an invalid rank.")
        if np.any(~np.any(valid & (support == owner[:, None]), axis=1)):
            raise ValueError("Every marker owner must participate in its support route.")
        self.marker_ids = jnp.asarray(ids, dtype=jnp.int64)
        self.owner_rank = jnp.asarray(owner, dtype=jnp.int32)
        self.support_rank = jnp.asarray(support, dtype=jnp.int32)
        self.support_valid = jnp.asarray(valid)
        self.rank_count = ranks
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-marker-ownership",
                "arrays": array_tree_fingerprint((ids, owner, support, valid)),
                "rank_count": ranks,
            }
        )

    def owner_mask(self, rank: int, /) -> Array:
        rank_ = int(rank)
        if rank_ < 0 or rank_ >= self.rank_count:
            raise ValueError("rank is outside the ownership communicator.")
        return self.owner_rank == rank_

    def support_mask(self, rank: int, /) -> Array:
        rank_ = int(rank)
        if rank_ < 0 or rank_ >= self.rank_count:
            raise ValueError("rank is outside the ownership communicator.")
        return jnp.any(self.support_valid & (self.support_rank == rank_), axis=-1)

    def canonical_order(self, rank: int, /) -> Array:
        mask = np.asarray(self.owner_rank) == int(rank)
        indices = np.flatnonzero(mask)
        order = np.argsort(np.asarray(self.marker_ids)[indices], kind="stable")
        return jnp.asarray(indices[order], dtype=jnp.int32)


class DistributedMarkerTransferDiagnostics(StrictModule):
    local: MACMarkerTransferDiagnostics
    owner_count: Array
    support_count: Array
    duplicated_owner_count: Array
    global_force_residual: Array
    global_work_residual: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class DistributedMACMarkerTransfer(StrictModule, NonTrainableState):
    """Owner-computes marker coupling with explicit halo exchange boundaries."""

    local: PreparedMACMarkerTransfer
    ownership: DistributedMarkerOwnershipPlan
    rank: int = eqx.field(static=True)
    owner_mask_active: Array
    support_mask_active: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        local: PreparedMACMarkerTransfer,
        ownership: DistributedMarkerOwnershipPlan,
        rank: int,
        /,
    ):
        if local.markers.capacity != ownership.marker_ids.size or not np.array_equal(
            np.asarray(local.markers.plan.marker_ids),
            np.asarray(ownership.marker_ids),
        ):
            raise ValueError("Local transfer and ownership marker identities differ.")
        rank_ = int(rank)
        owner = ownership.owner_mask(rank_)[local.markers.active_indices]
        support = ownership.support_mask(rank_)[local.markers.active_indices]
        self.local = local
        self.ownership = ownership
        self.rank = rank_
        self.owner_mask_active = owner
        self.support_mask_active = support
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-mac-marker-transfer",
                "local": local.prepared_id,
                "ownership": ownership.plan_id,
                "rank": rank_,
            }
        )

    def gather_owned(
        self,
        relation: MACMarkerRelation,
        velocity: FaceVelocity,
        exchange: DistributedMarkerExchange,
        /,
    ) -> Array:
        if not callable(exchange):
            raise TypeError("exchange must be callable.")
        local_value = self.local.gather(relation, velocity)
        owned = jnp.where(self.owner_mask_active[:, None], local_value, 0.0)
        exchanged = jnp.asarray(exchange("marker-gather", owned))
        expected = local_value.shape
        if exchanged.shape != expected:
            raise ValueError(f"marker-gather exchange must return shape {expected}.")
        return jnp.where(self.support_mask_active[:, None], exchanged, 0.0)

    def spread_owned(
        self,
        relation: MACMarkerRelation,
        marker_force: ArrayLike,
        exchange: DistributedMarkerExchange,
        /,
    ) -> FaceVelocity:
        if not callable(exchange):
            raise TypeError("exchange must be callable.")
        raw = jnp.asarray(marker_force)
        active = (
            self.local.markers.active_values(raw)
            if raw.shape
            == (self.local.markers.capacity, self.local.markers.ambient_dimension)
            else self.local.markers.active_velocity_space.validate(raw)
        )
        owned_active = jnp.where(self.owner_mask_active[:, None], active, 0.0)
        local_spread = self.local.spread(
            relation, self.local.markers.expand_active(owned_active)
        )
        exchanged = exchange("face-spread", local_spread)
        values = tuple(jnp.asarray(value) for value in exchanged)
        return self.local.operators.validate_velocity(values)

    def diagnostics(
        self,
        relation: MACMarkerRelation,
        velocity: FaceVelocity,
        marker_force: ArrayLike,
        exchange: DistributedMarkerExchange,
        /,
    ) -> DistributedMarkerTransferDiagnostics:
        raw = jnp.asarray(marker_force)
        active = (
            self.local.markers.active_values(raw)
            if raw.shape
            == (self.local.markers.capacity, self.local.markers.ambient_dimension)
            else self.local.markers.active_velocity_space.validate(raw)
        )
        full_force = self.local.markers.expand_active(active)
        local_diagnostics = self.local.diagnostics(relation, velocity, full_force)
        gathered = self.gather_owned(relation, velocity, exchange)
        spread = self.spread_owned(relation, full_force, exchange)
        marker_work = jnp.real(
            self.local.markers.active_velocity_space.inner(gathered, active)
        )
        face_work = jnp.real(self.local.operators.velocity_space.inner(velocity, spread))
        work_residual = marker_work - face_work
        global_values = jnp.asarray(
            exchange(
                "diagnostic-reduction",
                jnp.concatenate(
                    (
                        local_diagnostics.force_residual.reshape((-1,)),
                        work_residual.reshape((1,)),
                    )
                ),
            )
        )
        dimension = self.local.markers.ambient_dimension
        if global_values.shape != (dimension + 1,):
            raise ValueError(
                "diagnostic-reduction must return force components plus work residual."
            )
        force_residual = global_values[:dimension]
        global_work = global_values[-1]
        finite = local_diagnostics.finite & jnp.all(jnp.isfinite(global_values))
        tolerance = local_diagnostics.tolerance
        successful = (
            local_diagnostics.successful
            & finite
            & (jnp.max(jnp.abs(force_residual)) <= tolerance)
            & (jnp.abs(global_work) <= tolerance)
        )
        return DistributedMarkerTransferDiagnostics(
            local_diagnostics,
            jnp.sum(self.owner_mask_active),
            jnp.sum(self.support_mask_active),
            jnp.asarray(0, dtype=jnp.int32),
            force_residual,
            global_work,
            finite,
            successful,
            self.plan_id,
        )


__all__ = [
    "DistributedMACMarkerTransfer",
    "DistributedMarkerExchange",
    "DistributedMarkerOwnershipPlan",
    "DistributedMarkerTransferDiagnostics",
]
