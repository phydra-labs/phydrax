#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class VortexFilamentTopology(StrictModule, NonTrainableState):
    """Fixed-capacity oriented segments indexing a dynamic vertex array.

    Each active segment is directed from ``start_indices`` to ``end_indices``.
    Reversing those indices therefore reverses its positive-circulation velocity.
    Inactive slots use the canonical inert connectivity ``0 -> 0`` and ID ``-1``.
    """

    start_indices: Array
    end_indices: Array
    active: Array
    segment_ids: Array
    vertex_capacity: int = eqx.field(static=True)
    segment_capacity: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertex_capacity: int,
        start_indices: ArrayLike,
        end_indices: ArrayLike,
        /,
        *,
        active: ArrayLike | None = None,
        segment_ids: ArrayLike | None = None,
        topology_id: str | None = None,
    ):
        vertices = int(vertex_capacity)
        start = np.asarray(start_indices, dtype=np.int32)
        end = np.asarray(end_indices, dtype=np.int32)
        if (
            vertices <= 0
            or start.ndim != 1
            or end.shape != start.shape
            or start.size == 0
        ):
            raise ValueError(
                "Filament topology requires a positive vertex capacity and matching "
                "nonempty one-dimensional start/end arrays."
            )
        mask = (
            np.ones(start.shape, dtype=bool)
            if active is None
            else np.asarray(active, dtype=bool)
        )
        if mask.shape != start.shape:
            raise ValueError("active must have one entry per filament segment slot.")
        if np.any(start[mask] < 0) or np.any(start[mask] >= vertices):
            raise ValueError("Active filament start indices are outside vertex capacity.")
        if np.any(end[mask] < 0) or np.any(end[mask] >= vertices):
            raise ValueError("Active filament end indices are outside vertex capacity.")
        if np.any(start[mask] == end[mask]):
            raise ValueError("Active filament segments must join distinct vertex slots.")
        if np.any((start[~mask] != 0) | (end[~mask] != 0)):
            raise ValueError(
                "Inactive filament connectivity must use the inert 0 -> 0 sentinel."
            )
        ids = (
            np.where(mask, np.arange(start.size, dtype=np.int64), -1)
            if segment_ids is None
            else np.asarray(segment_ids, dtype=np.int64)
        )
        if ids.shape != start.shape:
            raise ValueError("segment_ids must have one entry per filament segment slot.")
        if np.any(ids[~mask] != -1):
            raise ValueError("Inactive filament segment IDs must be -1.")
        active_ids = ids[mask]
        if np.any(active_ids < 0) or np.unique(active_ids).size != active_ids.size:
            raise ValueError(
                "Active filament segment IDs must be unique and non-negative."
            )
        generated_id = canonical_fingerprint(
            {
                "kind": "oriented-vortex-filament-topology-v1",
                "vertex_capacity": vertices,
                "start_indices": array_tree_fingerprint(start),
                "end_indices": array_tree_fingerprint(end),
                "active": array_tree_fingerprint(mask),
                "segment_ids": array_tree_fingerprint(ids),
            }
        )
        identifier = generated_id if topology_id is None else str(topology_id)
        if not identifier:
            raise ValueError("topology_id must be nonempty when supplied.")
        self.start_indices = jnp.asarray(start)
        self.end_indices = jnp.asarray(end)
        self.active = jnp.asarray(mask)
        self.segment_ids = jnp.asarray(ids)
        self.vertex_capacity = vertices
        self.segment_capacity = int(start.size)
        self.topology_id = identifier

    @classmethod
    def from_segments(
        cls,
        segments: Sequence[tuple[int, int]],
        /,
        *,
        vertex_capacity: int,
        segment_capacity: int | None = None,
        topology_id: str | None = None,
    ) -> VortexFilamentTopology:
        """Pack oriented connectivity into a canonical fixed-capacity topology."""

        pairs = tuple((int(start), int(end)) for start, end in segments)
        capacity = len(pairs) if segment_capacity is None else int(segment_capacity)
        if capacity <= 0 or len(pairs) > capacity:
            raise ValueError("segment_capacity must be positive and hold every segment.")
        start = np.zeros((capacity,), dtype=np.int32)
        end = np.zeros((capacity,), dtype=np.int32)
        active = np.zeros((capacity,), dtype=bool)
        for index, pair in enumerate(pairs):
            start[index], end[index] = pair
            active[index] = True
        return cls(
            vertex_capacity,
            start,
            end,
            active=active,
            topology_id=topology_id,
        )


class OrientedFilamentGeometry(StrictModule):
    """Realized segment endpoints and explicit orientation evidence."""

    start: Array
    end: Array
    tangent: Array
    length: Array
    active: Array
    finite: Array
    nondegenerate: Array
    minimum_active_length: Array
    geometry_id: str = eqx.field(static=True)


class VortexFilamentState(StrictModule):
    """Dynamic vertices, circulation, and core radii on a stable topology."""

    topology: VortexFilamentTopology
    vertex_position: Array
    circulation: Array
    core_radius: Array
    state_layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: VortexFilamentTopology,
        vertex_position: ArrayLike,
        circulation: ArrayLike,
        core_radius: ArrayLike,
        /,
    ):
        if not isinstance(topology, VortexFilamentTopology):
            raise TypeError("topology must be a VortexFilamentTopology.")
        vertices = jnp.asarray(vertex_position)
        strength = jnp.asarray(circulation)
        core = jnp.asarray(core_radius, dtype=vertices.dtype)
        if vertices.shape != (topology.vertex_capacity, 3):
            raise ValueError("vertex_position must have shape (vertex_capacity, 3).")
        if strength.shape != (topology.segment_capacity,):
            raise ValueError("circulation must have shape (segment_capacity,).")
        if core.ndim == 0:
            core = jnp.broadcast_to(core, (topology.segment_capacity,))
        if core.shape != (topology.segment_capacity,):
            raise ValueError(
                "core_radius must be scalar or have shape (segment_capacity,)."
            )
        self.topology = topology
        self.vertex_position = vertices
        self.circulation = strength
        self.core_radius = core
        self.state_layout_id = canonical_fingerprint(
            {
                "kind": "vortex-filament-state-layout-v1",
                "topology": topology.topology_id,
                "position_shape": list(vertices.shape),
                "circulation_shape": list(strength.shape),
                "core_shape": list(core.shape),
            }
        )

    @property
    def safe_circulation(self) -> Array:
        """Circulation with inactive padding made exactly inert."""

        return jnp.where(self.topology.active, self.circulation, 0.0)

    @property
    def safe_core_radius(self) -> Array:
        """Non-negative core values for active slots and zero inactive padding."""

        return jnp.where(
            self.topology.active,
            jnp.maximum(self.core_radius, 0.0),
            0.0,
        )

    def geometry(self, /) -> OrientedFilamentGeometry:
        start = self.vertex_position[self.topology.start_indices]
        end = self.vertex_position[self.topology.end_indices]
        tangent = end - start
        length = jnp.linalg.norm(tangent, axis=-1)
        active = self.topology.active
        finite_slots = (
            jnp.all(jnp.isfinite(start), axis=-1)
            & jnp.all(jnp.isfinite(end), axis=-1)
            & jnp.isfinite(self.circulation)
            & jnp.isfinite(self.core_radius)
            & (self.core_radius >= 0.0)
        )
        nondegenerate_slots = length > jnp.finfo(length.dtype).eps
        safe_length = jnp.where(active, length, jnp.asarray(jnp.inf, dtype=length.dtype))
        return OrientedFilamentGeometry(
            start=start,
            end=end,
            tangent=tangent,
            length=length,
            active=active,
            finite=jnp.all(jnp.where(active, finite_slots, True)),
            nondegenerate=jnp.all(jnp.where(active, nondegenerate_slots, True)),
            minimum_active_length=jnp.where(
                jnp.any(active),
                jnp.min(safe_length),
                jnp.asarray(0.0, dtype=length.dtype),
            ),
            geometry_id=canonical_fingerprint(
                {
                    "kind": "oriented-vortex-filament-geometry-v1",
                    "topology": self.topology.topology_id,
                }
            ),
        )


__all__ = [
    "OrientedFilamentGeometry",
    "VortexFilamentState",
    "VortexFilamentTopology",
]
