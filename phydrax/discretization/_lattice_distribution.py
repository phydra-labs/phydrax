#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite Cartesian lattice ownership and parity-aware halo execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._distributed_field import DistributedHaloPlan


LatticeParity: TypeAlias = Literal[0, 1]


def _positive_shape(values: Sequence[int], name: str, /) -> tuple[int, ...]:
    shape = tuple(values)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError(f"{name} must contain positive extents.")
    return shape


def _partition_axis(extent: int, count: int, /) -> tuple[tuple[int, int], ...]:
    quotient, remainder = divmod(extent, count)
    sizes = tuple(quotient + (index < remainder) for index in range(count))
    cursor = 0
    ranges = []
    for size in sizes:
        ranges.append((cursor, cursor + size))
        cursor += size
    return tuple(ranges)


def _coordinate_owner(
    coordinate: tuple[int, ...],
    axis_ranges: tuple[tuple[tuple[int, int], ...], ...],
    partition_shape: tuple[int, ...],
    /,
) -> int:
    partition_coordinate = tuple(
        next(
            index
            for index, (start, stop) in enumerate(ranges)
            if start <= coordinate[axis] < stop
        )
        for axis, ranges in enumerate(axis_ranges)
    )
    return int(np.ravel_multi_index(partition_coordinate, partition_shape, order="C"))


def _neighbor(
    coordinate: tuple[int, ...],
    axis: int,
    orientation: int,
    shape: tuple[int, ...],
    periodic: tuple[bool, ...],
    /,
) -> tuple[int, ...] | None:
    displacement = -1 if orientation == 0 else 1
    values = list(coordinate)
    candidate = values[axis] + displacement
    if 0 <= candidate < shape[axis]:
        values[axis] = candidate
        return tuple(values)
    if periodic[axis]:
        values[axis] = candidate % shape[axis]
        return tuple(values)
    return None


class LatticeOwnership(StrictModule, NonTrainableState):
    """Canonical owners for sites, outgoing links, and positive plaquettes."""

    site_owner: Array
    link_owner: Array
    face_owner: Array
    face_valid: Array
    face_axes: Array
    site_count: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    face_orientation_count: int = eqx.field(static=True)
    ownership_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_owner: ArrayLike,
        link_owner: ArrayLike,
        face_owner: ArrayLike,
        face_valid: ArrayLike,
        face_axes: ArrayLike,
        /,
    ) -> None:
        sites = np.asarray(site_owner)
        links = np.asarray(link_owner)
        faces = np.asarray(face_owner)
        valid = np.asarray(face_valid)
        axes = np.asarray(face_axes)
        if sites.ndim != 1 or sites.size == 0:
            raise ValueError("site_owner must contain one owner per site.")
        if links.ndim != 2 or links.shape[0] != sites.size:
            raise ValueError("link_owner must have shape (site_count, dimension).")
        if faces.ndim != 2 or faces.shape[0] != sites.size or valid.shape != faces.shape:
            raise ValueError("face ownership and validity shapes must agree.")
        if axes.shape != (faces.shape[1], 2):
            raise ValueError(
                "face_axes must identify every positive plaquette orientation."
            )
        if not all(
            np.issubdtype(value.dtype, np.integer)
            for value in (sites, links, faces, axes)
        ):
            raise TypeError("Lattice ownership arrays must use integer indices.")
        if valid.dtype != np.dtype(np.bool_):
            raise TypeError("face_valid must be boolean.")
        self.site_owner = jnp.asarray(sites, dtype=jnp.int32)
        self.link_owner = jnp.asarray(links, dtype=jnp.int32)
        self.face_owner = jnp.asarray(faces, dtype=jnp.int32)
        self.face_valid = jnp.asarray(valid, dtype=jnp.bool_)
        self.face_axes = jnp.asarray(axes, dtype=jnp.int32)
        self.site_count = sites.size
        self.dimension = links.shape[1]
        self.face_orientation_count = faces.shape[1]
        self.ownership_id = canonical_fingerprint(
            {
                "kind": "cartesian-lattice-ownership",
                "site_owner": sites,
                "link_owner": links,
                "face_owner": faces,
                "face_valid": valid,
                "face_axes": axes,
            }
        )

    def site_mask(self, partition: int, /) -> Array:
        return self.site_owner == int(partition)

    def link_mask(self, partition: int, /) -> Array:
        return self.link_owner == int(partition)

    def face_mask(self, partition: int, /) -> Array:
        return self.face_valid & (self.face_owner == int(partition))


class PackedLatticeHalo(StrictModule):
    payload: Array
    valid: Array
    plan_id: str = eqx.field(static=True)
    parity: int = eqx.field(static=True)
    rank_local: bool = eqx.field(static=True)
    payload_kind: str = eqx.field(static=True)


class StartedLatticeHalo(StrictModule):
    payload: Array
    valid: Array
    plan_id: str = eqx.field(static=True)
    parity: int = eqx.field(static=True)
    rank_local: bool = eqx.field(static=True)
    payload_kind: str = eqx.field(static=True)


class ProjectedSpinorHalo(StrictModule):
    values: Array
    valid: Array
    plan_id: str = eqx.field(static=True)
    target_parity: int = eqx.field(static=True)
    rank_local: bool = eqx.field(static=True)


class LatticeHaloPlan(StrictModule, NonTrainableState):
    """Nearest-neighbor halo routes with parity and projected-spinor metadata."""

    distributed: DistributedHaloPlan
    global_site_parity: Array
    local_site_parity: Array
    local_interior: Array
    local_boundary: Array
    projected_send_indices: Array
    projected_receive_indices: Array
    projected_send_direction: Array
    projected_receive_direction: Array
    projected_send_orientation: Array
    projected_receive_orientation: Array
    projected_send_target_parity: Array
    projected_receive_target_parity: Array
    projected_send_valid: Array
    projected_receive_valid: Array
    dimension: int = eqx.field(static=True)
    projected_message_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        distributed: DistributedHaloPlan,
        global_site_parity: ArrayLike,
        neighbor_ids: ArrayLike,
        neighbor_valid: ArrayLike,
        /,
    ) -> None:
        if not isinstance(distributed, DistributedHaloPlan):
            raise TypeError("distributed must be a DistributedHaloPlan.")
        parity = np.asarray(global_site_parity)
        neighbors = np.asarray(neighbor_ids)
        neighbor_mask = np.asarray(neighbor_valid)
        if parity.shape != (distributed.entity_count,) or not np.all(
            (parity == 0) | (parity == 1)
        ):
            raise ValueError("global_site_parity must contain one binary value per site.")
        if (
            neighbors.ndim != 3
            or neighbors.shape[0] != distributed.entity_count
            or neighbors.shape[2] != 2
            or neighbor_mask.shape != neighbors.shape
        ):
            raise ValueError(
                "Neighbor arrays must have shape (site_count, dimension, 2)."
            )
        ids = np.asarray(distributed.local_global_ids)
        valid = np.asarray(distributed.local_valid)
        owned = np.asarray(distributed.local_owned)
        owner = np.asarray(distributed.entity_owner)
        parts, capacity = ids.shape
        local_parity = parity[ids]
        local_parity = np.where(valid, local_parity, 0).astype(np.int32)
        interior = np.zeros((parts, capacity), dtype=np.bool_)
        boundary = np.zeros_like(interior)
        local_maps = [
            {
                int(global_id): local
                for local, global_id in enumerate(ids[part][valid[part]])
            }
            for part in range(parts)
        ]
        for part in range(parts):
            for local_index in np.flatnonzero(owned[part]):
                global_id = int(ids[part, local_index])
                is_boundary = bool(np.any(~neighbor_mask[global_id]))
                for remote in neighbors[global_id][neighbor_mask[global_id]]:
                    is_boundary = is_boundary or int(owner[int(remote)]) != part
                boundary[part, local_index] = is_boundary
                interior[part, local_index] = not is_boundary

        pair_edges: dict[tuple[int, int], list[tuple[int, int, int, int, int]]] = {}
        for receiver in range(parts):
            for target_local in np.flatnonzero(owned[receiver]):
                target_global = int(ids[receiver, target_local])
                for direction in range(neighbors.shape[1]):
                    for orientation in range(2):
                        if not neighbor_mask[target_global, direction, orientation]:
                            continue
                        source_global = int(
                            neighbors[target_global, direction, orientation]
                        )
                        source = int(owner[source_global])
                        if source == receiver:
                            continue
                        source_local = local_maps[source][source_global]
                        pair_edges.setdefault((source, receiver), []).append(
                            (
                                source_local,
                                int(target_local),
                                direction,
                                orientation,
                                int(parity[target_global]),
                            )
                        )
        pair_phase = {
            pair: phase
            for phase, permutation in enumerate(distributed.permutations)
            for pair in permutation
        }
        message_capacity = max((len(values) for values in pair_edges.values()), default=1)
        phase_count = len(distributed.permutations)
        message_shape = (phase_count, parts, message_capacity)
        send = np.zeros(message_shape, dtype=np.int32)
        receive = np.zeros(message_shape, dtype=np.int32)
        send_direction = np.zeros(message_shape, dtype=np.int32)
        receive_direction = np.zeros(message_shape, dtype=np.int32)
        send_orientation = np.zeros(message_shape, dtype=np.int32)
        receive_orientation = np.zeros(message_shape, dtype=np.int32)
        send_target_parity = np.zeros(message_shape, dtype=np.int32)
        receive_target_parity = np.zeros(message_shape, dtype=np.int32)
        send_valid = np.zeros(message_shape, dtype=np.bool_)
        receive_valid = np.zeros(message_shape, dtype=np.bool_)
        for pair, messages in pair_edges.items():
            phase = pair_phase[pair]
            source, receiver = pair
            count = len(messages)
            send[phase, source, :count] = [value[0] for value in messages]
            receive[phase, receiver, :count] = [value[1] for value in messages]
            send_direction[phase, source, :count] = [value[2] for value in messages]
            receive_direction[phase, receiver, :count] = [value[2] for value in messages]
            send_orientation[phase, source, :count] = [value[3] for value in messages]
            receive_orientation[phase, receiver, :count] = [
                value[3] for value in messages
            ]
            send_target_parity[phase, source, :count] = [value[4] for value in messages]
            receive_target_parity[phase, receiver, :count] = [
                value[4] for value in messages
            ]
            send_valid[phase, source, :count] = True
            receive_valid[phase, receiver, :count] = True
        self.distributed = distributed
        self.global_site_parity = jnp.asarray(parity, dtype=jnp.int32)
        self.local_site_parity = jnp.asarray(local_parity)
        self.local_interior = jnp.asarray(interior)
        self.local_boundary = jnp.asarray(boundary)
        self.projected_send_indices = jnp.asarray(send)
        self.projected_receive_indices = jnp.asarray(receive)
        self.projected_send_direction = jnp.asarray(send_direction)
        self.projected_receive_direction = jnp.asarray(receive_direction)
        self.projected_send_orientation = jnp.asarray(send_orientation)
        self.projected_receive_orientation = jnp.asarray(receive_orientation)
        self.projected_send_target_parity = jnp.asarray(send_target_parity)
        self.projected_receive_target_parity = jnp.asarray(receive_target_parity)
        self.projected_send_valid = jnp.asarray(send_valid)
        self.projected_receive_valid = jnp.asarray(receive_valid)
        self.dimension = neighbors.shape[1]
        self.projected_message_capacity = message_capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "parity-aware-lattice-halo",
                "distributed": distributed.plan_id,
                "site_parity": parity,
                "local_interior": interior,
                "projected_send": send,
                "projected_receive": receive,
                "projected_send_direction": send_direction,
                "projected_receive_direction": receive_direction,
                "projected_send_orientation": send_orientation,
                "projected_receive_orientation": receive_orientation,
            }
        )

    @property
    def partition_count(self) -> int:
        return self.distributed.part_count

    @property
    def local_capacity(self) -> int:
        return self.distributed.local_capacity

    @property
    def phase_count(self) -> int:
        return len(self.distributed.permutations)

    def _parity(self, parity: LatticeParity | None, /) -> int:
        if parity is not None and parity not in (0, 1):
            raise ValueError("parity must be zero, one, or None.")
        return -1 if parity is None else int(parity)

    def pack(
        self,
        local_values: ArrayLike,
        /,
        *,
        parity: LatticeParity | None = None,
    ) -> PackedLatticeHalo:
        values = jnp.asarray(local_values)
        expected = (self.partition_count, self.local_capacity)
        if values.shape[:2] != expected:
            raise ValueError(f"local_values must begin with shape {expected}.")
        indices = self.distributed.phase_send_indices
        partition = jnp.arange(self.partition_count)[None, :, None]
        payload = values[partition, indices]
        valid = self.distributed.phase_send_valid
        parity_value = self._parity(parity)
        if parity_value >= 0:
            site_parity = self.local_site_parity[partition, indices]
            valid = valid & (site_parity == parity_value)
        mask = valid.reshape(valid.shape + (1,) * (values.ndim - 2))
        return PackedLatticeHalo(
            jnp.where(mask, payload, 0),
            valid,
            self.plan_id,
            parity_value,
            False,
            "full-site",
        )

    def pack_rank(
        self,
        local_values: ArrayLike,
        partition: ArrayLike,
        /,
        *,
        parity: LatticeParity | None = None,
    ) -> PackedLatticeHalo:
        values = jnp.asarray(local_values)
        if values.shape[:1] != (self.local_capacity,):
            raise ValueError("Rank-local values must begin with local_capacity.")
        part = jnp.asarray(partition, dtype=jnp.int32)
        indices = self.distributed.phase_send_indices[:, part]
        payload = values[indices]
        valid = self.distributed.phase_send_valid[:, part]
        parity_value = self._parity(parity)
        if parity_value >= 0:
            valid = valid & (self.local_site_parity[part, indices] == parity_value)
        mask = valid.reshape(valid.shape + (1,) * (values.ndim - 1))
        return PackedLatticeHalo(
            jnp.where(mask, payload, 0),
            valid,
            self.plan_id,
            parity_value,
            True,
            "full-site",
        )

    def start_reference(self, packed: PackedLatticeHalo, /) -> StartedLatticeHalo:
        if (
            packed.plan_id != self.plan_id
            or packed.rank_local
            or packed.payload_kind != "full-site"
        ):
            raise ValueError("Reference halo start requires a matching all-rank pack.")
        received = jnp.zeros_like(packed.payload)
        valid = jnp.zeros_like(packed.valid)
        for phase, permutation in enumerate(self.distributed.permutations):
            for source, target in permutation:
                received = received.at[phase, target].set(packed.payload[phase, source])
                valid = valid.at[phase, target].set(packed.valid[phase, source])
        valid = valid & self.distributed.phase_receive_valid
        return StartedLatticeHalo(
            received,
            valid,
            self.plan_id,
            packed.parity,
            False,
            "full-site",
        )

    def start(
        self,
        packed: PackedLatticeHalo,
        partition: ArrayLike,
        /,
        *,
        axis_name: str,
    ) -> StartedLatticeHalo:
        if (
            packed.plan_id != self.plan_id
            or not packed.rank_local
            or packed.payload_kind != "full-site"
        ):
            raise ValueError(
                "Distributed halo start requires a matching rank-local pack."
            )
        part = jnp.asarray(partition, dtype=jnp.int32)
        if self.phase_count == 0:
            return StartedLatticeHalo(
                jnp.zeros_like(packed.payload),
                jnp.zeros_like(packed.valid),
                self.plan_id,
                packed.parity,
                True,
                "full-site",
            )
        received = []
        valid = []
        for phase, permutation in enumerate(self.distributed.permutations):
            received.append(
                jax.lax.ppermute(packed.payload[phase], axis_name, permutation)
            )
            routed_valid = jax.lax.ppermute(
                packed.valid[phase].astype(jnp.int32), axis_name, permutation
            ).astype("bool")
            valid.append(routed_valid & self.distributed.phase_receive_valid[phase, part])
        return StartedLatticeHalo(
            jnp.stack(received),
            jnp.stack(valid),
            self.plan_id,
            packed.parity,
            True,
            "full-site",
        )

    def complete(
        self,
        started: StartedLatticeHalo,
        local_owned_values: ArrayLike,
        /,
        *,
        partition: ArrayLike | None = None,
    ) -> Array:
        if started.plan_id != self.plan_id or started.payload_kind != "full-site":
            raise ValueError("Started halo belongs to another plan.")
        values = jnp.asarray(local_owned_values)
        if started.rank_local:
            if partition is None or values.shape[:1] != (self.local_capacity,):
                raise ValueError(
                    "Rank-local completion requires partition and local values."
                )
            part = jnp.asarray(partition, dtype=jnp.int32)
            result = values
            for phase in range(self.phase_count):
                indices = self.distributed.phase_receive_indices[phase, part]
                mask = started.valid[phase].reshape(
                    started.valid[phase].shape + (1,) * (values.ndim - 1)
                )
                result = result.at[indices].add(
                    jnp.where(mask, started.payload[phase], 0)
                )
            return result
        expected = (self.partition_count, self.local_capacity)
        if values.shape[:2] != expected or partition is not None:
            raise ValueError("All-rank completion requires all-rank local values.")
        result = values
        for phase in range(self.phase_count):
            for part in range(self.partition_count):
                indices = self.distributed.phase_receive_indices[phase, part]
                mask = started.valid[phase, part].reshape(
                    started.valid[phase, part].shape + (1,) * (values.ndim - 2)
                )
                result = result.at[part, indices].add(
                    jnp.where(mask, started.payload[phase, part], 0)
                )
        return result

    def pack_projected_spinor(
        self,
        local_spinor: ArrayLike,
        projector_basis: ArrayLike,
        /,
        *,
        target_parity: LatticeParity | None = None,
        maximum_payload_bytes: int = 1_073_741_824,
    ) -> PackedLatticeHalo:
        spinor = jnp.asarray(local_spinor)
        projectors = jnp.asarray(projector_basis)
        expected = (self.partition_count, self.local_capacity)
        if spinor.shape[:2] != expected or spinor.ndim < 4:
            raise ValueError(
                "Spinors must have shape (part, local_site, spin, color, ...)."
            )
        if (
            projectors.ndim != 4
            or projectors.shape[:2] != (self.dimension, 2)
            or projectors.shape[3] != spinor.shape[2]
        ):
            raise ValueError(
                "projector_basis must have shape (dimension, 2, half_spin, spin)."
            )
        maximum = int(maximum_payload_bytes)
        payload_bytes = (
            self.phase_count
            * self.partition_count
            * self.projected_message_capacity
            * projectors.shape[2]
            * prod(spinor.shape[3:])
            * spinor.dtype.itemsize
        )
        if maximum <= 0 or payload_bytes > maximum:
            raise ValueError("Projected spinor halo exceeds maximum_payload_bytes.")
        indices = self.projected_send_indices
        partition = jnp.arange(self.partition_count)[None, :, None]
        gathered = spinor[partition, indices]
        selected = projectors[
            self.projected_send_direction,
            self.projected_send_orientation,
        ]
        flat = gathered.reshape(gathered.shape[:3] + (spinor.shape[2], -1))
        payload = contract("prmhs,prmsq->prmhq", selected, flat, backend="jax")
        payload = payload.reshape(payload.shape[:4] + spinor.shape[3:])
        parity_value = self._parity(target_parity)
        valid = self.projected_send_valid
        if parity_value >= 0:
            valid = valid & (self.projected_send_target_parity == parity_value)
        mask = valid.reshape(valid.shape + (1,) * (payload.ndim - 3))
        return PackedLatticeHalo(
            jnp.where(mask, payload, 0),
            valid,
            self.plan_id,
            parity_value,
            False,
            "projected-spinor",
        )

    def pack_projected_spinor_rank(
        self,
        local_spinor: ArrayLike,
        projector_basis: ArrayLike,
        partition: ArrayLike,
        /,
        *,
        target_parity: LatticeParity | None = None,
        maximum_payload_bytes: int = 1_073_741_824,
    ) -> PackedLatticeHalo:
        spinor = jnp.asarray(local_spinor)
        projectors = jnp.asarray(projector_basis)
        if spinor.shape[:1] != (self.local_capacity,) or spinor.ndim < 3:
            raise ValueError(
                "Rank-local spinors must have shape (local_site, spin, color, ...)."
            )
        if (
            projectors.ndim != 4
            or projectors.shape[:2] != (self.dimension, 2)
            or projectors.shape[3] != spinor.shape[1]
        ):
            raise ValueError(
                "projector_basis must have shape (dimension, 2, half_spin, spin)."
            )
        maximum = int(maximum_payload_bytes)
        payload_bytes = (
            self.phase_count
            * self.projected_message_capacity
            * projectors.shape[2]
            * prod(spinor.shape[2:])
            * spinor.dtype.itemsize
        )
        if maximum <= 0 or payload_bytes > maximum:
            raise ValueError("Projected spinor halo exceeds maximum_payload_bytes.")
        part = jnp.asarray(partition, dtype=jnp.int32)
        indices = self.projected_send_indices[:, part]
        gathered = spinor[indices]
        selected = projectors[
            self.projected_send_direction[:, part],
            self.projected_send_orientation[:, part],
        ]
        flat = gathered.reshape(gathered.shape[:2] + (spinor.shape[1], -1))
        payload = contract("pmhs,pmsq->pmhq", selected, flat, backend="jax")
        payload = payload.reshape(payload.shape[:3] + spinor.shape[2:])
        parity_value = self._parity(target_parity)
        valid = self.projected_send_valid[:, part]
        if parity_value >= 0:
            valid = valid & (self.projected_send_target_parity[:, part] == parity_value)
        mask = valid.reshape(valid.shape + (1,) * (payload.ndim - 2))
        return PackedLatticeHalo(
            jnp.where(mask, payload, 0),
            valid,
            self.plan_id,
            parity_value,
            True,
            "projected-spinor",
        )

    def start_projected(
        self,
        packed: PackedLatticeHalo,
        partition: ArrayLike,
        /,
        *,
        axis_name: str,
    ) -> StartedLatticeHalo:
        if (
            packed.plan_id != self.plan_id
            or not packed.rank_local
            or packed.payload_kind != "projected-spinor"
        ):
            raise ValueError(
                "Projected distributed start requires a matching rank-local pack."
            )
        part = jnp.asarray(partition, dtype=jnp.int32)
        if self.phase_count == 0:
            return StartedLatticeHalo(
                jnp.zeros_like(packed.payload),
                jnp.zeros_like(packed.valid),
                self.plan_id,
                packed.parity,
                True,
                "projected-spinor",
            )
        received = []
        received_valid = []
        for phase, permutation in enumerate(self.distributed.permutations):
            received.append(
                jax.lax.ppermute(packed.payload[phase], axis_name, permutation)
            )
            routed = jax.lax.ppermute(
                packed.valid[phase].astype(jnp.int32),
                axis_name,
                permutation,
            ).astype("bool")
            received_valid.append(routed & self.projected_receive_valid[phase, part])
        return StartedLatticeHalo(
            jnp.stack(received),
            jnp.stack(received_valid),
            self.plan_id,
            packed.parity,
            True,
            "projected-spinor",
        )

    def reconstruct_projected_spinor(
        self,
        started: StartedLatticeHalo,
        reconstructor_basis: ArrayLike,
        partition: ArrayLike,
        /,
    ) -> ProjectedSpinorHalo:
        if (
            started.plan_id != self.plan_id
            or not started.rank_local
            or started.payload_kind != "projected-spinor"
        ):
            raise ValueError(
                "Projected distributed completion requires a matching started halo."
            )
        reconstructors = jnp.asarray(reconstructor_basis)
        if reconstructors.ndim != 4 or reconstructors.shape[:2] != (
            self.dimension,
            2,
        ):
            raise ValueError(
                "reconstructor_basis must have shape (dimension, 2, spin, half_spin)."
            )
        if started.payload.shape[2] != reconstructors.shape[3]:
            raise ValueError("Projected payload width and reconstructor width disagree.")
        part = jnp.asarray(partition, dtype=jnp.int32)
        tail = started.payload.shape[3:]
        result = jnp.zeros(
            (
                self.local_capacity,
                self.dimension,
                2,
                reconstructors.shape[2],
                *tail,
            ),
            dtype=started.payload.dtype,
        )
        valid = jnp.zeros(
            (self.local_capacity, self.dimension, 2),
            dtype=jnp.bool_,
        )
        for phase in range(self.phase_count):
            indices = self.projected_receive_indices[phase, part]
            directions = self.projected_receive_direction[phase, part]
            orientations = self.projected_receive_orientation[phase, part]
            selected = reconstructors[directions, orientations]
            payload = started.payload[phase]
            flat = payload.reshape(payload.shape[:1] + (payload.shape[1], -1))
            reconstructed = contract("msh,mhq->msq", selected, flat, backend="jax")
            reconstructed = reconstructed.reshape(
                (payload.shape[0], reconstructors.shape[2], *tail)
            )
            message_valid = started.valid[phase]
            for slot in range(self.projected_message_capacity):
                value = jnp.where(message_valid[slot], reconstructed[slot], 0)
                result = result.at[
                    indices[slot],
                    directions[slot],
                    orientations[slot],
                ].add(value)
                valid = valid.at[
                    indices[slot],
                    directions[slot],
                    orientations[slot],
                ].set(message_valid[slot])
        return ProjectedSpinorHalo(
            result,
            valid,
            self.plan_id,
            started.parity,
            True,
        )

    def reconstruct_projected_spinor_reference(
        self,
        packed: PackedLatticeHalo,
        reconstructor_basis: ArrayLike,
        /,
    ) -> ProjectedSpinorHalo:
        if (
            packed.plan_id != self.plan_id
            or packed.rank_local
            or packed.payload_kind != "projected-spinor"
        ):
            raise ValueError(
                "Projected reconstruction requires a matching all-rank pack."
            )
        reconstructors = jnp.asarray(reconstructor_basis)
        if reconstructors.ndim != 4 or reconstructors.shape[:2] != (
            self.dimension,
            2,
        ):
            raise ValueError(
                "reconstructor_basis must have shape (dimension, 2, spin, half_spin)."
            )
        if packed.payload.shape[3] != reconstructors.shape[3]:
            raise ValueError("Projected payload width and reconstructor width disagree.")
        received = jnp.zeros_like(packed.payload)
        received_valid = jnp.zeros_like(packed.valid)
        for phase, permutation in enumerate(self.distributed.permutations):
            for source, target in permutation:
                received = received.at[phase, target].set(packed.payload[phase, source])
                received_valid = received_valid.at[phase, target].set(
                    packed.valid[phase, source]
                )
        received_valid = received_valid & self.projected_receive_valid
        tail = packed.payload.shape[4:]
        result = jnp.zeros(
            (
                self.partition_count,
                self.local_capacity,
                self.dimension,
                2,
                reconstructors.shape[2],
                *tail,
            ),
            dtype=packed.payload.dtype,
        )
        valid = jnp.zeros(
            (self.partition_count, self.local_capacity, self.dimension, 2),
            dtype=jnp.bool_,
        )
        for phase, permutation in enumerate(self.distributed.permutations):
            for _, target in permutation:
                indices = self.projected_receive_indices[phase, target]
                directions = self.projected_receive_direction[phase, target]
                orientations = self.projected_receive_orientation[phase, target]
                selected = reconstructors[directions, orientations]
                payload = received[phase, target]
                flat = payload.reshape(payload.shape[:1] + (payload.shape[1], -1))
                reconstructed = contract("msh,mhq->msq", selected, flat, backend="jax")
                reconstructed = reconstructed.reshape(
                    (payload.shape[0], reconstructors.shape[2], *tail)
                )
                message_valid = received_valid[phase, target]
                for slot in range(self.projected_message_capacity):
                    value = jnp.where(message_valid[slot], reconstructed[slot], 0)
                    result = result.at[
                        target,
                        indices[slot],
                        directions[slot],
                        orientations[slot],
                    ].add(value)
                    valid = valid.at[
                        target,
                        indices[slot],
                        directions[slot],
                        orientations[slot],
                    ].set(message_valid[slot])
        return ProjectedSpinorHalo(
            result,
            valid,
            self.plan_id,
            packed.parity,
            False,
        )


class LatticeDecompositionPlan(StrictModule, NonTrainableState):
    """Immutable block decomposition of one finite Cartesian lattice."""

    ownership: LatticeOwnership
    halo: LatticeHaloPlan
    global_coordinates: Array
    neighbor_ids: Array
    neighbor_valid: Array
    owned_global_ids: Array
    owned_valid: Array
    owned_starts: Array
    owned_stops: Array
    global_shape: tuple[int, ...] = eqx.field(static=True)
    partition_shape: tuple[int, ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    owned_slices: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    site_count: int = eqx.field(static=True)
    partition_count: int = eqx.field(static=True)
    owned_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        global_shape: Sequence[int],
        partition_shape: Sequence[int],
        /,
        *,
        periodic: Sequence[bool] | None = None,
        maximum_global_sites: int = 16_777_216,
        maximum_local_sites: int = 4_194_304,
        maximum_halo_sites: int = 4_194_304,
    ) -> None:
        shape = _positive_shape(global_shape, "global_shape")
        partitions = _positive_shape(partition_shape, "partition_shape")
        if len(partitions) != len(shape) or any(
            p > n for p, n in zip(partitions, shape, strict=True)
        ):
            raise ValueError(
                "partition_shape must match rank and cannot split an axis into empty parts."
            )
        periodic_ = (
            (True,) * len(shape)
            if periodic is None
            else tuple(bool(value) for value in periodic)
        )
        if len(periodic_) != len(shape):
            raise ValueError("periodic must contain one flag per lattice axis.")
        maximum_global = int(maximum_global_sites)
        maximum_local = int(maximum_local_sites)
        maximum_halo = int(maximum_halo_sites)
        if min(maximum_global, maximum_local, maximum_halo) <= 0:
            raise ValueError("Lattice resource limits must be positive.")
        site_count = prod(shape)
        part_count = prod(partitions)
        axis_ranges = tuple(
            _partition_axis(extent, count)
            for extent, count in zip(shape, partitions, strict=True)
        )
        local_extents = [
            prod(stop - start for start, stop in block)
            for block in (
                tuple(
                    axis_ranges[axis][part_coordinate[axis]] for axis in range(len(shape))
                )
                for part_coordinate in np.ndindex(partitions)
            )
        ]
        if site_count > maximum_global or max(local_extents) > maximum_local:
            raise ValueError(
                "Lattice decomposition exceeds its configured site resource limit."
            )
        coordinates = np.asarray(tuple(np.ndindex(shape)), dtype=np.int32)
        owner = np.asarray(
            [
                _coordinate_owner(tuple(coordinate), axis_ranges, partitions)
                for coordinate in coordinates
            ],
            dtype=np.int32,
        )
        neighbors = np.zeros((site_count, len(shape), 2), dtype=np.int32)
        neighbor_valid = np.zeros_like(neighbors, dtype=np.bool_)
        adjacency: set[tuple[int, int]] = set()
        for global_id, coordinate_array in enumerate(coordinates):
            coordinate = tuple(coordinate_array)
            for axis in range(len(shape)):
                for orientation in range(2):
                    neighbor = _neighbor(coordinate, axis, orientation, shape, periodic_)
                    if neighbor is None:
                        continue
                    neighbor_id = int(np.ravel_multi_index(neighbor, shape, order="C"))
                    neighbors[global_id, axis, orientation] = neighbor_id
                    neighbor_valid[global_id, axis, orientation] = True
                    if neighbor_id != global_id:
                        adjacency.add(tuple(sorted((global_id, neighbor_id))))
        adjacency_array = np.asarray(sorted(adjacency), dtype=np.int32).reshape((-1, 2))
        halo_sets: list[set[int]] = [set() for _ in range(part_count)]
        for left, right in adjacency_array:
            left_part = int(owner[int(left)])
            right_part = int(owner[int(right)])
            if left_part != right_part:
                halo_sets[left_part].add(int(right))
                halo_sets[right_part].add(int(left))
        if max((len(values) for values in halo_sets), default=0) > maximum_halo:
            raise ValueError("Lattice decomposition exceeds maximum_halo_sites.")
        distributed = DistributedHaloPlan(owner, adjacency_array, part_count)
        face_axes = np.asarray(
            [
                (left, right)
                for left in range(len(shape))
                for right in range(left + 1, len(shape))
            ],
            dtype=np.int32,
        ).reshape((-1, 2))
        face_valid = np.zeros((site_count, face_axes.shape[0]), dtype=np.bool_)
        for site in range(site_count):
            for face, (left, right) in enumerate(face_axes):
                face_valid[site, face] = bool(
                    neighbor_valid[site, int(left), 1]
                    and neighbor_valid[site, int(right), 1]
                )
        face_owner = np.broadcast_to(owner[:, None], face_valid.shape).copy()
        face_owner[~face_valid] = -1
        ownership = LatticeOwnership(
            owner,
            np.broadcast_to(owner[:, None], (site_count, len(shape))).copy(),
            face_owner,
            face_valid,
            face_axes,
        )
        parity = np.mod(np.sum(coordinates, axis=1), 2).astype(np.int32)
        halo = LatticeHaloPlan(distributed, parity, neighbors, neighbor_valid)
        owned_counts = np.bincount(owner, minlength=part_count)
        owned_capacity = int(np.max(owned_counts))
        owned_ids = np.zeros((part_count, owned_capacity), dtype=np.int32)
        owned_valid = np.zeros_like(owned_ids, dtype=np.bool_)
        owned_slices = []
        starts = np.zeros((part_count, len(shape)), dtype=np.int32)
        stops = np.zeros_like(starts)
        for part, part_coordinate in enumerate(np.ndindex(partitions)):
            block = tuple(
                axis_ranges[axis][part_coordinate[axis]] for axis in range(len(shape))
            )
            owned_slices.append(block)
            starts[part] = [value[0] for value in block]
            stops[part] = [value[1] for value in block]
            ids = np.flatnonzero(owner == part)
            owned_ids[part, : ids.size] = ids
            owned_valid[part, : ids.size] = True
        self.ownership = ownership
        self.halo = halo
        self.global_coordinates = jnp.asarray(coordinates)
        self.neighbor_ids = jnp.asarray(neighbors)
        self.neighbor_valid = jnp.asarray(neighbor_valid)
        self.owned_global_ids = jnp.asarray(owned_ids)
        self.owned_valid = jnp.asarray(owned_valid)
        self.owned_starts = jnp.asarray(starts)
        self.owned_stops = jnp.asarray(stops)
        self.global_shape = shape
        self.partition_shape = partitions
        self.periodic = periodic_
        self.owned_slices = tuple(owned_slices)
        self.dimension = len(shape)
        self.site_count = site_count
        self.partition_count = part_count
        self.owned_capacity = owned_capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "global-cartesian-lattice-decomposition",
                "global_shape": shape,
                "partition_shape": partitions,
                "periodic": periodic_,
                "ownership": ownership.ownership_id,
                "halo": halo.plan_id,
                "owned_slices": owned_slices,
            }
        )

    def flatten_sites(self, values: ArrayLike, /) -> Array:
        array = jnp.asarray(values)
        if array.shape[: self.dimension] != self.global_shape:
            raise ValueError("Lattice field leading axes must match global_shape.")
        return array.reshape((self.site_count,) + array.shape[self.dimension :])

    def unflatten_sites(self, values: ArrayLike, /) -> Array:
        array = jnp.asarray(values)
        if array.shape[:1] != (self.site_count,):
            raise ValueError("Flattened lattice field must begin with site_count.")
        return array.reshape(self.global_shape + array.shape[1:])

    def pack_owned_sites(self, values: ArrayLike, /) -> Array:
        flat = jnp.asarray(values)
        if flat.shape[:1] != (self.site_count,):
            flat = self.flatten_sites(flat)
        return self.halo.distributed.pack_owned(flat)

    def pack_reference_sites(self, values: ArrayLike, /) -> Array:
        flat = jnp.asarray(values)
        if flat.shape[:1] != (self.site_count,):
            flat = self.flatten_sites(flat)
        return self.halo.distributed.pack_reference(flat)

    def unpack_owned_sites(self, local_values: ArrayLike, /) -> Array:
        return self.halo.distributed.unpack_owned(local_values)

    def assemble_owned_sites(self, owned_values: ArrayLike, /) -> Array:
        values = jnp.asarray(owned_values)
        expected = (self.partition_count, self.owned_capacity)
        if values.shape[:2] != expected:
            raise ValueError(f"owned_values must begin with shape {expected}.")
        result = jnp.zeros((self.site_count,) + values.shape[2:], dtype=values.dtype)
        for part in range(self.partition_count):
            ids = self.owned_global_ids[part]
            mask = self.owned_valid[part].reshape(
                (self.owned_capacity,) + (1,) * (values.ndim - 2)
            )
            result = result.at[ids].add(jnp.where(mask, values[part], 0))
        return result


class LatticeStencilExecutionPlan(StrictModule, NonTrainableState):
    """Static resource contract for explicit overlap of halo and stencil work."""

    decomposition: LatticeDecompositionPlan
    site_value_shape: tuple[int, ...] = eqx.field(static=True)
    dtype_name: str = eqx.field(static=True)
    parity: int = eqx.field(static=True)
    payload_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        decomposition: LatticeDecompositionPlan,
        site_value_shape: Sequence[int],
        dtype: object,
        /,
        *,
        parity: LatticeParity | None = None,
        maximum_payload_bytes: int = 1_073_741_824,
    ) -> None:
        if not isinstance(decomposition, LatticeDecompositionPlan):
            raise TypeError("decomposition must be LatticeDecompositionPlan.")
        value_shape = tuple(site_value_shape)
        if any(value <= 0 for value in value_shape):
            raise ValueError("site_value_shape extents must be positive.")
        dtype_ = np.dtype(dtype)
        if dtype_.hasobject:
            raise TypeError("Lattice stencil values cannot have object dtype.")
        parity_value = decomposition.halo._parity(parity)
        payload_bytes = (
            decomposition.halo.phase_count
            * decomposition.partition_count
            * decomposition.halo.distributed.message_capacity
            * prod(value_shape)
            * dtype_.itemsize
        )
        if int(maximum_payload_bytes) <= 0 or payload_bytes > int(maximum_payload_bytes):
            raise ValueError("Planned lattice halo exceeds maximum_payload_bytes.")
        self.decomposition = decomposition
        self.site_value_shape = value_shape
        self.dtype_name = dtype_.str
        self.parity = parity_value
        self.payload_bytes = payload_bytes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lattice-stencil-execution-plan",
                "decomposition": decomposition.plan_id,
                "site_value_shape": value_shape,
                "dtype": dtype_.str,
                "parity": parity_value,
                "payload_bytes": payload_bytes,
            }
        )


class PreparedLatticeStencilExecution(StrictModule, NonTrainableState):
    """Prepared interior/boundary actions with explicit five-phase execution."""

    plan: LatticeStencilExecutionPlan
    interior_action: Callable = eqx.field(static=True)
    boundary_action: Callable = eqx.field(static=True)
    interior_action_id: str = eqx.field(static=True)
    boundary_action_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: LatticeStencilExecutionPlan,
        interior_action: Callable,
        boundary_action: Callable,
        /,
        *,
        interior_action_id: str,
        boundary_action_id: str,
    ) -> None:
        if not isinstance(plan, LatticeStencilExecutionPlan):
            raise TypeError("plan must be LatticeStencilExecutionPlan.")
        if not callable(interior_action) or not callable(boundary_action):
            raise TypeError("Both interior and boundary actions must be callable.")
        interior_id = str(interior_action_id).strip()
        boundary_id = str(boundary_action_id).strip()
        if not interior_id or not boundary_id:
            raise ValueError("Stencil action identities must be non-empty.")
        self.plan = plan
        self.interior_action = interior_action
        self.boundary_action = boundary_action
        self.interior_action_id = interior_id
        self.boundary_action_id = boundary_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-lattice-stencil-execution",
                "plan": plan.plan_id,
                "interior_action": interior_id,
                "boundary_action": boundary_id,
            }
        )

    def pack(self, local_owned: ArrayLike, /) -> PackedLatticeHalo:
        parity = None if self.plan.parity < 0 else self.plan.parity
        return self.plan.decomposition.halo.pack(local_owned, parity=parity)

    def pack_rank(
        self,
        local_owned: ArrayLike,
        partition: ArrayLike,
        /,
    ) -> PackedLatticeHalo:
        parity = None if self.plan.parity < 0 else self.plan.parity
        return self.plan.decomposition.halo.pack_rank(
            local_owned,
            partition,
            parity=parity,
        )

    def start(
        self,
        packed: PackedLatticeHalo,
        partition: ArrayLike,
        /,
        *,
        axis_name: str,
    ) -> StartedLatticeHalo:
        return self.plan.decomposition.halo.start(
            packed,
            partition,
            axis_name=axis_name,
        )

    def interior_rank(
        self,
        local_owned: ArrayLike,
        partition: ArrayLike,
        /,
    ) -> Array:
        values = jnp.asarray(local_owned)
        decomposition = self.plan.decomposition
        part = jnp.asarray(partition, dtype=jnp.int32)
        active = decomposition.halo.local_interior[part]
        if self.plan.parity >= 0:
            active = active & (
                decomposition.halo.local_site_parity[part] == self.plan.parity
            )
        return self.interior_action(
            part,
            values,
            decomposition.halo.distributed.local_global_ids[part],
            active,
        )

    def complete_rank(
        self,
        started: StartedLatticeHalo,
        local_owned: ArrayLike,
        partition: ArrayLike,
        /,
    ) -> Array:
        return self.plan.decomposition.halo.complete(
            started,
            local_owned,
            partition=partition,
        )

    def boundary_rank(
        self,
        completed: ArrayLike,
        interior_result: ArrayLike,
        partition: ArrayLike,
        /,
    ) -> Array:
        values = jnp.asarray(completed)
        interior = jnp.asarray(interior_result)
        if values.shape != interior.shape:
            raise ValueError("Completed values and interior result shapes must agree.")
        decomposition = self.plan.decomposition
        part = jnp.asarray(partition, dtype=jnp.int32)
        active = decomposition.halo.local_boundary[part]
        if self.plan.parity >= 0:
            active = active & (
                decomposition.halo.local_site_parity[part] == self.plan.parity
            )
        boundary = self.boundary_action(
            part,
            values,
            decomposition.halo.distributed.local_global_ids[part],
            active,
        )
        mask = active.reshape(active.shape + (1,) * (values.ndim - 1))
        return jnp.where(mask, boundary, interior)

    def start_reference(self, packed: PackedLatticeHalo, /) -> StartedLatticeHalo:
        return self.plan.decomposition.halo.start_reference(packed)

    def interior(self, local_owned: ArrayLike, /) -> Array:
        values = jnp.asarray(local_owned)
        decomposition = self.plan.decomposition
        outputs = []
        for part in range(decomposition.partition_count):
            active = decomposition.halo.local_interior[part]
            if self.plan.parity >= 0:
                active = active & (
                    decomposition.halo.local_site_parity[part] == self.plan.parity
                )
            outputs.append(
                self.interior_action(
                    jnp.asarray(part, dtype=jnp.int32),
                    values[part],
                    decomposition.halo.distributed.local_global_ids[part],
                    active,
                )
            )
        return jnp.stack(outputs)

    def complete(self, started: StartedLatticeHalo, local_owned: ArrayLike, /) -> Array:
        return self.plan.decomposition.halo.complete(started, local_owned)

    def boundary(self, completed: ArrayLike, interior_result: ArrayLike, /) -> Array:
        values = jnp.asarray(completed)
        interior = jnp.asarray(interior_result)
        if values.shape != interior.shape:
            raise ValueError("Completed values and interior result shapes must agree.")
        decomposition = self.plan.decomposition
        outputs = []
        for part in range(decomposition.partition_count):
            active = decomposition.halo.local_boundary[part]
            if self.plan.parity >= 0:
                active = active & (
                    decomposition.halo.local_site_parity[part] == self.plan.parity
                )
            boundary = self.boundary_action(
                jnp.asarray(part, dtype=jnp.int32),
                values[part],
                decomposition.halo.distributed.local_global_ids[part],
                active,
            )
            mask = active.reshape(active.shape + (1,) * (values.ndim - 2))
            outputs.append(jnp.where(mask, boundary, interior[part]))
        return jnp.stack(outputs)

    def reference(self, global_values: ArrayLike, /) -> Array:
        decomposition = self.plan.decomposition
        local = decomposition.pack_owned_sites(global_values)
        packed = self.pack(local)
        started = self.start_reference(packed)
        interior = self.interior(local)
        completed = self.complete(started, local)
        return decomposition.unpack_owned_sites(self.boundary(completed, interior))


def prepare_lattice_stencil_execution(
    plan: LatticeStencilExecutionPlan,
    interior_action: Callable,
    boundary_action: Callable,
    /,
    *,
    interior_action_id: str,
    boundary_action_id: str,
) -> PreparedLatticeStencilExecution:
    return PreparedLatticeStencilExecution(
        plan,
        interior_action,
        boundary_action,
        interior_action_id=interior_action_id,
        boundary_action_id=boundary_action_id,
    )


__all__ = [
    "LatticeDecompositionPlan",
    "LatticeHaloPlan",
    "LatticeOwnership",
    "LatticeParity",
    "LatticeStencilExecutionPlan",
    "PackedLatticeHalo",
    "PreparedLatticeStencilExecution",
    "ProjectedSpinorHalo",
    "StartedLatticeHalo",
    "prepare_lattice_stencil_execution",
]
