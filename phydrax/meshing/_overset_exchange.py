#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-epoch owner-local overset field traffic, without global field materialization.

Connectivity and field-query preparation retain their existing owners. This owner
exchanges only unique scalar coefficient support of locally owned receptors. Its
bounded request/response routes are reused for values and reversed for transpose
contributions. Interpolation remains nonconservative. Nonlinear FV routes whose
owner computes a global reconstruction do not admit this sparse linear profile.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import prod
from typing import assert_never, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, PartitionSpec
from jax.typing import ArrayLike

from .._execution_runtime import ExecutionGroup
from .._fingerprint import canonical_fingerprint
from .._interpolation import GatherStencil
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._distributed_field import (
    _color_directed_pairs,
    _exchange_owner_local_packets,
    make_owner_local_field_array,
)
from ..discretization._field_query import PreparedFieldQuery
from ..discretization.fem._point_interpolation import (
    FiniteElementFieldReconstructionKernel,
)
from ..discretization.finite_volume._field_view import (
    UnstructuredFiniteVolumeFieldReconstructionKernel,
)
from ._coupling import OversetCoupling
from ._overset import PreparedOversetFieldTransfer


def _query_scalar_stencil(query: PreparedFieldQuery, /) -> GatherStencil:
    if not query.complete or not query.coefficient_linear:
        raise ValueError(
            "Owner-local overset requires complete coefficient-linear fixed queries."
        )
    kernel = query.reconstruction.kernel
    if isinstance(kernel, FiniteElementFieldReconstructionKernel):
        return kernel.scalar_query_stencil(query.route)
    if not isinstance(
        kernel, UnstructuredFiniteVolumeFieldReconstructionKernel
    ) or not isinstance(query.route, GatherStencil):
        raise ValueError(
            "This field owner has no bounded scalar coefficient-read stencil."
        )
    route = query.route
    components = prod(query.coefficient_shape[1:])
    indices = np.asarray(route.indices)
    weights = np.asarray(route.weights)
    valid = np.asarray(route.valid) & (weights != 0)
    shape = (indices.shape[0], components, indices.shape[1])
    scalar_indices = np.broadcast_to(
        indices[:, None, :] * components
        + np.arange(components, dtype=np.int32)[None, :, None],
        shape,
    ).reshape((-1, indices.shape[1]))
    return GatherStencil(
        indices=scalar_indices,
        weights=np.broadcast_to(weights[:, None, :], shape).reshape(scalar_indices.shape),
        valid=np.broadcast_to(valid[:, None, :], shape).reshape(scalar_indices.shape),
        source_size=prod(query.coefficient_shape),
    )


def _scalar_stencil(
    query: PreparedFieldQuery, coupling: OversetCoupling, /
) -> GatherStencil:
    """Fold only the overlay's declared scientific value action into its route."""
    if coupling.field_query is None or coupling.field_query.query_id != query.query_id:
        raise ValueError(
            "Sparse overset stencils must retain their canonical field overlay."
        )
    stencil = _query_scalar_stencil(query)
    match coupling.value_action:
        case "invariant":
            return stencil
        case "polar-vector" | "contravariant-vector":
            if coupling.rotation is None:
                return stencil
            rotation = np.asarray(coupling.rotation)
            dimension = rotation.shape[0]
            points, width = query.admitted_count, stencil.indices.shape[1]
            shape = (points, dimension, dimension, width)
            indices = np.asarray(stencil.indices).reshape((points, dimension, width))
            weights = np.asarray(stencil.weights).reshape(indices.shape)
            valid = np.asarray(stencil.valid).reshape(indices.shape)
            # The source-component axis expands only the local stencil width,
            # never the donor's complete coefficient extent.
            mixed_indices = np.broadcast_to(indices[:, None, :, :], shape)
            mixed_weights = rotation[None, :, :, None] * weights[:, None, :, :]
            mixed_valid = np.broadcast_to(valid[:, None, :, :], shape) & (
                mixed_weights != 0
            )
            output_shape = (points * dimension, dimension * width)
            return GatherStencil(
                indices=mixed_indices.reshape(output_shape),
                weights=mixed_weights.reshape(output_shape),
                valid=mixed_valid.reshape(output_shape),
                source_size=stencil.source_size,
            )
        case _:
            assert_never(coupling.value_action)


def _place(
    value: ArrayLike, rank: int, devices: tuple[jax.Device, ...], axis: str, /
) -> Array:
    return make_owner_local_field_array((value,), (rank,), devices, axis_name=axis)


def _local(value: Array, /) -> np.ndarray:
    if len(value.addressable_shards) != 1:
        raise ValueError(
            "Overset exchange requires exactly one process-local owner bucket."
        )
    return np.asarray(value.addressable_shards[0].data)[0]


def _admit_capacities(
    sizes: np.ndarray,
    identity: str,
    bound: int,
    rank: int,
    devices: tuple[jax.Device, ...],
    axis: str,
    /,
) -> tuple[int, ...]:
    """Agree on the epoch and allocate only actual sparse-route maxima."""
    digest = np.asarray(
        [int(identity[start : start + 7], 16) for start in range(0, 64, 7)],
        dtype=np.int32,
    )
    mesh = Mesh(np.asarray(devices, dtype=object), (axis,))
    spec = PartitionSpec(axis)

    def admit(local_sizes: Array, fingerprint: Array) -> tuple[Array, Array]:
        agreement = jnp.all(
            jax.lax.pmin(fingerprint[0], axis) == jax.lax.pmax(fingerprint[0], axis)
        )
        return jax.lax.pmax(local_sizes[0], axis), agreement

    maxima, agreement = jax.shard_map(
        admit,
        mesh=mesh,
        in_specs=(spec, spec),
        out_specs=(PartitionSpec(), PartitionSpec()),
        check_vma=False,
    )(_place(sizes, rank, devices, axis), _place(digest, rank, devices, axis))
    capacities = np.asarray(maxima)
    if not bool(np.asarray(agreement)) or capacities[4] > bound:
        raise ValueError(
            "Collective overset epoch agreement or sparse request bound failed."
        )
    return tuple(max(int(value), 1) for value in capacities)


def _prepare_request_routes(
    requests: list[tuple[tuple[int, int], ...]],
    support_lookup: dict[tuple[int, int], int],
    owned_lookup: dict[tuple[int, int], int],
    width: int,
    rank: int,
    devices: tuple[jax.Device, ...],
    axis: str,
    permutations: tuple[tuple[tuple[int, int], ...], ...],
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve real peer stable-ID requests only against locally owned rows."""
    request_packet = np.full((len(permutations), width, 2), -1, dtype=np.int64)
    receive = np.zeros((len(permutations), width), dtype=np.int32)
    receive_valid = np.zeros(receive.shape, dtype=np.bool_)
    for phase, request in enumerate(requests):
        for column, key in enumerate(request):
            request_packet[phase, column] = key
            receive[phase, column] = support_lookup[key]
            receive_valid[phase, column] = True
    mesh, spec = Mesh(np.asarray(devices, dtype=object), (axis,)), PartitionSpec(axis)
    incoming = _local(
        _exchange_owner_local_packets(
            _place(request_packet, rank, devices, axis),
            permutations,
            mesh,
            axis,
        )
    )
    send = np.zeros(receive.shape, dtype=np.int32)
    # Nonparticipants receive ppermute's zero fill, not a stable-ID request.
    destinations = np.asarray(
        [any(target == rank for _, target in phase) for phase in permutations],
        dtype=np.bool_,
    )
    send_valid = (incoming[:, :, 0] >= 0) & destinations[:, None]
    missing = False
    for phase, column in zip(*np.nonzero(send_valid), strict=True):
        key = (int(incoming[phase, column, 0]), int(incoming[phase, column, 1]))
        row = owned_lookup.get(key)
        if row is None:
            missing = True
        else:
            send[phase, column] = row
    failed = jax.shard_map(
        lambda flag: jax.lax.pmax(flag[0].astype(jnp.int32), axis),
        mesh=mesh,
        in_specs=spec,
        out_specs=PartitionSpec(),
        check_vma=False,
    )(_place(jnp.asarray(missing), rank, devices, axis))
    if bool(np.asarray(failed)):
        raise ValueError(
            "A requested stable coefficient ID is absent from its actual donor owner."
        )
    return send, send_valid, receive, receive_valid


@final
class PreparedOwnerLocalOversetExchange(StrictModule, NonTrainableState):
    """Actual device-placed sparse support routes of one native overset epoch.

    One execution-group device owns each process/partition. ``coefficient_ids``
    explicitly names each scalar coefficient in its actual FE/FV flattened
    order, scoped by donor field and revision; numbering is never inferred from
    geometry vertices. Supply IDs only for this rank's donor fields and the
    fields referenced by its receptor queries. Dynamic calls accept only locally
    owned donor arrays and return only locally owned receptor outputs.

    Local geometry/query metadata may have been prepared serially. This API does
    not claim distributed hole cutting or distributed donor search.
    """

    send_indices: Array
    send_valid: Array
    receive_indices: Array
    receive_valid: Array
    direct_indices: Array
    direct_valid: Array
    route_indices: Array
    route_weights: Array
    permutations: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(static=True)
    reverse_permutations: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(static=True)
    devices: tuple[jax.Device, ...] = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    partition_index: int = eqx.field(static=True)
    donor_layouts: tuple[tuple[str, tuple[int, ...], int], ...] = eqx.field(static=True)
    receptor_layouts: tuple[tuple[str, tuple[int, ...], int], ...] = eqx.field(
        static=True
    )
    owned_capacity: int = eqx.field(static=True)
    support_capacity: int = eqx.field(static=True)
    output_capacity: int = eqx.field(static=True)
    message_capacity: int = eqx.field(static=True)
    requested_support_count: int = eqx.field(static=True)
    served_support_count: int = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)
    exchange_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: PreparedOversetFieldTransfer,
        coefficient_ids: Mapping[str, ArrayLike],
        execution_group: ExecutionGroup,
        /,
        *,
        axis_name: str,
        message_capacity: int,
    ) -> None:
        devices = execution_group.devices
        rank, parts = jax.process_index(), jax.process_count()
        axis = str(axis_name).strip()
        bound = int(message_capacity)
        if (
            not axis
            or bound < 1
            or len(devices) != parts
            or tuple(device.process_index for device in devices) != tuple(range(parts))
        ):
            raise ValueError(
                "Overset placement requires one actual execution-group device per process and a positive packet bound."
            )
        names = transfer.donor_names
        owners = {spec.part_name: spec.owner_rank for spec in transfer.connectivity.specs}
        if any(not 0 <= owner < parts for owner in owners.values()):
            raise ValueError(
                "Donor/fringe packet ownership must bind actual execution partitions."
            )
        if any(
            packet.donor_rank != owners[packet.donor_part]
            or packet.receptor_rank != owners[packet.receptor_part]
            for packet in transfer.packets
        ):
            raise ValueError(
                "A donor packet disagrees with its registered donor/fringe owner."
            )
        fields = {
            name: (index, shape)
            for index, (name, shape) in enumerate(
                zip(names, transfer.coefficient_shapes, strict=True)
            )
        }
        packets = tuple(
            (packet, query, coupling)
            for packet, query, coupling in zip(
                transfer.packets, transfer.queries, transfer.field_couplings, strict=True
            )
            if packet.receptor_rank == rank
        )
        required_names = {name for name in names if owners[name] == rank} | {
            packet.donor_part for packet, _, _ in packets
        }
        if set(coefficient_ids) != required_names:
            raise ValueError(
                "Stable coefficient IDs must cover exactly local donor fields and local receptor query support fields."
            )
        ids = {}
        for name in sorted(required_names):
            values = np.asarray(coefficient_ids[name])
            if (
                values.shape != (prod(fields[name][1]),)
                or not np.issubdtype(values.dtype, np.integer)
                or np.any(values < 0)
                or np.any(values > np.iinfo(np.int64).max)
                or np.unique(values).size != values.size
            ):
                raise ValueError(
                    "Each field requires unique nonnegative int64 stable scalar IDs in exact flattened coefficient order."
                )
            ids[name] = values.astype(np.int64)
        owned_keys, donor_layouts = [], []
        for name in names:
            if owners[name] == rank:
                donor_layouts.append((name, fields[name][1], len(owned_keys)))
                owned_keys.extend(
                    (fields[name][0], int(identifier)) for identifier in ids[name]
                )
        owned_lookup = {key: row for row, key in enumerate(owned_keys)}
        support_keys: set[tuple[int, int]] = set()
        stencils = []
        for packet, query, coupling in packets:
            stencil = _scalar_stencil(query, coupling)
            rows = np.asarray(stencil.indices)
            valid = np.asarray(stencil.valid)
            field = fields[packet.donor_part][0]
            keys = {
                (field, int(ids[packet.donor_part][row]))
                for row in np.unique(rows[valid])
            }
            support_keys.update(keys)
            stencils.append((packet, query, stencil))
        support = tuple(sorted(support_keys))
        support_lookup = {key: row for row, key in enumerate(support)}
        requests = []
        permutations = _color_directed_pairs(
            (packet.receptor_rank, packet.donor_rank)
            for packet in transfer.packets
            if packet.receptor_rank != packet.donor_rank
        )
        for phase in permutations:
            outgoing = tuple(target for source, target in phase if source == rank)
            requests.append(
                tuple(
                    key
                    for key in support
                    if outgoing and owners[names[key[0]]] == outgoing[0]
                )
            )
        receptor_layouts, output_count = [], 0
        receptor_offsets = {}
        for evidence, spec in zip(
            transfer.receptors, transfer.connectivity.specs, strict=True
        ):
            if spec.owner_rank == rank:
                shape = (evidence.receptor_ids.size, *transfer.value_shape)
                receptor_layouts.append((evidence.part_name, shape, output_count))
                receptor_offsets[evidence.part_name] = output_count
                output_count += prod(shape)
        local_sizes = np.asarray(
            [
                len(owned_keys),
                len(support),
                output_count,
                max((stencil.indices.shape[1] for _, _, stencil in stencils), default=1),
                max((len(request) for request in requests), default=0),
            ],
            dtype=np.int64,
        )
        identity = canonical_fingerprint(
            {
                "transfer": transfer.transfer_id,
                "fields": fields,
                "owners": owners,
                "parts": parts,
                "axis": axis,
                "bound": bound,
            }
        )
        owned_capacity, support_capacity, output_capacity, route_width, packet_width = (
            _admit_capacities(local_sizes, identity, bound, rank, devices, axis)
        )
        send, send_valid, receive, receive_valid = _prepare_request_routes(
            requests,
            support_lookup,
            owned_lookup,
            packet_width,
            rank,
            devices,
            axis,
            permutations,
        )
        direct = np.zeros(support_capacity, dtype=np.int32)
        direct_valid = np.zeros(support_capacity, dtype=np.bool_)
        for row, key in enumerate(support):
            if key in owned_lookup:
                direct[row], direct_valid[row] = owned_lookup[key], True
        route_indices = np.zeros((output_capacity, route_width), dtype=np.int32)
        route_weights = np.zeros(route_indices.shape, dtype=np.float64)
        output_covered = np.zeros(output_count, dtype=np.bool_)
        value_count = prod(transfer.value_shape)
        for packet, query, stencil in stencils:
            destinations = receptor_offsets[packet.receptor_part] + (
                np.asarray(packet.receptor_slots)[:, None] * value_count
                + np.arange(value_count, dtype=np.int32)[None, :]
            ).reshape(-1)
            if np.any(output_covered[destinations]):
                raise ValueError(
                    "Overset fixed routes assign a receptor scalar more than once."
                )
            output_covered[destinations] = True
            field = fields[packet.donor_part][0]
            indices, weights, valid = (
                np.asarray(stencil.indices),
                np.asarray(stencil.weights),
                np.asarray(stencil.valid),
            )
            for output, column in zip(*np.nonzero(valid), strict=True):
                route_indices[destinations[output], column] = support_lookup[
                    field, int(ids[packet.donor_part][indices[output, column]])
                ]
                route_weights[destinations[output], column] = weights[output, column]
        if not np.all(output_covered):
            raise ValueError(
                "Fixed donor routes must cover every locally owned receptor scalar."
            )
        self.send_indices, self.send_valid = (
            _place(send, rank, devices, axis),
            _place(send_valid, rank, devices, axis),
        )
        self.receive_indices, self.receive_valid = (
            _place(receive, rank, devices, axis),
            _place(receive_valid, rank, devices, axis),
        )
        self.direct_indices, self.direct_valid = (
            _place(direct, rank, devices, axis),
            _place(direct_valid, rank, devices, axis),
        )
        self.route_indices, self.route_weights = (
            _place(route_indices, rank, devices, axis),
            _place(route_weights, rank, devices, axis),
        )
        self.permutations = tuple(
            tuple((target, source) for source, target in phase) for phase in permutations
        )
        self.reverse_permutations = permutations
        self.devices, self.axis_name, self.partition_index = devices, axis, rank
        self.donor_layouts, self.receptor_layouts = (
            tuple(donor_layouts),
            tuple(receptor_layouts),
        )
        self.owned_capacity, self.support_capacity, self.output_capacity = (
            owned_capacity,
            support_capacity,
            output_capacity,
        )
        self.message_capacity = packet_width
        self.requested_support_count, self.served_support_count = (
            int(np.count_nonzero(receive_valid)),
            int(np.count_nonzero(send_valid)),
        )
        self.transfer_id = transfer.transfer_id
        self.exchange_id = canonical_fingerprint(
            {
                "epoch": identity,
                "rank": rank,
                "owned": owned_keys,
                "support": support,
                "send": send,
                "receive": receive,
                "routes": route_indices,
                "weights": route_weights,
            }
        )

    def apply(self, coefficients: Mapping[str, ArrayLike], /) -> dict[str, Array]:
        """Request only fixed unique support values; return local fringe outputs."""
        values = self._pack(coefficients, self.donor_layouts, self.owned_capacity)
        result = _owner_local_overset_action(self, values, False)
        local = result.addressable_shards[0].data[0]
        return {
            name: local[offset : offset + prod(shape)].reshape(shape)
            for name, shape, offset in self.receptor_layouts
        }

    def transpose(self, cotangents: Mapping[str, ArrayLike], /) -> dict[str, Array]:
        """Return exact fixed-route contributions to their real donor owners."""
        values = self._pack(cotangents, self.receptor_layouts, self.output_capacity)
        result = _owner_local_overset_action(self, values, True)
        local = result.addressable_shards[0].data[0]
        return {
            name: local[offset : offset + prod(shape)].reshape(shape)
            for name, shape, offset in self.donor_layouts
        }

    def _pack(
        self,
        values: Mapping[str, ArrayLike],
        layouts: tuple[tuple[str, tuple[int, ...], int], ...],
        capacity: int,
        /,
    ) -> Array:
        if set(values) != {name for name, _, _ in layouts}:
            raise ValueError(
                "Dynamic overset values must name exactly this process's owned fields."
            )
        arrays = {name: jnp.asarray(value) for name, value in values.items()}
        dtype = jnp.result_type(jnp.float64, *(array.dtype for array in arrays.values()))
        pieces = []
        for name, shape, _ in layouts:
            array = arrays[name]
            if array.shape != shape or not array.is_fully_addressable:
                raise ValueError(
                    "Overset fields must retain their local owned shape and be process-addressable."
                )
            if array.size:
                pieces.append(array.reshape(-1).astype(dtype))
        if pieces:
            packed = pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces)
            if packed.shape[0] < capacity:
                packed = jnp.pad(packed, (0, capacity - packed.shape[0]))
        else:
            packed = jnp.zeros((capacity,), dtype=dtype)
        return _place(packed, self.partition_index, self.devices, self.axis_name)


@eqx.filter_jit
def _owner_local_overset_action(
    exchange: PreparedOwnerLocalOversetExchange,
    local: Array,
    transpose: bool,
    /,
) -> Array:
    """Compiled sparse action with numerical exchange leaves kept dynamic."""
    axis, spec = exchange.axis_name, PartitionSpec(exchange.axis_name)
    mesh = Mesh(np.asarray(exchange.devices, dtype=object), (axis,))

    def action(
        values: Array,
        send: Array,
        send_valid: Array,
        receive: Array,
        receive_valid: Array,
        direct: Array,
        direct_valid: Array,
        indices: Array,
        weights: Array,
    ) -> Array:
        (
            values,
            send,
            send_valid,
            receive,
            receive_valid,
            direct,
            direct_valid,
            indices,
            weights,
        ) = (
            item[0]
            for item in (
                values,
                send,
                send_valid,
                receive,
                receive_valid,
                direct,
                direct_valid,
                indices,
                weights,
            )
        )
        if transpose:
            support = (
                jnp.zeros((exchange.support_capacity,), dtype=values.dtype)
                .at[indices]
                .add(weights * values[:, None])
            )
            owned = (
                jnp.zeros((exchange.owned_capacity,), dtype=values.dtype)
                .at[direct]
                .add(jnp.where(direct_valid, support, 0))
            )
            for phase in reversed(range(len(exchange.reverse_permutations))):
                payload = jnp.where(receive_valid[phase], support[receive[phase]], 0)
                response = jax.lax.ppermute(
                    payload, axis, exchange.reverse_permutations[phase]
                )
                owned = owned.at[send[phase]].add(
                    jnp.where(send_valid[phase], response, 0)
                )
            return owned[None, :]
        support = jnp.where(direct_valid, values[direct], 0)
        for phase, permutation in enumerate(exchange.permutations):
            payload = jnp.where(send_valid[phase], values[send[phase]], 0)
            response = jax.lax.ppermute(payload, axis, permutation)
            support = support.at[receive[phase]].add(
                jnp.where(receive_valid[phase], response, 0)
            )
        return jnp.sum(weights * support[indices], axis=1)[None, :]

    return jax.shard_map(
        action, mesh=mesh, in_specs=(spec,) * 9, out_specs=spec, check_vma=False
    )(
        local,
        exchange.send_indices,
        exchange.send_valid,
        exchange.receive_indices,
        exchange.receive_valid,
        exchange.direct_indices,
        exchange.direct_valid,
        exchange.route_indices,
        exchange.route_weights,
    )
