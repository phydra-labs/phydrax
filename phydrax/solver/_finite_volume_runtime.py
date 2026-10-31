#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._differentiation import BranchDifferentiationPolicy
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._trainable import fixed_field, NonTrainableState
from ..discretization import TopologyEpoch
from ..discretization._conservation_ledger import (
    AcceptedConservationFluxIntegralBlock,
    AcceptedConservationIntegralLedger,
    ConservationStageFluxRateBlock,
    ConservationStageLedger,
)
from ..discretization.finite_volume import (
    AbstractNumericalFluxPlan,
    AbstractWavePropagationPlan,
    ConservativeSmallCellRedistributionPlan,
    ConservativeSmallCellRedistributionReport,
    DyadicFiniteVolumeDiscretization,
    FiniteVolumeAdmissibilityReport,
    FiniteVolumeMethodPlan,
    FiniteVolumePrecisionPolicy,
    FluxPositivityPlan,
    lower_embedded_stage_metrics,
    lower_static_unstructured_stage_metrics,
    PeriodicSlidingCoupling,
    PeriodicSlidingInterfacePlan,
    PeriodicSlidingRefreshArtifact,
    PiecewiseConstantReconstruction,
    PreparedFiniteVolumeDynamics,
    PreparedTriangleFiniteVolumeDynamics,
    PreparedUnstructuredFiniteVolumeDynamics,
    TriangleFiniteVolumeDiscretization,
    UnstructuredEmbeddedBoundarySet,
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumePlan,
)
from ..discretization.finite_volume._geometry_protocol import (
    FiniteVolumeGeometryStatus,
    FiniteVolumeStageMetrics,
)
from ..discretization.finite_volume._positivity import (
    BalancedPositivityBlendResult,
    PositivityBlendResult,
)
from ..discretization.finite_volume._shallow_water import (
    PreparedShallowWaterBathymetry,
    ShallowWaterAcceptedFaceIntegrals,
    ShallowWaterBalancedFaceResult,
    ShallowWaterHydrostaticHLLPlan,
)
from ..discretization.finite_volume._unstructured_motion import (
    UnstructuredALEStepGeometry,
)
from ..lifecycle import CompositionRebindReceipt
from ..typing import checked
from ._finite_volume import (
    collective_branch,
    FiniteVolumeStageDecisionProvider,
    FiniteVolumeStageRouteProvider,
    FiniteVolumeStageStateProvider,
    unstructured_ale_ssprk33_candidate,
    unstructured_ssprk33_content_candidate,
    zero_unstructured_ale_ssprk33_candidate,
    zero_unstructured_ssprk33_content_candidate,
)
from ._finite_volume_content import FiniteVolumeConservativeContentState
from ._finite_volume_topology_events import (
    _OwnerLocalRepartitionTransfer,
    FiniteVolumeContentTransferSemantics,
    FiniteVolumeTopologyArtifacts,
    FiniteVolumeTopologyEventJournal,
    FiniteVolumeTopologyEventRequest,
    FiniteVolumeTopologyEventScheduler,
    TopologyEventKind,
    TopologyEventStatus,
)


_TreeT = TypeVar("_TreeT")


@dataclass(frozen=True, slots=True)
class FiniteVolumeOwnerLocalCellStateBankSpec:
    """Declare stable-ID rows along one array axis of an owner-local state entry.

    ``cell_ids`` labels either owned rows or every resident owned/ghost row in
    the source entry. The target entry stores the migrated owned rows only.
    """

    entry_id: str
    cell_ids: ArrayLike
    cell_axis: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.entry_id, str)
            or not self.entry_id
            or self.entry_id != self.entry_id.strip()
            or isinstance(self.cell_axis, bool)
            or not isinstance(self.cell_axis, (int, np.integer))
        ):
            raise ValueError("Owner-local cell-state bank schema is malformed.")
        raw_ids = np.asarray(self.cell_ids)
        if (
            raw_ids.dtype.kind not in "iu"
            or raw_ids.ndim != 1
            or (
                raw_ids.dtype.kind == "u"
                and raw_ids.size
                and int(np.max(raw_ids)) > np.iinfo(np.int64).max
            )
            or (raw_ids.size and np.any(raw_ids < 0))
            or np.unique(raw_ids).size != raw_ids.size
        ):
            raise ValueError("Owner-local cell-state bank IDs must be unique int64 keys.")
        object.__setattr__(
            self, "cell_ids", tuple(int(value) for value in raw_ids.tolist())
        )
        object.__setattr__(self, "cell_axis", int(self.cell_axis))


def _owner_local_view_types() -> tuple[type, ...]:
    """Owner-local FV layouts: one part view, or a disjoint multi-part composite."""
    from ..discretization._view_support import (
        OwnerLocalCellGeometry,
        OwnerLocalCompositeCellGeometry,
    )

    return (OwnerLocalCellGeometry, OwnerLocalCompositeCellGeometry)


def _prepare_owner_local_fv_stage_state_provider(
    owner_local_view: Any,
    execution_group: Any,
    plan_cell_ids: Any,
    /,
    *,
    axis_name: str,
    required_stencil_depth: int,
) -> tuple[FiniteVolumeStageStateProvider, Array, Any]:
    """Prepare sparse per-stage owner-to-ghost exchange for the local cell axis."""
    import jax

    from .._execution_runtime import ExecutionGroup
    from ..discretization._distributed_field import DistributedHaloPlan
    from ..meshing._distribution import (
        _owner_exchange_message_banks,
        _owner_group_allgather,
        _owner_pair_channel_plan,
    )

    if not isinstance(owner_local_view, _owner_local_view_types()):
        raise TypeError("owner_local_view must be an owner-local FV cell layout.")
    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be ExecutionGroup.")
    axis = str(axis_name).strip()
    if not axis:
        raise ValueError("owner-local FV halo axis_name must be non-empty.")
    members = execution_group.spec.process_indices
    local_rank = owner_local_view.owner_rank
    if (
        not execution_group.is_member
        or len(execution_group.devices) != len(members)
        or tuple(device.process_index for device in execution_group.devices) != members
        or not 0 <= local_rank < len(members)
        or members[local_rank] != jax.process_index()
    ):
        raise ValueError("Owner-local FV halo view is not on its accepted group rank.")
    local_ids = np.asarray(owner_local_view.cell_ids, dtype=np.int64)
    local_owners = np.asarray(owner_local_view.cell_owner_ranks, dtype=np.int32)
    local_owned = np.asarray(owner_local_view.owned, dtype=np.bool_)
    if (
        local_owners.shape != local_ids.shape
        or local_owned.shape != local_ids.shape
        or np.unique(local_ids).size != local_ids.size
        or np.any(local_owners < 0)
        or np.any(local_owners >= len(members))
        or np.any(local_owned != (local_owners == local_rank))
    ):
        raise ValueError("Owner-local FV cell ownership rows are malformed.")
    requested_plan_ids = np.asarray(plan_cell_ids, dtype=np.int64)
    if requested_plan_ids.shape != local_ids.shape or set(
        int(value) for value in requested_plan_ids
    ) != set(int(value) for value in local_ids):
        raise ValueError("Owner-local FV plan cells differ from the accepted local view.")
    view_row_by_id = {int(identifier): row for row, identifier in enumerate(local_ids)}
    plan_to_view = np.asarray(
        [view_row_by_id[int(identifier)] for identifier in requested_plan_ids],
        dtype=np.int32,
    )
    owner_row_by_id = {
        int(identifier): row
        for row, identifier in enumerate(local_ids)
        if local_owned[row]
    }
    # The view's face-neighbor CSR is keyed by its sorted owned cells; ghost
    # rows are resident carriers without their own neighbor rows.
    graph_ids = np.asarray(owner_local_view.owned_cell_ids, dtype=np.int64)
    offsets = np.asarray(owner_local_view.cell_neighbor_offsets, dtype=np.int64)
    neighbor_ids = np.asarray(owner_local_view.cell_neighbor_ids, dtype=np.int64)
    if (
        offsets.shape != (graph_ids.size + 1,)
        or offsets[0] != 0
        or offsets[-1] != neighbor_ids.size
        or np.any(np.diff(offsets) < 0)
        or set(int(value) for value in graph_ids)
        != set(int(value) for value in local_ids[local_owned])
    ):
        raise ValueError("Owner-local FV reconstruction-neighbor routes are malformed.")
    graph_row_by_id = {int(identifier): row for row, identifier in enumerate(graph_ids)}
    if (
        isinstance(required_stencil_depth, bool)
        or not isinstance(required_stencil_depth, (int, np.integer))
        or required_stencil_depth < 1
    ):
        raise ValueError("Owner-local FV stencil depth must be a positive integer.")
    resident = set(int(value) for value in local_ids)
    frontier = {int(local_ids[row]) for row in np.flatnonzero(local_owned)}
    reached = set(frontier)
    for _ in range(int(required_stencil_depth)):
        next_frontier: set[int] = set()
        for identifier in frontier:
            row = graph_row_by_id.get(identifier)
            if row is None:
                raise ValueError(
                    "Owner-local FV view lacks neighbor rows for a ghost cell inside "
                    "the reconstruction-stencil depth."
                )
            neighbors = neighbor_ids[offsets[row] : offsets[row + 1]]
            missing = [int(value) for value in neighbors if int(value) not in resident]
            if missing:
                raise ValueError(
                    "Owner-local FV view omits required reconstruction-stencil halo cells."
                )
            next_frontier.update(int(value) for value in neighbors)
        next_frontier.difference_update(reached)
        reached.update(next_frontier)
        frontier = next_frontier
    outgoing_requests: list[np.ndarray] = [
        np.empty((0, 2), dtype=np.int64) for _ in members
    ]
    request_rows: dict[int, list[tuple[int, int]]] = {}
    for row, (identifier, owner, owned) in enumerate(
        zip(local_ids, local_owners, local_owned, strict=True)
    ):
        if owned:
            continue
        if owner == local_rank:
            raise ValueError("A locally owned FV cell is marked as a ghost.")
        request_rows.setdefault(int(owner), []).append((int(identifier), row))
    for owner, rows in request_rows.items():
        rows.sort()
        outgoing_requests[owner] = np.asarray(rows, dtype=np.int64).reshape((-1, 2))
    incoming_requests = _owner_exchange_message_banks(
        tuple(outgoing_requests),
        local_rank,
        execution_group,
        f"{axis}_requests",
    )
    local_counts = np.asarray(
        [values.shape[0] for values in outgoing_requests], dtype=np.int64
    )
    count_matrix = _owner_group_allgather(
        local_counts,
        execution_group,
        f"{axis}_route_counts",
    )
    if count_matrix.shape != (len(members), len(members)) or np.any(count_matrix < 0):
        raise ValueError("Owner-local FV halo route counts are malformed.")
    if not np.array_equal(count_matrix[local_rank], local_counts):
        raise ValueError("Owner-local FV halo route counts changed on this rank.")
    routes = []
    for sender in range(len(members)):
        for receiver in range(len(members)):
            count = int(count_matrix[receiver, sender])
            if not count:
                continue
            pair = _owner_pair_channel_plan(sender, receiver, len(members))
            if not isinstance(pair, DistributedHaloPlan):
                raise TypeError("Owner channel builder returned an invalid halo plan.")
            if local_rank == sender:
                requests = incoming_requests.get(receiver)
                if requests is None or requests.shape != (count, 2):
                    raise ValueError("FV ghost requests do not match their owner route.")
                try:
                    send_rows = np.asarray(
                        [
                            owner_row_by_id[int(identifier)]
                            for identifier in requests[:, 0]
                        ],
                        dtype=np.int32,
                    )
                except KeyError as error:
                    raise ValueError(
                        "A requested FV stencil cell is absent from its accepted owner."
                    ) from error
            else:
                send_rows = np.zeros((count,), dtype=np.int32)
            if local_rank == receiver:
                requests = outgoing_requests[sender]
                if requests.shape != (count, 2):
                    raise ValueError("FV ghost rows do not match their owner route.")
                ghost_rows = np.asarray(requests[:, 1], dtype=np.int32)
            else:
                ghost_rows = np.zeros((count,), dtype=np.int32)
            pair_ids = np.asarray(pair.local_global_ids, dtype=np.int64)
            pair_owned = np.asarray(pair.local_owned, dtype=np.bool_)
            send_slots = np.flatnonzero((pair_ids[sender] == 0) & pair_owned[sender])
            receive_slots = np.flatnonzero(
                (pair_ids[receiver] == 0) & ~pair_owned[receiver]
            )
            if send_slots.size != 1 or receive_slots.size != 1:
                raise ValueError(
                    "FV owner-pair channel does not expose one exact payload slot."
                )
            routes.append(
                (
                    pair,
                    sender,
                    receiver,
                    send_rows,
                    ghost_rows,
                    int(send_slots[0]),
                    int(receive_slots[0]),
                    count,
                )
            )
    local_counts_by_rank = _owner_group_allgather(
        np.asarray((local_ids.size,), dtype=np.int32),
        execution_group,
        f"{axis}_local_capacity",
    )
    capacity = max(1, int(np.max(local_counts_by_rank, initial=0)))
    local_cell_count = int(local_ids.size)
    provider_id = canonical_fingerprint(
        {
            "kind": "owner-local-fv-stage-state-provider",
            "group": execution_group.spec.group_id,
            "view": owner_local_view.local_view_id,
            "plan_cell_ids": requested_plan_ids,
            "plan_to_view": plan_to_view,
            "routes": tuple(
                (
                    sender,
                    receiver,
                    count,
                    array_tree_fingerprint(send_rows),
                    array_tree_fingerprint(ghost_rows),
                )
                for (
                    _,
                    sender,
                    receiver,
                    send_rows,
                    ghost_rows,
                    _,
                    _,
                    count,
                ) in routes
            ),
        }
    )
    devices = execution_group.devices
    stage_axis = f"{axis}_stage"

    def exchange_stage_state(time: Array, state: Array, /) -> Array:
        del time
        if state.ndim < 2 or state.shape[0] != local_cell_count:
            raise ValueError(
                "Owner-local stage state does not match local cell ownership."
            )
        components = int(np.prod(state.shape[1:]))
        flattened = state.reshape((local_cell_count, components))
        padded = jnp.pad(
            flattened,
            ((0, capacity - local_cell_count), (0, 0)),
        )

        def exchange(local_state: Array, /) -> Array:
            # pmap strips the single local-device axis: one (capacity, c) block.
            values = local_state
            rank = jax.lax.axis_index(stage_axis)
            for (
                pair,
                sender,
                receiver,
                send_rows,
                ghost_rows,
                send_slot,
                receive_slot,
                count,
            ) in routes:
                payload_width = count * components
                source_values = values[jnp.asarray(send_rows, dtype=jnp.int32)]
                packed = jnp.where(
                    rank == sender,
                    source_values.reshape((payload_width,)),
                    jnp.zeros((payload_width,), dtype=values.dtype),
                )
                pair_values = (
                    jnp.zeros((pair.local_capacity, payload_width), dtype=values.dtype)
                    .at[send_slot]
                    .set(packed)
                )
                received = pair.exchange(
                    pair_values,
                    rank,
                    axis_name=stage_axis,
                )
                ghost_values = received[receive_slot].reshape((count, components))
                values = jax.lax.cond(
                    rank == receiver,
                    lambda current: current.at[
                        jnp.asarray(ghost_rows, dtype=jnp.int32)
                    ].set(ghost_values),
                    lambda current: current,
                    values,
                )
            return values

        mapped = jax.pmap(
            exchange,
            axis_name=stage_axis,
            devices=devices,
        )
        exchanged = mapped(padded[None, ...])[0, :local_cell_count]
        view_values = exchanged
        plan_values = view_values[jnp.asarray(plan_to_view, dtype=jnp.int32)]
        return plan_values.reshape(state.shape)

    provider = FiniteVolumeStageStateProvider(
        exchange_stage_state,
        provider_id=provider_id,
    )
    return provider, jnp.asarray(local_owned[plan_to_view]), plan_to_view


def _prepare_owner_local_fv_stage_route_provider(
    owner_local_view: Any,
    execution_group: Any,
    dynamics: PreparedUnstructuredFiniteVolumeDynamics,
    /,
    *,
    axis_name: str,
) -> FiniteVolumeStageRouteProvider | None:
    """Route overset donor debits evaluated on ghost donor rows to their owners.

    An overset route lives on the owner of its fringe receptor. When its donor
    row there is a ghost (a halo or remote donor cell), the receptor's credit
    is applied locally while the donor's debit must reach the donor's owner.
    Returns ``None`` collectively when no owner holds such a route.
    """
    import jax

    from ..discretization._distributed_field import DistributedHaloPlan
    from ..meshing._distribution import (
        _owner_exchange_message_banks,
        _owner_group_allgather,
        _owner_pair_channel_plan,
    )

    if dynamics.coupling.overset is None:
        # Every owner compiles the same coupling, so this exit is collective.
        return None
    template = dynamics.overset_rate_block_template
    members = execution_group.spec.process_indices
    local_rank = owner_local_view.owner_rank
    discretization = dynamics.discretization
    if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
        raise TypeError("Owner-local overset routes require unstructured cell IDs.")
    plan_ids = np.asarray(discretization.cell_global_ids, dtype=np.int64)
    view_ids = np.asarray(owner_local_view.cell_ids, dtype=np.int64)
    view_owners = np.asarray(owner_local_view.cell_owner_ranks, dtype=np.int32)
    owner_by_id = dict(zip(view_ids.tolist(), view_owners.tolist(), strict=True))
    plan_owner = np.asarray(
        [owner_by_id[int(identifier)] for identifier in plan_ids], dtype=np.int32
    )
    plan_row_by_id = {int(identifier): row for row, identifier in enumerate(plan_ids)}
    # An owner without overset routes (for example an empty owner) still joins
    # every collective below, so the group agrees on one provider.
    route_owners = (
        np.empty((0,), dtype=np.int64)
        if template is None
        else np.asarray(template.owner_cells, dtype=np.int64)
    )
    route_active = (
        np.empty((0,), dtype=np.bool_)
        if template is None
        else np.asarray(template.active_mask, dtype=np.bool_)
    )
    ghost_routes = np.flatnonzero(
        route_active & (plan_owner[route_owners] != local_rank)
    ).astype(np.int32)
    ghost_rows = np.unique(route_owners[ghost_routes])
    outgoing: list[np.ndarray] = [np.empty((0, 2), dtype=np.int64) for _ in members]
    for owner in np.unique(plan_owner[ghost_rows]).tolist():
        rows = ghost_rows[plan_owner[ghost_rows] == owner]
        order = np.argsort(plan_ids[rows], kind="stable")
        outgoing[int(owner)] = np.stack(
            (plan_ids[rows][order], rows[order].astype(np.int64)), axis=1
        )
    incoming = _owner_exchange_message_banks(
        tuple(outgoing),
        local_rank,
        execution_group,
        f"{axis_name}_overset_route_requests",
    )
    local_counts = np.asarray([values.shape[0] for values in outgoing], dtype=np.int64)
    count_matrix = _owner_group_allgather(
        local_counts, execution_group, f"{axis_name}_overset_route_counts"
    )
    if count_matrix.shape != (len(members), len(members)) or np.any(count_matrix < 0):
        raise ValueError("Owner-local overset route counts are malformed.")
    if not np.any(count_matrix):
        return None
    routes = []
    # count_matrix[sender, receiver]: ghost debits the sender returns to owners.
    for sender in range(len(members)):
        for receiver in range(len(members)):
            count = int(count_matrix[sender, receiver])
            if not count:
                continue
            pair = _owner_pair_channel_plan(sender, receiver, len(members))
            if not isinstance(pair, DistributedHaloPlan):
                raise TypeError("Owner channel builder returned an invalid halo plan.")
            send_rows = np.zeros((count,), dtype=np.int32)
            receive_rows = np.zeros((count,), dtype=np.int32)
            if local_rank == sender:
                send_rows = np.asarray(outgoing[receiver][:, 1], dtype=np.int32)
            if local_rank == receiver:
                requests = incoming.get(sender)
                if requests is None or requests.shape != (count, 2):
                    raise ValueError("Overset debit routes do not match their owner.")
                try:
                    receive_rows = np.asarray(
                        [
                            plan_row_by_id[int(identifier)]
                            for identifier in requests[:, 0]
                        ],
                        dtype=np.int32,
                    )
                except KeyError as error:
                    raise ValueError(
                        "An overset donor cell is absent from its accepted owner."
                    ) from error
                if np.any(plan_owner[receive_rows] != local_rank):
                    raise ValueError("An overset donor debit targets a non-owned cell.")
            pair_ids = np.asarray(pair.local_global_ids, dtype=np.int64)
            pair_owned = np.asarray(pair.local_owned, dtype=np.bool_)
            send_slots = np.flatnonzero((pair_ids[sender] == 0) & pair_owned[sender])
            receive_slots = np.flatnonzero(
                (pair_ids[receiver] == 0) & ~pair_owned[receiver]
            )
            if send_slots.size != 1 or receive_slots.size != 1:
                raise ValueError(
                    "Overset owner-pair channel does not expose one exact payload slot."
                )
            routes.append(
                (
                    pair,
                    sender,
                    receiver,
                    send_rows,
                    receive_rows,
                    int(send_slots[0]),
                    int(receive_slots[0]),
                    count,
                )
            )
    capacity = max(
        1,
        int(
            np.max(
                _owner_group_allgather(
                    np.asarray((plan_ids.size,), dtype=np.int32),
                    execution_group,
                    f"{axis_name}_overset_route_capacity",
                ),
                initial=0,
            )
        ),
    )
    cell_count = int(plan_ids.size)
    block_route_id = None if template is None else template.route_id
    block_kind = None if template is None else template.block_kind
    devices = execution_group.devices
    stage_axis = f"{axis_name}_stage"

    def route_overset_debits(ledger: Any, /) -> Array:
        components = int(np.prod(ledger.component_shape))
        debits = jnp.zeros((capacity, components), dtype=ledger.source_rate.dtype)
        if block_kind is not None:
            blocks = [
                block
                for block in ledger.blocks
                if block.block_kind == block_kind and block.route_id == block_route_id
            ]
            if len(blocks) != 1:
                raise ValueError("Stage ledger lacks its exact overset correction block.")
            rates = blocks[0].flux_rate
            flat = rates.reshape((rates.shape[0], components)).astype(debits.dtype)
            # The route debits its owner (donor) row; see the ledger scatter.
            debits = debits.at[
                jnp.asarray(route_owners[ghost_routes], dtype=jnp.int32)
            ].add(-flat[jnp.asarray(ghost_routes, dtype=jnp.int32)])

        def exchange(local_debits: Array, /) -> Array:
            rank = jax.lax.axis_index(stage_axis)
            received = jnp.zeros_like(local_debits)
            for (
                pair,
                sender,
                receiver,
                send_rows,
                receive_rows,
                send_slot,
                receive_slot,
                count,
            ) in routes:
                width = count * components
                packed = jnp.where(
                    rank == sender,
                    local_debits[jnp.asarray(send_rows, dtype=jnp.int32)].reshape(
                        (width,)
                    ),
                    jnp.zeros((width,), dtype=local_debits.dtype),
                )
                payload = (
                    jnp.zeros((pair.local_capacity, width), dtype=local_debits.dtype)
                    .at[send_slot]
                    .set(packed)
                )
                arrived = pair.exchange(payload, rank, axis_name=stage_axis)
                values = arrived[receive_slot].reshape((count, components))
                received = jax.lax.cond(
                    rank == receiver,
                    lambda current: current.at[
                        jnp.asarray(receive_rows, dtype=jnp.int32)
                    ].add(values),
                    lambda current: current,
                    received,
                )
                # Every owner must issue the pair permutes in one order; the
                # barrier makes the next payload depend on this delivery.
                local_debits, received = jax.lax.optimization_barrier(
                    (local_debits, received)
                )
            return received

        mapped = jax.pmap(exchange, axis_name=stage_axis, devices=devices)
        received = mapped(debits[None, ...])[0, :cell_count]
        return received.reshape((cell_count, *ledger.component_shape))

    provider_id = canonical_fingerprint(
        {
            "kind": "owner-local-fv-overset-debit-routes",
            "group": execution_group.spec.group_id,
            "view": owner_local_view.local_view_id,
            "overset_routes": block_route_id,
            "routes": tuple(
                (
                    sender,
                    receiver,
                    count,
                    array_tree_fingerprint(send_rows),
                    array_tree_fingerprint(receive_rows),
                )
                for (_, sender, receiver, send_rows, receive_rows, _, _, count) in routes
            ),
        }
    )
    return FiniteVolumeStageRouteProvider(route_overset_debits, provider_id=provider_id)


def _owner_local_global_decision_provider(
    execution_group: Any,
    /,
    *,
    axis_name: str,
) -> FiniteVolumeStageDecisionProvider:
    import jax

    axis = str(axis_name).strip()
    if not axis:
        raise ValueError("owner-local decision axis_name must be non-empty.")
    devices = execution_group.devices
    if len(devices) == 1:
        callback = lambda value: jnp.asarray(value, dtype=jnp.bool_)
    else:

        def reduce(local_value: Array, /) -> Array:
            return jax.lax.pmin(local_value.astype(jnp.int32), axis)

        mapped = jax.pmap(reduce, axis_name=axis, devices=devices)

        def callback(value: Array, /) -> Array:
            local_value = jnp.asarray(value, dtype=jnp.int32).reshape((1,))
            return mapped(local_value)[0].astype(jnp.bool_)

    return FiniteVolumeStageDecisionProvider(
        callback,
        provider_id=canonical_fingerprint(
            {
                "kind": "owner-local-stage-decision-provider",
                "group": execution_group.spec.group_id,
                "axis": axis,
            }
        ),
    )


def _require_owner_local_fv_geometry_current(
    view: Any,
    dynamics: PreparedUnstructuredFiniteVolumeDynamics,
    execution_group: Any,
    axis_name: str,
    /,
) -> None:
    from ..discretization._view_support import (
        OwnerLocalCellGeometry,
        OwnerLocalCompositeCellGeometry,
    )

    if not isinstance(view, _owner_local_view_types()) or not isinstance(
        dynamics.discretization, UnstructuredFiniteVolumeDiscretization
    ):
        raise TypeError(
            "Owner-local FV geometry requires its typed view and discretization."
        )
    prepared_view = dynamics.discretization.owner_local_view
    if (
        prepared_view is None
        or type(prepared_view) is not type(view)
        or prepared_view.local_view_id != view.local_view_id
        or prepared_view.distribution_id != view.distribution_id
        or prepared_view.part_id != view.part_id
        or prepared_view.source_topology_id != view.source_topology_id
        or prepared_view.source_geometry_id != view.source_geometry_id
        or (
            isinstance(view, OwnerLocalCellGeometry)
            and isinstance(prepared_view, OwnerLocalCellGeometry)
            and (
                prepared_view.source_map_id != view.source_map_id
                or prepared_view.source_context_id != view.source_context_id
            )
        )
    ):
        raise ValueError(
            "Prepared FV dynamics differ from their exact compact geometry view."
        )
    view.require_owner_directory_current(
        execution_group,
        axis_name=axis_name,
    )
    prepared_mesh = dynamics.discretization.mesh
    if isinstance(view, OwnerLocalCompositeCellGeometry):
        carrier_mesh = view.union_local_mesh()
    else:
        carrier_mesh, local_geometry = view.reconstruct_local_geometry()
        if carrier_mesh is None:
            if (
                prepared_mesh is not None
                or local_geometry is not None
                or view.local_mesh_id is not None
                or np.asarray(view.cell_ids).size
                or np.asarray(view.vertex_ids).size
            ):
                raise ValueError(
                    "Empty owner-local FV geometry differs from its prepared carrier."
                )
            return
        if local_geometry is None or view.local_mesh_id != carrier_mesh.mesh_id:
            raise ValueError(
                "Prepared FV mesh differs from the exact reconstructed owner geometry."
            )
    # The FV lowering re-blocks carriers by cell kind (and a composite unions
    # its parts), so carriers are compared through their stable vertex/cell
    # identities, coordinates and cell-vertex incidence.
    if (carrier_mesh is None) != (prepared_mesh is None):
        raise ValueError(
            "Prepared FV mesh differs from the exact reconstructed owner geometry."
        )
    if carrier_mesh is None or prepared_mesh is None:
        return
    if not _same_cell_mesh_identities(carrier_mesh, prepared_mesh):
        raise ValueError(
            "Prepared FV mesh differs from the exact reconstructed owner geometry."
        )


def _same_cell_mesh_identities(first: Any, second: Any, /) -> bool:
    """Equal vertex ids, coordinates by id, cell ids and cell vertex-id sets."""

    def identities(mesh: Any, /) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        order = np.argsort(vertex_ids, kind="stable")
        cells = sorted(
            (int(identifier), tuple(sorted(vertex_ids[np.asarray(row)].tolist())))
            for block in mesh.blocks
            for identifier, row in zip(
                np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
            )
        )
        return (
            vertex_ids[order],
            np.asarray(mesh.coordinates)[order],
            np.asarray([identifier for identifier, _ in cells], dtype=np.int64),
            [vertices for _, vertices in cells],
        )

    first_ids = identities(first)
    second_ids = identities(second)
    return (
        np.array_equal(first_ids[0], second_ids[0])
        and np.array_equal(first_ids[1], second_ids[1])
        and np.array_equal(first_ids[2], second_ids[2])
        and first_ids[3] == second_ids[3]
    )


def _owner_local_reconstruction_stencil_depth(
    dynamics: PreparedUnstructuredFiniteVolumeDynamics, /
) -> int:
    from ..discretization.finite_volume._cell_polynomial import (
        PreparedCellPolynomialReconstruction,
    )
    from ..discretization.finite_volume._unstructured_weno import (
        PreparedUnstructuredWENOZReconstruction,
    )

    reconstruction = dynamics.method.reconstruction
    if isinstance(reconstruction, PiecewiseConstantReconstruction):
        return 1
    if isinstance(reconstruction, PreparedCellPolynomialReconstruction):
        depths = (int(np.asarray(reconstruction.report.maximum_stencil_depth)),)
    elif isinstance(reconstruction, PreparedUnstructuredWENOZReconstruction):
        depths = tuple(
            int(np.asarray(item.report.maximum_stencil_depth))
            for item in (reconstruction.optimal, *reconstruction.sectors)
        )
    else:
        raise TypeError(
            "Owner-local FV requires a reconstruction with an explicit prepared stencil."
        )
    return max(1, max(depths, default=0) + 1)


def _minimum_or_infinity(values: Array, dtype: Any, /) -> Array:
    target_dtype = jnp.dtype(dtype)
    if values.size == 0:
        return jnp.asarray(jnp.inf, dtype=target_dtype)
    return jnp.min(values).astype(target_dtype)


def _owner_local_global_scalar_min_provider(
    execution_group: Any,
    /,
    *,
    axis_name: str,
) -> Callable[[Array], Array]:
    import jax

    axis = str(axis_name).strip()
    if not axis:
        raise ValueError("owner-local scalar reduction axis_name must be non-empty.")
    devices = execution_group.devices
    if len(devices) == 1:
        return lambda value: jnp.asarray(value)

    def reduce(local_value: Array, /) -> Array:
        return jax.lax.pmin(local_value, axis)

    mapped = jax.pmap(reduce, axis_name=axis, devices=devices)

    def collective(value: Array, /) -> Array:
        local_value = jnp.asarray(value).reshape((1,))
        return mapped(local_value)[0]

    return collective


class FiniteVolumeRunStatus(IntEnum):
    SUCCESS = 0
    RECOVERED_REJECTION = 1
    INVALID_INITIAL_STATE = 2
    RETRY_LIMIT_REACHED = 3
    MINIMUM_STEP_REACHED = 4
    NONFINITE_STATE = 5
    STABILITY_LIMIT_EXCEEDED = 6
    PRESCRIBED_STEP_REJECTED = 7


class FiniteVolumeStepPolicy(StrictModule, NonTrainableState):
    cfl: float = eqx.field(static=True)
    maximum_retries: int = eqx.field(static=True)
    reduction_factor: float = eqx.field(static=True)
    minimum_step_size: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        cfl: float = 0.45,
        maximum_retries: int = 4,
        reduction_factor: float = 0.5,
        minimum_step_size: float = 1e-12,
    ) -> None:
        cfl_ = float(cfl)
        retries = int(maximum_retries)
        reduction = float(reduction_factor)
        minimum = float(minimum_step_size)
        if (
            not np.isfinite(cfl_)
            or cfl_ <= 0.0
            or retries < 0
            or not 0.0 < reduction < 1.0
            or not np.isfinite(minimum)
            or minimum <= 0.0
        ):
            raise ValueError("Finite-volume step policy is invalid.")
        self.cfl = cfl_
        self.maximum_retries = retries
        self.reduction_factor = reduction
        self.minimum_step_size = minimum
        self.policy_id = canonical_fingerprint(
            {
                "kind": "finite-volume-step-policy",
                "cfl": cfl_,
                "maximum_retries": retries,
                "reduction_factor": reduction,
                "minimum_step_size": minimum,
            }
        )


class FiniteVolumeStageFlux(StrictModule):
    """One stage-specific replacement of selected normal face fluxes."""

    replacement_normal_fluxes: tuple[Array, ...]
    replacement_masks: tuple[Array, ...]
    interface_conservative_flux: Array
    evidence: Any

    def __init__(
        self,
        replacement_normal_fluxes: tuple[ArrayLike, ...],
        replacement_masks: tuple[ArrayLike, ...],
        interface_conservative_flux: ArrayLike,
        evidence: Any,
        /,
    ) -> None:
        fluxes = tuple(jnp.asarray(value) for value in replacement_normal_fluxes)
        masks = tuple(jnp.asarray(value, dtype=jnp.bool_) for value in replacement_masks)
        interface_flux = jnp.asarray(interface_conservative_flux)
        if not fluxes or len(fluxes) != len(masks):
            raise ValueError(
                "Stage-flux replacements require aligned nonempty flux and mask tuples."
            )
        if any(
            flux.ndim == 0 or mask.shape != flux.shape[:-1]
            for flux, mask in zip(fluxes, masks, strict=True)
        ):
            raise ValueError(
                "Each stage-flux mask must match its replacement face layout."
            )
        if interface_flux.ndim == 0:
            raise ValueError("interface_conservative_flux must retain a component axis.")
        self.replacement_normal_fluxes = fluxes
        self.replacement_masks = masks
        self.interface_conservative_flux = interface_flux
        self.evidence = evidence


class FiniteVolumeStageFluxProvider(StrictModule, NonTrainableState):
    """Immutable provider for selected common-flux replacements at SSPRK stages."""

    callback: Callable[[int, Array, Array], FiniteVolumeStageFlux]
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        callback: Callable[[int, Array, Array], FiniteVolumeStageFlux],
        /,
        *,
        provider_id: str,
    ) -> None:
        identity = str(provider_id)
        if not callable(callback):
            raise TypeError("callback must be callable.")
        if not identity:
            raise ValueError("provider_id must be non-empty.")
        self.callback = callback
        self.provider_id = identity

    def evaluate(
        self,
        stage_index: int,
        time: Array,
        state: Array,
        reference_fluxes: tuple[Array, ...],
        /,
    ) -> FiniteVolumeStageFlux:
        stage = int(stage_index)
        if stage not in (0, 1, 2):
            raise ValueError("SSPRK3 stage_index must be 0, 1, or 2.")
        supplied = self.callback(stage, time, state)
        if not isinstance(supplied, FiniteVolumeStageFlux):
            raise TypeError("Stage-flux callback must return FiniteVolumeStageFlux.")
        if len(supplied.replacement_normal_fluxes) != len(reference_fluxes):
            raise ValueError(
                "Stage-flux replacements must match the finite-volume face blocks."
            )
        fluxes = []
        masks = []
        for replacement, mask, reference in zip(
            supplied.replacement_normal_fluxes,
            supplied.replacement_masks,
            reference_fluxes,
            strict=True,
        ):
            if replacement.shape != reference.shape or mask.shape != reference.shape[:-1]:
                raise ValueError(
                    "Stage-flux replacement shapes must match the finite-volume faces."
                )
            replacement_ = jnp.asarray(replacement, dtype=reference.dtype)
            replacement_ = eqx.error_if(
                replacement_,
                jnp.any(mask & ~jnp.all(jnp.isfinite(replacement_), axis=-1)),
                "Active stage-flux replacements must be finite.",
            )
            fluxes.append(replacement_)
            masks.append(mask)
        interface_flux = jnp.asarray(
            supplied.interface_conservative_flux, dtype=state.dtype
        )
        if interface_flux.shape[-1] != state.shape[-1] or interface_flux.shape[:-1] == ():
            raise ValueError(
                "Stage interface fluxes must have shape (interfaces, components)."
            )
        return FiniteVolumeStageFlux(
            tuple(fluxes),
            tuple(masks),
            interface_flux,
            supplied.evidence,
        )


class FiniteVolumeStageFluxTrace(StrictModule):
    """The exact three provider outputs retained by one SSPRK3 attempt."""

    stages: tuple[
        FiniteVolumeStageFlux,
        FiniteVolumeStageFlux,
        FiniteVolumeStageFlux,
    ]
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        stages: tuple[
            FiniteVolumeStageFlux,
            FiniteVolumeStageFlux,
            FiniteVolumeStageFlux,
        ],
        provider_id: str,
        /,
    ) -> None:
        if (
            not isinstance(stages, tuple)
            or len(stages) != 3
            or any(not isinstance(value, FiniteVolumeStageFlux) for value in stages)
        ):
            raise TypeError("Stage-flux traces require exactly three provider outputs.")
        identity = str(provider_id)
        if not identity:
            raise ValueError("provider_id must be non-empty.")
        self.stages = stages
        self.provider_id = identity


class FiniteVolumeRuntimeState(StrictModule):
    content_state: FiniteVolumeConservativeContentState
    topology_journal: FiniteVolumeTopologyEventJournal
    accepted_step: Array
    step_size: Array
    last_status: Array
    controller_state: Array
    integrator_state: Array
    output_cursor: Array
    sliding_coupling: PeriodicSlidingCoupling | None
    sliding_shift: Array
    sliding_coupling_id: str | None = eqx.field(static=True)
    sliding_event_id: str | None = eqx.field(static=True)

    @checked
    def __init__(
        self,
        content_state: FiniteVolumeConservativeContentState,
        topology_journal: FiniteVolumeTopologyEventJournal,
        step_size: ArrayLike,
        /,
        *,
        accepted_step: ArrayLike = 0,
        last_status: ArrayLike = FiniteVolumeRunStatus.SUCCESS,
        controller_state: ArrayLike | None = None,
        integrator_state: ArrayLike | None = None,
        output_cursor: ArrayLike = 0,
        sliding_coupling: PeriodicSlidingCoupling | None = None,
        sliding_shift: ArrayLike = 0.0,
        sliding_event_id: str | None = None,
    ) -> None:
        if topology_journal.current_epoch_id != content_state.topology_epoch_id:
            raise ValueError(
                "Runtime content and topology journal must own the same current epoch."
            )
        if sliding_coupling is not None and not isinstance(
            sliding_coupling, PeriodicSlidingCoupling
        ):
            raise TypeError("sliding_coupling must be PeriodicSlidingCoupling or None.")
        shift = jnp.asarray(
            sliding_shift, dtype=content_state.precision.reduction_dtype
        ).reshape(())
        shift = eqx.error_if(
            shift,
            ~jnp.isfinite(shift),
            "sliding_shift must be finite.",
        )
        if sliding_event_id is not None and (
            not isinstance(sliding_event_id, str) or not sliding_event_id
        ):
            raise ValueError("sliding_event_id must be nonempty or None.")
        self.content_state = content_state
        self.topology_journal = topology_journal
        self.accepted_step = jnp.asarray(accepted_step, dtype=jnp.int32).reshape(())
        self.step_size = jnp.asarray(step_size).reshape(())
        self.last_status = jnp.asarray(last_status, dtype=jnp.int32).reshape(())
        self.controller_state = jnp.asarray(
            () if controller_state is None else controller_state
        )
        self.integrator_state = jnp.asarray(
            () if integrator_state is None else integrator_state
        )
        self.output_cursor = jnp.asarray(output_cursor, dtype=jnp.int32).reshape(())
        self.sliding_coupling = sliding_coupling
        self.sliding_shift = shift
        self.sliding_coupling_id = (
            None if sliding_coupling is None else sliding_coupling.coupling_id
        )
        self.sliding_event_id = sliding_event_id

    @property
    def time(self) -> Array:
        return self.content_state.time

    def cell_average(self) -> Array:
        """Derive cell averages from the authoritative conservative content."""

        return self.content_state.cell_average()


class FiniteVolumeValidatedRuntimeState(StrictModule):
    """Runtime-ordered accepted state bound to stable source IDs and authority."""

    runtime_state: FiniteVolumeRuntimeState
    source_cell_ids: Array
    runtime_id: str = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        source_cell_ids: ArrayLike,
        runtime_id: str,
        epoch_id: str,
        state_id: str,
        /,
    ) -> None:
        ids = jnp.asarray(source_cell_ids)
        host_ids = np.asarray(ids)
        if (
            not isinstance(runtime_state, FiniteVolumeRuntimeState)
            or ids.ndim != 1
            or host_ids.dtype.kind not in "iu"
            or (host_ids.size and np.any(host_ids < 0))
            or np.unique(host_ids).size != host_ids.size
            or ids.shape != (runtime_state.content_state.cell_count,)
        ):
            raise ValueError("Validated FV runtime state has malformed stable cell IDs.")
        identities = (runtime_id, epoch_id, state_id)
        if any(not isinstance(value, str) or not value for value in identities):
            raise ValueError("Validated FV runtime state identities must be non-empty.")
        self.runtime_state = runtime_state
        self.source_cell_ids = ids
        self.runtime_id = runtime_id
        self.epoch_id = epoch_id
        self.state_id = state_id


class FiniteVolumeALEAdvanceEvidence(StrictModule):
    """ALE geometry and stage-rate evidence for the public accepted ledger."""

    accepted: Array
    stage_rate_ledgers: tuple[
        ConservationStageLedger,
        ConservationStageLedger,
        ConservationStageLedger,
    ]
    geometry: UnstructuredALEStepGeometry
    maximum_relative_rate: Array
    relative_cfl_step: Array
    geometry_reduction_factor: Array


class FiniteVolumeEmbeddedAdvanceEvidence(StrictModule):
    """Stationary embedded metrics, redistribution, and effective CFL evidence."""

    accepted: Array
    stage_rate_ledgers: tuple[
        ConservationStageLedger,
        ConservationStageLedger,
        ConservationStageLedger,
    ]
    stage_metrics: tuple[
        FiniteVolumeStageMetrics,
        FiniteVolumeStageMetrics,
        FiniteVolumeStageMetrics,
    ]
    accepted_metrics: FiniteVolumeStageMetrics
    redistribution: ConservativeSmallCellRedistributionReport
    stage_maximum_relative_rates: tuple[Array, Array, Array]
    maximum_relative_rate: Array
    relative_cfl_step: Array


class FiniteVolumeTopologyEventRefusal(StrictModule):
    """Typed reason a step-boundary topology event published no successor runtime.

    The accepted PDE step stays accepted and its state is rolled back to the
    accepted source epoch; this record names why the event itself was refused.
    """

    status: TopologyEventStatus = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        status: TopologyEventStatus,
        reason: str,
        request_id: str,
        evidence_id: str,
    ) -> None:
        if not isinstance(status, TopologyEventStatus) or status in (
            TopologyEventStatus.PENDING,
            TopologyEventStatus.SUCCESS,
        ):
            raise ValueError("Topology event refusals require a failed event status.")
        if any(
            not isinstance(value, str) or not value
            for value in (reason, request_id, evidence_id)
        ):
            raise ValueError("Topology event refusal identities must be non-empty.")
        self.status = status
        self.reason = reason
        self.request_id = request_id
        self.evidence_id = evidence_id


class FiniteVolumeAdvanceResult(StrictModule):
    runtime_state: FiniteVolumeRuntimeState = fixed_field()
    accepted: Array = fixed_field()
    retries: Array = fixed_field()
    attempted_step_size: Array = fixed_field()
    accepted_step_size: Array = fixed_field()
    positivity: FiniteVolumeAdmissibilityReport = fixed_field()
    accepted_flux_integrals: AcceptedConservationIntegralLedger = fixed_field()
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    ale: FiniteVolumeALEAdvanceEvidence | None = fixed_field()
    embedded: FiniteVolumeEmbeddedAdvanceEvidence | None = fixed_field()
    successor_runtime: PreparedFiniteVolumeRuntime | None = None
    shallow_water_integrals: ShallowWaterAcceptedFaceIntegrals | None = fixed_field(
        default=None
    )
    stage_flux_trace: FiniteVolumeStageFluxTrace | None = None
    overset_motion_receipt: CompositionRebindReceipt | None = fixed_field(default=None)
    topology_event_refusal: FiniteVolumeTopologyEventRefusal | None = None


class FiniteVolumeOwnerLocalRepartitionResult(StrictModule):
    """REPARTITION event outcome, distinct from any prior PDE advance result."""

    runtime_state: FiniteVolumeRuntimeState = fixed_field()
    accepted: Array = fixed_field()
    event_status: Array = fixed_field()
    successor_runtime: PreparedFiniteVolumeRuntime | None = None
    repartition_receipt: CompositionRebindReceipt | None = fixed_field(default=None)
    refusal: str | None = eqx.field(static=True, default=None)


class FiniteVolumeScheduledAdvanceResult(StrictModule):
    """One exact prescribed-step attempt with explicit stability evidence."""

    runtime_state: FiniteVolumeRuntimeState = fixed_field()
    attempted: FiniteVolumeAdvanceResult
    accepted: Array = fixed_field()
    requested_step_size: Array = fixed_field()
    stable_step_size: Array = fixed_field()
    stability_margin: Array = fixed_field()


class PreparedFiniteVolumeRuntime(StrictModule):
    """Content-authoritative SSPRK3 runtime with bounded positivity retries."""

    dynamics: (
        PreparedFiniteVolumeDynamics
        | PreparedTriangleFiniteVolumeDynamics
        | PreparedUnstructuredFiniteVolumeDynamics
    )
    fallback_dynamics: (
        PreparedFiniteVolumeDynamics
        | PreparedTriangleFiniteVolumeDynamics
        | PreparedUnstructuredFiniteVolumeDynamics
    )
    positivity: FluxPositivityPlan
    policy: FiniteVolumeStepPolicy
    stage_state_provider: FiniteVolumeStageStateProvider | None
    stage_flux_provider: FiniteVolumeStageFluxProvider | None
    effective_cell_volumes: Array = fixed_field()
    active_cell_mask: Array = fixed_field()
    owner_local_geometry: Any = fixed_field()
    owner_local_execution_group: Any = eqx.field(static=True)
    owner_local_owned_cell_mask: Array | None = fixed_field()
    owner_local_decision_provider: Any = eqx.field(static=True)
    owner_local_scalar_min_provider: Any = eqx.field(static=True)
    owner_local_route_provider: Any = eqx.field(static=True)
    owner_local_axis_name: str = eqx.field(static=True)
    precision: FiniteVolumePrecisionPolicy
    static_flux_rate_block_templates: tuple[ConservationStageFluxRateBlock, ...] = (
        fixed_field()
    )
    embedded_redistribution: ConservativeSmallCellRedistributionPlan | None
    embedded_stage_template: FiniteVolumeStageMetrics | None
    sliding_plan: PeriodicSlidingInterfacePlan | None
    sliding_initial_coupling: PeriodicSlidingCoupling | None
    initial_topology_epoch: TopologyEpoch = eqx.field(static=True)
    initial_topology_artifacts: FiniteVolumeTopologyArtifacts = eqx.field(static=True)
    topology_epoch_id: str = eqx.field(static=True)
    geometry_family_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    evidence_policy_id: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        dynamics: (
            PreparedFiniteVolumeDynamics
            | PreparedTriangleFiniteVolumeDynamics
            | PreparedUnstructuredFiniteVolumeDynamics
        ),
        positivity: FluxPositivityPlan,
        policy: FiniteVolumeStepPolicy | None = None,
        *,
        topology_epoch: TopologyEpoch | None = None,
        topology_artifacts: FiniteVolumeTopologyArtifacts | None = None,
        stage_state_provider: FiniteVolumeStageStateProvider | None = None,
        stage_flux_provider: FiniteVolumeStageFluxProvider | None = None,
        execution_group: Any = None,
        owner_local_geometry: Any = None,
        owner_local_axis_name: str = "finite_volume_owner_local",
    ) -> None:
        if not isinstance(
            dynamics,
            (
                PreparedFiniteVolumeDynamics,
                PreparedTriangleFiniteVolumeDynamics,
                PreparedUnstructuredFiniteVolumeDynamics,
            ),
        ):
            raise TypeError("dynamics must be prepared finite-volume dynamics.")
        if isinstance(dynamics, PreparedFiniteVolumeDynamics) and isinstance(
            dynamics.method.interface_solver, AbstractWavePropagationPlan
        ):
            raise ValueError(
                "Wave-propagation dynamics do not expose the face fluxes required by PreparedFiniteVolumeRuntime."
            )
        policy_ = FiniteVolumeStepPolicy() if policy is None else policy
        if not isinstance(policy_, FiniteVolumeStepPolicy):
            raise TypeError("policy must be FiniteVolumeStepPolicy.")
        owner_local_view = owner_local_geometry
        if (
            owner_local_view is None
            and isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            and isinstance(
                dynamics.discretization, UnstructuredFiniteVolumeDiscretization
            )
        ):
            owner_local_view = dynamics.discretization.owner_local_view
        owner_local_group = execution_group
        owner_local_owned_mask = None
        owner_local_decision_provider = None
        owner_local_scalar_min_provider = None
        owner_local_route_provider = None
        owner_axis = str(owner_local_axis_name).strip()
        if owner_local_view is None:
            if execution_group is not None:
                raise ValueError(
                    "execution_group requires a canonical owner-local FV geometry view."
                )
            owner_axis = ""
        else:
            from .._execution_runtime import ExecutionGroup

            if not isinstance(owner_local_view, _owner_local_view_types()):
                raise TypeError(
                    "owner_local_geometry must be an owner-local FV cell layout."
                )
            if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
                raise TypeError("Owner-local geometry requires unstructured FV dynamics.")
            if not isinstance(
                dynamics.discretization, UnstructuredFiniteVolumeDiscretization
            ):
                raise TypeError(
                    "Owner-local geometry requires unstructured discretization."
                )
            if not isinstance(owner_local_group, ExecutionGroup):
                raise TypeError(
                    "Owner-local finite-volume runtime requires ExecutionGroup."
                )
            if stage_state_provider is not None:
                raise ValueError(
                    "Owner-local FV runtimes require their canonical stage halo provider."
                )
            _require_owner_local_fv_geometry_current(
                owner_local_view,
                dynamics,
                owner_local_group,
                f"{owner_axis}_directory",
            )
            stage_state_provider, owner_local_owned_mask, _ = (
                _prepare_owner_local_fv_stage_state_provider(
                    owner_local_view,
                    owner_local_group,
                    dynamics.discretization.cell_global_ids,
                    axis_name=owner_axis,
                    required_stencil_depth=_owner_local_reconstruction_stencil_depth(
                        dynamics
                    ),
                )
            )
            # Every collective traced into one advance program must name the
            # same device mesh axis, so the providers share the stage axis.
            owner_local_decision_provider = _owner_local_global_decision_provider(
                owner_local_group,
                axis_name=f"{owner_axis}_stage",
            )
            owner_local_scalar_min_provider = _owner_local_global_scalar_min_provider(
                owner_local_group,
                axis_name=f"{owner_axis}_stage",
            )
            owner_local_route_provider = _prepare_owner_local_fv_stage_route_provider(
                owner_local_view,
                owner_local_group,
                dynamics,
                axis_name=owner_axis,
            )
        if stage_state_provider is not None:
            if not isinstance(stage_state_provider, FiniteVolumeStageStateProvider):
                raise TypeError(
                    "stage_state_provider must be FiniteVolumeStageStateProvider or None."
                )
            if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
                raise ValueError(
                    "Stage-state providers are supported only by unstructured finite-volume runtimes."
                )
        if stage_flux_provider is not None:
            if not isinstance(stage_flux_provider, FiniteVolumeStageFluxProvider):
                raise TypeError(
                    "stage_flux_provider must be FiniteVolumeStageFluxProvider or None."
                )
            if not isinstance(dynamics, PreparedFiniteVolumeDynamics) or not isinstance(
                dynamics.method.interface_solver, AbstractNumericalFluxPlan
            ):
                raise ValueError(
                    "Stage-flux providers require stationary structured "
                    "finite-volume dynamics with explicit numerical fluxes."
                )
        sliding_plan = (
            dynamics.coupling.sliding
            if isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            else None
        )
        sliding_initial_coupling = (
            dynamics.coupling.sliding_coupling
            if isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            else None
        )
        if isinstance(
            dynamics,
            (
                PreparedTriangleFiniteVolumeDynamics,
                PreparedUnstructuredFiniteVolumeDynamics,
            ),
        ):
            fallback = dynamics.make_fallback_dynamics(positivity.fallback_flux)
        elif isinstance(dynamics.method.interface_solver, ShallowWaterHydrostaticHLLPlan):
            if not isinstance(dynamics.bathymetry, PreparedShallowWaterBathymetry):
                raise TypeError(
                    "Hydrostatic shallow-water runtime requires prepared bathymetry."
                )
            fallback_method = FiniteVolumeMethodPlan(
                PiecewiseConstantReconstruction(),
                dynamics.method.interface_solver,
                differentiability=BranchDifferentiationPolicy.BRANCHWISE,
            )
            fallback = PreparedFiniteVolumeDynamics(
                dynamics.system,
                dynamics.discretization,
                fallback_method,
                dynamics.boundaries,
                bathymetry=dynamics.bathymetry.values,
                precision=dynamics.precision,
                source=dynamics.source,
                source_id=dynamics.source_id,
            )
        else:
            fallback_method = FiniteVolumeMethodPlan(
                PiecewiseConstantReconstruction(),
                positivity.fallback_flux,
                viscous=dynamics.method.viscous,
                differentiability=BranchDifferentiationPolicy.BRANCHWISE,
            )
            fallback = PreparedFiniteVolumeDynamics(
                dynamics.system,
                dynamics.discretization,
                fallback_method,
                dynamics.boundaries,
                capacity=dynamics.capacity,
                bathymetry=dynamics.bathymetry,
                precision=dynamics.precision,
                source=dynamics.source,
                source_id=dynamics.source_id,
            )
        discretization = dynamics.discretization
        embedded_redistribution = None
        embedded_stage_template = None
        embedded_boundaries: UnstructuredEmbeddedBoundarySet | None = None
        embedded_metrics = (
            dynamics.coupling.embedded_metrics
            if isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            else None
        )
        overset_effective_volumes = (
            dynamics.overset_effective_cell_volumes
            if isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            and dynamics.overset_mapping_id is not None
            else None
        )
        if isinstance(dynamics, PreparedFiniteVolumeDynamics):
            effective_volumes = dynamics.effective_volumes
        elif isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
            effective_volumes = (
                dynamics.overset_effective_cell_volumes
                if overset_effective_volumes is not None
                else (
                    embedded_metrics.fluid_cell_volumes
                    if embedded_metrics is not None
                    else discretization.cell_volumes
                )
            )
        else:
            effective_volumes = discretization.cell_volumes
        bound_effective_volumes = dynamics.precision.reduction(effective_volumes).reshape(
            (-1,)
        )
        if isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            topology_id = discretization.topology_id
            geometry_id = discretization.geometry_id
            geometry_family_id = geometry_id
            motion = (
                dynamics.coupling.motion
                if isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
                else None
            )
            if embedded_metrics is not None:
                if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
                    raise TypeError("Embedded metrics require unstructured dynamics.")
                embedded_boundaries = dynamics.coupling.embedded_boundaries
                stabilization = dynamics.coupling.embedded_stabilization_policy
                if (
                    not isinstance(
                        embedded_boundaries,
                        UnstructuredEmbeddedBoundarySet,
                    )
                    or stabilization is None
                ):
                    raise TypeError(
                        "Prepared embedded coupling requires wall ownership and stabilization."
                    )
                embedded_stage_template = lower_embedded_stage_metrics(
                    discretization,
                    embedded_metrics,
                    embedded_boundaries,
                    discretization.topology_id,
                    0,
                    0,
                    time=0.0,
                )
                geometry_layout_id = embedded_stage_template.geometry_layout_id
                evidence_policy_id = embedded_stage_template.evidence.policy_id
                embedded_redistribution = ConservativeSmallCellRedistributionPlan(
                    discretization,
                    embedded_metrics,
                    stabilization,
                )
            elif motion is None:
                static_metrics = lower_static_unstructured_stage_metrics(discretization)
                geometry_layout_id = static_metrics.geometry_layout_id
                evidence_policy_id = static_metrics.evidence.policy_id
            else:
                geometry_family_id = motion.plan_id
                geometry_layout_id = motion.geometry_layout_id
                evidence_policy_id = motion.consistency_policy.policy_id
        else:
            topology_id = discretization.prepared_id
            geometry_id = discretization.prepared_id
            geometry_family_id = geometry_id
            geometry_layout_id = canonical_fingerprint(
                {
                    "kind": "static-finite-volume-runtime-geometry-layout",
                    "topology": topology_id,
                    "support": discretization.support.support_id,
                    "cell_layout": discretization.cell_space.layout.layout_id,
                    "effective_cell_volumes": array_tree_fingerprint(
                        np.asarray(bound_effective_volumes)
                    ),
                }
            )
            evidence_policy_id = canonical_fingerprint(
                {
                    "kind": "static-finite-volume-runtime-geometry-evidence",
                    "geometry_layout": geometry_layout_id,
                    "precision": dynamics.precision.policy_id,
                    "requirements": (
                        "all-cells-active",
                        "finite-positive-effective-cell-volumes",
                        "fixed-topology-and-geometry",
                    ),
                }
            )
        if isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
            static_flux_rate_block_templates = dynamics.stage_rate_block_templates
        elif isinstance(discretization, TriangleFiniteVolumeDiscretization):
            face_block = discretization.face_block
            static_flux_rate_block_templates = (
                ConservationStageFluxRateBlock(
                    jnp.zeros(
                        (
                            discretization.face_measures.size,
                            discretization.component_count,
                        ),
                        dtype=jnp.dtype(dynamics.precision.reduction_dtype),
                    ),
                    face_block.owner_cells,
                    face_block.neighbor_cells,
                    face_block.active_mask,
                    face_block.block_id,
                    "static-explicit-face",
                ),
            )
        else:
            if not isinstance(dynamics, PreparedFiniteVolumeDynamics):
                raise TypeError("Structured templates require prepared FV dynamics.")
            discretization = dynamics.discretization
            cell_shape = tuple(discretization.state_shape[:-1])
            cell_ids = np.arange(np.prod(cell_shape), dtype=np.int32).reshape(cell_shape)
            templates = []
            for axis, layout in enumerate(discretization.face_layouts):
                face_shape = tuple(layout.shape)
                periodic = discretization.grid.structured_axes[axis].periodic
                if periodic:
                    owners = np.roll(cell_ids, 1, axis=axis)
                    neighbors = cell_ids
                else:
                    owners = np.empty(face_shape, dtype=np.int32)
                    neighbors = np.full(face_shape, -1, dtype=np.int32)
                    lower_face: list[slice | int] = [slice(None)] * len(face_shape)
                    upper_face: list[slice | int] = [slice(None)] * len(face_shape)
                    interior_faces = [slice(None)] * len(face_shape)
                    lower_cells = [slice(None)] * len(cell_shape)
                    upper_cells = [slice(None)] * len(cell_shape)
                    lower_face[axis] = 0
                    upper_face[axis] = -1
                    interior_faces[axis] = slice(1, -1)
                    lower_cells[axis] = slice(0, -1)
                    upper_cells[axis] = slice(1, None)
                    owners[tuple(lower_face)] = cell_ids[tuple(lower_face)]
                    owners[tuple(upper_face)] = cell_ids[tuple(upper_face)]
                    owners[tuple(interior_faces)] = cell_ids[tuple(lower_cells)]
                    neighbors[tuple(interior_faces)] = cell_ids[tuple(upper_cells)]
                active = owners != neighbors
                block_id = canonical_fingerprint(
                    {
                        "kind": "static-structured-face-route",
                        "geometry_layout": geometry_layout_id,
                        "axis": axis,
                        "periodic": periodic,
                        "owner_cells": array_tree_fingerprint(owners),
                        "neighbor_cells": array_tree_fingerprint(neighbors),
                    }
                )
                templates.append(
                    ConservationStageFluxRateBlock(
                        jnp.zeros(
                            (int(np.prod(face_shape)), discretization.component_count),
                            dtype=jnp.dtype(dynamics.precision.reduction_dtype),
                        ),
                        owners.reshape((-1,)),
                        neighbors.reshape((-1,)),
                        active.reshape((-1,)),
                        block_id,
                        "static-structured-face",
                    )
                )
            static_flux_rate_block_templates = tuple(templates)
        native_overset_id = (
            None
            if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            or dynamics.coupling.overset is None
            else dynamics.coupling.overset.native_connectivity_id
        )
        epoch_topology_id = topology_id
        epoch_geometry_id = geometry_id
        if owner_local_view is not None:
            epoch_topology_id = owner_local_view.source_topology_id
            epoch_geometry_id = owner_local_view.source_geometry_id
        partition_id = (
            "serial" if owner_local_view is None else owner_local_view.distribution_id
        )
        default_topology_epoch = TopologyEpoch(
            0,
            epoch_geometry_id,
            epoch_topology_id,
            partition_id,
        )
        default_topology_artifacts = FiniteVolumeTopologyArtifacts(
            default_topology_epoch,
            discretization.prepared_id,
            topology_artifact_id=(
                native_overset_id
                if native_overset_id is not None
                else (
                    None
                    if sliding_initial_coupling is None
                    else sliding_initial_coupling.coupling_id
                )
            ),
            metrics_artifact_id=(
                None
                if sliding_initial_coupling is None
                else sliding_initial_coupling.evidence_id
            ),
            operators_artifact_id=(
                None if sliding_plan is None else sliding_plan.plan_id
            ),
        )
        if (topology_epoch is None) != (topology_artifacts is None):
            raise ValueError("Topology epochs and artifacts must be supplied together.")
        initial_topology_epoch = (
            default_topology_epoch if topology_epoch is None else topology_epoch
        )
        initial_topology_artifacts = (
            default_topology_artifacts
            if topology_artifacts is None
            else topology_artifacts
        )
        if not isinstance(initial_topology_epoch, TopologyEpoch):
            raise TypeError("topology_epoch must be TopologyEpoch.")
        if not isinstance(initial_topology_artifacts, FiniteVolumeTopologyArtifacts):
            raise TypeError("topology_artifacts must be FiniteVolumeTopologyArtifacts.")
        if (
            initial_topology_artifacts.epoch_id != initial_topology_epoch.epoch_id
            or initial_topology_artifacts.prepared_id != discretization.prepared_id
            or initial_topology_epoch.geometry_id != epoch_geometry_id
            or initial_topology_epoch.topology_id != epoch_topology_id
            or initial_topology_epoch.partition_id != partition_id
        ):
            raise ValueError("Reprepared topology does not match runtime geometry.")
        if (
            native_overset_id is not None
            and initial_topology_artifacts.topology_artifact_id != native_overset_id
        ):
            raise ValueError(
                "Initial topology artifacts do not bind the native overset connectivity."
            )
        topology_epoch_id = initial_topology_epoch.epoch_id
        if (
            embedded_metrics is not None
            and isinstance(discretization, UnstructuredFiniteVolumeDiscretization)
            and isinstance(embedded_boundaries, UnstructuredEmbeddedBoundarySet)
        ):
            embedded_stage_template = lower_embedded_stage_metrics(
                discretization,
                embedded_metrics,
                embedded_boundaries,
                topology_epoch_id,
                0,
                0,
                time=0.0,
            )
            geometry_layout_id = embedded_stage_template.geometry_layout_id
        self.dynamics = dynamics
        self.fallback_dynamics = fallback
        self.positivity = positivity
        self.precision = dynamics.precision
        self.effective_cell_volumes = bound_effective_volumes
        if (
            isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            and overset_effective_volumes is not None
        ):
            active_mask = dynamics.overset_active_cell_mask
        elif embedded_metrics is not None:
            active_mask = embedded_metrics.active_fluid_cells
        else:
            active_mask = jnp.ones(self.effective_cell_volumes.shape, dtype=jnp.bool_)
        self.active_cell_mask = jnp.asarray(active_mask, dtype=jnp.bool_)
        self.owner_local_geometry = owner_local_view
        self.owner_local_execution_group = owner_local_group
        self.owner_local_owned_cell_mask = owner_local_owned_mask
        self.owner_local_decision_provider = owner_local_decision_provider
        self.owner_local_scalar_min_provider = owner_local_scalar_min_provider
        self.owner_local_route_provider = owner_local_route_provider
        self.owner_local_axis_name = owner_axis

        self.policy = policy_
        self.stage_state_provider = stage_state_provider
        self.stage_flux_provider = stage_flux_provider
        self.topology_epoch_id = topology_epoch_id
        self.geometry_family_id = geometry_family_id
        self.geometry_layout_id = geometry_layout_id
        self.evidence_policy_id = evidence_policy_id
        self.static_flux_rate_block_templates = static_flux_rate_block_templates
        self.embedded_redistribution = embedded_redistribution
        self.embedded_stage_template = embedded_stage_template
        self.sliding_plan = sliding_plan
        self.sliding_initial_coupling = sliding_initial_coupling
        self.initial_topology_epoch = initial_topology_epoch
        self.initial_topology_artifacts = initial_topology_artifacts
        owner_local_identity = None
        if owner_local_view is not None:
            if (
                owner_local_group is None
                or owner_local_owned_mask is None
                or owner_local_decision_provider is None
            ):
                raise RuntimeError(
                    "Owner-local runtime identity lacks its prepared providers."
                )
            owner_local_identity = {
                "group": owner_local_group.spec.group_id,
                "view": owner_local_view.local_view_id,
                "owned_mask": array_tree_fingerprint(np.asarray(owner_local_owned_mask)),
                "decision_provider": owner_local_decision_provider.provider_id,
                "axis": owner_axis,
            }
            if owner_local_route_provider is not None:
                owner_local_identity["route_provider"] = (
                    owner_local_route_provider.provider_id
                )
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-volume-runtime",
                "dynamics": dynamics.dynamics_id,
                "fallback": fallback.dynamics_id,
                "positivity": positivity.plan_id,
                "policy": policy_.policy_id,
                "precision": dynamics.precision.policy_id,
                "topology_epoch": topology_epoch_id,
                "topology_artifacts": initial_topology_artifacts.artifacts_id,
                "geometry_family": geometry_family_id,
                "geometry_layout": geometry_layout_id,
                "evidence_policy": evidence_policy_id,
                "owner_local": owner_local_identity,
                "stage_state_provider": (
                    None
                    if stage_state_provider is None
                    else stage_state_provider.provider_id
                ),
                "stage_flux_provider": (
                    None
                    if stage_flux_provider is None
                    else stage_flux_provider.provider_id
                ),
                "embedded_redistribution": (
                    None
                    if embedded_redistribution is None
                    else embedded_redistribution.plan_id
                ),
                "sliding_plan": None if sliding_plan is None else sliding_plan.plan_id,
                "sliding_coupling": (
                    None
                    if sliding_initial_coupling is None
                    else sliding_initial_coupling.coupling_id
                ),
                "overset": (
                    None
                    if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
                    or dynamics.coupling.overset is None
                    else dynamics.coupling.overset.plan_id
                ),
                "overset_policy": (
                    None
                    if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
                    else dynamics.overset_policy_id
                ),
                "overset_mapping": (
                    None
                    if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
                    else dynamics.overset_mapping_id
                ),
                "overset_epoch": (
                    None
                    if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
                    else dynamics.overset_epoch_id
                ),
            }
        )

    def with_stage_state_provider(
        self,
        provider: FiniteVolumeStageStateProvider | None,
        /,
    ) -> PreparedFiniteVolumeRuntime:
        """Return an immutable runtime bound to one stage-state trace."""
        if provider is not None and not isinstance(
            provider, FiniteVolumeStageStateProvider
        ):
            raise TypeError("provider must be FiniteVolumeStageStateProvider or None.")
        if provider is self.stage_state_provider:
            return self
        return PreparedFiniteVolumeRuntime(
            self.dynamics,
            self.positivity,
            self.policy,
            topology_epoch=self.initial_topology_epoch,
            topology_artifacts=self.initial_topology_artifacts,
            stage_state_provider=provider,
            stage_flux_provider=self.stage_flux_provider,
        )

    def with_stage_flux_provider(
        self,
        provider: FiniteVolumeStageFluxProvider | None,
        /,
    ) -> PreparedFiniteVolumeRuntime:
        """Return an immutable runtime bound to one stage-specific flux trace."""

        if provider is not None and not isinstance(
            provider, FiniteVolumeStageFluxProvider
        ):
            raise TypeError("provider must be FiniteVolumeStageFluxProvider or None.")
        if provider is self.stage_flux_provider:
            return self
        return PreparedFiniteVolumeRuntime(
            self.dynamics,
            self.positivity,
            self.policy,
            topology_epoch=self.initial_topology_epoch,
            topology_artifacts=self.initial_topology_artifacts,
            stage_state_provider=self.stage_state_provider,
            stage_flux_provider=provider,
        )

    def _provide_stage_state(self, time: Array, state: Array, /) -> Array:
        provider = self.stage_state_provider
        if provider is None:
            return state
        return self.precision.storage(
            provider(
                self.precision.decision(time),
                self.precision.storage(state),
            )
        )

    def _effective_cell_volumes(self) -> Array:
        return self.effective_cell_volumes

    def _state_shape(self) -> tuple[int, ...]:
        return tuple(self.dynamics.discretization.state_shape)

    def _topology_journal_capacity(self) -> int:
        coupling = (
            self.dynamics.coupling
            if isinstance(self.dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            else None
        )
        return (
            coupling.topology_event_capacity
            if coupling is not None and coupling.topology_event_policy == "accepted_step"
            else 1
        )

    def _dynamics_cell_average(
        self,
        content_state: FiniteVolumeConservativeContentState,
        /,
    ) -> Array:
        return content_state.cell_average().reshape(self._state_shape())

    def _active_state_is_admissible(
        self,
        cell_average: Array,
        active_cell_mask: Array,
        /,
    ) -> Array:
        average = cell_average.reshape((-1, cell_average.shape[-1]))
        active = active_cell_mask.reshape((-1,))
        if average.shape[0] == 0:
            return jnp.asarray(True)
        first_active = jnp.argmax(active.astype(jnp.int32))

        def evaluate(_: None) -> Array:
            seed = average[first_active]
            safe_average = jnp.where(active[:, None], average, seed[None, :])
            return jnp.all(
                jnp.where(
                    active,
                    self.dynamics.system.admissible(safe_average),
                    True,
                )
            )

        return jax.lax.cond(
            jnp.any(active),
            evaluate,
            lambda _: jnp.asarray(True),
            operand=None,
        )

    @checked
    def _validate_state(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        /,
    ) -> None:
        if not isinstance(runtime_state, FiniteVolumeRuntimeState):
            raise TypeError("runtime_state must be FiniteVolumeRuntimeState.")
        content = runtime_state.content_state
        journal = runtime_state.topology_journal
        if content.precision.policy_id != self.precision.policy_id:
            raise ValueError("Runtime content precision does not match the runtime.")
        if (
            journal.capacity < self._topology_journal_capacity()
            or journal.current_epoch_id != self.initial_topology_epoch.epoch_id
            or content.topology_epoch_id != journal.current_epoch_id
        ):
            raise ValueError("Runtime topology journal does not match the runtime.")
        if not journal.artifact_table:
            raise ValueError("Runtime topology journal has no current artifacts.")
        current_artifacts = journal.artifact_table[-1]
        if (
            current_artifacts.epoch_id != self.initial_topology_epoch.epoch_id
            or current_artifacts.prepared_id
            != self.initial_topology_artifacts.prepared_id
        ):
            raise ValueError("Runtime topology artifacts do not match the runtime.")
        if (
            isinstance(self.dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            and self.dynamics.coupling.overset is not None
            and self.dynamics.coupling.overset.native_connectivity_id is not None
            and current_artifacts.topology_artifact_id
            != self.dynamics.coupling.overset.native_connectivity_id
        ):
            raise ValueError(
                "Runtime topology journal does not bind the prepared native overset connectivity."
            )
        if content.topology_epoch_id != self.topology_epoch_id:
            raise ValueError("Runtime content topology epoch does not match the runtime.")
        if content.geometry_family_id != self.geometry_family_id:
            raise ValueError(
                "Runtime content geometry family does not match the runtime."
            )
        if content.geometry_layout_id != self.geometry_layout_id:
            raise ValueError(
                "Runtime content geometry layout does not match the runtime."
            )
        if content.evidence_policy_id != self.evidence_policy_id:
            raise ValueError(
                "Runtime content evidence policy does not match the runtime."
            )
        expected_content_shape = (
            int(np.prod(self._state_shape()[:-1])),
            self._state_shape()[-1],
        )
        if content.conservative_content.shape != expected_content_shape:
            raise ValueError(
                "Runtime content coordinates do not match the finite-volume layout."
            )
        if (
            content.effective_cell_volumes.shape != self.effective_cell_volumes.shape
            or content.active_cell_mask.shape != self.active_cell_mask.shape
        ):
            raise ValueError(
                "Runtime content metrics do not match the finite-volume layout."
            )
        if self.owner_local_geometry is not None:
            owner_mask = self.owner_local_owned_cell_mask
            if owner_mask is None:
                raise ValueError("Owner-local runtime lacks its prepared ownership mask.")
            if (
                content.owned_cell_mask is None
                or content.owned_cell_mask.shape != owner_mask.shape
                or not np.array_equal(
                    np.asarray(content.owned_cell_mask, dtype=np.bool_),
                    np.asarray(owner_mask, dtype=np.bool_),
                )
            ):
                raise ValueError(
                    "Runtime content ownership does not match its accepted local cell view."
                )
        elif content.owned_cell_mask is not None and not bool(
            np.all(np.asarray(content.owned_cell_mask))
        ):
            raise ValueError(
                "Partially owned content requires its accepted owner-local geometry binding."
            )
        if self.sliding_plan is not None:
            self._validate_sliding_state(runtime_state)

    def _owner_local_runtime_cell_ids(self) -> np.ndarray:
        if not isinstance(self.owner_local_geometry, _owner_local_view_types()):
            raise ValueError("Owner-local runtime lacks its typed geometry view.")
        if not isinstance(self.dynamics, PreparedUnstructuredFiniteVolumeDynamics):
            raise TypeError("Owner-local runtime requires unstructured FV dynamics.")
        discretization = self.dynamics.discretization
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError("Owner-local runtime requires unstructured cell geometry.")
        cell_ids = np.asarray(discretization.cell_global_ids, dtype=np.int64)
        view_ids = np.asarray(self.owner_local_geometry.cell_ids, dtype=np.int64)
        if (
            cell_ids.ndim != 1
            or np.unique(cell_ids).size != cell_ids.size
            or view_ids.shape != cell_ids.shape
            or not np.array_equal(np.sort(view_ids), np.sort(cell_ids))
        ):
            raise ValueError(
                "Runtime plan cells differ from its accepted owner-local view."
            )
        return cell_ids

    @checked
    def owner_local_cell_ids(self) -> np.ndarray:
        """Return exact runtime row IDs for owner-local accepted state arrays."""
        return self._owner_local_runtime_cell_ids()

    def _owner_local_state_rows(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        source_cell_ids: ArrayLike,
        /,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(runtime_state, FiniteVolumeRuntimeState):
            raise TypeError("runtime_state must be FiniteVolumeRuntimeState.")
        raw_ids = np.asarray(source_cell_ids)
        if (
            raw_ids.dtype.kind not in "iu"
            or raw_ids.ndim != 1
            or raw_ids.shape != (runtime_state.content_state.cell_count,)
            or np.unique(raw_ids).size != raw_ids.size
            or (raw_ids.size and np.any(raw_ids < 0))
            or (
                raw_ids.dtype.kind == "u"
                and raw_ids.size
                and np.any(raw_ids > np.iinfo(np.int64).max)
            )
        ):
            raise ValueError("Source FV rows require unique nonnegative stable cell IDs.")
        source_ids = np.asarray(raw_ids, dtype=np.int64)
        runtime_ids = self._owner_local_runtime_cell_ids()
        if source_ids.shape != runtime_ids.shape or not np.array_equal(
            np.sort(source_ids), np.sort(runtime_ids)
        ):
            raise ValueError(
                "Source stable cell IDs differ from the prepared owner-local plan."
            )
        source_rows = {int(identifier): row for row, identifier in enumerate(source_ids)}
        runtime_rows = np.asarray(
            [source_rows[int(identifier)] for identifier in runtime_ids],
            dtype=np.int32,
        )
        return source_ids, runtime_ids, runtime_rows

    def _runtime_ordered_state(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        source_cell_ids: ArrayLike,
        /,
    ) -> tuple[FiniteVolumeRuntimeState, np.ndarray]:
        source_ids, runtime_ids, runtime_rows = self._owner_local_state_rows(
            runtime_state, source_cell_ids
        )
        if np.array_equal(source_ids, runtime_ids):
            return runtime_state, runtime_ids
        source = runtime_state.content_state
        rows = jnp.asarray(runtime_rows, dtype=jnp.int32)
        content = FiniteVolumeConservativeContentState(
            jnp.asarray(source.conservative_content)[rows],
            jnp.asarray(source.effective_cell_volumes)[rows],
            jnp.asarray(source.active_cell_mask)[rows],
            source.time,
            topology_epoch_id=source.topology_epoch_id,
            geometry_family_id=source.geometry_family_id,
            geometry_layout_id=source.geometry_layout_id,
            geometry_version=source.geometry_version,
            evidence_policy_id=source.evidence_policy_id,
            evidence_version=source.evidence_version,
            precision=source.precision,
            owned_cell_mask=(
                None
                if source.owned_cell_mask is None
                else jnp.asarray(source.owned_cell_mask)[rows]
            ),
        )
        return (
            FiniteVolumeRuntimeState(
                content,
                runtime_state.topology_journal,
                runtime_state.step_size,
                accepted_step=runtime_state.accepted_step,
                last_status=runtime_state.last_status,
                controller_state=runtime_state.controller_state,
                integrator_state=runtime_state.integrator_state,
                output_cursor=runtime_state.output_cursor,
                sliding_coupling=runtime_state.sliding_coupling,
                sliding_shift=runtime_state.sliding_shift,
                sliding_event_id=runtime_state.sliding_event_id,
            ),
            runtime_ids,
        )

    @checked
    def accepted_state_id(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        source_cell_ids: ArrayLike,
        /,
    ) -> str:
        """Hash core rows with their explicit IDs; external banks are separate.

        The root binds this runtime/view and source geometry, keyed conservative
        content, volumes, activity, owner-mask presence/data, complete journal,
        accepted controls, and sliding state. It is stable under row order.
        """
        cell_ids, _, _ = self._owner_local_state_rows(runtime_state, source_cell_ids)
        content = runtime_state.content_state
        order = np.argsort(cell_ids, kind="stable")
        journal = runtime_state.topology_journal
        journal_static = {
            "current_epoch_id": journal.current_epoch_id,
            "epochs": tuple(
                (
                    epoch.index,
                    epoch.geometry_id,
                    epoch.topology_id,
                    epoch.partition_id,
                    epoch.epoch_id,
                )
                for epoch in journal.epoch_table
            ),
            "artifacts": tuple(
                artifact.artifacts_id for artifact in journal.artifact_table
            ),
            "input_epoch_ids": journal.input_epoch_ids,
            "requested_ids": journal.requested_ids,
            "result_ids": journal.result_ids,
            "payload_ids": journal.payload_ids,
            "capacity": journal.capacity,
            "journal_id": journal.journal_id,
        }
        return canonical_fingerprint(
            {
                "kind": "finite-volume-accepted-runtime-state",
                "runtime_id": self.runtime_id,
                "owner_local_view_id": self.owner_local_geometry.local_view_id,
                "source_topology_id": self.owner_local_geometry.source_topology_id,
                "source_geometry_id": self.owner_local_geometry.source_geometry_id,
                "distribution_id": self.owner_local_geometry.distribution_id,
                "cell_ids": cell_ids[order],
                "conservative_content": array_tree_fingerprint(
                    np.asarray(content.conservative_content)[order]
                ),
                "effective_cell_volumes": array_tree_fingerprint(
                    np.asarray(content.effective_cell_volumes)[order]
                ),
                "active_cell_mask": array_tree_fingerprint(
                    np.asarray(content.active_cell_mask)[order]
                ),
                "owned_cell_mask_present": content.owned_cell_mask is not None,
                "owned_cell_mask": (
                    None
                    if content.owned_cell_mask is None
                    else array_tree_fingerprint(
                        np.asarray(content.owned_cell_mask)[order]
                    )
                ),
                "content": {
                    "time": array_tree_fingerprint(content.time),
                    "geometry_version": array_tree_fingerprint(content.geometry_version),
                    "evidence_version": array_tree_fingerprint(content.evidence_version),
                    "topology_epoch_id": content.topology_epoch_id,
                    "geometry_family_id": content.geometry_family_id,
                    "geometry_layout_id": content.geometry_layout_id,
                    "evidence_policy_id": content.evidence_policy_id,
                    "precision_policy_id": content.precision.policy_id,
                },
                "topology_journal": {
                    "arrays": array_tree_fingerprint(
                        (
                            journal.kinds,
                            journal.states,
                            journal.statuses,
                            journal.accepted_steps,
                            journal.times,
                            journal.next_sequence,
                            journal.count,
                            journal.overflowed,
                        )
                    ),
                    "static": journal_static,
                },
                "accepted_step": array_tree_fingerprint(runtime_state.accepted_step),
                "step_size": array_tree_fingerprint(runtime_state.step_size),
                "last_status": array_tree_fingerprint(runtime_state.last_status),
                "controller_state": array_tree_fingerprint(
                    runtime_state.controller_state
                ),
                "integrator_state": array_tree_fingerprint(
                    runtime_state.integrator_state
                ),
                "output_cursor": array_tree_fingerprint(runtime_state.output_cursor),
                "sliding": {
                    "coupling_id": runtime_state.sliding_coupling_id,
                    "coupling": (
                        None
                        if runtime_state.sliding_coupling is None
                        else array_tree_fingerprint(runtime_state.sliding_coupling)
                    ),
                    "shift": array_tree_fingerprint(runtime_state.sliding_shift),
                    "event_id": runtime_state.sliding_event_id,
                },
            }
        )

    @checked
    def validate_restored_accepted_state(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        source_cell_ids: ArrayLike,
        /,
        *,
        expected_runtime_id: str,
        expected_epoch_id: str,
        expected_state_id: str | None = None,
    ) -> FiniteVolumeValidatedRuntimeState:
        """Validate a decoded accepted shard against this exact source runtime."""
        from .._execution_runtime import ExecutionGroup
        from ..discretization._view_support import (
            OwnerLocalCellGeometry,
            OwnerLocalCompositeCellGeometry,
        )
        from ..meshing._distribution import _owner_group_fail_if_any

        execution_group = self.owner_local_execution_group
        if not isinstance(execution_group, ExecutionGroup):
            raise TypeError("Restored owner-local state requires its ExecutionGroup.")
        owner_view: OwnerLocalCellGeometry | OwnerLocalCompositeCellGeometry | None = None
        normalized_state: FiniteVolumeRuntimeState | None = None
        runtime_cell_ids: np.ndarray | None = None
        state_id: str | None = None
        local_error: str | None = None
        try:
            owner_view = self.owner_local_geometry
            if not isinstance(owner_view, _owner_local_view_types()):
                raise TypeError(
                    "Restored owner-local state requires its source geometry view."
                )
            state_id = self.accepted_state_id(runtime_state, source_cell_ids)
            if expected_state_id is not None and state_id != expected_state_id:
                raise ValueError("Restored source state identity changed.")
            normalized_state, runtime_cell_ids = self._runtime_ordered_state(
                runtime_state, source_cell_ids
            )
            self._validate_state(normalized_state)
            if expected_runtime_id != self.runtime_id:
                raise ValueError("Restored source runtime identity changed.")
            if (
                expected_epoch_id != self.topology_epoch_id
                or normalized_state.topology_journal.current_epoch_id != expected_epoch_id
            ):
                raise ValueError(
                    "Restored source epoch differs from its prepared runtime."
                )
            if (
                owner_view.source_topology_id != self.initial_topology_epoch.topology_id
                or owner_view.source_geometry_id
                != self.initial_topology_epoch.geometry_id
                or owner_view.distribution_id != self.initial_topology_epoch.partition_id
            ):
                raise ValueError("Restored source view differs from its runtime epoch.")
            status = int(np.asarray(normalized_state.last_status))
            if status not in (
                int(FiniteVolumeRunStatus.SUCCESS),
                int(FiniteVolumeRunStatus.RECOVERED_REJECTION),
            ):
                raise ValueError("Restored runtime state is not an accepted boundary.")
            if (
                int(np.asarray(normalized_state.accepted_step)) < 0
                or int(np.asarray(normalized_state.output_cursor)) < 0
                or not np.isfinite(float(np.asarray(normalized_state.time)))
                or not np.isfinite(float(np.asarray(normalized_state.step_size)))
                or float(np.asarray(normalized_state.step_size)) <= 0.0
            ):
                raise ValueError(
                    "Restored runtime controls are outside the accepted domain."
                )
            content = normalized_state.content_state
            if not np.array_equal(
                np.asarray(content.effective_cell_volumes),
                np.asarray(self.effective_cell_volumes),
            ) or not np.array_equal(
                np.asarray(content.active_cell_mask),
                np.asarray(self.active_cell_mask),
            ):
                raise ValueError(
                    "Restored FV state differs from the prepared source geometry."
                )
            self.precision.validate_state(content.conservative_content)
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            f"{self.owner_local_axis_name}_restored_state_admission",
            "restored accepted finite-volume source state",
        )
        if owner_view is None:
            raise ValueError("Restored source geometry view admission produced no view.")
        if normalized_state is None or runtime_cell_ids is None or state_id is None:
            raise ValueError(
                "Restored source-state normalization produced incomplete typed data."
            )
        if not isinstance(self.dynamics, PreparedUnstructuredFiniteVolumeDynamics):
            raise TypeError(
                "Restored owner-local FV geometry requires unstructured dynamics."
            )
        authority_error: str | None = None
        try:
            _require_owner_local_fv_geometry_current(
                owner_view,
                self.dynamics,
                execution_group,
                f"{self.owner_local_axis_name}_restored_source_directory",
            )
        except Exception as error:
            authority_error = f"{type(error).__name__}: {error}"
        _owner_group_fail_if_any(
            authority_error,
            execution_group,
            f"{self.owner_local_axis_name}_restored_source_authority",
            "restored source geometry authority",
        )
        local_admissible: Array | None = None
        admissibility_error: str | None = None
        try:
            content = normalized_state.content_state
            active = content.active_cell_mask
            owner_mask = self.owner_local_owned_cell_mask
            if owner_mask is None:
                raise ValueError("Restored owner-local runtime has no ownership mask.")
            active = active & owner_mask
            local_admissible = self._active_state_is_admissible(
                content.cell_average(),
                active,
            )
        except Exception as error:
            admissibility_error = f"{type(error).__name__}: {error}"
        _owner_group_fail_if_any(
            admissibility_error,
            execution_group,
            f"{self.owner_local_axis_name}_restored_state_admissibility",
            "restored accepted finite-volume source admissibility",
        )
        if local_admissible is None:
            raise ValueError("Restored source admissibility produced no typed result.")
        valid = local_admissible
        if self.owner_local_decision_provider is None:
            raise ValueError(
                "Restored owner-local runtime has no collective decision provider."
            )
        valid = self.owner_local_decision_provider(valid)
        if not bool(np.asarray(valid)):
            raise ValueError(
                "Restored FV state is not admissible on its owned active cells."
            )
        return FiniteVolumeValidatedRuntimeState(
            normalized_state,
            runtime_cell_ids,
            self.runtime_id,
            self.topology_epoch_id,
            state_id,
        )

    def reprepare_for_epoch(
        self,
        topology_epoch: TopologyEpoch,
        topology_artifacts: FiniteVolumeTopologyArtifacts,
        /,
    ) -> PreparedFiniteVolumeRuntime:
        """Rebind the fixed geometry runtime to a validated realized topology."""
        return PreparedFiniteVolumeRuntime(
            self.dynamics,
            self.positivity,
            self.policy,
            topology_epoch=topology_epoch,
            topology_artifacts=topology_artifacts,
            stage_state_provider=(
                None
                if self.owner_local_geometry is not None
                else self.stage_state_provider
            ),
            stage_flux_provider=self.stage_flux_provider,
            execution_group=self.owner_local_execution_group,
            owner_local_geometry=self.owner_local_geometry,
            owner_local_axis_name=(
                "finite_volume_owner_local"
                if not self.owner_local_axis_name
                else self.owner_local_axis_name
            ),
        )

    def initialize_state(
        self,
        cell_average: ArrayLike,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        motion_args: Any = None,
        accepted_step: ArrayLike = 0,
        last_status: ArrayLike = FiniteVolumeRunStatus.SUCCESS,
        controller_state: ArrayLike | None = None,
        integrator_state: ArrayLike | None = None,
        output_cursor: ArrayLike = 0,
    ) -> FiniteVolumeRuntimeState:
        time_value = self.precision.decision(time)
        volumes = self._effective_cell_volumes()
        if (
            isinstance(self.dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            and self.dynamics.coupling.motion is not None
        ):
            if time_value.shape != () or float(np.asarray(time_value)) != 0.0:
                raise ValueError(
                    "Fresh moving finite-volume initialization requires time exactly "
                    "t=0; use checkpoint/restart for nonzero starts."
                )
            motion = self.dynamics.coupling.motion
            evaluated_plan = motion.evaluate_plan(time_value, motion_args)
            evaluated_discretization = evaluated_plan.prepare()
            base_discretization = self.dynamics.discretization
            if evaluated_discretization.topology_id != base_discretization.topology_id:
                raise ValueError(
                    "Motion evaluated at t=0 must match the compiled base topology."
                )

            absolute_tolerance = motion.consistency_policy.absolute_tolerance
            relative_tolerance = motion.consistency_policy.relative_tolerance

            def matches_base_geometry(evaluated: Array, compiled: Array, /) -> bool:
                evaluated_array = np.asarray(evaluated)
                compiled_array = np.asarray(compiled)
                if evaluated_array.shape != compiled_array.shape:
                    return False
                reference = np.maximum(
                    np.abs(evaluated_array),
                    np.abs(compiled_array),
                )
                tolerance = absolute_tolerance + relative_tolerance * reference
                return bool(
                    np.all(np.isfinite(evaluated_array))
                    and np.all(np.isfinite(compiled_array))
                    and np.all(np.abs(evaluated_array - compiled_array) <= tolerance)
                )

            if not matches_base_geometry(
                evaluated_discretization.vertices,
                base_discretization.vertices,
            ):
                raise ValueError(
                    "Motion evaluated at t=0 must match the compiled base geometry within ALE precision evidence."
                )
            if not matches_base_geometry(
                evaluated_discretization.cell_volumes,
                volumes,
            ):
                raise ValueError(
                    "Motion evaluated at t=0 must reproduce the compiled base geometry "
                    "cell volumes within ALE precision evidence."
                )

        journal_capacity = self._topology_journal_capacity()
        topology_journal = FiniteVolumeTopologyEventJournal.allocate(
            self.initial_topology_epoch,
            self.initial_topology_artifacts,
            capacity=journal_capacity,
            time=time_value,
        )
        average = self.precision.storage(cell_average)
        state_shape = self._state_shape()
        if average.shape != state_shape:
            raise ValueError(
                f"Finite-volume cell average must have exact shape {state_shape}."
            )
        flattened = average.reshape((-1, state_shape[-1]))
        active = self.active_cell_mask
        content_state = FiniteVolumeConservativeContentState.from_cell_average(
            flattened,
            volumes,
            active,
            time_value,
            topology_epoch_id=self.topology_epoch_id,
            geometry_family_id=self.geometry_family_id,
            geometry_layout_id=self.geometry_layout_id,
            geometry_version=jnp.asarray(0, dtype=jnp.int32),
            evidence_policy_id=self.evidence_policy_id,
            evidence_version=jnp.asarray(0, dtype=jnp.int32),
            precision=self.precision,
            owned_cell_mask=self.owner_local_owned_cell_mask,
        )
        if self.owner_local_decision_provider is not None:
            refreshed_average = self._provide_stage_state(
                time_value,
                content_state.cell_average(),
            )
            refreshed_content = (
                refreshed_average
                * content_state.effective_cell_volumes.reshape(
                    (-1,) + (1,) * (refreshed_average.ndim - 1)
                )
            )
            content_state = content_state.with_content(refreshed_content)
        return FiniteVolumeRuntimeState(
            content_state,
            topology_journal,
            self.precision.decision(step_size),
            accepted_step=accepted_step,
            last_status=last_status,
            controller_state=controller_state,
            integrator_state=integrator_state,
            output_cursor=output_cursor,
            sliding_coupling=self.sliding_initial_coupling,
            sliding_shift=(
                0.0
                if self.sliding_initial_coupling is None
                else self.sliding_initial_coupling.normalized_shift
            ),
        )

    def _face_measures(self) -> tuple[Array, ...]:
        discretization = self.dynamics.discretization
        if isinstance(
            discretization,
            (
                TriangleFiniteVolumeDiscretization,
                UnstructuredFiniteVolumeDiscretization,
                DyadicFiniteVolumeDiscretization,
            ),
        ):
            return (discretization.face_measures,)
        return tuple(discretization.face_measures)

    def _face_fluxes(
        self,
        dynamics: (
            PreparedFiniteVolumeDynamics
            | PreparedTriangleFiniteVolumeDynamics
            | PreparedUnstructuredFiniteVolumeDynamics
        ),
        time: Array,
        state: Array,
        args: Any,
        /,
    ) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
        if isinstance(
            dynamics,
            (
                PreparedTriangleFiniteVolumeDynamics,
                PreparedUnstructuredFiniteVolumeDynamics,
            ),
        ):
            flux, speed = dynamics.face_fluxes(time, state, args)
            return (flux,), (speed,)
        fluxes, speeds = dynamics.face_fluxes(time, state, args)
        return fluxes, speeds

    def _flux_residual(
        self,
        dynamics: (
            PreparedFiniteVolumeDynamics
            | PreparedTriangleFiniteVolumeDynamics
            | PreparedUnstructuredFiniteVolumeDynamics
        ),
        fluxes: tuple[Array, ...],
        /,
    ) -> Array:
        if isinstance(
            dynamics,
            (
                PreparedTriangleFiniteVolumeDynamics,
                PreparedUnstructuredFiniteVolumeDynamics,
            ),
        ):
            if len(fluxes) != 1:
                raise ValueError("Explicit-face runtime requires one face-flux block.")
            return dynamics.residual_from_fluxes(fluxes[0])
        return dynamics._flux_residual(fluxes)

    def _zero_static_flux_rates(self) -> tuple[Array, ...]:
        discretization = self.dynamics.discretization
        dtype = jnp.dtype(self.precision.reduction_dtype)
        if isinstance(
            discretization,
            (
                TriangleFiniteVolumeDiscretization,
                UnstructuredFiniteVolumeDiscretization,
                DyadicFiniteVolumeDiscretization,
            ),
        ):
            return (
                jnp.zeros(
                    (
                        discretization.face_measures.size,
                        discretization.component_count,
                    ),
                    dtype=dtype,
                ),
            )
        return tuple(
            jnp.zeros(
                layout.shape + (discretization.component_count,),
                dtype=dtype,
            )
            for layout in discretization.face_layouts
        )

    def _zero_shallow_water_integrals(
        self,
    ) -> ShallowWaterAcceptedFaceIntegrals | None:
        if not isinstance(self.dynamics, PreparedFiniteVolumeDynamics) or not isinstance(
            self.dynamics.method.interface_solver,
            ShallowWaterHydrostaticHLLPlan,
        ):
            return None
        if not isinstance(self.dynamics.bathymetry, PreparedShallowWaterBathymetry):
            raise TypeError(
                "Hydrostatic shallow-water runtime requires prepared bathymetry."
            )
        dtype = jnp.dtype(self.precision.reduction_dtype)
        contributions = []
        for layout in self.dynamics.discretization.face_layouts:
            shape = layout.shape + (self.dynamics.discretization.component_count,)
            state = jnp.zeros(shape, dtype=dtype)
            speed = jnp.zeros(layout.shape, dtype=dtype)
            contributions.append(
                ShallowWaterBalancedFaceResult(
                    state,
                    state,
                    state,
                    speed,
                    state,
                    state,
                    jnp.ones(layout.shape, dtype=jnp.bool_),
                )
            )
        return ShallowWaterAcceptedFaceIntegrals(
            tuple(contributions),
            tuple(self.dynamics.discretization.face_measures),
            jnp.asarray(0.0, dtype=dtype),
            axis_names=self.dynamics.discretization.grid.axis_names,
            bed_id=self.dynamics.bathymetry.bed_id,
            plan_id=self.dynamics.method.interface_solver.plan_id,
        )

    def _static_accepted_flux_integral_ledger(
        self,
        original_content: FiniteVolumeConservativeContentState,
        next_content: FiniteVolumeConservativeContentState,
        integrated_flux_rates: tuple[Array, ...],
        step_size: Array,
        end_time: Array,
        accepted_step: Array,
        /,
    ) -> AcceptedConservationIntegralLedger:
        if len(integrated_flux_rates) != len(self.static_flux_rate_block_templates):
            raise ValueError("Static flux rates must match the prepared ledger routes.")
        discretization = self.dynamics.discretization
        step = self.precision.reduction(step_size)
        blocks = []
        for axis, (integrated_rate, template) in enumerate(
            zip(
                integrated_flux_rates,
                self.static_flux_rate_block_templates,
                strict=True,
            )
        ):
            integral = step * self.precision.reduction(integrated_rate)
            if (
                not isinstance(
                    discretization,
                    (
                        TriangleFiniteVolumeDiscretization,
                        UnstructuredFiniteVolumeDiscretization,
                        DyadicFiniteVolumeDiscretization,
                    ),
                )
                and not discretization.grid.structured_axes[axis].periodic
            ):
                lower_face: list[slice | int] = [slice(None)] * integral.ndim
                lower_face[axis] = 0
                integral = integral.at[tuple(lower_face)].multiply(-1)
            blocks.append(
                AcceptedConservationFluxIntegralBlock._from_stage_rate_block(
                    integral.reshape(template.flux_rate.shape),
                    template,
                )
            )
        content_change = self.precision.reduction(
            next_content.conservative_content
        ) - self.precision.reduction(original_content.conservative_content)
        face_change = jnp.zeros_like(content_change)
        for block in blocks:
            safe_neighbor = jnp.maximum(block.neighbor_cells, 0)
            face_change = face_change.at[block.owner_cells].add(-block.flux_integral)
            face_change = face_change.at[safe_neighbor].add(
                jnp.where(
                    (block.neighbor_cells >= 0)[:, None],
                    block.flux_integral,
                    jnp.zeros((), dtype=block.flux_integral.dtype),
                )
            )
        source_integral = content_change - face_change
        return AcceptedConservationIntegralLedger(
            tuple(blocks),
            source_integral,
            original_content.active_cell_mask,
            geometry_family_id=original_content.geometry_family_id,
            geometry_layout_id=original_content.geometry_layout_id,
            stage_geometry_versions=(
                original_content.geometry_version,
                original_content.geometry_version,
                original_content.geometry_version,
            ),
            start_geometry_version=original_content.geometry_version,
            end_geometry_version=next_content.geometry_version,
            evidence_policy_id=original_content.evidence_policy_id,
            stage_evidence_versions=(
                original_content.evidence_version,
                original_content.evidence_version,
                original_content.evidence_version,
            ),
            start_evidence_version=original_content.evidence_version,
            end_evidence_version=next_content.evidence_version,
            start_topology_epoch_id=original_content.topology_epoch_id,
            end_topology_epoch_id=next_content.topology_epoch_id,
            start_time=original_content.time,
            end_time=end_time,
            accepted_step=accepted_step,
        )

    def _limited_euler(
        self,
        stage_index: int,
        time: Array,
        evaluation_state: Array,
        combination_base: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> tuple[
        PositivityBlendResult | BalancedPositivityBlendResult,
        FiniteVolumeStageFlux | None,
    ]:
        if isinstance(
            self.dynamics,
            PreparedFiniteVolumeDynamics,
        ) and isinstance(
            self.dynamics.method.interface_solver,
            ShallowWaterHydrostaticHLLPlan,
        ):
            if not isinstance(self.fallback_dynamics, PreparedFiniteVolumeDynamics):
                raise TypeError(
                    "Hydrostatic shallow-water fallback must use structured dynamics."
                )
            high_contributions = self.dynamics.balanced_face_contributions(
                time, evaluation_state, args
            )
            fallback_contributions = self.fallback_dynamics.balanced_face_contributions(
                time, evaluation_state, args
            )
            high_residual = self.precision.reduction(
                self.dynamics(time, evaluation_state, args)
            )
            common_residual = self.precision.storage(
                high_residual
                - self.precision.reduction(
                    self.dynamics._balanced_residual(high_contributions)
                )
            )
            result = self.positivity.limit_balanced_face_contributions(
                self.dynamics.system,
                combination_base,
                high_contributions,
                fallback_contributions,
                common_residual,
                step_size,
                self.dynamics.discretization,
            )
            return result, None
        high_fluxes, _ = self._face_fluxes(self.dynamics, time, evaluation_state, args)
        fallback_fluxes, _ = self._face_fluxes(
            self.fallback_dynamics, time, evaluation_state, args
        )
        high_residual = self.precision.reduction(
            self.dynamics(time, evaluation_state, args)
        )
        common_residual = self.precision.storage(
            high_residual
            - self.precision.reduction(self._flux_residual(self.dynamics, high_fluxes))
        )
        stage_flux = None
        if self.stage_flux_provider is not None:
            stage_flux = self.stage_flux_provider.evaluate(
                stage_index,
                self.precision.decision(time),
                self.precision.storage(evaluation_state),
                high_fluxes,
            )
            high_fluxes = tuple(
                jnp.where(mask[..., None], replacement, original)
                for replacement, mask, original in zip(
                    stage_flux.replacement_normal_fluxes,
                    stage_flux.replacement_masks,
                    high_fluxes,
                    strict=True,
                )
            )
            fallback_fluxes = tuple(
                jnp.where(mask[..., None], replacement, original)
                for replacement, mask, original in zip(
                    stage_flux.replacement_normal_fluxes,
                    stage_flux.replacement_masks,
                    fallback_fluxes,
                    strict=True,
                )
            )
        result = self.positivity.limit_face_fluxes(
            self.dynamics.system,
            combination_base,
            high_fluxes,
            fallback_fluxes,
            common_residual,
            step_size,
            self.dynamics.discretization,
        )
        return result, stage_flux

    def _precision_report(
        self,
        report: FiniteVolumeAdmissibilityReport,
        /,
    ) -> FiniteVolumeAdmissibilityReport:
        return jax.tree.map(
            lambda value: (
                self.precision.reduction(value) if eqx.is_inexact_array(value) else value
            ),
            report,
        )

    def _candidate_with_stage_flux_trace(
        self,
        time: Array,
        state: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> tuple[
        PositivityBlendResult | BalancedPositivityBlendResult,
        FiniteVolumeStageFluxTrace | None,
    ]:
        stage_initial = self._provide_stage_state(time, state)
        first, first_stage_flux = self._limited_euler(
            0, time, stage_initial, stage_initial, step_size, args
        )
        second_base = self.precision.storage(
            0.75 * self.precision.reduction(stage_initial)
            + 0.25 * self.precision.reduction(first.state)
        )
        second_time = self.precision.decision(time + step_size)
        second_state = self._provide_stage_state(
            second_time,
            self.precision.storage(first.state),
        )
        second, second_stage_flux = self._limited_euler(
            1,
            second_time,
            second_state,
            second_base,
            self.precision.decision(0.25 * step_size),
            args,
        )
        third_base = self.precision.storage(
            (1.0 / 3.0) * self.precision.reduction(stage_initial)
            + (2.0 / 3.0) * self.precision.reduction(second.state)
        )
        third_time = self.precision.decision(time + 0.5 * step_size)
        third_state = self._provide_stage_state(
            third_time,
            self.precision.storage(second.state),
        )
        third, third_stage_flux = self._limited_euler(
            2,
            third_time,
            third_state,
            third_base,
            self.precision.decision((2.0 / 3.0) * step_size),
            args,
        )
        integrated_flux_rates = tuple(
            self.precision.reduction(
                (1.0 / 6.0) * self.precision.reduction(first_flux)
                + (1.0 / 6.0) * self.precision.reduction(second_flux)
                + (2.0 / 3.0) * self.precision.reduction(third_flux)
            )
            for first_flux, second_flux, third_flux in zip(
                first.integrated_fluxes,
                second.integrated_fluxes,
                third.integrated_fluxes,
                strict=True,
            )
        )
        normal_fluxes = tuple(
            self.precision.flux(
                integrated_rate / self.precision.reduction(measure[..., None])
            )
            for integrated_rate, measure in zip(
                integrated_flux_rates,
                self._face_measures(),
                strict=True,
            )
        )
        if isinstance(third, BalancedPositivityBlendResult):
            if not isinstance(first, BalancedPositivityBlendResult) or not isinstance(
                second, BalancedPositivityBlendResult
            ):
                raise TypeError(
                    "All hydrostatic shallow-water stages must use balanced positivity."
                )
            accepted_contributions = tuple(
                ShallowWaterBalancedFaceResult(
                    self.precision.flux(
                        (1.0 / 6.0)
                        * self.precision.reduction(first_contribution.normal_flux)
                        + (1.0 / 6.0)
                        * self.precision.reduction(second_contribution.normal_flux)
                        + (2.0 / 3.0)
                        * self.precision.reduction(third_contribution.normal_flux)
                    ),
                    self.precision.flux(
                        (1.0 / 6.0)
                        * self.precision.reduction(first_contribution.left_correction)
                        + (1.0 / 6.0)
                        * self.precision.reduction(second_contribution.left_correction)
                        + (2.0 / 3.0)
                        * self.precision.reduction(third_contribution.left_correction)
                    ),
                    self.precision.flux(
                        (1.0 / 6.0)
                        * self.precision.reduction(first_contribution.right_correction)
                        + (1.0 / 6.0)
                        * self.precision.reduction(second_contribution.right_correction)
                        + (2.0 / 3.0)
                        * self.precision.reduction(third_contribution.right_correction)
                    ),
                    third_contribution.max_speed,
                    third_contribution.reconstructed_left,
                    third_contribution.reconstructed_right,
                    third_contribution.dry_face,
                )
                for (
                    first_contribution,
                    second_contribution,
                    third_contribution,
                ) in zip(
                    first.contributions,
                    second.contributions,
                    third.contributions,
                    strict=True,
                )
            )
            candidate = BalancedPositivityBlendResult(
                state=self.precision.storage(third.state),
                report=self._precision_report(third.report),
                contributions=accepted_contributions,
                normal_fluxes=normal_fluxes,
                integrated_fluxes=integrated_flux_rates,
                face_blend_factors=third.face_blend_factors,
            )
        else:
            candidate = type(third)(
                state=self.precision.storage(third.state),
                report=self._precision_report(third.report),
                normal_fluxes=normal_fluxes,
                integrated_fluxes=integrated_flux_rates,
                face_blend_factors=third.face_blend_factors,
            )
        provider = self.stage_flux_provider
        if provider is None:
            return candidate, None
        if (
            first_stage_flux is None
            or second_stage_flux is None
            or third_stage_flux is None
        ):
            raise RuntimeError("Stage-flux provider did not produce all SSPRK3 stages.")
        return candidate, FiniteVolumeStageFluxTrace(
            (first_stage_flux, second_stage_flux, third_stage_flux),
            provider.provider_id,
        )

    def _candidate(
        self,
        time: Array,
        state: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> PositivityBlendResult | BalancedPositivityBlendResult:
        candidate, _ = self._candidate_with_stage_flux_trace(time, state, step_size, args)
        return candidate

    def _validate_sliding_state(self, runtime_state: FiniteVolumeRuntimeState, /) -> None:
        """Reject a stale or absent map before any stage evaluates physics."""
        if self.sliding_plan is None:
            if runtime_state.sliding_coupling is not None:
                raise ValueError("Static runtime cannot carry a sliding coupling.")
            return
        coupling = runtime_state.sliding_coupling
        if not isinstance(coupling, PeriodicSlidingCoupling):
            raise ValueError("Sliding runtime state has no certified coupling.")
        if coupling.evidence_id == "" or not bool(np.asarray(coupling.coverage_passed)):
            raise ValueError("Sliding runtime state carries failed overlap evidence.")
        if runtime_state.sliding_coupling_id != coupling.coupling_id:
            raise ValueError("Sliding runtime coupling identity is stale.")
        if coupling.shift_precision != self.sliding_plan.shift_precision:
            raise ValueError("Sliding runtime shift precision is stale.")
        prepared_coupling = self.sliding_initial_coupling
        if (
            prepared_coupling is None
            or coupling.coupling_id != prepared_coupling.coupling_id
        ):
            raise ValueError(
                "Sliding runtime state does not match this prepared runtime; "
                "advance with the accepted successor runtime."
            )

    def _refresh_sliding_after_accept(
        self,
        prior_state: FiniteVolumeRuntimeState,
        result: FiniteVolumeAdvanceResult,
        args: Any,
        /,
    ) -> FiniteVolumeAdvanceResult:
        """Run exactly one host-side sliding transaction after acceptance."""
        if self.sliding_plan is None or not bool(np.asarray(result.accepted)):
            return result
        plan = self.sliding_plan
        accepted_state = result.runtime_state
        shift = plan.evaluate_shift(accepted_state.time, args)
        try:
            coupling = plan.coupling(shift)
        except (TypeError, ValueError, FloatingPointError) as error:
            raise RuntimeError(
                "Accepted sliding refresh failed overlap coverage; "
                "the predecessor state remains the only valid runtime state."
            ) from error

        # One accepted boundary produces one request and one transaction.  The
        # scheduler owns append/commit ordering and keeps failed artifacts out of
        # the runtime state.
        event_kind = TopologyEventKind.SLIDING_REFRESH
        request = FiniteVolumeTopologyEventRequest(
            event_kind,
            prior_state.content_state.topology_epoch_id,
            plan.plan_id,
            payload_id=coupling.evidence_id,
            reason="accepted-step sliding coupling refresh",
        )
        accepted_step = int(np.asarray(accepted_state.accepted_step))
        event_time = float(np.asarray(accepted_state.time))
        discretization = self.dynamics.discretization
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise ValueError("Sliding refresh requires unstructured geometry.")
        current_epoch = prior_state.topology_journal.epoch_table[-1]
        successor = current_epoch
        native_overset_id = (
            self.dynamics.coupling.overset.native_connectivity_id
            if isinstance(self.dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            and self.dynamics.coupling.overset is not None
            else None
        )
        successor_artifacts = FiniteVolumeTopologyArtifacts(
            successor,
            discretization.prepared_id,
            topology_artifact_id=(
                coupling.coupling_id if native_overset_id is None else native_overset_id
            ),
            metrics_artifact_id=coupling.evidence_id,
            operators_artifact_id=plan.plan_id,
        )
        remapped_content = FiniteVolumeConservativeContentState(
            accepted_state.content_state.conservative_content,
            accepted_state.content_state.effective_cell_volumes,
            accepted_state.content_state.active_cell_mask,
            accepted_state.content_state.time,
            topology_epoch_id=successor.epoch_id,
            geometry_family_id=accepted_state.content_state.geometry_family_id,
            geometry_layout_id=accepted_state.content_state.geometry_layout_id,
            geometry_version=accepted_state.content_state.geometry_version,
            evidence_policy_id=accepted_state.content_state.evidence_policy_id,
            evidence_version=accepted_state.content_state.evidence_version,
            precision=accepted_state.content_state.precision,
            owned_cell_mask=accepted_state.content_state.owned_cell_mask,
        )
        scheduler = FiniteVolumeTopologyEventScheduler(prior_state.topology_journal)
        scheduler.submit(request, accepted_step, event_time, accepted=True)
        artifact = PeriodicSlidingRefreshArtifact(
            content_state=remapped_content,
            remap=coupling,
            metrics=coupling,
            evidence=coupling,
            status=jnp.asarray(0, dtype=jnp.int32),
            result_id=successor.epoch_id,
        )
        transaction = scheduler.transact(
            accepted=True,
            source_content=accepted_state.content_state,
            artifact=artifact,
            candidate_epoch=successor,
            candidate_artifacts=successor_artifacts,
            remap=coupling,
            metrics=coupling,
            evidence=coupling,
            status=TopologyEventStatus.SUCCESS,
            coverage_tolerance=plan.coverage_tolerance,
        )
        if (
            not transaction.committed
            or transaction.result_epoch is None
            or transaction.result_artifacts is None
        ):
            raise RuntimeError(
                "Accepted sliding refresh transaction failed; "
                "the predecessor state remains the only valid runtime state."
            )
        journal = transaction.journal
        if not transaction.events:
            raise RuntimeError("Sliding refresh committed without an event successor.")
        event_id = transaction.events[-1].event_id
        refreshed_state = FiniteVolumeRuntimeState(
            remapped_content,
            journal,
            accepted_state.step_size,
            accepted_step=accepted_state.accepted_step,
            last_status=accepted_state.last_status,
            controller_state=accepted_state.controller_state,
            integrator_state=accepted_state.integrator_state,
            output_cursor=accepted_state.output_cursor,
            sliding_coupling=coupling,
            sliding_shift=shift,
            sliding_event_id=event_id,
        )
        if not isinstance(
            self.dynamics,
            PreparedUnstructuredFiniteVolumeDynamics,
        ):
            raise RuntimeError("Sliding refresh requires unstructured dynamics.")
        successor_dynamics = self.dynamics.with_sliding_coupling(coupling)
        successor_runtime = PreparedFiniteVolumeRuntime(
            successor_dynamics,
            self.positivity,
            self.policy,
            topology_epoch=transaction.result_epoch,
            topology_artifacts=transaction.result_artifacts,
            stage_state_provider=(
                None
                if self.owner_local_geometry is not None
                else self.stage_state_provider
            ),
            execution_group=self.owner_local_execution_group,
            owner_local_geometry=self.owner_local_geometry,
            owner_local_axis_name=(
                "finite_volume_owner_local"
                if not self.owner_local_axis_name
                else self.owner_local_axis_name
            ),
        )
        refreshed_result = eqx.tree_at(
            lambda answer: answer.runtime_state,
            result,
            refreshed_state,
        )
        return eqx.tree_at(
            lambda answer: answer.successor_runtime,
            refreshed_result,
            successor_runtime,
            is_leaf=lambda node: node is None,
        )

    def commit_moving_overset_update(
        self,
        accepted_result: FiniteVolumeAdvanceResult,
        previous_connectivity: Any,
        candidate_connectivity: Any,
        motion_advances: Any,
        successor_dynamics: PreparedUnstructuredFiniteVolumeDynamics | None,
        /,
        *,
        args: Any = None,
        accepted_composition: Any = None,
        execution_group: Any = None,
        owner_local_source_views: Any = None,
        owner_local_successor_assembly: Any = None,
        owner_local_successor_specs: Any = None,
        owner_local_field_reconstructions: Any = None,
        owner_local_coefficient_ids: Any = None,
        owner_local_field_name: str = "state",
        owner_local_part_name: str | None = None,
        retained_state_entry_ids: tuple[str, ...] = (),
        reprepare_entries: tuple[Any, ...] = (),
        invalidate_entry_ids: tuple[str, ...] = (),
    ) -> FiniteVolumeAdvanceResult:
        """Commit one accepted native overset motion/rezone at the step boundary.

        Eulerian steps use a fixed-time conservative rezone and keep physical
        boundary coordinates fixed. An ALE result instead supplies the exact
        accepted endpoint geometry; the native motion transition must realize
        that geometry, and a successor ALE plan must continue from it. Both
        routes transfer the actual accepted post-step content and publish only
        after the finite-volume event and native lifecycle rebind commit.

        Accepted Composition entries other than the overset registration and the
        FV state take the caller's explicit dispositions
        (`retained_state_entry_ids`, `reprepare_entries`, `invalidate_entry_ids`).
        A refused event keeps the accepted state and returns its typed reason in
        `topology_event_refusal`.
        """
        from collections.abc import Mapping

        from ..discretization.finite_volume._unstructured_remap import (
            UnstructuredConservativeRemapPlan,
        )
        from ..lifecycle import (
            Composition,
            CompositionEntry,
            CompositionTransport,
        )
        from ..meshing._assembly import MeshPart
        from ..meshing._canonical import certify_cell_mesh
        from ..meshing._motion import MeshMotionAdvance
        from ..meshing._overset import (
            OversetConnectivity,
            prepare_overset_conservative_remap,
            prepare_overset_motion_rebind,
        )
        from ..meshing._result import CellMeshingResult
        from ._finite_volume_topology_events import (
            FiniteVolumeTopologyArtifactEvidence,
        )

        if not isinstance(accepted_result, FiniteVolumeAdvanceResult):
            raise TypeError("accepted_result must be FiniteVolumeAdvanceResult.")
        if not bool(np.asarray(accepted_result.accepted)):
            return accepted_result
        if accepted_composition is not None and not isinstance(
            accepted_composition, Composition
        ):
            raise TypeError("accepted_composition must be Composition or None.")
        if (
            not isinstance(retained_state_entry_ids, tuple)
            or not isinstance(reprepare_entries, tuple)
            or not isinstance(invalidate_entry_ids, tuple)
        ):
            raise TypeError("Motion Composition dispositions must be tuples.")
        if not isinstance(previous_connectivity, OversetConnectivity):
            raise TypeError("previous_connectivity must be OversetConnectivity.")
        if candidate_connectivity is not None and not isinstance(
            candidate_connectivity, OversetConnectivity
        ):
            raise TypeError("candidate_connectivity must be OversetConnectivity or None.")
        if not isinstance(motion_advances, Mapping) or (
            not motion_advances and not previous_connectivity.owner_local
        ):
            raise TypeError("motion_advances must map accepted moved-part advances.")
        if (
            not isinstance(self.dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            or self.dynamics.coupling.overset is None
        ):
            raise ValueError(
                "Native overset motion requires prepared unstructured overset dynamics."
            )
        ale_result = accepted_result.ale
        if ale_result is None:
            ale_error = (
                "moving runtime result lacks accepted ALE geometry"
                if self.dynamics.coupling.motion is not None
                else None
            )
        elif (
            self.dynamics.coupling.motion is None
            or ale_result.geometry.motion_plan_id != self.dynamics.coupling.motion.plan_id
        ):
            ale_error = "ALE evidence does not bind the predecessor runtime motion plan"
        elif not bool(np.asarray(ale_result.accepted)):
            ale_error = "ALE geometry was not accepted"
        else:
            ale_error = None
        if self.dynamics.coupling.topology_event_policy != "accepted_step":
            raise ValueError(
                "Native overset motion requires accepted-step topology events."
            )
        source_map = self.dynamics.coupling.overset
        if (
            source_map.native_connectivity_id is None
            or source_map.native_connectivity_id != previous_connectivity.connectivity_id
        ):
            raise ValueError(
                "The predecessor FV map does not bind the accepted native connectivity."
            )
        candidate_error: str | None = None
        if previous_connectivity.owner_local and owner_local_source_views is None:
            owner_local_source_views = {
                part.name: part.carrier for part in previous_connectivity.assembly.parts
            }
        try:
            if previous_connectivity.owner_local:
                if (
                    execution_group is None
                    or owner_local_source_views is None
                    or owner_local_successor_assembly is None
                    or accepted_composition is None
                    or owner_local_successor_specs is None
                ):
                    raise ValueError(
                        "Owner-local motion requires the accepted Composition, source views, successor assembly/specs, and execution group."
                    )
                if (
                    owner_local_field_reconstructions is None
                    and owner_local_coefficient_ids is None
                ):
                    from ..discretization.finite_volume._field_view import (
                        prepare_owner_local_finite_volume_field_reconstruction,
                    )
                    from ..meshing._overset_exchange import (
                        _synchronize_owner_search_failure,
                    )

                    local_reconstructions = {}
                    local_coefficient_ids = {}
                    local_error: Exception | None = None
                    try:
                        for part in owner_local_successor_assembly.parts:
                            view = part.carrier
                            nonempty_blocks = tuple(
                                block
                                for block in view.blocks
                                if np.asarray(block.cell_ids).size
                            )
                            if len(nonempty_blocks) > 1:
                                raise ValueError(
                                    "Automatic owner-local donor reconstruction requires one canonical cell block per part; pass an exact prepared field binding for multiblock views."
                                )
                            if nonempty_blocks:
                                block = nonempty_blocks[0]
                                local_reconstructions[part.name] = (
                                    prepare_owner_local_finite_volume_field_reconstruction(
                                        view,
                                        block.name,
                                        owner_local_field_name,
                                    )
                                )
                                local_coefficient_ids[part.name] = np.asarray(
                                    block.cell_ids, dtype=np.int64
                                )
                    except Exception as error:
                        local_error = error
                    if _synchronize_owner_search_failure(
                        local_error is not None,
                        execution_group.devices,
                        "finite_volume_owner_local_reconstruction",
                    ):
                        detail = (
                            str(local_error)
                            if local_error is not None
                            else "another rank rejected its owner-local donor reconstruction"
                        )
                        raise ValueError(
                            "Owner-local donor reconstruction failed collectively: "
                            + detail
                        )
                    if local_error is not None:
                        raise ValueError(
                            "Owner-local donor reconstruction failed collectively."
                        )
                    owner_local_field_reconstructions = local_reconstructions
                    owner_local_coefficient_ids = local_coefficient_ids
                derived_candidate = previous_connectivity.moved_from_advances(
                    motion_advances,
                    successor_specs=owner_local_successor_specs,
                    owner_local_source_views=owner_local_source_views,
                    successor_assembly=owner_local_successor_assembly,
                    execution_group=execution_group,
                    owner_local_field_reconstructions=owner_local_field_reconstructions,
                    owner_local_coefficient_ids=owner_local_coefficient_ids,
                    axis_name="finite_volume_motion",
                    owner_local_field_name=owner_local_field_name,
                )
                if candidate_connectivity is not None and (
                    candidate_connectivity.complete
                    and candidate_connectivity.connectivity_id
                    != derived_candidate.connectivity_id
                ):
                    candidate_error = "supplied resolved owner-local connectivity differs from the accepted motion result"
                candidate_connectivity = derived_candidate
            else:
                derived_candidate = previous_connectivity.moved_from_advances(
                    motion_advances
                )
                if candidate_connectivity is not None and (
                    candidate_connectivity.connectivity_id
                    != derived_candidate.connectivity_id
                ):
                    candidate_error = (
                        "supplied candidate differs from moved_from_advances result"
                    )
                candidate_connectivity = derived_candidate
        except (KeyError, TypeError, ValueError) as error:
            candidate_error = str(error)
            candidate_connectivity = None

        accepted_state = accepted_result.runtime_state
        current_epoch = accepted_state.topology_journal.epoch_table[-1]
        current_artifacts = accepted_state.topology_journal.artifact_table[-1]
        candidate_id = (
            None
            if candidate_connectivity is None
            else candidate_connectivity.connectivity_id
        )
        request = FiniteVolumeTopologyEventRequest(
            TopologyEventKind.OVERSET_DONOR_REBUILD,
            current_epoch.epoch_id,
            previous_connectivity.connectivity_id,
            payload_id=candidate_id,
            reason=(
                "accepted ALE overset motion and endpoint physical remap"
                if ale_result is not None
                else "accepted Eulerian overset rezone and physical field remap"
            ),
        )
        accepted_step = int(np.asarray(accepted_state.accepted_step))
        event_time = float(np.asarray(accepted_state.time))

        def with_journal(
            journal: FiniteVolumeTopologyEventJournal,
            content: FiniteVolumeConservativeContentState,
        ) -> FiniteVolumeRuntimeState:
            return FiniteVolumeRuntimeState(
                content,
                journal,
                accepted_state.step_size,
                accepted_step=accepted_state.accepted_step,
                last_status=accepted_state.last_status,
                controller_state=accepted_state.controller_state,
                integrator_state=accepted_state.integrator_state,
                output_cursor=accepted_state.output_cursor,
                sliding_coupling=accepted_state.sliding_coupling,
                sliding_shift=accepted_state.sliding_shift,
                sliding_event_id=accepted_state.sliding_event_id,
            )

        def fail(
            status: TopologyEventStatus,
            reason: str,
            /,
            *,
            coverage_error: Any = 0.0,
            conservation_defect: Any = 0.0,
        ) -> FiniteVolumeAdvanceResult:
            evidence = FiniteVolumeTopologyArtifactEvidence(
                passed=jnp.asarray(False),
                status=jnp.asarray(int(status), dtype=jnp.int32),
                coverage_error=jnp.asarray(coverage_error),
                conservation_defect=jnp.asarray(conservation_defect),
                evidence_id=canonical_fingerprint(
                    {
                        "kind": "failed-native-overset-motion",
                        "request": request.request_id,
                        "candidate": candidate_id,
                        "reason": reason,
                        "status": int(status),
                    }
                ),
            )
            scheduler = FiniteVolumeTopologyEventScheduler(
                accepted_state.topology_journal
            )
            scheduler.submit(request, accepted_step, event_time)
            transaction = scheduler.transact(
                accepted=True,
                source_content=accepted_state.content_state,
                candidate_epoch=current_epoch,
                candidate_artifacts=current_artifacts,
                remap=evidence,
                metrics=evidence,
                evidence=evidence,
                status=status,
            )
            rolled_back = with_journal(transaction.journal, transaction.content_state)
            answer = eqx.tree_at(
                lambda result: result.runtime_state,
                accepted_result,
                rolled_back,
            )
            return eqx.tree_at(
                lambda result: result.topology_event_refusal,
                answer,
                FiniteVolumeTopologyEventRefusal(
                    status, reason, request.request_id, evidence.evidence_id
                ),
                is_leaf=lambda node: node is None,
            )

        def commit_owner_local(
            candidate_connectivity_for_commit: OversetConnectivity,
            successor_dynamics_for_commit: PreparedUnstructuredFiniteVolumeDynamics,
        ) -> FiniteVolumeAdvanceResult:
            from ..discretization._cell_complex import (
                PolygonalConnectivity,
                PolyhedralConnectivity,
                TetrahedralConnectivity,
            )
            from ..discretization._view_support import (
                OwnerLocalCellGeometry,
                OwnerLocalCompositeCellGeometry,
            )
            from ..discretization.finite_volume._owner_local_remap import (
                prepare_owner_local_conservative_remap,
            )
            from ..meshing._overset_exchange import _synchronize_owner_search_failure

            def collective_error(
                error: Exception | None,
                axis_name: str,
                /,
            ) -> str | None:
                any_failed = _synchronize_owner_search_failure(
                    error is not None,
                    execution_group.devices,
                    axis_name,
                )
                if not any_failed:
                    return None
                return (
                    str(error)
                    if error is not None
                    else "another owner rejected the same accepted topology event"
                )

            if (
                execution_group is None
                or owner_local_successor_assembly is None
                or owner_local_source_views is None
                or accepted_composition is None
                or self.owner_local_geometry is None
            ):
                return fail(
                    TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                    "Owner-local accepted-state rebind lacks its accepted composition or geometry views.",
                )
            source_layout_value = self.owner_local_geometry
            composite_layout = isinstance(
                source_layout_value, OwnerLocalCompositeCellGeometry
            )
            if composite_layout and owner_local_part_name is None:
                # A composite layout moves one named part; the others keep their
                # geometry and are carried by stable cell ID.
                return fail(
                    TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                    "Composite owner-local motion must name its moved part.",
                )
            part_name = (
                str(owner_local_part_name)
                if owner_local_part_name is not None
                else source_layout_value.part_name
            )
            local_error: Exception | None = None
            try:
                source_dynamics = self.dynamics
                if not isinstance(
                    source_dynamics, PreparedUnstructuredFiniteVolumeDynamics
                ):
                    raise TypeError("Owner-local motion source dynamics changed type.")
                source_geometry = source_dynamics.discretization
                target_geometry = successor_dynamics_for_commit.discretization
                if not isinstance(
                    source_geometry, UnstructuredFiniteVolumeDiscretization
                ) or not isinstance(
                    target_geometry, UnstructuredFiniteVolumeDiscretization
                ):
                    raise TypeError(
                        "Owner-local motion requires unstructured cell geometry."
                    )
                source_geometry_view = source_geometry.owner_local_view
                target_geometry_view = target_geometry.owner_local_view
                layout_type = (
                    OwnerLocalCompositeCellGeometry
                    if composite_layout
                    else OwnerLocalCellGeometry
                )
                if not isinstance(source_geometry_view, layout_type) or not isinstance(
                    target_geometry_view, layout_type
                ):
                    raise ValueError("Prepared FV geometry lost its owner-local view.")
                source_layout = source_geometry_view
                target_layout = target_geometry_view
                runtime_part = (
                    source_layout_value.part(part_name)
                    if isinstance(source_layout_value, OwnerLocalCompositeCellGeometry)
                    and part_name in source_layout_value.part_names
                    else source_layout_value
                )
                if (
                    not part_name
                    or not isinstance(runtime_part, OwnerLocalCellGeometry)
                    or part_name != runtime_part.part_name
                    or owner_local_source_views is None
                    or owner_local_successor_assembly is None
                ):
                    raise ValueError(
                        "Owner-local runtime part differs from its accepted source view."
                    )
                source_view_value = owner_local_source_views[part_name]
                target_view_value = owner_local_successor_assembly.part(part_name).carrier
                candidate_view_value = candidate_connectivity_for_commit.assembly.part(
                    part_name
                ).carrier
                if (
                    not isinstance(source_view_value, OwnerLocalCellGeometry)
                    or not isinstance(target_view_value, OwnerLocalCellGeometry)
                    or not isinstance(candidate_view_value, OwnerLocalCellGeometry)
                ):
                    raise TypeError(
                        "Owner-local motion lost a typed source or target view."
                    )
                source_view = source_view_value
                target_view = target_view_value
                candidate_view = candidate_view_value
                local_advance = motion_advances.get(part_name)
                if isinstance(source_layout, OwnerLocalCompositeCellGeometry):
                    if (
                        not isinstance(target_layout, OwnerLocalCompositeCellGeometry)
                        or source_layout.part_names != target_layout.part_names
                        or source_layout.part(part_name).local_view_id
                        != source_view.local_view_id
                        or target_layout.part(part_name).local_view_id
                        != target_view.local_view_id
                        or any(
                            target_layout.part(name).local_view_id
                            != source_layout.part(name).local_view_id
                            for name in source_layout.part_names
                            if name != part_name
                        )
                    ):
                        raise ValueError(
                            "Successor composite layout must move only the named part."
                        )
                if (
                    source_view.local_view_id != runtime_part.local_view_id
                    or target_view.local_view_id != candidate_view.local_view_id
                    or source_layout.local_view_id != source_layout_value.local_view_id
                    or (
                        not composite_layout
                        and target_layout.local_view_id != target_view.local_view_id
                    )
                ):
                    raise ValueError(
                        "Owner-local FV geometry does not bind the accepted source/successor views."
                    )
                if (
                    source_layout.source_topology_id != current_epoch.topology_id
                    or source_layout.source_geometry_id != current_epoch.geometry_id
                ):
                    raise ValueError(
                        "Owner-local source view differs from the current global target authority."
                    )

                def layout_parts(layout: Any, /) -> tuple[OwnerLocalCellGeometry, ...]:
                    return (
                        layout.parts
                        if isinstance(layout, OwnerLocalCompositeCellGeometry)
                        else (layout,)
                    )

                def remote_ghost_ids(layout: Any, /) -> set[int]:
                    ghosts = (
                        layout.remote_ghosts
                        if isinstance(layout, OwnerLocalCompositeCellGeometry)
                        else None
                    )
                    if ghosts is None:
                        return set()
                    return set(np.asarray(ghosts.cell_ids, dtype=np.int64).tolist())

                source_ids = np.asarray(source_geometry.cell_global_ids, dtype=np.int64)
                target_ids = np.asarray(target_geometry.cell_global_ids, dtype=np.int64)
                source_view_ids = np.asarray(source_view.cell_ids, dtype=np.int64)
                target_view_ids = np.asarray(target_view.cell_ids, dtype=np.int64)
                if set(source_ids.tolist()) != set(
                    np.asarray(source_layout.cell_ids, dtype=np.int64).tolist()
                ) or set(target_ids.tolist()) != set(
                    np.asarray(target_layout.cell_ids, dtype=np.int64).tolist()
                ):
                    raise ValueError(
                        "Owner-local FV plan cells differ from their local views."
                    )
                for layout, geometry, label in (
                    (source_layout, source_geometry, "source"),
                    (target_layout, target_geometry, "target"),
                ):
                    plan_row = {
                        int(identifier): row
                        for row, identifier in enumerate(
                            np.asarray(geometry.cell_global_ids, dtype=np.int64).tolist()
                        )
                    }
                    plan_volumes = np.asarray(geometry.cell_volumes)
                    tolerance = (
                        64.0
                        * np.finfo(np.float64).eps
                        * max(1.0, float(np.max(np.abs(plan_volumes), initial=0.0)))
                    )
                    for part_view in layout_parts(layout):
                        rows = np.asarray(
                            [
                                plan_row[int(identifier)]
                                for identifier in np.asarray(
                                    part_view.cell_ids, dtype=np.int64
                                ).tolist()
                            ],
                            dtype=np.int64,
                        )
                        measures = np.asarray(part_view.cell_measures)
                        if measures.shape != rows.shape or not np.allclose(
                            measures, plan_volumes[rows], rtol=0.0, atol=tolerance
                        ):
                            raise ValueError(
                                f"Owner-local {label} FV volumes differ from the accepted local geometry."
                            )
                source_content = accepted_state.content_state
                if source_content.owned_cell_mask is None or not np.array_equal(
                    np.asarray(source_content.owned_cell_mask),
                    np.asarray(self.owner_local_owned_cell_mask),
                ):
                    raise ValueError(
                        "Accepted finite-volume state lost its owner-local cell ownership."
                    )
                source_active_by_id = {
                    int(identifier): bool(source_content.active_cell_mask[row])
                    for row, identifier in enumerate(source_ids)
                }
                target_active_by_id = {
                    int(identifier): bool(
                        successor_dynamics_for_commit.overset_active_cell_mask[row]
                    )
                    for row, identifier in enumerate(target_ids)
                }
                # Remote donor ghosts follow the moved receptors' overlap; every
                # part row keeps its identity and physical activity.
                if set(source_active_by_id).difference(
                    remote_ghost_ids(source_layout)
                ) != set(target_active_by_id).difference(
                    remote_ghost_ids(target_layout)
                ) or any(
                    source_active_by_id[identifier] != target_active_by_id[identifier]
                    for identifier in set(source_active_by_id).intersection(
                        target_active_by_id
                    )
                ):
                    raise ValueError(
                        "Owner-local motion cannot change physical active-cell ownership."
                    )
                accepted_volumes = np.asarray(source_content.effective_cell_volumes)
                source_plan_volumes = np.asarray(source_geometry.cell_volumes)
                source_active = np.asarray(
                    source_content.active_cell_mask, dtype=np.bool_
                )
                volume_tolerance = (
                    64.0
                    * np.finfo(np.float64).eps
                    * max(1.0, float(np.max(np.abs(source_plan_volumes), initial=0.0)))
                )
                if (
                    accepted_volumes.shape != source_plan_volumes.shape
                    or not np.allclose(
                        accepted_volumes[source_active],
                        source_plan_volumes[source_active],
                        rtol=0.0,
                        atol=volume_tolerance,
                    )
                    or np.any(accepted_volumes[~source_active] != 0.0)
                ):
                    raise ValueError(
                        "Accepted owner-local content volumes do not bind the source geometry."
                    )
                if ale_result is not None:
                    endpoint = ale_result.geometry.end_geometry
                    if (
                        ale_result.geometry.topology_epoch_id
                        != source_content.topology_epoch_id
                        or not np.array_equal(
                            np.asarray(endpoint.vertices),
                            np.asarray(source_geometry.vertices),
                        )
                    ):
                        raise ValueError(
                            "Accepted ALE endpoint differs from the current owner-local FV geometry."
                        )
                elif local_advance is None:
                    if target_view.local_view_id != source_view.local_view_id:
                        raise ValueError(
                            "An unchanged Eulerian part cannot claim a moved geometry view."
                        )
                else:
                    if local_advance.local_fixed_vertex_ids is None:
                        raise ValueError(
                            "Eulerian MeshMotionAdvance lacks its exact fixed-boundary request."
                        )
                    fixed_ids = np.asarray(
                        local_advance.local_fixed_vertex_ids, dtype=np.int64
                    )
                    if fixed_ids.ndim != 1 or np.unique(fixed_ids).size != fixed_ids.size:
                        raise ValueError("Owner-local fixed vertex IDs are malformed.")
                    # A native periodic part closes its periodic cell and has no
                    # physical boundary; the unwelded seam faces of its resident
                    # lifted patch are not one.
                    if (
                        source_geometry.cell_count
                        and source_view.periodic_topology_id is None
                    ):
                        patch_names = source_geometry.boundary_patch_names
                        if "__physical_boundary__" not in patch_names:
                            raise ValueError(
                                "Owner-local FV plan lacks its separate physical-boundary patch."
                            )
                        physical_patch = patch_names.index("__physical_boundary__")
                        boundary_face_ids = np.flatnonzero(
                            np.asarray(source_geometry.boundary_patch_ids)
                            == physical_patch
                        )
                        connectivity = source_geometry.connectivity
                        if isinstance(connectivity, PolygonalConnectivity):
                            face_vertices = np.asarray(connectivity.edges, dtype=np.int64)
                            physical_rows = face_vertices[boundary_face_ids]
                        elif isinstance(connectivity, TetrahedralConnectivity):
                            face_vertices = np.asarray(connectivity.faces, dtype=np.int64)
                            physical_rows = face_vertices[boundary_face_ids]
                        elif isinstance(connectivity, PolyhedralConnectivity):
                            face_offsets = np.asarray(
                                connectivity.face_vertex_offsets, dtype=np.int64
                            )
                            face_values = np.asarray(
                                connectivity.face_vertex_values, dtype=np.int64
                            )
                            physical_rows = tuple(
                                face_values[face_offsets[face] : face_offsets[face + 1]]
                                for face in boundary_face_ids
                            )
                        else:
                            raise TypeError(
                                "Owner-local physical-boundary validation requires canonical cell faces."
                            )
                        if composite_layout:
                            # Union-mesh vertex rows are disjoint per part in part
                            # order; global vertex IDs may repeat across parts, so
                            # only the moved part's rows constrain the request.
                            row_start = 0
                            row_count = 0
                            for name in source_layout.part_names:
                                part_mesh = source_layout.part(
                                    name
                                ).reconstruct_local_geometry()[0]
                                row_count = (
                                    0
                                    if part_mesh is None
                                    else int(np.asarray(part_mesh.coordinates).shape[0])
                                )
                                if name == part_name:
                                    break
                                row_start += row_count
                            physical_rows = tuple(
                                row
                                for row in physical_rows
                                if np.all(
                                    (np.asarray(row) >= row_start)
                                    & (np.asarray(row) < row_start + row_count)
                                )
                            )
                        physical_ids = np.unique(
                            np.concatenate(
                                [
                                    source_geometry.vertex_global_ids[row]
                                    for row in physical_rows
                                ]
                            )
                            if len(physical_rows)
                            else np.empty((0,), dtype=np.int64)
                        )
                        if not set(int(value) for value in physical_ids).issubset(
                            set(int(value) for value in fixed_ids)
                        ):
                            raise ValueError(
                                "Eulerian motion request does not fix every physical boundary vertex."
                            )
                    source_vertex_ids = np.asarray(source_view.vertex_ids, dtype=np.int64)
                    target_vertex_ids = np.asarray(target_view.vertex_ids, dtype=np.int64)
                    source_vertex_row = {
                        int(identifier): row
                        for row, identifier in enumerate(source_vertex_ids)
                    }
                    target_vertex_row = {
                        int(identifier): row
                        for row, identifier in enumerate(target_vertex_ids)
                    }
                    for identifier in fixed_ids:
                        source_row = source_vertex_row.get(int(identifier))
                        target_row = target_vertex_row.get(int(identifier))
                        if (
                            source_row is None
                            or target_row is None
                            or not np.array_equal(
                                np.asarray(source_view.vertex_coordinates)[source_row],
                                np.asarray(target_view.vertex_coordinates)[target_row],
                            )
                        ):
                            raise ValueError(
                                "Eulerian motion changed a fixed physical boundary vertex."
                            )
            except Exception as error:
                local_error = error

            if _synchronize_owner_search_failure(
                local_error is not None,
                execution_group.devices,
                "finite_volume_owner_local_state_admission",
            ):
                detail = (
                    str(local_error)
                    if local_error is not None
                    else "another owner rejected its accepted local FV state binding"
                )
                return fail(
                    TopologyEventStatus.FAILED_STALE_EPOCH,
                    "Owner-local state admission failed collectively: " + detail,
                )
            if local_error is not None:
                return fail(
                    TopologyEventStatus.FAILED_STALE_EPOCH,
                    "Owner-local state admission failed collectively.",
                )

            target_map_for_commit = successor_dynamics_for_commit.coupling.overset
            target_map_error = None
            if (
                target_map_for_commit is None
                or target_map_for_commit.native_connectivity_id
                != candidate_connectivity_for_commit.connectivity_id
            ):
                target_map_error = ValueError(
                    "Owner-local successor map does not bind its native candidate."
                )
            target_map_error_text = collective_error(
                target_map_error,
                "finite_volume_owner_local_target_map_admission",
            )
            if target_map_error_text is not None:
                return fail(
                    TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                    target_map_error_text,
                )
            if target_map_for_commit is None:
                return fail(
                    TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                    "Owner-local target-map admission produced no typed map.",
                )

            advance = motion_advances.get(part_name)
            transition = None if advance is None else advance.geometry_transition
            # Local advance IDs bind rank-local meshes; the remap binds the
            # group-wide motion between the accepted source and target parts.
            group_advance_id = (
                None
                if advance is None
                else canonical_fingerprint(
                    {
                        "kind": "owner-local-finite-volume-motion",
                        "source": source_view.part_id,
                        "target": target_view.part_id,
                        "driver": advance.motion_driver_id,
                    }
                )
            )
            source_average_plan = self._dynamics_cell_average(
                accepted_state.content_state
            ).reshape((-1, self.dynamics.system.component_count))
            source_average_view = source_average_plan[
                jnp.asarray(
                    [
                        np.flatnonzero(source_ids == int(identifier))[0]
                        for identifier in source_view_ids
                    ],
                    dtype=jnp.int32,
                )
            ]
            try:
                # The remap synchronizes its own validation, so a refusal is
                # raised on every owner at the same collective step.
                local_remap = prepare_owner_local_conservative_remap(
                    source_view,
                    target_view,
                    execution_group,
                    transition=transition,
                    advance_id=group_advance_id,
                    axis_name="finite_volume_owner_local_rezone",
                    require_complete=True,
                )
                remap_result = local_remap.apply_averages(
                    source_average_view,
                    source_view,
                    target_view,
                    execution_group,
                    axis_name="finite_volume_owner_local_rezone",
                )
            except (TypeError, ValueError, RuntimeError) as error:
                return fail(
                    TopologyEventStatus.FAILED_COVERAGE,
                    "Owner-local remap preparation failed collectively: " + str(error),
                )
            target_average_view = remap_result.local_values
            # Moved-part rows take the conservative remap; rows of unmoved parts
            # are carried by stable cell ID. Remote donor ghost rows are refreshed
            # from their owners once the successor layout exists.
            target_view_row = {
                int(identifier): row for row, identifier in enumerate(target_view_ids)
            }
            source_plan_row = {
                int(identifier): row for row, identifier in enumerate(source_ids)
            }
            remapped_rows = np.asarray(
                [target_view_row.get(int(identifier), -1) for identifier in target_ids],
                dtype=np.int32,
            )
            carried_rows = np.asarray(
                [
                    -1
                    if int(identifier) in target_view_row
                    else source_plan_row.get(int(identifier), -1)
                    for identifier in target_ids
                ],
                dtype=np.int32,
            )
            target_average_plan = jnp.where(
                jnp.asarray(remapped_rows >= 0)[:, None],
                target_average_view[jnp.asarray(np.maximum(remapped_rows, 0))],
                jnp.where(
                    jnp.asarray(carried_rows >= 0)[:, None],
                    source_average_plan[jnp.asarray(np.maximum(carried_rows, 0))],
                    jnp.zeros((), dtype=source_average_plan.dtype),
                ),
            )

            target_topology_id = target_layout.source_topology_id
            target_geometry_id = target_layout.source_geometry_id
            target_partition_id = target_layout.distribution_id
            same_epoch = (
                target_topology_id == current_epoch.topology_id
                and target_geometry_id == current_epoch.geometry_id
                and target_partition_id == current_epoch.partition_id
            )
            candidate_epoch = TopologyEpoch(
                current_epoch.index if same_epoch else current_epoch.index + 1,
                target_geometry_id,
                target_topology_id,
                target_partition_id,
            )
            coverage_error = max(
                float(
                    np.max(
                        np.abs(np.asarray(local_remap.evidence.source_coverage_defects)),
                        initial=0.0,
                    )
                ),
                float(
                    np.max(
                        np.abs(np.asarray(local_remap.evidence.target_coverage_defects)),
                        initial=0.0,
                    )
                ),
            )
            conservation_defect = remap_result.conservation_residual
            remap_tolerance = float(local_remap.tolerance)
            residual_host = np.asarray(conservation_defect, dtype=np.float64)
            remap_error = None
            if (
                not bool(np.asarray(local_remap.evidence.coverage_complete))
                or np.any(~np.isfinite(residual_host))
                or np.any(np.abs(residual_host) > remap_tolerance)
            ):
                remap_error = ValueError(
                    "Owner-local common refinement did not prove complete conservative coverage."
                )
            remap_error_text = collective_error(
                remap_error,
                "finite_volume_owner_local_remap_admission",
            )
            if remap_error_text is not None:
                return fail(
                    TopologyEventStatus.FAILED_COVERAGE,
                    "Owner-local remap failed collective admission: " + remap_error_text,
                    coverage_error=coverage_error,
                    conservation_defect=conservation_defect,
                )

            candidate_artifacts = FiniteVolumeTopologyArtifacts(
                candidate_epoch,
                target_geometry.prepared_id,
                topology_artifact_id=candidate_connectivity_for_commit.connectivity_id,
                metrics_artifact_id=local_remap.evidence.evidence_id,
                operators_artifact_id=target_map_for_commit.plan_id,
            )
            successor_sliding_coupling = None
            successor_sliding_shift = jnp.asarray(
                0.0, dtype=self.precision.reduction_dtype
            )
            sliding_plan = successor_dynamics_for_commit.coupling.sliding
            sliding_error = None
            try:
                if sliding_plan is not None:
                    successor_sliding_shift = sliding_plan.evaluate_shift(
                        accepted_state.time, args
                    )
                    successor_sliding_coupling = sliding_plan.coupling(
                        successor_sliding_shift
                    )
                    successor_dynamics_for_commit = (
                        successor_dynamics_for_commit.with_sliding_coupling(
                            successor_sliding_coupling
                        )
                    )
                elif self.sliding_plan is not None:
                    raise ValueError(
                        "Successor dynamics omit the predecessor sliding coupling."
                    )
            except Exception as error:
                sliding_error = error
            sliding_error_text = collective_error(
                sliding_error,
                "finite_volume_owner_local_sliding_admission",
            )
            if sliding_error_text is not None:
                return fail(
                    TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                    "Owner-local sliding rebind failed collective admission: "
                    + sliding_error_text,
                )
            successor_runtime = None
            target_content = None
            target_runtime_error = None
            try:
                successor_runtime = PreparedFiniteVolumeRuntime(
                    successor_dynamics_for_commit,
                    self.positivity,
                    self.policy,
                    topology_epoch=candidate_epoch,
                    topology_artifacts=candidate_artifacts,
                    execution_group=execution_group,
                    owner_local_geometry=target_layout,
                    owner_local_axis_name=self.owner_local_axis_name,
                )
                successor_owned_cell_mask = successor_runtime.owner_local_owned_cell_mask
                if successor_owned_cell_mask is None:
                    raise ValueError(
                        "Owner-local successor runtime lost its ownership mask."
                    )
                if ale_result is not None:
                    successor_motion = successor_dynamics_for_commit.coupling.motion
                    if successor_motion is None:
                        raise ValueError(
                            "Continuous owner-local ALE update has no successor motion plan."
                        )
                    continuation = successor_motion.prepare_ssprk33_step(
                        accepted_state.time,
                        accepted_state.step_size,
                        candidate_epoch.epoch_id,
                        accepted_state.content_state.geometry_version,
                        accepted_state.content_state.evidence_version,
                        args,
                        prior_effective_cell_volumes=successor_runtime.effective_cell_volumes,
                    )
                    if not np.array_equal(
                        np.asarray(continuation.start_geometry.vertices),
                        np.asarray(target_geometry.vertices),
                    ):
                        raise ValueError(
                            "Successor owner-local ALE plan does not start at the accepted target geometry."
                        )
                target_content = FiniteVolumeConservativeContentState.from_cell_average(
                    target_average_plan,
                    successor_runtime.effective_cell_volumes,
                    successor_runtime.active_cell_mask,
                    accepted_state.time,
                    topology_epoch_id=candidate_epoch.epoch_id,
                    geometry_family_id=successor_runtime.geometry_family_id,
                    geometry_layout_id=successor_runtime.geometry_layout_id,
                    geometry_version=jnp.asarray(0, dtype=jnp.int32),
                    evidence_policy_id=successor_runtime.evidence_policy_id,
                    evidence_version=jnp.asarray(0, dtype=jnp.int32),
                    precision=successor_runtime.precision,
                    owned_cell_mask=successor_owned_cell_mask,
                )
            except Exception as error:
                target_runtime_error = error
            target_runtime_error_text = collective_error(
                target_runtime_error,
                "finite_volume_owner_local_target_runtime_admission",
            )
            if target_runtime_error_text is not None:
                return fail(
                    TopologyEventStatus.FAILED_COVERAGE,
                    "Owner-local target runtime failed collective admission: "
                    + target_runtime_error_text,
                    coverage_error=coverage_error,
                    conservation_defect=conservation_defect,
                )
            if not isinstance(
                successor_runtime, PreparedFiniteVolumeRuntime
            ) or not isinstance(target_content, FiniteVolumeConservativeContentState):
                return fail(
                    TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                    "Owner-local target runtime admission produced no accepted state.",
                    coverage_error=coverage_error,
                    conservation_defect=conservation_defect,
                )
            refresh_error = None
            try:
                if successor_runtime.owner_local_decision_provider is not None:
                    refreshed_average = successor_runtime._provide_stage_state(
                        accepted_state.time, target_content.cell_average()
                    )
                    target_content = target_content.with_content(
                        refreshed_average
                        * target_content.effective_cell_volumes.reshape(
                            (-1,) + (1,) * (refreshed_average.ndim - 1)
                        )
                    )
            except Exception as error:
                refresh_error = error
            refresh_error_text = collective_error(
                refresh_error,
                "finite_volume_owner_local_target_ghost_refresh",
            )
            if refresh_error_text is not None:
                return fail(
                    TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                    "Owner-local target ghost refresh failed collective admission: "
                    + refresh_error_text,
                    coverage_error=coverage_error,
                    conservation_defect=conservation_defect,
                )
            event_evidence = FiniteVolumeTopologyArtifactEvidence(
                passed=jnp.asarray(True),
                status=jnp.asarray(int(TopologyEventStatus.SUCCESS), dtype=jnp.int32),
                coverage_error=jnp.asarray(coverage_error),
                conservation_defect=conservation_defect,
                evidence_id=canonical_fingerprint(
                    {
                        "kind": "accepted-owner-local-overset-rezone-evidence",
                        "connectivity": candidate_connectivity_for_commit.connectivity_id,
                        "remap": local_remap.remap_id,
                        "action": remap_result.result_id,
                        "source_state": array_tree_fingerprint(
                            accepted_state.content_state.conservative_content
                        ),
                    }
                ),
            )
            transaction = None
            transaction_error = None
            try:
                scheduler = FiniteVolumeTopologyEventScheduler(
                    accepted_state.topology_journal
                )
                scheduler.submit(request, accepted_step, event_time)
                successor_owned_cell_mask = successor_runtime.owner_local_owned_cell_mask
                if successor_owned_cell_mask is None:
                    raise ValueError(
                        "Owner-local successor runtime lost its ownership mask."
                    )
                active_owned_mask = (
                    successor_runtime.active_cell_mask & successor_owned_cell_mask
                )
                transaction = scheduler.transact(
                    accepted=True,
                    source_content=accepted_state.content_state,
                    candidate_epoch=candidate_epoch,
                    candidate_artifacts=candidate_artifacts,
                    remap=local_remap,
                    metrics=event_evidence,
                    evidence=event_evidence,
                    status=TopologyEventStatus.SUCCESS,
                    coverage_tolerance=remap_tolerance,
                    active_cell_mask=successor_runtime.active_cell_mask,
                    admissibility=lambda content: jnp.all(
                        jnp.where(
                            active_owned_mask,
                            successor_dynamics_for_commit.system.admissible(
                                content.cell_average()
                            ),
                            True,
                        )
                    ),
                    transfer=lambda _content, _remap: target_content,
                    result_id=candidate_epoch.epoch_id,
                    payload_ids=(request.payload_id,),
                )
            except Exception as error:
                transaction_error = error
            if transaction is None and transaction_error is None:
                transaction_error = ValueError(
                    "Owner-local topology transaction produced no result."
                )
            transaction_error_text = collective_error(
                transaction_error,
                "finite_volume_owner_local_event_transaction",
            )
            if transaction_error_text is not None:
                return fail(
                    TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                    "Owner-local topology transaction failed collective admission: "
                    + transaction_error_text,
                    coverage_error=coverage_error,
                    conservation_defect=conservation_defect,
                )
            if transaction is None:
                return fail(
                    TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                    "Owner-local topology transaction produced no result.",
                    coverage_error=coverage_error,
                    conservation_defect=conservation_defect,
                )
            transaction_error = (
                None
                if transaction.committed and transaction.events
                else ValueError(
                    "Owner-local native overset transaction did not commit one event."
                )
            )
            transaction_error_text = collective_error(
                transaction_error,
                "finite_volume_owner_local_event_commit",
            )
            if transaction_error_text is not None:
                return fail(
                    TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                    "Owner-local topology event failed collective admission: "
                    + transaction_error_text,
                    coverage_error=coverage_error,
                    conservation_defect=conservation_defect,
                )
            event_id = transaction.events[-1].event_id
            next_state = FiniteVolumeRuntimeState(
                transaction.content_state,
                transaction.journal,
                accepted_state.step_size,
                accepted_step=accepted_state.accepted_step,
                last_status=accepted_state.last_status,
                controller_state=accepted_state.controller_state,
                integrator_state=accepted_state.integrator_state,
                output_cursor=accepted_state.output_cursor,
                sliding_coupling=successor_sliding_coupling,
                sliding_shift=successor_sliding_shift,
                sliding_event_id=(
                    event_id
                    if successor_sliding_coupling is not None
                    else accepted_state.sliding_event_id
                ),
            )
            composition = None
            transport = None
            composition_error = None
            try:
                previous_entries = previous_connectivity.composition_entries()
                candidate_entries = (
                    candidate_connectivity_for_commit.composition_entries()
                )
                previous_route = next(
                    entry for entry in previous_entries if entry.role == "interface-route"
                )
                candidate_route = next(
                    entry
                    for entry in candidate_entries
                    if entry.role == "interface-route"
                )
                accepted_route = accepted_composition.entry(previous_route.entry_id)
                if accepted_route.revision_id != previous_route.revision_id:
                    raise ValueError(
                        "Accepted Composition does not bind the predecessor overset route."
                    )

                def state_entry(
                    state: FiniteVolumeRuntimeState,
                    route: Any,
                    layout_id: str,
                    runtime_id: str,
                    /,
                ) -> CompositionEntry:
                    content = state.content_state
                    revision_id = canonical_fingerprint(
                        {
                            "kind": "finite-volume-accepted-state",
                            "runtime": runtime_id,
                            "time": float(np.asarray(state.time)).hex(),
                            "accepted_step": int(np.asarray(state.accepted_step)),
                            "content": array_tree_fingerprint(
                                content.conservative_content
                            ),
                            "controller": array_tree_fingerprint(state.controller_state),
                            "integrator": array_tree_fingerprint(state.integrator_state),
                            "output_cursor": int(np.asarray(state.output_cursor)),
                        }
                    )
                    return CompositionEntry(
                        state,
                        entry_id="finite-volume/state",
                        role="physical-state",
                        owner_id="phydrax.solver.PreparedFiniteVolumeRuntime",
                        structure_id=layout_id,
                        revision_id=revision_id,
                        semantics_id=successor_dynamics_for_commit.system.system_id,
                        dependencies=(route.binding("revision"),),
                    )

                source_state_entry = state_entry(
                    accepted_state,
                    previous_route,
                    accepted_state.content_state.geometry_layout_id,
                    self.runtime_id,
                )
                target_state_entry = state_entry(
                    next_state,
                    candidate_route,
                    transaction.content_state.geometry_layout_id,
                    successor_runtime.runtime_id,
                )
                preserved_entries = tuple(
                    entry
                    for entry in accepted_composition.entries
                    if entry.entry_id != source_state_entry.entry_id
                )
                composition = Composition(
                    (*preserved_entries, source_state_entry),
                    boundary_id=canonical_fingerprint(
                        {
                            "kind": "finite-volume-accepted-boundary",
                            "epoch": current_epoch.epoch_id,
                            "accepted_step": accepted_step,
                        }
                    ),
                )
                source_total = remap_result.source_integral
                target_total = remap_result.target_integral
                content_tolerance = jnp.asarray(
                    remap_tolerance, dtype=source_total.dtype
                ) * jnp.maximum(jnp.abs(source_total), 1.0)
                transport = CompositionTransport(
                    "physical-remap",
                    ("finite-volume/state",),
                    (target_state_entry,),
                    source_structure_ids=(source_state_entry.structure_id,),
                    route_id=local_remap.remap_id,
                    successful=jnp.asarray(True),
                    source_content=source_total,
                    target_content=target_total,
                    content_tolerance=content_tolerance,
                )
            except Exception as error:
                composition_error = error
            composition_error_text = collective_error(
                composition_error,
                "finite_volume_owner_local_composition_preparation",
            )
            if composition_error_text is not None:
                return fail(
                    TopologyEventStatus.FAILED_COVERAGE,
                    "Owner-local accepted-state composition failed collective admission: "
                    + composition_error_text,
                    coverage_error=coverage_error,
                    conservation_defect=conservation_defect,
                )
            if not isinstance(composition, Composition) or not isinstance(
                transport, CompositionTransport
            ):
                return fail(
                    TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                    "Owner-local accepted-state transport was not prepared.",
                    coverage_error=coverage_error,
                    conservation_defect=conservation_defect,
                )
            receipt = None
            rebind_error = None
            try:
                prepared_motion = prepare_overset_motion_rebind(
                    composition,
                    previous_connectivity,
                    candidate_connectivity_for_commit,
                    transports=(transport,),
                    reprepare=reprepare_entries,
                    retain=retained_state_entry_ids,
                    invalidate=invalidate_entry_ids,
                )
                receipt = prepared_motion.commit(accepted_boundary=True)
            except Exception as error:
                rebind_error = error
            rebind_error_text = collective_error(
                rebind_error,
                "finite_volume_owner_local_composition_rebind",
            )
            if rebind_error_text is not None:
                return fail(
                    TopologyEventStatus.FAILED_COVERAGE,
                    "Owner-local accepted-state rebind failed collective admission: "
                    + rebind_error_text,
                    coverage_error=coverage_error,
                    conservation_defect=conservation_defect,
                )
            receipt_error = (
                None
                if isinstance(receipt, CompositionRebindReceipt) and receipt.published
                else ValueError("Owner-local accepted-state transport was not admitted.")
            )
            receipt_error_text = collective_error(
                receipt_error,
                "finite_volume_owner_local_receipt_admission",
            )
            if receipt_error_text is not None:
                return fail(
                    TopologyEventStatus.FAILED_COVERAGE,
                    "Owner-local accepted-state receipt failed collective admission: "
                    + receipt_error_text,
                    coverage_error=coverage_error,
                    conservation_defect=conservation_defect,
                )
            if not isinstance(receipt, CompositionRebindReceipt) or not receipt.published:
                return fail(
                    TopologyEventStatus.FAILED_COVERAGE,
                    "Owner-local accepted-state receipt was not published.",
                    coverage_error=coverage_error,
                    conservation_defect=conservation_defect,
                )
            answer = eqx.tree_at(
                lambda result: result.runtime_state,
                accepted_result,
                next_state,
            )
            answer = eqx.tree_at(
                lambda result: result.successor_runtime,
                answer,
                successor_runtime,
                is_leaf=lambda node: node is None,
            )
            return eqx.tree_at(
                lambda result: result.overset_motion_receipt,
                answer,
                receipt,
                is_leaf=lambda node: node is None,
            )

        if previous_connectivity.owner_local:
            if execution_group is None:
                return fail(
                    TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                    "Owner-local motion requires its execution group for collective admission.",
                )
            from ..meshing._overset_exchange import _synchronize_owner_search_failure

            admission_error = ale_error or candidate_error
            if candidate_connectivity is None:
                admission_error = admission_error or (
                    "native donor reanalysis produced no owner-local candidate"
                )
            else:
                if not candidate_connectivity.owner_local:
                    admission_error = admission_error or (
                        "owner-local source produced a root-geometry candidate"
                    )
                if (
                    candidate_connectivity.epoch != previous_connectivity.epoch + 1
                    or candidate_connectivity.predecessor_id
                    != previous_connectivity.connectivity_id
                ):
                    admission_error = admission_error or (
                        "candidate native connectivity does not continue its predecessor"
                    )
                if not candidate_connectivity.complete:
                    admission_error = admission_error or (
                        "candidate native donor search contains orphan receptors"
                    )
            if successor_dynamics is None or not isinstance(
                successor_dynamics, PreparedUnstructuredFiniteVolumeDynamics
            ):
                admission_error = admission_error or (
                    "successor finite-volume dynamics are missing"
                )
            else:
                target_map = successor_dynamics.coupling.overset
                if (
                    target_map is None
                    or candidate_connectivity is None
                    or target_map.native_connectivity_id
                    != candidate_connectivity.connectivity_id
                    or successor_dynamics.source_id != self.dynamics.source_id
                    or successor_dynamics.system.system_id
                    != self.dynamics.system.system_id
                ):
                    admission_error = admission_error or (
                        "successor dynamics are stale or change the accepted source/system"
                    )
                if ale_result is not None and successor_dynamics.coupling.motion is None:
                    admission_error = admission_error or (
                        "continuous ALE motion requires a successor motion plan"
                    )
                if not isinstance(
                    self.dynamics.discretization,
                    UnstructuredFiniteVolumeDiscretization,
                ) or not isinstance(
                    successor_dynamics.discretization,
                    UnstructuredFiniteVolumeDiscretization,
                ):
                    admission_error = admission_error or (
                        "owner-local physical rezone requires unstructured cell geometry"
                    )
            # A refused motion is a failed event of the current epoch (journaled),
            # not a stale request; motion acceptance is itself collective.
            motion_refused = _synchronize_owner_search_failure(
                any(
                    isinstance(advance, MeshMotionAdvance) and not advance.accepted
                    for advance in motion_advances.values()
                ),
                execution_group.devices,
                "finite_volume_owner_local_motion_acceptance",
            )
            if _synchronize_owner_search_failure(
                admission_error is not None,
                execution_group.devices,
                "finite_volume_owner_local_candidate_admission",
            ):
                return fail(
                    TopologyEventStatus.FAILED
                    if motion_refused
                    else TopologyEventStatus.FAILED_STALE_EPOCH,
                    "Owner-local motion candidate failed collective admission: "
                    + (
                        admission_error
                        if admission_error is not None
                        else "another owner rejected its candidate"
                    ),
                )
        if ale_error is not None:
            return fail(TopologyEventStatus.FAILED_STALE_EPOCH, ale_error)
        if candidate_error is not None:
            return fail(
                TopologyEventStatus.FAILED_STALE_EPOCH,
                "native motion candidate was not derived from the supplied advances: "
                + candidate_error,
            )

        if candidate_connectivity is None:
            return fail(
                TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                "native donor reanalysis produced no candidate",
            )
        if (
            candidate_connectivity.epoch != previous_connectivity.epoch + 1
            or candidate_connectivity.predecessor_id
            != previous_connectivity.connectivity_id
        ):
            return fail(
                TopologyEventStatus.FAILED_STALE_EPOCH,
                "candidate native connectivity does not continue its predecessor",
            )
        if not candidate_connectivity.complete:
            return fail(
                TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                "candidate native donor search contains orphan receptors",
                coverage_error=candidate_connectivity.orphan_count,
            )
        if successor_dynamics is None or not isinstance(
            successor_dynamics, PreparedUnstructuredFiniteVolumeDynamics
        ):
            return fail(
                TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                "successor finite-volume dynamics are missing",
            )
        target_map = successor_dynamics.coupling.overset
        if (
            target_map is None
            or target_map.native_connectivity_id != candidate_connectivity.connectivity_id
            or successor_dynamics.source_id != self.dynamics.source_id
            or successor_dynamics.system.system_id != self.dynamics.system.system_id
        ):
            return fail(
                TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                "successor dynamics are stale or change the accepted source/system",
            )
        if ale_result is not None and successor_dynamics.coupling.motion is None:
            return fail(
                TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                "continuous ALE motion requires a successor motion plan",
            )
        if previous_connectivity.owner_local != candidate_connectivity.owner_local:
            return fail(
                TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                "native motion candidate changed between local and root geometry",
            )
        if previous_connectivity.owner_local:
            return commit_owner_local(candidate_connectivity, successor_dynamics)
        if not isinstance(
            self.dynamics.discretization, UnstructuredFiniteVolumeDiscretization
        ) or not isinstance(
            successor_dynamics.discretization,
            UnstructuredFiniteVolumeDiscretization,
        ):
            return fail(
                TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                "physical overset rezone requires unstructured cell geometry",
            )
        source_geometry = self.dynamics.discretization
        target_geometry = successor_dynamics.discretization

        def part_matches_geometry(part: Any, geometry: Any, /) -> bool:
            if not isinstance(part.carrier, CellMeshingResult):
                return False
            mesh = part.carrier.mesh
            return (
                mesh.topology_id == geometry.mesh.topology_id
                and np.array_equal(
                    np.asarray(mesh.coordinates), np.asarray(geometry.vertices)
                )
                and np.array_equal(
                    np.asarray(mesh.vertex_global_ids),
                    np.asarray(geometry.vertex_global_ids),
                )
                and np.array_equal(
                    np.asarray(mesh.entity_set(mesh.topological_dimension).entity_ids),
                    np.asarray(geometry.cell_global_ids),
                )
            )

        source_part = None
        target_part = None
        for name, advance in motion_advances.items():
            if not isinstance(name, str) or not isinstance(advance, MeshMotionAdvance):
                return fail(
                    TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                    "motion advances must contain named MeshMotionAdvance values",
                )
            try:
                prior_part = previous_connectivity.assembly.part(name)
                next_part = candidate_connectivity.assembly.part(name)
            except KeyError:
                return fail(
                    TopologyEventStatus.FAILED_STALE_EPOCH,
                    "motion advance names an unregistered assembly part",
                )
            motion_transition = advance.geometry_transition
            if (
                not advance.accepted
                or advance.result is None
                or motion_transition is None
                or not isinstance(prior_part.carrier, CellMeshingResult)
                or not isinstance(next_part.carrier, CellMeshingResult)
                or advance.source_mesh_id != prior_part.carrier.mesh.mesh_id
                or advance.source_result_id != prior_part.carrier.result_id
                or motion_transition.source_topology_id
                != prior_part.carrier.mesh.topology_id
                or next_part.carrier.result_id != advance.result.result_id
            ):
                return fail(
                    TopologyEventStatus.FAILED_STALE_EPOCH,
                    "MeshMotionAdvance does not bind the candidate native geometry",
                )
            boundary_mask = np.asarray(
                prior_part.carrier.mesh.topology.entities(0).subset("boundary").mask,
                dtype=np.bool_,
            )
            prior_coordinates = np.asarray(prior_part.carrier.mesh.coordinates)
            moved_coordinates = np.asarray(motion_transition.target_mesh.coordinates)
            if ale_result is None:
                if not np.array_equal(
                    prior_coordinates[boundary_mask],
                    moved_coordinates[boundary_mask],
                ):
                    return fail(
                        TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                        "physical boundary motion requires accepted ALE geometry",
                    )
                source_matches_runtime = part_matches_geometry(
                    prior_part, source_geometry
                )
            else:
                ale_start = np.asarray(ale_result.geometry.start_geometry.vertices)
                ale_end = np.asarray(ale_result.geometry.end_geometry.vertices)
                if (
                    source_geometry.topology_id
                    != ale_result.geometry.start_geometry.topology_id
                    or ale_result.geometry.topology_epoch_id
                    != accepted_state.content_state.topology_epoch_id
                ):
                    return fail(
                        TopologyEventStatus.FAILED_STALE_EPOCH,
                        "ALE stage geometry does not bind the accepted predecessor FV topology",
                    )
                if (
                    not np.array_equal(prior_coordinates, ale_start)
                    or not np.array_equal(moved_coordinates, ale_end)
                    or motion_transition.target_mesh.topology_id
                    != prior_part.carrier.mesh.topology_id
                ):
                    return fail(
                        TopologyEventStatus.FAILED_STALE_EPOCH,
                        "MeshMotionAdvance does not realize the accepted ALE endpoint",
                    )
                source_matches_runtime = part_matches_geometry(
                    prior_part, source_geometry
                )
            if source_matches_runtime:
                if source_part is not None:
                    return fail(
                        TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                        "one runtime update cannot own multiple independent mesh parts",
                    )
                source_part = prior_part
                target_part = next_part
        if (
            source_part is None
            or target_part is None
            or not part_matches_geometry(target_part, target_geometry)
        ):
            return fail(
                TopologyEventStatus.FAILED_STALE_EPOCH,
                "source and successor FV geometry do not match the moved native part",
            )

        remap_source_geometry = source_geometry
        remap_source_part = source_part
        try:
            if ale_result is not None:
                if len(motion_advances) != 1:
                    return fail(
                        TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                        "one ALE runtime update must bind exactly one moved FV part",
                    )
                motion_plan = self.dynamics.coupling.motion
                if motion_plan is None:
                    return fail(
                        TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                        "accepted ALE evidence has no predecessor motion plan",
                    )
                end_vertices = np.asarray(ale_result.geometry.end_geometry.vertices)
                source_end_mesh = source_part.carrier.mesh.with_coordinates(
                    end_vertices,
                    numeric_version=(
                        "accepted-ale:" + ale_result.geometry.motion_plan_id
                    ),
                )
                source_end_result = certify_cell_mesh(
                    source_end_mesh,
                    source_part.carrier.coordinate_contract,
                )
                remap_source_part = MeshPart(source_part.name, source_end_result)
                base_plan = motion_plan.base_plan
                endpoint_plan = UnstructuredFiniteVolumePlan(
                    end_vertices,
                    triangles=base_plan.triangles,
                    quadrilaterals=base_plan.quadrilaterals,
                    tetrahedra=base_plan.tetrahedra,
                    vertex_global_ids=base_plan.vertex_global_ids,
                    cell_global_ids=base_plan.cell_global_ids,
                    boundary_patches={
                        name: faces
                        for name, faces in zip(
                            base_plan.patch_names, base_plan.patch_faces, strict=True
                        )
                    },
                    field_name=base_plan.field_name,
                    component_names=base_plan.component_names,
                )
                remap_source_geometry = endpoint_plan.prepare(
                    numeric_version=ale_result.geometry.motion_plan_id
                )
                if (
                    remap_source_geometry.topology_id != source_geometry.topology_id
                    or not np.array_equal(
                        np.asarray(remap_source_geometry.cell_global_ids),
                        np.asarray(source_geometry.cell_global_ids),
                    )
                ):
                    return fail(
                        TopologyEventStatus.FAILED_STALE_EPOCH,
                        "realized ALE endpoint changed the predecessor FV topology",
                    )
                accepted_volumes = np.asarray(
                    accepted_state.content_state.effective_cell_volumes
                )
                endpoint_volumes = np.asarray(remap_source_geometry.cell_volumes)
                volume_tolerance = (
                    64.0
                    * np.finfo(accepted_volumes.dtype).eps
                    * max(1.0, float(np.max(np.abs(endpoint_volumes))))
                )
                if accepted_volumes.shape != endpoint_volumes.shape or not np.allclose(
                    accepted_volumes,
                    endpoint_volumes,
                    rtol=0.0,
                    atol=volume_tolerance,
                ):
                    return fail(
                        TopologyEventStatus.FAILED_STALE_EPOCH,
                        "accepted ALE content volumes do not bind the realized endpoint",
                    )
            native_remap = prepare_overset_conservative_remap(
                remap_source_part,
                target_part,
            )
            if not native_remap.complete:
                return fail(
                    TopologyEventStatus.FAILED_COVERAGE,
                    "native physical overlap is incomplete",
                )
            refinement = native_remap.refinement
            remap_tolerance = float(refinement.policy.coverage_tolerance)
            remap = UnstructuredConservativeRemapPlan(
                remap_source_geometry,
                target_geometry,
                refinement.target_offsets,
                refinement.source_cells,
                refinement.volumes,
                method="native-common-refinement",
                provenance=native_remap.transfer_id,
                tolerance=remap_tolerance,
            )
        except (TypeError, ValueError, FloatingPointError):
            return fail(
                TopologyEventStatus.FAILED_COVERAGE,
                "native physical overlap preparation failed",
            )
        source_active = np.asarray(accepted_state.content_state.active_cell_mask)
        target_active = np.asarray(successor_dynamics.overset_active_cell_mask)
        same_physical_ownership = (np.all(source_active) and np.all(target_active)) or (
            source_active.shape == target_active.shape
            and np.array_equal(source_active, target_active)
        )
        if not same_physical_ownership:
            return fail(
                TopologyEventStatus.FAILED_COVERAGE,
                "serial physical rezone cannot change active cell ownership",
            )

        next_index = (
            current_epoch.index
            if (
                target_geometry.topology_id == current_epoch.topology_id
                and target_geometry.geometry_id == current_epoch.geometry_id
            )
            else current_epoch.index + 1
        )
        candidate_epoch = TopologyEpoch(
            next_index,
            target_geometry.geometry_id,
            target_geometry.topology_id,
            current_epoch.partition_id,
        )
        candidate_artifacts = FiniteVolumeTopologyArtifacts(
            candidate_epoch,
            target_geometry.prepared_id,
            topology_artifact_id=candidate_connectivity.connectivity_id,
            metrics_artifact_id=native_remap.evidence.evidence_id,
            operators_artifact_id=target_map.plan_id,
        )
        successor_sliding_coupling = None
        successor_sliding_shift = jnp.asarray(0.0, dtype=self.precision.reduction_dtype)
        sliding_plan = successor_dynamics.coupling.sliding
        if sliding_plan is not None:
            try:
                successor_sliding_shift = sliding_plan.evaluate_shift(
                    accepted_state.time, args
                )
                successor_sliding_coupling = sliding_plan.coupling(
                    successor_sliding_shift
                )
                successor_dynamics = successor_dynamics.with_sliding_coupling(
                    successor_sliding_coupling
                )
            except (TypeError, ValueError, FloatingPointError):
                return fail(
                    TopologyEventStatus.FAILED_COVERAGE,
                    "successor ALE/overset sliding map could not be prepared",
                )
        elif self.sliding_plan is not None:
            return fail(
                TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                "successor dynamics omit the predecessor sliding coupling",
            )
        try:
            successor_runtime = PreparedFiniteVolumeRuntime(
                successor_dynamics,
                self.positivity,
                self.policy,
                topology_epoch=candidate_epoch,
                topology_artifacts=candidate_artifacts,
                stage_state_provider=self.stage_state_provider,
                stage_flux_provider=self.stage_flux_provider,
            )
            if ale_result is not None:
                successor_motion = successor_dynamics.coupling.motion
                if successor_motion is None:
                    return fail(
                        TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                        "continuous ALE update has no successor motion plan",
                    )
                continuation = successor_motion.prepare_ssprk33_step(
                    accepted_state.time,
                    accepted_state.step_size,
                    candidate_epoch.epoch_id,
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32),
                    args,
                    prior_effective_cell_volumes=successor_runtime.effective_cell_volumes,
                )
                if not np.array_equal(
                    np.asarray(continuation.start_geometry.vertices),
                    np.asarray(target_geometry.vertices),
                ):
                    return fail(
                        TopologyEventStatus.FAILED_STALE_EPOCH,
                        "successor ALE plan does not continue from the accepted physical geometry",
                    )
            source_average = accepted_state.content_state.cell_average()
            transferred_average = native_remap.apply_intensive(source_average)
            target_content = FiniteVolumeConservativeContentState.from_cell_average(
                transferred_average,
                successor_runtime.effective_cell_volumes,
                successor_runtime.active_cell_mask,
                accepted_state.time,
                topology_epoch_id=candidate_epoch.epoch_id,
                geometry_family_id=successor_runtime.geometry_family_id,
                geometry_layout_id=successor_runtime.geometry_layout_id,
                geometry_version=jnp.asarray(0, dtype=jnp.int32),
                evidence_policy_id=successor_runtime.evidence_policy_id,
                evidence_version=jnp.asarray(0, dtype=jnp.int32),
                precision=successor_runtime.precision,
            )
        except (TypeError, ValueError, FloatingPointError):
            return fail(
                TopologyEventStatus.FAILED_COVERAGE,
                "native physical state transfer failed",
            )

        conservation_defect = remap.conservation_defect(
            source_average, transferred_average
        )
        coverage_error = jnp.maximum(
            remap.report.maximum_source_coverage_defect,
            remap.report.maximum_target_coverage_defect,
        )
        event_evidence = FiniteVolumeTopologyArtifactEvidence(
            passed=jnp.asarray(True),
            status=jnp.asarray(int(TopologyEventStatus.SUCCESS), dtype=jnp.int32),
            coverage_error=coverage_error,
            conservation_defect=conservation_defect,
            evidence_id=canonical_fingerprint(
                {
                    "kind": "accepted-native-overset-rezone-evidence",
                    "connectivity": candidate_connectivity.connectivity_id,
                    "transfer": native_remap.transfer_id,
                    "remap": remap.plan_id,
                    "source_state": array_tree_fingerprint(
                        accepted_state.content_state.conservative_content
                    ),
                }
            ),
        )
        scheduler = FiniteVolumeTopologyEventScheduler(accepted_state.topology_journal)
        scheduler.submit(request, accepted_step, event_time)
        transaction = scheduler.transact(
            accepted=True,
            source_content=accepted_state.content_state,
            source_geometry=source_geometry,
            target_geometry=target_geometry,
            candidate_epoch=candidate_epoch,
            candidate_artifacts=candidate_artifacts,
            remap=remap,
            metrics=event_evidence,
            evidence=event_evidence,
            status=TopologyEventStatus.SUCCESS,
            coverage_tolerance=remap_tolerance,
            active_cell_mask=successor_runtime.active_cell_mask,
            admissibility=lambda content: jnp.all(
                successor_dynamics.system.admissible(content.cell_average())
            ),
            transfer=lambda _content, _remap: target_content,
            result_id=candidate_epoch.epoch_id,
            payload_ids=(request.payload_id,),
        )
        if not transaction.committed:
            failure_status = (
                transaction.failure
                if transaction.failure is not None
                else TopologyEventStatus.FAILED_COVERAGE
            )
            return fail(
                failure_status,
                "Native overset topology transaction was rejected: "
                + failure_status.name,
                coverage_error=coverage_error,
                conservation_defect=conservation_defect,
            )

        if not transaction.events:
            return fail(
                TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                "accepted native overset transaction committed without an event",
            )
        event_id = transaction.events[-1].event_id
        next_state = FiniteVolumeRuntimeState(
            transaction.content_state,
            transaction.journal,
            accepted_state.step_size,
            accepted_step=accepted_state.accepted_step,
            last_status=accepted_state.last_status,
            controller_state=accepted_state.controller_state,
            integrator_state=accepted_state.integrator_state,
            output_cursor=accepted_state.output_cursor,
            sliding_coupling=successor_sliding_coupling,
            sliding_shift=successor_sliding_shift,
            sliding_event_id=(
                event_id
                if successor_sliding_coupling is not None
                else accepted_state.sliding_event_id
            ),
        )
        try:
            previous_entries = previous_connectivity.composition_entries()
            candidate_entries = candidate_connectivity.composition_entries()
            previous_route = next(
                entry for entry in previous_entries if entry.role == "interface-route"
            )
            candidate_route = next(
                entry for entry in candidate_entries if entry.role == "interface-route"
            )

            def state_entry(
                state: FiniteVolumeRuntimeState,
                route: CompositionEntry,
                geometry_layout_id: str,
                runtime_id: str,
                /,
            ) -> CompositionEntry:
                content = state.content_state
                revision_id = canonical_fingerprint(
                    {
                        "kind": "finite-volume-accepted-state",
                        "runtime": runtime_id,
                        "time": float(np.asarray(state.time)).hex(),
                        "accepted_step": int(np.asarray(state.accepted_step)),
                        "content": array_tree_fingerprint(content.conservative_content),
                        "controller": array_tree_fingerprint(state.controller_state),
                        "integrator": array_tree_fingerprint(state.integrator_state),
                        "output_cursor": int(np.asarray(state.output_cursor)),
                    }
                )
                return CompositionEntry(
                    state,
                    entry_id="finite-volume/state",
                    role="physical-state",
                    owner_id="phydrax.solver.PreparedFiniteVolumeRuntime",
                    structure_id=geometry_layout_id,
                    revision_id=revision_id,
                    semantics_id=self.dynamics.system.system_id,
                    dependencies=(route.binding("revision"),),
                )

            source_state_entry = state_entry(
                accepted_state,
                previous_route,
                accepted_state.content_state.geometry_layout_id,
                self.runtime_id,
            )
            target_state_entry = state_entry(
                next_state,
                candidate_route,
                transaction.content_state.geometry_layout_id,
                successor_runtime.runtime_id,
            )
            if accepted_composition is None:
                composition_entries = previous_entries
            else:
                accepted_route = accepted_composition.entry(previous_route.entry_id)
                if accepted_route.revision_id != previous_route.revision_id:
                    raise ValueError(
                        "accepted Composition does not bind the predecessor native connectivity."
                    )
                composition_entries = tuple(
                    entry
                    for entry in accepted_composition.entries
                    if entry.entry_id != source_state_entry.entry_id
                )
            composition = Composition(
                (*composition_entries, source_state_entry),
                boundary_id=canonical_fingerprint(
                    {
                        "kind": "finite-volume-accepted-boundary",
                        "epoch": current_epoch.epoch_id,
                        "accepted_step": accepted_step,
                    }
                ),
            )
            source_total = accepted_state.content_state.volume_integral()
            target_total = transaction.content_state.volume_integral()
            content_tolerance = jnp.asarray(
                remap_tolerance, dtype=source_total.dtype
            ) * jnp.maximum(jnp.abs(source_total), 1.0)
            transport = CompositionTransport(
                "physical-remap",
                ("finite-volume/state",),
                (target_state_entry,),
                source_structure_ids=(source_state_entry.structure_id,),
                route_id=remap.plan_id,
                successful=jnp.asarray(True),
                source_content=source_total,
                target_content=target_total,
                content_tolerance=content_tolerance,
            )
            prepared_motion = prepare_overset_motion_rebind(
                composition,
                previous_connectivity,
                candidate_connectivity,
                transports=(transport,),
                reprepare=reprepare_entries,
                retain=retained_state_entry_ids,
                invalidate=invalidate_entry_ids,
            )
            receipt = prepared_motion.commit(accepted_boundary=True)
        except (KeyError, TypeError, ValueError, FloatingPointError):
            return fail(
                TopologyEventStatus.FAILED_COVERAGE,
                "accepted physical-state ownership rebind failed",
                coverage_error=coverage_error,
                conservation_defect=conservation_defect,
            )
        if not isinstance(receipt, CompositionRebindReceipt):
            return fail(
                TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                "accepted motion rebind returned no typed composition receipt",
                coverage_error=coverage_error,
                conservation_defect=conservation_defect,
            )
        if not receipt.published:
            return fail(
                TopologyEventStatus.FAILED_COVERAGE,
                "accepted physical-state transport was not admitted",
                coverage_error=coverage_error,
                conservation_defect=conservation_defect,
            )
        answer = eqx.tree_at(
            lambda result: result.runtime_state,
            accepted_result,
            next_state,
        )
        answer = eqx.tree_at(
            lambda result: result.successor_runtime,
            answer,
            successor_runtime,
            is_leaf=lambda node: node is None,
        )
        return eqx.tree_at(
            lambda result: result.overset_motion_receipt,
            answer,
            receipt,
            is_leaf=lambda node: node is None,
        )

    def commit_remesh(
        self,
        accepted_result: FiniteVolumeAdvanceResult,
        transition: Any,
        successor_dynamics: PreparedUnstructuredFiniteVolumeDynamics,
        /,
        *,
        content_semantics: FiniteVolumeContentTransferSemantics = (
            "eulerian_physical_overlap"
        ),
    ) -> FiniteVolumeAdvanceResult:
        """Commit one accepted remesh/adaptation transition at the step boundary.

        ``content_semantics`` selects how the accepted conservative content
        crosses the transition: the exact physical overlap of the endpoint
        geometries for Eulerian content, or the transition's accepted volume
        material chain for material-following content. The topology transaction
        prepares that route, checks coverage and conservation, and publishes the
        successor runtime only after the event commits; a rejected event keeps
        the accepted predecessor state runnable.
        """
        from ..geometry._supermesh import CommonRefinementPolicy
        from ..meshing import CellMeshTransition
        from ._finite_volume_topology_events import (
            FiniteVolumeRemeshArtifact,
            FiniteVolumeTopologyArtifactEvidence,
        )

        if not isinstance(accepted_result, FiniteVolumeAdvanceResult):
            raise TypeError("accepted_result must be FiniteVolumeAdvanceResult.")
        if not isinstance(transition, CellMeshTransition):
            raise TypeError("transition must be a CellMeshTransition.")
        if not isinstance(successor_dynamics, PreparedUnstructuredFiniteVolumeDynamics):
            raise TypeError(
                "successor_dynamics must be PreparedUnstructuredFiniteVolumeDynamics."
            )
        if not bool(np.asarray(accepted_result.accepted)):
            return accepted_result
        if self.owner_local_geometry is not None:
            raise ValueError(
                "Owner-local FV remeshing commits through the owner-local adaptation "
                "and repartition transactions, not a single-rank remesh."
            )
        if not isinstance(self.dynamics, PreparedUnstructuredFiniteVolumeDynamics):
            raise TypeError("FV remeshing requires prepared unstructured dynamics.")
        if (
            self.dynamics.coupling.topology_event_policy != "accepted_step"
            or successor_dynamics.coupling.topology_event_policy != "accepted_step"
        ):
            raise ValueError("FV remeshing requires accepted-step topology events.")
        if (
            self.sliding_plan is not None
            or successor_dynamics.coupling.sliding is not None
            or successor_dynamics.coupling.overset is not None
            or successor_dynamics.coupling.motion is not None
        ):
            raise ValueError(
                "Remesh commits rebind no sliding, overset, or motion coupling; "
                "use the coupling's own topology transaction."
            )
        if (
            successor_dynamics.system.system_id != self.dynamics.system.system_id
            or successor_dynamics.precision.policy_id != self.precision.policy_id
        ):
            raise ValueError("Successor dynamics change the FV system or precision.")
        accepted_state = accepted_result.runtime_state
        current_epoch = accepted_state.topology_journal.epoch_table[-1]
        current_artifacts = accepted_state.topology_journal.artifact_table[-1]
        source_geometry = self.dynamics.discretization
        target_geometry = successor_dynamics.discretization
        request = FiniteVolumeTopologyEventRequest(
            TopologyEventKind.REMESH,
            current_epoch.epoch_id,
            transition.transition_id,
            payload_id=transition.transition_id,
            reason=f"accepted {content_semantics} remesh",
        )
        accepted_step = int(np.asarray(accepted_state.accepted_step))
        event_time = float(np.asarray(accepted_state.time))

        def with_state(
            content: FiniteVolumeConservativeContentState,
            journal: FiniteVolumeTopologyEventJournal,
            /,
        ) -> FiniteVolumeRuntimeState:
            return FiniteVolumeRuntimeState(
                content,
                journal,
                accepted_state.step_size,
                accepted_step=accepted_state.accepted_step,
                last_status=accepted_state.last_status,
                controller_state=accepted_state.controller_state,
                integrator_state=accepted_state.integrator_state,
                output_cursor=accepted_state.output_cursor,
                sliding_coupling=accepted_state.sliding_coupling,
                sliding_shift=accepted_state.sliding_shift,
                sliding_event_id=accepted_state.sliding_event_id,
            )

        def fail(
            status: TopologyEventStatus, reason: str, /
        ) -> FiniteVolumeAdvanceResult:
            evidence = FiniteVolumeTopologyArtifactEvidence(
                passed=jnp.asarray(False),
                status=jnp.asarray(int(status), dtype=jnp.int32),
                coverage_error=jnp.asarray(0.0),
                conservation_defect=jnp.asarray(0.0),
                evidence_id=canonical_fingerprint(
                    {
                        "kind": "failed-finite-volume-remesh",
                        "request": request.request_id,
                        "reason": reason,
                        "status": int(status),
                    }
                ),
            )
            scheduler = FiniteVolumeTopologyEventScheduler(
                accepted_state.topology_journal
            )
            scheduler.submit(request, accepted_step, event_time)
            rejected = scheduler.transact(
                accepted=True,
                source_content=accepted_state.content_state,
                candidate_epoch=current_epoch,
                candidate_artifacts=current_artifacts,
                remap=evidence,
                metrics=evidence,
                evidence=evidence,
                status=status,
            )
            answer = eqx.tree_at(
                lambda result: result.runtime_state,
                accepted_result,
                with_state(rejected.content_state, rejected.journal),
            )
            return eqx.tree_at(
                lambda result: result.topology_event_refusal,
                answer,
                FiniteVolumeTopologyEventRefusal(
                    status, reason, request.request_id, evidence.evidence_id
                ),
                is_leaf=lambda node: node is None,
            )

        candidate_epoch = TopologyEpoch(
            current_epoch.index + 1,
            target_geometry.geometry_id,
            target_geometry.topology_id,
            current_epoch.partition_id,
        )
        candidate_artifacts = FiniteVolumeTopologyArtifacts(
            candidate_epoch,
            target_geometry.prepared_id,
            topology_artifact_id=transition.transition_id,
            operators_artifact_id=successor_dynamics.dynamics_id,
        )
        try:
            successor_runtime = PreparedFiniteVolumeRuntime(
                successor_dynamics,
                self.positivity,
                self.policy,
                topology_epoch=candidate_epoch,
                topology_artifacts=candidate_artifacts,
                stage_state_provider=self.stage_state_provider,
                stage_flux_provider=self.stage_flux_provider,
            )
        except (TypeError, ValueError) as error:
            return fail(
                TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                f"successor runtime preparation failed: {error}",
            )
        artifact = FiniteVolumeRemeshArtifact(
            transition,
            candidate_epoch,
            candidate_artifacts,
            None,
            None,
            None,
            TopologyEventStatus.SUCCESS,
            candidate_epoch.epoch_id,
            payload_ids=(request.payload_id,),
            target_geometry=target_geometry,
            content_semantics=content_semantics,
        )

        def transfer(
            source: FiniteVolumeConservativeContentState, remap: Any, /
        ) -> FiniteVolumeConservativeContentState:
            return FiniteVolumeConservativeContentState.from_cell_average(
                remap.apply(source.cell_average()),
                successor_runtime.effective_cell_volumes,
                successor_runtime.active_cell_mask,
                accepted_state.time,
                topology_epoch_id=candidate_epoch.epoch_id,
                geometry_family_id=successor_runtime.geometry_family_id,
                geometry_layout_id=successor_runtime.geometry_layout_id,
                geometry_version=jnp.asarray(0, dtype=jnp.int32),
                evidence_policy_id=successor_runtime.evidence_policy_id,
                evidence_version=jnp.asarray(0, dtype=jnp.int32),
                precision=successor_runtime.precision,
            )

        if content_semantics == "material_following_content":
            # Material transfer rounds once per moved product and compensated
            # accumulation; bound the admitted content defect accordingly.
            source_total = np.asarray(
                accepted_state.content_state.conservative_content, dtype=np.float64
            )
            cell_rows = source_geometry.cell_count + target_geometry.cell_count
            map_count = (
                1
                if transition.volume_material_chain is None
                else len(transition.volume_material_chain.maps)
            )
            transfer_tolerance = float(
                8.0
                * np.finfo(np.float64).eps
                * cell_rows
                * map_count
                * max(float(np.max(np.sum(np.abs(source_total), axis=0))), 1.0)
            )
            remap_policy = None
        else:
            # The Eulerian overlap is certified to its common-refinement policy.
            remap_policy = CommonRefinementPolicy()
            transfer_tolerance = float(remap_policy.coverage_tolerance)
        scheduler = FiniteVolumeTopologyEventScheduler(accepted_state.topology_journal)
        scheduler.submit(request, accepted_step, event_time)
        transaction = scheduler.transact(
            accepted=True,
            source_content=accepted_state.content_state,
            artifact=artifact,
            source_geometry=source_geometry,
            target_geometry=target_geometry,
            coverage_tolerance=transfer_tolerance,
            remap_policy=remap_policy,
            active_cell_mask=successor_runtime.active_cell_mask,
            admissibility=lambda content: jnp.all(
                successor_dynamics.system.admissible(content.cell_average())
            ),
            transfer=transfer,
            result_id=candidate_epoch.epoch_id,
            payload_ids=(request.payload_id,),
        )
        if not transaction.committed:
            failure_status = (
                transaction.failure
                if transaction.failure is not None
                else TopologyEventStatus.FAILED_COVERAGE
            )
            return fail(
                failure_status,
                "FV remesh transaction was rejected: " + failure_status.name,
            )
        answer = eqx.tree_at(
            lambda result: result.runtime_state,
            accepted_result,
            with_state(transaction.content_state, transaction.journal),
        )
        return eqx.tree_at(
            lambda result: result.successor_runtime,
            answer,
            successor_runtime,
            is_leaf=lambda node: node is None,
        )

    @checked
    def commit_owner_local_repartition(
        self,
        accepted_source: FiniteVolumeValidatedRuntimeState,
        previous_connectivity: Any,
        candidate_connectivity: Any,
        target_dynamics: PreparedUnstructuredFiniteVolumeDynamics,
        /,
        *,
        accepted_composition: Any = None,
        execution_group: Any = None,
        cell_state_banks: tuple[FiniteVolumeOwnerLocalCellStateBankSpec, ...] = (),
        retained_state_entry_ids: tuple[str, ...] = (),
        reprepare_entries: tuple[Any, ...] = (),
        invalidate_entry_ids: tuple[str, ...] = (),
        owner_local_part_name: str | None = None,
    ) -> FiniteVolumeOwnerLocalRepartitionResult:
        """Commit exact owner-local cell state to a graph-migrated FV partition."""
        from .._execution_runtime import ExecutionGroup
        from ..discretization._view_support import (
            OwnerLocalCellGeometry,
            OwnerLocalCompositeCellGeometry,
        )
        from ..lifecycle import Composition, CompositionEntry, CompositionTransport
        from ..meshing._distribution import (
            _owner_group_allgather,
            OwnerLocalGraphMigrationEvidence,
        )
        from ..meshing._overset import (
            OversetConnectivity,
            prepare_overset_motion_rebind,
        )
        from ..meshing._overset_exchange import _synchronize_owner_search_failure
        from ._finite_volume_topology_events import FiniteVolumeTopologyArtifactEvidence

        if not isinstance(previous_connectivity, OversetConnectivity) or not isinstance(
            candidate_connectivity, OversetConnectivity
        ):
            raise TypeError(
                "Repartition requires predecessor and candidate overset connectivity."
            )
        if accepted_composition is not None and not isinstance(
            accepted_composition, Composition
        ):
            raise TypeError("accepted_composition must be Composition or None.")
        if not isinstance(execution_group, ExecutionGroup):
            raise TypeError("Owner-local repartition requires an ExecutionGroup.")
        if (
            not isinstance(self.owner_local_execution_group, ExecutionGroup)
            or execution_group.spec.group_id
            != self.owner_local_execution_group.spec.group_id
        ):
            raise ValueError(
                "Repartition execution group differs from its source runtime."
            )
        if not isinstance(accepted_source, FiniteVolumeValidatedRuntimeState):
            raise TypeError("accepted_source must be FiniteVolumeValidatedRuntimeState.")
        if not isinstance(cell_state_banks, tuple) or any(
            not isinstance(spec, FiniteVolumeOwnerLocalCellStateBankSpec)
            for spec in cell_state_banks
        ):
            raise TypeError(
                "cell_state_banks must contain explicit FV cell-bank schemas."
            )
        if (
            not isinstance(retained_state_entry_ids, tuple)
            or not isinstance(reprepare_entries, tuple)
            or not isinstance(invalidate_entry_ids, tuple)
        ):
            raise TypeError("Repartition Composition dispositions must be tuples.")
        validated_source = self.validate_restored_accepted_state(
            accepted_source.runtime_state,
            accepted_source.source_cell_ids,
            expected_runtime_id=accepted_source.runtime_id,
            expected_epoch_id=accepted_source.epoch_id,
            expected_state_id=accepted_source.state_id,
        )
        accepted_state = validated_source.runtime_state
        source_content = accepted_state.content_state
        current_epoch = accepted_state.topology_journal.epoch_table[-1]
        current_artifacts = accepted_state.topology_journal.artifact_table[-1]
        if not isinstance(current_epoch, TopologyEpoch) or not isinstance(
            current_artifacts, FiniteVolumeTopologyArtifacts
        ):
            raise TypeError("Accepted runtime state has no typed topology authority.")
        request = FiniteVolumeTopologyEventRequest(
            TopologyEventKind.REPARTITION,
            current_epoch.epoch_id,
            candidate_connectivity.connectivity_id,
            payload_id=candidate_connectivity.connectivity_id,
            reason="accepted owner-local graph repartition",
        )
        accepted_step = int(np.asarray(accepted_state.accepted_step))
        event_time = float(np.asarray(accepted_state.time))

        def with_journal(
            journal: FiniteVolumeTopologyEventJournal,
            content: FiniteVolumeConservativeContentState,
        ) -> FiniteVolumeRuntimeState:
            return FiniteVolumeRuntimeState(
                content,
                journal,
                accepted_state.step_size,
                accepted_step=accepted_state.accepted_step,
                last_status=accepted_state.last_status,
                controller_state=accepted_state.controller_state,
                integrator_state=accepted_state.integrator_state,
                output_cursor=accepted_state.output_cursor,
                sliding_coupling=accepted_state.sliding_coupling,
                sliding_shift=accepted_state.sliding_shift,
                sliding_event_id=accepted_state.sliding_event_id,
            )

        def rollback_result(
            journal: FiniteVolumeTopologyEventJournal,
            content: FiniteVolumeConservativeContentState,
            event_status: TopologyEventStatus,
            reason: str,
            /,
        ) -> FiniteVolumeOwnerLocalRepartitionResult:
            return FiniteVolumeOwnerLocalRepartitionResult(
                with_journal(journal, content),
                jnp.asarray(False),
                jnp.asarray(int(event_status), dtype=jnp.int32),
                refusal=reason,
            )

        def fail(
            status: TopologyEventStatus,
            reason: str,
            /,
        ) -> FiniteVolumeOwnerLocalRepartitionResult:
            evidence = FiniteVolumeTopologyArtifactEvidence(
                passed=jnp.asarray(False),
                status=jnp.asarray(int(status), dtype=jnp.int32),
                coverage_error=jnp.asarray(0.0),
                conservation_defect=jnp.asarray(0.0),
                evidence_id=canonical_fingerprint(
                    {
                        "kind": "failed-owner-local-fv-repartition",
                        "request": request.request_id,
                        "reason": reason,
                        "status": int(status),
                    }
                ),
            )
            scheduler = FiniteVolumeTopologyEventScheduler(
                accepted_state.topology_journal
            )
            scheduler.submit(request, accepted_step, event_time)
            transaction = scheduler.transact(
                accepted=True,
                source_content=source_content,
                candidate_epoch=current_epoch,
                candidate_artifacts=current_artifacts,
                remap=evidence,
                metrics=evidence,
                evidence=evidence,
                status=status,
            )
            return rollback_result(
                transaction.journal, transaction.content_state, status, reason
            )

        def collective_error(
            error: Exception | None,
            axis_name: str,
            /,
        ) -> str | None:
            failed = _synchronize_owner_search_failure(
                error is not None,
                execution_group.devices,
                axis_name,
            )
            if not failed:
                return None
            return (
                f"{type(error).__name__}: {error}"
                if error is not None
                else "another owner rejected the same repartition transaction"
            )

        part_name = (
            self.owner_local_geometry.part_name
            if owner_local_part_name is None and self.owner_local_geometry is not None
            else str(owner_local_part_name)
        )
        local_rank = -1
        members = execution_group.spec.process_indices
        preflight_error: Exception | None = None
        source_view = target_view = graph_migration = None
        candidate_epoch = current_epoch
        candidate_artifacts = current_artifacts
        route_identity_id = ""
        source_sliding_plan_id: str | None = None
        source_sliding_coupling_id = accepted_state.sliding_coupling_id
        target_sliding_coupling_id: str | None = None
        previous_route = candidate_route = None
        source_state_entry = None
        source_composition = None
        external_specs: dict[str, FiniteVolumeOwnerLocalCellStateBankSpec] = {}
        retained_ids: tuple[str, ...] = ()
        invalidated_ids: tuple[str, ...] = ()
        prepared_entries: tuple[CompositionEntry, ...] = ()
        bank_inputs: list[tuple[str, CompositionEntry, np.ndarray, Any, Any]] = []
        source_plan_ids = source_owned_ids = target_plan_ids = target_owned_ids = None
        target_owned_mask: np.ndarray | None = None
        source_owned_rows = target_owned_rows = None
        kept_source_rows = kept_target_rows = np.zeros((0,), dtype=np.int32)
        target_layout: Any = None
        target_runtime: PreparedFiniteVolumeRuntime | None = None
        try:
            local_rank = members.index(jax.process_index())
            if (
                not part_name
                or not isinstance(self.dynamics, PreparedUnstructuredFiniteVolumeDynamics)
                or self.dynamics.coupling.overset is None
                or self.owner_local_geometry is None
                or not isinstance(
                    self.owner_local_geometry,
                    (OwnerLocalCellGeometry, OwnerLocalCompositeCellGeometry),
                )
                or not previous_connectivity.owner_local
                or not candidate_connectivity.owner_local
                or candidate_connectivity.epoch != previous_connectivity.epoch + 1
                or candidate_connectivity.predecessor_id
                != previous_connectivity.connectivity_id
                or self.dynamics.coupling.overset.native_connectivity_id
                != previous_connectivity.connectivity_id
                or accepted_state.topology_journal.current_epoch_id
                != current_epoch.epoch_id
                or source_content.topology_epoch_id != current_epoch.epoch_id
                or self.initial_topology_epoch.epoch_id != current_epoch.epoch_id
                or self.initial_topology_artifacts.artifacts_id
                != current_artifacts.artifacts_id
            ):
                raise ValueError(
                    "Accepted FV runtime does not bind the predecessor graph partition."
                )
            self._validate_sliding_state(accepted_state)
            source_dynamics = self.dynamics
            if not isinstance(source_dynamics, PreparedUnstructuredFiniteVolumeDynamics):
                raise TypeError("Owner-local source runtime is not unstructured FV.")
            source_geometry = source_dynamics.discretization
            if not isinstance(source_geometry, UnstructuredFiniteVolumeDiscretization):
                raise TypeError("Owner-local source plan lacks unstructured cell IDs.")
            source_layout = self.owner_local_geometry
            if not isinstance(
                source_layout, (OwnerLocalCellGeometry, OwnerLocalCompositeCellGeometry)
            ):
                raise TypeError("Accepted runtime lost its owner-local source view.")
            source_geometry_layout = source_geometry.owner_local_view
            if not isinstance(
                source_geometry_layout,
                (OwnerLocalCellGeometry, OwnerLocalCompositeCellGeometry),
            ):
                raise TypeError("Accepted FV geometry lost its owner-local source view.")
            if source_geometry_layout.local_view_id != source_layout.local_view_id:
                raise ValueError(
                    "Accepted FV geometry differs from its owner-local source view."
                )
            if isinstance(source_layout, OwnerLocalCompositeCellGeometry):
                # A composite layout migrates one named part; the other parts
                # keep their owned cells and are carried by stable cell ID.
                if owner_local_part_name is None:
                    raise ValueError(
                        "Composite FV repartition must name its migrated part."
                    )
                part_name = str(owner_local_part_name)
                source_view = source_layout.part(part_name)
            else:
                source_view = source_layout
                part_name = (
                    source_view.part_name
                    if owner_local_part_name is None
                    else str(owner_local_part_name)
                )
            target_view_candidate = candidate_connectivity.assembly.part(
                part_name
            ).carrier
            if not isinstance(target_view_candidate, OwnerLocalCellGeometry):
                raise TypeError(
                    "Candidate connectivity lacks its owner-local target view."
                )
            target_view = target_view_candidate
            target_geometry = target_dynamics.discretization
            if not isinstance(target_geometry, UnstructuredFiniteVolumeDiscretization):
                raise TypeError("Target FV plan lacks unstructured cell IDs.")
            target_layout_candidate = target_geometry.owner_local_view
            if not isinstance(
                target_layout_candidate,
                (OwnerLocalCellGeometry, OwnerLocalCompositeCellGeometry),
            ):
                raise TypeError("Target FV geometry lacks its owner-local view.")
            target_layout = target_layout_candidate
            if isinstance(source_layout, OwnerLocalCompositeCellGeometry):
                if (
                    not isinstance(target_layout, OwnerLocalCompositeCellGeometry)
                    or target_layout.part_names != source_layout.part_names
                    or target_layout.part(part_name).local_view_id
                    != target_view.local_view_id
                    or any(
                        target_layout.part(name).local_view_id
                        != source_layout.part(name).local_view_id
                        for name in source_layout.part_names
                        if name != part_name
                    )
                ):
                    raise ValueError(
                        "Target composite layout must migrate only the named part."
                    )
            elif target_layout.local_view_id != target_view.local_view_id:
                raise ValueError(
                    "Target dynamics do not bind the candidate connectivity view."
                )
            target_overset = target_dynamics.coupling.overset
            if (
                target_overset is None
                or target_overset.native_connectivity_id
                != candidate_connectivity.connectivity_id
            ):
                raise ValueError(
                    "Target dynamics do not bind the candidate connectivity view."
                )
            if source_dynamics.coupling.overset is None or (
                source_dynamics.coupling.overset.native_connectivity_id
                != previous_connectivity.connectivity_id
            ):
                raise ValueError("Source dynamics do not bind predecessor connectivity.")
            graph_migration = target_view.graph_migration_evidence
            if not isinstance(graph_migration, OwnerLocalGraphMigrationEvidence):
                raise ValueError(
                    "Candidate owner-local view lacks native graph migration evidence."
                )
            if (
                source_view.owner_rank != local_rank
                or target_view.owner_rank != local_rank
                or source_view.part_name != target_view.part_name
                or source_layout.source_topology_id != current_epoch.topology_id
                or source_layout.source_geometry_id != current_epoch.geometry_id
                or target_layout.source_topology_id != current_epoch.topology_id
                or target_layout.source_geometry_id != current_epoch.geometry_id
                or source_layout.distribution_id != current_epoch.partition_id
                or target_layout.distribution_id == current_epoch.partition_id
                or graph_migration.local_rank != local_rank
                or graph_migration.part_count != len(members)
                or graph_migration.group_id != execution_group.spec.group_id
                or graph_migration.source_part_id != source_view.part_id
                or graph_migration.source_directory_id != source_view.distribution_id
                or graph_migration.source_candidate_id
                != source_view.candidate_revision_id
                or graph_migration.periodic_face_coverage_id
                != source_view.periodic_face_coverage_id
                or graph_migration.target_part_id != target_view.part_id
                or graph_migration.target_directory_id != target_view.distribution_id
                or not graph_migration.result.feasible
                or graph_migration.result.migration_count <= 0
            ):
                raise ValueError(
                    "Native graph migration does not bind the accepted source/target."
                )
            source_plan_ids = np.asarray(source_geometry.cell_global_ids, dtype=np.int64)
            target_plan_ids = np.asarray(target_geometry.cell_global_ids, dtype=np.int64)
            source_owned_plan_mask = self.owner_local_owned_cell_mask
            if source_owned_plan_mask is None:
                raise ValueError("Accepted owner-local runtime lacks its ownership mask.")
            source_owned_mask = np.asarray(source_owned_plan_mask, dtype=np.bool_)

            def layout_owned_mask(layout: Any, plan_ids: np.ndarray, /) -> np.ndarray:
                layout_ids = np.asarray(layout.cell_ids, dtype=np.int64)
                layout_owned = np.asarray(layout.owned, dtype=np.bool_)
                layout_row = {
                    int(identifier): row
                    for row, identifier in enumerate(layout_ids.tolist())
                }
                return np.asarray(
                    [
                        layout_owned[layout_row[int(identifier)]]
                        for identifier in plan_ids
                    ],
                    dtype=np.bool_,
                )

            target_owned_mask = layout_owned_mask(target_layout, target_plan_ids)
            source_view_owned_mask = layout_owned_mask(source_layout, source_plan_ids)
            if (
                source_plan_ids.ndim != 1
                or target_plan_ids.ndim != 1
                or source_owned_mask.shape != source_plan_ids.shape
                or target_owned_mask.shape != target_plan_ids.shape
                or np.unique(source_plan_ids).size != source_plan_ids.size
                or np.unique(target_plan_ids).size != target_plan_ids.size
                or not np.array_equal(
                    np.sort(source_plan_ids),
                    np.sort(np.asarray(source_layout.cell_ids, dtype=np.int64)),
                )
                or not np.array_equal(
                    np.sort(target_plan_ids),
                    np.sort(np.asarray(target_layout.cell_ids, dtype=np.int64)),
                )
                or not np.array_equal(source_owned_mask, source_view_owned_mask)
            ):
                raise ValueError(
                    "FV plan rows do not match the accepted owner-local cell views."
                )
            source_part_mask = np.isin(
                source_plan_ids, np.asarray(source_view.cell_ids, dtype=np.int64)
            )
            target_part_mask = np.isin(
                target_plan_ids, np.asarray(target_view.cell_ids, dtype=np.int64)
            )
            source_owned_rows = np.flatnonzero(source_owned_mask & source_part_mask)
            target_owned_rows = np.flatnonzero(target_owned_mask & target_part_mask)
            source_owned_ids = np.sort(source_plan_ids[source_owned_rows])
            target_owned_ids = np.sort(target_plan_ids[target_owned_rows])
            # Owned cells of the parts that do not migrate stay on this owner.
            source_kept_ids = np.sort(
                source_plan_ids[source_owned_mask & ~source_part_mask]
            )
            target_kept_ids = np.sort(
                target_plan_ids[target_owned_mask & ~target_part_mask]
            )
            if not np.array_equal(source_kept_ids, target_kept_ids):
                raise ValueError(
                    "Composite repartition changed the owned cells of a kept part."
                )
            source_plan_row_of = {
                int(identifier): row
                for row, identifier in enumerate(source_plan_ids.tolist())
            }
            target_plan_row_of = {
                int(identifier): row
                for row, identifier in enumerate(target_plan_ids.tolist())
            }
            kept_source_rows = np.asarray(
                [source_plan_row_of[int(identifier)] for identifier in source_kept_ids],
                dtype=np.int32,
            )
            kept_target_rows = np.asarray(
                [target_plan_row_of[int(identifier)] for identifier in target_kept_ids],
                dtype=np.int32,
            )
            if (
                not np.array_equal(
                    source_owned_ids,
                    np.sort(
                        np.asarray(graph_migration.result.entity_ids, dtype=np.int64)
                    ),
                )
                or not np.array_equal(
                    source_owned_ids,
                    np.sort(np.asarray(source_view.owned_cell_ids, dtype=np.int64)),
                )
                or not np.array_equal(
                    target_owned_ids,
                    np.asarray(graph_migration.target_cell_ids, dtype=np.int64),
                )
                or not np.array_equal(
                    target_owned_ids,
                    np.sort(np.asarray(target_view.owned_cell_ids, dtype=np.int64)),
                )
                or source_content.owned_cell_mask is None
                or not np.array_equal(
                    np.asarray(source_content.owned_cell_mask, dtype=np.bool_),
                    source_owned_mask,
                )
                or target_dynamics.system.system_id != self.dynamics.system.system_id
                or target_dynamics.precision.policy_id != self.precision.policy_id
            ):
                raise ValueError(
                    "Source/target FV owner masks or graph cell keys differ."
                )
            candidate_epoch = TopologyEpoch(
                current_epoch.index + 1,
                current_epoch.geometry_id,
                current_epoch.topology_id,
                target_layout.distribution_id,
            )
            if candidate_epoch.partition_id == current_epoch.partition_id:
                raise ValueError(
                    "Graph repartition did not change the global partition identity."
                )
            source_sliding_plan = self.sliding_plan
            target_sliding_plan = target_dynamics.coupling.sliding
            source_sliding_plan_id = (
                None if source_sliding_plan is None else source_sliding_plan.plan_id
            )
            target_sliding_plan_id = (
                None if target_sliding_plan is None else target_sliding_plan.plan_id
            )
            if source_sliding_plan_id != target_sliding_plan_id:
                raise ValueError(
                    "Repartition changes the declared sliding-interface plan."
                )
            if target_sliding_plan is not None:
                target_sliding_coupling = target_sliding_plan.coupling(
                    accepted_state.sliding_shift
                )
                target_dynamics = target_dynamics.with_sliding_coupling(
                    target_sliding_coupling
                )
                target_sliding_coupling_id = target_sliding_coupling.coupling_id
            candidate_artifacts = FiniteVolumeTopologyArtifacts(
                candidate_epoch,
                target_geometry.prepared_id,
                topology_artifact_id=candidate_connectivity.connectivity_id,
                metrics_artifact_id=graph_migration.evidence_id,
                operators_artifact_id=target_dynamics.dynamics_id,
            )
            route_identity_id = canonical_fingerprint(
                {
                    "kind": "finite-volume-owner-local-graph-repartition",
                    "source_part": graph_migration.source_part_id,
                    "source_directory": graph_migration.source_directory_id,
                    "target_part": graph_migration.target_part_id,
                    "target_directory": graph_migration.target_directory_id,
                    "graph_plan": graph_migration.result.plan_id,
                    "group": graph_migration.group_id,
                }
            )
            previous_entries = previous_connectivity.composition_entries()
            candidate_entries = candidate_connectivity.composition_entries()
            previous_route = next(
                entry for entry in previous_entries if entry.role == "interface-route"
            )
            candidate_route = next(
                entry for entry in candidate_entries if entry.role == "interface-route"
            )
            source_entries = (
                previous_entries
                if accepted_composition is None
                else accepted_composition.entries
            )
            source_entry_by_id = {entry.entry_id: entry for entry in source_entries}
            if accepted_composition is not None:
                for previous_entry in previous_entries:
                    if (
                        accepted_composition.entry(previous_entry.entry_id).record_id
                        != previous_entry.record_id
                    ):
                        raise ValueError(
                            "Accepted Composition does not bind predecessor connectivity."
                        )
                accepted_route = accepted_composition.entry(previous_route.entry_id)
                if accepted_route.revision_id != previous_route.revision_id:
                    raise ValueError(
                        "Accepted Composition does not bind the predecessor route."
                    )
            retained_ids = tuple(sorted(set(retained_state_entry_ids)))
            invalidated_ids = tuple(sorted(set(invalidate_entry_ids)))
            prepared_entries = tuple(reprepare_entries)
            prepared_by_id = {entry.entry_id: entry for entry in prepared_entries}
            if (
                len(retained_ids) != len(retained_state_entry_ids)
                or len(invalidated_ids) != len(invalidate_entry_ids)
                or len(prepared_by_id) != len(prepared_entries)
                or set(retained_ids) & set(invalidated_ids)
                or set(retained_ids) & set(prepared_by_id)
                or set(invalidated_ids) & set(prepared_by_id)
            ):
                raise ValueError("Composition repartition dispositions are duplicated.")
            external_entries = tuple(
                entry
                for entry in source_entries
                if entry.entry_id not in {item.entry_id for item in previous_entries}
                and entry.entry_id != "finite-volume/state"
            )
            external_by_id = {entry.entry_id: entry for entry in external_entries}
            external_specs = {spec.entry_id: spec for spec in cell_state_banks}
            if (
                len(external_specs) != len(cell_state_banks)
                or set(external_specs) & set(retained_ids)
                or set(external_specs) & set(invalidated_ids)
                or set(external_specs) & set(prepared_by_id)
            ):
                raise ValueError(
                    "Cell-state bank declarations overlap other dispositions."
                )
            state_roles = {
                "physical-state",
                "exchange-state",
                "budget",
                "model-parameter",
                "model-state",
                "optimizer-state",
                "history",
                "statistics",
                "rng",
            }
            derived_roles = {
                "topology",
                "discretization",
                "interface-route",
                "transfer",
                "preconditioner",
                "observation",
                "prepared-graph",
                "workset",
            }
            for entry_id, spec in external_specs.items():
                entry = external_by_id.get(entry_id)
                if entry is None or entry.role not in state_roles:
                    raise ValueError("Cell-state bank must name an accepted state entry.")
                if spec.cell_axis >= np.ndim(entry.value) or spec.cell_axis < -np.ndim(
                    entry.value
                ):
                    raise ValueError(
                        "Cell-state bank axis is outside its typed value rank."
                    )
            for entry_id in retained_ids:
                entry = external_by_id.get(entry_id)
                if entry is None or entry.role not in state_roles:
                    raise ValueError(
                        "Only explicitly declared global state entries may be retained."
                    )
            for entry_id in invalidated_ids:
                entry = external_by_id.get(entry_id)
                if entry is None or entry.role not in derived_roles:
                    raise ValueError(
                        "Only derived artifacts may be explicitly invalidated."
                    )
            for entry_id, target_entry in prepared_by_id.items():
                entry = external_by_id.get(entry_id)
                if (
                    entry is None
                    or entry.role not in derived_roles
                    or target_entry.role != entry.role
                    or target_entry.owner_id != entry.owner_id
                ):
                    raise ValueError(
                        "Reprepared entries must replace matching derived artifacts."
                    )
            dispositions = (
                set(external_specs)
                | set(retained_ids)
                | set(invalidated_ids)
                | set(prepared_by_id)
            )
            if dispositions != set(external_by_id):
                unknown = sorted(set(external_by_id) - dispositions)
                raise ValueError(
                    "Every non-connectivity Composition entry needs an explicit repartition "
                    f"disposition; unresolved entries: {unknown!r}."
                )
            source_state_entry = CompositionEntry(
                accepted_state,
                entry_id="finite-volume/state",
                role="physical-state",
                owner_id="phydrax.solver.PreparedFiniteVolumeRuntime",
                structure_id=source_content.geometry_layout_id,
                revision_id=canonical_fingerprint(
                    {
                        "kind": "finite-volume-accepted-state",
                        "runtime": self.runtime_id,
                        "time": float(np.asarray(accepted_state.time)).hex(),
                        "accepted_step": accepted_step,
                        "content": array_tree_fingerprint(
                            source_content.conservative_content
                        ),
                        "controller": array_tree_fingerprint(
                            accepted_state.controller_state
                        ),
                        "integrator": array_tree_fingerprint(
                            accepted_state.integrator_state
                        ),
                        "output_cursor": int(np.asarray(accepted_state.output_cursor)),
                    }
                ),
                semantics_id=self.dynamics.system.system_id,
                dependencies=(previous_route.binding("revision"),),
            )
            source_composition = Composition(
                (
                    *(
                        entry
                        for entry in source_entries
                        if entry.entry_id != "finite-volume/state"
                    ),
                    source_state_entry,
                ),
                boundary_id=canonical_fingerprint(
                    {
                        "kind": "finite-volume-accepted-boundary",
                        "epoch": current_epoch.epoch_id,
                        "accepted_step": accepted_step,
                    }
                ),
            )
            source_plan_row = {
                int(identifier): row for row, identifier in enumerate(source_plan_ids)
            }
            source_rows = np.asarray(
                [source_plan_row[int(identifier)] for identifier in source_owned_ids],
                dtype=np.int32,
            )
            if (
                np.asarray(source_content.conservative_content).shape[0]
                != source_plan_ids.size
                or np.asarray(source_content.effective_cell_volumes).shape
                != source_plan_ids.shape
                or np.asarray(source_content.active_cell_mask).shape
                != source_plan_ids.shape
                or not np.array_equal(
                    np.asarray(source_content.effective_cell_volumes),
                    np.asarray(self.effective_cell_volumes),
                )
                or not np.array_equal(
                    np.asarray(source_content.active_cell_mask),
                    np.asarray(self.active_cell_mask),
                )
                or float(np.asarray(source_content.time))
                != float(np.asarray(accepted_state.time))
            ):
                raise ValueError(
                    "Accepted owner-local content differs from its source runtime."
                )

            def add_bank(
                name: str,
                source_entry: CompositionEntry,
                source_ids: np.ndarray,
                source_values: Any,
                /,
            ) -> None:
                if not isinstance(source_values, Array) and not hasattr(
                    source_values, "shape"
                ):
                    raise TypeError("Graph-migrated cell bank values must be arrays.")
                values_shape = tuple(int(value) for value in source_values.shape)
                if not values_shape or values_shape[0] != source_ids.size:
                    raise ValueError(
                        "Graph-migrated cell bank rows lack explicit stable IDs."
                    )
                bank_inputs.append(
                    (name, source_entry, source_ids, source_values, values_shape[1:])
                )

            core_names = (
                "conservative_content",
                "effective_cell_volumes",
                "active_cell_mask",
            )
            core_values = (
                jnp.asarray(source_content.conservative_content)[source_rows],
                jnp.asarray(source_content.effective_cell_volumes)[source_rows],
                jnp.asarray(source_content.active_cell_mask)[source_rows],
            )
            for name, values in zip(core_names, core_values, strict=True):
                entry_id = f"finite-volume/repartition/{name}"
                temp_entry = CompositionEntry(
                    values,
                    entry_id=entry_id,
                    role="physical-state",
                    owner_id="phydrax.solver.PreparedFiniteVolumeRuntime",
                    structure_id=source_view.geometry_layout_id,
                    revision_id=canonical_fingerprint(
                        {
                            "kind": "finite-volume-repartition-cell-bank-schema",
                            "source_geometry": current_epoch.geometry_id,
                            "name": name,
                            "tail_shape": tuple(int(value) for value in values.shape[1:]),
                            "dtype": str(values.dtype),
                        }
                    ),
                    semantics_id=canonical_fingerprint(
                        {
                            "kind": "finite-volume-cell-bank-semantics",
                            "system": self.dynamics.system.system_id,
                            "name": name,
                        }
                    ),
                )
                add_bank(name, temp_entry, source_owned_ids, values)
            for entry_id in sorted(external_specs):
                spec = external_specs[entry_id]
                source_entry = external_by_id[entry_id]
                row_ids = np.asarray(spec.cell_ids, dtype=np.int64)
                source_local_ids = np.asarray(source_view.cell_ids, dtype=np.int64)
                if not (
                    np.array_equal(np.sort(row_ids), source_owned_ids)
                    or np.array_equal(np.sort(row_ids), np.sort(source_local_ids))
                ):
                    raise ValueError(
                        f"Cell-state bank {entry_id!r} must label every source-owned "
                        "row or every resident source row explicitly."
                    )
                axis = spec.cell_axis
                source_array = jnp.asarray(source_entry.value)
                values = jnp.moveaxis(source_array, axis, 0)
                id_to_row = {
                    int(identifier): row for row, identifier in enumerate(row_ids)
                }
                ordered_rows = jnp.asarray(
                    [id_to_row[int(identifier)] for identifier in source_owned_ids],
                    dtype=jnp.int32,
                )
                values = values[ordered_rows]
                transfer_entry = CompositionEntry(
                    values,
                    entry_id=f"finite-volume/repartition/cell-bank/{entry_id}",
                    role=source_entry.role,
                    owner_id=source_entry.owner_id,
                    structure_id=source_view.geometry_layout_id,
                    revision_id=canonical_fingerprint(
                        {
                            "kind": "finite-volume-repartition-cell-bank-schema",
                            "entry": entry_id,
                            "source_geometry": current_epoch.geometry_id,
                            "role": source_entry.role,
                            "owner": source_entry.owner_id,
                            "semantics": source_entry.semantics_id,
                            "cell_axis": axis,
                            "tail_shape": tuple(int(value) for value in values.shape[1:]),
                            "dtype": str(values.dtype),
                        }
                    ),
                    semantics_id=source_entry.semantics_id,
                )
                add_bank(f"entry:{entry_id}", transfer_entry, source_owned_ids, values)
            schema_id = canonical_fingerprint(
                tuple(
                    (
                        name,
                        entry.entry_id,
                        entry.role,
                        entry.owner_id,
                        entry.structure_id,
                        entry.revision_id,
                        entry.semantics_id,
                        shape,
                        str(values.dtype),
                    )
                    for name, entry, _, values, shape in bank_inputs
                )
            )
        except Exception as error:
            preflight_error = error
            schema_id = canonical_fingerprint(
                {"kind": "invalid-finite-volume-repartition-bank-schema"}
            )
        try:
            group_state_id = canonical_fingerprint(
                {
                    "group": execution_group.spec.group_id,
                    "request": request.request_id,
                    "epoch": current_epoch.epoch_id,
                    "accepted_step": accepted_step,
                    "time": float(event_time).hex(),
                    "journal_capacity": accepted_state.topology_journal.capacity,
                    "journal_count": int(
                        np.asarray(accepted_state.topology_journal.count)
                    ),
                }
            )
            group_rows = _owner_group_allgather(
                np.frombuffer(bytes.fromhex(group_state_id), dtype=np.uint8),
                execution_group,
                f"{self.owner_local_axis_name}_repartition_boundary",
            )
            if any(
                np.asarray(row, dtype=np.uint8).tobytes().hex() != group_state_id
                for row in group_rows
            ):
                preflight_error = ValueError(
                    "Owner ranks do not share one accepted FV event boundary."
                )
            schema_rows = _owner_group_allgather(
                np.frombuffer(bytes.fromhex(schema_id), dtype=np.uint8),
                execution_group,
                f"{self.owner_local_axis_name}_repartition_schema",
            )
            if any(
                np.asarray(row, dtype=np.uint8).tobytes().hex() != schema_id
                for row in schema_rows
            ):
                preflight_error = ValueError(
                    "Owner ranks declared different cell-state bank schemas."
                )
        except Exception as error:
            preflight_error = error
        preflight_error_text = collective_error(
            preflight_error, f"{self.owner_local_axis_name}_repartition_admission"
        )
        if preflight_error_text is not None:
            return fail(TopologyEventStatus.FAILED_MISSING_ARTIFACT, preflight_error_text)
        if (
            not isinstance(source_view, OwnerLocalCellGeometry)
            or not isinstance(target_view, OwnerLocalCellGeometry)
            or not isinstance(graph_migration, OwnerLocalGraphMigrationEvidence)
            or previous_route is None
            or candidate_route is None
            or source_state_entry is None
            or source_composition is None
            or source_plan_ids is None
            or source_owned_ids is None
            or target_plan_ids is None
            or target_owned_ids is None
            or source_owned_rows is None
            or target_owned_rows is None
            or target_owned_mask is None
            or target_layout is None
        ):
            return fail(
                TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                "Owner-local repartition preflight produced incomplete typed state.",
            )

        target_runtime_error: Exception | None = None
        try:
            target_runtime = PreparedFiniteVolumeRuntime(
                target_dynamics,
                self.positivity,
                self.policy,
                topology_epoch=candidate_epoch,
                topology_artifacts=candidate_artifacts,
                execution_group=execution_group,
                owner_local_geometry=target_layout,
                owner_local_axis_name=self.owner_local_axis_name,
            )
            runtime_sliding_plan_id = (
                None
                if target_runtime.sliding_plan is None
                else target_runtime.sliding_plan.plan_id
            )
            runtime_sliding_coupling_id = (
                None
                if target_runtime.sliding_initial_coupling is None
                else target_runtime.sliding_initial_coupling.coupling_id
            )
            # The owner-local geometry family is the rank-local carrier, which a
            # graph repartition changes by definition; the global source geometry
            # and topology are bound by the candidate epoch above.
            changed = tuple(
                name
                for name, differs in (
                    (
                        "owned_cell_mask",
                        target_runtime.owner_local_owned_cell_mask is None
                        or not np.array_equal(
                            np.asarray(target_runtime.owner_local_owned_cell_mask),
                            target_owned_mask,
                        ),
                    ),
                    (
                        "evidence_policy",
                        target_runtime.evidence_policy_id
                        != source_content.evidence_policy_id,
                    ),
                    ("sliding_plan", runtime_sliding_plan_id != source_sliding_plan_id),
                    (
                        "sliding_coupling",
                        runtime_sliding_coupling_id != target_sliding_coupling_id,
                    ),
                )
                if differs
            )
            if changed:
                raise ValueError(
                    f"Target runtime changes preserved FV state semantics: {changed}."
                )
        except Exception as error:
            target_runtime_error = error
        target_runtime_error_text = collective_error(
            target_runtime_error,
            f"{self.owner_local_axis_name}_repartition_target_runtime",
        )
        if target_runtime_error_text is not None:
            return fail(
                TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                target_runtime_error_text,
            )
        if target_runtime is None:
            return fail(
                TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                "Target FV runtime preparation produced no runtime.",
            )
        target_owned_plan_mask = target_runtime.owner_local_owned_cell_mask
        if target_owned_plan_mask is None:
            return fail(
                TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                "Target owner-local FV runtime has no ownership mask.",
            )

        retained_entries = tuple(
            source_entry_by_id[entry_id] for entry_id in retained_ids
        )
        retained_state_identity = tuple(
            (
                entry.entry_id,
                entry.role,
                entry.owner_id,
                entry.structure_id,
                entry.revision_id,
                entry.semantics_id,
                array_tree_fingerprint(entry.value),
                tuple(
                    (dependency.entry_id, dependency.facet, dependency.bound_id)
                    for dependency in entry.dependencies
                ),
            )
            for entry in retained_entries
        )
        state_global_payload = {
            "time": float(np.asarray(accepted_state.time)).hex(),
            "accepted_step": accepted_step,
            "step_size": array_tree_fingerprint(accepted_state.step_size),
            "last_status": int(np.asarray(accepted_state.last_status)),
            "controller_state": array_tree_fingerprint(accepted_state.controller_state),
            "integrator_state": array_tree_fingerprint(accepted_state.integrator_state),
            "output_cursor": int(np.asarray(accepted_state.output_cursor)),
            "sliding_plan_id": source_sliding_plan_id,
            "sliding_shift": array_tree_fingerprint(accepted_state.sliding_shift),
            "sliding_event_id": accepted_state.sliding_event_id,
            "geometry_family": source_content.geometry_family_id,
            "evidence_policy": source_content.evidence_policy_id,
            "geometry_version": array_tree_fingerprint(source_content.geometry_version),
            "evidence_version": array_tree_fingerprint(source_content.evidence_version),
            "precision": source_content.precision.policy_id,
            "owned_cell_mask_present": source_content.owned_cell_mask is not None,
            "retained_global_entries": retained_state_identity,
        }
        key_identity_id = canonical_fingerprint(
            {
                "kind": "finite-volume-stable-cell-global-id",
                "topology": current_epoch.topology_id,
            }
        )
        transfer_holder: dict[str, Any] = {}

        def failed_transfer(
            reason: str,
            /,
            *,
            content: FiniteVolumeConservativeContentState = source_content,
        ) -> _OwnerLocalRepartitionTransfer:
            transfer_holder["refusal"] = reason
            rejected_root = canonical_fingerprint(
                {
                    "kind": "rejected-owner-local-fv-cell-transfer",
                    "request": request.request_id,
                    "reason": reason,
                }
            )
            return _OwnerLocalRepartitionTransfer(
                content,
                graph_migration.evidence_id,
                route_identity_id,
                rejected_root,
                rejected_root,
                (),
                (),
                False,
            )

        def transfer_cells(
            content: Any,
            migration: Any,
            /,
        ) -> _OwnerLocalRepartitionTransfer:
            local_error: Exception | None = None
            if content is not source_content or migration is not graph_migration:
                local_error = ValueError(
                    "Repartition transfer source or graph evidence changed after admission."
                )
            error_text = collective_error(
                local_error, f"{self.owner_local_axis_name}_repartition_transfer_source"
            )
            if error_text is not None:
                return failed_transfer(error_text)
            transfer_rows: dict[str, Any] = {}
            for ordinal, (name, entry, source_ids, values, _) in enumerate(bank_inputs):
                transfer_error: Exception | None = None
                bank_transfer = None
                try:
                    source_owners = np.full(source_ids.size, local_rank, dtype=np.int32)
                    target_owners = np.full(
                        target_owned_ids.size, local_rank, dtype=np.int32
                    )
                    bank_transfer = graph_migration.transfer_keyed_values(
                        entry,
                        source_ids,
                        source_owners,
                        values,
                        target_owned_ids,
                        target_owners,
                        execution_group,
                        target_structure_id=target_view.geometry_layout_id,
                        key_identity_id=key_identity_id,
                        axis_name=(
                            f"{self.owner_local_axis_name}_repartition_bank_{ordinal}"
                        ),
                    )
                    if (
                        bank_transfer.migration_evidence_id != graph_migration.evidence_id
                        or bank_transfer.source_key_set_id
                        != bank_transfer.target_key_set_id
                        or bank_transfer.source_content_root
                        != bank_transfer.target_content_root
                        or not bool(np.asarray(bank_transfer.successful))
                        or not np.array_equal(
                            np.asarray(bank_transfer.target_entity_keys)[:, 0],
                            target_owned_ids,
                        )
                    ):
                        raise ValueError(
                            f"Graph transfer for {name!r} did not preserve exact keyed rows."
                        )
                except Exception as error:
                    transfer_error = error
                error_text = collective_error(
                    transfer_error,
                    f"{self.owner_local_axis_name}_repartition_bank_{ordinal}_admission",
                )
                if error_text is not None or bank_transfer is None:
                    return failed_transfer(
                        error_text or "A graph cell-row transfer failed."
                    )
                transfer_rows[name] = bank_transfer

            target_content_state: FiniteVolumeConservativeContentState | None = None
            content_error: Exception | None = None
            try:
                target_row_by_id = {
                    int(identifier): row
                    for row, identifier in enumerate(target_plan_ids.tolist())
                }
                target_rows = np.asarray(
                    [
                        target_row_by_id[int(identifier)]
                        for identifier in target_owned_ids
                    ],
                    dtype=np.int32,
                )
                if not np.array_equal(
                    np.sort(target_plan_ids[target_owned_rows]), target_owned_ids
                ):
                    raise ValueError(
                        "Target plan owned rows differ from graph migration."
                    )
                migrated_content = np.asarray(
                    transfer_rows["conservative_content"].target_values
                )
                migrated_volumes = np.asarray(
                    transfer_rows["effective_cell_volumes"].target_values
                )
                migrated_active = np.asarray(
                    transfer_rows["active_cell_mask"].target_values,
                    dtype=np.bool_,
                )
                target_volumes = np.asarray(target_runtime.effective_cell_volumes)
                target_active = np.asarray(
                    target_runtime.active_cell_mask, dtype=np.bool_
                )
                source_kept_content = np.asarray(source_content.conservative_content)[
                    kept_source_rows
                ]
                if (
                    not np.array_equal(migrated_volumes, target_volumes[target_rows])
                    or not np.array_equal(migrated_active, target_active[target_rows])
                    or migrated_content.shape[0] != target_owned_ids.size
                    or not np.array_equal(
                        np.asarray(source_content.effective_cell_volumes)[
                            kept_source_rows
                        ],
                        target_volumes[kept_target_rows],
                    )
                    or not np.array_equal(
                        np.asarray(source_content.active_cell_mask, dtype=np.bool_)[
                            kept_source_rows
                        ],
                        target_active[kept_target_rows],
                    )
                ):
                    raise ValueError(
                        "Migrated cell measures/activity differ from target geometry."
                    )
                content_values = np.zeros(
                    (target_plan_ids.size, *migrated_content.shape[1:]),
                    dtype=migrated_content.dtype,
                )
                content_values[target_rows] = migrated_content
                content_values[kept_target_rows] = source_kept_content
                safe_volumes = np.where(target_active, target_volumes, 1.0)
                trailing = (1,) * (content_values.ndim - 1)
                owned_average = np.where(
                    target_active.reshape((-1,) + trailing),
                    content_values / safe_volumes.reshape((-1,) + trailing),
                    0.0,
                )
                refreshed_average = target_runtime._provide_stage_state(
                    source_content.time, jnp.asarray(owned_average)
                )
                target_runtime.precision.validate_state(refreshed_average)
                refreshed_content = np.asarray(
                    refreshed_average
                    * jnp.asarray(target_volumes).reshape((-1,) + trailing)
                )
                refreshed_content = np.where(
                    target_active.reshape((-1,) + trailing), refreshed_content, 0.0
                )
                refreshed_content[target_rows] = migrated_content
                refreshed_content[kept_target_rows] = source_kept_content
                target_content_state = FiniteVolumeConservativeContentState(
                    refreshed_content,
                    target_runtime.effective_cell_volumes,
                    target_runtime.active_cell_mask,
                    source_content.time,
                    topology_epoch_id=candidate_epoch.epoch_id,
                    geometry_family_id=target_runtime.geometry_family_id,
                    geometry_layout_id=target_runtime.geometry_layout_id,
                    geometry_version=source_content.geometry_version,
                    evidence_policy_id=target_runtime.evidence_policy_id,
                    evidence_version=source_content.evidence_version,
                    precision=target_runtime.precision,
                    owned_cell_mask=target_owned_plan_mask,
                )
                target_runtime.precision.validate_state(
                    target_content_state.conservative_content
                )
                active_owned = target_runtime.active_cell_mask & target_owned_plan_mask
                valid = jnp.all(
                    jnp.where(
                        active_owned,
                        target_runtime.dynamics.system.admissible(
                            target_content_state.cell_average()
                        ),
                        True,
                    )
                )
                valid = target_runtime.owner_local_decision_provider(valid)
                if not bool(np.asarray(valid)):
                    raise ValueError("Migrated target FV state is inadmissible.")
            except Exception as error:
                content_error = error
            error_text = collective_error(
                content_error,
                f"{self.owner_local_axis_name}_repartition_content_admission",
            )
            if error_text is not None or target_content_state is None:
                return failed_transfer(
                    error_text or "Target content construction failed."
                )

            source_banks = tuple(
                (name, row.source_content_root)
                for name, row in sorted(transfer_rows.items())
            )
            target_banks = tuple(
                (name, row.target_content_root)
                for name, row in sorted(transfer_rows.items())
            )
            source_root = canonical_fingerprint(
                {
                    "kind": "finite-volume-owner-local-runtime-state-bank-v1",
                    "global_fields": state_global_payload,
                    "cell_banks": source_banks,
                }
            )
            target_root = canonical_fingerprint(
                {
                    "kind": "finite-volume-owner-local-runtime-state-bank-v1",
                    "global_fields": state_global_payload,
                    "cell_banks": target_banks,
                }
            )
            if source_root != target_root:
                return failed_transfer(
                    "Exact runtime state roots changed during owner-local migration."
                )
            transfer_holder["rows"] = transfer_rows
            bank_route_evidence = tuple(
                (
                    name,
                    row.transfer_id,
                    row.source_owner_map_id,
                    row.target_owner_map_id,
                )
                for name, row in sorted(transfer_rows.items())
            )
            return _OwnerLocalRepartitionTransfer(
                target_content_state,
                graph_migration.evidence_id,
                route_identity_id,
                source_root,
                target_root,
                tuple(sorted(transfer_rows)),
                bank_route_evidence,
                True,
            )

        def target_state_entry(
            state: FiniteVolumeRuntimeState,
            route: CompositionEntry,
            runtime: PreparedFiniteVolumeRuntime,
            /,
        ) -> CompositionEntry:
            content = state.content_state
            revision_id = canonical_fingerprint(
                {
                    "kind": "finite-volume-accepted-state",
                    "runtime": runtime.runtime_id,
                    "time": float(np.asarray(state.time)).hex(),
                    "accepted_step": int(np.asarray(state.accepted_step)),
                    "content": array_tree_fingerprint(content.conservative_content),
                    "controller": array_tree_fingerprint(state.controller_state),
                    "integrator": array_tree_fingerprint(state.integrator_state),
                    "output_cursor": int(np.asarray(state.output_cursor)),
                }
            )
            return CompositionEntry(
                state,
                entry_id="finite-volume/state",
                role="physical-state",
                owner_id="phydrax.solver.PreparedFiniteVolumeRuntime",
                structure_id=content.geometry_layout_id,
                revision_id=revision_id,
                semantics_id=target_runtime.dynamics.system.system_id,
                dependencies=(route.binding("revision"),),
            )

        def remap_dependencies(
            source_entry: CompositionEntry,
            target_entry: CompositionEntry,
            target_entries: dict[str, CompositionEntry],
            /,
        ) -> CompositionEntry:
            dependencies = []
            for dependency in source_entry.dependencies:
                target = target_entries.get(dependency.entry_id)
                if target is None:
                    raise ValueError(
                        f"Cell-state entry {source_entry.entry_id!r} has an "
                        f"unresolved dependency {dependency.entry_id!r}."
                    )
                dependencies.append(target.binding(dependency.facet))
            return CompositionEntry(
                target_entry.value,
                entry_id=target_entry.entry_id,
                role=target_entry.role,
                owner_id=target_entry.owner_id,
                structure_id=target_entry.structure_id,
                revision_id=target_entry.revision_id,
                semantics_id=target_entry.semantics_id,
                dependencies=tuple(dependencies),
            )

        commit_holder: dict[str, Any] = {}

        def commit_rebind(
            _prepared_event: Any,
            transfer: _OwnerLocalRepartitionTransfer,
            committed_journal: FiniteVolumeTopologyEventJournal,
            /,
        ) -> bool:
            local_error: Exception | None = None
            receipt = None
            next_state = None
            try:
                next_state = FiniteVolumeRuntimeState(
                    transfer.content_state,
                    committed_journal,
                    accepted_state.step_size,
                    accepted_step=accepted_state.accepted_step,
                    last_status=accepted_state.last_status,
                    controller_state=accepted_state.controller_state,
                    integrator_state=accepted_state.integrator_state,
                    output_cursor=accepted_state.output_cursor,
                    sliding_coupling=target_runtime.sliding_initial_coupling,
                    sliding_shift=accepted_state.sliding_shift,
                    sliding_event_id=accepted_state.sliding_event_id,
                )
                target_runtime._validate_sliding_state(next_state)
                target_state = target_state_entry(
                    next_state, candidate_route, target_runtime
                )
                target_entry_map = {
                    entry.entry_id: entry
                    for entry in candidate_connectivity.composition_entries()
                }
                target_entry_map.update(
                    {entry.entry_id: entry for entry in prepared_entries}
                )
                target_entry_map.update(
                    {entry.entry_id: entry for entry in retained_entries}
                )
                target_entry_map["finite-volume/state"] = target_state
                provisional_targets: dict[str, CompositionEntry] = {}
                external_transports = []
                transfers = transfer_holder["rows"]
                for entry_id in sorted(external_specs):
                    spec = external_specs[entry_id]
                    source_entry = external_by_id[entry_id]
                    bank_transfer = transfers[f"entry:{entry_id}"]
                    axis = spec.cell_axis
                    values = jnp.moveaxis(bank_transfer.target_values, 0, axis)
                    revision_id = canonical_fingerprint(
                        {
                            "kind": "finite-volume-owner-local-cell-state",
                            "entry": entry_id,
                            "source_revision": source_entry.revision_id,
                            "target_partition": candidate_epoch.partition_id,
                            "graph_plan": graph_migration.result.plan_id,
                        }
                    )
                    provisional_targets[entry_id] = CompositionEntry(
                        values,
                        entry_id=entry_id,
                        role=source_entry.role,
                        owner_id=source_entry.owner_id,
                        structure_id=target_view.geometry_layout_id,
                        revision_id=revision_id,
                        semantics_id=source_entry.semantics_id,
                    )
                target_entry_map.update(provisional_targets)
                for entry_id in sorted(external_specs):
                    source_entry = external_by_id[entry_id]
                    target_entry = remap_dependencies(
                        source_entry,
                        provisional_targets[entry_id],
                        target_entry_map,
                    )
                    bank_transfer = transfers[f"entry:{entry_id}"]
                    if (
                        bank_transfer.source_content_root
                        != bank_transfer.target_content_root
                    ):
                        raise ValueError(
                            f"Cell-state bank {entry_id!r} failed exact content preservation."
                        )
                    external_transports.append(
                        CompositionTransport(
                            "ownership-migration",
                            (entry_id,),
                            (target_entry,),
                            source_structure_ids=(source_entry.structure_id,),
                            route_id=canonical_fingerprint(
                                {
                                    "kind": "finite-volume-owner-cell-state-transfer",
                                    "transfer": bank_transfer.transfer_id,
                                }
                            ),
                            successful=bank_transfer.successful,
                            source_content_root=bank_transfer.source_content_root,
                            target_content_root=bank_transfer.target_content_root,
                        )
                    )
                source_composition_value = Composition(
                    (
                        *(
                            entry
                            for entry in source_composition.entries
                            if entry.entry_id != "finite-volume/state"
                        ),
                        source_state_entry,
                    ),
                    boundary_id=source_composition.boundary_id,
                )
                state_transport = CompositionTransport(
                    "ownership-migration",
                    ("finite-volume/state",),
                    (target_state,),
                    source_structure_ids=(source_state_entry.structure_id,),
                    route_id=canonical_fingerprint(
                        {
                            "kind": "finite-volume-owner-runtime-state-migration",
                            "migration": graph_migration.evidence_id,
                            "route": transfer.route_identity_id,
                            "bank_routes": transfer.bank_route_evidence,
                            "target_state": target_state.record_id,
                            "source_sliding_coupling": source_sliding_coupling_id,
                            "target_sliding_coupling": target_sliding_coupling_id,
                        }
                    ),
                    successful=jnp.asarray(True),
                    source_content_root=transfer.source_content_root,
                    target_content_root=transfer.target_content_root,
                )
                prepared_rebind = prepare_overset_motion_rebind(
                    source_composition_value,
                    previous_connectivity,
                    candidate_connectivity,
                    transports=(state_transport, *external_transports),
                    reprepare=prepared_entries,
                    retain=retained_ids,
                    invalidate=invalidated_ids,
                )
                receipt = prepared_rebind.commit(accepted_boundary=True)
                if (
                    not isinstance(receipt, CompositionRebindReceipt)
                    or not receipt.published
                ):
                    raise ValueError("Owner-local FV state rebind did not publish.")
            except Exception as error:
                local_error = error
            error_text = collective_error(
                local_error, f"{self.owner_local_axis_name}_repartition_rebind"
            )
            if error_text is not None or next_state is None or receipt is None:
                commit_holder["refusal"] = (
                    error_text or "Owner-local FV state rebind returned no state."
                )
                return False
            commit_holder["state"] = next_state
            commit_holder["receipt"] = receipt
            return True

        transaction: Any = None
        transaction_error: Exception | None = None
        try:
            scheduler = FiniteVolumeTopologyEventScheduler(
                accepted_state.topology_journal
            )
            scheduler.submit(request, accepted_step, event_time)
            transaction = scheduler.transact(
                accepted=True,
                source_content=source_content,
                candidate_epoch=candidate_epoch,
                candidate_artifacts=candidate_artifacts,
                remap=graph_migration,
                metrics=graph_migration,
                evidence=graph_migration,
                status=TopologyEventStatus.SUCCESS,
                active_cell_mask=target_runtime.active_cell_mask,
                transfer=transfer_cells,
                admissibility=True,
                result_id=candidate_epoch.epoch_id,
                payload_ids=(route_identity_id,),
                commit_action=commit_rebind,
                migration_partition_ids=(
                    (source_view.distribution_id, target_view.distribution_id)
                    if isinstance(
                        self.owner_local_geometry, OwnerLocalCompositeCellGeometry
                    )
                    else None
                ),
            )
        except Exception as error:
            transaction_error = error
        transaction_error_text = collective_error(
            transaction_error, f"{self.owner_local_axis_name}_repartition_transaction"
        )
        if transaction_error_text is not None or transaction is None:
            return fail(
                TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                transaction_error_text
                or "Owner-local repartition produced no transaction.",
            )
        commit_error = (
            None
            if transaction.committed and transaction.events
            else ValueError("Owner-local FV graph repartition did not commit.")
        )
        commit_error_text = collective_error(
            commit_error, f"{self.owner_local_axis_name}_repartition_commit"
        )
        if commit_error_text is not None:
            failure = transaction.failure
            return rollback_result(
                transaction.journal,
                source_content,
                transaction.event_status,
                "Owner-local FV graph repartition did not commit ("
                + ("no failure status" if failure is None else failure.name)
                + "): "
                + str(
                    transfer_holder.get("refusal")
                    or commit_holder.get("refusal")
                    or commit_error_text
                ),
            )
        next_state = commit_holder.get("state")
        receipt = commit_holder.get("receipt")
        receipt_error = (
            None
            if isinstance(next_state, FiniteVolumeRuntimeState)
            and isinstance(receipt, CompositionRebindReceipt)
            and receipt.published
            else ValueError("Owner-local FV repartition has no committed target state.")
        )
        receipt_error_text = collective_error(
            receipt_error, f"{self.owner_local_axis_name}_repartition_receipt"
        )
        if receipt_error_text is not None:
            return rollback_result(
                transaction.journal,
                source_content,
                transaction.event_status,
                receipt_error_text,
            )
        if (
            not isinstance(next_state, FiniteVolumeRuntimeState)
            or not isinstance(receipt, CompositionRebindReceipt)
            or not isinstance(target_runtime, PreparedFiniteVolumeRuntime)
        ):
            return rollback_result(
                transaction.journal,
                source_content,
                TopologyEventStatus.FAILED_MISSING_ARTIFACT,
                "Owner-local FV repartition lost its typed committed outputs.",
            )
        return FiniteVolumeOwnerLocalRepartitionResult(
            next_state,
            jnp.asarray(True),
            jnp.asarray(int(TopologyEventStatus.SUCCESS), dtype=jnp.int32),
            successor_runtime=target_runtime,
            repartition_receipt=receipt,
        )

    @checked
    def advance(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        args: Any = None,
        /,
    ) -> FiniteVolumeAdvanceResult:
        self._validate_state(runtime_state)
        content_state = runtime_state.content_state
        average = self._dynamics_cell_average(content_state)
        stage_average = self._provide_stage_state(content_state.time, average)
        if (
            isinstance(self.dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            and self.dynamics.coupling.overset is not None
            and self.embedded_redistribution is None
            and self.dynamics.coupling.motion is None
        ):
            self.precision.validate_state(stage_average)
            valid = self._active_state_is_admissible(
                stage_average,
                content_state.active_cell_mask,
            )
            if self.owner_local_decision_provider is not None:
                valid = self.owner_local_decision_provider(valid)
            return self._refresh_sliding_after_accept(
                runtime_state,
                self._advance_overset(runtime_state, average, valid, args),
                args,
            )
        if self.embedded_redistribution is not None:
            average = eqx.error_if(
                average,
                jnp.any(content_state.active_cell_mask != self.active_cell_mask)
                | jnp.any(
                    content_state.effective_cell_volumes != self._effective_cell_volumes()
                ),
                "Embedded runtime content must retain certified fluid ownership.",
            )
            self.precision.validate_state(stage_average)
            valid = self._active_state_is_admissible(
                stage_average,
                content_state.active_cell_mask,
            )
            return self._refresh_sliding_after_accept(
                runtime_state,
                self._advance_embedded(runtime_state, average, valid, args),
                args,
            )
        if (
            isinstance(self.dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            and self.dynamics.coupling.motion is not None
        ):
            average = eqx.error_if(
                average,
                ~jnp.all(content_state.active_cell_mask),
                "Fixed-topology ALE runtimes require every cell to remain active.",
            )
            self.precision.validate_state(stage_average)
            valid = jnp.all(self.dynamics.system.admissible(stage_average))
            if self.owner_local_decision_provider is not None:
                valid = self.owner_local_decision_provider(valid)
            return self._refresh_sliding_after_accept(
                runtime_state,
                self._advance_ale(runtime_state, average, valid, args),
                args,
            )
        average = eqx.error_if(
            average,
            ~jnp.all(content_state.active_cell_mask),
            "Static finite-volume runtimes require every cell to remain active.",
        )
        average = eqx.error_if(
            average,
            ~jnp.all(
                content_state.effective_cell_volumes == self._effective_cell_volumes()
            ),
            "Runtime effective cell volumes do not match the fixed geometry.",
        )
        self.precision.validate_state(stage_average)
        valid = jnp.all(self.dynamics.system.admissible(stage_average))
        if self.owner_local_decision_provider is not None:
            valid = self.owner_local_decision_provider(valid)

        def valid_branch(_: None) -> FiniteVolumeAdvanceResult:
            return self._advance_valid(runtime_state, average, args)

        def invalid_branch(_: None) -> FiniteVolumeAdvanceResult:
            state = FiniteVolumeRuntimeState(
                content_state,
                runtime_state.topology_journal,
                self.precision.decision(runtime_state.step_size),
                accepted_step=runtime_state.accepted_step,
                last_status=int(FiniteVolumeRunStatus.INVALID_INITIAL_STATE),
                controller_state=runtime_state.controller_state,
                integrator_state=runtime_state.integrator_state,
                output_cursor=runtime_state.output_cursor,
                sliding_coupling=runtime_state.sliding_coupling,
                sliding_shift=runtime_state.sliding_shift,
                sliding_event_id=runtime_state.sliding_event_id,
            )
            report = FiniteVolumeAdmissibilityReport(
                high_order_valid=jnp.asarray(False),
                fallback_valid=jnp.asarray(False),
                blend_factor=jnp.asarray(
                    0.0,
                    dtype=jnp.dtype(self.precision.reduction_dtype),
                ),
                activated=jnp.asarray(False),
                minimum_density=self.precision.decision(
                    _minimum_or_infinity(
                        stage_average[..., 0], self.precision.reduction_dtype
                    )
                ),
                limited_state_valid=jnp.asarray(False),
                secondary_reduction_applied=jnp.asarray(False),
                secondary_reduction_factor=jnp.asarray(
                    0.0,
                    dtype=jnp.dtype(self.precision.reduction_dtype),
                ),
            )
            return FiniteVolumeAdvanceResult(
                runtime_state=state,
                accepted=jnp.asarray(False),
                retries=jnp.asarray(0, dtype=jnp.int32),
                attempted_step_size=self.precision.decision(runtime_state.step_size),
                accepted_step_size=jnp.asarray(
                    0.0, dtype=jnp.dtype(self.precision.reduction_dtype)
                ),
                positivity=report,
                accepted_flux_integrals=self._static_accepted_flux_integral_ledger(
                    content_state,
                    content_state,
                    self._zero_static_flux_rates(),
                    jnp.asarray(0.0, dtype=jnp.dtype(self.precision.reduction_dtype)),
                    self.precision.decision(runtime_state.time + runtime_state.step_size),
                    runtime_state.accepted_step,
                ),
                precision_evidence=self.precision.evidence(),
                ale=None,
                embedded=None,
                shallow_water_integrals=self._zero_shallow_water_integrals(),
            )

        if self.stage_flux_provider is None:
            selected = jax.lax.cond(valid, valid_branch, invalid_branch, operand=None)
        else:
            valid_result = valid_branch(None)
            invalid_result = eqx.tree_at(
                lambda result: result.stage_flux_trace,
                invalid_branch(None),
                valid_result.stage_flux_trace,
                is_leaf=lambda value: value is None,
            )
            selected = jax.lax.cond(
                valid,
                lambda _: valid_result,
                lambda _: invalid_result,
                operand=None,
            )
        return self._refresh_sliding_after_accept(
            runtime_state,
            selected,
            args,
        )

    def advance_prescribed(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> FiniteVolumeScheduledAdvanceResult:
        """Attempt exactly ``step_size`` without accepting a clamp or retry."""

        if not isinstance(self.dynamics, PreparedFiniteVolumeDynamics):
            raise ValueError(
                "Prescribed finite-volume replay currently requires stationary structured dynamics."
            )
        if self.embedded_redistribution is not None or self.sliding_plan is not None:
            raise ValueError(
                "Prescribed finite-volume replay does not support embedded or sliding topology."
            )
        requested = self.precision.decision(jnp.asarray(step_size).reshape(()))
        requested = eqx.error_if(
            requested,
            ~jnp.isfinite(requested) | (requested <= 0.0),
            "Prescribed finite-volume step_size must be positive and finite.",
        )
        replay_state = FiniteVolumeRuntimeState(
            runtime_state.content_state,
            runtime_state.topology_journal,
            requested,
            accepted_step=runtime_state.accepted_step,
            last_status=runtime_state.last_status,
            controller_state=runtime_state.controller_state,
            integrator_state=runtime_state.integrator_state,
            output_cursor=runtime_state.output_cursor,
            sliding_coupling=runtime_state.sliding_coupling,
            sliding_shift=runtime_state.sliding_shift,
            sliding_event_id=runtime_state.sliding_event_id,
        )
        stage_state = self._provide_stage_state(
            runtime_state.time,
            self._dynamics_cell_average(runtime_state.content_state),
        )
        stable = self.precision.decision(
            self.dynamics.stable_step(stage_state, args, cfl=self.policy.cfl)
        )
        attempted = self.advance(replay_state, args)
        tolerance = (
            32.0 * jnp.finfo(requested.dtype).eps * jnp.maximum(jnp.abs(requested), 1.0)
        )
        exact = (
            attempted.accepted
            & (attempted.retries == 0)
            & (jnp.abs(attempted.accepted_step_size - requested) <= tolerance)
        )
        status = jnp.where(
            exact,
            attempted.runtime_state.last_status,
            jnp.where(
                requested > stable + tolerance,
                int(FiniteVolumeRunStatus.STABILITY_LIMIT_EXCEEDED),
                int(FiniteVolumeRunStatus.PRESCRIBED_STEP_REJECTED),
            ),
        )
        content = jax.lax.cond(
            exact,
            lambda _: attempted.runtime_state.content_state,
            lambda _: runtime_state.content_state,
            operand=None,
        )
        final_state = FiniteVolumeRuntimeState(
            content,
            runtime_state.topology_journal,
            self.precision.decision(jnp.where(exact, requested, runtime_state.step_size)),
            accepted_step=jnp.where(
                exact,
                attempted.runtime_state.accepted_step,
                runtime_state.accepted_step,
            ),
            last_status=status,
            controller_state=runtime_state.controller_state,
            integrator_state=runtime_state.integrator_state,
            output_cursor=runtime_state.output_cursor,
            sliding_coupling=runtime_state.sliding_coupling,
            sliding_shift=runtime_state.sliding_shift,
            sliding_event_id=runtime_state.sliding_event_id,
        )
        margin = self.precision.decision(stable / requested - 1.0)
        return FiniteVolumeScheduledAdvanceResult(
            runtime_state=final_state,
            attempted=attempted,
            accepted=exact,
            requested_step_size=requested,
            stable_step_size=stable,
            stability_margin=margin,
        )

    def _advance_overset(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        original_average: Array,
        initial_valid: Array,
        args: Any,
        /,
    ) -> FiniteVolumeAdvanceResult:
        """Advance a stationary certified overset map with frozen stage routes."""
        dynamics = self.dynamics
        fallback = self.fallback_dynamics
        if not isinstance(
            dynamics, PreparedUnstructuredFiniteVolumeDynamics
        ) or not isinstance(fallback, PreparedUnstructuredFiniteVolumeDynamics):
            raise TypeError("Overset advancement requires unstructured dynamics.")

        original_content = runtime_state.content_state
        attempted = self.precision.decision(runtime_state.step_size)
        zero = jnp.asarray(0.0, dtype=jnp.dtype(self.precision.reduction_dtype))
        active_density = jnp.where(
            original_content.active_cell_mask,
            original_average.reshape((-1, original_average.shape[-1]))[..., 0],
            jnp.asarray(jnp.inf, dtype=original_average.dtype),
        )
        last_report = FiniteVolumeAdmissibilityReport(
            high_order_valid=jnp.asarray(False),
            fallback_valid=jnp.asarray(False),
            blend_factor=zero,
            activated=jnp.asarray(False),
            minimum_density=self.precision.decision(
                _minimum_or_infinity(active_density, self.precision.reduction_dtype)
            ),
            limited_state_valid=jnp.asarray(False),
            secondary_reduction_applied=jnp.asarray(False),
            secondary_reduction_factor=zero,
        )
        accepted = jnp.asarray(False)
        accepted_content = original_content
        accepted_dt = jnp.asarray(0.0, dtype=attempted.dtype)
        retries = jnp.asarray(0, dtype=jnp.int32)
        current_dt = attempted
        accepted_flux_ledger = None
        accepted_stage_ledgers = None

        def stage_at(stage_time: Array, version: Array) -> FiniteVolumeStageMetrics:
            base = lower_static_unstructured_stage_metrics(
                dynamics.discretization,
                time=self.precision.decision(stage_time),
                topology_epoch_id=original_content.topology_epoch_id,
            )
            face_blocks = tuple(
                eqx.tree_at(
                    lambda block: block.layout.active_mask,
                    block,
                    block.layout.active_mask
                    & self.active_cell_mask[block.layout.owner_cells]
                    & (
                        (block.layout.neighbor_cells < 0)
                        | self.active_cell_mask[
                            jnp.maximum(block.layout.neighbor_cells, 0)
                        ]
                    ),
                )
                for block in base.face_blocks
            )
            # ownership through the pytree avoids re-running constructor checks
            # while preserving the certified route identity for JIT stages.
            return eqx.tree_at(
                lambda stage: (
                    stage.geometry_version,
                    stage.time,
                    stage.effective_cell_volumes,
                    stage.coordinate_effective_cell_volumes,
                    stage.mesh_volume_rate,
                    stage.active_cell_mask,
                    stage.face_blocks,
                ),
                base,
                (
                    version,
                    self.precision.decision(stage_time),
                    self.effective_cell_volumes,
                    self.effective_cell_volumes,
                    jnp.zeros_like(self.effective_cell_volumes),
                    self.active_cell_mask,
                    face_blocks,
                ),
            )

        def select_tree(condition: Array, new: _TreeT, old: _TreeT, /) -> _TreeT:
            return jax.tree.map(
                lambda new_value, old_value: jnp.where(condition, new_value, old_value),
                new,
                old,
            )

        for retry in range(self.policy.maximum_retries + 1):
            stage_metrics = (
                stage_at(runtime_state.time, original_content.geometry_version),
                stage_at(
                    runtime_state.time + current_dt,
                    original_content.geometry_version + 1,
                ),
                stage_at(
                    runtime_state.time + 0.5 * current_dt,
                    original_content.geometry_version + 2,
                ),
            )
            zero_candidate = zero_unstructured_ssprk33_content_candidate(
                dynamics,
                original_content,
                stage_metrics,
                stage_metrics[1],
                current_dt,
                runtime_state.accepted_step,
                stage_state_provider=self.stage_state_provider,
                stage_decision_provider=self.owner_local_decision_provider,
                stage_route_provider=self.owner_local_route_provider,
            )
            candidate = collective_branch(
                initial_valid,
                lambda _: unstructured_ssprk33_content_candidate(
                    dynamics,
                    fallback,
                    self.positivity,
                    original_content,
                    stage_metrics,
                    stage_metrics[1],
                    current_dt,
                    runtime_state.accepted_step + jnp.asarray(1, dtype=jnp.int32),
                    args,
                    stage_state_provider=self.stage_state_provider,
                    stage_decision_provider=self.owner_local_decision_provider,
                    stage_route_provider=self.owner_local_route_provider,
                    cfl=self.policy.cfl,
                ),
                lambda _: zero_candidate,
            )
            if retry == 0:
                accepted_flux_ledger = zero_candidate.accepted_flux_integrals
                accepted_stage_ledgers = zero_candidate.stage_rate_ledgers
            record = ~accepted
            last_report = select_tree(record, candidate.positivity, last_report)
            accepted_flux_ledger = select_tree(
                record, zero_candidate.accepted_flux_integrals, accepted_flux_ledger
            )
            accepted_stage_ledgers = select_tree(
                record, candidate.stage_rate_ledgers, accepted_stage_ledgers
            )
            finite = jnp.all(jnp.isfinite(candidate.content_state.conservative_content))
            cfl_valid = current_dt <= candidate.relative_cfl_step
            candidate_valid = (
                initial_valid
                & finite
                & cfl_valid
                & candidate.positivity.fallback_valid
                & candidate.positivity.limited_state_valid
            )
            if self.owner_local_decision_provider is not None:
                candidate_valid = self.owner_local_decision_provider(candidate_valid)
            take = (~accepted) & candidate_valid
            accepted_content = select_tree(
                take, candidate.content_state, accepted_content
            )
            accepted_flux_ledger = select_tree(
                take, candidate.accepted_flux_integrals, accepted_flux_ledger
            )
            accepted_stage_ledgers = select_tree(
                take, candidate.stage_rate_ledgers, accepted_stage_ledgers
            )
            accepted_dt = jnp.where(take, current_dt, accepted_dt)
            retries = jnp.where(take, retry, retries)
            accepted = accepted | take
            current_dt = self.precision.decision(
                current_dt * self.policy.reduction_factor
            )

        if accepted_flux_ledger is None or accepted_stage_ledgers is None:
            raise RuntimeError("Overset retry loop did not form its evidence envelope.")
        minimum_reached = current_dt < self.precision.decision(
            self.policy.minimum_step_size
        )
        status = jnp.where(
            ~initial_valid,
            int(FiniteVolumeRunStatus.INVALID_INITIAL_STATE),
            jnp.where(
                accepted & (retries > 0),
                int(FiniteVolumeRunStatus.RECOVERED_REJECTION),
                jnp.where(
                    accepted,
                    int(FiniteVolumeRunStatus.SUCCESS),
                    jnp.where(
                        minimum_reached,
                        int(FiniteVolumeRunStatus.MINIMUM_STEP_REACHED),
                        int(FiniteVolumeRunStatus.RETRY_LIMIT_REACHED),
                    ),
                ),
            ),
        )
        next_content = jax.lax.cond(
            accepted,
            lambda _: accepted_content,
            lambda _: original_content,
            operand=None,
        )
        if self.owner_local_decision_provider is not None:
            refreshed_average = self._provide_stage_state(
                next_content.time,
                next_content.cell_average(),
            )
            refreshed_content = (
                refreshed_average
                * next_content.effective_cell_volumes.reshape(
                    (-1,) + (1,) * (refreshed_average.ndim - 1)
                )
            )
            next_content = next_content.with_content(refreshed_content)
        next_state = FiniteVolumeRuntimeState(
            next_content,
            runtime_state.topology_journal,
            self.precision.decision(jnp.where(accepted, accepted_dt, attempted)),
            accepted_step=runtime_state.accepted_step + accepted.astype(jnp.int32),
            last_status=status,
            controller_state=runtime_state.controller_state,
            integrator_state=runtime_state.integrator_state,
            output_cursor=runtime_state.output_cursor,
            sliding_coupling=runtime_state.sliding_coupling,
            sliding_shift=runtime_state.sliding_shift,
            sliding_event_id=runtime_state.sliding_event_id,
        )
        return FiniteVolumeAdvanceResult(
            runtime_state=next_state,
            accepted=accepted,
            retries=retries,
            attempted_step_size=attempted,
            accepted_step_size=accepted_dt,
            positivity=last_report,
            accepted_flux_integrals=accepted_flux_ledger,
            precision_evidence=self.precision.evidence(),
            ale=None,
            embedded=None,
        )

    def _advance_embedded(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        original_average: Array,
        initial_valid: Array,
        args: Any,
        /,
    ) -> FiniteVolumeAdvanceResult:
        dynamics = self.dynamics
        fallback = self.fallback_dynamics
        redistribution = self.embedded_redistribution
        if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
            raise TypeError("Embedded advancement requires unstructured dynamics.")
        if not isinstance(fallback, PreparedUnstructuredFiniteVolumeDynamics):
            raise TypeError(
                "Embedded fallback advancement requires unstructured dynamics."
            )
        if not isinstance(
            redistribution,
            ConservativeSmallCellRedistributionPlan,
        ):
            raise TypeError("Embedded advancement requires prepared redistribution.")
        embedded_metrics = dynamics.coupling.embedded_metrics
        embedded_boundaries = dynamics.coupling.embedded_boundaries
        if embedded_metrics is None or not isinstance(
            embedded_boundaries,
            UnstructuredEmbeddedBoundarySet,
        ):
            raise ValueError("Embedded advancement requires certified stationary cuts.")
        embedded_stage_template = self.embedded_stage_template
        if not isinstance(embedded_stage_template, FiniteVolumeStageMetrics):
            raise TypeError("Embedded advancement requires a prepared stage template.")

        original_content = runtime_state.content_state

        def stage_at(stage_time: Array) -> FiniteVolumeStageMetrics:
            return eqx.tree_at(
                lambda stage: (
                    stage.time,
                    stage.geometry_version,
                    stage.evidence.evidence_version,
                ),
                embedded_stage_template,
                (
                    self.precision.decision(stage_time),
                    original_content.geometry_version,
                    original_content.evidence_version,
                ),
            )

        attempted = self.precision.decision(runtime_state.step_size)
        accepted = jnp.asarray(False)
        accepted_content = original_content
        accepted_dt = jnp.asarray(0.0, dtype=attempted.dtype)
        retries = jnp.asarray(0, dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=jnp.dtype(self.precision.reduction_dtype))
        active_density = jnp.where(
            original_content.active_cell_mask,
            original_average.reshape((-1, original_average.shape[-1]))[..., 0],
            jnp.asarray(jnp.inf, dtype=original_average.dtype),
        )
        last_report = FiniteVolumeAdmissibilityReport(
            high_order_valid=jnp.asarray(False),
            fallback_valid=jnp.asarray(False),
            blend_factor=zero,
            activated=jnp.asarray(False),
            minimum_density=self.precision.decision(
                _minimum_or_infinity(active_density, self.precision.reduction_dtype)
            ),
            limited_state_valid=jnp.asarray(False),
            secondary_reduction_applied=jnp.asarray(False),
            secondary_reduction_factor=zero,
        )
        current_dt = self.precision.decision(attempted)
        accepted_flux_ledger = None
        accepted_stage_ledgers = None
        recorded_stage_metrics = None
        recorded_stage_rates: tuple[Array, Array, Array] = (
            zero,
            zero,
            zero,
        )
        recorded_maximum_rate = zero
        recorded_cfl_step = jnp.asarray(jnp.inf, dtype=zero.dtype)

        def select_tree(condition: Array, new: _TreeT, old: _TreeT, /) -> _TreeT:
            return jax.tree.map(
                lambda new_value, old_value: jnp.where(
                    condition,
                    new_value,
                    old_value,
                ),
                new,
                old,
            )

        for retry in range(self.policy.maximum_retries + 1):
            stage_1 = stage_at(runtime_state.time)
            stage_2 = stage_at(self.precision.decision(runtime_state.time + current_dt))
            stage_3 = stage_at(
                self.precision.decision(runtime_state.time + 0.5 * current_dt)
            )
            stage_metrics = (stage_1, stage_2, stage_3)
            zero_candidate = zero_unstructured_ssprk33_content_candidate(
                dynamics,
                original_content,
                stage_metrics,
                stage_2,
                current_dt,
                runtime_state.accepted_step,
                stage_state_provider=self.stage_state_provider,
                redistribution=redistribution,
                stage_decision_provider=self.owner_local_decision_provider,
                stage_route_provider=self.owner_local_route_provider,
            )
            metrics_success = jnp.all(
                jnp.stack(
                    tuple(
                        stage.evidence.passed
                        & (
                            stage.evidence.status
                            == int(FiniteVolumeGeometryStatus.SUCCESS)
                        )
                        for stage in stage_metrics
                    )
                )
            )
            if self.owner_local_decision_provider is not None:
                metrics_success = self.owner_local_decision_provider(metrics_success)
            precheck_state = (
                original_content
                if self.stage_state_provider is None
                else original_content.with_content(
                    self.stage_state_provider(
                        stage_1.time, original_content.cell_average()
                    )
                    * stage_1.effective_cell_volumes.reshape(
                        (-1,) + (1,) * (original_content.conservative_content.ndim - 1)
                    )
                )
            )
            precheck = dynamics.evaluate_stage(
                precheck_state,
                stage_1,
                args,
                cfl=self.policy.cfl,
                redistribution=redistribution,
            )
            precheck_cfl_valid = jnp.isinf(precheck.relative_cfl_step) | (
                current_dt <= precheck.relative_cfl_step
            )
            if self.owner_local_decision_provider is not None:
                precheck_cfl_valid = self.owner_local_decision_provider(
                    precheck_cfl_valid
                )
            evaluate_physics = initial_valid & metrics_success & precheck_cfl_valid
            if self.owner_local_decision_provider is not None:
                evaluate_physics = self.owner_local_decision_provider(evaluate_physics)
            candidate = collective_branch(
                evaluate_physics,
                lambda _: unstructured_ssprk33_content_candidate(
                    dynamics,
                    fallback,
                    self.positivity,
                    original_content,
                    stage_metrics,
                    stage_2,
                    current_dt,
                    runtime_state.accepted_step + jnp.asarray(1, dtype=jnp.int32),
                    args,
                    cfl=self.policy.cfl,
                    redistribution=redistribution,
                    stage_state_provider=self.stage_state_provider,
                    stage_decision_provider=self.owner_local_decision_provider,
                    stage_route_provider=self.owner_local_route_provider,
                ),
                lambda _: zero_candidate,
            )
            if retry == 0:
                accepted_flux_ledger = zero_candidate.accepted_flux_integrals
                accepted_stage_ledgers = zero_candidate.stage_rate_ledgers
                recorded_stage_metrics = stage_metrics
                recorded_stage_rates = candidate.stage_maximum_relative_rates
            record_attempt = ~accepted
            recorded_stage_metrics = select_tree(
                record_attempt,
                stage_metrics,
                recorded_stage_metrics,
            )
            new_stage_rates = candidate.stage_maximum_relative_rates
            recorded_stage_rates = (
                jnp.where(record_attempt, new_stage_rates[0], recorded_stage_rates[0]),
                jnp.where(record_attempt, new_stage_rates[1], recorded_stage_rates[1]),
                jnp.where(record_attempt, new_stage_rates[2], recorded_stage_rates[2]),
            )
            recorded_maximum_rate = jnp.where(
                record_attempt,
                candidate.maximum_relative_rate,
                recorded_maximum_rate,
            )
            recorded_cfl_step = jnp.where(
                record_attempt,
                candidate.relative_cfl_step,
                recorded_cfl_step,
            )
            last_report = select_tree(
                record_attempt,
                self._precision_report(candidate.positivity),
                last_report,
            )
            accepted_flux_ledger = select_tree(
                record_attempt,
                zero_candidate.accepted_flux_integrals,
                accepted_flux_ledger,
            )
            accepted_stage_ledgers = select_tree(
                record_attempt,
                candidate.stage_rate_ledgers,
                accepted_stage_ledgers,
            )

            finite = jnp.all(jnp.isfinite(candidate.content_state.conservative_content))
            cfl_valid = current_dt <= candidate.relative_cfl_step
            candidate_valid = (
                evaluate_physics
                & finite
                & cfl_valid
                & candidate.positivity.fallback_valid
                & candidate.positivity.limited_state_valid
            )
            if self.owner_local_decision_provider is not None:
                candidate_valid = self.owner_local_decision_provider(candidate_valid)
            take = (~accepted) & candidate_valid
            accepted_content = select_tree(
                take,
                candidate.content_state,
                accepted_content,
            )
            accepted_flux_ledger = select_tree(
                take,
                candidate.accepted_flux_integrals,
                accepted_flux_ledger,
            )
            accepted_stage_ledgers = select_tree(
                take,
                candidate.stage_rate_ledgers,
                accepted_stage_ledgers,
            )
            accepted_dt = jnp.where(take, current_dt, accepted_dt)
            retries = jnp.where(take, retry, retries)
            accepted = accepted | take
            current_dt = self.precision.decision(
                current_dt * self.policy.reduction_factor
            )

        minimum_reached = current_dt < self.precision.decision(
            self.policy.minimum_step_size
        )
        status = jnp.where(
            ~initial_valid,
            int(FiniteVolumeRunStatus.INVALID_INITIAL_STATE),
            jnp.where(
                accepted & (retries > 0),
                int(FiniteVolumeRunStatus.RECOVERED_REJECTION),
                jnp.where(
                    accepted,
                    int(FiniteVolumeRunStatus.SUCCESS),
                    jnp.where(
                        minimum_reached,
                        int(FiniteVolumeRunStatus.MINIMUM_STEP_REACHED),
                        int(FiniteVolumeRunStatus.RETRY_LIMIT_REACHED),
                    ),
                ),
            ),
        )
        next_content = jax.lax.cond(
            accepted,
            lambda _: accepted_content,
            lambda _: original_content,
            operand=None,
        )
        next_state = FiniteVolumeRuntimeState(
            next_content,
            runtime_state.topology_journal,
            self.precision.decision(
                jnp.where(accepted, accepted_dt, runtime_state.step_size)
            ),
            accepted_step=(runtime_state.accepted_step + accepted.astype(jnp.int32)),
            last_status=status,
            controller_state=runtime_state.controller_state,
            integrator_state=runtime_state.integrator_state,
            output_cursor=runtime_state.output_cursor,
            sliding_coupling=runtime_state.sliding_coupling,
            sliding_shift=runtime_state.sliding_shift,
            sliding_event_id=runtime_state.sliding_event_id,
        )
        if (
            accepted_flux_ledger is None
            or accepted_stage_ledgers is None
            or recorded_stage_metrics is None
            or recorded_stage_rates is None
        ):
            raise RuntimeError("Embedded retry loop did not form its evidence envelope.")
        return FiniteVolumeAdvanceResult(
            runtime_state=next_state,
            accepted=accepted,
            retries=retries,
            attempted_step_size=attempted,
            accepted_step_size=accepted_dt,
            positivity=last_report,
            accepted_flux_integrals=accepted_flux_ledger,
            precision_evidence=self.precision.evidence(),
            ale=None,
            embedded=FiniteVolumeEmbeddedAdvanceEvidence(
                accepted=accepted,
                stage_rate_ledgers=accepted_stage_ledgers,
                stage_metrics=recorded_stage_metrics,
                accepted_metrics=recorded_stage_metrics[1],
                redistribution=redistribution.report,
                stage_maximum_relative_rates=recorded_stage_rates,
                maximum_relative_rate=recorded_maximum_rate,
                relative_cfl_step=recorded_cfl_step,
            ),
        )

    def _advance_ale(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        original_average: Array,
        initial_valid: Array,
        args: Any,
        /,
    ) -> FiniteVolumeAdvanceResult:
        dynamics = self.dynamics
        fallback = self.fallback_dynamics
        if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
            raise TypeError("ALE advancement requires unstructured dynamics.")
        if not isinstance(fallback, PreparedUnstructuredFiniteVolumeDynamics):
            raise TypeError("ALE fallback advancement requires unstructured dynamics.")
        motion = dynamics.coupling.motion
        if motion is None:
            raise ValueError(
                "ALE advancement requires prepared fixed-connectivity motion."
            )

        original_content = runtime_state.content_state
        attempted = self.precision.decision(runtime_state.step_size)
        accepted = jnp.asarray(False)
        accepted_content = original_content
        accepted_dt = jnp.asarray(0.0, dtype=attempted.dtype)
        retries = jnp.asarray(0, dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=jnp.dtype(self.precision.reduction_dtype))
        last_report = FiniteVolumeAdmissibilityReport(
            high_order_valid=jnp.asarray(False),
            fallback_valid=jnp.asarray(False),
            blend_factor=zero,
            activated=jnp.asarray(False),
            minimum_density=self.precision.decision(
                _minimum_or_infinity(
                    original_average[..., 0], self.precision.reduction_dtype
                )
            ),
            limited_state_valid=jnp.asarray(False),
            secondary_reduction_applied=jnp.asarray(False),
            secondary_reduction_factor=zero,
        )
        current_dt = self.precision.decision(attempted)
        accepted_flux_ledger = None
        accepted_stage_ledgers = None
        recorded_geometry = None
        recorded_maximum_rate = zero
        recorded_cfl_step = jnp.asarray(jnp.inf, dtype=zero.dtype)
        recorded_geometry_reduction = jnp.asarray(1.0, dtype=zero.dtype)
        minimum_geometry_reduction = jnp.asarray(1.0, dtype=zero.dtype)

        def select_tree(condition: Array, new: _TreeT, old: _TreeT, /) -> _TreeT:
            return jax.tree.map(
                lambda new_value, old_value: jnp.where(
                    condition,
                    new_value,
                    old_value,
                ),
                new,
                old,
            )

        for retry in range(self.policy.maximum_retries + 1):
            geometry = motion.prepare_ssprk33_step(
                runtime_state.time,
                current_dt,
                original_content.topology_epoch_id,
                original_content.geometry_version,
                original_content.evidence_version,
                args,
                prior_effective_cell_volumes=(original_content.effective_cell_volumes),
            )
            zero_candidate = zero_unstructured_ale_ssprk33_candidate(
                dynamics,
                original_content,
                geometry,
                current_dt,
                runtime_state.accepted_step,
                stage_state_provider=self.stage_state_provider,
                stage_decision_provider=self.owner_local_decision_provider,
                stage_route_provider=self.owner_local_route_provider,
            )
            geometry_success = geometry.passed & (
                geometry.status == int(FiniteVolumeGeometryStatus.SUCCESS)
            )
            if self.owner_local_decision_provider is not None:
                geometry_success = self.owner_local_decision_provider(geometry_success)
            geometry_reduction = geometry.proposed_reduction_factor
            if self.owner_local_scalar_min_provider is not None:
                geometry_reduction = self.owner_local_scalar_min_provider(
                    geometry_reduction
                )
            evaluate_physics = initial_valid & geometry_success
            candidate = jax.lax.cond(
                evaluate_physics,
                lambda _: unstructured_ale_ssprk33_candidate(
                    dynamics,
                    fallback,
                    self.positivity,
                    original_content,
                    geometry,
                    current_dt,
                    runtime_state.accepted_step + jnp.asarray(1, dtype=jnp.int32),
                    args,
                    cfl=self.policy.cfl,
                    stage_state_provider=self.stage_state_provider,
                    stage_decision_provider=self.owner_local_decision_provider,
                    stage_route_provider=self.owner_local_route_provider,
                ),
                lambda _: zero_candidate,
                operand=None,
            )
            if retry == 0:
                accepted_flux_ledger = zero_candidate.accepted_flux_integrals
                recorded_geometry = geometry
                accepted_stage_ledgers = zero_candidate.stage_rate_ledgers
            record_attempt = ~accepted
            recorded_geometry = select_tree(
                record_attempt,
                geometry,
                recorded_geometry,
            )
            recorded_maximum_rate = jnp.where(
                record_attempt,
                candidate.maximum_relative_rate,
                recorded_maximum_rate,
            )
            recorded_cfl_step = jnp.where(
                record_attempt,
                candidate.relative_cfl_step,
                recorded_cfl_step,
            )
            recorded_geometry_reduction = jnp.where(
                record_attempt,
                geometry_reduction,
                recorded_geometry_reduction,
            )
            minimum_geometry_reduction = jnp.where(
                record_attempt,
                jnp.minimum(
                    minimum_geometry_reduction,
                    geometry_reduction,
                ),
                minimum_geometry_reduction,
            )
            last_report = select_tree(
                record_attempt,
                self._precision_report(candidate.positivity),
                last_report,
            )
            accepted_flux_ledger = select_tree(
                record_attempt,
                zero_candidate.accepted_flux_integrals,
                accepted_flux_ledger,
            )
            accepted_stage_ledgers = select_tree(
                record_attempt,
                candidate.stage_rate_ledgers,
                accepted_stage_ledgers,
            )

            finite = jnp.all(jnp.isfinite(candidate.content_state.conservative_content))
            cfl_valid = current_dt <= candidate.relative_cfl_step
            candidate_valid = (
                evaluate_physics
                & finite
                & cfl_valid
                & candidate.positivity.fallback_valid
                & candidate.positivity.limited_state_valid
            )
            if self.owner_local_decision_provider is not None:
                candidate_valid = self.owner_local_decision_provider(candidate_valid)
            take = (~accepted) & candidate_valid
            accepted_content = select_tree(
                take,
                candidate.content_state,
                accepted_content,
            )
            accepted_flux_ledger = select_tree(
                take,
                candidate.accepted_flux_integrals,
                accepted_flux_ledger,
            )
            accepted_stage_ledgers = select_tree(
                take,
                candidate.stage_rate_ledgers,
                accepted_stage_ledgers,
            )
            accepted_dt = jnp.where(take, current_dt, accepted_dt)
            retries = jnp.where(take, retry, retries)
            accepted = accepted | take
            retry_factor = jnp.where(
                geometry_success,
                jnp.asarray(self.policy.reduction_factor, dtype=current_dt.dtype),
                geometry_reduction.astype(current_dt.dtype),
            )
            current_dt = self.precision.decision(current_dt * retry_factor)

        minimum_reached = current_dt < self.precision.decision(
            self.policy.minimum_step_size
        )
        status = jnp.where(
            ~initial_valid,
            int(FiniteVolumeRunStatus.INVALID_INITIAL_STATE),
            jnp.where(
                accepted & (retries > 0),
                int(FiniteVolumeRunStatus.RECOVERED_REJECTION),
                jnp.where(
                    accepted,
                    int(FiniteVolumeRunStatus.SUCCESS),
                    jnp.where(
                        minimum_reached,
                        int(FiniteVolumeRunStatus.MINIMUM_STEP_REACHED),
                        int(FiniteVolumeRunStatus.RETRY_LIMIT_REACHED),
                    ),
                ),
            ),
        )
        next_content = jax.lax.cond(
            accepted,
            lambda _: accepted_content,
            lambda _: original_content,
            operand=None,
        )
        if self.owner_local_decision_provider is not None:
            refreshed_average = self._provide_stage_state(
                next_content.time,
                next_content.cell_average(),
            )
            refreshed_content = (
                refreshed_average
                * next_content.effective_cell_volumes.reshape(
                    (-1,) + (1,) * (refreshed_average.ndim - 1)
                )
            )
            next_content = next_content.with_content(refreshed_content)
        next_state = FiniteVolumeRuntimeState(
            next_content,
            runtime_state.topology_journal,
            self.precision.decision(
                jnp.where(accepted, accepted_dt, runtime_state.step_size)
            ),
            accepted_step=(runtime_state.accepted_step + accepted.astype(jnp.int32)),
            last_status=status,
            controller_state=runtime_state.controller_state,
            integrator_state=runtime_state.integrator_state,
            output_cursor=runtime_state.output_cursor,
            sliding_coupling=runtime_state.sliding_coupling,
            sliding_shift=runtime_state.sliding_shift,
            sliding_event_id=runtime_state.sliding_event_id,
        )
        if (
            accepted_flux_ledger is None
            or accepted_stage_ledgers is None
            or recorded_geometry is None
        ):
            raise RuntimeError("ALE retry loop did not form its evidence envelope.")
        return FiniteVolumeAdvanceResult(
            runtime_state=next_state,
            accepted=accepted,
            retries=retries,
            attempted_step_size=attempted,
            accepted_step_size=accepted_dt,
            positivity=last_report,
            accepted_flux_integrals=accepted_flux_ledger,
            precision_evidence=self.precision.evidence(),
            ale=FiniteVolumeALEAdvanceEvidence(
                accepted=accepted,
                geometry=recorded_geometry,
                maximum_relative_rate=recorded_maximum_rate,
                stage_rate_ledgers=accepted_stage_ledgers,
                relative_cfl_step=recorded_cfl_step,
                geometry_reduction_factor=jnp.where(
                    accepted,
                    minimum_geometry_reduction,
                    recorded_geometry_reduction,
                ),
            ),
            embedded=None,
        )

    def _advance_valid(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        original_average: Array,
        args: Any = None,
        /,
    ) -> FiniteVolumeAdvanceResult:
        initial_stage_average = self._provide_stage_state(
            runtime_state.time, original_average
        )
        self.precision.validate_state(initial_stage_average)
        stable = self.precision.decision(
            self.dynamics.stable_step(
                initial_stage_average,
                args,
                cfl=self.policy.cfl,
            )
        )
        attempted = self.precision.decision(
            jnp.minimum(self.precision.decision(runtime_state.step_size), stable)
        )
        accepted = jnp.asarray(False)
        accepted_average = original_average
        accepted_dt = jnp.asarray(0.0, dtype=attempted.dtype)
        retries = jnp.asarray(0, dtype=jnp.int32)
        last_report = FiniteVolumeAdmissibilityReport(
            high_order_valid=jnp.asarray(False),
            fallback_valid=jnp.asarray(False),
            blend_factor=jnp.asarray(
                0.0,
                dtype=jnp.dtype(self.precision.reduction_dtype),
            ),
            activated=jnp.asarray(False),
            minimum_density=self.precision.decision(
                _minimum_or_infinity(
                    initial_stage_average[..., 0], self.precision.reduction_dtype
                )
            ),
            limited_state_valid=jnp.asarray(False),
            secondary_reduction_applied=jnp.asarray(False),
            secondary_reduction_factor=jnp.asarray(
                0.0,
                dtype=jnp.dtype(self.precision.reduction_dtype),
            ),
        )
        accepted_flux_rates = self._zero_static_flux_rates()
        accepted_balanced_contributions: (
            tuple[ShallowWaterBalancedFaceResult, ...] | None
        ) = None
        accepted_stage_flux_trace: FiniteVolumeStageFluxTrace | None = None
        attempted_stage_flux_trace: FiniteVolumeStageFluxTrace | None = None
        current_dt = self.precision.decision(attempted)
        for retry in range(self.policy.maximum_retries + 1):
            candidate, stage_flux_trace = self._candidate_with_stage_flux_trace(
                runtime_state.time, original_average, current_dt, args
            )
            finite = jnp.all(jnp.isfinite(candidate.state))
            valid = (
                finite
                & candidate.report.fallback_valid
                & candidate.report.limited_state_valid
            )
            take = (~accepted) & valid
            if self.stage_flux_provider is not None:
                if stage_flux_trace is None:
                    raise RuntimeError(
                        "Bound stage-flux provider did not produce an SSPRK3 trace."
                    )
                attempted_stage_flux_trace = stage_flux_trace
                if accepted_stage_flux_trace is None:
                    accepted_stage_flux_trace = jax.tree.map(
                        jnp.zeros_like, stage_flux_trace
                    )
                accepted_stage_flux_trace = jax.tree.map(
                    lambda new, old: jnp.where(take, new, old),
                    stage_flux_trace,
                    accepted_stage_flux_trace,
                )
            accepted_average = jnp.where(
                take,
                self.precision.storage(candidate.state),
                accepted_average,
            )
            accepted_dt = jnp.where(
                take,
                self.precision.decision(current_dt),
                accepted_dt,
            )
            retries = jnp.where(take, retry, retries)
            last_report = jax.tree.map(
                lambda new, old: jnp.where(take, new, old),
                candidate.report,
                last_report,
            )
            accepted_flux_rates = tuple(
                jnp.where(
                    take,
                    self.precision.reduction(new),
                    old,
                )
                for new, old in zip(
                    candidate.integrated_fluxes,
                    accepted_flux_rates,
                    strict=True,
                )
            )
            if isinstance(candidate, BalancedPositivityBlendResult):
                if accepted_balanced_contributions is None:
                    accepted_balanced_contributions = jax.tree.map(
                        jnp.zeros_like, candidate.contributions
                    )
                accepted_balanced_contributions = jax.tree.map(
                    lambda new, old: jnp.where(take, new, old),
                    candidate.contributions,
                    accepted_balanced_contributions,
                )
            accepted = accepted | take
            current_dt = self.precision.decision(
                current_dt * self.policy.reduction_factor
            )

        minimum_reached = current_dt < self.precision.decision(
            self.policy.minimum_step_size
        )
        status = jnp.where(
            accepted & (retries > 0),
            int(FiniteVolumeRunStatus.RECOVERED_REJECTION),
            jnp.where(
                accepted,
                int(FiniteVolumeRunStatus.SUCCESS),
                jnp.where(
                    minimum_reached,
                    int(FiniteVolumeRunStatus.MINIMUM_STEP_REACHED),
                    int(FiniteVolumeRunStatus.RETRY_LIMIT_REACHED),
                ),
            ),
        )
        original_content = runtime_state.content_state
        accepted_content = FiniteVolumeConservativeContentState.from_cell_average(
            accepted_average.reshape(original_content.conservative_content.shape),
            original_content.effective_cell_volumes,
            original_content.active_cell_mask,
            self.precision.decision(runtime_state.time + accepted_dt),
            topology_epoch_id=original_content.topology_epoch_id,
            geometry_family_id=original_content.geometry_family_id,
            geometry_layout_id=original_content.geometry_layout_id,
            geometry_version=original_content.geometry_version,
            evidence_policy_id=original_content.evidence_policy_id,
            evidence_version=original_content.evidence_version,
            precision=original_content.precision,
            owned_cell_mask=original_content.owned_cell_mask,
        )
        next_content = jax.lax.cond(
            accepted,
            lambda _: accepted_content,
            lambda _: original_content,
            operand=None,
        )
        next_state = FiniteVolumeRuntimeState(
            next_content,
            runtime_state.topology_journal,
            self.precision.decision(
                jnp.where(accepted, accepted_dt, runtime_state.step_size)
            ),
            accepted_step=(runtime_state.accepted_step + accepted.astype(jnp.int32)),
            last_status=status,
            controller_state=runtime_state.controller_state,
            integrator_state=runtime_state.integrator_state,
            output_cursor=runtime_state.output_cursor,
            sliding_coupling=runtime_state.sliding_coupling,
            sliding_shift=runtime_state.sliding_shift,
            sliding_event_id=runtime_state.sliding_event_id,
        )
        shallow_water_integrals = None
        if isinstance(self.dynamics, PreparedFiniteVolumeDynamics) and isinstance(
            self.dynamics.method.interface_solver,
            ShallowWaterHydrostaticHLLPlan,
        ):
            if accepted_balanced_contributions is None or not isinstance(
                self.dynamics.bathymetry, PreparedShallowWaterBathymetry
            ):
                raise RuntimeError(
                    "Hydrostatic shallow-water acceptance lacks face evidence."
                )
            shallow_water_integrals = ShallowWaterAcceptedFaceIntegrals(
                accepted_balanced_contributions,
                tuple(self.dynamics.discretization.face_measures),
                accepted_dt,
                axis_names=self.dynamics.discretization.grid.axis_names,
                bed_id=self.dynamics.bathymetry.bed_id,
                plan_id=self.dynamics.method.interface_solver.plan_id,
            )
        stage_flux_trace = None
        if self.stage_flux_provider is not None:
            if accepted_stage_flux_trace is None or attempted_stage_flux_trace is None:
                raise RuntimeError(
                    "Stage-flux retry loop did not retain complete attempt evidence."
                )
            stage_flux_trace = jax.tree.map(
                lambda accepted_value, attempted_value: jnp.where(
                    accepted, accepted_value, attempted_value
                ),
                accepted_stage_flux_trace,
                attempted_stage_flux_trace,
            )
        return FiniteVolumeAdvanceResult(
            runtime_state=next_state,
            accepted=accepted,
            retries=retries,
            attempted_step_size=attempted,
            accepted_step_size=accepted_dt,
            positivity=last_report,
            accepted_flux_integrals=self._static_accepted_flux_integral_ledger(
                original_content,
                next_content,
                accepted_flux_rates,
                accepted_dt,
                self.precision.decision(
                    runtime_state.time + jnp.where(accepted, accepted_dt, attempted)
                ),
                runtime_state.accepted_step + accepted.astype(jnp.int32),
            ),
            precision_evidence=self.precision.evidence(),
            ale=None,
            embedded=None,
            shallow_water_integrals=shallow_water_integrals,
            stage_flux_trace=stage_flux_trace,
        )


__all__ = [
    "FiniteVolumeALEAdvanceEvidence",
    "FiniteVolumeEmbeddedAdvanceEvidence",
    "FiniteVolumeOwnerLocalCellStateBankSpec",
    "FiniteVolumeAdvanceResult",
    "FiniteVolumeOwnerLocalRepartitionResult",
    "FiniteVolumeTopologyEventRefusal",
    "FiniteVolumeScheduledAdvanceResult",
    "FiniteVolumeStageFlux",
    "FiniteVolumeStageFluxProvider",
    "FiniteVolumeStageFluxTrace",
    "FiniteVolumeRunStatus",
    "FiniteVolumeRuntimeState",
    "FiniteVolumeStepPolicy",
    "FiniteVolumeValidatedRuntimeState",
    "PreparedFiniteVolumeRuntime",
]
