# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Meshfree checkpoint inventory and restart transports for production runs.

Encoding, persistence, trust, outputs and transactions stay with the solver's
production stores and ``phydrax.lifecycle``. This module owns only which
native meshfree identities a restored run must match, how a moving surface
binds to the fixed-step runtime and its acknowledged archive, which committed
native transports (distributed ownership migration, support-epoch rebase) may
move a checkpoint between two identity inventories, and how a restarting
process re-derives its current support epoch from the migrations committed
with the store's checkpoint lineage.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, assert_never, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver._fixed_step import AbstractFixedStepMethod, FixedStepResult
from ...solver._production_runtime import (
    ArtifactCheckpointStore,
    ProductionArchivePolicy,
)
from ...solver._runtime_lifecycle import (
    RuntimeIdentityInventory,
    RuntimeMigrationReceipt,
    RuntimeRestartRelation,
    StaleRuntimeCheckpointError,
)
from ...solver.coupling._surface_exchange import MeshfreeBulkSurfaceMethod
from ...typing import checked
from .._point_cloud import PreparedPointCloudDiscretization
from ..spatial._distributed_relations import DistributedPointLayout
from ._distributed import DistributedMeshfreeOperator
from ._evolution import PreparedMeshfreeEvolution
from ._moving import MovingSurfacePlan, MovingSurfaceState
from ._transport import ConservativeTransport


def _content(kind: str, tree: Any, /) -> str:
    """Content identity of a host-visible array tree under one named role."""
    return canonical_fingerprint({"kind": kind, "arrays": array_tree_fingerprint(tree)})


def _evolution_identities(evolution: PreparedMeshfreeEvolution, /) -> dict[str, str]:
    plan = evolution.plan
    capacity = evolution.capacity
    motion = plan.motion
    material = None if motion is None else motion.material
    masses = () if material is None else (material.masses,)
    identities = {
        "point-ids": _content("meshfree-stable-ids", capacity.stable_ids),
        "support": _content("meshfree-support-trust", capacity.support_trust),
        "capacity": canonical_fingerprint(
            {
                "kind": "meshfree-evolution-capacity",
                "state_size": capacity.state_size,
                "point_count": capacity.point_count,
                "edge_count": capacity.edge_count,
                "dimension": capacity.dimension,
                "layout": capacity.layout,
                "measure": capacity.measure,
            }
        ),
    }
    spatial = plan.spatial
    match spatial:
        case PreparedPointCloudDiscretization():
            cloud = spatial.plan
            identities |= {
                "cloud": spatial.plan_id,
                "geometry": _content("meshfree-anchored-geometry", cloud.points),
                "topology": _content("meshfree-support-relation", spatial.relation),
                "measure": _content(
                    "meshfree-measure", (cloud.quadrature_weights, *masses)
                ),
                "stencil": spatial.prepared_id,
                "boundary": canonical_fingerprint(
                    {
                        "kind": "meshfree-collocation-boundary",
                        "arrays": array_tree_fingerprint(
                            (
                                cloud.boundary_mask,
                                cloud.boundary_normals,
                                cloud.boundary_quadrature_weights,
                            )
                        ),
                        "rows": None
                        if plan.boundary is None
                        else plan.boundary.boundary_id,
                    }
                ),
            }
        case ConservativeTransport():
            exterior = spatial.exterior
            identities |= {
                "cloud": spatial.transport_id,
                "geometry": _content("meshfree-graph-geometry", exterior.points),
                "topology": _content("meshfree-graph-edges", exterior.pairs),
                "measure": _content("meshfree-measure", (exterior.node_volumes, *masses)),
                "metric": _content(
                    "meshfree-graph-metric", exterior.metric_result.weights
                ),
                "boundary": _content("meshfree-graph-boundary", spatial.boundary_mask),
            }
            if spatial.reconstruction is not None:
                identities["stencil"] = spatial.reconstruction.prepared_id
        case unknown:
            assert_never(unknown)
    return identities


def _moving_identities(plan: MovingSurfacePlan, /) -> dict[str, str]:
    epoch = plan.epoch
    return {
        "cloud": plan.plan_id,
        "geometry": epoch.geometry_id,
        "topology": epoch.topology_id,
        "point-ids": _content("meshfree-stable-ids", plan.capacity.stable_ids),
        "capacity": plan.capacity.mapping_id,
    }


def _bulk_surface_identities(method: MeshfreeBulkSurfaceMethod, /) -> dict[str, str]:
    return {
        "cloud": method.transport.transport_id,
        "geometry": _content("meshfree-surface-geometry", method.query.points),
        "measure": _content(
            "meshfree-bulk-surface-measure",
            (method.bulk_volumes, method.surface_measures),
        ),
        "metric": _content(
            "meshfree-surface-metric", (method.surface_conductances, method.metric_exact)
        ),
        "query": method.query.query_id,
        "interface": method.transport.transport_id,
    }


def _distributed_identities(operator: DistributedMeshfreeOperator, /) -> dict[str, str]:
    layout = operator.target_layout
    if operator.source_layout.partition_fingerprint() != layout.partition_fingerprint():
        raise ValueError("A distributed runtime state lives on one owner partition.")
    plan = layout.plan
    return {
        "stencil": operator.operator_id,
        # Logical-order identities are independent of the owner partition.
        "geometry": _content("meshfree-logical-geometry", layout.collect(layout.points)),
        "point-ids": _content("meshfree-stable-ids", layout.collect(layout.stable_ids)),
        "partition": layout.partition_fingerprint(),
        # Route/halo buckets follow the partition; an ownership migration may move them.
        "capacity": canonical_fingerprint(
            {
                "kind": "distributed-meshfree-capacity",
                "ownership": plan.plan_id,
                "logical_count": layout.logical_count,
                "binding": operator.binding_id,
            }
        ),
    }


@checked
def meshfree_runtime_inventory(
    participant: PreparedMeshfreeEvolution
    | MovingSurfacePlan
    | MeshfreeBulkSurfaceMethod
    | DistributedMeshfreeOperator,
    /,
    *,
    source: str,
    program: str,
    method: AbstractFixedStepMethod,
    controller: str,
    precision: str,
    rng: str | None = None,
    lineage: str | None = None,
    hierarchy: str | None = None,
) -> RuntimeIdentityInventory:
    """Identity inventory a restored meshfree production run must match.

    Discretization identities come from the prepared native owners: anchored
    geometry, support relation, stable point IDs, measure realization
    (quadrature weights, graph volumes or material masses), support trust,
    stencil, metric, boundary rows and query/interface sources, the owner
    partition of a distributed operator, and the static state capacity.
    ``source`` is the build identity, ``program`` the prepared
    ``ProductionRunPlan.plan_id``, ``controller`` the retry/epoch controller
    identity, ``precision`` the declared precision policy and ``rng`` the RNG
    addressing. Any differing role refuses a restore before use.
    """
    identities: dict[str, str] = {
        "source": source,
        "program": program,
        "method": method.method_id,
        "controller": controller,
        "precision": precision,
    }
    match participant:
        case PreparedMeshfreeEvolution():
            identities |= _evolution_identities(participant)
        case MovingSurfacePlan():
            identities |= _moving_identities(participant)
        case MeshfreeBulkSurfaceMethod():
            identities |= _bulk_surface_identities(participant)
        case DistributedMeshfreeOperator():
            identities |= _distributed_identities(participant)
        case unknown:
            assert_never(unknown)
    optional: Mapping[str, str | None] = {
        "rng": rng,
        "lineage": lineage,
        "hierarchy": hierarchy,
    }
    identities |= {role: value for role, value in optional.items() if value is not None}
    return RuntimeIdentityInventory(identities)


@final
class _OwnershipMigrationTransport(StrictModule, NonTrainableState):
    """Replay native ownership packets with dynamic layout and destination data."""

    source_layout: DistributedPointLayout
    destinations: Array
    packet_capacity: int = eqx.field(static=True)
    target_partition: str = eqx.field(static=True)

    def __call__(self, state: Any, /) -> Any:
        replayed = self.source_layout.migrate(
            self.destinations, packet_capacity=self.packet_capacity, payload=state
        )
        if (
            not bool(np.asarray(jax.device_get(replayed.evidence.committed)))
            or replayed.layout.partition_fingerprint() != self.target_partition
        ):
            raise ValueError("Replayed ownership migration differs from its receipt.")
        return replayed.payload


@checked
def ownership_migration_relation(
    source: RuntimeIdentityInventory,
    target: RuntimeIdentityInventory,
    source_layout: DistributedPointLayout,
    destinations: ArrayLike,
    /,
    *,
    packet_capacity: int,
    source_template: Any,
) -> RuntimeRestartRelation:
    """Bitwise restart relation across one committed distributed ownership change.

    The native ``DistributedPointLayout.migrate`` transfer is executed once on
    ``source_layout`` and must commit into exactly the partition bound by
    ``target``; its packet/capacity/epoch evidence becomes the receipt's
    transport identity. Restore replays the same transfer on the archived
    owner-blocked state (every state leaf leads with the owner-slot axis), so
    no value is re-derived and the result is refused unless it commits again.
    """
    if source.identity("partition") != source_layout.partition_fingerprint():
        raise ValueError("Source inventory does not bind the source ownership layout.")
    committed = source_layout.migrate(destinations, packet_capacity=packet_capacity)
    if not bool(np.asarray(jax.device_get(committed.evidence.committed))):
        raise ValueError("The ownership migration rolled back; no restart relation.")
    target_partition = committed.layout.partition_fingerprint()
    if target.identity("partition") != target_partition:
        raise ValueError("Target inventory does not bind the migrated ownership layout.")
    receipt = RuntimeMigrationReceipt(
        "ownership",
        source,
        target,
        transport_id=canonical_fingerprint(
            {
                "kind": "distributed-ownership-migration",
                "source": source.identity("partition"),
                "target": target_partition,
                "packet_capacity": packet_capacity,
                "evidence": array_tree_fingerprint(committed.evidence),
            }
        ),
    )

    transport = _OwnershipMigrationTransport(
        source_layout, jnp.asarray(destinations), packet_capacity, target_partition
    )

    return RuntimeRestartRelation.from_migration(
        receipt, transport, source_template=source_template, classification="bitwise"
    )


@final
class _SupportEpochTransport(StrictModule, NonTrainableState):
    """Replay only the source state whose content the epoch receipt certifies."""

    evolution: PreparedMeshfreeEvolution
    source_anchor_id: str = eqx.field(static=True)

    def __call__(self, state: Any, /) -> Array:
        restored_id = _content("meshfree-support-epoch-source-anchor", state)
        if restored_id != self.source_anchor_id:
            raise ValueError(
                "Restored support-epoch source anchor differs from the migration "
                f"receipt: expected {self.source_anchor_id}, restored {restored_id}."
            )
        return self.evolution.repacked(state)


@checked
def support_epoch_relation(
    source: RuntimeIdentityInventory,
    target: RuntimeIdentityInventory,
    evolution: PreparedMeshfreeEvolution,
    successor: PreparedMeshfreeEvolution,
    anchor: ArrayLike,
    /,
    *,
    source_template: Any,
) -> RuntimeRestartRelation:
    """Bitwise restart relation across one support-epoch rebase.

    ``successor`` must be ``evolution.rebase(anchor)``: the same evolution
    re-anchored at the admitted ``anchor`` coordinates (periodic coordinates
    wrapped). The receipt binds the complete source anchor's canonical content;
    restore refuses any other archived state before applying the rebase's
    ``evolution.repacked`` state map. A refused or foreign rebase admits no relation.
    """
    if successor.evolution_id != evolution.evolution_id:
        raise ValueError("A support epoch keeps the evolution plan; prepare a new run.")
    repacked = evolution.repacked(anchor)
    source_anchor_id = _content(
        "meshfree-support-epoch-source-anchor", jnp.asarray(anchor)
    )
    successor_spatial = successor.plan.spatial
    if not isinstance(successor_spatial, PreparedPointCloudDiscretization) or not (
        np.array_equal(
            np.asarray(successor_spatial.points),
            np.asarray(successor.fields(repacked).points),
        )
    ):
        raise ValueError("The successor is not anchored at the rebased state.")
    receipt = RuntimeMigrationReceipt(
        "epoch",
        source,
        target,
        transport_id=canonical_fingerprint(
            {
                "kind": "meshfree-support-epoch-rebase",
                "evolution": evolution.evolution_id,
                "anchor": array_tree_fingerprint(repacked),
                "source_anchor": source_anchor_id,
            }
        ),
    )
    return RuntimeRestartRelation.from_migration(
        receipt,
        _SupportEpochTransport(evolution, source_anchor_id),
        source_template=source_template,
        classification="bitwise",
    )


@final
class SupportEpochChain(StrictModule, NonTrainableState):
    """Current support epoch re-derived from a checkpoint's committed lineage.

    ``evolution`` is the successor of the last committed rebase (the epoch-0
    evolution when none was committed), ``inventory`` its identity inventory
    and ``epoch`` the number of committed rebases. The store's checkpoint
    belongs to this epoch and resumes through the identity relation.
    """

    evolution: PreparedMeshfreeEvolution
    inventory: RuntimeIdentityInventory
    epoch: int = eqx.field(static=True)


@checked
def resume_support_epochs(
    evolution: PreparedMeshfreeEvolution,
    store: ArtifactCheckpointStore,
    inventory: Callable[[PreparedMeshfreeEvolution], RuntimeIdentityInventory],
    /,
) -> SupportEpochChain:
    """Re-derive the support epoch of a store's committed checkpoint.

    ``evolution`` is epoch 0 and ``inventory`` builds a run's identity
    inventory for one epoch (build source, program, method, controller,
    precision and RNG as when the run was prepared). Every restore through a
    ``support_epoch_relation`` committed its receipt and held anchor with the
    store's checkpoint lineage; they are replayed oldest first. Each epoch's
    inventory must equal the receipt source and its rebase at the archived
    anchor must reproduce the receipt target, or ``StaleRuntimeCheckpointError``
    names the changed roles; the re-derived relation must reproduce the
    committed receipt bitwise.
    """
    current = evolution
    current_inventory = inventory(evolution)
    lineage = store.migration_lineage()
    for record in lineage:
        receipt = record.receipt
        match receipt.kind:
            case "epoch":
                pass
            case "ownership":
                raise ValueError("The checkpoint lineage crossed an ownership migration.")
            case unknown:
                assert_never(unknown)
        stale = current_inventory.changed_roles(receipt.source)
        if stale:
            raise StaleRuntimeCheckpointError(stale)
        anchor = record.source_state(
            current.initial_state(jnp.zeros(current.point_count, dtype=jnp.float64))
        )
        successor = current.rebase(anchor)
        successor_inventory = inventory(successor)
        stale = successor_inventory.changed_roles(receipt.target)
        if stale:
            raise StaleRuntimeCheckpointError(stale)
        relation = support_epoch_relation(
            current_inventory,
            successor_inventory,
            current,
            successor,
            anchor,
            source_template=anchor,
        )
        if relation.relation_id != receipt.receipt_id:
            raise ValueError("The re-derived rebase differs from its committed receipt.")
        current, current_inventory = successor, successor_inventory
    return SupportEpochChain(current, current_inventory, len(lineage))


@final
class MovingSurfaceFixedStepMethod(AbstractFixedStepMethod, NonTrainableState):
    """Fixed-epoch moving-surface runtime published to native fixed-step consumers.

    The candidate, accepted state, success and ``MovingSurfaceEvidence`` are
    the plan's own; nothing is retried, clipped or repaired here. The runtime
    clock and the state's own clock advance by the same accepted steps.
    """

    plan: MovingSurfacePlan
    method_id: str = eqx.field(static=True)

    @checked
    def __init__(self, plan: MovingSurfacePlan, /) -> None:
        self.plan = plan
        self.method_id = canonical_fingerprint(
            {"kind": "moving-surface-fixed-step", "plan": plan.plan_id}
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: MovingSurfaceState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        del step_index, time
        result = self.plan.step(state, step_size, args=args)
        evidence = result.evidence
        return FixedStepResult(
            result.candidate,
            result.state,
            evidence.successful,
            evidence.implicit_residual,
            evidence.implicit_iterations,
            jnp.asarray(self.plan.tableau.stage_count, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.zeros((), dtype=state.content.dtype),
            evidence=evidence,
        )


@checked
def moving_archive_policy(plan: MovingSurfacePlan, /) -> ProductionArchivePolicy:
    """Acknowledged moving archive bound to the runtime's scheduled checkpoints.

    Every scheduled checkpoint commits the state whose archive cursor covers
    its newest lifetime. The current state occupies one ring slot, so
    ``history_capacity - 1`` accepted steps fit between acknowledgements: that
    is the hard window. A ``rolling`` archive evicts without acknowledgement
    and binds no policy.
    """
    if plan.archive != "acknowledged":
        raise ValueError("Only an acknowledged moving archive binds acknowledgement.")

    def acknowledge(state: MovingSurfaceState) -> MovingSurfaceState:
        return eqx.tree_at(
            lambda item: item.archive_cursor, state, state.accepted_steps + 1
        )

    return ProductionArchivePolicy(
        acknowledge,
        window=plan.history_capacity - 1,
        policy_id=canonical_fingerprint(
            {"kind": "moving-surface-acknowledged-archive", "plan": plan.plan_id}
        ),
    )


__all__ = [
    "MovingSurfaceFixedStepMethod",
    "SupportEpochChain",
    "meshfree_runtime_inventory",
    "moving_archive_policy",
    "ownership_migration_relation",
    "resume_support_epochs",
    "support_epoch_relation",
]
