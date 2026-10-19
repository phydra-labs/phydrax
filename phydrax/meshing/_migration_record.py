#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Topology-neutral migration inputs and independently replayed numerical receipts."""

from __future__ import annotations

from dataclasses import fields
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax
from jax import Array

import phydrax

from .._fingerprint import (
    canonical_fingerprint,
    logical_array_value_collection_digest,
    tree_content_equal,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..typing import Dim, Size, validate
from ._migration_anchor import MigrationAnchor
from ._migration_state_record import (
    MigrationTransportEvidence,
    MigrationWorkset,
    MigrationWorkState,
)


if TYPE_CHECKING:
    from .._meshcore import NativeExecutionScope
    from ..discretization._adaptive_simplex import AdaptiveSimplexParts
    from ..discretization._coordinate_budget import CoordinateEnclosureBudget
    from ..lifecycle._restart_topology import TopologyRestartPolicy
    from ._device_adaptation import PartitionedAdaptiveSimplex
    from ._distribution_migration import PartitionedEpochMigration


class _MigrationPartDim(Dim, minimum=1):
    """Logical partitions retained by one migration source."""


@final
class MigrationPlacement(StrictModule, NonTrainableState):
    """Plain execution-layout metadata admitted before live mesh rebinding."""

    __strict_contract__ = True
    part_count: Size[_MigrationPartDim] = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    axis_names: tuple[str, ...] = eqx.field(static=True)
    mesh_shape: tuple[int, ...] = eqx.field(static=True)
    device_keys: tuple[tuple[int, int], ...] = eqx.field(static=True)
    runtime_device_keys: tuple[tuple[int, int], ...] = eqx.field(static=True)
    neighbor_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    placement_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis_name: str,
        axis_names: tuple[str, ...],
        mesh_shape: tuple[int, ...],
        device_keys: tuple[tuple[int, int], ...],
        runtime_device_keys: tuple[tuple[int, int], ...],
        neighbor_pairs: tuple[tuple[int, int], ...],
        /,
    ) -> None:
        self.part_count = len(device_keys)
        self.axis_name = canonical_identifier(axis_name, "MigrationPlacement.axis_name")
        self.axis_names = axis_names
        self.mesh_shape = mesh_shape
        self.device_keys = device_keys
        self.runtime_device_keys = runtime_device_keys
        self.neighbor_pairs = neighbor_pairs
        self.placement_id = self._identity()
        self.validate_restored()

    def _identity(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "collective-migration-placement",
                "axis_name": self.axis_name,
                "axis_names": self.axis_names,
                "mesh_shape": self.mesh_shape,
                "device_keys": self.device_keys,
                "runtime_device_keys": self.runtime_device_keys,
                "neighbor_pairs": self.neighbor_pairs,
            }
        )

    @classmethod
    def from_live(cls, parts: AdaptiveSimplexParts) -> MigrationPlacement:
        return cls(
            parts.axis_name,
            tuple(parts.mesh.axis_names),
            tuple(parts.mesh.devices.shape),
            tuple(
                (device.process_index, device.id) for device in parts.mesh.devices.flat
            ),
            tuple((device.process_index, device.id) for device in jax.devices()),
            parts.neighbor_pairs,
        )

    @classmethod
    def from_stored_fields(cls, value: MigrationPlacement, /) -> MigrationPlacement:
        value.validate_restored()
        return cls(
            value.axis_name,
            value.axis_names,
            value.mesh_shape,
            value.device_keys,
            value.runtime_device_keys,
            value.neighbor_pairs,
        )

    def validate_restored(self) -> None:
        validate(self)
        from .._execution_resources import ExecutionGroupSpec

        if (
            self.part_count != len(self.device_keys)
            or self.axis_name not in self.axis_names
        ):
            raise ValueError(
                "MigrationPlacement.axes must bind the logical partition extent; got inconsistent axes"
            )
        if any(type(size) is not int or size < 1 for size in self.mesh_shape):
            raise ValueError(
                "MigrationPlacement.mesh_shape must have positive integer extents; got invalid dimensions"
            )
        if len(self.axis_names) != len(self.mesh_shape):
            raise ValueError(
                "MigrationPlacement.mesh_shape must match its axis names; got mismatched rank"
            )
        ExecutionGroupSpec(
            self.placement_id,
            tuple(sorted({process for process, _ in self.device_keys})),
            self.device_keys,
            mesh_axes=tuple(zip(self.axis_names, self.mesh_shape, strict=True)),
        )
        if len(set(self.runtime_device_keys)) != len(self.runtime_device_keys) or not set(
            self.device_keys
        ) <= set(self.runtime_device_keys):
            raise ValueError(
                "MigrationPlacement.runtime_device_keys must contain each selected device once; got inconsistent device ordinals"
            )
        pairs = set(self.neighbor_pairs)
        if any(
            first == second
            or min(first, second) < 0
            or max(first, second) >= self.part_count
            or (second, first) not in pairs
            for first, second in pairs
        ):
            raise ValueError(
                "MigrationPlacement.neighbor_pairs must be bounded reciprocal routes; got invalid routes"
            )
        if self._identity() != self.placement_id:
            raise ValueError(
                "MigrationPlacement.placement_id must bind its execution metadata; got changed fields"
            )

    def rebind(self, policy: TopologyRestartPolicy | None = None) -> AdaptiveSimplexParts:
        from .._execution_resources import ExecutionGroupSpec
        from .._execution_runtime import bind_execution_group
        from ..discretization._adaptive_simplex import AdaptiveSimplexParts
        from ..lifecycle._restart_topology import (
            admit_topology_restart,
            TopologyRestartPolicy,
            TopologyRestartRelation,
        )

        self.validate_restored()
        current = tuple((device.process_index, device.id) for device in jax.devices())
        target_id = canonical_fingerprint(
            {
                "kind": "collective-migration-topology",
                "runtime_device_keys": current,
            }
        )
        source_id = canonical_fingerprint(
            {
                "kind": "collective-migration-topology",
                "runtime_device_keys": self.runtime_device_keys,
            }
        )
        relation_class = "bitwise" if len(current) >= self.part_count else "unsupported"
        relation = TopologyRestartRelation(
            source_id,
            target_id,
            relation_class,
            reason=""
            if relation_class == "bitwise"
            else "Logical migration partition count exceeds available devices.",
        )
        admitted = admit_topology_restart(
            relation,
            TopologyRestartPolicy(allow_topology_change=False)
            if policy is None
            else policy,
        )
        if not admitted.admitted:
            raise ValueError(
                f"MigrationPlacement.topology must admit its restart relation; got {admitted.reason}"
            )
        selected = (
            self.device_keys
            if set(self.device_keys) <= set(current)
            else current[: self.part_count]
        )
        group = bind_execution_group(
            ExecutionGroupSpec(
                canonical_fingerprint(
                    {
                        "kind": "rebound-migration-group",
                        "devices": selected,
                        "axes": self.axis_names,
                    }
                ),
                tuple(sorted({process for process, _ in selected})),
                selected,
                mesh_axes=tuple(zip(self.axis_names, self.mesh_shape, strict=True)),
            )
        )
        return AdaptiveSimplexParts(
            group.devices, axis_name=self.axis_name, neighbor_pairs=self.neighbor_pairs
        )


@final
class MigrationSourceRecord(StrictModule, NonTrainableState):
    """Logical source epoch and explicit anchor banks needed by migration replay."""

    __strict_contract__ = True
    part_count: Size[_MigrationPartDim] = eqx.field(static=True)
    adaptation: phydrax.meshing._adaptation.PreparedMeshAdaptation
    layout: phydrax.discretization._adaptive_simplex.AdaptiveSimplexLayout
    preparation_state: phydrax.discretization._adaptive_simplex.AdaptiveSimplexState
    anchor: MigrationAnchor
    preparation_evidence: phydrax.meshing._measurements.NativeExecutionRecord | None
    placement: MigrationPlacement
    states: phydrax.discretization._adaptive_simplex.AdaptiveSimplexState
    source_exterior: Array
    slot_origins: Array
    solver_cell_owners: Array
    placement_arrays: tuple[tuple[str, Array], ...]
    solver_neighborhood: MigrationWorkset | None
    solver_forest: phydrax.discretization._adaptive_simplex.AdaptiveSimplexState | None
    execution_evidence: phydrax.meshing._measurements.NativeExecutionRecord | None
    prepared_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    source_cache_id: str = eqx.field(static=True)
    initial_state_id: str = eqx.field(static=True)
    partitioned_id: str = eqx.field(static=True)
    restart_content_id: str | None = eqx.field(static=True)
    rebalance_proposal: phydrax.meshing._distribution.PartitionedRebalanceProposal | None
    prior_migration: CollectiveMigrationRecord | None

    def __init__(self, source: PartitionedAdaptiveSimplex, /) -> None:
        self.part_count = source.parts.part_count
        self.adaptation = source.prepared.adaptation
        self.layout = source.layout
        self.preparation_state = source.prepared.state
        self.anchor = MigrationAnchor.from_live(source.prepared.anchor)
        self.preparation_evidence = source.prepared.execution_evidence
        self.placement = MigrationPlacement.from_live(source.parts)
        self.states = source.states
        self.source_exterior = source.source_exterior
        self.slot_origins = source.slot_origins
        self.solver_cell_owners = source.solver_cell_owners
        self.placement_arrays = source.placement_arrays
        self.solver_neighborhood = (
            None
            if source.solver_neighborhood is None
            else MigrationWorkset.from_live(source.solver_neighborhood)
        )
        self.solver_forest = source.solver_forest
        self.execution_evidence = source.execution_evidence
        self.prepared_id = source.prepared.prepared_id
        self.source_topology_id = source.prepared.source_topology_id
        self.source_cache_id = source.source_cache_id
        self.initial_state_id = source.initial_state_id
        self.partitioned_id = source.partitioned_id
        self.restart_content_id = (
            None if source.restart_proof is None else source.restart_proof.content_id
        )
        self.rebalance_proposal = source.rebalance_proposal
        self.prior_migration = (
            None
            if source.migration is None
            else CollectiveMigrationRecord(source.migration)
        )
        self.validate_restored()

    def validate_restored(self) -> None:
        validate(self)
        self.anchor.validate_restored()
        self.placement.validate_restored()
        if (
            self.part_count != self.placement.part_count
            or self.states.mesh.cell_ids.shape[0] != self.part_count
        ):
            raise ValueError(
                "MigrationSourceRecord.part_count must bind its placement and state axes; got inconsistent partitions"
            )
        if (
            self.source_exterior.shape != self.states.mesh.cells.shape
            or self.slot_origins.shape != self.states.mesh.cell_ids.shape
            or self.solver_cell_owners.shape != self.states.mesh.cell_ids.shape
        ):
            raise ValueError(
                "MigrationSourceRecord.routing must align with source cell axes; got inconsistent banks"
            )
        if self.prepared_id != canonical_fingerprint(
            {
                "kind": "prepared-adaptive-simplex",
                "adaptation": self.adaptation.prepared_id,
                "layout": self.layout.signature_id,
            }
        ):
            raise ValueError(
                "MigrationSourceRecord.prepared_id must bind adaptation and layout; got changed identity"
            )
        if self.source_topology_id != self.adaptation.source.mesh.topology_id:
            raise ValueError(
                "MigrationSourceRecord.source_topology_id must bind its accepted source; got changed topology"
            )
        if self.solver_neighborhood is not None:
            self.solver_neighborhood.validate_restored()
        for evidence in (self.preparation_evidence, self.execution_evidence):
            if evidence is not None:
                evidence.require_valid()

    @classmethod
    def from_stored_fields(cls, value: MigrationSourceRecord, /) -> MigrationSourceRecord:
        from .._model._structure import restore_registered_fields

        value.validate_restored()
        return restore_registered_fields(
            cls, {field.name: getattr(value, field.name) for field in fields(value)}
        )

    def rebind(
        self,
        *,
        scope: NativeExecutionScope,
        budget: CoordinateEnclosureBudget,
        policy: TopologyRestartPolicy | None = None,
    ) -> PartitionedAdaptiveSimplex:
        from ._device_adaptation import (
            PartitionedAdaptiveSimplex,
            PreparedAdaptiveSimplex,
        )

        self.validate_restored()
        parts = self.placement.rebind(policy)
        prepared = PreparedAdaptiveSimplex(
            self.adaptation,
            self.layout,
            self.preparation_state,
            self.anchor.to_live(),
            execution_evidence=self.preparation_evidence,
        )
        source = PartitionedAdaptiveSimplex(
            prepared,
            self.layout,
            parts,
            _place_partition_tree(self.states, parts),
            _place_partition_tree(self.source_exterior, parts),
            _place_partition_tree(self.slot_origins, parts),
            solver_cell_owners=_place_partition_tree(self.solver_cell_owners, parts),
            placement_arrays=self.placement_arrays,
            solver_neighborhood=None
            if self.solver_neighborhood is None
            else _place_partition_tree(self.solver_neighborhood.to_live(), parts),
            solver_forest=None
            if self.solver_forest is None
            else _place_partition_tree(self.solver_forest, parts),
            migration=None
            if self.prior_migration is None
            else self.prior_migration.replay(scope=scope, budget=budget, policy=policy),
            rebalance_proposal=self.rebalance_proposal,
            execution_evidence=self.execution_evidence,
            scope=scope,
            budget=budget,
        )
        if any(
            getattr(source, name) != getattr(self, name)
            for name in ("source_cache_id", "initial_state_id", "partitioned_id")
        ):
            raise ValueError(
                "MigrationSourceRecord.identities must reproduce source preparation bindings; got changed source records"
            )
        return source


@final
class CollectiveMigrationRecord(StrictModule, NonTrainableState):
    """Portable migration inputs and numerical outputs authenticated before replay."""

    __strict_contract__ = True
    part_count: Size[_MigrationPartDim] = eqx.field(static=True)
    source: MigrationSourceRecord
    saved_states: phydrax.discretization._adaptive_simplex.AdaptiveSimplexState
    saved_migration_state: MigrationWorkState
    requested_target_owners: Array
    states: phydrax.discretization._adaptive_simplex.AdaptiveSimplexState
    initial_states: phydrax.discretization._adaptive_simplex.AdaptiveSimplexState
    source_exterior: Array
    slot_origins: Array
    placement: MigrationPlacement
    solver_cell_owners: Array
    initial_solver_cell_owners: Array
    solver_neighborhood: MigrationWorkset
    solver_forest: phydrax.discretization._adaptive_simplex.AdaptiveSimplexState
    solver_initial_states: phydrax.discretization._adaptive_simplex.AdaptiveSimplexState
    solver_source_exterior: Array
    solver_slot_origins: Array
    transport_evidence: MigrationTransportEvidence
    migration_id: str = eqx.field(static=True)

    def __init__(self, migration: PartitionedEpochMigration, /) -> None:
        self.part_count = migration.parts.part_count
        self.source = MigrationSourceRecord(migration.source_partitioned)
        self.saved_states = migration.saved_states
        self.saved_migration_state = MigrationWorkState.from_live(
            migration.saved_migration_state
        )
        self.requested_target_owners = migration.requested_target_owners
        self.states = migration.states
        self.initial_states = migration.initial_states
        self.source_exterior = migration.source_exterior
        self.slot_origins = migration.slot_origins
        self.placement = MigrationPlacement.from_live(migration.parts)
        self.solver_cell_owners = migration.solver_cell_owners
        self.initial_solver_cell_owners = migration.initial_solver_cell_owners
        self.solver_neighborhood = MigrationWorkset.from_live(
            migration.solver_neighborhood
        )
        self.solver_forest = migration.solver_forest
        self.solver_initial_states = migration.solver_initial_states
        self.solver_source_exterior = migration.solver_source_exterior
        self.solver_slot_origins = migration.solver_slot_origins
        self.transport_evidence = MigrationTransportEvidence.from_live(
            migration.migration.evidence
        )
        self.migration_id = self._identity()
        self.validate_restored()

    def _identity(self) -> str:
        arrays = {
            "requested_target_owners": self.requested_target_owners,
            "source_exterior": self.source_exterior,
            "slot_origins": self.slot_origins,
            "solver_cell_owners": self.solver_cell_owners,
            "initial_solver_cell_owners": self.initial_solver_cell_owners,
            "solver_source_exterior": self.solver_source_exterior,
            "solver_slot_origins": self.solver_slot_origins,
        }
        for prefix, value in (
            ("states", self.states),
            ("initial_states", self.initial_states),
            ("solver_forest", self.solver_forest),
            ("solver_initial", self.solver_initial_states),
        ):
            arrays.update(
                {
                    f"{prefix}/{jax.tree_util.keystr(path)}": leaf
                    for path, leaf in jax.tree_util.tree_flatten_with_path(value)[0]
                    if isinstance(leaf, Array)
                }
            )
        return canonical_fingerprint(
            {
                "kind": "collective-migration-record",
                "source_prepared": self.source.prepared_id,
                "source_state": self.source.initial_state_id,
                "source_routing": self.source.source_cache_id,
                "part_count": self.part_count,
                "axis_name": self.placement.axis_name,
                "neighbor_pairs": self.placement.neighbor_pairs,
                "numerical_receipts": logical_array_value_collection_digest(arrays),
            }
        )

    def validate_restored(self) -> None:
        validate(self)
        self.source.validate_restored()
        self.saved_migration_state.validate_restored()
        self.solver_neighborhood.validate_restored()
        self.transport_evidence.validate_restored()
        self.placement.validate_restored()
        if (
            self.part_count != self.source.part_count
            or self.part_count != self.placement.part_count
        ):
            raise ValueError(
                "CollectiveMigrationRecord.part_count must bind source and output placement; got inconsistent partition count"
            )
        if (
            self.source_exterior.shape != self.initial_states.mesh.cells.shape
            or self.slot_origins.shape != self.initial_states.mesh.cell_ids.shape
        ):
            raise ValueError(
                "CollectiveMigrationRecord.initial_routing must bind retained initial states; got inconsistent banks"
            )
        if (
            self.solver_source_exterior.shape
            != self.solver_initial_states.mesh.cells.shape
            or self.solver_slot_origins.shape
            != self.solver_initial_states.mesh.cell_ids.shape
        ):
            raise ValueError(
                "CollectiveMigrationRecord.solver_routing must bind retained solver states; got inconsistent banks"
            )
        if self._identity() != self.migration_id:
            raise ValueError(
                "CollectiveMigrationRecord.migration_id must bind its numerical receipts; got changed content"
            )

    @classmethod
    def from_stored_fields(
        cls, value: CollectiveMigrationRecord, /
    ) -> CollectiveMigrationRecord:
        from .._model._structure import restore_registered_fields

        value.validate_restored()
        return restore_registered_fields(
            cls, {field.name: getattr(value, field.name) for field in fields(value)}
        )

    def replay(
        self,
        *,
        scope: NativeExecutionScope,
        budget: CoordinateEnclosureBudget,
        policy: TopologyRestartPolicy | None = None,
    ) -> PartitionedEpochMigration:
        from ._distribution_migration import migrate_partitioned_epoch

        self.validate_restored()
        parts = self.placement.rebind(policy)
        source = self.source.rebind(scope=scope, budget=budget, policy=policy)
        replayed = migrate_partitioned_epoch(
            source,
            _place_partition_tree(self.saved_states, parts),
            _place_partition_tree(self.saved_migration_state.to_live(), parts),
            _place_partition_tree(self.requested_target_owners, parts),
        )
        renewed = CollectiveMigrationRecord(replayed)
        if renewed.migration_id != self.migration_id or not tree_content_equal(
            renewed.transport_evidence, self.transport_evidence
        ):
            raise ValueError(
                "CollectiveMigrationRecord.replay must reproduce its migration receipts; got changed mathematical or transport evidence"
            )
        return replayed


def _place_partition_tree[T](value: T, parts: AdaptiveSimplexParts) -> T:
    """Place admitted logical partition banks on the rebound execution mesh."""
    sharding = jax.sharding.NamedSharding(
        parts.mesh, jax.sharding.PartitionSpec(parts.axis_name)
    )
    replicated = jax.sharding.NamedSharding(parts.mesh, jax.sharding.PartitionSpec())

    def place(leaf: object) -> object:
        if isinstance(leaf, Array):
            return jax.device_put(
                leaf,
                sharding
                if leaf.ndim and leaf.shape[0] == parts.part_count
                else replicated,
            )
        return leaf

    return jax.tree.map(place, value)


MIGRATION_RECORD_ARTIFACT_TYPES = (
    MigrationPlacement,
    MigrationSourceRecord,
    CollectiveMigrationRecord,
)
