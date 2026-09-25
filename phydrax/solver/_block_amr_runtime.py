#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum
from math import prod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._trainable import fixed_field, NonTrainableState
from ..discretization._conservation_ledger import (
    AcceptedConservationFluxIntegralBlock,
    AcceptedConservationIntegralLedger,
    ConservationStageLedger,
)
from ..discretization.amr._core import (
    BlockHierarchyState,
    BlockHierarchyTopology,
    BlockLevelState,
)
from ..discretization.amr._fd_runtime import PreparedFDAMRHierarchy
from ..discretization.amr._reflux import FluxRegister
from ..discretization.finite_volume._amr import BlockAMRConservationPlan
from ..discretization.finite_volume._block_amr import (
    BlockAMRFiniteVolumePlan,
    PreparedBlockAMRFiniteVolumeDynamics,
)
from ._finite_volume_topology_events import (
    FiniteVolumeTopologyArtifacts,
    FiniteVolumeTopologyEventJournal,
    FiniteVolumeTopologyEventRequest,
    FiniteVolumeTopologyEventTransaction,
    FiniteVolumeTopologyEventTransactionResult,
    TopologyEventStatus,
)


class BlockAMRAdvancePhase(IntEnum):
    """First failed phase of one atomic hierarchy interval."""

    SUCCESS = 0
    INVALID_INTERVAL = 1
    FILL_PATCH = 2
    STAGE_LEDGER = 3
    NONFINITE = 4
    ADMISSIBILITY = 5
    REFLUX = 6
    RESTRICTION = 7
    SPECIALIST_SYNCHRONIZATION = 8


class AMRTimeSchedulePlan(StrictModule, NonTrainableState):
    """Static hierarchy time schedule for exact SSPRK(3,3) intervals.

    By default each edge uses its coarse level's spatial refinement ratio.  The
    explicit no-subcycling mode changes only temporal subdivision and assigns one
    child interval to every edge.
    """

    hierarchy: PreparedFDAMRHierarchy
    edge_substeps: tuple[int, ...] = eqx.field(static=True)
    level_substeps_per_root: tuple[int, ...] = eqx.field(static=True)
    temporal_method_id: str = eqx.field(static=True)
    subcycling: bool = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        hierarchy: PreparedFDAMRHierarchy,
        /,
        *,
        subcycling: bool = True,
        temporal_method_id: str = "temporal:ssprk33",
        edge_substeps: Sequence[int] | None = None,
    ):
        if not isinstance(hierarchy, PreparedFDAMRHierarchy):
            raise TypeError("hierarchy must be PreparedFDAMRHierarchy.")
        if not isinstance(subcycling, bool):
            raise TypeError("subcycling must be boolean.")
        method_id = str(temporal_method_id)
        if not method_id or method_id != method_id.strip():
            raise ValueError("temporal_method_id must be a canonical identifier.")
        level_plans = hierarchy.plan.hierarchy.levels
        default_edges = tuple(
            level.refinement_ratio if subcycling else 1 for level in level_plans[:-1]
        )
        edges = default_edges if edge_substeps is None else tuple(edge_substeps)
        if len(edges) != len(level_plans) - 1 or any(value <= 0 for value in edges):
            raise ValueError("AMR edge substeps must be positive and align with levels.")
        if not subcycling and any(value != 1 for value in edges):
            raise ValueError("No-subcycling schedules require one substep per edge.")
        counts = [1]
        for value in edges:
            counts.append(counts[-1] * value)
        self.hierarchy = hierarchy
        self.edge_substeps = edges
        self.level_substeps_per_root = tuple(counts)
        self.temporal_method_id = method_id
        self.subcycling = subcycling
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "block-amr-time-schedule",
                "hierarchy": hierarchy.prepared_id,
                "edge_substeps": edges,
                "level_substeps_per_root": counts,
                "temporal_method_id": method_id,
            }
        )


class BlockAMRRuntimeState(StrictModule):
    """Synchronized hierarchy payload and its single canonical topology journal."""

    hierarchy_state: BlockHierarchyState
    topology_journal: FiniteVolumeTopologyEventJournal
    time: Array
    accepted_step: Array
    level_accepted_steps: Array
    last_status: Array

    def __init__(
        self,
        hierarchy_state: BlockHierarchyState,
        topology_journal: FiniteVolumeTopologyEventJournal,
        time: ArrayLike,
        /,
        *,
        accepted_step: ArrayLike = 0,
        level_accepted_steps: ArrayLike | None = None,
        last_status: ArrayLike = BlockAMRAdvancePhase.SUCCESS,
    ):
        if not isinstance(hierarchy_state, BlockHierarchyState):
            raise TypeError("hierarchy_state must be BlockHierarchyState.")
        if not isinstance(topology_journal, FiniteVolumeTopologyEventJournal):
            raise TypeError("topology_journal must be FiniteVolumeTopologyEventJournal.")
        epoch_id = hierarchy_state.topology.epoch.epoch_id
        if topology_journal.current_epoch_id != epoch_id:
            raise ValueError(
                "Block AMR state and topology journal have different epochs."
            )
        time_ = jnp.asarray(time)
        if time_.shape != () or time_.dtype.kind not in "iuf":
            raise TypeError("Block AMR time must be a real scalar.")
        time_ = eqx.error_if(
            time_, ~jnp.isfinite(time_), "Block AMR time must be finite."
        )
        accepted = jnp.asarray(accepted_step, dtype=jnp.int32).reshape(())
        accepted = eqx.error_if(
            accepted,
            accepted < 0,
            "Block AMR accepted_step must be nonnegative.",
        )
        level_count = len(hierarchy_state.levels)
        level_steps = jnp.asarray(
            jnp.full((level_count,), accepted, dtype=jnp.int32)
            if level_accepted_steps is None
            else level_accepted_steps,
            dtype=jnp.int32,
        )
        if level_steps.shape != (level_count,):
            raise ValueError("level_accepted_steps must contain one value per level.")
        level_steps = eqx.error_if(
            level_steps,
            jnp.any(level_steps < 0) | (level_steps[0] != accepted),
            "Per-level accepted steps must be nonnegative and agree at level zero.",
        )
        self.hierarchy_state = hierarchy_state
        self.topology_journal = topology_journal
        self.time = time_
        self.accepted_step = accepted
        self.level_accepted_steps = level_steps
        self.last_status = jnp.asarray(last_status, dtype=jnp.int32).reshape(())


class BlockAMRAdvanceResult(StrictModule):
    """Atomic root-interval result with accepted ledgers and synchronization evidence."""

    runtime_state: BlockAMRRuntimeState = fixed_field()
    accepted: Array = fixed_field()
    attempted_step_size: Array = fixed_field()
    accepted_step_size: Array = fixed_field()
    accepted_ledgers: tuple[AcceptedConservationIntegralLedger, ...] = fixed_field()
    edge_accepted_ledgers: tuple[AcceptedConservationIntegralLedger, ...] = fixed_field()
    flux_registers: tuple[FluxRegister, ...] = fixed_field()
    maximum_rate: Array = fixed_field()
    precision_evidence: PrecisionEvidenceEnvelope = fixed_field()
    composite_conservation_defect: Array = fixed_field()
    failed_level: Array = fixed_field()
    failed_phase: Array = fixed_field()
    topology_status: Array = fixed_field()
    topology_event_request: FiniteVolumeTopologyEventRequest | None
    successor_runtime: PreparedBlockAMRRuntime | None
    synchronization_order: tuple[int, ...] = eqx.field(static=True)
    stage_times: tuple[tuple[Array, Array, Array], ...] = fixed_field()
    level_attempt_order: tuple[int, ...] = eqx.field(static=True)
    temporal_method_id: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class _BlockAMREdgeRoute(StrictModule, NonTrainableState):
    """Host-prepared mapping from fine outward faces to one coarse route."""

    level: int = eqx.field(static=True)
    coarse_block_id: str = eqx.field(static=True)
    fine_block_id: str = eqx.field(static=True)
    fine_to_coarse: Array
    orientation_factor: Array
    interface_mask: Array
    route_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coarse: Any,
        fine: Any,
        coarse_spacing: Sequence[float],
        /,
    ):
        if coarse.level + 1 != fine.level or coarse.axis != fine.axis:
            raise ValueError(
                "Coarse/fine face routes must describe one adjacent edge/axis."
            )
        coarse_coordinates = np.asarray(coarse.coordinates, dtype=np.float64)
        fine_coordinates = np.asarray(fine.coordinates, dtype=np.float64)
        if coarse_coordinates.ndim != 2 or fine_coordinates.ndim != 2:
            raise ValueError("Coarse/fine face coordinates must be rank-two arrays.")
        if coarse_coordinates.shape[1] != fine_coordinates.shape[1]:
            raise ValueError("Coarse/fine face coordinate dimensions changed.")
        dimension = coarse_coordinates.shape[1]
        spacing = np.asarray(tuple(coarse_spacing), dtype=np.float64)
        if (
            spacing.shape != (dimension,)
            or np.any(~np.isfinite(spacing))
            or np.any(spacing <= 0.0)
        ):
            raise ValueError("Coarse interface spacing must be finite and positive.")
        mapping = np.full((fine_coordinates.shape[0],), -1, dtype=np.int32)
        for fine_face, point in enumerate(fine_coordinates):
            scale = max(1.0, float(np.max(np.abs(point), initial=0.0)))
            tolerance = 256.0 * np.finfo(np.float64).eps * scale
            normal = np.abs(coarse_coordinates[:, coarse.axis] - point[coarse.axis])
            candidates = normal <= tolerance
            for axis in range(dimension):
                if axis == coarse.axis:
                    continue
                candidates &= np.abs(coarse_coordinates[:, axis] - point[axis]) <= (
                    0.5 * spacing[axis] + tolerance
                )
            matches = np.flatnonzero(candidates)
            if matches.size != 1:
                raise ValueError("Fine interface face does not map to one coarse face.")
            mapping[fine_face] = int(matches[0])
        if coarse_coordinates.shape[0] and not np.array_equal(
            np.unique(mapping), np.arange(coarse_coordinates.shape[0], dtype=np.int32)
        ):
            raise ValueError("Every coarse interface face must receive fine subfaces.")
        coarse_orientation = np.asarray(coarse.orientation, dtype=np.float64)
        fine_orientation = np.asarray(fine.orientation, dtype=np.float64)
        if np.any(np.abs(coarse_orientation) != 1.0) or np.any(
            np.abs(fine_orientation) != 1.0
        ):
            raise ValueError("AMR interface orientations must be unit signs.")
        factors = coarse_orientation[mapping] / fine_orientation
        mask = np.ones((coarse_coordinates.shape[0],), dtype=np.bool_)
        self.level = int(coarse.level)
        self.coarse_block_id = coarse.block_id
        self.fine_block_id = fine.block_id
        self.fine_to_coarse = jnp.asarray(mapping)
        self.orientation_factor = jnp.asarray(factors)
        self.interface_mask = jnp.asarray(mask)
        self.route_plan_id = canonical_fingerprint(
            {
                "kind": "block-amr-edge-route",
                "level": int(coarse.level),
                "coarse": coarse.block_id,
                "fine": fine.block_id,
                "mapping": mapping.tolist(),
                "orientation": factors.tolist(),
            }
        )

    def restrict(self, fine_flux: Array, /) -> Array:
        values = jnp.asarray(fine_flux)
        component_rank = values.ndim - 1
        factor = self.orientation_factor.reshape(
            self.orientation_factor.shape + (1,) * component_rank
        )
        target = jnp.zeros(
            (self.interface_mask.size,) + values.shape[1:], dtype=values.dtype
        )
        return target.at[self.fine_to_coarse].add(values * factor)


SpecialistSynchronization = Callable[
    [int, BlockHierarchyState, Array, Array, Any],
    BlockHierarchyState | tuple[BlockHierarchyState, ArrayLike],
]


class BlockAMRRuntimePlan(StrictModule):
    """Solver-owned fixed-topology block finite-volume advancement policy."""

    finite_volume: BlockAMRFiniteVolumePlan
    schedule: AMRTimeSchedulePlan
    specialist_synchronization: SpecialistSynchronization | None = eqx.field(static=True)
    specialist_id: str | None = eqx.field(static=True)
    indicator: Callable[..., Any] | None = eqx.field(static=True)
    indicator_id: str | None = eqx.field(static=True)
    topology_transaction: Callable[..., Any] | None = eqx.field(static=True)
    topology_transaction_id: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        finite_volume: BlockAMRFiniteVolumePlan,
        schedule: AMRTimeSchedulePlan | None = None,
        /,
        *,
        subcycling: bool | None = None,
        specialist_synchronization: SpecialistSynchronization | None = None,
        specialist_id: str | None = None,
        indicator: Callable[..., Any] | None = None,
        indicator_id: str | None = None,
        topology_transaction: Callable[..., Any] | None = None,
        topology_transaction_id: str | None = None,
    ):
        if not isinstance(finite_volume, BlockAMRFiniteVolumePlan):
            raise TypeError("finite_volume must be BlockAMRFiniteVolumePlan.")
        if subcycling is not None and not isinstance(subcycling, bool):
            raise TypeError("subcycling must be boolean or None.")
        schedule_ = (
            AMRTimeSchedulePlan(
                finite_volume.hierarchy,
                subcycling=True if subcycling is None else subcycling,
            )
            if schedule is None
            else schedule
        )
        if not isinstance(schedule_, AMRTimeSchedulePlan) or (
            schedule_.hierarchy.prepared_id != finite_volume.hierarchy.prepared_id
        ):
            raise ValueError("AMR time schedule must belong to the FV hierarchy.")
        if (
            schedule is not None
            and subcycling is not None
            and schedule_.subcycling != subcycling
        ):
            raise ValueError("Explicit schedule and subcycling selection disagree.")

        def identified(
            callback: Callable[..., Any] | None,
            identifier: str | None,
            name: str,
        ) -> tuple[Callable[..., Any] | None, str | None]:
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable or None.")
            value = None if identifier is None else str(identifier)
            if (callback is None) != (value is None) or value == "":
                raise ValueError(f"{name} requires exactly one non-empty identity.")
            return callback, value

        specialist, specialist_identity = identified(
            specialist_synchronization,
            specialist_id,
            "specialist_synchronization",
        )
        indicator_, indicator_identity = identified(
            indicator,
            indicator_id,
            "indicator",
        )
        transaction, transaction_identity = identified(
            topology_transaction,
            topology_transaction_id,
            "topology_transaction",
        )
        if transaction is not None and indicator_ is None:
            raise ValueError(
                "A topology transaction requires an accepted-step indicator."
            )
        self.finite_volume = finite_volume
        self.schedule = schedule_
        self.specialist_synchronization = specialist
        self.specialist_id = specialist_identity
        self.indicator = indicator_
        self.indicator_id = indicator_identity
        self.topology_transaction = transaction
        self.topology_transaction_id = transaction_identity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "block-amr-runtime-plan",
                "finite_volume": finite_volume.plan_id,
                "schedule": schedule_.schedule_id,
                "specialist": specialist_identity,
                "indicator": indicator_identity,
                "topology_transaction": transaction_identity,
            }
        )

    def prepare(self, topology: BlockHierarchyTopology, /) -> PreparedBlockAMRRuntime:
        return PreparedBlockAMRRuntime(self, topology)


class PreparedBlockAMRRuntime(StrictModule):
    """Prepared N-level SSPRK runtime for one immutable topology epoch."""

    plan: BlockAMRRuntimePlan
    dynamics: PreparedBlockAMRFiniteVolumeDynamics
    conservation: BlockAMRConservationPlan
    edge_routes: tuple[_BlockAMREdgeRoute, ...]
    covered_cell_masks: tuple[Array, ...] = fixed_field()
    topology_artifacts: FiniteVolumeTopologyArtifacts
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: BlockAMRRuntimePlan, topology: BlockHierarchyTopology, /):
        if not isinstance(plan, BlockAMRRuntimePlan):
            raise TypeError("plan must be BlockAMRRuntimePlan.")
        if not isinstance(topology, BlockHierarchyTopology) or (
            topology.plan.plan_id != plan.finite_volume.hierarchy.plan.hierarchy.plan_id
        ):
            raise ValueError("Runtime topology does not belong to the hierarchy plan.")
        dynamics = plan.finite_volume.prepare(topology)
        conservation = BlockAMRConservationPlan(
            plan.finite_volume.hierarchy,
            topology,
            precision=plan.finite_volume.precision,
        )
        routes = {route.block_id: route for route in dynamics.face_routes}
        edge_routes = tuple(
            _BlockAMREdgeRoute(
                routes[coarse_id],
                routes[fine_id],
                topology.plan.level_spacings[routes[coarse_id].level],
            )
            for coarse_id, fine_id in dynamics.coarse_fine_route_pairs
        )
        covered = tuple(
            conservation.covered_cell_mask(level)
            for level in range(len(topology.plan.levels) - 1)
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-block-amr-runtime",
                "plan": plan.plan_id,
                "epoch": topology.epoch.epoch_id,
                "dynamics": dynamics.dynamics_id,
                "conservation": conservation.plan_id,
                "edge_routes": [route.route_plan_id for route in edge_routes],
            }
        )
        artifacts = FiniteVolumeTopologyArtifacts(
            topology.epoch,
            prepared_id,
            topology_artifact_id=topology.topology_id,
            metrics_artifact_id=conservation.plan_id,
            operators_artifact_id=dynamics.dynamics_id,
        )
        self.plan = plan
        self.dynamics = dynamics
        self.conservation = conservation
        self.edge_routes = edge_routes
        self.covered_cell_masks = covered
        self.topology_artifacts = artifacts
        self.prepared_id = prepared_id

    def initial_state(
        self,
        hierarchy_state: BlockHierarchyState,
        /,
        *,
        time: ArrayLike = 0.0,
        accepted_step: ArrayLike = 0,
        level_accepted_steps: ArrayLike | None = None,
        journal: FiniteVolumeTopologyEventJournal | None = None,
        journal_capacity: int = 16,
    ) -> BlockAMRRuntimeState:
        self._validate_hierarchy_state(hierarchy_state)
        journal_ = (
            FiniteVolumeTopologyEventJournal.allocate(
                hierarchy_state.topology.epoch,
                self.topology_artifacts,
                capacity=journal_capacity,
                time=time,
            )
            if journal is None
            else journal
        )
        state = BlockAMRRuntimeState(
            hierarchy_state,
            journal_,
            time,
            accepted_step=accepted_step,
            level_accepted_steps=level_accepted_steps,
        )
        self._validate_state(state)
        return state

    def _validate_hierarchy_state(self, state: BlockHierarchyState, /) -> None:
        if not isinstance(state, BlockHierarchyState) or (
            state.topology.epoch.epoch_id != self.dynamics.topology.epoch.epoch_id
        ):
            raise ValueError("Block AMR payload has a stale topology epoch.")
        if state.topology.partition_id != self.dynamics.topology.partition_id:
            raise ValueError("Block AMR payload has a stale partition route.")
        self.dynamics._validate_state(state)

    def _validate_state(self, state: BlockAMRRuntimeState, /) -> None:
        if not isinstance(state, BlockAMRRuntimeState):
            raise TypeError("state must be BlockAMRRuntimeState.")
        self._validate_hierarchy_state(state.hierarchy_state)
        if (
            state.topology_journal.current_epoch_id
            != self.dynamics.topology.epoch.epoch_id
        ):
            raise ValueError("Block AMR topology journal is stale.")
        if state.topology_journal.artifact_table[-1].artifacts_id != (
            self.topology_artifacts.artifacts_id
        ):
            raise ValueError("Block AMR prepared routes do not match journal artifacts.")

    @staticmethod
    def _replace_level(
        state: BlockHierarchyState,
        level: int,
        values: Array,
        /,
    ) -> BlockHierarchyState:
        level_state = state.levels[level]
        active = level_state.metadata.active.reshape(
            (level_state.plan.maximum_blocks,) + (1,) * (values.ndim - 1)
        )
        masked = jnp.where(active, values, jnp.zeros((), dtype=values.dtype))
        levels = list(state.levels)
        levels[level] = BlockLevelState(
            level_state.plan,
            level_state.metadata,
            masked,
        )
        return BlockHierarchyState(state.topology, tuple(levels))

    @staticmethod
    def _select_hierarchy(
        condition: Array,
        accepted: BlockHierarchyState,
        rejected: BlockHierarchyState,
        /,
    ) -> BlockHierarchyState:
        levels = tuple(
            BlockLevelState(
                original.plan,
                original.metadata,
                jnp.where(condition, candidate.values, original.values),
            )
            for candidate, original in zip(accepted.levels, rejected.levels, strict=True)
        )
        return BlockHierarchyState(rejected.topology, levels)

    def _level_admissible(self, level: int, values: Array, /) -> Array:
        active = self.dynamics.topology.levels[level].active.reshape(
            (values.shape[0],) + (1,) * (values.ndim - 2)
        )
        admissible = jnp.asarray(
            self.dynamics.plan.system.admissible(values), dtype=jnp.bool_
        )
        if admissible.shape != values.shape[:-1]:
            raise ValueError("Conservation-system admissibility has the wrong shape.")
        return jnp.all(jnp.where(active, admissible, True))

    def _level_stage_ledger(
        self,
        ledger: ConservationStageLedger,
        level: int,
        /,
    ) -> ConservationStageLedger:
        route_positions = tuple(
            index
            for index, route in enumerate(self.dynamics.face_routes)
            if route.level == level
        )
        level_plan = self.dynamics.topology.plan.levels[level]
        begin = self.dynamics.level_cell_offsets[level]
        count = level_plan.maximum_blocks * prod(level_plan.block_shape)
        end = begin + count

        def level_values(values: Array) -> Array:
            return jnp.zeros_like(values).at[begin:end].set(values[begin:end])

        return ConservationStageLedger(
            tuple(ledger.blocks[index] for index in route_positions),
            level_values(ledger.source_rate),
            ledger.active_cell_mask,
            geometry_family_id=ledger.geometry_family_id,
            geometry_layout_id=ledger.geometry_layout_id,
            geometry_version=ledger.geometry_version,
            evidence_policy_id=ledger.evidence_policy_id,
            evidence_version=ledger.evidence_version,
            topology_epoch_id=ledger.topology_epoch_id,
            high_order_blocks=tuple(
                ledger.high_order_blocks[index] for index in route_positions
            ),
            low_order_blocks=tuple(
                ledger.low_order_blocks[index] for index in route_positions
            ),
            blend_factors=tuple(ledger.blend_factors[index] for index in route_positions),
            entropy_production=level_values(ledger.entropy_production),
            troubled_cell_mask=level_values(ledger.troubled_cell_mask),
            correction_level=level_values(ledger.correction_level),
            accepted=ledger.accepted,
            differentiability_policy_id=ledger.differentiability_policy_id,
        )

    @staticmethod
    def _block_by_id(
        ledger: AcceptedConservationIntegralLedger,
        block_id: str,
        /,
    ) -> AcceptedConservationFluxIntegralBlock:
        matches = tuple(block for block in ledger.blocks if block.block_id == block_id)
        if len(matches) != 1:
            raise ValueError(
                f"Accepted ledger must contain route block {block_id!r} exactly once."
            )
        return matches[0]

    def _aggregate_ledgers(
        self,
        ledgers: Sequence[AcceptedConservationIntegralLedger],
        /,
    ) -> AcceptedConservationIntegralLedger:
        values, token = self.conservation._validate_ledgers(ledgers)
        first = values[0]
        blocks = []
        for index, reference in enumerate(first.blocks):
            total = jnp.zeros_like(
                self.dynamics.plan.precision.reduction(reference.flux_integral)
            )
            for ledger in values:
                block = ledger.blocks[index]
                if (
                    block.block_id != reference.block_id
                    or block.block_kind != reference.block_kind
                    or block.route_id != reference.route_id
                ):
                    raise ValueError("Accepted ledger aggregation changed a route.")
                total = self.dynamics.plan.precision.reduction(
                    total + self.dynamics.plan.precision.reduction(block.flux_integral)
                )
            blocks.append(
                eqx.tree_at(
                    lambda value: value.flux_integral,
                    reference,
                    total,
                )
            )
        source = jnp.zeros_like(
            self.dynamics.plan.precision.reduction(first.source_integral)
        )
        source = source + jnp.asarray(0.0, dtype=source.dtype) * jnp.sum(token)
        for ledger in values:
            source = self.dynamics.plan.precision.reduction(
                source + self.dynamics.plan.precision.reduction(ledger.source_integral)
            )
        return AcceptedConservationIntegralLedger(
            tuple(blocks),
            source,
            first.active_cell_mask,
            geometry_family_id=first.geometry_family_id,
            geometry_layout_id=first.geometry_layout_id,
            stage_geometry_versions=first.stage_geometry_versions,
            start_geometry_version=first.start_geometry_version,
            end_geometry_version=values[-1].end_geometry_version,
            evidence_policy_id=first.evidence_policy_id,
            stage_evidence_versions=first.stage_evidence_versions,
            start_evidence_version=first.start_evidence_version,
            end_evidence_version=values[-1].end_evidence_version,
            start_topology_epoch_id=first.start_topology_epoch_id,
            end_topology_epoch_id=values[-1].end_topology_epoch_id,
            start_time=first.start_time,
            end_time=values[-1].end_time,
            accepted_step=values[-1].accepted_step,
        )

    @staticmethod
    def _mask_ledger(
        ledger: AcceptedConservationIntegralLedger,
        accepted: Array,
        /,
    ) -> AcceptedConservationIntegralLedger:
        blocks = tuple(
            eqx.tree_at(
                lambda value: value.flux_integral,
                reference,
                jnp.where(accepted, reference.flux_integral, 0.0),
            )
            for reference in ledger.blocks
        )
        return AcceptedConservationIntegralLedger(
            blocks,
            jnp.where(accepted, ledger.source_integral, 0.0),
            ledger.active_cell_mask,
            geometry_family_id=ledger.geometry_family_id,
            geometry_layout_id=ledger.geometry_layout_id,
            stage_geometry_versions=ledger.stage_geometry_versions,
            start_geometry_version=ledger.start_geometry_version,
            end_geometry_version=ledger.end_geometry_version,
            evidence_policy_id=ledger.evidence_policy_id,
            stage_evidence_versions=ledger.stage_evidence_versions,
            start_evidence_version=ledger.start_evidence_version,
            end_evidence_version=ledger.end_evidence_version,
            start_topology_epoch_id=ledger.start_topology_epoch_id,
            end_topology_epoch_id=ledger.end_topology_epoch_id,
            start_time=ledger.start_time,
            end_time=ledger.end_time,
            accepted_step=ledger.accepted_step,
        )

    @staticmethod
    def _mask_register(register: FluxRegister, accepted: Array, /) -> FluxRegister:
        return FluxRegister(
            jnp.where(accepted, register.coarse_flux, 0.0),
            jnp.where(accepted, register.fine_flux, 0.0),
            register.interface_mask,
            accumulated_time=register.accumulated_time,
            orientation=register.orientation,
            refinement_ratio=register.refinement_ratio,
            register_id=register.register_id,
            owner_id=register.owner_id,
        )

    def _composite_integral(self, state: BlockHierarchyState, /) -> Array:
        total = jnp.zeros(
            (self.dynamics.plan.system.component_count,),
            dtype=self.dynamics.plan.precision.reduction_dtype,
        )
        for level, level_state in enumerate(state.levels):
            values = self.dynamics.plan.precision.reduction(level_state.safe_values())
            active = level_state.metadata.active.reshape(
                (level_state.plan.maximum_blocks,)
                + (1,) * len(level_state.plan.block_shape)
            )
            owned = jnp.broadcast_to(active, values.shape[:-1])
            if level < len(self.covered_cell_masks):
                owned = owned & ~self.covered_cell_masks[level]
            volume = prod(state.plan.level_spacings[level])
            total = total + jnp.sum(
                jnp.where(owned[..., None], values * volume, 0.0),
                axis=tuple(range(values.ndim - 1)),
            )
        return total

    def _topology_event(
        self,
        state: BlockAMRRuntimeState,
        args: Any,
        /,
    ) -> tuple[
        BlockAMRRuntimeState,
        FiniteVolumeTopologyEventRequest | None,
        PreparedBlockAMRRuntime | None,
        Array,
    ]:
        indicator = self.plan.indicator
        if indicator is None:
            return state, None, None, jnp.asarray(-1, dtype=jnp.int32)
        request = indicator(state, args)
        if request is None:
            return state, None, None, jnp.asarray(-1, dtype=jnp.int32)
        if not isinstance(request, FiniteVolumeTopologyEventRequest):
            raise TypeError("Block AMR indicator must return a topology request or None.")
        if request.input_epoch_id != self.dynamics.topology.epoch.epoch_id:
            raise ValueError("Block AMR indicator returned a stale topology request.")
        callback = self.plan.topology_transaction
        if callback is None:
            journal = state.topology_journal.append_requested(
                request,
                state.accepted_step,
                state.time,
            )
            updated = BlockAMRRuntimeState(
                state.hierarchy_state,
                journal,
                state.time,
                accepted_step=state.accepted_step,
                level_accepted_steps=state.level_accepted_steps,
                last_status=state.last_status,
            )
            status = jnp.where(
                journal.overflowed,
                int(TopologyEventStatus.FAILED_RESOURCE_LIMIT),
                int(TopologyEventStatus.PENDING),
            )
            return updated, request, None, status
        transaction = FiniteVolumeTopologyEventTransaction(
            state.topology_journal,
            (request,),
            state.accepted_step,
            state.time,
            accepted=True,
        )
        outcome = callback(transaction, state, args)
        if not isinstance(outcome, FiniteVolumeTopologyEventTransactionResult):
            raise TypeError(
                "topology_transaction must return FiniteVolumeTopologyEventTransactionResult."
            )
        successor = None
        hierarchy_state = state.hierarchy_state
        if outcome.committed:
            if not isinstance(outcome.content_state, BlockHierarchyState):
                raise TypeError(
                    "Committed block topology must return BlockHierarchyState."
                )
            hierarchy_state = outcome.content_state
            successor = self.plan.prepare(hierarchy_state.topology)
            if outcome.journal.artifact_table[-1].artifacts_id != (
                successor.topology_artifacts.artifacts_id
            ):
                raise ValueError("Committed block topology artifacts are stale.")
        updated = BlockAMRRuntimeState(
            hierarchy_state,
            outcome.journal,
            state.time,
            accepted_step=state.accepted_step,
            level_accepted_steps=state.level_accepted_steps,
            last_status=state.last_status,
        )
        status = (
            int(TopologyEventStatus.SUCCESS)
            if outcome.committed
            else int(outcome.failure_status or TopologyEventStatus.FAILED)
        )
        return updated, request, successor, jnp.asarray(status, dtype=jnp.int32)

    def advance(
        self,
        state: BlockAMRRuntimeState,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> BlockAMRAdvanceResult:
        """Attempt one exact root interval and roll the complete hierarchy back on failure."""
        self._validate_state(state)
        original = state.hierarchy_state
        dtype = self.dynamics.plan.precision.reduction_dtype
        time = self.dynamics.plan.precision.decision(state.time)
        attempted_dt = self.dynamics.plan.precision.decision(step_size)
        if attempted_dt.shape != ():
            raise ValueError("Block AMR step_size must be scalar.")
        interval_ok = jnp.isfinite(attempted_dt) & (attempted_dt > 0.0)
        dt = jnp.where(interval_ok, attempted_dt, jnp.asarray(1.0, dtype=dtype))

        successful = jnp.asarray(True)
        failed_level = jnp.asarray(-1, dtype=jnp.int32)
        failed_phase = jnp.asarray(int(BlockAMRAdvancePhase.SUCCESS), dtype=jnp.int32)
        working = original
        level_steps = [
            state.level_accepted_steps[level] for level in range(len(original.levels))
        ]
        accepted_ledgers: list[AcceptedConservationIntegralLedger] = []
        stage_times: list[tuple[Array, Array, Array]] = []
        level_attempt_order: list[int] = []
        edge_ledgers: list[AcceptedConservationIntegralLedger] = []
        registers: list[FluxRegister] = []
        synchronization_order: list[int] = []
        maximum_rate = jnp.asarray(0.0, dtype=dtype)

        def record(condition: ArrayLike, level: int, phase: BlockAMRAdvancePhase) -> None:
            nonlocal successful, failed_level, failed_phase
            condition_ = jnp.asarray(condition, dtype=jnp.bool_).reshape(())
            first_failure = successful & ~condition_
            failed_level = jnp.where(first_failure, level, failed_level)
            failed_phase = jnp.where(first_failure, int(phase), failed_phase)
            successful = successful & condition_

        record(interval_ok, 0, BlockAMRAdvancePhase.INVALID_INTERVAL)

        def advance_level(
            level: int,
            interval_time: Array,
            interval_dt: Array,
            parent_old: BlockHierarchyState | None,
            parent_new: BlockHierarchyState | None,
            parent_start: Array | None,
            parent_end: Array | None,
        ) -> tuple[BlockHierarchyState, AcceptedConservationIntegralLedger]:
            nonlocal working, maximum_rate
            interval_initial = working
            initial_values = interval_initial.levels[level].safe_values()
            start_evidence_version = level_steps[level]
            accepted_id = start_evidence_version + jnp.asarray(1, dtype=jnp.int32)
            level_steps[level] = accepted_id
            local_ok = jnp.asarray(True)
            stage_ledgers: list[ConservationStageLedger] = []

            def stage(
                stage_time: Array,
                stage_state: BlockHierarchyState,
            ):
                nonlocal local_ok, maximum_rate
                if parent_old is None or parent_new is None:
                    coarse_old = stage_state
                    coarse_new = stage_state
                    coarse_start = stage_time
                    coarse_end = stage_time
                else:
                    coarse_old = parent_old
                    coarse_new = parent_new
                    coarse_start = parent_start
                    coarse_end = parent_end
                fill = self.dynamics.fill_patch(
                    stage_time,
                    stage_state,
                    args,
                    coarse_old=coarse_old,
                    coarse_new=coarse_new,
                    coarse_old_time=coarse_start,
                    coarse_new_time=coarse_end,
                )
                fill_ok = jnp.asarray(fill.complete, dtype=jnp.bool_).reshape(())
                record(fill_ok, level, BlockAMRAdvancePhase.FILL_PATCH)
                local_ok = local_ok & fill_ok
                result = self.dynamics.evaluate(
                    stage_time,
                    stage_state,
                    fill,
                    args,
                    geometry_version=self.dynamics.topology.epoch.index,
                    evidence_version=start_evidence_version,
                )
                level_ledger = self._level_stage_ledger(result.ledger, level)
                ledger_ok = jnp.asarray(level_ledger.accepted, dtype=jnp.bool_).reshape(
                    ()
                )
                record(ledger_ok, level, BlockAMRAdvancePhase.STAGE_LEDGER)
                local_ok = local_ok & ledger_ok
                rate_ok = jnp.isfinite(result.maximum_rate)
                residual_ok = jnp.all(jnp.isfinite(result.residuals[level]))
                finite = rate_ok & residual_ok
                record(finite, level, BlockAMRAdvancePhase.NONFINITE)
                local_ok = local_ok & finite
                maximum_rate = jnp.maximum(maximum_rate, result.maximum_rate)
                stage_ledgers.append(level_ledger)
                return result

            first = stage(interval_time, interval_initial)
            first_candidate = self.dynamics.plan.precision.storage(
                self.dynamics.plan.precision.reduction(initial_values)
                + self.dynamics.plan.precision.reduction(
                    interval_dt * first.residuals[level]
                )
            )
            first_finite = jnp.all(jnp.isfinite(first_candidate))
            first_admissible = self._level_admissible(level, first_candidate)
            record(first_finite, level, BlockAMRAdvancePhase.NONFINITE)
            record(first_admissible, level, BlockAMRAdvancePhase.ADMISSIBILITY)
            first_ok = first_finite & first_admissible
            local_ok = local_ok & first_ok
            first_values = jnp.where(first_ok, first_candidate, initial_values)
            first_state = self._replace_level(interval_initial, level, first_values)

            second_time = interval_time + interval_dt
            second = stage(second_time, first_state)
            second_candidate = self.dynamics.plan.precision.storage(
                0.75 * self.dynamics.plan.precision.reduction(initial_values)
                + 0.25
                * (
                    self.dynamics.plan.precision.reduction(first_values)
                    + self.dynamics.plan.precision.reduction(
                        interval_dt * second.residuals[level]
                    )
                )
            )
            second_finite = jnp.all(jnp.isfinite(second_candidate))
            second_admissible = self._level_admissible(level, second_candidate)
            record(second_finite, level, BlockAMRAdvancePhase.NONFINITE)
            record(second_admissible, level, BlockAMRAdvancePhase.ADMISSIBILITY)
            second_ok = second_finite & second_admissible
            local_ok = local_ok & second_ok
            second_values = jnp.where(second_ok, second_candidate, initial_values)
            second_state = self._replace_level(interval_initial, level, second_values)

            third_time = interval_time + 0.5 * interval_dt
            third = stage(third_time, second_state)
            final_candidate = self.dynamics.plan.precision.storage(
                (1.0 / 3.0) * self.dynamics.plan.precision.reduction(initial_values)
                + (2.0 / 3.0)
                * (
                    self.dynamics.plan.precision.reduction(second_values)
                    + self.dynamics.plan.precision.reduction(
                        interval_dt * third.residuals[level]
                    )
                )
            )
            final_finite = jnp.all(jnp.isfinite(final_candidate))
            final_admissible = self._level_admissible(level, final_candidate)
            record(final_finite, level, BlockAMRAdvancePhase.NONFINITE)
            record(final_admissible, level, BlockAMRAdvancePhase.ADMISSIBILITY)
            final_ok = final_finite & final_admissible
            local_ok = local_ok & final_ok
            final_values = jnp.where(local_ok, final_candidate, initial_values)
            parent_endpoint = self._replace_level(
                interval_initial,
                level,
                final_values,
            )
            ledger = AcceptedConservationIntegralLedger.integrate_ssprk33(
                stage_ledgers[0],
                stage_ledgers[1],
                stage_ledgers[2],
                interval_dt,
                start_geometry_version=self.dynamics.topology.epoch.index,
                end_geometry_version=self.dynamics.topology.epoch.index,
                start_evidence_version=start_evidence_version,
                end_evidence_version=accepted_id,
                start_topology_epoch_id=self.dynamics.topology.epoch.epoch_id,
                end_topology_epoch_id=self.dynamics.topology.epoch.epoch_id,
                start_time=interval_time,
                end_time=interval_time + interval_dt,
                accepted_step=accepted_id,
            )
            accepted_ledgers.append(ledger)
            stage_times.append(
                (
                    interval_time,
                    interval_time + interval_dt,
                    interval_time + 0.5 * interval_dt,
                )
            )
            level_attempt_order.append(level)
            working = parent_endpoint
            if level == len(working.levels) - 1:
                return working, ledger

            substeps = self.plan.schedule.edge_substeps[level]
            child_dt = interval_dt / substeps
            child_ledgers: list[AcceptedConservationIntegralLedger] = []
            for substep in range(substeps):
                child_time = interval_time + substep * child_dt
                working, child_ledger = advance_level(
                    level + 1,
                    child_time,
                    child_dt,
                    interval_initial,
                    parent_endpoint,
                    interval_time,
                    interval_time + interval_dt,
                )
                child_ledgers.append(child_ledger)
            aggregated = self._aggregate_ledgers(tuple(child_ledgers))
            edge_ledgers.append(aggregated)

            edge_values = tuple(item.safe_values() for item in working.levels)
            for route in self.edge_routes:
                if route.level != level:
                    continue
                coarse_block = self._block_by_id(ledger, route.coarse_block_id)
                fine_block = self._block_by_id(child_ledgers[0], route.fine_block_id)
                register = self.conservation.flux_register(
                    ledger,
                    tuple(child_ledgers),
                    coarse_block.route_id,
                    fine_block.route_id,
                    route.restrict,
                    route.interface_mask,
                )
                registers.append(register)
                edge_values = self.conservation.reflux(edge_values, register)
            for edge_level, values in enumerate(edge_values):
                working = self._replace_level(working, edge_level, values)
            reflux_finite = jnp.all(
                jnp.stack(
                    tuple(
                        jnp.all(jnp.isfinite(item.safe_values()))
                        for item in working.levels
                    )
                )
            )
            record(reflux_finite, level, BlockAMRAdvancePhase.REFLUX)

            restricted = self.conservation.restrict_covered(
                working.levels[level].safe_values(),
                working.levels[level + 1].safe_values(),
                level,
            )
            restriction_finite = jnp.all(jnp.isfinite(restricted))
            restriction_admissible = self._level_admissible(level, restricted)
            record(restriction_finite, level, BlockAMRAdvancePhase.RESTRICTION)
            record(restriction_admissible, level, BlockAMRAdvancePhase.ADMISSIBILITY)
            restricted = jnp.where(
                restriction_finite & restriction_admissible,
                restricted,
                working.levels[level].safe_values(),
            )
            working = self._replace_level(working, level, restricted)

            specialist = self.plan.specialist_synchronization
            if specialist is not None:
                specialist_result = specialist(
                    level,
                    working,
                    interval_time + interval_dt,
                    interval_dt,
                    args,
                )
                if isinstance(specialist_result, BlockHierarchyState):
                    specialist_state = specialist_result
                    specialist_ok = jnp.asarray(True)
                elif (
                    isinstance(specialist_result, tuple)
                    and len(specialist_result) == 2
                    and isinstance(specialist_result[0], BlockHierarchyState)
                ):
                    specialist_state = specialist_result[0]
                    specialist_ok = jnp.asarray(
                        specialist_result[1], dtype=jnp.bool_
                    ).reshape(())
                else:
                    raise TypeError(
                        "specialist_synchronization must return BlockHierarchyState or (BlockHierarchyState, accepted)."
                    )
                if specialist_state.topology.epoch.epoch_id != (
                    working.topology.epoch.epoch_id
                ):
                    raise ValueError("Specialist synchronization changed topology epoch.")
                specialist_finite = jnp.all(
                    jnp.stack(
                        tuple(
                            jnp.all(jnp.isfinite(item.safe_values()))
                            for item in specialist_state.levels
                        )
                    )
                )
                specialist_ok = specialist_ok & specialist_finite
                record(
                    specialist_ok,
                    level,
                    BlockAMRAdvancePhase.SPECIALIST_SYNCHRONIZATION,
                )
                working = self._select_hierarchy(
                    specialist_ok,
                    specialist_state,
                    working,
                )
            synchronization_order.append(level)
            return working, ledger

        working, _ = advance_level(0, time, dt, None, None, None, None)
        accepted = successful
        selected_hierarchy = self._select_hierarchy(accepted, working, original)
        next_time = jnp.where(accepted, time + attempted_dt, state.time)
        next_root_step = jnp.where(
            accepted,
            state.accepted_step + jnp.asarray(1, dtype=jnp.int32),
            state.accepted_step,
        )
        candidate_level_steps = jnp.stack(tuple(level_steps))
        next_level_steps = jnp.where(
            accepted,
            candidate_level_steps,
            state.level_accepted_steps,
        )
        next_status = jnp.where(
            accepted,
            int(BlockAMRAdvancePhase.SUCCESS),
            failed_phase,
        )
        next_state = BlockAMRRuntimeState(
            selected_hierarchy,
            state.topology_journal,
            next_time,
            accepted_step=next_root_step,
            level_accepted_steps=next_level_steps,
            last_status=next_status,
        )
        request = None
        successor = None
        topology_status = jnp.asarray(-1, dtype=jnp.int32)
        if self.plan.indicator is not None and bool(np.asarray(accepted)):
            next_state, request, successor, topology_status = self._topology_event(
                next_state,
                args,
            )
        original_composite = self._composite_integral(original)
        final_composite = self._composite_integral(next_state.hierarchy_state)
        composite_defect = self.dynamics.plan.precision.reduction(
            final_composite - original_composite
        )
        return BlockAMRAdvanceResult(
            runtime_state=next_state,
            accepted=accepted,
            attempted_step_size=attempted_dt,
            accepted_step_size=jnp.where(accepted, attempted_dt, 0.0),
            accepted_ledgers=tuple(
                self._mask_ledger(ledger, accepted) for ledger in accepted_ledgers
            ),
            edge_accepted_ledgers=tuple(
                self._mask_ledger(ledger, accepted) for ledger in edge_ledgers
            ),
            flux_registers=tuple(
                self._mask_register(register, accepted) for register in registers
            ),
            maximum_rate=jnp.where(accepted, maximum_rate, 0.0),
            composite_conservation_defect=composite_defect,
            failed_level=jnp.where(accepted, -1, failed_level),
            precision_evidence=self.dynamics.plan.precision.evidence(),
            failed_phase=jnp.where(
                accepted,
                int(BlockAMRAdvancePhase.SUCCESS),
                failed_phase,
            ),
            topology_status=topology_status,
            topology_event_request=request,
            successor_runtime=successor,
            stage_times=tuple(stage_times),
            level_attempt_order=tuple(level_attempt_order),
            synchronization_order=tuple(synchronization_order),
            temporal_method_id=self.plan.schedule.temporal_method_id,
            schedule_id=self.plan.schedule.schedule_id,
            prepared_id=self.prepared_id,
        )


__all__ = [
    "AMRTimeSchedulePlan",
    "BlockAMRAdvancePhase",
    "BlockAMRAdvanceResult",
    "BlockAMRRuntimePlan",
    "BlockAMRRuntimeState",
    "PreparedBlockAMRRuntime",
]
