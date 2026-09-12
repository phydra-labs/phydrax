#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._conservation_boundary import PrescribedNormalFluxBoundary
from .._conservation_ledger import (
    ConservationStageFluxRateBlock,
    ConservationStageLedger,
)
from ..amr import (
    BlockHierarchyState,
    BlockHierarchyTopology,
    FDAMRFillPatchResult,
    FDAMRFillPatchWorkspace,
    PreparedFDAMRHierarchy,
)
from ._boundary import FiniteVolumeBoundarySet
from ._dynamics import (
    evaluate_cartesian_numerical_flux,
    FiniteVolumeMethodPlan,
    reconstruct_cartesian_ghosted_axis,
    SourceFunction,
)
from ._precision import FiniteVolumePrecisionPolicy
from ._riemann import AbstractNumericalFluxPlan


class _BlockAMRFaceRoute(StrictModule, NonTrainableState):
    level: int = eqx.field(static=True)
    axis: int = eqx.field(static=True)
    face_indices: Array
    owner_cells: Array
    neighbour_cells: Array
    orientation: Array
    coordinates: Array
    lower_positions: tuple[int, ...] = eqx.field(static=True)
    upper_positions: tuple[int, ...] = eqx.field(static=True)
    block_id: str = eqx.field(static=True)
    block_kind: str = eqx.field(static=True)

    def __init__(
        self,
        level: int,
        axis: int,
        face_indices: np.ndarray,
        owner_cells: np.ndarray,
        neighbour_cells: np.ndarray,
        orientation: np.ndarray,
        coordinates: np.ndarray,
        sides: np.ndarray,
        block_id: str,
        block_kind: str,
        /,
    ):
        self.level = int(level)
        self.axis = int(axis)
        self.face_indices = jnp.asarray(face_indices, dtype=jnp.int32)
        self.owner_cells = jnp.asarray(owner_cells, dtype=jnp.int32)
        self.neighbour_cells = jnp.asarray(neighbour_cells, dtype=jnp.int32)
        self.orientation = jnp.asarray(orientation)
        self.coordinates = jnp.asarray(coordinates)
        self.lower_positions = tuple(int(value) for value in np.flatnonzero(sides == -1))
        self.upper_positions = tuple(int(value) for value in np.flatnonzero(sides == 1))
        self.block_id = str(block_id)
        self.block_kind = str(block_kind)


class BlockAMRFiniteVolumeStageResult(StrictModule):
    """Block-shaped residuals and their authoritative routed stage ledger."""

    residuals: tuple[Array, ...]
    ledger: ConservationStageLedger
    maximum_rate: Array
    precision_evidence: PrecisionEvidenceEnvelope


class BlockAMRFiniteVolumePlan(StrictModule, NonTrainableState):
    """Cartesian cell-centred FV method bound to one prepared fixed-block hierarchy."""

    hierarchy: PreparedFDAMRHierarchy
    system: Any
    method: FiniteVolumeMethodPlan
    boundaries: FiniteVolumeBoundarySet
    precision: FiniteVolumePrecisionPolicy
    source: SourceFunction | None = eqx.field(static=True)
    source_id: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        hierarchy: PreparedFDAMRHierarchy,
        system: Any,
        method: FiniteVolumeMethodPlan,
        boundaries: FiniteVolumeBoundarySet,
        /,
        *,
        source: SourceFunction | None = None,
        source_id: str | None = None,
        precision: FiniteVolumePrecisionPolicy | None = None,
    ):
        if not isinstance(hierarchy, PreparedFDAMRHierarchy):
            raise TypeError("hierarchy must be PreparedFDAMRHierarchy.")
        if not isinstance(method, FiniteVolumeMethodPlan):
            raise TypeError("method must be FiniteVolumeMethodPlan.")
        if not isinstance(method.interface_solver, AbstractNumericalFluxPlan):
            raise TypeError("Block AMR finite volumes require a numerical-flux method.")
        if method.viscous is not None:
            raise ValueError(
                "Block AMR currently accepts inviscid finite-volume methods only."
            )
        if not isinstance(boundaries, FiniteVolumeBoundarySet):
            raise TypeError("boundaries must be FiniteVolumeBoundarySet.")
        hierarchy_plan = hierarchy.plan.hierarchy
        if boundaries.axis_names != hierarchy_plan.grid.axis_names:
            raise ValueError("Boundary axes must match the block hierarchy geometry.")
        if int(system.dimension) != len(hierarchy_plan.grid.shape):
            raise ValueError("Conservation-system dimension must match AMR block rank.")
        for periodic, pair in zip(
            hierarchy_plan.periodic_axes, boundaries.pairs, strict=True
        ):
            if periodic != (pair is None):
                raise ValueError(
                    "Periodic axes use no physical pair; bounded axes require one."
                )
        required_halo = int(method.reconstruction.ghost_width)
        if any(
            width < required_halo
            for level in hierarchy_plan.levels
            for width in level.halo_width
        ):
            raise ValueError(
                "Every block halo must contain the reconstruction ghost width."
            )
        precision_ = (
            FiniteVolumePrecisionPolicy(hierarchy.plan.precision.field_dtype)
            if precision is None
            else precision
        )
        if not isinstance(precision_, FiniteVolumePrecisionPolicy):
            raise TypeError("precision must be FiniteVolumePrecisionPolicy.")
        if precision_.storage_dtype != hierarchy.plan.precision.field_dtype:
            raise ValueError(
                "Finite-volume storage precision must match the prepared hierarchy field precision."
            )
        if source is not None and not callable(source):
            raise TypeError("source must be callable or None.")
        source_identifier = None if source_id is None else str(source_id)
        if (source is None) != (source_identifier is None) or source_identifier == "":
            raise ValueError(
                "A source callable requires exactly one non-empty source_id."
            )
        self.hierarchy = hierarchy
        self.system = system
        self.method = method
        self.boundaries = boundaries
        self.precision = precision_
        self.source = source
        self.source_id = source_identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "block-amr-finite-volume-plan",
                "hierarchy": hierarchy.prepared_id,
                "system": system.system_id,
                "method": method.method_id,
                "boundaries": boundaries.boundary_set_id,
                "source": source_identifier,
                "precision": precision_.policy_id,
            }
        )

    def prepare(
        self, topology: BlockHierarchyTopology, /
    ) -> "PreparedBlockAMRFiniteVolumeDynamics":
        return PreparedBlockAMRFiniteVolumeDynamics(self, topology)


class PreparedBlockAMRFiniteVolumeDynamics(StrictModule, NonTrainableState):
    """Topology-bound block-local Cartesian finite-volume dynamics."""

    plan: BlockAMRFiniteVolumePlan
    topology: BlockHierarchyTopology
    fill_patch_plans: tuple[Any, ...]
    face_routes: tuple[_BlockAMRFaceRoute, ...]
    coarse_fine_route_pairs: tuple[tuple[str, str], ...] = eqx.field(static=True)
    physical_boundary_slots: tuple[
        tuple[tuple[tuple[int, ...], tuple[int, ...]], ...], ...
    ] = eqx.field(static=True)
    cell_coordinates: tuple[Array, ...]
    active_cell_mask: Array
    level_cell_offsets: tuple[int, ...] = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: BlockAMRFiniteVolumePlan,
        topology: BlockHierarchyTopology,
        /,
    ):
        if not isinstance(plan, BlockAMRFiniteVolumePlan):
            raise TypeError("plan must be BlockAMRFiniteVolumePlan.")
        hierarchy_plan = plan.hierarchy.plan.hierarchy
        if not isinstance(topology, BlockHierarchyTopology) or (
            topology.plan.plan_id != hierarchy_plan.plan_id
        ):
            raise ValueError("Topology does not belong to the finite-volume hierarchy.")
        routes, offsets = self._build_routes(plan, topology)
        coordinates = tuple(
            self._level_cell_coordinates(topology, level)
            for level in range(len(hierarchy_plan.levels))
        )
        active_chunks = []
        for level_plan, metadata in zip(
            hierarchy_plan.levels, topology.levels, strict=True
        ):
            active_chunks.append(
                np.repeat(
                    np.asarray(metadata.active, dtype=bool), prod(level_plan.block_shape)
                )
            )
        active = np.concatenate(active_chunks)
        self.plan = plan
        self.topology = topology
        self.fill_patch_plans = plan.hierarchy.prepare_fill_patch(topology)
        self.face_routes = routes
        route_ids = {route.block_id for route in routes}
        self.coarse_fine_route_pairs = tuple(
            (coarse, fine)
            for level in range(len(hierarchy_plan.levels) - 1)
            for axis in range(len(hierarchy_plan.grid.shape))
            for coarse, fine in (
                (
                    f"block-amr:transition-{level}-{level + 1}:axis-{axis}:coarse",
                    f"block-amr:transition-{level}-{level + 1}:axis-{axis}:fine",
                ),
            )
            if coarse in route_ids and fine in route_ids
        )
        physical_slots = []
        for metadata, lattice in zip(
            topology.levels,
            hierarchy_plan.block_lattice_shapes,
            strict=True,
        ):
            active_host = np.asarray(metadata.active, dtype=bool)
            logical_host = np.asarray(metadata.logical_indices, dtype=np.int32)
            level_slots = []
            for axis in range(len(hierarchy_plan.grid.shape)):
                level_slots.append(
                    (
                        tuple(
                            int(slot)
                            for slot in np.flatnonzero(active_host)
                            if logical_host[slot, axis] == 0
                        ),
                        tuple(
                            int(slot)
                            for slot in np.flatnonzero(active_host)
                            if logical_host[slot, axis] == lattice[axis] - 1
                        ),
                    )
                )
            physical_slots.append(tuple(level_slots))
        self.physical_boundary_slots = tuple(physical_slots)
        self.cell_coordinates = coordinates
        self.active_cell_mask = jnp.asarray(active)
        self.level_cell_offsets = offsets
        self.dynamics_id = canonical_fingerprint(
            {
                "kind": "prepared-block-amr-finite-volume-dynamics",
                "plan": plan.plan_id,
                "epoch": topology.epoch.epoch_id,
                "routes": [
                    {
                        "block": route.block_id,
                        "owners": array_tree_fingerprint(np.asarray(route.owner_cells)),
                        "neighbours": array_tree_fingerprint(
                            np.asarray(route.neighbour_cells)
                        ),
                    }
                    for route in routes
                ],
            }
        )

    @staticmethod
    def _level_cell_coordinates(topology: BlockHierarchyTopology, level: int, /) -> Array:
        hierarchy = topology.plan
        level_plan = hierarchy.levels[level]
        metadata = topology.levels[level]
        spacing = np.asarray(hierarchy.level_spacings[level], dtype=float)
        lower = np.asarray(
            [axis.bounds[0] for axis in hierarchy.grid.structured_axes], dtype=float
        )
        coordinates = np.zeros(
            (level_plan.maximum_blocks,) + level_plan.block_shape + (len(spacing),),
            dtype=float,
        )
        logical = np.asarray(metadata.logical_indices)
        for slot in range(level_plan.maximum_blocks):
            if not bool(np.asarray(metadata.active)[slot]):
                continue
            origin = logical[slot] * np.asarray(level_plan.block_shape)
            for local in np.ndindex(level_plan.block_shape):
                coordinates[(slot,) + local] = (
                    lower + (origin + np.asarray(local) + 0.5) * spacing
                )
        return jnp.asarray(coordinates)

    @staticmethod
    def _build_routes(
        plan: BlockAMRFiniteVolumePlan,
        topology: BlockHierarchyTopology,
        /,
    ) -> tuple[tuple[_BlockAMRFaceRoute, ...], tuple[int, ...]]:
        hierarchy = topology.plan
        dimension = len(hierarchy.grid.shape)
        lower_bounds = np.asarray(
            [axis.bounds[0] for axis in hierarchy.grid.structured_axes], dtype=float
        )
        offsets = []
        next_offset = 0
        for level_plan in hierarchy.levels:
            offsets.append(next_offset)
            next_offset += level_plan.maximum_blocks * prod(level_plan.block_shape)
        routes: list[_BlockAMRFaceRoute] = []
        for level, (level_plan, metadata) in enumerate(
            zip(hierarchy.levels, topology.levels, strict=True)
        ):
            active = np.asarray(metadata.active, dtype=bool)
            logical = np.asarray(metadata.logical_indices, dtype=np.int32)
            neighbors = np.asarray(metadata.neighbor_slots, dtype=np.int32)
            interfaces = np.asarray(topology.interfaces[level], dtype=bool)
            spacing = np.asarray(hierarchy.level_spacings[level], dtype=float)
            lattice = hierarchy.block_lattice_shapes[level]
            block_shape = level_plan.block_shape
            block_cells = prod(block_shape)

            def cell(
                slot: int,
                local: tuple[int, ...],
                *,
                level: int = level,
                block_cells: int = block_cells,
                block_shape: tuple[int, ...] = block_shape,
            ) -> int:
                return (
                    offsets[level]
                    + slot * block_cells
                    + int(np.ravel_multi_index(local, block_shape))
                )

            def coordinate(
                slot: int,
                face: tuple[int, ...],
                axis: int,
                *,
                logical: np.ndarray = logical,
                block_shape: tuple[int, ...] = block_shape,
                spacing: np.ndarray = spacing,
            ) -> np.ndarray:
                global_face = logical[slot] * np.asarray(block_shape) + np.asarray(face)
                value = lower_bounds + (global_face + 0.5) * spacing
                value[axis] = lower_bounds[axis] + global_face[axis] * spacing[axis]
                return value

            def separates_finer_coverage(
                owner_slot: int,
                owner: tuple[int, ...],
                neighbour_slot: int,
                neighbour: tuple[int, ...],
                *,
                covered: np.ndarray | None = (
                    None
                    if level + 1 == len(hierarchy.levels)
                    else np.asarray(topology.covered_cells[level], dtype=bool)
                ),
            ) -> bool:
                if covered is None:
                    return False
                return bool(covered[(owner_slot,) + owner]) != bool(
                    covered[(neighbour_slot,) + neighbour]
                )

            for axis in range(dimension):
                bucket_names = ["same-level", "physical"]
                if level > 0:
                    bucket_names.append(f"transition-{level - 1}-{level}:fine")
                if level < len(hierarchy.levels) - 1:
                    bucket_names.append(f"transition-{level}-{level + 1}:coarse")
                buckets: dict[str, dict[str, list[Any]]] = {
                    name: {
                        "faces": [],
                        "owner": [],
                        "neighbour": [],
                        "sign": [],
                        "coordinates": [],
                        "sides": [],
                    }
                    for name in bucket_names
                }

                def append(
                    kind: str,
                    slot: int,
                    face: tuple[int, ...],
                    owner: tuple[int, ...],
                    neighbour_slot: int,
                    neighbour: tuple[int, ...] | None,
                    sign: int,
                    *,
                    buckets: dict[str, dict[str, list[Any]]] = buckets,
                    cell=cell,
                    coordinate=coordinate,
                    axis: int = axis,
                ) -> None:
                    bucket = buckets[kind]
                    bucket["faces"].append((slot,) + face)
                    bucket["owner"].append(cell(slot, owner))
                    bucket["neighbour"].append(
                        -1 if neighbour is None else cell(neighbour_slot, neighbour)
                    )
                    bucket["sign"].append(sign)
                    bucket["coordinates"].append(coordinate(slot, face, axis))
                    bucket["sides"].append(sign if kind == "physical" else 0)

                transverse_shape = block_shape[:axis] + block_shape[axis + 1 :]
                for slot in np.flatnonzero(active):
                    for transverse in np.ndindex(transverse_shape):

                        def local_with_axis(
                            value: int,
                            *,
                            transverse: tuple[int, ...] = transverse,
                            axis: int = axis,
                        ) -> tuple[int, ...]:
                            values = list(transverse)
                            values.insert(axis, value)
                            return tuple(values)

                        for face_index in range(1, block_shape[axis]):
                            owner_local = local_with_axis(face_index - 1)
                            neighbour_local = local_with_axis(face_index)
                            kind = (
                                f"transition-{level}-{level + 1}:coarse"
                                if separates_finer_coverage(
                                    int(slot),
                                    owner_local,
                                    int(slot),
                                    neighbour_local,
                                )
                                else "same-level"
                            )
                            append(
                                kind,
                                int(slot),
                                local_with_axis(face_index),
                                owner_local,
                                int(slot),
                                neighbour_local,
                                1,
                            )
                        upper_slot = int(neighbors[slot, axis, 1])
                        if upper_slot >= 0:
                            owner_local = local_with_axis(block_shape[axis] - 1)
                            neighbour_local = local_with_axis(0)
                            kind = (
                                f"transition-{level}-{level + 1}:coarse"
                                if separates_finer_coverage(
                                    int(slot),
                                    owner_local,
                                    upper_slot,
                                    neighbour_local,
                                )
                                else "same-level"
                            )
                            append(
                                kind,
                                int(slot),
                                local_with_axis(block_shape[axis]),
                                owner_local,
                                upper_slot,
                                neighbour_local,
                                1,
                            )
                        for side, face_index, owner_index in (
                            (0, 0, 0),
                            (1, block_shape[axis], block_shape[axis] - 1),
                        ):
                            if neighbors[slot, axis, side] >= 0:
                                continue
                            at_domain = not hierarchy.periodic_axes[axis] and logical[
                                slot, axis
                            ] == (0 if side == 0 else lattice[axis] - 1)
                            kind = (
                                "physical"
                                if at_domain
                                else f"transition-{level - 1}-{level}:fine"
                            )
                            if kind != "physical" and not interfaces[slot, axis, side]:
                                continue
                            append(
                                kind,
                                int(slot),
                                local_with_axis(face_index),
                                local_with_axis(owner_index),
                                -1,
                                None,
                                -1 if side == 0 else 1,
                            )
                for kind, bucket in buckets.items():
                    if not bucket["faces"]:
                        continue
                    if kind.startswith("transition-"):
                        transition, role = kind.rsplit(":", 1)
                        block_id = f"block-amr:{transition}:axis-{axis}:{role}"
                    else:
                        block_id = f"block-amr:level-{level}:axis-{axis}:{kind}"
                    routes.append(
                        _BlockAMRFaceRoute(
                            level,
                            axis,
                            np.asarray(bucket["faces"], dtype=np.int32),
                            np.asarray(bucket["owner"], dtype=np.int32),
                            np.asarray(bucket["neighbour"], dtype=np.int32),
                            np.asarray(bucket["sign"], dtype=float),
                            np.asarray(bucket["coordinates"], dtype=float),
                            np.asarray(bucket["sides"], dtype=np.int8),
                            block_id,
                            "coarse-fine" if kind.startswith("transition-") else kind,
                        )
                    )
        return tuple(routes), tuple(offsets)

    def _validate_state(self, state: BlockHierarchyState, /) -> None:
        if not isinstance(state, BlockHierarchyState) or (
            state.topology.epoch.epoch_id != self.topology.epoch.epoch_id
        ):
            raise ValueError(
                "Finite-volume state must share the prepared topology epoch."
            )
        for level_plan, level in zip(
            self.topology.plan.levels, state.levels, strict=True
        ):
            expected = (
                level_plan.maximum_blocks,
                *level_plan.block_shape,
                self.plan.system.component_count,
            )
            if level.values.shape != expected:
                raise ValueError(
                    f"Block finite-volume values must have shape {expected}."
                )
            self.plan.precision.validate_state(level.values)

    def _physical_boundary_values(
        self,
        time: ArrayLike,
        state: BlockHierarchyState,
        workspaces: tuple[FDAMRFillPatchWorkspace, ...],
        args: Any,
        /,
    ) -> tuple[Array, ...]:
        hierarchy = self.topology.plan
        output: list[Array] = []
        for level, (level_plan, workspace) in enumerate(
            zip(hierarchy.levels, workspaces, strict=True)
        ):
            safe = state.levels[level].safe_values()
            values = jnp.zeros_like(workspace.values)
            logical = self.topology.levels[level].logical_indices
            widths = level_plan.halo_width
            block_shape = level_plan.block_shape
            spacing = jnp.asarray(hierarchy.level_spacings[level], dtype=values.dtype)
            bounds = jnp.stack(
                tuple(axis.bounds for axis in hierarchy.grid.structured_axes)
            ).astype(values.dtype)
            for axis, pair in enumerate(self.plan.boundaries.pairs):
                if pair is None:
                    continue
                for side_index, (side, boundary) in enumerate(
                    ((-1, pair.lower), (1, pair.upper))
                ):
                    touching = self.physical_boundary_slots[level][axis][side_index]
                    if not touching:
                        continue
                    transverse_shape = block_shape[:axis] + block_shape[axis + 1 :]
                    interiors = []
                    coordinates = []
                    targets: list[tuple[int, tuple[int, ...]]] = []
                    for slot in touching:
                        for transverse in np.ndindex(transverse_shape):
                            local_values = list(transverse)
                            local_values.insert(
                                axis, 0 if side < 0 else block_shape[axis] - 1
                            )
                            local = tuple(local_values)
                            interiors.append(safe[(slot,) + local])
                            global_cell = logical[slot] * jnp.asarray(
                                block_shape
                            ) + jnp.asarray(local)
                            point = bounds[:, 0] + (global_cell + 0.5) * spacing
                            point = point.at[axis].set(bounds[axis, 0 if side < 0 else 1])
                            coordinates.append(point)
                            targets.append((slot, transverse))
                    interior = jnp.stack(interiors)
                    exterior = (
                        interior
                        if isinstance(boundary, PrescribedNormalFluxBoundary)
                        else boundary.exterior_state(
                            self.plan.system,
                            self.plan.precision.decision(time),
                            self.plan.precision.reconstruction(interior),
                            self.plan.precision.reconstruction(jnp.stack(coordinates)),
                            side * jnp.eye(len(block_shape))[axis],
                            axis,
                            args,
                        )
                    )
                    exterior = self.plan.precision.storage(exterior)
                    depth = widths[axis]
                    for row, (slot, transverse) in enumerate(targets):
                        for layer in range(depth):
                            padded_local = list(transverse)
                            padded_local.insert(axis, 0)
                            padded = [
                                value + widths[index]
                                for index, value in enumerate(padded_local)
                            ]
                            padded[axis] = (
                                layer
                                if side < 0
                                else widths[axis] + block_shape[axis] + layer
                            )
                            values = values.at[(slot,) + tuple(padded)].set(exterior[row])
            output.append(values)
        return tuple(output)

    def _execute_fill_patch(
        self,
        state: BlockHierarchyState,
        coarse_old: BlockHierarchyState,
        coarse_new: BlockHierarchyState,
        coarse_old_time: ArrayLike,
        coarse_new_time: ArrayLike,
        fill_time: ArrayLike,
        physical_boundary_values: tuple[Array, ...] | None,
        /,
    ) -> FDAMRFillPatchResult:
        boundaries = (
            (None,) * len(self.fill_patch_plans)
            if physical_boundary_values is None
            else physical_boundary_values
        )
        executed = tuple(
            fill.execute(
                state,
                coarse_old,
                coarse_new,
                coarse_old_time,
                coarse_new_time,
                fill_time,
                boundary,
            )
            for fill, boundary in zip(self.fill_patch_plans, boundaries, strict=True)
        )
        workspaces = tuple(value[0] for value in executed)
        requests = tuple(value[1] for value in executed)
        complete_by_level = tuple(
            jnp.all(
                workspace.valid
                | ~metadata.active.reshape(
                    (metadata.active.shape[0],) + (1,) * (workspace.valid.ndim - 1)
                )
            )
            for metadata, workspace in zip(self.topology.levels, workspaces, strict=True)
        )
        return FDAMRFillPatchResult(
            workspaces=workspaces,
            physical_boundary_requests=requests,
            complete=jnp.all(jnp.stack(complete_by_level)),
            result_id=canonical_fingerprint(
                {
                    "kind": "block-amr-fv-fill-patch-result",
                    "dynamics": self.dynamics_id,
                    "physical_values_supplied": physical_boundary_values is not None,
                }
            ),
        )

    def fill_patch(
        self,
        time: ArrayLike,
        state: BlockHierarchyState,
        args: Any = None,
        /,
        *,
        coarse_old: BlockHierarchyState | None = None,
        coarse_new: BlockHierarchyState | None = None,
        coarse_old_time: ArrayLike = 0.0,
        coarse_new_time: ArrayLike = 0.0,
    ) -> FDAMRFillPatchResult:
        """Complete FillPatch, asking FV boundary policies only for physical faces."""
        self._validate_state(state)
        old = state if coarse_old is None else coarse_old
        new = state if coarse_new is None else coarse_new
        self._validate_state(old)
        self._validate_state(new)
        initial = self._execute_fill_patch(
            state,
            old,
            new,
            coarse_old_time,
            coarse_new_time,
            time,
            None,
        )
        physical_values = self._physical_boundary_values(
            time, state, initial.workspaces, args
        )
        return self._execute_fill_patch(
            state,
            old,
            new,
            coarse_old_time,
            coarse_new_time,
            time,
            physical_values,
        )

    def _face_fluxes(
        self,
        workspace: FDAMRFillPatchWorkspace,
        level: int,
        args: Any,
        /,
    ) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
        level_plan = self.topology.plan.levels[level]
        active_cells = self.topology.levels[level].active.reshape(
            (level_plan.maximum_blocks,) + (1,) * (workspace.valid.ndim - 1)
        )
        active_values = active_cells.reshape(
            active_cells.shape + (1,) * (workspace.values.ndim - workspace.valid.ndim)
        )
        values = jnp.where(active_values, workspace.values, 0.0)
        values = eqx.error_if(
            values,
            jnp.any(active_cells & ~workspace.valid),
            "Completed FillPatch workspace has an invalid active stencil cell.",
        )
        fluxes = []
        speeds = []
        for axis, (size, width, spacing) in enumerate(
            zip(
                level_plan.block_shape,
                level_plan.halo_width,
                self.topology.plan.level_spacings[level],
                strict=True,
            )
        ):
            array_axis = axis + 1
            coordinates = (
                jnp.arange(size + 2 * width, dtype=values.dtype) - width + 0.5
            ) * spacing
            left, right = reconstruct_cartesian_ghosted_axis(
                self.plan.method,
                self.plan.system,
                self.plan.precision,
                values,
                array_axis,
                interior_cell_count=size,
                ghost_depth=width,
                periodic=False,
                axis_coordinates=coordinates,
            )
            crop: list[slice] = [slice(None)]
            for transverse, (other_size, other_width) in enumerate(
                zip(level_plan.block_shape, level_plan.halo_width, strict=True)
            ):
                crop.append(
                    slice(None)
                    if transverse == axis
                    else slice(other_width, other_width + other_size)
                )
            crop.append(slice(None))
            left = left[tuple(crop)]
            right = right[tuple(crop)]
            flux, speed = evaluate_cartesian_numerical_flux(
                self.plan.method,
                self.plan.system,
                self.plan.precision,
                left,
                right,
                axis,
                args,
            )
            fluxes.append(flux)
            speeds.append(speed)
        return tuple(fluxes), tuple(speeds)

    def evaluate(
        self,
        time: ArrayLike,
        state: BlockHierarchyState,
        fill_patch: FDAMRFillPatchResult,
        args: Any = None,
        /,
        *,
        geometry_version: ArrayLike | None = None,
        evidence_version: ArrayLike = 0,
    ) -> BlockAMRFiniteVolumeStageResult:
        self._validate_state(state)
        if not isinstance(fill_patch, FDAMRFillPatchResult):
            raise TypeError("fill_patch must be FDAMRFillPatchResult.")
        # Completion is enforced by the per-workspace validity error below; avoid
        # Python truth conversion of the dynamic aggregate flag under JIT.
        workspaces = fill_patch.workspaces
        if len(workspaces) != len(self.fill_patch_plans) or any(
            workspace.workspace_id
            != canonical_fingerprint(
                {
                    "kind": "fd-amr-fill-patch-workspace",
                    "plan": expected.plan_id,
                    "shape": list(workspace.values.shape),
                }
            )
            for workspace, expected in zip(workspaces, self.fill_patch_plans, strict=True)
        ):
            raise ValueError("FillPatch workspaces do not match this topology route.")
        by_level = tuple(
            self._face_fluxes(workspace, level, args)
            for level, workspace in enumerate(workspaces)
        )
        flattened_state = jnp.concatenate(
            tuple(
                level.safe_values().reshape((-1, self.plan.system.component_count))
                for level in state.levels
            ),
            axis=0,
        )
        blocks = []
        maximum_rate = jnp.asarray(0.0, dtype=self.plan.precision.reduction_dtype)
        for route in self.face_routes:
            fluxes, speeds = by_level[route.level]
            face_index = tuple(
                route.face_indices[:, index]
                for index in range(route.face_indices.shape[1])
            )
            flux = fluxes[route.axis][face_index]
            speed = speeds[route.axis][face_index]
            if route.block_kind == "physical":
                pair = self.plan.boundaries.pairs[route.axis]
                if pair is None:
                    raise RuntimeError("Physical FV route lost its boundary pair.")
                for side, boundary in ((-1, pair.lower), (1, pair.upper)):
                    if not isinstance(boundary, PrescribedNormalFluxBoundary):
                        continue
                    selected_positions = (
                        route.lower_positions if side < 0 else route.upper_positions
                    )
                    if not selected_positions:
                        continue
                    selected = jnp.asarray(selected_positions, dtype=jnp.int32)
                    outward = boundary.normal_flux(
                        self.plan.precision.decision(time),
                        self.plan.precision.reconstruction(
                            flattened_state[route.owner_cells[selected]]
                        ),
                        self.plan.precision.reconstruction(route.coordinates[selected]),
                        side * jnp.eye(self.plan.system.dimension)[route.axis],
                        args,
                    )
                    flux = flux.at[selected].set(side * self.plan.precision.flux(outward))
            face_measure = prod(
                spacing
                for axis, spacing in enumerate(
                    self.topology.plan.level_spacings[route.level]
                )
                if axis != route.axis
            )
            integrated = self.plan.precision.reduction(flux) * face_measure
            outward = integrated * route.orientation[:, None]
            blocks.append(
                ConservationStageFluxRateBlock(
                    outward,
                    route.owner_cells,
                    route.neighbour_cells,
                    np.ones(route.owner_cells.shape, dtype=bool),
                    route.block_id,
                    route.block_kind,
                )
            )
            cell_width = self.topology.plan.level_spacings[route.level][route.axis]
            maximum_rate = jnp.maximum(
                maximum_rate,
                jnp.max(self.plan.precision.decision(speed)) / cell_width,
            )
        source_chunks = []
        for level, level_state in enumerate(state.levels):
            safe = level_state.safe_values()
            if self.plan.source is None:
                source = jnp.zeros_like(safe)
            else:
                source = self.plan.source(
                    self.plan.precision.decision(time),
                    self.plan.precision.reconstruction(safe),
                    self.plan.precision.reconstruction(self.cell_coordinates[level]),
                    args,
                )
                source = self.plan.precision.reduction(source)
                if source.shape != safe.shape:
                    raise ValueError(
                        "Block finite-volume source must match level values."
                    )
            active = self.topology.levels[level].active.reshape(
                (level_state.plan.maximum_blocks,) + (1,) * (source.ndim - 1)
            )
            cell_volume = prod(self.topology.plan.level_spacings[level])
            source_chunks.append(
                jnp.where(active, source * cell_volume, 0.0).reshape(
                    (-1, self.plan.system.component_count)
                )
            )
        source_rate = jnp.concatenate(tuple(source_chunks), axis=0)
        version = (
            self.topology.epoch.index if geometry_version is None else geometry_version
        )
        ledger = ConservationStageLedger(
            tuple(blocks),
            source_rate,
            self.active_cell_mask,
            geometry_family_id=self.topology.plan.geometry_id,
            geometry_layout_id=self.topology.partition_id,
            geometry_version=jnp.asarray(version),
            evidence_policy_id=self.plan.precision.policy_id,
            evidence_version=jnp.asarray(evidence_version),
            topology_epoch_id=self.topology.epoch.epoch_id,
            differentiability_policy_id=self.plan.method.differentiability,
        )
        content_rate = ledger.scatter_content_rate()
        residuals = []
        for level, level_plan in enumerate(self.topology.plan.levels):
            begin = self.level_cell_offsets[level]
            count = level_plan.maximum_blocks * prod(level_plan.block_shape)
            cell_volume = prod(self.topology.plan.level_spacings[level])
            residual = (content_rate[begin : begin + count] / cell_volume).reshape(
                (
                    level_plan.maximum_blocks,
                    *level_plan.block_shape,
                    self.plan.system.component_count,
                )
            )
            active = self.topology.levels[level].active.reshape(
                (level_plan.maximum_blocks,) + (1,) * (residual.ndim - 1)
            )
            residuals.append(
                self.plan.precision.storage(jnp.where(active, residual, 0.0))
            )
        return BlockAMRFiniteVolumeStageResult(
            tuple(residuals),
            ledger,
            self.plan.precision.decision(maximum_rate),
            self.plan.precision.evidence(),
        )

    def __call__(
        self,
        time: ArrayLike,
        state: BlockHierarchyState,
        fill_patch: FDAMRFillPatchResult,
        args: Any = None,
        /,
    ) -> tuple[Array, ...]:
        return self.evaluate(time, state, fill_patch, args).residuals


__all__ = [
    "BlockAMRFiniteVolumePlan",
    "BlockAMRFiniteVolumeStageResult",
    "PreparedBlockAMRFiniteVolumeDynamics",
]
