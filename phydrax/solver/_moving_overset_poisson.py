#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Atomic moving overset finite-element Poisson reanalysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._execution_runtime import ExecutionGroup
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_mesh import CellMesh
from ..discretization._distributed_field import DistributedHaloPlan
from ..discretization._mapped_locator import prepare_owner_local_mapped_cell_locator
from ..discretization._view_support import OwnerLocalCellGeometry, OwnerLocalFieldSupport
from ..discretization.fem import (
    dirichlet_constraint,
    FiniteElementDiscretization,
    FiniteElementPlan,
    prepare_finite_element_field_reconstruction,
)
from ..discretization.finite_volume._owner_local_remap import (
    OwnerLocalConservativeRemapResult,
    prepare_owner_local_conservative_remap,
    PreparedOwnerLocalConservativeRemap,
)
from ..equations import (
    compile_finite_element_problem,
    DiffusionAction,
    FiniteElementForm,
    SourceAction,
)
from ..geometry._common_root_volume_overlap import PhysicalOverlapActionEvidence
from ..lifecycle import (
    Composition,
    CompositionEntry,
    CompositionRebindReceipt,
    CompositionTransport,
)
from ..linalg import LinearSolvePolicy, solve, TolerancePolicy
from ..linalg._distributed import (
    DistributedKrylovPolicy,
    DistributedLinearOperator,
    DistributedPairing,
    solve_distributed_pcg,
)
from ..meshing._assembly import MeshAssembly, MeshPart
from ..meshing._motion import MeshMotionAdvance
from ..meshing._overset import (
    OversetConnectivity,
    OversetPartSpec,
    prepare_overset_conservative_remap,
    prepare_overset_field_transfer,
    prepare_overset_motion_rebind,
    PreparedOversetConservativeRemap,
    PreparedOversetMotion,
)
from ..meshing._overset_exchange import (
    _owner_local_all_gather_rows,
    _synchronize_owner_search_failure,
)
from ..meshing._result import CellMeshingResult
from ..typing import checked


@final
class MovingOversetPoissonPart(StrictModule, NonTrainableState):
    """One moved part's FE solution, source-overlap, and boundary-flux evidence.

    Serial parts carry a whole-part common refinement; owner-local parts carry
    the owner-routed refinement and its owned-target action result.
    """

    part_name: str = eqx.field(static=True)
    discretization: FiniteElementDiscretization
    coefficients: Array
    conservative_remap: (
        PreparedOversetConservativeRemap | PreparedOwnerLocalConservativeRemap
    )
    overlap_evidence: PhysicalOverlapActionEvidence | OwnerLocalConservativeRemapResult
    remap_complete: bool = eqx.field(static=True)
    source_content: float = eqx.field(static=True)
    target_content: float = eqx.field(static=True)
    boundary_flux: float = eqx.field(static=True)
    overset_flux: float = eqx.field(static=True)
    linear_residual: float = eqx.field(static=True)
    flux_defect: float = eqx.field(static=True)
    overlap_defect: float = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    part_update_id: str = eqx.field(static=True)

    def __init__(
        self,
        part_name: str,
        discretization: FiniteElementDiscretization,
        coefficients: Array,
        conservative_remap: PreparedOversetConservativeRemap
        | PreparedOwnerLocalConservativeRemap,
        overlap_evidence: PhysicalOverlapActionEvidence
        | OwnerLocalConservativeRemapResult,
        /,
        *,
        source_content: float,
        target_content: float,
        boundary_flux: float,
        overset_flux: float,
        linear_residual: float,
        flux_defect: float,
        overlap_defect: float,
        accepted: bool,
    ) -> None:
        values = (
            source_content,
            target_content,
            boundary_flux,
            overset_flux,
            linear_residual,
            flux_defect,
            overlap_defect,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("Moving overset Poisson evidence must be finite.")
        if not isinstance(accepted, bool):
            raise TypeError("accepted must be a host bool.")
        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("discretization must be FiniteElementDiscretization.")
        if isinstance(conservative_remap, PreparedOversetConservativeRemap):
            if not isinstance(overlap_evidence, PhysicalOverlapActionEvidence):
                raise TypeError("Serial overlap evidence must be a physical action.")
            remap_complete = conservative_remap.complete
            remap_id = conservative_remap.transfer_id
            overlap_values = overlap_evidence.values
        elif isinstance(conservative_remap, PreparedOwnerLocalConservativeRemap):
            if not isinstance(overlap_evidence, OwnerLocalConservativeRemapResult):
                raise TypeError("Owner-local overlap evidence must be a remap result.")
            remap_complete = (
                conservative_remap.evidence.coverage_complete
                and overlap_evidence.evidence_id
                == conservative_remap.evidence.evidence_id
            )
            remap_id = conservative_remap.remap_id
            overlap_values = overlap_evidence.owned_values
        else:
            raise TypeError(
                "conservative_remap must be a serial or owner-local overlap remap."
            )
        if not isinstance(coefficients, Array):
            raise TypeError("coefficients must be a JAX array.")
        self.part_name = str(part_name)
        self.discretization = discretization
        self.coefficients = coefficients
        self.conservative_remap = conservative_remap
        self.overlap_evidence = overlap_evidence
        self.remap_complete = bool(remap_complete)
        self.source_content = float(source_content)
        self.target_content = float(target_content)
        self.boundary_flux = float(boundary_flux)
        self.overset_flux = float(overset_flux)
        self.linear_residual = float(linear_residual)
        self.flux_defect = float(flux_defect)
        self.overlap_defect = float(overlap_defect)
        self.accepted = accepted
        self.part_update_id = canonical_fingerprint(
            {
                "kind": "moving-overset-poisson-part",
                "part": self.part_name,
                "connectivity": remap_id,
                "discretization": discretization.prepared_id,
                "coefficients": array_tree_fingerprint(coefficients),
                "overlap": array_tree_fingerprint(overlap_values),
                "source_content": self.source_content,
                "target_content": self.target_content,
                "boundary_flux": self.boundary_flux,
                "overset_flux": self.overset_flux,
                "linear_residual": self.linear_residual,
                "flux_defect": self.flux_defect,
                "overlap_defect": self.overlap_defect,
                "accepted": accepted,
            }
        )


@final
class PreparedMovingOversetPoissonUpdate(StrictModule, NonTrainableState):
    """Unpublished all-part Poisson update with one composition rebind."""

    previous: OversetConnectivity
    candidate: OversetConnectivity
    parts: tuple[MovingOversetPoissonPart, ...]
    motion: PreparedOversetMotion
    execution_group: ExecutionGroup | None
    accepted: bool = eqx.field(static=True)
    update_id: str = eqx.field(static=True)

    def __init__(
        self,
        previous: OversetConnectivity,
        candidate: OversetConnectivity,
        parts: tuple[MovingOversetPoissonPart, ...],
        motion: PreparedOversetMotion,
        execution_group: ExecutionGroup | None = None,
        /,
        *,
        collective_accepted: bool | None = None,
    ) -> None:
        if not isinstance(previous, OversetConnectivity) or not isinstance(
            candidate, OversetConnectivity
        ):
            raise TypeError("previous and candidate must be OversetConnectivity.")
        if not isinstance(motion, PreparedOversetMotion):
            raise TypeError("motion must be PreparedOversetMotion.")
        if execution_group is not None and not isinstance(
            execution_group, ExecutionGroup
        ):
            raise TypeError("execution_group must be an ExecutionGroup or None.")
        if collective_accepted is not None and not isinstance(collective_accepted, bool):
            raise TypeError("collective_accepted must be a host bool or None.")
        ordered = tuple(sorted(parts, key=lambda item: item.part_name))
        if len(ordered) != len(previous.global_part_names) or tuple(
            item.part_name for item in ordered
        ) != tuple(sorted(previous.global_part_names)):
            raise ValueError(
                "Moving Poisson evidence must cover every overset part exactly once."
            )
        accepted = candidate.complete and all(item.accepted for item in ordered)
        if collective_accepted is not None:
            accepted = accepted and collective_accepted
        self.previous = previous
        self.candidate = candidate
        self.parts = ordered
        self.motion = motion
        self.execution_group = execution_group
        self.accepted = accepted
        self.update_id = canonical_fingerprint(
            {
                "kind": "moving-overset-poisson-update",
                "predecessor": previous.connectivity_id,
                "candidate": candidate.connectivity_id,
                "parts": [item.part_update_id for item in ordered],
                "execution_group": (
                    None if execution_group is None else execution_group.spec.group_id
                ),
                "accepted": accepted,
            }
        )

    def commit(self, /, *, accepted_boundary: bool) -> CompositionRebindReceipt:
        if not isinstance(accepted_boundary, bool):
            raise TypeError("accepted_boundary must be a host bool.")
        boundary = accepted_boundary and self.accepted
        if self.execution_group is not None:
            rejected = _synchronize_owner_search_failure(
                not boundary,
                self.execution_group.devices,
                "moving_overset_poisson_commit",
            )
            boundary = not rejected
        return self.motion.commit(accepted_boundary=boundary)


def _require_owner_local_face_halos(
    names: tuple[str, ...],
    views: Mapping[str, OwnerLocalCellGeometry],
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> None:
    """Require owned rows plus every face neighbor of each owned cell per rank."""
    local_error: str | None = None
    try:
        if set(views) != set(names) or any(
            not isinstance(view, OwnerLocalCellGeometry) for view in views.values()
        ):
            raise ValueError("one source-bound view per part is required on every rank")
        for name in names:
            view = views[name]
            cell_ids = np.asarray(view.cell_ids, dtype=np.int64)
            owned = np.asarray(view.owned, dtype=np.bool_)
            if set(np.asarray(view.owned_cell_ids, dtype=np.int64).tolist()) != set(
                cell_ids[owned].tolist()
            ):
                raise ValueError(f"view {name!r} lacks its owned face-neighbor graph")
            resident = set(cell_ids.tolist())
            if any(
                int(neighbor) not in resident
                for neighbor in np.asarray(view.cell_neighbor_ids, dtype=np.int64)
            ):
                raise ValueError(
                    f"view {name!r} omits a face neighbor of one of its owned cells"
                )
    except Exception as error:
        local_error = str(error)
    if _synchronize_owner_search_failure(
        local_error is not None, execution_group.devices, f"{axis_name}_halo"
    ):
        raise ValueError(
            "Owner-local Poisson reanalysis requires owned cells plus their exact "
            "face-neighbor halo: "
            + (local_error if local_error is not None else "another rank refused")
        )


class _OwnerRowRoute(NamedTuple):
    sender: int
    receiver: int
    count: int
    forward: DistributedHaloPlan
    forward_send_slot: int
    forward_receive_slot: int
    reverse: DistributedHaloPlan
    reverse_send_slot: int
    reverse_receive_slot: int
    send_rows: np.ndarray
    ghost_rows: np.ndarray


def _pair_slot(plan: DistributedHaloPlan, rank: int, owned: bool, /) -> int:
    identifiers = np.asarray(plan.local_global_ids, dtype=np.int64)[rank]
    owned_rows = np.asarray(plan.local_owned, dtype=np.bool_)[rank]
    slots = np.flatnonzero((identifiers == 0) & (owned_rows == owned))
    if slots.size != 1:
        raise ValueError("Owner row channel does not expose one exact payload slot.")
    return int(slots[0])


def _owner_local_row_routes(
    row_ids: np.ndarray,
    row_owners: np.ndarray,
    local_rank: int,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> tuple[_OwnerRowRoute, ...]:
    """Owner-to-ghost row routes keyed by stable vertex IDs, without gathers."""
    from ..meshing._distribution import (
        _owner_exchange_message_banks,
        _owner_group_allgather,
        _owner_pair_channel_plan,
    )

    parts = len(execution_group.spec.process_indices)
    owned_row = {
        int(identifier): row
        for row, (identifier, owner) in enumerate(
            zip(row_ids.tolist(), row_owners.tolist(), strict=True)
        )
        if owner == local_rank
    }
    outgoing: list[np.ndarray] = [np.empty((0, 2), dtype=np.int64) for _ in range(parts)]
    for owner in range(parts):
        if owner == local_rank:
            continue
        rows = np.flatnonzero(row_owners == owner)
        outgoing[owner] = np.stack((row_ids[rows], rows), axis=1).astype(np.int64)
    incoming = _owner_exchange_message_banks(
        tuple(outgoing), local_rank, execution_group, f"{axis_name}_requests"
    )
    counts = _owner_group_allgather(
        np.asarray([values.shape[0] for values in outgoing], dtype=np.int64),
        execution_group,
        f"{axis_name}_counts",
    )
    routes = []
    for sender in range(parts):
        for receiver in range(parts):
            count = int(counts[receiver, sender])
            if not count:
                continue
            send_rows = np.zeros((count,), dtype=np.int32)
            ghost_rows = np.zeros((count,), dtype=np.int32)
            if local_rank == sender:
                requests = np.asarray(incoming[receiver], dtype=np.int64)
                send_rows = np.asarray(
                    [owned_row[int(identifier)] for identifier in requests[:, 0]],
                    dtype=np.int32,
                )
            if local_rank == receiver:
                ghost_rows = np.asarray(outgoing[sender][:, 1], dtype=np.int32)
            forward = _owner_pair_channel_plan(sender, receiver, parts)
            reverse = _owner_pair_channel_plan(receiver, sender, parts)
            routes.append(
                _OwnerRowRoute(
                    sender,
                    receiver,
                    count,
                    forward,
                    _pair_slot(forward, sender, True),
                    _pair_slot(forward, receiver, False),
                    reverse,
                    _pair_slot(reverse, receiver, True),
                    _pair_slot(reverse, sender, False),
                    send_rows,
                    ghost_rows,
                )
            )
    return tuple(routes)


def _route_forward(
    routes: tuple[_OwnerRowRoute, ...], values: Array, rank: Array, axis: str, /
) -> Array:
    """Overwrite ghost rows with their owners' values."""
    for route in routes:
        payload = jnp.where(
            rank == route.sender,
            values[jnp.asarray(route.send_rows)],
            jnp.zeros((route.count,), dtype=values.dtype),
        )
        packed = (
            jnp.zeros((route.forward.local_capacity, route.count), dtype=values.dtype)
            .at[route.forward_send_slot]
            .set(payload)
        )
        received = route.forward.exchange(packed, rank, axis_name=axis)[
            route.forward_receive_slot
        ]
        ghost_rows = jnp.asarray(route.ghost_rows)
        values = jax.lax.cond(
            rank == route.receiver,
            lambda current, rows=ghost_rows, data=received: current.at[rows].set(data),
            lambda current: current,
            values,
        )
    return values


def _route_reverse(
    routes: tuple[_OwnerRowRoute, ...],
    values: Array,
    owned_mask: Array,
    rank: Array,
    axis: str,
    /,
) -> Array:
    """Add ghost-row contributions into their owners' rows; ghosts become zero."""
    for route in routes:
        payload = jnp.where(
            rank == route.receiver,
            values[jnp.asarray(route.ghost_rows)],
            jnp.zeros((route.count,), dtype=values.dtype),
        )
        packed = (
            jnp.zeros((route.reverse.local_capacity, route.count), dtype=values.dtype)
            .at[route.reverse_send_slot]
            .set(payload)
        )
        received = route.reverse.exchange(packed, rank, axis_name=axis)[
            route.reverse_receive_slot
        ]
        send_rows = jnp.asarray(route.send_rows)
        values = jax.lax.cond(
            rank == route.sender,
            lambda current, rows=send_rows, data=received: current.at[rows].add(data),
            lambda current: current,
            values,
        )
    return jnp.where(owned_mask, values, jnp.zeros_like(values))


def _owned_cell_submesh(
    view: OwnerLocalCellGeometry,
    mesh: CellMesh,
    execution_group: ExecutionGroup,
    /,
) -> CellMesh:
    """Exactly this rank's owned cells; each global cell is assembled once."""
    from ..discretization import CellBlock, EmptyLocalCellMeshAuthority

    owned_ids = set(
        np.asarray(view.cell_ids, dtype=np.int64)[
            np.asarray(view.owned, dtype=np.bool_)
        ].tolist()
    )
    (block,) = mesh.blocks
    global_ids = np.asarray(block.global_ids, dtype=np.int64)
    selected = np.flatnonzero(np.isin(global_ids, list(owned_ids)))
    if not selected.size:
        return CellMesh.empty_local(
            EmptyLocalCellMeshAuthority(
                view.source_topology_id,
                view.source_geometry_id,
                view.distribution_id,
                execution_group.spec.group_id,
                view.owner_rank,
                view.owner_directory_data.part_count,
                block.name,
                block.cell_kind,
                mesh.topological_dimension,
                mesh.ambient_dimension,
            )
        )
    cells = np.asarray(block.vertices, dtype=np.int64)[selected]
    vertex_rows = np.unique(cells)
    local = np.searchsorted(vertex_rows, cells)
    return CellMesh(
        np.asarray(mesh.coordinates, dtype=np.float64)[vertex_rows],
        (CellBlock(block.name, block.cell_kind, local, global_ids=global_ids[selected]),),
        vertex_global_ids=np.asarray(mesh.vertex_global_ids, dtype=np.int64)[vertex_rows],
        numeric_version=f"owned:{view.local_view_id}",
    )


def _owner_local_poisson_part(
    part_name: str,
    source_view: OwnerLocalCellGeometry,
    target_view: OwnerLocalCellGeometry,
    source_discretization: FiniteElementDiscretization,
    target_discretization: FiniteElementDiscretization,
    source_coefficients: Array,
    receptor_ids: np.ndarray,
    receptor_values: np.ndarray,
    conservative_remap: PreparedOwnerLocalConservativeRemap,
    field_name: str,
    fields: tuple[Any, ...],
    execution_group: ExecutionGroup,
    /,
    *,
    axis_name: str,
    solve_tolerance: float,
    flux_tolerance: float,
) -> MovingOversetPoissonPart:
    """Solve one sharded P1 Poisson part over owned rows and their face halo.

    Each owner assembles only its owned cells; ghost rows are refreshed from
    their owners and assembled contributions are reduced into owner rows by
    sparse owner-pair routes, so the distributed PCG never gathers the mesh.
    """
    field_index = target_discretization._field_index(field_name)
    target_dof_map = target_discretization.dof_maps[field_index]
    source_dof_map = source_discretization.dof_maps[
        source_discretization._field_index(field_name)
    ]
    if (
        source_dof_map.dof_map_id != target_dof_map.dof_map_id
        or source_coefficients.shape != (source_dof_map.global_dof_count,)
    ):
        raise ValueError(
            f"Moving Poisson part {part_name!r} must retain its canonical P1 DOF layout."
        )
    rank = target_view.owner_rank
    devices = execution_group.devices
    rows_by_vertex = _vertex_dof_rows(target_discretization, field_index)
    row_count = target_dof_map.global_dof_count
    row_ids = np.empty((row_count,), dtype=np.int64)
    for identifier, row in rows_by_vertex.items():
        row_ids[row] = identifier
    owner_by_vertex = {
        int(identifier): int(owner)
        for identifier, owner in zip(
            np.asarray(target_view.vertex_ids, dtype=np.int64).tolist(),
            np.asarray(target_view.vertex_owner_ranks, dtype=np.int32).tolist(),
            strict=True,
        )
    }
    row_owners = np.asarray(
        [owner_by_vertex[int(identifier)] for identifier in row_ids], dtype=np.int32
    )
    routes = _owner_local_row_routes(
        row_ids, row_owners, rank, execution_group, f"{axis_name}_rows"
    )

    # Owned-cell assembly: every global cell contributes exactly once.
    target_mesh = target_discretization.mesh
    submesh = _owned_cell_submesh(target_view, target_mesh, execution_group)
    form = FiniteElementForm(
        "moving-overset-poisson",
        field_name,
        (DiffusionAction(field_name, 1.0), SourceAction(field_name, 1.0)),
    )
    coo_rows = np.empty((0,), dtype=np.int64)
    coo_cols = np.empty((0,), dtype=np.int64)
    coo_values = np.empty((0,), dtype=np.float64)
    zero_residual = np.zeros((row_count,), dtype=np.float64)
    boundary_flags = np.zeros((row_count,), dtype=np.float64)
    if submesh.blocks[0].cell_count:
        sub_discretization = FiniteElementPlan(submesh, fields).prepare()
        problem = compile_finite_element_problem(form, sub_discretization)
        sub_rows = _vertex_dof_rows(sub_discretization, 0)
        to_local = np.empty((len(sub_rows),), dtype=np.int64)
        for identifier, sub_row in sub_rows.items():
            to_local[sub_row] = rows_by_vertex[identifier]
        matrix = problem.to_scipy_csr().tocoo()
        coo_rows = to_local[np.asarray(matrix.row, dtype=np.int64)]
        coo_cols = to_local[np.asarray(matrix.col, dtype=np.int64)]
        coo_values = np.asarray(matrix.data, dtype=np.float64)
        residual_zero = np.asarray(
            problem.full_residual(jnp.zeros((len(sub_rows),), dtype=jnp.float64)),
            dtype=np.float64,
        )
        np.add.at(zero_residual, to_local, residual_zero)
        # A facet of an owned cell met by no other resident cell is physical
        # boundary: the view holds every face neighbor of its owned cells.
        (block,) = target_mesh.blocks
        cell_vertices = np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)[
            np.asarray(block.vertices, dtype=np.int64)
        ]
        owned_cells = set(
            np.asarray(submesh.blocks[0].global_ids, dtype=np.int64).tolist()
        )
        facet_count: dict[tuple[int, ...], int] = {}
        owned_facets: set[tuple[int, ...]] = set()
        arity = cell_vertices.shape[1]
        for cell_id, vertices in zip(
            np.asarray(block.global_ids, dtype=np.int64).tolist(),
            cell_vertices.tolist(),
            strict=True,
        ):
            for skip in range(arity):
                key = tuple(sorted(vertices[:skip] + vertices[skip + 1 :]))
                facet_count[key] = facet_count.get(key, 0) + 1
                if cell_id in owned_cells:
                    owned_facets.add(key)
        for key in owned_facets:
            if facet_count[key] == 1:
                for vertex in key:
                    boundary_flags[rows_by_vertex[vertex]] = 1.0
    receptor_flags = np.zeros((row_count,), dtype=np.float64)
    receptor_sums = np.zeros((row_count,), dtype=np.float64)
    for identifier, value in zip(
        receptor_ids.tolist(), receptor_values.tolist(), strict=True
    ):
        row = rows_by_vertex.get(int(identifier))
        if row is None:
            raise ValueError(
                "Overset receptor identity is absent from its target P1 field layout."
            )
        receptor_flags[row] += 1.0
        receptor_sums[row] += float(value)

    capacity_rows = _owner_local_all_gather_rows(
        np.asarray(((row_count, coo_values.size),), dtype=np.int64),
        devices,
        f"{axis_name}_capacity",
    )
    capacity = max(1, max(int(rows[0, 0]) for rows in capacity_rows))
    nonzero_capacity = max(1, max(int(rows[0, 1]) for rows in capacity_rows))
    owned_counts = _owner_local_all_gather_rows(
        np.asarray(((int(np.count_nonzero(row_owners == rank)),),), dtype=np.int64),
        devices,
        f"{axis_name}_owned",
    )
    maximum_steps = max(32, 4 * sum(int(rows[0, 0]) for rows in owned_counts))

    def pad(values: np.ndarray, size: int, dtype: Any, /) -> np.ndarray:
        padded = np.zeros((size,), dtype=dtype)
        padded[: values.size] = values
        return padded

    arguments = tuple(
        jnp.asarray(value[None, ...])
        for value in (
            pad(coo_rows, nonzero_capacity, np.int32),
            pad(coo_cols, nonzero_capacity, np.int32),
            pad(coo_values, nonzero_capacity, np.float64),
            pad(zero_residual, capacity, np.float64),
            pad(np.asarray(source_coefficients, np.float64), capacity, np.float64),
            pad(boundary_flags, capacity, np.float64),
            pad(receptor_flags, capacity, np.float64),
            pad(receptor_sums, capacity, np.float64),
            pad(row_owners == rank, capacity, np.bool_),
        )
    )
    axis = f"{axis_name}_pcg"
    policy = DistributedKrylovPolicy(
        maximum_steps, relative_tolerance=0.0, absolute_tolerance=solve_tolerance
    )
    operator_id = canonical_fingerprint(
        {"kind": "owner-local-moving-poisson-operator", "view": target_view.local_view_id}
    )

    def kernel(
        rows: Array,
        cols: Array,
        values: Array,
        residual_zero: Array,
        carried: Array,
        boundary: Array,
        receptor_flag: Array,
        receptor_sum: Array,
        owned: Array,
    ) -> tuple[Array, ...]:
        index = jax.lax.axis_index(axis)

        def assemble(local: Array, /) -> Array:
            return _route_reverse(routes, local, owned, index, axis)

        def stiffness(state: Array, /) -> Array:
            refreshed = _route_forward(routes, state, index, axis)
            return assemble(
                jax.ops.segment_sum(values * refreshed[cols], rows, num_segments=capacity)
            )

        boundary_rows = owned & (assemble(boundary) > 0.0)
        receptor_count = assemble(receptor_flag)
        receptor_rows = owned & (receptor_count > 0.0)
        receptor_value = jnp.where(
            receptor_rows,
            assemble(receptor_sum) / jnp.where(receptor_rows, receptor_count, 1.0),
            0.0,
        )
        constrained = boundary_rows | receptor_rows
        free = owned & ~constrained
        dirichlet = jnp.where(
            receptor_rows, receptor_value, jnp.where(boundary_rows, carried, 0.0)
        )
        load = assemble(residual_zero)
        right_hand_side = jnp.where(free, -(stiffness(dirichlet) + load), 0.0)

        def action(state: Array, /) -> Array:
            return jnp.where(free, stiffness(jnp.where(free, state, 0.0)), 0.0)

        result = solve_distributed_pcg(
            DistributedLinearOperator(
                action, action, (capacity,), (capacity,), operator_id=operator_id
            ),
            right_hand_side,
            DistributedPairing(free, axis_name=axis, pairing_id=operator_id),
            policy,
        )
        solution = jnp.where(free, result.value, dirichlet)
        residual = stiffness(solution) + load
        physical = boundary_rows & ~receptor_rows
        return (
            _route_forward(routes, solution, index, axis),
            result.converged,
            jax.lax.pmax(
                jnp.max(jnp.where(free, jnp.abs(residual), 0.0), initial=0.0), axis
            ),
            jax.lax.psum(jnp.sum(jnp.where(physical, residual, 0.0)), axis),
            jax.lax.psum(jnp.sum(jnp.where(receptor_rows, residual, 0.0)), axis),
        )

    solved = jax.pmap(kernel, axis_name=axis, devices=devices)(*arguments)
    solution, converged, linear_residual, boundary_flux, overset_flux = (
        value[0] for value in solved
    )
    if not bool(converged):
        raise RuntimeError(f"Moving Poisson solve failed on overset part {part_name!r}.")
    coefficients = jnp.asarray(solution[:row_count])

    # Physical overlap of the unit source: owner-routed common refinement.
    source_measures = np.where(
        np.asarray(source_view.owned, dtype=np.bool_),
        np.asarray(source_view.cell_measures, dtype=np.float64),
        0.0,
    )
    overlap = conservative_remap.apply_content(
        jnp.asarray(source_measures),
        source_view,
        target_view,
        execution_group,
    )
    target_measure_by_id = {
        int(identifier): float(measure)
        for identifier, measure in zip(
            np.asarray(target_view.cell_ids, dtype=np.int64).tolist(),
            np.asarray(target_view.cell_measures, dtype=np.float64).tolist(),
            strict=True,
        )
    }
    owned_target_ids = np.asarray(overlap.owned_target_cell_ids, dtype=np.int64)
    owned_measures = np.asarray(
        [target_measure_by_id[int(identifier)] for identifier in owned_target_ids],
        dtype=np.float64,
    )
    remapped = np.asarray(overlap.owned_values, dtype=np.float64).reshape((-1,))
    totals = _owner_local_all_gather_rows(
        np.asarray(
            (
                (
                    float(np.max(np.abs(remapped - owned_measures), initial=0.0)),
                    float(np.sum(source_measures)),
                    float(np.sum(remapped)),
                    float(np.sum(owned_measures)),
                ),
            ),
            dtype=np.float64,
        ),
        devices,
        f"{axis_name}_ledger",
    )
    overlap_defect = max(float(rows[0, 0]) for rows in totals)
    source_content = float(np.sum([rows[0, 1] for rows in totals]))
    target_content = float(np.sum([rows[0, 2] for rows in totals]))
    target_measure = float(np.sum([rows[0, 3] for rows in totals]))
    boundary_flux_value = float(boundary_flux)
    overset_flux_value = float(overset_flux)
    flux_defect = abs(boundary_flux_value + overset_flux_value + target_measure)
    complete = bool(conservative_remap.evidence.coverage_complete)
    accepted = (
        complete
        and float(linear_residual) <= solve_tolerance
        and overlap_defect <= flux_tolerance * max(1.0, target_measure)
        and flux_defect <= flux_tolerance * max(1.0, target_measure)
    )
    return MovingOversetPoissonPart(
        part_name,
        target_discretization,
        coefficients,
        conservative_remap,
        overlap,
        source_content=source_content,
        target_content=target_content,
        boundary_flux=boundary_flux_value,
        overset_flux=overset_flux_value,
        linear_residual=float(linear_residual),
        flux_defect=flux_defect,
        overlap_defect=overlap_defect,
        accepted=accepted,
    )


def _owner_local_reconstruction(
    view: OwnerLocalCellGeometry,
    discretization: FiniteElementDiscretization,
    field_name: str,
    /,
) -> Any:
    blocks = tuple(block for block in view.blocks if np.asarray(block.cell_ids).size)
    if len(blocks) != 1:
        raise ValueError(
            "Owner-local Poisson reconstruction requires exactly one non-empty simplex block per part."
        )
    block_name = blocks[0].name
    locator = prepare_owner_local_mapped_cell_locator(view, block_name)
    support = OwnerLocalFieldSupport(view, locator, block_name)
    return prepare_finite_element_field_reconstruction(
        discretization,
        field_name,
        block_name=block_name,
        locator=locator,
        support_geometry=support,
    )


def _owner_local_coefficient_ids(
    discretization: FiniteElementDiscretization, field_name: str, /
) -> np.ndarray:
    field_index = discretization._field_index(field_name)
    dof_map = discretization.dof_maps[field_index]
    rows = _vertex_dof_rows(discretization, field_index)
    identifiers = np.empty((dof_map.global_dof_count,), dtype=np.int64)
    for identifier, row in rows.items():
        identifiers[row] = identifier
    if np.unique(identifiers).size != identifiers.size:
        raise ValueError("Owner-local P1 coefficient identities must be unique.")
    return identifiers


def _rebind_successor_spec(
    specification: OversetPartSpec, target: MeshPart, /
) -> OversetPartSpec:
    def rebind(scope: Any) -> Any:
        return (
            None
            if scope is None
            else target.scope(scope.entity_dimension, np.asarray(scope.entity_ids))
        )

    return OversetPartSpec(
        target.name,
        wall=rebind(specification.wall),
        boundary=rebind(specification.boundary),
        excluded=rebind(specification.excluded),
        solid_query=specification.solid_query,
        solid_index=specification.solid_index,
        priority=specification.priority,
        owner_rank=specification.owner_rank,
        image_rotation=specification.image_rotation,
        image_translation=specification.image_translation,
    )


def _prepare_owner_local_moving_poisson_update(
    composition: Composition,
    previous: OversetConnectivity,
    advances: Mapping[str, MeshMotionAdvance],
    plans: Mapping[str, FiniteElementPlan],
    field_name: str,
    execution_group: ExecutionGroup,
    successor_assembly: MeshAssembly,
    solve_tolerance: float,
    flux_tolerance: float,
    retain: Sequence[str],
    reprepare: Sequence[CompositionEntry],
    transports: Sequence[CompositionTransport],
    invalidate: Sequence[str],
    /,
) -> PreparedMovingOversetPoissonUpdate:
    names = tuple(sorted(previous.global_part_names))
    source_parts = {part.name: part for part in previous.assembly.parts}
    source_views = {
        name: part.carrier
        for name, part in source_parts.items()
        if isinstance(part.carrier, OwnerLocalCellGeometry)
    }
    target_parts = {part.name: part for part in successor_assembly.parts}
    target_views = {
        name: part.carrier
        for name, part in target_parts.items()
        if isinstance(part.carrier, OwnerLocalCellGeometry)
    }
    _require_owner_local_face_halos(
        names, source_views, execution_group, "moving_overset_poisson_source"
    )
    _require_owner_local_face_halos(
        names, target_views, execution_group, "moving_overset_poisson_target"
    )
    if (
        set(plans) != set(names)
        or set(source_parts) != set(names)
        or set(target_parts) != set(names)
        or successor_assembly.local_rank != previous.assembly.local_rank
    ):
        raise ValueError(
            "Owner-local FE plans and source/successor views must cover every overset part."
        )
    if any(not isinstance(plan, FiniteElementPlan) for plan in plans.values()):
        raise TypeError("Owner-local plans must contain FiniteElementPlan values.")

    source_discretizations: dict[str, FiniteElementDiscretization] = {}
    source_coefficients: dict[str, Array] = {}
    field_entries: dict[str, CompositionEntry] = {}
    state_entries: dict[str, CompositionEntry] = {}
    target_discretizations: dict[str, FiniteElementDiscretization] = {}
    target_reconstructions: dict[str, Any] = {}
    coefficient_ids: dict[str, np.ndarray] = {}
    for name in names:
        source_view = source_views[name]
        target_view = target_views[name]
        source_mesh, source_geometry = source_view.reconstruct_local_geometry()
        target_mesh, target_geometry = target_view.reconstruct_local_geometry()
        if source_mesh is None or source_geometry is None:
            raise ValueError(f"Owner-local source part {name!r} has no FE mesh.")
        if target_mesh is None or target_geometry is None:
            raise ValueError(f"Owner-local successor part {name!r} has no FE mesh.")
        if (
            source_view.source_topology_id != target_view.source_topology_id
            or not np.array_equal(
                np.asarray(source_view.cell_ids), np.asarray(target_view.cell_ids)
            )
        ):
            raise ValueError(
                f"Owner-local Poisson part {name!r} changed topology; transfer its FE state through FiniteElementTopologyTransaction.adapt_owner_local first."
            )
        plan = plans[name]
        if (
            plan.mesh.mesh_id != source_mesh.mesh_id
            or plan.coordinate_spec.geometry_layout_id
            != source_geometry.geometry_layout_id
        ):
            raise ValueError(f"Accepted local FE plan for {name!r} is stale.")
        fe_entry = composition.entry(f"fe:{name}")
        if fe_entry.role != "discretization" or not isinstance(
            fe_entry.value, FiniteElementDiscretization
        ):
            raise TypeError(f"Composition FE entry for {name!r} is malformed.")
        source_discretization = fe_entry.value
        if source_discretization.plan_id != plan.plan_id:
            raise ValueError(f"Composition FE entry for {name!r} binds another plan.")
        _require_scalar_simplex_p1(plan, source_discretization, field_name, name)
        state_entry = _state_entry(composition, fe_entry.entry_id, name)
        coefficients = state_entry.value
        if not isinstance(coefficients, Array) or coefficients.shape != (
            source_discretization.dof_maps[0].global_dof_count,
        ):
            raise ValueError(
                f"Accepted local FE state for {name!r} has a stale DOF layout."
            )
        target_plan = FiniteElementPlan(
            target_mesh,
            plan.fields,
            precision_policy=plan.precision_policy,
            coordinate_spec=target_geometry,
            coefficient_dtype=plan.coefficient_dtype,
            coordinate_measure=plan.coordinate_measure,
        )
        target_discretization = target_plan.prepare(
            numeric_version=f"owner-local-overset-motion:{target_view.local_view_id}"
        )
        _require_scalar_simplex_p1(target_plan, target_discretization, field_name, name)
        if source_discretization.dof_maps[0].dof_map_id != (
            target_discretization.dof_maps[0].dof_map_id
        ):
            raise ValueError(
                f"Owner-local Poisson part {name!r} changed FE DOF identity; use adapt_owner_local."
            )
        source_discretizations[name] = source_discretization
        source_coefficients[name] = coefficients
        field_entries[name] = fe_entry
        state_entries[name] = state_entry
        target_discretizations[name] = target_discretization
        target_reconstructions[name] = _owner_local_reconstruction(
            target_view, target_discretization, field_name
        )
        coefficient_ids[name] = _owner_local_coefficient_ids(
            target_discretization, field_name
        )

    successor_specs = {
        name: _rebind_successor_spec(
            next(spec for spec in previous.specs if spec.part_name == name),
            target_parts[name],
        )
        for name in advances
    }
    candidate = previous.moved_from_advances(
        advances,
        successor_specs=successor_specs,
        owner_local_source_views=source_views,
        successor_assembly=successor_assembly,
        execution_group=execution_group,
        owner_local_field_reconstructions=target_reconstructions,
        owner_local_coefficient_ids=coefficient_ids,
        axis_name="moving_overset_poisson_candidate",
        owner_local_field_name=field_name,
    )
    candidate.require_complete()
    transfer = prepare_overset_field_transfer(
        candidate, target_reconstructions, field_name
    )
    exchange = transfer.prepare_owner_local_exchange(
        coefficient_ids,
        execution_group,
        target_views,
        axis_name="moving_overset_poisson_field",
    )
    # The exchange consumes this owner's donor coefficients in the order of its
    # owned coefficient identities; its layout rows address the packed halo.
    donor_field_ids = dict(exchange.field_ids)
    local_donors = {}
    for name, _, _ in exchange.donor_layouts:
        row_by_id = {
            int(identifier): row
            for row, identifier in enumerate(np.asarray(coefficient_ids[name]).tolist())
        }
        local_donors[name] = source_coefficients[name][
            np.asarray(
                [row_by_id[int(identifier)] for identifier in donor_field_ids[name]],
                dtype=np.int64,
            )
        ]
    local_receptor_values = exchange.apply(local_donors)
    local_receptor_ids = dict(exchange.receptor_global_ids)

    part_evidence: list[MovingOversetPoissonPart] = []
    for index, name in enumerate(names):
        source_view = source_views[name]
        target_view = target_views[name]
        advance = advances.get(name)
        transition = None if advance is None else advance.geometry_transition
        remap = prepare_owner_local_conservative_remap(
            source_view,
            target_view,
            execution_group,
            transition=transition,
            # Local advance IDs bind rank-local meshes; the remap binds the
            # group-wide motion between the accepted source and target parts.
            advance_id=(
                None
                if advance is None
                else canonical_fingerprint(
                    {
                        "kind": "owner-local-moving-poisson-motion",
                        "source": source_view.part_id,
                        "target": target_view.part_id,
                        "driver": advance.motion_driver_id,
                    }
                )
            ),
            axis_name=f"moving_overset_poisson_remap_{index}",
            require_complete=True,
        )
        part_evidence.append(
            _owner_local_poisson_part(
                name,
                source_view,
                target_view,
                source_discretizations[name],
                target_discretizations[name],
                source_coefficients[name],
                np.asarray(local_receptor_ids.get(name, ()), dtype=np.int64),
                np.asarray(
                    local_receptor_values.get(name, jnp.empty((0,), jnp.float64)),
                    dtype=np.float64,
                ).reshape((-1,)),
                remap,
                field_name,
                plans[name].fields,
                execution_group,
                axis_name=f"moving_overset_poisson_solve_{index}",
                solve_tolerance=solve_tolerance,
                flux_tolerance=flux_tolerance,
            )
        )

    target_part_entries = {
        entry.entry_id: entry for entry in candidate.composition_entries()
    }
    new_fe_entries: list[CompositionEntry] = []
    state_transports: list[CompositionTransport] = []
    for part in part_evidence:
        name = part.part_name
        old_fe = field_entries[name]
        old_state = state_entries[name]
        target_part_entry = target_part_entries[f"overset:part:{name}"]
        dof_map = part.discretization.dof_maps[
            part.discretization._field_index(field_name)
        ]
        new_fe = CompositionEntry(
            part.discretization,
            entry_id=old_fe.entry_id,
            role=old_fe.role,
            owner_id=old_fe.owner_id,
            structure_id=dof_map.dof_map_id,
            revision_id=part.discretization.prepared_id,
            semantics_id=old_fe.semantics_id,
            dependencies=(target_part_entry.binding("revision"),),
        )
        state_revision = canonical_fingerprint(
            {
                "kind": "moving-overset-poisson-state",
                "update": candidate.connectivity_id,
                "part": name,
                "coefficients": array_tree_fingerprint(part.coefficients),
            }
        )
        new_state = CompositionEntry(
            part.coefficients,
            entry_id=old_state.entry_id,
            role=old_state.role,
            owner_id=old_state.owner_id,
            structure_id=dof_map.dof_map_id,
            revision_id=state_revision,
            semantics_id=old_state.semantics_id,
            dependencies=(new_fe.binding("revision"),),
        )
        new_fe_entries.append(new_fe)
        state_transports.append(
            CompositionTransport(
                "physical-remap",
                (old_state.entry_id,),
                (new_state,),
                source_structure_ids=(old_state.structure_id,),
                route_id=f"moving-overset-poisson:{name}",
                successful=jnp.asarray(part.accepted, dtype=jnp.bool_),
            )
        )
    motion = prepare_overset_motion_rebind(
        composition,
        previous,
        candidate,
        transports=(*transports, *state_transports),
        reprepare=(*reprepare, *new_fe_entries),
        retain=tuple(retain),
        invalidate=tuple(invalidate),
    )
    local_accepted = candidate.complete and all(part.accepted for part in part_evidence)
    globally_rejected = _synchronize_owner_search_failure(
        not local_accepted,
        execution_group.devices,
        "moving_overset_poisson_evidence",
    )
    return PreparedMovingOversetPoissonUpdate(
        previous,
        candidate,
        tuple(part_evidence),
        motion,
        execution_group,
        collective_accepted=not globally_rejected,
    )


def _cell_result(part: MeshPart, /) -> CellMeshingResult:
    if not isinstance(part.carrier, CellMeshingResult):
        raise TypeError(f"Overset part {part.name!r} requires a CellMeshingResult.")
    return part.carrier


def _state_entry(
    composition: Composition, field_entry_id: str, part_name: str, /
) -> CompositionEntry:
    candidates = tuple(
        entry
        for entry in composition.entries
        if entry.role == "physical-state"
        and any(
            dependency.entry_id == field_entry_id and dependency.facet == "revision"
            for dependency in entry.dependencies
        )
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Composition must bind one accepted FE state to part {part_name!r}."
        )
    return candidates[0]


def _require_scalar_simplex_p1(
    plan: FiniteElementPlan,
    discretization: FiniteElementDiscretization,
    field_name: str,
    part_name: str,
    /,
) -> int:
    if len(plan.fields) != 1 or plan.fields[0].name != field_name:
        raise ValueError(
            f"Moving Poisson part {part_name!r} requires exactly one {field_name!r} field."
        )
    field = plan.fields[0]
    if field.component_shape:
        raise ValueError("Moving Poisson reanalysis requires a scalar FE field.")
    if plan.mesh.topological_dimension != plan.mesh.ambient_dimension:
        raise ValueError("Moving Poisson requires full-dimensional simplex meshes.")
    if any(
        block.cell_kind not in ("triangle", "tetrahedron")
        or element.family != "Lagrange"
        or element.degree != 1
        or element.continuity != "conforming"
        or element.representation != "point_value"
        for block, element in zip(plan.mesh.blocks, field.elements, strict=True)
    ):
        raise ValueError(
            "Moving Poisson reanalysis currently requires conforming scalar P1 simplex fields."
        )
    if len(discretization.dof_maps) != 1:
        raise ValueError("Moving Poisson discretization must have one scalar DOF map.")
    field_index = discretization._field_index(field_name)
    dof_map = discretization.dof_maps[field_index]
    if dof_map.association != "vertex" or dof_map.component_shape:
        raise ValueError(
            "Scalar P1 Poisson fields must use one vertex DOF per mesh vertex."
        )
    return field_index


def _vertex_dof_rows(
    discretization: FiniteElementDiscretization, field_index: int, /
) -> dict[int, int]:
    dof_map = discretization.dof_maps[field_index]
    vertex_ids = np.asarray(discretization.mesh.vertex_global_ids, dtype=np.int64)
    rows: dict[int, int] = {}
    for block_index, block in enumerate(discretization.mesh.blocks):
        vertices = np.asarray(block.vertices, dtype=np.int64)
        dofs = np.asarray(dof_map.cell_dofs[block_index], dtype=np.int64)
        if vertices.shape != dofs.shape:
            raise ValueError("Scalar P1 cell vertex and DOF routes must align exactly.")
        for vertex_row, dof_row in zip(
            vertices.reshape(-1), dofs.reshape(-1), strict=True
        ):
            identifier = int(vertex_ids[vertex_row])
            previous = rows.setdefault(identifier, int(dof_row))
            if previous != int(dof_row):
                raise ValueError("Shared P1 vertices must have one canonical global DOF.")
    if set(rows) != set(vertex_ids.tolist()):
        raise ValueError("P1 DOF routes must cover every mesh vertex exactly.")
    return rows


def _poisson_part(
    part_name: str,
    source_discretization: FiniteElementDiscretization,
    target_discretization: FiniteElementDiscretization,
    source_coefficients: Array,
    receptor_ids: np.ndarray,
    receptor_values: Array,
    conservative_remap: PreparedOversetConservativeRemap,
    field_name: str,
    /,
    *,
    solve_tolerance: float,
    flux_tolerance: float,
) -> MovingOversetPoissonPart:
    field_index = target_discretization._field_index(field_name)
    target_dof_map = target_discretization.dof_maps[field_index]
    source_dof_map = source_discretization.dof_maps[
        source_discretization._field_index(field_name)
    ]
    if (
        source_dof_map.dof_map_id != target_dof_map.dof_map_id
        or source_coefficients.shape != (source_dof_map.global_dof_count,)
    ):
        raise ValueError(
            f"Moving Poisson part {part_name!r} must retain its canonical P1 DOF layout."
        )
    source_measures = np.asarray(
        source_discretization.measures[0].weights, dtype=np.float64
    )
    target_measures = np.asarray(
        target_discretization.measures[0].weights, dtype=np.float64
    )
    overlap_evidence = conservative_remap.apply_extensive_with_evidence(source_measures)
    remapped_source = np.asarray(overlap_evidence.values, dtype=np.float64)
    overlap_defect = float(np.max(np.abs(remapped_source - target_measures), initial=0.0))
    source_content = float(np.asarray(overlap_evidence.source_total))
    target_content = float(np.asarray(overlap_evidence.target_total))
    rows = _vertex_dof_rows(target_discretization, field_index)
    carried = jnp.asarray(source_coefficients)
    receptor_mask = np.zeros((target_dof_map.global_dof_count,), dtype=np.bool_)
    for identifier, value in zip(
        receptor_ids.tolist(), np.asarray(receptor_values).reshape((-1,)), strict=True
    ):
        try:
            dof_row = rows[int(identifier)]
        except KeyError as error:
            raise ValueError(
                "Overset receptor identity is absent from its target P1 field layout."
            ) from error
        carried = carried.at[dof_row].set(value)
        receptor_mask[dof_row] = True
    boundary_mask = np.asarray(target_dof_map.boundary_dof_mask, dtype=np.bool_)
    constrained_mask = boundary_mask | receptor_mask
    constraint = dirichlet_constraint(
        target_discretization, field_name, boundary_mask=constrained_mask
    )
    form = FiniteElementForm(
        "moving-overset-poisson",
        field_name,
        (
            DiffusionAction(field_name, 1.0),
            SourceAction(field_name, 1.0),
        ),
    )
    problem = compile_finite_element_problem(
        form,
        target_discretization,
        constraint=constraint,
        dirichlet_values=carried,
    )
    system, rhs = problem.linear_system()
    if not isinstance(rhs, Array):
        raise TypeError("Compiled Poisson right-hand side must be a JAX array.")
    policy = LinearSolvePolicy(
        tolerance=TolerancePolicy(
            relative=0.0,
            absolute=solve_tolerance,
            max_steps=max(32, 4 * target_dof_map.global_dof_count),
        )
    )
    solution = solve(system, rhs, policy=policy)
    if not bool(jnp.all(solution.successful)) or not isinstance(solution.value, Array):
        raise RuntimeError(f"Moving Poisson solve failed on overset part {part_name!r}.")
    operator_action = system.operator.mv(solution.value)
    linear_residual = float(jnp.max(jnp.abs(operator_action - rhs), initial=0.0))
    coefficients = problem.expand(solution.value)
    if not isinstance(coefficients, Array):
        raise TypeError("Expanded Poisson state must be a JAX array.")
    full_residual = problem.full_residual(coefficients)
    if not isinstance(full_residual, Array):
        raise TypeError("Full Poisson residual must be a JAX array.")
    residual = np.asarray(full_residual, dtype=np.float64)
    source_content_target = float(np.sum(target_measures))
    physical_dofs = boundary_mask & ~receptor_mask
    boundary_flux = float(np.sum(residual[physical_dofs]))
    overset_flux = float(np.sum(residual[receptor_mask]))
    flux_defect = abs(boundary_flux + overset_flux + source_content_target)
    accepted = (
        conservative_remap.complete
        and linear_residual <= solve_tolerance
        and overlap_defect <= flux_tolerance * max(1.0, source_content_target)
        and flux_defect <= flux_tolerance * max(1.0, source_content_target)
    )
    return MovingOversetPoissonPart(
        part_name,
        target_discretization,
        coefficients,
        conservative_remap,
        overlap_evidence,
        source_content=source_content,
        target_content=target_content,
        boundary_flux=boundary_flux,
        overset_flux=overset_flux,
        linear_residual=linear_residual,
        flux_defect=flux_defect,
        overlap_defect=overlap_defect,
        accepted=accepted,
    )


@checked
def prepare_moving_overset_poisson_update(
    composition: Composition,
    previous: OversetConnectivity,
    advances: Mapping[str, MeshMotionAdvance],
    plans: Mapping[str, FiniteElementPlan],
    field_name: str,
    /,
    *,
    solve_tolerance: float = 1e-11,
    flux_tolerance: float = 1e-9,
    retain: Sequence[str] = (),
    reprepare: Sequence[CompositionEntry] = (),
    transports: Sequence[CompositionTransport] = (),
    invalidate: Sequence[str] = (),
    execution_group: ExecutionGroup | None = None,
    successor_assembly: MeshAssembly | None = None,
) -> PreparedMovingOversetPoissonUpdate:
    """Rebuild, remap, solve, and stage every part before one accepted commit.

    Each source part must have a `fe:<part>` discretization entry and one
    `physical-state` entry depending on that FE revision. P1 state crosses the
    motion through the declared FE layout; the unit Poisson source is separately
    transported by exact physical overlap and its boundary plus overset reaction
    fluxes must balance that source content. Owner-local updates require complete
    source and successor cell views on every group rank so overlap and flux
    evidence covers the global part without assembling coordinate payloads.
    Topology changes are transferred by `FiniteElementTopologyTransaction`
    before reanalysis. The caller's composition is never mutated during
    preparation.
    """

    if not isinstance(composition, Composition):
        raise TypeError("composition must be Composition.")
    if not isinstance(previous, OversetConnectivity):
        raise TypeError("previous must be OversetConnectivity.")
    if not isinstance(advances, Mapping) or not advances:
        raise TypeError("advances must map at least one moved part to MeshMotionAdvance.")
    if not isinstance(plans, Mapping):
        raise TypeError("plans must map every overset part to its accepted FE plan.")
    field = str(field_name).strip()
    if not field or field != field_name:
        raise ValueError("field_name must be a non-empty exact identifier.")
    tolerances = (float(solve_tolerance), float(flux_tolerance))
    if any(not np.isfinite(value) or value <= 0.0 for value in tolerances):
        raise ValueError("Poisson solve and flux tolerances must be positive and finite.")
    names = tuple(sorted(previous.global_part_names))
    if set(plans) != set(names):
        raise ValueError(
            "Finite-element plans must cover every overset part exactly once."
        )
    if any(not isinstance(plan, FiniteElementPlan) for plan in plans.values()):
        raise TypeError("plans must contain FiniteElementPlan values.")
    if not set(advances).issubset(names):
        raise ValueError("Motion advances may name only registered overset parts.")
    if any(not isinstance(value, MeshMotionAdvance) for value in advances.values()):
        raise TypeError("Every advance must be a MeshMotionAdvance.")
    if previous.owner_local:
        if not isinstance(execution_group, ExecutionGroup) or not isinstance(
            successor_assembly, MeshAssembly
        ):
            raise TypeError(
                "Owner-local moving Poisson requires an ExecutionGroup and successor MeshAssembly."
            )
        return _prepare_owner_local_moving_poisson_update(
            composition,
            previous,
            advances,
            plans,
            field,
            execution_group,
            successor_assembly,
            tolerances[0],
            tolerances[1],
            retain,
            reprepare,
            transports,
            invalidate,
        )
    if execution_group is not None or successor_assembly is not None:
        raise ValueError(
            "Execution-group successor inputs apply only to owner-local connectivity."
        )
    candidate = previous.moved_from_advances(advances)
    candidate.require_complete()
    if candidate.owner_local:
        raise ValueError(
            "Serial moving FE-Poisson reanalysis cannot publish owner-local successors."
        )
    source_parts = {part.name: part for part in previous.assembly.parts}
    target_parts = {part.name: part for part in candidate.assembly.parts}
    field_entries: dict[str, CompositionEntry] = {}
    state_entries: dict[str, CompositionEntry] = {}
    source_discretizations: dict[str, FiniteElementDiscretization] = {}
    source_coefficients: dict[str, Array] = {}
    target_discretizations: dict[str, FiniteElementDiscretization] = {}
    for name in names:
        source_part = source_parts[name]
        source_result = _cell_result(source_part)
        plan = plans[name]
        if (
            plan.mesh.mesh_id != source_result.mesh.mesh_id
            or plan.coordinate_spec.geometry_layout_id
            != source_result.geometry.geometry_layout_id
        ):
            raise ValueError(f"Accepted FE plan for {name!r} is stale against its mesh.")
        fe_entry = composition.entry(f"fe:{name}")
        if fe_entry.role != "discretization" or not isinstance(
            fe_entry.value, FiniteElementDiscretization
        ):
            raise TypeError(f"Composition FE entry for {name!r} is malformed.")
        discretization = fe_entry.value
        if discretization.plan_id != plan.plan_id:
            raise ValueError(f"Composition FE entry for {name!r} binds another plan.")
        field_index = _require_scalar_simplex_p1(plan, discretization, field, name)
        state_entry = _state_entry(composition, fe_entry.entry_id, name)
        values = state_entry.value
        if not isinstance(values, Array) or values.shape != (
            discretization.dof_maps[field_index].global_dof_count,
        ):
            raise ValueError(
                f"Accepted scalar state for {name!r} has the wrong DOF layout."
            )
        target_result = _cell_result(target_parts[name])
        if target_result.mesh.topology_id != source_result.mesh.topology_id:
            raise ValueError(
                f"Moving Poisson part {name!r} changed topology and requires its declared FE transfer owner."
            )
        target_plan = FiniteElementPlan(
            target_result.mesh,
            plan.fields,
            precision_policy=plan.precision_policy,
            coordinate_spec=target_result.geometry,
            coefficient_dtype=plan.coefficient_dtype,
            coordinate_measure=plan.coordinate_measure,
        )
        target_discretization = target_plan.prepare(
            numeric_version=f"overset-motion:{candidate.epoch}"
        )
        target_field_index = _require_scalar_simplex_p1(
            target_plan, target_discretization, field, name
        )
        if discretization.dof_maps[field_index].dof_map_id != (
            target_discretization.dof_maps[target_field_index].dof_map_id
        ):
            raise ValueError(f"Moving Poisson part {name!r} changed its FE DOF identity.")
        field_entries[name] = fe_entry
        state_entries[name] = state_entry
        source_discretizations[name] = discretization
        source_coefficients[name] = values
        target_discretizations[name] = target_discretization

    donor_names = tuple(sorted({packet.donor_part for packet in candidate.packets}))
    field_transfer = prepare_overset_field_transfer(
        candidate,
        {name: target_discretizations[name] for name in donor_names},
        field,
    )
    donor_values = {
        name: source_coefficients[name] for name in field_transfer.donor_names
    }
    receptor_values = field_transfer.apply(donor_values)
    receptors_by_name = {item.part_name: item for item in candidate.receptors}
    part_evidence: list[MovingOversetPoissonPart] = []
    for name in names:
        source_part = source_parts[name]
        target_part = target_parts[name]
        remap = prepare_overset_conservative_remap(source_part, target_part)
        evidence = receptors_by_name[name]
        transferred = receptor_values[name]
        part_evidence.append(
            _poisson_part(
                name,
                source_discretizations[name],
                target_discretizations[name],
                source_coefficients[name],
                np.asarray(evidence.receptor_ids, dtype=np.int64),
                transferred,
                remap,
                field,
                solve_tolerance=tolerances[0],
                flux_tolerance=tolerances[1],
            )
        )

    target_part_entries = {
        entry.entry_id: entry for entry in candidate.composition_entries()
    }
    new_fe_entries: list[CompositionEntry] = []
    state_transports: list[CompositionTransport] = []
    for part in part_evidence:
        name = part.part_name
        old_fe = field_entries[name]
        old_state = state_entries[name]
        target_part_entry = target_part_entries[f"overset:part:{name}"]
        dof_map = part.discretization.dof_maps[part.discretization._field_index(field)]
        new_fe = CompositionEntry(
            part.discretization,
            entry_id=old_fe.entry_id,
            role=old_fe.role,
            owner_id=old_fe.owner_id,
            structure_id=dof_map.dof_map_id,
            revision_id=part.discretization.prepared_id,
            semantics_id=old_fe.semantics_id,
            dependencies=(target_part_entry.binding("revision"),),
        )
        state_revision = canonical_fingerprint(
            {
                "kind": "moving-overset-poisson-state",
                "update": candidate.connectivity_id,
                "part": name,
                "coefficients": array_tree_fingerprint(part.coefficients),
            }
        )
        new_state = CompositionEntry(
            part.coefficients,
            entry_id=old_state.entry_id,
            role=old_state.role,
            owner_id=old_state.owner_id,
            structure_id=dof_map.dof_map_id,
            revision_id=state_revision,
            semantics_id=old_state.semantics_id,
            dependencies=(new_fe.binding("revision"),),
        )
        new_fe_entries.append(new_fe)
        state_transports.append(
            CompositionTransport(
                "physical-remap",
                (old_state.entry_id,),
                (new_state,),
                source_structure_ids=(old_state.structure_id,),
                route_id=f"moving-overset-poisson:{name}",
                successful=jnp.asarray(part.accepted, dtype=jnp.bool_),
            )
        )
    motion = prepare_overset_motion_rebind(
        composition,
        previous,
        candidate,
        transports=(*transports, *state_transports),
        reprepare=(*reprepare, *new_fe_entries),
        retain=tuple(retain),
        invalidate=tuple(invalidate),
    )
    return PreparedMovingOversetPoissonUpdate(
        previous,
        candidate,
        tuple(part_evidence),
        motion,
    )


__all__ = [
    "MovingOversetPoissonPart",
    "PreparedMovingOversetPoissonUpdate",
    "prepare_moving_overset_poisson_update",
]
