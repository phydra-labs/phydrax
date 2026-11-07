#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Owner-local sparse overset search verification and field exchange."""

from __future__ import annotations

from collections.abc import Mapping
from math import prod
from typing import Any, assert_never, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from .._execution_runtime import ExecutionGroup
from .._fingerprint import canonical_fingerprint
from .._interpolation import GatherStencil
from .._owned_programs import run_owned_program
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._distributed_field import DistributedHaloPlan
from ..discretization._field_query import PreparedFieldQuery
from ..discretization._mapped_locator import (
    PreparedMappedCellLocator,
    PreparedOwnerLocalMappedCellMap,
)
from ..discretization._periodic_cell import PeriodicCell
from ..discretization._polyhedral_locator import PreparedPolyhedralCellLocator
from ..discretization._view_support import OwnerLocalCellGeometry
from ..discretization.fem._point_interpolation import (
    _FiniteElementRoute,
    FiniteElementFieldReconstructionKernel,
)
from ..discretization.finite_volume._field_view import (
    UnstructuredFiniteVolumeFieldReconstructionKernel,
)
from ..typing import checked, ConvertibleToArray
from ._coupling import OversetCoupling, OversetValueAction
from ._overset import (
    OversetCellStatus,
    OversetConnectivity,
    OversetReceptorEvidence,
    PreparedOversetFieldTransfer,
)


type _LocalLayout = tuple[str, tuple[int, ...], tuple[int, ...]]


def _base_stencil(
    query: PreparedFieldQuery, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not query.complete or not query.coefficient_linear:
        raise ValueError(
            "Owner-local overset exchange requires a complete coefficient-linear query."
        )
    kernel = query.reconstruction.kernel
    route = query.route
    match kernel:
        case FiniteElementFieldReconstructionKernel():
            if not isinstance(route, _FiniteElementRoute):
                raise TypeError("FE query route does not retain its scalar basis gather.")
            indices = np.asarray(route.dof_routes, dtype=np.int64).reshape(
                (query.admitted_count, -1)
            )
            weights = np.asarray(route.weights, dtype=np.float64).reshape(indices.shape)
            valid = weights != 0.0
        case UnstructuredFiniteVolumeFieldReconstructionKernel():
            if not isinstance(route, GatherStencil):
                raise TypeError(
                    "FV query route does not retain its sparse gather stencil."
                )
            indices = np.asarray(route.indices, dtype=np.int64)
            weights = np.asarray(route.weights, dtype=np.float64)
            valid = np.asarray(route.valid, dtype=np.bool_) & (weights != 0.0)
        case _:
            raise ValueError(
                "Field owner has no complete coefficient-linear sparse query route."
            )
    if (
        indices.ndim != 2
        or indices.shape[0] != query.admitted_count
        or weights.shape != indices.shape
        or valid.shape != indices.shape
        or np.any(indices[valid] < 0)
        or np.any(indices[valid] >= query.coefficient_shape[0])
    ):
        raise ValueError(
            "Field query gather indices do not match the coefficient layout."
        )
    return indices, weights, valid


def _scalar_stencil(
    query: PreparedFieldQuery, coupling: OversetCoupling, /
) -> GatherStencil:
    if coupling.field_query is None or coupling.field_query.query_id != query.query_id:
        raise ValueError(
            "Overset field coupling is missing its identical prepared query."
        )
    return _scalar_stencil_action(query, coupling.value_action, coupling.rotation)


def _scalar_stencil_action(
    query: PreparedFieldQuery,
    value_action: OversetValueAction,
    rotation: Any,
    /,
) -> GatherStencil:
    indices, weights, valid = _base_stencil(query)
    source_components = prod(query.coefficient_shape[1:])
    output_components = prod(query.value_shape)
    if source_components < 1 or output_components < 1:
        raise ValueError(
            "Overset coefficient and field component layouts must be nonempty."
        )
    match value_action:
        case "invariant":
            if source_components != output_components:
                raise ValueError(
                    "Invariant field actions require matching coefficient/value components."
                )
            component = np.arange(source_components, dtype=np.int64)
            scalar_indices = (
                indices[:, None, :] * source_components + component[None, :, None]
            ).reshape((-1, indices.shape[1]))
            scalar_weights = np.broadcast_to(
                weights[:, None, :],
                (indices.shape[0], source_components, indices.shape[1]),
            ).reshape(scalar_indices.shape)
            scalar_valid = np.broadcast_to(
                valid[:, None, :],
                (indices.shape[0], source_components, indices.shape[1]),
            ).reshape(scalar_indices.shape)
        case "polar-vector":
            matrix = np.asarray(rotation, dtype=np.float64)
            dimension = matrix.shape[0]
            if (
                matrix.shape != (dimension, dimension)
                or source_components != dimension
                or output_components != dimension
            ):
                raise ValueError(
                    "Polar-vector overset action must match its frame dimension."
                )
            component = np.arange(dimension, dtype=np.int64)
            scalar_indices = np.broadcast_to(
                indices[:, None, None, :] * dimension + component[None, None, :, None],
                (indices.shape[0], dimension, dimension, indices.shape[1]),
            ).reshape((indices.shape[0] * dimension, dimension * indices.shape[1]))
            scalar_weights = (
                matrix[None, :, :, None]
                * np.broadcast_to(
                    weights[:, None, None, :],
                    (indices.shape[0], dimension, dimension, indices.shape[1]),
                )
            ).reshape(scalar_indices.shape)
            scalar_valid = (
                np.broadcast_to(
                    valid[:, None, None, :],
                    (indices.shape[0], dimension, dimension, indices.shape[1]),
                )
                & (
                    scalar_weights.reshape(
                        (indices.shape[0], dimension, dimension, indices.shape[1])
                    )
                    != 0
                )
            ).reshape(scalar_indices.shape)
        case _:
            assert_never(value_action)
    return GatherStencil(
        indices=scalar_indices.astype(np.int32),
        weights=scalar_weights,
        valid=scalar_valid,
        source_size=prod(query.coefficient_shape),
    )


def _owner_local_query_layout(
    connectivity: OversetConnectivity,
    receptor_evidence: tuple[OversetReceptorEvidence, ...],
    group_ranks: tuple[int, ...],
    /,
) -> tuple[
    tuple[tuple[str, int], ...],
    DistributedHaloPlan,
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
]:
    group_size = len(group_ranks)
    global_names = set(connectivity.assembly.global_part_names)
    owner_by_query: dict[tuple[str, int], int] = {}
    for evidence in receptor_evidence:
        if evidence.part_name not in global_names:
            raise ValueError("Receptor evidence names an undeclared assembly part.")
        ids = np.asarray(evidence.receptor_ids, dtype=np.int64)
        owners = np.asarray(evidence.owner_ranks, dtype=np.int32)
        if ids.shape != owners.shape:
            raise ValueError("Receptor owner metadata does not align with stable IDs.")
        for identifier, owner in zip(ids.tolist(), owners.tolist(), strict=True):
            key = (evidence.part_name, int(identifier))
            if key in owner_by_query:
                raise ValueError("Owner-local receptor IDs must be unique by part.")
            if not 0 <= int(owner) < group_size:
                raise ValueError("A receptor point owner is outside the execution group.")
            owner_by_query[key] = int(owner)
    queries = tuple(sorted(owner_by_query, key=lambda item: (item[0], item[1])))
    if not queries:
        empty_adjacency = np.empty((0, 2), dtype=np.int32)
        plan = DistributedHaloPlan(
            np.zeros((1,), dtype=np.int32), empty_adjacency, group_size
        )
        return queries, plan, (), ()
    query_owners = tuple(owner_by_query[key] for key in queries)
    candidate_owners = tuple(tuple(range(group_size)) for _ in queries)
    owners = list(query_owners)
    adjacency: list[tuple[int, int]] = []
    for query_index, partitions in enumerate(candidate_owners):
        for partition in partitions:
            token = len(owners)
            owners.append(partition)
            adjacency.append((query_index, token))
    plan = DistributedHaloPlan(
        np.asarray(owners, dtype=np.int32),
        np.asarray(adjacency, dtype=np.int32).reshape((-1, 2)),
        group_size,
    )
    return queries, plan, query_owners, candidate_owners


@eqx.filter_jit
def _exchange_owner_local_halo(
    plan: DistributedHaloPlan,
    values: Array,
    mesh: Mesh,
    axis_name: str,
    /,
) -> Array:
    """Compile one complete collective; halo arrays remain dynamic arguments."""
    spec = PartitionSpec(axis_name)

    def exchange(local_values: Array, /) -> Array:
        partition = jax.lax.axis_index(axis_name)
        return plan.exchange(local_values[0], partition, axis_name=axis_name)[None, ...]

    return jax.shard_map(
        exchange,
        mesh=mesh,
        in_specs=spec,
        out_specs=spec,
        check_vma=False,
    )(values)


def _owner_local_halo_exchange(
    plan: DistributedHaloPlan,
    local_values: np.ndarray,
    devices: tuple[jax.Device, ...],
    axis_name: str,
    /,
) -> np.ndarray:
    # The axis is internal to this standalone shard_map. A call-site-specific
    # name needlessly specializes otherwise identical collective executables.
    del axis_name
    axis_name = "owner_local_halo"
    mesh = Mesh(np.asarray(devices, dtype=object), (axis_name,))
    spec = PartitionSpec(axis_name)
    distributed = jax.make_array_from_process_local_data(
        NamedSharding(mesh, spec),
        local_values[None, ...],
        global_shape=(len(devices), *local_values.shape),
    )
    exchanged = _exchange_owner_local_halo(plan, distributed, mesh, axis_name)

    if len(exchanged.addressable_shards) != 1:
        raise ValueError("Owner-local search requires one addressable process shard.")
    return np.asarray(jax.device_get(exchanged.addressable_shards[0].data))[0]


def _periodic_donor_image_frame(
    view: OwnerLocalCellGeometry,
    block: Any,
    search_mask: np.ndarray,
    /,
) -> tuple[PeriodicCell, np.ndarray, np.ndarray, float] | None:
    """Fractional reach of this rank's searchable donor cells on a periodic source.

    Periodic owner-local donors cover the quotient once, each in its own lifted
    image, so a receptor point must be located over every lattice image that can
    reach those lifted cells. Returns the source cell, the fractional lower/upper
    bounds of the searchable donor vertices and their largest cell diameter.
    """
    from ._periodic import PeriodicPointOrbits
    from .providers._native_periodic import NativePeriodicSource

    source = view.source_value
    if (
        view.periodic_lift_matrix is None
        or not isinstance(source, NativePeriodicSource)
        or not isinstance(source.domain, PeriodicPointOrbits)
        or not np.any(search_mask)
    ):
        return None
    cell = source.domain.cell
    if cell.cell_id != view.periodic_lift_id:
        raise ValueError("Periodic donor view does not bind its source lattice.")
    coordinates = np.asarray(view.vertex_coordinates, dtype=np.float64)
    rows = np.asarray(block.vertices, dtype=np.int64)[search_mask]
    valid = np.asarray(block.vertex_valid, dtype=np.bool_)[search_mask]
    cell_points = coordinates[rows]
    shifts = block.cell_vertex_shifts
    if not view.coordinates_are_lifted and shifts is not None:
        cell_points = cell_points + np.asarray(shifts, dtype=np.float64)[
            search_mask
        ] @ np.asarray(cell.vectors, dtype=np.float64)
    fractional = np.asarray(cell.fractional(cell_points[valid]), dtype=np.float64)
    diameter = 0.0
    for points, mask in zip(cell_points, valid, strict=True):
        selected = points[mask]
        diameter = max(
            diameter,
            float(
                np.max(
                    np.linalg.norm(selected[:, None, :] - selected[None, :, :], axis=-1)
                )
            ),
        )
    return cell, fractional.min(axis=0), fractional.max(axis=0), diameter


def _periodic_donor_images(
    frame: tuple[PeriodicCell, np.ndarray, np.ndarray, float] | None,
    point: np.ndarray,
    /,
) -> np.ndarray:
    """Every lattice image of ``point`` that can lie in the framed donor cells.

    The query is first translated by the integer lattice shift that brings it
    nearest the donor window, so the enumerated set is bounded by the donor
    support width rather than by the query's distance from the origin cell.
    """
    if frame is None:
        return point
    cell, lower, upper, diameter = frame
    vectors = np.asarray(cell.vectors, dtype=np.float64)
    query = np.asarray(cell.fractional(point), dtype=np.float64)[0]
    base = np.round(0.5 * (lower + upper) - query)
    centered = query + base
    excursion = tuple(
        float(value)
        for value in np.maximum(np.abs(centered - lower), np.abs(upper - centered))
    )
    stencil = cell.image_stencil(diameter, fractional_excursion=excursion)
    translations = (
        np.asarray(stencil.shifts, dtype=np.float64) + base[None, :]
    ) @ vectors
    return point + translations


def _local_donor_candidates(
    connectivity: OversetConnectivity,
    geometries: Mapping[str, OwnerLocalCellGeometry],
    queries: tuple[tuple[str, int], ...],
    query_points: Mapping[int, np.ndarray],
    process_rank: int,
    /,
) -> tuple[
    dict[int, tuple[Any, ...]],
    dict[int, set[str]],
    set[int],
]:
    from ..discretization._cell_locator import (
        CellInverseRefusal,
        CellLocationRefusalError,
        CellLocationStatus,
    )
    from ..discretization._mapped_locator import prepare_owner_local_mapped_cell_locator
    from ..discretization._polyhedral_locator import PreparedPolyhedralCellLocator
    from ._overset import _to_source

    specs = connectivity.specs
    spec_by_name = {spec.part_name: spec for spec in specs}
    part_order = {
        name: index for index, name in enumerate(connectivity.assembly.global_part_names)
    }
    priority_values = sorted({spec.priority for spec in specs}, reverse=True)
    priority_order = {value: index for index, value in enumerate(priority_values)}
    blanking_by_name = {value.part_name: value for value in connectivity.blanking}
    measure_by_part: dict[str, dict[int, float]] = {}
    active_by_part: dict[str, dict[int, bool]] = {}
    owner_by_id: dict[str, dict[int, int]] = {}
    locators_by_part: dict[
        str,
        tuple[
            tuple[
                Any,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                str,
                tuple[PeriodicCell, np.ndarray, np.ndarray, float] | None,
            ],
            ...,
        ],
    ] = {}
    for name, view in geometries.items():
        spec = spec_by_name.get(name)
        if spec is None or view.part_name != name or view.owner_rank != process_rank:
            raise ValueError(
                "Owner-local geometry does not match its declared donor owner."
            )
        blanking = blanking_by_name.get(name)
        if blanking is None:
            raise ValueError(
                "Owner-local donor lacks canonical overset blanking evidence."
            )
        status_by_id = dict(
            zip(
                np.asarray(blanking.cell_ids, dtype=np.int64).tolist(),
                np.asarray(blanking.cell_status, dtype=np.int32).tolist(),
                strict=True,
            )
        )
        ambiguous_by_id = dict(
            zip(
                np.asarray(blanking.cell_ids, dtype=np.int64).tolist(),
                np.asarray(blanking.ambiguous_cells, dtype=np.bool_).tolist(),
                strict=True,
            )
        )
        owner_by_id[name] = dict(
            zip(
                np.asarray(view.cell_ids, dtype=np.int64).tolist(),
                np.asarray(view.cell_owner_ranks, dtype=np.int32).tolist(),
                strict=True,
            )
        )
        measure_by_part[name] = dict(
            zip(
                np.asarray(view.cell_ids, dtype=np.int64).tolist(),
                np.asarray(view.cell_measures, dtype=np.float64).tolist(),
                strict=True,
            )
        )
        support_by_id = dict(
            zip(
                np.asarray(view.cell_ids, dtype=np.int64).tolist(),
                np.asarray(view.support_cell_mask, dtype=np.bool_).tolist(),
                strict=True,
            )
        )
        if (
            set(status_by_id) != set(owner_by_id[name])
            or set(support_by_id) != set(owner_by_id[name])
            or set(ambiguous_by_id) != set(owner_by_id[name])
        ):
            raise ValueError(
                "Overset blanking/support/ambiguity do not cover compact local cell IDs."
            )
        active_by_part[name] = {
            identifier: (
                status_by_id[identifier] == int(OversetCellStatus.ACTIVE)
                and not ambiguous_by_id[identifier]
            )
            for identifier in owner_by_id[name]
        }
        locators = []
        for block in view.blocks:
            block_ids = np.asarray(block.cell_ids, dtype=np.int64)
            if not block_ids.size:
                continue
            locator = (
                PreparedPolyhedralCellLocator.from_owner_local_geometry(view, block.name)
                if block.cell_kind == "polyhedron"
                else prepare_owner_local_mapped_cell_locator(view, block.name)
            )
            search_mask = np.asarray(
                [
                    owner_by_id[name][int(identifier)] == process_rank
                    for identifier in block_ids
                ],
                dtype=np.bool_,
            )
            supported = np.asarray(
                [support_by_id[int(identifier)] for identifier in block_ids],
                dtype=np.bool_,
            )
            active = np.asarray(
                [
                    active_by_part[name][int(identifier)] and search_mask[row]
                    for row, identifier in enumerate(block_ids)
                ],
                dtype=np.bool_,
            )
            locators.append(
                (
                    locator,
                    search_mask,
                    active,
                    supported,
                    block.name,
                    _periodic_donor_image_frame(view, block, search_mask),
                )
            )
        locators_by_part[name] = tuple(locators)
    best: dict[int, tuple[Any, ...]] = {}
    candidate_parts: dict[int, set[str]] = {}
    blocked: set[int] = set()
    local_queries = [index for index in range(len(queries)) if index in query_points]
    for query_index in local_queries:
        candidate_parts[query_index] = set()
    for donor_name in geometries:
        donor_spec = spec_by_name[donor_name]
        priority = priority_order[donor_spec.priority]
        donor_queries = [
            index for index in local_queries if queries[index][0] != donor_name
        ]
        if not donor_queries:
            continue
        donor_points = {
            index: _to_source(
                np.asarray(query_points[index], dtype=np.float64)[None, :],
                spec_by_name[queries[index][0]],
                donor_spec,
            )
            for index in donor_queries
        }
        for (
            locator,
            search_mask,
            active,
            supported,
            block_name,
            image_frame,
        ) in locators_by_part[donor_name]:
            # One bounded location batch per donor block: every query's donor
            # point with its complete lattice image set, grouped by query.
            image_rows = [
                _periodic_donor_images(image_frame, donor_points[index])
                for index in donor_queries
            ]
            image_queries = np.repeat(
                np.asarray(donor_queries, dtype=np.int64),
                [rows.shape[0] for rows in image_rows],
            )
            images = np.concatenate(image_rows, axis=0)
            location = locator.locate(images, cell_mask=search_mask)
            complete = np.asarray(location.candidates_complete, dtype=np.bool_)
            status = np.asarray(location.status, dtype=np.int32)
            if not np.all(complete) or not np.all(
                np.isin(
                    status,
                    (
                        int(CellLocationStatus.LOCATED),
                        int(CellLocationStatus.OUTSIDE),
                    ),
                )
            ):
                # Refusal points are re-indexed from lattice images to receptor queries.
                refusals = sorted(
                    (
                        CellInverseRefusal(
                            int(image_queries[refusal.point]),
                            refusal.cell,
                            refusal.reason,
                            refusal.lower,
                            refusal.upper,
                            refusal.witness,
                        )
                        for refusal in location.inverse_refusals
                    ),
                    key=lambda refusal: (refusal.point, refusal.cell),
                )
                raise CellLocationRefusalError(
                    "Owner-local donor search has incomplete source-bound candidates "
                    f"in donor block {block_name!r} of part {donor_name!r}; refusal "
                    "points index receptor queries and cells the block's local cells.",
                    tuple(refusals),
                )
            candidates = np.asarray(location.candidate_cells, dtype=np.int64)
            references = np.asarray(location.candidate_reference, dtype=np.float64)
            residuals = np.asarray(location.geometry_residual, dtype=np.float64)
            if (
                candidates.ndim != 2
                or candidates.shape[0] != images.shape[0]
                or references.ndim != 3
                or references.shape[:2] != candidates.shape
                or residuals.shape != (images.shape[0],)
            ):
                raise ValueError(
                    "Owner-local candidate rows do not match receptor points."
                )
            global_ids = np.asarray(locator.cell_map.cell_global_ids, dtype=np.int64)
            for image, slot in zip(*np.nonzero(candidates >= 0), strict=True):
                query_index = int(image_queries[image])
                local_cell_ = int(candidates[image, slot])
                cell_id = int(global_ids[local_cell_])
                if not active[local_cell_]:
                    blocked.add(query_index)
                    continue
                if not supported[local_cell_]:
                    raise ValueError(
                        "Owner-local donor candidate lacks completed source support "
                        f"for receptor {queries[query_index]!r} in cell {cell_id}."
                    )
                candidate_parts[query_index].add(donor_name)
                measure = measure_by_part[donor_name].get(cell_id)
                if measure is None:
                    raise ValueError(
                        "Owner-local candidate lacks its source cell measure."
                    )
                key = (
                    (measure, priority, part_order[donor_name], cell_id)
                    if connectivity.policy.donor_priority == "finest-donor"
                    else (priority, measure, part_order[donor_name], cell_id)
                )
                previous = best.get(query_index)
                if previous is None or key < previous[0]:
                    best[query_index] = (
                        key,
                        donor_name,
                        cell_id,
                        block_name,
                        local_cell_,
                        references[image, slot],
                        float(residuals[image]),
                        # The donor-frame lattice image the cell was found in.
                        images[image],
                    )
    return best, candidate_parts, blocked


@eqx.filter_jit
def _candidate_winner_program(
    distributed: tuple[Array, ...], mesh: Mesh, axis_name: str, finest: bool, /
) -> tuple[Array, ...]:
    def reduce(
        local_found: Array,
        local_measure: Array,
        local_priority: Array,
        local_part: Array,
        local_cell: Array,
        local_rank: Array,
        local_residual: Array,
        /,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        present = local_found[0]
        present_any = jax.lax.psum(present.astype(jnp.int32), axis_name) > 0
        if finest:
            best_measure = jax.lax.pmin(
                jnp.where(present, local_measure[0], jnp.inf), axis_name
            )
            measure_match = present & (local_measure[0] == best_measure)
            best_priority = jax.lax.pmin(
                jnp.where(
                    measure_match,
                    local_priority[0],
                    jnp.iinfo(jnp.int32).max,
                ),
                axis_name,
            )
            tie = measure_match & (local_priority[0] == best_priority)
        else:
            best_priority = jax.lax.pmin(
                jnp.where(
                    present,
                    local_priority[0],
                    jnp.iinfo(jnp.int32).max,
                ),
                axis_name,
            )
            priority_match = present & (local_priority[0] == best_priority)
            best_measure = jax.lax.pmin(
                jnp.where(priority_match, local_measure[0], jnp.inf), axis_name
            )
            tie = priority_match & (local_measure[0] == best_measure)
        best_part = jax.lax.pmin(
            jnp.where(tie, local_part[0], jnp.iinfo(jnp.int32).max), axis_name
        )
        tie = tie & (local_part[0] == best_part)
        best_cell = jax.lax.pmin(
            jnp.where(tie, local_cell[0], jnp.iinfo(jnp.int64).max), axis_name
        )
        tie = tie & (local_cell[0] == best_cell)
        best_rank = jax.lax.pmin(
            jnp.where(tie, local_rank[0], jnp.iinfo(jnp.int32).max), axis_name
        )
        winner = tie & (local_rank[0] == best_rank)
        best_residual = jax.lax.pmin(
            jnp.where(winner, local_residual[0], jnp.inf), axis_name
        )
        return (
            present_any,
            best_part,
            best_cell,
            best_rank,
            best_measure,
            best_residual,
        )

    spec = PartitionSpec(axis_name)
    return jax.shard_map(
        reduce,
        mesh=mesh,
        in_specs=(spec,) * 7,
        out_specs=(PartitionSpec(),) * 6,
        check_vma=False,
    )(*distributed)


def _collective_candidate_winners(
    connectivity: OversetConnectivity,
    best: Mapping[int, tuple[Any, ...]],
    query_count: int,
    partition_index: int,
    devices: tuple[jax.Device, ...],
    axis_name: str,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    specs = connectivity.specs
    part_order = {
        name: index for index, name in enumerate(connectivity.assembly.global_part_names)
    }
    priority_values = sorted({spec.priority for spec in specs}, reverse=True)
    priority_order = {value: index for index, value in enumerate(priority_values)}
    found = np.zeros((query_count,), dtype=np.bool_)
    measure = np.full((query_count,), np.inf, dtype=np.float64)
    priority = np.full((query_count,), np.iinfo(np.int32).max, dtype=np.int32)
    part = np.full((query_count,), np.iinfo(np.int32).max, dtype=np.int32)
    cell = np.full((query_count,), np.iinfo(np.int64).max, dtype=np.int64)
    rank = np.full((query_count,), np.iinfo(np.int32).max, dtype=np.int32)
    residual = np.full((query_count,), np.inf, dtype=np.float64)
    spec_by_name = {spec.part_name: spec for spec in specs}
    for query_index, candidate in best.items():
        _, donor_name, cell_id = candidate[:3]
        spec = spec_by_name[donor_name]
        found[query_index] = True
        priority[query_index] = priority_order[spec.priority]
        part[query_index] = part_order[donor_name]
        cell[query_index] = cell_id
        rank[query_index] = partition_index
        measure[query_index] = float(
            candidate[0][0]
            if connectivity.policy.donor_priority == "finest-donor"
            else candidate[0][1]
        )
        residual[query_index] = float(candidate[6])
    local_arrays = tuple(
        jnp.asarray(value)
        for value in (found, measure, priority, part, cell, rank, residual)
    )
    mesh = Mesh(np.asarray(devices, dtype=object), (axis_name,))
    spec = PartitionSpec(axis_name)
    distributed = tuple(
        jax.make_array_from_process_local_data(
            NamedSharding(mesh, spec),
            value[None, :],
            global_shape=(len(devices), query_count),
        )
        for value in local_arrays
    )
    result = _candidate_winner_program(
        distributed,
        mesh,
        axis_name,
        connectivity.policy.donor_priority == "finest-donor",
    )
    return tuple(
        np.asarray(jax.device_get(value.addressable_shards[0].data)) for value in result
    )


@eqx.filter_jit
def _candidate_count_program(
    candidates: Array, blocked: Array, mesh: Mesh, axis_name: str, /
) -> tuple[Array, Array]:
    def reduce(
        local_presence: Array, local_blocked_values: Array, /
    ) -> tuple[Array, Array]:
        presence = jax.lax.psum(local_presence[0], axis_name)
        blocked_ = jax.lax.psum(local_blocked_values[0], axis_name) > 0
        return presence > 0, blocked_

    spec = PartitionSpec(axis_name)
    return jax.shard_map(
        reduce,
        mesh=mesh,
        in_specs=(spec, spec),
        out_specs=(PartitionSpec(), PartitionSpec()),
        check_vma=False,
    )(candidates, blocked)


def _collective_candidate_counts(
    connectivity: OversetConnectivity,
    local_candidate_parts: Mapping[int, set[str]],
    local_blocked: set[int],
    query_count: int,
    devices: tuple[jax.Device, ...],
    axis_name: str,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    if not query_count:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.bool_)
    names = tuple(sorted(connectivity.assembly.global_part_names))
    part_order = {name: index for index, name in enumerate(names)}
    candidates = np.zeros((query_count, len(names)), dtype=np.int32)
    blocked = np.zeros((query_count,), dtype=np.int32)
    for query, parts in local_candidate_parts.items():
        for name in parts:
            candidates[query, part_order[name]] = 1
    for query in local_blocked:
        blocked[query] = 1
    mesh = Mesh(np.asarray(devices, dtype=object), (axis_name,))
    spec = PartitionSpec(axis_name)
    distributed_candidates = jax.make_array_from_process_local_data(
        NamedSharding(mesh, spec),
        candidates[None, ...],
        global_shape=(len(devices), *candidates.shape),
    )
    distributed_blocked = jax.make_array_from_process_local_data(
        NamedSharding(mesh, spec),
        blocked[None, :],
        global_shape=(len(devices), query_count),
    )

    result = _candidate_count_program(
        distributed_candidates, distributed_blocked, mesh, axis_name
    )
    presence = np.asarray(jax.device_get(result[0].addressable_shards[0].data))
    blocked_result = np.asarray(jax.device_get(result[1].addressable_shards[0].data))
    return np.sum(presence, axis=-1, dtype=np.int32), blocked_result.astype(np.bool_)


def _owner_local_donor_results(
    connectivity: OversetConnectivity,
    geometries: Mapping[str, OwnerLocalCellGeometry],
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> tuple[Any, ...]:
    from ._coupling import CouplingSearchStatus
    from ._overset import OversetConnectivity as Connectivity

    devices = execution_group.devices
    group_ranks = execution_group.spec.process_indices
    process_rank = jax.process_index()
    local_error: Exception | None = None
    partition_index = -1
    queries: tuple[tuple[str, int], ...] = ()
    plan: DistributedHaloPlan | None = None
    directories: list[tuple[str, OwnerLocalCellGeometry]] = []
    query_owners: tuple[int, ...] = ()
    candidate_owners: tuple[tuple[int, ...], ...] = ()
    try:
        if (
            len(devices) != len(group_ranks)
            or tuple(device.process_index for device in devices) != group_ranks
            or not execution_group.is_member
            or process_rank not in group_ranks
        ):
            raise ValueError(
                "Owner-local donor search has an invalid execution-group layout."
            )
        partition_index = group_ranks.index(process_rank)
        assembly = connectivity.assembly
        if assembly.owner_local and assembly.local_rank != partition_index:
            raise ValueError("Owner-local connectivity belongs to another group shard.")
        expected_local_names = {part.name for part in assembly.parts}
        if set(geometries) != expected_local_names:
            raise ValueError(
                "Owner-local geometry must cover this assembly's exact local parts."
            )
        if not set(geometries).issubset(set(assembly.global_part_names)):
            raise ValueError("Owner-local geometry contains an undeclared assembly part.")
        for name, view in geometries.items():
            if (
                not isinstance(view, OwnerLocalCellGeometry)
                or view.part_name != name
                or view.owner_rank != partition_index
                or view.serial_source_view
                and len(group_ranks) > 1
            ):
                raise ValueError(
                    "Owner-local search requires this rank's native source-bound cell views."
                )
            if assembly.owner_local:
                part = assembly.part(name)
                carrier = part.carrier
                if not isinstance(carrier, OwnerLocalCellGeometry):
                    raise TypeError(
                        "Owner-local search requires compact assembly carriers."
                    )
                if (
                    carrier.local_view_id != view.local_view_id
                    or carrier.distribution_id != view.distribution_id
                    or carrier.part_id != view.part_id
                    or carrier.source_topology_id != view.source_topology_id
                    or carrier.source_geometry_id != view.source_geometry_id
                    or carrier.source_id != view.source_id
                    or carrier.source_revision != view.source_revision
                    or carrier.source_map_id != view.source_map_id
                    or carrier.source_context_id != view.source_context_id
                ):
                    raise ValueError(
                        "Owner-local search geometry differs from its accepted assembly view."
                    )
        if assembly.owner_local:
            for identity in assembly.part_manifest:
                view = geometries.get(identity.name)
                if (
                    not isinstance(view, OwnerLocalCellGeometry)
                    or identity.part_id != view.part_id
                    or identity.distribution_id != view.distribution_id
                    or identity.source_topology_id != view.source_topology_id
                    or identity.source_geometry_id != view.source_geometry_id
                ):
                    raise ValueError(
                        "Owner-local search manifest differs from its current view."
                    )
                directories.append((identity.name, view))
        queries, plan, query_owners, candidate_owners = _owner_local_query_layout(
            connectivity, connectivity.receptors, group_ranks
        )
    except MemoryError:
        local_error = MemoryError(
            "Owner-local donor-search preparation failed allocation."
        )
    except Exception as error:
        local_error = error
    if _synchronize_owner_search_failure(local_error is not None, devices, axis_name):
        detail = (
            str(local_error)
            if local_error is not None
            else "another owner rejected its local search metadata"
        )
        raise ValueError(
            "Owner-local donor-search metadata failed collectively: " + detail
        )
    if local_error is not None or plan is None:
        raise ValueError("Owner-local donor-search metadata failed collectively.")
    for name, view in directories:
        view.require_owner_directory_current(
            execution_group, axis_name=f"{axis_name}_{name}_directory"
        )
    query_count = len(queries)
    if not query_count:
        resolved = Connectivity(
            connectivity.assembly,
            connectivity.specs,
            connectivity.policy,
            connectivity.blanking,
            connectivity.receptors,
            (),
            connectivity.geometries,
            epoch=connectivity.epoch,
            predecessor_id=connectivity.predecessor_id,
            connectivity_id=canonical_fingerprint(
                {
                    "kind": "owner-local-resolved-overset-connectivity",
                    "preparation": connectivity.connectivity_id,
                    "receptors": tuple(
                        item.evidence_id for item in connectivity.receptors
                    ),
                }
            ),
            global_conflict_count=connectivity.conflict_count,
            global_ambiguous_count=connectivity.ambiguous_count,
            donor_search_complete=True,
        )
        empty_i32 = np.empty((0,), dtype=np.int32)
        empty_i64 = np.empty((0,), dtype=np.int64)
        empty_f64 = np.empty((0,), dtype=np.float64)
        empty_bool = np.empty((0,), dtype=np.bool_)
        return (
            resolved,
            queries,
            query_owners,
            candidate_owners,
            {},
            {},
            empty_bool,
            empty_i32,
            empty_i64,
            empty_i32,
            empty_f64,
            empty_f64,
            empty_i32,
            empty_bool,
        )
    point_dimension = connectivity.assembly.part_manifest[0].ambient_dimension
    capacity = max(1, plan.local_capacity)
    local_points = np.zeros((capacity, point_dimension), dtype=np.float64)
    local_ids = np.asarray(plan.local_global_ids, dtype=np.int64)[partition_index]
    local_valid = np.asarray(plan.local_valid, dtype=np.bool_)[partition_index]
    local_owned = np.asarray(plan.local_owned, dtype=np.bool_)[partition_index]
    local_slots = {
        int(identifier): row
        for row, (identifier, valid) in enumerate(
            zip(local_ids, local_valid, strict=True)
        )
        if valid
    }
    receptor_error: Exception | None = None
    try:
        for query_index, ((part_name, receptor_id), owner) in enumerate(
            zip(queries, query_owners, strict=True)
        ):
            if owner != partition_index:
                continue
            view = geometries.get(part_name)
            if view is None:
                raise ValueError(
                    "Receptor owner lacks its accepted local source geometry."
                )
            row = local_slots.get(query_index)
            if row is None or not local_owned[row]:
                raise ValueError("Receptor owner lacks its owned stable query row.")
            point = np.asarray(
                view.point_coordinates(np.asarray((receptor_id,), dtype=np.int64)),
                dtype=np.float64,
            )
            if point.shape != (1, point_dimension) or not np.all(np.isfinite(point)):
                raise ValueError("Owner-local receptor point is malformed.")
            local_points[row] = point[0]
    except MemoryError:
        receptor_error = MemoryError("Owner-local receptor query failed allocation.")
    except Exception as error:
        receptor_error = error
    if _synchronize_owner_search_failure(receptor_error is not None, devices, axis_name):
        detail = (
            str(receptor_error)
            if receptor_error is not None
            else "another owner lacks a receptor point"
        )
        raise ValueError("Owner-local receptor query failed collectively: " + detail)
    points = _owner_local_halo_exchange(plan, local_points, devices, axis_name)
    # Candidate tokens only create the owner-to-candidate halo routes; each
    # candidate reads the query point from the query entity row itself, which is
    # owned on the query owner and a received halo copy everywhere else.
    points_by_query = {
        query_index: points[local_slots[query_index]]
        for query_index in range(query_count)
        if partition_index in candidate_owners[query_index] and query_index in local_slots
    }
    point_error = (
        None
        if len(points_by_query) == query_count
        else ValueError("Owner-local query exchange omitted a candidate owner.")
    )
    if _synchronize_owner_search_failure(point_error is not None, devices, axis_name):
        raise ValueError("Owner-local query exchange failed collectively.")
    local_best: dict[int, tuple[Any, ...]] = {}
    local_candidate_parts: dict[int, set[str]] = {}
    local_blocked: set[int] = set()
    search_error: Exception | None = None
    try:
        local_best, local_candidate_parts, local_blocked = _local_donor_candidates(
            connectivity,
            geometries,
            queries,
            points_by_query,
            partition_index,
        )
    except MemoryError:
        search_error = MemoryError("Owner-local source search failed allocation.")
    except Exception as error:
        search_error = error
    if _synchronize_owner_search_failure(search_error is not None, devices, axis_name):
        detail = (
            str(search_error)
            if search_error is not None
            else "another owner reported incomplete source-location evidence"
        )
        raise ValueError("Owner-local donor search failed collectively: " + detail)
    (
        found,
        winner_part,
        winner_cell,
        winner_rank,
        winner_measure,
        winner_residual,
    ) = _collective_candidate_winners(
        connectivity,
        local_best,
        query_count,
        partition_index,
        devices,
        axis_name,
    )
    candidate_counts, blocked = _collective_candidate_counts(
        connectivity,
        local_candidate_parts,
        local_blocked,
        query_count,
        devices,
        axis_name,
    )
    status = np.where(
        found,
        int(CouplingSearchStatus.FOUND),
        np.where(
            blocked,
            int(CouplingSearchStatus.EXCLUDED_DONOR),
            int(CouplingSearchStatus.OUTSIDE),
        ),
    ).astype(np.int32)
    donor_parts = np.where(found, winner_part, -1).astype(np.int32)
    donor_cells = np.where(found, winner_cell, -1).astype(np.int64)
    measures = np.where(found, winner_measure, np.inf).astype(np.float64)
    residuals = np.where(found, winner_residual, np.inf).astype(np.float64)
    query_index_by_key = {key: index for index, key in enumerate(queries)}
    receptor_rows_by_key: dict[tuple[str, int], tuple[int, int]] = {}
    for evidence in connectivity.receptors:
        for identifier, row, owner in zip(
            np.asarray(evidence.receptor_ids, dtype=np.int64),
            np.asarray(evidence.receptor_rows, dtype=np.int32),
            np.asarray(evidence.owner_ranks, dtype=np.int32),
            strict=True,
        ):
            receptor_rows_by_key[(evidence.part_name, int(identifier))] = (
                int(row),
                int(owner),
            )
    resolved_receptors = []
    for evidence in connectivity.receptors:
        keys = tuple(
            (evidence.part_name, int(identifier))
            for identifier in np.asarray(evidence.receptor_ids, dtype=np.int64)
        )
        indices = np.asarray([query_index_by_key[key] for key in keys], dtype=np.int64)
        owners = np.asarray(
            [receptor_rows_by_key[key][1] for key in keys], dtype=np.int32
        )
        resolved_receptors.append(
            OversetReceptorEvidence(
                evidence.part_name,
                evidence.receptor_ids,
                evidence.receptor_rows,
                status[indices],
                donor_parts[indices],
                donor_cells[indices],
                candidate_counts[indices],
                residuals[indices],
                measures[indices],
                owners,
                # The owner-local donor search raises with its refusals instead.
                donor_refusals=(),
            )
        )
    resolved_receptors_tuple = tuple(resolved_receptors)
    resolved = Connectivity(
        connectivity.assembly,
        connectivity.specs,
        connectivity.policy,
        connectivity.blanking,
        resolved_receptors_tuple,
        (),
        connectivity.geometries,
        epoch=connectivity.epoch,
        predecessor_id=connectivity.predecessor_id,
        connectivity_id=canonical_fingerprint(
            {
                "kind": "owner-local-resolved-overset-connectivity",
                "preparation": connectivity.connectivity_id,
                "receptors": tuple(item.evidence_id for item in resolved_receptors_tuple),
            }
        ),
        global_conflict_count=connectivity.conflict_count,
        global_ambiguous_count=connectivity.ambiguous_count,
        donor_search_complete=True,
    )
    return (
        resolved,
        queries,
        query_owners,
        candidate_owners,
        points_by_query,
        local_best,
        found,
        winner_part,
        winner_cell,
        winner_rank,
        winner_measure,
        winner_residual,
        candidate_counts,
        blocked,
    )


def _validate_local_donor_search(
    transfer: PreparedOversetFieldTransfer,
    geometries: Mapping[str, OwnerLocalCellGeometry],
    execution_group: ExecutionGroup,
    process_rank: int,
    partition_index: int,
    /,
    *,
    axis_name: str,
) -> None:
    """Collectively search only owner-local geometry and validate winner metadata."""
    specs = transfer.connectivity.specs
    spec_by_name = {spec.part_name: spec for spec in specs}
    group_ranks = execution_group.spec.process_indices
    global_names = set(spec_by_name)
    local_error: Exception | None = None
    queries: tuple[tuple[str, int], ...] = ()
    plan: DistributedHaloPlan | None = None
    query_owners: tuple[int, ...] = ()
    candidate_owners: tuple[tuple[int, ...], ...] = ()
    points_by_query: dict[int, np.ndarray] = {}
    try:
        expected_local_names = {
            part.name for part in transfer.connectivity.assembly.parts
        }
        if set(geometries) != expected_local_names:
            raise ValueError(
                "Owner-local geometry must cover this assembly's exact local part inventory."
            )
        if not set(geometries).issubset(global_names):
            raise ValueError("Owner-local geometry contains an undeclared part name.")
        if any(
            not isinstance(view, OwnerLocalCellGeometry)
            or view.owner_rank != partition_index
            or (view.serial_source_view and len(group_ranks) > 1)
            for name, view in geometries.items()
        ):
            raise ValueError(
                "Owner-local exchange requires native source-bound local parts, not serial mesh slices."
            )
        queries, plan, query_owners, candidate_owners = _owner_local_query_layout(
            transfer.connectivity, transfer.receptors, group_ranks
        )
    except MemoryError:
        local_error = MemoryError("Owner-local source preparation failed allocation.")
    except Exception as error:
        local_error = error
    if _synchronize_owner_search_failure(
        local_error is not None, execution_group.devices, axis_name
    ):
        detail = (
            str(local_error)
            if local_error is not None
            else "another owner rejected its local source view"
        )
        raise ValueError("Owner-local source metadata failed collectively: " + detail)
    if local_error is not None or plan is None:
        raise ValueError("Owner-local source metadata failed collectively.")
    if not queries:
        return
    query_count = len(queries)
    point_dimension = transfer.connectivity.assembly.part_manifest[0].ambient_dimension
    capacity = max(1, plan.local_capacity)
    local_points = np.zeros((capacity, point_dimension), dtype=np.float64)
    local_ids = np.asarray(plan.local_global_ids, dtype=np.int64)[partition_index]
    local_valid = np.asarray(plan.local_valid, dtype=np.bool_)[partition_index]
    local_owned = np.asarray(plan.local_owned, dtype=np.bool_)[partition_index]
    local_slots = {
        int(identifier): row
        for row, (identifier, valid) in enumerate(
            zip(local_ids, local_valid, strict=True)
        )
        if valid
    }
    receptor_views_error: Exception | None = None
    try:
        for query_index, ((part_name, receptor_id), owner) in enumerate(
            zip(queries, query_owners, strict=True)
        ):
            if owner != partition_index:
                continue
            view = geometries.get(part_name)
            if view is None:
                raise ValueError("Receptor owner lacks its local source geometry.")
            row = local_slots.get(query_index)
            if row is None or not local_owned[row]:
                raise ValueError("Receptor owner lacks its owned stable query row.")
            point = np.asarray(
                view.point_coordinates(np.asarray((receptor_id,), dtype=np.int64)),
                dtype=np.float64,
            )
            if point.shape != (1, point_dimension) or not np.all(np.isfinite(point)):
                raise ValueError("Owner-local receptor point is malformed.")
            local_points[row] = point[0]
    except MemoryError:
        receptor_views_error = MemoryError(
            "Owner-local receptor preparation failed allocation."
        )
    except Exception as error:
        receptor_views_error = error
    if _synchronize_owner_search_failure(
        receptor_views_error is not None, execution_group.devices, axis_name
    ):
        detail = (
            str(receptor_views_error)
            if receptor_views_error is not None
            else "another owner lacks a receptor query row"
        )
        raise ValueError(
            "Owner-local receptor preparation failed collectively: " + detail
        )
    points = _owner_local_halo_exchange(
        plan, local_points, execution_group.devices, axis_name
    )
    local_query_indices = tuple(
        index
        for index, owners in enumerate(candidate_owners)
        if partition_index in owners
    )
    points_by_query: dict[int, np.ndarray] = {}
    for query_index in local_query_indices:
        row = local_slots.get(query_index)
        if row is not None:
            points_by_query[query_index] = points[row]
    point_error = (
        None
        if set(points_by_query) == set(local_query_indices)
        else ValueError("Owner-local query exchange omitted a candidate owner.")
    )
    if _synchronize_owner_search_failure(
        point_error is not None, execution_group.devices, axis_name
    ):
        raise ValueError("Owner-local query exchange failed collectively.")
    local_best: dict[int, tuple[Any, ...]] = {}
    search_error: Exception | None = None
    try:
        local_best, local_candidate_parts, local_blocked = _local_donor_candidates(
            transfer.connectivity,
            geometries,
            queries,
            points_by_query,
            partition_index,
        )
    except MemoryError:
        search_error = MemoryError("Owner-local source search failed allocation.")
    except Exception as error:
        search_error = error
    if _synchronize_owner_search_failure(
        search_error is not None, execution_group.devices, axis_name
    ):
        detail = (
            str(search_error)
            if search_error is not None
            else "another owner reported incomplete source-location evidence"
        )
        raise ValueError("Owner-local donor search failed collectively: " + detail)
    found, winner_part, winner_cell, winner_rank, winner_measure, winner_residual = (
        _collective_candidate_winners(
            transfer.connectivity,
            local_best,
            query_count,
            partition_index,
            execution_group.devices,
            axis_name,
        )
    )
    candidate_counts, _blocked = _collective_candidate_counts(
        transfer.connectivity,
        local_candidate_parts,
        local_blocked,
        query_count,
        execution_group.devices,
        axis_name,
    )
    evidence_by_key: dict[tuple[str, int], tuple[int, float, float]] = {}
    for evidence in transfer.receptors:
        for receptor_id, candidates, measure, residual in zip(
            np.asarray(evidence.receptor_ids, dtype=np.int64),
            np.asarray(evidence.candidate_parts, dtype=np.int32),
            np.asarray(evidence.donor_measures, dtype=np.float64),
            np.asarray(evidence.residuals, dtype=np.float64),
            strict=True,
        ):
            evidence_by_key[(evidence.part_name, int(receptor_id))] = (
                int(candidates),
                float(measure),
                float(residual),
            )
    part_names = tuple(sorted(spec.part_name for spec in specs))
    published: dict[tuple[str, int], tuple[str, int, int]] = {}
    for route in transfer.routes:
        packet = route.packet
        for receptor_id, donor_cell in zip(
            np.asarray(packet.receptor_ids, dtype=np.int64),
            np.asarray(packet.donor_cells, dtype=np.int64),
            strict=True,
        ):
            key = (packet.receptor_part, int(receptor_id))
            if key in published:
                raise ValueError("A receptor has more than one published donor route.")
            published[key] = (
                packet.donor_part,
                int(donor_cell),
                packet.donor_rank,
            )
    for query_index, key in enumerate(queries):
        expected = published.get(key)
        expected_evidence = evidence_by_key.get(key)
        if expected is None or expected_evidence is None or not bool(found[query_index]):
            raise ValueError("Complete overset transfer has an orphan local receptor.")
        part_index = int(winner_part[query_index])
        cell_id = int(winner_cell[query_index])
        if (
            part_index < 0
            or part_index >= len(part_names)
            or (part_names[part_index], cell_id, int(winner_rank[query_index]))
            != expected
        ):
            raise ValueError(
                "Owner-local donor winner differs from published connectivity."
            )
        if (
            int(candidate_counts[query_index]) != expected_evidence[0]
            or not np.isclose(
                float(winner_measure[query_index]),
                expected_evidence[1],
                rtol=1e-10,
                atol=1e-12,
            )
            or not np.isclose(
                float(winner_residual[query_index]),
                expected_evidence[2],
                rtol=1e-10,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "Owner-local candidate evidence differs from registered connectivity."
            )


@eqx.filter_jit
def _flag_sum_program(values: Array, mesh: Mesh, axis_name: str, /) -> Array:
    def reduce(local_values: Array, /) -> Array:
        return jax.lax.psum(local_values, axis_name=axis_name)

    return jax.shard_map(
        reduce,
        mesh=mesh,
        in_specs=PartitionSpec(axis_name),
        out_specs=PartitionSpec(axis_name),
        check_vma=False,
    )(values)


def _metadata_agree_local(local_values: Array, /) -> Array:
    local_digest = local_values[0]
    lower = jax.lax.pmin(local_digest, axis_name="overset_metadata")
    upper = jax.lax.pmax(local_digest, axis_name="overset_metadata")
    return jnp.asarray((jnp.all(lower == upper),), dtype=jnp.bool_)


@eqx.filter_jit
def _metadata_agreement_program(values: Array, mesh: Mesh, /) -> Array:
    return jax.shard_map(
        _metadata_agree_local,
        mesh=mesh,
        in_specs=PartitionSpec("overset_metadata"),
        out_specs=PartitionSpec("overset_metadata"),
        check_vma=False,
    )(values)


def _synchronize_owner_search_failure(
    failed: bool, devices: tuple[jax.Device, ...], axis_name: str, /
) -> bool:
    mesh = Mesh(np.asarray(devices, dtype=object), (axis_name,))
    sharding = NamedSharding(mesh, PartitionSpec(axis_name))
    local = np.asarray((int(failed),), dtype=np.int32)
    distributed = jax.make_array_from_process_local_data(
        sharding, local, global_shape=(len(devices),)
    )

    result = _flag_sum_program(distributed, mesh, axis_name)
    return int(np.asarray(jax.device_get(result.addressable_shards[0].data))[0]) != 0


def _owner_metadata_mismatch(
    metadata_id: str, devices: tuple[jax.Device, ...], /
) -> bool:
    digest = np.frombuffer(bytes.fromhex(metadata_id), dtype=np.uint8).astype(np.uint32)
    mesh = Mesh(np.asarray(devices, dtype=object), ("overset_metadata",))
    sharding = NamedSharding(mesh, PartitionSpec("overset_metadata"))
    distributed = jax.make_array_from_process_local_data(
        sharding,
        digest[None, :],
        global_shape=(len(devices), digest.size),
    )

    result = _metadata_agreement_program(distributed, mesh)
    agreed = bool(np.asarray(jax.device_get(result.addressable_shards[0].data))[0])
    return not agreed


@eqx.filter_jit
def _gather_owner_local_rows(values: Array, mesh: Mesh, axis_name: str, /) -> Array:
    def gather(local: Array, /) -> Array:
        return jax.lax.all_gather(local[0], axis_name, axis=0)

    return jax.shard_map(
        gather,
        mesh=mesh,
        in_specs=PartitionSpec(axis_name),
        out_specs=PartitionSpec(),
        check_vma=False,
    )(values)


def _owner_local_all_gather_rows(
    local_values: np.ndarray,
    devices: tuple[jax.Device, ...],
    axis_name: str,
    /,
) -> tuple[np.ndarray, ...]:
    """Allgather variable-length query metadata, never cell or field payloads."""
    values = np.asarray(local_values)
    if values.ndim < 1:
        raise ValueError("Owner-local collective rows require a leading row dimension.")
    mesh = Mesh(np.asarray(devices, dtype=object), (axis_name,))
    sharding = NamedSharding(mesh, PartitionSpec(axis_name))
    local_count = np.asarray((values.shape[0],), dtype=np.int64)
    distributed_count = jax.make_array_from_process_local_data(
        sharding, local_count[None, :], global_shape=(len(devices), 1)
    )

    gathered_count = _gather_owner_local_rows(distributed_count, mesh, axis_name)
    counts = np.asarray(jax.device_get(gathered_count.addressable_shards[0].data))[:, 0]
    padded_count = max(1, int(np.max(counts, initial=0)))
    padded = np.zeros((padded_count, *values.shape[1:]), dtype=values.dtype)
    padded[: values.shape[0]] = values
    distributed = jax.make_array_from_process_local_data(
        sharding,
        padded[None, ...],
        global_shape=(len(devices), *padded.shape),
    )

    gathered = _gather_owner_local_rows(distributed, mesh, axis_name)
    rows = np.asarray(jax.device_get(gathered.addressable_shards[0].data))
    return tuple(
        np.asarray(rows[rank, : int(counts[rank])]) for rank in range(len(devices))
    )


def _owner_local_field_exchange_state(
    transfer: PreparedOversetFieldTransfer,
    coefficient_ids: Mapping[str, ConvertibleToArray],
    execution_group: ExecutionGroup,
    owner_local_geometry: Mapping[str, OwnerLocalCellGeometry],
    /,
    *,
    axis_name: str,
) -> tuple[Any, ...]:
    from ._overset import (
        _to_target,
        OversetConnectivityError,
        OversetDonorPacket,
    )

    connectivity = transfer.connectivity
    devices = execution_group.devices
    group_ranks = execution_group.spec.process_indices
    process_rank = jax.process_index()
    partition_index = group_ranks.index(process_rank)
    part_count = len(group_ranks)
    axis = str(axis_name).strip()
    request_id = canonical_fingerprint(
        {
            "kind": "owner-local-overset-exchange-request",
            "connectivity": connectivity.connectivity_id,
            "transfer": transfer.transfer_id,
            "field": transfer.field_name,
            "value_action": transfer.value_action,
            "group": execution_group.spec.group_id,
            "processes": group_ranks,
            "axis": axis,
        }
    )
    if _owner_metadata_mismatch(request_id, devices):
        raise OversetConnectivityError(
            "Owner-local overset field-transfer metadata differ across processes.",
            connectivity,
        )
    if not axis:
        raise OversetConnectivityError(
            "Owner-local overset exchange axis_name must be non-empty.",
            connectivity,
        )
    try:
        (
            resolved,
            queries,
            query_owners,
            _candidate_owners,
            points_by_query,
            local_best,
            found,
            winner_parts,
            winner_cells,
            winner_ranks,
            _winner_measures,
            _winner_residuals,
            _candidate_counts,
            _blocked,
        ) = _owner_local_donor_results(
            connectivity, owner_local_geometry, execution_group, axis
        )
    except Exception as error:
        if isinstance(error, OversetConnectivityError):
            raise
        raise OversetConnectivityError(str(error), connectivity) from error
    if not resolved.complete:
        raise OversetConnectivityError(
            f"Owner-local overset has {resolved.orphan_count} orphan receptors and "
            f"{resolved.conflict_count} protected-wall conflicts.",
            resolved,
        )
    global_names = resolved.assembly.global_part_names
    part_order = {name: index for index, name in enumerate(global_names)}
    spec_by_name = {spec.part_name: spec for spec in resolved.specs}
    selected_donors = tuple(
        sorted(
            {
                global_names[int(part)]
                for evidence in resolved.receptors
                for part in np.asarray(evidence.donor_parts, dtype=np.int32)
                if int(part) >= 0
            }
        )
    )
    if not selected_donors:
        halo = DistributedHaloPlan(
            np.zeros((1,), dtype=np.int32),
            np.empty((0, 2), dtype=np.int32),
            part_count,
        )
        route_indices = np.zeros((part_count, 1, 1), dtype=np.int32)
        route_weights = np.zeros((part_count, 1, 1), dtype=np.float64)
        route_valid = np.zeros((part_count, 1, 1), dtype=np.bool_)
        output_local_indices = np.zeros((part_count, 1), dtype=np.int32)
        output_valid = np.zeros((part_count, 1), dtype=np.bool_)
        return (
            resolved,
            halo,
            jnp.asarray(route_indices),
            jnp.asarray(route_weights),
            jnp.asarray(route_valid),
            jnp.asarray(output_local_indices),
            jnp.asarray(output_valid),
            devices,
            axis,
            partition_index,
            (),
            (),
            (),
            (),
            halo.local_capacity,
            1,
            transfer.transfer_id,
            canonical_fingerprint(
                {
                    "kind": "owner-local-overset-exchange",
                    "connectivity": resolved.connectivity_id,
                    "transfer": transfer.transfer_id,
                    "empty": True,
                }
            ),
            (),
        )

    local_reconstructions = dict(transfer.local_reconstructions)
    local_error: Exception | None = None
    local_ids: dict[str, np.ndarray] = {}
    local_owners: dict[str, np.ndarray] = {}
    local_shapes: dict[str, tuple[int, ...]] = {}
    local_components: dict[str, int] = {}
    local_kinds: dict[str, int] = {}
    try:
        if not isinstance(coefficient_ids, Mapping) or set(coefficient_ids) != set(
            local_reconstructions
        ):
            raise ValueError(
                "Local coefficient IDs must match this rank's prepared field bindings."
            )
        for name, reconstruction in local_reconstructions.items():
            if name not in selected_donors:
                continue
            view = owner_local_geometry.get(name)
            if view is None or view.part_name != name:
                raise ValueError(
                    "A local donor reconstruction lacks its source-bound cell view."
                )
            shape = tuple(int(value) for value in reconstruction.coefficient_shape)
            ids = np.asarray(coefficient_ids[name])
            if (
                not shape
                or ids.shape != (shape[0],)
                or ids.dtype.kind not in "iu"
                or np.any(ids < 0)
                or np.unique(ids).size != ids.size
                or ids.size
                and int(np.max(ids)) > np.iinfo(np.int64).max
            ):
                raise ValueError(
                    "Stable local coefficient IDs must provide one unique nonnegative ID per row."
                )
            ids = ids.astype(np.int64, copy=False)
            cell_ids = np.asarray(view.cell_ids, dtype=np.int64)
            vertex_ids = np.asarray(view.vertex_ids, dtype=np.int64)
            kernel = reconstruction.kernel
            cell_binding = isinstance(
                kernel, UnstructuredFiniteVolumeFieldReconstructionKernel
            )
            matches_cells = bool(np.all(np.isin(ids, cell_ids)))
            matches_vertices = bool(np.all(np.isin(ids, vertex_ids)))
            if cell_binding and matches_cells:
                directory_kind, kind = "mesh-cells", 0
            elif matches_vertices:
                directory_kind, kind = "mesh-vertices", 1
            elif matches_cells:
                directory_kind, kind = "mesh-cells", 0
            else:
                raise ValueError(
                    "Local field coefficient IDs must bind the exact source cell or vertex bank."
                )
            owners = view.owner_ranks_for_ids(ids, directory_kind=directory_kind)
            components = prod(shape[1:])
            if tuple(shape[1:]) != tuple(reconstruction.value_shape):
                raise ValueError(
                    "Stable local coefficients do not match the prepared field component shape."
                )
            if components < 1:
                raise ValueError(
                    "Local field coefficient rows must have scalar components."
                )
            local_ids[name] = ids
            local_owners[name] = np.asarray(owners, dtype=np.int32)
            local_shapes[name] = tuple(reconstruction.value_shape)
            local_components[name] = components
            local_kinds[name] = kind
    except MemoryError:
        local_error = MemoryError(
            "Owner-local coefficient preparation failed allocation."
        )
    except Exception as error:
        local_error = error
    if _synchronize_owner_search_failure(
        local_error is not None, devices, f"{axis}_field_preflight"
    ):
        detail = (
            str(local_error)
            if local_error is not None
            else "another owner rejected its coefficient IDs"
        )
        raise OversetConnectivityError(
            "Owner-local coefficient metadata failed collectively: " + detail,
            resolved,
        )
    if local_error is not None:
        raise OversetConnectivityError(
            "Owner-local coefficient metadata failed collectively.", resolved
        )

    local_max_value_rank = max((len(shape) for shape in local_shapes.values()), default=0)
    rank_rows = _owner_local_all_gather_rows(
        np.asarray((local_max_value_rank,), dtype=np.int64),
        devices,
        f"{axis}_field_metadata",
    )
    max_value_rank = max((int(rows[0]) for rows in rank_rows if rows.size), default=0)
    local_shape_rows: list[np.ndarray] = []
    local_coefficient_rows: list[tuple[int, int, int, int, int]] = []
    local_reconstruction_rows: list[np.ndarray] = []
    for name in selected_donors:
        reconstruction = local_reconstructions.get(name)
        if reconstruction is None:
            continue
        name_index = part_order[name]
        shape = local_shapes[name]
        row = np.full((4 + max_value_rank,), -1, dtype=np.int64)
        row[0] = name_index
        row[1] = local_kinds[name]
        row[2] = len(shape)
        row[3 : 3 + len(shape)] = shape
        row[-1] = local_components[name]
        local_shape_rows.append(row)
        reconstruction_digest = np.frombuffer(
            bytes.fromhex(reconstruction.reconstruction_id), dtype=np.uint8
        ).astype(np.int64)
        local_reconstruction_rows.append(
            np.concatenate(
                (
                    np.asarray((name_index, partition_index), dtype=np.int64),
                    reconstruction_digest,
                )
            )
        )
        local_coefficient_rows.extend(
            (
                name_index,
                int(identifier),
                int(owner),
                local_kinds[name],
                local_components[name],
            )
            for identifier, owner in zip(local_ids[name], local_owners[name], strict=True)
        )
    shape_rows = _owner_local_all_gather_rows(
        np.asarray(local_shape_rows, dtype=np.int64).reshape((-1, 4 + max_value_rank)),
        devices,
        f"{axis}_field_metadata",
    )
    coefficient_rows = _owner_local_all_gather_rows(
        np.asarray(local_coefficient_rows, dtype=np.int64).reshape((-1, 5)),
        devices,
        f"{axis}_field_metadata",
    )
    gathered_reconstructions = _owner_local_all_gather_rows(
        np.asarray(local_reconstruction_rows, dtype=np.int64).reshape((-1, 34)),
        devices,
        f"{axis}_field_metadata",
    )
    reconstruction_ids = tuple(
        sorted(
            (
                global_names[int(row[0])],
                int(row[1]),
                np.asarray(row[2:], dtype=np.uint8).tobytes().hex(),
            )
            for rows in gathered_reconstructions
            for row in rows
        )
    )
    shape_by_part: dict[str, tuple[int, ...]] = {}
    components_by_part: dict[str, int] = {}
    kinds_by_part: dict[str, int] = {}
    metadata_error: str | None = None
    for rows in shape_rows:
        for row in rows:
            name_index = int(row[0])
            if not 0 <= name_index < len(global_names):
                metadata_error = "Local field metadata names an undeclared donor part."
                continue
            name = global_names[name_index]
            rank = int(row[2])
            if rank < 0 or rank > max_value_rank:
                metadata_error = "Local field value shape metadata is malformed."
                continue
            value_shape = tuple(int(value) for value in row[3 : 3 + rank])
            kind = int(row[1])
            components = int(row[-1])
            if (
                any(value < 1 for value in value_shape)
                or kind not in (0, 1)
                or components < 1
                or name in shape_by_part
                and (
                    shape_by_part[name] != value_shape
                    or components_by_part[name] != components
                    or kinds_by_part[name] != kind
                )
            ):
                metadata_error = "Donor field layouts differ across owner-local views."
                continue
            shape_by_part[name] = value_shape
            components_by_part[name] = components
            kinds_by_part[name] = kind
    if any(name not in shape_by_part for name in selected_donors):
        metadata_error = (
            "A selected donor has no source-bound local field reconstruction."
        )
    selected_shapes = {
        shape_by_part[name] for name in selected_donors if name in shape_by_part
    }
    if len(selected_shapes) != 1:
        metadata_error = "Selected donor fields must share one output value shape."
    if metadata_error is not None:
        raise OversetConnectivityError(metadata_error, resolved)
    value_shape = next(iter(selected_shapes))
    value_components = prod(value_shape)
    if value_components < 1:
        raise OversetConnectivityError(
            "Overset receptor fields must have at least one scalar component.",
            resolved,
        )
    owner_by_coefficient: dict[tuple[str, int], int] = {}
    for rows in coefficient_rows:
        for name_index, identifier, owner, kind, components in rows:
            name = global_names[int(name_index)]
            if name not in selected_donors:
                continue
            if (
                not 0 <= int(owner) < part_count
                or int(kind) != kinds_by_part[name]
                or int(components) != components_by_part[name]
            ):
                raise OversetConnectivityError(
                    "Local coefficient ownership differs from the source directory.",
                    resolved,
                )
            key = (name, int(identifier))
            previous = owner_by_coefficient.get(key)
            if previous is not None and previous != int(owner):
                raise OversetConnectivityError(
                    "Stable field coefficient has conflicting owner ranks.", resolved
                )
            owner_by_coefficient[key] = int(owner)
    if any(
        not any(part_name == name for part_name, _ in owner_by_coefficient)
        for name in selected_donors
    ):
        raise OversetConnectivityError(
            "A selected donor has no complete local coefficient owner bank.", resolved
        )
    for name in selected_donors:
        view = owner_local_geometry.get(name)
        if view is None or view.part_name != name:
            local_error = ValueError(
                f"Selected donor {name!r} has no current compact local geometry view."
            )
            break
        if kinds_by_part[name] == 0:
            ids = np.asarray(view.cell_ids, dtype=np.int64)
            owners = np.asarray(view.cell_owner_ranks, dtype=np.int32)
        else:
            ids = np.asarray(view.vertex_ids, dtype=np.int64)
            owners = np.asarray(view.vertex_owner_ranks, dtype=np.int32)
        locally_owned = ids[owners == partition_index]
        if locally_owned.size and name not in local_ids:
            local_error = ValueError(
                f"Rank {partition_index} owns donor field coefficients for {name!r} "
                "without a local source-bound reconstruction."
            )
            break
    if _synchronize_owner_search_failure(
        local_error is not None, devices, f"{axis}_field_preflight"
    ):
        detail = (
            str(local_error)
            if local_error is not None
            else "another owner lacks local field coefficients"
        )
        raise OversetConnectivityError(
            "Owner-local field ownership failed collectively: " + detail, resolved
        )

    route_rows_local: list[tuple[int, int, int, int, int]] = []
    route_weights_local: list[float] = []
    route_error: Exception | None = None
    try:
        for query_index, (
            (receptor_name, receptor_id),
            is_found,
            part_index,
            cell_id,
            winner_rank,
        ) in enumerate(
            zip(
                queries,
                np.asarray(found, dtype=np.bool_),
                np.asarray(winner_parts, dtype=np.int32),
                np.asarray(winner_cells, dtype=np.int64),
                np.asarray(winner_ranks, dtype=np.int32),
                strict=True,
            )
        ):
            if not is_found or int(winner_rank) != partition_index:
                continue
            donor_name = global_names[int(part_index)]
            reconstruction = local_reconstructions.get(donor_name)
            if reconstruction is None:
                raise ValueError(
                    "Selected donor cell owner lacks its local field reconstruction."
                )
            candidate = local_best.get(query_index)
            if (
                candidate is None
                or candidate[1] != donor_name
                or int(candidate[2]) != int(cell_id)
            ):
                raise ValueError(
                    "Collective donor winner lacks local source-location evidence."
                )
            kernel = reconstruction.kernel
            if not isinstance(
                kernel,
                (
                    FiniteElementFieldReconstructionKernel,
                    UnstructuredFiniteVolumeFieldReconstructionKernel,
                ),
            ):
                raise TypeError(
                    "Selected donor requires a source-bound FE or FV reconstruction."
                )
            locator = kernel.locator.canonical_locator
            if not isinstance(
                locator, (PreparedMappedCellLocator, PreparedPolyhedralCellLocator)
            ):
                raise TypeError(
                    "Selected donor requires a mapped or polyhedral source locator."
                )
            if isinstance(locator, PreparedPolyhedralCellLocator):
                cell_ids = np.asarray(locator.cell_map.cell_global_ids, dtype=np.int64)
            else:
                cell_map = locator.cell_map
                if not isinstance(cell_map, PreparedOwnerLocalMappedCellMap):
                    raise TypeError(
                        "Selected donor map must retain its owner-local source binding."
                    )
                cell_ids = np.asarray(cell_map.cell_global_ids, dtype=np.int64)
            cell_rows = np.flatnonzero(cell_ids == int(cell_id))
            if cell_rows.size != 1:
                raise ValueError(
                    "Selected donor cell is absent or duplicated in its field locator."
                )
            cell_row = int(cell_rows[0])
            target_spec = spec_by_name[receptor_name]
            donor_spec = spec_by_name[donor_name]
            # Evaluate at the exact lattice image in which the donor was located;
            # its receptor-frame site is the same periodic point's matching image.
            donor_point = np.asarray(candidate[7], dtype=np.float64)[None, :]
            target_point = _to_target(donor_point, donor_spec, target_spec)[0]
            packet = OversetDonorPacket(
                resolved.assembly.declared_part(donor_name),
                resolved.assembly.declared_part(receptor_name),
                donor_spec,
                target_spec,
                np.asarray((receptor_id,), dtype=np.int64),
                np.asarray((0,), dtype=np.int32),
                donor_point,
                target_point[None, :],
                np.asarray((cell_id,), dtype=np.int64),
                np.asarray((candidate[4],), dtype=np.int32),
                np.asarray(candidate[5], dtype=np.float64)[None, :],
                donor_block=str(candidate[3]),
                donor_rank=partition_index,
                receptor_rank=int(query_owners[query_index]),
            )
            if reconstruction.trace_policy.kind == "cell-sided":
                query = reconstruction.prepare_query(
                    packet.query_points,
                    side="owner",
                    cell_ids=np.asarray((cell_row,), dtype=np.int32),
                )
            else:
                query = reconstruction.prepare_query(packet.query_points)
            stencil = _scalar_stencil_action(
                query, transfer.value_action, packet.rotation
            )
            indices = np.asarray(stencil.indices, dtype=np.int64)
            weights = np.asarray(stencil.weights, dtype=np.float64)
            valid = np.asarray(stencil.valid, dtype=np.bool_)
            source_components = local_components[donor_name]
            if (
                indices.shape != weights.shape
                or valid.shape != indices.shape
                or indices.shape[0] != value_components
            ):
                raise ValueError(
                    "Prepared local field stencil does not match its value layout."
                )
            for output_component in range(value_components):
                for column in np.flatnonzero(valid[output_component]):
                    coefficient = int(indices[output_component, column])
                    source_row, source_component = divmod(coefficient, source_components)
                    if source_row < 0 or source_row >= local_ids[donor_name].size:
                        raise ValueError(
                            "Local field route requests an absent stable coefficient row."
                        )
                    route_rows_local.append(
                        (
                            query_index,
                            output_component,
                            part_order[donor_name],
                            int(local_ids[donor_name][source_row]),
                            source_component,
                        )
                    )
                    route_weights_local.append(float(weights[output_component, column]))
    except MemoryError:
        route_error = MemoryError(
            "Owner-local field query preparation failed allocation."
        )
    except Exception as error:
        route_error = error
    if _synchronize_owner_search_failure(
        route_error is not None, devices, f"{axis}_field_routes"
    ):
        detail = (
            str(route_error)
            if route_error is not None
            else "another donor owner rejected its field query"
        )
        raise OversetConnectivityError(
            "Owner-local field query failed collectively: " + detail, resolved
        )
    if route_error is not None:
        raise OversetConnectivityError(
            "Owner-local field query failed collectively.", resolved
        )
    local_route_rows = np.asarray(route_rows_local, dtype=np.int64).reshape((-1, 5))
    local_route_weights = np.asarray(route_weights_local, dtype=np.float64).reshape(
        (-1, 1)
    )
    gathered_routes = _owner_local_all_gather_rows(
        local_route_rows, devices, f"{axis}_field_routes"
    )
    gathered_weights = _owner_local_all_gather_rows(
        local_route_weights, devices, f"{axis}_field_routes"
    )

    entity_keys: list[tuple[str, str, int, int]] = []
    entity_owners: list[int] = []
    coefficient_rows: dict[tuple[str, int, int], int] = {}
    for name in selected_donors:
        identifiers = sorted(
            identifier
            for part_name, identifier in owner_by_coefficient
            if part_name == name
        )
        for identifier in identifiers:
            for component in range(components_by_part[name]):
                row = len(entity_keys)
                entity_keys.append(("coefficient", name, identifier, component))
                entity_owners.append(owner_by_coefficient[name, identifier])
                coefficient_rows[name, identifier, component] = row
    receptor_keys: list[tuple[str, int, int]] = []
    output_entity_rows: dict[tuple[str, int, int], int] = {}
    for evidence in resolved.receptors:
        ids = np.asarray(evidence.receptor_ids, dtype=np.int64)
        owners = np.asarray(evidence.owner_ranks, dtype=np.int32)
        if owners.shape != ids.shape:
            raise OversetConnectivityError(
                "Receptor owner ranks do not align with stable IDs.", resolved
            )
        for identifier, owner in zip(ids, owners, strict=True):
            for component in range(value_components):
                key = (evidence.part_name, int(identifier), component)
                output_entity_rows[key] = len(entity_keys)
                receptor_keys.append(key)
                entity_keys.append(("receptor", *key))
                entity_owners.append(int(owner))
    route_table: dict[int, list[tuple[int, float]]] = {}
    route_sender: dict[int, int] = {}
    adjacency: set[tuple[int, int]] = set()
    for sender, (rows, weights) in enumerate(
        zip(gathered_routes, gathered_weights, strict=True)
    ):
        if rows.shape[0] != weights.shape[0]:
            raise OversetConnectivityError(
                "Collective field stencil rows lost their weights.", resolved
            )
        for metadata, weight_row in zip(rows, weights, strict=True):
            query_index, output_component, donor_index, identifier, source_component = (
                int(value) for value in metadata
            )
            if (
                not 0 <= query_index < len(queries)
                or not 0 <= donor_index < len(global_names)
                or not 0 <= output_component < value_components
            ):
                raise OversetConnectivityError(
                    "Collective field stencil metadata is malformed.", resolved
                )
            receptor_name, receptor_id = queries[query_index]
            output_global = output_entity_rows[
                (receptor_name, receptor_id, output_component)
            ]
            previous_sender = route_sender.get(output_global)
            if previous_sender is not None and previous_sender != sender:
                raise OversetConnectivityError(
                    "A receptor field route has multiple donor owners.", resolved
                )
            route_sender[output_global] = sender
            donor_name = global_names[donor_index]
            coefficient_global = coefficient_rows.get(
                (donor_name, identifier, source_component)
            )
            if coefficient_global is None:
                raise OversetConnectivityError(
                    "Field stencil references an unowned stable coefficient ID.",
                    resolved,
                )
            weight = float(weight_row[0])
            if not np.isfinite(weight) or weight == 0.0:
                raise OversetConnectivityError(
                    "Field stencil contains an invalid scalar weight.", resolved
                )
            route_table.setdefault(output_global, []).append((coefficient_global, weight))
            adjacency.add((coefficient_global, output_global))
    if len(route_table) != len(receptor_keys) or any(
        not route_table.get(index)
        for index in range(len(entity_keys) - len(receptor_keys), len(entity_keys))
    ):
        raise OversetConnectivityError(
            "Every local receptor coefficient requires one complete field route.",
            resolved,
        )
    if len(entity_keys) > np.iinfo(np.int32).max:
        raise OversetConnectivityError(
            "Owner-local field relation exceeds the canonical int32 index domain.",
            resolved,
        )
    owners = np.asarray(entity_owners, dtype=np.int32)
    adjacency_rows = np.asarray(sorted(adjacency), dtype=np.int32).reshape((-1, 2))
    halo = DistributedHaloPlan(owners, adjacency_rows, part_count)
    local_ids_grid = np.asarray(halo.local_global_ids, dtype=np.int64)
    local_valid_grid = np.asarray(halo.local_valid, dtype=np.bool_)
    local_lookups = [
        {
            int(global_id): row
            for row, global_id in enumerate(local_ids_grid[index])
            if local_valid_grid[index, row]
        }
        for index in range(part_count)
    ]
    outputs_by_part = [
        [
            index
            for index, owner in enumerate(owners)
            if owner == part and entity_keys[index][0] == "receptor"
        ]
        for part in range(part_count)
    ]
    output_capacity = max(1, max((len(rows) for rows in outputs_by_part), default=0))
    route_width = max(
        1,
        max(
            (len(route_table[index]) for rows in outputs_by_part for index in rows),
            default=0,
        ),
    )
    route_indices = np.zeros((part_count, output_capacity, route_width), dtype=np.int32)
    route_weights = np.zeros((part_count, output_capacity, route_width), dtype=np.float64)
    route_valid = np.zeros(route_indices.shape, dtype=np.bool_)
    output_local_indices = np.zeros((part_count, output_capacity), dtype=np.int32)
    output_valid = np.zeros(output_local_indices.shape, dtype=np.bool_)
    for part, output_rows in enumerate(outputs_by_part):
        for slot, output_global in enumerate(output_rows):
            output_local = local_lookups[part].get(output_global)
            if output_local is None:
                raise OversetConnectivityError(
                    "Receptor owner lacks its local output scalar row.", resolved
                )
            output_local_indices[part, slot] = output_local
            output_valid[part, slot] = True
            for column, (coefficient_global, weight) in enumerate(
                route_table[output_global]
            ):
                row = local_lookups[part].get(coefficient_global)
                if row is None:
                    raise OversetConnectivityError(
                        "Receptor owner lacks a requested donor coefficient halo row.",
                        resolved,
                    )
                route_indices[part, slot, column] = row
                route_weights[part, slot, column] = weight
                route_valid[part, slot, column] = True

    local_donor_layouts = []
    local_field_ids = []
    for name in selected_donors:
        reconstruction = local_reconstructions.get(name)
        if reconstruction is None:
            continue
        ids = local_ids[name]
        row_owners = local_owners[name]
        owned_rows = np.flatnonzero(row_owners == partition_index)
        if not owned_rows.size:
            continue
        scalar_rows = []
        for input_row in owned_rows:
            for component in range(local_components[name]):
                global_row = coefficient_rows[name, int(ids[input_row]), component]
                local_row = local_lookups[partition_index].get(global_row)
                if local_row is None or not bool(
                    np.asarray(halo.local_owned)[partition_index, local_row]
                ):
                    raise ValueError(
                        "Donor field owner lacks its canonical owned coefficient row."
                    )
                scalar_rows.append(local_row)
        shape = (owned_rows.size, *reconstruction.coefficient_shape[1:])
        local_donor_layouts.append((name, shape, tuple(scalar_rows)))
        local_field_ids.append((name, tuple(int(value) for value in ids[owned_rows])))
    local_receptor_layouts = []
    receptor_global_ids = []
    for evidence in resolved.receptors:
        ids = np.asarray(evidence.receptor_ids, dtype=np.int64)
        row_owners = np.asarray(evidence.owner_ranks, dtype=np.int32)
        owned_ids = tuple(
            int(identifier)
            for identifier, owner in zip(ids, row_owners, strict=True)
            if int(owner) == partition_index
        )
        if not owned_ids:
            continue
        scalar_rows = tuple(
            local_lookups[partition_index][
                output_entity_rows[(evidence.part_name, identifier, component)]
            ]
            for identifier in owned_ids
            for component in range(value_components)
        )
        local_receptor_layouts.append(
            (
                evidence.part_name,
                (len(owned_ids), *value_shape),
                scalar_rows,
            )
        )
        receptor_global_ids.append((evidence.part_name, owned_ids))
    global_field_ids = tuple(
        (
            name,
            tuple(
                sorted(
                    identifier
                    for part_name, identifier in owner_by_coefficient
                    if part_name == name
                )
            ),
        )
        for name in selected_donors
    )
    exchange_id = canonical_fingerprint(
        {
            "kind": "owner-local-overset-exchange",
            "transfer": transfer.transfer_id,
            "connectivity": resolved.connectivity_id,
            "field_shapes": tuple(
                (name, shape_by_part[name], components_by_part[name])
                for name in selected_donors
            ),
            "field_ids": global_field_ids,
            "reconstructions": reconstruction_ids,
            "entities": entity_keys,
            "owners": owners,
            "adjacency": adjacency_rows,
            "routes": tuple(
                (output, tuple(route_table[output])) for output in sorted(route_table)
            ),
        }
    )
    return (
        resolved,
        halo,
        jnp.asarray(route_indices),
        jnp.asarray(route_weights),
        jnp.asarray(route_valid),
        jnp.asarray(output_local_indices),
        jnp.asarray(output_valid),
        devices,
        axis,
        partition_index,
        tuple(local_donor_layouts),
        tuple(local_receptor_layouts),
        tuple(receptor_global_ids),
        tuple(local_field_ids),
        halo.local_capacity,
        output_capacity,
        transfer.transfer_id,
        exchange_id,
        value_shape,
    )


@final
class PreparedOwnerLocalOversetExchange(StrictModule, NonTrainableState):
    """Collective owner-local interpolation using canonical halo and transpose routes.

    Each process supplies only its owned donor or receptor field arrays. The
    prepared sparse route metadata is shared, while coefficient payloads remain
    on their owner and move only along the actual support relation.
    """

    halo: DistributedHaloPlan
    route_indices: Array
    route_weights: Array
    route_valid: Array
    output_local_indices: Array
    output_valid: Array
    devices: tuple[jax.Device, ...] = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    partition_index: int = eqx.field(static=True)
    donor_layouts: tuple[_LocalLayout, ...] = eqx.field(static=True)
    receptor_layouts: tuple[_LocalLayout, ...] = eqx.field(static=True)
    receptor_global_ids: tuple[tuple[str, tuple[int, ...]], ...] = eqx.field(static=True)
    field_ids: tuple[tuple[str, tuple[int, ...]], ...] = eqx.field(static=True)
    local_capacity: int = eqx.field(static=True)
    output_capacity: int = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)
    exchange_id: str = eqx.field(static=True)
    connectivity: OversetConnectivity
    value_shape: tuple[int, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        transfer: PreparedOversetFieldTransfer,
        coefficient_ids: Mapping[str, ConvertibleToArray],
        execution_group: ExecutionGroup,
        owner_local_geometry: Mapping[str, OwnerLocalCellGeometry],
        /,
        *,
        axis_name: str = "overset",
    ) -> None:
        if not isinstance(owner_local_geometry, Mapping):
            raise TypeError(
                "owner_local_geometry must map local part names to source-bound views."
            )
        if transfer.owner_local:
            (
                self.connectivity,
                self.halo,
                self.route_indices,
                self.route_weights,
                self.route_valid,
                self.output_local_indices,
                self.output_valid,
                self.devices,
                self.axis_name,
                self.partition_index,
                self.donor_layouts,
                self.receptor_layouts,
                self.receptor_global_ids,
                self.field_ids,
                self.local_capacity,
                self.output_capacity,
                self.transfer_id,
                self.exchange_id,
                self.value_shape,
            ) = _owner_local_field_exchange_state(
                transfer,
                coefficient_ids,
                execution_group,
                owner_local_geometry,
                axis_name=axis_name,
            )
            return
        devices = execution_group.devices
        parts = len(devices)
        process_rank = jax.process_index()
        group_ranks = execution_group.spec.process_indices
        axis = str(axis_name).strip()
        if (
            len(devices) != len(group_ranks)
            or tuple(device.process_index for device in devices) != group_ranks
            or not execution_group.is_member
        ):
            raise ValueError(
                "Overset owner exchange requires one group device per registered owner process."
            )
        owner_by_name = {
            spec.part_name: spec.owner_rank for spec in transfer.connectivity.specs
        }
        receptor_owners = tuple(
            int(owner)
            for evidence in transfer.receptors
            for owner in np.asarray(evidence.owner_ranks, dtype=np.int32)
        )
        local_group_error: Exception | None = None
        if not axis:
            local_group_error = ValueError("axis_name must be non-empty.")
        elif any(
            not isinstance(view, OwnerLocalCellGeometry)
            for view in owner_local_geometry.values()
        ):
            local_group_error = TypeError(
                "owner_local_geometry must contain canonical source-bound views."
            )
        else:
            local_cell_owners = {
                int(rank)
                for view in owner_local_geometry.values()
                for rank in np.asarray(view.cell_owner_ranks, dtype=np.int32)
            }
            registered_owners = (
                set(owner_by_name.values()) | set(receptor_owners) | local_cell_owners
            )
            if any(
                owner < 0 or owner >= jax.process_count() for owner in registered_owners
            ) or not registered_owners.issubset(set(group_ranks)):
                local_group_error = ValueError(
                    "Execution group omits a declared part, receptor, or local-cell owner."
                )
        if _synchronize_owner_search_failure(
            local_group_error is not None, devices, "overset_preflight"
        ):
            detail = (
                str(local_group_error)
                if local_group_error is not None
                else "another process has an invalid owner-group contract"
            )
            raise ValueError(
                "Overset owner-group validation failed collectively: " + detail
            )
        partition_by_rank = {process: part for part, process in enumerate(group_ranks)}
        owner_partition_by_name = {
            name: partition_by_rank[owner] for name, owner in owner_by_name.items()
        }
        partition_index = partition_by_rank[process_rank]
        id_banks: dict[str, np.ndarray] = {}
        coefficient_shapes = dict(
            zip(transfer.donor_names, transfer.coefficient_shapes, strict=True)
        )
        local_metadata_error: Exception | None = None
        local_memory_error = False
        try:
            if set(coefficient_ids) != set(transfer.donor_names):
                raise ValueError(
                    "Stable coefficient IDs must name every field used by this transfer."
                )
            for name in transfer.donor_names:
                raw = np.asarray(coefficient_ids[name])
                if (
                    raw.shape != (coefficient_shapes[name][0],)
                    or raw.dtype.kind not in "iu"
                    or np.any(raw < 0)
                    or np.unique(raw).size != raw.size
                    or raw.size
                    and int(np.max(raw)) > np.iinfo(np.int64).max
                ):
                    raise ValueError(
                        "Stable coefficient IDs must provide one unique nonnegative int64 ID per field coefficient row."
                    )
                id_banks[name] = raw.astype(np.int64)
        except MemoryError:
            local_memory_error = True
        except Exception as error:
            local_metadata_error = error
        if _synchronize_owner_search_failure(
            local_memory_error, devices, "overset_preflight"
        ):
            raise MemoryError(
                "Overset coefficient preparation failed allocation collectively."
            )
        if _synchronize_owner_search_failure(
            local_metadata_error is not None, devices, "overset_preflight"
        ):
            detail = (
                str(local_metadata_error)
                if local_metadata_error is not None
                else "another owner supplied invalid coefficient IDs"
            )
            raise ValueError(
                "Overset coefficient metadata failed collectively: " + detail
            )
        metadata_id = canonical_fingerprint(
            {
                "kind": "owner-local-overset-metadata",
                "transfer": transfer.transfer_id,
                "connectivity": transfer.connectivity.connectivity_id,
                "donors": transfer.donor_names,
                "coefficient_shapes": transfer.coefficient_shapes,
                "value_shape": transfer.value_shape,
                "owners": tuple(sorted(owner_by_name.items())),
                "processes": group_ranks,
                "mesh_axes": execution_group.spec.mesh_axes,
                "axis": axis,
                "coefficient_ids": tuple(
                    (name, id_banks[name]) for name in transfer.donor_names
                ),
            }
        )
        if _owner_metadata_mismatch(metadata_id, devices):
            raise ValueError(
                "Overset owner metadata differ across execution-group processes."
            )
        entity_keys: list[tuple[str, str, int, int]] = []
        entity_owners: list[int] = []
        coefficient_rows: dict[tuple[str, int, int], int] = {}
        field_entity_rows: dict[str, tuple[int, ...]] = {}
        for name in transfer.donor_names:
            ids = id_banks[name]
            owner = owner_partition_by_name[name]
            rows = []
            component_count = prod(coefficient_shapes[name][1:])
            for identifier in sorted(int(value) for value in ids):
                for component in range(component_count):
                    global_row = len(entity_keys)
                    entity_keys.append(("coefficient", name, identifier, component))
                    entity_owners.append(owner)
                    coefficient_rows[name, identifier, component] = global_row
            rows.extend(
                coefficient_rows[name, int(value), component]
                for value in ids
                for component in range(component_count)
            )
            field_entity_rows[name] = tuple(rows)
        value_components = prod(transfer.value_shape)
        if value_components < 1:
            raise ValueError(
                "Overset field outputs require at least one scalar component."
            )
        receptor_keys: list[tuple[str, int, int]] = []
        for evidence in transfer.receptors:
            ids = np.asarray(evidence.receptor_ids, dtype=np.int64)
            receptor_owners = np.asarray(evidence.owner_ranks, dtype=np.int32)
            if receptor_owners.shape != ids.shape:
                raise ValueError("Receptor owner ranks do not align with stable IDs.")
            for identifier, owner_rank in zip(ids, receptor_owners, strict=True):
                owner = partition_by_rank[int(owner_rank)]
                for component in range(value_components):
                    receptor_keys.append((evidence.part_name, int(identifier), component))
                    entity_keys.append(
                        ("receptor", evidence.part_name, int(identifier), component)
                    )
                    entity_owners.append(owner)
        output_entity_rows = {
            key: len(entity_keys) - len(receptor_keys) + index
            for index, key in enumerate(receptor_keys)
        }
        route_table: dict[int, tuple[tuple[int, float], ...]] = {}
        adjacency: set[tuple[int, int]] = set()
        for route in transfer.routes:
            packet = route.packet
            stencil = _scalar_stencil(route.query, route.coupling)
            indices = np.asarray(stencil.indices, dtype=np.int64)
            weights = np.asarray(stencil.weights, dtype=np.float64)
            valid = np.asarray(stencil.valid, dtype=np.bool_)
            if indices.shape != weights.shape or valid.shape != indices.shape:
                raise ValueError(
                    "Overset scalar routes have incompatible CSR stencil arrays."
                )
            if indices.shape[0] != packet.receptor_ids.size * value_components:
                raise ValueError(
                    "Overset query outputs do not align with packet receptor IDs."
                )
            field_ids = id_banks[packet.donor_part]
            source_components = prod(coefficient_shapes[packet.donor_part][1:])
            for point, receptor_id in enumerate(
                np.asarray(packet.receptor_ids, dtype=np.int64)
            ):
                for component in range(value_components):
                    row = point * value_components + component
                    output_key = (packet.receptor_part, int(receptor_id), component)
                    output_global = output_entity_rows[output_key]
                    if output_global in route_table:
                        raise ValueError(
                            "Overset receptor scalar has more than one owner route."
                        )
                    support = []
                    for column in np.flatnonzero(valid[row]):
                        coefficient_row = int(indices[row, column])
                        if coefficient_row < 0 or coefficient_row >= prod(
                            coefficient_shapes[packet.donor_part]
                        ):
                            raise ValueError(
                                "Overset query requests an absent flattened coefficient row."
                            )
                        source_row, source_component = divmod(
                            coefficient_row, source_components
                        )
                        coefficient_global = coefficient_rows[
                            packet.donor_part,
                            int(field_ids[source_row]),
                            source_component,
                        ]
                        weight = float(weights[row, column])
                        support.append((coefficient_global, weight))
                        adjacency.add((coefficient_global, output_global))
                    if not support:
                        raise ValueError(
                            "Every overset receptor scalar requires a nonempty donor action."
                        )
                    route_table[output_global] = tuple(support)
        if len(route_table) != len(receptor_keys):
            raise ValueError(
                "Every receptor scalar must have exactly one owner-local field action."
            )
        if len(entity_keys) > np.iinfo(np.int32).max:
            raise ValueError(
                "Owner-local overset relation exceeds the canonical int32 local-index domain."
            )
        owners = np.asarray(entity_owners, dtype=np.int32)
        adjacency_rows = np.asarray(sorted(adjacency), dtype=np.int32).reshape((-1, 2))
        halo = DistributedHaloPlan(owners, adjacency_rows, parts)
        local_ids = np.asarray(halo.local_global_ids, dtype=np.int64)
        local_valid = np.asarray(halo.local_valid, dtype=np.bool_)
        local_lookups: list[dict[int, int]] = []
        for part in range(parts):
            local_lookups.append(
                {
                    int(global_id): row
                    for row, global_id in enumerate(local_ids[part])
                    if local_valid[part, row]
                }
            )
        outputs_by_part = [
            [
                index
                for index, owner in enumerate(owners)
                if owner == part and entity_keys[index][0] == "receptor"
            ]
            for part in range(parts)
        ]
        output_capacity = max(1, max((len(rows) for rows in outputs_by_part), default=0))
        route_width = max(
            1,
            max(
                (len(route_table[index]) for rows in outputs_by_part for index in rows),
                default=0,
            ),
        )
        if receptor_keys and (not any(outputs_by_part) or not any(route_table.values())):
            raise ValueError(
                "Owner-local overset exchange lacks receptor routes or donor support."
            )
        route_indices = np.zeros((parts, output_capacity, route_width), dtype=np.int32)
        route_weights = np.zeros((parts, output_capacity, route_width), dtype=np.float64)
        route_valid = np.zeros(route_indices.shape, dtype=np.bool_)
        output_local_indices = np.zeros((parts, output_capacity), dtype=np.int32)
        output_valid = np.zeros(output_local_indices.shape, dtype=np.bool_)
        for part, output_rows in enumerate(outputs_by_part):
            for slot, output_global in enumerate(output_rows):
                output_key = entity_keys[output_global]
                output_local = local_lookups[part].get(output_global)
                if output_local is None:
                    raise ValueError("Receptor owner lacks its local output scalar row.")
                output_local_indices[part, slot] = output_local
                output_valid[part, slot] = True
                for column, (coefficient_global, weight) in enumerate(
                    route_table[output_global]
                ):
                    row = local_lookups[part].get(coefficient_global)
                    if row is None:
                        raise ValueError(
                            "Receptor owner lacks a requested donor coefficient halo row."
                        )
                    route_indices[part, slot, column] = row
                    route_weights[part, slot, column] = weight
                    route_valid[part, slot, column] = True
        owner_specs = tuple((name, owner_by_name[name]) for name in transfer.donor_names)
        donor_layouts = []
        receptor_layouts = []
        local_search_error: Exception | None = None
        local_memory_error = False
        try:
            for name, owner in owner_specs:
                if owner != process_rank:
                    continue
                field_index = transfer.donor_names.index(name)
                shape = transfer.coefficient_shapes[field_index]
                local_rows = tuple(
                    local_lookups[partition_index][global_row]
                    for global_row in field_entity_rows[name]
                )
                if any(
                    not bool(np.asarray(halo.local_owned)[partition_index, row])
                    for row in local_rows
                ):
                    raise ValueError(
                        "Donor field coefficients are not owned by their declared process."
                    )
                donor_layouts.append((name, shape, local_rows))
            receptor_global_ids = []
            for evidence in transfer.receptors:
                ids_all = np.asarray(evidence.receptor_ids, dtype=np.int64)
                owner_ranks = np.asarray(evidence.owner_ranks, dtype=np.int32)
                ids = tuple(
                    int(identifier)
                    for identifier, owner_rank in zip(ids_all, owner_ranks, strict=True)
                    if int(owner_rank) == process_rank
                )
                if not ids:
                    continue
                rows = tuple(
                    local_lookups[partition_index][
                        output_entity_rows[(evidence.part_name, identifier, component)]
                    ]
                    for identifier in ids
                    for component in range(value_components)
                )
                receptor_layouts.append(
                    (evidence.part_name, (len(ids), *transfer.value_shape), rows)
                )
                receptor_global_ids.append((evidence.part_name, ids))
            _validate_local_donor_search(
                transfer,
                owner_local_geometry,
                execution_group,
                process_rank,
                partition_index,
                axis_name=axis,
            )
        except MemoryError:
            local_memory_error = True
        except Exception as error:
            local_search_error = error
        if _synchronize_owner_search_failure(local_memory_error, devices, axis):
            raise MemoryError(
                "Owner-local overset preparation failed allocation collectively."
            )
        if _synchronize_owner_search_failure(
            local_search_error is not None, devices, axis
        ):
            detail = (
                str(local_search_error)
                if local_search_error is not None
                else "another owner reported incomplete source-location evidence"
            )
            raise ValueError("Owner-local overset search failed collectively: " + detail)
        self.halo = halo
        self.route_indices = jnp.asarray(route_indices)
        self.route_weights = jnp.asarray(route_weights)
        self.route_valid = jnp.asarray(route_valid)
        self.output_local_indices = jnp.asarray(output_local_indices)
        self.output_valid = jnp.asarray(output_valid)
        self.devices = devices
        self.axis_name = axis
        self.partition_index = partition_index
        self.donor_layouts = tuple(donor_layouts)
        self.receptor_layouts = tuple(receptor_layouts)
        self.receptor_global_ids = tuple(receptor_global_ids)
        self.field_ids = tuple(
            (name, tuple(int(value) for value in id_banks[name]))
            for name in transfer.donor_names
        )
        self.local_capacity = halo.local_capacity
        self.output_capacity = output_capacity
        self.transfer_id = transfer.transfer_id
        self.exchange_id = canonical_fingerprint(
            {
                "kind": "owner-local-overset-exchange",
                "transfer": transfer.transfer_id,
                "fields": self.field_ids,
                "owners": owner_specs,
                "entities": entity_keys,
                "adjacency": adjacency_rows,
                "routes": route_indices,
                "weights": route_weights,
            }
        )
        self.connectivity = transfer.connectivity
        self.value_shape = transfer.value_shape

    def apply(
        self, coefficients: Mapping[str, ConvertibleToArray], /
    ) -> dict[str, Array]:
        """Collectively evaluate this process's receptors from owned donor fields."""
        values = self._pack(coefficients, self.donor_layouts)
        distributed = self._distributed(values)
        result = owner_local_overset_action(self, distributed, False)
        local = _local_shard(result)
        return {
            name: jnp.asarray(local[np.asarray(rows, dtype=np.int32)]).reshape(shape)
            for name, shape, rows in self.receptor_layouts
        }

    def transpose(
        self, cotangents: Mapping[str, ConvertibleToArray], /
    ) -> dict[str, Array]:
        """Collectively scatter this process's receptor duals to donor owners."""
        values = self._pack(cotangents, self.receptor_layouts)
        distributed = self._distributed(values)
        result = owner_local_overset_action(self, distributed, True)
        local = _local_shard(result)
        return {
            name: jnp.asarray(local[np.asarray(rows, dtype=np.int32)]).reshape(shape)
            for name, shape, rows in self.donor_layouts
        }

    def _pack(
        self,
        values: Mapping[str, ConvertibleToArray],
        layouts: tuple[_LocalLayout, ...],
        /,
    ) -> np.ndarray:
        if set(values) != {name for name, _, _ in layouts}:
            raise ValueError(
                "Owner-local overset values must match this process's exact field inventory."
            )
        arrays = {name: jnp.asarray(value) for name, value in values.items()}
        if any(
            not array.is_fully_addressable or array.shape != shape
            for name, shape, _ in layouts
            for array in (arrays[name],)
        ):
            raise ValueError(
                "Owner-local overset arrays must have their exact local shapes and be addressable."
            )
        dtype = np.dtype(
            jnp.result_type(jnp.float64, *(array.dtype for array in arrays.values()))
        )
        packed = np.zeros((self.local_capacity,), dtype=dtype)
        for name, shape, rows in layouts:
            local = np.asarray(jax.device_get(arrays[name])).reshape(-1)
            if local.size != len(rows):
                raise ValueError(
                    "Owner-local field rows do not match their scalar entity layout."
                )
            packed[np.asarray(rows, dtype=np.int32)] = local.astype(dtype, copy=False)
        return packed

    def _distributed(self, local: np.ndarray, /) -> Array:
        mesh = Mesh(np.asarray(self.devices, dtype=object), (self.axis_name,))
        sharding = NamedSharding(mesh, PartitionSpec(self.axis_name))
        return jax.make_array_from_process_local_data(
            sharding,
            local[None, :],
            global_shape=(len(self.devices), self.local_capacity),
        )


def _local_shard(value: Array, /) -> np.ndarray:
    if len(value.addressable_shards) != 1:
        raise ValueError(
            "Owner-local overset action requires one addressable process shard."
        )
    return np.asarray(jax.device_get(value.addressable_shards[0].data))[0]


def owner_local_overset_action(
    exchange: PreparedOwnerLocalOversetExchange,
    values: Array,
    transpose: bool,
    /,
) -> Array:
    """Sparse forward or exact transpose action over an owner-local halo exchange.

    Each direction runs as one compiled program owned by ``exchange``.
    """
    return run_owned_program(
        exchange,
        "transpose-action" if transpose else "action",
        _overset_action_program,
        values,
        bool(transpose),
    )


def _overset_action_program(
    exchange: PreparedOwnerLocalOversetExchange,
    values: Array,
    transpose: bool,
    /,
) -> Array:
    axis = exchange.axis_name
    mesh = Mesh(np.asarray(exchange.devices, dtype=object), (axis,))
    spec = PartitionSpec(axis)

    def action(local: Array, /) -> Array:
        part = jax.lax.axis_index(axis)
        current = local[0]
        output_rows = exchange.output_local_indices[part]
        output_mask = exchange.output_valid[part]
        route_rows = exchange.route_indices[part]
        route_weights = exchange.route_weights[part]
        route_mask = exchange.route_valid[part] & output_mask[:, None]
        if transpose:
            dual = jnp.where(output_mask, current[output_rows], 0.0)
            scatter = (
                jnp.zeros((exchange.local_capacity,), dtype=current.dtype)
                .at[route_rows]
                .add(jnp.where(route_mask, dual[:, None] * route_weights, 0.0))
            )
            result = exchange.halo.accumulate_halo(scatter, part, axis_name=axis)
        else:
            support = exchange.halo.exchange(current, part, axis_name=axis)
            gathered = support[route_rows]
            routed = jnp.sum(jnp.where(route_mask, gathered * route_weights, 0.0), axis=1)
            result = (
                jnp.zeros_like(current)
                .at[output_rows]
                .add(jnp.where(output_mask, routed, 0.0))
            )
        return result[None, :]

    return jax.shard_map(
        action,
        mesh=mesh,
        in_specs=spec,
        out_specs=spec,
        check_vma=False,
    )(values)


def prepare_owner_local_overset_exchange(
    transfer: PreparedOversetFieldTransfer,
    coefficient_ids: Mapping[str, ConvertibleToArray],
    execution_group: ExecutionGroup,
    owner_local_geometry: Mapping[str, OwnerLocalCellGeometry],
    /,
    *,
    axis_name: str = "overset",
) -> PreparedOwnerLocalOversetExchange:
    return PreparedOwnerLocalOversetExchange(
        transfer,
        coefficient_ids,
        execution_group,
        owner_local_geometry,
        axis_name=axis_name,
    )


__all__ = [
    "PreparedOwnerLocalOversetExchange",
    "owner_local_overset_action",
    "prepare_owner_local_overset_exchange",
]
