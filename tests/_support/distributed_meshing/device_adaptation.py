# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Device adaptation scenarios returning numerical and lifecycle observations."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization import AdaptiveSimplexPolicy, AdaptiveSimplexUpdate, CellMesh
from phydrax.discretization._adaptive_simplex import (
    adaptive_simplex_state,
    AdaptiveSimplexLayout,
    AdaptiveSimplexParts,
    AdaptiveSimplexState,
    refine_adaptive_simplex,
    refine_adaptive_simplex_parts,
)
from phydrax.meshing import (
    BisectionHierarchy,
    CellMeshingResult,
    CollectiveMeshEvidence,
    commit_adaptive_simplex,
    commit_partitioned_adaptive_simplex,
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    MeshingFailure,
    MeshPart,
    MeshPartitionKind,
    MeshPartitionPolicy,
    partition_adaptive_simplex,
    PartitionedAdaptiveSimplex,
    prepare_adaptive_simplex,
    prepare_mesh_adaptation,
    prepare_mesh_distribution,
    PreparedAdaptiveSimplex,
)
from tests._support.cell_meshes import triangle_grid


def _certified(mesh: CellMesh) -> CellMeshingResult:
    return phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


def _policy(route: MeshAdaptationRoute, **options: Any) -> MeshAdaptationPolicy:
    if route is MeshAdaptationRoute.DEVICE_BISECTION:
        options.setdefault("device_policy", AdaptiveSimplexPolicy())
    return MeshAdaptationPolicy(route, **options)


def _adapt(
    route: Any,
    source: Any,
    refine: Any = (),
    coarsen: Any = (),
    /,
    *,
    hierarchy: Any = None,
    **options: Any,
) -> Any:
    request = MarkedMeshAdaptation(
        np.asarray(refine, dtype=np.int64),
        np.asarray(coarsen, dtype=np.int64),
        hierarchy=hierarchy,
    )
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(source, request, policy=_policy(route, **options))
    )


def _cells(mesh: CellMesh) -> np.ndarray:
    return np.concatenate(tuple(np.asarray(block.vertices) for block in mesh.blocks))


def _cell_ids(mesh: CellMesh) -> np.ndarray:
    return np.concatenate(tuple(np.asarray(block.global_ids) for block in mesh.blocks))


def _chain_state(
    indices: np.ndarray,
    layout: AdaptiveSimplexLayout,
    points: np.ndarray,
    rows: np.ndarray,
    protected_last: bool,
) -> AdaptiveSimplexState:
    global_rows = rows[indices]
    vertex_ids = np.unique(global_rows).astype(np.int64)
    local_rows = np.searchsorted(vertex_ids, global_rows).astype(np.int32)
    count = indices.size
    protected = np.empty((0, 2), dtype=np.int32)
    if protected_last and 3 in indices:
        protected = np.searchsorted(vertex_ids, np.asarray([[0, 5]])).astype(np.int32)
    return adaptive_simplex_state(
        layout,
        coordinates=points[vertex_ids],
        vertex_ids=vertex_ids,
        vertex_active=np.ones(vertex_ids.size, dtype=np.bool_),
        vertex_parents=np.full((vertex_ids.size, 2), -1, dtype=np.int32),
        vertex_protected=np.zeros(vertex_ids.size, dtype=np.bool_),
        cells=local_rows,
        tuples=local_rows,
        tags=np.full(count, 2, dtype=np.int32),
        blocks=np.zeros(count, dtype=np.int32),
        generations=np.zeros(count, dtype=np.int32),
        parents=np.full(count, -1, dtype=np.int32),
        children=np.full((count, 2), -1, dtype=np.int32),
        bisection_vertices=np.full(count, -1, dtype=np.int32),
        cell_ids=indices.astype(np.int64) + 10,
        cell_active=np.ones(count, dtype=np.bool_),
        cell_classes=np.zeros(count, dtype=np.int32),
        facet_classes=np.zeros((count, 3), dtype=np.int32),
        protected_edges=protected,
        next_vertex_id=6,
        next_cell_id=14,
    )


def _neighbor_closure(
    owners: tuple[int, ...], *, cell_capacity: int, protected_last: bool
) -> dict[str, Any]:
    angles = np.linspace(0.0, np.pi, 5)
    points = np.vstack(
        (np.zeros((1, 2)), np.stack((np.cos(angles), np.sin(angles)), axis=1))
    )
    rows = np.asarray([(0, index, index + 1) for index in range(1, 5)], dtype=np.int32)
    owner_array = np.asarray(owners, dtype=np.int32)
    layout = AdaptiveSimplexLayout(
        2,
        2,
        vertex_capacity=64,
        cell_capacity=cell_capacity,
        protected_edge_capacity=4,
        maximum_closure_iterations=64,
        maximum_coarsening_passes=1,
    )
    pieces = [
        _chain_state(
            np.flatnonzero(owner_array == part), layout, points, rows, protected_last
        )
        for part in range(4)
    ]
    stacked = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *pieces)
    # Center-only overlap is not a topology route; closure traverses the chain.
    neighbors = tuple(
        pair
        for index in range(3)
        for pair in (
            (owners[index], owners[index + 1]),
            (owners[index + 1], owners[index]),
        )
    )
    parts = AdaptiveSimplexParts(jax.devices("cpu"), neighbor_pairs=neighbors)
    marks = np.zeros((4, layout.cell_capacity), dtype=np.bool_)
    marks[owners[0], 0] = True
    update = refine_adaptive_simplex_parts(layout, parts, stacked, marks)
    report: dict[str, Any] = {"status": np.asarray(update.report.status).tolist()}
    if protected_last or cell_capacity == 3:
        cleared = eqx.tree_at(
            lambda state: state.clocks, update.state, update.state.clocks.at[:, 2].set(0)
        )
        refused = refine_adaptive_simplex_parts(layout, parts, update.state, marks)
        report.update(
            accepted=np.asarray(update.report.accepted).tolist(),
            operations=np.asarray(update.report.operations).tolist(),
            rollback_leaves=[
                np.asarray(value).tolist() for value in jax.tree_util.tree_leaves(cleared)
            ],
            source_leaves=[
                np.asarray(value).tolist() for value in jax.tree_util.tree_leaves(stacked)
            ],
            refused_leaves=[
                np.asarray(value).tolist()
                for value in jax.tree_util.tree_leaves(refused.state)
            ],
            failed_leaves=[
                np.asarray(value).tolist()
                for value in jax.tree_util.tree_leaves(update.state)
            ],
            refused_failed=np.asarray(refused.report.failed).tolist(),
        )
        return report
    serial_state = _chain_state(np.arange(4, dtype=np.int32), layout, points, rows, False)
    serial_marks = np.zeros(layout.cell_capacity, dtype=np.bool_)
    serial_marks[0] = True
    serial = refine_adaptive_simplex(layout, serial_state, serial_marks)
    mesh, reference = update.state.mesh, serial.state.mesh
    active = np.asarray(mesh.cell_active)
    cell_ids = np.asarray(mesh.cell_ids)[active]
    connectivity = np.asarray(mesh.vertex_ids)[
        np.arange(4)[:, None, None], np.asarray(mesh.cells)
    ][active]
    order = np.argsort(cell_ids)
    reference_active = np.asarray(reference.cell_active)
    reference_order = np.argsort(np.asarray(reference.cell_ids)[reference_active])
    held = np.asarray(mesh.vertex_active)
    ids = np.asarray(mesh.vertex_ids)[held]
    coordinates = np.asarray(mesh.coordinates)[held]
    unique, first = np.unique(ids, return_index=True)
    reference_held = np.asarray(reference.vertex_active)
    report.update(
        grown=np.asarray(update.state.cursors[:, 1] - stacked.cursors[:, 1]).tolist(),
        iterations=np.asarray(update.report.iterations).tolist(),
        serial_status=int(serial.report.status),
        cell_ids=cell_ids[order].tolist(),
        reference_cell_ids=np.asarray(reference.cell_ids)[reference_active][
            reference_order
        ].tolist(),
        connectivity=connectivity[order].tolist(),
        reference_connectivity=np.asarray(reference.vertex_ids)[
            np.asarray(reference.cells)[reference_active]
        ][reference_order].tolist(),
        coordinates=coordinates.tolist(),
        shared_coordinates=coordinates[first[np.searchsorted(unique, ids)]].tolist(),
        vertex_ids=unique.tolist(),
        reference_vertex_ids=np.asarray(reference.vertex_ids)[reference_held].tolist(),
        unique_coordinates=coordinates[first].tolist(),
        reference_coordinates=np.asarray(reference.coordinates)[reference_held].tolist(),
    )
    return report


def neighbor_chain() -> dict[str, Any]:
    return _neighbor_closure((0, 1, 2, 3), cell_capacity=64, protected_last=False)


def neighbor_owners() -> dict[str, Any]:
    return _neighbor_closure((3, 0, 2, 1), cell_capacity=64, protected_last=False)


def neighbor_protected() -> dict[str, Any]:
    return _neighbor_closure((0, 1, 2, 3), cell_capacity=64, protected_last=True)


def neighbor_capacity() -> dict[str, Any]:
    return _neighbor_closure((0, 1, 2, 3), cell_capacity=3, protected_last=False)


def _public_preparation(
    capacity: bool,
) -> tuple[
    CellMeshingResult,
    MeshPartitionPolicy,
    np.ndarray,
    PreparedAdaptiveSimplex,
    PartitionedAdaptiveSimplex,
    AdaptiveSimplexUpdate,
]:
    source = _certified(triangle_grid(8, 8))
    partition = MeshPartitionPolicy(MeshPartitionKind.MORTON, 4)
    distribution = prepare_mesh_distribution(MeshPart("domain", source), policy=partition)
    owners = np.asarray(distribution.partition.cell_owner)
    marks = np.asarray(distribution.cell_global_ids)[owners == 0]
    device_policy = (
        AdaptiveSimplexPolicy(growth_factor=1.0)
        if capacity
        else AdaptiveSimplexPolicy(vertex_capacity=1024, cell_capacity=1024)
    )
    prepared = prepare_adaptive_simplex(
        source,
        policy=_policy(
            MeshAdaptationRoute.DEVICE_BISECTION,
            device_policy=device_policy,
            distribution=distribution,
            partition_policy=partition,
        ),
    )
    partitioned = partition_adaptive_simplex(prepared)
    first = refine_adaptive_simplex_parts(
        partitioned.layout,
        partitioned.parts,
        partitioned.states,
        partitioned.cell_marks(marks),
    )
    return source, partition, marks, prepared, partitioned, first


def public_capacity() -> dict[str, Any]:
    _, _, marks, _, partitioned, first = _public_preparation(True)
    refused = refine_adaptive_simplex_parts(
        partitioned.layout, partitioned.parts, first.state, partitioned.cell_marks(marks)
    )
    category = None
    try:
        commit_partitioned_adaptive_simplex(partitioned, first.state)
    except MeshingFailure as rejection:
        category = rejection.category.value
    return {
        "status": np.asarray(first.report.status).tolist(),
        "flags": np.asarray(first.state.status_flags).tolist(),
        "cursors": np.asarray(first.state.cursors).tolist(),
        "source_cursors": np.asarray(partitioned.states.cursors).tolist(),
        "refused_failed": np.asarray(refused.report.failed).tolist(),
        "refused_operations": int(np.sum(refused.report.operations)),
        "refused_cursors": np.asarray(refused.state.cursors).tolist(),
        "category": category,
    }


def public_identity() -> dict[str, Any]:
    source, partition, marks, prepared, partitioned, first = _public_preparation(False)
    again = np.asarray(first.state.mesh.cell_active).copy()
    again[1:] = False
    second = refine_adaptive_simplex_parts(
        partitioned.layout, partitioned.parts, first.state, again
    )
    result = commit_partitioned_adaptive_simplex(partitioned, second.state)
    held = np.asarray(first.state.mesh.cell_ids[0])[again[0]]
    single = refine_adaptive_simplex(
        prepared.layout, prepared.state, prepared.cell_marks(marks)
    )
    single = refine_adaptive_simplex(
        prepared.layout,
        single.state,
        np.asarray(single.state.mesh.cell_active)
        & np.isin(np.asarray(single.state.mesh.cell_ids), held),
    )
    reference = commit_adaptive_simplex(prepared, single.state)
    if reference.lineage is None or not isinstance(
        reference.hierarchy, BisectionHierarchy
    ):
        raise AssertionError(
            "device_scenario.serial_reference must retain transition, lineage, transfer, and hierarchy; got missing adaptation identity"
        )
    if partitioned.source_witness is None:
        raise AssertionError(
            "device_scenario.partitioned_source must retain source support; got absent source support"
        )
    host = _adapt(MeshAdaptationRoute.NATIVE_BISECTION, source, np.sort(marks))
    host = _adapt(
        MeshAdaptationRoute.NATIVE_BISECTION,
        host.target,
        np.sort(held),
        hierarchy=host.hierarchy,
    )
    host_mesh = host.target.mesh
    cell_ids, cell_vertices, vertex_ids, coordinates = [], [], [], []
    local_records = []
    for local in result.local_results:
        target = local.target.mesh
        storage = target.storage
        if storage is None:
            raise AssertionError(
                "device_scenario.partitioned_target.storage must retain resident data; got absent storage"
            )
        evidence = local.target.collective_evidence
        if not isinstance(evidence, CollectiveMeshEvidence):
            raise AssertionError(
                "device_scenario.partitioned_target.collective_evidence must bind the accepted epoch; got absent or incompatible evidence"
            )
        if (
            evidence.source_embedding is None
            or evidence.migration_record is None
            or evidence.rebalance_proposal is None
        ):
            raise AssertionError(
                "device_scenario.partitioned_target.collective_evidence must retain embedding and migration records; got absent or inconsistent evidence"
            )
        owned_cells = np.asarray(storage.entity_owned[2])
        owned_vertices = np.asarray(storage.entity_owned[0])
        cell_ids.append(_cell_ids(target)[owned_cells])
        cell_vertices.append(
            np.asarray(target.vertex_global_ids)[_cells(target)][owned_cells]
        )
        vertex_ids.append(np.asarray(target.vertex_global_ids)[owned_vertices])
        coordinates.append(np.asarray(target.coordinates)[owned_vertices])
        if local.distribution is None:
            raise AssertionError(
                "device_scenario.partitioned_target.distribution must bind target ownership; got absent distribution"
            )
        local_records.append(
            {
                "storage_present": True,
                "distribution_present": True,
                "topology_id": target.topology_id,
                "embedding_status": evidence.source_embedding.status,
                "source_audit_id": evidence.source_audit_id,
                "source_premise_id": evidence.source_premise_id,
                "source_result_id": local.source.result_id,
                "rebalanced": local.distribution.rebalanced,
                "imbalance": float(local.distribution.target.evidence.imbalance),
                "migration_accepted": np.asarray(
                    evidence.migration_record.transport_evidence.accepted
                ).tolist(),
                "sent_cells": int(
                    np.sum(
                        np.asarray(
                            evidence.migration_record.transport_evidence.sent_cells
                        )
                    )
                ),
                "part_weights": np.asarray(
                    local.distribution.target.evidence.part_weights
                ).tolist(),
                "proposed_weights": np.asarray(
                    evidence.rebalance_proposal.part_weights
                ).tolist(),
            }
        )
    joined_cells, joined_vertices = np.concatenate(cell_ids), np.concatenate(vertex_ids)
    cell_order, vertex_order = np.argsort(joined_cells), np.argsort(joined_vertices)
    host_cell_ids = _cell_ids(host_mesh)
    host_cell_order = np.argsort(host_cell_ids)
    return {
        "first_status": np.asarray(first.report.status).tolist(),
        "second_status": np.asarray(second.report.status).tolist(),
        "grown": np.asarray(
            second.state.cursors[:, 1] - first.state.cursors[:, 1]
        ).tolist(),
        "topology_id": result.identity.target_topology_id,
        "reference_topology_id": reference.target.mesh.topology_id,
        "lineage_id": result.identity.lineage_id,
        "reference_lineage_id": reference.lineage.lineage_id,
        "hierarchy_id": result.identity.hierarchy_id,
        "reference_hierarchy_id": reference.hierarchy.hierarchy_id,
        "source_certification_absent": source.certification is None,
        "source_result_id": source.result_id,
        "source_audit_id": source.audit.report_id,
        "source_premise_id": partitioned.source_witness.scientific_certification_id,
        "maximum_imbalance": partition.maximum_imbalance,
        "locals": local_records,
        "cell_ids": joined_cells[cell_order].tolist(),
        "host_cell_ids": host_cell_ids[host_cell_order].tolist(),
        "connectivity": np.concatenate(cell_vertices)[cell_order].tolist(),
        "host_connectivity": np.asarray(host_mesh.vertex_global_ids)[_cells(host_mesh)][
            host_cell_order
        ].tolist(),
        "vertex_ids": joined_vertices[vertex_order].tolist(),
        "host_vertex_ids": np.asarray(host_mesh.vertex_global_ids).tolist(),
        "coordinates": np.concatenate(coordinates)[vertex_order].tolist(),
        "host_coordinates": np.asarray(host_mesh.coordinates).tolist(),
    }
