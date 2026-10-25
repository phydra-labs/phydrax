# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Unequal owner-local FV closures on collectively agreed execution buckets.

A jittered 4x4 square is split at x = 0.3, so the two depth-two owner
closures differ in cells, faces, owned facets and halo routes. Every scenario
compares owned rows with an independent serial advance of the logical mesh.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import PartitionSpec

import phydrax as phx
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import ExecutionGroup
from phydrax.discretization._cell_complex import PolygonalConnectivity
from phydrax.discretization._cell_mesh import CellMesh, CellMeshStorage
from phydrax.discretization._distributed_field import (
    DistributedHaloPlan,
    prepare_owner_local_halo_plans,
)
from phydrax.discretization._distributed_preparation import replicated_host_array
from phydrax.discretization._reference_cell_types import CellBlock
from phydrax.discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumeDiscretization,
)
from phydrax.discretization.finite_volume._unstructured_dynamics import (
    execute_owner_local_finite_volume_advance,
    OwnerLocalFiniteVolumeExecution,
    prepare_owner_local_finite_volume_execution,
    PreparedUnstructuredFiniteVolumeDynamics,
)


_EVIDENCE = "e" * 64
_SOURCE_ID = "owner-local-kinetic-heating"
_STEP = 0.01
_STEPS = 3
_BOUNDS = {
    "maximum_cells": 32,
    "maximum_faces": 64,
    "maximum_message_entries": 64,
    "maximum_storage_bytes": 1 << 30,
}
# Owned rows match the serial advance up to reordered face-sum rounding.
_RTOL = 1e-12
_ATOL = 1e-12


def _kinetic_heating(time: Array, state: Array, centers: Array, args: Any) -> Array:
    """Energy source 0.05 |m|^2 / rho: 0/0 on any zero-density lane."""
    momentum = state[:, 1:3]
    heating = 0.05 * jnp.sum(momentum * momentum, axis=1) / state[:, 0]
    return jnp.zeros_like(state).at[:, 3].set(heating)


class _Problem(NamedTuple):
    system: Any
    source: CellMesh
    serial: PreparedUnstructuredFiniteVolumeDynamics
    owners: np.ndarray
    rows: dict[int, int]
    state: Array
    volumes: np.ndarray


def _dynamics(
    system: Any, geometry: Any, flux: Any = None
) -> PreparedUnstructuredFiniteVolumeDynamics:
    return PreparedUnstructuredFiniteVolumeDynamics(
        system,
        geometry,
        phx.discretization.UnstructuredFiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.HLLFluxPlan() if flux is None else flux,
        ),
        phx.discretization.UnstructuredFiniteVolumeBoundarySet(
            geometry.boundary_patch_names,
            {
                name: phx.discretization.SlipWallBoundary()
                for name in geometry.boundary_patch_names
            },
        ),
        source=_kinetic_heating,
        source_id=_SOURCE_ID,
    )


def _problem() -> _Problem:
    system = phx.equations.EulerSystem(2)
    # Deterministic interior jitter keeps every triangle positively oriented.
    axis = np.linspace(0.0, 1.0, 5)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape((-1, 2))
    interior = np.all((points > 0.0) & (points < 1.0), axis=1)
    shift = 0.05 * np.stack((np.sin(7.0 * points[:, 1]), np.cos(5.0 * points[:, 0])), 1)
    points = points + np.where(interior[:, None], shift, 0.0)
    triangles = np.asarray(
        [
            cell
            for i, j in np.ndindex(4, 4)
            for cell in (
                (5 * i + j, 5 * i + j + 5, 5 * i + j + 6),
                (5 * i + j, 5 * i + j + 6, 5 * i + j + 1),
            )
        ],
        dtype=np.int32,
    )
    corners = points[triangles]
    edges = corners[:, 1:] - corners[:, :1]
    negative = edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0] < 0
    triangles[negative] = triangles[negative][:, (0, 2, 1)]
    source = CellMesh.from_triangles(
        points,
        triangles,
        vertex_global_ids=500 + 7 * np.arange(25, dtype=np.int64),
        cell_global_ids=900 + 11 * np.arange(32, dtype=np.int64),
    )
    dense = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
        source, component_names=system.component_names
    ).prepare()
    centers = np.asarray(dense.cell_centers)
    x, y = jnp.asarray(centers[:, 0]), jnp.asarray(centers[:, 1])
    primitive = jnp.stack(
        (1 + 0.3 * x + 0.1 * y, 0.2 - 0.1 * y, 0.05 + 0.1 * x, 1 + 0.2 * y), axis=1
    )
    return _Problem(
        system,
        source,
        _dynamics(system, dense),
        (centers[:, 0] > 0.3).astype(np.int32),
        {
            int(identifier): row
            for row, identifier in enumerate(np.asarray(dense.cell_global_ids))
        },
        system.primitive_to_conserved(primitive),
        np.asarray(dense.cell_volumes),
    )


def _closure(problem: _Problem, part: int, parts: int) -> CellMesh:
    """Depth-two owner closure: owned cells first, then ghosts by global row."""
    source, owners = problem.source, problem.owners
    block = source.blocks[0]
    triangles = np.asarray(block.vertices)
    cell_ids = np.asarray(block.global_ids)
    cell_owner = owners[[problem.rows[int(value)] for value in cell_ids]]
    vertex_ids = np.asarray(source.vertex_global_ids)
    edges: dict[frozenset[int], list[int]] = {}
    for cell, rows in enumerate(triangles):
        for corner in range(3):
            key = frozenset((int(rows[corner]), int(rows[(corner + 1) % 3])))
            edges.setdefault(key, []).append(cell)
    adjacency: list[set[int]] = [set() for _ in range(cell_ids.size)]
    for cells in edges.values():
        if len(cells) == 2:
            adjacency[cells[0]].add(cells[1])
            adjacency[cells[1]].add(cells[0])
    owned = [int(cell) for cell in np.flatnonzero(cell_owner == part)]
    first = set(owned).union(*(adjacency[cell] for cell in owned))
    second = first.union(*(adjacency[cell] for cell in first))
    closure = np.asarray(owned + sorted(second - set(owned)), dtype=np.int64)
    local_vertices, local_rows = np.unique(triangles[closure], return_inverse=True)
    coordinates = np.asarray(source.coordinates)[local_vertices]
    probe = CellMesh(
        coordinates,
        (
            CellBlock(
                "triangles",
                "triangle",
                local_rows.reshape((-1, 3)).astype(np.int32),
                global_ids=cell_ids[closure],
            ),
        ),
        vertex_global_ids=vertex_ids[local_vertices],
    )
    # Logical owners: an entity belongs to the lowest owner among its cells.
    vertex_owner = np.full(vertex_ids.size, parts, dtype=np.int32)
    for cell, rows in enumerate(triangles):
        vertex_owner[rows] = np.minimum(vertex_owner[rows], cell_owner[cell])
    edge_owner = {
        key: min(int(cell_owner[cell]) for cell in cells) for key, cells in edges.items()
    }
    edge_boundary = {key: len(cells) == 1 for key, cells in edges.items()}
    vertex_lookup = {int(value): row for row, value in enumerate(vertex_ids)}
    connectivity = probe.connectivity
    assert isinstance(connectivity, PolygonalConnectivity)
    local_edges = np.asarray(connectivity.edges)
    local_ids = vertex_ids[local_vertices]
    edge_keys = [
        frozenset(vertex_lookup[int(local_ids[corner])] for corner in edge)
        for edge in local_edges
    ]
    storage = CellMeshStorage(
        tuple(entity.count for entity in source.topology.entity_sets),
        tuple(np.asarray(entity.entity_ids) for entity in probe.topology.entity_sets),
        (
            vertex_owner[local_vertices],
            np.asarray([edge_owner[key] for key in edge_keys], dtype=np.int32),
            cell_owner[closure].astype(np.int32),
        ),
        partition_index=part,
        partition_count=parts,
        logical_topology_id=source.topology_id,
        logical_geometry_id=source.geometry_id,
        evidence_id=_EVIDENCE,
        logical_arrays=(("coordinates", jnp.asarray(source.coordinates)),),
        local_coordinates=coordinates,
        local_blocks=probe.blocks,
        local_physical_boundary_facets=np.asarray(
            [edge_boundary[key] for key in edge_keys], dtype=np.bool_
        ),
        local_neighborhood_complete=np.asarray(
            [adjacency[int(cell)] <= second for cell in closure], dtype=np.bool_
        ),
    )
    return CellMesh(coordinates, probe.blocks, storage=storage)


def _program(
    problem: _Problem, mesh: CellMesh, buffered: bool, flux: Any = None
) -> PreparedUnstructuredFiniteVolumeDynamics:
    storage = mesh.storage
    assert storage is not None
    plan = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
        mesh,
        component_names=problem.system.component_names,
        neighborhood_complete=storage.local_neighborhood_complete,
        physical_boundary_faces=storage.local_physical_boundary_facets,
        closure_evidence_id=storage.evidence_id,
    )
    # Per-part slot buckets differ, so owned-facet slots are padded too.
    owned = int(np.count_nonzero(np.asarray(storage.entity_owned[1])))
    geometry = plan.prepare(owned_face_capacity=owned + 3 if buffered else None)
    return _dynamics(problem.system, geometry, flux)


def _group(devices: tuple[jax.Device, ...]) -> ExecutionGroup:
    return ExecutionGroup(
        ExecutionGroupSpec(
            "owner-local-fv-unequal-closures",
            tuple(sorted({device.process_index for device in devices})),
            tuple((device.process_index, device.id) for device in devices),
            mesh_axes=(("parts", len(devices)),),
        ),
        devices,
    )


def _prepare(
    programs: tuple[tuple[Any, DistributedHaloPlan], ...],
    group: ExecutionGroup,
    **bounds: int,
) -> OwnerLocalFiniteVolumeExecution:
    return prepare_owner_local_finite_volume_execution(
        programs, group, axis_name="parts", **{**_BOUNDS, **bounds}
    )


def _refused(call: Callable[[], object], expected: str) -> str:
    """Return ``expected`` when ``call`` refuses with it, else the observed outcome."""
    try:
        call()
    except ValueError as error:
        return expected if expected in str(error) else str(error)
    return "admitted"


def _with_evidence(halo: DistributedHaloPlan, evidence: str) -> DistributedHaloPlan:
    return DistributedHaloPlan.from_owner_local(
        halo.part_count,
        local_global_ids=halo.local_global_ids,
        local_owned=halo.local_owned,
        local_valid=halo.local_valid,
        global_entity_count=halo.entity_count,
        partition_index=int(np.asarray(halo.partition_index)),
        phase_send_indices=halo.phase_send_indices,
        phase_receive_indices=halo.phase_receive_indices,
        phase_send_valid=halo.phase_send_valid,
        phase_receive_valid=halo.phase_receive_valid,
        permutations=halo.permutations,
        evidence_id=evidence,
    )


def _padding_is_zero(value: Array, execution: OwnerLocalFiniteVolumeExecution) -> bool:
    counts = dict(zip(execution.local_parts, execution.local_cell_counts, strict=True))
    placed = jax.device_put(
        value,
        execution.execution_group.named_sharding(PartitionSpec("parts", None, None)),
    )
    return all(
        not np.any(np.asarray(shard.data)[0, counts[shard.index[0].start or 0] :])
        for shard in placed.addressable_shards
    )


def _advance_and_compare(
    problem: _Problem,
    programs: tuple[PreparedUnstructuredFiniteVolumeDynamics, ...],
    execution: OwnerLocalFiniteVolumeExecution,
    *,
    zero_density: bool,
) -> dict[str, Any]:
    """Advance, differentiate and refuse a zero density against the serial mesh."""
    geometries: list[UnstructuredFiniteVolumeDiscretization] = []
    for program in programs:
        geometry = program.discretization
        assert isinstance(geometry, UnstructuredFiniteVolumeDiscretization)
        geometries.append(geometry)
    rows = tuple(
        np.asarray([problem.rows[int(value)] for value in np.asarray(g.cell_global_ids)])
        for g in geometries
    )
    owned = tuple(np.asarray(geometry.cell_owned) for geometry in geometries)
    state = problem.state
    # Stale ghost seeds prove every advance refreshes halos before its solve.
    seeds = tuple(
        jnp.where(mask[:, None], state[row], 2 * state[row])
        for mask, row in zip(owned, rows, strict=True)
    )
    time, step = jnp.asarray(0.0), jnp.asarray(_STEP)
    reference = state
    for _ in range(_STEPS):
        reference = reference + _STEP * problem.serial(time, reference)
    current = execution.pack_state(seeds)
    # Iterations never synchronize with the host after preparation.
    with jax.transfer_guard_device_to_host("disallow"):
        for _ in range(_STEPS):
            advance = execute_owner_local_finite_volume_advance(
                execution, current, time, step
            )
            current = advance.state
    logical = execution.unpack_state(current)
    for value, seed, mask, row in zip(logical, seeds, owned, rows, strict=True):
        np.testing.assert_allclose(
            value[mask], reference[row[mask]], rtol=_RTOL, atol=_ATOL
        )
        np.testing.assert_array_equal(value[~mask], seed[~mask])
    assert _padding_is_zero(current, execution)
    inventory = np.sum(problem.volumes[:, None] * np.asarray(reference), axis=0)
    after = replicated_host_array(advance.inventory_after)
    np.testing.assert_allclose(after, inventory, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(
        after[0],
        np.sum(problem.volumes * np.asarray(state[:, 0])),
        rtol=_RTOL,
        atol=_ATOL,
    )
    assert bool(replicated_host_array(advance.finite))
    assert bool(replicated_host_array(advance.admissible))

    weight = jnp.asarray((1.0, 0.5, 0.25, 0.125))
    weights = execution.pack_state(
        tuple(jnp.where(mask[:, None], weight, 0.0) for mask in owned)
    )
    packed = execution.pack_state(seeds)

    def loss(value: Array) -> Array:
        advanced = execute_owner_local_finite_volume_advance(
            execution, value, time, step
        ).state
        return jnp.sum(weights * advanced)

    gradient = jax.grad(loss)(packed)
    expected = jax.grad(
        lambda value: jnp.sum(weight * (value + _STEP * problem.serial(time, value)))
    )(state)
    assert _padding_is_zero(gradient, execution)
    for value, mask, row in zip(
        execution.unpack_state(gradient), owned, rows, strict=True
    ):
        assert np.all(np.isfinite(np.asarray(value)))
        np.testing.assert_allclose(
            value[mask], expected[row[mask]], rtol=_RTOL, atol=_ATOL
        )
        np.testing.assert_array_equal(value[~mask], 0.0)

    broken = list(seeds)
    if zero_density:
        broken[-1] = broken[-1].at[0, 0].set(0.0)
    refused = execute_owner_local_finite_volume_advance(
        execution, execution.pack_state(tuple(broken)), time, step
    )
    return {
        "cell_capacity": execution.cell_capacity,
        "logical_cell_count": execution.logical_cell_count,
        "logical_owned_face_count": execution.logical_owned_face_count,
        "global_face_count": problem.source.topology.entity_sets[1].count,
        "inventory_after": after.tolist(),
        "zero_density": {
            "finite": bool(replicated_host_array(refused.finite)),
            "admissible": bool(replicated_host_array(refused.admissible)),
        },
    }


def two_device_unequal_closures(mode: str) -> dict[str, Any]:
    """Two unequal closures on one process's two devices."""
    problem = _problem()
    devices = tuple(jax.devices()[:2])
    group = _group(devices)
    meshes = tuple(_closure(problem, part, 2) for part in range(2))
    buffered = mode == "buffered"
    programs = tuple(_program(problem, mesh, buffered) for mesh in meshes)
    planned = prepare_owner_local_halo_plans(
        meshes,
        2,
        devices=devices,
        axis_name="parts",
        message_capacity=_BOUNDS["maximum_message_entries"],
    )
    # Part zero declares a wider, still exact, message bucket.
    halos = (
        planned[0].execution_bucket(
            planned[0].local_capacity, planned[0].message_capacity + 3
        ),
        planned[1],
    )
    pairs = tuple(zip(programs, halos, strict=True))
    execution = _prepare(pairs, group)
    record = _advance_and_compare(problem, programs, execution, zero_density=True)
    record["cell_counts"] = list(execution.local_cell_counts)
    record["refusals"] = {
        "overflow": _refused(
            lambda: _prepare(pairs, group, maximum_cells=execution.cell_capacity - 1),
            "owner_local.capacities",
        ),
        "storage": _refused(
            lambda: _prepare(pairs, group, maximum_storage_bytes=1),
            "owner_local.storage_bytes",
        ),
        "halo": _refused(
            lambda: _prepare(((programs[0], halos[1]), (programs[1], halos[0])), group),
            "halo and closure stable IDs",
        ),
        "missing": _refused(
            lambda: _prepare(pairs[:1], group), "a local program admission failure"
        ),
        "family": _refused(
            lambda: _prepare(
                (
                    pairs[0],
                    (
                        _program(
                            problem,
                            meshes[1],
                            buffered,
                            phx.discretization.RusanovFluxPlan(),
                        ),
                        halos[1],
                    ),
                ),
                group,
            ),
            "one execution family",
        ),
    }
    return record


def two_process_unequal_closures() -> dict[str, Any]:
    """One unequal closure per Gloo process; rank one alone supplies refusals."""
    rank = jax.process_index()
    problem = _problem()
    devices = tuple(jax.devices())
    group = _group(devices)
    mesh = _closure(problem, rank, 2)
    program = _program(problem, mesh, True)
    (halo,) = prepare_owner_local_halo_plans(
        (mesh,),
        2,
        devices=devices,
        axis_name="parts",
        message_capacity=_BOUNDS["maximum_message_entries"],
    )
    execution = _prepare(((program, halo),), group)
    record = _advance_and_compare(problem, (program,), execution, zero_density=rank == 1)
    record["cell_count"] = execution.local_cell_counts[0]
    one_sided = rank == 1
    record["refusals"] = {
        "halo": _refused(
            lambda: _prepare(
                ((program, _with_evidence(halo, "0" * 64) if one_sided else halo),),
                group,
            ),
            "halo and closure stable IDs",
        ),
        "missing": _refused(
            lambda: _prepare(() if one_sided else ((program, halo),), group),
            "a local program admission failure",
        ),
        "family": _refused(
            lambda: _prepare(
                (
                    (
                        _program(
                            problem, mesh, True, phx.discretization.RusanovFluxPlan()
                        )
                        if one_sided
                        else program,
                        halo,
                    ),
                ),
                group,
            ),
            "one execution family",
        ),
    }
    return record
