from fractions import Fraction
from itertools import permutations
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax._geometry_predicates import PredicateMode
from phydrax._meshcore import (
    PlcBoundaryPolicy,
    TetMesh3D,
    TetMeshEdit,
    TetMeshEditKind,
    TetMeshEditStatus,
)
from phydrax.discretization._cell_complex import PolygonalConnectivity
from phydrax.discretization._cell_geometry import (
    PerCellRestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._coordinate_enclosure import ExpressionInput
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_exact import apply_periodic
from phydrax.discretization._periodic_topology import (
    _entity_vertex_rows,
    PeriodicMeshTopology,
)
from phydrax.meshing._assembly import MeshPart
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing._distribution import (
    MeshPartitionKind,
    MeshPartitionPolicy,
    prepare_distribution_transition,
    prepare_mesh_distribution,
)
from phydrax.meshing._lineage import EntityLineageKind, VertexInterpolationStencil
from phydrax.meshing._local_metric import execute_local_metric_adaptation
from phydrax.meshing._periodic import (
    _cell_edge_rows,
    _entity_layout,
    _periodic_cell_layout,
    _periodic_edge_operation_signatures,
    _periodic_quotient_adjacency,
    periodic_delaunay_mesh,
    periodic_entity_orbit_map,
    periodic_orbit_closure,
    periodic_topology_after_edit,
    PeriodicMeshConstruction,
    PeriodicPointOrbits,
    prepare_periodic_refinement_edge_lock,
    stage_periodic_topology_edit,
    transfer_periodic_p1_vertices,
)
from phydrax.meshing._topology_edit import (
    CellTopologyEdit,
    entity_keys,
    EntityRelations,
    TopologyEditBlock,
)
from phydrax.meshing.providers._native_periodic_tet import (
    tet_mesh_periodic_edit,
    tet_mesh_periodic_operation_members,
    tet_mesh_periodic_state,
)


def _skew_periodic_mesh() -> CellMesh:
    vectors = np.asarray(((1.0, 0.2), (0.3, 1.1)), dtype=np.float64)
    periodic_cell = PeriodicCell(vectors)
    shape = 5
    vertex_rows = np.arange(shape * shape, dtype=np.int64).reshape(shape, shape)
    lattice_coordinates = np.stack(
        np.meshgrid(np.arange(shape), np.arange(shape), indexing="ij"), axis=-1
    ).reshape((-1, 2))
    representatives = vertex_rows[
        np.arange(shape)[:, None] % 2, np.arange(shape)[None, :] % 2
    ].reshape(-1)
    shifts = np.stack(
        np.meshgrid(np.arange(shape) // 2, np.arange(shape) // 2, indexing="ij"),
        axis=-1,
    ).reshape((-1, 2))
    representative_coordinates = (
        (lattice_coordinates % 2).astype(np.float64) / 2.0
    ) @ vectors
    coordinates = apply_periodic(
        periodic_cell,
        representative_coordinates[representatives],
        shifts,
    )
    triangles = []
    for i in range(shape - 1):
        for j in range(shape - 1):
            a = int(vertex_rows[i, j])
            b = int(vertex_rows[i + 1, j])
            c = int(vertex_rows[i + 1, j + 1])
            d = int(vertex_rows[i, j + 1])
            triangles.extend(((a, b, c), (a, c, d)))
    block = CellBlock(
        "triangles",
        "triangle",
        np.asarray(triangles, dtype=np.int32),
        global_ids=np.arange(500, 500 + len(triangles), dtype=np.int64),
    )
    vertex_ids = np.arange(shape * shape, dtype=np.int64)
    lifted = CellMesh(coordinates, (block,), vertex_global_ids=vertex_ids)
    topology = PeriodicMeshTopology(lifted, periodic_cell, representatives, shifts)
    return CellMesh(
        coordinates,
        (block,),
        vertex_global_ids=vertex_ids,
        periodic_topology=topology,
    )


def test_periodic_vertex_orbit_rejects_a_nonzero_one_ulp_mismatch() -> None:
    from phydrax.discretization._periodic_topology import (
        validate_periodic_vertex_orbits,
    )

    mesh = _skew_periodic_mesh()
    topology = mesh.periodic_topology
    if not isinstance(topology, PeriodicMeshTopology):
        raise TypeError("The skew periodic fixture lost its quotient topology.")
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64).copy()
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    shifted_rows = np.flatnonzero(np.any(shifts != 0, axis=1))
    if not shifted_rows.size:
        raise ValueError("The periodic fixture needs a nonzero lifted winding.")
    row = int(shifted_rows[0])
    axis = int(np.flatnonzero(shifts[row] != 0)[0])
    original = float(coordinates[row, axis])
    coordinates[row, axis] = np.nextafter(original, np.inf)
    assert coordinates[row, axis] != original
    with pytest.raises(ValueError, match="vertex_shifts"):
        validate_periodic_vertex_orbits(
            coordinates,
            topology.cell,
            topology.vertex_representatives,
            topology.vertex_shifts,
        )


def _periodic_tetra_mesh(shape: int = 3, /) -> CellMesh:
    vertex_rows = np.arange(shape**3, dtype=np.int64).reshape((shape,) * 3)
    lattice = np.stack(
        np.meshgrid(*(np.arange(shape) for _ in range(3)), indexing="ij"), axis=-1
    ).reshape((-1, 3))
    vectors = np.asarray(
        ((1.0, 0.125, 0.0), (0.0, 1.0, 0.25), (0.125, 0.0, 1.0)),
        dtype=np.float64,
    )
    periodic_cell = PeriodicCell(vectors)
    indices = np.arange(shape, dtype=np.int64)
    representatives = vertex_rows[
        indices[:, None, None] % 2,
        indices[None, :, None] % 2,
        indices[None, None, :] % 2,
    ].reshape(-1)
    shifts = np.stack(
        np.meshgrid(*(indices // 2 for _ in range(3)), indexing="ij"), axis=-1
    ).reshape((-1, 3))
    representative_coordinates = ((lattice % 2).astype(np.float64) / 2.0) @ vectors
    coordinates = apply_periodic(
        periodic_cell,
        representative_coordinates[representatives],
        shifts,
    )
    tetrahedra = []
    for cube in np.ndindex((shape - 1,) * 3):
        for order in permutations(range(3)):
            point = np.asarray(cube, dtype=np.int64)
            path = [point.copy()]
            for axis in order:
                point = point.copy()
                point[axis] += 1
                path.append(point)
            tetrahedron = np.asarray(
                [vertex_rows[tuple(row)] for row in path], dtype=np.int32
            )
            if (
                np.linalg.det(coordinates[tetrahedron[1:]] - coordinates[tetrahedron[0]])
                < 0.0
            ):
                tetrahedron[[1, 2]] = tetrahedron[[2, 1]]
            tetrahedra.append(tetrahedron)
    block = CellBlock(
        "tetrahedra",
        "tetrahedron",
        np.asarray(tetrahedra, dtype=np.int32),
        global_ids=np.arange(3000, 3000 + len(tetrahedra), dtype=np.int64),
    )
    vertex_ids = np.arange(shape**3, dtype=np.int64)
    lifted = CellMesh(coordinates, (block,), vertex_global_ids=vertex_ids)
    topology = PeriodicMeshTopology(
        lifted,
        periodic_cell,
        representatives,
        shifts,
    )
    return CellMesh(
        coordinates,
        (block,),
        vertex_global_ids=vertex_ids,
        periodic_topology=topology,
    )


def _periodic_grid_mesh(subdivisions: int, /, *, perturbation: float = 0.0) -> CellMesh:
    shape = 2 * subdivisions + 1
    vectors = np.asarray(((1.0, 0.2), (0.3, 1.1)), dtype=np.float64)
    periodic_cell = PeriodicCell(vectors)
    vertex_rows = np.arange(shape * shape, dtype=np.int64).reshape(shape, shape)
    lattice = np.stack(
        np.meshgrid(np.arange(shape), np.arange(shape), indexing="ij"), axis=-1
    ).reshape((-1, 2))
    root = lattice % subdivisions
    representatives = vertex_rows[root[:, 0], root[:, 1]]
    shifts = lattice // subdivisions
    representative_coordinates = (root.astype(np.float64) / subdivisions) @ vectors
    if perturbation:
        phase = root.astype(np.float64) / subdivisions
        active = np.all(root > 0, axis=1)
        offsets = np.stack(
            (
                np.sin(5.0 * phase[:, 0] + 2.0 * phase[:, 1]),
                np.cos(3.0 * phase[:, 0] - 4.0 * phase[:, 1]),
            ),
            axis=1,
        )
        representative_coordinates += (
            (perturbation / subdivisions) * active[:, None] * (offsets @ vectors)
        )
    coordinates = apply_periodic(
        periodic_cell,
        representative_coordinates[representatives],
        shifts,
    )
    triangles = []
    for i in range(shape - 1):
        for j in range(shape - 1):
            a = int(vertex_rows[i, j])
            b = int(vertex_rows[i + 1, j])
            c = int(vertex_rows[i + 1, j + 1])
            d = int(vertex_rows[i, j + 1])
            triangles.extend(((a, b, c), (a, c, d)))
    block = CellBlock(
        "triangles",
        "triangle",
        np.asarray(triangles, dtype=np.int32),
        global_ids=np.arange(8000, 8000 + len(triangles), dtype=np.int64),
    )
    vertex_ids = np.arange(shape * shape, dtype=np.int64)
    lifted = CellMesh(coordinates, (block,), vertex_global_ids=vertex_ids)
    topology = PeriodicMeshTopology(
        lifted,
        periodic_cell,
        representatives,
        shifts,
    )
    return CellMesh(
        coordinates,
        (block,),
        vertex_global_ids=vertex_ids,
        periodic_topology=topology,
    )


def _periodic_boundary_controls(mesh: CellMesh) -> tuple[np.ndarray, np.ndarray]:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError("Periodic boundary controls require polygonal connectivity.")
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    boundary_edges = periodic_orbit_closure(
        mesh, 1, edge_ids[np.asarray(connectivity.boundary_edges, dtype=np.bool_)]
    ).require_source(mesh, 1)
    boundary_vertices = periodic_orbit_closure(
        mesh, 0, vertex_ids[np.asarray(connectivity.boundary_vertices, dtype=np.bool_)]
    ).require_source(mesh, 0)
    return np.isin(edge_ids, boundary_edges).astype(np.int64), np.isin(
        vertex_ids, boundary_vertices
    )


def _periodic_metric_adaptation(
    mesh: CellMesh,
    metric: np.ndarray,
    /,
    *,
    topology_operations: bool,
    relocation: bool,
) -> Any:
    edge_classes, fixed_vertices = _periodic_boundary_controls(mesh)
    edge_count = mesh.entity_set(1).entity_ids.size
    return execute_local_metric_adaptation(
        mesh,
        metric,
        cell_classes=np.zeros((mesh.entity_set(2).entity_ids.size,), dtype=np.int64),
        edge_classes=edge_classes,
        protected_edges=np.zeros((edge_count,), dtype=np.bool_),
        fixed_vertices=fixed_vertices,
        predicate_mode=PredicateMode.EXACT,
        topology_operations=topology_operations,
        relocation=relocation,
    )


def _require_periodic_outcome(mesh: CellMesh, outcome: Any, /) -> CellMesh:
    target = stage_periodic_topology_edit(mesh, outcome.edit)
    assert target is not None
    topology = target.periodic_topology
    assert topology is not None
    topology.require_lifted(target.topology, target.lifted_topology_id, target.blocks)
    topology.require_coordinates(target.coordinates)
    return target


def _reordered_periodic_mesh() -> CellMesh:
    base = _skew_periodic_mesh()
    source_topology = base.periodic_topology
    if source_topology is None:
        raise ValueError("The periodic fixture requires its quotient descriptor.")
    base_representatives = np.asarray(
        source_topology.vertex_representatives, dtype=np.int64
    )
    base_shifts = np.asarray(source_topology.vertex_shifts, dtype=np.int64)
    vertex_ids = np.asarray(base.vertex_global_ids, dtype=np.int64)
    representatives = np.empty_like(base_representatives)
    shifts = np.empty_like(base_shifts)
    for root in np.unique(base_representatives):
        members = np.flatnonzero(base_representatives == root)
        selected = int(members[np.argmax(vertex_ids[members])])
        representatives[members] = selected
        shifts[members] = base_shifts[members] - base_shifts[selected]
    # Re-rooting moves the canonical lift owner: every image is the exact
    # image of its new root, which can differ from the old lift by roundoff.
    vertex_coordinates = apply_periodic(
        source_topology.cell,
        np.asarray(base.coordinates, dtype=np.float64)[representatives],
        shifts,
    )

    source_block = base.blocks[0]
    cells = np.asarray(source_block.vertices, dtype=np.int32)
    cell_ids = np.asarray(source_block.global_ids, dtype=np.int64)
    blocks = (
        CellBlock(
            "zeta",
            "triangle",
            cells[::2][::-1],
            global_ids=cell_ids[::2][::-1],
        ),
        CellBlock(
            "alpha",
            "triangle",
            cells[1::2][::-1],
            global_ids=cell_ids[1::2][::-1],
        ),
    )
    lifted = CellMesh(
        vertex_coordinates, blocks, vertex_global_ids=base.vertex_global_ids
    )
    base_geometry = phx.discretization.CellGeometrySpec.affine(lifted)
    elements, routes, coordinates = base_geometry.resolve(lifted)
    geometry_elements = {}
    geometry_routes = {}
    identity_chart: tuple[ExpressionInput, ...] = (
        {(1, 0): Fraction(1)},
        {(0, 1): Fraction(1)},
    )
    for block, source_element, route in zip(lifted.blocks, elements, routes, strict=True):
        block_ids = np.asarray(block.global_ids, dtype=np.int64)
        if not isinstance(source_element, ScalarCoordinateElement):
            raise TypeError(
                "Periodic geometry fixture requires a scalar coordinate element."
            )
        geometry_elements[block.name] = PerCellRestrictedCellGeometryElement(
            source_element,
            block.cell_kind,
            block_ids,
            block_ids,
            tuple(identity_chart for _ in block_ids),
        )
        geometry_routes[block.name] = route
    actual_geometry = phx.discretization.CellGeometrySpec(
        geometry_elements, geometry_routes, coordinates
    )
    initial = PeriodicMeshTopology(lifted, source_topology.cell, representatives, shifts)
    quotient_ids = {
        degree: np.arange(
            100_000 * (degree + 1),
            100_000 * (degree + 1) + entity.count,
            dtype=np.int64,
        )
        for degree, entity in enumerate(initial.quotient.entity_sets)
    }
    topology = PeriodicMeshTopology(
        lifted,
        source_topology.cell,
        representatives,
        shifts,
        entity_global_ids=quotient_ids,
        entity_allocator_next_ids={
            degree: int(ids[-1]) + 1 for degree, ids in quotient_ids.items()
        },
        actual_geometry=actual_geometry,
    )
    return CellMesh(
        vertex_coordinates,
        blocks,
        vertex_global_ids=base.vertex_global_ids,
        periodic_topology=topology,
    )


def test_periodic_canonicalization_rebinds_reordered_quotient_mesh() -> None:
    source = _reordered_periodic_mesh()
    source_topology = source.periodic_topology
    assert source_topology is not None

    canonical = phx.meshing.canonicalize_cell_mesh(source)
    target_topology = canonical.periodic_topology
    assert target_topology is not None
    assert canonical is not source
    assert tuple(block.name for block in canonical.blocks) == ("alpha", "zeta")
    for block in canonical.blocks:
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        assert np.all(identifiers[:-1] <= identifiers[1:])
    target_topology.require_lifted(
        canonical.topology, canonical.topology_id, canonical.blocks
    )
    target_topology.require_coordinates(canonical.coordinates)
    assert isinstance(target_topology.cell, PeriodicCell)
    assert isinstance(source_topology.cell, PeriodicCell)
    assert target_topology.cell.cell_id == source_topology.cell.cell_id
    np.testing.assert_array_equal(
        target_topology.cell.vectors, source_topology.cell.vectors
    )
    source_vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    target_vertex_ids = np.asarray(canonical.vertex_global_ids, dtype=np.int64)
    source_vertex_rows = {
        int(identifier): row for row, identifier in enumerate(source_vertex_ids)
    }
    source_root_ids = source_vertex_ids[
        np.asarray(source_topology.vertex_representatives, dtype=np.int64)
    ]
    target_root_ids = target_vertex_ids[
        np.asarray(target_topology.vertex_representatives, dtype=np.int64)
    ]
    assert np.any(source_root_ids > source_vertex_ids)
    for target_row, identifier in enumerate(target_vertex_ids):
        source_row = source_vertex_rows[int(identifier)]
        assert target_root_ids[target_row] == source_root_ids[source_row]
        np.testing.assert_array_equal(
            target_topology.vertex_shifts[target_row],
            source_topology.vertex_shifts[source_row],
        )

    source_geometry = source_topology._actual_geometry
    target_geometry = target_topology._actual_geometry
    assert isinstance(source_geometry, phx.discretization.CellGeometrySpec)
    assert isinstance(target_geometry, phx.discretization.CellGeometrySpec)
    cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in source.blocks)
    )
    reference_points = np.broadcast_to(
        np.asarray(((0.2, 0.3), (0.3, 0.2)), dtype=np.float64),
        (cell_ids.size, 2, 2),
    )
    np.testing.assert_array_equal(
        source_geometry.evaluate(source, cell_ids, reference_points),
        target_geometry.evaluate(canonical, cell_ids, reference_points),
    )
    assert target_topology.quotient.topology_id == source_topology.quotient.topology_id

    source_lifted_offsets = np.asarray(source_topology.lifted_offsets, dtype=np.int64)
    source_quotient_offsets = np.asarray(source_topology.quotient_offsets, dtype=np.int64)
    target_lifted_offsets = np.asarray(target_topology.lifted_offsets, dtype=np.int64)
    target_quotient_offsets = np.asarray(target_topology.quotient_offsets, dtype=np.int64)
    for degree in range(source.topological_dimension + 1):
        source_lifted_ids = np.asarray(
            source.entity_set(degree).entity_ids, dtype=np.int64
        )
        target_lifted_ids = np.asarray(
            canonical.entity_set(degree).entity_ids, dtype=np.int64
        )
        np.testing.assert_array_equal(
            np.sort(target_lifted_ids), np.sort(source_lifted_ids)
        )
        source_entity_rows = _entity_vertex_rows(source)[degree]
        target_entity_rows = _entity_vertex_rows(canonical)[degree]
        source_ids_by_vertices = {
            tuple(sorted(int(source_vertex_ids[int(vertex)]) for vertex in row)): int(
                identifier
            )
            for row, identifier in zip(source_entity_rows, source_lifted_ids, strict=True)
        }
        target_ids_by_vertices = {
            tuple(sorted(int(target_vertex_ids[int(vertex)]) for vertex in row)): int(
                identifier
            )
            for row, identifier in zip(target_entity_rows, target_lifted_ids, strict=True)
        }
        assert source_ids_by_vertices == target_ids_by_vertices
        source_rows = {
            int(identifier): row for row, identifier in enumerate(source_lifted_ids)
        }
        source_lifted_start, source_lifted_stop = map(
            int, source_lifted_offsets[degree : degree + 2]
        )
        source_quotient_start = int(source_quotient_offsets[degree])
        target_lifted_start, target_lifted_stop = map(
            int, target_lifted_offsets[degree : degree + 2]
        )
        target_quotient_start = int(target_quotient_offsets[degree])
        source_orbits = (
            np.asarray(source_topology.orbit_indices, dtype=np.int64)[
                source_lifted_start:source_lifted_stop
            ]
            - source_quotient_start
        )
        target_orbits = (
            np.asarray(target_topology.orbit_indices, dtype=np.int64)[
                target_lifted_start:target_lifted_stop
            ]
            - target_quotient_start
        )
        source_quotient_ids = np.asarray(
            source_topology.quotient.entity_sets[degree].entity_ids, dtype=np.int64
        )
        target_quotient_ids = np.asarray(
            target_topology.quotient.entity_sets[degree].entity_ids, dtype=np.int64
        )
        source_orientations = np.asarray(source_topology.orbit_orientations)[
            source_lifted_start:source_lifted_stop
        ]
        target_orientations = np.asarray(target_topology.orbit_orientations)[
            target_lifted_start:target_lifted_stop
        ]
        source_windings = np.asarray(source_topology.orbit_shifts)[
            source_lifted_start:source_lifted_stop
        ]
        target_windings = np.asarray(target_topology.orbit_shifts)[
            target_lifted_start:target_lifted_stop
        ]
        for target_row, identifier in enumerate(target_lifted_ids):
            source_row = source_rows[int(identifier)]
            source_orbit = int(source_orbits[source_row])
            target_orbit = int(target_orbits[target_row])
            assert source_quotient_ids[source_orbit] == target_quotient_ids[target_orbit]
            assert source_orientations[source_row] == target_orientations[target_row]
            np.testing.assert_array_equal(
                source_windings[source_row], target_windings[target_row]
            )

    assert phx.meshing.canonicalize_cell_mesh(canonical) is canonical


def _periodic_part(mesh: CellMesh) -> MeshPart:
    result = certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    return MeshPart("skew-periodic", result)


def test_periodic_cell_partition_uses_quotient_and_synchronizes_lifted_owners() -> None:
    mesh = _skew_periodic_mesh()
    layout = _periodic_cell_layout(mesh)
    assert layout is not None
    orbit_counts = np.bincount(layout.orbit_rows, minlength=layout.entity_ids.size)
    assert np.max(orbit_counts) > 1
    quotient_weights = 1.0 + np.arange(layout.entity_ids.size, dtype=np.float64)
    lifted_weights = quotient_weights[layout.orbit_rows]
    policy = MeshPartitionPolicy(
        MeshPartitionKind.GRAPH,
        2,
        halo_width=1,
        maximum_imbalance=2.0,
    )

    distribution = prepare_mesh_distribution(
        _periodic_part(mesh), policy=policy, cell_weights=lifted_weights
    )
    owner = np.asarray(distribution.partition.cell_owner, dtype=np.int64)
    quotient_owner = np.empty((layout.entity_ids.size,), dtype=np.int64)
    for orbit in range(layout.entity_ids.size):
        values = np.unique(owner[layout.orbit_rows == orbit])
        assert values.size == 1
        quotient_owner[orbit] = values[0]
    np.testing.assert_allclose(
        distribution.evidence.part_weights,
        np.bincount(quotient_owner, weights=quotient_weights, minlength=2),
    )
    adjacency = _periodic_quotient_adjacency(mesh)
    pairs = adjacency.undirected_pairs()
    expected_cut = np.count_nonzero(
        quotient_owner[pairs[:, 0]] != quotient_owner[pairs[:, 1]]
    )
    assert int(distribution.evidence.edge_cut) == expected_cut
    assert len(distribution.halo_rows) == 2


def test_periodic_orbit_closure_keeps_distinct_winding_edges_separate() -> None:
    mesh = _skew_periodic_mesh()
    topology = mesh.periodic_topology
    assert topology is not None
    edge_orbits = np.asarray(topology.orbit_indices, dtype=np.int64)[
        int(topology.lifted_offsets[1]) : int(topology.lifted_offsets[2])
    ] - int(topology.quotient_offsets[1])
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    quotient_start = int(topology.quotient_offsets[1])
    key_offsets = np.asarray(topology.entity_key_offsets, dtype=np.int64)
    key_values = np.asarray(topology.entity_key_values, dtype=np.int64)
    rank = np.asarray(topology.vertex_shifts).shape[1]
    same_representatives: dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]] = {}
    edge_count = topology.quotient.entity_sets[1].count
    for orbit in range(edge_count):
        row = quotient_start + orbit
        key = tuple(
            int(value) for value in key_values[key_offsets[row] : key_offsets[row + 1]]
        )
        vertices = np.asarray(key, dtype=np.int64).reshape((2, rank + 1))
        representative_pair = tuple(sorted(int(value) for value in vertices[:, 0]))
        same_representatives.setdefault(representative_pair, []).append((orbit, key))
    winding_pair = next(
        (first, second)
        for values in same_representatives.values()
        for first in values
        for second in values
        if first[0] != second[0] and first[1] != second[1]
    )
    first_orbit, second_orbit = winding_pair[0][0], winding_pair[1][0]
    seed_row = int(np.flatnonzero(edge_orbits == first_orbit)[0])
    closure = periodic_orbit_closure(mesh, 1, edge_ids[[seed_row]])
    closed_ids = closure.require_source(mesh, 1)
    closed_rows = np.flatnonzero(np.isin(edge_ids, closed_ids))
    assert np.all(edge_orbits[closed_rows] == first_orbit)
    assert second_orbit not in set(edge_orbits[closed_rows].tolist())

    lifted_edge_vertices = np.stack(_entity_vertex_rows(mesh)[1])
    lifted_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    first_rows = np.flatnonzero(edge_orbits == first_orbit)
    second_row = int(np.flatnonzero(edge_orbits == second_orbit)[0])
    first_signatures = _periodic_edge_operation_signatures(
        mesh, lifted_vertex_ids[lifted_edge_vertices[first_rows]]
    )
    second_signature = _periodic_edge_operation_signatures(
        mesh, lifted_vertex_ids[lifted_edge_vertices[[second_row]]]
    )[0]
    assert len(set(first_signatures)) == 1
    assert first_signatures[0] != second_signature

    entity_map = periodic_entity_orbit_map(mesh, 1)
    quotient_coefficients = np.arange(
        entity_map.quotient_entity_ids.size, dtype=np.float64
    )
    lifted_coefficients = entity_map.lift_oriented_scalars(quotient_coefficients)
    collapsed_coefficients = entity_map.collapse_oriented_scalars(lifted_coefficients)
    np.testing.assert_array_equal(collapsed_coefficients, quotient_coefficients)
    corner_offsets = np.asarray(entity_map.corner_permutation_offsets, dtype=np.int64)
    assert corner_offsets.shape == (edge_ids.size + 1,)
    assert np.all(np.diff(corner_offsets) == 2)
    assert np.asarray(entity_map.corner_permutations).size == 2 * edge_ids.size
    assert entity_map.winding_shifts.shape[0] == edge_ids.size

    vertex_map = periodic_entity_orbit_map(mesh, 0)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    identity_stencil = VertexInterpolationStencil(
        mesh.entity_set(0).entity_set_id,
        mesh.entity_set(0).entity_set_id,
        vertex_ids,
        vertex_ids[:, None],
        np.ones((vertex_ids.size, 1), dtype=np.float64),
        np.ones((vertex_ids.size, 1), dtype=np.bool_),
    )
    vertex_quotient_values = np.arange(
        vertex_map.quotient_entity_ids.size, dtype=np.float64
    )
    np.testing.assert_array_equal(
        transfer_periodic_p1_vertices(
            mesh, mesh, identity_stencil, vertex_quotient_values
        ),
        vertex_quotient_values,
    )
    stale = CellMesh(
        mesh.coordinates,
        mesh.blocks,
        vertex_global_ids=mesh.vertex_global_ids,
        numeric_version="stale-periodic-revision",
        periodic_topology=topology,
    )
    with pytest.raises(ValueError, match="different source mesh revision"):
        closure.require_source(stale, 1)


def test_periodic_refinement_edge_lock_enforces_cell_and_edge_orbits() -> None:
    mesh = _skew_periodic_mesh()
    cell_layout = _periodic_cell_layout(mesh)
    assert cell_layout is not None
    edge_layout = _entity_layout(mesh, 1)
    cell_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    cell_edges = _cell_edge_rows(mesh)

    prepared = None
    selected_cells = selected_edges = None
    for orbit in np.unique(cell_layout.orbit_rows):
        rows = np.flatnonzero(cell_layout.orbit_rows == orbit)
        if rows.size < 2:
            continue
        edge_orbits = [set(edge_layout.orbit_rows[cell_edges[row]]) for row in rows]
        common = set.intersection(*edge_orbits)
        for edge_orbit in sorted(common):
            edge_rows = np.asarray(
                [
                    cell_edges[row][
                        edge_layout.orbit_rows[cell_edges[row]] == edge_orbit
                    ][0]
                    for row in rows
                ],
                dtype=np.int64,
            )
            try:
                prepared = prepare_periodic_refinement_edge_lock(
                    mesh, cell_ids[rows], edge_ids[edge_rows]
                )
            except ValueError:
                continue
            selected_cells, selected_edges = cell_ids[rows], edge_ids[edge_rows]
            break
        if prepared is not None:
            break

    assert prepared is not None
    assert selected_cells is not None
    assert selected_edges is not None
    locked_cells, locked_edges = prepared.require_source(mesh)
    np.testing.assert_array_equal(locked_cells, np.sort(selected_cells))
    np.testing.assert_array_equal(
        locked_edges, selected_edges[np.argsort(selected_cells)]
    )

    rows = np.flatnonzero(np.isin(cell_ids, selected_cells))
    source_edge_row = int(np.flatnonzero(edge_ids == selected_edges[0])[0])
    source_edge_orbit = edge_layout.orbit_rows[source_edge_row]
    first_row = int(rows[0])
    alternate = cell_edges[first_row][
        edge_layout.orbit_rows[cell_edges[first_row]] != source_edge_orbit
    ]
    if alternate.size:
        mismatched_edges = selected_edges.copy()
        selected_position = int(np.flatnonzero(selected_cells == cell_ids[first_row])[0])
        mismatched_edges[selected_position] = edge_ids[alternate[0]]
        with pytest.raises(ValueError, match="different quotient edges"):
            prepare_periodic_refinement_edge_lock(mesh, selected_cells, mismatched_edges)

    stale = CellMesh(
        mesh.coordinates,
        mesh.blocks,
        vertex_global_ids=mesh.vertex_global_ids,
        numeric_version="stale-periodic-edge-lock",
        periodic_topology=mesh.periodic_topology,
    )

    with pytest.raises(ValueError, match="different source mesh revision"):
        prepared.require_source(stale)


def test_periodic_tetra_refinement_edge_lock_uses_all_six_cell_edges() -> None:
    mesh = _periodic_tetra_mesh()
    edge_layout = _entity_layout(mesh, 1)
    cell_ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    cell_edges = _cell_edge_rows(mesh)
    assert cell_edges.shape == (cell_ids.size, 6)
    cell_map = periodic_entity_orbit_map(mesh, 3)
    corner_offsets = np.asarray(cell_map.corner_permutation_offsets, dtype=np.int64)
    assert corner_offsets.shape == (cell_ids.size + 1,)
    assert np.all(np.diff(corner_offsets) == 4)
    assert np.asarray(cell_map.corner_permutations).size == 4 * cell_ids.size

    lock = None
    for edge_orbit in np.unique(edge_layout.orbit_rows):
        edge_rows = np.flatnonzero(edge_layout.orbit_rows == edge_orbit)
        cell_rows = np.flatnonzero(np.any(np.isin(cell_edges, edge_rows), axis=1))
        selected_rows = np.asarray(
            [
                cell_edges[row][np.isin(cell_edges[row], edge_rows)][0]
                for row in cell_rows
            ],
            dtype=np.int64,
        )
        try:
            lock = prepare_periodic_refinement_edge_lock(
                mesh, cell_ids[cell_rows], edge_ids[selected_rows]
            )
        except ValueError:
            continue
        break

    assert lock is not None
    locked_cells, locked_edges = lock.require_source(mesh)
    assert locked_cells.size == locked_edges.size


def test_periodic_local_metric_stages_a_quotient_checked_outcome() -> None:
    mesh = _skew_periodic_mesh()
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    edges = np.stack(_entity_vertex_rows(mesh)[1])
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    scale = 0.5 * (1.0 / np.sqrt(2.0) / np.min(lengths) + np.sqrt(2.0) / np.max(lengths))
    metric = np.broadcast_to(scale**2 * np.eye(2), (points.shape[0], 2, 2)).copy()
    metric_lengths = scale * lengths
    assert np.all(
        (metric_lengths >= 1.0 / np.sqrt(2.0)) & (metric_lengths <= np.sqrt(2.0))
    )

    outcome = execute_local_metric_adaptation(
        mesh,
        metric,
        cell_classes=np.zeros((mesh.entity_set(2).entity_ids.size,), dtype=np.int64),
        edge_classes=np.ones((mesh.entity_set(1).entity_ids.size,), dtype=np.int64),
        protected_edges=np.zeros((edges.shape[0],), dtype=np.bool_),
        fixed_vertices=np.ones((points.shape[0],), dtype=np.bool_),
        predicate_mode=PredicateMode.EXACT,
        topology_operations=False,
        relocation=False,
    )

    assert outcome.evidence.converged
    assert outcome.evidence.splits == 0
    assert outcome.evidence.collapses == 0
    assert outcome.evidence.flips == 0
    assert outcome.evidence.relocations == 0
    assert mesh.periodic_topology is not None

    layout = _periodic_cell_layout(mesh)
    assert layout is not None
    source_cell_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    orbit = next(
        orbit_id
        for orbit_id in np.unique(layout.orbit_rows)
        if np.count_nonzero(layout.orbit_rows == orbit_id) > 1
    )
    member = int(np.flatnonzero(layout.orbit_rows == orbit)[0])
    cell_id = int(source_cell_ids[member])
    asymmetric_relation = EntityRelations(
        2,
        np.asarray([[cell_id]], dtype=np.int64),
        np.asarray([[cell_id]], dtype=np.int64),
        np.asarray([int(EntityLineageKind.RELOCATED)], dtype=np.int32),
    )
    relations = list(outcome.edit.relations)
    relations[2] = asymmetric_relation
    original_source_id = mesh.mesh_id
    with pytest.raises(ValueError, match="same operation"):
        stage_periodic_topology_edit(
            mesh, outcome.edit._replace(relations=tuple(relations))
        )
    assert mesh.mesh_id == original_source_id


def test_periodic_local_metric_commits_a_full_split_orbit() -> None:
    mesh = _skew_periodic_mesh()
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    edge_count = mesh.entity_set(1).entity_ids.size
    metric = np.broadcast_to(4.0 * np.eye(2), (points.shape[0], 2, 2)).copy()
    outcome = execute_local_metric_adaptation(
        mesh,
        metric,
        cell_classes=np.zeros((mesh.entity_set(2).entity_ids.size,), dtype=np.int64),
        edge_classes=np.ones((edge_count,), dtype=np.int64),
        protected_edges=np.zeros((edge_count,), dtype=np.bool_),
        fixed_vertices=np.ones((points.shape[0],), dtype=np.bool_),
        predicate_mode=PredicateMode.EXACT,
        topology_operations=True,
        relocation=False,
    )

    target = _require_periodic_outcome(mesh, outcome)
    assert outcome.evidence.splits > 0
    assert target.vertex_global_ids.size > mesh.vertex_global_ids.size
    target_topology = target.periodic_topology
    source_topology = mesh.periodic_topology
    assert target_topology is not None and source_topology is not None
    assert target_topology.vertex_orbit_count > source_topology.vertex_orbit_count


def test_periodic_topology_edit_stages_non_binary64_split_roots_with_exact_bank() -> None:
    from phydrax.discretization._exact_plc_geometry import (
        ExactPlcCellGeometrySource,
    )
    from phydrax.discretization._periodic_exact import exact_periodic_element

    # The unlocked 2D staging path: a split orbit on the skew (non-dyadic) lattice
    # whose roots are exact quotient-chart midpoints. Their lifted members are the
    # canonical roundings of exact translates, which are not always translates of
    # the rounded roots, so only the exact vertex bank witnesses the quotient.
    mesh = _skew_periodic_mesh()
    source_topology = mesh.periodic_topology
    assert source_topology is not None
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    edge_count = mesh.entity_set(1).entity_ids.size
    outcome = execute_local_metric_adaptation(
        mesh,
        np.broadcast_to(4.0 * np.eye(2), (points.shape[0], 2, 2)).copy(),
        cell_classes=np.zeros((mesh.entity_set(2).entity_ids.size,), dtype=np.int64),
        edge_classes=np.ones((edge_count,), dtype=np.int64),
        protected_edges=np.zeros((edge_count,), dtype=np.bool_),
        fixed_vertices=np.ones((points.shape[0],), dtype=np.bool_),
        predicate_mode=PredicateMode.EXACT,
        topology_operations=True,
        relocation=False,
    )
    assert outcome.evidence.splits > 0
    edit = outcome.edit
    rounded_target = _require_periodic_outcome(mesh, outcome)
    target_topology = rounded_target.periodic_topology
    assert target_topology is not None
    cell = target_topology.cell

    def exact_lift(
        root: tuple[Fraction, ...], shift: np.ndarray, /
    ) -> tuple[Fraction, ...]:
        matrix = exact_periodic_element(cell, shift)
        base = (*root, Fraction(1))
        return tuple(
            sum((matrix[axis][column] * base[column] for column in range(3)), Fraction(0))
            for axis in range(2)
        )

    source_representatives = np.asarray(
        source_topology.vertex_representatives, dtype=np.int64
    )
    source_shifts = np.asarray(source_topology.vertex_shifts, dtype=np.int64)
    source_exact = [
        exact_lift(
            tuple(Fraction(float(value)) for value in points[root]), source_shifts[row]
        )
        for row, root in enumerate(source_representatives.tolist())
    ]
    assert all(
        tuple(float(value) for value in row) == tuple(points[index])
        for index, row in enumerate(source_exact)
    )
    edge_vertices = np.stack(_entity_vertex_rows(mesh)[1])
    exact_midpoints = {}
    for first, second in edge_vertices.tolist():
        midpoint = tuple(
            (source_exact[first][axis] + source_exact[second][axis]) / 2
            for axis in range(2)
        )
        exact_midpoints.setdefault(tuple(float(value) for value in midpoint), midpoint)
    source_rows = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(mesh.vertex_global_ids).tolist())
    }
    target_ids = np.asarray(edit.vertex_global_ids, dtype=np.int64)
    coordinates = np.asarray(edit.coordinates, dtype=np.float64)
    representatives = np.asarray(target_topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(target_topology.vertex_shifts, dtype=np.int64)
    roots: dict[int, tuple[Fraction, ...]] = {}
    for root in np.unique(representatives).tolist():
        source_row = source_rows.get(int(target_ids[root]))
        if source_row is not None:
            roots[root] = source_exact[source_row]
        else:
            roots[root] = exact_midpoints.get(
                tuple(coordinates[root]),
                tuple(Fraction(float(value)) for value in coordinates[root]),
            )
    exact_rows = tuple(
        exact_lift(roots[root], shifts[row])
        for row, root in enumerate(representatives.tolist())
    )
    assert any(Fraction(float(value)) != value for row in exact_rows for value in row)
    canonical = np.asarray(
        [[float(value) for value in row] for row in exact_rows], dtype=np.float64
    )
    assert np.any(canonical != coordinates)
    canonical_edit = edit._replace(coordinates=canonical)
    bank = ExactPlcCellGeometrySource(
        "periodic-quotient-chart",
        mesh.mesh_id,
        target_ids,
        exact_rows,
        vertex_source_entity_ids=(None,) * target_ids.size,
        vertex_occurrence_paths=((),) * target_ids.size,
    )

    with pytest.raises(ValueError, match="vertex_shifts"):
        stage_periodic_topology_edit(mesh, canonical_edit)
    target = stage_periodic_topology_edit(
        mesh, canonical_edit, exact_coordinate_source=bank
    )
    assert target is not None
    staged_topology = target.periodic_topology
    assert staged_topology is not None
    assert staged_topology.coordinate_source_map_id == bank.source_map_id
    np.testing.assert_array_equal(target.coordinates, canonical)
    staged_topology.require_lifted(
        target.topology, target.lifted_topology_id, target.blocks
    )
    staged_topology.require_coordinates(target.coordinates)
    assert staged_topology.vertex_orbit_count == target_topology.vertex_orbit_count


def test_periodic_local_metric_commits_a_full_collapse_orbit() -> None:
    mesh = _periodic_grid_mesh(4, perturbation=0.02)
    metric = np.broadcast_to(0.01 * np.eye(2), (mesh.coordinates.shape[0], 2, 2)).copy()
    outcome = _periodic_metric_adaptation(
        mesh, metric, topology_operations=True, relocation=False
    )

    target = _require_periodic_outcome(mesh, outcome)
    assert outcome.evidence.collapses > 0
    assert target.vertex_global_ids.size < mesh.vertex_global_ids.size
    target_topology = target.periodic_topology
    source_topology = mesh.periodic_topology
    assert target_topology is not None and source_topology is not None
    assert target_topology.vertex_orbit_count < source_topology.vertex_orbit_count


def test_periodic_local_metric_commits_a_full_flip_orbit() -> None:
    mesh = _periodic_grid_mesh(2)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    edges = np.stack(_entity_vertex_rows(mesh)[1])
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    scale = 0.5 * (1.0 / np.sqrt(2.0) / np.min(lengths) + np.sqrt(2.0) / np.max(lengths))
    metric = np.broadcast_to(scale**2 * np.eye(2), (points.shape[0], 2, 2)).copy()
    outcome = _periodic_metric_adaptation(
        mesh, metric, topology_operations=True, relocation=False
    )

    target = _require_periodic_outcome(mesh, outcome)
    assert outcome.evidence.flips > 0
    assert target.topology_id != mesh.topology_id


def test_periodic_metric_edge_classes_use_quotient_incidence_on_multi_period_covers() -> (
    None
):
    from phydrax.meshing._adaptation import _edge_classes
    from phydrax.meshing._periodic import periodic_quotient_facet_cell_classes

    # A 2 x 2 cover of the torus: its outer boundary edges have one incident
    # lifted cell, while their orbit mates inside the cover have two.
    mesh = _periodic_grid_mesh(2)
    edge_orbits = np.asarray(
        periodic_entity_orbit_map(mesh, 1).lifted_to_quotient, dtype=np.int64
    )
    cell_orbits = np.asarray(
        periodic_entity_orbit_map(mesh, 2).lifted_to_quotient, dtype=np.int64
    )
    relation = mesh.topology.incidences[1].relation
    valid = np.asarray(relation.valid, dtype=np.bool_)
    edge_rows = np.asarray(relation.source_indices, dtype=np.int64)[valid]
    cell_rows = np.asarray(relation.target_indices, dtype=np.int64)[valid]
    lifted_incidence = np.bincount(edge_rows, minlength=edge_orbits.size)
    assert np.any(lifted_incidence == 1)
    assert np.any(lifted_incidence == 2)

    free = _edge_classes(mesh, (), np.zeros((cell_orbits.size,), dtype=np.int64))
    np.testing.assert_array_equal(free, 0)

    cell_classes = (cell_orbits % 2).astype(np.int64)
    incident, lowest, highest = periodic_quotient_facet_cell_classes(mesh, cell_classes)
    np.testing.assert_array_equal(incident, 2)
    # Oracle: the cover holds every quotient incidence of an edge orbit through
    # some member, so an orbit is an interface iff its members see two classes.
    orbit_classes: dict[int, set[int]] = {}
    for edge_row, cell_row in zip(edge_rows.tolist(), cell_rows.tolist(), strict=True):
        orbit_classes.setdefault(int(edge_orbits[edge_row]), set()).add(
            int(cell_classes[cell_row])
        )
    interface = np.asarray(
        [len(orbit_classes[int(orbit)]) > 1 for orbit in edge_orbits], dtype=np.bool_
    )
    assert np.any(interface) and not np.all(interface)
    np.testing.assert_array_equal(lowest != highest, interface)
    classes = _edge_classes(mesh, (), cell_classes)
    np.testing.assert_array_equal(classes > 0, interface)
    for orbit in np.unique(edge_orbits):
        assert np.unique(classes[edge_orbits == orbit]).size == 1

    source = certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    vertices = source.mesh.entity_set(0)
    scope = phx.meshing.MeshingScope(
        source.mesh.mesh_id,
        source.mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        0,
        vertices.entity_set_id,
        vertices.entity_ids,
    )
    metric = phx.meshing.MeshMetricField(
        scope,
        np.broadcast_to(16.0 * np.eye(2), (vertices.entity_ids.shape[0], 2, 2)),
        minimum_size=1.0e-3,
        maximum_size=10.0,
    )
    result = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MetricMeshAdaptation(metric),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_METRIC_2D
            ),
        )
    )
    assert result.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    target = result.target.mesh
    assert target.periodic_topology is not None
    assert target.entity_set(2).count > mesh.entity_set(2).count


def test_periodic_local_metric_commits_a_full_relocation_orbit() -> None:
    mesh = _periodic_grid_mesh(4, perturbation=0.08)
    metric = np.broadcast_to(4.0 * np.eye(2), (mesh.coordinates.shape[0], 2, 2)).copy()
    outcome = _periodic_metric_adaptation(
        mesh, metric, topology_operations=False, relocation=True
    )

    target = _require_periodic_outcome(mesh, outcome)
    assert outcome.evidence.relocations > 0
    assert not np.array_equal(target.coordinates, mesh.coordinates)
    target_ids = np.asarray(outcome.edit.vertex_global_ids, dtype=np.int64)
    source_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    source_rows = {
        int(identifier): row for row, identifier in enumerate(source_ids.tolist())
    }
    source_points = np.asarray(mesh.coordinates, dtype=np.float64)
    stencil_sources = np.asarray(outcome.edit.stencil_sources, dtype=np.int64)
    stencil_weights = np.asarray(outcome.edit.stencil_weights, dtype=np.float64)
    stencil_valid = np.asarray(outcome.edit.stencil_valid, dtype=np.bool_)
    moved_rows = []
    for row, identifier in enumerate(target_ids.tolist()):
        source_row = source_rows.get(identifier)
        if source_row is None or np.array_equal(
            source_points[source_row], outcome.edit.coordinates[row]
        ):
            continue
        moved_rows.append(row)
        support = np.flatnonzero(stencil_valid[row])
        # Binary64 barycentric weights are roundings; the exact relocation
        # stencil owns the partition of unity.
        exact_weights = outcome.edit.stencil_exact_weights
        assert exact_weights is not None
        assert (
            sum(
                (exact_weights[row][column] for column in support.tolist()),
                Fraction(0),
            )
            == 1
        )
        assert not (
            support.size == 1
            and int(stencil_sources[row, support[0]]) == identifier
            and stencil_weights[row, support[0]] == 1.0
        )
    assert moved_rows

    periodic = target.periodic_topology
    assert periodic is not None
    coordinates = np.asarray(target.coordinates, dtype=np.float64)
    representatives = np.asarray(periodic.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(periodic.vertex_shifts, dtype=np.int64)
    for row, representative in enumerate(representatives.tolist()):
        expected = apply_periodic(
            periodic.cell,
            coordinates[representative][None, :],
            shifts[row : row + 1],
        )[0]
        np.testing.assert_array_equal(coordinates[row], expected)


def test_periodic_descriptor_rebuilds_from_stable_identity_stencils() -> None:
    source = _skew_periodic_mesh()
    target = CellMesh(
        source.coordinates,
        source.blocks,
        vertex_global_ids=source.vertex_global_ids,
    )
    vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    stencil = VertexInterpolationStencil(
        source.entity_set(0).entity_set_id,
        target.entity_set(0).entity_set_id,
        vertex_ids,
        vertex_ids[:, None],
        np.ones((vertex_ids.size, 1), dtype=np.float64),
        np.ones((vertex_ids.size, 1), dtype=np.bool_),
    )

    rebuilt = periodic_topology_after_edit(source, target, stencil)

    assert isinstance(rebuilt, PeriodicMeshTopology)
    original = source.periodic_topology
    assert original is not None
    np.testing.assert_array_equal(
        rebuilt.vertex_representatives, original.vertex_representatives
    )
    np.testing.assert_array_equal(rebuilt.vertex_shifts, original.vertex_shifts)
    assert rebuilt.quotient.topology_id == original.quotient.topology_id


def test_periodic_explicit_ancestry_requires_real_stencil_support() -> None:
    source = _skew_periodic_mesh()
    target = CellMesh(
        source.coordinates,
        source.blocks,
        vertex_global_ids=source.vertex_global_ids,
    )
    vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    vertex_map = periodic_entity_orbit_map(source, 0)
    quotient_ids = np.asarray(vertex_map.quotient_entity_ids, dtype=np.int64)[
        np.asarray(vertex_map.lifted_to_quotient, dtype=np.int64)
    ]
    keys = np.column_stack((quotient_ids, np.zeros((vertex_ids.size,), dtype=np.int64)))
    windings = np.asarray(vertex_map.winding_shifts, dtype=np.int64)
    target_set_id = target.entity_set(0).entity_set_id
    source_set_id = source.entity_set(0).entity_set_id
    stencil = VertexInterpolationStencil(
        source_set_id,
        target_set_id,
        vertex_ids,
        vertex_ids[:, None],
        np.ones((vertex_ids.size, 1), dtype=np.float64),
        np.ones((vertex_ids.size, 1), dtype=np.bool_),
    )

    rebuilt = periodic_topology_after_edit(
        source,
        target,
        stencil,
        vertex_orbit_keys=keys,
        vertex_windings=windings,
        source_entity_degree=0,
    )

    assert isinstance(rebuilt, PeriodicMeshTopology)
    invalid_stencil = VertexInterpolationStencil(
        source_set_id,
        target_set_id,
        vertex_ids,
        np.full((vertex_ids.size, 1), -1, dtype=np.int64),
        np.zeros((vertex_ids.size, 1), dtype=np.float64),
        np.zeros((vertex_ids.size, 1), dtype=np.bool_),
        preserves_constants=False,
    )
    with pytest.raises(ValueError, match="source interpolation ancestry"):
        periodic_topology_after_edit(
            source,
            target,
            invalid_stencil,
            vertex_orbit_keys=keys,
            vertex_windings=windings,
            source_entity_degree=0,
        )


def test_native_periodic_delaunay_constructs_a_closed_skew_quotient() -> None:
    cell = PeriodicCell(np.asarray(((1.2, 0.1), (0.35, 1.05)), dtype=np.float64))
    fractional = np.asarray(
        ((0.11, 0.13), (0.58, 0.17), (0.31, 0.62), (0.84, 0.77)),
        dtype=np.float64,
    )
    points = np.asarray(cell.cartesian(fractional), dtype=np.float64)
    point_ids = np.asarray((91, 4, 53, 17), dtype=np.int64)
    first = periodic_delaunay_mesh(PeriodicPointOrbits(cell, points, point_ids))
    order = np.asarray((2, 0, 3, 1), dtype=np.int64)
    reordered = periodic_delaunay_mesh(
        PeriodicPointOrbits(cell, points[order], point_ids[order])
    )

    assert isinstance(first, PeriodicMeshConstruction)
    assert first.mesh.periodic_topology is not None
    assert first.mesh.periodic_topology.vertex_orbit_count == point_ids.size
    assert first.mesh.mesh_id == reordered.mesh.mesh_id
    np.testing.assert_array_equal(
        first.mesh.vertex_global_ids, reordered.mesh.vertex_global_ids
    )
    block = first.mesh.blocks[0]
    assert block.cell_kind == "triangle"
    cells = np.asarray(block.vertices, dtype=np.int64)
    coordinates = np.asarray(first.mesh.coordinates, dtype=np.float64)
    triangles = coordinates[cells]
    signed_areas = 0.5 * (
        (triangles[:, 1, 0] - triangles[:, 0, 0])
        * (triangles[:, 2, 1] - triangles[:, 0, 1])
        - (triangles[:, 1, 1] - triangles[:, 0, 1])
        * (triangles[:, 2, 0] - triangles[:, 0, 0])
    )
    assert np.all(signed_areas > 0.0)
    np.testing.assert_allclose(
        np.sum(signed_areas), cell.cell_measure, rtol=0.0, atol=1.0e-10
    )


def test_periodic_descriptor_rebuilds_new_vertices_from_edge_stencils() -> None:
    source = _skew_periodic_mesh()
    topology = source.periodic_topology
    assert topology is not None
    lifted_offsets = np.asarray(topology.lifted_offsets, dtype=np.int64)
    quotient_offsets = np.asarray(topology.quotient_offsets, dtype=np.int64)
    edge_orbits = (
        np.asarray(topology.orbit_indices, dtype=np.int64)[
            lifted_offsets[1] : lifted_offsets[2]
        ]
        - quotient_offsets[1]
    )
    orbit_counts = np.bincount(
        edge_orbits, minlength=topology.quotient.entity_sets[1].count
    )
    edge_orbit = int(np.flatnonzero(orbit_counts > 1)[0])
    edge_rows = np.asarray(_entity_vertex_rows(source)[1], dtype=object)
    members = np.flatnonzero(edge_orbits == edge_orbit)
    source_edge_ids = np.asarray(source.entity_set(1).entity_ids, dtype=np.int64)
    members = members[np.argsort(source_edge_ids[members], kind="stable")]
    source_vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    next_vertex_id = int(np.max(source_vertex_ids)) + 1
    target_vertex_ids = np.concatenate(
        (
            source_vertex_ids,
            np.arange(next_vertex_id, next_vertex_id + members.size, dtype=np.int64),
        )
    )
    # The first new vertex is its orbit root; every other image is its exact
    # lattice image, the one canonical lift the quotient descriptor accepts.
    vertex_roots = np.asarray(topology.vertex_representatives, dtype=np.int64)
    vertex_shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    root_endpoints = np.asarray(edge_rows[int(members[0])], dtype=np.int64)
    root_midpoint = np.mean(np.asarray(source.coordinates)[root_endpoints], axis=0)

    def edge_translation(edge_row: int, /) -> np.ndarray:
        endpoints = np.asarray(edge_rows[edge_row], dtype=np.int64)
        for ordered in (endpoints, endpoints[::-1]):
            deltas = vertex_shifts[ordered] - vertex_shifts[root_endpoints]
            if np.array_equal(
                vertex_roots[ordered], vertex_roots[root_endpoints]
            ) and np.array_equal(deltas[0], deltas[1]):
                return deltas[0]
        raise ValueError("The edge-orbit member is not a lattice image of its root.")

    midpoint_coordinates = apply_periodic(
        topology.cell,
        np.broadcast_to(root_midpoint, (members.size, root_midpoint.size)),
        np.stack(tuple(edge_translation(int(edge_row)) for edge_row in members)),
    )
    target_coordinates = np.concatenate(
        (np.asarray(source.coordinates), midpoint_coordinates), axis=0
    )
    target = CellMesh(
        target_coordinates,
        source.blocks,
        vertex_global_ids=target_vertex_ids,
    )
    stencil_sources = np.full((target_vertex_ids.size, 2), -1, dtype=np.int64)
    stencil_weights = np.zeros((target_vertex_ids.size, 2), dtype=np.float64)
    stencil_valid = np.zeros((target_vertex_ids.size, 2), dtype=np.bool_)
    stencil_sources[: source_vertex_ids.size, 0] = source_vertex_ids
    stencil_weights[: source_vertex_ids.size, 0] = 1.0
    stencil_valid[: source_vertex_ids.size, 0] = True
    for index, edge_row in enumerate(members, start=source_vertex_ids.size):
        endpoints = np.asarray(edge_rows[int(edge_row)], dtype=np.int64)
        stencil_sources[index] = source_vertex_ids[endpoints]
        stencil_weights[index] = 0.5
        stencil_valid[index] = True
    stencil = VertexInterpolationStencil(
        source.entity_set(0).entity_set_id,
        target.entity_set(0).entity_set_id,
        target_vertex_ids,
        stencil_sources,
        stencil_weights,
        stencil_valid,
    )

    rebuilt = periodic_topology_after_edit(source, target, stencil)

    assert isinstance(rebuilt, PeriodicMeshTopology)
    new_rows = np.arange(source_vertex_ids.size, target_vertex_ids.size)
    assert np.unique(np.asarray(rebuilt.vertex_representatives)[new_rows]).size == 1
    assert (
        np.unique(np.asarray(rebuilt.vertex_shifts)[new_rows], axis=0).shape[0]
        == members.size
    )
    assert (
        rebuilt.quotient.entity_sets[0].count
        == topology.quotient.entity_sets[0].count + 1
    )


def test_periodic_halo_expansion_follows_discovered_orbits_without_global_gather() -> (
    None
):
    mesh = _skew_periodic_mesh()
    layout = _periodic_cell_layout(mesh)
    assert layout is not None
    policy = MeshPartitionPolicy(
        MeshPartitionKind.GRAPH,
        2,
        halo_width=0,
        maximum_imbalance=2.0,
    )
    distribution = prepare_mesh_distribution(_periodic_part(mesh), policy=policy)
    owner = np.asarray(distribution.partition.cell_owner, dtype=np.int64)
    cell_ids = np.asarray(distribution.cell_global_ids, dtype=np.int64)
    remote = np.flatnonzero(owner != 0)
    assert remote.size
    requested_row = int(remote[0])
    requested_orbit = int(layout.orbit_rows[requested_row])
    requests = [np.zeros((0,), dtype=np.int64) for _ in range(2)]
    requests[0] = cell_ids[[requested_row]]

    expanded = distribution.expand_halos(tuple(requests))

    expected = cell_ids[layout.orbit_rows == requested_orbit]
    assert set(np.asarray(expanded.halo_global_ids[0]).tolist()) == set(expected.tolist())
    assert expanded.halo_global_ids[0].size < cell_ids.size
    assert distribution.halo_global_ids[0].size == 0


def test_periodic_native_bisection_uses_one_orbit_locked_edge_plan() -> None:

    mesh = _skew_periodic_mesh()
    cell_layout = _periodic_cell_layout(mesh)
    edge_layout = _entity_layout(mesh, 1)
    assert cell_layout is not None
    source = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    source_edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    cell_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    edge_orbits = np.asarray(edge_layout.orbit_rows, dtype=np.int64)
    cell_edges = _cell_edge_rows(mesh)
    selected = None
    result = None
    for orbit in np.unique(np.asarray(cell_layout.orbit_rows, dtype=np.int64)):
        rows = np.flatnonzero(np.asarray(cell_layout.orbit_rows) == orbit)
        orbit_cells = cell_ids[rows]
        common = set(int(value) for value in edge_orbits[cell_edges[rows[0]]])
        for row in rows[1:]:
            common &= set(int(value) for value in edge_orbits[cell_edges[row]])
        for edge_orbit in sorted(common):
            chosen_edges = []
            for row in rows:
                edge_rows = cell_edges[row]
                member = int(
                    edge_rows[np.flatnonzero(edge_orbits[edge_rows] == edge_orbit)[0]]
                )
                chosen_edges.append(int(source_edge_ids[member]))
            try:
                lock = prepare_periodic_refinement_edge_lock(
                    mesh,
                    orbit_cells,
                    np.asarray(chosen_edges, dtype=np.int64),
                )
                request = phx.meshing.MarkedMeshAdaptation(
                    orbit_cells, periodic_edge_lock=lock
                )
                prepared = phx.meshing.prepare_mesh_adaptation(
                    source,
                    request,
                    policy=phx.meshing.MeshAdaptationPolicy(
                        phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
                        compatibility=phx.meshing.BisectionCompatibility.REJECT,
                    ),
                )
                result = phx.meshing.execute_mesh_adaptation(prepared)
            except (ValueError, phx.meshing.MeshingFailure):
                continue
            selected = lock
            break
        if result is not None:
            break

    assert selected is not None and result is not None
    transition = result.transition
    assert transition is not None
    target = result.target.mesh
    target_topology = target.periodic_topology
    assert target_topology is not None
    target_topology.require_lifted(
        target.topology, target.lifted_topology_id, target.blocks
    )
    target_topology.require_coordinates(target.coordinates)
    lineage = transition.lineage
    assert lineage is not None
    assert lineage.target_topology_id == target.topology_id

    distribution_policy = MeshPartitionPolicy(
        MeshPartitionKind.GRAPH,
        2,
        halo_width=0,
        maximum_imbalance=2.0,
    )
    source_distribution = prepare_mesh_distribution(
        _periodic_part(mesh), policy=distribution_policy
    )
    # The target's non-binary64 bisection midpoints are canonical roundings of
    # exact quotient-chart translates: it is published with its exact vertex bank,
    # so the certified adaptation result (not a re-certified bare mesh) is the part.
    distribution_transition = prepare_distribution_transition(
        source_distribution,
        MeshPart("skew-periodic", result.target),
        lineage,
        policy=distribution_policy,
    )
    target_layout = _periodic_cell_layout(target)
    assert target_layout is not None
    target_owner = np.asarray(
        distribution_transition.target.partition.cell_owner, dtype=np.int64
    )
    target_cell_orbits = np.asarray(target_layout.orbit_rows, dtype=np.int64)
    assert all(
        np.unique(target_owner[target_cell_orbits == quotient_row]).size == 1
        for quotient_row in range(target_layout.entity_ids.size)
    )
    source_values = np.asarray(source_distribution.cell_global_ids, dtype=np.float64)
    migrated_values = np.asarray(distribution_transition.transfer(source_values))
    assert np.all(np.asarray(distribution_transition.target_defined))
    assert np.all(np.isin(migrated_values, source_values))

    geometry_transition = transition.geometry_transition
    assert (
        geometry_transition is not None and geometry_transition.reference_map is not None
    )
    source_complex = phx.discretization.fem.FiniteElementDeRhamComplex(
        mesh,
        family="trimmed",
        order=1,
    )
    source_quotient = source_complex.transfer_periodic_quotient(
        source_complex
    ).complex_map.source
    edge_form = phx.discretization.fem.form_element(
        "triangle",
        1,
        1,
        family="trimmed",
        twist="untwisted",
        proxy="circulation",
    )
    field_spec = phx.discretization.FiniteElementFieldSpec("hcurl", edge_form)
    quotient_layout = phx.solver.FiniteElementAcceptedFieldLayout(
        field_spec,
        periodic_quotient=True,
    )
    quotient_state = phx.solver.FiniteElementAcceptedState(
        (np.ones((source_quotient.space(1).size,), dtype=np.float64),),
        0.0,
        0,
        mesh.topology_id,
        "prepared-periodic-quotient",
        "compiled-periodic-quotient",
        field_layouts=(quotient_layout,),
    )
    transaction = phx.solver.FiniteElementTopologyTransaction(
        lambda candidate, fields, materials, lineage, args: (
            candidate.topology_id == lineage.target_topology_id and len(fields) == 1
        )
    )
    advanced = transaction.execute(quotient_state, mesh, result)
    assert bool(advanced.committed)
    assert len(advanced.periodic_transfers) == 1
    quotient_transfer = advanced.periodic_transfers[0]
    assert (
        quotient_transfer.complex_map.source.space(1).size
        == quotient_state.fields[0].size
    )
    assert (
        quotient_transfer.complex_map.target.space(1).size
        == advanced.state.fields[0].size
    )

    with pytest.raises(ValueError, match="explicitly declare lifted or quotient"):
        phx.solver.FiniteElementAcceptedFieldLayout(field_spec)
    lifted_layout = phx.solver.FiniteElementAcceptedFieldLayout(
        field_spec,
        periodic_quotient=False,
    )
    lifted_state = phx.solver.FiniteElementAcceptedState(
        quotient_state.fields,
        quotient_state.time,
        quotient_state.step,
        mesh.topology_id,
        "prepared-lifted-cover",
        "compiled-lifted-cover",
        field_layouts=(lifted_layout,),
    )
    refused = transaction.execute(lifted_state, mesh, result)
    assert not bool(refused.committed)
    assert refused.state is lifted_state
    assert isinstance(refused.diagnostics, str)
    assert "declared-de-rham-layout-transfer-refused" in refused.diagnostics


def test_periodic_tetra_stage_rebuilds_a_relocated_vertex_orbit() -> None:
    source = _periodic_tetra_mesh()
    source_topology = source.periodic_topology
    assert source_topology is not None
    vertex_map = periodic_entity_orbit_map(source, 0)
    lifted_to_quotient = np.asarray(vertex_map.lifted_to_quotient, dtype=np.int64)
    orbit_counts = np.bincount(
        lifted_to_quotient, minlength=vertex_map.quotient_entity_ids.size
    )
    quotient_row = int(np.flatnonzero(orbit_counts > 1)[0])
    member_rows = np.flatnonzero(lifted_to_quotient == quotient_row)
    member_ids = np.asarray(vertex_map.lifted_entity_ids, dtype=np.int64)[member_rows]
    coordinate_row_by_id = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(source.vertex_global_ids, dtype=np.int64)
        )
    }
    target_coordinates = np.array(source.coordinates, dtype=np.float64, copy=True)
    target_rows = np.asarray(
        [coordinate_row_by_id[int(identifier)] for identifier in member_ids],
        dtype=np.int64,
    )
    target_coordinates[target_rows] += np.asarray((2.0**-10, -(2.0**-11), 2.0**-12))
    vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    blocks = tuple(
        TopologyEditBlock(
            block.name,
            block.cell_kind,
            block.cell_kind,
            np.asarray(block.vertices, dtype=np.int32),
            np.asarray(block.global_ids, dtype=np.int64),
        )
        for block in source.blocks
    )
    relations = []
    selected_vertex_ids = set(map(int, member_ids))
    for degree in range(source.topological_dimension + 1):
        keys = np.asarray(entity_keys(source, degree), dtype=np.int64)
        kinds = np.full((keys.shape[0],), EntityLineageKind.PRESERVED, dtype=np.int32)
        if degree == 0:
            kinds = np.asarray(
                [
                    EntityLineageKind.RELOCATED
                    if int(key[0]) in selected_vertex_ids
                    else EntityLineageKind.PRESERVED
                    for key in keys
                ],
                dtype=np.int32,
            )
        relations.append(EntityRelations(degree, keys, keys, kinds))
    edit = CellTopologyEdit(
        target_coordinates,
        vertex_ids,
        blocks,
        vertex_ids[:, None],
        np.ones((vertex_ids.size, 1), dtype=np.float64),
        np.ones((vertex_ids.size, 1), dtype=np.bool_),
        tuple(relations),
    )

    target = stage_periodic_topology_edit(source, edit)

    assert target is not None
    target_topology = target.periodic_topology
    assert target_topology is not None
    target_topology.require_lifted(
        target.topology, target.lifted_topology_id, target.blocks
    )
    target_topology.require_coordinates(target.coordinates)
    assert not np.array_equal(target.coordinates, source.coordinates)
    for degree in range(source.topological_dimension + 1):
        source_map = periodic_entity_orbit_map(source, degree)
        target_map = periodic_entity_orbit_map(target, degree)
        np.testing.assert_array_equal(
            target_map.quotient_entity_ids, source_map.quotient_entity_ids
        )
        np.testing.assert_array_equal(
            target_map.lifted_to_quotient, source_map.lifted_to_quotient
        )
        np.testing.assert_array_equal(
            target_map.winding_shifts, source_map.winding_shifts
        )

    state = tet_mesh_periodic_state(source, None)
    assert len(state.identity_map.entity_maps) == 4
    assert (
        len(state.quotient_cells)
        == periodic_entity_orbit_map(source, 3).quotient_entity_ids.size
    )
    operation_members = tet_mesh_periodic_operation_members(source, target, edit)
    assert {int(member.vertex_id) for member in operation_members} == set(
        int(identifier) for identifier in member_ids
    )
    native_edit = tet_mesh_periodic_edit(source, target, edit, operation_members)
    assert [int(member.vertex_id) for member in native_edit.periodic_members] == [
        int(member.vertex_id) for member in operation_members
    ]
    for actual, expected in zip(
        native_edit.periodic_members, operation_members, strict=True
    ):
        np.testing.assert_array_equal(actual.winding, expected.winding)
        np.testing.assert_array_equal(
            actual.representative_to_member, expected.representative_to_member
        )
    assert np.asarray(native_edit.removed_periodic_orbit_ids).size == 0
    assert native_edit.replacement_periodic_orbits == ()
    assert np.asarray(native_edit.removed_quotient_cell_ids).size == 0
    assert native_edit.replacement_quotient_cells == ()
    assert (
        native_edit.target_identity_map.periodic_topology_id
        == target_topology.periodic_topology_id
    )

    block = source.blocks[0]
    # The native complex declares every carrier boundary facet; the lifted
    # fundamental cover's open faces are its boundary (outward corner order).
    cell_vertices = np.asarray(block.vertices, dtype=np.int32)
    outward_corners = ((1, 3, 2), (0, 2, 3), (0, 3, 1), (0, 1, 2))
    oriented_faces = [
        cell_vertices[row, list(corners)]
        for row in range(cell_vertices.shape[0])
        for corners in outward_corners
    ]
    face_counts: dict[tuple[int, ...], int] = {}
    for face in oriented_faces:
        key = tuple(sorted(int(value) for value in face))
        face_counts[key] = face_counts.get(key, 0) + 1
    boundary_faces = np.asarray(
        [
            face
            for face in oriented_faces
            if face_counts[tuple(sorted(int(value) for value in face))] == 1
        ],
        dtype=np.int32,
    )
    with TetMesh3D(
        source.coordinates,
        block.vertices,
        np.zeros((block.cell_count,), dtype=np.int32),
        boundary_faces,
        np.zeros((boundary_faces.shape[0],), dtype=np.int32),
        np.empty((0, 2), dtype=np.int32),
        np.empty((0,), dtype=np.int32),
        boundary_policy=PlcBoundaryPolicy.CONFORMING,
        vertex_ids=source.vertex_global_ids,
        cell_ids=block.global_ids,
        source_id=source.mesh_id,
        source_revision=source.numeric_version,
        source_map_id=source.topology_id,
        frame_id="periodic-test-world-frame",
        periodic_state=state,
    ) as native:
        snapshot = native.arrays()
        changed_rows = np.asarray(
            [coordinate_row_by_id[int(identifier)] for identifier in member_ids],
            dtype=np.int64,
        )
        # A relocation names its star: every lifted cell incident to the orbit.
        star_ids = np.asarray(block.global_ids, dtype=np.int64)[
            np.isin(cell_vertices, changed_rows).any(axis=1)
        ]
        native_operation = TetMeshEdit(
            TetMeshEditKind.RELOCATE,
            native.generation,
            source_id=native.source_id,
            source_revision=native.source_revision,
            source_map_id=native.source_map_id,
            frame_id=native.frame_id,
            removed_cell_ids=star_ids,
            changed_vertex_ids=member_ids,
            changed_vertex_points=target_coordinates[changed_rows],
            periodic=native_edit,
        )
        assert native.apply_edit(native_operation) is TetMeshEditStatus.APPLIED
        moved = native.arrays()
        assert moved.generation == 1
        np.testing.assert_array_equal(
            moved.points[changed_rows], target_coordinates[changed_rows]
        )
        assert moved.periodic is not None
        assert (
            moved.periodic.identity_map.periodic_topology_id
            == target_topology.periodic_topology_id
        )
    assert snapshot.periodic is not None
    assert (
        snapshot.periodic.identity_map.periodic_topology_id
        == source_topology.periodic_topology_id
    )
    np.testing.assert_array_equal(
        snapshot.periodic.identity_map.entity_maps[3].quotient_entity_ids,
        state.identity_map.entity_maps[3].quotient_entity_ids,
    )


def test_periodic_tetra_distribution_keeps_owner_local_orbit_halos() -> None:
    mesh = _periodic_tetra_mesh(5)
    layout = _periodic_cell_layout(mesh)
    assert layout is not None
    distribution = prepare_mesh_distribution(
        _periodic_part(mesh),
        policy=MeshPartitionPolicy(
            MeshPartitionKind.GRAPH,
            2,
            halo_width=0,
            maximum_imbalance=2.0,
        ),
    )
    owner = np.asarray(distribution.partition.cell_owner, dtype=np.int64)
    cell_ids = np.asarray(distribution.cell_global_ids, dtype=np.int64)
    cell_orbits = np.asarray(layout.orbit_rows, dtype=np.int64)
    orbit_counts = np.bincount(cell_orbits, minlength=layout.entity_ids.size)
    orbit = int(np.flatnonzero(orbit_counts > 1)[0])
    orbit_cell_rows = np.flatnonzero(cell_orbits == orbit)
    assert np.unique(owner[orbit_cell_rows]).size == 1
    owner_rank = int(owner[orbit_cell_rows[0]])
    requester = 1 - owner_rank
    requests = [np.zeros((0,), dtype=np.int64) for _ in range(2)]
    requests[requester] = cell_ids[[int(orbit_cell_rows[0])]]

    expanded = distribution.expand_halos(tuple(requests))

    expected_ids = set(map(int, cell_ids[orbit_cell_rows]))
    assert expected_ids <= set(map(int, expanded.halo_global_ids[requester]))
    local = distribution.local_cell_geometry(owner_rank)
    assert expected_ids <= set(map(int, local["cell_ids"]))
    assert all(
        np.unique(owner[cell_orbits == quotient_row]).size == 1
        for quotient_row in range(layout.entity_ids.size)
    )
