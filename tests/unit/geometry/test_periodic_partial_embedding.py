from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from itertools import permutations, product

import numpy as np

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import (
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_exact import PeriodicIsometryGroup
from phydrax.discretization._periodic_topology import PeriodicMeshTopology
from phydrax.geometry._global_embedding import certify_global_embedding


type GridPoint = tuple[int, ...]


def _quotient_mesh(
    coordinates: Sequence[GridPoint],
    blocks: tuple[tuple[str, str, Sequence[tuple[int, ...]]], ...],
    identification: PeriodicCell | PeriodicIsometryGroup,
    representatives: Sequence[int],
    shifts: Sequence[tuple[int, ...]],
) -> CellMesh:
    points = np.asarray(coordinates, dtype=np.float64)
    vertex_ids = np.arange(len(coordinates), dtype=np.int64)
    cell_blocks = []
    first_cell_id = 100
    for name, kind, rows in blocks:
        cell_blocks.append(
            CellBlock(
                name,
                kind,
                np.asarray(rows, dtype=np.int32),
                global_ids=np.arange(
                    first_cell_id, first_cell_id + len(rows), dtype=np.int64
                ),
            )
        )
        first_cell_id += len(rows)
    lifted = CellMesh(points, tuple(cell_blocks), vertex_global_ids=vertex_ids)
    topology = PeriodicMeshTopology(
        lifted,
        identification,
        np.asarray(representatives, dtype=np.int32),
        np.asarray(shifts, dtype=np.int32),
    )
    return CellMesh(
        points,
        tuple(cell_blocks),
        vertex_global_ids=vertex_ids,
        periodic_topology=topology,
    )


def _periodic_quotient(
    coordinates: Sequence[GridPoint],
    blocks: tuple[tuple[str, str, Sequence[tuple[int, ...]]], ...],
    periodic_axes: tuple[int, ...],
    combinatorial_period: int,
    winding_multiplier: int,
) -> CellMesh:
    """Build a lifted mesh whose periodic axes identify grid layers exactly.

    A grid point ``p`` on a periodic axis has root coordinate
    ``p % combinatorial_period`` and winding
    ``winding_multiplier * (p // combinatorial_period)``; the geometric lattice
    period is ``combinatorial_period / winding_multiplier``. A multiplier of
    one is a degree-one quotient; two wraps the quotient twice.
    """
    dimension = len(coordinates[0])
    index = {point: row for row, point in enumerate(coordinates)}
    representatives = []
    shifts = []
    for point in coordinates:
        root = list(point)
        winding = []
        for axis in periodic_axes:
            root[axis] = point[axis] % combinatorial_period
            winding.append(winding_multiplier * (point[axis] // combinatorial_period))
        representatives.append(index[tuple(root)])
        shifts.append(tuple(winding))
    lattice = np.zeros((len(periodic_axes), dimension), dtype=np.float64)
    for row, axis in enumerate(periodic_axes):
        lattice[row, axis] = combinatorial_period / winding_multiplier
    return _quotient_mesh(
        coordinates, blocks, PeriodicCell(lattice), representatives, shifts
    )


def _kuhn_tetrahedra(
    cube: GridPoint, index: dict[GridPoint, int]
) -> list[tuple[int, int, int, int]]:
    rows = []
    for permutation in permutations(range(3)):
        path = [cube]
        current = list(cube)
        for axis in permutation:
            current[axis] += 1
            path.append(tuple(current))
        row = [index[point] for point in path]
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(3)
            for right in range(left + 1, 3)
        )
        if inversions % 2:
            row[1], row[2] = row[2], row[1]
        rows.append((row[0], row[1], row[2], row[3]))
    return rows


def _cube_grid(
    material_cubes: tuple[tuple[str, tuple[GridPoint, ...]], ...],
) -> tuple[
    list[GridPoint],
    tuple[tuple[str, str, Sequence[tuple[int, ...]]], ...],
]:
    coordinates = sorted(
        {
            tuple(cube[axis] + corner[axis] for axis in range(3))
            for _, cubes in material_cubes
            for cube in cubes
            for corner in product((0, 1), repeat=3)
        }
    )
    index = {point: row for row, point in enumerate(coordinates)}
    blocks = tuple(
        (
            name,
            "tetrahedron",
            [row for cube in cubes for row in _kuhn_tetrahedra(cube, index)],
        )
        for name, cubes in material_cubes
    )
    return coordinates, blocks


def _tetrahedral_quotient(
    material_cubes: tuple[tuple[str, tuple[GridPoint, ...]], ...],
    periodic_axes: tuple[int, ...],
    *,
    winding_multiplier: int = 1,
) -> tuple[CellMesh, CellGeometrySpec]:
    coordinates, blocks = _cube_grid(material_cubes)
    mesh = _periodic_quotient(coordinates, blocks, periodic_axes, 2, winding_multiplier)
    return mesh, CellGeometrySpec.affine(mesh)


def _material_l_channel(
    *, winding_multiplier: int = 1
) -> tuple[CellMesh, CellGeometrySpec]:
    """Two-material L cross-section in xy, periodic along z with two layers."""
    foot = tuple((i, 0, k) for i in range(2) for k in range(2))
    leg = tuple((0, 1, k) for k in range(2))
    return _tetrahedral_quotient(
        (("material-foot", foot), ("material-leg", leg)),
        (2,),
        winding_multiplier=winding_multiplier,
    )


def _quarter_turn() -> PeriodicIsometryGroup:
    rotation = np.eye(4, dtype=np.float64)
    rotation[:2, :2] = ((0.0, -1.0), (1.0, 0.0))
    return PeriodicIsometryGroup(rotation[None, :, :])


def _rotational_wedge(*, doubled: bool = False) -> tuple[CellMesh, CellGeometrySpec]:
    """Wedge of a square frame about the z axis under the quarter-turn group.

    The quarter wedge glues its ``y = 0`` cut to its ``x = 0`` cut by one quarter
    turn. The doubled wedge is the quarter wedge joined to its own quarter-turn
    image (so both cuts stay conforming) and glues its two remaining cuts by two
    quarter turns, so the order-four group wraps it twice around the axis.
    """
    coordinates, blocks = _cube_grid(
        (("wedge", tuple((i, j, 0) for i, j in ((1, 0), (1, 1), (0, 1)))),)
    )
    if doubled:
        ((name, kind, rows),) = blocks
        rotated = [(-point[1], point[0], point[2]) for point in coordinates]
        merged = sorted(set(coordinates) | set(rotated))
        merged_index = {point: row for row, point in enumerate(merged)}
        doubled_rows = [
            tuple(merged_index[bank[vertex]] for vertex in row)
            for bank in (coordinates, rotated)
            for row in rows
        ]
        coordinates, blocks = merged, ((name, kind, doubled_rows),)
    index = {point: row for row, point in enumerate(coordinates)}
    representatives = []
    shifts = []
    for x, y, z in coordinates:
        if doubled and y == 0 and x < 0:
            representatives.append(index[(-x, 0, z)])
            shifts.append((2,))
        elif not doubled and x == 0:
            representatives.append(index[(y, 0, z)])
            shifts.append((1,))
        else:
            representatives.append(index[(x, y, z)])
            shifts.append((0,))
    mesh = _quotient_mesh(coordinates, blocks, _quarter_turn(), representatives, shifts)
    return mesh, CellGeometrySpec.affine(mesh)


def _tetrahedron_volume(points: np.ndarray, /) -> Fraction:
    exact = tuple(tuple(Fraction(float(value)) for value in point) for point in points)
    edges = tuple(
        tuple(exact[row][axis] - exact[0][axis] for axis in range(3)) for row in (1, 2, 3)
    )
    return (
        abs(
            edges[0][0] * (edges[1][1] * edges[2][2] - edges[1][2] * edges[2][1])
            - edges[0][1] * (edges[1][0] * edges[2][2] - edges[1][2] * edges[2][0])
            + edges[0][2] * (edges[1][0] * edges[2][1] - edges[1][1] * edges[2][0])
        )
        / 6
    )


def test_z_periodic_material_l_channel_certifies_partial_quotient_embedding() -> None:
    mesh, geometry = _material_l_channel()
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)

    topology = mesh.periodic_topology
    assert topology is not None
    assert topology.cell.rank == 1
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    independent_volume = sum(
        (
            _tetrahedron_volume(coordinates[list(row)])
            for block in mesh.blocks
            for row in np.asarray(block.vertices, dtype=np.int64).tolist()
        ),
        Fraction(0),
    )

    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings
    assert {
        "periodic_exact_cell_contacts",
        "periodic_subdomain_boundary",
    } <= set(certificate.evaluated_checks)
    # Three unit squares of L cross-section over one z period of length two.
    assert independent_volume == Fraction(6)


def test_doubly_wrapped_material_l_channel_is_refused() -> None:
    mesh, geometry = _material_l_channel(winding_multiplier=2)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "violated"
    assert any(
        finding.check == "periodic_cell_contact" and finding.status == "violated"
        for finding in certificate.findings
    )


def test_folded_material_l_channel_is_invalid() -> None:
    mesh, _ = _material_l_channel()
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64).copy()
    topology = mesh.periodic_topology
    assert topology is not None
    representatives = np.asarray(topology.vertex_representatives, dtype=np.int64)
    folded_root = next(
        row for row, point in enumerate(coordinates.tolist()) if point == [1.0, 1.0, 1.0]
    )
    for row in np.flatnonzero(representatives == folded_root):
        coordinates[row, 0] += 1.5
    lifted = CellMesh(coordinates, mesh.blocks, vertex_global_ids=mesh.vertex_global_ids)
    folded_topology = PeriodicMeshTopology(
        lifted,
        topology.cell,
        topology.vertex_representatives,
        topology.vertex_shifts,
    )
    folded = CellMesh(
        coordinates,
        mesh.blocks,
        vertex_global_ids=mesh.vertex_global_ids,
        periodic_topology=folded_topology,
    )
    folded_geometry = CellGeometrySpec.affine(folded)
    validity = certify_cell_geometry_validity(folded_geometry, mesh=folded)
    certificate = certify_global_embedding(folded, folded_geometry, validity)

    assert np.any(np.asarray(validity.status) == CellValidityStatus.INVALID)
    assert certificate.status == "violated"
    assert any(
        finding.check == "cell_validity" and finding.status == "violated"
        for finding in certificate.findings
    )


def test_xy_periodic_slab_certifies_rank_two_partial_quotient() -> None:
    cubes = tuple((i, j, 0) for i in range(2) for j in range(2))
    mesh, geometry = _tetrahedral_quotient((("slab", cubes),), (0, 1))
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)

    topology = mesh.periodic_topology
    assert topology is not None
    assert topology.cell.rank == 2
    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings
    assert "periodic_subdomain_boundary" in certificate.evaluated_checks


def _x_periodic_strip(
    *, winding_multiplier: int = 1
) -> tuple[CellMesh, CellGeometrySpec]:
    coordinates = [(i, j) for i in range(3) for j in range(2)]
    index = {point: row for row, point in enumerate(coordinates)}
    triangles = []
    for i in range(2):
        lower_left = index[(i, 0)]
        lower_right = index[(i + 1, 0)]
        upper_right = index[(i + 1, 1)]
        upper_left = index[(i, 1)]
        triangles.extend(
            (
                (lower_left, lower_right, upper_right),
                (lower_left, upper_right, upper_left),
            )
        )
    mesh = _periodic_quotient(
        coordinates,
        (("strip", "triangle", triangles),),
        (0,),
        2,
        winding_multiplier,
    )
    return mesh, CellGeometrySpec.affine(mesh)


def test_x_periodic_triangle_strip_certifies_rank_one_quotient() -> None:
    mesh, geometry = _x_periodic_strip()
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings
    assert {
        "periodic_exact_cell_contacts",
        "periodic_subdomain_boundary",
    } <= set(certificate.evaluated_checks)


def test_doubly_wrapped_triangle_strip_is_refused() -> None:
    mesh, geometry = _x_periodic_strip(winding_multiplier=2)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "violated"
    assert any(
        finding.check == "periodic_cell_contact" and finding.status == "violated"
        for finding in certificate.findings
    )


def test_rotational_wedge_certifies_finite_group_quotient() -> None:
    mesh, geometry = _rotational_wedge()
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)

    topology = mesh.periodic_topology
    assert topology is not None
    assert isinstance(topology.cell, PeriodicIsometryGroup)
    assert topology.cell.orders == (4,)
    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings
    assert {
        "periodic_exact_cell_contacts",
        "periodic_subdomain_boundary",
    } <= set(certificate.evaluated_checks)


def test_doubly_wrapped_rotational_wedge_is_refused() -> None:
    mesh, geometry = _rotational_wedge(doubled=True)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "violated"
    assert any(
        finding.check == "periodic_cell_contact" and finding.status == "violated"
        for finding in certificate.findings
    )


def test_rotational_pie_wedge_with_fixed_axis_certifies() -> None:
    # One unit cube on the rotation axis: its x = 0 cut is the quarter-turn
    # image of its y = 0 cut, and the axis vertices are fixed by the group.
    coordinates, blocks = _cube_grid((("pie", ((0, 0, 0),)),))
    index = {point: row for row, point in enumerate(coordinates)}
    representatives = []
    shifts = []
    for x, y, z in coordinates:
        if x == 0 and y > 0:
            representatives.append(index[(y, 0, z)])
            shifts.append((1,))
        else:
            representatives.append(index[(x, y, z)])
            shifts.append((0,))
    mesh = _quotient_mesh(coordinates, blocks, _quarter_turn(), representatives, shifts)
    geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)

    topology = mesh.periodic_topology
    assert topology is not None
    assert np.any(np.asarray(topology.vertex_fixed_generators, dtype=np.bool_))
    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings
    assert "periodic_subdomain_boundary" in certificate.evaluated_checks


def _has_loop_edge(mesh: CellMesh, /) -> bool:
    topology = mesh.periodic_topology
    assert topology is not None
    incidence = topology.quotient.incidences[0]
    valid = np.asarray(incidence.relation.valid, dtype=np.bool_)
    vertices = np.asarray(incidence.relation.source_indices, dtype=np.int64)[valid]
    edges = np.asarray(incidence.relation.target_indices, dtype=np.int64)[valid]
    return any(np.unique(vertices[edges == edge]).size == 1 for edge in np.unique(edges))


def test_single_layer_z_periodic_material_l_certifies_loop_edge_quotient() -> None:
    # One cell layer spans the whole z period, so every vertical edge is a
    # quotient loop and every vertical boundary face repeats a vertex orbit.
    foot = tuple((i, 0, 0) for i in range(2))
    leg = ((0, 1, 0),)
    coordinates, blocks = _cube_grid((("material-foot", foot), ("material-leg", leg)))
    mesh = _periodic_quotient(coordinates, blocks, (2,), 1, 1)
    geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)

    assert _has_loop_edge(mesh)
    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings
    assert {
        "periodic_exact_cell_contacts",
        "periodic_subdomain_boundary",
    } <= set(certificate.evaluated_checks)


def test_single_column_x_periodic_strip_certifies_boundary_loops() -> None:
    # Each boundary component of the annulus is one loop edge at one vertex.
    coordinates = [(i, j) for i in range(2) for j in range(2)]
    index = {point: row for row, point in enumerate(coordinates)}
    triangles = [
        (index[(0, 0)], index[(1, 0)], index[(1, 1)]),
        (index[(0, 0)], index[(1, 1)], index[(0, 1)]),
    ]
    mesh = _periodic_quotient(
        coordinates, (("strip", "triangle", triangles),), (0,), 1, 1
    )
    geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)

    assert _has_loop_edge(mesh)
    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings
    assert "periodic_subdomain_boundary" in certificate.evaluated_checks


def test_edge_pinched_single_layer_quotient_is_unresolved() -> None:
    # Two columns meet only along one vertical loop edge: its boundary vertex
    # link is two cycles, so the quotient boundary is not a manifold.
    coordinates, blocks = _cube_grid((("pinched", ((0, 0, 0), (1, 1, 0))),))
    mesh = _periodic_quotient(coordinates, blocks, (2,), 1, 1)
    geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status != "certified"
    assert any(
        finding.check == "periodic_quotient_incidence_unresolved"
        for finding in certificate.findings
    )
