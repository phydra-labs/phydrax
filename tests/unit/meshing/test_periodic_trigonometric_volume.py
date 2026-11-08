#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from functools import cache

import numpy as np
import pytest

import phydrax as phx
from phydrax._geometry_predicates import PredicateMode
from phydrax.discretization import PeriodicCell
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurfacePolicy,
    discover_periodic_trigonometric_surface,
)
from phydrax.meshing._lineage import EntityLineageKind
from phydrax.meshing._periodic import stage_periodic_topology_edit
from phydrax.meshing._periodic_implicit_volume import (
    mesh_periodic_trigonometric_region,
    PeriodicImplicitVolume,
)
from phydrax.meshing._tetra_metric import execute_tetra_metric_adaptation
from phydrax.meshing._topology_edit import entity_keys


_G = phx.geometry

type _Key = tuple[int, ...]


@cache
def _volume(family: _G.TrigonometricFamily, diameter: float) -> PeriodicImplicitVolume:
    source = _G.TrigonometricLevelSet(
        _G.TrigonometricTable.family(family), PeriodicCell(np.eye(3))
    )
    surface = discover_periodic_trigonometric_surface(
        source,
        policy=AdaptiveImplicitSurfacePolicy(
            relative_cell_diameter=diameter, root_tolerance=1.0e-8
        ),
        source_id=f"{family}-network",
    )
    return mesh_periodic_trigonometric_region(surface)


def _tetrahedra(mesh: CellMesh) -> np.ndarray:
    return np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )


def _quotient_boundary(mesh: CellMesh) -> list[tuple[_Key, ...]]:
    """Faces with one incident tetrahedron after identifying lattice translates.

    A face is keyed by its vertex orbits and their windings relative to the
    face's least winding, so every lattice translate of a face has one key.
    """
    topology = mesh.periodic_topology
    assert topology is not None
    representatives = np.asarray(topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    counts: dict[tuple[_Key, ...], int] = {}
    for cell in _tetrahedra(mesh).tolist():
        for omitted in range(4):
            face = [cell[index] for index in range(4) if index != omitted]
            anchor = shifts[face].min(axis=0)
            key = tuple(
                sorted(
                    (int(representatives[v]), *(shifts[v] - anchor).tolist())
                    for v in face
                )
            )
            counts[key] = counts.get(key, 0) + 1
    assert max(counts.values()) <= 2
    return [key for key, count in counts.items() if count == 1]


def _surface_census(faces: list[tuple[_Key, ...]]) -> tuple[int, int]:
    """Euler characteristic and component count of a quotient face set."""

    def edge(first: _Key, second: _Key, /) -> _Key:
        step = tuple(b - a for a, b in zip(first[1:], second[1:], strict=True))
        forward = (first[0], second[0], *step)
        backward = (second[0], first[0], *(-value for value in step))
        return min(forward, backward)

    vertices = {corner[0] for face in faces for corner in face}
    owners: dict[_Key, list[int]] = {}
    for row, face in enumerate(faces):
        for first, second in ((0, 1), (1, 2), (0, 2)):
            owners.setdefault(edge(face[first], face[second]), []).append(row)
    parent = list(range(len(faces)))

    def find(item: int, /) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for rows in owners.values():
        assert len(rows) == 2
        parent[find(rows[0])] = find(rows[1])
    components = len({find(row) for row in range(len(faces))})
    return len(vertices) - len(owners) + len(faces), components


def _cell_face_pairs(mesh: CellMesh, domain: np.ndarray, axis: int, /) -> int:
    """Require the upper cell-face triangulation to be the lower one's translate."""
    topology = mesh.periodic_topology
    assert topology is not None
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    representatives = np.asarray(topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    lower, upper = domain[:, axis]
    in_plane = [index for index in range(3) if index != axis]
    counts: dict[frozenset[int], int] = {}
    for cell in _tetrahedra(mesh).tolist():
        for omitted in range(4):
            face = frozenset(cell[index] for index in range(4) if index != omitted)
            counts[face] = counts.get(face, 0) + 1
    sides: dict[float, dict[frozenset[_Key], tuple[bytes, ...]]] = {
        lower: {},
        upper: {},
    }
    step = np.zeros(3, dtype=np.int64)
    step[axis] = 1
    for face, count in counts.items():
        rows = sorted(face)
        plane = coordinates[rows, axis]
        if count != 1 or not np.all(plane == plane[0]) or plane[0] not in sides:
            continue
        moved = step if plane[0] == lower else np.zeros(3, dtype=np.int64)
        key = frozenset(
            (int(representatives[row]), *(shifts[row] + moved).tolist()) for row in rows
        )
        # Translation along the axis leaves the in-plane coordinates bit-identical.
        sides[float(plane[0])][key] = tuple(
            sorted(coordinates[row, in_plane].tobytes() for row in rows)
        )
    assert sides[lower] and sides[lower] == sides[upper]
    return 2 * len(sides[lower])


@pytest.mark.parametrize(("family", "diameter"), (("schwarz_p", 0.2), ("gyroid", 0.1)))
def test_periodic_network_mesh_certifies_and_closes_through_paired_cell_faces(
    family: _G.TrigonometricFamily, diameter: float
) -> None:
    volume = _volume(family, diameter)
    surface = volume.surface
    assert surface.topology is not None

    assert volume.result.audit.passed
    assert volume.mesh.periodic_topology is not None
    # Every cap triangle on a lower cell face has its exact lattice translate
    # on the paired upper face, and no other boundary triangle lies there.
    assert (
        sum(
            _cell_face_pairs(volume.mesh, surface.reduced_domain, axis)
            for axis in range(3)
        )
        == volume.cap_triangle_count
    )
    # Paired caps become interior quotient faces: only the contour bounds the
    # quotient region, with the topology of the discovery's surface report.
    boundary = _quotient_boundary(volume.mesh)
    assert len(boundary) == surface.faces.shape[0]
    assert _surface_census(boundary) == (
        surface.topology.euler_characteristic,
        surface.topology.component_count,
    )
    assert volume.quotient_region_components == 1
    # c = 0 has a proved volume fraction of 1/2; the mesh measure lies within
    # the contour's discretization of it, and inside the certified cover range.
    enclosure = surface.cover.volume_enclosure()
    assert enclosure.fraction_lower <= volume.measure <= enclosure.fraction_upper
    assert volume.measure == pytest.approx(0.5, abs=0.02)
    # Shape targets on the fixed lattice contour are soft and reported.
    achieved = dict(volume.construction.compliance.achieved)
    assert achieved["soft_shape_target_unmet_count"] >= 0.0
    assert volume.construction.compliance.issues == ()


def test_periodic_network_public_adaptation_edits_complete_lattice_orbits() -> None:
    volume = _volume("schwarz_p", 0.2)
    source = volume.result
    mesh = source.mesh
    topology = mesh.periodic_topology
    assert topology is not None
    boundary = _quotient_boundary(mesh)
    contour_orbits = {corner[0] for face in boundary for corner in face}
    representatives = np.asarray(topology.vertex_representatives, dtype=np.int64)
    fixed = np.isin(representatives, np.asarray(sorted(contour_orbits)))
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    row_of = {int(identifier): row for row, identifier in enumerate(vertex_ids)}
    edges = np.asarray(entity_keys(mesh, 1), dtype=np.int64)
    protected = np.asarray(
        [fixed[row_of[int(a)]] and fixed[row_of[int(b)]] for a, b in edges.tolist()],
        dtype=np.bool_,
    )
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    rows = np.asarray(
        [[row_of[int(a)], row_of[int(b)]] for a, b in edges.tolist()], dtype=np.int64
    )
    lengths = np.linalg.norm(coordinates[rows[:, 1]] - coordinates[rows[:, 0]], axis=1)
    # A uniform isotropic metric is lattice-equivariant; it asks for edges
    # shorter than the mean, so adaptation splits and collapses complete edge
    # orbits, including seam orbits on the paired cell faces.
    size = 0.75 * float(np.mean(lengths))
    metric = np.broadcast_to(np.eye(3) / size**2, (vertex_ids.size, 3, 3)).copy()

    outcome = execute_tetra_metric_adaptation(
        source,
        metric,
        cell_classes=np.zeros(mesh.entity_set(3).entity_ids.size, dtype=np.int64),
        facet_classes=np.zeros(mesh.entity_set(2).entity_ids.size, dtype=np.int64),
        edge_classes=np.zeros(edges.shape[0], dtype=np.int64),
        protected_edges=protected,
        fixed_vertices=fixed,
        predicate_mode=PredicateMode.EXACT,
        topology_operations=True,
        relocation=False,
    )

    evidence = outcome.evidence
    assert evidence.splits > 0 and evidence.collapses > 0
    assert evidence.maximum_region_volume_error <= 1.0e-12
    target = stage_periodic_topology_edit(mesh, outcome.edit)
    assert target is not None and target.periodic_topology is not None
    target_topology = target.periodic_topology
    target_topology.require_lifted(
        target.topology, target.lifted_topology_id, target.blocks
    )
    target_topology.require_coordinates(target.coordinates)
    # Seam edits keep or remove whole vertex orbits: a source vertex orbit
    # never loses only some of its lifted copies on the paired cell faces.
    copies = np.bincount(representatives, minlength=vertex_ids.size)
    retained = np.isin(vertex_ids, np.asarray(target.vertex_global_ids))
    for root in np.unique(representatives).tolist():
        members = retained[representatives == root]
        assert members.all() or not members.any()
    # Moved collapse survivors are the only relocated vertices without the
    # relocation phase: a periodic survivor moves when its orbit is one copy.
    vertex_relation = outcome.edit.relations[0]
    relocated = {
        int(source_key[0])
        for source_key, kind in zip(
            np.asarray(vertex_relation.source_keys).tolist(),
            np.asarray(vertex_relation.kinds).tolist(),
            strict=True,
        )
        if kind == int(EntityLineageKind.RELOCATED)
    }
    assert relocated and all(copies[representatives[row_of[v]]] == 1 for v in relocated)
    # The upper cell-face triangulations stay the exact translates of the lower.
    domain = volume.surface.reduced_domain
    assert sum(_cell_face_pairs(target, domain, axis) for axis in range(3)) > 0
    # The fixed contour still alone bounds the quotient region, unchanged.
    target_boundary = _quotient_boundary(target)
    assert len(target_boundary) == len(boundary)
    assert _surface_census(target_boundary) == _surface_census(boundary)
    cells = np.asarray(target.coordinates, dtype=np.float64)[_tetrahedra(target)]
    measure = float(np.sum(np.abs(np.linalg.det(cells[:, 1:] - cells[:, :1])) / 6.0))
    assert measure == pytest.approx(volume.measure, rel=1.0e-12)
