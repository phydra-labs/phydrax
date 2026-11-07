"""Cold implicit target coverage against the canonical facet owners."""

import numpy as np
import pytest

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._approximate_source_coverage import (
    _target_face_chart_keys,
    _target_facets,
    _volume_face_owners,
    SourceCoverageUnavailable,
)
from phydrax.geometry._implicit_source_coverage import _SortedIntegerKeys
from phydrax.geometry._implicit_target_coverage import prove_implicit_target_coverage
from phydrax.geometry._source_boundary_records import SourceBoundaryChartWitness
from phydrax.geometry.implicit._adaptive_discovery import (
    _FieldEnclosures,
    AdaptiveImplicitScaffold,
)
from tests.unit.geometry.test_implicit_source_charts import _adaptive_source


Target = tuple[
    CellMesh,
    CellGeometrySpec,
    np.ndarray,
    SourceBoundaryChartWitness,
    _FieldEnclosures,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    _SortedIntegerKeys,
    AdaptiveImplicitScaffold,
]


@pytest.fixture(scope="module")
def target() -> Target:
    # Cone each actual root-scaffold triangle to the centre of its convex
    # superquadric. This is an exact conforming volume, unlike the native
    # remeshed carrier, which belongs to the separate chart-degree route.
    surface, _source, _contract, _surface_mesh, _surface_geometry, source_witness = (
        _adaptive_source()
    )
    assert surface.mesh is not None
    vertices = np.concatenate(
        (np.zeros((1, 3)), np.asarray(surface.mesh.vertices)), axis=0
    )
    source_faces = np.asarray(surface.mesh.faces, dtype=np.int64)
    cells = np.column_stack(
        (np.zeros(len(source_faces), dtype=np.int64), source_faces + 1)
    )
    mesh = CellMesh.from_tetrahedra(vertices, cells.astype(np.int32))
    geometry = CellGeometrySpec.affine(mesh)
    ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    reference = np.asarray(
        reference_cell_topology("tetrahedron").vertices, dtype=np.float64
    )
    target_vertices = np.broadcast_to(reference[[1, 2, 3]], (len(ids), 3, 3))
    witness = SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source_witness.source_id,
        source_revision=source_witness.source_revision,
        source_query_id=source_witness.source_query_id,
        chart_dimension=2,
        reference_dimension=3,
        cell_global_ids=ids,
        entity_global_ids=ids,
        local_entity_indices=np.zeros(len(ids), dtype=np.int32),
        target_reference_vertices=target_vertices,
        source_chart_ids=source_witness.source_chart_ids,
        source_entity_ids=source_witness.source_entity_ids,
        occurrence_paths=source_witness.occurrence_paths,
        source_parameters=source_witness.source_parameters,
        orientations=np.ones(len(ids), dtype=np.int8),
        implicit_cell_keys=source_witness.implicit_cell_keys,
        implicit_root_edge_keys=source_witness.implicit_root_edge_keys,
    )
    scaffold = surface.scaffold
    return (
        mesh,
        geometry,
        np.zeros(len(ids), dtype=np.int64),
        witness,
        _FieldEnclosures(surface.geometry, surface.policy.enclosure),
        np.asarray(surface.mesh.vertices, dtype=np.float64),
        np.asarray(surface.mesh.faces, dtype=np.int64),
        np.asarray(scaffold.triangle_cell_rows, dtype=np.int64),
        np.asarray(scaffold.triangle_root_edge_rows, dtype=np.int64),
        _SortedIntegerKeys(np.asarray(scaffold.root_edge_ids, dtype=np.int64)),
        scaffold,
    )


def test_target_matches_canonical_boundary_owners(target: Target) -> None:
    mesh, geometry, regions, witness, *_ = target
    facets, issues = _target_facets(mesh, geometry, regions)
    assert not issues
    owners, by_face, _ = _volume_face_owners(mesh, regions)
    required = {
        key for key, rows in owners.items() if len(rows) == 1 or rows[0][1] != rows[1][1]
    }
    charts = _target_face_chart_keys(mesh, geometry)
    assert len(facets) == len(required) == len(witness.cell_global_ids)
    assert {
        by_face[int(cell), int(face)]
        for cell, face in zip(
            witness.cell_global_ids, witness.local_entity_indices, strict=True
        )
    } == required
    assert {tuple(sorted(facet.vertices)) for facet in facets} == {
        charts[int(cell), int(face)]
        for cell, face in zip(
            witness.cell_global_ids, witness.local_entity_indices, strict=True
        )
    }
    prove_implicit_target_coverage(*target)


@pytest.mark.parametrize(
    "mutation", ("missing", "duplicate", "reversed", "changed", "ancestry")
)
def test_target_rejects_forged_witness(target: Target, mutation: str) -> None:
    witness = target[3]
    if mutation == "missing":
        name = "cell_global_ids"
        replacement = np.asarray(getattr(witness, name))[:-1]
    elif mutation == "duplicate":
        name = "cell_global_ids"
        replacement = np.array(getattr(witness, name), copy=True)
        replacement[1] = replacement[0]
        # Duplicate the complete face address, even when both faces share a cell.
        faces = np.array(witness.local_entity_indices, copy=True)
        faces[1] = faces[0]
    elif mutation == "reversed":
        name = "orientations"
        replacement = np.array(getattr(witness, name), copy=True)
        replacement[0] *= -1
    elif mutation == "changed":
        name = "cell_global_ids"
        replacement = np.array(getattr(witness, name), copy=True)
        replacement[0] = np.iinfo(np.int64).max
    else:
        name = "implicit_root_edge_keys"
        replacement = np.array(getattr(witness, name), copy=True)
        replacement[0, 0, 0, 0] += 1
    original = getattr(witness, name)
    original_faces = witness.local_entity_indices
    try:
        object.__setattr__(witness, name, replacement)
        if mutation == "duplicate":
            object.__setattr__(witness, "local_entity_indices", faces)
        with pytest.raises(SourceCoverageUnavailable) as error:
            prove_implicit_target_coverage(*target)
        assert error.value.status == "violated"
    finally:
        object.__setattr__(witness, name, original)
        object.__setattr__(witness, "local_entity_indices", original_faces)


def test_target_rejects_reversed_source_face(target: Target) -> None:
    faces = np.array(target[6], copy=True)
    faces[0, [1, 2]] = faces[0, [2, 1]]
    args = (*target[:6], faces, *target[7:])
    with pytest.raises(SourceCoverageUnavailable, match="winding reverses"):
        prove_implicit_target_coverage(*args)


def test_target_rejects_changed_facet_coordinates(target: Target) -> None:
    mesh, geometry, _regions, witness, *_ = target
    cell_id = int(witness.cell_global_ids[0])
    local_face = int(witness.local_entity_indices[0])
    elements, routes, _ = geometry.resolve(mesh)
    for block, _element, route in zip(mesh.blocks, elements, routes, strict=True):
        matches = np.flatnonzero(np.asarray(block.global_ids) == cell_id)
        if matches.size:
            slot = reference_cell_topology(block.cell_kind).entities[2][local_face][0]
            coordinate_row = int(np.asarray(route)[int(matches[0]), int(slot)])
            break
    else:
        raise AssertionError("Witness target cell is missing.")
    coordinates = geometry.coordinates
    changed = np.array(coordinates, dtype=np.float64, copy=True)
    changed[coordinate_row, 0] += 0.125
    try:
        object.__setattr__(geometry, "coordinates", changed)
        with pytest.raises(SourceCoverageUnavailable):
            prove_implicit_target_coverage(*target)
    finally:
        object.__setattr__(geometry, "coordinates", coordinates)
