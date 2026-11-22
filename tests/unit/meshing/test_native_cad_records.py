# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native CAD solid records and the consumers of their associations."""

from __future__ import annotations

import re
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax._array_archive import ArrayArchiveCorruptionError
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry.brep import brep_box, brep_entity_id
from phydrax.meshing._adaptation import (
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationResult,
    MeshAdaptationRoute,
    prepare_mesh_adaptation,
)
from phydrax.meshing._association import GeometryAssociation, GeometryAssociationKind
from phydrax.meshing._native_cad_records import NativeCadRecords
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing.providers._native_plc_geometry import simplex_volume_entities
from tests.unit.meshing.test_native_cad_mesh import _cad_volume_plan, _sphere_model


_CONTRACT = phx.SpatialCoordinateContract.si()


def _box() -> phx.geometry.brep.BRepModel:
    return brep_box(
        np.zeros(3),
        np.ones(3),
        coordinate_contract=_CONTRACT,
        source_id="test:native-cad-records-box",
    )


def _native(model: phx.geometry.brep.BRepModel, size: float, /) -> CellMeshingResult:
    result = _cad_volume_plan(model, size, 1, radius_edge=8.0, dihedral=0.0).execute(
        prove=True
    )
    assert isinstance(result, CellMeshingResult)
    assert result.mesh.blocks[0].name == "native-cad-tetrahedra"
    return result


def _records(result: CellMeshingResult, /) -> NativeCadRecords:
    assert result.certification is not None
    records = result.certification.inputs.native_cad_records
    assert records is not None
    return records


def _table(result: CellMeshingResult, dimension: int, /) -> GeometryAssociation:
    entity_set = result.mesh.entity_set(dimension).entity_set_id
    (table,) = (
        item
        for item in result.associations
        if item.association_kind is GeometryAssociationKind.BREP
        and item.target_entity_set_id == entity_set
    )
    return table


def _curve(result: CellMeshingResult, /) -> GeometryAssociation:
    (curve,) = (
        item
        for item in result.associations
        if item.association_kind is GeometryAssociationKind.CURVE
    )
    return curve


def _transfer(
    model: phx.geometry.brep.BRepModel, /
) -> phx.meshing.BRepAssociationTransfer:
    return phx.meshing.BRepAssociationTransfer(
        phx.geometry.prepare_brep_projection(model)
    )


@pytest.fixture(scope="module")
def box() -> tuple[phx.geometry.brep.BRepModel, CellMeshingResult]:
    model = _box()
    # Small enough that both routes place interior vertices (stratum 3).
    return model, _native(model, 0.3)


def test_native_box_records_classify_every_entity(
    box: tuple[phx.geometry.brep.BRepModel, CellMeshingResult],
) -> None:
    # Reference: the unit box's closed-form strata (a point's stratum is
    # three minus the number of coordinates on {0, 1}) and outward normals.
    # Faults caught: a misrouted model row (curve row as domain curve), an
    # inverted face orientation parity, a lost region side, a curve interval
    # in the wrong gauge.
    _, result = box
    mesh = result.mesh
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    on_plane = np.isclose(points, 0.0, atol=1e-12) | np.isclose(points, 1.0, atol=1e-12)
    vertices = _table(result, 0)
    expected = 3 - np.count_nonzero(on_plane, axis=1)
    np.testing.assert_array_equal(np.asarray(vertices.source_dimensions), expected)
    curve_rows = np.flatnonzero(expected == 1)
    parameters = np.asarray(vertices.parameters, dtype=np.float64)[curve_rows, 0]
    free = points[curve_rows][~on_plane[curve_rows]]
    assert np.all(
        np.isclose(parameters, free, atol=1e-12)
        | np.isclose(parameters, 1.0 - free, atol=1e-12)
    )
    assert (
        len({vertices.source_entity_ids[row] for row in np.flatnonzero(expected == 0)})
        == 8
    )
    assert len({vertices.source_entity_ids[row] for row in curve_rows}) == 12

    _, faces = simplex_volume_entities(mesh)
    table = _table(result, 2)
    boundary = np.flatnonzero(np.asarray(table.source_dimensions) == 2)
    assert len({table.source_entity_ids[row] for row in boundary}) == 6
    corners = points[faces[boundary]]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    outward = corners.mean(axis=1) - 0.5
    signs = np.sign(np.einsum("ij,ij->i", normals, outward))
    np.testing.assert_array_equal(np.asarray(table.orientations)[boundary], signs)
    region = MeshingDomain.from_brep(box[0]).entity_id(3, 0)
    assert {table.source_region_sides[row] for row in boundary} == {(None, region)}
    interior = np.flatnonzero(np.asarray(table.source_dimensions) == 3)
    assert np.all(np.asarray(table.orientations)[interior] == 0)
    assert len(_table(result, 3).source_entity_ids) == sum(
        block.cell_count for block in mesh.blocks
    )

    records = _records(result)
    curve = _curve(result)
    records.require_curve(curve)
    assert curve.parent_association_id == records.root_id
    by_entity: dict[str, list[tuple[Fraction, Fraction]]] = {}
    for entity, interval in zip(
        curve.source_entity_ids, curve.source_parameter_intervals, strict=True
    ):
        assert interval is not None
        by_entity.setdefault(entity, []).append(interval)
    assert len(by_entity) == 12
    for intervals in by_entity.values():
        intervals.sort()
        assert intervals[0][0] == 0 and intervals[-1][1] == 1
        assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))
    # The edge table names the curve rows' source edges with their orientation.
    edges = _table(result, 1)
    rows = {
        int(value): row for row, value in enumerate(np.asarray(edges.target_global_ids))
    }
    for row, target in enumerate(np.asarray(curve.target_global_ids).tolist()):
        assert edges.source_entity_ids[rows[target]] == curve.source_entity_ids[row]
        assert int(edges.orientations[rows[target]]) == int(curve.orientations[row])


def test_native_box_occurrences_name_the_existing_route_entities(
    box: tuple[phx.geometry.brep.BRepModel, CellMeshingResult],
) -> None:
    # Reference: the existing CAD volume route (an independent implementation
    # of the same B-Rep occurrence naming) on the same model. Fault caught: a
    # native record keyed by model row instead of the B-Rep occurrence.
    from phydrax.meshing._native_construction import NativeMeshConstruction
    from phydrax.meshing.providers._native import _PreparedDirectBRepCadVolume

    model, native = box
    prepared = _cad_volume_plan(model, 0.3, 1, radius_edge=8.0, dihedral=0.0).prepared
    assert isinstance(prepared, _PreparedDirectBRepCadVolume)
    existing = prepared.execute()
    assert isinstance(existing, NativeMeshConstruction)
    assert existing.mesh.blocks[0].name != "native-cad-tetrahedra"
    associations = existing.organization.associations
    for dimension in range(4):
        entity_set = existing.mesh.entity_set(dimension).entity_set_id
        ours = _table(native, dimension)
        (theirs,) = (
            item
            for item in associations
            if item.association_kind is GeometryAssociationKind.BREP
            and item.target_entity_set_id == entity_set
        )
        for dimension_value in range(4):
            assert {
                (entity, path)
                for entity, path, stratum in zip(
                    ours.source_entity_ids,
                    ours.source_occurrence_paths,
                    np.asarray(ours.source_dimensions).tolist(),
                    strict=True,
                )
                if stratum == dimension_value
            } == {
                (entity, path)
                for entity, path, stratum in zip(
                    theirs.source_entity_ids,
                    theirs.source_occurrence_paths,
                    np.asarray(theirs.source_dimensions).tolist(),
                    strict=True,
                )
                if stratum == dimension_value
            }
    (curve,) = (
        item
        for item in associations
        if item.association_kind is GeometryAssociationKind.CURVE
    )
    assert set(_curve(native).source_entity_ids) == set(curve.source_entity_ids)


def test_native_sphere_records_pair_pole_slivers() -> None:
    # Reference: the sphere's poles (0, 0, +-r). Fault caught: a sliver pair
    # keyed by chart row instead of its {P, Q} mesh vertices.
    result = _native(_sphere_model(), 0.3)
    records = _records(result)
    pairs = np.asarray(records.sliver_vertices)
    assert pairs.size and np.all(pairs[:, 0] < pairs[:, 1])
    points = np.asarray(result.mesh.coordinates, dtype=np.float64)
    rows = {
        int(value): row
        for row, value in enumerate(np.asarray(result.mesh.vertex_global_ids).tolist())
    }
    poles = set(
        np.flatnonzero(
            np.all(np.isclose(points[:, :2], 0.0, atol=1e-12), axis=1)
            & np.isclose(np.abs(points[:, 2]), 0.5)
        ).tolist()
    )
    assert len(poles) == 2
    assert all(
        len({rows[int(value)] for value in pair} & poles) == 1 for pair in pairs.tolist()
    )


@pytest.mark.parametrize(
    "forgery",
    ("absent", "missing-child", "wrong-parent", "wrong-target", "swapped-target"),
)
def test_native_curve_association_transfer_rejects_forgeries(
    box: tuple[phx.geometry.brep.BRepModel, CellMeshingResult], forgery: str
) -> None:
    # Fault caught: a transfer that accepts a curve association its native
    # records do not certify.
    model, source = box
    transfer = _transfer(model)
    transfer.validate_source(source)
    foreign = brep_box(
        np.zeros(3),
        np.ones(3),
        coordinate_contract=_CONTRACT,
        source_id="test:native-cad-records-foreign",
    )
    with pytest.raises(ValueError, match="vertex association"):
        _transfer(foreign).validate_source(source)
    curve = _curve(source)
    index = source.associations.index(curve)
    if forgery == "absent":
        forged = source.associations[:index] + source.associations[index + 1 :]
    else:
        rows = np.arange(len(curve.target_global_ids), dtype=np.int64)
        if forgery == "missing-child":
            rows = rows[1:]
        parent_ids = np.asarray(curve.parent_ids)[rows].copy()
        if forgery == "wrong-parent":
            parent_ids[0] = next(value for value in parent_ids if value != parent_ids[0])
        target_ids = np.asarray(curve.target_global_ids)[rows].copy()
        if forgery == "wrong-target":
            target_ids[0] = next(
                identifier
                for identifier in np.asarray(source.mesh.entity_set(1).entity_ids)
                if identifier not in target_ids
            )
        if forgery == "swapped-target":
            first, second = next(
                (i, j)
                for i in range(len(target_ids))
                for j in range(i + 1, len(target_ids))
                if curve.source_entity_ids[i] == curve.source_entity_ids[j]
            )
            target_ids[first], target_ids[second] = target_ids[second], target_ids[first]

        def selected[T](values: tuple[T, ...]) -> tuple[T, ...]:
            return tuple(values[int(row)] for row in rows)

        replacement = GeometryAssociation(
            curve.association_kind,
            curve.source_id,
            curve.source_revision,
            curve.target_entity_set_id,
            target_ids,
            selected(curve.source_entity_ids),
            np.asarray(curve.residuals)[rows],
            exact=curve.exact,
            source_dimensions=np.asarray(curve.source_dimensions)[rows],
            source_indices=np.asarray(curve.source_indices)[rows],
            source_roles=selected(curve.source_roles),
            source_occurrence_paths=selected(curve.source_occurrence_paths),
            source_parameter_intervals=selected(curve.source_parameter_intervals),
            orientations=np.asarray(curve.orientations)[rows],
            parent_dimensions=np.asarray(curve.parent_dimensions)[rows],
            parent_ids=parent_ids,
            parent_association_id=curve.parent_association_id,
            provenance=curve.provenance,
        )
        forged = (
            source.associations[:index]
            + (replacement,)
            + source.associations[index + 1 :]
        )
    stale = eqx.tree_at(lambda value: value.associations, source, forged)
    with pytest.raises(ValueError):
        transfer.validate_source(stale)


def _rebuild_brep_table(
    table: GeometryAssociation, /, *, change: str
) -> GeometryAssociation:
    """Author a fresh table, giving edited source rows canonical entity keys."""
    dimensions = np.asarray(table.source_dimensions)
    indices = np.asarray(table.source_indices).copy()
    paths = list(table.source_occurrence_paths)
    sides = list(table.source_region_sides)
    residuals = np.asarray(table.residuals).copy()
    rows = np.arange(len(indices), dtype=np.int64)
    source_id = table.source_id
    match change:
        case "unchanged":
            pass
        case "face-entity":
            first = int(np.flatnonzero(dimensions == 2)[0])
            other = int(
                np.flatnonzero((dimensions == 2) & (indices != indices[first]))[0]
            )
            assert paths[first] == paths[other]
            indices[first] = indices[other]
        case "face-occurrence":
            first = int(np.flatnonzero(dimensions == 2)[0])
            paths[first] = (*paths[first], "another-occurrence")
        case "face-region-side":
            first = next(row for row, (minus, plus) in enumerate(sides) if minus != plus)
            minus, plus = sides[first]
            sides[first] = (plus, minus)
        case "face-residual":
            first = int(np.flatnonzero(dimensions == 2)[0])
            residuals[first] = 0.0
        case "cell-occurrence":
            paths[0] = (*paths[0], "another-occurrence")
        case "missing-row":
            first = int(np.flatnonzero(dimensions == 2)[0])
            rows = rows[rows != first]
        case "foreign-table":
            source_id = "test:native-cad-records-foreign"
        case _:
            raise ValueError(f"Unknown B-Rep table test change: {change}")

    def selected[T](values: Sequence[T]) -> tuple[T, ...]:
        return tuple(values[int(row)] for row in rows)

    return GeometryAssociation(
        table.association_kind,
        source_id,
        table.source_revision,
        table.target_entity_set_id,
        np.asarray(table.target_global_ids)[rows],
        tuple(
            brep_entity_id(
                source_id,
                table.source_revision,
                int(dimensions[row]),
                int(indices[row]),
                occurrence_path=paths[int(row)],
            )
            for row in rows
        ),
        residuals[rows],
        resolved=np.asarray(table.resolved)[rows],
        ambiguous=np.asarray(table.ambiguous)[rows],
        exact=table.exact,
        source_dimensions=dimensions[rows],
        source_indices=indices[rows],
        source_roles=selected(table.source_roles),
        source_occurrence_paths=selected(paths),
        source_region_sides=selected(sides),
        source_parameter_intervals=selected(table.source_parameter_intervals),
        target_parameter_intervals=selected(table.target_parameter_intervals),
        target_reference_pieces=selected(table.target_reference_pieces),
        source_boundary_witness_id=table.source_boundary_witness_id,
        source_parameter_points=selected(table.source_parameter_points),
        parameters=np.asarray(table.parameters)[rows],
        orientations=np.asarray(table.orientations)[rows],
        parent_dimensions=np.asarray(table.parent_dimensions)[rows],
        parent_ids=np.asarray(table.parent_ids)[rows],
        parent_association_id=table.parent_association_id,
        provenance=table.provenance,
    )


def _cad_association_input(
    source: CellMeshingResult, change: str, /
) -> tuple[GeometryAssociation, ...]:
    dimension = 3 if change == "cell-occurrence" else 2
    table = _table(source, dimension)
    if change == "missing-namespace":
        return tuple(
            item
            for item in source.associations
            if (item.source_id, item.source_revision)
            != (table.source_id, table.source_revision)
        )
    if change == "missing-table":
        return tuple(item for item in source.associations if item is not table)
    replacement = _rebuild_brep_table(table, change=change)
    replacement.validate_target(source.mesh.entity_set(dimension))
    assert replacement is not table
    # GeometryAssociation's general identity does not encode residuals.
    assert (replacement.association_id == table.association_id) == (
        change in ("unchanged", "face-residual")
    )
    return tuple(replacement if item is table else item for item in source.associations)


def _recertify(
    source: CellMeshingResult, associations: tuple[GeometryAssociation, ...], /
) -> CellMeshingResult:
    certification = source.certification
    if certification is None:
        raise ValueError("The native CAD fixture requires a certificate.")
    return phx.meshing.certify_cell_mesh(
        source.mesh,
        source.coordinate_contract,
        geometry=source.geometry,
        audit_policy=certification.audit_policy,
        certification_inputs=certification.inputs,
        boundary=source.boundary,
        patches=source.patches,
        zones=source.zones,
        labels=source.labels,
        attributes=source.attributes,
        associations=associations,
        surface_source=source.surface_source,
        region_boundary_evidence=source.region_boundary_evidence,
    )


_ASSOCIATION_AUTHORITY_ERROR = r"Native CAD.*association"


@pytest.mark.parametrize(
    "change",
    (
        "face-entity",
        "face-occurrence",
        "face-region-side",
        "face-residual",
        "cell-occurrence",
        "missing-namespace",
        "missing-table",
        "missing-row",
        "foreign-table",
    ),
)
def test_native_cad_recertification_rejects_changed_source_tables(
    box: tuple[phx.geometry.brep.BRepModel, CellMeshingResult], change: str
) -> None:
    # Curve-only validation admits these changes (or no owning tables at all).
    _, source = box
    associations = _cad_association_input(source, change)
    with pytest.raises(ValueError, match=_ASSOCIATION_AUTHORITY_ERROR):
        _recertify(source, associations)


def test_native_cad_accepts_independently_rebuilt_source_table(
    box: tuple[phx.geometry.brep.BRepModel, CellMeshingResult],
) -> None:
    # Authority binds content, not the Python identity of a table.
    model, source = box
    associations = _cad_association_input(source, "unchanged")
    recertified = _recertify(source, associations)
    assert recertified.certification is not None and recertified.certification.passed
    _transfer(model).validate_source(recertified)


@pytest.mark.parametrize("change", ("face-entity", "cell-occurrence", "missing-table"))
def test_native_cad_adaptation_rejects_changed_tables_before_transfer(
    box: tuple[phx.geometry.brep.BRepModel, CellMeshingResult], change: str
) -> None:
    # Higher-dimensional target rows are rebuilt from vertex data, so waiting
    # until target certification can erase the evidence of a forged source.
    model, source = box
    transfer = _transfer(model)
    request = MarkedMeshAdaptation(np.asarray(source.mesh.blocks[0].global_ids)[:1])
    policy = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_BISECTION, association_transfer=transfer
    )
    prepare_mesh_adaptation(source, request, policy=policy)
    associations = _cad_association_input(source, change)
    forged = eqx.tree_at(lambda value: value.associations, source, associations)
    with pytest.raises(ValueError, match=_ASSOCIATION_AUTHORITY_ERROR):
        prepare_mesh_adaptation(forged, request, policy=policy)


def _boundary_triangles(mesh: phx.discretization.CellMesh, /) -> np.ndarray:
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    faces, counts = np.unique(
        np.sort(
            cells[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]].reshape(-1, 3),
            axis=1,
        ),
        axis=0,
        return_counts=True,
    )
    return faces[counts == 1]


def test_native_cad_adaptation_round_trip_keeps_records_and_certificate() -> None:
    # Two successive refinements of every cell. Bisection splits boundary
    # edges at their physical midpoints (stencil weights 1/2, 1/2); the chart
    # transfer maps each child to the chart sub-triangle with the same corner
    # weights (chord and parameter midpoints on curve edges), the records
    # split curve edges and pole slivers, and the native verifiers re-prove
    # the refined boundary. Faults caught: a chart child off its row (V4/V3
    # refuse), a curve child outside its root interval, a sliver left unsplit,
    # seam curve edges reclassified as face edges (the next transfer refuses).
    # Then a split whose ancestry leaves its chart row is refused with reason.
    from phydrax.meshing._lineage import VertexInterpolationStencil
    from phydrax.meshing._native_cad_records import retarget_native_cad_surface

    model = _sphere_model()
    source = _native(model, 0.3)
    transfer = _transfer(model)
    original = _records(source)
    current = source
    split = None
    for _ in range(2):
        result = execute_mesh_adaptation(
            prepare_mesh_adaptation(
                current,
                MarkedMeshAdaptation(np.asarray(current.mesh.blocks[0].global_ids)),
                policy=MeshAdaptationPolicy(
                    MeshAdaptationRoute.NATIVE_BISECTION, association_transfer=transfer
                ),
            )
        )
        target = result.target
        assert target.mesh.blocks[0].cell_count > current.mesh.blocks[0].cell_count
        certification = target.certification
        assert certification is not None and certification.passed
        native = certification.native_cad
        assert native is not None and native.status == "certified"
        assert native.maximum_deviation is not None and native.maximum_deviation <= 0.05
        before = _records(current)
        records = _records(target)
        assert records.root_id == original.root_id
        assert records.curve is not None and before.curve is not None
        assert len(records.curve.target_global_ids) >= len(before.curve.target_global_ids)
        assert len(records.sliver_vertices) >= len(before.sliver_vertices)
        curve = _curve(target)
        records.require_curve(curve)
        edges = _table(target, 1)
        rows = {
            int(value): row
            for row, value in enumerate(np.asarray(edges.target_global_ids).tolist())
        }
        assert all(
            int(edges.source_dimensions[rows[int(value)]]) == 1
            for value in np.asarray(curve.target_global_ids)
        )
        if len(_boundary_triangles(target.mesh)) > len(_boundary_triangles(current.mesh)):
            split = result
        transfer.validate_source(target)
        current = target
    assert split is not None, "refining every cell twice must split the boundary"
    assert len(_boundary_triangles(current.mesh)) > len(_boundary_triangles(source.mesh))

    # Refusal: move one new boundary vertex's ancestry off its chart row.
    stencil = split.stencil
    assert stencil is not None
    parent = split.source
    assert parent.certification is not None and parent.certification.native_cad
    witness = parent.certification.native_cad.witness
    parent_ids = np.asarray(parent.mesh.vertex_global_ids, dtype=np.int64)
    on_boundary = set(parent_ids[np.unique(_boundary_triangles(parent.mesh))].tolist())
    sources = np.asarray(stencil.source_global_ids, dtype=np.int64).copy()
    valid = np.asarray(stencil.valid, dtype=np.bool_)
    target_boundary = set(
        np.asarray(split.target.mesh.vertex_global_ids)[
            np.unique(_boundary_triangles(split.target.mesh))
        ].tolist()
    )
    row = next(
        index
        for index, identifier in enumerate(np.asarray(stencil.target_global_ids).tolist())
        if identifier in target_boundary and int(valid[index].sum()) == 2
    )
    slot = int(np.flatnonzero(valid[row])[1])
    sources[row, slot] = next(
        int(value)
        for value in parent_ids.tolist()
        if value not in on_boundary and value not in sources[row]
    )
    forged = VertexInterpolationStencil(
        stencil.source_entity_set_id,
        stencil.target_entity_set_id,
        np.asarray(stencil.target_global_ids),
        sources,
        np.asarray(stencil.weights),
        valid,
        preserves_constants=stencil.preserves_constants,
        exact_weights=stencil.exact_weights,
    )
    with pytest.raises(ValueError, match="bound no single source chart triangle"):
        retarget_native_cad_surface(witness, parent.mesh, split.target.mesh, forged)


def _refine(
    result: CellMeshingResult,
    transfer: phx.meshing.BRepAssociationTransfer,
    vertex_rows: np.ndarray,
    /,
) -> MeshAdaptationResult:
    """Bisect every cell incident to ``vertex_rows``."""
    cells = np.asarray(result.mesh.blocks[0].vertices, dtype=np.int64)
    marked = np.flatnonzero(np.isin(cells, vertex_rows).any(axis=1))
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(
            result,
            MarkedMeshAdaptation(np.asarray(result.mesh.blocks[0].global_ids)[marked]),
            policy=MeshAdaptationPolicy(
                MeshAdaptationRoute.NATIVE_BISECTION, association_transfer=transfer
            ),
        )
    )


def _curve_vertex_rows(result: CellMeshingResult, /) -> np.ndarray:
    curve = _records(result).curve
    assert curve is not None
    edges, _ = simplex_volume_entities(result.mesh)
    ids = np.asarray(result.mesh.entity_set(1).entity_ids)
    return np.unique(edges[np.isin(ids, np.asarray(curve.target_global_ids))])


def test_native_cad_adaptation_splits_curve_edges_exactly() -> None:
    # Refining the cells at the box's curve edges bisects curve edges (box
    # h 0.5: 24 -> 25 -> 45 curve edges). Reference: bisection halves an
    # edge, so every child interval is its root interval scaled by a power of
    # two at a multiple of its width, and the children partition the root
    # exactly (records). The native certificate re-proves each target.
    # Then a curve midpoint whose stencil claims weights 1/4, 3/4 is refused
    # (it does not lie at that combination). Faults caught: a chord range
    # split off the vertex, a child interval not a half, a forged stencil
    # mapped anyway.
    from phydrax.meshing._lineage import VertexInterpolationStencil
    from phydrax.meshing._native_cad_records import retarget_native_cad_surface

    model = _box()
    current = _native(model, 0.5)
    transfer = _transfer(model)
    original = _records(current)
    assert original.curve is not None
    split = None
    for _ in range(2):
        result = _refine(current, transfer, _curve_vertex_rows(current))
        target = result.target
        certification = target.certification
        assert certification is not None and certification.passed
        assert certification.native_cad is not None
        assert certification.native_cad.status == "certified"
        records = _records(target)
        assert records.root_id == original.root_id
        curve = records.curve
        assert curve is not None
        before = _records(current).curve
        assert before is not None
        if len(curve.target_global_ids) > len(before.target_global_ids):
            split = result
        records.require_curve(_curve(target))
        for parent, interval in zip(
            np.asarray(curve.parent_ids).tolist(),
            curve.source_parameter_intervals,
            strict=True,
        ):
            assert interval is not None
            low, high = sorted(
                Fraction.from_float(value) for value in original.root_parameters[parent]
            )
            width = interval[1] - interval[0]
            ratio = (high - low) / width
            assert ratio.denominator == 1
            assert ratio.numerator & (ratio.numerator - 1) == 0
            assert ((interval[0] - low) / width).denominator == 1
        transfer.validate_source(target)
        current = target
    assert split is not None, "refining at the curve edges must split one"

    stencil = split.stencil
    assert stencil is not None
    parent = split.source
    assert parent.certification is not None
    assert parent.certification.native_cad is not None
    parent_curve = _records(parent).curve
    assert parent_curve is not None
    edges, _ = simplex_volume_entities(parent.mesh)
    entity_ids = np.asarray(parent.mesh.entity_set(1).entity_ids)
    parent_ids = np.asarray(parent.mesh.vertex_global_ids)
    curve_pairs = {
        frozenset(parent_ids[edge].tolist())
        for edge in edges[np.isin(entity_ids, np.asarray(parent_curve.target_global_ids))]
    }
    sources = np.asarray(stencil.source_global_ids, dtype=np.int64)
    valid = np.asarray(stencil.valid, dtype=np.bool_)
    row = next(
        index
        for index in range(sources.shape[0])
        if int(valid[index].sum()) == 2
        and frozenset(sources[index][valid[index]].tolist()) in curve_pairs
    )
    weights = np.asarray(stencil.weights, dtype=np.float64).copy()
    exact = [list(values) for values in stencil.exact_weights]
    first, second = np.flatnonzero(valid[row])[:2]
    weights[row, first], weights[row, second] = 0.25, 0.75
    exact[row][first], exact[row][second] = Fraction(1, 4), Fraction(3, 4)
    forged = VertexInterpolationStencil(
        stencil.source_entity_set_id,
        stencil.target_entity_set_id,
        np.asarray(stencil.target_global_ids),
        sources,
        weights,
        valid,
        preserves_constants=stencil.preserves_constants,
        exact_weights=tuple(tuple(values) for values in exact),
    )
    with pytest.raises(ValueError, match="does not lie at its stencil combination"):
        retarget_native_cad_surface(
            parent.certification.native_cad.witness,
            parent.mesh,
            split.target.mesh,
            forged,
        )


def test_native_cad_adaptation_refuses_a_split_pole_segment() -> None:
    # Refining the cells at the sphere's pole slivers bisects a sliver's P-Q
    # edge in the third round (sphere h 0.3). Its midpoint M would need two
    # chart points joined by a collapsed edge off the pole, which V3's
    # chart-mesh link rejects (measured before this refusal: every chart row
    # unresolved, reason 4, CHART_MESH_LINK findings on the split slivers),
    # so the transfer refuses with its reason instead. The first two rounds,
    # which keep every P-Q edge, are certified with the slivers kept.
    model = _sphere_model()
    source = _native(model, 0.3)
    transfer = _transfer(model)

    def pole_rows(result: CellMeshingResult, /) -> np.ndarray:
        ids = np.asarray(result.mesh.vertex_global_ids)
        return np.flatnonzero(np.isin(ids, _records(result).sliver_vertices.reshape(-1)))

    current = source
    for _ in range(2):
        refined = _refine(current, transfer, pole_rows(current)).target
        assert refined.certification is not None and refined.certification.passed
        np.testing.assert_array_equal(
            _records(refined).sliver_vertices, _records(source).sliver_vertices
        )
        current = refined
    with pytest.raises(ValueError, match="split pole segment"):
        _refine(current, transfer, pole_rows(current))


@pytest.mark.parametrize("moved", ("root", "midpoint-normal", "midpoint-ulp"))
def test_curve_archive_replay_refuses_unrealized_stencil(moved: str) -> None:
    from phydrax.meshing._native_cad_records import (
        _CurveSegment,
        _replay_curve_intervals,
        _replay_curve_splits,
    )

    abscissa = float(1 << 40)
    start = np.array((abscissa, 0.0, 0.0), dtype=np.float64)
    stop = np.array((abscissa, 2.0, 0.0), dtype=np.float64)
    midpoint = np.array((abscissa, 1.0, 0.0), dtype=np.float64)
    root = _CurveSegment(0, 0.0, 1.0, 0, 2)
    parts = (_CurveSegment(0, 0.0, 0.5, 0, 1), _CurveSegment(0, 0.5, 1.0, 1, 2))
    origin = {0: start, 2: stop}
    points = {0: start, 1: midpoint, 2: stop}
    expected, addresses = _replay_curve_splits(
        (root,), origin, ((1, 0, 2, Fraction(1, 2)),)
    )
    assert _replay_curve_intervals(0, root, parts, points, expected, addresses) == (
        (Fraction(0), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(1)),
    )

    row = 0 if moved == "root" else 1
    points[row] = points[row].copy()
    if moved == "midpoint-ulp":
        points[row][0] = np.nextafter(abscissa, np.inf)
    else:
        points[row][2] = 2.0**-16
    reason = "moved a retained root endpoint" if row == 0 else "stencil combination"
    with pytest.raises(ValueError, match=reason):
        _replay_curve_intervals(0, root, parts, points, expected, addresses)


@pytest.mark.parametrize("composed", (True, False), ids=("one-epoch", "two-epochs"))
def test_curve_archive_replay_preserves_stencil_grouping(composed: bool) -> None:
    from phydrax.meshing._native_cad_records import _CurveSegment, _replay_curve_splits

    base = float(1 << 40)
    ulp = np.spacing(base)
    origin = {
        0: np.array((base + ulp, 0.0, 0.0), dtype=np.float64),
        1: np.array((base + 7 * ulp, 2.0, 0.0), dtype=np.float64),
    }
    quarter = (3, 0, 1, Fraction(1, 4)) if composed else (3, 0, 2, Fraction(1, 2))
    points, addresses = _replay_curve_splits(
        (_CurveSegment(0, 0.0, 1.0, 0, 1),),
        origin,
        ((2, 0, 1, Fraction(1, 2)), quarter),
    )
    # Binary64 round-to-even gives different x coordinates at the same exact
    # quarter parameter when the midpoint is rounded in an earlier epoch.
    np.testing.assert_array_equal(
        points[3], (base + (3 if composed else 2) * ulp, 0.5, 0.0)
    )
    assert addresses[3] == {0: Fraction(1, 4)}


@pytest.mark.parametrize("adapted", (False, True), ids=("root", "split"))
def test_native_cad_records_survive_the_archive(
    box: tuple[phx.geometry.brep.BRepModel, CellMeshingResult],
    tmp_path: Path,
    adapted: bool,
) -> None:
    # Fault caught: records dropped or rebuilt differently by the codec.
    from phydrax.meshing._model_codec import (
        model_archive_content_id,
        MODEL_ARCHIVE_LIMITS,
        read_model_artifact,
        restore_model_archive_value,
        write_model_artifact,
    )

    model, result = box
    if adapted:
        result = _native(model, 0.5)
        root = _records(result)
        transfer = _transfer(model)
        for _ in range(2):
            result = _refine(result, transfer, _curve_vertex_rows(result)).target
        assert _records(result).root_id == root.root_id
        assert len(_curve(result).target_global_ids) > root.root_curves.size
    path = tmp_path / "native-cad-records.phx"
    write_model_artifact(path, result, limits=MODEL_ARCHIVE_LIMITS)
    restored = read_model_artifact(
        path,
        expected_type=CellMeshingResult,
        restore=restore_model_archive_value,
        expected_content_id=model_archive_content_id(path, limits=MODEL_ARCHIVE_LIMITS),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    assert _records(restored).records_id == _records(result).records_id
    assert [item.association_id for item in restored.associations] == [
        item.association_id for item in result.associations
    ]
    _transfer(model).validate_source(restored)


@pytest.mark.parametrize(
    "change",
    (
        "face-region-side",
        "cell-occurrence",
        "missing-namespace",
        "foreign-table",
        "rehashed-face",
    ),
)
def test_native_cad_archive_rejects_changed_source_tables(
    box: tuple[phx.geometry.brep.BRepModel, CellMeshingResult],
    tmp_path: Path,
    change: str,
) -> None:
    # An archive can present the same changed tables in the result and the
    # stored report. Agreement between these two envelopes is not authority.
    from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
    from phydrax.meshing._model_codec import (
        _write_model_archive_recipe,
        model_archive_content_id,
        MODEL_ARCHIVE_LIMITS,
        read_model_archive,
        read_model_artifact,
        restore_model_archive_value,
        write_model_artifact,
    )
    from phydrax.meshing._native_cad_records import _association_key

    _, source = box
    report = source.certification
    if report is None:
        raise ValueError("The native CAD fixture requires a certificate.")
    associations = _cad_association_input(
        source, "face-entity" if change == "rehashed-face" else change
    )
    stored_report = eqx.tree_at(lambda value: value.associations, report, associations)
    forged = eqx.tree_at(
        lambda value: (value.associations, value.certification),
        source,
        (associations, stored_report),
    )
    # Re-encode the complete manifest and array envelope from ordinary rows.
    path = write_model_artifact(
        tmp_path / "changed-cad-tables.phx", forged, limits=MODEL_ARCHIVE_LIMITS
    )
    if change == "rehashed-face":
        records = _records(source)
        assert records.origin_chart is None and not records.curve_splits
        manifest, arrays = read_model_archive(path, limits=MODEL_ARCHIVE_LIMITS)
        recipe = manifest["model_recipe"]
        certificate = recipe["fields"]["certification"]["fields"]
        saved = certificate["inputs"]["fields"]["native_cad_records"]["fields"]
        keys = tuple(_association_key(item) for item in associations)
        assert keys != records.association_keys
        for item, key in zip(saved["association_keys"]["items"], keys, strict=True):
            item["value"] = key
        # Mutate the serialized expectation too and rehash it; the outer
        # request/result IDs must not be the owner that discovers this forgery.
        claimed_id = canonical_fingerprint(
            {
                "kind": "native-cad-records",
                "mesh": records.mesh_id,
                "tolerance": records.tolerance,
                "topology": records.topology_id,
                "witness": records.witness_id,
                "origin": None,
                "curve_splits": (),
                "roots": records.root_id,
                "associations": keys,
                "slivers": array_tree_fingerprint(records.sliver_vertices),
            }
        )
        saved["records_id"]["value"] = claimed_id
        certificate["native_cad"]["fields"]["records_id"]["value"] = claimed_id
        path = _write_model_archive_recipe(
            tmp_path / "rehashed-cad-tables.phx", recipe, arrays, MODEL_ARCHIVE_LIMITS
        )
    with pytest.raises(ArrayArchiveCorruptionError) as refusal:
        read_model_artifact(
            path,
            expected_type=CellMeshingResult,
            restore=restore_model_archive_value,
            expected_content_id=model_archive_content_id(
                path, limits=MODEL_ARCHIVE_LIMITS
            ),
            limits=MODEL_ARCHIVE_LIMITS,
        )
    # The owner must refuse the CAD associations, not later catch a stale
    # outer result ID or reject the archive structure before seeing the tables.
    cause = refusal.value.__cause__
    assert isinstance(cause, ValueError)
    assert re.search(_ASSOCIATION_AUTHORITY_ERROR, str(cause))
    if change == "rehashed-face":
        assert "differs from replayed source authority" in str(cause)
