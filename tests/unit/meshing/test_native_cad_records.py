# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native CAD solid records and the consumers of their associations."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry.brep import brep_box
from phydrax.meshing._adaptation import (
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
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
    result = _cad_volume_plan(model, size, 1, radius_edge=8.0, dihedral=0.0).execute()
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
    from phydrax.meshing._controls import RegionControl
    from phydrax.meshing._organization import RegionRole

    model, native = box
    plan = _cad_volume_plan(model, 0.3, 1, radius_edge=8.0, dihedral=0.0)
    specification = eqx.tree_at(
        lambda value: value.region_controls,
        plan.specification,
        (
            RegionControl(
                phx.meshing.MeshingScope.from_geometry(model, model.solid_ids[0]),
                "region-0",
                "material-0",
                RegionRole.SOLID,
            ),
        ),
        is_leaf=lambda value: value is None,
    )
    existing = (
        phx.meshing.NativeMeshingProvider(plan.options)
        .plan(model, specification)
        .execute()
    )
    assert isinstance(existing, CellMeshingResult)
    assert existing.mesh.blocks[0].name != "native-cad-tetrahedra"
    for dimension in range(4):
        ours, theirs = _table(native, dimension), _table(existing, dimension)
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
    assert set(_curve(native).source_entity_ids) == set(
        _curve(existing).source_entity_ids
    )


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


def test_native_cad_adaptation_round_trip_keeps_records_and_certificate() -> None:
    # Two successive refinements of the cells off the boundary: bisection that
    # keeps the CAD boundary carries the source chart row for row; the native
    # verifiers re-prove it, and the target is a valid source of the next
    # adaptation. (Marking every cell may bisect a boundary edge, which the
    # chart transfer refuses.) Faults caught: a chart left on stale vertex rows
    # (V2/V4 refuse), seam curve edges reclassified as face edges by projection
    # (the second transfer refuses), records dropped by the adaptation.
    model = _sphere_model()
    source = _native(model, 0.2)
    transfer = _transfer(model)
    original = _records(source)
    assert source.certification is not None
    assert source.certification.native_cad is not None
    chart = source.certification.native_cad.witness
    source_ids = np.asarray(source.mesh.vertex_global_ids)
    current = source
    for _ in range(2):
        cells = np.asarray(current.mesh.blocks[0].vertices, dtype=np.int64)
        faces, counts = np.unique(
            np.sort(
                cells[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]].reshape(-1, 3),
                axis=1,
            ),
            axis=0,
            return_counts=True,
        )
        interior = ~np.isin(cells, faces[counts == 1]).any(axis=1)
        assert interior.any()
        target = execute_mesh_adaptation(
            prepare_mesh_adaptation(
                current,
                MarkedMeshAdaptation(
                    np.asarray(current.mesh.blocks[0].global_ids)[interior]
                ),
                policy=MeshAdaptationPolicy(
                    MeshAdaptationRoute.NATIVE_BISECTION, association_transfer=transfer
                ),
            )
        ).target
        assert target.mesh.blocks[0].cell_count > current.mesh.blocks[0].cell_count
        certification = target.certification
        assert certification is not None and certification.passed
        native = certification.native_cad
        assert native is not None and native.status == "certified"
        assert native.maximum_deviation is not None and native.maximum_deviation <= 0.05
        target_ids = np.asarray(target.mesh.vertex_global_ids)
        np.testing.assert_array_equal(
            target_ids[np.asarray(native.witness.chart_triangles)],
            source_ids[np.asarray(chart.chart_triangles)],
        )
        np.testing.assert_array_equal(native.witness.chart_uv, chart.chart_uv)
        records = _records(target)
        assert records.root_id == original.root_id
        np.testing.assert_array_equal(records.sliver_vertices, original.sliver_vertices)
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
        transfer.validate_source(target)
        current = target


def test_native_cad_records_survive_the_archive(
    box: tuple[phx.geometry.brep.BRepModel, CellMeshingResult], tmp_path: Path
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
