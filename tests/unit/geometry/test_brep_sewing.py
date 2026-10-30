from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep._model import BRepGeometry, BRepImportReport, BRepModel
from phydrax.geometry.brep._patches import ExactLineCurve, PlanePatch
from phydrax.geometry.brep._sewing import BRepSewingFailure, sew_brep


_CONTRACT = SpatialCoordinateContract.si()
_FACE_FRAMES = (
    ((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 2)),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1)),
    ((1, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1)),
    ((0, 0, 0), (1, 0, 0), (0, 0, 1), (2, 1)),
    ((2, 1, 0), (-1, 0, 0), (0, 0, 1), (2, 1)),
    ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1)),
    ((2, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1)),
)


def _face_soup(top_height: float = 1.0) -> BRepModel:
    points: list[tuple[float, float, float]] = []
    exact_points: list[tuple[Fraction, Fraction, Fraction]] = []
    curves: list[ExactLineCurve] = []
    ranges: list[tuple[float, float]] = []
    edge_vertices: list[tuple[int, int]] = []
    pcurves: list[ExactLineCurve] = []
    coedge_edges: list[int] = []
    face_loops: list[tuple[tuple[int, ...], ...]] = []
    patches: list[PlanePatch] = []
    bounds: list[tuple[tuple[float, float], tuple[float, float]]] = []

    for face, (origin_raw, first_raw, second_raw, size) in enumerate(_FACE_FRAMES):
        origin = tuple(Fraction.from_float(float(value)) for value in origin_raw)
        first = tuple(Fraction.from_float(float(value)) for value in first_raw)
        second = tuple(Fraction.from_float(float(value)) for value in second_raw)
        if face in (1, 2):
            origin = (origin[0], origin[1], Fraction.from_float(top_height))
        uv = ((0, 0), (size[0], 0), (size[0], size[1]), (0, size[1]))
        vertices: list[int] = []
        exact_face_points: list[tuple[Fraction, Fraction, Fraction]] = []
        for u, v in uv:
            point = (
                origin[0] + Fraction(u) * first[0] + Fraction(v) * second[0],
                origin[1] + Fraction(u) * first[1] + Fraction(v) * second[1],
                origin[2] + Fraction(u) * first[2] + Fraction(v) * second[2],
            )
            exact_face_points.append(point)
            vertices.append(len(points))
            exact_points.append(point)
            points.append((float(point[0]), float(point[1]), float(point[2])))
        loop: list[int] = []
        for edge_position in range(4):
            next_position = (edge_position + 1) % 4
            edge = len(curves)
            curves.append(ExactLineCurve(exact_face_points[edge_position], exact_face_points[next_position]))
            ranges.append((0.0, 1.0))
            edge_vertices.append((vertices[edge_position], vertices[next_position]))
            pcurves.append(ExactLineCurve(uv[edge_position], uv[next_position]))
            coedge_edges.append(edge)
            loop.append(len(coedge_edges) - 1)
        face_loops.append((tuple(loop),))
        patches.append(
            PlanePatch(
                tuple(float(value) for value in origin),
                tuple(float(value) for value in first),
                tuple(float(value) for value in second),
            )
        )
        bounds.append(((0.0, 0.0), (float(size[0]), float(size[1]))))

    geometry = BRepGeometry(
        vertex_points=np.asarray(points, dtype=np.float64),
        vertex_exact_points=tuple(exact_points),
        curves=tuple(curves),
        edge_curves=tuple(range(len(curves))),
        edge_ranges=np.asarray(ranges, dtype=np.float64),
        edge_vertices=tuple(edge_vertices),
        pcurves=tuple(pcurves),
        coedge_edges=tuple(coedge_edges),
        coedge_senses=(1,) * len(coedge_edges),
        face_loops=tuple(face_loops),
        shell_faces=(),
        shell_orientations=(),
        solid_shells=(),
    )
    report = BRepImportReport(
        source_id="sewing-test-source",
        source_digest="a" * 64,
        source_format="native-test",
        coordinate_contract=_CONTRACT,
        import_policy_id="b" * 64,
        num_solids=0,
        num_faces=len(face_loops),
        num_edges=len(curves),
        num_vertices=len(points),
    )
    return BRepModel(
        geometry=geometry,
        patches=tuple(patches),
        parameter_bounds=np.asarray(bounds, dtype=np.float64),
        orientation=np.ones((len(face_loops),), dtype=np.float64),
        physical_tags=tuple(f"face-{index}" for index in range(len(face_loops))),
        coordinate_contract=_CONTRACT,
        report=report,
    )


def _model_for_result(source: BRepModel, result_geometry: BRepGeometry) -> BRepModel:
    report = BRepImportReport(
        source_id="native-sewing-result",
        source_digest="c" * 64,
        source_format="native-test",
        coordinate_contract=_CONTRACT,
        import_policy_id="d" * 64,
        num_solids=result_geometry.topology.num_solids,
        num_faces=result_geometry.topology.num_faces,
        num_edges=result_geometry.topology.num_edges,
        num_vertices=result_geometry.topology.num_vertices,
    )
    return BRepModel(
        geometry=result_geometry,
        patches=source.patches,
        parameter_bounds=source.parameter_bounds,
        orientation=source.orientation,
        physical_tags=source.physical_tags,
        coordinate_contract=_CONTRACT,
        report=report,
    )


def test_exact_split_face_shell_sewing_publishes_verified_lineage() -> None:
    source_model = _face_soup()
    source = source_model.geometry
    source_id = source.geometry_id
    original_points = np.asarray(source.vertex_points).copy()

    result = sew_brep(source, source_model.orientation, (tuple(range(7)),))
    target = result.geometry

    assert source.geometry_id == source_id
    np.testing.assert_array_equal(np.asarray(source.vertex_points), original_points)
    assert target.solid_shells == ((0,),)
    assert (target.topology.num_vertices, target.topology.num_edges, target.topology.num_faces) == (
        10,
        15,
        7,
    )
    assert len(result.lineage.vertex_sources) == target.topology.num_vertices
    assert sorted(vertex for values in result.lineage.vertex_sources for vertex in values) == list(
        range(source.topology.num_vertices)
    )
    assert {image.source_edge for image in result.lineage.edge_images} == set(
        range(source.topology.num_edges)
    )
    assert set(result.lineage.coedge_sources) == set(range(len(source.coedge_edges)))
    assert sum(len(result.lineage.source_images(edge)) == 2 for edge in range(source.topology.num_edges)) == 2
    assert any(len(result.lineage.target_sources(edge)) > 1 for edge in range(target.topology.num_edges))
    assert {image.orientation for image in result.lineage.edge_images} == {-1, 1}

    for image in result.lineage.edge_images:
        source_curve = source.curves[source.edge_curves[image.source_edge]]
        target_curve = target.curves[target.edge_curves[image.target_edge]]
        assert isinstance(source_curve, ExactLineCurve)
        assert isinstance(target_curve, ExactLineCurve)
        assert image.offset_turns == 0
        for target_parameter in np.asarray(target.edge_ranges)[image.target_edge]:
            exact_target = Fraction.from_float(float(target_parameter))
            exact_source = image.scale * exact_target + image.offset_rational
            assert source_curve.evaluate_exact(exact_source) == target_curve.evaluate_exact(exact_target)

    rebuilt = _model_for_result(source_model, target)
    assert rebuilt.geometry is target
    assert rebuilt.topology.num_solids == 1


def test_noncoincident_exact_shell_fails_without_mutating_source() -> None:
    source_model = _face_soup(float(np.nextafter(1.0, 2.0)))
    source = source_model.geometry
    source_id = source.geometry_id
    original_points = np.asarray(source.vertex_points).copy()

    with pytest.raises(BRepSewingFailure, match="open") as raised:
        sew_brep(source, source_model.orientation, (tuple(range(7)),))
    assert raised.value.contacts_complete

    assert source.geometry_id == source_id
    np.testing.assert_array_equal(np.asarray(source.vertex_points), original_points)
