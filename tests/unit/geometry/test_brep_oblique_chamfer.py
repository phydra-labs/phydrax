"""Independent wedge-volume and rational plane-incidence chamfer oracles."""

from fractions import Fraction

import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry.brep._constructors import Builder, finish
from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.geometry.brep._fillet import brep_chamfer
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._patches import LineCurve, PlanePatch
from phydrax.geometry.brep._request_preparations import TrimRequestPreparations


# Empirical `test_oblique_wedge_chamfer_matches_closed_form_volume`: matched max |delta|=3.55271e-15; joint allowance fraction=5.87327e-05, margin=99.9941%.
_OBLIQUE_CHAMFER_VOLUME_RTOL = 2e-12


_CONTRACT = phx.SpatialCoordinateContract.si()
_POLICY = phx.geometry.BRepTessellationPolicy(realize=False)


def _wedge_uv(face: int, point: np.ndarray, /) -> np.ndarray:
    match face:
        case 0 | 4:
            return np.asarray((point[0], point[2]), dtype=np.float64)
        case 1 | 3:
            return np.asarray((point[1], point[2]), dtype=np.float64)
        case 2:
            return np.asarray((point[0], point[1]), dtype=np.float64)
        case 5:
            return np.asarray((point[0], 3.0 - point[1]), dtype=np.float64)
        case _:
            raise ValueError(
                "WedgeFixture.face must identify one of its six planes; got an unknown face index"
            )


def _wedge() -> tuple[BRepModel, int]:
    """Author x∈[0,2], y∈[0,3], z∈[0,4+x] without a blend helper."""
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 3.0, 0.0),
            (0.0, 3.0, 0.0),
            (0.0, 0.0, 4.0),
            (2.0, 0.0, 6.0),
            (2.0, 3.0, 6.0),
            (0.0, 3.0, 4.0),
        ),
        dtype=np.float64,
    )
    # Incident face order y=0, x=0 makes (a,b) mean x setback, y setback.
    faces = (
        (
            (0, 1, 5, 4),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1,
            "source:y0",
        ),
        (
            (0, 3, 7, 4),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            -1,
            "source:x0",
        ),
        (
            (0, 1, 2, 3),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            -1,
            "source:bottom",
        ),
        (
            (1, 2, 6, 5),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            1,
            "source:x2",
        ),
        (
            (3, 2, 6, 7),
            (0.0, 3.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            -1,
            "source:y3",
        ),
        # Raw normal is (1,0,-1); its irrational unit normalization is unused.
        (
            (7, 6, 5, 4),
            (0.0, 3.0, 4.0),
            (1.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
            -1,
            "source:top",
        ),
    )
    builder = Builder()
    builder.vertices = [point for point in vertices]
    edge_map: dict[tuple[int, int], int] = {}
    preparations = TrimRequestPreparations()
    with coordinate_source_request((vertices, faces)) as budget:
        for face, (
            indices,
            origin,
            first_axis,
            second_axis,
            orientation,
            tag,
        ) in enumerate(faces):
            loop = []
            for first, last in zip(indices, indices[1:] + indices[:1], strict=True):
                key = min(first, last), max(first, last)
                if key not in edge_map:
                    start, end = key
                    edge_map[key] = builder.edge(
                        LineCurve(vertices[start], vertices[end] - vertices[start]),
                        0.0,
                        1.0,
                        start,
                        end,
                    )
                edge = edge_map[key]
                start, end = builder.edge_vertices[edge]
                uv = _wedge_uv(face, vertices[start])
                direction = _wedge_uv(face, vertices[end]) - uv
                loop.append(
                    builder.coedge(
                        edge,
                        1 if first == start else -1,
                        LineCurve(uv, direction),
                        preparations=preparations,
                        budget=budget,
                    )
                )
            points = np.stack([_wedge_uv(face, vertices[index]) for index in indices])
            builder.face(
                PlanePatch(origin, first_axis, second_axis),
                np.stack((points.min(axis=0), points.max(axis=0))),
                [loop],
                tag,
                None,
                orientation=orientation,
            )
        model = finish(
            builder,
            solid=True,
            coordinate_contract=_CONTRACT,
            source_id="oblique-chamfer-wedge",
            tessellation=_POLICY,
            preparations=preparations,
            budget=budget,
        )
    return model, edge_map[(0, 4)]


def _chamfer(source: BRepModel, edge: int, setbacks: tuple[float, float], /) -> BRepModel:
    result = brep_chamfer(
        source, (source.edge_ids[edge],), setbacks, tessellation=_POLICY
    )
    assert result.status == "certified", result.reason
    assert result.model is not None
    assert result.g1_angle_bounds.shape == (0, 2)
    return result.model


def _face_edges(model: BRepModel, face: int, /) -> frozenset[int]:
    geometry = model.require_geometry()
    return frozenset(
        geometry.coedge_edges[coedge]
        for loop in geometry.face_loops[face]
        for coedge in loop
    )


def _exact_lift(model: BRepModel, face: int, coedge: int, /) -> None:
    """Compare rational affine coefficients, proving equality on the whole line."""
    geometry = model.require_geometry()
    patch = model.patches[face]
    edge = geometry.coedge_edges[coedge]
    curve = geometry.curves[geometry.edge_curves[edge]]
    pcurve = geometry.pcurves[coedge]
    assert isinstance(patch, PlanePatch)
    assert isinstance(curve, LineCurve)
    assert isinstance(pcurve, LineCurve)
    origin, first_axis, second_axis = (
        tuple(Fraction(float(value)) for value in np.asarray(vector))
        for vector in (patch.origin, patch.first_axis, patch.second_axis)
    )
    uv, direction = (
        tuple(Fraction(float(value)) for value in np.asarray(vector))
        for vector in (pcurve.origin, pcurve.direction)
    )
    lifted_origin = tuple(
        value + uv[0] * first + uv[1] * second
        for value, first, second in zip(origin, first_axis, second_axis, strict=True)
    )
    lifted_direction = tuple(
        direction[0] * first + direction[1] * second
        for first, second in zip(first_axis, second_axis, strict=True)
    )
    assert lifted_origin == tuple(
        Fraction(float(value)) for value in np.asarray(curve.origin)
    )
    assert lifted_direction == tuple(
        Fraction(float(value)) for value in np.asarray(curve.direction)
    )


@pytest.mark.parametrize(
    "setbacks",
    [(0.25, 0.5), (0.5, 0.25)],
    ids=["x-quarter-y-half", "x-half-y-quarter"],
)
def test_oblique_wedge_chamfer_matches_closed_form_volume(
    setbacks: tuple[float, float],
) -> None:
    source, edge = _wedge()
    result = _chamfer(source, edge, setbacks)
    a, b = setbacks
    # Integrate (4+x) over the removed right triangle: area*a-centroid law.
    loss = (a * b / 2.0) * (4.0 + a / 3.0)
    source_volume = float(
        phx.geometry.prepare_brep_query(source).measures.solid_volumes[0]
    )
    volume = float(phx.geometry.prepare_brep_query(result).measures.solid_volumes[0])
    assert source_volume == pytest.approx(30.0, rel=_OBLIQUE_CHAMFER_VOLUME_RTOL)
    assert volume == pytest.approx(30.0 - loss, rel=_OBLIQUE_CHAMFER_VOLUME_RTOL)


def test_oblique_wedge_chamfer_caps_are_exact_identity_shared_intersections() -> None:
    source, selected = _wedge()
    a, b = 0.25, 0.5
    result = _chamfer(source, selected, (a, b))
    geometry = result.require_geometry()
    chamfer = len(source.patches)
    assert len(geometry.vertex_points) == 10
    assert len(geometry.edge_vertices) == 15
    assert len(result.patches) == 7
    for tag, expected in (
        ("source:bottom", {(a, 0.0, 0.0), (0.0, b, 0.0)}),
        ("source:top", {(a, 0.0, 4.0 + a), (0.0, b, 4.0)}),
    ):
        cap = result.physical_tags.index(tag)
        shared = _face_edges(result, cap) & _face_edges(result, chamfer)
        assert len(shared) == 1
        edge = next(iter(shared))
        points = {
            tuple(np.asarray(geometry.vertex_points[index]).tolist())
            for index in geometry.edge_vertices[edge]
        }
        assert points == expected
        owners = []
        for face, loops in enumerate(geometry.face_loops):
            for loop in loops:
                for coedge in loop:
                    if geometry.coedge_edges[coedge] == edge:
                        owners.append(face)
                        _exact_lift(result, face, coedge)
        assert sorted(owners) == sorted((cap, chamfer))
    for edge in range(len(geometry.edge_vertices)):
        assert geometry.coedge_edges.count(edge) == 2
    assert geometry.edge_vertex_gap_bounds is None
    assert geometry.edge_face_gap_bounds is None


def test_oblique_wedge_chamfer_retains_source_planes_and_named_edge_incidence() -> None:
    source, selected = _wedge()
    result = _chamfer(source, selected, (0.25, 0.5))
    source_geometry, geometry = source.require_geometry(), result.require_geometry()
    assert result.physical_tags[:-1] == source.physical_tags
    for face, patch in enumerate(source.patches):
        assert result.patches[face] is patch
    endpoints = frozenset(source_geometry.edge_vertices[selected])
    for source_edge, indices in enumerate(source_geometry.edge_vertices):
        if endpoints & frozenset(indices):
            continue
        curve = source_geometry.curves[source_geometry.edge_curves[source_edge]]
        retained = tuple(
            edge
            for edge, curve_index in enumerate(geometry.edge_curves)
            if geometry.curves[curve_index] is curve
        )
        assert len(retained) == 1
        old_owners = frozenset(
            source.physical_tags[face]
            for face in range(len(source.patches))
            if source_edge in _face_edges(source, face)
        )
        new_owners = frozenset(
            result.physical_tags[face]
            for face in range(len(result.patches))
            if retained[0] in _face_edges(result, face)
        )
        assert new_owners == old_owners


def test_oblique_wedge_chamfer_refuses_disappearing_neighbor() -> None:
    source, edge = _wedge()
    result = brep_chamfer(
        source, (source.edge_ids[edge],), (2.0, 0.5), tessellation=_POLICY
    )
    assert result.status == "unsupported"
    assert result.model is None
    assert "face vanishes" in result.reason


def test_oblique_wedge_setback_scale_limit_is_exact_and_straddles_construction() -> None:
    source, edge = _wedge()
    setbacks = (0.25, 0.5)
    # x∈[0,2], y∈[0,3]. The horizontal setback reaches x=2 at scale
    # 8, while the other reaches y=3 at scale 6. Slanted top contact
    # coordinates remain coplanar and retain height 4+x throughout.
    upper = min(2.0 / setbacks[0], 3.0 / setbacks[1])
    original = brep_chamfer(
        source,
        (source.edge_ids[edge],),
        setbacks,
        tessellation=_POLICY,
        include_parameter_range=True,
    )
    assert original.status == "certified", original.reason
    certificate = original.parameter_range
    assert certificate is not None and certificate.parameter == "setback_scale"
    assert certificate.interval == (0.0, upper)
    assert certificate.end_status == ("attained_exactly", "attained_exactly")
    for scale, expected in (
        (upper - 2.0**-10, "certified"),
        (upper + 2.0**-10, "unsupported"),
    ):
        result = brep_chamfer(
            source,
            (source.edge_ids[edge],),
            (setbacks[0] * scale, setbacks[1] * scale),
            tessellation=_POLICY,
            include_parameter_range=True,
        )
        assert result.status == expected, result.reason
        assert result.parameter_range is not None
