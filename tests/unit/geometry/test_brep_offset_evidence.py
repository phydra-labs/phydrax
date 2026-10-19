"""Binary64 offset publication gaps against independent rational-rational intent."""

from fractions import Fraction

import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry.brep._patches import LineCurve, PlanePatch


_CONTRACT = phx.SpatialCoordinateContract.si()
_LAZY = phx.geometry.BRepTessellationPolicy(realize=False)


def _source(family: str) -> phx.geometry.BRepModel:
    if family == "box":
        return phx.geometry.brep_box(
            (0.0, 0.0, 0.0),
            (2.0, 3.0, 4.0),
            coordinate_contract=_CONTRACT,
            tessellation=_LAZY,
        )
    profile = phx.geometry.PlanarProfile(
        phx.geometry.ProfilePlane(),
        phx.geometry.ProfileLoop.polygon(
            ((0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (1.0, 1.0), (1.0, 3.0), (0.0, 3.0))
        ),
    )
    return phx.geometry.brep_extrusion(
        profile, (0.0, 0.0, 2.0), coordinate_contract=_CONTRACT, tessellation=_LAZY
    )


def _intent(
    point: np.ndarray, family: str, delta: Fraction
) -> tuple[Fraction, Fraction, Fraction]:
    x, y, z = (Fraction(float(value)) for value in point)
    if family == "box":
        return (
            x + (-delta if x == 0 else delta),
            y + (-delta if y == 0 else delta),
            z + (-delta if z == 0 else delta),
        )
    return (
        x + (-delta if x == 0 else delta),
        y + (-delta if y == 0 else delta),
        z + (-delta if z == 0 else delta),
    )


@pytest.mark.parametrize("family", ["box", "l_prism"])
def test_binary_distance_offset_publishes_certified_intent_and_gap_bounds(
    family: str,
) -> None:
    source = _source(family)
    distance = 0.1
    delta = Fraction(distance)
    result = phx.geometry.brep_offset(source, distance, tessellation=_LAZY)
    model, evidence = result.model, result.evidence
    assert evidence.face_rounding_bounds is not None
    assert evidence.vertex_rounding_bounds is not None
    expected = (
        (2 + 2 * delta) * (3 + 2 * delta) * (4 + 2 * delta)
        if family == "box"
        else (5 + 12 * delta + 4 * delta**2) * (2 + 2 * delta)
    )
    assert evidence.volume_status == "certified"
    enclosure = evidence.volume_enclosure
    assert enclosure is not None
    assert Fraction(float(enclosure[0])) <= expected <= Fraction(float(enclosure[1]))
    original, geometry = source.require_geometry(), model.require_geometry()
    assert geometry.edge_vertices == original.edge_vertices
    assert geometry.face_loops == original.face_loops
    assert np.any(np.asarray(evidence.vertex_rounding_bounds) > 0.0)
    for index, point in enumerate(np.asarray(original.vertex_points)):
        exact = _intent(point, family, delta)
        rounded = tuple(
            Fraction(float(value)) for value in np.asarray(geometry.vertex_points[index])
        )
        squared = sum(
            ((a - b) ** 2 for a, b in zip(exact, rounded, strict=True)), Fraction()
        )
        assert squared <= Fraction(float(evidence.vertex_rounding_bounds[index])) ** 2
    gaps = geometry.edge_vertex_gap_bounds
    assert gaps is not None
    for edge, (first, last) in enumerate(geometry.edge_vertices):
        curve = geometry.curves[geometry.edge_curves[edge]]
        assert isinstance(curve, LineCurve)
        origin, direction = (
            tuple(Fraction(float(value)) for value in np.asarray(values))
            for values in (curve.origin, curve.direction)
        )
        for endpoint, vertex in enumerate((first, last)):
            parameter = Fraction(float(geometry.edge_ranges[edge, endpoint]))
            point = tuple(
                Fraction(float(value))
                for value in np.asarray(geometry.vertex_points[vertex])
            )
            squared = sum(
                (
                    (a + parameter * b - c) ** 2
                    for a, b, c in zip(origin, direction, point, strict=True)
                ),
                Fraction(),
            )
            assert squared <= Fraction(gaps[edge][endpoint]) ** 2
    for face, (old_patch, patch) in enumerate(
        zip(source.patches, model.patches, strict=True)
    ):
        assert isinstance(old_patch, PlanePatch) and isinstance(patch, PlanePatch)
        normal = np.sign(
            np.cross(np.asarray(old_patch.first_axis), np.asarray(old_patch.second_axis))
        ) * int(np.asarray(source.orientation)[face])
        exact_height = (
            sum(
                (
                    Fraction(float(a)) * Fraction(float(b))
                    for a, b in zip(normal, np.asarray(old_patch.origin), strict=True)
                ),
                Fraction(),
            )
            + delta
        )
        represented_height = sum(
            (
                Fraction(float(a)) * Fraction(float(b))
                for a, b in zip(normal, np.asarray(patch.origin), strict=True)
            ),
            Fraction(),
        )
        assert abs(exact_height - represented_height) <= Fraction(
            float(evidence.face_rounding_bounds[face])
        )


def test_exactly_representable_offset_preserves_absent_intrinsic_gaps() -> None:
    result = phx.geometry.brep_offset(_source("box"), 0.25, tessellation=_LAZY)
    assert result.model.require_geometry().edge_vertex_gap_bounds is None
    assert np.all(np.asarray(result.evidence.vertex_rounding_bounds) == 0.0)
    assert np.all(np.asarray(result.evidence.face_rounding_bounds) == 0.0)
