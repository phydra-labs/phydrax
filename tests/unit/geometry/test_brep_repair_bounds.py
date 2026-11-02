from __future__ import annotations

from fractions import Fraction as F
from typing import cast

import pytest

from phydrax.geometry.brep._patches import CircleCurve, LineCurve, TorusPatch
from phydrax.geometry.brep._repair_bounds import (
    circle_displacement_bound,
    torus_displacement_bound,
    torus_parameter_displacement_bound,
    vertex_displacement_bound,
)
from phydrax.geometry.brep._source_repair_trim import (
    annulus_bound,
    annulus_enclosure_bounds,
    AnnulusBoundary,
    chart_connector_bound,
    circle_coedge_completion_bounds,
    completed_profile_enclosure,
    cylinder_enclosure_bounds,
    cylinder_strip_bound,
    line_coedge_completion_bounds,
    periodic_partition_bound,
    ProfileEnclosure,
    StripBoundary,
    torus_enclosure_bounds,
    torus_strip_bound,
)


ZERO = (F(0), F(0), F(0))
FRAME = ((F(1), F(0), F(0)), (F(0), F(1), F(0)), (F(0), F(0), F(1)))


def point(value: F) -> tuple[F, F]:
    return value, value


def strip(lower: F, upper: F, seam: F = F(0)) -> StripBoundary:
    return StripBoundary(
        point(lower), point(upper), point(seam), frozenset({1}), frozenset({2}), 3
    )


def circle(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 1.0,
    first: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> CircleCurve:
    return CircleCurve(center, first, (0.0, 1.0, 0.0), radius)


def torus(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    major: float = 3.0,
    minor: float = 1.0,
    first: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> TorusPatch:
    return TorusPatch(center, first, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), major, minor)


def test_whole_carrier_bounds_are_positive_exact_rationals() -> None:
    assert vertex_displacement_bound(ZERO, (F(3), F(4), F(0))) == F(5)
    assert vertex_displacement_bound(ZERO, ZERO) == F(0)
    source_circle = circle()
    target_circle = circle(center=(0.25, 0.0, 0.0), radius=1.5)
    bound = circle_displacement_bound(source_circle, target_circle)
    assert isinstance(bound, F) and bound == F(3, 4)
    # At the positive first-axis point the entire bound is attained.
    assert bound >= F(1, 4) + F(1, 2)
    source_torus = torus()
    target_torus = torus(center=(0.25, 0.0, 0.0), major=3.5, minor=1.25)
    bound = torus_displacement_bound(source_torus, target_torus)
    assert isinstance(bound, F) and bound == F(1)
    assert torus_parameter_displacement_bound(source_torus, F(1, 4), F(1, 2)) == F(3, 2)


def test_carrier_bounds_reject_incompatible_frames_and_inexact_inputs() -> None:
    with pytest.raises(ValueError, match="axes"):
        circle_displacement_bound(circle(), circle(first=(-1.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="axes"):
        torus_displacement_bound(torus(), torus(first=(-1.0, 0.0, 0.0)))
    with pytest.raises(TypeError):
        vertex_displacement_bound((cast(F, 0.0),), (F(0),))
    with pytest.raises(TypeError):
        torus_parameter_displacement_bound(torus(), cast(F, 0.25), F(0))


def test_strip_and_annulus_correspondence_bounds() -> None:
    source, target = strip(F(0), F(2)), strip(F(1, 4), F(9, 4), F(1, 8))
    cylinder = cylinder_strip_bound(ZERO, ZERO, FRAME, FRAME, F(2), F(2), source, target)
    toroidal = torus_strip_bound(
        ZERO, ZERO, FRAME, FRAME, F(3), F(3), F(1), F(1), source, target
    )
    assert cylinder == F(1, 2)
    assert toroidal == F(3, 4)
    first = AnnulusBoundary(F(1), F(3), frozenset({1}), frozenset({2}))
    second = AnnulusBoundary(F(5, 4), F(7, 2), frozenset({1}), frozenset({2}))
    assert annulus_bound(ZERO, ZERO, FRAME, FRAME, first, second) == F(1, 2)
    with pytest.raises(ValueError, match="frame"):
        cylinder_strip_bound(
            ZERO, ZERO, FRAME, (FRAME[1], FRAME[0], FRAME[2]), F(2), F(2), source, target
        )
    with pytest.raises(ValueError, match="incidence"):
        annulus_bound(
            ZERO,
            ZERO,
            FRAME,
            FRAME,
            first,
            AnnulusBoundary(F(1), F(3), frozenset({1}), frozenset({1})),
        )


def test_partition_and_strip_reject_ambiguous_limits() -> None:
    assert periodic_partition_bound(
        (point(F(0)), point(F(2))), (point(F(1, 4)), point(F(9, 4))), (1, 2), (1, 2)
    ) == F(1, 4)
    with pytest.raises(ValueError, match="ordered"):
        periodic_partition_bound(
            ((F(0), F(1)), (F(1), F(2))), (point(F(0)), point(F(2))), (1, 2), (1, 2)
        )
    with pytest.raises(ValueError, match="partition"):
        periodic_partition_bound(
            (point(F(0)), point(F(2))), (point(F(0)), point(F(2))), (1, 2), (2, 1)
        )
    ambiguous = StripBoundary(
        (F(0), F(1)), (F(1), F(2)), point(F(0)), frozenset({1}), frozenset({2}), 3
    )
    with pytest.raises(ValueError, match="order"):
        ambiguous.validate(periodic_profile=False)
    with pytest.raises(ValueError, match="period"):
        strip(F(0), F(7)).validate(periodic_profile=True)


def test_completed_profile_enclosure_and_directed_face_bounds() -> None:
    source = completed_profile_enclosure((F(1), F(5, 4)), (F(11, 4), F(3)), radial=True)
    assert source == ProfileEnclosure(
        point(F(1)), point(F(5, 4)), point(F(11, 4)), point(F(3))
    )
    target = strip(F(9, 8), F(23, 8))
    cylinder = cylinder_enclosure_bounds(
        ZERO, ZERO, FRAME, FRAME, F(2), F(2), source, target
    )
    toroidal = torus_enclosure_bounds(
        ZERO, ZERO, FRAME, FRAME, F(3), F(3), F(2), F(2), source, target
    )
    annulus = annulus_enclosure_bounds(
        ZERO,
        ZERO,
        FRAME,
        FRAME,
        source,
        AnnulusBoundary(F(9, 8), F(23, 8), frozenset({1}), frozenset({2})),
    )
    assert (cylinder.source_to_target, cylinder.target_to_source) == (F(1, 8), F(1, 8))
    assert (toroidal.source_to_target, toroidal.target_to_source) == (F(1, 4), F(1, 4))
    assert (annulus.source_to_target, annulus.target_to_source) == (F(1, 8), F(1, 8))
    assert all(
        isinstance(value, F)
        for value in (
            cylinder.source_to_target,
            toroidal.target_to_source,
            annulus.source_to_target,
        )
    )


def test_enclosure_rejects_unproved_order_and_incompatible_supports() -> None:
    with pytest.raises(ValueError, match="Nested"):
        completed_profile_enclosure((F(1), F(2)), (F(2), F(3)))
    with pytest.raises(ValueError, match="positive inner"):
        completed_profile_enclosure(point(F(0)), point(F(2)), radial=True)
    with pytest.raises(ValueError, match="turn"):
        completed_profile_enclosure(point(F(0)), point(F(7)), periodic=True)
    source = completed_profile_enclosure(point(F(1)), point(F(2)))
    with pytest.raises(ValueError, match="frame"):
        cylinder_enclosure_bounds(
            ZERO,
            ZERO,
            FRAME,
            (FRAME[1], FRAME[0], FRAME[2]),
            F(1),
            F(1),
            source,
            strip(F(1), F(2)),
        )
    with pytest.raises(ValueError, match="ring tori"):
        torus_enclosure_bounds(
            ZERO, ZERO, FRAME, FRAME, F(1), F(1), F(1), F(1), source, strip(F(1), F(2))
        )


def test_chart_connector_bounds_cover_interval_endpoint_uncertainty() -> None:
    bound = chart_connector_bound(
        ((F(0), F(1, 8)), point(F(0))), (point(F(1, 2)), (F(1, 4), F(3, 8))), F(2), F(4)
    )
    assert isinstance(bound, F) and bound == F(5, 2)
    with pytest.raises(ValueError, match="nonnegative"):
        chart_connector_bound(
            (point(F(0)), point(F(0))), (point(F(1)), point(F(1))), F(-1), F(1)
        )
    with pytest.raises(ValueError):
        chart_connector_bound(
            ((F(1), F(0)), point(F(0))), (point(F(1)), point(F(1))), F(1), F(1)
        )


def test_coedge_completion_bounds_include_both_connectors() -> None:
    source = LineCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    target = LineCurve((0.0, 0.25, 0.0), (1.0, 0.0, 0.0))
    endpoints = (point(F(0)), point(F(1)))
    shifted = (point(F(1, 8)), point(F(5, 4)))
    line = line_coedge_completion_bounds(source, target, endpoints, shifted)
    assert line.connectors == (F(3, 8), F(1, 2))
    assert line.occurrence == F(1, 2)
    circular = circle_coedge_completion_bounds(
        circle(), circle(radius=1.5), endpoints, shifted
    )
    assert circular.connectors == (F(5, 8), F(3, 4))
    assert circular.occurrence == F(3, 4)
    with pytest.raises(ValueError, match="ordering"):
        line_coedge_completion_bounds(source, target, (point(F(0)), point(F(0))), shifted)
    with pytest.raises(ValueError, match="axes"):
        circle_coedge_completion_bounds(
            circle(), circle(first=(-1.0, 0.0, 0.0)), endpoints, shifted
        )
