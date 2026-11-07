"""Independent hand-authored DXF fixtures for the native planar codec."""

import math
from pathlib import Path

import mpmath
import numpy as np
import pytest

from phydrax.geometry.brep._constructors import ProfileCurveUse
from phydrax.geometry.brep._patches import BSplineCurve, EllipseCurve
from phydrax.interchange._dxf import (
    decode_dxf_bytes,
    DxfImportError,
    DxfImportPolicy,
    encode_dxf,
    read_dxf,
    write_dxf,
)
from phydrax.units import MILLIMETER


def _drawing(entities: str) -> bytes:
    return (
        "0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n70\n4\n0\nENDSEC\n"
        "0\nSECTION\n2\nENTITIES\n" + entities + "0\nENDSEC\n0\nEOF\n"
    ).encode("ascii")


# Hand-authored closed wire: line, circular arc, line, rational bulge and
# quadratic B-spline.  The controls and knots are independent of the writer.
MIXED = """0
LINE
10
0
20
0
11
4
21
0
0
ARC
10
4
20
1
40
1
50
270
51
90
0
LINE
10
4
20
2
11
2
21
2
0
LWPOLYLINE
90
2
70
0
10
2
20
2
42
1
10
0
20
2
0
SPLINE
70
0
71
2
72
6
73
3
74
0
40
0
40
0
40
0
40
1
40
1
40
1
10
0
20
2
10
-1
20
1
10
0
20
0
"""


def test_mixed_independent_source_round_trips_through_native_codec(
    tmp_path: Path,
) -> None:
    policy = DxfImportPolicy()
    source = _drawing(MIXED)
    imported = decode_dxf_bytes(source, policy)
    assert imported.coordinate_unit == MILLIMETER
    assert len(imported.profile.outer.segments) == 5
    assert imported.report.valid

    encoded = encode_dxf(imported.profile, coordinate_unit=MILLIMETER)
    restored = decode_dxf_bytes(encoded, policy)
    assert restored.coordinate_unit == MILLIMETER
    assert len(restored.profile.outer.segments) == 5
    assert restored.report.valid

    source_path = tmp_path / "fixture.dxf"
    source_path.write_bytes(source)
    indexed = read_dxf(source_path, policy, trusted_root=tmp_path)
    assert len(indexed.profile.outer.segments) == 5

    destination = tmp_path / "roundtrip.dxf"
    published = write_dxf(imported.profile, destination, coordinate_unit=MILLIMETER)
    assert published.path == destination
    assert published.report.valid
    assert decode_dxf_bytes(destination.read_bytes(), policy).report.valid


def test_spline_declared_count_cannot_drive_allocation() -> None:
    malformed = """0
SPLINE
70
0
71
2
72
1000000000
73
3
74
0
40
0
40
0
40
0
40
1
40
1
40
1
10
0
20
0
10
1
20
1
10
2
20
0
"""
    with pytest.raises(DxfImportError, match="declared counts disagree"):
        decode_dxf_bytes(_drawing(malformed), DxfImportPolicy())


# Exact axis points of the ellipse centered at 0 with major (2, 0), ratio 0.5.
_AXIS_POINTS = ((2.0, 0.0), (0.0, 1.0), (-2.0, 0.0), (0.0, -1.0), (2.0, 0.0))


def _right_angle(k: int) -> float:
    with mpmath.workprec(300):
        return float(mpmath.mpf(k) * mpmath.pi / 2)


def _wedge(first: float, last: float, k_first: int, k_last: int) -> bytes:
    # ELLIPSE first -> last closed through the center by lines that meet the
    # exact axis points of the right angles k_first and k_last.
    end, start = _AXIS_POINTS[k_last], _AXIS_POINTS[k_first]
    return _drawing(
        "0\nELLIPSE\n10\n0.0\n20\n0.0\n30\n0.0\n11\n2.0\n21\n0.0\n31\n0.0\n"
        f"40\n0.5\n41\n{first!r}\n42\n{last!r}\n"
        f"0\nLINE\n10\n{end[0]!r}\n20\n{end[1]!r}\n30\n0.0\n"
        "11\n0.0\n21\n0.0\n31\n0.0\n"
        "0\nLINE\n10\n0.0\n20\n0.0\n30\n0.0\n"
        f"11\n{start[0]!r}\n21\n{start[1]!r}\n31\n0.0\n"
    )


@pytest.mark.parametrize(
    ("k_first", "k_last"), ((0, 1), (0, 2), (0, 3), (1, 3), (3, 1), (2, 4))
)
def test_right_angle_parameters_denote_exact_axis_points(
    k_first: int, k_last: int
) -> None:
    policy = DxfImportPolicy()
    imported = decode_dxf_bytes(
        _wedge(_right_angle(k_first), _right_angle(k_last), k_first, k_last), policy
    )
    assert imported.report.losses == ()
    assert imported.profile.endpoint_gap_bounds is None
    (spline,) = (
        segment
        for segment in imported.profile.outer.segments
        if isinstance(segment, BSplineCurve)
    )
    controls = np.asarray(spline.control_points)
    assert {tuple(controls[0].tolist()), tuple(controls[-1].tolist())} == {
        _AXIS_POINTS[k_first],
        _AXIS_POINTS[k_last],
    }
    restored = decode_dxf_bytes(
        encode_dxf(imported.profile, coordinate_unit=MILLIMETER), policy
    )
    assert restored.report.losses == ()
    assert restored.profile.outer.vertices == imported.profile.outer.vertices
    assert [type(segment) for segment in restored.profile.outer.segments] == [
        type(segment) for segment in imported.profile.outer.segments
    ]


def _neighbors() -> list[tuple[float, float, int, int]]:
    cases = []
    for k_first, k_last in ((0, 1), (0, 2), (1, 3), (3, 1), (2, 4)):
        first, last = _right_angle(k_first), _right_angle(k_last)
        cases += [
            (first, math.nextafter(last, 0.0), k_first, k_last),
            (math.nextafter(first, math.inf), last, k_first, k_last),
        ]
        if k_first:
            cases.append((math.nextafter(first, -math.inf), last, k_first, k_last))
        if k_last != 4:
            cases.append((first, math.nextafter(last, math.inf), k_first, k_last))
    return cases


@pytest.mark.parametrize("k_last", (1, 2))
def test_signed_zero_start_keeps_the_analytic_carrier(k_last: int) -> None:
    # -0.0 is not bit-equal to the right angle 0, so no quarter arc is formed;
    # its analytic end is nevertheless exact and the right-angle end is rooted.
    imported = decode_dxf_bytes(
        _wedge(-0.0, _right_angle(k_last), 0, k_last), DxfImportPolicy()
    )
    assert imported.report.losses == ()
    assert any(
        isinstance(segment, ProfileCurveUse)
        for segment in imported.profile.outer.segments
    )


@pytest.mark.parametrize(("first", "last", "k_first", "k_last"), _neighbors())
def test_neighboring_parameters_keep_declared_gap_closure(
    first: float, last: float, k_first: int, k_last: int
) -> None:
    source = _wedge(first, last, k_first, k_last)
    with pytest.raises(DxfImportError):
        decode_dxf_bytes(source, DxfImportPolicy())
    imported = decode_dxf_bytes(source, DxfImportPolicy(maximum_junction_gap=1e-14))
    assert [loss.category for loss in imported.report.losses] == ["transformed"]


def test_parameter_past_the_full_turn_is_out_of_period() -> None:
    source = _wedge(_right_angle(2), math.nextafter(_right_angle(4), math.inf), 2, 4)
    with pytest.raises(DxfImportError, match="one declared period"):
        decode_dxf_bytes(source, DxfImportPolicy(maximum_junction_gap=1e-14))


def _mixed_right_angles() -> list[tuple[float, float, int, int]]:
    cases = []
    for k in (1, 2, 3):
        # The general end lies one ulp off the neighboring axis parameter.
        before = _right_angle(k - 1)
        start = math.nextafter(before, math.inf) if k > 1 else math.nextafter(0.0, 1.0)
        cases.append((start, _right_angle(k), k - 1, k))
        cases.append(
            (_right_angle(k), math.nextafter(_right_angle(k + 1), 0.0), k, k + 1)
        )
    return cases


@pytest.mark.parametrize(("first", "last", "k_first", "k_last"), _mixed_right_angles())
def test_single_right_angle_end_joins_at_its_exact_axis_point(
    first: float, last: float, k_first: int, k_last: int
) -> None:
    source = _wedge(first, last, k_first, k_last)
    with pytest.raises(DxfImportError):
        decode_dxf_bytes(source, DxfImportPolicy())
    imported = decode_dxf_bytes(source, DxfImportPolicy(maximum_junction_gap=1e-14))
    assert [loss.category for loss in imported.report.losses] == ["transformed"]
    rows = imported.profile.endpoint_gap_bounds
    assert rows is not None
    (use,) = (
        index
        for index, segment in enumerate(imported.profile.outer.segments)
        if isinstance(segment, ProfileCurveUse)
    )
    # The right-angle end is the exact axis point; only the general end has a gap.
    assert min(rows[0][use]) == 0.0 < max(rows[0][use]) <= 1e-14


_TWO_PI = 2 * math.pi


def _ellipse(first: float, last: float) -> bytes:
    return _drawing(
        "0\nELLIPSE\n10\n0.0\n20\n0.0\n30\n0.0\n11\n2.0\n21\n0.0\n31\n0.0\n"
        f"40\n0.5\n41\n{first!r}\n42\n{last!r}\n"
    )


def test_full_ellipse_convention_is_a_closed_periodic_carrier() -> None:
    imported = decode_dxf_bytes(_ellipse(0.0, _TWO_PI), DxfImportPolicy())
    assert imported.report.losses == ()
    assert imported.profile.endpoint_gap_bounds is None
    loop = imported.profile.outer
    (segment,) = loop.segments
    assert loop.vertices == ((2.0, 0.0),)
    assert isinstance(segment, ProfileCurveUse)
    assert isinstance(segment.curve, EllipseCurve)
    assert segment.curve.native_period_symbol() == "two_pi"
    assert (segment.first, segment.last) == (0.0, _TWO_PI)


@pytest.mark.parametrize(
    ("first", "last"),
    (
        (0.0, math.nextafter(_TWO_PI, 0.0)),
        (math.nextafter(0.0, 1.0), _TWO_PI),
    ),
    ids=("short-end", "late-start"),
)
def test_other_ellipse_parameters_keep_declared_gap_closure(
    first: float, last: float
) -> None:
    with pytest.raises(DxfImportError):
        decode_dxf_bytes(_ellipse(first, last), DxfImportPolicy())
    imported = decode_dxf_bytes(
        _ellipse(first, last), DxfImportPolicy(maximum_junction_gap=1e-14)
    )
    assert [loss.category for loss in imported.report.losses] == ["transformed"]


def test_full_ellipse_and_circle_round_trip_as_closed_carriers() -> None:
    policy = DxfImportPolicy()
    source = decode_dxf_bytes(_ellipse(0.0, _TWO_PI), policy)
    restored = decode_dxf_bytes(
        encode_dxf(source.profile, coordinate_unit=MILLIMETER), policy
    )
    assert restored.report.losses == ()
    assert restored.profile.endpoint_gap_bounds is None
    (segment,) = restored.profile.outer.segments
    (original,) = source.profile.outer.segments
    assert restored.profile.outer.vertices == ((2.0, 0.0),)
    assert isinstance(segment, ProfileCurveUse)
    assert isinstance(original, ProfileCurveUse)
    assert (segment.first, segment.last) == (0.0, _TWO_PI)
    assert segment.curve.identity_payload() == original.curve.identity_payload()
    circle = decode_dxf_bytes(
        _drawing("0\nCIRCLE\n10\n1.0\n20\n2.0\n30\n0.0\n40\n3.0\n"), policy
    )
    circle_restored = decode_dxf_bytes(
        encode_dxf(circle.profile, coordinate_unit=MILLIMETER), policy
    )
    assert circle_restored.report.losses == ()
    assert circle_restored.profile.outer == circle.profile.outer
