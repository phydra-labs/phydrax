"""Independent hand-authored DXF fixtures for the native planar codec."""

import math
from pathlib import Path

import mpmath
import pytest

from phydrax.geometry.brep._constructors import ProfileCurveUse
from phydrax.geometry.brep._patches import EllipseCurve
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


def _quarter_ellipse(corner: tuple[float, float]) -> bytes:
    # ELLIPSE 0 -> fl(pi/2) with semi-axes 2 and 1, closed by two lines.
    return _drawing(
        "0\nELLIPSE\n10\n0.0\n20\n0.0\n30\n0.0\n11\n2.0\n21\n0.0\n31\n0.0\n"
        "40\n0.5\n41\n0.0\n42\n1.5707963267948966\n"
        f"0\nLINE\n10\n{corner[0]!r}\n20\n{corner[1]!r}\n30\n0.0\n"
        "11\n0.0\n21\n0.0\n31\n0.0\n"
        "0\nLINE\n10\n0.0\n20\n0.0\n30\n0.0\n11\n2.0\n21\n0.0\n31\n0.0\n"
    )


def test_float_quarter_angle_keeps_its_literal_ellipse_endpoint() -> None:
    gap = 1e-15
    with mpmath.workprec(200):
        parameter = mpmath.mpf(1.5707963267948966)
        exact = (2 * mpmath.cos(parameter), mpmath.sin(parameter))
        literal = (float(exact[0]), float(exact[1]))
        displacement = mpmath.hypot(exact[0] - literal[0], exact[1] - literal[1])
        assert 0 < displacement <= gap
    imported = decode_dxf_bytes(
        _quarter_ellipse(literal), DxfImportPolicy(maximum_junction_gap=gap)
    )
    assert [loss.category for loss in imported.report.losses] == ["transformed"]
    # fl(pi/2) is not an exact quarter turn: the exact axis point (0, 1) lies
    # 1.2e-16 from the authored endpoint and needs a declared gap.
    with pytest.raises(DxfImportError):
        decode_dxf_bytes(_quarter_ellipse((0.0, 1.0)), DxfImportPolicy())
    axis_corner = decode_dxf_bytes(
        _quarter_ellipse((0.0, 1.0)), DxfImportPolicy(maximum_junction_gap=gap)
    )
    assert [loss.category for loss in axis_corner.report.losses] == ["transformed"]


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
