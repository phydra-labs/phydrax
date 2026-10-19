"""Hand-authored DXF sources and independent closed-form conic references."""

import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep._constructors import (
    brep_extrusion,
    ProfileArc,
    ProfileCurveUse,
)
from phydrax.geometry.brep._patches import BSplineCurve, CircleCurve
from phydrax.interchange._catalog import format_capabilities
from phydrax.interchange._dxf import (
    decode_dxf_bytes,
    DxfImportError,
    DxfImportPolicy,
    encode_dxf,
    read_dxf,
    write_dxf,
)
from phydrax.interchange._report import AdapterStatus
from phydrax.units import MILLIMETER
from tests._support.cad_models import solid_volumes


# Empirical `test_full_ellipse_exact_rational_identity`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_DXF_ELLIPSE_RATIONAL_POINT_ATOL = 1e-15
# Empirical `test_extrusion_closed_form_area`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_DXF_EXTRUDED_AREA_RTOL = 1e-12


def _drawing(entities: str, *, unit: bool = True) -> bytes:
    header = "0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n70\n4\n0\nENDSEC\n" if unit else ""
    return (
        header + "0\nSECTION\n2\nENTITIES\n" + entities + "0\nENDSEC\n0\nEOF\n"
    ).encode("ascii")


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


RECTANGLE = """0
LWPOLYLINE
90
4
70
1
10
0
20
0
10
4
20
0
10
4
20
2
10
0
20
2
"""


ELLIPSE = """0
ELLIPSE
10
0
20
0
11
2
21
0
40
0.5
41
0
42
6.283185307179586
"""


def test_mixed_exact_round_trip(tmp_path: Path) -> None:
    policy = DxfImportPolicy()
    first = decode_dxf_bytes(_drawing(MIXED), policy)
    path = tmp_path / "mixed.dxf"
    exported = write_dxf(
        first.profile, path, coordinate_unit=first.coordinate_unit, policy=policy
    )
    second = read_dxf(path, policy, trusted_root=tmp_path)
    assert (
        first.report.status
        == second.report.status
        == exported.report.status
        == AdapterStatus.LOSSLESS
    )
    assert (
        first.coordinate_unit.unit_id
        == second.coordinate_unit.unit_id
        == MILLIMETER.unit_id
    )
    assert first.profile.endpoint_gap_bounds is None
    assert first.profile.outer.vertices == second.profile.outer.vertices
    for source, target in zip(
        first.profile.outer.segments, second.profile.outer.segments, strict=True
    ):
        assert type(source) is type(target)
        if isinstance(source, BSplineCurve) and isinstance(target, BSplineCurve):
            np.testing.assert_array_equal(source.control_points, target.control_points)
            np.testing.assert_array_equal(source.weights, target.weights)
            np.testing.assert_array_equal(source.knots, target.knots)
            assert source.degree == target.degree
        elif isinstance(source, ProfileArc) and isinstance(target, ProfileArc):
            assert source == target
    coverage = dict(first.report.source_profile.qualifiers)
    assert len(json.loads(coverage["coverage-consumed"])) == 5
    assert json.loads(coverage["coverage-refused"]) == []


def test_extrusion_closed_form_area() -> None:
    imported = decode_dxf_bytes(_drawing(RECTANGLE), DxfImportPolicy())
    solid = brep_extrusion(
        imported.profile,
        (0.0, 0.0, 3.0),
        coordinate_contract=SpatialCoordinateContract(imported.coordinate_unit),
    )
    assert solid.geometry is not None
    assert solid.geometry.edge_vertex_gap_bounds is None
    assert solid_volumes(solid)[0] / 3.0 == pytest.approx(
        8.0, rel=_DXF_EXTRUDED_AREA_RTOL
    )


def test_full_ellipse_exact_rational_identity() -> None:
    imported = decode_dxf_bytes(_drawing(ELLIPSE), DxfImportPolicy())
    curve = imported.profile.outer.segments[0]
    assert isinstance(curve, BSplineCurve)
    # Exact rational quarter parametrization: x=2(1-t²)/(1+t²), y=2t/(1+t²).
    t = 0.5
    point = np.asarray(curve.evaluate(jnp.asarray(t, dtype=jnp.float64)))
    np.testing.assert_allclose(
        point,
        (2 * (1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)),
        rtol=0.0,
        atol=_DXF_ELLIPSE_RATIONAL_POINT_ATOL,
    )
    restored = decode_dxf_bytes(
        encode_dxf(
            imported.profile, coordinate_unit=MILLIMETER, policy=DxfImportPolicy()
        ),
        DxfImportPolicy(),
    )
    target = restored.profile.outer.segments[0]
    assert isinstance(target, BSplineCurve)
    np.testing.assert_array_equal(curve.control_points, target.control_points)
    np.testing.assert_array_equal(curve.weights, target.weights)
    np.testing.assert_array_equal(curve.knots, target.knots)


def test_fit_point_spline_refuses() -> None:
    fit_only = (
        "0\nSPLINE\n70\n0\n71\n3\n72\n0\n73\n0\n74\n2\n11\n0\n21\n0\n11\n1\n21\n1\n"
    )
    with pytest.raises(DxfImportError, match="fit-point-only") as failure:
        decode_dxf_bytes(_drawing(fit_only), DxfImportPolicy())
    assert failure.value.report.status == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
    coverage = dict(failure.value.report.source_profile.qualifiers)
    assert json.loads(coverage["coverage-refused"]) == ["SPLINE:0"]


def test_units_are_required_and_caller_can_declare() -> None:
    data = _drawing(RECTANGLE, unit=False)
    with pytest.raises(DxfImportError, match="unit must be declared"):
        decode_dxf_bytes(data, DxfImportPolicy())
    assert (
        decode_dxf_bytes(
            data, DxfImportPolicy(), source_length_unit=MILLIMETER
        ).coordinate_unit.unit_id
        == MILLIMETER.unit_id
    )


def test_open_endpoints_are_reported() -> None:
    with pytest.raises(DxfImportError, match="open or ambiguous") as failure:
        decode_dxf_bytes(
            _drawing("0\nLINE\n10\n0\n20\n0\n11\n1\n21\n0\n"), DxfImportPolicy()
        )
    assert failure.value.open_endpoints == ((0.0, 0.0), (1.0, 0.0))


def test_coverage_and_resource_limits() -> None:
    annotated = RECTANGLE + "0\nTEXT\n10\n1\n20\n1\n1\nannotation\n"
    imported = decode_dxf_bytes(_drawing(annotated), DxfImportPolicy())
    coverage = dict(imported.report.source_profile.qualifiers)
    assert json.loads(coverage["coverage-ignored-by-design"]) == ["TEXT:1"]
    with pytest.raises(DxfImportError, match="maximum_bytes"):
        decode_dxf_bytes(_drawing(RECTANGLE), DxfImportPolicy(maximum_bytes=8))
    with pytest.raises(DxfImportError, match="maximum_groups"):
        decode_dxf_bytes(_drawing(RECTANGLE), DxfImportPolicy(maximum_groups=4))
    with pytest.raises(DxfImportError, match="Unsupported required DXF entity"):
        decode_dxf_bytes(_drawing("0\nINSERT\n2\nblock\n"), DxfImportPolicy())


def _angular_drawing() -> bytes:
    # Independent authored points are nearest binary64 coordinates of 30°/60°.
    a = (math.sqrt(3.0) / 2.0, 0.5)
    b = (0.5, math.sqrt(3.0) / 2.0)
    lines = f"0\nLINE\n10\n0\n20\n0\n11\n{a[0]:.17g}\n21\n{a[1]:.17g}\n0\nARC\n10\n0\n20\n0\n40\n1\n50\n30\n51\n60\n0\nLINE\n10\n{b[0]:.17g}\n20\n{b[1]:.17g}\n11\n0\n21\n0\n"
    return _drawing(lines)


def _decimal_trig(angle: Decimal) -> tuple[Decimal, Decimal]:
    cosine = sum(
        (
            (-1) ** i * angle ** (2 * i) / Decimal(math.factorial(2 * i))
            for i in range(100)
        ),
        Decimal(0),
    )
    sine = sum(
        (
            (-1) ** i * angle ** (2 * i + 1) / Decimal(math.factorial(2 * i + 1))
            for i in range(100)
        ),
        Decimal(0),
    )
    return cosine, sine


def test_angular_junction_default_refusal_and_declared_gap() -> None:
    data = _angular_drawing()
    with pytest.raises(DxfImportError) as failure:
        decode_dxf_bytes(data, DxfImportPolicy())
    refusal = dict(failure.value.report.source_profile.qualifiers)
    assert json.loads(refusal["junction-gap-bounds"])
    imported = decode_dxf_bytes(data, DxfImportPolicy(maximum_junction_gap=1e-12))
    assert imported.report.status == AdapterStatus.DECLARED_LOSS
    declaration = dict(imported.report.source_profile.qualifiers)
    assert float(declaration["declared-maximum-junction-gap"]) == 1e-12
    assert imported.profile.endpoint_gap_bounds is not None
    solid = brep_extrusion(
        imported.profile,
        (0.0, 0.0, 1.0),
        coordinate_contract=SpatialCoordinateContract(MILLIMETER),
    )
    assert solid.geometry is not None
    assert solid.geometry.edge_vertex_gap_bounds is not None
    with localcontext() as context:
        context.prec = 80
        # An independent 80-digit Taylor reference evaluates the authored binary64 parameter.
        for index, segment in enumerate(imported.profile.outer.segments):
            if not isinstance(segment, ProfileCurveUse):
                continue
            vertices = imported.profile.outer.vertices
            for endpoint, angle in enumerate((segment.first, segment.last)):
                cosine, sine = _decimal_trig(Decimal.from_float(angle))
                vertex = vertices[(index + endpoint) % len(vertices)]
                displacement = (
                    (cosine - Decimal.from_float(vertex[0])) ** 2
                    + (sine - Decimal.from_float(vertex[1])) ** 2
                ).sqrt()
                bound = imported.profile.endpoint_gap_bounds[0][index][endpoint]
                assert Decimal.from_float(bound) >= displacement
        geometry = solid.geometry
        gap_rows = geometry.edge_vertex_gap_bounds
        assert gap_rows is not None
        points = np.asarray(geometry.vertex_points)
        ranges = np.asarray(geometry.edge_ranges)
        for edge, curve_index in enumerate(geometry.edge_curves):
            curve = geometry.curves[curve_index]
            if not isinstance(curve, CircleCurve):
                continue
            center = np.asarray(curve.center)
            first_axis, second_axis = (
                np.asarray(curve.first_axis),
                np.asarray(curve.second_axis),
            )
            radius = Decimal.from_float(float(curve.radius))
            for endpoint, vertex in enumerate(geometry.edge_vertices[edge]):
                cosine, sine = _decimal_trig(
                    Decimal.from_float(float(ranges[edge, endpoint]))
                )
                squared = sum(
                    (
                        (
                            Decimal.from_float(float(center[i]))
                            + radius
                            * (
                                cosine * Decimal.from_float(float(first_axis[i]))
                                + sine * Decimal.from_float(float(second_axis[i]))
                            )
                            - Decimal.from_float(float(points[vertex, i]))
                        )
                        ** 2
                        for i in range(3)
                    ),
                    Decimal(0),
                )
                bound = gap_rows[edge][endpoint]
                assert Decimal.from_float(bound) >= squared.sqrt()
        published = max(
            bound for row in solid.geometry.edge_vertex_gap_bounds for bound in row
        )
    assert published <= 1e-12


def test_dxf_format_registered() -> None:
    capabilities = format_capabilities()
    assert any(
        item.profile.format == "dxf" and ".dxf" in item.extensions
        for item in capabilities
    )


def test_circle_hole_containment_and_orientation() -> None:
    entities = "0\nCIRCLE\n10\n0\n20\n0\n40\n1\n0\nCIRCLE\n10\n0\n20\n0\n40\n2\n"
    imported = decode_dxf_bytes(_drawing(entities), DxfImportPolicy())
    assert imported.profile.outer.vertices == ((2.0, 0.0),)
    assert imported.profile.holes[0].vertices == ((1.0, 0.0),)
    outer = imported.profile.outer.segments[0]
    hole = imported.profile.holes[0].segments[0]
    assert isinstance(outer, ProfileArc) and isinstance(hole, ProfileArc)
    assert outer.counterclockwise and not hole.counterclockwise
    assert imported.profile.endpoint_gap_bounds is None


def test_rational_bulge_preserves_center_without_angle_fitting() -> None:
    entities = (
        "0\nLWPOLYLINE\n90\n2\n70\n0\n10\n0\n20\n0\n42\n0.5\n10\n4\n20\n0\n"
        "0\nLINE\n10\n4\n20\n0\n11\n0\n21\n0\n"
    )
    first = decode_dxf_bytes(_drawing(entities), DxfImportPolicy())
    arc = first.profile.outer.segments[0]
    assert isinstance(arc, ProfileArc)
    assert arc.center == (2.0, 1.5)
    second = decode_dxf_bytes(
        encode_dxf(first.profile, coordinate_unit=MILLIMETER, policy=DxfImportPolicy()),
        DxfImportPolicy(),
    )
    assert first.profile.outer.vertices == second.profile.outer.vertices
    assert first.profile.outer.segments == second.profile.outer.segments


def test_gap_free_model_matches_prechange_identity() -> None:
    from phydrax.geometry.brep._constructors import brep_box
    from phydrax.interchange._cad_archive import load_brep_archive

    fixture = Path(__file__).parents[2] / "data" / "brep_no_gap_prechange.phx"
    old = load_brep_archive(fixture)
    new = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 2.0, 3.0),
        coordinate_contract=SpatialCoordinateContract(MILLIMETER),
    )
    assert old.coordinate_contract.length_unit.unit_id == MILLIMETER.unit_id
    assert old.model_id == new.model_id
    assert old.require_geometry().geometry_id == new.require_geometry().geometry_id
    assert old.face_ids == new.face_ids
    assert old.edge_ids == new.edge_ids
    assert old.vertex_ids == new.vertex_ids
    assert old.solid_ids == new.solid_ids


def test_integer_group_does_not_truncate_fractional_count() -> None:
    malformed = "0\nLWPOLYLINE\n90\n2.5\n70\n0\n10\n0\n20\n0\n10\n1\n20\n0\n"
    with pytest.raises(DxfImportError, match="invalid literal"):
        decode_dxf_bytes(_drawing(malformed), DxfImportPolicy())


def test_mutable_source_bytes_refuse() -> None:
    with pytest.raises(TypeError, match="bytes"):
        decode_dxf_bytes(bytearray(_drawing(RECTANGLE)), DxfImportPolicy())  # ty: ignore[invalid-argument-type]


def test_descriptor_reader_refuses_symbolic_link(tmp_path: Path) -> None:
    target = tmp_path / "authored.dxf"
    target.write_bytes(_drawing(RECTANGLE))
    link = tmp_path / "alias.dxf"
    link.symlink_to(target.name)
    with pytest.raises(DxfImportError):
        read_dxf(link, DxfImportPolicy(), trusted_root=tmp_path)


def test_request_budget_admits_work_before_decoding() -> None:
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget

    with HostBatchTicket(1, bytes_upper=1_000_000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=1, maximum_memory_bytes=1_000_000
        )
        with budget.activate(), pytest.raises(DxfImportError) as failure:
            decode_dxf_bytes(
                _drawing(RECTANGLE), DxfImportPolicy(), coordinate_budget=budget
            )
        assert budget.work_units == 0
        coverage = dict(failure.value.report.source_profile.qualifiers)
        assert json.loads(coverage["coverage-consumed"]) == []
