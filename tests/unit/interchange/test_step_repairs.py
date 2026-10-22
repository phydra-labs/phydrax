"""STEP repairs use authored fixtures and independent native box geometry."""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from phydrax._external_resource import ResourceLimits
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry.brep._gap_evidence import certify_brep_gaps
from phydrax.geometry.brep._intersection_curve import PeriodicPCurve
from phydrax.geometry.brep._patches import (
    CircleCurve,
    CylinderPatch,
    LineCurve,
    PlanePatch,
    SurfaceIsoparametricCurve,
    TorusPatch,
    curve_evaluate_host,
    surface_evaluate_host,
    surface_value_and_differential_host,
)
from phydrax.geometry.brep._constructors import brep_box, BRepTessellationPolicy
from phydrax.geometry.brep._query import prepare_brep_query
from phydrax.interchange._cad import (
    CadImportPolicy,
    CadInterchangeError,
    CadStage,
    explicit_carrier,
    StagedCoedge,
    StagedFace,
)
from phydrax.interchange._report import (
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
    compose_adapter_reports,
)
from phydrax.interchange._step import decode_step_bytes, read_step, write_step
from phydrax.interchange._step_repair import StepRepairPolicy
from tests._support.cad_models import CONTRACT, native_solid


POLICY = CadImportPolicy(
    CONTRACT,
    ResourceLimits(
        max_bytes=1 << 24,
        max_depth=64,
        max_nodes=200_000,
        max_attributes=4_000_000,
        max_losses=8,
    ),
)


@contextmanager
def _budget() -> Iterator[CoordinateEnclosureBudget]:
    with HostBatchTicket(100_000_000, bytes_upper=256 << 20) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=100_000_000,
            maximum_memory_bytes=256 << 20,
        )
        with budget.activate():
            yield budget


def _box_source(tmp_path: Path) -> str:
    path = tmp_path / "native-box.step"
    write_step(native_solid("box"), path)
    return path.read_text()


def _inverted_box(tmp_path: Path) -> bytes:
    source = _box_source(tmp_path)
    match = re.search(r"(MANIFOLD_SOLID_BREP\('[^']*',)#(\d+)(\))", source)
    assert match is not None
    shell = match.group(2)
    source = (
        source[: match.start()]
        + match.group(1)
        + "#999999"
        + match.group(3)
        + source[match.end() :]
    )
    source = source.replace(
        "ENDSEC;\nEND-ISO",
        f"#999999=ORIENTED_CLOSED_SHELL('',*,#{shell},.F.);\nENDSEC;\nEND-ISO",
    )
    return source.encode()


# Independent source: an open cylinder band bounded by full native circles.
# The longitudinal seam and its two uses are deliberately absent.
_CYLINDER = b"""ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('missing longitudinal seam'),'2;1');
FILE_NAME('band.step','2026-10-14T00:00:00',('PHYDRA'),('PHYDRA'),'handwritten','handwritten','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
#3=DIRECTION('',(1.,0.,0.));
#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);
#5=CYLINDRICAL_SURFACE('',#4,1.);
#6=CIRCLE('',#4,1.);
#7=CARTESIAN_POINT('',(0.,0.,2.));
#8=AXIS2_PLACEMENT_3D('',#7,#2,#3);
#9=CIRCLE('',#8,1.);
#10=CARTESIAN_POINT('',(1.,0.,0.));
#11=CARTESIAN_POINT('',(1.,0.,2.));
#12=VERTEX_POINT('',#10);
#13=VERTEX_POINT('',#11);
#14=EDGE_CURVE('',#12,#12,#6,.T.);
#15=EDGE_CURVE('',#13,#13,#9,.T.);
#16=ORIENTED_EDGE('',*,*,#14,.T.);
#17=ORIENTED_EDGE('',*,*,#15,.F.);
#18=EDGE_LOOP('',(#16));
#19=EDGE_LOOP('',(#17));
#20=FACE_OUTER_BOUND('',#18,.T.);
#21=FACE_BOUND('',#19,.T.);
#22=ADVANCED_FACE('',(#20,#21),#5,.T.);
#23=OPEN_SHELL('',(#22));
#24=SHELL_BASED_SURFACE_MODEL('',(#23));
#25=MANIFOLD_SURFACE_SHAPE_REPRESENTATION('',(#24),#29);
#26=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.));
#27=(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.));
#28=(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT());
#29=(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#26,#27,#28)) REPRESENTATION_CONTEXT('',''));
ENDSEC;
END-ISO-10303-21;
"""


def test_missing_longitudinal_seam_is_exact_and_recorded() -> None:
    native = decode_step_bytes(_CYLINDER, POLICY)
    assert native.report.status == AdapterStatus.LOSSLESS
    assert not native.report.losses
    assert any(
        "DERIVED:periodic-chart-identification" in label
        for _, label in native.provenance
    )
    with _budget() as budget:
        result = decode_step_bytes(
            _CYLINDER,
            POLICY,
            repair_policy=StepRepairPolicy(add_missing_seams=True),
            coordinate_budget=budget,
        )
    assert result.report.status == AdapterStatus.REPAIRED
    assert result.report.valid
    records = [loss for loss in result.report.losses if "step-repairs" in loss.path]
    assert len(records) == 1
    assert "#22" in records[0].path
    assert "longitudinal" in records[0].rationale
    geometry = result.model.require_geometry()
    assert len(geometry.edge_curves) == 3
    seam = geometry.coedge_edges[-1]
    assert geometry.coedge_edges.count(seam) == 2
    np.testing.assert_array_equal(
        geometry.vertex_points, ((1.0, 0.0, 0.0), (1.0, 0.0, 2.0))
    )
    bounds = np.asarray(result.model.parameter_bounds[0])
    assert np.all(bounds[0] <= (0.0, 0.0))
    assert np.all(bounds[1] >= (2 * np.pi, 2.0))


@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.parametrize("fixed", (0.0, 0.375))
def test_cylinder_isoline_export_retains_the_native_parameter(
    axis: int, fixed: float
) -> None:
    cylinder = CylinderPatch(
        (0.25, -0.5, 0.75), (0, 1, 0), (0, 0, 1), (1, 0, 0), 2.0
    )
    source = SurfaceIsoparametricCurve(
        cylinder, axis, fixed, parameter_range=(-0.5, 1.5)
    )
    carrier = explicit_carrier(source)
    assert isinstance(carrier, LineCurve if axis == 0 else CircleCurve)
    parameters = np.linspace(-0.5, 1.5, 17)
    np.testing.assert_allclose(
        curve_evaluate_host(carrier, parameters),
        curve_evaluate_host(source, parameters),
        rtol=0,
        atol=1e-14,
    )
    assert carrier.period == source.period


@pytest.mark.parametrize("torus", (False, True), ids=("cylinder", "torus"))
@pytest.mark.parametrize("reverse_carriers", (False, True))
def test_periodic_band_chart_seam_preserves_area_winding_and_source(
    torus: bool, reverse_carriers: bool
) -> None:
    patch = (
        TorusPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 3.0, 1.0)
        if torus
        else CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 3.0)
    )
    stage = CadStage(10.0, None)
    limits = (0.0, 0.5)
    rings = []
    direction = -1 if reverse_carriers else 1
    for index, value in enumerate(limits):
        point = surface_evaluate_host(patch, np.asarray((0.0, value)))
        vertex = stage.vertex(point, f"authored-vertex-{index}")
        radius = 3.0 + np.cos(value) if torus else 3.0
        height = np.sin(value) if torus else value
        edge = stage.edge(
            CircleCurve((0, 0, height), (1, 0, 0), (0, direction, 0), radius),
            vertex,
            vertex,
            f"authored-ring-{index}",
            parameter_range=(0.0, 2 * np.pi),
        )
        rings.append(
            [
                StagedCoedge(
                    edge, direction * (1 if index == 0 else -1), None, f"ring-{index}"
                )
            ]
        )
    face = StagedFace(patch, rings, 1, "band", "authored-face", True)
    loops, _ = stage._resolve_face(face)
    assert len(loops) == 1 and len(loops[0]) == 4
    loop = loops[0]
    assert [loop[index][0].edge for index in (0, 2)] == [0, 1]
    assert [loop[index][0].sense for index in (0, 2)] == [direction, -direction]
    assert len(stage.vertices) == 2 and len(stage.edge_curves) == 3
    assert stage.provenance[:4] == [
        ("vertex:0", "authored-vertex-0"),
        ("edge:0", "authored-ring-0"),
        ("vertex:1", "authored-vertex-1"),
        ("edge:1", "authored-ring-1"),
    ]
    assert loop[1][0].edge == loop[3][0].edge
    assert loop[1][0].sense == -loop[3][0].sense
    assert all(isinstance(curve, PeriodicPCurve) for _, curve in loop)
    for current, following in zip(loop, loop[1:] + loop[:1], strict=True):
        np.testing.assert_allclose(
            stage._ends(*current)[1], stage._ends(*following)[0], rtol=0, atol=1e-14
        )
    samples = np.linspace(*limits, 17)
    seam_points = [
        surface_evaluate_host(patch, curve_evaluate_host(loop[index][1], samples))
        for index in (1, 3)
    ]
    np.testing.assert_allclose(seam_points[0], seam_points[1], rtol=0, atol=2e-14)
    nodes, weights = np.polynomial.legendre.leggauss(16)
    v = limits[0] + (nodes + 1) * (limits[1] - limits[0]) / 2
    uv = np.stack((np.ones_like(v), v), axis=-1)
    _, differential = surface_value_and_differential_host(patch, uv)
    density = np.linalg.norm(
        np.cross(differential[..., 0], differential[..., 1]), axis=-1
    )
    area = 2 * np.pi * (limits[1] - limits[0]) / 2 * (weights @ density)
    expected = 2 * np.pi * (
        3 * (limits[1] - limits[0])
        + (np.sin(limits[1]) - np.sin(limits[0]) if torus else 0)
    )
    assert area == pytest.approx(expected, rel=1e-12)
    assert stage._signed_area(stage._loop_polygon(loop)) == pytest.approx(np.pi)


@pytest.mark.parametrize("defect", ("partial-turn", "near-period", "nonperiodic"))
def test_periodic_band_requires_exact_native_turn(defect: str) -> None:
    patch = (
        PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0))
        if defect == "nonperiodic"
        else CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    )
    stage = CadStage(2.0, None)
    rings = []
    for index in range(2):
        vertex = stage.vertex(np.asarray((1.0, 0.0, float(index))), f"v{index}")
        edge = stage.edge(
            CircleCurve((0, 0, index), (1, 0, 0), (0, 1, 0), 1.0),
            vertex,
            vertex,
            f"e{index}",
            parameter_range=(0.0, 2 * np.pi),
        )
        rings.append(
            [
                StagedCoedge(
                    edge,
                    1 if index == 0 else -1,
                    LineCurve((0, index), (1, 0)),
                    f"c{index}",
                )
            ]
        )
    if defect in ("partial-turn", "near-period"):
        stage.edge_ranges[0] = (
            0.0,
            np.pi if defect == "partial-turn" else np.nextafter(2 * np.pi, 0.0),
        )
    face = StagedFace(patch, rings, 1, "band", "authored-face", True)
    assert stage._periodic_band(face) is None
    assert len(stage.edge_curves) == 2 and len(stage.vertices) == 2
    with pytest.raises(CadInterchangeError):
        stage._resolve_face(face)


@pytest.mark.parametrize("point", ((0.0, 1.0), (0.6, 0.8)))
def test_misaligned_full_band_retains_authored_vertex_as_subdivision(
    point: tuple[float, float],
) -> None:
    source = _CYLINDER.replace(
        b"#11=CARTESIAN_POINT('',(1.,0.,2.));",
        f"#11=CARTESIAN_POINT('',({point[0]},{point[1]},2.));".encode(),
    )
    result = decode_step_bytes(
        source,
        CadImportPolicy(
            CONTRACT, POLICY.limits, tessellation=BRepTessellationPolicy(realize=False)
        ),
    )
    assert result.report.status == AdapterStatus.LOSSLESS
    geometry = result.model.require_geometry()
    np.testing.assert_array_equal(
        geometry.vertex_points[:2], ((1.0, 0.0, 0.0), (*point, 2.0))
    )
    assert len(geometry.vertex_points) == 3
    assert len(geometry.edge_curves) == 4
    assert len(geometry.face_loops[0]) == 1
    assert len(geometry.face_loops[0][0]) == 5
    assert any(
        key.startswith("curve:") and label.startswith("#15 ")
        for key, label in result.provenance
    )
    assert sum(
        key.startswith("edge:") and "#15 " in label and "DERIVED:subedge" in label
        for key, label in result.provenance
    ) == 2


def test_nist_periodic_bands_preserve_authored_vertices_and_closed_boundary(
    request: pytest.FixtureRequest,
) -> None:
    source = Path(__file__).parents[2] / "data/cad/nist_ftc_11_asme1_rb.stp"
    result = decode_step_bytes(
        source.read_bytes(),
        CadImportPolicy(
            CONTRACT, POLICY.limits, tessellation=BRepTessellationPolicy(realize=False)
        ),
    )
    assert result.report.valid
    geometry = result.model.require_geometry()
    assert len(result.model.patches) == 6
    gaps = certify_brep_gaps(result.model)
    for kind in ("edge-face", "vertex-edge"):
        maximum = max(
            float(bound)
            for category, bound in zip(gaps.kinds, gaps.upper_bounds, strict=True)
            if category == kind
        )
        # The source is an authored tolerance B-Rep, not a zero-gap theorem.
        request.node.user_properties.append((f"maximum_{kind}_gap_meters", maximum))
    provenance = dict(result.provenance)
    for external in (74, 85, 102, 122, 142, 153):
        assert any(
            key.startswith("vertex:") and label.startswith(f"#{external} ")
            for key, label in result.provenance
        )
    seams = [
        int(key.split(":")[1])
        for key, label in provenance.items()
        if key.startswith("edge:") and "periodic-chart-identification" in label
    ]
    assert len(seams) == 4
    for edge in seams:
        uses = [
            index for index, owner in enumerate(geometry.coedge_edges) if owner == edge
        ]
        assert len(uses) == 2
        assert sum(geometry.coedge_senses[index] for index in uses) == 0
    orientation = np.asarray(result.model.orientation)
    for edge in range(len(geometry.edge_curves)):
        assert sum(
            geometry.coedge_senses[use] * orientation[geometry.coedge_faces[use]]
            for use, owner in enumerate(geometry.coedge_edges)
            if owner == edge
        ) == 0
    for loops in geometry.face_loops:
        for loop in loops:
            for first, second in zip(loop, (*loop[1:], loop[0]), strict=True):
                edge0, edge1 = geometry.coedge_edges[first], geometry.coedge_edges[second]
                end = 1 if geometry.coedge_senses[first] > 0 else 0
                start = 0 if geometry.coedge_senses[second] > 0 else 1
                assert geometry.edge_vertices[edge0][end] == geometry.edge_vertices[edge1][start]
                np.testing.assert_allclose(
                    curve_evaluate_host(
                        geometry.pcurves[first], np.asarray(geometry.edge_ranges[edge0, end])
                    ),
                    curve_evaluate_host(
                        geometry.pcurves[second], np.asarray(geometry.edge_ranges[edge1, start])
                    ),
                    rtol=0,
                    atol=1e-12,
                )
    areas = np.asarray(prepare_brep_query(result.model).measures.face_areas)
    angle = np.pi - np.arcsin(0.425 / 1.5)
    for index, patch in enumerate(result.model.patches):
        if isinstance(patch, CylinderPatch):
            expected = 2 * np.pi * float(patch.radius) * 0.0015
        elif isinstance(patch, TorusPatch):
            radius = float(patch.major_radius)
            height = 0.000425 if radius > 0.02 else -0.000425
            expected = 2 * np.pi * 0.0015 * (radius * angle + height)
        else:
            continue
        assert areas[index] == pytest.approx(expected, rel=1e-9)


def test_negative_shell_is_refused_by_default_and_reversed_with_certificate(
    tmp_path: Path,
) -> None:
    source = _inverted_box(tmp_path)
    with pytest.raises(CadInterchangeError) as caught:
        decode_step_bytes(source, POLICY)
    assert caught.value.refusal.reason == "inverted-shell"
    with _budget() as budget:
        result = decode_step_bytes(
            source,
            POLICY,
            repair_policy=StepRepairPolicy(reverse_inverted_shells=True),
            coordinate_budget=budget,
        )
    assert result.report.status == AdapterStatus.REPAIRED
    records = [loss for loss in result.report.losses if "step-repairs" in loss.path]
    assert len(records) == 1
    assert "Reversed shell" in records[0].rationale
    assert "certified signed volume" in records[0].rationale
    assert "nesting depth 0" in records[0].rationale
    assert all(
        value == 0
        for value in result.model.require_geometry().edge_use_balance(
            result.model.orientation
        )
    )


def test_face_sense_repair_uses_exact_outward_supporting_plane(tmp_path: Path) -> None:
    source = _box_source(tmp_path)
    match = re.search(r"(#\d+=ADVANCED_FACE\('[^']*',\([^;]+,\.)([TF])(\.\);)", source)
    assert match is not None
    changed = (
        source[: match.start(2)]
        + ("F" if match.group(2) == "T" else "T")
        + source[match.end(2) :]
    )
    with pytest.raises(CadInterchangeError):
        decode_step_bytes(changed.encode(), POLICY)
    path = tmp_path / "bad-sense.step"
    path.write_text(changed)
    with _budget() as budget:
        result = read_step(
            path,
            POLICY,
            trusted_root=tmp_path,
            repair_policy=StepRepairPolicy(orient_face_senses=True),
            coordinate_budget=budget,
        )
    assert result.report.status == AdapterStatus.REPAIRED
    records = [loss for loss in result.report.losses if "step-repairs" in loss.path]
    assert len(records) == 1
    assert "supporting-plane outward test" in records[0].rationale
    assert match.group(1).split("=")[0] in records[0].path


def test_curved_face_senses_refuse_without_sampled_guess() -> None:
    with _budget() as budget, pytest.raises(CadInterchangeError) as caught:
        decode_step_bytes(
            _CYLINDER,
            POLICY,
            repair_policy=StepRepairPolicy(orient_face_senses=True),
            coordinate_budget=budget,
        )
    assert caught.value.refusal.reason == "unsupported-entity"


def test_repairs_require_explicit_budget() -> None:
    with pytest.raises(CadInterchangeError) as caught:
        decode_step_bytes(
            _CYLINDER, POLICY, repair_policy=StepRepairPolicy(add_missing_seams=True)
        )
    assert caught.value.refusal.reason == "limit"


def test_geometry_moving_seam_repair_refuses() -> None:
    source = _CYLINDER.replace(
        b"#11=CARTESIAN_POINT('',(1.,0.,2.));",
        b"#11=CARTESIAN_POINT('',(1.0000000001,0.,2.));",
    )
    with _budget() as budget, pytest.raises(CadInterchangeError):
        decode_step_bytes(
            source,
            POLICY,
            repair_policy=StepRepairPolicy(add_missing_seams=True),
            coordinate_budget=budget,
        )


def test_repaired_status_survives_report_composition() -> None:
    record = AdapterLoss(
        "geometry.#1",
        "import",
        "transformed",
        "Exact shell reversal",
        changes_interpretation=False,
    )
    repaired = AdapterReport(
        AdapterStatus.REPAIRED,
        "step",
        "native",
        source_id="a",
        target_id="b",
        losses=(record,),
    )
    exact = AdapterReport(
        AdapterStatus.LOSSLESS, "native", "native", source_id="b", target_id="c"
    )
    combined = compose_adapter_reports((repaired, exact))
    assert repaired.valid and combined.valid
    assert combined.status == AdapterStatus.REPAIRED
    assert combined.losses == (record,)


def test_cavity_shell_requires_negative_sign_from_certified_nesting(
    tmp_path: Path,
) -> None:
    outer = _box_source(tmp_path)
    inner = brep_box(
        (0.25, 0.5, 0.75),
        (0.75, 1.5, 2.25),
        coordinate_contract=CONTRACT,
    )
    path = tmp_path / "inner.step"
    write_step(inner, path)
    text = path.read_text().split("DATA;\n", 1)[1]
    match = re.search(r"#(\d+)=CLOSED_SHELL\([^;]+;", text)
    assert match is not None
    inner_shell = int(match.group(1)) + 100_000
    geometry_records = re.sub(
        r"#(\d+)",
        lambda match: f"#{int(match.group(1)) + 100_000}",
        text[: match.end()],
    )
    source = re.sub(
        r"MANIFOLD_SOLID_BREP\(('[^']*'),#(\d+)\)",
        lambda match: (
            f"BREP_WITH_VOIDS({match.group(1)},#{match.group(2)},(#{inner_shell}))"
        ),
        outer,
    )
    source = source.replace("ENDSEC;\nEND-ISO", geometry_records + "\nENDSEC;\nEND-ISO")
    with pytest.raises(CadInterchangeError) as caught:
        decode_step_bytes(source.encode(), POLICY)
    assert caught.value.refusal.reason == "inverted-shell"
    with _budget() as budget:
        result = decode_step_bytes(
            source.encode(),
            POLICY,
            repair_policy=StepRepairPolicy(reverse_inverted_shells=True),
            coordinate_budget=budget,
        )
    assert result.report.status == AdapterStatus.REPAIRED
    records = [loss for loss in result.report.losses if "step-repairs" in loss.path]
    assert len(records) == 1
    assert f"#{inner_shell}" in records[0].path
    assert "nesting depth 1" in records[0].rationale


def test_repaired_report_requires_a_repair_record() -> None:
    with pytest.raises(ValueError, match="enumerate"):
        AdapterReport(
            AdapterStatus.REPAIRED, "step", "native", source_id="a", target_id="b"
        )


def test_none_policy_leaves_clean_source_unchanged(tmp_path: Path) -> None:
    source = _box_source(tmp_path).encode()
    default = decode_step_bytes(source, POLICY)
    explicit_none = decode_step_bytes(source, POLICY, repair_policy=StepRepairPolicy())
    assert explicit_none.model.model_id == default.model.model_id
    assert explicit_none.report.report_id == default.report.report_id
    assert default.report.status == AdapterStatus.LOSSLESS


def test_no_policy_can_authorize_tolerance_growth() -> None:
    with pytest.raises(TypeError, match="maximum_gap"):
        StepRepairPolicy(maximum_gap=1e-6)  # ty: ignore[unknown-argument]


def test_unsupported_conical_missing_seam_refuses_explicitly() -> None:
    source = _CYLINDER.replace(
        b"CYLINDRICAL_SURFACE('',#4,1.)", b"CONICAL_SURFACE('',#4,1.,0.25)"
    )
    with _budget() as budget, pytest.raises(CadInterchangeError) as caught:
        decode_step_bytes(
            source,
            POLICY,
            repair_policy=StepRepairPolicy(add_missing_seams=True),
            coordinate_budget=budget,
        )
    assert caught.value.refusal.reason == "unsupported-entity"


def test_missing_natural_torus_seams_are_exactly_recreated() -> None:
    lines = _CYLINDER.splitlines()
    retained = []
    for line in lines:
        match = re.match(rb"#(\d+)=", line)
        if match is not None and (
            6 <= int(match.group(1)) <= 19 or int(match.group(1)) == 21
        ):
            continue
        retained.append(line)
    source = (
        b"\n".join(retained)
        .replace(b"#20=FACE_OUTER_BOUND", b"#18=EDGE_LOOP('',());\n#20=FACE_OUTER_BOUND")
        .replace(b"CYLINDRICAL_SURFACE('',#4,1.)", b"TOROIDAL_SURFACE('',#4,3.,1.)")
        .replace(b"(#20,#21)", b"(#20)")
    )
    with pytest.raises(CadInterchangeError):
        decode_step_bytes(source, POLICY)
    with _budget() as budget:
        result = decode_step_bytes(
            source,
            POLICY,
            repair_policy=StepRepairPolicy(add_missing_seams=True),
            coordinate_budget=budget,
        )
    assert result.report.status == AdapterStatus.REPAIRED
    records = [loss for loss in result.report.losses if "step-repairs" in loss.path]
    assert len(records) == 1 and "#22" in records[0].path
    assert "native symbolic periods" in records[0].rationale


def test_default_audit_exhaustion_admits_source_with_unresolved_orientation(
    tmp_path: Path,
) -> None:
    source = _box_source(tmp_path).encode()
    result = decode_step_bytes(
        source, POLICY, repair_policy=StepRepairPolicy(audit_maximum_work_units=1)
    )
    assert result.report.valid
    assert result.report.status == AdapterStatus.LOSSLESS
    assert any("orientation_unresolved" in note for note in result.report.assumptions)
    assert not any("step-repairs" in record.path for record in result.report.losses)


@pytest.mark.parametrize("shell_reversal", (False, True), ids=("seam", "shell"))
def test_explicit_repair_budget_exhaustion_reports_resource_limit(
    shell_reversal: bool, tmp_path: Path
) -> None:
    source = _inverted_box(tmp_path) if shell_reversal else _CYLINDER
    with HostBatchTicket(1, bytes_upper=16 << 20) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=1, maximum_memory_bytes=16 << 20
        )
        with budget.activate(), pytest.raises(CadInterchangeError) as caught:
            decode_step_bytes(
                source,
                POLICY,
                repair_policy=StepRepairPolicy(
                    add_missing_seams=not shell_reversal,
                    reverse_inverted_shells=shell_reversal,
                ),
                coordinate_budget=budget,
            )
    assert caught.value.refusal.reason == "limit"
    assert caught.value.resource_manifest is not None
