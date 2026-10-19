"""S2-029 writer defects, authored here rather than imported from a corpus.

The cylinder band and empty-loop torus are handwritten Part 21 sources. Box
shells and apex cones come from native constructors with dimensions declared
below; mutations change only STEP topology/senses, except the deliberate invalid
apex displacement. Independent oracles are dyadic box volumes, the cone
formula pi*r**2*h/3, the torus formula 2*pi**2*R*r**2, and analytic extrema.
"""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from phydrax._external_resource import ResourceLimits
from phydrax._physical import SpatialCoordinateContract
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry.brep._constructors import brep_box, brep_cone
from phydrax.geometry.brep._measure import brep_signed_shell_volume_enclosures
from phydrax.geometry.brep._model import BRepModel
from phydrax.interchange._cad import CadImportPolicy, CadInterchangeError
from phydrax.interchange._report import AdapterLoss, AdapterReport, AdapterStatus
from phydrax.interchange._step import decode_step_bytes, write_step
from phydrax.interchange._step_repair import StepRepairPolicy


_CONTRACT = SpatialCoordinateContract.si()
_POLICY = CadImportPolicy(
    _CONTRACT,
    ResourceLimits(
        max_bytes=1 << 24,
        max_depth=64,
        max_nodes=200_000,
        max_attributes=4_000_000,
        max_losses=8,
    ),
)

# Radius 1/2, translated axis (2,-1,z), axial interval [1/2,5/2].
# Each circle has a source-owned endpoint, but neither longitudinal seam use.
_CYLINDER_BAND = b"""ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('authored periodic defect'),'2;1');
FILE_NAME('translated-band.step','2026-10-14T00:00:00',('PHYDRA'),('PHYDRA'),'native corpus','native corpus','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#101=CARTESIAN_POINT('',(2.,-1.,0.5));
#102=DIRECTION('',(0.,0.,1.));
#103=DIRECTION('',(1.,0.,0.));
#104=AXIS2_PLACEMENT_3D('',#101,#102,#103);
#105=CYLINDRICAL_SURFACE('',#104,0.5);
#106=CIRCLE('',#104,0.5);
#107=CARTESIAN_POINT('',(2.,-1.,2.5));
#108=AXIS2_PLACEMENT_3D('',#107,#102,#103);
#109=CIRCLE('',#108,0.5);
#110=CARTESIAN_POINT('',(2.5,-1.,0.5));
#111=CARTESIAN_POINT('',(2.5,-1.,2.5));
#112=VERTEX_POINT('',#110);
#113=VERTEX_POINT('',#111);
#114=EDGE_CURVE('',#112,#112,#106,.T.);
#115=EDGE_CURVE('',#113,#113,#109,.T.);
#116=ORIENTED_EDGE('',*,*,#114,.T.);
#117=ORIENTED_EDGE('',*,*,#115,.F.);
#118=EDGE_LOOP('',(#116));
#119=EDGE_LOOP('',(#117));
#120=FACE_OUTER_BOUND('',#118,.T.);
#121=FACE_BOUND('',#119,.T.);
#122=ADVANCED_FACE('',(#120,#121),#105,.T.);
#123=OPEN_SHELL('',(#122));
#124=SHELL_BASED_SURFACE_MODEL('',(#123));
#125=MANIFOLD_SURFACE_SHAPE_REPRESENTATION('',(#124),#129);
#126=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.));
#127=(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.));
#128=(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT());
#129=(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#126,#127,#128)) REPRESENTATION_CONTEXT('',''));
ENDSEC;
END-ISO-10303-21;
"""

# A full torus boundary whose writer omitted both periodic seam pairs.
_TORUS_WITHOUT_SEAMS = b"""ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('authored empty periodic trim'),'2;1');
FILE_NAME('torus.step','2026-10-14T00:00:00',('PHYDRA'),('PHYDRA'),'native corpus','native corpus','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#201=CARTESIAN_POINT('',(-1.,2.,0.5));
#202=DIRECTION('',(0.,0.,1.));
#203=DIRECTION('',(1.,0.,0.));
#204=AXIS2_PLACEMENT_3D('',#201,#202,#203);
#205=TOROIDAL_SURFACE('',#204,2.,0.25);
#206=EDGE_LOOP('',());
#207=FACE_OUTER_BOUND('',#206,.T.);
#208=ADVANCED_FACE('',(#207),#205,.T.);
#209=CLOSED_SHELL('',(#208));
#210=MANIFOLD_SOLID_BREP('',#209);
#211=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#210),#215);
#212=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.));
#213=(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.));
#214=(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT());
#215=(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#212,#213,#214)) REPRESENTATION_CONTEXT('',''));
ENDSEC;
END-ISO-10303-21;
"""


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


def _native_text(model: BRepModel, path: Path) -> str:
    written = write_step(model, path)
    assert written.report.valid
    return path.read_text()


def _repairs(report: AdapterReport) -> tuple[AdapterLoss, ...]:
    assert report.valid
    assert report.status == AdapterStatus.REPAIRED
    records = tuple(loss for loss in report.losses if "step-repairs" in loss.path)
    assert records
    assert all(not record.changes_interpretation for record in records)
    return records


def _shell_bounds(
    model: BRepModel, budget: CoordinateEnclosureBudget, *, subdivisions: int = 8
) -> np.ndarray:
    result = brep_signed_shell_volume_enclosures(
        model,
        model.require_geometry(),
        subdivisions=subdivisions,
        coordinate_budget=budget,
    )
    assert result.model_id == model.model_id
    bounds = np.asarray(result.volume_enclosures)
    assert np.all(np.isfinite(bounds))
    assert np.all(bounds[:, 0] <= bounds[:, 1])
    return bounds


def _bad_face_senses(tmp_path: Path) -> tuple[bytes, tuple[str, ...]]:
    text = _native_text(
        brep_box((-2.0, -1.0, 0.5), (1.0, 3.0, 2.5), coordinate_contract=_CONTRACT),
        tmp_path / "sense-source.step",
    )
    pattern = re.compile(r"(#\d+)=ADVANCED_FACE\('[^']*',\([^;]+,\.([TF])\.\);")
    faces = tuple(pattern.finditer(text))
    assert len(faces) == 6
    selected = (faces[0], faces[-1])
    for face in reversed(selected):
        text = (
            text[: face.start(2)] + ("F" if face[2] == "T" else "T") + text[face.end(2) :]
        )
    return text.encode(), tuple(face[1] for face in selected)


def _near_wall_cavity(tmp_path: Path, *, inverted: bool) -> tuple[bytes, str, Fraction]:
    # Dyadic clearance to the outer x=2 plane: 2**-30 metres.
    # Even the inner shell's least-x vertex is within 2**-12 of that plane.
    lower = 2.0 - 2.0**-12
    upper = 2.0 - 2.0**-30
    volume = Fraction(upper) - Fraction(lower)  # y and z extents are exactly 1.
    outer = _native_text(
        brep_box((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0), coordinate_contract=_CONTRACT),
        tmp_path / "outer.step",
    )
    inner = _native_text(
        brep_box((lower, -0.5, -0.5), (upper, 0.5, 0.5), coordinate_contract=_CONTRACT),
        tmp_path / "near-wall-inner.step",
    ).split("DATA;\n", 1)[1]
    shell = re.search(r"#(\d+)=CLOSED_SHELL\([^;]+;", inner)
    assert shell is not None
    offset = 100_000
    inner_shell = f"#{int(shell[1]) + offset}"
    records = re.sub(
        r"#(\d+)",
        lambda match: f"#{int(match[1]) + offset}",
        inner[: shell.end()],
    )
    cavity_shell = inner_shell
    if inverted:
        cavity_shell = "#900000"
        records += f"\n{cavity_shell}=ORIENTED_CLOSED_SHELL('',*,{inner_shell},.F.);"
    outer, count = re.subn(
        r"MANIFOLD_SOLID_BREP\(('[^']*'),#(\d+)\)",
        lambda match: f"BREP_WITH_VOIDS({match[1]},#{match[2]},({cavity_shell}))",
        outer,
    )
    assert count == 1
    outer = outer.replace("ENDSEC;\nEND-ISO", records + "\nENDSEC;\nEND-ISO")
    return outer.encode(), cavity_shell, volume


def test_missing_cylinder_seam_default_refuses_with_status() -> None:
    with pytest.raises(CadInterchangeError) as caught:
        decode_step_bytes(_CYLINDER_BAND, _POLICY)
    assert caught.value.refusal.reason == "inconsistent-geometry"
    assert caught.value.status == AdapterStatus.INCONSISTENT_SOURCE
    assert caught.value.refusal.entity == "#116 ORIENTED_EDGE"
    assert caught.value.refusal.chain == ("#122 ADVANCED_FACE",)


def test_missing_cylinder_seam_repair_preserves_boundary_and_chart() -> None:
    with _budget() as budget:
        result = decode_step_bytes(
            _CYLINDER_BAND,
            _POLICY,
            repair_policy=StepRepairPolicy(add_missing_seams=True),
            coordinate_budget=budget,
        )
        box = result.model.patches[0].bounding_box(
            np.asarray(result.model.parameter_bounds[0]), budget=budget
        )
    records = _repairs(result.report)
    assert len(records) == 1
    assert "#122" in records[0].path
    assert "longitudinal periodic seam" in records[0].rationale
    geometry = result.model.require_geometry()
    np.testing.assert_array_equal(
        geometry.vertex_points, ((2.5, -1.0, 0.5), (2.5, -1.0, 2.5))
    )
    assert len(geometry.edge_curves) == 3
    seam_uses = tuple(c for c, e in enumerate(geometry.coedge_edges) if e == 2)
    assert len(seam_uses) == 2
    assert tuple(geometry.coedge_senses[c] for c in seam_uses) == (1, -1)
    assert geometry.edge_vertices[2] == (0, 1)
    # Analytic extrema of the entire band, not samples of the repaired seam.
    assert np.all(box[0] <= (1.5, -1.5, 0.5))
    assert np.all(box[1] >= (2.5, -0.5, 2.5))
    assert result.coverage.pcurves_fitted == 0


def test_missing_torus_seams_default_refuses_with_status() -> None:
    with pytest.raises(CadInterchangeError) as caught:
        decode_step_bytes(_TORUS_WITHOUT_SEAMS, _POLICY)
    assert caught.value.refusal.reason == "malformed"
    assert caught.value.status == AdapterStatus.MALFORMED_SOURCE
    assert "#206" in caught.value.refusal.entity


def test_missing_torus_seams_repair_encloses_closed_form() -> None:
    with _budget() as budget:
        result = decode_step_bytes(
            _TORUS_WITHOUT_SEAMS,
            _POLICY,
            repair_policy=StepRepairPolicy(add_missing_seams=True),
            coordinate_budget=budget,
        )
        bounds = _shell_bounds(result.model, budget, subdivisions=32)
    records = _repairs(result.report)
    assert len(records) == 1
    assert "#208" in records[0].path
    assert "native symbolic periods" in records[0].rationale
    assert bounds.shape == (1, 2)
    reference = np.pi**2 / 4  # 2*pi**2 * 2 * (1/4)**2.
    assert 0 < bounds[0, 0] <= reference <= bounds[0, 1]
    geometry = result.model.require_geometry()
    assert len(geometry.edge_curves) == 2
    assert len(geometry.face_loops[0]) == 1
    assert len(geometry.face_loops[0][0]) == 4
    assert geometry.coedge_edges.count(0) == geometry.coedge_edges.count(1) == 2
    assert all(
        value == 0 for value in geometry.edge_use_balance(result.model.orientation)
    )
    assert result.coverage.pcurves_fitted == 0


@pytest.mark.parametrize("repair", [False, True], ids=["default", "seam-policy"])
def test_cone_apex_trim_import_encloses_closed_form(tmp_path: Path, repair: bool) -> None:
    source = _native_text(
        brep_cone(1.5, 0.0, 3.0, coordinate_contract=_CONTRACT),
        tmp_path / "apex.step",
    ).encode()
    with _budget() as budget:
        result = decode_step_bytes(
            source,
            _POLICY,
            repair_policy=StepRepairPolicy(add_missing_seams=True) if repair else None,
            coordinate_budget=budget,
        )
        bounds = _shell_bounds(result.model, budget, subdivisions=16)
    assert result.report.valid
    assert result.report.status == AdapterStatus.LOSSLESS
    assert not any("step-repairs" in loss.path for loss in result.report.losses)
    assert bounds.shape == (1, 2)
    assert 0 < bounds[0, 0] <= 2.25 * np.pi <= bounds[0, 1]
    geometry = result.model.require_geometry()
    poles = tuple(e for e, curve in enumerate(geometry.edge_curves) if curve == -1)
    assert len(poles) == 1
    start, end = geometry.edge_vertices[poles[0]]
    assert start == end
    np.testing.assert_array_equal(geometry.vertex_points[start], (0.0, 0.0, 3.0))
    balances = geometry.edge_use_balance(result.model.orientation)
    assert abs(balances[poles[0]]) == 1
    assert all(value == 0 for edge, value in enumerate(balances) if edge not in poles)
    assert result.coverage.pcurves_fitted == 0


@pytest.mark.parametrize("repair", [False, True], ids=["default", "seam-policy"])
def test_displaced_cone_apex_trim_refuses_without_repair(
    tmp_path: Path, repair: bool
) -> None:
    source = _native_text(
        brep_cone(1.5, 0.0, 3.0, coordinate_contract=_CONTRACT),
        tmp_path / "displaced-apex.step",
    )
    points = re.finditer(r"#\d+=CARTESIAN_POINT\('[^']*',\(([^)]*)\)\);", source)
    apex = next(
        point
        for point in points
        if tuple(float(x) for x in point[1].split(",")) == (0.0, 0.0, 3.0)
    )
    # A dyadic 2**-10 displacement cannot be authorized by a topology-only repair.
    source = source[: apex.start(1)] + "0.,0.,3.0009765625" + source[apex.end(1) :]
    with _budget() as budget, pytest.raises(CadInterchangeError) as caught:
        decode_step_bytes(
            source.encode(),
            _POLICY,
            repair_policy=StepRepairPolicy(add_missing_seams=True) if repair else None,
            coordinate_budget=budget,
        )
    assert caught.value.refusal.reason == "inconsistent-geometry"
    assert caught.value.status == AdapterStatus.INCONSISTENT_SOURCE


def test_inconsistent_face_senses_default_refuses_with_status(tmp_path: Path) -> None:
    source, entities = _bad_face_senses(tmp_path)
    with pytest.raises(CadInterchangeError) as caught:
        decode_step_bytes(source, _POLICY)
    assert caught.value.refusal.reason == "inconsistent-geometry"
    assert caught.value.status == AdapterStatus.INCONSISTENT_SOURCE
    assert any(entity in caught.value.refusal.entity for entity in entities)


def test_inconsistent_face_senses_repair_encloses_box_volume(tmp_path: Path) -> None:
    source, entities = _bad_face_senses(tmp_path)
    with _budget() as budget:
        result = decode_step_bytes(
            source,
            _POLICY,
            repair_policy=StepRepairPolicy(orient_face_senses=True),
            coordinate_budget=budget,
        )
        bounds = _shell_bounds(result.model, budget)
    records = _repairs(result.report)
    assert len(records) == len(entities) == 2
    assert all(any(entity in record.path for record in records) for entity in entities)
    assert all(
        "rational supporting-plane outward test" in record.rationale for record in records
    )
    assert bounds.shape == (1, 2)
    assert 0 < bounds[0, 0] <= 24.0 <= bounds[0, 1]  # 3 * 4 * 2.
    geometry = result.model.require_geometry()
    assert all(
        value == 0 for value in geometry.edge_use_balance(result.model.orientation)
    )
    assert result.coverage.pcurves_fitted == 0


def test_near_outer_cavity_wrong_sense_default_refuses_with_status(
    tmp_path: Path,
) -> None:
    source, entity, _ = _near_wall_cavity(tmp_path, inverted=False)
    with pytest.raises(CadInterchangeError) as caught:
        decode_step_bytes(source, _POLICY)
    assert caught.value.refusal.reason == "inverted-shell"
    assert caught.value.status == AdapterStatus.INCONSISTENT_SOURCE
    assert entity in caught.value.refusal.entity
    assert "nesting depth 1" in caught.value.refusal.message


def test_near_outer_cavity_repair_certifies_negative_inner_volume(tmp_path: Path) -> None:
    source, entity, volume = _near_wall_cavity(tmp_path, inverted=False)
    with _budget() as budget:
        result = decode_step_bytes(
            source,
            _POLICY,
            repair_policy=StepRepairPolicy(reverse_inverted_shells=True),
            coordinate_budget=budget,
        )
        bounds = _shell_bounds(result.model, budget)
    records = _repairs(result.report)
    assert len(records) == 1
    assert entity in records[0].path
    assert "certified signed volume" in records[0].rationale
    assert "nesting depth 1" in records[0].rationale
    assert bounds.shape == (2, 2)
    assert 0 < bounds[0, 0] <= 64.0 <= bounds[0, 1]
    assert bounds[1, 0] <= -float(volume) <= bounds[1, 1] < 0
    geometry = result.model.require_geometry()
    # Repair must leave the subnanometre wall clearance exactly source-authored.
    inner_vertices = np.asarray(geometry.vertex_points)[8:]
    assert np.max(inner_vertices[:, 0]) == 2.0 - 2.0**-30
    assert Fraction(2) - Fraction(float(np.max(inner_vertices[:, 0]))) == Fraction(
        1, 2**30
    )
    assert all(
        value == 0 for value in geometry.edge_use_balance(result.model.orientation)
    )


def test_near_outer_cavity_correct_sense_imports_without_repair(tmp_path: Path) -> None:
    source, _, volume = _near_wall_cavity(tmp_path, inverted=True)
    with _budget() as budget:
        result = decode_step_bytes(source, _POLICY, coordinate_budget=budget)
        bounds = _shell_bounds(result.model, budget)
    assert result.report.valid
    assert result.report.status == AdapterStatus.LOSSLESS
    assert not any("orientation_unresolved" in note for note in result.report.assumptions)
    assert not any("step-repairs" in loss.path for loss in result.report.losses)
    assert bounds.shape == (2, 2)
    assert 0 < bounds[0, 0] <= 64.0 <= bounds[0, 1]
    assert bounds[1, 0] <= -float(volume) <= bounds[1, 1] < 0


def test_near_outer_cavity_exhausted_default_audit_reports_unresolved(
    tmp_path: Path,
) -> None:
    source, entity, _ = _near_wall_cavity(tmp_path, inverted=False)
    result = decode_step_bytes(
        source, _POLICY, repair_policy=StepRepairPolicy(audit_maximum_work_units=1)
    )
    assert result.report.valid
    assert result.report.status == AdapterStatus.LOSSLESS
    assert any(
        "orientation_unresolved" in note and entity in note
        for note in result.report.assumptions
    )
    assert not any("step-repairs" in loss.path for loss in result.report.losses)
