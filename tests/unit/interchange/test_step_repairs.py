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
from phydrax.geometry.brep._constructors import brep_box
from phydrax.interchange._cad import CadImportPolicy, CadInterchangeError
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
    with pytest.raises(CadInterchangeError):
        decode_step_bytes(_CYLINDER, POLICY)
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
