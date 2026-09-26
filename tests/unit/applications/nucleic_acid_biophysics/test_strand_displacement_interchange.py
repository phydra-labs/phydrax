import csv
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from phydrax.applications.nucleic_acid_biophysics.interchange import (
    admit_prepared_strand_displacement_csv,
    prepare_strand_displacement_cohort,
    StrandDisplacementSourceManifest,
    StrandDisplacementSourceMember,
    StrandDisplacementWellManifest,
)
from phydrax.applications.nucleic_acid_biophysics.interchange._strand_displacement import (
    _parse_admitted_workbooks,
)
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import MILLIMOLAR


_COLUMNS = (
    "case_id",
    "sample_label",
    "experiment_id",
    "plate_id",
    "well_id",
    "preparation_id",
    "replicate_id",
    "reporter_id",
    "sequence_family_id",
    "condition_id",
    "chemistry_direction",
    "temperature_kelvin",
    "time_seconds",
    "intensity",
    "intensity_unit_id",
    "saturation_state",
    "injection_reference_seconds",
    "saturation_threshold_intensity",
    "construct_ids_json",
    "initial_concentrations_molar_json",
    "source_manifest_ids_json",
)
_USE = {
    "commercial_use": True,
    "redistribution": False,
    "training_use": False,
    "export": False,
}


def _manifest(name, content, *, lineage=("independent-test-source",)):
    return ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        license_id="CC-BY-4.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"identity": 1.0},
        uncertainty=None,
        lineage_ids=lineage,
    )


def _source(*, leaking_family=False):
    raw = StrandDisplacementSourceMember(
        "raw-workbook", "raw.xlsx", _manifest("raw.xlsx", b"raw")
    )
    layout = StrandDisplacementSourceMember(
        "plate-layout", "layout.xlsx", _manifest("layout.xlsx", b"layout")
    )
    wells = (
        StrandDisplacementWellManifest(
            "Sample X1",
            "reporter calibration",
            "calibration-case",
            "calibration-preparation",
            "replicate-1",
            "reporter-1",
            "family-shared" if leaking_family else "family-calibration",
            ("invader", "substrate"),
            (0.0001, 0.0001),
            concentration_unit=MILLIMOLAR,
            role="calibration",
            saturation_threshold_intensity=100.0,
        ),
        StrandDisplacementWellManifest(
            "Sample X2",
            "locked displacement",
            "locked-case",
            "locked-preparation",
            "replicate-1",
            "reporter-1",
            "family-shared" if leaking_family else "family-locked",
            ("invader", "substrate"),
            (0.0001, 0.0001),
            concentration_unit=MILLIMOLAR,
            role="locked_evaluation",
            saturation_threshold_intensity=100.0,
        ),
    )
    return StrandDisplacementSourceManifest(
        archive=None,
        raw_workbook=raw,
        plate_layout=layout,
        processed_csv=None,
        experiment_id="experiment-1",
        plate_id="plate-1",
        condition_id="buffer-1",
        chemistry_direction="RNA>DNA",
        temperature_kelvin=298.15,
        wells=wells,
    )


def _csv_bytes(source, *, duplicate_well=False):
    source_ids = sorted(
        (
            source.raw_workbook.manifest.manifest_id,
            source.plate_layout.manifest.manifest_id,
        )
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for index, well in enumerate(source.wells):
        for time, intensity, state in (
            (0.0, "10.0", "observed"),
            (2.0, "", "right-censored"),
        ):
            writer.writerow(
                {
                    "case_id": well.case_id,
                    "sample_label": well.sample_label,
                    "experiment_id": source.experiment_id,
                    "plate_id": source.plate_id,
                    "well_id": "A1" if duplicate_well else f"A{index + 1}",
                    "preparation_id": well.preparation_id,
                    "replicate_id": well.replicate_id,
                    "reporter_id": well.reporter_id,
                    "sequence_family_id": well.sequence_family_id,
                    "condition_id": source.condition_id,
                    "chemistry_direction": source.chemistry_direction,
                    "temperature_kelvin": source.temperature_kelvin,
                    "time_seconds": time,
                    "intensity": intensity,
                    "intensity_unit_id": "instrument-fluorescence-unit",
                    "saturation_state": state,
                    "injection_reference_seconds": 0.0,
                    "saturation_threshold_intensity": 100.0,
                    "construct_ids_json": json.dumps(well.construct_ids),
                    "initial_concentrations_molar_json": json.dumps(
                        well.initial_concentrations_molar
                    ),
                    "source_manifest_ids_json": json.dumps(source_ids),
                }
            )
    return stream.getvalue().encode()


def _admit(tmp_path, source, content):
    path = tmp_path / "traces.csv"
    path.write_bytes(content)
    raw_ids = (
        source.raw_workbook.manifest.manifest_id,
        source.plate_layout.manifest.manifest_id,
    )
    prepared = _manifest("traces.csv", content, lineage=(source.manifest_id, *raw_ids))
    return admit_prepared_strand_displacement_csv(
        path, source, prepared, requested_use=_USE
    )


def test_prepared_csv_preserves_order_units_saturation_and_provenance(tmp_path):
    source = _source()
    admission = _admit(tmp_path, source, _csv_bytes(source))
    first, second = admission.traces

    assert tuple(trace.case_id for trace in admission.traces) == (
        "calibration-case",
        "locked-case",
    )
    assert (first.identity.well_id, second.identity.well_id) == ("A1", "A2")
    np.testing.assert_array_equal(first.time_seconds, [0.0, 2.0])
    assert first.initial_concentrations_molar[0] == pytest.approx(1e-7)
    np.testing.assert_array_equal(first.saturation_mask, [False, True])
    assert np.isnan(np.asarray(first.intensity)[1])
    assert first.chemistry_direction == "RNA>DNA"
    assert source.raw_workbook.manifest.manifest_id in first.source_manifest_ids
    assert len(first.source_manifest_ids) == 3
    with pytest.raises(ValueError, match="Plate groups cross campaign roles"):
        prepare_strand_displacement_cohort((admission,))


def test_prepared_csv_refuses_duplicate_wells_and_family_role_leakage(tmp_path):
    source = _source()
    with pytest.raises(ValueError, match="duplicate physical well"):
        _admit(tmp_path, source, _csv_bytes(source, duplicate_well=True))

    leaking = _source(leaking_family=True)
    admission = _admit(tmp_path, leaking, _csv_bytes(leaking))
    with pytest.raises(ValueError, match="Sequence Family|independent_unit_id"):
        prepare_strand_displacement_cohort((admission,))


def test_prepared_csv_refuses_schema_or_digest_disagreement(tmp_path):
    source = _source()
    content = _csv_bytes(source)
    malformed = content.replace(b"time_seconds", b"time_minutes", 1)
    with pytest.raises(ValueError, match="columns and order"):
        _admit(tmp_path, source, malformed)

    path = tmp_path / "traces.csv"
    path.write_bytes(content + b"\n")
    prepared = _manifest(
        "traces.csv",
        content,
        lineage=(
            source.manifest_id,
            source.raw_workbook.manifest.manifest_id,
            source.plate_layout.manifest.manifest_id,
        ),
    )
    with pytest.raises(ValueError, match="Reference artifact size mismatch"):
        admit_prepared_strand_displacement_csv(path, source, prepared, requested_use=_USE)


_XLSX_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_XLSX_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_XLSX_PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def _xlsx_cell(coordinate, value):
    if value is None:
        return f'<c r="{coordinate}"/>'
    if isinstance(value, str):
        return f'<c r="{coordinate}" t="inlineStr"><is><t>{value}</t></is></c>'
    return f'<c r="{coordinate}"><v>{value}</v></c>'


def _xlsx_bytes(sheet_name, rows):
    body = "".join(
        f'<row r="{index}">'
        + "".join(_xlsx_cell(f"{column}{index}", value) for column, value in row.items())
        + "</row>"
        for index, row in enumerate(rows, start=1)
    )
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as workbook:
        workbook.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{_XLSX_MAIN}" xmlns:r="{_XLSX_REL}"><sheets>'
            f'<sheet name="{sheet_name}" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        workbook.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships xmlns="{_XLSX_PACKAGE_REL}">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        workbook.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{_XLSX_MAIN}"><sheetData>{body}</sheetData></worksheet>',
        )
    return stream.getvalue()


def test_raw_workbook_rows_with_empty_sample_cells_are_not_samples():
    source = _source()
    plate = _xlsx_bytes(
        "Sheet1",
        (
            {"A": "Sample X1", "B": "reporter calibration"},
            {"A": "Sample X2", "B": "locked displacement"},
        ),
    )
    raw = _xlsx_bytes(
        "Table All Cycles",
        (
            {"A": "Well", "B": "Time [s]", "C": 0, "D": 10, "E": 20},
            {"A": "A01", "B": "Sample X1", "C": 1.0, "D": "Inj.", "E": 2.0},
            {"A": "A02", "B": "Sample X2", "C": 3.0, "D": "Inj.", "E": 4.0},
            {"A": "Notes", "B": None},
        ),
    )
    admission = _parse_admitted_workbooks(
        raw, plate, source, saturation_markers=("OVER",), injection_marker="Inj."
    )

    assert tuple(trace.identity.well_id for trace in admission.traces) == (
        "A01",
        "A02",
    )
    np.testing.assert_array_equal(admission.traces[0].time_seconds, [0.0, 20.0])
