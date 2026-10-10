#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Retained campaign reports aggregate into closure without executing physics."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from phydrax.qualification._nuclear_fuel_release_matrix import FUEL_VERIFICATION_CASES
from tools.nuclear_fuel_qualification import (
    _scientific_wire_artifact,
    aggregate_reports,
    main,
)


_CONTRACT_IDENTITY = ("contract-build", "contract-driver", "contract-environment")
_ARTIFACT_DIRECTORY = "literal-contract-artifact-location"


def _report_contract_fixture(
    case_ids: tuple[str, ...],
    /,
    *,
    design: str = "smoke",
    identity: tuple[str, str, str] = _CONTRACT_IDENTITY,
    outcome: str = "passed",
    detected: bool = True,
    measurement: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Interchange contract fixtures are not observed physical or fault evidence.

    Rows are content-addressed by the campaign's own wire writer, as a real
    campaign publishes them.
    """
    records: list[dict[str, object]] = []
    for case_id in case_ids:
        case = next(row for row in FUEL_VERIFICATION_CASES if row.case_id == case_id)
        source = dict(
            zip(("source_build_id", "driver_id", "environment_id"), identity, strict=True)
        )
        numeric, numeric_id = _scientific_wire_artifact(
            {
                **source,
                "case_id": case_id,
                "design": design,
                "reference": case.reference,
                "metric": case.metric,
                "outcome": outcome,
                "scope": "synthetic-numerical-equation-reproduction-not-scientific-validation",
                "measurement": {"contract-fixture-not-physical-evidence": True}
                if measurement is None
                else measurement,
            }
        )
        fault, fault_id = _scientific_wire_artifact(
            {
                **source,
                "case_id": case_id,
                "design": design,
                "detected": detected,
                "fault": "interchange-contract-fixture-not-physical-fault-proof",
            }
        )
        records.append(
            {
                **numeric,
                "raw_artifact_id": numeric_id,
                "fault_adequacy": fault,
                "fault_artifact_id": fault_id,
            }
        )
    return {
        "cases": records,
        "campaign_designs": [design],
        "source_receipt": {
            "build_before": identity[0],
            "build_after": identity[0],
            "driver_before": identity[1],
            "driver_after": identity[1],
            "environment": {"fingerprint": identity[2]},
            "capture": "before-and-after-actual-campaign",
        },
        "artifact_directory": _ARTIFACT_DIRECTORY,
        "successful": True,
    }


def _write_contract_report(path: Path, report: dict[str, object], /) -> Path:
    path.write_text(json.dumps(report, sort_keys=True, allow_nan=False), encoding="utf-8")
    return path


def _first_row(report: dict[str, object], /) -> dict[str, object]:
    rows = report["cases"]
    assert isinstance(rows, list)
    row = rows[0]
    assert isinstance(row, dict)
    return row


def _split_campaign(tmp_path: Path, /) -> tuple[Path, Path, dict[str, object]]:
    """Smoke cases V1-V11 and full-design cases V12-V15, aggregated out of order."""
    smoke = _write_contract_report(
        tmp_path / "isolated.json",
        _report_contract_fixture(
            tuple(row.case_id for row in FUEL_VERIFICATION_CASES[:11])
        ),
    )
    full = _write_contract_report(
        tmp_path / "full.json",
        _report_contract_fixture(
            tuple(row.case_id for row in FUEL_VERIFICATION_CASES[11:]),
            design="default-full",
        ),
    )
    return smoke, full, aggregate_reports((full, smoke))


def test_report_aggregation_orders_cases_and_keeps_literal_designs(
    tmp_path: Path,
) -> None:
    _, _, result = _split_campaign(tmp_path)
    cases = result["cases"]
    assert isinstance(cases, list)
    assert [row["case_id"] for row in cases] == [
        row.case_id for row in FUEL_VERIFICATION_CASES
    ]
    assert result["campaign_designs"] == ("default-full", "smoke")
    assert result["aggregation_scope"] == "retained-observed-reports-no-physics-execution"


def test_report_aggregation_closes_implementation_but_not_release(
    tmp_path: Path,
) -> None:
    _, _, result = _split_campaign(tmp_path)
    implementation = result["implementation_closure"]
    scientific = result["scientific_release_closure"]
    assert isinstance(implementation, dict) and isinstance(scientific, dict)
    assert implementation["implementation_closed"]
    assert not implementation["release_closed"]
    assert not scientific["release_closed"]


def test_report_aggregation_retains_each_report_artifact_location(
    tmp_path: Path,
) -> None:
    smoke, full, result = _split_campaign(tmp_path)
    assert result["artifact_locations"] == [
        {"input_report": str(full), "artifact_directory": _ARTIFACT_DIRECTORY},
        {"input_report": str(smoke), "artifact_directory": _ARTIFACT_DIRECTORY},
    ]


def test_report_aggregation_accepts_numpy_scalar_measurements_as_json_numbers(
    tmp_path: Path,
) -> None:
    report = _report_contract_fixture(
        ("V7",),
        measurement={
            "absolute_error": np.float64(0.125),
            "nested_source_error": [np.float64(0.25)],
        },
    )
    written = _first_row(report)
    path = _write_contract_report(tmp_path / "numpy.json", report)
    observed = aggregate_reports((path,))["cases"]
    assert isinstance(observed, list)
    assert observed[0]["raw_artifact_id"] == written["raw_artifact_id"]
    assert observed[0]["fault_artifact_id"] == written["fault_artifact_id"]
    assert observed[0]["measurement"] == {
        "absolute_error": 0.125,
        "nested_source_error": [0.25],
    }


def test_report_aggregation_refuses_mutated_content_address(tmp_path: Path) -> None:
    report = _report_contract_fixture(
        ("V7",), measurement={"absolute_error": np.float64(0.125)}
    )
    _first_row(report)["measurement"] = {"absolute_error": 0.375}
    path = _write_contract_report(tmp_path / "changed.json", report)
    with pytest.raises(ValueError, match="content address"):
        aggregate_reports((path,))


def test_report_aggregation_refuses_duplicate_case_observations(tmp_path: Path) -> None:
    report = _report_contract_fixture(("V1",))
    first = _write_contract_report(tmp_path / "first.json", report)
    second = _write_contract_report(tmp_path / "second.json", report)
    with pytest.raises(ValueError, match="Duplicate observed case"):
        aggregate_reports((first, second))


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["build", "driver", "environment"])
def test_report_aggregation_refuses_mixed_source_identities(
    tmp_path: Path, axis: int
) -> None:
    identity = list(_CONTRACT_IDENTITY)
    identity[axis] = "foreign-contract-identity"
    first = _write_contract_report(
        tmp_path / "first.json", _report_contract_fixture(("V1",))
    )
    second = _write_contract_report(
        tmp_path / "second.json",
        _report_contract_fixture(
            ("V2",), identity=(identity[0], identity[1], identity[2])
        ),
    )
    with pytest.raises(ValueError, match="mixed global"):
        aggregate_reports((first, second))


@pytest.mark.parametrize(
    ("outcome", "detected", "message"),
    [("failed", True, "numerical case"), ("passed", False, "fault adequacy")],
    ids=["failed-numeric", "undetected-fault"],
)
def test_report_aggregation_refuses_failed_case_or_fault(
    tmp_path: Path, outcome: str, detected: bool, message: str
) -> None:
    path = _write_contract_report(
        tmp_path / "failed.json",
        _report_contract_fixture(("V1",), outcome=outcome, detected=detected),
    )
    with pytest.raises(ValueError, match=message):
        aggregate_reports((path,))


def test_report_aggregation_refuses_source_change_during_campaign(tmp_path: Path) -> None:
    report = _report_contract_fixture(("V1",))
    receipt = report["source_receipt"]
    assert isinstance(receipt, dict)
    receipt["build_after"] = "changed-during-contract-campaign"
    path = _write_contract_report(tmp_path / "changed.json", report)
    with pytest.raises(ValueError, match="source changed during"):
        aggregate_reports((path,))


def test_report_aggregation_refuses_smoke_scope_relabeling(tmp_path: Path) -> None:
    report = _report_contract_fixture(("V1",))
    report["campaign_designs"] = ["default-full"]
    path = _write_contract_report(tmp_path / "changed.json", report)
    with pytest.raises(ValueError, match="literal declaration"):
        aggregate_reports((path,))


@pytest.mark.parametrize("selector", ["--smoke", "--cases", "--artifact-directory"])
def test_report_aggregation_cli_cannot_execute_physics(
    monkeypatch: pytest.MonkeyPatch, selector: str
) -> None:
    values = (
        []
        if selector == "--smoke"
        else ["V1"]
        if selector == "--cases"
        else ["unused-directory"]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nuclear-fuel-qualification",
            "--input-reports",
            "unused.json",
            selector,
            *values,
        ],
    )
    with pytest.raises(SystemExit) as failure:
        main()
    assert failure.value.code == 2


def test_report_aggregation_refuses_reused_numeric_fault_address(tmp_path: Path) -> None:
    report = _report_contract_fixture(("V1",))
    row = _first_row(report)
    row["fault_artifact_id"] = row["raw_artifact_id"]
    path = _write_contract_report(tmp_path / "reused.json", report)
    with pytest.raises(ValueError, match="content address"):
        aggregate_reports((path,))


def test_report_aggregation_rejects_wrong_report_kind(tmp_path: Path) -> None:
    path = tmp_path / "wrong.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="report mapping"):
        aggregate_reports((path,))


def test_report_aggregation_rejects_incomplete_source_receipt(tmp_path: Path) -> None:
    report = _report_contract_fixture(("V1",))
    receipt = report["source_receipt"]
    assert isinstance(receipt, dict)
    del receipt["environment"]
    path = _write_contract_report(tmp_path / "incomplete.json", report)
    with pytest.raises(ValueError, match="source identity fields"):
        aggregate_reports((path,))
