"""Pure campaign aggregation and metadata parsing; never launch GPU children."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from benchmarks.cuest_molecular_campaign import (
    _driver_evidence_fields,
    _load_child_record,
    aggregate_cases,
    capacity_grid,
)


_IDENTITY = {
    "fingerprint": "current-identity",
    "source_build_fingerprint": "current-source",
    "benchmark_driver_fingerprint": "current-driver",
    "evidence_schema": {"fields": ["scientific_checks"], "signature": "current-schema"},
}
_CHECKS = {
    "reference_converged": True,
    "native_jk_successful": True,
    "energies_finite": True,
    "forces_finite": True,
    "coulomb_finite": True,
    "exchange_finite": True,
    "matched_df_energy": True,
    "matched_df_forces": True,
    "matched_df_coulomb": True,
    "matched_df_exchange": True,
}


def _compiler_fixture(repeats: int) -> dict[str, Any]:
    kernels: dict[str, Any] = {}
    for name, observables in {
        "one_electron": ("overlap", "kinetic", "nuclear_attraction"),
        "jk": ("coulomb", "exchange"),
        "gradient": ("one_electron", "two_electron", "pulay", "total_gradient", "forces"),
    }.items():
        checks = {"status_zero": True}
        checks.update(
            {
                f"{observable}_{suffix}": True
                for observable in observables
                for suffix in ("finite", "matches_host")
            }
        )
        sample = {"scientific_checks": checks, "maximum_absolute_errors": {}}
        kernels[name] = {
            "timings_seconds": {
                "lowering": 0.1,
                "compile": 0.1,
                "first_execution": 0.1,
                "warm": {"count": repeats, "samples_seconds": [0.1] * repeats},
            },
            "compiler": {
                "source": "unit-test compiler record",
                "argument_bytes": 32,
                "output_bytes": 32,
                "temporary_bytes": 0,
            },
            "first_validation": deepcopy(sample),
            "warm_validation": [deepcopy(sample) for _ in range(repeats)],
        }
    return {
        "scientific_checks": {"all_first_and_warm_samples_qualified": True},
        "kernels": kernels,
    }


def _rows() -> list[dict[str, Any]]:
    rows = []
    for capacity, orbitals, auxiliaries in zip(
        capacity_grid((0, 1), (1,), repeats=2), (2, 8), (10, 90), strict=True
    ):
        rows.append(
            {
                "requested_capacity": capacity,
                "returncode": 0,
                "output_path": f"angular-{capacity['angular']}.json",
                "stdout_path": f"angular-{capacity['angular']}.stdout.log",
                "stderr_path": f"angular-{capacity['angular']}.stderr.log",
                "record": {
                    "provider": "cuEST",
                    "identity": deepcopy(_IDENTITY),
                    "requested_capacity": dict(capacity),
                    "status": "completed",
                    "scientific_checks": dict(_CHECKS),
                    "independent_reference": {"status": "qualified"},
                    "orbital_count": orbitals,
                    "auxiliary_count": auxiliaries,
                    "execution_flags": {"CUDA_VISIBLE_DEVICES": "0"},
                    "environment": {"fingerprint": "unit-test environment"},
                    "dependencies": {
                        name: "unit-test version"
                        for name in (
                            "phydrax",
                            "phydrax-cuda13",
                            "phydrax-cuest",
                            "jax",
                            "jaxlib",
                            "pyscf",
                        )
                    },
                    "native_component": {
                        "build_hash": "unit-test native build",
                        "companion_binary": {"sha256": "0" * 64, "bytes": 32},
                        "vendor_binary": {"sha256": "1" * 64, "bytes": 64},
                    },
                    "selected_devices": [
                        {"platform": "cuda", "id": 0, "kind": "unit-test device"}
                    ],
                    "compiled_kernels": _compiler_fixture(capacity["repeats"]),
                },
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return aggregate_cases(
        rows,
        angulars=(0, 1),
        molecule_counts=(1,),
        repeats=2,
        expected_child_identity=_IDENTITY,
    )


def test_capacity_grid_has_canonical_angular_major_order() -> None:
    assert capacity_grid((2, 0, 1, 0), (4, 1, 2, 1), repeats=3) == tuple(
        {"angular": angular, "molecules": molecules, "repeats": 3}
        for angular in (0, 1, 2)
        for molecules in (1, 2, 4)
    )


@pytest.mark.parametrize(
    "angulars,molecules,repeats",
    (
        ((), (1,), 1),
        ((-1,), (1,), 1),
        ((3,), (1,), 1),
        ((True,), (1,), 1),
        ((0,), (), 1),
        ((0,), (0,), 1),
        ((0,), (-2,), 1),
        ((0,), (True,), 1),
        ((0,), (1,), 0),
        ((0,), (1,), True),
    ),
)
def test_capacity_grid_rejects_invalid_capacity_lists(
    angulars: tuple[int, ...], molecules: tuple[int, ...], repeats: int
) -> None:
    with pytest.raises(ValueError):
        capacity_grid(angulars, molecules, repeats=repeats)


def test_campaign_completion_requires_and_retains_all_qualified_children() -> None:
    rows = _rows()
    result = _aggregate(rows)
    assert result["status"] == "completed"
    assert all(result["qualification_checks"].values())
    assert result["controlling_capacities"]["qualified_observed_orbital_counts"] == [2, 8]
    assert result["controlling_capacities"]["qualified_observed_auxiliary_counts"] == [
        10,
        90,
    ]
    for retained, original in zip(result["cases"], rows, strict=True):
        assert retained["record"] == original["record"]
        assert retained["stdout_path"] == original["stdout_path"]
        assert retained["stderr_path"] == original["stderr_path"]
        assert retained["status"] == "qualified"
        assert retained["qualification_errors"] == []
        assert "qualification_errors" not in original
    assert "No single-H2 speedup extrapolation" in result["qualification_scope"]


@pytest.mark.parametrize("field", ("orbital_count", "auxiliary_count"))
def test_campaign_requires_both_observed_capacities_to_vary(field: str) -> None:
    rows = _rows()
    rows[1]["record"][field] = rows[0]["record"][field]
    result = _aggregate(rows)
    assert result["status"] == "unqualified"
    assert result["qualification_checks"]["all_children_scientifically_qualified"]
    assert not result["qualification_checks"][
        field.removesuffix("_count") + "_capacity_varies"
    ]


def test_one_h2_cannot_qualify_as_a_capacity_campaign() -> None:
    result = aggregate_cases(
        _rows()[:1],
        angulars=(0,),
        molecule_counts=(1,),
        repeats=2,
        expected_child_identity=_IDENTITY,
    )
    assert result["status"] == "unqualified"
    assert not result["qualification_checks"]["orbital_capacity_varies"]
    assert not result["qualification_checks"]["auxiliary_capacity_varies"]


@pytest.mark.parametrize("returncode", (2, -9, None, False))
def test_child_process_failure_cannot_be_hidden_by_completed_json(
    returncode: int | None,
) -> None:
    rows = _rows()
    rows[0]["returncode"] = returncode
    result = _aggregate(rows)
    assert result["status"] == "failed"
    assert result["cases"][0]["returncode"] is returncode
    assert result["cases"][0]["qualification_errors"]
    assert result["cases"][1]["status"] == "qualified"


@pytest.mark.parametrize(
    "field,value",
    (
        ("provider", "other"),
        ("identity", None),
        ("identity", {"source_build_fingerprint": "stale"}),
        ("requested_capacity", {"angular": 0, "molecules": 99, "repeats": 2}),
        ("requested_capacity", {"angular": False, "molecules": 1, "repeats": 2}),
        ("status", "unqualified"),
        ("scientific_checks", {}),
        ("scientific_checks", {**_CHECKS, "matched_df_energy": False}),
        ("scientific_checks", {**_CHECKS, "matched_df_energy": 1}),
        ("scientific_checks", {**_CHECKS, "unexpected_check": True}),
        ("independent_reference", {"status": "dependency_absent"}),
        ("orbital_count", 0),
        ("orbital_count", True),
        ("auxiliary_count", 1.5),
    ),
)
def test_campaign_rejects_stale_malformed_or_unqualified_child_data(
    field: str, value: object
) -> None:
    rows = _rows()
    rows[0]["record"][field] = value
    result = _aggregate(rows)
    assert result["status"] == "failed"
    assert result["cases"][0]["status"] == "failed"
    assert result["cases"][0]["record"][field] == value
    assert result["cases"][1]["status"] == "qualified"


@pytest.mark.parametrize("record", (None, [], "not an object", 1))
def test_missing_or_nonobject_child_record_is_retained_and_fails(record: object) -> None:
    rows = _rows()
    rows[0]["record"] = record
    rows[0]["record_error"] = "Unreadable child artifact"
    result = _aggregate(rows)
    assert result["status"] == "failed"
    assert result["cases"][0]["record"] == record
    assert "Unreadable child artifact" in result["cases"][0]["qualification_errors"]


def test_campaign_rejects_missing_extra_duplicate_or_reordered_rows() -> None:
    rows = _rows()
    for invalid in (
        [],
        rows[:1],
        rows + rows[:1],
        [rows[0], rows[0]],
        list(reversed(rows)),
    ):
        result = _aggregate(invalid)
        assert result["status"] == "failed"
        assert len(result["cases"]) == len(invalid)


def test_controller_row_capacity_must_match_canonical_request() -> None:
    rows = _rows()
    rows[0]["requested_capacity"] = {"angular": 1, "molecules": 1, "repeats": 2}
    assert _aggregate(rows)["status"] == "failed"


@pytest.mark.parametrize(
    "payload",
    (
        '{"status":',
        '{"orbital_count": NaN}',
        '{"value": Infinity}',
        '{"value": 1e400}',
        "\xff",
    ),
)
def test_malformed_child_json_is_an_error_not_a_success(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "child.json"
    path.write_bytes(payload.encode("latin-1"))
    record, error = _load_child_record(path)
    assert record is None
    assert error
    assert path.read_bytes() == payload.encode("latin-1")


def test_missing_child_json_is_an_error(tmp_path: Path) -> None:
    record, error = _load_child_record(tmp_path / "absent.json")
    assert record is None
    assert error is not None
    assert "FileNotFoundError" in error


def test_child_json_payload_is_retained(tmp_path: Path) -> None:
    path = tmp_path / "child.json"
    path.write_text('{"status": "failed", "error": "scientific gate"}', encoding="utf-8")
    record, error = _load_child_record(path)
    assert record == {"status": "failed", "error": "scientific gate"}
    assert error is None


def test_driver_schema_is_read_without_importing_driver(tmp_path: Path) -> None:
    path = tmp_path / "driver.py"
    path.write_text(
        'raise RuntimeError("must not execute")\n_EVIDENCE_FIELDS = ("identity", "status")\n',
        encoding="utf-8",
    )
    assert _driver_evidence_fields(path) == ("identity", "status")


@pytest.mark.parametrize(
    "source",
    (
        '_EVIDENCE_FIELDS = ["status"]',
        "_EVIDENCE_FIELDS = make_fields()",
        'OTHER = ("status",)',
    ),
)
def test_driver_schema_missing_or_nonliteral_is_rejected(
    tmp_path: Path, source: str
) -> None:
    path = tmp_path / "driver.py"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError):
        _driver_evidence_fields(path)


@pytest.mark.parametrize(
    "field",
    (
        "environment",
        "dependencies",
        "native_component",
        "selected_devices",
        "compiled_kernels",
    ),
)
def test_campaign_rejects_missing_required_measurement_provenance(field: str) -> None:
    rows = _rows()
    rows[0]["record"].pop(field)
    result = _aggregate(rows)
    assert result["status"] == "failed"
    assert result["cases"][0]["qualification_errors"]


@pytest.mark.parametrize("phase", ("lowering", "compile", "first_execution"))
def test_campaign_rejects_missing_real_compiler_phase(phase: str) -> None:
    rows = _rows()
    rows[0]["record"]["compiled_kernels"]["kernels"]["jk"]["timings_seconds"][phase] = (
        None
    )
    assert _aggregate(rows)["status"] == "failed"


def test_campaign_requires_every_measured_sample_to_be_qualified() -> None:
    rows = _rows()
    sample = rows[0]["record"]["compiled_kernels"]["kernels"]["gradient"][
        "warm_validation"
    ][0]
    sample["scientific_checks"]["forces_matches_host"] = False
    assert _aggregate(rows)["status"] == "failed"


def test_unavailable_compiler_memory_requires_actual_api_reason() -> None:
    rows = _rows()
    compiler = rows[0]["record"]["compiled_kernels"]["kernels"]["jk"]["compiler"]
    for field in ("argument_bytes", "output_bytes", "temporary_bytes"):
        compiler[field] = None
    assert _aggregate(rows)["status"] == "failed"
    compiler["unavailable_reason"] = "Compiled.memory_analysis() returned None"
    assert _aggregate(rows)["status"] == "completed"


@pytest.mark.parametrize("field", ("execution_flags", "dependencies", "selected_devices"))
def test_capacity_series_cannot_mix_runtime_or_device_profiles(field: str) -> None:
    rows = _rows()
    if field == "execution_flags":
        rows[1]["record"][field]["CUDA_VISIBLE_DEVICES"] = "1"
    elif field == "dependencies":
        rows[1]["record"][field]["jaxlib"] = "different-unit-test-version"
    else:
        rows[1]["record"][field][0]["id"] = 1
    result = _aggregate(rows)
    assert result["status"] == "unqualified"
    assert result["qualification_checks"]["all_children_scientifically_qualified"]
    assert not result["qualification_checks"]["comparison_environment_consistent"]
