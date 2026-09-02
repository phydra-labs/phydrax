#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
from collections.abc import Mapping

import pytest

from phydrax.lifecycle._migration import (
    AmbiguousMigrationError,
    CompatibilityRegistry,
    CyclicMigrationError,
    load_and_resolve_migration,
    LossyMigrationError,
    MigrationEdge,
    MigrationPurityError,
    MigrationReport,
    UnsupportedMigrationError,
)


def _rename_value(record: Mapping[str, object]) -> Mapping[str, object]:
    return {"renamed": record["value"]}


def _wrap_value(record: Mapping[str, object]) -> Mapping[str, object]:
    return {"payload": record["renamed"]}


def _direct_value(record: Mapping[str, object]) -> Mapping[str, object]:
    return {"payload": record["value"]}


def _identity_copy(record: Mapping[str, object]) -> Mapping[str, object]:
    return dict(record)


def _mutate_input(record: Mapping[str, object]) -> Mapping[str, object]:
    record["mutated"] = True  # type: ignore[index]
    return {"value": record["value"]}


def _edges() -> tuple[MigrationEdge, MigrationEdge]:
    return (
        MigrationEdge(
            "format-v1",
            "format-v2",
            _rename_value,
            migration_id="rename-value",
        ),
        MigrationEdge(
            "format-v2",
            "format-v3",
            _wrap_value,
            migration_id="wrap-value",
        ),
    )


def test_registry_migrates_to_current_writer_with_digest_lineage() -> None:
    first, second = _edges()
    registry = CompatibilityRegistry("format-v3", (second, first))

    report = registry.resolve({"value": 7}, source_format_id="format-v1")

    assert report.output_format_id == "format-v3"
    assert report.output_record == {"payload": 7}
    assert report.migration_ids == ("rename-value", "wrap-value")
    assert report.lineage == (
        report.input_digest,
        report.lineage[1],
        report.output_digest,
    )
    assert len(set(report.lineage)) == 3
    assert not report.lossy


def test_registry_preserves_existing_lineage_and_requires_continuity() -> None:
    registry = CompatibilityRegistry("format-v3", _edges())
    initial = registry.resolve({"value": 7}, source_format_id="format-v1")
    ancestor = "0" * 64

    report = registry.resolve(
        {"value": 7},
        source_format_id="format-v1",
        lineage=(ancestor, initial.input_digest),
    )

    assert report.lineage[:2] == (ancestor, initial.input_digest)
    assert registry.rollback(report)["lineage"] == [ancestor, initial.input_digest]
    with pytest.raises(ValueError, match="does not terminate"):
        registry.resolve(
            {"value": 7},
            source_format_id="format-v1",
            lineage=(ancestor,),
        )


def test_registry_selects_unique_shortest_path_deterministically() -> None:
    first, second = _edges()
    direct = MigrationEdge(
        "format-v1",
        "format-v3",
        _direct_value,
        migration_id="direct-value",
    )
    registry = CompatibilityRegistry("format-v3", (second, direct, first))

    report = registry.resolve({"value": 11}, source_format_id="format-v1")

    assert report.migration_ids == ("direct-value",)
    assert report.output_record == {"payload": 11}


def test_registry_rejects_ambiguous_and_unsupported_paths() -> None:
    edges = (
        MigrationEdge("v1", "left", _identity_copy, migration_id="to-left"),
        MigrationEdge("left", "v3", _identity_copy, migration_id="left-current"),
        MigrationEdge("v1", "right", _identity_copy, migration_id="to-right"),
        MigrationEdge("right", "v3", _identity_copy, migration_id="right-current"),
    )
    registry = CompatibilityRegistry("v3", edges)

    with pytest.raises(AmbiguousMigrationError, match="multiple shortest"):
        registry.resolve({"value": 1}, source_format_id="v1")
    with pytest.raises(UnsupportedMigrationError, match="cannot migrate"):
        registry.resolve({"value": 1}, source_format_id="unknown")


def test_registry_rejects_cycles_and_outgoing_current_writer_edges() -> None:
    cycle = (
        MigrationEdge("v1", "v2", _identity_copy, migration_id="forward"),
        MigrationEdge("v2", "v1", _identity_copy, migration_id="backward"),
    )

    with pytest.raises(CyclicMigrationError, match="acyclic"):
        CompatibilityRegistry("v3", cycle)
    with pytest.raises(ValueError, match="current writer"):
        CompatibilityRegistry(
            "v3",
            (MigrationEdge("v3", "v2", _identity_copy, migration_id="reverse-write"),),
        )


def test_lossy_migration_requires_explicit_authorization() -> None:
    registry = CompatibilityRegistry(
        "v2",
        (
            MigrationEdge(
                "v1",
                "v2",
                _identity_copy,
                migration_id="lossy-transition",
                lossy=True,
            ),
        ),
    )

    with pytest.raises(LossyMigrationError, match="authorization"):
        registry.resolve({"value": 1}, source_format_id="v1")
    report = registry.resolve({"value": 1}, source_format_id="v1", allow_lossy=True)
    assert report.lossy


def test_report_reconstruction_and_rollback_select_the_parent_artifact() -> None:
    registry = CompatibilityRegistry("format-v3", _edges())
    report = registry.resolve({"value": 19}, source_format_id="format-v1")

    restored = MigrationReport.from_json(report.to_json())
    parent = registry.rollback(restored)

    assert restored.report_id == report.report_id
    assert restored.to_json() == report.to_json()
    assert parent == {
        "format_id": "format-v1",
        "record": {"value": 19},
        "artifact_id": report.input_digest,
        "lineage": [report.input_digest],
    }
    parent["record"]["value"] = -1
    assert report.input_record == {"value": 19}
    assert restored.rollback_artifact_id == report.input_digest


def test_registry_identity_is_independent_of_edge_order() -> None:
    first, second = _edges()

    left = CompatibilityRegistry("format-v3", (first, second))
    right = CompatibilityRegistry("format-v3", (second, first))

    assert left.registry_id == right.registry_id
    assert left.to_record() == right.to_record()


def test_canonical_load_is_strict_and_rejects_nonfinite_values() -> None:
    registry = CompatibilityRegistry("format-v3", _edges())
    request = {
        "format_id": "format-v1",
        "record": {"value": 3},
        "lineage": [],
    }

    report = load_and_resolve_migration(registry, json.dumps(request))

    assert report.output_record == {"payload": 3}
    with pytest.raises(ValueError, match="unknown fields"):
        registry.load(json.dumps({**request, "source": "alias"}))
    missing = dict(request)
    del missing["format_id"]
    with pytest.raises(ValueError, match="missing fields"):
        registry.load(json.dumps(missing))
    with pytest.raises(ValueError, match="Non-finite"):
        registry.load(
            json.dumps(
                {
                    "format_id": "format-v1",
                    "record": {"value": float("inf")},
                    "lineage": [],
                }
            )
        )


def test_migration_transform_cannot_rewrite_its_input_in_place() -> None:
    registry = CompatibilityRegistry(
        "v2",
        (MigrationEdge("v1", "v2", _mutate_input, migration_id="mutating-transition"),),
    )
    original = {"value": 1}

    with pytest.raises(MigrationPurityError, match="mutated"):
        registry.resolve(original, source_format_id="v1")
    assert original == {"value": 1}
