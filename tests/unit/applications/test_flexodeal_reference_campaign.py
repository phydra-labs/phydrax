# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Evidence-integrity checks; synthetic tables are not physiological oracles."""

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from tools.flexodeal_reference_campaign import validate_history, verify_files


ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "tests/fixtures/flexodeal_0698e3d/reference_outputs"


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def test_frozen_executable_packages_match_content_identified_manifest():
    manifest = json.loads((FROZEN / "manifest.json").read_text())
    payload = dict(manifest)
    content_id = payload.pop("content_id")
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == content_id

    records = [
        *manifest["packages"],
        {
            **manifest["material_points"],
            "retained_members": [
                manifest["material_points"]["source_record_member"],
                manifest["material_points"]["inputs_member"],
                manifest["material_points"]["observations_member"],
            ],
        },
    ]
    for record in records:
        archive_path = ROOT / record["path"]
        assert archive_path.stat().st_size == record["byte_count"]
        assert _digest(archive_path) == record["sha256"]
        with tarfile.open(archive_path, "r:gz") as archive:
            names = set(archive.getnames())
            assert set(record["retained_members"]) <= names
            source_record = json.load(
                archive.extractfile(
                    record.get("full_raw_artifact_inventory_member", "run-record.json")
                )
            )
        assert source_record["content_id"] == record["source_record_content_id"]


def test_reference_asset_rejects_same_size_content_tampering(tmp_path):
    path = tmp_path / "activation.dat"
    original = b"0.0 0.0\n1.0 1.0\n"
    path.write_bytes(original)
    entries = {
        path.name: {
            "bytes": len(original),
            "sha256": hashlib.sha256(original).hexdigest(),
        }
    }
    verify_files(tmp_path, entries)
    path.write_bytes(original.replace(b"1.0 1.0", b"1.0 0.0"))
    with pytest.raises(ValueError, match="content mismatch"):
        verify_files(tmp_path, entries)


def test_reference_asset_rejects_indirect_mutable_source(tmp_path):
    original = tmp_path / "source.dat"
    original.write_bytes(b"0 0\n")
    link = tmp_path / "reference.dat"
    link.symlink_to(original)
    entries = {
        link.name: {
            "bytes": 4,
            "sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        }
    }
    with pytest.raises(ValueError, match="symbolic reference asset"):
        verify_files(tmp_path, entries)


@pytest.mark.parametrize(
    "last_row, expected_error",
    [
        ("", "Incomplete reference history"),
        ("0.015,2,1e-4\n", "Wrong reference time grid"),
        ("0.02,nan,1e-4\n", "Nonfinite reference history"),
        ("0.02,2,-1e-4\n", "Nonpositive reference volume"),
        ("0.02,2\n", "Malformed reference row"),
    ],
)
def test_reference_history_rejects_incomplete_or_invalid_output(
    tmp_path, last_row, expected_error
):
    path = tmp_path / "force.csv"
    beginning = "Time [s],Total [N],Volume [m^3]\n0,0,1e-4\n0.01,1,1e-4\n"
    times = [0.0, 0.01, 0.02]
    path.write_text(beginning + "0.02,2,1e-4\n")
    assert validate_history(path, times) == {"samples": 3, "start_s": 0.0, "end_s": 0.02}
    path.write_text(beginning + last_row)
    with pytest.raises(ValueError, match=expected_error):
        validate_history(path, times)
