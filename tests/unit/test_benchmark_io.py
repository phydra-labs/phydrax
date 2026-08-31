#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks._io import atomic_write, write_json_atomic


def test_json_output_is_sorted_finite_and_newline_terminated(tmp_path):
    destination = tmp_path / "nested" / "artifact.json"

    observed = write_json_atomic(destination, {"z": 2, "a": [1, 3]})

    assert observed == destination
    payload = destination.read_text(encoding="utf-8")
    assert payload.endswith("\n")
    assert payload.index('"a"') < payload.index('"z"')
    assert json.loads(payload) == {"a": [1, 3], "z": 2}


def test_nonfinite_json_leaves_existing_destination_untouched(tmp_path):
    destination = tmp_path / "artifact.json"
    destination.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError):
        write_json_atomic(destination, {"invalid": float("nan")})

    assert destination.read_text(encoding="utf-8") == "existing\n"


def test_writer_failure_preserves_destination_and_removes_temporary_file(tmp_path):
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"existing")

    def failing_writer(temporary: Path) -> None:
        temporary.write_bytes(b"partial")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        atomic_write(destination, failing_writer)

    assert destination.read_bytes() == b"existing"
    assert tuple(tmp_path.glob(".artifact.bin.*.tmp")) == ()


def test_successful_atomic_write_replaces_existing_content(tmp_path):
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"old")

    atomic_write(destination, lambda temporary: temporary.write_bytes(b"new"))

    assert destination.read_bytes() == b"new"
    assert tuple(tmp_path.glob(".artifact.bin.*.tmp")) == ()
