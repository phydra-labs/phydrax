#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phydrax._array_archive import ArrayArchiveCorruptionError, read_array_archive
from phydrax.lifecycle._archive import (
    collection_digest,
    create,
    open as open_lifecycle_archive,
    support_bundle,
    SupportBundleAuthorization,
)
from phydrax.lifecycle._models import NumericRevision


_FULL_DISCLOSURE = frozenset(
    {"arrays", "payloads", "paths", "identifiers", "free-text", "secrets"}
)


def _sensitive_archive(tmp_path: Path):
    values = np.asarray((1.0, 2.0, 3.0))
    revision = NumericRevision(
        collection_digest({"patient-waveform": values}),
        label="patient Jane Doe free text",
        metadata={
            "source_path": "/private/patient/jane.npy",
            "api_secret": "support-secret-value",
            "external_id": "patient-12345",
        },
    )
    return create(
        tmp_path / "patient-12345-source.zip",
        manifest=revision,
        arrays={"patient-waveform": values},
    )


def test_support_bundle_is_recursively_allowlisted_and_payload_free_by_default(
    tmp_path: Path,
):
    source = _sensitive_archive(tmp_path)
    destination = support_bundle(source, tmp_path / "sanitized-support.zip")

    manifest, arrays = read_array_archive(destination)
    raw_bundle = destination.read_bytes()

    assert manifest == {
        "kind": "lifecycle-support-bundle",
        "disclosure": "sanitized",
        "telemetry": {
            "record": {"kind": "numeric-revision"},
            "archive": {"array_count": 1, "array_bytes": 24},
        },
        "audit": {"data_owner_authorized": False},
        "arrays": {},
    }
    assert arrays == {}
    for forbidden in (
        b"patient-waveform",
        b"patient-12345",
        b"Jane Doe",
        b"/private/patient/jane.npy",
        b"support-secret-value",
        source.archive_id.encode(),
        source.manifest.revision_id.encode(),
    ):
        assert forbidden not in raw_bundle


def test_full_support_payload_requires_explicit_complete_owner_authorization(
    tmp_path: Path,
):
    source = _sensitive_archive(tmp_path)

    with pytest.raises(ValueError, match="every sensitive disclosure"):
        SupportBundleAuthorization(
            "authorization-1",
            "data-owner-1",
            source.archive_id,
            1_700_000_000,
            frozenset({"arrays", "payloads"}),
        )

    authorization = SupportBundleAuthorization(
        "authorization-1",
        "data-owner-1",
        source.archive_id,
        1_700_000_000,
        _FULL_DISCLOSURE,
    )
    wrong_source = SupportBundleAuthorization(
        "authorization-wrong-source",
        "data-owner-1",
        "0" * 64,
        1_700_000_000,
        _FULL_DISCLOSURE,
    )
    with pytest.raises(ValueError, match="not bound"):
        support_bundle(
            source,
            tmp_path / "wrong-source-support.zip",
            authorization=wrong_source,
        )

    destination = support_bundle(
        source,
        tmp_path / "authorized-support.zip",
        authorization=authorization,
    )
    manifest, arrays = read_array_archive(destination)

    assert manifest["disclosure"] == "data-owner-authorized"
    assert manifest["audit"]["data_owner_authorized"] is True
    assert manifest["audit"]["authorization_id"] == "authorization-1"
    assert manifest["audit"]["data_owner_id"] == "data-owner-1"
    assert manifest["audit"]["source_archive_id"] == source.archive_id
    assert manifest["audit"]["authorized_at"] == 1_700_000_000
    assert manifest["audit"]["disclosures"] == sorted(_FULL_DISCLOSURE)
    assert len(manifest["audit"]["authorization_fingerprint"]) == 64
    assert manifest["source"]["archive_id"] == source.archive_id
    assert arrays["archive"].tobytes() == source.path.read_bytes()


def test_numeric_revision_archive_requires_materialized_content(tmp_path: Path):
    revision = NumericRevision("0" * 64, label="unbacked")
    with pytest.raises(ValueError, match="materialized payload"):
        create(tmp_path / "unbacked.zip", manifest=revision, arrays={})


def test_parented_numeric_revision_requires_exact_parent_archive(tmp_path: Path):
    parent_values = np.asarray((1.0,))
    parent_revision = NumericRevision(
        collection_digest({"state": parent_values}), label="parent"
    )
    parent = create(
        tmp_path / "parent.zip",
        manifest=parent_revision,
        arrays={"state": parent_values},
    )
    child_values = np.asarray((2.0,))
    child_revision = NumericRevision(
        collection_digest({"state": child_values}),
        label="child",
        parent_digest=parent_revision.content_digest,
        parent_revision_id=parent_revision.revision_id,
    )
    with pytest.raises(ValueError, match="parent archive"):
        create(
            tmp_path / "child-without-parent.zip",
            manifest=child_revision,
            arrays={"state": child_values},
        )
    child = create(
        tmp_path / "child.zip",
        manifest=child_revision,
        arrays={"state": child_values},
        parent=parent,
    )
    assert child.manifest.parent_revision_id == parent_revision.revision_id
    with pytest.raises(ValueError, match="parent archive"):
        open_lifecycle_archive(child.path)
    assert (
        open_lifecycle_archive(child.path, parent=parent).archive_id == child.archive_id
    )


def test_authorized_support_copy_revalidates_source_snapshot(tmp_path: Path):
    source = _sensitive_archive(tmp_path)
    authorization = SupportBundleAuthorization(
        "authorization-1",
        "data-owner-1",
        source.archive_id,
        1_700_000_000,
        _FULL_DISCLOSURE,
    )
    replacement_values = np.asarray((9.0,))
    replacement = NumericRevision(
        collection_digest({"replacement": replacement_values}),
        label="replacement",
    )
    source.path.unlink()
    create(
        source.path,
        manifest=replacement,
        arrays={"replacement": replacement_values},
    )
    with pytest.raises(
        ArrayArchiveCorruptionError, match="changed after support-bundle admission"
    ):
        support_bundle(
            source,
            tmp_path / "replaced-support.zip",
            authorization=authorization,
        )
