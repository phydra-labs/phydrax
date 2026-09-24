#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phydrax._array_archive import (
    array_collection_digest,
    ArrayArchiveCorruptionError,
    read_array_archive,
    write_array_archive,
)
from phydrax._fingerprint import canonical_fingerprint
from phydrax._identity import (
    ArtifactBindingIdentity,
    ExecutableSignature,
    NumericRevision,
    SemanticProvenance,
)
from phydrax.lifecycle._archive import (
    create,
    open as open_lifecycle_archive,
    payload_digest,
    support_bundle,
    SupportBundleAuthorization,
)
from phydrax.lifecycle._models import ModelManifest, RevisionLineage


_FULL_DISCLOSURE = frozenset(
    {"arrays", "payloads", "paths", "identifiers", "free-text", "secrets"}
)


_SEMANTIC = SemanticProvenance({"kind": "lifecycle-test-state"})


def _lineage(arrays, /, **kwargs) -> RevisionLineage:
    return RevisionLineage(NumericRevision(_SEMANTIC, arrays), **kwargs)


def _sensitive_archive(tmp_path: Path):
    values = np.asarray((1.0, 2.0, 3.0))
    revision = _lineage(
        {"patient-waveform": values},
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
            "record": {"kind": "revision-lineage"},
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
        source.manifest.lineage_id.encode(),
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


def test_revision_lineage_archive_requires_materialized_content(tmp_path: Path):
    lineage = _lineage({"state": np.asarray((1.0,))}, label="unbacked")
    with pytest.raises(ValueError, match="materialized payload"):
        create(tmp_path / "unbacked.zip", manifest=lineage, arrays={})
    with pytest.raises(ValueError, match="canonical numeric revision"):
        create(
            tmp_path / "other-content.zip",
            manifest=lineage,
            arrays={"state": np.asarray((2.0,))},
        )


def test_revision_lineage_round_trips_canonical_revision_ids(tmp_path: Path):
    parent_arrays = {"state": np.asarray((1.0, 2.0))}
    parent_revision = NumericRevision(_SEMANTIC, parent_arrays)
    parent = create(
        tmp_path / "parent.zip",
        manifest=RevisionLineage(parent_revision, label="parent"),
        arrays=parent_arrays,
    )
    child_arrays = {"state": np.asarray((3.0, 4.0)), "count": np.asarray(2)}
    child_revision = NumericRevision(_SEMANTIC, child_arrays)
    child_lineage = RevisionLineage(
        child_revision,
        label="child",
        parent_revision_id=parent.manifest.revision_id,
        parent_lineage_id=parent.manifest.lineage_id,
        metadata={"round": "2"},
    )
    child = create(
        tmp_path / "child.zip",
        manifest=child_lineage,
        arrays=child_arrays,
        parent=parent,
    )

    reopened = open_lifecycle_archive(child.path, parent=parent)
    assert reopened.manifest == child_lineage
    assert reopened.manifest.revision_id == child_revision.revision_id
    assert reopened.manifest.semantic_id == _SEMANTIC.semantic_id
    assert reopened.manifest.parent_revision_id == parent_revision.revision_id
    # The archive payload is the canonical numeric content of the revision.
    assert (
        NumericRevision(_SEMANTIC, dict(reopened.arrays)).revision_id
        == child_revision.revision_id
    )
    with pytest.raises(ValueError, match="supplied together"):
        RevisionLineage(child_revision, parent_revision_id=parent_revision.revision_id)


def test_parented_revision_lineage_requires_exact_parent_archive(tmp_path: Path):
    parent_values = {"state": np.asarray((1.0,))}
    parent_lineage = _lineage(parent_values, label="parent")
    parent = create(
        tmp_path / "parent.zip", manifest=parent_lineage, arrays=parent_values
    )
    sibling_values = {"state": np.asarray((5.0,))}
    sibling = create(
        tmp_path / "sibling.zip",
        manifest=_lineage(sibling_values, label="sibling"),
        arrays=sibling_values,
    )
    child_values = {"state": np.asarray((2.0,))}
    child_lineage = _lineage(
        child_values,
        label="child",
        parent_revision_id=parent_lineage.revision_id,
        parent_lineage_id=parent_lineage.lineage_id,
    )
    with pytest.raises(ValueError, match="parent archive"):
        create(
            tmp_path / "child-without-parent.zip",
            manifest=child_lineage,
            arrays=child_values,
        )
    with pytest.raises(ValueError, match="parent archive identity"):
        create(
            tmp_path / "child-wrong-parent.zip",
            manifest=child_lineage,
            arrays=child_values,
            parent=sibling,
        )
    child = create(
        tmp_path / "child.zip",
        manifest=child_lineage,
        arrays=child_values,
        parent=parent,
    )
    assert child.manifest.parent_lineage_id == parent_lineage.lineage_id
    with pytest.raises(ValueError, match="parent archive"):
        open_lifecycle_archive(child.path)
    assert (
        open_lifecycle_archive(child.path, parent=parent).archive_id == child.archive_id
    )


def test_legacy_numeric_revision_archive_fails_closed(tmp_path: Path):
    values = np.asarray((1.0, 2.0))
    legacy_record = {
        "kind": "numeric-revision",
        "content_digest": array_collection_digest({"state": values}),
        "label": "legacy",
        "parent_digest": None,
        "parent_revision_id": None,
        "metadata": [],
        "revision_id": "0" * 64,
    }
    record_digest = canonical_fingerprint(legacy_record)
    path = tmp_path / "legacy.zip"
    # A self-consistent container: only the retired record kind is wrong.
    write_array_archive(
        path,
        manifest={
            "kind": "lifecycle-archive",
            "record": legacy_record,
            "record_digest": record_digest,
            "archive_id": canonical_fingerprint(
                {
                    "kind": "lifecycle-archive",
                    "record_digest": record_digest,
                    "payload_digest": array_collection_digest({"state": values}),
                }
            ),
        },
        arrays={"state": values},
    )
    with pytest.raises(ArrayArchiveCorruptionError, match="Unknown lifecycle record"):
        open_lifecycle_archive(path)


def test_model_manifest_binding_identity_round_trips(tmp_path: Path):
    values = np.arange(3.0)
    revision = NumericRevision(_SEMANTIC, {"weights": values})
    signature = ExecutableSignature(shapes={"weights": (3,)}, dtypes={"weights": "f8"})
    binding = ArtifactBindingIdentity(_SEMANTIC, revision, signature)
    manifest = ModelManifest(
        "model",
        "analysis",
        revision.revision_id,
        {"weights": payload_digest(values)},
        binding=binding,
    )
    unbound = ModelManifest(
        "model", "analysis", revision.revision_id, {"weights": payload_digest(values)}
    )
    assert manifest.manifest_id != unbound.manifest_id

    archive = create(tmp_path / "model.zip", manifest=manifest, arrays={"weights": values})
    reopened = open_lifecycle_archive(archive.path)
    assert reopened.manifest.binding == binding
    assert reopened.manifest.manifest_id == manifest.manifest_id
    with pytest.raises(ValueError, match="manifest numeric revision"):
        ModelManifest(
            "model",
            "analysis",
            "other-revision",
            {"weights": payload_digest(values)},
            binding=binding,
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
    replacement = _lineage({"replacement": replacement_values}, label="replacement")
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
