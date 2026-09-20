#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from phydrax._external_resource import (
    bounded_resource_from_bytes,
    open_bounded_resource,
    ResourceLimits,
    ResourceReadError,
)
from phydrax._publication import publish_bytes, publish_resource_set
from phydrax._resource_archive import admit_zip_resource, ArchiveLimits, read_zip_members
from phydrax._resource_set import (
    read_bounded_resource_set,
    ResourceSetLimits,
    ResourceSetReadError,
)
from phydrax.interchange import (
    decode_json_resource,
    decode_npy_resource,
    decode_npz_resource,
    decode_xml_resource,
    format_capabilities,
    HDF5Limits,
    inspect_hdf5_resource,
    NumpyFormatLimits,
)
from phydrax.qualification import (
    read_reference_artifact,
    ReferenceArtifactManifest,
)


def _resource_limits(maximum_bytes: int = 1_000_000) -> ResourceLimits:
    return ResourceLimits(maximum_bytes, 16, 1000, 1000, 100)


def _archive_limits(maximum_bytes: int = 1_000_000) -> ArchiveLimits:
    return ArchiveLimits(
        maximum_bytes,
        16,
        maximum_bytes,
        maximum_bytes,
        1024,
        8,
        100,
    )


def test_opened_resource_is_seekable_and_content_identified(tmp_path: Path):
    payload = b"seekable resource bytes"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)

    with open_bounded_resource(
        source.name,
        trusted_root=tmp_path,
        limits=_resource_limits(),
    ) as resource:
        assert resource.stream.read(8) == payload[:8]
        resource.stream.seek(0)
        assert resource.stream.read() == payload
        assert resource.manifest.size_bytes == len(payload)
        manifest_id = resource.manifest.manifest_id

    assert manifest_id


def test_resource_set_accounts_exact_members_and_rejects_links(tmp_path: Path):
    root = tmp_path / "root"
    dataset = root / "dataset"
    nested = dataset / "nested"
    nested.mkdir(parents=True)
    (dataset / "metadata.json").write_bytes(b"{}")
    (nested / "values.bin").write_bytes(b"1234")
    limits = ResourceSetLimits(1024, 512, 4, 4)

    admitted = read_bounded_resource_set(
        "dataset",
        trusted_root=root,
        limits=limits,
    )

    assert tuple(member.relative_path for member in admitted.manifest.members) == (
        "metadata.json",
        "nested/values.bin",
    )
    assert admitted.manifest.total_size_bytes == 6

    (dataset / "unsafe").symlink_to(tmp_path / "outside")
    with pytest.raises(ResourceSetReadError) as caught:
        read_bounded_resource_set("dataset", trusted_root=root, limits=limits)
    assert caught.value.reason == "policy"


def test_external_archive_preflights_paths_and_reads_exact_members():
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", b"{}")
        archive.writestr("arrays/value.npy", b"payload")
    resource = bounded_resource_from_bytes(
        output.getvalue(),
        limits=_resource_limits(),
    )

    admitted = admit_zip_resource(resource, limits=_archive_limits())
    values = read_zip_members(admitted, ("manifest.json", "arrays/value.npy"))

    assert values == {"manifest.json": b"{}", "arrays/value.npy": b"payload"}

    malicious = io.BytesIO()
    with zipfile.ZipFile(malicious, mode="w") as archive:
        archive.writestr("../escape", b"bad")
    with pytest.raises(ResourceReadError) as caught:
        admit_zip_resource(
            bounded_resource_from_bytes(
                malicious.getvalue(),
                limits=_resource_limits(),
            ),
            limits=_archive_limits(),
        )
    assert caught.value.reason == "policy"


def test_publication_exposes_only_complete_file_and_resource_set(tmp_path: Path):
    destination = tmp_path / "artifact.bin"
    first = publish_bytes(destination, b"first", maximum_bytes=16)
    second = publish_bytes(
        destination,
        b"second",
        maximum_bytes=16,
        mode="atomic_replace",
    )

    assert destination.read_bytes() == b"second"
    assert first.content_sha256 != second.content_sha256
    with pytest.raises(FileExistsError):
        publish_bytes(destination, b"third", maximum_bytes=16)

    bundle = tmp_path / "bundle"
    receipt = publish_resource_set(
        bundle,
        {"manifest.json": b"{}", "data/values.bin": b"123"},
        limits=ResourceSetLimits(1024, 512, 4, 4),
    )
    assert (bundle / "manifest.json").read_bytes() == b"{}"
    assert (bundle / "data" / "values.bin").read_bytes() == b"123"
    assert receipt.total_size_bytes == 5
    with pytest.raises(FileExistsError):
        publish_resource_set(
            bundle,
            {"manifest.json": b"{}"},
            limits=ResourceSetLimits(1024, 512, 4, 4),
        )
    replacement = publish_resource_set(
        bundle,
        {"manifest.json": b'{"generation": 2}'},
        limits=ResourceSetLimits(1024, 512, 4, 4),
        mode="atomic_replace",
    )
    assert replacement.replaced_existing is True
    assert (bundle / "manifest.json").read_bytes() == b'{"generation": 2}'
    assert not (bundle / "data").exists()


def test_document_decoders_reject_duplicate_json_and_xml_doctype():
    limits = _resource_limits()
    with pytest.raises(ResourceReadError):
        decode_json_resource(
            bounded_resource_from_bytes(b'{"value": 1, "value": 2}', limits=limits)
        )

    decoded = decode_json_resource(
        bounded_resource_from_bytes(
            json.dumps({"value": [1, 2, 3]}).encode(),
            limits=limits,
        )
    )
    assert decoded.value == {"value": [1, 2, 3]}
    assert decoded.resource.manifest.observed_nodes == 5

    with pytest.raises(ResourceReadError) as caught:
        decode_xml_resource(
            bounded_resource_from_bytes(
                b'<!DOCTYPE data [<!ENTITY x "boom">]><data>&x;</data>',
                limits=limits,
            )
        )
    assert caught.value.reason == "policy"


def test_numpy_decoders_preflight_pickle_and_preserve_read_only_arrays():
    limits = _resource_limits()
    encoded = io.BytesIO()
    np.save(encoded, np.arange(6, dtype=np.float64).reshape(2, 3), allow_pickle=False)
    decoded = decode_npy_resource(
        bounded_resource_from_bytes(encoded.getvalue(), limits=limits)
    )
    np.testing.assert_array_equal(decoded.value, np.arange(6).reshape(2, 3))
    assert not decoded.value.flags.writeable

    archive = io.BytesIO()
    np.savez(archive, values=np.arange(4, dtype=np.int32))
    decoded_archive = decode_npz_resource(
        bounded_resource_from_bytes(archive.getvalue(), limits=limits),
        expected_names=("values",),
    )
    np.testing.assert_array_equal(decoded_archive.arrays["values"], np.arange(4))

    objects = io.BytesIO()
    np.save(objects, np.asarray([{"unsafe": True}], dtype=object), allow_pickle=True)
    with pytest.raises(ResourceReadError) as caught:
        decode_npy_resource(
            bounded_resource_from_bytes(objects.getvalue(), limits=limits),
            limits=NumpyFormatLimits(max_container_bytes=1_000_000),
        )
    assert caught.value.reason == "policy"


def test_hdf5_inspection_and_reference_admission(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "values.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset(
            "values", data=np.arange(12, dtype=np.float64).reshape(3, 4)
        )
    payload = source.read_bytes()
    reference = ReferenceArtifactManifest(
        source.name,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"value": 1.0},
        uncertainty={"value": 0.0},
        lineage_ids=("unit-test",),
    )
    admitted = read_reference_artifact(source, reference)
    assert admitted.data == payload

    with open_bounded_resource(
        source.name,
        trusted_root=tmp_path,
        limits=_resource_limits(maximum_bytes=len(payload)),
    ) as resource:
        manifest = inspect_hdf5_resource(
            resource,
            limits=HDF5Limits(
                16,
                8,
                16,
                1024,
                4,
                16,
                1024,
                1024,
                1024,
                16,
            ),
        )
    assert tuple(dataset.path for dataset in manifest.datasets) == ("/values",)
    assert manifest.total_dataset_bytes == 96


def test_format_catalog_is_deterministic_and_non_dispatching():
    first = format_capabilities()
    second = format_capabilities()
    assert first == second
    keys = tuple((item.profile.format, item.profile.profile_id) for item in first)
    assert keys == tuple(sorted(keys))
    assert len(keys) == len(set(keys))
    assert {"vtu", "segy-rev1-ieee", "nifti", "onnx"}.issubset(
        {item.profile.format for item in first}
    )
