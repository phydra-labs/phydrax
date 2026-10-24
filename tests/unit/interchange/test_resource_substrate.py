#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import stat
import struct
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import phydrax._host_io as host_io
import phydrax._publication as publication
import phydrax._resource_archive as resource_archive
from phydrax._external_resource import (
    bounded_resource_from_bytes,
    open_bounded_resource,
    read_bounded_resource,
    ResourceLimits,
    ResourceReadError,
)
from phydrax._publication import publish_bytes, publish_resource_set
from phydrax._resource_archive import admit_zip_resource, ArchiveLimits, read_zip_members
from phydrax._resource_set import (
    open_bounded_resource_set,
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
from tests._support.filesystem import create_directory_link


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


def test_opened_resource_is_seekable_and_content_identified(tmp_path: Path) -> None:
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


def test_opened_resource_preserves_consumer_errors(tmp_path: Path) -> None:
    class ConsumerError(ValueError):
        pass

    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")

    with pytest.raises(ConsumerError, match="consumer failure"):
        with open_bounded_resource(
            source.name,
            trusted_root=tmp_path,
            limits=_resource_limits(),
        ):
            raise ConsumerError("consumer failure")


def test_trusted_root_walk_rejects_directory_link_ancestors(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    root = actual / "root"
    root.mkdir(parents=True)
    (root / "resource.bin").write_bytes(b"outside")
    alias = tmp_path / "alias"
    create_directory_link(alias, actual)

    with pytest.raises(ResourceReadError) as caught:
        read_bounded_resource(
            "resource.bin",
            trusted_root=alias / "root",
            limits=_resource_limits(),
        )

    assert caught.value.reason == "policy"
    assert (root / "resource.bin").read_bytes() == b"outside"


def test_trusted_root_walk_stays_on_held_ancestor_during_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    trusted = parent / "trusted"
    trusted.mkdir(parents=True)
    (trusted / "resource.bin").write_bytes(b"original")
    replacement_parent = tmp_path / "replacement-parent"
    replacement_trusted = replacement_parent / "trusted"
    replacement_trusted.mkdir(parents=True)
    (replacement_trusted / "resource.bin").write_bytes(b"replacement")
    detached = tmp_path / "detached"
    operation_owner = host_io
    if os.name == "nt":
        import phydrax._windows_host_io as windows_host_io

        operation_owner = windows_host_io
    original_open = operation_owner.open_directory_at
    attempted = False

    def replace_ancestor(parent_fd: int, name: str, *, create: bool = False) -> int:
        nonlocal attempted
        if name == "trusted" and not attempted:
            attempted = True
            if os.name == "nt":
                with pytest.raises(PermissionError):
                    parent.rename(detached)
            else:
                parent.rename(detached)
                replacement_parent.rename(parent)
        return original_open(parent_fd, name, create=create)

    monkeypatch.setattr(operation_owner, "open_directory_at", replace_ancestor)
    resource = read_bounded_resource(
        "resource.bin",
        trusted_root=trusted,
        limits=_resource_limits(),
    )

    assert attempted
    assert resource.data == b"original"


def test_resource_set_accounts_exact_members_and_rejects_links(tmp_path: Path) -> None:
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
    assert admitted.manifest.total_entry_count == 3

    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"outside resource set")
    create_directory_link(dataset / "unsafe", outside)
    with pytest.raises(ResourceSetReadError) as caught:
        read_bounded_resource_set("dataset", trusted_root=root, limits=limits)
    assert caught.value.reason == "policy"
    assert sentinel.read_bytes() == b"outside resource set"


def test_resource_set_counts_directories_against_total_entry_limit(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for index in range(4):
        (dataset / f"empty-{index}").mkdir()

    with pytest.raises(ResourceSetReadError) as caught:
        read_bounded_resource_set(
            dataset.name,
            trusted_root=tmp_path,
            limits=ResourceSetLimits(1024, 512, 3, 4),
        )

    assert caught.value.reason == "limit"


def test_opened_resource_set_reads_from_admitted_directory_generation(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "module.bin").write_bytes(b"admitted")

    with open_bounded_resource_set(
        dataset.name,
        trusted_root=tmp_path,
        limits=ResourceSetLimits(1024, 512, 1, 4),
    ) as opened:
        with pytest.raises(ResourceSetReadError) as caught:
            opened.read_member("module.bin", maximum_bytes=1)
        assert caught.value.reason == "limit"
        if os.name == "nt":
            with pytest.raises(PermissionError):
                dataset.rename(tmp_path / "admitted-generation")
        else:
            dataset.rename(tmp_path / "admitted-generation")
            dataset.mkdir()
            (dataset / "module.bin").write_bytes(b"replacement")
        payload = opened.read_member("module.bin")

    assert payload == b"admitted"


def test_host_file_descriptor_supports_binary_stream_and_identity(tmp_path: Path) -> None:
    parent = host_io.open_directory_descriptor(tmp_path, create=False)
    payload = b"binary\r\n\x1a\x00payload"
    try:
        descriptor = host_io.open_file_at(
            parent, "resource.bin", write=True, create=True, exclusive=True
        )
        with os.fdopen(descriptor, "w+b") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            status = os.fstat(stream.fileno())
            assert stat.S_ISREG(status.st_mode)
            assert host_io.stat_identity(host_io.stat_at(parent, "resource.bin")) == (
                host_io.stat_identity(status)
            )
            duplicate = os.dup(stream.fileno())
            with os.fdopen(duplicate, "rb") as copy:
                copy.seek(0)
                assert copy.read() == payload
        host_io.sync_directory(parent)
        assert host_io.list_directory(parent) == ["resource.bin"]
        with host_io.open_regular_file(
            host_io.descriptor_relative_path(parent, "resource.bin")
        ) as source:
            assert source.read() == payload
    finally:
        os.close(parent)


def test_host_exclusive_directory_rename_preserves_existing_target(
    tmp_path: Path,
) -> None:
    parent = host_io.open_directory_descriptor(tmp_path, create=False)
    try:
        host_io.create_directory_at(parent, "staged")
        host_io.create_directory_at(parent, "existing")
        with pytest.raises(FileExistsError):
            host_io.rename_at(parent, "staged", "existing", replace=False)
        assert set(host_io.list_directory(parent)) == {"staged", "existing"}
        host_io.rename_at(parent, "staged", "committed", replace=False)
        host_io.sync_directory(parent)
        assert set(host_io.list_directory(parent)) == {"committed", "existing"}
    finally:
        os.close(parent)


@pytest.mark.parametrize(
    "name",
    ["nested\\file", "file:stream", "name.", "name ", "NUL", "COM1.bin"],
)
def test_descriptor_relative_names_preserve_host_rules(name: str) -> None:
    if os.name == "nt":
        with pytest.raises(ValueError):
            host_io.descriptor_relative_path(0, name)
    else:
        assert host_io.descriptor_relative_path(0, name).name == name


@pytest.mark.parametrize("portable_spelling", [False, True], ids=["native", "portable"])
def test_resource_components_accepts_native_and_portable_absolute_paths(
    tmp_path: Path, portable_spelling: bool
) -> None:
    source = tmp_path / "nested" / "resource.bin"
    path = source.as_posix() if portable_spelling else str(source)
    root, components = host_io.resource_components(path, tmp_path, 4)
    assert root == str(tmp_path)
    assert components == ("nested", "resource.bin")


@pytest.mark.parametrize("portable_spelling", [False, True], ids=["native", "portable"])
def test_resource_components_refuses_traversal_before_normalization(
    tmp_path: Path, portable_spelling: bool
) -> None:
    separator = "/" if portable_spelling else os.sep
    path = separator.join(("nested", "..", "resource.bin"))
    with pytest.raises(ValueError, match="traversal"):
        host_io.resource_components(path, tmp_path, 4)


def test_resource_components_refuses_trusted_root_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="traversal"):
        host_io.resource_components("resource.bin", tmp_path / "nested" / "..", 4)


@pytest.mark.parametrize(
    "path",
    [
        "https://example.invalid/resource",
        "//server/share/file",
        "\\\\server\\share\\file",
    ],
)
def test_resource_components_refuses_network_paths(tmp_path: Path, path: str) -> None:
    with pytest.raises(ValueError, match="Network"):
        host_io.resource_components(path, tmp_path, 4)


def test_opened_resource_set_reads_nested_manifest_names_on_each_host(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "dataset" / "nested"
    nested.mkdir(parents=True)
    (nested / "resource.bin").write_bytes(b"nested payload")
    with open_bounded_resource_set(
        "dataset",
        trusted_root=tmp_path,
        limits=ResourceSetLimits(1024, 512, 2, 4),
    ) as opened:
        assert tuple(member.relative_path for member in opened.manifest.members) == (
            "nested/resource.bin",
        )
        assert opened.read_member("nested/resource.bin") == b"nested payload"


def test_resource_substrate_scenario_1() -> None:
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
    assert admitted.directory_entry_count == 2

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
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w") as archive:
        archive.mkdir("one/")
        archive.mkdir("two/")
        archive.writestr("payload.bin", b"payload")
    limits = ArchiveLimits(
        1_000_000,
        2,
        1_000_000,
        1_000_000,
        1024,
        8,
        100,
    )

    with pytest.raises(ResourceReadError) as caught:
        admit_zip_resource(
            bounded_resource_from_bytes(
                output.getvalue(),
                limits=_resource_limits(),
            ),
            limits=limits,
        )

    assert caught.value.reason == "limit"
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


def test_external_archive_counts_headers_before_zipinfo_materialization(
    monkeypatch: Any,
) -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w") as archive:
        archive.writestr("one.bin", b"1")
        archive.writestr("two.bin", b"2")
        archive.writestr("three.bin", b"3")
    payload = bytearray(output.getvalue())
    end_offset = payload.rfind(b"PK\x05\x06")
    struct.pack_into("<HH", payload, end_offset + 8, 1, 1)

    def unexpected_zipfile(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("ZipFile materialized entries before central-directory preflight")

    monkeypatch.setattr(resource_archive.zipfile, "ZipFile", unexpected_zipfile)
    with pytest.raises(ResourceReadError) as caught:
        admit_zip_resource(
            bounded_resource_from_bytes(
                bytes(payload),
                limits=_resource_limits(),
            ),
            limits=_archive_limits(),
        )

    assert caught.value.reason == "malformed"


def test_resource_publication_preflights_parent_directories_as_entries(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "bundle"

    with pytest.raises(ValueError, match="total entry limit"):
        publish_resource_set(
            destination,
            {"nested/value.bin": b"value"},
            limits=ResourceSetLimits(1024, 512, 1, 4),
        )

    assert not destination.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX rename compatibility contract.")
@pytest.mark.parametrize(
    ("mode", "existing_empty"),
    [
        pytest.param("exclusive", False, id="exclusive-new"),
        pytest.param("exclusive", True, id="exclusive-empty"),
        pytest.param("atomic_replace", False, id="atomic-replace-new"),
    ],
)
def test_posix_resource_publication_without_rename_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: publication.PublicationMode,
    existing_empty: bool,
) -> None:
    destination = tmp_path / "bundle"
    if existing_empty:
        destination.mkdir()

    def reject_rename_library(*args: Any, **kwargs: Any) -> None:
        raise OSError(errno.EINVAL, "Filesystem rejects rename flags")

    monkeypatch.setattr(host_io.ctypes, "CDLL", reject_rename_library)
    limits = ResourceSetLimits(1024, 512, 4, 4)
    receipt = publish_resource_set(
        destination,
        {"manifest.json": b"{}", "data/values.bin": b"123"},
        limits=limits,
        mode=mode,
    )

    assert (destination / "manifest.json").read_bytes() == b"{}"
    assert (destination / "data" / "values.bin").read_bytes() == b"123"
    assert receipt.total_size_bytes == 5
    assert receipt.replaced_existing is existing_empty
    assert {entry.name for entry in tmp_path.iterdir()} == {"bundle"}


@pytest.mark.skipif(os.name != "posix", reason="POSIX rename compatibility contract.")
@pytest.mark.parametrize("concurrent_nonempty", [False, True], ids=["empty", "nonempty"])
def test_posix_resource_publication_preserves_concurrent_directory_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    concurrent_nonempty: bool,
) -> None:
    destination = tmp_path / "bundle"
    open_directory = publication.open_directory_at

    def appear_after_absence(parent: int, name: str, *, create: bool = False) -> int:
        try:
            return open_directory(parent, name, create=create)
        except FileNotFoundError:
            if name == destination.name:
                destination.mkdir()
                if concurrent_nonempty:
                    (destination / "existing.bin").write_bytes(b"existing generation")
            raise

    monkeypatch.setattr(publication, "open_directory_at", appear_after_absence)
    limits = ResourceSetLimits(1024, 512, 4, 4)
    if concurrent_nonempty:
        with pytest.raises(FileExistsError) as raised:
            publish_resource_set(destination, {"value.bin": b"new"}, limits=limits)
        assert str(raised.value) == str(destination)
        assert (destination / "existing.bin").read_bytes() == b"existing generation"
        assert not (destination / "value.bin").exists()
    else:
        receipt = publish_resource_set(
            destination, {"value.bin": b"new"}, limits=limits
        )
        assert not receipt.replaced_existing
        assert (destination / "value.bin").read_bytes() == b"new"
    assert {entry.name for entry in tmp_path.iterdir()} == {"bundle"}


def test_resource_publication_empty_destination_flush_failure_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "bundle"
    destination.mkdir()
    failure = OSError(errno.EIO, "Empty destination removal synchronization failed")
    sync = publication.sync_directory

    def fail_after_empty_destination_removal(descriptor: int) -> None:
        if not destination.exists():
            raise failure
        sync(descriptor)

    monkeypatch.setattr(
        publication, "sync_directory", fail_after_empty_destination_removal
    )
    with pytest.raises(OSError) as raised:
        publish_resource_set(
            destination,
            {"value.bin": b"new"},
            limits=ResourceSetLimits(1024, 512, 4, 4),
        )

    assert raised.value is failure
    assert list(tmp_path.iterdir()) == []


def test_publication_exposes_only_complete_file_and_resource_set(tmp_path: Path) -> None:
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
    if os.name == "nt":
        with pytest.raises(OSError, match="Atomic directory exchange is unsupported"):
            publish_resource_set(
                bundle,
                {"manifest.json": b'{"generation": 2}'},
                limits=ResourceSetLimits(1024, 512, 4, 4),
                mode="atomic_replace",
            )
        assert (bundle / "manifest.json").read_bytes() == b"{}"
        assert (bundle / "data" / "values.bin").read_bytes() == b"123"
        assert {path.name for path in tmp_path.iterdir()} == {"artifact.bin", "bundle"}
        return
    replacement = publish_resource_set(
        bundle,
        {"manifest.json": b'{"generation": 2}'},
        limits=ResourceSetLimits(1024, 512, 4, 4),
        mode="atomic_replace",
    )
    assert replacement.replaced_existing is True
    assert (bundle / "manifest.json").read_bytes() == b'{"generation": 2}'
    assert not (bundle / "data").exists()


def test_hdf5_inspection_and_reference_admission(tmp_path: Path) -> None:
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


def test_format_catalog_is_deterministic_and_non_dispatching() -> None:
    first = format_capabilities()
    second = format_capabilities()
    assert first == second
    keys = tuple((item.profile.format, item.profile.profile_id) for item in first)
    assert keys == tuple(sorted(keys))
    assert len(keys) == len(set(keys))
    assert {"vtu", "segy-rev1-ieee", "nifti", "onnx"}.issubset(
        {item.profile.format for item in first}
    )


@pytest.mark.parametrize("fortran", [False, True], ids=["C", "F"])
def test_npy_decoding_owns_a_readonly_immutable_resource_view(fortran: bool) -> None:
    original = np.arange(12, dtype=np.float64).reshape(3, 4)
    value = np.asfortranarray(original) if fortran else original
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    resource = bounded_resource_from_bytes(buffer.getvalue(), limits=_resource_limits())
    decoded = decode_npy_resource(resource)
    np.testing.assert_array_equal(decoded.value, original)
    assert not decoded.value.flags.writeable
    assert not decoded.value.flags.owndata
    with pytest.raises(ValueError):
        decoded.value.setflags(write=True)


def test_npz_decoding_returns_exact_readonly_inventory() -> None:
    arrays = {f"member{i:02d}": np.arange(128, dtype=np.float64) + i for i in range(16)}
    buffer = io.BytesIO()
    np.savez(buffer, allow_pickle=False, **arrays)
    resource = bounded_resource_from_bytes(buffer.getvalue(), limits=_resource_limits())
    decoded = decode_npz_resource(resource, expected_names=tuple(arrays))
    assert set(decoded.arrays) == set(arrays)
    for name, value in decoded.arrays.items():
        np.testing.assert_array_equal(value, arrays[name])
        assert not value.flags.writeable
    with pytest.raises(ResourceReadError, match="inventory"):
        decode_npz_resource(resource, expected_names=("wrong",))


def test_npz_headers_are_all_admitted_before_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax._numpy_resource as numpy_resource

    buffer = io.BytesIO()
    np.savez(
        buffer,
        first=np.arange(8, dtype=np.float64),
        second=np.arange(8, dtype=np.float64),
    )
    resource = bounded_resource_from_bytes(buffer.getvalue(), limits=_resource_limits())

    def refuse_decode(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Array decoding reached before aggregate admission")

    monkeypatch.setattr(numpy_resource.np, "load", refuse_decode)
    with pytest.raises(ResourceReadError, match="aggregate element"):
        decode_npz_resource(resource, limits=NumpyFormatLimits(max_total_elements=12))


def test_npz_corrupt_member_is_refused_before_any_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax._numpy_resource as numpy_resource

    buffer = io.BytesIO()
    np.savez(
        buffer,
        first=np.arange(8, dtype=np.float64),
        second=np.arange(8, dtype=np.float64),
    )
    payload = bytearray(buffer.getvalue())
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = archive.getinfo("second.npy")
        name_size, extra_size = struct.unpack_from(
            "<HH", payload, member.header_offset + 26
        )
        payload[member.header_offset + 30 + name_size + extra_size + 128] ^= 1

    def refuse_decode(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Array decoding reached before archive checksum admission")

    monkeypatch.setattr(numpy_resource.np, "load", refuse_decode)
    resource = bounded_resource_from_bytes(bytes(payload), limits=_resource_limits())
    with pytest.raises(ResourceReadError):
        decode_npz_resource(resource)


def test_npy_view_preserves_admitted_zero_itemsize_structured_dtype() -> None:
    value = np.empty((3,), dtype=[("empty", "V0")])
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    resource = bounded_resource_from_bytes(buffer.getvalue(), limits=_resource_limits())
    decoded = decode_npy_resource(
        resource,
        limits=NumpyFormatLimits(
            allow_structured_dtypes=True, allowed_dtype_kinds=frozenset({"V"})
        ),
    )
    assert decoded.value.shape == value.shape
    assert decoded.value.dtype == value.dtype
    assert not decoded.value.flags.writeable
