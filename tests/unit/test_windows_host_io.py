"""Real Windows handle, namespace, ACL, flush, and publication behavior.

Requires a local NTFS temporary directory and Windows 10 1709 or newer. Junction
creation uses FSCTL_SET_REPARSE_POINT, not privileged symbolic-link creation.
These tests exercise flush completion/error reporting, not simulated power loss.
Fault injection supplements real I/O to cover cleanup and native error reporting.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import stat
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from threading import Barrier

import pytest

from phydrax import _publication as publication, _windows_host_io as native
from phydrax._host_io import stat_identity
from phydrax._resource_set import ResourceSetLimits
from tests._support.filesystem import create_directory_link


pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Real Windows filesystem APIs"
)


@cache
def _test_apis() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True, winmode=0x00000800)
    security = ctypes.WinDLL("advapi32", use_last_error=True, winmode=0x00000800)
    kernel.GetCurrentProcess.argtypes = ()
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessHandleCount.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    )
    kernel.GetProcessHandleCount.restype = ctypes.c_int32
    security.GetSecurityDescriptorDacl.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int32),
    )
    security.GetSecurityDescriptorDacl.restype = ctypes.c_int32
    security.SetSecurityInfo.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    security.SetSecurityInfo.restype = ctypes.c_uint32
    return kernel, security


@pytest.fixture
def root(tmp_path: Path) -> Iterator[int]:
    chain = native.open_directory_chain(tmp_path, create=False)
    child = -1
    try:
        native.create_directory_at(chain[-1], "owned")
        child = native.open_directory_at(chain[-1], "owned")
        yield child
    finally:
        if child >= 0:
            os.close(child)
        for descriptor in reversed(chain):
            os.close(descriptor)


def _write_file(root: int, name: str, data: bytes) -> None:
    descriptor = native.open_file_at(root, name, write=True, create=True, exclusive=True)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _read_file(root: int, name: str) -> bytes:
    with os.fdopen(native.open_file_at(root, name), "rb") as stream:
        return stream.read()


def _handle_count() -> int:
    kernel, _ = _test_apis()
    count = ctypes.c_uint32()
    assert kernel.GetProcessHandleCount(kernel.GetCurrentProcess(), ctypes.byref(count))
    return count.value


@contextmanager
def _temporary_dacl(root: int, sddl: str | None) -> Iterator[None]:
    """Change a real held directory's DACL and restore it even after assertions."""
    kernel, _, security = native._apis()
    _, mutation = _test_apis()
    handle = native._reopen_directory(root, 0x00040000 | 0x00020000 | 0x80)
    original = ctypes.c_void_p()
    original_acl = ctypes.c_void_p()
    replacement = ctypes.c_void_p()
    changed = False
    try:
        code = security.GetSecurityInfo(
            handle,
            1,
            4,
            None,
            None,
            ctypes.byref(original_acl),
            None,
            ctypes.byref(original),
        )
        if code:
            raise ctypes.WinError(code)
        acl = ctypes.c_void_p()
        if sddl is not None:
            if not security.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl,
                1,
                ctypes.byref(replacement),
                None,
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            present, defaulted = ctypes.c_int32(), ctypes.c_int32()
            if not mutation.GetSecurityDescriptorDacl(
                replacement,
                ctypes.byref(present),
                ctypes.byref(acl),
                ctypes.byref(defaulted),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            assert present.value
        code = mutation.SetSecurityInfo(handle, 1, 4 | 0x80000000, None, None, acl, None)
        if code:
            raise ctypes.WinError(code)
        changed = True
        yield
    finally:
        try:
            if changed:
                code = mutation.SetSecurityInfo(
                    handle,
                    1,
                    4 | 0x80000000,
                    None,
                    None,
                    original_acl,
                    None,
                )
                if code:
                    raise ctypes.WinError(code)
        finally:
            if replacement.value is not None:
                kernel.LocalFree(replacement)
            if original.value is not None:
                kernel.LocalFree(original)
            native._close_handle(handle)


def test_binary_roundtrip_crt_descriptor_ownership_and_stat(root: int) -> None:
    payload = b"binary\x00\r\n\x1a\xff" * 1024
    descriptor = native.open_file_at(
        root,
        "data-\U00020000.bin",
        write=True,
        create=True,
        exclusive=True,
    )
    duplicate = os.dup(descriptor)
    try:
        assert not os.get_inheritable(descriptor)
        with os.fdopen(descriptor, "w+b") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            assert stat_identity(
                native.stat_at(root, "data-\U00020000.bin")
            ) == stat_identity(os.fstat(stream.fileno()))
            assert stat.S_ISREG(os.fstat(stream.fileno()).st_mode)
        # os.fdopen closed the original descriptor, not the owned duplicate.
        os.lseek(duplicate, 0, os.SEEK_SET)
        assert os.read(duplicate, len(payload) + 1) == payload
    finally:
        os.close(duplicate)
    assert _read_file(root, "data-\U00020000.bin") == payload
    native.sync_directory(root)


def test_native_exclusive_creation_and_atomic_rename_refusal(root: int) -> None:
    _write_file(root, "source", b"new")
    _write_file(root, "destination", b"old")
    with pytest.raises(FileExistsError):
        native.open_file_at(root, "source", write=True, create=True, exclusive=True)
    with pytest.raises(FileExistsError):
        native.rename_at(root, "source", "destination", replace=False)
    assert _read_file(root, "source") == b"new"
    assert _read_file(root, "destination") == b"old"
    native.rename_at(root, "source", "destination", replace=True)
    assert _read_file(root, "destination") == b"new"
    with pytest.raises(FileNotFoundError):
        native.open_file_at(root, "source")
    native.rename_at(root, "destination", "published", replace=False)
    native.sync_directory(root)
    assert native.list_directory(root) == ["published"]


def test_atomic_exclusive_rename_under_real_contention(root: int) -> None:
    for name in ("first", "second"):
        _write_file(root, name, name.encode())
    barrier = Barrier(2)

    def publish(name: str) -> tuple[str, bool]:
        barrier.wait()
        try:
            native.rename_at(root, name, "winner", replace=False)
        except FileExistsError:
            return name, False
        return name, True

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(pool.map(publish, ("first", "second")))
    winners = [name for name, succeeded in outcomes if succeeded]
    losers = [name for name, succeeded in outcomes if not succeeded]
    assert len(winners) == len(losers) == 1
    assert _read_file(root, "winner") == winners[0].encode()
    assert _read_file(root, losers[0]) == losers[0].encode()
    native.sync_directory(root)


def test_native_hardlink_and_unlink(root: int) -> None:
    _write_file(root, "source", b"linked")
    native.link_at(root, "source", "alias")
    source, alias = native.stat_at(root, "source"), native.stat_at(root, "alias")
    assert (source.st_dev, source.st_ino) == (alias.st_dev, alias.st_ino)
    assert alias.st_nlink == 2
    with pytest.raises(FileExistsError):
        native.link_at(root, "source", "alias")
    native.unlink_at(root, "source")
    assert _read_file(root, "alias") == b"linked"
    native.unlink_at(root, "alias")
    native.sync_directory(root)
    assert native.list_directory(root) == []
    with pytest.raises(FileNotFoundError):
        native.unlink_at(root, "alias")


def test_directory_creation_enumeration_and_removal(root: int) -> None:
    child = native.open_directory_at(root, "child", create=True)
    try:
        native.require_owned_directory(child)
        with pytest.raises(FileExistsError):
            native.create_directory_at(root, "child")
        _write_file(child, "member", b"inside")
        with pytest.raises(PermissionError):
            native.unlink_at(root, "child", directory=True)
    finally:
        os.close(child)
    with pytest.raises(OSError) as error:
        native.unlink_at(root, "child", directory=True)
    assert error.value.errno == errno.ENOTEMPTY
    native.rename_at(root, "child", "renamed", replace=False)
    child = native.open_directory_at(root, "renamed", create=False)
    try:
        native.unlink_at(child, "member")
        native.sync_directory(child)
    finally:
        os.close(child)
    native.unlink_at(root, "renamed", directory=True)
    native.sync_directory(root)
    assert native.list_directory(root) == []


def test_directory_descriptor_duplication_and_close(root: int) -> None:
    duplicate = os.dup(root)
    try:
        assert stat.S_ISDIR(os.fstat(duplicate).st_mode)
        assert not os.get_inheritable(duplicate)
        native.require_owned_directory(duplicate)
        assert native.list_directory(duplicate) == []
    finally:
        os.close(duplicate)
    assert native.list_directory(root) == []


def test_enumeration_has_independent_cursors(root: int) -> None:
    names = {f"entry-{index:04d}-{'x' * 100}" for index in range(600)}
    for name in names:
        _write_file(root, name, b"")
    with ThreadPoolExecutor(max_workers=4) as pool:
        enumerations = tuple(pool.map(native.list_directory, [root] * 8))
    assert all(
        set(entries) == names and len(entries) == len(names) for entries in enumerations
    )


def test_junction_lstat_and_reparse_traversal_refusal(root: int, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel").write_bytes(b"untouched")
    junction = tmp_path / "owned" / "junction"
    create_directory_link(junction, outside)
    try:
        information = native.stat_at(root, "junction")
        assert stat.S_ISLNK(information.st_mode)
        assert information.st_reparse_tag == 0xA0000003
        assert information.st_file_attributes & 0x400
        with pytest.raises(OSError) as error:
            native.open_directory_at(root, "junction")
        assert error.value.errno == errno.ELOOP
        with pytest.raises(OSError):
            native.open_file_at(root, "junction")
        with pytest.raises(OSError) as error:
            native.open_directory_chain(junction / "escaped", create=True)
        assert error.value.errno == errno.ELOOP
        with pytest.raises(OSError) as error:
            native.rename_at(root, "junction", "moved", replace=False)
        assert error.value.errno == errno.ELOOP
        assert (outside / "sentinel").read_bytes() == b"untouched"
        assert not (outside / "escaped").exists()
    finally:
        junction.rmdir()


def test_held_directory_cannot_be_replaced_by_a_junction(
    root: int, tmp_path: Path
) -> None:
    _write_file(root, "sentinel", b"original")
    held = tmp_path / "owned"
    with pytest.raises(PermissionError):
        held.rename(tmp_path / "moved")
    with pytest.raises(ValueError):
        native.rename_at(root, "sentinel", "missing\\escape", replace=False)
    assert _read_file(root, "sentinel") == b"original"


def test_owned_directory_admission_rejects_another_owner() -> None:
    # The Windows installation directory is system-owned. This only reads it.
    chain = native.open_directory_chain(Path(os.environ["SystemRoot"]), create=False)
    try:
        with pytest.raises(PermissionError, match="owned"):
            native.require_owned_directory(chain[-1])
    finally:
        for descriptor in reversed(chain):
            os.close(descriptor)


@pytest.mark.parametrize("untrusted", ["(A;;FW;;;WD)", "(A;OICIIO;FW;;;WD)"])
def test_owned_directory_admission_rejects_untrusted_writers(
    root: int, untrusted: str
) -> None:
    native.require_owned_directory(root)
    user = native._sid_text(native._current_user_sid())
    with _temporary_dacl(root, f"D:P(A;;FA;;;{user}){untrusted}"):
        with pytest.raises(PermissionError, match="another trustee"):
            native.require_owned_directory(root)
    native.require_owned_directory(root)


def test_owned_directory_admission_rejects_null_dacl(root: int) -> None:
    with _temporary_dacl(root, None):
        with pytest.raises(PermissionError, match="non-NULL DACL"):
            native.require_owned_directory(root)


def test_real_durability_access_failure_is_reported_without_handle_leak(
    root: int,
) -> None:
    user = native._sid_text(native._current_user_sid())
    with _temporary_dacl(root, f"D:P(D;;GW;;;{user})(A;;FA;;;{user})"):
        before = _handle_count()
        for _ in range(32):
            with pytest.raises(PermissionError):
                native.sync_directory(root)
        assert _handle_count() == before
    native.sync_directory(root)


def test_resource_lifetime_on_success_and_real_failures(
    root: int, tmp_path: Path
) -> None:
    _write_file(root, "existing", b"bytes")
    native.require_owned_directory(root)
    native.list_directory(root)
    before = _handle_count()
    for _ in range(32):
        descriptor = native.open_file_at(root, "existing")
        duplicate = os.dup(descriptor)
        os.close(descriptor)
        assert os.read(duplicate, 5) == b"bytes"
        os.close(duplicate)
        native.stat_at(root, "existing")
        native.list_directory(root)
        native.require_owned_directory(root)
        with pytest.raises(FileNotFoundError):
            native.open_directory_chain(tmp_path / "missing" / "child", create=False)
        with pytest.raises(FileExistsError):
            native.open_file_at(root, "existing", write=True, create=True, exclusive=True)
    assert _handle_count() == before


@pytest.mark.parametrize(
    "name", ["..", "a/b", "a\\b", "x:stream", "NUL", "x.", "x ", "x\x00"]
)
def test_invalid_components_cannot_escape_held_parent(root: int, name: str) -> None:
    with pytest.raises(ValueError):
        native.open_file_at(root, name, write=True, create=True, exclusive=True)
    assert native.list_directory(root) == []


@pytest.mark.parametrize(
    "path",
    [
        "C:..\\outside",
        "C:\\safe\\.. \\outside",
        "C:\\alias.\\outside",
        "\\\\server\\share",
    ],
)
def test_raw_chain_components_are_validated_before_normalization(path: str) -> None:
    with pytest.raises(ValueError):
        native.open_directory_chain(path, create=True)


def test_native_directory_reopen_preserves_identity_and_minimum_flush_rights(
    root: int,
) -> None:
    before = _handle_count()
    handle = native._reopen_directory(root, native._FILE_ADD_FILE)
    try:
        assert native._file_identity(handle) == native._file_identity(
            native._handle(root)
        )
        _, api, _ = native._apis()
        status = native._IoStatusBlock()
        native._check_status(
            api.NtFlushBuffersFileEx(handle, 0, None, 0, ctypes.byref(status))
        )
    finally:
        native._close_handle(handle)
    assert _handle_count() == before
    assert native.list_directory(root) == []


def test_reopen_identity_mismatch_closes_handle(
    root: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = native._file_identity
    calls = 0

    def changed_identity(handle: int) -> tuple[int, bytes]:
        nonlocal calls
        calls += 1
        volume, identifier = original(handle)
        return (volume, identifier) if calls == 1 else (volume + 1, identifier)

    before = _handle_count()
    monkeypatch.setattr(native, "_file_identity", changed_identity)
    with pytest.raises(OSError, match="identity does not match"):
        native._reopen_directory(root, native._DIRECTORY_ACCESS)
    assert _handle_count() == before


def test_flush_native_failure_survives_close_failure(
    root: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel, api, _ = native._apis()
    close = kernel.CloseHandle

    def failed_flush(
        handle: int,
        flags: int,
        parameters: object,
        size: int,
        status: object,
    ) -> int:
        assert flags == 0 and parameters is None and size == 0
        return ctypes.c_int32(0xC0000022).value  # STATUS_ACCESS_DENIED

    def failed_close(handle: int) -> int:
        assert close(handle)
        ctypes.set_last_error(6)
        return 0

    before = _handle_count()
    monkeypatch.setattr(api, "NtFlushBuffersFileEx", failed_flush)
    monkeypatch.setattr(kernel, "CloseHandle", failed_close)
    with pytest.raises(PermissionError) as raised:
        native.sync_directory(root)
    assert raised.value.winerror == 5
    assert any("Closing Windows handle" in note for note in raised.value.__notes__)
    assert _handle_count() == before


def test_publication_roundtrip_exclusive_replace_and_cleanup(tmp_path: Path) -> None:
    destination = tmp_path / "publication.bin"
    payload = b"windows-cuda-owned-publication"
    receipt = publication.publish_bytes(destination, payload, maximum_bytes=1024)
    assert destination.read_bytes() == payload
    assert receipt.size_bytes == len(payload)
    assert receipt.content_sha256 == hashlib.sha256(payload).hexdigest()
    assert not receipt.replaced_existing
    with pytest.raises(FileExistsError):
        publication.publish_bytes(destination, b"collision", maximum_bytes=1024)
    assert destination.read_bytes() == payload
    replacement = publication.publish_bytes(
        destination,
        b"replacement",
        maximum_bytes=1024,
        mode="atomic_replace",
    )
    assert replacement.replaced_existing
    assert destination.read_bytes() == b"replacement"
    assert {entry.name for entry in tmp_path.iterdir()} == {"publication.bin"}
    destination.unlink()


def test_publication_validation_error_preserves_error_and_removes_staging(
    tmp_path: Path,
) -> None:
    failure = ValueError("validator rejected staged bytes")

    def reject(path: os.PathLike[str]) -> None:
        raise failure

    before = _handle_count()
    with pytest.raises(ValueError) as raised:
        publication.publish_bytes(
            tmp_path / "rejected.bin",
            b"rejected",
            maximum_bytes=1024,
            validator=reject,
        )
    assert raised.value is failure
    assert list(tmp_path.iterdir()) == []
    assert _handle_count() == before


def test_publication_flush_error_survives_cleanup_error_and_releases_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "publication.bin"
    failure = OSError(errno.EIO, "directory flush failed")
    unlink = publication.unlink_at
    calls = 0

    def fail_flush(descriptor: int) -> None:
        raise failure

    def fail_cleanup(parent: int, name: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PermissionError(errno.EACCES, "staging cleanup denied")
        unlink(parent, name)

    before = _handle_count()
    monkeypatch.setattr(publication, "sync_directory", fail_flush)
    monkeypatch.setattr(publication, "unlink_at", fail_cleanup)
    with pytest.raises(OSError) as raised:
        publication.publish_bytes(
            destination, b"visible but not durable", maximum_bytes=1024
        )
    assert raised.value is failure
    assert any("staging cleanup denied" in note for note in failure.__notes__)
    assert destination.read_bytes() == b"visible but not durable"
    assert {entry.name for entry in tmp_path.iterdir()} == {"publication.bin"}
    assert _handle_count() == before
    destination.unlink()


def test_resource_set_publication_and_failed_staging_cleanup(tmp_path: Path) -> None:
    limits = ResourceSetLimits(
        max_total_bytes=1024,
        max_member_bytes=512,
        max_members=8,
        max_depth=4,
    )
    destination = tmp_path / "resources"
    receipt = publication.publish_resource_set(
        destination,
        {"nested/member": b"payload", "root": b"bytes"},
        limits=limits,
    )
    assert receipt.total_size_bytes == 12
    assert (destination / "nested" / "member").read_bytes() == b"payload"
    assert (destination / "root").read_bytes() == b"bytes"
    before = _handle_count()
    with pytest.raises(ValueError, match="byte limit"):
        publication.publish_resource_set(
            tmp_path / "rejected",
            {"oversize": b"x" * 513},
            limits=limits,
        )
    assert {entry.name for entry in tmp_path.iterdir()} == {"resources"}
    assert _handle_count() == before


def test_stat_preserves_native_timestamps_and_file_identity(root: int) -> None:
    _write_file(root, "metadata.bin", b"metadata")
    descriptor = native.open_file_at(root, "metadata.bin")
    try:
        expected = os.fstat(descriptor)
        actual = native.stat_at(root, "metadata.bin")
        assert tuple(actual) == tuple(expected)
        assert actual.st_atime_ns == expected.st_atime_ns
        assert actual.st_mtime_ns == expected.st_mtime_ns
        assert actual.st_ctime_ns == expected.st_ctime_ns
        assert actual.st_birthtime_ns == expected.st_birthtime_ns
        assert actual.st_file_attributes == expected.st_file_attributes
        assert actual.st_reparse_tag == expected.st_reparse_tag
    finally:
        os.close(descriptor)


def test_non_ntfs_directory_flush_is_not_reported_as_durable(
    root: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel, api, _ = native._apis()

    def other_filesystem(
        handle: int,
        volume: object,
        volume_size: int,
        serial: object,
        maximum_component: object,
        flags: object,
        filesystem: ctypes.Array[ctypes.c_wchar],
        filesystem_size: int,
    ) -> int:
        for index, character in enumerate("ReFS\0"):
            filesystem[index] = character
        return 1

    def unexpected_flush(*arguments: object) -> int:
        raise AssertionError("Unsupported namespace must not be flushed as if supported")

    before = _handle_count()
    monkeypatch.setattr(kernel, "GetVolumeInformationByHandleW", other_filesystem)
    monkeypatch.setattr(api, "NtFlushBuffersFileEx", unexpected_flush)
    with pytest.raises(OSError, match="requires NTFS") as raised:
        native.sync_directory(root)
    assert raised.value.errno == errno.ENOTSUP
    assert _handle_count() == before
