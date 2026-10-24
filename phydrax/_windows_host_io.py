#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Held-handle Windows filesystem operations; no pathname-based emulation.

Public integers are owned, noninheritable CRT descriptors, not Win32 HANDLEs.
Callers close them with ``os.close``; regular files also support ``os.fdopen``,
``os.read``, and ``os.fsync``. Native handles never escape this module.

Requires Windows 10 1709 or later and a local filesystem implementing the
requested NT operations. Drive letters must identify volume devices directly;
UNC, device paths, SUBST drives, all filesystem reparse points (including cloud
placeholders), and alternate data streams are refused. Open directory handles
exclude delete sharing: an admitted directory cannot be moved or replaced while
held. Close a publication's staging-directory descriptor before renaming it.
File handles permit delete sharing, but namespace-operation source handles do
not, so the source cannot be substituted between open and mutation.

New objects receive a protected owner/SYSTEM/Administrators DACL. Admission of
existing directories requires the effective user's ownership and rejects write
permissions for other trustees, NULL DACLs, and unfamiliar ACE forms. SYSTEM and
local Administrators are trusted like POSIX root; this is not a sandbox against
administrators, the same user, or existing hostile handles. ACL admission is a
snapshot, just as POSIX ownership/mode admission is.

``sync_directory`` reopens the *held object* with an empty native relative name,
verifies its volume/file identity, and issues NtFlushBuffersFileEx with normal
flags (data, metadata, and storage-cache flush). It requires NTFS: MS-FSA's
directory-namespace persistence behavior is NTFS-specific. Denied access and
failed flushes raise; there is no successful no-op or pathname fallback.
Completion is not power-loss qualification of this local API sequence, and
depends on the filesystem, filters, and storage honoring flush requests.
Namespace mutations do not themselves imply durability, except newly created
components of ``open_directory_at(create=True)``/``open_directory_chain``.

Microsoft API contracts used here:
https://learn.microsoft.com/windows/win32/api/winternl/nf-winternl-ntcreatefile
https://learn.microsoft.com/windows-hardware/drivers/ddi/ntifs/nf-ntifs-ntflushbuffersfileex
https://learn.microsoft.com/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_file_rename_information
https://learn.microsoft.com/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_file_link_information
https://learn.microsoft.com/windows-hardware/drivers/ddi/ntddk/ns-ntddk-_file_disposition_information_ex
https://learn.microsoft.com/windows/win32/api/aclapi/nf-aclapi-getsecurityinfo
https://learn.microsoft.com/openspecs/windows_protocols/ms-fsa/0de7dc40-9627-437e-a4df-c4696cdc3d02
"""

from __future__ import annotations

import ctypes
import errno
import ntpath
import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import NoReturn


_HANDLE = ctypes.c_void_p
_ULONG = ctypes.c_uint32
_NTSTATUS = ctypes.c_int32
_BOOLEAN = ctypes.c_ubyte
_READ_CONTROL = 0x00020000
_SYNCHRONIZE = 0x00100000
_DELETE = 0x00010000
_FILE_READ_ATTRIBUTES = 0x80
_FILE_LIST_DIRECTORY = 0x01
_FILE_TRAVERSE = 0x20
_FILE_ADD_FILE = 0x02  # Directory equivalent of FILE_WRITE_DATA, required to flush.
_DIRECTORY_ACCESS = (
    _READ_CONTROL
    | _SYNCHRONIZE
    | _FILE_READ_ATTRIBUTES
    | _FILE_LIST_DIRECTORY
    | _FILE_TRAVERSE
)
_FILE_DIRECTORY_FILE = 0x01
_FILE_NON_DIRECTORY_FILE = 0x40
_FILE_SYNCHRONOUS_IO_NONALERT = 0x20
_FILE_OPEN_REPARSE_POINT = 0x00200000
_FILE_ATTRIBUTE_DIRECTORY = 0x10
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_FILE_OPEN = 1
_FILE_CREATE = 2
_FILE_OPEN_IF = 3
_FILE_CREATED = 2
_SHARE_READ_WRITE = 3
_SHARE_ALL = 7
_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_MUTATING_RIGHTS = 0x10000000 | 0x40000000 | 0x000D0156


class _UnicodeString(ctypes.Structure):
    _fields_ = (
        ("Length", ctypes.c_uint16),
        ("MaximumLength", ctypes.c_uint16),
        ("Buffer", ctypes.c_void_p),
    )


class _ObjectAttributes(ctypes.Structure):
    _fields_ = (
        ("Length", _ULONG),
        ("RootDirectory", _HANDLE),
        ("ObjectName", ctypes.POINTER(_UnicodeString)),
        ("Attributes", _ULONG),
        ("SecurityDescriptor", ctypes.c_void_p),
        ("SecurityQualityOfService", ctypes.c_void_p),
    )


class _IoStatusUnion(ctypes.Union):
    _fields_ = (("Status", _NTSTATUS), ("Pointer", ctypes.c_void_p))


class _IoStatusBlock(ctypes.Structure):
    _anonymous_ = ("Result",)
    _fields_ = (("Result", _IoStatusUnion), ("Information", ctypes.c_size_t))


class _AttributeTagInformation(ctypes.Structure):
    _fields_ = (("FileAttributes", _ULONG), ("ReparseTag", _ULONG))


class _FileIdInformation(ctypes.Structure):
    _fields_ = (
        ("VolumeSerialNumber", ctypes.c_uint64),
        ("FileId", ctypes.c_ubyte * 16),
    )


class _NameInformation(ctypes.Structure):
    # The union's ULONG member also gives the correct layout for BOOLEAN-based
    # FileLinkInformation, whose remaining union bytes are zero.
    _fields_ = (
        ("Flags", _ULONG),
        ("RootDirectory", _HANDLE),
        ("FileNameLength", _ULONG),
        ("FileName", ctypes.c_uint16 * 1),
    )


class _FileNamesInformation(ctypes.Structure):
    _fields_ = (
        ("NextEntryOffset", _ULONG),
        ("FileIndex", _ULONG),
        ("FileNameLength", _ULONG),
        ("FileName", ctypes.c_uint16 * 1),
    )


class _TokenUser(ctypes.Structure):
    _fields_ = (("Sid", ctypes.c_void_p), ("Attributes", _ULONG))


class _AclSizeInformation(ctypes.Structure):
    _fields_ = (
        ("AceCount", _ULONG),
        ("AclBytesInUse", _ULONG),
        ("AclBytesFree", _ULONG),
    )


class _AceHeader(ctypes.Structure):
    _fields_ = (
        ("AceType", _BOOLEAN),
        ("AceFlags", _BOOLEAN),
        ("AceSize", ctypes.c_uint16),
    )


class _AccessAllowedAce(ctypes.Structure):
    _fields_ = (("Header", _AceHeader), ("Mask", _ULONG), ("SidStart", _ULONG))


@cache
def _apis() -> tuple[ctypes.CDLL, ctypes.CDLL, ctypes.CDLL]:
    if sys.platform != "win32":
        raise OSError(errno.ENOSYS, "The Windows filesystem backend requires Windows.")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True, winmode=0x00000800)
    native = ctypes.WinDLL("ntdll", use_last_error=True, winmode=0x00000800)
    security = ctypes.WinDLL("advapi32", use_last_error=True, winmode=0x00000800)
    kernel.CloseHandle.argtypes = (_HANDLE,)
    kernel.CloseHandle.restype = ctypes.c_int32
    kernel.GetCurrentProcess.argtypes = ()
    kernel.GetCurrentProcess.restype = _HANDLE
    kernel.GetCurrentThread.argtypes = ()
    kernel.GetCurrentThread.restype = _HANDLE
    kernel.LocalFree.argtypes = (ctypes.c_void_p,)
    kernel.LocalFree.restype = ctypes.c_void_p
    kernel.GetFileType.argtypes = (_HANDLE,)
    kernel.GetFileType.restype = _ULONG
    kernel.GetFileInformationByHandleEx.argtypes = (
        _HANDLE,
        ctypes.c_int32,
        ctypes.c_void_p,
        _ULONG,
    )
    kernel.GetFileInformationByHandleEx.restype = ctypes.c_int32
    kernel.GetVolumeInformationByHandleW.argtypes = (
        _HANDLE,
        ctypes.POINTER(ctypes.c_wchar),
        _ULONG,
        ctypes.POINTER(_ULONG),
        ctypes.POINTER(_ULONG),
        ctypes.POINTER(_ULONG),
        ctypes.POINTER(ctypes.c_wchar),
        _ULONG,
    )
    kernel.GetVolumeInformationByHandleW.restype = ctypes.c_int32
    kernel.QueryDosDeviceW.argtypes = (
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_wchar),
        _ULONG,
    )
    kernel.QueryDosDeviceW.restype = _ULONG
    native.NtCreateFile.argtypes = (
        ctypes.POINTER(_HANDLE),
        _ULONG,
        ctypes.POINTER(_ObjectAttributes),
        ctypes.POINTER(_IoStatusBlock),
        ctypes.POINTER(ctypes.c_int64),
        _ULONG,
        _ULONG,
        _ULONG,
        _ULONG,
        ctypes.c_void_p,
        _ULONG,
    )
    native.NtCreateFile.restype = _NTSTATUS
    native.NtSetInformationFile.argtypes = (
        _HANDLE,
        ctypes.POINTER(_IoStatusBlock),
        ctypes.c_void_p,
        _ULONG,
        ctypes.c_int32,
    )
    native.NtSetInformationFile.restype = _NTSTATUS
    native.NtQueryDirectoryFile.argtypes = (
        _HANDLE,
        _HANDLE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(_IoStatusBlock),
        ctypes.c_void_p,
        _ULONG,
        ctypes.c_int32,
        _BOOLEAN,
        ctypes.POINTER(_UnicodeString),
        _BOOLEAN,
    )
    native.NtQueryDirectoryFile.restype = _NTSTATUS
    native.NtFlushBuffersFileEx.argtypes = (
        _HANDLE,
        _ULONG,
        ctypes.c_void_p,
        _ULONG,
        ctypes.POINTER(_IoStatusBlock),
    )
    native.NtFlushBuffersFileEx.restype = _NTSTATUS
    native.RtlNtStatusToDosError.argtypes = (_NTSTATUS,)
    native.RtlNtStatusToDosError.restype = _ULONG
    security.OpenProcessToken.argtypes = (_HANDLE, _ULONG, ctypes.POINTER(_HANDLE))
    security.OpenProcessToken.restype = ctypes.c_int32
    security.OpenThreadToken.argtypes = (
        _HANDLE,
        _ULONG,
        ctypes.c_int32,
        ctypes.POINTER(_HANDLE),
    )
    security.OpenThreadToken.restype = ctypes.c_int32
    security.GetTokenInformation.argtypes = (
        _HANDLE,
        ctypes.c_int32,
        ctypes.c_void_p,
        _ULONG,
        ctypes.POINTER(_ULONG),
    )
    security.GetTokenInformation.restype = ctypes.c_int32
    security.GetLengthSid.argtypes = (ctypes.c_void_p,)
    security.GetLengthSid.restype = _ULONG
    security.IsValidSid.argtypes = (ctypes.c_void_p,)
    security.IsValidSid.restype = ctypes.c_int32
    security.EqualSid.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    security.EqualSid.restype = ctypes.c_int32
    security.IsWellKnownSid.argtypes = (ctypes.c_void_p, ctypes.c_int32)
    security.IsWellKnownSid.restype = ctypes.c_int32
    security.ConvertSidToStringSidW.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    security.ConvertSidToStringSidW.restype = ctypes.c_int32
    security.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = (
        ctypes.c_wchar_p,
        _ULONG,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(_ULONG),
    )
    security.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = ctypes.c_int32
    security.GetSecurityInfo.argtypes = (
        _HANDLE,
        ctypes.c_int32,
        _ULONG,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    )
    security.GetSecurityInfo.restype = _ULONG
    security.IsValidAcl.argtypes = (ctypes.c_void_p,)
    security.IsValidAcl.restype = ctypes.c_int32
    security.GetAclInformation.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        _ULONG,
        ctypes.c_int32,
    )
    security.GetAclInformation.restype = ctypes.c_int32
    security.GetAce.argtypes = (
        ctypes.c_void_p,
        _ULONG,
        ctypes.POINTER(ctypes.c_void_p),
    )
    security.GetAce.restype = ctypes.c_int32
    return kernel, native, security


def _raise_windows(code: int, name: str | None = None) -> NoReturn:
    # Windows OSError's winerror constructor performs Python's standard errno
    # and subclass mapping (including sharing violations -> PermissionError).
    raise OSError(0, ctypes.FormatError(code), name, code)


def _check_status(status: int, name: str | None = None) -> None:
    if status == 0:
        return
    # Do not treat STATUS_PENDING or informational statuses as completed I/O.
    unsigned = status & 0xFFFFFFFF
    if unsigned == 0xC00000BA:
        raise IsADirectoryError(errno.EISDIR, "Expected a regular file.", name)
    if unsigned == 0xC0000103:
        raise NotADirectoryError(errno.ENOTDIR, "Expected a directory.", name)
    _, native, _ = _apis()
    code = native.RtlNtStatusToDosError(status)
    _raise_windows(code, name)


def _close_handle(handle: int) -> None:
    kernel, _, _ = _apis()
    if not kernel.CloseHandle(handle):
        original = sys.exception()
        try:
            _raise_windows(ctypes.get_last_error())
        except OSError as cleanup_error:
            if original is None:
                raise
            original.add_note(f"Closing Windows handle also failed: {cleanup_error!r}")


def _handle(descriptor: int) -> int:
    import msvcrt

    return msvcrt.get_osfhandle(descriptor)


def _adopt_handle(handle: int, *, write: bool = False) -> int:
    import msvcrt

    try:
        return msvcrt.open_osfhandle(
            handle, os.O_BINARY | os.O_NOINHERIT | (os.O_RDWR if write else os.O_RDONLY)
        )
    except BaseException:
        _close_handle(handle)
        raise


def _attributes(handle: int) -> _AttributeTagInformation:
    kernel, _, _ = _apis()
    information = _AttributeTagInformation()
    if not kernel.GetFileInformationByHandleEx(
        handle, 9, ctypes.byref(information), ctypes.sizeof(information)
    ):
        _raise_windows(ctypes.get_last_error())
    return information


def _require_not_reparse(information: _AttributeTagInformation) -> None:
    if information.FileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT:
        raise OSError(errno.ELOOP, "Windows reparse points are not admitted.")


def _directory_handle(descriptor: int) -> int:
    handle = _handle(descriptor)
    information = _attributes(handle)
    _require_not_reparse(information)
    if not information.FileAttributes & _FILE_ATTRIBUTE_DIRECTORY:
        raise NotADirectoryError(errno.ENOTDIR, "Expected a held directory.")
    return handle


def _validate_name(name: str) -> None:
    from ._host_io import _validate_component

    _validate_component(name)


def _current_user_sid() -> bytes:
    kernel, _, security = _apis()
    token = _HANDLE()
    if not security.OpenThreadToken(
        kernel.GetCurrentThread(), 0x0008, True, ctypes.byref(token)
    ):
        code = ctypes.get_last_error()
        if code != 1008:  # ERROR_NO_TOKEN: no thread impersonation.
            _raise_windows(code)
        if not security.OpenProcessToken(
            kernel.GetCurrentProcess(), 0x0008, ctypes.byref(token)
        ):
            _raise_windows(ctypes.get_last_error())
    try:
        size = _ULONG()
        if security.GetTokenInformation(token, 1, None, 0, ctypes.byref(size)):
            raise OSError(errno.EIO, "TokenUser unexpectedly returned no buffer.")
        code = ctypes.get_last_error()
        if code != 122:  # ERROR_INSUFFICIENT_BUFFER
            _raise_windows(code)
        buffer = ctypes.create_string_buffer(size.value)
        if not security.GetTokenInformation(token, 1, buffer, size, ctypes.byref(size)):
            _raise_windows(ctypes.get_last_error())
        sid = _TokenUser.from_buffer(buffer).Sid
        if not security.IsValidSid(sid):
            raise OSError(errno.EIO, "TokenUser returned an invalid SID.")
        return ctypes.string_at(sid, security.GetLengthSid(sid))
    finally:
        if token.value is not None:
            _close_handle(token.value)


def _sid_text(sid: bytes) -> str:
    kernel, _, security = _apis()
    buffer = ctypes.create_string_buffer(sid)
    text = ctypes.c_void_p()
    if not security.ConvertSidToStringSidW(buffer, ctypes.byref(text)):
        _raise_windows(ctypes.get_last_error())
    try:
        return ctypes.wstring_at(text)
    finally:
        kernel.LocalFree(text)


@contextmanager
def _creation_security(enabled: bool) -> Iterator[ctypes.c_void_p]:
    descriptor = ctypes.c_void_p()
    kernel, _, security = _apis()
    if enabled:
        owner = _sid_text(_current_user_sid())
        sddl = f"O:{owner}D:P(A;OICI;FA;;;{owner})(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)"
        if not security.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl, 1, ctypes.byref(descriptor), None
        ):
            _raise_windows(ctypes.get_last_error())
    try:
        yield descriptor
    finally:
        if descriptor.value is not None:
            kernel.LocalFree(descriptor)


def _open_native(
    parent: int | None,
    name: str,
    *,
    access: int,
    disposition: int = _FILE_OPEN,
    options: int = 0,
    share: int = _SHARE_READ_WRITE,
) -> tuple[int, bool]:
    if parent is not None:
        _validate_name(name)
    encoded = name.encode("utf-16-le", "surrogatepass")
    if not encoded or len(encoded) > 65532:
        raise ValueError("A native filename must fit a counted UTF-16 string.")
    buffer = ctypes.create_string_buffer(encoded + b"\0\0")
    unicode_name = _UnicodeString(
        len(encoded), len(encoded) + 2, ctypes.addressof(buffer)
    )
    handle = _HANDLE()
    status_block = _IoStatusBlock()
    _, native, _ = _apis()
    try:
        with _creation_security(disposition != _FILE_OPEN) as security:
            attributes = _ObjectAttributes(
                ctypes.sizeof(_ObjectAttributes),
                parent,
                ctypes.pointer(unicode_name),
                0x40,
                security,
                None,  # OBJ_CASE_INSENSITIVE; never OBJ_INHERIT.
            )
            result = native.NtCreateFile(
                ctypes.byref(handle),
                access | _SYNCHRONIZE,
                ctypes.byref(attributes),
                ctypes.byref(status_block),
                None,
                0x80,
                share,
                disposition,
                options | _FILE_SYNCHRONOUS_IO_NONALERT | _FILE_OPEN_REPARSE_POINT,
                None,
                0,
            )
        _check_status(result, name)
        if handle.value is None:
            raise OSError(errno.EIO, "NtCreateFile succeeded without a handle.", name)
        return handle.value, status_block.Information == _FILE_CREATED
    except BaseException:
        if handle.value is not None:
            _close_handle(handle.value)
        raise


def _open_directory(parent: int | None, name: str, disposition: int) -> tuple[int, bool]:
    handle, created = _open_native(
        parent,
        name,
        access=_DIRECTORY_ACCESS,
        disposition=disposition,
        options=_FILE_DIRECTORY_FILE,
    )
    try:
        _require_not_reparse(_attributes(handle))
    except BaseException:
        _close_handle(handle)
        raise
    return _adopt_handle(handle), created


def open_directory_at(parent: int, name: str, *, create: bool = False) -> int:
    """Open one pinned child directory, durably publishing it if newly created."""
    descriptor, created = _open_directory(
        _directory_handle(parent),
        name,
        _FILE_OPEN_IF if create else _FILE_OPEN,
    )
    try:
        if created:
            sync_directory(parent)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def create_directory_at(parent: int, name: str) -> None:
    """Create one owner-controlled directory exclusively; caller owns flushing."""
    descriptor, _ = _open_directory(_directory_handle(parent), name, _FILE_CREATE)
    os.close(descriptor)


def open_directory_chain(path: str | os.PathLike[str], *, create: bool) -> list[int]:
    """Pin every component from a directly mapped local volume's root."""
    text = os.fspath(path)
    if not isinstance(text, str) or not text or "\x00" in text:
        raise ValueError("A nonempty Windows text path is required.")
    # Validate before Win32 absolute-path normalization can erase aliases.
    for component in ntpath.splitdrive(text)[1].replace("/", "\\").split("\\"):
        if component not in {"", "."}:
            _validate_name(component)
    if text.startswith(("\\\\", "//")):
        raise ValueError("UNC and device paths are not admitted.")
    absolute = ntpath.abspath(text)
    drive, tail = ntpath.splitdrive(absolute)
    if (
        len(drive) != 2
        or drive[1] != ":"
        or not drive[0].isascii()
        or not drive[0].isalpha()
    ):
        raise ValueError("Host paths require a local drive letter.")
    kernel, _, _ = _apis()
    target = ctypes.create_unicode_buffer(32768)
    if not kernel.QueryDosDeviceW(drive, target, len(target)):
        _raise_windows(ctypes.get_last_error(), drive)
    device = target.value
    if not device.startswith("\\Device\\") or "\\" in device[len("\\Device\\") :]:
        raise ValueError("Only directly mapped local volume devices are admitted.")
    descriptors: list[int] = []
    try:
        descriptor, _ = _open_directory(None, device + "\\", _FILE_OPEN)
        descriptors.append(descriptor)
        for component in tail.replace("/", "\\").split("\\"):
            if component in {"", "."}:
                continue
            descriptors.append(
                open_directory_at(descriptors[-1], component, create=create)
            )
        return descriptors
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def open_file_at(
    parent: int,
    name: str,
    *,
    write: bool = False,
    create: bool = False,
    exclusive: bool = False,
) -> int:
    """Open a regular file without truncation, following no reparse point."""
    if exclusive and not create:
        raise ValueError("Exclusive file creation requires create=True.")
    disposition = _FILE_CREATE if exclusive else _FILE_OPEN_IF if create else _FILE_OPEN
    handle, _ = _open_native(
        _directory_handle(parent),
        name,
        access=_GENERIC_READ | (_GENERIC_WRITE if write else 0),
        disposition=disposition,
        options=_FILE_NON_DIRECTORY_FILE,
        share=_SHARE_ALL,
    )
    try:
        _require_not_reparse(_attributes(handle))
        kernel, _, _ = _apis()
        if kernel.GetFileType(handle) != 1:  # FILE_TYPE_DISK
            raise ValueError("Host resource must be a regular file.")
    except BaseException:
        _close_handle(handle)
        raise
    return _adopt_handle(handle, write=write)


def stat_at(parent: int, name: str) -> os.stat_result:
    """Return lstat information obtained from the entry itself, never its target."""
    handle, _ = _open_native(
        _directory_handle(parent), name, access=_FILE_READ_ATTRIBUTES
    )
    try:
        information = _attributes(handle)
    except BaseException:
        _close_handle(handle)
        raise
    descriptor = _adopt_handle(handle)
    try:
        result = os.fstat(descriptor)
        if not information.ReparseTag & 0x20000000:  # IsReparseTagNameSurrogate
            return result
        fields = list(result)
        fields[0] = stat.S_IFLNK | stat.S_IMODE(result.st_mode)
        return os.stat_result(
            fields,
            {
                "st_atime": result.st_atime,
                "st_mtime": result.st_mtime,
                "st_ctime": result.st_ctime_ns / 1_000_000_000,
                "st_atime_ns": result.st_atime_ns,
                "st_mtime_ns": result.st_mtime_ns,
                "st_ctime_ns": result.st_ctime_ns,
                "st_birthtime": result.st_birthtime,
                "st_birthtime_ns": result.st_birthtime_ns,
                "st_file_attributes": information.FileAttributes,
                "st_reparse_tag": information.ReparseTag,
            },
        )
    finally:
        os.close(descriptor)


def _file_identity(handle: int) -> tuple[int, bytes]:
    kernel, _, _ = _apis()
    information = _FileIdInformation()
    if not kernel.GetFileInformationByHandleEx(
        handle,
        18,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        _raise_windows(ctypes.get_last_error())
    return information.VolumeSerialNumber, bytes(information.FileId)


def _reopen_directory(descriptor: int, access: int) -> int:
    """Reopen the held directory itself with explicit native directory options.

    An empty counted name relative to the original handle identifies the same
    object, not a reconstructed pathname. Do not use the Win32 file-reopening
    route, which can impose non-directory open semantics.
    """
    _, native, _ = _apis()
    original = _directory_handle(descriptor)
    identity = _file_identity(original)
    name = _UnicodeString()
    attributes = _ObjectAttributes(
        ctypes.sizeof(_ObjectAttributes),
        original,
        ctypes.pointer(name),
        0x40,
        None,
        None,
    )
    handle = _HANDLE()
    status_block = _IoStatusBlock()
    try:
        result = native.NtCreateFile(
            ctypes.byref(handle),
            access | _SYNCHRONIZE | _FILE_READ_ATTRIBUTES,
            ctypes.byref(attributes),
            ctypes.byref(status_block),
            None,
            0,
            _SHARE_READ_WRITE,
            _FILE_OPEN,
            _FILE_DIRECTORY_FILE
            | _FILE_SYNCHRONOUS_IO_NONALERT
            | _FILE_OPEN_REPARSE_POINT,
            None,
            0,
        )
        _check_status(result)
        if handle.value is None:
            raise OSError(errno.EIO, "NtCreateFile directory reopen returned no handle.")
        information = _attributes(handle.value)
        _require_not_reparse(information)
        if not information.FileAttributes & _FILE_ATTRIBUTE_DIRECTORY:
            raise NotADirectoryError(errno.ENOTDIR, "Reopened object is not a directory.")
        if _file_identity(handle.value) != identity:
            raise OSError(
                errno.EIO, "Reopened directory identity does not match held object."
            )
        return handle.value
    except BaseException:
        if handle.value is not None:
            _close_handle(handle.value)
        raise


def list_directory(descriptor: int) -> list[str]:
    """Enumerate the held directory using a private, independent native cursor."""
    handle = _reopen_directory(descriptor, _DIRECTORY_ACCESS)
    try:
        _, native, _ = _apis()
        # Enough for many maximum-length local filesystem components.
        buffer = ctypes.create_string_buffer(65536)
        status_block = _IoStatusBlock()
        result: list[str] = []
        restart = True
        while True:
            status = native.NtQueryDirectoryFile(
                handle,
                None,
                None,
                None,
                ctypes.byref(status_block),
                buffer,
                ctypes.sizeof(buffer),
                12,
                False,
                None,
                restart,
            )
            if status & 0xFFFFFFFF == 0x80000006:  # STATUS_NO_MORE_FILES
                return result
            _check_status(status)
            used = status_block.Information
            if not used or used > ctypes.sizeof(buffer):
                raise OSError(errno.EIO, "Invalid native directory enumeration length.")
            offset = 0
            while True:
                start = offset + _FileNamesInformation.FileName.offset
                if start > used:
                    raise OSError(errno.EIO, "Truncated native directory entry.")
                entry = _FileNamesInformation.from_buffer(buffer, offset)
                length = entry.FileNameLength
                if length % 2 or start + length > used:
                    raise OSError(errno.EIO, "Invalid native directory filename length.")
                name = ctypes.string_at(ctypes.addressof(buffer) + start, length).decode(
                    "utf-16-le", "surrogatepass"
                )
                if name not in {".", ".."}:
                    result.append(name)
                if entry.NextEntryOffset == 0:
                    break
                if entry.NextEntryOffset < start + length - offset:
                    raise OSError(errno.EIO, "Overlapping native directory entries.")
                offset += entry.NextEntryOffset
            restart = False
    finally:
        _close_handle(handle)


def _namespace_source(parent: int, name: str, *, delete: bool) -> int:
    handle, _ = _open_native(
        parent,
        name,
        access=_FILE_READ_ATTRIBUTES | (_DELETE if delete else 0),
    )
    try:
        _require_not_reparse(_attributes(handle))
        return handle
    except BaseException:
        _close_handle(handle)
        raise


def unlink_at(parent: int, name: str, *, directory: bool = False) -> None:
    """Remove the opened entry, refusing links and nonempty directories."""
    handle = _namespace_source(_directory_handle(parent), name, delete=True)
    try:
        is_directory = bool(
            _attributes(handle).FileAttributes & _FILE_ATTRIBUTE_DIRECTORY
        )
        if directory != is_directory:
            error = errno.ENOTDIR if directory else errno.EISDIR
            raise OSError(error, "Unlink entry type does not match the request.", name)
        _, native, _ = _apis()
        flags = _ULONG(0x01 | 0x02)  # DELETE | POSIX_SEMANTICS; removed on this close.
        status_block = _IoStatusBlock()
        _check_status(
            native.NtSetInformationFile(
                handle,
                ctypes.byref(status_block),
                ctypes.byref(flags),
                ctypes.sizeof(flags),
                64,
            ),
            name,
        )
    finally:
        _close_handle(handle)


def _set_name(
    handle: int, parent: int, destination: str, *, flags: int, kind: int
) -> None:
    _validate_name(destination)
    encoded = destination.encode("utf-16-le", "surrogatepass")
    if len(encoded) > 65532:
        raise ValueError("A native filename must fit a counted UTF-16 string.")
    buffer = ctypes.create_string_buffer(ctypes.sizeof(_NameInformation) + len(encoded))
    information = _NameInformation.from_buffer(buffer)
    information.Flags = flags
    information.RootDirectory = parent
    information.FileNameLength = len(encoded)
    ctypes.memmove(
        ctypes.addressof(buffer) + _NameInformation.FileName.offset, encoded, len(encoded)
    )
    status_block = _IoStatusBlock()
    _, native, _ = _apis()
    _check_status(
        native.NtSetInformationFile(
            handle,
            ctypes.byref(status_block),
            buffer,
            ctypes.sizeof(buffer),
            kind,
        ),
        destination,
    )


def rename_at(parent: int, source: str, destination: str, *, replace: bool) -> None:
    """Atomically rename within the held parent; false means atomic no-replace."""
    parent_handle = _directory_handle(parent)
    _validate_name(destination)
    handle = _namespace_source(parent_handle, source, delete=True)
    try:
        _set_name(handle, parent_handle, destination, flags=3 if replace else 0, kind=65)
    finally:
        _close_handle(handle)


def link_at(parent: int, source: str, destination: str) -> None:
    """Exclusively create a hard link to the pinned regular source file."""
    parent_handle = _directory_handle(parent)
    _validate_name(destination)
    handle = _namespace_source(parent_handle, source, delete=False)
    try:
        if _attributes(handle).FileAttributes & _FILE_ATTRIBUTE_DIRECTORY:
            raise IsADirectoryError(errno.EISDIR, "Cannot hardlink a directory.", source)
        _set_name(handle, parent_handle, destination, flags=0, kind=11)
    finally:
        _close_handle(handle)


def sync_directory(descriptor: int) -> None:
    """Request a full NTFS directory flush; raise rather than weaken durability."""
    kernel, _, _ = _apis()
    filesystem = ctypes.create_unicode_buffer(32)
    if not kernel.GetVolumeInformationByHandleW(
        _directory_handle(descriptor),
        None,
        0,
        None,
        None,
        None,
        filesystem,
        len(filesystem),
    ):
        _raise_windows(ctypes.get_last_error())
    if filesystem.value != "NTFS":
        raise OSError(errno.ENOTSUP, "Directory namespace flushing requires NTFS.")
    handle = _reopen_directory(descriptor, _FILE_ADD_FILE | _FILE_READ_ATTRIBUTES)
    try:
        _, native, _ = _apis()
        status_block = _IoStatusBlock()
        _check_status(
            native.NtFlushBuffersFileEx(handle, 0, None, 0, ctypes.byref(status_block))
        )
    finally:
        _close_handle(handle)


def require_owned_directory(descriptor: int) -> None:
    """Require effective-user ownership and a conservative owner-controlled DACL."""
    handle = _directory_handle(descriptor)
    kernel, _, security = _apis()
    user = ctypes.create_string_buffer(_current_user_sid())
    owner = ctypes.c_void_p()
    dacl = ctypes.c_void_p()
    security_descriptor = ctypes.c_void_p()
    code = security.GetSecurityInfo(
        handle,
        1,
        0x01 | 0x04,
        ctypes.byref(owner),
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(security_descriptor),
    )
    if code:
        _raise_windows(code)
    try:
        if not owner.value or not security.EqualSid(owner, user):
            raise PermissionError(
                errno.EACCES, "Directory is not owned by the effective user."
            )
        if not dacl.value or not security.IsValidAcl(dacl):
            raise PermissionError(
                errno.EACCES, "Directory requires a valid non-NULL DACL."
            )
        size = _AclSizeInformation()
        if not security.GetAclInformation(
            dacl, ctypes.byref(size), ctypes.sizeof(size), 2
        ):
            _raise_windows(ctypes.get_last_error())
        for index in range(size.AceCount):
            pointer = ctypes.c_void_p()
            if not security.GetAce(dacl, index, ctypes.byref(pointer)):
                _raise_windows(ctypes.get_last_error())
            if pointer.value is None:
                raise OSError(errno.EIO, "GetAce returned a NULL entry.")
            header = _AceHeader.from_address(pointer.value)
            if header.AceType == 1:  # ACCESS_DENIED_ACE_TYPE cannot grant rights.
                continue
            if header.AceType != 0 or header.AceSize < ctypes.sizeof(_AccessAllowedAce):
                raise PermissionError(
                    errno.EACCES, "Directory has an unsupported DACL ACE."
                )
            entry = _AccessAllowedAce.from_address(pointer.value)
            if not entry.Mask & _MUTATING_RIGHTS:
                continue
            sid = pointer.value + _AccessAllowedAce.SidStart.offset
            if not security.IsValidSid(sid):
                raise PermissionError(
                    errno.EACCES, "Directory DACL has an invalid trustee."
                )
            trusted = (
                security.EqualSid(sid, user)
                or security.IsWellKnownSid(sid, 22)  # WinLocalSystemSid
                or security.IsWellKnownSid(sid, 26)  # WinBuiltinAdministratorsSid
            )
            if not trusted:
                raise PermissionError(
                    errno.EACCES, "Directory grants mutation to another trustee."
                )
    finally:
        if security_descriptor.value is not None:
            kernel.LocalFree(security_descriptor)
