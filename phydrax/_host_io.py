#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Descriptor-relative host I/O with no-follow admission and explicit durability.

All returned handles are owned CRT/POSIX descriptors usable with ``os.close``,
``os.dup`` and ``os.fstat``. Windows namespace operations use native relative
handles, never reconstructed paths; POSIX operations retain ``dir_fd`` and
``O_NOFOLLOW``. Namespace mutation is not a durability acknowledgement until
``sync_directory`` succeeds.
"""

from __future__ import annotations

import ctypes
import errno
import io
import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


if os.name == "nt":
    from . import _windows_host_io
else:
    _DIRECTORY_OPEN_FLAGS = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    )
    _REGULAR_FILE_OPEN_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


def _validate_component(name: str) -> None:
    """Reject traversal and, on Windows, separators, streams and DOS aliases."""

    if (
        type(name) is not str
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\x00" in name
    ):
        raise ValueError("Descriptor-relative path is invalid.")
    if os.name == "nt" and (
        any(character in '\\:<>|"?*' or ord(character) < 32 for character in name)
        or name.endswith((".", " "))
        or name.split(".", 1)[0].upper()
        in {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "CONIN$",
            "CONOUT$",
            *(f"COM{number}" for number in "123456789¹²³"),
            *(f"LPT{number}" for number in "123456789¹²³"),
        }
    ):
        raise ValueError("Descriptor-relative Windows filename is invalid.")


def open_directory_at(parent: int, name: str, *, create: bool = False) -> int:
    """Open a no-follow child directory, optionally creating its durable entry."""

    _validate_component(name)
    if os.name == "nt":
        return _windows_host_io.open_directory_at(parent, name, create=create)
    if create:
        try:
            create_directory_at(parent, name)
        except FileExistsError:
            pass
        else:
            sync_directory(parent)
    return os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent)


def create_directory_at(parent: int, name: str) -> None:
    """Create an exclusive private child directory; the caller owns its flush."""

    _validate_component(name)
    if os.name == "nt":
        _windows_host_io.create_directory_at(parent, name)
    else:
        os.mkdir(name, mode=0o700, dir_fd=parent)


def open_file_at(
    parent: int,
    name: str,
    *,
    write: bool = False,
    create: bool = False,
    exclusive: bool = False,
) -> int:
    """Open only a no-follow regular file, without truncation, as an owned fd.

    ``write`` grants read/write access. Creation uses private permissions;
    ``exclusive`` requires creation and refuses an existing namespace entry.
    """

    _validate_component(name)
    if exclusive and not create:
        raise ValueError("Exclusive file opening requires creation.")
    if os.name == "nt":
        return _windows_host_io.open_file_at(
            parent, name, write=write, create=create, exclusive=exclusive
        )
    flags = _REGULAR_FILE_OPEN_FLAGS
    if write:
        flags = (flags & ~os.O_ACCMODE) | os.O_RDWR
    if create:
        flags |= os.O_CREAT
    if exclusive:
        flags |= os.O_EXCL
    descriptor = os.open(name, flags, mode=0o600, dir_fd=parent)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Host resource must be a regular file.")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def stat_at(parent: int, name: str) -> os.stat_result:
    """Read one bound entry's lstat identity without following links/reparse."""

    _validate_component(name)
    if os.name == "nt":
        return _windows_host_io.stat_at(parent, name)
    return os.stat(name, dir_fd=parent, follow_symlinks=False)


def list_directory(descriptor: int) -> list[str]:
    """List names from the held directory, independent of pathname replacement."""

    if os.name == "nt":
        return _windows_host_io.list_directory(descriptor)
    return os.listdir(descriptor)


def unlink_at(parent: int, name: str, *, directory: bool = False) -> None:
    """Remove one bound entry, never its symlink or reparse target."""

    _validate_component(name)
    if os.name == "nt":
        _windows_host_io.unlink_at(parent, name, directory=directory)
    elif directory:
        os.rmdir(name, dir_fd=parent)
    else:
        os.unlink(name, dir_fd=parent)


def rename_at(parent: int, source: str, destination: str, *, replace: bool) -> None:
    """Atomically rename siblings, refusing any existing target unless replacing.

    No-replace uses the host's atomic exclusion primitive, not a pathname check
    followed by rename. Unsupported kernels/filesystems fail closed.
    Ordinary POSIX rename semantics require ``replace=True``; they do not
    require rename flags. Exclusive regular-file publication can instead use
    ``link_at`` followed by ``unlink_at`` before synchronizing the directory.
    """

    _validate_component(source)
    _validate_component(destination)
    if os.name == "nt":
        _windows_host_io.rename_at(parent, source, destination, replace=replace)
        return
    if replace:
        os.replace(source, destination, src_dir_fd=parent, dst_dir_fd=parent)
        return
    library = ctypes.CDLL(None, use_errno=True)
    try:
        if sys.platform == "darwin":
            operation = library.renameatx_np
            flags = 0x00000004  # RENAME_EXCL
        elif sys.platform.startswith("linux"):
            operation = library.renameat2
            flags = 0x00000001  # RENAME_NOREPLACE
        else:
            raise OSError(errno.ENOTSUP, "Atomic exclusive rename is unsupported.")
    except AttributeError as error:
        raise OSError(errno.ENOTSUP, "Atomic exclusive rename is unavailable.") from error
    operation.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    operation.restype = ctypes.c_int
    if operation(parent, os.fsencode(source), parent, os.fsencode(destination), flags):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), destination)


def link_at(parent: int, source: str, destination: str) -> None:
    """Create an exclusive sibling hard link without following the source."""

    _validate_component(source)
    _validate_component(destination)
    if os.name == "nt":
        _windows_host_io.link_at(parent, source, destination)
    else:
        os.link(
            source,
            destination,
            src_dir_fd=parent,
            dst_dir_fd=parent,
            follow_symlinks=False,
        )


def sync_directory(descriptor: int) -> None:
    """Durably flush a directory namespace, propagating unsupported/failure."""

    if os.name == "nt":
        _windows_host_io.sync_directory(descriptor)
    else:
        os.fsync(descriptor)


def require_owned_directory(descriptor: int) -> None:
    """Require current ownership and no untrusted directory mutation rights.

    POSIX checks effective uid and group/other write bits. Windows checks the
    current token owner and the held directory's DACL, not synthetic mode bits.
    """

    if os.name == "nt":
        _windows_host_io.require_owned_directory(descriptor)
        return
    information = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(information.st_mode)
        or information.st_uid != os.geteuid()
        or information.st_mode & 0o022
    ):
        raise PermissionError(
            "Host directory must be owner-controlled and not group- or world-writable."
        )


@dataclass(frozen=True, slots=True)
class DescriptorRelativePath(os.PathLike[str]):
    """One validated filename relative to a caller-owned directory descriptor.

    Windows names additionally exclude separators, ADS and DOS aliases; POSIX
    filenames retain their native spelling and no-follow semantics.
    """

    directory_descriptor: int
    name: str

    def __post_init__(self) -> None:
        if type(self.directory_descriptor) is not int or self.directory_descriptor < 0:
            raise ValueError("Descriptor-relative path is invalid.")
        _validate_component(self.name)

    def __fspath__(self) -> str:
        return self.name


def descriptor_relative_path(
    directory_descriptor: int,
    name: str,
    /,
) -> DescriptorRelativePath:
    """Construct one validated descriptor-relative path."""

    return DescriptorRelativePath(directory_descriptor, name)


def _open_directory_chain(
    path: str | os.PathLike[str],
    /,
    *,
    create: bool,
) -> list[int]:
    """Hold every ancestor, with the existing root-owned Darwin alias policy."""

    if os.name == "nt":
        return _windows_host_io.open_directory_chain(path, create=create)
    value = Path(path)
    parts = value.parts
    if value.is_absolute() and len(parts) > 1 and parts[1] in {"var", "tmp"}:
        alias = Path("/") / parts[1]
        information = os.lstat(alias)
        expected_target = f"private/{parts[1]}"
        if (
            stat.S_ISLNK(information.st_mode)
            and information.st_uid == 0
            and information.st_mode & 0o022 == 0
            and os.readlink(alias) == expected_target
        ):
            parts = ("/", "private", parts[1], *parts[2:])
    descriptors = [os.open("/" if value.is_absolute() else ".", _DIRECTORY_OPEN_FLAGS)]
    start = 1 if value.is_absolute() else 0
    try:
        for part in parts[start:]:
            if part in {"", "."}:
                continue
            if part == "..":
                raise ValueError("Host paths cannot contain parent traversal.")
            descriptors.append(open_directory_at(descriptors[-1], part, create=create))
        return descriptors
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def open_directory_descriptor(
    path: str | os.PathLike[str],
    /,
    *,
    create: bool,
) -> int:
    """Open a directory component-by-component without following links."""

    descriptors = _open_directory_chain(path, create=create)
    for descriptor in descriptors[:-1]:
        os.close(descriptor)
    return descriptors[-1]


def open_parent_descriptor(
    path: str | os.PathLike[str],
    /,
    *,
    create: bool,
) -> tuple[int, str]:
    """Open a path's parent and return its descriptor plus final filename."""

    if isinstance(path, DescriptorRelativePath):
        return os.dup(path.directory_descriptor), path.name
    value = Path(path)
    if not value.name or value.name in {".", ".."}:
        raise ValueError("Host path must name one file.")
    _validate_component(value.name)
    return open_directory_descriptor(value.parent, create=create), value.name


@contextmanager
def open_regular_file(path: str | os.PathLike[str], /) -> Iterator[io.BufferedReader]:
    """Open one regular file without following it or any parent link."""

    directory_descriptor, name = open_parent_descriptor(path, create=False)
    descriptor = -1
    try:
        descriptor = open_file_at(directory_descriptor, name)
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            yield stream
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_descriptor)


def resource_components(
    path: str | os.PathLike[str],
    trusted_root: str | os.PathLike[str],
    maximum_depth: int,
    /,
) -> tuple[str, tuple[str, ...]]:
    """Normalize a local confined path without erasing traversal evidence.

    Windows accepts either separator but refuses UNC/device paths, drive-relative
    names and alternate streams. POSIX preserves its native filename semantics.
    """

    path_text = os.fspath(path)
    root_input = os.fspath(trusted_root)
    if not isinstance(path_text, str) or not isinstance(root_input, str):
        raise TypeError("Resource paths and trusted roots must be text paths.")
    if remote_location(path_text) or remote_location(root_input):
        raise ValueError("Network resource locations are disabled.")
    if "\x00" in path_text or "\x00" in root_input:
        raise ValueError("Resource paths cannot contain null bytes.")
    expanded_root = os.path.expanduser(root_input)
    expanded_path = os.path.expanduser(path_text)
    if remote_location(expanded_path) or remote_location(expanded_root):
        raise ValueError("Network resource locations are disabled.")
    separator = os.sep
    if os.name == "nt":
        expanded_root = expanded_root.replace("\\", "/")
        expanded_path = expanded_path.replace("\\", "/")
        separator = "/"
        for value in (expanded_path, expanded_root):
            drive, remainder = os.path.splitdrive(value)
            if (drive and not remainder.startswith("/")) or (
                not drive and remainder.startswith("/")
            ):
                raise ValueError("Resource paths cannot be drive- or root-relative.")
    for value in (expanded_path, expanded_root):
        if ".." in value.split(separator):
            raise ValueError("Resource traversal components are disabled.")
    root_text = os.path.abspath(expanded_root)
    relative_text = (
        os.path.relpath(expanded_path, root_text)
        if os.path.isabs(expanded_path)
        else expanded_path
    )
    if os.name == "nt":
        relative_text = relative_text.replace("\\", "/")
    components = tuple(
        part for part in relative_text.split(separator) if part and part != "."
    )
    if not components:
        raise ValueError("A resource file path is required.")
    if any(part == ".." for part in components):
        raise ValueError("The resource path escapes its trusted root.")
    for component in components:
        _validate_component(component)
    if len(components) > maximum_depth:
        raise OverflowError("Resource path nesting exceeds its depth limit.")
    return root_text, components


def remote_location(value: str, /) -> bool:
    """Return whether text denotes a remote or UNC resource location."""

    normalized = value.replace("\\", "/") if os.name == "nt" else value
    return "://" in normalized or normalized.startswith("//") or value.startswith("\\\\")


def stat_identity(status: os.stat_result, /) -> tuple[int, ...]:
    """Return the identity fields needed to detect file mutation."""

    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_mode),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
    )


def directory_identity(status: os.stat_result, /) -> tuple[int, ...]:
    """Return the identity fields needed to detect directory replacement."""

    return stat_identity(status)


def _held_directory_identity(status: os.stat_result, /) -> tuple[int, int, int]:
    return (int(status.st_dev), int(status.st_ino), int(status.st_mode))


def _verify_bound_entry(
    parent_descriptor: int,
    name: str,
    expected: os.stat_result,
    /,
) -> None:
    current = stat_at(parent_descriptor, name)
    if _held_directory_identity(expected) != _held_directory_identity(current):
        raise RuntimeError("A resource path component changed during admission.")


@dataclass(slots=True)
class OpenedHostFile:
    """Held descriptor and path identities for one admitted regular file."""

    descriptor: int
    root_path: str
    components: tuple[str, ...]
    root_status: os.stat_result
    file_status: os.stat_result
    directory_states: tuple[tuple[int, os.stat_result], ...]

    def rewind(self) -> None:
        os.lseek(self.descriptor, 0, os.SEEK_SET)

    @contextmanager
    def duplicate_stream(self) -> Iterator[BinaryIO]:
        """Yield a seekable stream without transferring the held descriptor."""

        with os.fdopen(os.dup(self.descriptor), "rb") as stream:
            yield stream

    def verify_stable(self) -> os.stat_result:
        """Verify file and traversed directory identities remain unchanged."""

        current = os.fstat(self.descriptor)
        if stat_identity(self.file_status) != stat_identity(current):
            raise RuntimeError("The resource changed while it was open.")
        for descriptor, initial in self.directory_states:
            if _held_directory_identity(initial) != _held_directory_identity(
                os.fstat(descriptor)
            ):
                raise RuntimeError("A resource path component changed while it was open.")
        return current


@dataclass(slots=True)
class OpenedHostDirectory:
    """Held descriptor and path identities for one admitted directory."""

    descriptor: int
    root_path: str
    components: tuple[str, ...]
    root_status: os.stat_result
    directory_status: os.stat_result
    directory_states: tuple[tuple[int, os.stat_result], ...]

    def verify_stable(self) -> os.stat_result:
        """Verify target and traversed directory identities remain unchanged."""

        current = os.fstat(self.descriptor)
        if _held_directory_identity(self.directory_status) != _held_directory_identity(
            current
        ):
            raise RuntimeError("The resource directory changed while it was open.")
        for descriptor, initial in self.directory_states:
            if _held_directory_identity(initial) != _held_directory_identity(
                os.fstat(descriptor)
            ):
                raise RuntimeError("A resource path component changed while it was open.")
        return current


@contextmanager
def open_directory_beneath(
    path: str | os.PathLike[str],
    /,
    *,
    trusted_root: str | os.PathLike[str],
    maximum_depth: int,
) -> Iterator[OpenedHostDirectory]:
    """Hold one directory and every traversed parent beneath a trusted root."""

    root_text, components = resource_components(
        path,
        trusted_root,
        maximum_depth,
    )
    descriptors: list[int] = []
    directory_states: list[tuple[int, os.stat_result]] = []
    component_bindings: list[tuple[int, str, os.stat_result]] = []
    try:
        root_descriptors = _open_directory_chain(root_text, create=False)
        descriptors.extend(root_descriptors)
        root_descriptor = root_descriptors[-1]
        root_status = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_status.st_mode):
            raise ValueError("The trusted resource root must be a real directory.")
        directory_states.extend(
            (descriptor, os.fstat(descriptor)) for descriptor in root_descriptors
        )
        parent_descriptor = root_descriptor
        for index, component in enumerate(components):
            descriptor = open_directory_at(parent_descriptor, component)
            descriptors.append(descriptor)
            status = os.fstat(descriptor)
            if not stat.S_ISDIR(status.st_mode):
                raise ValueError("Resource path components must be real directories.")
            component_bindings.append((parent_descriptor, component, status))
            parent_descriptor = descriptor
            if index + 1 < len(components):
                directory_states.append((descriptor, status))
        opened = OpenedHostDirectory(
            parent_descriptor,
            root_text,
            components,
            root_status,
            os.fstat(parent_descriptor),
            tuple(directory_states),
        )
        for parent, component, status in component_bindings:
            _verify_bound_entry(parent, component, status)
        opened.verify_stable()
        yield opened
        opened.verify_stable()
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


@contextmanager
def open_regular_beneath(
    path: str | os.PathLike[str],
    /,
    *,
    trusted_root: str | os.PathLike[str],
    maximum_depth: int,
) -> Iterator[OpenedHostFile]:
    """Hold one regular file and every traversed directory beneath a root."""

    root_text, components = resource_components(
        path,
        trusted_root,
        maximum_depth,
    )
    descriptors: list[int] = []
    directory_states: list[tuple[int, os.stat_result]] = []
    component_bindings: list[tuple[int, str, os.stat_result]] = []
    try:
        root_descriptors = _open_directory_chain(root_text, create=False)
        descriptors.extend(root_descriptors)
        root_descriptor = root_descriptors[-1]
        root_status = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_status.st_mode):
            raise ValueError("The trusted resource root must be a real directory.")
        directory_states.extend(
            (descriptor, os.fstat(descriptor)) for descriptor in root_descriptors
        )
        parent_descriptor = root_descriptor
        for component in components[:-1]:
            descriptor = open_directory_at(parent_descriptor, component)
            descriptors.append(descriptor)
            status = os.fstat(descriptor)
            if not stat.S_ISDIR(status.st_mode):
                raise ValueError("Resource path components must be real directories.")
            component_bindings.append((parent_descriptor, component, status))
            directory_states.append((descriptor, status))
            parent_descriptor = descriptor
        file_descriptor = open_file_at(parent_descriptor, components[-1])
        descriptors.append(file_descriptor)
        file_status = os.fstat(file_descriptor)
        if not stat.S_ISREG(file_status.st_mode):
            raise ValueError("The requested resource must be a regular file.")
        opened = OpenedHostFile(
            file_descriptor,
            root_text,
            components,
            root_status,
            file_status,
            tuple(directory_states),
        )
        for parent, component, status in component_bindings:
            _verify_bound_entry(parent, component, status)
        _verify_bound_entry(parent_descriptor, components[-1], file_status)
        opened.verify_stable()
        yield opened
        opened.verify_stable()
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


__all__ = [
    "DescriptorRelativePath",
    "OpenedHostDirectory",
    "OpenedHostFile",
    "create_directory_at",
    "descriptor_relative_path",
    "directory_identity",
    "link_at",
    "list_directory",
    "open_directory_at",
    "open_directory_beneath",
    "open_directory_descriptor",
    "open_file_at",
    "open_parent_descriptor",
    "open_regular_beneath",
    "open_regular_file",
    "remote_location",
    "rename_at",
    "require_owned_directory",
    "resource_components",
    "stat_at",
    "stat_identity",
    "sync_directory",
    "unlink_at",
]
