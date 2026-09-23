#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Descriptor-safe host filesystem primitives shared by I/O substrates."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
)
_REGULAR_FILE_OPEN_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


@dataclass(frozen=True, slots=True)
class DescriptorRelativePath(os.PathLike[str]):
    """One validated filename relative to a caller-owned directory descriptor."""

    directory_descriptor: int
    name: str

    def __post_init__(self) -> None:
        if (
            type(self.directory_descriptor) is not int
            or self.directory_descriptor < 0
            or type(self.name) is not str
            or not self.name
            or self.name in {".", ".."}
            or "/" in self.name
            or "\x00" in self.name
        ):
            raise ValueError("Descriptor-relative path is invalid.")

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
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptors[-1])
                    os.fsync(descriptors[-1])
                except FileExistsError:
                    pass
            descriptors.append(
                os.open(part, _DIRECTORY_OPEN_FLAGS, dir_fd=descriptors[-1])
            )
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
    return open_directory_descriptor(value.parent, create=create), value.name


@contextmanager
def open_regular_file(path: str | os.PathLike[str], /) -> Iterator[BinaryIO]:
    """Open one regular file without following it or any parent link."""

    directory_descriptor, name = open_parent_descriptor(path, create=False)
    descriptor = -1
    try:
        descriptor = os.open(name, _REGULAR_FILE_OPEN_FLAGS, dir_fd=directory_descriptor)
        information = os.fstat(descriptor)
        if not stat.S_ISREG(information.st_mode):
            raise ValueError("Host resource must be a regular file.")
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
    """Normalize one local resource path beneath an explicit trusted root."""

    path_text = os.fspath(path)
    root_input = os.fspath(trusted_root)
    if not isinstance(path_text, str) or not isinstance(root_input, str):
        raise TypeError("Resource paths and trusted roots must be text paths.")
    if remote_location(path_text) or remote_location(root_input):
        raise ValueError("Network resource locations are disabled.")
    if "\x00" in path_text or "\x00" in root_input:
        raise ValueError("Resource paths cannot contain null bytes.")
    root_text = os.path.abspath(os.path.expanduser(root_input))
    expanded_path = os.path.expanduser(path_text)
    raw_parts = tuple(part for part in expanded_path.split(os.sep) if part)
    if ".." in raw_parts:
        raise ValueError("Resource traversal components are disabled.")
    relative_text = (
        os.path.relpath(expanded_path, root_text)
        if os.path.isabs(expanded_path)
        else expanded_path
    )
    components = tuple(
        part for part in relative_text.split(os.sep) if part and part != "."
    )
    if not components:
        raise ValueError("A resource file path is required.")
    if any(part == ".." for part in components):
        raise ValueError("The resource path escapes its trusted root.")
    if len(components) > maximum_depth:
        raise OverflowError("Resource path nesting exceeds its depth limit.")
    return root_text, components


def remote_location(value: str, /) -> bool:
    """Return whether text denotes a remote or UNC resource location."""

    return "://" in value or value.startswith(("//", "\\\\"))


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
    current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
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
            descriptor = os.open(
                component,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=parent_descriptor,
            )
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
            descriptor = os.open(
                component,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=parent_descriptor,
            )
            descriptors.append(descriptor)
            status = os.fstat(descriptor)
            if not stat.S_ISDIR(status.st_mode):
                raise ValueError("Resource path components must be real directories.")
            component_bindings.append((parent_descriptor, component, status))
            directory_states.append((descriptor, status))
            parent_descriptor = descriptor
        file_descriptor = os.open(
            components[-1],
            _REGULAR_FILE_OPEN_FLAGS,
            dir_fd=parent_descriptor,
        )
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
    "descriptor_relative_path",
    "directory_identity",
    "open_directory_beneath",
    "open_directory_descriptor",
    "open_parent_descriptor",
    "open_regular_beneath",
    "open_regular_file",
    "remote_location",
    "resource_components",
    "stat_identity",
]
