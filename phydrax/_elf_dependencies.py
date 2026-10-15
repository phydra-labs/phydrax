#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-only ELF dependency inspection, without consulting process loader state."""

from __future__ import annotations

import os
import struct
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, final, Literal, TypeAlias

import equinox as eqx

from ._strict import StrictModule
from .typing import parse


ElfMachine: TypeAlias = Literal["x86_64", "aarch64"]


def _text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or "\0" in value:
        raise ValueError(f"{name} must be nonempty and contain no NUL.")
    return value


def _strings(values: tuple[str, ...], name: str, *, empty: bool = False) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple.")
    for value in values:
        if empty and value == "":
            continue
        _text(value, name)


@final
class ElfDynamic(StrictModule):
    """Unexpanded ELF dynamic metadata; RUNPATH suppresses RPATH."""

    path: str = eqx.field(static=True)
    machine: ElfMachine = eqx.field(static=True)
    soname: str | None = eqx.field(static=True)
    needed: tuple[str, ...] = eqx.field(static=True)
    rpath: tuple[str, ...] = eqx.field(static=True)
    runpath: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        path: str,
        machine: ElfMachine,
        soname: str | None,
        needed: tuple[str, ...],
        rpath: tuple[str, ...],
        runpath: tuple[str, ...],
    ) -> None:
        path = _text(path, "path")
        machine = parse(machine, ElfMachine, "machine")
        if soname is not None:
            _text(soname, "soname")
        _strings(needed, "needed")
        _strings(rpath, "rpath", empty=True)
        _strings(runpath, "runpath", empty=True)
        if rpath and runpath:
            raise ValueError("RPATH must be empty when RUNPATH exists.")
        self.path = path
        self.machine = machine
        self.soname = soname
        self.needed = needed
        self.rpath = rpath
        self.runpath = runpath


@final
class ResolvedDependency(StrictModule):
    """First loader request satisfied by a file or a preloaded soname."""

    soname: str = eqx.field(static=True)
    path: str = eqx.field(static=True)
    requested_by: str = eqx.field(static=True)
    preloaded: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        soname: str,
        path: str,
        requested_by: str,
        preloaded: bool,
    ) -> None:
        soname = _text(soname, "soname")
        path = _text(path, "path")
        requested_by = _text(requested_by, "requested_by")
        if not isinstance(preloaded, bool):
            raise TypeError("preloaded must be a bool.")
        self.soname = soname
        self.path = path
        self.requested_by = requested_by
        self.preloaded = preloaded


@final
class UnresolvedDependency(StrictModule):
    """A missing dependency and its first requesting object."""

    soname: str = eqx.field(static=True)
    requested_by: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)

    def __init__(self, *, soname: str, requested_by: str, reason: str) -> None:
        soname = _text(soname, "soname")
        requested_by = _text(requested_by, "requested_by")
        reason = _text(reason, "reason")
        self.soname = soname
        self.requested_by = requested_by
        self.reason = reason


@final
class DependencyClosure(StrictModule):
    """Fresh-process closure, with dependencies before their dependents."""

    root: str = eqx.field(static=True)
    machine: ElfMachine = eqx.field(static=True)
    resolved: tuple[ResolvedDependency, ...] = eqx.field(static=True)
    unresolved: tuple[UnresolvedDependency, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        root: str,
        machine: ElfMachine,
        resolved: tuple[ResolvedDependency, ...],
        unresolved: tuple[UnresolvedDependency, ...],
    ) -> None:
        root = _text(root, "root")
        machine = parse(machine, ElfMachine, "machine")
        if not isinstance(resolved, tuple) or any(
            not isinstance(item, ResolvedDependency) for item in resolved
        ):
            raise TypeError("resolved must be a tuple of ResolvedDependency.")
        if not isinstance(unresolved, tuple) or any(
            not isinstance(item, UnresolvedDependency) for item in unresolved
        ):
            raise TypeError("unresolved must be a tuple of UnresolvedDependency.")
        self.root = root
        self.machine = machine
        self.resolved = resolved
        self.unresolved = unresolved


def _read_at(stream: BinaryIO, offset: int, size: int) -> bytes:
    stream.seek(offset)
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated ELF data")
    return data


def _elf_headers(stream: BinaryIO) -> tuple[ElfMachine, list[tuple[int, ...]]]:
    header = _read_at(stream, 0, 64)
    if header[:4] != b"\x7fELF":
        raise ValueError("not an ELF file")
    if header[4] != 2:
        raise ValueError("requires ELF64")
    if header[5] != 1:
        raise ValueError("requires little-endian ELF")
    machine_number = struct.unpack_from("<H", header, 18)[0]
    match machine_number:
        case 62:
            machine: ElfMachine = "x86_64"
        case 183:
            machine = "aarch64"
        case _:
            raise ValueError(f"unsupported ELF machine {machine_number}")
    offset = struct.unpack_from("<Q", header, 32)[0]
    entry_size, count = struct.unpack_from("<HH", header, 54)
    if entry_size < 56:
        raise ValueError("invalid program header size")
    headers: list[tuple[int, ...]] = [
        struct.unpack("<IIQQQQQQ", _read_at(stream, offset + i * entry_size, 56))
        for i in range(count)
    ]
    return machine, headers


def _dynamic_tags(
    stream: BinaryIO, headers: list[tuple[int, ...]]
) -> dict[int, list[int]]:
    dynamic = next((header for header in headers if header[0] == 2), None)
    if dynamic is None:
        raise ValueError("no PT_DYNAMIC segment")
    tags: dict[int, list[int]] = {}
    data = _read_at(stream, dynamic[2], dynamic[5])
    if len(data) % 16:
        raise ValueError("invalid PT_DYNAMIC size")
    for tag, value in struct.iter_unpack("<qQ", data):
        if tag == 0:
            break
        tags.setdefault(tag, []).append(value)
    return tags


def _string_table(
    stream: BinaryIO,
    headers: list[tuple[int, ...]],
    tags: dict[int, list[int]],
) -> bytes:
    if 5 not in tags or 10 not in tags:
        raise ValueError("missing DT_STRTAB or DT_STRSZ")
    address, size = tags[5][0], tags[10][0]
    for header in headers:
        if (
            header[0] == 1
            and header[3] <= address
            and address + size <= header[3] + header[5]
        ):
            return _read_at(stream, header[2] + address - header[3], size)
    raise ValueError("DT_STRTAB is not backed by PT_LOAD")


def _cstring(data: bytes, offset: int) -> str:
    if offset < 0 or offset >= len(data):
        raise ValueError("string offset outside table")
    end = data.find(b"\0", offset)
    if end < 0:
        raise ValueError("unterminated string")
    return os.fsdecode(data[offset:end])


def _read_elf(path: str) -> tuple[ElfDynamic, bool]:
    try:
        with open(path, "rb") as stream:
            machine, headers = _elf_headers(stream)
            tags = _dynamic_tags(stream, headers)
            strings = _string_table(stream, headers, tags)
        needed = tuple(_cstring(strings, offset) for offset in tags.get(1, ()))
        soname = _cstring(strings, tags[14][0]) if 14 in tags else None
        rpath = (
            tuple(_cstring(strings, tags[15][0]).split(":"))
            if 15 in tags and 29 not in tags
            else ()
        )
        runpath = tuple(_cstring(strings, tags[29][0]).split(":")) if 29 in tags else ()
        return ElfDynamic(
            path=path,
            machine=machine,
            soname=soname,
            needed=needed,
            rpath=rpath,
            runpath=runpath,
        ), 29 in tags
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def read_dynamic_section(path: str | os.PathLike[str], /) -> ElfDynamic:
    """Read an ELF64-LE dynamic section without loading native code."""
    return _read_elf(os.fspath(path))[0]


def _cache_entries(path: str, machine: ElfMachine) -> dict[str, list[str]]:
    try:
        data = Path(path).read_bytes()
    except OSError:
        return {}
    magic = b"glibc-ld.so.cache1.1"
    base = 0
    if data.startswith(b"ld.so-1.7.0") and len(data) >= 16:
        old_count = struct.unpack_from("<I", data, 12)[0]
        base = (16 + old_count * 12 + 7) & ~7
    if len(data) < base + 48 or data[base : base + len(magic)] != magic:
        return {}
    count = struct.unpack_from("<I", data, base + 20)[0]
    if count > (len(data) - base - 48) // 24:
        return {}
    arch = 0x0300 if machine == "x86_64" else 0x0A00
    result: dict[str, list[str]] = {}
    for index in range(count):
        flags, key, value, _os_version, hwcap = struct.unpack_from(
            "<iIIIQ",
            data,
            base + 48 + index * 24,
        )
        if flags & 0xFF00 != arch or hwcap != 0:
            continue
        try:
            soname = _cstring(data, base + key)
            candidate = _cstring(data, base + value)
        except ValueError:
            continue
        result.setdefault(soname, []).append(candidate)
    return result


def _candidate(path: str, machine: ElfMachine) -> tuple[ElfDynamic, bool] | None:
    try:
        dynamic, has_runpath = _read_elf(path)
    except (OSError, ValueError):
        return None
    return (dynamic, has_runpath) if dynamic.machine == machine else None


def _expand(entry: str, owner: str) -> str | None:
    if any(token in entry for token in ("$LIB", "${LIB}", "$PLATFORM", "${PLATFORM}")):
        return None
    return entry.replace("${ORIGIN}", os.path.dirname(owner)).replace(
        "$ORIGIN", os.path.dirname(owner)
    )


@dataclass
class _Loaded:
    dynamic: ElfDynamic
    has_runpath: bool
    parent: int | None
    dependency: ResolvedDependency | None = None
    children: list[int] = field(default_factory=list)


def _rpath_chain(
    nodes: list[_Loaded], index: int, executable: ElfDynamic | None
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    node = nodes[index]
    if node.has_runpath:
        return result
    while True:
        result.extend((entry, node.dynamic.path) for entry in node.dynamic.rpath)
        if node.parent is None:
            break
        node = nodes[node.parent]
    if executable is not None:
        result.extend((entry, executable.path) for entry in executable.rpath)
    return result


def _find_dependency(
    soname: str,
    requester: ElfDynamic,
    entries: list[tuple[str, str]],
    cache: dict[str, list[str]],
) -> tuple[tuple[ElfDynamic, bool] | None, str]:
    unsupported = False
    if "/" in soname:
        direct = _expand(soname, requester.path)
        found = None if direct is None else _candidate(direct, requester.machine)
        return (
            found,
            "requires $LIB/$PLATFORM expansion" if direct is None else "not found",
        )
    for entry, owner in entries:
        directory = _expand(entry, owner)
        if directory is None:
            unsupported = True
            continue
        if directory:
            found = _candidate(os.path.join(directory, soname), requester.machine)
            if found is not None:
                return found, ""
    for path in cache.get(soname, ()):
        found = _candidate(path, requester.machine)
        if found is not None:
            return found, ""
    triplet = "x86_64-linux-gnu" if requester.machine == "x86_64" else "aarch64-linux-gnu"
    for directory in (
        f"/lib/{triplet}",
        f"/usr/lib/{triplet}",
        "/lib64",
        "/usr/lib64",
        "/lib",
        "/usr/lib",
    ):
        found = _candidate(os.path.join(directory, soname), requester.machine)
        if found is not None:
            return found, ""
    return None, "requires $LIB/$PLATFORM expansion" if unsupported else "not found"


def _dependency_order(nodes: list[_Loaded]) -> tuple[ResolvedDependency, ...]:
    visited: set[int] = set()
    result: list[ResolvedDependency] = []
    stack = [(0, False)]
    while stack:
        index, expanded = stack.pop()
        node = nodes[index]
        if expanded:
            if node.dependency is not None:
                result.append(node.dependency)
        elif index not in visited:
            visited.add(index)
            stack.append((index, True))
            stack.extend((child, False) for child in reversed(node.children))
    return tuple(result)


def _validate_resolution_inputs(
    library_path: tuple[str, ...],
    executable: str | None,
    preloaded: tuple[tuple[str, str], ...],
    ld_so_cache: str,
) -> None:
    _strings(library_path, "library_path", empty=True)
    if executable is not None:
        _text(executable, "executable")
    _text(ld_so_cache, "ld_so_cache")
    if not isinstance(preloaded, tuple):
        raise TypeError("preloaded must be a tuple of (soname, path) tuples.")
    for pair in preloaded:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError("preloaded must contain (soname, path) tuples.")
        _text(pair[0], "preloaded soname")
        _text(pair[1], "preloaded path")


def _load_dependencies(
    nodes: list[_Loaded],
    library_path: tuple[str, ...],
    executable: ElfDynamic | None,
    preloaded: tuple[tuple[str, str], ...],
    cache: dict[str, list[str]],
) -> tuple[UnresolvedDependency, ...]:
    root = nodes[0].dynamic
    aliases = {root.path: 0}
    if root.soname is not None:
        aliases[root.soname] = 0
    seeds: dict[str, str] = {}
    for soname, path in preloaded:
        seeds.setdefault(soname, path)
    unresolved: list[UnresolvedDependency] = []
    queue = deque([0])
    while queue:
        index = queue.popleft()
        node = nodes[index]
        entries = _rpath_chain(nodes, index, executable)
        entries.extend((entry, root.path) for entry in library_path)
        entries.extend((entry, node.dynamic.path) for entry in node.dynamic.runpath)
        for soname in node.dynamic.needed:
            if soname in aliases:
                node.children.append(aliases[soname])
                continue
            if soname in seeds:
                dynamic = ElfDynamic(
                    path=seeds[soname],
                    machine=root.machine,
                    soname=soname,
                    needed=(),
                    rpath=(),
                    runpath=(),
                )
                found = (dynamic, False)
            else:
                found, reason = _find_dependency(soname, node.dynamic, entries, cache)
                if found is None:
                    unresolved.append(
                        UnresolvedDependency(
                            soname=soname, requested_by=node.dynamic.path, reason=reason
                        )
                    )
                    continue
            dynamic, has_runpath = found
            child = len(nodes)
            aliases[soname] = child
            if dynamic.soname is not None:
                aliases.setdefault(dynamic.soname, child)
            dependency = ResolvedDependency(
                soname=soname,
                path=dynamic.path,
                requested_by=node.dynamic.path,
                preloaded=soname in seeds,
            )
            nodes.append(_Loaded(dynamic, has_runpath, index, dependency))
            node.children.append(child)
            if not dependency.preloaded:
                queue.append(child)
    return tuple(unresolved)


def resolve_dependency_closure(
    path: str | os.PathLike[str],
    /,
    *,
    library_path: tuple[str, ...],
    executable: str | None,
    preloaded: tuple[tuple[str, str], ...] = (),
    ld_so_cache: str = "/etc/ld.so.cache",
) -> DependencyClosure:
    """Model glibc's fresh-process breadth-first loads, never the live process."""
    _validate_resolution_inputs(library_path, executable, preloaded, ld_so_cache)
    root, has_runpath = _read_elf(os.fspath(path))
    main = _candidate(executable, root.machine) if executable is not None else None
    main_dynamic = main[0] if main is not None and not main[1] else None
    nodes = [_Loaded(root, has_runpath, None)]
    unresolved = _load_dependencies(
        nodes,
        library_path,
        main_dynamic,
        preloaded,
        _cache_entries(ld_so_cache, root.machine),
    )
    return DependencyClosure(
        root=root.path,
        machine=root.machine,
        resolved=_dependency_order(nodes),
        unresolved=unresolved,
    )
