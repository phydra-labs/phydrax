# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Host-only native dependency evidence shared by ELF and Windows providers."""

from __future__ import annotations

import os
from typing import final, Literal, TypeAlias

import equinox as eqx

from ._strict import StrictModule
from .typing import parse


NativeMachine: TypeAlias = Literal["x86_64", "aarch64"]
NativeObjectFormat: TypeAlias = Literal["elf", "pe"]


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
class ResolvedDependency(StrictModule):
    """A loader request satisfied by an owned file or resident library."""

    soname: str = eqx.field(static=True)
    path: str = eqx.field(static=True)
    requested_by: str = eqx.field(static=True)
    preloaded: bool = eqx.field(static=True)

    def __init__(
        self, *, soname: str, path: str, requested_by: str, preloaded: bool
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
    """A missing dependency and its requesting object."""

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
    """Format-qualified dependency evidence, with no implied runtime admission."""

    root: str = eqx.field(static=True)
    machine: NativeMachine = eqx.field(static=True)
    object_format: NativeObjectFormat = eqx.field(static=True)
    resolved: tuple[ResolvedDependency, ...] = eqx.field(static=True)
    unresolved: tuple[UnresolvedDependency, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        root: str,
        machine: NativeMachine,
        object_format: NativeObjectFormat,
        resolved: tuple[ResolvedDependency, ...],
        unresolved: tuple[UnresolvedDependency, ...],
    ) -> None:
        root = _text(root, "root")
        machine = parse(machine, NativeMachine, "machine")
        object_format = parse(object_format, NativeObjectFormat, "object_format")
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
        self.object_format = object_format
        self.resolved = resolved
        self.unresolved = unresolved


def resolve_native_dependencies(
    path: str | os.PathLike[str],
    /,
    *,
    library_path: tuple[str, ...],
    executable: str | None,
    preloaded: tuple[tuple[str, str], ...] = (),
) -> DependencyClosure:
    """Inspect ELF loader rules or the Windows runtime's controlled DLL closure."""
    with open(path, "rb") as stream:
        signature = stream.read(4)
    if signature == b"\x7fELF":
        from ._elf_dependencies import resolve_dependency_closure

        return resolve_dependency_closure(
            path, library_path=library_path, executable=executable, preloaded=preloaded
        )
    if signature[:2] != b"MZ":
        raise ValueError(f"{os.fspath(path)}: unsupported native object format")
    try:
        from phydrax_cuda13_pjrt.runtime import dependency_closure

        closure = dependency_closure(os.fspath(path))
    except (ImportError, RuntimeError) as error:
        raise ValueError(f"Windows dependency inspection refused: {error}") from error
    if closure.machine != 0x8664:
        raise ValueError(f"Unsupported Windows CUDA machine: {closure.machine:#x}")
    resolved = tuple(
        ResolvedDependency(
            soname=edge.name,
            path=edge.path,
            requested_by=edge.requester,
            preloaded=edge.kind == "resident",
        )
        for edge in closure.edges
        if edge.path is not None and edge.kind in ("owned", "resident", "system")
    )
    unresolved = tuple(
        UnresolvedDependency(
            soname=edge.name,
            requested_by=edge.requester,
            reason="not resolved by the owned Windows CUDA runtime",
        )
        for edge in closure.edges
        if edge.kind == "unresolved"
    )
    if not closure.complete or closure.issues:
        raise ValueError(
            "Windows dependency inspection is incomplete: " + "; ".join(closure.issues)
        )
    return DependencyClosure(
        root=closure.root,
        machine="x86_64",
        object_format="pe",
        resolved=resolved,
        unresolved=unresolved,
    )
