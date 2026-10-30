#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Descriptor-owned native CAD resource input and format dispatch."""

from __future__ import annotations

import hashlib
import mmap
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from .._host_io import open_regular_beneath
from ..typing import checked
from ..units._dimension import LENGTH
from ..units._unit import UnitDefinition
from ._cad import CadImportPolicy, CadImportResult, refuse


@dataclass(frozen=True, slots=True)
class CadResource:
    """One parser input backed by caller bytes or a held read-only file mapping."""

    data: bytes | mmap.mmap
    digest: str
    source_path: str | None = None


type CadResourceDecoder = Callable[[CadResource, CadImportPolicy], CadImportResult]


@contextmanager
def _mapped_resource(
    path: str | os.PathLike[str], trusted_root: str | os.PathLike[str], /
) -> Iterator[CadResource]:
    with open_regular_beneath(
        path,
        trusted_root=trusted_root,
        maximum_depth=max(1, len(Path(path).parts)),
    ) as opened:
        with opened.duplicate_stream() as stream:
            hasher = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                hasher.update(chunk)
            digest = hasher.hexdigest()
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(0)
            if size:
                with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                    yield CadResource(mapped, digest, os.fspath(path))
            else:
                yield CadResource(b"", digest, os.fspath(path))
        opened.verify_stable()


@checked
def decode_cad_bytes(
    data: bytes,
    policy: CadImportPolicy,
    /,
    *,
    decoder: CadResourceDecoder,
    source_path: str | None = None,
) -> CadImportResult:
    """Decode caller-owned bytes with the native CAD codec."""
    if not isinstance(data, bytes):
        raise TypeError("CAD input must be bytes.")
    resource = CadResource(data, hashlib.sha256(data).hexdigest(), source_path)
    return decoder(resource, policy)


@checked
def read_cad_file(
    path: str | os.PathLike[str],
    policy: CadImportPolicy,
    /,
    *,
    decoder: CadResourceDecoder,
    trusted_root: str | os.PathLike[str],
) -> CadImportResult:
    """Decode one indexed file while its confined source descriptor remains held."""
    with _mapped_resource(path, trusted_root) as resource:
        return decoder(resource, policy)


@checked
def read_cad(
    path: str | os.PathLike[str],
    policy: CadImportPolicy,
    /,
    *,
    trusted_root: str | os.PathLike[str],
    source_length_unit: UnitDefinition | None = None,
) -> CadImportResult:
    """Read STEP, IGES, or OCCT BRep text through its native codec.

    BRep text requires an explicit length unit; embedded STEP and IGES units
    cannot be overridden.
    """
    if source_length_unit is not None and source_length_unit.dimension != LENGTH:
        raise ValueError(
            f"source_length_unit.dimension must be length; got {source_length_unit.dimension!r}"
        )
    suffix = Path(path).suffix.lower()
    match suffix:
        case ".step" | ".stp":
            format_name = "step"
        case ".iges" | ".igs":
            format_name = "iges"
        case ".brep" | ".brp":
            format_name = "cad_brep_text"
        case _:
            raise refuse(
                "unsupported-entity",
                suffix or "file",
                "Native CAD dispatch supports .step/.stp, .iges/.igs and .brep/.brp.",
            )
    if format_name != "cad_brep_text" and source_length_unit is not None:
        raise refuse(
            "units",
            format_name.upper(),
            f"{format_name.upper()} embedded units cannot be overridden.",
        )
    if format_name == "cad_brep_text" and source_length_unit is None:
        raise refuse("units", "OCCT BRep text", "BRep text requires source_length_unit.")
    codec = import_module(f"._{format_name}", __package__)
    if format_name == "cad_brep_text":
        return codec.read_brep_text(
            path,
            policy,
            trusted_root=trusted_root,
            source_length_unit=source_length_unit,
        )
    if format_name == "step":
        return codec.read_step(path, policy, trusted_root=trusted_root)
    return codec.read_iges(path, policy, trusted_root=trusted_root)


__all__ = [
    "CadResource",
    "CadResourceDecoder",
    "decode_cad_bytes",
    "read_cad_file",
    "read_cad",
]
