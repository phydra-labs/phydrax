#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded CAD resource admission and suffix dispatch."""

from __future__ import annotations

import os
from collections.abc import Callable
from importlib import import_module
from pathlib import Path

from .._external_resource import (
    bounded_resource_from_bytes,
    BoundedResource,
    read_bounded_resource,
    ResourceReadError,
)
from ..typing import checked
from ..units._dimension import LENGTH
from ..units._unit import UnitDefinition
from ._cad import CadImportPolicy, CadImportResult, refuse, resource_refusal


type CadResourceDecoder = Callable[[BoundedResource, CadImportPolicy], CadImportResult]


@checked
def decode_cad_bytes(
    data: bytes,
    policy: CadImportPolicy,
    /,
    *,
    decoder: CadResourceDecoder,
    source_path: str | None = None,
) -> CadImportResult:
    """Admit a bounded byte resource and pass it to its format decoder."""
    try:
        resource = bounded_resource_from_bytes(
            data, limits=policy.limits, source_path=source_path
        )
    except ResourceReadError as error:
        resource_refusal(error)
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
    """Descriptor-read a bounded local resource and decode its admitted format."""
    try:
        resource = read_bounded_resource(
            path, trusted_root=trusted_root, limits=policy.limits
        )
    except ResourceReadError as error:
        resource_refusal(error)
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
    """Read STEP, IGES, or OCCT BRep text under bounded resource admission.

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
    # Codecs import this downward resource owner; resolve one selected codec at
    # the format boundary rather than introducing an eager import cycle.
    codec = import_module(f"._{format_name}", __package__)
    if format_name == "cad_brep_text":
        return codec.read_brep_text(
            path, policy, trusted_root=trusted_root, source_length_unit=source_length_unit
        )
    if format_name == "step":
        return codec.read_step(path, policy, trusted_root=trusted_root)
    return codec.read_iges(path, policy, trusted_root=trusted_root)


__all__ = ["CadResourceDecoder", "decode_cad_bytes", "read_cad_file", "read_cad"]
