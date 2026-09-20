#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit meshio-backed file profiles; never provider-wide autodetection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


MeshCarrier = Literal["single-file", "resource-set"]


@dataclass(frozen=True, slots=True)
class MeshFileProfile:
    """One explicit meshio format name, carrier, and direction contract."""

    profile_id: str
    meshio_format: str
    extensions: tuple[str, ...]
    carrier: MeshCarrier
    readable: bool
    writable: bool


_PROFILES = tuple(
    MeshFileProfile(*record)
    for record in (
        ("abaqus", "abaqus", (".inp",), "single-file", True, True),
        ("ansys", "ansys", (".msh",), "single-file", True, True),
        ("avsucd", "avsucd", (".avs",), "single-file", True, True),
        ("cgns", "cgns", (".cgns",), "single-file", True, True),
        ("dolfin-xml", "dolfin-xml", (".xml",), "single-file", True, True),
        ("exodus", "exodus", (".e", ".exo", ".ex2"), "single-file", True, True),
        ("flac3d", "flac3d", (".f3grid",), "single-file", True, True),
        ("gmsh", "gmsh", (".msh",), "single-file", True, True),
        ("h5m", "h5m", (".h5m",), "single-file", True, True),
        ("hmf", "hmf", (".hmf",), "single-file", True, True),
        ("mdpa", "mdpa", (".mdpa",), "single-file", True, True),
        ("med", "med", (".med",), "single-file", True, True),
        ("medit", "medit", (".mesh", ".meshb"), "single-file", True, True),
        ("nastran", "nastran", (".bdf", ".fem", ".nas"), "single-file", True, True),
        ("netgen", "netgen", (".vol",), "single-file", True, True),
        ("neuroglancer", "neuroglancer", (".precomputed",), "resource-set", True, True),
        ("obj", "obj", (".obj",), "single-file", True, True),
        ("off", "off", (".off",), "single-file", True, True),
        ("permas", "permas", (".post", ".dato"), "single-file", True, True),
        ("ply", "ply", (".ply",), "single-file", True, True),
        ("stl", "stl", (".stl",), "single-file", True, True),
        ("su2", "su2", (".su2",), "single-file", True, True),
        ("svg", "svg", (".svg",), "single-file", False, True),
        ("tecplot", "tecplot", (".dat", ".tec"), "single-file", True, True),
        ("tetgen", "tetgen", (".node", ".ele"), "resource-set", True, False),
        ("ugrid", "ugrid", (".ugrid",), "single-file", True, True),
        ("vtk", "vtk", (".vtk",), "single-file", True, True),
        ("vtu", "vtu", (".vtu",), "single-file", True, True),
        ("wkt", "wkt", (".wkt",), "single-file", True, False),
        ("xdmf", "xdmf", (".xdmf", ".xmf"), "resource-set", True, True),
    )
)
_BY_ID = {profile.profile_id: profile for profile in _PROFILES}
_BY_EXTENSION: dict[str, tuple[MeshFileProfile, ...]] = {}
for _profile in _PROFILES:
    for _extension in _profile.extensions:
        _BY_EXTENSION[_extension] = (*_BY_EXTENSION.get(_extension, ()), _profile)
_DEFAULT_BY_EXTENSION = {".msh": "gmsh"}


def mesh_file_profiles() -> tuple[MeshFileProfile, ...]:
    """Return every explicitly declared meshio-backed profile."""

    return _PROFILES


def resolve_mesh_file_profile(
    path: str | Path,
    declared_profile: str | None,
    /,
    *,
    direction: Literal["read", "write"],
) -> MeshFileProfile:
    """Resolve an explicit profile or an unambiguous extension hint."""

    if declared_profile is not None:
        profile = _BY_ID.get(str(declared_profile))
        if profile is None:
            raise ValueError(f"Unknown mesh file profile {declared_profile!r}.")
    else:
        extension = Path(path).suffix.lower()
        default = _DEFAULT_BY_EXTENSION.get(extension)
        candidates = _BY_EXTENSION.get(extension, ())
        if default is not None:
            profile = _BY_ID[default]
        elif len(candidates) == 1:
            profile = candidates[0]
        else:
            raise ValueError(
                "Mesh file format is ambiguous; declare a mesh file profile."
            )
    if direction == "read" and not profile.readable:
        raise ValueError(f"Mesh profile {profile.profile_id!r} is not readable.")
    if direction == "write" and not profile.writable:
        raise ValueError(f"Mesh profile {profile.profile_id!r} is not writable.")
    return profile


__all__ = [
    "MeshCarrier",
    "MeshFileProfile",
    "mesh_file_profiles",
    "resolve_mesh_file_profile",
]
