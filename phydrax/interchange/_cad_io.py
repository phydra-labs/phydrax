#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native CAD suffix admission; format graphs remain with their codec owners."""

from __future__ import annotations

import os
from pathlib import Path

from ..units import LENGTH, UnitDefinition
from ._cad import CadImportPolicy, CadImportResult, refuse


def read_cad(
    path: str | os.PathLike[str],
    policy: CadImportPolicy,
    /,
    *,
    trusted_root: str | os.PathLike[str],
    source_length_unit: UnitDefinition | None = None,
) -> CadImportResult:
    """Decode native STEP, IGES, or the supported external OCCT BRep text profile.

    STEP and IGES own embedded units and placements; their units cannot be
    overridden here. OCCT BRep text carries no length unit and therefore requires
    ``source_length_unit`` explicitly. Each codec owns bounded graph admission,
    exact geometry/topology construction, coverage, refusals and source identity.
    This dispatch never loads the optional OCCT interoperability engine.
    """
    if not isinstance(policy, CadImportPolicy):
        raise TypeError("policy must be a CadImportPolicy.")
    if source_length_unit is not None:
        if not isinstance(source_length_unit, UnitDefinition):
            raise TypeError("source_length_unit must be a UnitDefinition or None.")
        if source_length_unit.dimension != LENGTH:
            raise ValueError("source_length_unit must have length dimension.")
    suffix = Path(path).suffix.lower()
    match suffix:
        case ".step" | ".stp":
            if source_length_unit is not None:
                raise refuse("units", "STEP", "STEP embedded units cannot be overridden.")
            from ._step import read_step

            return read_step(path, policy, trusted_root=trusted_root)
        case ".iges" | ".igs":
            if source_length_unit is not None:
                raise refuse("units", "IGES", "IGES embedded units cannot be overridden.")
            from ._iges import read_iges

            return read_iges(path, policy, trusted_root=trusted_root)
        case ".brep" | ".brp":
            if source_length_unit is None:
                raise refuse(
                    "units", "OCCT BRep text", "BRep text requires source_length_unit."
                )
            from ._cad_brep_text import read_brep_text

            return read_brep_text(
                path,
                policy,
                trusted_root=trusted_root,
                source_length_unit=source_length_unit,
            )
        case _:
            raise refuse(
                "unsupported-entity",
                suffix or "file",
                "Native CAD dispatch supports .step/.stp, .iges/.igs and .brep/.brp.",
            )


__all__ = ["read_cad"]
