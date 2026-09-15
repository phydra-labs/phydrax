#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

from ...particle_physics import EventWeightSet, PreparedParticleEvents
from ._common import (
    HEPColumnProfile,
    HEPImportResult,
    HEPOptionalDependencyError,
    import_event_columns,
)


def read_root_event_tree(
    path: str | Path,
    tree_name: str,
    profile: HEPColumnProfile,
    prepared: PreparedParticleEvents,
    weights: EventWeightSet,
    /,
    *,
    source_id: str,
) -> HEPImportResult:
    """Read one explicitly profiled ROOT tree through optional uproot."""
    if importlib.util.find_spec("uproot") is None:
        raise HEPOptionalDependencyError(
            "ROOT event import requires the optional radiation-interop dependency."
        )
    tree_name_ = str(tree_name).strip()
    if not tree_name_:
        raise ValueError("tree_name must be non-empty.")
    uproot = importlib.import_module("uproot")
    source = Path(path)
    with uproot.open(source) as root_file:
        tree = root_file[tree_name_]
        branch_names = tuple(profile.fields.values())
        arrays = tree.arrays(branch_names, library="np")
    return import_event_columns(
        arrays,
        prepared,
        profile,
        weights,
        source_id=source_id,
    )


__all__ = ["read_root_event_tree"]
