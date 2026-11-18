#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native composition persistence through registered model recipes."""

from __future__ import annotations

from pathlib import Path

from .._array_archive import ArrayArchiveLimits
from ..lifecycle import Composition


def save_composition_archive(
    composition: Composition,
    path: str | Path,
    /,
    *,
    ingress_limits: ArrayArchiveLimits | None = None,
) -> Path:
    """Write a composition and its registered model state."""
    if type(composition) is not Composition:
        raise TypeError("composition must be the canonical Composition type.")
    from ..meshing._model_codec import write_model_artifact

    return write_model_artifact(path, composition, limits=ingress_limits)


def load_composition_archive(
    path: str | Path,
    /,
    *,
    ingress_limits: ArrayArchiveLimits | None = None,
) -> Composition:
    """Restore a composition through its owning model constructors."""
    from ..meshing._model_codec import read_model_artifact, restore_model_archive_value

    return read_model_artifact(
        path,
        expected_type=Composition,
        restore=restore_model_archive_value,
        limits=ingress_limits,
    )
