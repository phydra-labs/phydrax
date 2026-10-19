#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Specification-compliance refusals shared by meshing construction phases."""

from __future__ import annotations

from ._failures import MeshingFailure, MeshingFailureCategory
from ._trace import MeshingStageKind


def compliance_failure(
    message: str,
    /,
    *,
    entity_ids: tuple[int, ...] = (),
    requested: tuple[tuple[str, float], ...] = (),
    achieved: tuple[tuple[str, float], ...] = (),
    stage: str = MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
) -> MeshingFailure:
    """Build a compliance refusal without changing its phase or evidence rows."""
    return MeshingFailure(
        MeshingFailureCategory.COMPLIANCE_FAILED,
        message,
        stage=stage,
        entity_ids=entity_ids,
        requested=requested,
        achieved=achieved,
    )
