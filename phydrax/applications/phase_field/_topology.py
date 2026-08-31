#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

from jaxtyping import ArrayLike

from ...topology import (
    CellSubcomplex,
    CellVertexSupport,
    FieldTopologyPlan,
    PrimeField,
)


def phase_field_topology_plan(
    complex: CellSubcomplex,
    support: CellVertexSupport,
    thresholds: ArrayLike,
    /,
    *,
    phase: Literal["occupied", "void"] = "occupied",
    coefficients: PrimeField | None = None,
) -> FieldTopologyPlan:
    """Bind occupied- or void-phase topology to an explicit star convention."""
    if phase not in ("occupied", "void"):
        raise ValueError("Phase topology must analyze occupied or void material.")
    return FieldTopologyPlan(
        complex,
        support,
        PrimeField(2) if coefficients is None else coefficients,
        thresholds,
        direction="superlevel" if phase == "occupied" else "sublevel",
    )


__all__ = ["phase_field_topology_plan"]
