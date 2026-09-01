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


def diffuse_fracture_topology_plan(
    complex: CellSubcomplex,
    support: CellVertexSupport,
    thresholds: ArrayLike,
    /,
    *,
    region: Literal["damage", "intact"] = "damage",
    coefficients: PrimeField | None = None,
) -> FieldTopologyPlan:
    """Analyze damaged or intact material under an explicit threshold convention."""
    if region not in ("damage", "intact"):
        raise ValueError("Fracture topology region must be damage or intact.")
    return FieldTopologyPlan(
        complex,
        support,
        PrimeField(2) if coefficients is None else coefficients,
        thresholds,
        direction="superlevel" if region == "damage" else "sublevel",
    )


__all__ = ["diffuse_fracture_topology_plan"]
