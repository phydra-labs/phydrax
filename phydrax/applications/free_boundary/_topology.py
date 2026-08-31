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


def free_boundary_topology_plan(
    complex: CellSubcomplex,
    support: CellVertexSupport,
    thresholds: ArrayLike,
    /,
    *,
    region: Literal["inside", "outside"] = "inside",
    coefficients: PrimeField | None = None,
) -> FieldTopologyPlan:
    """Bind signed-field inside/outside topology to an explicit sublevel convention."""
    if region not in ("inside", "outside"):
        raise ValueError("Free-boundary topology region must be inside or outside.")
    return FieldTopologyPlan(
        complex,
        support,
        PrimeField(2) if coefficients is None else coefficients,
        thresholds,
        direction="sublevel" if region == "inside" else "superlevel",
    )


__all__ = ["free_boundary_topology_plan"]
