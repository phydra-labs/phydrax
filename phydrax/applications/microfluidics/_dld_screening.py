#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._admissibility import AdmissibilityHeader, AdmissibilityReason
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._dld_geometry import DLDDesign, DLDTopology


class DLDEmpiricalScreenResult(StrictModule):
    row_shift_fraction: Array
    gap: Array
    critical_diameter: Array
    header: AdmissibilityHeader
    plan_id: str = eqx.field(static=True)


class DLDEmpiricalScreenPlan(StrictModule, NonTrainableState):
    coefficient: float = eqx.field(static=True)
    exponent: float = eqx.field(static=True)
    minimum_shift_fraction: float = eqx.field(static=True)
    maximum_shift_fraction: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        coefficient: float,
        exponent: float,
        minimum_shift_fraction: float,
        maximum_shift_fraction: float,
        source_id: str,
    ) -> None:
        values = tuple(
            float(value)
            for value in (
                coefficient,
                exponent,
                minimum_shift_fraction,
                maximum_shift_fraction,
            )
        )
        source = str(source_id)
        if (
            any(not np.isfinite(value) for value in values)
            or values[0] <= 0.0
            or values[1] <= 0.0
            or not 0.0 < values[2] < values[3] < 1.0
            or not source
        ):
            raise ValueError("DLD empirical screening coefficients are invalid.")
        self.coefficient, self.exponent = values[:2]
        self.minimum_shift_fraction, self.maximum_shift_fraction = values[2:]
        self.source_id = source
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dld-empirical-screen",
                "coefficient": values[0],
                "exponent": values[1],
                "support": values[2:],
                "source": source,
            }
        )

    def evaluate(
        self, topology: DLDTopology, design: DLDDesign, /
    ) -> DLDEmpiricalScreenResult:
        if not isinstance(topology, DLDTopology) or not isinstance(design, DLDDesign):
            raise TypeError("DLD screen requires topology and design.")
        shift_fraction = jnp.asarray(design.row_shift / design.lateral_pitch)
        gap = jnp.asarray(design.lateral_pitch - 2.0 * design.post_radius)
        critical = self.coefficient * gap * shift_fraction**self.exponent
        supported = (
            (shift_fraction >= self.minimum_shift_fraction)
            & (shift_fraction <= self.maximum_shift_fraction)
            & (gap > 0.0)
            & jnp.isfinite(critical)
        )
        reasons = jnp.where(
            supported,
            jnp.asarray(0, dtype=jnp.uint32),
            jnp.asarray(int(AdmissibilityReason.OUTSIDE_SUPPORT), dtype=jnp.uint32),
        )
        margin = jnp.minimum(
            shift_fraction - self.minimum_shift_fraction,
            self.maximum_shift_fraction - shift_fraction,
        )
        header = AdmissibilityHeader(
            jnp.where(supported, margin, jnp.minimum(margin, -1.0)),
            reasons,
            self.plan_id,
            canonical_fingerprint({"kind": "dld-screen-evidence", "plan": self.plan_id}),
        )
        return DLDEmpiricalScreenResult(
            shift_fraction,
            gap,
            critical,
            header,
            self.plan_id,
        )


__all__ = ["DLDEmpiricalScreenPlan", "DLDEmpiricalScreenResult"]
