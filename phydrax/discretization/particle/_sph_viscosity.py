#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class MorrisViscosityPlan(StrictModule, NonTrainableState):
    """Symmetric Morris physical viscosity with constant kinematic viscosity."""

    kinematic_viscosity: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kinematic_viscosity: float,
        /,
        *,
        regularization: float = 0.01,
        plan_id: str | None = None,
    ):
        viscosity = float(kinematic_viscosity)
        epsilon = float(regularization)
        if not np.isfinite(viscosity) or viscosity < 0.0:
            raise ValueError("kinematic_viscosity must be finite and non-negative.")
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("regularization must be finite and positive.")
        generated = canonical_fingerprint(
            {
                "kind": "morris-viscosity-plan",
                "kinematic_viscosity": viscosity,
                "regularization": epsilon,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.kinematic_viscosity = viscosity
        self.regularization = epsilon
        self.plan_id = identifier


__all__ = ["MorrisViscosityPlan"]
