#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class FLIPResourcePolicy(StrictModule, NonTrainableState):
    maximum_state_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_state_bytes: int = 1024**3,
        maximum_workspace_bytes: int = 2 * 1024**3,
    ):
        state = int(maximum_state_bytes)
        workspace = int(maximum_workspace_bytes)
        if state <= 0 or workspace <= 0:
            raise ValueError("FLIP resource limits must be positive.")
        self.maximum_state_bytes = state
        self.maximum_workspace_bytes = workspace
        self.policy_id = canonical_fingerprint(
            {"kind": "flip-resource-policy", "state": state, "workspace": workspace}
        )


class FLIPMethodPlan(StrictModule, NonTrainableState):
    """Fixed-population PIC/FLIP transfer and stability policy."""

    pic_fraction: float = eqx.field(static=True)
    liquid_fraction_threshold: float = eqx.field(static=True)
    extrapolation_layers: int = eqx.field(static=True)
    cfl_fraction: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        pic_fraction: float,
        /,
        *,
        liquid_fraction_threshold: float = 0.05,
        extrapolation_layers: int = 3,
        cfl_fraction: float = 0.5,
    ):
        pic = float(pic_fraction)
        threshold = float(liquid_fraction_threshold)
        layers = int(extrapolation_layers)
        cfl = float(cfl_fraction)
        if not np.isfinite(pic) or not 0.0 <= pic <= 1.0:
            raise ValueError("pic_fraction must lie in [0,1].")
        if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError("liquid_fraction_threshold must lie in (0,1].")
        if layers < 0:
            raise ValueError("extrapolation_layers must be nonnegative.")
        if not np.isfinite(cfl) or cfl <= 0.0:
            raise ValueError("cfl_fraction must be positive and finite.")
        self.pic_fraction = pic
        self.liquid_fraction_threshold = threshold
        self.extrapolation_layers = layers
        self.cfl_fraction = cfl
        self.method_id = canonical_fingerprint(
            {
                "kind": "flip-method",
                "pic_fraction": pic,
                "liquid_fraction_threshold": threshold,
                "extrapolation_layers": layers,
                "cfl_fraction": cfl,
            }
        )


__all__ = ["FLIPMethodPlan", "FLIPResourcePolicy"]
