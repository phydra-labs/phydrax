#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class PanelCompressibilityPolicy(StrictModule, NonTrainableState):
    """Explicit subsonic pressure correction for native potential-flow panels."""

    kind: Literal["incompressible", "prandtl-glauert", "karman-tsien"] = eqx.field(
        static=True
    )
    mach_number: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: Literal[
            "incompressible", "prandtl-glauert", "karman-tsien"
        ] = "incompressible",
        mach_number: float = 0.0,
        /,
    ):
        mach = float(mach_number)
        if (
            kind not in ("incompressible", "prandtl-glauert", "karman-tsien")
            or not np.isfinite(mach)
            or mach < 0.0
            or (kind == "incompressible" and mach != 0.0)
            or (kind != "incompressible" and not 0.0 < mach < 1.0)
        ):
            raise ValueError(
                "Panel compressibility must be incompressible or strictly subsonic."
            )
        beta = 1.0 if kind == "incompressible" else float(np.sqrt(1.0 - mach**2))
        self.kind = kind
        self.mach_number = mach
        self.beta = beta
        self.policy_id = canonical_fingerprint(
            {
                "kind": "panel-pressure-compressibility",
                "correction": kind,
                "mach_number": mach,
                "beta": beta,
                "scope": "pressure-postprocessing-only",
            }
        )

    def correct_pressure_coefficient(
        self, incompressible_pressure_coefficient: ArrayLike, /
    ) -> Array:
        pressure = jnp.asarray(incompressible_pressure_coefficient)
        if self.kind == "incompressible":
            return pressure
        if self.kind == "prandtl-glauert":
            return pressure / self.beta
        denominator = self.beta + (
            self.mach_number**2 * pressure / (2.0 * (1.0 + self.beta))
        )
        denominator = eqx.error_if(
            denominator,
            jnp.any(~jnp.isfinite(denominator) | (denominator <= 0.0)),
            "Karman-Tsien correction left its admissible pressure range.",
        )
        return pressure / denominator


__all__ = ["PanelCompressibilityPolicy"]
