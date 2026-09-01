#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._mac_interface_state import MACFreeSurfaceGeometryState


class MACCapillaryResult(StrictModule):
    pressure_jump: jnp.ndarray
    surface_area: jnp.ndarray
    surface_energy: jnp.ndarray
    maximum_curvature: jnp.ndarray
    finite: jnp.ndarray
    successful: jnp.ndarray
    plan_id: str = eqx.field(static=True)


class MACGhostFluidCapillaryPlan(StrictModule, NonTrainableState):
    surface_tension: float = eqx.field(static=True)
    interface_width: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, surface_tension: float, /, *, interface_width: float):
        tension = float(surface_tension)
        width = float(interface_width)
        if (
            not np.isfinite(tension)
            or tension < 0.0
            or not np.isfinite(width)
            or width <= 0.0
        ):
            raise ValueError("Surface-tension policy is invalid.")
        self.surface_tension = tension
        self.interface_width = width
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-ghost-fluid-capillary",
                "surface_tension": tension,
                "interface_width": width,
            }
        )

    def evaluate(self, geometry: MACFreeSurfaceGeometryState, /) -> MACCapillaryResult:
        phi = geometry.signed_distance
        width = self.interface_width
        inside = jnp.abs(phi) <= width
        delta = jnp.where(
            inside,
            (1.0 + jnp.cos(jnp.pi * phi / width)) / (2.0 * width),
            0.0,
        )
        pressure_jump = self.surface_tension * geometry.curvature
        surface_area = jnp.sum(delta)
        surface_energy = self.surface_tension * surface_area
        maximum = jnp.max(
            jnp.abs(jnp.where(geometry.valid_band, geometry.curvature, 0.0)), initial=0.0
        )
        finite = (
            geometry.finite
            & jnp.all(jnp.isfinite(pressure_jump))
            & jnp.isfinite(surface_area + surface_energy + maximum)
        )
        return MACCapillaryResult(
            pressure_jump,
            surface_area,
            surface_energy,
            maximum,
            finite,
            geometry.successful & finite,
            self.plan_id,
        )


__all__ = ["MACCapillaryResult", "MACGhostFluidCapillaryPlan"]
