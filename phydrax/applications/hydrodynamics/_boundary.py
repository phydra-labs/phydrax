#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver._mac_ale import MACALEStageGeometry
from ._free_surface_ale import FaceTuple, PreparedGraphSurfaceALE


class FreeSurfaceBoundaryStage(StrictModule):
    free_velocity_mask: FaceTuple
    prescribed_velocity: FaceTuple
    surface_pressure_head: Array
    gas_pressure_anomaly_head: Array
    boundary_volume_flux: Array
    boundary_power: Array
    finite: Array
    valid: Array
    layout_id: str = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)


class FreeSurfaceBoundaryPlan(StrictModule, NonTrainableState):
    """Single owner of graph ALE velocity and pressure boundary constraints."""

    gas_pressure: float = eqx.field(static=True)
    reference_pressure: float = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        gas_pressure: float = 101_325.0,
        reference_pressure: float = 101_325.0,
    ):
        gas = float(gas_pressure)
        reference = float(reference_pressure)
        if not np.isfinite(gas) or not np.isfinite(reference):
            raise ValueError("Gas and reference pressures must be finite.")
        self.gas_pressure = gas
        self.reference_pressure = reference
        self.layout_id = canonical_fingerprint(
            {
                "kind": "free-surface-boundary-plan-v2",
                "gas_pressure": gas,
                "reference_pressure": reference,
                "lateral": "periodic-or-closed",
                "bottom": "impermeable",
                "top": "pressure-dirichlet",
            }
        )

    def stage(
        self,
        surface: PreparedGraphSurfaceALE,
        geometry: MACALEStageGeometry,
        eta: Array,
        /,
        *,
        gravity: float,
        density: float,
        capillary_head: Array | None = None,
        wave_pressure_head: Array | None = None,
        prescribed_velocity: FaceTuple | None = None,
        stage_tag: Any = None,
    ) -> FreeSurfaceBoundaryStage:
        if not isinstance(surface, PreparedGraphSurfaceALE):
            raise TypeError("surface must be PreparedGraphSurfaceALE.")
        eta_ = jnp.asarray(eta, dtype=geometry.cell_volumes.dtype)
        if eta_.shape != surface.eta_shape:
            raise ValueError("Boundary eta shape is invalid.")
        masks = [jnp.ones_like(value) for value in geometry.face_measures]
        axes = surface.plan.reference.grid.structured_axes
        for axis, grid_axis in enumerate(axes):
            if grid_axis.periodic:
                continue
            lower = [slice(None)] * masks[axis].ndim
            upper = [slice(None)] * masks[axis].ndim
            lower[axis] = 0
            upper[axis] = masks[axis].shape[axis] - 1
            masks[axis] = masks[axis].at[tuple(lower)].set(0.0)
            masks[axis] = masks[axis].at[tuple(upper)].set(0.0)
        top = [slice(None)] * masks[2].ndim
        top[2] = masks[2].shape[2] - 1
        masks[2] = masks[2].at[tuple(top)].set(1.0)
        velocities = (
            tuple(jnp.zeros_like(value) for value in geometry.face_measures)
            if prescribed_velocity is None
            else geometry.validate_velocity(prescribed_velocity)
        )
        capillary = (
            jnp.zeros_like(eta_)
            if capillary_head is None
            else jnp.asarray(capillary_head, dtype=eta_.dtype)
        )
        wave = (
            jnp.zeros_like(eta_)
            if wave_pressure_head is None
            else jnp.asarray(wave_pressure_head, dtype=eta_.dtype)
        )
        anomaly = jnp.asarray(
            (self.gas_pressure - self.reference_pressure) / density,
            dtype=eta_.dtype,
        )
        pressure_head = gravity * eta_ + anomaly + capillary + wave
        finite = jnp.all(jnp.isfinite(pressure_head)) & jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in velocities))
        )
        stage_id = canonical_fingerprint(
            {
                "kind": "free-surface-boundary-stage-v2",
                "layout": self.layout_id,
                "surface": surface.surface_id,
                "stage_tag": str(stage_tag),
            }
        )
        return FreeSurfaceBoundaryStage(
            free_velocity_mask=tuple(masks),
            prescribed_velocity=velocities,
            surface_pressure_head=pressure_head,
            gas_pressure_anomaly_head=jnp.broadcast_to(anomaly, eta_.shape),
            boundary_volume_flux=jnp.asarray(0.0, dtype=eta_.dtype),
            boundary_power=jnp.asarray(0.0, dtype=eta_.dtype),
            finite=finite,
            valid=finite & geometry.passed,
            layout_id=self.layout_id,
            stage_id=stage_id,
        )


__all__ = ["FreeSurfaceBoundaryPlan", "FreeSurfaceBoundaryStage"]
