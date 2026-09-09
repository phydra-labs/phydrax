#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conserved porous states, inspectable root failure, and derivative acceptance."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from ...nonlinear import NonlinearResult


@jax.custom_jvp
def _successful_root_value(value, successful):
    """Keep failed primal candidates inspectable, never differentiate them."""
    return value


@_successful_root_value.defjvp
def _successful_root_value_jvp(primals, tangents):
    value, successful = primals
    tangent, _ = tangents
    tangent = eqx.error_if(
        tangent, ~successful, "Porous step failed; derivatives require a successful root."
    )
    return value, tangent


class PorousFluxes(StrictModule):
    """Owner-oriented face rates and cell-local outward rates (area integrated)."""

    volumetric_face_rates: Array
    mass_face_rates: Array
    local_volumetric_rates: Array
    local_mass_rates: Array


class PorousState(StrictModule):
    """SI cell and shared-face potentials plus conserved cell liquid inventory."""

    pressure_Pa: Array
    face_pressure_Pa: Array
    temperature_K: Array
    face_temperature_K: Array
    water_mass_kg: Array
    water_volume_m3: Array
    time_s: Array

    @property
    def unknown(self):
        return jnp.concatenate((self.pressure_Pa, self.face_pressure_Pa))


class WaterHeatState(StrictModule):
    """Aqueous sensible energy and fixed-skeleton sensible energy in joules."""

    water: PorousState
    energy_J: Array

    @property
    def pressure_Pa(self):
        return self.water.pressure_Pa

    @property
    def face_pressure_Pa(self):
        return self.water.face_pressure_Pa

    @property
    def temperature_K(self):
        return self.water.temperature_K

    @property
    def face_temperature_K(self):
        return self.water.face_temperature_K

    @property
    def water_mass_kg(self):
        return self.water.water_mass_kg

    @property
    def water_volume_m3(self):
        return self.water.water_volume_m3

    @property
    def time_s(self):
        return self.water.time_s

    @property
    def unknown(self):
        return jnp.concatenate(
            (self.water.unknown, self.temperature_K, self.face_temperature_K)
        )


class PorousStepResult(StrictModule):
    """State is committed only on success; candidate and root retain evidence."""

    state: PorousState | WaterHeatState
    candidate: PorousState | WaterHeatState
    fluxes: PorousFluxes
    root: NonlinearResult
    residual: Array
    successful: Array

    @property
    def status(self):
        return self.root.status


__all__ = ["PorousFluxes", "PorousState", "PorousStepResult", "WaterHeatState"]
