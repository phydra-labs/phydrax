#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class WellControl(StrictModule, NonTrainableState):
    mode: Literal["rate", "pressure"] = eqx.field(static=True)
    target: float = eqx.field(static=True)
    minimum_pressure_Pa: float = eqx.field(static=True)
    maximum_pressure_Pa: float = eqx.field(static=True)

    def __init__(
        self,
        mode: Literal["rate", "pressure"],
        target: float,
        /,
        *,
        minimum_pressure_Pa: float = -np.inf,
        maximum_pressure_Pa: float = np.inf,
    ):
        target_, minimum, maximum = (
            float(target),
            float(minimum_pressure_Pa),
            float(maximum_pressure_Pa),
        )
        if (
            mode not in ("rate", "pressure")
            or not np.isfinite(target_)
            or np.isnan(minimum)
            or np.isnan(maximum)
            or minimum >= maximum
        ):
            raise ValueError("Well control mode/target/pressure limits are invalid.")
        if mode == "pressure" and not minimum <= target_ <= maximum:
            raise ValueError("Pressure-control target lies outside pressure limits.")
        self.mode, self.target = mode, target_
        self.minimum_pressure_Pa, self.maximum_pressure_Pa = minimum, maximum


class WellResult(StrictModule):
    bottom_hole_pressure_Pa: Array
    completion_volume_rate_m3_s: Array
    cell_component_source_kg_s: Array
    cell_energy_source_W: Array
    active_mode: Array
    switched: Array
    derivative_available: Array
    successful: Array


class WellCompletionPlan(StrictModule, NonTrainableState):
    """Peaceman/effective-index completions with rate/pressure control switching."""

    cell_indices: Array
    well_indices_m: Array
    phase_composition: Array
    phase_density_kg_m3: Array
    phase_enthalpy_J_kg: Array
    cell_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_indices: ArrayLike,
        well_indices_m: ArrayLike,
        phase_composition: ArrayLike,
        phase_density_kg_m3: ArrayLike,
        phase_enthalpy_J_kg: ArrayLike,
        cell_count: int,
        /,
    ):
        cells = np.asarray(cell_indices)
        indices = np.asarray(well_indices_m, dtype=float)
        composition = np.asarray(phase_composition, dtype=float)
        density = np.asarray(phase_density_kg_m3, dtype=float)
        enthalpy = np.asarray(phase_enthalpy_J_kg, dtype=float)
        count = int(cell_count)
        completions = cells.size
        if (
            cells.ndim != 1
            or completions == 0
            or not np.issubdtype(cells.dtype, np.integer)
            or np.any(cells < 0)
            or np.any(cells >= count)
            or indices.shape != (completions,)
            or np.any(~np.isfinite(indices))
            or np.any(indices <= 0)
            or composition.ndim != 2
            or composition.shape[0] != completions
            or np.any(~np.isfinite(composition))
            or np.any(composition < 0)
            or not np.allclose(np.sum(composition, axis=1), 1.0)
            or density.shape != (completions,)
            or np.any(~np.isfinite(density))
            or np.any(density <= 0)
            or enthalpy.shape != (completions,)
            or np.any(~np.isfinite(enthalpy))
        ):
            raise ValueError("Well completion geometry/fluid properties are invalid.")
        self.cell_indices = jnp.asarray(cells, dtype=jnp.int32)
        self.well_indices_m = jnp.asarray(indices)
        self.phase_composition = jnp.asarray(composition)
        self.phase_density_kg_m3 = jnp.asarray(density)
        self.phase_enthalpy_J_kg = jnp.asarray(enthalpy)
        self.cell_count, self.component_count = count, composition.shape[1]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "well-completion-plan",
                "cells": cells,
                "indices_m": indices,
                "composition": composition,
                "density": density,
                "enthalpy": enthalpy,
                "cell_count": count,
            }
        )

    @classmethod
    def peaceman_isotropic(
        cls,
        cell_indices: ArrayLike,
        permeability_m2: ArrayLike,
        thickness_m: ArrayLike,
        cell_dx_m: ArrayLike,
        cell_dy_m: ArrayLike,
        well_radius_m: ArrayLike,
        phase_composition: ArrayLike,
        phase_density_kg_m3: ArrayLike,
        phase_enthalpy_J_kg: ArrayLike,
        cell_count: int,
        /,
    ) -> WellCompletionPlan:
        permeability, thickness, dx, dy, radius = np.broadcast_arrays(
            np.asarray(permeability_m2, dtype=float),
            np.asarray(thickness_m, dtype=float),
            np.asarray(cell_dx_m, dtype=float),
            np.asarray(cell_dy_m, dtype=float),
            np.asarray(well_radius_m, dtype=float),
        )
        if (
            np.any(permeability <= 0)
            or np.any(thickness <= 0)
            or np.any(dx <= 0)
            or np.any(dy <= 0)
            or np.any(radius <= 0)
        ):
            raise ValueError("Peaceman permeability/geometry must be positive.")
        equivalent = 0.14 * np.sqrt(dx**2 + dy**2)
        if np.any(radius >= equivalent):
            raise ValueError(
                "Well radius must be smaller than Peaceman equivalent radius."
            )
        well_index = 2 * np.pi * permeability * thickness / np.log(equivalent / radius)
        return cls(
            cell_indices,
            well_index,
            phase_composition,
            phase_density_kg_m3,
            phase_enthalpy_J_kg,
            cell_count,
        )

    def evaluate(
        self,
        cell_pressure_Pa: ArrayLike,
        phase_mobility_Pa_s_inverse: ArrayLike,
        control: WellControl,
        /,
    ) -> WellResult:
        pressure = jnp.asarray(cell_pressure_Pa)
        mobility = jnp.asarray(phase_mobility_Pa_s_inverse)
        if pressure.shape != (self.cell_count,) or mobility.shape != (
            self.cell_indices.size,
        ):
            raise ValueError("Well pressure/mobility shapes are invalid.")
        if not isinstance(control, WellControl):
            raise TypeError("Well evaluation requires WellControl.")
        pressure = eqx.error_if(
            pressure,
            jnp.any(~jnp.isfinite(pressure))
            | jnp.any(~jnp.isfinite(mobility))
            | jnp.any(mobility < 0)
            | (jnp.sum(mobility) <= 0),
            "Well pressure and mobility must be finite with positive total mobility.",
        )
        transmissibility = self.well_indices_m * mobility
        total = jnp.sum(transmissibility)
        if control.mode == "rate":
            unconstrained_pressure = (
                control.target + jnp.sum(transmissibility * pressure[self.cell_indices])
            ) / total
            bottom = jnp.minimum(
                jnp.maximum(unconstrained_pressure, control.minimum_pressure_Pa),
                control.maximum_pressure_Pa,
            )
            switched = bottom != unconstrained_pressure
            active_mode = jnp.where(switched, 1, 0)
        else:
            bottom = jnp.asarray(control.target)
            switched = jnp.asarray(False)
            active_mode = jnp.asarray(1)
        rate = transmissibility * (bottom - pressure[self.cell_indices])
        mass = rate * self.phase_density_kg_m3
        component = mass[:, None] * self.phase_composition
        energy = mass * self.phase_enthalpy_J_kg
        cell_component = (
            jnp.zeros((self.cell_count, self.component_count))
            .at[self.cell_indices]
            .add(component)
        )
        cell_energy = jnp.zeros((self.cell_count,)).at[self.cell_indices].add(energy)
        target_residual = jnp.where(active_mode == 0, jnp.sum(rate) - control.target, 0.0)
        successful = (
            jnp.all(jnp.isfinite(rate))
            & jnp.isfinite(bottom)
            & (jnp.abs(target_residual) <= 1e-10 * jnp.maximum(abs(control.target), 1.0))
        )
        return WellResult(
            bottom,
            rate,
            cell_component,
            cell_energy,
            active_mode,
            switched,
            successful & ~switched,
            successful,
        )


__all__ = ["WellCompletionPlan", "WellControl", "WellResult"]
