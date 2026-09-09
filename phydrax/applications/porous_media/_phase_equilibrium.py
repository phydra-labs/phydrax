#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class PhaseEquilibriumResult(StrictModule):
    vapor_fraction: Array
    liquid_composition: Array
    vapor_composition: Array
    equilibrium_residual: Array
    liquid_margin: Array
    vapor_margin: Array
    phase_state: Array
    successful: Array
    derivative_available: Array


class RachfordRiceFlashPlan(StrictModule, NonTrainableState):
    component_names: tuple[str, ...] = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        component_names: tuple[str, ...],
        /,
        *,
        maximum_iterations: int = 80,
        tolerance: float = 1e-12,
    ):
        names = tuple(str(value).strip() for value in component_names)
        steps, tolerance_ = int(maximum_iterations), float(tolerance)
        if (
            len(names) < 2
            or any(not value for value in names)
            or len(set(names)) != len(names)
            or steps <= 0
            or not jnp.isfinite(tolerance_)
            or tolerance_ <= 0
        ):
            raise ValueError(
                "Flash components, iteration count, or tolerance are invalid."
            )
        self.component_names, self.maximum_iterations, self.tolerance = (
            names,
            steps,
            tolerance_,
        )

    def solve(
        self, overall_composition: ArrayLike, equilibrium_ratios: ArrayLike, /
    ) -> PhaseEquilibriumResult:
        composition = jnp.asarray(overall_composition)
        ratios = jnp.asarray(equilibrium_ratios)
        expected = (len(self.component_names),)
        if composition.shape != expected or ratios.shape != expected:
            raise ValueError("Flash composition/K-value shapes disagree with components.")
        composition = eqx.error_if(
            composition,
            jnp.any(~jnp.isfinite(composition))
            | jnp.any(composition < 0)
            | (jnp.abs(jnp.sum(composition) - 1.0) > 1e-10)
            | jnp.any(~jnp.isfinite(ratios))
            | jnp.any(ratios <= 0),
            "Flash composition must be normalized and K values positive finite.",
        )

        def residual(fraction):
            return jnp.sum(
                composition * (ratios - 1.0) / (1.0 + fraction * (ratios - 1.0))
            )

        at_liquid = residual(jnp.asarray(0.0))
        at_vapor = residual(jnp.asarray(1.0))
        liquid_only = at_liquid <= 0
        vapor_only = at_vapor >= 0
        left, right = jnp.asarray(0.0), jnp.asarray(1.0)
        for _ in range(self.maximum_iterations):
            middle = 0.5 * (left + right)
            value = residual(middle)
            left, right = (
                jnp.where(value > 0, middle, left),
                jnp.where(value > 0, right, middle),
            )
        interior = 0.5 * (left + right)
        fraction = jnp.where(liquid_only, 0.0, jnp.where(vapor_only, 1.0, interior))
        denominator = 1.0 + fraction * (ratios - 1.0)
        liquid = composition / denominator
        vapor = ratios * liquid
        liquid = liquid / jnp.sum(liquid)
        vapor = vapor / jnp.sum(vapor)
        equilibrium = residual(fraction)
        phase_state = jnp.where(liquid_only, 0, jnp.where(vapor_only, 2, 1)).astype(
            jnp.int32
        )
        successful = (
            jnp.all(jnp.isfinite(liquid))
            & jnp.all(jnp.isfinite(vapor))
            & (liquid_only | vapor_only | (jnp.abs(equilibrium) <= self.tolerance))
        )
        margin = jnp.minimum(jnp.abs(at_liquid), jnp.abs(at_vapor))
        derivative_available = (
            successful & ~liquid_only & ~vapor_only & (margin > 10 * self.tolerance)
        )
        return PhaseEquilibriumResult(
            fraction,
            liquid,
            vapor,
            equilibrium,
            -at_liquid,
            at_vapor,
            phase_state,
            successful,
            derivative_available,
        )


class CompositionalFlashPlan(StrictModule, NonTrainableState):
    flash: RachfordRiceFlashPlan
    equilibrium_ratio_model: Callable[[Array, Array, Array], Array] = eqx.field(
        static=True
    )
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        flash: RachfordRiceFlashPlan,
        equilibrium_ratio_model: Callable[[Array, Array, Array], Array],
        /,
        *,
        model_id: str,
    ):
        if not isinstance(flash, RachfordRiceFlashPlan) or not callable(
            equilibrium_ratio_model
        ):
            raise TypeError(
                "Compositional flash requires Rachford-Rice and K-value model."
            )
        identifier = str(model_id).strip()
        if not identifier:
            raise ValueError("Equilibrium-ratio model identity is required.")
        self.flash, self.equilibrium_ratio_model = flash, equilibrium_ratio_model
        self.model_id = identifier

    def solve(
        self,
        pressure_Pa: ArrayLike,
        temperature_K: ArrayLike,
        overall_composition: ArrayLike,
        /,
    ) -> PhaseEquilibriumResult:
        pressure, temperature, composition = (
            jnp.asarray(pressure_Pa),
            jnp.asarray(temperature_K),
            jnp.asarray(overall_composition),
        )
        if pressure.shape != () or temperature.shape != ():
            raise ValueError(
                "Local compositional flash pressure/temperature must be scalar."
            )
        pressure = eqx.error_if(
            pressure,
            ~jnp.isfinite(pressure)
            | (pressure <= 0)
            | ~jnp.isfinite(temperature)
            | (temperature <= 0),
            "Flash pressure and temperature must be finite and positive.",
        )
        ratios = self.equilibrium_ratio_model(pressure, temperature, composition)
        return self.flash.solve(composition, ratios)


__all__ = [
    "CompositionalFlashPlan",
    "PhaseEquilibriumResult",
    "RachfordRiceFlashPlan",
]
