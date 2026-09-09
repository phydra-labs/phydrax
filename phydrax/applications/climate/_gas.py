#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule


MODEL_YEAR_SECONDS = 365.25 * 86400.0
GAS_NAMES = ("CO2", "CH4", "N2O")


def decay_average(x: ArrayLike, /) -> Array:
    """(1-exp(-x))/x, with its analytic zero limit and finite derivatives."""
    x = jnp.asarray(x)
    small = jnp.abs(x) < 1.0e-3
    safe = jnp.where(small, jnp.ones_like(x), x)
    series = 1.0 + x * (-0.5 + x * (1.0 / 6.0 + x * (-1.0 / 24.0 + x / 120.0)))
    return jnp.where(small, series, -jnp.expm1(-safe) / safe)


class LifetimeResult(StrictModule):
    multiplier: Array
    target: Array
    residual: Array
    bracketed: Array
    successful: Array


class GasBoxResult(StrictModule):
    boxes: Array
    emissions: Array
    sink_increment: Array
    lifetime: LifetimeResult
    successful: Array


class GasBoxModel(StrictModule):
    """Perturbation inventories for CO2, CH4 and N2O, never absolute gas stocks.

    Units by gas: GtC, TgCH4, TgN2O; concentrations: ppm, ppb, ppb.
    Rates are inverse 365.25-day model years. Signed emissions and net sinks
    are supported, provided absolute concentration remains positive.
    """

    fractions: Array
    decay_rates: Array
    background: Array
    inventory_per_concentration: Array
    response_coefficients: Array
    horizon: float = eqx.field(static=True)
    alpha_bounds: tuple[float, float] = eqx.field(static=True)
    solve_iterations: int = eqx.field(static=True)
    solve_tolerance: float = eqx.field(static=True)
    response_active: tuple[bool, ...] = eqx.field(static=True)
    gas_names: tuple[str, ...] = eqx.field(static=True)
    inventory_units: tuple[str, ...] = eqx.field(static=True)
    concentration_units: tuple[str, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        fractions: ArrayLike = (
            (0.2173, 0.2240, 0.2824, 0.2763),
            (1.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
        ),
        decay_rates: ArrayLike = (
            (0.0, 1.0 / 394.4, 1.0 / 36.54, 1.0 / 4.304),
            (1.0 / 9.3, 0.0, 0.0, 0.0),
            (1.0 / 121.0, 0.0, 0.0, 0.0),
        ),
        /,
        *,
        background: ArrayLike = (278.3, 729.2, 270.1),
        inventory_per_concentration: ArrayLike = (2.124, 2.78, 7.8),
        response_coefficients: ArrayLike = (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        horizon: float = 100.0,
        alpha_bounds: tuple[float, float] = (1.0e-4, 1.0e4),
        solve_iterations: int = 64,
        solve_tolerance: float = 1.0e-6,
    ):
        fractions_ = np.asarray(fractions, dtype=float)
        rates = np.asarray(decay_rates, dtype=float)
        background_ = np.asarray(background, dtype=float)
        conversion = np.asarray(inventory_per_concentration, dtype=float)
        response = np.asarray(response_coefficients, dtype=float)
        if (
            fractions_.ndim != 2
            or fractions_.shape[0] != 3
            or fractions_.shape[1] < 1
            or rates.shape != fractions_.shape
        ):
            raise ValueError(
                "Gas fractions and rates require matching (3, boxes) shapes."
            )
        if not all(
            np.all(np.isfinite(x))
            for x in (fractions_, rates, background_, conversion, response)
        ):
            raise ValueError("Gas model coefficients must be finite.")
        if (
            np.any(fractions_ < 0)
            or not np.allclose(fractions_.sum(axis=-1), 1.0, atol=1.0e-12, rtol=1.0e-12)
            or np.any(rates < 0)
        ):
            raise ValueError(
                "Fractions must be nonnegative and sum to one; decay rates must be nonnegative."
            )
        if (
            background_.shape != (3,)
            or conversion.shape != (3,)
            or np.any(background_ <= 0)
            or np.any(conversion <= 0)
        ):
            raise ValueError(
                "Background concentrations and inventory conversions require three positive entries."
            )
        if response.shape != (3, 3):
            raise ValueError(
                "Lifetime response columns are cumulative sink, temperature, airborne inventory."
            )
        lo, hi = map(float, alpha_bounds)
        if (
            not np.isfinite(horizon)
            or horizon <= 0
            or not 0 < lo < hi
            or not np.isfinite(hi)
        ):
            raise ValueError(
                "Lifetime horizon and finite positive ordered alpha bounds are required."
            )
        if (
            not lo <= 1.0 <= hi
            or solve_iterations < 1
            or not np.isfinite(solve_tolerance)
            or solve_tolerance <= 0
        ):
            raise ValueError(
                "Lifetime bounds must contain one; solve controls must be positive."
            )
        active = np.any(response != 0.0, axis=-1)
        if np.any(active & ~np.any((rates > 0.0) & (fractions_ > 0.0), axis=-1)):
            raise ValueError("State-dependent lifetime requires a decaying reservoir.")
        self.fractions = jnp.asarray(fractions_)
        self.decay_rates = jnp.asarray(rates)
        self.background = jnp.asarray(background_)
        self.inventory_per_concentration = jnp.asarray(conversion)
        self.response_coefficients = jnp.asarray(response)
        self.horizon = float(horizon)
        self.alpha_bounds = (lo, hi)
        self.solve_iterations = int(solve_iterations)
        self.solve_tolerance = float(solve_tolerance)
        self.response_active = tuple(bool(value) for value in active)
        self.gas_names = GAS_NAMES
        self.inventory_units = ("GtC", "TgCH4", "TgN2O")
        self.concentration_units = ("ppm", "ppb", "ppb")
        self.model_id = canonical_fingerprint(
            {
                "kind": "reduced-gas-box-model",
                "gases": self.gas_names,
                "inventory_units": self.inventory_units,
                "concentration_units": self.concentration_units,
                "coefficients": array_tree_fingerprint(
                    (
                        self.fractions,
                        self.decay_rates,
                        self.background,
                        self.inventory_per_concentration,
                        self.response_coefficients,
                    )
                ),
                "horizon": self.horizon,
                "alpha_bounds": self.alpha_bounds,
                "solve_iterations": self.solve_iterations,
                "solve_tolerance": self.solve_tolerance,
                "year_seconds": MODEL_YEAR_SECONDS,
            }
        )

    def concentration(self, boxes: Array, /) -> Array:
        return (
            self.background + jnp.sum(boxes, axis=-1) / self.inventory_per_concentration
        )

    def integrated_response(self, multiplier: Array, /) -> Array:
        return jnp.sum(
            self.fractions
            * self.horizon
            * decay_average(self.horizon * self.decay_rates / multiplier[..., None]),
            axis=-1,
        )

    def lifetime(
        self,
        boxes: Array,
        cumulative_sink: Array,
        surface_temperature: Array,
        /,
        *,
        active_gases: tuple[bool, ...] = (True, True, True),
    ) -> LifetimeResult:
        one = jnp.ones_like(self.background)
        base = self.integrated_response(one)
        zero = jnp.zeros_like(base)
        response_active = tuple(
            enabled and selected
            for enabled, selected in zip(self.response_active, active_gases, strict=True)
        )
        if not any(response_active):
            return LifetimeResult(
                one, base, zero, jnp.ones_like(base, dtype=bool), jnp.asarray(True)
            )
        target = (
            base
            + self.response_coefficients[:, 0] * cumulative_sink
            + self.response_coefficients[:, 1] * surface_temperature
            + self.response_coefficients[:, 2] * jnp.sum(boxes, axis=-1)
        )
        lower = jnp.full_like(base, self.alpha_bounds[0])
        upper = jnp.full_like(base, self.alpha_bounds[1])
        bracketed = (
            (target >= self.integrated_response(lower))
            & (target <= self.integrated_response(upper))
            & jnp.isfinite(target)
        )

        def bisect(_, bracket):
            lo, hi = bracket
            mid = jnp.sqrt(lo * hi)
            below = self.integrated_response(mid) < target
            return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

        lower, upper = jax.lax.fori_loop(0, self.solve_iterations, bisect, (lower, upper))
        root = jax.lax.stop_gradient(jnp.sqrt(lower * upper))
        value, slope = jax.jvp(self.integrated_response, (root,), (one,))
        # Exact implicit derivative on a certified smooth root, not the derivative
        # of bisection's discrete comparisons. Forward value remains the root.
        difference = target - value
        safe_slope = jax.lax.stop_gradient(jnp.where(slope > 0.0, slope, one))
        root = root + (difference - jax.lax.stop_gradient(difference)) / safe_slope
        active = jnp.asarray(response_active)
        multiplier = jnp.where(active, root, one)
        residual = jnp.where(
            active, jnp.abs(self.integrated_response(multiplier) - target), zero
        )
        bracketed = ~active | bracketed
        certified = (
            bracketed
            & (residual <= self.solve_tolerance * jnp.maximum(1.0, jnp.abs(target)))
            & (~active | (slope > 0.0))
        )
        return LifetimeResult(multiplier, target, residual, bracketed, jnp.all(certified))

    def advance(
        self,
        boxes: Array,
        cumulative_sink: Array,
        surface_temperature: Array,
        duration: Array,
        emissions: Array,
        concentrations: Array,
        roles: tuple[str, ...],
        /,
    ) -> GasBoxResult:
        """Exact constant-source reservoir map with coefficients frozen at entry.

        Concentration drivers prescribe the endpoint and infer the unique
        constant interval emission, using exactly the same frozen lifetime.
        Forcing-driven gases keep their inventories unchanged.
        """
        if (
            boxes.shape != self.fractions.shape
            or cumulative_sink.shape != (3,)
            or emissions.shape != (3,)
            or concentrations.shape != (3,)
        ):
            raise ValueError(
                "Gas state and drivers must match the three-gas reservoir layout."
            )
        if len(roles) != 3 or any(
            role not in ("emissions", "concentration", "forcing") for role in roles
        ):
            raise ValueError("Each gas requires exactly one supported driver role.")
        concentration_driven = jnp.asarray(
            tuple(role == "concentration" for role in roles)
        )
        emissions_driven = jnp.asarray(tuple(role == "emissions" for role in roles))
        forcing_driven = jnp.asarray(tuple(role == "forcing" for role in roles))
        # Mask unused inputs before arithmetic: masking an inverse containing
        # NaN only at its output still poisons reverse-mode parameter gradients.
        concentrations = jnp.where(concentration_driven, concentrations, self.background)
        emissions = jnp.where(emissions_driven, emissions, 0.0)
        lifetime = self.lifetime(
            boxes,
            cumulative_sink,
            surface_temperature,
            active_gases=tuple(role != "forcing" for role in roles),
        )
        x = self.decay_rates * duration / lifetime.multiplier[:, None]
        decayed = jnp.exp(-x) * boxes
        gain = self.fractions * duration * decay_average(x)
        target_inventory = (
            concentrations - self.background
        ) * self.inventory_per_concentration
        gain_sum = jnp.sum(gain, axis=-1)
        inverse = (target_inventory - jnp.sum(decayed, axis=-1)) / jnp.where(
            gain_sum > 0.0, gain_sum, 1.0
        )
        inferred = jnp.where(concentration_driven, inverse, emissions)
        inferred = jnp.where(forcing_driven, 0.0, inferred)
        candidate = decayed + gain * inferred[:, None]
        candidate = jnp.where(forcing_driven[:, None], boxes, candidate)
        # Independently integrate physical decay losses, rather than defining
        # sinks as the budget remainder. The source-loss series avoids losing
        # a tiny but nonzero sink to cancellation at nearly immortal rates.
        loss_average = jnp.where(
            jnp.abs(x) < 1.0e-3,
            x * (0.5 + x * (-1.0 / 6.0 + x * (1.0 / 24.0 - x / 120.0))),
            1.0 - decay_average(x),
        )
        sink = jnp.sum(
            -jnp.expm1(-x) * boxes
            + self.fractions * duration * inferred[:, None] * loss_average,
            axis=-1,
        )
        sink = jnp.where(forcing_driven, 0.0, sink)
        valid = (
            (duration > 0.0)
            & jnp.isfinite(duration)
            & jnp.all(jnp.isfinite(candidate))
            & jnp.all(jnp.isfinite(inferred))
            & jnp.all(jnp.isfinite(sink))
            & jnp.all(self.concentration(candidate) > 0.0)
        )
        successful = valid & lifetime.successful
        return GasBoxResult(
            jnp.where(successful, candidate, boxes),
            jnp.where(successful, inferred, 0.0),
            jnp.where(successful, sink, 0.0),
            lifetime,
            successful,
        )


__all__ = [
    "GAS_NAMES",
    "MODEL_YEAR_SECONDS",
    "GasBoxModel",
    "GasBoxResult",
    "LifetimeResult",
    "decay_average",
]
