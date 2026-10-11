#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Declared irradiation and coolant histories of one rod.

Power, fast flux, and coolant conditions are piecewise linear in time between
declared breakpoints. Step integrals of linear power are exact (piecewise
quadratic cumulative energy), so burnup and deposited energy never depend on
the time-step sequence. Breakpoints are published as an exact-time schedule
that the accepted-step runtime lands on.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import linear_interpolate
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver import ExactTimeSchedule
from ...typing import ConvertibleToArray


def require_increasing_from_zero(values: NDArray[np.float64], name: str, /) -> None:
    """Refuse ``values`` unless finite, rank 1, at least two long, from 0 increasing."""
    if (
        values.ndim != 1
        or values.size < 2
        or values[0] != 0.0
        or np.any(np.diff(values) <= 0.0)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(f"{name} must start at 0 and increase strictly.")


def _cumulative_trapezoid(
    times: NDArray[np.float64], series: NDArray[np.float64], /
) -> NDArray[np.float64]:
    """Exact running integral of a piecewise-linear ``series (times, zones)``."""

    increments = np.diff(times)[:, None] * 0.5 * (series[1:] + series[:-1])
    return np.concatenate(
        (
            np.zeros((1, series.shape[1]), dtype=np.float64),
            np.cumsum(increments, axis=0),
        ),
        axis=0,
    )


@final
class IrradiationConditions(StrictModule):
    """History values at one time.

    ``linear_power`` [W/m] and ``fast_flux`` [n/(m^2 s)] hold one value per
    axial zone; the coolant inlet temperature [K], pressure [Pa], and mass flux
    [kg/(m^2 s)] are rod-wide.
    """

    linear_power: Array
    fast_flux: Array
    coolant_inlet_temperature: Array
    coolant_pressure: Array
    coolant_mass_flux: Array


@final
class IrradiationHistory(StrictModule, NonTrainableState):
    """Piecewise-linear rod history on axial zones.

    ``zone_edges`` [m] partition the fuel stack from its bottom (axial zone
    ``k`` spans ``zone_edges[k]..zone_edges[k+1]``). ``linear_power`` [W/m] is
    the total recoverable fission power per unit reference stack length of each
    zone; ``fast_flux`` [n/(m^2 s), E > 1 MeV] is the zone's fast flux. Coolant
    inlet temperature [K], pressure [Pa], and mass flux [kg/(m^2 s)] are
    rod-wide. The mass flux may be zero, for histories whose coolant boundary
    does not use the flow.
    """

    times: Array
    zone_edges: Array
    linear_power: Array
    fast_flux: Array
    coolant_inlet_temperature: Array
    coolant_pressure: Array
    coolant_mass_flux: Array
    cumulative_energy: Array
    cumulative_fast_fluence: Array
    history_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        times: ConvertibleToArray,
        zone_edges: ConvertibleToArray,
        linear_power: ConvertibleToArray,
        fast_flux: ConvertibleToArray,
        coolant_inlet_temperature: ConvertibleToArray,
        coolant_pressure: ConvertibleToArray,
        coolant_mass_flux: ConvertibleToArray,
    ) -> None:
        times_ = np.asarray(times, dtype=np.float64)
        edges = np.asarray(zone_edges, dtype=np.float64)
        power = np.asarray(linear_power, dtype=np.float64)
        flux = np.asarray(fast_flux, dtype=np.float64)
        inlet = np.asarray(coolant_inlet_temperature, dtype=np.float64)
        pressure = np.asarray(coolant_pressure, dtype=np.float64)
        mass_flux = np.asarray(coolant_mass_flux, dtype=np.float64)
        require_increasing_from_zero(times_, "times")
        require_increasing_from_zero(edges, "zone_edges")
        shape = (times_.size, edges.size - 1)
        if power.shape != shape or flux.shape != shape:
            raise ValueError(
                f"linear_power and fast_flux must have shape (times, zones) = {shape}."
            )
        if not np.all(np.isfinite(power)) or np.any(power < 0.0):
            raise ValueError("linear_power must be finite and nonnegative.")
        if not np.all(np.isfinite(flux)) or np.any(flux < 0.0):
            raise ValueError("fast_flux must be finite and nonnegative.")
        for name, series, zero_allowed in (
            ("coolant_inlet_temperature", inlet, False),
            ("coolant_pressure", pressure, False),
            ("coolant_mass_flux", mass_flux, True),
        ):
            if series.shape != times_.shape or not np.all(np.isfinite(series)):
                raise ValueError(f"{name} must be finite with one value per time.")
            if np.any(series < 0.0) or (not zero_allowed and np.any(series == 0.0)):
                raise ValueError(f"{name} has non-physical values.")
        self.times = jnp.asarray(times_)
        self.zone_edges = jnp.asarray(edges)
        self.linear_power = jnp.asarray(power)
        self.fast_flux = jnp.asarray(flux)
        self.coolant_inlet_temperature = jnp.asarray(inlet)
        self.coolant_pressure = jnp.asarray(pressure)
        self.coolant_mass_flux = jnp.asarray(mass_flux)
        self.cumulative_energy = jnp.asarray(_cumulative_trapezoid(times_, power))
        self.cumulative_fast_fluence = jnp.asarray(_cumulative_trapezoid(times_, flux))
        self.history_id = canonical_fingerprint(
            {
                "kind": "fuel-irradiation-history",
                "arrays": array_tree_fingerprint(
                    (times_, edges, power, flux, inlet, pressure, mass_flux)
                ),
            }
        )

    @property
    def zone_count(self) -> int:
        return self.zone_edges.shape[0] - 1

    @property
    def zone_lengths(self) -> Array:
        return jnp.diff(self.zone_edges)

    @property
    def end_time(self) -> float:
        return float(np.asarray(self.times)[-1])

    def schedule(self, /) -> ExactTimeSchedule:
        """Exact-time schedule of every breakpoint after the start."""

        return ExactTimeSchedule(np.asarray(self.times)[1:])

    def conditions(self, time: ArrayLike, /) -> IrradiationConditions:
        """Interpolated values at one finite time within the declared history."""

        query = self._checked_time(time)

        def interpolate(series: Array) -> Array:
            return linear_interpolate(self.times, series, query, bounds="error").values

        return IrradiationConditions(
            linear_power=interpolate(self.linear_power),
            fast_flux=interpolate(self.fast_flux),
            coolant_inlet_temperature=interpolate(self.coolant_inlet_temperature),
            coolant_pressure=interpolate(self.coolant_pressure),
            coolant_mass_flux=interpolate(self.coolant_mass_flux),
        )

    def energy_per_length(self, time: ArrayLike, /) -> Array:
        """Exact cumulative recoverable fission energy [J/m] per axial zone."""

        return self._integral(self.linear_power, self.cumulative_energy, time)

    def step_average_linear_power(self, start: ArrayLike, end: ArrayLike, /) -> Array:
        """Exact mean linear power [W/m] per zone over ``[start, end]``."""

        return self._interval_average(
            self.linear_power, self.cumulative_energy, start, end
        )

    def fast_fluence(self, time: ArrayLike, /) -> Array:
        """Exact cumulative fast fluence [n/m^2, E > 1 MeV] per axial zone."""

        return self._integral(self.fast_flux, self.cumulative_fast_fluence, time)

    def step_average_fast_flux(self, start: ArrayLike, end: ArrayLike, /) -> Array:
        """Exact mean fast flux [n/(m^2 s)] per zone over ``[start, end]``."""

        return self._interval_average(
            self.fast_flux, self.cumulative_fast_fluence, start, end
        )

    def _interval_average(
        self,
        series: Array,
        cumulative: Array,
        start: ArrayLike,
        end: ArrayLike,
        /,
    ) -> Array:
        """Exact mean of the piecewise-linear ``series`` over ``[start, end]``."""

        start_ = self._checked_time(start)
        end_ = self._checked_time(end)
        duration = eqx.error_if(
            end_ - start_,
            end_ <= start_,
            "A history integration interval must have strictly increasing endpoints.",
        )
        return (
            self._integral(series, cumulative, end_)
            - self._integral(series, cumulative, start_)
        ) / duration

    def _integral(self, values: Array, prefix: Array, time: ArrayLike, /) -> Array:
        """Exact running integral of ``values`` at ``time``.

        The stored breakpoint ``prefix`` plus the quadratic primitive of the
        linear segment containing ``time``.
        """

        query = self._checked_time(time)
        index = jnp.clip(
            jnp.searchsorted(self.times, query, side="right") - 1,
            0,
            self.times.shape[0] - 2,
        )
        span = self.times[index + 1] - self.times[index]
        offset = query - self.times[index]
        low = values[index]
        slope = (values[index + 1] - low) / span
        return prefix[index] + low * offset + 0.5 * slope * offset**2

    def _checked_time(self, time: ArrayLike, /) -> Array:
        query = jnp.asarray(time, dtype=self.times.dtype)
        if query.shape != ():
            raise ValueError("A history query requires one scalar physical time.")
        return eqx.error_if(
            query,
            ~jnp.isfinite(query) | (query < self.times[0]) | (query > self.times[-1]),
            "History time must be finite and within the declared time interval.",
        )


__all__ = ["IrradiationConditions", "IrradiationHistory"]
