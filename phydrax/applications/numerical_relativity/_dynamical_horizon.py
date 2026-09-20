#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, IntFlag

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._horizon_tracking import HorizonGeometryEvidence
from ._mots import MOTSSolveResult
from ._surfaces import SphericalSpectralSurface


class HorizonWorldtubeRegime(IntEnum):
    """Slice-local regime of a quasilocal marginal-surface worldtube."""

    INVALID = 0
    ISOLATED = 1
    DYNAMICAL = 2
    TRANSITIONAL = 3


class DynamicalHorizonStatus(IntFlag):
    """Status bits for isolated/dynamical-horizon balance evidence."""

    SUCCESS = 0
    NONFINITE = 1
    SLICE_UNQUALIFIED = 2
    BALANCE_NOT_CONVERGED = 4
    NONPHYSICAL_ENERGY_FLUX = 8
    TIMELIKE_WORLDTUBE = 16
    AREA_DECREASE = 32
    DERIVATIVE_INVALID = 64


def _time_derivative(values: Array, times: Array, /) -> Array:
    previous_width = times[1:-1] - times[:-2]
    next_width = times[2:] - times[1:-1]
    previous_weight = -next_width / (previous_width * (previous_width + next_width))
    center_weight = (next_width - previous_width) / (previous_width * next_width)
    next_weight = previous_width / (next_width * (previous_width + next_width))
    interior = (
        previous_weight * values[:-2]
        + center_weight * values[1:-1]
        + next_weight * values[2:]
    )
    first = (values[1] - values[0]) / (times[1] - times[0])
    last = (values[-1] - values[-2]) / (times[-1] - times[-2])
    return jnp.concatenate((first[None], interior, last[None]), axis=0)


class QuasilocalHorizonWorldtube(StrictModule, NonTrainableState):
    """Fixed-capacity history of qualified marginal-surface candidates.

    The worldtube is built from ``MOTSSolveResult`` and
    ``HorizonGeometryEvidence`` objects.  It is therefore a quasilocal product;
    neither this class nor its balance plan assigns global/event-horizon status.
    ``worldtube_signature`` is positive for spacelike portions, zero for null
    portions, and negative for timelike portions in the caller's fixed
    normalization.
    """

    times: Array
    surfaces: tuple[SphericalSpectralSurface, ...]
    area: Array
    angular_momentum: Array
    horizon_mass: Array
    matter_energy_flux: Array
    gravitational_energy_flux: Array
    matter_angular_momentum_flux: Array
    gravitational_angular_momentum_flux: Array
    worldtube_signature: Array
    slice_finite: Array
    slice_converged: Array
    slice_physically_valid: Array
    slice_qualified: Array
    slice_derivative_valid: Array
    time_capacity: int = eqx.field(static=True)
    surface_plan_id: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    worldtube_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        mots_slices: tuple[MOTSSolveResult, ...],
        geometry_slices: tuple[HorizonGeometryEvidence, ...],
        matter_energy_flux: ArrayLike,
        gravitational_energy_flux: ArrayLike,
        matter_angular_momentum_flux: ArrayLike,
        gravitational_angular_momentum_flux: ArrayLike,
        worldtube_signature: ArrayLike,
        /,
        *,
        worldtube_name: str = "quasilocal-horizon-worldtube",
    ):
        times_host = np.asarray(times, dtype=np.float64)
        if (
            times_host.ndim != 1
            or times_host.size < 3
            or np.any(~np.isfinite(times_host))
            or np.any(np.diff(times_host) <= 0.0)
        ):
            raise ValueError(
                "Worldtube times must contain three finite increasing nodes."
            )
        time_count = times_host.size
        if not isinstance(mots_slices, tuple) or not isinstance(geometry_slices, tuple):
            raise TypeError("MOTS and geometry histories must be fixed host tuples.")
        if len(mots_slices) != time_count or len(geometry_slices) != time_count:
            raise ValueError("MOTS and geometry histories must match time capacity.")
        if any(not isinstance(item, MOTSSolveResult) for item in mots_slices):
            raise TypeError("mots_slices must contain only MOTSSolveResult objects.")
        if any(not isinstance(item, HorizonGeometryEvidence) for item in geometry_slices):
            raise TypeError(
                "geometry_slices must contain only HorizonGeometryEvidence objects."
            )
        surface_plan_ids = {item.surface.plan_id for item in mots_slices}
        convention_ids = {item.convention_id for item in geometry_slices}
        if len(surface_plan_ids) != 1 or len(convention_ids) != 1:
            raise ValueError("One worldtube requires one surface plan and convention.")
        flux_inputs = (
            matter_energy_flux,
            gravitational_energy_flux,
            matter_angular_momentum_flux,
            gravitational_angular_momentum_flux,
            worldtube_signature,
        )
        flux_names = (
            "matter_energy_flux",
            "gravitational_energy_flux",
            "matter_angular_momentum_flux",
            "gravitational_angular_momentum_flux",
            "worldtube_signature",
        )
        fluxes = tuple(np.asarray(value, dtype=np.float64) for value in flux_inputs)
        for name, values in zip(flux_names, fluxes):
            if values.shape != (time_count,):
                raise ValueError(f"{name} must match fixed worldtube time capacity.")
        if not isinstance(worldtube_name, str) or not worldtube_name:
            raise ValueError("worldtube_name must be a nonempty string.")

        surfaces = tuple(item.surface for item in mots_slices)
        area = jnp.stack(tuple(item.area for item in geometry_slices))
        angular_momentum = jnp.stack(
            tuple(item.angular_momentum for item in geometry_slices)
        )
        horizon_mass = jnp.stack(
            tuple(item.christodoulou_mass for item in geometry_slices)
        )
        slice_finite = jnp.stack(
            tuple(
                item.finite & geometry.finite
                for item, geometry in zip(mots_slices, geometry_slices)
            )
        )
        slice_converged = jnp.stack(tuple(item.converged for item in mots_slices))
        slice_physical = jnp.stack(
            tuple(
                item.physically_valid & geometry.physically_valid
                for item, geometry in zip(mots_slices, geometry_slices)
            )
        )
        slice_qualified = jnp.stack(
            tuple(
                item.qualified & geometry.qualified
                for item, geometry in zip(mots_slices, geometry_slices)
            )
        )
        slice_derivative = jnp.stack(
            tuple(
                item.derivative_valid & geometry.derivative_valid
                for item, geometry in zip(mots_slices, geometry_slices)
            )
        )

        self.times = jnp.asarray(times_host, dtype=area.dtype)
        self.surfaces = surfaces
        self.area = area
        self.angular_momentum = angular_momentum
        self.horizon_mass = horizon_mass
        self.matter_energy_flux = jnp.asarray(fluxes[0], dtype=area.dtype)
        self.gravitational_energy_flux = jnp.asarray(fluxes[1], dtype=area.dtype)
        self.matter_angular_momentum_flux = jnp.asarray(fluxes[2], dtype=area.dtype)
        self.gravitational_angular_momentum_flux = jnp.asarray(
            fluxes[3], dtype=area.dtype
        )
        self.worldtube_signature = jnp.asarray(fluxes[4], dtype=area.dtype)
        self.slice_finite = slice_finite
        self.slice_converged = slice_converged
        self.slice_physically_valid = slice_physical
        self.slice_qualified = slice_qualified
        self.slice_derivative_valid = slice_derivative
        self.time_capacity = time_count
        self.surface_plan_id = next(iter(surface_plan_ids))
        self.convention_id = next(iter(convention_ids))
        self.worldtube_id = canonical_fingerprint(
            {
                "kind": "fixed-capacity-quasilocal-horizon-worldtube",
                "name": worldtube_name,
                "times": times_host,
                "surface_plan": self.surface_plan_id,
                "surface_coefficients": tuple(item.coefficients for item in surfaces),
                "surface_centers": tuple(item.center for item in surfaces),
                "area": area,
                "angular_momentum": angular_momentum,
                "horizon_mass": horizon_mass,
                "matter_energy_flux": fluxes[0],
                "gravitational_energy_flux": fluxes[1],
                "matter_angular_momentum_flux": fluxes[2],
                "gravitational_angular_momentum_flux": fluxes[3],
                "worldtube_signature": fluxes[4],
                "slice_qualified": slice_qualified,
                "convention": self.convention_id,
            }
        )


class DynamicalHorizonBalanceProduct(StrictModule):
    """Energy/angular-momentum flux-law evidence for one quasilocal worldtube."""

    times: Array
    irreducible_mass: Array
    total_energy_flux: Array
    total_angular_momentum_flux: Array
    cumulative_energy_flux: Array
    cumulative_angular_momentum_flux: Array
    energy_balance_residual: Array
    angular_momentum_balance_residual: Array
    energy_balance_ratio: Array
    angular_momentum_balance_ratio: Array
    area_rate: Array
    area_interval_rate: Array
    mass_rate: Array
    derivative_mask: Array
    regime: Array
    isolated: Array
    dynamical: Array
    status: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    worldtube_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)


class DynamicalHorizonBalancePlan(StrictModule, NonTrainableState):
    """Fixed-capacity isolated/dynamical-horizon flux-law audit."""

    time_capacity: int = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    isolation_flux_tolerance: float = eqx.field(static=True)
    isolation_rate_tolerance: float = eqx.field(static=True)
    signature_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        time_capacity: int,
        /,
        *,
        absolute_tolerance: float = 1.0e-9,
        relative_tolerance: float = 1.0e-7,
        isolation_flux_tolerance: float = 1.0e-10,
        isolation_rate_tolerance: float = 1.0e-9,
        signature_tolerance: float = 1.0e-10,
        plan_name: str = "dynamical-horizon-balance",
    ):
        raw_capacity = np.asarray(time_capacity)
        if (
            raw_capacity.shape != ()
            or not np.issubdtype(raw_capacity.dtype, np.integer)
            or int(raw_capacity) < 3
        ):
            raise ValueError("time_capacity must be an integer of at least three.")
        tolerances = tuple(
            float(value)
            for value in (
                absolute_tolerance,
                relative_tolerance,
                isolation_flux_tolerance,
                isolation_rate_tolerance,
                signature_tolerance,
            )
        )
        if any(not np.isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("Horizon balance tolerances must be finite and nonnegative.")
        if (
            tolerances[0] <= 0.0
            or tolerances[2] <= 0.0
            or tolerances[3] <= 0.0
            or tolerances[4] <= 0.0
        ):
            raise ValueError(
                "Absolute, isolation, rate, and signature tolerances must be positive."
            )
        if not isinstance(plan_name, str) or not plan_name:
            raise ValueError("plan_name must be a nonempty string.")
        self.time_capacity = int(raw_capacity)
        self.absolute_tolerance = tolerances[0]
        self.relative_tolerance = tolerances[1]
        self.isolation_flux_tolerance = tolerances[2]
        self.isolation_rate_tolerance = tolerances[3]
        self.signature_tolerance = tolerances[4]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-capacity-dynamical-horizon-balance-plan",
                "name": plan_name,
                "time_capacity": self.time_capacity,
                "absolute_tolerance": self.absolute_tolerance,
                "relative_tolerance": self.relative_tolerance,
                "isolation_flux_tolerance": self.isolation_flux_tolerance,
                "isolation_rate_tolerance": self.isolation_rate_tolerance,
                "signature_tolerance": self.signature_tolerance,
            }
        )

    def evaluate(
        self, worldtube: QuasilocalHorizonWorldtube, /
    ) -> DynamicalHorizonBalanceProduct:
        if worldtube.time_capacity != self.time_capacity:
            raise ValueError("Worldtube does not match balance-plan time capacity.")
        total_energy_flux = (
            worldtube.matter_energy_flux + worldtube.gravitational_energy_flux
        )
        total_angular_flux = (
            worldtube.matter_angular_momentum_flux
            + worldtube.gravitational_angular_momentum_flux
        )
        time_step = worldtube.times[1:] - worldtube.times[:-1]
        energy_increment = (
            0.5 * (total_energy_flux[1:] + total_energy_flux[:-1]) * time_step
        )
        angular_increment = (
            0.5 * (total_angular_flux[1:] + total_angular_flux[:-1]) * time_step
        )
        cumulative_energy = jnp.concatenate(
            (
                jnp.zeros((1,), dtype=total_energy_flux.dtype),
                jnp.cumsum(energy_increment),
            )
        )
        cumulative_angular = jnp.concatenate(
            (
                jnp.zeros((1,), dtype=total_angular_flux.dtype),
                jnp.cumsum(angular_increment),
            )
        )
        mass_change = worldtube.horizon_mass - worldtube.horizon_mass[0]
        angular_change = worldtube.angular_momentum - worldtube.angular_momentum[0]
        energy_residual = mass_change - cumulative_energy
        angular_residual = angular_change - cumulative_angular
        energy_limit = self.absolute_tolerance + self.relative_tolerance * jnp.maximum(
            jnp.maximum(jnp.abs(mass_change), jnp.abs(cumulative_energy)),
            jnp.abs(worldtube.horizon_mass),
        )
        angular_limit = self.absolute_tolerance + self.relative_tolerance * jnp.maximum(
            jnp.maximum(jnp.abs(angular_change), jnp.abs(cumulative_angular)),
            jnp.maximum(jnp.abs(worldtube.angular_momentum), 1.0),
        )
        energy_ratio = jnp.abs(energy_residual) / energy_limit
        angular_ratio = jnp.abs(angular_residual) / angular_limit
        converged = jnp.all(energy_ratio <= 1.0) & jnp.all(angular_ratio <= 1.0)

        area_rate = _time_derivative(worldtube.area, worldtube.times)
        area_interval_rate = (worldtube.area[1:] - worldtube.area[:-1]) / time_step
        mass_rate = _time_derivative(worldtube.horizon_mass, worldtube.times)
        derivative_mask = (jnp.arange(self.time_capacity) > 0) & (
            jnp.arange(self.time_capacity) < self.time_capacity - 1
        )
        slice_ready = (
            worldtube.slice_finite
            & worldtube.slice_converged
            & worldtube.slice_physically_valid
            & worldtube.slice_qualified
        )
        isolated = (
            slice_ready
            & (jnp.abs(worldtube.worldtube_signature) <= self.signature_tolerance)
            & (jnp.abs(total_energy_flux) <= self.isolation_flux_tolerance)
            & (jnp.abs(total_angular_flux) <= self.isolation_flux_tolerance)
            & (jnp.abs(area_rate) <= self.isolation_rate_tolerance)
            & (jnp.abs(mass_rate) <= self.isolation_rate_tolerance)
        )
        dynamical = (
            slice_ready
            & (worldtube.worldtube_signature > self.signature_tolerance)
            & (area_rate >= -self.isolation_rate_tolerance)
            & (
                (total_energy_flux > self.isolation_flux_tolerance)
                | (jnp.abs(total_angular_flux) > self.isolation_flux_tolerance)
                | (area_rate > self.isolation_rate_tolerance)
            )
        )
        regime = jnp.where(
            isolated,
            int(HorizonWorldtubeRegime.ISOLATED),
            jnp.where(
                dynamical,
                int(HorizonWorldtubeRegime.DYNAMICAL),
                jnp.where(
                    slice_ready,
                    int(HorizonWorldtubeRegime.TRANSITIONAL),
                    int(HorizonWorldtubeRegime.INVALID),
                ),
            ),
        ).astype(jnp.int32)

        finite = (
            jnp.all(worldtube.slice_finite)
            & jnp.all(jnp.isfinite(total_energy_flux))
            & jnp.all(jnp.isfinite(total_angular_flux))
            & jnp.all(jnp.isfinite(energy_residual))
            & jnp.all(jnp.isfinite(angular_residual))
            & jnp.all(jnp.isfinite(area_rate))
            & jnp.all(jnp.isfinite(area_interval_rate))
            & jnp.all(jnp.isfinite(mass_rate))
        )
        energy_flux_physical = jnp.all(
            worldtube.matter_energy_flux >= -self.isolation_flux_tolerance
        ) & jnp.all(worldtube.gravitational_energy_flux >= -self.isolation_flux_tolerance)
        signature_physical = jnp.all(
            worldtube.worldtube_signature >= -self.signature_tolerance
        )
        area_nondecreasing = jnp.all(area_interval_rate >= -self.isolation_rate_tolerance)
        physically_valid = (
            jnp.all(worldtube.slice_physically_valid)
            & energy_flux_physical
            & signature_physical
            & area_nondecreasing
            & jnp.all(worldtube.area > 0.0)
            & jnp.all(worldtube.horizon_mass > 0.0)
        )
        derivative_valid = (
            finite
            & jnp.all(worldtube.slice_derivative_valid)
            & jnp.all(jnp.isfinite(area_rate[derivative_mask]))
            & jnp.all(jnp.isfinite(mass_rate[derivative_mask]))
        )
        qualified = (
            finite & converged & physically_valid & jnp.all(worldtube.slice_qualified)
        )
        status = jnp.asarray(int(DynamicalHorizonStatus.SUCCESS), dtype=jnp.int32)
        status = status | jnp.where(
            finite, 0, int(DynamicalHorizonStatus.NONFINITE)
        ).astype(jnp.int32)
        status = status | jnp.where(
            jnp.all(worldtube.slice_qualified),
            0,
            int(DynamicalHorizonStatus.SLICE_UNQUALIFIED),
        ).astype(jnp.int32)
        status = status | jnp.where(
            converged, 0, int(DynamicalHorizonStatus.BALANCE_NOT_CONVERGED)
        ).astype(jnp.int32)
        status = status | jnp.where(
            energy_flux_physical,
            0,
            int(DynamicalHorizonStatus.NONPHYSICAL_ENERGY_FLUX),
        ).astype(jnp.int32)
        status = status | jnp.where(
            signature_physical, 0, int(DynamicalHorizonStatus.TIMELIKE_WORLDTUBE)
        ).astype(jnp.int32)
        status = status | jnp.where(
            area_nondecreasing, 0, int(DynamicalHorizonStatus.AREA_DECREASE)
        ).astype(jnp.int32)
        status = status | jnp.where(
            derivative_valid, 0, int(DynamicalHorizonStatus.DERIVATIVE_INVALID)
        ).astype(jnp.int32)
        irreducible_mass = jnp.sqrt(worldtube.area / (16.0 * jnp.pi))
        product_id = canonical_fingerprint(
            {
                "kind": "isolated-dynamical-horizon-balance-product",
                "worldtube": worldtube.worldtube_id,
                "plan": self.plan_id,
            }
        )
        return DynamicalHorizonBalanceProduct(
            worldtube.times,
            irreducible_mass,
            total_energy_flux,
            total_angular_flux,
            cumulative_energy,
            cumulative_angular,
            energy_residual,
            angular_residual,
            energy_ratio,
            angular_ratio,
            area_rate,
            area_interval_rate,
            mass_rate,
            derivative_mask,
            regime,
            isolated,
            dynamical,
            status,
            finite,
            converged,
            physically_valid,
            qualified,
            derivative_valid,
            worldtube.worldtube_id,
            self.plan_id,
            product_id,
        )
