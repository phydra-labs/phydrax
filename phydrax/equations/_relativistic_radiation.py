#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._physical import RelativityScaleContract
from .._strict import StrictModule
from ..metrix._adm_exchange import ADMGridGeometry, StressEnergyProjection
from ..metrix._spacetime_conventions import RelativityConvention
from ._hyperbolic_systems import AbstractAdmissibleSystem
from ._radiation_moments import MultigroupM1RadiationSystem


def _m1_pressure(
    energy: Array,
    flux_vector: Array,
    inverse_metric: Array,
    physical_light_speed: float,
    energy_floor: float,
    /,
) -> tuple[Array, Array, Array, Array]:
    flux_squared = contract(
        "...i,...ij,...j->...",
        flux_vector,
        inverse_metric,
        flux_vector,
        backend="jax",
    )
    flux_norm = jnp.sqrt(jnp.maximum(flux_squared, 0.0))
    light_speed = jnp.asarray(physical_light_speed, dtype=energy.dtype)
    safe_energy = jnp.maximum(energy, jnp.asarray(energy_floor, dtype=energy.dtype))
    unconstrained_flux_factor = flux_norm / (light_speed * safe_energy)
    flux_factor = jnp.clip(unconstrained_flux_factor, 0.0, 1.0)
    eddington = (3.0 + 4.0 * flux_factor**2) / (
        5.0 + 2.0 * jnp.sqrt(jnp.maximum(4.0 - 3.0 * flux_factor**2, 0.0))
    )
    safe_norm = jnp.where(flux_norm > 0.0, flux_norm, 1.0)
    direction = (
        contract("...ij,...j->...i", inverse_metric, flux_vector, backend="jax")
        / safe_norm[..., None]
    )
    pressure = (
        0.5
        * (1.0 - eddington)[..., None, None]
        * energy[..., None, None]
        * inverse_metric
        + 0.5
        * (3.0 * eddington - 1.0)[..., None, None]
        * energy[..., None, None]
        * direction[..., :, None]
        * direction[..., None, :]
    )
    return pressure, flux_norm, unconstrained_flux_factor, eddington


class GRGreyM1ClosureEvaluation(StrictModule):
    pressure_tensor: Array
    flux_vector: Array
    flux_norm: Array
    reduced_flux: Array
    eddington_factor: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    system_id: str = eqx.field(static=True)


class GRRadiationMatterExchange(StrictModule):
    """Physical four-force split into transport-state and ADM source units.

    ``radiation_flux_source`` advances the M1 flux variable ``F_i``;
    ``radiation_momentum_source = radiation_flux_source / c`` is the Eulerian
    ADM momentum source balanced by ``matter_momentum_source``.
    """

    radiation_energy_source: Array
    radiation_flux_source: Array
    radiation_momentum_source: Array
    matter_energy_source: Array
    matter_momentum_source: Array
    comoving_energy_density: Array
    comoving_flux_four_vector: Array
    interaction_four_force: Array
    energy_balance_residual: Array
    momentum_balance_residual: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    system_id: str = eqx.field(static=True)


class GRGreyM1RadiationSystem(AbstractAdmissibleSystem):
    """Grey M1 radiation and covariant matter exchange in a 3+1 frame.

    The local conserved state is ``(E, F^x, F^y, F^z)`` and reuses the native
    M1 hyperbolic system.  Metric-aware closure methods take spatial covariant
    flux instead, avoiding an implicit Euclidean index convention.  Absorption
    and scattering coefficients are inverse code lengths.  Reduced light speed
    changes only hyperbolic transport; frame boosts, stress-energy projection,
    and interaction sources use the physical light speed from ``scale``.  Matter
    sources are the exact negatives of radiation sources.
    """

    scale: RelativityScaleContract
    convention: RelativityConvention
    local_system: MultigroupM1RadiationSystem
    absorption_coefficient: float = eqx.field(static=True)
    scattering_coefficient: float = eqx.field(static=True)
    radiation_constant: float = eqx.field(static=True)
    metric_tolerance: float = eqx.field(static=True)
    source_convention: str = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        convention: RelativityConvention,
        /,
        *,
        reduced_light_speed: float | None = None,
        energy_floor: float = 1.0e-12,
        absorption_coefficient: float = 0.0,
        scattering_coefficient: float = 0.0,
        radiation_constant: float = 1.0,
        metric_tolerance: float = 1.0e-9,
    ) -> None:
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be RelativityScaleContract.")
        if not isinstance(convention, RelativityConvention):
            raise TypeError("convention must be RelativityConvention.")
        if convention.metric_signature != "mostly_plus":
            raise ValueError("GR grey M1 requires the mostly-plus convention.")
        physical_light_speed = float(scale.speed_of_light)
        reduced = (
            physical_light_speed
            if reduced_light_speed is None
            else float(reduced_light_speed)
        )
        absorption = float(absorption_coefficient)
        scattering = float(scattering_coefficient)
        constant = float(radiation_constant)
        tolerance = float(metric_tolerance)
        if (
            not np.isfinite(reduced)
            or reduced <= 0.0
            or reduced > physical_light_speed
            or not np.isfinite(absorption)
            or absorption < 0.0
            or not np.isfinite(scattering)
            or scattering < 0.0
            or not np.isfinite(constant)
            or constant <= 0.0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("GR grey M1 material coefficients are invalid.")
        local = MultigroupM1RadiationSystem(
            1,
            3,
            reduced_light_speed=reduced,
            energy_floor=energy_floor,
        )
        self.dimension = 3
        self.component_names = (
            "radiation_energy",
            "radiation_flux_x",
            "radiation_flux_y",
            "radiation_flux_z",
        )
        self.scale = scale
        self.convention = convention
        self.local_system = local
        self.absorption_coefficient = absorption
        self.scattering_coefficient = scattering
        self.radiation_constant = constant
        self.metric_tolerance = tolerance
        self.source_convention = "physical-light-speed-unscaled-four-force"
        self.system_id = canonical_fingerprint(
            {
                "kind": "gr-grey-m1-radiation-system",
                "scale": scale.scale_id,
                "convention": convention.convention_id,
                "local_system": local.system_id,
                "absorption_coefficient": absorption,
                "scattering_coefficient": scattering,
                "radiation_constant": constant,
                "metric_tolerance": tolerance,
                "source_convention": self.source_convention,
            }
        )

    @property
    def reduced_light_speed(self) -> float:
        return self.local_system.reduced_light_speed

    @property
    def physical_light_speed(self) -> float:
        return float(self.scale.speed_of_light)

    @property
    def energy_floor(self) -> float:
        return self.local_system.energy_floor

    @staticmethod
    def _state(state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape[-1:] != (4,):
            raise ValueError("Grey M1 state must have four trailing components.")
        return value

    def conserved_to_primitive(self, state: Array, /) -> Array:
        return self.local_system.conserved_to_primitive(self._state(state))

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        return self.local_system.primitive_to_conserved(self._state(primitive))

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        del args
        value = self._state(state)
        axis_ = int(axis)
        if axis_ not in (0, 1, 2):
            raise ValueError("Grey M1 flux axis must be zero, one, or two.")
        energy = value[..., 0]
        flux = value[..., 1:]
        identity = jnp.broadcast_to(
            jnp.eye(3, dtype=value.dtype), value.shape[:-1] + (3, 3)
        )
        pressure, _, _, _ = _m1_pressure(
            energy,
            flux,
            identity,
            self.physical_light_speed,
            self.energy_floor,
        )
        result = jnp.zeros_like(value)
        result = result.at[..., 0].set(flux[..., axis_])
        return result.at[..., 1:].set(
            self.reduced_light_speed**2 * pressure[..., :, axis_]
        )

    def max_wave_speed(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        return self.local_system.max_wave_speed(
            self._state(left), self._state(right), axis, args
        )

    def signal_bounds(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        return self.local_system.signal_bounds(
            self._state(left), self._state(right), axis, args
        )

    def normal_signal_bounds(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        return self.local_system.normal_signal_bounds(
            self._state(left), self._state(right), normal, args
        )

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        return self.local_system.reflect_state(self._state(state), axis)

    def admissible(self, state: Array, /) -> Array:
        value = self._state(state)
        energy = value[..., 0]
        flux_norm = jnp.sqrt(jnp.sum(value[..., 1:] ** 2, axis=-1))
        return (
            jnp.all(jnp.isfinite(value), axis=-1)
            & (energy > self.energy_floor)
            & (flux_norm <= self.physical_light_speed * energy)
        )

    def closure(
        self,
        energy_density: ArrayLike,
        flux_covector: ArrayLike,
        geometry: ADMGridGeometry,
        /,
    ) -> GRGreyM1ClosureEvaluation:
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be ADMGridGeometry.")
        if (
            geometry.scale_id != self.scale.scale_id
            or geometry.convention_id != self.convention.convention_id
        ):
            raise ValueError("ADM geometry and radiation system contracts differ.")
        energy = jnp.asarray(energy_density)
        flux_covector_ = jnp.asarray(flux_covector, dtype=energy.dtype)
        metric = geometry.spatial_metric.astype(energy.dtype)
        inverse = geometry.inverse_spatial_metric.astype(energy.dtype)
        cell_shape = geometry.leading_shape
        if energy.shape != cell_shape or flux_covector_.shape != cell_shape + (3,):
            raise ValueError("Grey M1 moments must match ADM geometry.")
        flux_vector = contract("...ij,...j->...i", inverse, flux_covector_, backend="jax")
        flux_squared = contract(
            "...i,...i->...", flux_covector_, flux_vector, backend="jax"
        )
        (
            pressure,
            flux_norm,
            unconstrained_reduced_flux,
            eddington,
        ) = _m1_pressure(
            energy,
            flux_covector_,
            inverse,
            self.physical_light_speed,
            self.energy_floor,
        )
        reduced_flux = jnp.clip(unconstrained_reduced_flux, 0.0, 1.0)
        identity = contract("...ik,...kj->...ij", metric, inverse, backend="jax")
        metric_residual = jnp.max(
            jnp.abs(identity - jnp.eye(3, dtype=energy.dtype)), axis=(-2, -1)
        )
        finite = (
            geometry.finite
            & jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(flux_covector_), axis=-1)
            & jnp.isfinite(flux_norm)
            & jnp.all(jnp.isfinite(pressure), axis=(-2, -1))
        )
        physically_valid = (
            finite
            & geometry.physically_valid
            & (energy > self.energy_floor)
            & (flux_squared >= 0.0)
            & (unconstrained_reduced_flux <= 1.0)
        )
        qualified = physically_valid & (metric_residual <= self.metric_tolerance)
        derivative_valid = qualified & (
            unconstrained_reduced_flux < 1.0 - 32.0 * jnp.finfo(energy.dtype).eps
        )
        return GRGreyM1ClosureEvaluation(
            pressure,
            flux_vector,
            flux_norm,
            reduced_flux,
            eddington,
            finite,
            physically_valid,
            qualified,
            derivative_valid,
            self.system_id,
        )

    def stress_energy_projection(
        self,
        energy_density: ArrayLike,
        flux_covector: ArrayLike,
        geometry: ADMGridGeometry,
        /,
    ) -> StressEnergyProjection:
        closure = self.closure(energy_density, flux_covector, geometry)
        energy = jnp.asarray(energy_density)
        flux = jnp.asarray(flux_covector, dtype=energy.dtype)
        metric = geometry.spatial_metric.astype(energy.dtype)
        stress_covariant = contract(
            "...ik,...jl,...kl->...ij",
            metric,
            metric,
            closure.pressure_tensor,
            backend="jax",
        )
        projection_defect = jnp.max(
            jnp.abs(stress_covariant - jnp.swapaxes(stress_covariant, -1, -2)),
            axis=(-2, -1),
        )
        conservation_defect = jnp.zeros_like(energy)
        projection_id = canonical_fingerprint(
            {
                "kind": "gr-grey-m1-stress-energy-projection",
                "system": self.system_id,
                "geometry_lineage": geometry.geometry_lineage_id,
            }
        )
        return StressEnergyProjection(
            energy,
            flux / jnp.asarray(self.physical_light_speed, dtype=energy.dtype),
            stress_covariant,
            geometry.active,
            closure.qualified,
            projection_defect,
            conservation_defect,
            snapshot_token=geometry.snapshot_token,
            geometry_lineage_id=geometry.geometry_lineage_id,
            convention_id=geometry.convention_id,
            scale_id=geometry.scale_id,
            topology_id=geometry.topology_id,
            projection_id=projection_id,
        )

    def coordinate_flux(
        self,
        energy_density: ArrayLike,
        flux_covector: ArrayLike,
        axis: int,
        geometry: ADMGridGeometry,
        /,
    ) -> Array:
        closure = self.closure(energy_density, flux_covector, geometry)
        energy = jnp.asarray(energy_density)
        flux_covector_ = jnp.asarray(flux_covector, dtype=energy.dtype)
        lapse_ = geometry.alpha.astype(energy.dtype)
        shift_ = geometry.beta_contravariant.astype(energy.dtype)
        metric = geometry.spatial_metric.astype(energy.dtype)
        axis_ = int(axis)
        if axis_ not in (0, 1, 2):
            raise ValueError("Grey M1 flux axis must be zero, one, or two.")
        mixed_pressure = contract(
            "...ik,...ka->...ia", metric, closure.pressure_tensor, backend="jax"
        )
        energy_flux = (
            lapse_ * closure.flux_vector[..., axis_] - shift_[..., axis_] * energy
        )
        momentum_flux = (
            lapse_[..., None]
            * self.reduced_light_speed**2
            * mixed_pressure[..., :, axis_]
            - shift_[..., axis_, None] * flux_covector_
        )
        return jnp.concatenate((energy_flux[..., None], momentum_flux), axis=-1)

    def coordinate_characteristic_bounds(
        self,
        unit_covector: ArrayLike,
        geometry: ADMGridGeometry,
        /,
    ) -> tuple[Array, Array]:
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be ADMGridGeometry.")
        if (
            geometry.scale_id != self.scale.scale_id
            or geometry.convention_id != self.convention.convention_id
        ):
            raise ValueError("ADM geometry and radiation system contracts differ.")
        normal = jnp.asarray(unit_covector)
        lapse_ = geometry.alpha.astype(normal.dtype)
        shift_ = geometry.beta_contravariant.astype(normal.dtype)
        inverse = geometry.inverse_spatial_metric.astype(normal.dtype)
        if normal.shape != geometry.leading_shape + (3,):
            raise ValueError("Radiation characteristic covector must match ADM geometry.")
        normal_squared = contract(
            "...i,...ij,...j->...", normal, inverse, normal, backend="jax"
        )
        light_cone = (
            lapse_ * self.reduced_light_speed * jnp.sqrt(jnp.maximum(normal_squared, 0.0))
        )
        transport = -contract("...i,...i->...", shift_, normal, backend="jax")
        return transport - light_cone, transport + light_cone

    def matter_exchange(
        self,
        energy_density: ArrayLike,
        flux_covector: ArrayLike,
        fluid_velocity: ArrayLike,
        equilibrium_temperature: ArrayLike,
        geometry: ADMGridGeometry,
        /,
    ) -> GRRadiationMatterExchange:
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be ADMGridGeometry.")
        energy = jnp.asarray(energy_density)
        flux_covector_ = jnp.asarray(flux_covector, dtype=energy.dtype)
        velocity = jnp.asarray(fluid_velocity, dtype=energy.dtype)
        temperature = jnp.asarray(equilibrium_temperature, dtype=energy.dtype)
        metric = geometry.spatial_metric.astype(energy.dtype)
        inverse = geometry.inverse_spatial_metric.astype(energy.dtype)
        closure = self.closure(energy, flux_covector_, geometry)
        cell_shape = geometry.leading_shape
        if (
            energy.shape != cell_shape
            or velocity.shape != cell_shape + (3,)
            or temperature.shape != cell_shape
        ):
            raise ValueError("Fluid and radiation fields must match ADM geometry.")
        light_speed = jnp.asarray(self.physical_light_speed, dtype=energy.dtype)
        normalized_velocity = velocity / light_speed
        velocity_covector = contract(
            "...ij,...j->...i", metric, normalized_velocity, backend="jax"
        )
        speed_squared = contract(
            "...i,...i->...", normalized_velocity, velocity_covector, backend="jax"
        )
        lorentz = 1.0 / jnp.sqrt(
            jnp.maximum(1.0 - speed_squared, jnp.finfo(energy.dtype).tiny)
        )
        four_velocity = jnp.concatenate(
            (lorentz[..., None], lorentz[..., None] * normalized_velocity), axis=-1
        )
        lower_four_velocity = jnp.concatenate(
            ((-lorentz)[..., None], lorentz[..., None] * velocity_covector), axis=-1
        )
        stress_energy = jnp.zeros(cell_shape + (4, 4), dtype=energy.dtype)
        stress_energy = stress_energy.at[..., 0, 0].set(energy)
        stress_energy = stress_energy.at[..., 0, 1:].set(
            closure.flux_vector / light_speed
        )
        stress_energy = stress_energy.at[..., 1:, 0].set(
            closure.flux_vector / light_speed
        )
        stress_energy = stress_energy.at[..., 1:, 1:].set(closure.pressure_tensor)
        comoving_energy = contract(
            "...m,...mn,...n->...",
            lower_four_velocity,
            stress_energy,
            lower_four_velocity,
            backend="jax",
        )
        energy_current = -contract(
            "...mn,...n->...m", stress_energy, lower_four_velocity, backend="jax"
        )
        comoving_flux = energy_current - comoving_energy[..., None] * four_velocity
        equilibrium_energy = (
            jnp.asarray(self.radiation_constant, dtype=energy.dtype) * temperature**4
        )
        total_extinction = self.absorption_coefficient + self.scattering_coefficient
        interaction = (
            self.absorption_coefficient
            * (comoving_energy - equilibrium_energy)[..., None]
            * four_velocity
            + total_extinction * comoving_flux
        )
        radiation_energy_source = -light_speed * interaction[..., 0]
        radiation_flux_source_vector = -(light_speed**2) * interaction[..., 1:]
        radiation_flux_source = contract(
            "...ij,...j->...i", metric, radiation_flux_source_vector, backend="jax"
        )
        radiation_momentum_source = radiation_flux_source / light_speed
        matter_energy_source = -radiation_energy_source
        matter_momentum_source = -radiation_momentum_source
        energy_residual = radiation_energy_source + matter_energy_source
        momentum_residual = radiation_momentum_source + matter_momentum_source
        finite = (
            geometry.finite
            & closure.finite
            & jnp.all(jnp.isfinite(velocity), axis=-1)
            & jnp.isfinite(temperature)
            & jnp.isfinite(comoving_energy)
            & jnp.all(jnp.isfinite(comoving_flux), axis=-1)
            & jnp.all(jnp.isfinite(interaction), axis=-1)
            & jnp.isfinite(energy_residual)
            & jnp.all(jnp.isfinite(momentum_residual), axis=-1)
        )
        physically_valid = (
            finite
            & geometry.physically_valid
            & closure.physically_valid
            & (speed_squared >= 0.0)
            & (speed_squared < 1.0)
            & (temperature >= 0.0)
            & (comoving_energy >= 0.0)
        )
        balance_scale = jnp.maximum(
            jnp.maximum(jnp.abs(radiation_energy_source), jnp.abs(matter_energy_source)),
            jnp.asarray(1.0, dtype=energy.dtype),
        )
        momentum_scale = jnp.maximum(
            jnp.max(jnp.abs(radiation_momentum_source), axis=-1),
            jnp.asarray(1.0, dtype=energy.dtype),
        )
        tolerance = 64.0 * jnp.finfo(energy.dtype).eps
        converged = (
            finite
            & (jnp.abs(energy_residual) <= tolerance * balance_scale)
            & jnp.all(
                jnp.abs(momentum_residual) <= (tolerance * momentum_scale)[..., None],
                axis=-1,
            )
        )
        qualified = physically_valid & closure.qualified & converged
        derivative_valid = (
            qualified
            & closure.derivative_valid
            & (speed_squared < 1.0 - 32.0 * jnp.finfo(energy.dtype).eps)
        )
        return GRRadiationMatterExchange(
            radiation_energy_source,
            radiation_flux_source,
            radiation_momentum_source,
            matter_energy_source,
            matter_momentum_source,
            comoving_energy,
            comoving_flux,
            interaction,
            energy_residual,
            momentum_residual,
            finite,
            converged,
            physically_valid,
            qualified,
            derivative_valid,
            self.system_id,
        )
