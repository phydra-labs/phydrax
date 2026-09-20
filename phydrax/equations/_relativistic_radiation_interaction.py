#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..metrix._adm_exchange import ADMGridGeometry
from ._relativistic_radiation import GRGrayM1RadiationSystem


class GRGrayOpacityEvaluation(StrictModule):
    """Gray interaction coefficients in inverse code length units."""

    planck_emission: Array
    planck_absorption: Array
    rosseland_transport: Array
    scattering: Array
    photon_absorption: Array
    photon_emission_rate: Array
    compton_coefficient: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    opacity_id: str = eqx.field(static=True)


class AbstractGRGrayOpacityPlan(StrictModule, NonTrainableState):
    """State-dependent gray opacity contract for relativistic radiation."""

    opacity_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def evaluate(
        self,
        rest_mass_density: ArrayLike,
        matter_temperature: ArrayLike,
        radiation_temperature: ArrayLike,
        magnetic_squared: ArrayLike,
        composition: ArrayLike | None = None,
        /,
    ) -> GRGrayOpacityEvaluation:
        raise NotImplementedError


class ConstantGRGrayOpacityPlan(AbstractGRGrayOpacityPlan):
    """Constant emission, absorption, transport, and scattering extinctions."""

    planck_emission: float = eqx.field(static=True)
    planck_absorption: float = eqx.field(static=True)
    rosseland_transport: float = eqx.field(static=True)
    scattering: float = eqx.field(static=True)
    photon_absorption: float = eqx.field(static=True)
    photon_emission_rate: float = eqx.field(static=True)
    compton_coefficient: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        planck_absorption: float = 0.0,
        planck_emission: float | None = None,
        rosseland_transport: float | None = None,
        scattering: float = 0.0,
        photon_absorption: float = 0.0,
        photon_emission_rate: float = 0.0,
        compton_coefficient: float = 0.0,
    ) -> None:
        absorption = float(planck_absorption)
        emission = absorption if planck_emission is None else float(planck_emission)
        transport = (
            absorption if rosseland_transport is None else float(rosseland_transport)
        )
        values = (
            emission,
            absorption,
            transport,
            float(scattering),
            float(photon_absorption),
            float(photon_emission_rate),
            float(compton_coefficient),
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "Constant GR gray coefficients must be finite and nonnegative."
            )
        (
            self.planck_emission,
            self.planck_absorption,
            self.rosseland_transport,
            self.scattering,
            self.photon_absorption,
            self.photon_emission_rate,
            self.compton_coefficient,
        ) = values
        self.opacity_id = canonical_fingerprint(
            {
                "kind": "constant-gr-gray-opacity",
                "planck_emission": emission,
                "planck_absorption": absorption,
                "rosseland_transport": transport,
                "scattering": values[3],
                "photon_absorption": values[4],
                "photon_emission_rate": values[5],
                "compton_coefficient": values[6],
            }
        )

    def evaluate(
        self,
        rest_mass_density: ArrayLike,
        matter_temperature: ArrayLike,
        radiation_temperature: ArrayLike,
        magnetic_squared: ArrayLike,
        composition: ArrayLike | None = None,
        /,
    ) -> GRGrayOpacityEvaluation:
        density, matter, radiation, magnetic = jnp.broadcast_arrays(
            jnp.asarray(rest_mass_density),
            jnp.asarray(matter_temperature),
            jnp.asarray(radiation_temperature),
            jnp.asarray(magnetic_squared),
        )
        if composition is not None:
            composition_ = jnp.broadcast_to(jnp.asarray(composition), density.shape)
            composition_finite = jnp.isfinite(composition_)
        else:
            composition_finite = jnp.ones_like(density, dtype=jnp.bool_)
        dtype = jnp.result_type(density, matter, radiation, magnetic)
        coefficient = lambda value: jnp.full(density.shape, value, dtype=dtype)
        finite = (
            jnp.isfinite(density)
            & jnp.isfinite(matter)
            & jnp.isfinite(radiation)
            & jnp.isfinite(magnetic)
            & composition_finite
        )
        physical = (
            finite
            & (density >= 0.0)
            & (matter >= 0.0)
            & (radiation >= 0.0)
            & (magnetic >= 0.0)
        )
        return GRGrayOpacityEvaluation(
            coefficient(self.planck_emission),
            coefficient(self.planck_absorption),
            coefficient(self.rosseland_transport),
            coefficient(self.scattering),
            coefficient(self.photon_absorption),
            coefficient(self.photon_emission_rate),
            coefficient(self.compton_coefficient),
            finite,
            physical,
            physical,
            physical,
            self.opacity_id,
        )


class CompositeGRGrayOpacityPlan(AbstractGRGrayOpacityPlan):
    """Add independent gray processes without merging their provenance."""

    processes: tuple[AbstractGRGrayOpacityPlan, ...]

    def __init__(self, processes: tuple[AbstractGRGrayOpacityPlan, ...], /) -> None:
        values = tuple(processes)
        if not values or any(
            not isinstance(value, AbstractGRGrayOpacityPlan) for value in values
        ):
            raise TypeError("processes must contain GR gray opacity plans.")
        self.processes = values
        self.opacity_id = canonical_fingerprint(
            {
                "kind": "composite-gr-gray-opacity",
                "processes": [value.opacity_id for value in values],
            }
        )

    def evaluate(
        self,
        rest_mass_density: ArrayLike,
        matter_temperature: ArrayLike,
        radiation_temperature: ArrayLike,
        magnetic_squared: ArrayLike,
        composition: ArrayLike | None = None,
        /,
    ) -> GRGrayOpacityEvaluation:
        values = tuple(
            process.evaluate(
                rest_mass_density,
                matter_temperature,
                radiation_temperature,
                magnetic_squared,
                composition,
            )
            for process in self.processes
        )
        first = values[0]
        emission = first.planck_emission
        absorption = first.planck_absorption
        transport = first.rosseland_transport
        scattering = first.scattering
        photon_absorption = first.photon_absorption
        photon_emission = first.photon_emission_rate
        compton = first.compton_coefficient
        finite = first.finite
        physical = first.physically_valid
        qualified = first.qualified
        derivative = first.derivative_valid
        for value in values[1:]:
            emission = emission + value.planck_emission
            absorption = absorption + value.planck_absorption
            transport = transport + value.rosseland_transport
            scattering = scattering + value.scattering
            photon_absorption = photon_absorption + value.photon_absorption
            photon_emission = photon_emission + value.photon_emission_rate
            compton = compton + value.compton_coefficient
            finite = finite & value.finite
            physical = physical & value.physically_valid
            qualified = qualified & value.qualified
            derivative = derivative & value.derivative_valid
        return GRGrayOpacityEvaluation(
            emission,
            absorption,
            transport,
            scattering,
            photon_absorption,
            photon_emission,
            compton,
            finite,
            physical,
            qualified,
            derivative,
            self.opacity_id,
        )


class GRRadiationMatterExchange(StrictModule):
    """Covariant four-force and exactly opposite Eulerian source projections."""

    radiation_energy_source: Array
    radiation_flux_source: Array
    radiation_momentum_source: Array
    matter_energy_source: Array
    matter_momentum_source: Array
    comoving_energy_density: Array
    comoving_flux_four_vector: Array
    interaction_four_force: Array
    radiation_temperature: Array
    opacity: GRGrayOpacityEvaluation
    energy_balance_residual: Array
    momentum_balance_residual: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    interaction_id: str = eqx.field(static=True)


class GRGrayRadiationInteractionPlan(StrictModule, NonTrainableState):
    """Gray M1 matter interaction separated from radiation transport."""

    radiation: GRGrayM1RadiationSystem
    opacity: AbstractGRGrayOpacityPlan
    radiation_constant: float = eqx.field(static=True)
    interaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        radiation: GRGrayM1RadiationSystem,
        opacity: AbstractGRGrayOpacityPlan,
        /,
        *,
        radiation_constant: float = 1.0,
    ) -> None:
        if not isinstance(radiation, GRGrayM1RadiationSystem):
            raise TypeError("radiation must be GRGrayM1RadiationSystem.")
        if not isinstance(opacity, AbstractGRGrayOpacityPlan):
            raise TypeError("opacity must implement AbstractGRGrayOpacityPlan.")
        constant = float(radiation_constant)
        if not np.isfinite(constant) or constant <= 0.0:
            raise ValueError("radiation_constant must be finite and positive.")
        self.radiation = radiation
        self.opacity = opacity
        self.radiation_constant = constant
        self.interaction_id = canonical_fingerprint(
            {
                "kind": "gr-gray-radiation-interaction",
                "radiation": radiation.system_id,
                "opacity": opacity.opacity_id,
                "radiation_constant": constant,
            }
        )

    def matter_exchange(
        self,
        energy_density: ArrayLike,
        flux_covector: ArrayLike,
        rest_mass_density: ArrayLike,
        fluid_velocity: ArrayLike,
        matter_temperature: ArrayLike,
        geometry: ADMGridGeometry,
        /,
        *,
        magnetic_squared: ArrayLike = 0.0,
        composition: ArrayLike | None = None,
    ) -> GRRadiationMatterExchange:
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be ADMGridGeometry.")
        energy = jnp.asarray(energy_density)
        flux_covector_ = jnp.asarray(flux_covector, dtype=energy.dtype)
        density = jnp.asarray(rest_mass_density, dtype=energy.dtype)
        velocity = jnp.asarray(fluid_velocity, dtype=energy.dtype)
        temperature = jnp.asarray(matter_temperature, dtype=energy.dtype)
        magnetic = jnp.broadcast_to(
            jnp.asarray(magnetic_squared, dtype=energy.dtype), energy.shape
        )
        metric = geometry.spatial_metric.astype(energy.dtype)
        closure = self.radiation.closure(energy, flux_covector_, geometry)
        cell_shape = geometry.leading_shape
        if (
            energy.shape != cell_shape
            or density.shape != cell_shape
            or velocity.shape != cell_shape + (3,)
            or temperature.shape != cell_shape
        ):
            raise ValueError("Fluid and radiation fields must match ADM geometry.")
        light_speed = jnp.asarray(self.radiation.physical_light_speed, dtype=energy.dtype)
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
        constant = jnp.asarray(self.radiation_constant, dtype=energy.dtype)
        radiation_temperature = jnp.power(
            jnp.maximum(comoving_energy, 0.0) / constant, 0.25
        )
        opacity = self.opacity.evaluate(
            density,
            temperature,
            radiation_temperature,
            magnetic,
            composition,
        )
        equilibrium_energy = constant * temperature**4
        temperature_scale = jnp.maximum(
            jnp.maximum(temperature, radiation_temperature),
            jnp.finfo(energy.dtype).tiny,
        )
        compton_exchange = (
            opacity.compton_coefficient
            * comoving_energy
            * (radiation_temperature - temperature)
            / temperature_scale
        )
        interaction = (
            opacity.planck_absorption * comoving_energy
            - opacity.planck_emission * equilibrium_energy
            + compton_exchange
        )[..., None] * four_velocity + (opacity.rosseland_transport + opacity.scattering)[
            ..., None
        ] * comoving_flux
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
            & opacity.finite
            & jnp.isfinite(density)
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
            & opacity.physically_valid
            & (density >= 0.0)
            & (speed_squared >= 0.0)
            & (speed_squared < 1.0)
            & (temperature >= 0.0)
            & (comoving_energy >= 0.0)
        )
        scale = jnp.maximum(
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
            & (jnp.abs(energy_residual) <= tolerance * scale)
            & jnp.all(
                jnp.abs(momentum_residual) <= (tolerance * momentum_scale)[..., None],
                axis=-1,
            )
        )
        qualified = physically_valid & closure.qualified & opacity.qualified & converged
        derivative_valid = (
            qualified
            & closure.derivative_valid
            & opacity.derivative_valid
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
            radiation_temperature,
            opacity,
            energy_residual,
            momentum_residual,
            finite,
            converged,
            physically_valid,
            qualified,
            derivative_valid,
            self.interaction_id,
        )


__all__ = [
    "AbstractGRGrayOpacityPlan",
    "CompositeGRGrayOpacityPlan",
    "ConstantGRGrayOpacityPlan",
    "GRGrayOpacityEvaluation",
    "GRGrayRadiationInteractionPlan",
    "GRRadiationMatterExchange",
]
