#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import canonical_fingerprint
from ..._physical import RelativityScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume import FiniteVolumeDiscretization
from ...equations._relativistic_radiation_interaction import (
    AbstractGRGreyOpacityPlan,
    GRGreyOpacityEvaluation,
)
from ...solver._relativistic_finite_volume import ValenciaFiniteVolumeStageGeometry


def _fields(
    rest_mass_density: ArrayLike,
    matter_temperature: ArrayLike,
    radiation_temperature: ArrayLike,
    magnetic_squared: ArrayLike,
    composition: ArrayLike | None,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    density, matter, radiation, magnetic = jnp.broadcast_arrays(
        jnp.asarray(rest_mass_density),
        jnp.asarray(matter_temperature),
        jnp.asarray(radiation_temperature),
        jnp.asarray(magnetic_squared),
    )
    composition_finite = (
        jnp.ones_like(density, dtype=bool)
        if composition is None
        else jnp.isfinite(jnp.broadcast_to(jnp.asarray(composition), density.shape))
    )
    return density, matter, radiation, magnetic, composition_finite


def _opacity_evidence(
    density: Array,
    matter: Array,
    radiation: Array,
    magnetic: Array,
    composition_finite: Array,
    coefficients: tuple[Array, ...],
    /,
    *,
    minimum_temperature: float,
    maximum_temperature: float,
) -> tuple[Array, Array, Array, Array]:
    finite = (
        jnp.isfinite(density)
        & jnp.isfinite(matter)
        & jnp.isfinite(radiation)
        & jnp.isfinite(magnetic)
        & composition_finite
        & jnp.all(jnp.stack(tuple(jnp.isfinite(value) for value in coefficients)), axis=0)
    )
    physical = (
        finite
        & (density >= 0.0)
        & (matter > 0.0)
        & (radiation > 0.0)
        & (magnetic >= 0.0)
        & jnp.all(jnp.stack(tuple(value >= 0.0 for value in coefficients)), axis=0)
    )
    supported = (
        physical
        & (matter >= minimum_temperature)
        & (matter <= maximum_temperature)
        & (radiation >= minimum_temperature)
        & (radiation <= maximum_temperature)
    )
    derivative = (
        supported & (matter > minimum_temperature) & (matter < maximum_temperature)
    )
    return finite, physical, supported, derivative


class ThermalBremsstrahlungGreyOpacityPlan(AbstractGRGreyOpacityPlan):
    """Parameterized thermal free-free grey emission and absorption."""

    scale: RelativityScaleContract
    emission_prefactor: float = eqx.field(static=True)
    rosseland_ratio: float = eqx.field(static=True)
    minimum_temperature: float = eqx.field(static=True)
    maximum_temperature: float = eqx.field(static=True)
    radiation_constant: float = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        /,
        *,
        emission_prefactor: float,
        rosseland_ratio: float = 0.033,
        minimum_temperature: float,
        maximum_temperature: float,
        radiation_constant: float = 1.0,
    ) -> None:
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be RelativityScaleContract.")
        values = tuple(
            float(value)
            for value in (
                emission_prefactor,
                rosseland_ratio,
                minimum_temperature,
                maximum_temperature,
                radiation_constant,
            )
        )
        if (
            any(not np.isfinite(value) or value <= 0.0 for value in values)
            or values[3] <= values[2]
        ):
            raise ValueError("Bremsstrahlung grey-opacity controls are invalid.")
        self.scale = scale
        self.emission_prefactor = values[0]
        self.rosseland_ratio = values[1]
        self.minimum_temperature = values[2]
        self.maximum_temperature = values[3]
        self.radiation_constant = values[4]
        self.opacity_id = canonical_fingerprint(
            {
                "kind": "thermal-bremsstrahlung-grey-opacity",
                "scale": scale.scale_id,
                "emission_prefactor": values[0],
                "rosseland_ratio": values[1],
                "temperature_support": values[2:4],
                "radiation_constant": values[4],
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
    ) -> GRGreyOpacityEvaluation:
        density, matter, radiation, magnetic, composition_finite = _fields(
            rest_mass_density,
            matter_temperature,
            radiation_temperature,
            magnetic_squared,
            composition,
        )
        tiny = jnp.finfo(jnp.result_type(density, matter)).tiny
        safe_matter = jnp.maximum(matter, tiny)
        safe_radiation = jnp.maximum(radiation, tiny)
        emission = self.emission_prefactor * density**2 * safe_matter ** (-3.5)
        absorption = emission * (safe_matter / safe_radiation) ** 3
        transport = self.rosseland_ratio * emission
        zero = jnp.zeros_like(emission)
        boltzmann = jnp.asarray(float(self.scale.boltzmann_constant), emission.dtype)
        equilibrium = self.radiation_constant * safe_matter**4
        photon_emission = (
            emission * equilibrium / jnp.maximum(2.70118 * boltzmann * safe_matter, tiny)
        )
        coefficients = (
            emission,
            absorption,
            transport,
            zero,
            absorption,
            photon_emission,
            zero,
        )
        finite, physical, supported, derivative = _opacity_evidence(
            density,
            matter,
            radiation,
            magnetic,
            composition_finite,
            coefficients,
            minimum_temperature=self.minimum_temperature,
            maximum_temperature=self.maximum_temperature,
        )
        return GRGreyOpacityEvaluation(
            *coefficients,
            finite,
            physical,
            supported,
            derivative,
            self.opacity_id,
        )


class ThermalSynchrotronGreyOpacityPlan(AbstractGRGreyOpacityPlan):
    """Thermal synchrotron grey source tied to local magnetic energy."""

    scale: RelativityScaleContract
    electron_mass_per_particle: float = eqx.field(static=True)
    emission_prefactor: float = eqx.field(static=True)
    rosseland_ratio: float = eqx.field(static=True)
    minimum_temperature: float = eqx.field(static=True)
    maximum_temperature: float = eqx.field(static=True)
    radiation_constant: float = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        /,
        *,
        electron_mass_per_particle: float,
        emission_prefactor: float,
        rosseland_ratio: float = 1.0,
        minimum_temperature: float,
        maximum_temperature: float,
        radiation_constant: float = 1.0,
    ) -> None:
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be RelativityScaleContract.")
        values = tuple(
            float(value)
            for value in (
                electron_mass_per_particle,
                emission_prefactor,
                rosseland_ratio,
                minimum_temperature,
                maximum_temperature,
                radiation_constant,
            )
        )
        if (
            any(not np.isfinite(value) or value <= 0.0 for value in values)
            or values[4] <= values[3]
        ):
            raise ValueError("Synchrotron grey-opacity controls are invalid.")
        self.scale = scale
        self.electron_mass_per_particle = values[0]
        self.emission_prefactor = values[1]
        self.rosseland_ratio = values[2]
        self.minimum_temperature = values[3]
        self.maximum_temperature = values[4]
        self.radiation_constant = values[5]
        self.opacity_id = canonical_fingerprint(
            {
                "kind": "thermal-synchrotron-grey-opacity",
                "scale": scale.scale_id,
                "electron_mass_per_particle": values[0],
                "emission_prefactor": values[1],
                "rosseland_ratio": values[2],
                "temperature_support": values[3:5],
                "radiation_constant": values[5],
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
    ) -> GRGreyOpacityEvaluation:
        density, matter, radiation, magnetic, composition_finite = _fields(
            rest_mass_density,
            matter_temperature,
            radiation_temperature,
            magnetic_squared,
            composition,
        )
        tiny = jnp.finfo(jnp.result_type(density, matter)).tiny
        safe_matter = jnp.maximum(matter, tiny)
        safe_radiation = jnp.maximum(radiation, tiny)
        electron_number = density / self.electron_mass_per_particle
        emissivity = self.emission_prefactor * electron_number * magnetic * safe_matter**2
        equilibrium = self.radiation_constant * safe_matter**4
        emission = emissivity / jnp.maximum(equilibrium, tiny)
        absorption = emission * (safe_matter / safe_radiation) ** 3
        transport = self.rosseland_ratio * absorption
        zero = jnp.zeros_like(emission)
        boltzmann = jnp.asarray(float(self.scale.boltzmann_constant), emission.dtype)
        photon_emission = emissivity / jnp.maximum(
            2.70118 * boltzmann * safe_matter, tiny
        )
        coefficients = (
            emission,
            absorption,
            transport,
            zero,
            absorption,
            photon_emission,
            zero,
        )
        finite, physical, supported, derivative = _opacity_evidence(
            density,
            matter,
            radiation,
            magnetic,
            composition_finite,
            coefficients,
            minimum_temperature=self.minimum_temperature,
            maximum_temperature=self.maximum_temperature,
        )
        return GRGreyOpacityEvaluation(
            *coefficients,
            finite,
            physical,
            supported,
            derivative,
            self.opacity_id,
        )


class KleinNishinaScatteringPlan(AbstractGRGreyOpacityPlan):
    """Electron scattering with a bounded Klein-Nishina temperature reduction."""

    electron_mass_per_particle: float = eqx.field(static=True)
    thomson_cross_section: float = eqx.field(static=True)
    klein_nishina_temperature: float = eqx.field(static=True)
    compton_fraction: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        electron_mass_per_particle: float,
        thomson_cross_section: float,
        klein_nishina_temperature: float,
        compton_fraction: float = 1.0,
    ) -> None:
        values = tuple(
            float(value)
            for value in (
                electron_mass_per_particle,
                thomson_cross_section,
                klein_nishina_temperature,
                compton_fraction,
            )
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Klein-Nishina scattering controls are invalid.")
        self.electron_mass_per_particle = values[0]
        self.thomson_cross_section = values[1]
        self.klein_nishina_temperature = values[2]
        self.compton_fraction = values[3]
        self.opacity_id = canonical_fingerprint(
            {
                "kind": "klein-nishina-grey-scattering",
                "electron_mass_per_particle": values[0],
                "thomson_cross_section": values[1],
                "klein_nishina_temperature": values[2],
                "compton_fraction": values[3],
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
    ) -> GRGreyOpacityEvaluation:
        density, matter, radiation, magnetic, composition_finite = _fields(
            rest_mass_density,
            matter_temperature,
            radiation_temperature,
            magnetic_squared,
            composition,
        )
        electron_number = density / self.electron_mass_per_particle
        reduction = (1.0 + radiation / self.klein_nishina_temperature) ** (-0.86)
        scattering = electron_number * self.thomson_cross_section * reduction
        compton = self.compton_fraction * scattering
        zero = jnp.zeros_like(scattering)
        coefficients = (zero, zero, zero, scattering, zero, zero, compton)
        finite, physical, supported, derivative = _opacity_evidence(
            density,
            matter,
            radiation,
            magnetic,
            composition_finite,
            coefficients,
            minimum_temperature=0.0,
            maximum_temperature=float("inf"),
        )
        return GRGreyOpacityEvaluation(
            *coefficients,
            finite,
            physical,
            supported,
            derivative,
            self.opacity_id,
        )


class GRPhotonNumberState(StrictModule):
    densitized_number: Array
    time: Array
    accepted_steps: Array


class GRPhotonNumberLedger(StrictModule):
    transport_change: Array
    emission_change: Array
    absorption_change: Array
    total_change: Array
    balance_defect: Array
    minimum_number: Array
    finite: Array
    qualified: Array
    plan_id: str = eqx.field(static=True)


class GRPhotonNumberResult(StrictModule):
    candidate: GRPhotonNumberState
    state: GRPhotonNumberState
    radiation_temperature: Array
    ledger: GRPhotonNumberLedger
    accepted: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class GRPhotonNumberPlan(StrictModule, NonTrainableState):
    """Conservative M1-aligned photon transport plus implicit local production."""

    discretization: FiniteVolumeDiscretization
    scale: RelativityScaleContract
    minimum_number: float = eqx.field(static=True)
    balance_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteVolumeDiscretization,
        scale: RelativityScaleContract,
        /,
        *,
        minimum_number: float = 1.0e-30,
        balance_tolerance: float = 1.0e-9,
    ) -> None:
        if not isinstance(discretization, FiniteVolumeDiscretization):
            raise TypeError("discretization must be FiniteVolumeDiscretization.")
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be RelativityScaleContract.")
        minimum = float(minimum_number)
        tolerance = float(balance_tolerance)
        if (
            not np.isfinite(minimum)
            or minimum <= 0.0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("Photon-number controls are invalid.")
        self.discretization = discretization
        self.scale = scale
        self.minimum_number = minimum
        self.balance_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gr-photon-number-transport",
                "discretization": discretization.prepared_id,
                "scale": scale.scale_id,
                "minimum_number": minimum,
                "balance_tolerance": tolerance,
            }
        )

    def initialize(
        self,
        number_density: ArrayLike,
        geometry: ValenciaFiniteVolumeStageGeometry,
        /,
        *,
        time: ArrayLike = 0.0,
    ) -> GRPhotonNumberState:
        number = jnp.asarray(number_density)
        if number.shape != tuple(self.discretization.cell_shape):
            raise ValueError("Photon number density must match the finite-volume grid.")
        number = eqx.error_if(
            number,
            jnp.any(~jnp.isfinite(number) | (number < self.minimum_number)),
            "Initial photon number is invalid.",
        )
        return GRPhotonNumberState(
            geometry.cell.sqrt_det_spatial_metric * number,
            jnp.asarray(time, dtype=number.dtype).reshape(()),
            jnp.zeros((), dtype=jnp.int32),
        )

    def advance(
        self,
        state: GRPhotonNumberState,
        radiation_state: ArrayLike,
        opacity: GRGreyOpacityEvaluation,
        geometry: ValenciaFiniteVolumeStageGeometry,
        step_size: ArrayLike,
        /,
    ) -> GRPhotonNumberResult:
        if not isinstance(state, GRPhotonNumberState):
            raise TypeError("state must be GRPhotonNumberState.")
        radiation = jnp.asarray(radiation_state)
        shape = tuple(self.discretization.cell_shape)
        if radiation.shape != shape + (4,) or state.densitized_number.shape != shape:
            raise ValueError("Photon and radiation states must match the grid.")
        step = jnp.asarray(step_size, dtype=radiation.dtype).reshape(())
        cell_volume_density = geometry.cell.sqrt_det_spatial_metric
        number = state.densitized_number / cell_volume_density
        moments = radiation / cell_volume_density[..., None]
        energy = moments[..., 0]
        flux_covector = moments[..., 1:]
        flux_vector = ein.contract(
            "...ij,...j->...i",
            geometry.cell.inverse_spatial_metric,
            flux_covector,
        )
        physical_speed = jnp.asarray(float(self.scale.speed_of_light), radiation.dtype)
        transport_velocity = (
            geometry.cell.alpha[..., None]
            * flux_vector
            / jnp.maximum(physical_speed * energy, jnp.finfo(radiation.dtype).tiny)[
                ..., None
            ]
            - geometry.cell.beta_contravariant
        )
        residual = jnp.zeros_like(number)
        boundary_flux = jnp.asarray(0.0, dtype=number.dtype)
        volumes = self.discretization.cell_volumes.astype(number.dtype)
        for axis in range(len(shape)):
            periodic = self.discretization.grid.structured_axes[axis].periodic
            if periodic:
                left_number = number
                right_number = jnp.roll(number, -1, axis=axis)
                face_velocity = 0.5 * (
                    transport_velocity[..., axis]
                    + jnp.roll(transport_velocity[..., axis], -1, axis=axis)
                )
            else:
                lower = jnp.take(number, jnp.asarray([0]), axis=axis)
                upper = jnp.take(number, jnp.asarray([number.shape[axis] - 1]), axis=axis)
                left_number = jnp.concatenate((lower, number), axis=axis)
                right_number = jnp.concatenate((number, upper), axis=axis)
                velocity = transport_velocity[..., axis]
                velocity_lower = jnp.take(velocity, jnp.asarray([0]), axis=axis)
                velocity_upper = jnp.take(
                    velocity, jnp.asarray([velocity.shape[axis] - 1]), axis=axis
                )
                velocity_interior = 0.5 * (
                    jnp.take(velocity, jnp.arange(velocity.shape[axis] - 1), axis=axis)
                    + jnp.take(velocity, jnp.arange(1, velocity.shape[axis]), axis=axis)
                )
                face_velocity = jnp.concatenate(
                    (velocity_lower, velocity_interior, velocity_upper), axis=axis
                )
            upwind = jnp.where(face_velocity >= 0.0, left_number, right_number)
            flux = geometry.faces[axis].sqrt_det_spatial_metric * face_velocity * upwind
            integrated = flux * self.discretization.face_measures[axis]
            if periodic:
                residual = (
                    residual - (integrated - jnp.roll(integrated, 1, axis=axis)) / volumes
                )
            else:
                lower_indices = jnp.arange(integrated.shape[axis] - 1)
                upper_indices = jnp.arange(1, integrated.shape[axis])
                lower_flux = jnp.take(integrated, lower_indices, axis=axis)
                upper_flux = jnp.take(integrated, upper_indices, axis=axis)
                residual = residual - (upper_flux - lower_flux) / volumes
                outward = jnp.take(
                    integrated, integrated.shape[axis] - 1, axis=axis
                ) - jnp.take(integrated, 0, axis=axis)
                boundary_flux = boundary_flux + jnp.sum(outward)
        transported = state.densitized_number + step * residual
        lapse = geometry.cell.alpha
        emission_increment = (
            step * lapse * cell_volume_density * opacity.photon_emission_rate
        )
        denominator = 1.0 + step * lapse * opacity.photon_absorption
        candidate_number = (transported + emission_increment) / denominator
        absorption_change = candidate_number - transported - emission_increment
        total_change = candidate_number - state.densitized_number
        transport_change = step * residual
        balance_defect = (
            jnp.sum(volumes * total_change)
            + step * boundary_flux
            - jnp.sum(volumes * (emission_increment + absorption_change))
        )
        finite = (
            jnp.all(jnp.isfinite(candidate_number))
            & jnp.isfinite(balance_defect)
            & opacity.finite.all()
        )
        physical = finite & jnp.all(candidate_number > 0.0)
        scale = jnp.maximum(jnp.max(jnp.abs(total_change), initial=0.0), 1.0)
        tolerance = jnp.maximum(
            jnp.asarray(self.balance_tolerance, number.dtype),
            256.0 * jnp.finfo(number.dtype).eps * scale,
        )
        qualified = (
            physical & opacity.qualified.all() & (jnp.abs(balance_defect) <= tolerance)
        )
        accepted = qualified
        candidate = GRPhotonNumberState(
            candidate_number,
            state.time + step,
            state.accepted_steps + jnp.asarray(1, dtype=jnp.int32),
        )
        accepted_state = GRPhotonNumberState(
            jnp.where(accepted, candidate_number, state.densitized_number),
            jnp.where(accepted, state.time + step, state.time),
            state.accepted_steps + accepted.astype(jnp.int32),
        )
        local_number = candidate_number / cell_volume_density
        boltzmann = jnp.asarray(float(self.scale.boltzmann_constant), number.dtype)
        radiation_temperature = energy / jnp.maximum(
            2.70118 * boltzmann * local_number,
            jnp.finfo(number.dtype).tiny,
        )
        ledger = GRPhotonNumberLedger(
            transport_change,
            emission_increment,
            absorption_change,
            total_change,
            balance_defect,
            jnp.min(local_number),
            finite,
            qualified,
            self.plan_id,
        )
        return GRPhotonNumberResult(
            candidate,
            accepted_state,
            radiation_temperature,
            ledger,
            accepted,
            finite,
            physical,
            qualified,
            qualified & opacity.derivative_valid.all(),
        )


__all__ = [
    "GRPhotonNumberLedger",
    "GRPhotonNumberPlan",
    "GRPhotonNumberResult",
    "GRPhotonNumberState",
    "KleinNishinaScatteringPlan",
    "ThermalBremsstrahlungGreyOpacityPlan",
    "ThermalSynchrotronGreyOpacityPlan",
]
