#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax import ein

from ..._fingerprint import canonical_fingerprint
from ..._physical import ElectromagneticScaleContract, RelativityScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume import FiniteVolumeDiscretization
from ...electromagnetics import (
    GrayMeanOpacities,
    ThermalFreeFreeModel,
    ThermalSynchrotronModel,
)
from ...equations._relativistic_radiation_interaction import (
    AbstractGRGrayOpacityPlan,
    GRGrayOpacityEvaluation,
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
        jnp.ones_like(density, dtype=jnp.bool_)
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


_PLANCK_MEAN_PHOTON_ENERGY = 2.70118
"""Mean blackbody photon energy ``π⁴/(30 ζ(3))`` in units of ``k T``."""


def _radiation_constant(scale: ElectromagneticScaleContract, /) -> float:
    """``a = π² k⁴/(15 ħ³ c³)`` in the scale's energy-density-per-kelvin⁴ unit."""
    k = float(scale.relativity.boltzmann_constant)
    hbar = float(scale.reduced_planck_constant)
    c = float(scale.speed_of_light)
    return float(
        np.exp(
            2.0 * np.log(np.pi) + 4.0 * np.log(k) - np.log(15.0) - 3.0 * np.log(hbar * c)
        )
    )


def _gray_evaluation(
    means: GrayMeanOpacities,
    density: Array,
    matter: Array,
    radiation: Array,
    magnetic: Array,
    composition_finite: Array,
    scale: ElectromagneticScaleContract,
    opacity_id: str,
    /,
) -> GRGrayOpacityEvaluation:
    """Gray closure from spectral means; photon number uses the blackbody mean energy."""
    tiny = jnp.finfo(matter.dtype).tiny
    emission = means.planck_emission
    absorption = means.planck_absorption
    boltzmann = float(scale.relativity.boltzmann_constant)
    photon_emission = (
        emission
        * _radiation_constant(scale)
        * matter**4
        / jnp.maximum(_PLANCK_MEAN_PHOTON_ENERGY * boltzmann * matter, tiny)
    )
    zero = jnp.zeros_like(emission)
    coefficients = (
        emission,
        absorption,
        means.rosseland,
        zero,
        absorption,
        photon_emission,
        zero,
    )
    finite, physical, _, _ = _opacity_evidence(
        density,
        matter,
        radiation,
        magnetic,
        composition_finite,
        coefficients,
        minimum_temperature=0.0,
        maximum_temperature=float("inf"),
    )
    qualified = (
        physical
        & means.emission_supported
        & means.absorption_supported
        & means.rosseland_supported
    )
    return GRGrayOpacityEvaluation(
        *coefficients,
        finite,
        physical,
        qualified,
        qualified,
        opacity_id,
    )


class ThermalBremsstrahlungGrayOpacityPlan(AbstractGRGrayOpacityPlan):
    """Thermal free–free Planck and Rosseland means of the spectral owner.

    ``rest_mass_density / electron_mass_per_particle`` is the electron density and
    ions of charge ``ion_charge_number`` neutralize it. Temperatures are in
    kelvin; coefficients are per length of ``scale``. Qualification is the
    spectral support of every mean (`ThermalFreeFreeModel.gray_means`).
    """

    model: ThermalFreeFreeModel
    electron_mass_per_particle: float = eqx.field(static=True)

    def __init__(
        self,
        scale: ElectromagneticScaleContract,
        /,
        *,
        electron_mass_per_particle: float,
        ion_charge_number: float = 1.0,
    ) -> None:
        if not isinstance(scale, ElectromagneticScaleContract):
            raise TypeError("scale must be ElectromagneticScaleContract.")
        mass = float(electron_mass_per_particle)
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("electron_mass_per_particle must be finite and positive.")
        model = ThermalFreeFreeModel(scale, ion_charge_number=ion_charge_number)
        self.model = model
        self.electron_mass_per_particle = mass
        self.opacity_id = canonical_fingerprint(
            {
                "kind": "thermal-bremsstrahlung-gray-opacity",
                "model": model.model_id,
                "electron_mass_per_particle": mass,
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
        density, matter, radiation, magnetic, composition_finite = _fields(
            rest_mass_density,
            matter_temperature,
            radiation_temperature,
            magnetic_squared,
            composition,
        )
        electrons = density / self.electron_mass_per_particle
        means = self.model.gray_means(
            electrons,
            electrons / self.model.ion_charge_number,
            matter,
            radiation,
        )
        return _gray_evaluation(
            means,
            density,
            matter,
            radiation,
            magnetic,
            composition_finite,
            self.model.scale,
            self.opacity_id,
        )


class ThermalSynchrotronGrayOpacityPlan(AbstractGRGrayOpacityPlan):
    """Thermal synchrotron Planck and Rosseland means of the MNY96 spectral route.

    ``magnetic_squared`` is ``b² = B²/μ₀`` (twice the magnetic pressure) in the
    scale's energy-density unit. Fields are converted to SI through the scale's
    SI-referenced units, the SI `ThermalSynchrotronModel` means are taken, and the
    coefficients are returned per length of ``scale``. The synchrotron Rosseland
    mean lies outside the MNY96 frequency support whenever ``hν_s ≪ kT`` and is
    then unqualified.
    """

    scale: ElectromagneticScaleContract = eqx.field(static=True)
    model: ThermalSynchrotronModel
    electron_mass_per_particle: float = eqx.field(static=True)

    def __init__(
        self,
        scale: ElectromagneticScaleContract,
        /,
        *,
        electron_mass_per_particle: float,
    ) -> None:
        if not isinstance(scale, ElectromagneticScaleContract):
            raise TypeError("scale must be ElectromagneticScaleContract.")
        if scale.charge_unit.reference_system_id != "si":
            raise ValueError("scale units must be referenced to the SI system.")
        mass = float(electron_mass_per_particle)
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("electron_mass_per_particle must be finite and positive.")
        model = ThermalSynchrotronModel()
        self.scale = scale
        self.model = model
        self.electron_mass_per_particle = mass
        self.opacity_id = canonical_fingerprint(
            {
                "kind": "thermal-synchrotron-gray-opacity",
                "scale": scale.scale_id,
                "model": model.model_id,
                "electron_mass_per_particle": mass,
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
        density, matter, radiation, magnetic, composition_finite = _fields(
            rest_mass_density,
            matter_temperature,
            radiation_temperature,
            magnetic_squared,
            composition,
        )
        units = self.scale.unit_si_map()
        length_si = units["length"][0]
        field_si = units["magnetic_field"][0]
        permeability = float(self.scale.vacuum_permeability)
        electrons_si = density / self.electron_mass_per_particle / length_si**3
        field_tesla = jnp.sqrt(permeability * jnp.maximum(magnetic, 0.0)) * field_si
        means = self.model.gray_means(electrons_si, matter, field_tesla, radiation)
        means = GrayMeanOpacities(
            planck_emission=means.planck_emission * length_si,
            planck_absorption=means.planck_absorption * length_si,
            rosseland=means.rosseland * length_si,
            edge_fraction=means.edge_fraction,
            quadrature_error=means.quadrature_error,
            emission_supported=means.emission_supported,
            absorption_supported=means.absorption_supported,
            rosseland_supported=means.rosseland_supported,
        )
        return _gray_evaluation(
            means,
            density,
            matter,
            radiation,
            magnetic,
            composition_finite,
            self.scale,
            self.opacity_id,
        )


class KleinNishinaScatteringPlan(AbstractGRGrayOpacityPlan):
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
                "kind": "klein-nishina-gray-scattering",
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
    ) -> GRGrayOpacityEvaluation:
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
        return GRGrayOpacityEvaluation(
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
        opacity: GRGrayOpacityEvaluation,
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
    "ThermalBremsstrahlungGrayOpacityPlan",
    "ThermalSynchrotronGrayOpacityPlan",
]
