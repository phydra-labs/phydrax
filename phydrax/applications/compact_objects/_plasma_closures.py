#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import RelativityScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class PlasmaKineticRegime(StrEnum):
    """Declared kinetic content of a compact-object plasma closure."""

    TWO_TEMPERATURE_CALORIC = "two-temperature-caloric"
    ISOTROPIC_NONTHERMAL = "isotropic-nonthermal"
    ANISOTROPIC_GYROTROPIC = "anisotropic-gyrotropic"
    FULL_PHASE_SPACE = "full-phase-space"


class TwoTemperaturePlasmaState(StrictModule):
    electron_temperature: Array
    ion_temperature: Array


class TwoTemperatureExchangeResult(StrictModule):
    candidate: TwoTemperaturePlasmaState
    electron_energy_exchange: Array
    ion_energy_exchange: Array
    conservation_residual: Array
    equilibrium_temperature: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    closure_id: str = eqx.field(static=True)


class TwoTemperatureElectronIonClosure(StrictModule, NonTrainableState):
    """Exact electron-ion caloric relaxation with a closed energy ledger.

    Each species is an ideal caloric population with volumetric heat capacity
    ``n k_B / (gamma - 1)``.  The declared equilibration time is the e-folding
    time of ``T_e - T_i``.  This is a fluid closure: pressure anisotropy, pair
    creation and velocity-space kinetics are deliberately outside its support.
    """

    scale: RelativityScaleContract
    electron_adiabatic_index: float = eqx.field(static=True)
    ion_adiabatic_index: float = eqx.field(static=True)
    equilibration_time: float = eqx.field(static=True)
    minimum_temperature: float = eqx.field(static=True)
    maximum_temperature: float = eqx.field(static=True)
    regime: PlasmaKineticRegime = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        /,
        *,
        electron_adiabatic_index: float = 5.0 / 3.0,
        ion_adiabatic_index: float = 5.0 / 3.0,
        equilibration_time: float,
        minimum_temperature: float,
        maximum_temperature: float,
        regime: PlasmaKineticRegime = PlasmaKineticRegime.TWO_TEMPERATURE_CALORIC,
    ) -> None:
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be RelativityScaleContract.")
        if not isinstance(regime, PlasmaKineticRegime):
            raise TypeError("regime must be PlasmaKineticRegime.")
        if regime is not PlasmaKineticRegime.TWO_TEMPERATURE_CALORIC:
            raise NotImplementedError(
                "Two-temperature fluid exchange does not support anisotropic or "
                "velocity-space kinetic regimes."
            )
        gamma_e = float(electron_adiabatic_index)
        gamma_i = float(ion_adiabatic_index)
        timescale = float(equilibration_time)
        lower = float(minimum_temperature)
        upper = float(maximum_temperature)
        if (
            not np.isfinite(gamma_e)
            or gamma_e <= 1.0
            or not np.isfinite(gamma_i)
            or gamma_i <= 1.0
            or not np.isfinite(timescale)
            or timescale <= 0.0
            or not np.isfinite(lower)
            or lower <= 0.0
            or not np.isfinite(upper)
            or upper <= lower
        ):
            raise ValueError("Two-temperature closure parameters are invalid.")
        self.scale = scale
        self.electron_adiabatic_index = gamma_e
        self.ion_adiabatic_index = gamma_i
        self.equilibration_time = timescale
        self.minimum_temperature = lower
        self.maximum_temperature = upper
        self.regime = regime
        self.closure_id = canonical_fingerprint(
            {
                "kind": "two-temperature-electron-ion-caloric-closure",
                "scale": scale.scale_id,
                "electron_adiabatic_index": gamma_e,
                "ion_adiabatic_index": gamma_i,
                "equilibration_time": timescale,
                "temperature_support": (lower, upper),
                "regime": regime.value,
            }
        )

    def advance(
        self,
        electron_number_density: ArrayLike,
        ion_number_density: ArrayLike,
        state: TwoTemperaturePlasmaState,
        step_size: ArrayLike,
        /,
    ) -> TwoTemperatureExchangeResult:
        if not isinstance(state, TwoTemperaturePlasmaState):
            raise TypeError("state must be TwoTemperaturePlasmaState.")
        electron_density, ion_density, electron_temperature, ion_temperature, step = (
            jnp.broadcast_arrays(
                jnp.asarray(electron_number_density),
                jnp.asarray(ion_number_density),
                jnp.asarray(state.electron_temperature),
                jnp.asarray(state.ion_temperature),
                jnp.asarray(step_size),
            )
        )
        dtype = jnp.result_type(
            electron_density,
            ion_density,
            electron_temperature,
            ion_temperature,
            step,
        )
        electron_density = electron_density.astype(dtype)
        ion_density = ion_density.astype(dtype)
        electron_temperature = electron_temperature.astype(dtype)
        ion_temperature = ion_temperature.astype(dtype)
        step = step.astype(dtype)
        boltzmann = jnp.asarray(float(self.scale.boltzmann_constant), dtype=dtype)
        electron_capacity = (
            electron_density * boltzmann / (self.electron_adiabatic_index - 1.0)
        )
        ion_capacity = ion_density * boltzmann / (self.ion_adiabatic_index - 1.0)
        total_capacity = electron_capacity + ion_capacity
        safe_capacity = jnp.where(total_capacity > 0.0, total_capacity, 1.0)
        equilibrium = (
            electron_capacity * electron_temperature + ion_capacity * ion_temperature
        ) / safe_capacity
        decay = jnp.exp(-step / self.equilibration_time)
        electron_candidate = equilibrium + (electron_temperature - equilibrium) * decay
        electron_exchange = electron_capacity * (
            electron_candidate - electron_temperature
        )
        ion_exchange = -electron_exchange
        safe_ion_capacity = jnp.where(ion_capacity > 0.0, ion_capacity, 1.0)
        ion_candidate = ion_temperature + ion_exchange / safe_ion_capacity
        residual = electron_exchange + ion_exchange
        finite = (
            jnp.isfinite(electron_density)
            & jnp.isfinite(ion_density)
            & jnp.isfinite(electron_temperature)
            & jnp.isfinite(ion_temperature)
            & jnp.isfinite(step)
            & jnp.isfinite(electron_candidate)
            & jnp.isfinite(ion_candidate)
            & jnp.isfinite(residual)
        )
        physically_valid = (
            finite
            & (electron_density > 0.0)
            & (ion_density > 0.0)
            & (electron_temperature >= self.minimum_temperature)
            & (electron_temperature <= self.maximum_temperature)
            & (ion_temperature >= self.minimum_temperature)
            & (ion_temperature <= self.maximum_temperature)
            & (step >= 0.0)
        )
        candidate_in_support = (
            (electron_candidate >= self.minimum_temperature)
            & (electron_candidate <= self.maximum_temperature)
            & (ion_candidate >= self.minimum_temperature)
            & (ion_candidate <= self.maximum_temperature)
        )
        balance_scale = jnp.maximum(
            jnp.maximum(jnp.abs(electron_exchange), jnp.abs(ion_exchange)),
            jnp.asarray(1.0, dtype=dtype),
        )
        tolerance = 64.0 * jnp.finfo(dtype).eps * balance_scale
        converged = finite & (jnp.abs(residual) <= tolerance)
        qualified = physically_valid & candidate_in_support & converged
        derivative_valid = qualified
        return TwoTemperatureExchangeResult(
            TwoTemperaturePlasmaState(electron_candidate, ion_candidate),
            electron_exchange,
            ion_exchange,
            residual,
            equilibrium,
            finite,
            converged,
            physically_valid,
            qualified,
            derivative_valid,
            self.closure_id,
        )


class NonthermalParticleMoments(StrictModule):
    number_density: Array
    kinetic_energy_density: Array
    isotropic_pressure: Array
    mean_lorentz_factor: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    distribution_id: str = eqx.field(static=True)


class BoundedNonthermalParticleDistribution(StrictModule, NonTrainableState):
    """Fixed-bin isotropic nonthermal population on bounded Lorentz-factor support.

    ``bin_number_density`` stores bin-integrated particle number density, avoiding
    an implicit quadrature convention.  The geometric bin centers define the
    immutable moment rule.  Pitch-angle anisotropy and full phase-space kinetics
    are rejected rather than silently approximated.
    """

    scale: RelativityScaleContract
    lorentz_factor_edges: Array
    bin_number_density: Array
    particle_rest_energy: float = eqx.field(static=True)
    species: str = eqx.field(static=True)
    regime: PlasmaKineticRegime = eqx.field(static=True)
    distribution_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        lorentz_factor_edges: ArrayLike,
        bin_number_density: ArrayLike,
        /,
        *,
        particle_rest_energy: float,
        species: str,
        regime: PlasmaKineticRegime = PlasmaKineticRegime.ISOTROPIC_NONTHERMAL,
    ) -> None:
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be RelativityScaleContract.")
        if not isinstance(regime, PlasmaKineticRegime):
            raise TypeError("regime must be PlasmaKineticRegime.")
        if regime is not PlasmaKineticRegime.ISOTROPIC_NONTHERMAL:
            raise NotImplementedError(
                "Bounded nonthermal records support isotropic distributions only; "
                "anisotropic and full phase-space kinetics require a kinetic solver."
            )
        edges = np.asarray(lorentz_factor_edges, dtype=float)
        density = np.asarray(bin_number_density, dtype=float)
        rest_energy = float(particle_rest_energy)
        species_ = str(species).strip()
        if (
            edges.ndim != 1
            or edges.size < 2
            or density.ndim < 1
            or density.shape[-1] != edges.size - 1
            or np.any(~np.isfinite(edges))
            or edges[0] < 1.0
            or np.any(np.diff(edges) <= 0.0)
            or np.any(~np.isfinite(density))
            or np.any(density < 0.0)
            or not np.isfinite(rest_energy)
            or rest_energy <= 0.0
            or not species_
        ):
            raise ValueError("Bounded nonthermal distribution data are invalid.")
        self.scale = scale
        self.lorentz_factor_edges = jnp.asarray(edges)
        self.bin_number_density = jnp.asarray(density)
        self.particle_rest_energy = rest_energy
        self.species = species_
        self.regime = regime
        self.distribution_id = canonical_fingerprint(
            {
                "kind": "bounded-isotropic-nonthermal-particle-distribution",
                "scale": scale.scale_id,
                "lorentz_factor_edges": array_tree_fingerprint(edges),
                "bin_number_density": array_tree_fingerprint(density),
                "particle_rest_energy": rest_energy,
                "species": species_,
                "regime": regime.value,
            }
        )

    @property
    def bin_count(self) -> int:
        return int(self.lorentz_factor_edges.size - 1)

    def moments(self, /) -> NonthermalParticleMoments:
        density = self.bin_number_density
        edges = self.lorentz_factor_edges.astype(density.dtype)
        gamma = jnp.sqrt(edges[:-1] * edges[1:])
        rest_energy = jnp.asarray(self.particle_rest_energy, dtype=density.dtype)
        number = jnp.sum(density, axis=-1)
        kinetic = rest_energy * jnp.sum(density * (gamma - 1.0), axis=-1)
        pressure = rest_energy / 3.0 * jnp.sum(density * (gamma - 1.0 / gamma), axis=-1)
        safe_number = jnp.where(number > 0.0, number, 1.0)
        mean_gamma = jnp.sum(density * gamma, axis=-1) / safe_number
        mean_gamma = jnp.where(number > 0.0, mean_gamma, 1.0)
        finite = (
            jnp.all(jnp.isfinite(density), axis=-1)
            & jnp.isfinite(number)
            & jnp.isfinite(kinetic)
            & jnp.isfinite(pressure)
            & jnp.isfinite(mean_gamma)
        )
        physically_valid = (
            finite
            & jnp.all(density >= 0.0, axis=-1)
            & (number >= 0.0)
            & (kinetic >= 0.0)
            & (pressure >= 0.0)
            & (mean_gamma >= 1.0)
            & (mean_gamma <= edges[-1])
        )
        qualified = physically_valid
        derivative_valid = qualified & (number > 0.0)
        return NonthermalParticleMoments(
            number,
            kinetic,
            pressure,
            mean_gamma,
            finite,
            physically_valid,
            qualified,
            derivative_valid,
            self.distribution_id,
        )
