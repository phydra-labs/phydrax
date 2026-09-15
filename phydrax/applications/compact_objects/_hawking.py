#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualified Unruh-state Hawking fluxes and bounded Kerr evaporation."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from enum import IntEnum
from fractions import Fraction
from typing import Literal, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._physical import RelativityScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract


QuantumStatistics = Literal["boson", "fermion"]


class QuantumFieldSpecies(StrictModule, NonTrainableState):
    """Static identity and multiplicity of one free quantum field species.

    ``rest_mass_frequency`` is the Compton wavenumber in the inverse-length
    convention used by the scattering problem. ``multiplicity`` counts internal
    states not already represented as separate angular modes.
    """

    species_id: str = eqx.field(static=True)
    spin: float = eqx.field(static=True)
    statistics: QuantumStatistics = eqx.field(static=True)
    multiplicity: int = eqx.field(static=True)
    rest_mass_frequency: float = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        species_id: str,
        spin: float,
        statistics: QuantumStatistics,
        /,
        *,
        multiplicity: int = 1,
        rest_mass_frequency: float = 0.0,
    ):
        identifier = str(species_id).strip()
        spin_ = float(spin)
        twice_spin = round(2.0 * spin_)
        multiplicity_ = int(multiplicity)
        rest_mass_ = float(rest_mass_frequency)
        if not identifier:
            raise ValueError("Quantum species_id must be non-empty.")
        if (
            not math.isfinite(spin_)
            or spin_ < 0.0
            or abs(2.0 * spin_ - twice_spin) > 1.0e-12
        ):
            raise ValueError(
                "Quantum spin must be a nonnegative integer or half-integer."
            )
        if statistics not in ("boson", "fermion"):
            raise ValueError("Quantum statistics must be 'boson' or 'fermion'.")
        expected = "boson" if twice_spin % 2 == 0 else "fermion"
        if statistics != expected:
            raise ValueError(
                "Quantum statistics must obey the spin-statistics assignment."
            )
        if (
            isinstance(multiplicity, bool)
            or multiplicity_ != multiplicity
            or multiplicity_ < 1
        ):
            raise ValueError("Quantum multiplicity must be a positive integer.")
        if not math.isfinite(rest_mass_) or rest_mass_ < 0.0:
            raise ValueError("rest_mass_frequency must be finite and nonnegative.")
        self.species_id = identifier
        self.spin = spin_
        self.statistics = statistics
        self.multiplicity = multiplicity_
        self.rest_mass_frequency = rest_mass_
        self.record_id = canonical_fingerprint(
            {
                "kind": "quantum-field-species",
                "species_id": identifier,
                "spin": spin_,
                "statistics": statistics,
                "multiplicity": multiplicity_,
                "rest_mass_frequency": rest_mass_,
            }
        )


class HawkingTailEvidence(StrictModule, NonTrainableState):
    """Qualified upper bounds on omitted frequency and angular-mode flux.

    Bounds are ordered as number, Killing energy, and axial angular momentum.
    Angular-momentum bounds constrain the absolute omitted contribution.
    """

    frequency_remainder_upper: Array
    mode_remainder_upper: Array
    qualified: Array
    derivative_valid: Array
    qualification_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        frequency_remainder_upper: ArrayLike,
        mode_remainder_upper: ArrayLike,
        /,
        *,
        qualified: ArrayLike,
        derivative_valid: ArrayLike,
        qualification_id: str,
    ):
        frequency = np.asarray(frequency_remainder_upper, dtype=float)
        modes = np.asarray(mode_remainder_upper, dtype=float)
        qualified_host = np.asarray(qualified, dtype=bool)
        derivative_host = np.asarray(derivative_valid, dtype=bool)
        identifier = str(qualification_id).strip()
        if frequency.shape != (3,) or modes.shape != (3,):
            raise ValueError("Hawking tail bounds must have shape (3,).")
        if qualified_host.shape != () or derivative_host.shape != ():
            raise ValueError("Tail qualification flags must be scalar.")
        if not identifier:
            raise ValueError("Tail qualification_id must be non-empty.")
        self.frequency_remainder_upper = jnp.asarray(frequency)
        self.mode_remainder_upper = jnp.asarray(modes)
        self.qualified = jnp.asarray(qualified_host)
        self.derivative_valid = jnp.asarray(derivative_host)
        self.qualification_id = identifier
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "hawking-tail-evidence",
                "frequency_remainder_upper": frequency,
                "mode_remainder_upper": modes,
                "qualified": qualified_host,
                "derivative_valid": derivative_host,
                "qualification_id": identifier,
            }
        )


class HawkingSpectrumPlan(StrictModule, NonTrainableState):
    """Fixed-capacity frequency and angular-mode quadrature for Hawking flux."""

    scale: RelativityScaleContract
    species: tuple[QuantumFieldSpecies, ...]
    angular_frequencies: Array
    frequency_weights: Array
    species_indices: Array
    polar_mode_numbers: Array
    azimuthal_mode_numbers: Array
    active_modes: Array
    bosonic_modes: Array
    multiplicities: Array
    rest_mass_frequencies: Array
    absolute_tail_tolerances: Array
    relative_tail_tolerance: float = eqx.field(static=True)
    greybody_tolerance: float = eqx.field(static=True)
    corotation_tolerance: float = eqx.field(static=True)
    temperature_to_wavenumber: float = eqx.field(static=True)
    mode_species_ids: tuple[str, ...] = eqx.field(static=True)
    mode_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        species: Sequence[QuantumFieldSpecies],
        angular_frequencies: ArrayLike,
        mode_species_ids: Sequence[str],
        polar_mode_numbers: ArrayLike,
        azimuthal_mode_numbers: ArrayLike,
        /,
        *,
        mode_ids: Sequence[str],
        active_modes: ArrayLike | None = None,
        number_tail_tolerance: float = 0.0,
        energy_tail_tolerance: float = 0.0,
        angular_momentum_tail_tolerance: float = 0.0,
        relative_tail_tolerance: float = 1.0e-4,
        greybody_tolerance: float = 1.0e-10,
        corotation_tolerance: float = 1.0e-8,
    ):
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be a RelativityScaleContract.")
        if not scale.quantum_constants_explicit:
            raise ValueError("Hawking flux requires explicit hbar and k_B constants.")
        species_ = tuple(species)
        if not species_ or any(
            not isinstance(item, QuantumFieldSpecies) for item in species_
        ):
            raise TypeError(
                "species must be a non-empty sequence of QuantumFieldSpecies."
            )
        species_ids = tuple(item.species_id for item in species_)
        if len(set(species_ids)) != len(species_ids):
            raise ValueError("Hawking species IDs must be unique.")

        frequencies = np.asarray(angular_frequencies, dtype=float)
        if (
            frequencies.ndim != 1
            or frequencies.size < 2
            or np.any(~np.isfinite(frequencies))
            or frequencies[0] < 0.0
            or np.any(np.diff(frequencies) <= 0.0)
        ):
            raise ValueError(
                "Hawking angular frequencies must be a finite increasing vector starting at zero or above."
            )
        species_by_mode = tuple(str(value).strip() for value in mode_species_ids)
        modes = tuple(str(value).strip() for value in mode_ids)
        polar = np.asarray(polar_mode_numbers, dtype=float)
        azimuthal = np.asarray(azimuthal_mode_numbers, dtype=float)
        mode_capacity = len(species_by_mode)
        if mode_capacity < 1:
            raise ValueError("Hawking quadrature requires at least one angular mode.")
        if len(modes) != mode_capacity or any(not value for value in modes):
            raise ValueError("mode_ids must provide one non-empty ID per mode slot.")
        if len(set(modes)) != len(modes):
            raise ValueError("Hawking mode IDs must be unique.")
        if polar.shape != (mode_capacity,) or azimuthal.shape != (mode_capacity,):
            raise ValueError("Angular quantum-number arrays must match mode capacity.")
        if np.any(~np.isfinite(polar)) or np.any(~np.isfinite(azimuthal)):
            raise ValueError("Angular quantum numbers must be finite.")
        doubled_polar = 2.0 * polar
        doubled_azimuthal = 2.0 * azimuthal
        if (
            np.any(np.abs(doubled_polar - np.round(doubled_polar)) > 1.0e-12)
            or np.any(np.abs(doubled_azimuthal - np.round(doubled_azimuthal)) > 1.0e-12)
            or np.any(polar < 0.0)
            or np.any(np.abs(azimuthal) > polar)
        ):
            raise ValueError(
                "Mode l and m must be admissible integer or half-integer quantum numbers."
            )
        active = (
            np.ones((mode_capacity,), dtype=bool)
            if active_modes is None
            else np.asarray(active_modes, dtype=bool)
        )
        if active.shape != (mode_capacity,) or not np.any(active):
            raise ValueError(
                "active_modes must match mode capacity and activate at least one mode."
            )

        lookup = {identifier: index for index, identifier in enumerate(species_ids)}
        if any(identifier not in lookup for identifier in species_by_mode):
            raise ValueError(
                "Every mode species ID must name a declared quantum species."
            )
        indices = np.asarray([lookup[value] for value in species_by_mode], dtype=np.int32)
        spin = np.asarray([species_[int(index)].spin for index in indices])
        parity_invalid = (np.abs((polar - spin) - np.round(polar - spin)) > 1.0e-12) | (
            np.abs((polar - azimuthal) - np.round(polar - azimuthal)) > 1.0e-12
        )
        if np.any(active & ((polar < spin) | parity_invalid)):
            raise ValueError(
                "Active modes require l >= spin and integral l-spin and l-m."
            )

        intervals = np.diff(frequencies)
        weights = np.empty_like(frequencies)
        weights[0] = 0.5 * intervals[0]
        weights[-1] = 0.5 * intervals[-1]
        if frequencies.size > 2:
            weights[1:-1] = 0.5 * (frequencies[2:] - frequencies[:-2])

        tolerances = np.asarray(
            (
                number_tail_tolerance,
                energy_tail_tolerance,
                angular_momentum_tail_tolerance,
            ),
            dtype=float,
        )
        relative = float(relative_tail_tolerance)
        greybody = float(greybody_tolerance)
        corotation = float(corotation_tolerance)
        if np.any(~np.isfinite(tolerances)) or np.any(tolerances < 0.0):
            raise ValueError(
                "Absolute Hawking tail tolerances must be finite and nonnegative."
            )
        if not math.isfinite(relative) or relative < 0.0:
            raise ValueError("relative_tail_tolerance must be finite and nonnegative.")
        if not math.isfinite(greybody) or greybody <= 0.0:
            raise ValueError("greybody_tolerance must be positive and finite.")
        if not math.isfinite(corotation) or corotation <= 0.0:
            raise ValueError("corotation_tolerance must be positive and finite.")

        thermal_factor = scale.boltzmann_constant / (
            scale.reduced_planck_constant * scale.speed_of_light
        )
        bosonic = np.asarray(
            [species_[int(index)].statistics == "boson" for index in indices],
            dtype=bool,
        )
        multiplicities = np.asarray(
            [species_[int(index)].multiplicity for index in indices], dtype=float
        )
        rest_masses = np.asarray(
            [species_[int(index)].rest_mass_frequency for index in indices], dtype=float
        )

        self.scale = scale
        self.species = species_
        self.angular_frequencies = jnp.asarray(frequencies)
        self.frequency_weights = jnp.asarray(weights)
        self.species_indices = jnp.asarray(indices)
        self.polar_mode_numbers = jnp.asarray(polar)
        self.azimuthal_mode_numbers = jnp.asarray(azimuthal)
        self.active_modes = jnp.asarray(active)
        self.bosonic_modes = jnp.asarray(bosonic)
        self.multiplicities = jnp.asarray(multiplicities)
        self.rest_mass_frequencies = jnp.asarray(rest_masses)
        self.absolute_tail_tolerances = jnp.asarray(tolerances)
        self.relative_tail_tolerance = relative
        self.greybody_tolerance = greybody
        self.corotation_tolerance = corotation
        self.temperature_to_wavenumber = float(thermal_factor)
        self.mode_species_ids = species_by_mode
        self.mode_ids = modes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hawking-spectrum-plan",
                "scale": scale.scale_id,
                "species": [item.record_id for item in species_],
                "angular_frequencies": frequencies,
                "mode_species_ids": species_by_mode,
                "mode_ids": modes,
                "polar_mode_numbers": polar,
                "azimuthal_mode_numbers": azimuthal,
                "active_modes": active,
                "absolute_tail_tolerances": tolerances,
                "relative_tail_tolerance": relative,
                "greybody_tolerance": greybody,
                "corotation_tolerance": corotation,
            }
        )

    @property
    def mode_capacity(self) -> int:
        return len(self.mode_ids)

    @property
    def frequency_capacity(self) -> int:
        return int(self.angular_frequencies.shape[0])


class HawkingScatteringData(StrictModule, NonTrainableState):
    """Neutral fixed-shape carrier for independently qualified scattering modes.

    The greybody factor is signed: a qualified bosonic superradiant mode has a
    negative value below corotation. ``corotation_slopes`` supplies
    d(greybody)/d(angular_frequency) for the removable Bose singularity.
    """

    greybody_factors: Array
    corotation_slopes: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    tail_evidence: HawkingTailEvidence
    source_ids: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    scattering_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: HawkingSpectrumPlan,
        greybody_factors: ArrayLike,
        corotation_slopes: ArrayLike,
        /,
        *,
        finite: ArrayLike,
        converged: ArrayLike,
        physically_valid: ArrayLike,
        qualified: ArrayLike,
        derivative_valid: ArrayLike,
        tail_evidence: HawkingTailEvidence,
        source_ids: Sequence[Sequence[str]],
        qualification_id: str,
    ):
        if not isinstance(plan, HawkingSpectrumPlan):
            raise TypeError("plan must be a HawkingSpectrumPlan.")
        if not isinstance(tail_evidence, HawkingTailEvidence):
            raise TypeError("tail_evidence must be HawkingTailEvidence.")
        expected = (plan.mode_capacity, plan.frequency_capacity)
        greybody = np.asarray(greybody_factors, dtype=float)
        slopes = np.asarray(corotation_slopes, dtype=float)
        if greybody.shape != expected:
            raise ValueError(
                "greybody_factors must have shape (mode_capacity, frequency_capacity)."
            )
        if slopes.shape != (plan.mode_capacity,):
            raise ValueError("corotation_slopes must have shape (mode_capacity,).")
        flags = tuple(
            np.asarray(value, dtype=bool)
            for value in (
                finite,
                converged,
                physically_valid,
                qualified,
                derivative_valid,
            )
        )
        if any(value.shape != (plan.mode_capacity,) for value in flags):
            raise ValueError("Scattering status arrays must match mode capacity.")
        sources = tuple(tuple(str(value).strip() for value in row) for row in source_ids)
        if (
            len(sources) != plan.mode_capacity
            or any(len(row) != plan.frequency_capacity for row in sources)
            or any(not value for row in sources for value in row)
        ):
            raise ValueError(
                "source_ids must provide one non-empty source ID per mode-frequency cell."
            )
        qualification = str(qualification_id).strip()
        if not qualification:
            raise ValueError("Scattering qualification_id must be non-empty.")

        self.greybody_factors = jnp.asarray(greybody)
        self.corotation_slopes = jnp.asarray(slopes)
        self.finite = jnp.asarray(flags[0])
        self.converged = jnp.asarray(flags[1])
        self.physically_valid = jnp.asarray(flags[2])
        self.qualified = jnp.asarray(flags[3])
        self.derivative_valid = jnp.asarray(flags[4])
        self.tail_evidence = tail_evidence
        self.source_ids = sources
        self.qualification_id = qualification
        self.plan_id = plan.plan_id
        self.scattering_id = canonical_fingerprint(
            {
                "kind": "hawking-scattering-data",
                "plan": plan.plan_id,
                "greybody_factors": greybody,
                "corotation_slopes": slopes,
                "finite": flags[0],
                "converged": flags[1],
                "physically_valid": flags[2],
                "qualified": flags[3],
                "derivative_valid": flags[4],
                "tail_evidence": tail_evidence.evidence_id,
                "source_ids": sources,
                "qualification_id": qualification,
            }
        )


class HawkingSpectrumResult(StrictModule):
    source_state: KerrEvaporationState
    mode_number_flux_density: Array
    mode_energy_flux_density: Array
    mode_angular_momentum_flux_density: Array
    mode_number_flux: Array
    mode_energy_flux: Array
    mode_angular_momentum_flux: Array
    species_number_flux: Array
    species_energy_flux: Array
    species_angular_momentum_flux: Array
    number_flux: Array
    energy_flux: Array
    angular_momentum_flux: Array
    frequency_remainder_upper: Array
    mode_remainder_upper: Array
    tail_remainder_upper: Array
    mode_finite: Array
    mode_converged: Array
    mode_physically_valid: Array
    mode_qualified: Array
    mode_derivative_valid: Array
    horizon_finite: Array
    horizon_converged: Array
    horizon_physically_valid: Array
    horizon_qualified: Array
    horizon_derivative_valid: Array
    coverage_satisfied: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    species_ids: tuple[str, ...] = eqx.field(static=True)
    mode_ids: tuple[str, ...] = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    scattering_id: str = eqx.field(static=True)
    horizon_source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    spectrum_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.finite & self.converged & self.physically_valid & self.qualified

    def bound_to(self, state: KerrEvaporationState, /) -> Array:
        """Return whether this spectrum was evaluated for exactly ``state``."""

        if not isinstance(state, KerrEvaporationState):
            raise TypeError("state must be a KerrEvaporationState.")
        same_lineage = self.source_state.state_id == state.state_id
        return (
            jnp.asarray(same_lineage)
            & (self.source_state.mass == state.mass)
            & (self.source_state.angular_momentum == state.angular_momentum)
            & (self.source_state.elapsed_time == state.elapsed_time)
            & (self.source_state.step_index == state.step_index)
        )


def evaluate_hawking_spectrum(
    plan: HawkingSpectrumPlan,
    source_state: KerrEvaporationState,
    scattering: HawkingScatteringData,
    horizon_temperature: ArrayLike,
    horizon_angular_velocity: ArrayLike,
    /,
    *,
    horizon_source_id: str,
    horizon_finite: ArrayLike,
    horizon_converged: ArrayLike,
    horizon_physically_valid: ArrayLike,
    horizon_qualified: ArrayLike,
    horizon_derivative_valid: ArrayLike,
) -> HawkingSpectrumResult:
    """Evaluate mode-resolved Unruh-state flux and its qualified quadrature.

    Temperature uses ``plan.scale.temperature_unit``. Frequencies and horizon
    angular velocity use inverse ``plan.scale.dimensional_scale.length_unit``.
    The result retains the exact geometric Kerr source state.
    """

    if not isinstance(source_state, KerrEvaporationState):
        raise TypeError("source_state must be a KerrEvaporationState.")

    if not isinstance(plan, HawkingSpectrumPlan):
        raise TypeError("plan must be a HawkingSpectrumPlan.")
    if not isinstance(scattering, HawkingScatteringData):
        raise TypeError("scattering must be HawkingScatteringData.")
    if scattering.plan_id != plan.plan_id:
        raise ValueError("Scattering data belongs to a different Hawking spectrum plan.")
    horizon_id = str(horizon_source_id).strip()
    if not horizon_id:
        raise ValueError("horizon_source_id must be non-empty.")
    temperature = jnp.asarray(horizon_temperature)
    angular_velocity = jnp.asarray(horizon_angular_velocity)
    horizon_flags = tuple(
        jnp.asarray(value, dtype=bool)
        for value in (
            horizon_finite,
            horizon_converged,
            horizon_physically_valid,
            horizon_qualified,
            horizon_derivative_valid,
        )
    )
    if (
        temperature.shape != ()
        or angular_velocity.shape != ()
        or any(value.shape != () for value in horizon_flags)
    ):
        raise ValueError("Horizon values and scientific status flags must be scalar.")

    dtype = jnp.result_type(
        plan.angular_frequencies,
        scattering.greybody_factors,
        temperature,
        angular_velocity,
    )
    omega = plan.angular_frequencies.astype(dtype)[None, :]
    m = plan.azimuthal_mode_numbers.astype(dtype)[:, None]
    thermal_wavenumber = temperature.astype(dtype) * jnp.asarray(
        plan.temperature_to_wavenumber, dtype=dtype
    )
    safe_thermal = jnp.where(thermal_wavenumber > 0.0, thermal_wavenumber, 1.0)
    delta = omega - m * angular_velocity.astype(dtype)
    exponent = delta / safe_thermal
    greybody = scattering.greybody_factors.astype(dtype)

    small_scale = jnp.sqrt(jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))
    small = jnp.abs(exponent) <= small_scale
    safe_exponent = jnp.where(
        small,
        jnp.where(exponent < 0.0, -small_scale, small_scale),
        exponent,
    )
    decay = jnp.exp(-jnp.abs(safe_exponent))
    bose_reciprocal = jnp.where(
        safe_exponent > 0.0,
        decay / (1.0 - decay),
        -1.0 / (1.0 - decay),
    )
    bose_regular = greybody * bose_reciprocal
    bose_series = (
        scattering.corotation_slopes.astype(dtype)[:, None]
        * safe_thermal
        * (1.0 - 0.5 * exponent + exponent * exponent / 12.0 - exponent**4 / 720.0)
    )
    bose_ratio = jnp.where(small, bose_series, bose_regular)

    fermi_reciprocal = jnp.exp(-jnp.maximum(exponent, 0.0)) / (
        1.0 + jnp.exp(-jnp.abs(exponent))
    )
    fermi_ratio = greybody * fermi_reciprocal
    occupation_times_greybody = jnp.where(
        plan.bosonic_modes[:, None], bose_ratio, fermi_ratio
    )

    propagating = omega >= plan.rest_mass_frequencies.astype(dtype)[:, None]
    support = plan.active_modes[:, None] & propagating
    normalization = plan.multiplicities.astype(dtype)[:, None] / (2.0 * jnp.pi)
    number_density = jnp.where(support, normalization * occupation_times_greybody, 0.0)
    energy_density = omega * number_density
    angular_density = m * number_density

    weights = plan.frequency_weights.astype(dtype)
    mode_number = contract("mf,f->m", number_density, weights)
    mode_energy = contract("mf,f->m", energy_density, weights)
    mode_angular = contract("mf,f->m", angular_density, weights)
    selector = jax.nn.one_hot(
        plan.species_indices, len(plan.species), dtype=mode_number.dtype
    )
    species_number = contract("ms,m->s", selector, mode_number)
    species_energy = contract("ms,m->s", selector, mode_energy)
    species_angular = contract("ms,m->s", selector, mode_angular)
    number_flux = jnp.sum(species_number)
    energy_flux = jnp.sum(species_energy)
    angular_flux = jnp.sum(species_angular)

    active_grid = plan.active_modes[:, None]
    mode_finite = (
        scattering.finite
        & jnp.all(jnp.where(active_grid, jnp.isfinite(greybody), True), axis=-1)
        & jnp.isfinite(scattering.corotation_slopes)
    )
    threshold = jnp.asarray(plan.corotation_tolerance, dtype=dtype)
    greybody_tolerance = jnp.asarray(plan.greybody_tolerance, dtype=dtype)
    above = delta > threshold
    below = delta < -threshold
    boson_sign_valid = jnp.where(
        above,
        greybody >= -greybody_tolerance,
        jnp.where(
            below,
            greybody <= greybody_tolerance,
            jnp.abs(greybody)
            <= greybody_tolerance
            + jnp.abs(scattering.corotation_slopes.astype(dtype)[:, None]) * threshold,
        ),
    )
    boson_bounds = boson_sign_valid & (greybody <= 1.0 + greybody_tolerance)
    fermion_bounds = (greybody >= -greybody_tolerance) & (
        greybody <= 1.0 + greybody_tolerance
    )
    greybody_physical = jnp.where(
        plan.bosonic_modes[:, None], boson_bounds, fermion_bounds
    )
    density_physical = number_density >= 0.0
    mode_physical = scattering.physically_valid & jnp.all(
        jnp.where(active_grid, greybody_physical & density_physical, True), axis=-1
    )
    mode_converged = scattering.converged
    mode_qualified = scattering.qualified
    mode_derivative = scattering.derivative_valid

    tail_frequency = scattering.tail_evidence.frequency_remainder_upper.astype(dtype)
    tail_modes = scattering.tail_evidence.mode_remainder_upper.astype(dtype)
    tail_total = tail_frequency + tail_modes
    tail_finite = jnp.all(jnp.isfinite(tail_total))
    tail_physical = jnp.all(tail_frequency >= 0.0) & jnp.all(tail_modes >= 0.0)
    resolved = jnp.abs(jnp.stack((number_flux, energy_flux, angular_flux)))
    tail_limits = (
        plan.absolute_tail_tolerances.astype(dtype)
        + jnp.asarray(plan.relative_tail_tolerance, dtype=dtype) * resolved
    )
    coverage = tail_finite & tail_physical & jnp.all(tail_total <= tail_limits)

    active = plan.active_modes
    finite_modes = jnp.all(jnp.where(active, mode_finite, True))
    converged_modes = jnp.all(jnp.where(active, mode_converged, True))
    physical_modes = jnp.all(jnp.where(active, mode_physical, True))
    qualified_modes = jnp.all(jnp.where(active, mode_qualified, True))
    derivative_modes = jnp.all(jnp.where(active, mode_derivative, True))
    computed_horizon_finite = jnp.isfinite(temperature) & jnp.isfinite(angular_velocity)
    totals_finite = jnp.all(
        jnp.isfinite(jnp.stack((number_flux, energy_flux, angular_flux)))
    )
    finite = (
        horizon_flags[0]
        & computed_horizon_finite
        & finite_modes
        & tail_finite
        & totals_finite
    )
    physically_valid = (
        horizon_flags[2]
        & finite
        & (thermal_wavenumber > 0.0)
        & physical_modes
        & tail_physical
        & (number_flux >= 0.0)
        & (energy_flux >= 0.0)
    )
    converged = horizon_flags[1] & converged_modes & coverage
    qualified = horizon_flags[3] & qualified_modes & scattering.tail_evidence.qualified
    derivative_valid = (
        horizon_flags[4]
        & derivative_modes
        & scattering.tail_evidence.derivative_valid
        & finite
        & physically_valid
    )
    species_ids = tuple(item.species_id for item in plan.species)
    spectrum_id = canonical_fingerprint(
        {
            "kind": "hawking-spectrum-result",
            "plan": plan.plan_id,
            "scattering": scattering.scattering_id,
            "horizon_source": horizon_id,
            "species": species_ids,
            "source_state": source_state.state_id,
            "state_binding": "exact-mass-angular-momentum-time-step",
        }
    )
    return HawkingSpectrumResult(
        source_state=source_state,
        mode_number_flux_density=number_density,
        mode_energy_flux_density=energy_density,
        mode_angular_momentum_flux_density=angular_density,
        mode_number_flux=mode_number,
        mode_energy_flux=mode_energy,
        mode_angular_momentum_flux=mode_angular,
        species_number_flux=species_number,
        species_energy_flux=species_energy,
        species_angular_momentum_flux=species_angular,
        number_flux=number_flux,
        energy_flux=energy_flux,
        angular_momentum_flux=angular_flux,
        frequency_remainder_upper=tail_frequency,
        mode_remainder_upper=tail_modes,
        tail_remainder_upper=tail_total,
        mode_finite=mode_finite,
        mode_converged=mode_converged,
        mode_physically_valid=mode_physical,
        mode_qualified=mode_qualified,
        mode_derivative_valid=mode_derivative,
        horizon_finite=horizon_flags[0],
        horizon_converged=horizon_flags[1],
        horizon_physically_valid=horizon_flags[2],
        horizon_qualified=horizon_flags[3],
        horizon_derivative_valid=horizon_flags[4],
        coverage_satisfied=coverage,
        finite=finite,
        converged=converged,
        physically_valid=physically_valid,
        qualified=qualified,
        derivative_valid=derivative_valid,
        species_ids=species_ids,
        mode_ids=plan.mode_ids,
        scale_id=plan.scale.scale_id,
        scattering_id=scattering.scattering_id,
        horizon_source_id=horizon_id,
        plan_id=plan.plan_id,
        spectrum_id=spectrum_id,
    )


class HawkingEvaporationTermination(IntEnum):
    """Reason a bounded Kerr evaporation trajectory stopped."""

    CAPACITY_REACHED = 0
    NONFINITE_INITIAL_STATE = 1
    INVALID_KERR_STATE = 2
    INSUFFICIENT_COVERAGE = 3
    UNQUALIFIED_SCATTERING = 4
    INVALID_FLUX = 5
    ADIABATICITY_LIMIT = 6
    SEMICLASSICAL_BOUNDARY = 7
    NONFINITE_FLUX = 8
    STALE_SPECTRUM_BINDING = 9


class KerrEvaporationState(StrictModule, NonTrainableState):
    """One geometric Kerr state, with M in length and J in length squared."""

    mass: Array
    angular_momentum: Array
    elapsed_time: Array
    step_index: Array
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        mass: ArrayLike,
        angular_momentum: ArrayLike,
        /,
        *,
        elapsed_time: ArrayLike = 0.0,
        step_index: ArrayLike = 0,
        state_id: str,
    ):
        identifier = str(state_id).strip()
        if not identifier:
            raise ValueError("Kerr evaporation state_id must be non-empty.")
        mass_ = jnp.asarray(mass)
        angular_ = jnp.asarray(angular_momentum)
        elapsed_ = jnp.asarray(elapsed_time)
        index_ = jnp.asarray(step_index, dtype=jnp.int32)
        if (
            mass_.shape != ()
            or angular_.shape != ()
            or elapsed_.shape != ()
            or index_.shape != ()
        ):
            raise ValueError("Kerr evaporation state values must be scalar.")
        self.mass = mass_
        self.angular_momentum = angular_
        self.elapsed_time = elapsed_
        self.step_index = index_
        self.state_id = identifier

    @property
    def dimensionless_spin(self) -> Array:
        return self.angular_momentum / (self.mass * self.mass)


class KerrEvaporationPlan(StrictModule, NonTrainableState):
    """Fixed-capacity, fail-closed semiclassical Kerr evaporation plan."""

    scale: RelativityScaleContract
    time_offsets: Array
    planck_area: Array
    planck_length: Array
    semiclassical_mass_floor: Array
    maximum_adiabatic_parameter: float = eqx.field(static=True)
    maximum_mass_fraction_per_step: float = eqx.field(static=True)
    maximum_spin_change_per_step: float = eqx.field(static=True)
    extremality_margin: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        time_offsets: ArrayLike,
        /,
        *,
        semiclassical_mass_ratio: float = 100.0,
        maximum_adiabatic_parameter: float = 1.0e-3,
        maximum_mass_fraction_per_step: float = 1.0e-2,
        maximum_spin_change_per_step: float = 1.0e-2,
        extremality_margin: float = 1.0e-8,
    ):
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be a RelativityScaleContract.")
        if not scale.quantum_constants_explicit:
            raise ValueError("Kerr evaporation requires explicit hbar and k_B constants.")
        times = np.asarray(time_offsets, dtype=float)
        if (
            times.ndim != 1
            or times.size < 2
            or np.any(~np.isfinite(times))
            or times[0] != 0.0
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("Evaporation time offsets must start at zero and increase.")
        ratio = float(semiclassical_mass_ratio)
        adiabatic = float(maximum_adiabatic_parameter)
        mass_fraction = float(maximum_mass_fraction_per_step)
        spin_change = float(maximum_spin_change_per_step)
        margin = float(extremality_margin)
        values = (ratio, adiabatic, mass_fraction, spin_change, margin)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Kerr evaporation limits must be finite.")
        if ratio <= 1.0:
            raise ValueError("semiclassical_mass_ratio must exceed one Planck length.")
        if adiabatic <= 0.0 or mass_fraction <= 0.0 or spin_change <= 0.0:
            raise ValueError("Evaporation adiabatic and step bounds must be positive.")
        if margin <= 0.0 or margin >= 1.0:
            raise ValueError("extremality_margin must lie strictly between zero and one.")
        planck_area_fraction: Fraction = (
            scale.reduced_planck_constant
            * scale.gravitational_constant
            / scale.speed_of_light**3
        )
        planck_area = float(planck_area_fraction)
        planck_length = math.sqrt(planck_area)
        floor = ratio * planck_length
        self.scale = scale
        self.time_offsets = jnp.asarray(times)
        self.planck_area = jnp.asarray(planck_area)
        self.planck_length = jnp.asarray(planck_length)
        self.semiclassical_mass_floor = jnp.asarray(floor)
        self.maximum_adiabatic_parameter = adiabatic
        self.maximum_mass_fraction_per_step = mass_fraction
        self.maximum_spin_change_per_step = spin_change
        self.extremality_margin = margin
        self.plan_id = canonical_fingerprint(
            {
                "kind": "kerr-evaporation-plan",
                "scale": scale.scale_id,
                "time_offsets": times,
                "semiclassical_mass_ratio": ratio,
                "maximum_adiabatic_parameter": adiabatic,
                "maximum_mass_fraction_per_step": mass_fraction,
                "maximum_spin_change_per_step": spin_change,
                "extremality_margin": margin,
                "spectrum_state_binding": "exact-mass-angular-momentum-time-step",
            }
        )

    @property
    def step_capacity(self) -> int:
        return int(self.time_offsets.shape[0]) - 1


class HawkingSpectrumEvaluator(Protocol):
    def __call__(self, state: KerrEvaporationState, /) -> HawkingSpectrumResult: ...


class KerrEvaporationResult(StrictModule):
    times: Array
    masses: Array
    angular_momenta: Array
    dimensionless_spins: Array
    valid: Array
    attempted: Array
    accepted: Array
    mass_derivatives: Array
    angular_momentum_derivatives: Array
    step_finite: Array
    step_converged: Array
    step_physically_valid: Array
    step_qualified: Array
    step_derivative_valid: Array
    state_binding_satisfied: Array
    coverage_satisfied: Array
    adiabatic_satisfied: Array
    semiclassical_satisfied: Array
    completed_steps: Array
    termination: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    final_state: KerrEvaporationState
    scale_id: str = eqx.field(static=True)
    spectrum_source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    trajectory_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        clean_termination = (
            self.termination == int(HawkingEvaporationTermination.CAPACITY_REACHED)
        ) | (
            self.termination == int(HawkingEvaporationTermination.SEMICLASSICAL_BOUNDARY)
        )
        return (
            clean_termination
            & self.finite
            & self.converged
            & self.physically_valid
            & self.qualified
        )


def evolve_kerr_evaporation(
    plan: KerrEvaporationPlan,
    initial_state: KerrEvaporationState,
    spectrum_evaluator: Callable[[KerrEvaporationState], HawkingSpectrumResult],
    /,
    *,
    spectrum_source_id: str,
) -> KerrEvaporationResult:
    """Coevolve Kerr mass and angular momentum until a declared validity boundary.

    The evaluator must return fixed-shape qualified Hawking spectra for each
    proposed state. A step that would reach the excluded quantum-gravity regime
    is not committed; the last semiclassical state remains authoritative.
    """

    if not isinstance(plan, KerrEvaporationPlan):
        raise TypeError("plan must be a KerrEvaporationPlan.")
    if not isinstance(initial_state, KerrEvaporationState):
        raise TypeError("initial_state must be a KerrEvaporationState.")
    if not callable(spectrum_evaluator):
        raise TypeError("spectrum_evaluator must be callable.")
    source_id = str(spectrum_source_id).strip()
    if not source_id:
        raise ValueError("spectrum_source_id must be non-empty.")

    dtype = jnp.result_type(
        initial_state.mass, initial_state.angular_momentum, plan.time_offsets
    )
    mass0 = initial_state.mass.astype(dtype)
    angular0 = initial_state.angular_momentum.astype(dtype)
    time0 = initial_state.elapsed_time.astype(dtype)
    finite0 = (
        jnp.isfinite(mass0)
        & jnp.isfinite(angular0)
        & jnp.isfinite(time0)
        & (time0 >= 0.0)
        & (initial_state.step_index >= 0)
    )
    kerr0 = (mass0 > 0.0) & (
        jnp.abs(angular0) < (1.0 - plan.extremality_margin) * mass0 * mass0
    )
    semiclassical0 = mass0 > plan.semiclassical_mass_floor.astype(dtype)
    active0 = finite0 & kerr0 & semiclassical0
    status0 = jnp.where(
        ~finite0,
        int(HawkingEvaporationTermination.NONFINITE_INITIAL_STATE),
        jnp.where(
            ~kerr0,
            int(HawkingEvaporationTermination.INVALID_KERR_STATE),
            jnp.where(
                ~semiclassical0,
                int(HawkingEvaporationTermination.SEMICLASSICAL_BOUNDARY),
                int(HawkingEvaporationTermination.CAPACITY_REACHED),
            ),
        ),
    ).astype(jnp.int32)

    starts = plan.time_offsets[:-1].astype(dtype)
    ends = plan.time_offsets[1:].astype(dtype)
    indices = jnp.arange(plan.step_capacity, dtype=jnp.int32)
    quantum_area = plan.planck_area.astype(dtype)
    floor = plan.semiclassical_mass_floor.astype(dtype)

    def step(carry, interval):
        mass, angular_momentum, active, termination = carry
        start, end, local_index = interval
        dt = end - start
        attempted = active

        def evaluate_current(_):
            state = KerrEvaporationState(
                mass,
                angular_momentum,
                elapsed_time=time0 + start,
                step_index=initial_state.step_index + local_index,
                state_id=initial_state.state_id,
            )
            spectrum = spectrum_evaluator(state)
            if not isinstance(spectrum, HawkingSpectrumResult):
                raise TypeError("spectrum_evaluator must return HawkingSpectrumResult.")
            if spectrum.scale_id != plan.scale.scale_id:
                raise ValueError("Hawking spectrum and evaporation scale IDs differ.")
            return (
                spectrum.energy_flux.astype(dtype),
                spectrum.angular_momentum_flux.astype(dtype),
                spectrum.bound_to(state),
                spectrum.finite,
                spectrum.converged,
                spectrum.physically_valid,
                spectrum.qualified,
                spectrum.derivative_valid,
                spectrum.coverage_satisfied,
            )

        def hold(_):
            return (
                jnp.asarray(0.0, dtype=dtype),
                jnp.asarray(0.0, dtype=dtype),
                jnp.asarray(True),
                jnp.asarray(True),
                jnp.asarray(True),
                jnp.asarray(True),
                jnp.asarray(True),
                jnp.asarray(True),
                jnp.asarray(True),
            )

        (
            energy_flux,
            angular_flux,
            state_binding,
            spectrum_finite,
            spectrum_converged,
            spectrum_physical,
            spectrum_qualified,
            spectrum_derivative,
            coverage,
        ) = jax.lax.cond(active, evaluate_current, hold, operand=None)
        mass_derivative = -quantum_area * energy_flux
        angular_derivative = -quantum_area * angular_flux
        rates_finite = jnp.isfinite(mass_derivative) & jnp.isfinite(angular_derivative)
        finite = spectrum_finite & rates_finite
        flux_physical = spectrum_physical & (energy_flux >= 0.0)

        candidate_mass = mass + dt * mass_derivative
        candidate_angular = angular_momentum + dt * angular_derivative
        candidate_spin = candidate_angular / (candidate_mass * candidate_mass)
        current_spin = angular_momentum / (mass * mass)
        mass_fraction = jnp.abs(dt * mass_derivative) / mass
        spin_change = jnp.abs(candidate_spin - current_spin)
        adiabatic_parameter = jnp.maximum(
            jnp.abs(mass_derivative),
            jnp.abs(angular_derivative) / mass,
        )
        adiabatic = (
            (adiabatic_parameter <= plan.maximum_adiabatic_parameter)
            & (mass_fraction <= plan.maximum_mass_fraction_per_step)
            & (spin_change <= plan.maximum_spin_change_per_step)
        )
        semiclassical = candidate_mass > floor
        candidate_kerr = (
            jnp.isfinite(candidate_mass)
            & jnp.isfinite(candidate_angular)
            & (candidate_mass > 0.0)
            & (
                jnp.abs(candidate_angular)
                < (1.0 - plan.extremality_margin) * candidate_mass * candidate_mass
            )
        )
        physical = flux_physical & candidate_kerr
        step_ok = (
            finite
            & state_binding
            & spectrum_converged
            & coverage
            & spectrum_qualified
            & physical
            & adiabatic
            & semiclassical
        )
        accepted = active & step_ok
        next_mass = jnp.where(accepted, candidate_mass, mass)
        next_angular = jnp.where(accepted, candidate_angular, angular_momentum)
        failure = jnp.where(
            ~finite,
            int(HawkingEvaporationTermination.NONFINITE_FLUX),
            jnp.where(
                ~state_binding,
                int(HawkingEvaporationTermination.STALE_SPECTRUM_BINDING),
                jnp.where(
                    ~(spectrum_converged & coverage),
                    int(HawkingEvaporationTermination.INSUFFICIENT_COVERAGE),
                    jnp.where(
                        ~spectrum_qualified,
                        int(HawkingEvaporationTermination.UNQUALIFIED_SCATTERING),
                        jnp.where(
                            ~flux_physical,
                            int(HawkingEvaporationTermination.INVALID_FLUX),
                            jnp.where(
                                ~adiabatic,
                                int(HawkingEvaporationTermination.ADIABATICITY_LIMIT),
                                jnp.where(
                                    ~semiclassical,
                                    int(
                                        HawkingEvaporationTermination.SEMICLASSICAL_BOUNDARY
                                    ),
                                    int(HawkingEvaporationTermination.INVALID_KERR_STATE),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        next_termination = jnp.where(active & ~step_ok, failure, termination)
        next_active = active & step_ok
        output = (
            next_mass,
            next_angular,
            attempted,
            accepted,
            jnp.where(attempted, mass_derivative, 0.0),
            jnp.where(attempted, angular_derivative, 0.0),
            finite,
            state_binding,
            spectrum_converged,
            physical,
            spectrum_qualified,
            spectrum_derivative,
            coverage,
            adiabatic,
            semiclassical,
        )
        return (next_mass, next_angular, next_active, next_termination), output

    (_, _, _, termination), outputs = jax.lax.scan(
        step,
        (mass0, angular0, active0, status0),
        (starts, ends, indices),
    )
    masses = jnp.concatenate((mass0[None], outputs[0]))
    angular_momenta = jnp.concatenate((angular0[None], outputs[1]))
    attempted = outputs[2]
    accepted = outputs[3]
    valid = jnp.concatenate((active0[None], accepted))
    completed_steps = jnp.sum(accepted.astype(jnp.int32))
    final_index = completed_steps
    final_mass = masses[final_index]
    final_angular = angular_momenta[final_index]
    final_time = time0 + plan.time_offsets.astype(dtype)[final_index]
    trajectory_id = canonical_fingerprint(
        {
            "kind": "kerr-evaporation-trajectory",
            "plan": plan.plan_id,
            "initial_state": initial_state.state_id,
            "spectrum_source": source_id,
        }
    )
    final_state = KerrEvaporationState(
        final_mass,
        final_angular,
        elapsed_time=final_time,
        step_index=initial_state.step_index + completed_steps,
        state_id=trajectory_id,
    )
    step_finite = outputs[6]
    step_binding = outputs[7]
    step_converged = outputs[8]
    step_physical = outputs[9]
    step_qualified = outputs[10]
    step_derivative = outputs[11]
    aggregate_finite = finite0 & jnp.all(jnp.where(attempted, step_finite, True))
    aggregate_converged = active0 & jnp.all(jnp.where(attempted, step_converged, True))
    aggregate_physical = active0 & jnp.all(jnp.where(attempted, step_physical, True))
    aggregate_qualified = active0 & jnp.all(
        jnp.where(attempted, step_qualified & step_binding, True)
    )
    derivative_valid = (
        termination == int(HawkingEvaporationTermination.CAPACITY_REACHED)
    ) & jnp.all(jnp.where(attempted, step_derivative & step_binding, True))
    return KerrEvaporationResult(
        times=time0 + plan.time_offsets.astype(dtype),
        masses=masses,
        angular_momenta=angular_momenta,
        dimensionless_spins=angular_momenta / (masses * masses),
        valid=valid,
        attempted=attempted,
        accepted=accepted,
        mass_derivatives=outputs[4],
        angular_momentum_derivatives=outputs[5],
        step_finite=step_finite,
        step_converged=step_converged,
        step_physically_valid=step_physical,
        step_qualified=step_qualified,
        step_derivative_valid=step_derivative,
        state_binding_satisfied=step_binding,
        coverage_satisfied=outputs[12],
        adiabatic_satisfied=outputs[13],
        semiclassical_satisfied=outputs[14],
        completed_steps=completed_steps,
        termination=termination,
        finite=aggregate_finite,
        converged=aggregate_converged,
        physically_valid=aggregate_physical,
        qualified=aggregate_qualified,
        derivative_valid=derivative_valid,
        final_state=final_state,
        scale_id=plan.scale.scale_id,
        spectrum_source_id=source_id,
        plan_id=plan.plan_id,
        trajectory_id=trajectory_id,
    )


__all__ = [
    "HawkingEvaporationTermination",
    "HawkingScatteringData",
    "HawkingSpectrumEvaluator",
    "HawkingSpectrumPlan",
    "HawkingSpectrumResult",
    "HawkingTailEvidence",
    "KerrEvaporationPlan",
    "KerrEvaporationResult",
    "KerrEvaporationState",
    "QuantumFieldSpecies",
    "QuantumStatistics",
    "evaluate_hawking_spectrum",
    "evolve_kerr_evaporation",
]
