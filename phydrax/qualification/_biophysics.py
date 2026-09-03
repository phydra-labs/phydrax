#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Independent analytic qualifications for cellular and molecular biophysics.

Every function uses SI quantities named in its signature. Singular inverse
problems return typed fail-closed evidence rather than silently choosing a value.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._strict import StrictModule


BOLTZMANN_CONSTANT_J_PER_K = 1.380649e-23
ELEMENTARY_CHARGE_C = 1.602176634e-19
FARADAY_CONSTANT_C_PER_MOL = 96485.33212331001
GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
PLANCK_CONSTANT_J_S = 6.62607015e-34


def _floating(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    return jnp.asarray(array, dtype=jnp.result_type(array.dtype, jnp.float32))


def _scalar(value: ArrayLike, name: str, /) -> Array:
    array = _floating(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar.")
    return array


def spherical_membrane_capacitance(
    radius_m: ArrayLike,
    specific_capacitance_f_per_m2: ArrayLike,
    /,
) -> Array:
    """Return C = 4πr²c_m for a spherical membrane, in farads."""
    radius = _scalar(radius_m, "radius_m")
    specific = _scalar(specific_capacitance_f_per_m2, "specific_capacitance_f_per_m2")
    radius = eqx.error_if(
        radius,
        ~jnp.isfinite(radius) | (radius <= 0.0),
        "radius_m must be finite and positive.",
    )
    specific = eqx.error_if(
        specific,
        ~jnp.isfinite(specific) | (specific <= 0.0),
        "specific_capacitance_f_per_m2 must be finite and positive.",
    )
    return 4.0 * jnp.pi * radius * radius * specific


def spherical_membrane_ion_count(
    radius_m: ArrayLike,
    specific_capacitance_f_per_m2: ArrayLike,
    potential_v: ArrayLike,
    /,
    *,
    ion_valence: ArrayLike = 1.0,
) -> Array:
    """Return the magnitude of the ion count carrying C·V membrane charge."""
    potential = _scalar(potential_v, "potential_v")
    valence = _scalar(ion_valence, "ion_valence")
    potential = eqx.error_if(
        potential,
        ~jnp.isfinite(potential),
        "potential_v must be finite.",
    )
    valence = eqx.error_if(
        valence,
        ~jnp.isfinite(valence) | (valence == 0.0),
        "ion_valence must be finite and nonzero.",
    )
    capacitance = spherical_membrane_capacitance(radius_m, specific_capacitance_f_per_m2)
    return jnp.abs(capacitance * potential / (valence * ELEMENTARY_CHARGE_C))


def nernst_equilibrium_potential(
    concentration_inside_mol_per_m3: ArrayLike,
    concentration_outside_mol_per_m3: ArrayLike,
    ion_valence: ArrayLike,
    temperature_k: ArrayLike,
    /,
) -> Array:
    """Return ψ_in−ψ_out = RT/(zF) ln(c_out/c_in), in volts."""
    inside = _scalar(concentration_inside_mol_per_m3, "concentration_inside_mol_per_m3")
    outside = _scalar(
        concentration_outside_mol_per_m3, "concentration_outside_mol_per_m3"
    )
    valence = _scalar(ion_valence, "ion_valence")
    temperature = _scalar(temperature_k, "temperature_k")
    invalid = (
        ~jnp.isfinite(inside)
        | (inside <= 0.0)
        | ~jnp.isfinite(outside)
        | (outside <= 0.0)
        | ~jnp.isfinite(valence)
        | (valence == 0.0)
        | ~jnp.isfinite(temperature)
        | (temperature <= 0.0)
    )
    inside = eqx.error_if(
        inside,
        invalid,
        "Nernst concentrations and temperature must be finite and positive; valence must be finite and nonzero.",
    )
    return (
        GAS_CONSTANT_J_PER_MOL_K
        * temperature
        / (valence * FARADAY_CONSTANT_C_PER_MOL)
        * (jnp.log(outside) - jnp.log(inside))
    )


def eyring_rate(
    activation_free_energy_j_per_mol: ArrayLike,
    temperature_k: ArrayLike,
    /,
    *,
    transmission_coefficient: ArrayLike = 1.0,
) -> Array:
    """Return κk_B T/h exp(−ΔG‡/RT), in inverse seconds."""
    barrier = _scalar(
        activation_free_energy_j_per_mol, "activation_free_energy_j_per_mol"
    )
    temperature = _scalar(temperature_k, "temperature_k")
    transmission = _scalar(transmission_coefficient, "transmission_coefficient")
    invalid = (
        ~jnp.isfinite(barrier)
        | (barrier < 0.0)
        | ~jnp.isfinite(temperature)
        | (temperature <= 0.0)
        | ~jnp.isfinite(transmission)
        | (transmission <= 0.0)
        | (transmission > 1.0)
    )
    barrier = eqx.error_if(
        barrier,
        invalid,
        "Eyring barrier must be nonnegative, temperature positive, and transmission coefficient in (0, 1].",
    )
    prefactor = (
        transmission * BOLTZMANN_CONSTANT_J_PER_K * temperature / PLANCK_CONSTANT_J_S
    )
    return prefactor * jnp.exp(-barrier / (GAS_CONSTANT_J_PER_MOL_K * temperature))


class AntiporterBalanceResult(StrictModule):
    """Electrochemical antiporter balance and singularity evidence."""

    chemical_driving_energy_j_per_mol: Array
    charge_stoichiometry: Array
    electrochemical_energy_j_per_mol: Array
    equilibrium_potential_v: Array
    finite: Array
    identifiable: Array
    balanced: Array
    successful: Array


def antiporter_electrochemical_balance(
    substrate_inside_mol_per_m3: ArrayLike,
    substrate_outside_mol_per_m3: ArrayLike,
    driver_inside_mol_per_m3: ArrayLike,
    driver_outside_mol_per_m3: ArrayLike,
    substrate_valence: ArrayLike,
    driver_valence: ArrayLike,
    substrate_stoichiometry: ArrayLike,
    driver_stoichiometry: ArrayLike,
    temperature_k: ArrayLike,
    membrane_potential_v: ArrayLike = 0.0,
    /,
) -> AntiporterBalanceResult:
    """Balance substrate influx against oppositely directed driver efflux.

    The potential convention is ψ_in−ψ_out. A zero net transported charge makes
    the equilibrium potential unidentifiable; the returned value is then NaN.
    """
    values = tuple(
        _scalar(value, name)
        for value, name in (
            (substrate_inside_mol_per_m3, "substrate_inside_mol_per_m3"),
            (substrate_outside_mol_per_m3, "substrate_outside_mol_per_m3"),
            (driver_inside_mol_per_m3, "driver_inside_mol_per_m3"),
            (driver_outside_mol_per_m3, "driver_outside_mol_per_m3"),
            (substrate_valence, "substrate_valence"),
            (driver_valence, "driver_valence"),
            (substrate_stoichiometry, "substrate_stoichiometry"),
            (driver_stoichiometry, "driver_stoichiometry"),
            (temperature_k, "temperature_k"),
            (membrane_potential_v, "membrane_potential_v"),
        )
    )
    (
        substrate_inside,
        substrate_outside,
        driver_inside,
        driver_outside,
        substrate_charge,
        driver_charge,
        substrate_count,
        driver_count,
        temperature,
        potential,
    ) = values
    finite = jnp.all(jnp.stack(tuple(jnp.isfinite(value) for value in values)))
    physical = (
        (substrate_inside > 0.0)
        & (substrate_outside > 0.0)
        & (driver_inside > 0.0)
        & (driver_outside > 0.0)
        & (substrate_count > 0.0)
        & (driver_count > 0.0)
        & (temperature > 0.0)
    )
    valid = finite & physical
    safe_substrate_inside = jnp.where(valid, substrate_inside, 1.0)
    safe_substrate_outside = jnp.where(valid, substrate_outside, 1.0)
    safe_driver_inside = jnp.where(valid, driver_inside, 1.0)
    safe_driver_outside = jnp.where(valid, driver_outside, 1.0)
    chemical = (
        GAS_CONSTANT_J_PER_MOL_K
        * temperature
        * (
            substrate_count
            * (jnp.log(safe_substrate_inside) - jnp.log(safe_substrate_outside))
            - driver_count * (jnp.log(safe_driver_inside) - jnp.log(safe_driver_outside))
        )
    )
    charge_stoichiometry = (
        substrate_count * substrate_charge - driver_count * driver_charge
    )
    charge_coefficient = FARADAY_CONSTANT_C_PER_MOL * charge_stoichiometry
    tolerance = jnp.finfo(chemical.dtype).eps * jnp.maximum(
        1.0,
        jnp.abs(substrate_count * substrate_charge)
        + jnp.abs(driver_count * driver_charge),
    )
    electrochemical = chemical + charge_coefficient * potential
    computable = (
        valid
        & jnp.isfinite(chemical)
        & jnp.isfinite(charge_stoichiometry)
        & jnp.isfinite(charge_coefficient)
        & jnp.isfinite(electrochemical)
    )
    charge_identifiable = computable & (jnp.abs(charge_stoichiometry) > tolerance)
    equilibrium_candidate = -chemical / jnp.where(
        charge_identifiable, charge_coefficient, jnp.ones_like(charge_coefficient)
    )
    finite_result = computable & (
        ~charge_identifiable | jnp.isfinite(equilibrium_candidate)
    )
    identifiable = finite_result & charge_identifiable
    equilibrium = jnp.where(
        identifiable,
        equilibrium_candidate,
        jnp.asarray(jnp.nan, dtype=chemical.dtype),
    )
    balance_scale = jnp.maximum(
        1.0, jnp.abs(chemical) + jnp.abs(charge_coefficient * potential)
    )
    balanced = computable & (
        jnp.abs(electrochemical) <= 32.0 * jnp.finfo(chemical.dtype).eps * balance_scale
    )
    successful = finite_result & identifiable
    return AntiporterBalanceResult(
        chemical,
        charge_stoichiometry,
        electrochemical,
        equilibrium,
        finite_result,
        identifiable,
        balanced,
        successful,
    )


class BrownianTransportResult(StrictModule):
    """Recovered drift and diffusion from fixed-step Brownian trajectories."""

    drift_velocity_m_per_s: Array
    diffusion_coefficient_m2_per_s: Array
    increment_count: Array
    finite: Array
    identifiable: Array
    successful: Array


def recover_brownian_transport(
    positions_m: ArrayLike,
    time_step_s: ArrayLike,
    /,
) -> BrownianTransportResult:
    """Recover Gaussian-increment drift and scalar diffusion in SI units."""
    positions = _floating(positions_m)
    if positions.ndim != 3 or positions.shape[1] < 2 or positions.shape[2] < 1:
        raise ValueError(
            "positions_m must have shape (trajectory, sample, spatial_dimension) with at least two samples."
        )
    step = _scalar(time_step_s, "time_step_s")
    increments = positions[:, 1:, :] - positions[:, :-1, :]
    count = increments.shape[0] * increments.shape[1]
    mean_increment = jnp.mean(increments, axis=(0, 1))
    centered = increments - mean_increment
    squared_sum = contract("ntd,ntd->", centered, centered)
    dimension = positions.shape[2]
    valid = (
        jnp.all(jnp.isfinite(positions)) & jnp.isfinite(step) & (step > 0.0) & (count > 1)
    )
    safe_step = jnp.where(valid, step, jnp.ones_like(step))
    drift = mean_increment / safe_step
    diffusion = squared_sum / (2.0 * dimension * (count - 1) * safe_step)
    finite = valid & jnp.all(jnp.isfinite(drift)) & jnp.isfinite(diffusion)
    identifiable = finite & (count > 1)
    successful = finite & identifiable
    return BrownianTransportResult(
        drift,
        diffusion,
        jnp.asarray(count, dtype=jnp.int32),
        finite,
        identifiable,
        successful,
    )


class CensoredDwellTimeResult(StrictModule):
    """Independent exponential dwell-time likelihood qualification."""

    log_likelihood: Array
    maximum_likelihood_rate_per_s: Array
    event_count: Array
    total_exposure_s: Array
    finite: Array
    identifiable: Array
    successful: Array


def qualify_censored_dwell_times(
    durations_s: ArrayLike,
    event_observed: ArrayLike,
    rate_per_s: ArrayLike,
    /,
) -> CensoredDwellTimeResult:
    """Evaluate ∑δ log λ−λ∑t and the censor-aware rate MLE."""
    durations = _floating(durations_s)
    events_raw = jnp.asarray(event_observed)
    if durations.ndim != 1 or durations.size < 1:
        raise ValueError("durations_s must be a non-empty one-dimensional array.")
    if events_raw.shape != durations.shape:
        raise ValueError("event_observed must match durations_s.")
    rate = _scalar(rate_per_s, "rate_per_s")
    events_valid = jnp.all((events_raw == 0) | (events_raw == 1))
    events = events_raw.astype(jnp.bool_)
    event_count = jnp.sum(events)
    exposure = jnp.sum(durations)
    valid = (
        jnp.all(jnp.isfinite(durations))
        & jnp.all(durations >= 0.0)
        & jnp.isfinite(rate)
        & (rate > 0.0)
        & events_valid
        & (exposure > 0.0)
    )
    safe_rate = jnp.where(valid, rate, jnp.ones_like(rate))
    likelihood = (
        event_count.astype(durations.dtype) * jnp.log(safe_rate) - safe_rate * exposure
    )
    identifiable = valid & (event_count > 0)
    mle = event_count.astype(durations.dtype) / jnp.where(
        exposure > 0.0, exposure, jnp.ones_like(exposure)
    )
    mle = jnp.where(identifiable, mle, jnp.asarray(jnp.nan, dtype=durations.dtype))
    finite = valid & jnp.isfinite(likelihood)
    successful = finite & identifiable
    return CensoredDwellTimeResult(
        likelihood,
        mle,
        event_count,
        exposure,
        finite,
        identifiable,
        successful,
    )


__all__ = [
    "AntiporterBalanceResult",
    "BOLTZMANN_CONSTANT_J_PER_K",
    "BrownianTransportResult",
    "CensoredDwellTimeResult",
    "ELEMENTARY_CHARGE_C",
    "FARADAY_CONSTANT_C_PER_MOL",
    "GAS_CONSTANT_J_PER_MOL_K",
    "PLANCK_CONSTANT_J_S",
    "antiporter_electrochemical_balance",
    "eyring_rate",
    "nernst_equilibrium_potential",
    "qualify_censored_dwell_times",
    "recover_brownian_transport",
    "spherical_membrane_capacitance",
    "spherical_membrane_ion_count",
]
