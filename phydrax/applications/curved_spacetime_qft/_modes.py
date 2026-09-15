#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, ArrayLike

from phydrax.integration import GaussLegendreRule, interval_rule_data

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule


class FLRWModePlan(StrictModule):
    """Fixed-background, finite-mode plan in conformal time."""

    conformal_times: Array
    scale_factors: Array
    scale_factor_primes: Array
    scale_factor_seconds: Array
    comoving_wavenumbers: Array
    momentum_weights: Array
    mass: float = eqx.field(static=True)
    curvature_coupling: float = eqx.field(static=True)
    maximum_mode_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        conformal_times: ArrayLike,
        scale_factors: ArrayLike,
        scale_factor_primes: ArrayLike,
        scale_factor_seconds: ArrayLike,
        comoving_wavenumbers: ArrayLike,
        momentum_weights: ArrayLike,
        /,
        *,
        mass: float,
        curvature_coupling: float = 0.0,
        maximum_mode_steps: int = 1_000_000,
    ):
        times = np.asarray(conformal_times, dtype=float)
        scale = np.asarray(scale_factors, dtype=float)
        prime = np.asarray(scale_factor_primes, dtype=float)
        second = np.asarray(scale_factor_seconds, dtype=float)
        wavenumbers = np.asarray(comoving_wavenumbers, dtype=float)
        weights = np.asarray(momentum_weights, dtype=float)
        mass_ = float(mass)
        coupling = float(curvature_coupling)
        maximum = int(maximum_mode_steps)
        if times.ndim != 1 or times.size < 3 or np.any(np.diff(times) <= 0.0):
            raise ValueError(
                "conformal_times must be strictly increasing with at least three nodes."
            )
        if (
            scale.shape != times.shape
            or prime.shape != times.shape
            or second.shape != times.shape
        ):
            raise ValueError(
                "Scale factor and both derivatives must match conformal_times."
            )
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("scale_factors must be finite and positive.")
        if not np.all(np.isfinite(prime)) or not np.all(np.isfinite(second)):
            raise ValueError("Scale-factor derivatives must be finite.")
        if wavenumbers.ndim != 1 or wavenumbers.size == 0 or np.any(wavenumbers < 0.0):
            raise ValueError("comoving_wavenumbers must be one nonnegative vector.")
        if weights.shape != wavenumbers.shape or np.any(weights <= 0.0):
            raise ValueError("momentum_weights must be positive and match wavenumbers.")
        if not np.all(np.isfinite(wavenumbers)) or not np.all(np.isfinite(weights)):
            raise ValueError("Momentum quadrature data must be finite.")
        if mass_ < 0.0 or not np.isfinite(mass_) or not np.isfinite(coupling):
            raise ValueError("Mass/coupling must be finite and mass nonnegative.")
        required = times.size * wavenumbers.size
        if maximum < 1 or required > maximum:
            raise ValueError(
                f"Mode history requires {required} mode steps; capacity is {maximum}."
            )
        self.conformal_times = jnp.asarray(times)
        self.scale_factors = jnp.asarray(scale)
        self.scale_factor_primes = jnp.asarray(prime)
        self.scale_factor_seconds = jnp.asarray(second)
        self.comoving_wavenumbers = jnp.asarray(wavenumbers)
        self.momentum_weights = jnp.asarray(weights)
        self.mass = mass_
        self.curvature_coupling = coupling
        self.maximum_mode_steps = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-fixed-flrw-mode-plan",
                "conformal_times": array_tree_fingerprint(times),
                "scale_factors": array_tree_fingerprint(scale),
                "scale_factor_primes": array_tree_fingerprint(prime),
                "scale_factor_seconds": array_tree_fingerprint(second),
                "comoving_wavenumbers": array_tree_fingerprint(wavenumbers),
                "momentum_weights": array_tree_fingerprint(weights),
                "mass": mass_,
                "curvature_coupling": coupling,
                "maximum_mode_steps": maximum,
            }
        )


class PreparedFLRWModes(StrictModule):
    plan: FLRWModePlan
    interval_widths: Array
    frequency_squared: Array
    frequencies: Array
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class ModeInitialState(StrictModule):
    modes: Array
    derivatives: Array
    adiabatic_order: int = eqx.field(static=True)


class ModeEvolutionEvidence(StrictModule):
    modes: Array
    derivatives: Array
    wronskians: Array
    initial_wronskian_residual: Array
    maximum_wronskian_drift: Array
    finite: Array
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class BogoliubovEvidence(StrictModule):
    alpha: Array
    beta: Array
    occupation_numbers: Array
    normalization: Array
    normalization_residual: Array
    maximum_normalization_residual: Array
    normalized: Array
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def gauss_legendre_flrw_mode_plan(
    conformal_times: ArrayLike,
    scale_factors: ArrayLike,
    scale_factor_primes: ArrayLike,
    scale_factor_seconds: ArrayLike,
    /,
    *,
    momentum_cutoff: float,
    momentum_order: int,
    mass: float,
    curvature_coupling: float = 0.0,
    maximum_mode_steps: int = 1_000_000,
) -> FLRWModePlan:
    """Build the regulated momentum measure from Phydrax fixed quadrature."""
    cutoff = float(momentum_cutoff)
    order = int(momentum_order)
    maximum = int(maximum_mode_steps)
    times = np.asarray(conformal_times)
    if cutoff <= 0.0 or not np.isfinite(cutoff):
        raise ValueError("momentum_cutoff must be finite and positive.")
    if order < 1 or times.ndim != 1:
        raise ValueError("momentum_order must be positive and times one-dimensional.")
    if maximum < 1 or order * times.size > maximum:
        raise ValueError("Momentum quadrature exceeds maximum_mode_steps.")
    data = interval_rule_data(GaussLegendreRule(order))
    wavenumbers = 0.5 * cutoff * (data.nodes + 1.0)
    weights = 0.5 * cutoff * data.weights
    return FLRWModePlan(
        conformal_times,
        scale_factors,
        scale_factor_primes,
        scale_factor_seconds,
        wavenumbers,
        weights,
        mass=mass,
        curvature_coupling=curvature_coupling,
        maximum_mode_steps=maximum_mode_steps,
    )


def prepare_flrw_modes(plan: FLRWModePlan, /) -> PreparedFLRWModes:
    if not isinstance(plan, FLRWModePlan):
        raise TypeError("plan must be FLRWModePlan.")
    scale = plan.scale_factors[:, None]
    geometric = (6.0 * plan.curvature_coupling - 1.0) * (
        plan.scale_factor_seconds / plan.scale_factors
    )[:, None]
    frequency_squared = (
        plan.comoving_wavenumbers[None, :] ** 2 + (scale * plan.mass) ** 2 + geometric
    )
    if np.any(np.asarray(frequency_squared) <= 0.0):
        raise ValueError(
            "Prepared modes require positive frequency squared at every node."
        )
    return PreparedFLRWModes(
        plan=plan,
        interval_widths=jnp.diff(plan.conformal_times),
        frequency_squared=frequency_squared,
        frequencies=jnp.sqrt(frequency_squared),
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-fixed-flrw-modes",
                "plan": plan.plan_id,
                "equation": "v''+[k^2+a^2m^2+(6xi-1)a''/a]v=0",
            }
        ),
        claim="fixed-background-finite-mode-reference-only",
    )


def differentiate_time(values: Array, times: Array, /) -> Array:
    if values.shape[0] != times.shape[0] or times.ndim != 1 or times.size < 3:
        raise ValueError(
            "Time differentiation needs matching data at three or more nodes."
        )
    first = (values[1] - values[0]) / (times[1] - times[0])
    middle_scale = (times[2:] - times[:-2]).reshape((-1,) + (1,) * (values.ndim - 1))
    middle = (values[2:] - values[:-2]) / middle_scale
    last = (values[-1] - values[-2]) / (times[-1] - times[-2])
    return jnp.concatenate((first[None], middle, last[None]), axis=0)


def adiabatic_frequencies(prepared: PreparedFLRWModes, order: int, /) -> Array:
    """Return the fixed-grid WKB frequency after zero, one, or two recurrences."""
    if not isinstance(prepared, PreparedFLRWModes):
        raise TypeError("prepared must be PreparedFLRWModes.")
    order_ = int(order)
    if order_ not in (0, 2, 4):
        raise ValueError("adiabatic order must be zero, two, or four.")
    frequency = prepared.frequencies
    for _ in range(order_ // 2):
        prime = differentiate_time(frequency, prepared.plan.conformal_times)
        second = differentiate_time(prime, prepared.plan.conformal_times)
        effective = prepared.frequency_squared - 0.5 * second / frequency
        effective = effective + 0.75 * (prime / frequency) ** 2
        effective = eqx.error_if(
            effective,
            jnp.any(~jnp.isfinite(effective) | (effective <= 0.0)),
            "Adiabatic recurrence produced a nonpositive frequency squared.",
        )
        frequency = jnp.sqrt(effective)
    return frequency


def adiabatic_initial_state(
    prepared: PreparedFLRWModes, /, *, order: int = 0
) -> ModeInitialState:
    frequencies = adiabatic_frequencies(prepared, order)
    frequency_primes = differentiate_time(frequencies, prepared.plan.conformal_times)
    initial_frequency = frequencies[0]
    modes = 1.0 / jnp.sqrt(2.0 * initial_frequency)
    derivatives = (
        -0.5 * frequency_primes[0] / initial_frequency - 1.0j * initial_frequency
    ) * modes
    return ModeInitialState(
        modes=modes.astype(jnp.complex128),
        derivatives=derivatives.astype(jnp.complex128),
        adiabatic_order=int(order),
    )


def evolve_flrw_modes(
    prepared: PreparedFLRWModes,
    initial: ModeInitialState,
    /,
) -> ModeEvolutionEvidence:
    """Evolve all modes with a fixed symplectic Störmer--Verlet map."""
    if not isinstance(prepared, PreparedFLRWModes):
        raise TypeError("prepared must be PreparedFLRWModes.")
    if not isinstance(initial, ModeInitialState):
        raise TypeError("initial must be ModeInitialState.")
    mode_count = prepared.plan.comoving_wavenumbers.size
    if initial.modes.shape != (mode_count,) or initial.derivatives.shape != (mode_count,):
        raise ValueError("Initial modes must match the prepared momentum grid.")
    initial_mode = jnp.asarray(initial.modes)
    initial_derivative = jnp.asarray(initial.derivatives)

    def verlet_step(carry, interval_data):
        current_mode, current_derivative = carry
        width, current_frequency_squared, next_frequency_squared = interval_data
        half = current_derivative - 0.5 * width * current_frequency_squared * current_mode
        next_mode = current_mode + width * half
        next_derivative = half - 0.5 * width * next_frequency_squared * next_mode
        return (next_mode, next_derivative), (next_mode, next_derivative)

    _, (mode_tail, derivative_tail) = lax.scan(
        verlet_step,
        (initial_mode, initial_derivative),
        (
            prepared.interval_widths,
            prepared.frequency_squared[:-1],
            prepared.frequency_squared[1:],
        ),
    )
    mode_history = jnp.concatenate((initial_mode[None], mode_tail), axis=0)
    derivative_history = jnp.concatenate(
        (initial_derivative[None], derivative_tail), axis=0
    )
    wronskians = 1.0j * (
        jnp.conj(mode_history) * derivative_history
        - jnp.conj(derivative_history) * mode_history
    )
    initial_residual = jnp.max(jnp.abs(wronskians[0] - 1.0))
    drift = jnp.max(jnp.abs(wronskians - wronskians[0]))
    finite = jnp.all(jnp.isfinite(mode_history)) & jnp.all(
        jnp.isfinite(derivative_history)
    )
    return ModeEvolutionEvidence(
        modes=mode_history,
        derivatives=derivative_history,
        wronskians=wronskians.real,
        initial_wronskian_residual=initial_residual,
        maximum_wronskian_drift=drift,
        finite=finite,
        prepared_id=prepared.prepared_id,
        claim="fixed-grid-symplectic-mode-evolution-only",
    )


def bogoliubov_particle_production(
    prepared: PreparedFLRWModes,
    evolution: ModeEvolutionEvidence,
    /,
    *,
    tolerance: float = 1e-8,
) -> BogoliubovEvidence:
    if not isinstance(prepared, PreparedFLRWModes):
        raise TypeError("prepared must be PreparedFLRWModes.")
    if not isinstance(evolution, ModeEvolutionEvidence):
        raise TypeError("evolution must be ModeEvolutionEvidence.")
    tolerance_ = float(tolerance)
    if tolerance_ < 0.0 or not np.isfinite(tolerance_):
        raise ValueError("tolerance must be finite and nonnegative.")
    frequency = prepared.frequencies
    scale = jnp.sqrt(frequency / 2.0)
    inverse_scale = 1.0 / jnp.sqrt(2.0 * frequency)
    alpha = scale * evolution.modes + 1.0j * inverse_scale * evolution.derivatives
    beta = scale * evolution.modes - 1.0j * inverse_scale * evolution.derivatives
    occupation = jnp.abs(beta) ** 2
    normalization = jnp.abs(alpha) ** 2 - occupation
    residual = jnp.abs(normalization - 1.0)
    maximum = jnp.max(residual)
    normalized = jnp.all(jnp.isfinite(normalization)) & (maximum <= tolerance_)
    return BogoliubovEvidence(
        alpha=alpha,
        beta=beta,
        occupation_numbers=occupation,
        normalization=normalization,
        normalization_residual=residual,
        maximum_normalization_residual=maximum,
        normalized=normalized,
        prepared_id=prepared.prepared_id,
        claim="instantaneous-finite-mode-particle-number-reference-only",
    )


__all__ = [
    "BogoliubovEvidence",
    "FLRWModePlan",
    "ModeEvolutionEvidence",
    "ModeInitialState",
    "PreparedFLRWModes",
    "adiabatic_frequencies",
    "adiabatic_initial_state",
    "bogoliubov_particle_production",
    "differentiate_time",
    "evolve_flrw_modes",
    "gauss_legendre_flrw_mode_plan",
    "prepare_flrw_modes",
]
