#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Regulated finite-slice real-time path integrals.

This module estimates only the caller-declared finite-dimensional integral at a
strictly positive bridge regulator. It does not extrapolate the regulator to
zero or the slice count to infinity.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from ...discretization import TemporalMesh
from ._action import kinetic_action, potential_action
from ._sampling import _endpoints, brownian_bridge_from_noise


class RealTimePathIntegralPlan(StrictModule):
    """Immutable finite-population regulated real-time plan."""

    slicing: TemporalMesh
    mass: float = eqx.field(static=True)
    hbar: float = eqx.field(static=True)
    regulator: float = eqx.field(static=True)
    num_paths: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    minimum_mean_phase: float = eqx.field(static=True)

    def __init__(
        self,
        slicing: TemporalMesh,
        /,
        *,
        mass: float,
        hbar: float = 1.0,
        regulator: float,
        num_paths: int,
        chunk_size: int | None = None,
        minimum_mean_phase: float = 1e-3,
    ):
        if not isinstance(slicing, TemporalMesh) or slicing.role != "path":
            raise TypeError("slicing must be a uniform path TemporalMesh.")
        mass_, hbar_, regulator_ = float(mass), float(hbar), float(regulator)
        count = int(num_paths)
        chunk = count if chunk_size is None else int(chunk_size)
        phase = float(minimum_mean_phase)
        if not all(np.isfinite(v) and v > 0.0 for v in (mass_, hbar_, regulator_)):
            raise ValueError("mass, hbar, and regulator must be finite and positive.")
        if count <= 0 or chunk <= 0 or chunk > count:
            raise ValueError(
                "num_paths and chunk_size must satisfy 1 <= chunk_size <= num_paths."
            )
        if not np.isfinite(phase) or not 0.0 <= phase <= 1.0:
            raise ValueError("minimum_mean_phase must lie in [0, 1].")
        self.slicing = slicing
        self.mass = mass_
        self.hbar = hbar_
        self.regulator = regulator_
        self.num_paths = count
        self.chunk_size = chunk
        self.minimum_mean_phase = phase


class RealTimeRegulatorContinuation(StrictModule):
    """A finite positive regulator sequence; no zero-limit fit is performed."""

    regulators: tuple[float, ...] = eqx.field(static=True)

    def __init__(self, regulators: Sequence[float], /):
        values = tuple(float(value) for value in regulators)
        if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("regulators must be a nonempty finite positive sequence.")
        self.regulators = values


class OscillatoryPathIntegralEstimate(StrictModule):
    """Complex estimate and cancellation evidence for one finite regulator."""

    value: Array
    covariance: Array
    standard_error: Array
    mean_phase: Array
    phase_effective_sample_size: Array
    regulator: Array
    valid: Array
    unresolved_sign_problem: Array
    num_paths: int = eqx.field(static=True)
    num_slices: int = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class RealTimeContinuationResult(StrictModule):
    values: Array
    standard_errors: Array
    mean_phases: Array
    phase_effective_sample_sizes: Array
    successive_differences: Array
    valid: Array
    regulators: tuple[float, ...] = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def _regulator_volume(x0: Array, x1: Array, plan: RealTimePathIntegralPlan, /) -> Array:
    dimension = int(x0.shape[-1])
    steps = plan.slicing.num_steps
    dt = plan.slicing.dt
    displacement = jnp.sum((x1 - x0) ** 2, axis=-1)
    log_volume = (
        0.5 * dimension * (steps - 1) * jnp.log(2.0 * jnp.pi * dt / plan.regulator)
        - 0.5 * dimension * jnp.log(float(steps))
        - plan.regulator * displacement / (2.0 * plan.slicing.duration)
    )
    return jnp.exp(log_volume)


def _real_time_prefactor(plan: RealTimePathIntegralPlan, dimension: int, /) -> Array:
    base = plan.mass / (2.0j * jnp.pi * plan.hbar * plan.slicing.dt)
    return base ** (0.5 * dimension * plan.slicing.num_steps)


def real_time_kernel_from_noise(
    noise: ArrayLike,
    x0: ArrayLike,
    x1: ArrayLike,
    potential: Callable[[Array, Array], Array],
    /,
    *,
    plan: RealTimePathIntegralPlan,
) -> OscillatoryPathIntegralEstimate:
    """Evaluate the exact importance ratio for a regulated finite-slice integral."""
    if not isinstance(plan, RealTimePathIntegralPlan):
        raise TypeError("plan must be a RealTimePathIntegralPlan.")
    start, end = _endpoints(x0, x1)
    if start.ndim != 1 or end.ndim != 1:
        raise ValueError("real-time kernels currently require unbatched endpoints.")
    z = jnp.asarray(noise)
    expected = (plan.num_paths, plan.slicing.num_steps, int(start.shape[-1]))
    if z.shape != expected:
        raise ValueError(f"noise must have shape {expected}; got {z.shape}.")
    paths = brownian_bridge_from_noise(
        z,
        start,
        end,
        slicing=plan.slicing,
        diffusion=1.0 / plan.regulator,
    )
    kinetic = kinetic_action(paths, slicing=plan.slicing, mass=plan.mass)
    potential_values = potential_action(paths, potential, slicing=plan.slicing)
    action = kinetic - potential_values
    phase = jnp.exp(1j * action / plan.hbar)
    volume = _regulator_volume(start, end, plan)
    prefactor = _real_time_prefactor(plan, int(start.shape[-1]))
    samples = prefactor * volume * phase
    value = jnp.mean(samples)
    components = jnp.stack((jnp.real(samples), jnp.imag(samples)), axis=-1)
    centered = components - jnp.mean(components, axis=0)
    denominator = max(plan.num_paths - 1, 1)
    covariance = centered.T @ centered / denominator
    standard_error = jnp.sqrt(jnp.trace(covariance) / plan.num_paths)
    mean_phase = jnp.abs(jnp.mean(phase))
    phase_ess = jnp.abs(jnp.sum(phase)) ** 2 / plan.num_paths
    finite = jnp.all(jnp.isfinite(samples)) & jnp.all(jnp.isfinite(covariance))
    return OscillatoryPathIntegralEstimate(
        value=value,
        covariance=covariance,
        standard_error=jnp.where(plan.num_paths > 1, standard_error, jnp.nan),
        mean_phase=mean_phase,
        phase_effective_sample_size=phase_ess,
        regulator=jnp.asarray(plan.regulator),
        valid=finite,
        unresolved_sign_problem=mean_phase < plan.minimum_mean_phase,
        num_paths=plan.num_paths,
        num_slices=plan.slicing.num_steps,
        claim="regulated-finite-slice-only",
    )


def real_time_kernel(
    x0: ArrayLike,
    x1: ArrayLike,
    potential: Callable[[Array, Array], Array],
    /,
    *,
    plan: RealTimePathIntegralPlan,
    key: Key[Array, ""] = DOC_KEY0,
) -> OscillatoryPathIntegralEstimate:
    start, _ = _endpoints(x0, x1)
    if start.ndim != 1:
        raise ValueError("real-time kernels currently require unbatched endpoints.")
    noise = jr.normal(
        key,
        (plan.num_paths, plan.slicing.num_steps, int(start.shape[-1])),
        dtype=start.dtype,
    )
    return real_time_kernel_from_noise(noise, x0, x1, potential, plan=plan)


def continue_real_time_regulator_from_noise(
    noise: ArrayLike,
    x0: ArrayLike,
    x1: ArrayLike,
    potential: Callable[[Array, Array], Array],
    /,
    *,
    plan: RealTimePathIntegralPlan,
    continuation: RealTimeRegulatorContinuation,
) -> RealTimeContinuationResult:
    """Compare a finite regulator sequence with common-noise paired differences."""
    if not isinstance(continuation, RealTimeRegulatorContinuation):
        raise TypeError("continuation must be RealTimeRegulatorContinuation.")
    estimates = tuple(
        real_time_kernel_from_noise(
            noise,
            x0,
            x1,
            potential,
            plan=eqx.tree_at(lambda value: value.regulator, plan, regulator),
        )
        for regulator in continuation.regulators
    )
    values = jnp.stack([estimate.value for estimate in estimates])
    return RealTimeContinuationResult(
        values=values,
        standard_errors=jnp.stack([estimate.standard_error for estimate in estimates]),
        mean_phases=jnp.stack([estimate.mean_phase for estimate in estimates]),
        phase_effective_sample_sizes=jnp.stack(
            [estimate.phase_effective_sample_size for estimate in estimates]
        ),
        successive_differences=values[1:] - values[:-1],
        valid=jnp.stack([estimate.valid for estimate in estimates]),
        regulators=continuation.regulators,
        claim="finite-regulator-comparison-not-zero-limit-extrapolation",
    )


__all__ = [
    "OscillatoryPathIntegralEstimate",
    "RealTimeContinuationResult",
    "RealTimePathIntegralPlan",
    "RealTimeRegulatorContinuation",
    "continue_real_time_regulator_from_noise",
    "real_time_kernel",
    "real_time_kernel_from_noise",
]
