#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ._modes import differentiate_time
from ._renormalization import RenormalizedStressEvidence


class SemiclassicalEinsteinPlan(StrictModule):
    """Finite source-driven flat-FLRW semiclassical Einstein reference plan."""

    cosmic_times: Array
    initial_scale_factor: float = eqx.field(static=True)
    initial_hubble: float = eqx.field(static=True)
    newton_constant: float = eqx.field(static=True)
    cosmological_constant: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    maximum_time_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cosmic_times: ArrayLike,
        /,
        *,
        initial_scale_factor: float,
        initial_hubble: float,
        newton_constant: float,
        cosmological_constant: float = 0.0,
        residual_tolerance: float = 1e-6,
        maximum_time_steps: int = 100_000,
    ):
        times = np.asarray(cosmic_times, dtype=np.float64)
        scale = float(initial_scale_factor)
        hubble = float(initial_hubble)
        gravity = float(newton_constant)
        cosmological = float(cosmological_constant)
        tolerance = float(residual_tolerance)
        maximum = int(maximum_time_steps)
        if times.ndim != 1 or times.size < 3 or np.any(np.diff(times) <= 0.0):
            raise ValueError(
                "cosmic_times must be strictly increasing with at least three nodes."
            )
        if not np.all(np.isfinite(times)):
            raise ValueError("cosmic_times must be finite.")
        if scale <= 0.0 or gravity <= 0.0 or tolerance < 0.0:
            raise ValueError("Scale/gravity must be positive and tolerance nonnegative.")
        if not all(
            np.isfinite(value)
            for value in (scale, hubble, gravity, cosmological, tolerance)
        ):
            raise ValueError("Semiclassical plan scalars must be finite.")
        if maximum < times.size:
            raise ValueError("cosmic time grid exceeds maximum_time_steps.")
        self.cosmic_times = jnp.asarray(times)
        self.initial_scale_factor = scale
        self.initial_hubble = hubble
        self.newton_constant = gravity
        self.cosmological_constant = cosmological
        self.residual_tolerance = tolerance
        self.maximum_time_steps = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-source-driven-semiclassical-einstein-plan",
                "cosmic_times": array_tree_fingerprint(times),
                "initial_scale_factor": scale,
                "initial_hubble": hubble,
                "newton_constant": gravity,
                "cosmological_constant": cosmological,
                "residual_tolerance": tolerance,
                "maximum_time_steps": maximum,
            }
        )


class PreparedSemiclassicalEinstein(StrictModule):
    plan: SemiclassicalEinsteinPlan
    interval_widths: Array
    raychaudhuri_factor: Array
    prepared_id: str = eqx.field(static=True)


class SemiclassicalBackreactionEvidence(StrictModule):
    scale_factors: Array
    hubble_parameters: Array
    energy_density: Array
    pressure: Array
    friedmann_residual: Array
    relative_friedmann_residual: Array
    acceleration_residual: Array
    continuity_residual: Array
    maximum_relative_friedmann_residual: Array
    maximum_relative_continuity_residual: Array
    finite: Array
    self_consistent: Array
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def prepare_semiclassical_einstein(
    plan: SemiclassicalEinsteinPlan, /
) -> PreparedSemiclassicalEinstein:
    if not isinstance(plan, SemiclassicalEinsteinPlan):
        raise TypeError("plan must be SemiclassicalEinsteinPlan.")
    return PreparedSemiclassicalEinstein(
        plan=plan,
        interval_widths=jnp.diff(plan.cosmic_times),
        raychaudhuri_factor=jnp.asarray(4.0 * jnp.pi * plan.newton_constant),
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-finite-semiclassical-einstein-reference",
                "plan": plan.plan_id,
                "equations": ("Hdot=-4piG(rho+p)", "adot=aH"),
            }
        ),
    )


def semiclassical_einstein_backreaction(
    prepared: PreparedSemiclassicalEinstein,
    stress: RenormalizedStressEvidence,
    /,
) -> SemiclassicalBackreactionEvidence:
    """Evolve a finite source history and expose both Einstein constraints."""
    if not isinstance(prepared, PreparedSemiclassicalEinstein):
        raise TypeError("prepared must be PreparedSemiclassicalEinstein.")
    if not isinstance(stress, RenormalizedStressEvidence):
        raise TypeError("stress must be RenormalizedStressEvidence.")
    plan = prepared.plan
    energy = jnp.asarray(stress.renormalized_energy_density)
    pressure = jnp.asarray(stress.renormalized_pressure)
    expected = plan.cosmic_times.shape
    if energy.shape != expected or pressure.shape != expected:
        raise ValueError("Stress history must match the semiclassical cosmic time grid.")
    initial_scale = jnp.asarray(plan.initial_scale_factor, dtype=energy.dtype)
    initial_hubble = jnp.asarray(plan.initial_hubble, dtype=energy.dtype)
    interval_source = 0.5 * (energy[:-1] + pressure[:-1] + energy[1:] + pressure[1:])

    def backreaction_step(carry, interval_data):
        current_scale, current_hubble = carry
        width, average_source = interval_data
        next_hubble = current_hubble - (
            prepared.raychaudhuri_factor * width * average_source
        )
        next_scale = current_scale * jnp.exp(0.5 * width * (current_hubble + next_hubble))
        return (next_scale, next_hubble), (next_scale, next_hubble)

    _, (scale_tail, hubble_tail) = lax.scan(
        backreaction_step,
        (initial_scale, initial_hubble),
        (prepared.interval_widths, interval_source),
    )
    scale = jnp.concatenate((initial_scale[None], scale_tail), axis=0)
    hubble = jnp.concatenate((initial_hubble[None], hubble_tail), axis=0)
    matter_term = (8.0 * jnp.pi * plan.newton_constant / 3.0) * energy
    cosmological_term = plan.cosmological_constant / 3.0
    friedmann = hubble * hubble - matter_term - cosmological_term
    friedmann_scale = jnp.maximum(
        1.0, hubble * hubble + jnp.abs(matter_term) + abs(cosmological_term)
    )
    relative_friedmann = jnp.abs(friedmann) / friedmann_scale
    hubble_prime = jnp.diff(hubble) / prepared.interval_widths
    acceleration = hubble_prime + prepared.raychaudhuri_factor * interval_source
    energy_prime = differentiate_time(energy, plan.cosmic_times)
    continuity_expansion = 3.0 * hubble * (energy + pressure)
    continuity = energy_prime + continuity_expansion
    continuity_scale = jnp.maximum(
        1.0, jnp.abs(energy_prime) + jnp.abs(continuity_expansion)
    )
    relative_continuity = jnp.abs(continuity) / continuity_scale
    max_friedmann = jnp.max(relative_friedmann)
    max_continuity = jnp.max(relative_continuity)
    finite = (
        jnp.all(jnp.isfinite(scale))
        & jnp.all(jnp.isfinite(hubble))
        & jnp.all(jnp.isfinite(relative_friedmann))
        & jnp.all(jnp.isfinite(relative_continuity))
    )
    accepted = (
        finite
        & stress.conserved
        & (max_friedmann <= plan.residual_tolerance)
        & (max_continuity <= plan.residual_tolerance)
    )
    return SemiclassicalBackreactionEvidence(
        scale_factors=scale,
        hubble_parameters=hubble,
        energy_density=energy,
        pressure=pressure,
        friedmann_residual=friedmann,
        relative_friedmann_residual=relative_friedmann,
        acceleration_residual=acceleration,
        continuity_residual=continuity,
        maximum_relative_friedmann_residual=max_friedmann,
        maximum_relative_continuity_residual=max_continuity,
        finite=finite,
        self_consistent=accepted,
        prepared_id=prepared.prepared_id,
        claim=(
            "finite-source-driven-research-reference-only;"
            "self_consistent-is-false-unless-friedmann-and-conservation-evidence-pass"
        ),
    )


__all__ = [
    "PreparedSemiclassicalEinstein",
    "SemiclassicalBackreactionEvidence",
    "SemiclassicalEinsteinPlan",
    "prepare_semiclassical_einstein",
    "semiclassical_einstein_backreaction",
]
