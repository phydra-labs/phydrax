"""Deterministic D2Q9 decaying-shear qualification for Smagorinsky collision."""

from __future__ import annotations

import argparse
import json

import jax.numpy as jnp
import numpy as np

import phydrax as phx
import phydrax.ein as ein
from phydrax.discretization.lattice_boltzmann._collision import (
    quadratic_equilibrium,
)


def _macroscopic(populations, velocity_set):
    density = jnp.sum(populations, axis=-1)
    momentum = ein.contract("...q,qd->...d", populations, velocity_set.velocities)
    return density, momentum / density[..., None]


def _stream(populations, velocity_set):
    axes = tuple(range(velocity_set.dimension))
    return jnp.stack(
        tuple(
            jnp.roll(populations[..., index], shift=offset, axis=axes)
            for index, offset in enumerate(velocity_set.velocity_tuples)
        ),
        axis=-1,
    )


def _mode_amplitude(populations, velocity_set, mode):
    _, velocity = _macroscopic(populations, velocity_set)
    return 2.0 * jnp.mean(velocity[..., 0] * mode)


def _evolve(initial, velocity_set, precision, rate, coefficient, steps, mode):
    method = phx.discretization.LatticeBoltzmannMethodPlan(
        phx.discretization.SmagorinskyCollisionPlan(coefficient)
    ).prepare(velocity_set, precision)
    populations = initial
    initial_mass = jnp.sum(populations)
    initial_density, initial_velocity = _macroscopic(populations, velocity_set)
    initial_momentum = jnp.sum(initial_density[..., None] * initial_velocity, axis=(0, 1))
    maximum_mass_defect = jnp.asarray(0.0, dtype=populations.dtype)
    maximum_momentum_defect = jnp.asarray(0.0, dtype=populations.dtype)
    maximum_effective_viscosity = jnp.asarray(0.0, dtype=populations.dtype)
    coefficient_active = jnp.asarray(False)
    successful = jnp.asarray(True)
    evidence = None

    for _ in range(steps):
        density, velocity = _macroscopic(populations, velocity_set)
        result = method.collide(
            populations,
            density,
            velocity,
            jnp.zeros_like(velocity),
            rate,
            velocity_set,
            precision,
        )
        evidence = result.smagorinsky_evidence
        if evidence is None:
            raise RuntimeError("Smagorinsky collision did not return evidence.")
        maximum_mass_defect = jnp.maximum(
            maximum_mass_defect, result.diagnostics.mass_error
        )
        maximum_momentum_defect = jnp.maximum(
            maximum_momentum_defect, result.diagnostics.momentum_error
        )
        maximum_effective_viscosity = jnp.maximum(
            maximum_effective_viscosity,
            jnp.max(evidence.effective_kinematic_viscosity),
        )
        coefficient_active = coefficient_active | jnp.any(evidence.coefficient_active)
        successful = successful & result.successful & evidence.successful
        populations = _stream(result.populations, velocity_set)

    if evidence is None:
        raise ValueError("steps must be positive.")
    final_mass = jnp.sum(populations)
    final_density, final_velocity = _macroscopic(populations, velocity_set)
    final_momentum = jnp.sum(final_density[..., None] * final_velocity, axis=(0, 1))
    return {
        "final_amplitude": float(_mode_amplitude(populations, velocity_set, mode)),
        "global_mass_drift": float(jnp.abs(final_mass - initial_mass)),
        "global_momentum_drift": float(
            jnp.max(jnp.abs(final_momentum - initial_momentum))
        ),
        "maximum_collision_mass_defect": float(maximum_mass_defect),
        "maximum_collision_momentum_defect": float(maximum_momentum_defect),
        "maximum_effective_kinematic_viscosity": float(maximum_effective_viscosity),
        "coefficient_active": bool(coefficient_active),
        "finite": bool(evidence.finite),
        "successful": bool(successful),
        "support_satisfied": bool(evidence.support_satisfied),
        "support": {
            "coefficient_lower_bound": evidence.coefficient_lower_bound,
            "coefficient_requires_finite": evidence.coefficient_requires_finite,
            "base_relaxation_rate_bounds": list(evidence.base_relaxation_rate_bounds),
            "base_relaxation_rate_bounds_exclusive": (
                evidence.base_relaxation_rate_bounds_exclusive
            ),
            "density_lower_bound": evidence.density_lower_bound,
            "density_lower_bound_exclusive": evidence.density_lower_bound_exclusive,
            "filter_width_in_lattice_units": (evidence.filter_width_in_lattice_units),
        },
    }


def qualification(
    *,
    resolution=24,
    steps=24,
    amplitude=0.05,
    base_relaxation_rate=1.25,
    coefficient=0.18,
):
    """Run one periodic, athermal D2Q9 decaying-shear qualification case."""
    resolution = int(resolution)
    steps = int(steps)
    amplitude = float(amplitude)
    rate = float(base_relaxation_rate)
    coefficient = float(coefficient)
    if resolution < 8:
        raise ValueError("resolution must be at least 8.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("amplitude must be finite and positive.")
    if not np.isfinite(rate) or not 0.0 < rate < 2.0:
        raise ValueError("base_relaxation_rate must lie in (0, 2).")
    if not np.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("qualification coefficient must be finite and positive.")

    velocity_set = phx.discretization.D2Q9()
    precision = phx.discretization.LatticeBoltzmannPrecisionPolicy()
    wave_number = 2.0 * np.pi / resolution
    coordinate = jnp.arange(resolution, dtype=jnp.float64)
    mode = jnp.broadcast_to(
        jnp.sin(wave_number * coordinate)[None, :],
        (resolution, resolution),
    )
    density = jnp.ones((resolution, resolution), dtype=jnp.float64)
    velocity = jnp.zeros((resolution, resolution, 2), dtype=jnp.float64)
    velocity = velocity.at[..., 0].set(amplitude * mode)
    initial = quadratic_equilibrium(density, velocity, velocity_set, precision)
    molecular = _evolve(initial, velocity_set, precision, rate, 0.0, steps, mode)
    smagorinsky = _evolve(
        initial, velocity_set, precision, rate, coefficient, steps, mode
    )
    molecular_viscosity = float(velocity_set.sound_speed_squared * (1.0 / rate - 0.5))
    analytic_amplitude = amplitude * np.exp(-molecular_viscosity * wave_number**2 * steps)
    molecular_reference_relative_error = (
        abs(molecular["final_amplitude"] - analytic_amplitude) / analytic_amplitude
    )
    tolerance = 1.0e-11
    passed = (
        molecular["successful"]
        and smagorinsky["successful"]
        and molecular["support_satisfied"]
        and smagorinsky["support_satisfied"]
        and smagorinsky["coefficient_active"]
        and 0.0 < molecular["final_amplitude"] < amplitude
        and 0.0 < smagorinsky["final_amplitude"] < molecular["final_amplitude"]
        and molecular_reference_relative_error < 0.05
        and molecular["global_mass_drift"] < tolerance
        and smagorinsky["global_mass_drift"] < tolerance
        and molecular["global_momentum_drift"] < tolerance
        and smagorinsky["global_momentum_drift"] < tolerance
    )
    return {
        "case": "periodic-decaying-shear-d2q9",
        "parameters": {
            "resolution": resolution,
            "steps": steps,
            "initial_amplitude": amplitude,
            "wave_number": wave_number,
            "base_relaxation_rate": rate,
            "coefficient": coefficient,
        },
        "reference": {
            "model": "linear-molecular-shear-decay",
            "molecular_kinematic_viscosity": molecular_viscosity,
            "analytic_final_amplitude": analytic_amplitude,
            "relative_error": molecular_reference_relative_error,
        },
        "molecular": molecular,
        "smagorinsky": smagorinsky,
        "additional_amplitude_decay": (
            molecular["final_amplitude"] - smagorinsky["final_amplitude"]
        ),
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=24)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--amplitude", type=float, default=0.05)
    parser.add_argument("--base-relaxation-rate", type=float, default=1.25)
    parser.add_argument("--coefficient", type=float, default=0.18)
    arguments = parser.parse_args()
    payload = qualification(
        resolution=arguments.resolution,
        steps=arguments.steps,
        amplitude=arguments.amplitude,
        base_relaxation_rate=arguments.base_relaxation_rate,
        coefficient=arguments.coefficient,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
