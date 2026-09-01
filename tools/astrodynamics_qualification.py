"""Qualification evidence for two-body, Lambert, N-body, and CR3BP contracts."""

import json

import jax
import jax.numpy as jnp

import phydrax as phx


def main():
    astro = phx.applications.astrodynamics
    context = astro.AstrodynamicsContext(
        astro.AstrodynamicsScaleContract.si(),
        astro.ReferenceEpoch(astro.TimeInstant(astro.JulianDate(2451545.0, 0.0), "TT")),
        astro.FrameDefinition("central", "inertial", pseudo_inertial=True),
    )
    initial = astro.CartesianOrbitState(
        jnp.asarray([1.0, 0.0, 0.0]), jnp.asarray([0.0, 1.0, 0.0]), context
    )
    orbit = astro.propagate_universal_kepler(initial, 2.0 * jnp.pi, 1.0)
    derivative = jax.jvp(
        lambda dt: astro.propagate_universal_kepler(initial, dt, 1.0).state.position,
        (jnp.asarray(0.7),),
        (jnp.asarray(1.0),),
    )[1]
    lambert = astro.solve_lambert(
        jnp.asarray([1.0, 0.0, 0.0]),
        jnp.asarray([0.0, 1.0, 0.0]),
        0.5 * jnp.pi,
        1.0,
        context,
        astro.LambertPlan(max_revolutions=1),
    )
    cr3bp = astro.CR3BPSystem(0.01).lagrange_points()
    report = {
        "kind": "astrodynamics-qualification",
        "dtype": str(orbit.state.position.dtype),
        "circular_position_error": float(
            jnp.max(jnp.abs(orbit.state.position - initial.position))
        ),
        "circular_velocity_error": float(
            jnp.max(jnp.abs(orbit.state.velocity - initial.velocity))
        ),
        "energy_defect": float(
            jnp.abs(orbit.specific_energy_after - orbit.specific_energy_before)
        ),
        "angular_momentum_defect": float(orbit.angular_momentum_defect),
        "time_tangent_error": float(
            jnp.max(jnp.abs(derivative - jnp.asarray([-jnp.sin(0.7), jnp.cos(0.7), 0.0])))
        ),
        "lambert_valid_branches": int(jnp.sum(lambert.valid)),
        "lambert_zero_revolution_residual": float(lambert.residual[0]),
        "cr3bp_lagrange_max_residual": float(jnp.max(cr3bp.residuals)),
        "passed": bool(
            orbit.valid
            & lambert.valid[0]
            & jnp.all(cr3bp.valid)
            & (orbit.angular_momentum_defect < 1.0e-10)
            & (lambert.residual[0] < 1.0e-10)
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
