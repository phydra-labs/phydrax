"""Analytic, adaptive, and symplectic two-body propagation with one JVP."""

import jax
import jax.numpy as jnp

import phydrax as phx


def build_workflow():
    astro = phx.applications.astrodynamics
    context = astro.AstrodynamicsContext(
        astro.AstrodynamicsScaleContract.si(),
        astro.ReferenceEpoch(2451545.0, 0.0, "TT"),
        astro.AstrodynamicsFrame("central-body", "inertial", pseudo_inertial=True),
    )
    initial = astro.CartesianOrbitState(
        jnp.asarray([1.0, 0.0, 0.0]),
        jnp.asarray([0.0, 1.0, 0.0]),
        context,
    )
    times = jnp.linspace(0.0, 2.0 * jnp.pi, 65)
    force = astro.PointMassGravity(1.0, context)
    plan = astro.AstrodynamicsPropagationPlan(force, times)
    return context, initial, plan


def main():
    _, initial, plan = build_workflow()
    result = plan.solve_analytic_two_body(initial)
    if not bool(result.successful):
        raise RuntimeError("Two-body propagation failed.")
    tangent = jax.jvp(
        lambda dt: (
            phx.applications.astrodynamics.propagate_universal_kepler(
                initial, dt, 1.0
            ).state.position
        ),
        (jnp.asarray(1.0),),
        (jnp.asarray(1.0),),
    )[1]
    print(result.trajectory.states[-1])
    print(tangent)


if __name__ == "__main__":
    main()
