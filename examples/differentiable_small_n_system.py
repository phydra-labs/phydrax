"""Static-capacity direct N-body rollout with conservation diagnostics."""

import jax.numpy as jnp

import phydrax as phx


def main():
    astro = phx.applications.astrodynamics
    context = astro.AstrodynamicsContext(
        astro.AstrodynamicsScaleContract.si(),
        astro.ReferenceEpoch(2451545.0, 0.0, "TT"),
        astro.AstrodynamicsFrame("barycenter", "inertial", pseudo_inertial=True),
    )
    particles = phx.discretization.particle.ParticleSetPlan(
        jnp.asarray([0, 1]),
        jnp.asarray([1.0, 1.0]),
        ambient_dimension=3,
    ).prepare()
    state = astro.NBodyState(
        jnp.asarray([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        jnp.asarray([[0.0, -0.7, 0.0], [0.0, 0.7, 0.0]]),
        particles,
        context,
    )
    gravity = astro.DirectNBodyGravityPlan(particles, context)
    result = astro.NBodyPropagationPlan(gravity, jnp.linspace(0.0, 1.0, 257)).rollout(
        state
    )
    if not bool(result.successful):
        raise RuntimeError("N-body rollout failed.")
    print(jnp.max(jnp.abs(result.energy - result.energy[0])))
    print(result.linear_momentum[-1])


if __name__ == "__main__":
    main()
