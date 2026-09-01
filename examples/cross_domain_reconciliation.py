"""Shared core contracts across cosmology, astrodynamics, and astrophysics."""

from __future__ import annotations

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    cosmology = phx.applications.cosmology
    astrodynamics = phx.applications.astrodynamics
    scale = phx.DimensionalScaleContract(
        "m", "kg", "s", length_coordinate_kind="physical"
    )
    positions = jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    masses = jnp.ones((2,))
    direct, evidence = phx.solver.DirectParticleGravityPlan(
        phx.solver.NewtonianPairKernel(1.0, softening=1.0e-15)
    ).evaluate(positions, masses)
    tree = astrodynamics.PreparedOctree3D(positions, masses, leaf_capacity=1)
    hierarchical = astrodynamics.BarnesHutGravityPlan3D(tree, masses).evaluate(positions)

    source = phx.observation.CoordinateLayout(("source:0", "source:1"))
    target = phx.observation.CoordinateLayout(("target:0",))
    response = phx.observation.LinearObservationPlan([[1.0, 2.0]], source, target)
    predicted = response.apply(
        phx.observation.TheoryVector([2.0, 3.0], source, "example")
    )

    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2), masses, ambient_dimension=1
    ).prepare()
    kdk = cosmology.CosmologicalKDKPlan(particles, (10.0,))
    state = kdk.initialize([[1.0], [2.0]], [[0.0], [0.0]], 0.5)

    print("scale_id", scale.scale_id)
    print("direct_acceleration", direct)
    print("hierarchical_acceleration", hierarchical.acceleration)
    print("net_force", evidence.net_force)
    print("observed_product", predicted.values)
    print("cosmology_state_scale", state.scale_factor)


if __name__ == "__main__":
    main()
