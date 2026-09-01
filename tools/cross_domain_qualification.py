"""Qualification evidence for reconciled cross-domain substrates."""

from __future__ import annotations

import json

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    astro = phx.applications.astrodynamics
    astrophysics = phx.applications.astrophysics
    positions = jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    masses = jnp.ones((2,))
    direct, evidence = phx.solver.DirectParticleGravityPlan(
        phx.solver.NewtonianPairKernel(1.0, softening=1.0e-15)
    ).evaluate(positions, masses)
    tree = astro.PreparedOctree3D(positions, masses, leaf_capacity=1)
    hierarchical = astro.BarnesHutGravityPlan3D(tree, masses).evaluate(positions)

    binned = astrophysics.BinnedResponsePlan(
        [[1.0, 2.0], [0.5, 0.5]], response_id="fixture"
    ).evaluate([2.0, 3.0])
    source = phx.observation.CoordinateLayout(("s0", "s1"))
    target = phx.observation.CoordinateLayout(("d0", "d1"))
    core_response = phx.observation.LinearObservationPlan(
        [[1.0, 2.0], [0.5, 0.5]], source, target
    ).apply(phx.observation.TheoryVector([2.0, 3.0], source, "fixture"))

    amr = phx.discretization.TwoLevelAMRPlan((2,), 1)
    coarse = jnp.asarray([[1.0], [2.0]])
    restricted = amr.restrict(amr.prolong(coarse))

    report = {
        "gravity_adapter_error": float(
            jnp.max(jnp.abs(hierarchical.acceleration - direct))
        ),
        "gravity_net_force": float(jnp.max(jnp.abs(evidence.net_force))),
        "binned_response_error": float(
            jnp.max(jnp.abs(binned.predicted - core_response.values))
        ),
        "amr_prolong_restrict_error": float(jnp.max(jnp.abs(restricted - coarse))),
        "astrodynamics_scale_is_core": (
            astro.AstrodynamicsScaleContract is phx.DimensionalScaleContract
        ),
        "cosmology_scale_is_core": (
            phx.applications.cosmology.CosmologyScaleContract
            is phx.DimensionalScaleContract
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
