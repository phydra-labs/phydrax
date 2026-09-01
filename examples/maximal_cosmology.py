"""Bounded native maximal-physics profiles across cosmology and gravity."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import phydrax as phx


def artifact(kind):
    return phx.applications.cosmology.ScientificArtifactEnvelope(
        artifact_kind=kind,
        content_digest=f"{kind}-example",
        producer="example",
        producer_version="current",
        build_id="example",
        license_id="internal",
        resource_id="static",
        status="complete",
    )


def main() -> None:
    cosmo = phx.applications.cosmology
    profile = cosmo.ParityProfile(
        name="restricted-native-scalar",
        equations=("linear-Einstein-Boltzmann",),
        species=("photon", "baryon", "cdm", "massless-relic"),
        geometry="flat-FLRW",
        approximations=("scalar-adiabatic", "fixed-layout"),
        outputs=("transfer", "unlensed-CMB"),
        references=("qualified-external-oracles",),
        metrics=("state", "spectra"),
        negative_boundaries=("massive-relic", "curvature", "lensing"),
    )
    parity = cosmo.ParityEvidence(
        profile, [1.0e-5, 2.0e-5], [2.0e-5, 2.0e-5], artifact("parity")
    )

    positions = jnp.asarray(
        [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.8, 0.8, 0.8], [0.9, 0.8, 0.8]]
    )
    tree = cosmo.ParticleOctreePlan3D((1.0, 1.0, 1.0), 2).prepare(
        positions, jnp.ones((4,))
    )
    bh = cosmo.BarnesHutGravityPlan(1.0, softening=0.01, opening_angle=0.5).evaluate(tree)
    fmm = cosmo.UniformFMMPlan(
        1.0, cosmo.CartesianExpansionSpace(1), softening=0.01
    ).evaluate(tree)

    manifold = cosmo.S3ManifoldPlan(2.0)
    s3_point = jnp.asarray([[2.0, 0.0, 0.0, 0.0]])
    s3_target = manifold.exponential(s3_point, jnp.asarray([[0.0, 0.1, 0.0, 0.0]]))

    population = cosmo.CosmologicalPopulationPlan(4, 3).empty()
    stars = cosmo.StochasticStarFormationPlan(star_mass=0.5, maximum_events=1).apply(
        population,
        [1.0],
        [[0.0, 0.0, 0.0]],
        [1.0],
        [0.01],
        [[0.5, 0.5, 0.5]],
        [True],
        0.5,
        jax.random.key(1),
        0,
    )

    print("parity_profile", bool(parity.successful))
    print("barnes_hut_net_force", bh.evidence.net_force)
    print("fmm_error_indicator", fmm.evidence.estimated_relative_error)
    print("s3_distance", manifold.distance(s3_point, s3_target))
    print("star_events", stars.ledger.event_count)


if __name__ == "__main__":
    main()
