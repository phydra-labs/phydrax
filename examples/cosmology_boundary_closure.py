"""Boundary-closure contracts across cosmology theory, data, and simulation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    cosmo = phx.applications.cosmology
    background = cosmo.FLRWBackground(1.0, 0.3, curvature_density=0.01)
    validity = cosmo.LocalCurvatureValidityPlan(
        light_speed=1.0,
        geometry_error_budget=1.0e-3,
        support_kind="periodic-box-diagonal",
    ).evaluate(background, 0.1)

    source = cosmo.CoordinateLayout(("P0:k0", "P2:k0", "P4:k0"))
    observed = cosmo.CoordinateLayout(("d0", "d1"))
    observation = cosmo.LinearObservationPlan(
        [[1.0, 0.5, 0.0], [0.0, 0.25, 1.0]], source, observed
    )
    likelihood = cosmo.CorrelatedGaussianPlan(
        [2.0, 1.0],
        observation,
        cosmo.PrecisionCovarianceAction(jnp.eye(2), 0.0, observed),
    )
    likelihood_result = likelihood.evaluate(
        cosmo.TheoryVector([1.0, 2.0, 0.5], source, "example-theory")
    )

    hod = cosmo.Zheng07OccupationExpectation200m(12.0, 0.2, 1.0e11, 1.0e13, 1.0).evaluate(
        jnp.asarray([1.0e12, 1.0e13, 1.0e14])
    )

    ewald = cosmo.PeriodicEwaldForcePlan(
        (1.0, 1.0, 1.0),
        1.0,
        softening=0.02,
        alpha=5.0,
        real_shells=2,
        reciprocal_modes=4,
    )
    force = ewald.evaluate(
        jnp.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]),
        jnp.ones((2,)),
    )

    artifact = cosmo.ScientificArtifactEnvelope(
        artifact_kind="sht-fixture",
        content_digest="example",
        producer="example",
        producer_version="current",
        build_id="example",
        license_id="internal",
        resource_id="static",
        status="complete",
    )
    synthesis = jnp.zeros((36, 3))
    for pixel in range(12):
        synthesis = synthesis.at[3 * pixel : 3 * pixel + 3].set(jnp.eye(3))
    sky = cosmo.HarmonicSkySynthesisPlan(
        synthesis,
        jnp.eye(3),
        nside=1,
        lmax=2,
        pixelization="HEALPix-RING",
        artifact=artifact,
    ).realize(jax.random.key(0))

    print("local_curvature_valid", bool(validity.successful))
    print("gaussian_log_probability", float(likelihood_result.log_probability))
    print("hod_total_mean", hod.total_mean)
    print("ewald_net_force", force.evidence.net_force)
    print("sky_shape", sky.iqu.shape)


if __name__ == "__main__":
    main()
