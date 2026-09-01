"""Qualification evidence for cosmology boundary-closure contracts."""

from __future__ import annotations

import json

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    cosmo = phx.applications.cosmology
    first = cosmo.CosmologyPhysicalState(
        [70.0, 0.3, 2.1e-9],
        ("hubble_constant", "matter_density", "primordial_amplitude"),
        "scale",
    )
    second = cosmo.CosmologyPhysicalState(
        [70.0, 0.3, 3.0e-9],
        first.names,
        "scale",
    )
    geometry = cosmo.PhysicalDependencyProjection(("hubble_constant", "matter_density"))
    geometry_match = geometry.project(first).require_compatible(
        geometry.project(second), jnp.asarray(1.0)
    )

    curvature = cosmo.LocalCurvatureValidityPlan(
        light_speed=1.0,
        geometry_error_budget=1.0e-3,
        support_kind="qualification",
    ).evaluate(cosmo.FLRWBackground(1.0, 0.3, curvature_density=0.01), 0.1)

    ewald = cosmo.PeriodicEwaldForcePlan(
        (1.0, 1.0, 1.0),
        1.0,
        softening=0.02,
        alpha=5.0,
        real_shells=2,
        reciprocal_modes=4,
    ).evaluate(
        jnp.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]),
        jnp.ones((2,)),
    )

    source = cosmo.CoordinateLayout(("P0", "P2", "P4"))
    target = cosmo.CoordinateLayout(("d0", "d1"))
    likelihood = cosmo.CorrelatedGaussianPlan(
        [2.0, 1.0],
        cosmo.LinearObservationPlan([[1.0, 0.5, 0.0], [0.0, 0.25, 1.0]], source, target),
        cosmo.PrecisionCovarianceAction(jnp.eye(2), 0.0, target),
    ).evaluate(cosmo.TheoryVector([1.0, 2.0, 0.5], source, "qualification"))

    report = {
        "geometry_projection_match": float(geometry_match),
        "physical_state_ids_differ": first.content_id() != second.content_id(),
        "curvature_support_ratio": float(curvature.support_ratio),
        "curvature_within_budget": bool(curvature.successful),
        "ewald_net_force_defect": float(jnp.max(jnp.abs(ewald.evidence.net_force))),
        "ewald_finite": bool(ewald.successful),
        "gaussian_residual": float(jnp.max(jnp.abs(likelihood.residual))),
        "gaussian_successful": bool(likelihood.successful),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
