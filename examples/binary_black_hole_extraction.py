# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Extract candidate-surface geometry and Psi4 from Brill--Lindquist BBH data.

The coordinate spheres are quasilocal diagnostic surfaces, not certified MOTSs.
A single time-symmetric slice provides finite-radius Weyl multipoles, not an evolved
or asymptotic binary waveform.
"""

from __future__ import annotations

from jax import config


config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

import phydrax as phx


def metric_norm(vector, metric):
    return jnp.sqrt(phx.ein.contract("...i,...ij,...j->...", vector, metric, vector))


def main() -> None:
    nr = phx.applications.numerical_relativity
    punctures = (
        nr.Puncture(0.5, (-0.6, 0.0, 0.0)),
        nr.Puncture(0.5, (0.6, 0.0, 0.0)),
    )
    initial_data = nr.BrillLindquistInitialData(punctures)
    surface_plan = nr.SphericalSurfacePlan(4, max_precompute_bytes=8 * 1024**2)

    candidate_geometry = []
    candidate_null_expansions = []
    for puncture in punctures:
        coordinate_radius = 0.5 * puncture.bare_mass
        surface = surface_plan.constant(
            coordinate_radius,
            center=puncture.position,
        )
        coordinate_geometry = surface_plan.geometry(surface)
        sampled = initial_data(coordinate_geometry.points)
        geometry = surface_plan.geometry(surface, sampled.spatial_metric)
        conformal_gradient = jax.vmap(
            jax.grad(lambda point: initial_data(point).conformal_factor)
        )(coordinate_geometry.points.reshape((-1, 3))).reshape(
            surface_plan.sample_shape + (3,)
        )
        conformal_divergence = sampled.conformal_factor ** (-2) * (
            2.0 / coordinate_radius
            + 4.0
            * phx.ein.contract(
                "...i,...i->...",
                surface_plan.unit_radial,
                conformal_gradient,
            )
            / sampled.conformal_factor
        )
        candidate_null_expansions.append(
            nr.null_expansions(
                sampled.spatial_metric,
                sampled.extrinsic_curvature,
                geometry.outward_normal,
                conformal_divergence,
            )
        )
        candidate_geometry.append(
            nr.quasilocal_horizon_geometry(
                geometry,
                sampled.extrinsic_curvature,
                geometry.phi_tangent,
            )
        )

    extraction_surface = surface_plan.constant(5.0)
    coordinate_extraction = surface_plan.geometry(extraction_surface)
    extraction_data = initial_data(coordinate_extraction.points)
    extraction_geometry = surface_plan.geometry(
        extraction_surface, extraction_data.spatial_metric
    )
    cartesian_chart = phx.metrix.CoordinateChart("bbh-slice", ("x", "y", "z"))
    spatial_metric = phx.metrix.RiemannianMetric(
        lambda point: initial_data(point).spatial_metric,
        chart=cartesian_chart,
    )
    spatial_ricci = phx.metrix.ricci_tensor(spatial_metric, extraction_geometry.points)
    covariant_derivative_k = jnp.zeros(
        surface_plan.sample_shape + (3, 3, 3),
        dtype=extraction_data.spatial_metric.dtype,
    )
    weyl = nr.vacuum_weyl_curvature(
        extraction_data.spatial_metric,
        spatial_ricci,
        extraction_data.extrinsic_curvature,
        covariant_derivative_k,
    )

    radial = extraction_geometry.outward_normal
    theta = extraction_geometry.theta_tangent
    theta = theta / metric_norm(theta, extraction_data.spatial_metric)[..., None]
    phi = extraction_geometry.phi_tangent
    phi = phi / metric_norm(phi, extraction_data.spatial_metric)[..., None]
    multipole_plan = nr.SpinWeightedMultipolePlan(
        4,
        reconstruction_tolerance=5.0e-3,
        max_precompute_bytes=8 * 1024**2,
    )
    extraction = nr.Psi4ExtractionPlan(
        multipole_plan,
        tetrad_tolerance=1.0e-8,
    ).extract(
        weyl,
        radial,
        theta,
        phi,
        spatial_metric=extraction_data.spatial_metric,
    )

    amplitudes = jnp.where(
        multipole_plan.valid_modes,
        jnp.abs(extraction.multipoles.coefficients),
        -1.0,
    )
    dominant = int(jnp.argmax(amplitudes))
    ell = int(multipole_plan.degrees.reshape(-1)[dominant])
    order = int(multipole_plan.orders.reshape(-1)[dominant])
    print(
        "candidate_areal_radii",
        [float(item.areal_radius) for item in candidate_geometry],
    )
    print(
        "candidate_quasilocal_masses",
        [float(item.christodoulou_mass) for item in candidate_geometry],
    )
    print(
        "candidate_max_outgoing_expansion",
        [float(jnp.max(jnp.abs(item.outgoing))) for item in candidate_null_expansions],
    )
    print(
        "candidate_geometry_qualified",
        [bool(item.qualified) for item in candidate_geometry],
        "mots_certified",
        False,
    )
    print("dominant_psi4_mode", (ell, order), float(amplitudes.reshape(-1)[dominant]))
    print(
        "weyl_extraction_evidence",
        "finite",
        bool(extraction.finite),
        "physical",
        bool(extraction.physically_valid),
        "multipole_converged",
        bool(extraction.multipoles.converged),
        "finite_radius_only",
        True,
    )


if __name__ == "__main__":
    main()
