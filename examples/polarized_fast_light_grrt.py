# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Trace an exact ray through a fast-light plasma and integrate polarized transfer.

Stokes-I emission is evaluated inside the published MNY96 support. The active
linear-polarization and Faraday approximations are exercised but explicitly remain
reference-unqualified.
"""

from __future__ import annotations

from jax import config


config.update("jax_enable_x64", True)

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    astro = phx.applications.astrophysics
    scale = phx.RelativityScaleContract.si()
    convention = phx.metrix.RelativityConvention.canonical()
    ray_chart = phx.metrix.CoordinateChart("fast-light-cartesian", ("t", "x", "y", "z"))
    metric = phx.metrix.minkowski_metric(ray_chart)
    screen = astro.GRObserverScreenPlan(
        metric,
        jnp.asarray((0.0, -0.8, 0.0, 0.0)),
        jnp.asarray((1.0, 0.0, 0.0, 0.0)),
        jnp.asarray((0.0, 1.0, 0.0, 0.0)),
        jnp.asarray((0.0, 0.0, 0.0, 1.0)),
        jnp.asarray(((0.0, 0.0),)),
        scale=scale,
        convention=convention,
        coordinate_unit=scale.dimensional_scale.length_unit,
        affine_parameter_unit=scale.dimensional_scale.length_unit,
        temporal_direction="future",
    ).initialize()
    ray = astro.GRRayPlan.from_screen(
        metric,
        screen,
        jnp.linspace(0.0, 1.6, 7),
        scale=scale,
        convention=convention,
        coordinate_unit=scale.dimensional_scale.length_unit,
        affine_parameter_unit=scale.dimensional_scale.length_unit,
        transport_screen_basis=True,
        affine_budget_is_event=False,
        maximum_steps=128,
    ).trace()
    path = astro.PolarizedRayPath(
        ray,
        metric,
        ray_index=0,
        basis_tolerance=1.0e-10,
    )

    axis = jnp.linspace(-1.0, 1.0, 4)
    x, y, z = jnp.meshgrid(axis, axis, axis, indexing="ij")
    profile = jnp.exp(-1.5 * (x * x + y * y + z * z))
    shape = profile.shape
    fluid_velocity = jnp.broadcast_to(jnp.asarray((1.0, 0.0, 0.0, 0.0)), shape + (4,))
    magnetic_field = jnp.stack(
        (
            jnp.zeros_like(profile),
            0.8 + 0.2 * profile,
            0.1 * x,
            0.05 * jnp.ones_like(profile),
        ),
        axis=-1,
    )
    snapshot = astro.FastLightSnapshot(
        (axis, axis, axis),
        0.0,
        1.0e-18 * (1.0 + profile),
        1.0e6 * (1.0 + profile),
        4.0e10 * (1.0 + 0.2 * profile),
        fluid_velocity,
        magnetic_field,
        scale,
        convention,
        chart_id=path.chart_id,
        source_id="analytic-fast-light-plasma",
    )

    sampled = snapshot.sample(snapshot.prepare_path_sampling(path))

    magnetic_strength = jnp.sqrt(
        jnp.sum(sampled.magnetic_four_vector[..., 1:] ** 2, axis=-1)
    )
    frequency = jnp.full(sampled.electron_number_density.shape, 1.0e10)
    local = astro.ThermalSynchrotronModel(scale=scale).evaluate(
        sampled.electron_number_density,
        sampled.electron_temperature,
        magnetic_strength,
        frequency,
        jnp.full(frequency.shape, 0.5),
    )
    invariant = astro.invariant_synchrotron_coefficients(local, frequency)

    transfer = astro.PolarizedInvariantTransferPlan(
        path,
        astro.InvariantTransferUnitContract.si_affine_length(),
    )
    support = sampled.evidence.qualified & local.evidence.emission_reference_valid
    result = transfer.evaluate(
        invariant.emission,
        invariant.propagation_matrix,
        jnp.zeros((4,)),
        jnp.linspace(-0.2, 0.2, transfer.capacity),
        support=support,
    )

    if not bool(
        jnp.all(~transfer.active | support)
        & path.evidence.qualified
        & result.evidence.qualified
    ):
        raise RuntimeError("The evidence-bound numerical transfer was not qualified.")

    stokes = result.invariant_stokes
    linear_fraction = jnp.sqrt(stokes[1] ** 2 + stokes[2] ** 2) / stokes[0]
    print("sampled_segments", transfer.active_count)
    print("invariant_stokes", [float(value) for value in stokes])
    print("linear_polarization_fraction", float(linear_fraction))
    print("optical_depth", float(result.optical_depth))
    print(
        "numerical_transfer_qualified",
        bool(result.evidence.qualified),
        "basis_transport_error",
        float(result.evidence.maximum_basis_transport_error),
    )
    print(
        "microphysics_reference_support",
        "stokes_I",
        bool(jnp.all(local.evidence.emission_reference_valid)),
        "polarization",
        bool(jnp.all(local.evidence.polarization_reference_valid)),
        "faraday",
        bool(jnp.all(local.evidence.faraday_reference_valid)),
    )
    print(
        "polarized_prediction_reference_qualified",
        bool(jnp.all(local.evidence.qualified) & result.evidence.qualified),
    )


if __name__ == "__main__":
    main()
