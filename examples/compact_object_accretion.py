# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Construct Michel inflow and a magnetically seeded Fishbone--Moncrief torus."""

from __future__ import annotations

from jax import config


config.update("jax_enable_x64", True)

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    compact = phx.applications.compact_objects
    scale = phx.RelativityScaleContract.geometric(phx.units.KILOGRAM)
    eos = phx.equations.GammaLawEOS(scale, 4.0 / 3.0, minimum_density=0.0)

    michel_plan = compact.MichelBondiAccretionPlan(
        eos,
        1.0,
        1.0,
        0.01,
        root_iterations=56,
        bracket_samples=96,
    )
    radii = jnp.asarray(
        (
            0.7 * michel_plan.critical_radius,
            michel_plan.critical_radius,
            2.0 * michel_plan.critical_radius,
        )
    )
    michel = michel_plan.evaluate(radii)
    if not bool(jnp.all(michel.qualified)):
        raise RuntimeError("The Michel--Bondi branches were not qualified.")

    torus_plan = compact.FishboneMoncriefTorusPlan(
        eos,
        1.0,
        0.5,
        6.0,
        12.0,
        0.01,
        atmosphere_density=1.0e-8,
        atmosphere_pressure=1.0e-10,
        magnetic_seed_amplitude=0.2,
        magnetic_seed_cutoff=0.2,
    )
    radius = jnp.asarray((8.0, 10.0, 12.0, 16.0, 40.0))
    torus_points = jnp.stack(
        (
            radius,
            jnp.full_like(radius, jnp.pi / 2.0),
            jnp.linspace(0.0, 0.8, radius.shape[0]),
        ),
        axis=-1,
    )
    torus = torus_plan.evaluate(torus_points)
    if not bool(jnp.all(torus.qualified)):
        raise RuntimeError("The Fishbone--Moncrief sample was not qualified.")

    michel_residual = jnp.maximum(
        jnp.max(jnp.abs(michel.continuity_residual)),
        jnp.max(jnp.abs(michel.bernoulli_residual)),
    )
    inside_residual = jnp.max(
        jnp.where(torus.inside_torus, jnp.abs(torus.equilibrium_residual), 0.0)
    )
    print("michel_critical_radius", michel_plan.critical_radius)
    print("michel_accretion_rate", michel_plan.mass_accretion_rate)
    print(
        "michel_radial_four_velocity",
        [float(value) for value in michel.radial_four_velocity],
    )
    print("michel_max_integral_residual", float(michel_residual))
    print("torus_inside_samples", int(jnp.sum(torus.inside_torus)), "of", radius.shape[0])
    print("torus_max_equilibrium_residual", float(inside_residual))
    print(
        "torus_max_magnetic_seed",
        float(jnp.max(jnp.abs(torus.vector_potential_covector[..., 2]))),
    )


if __name__ == "__main__":
    main()
