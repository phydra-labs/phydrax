# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Evaluate a stationary Kerr Killing horizon and its SI thermodynamics."""

from __future__ import annotations

from jax import config


config.update("jax_enable_x64", True)

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    compact = phx.applications.compact_objects
    scale = phx.RelativityScaleContract.si()

    mass = 1_000.0  # geometric length in meters
    dimensionless_spin = 0.8
    parameters = compact.KerrInput(
        mass,
        dimensionless_spin * mass**2,
    )
    horizon = compact.evaluate_stationary_kerr_horizon(parameters)
    thermal = compact.evaluate_kerr_entropy_temperature(horizon, scale)
    first_law = compact.evaluate_kerr_first_law(
        parameters,
        jnp.asarray(0.25),
        jnp.asarray(-0.1),
    )

    if not bool(horizon.qualified & thermal.qualified & first_law.qualified):
        raise RuntimeError("The stationary Kerr evaluation was not qualified.")

    branch = compact.KerrBranchCode(int(horizon.branch.code)).name.lower()
    print("branch", branch)
    print("radii_m", float(horizon.inner_radius), float(horizon.outer_radius))
    print("area_m2", float(horizon.area))
    print("temperature_K", float(thermal.temperature))
    print("entropy_J_per_K", float(thermal.entropy))
    print(
        "first_law_smarr",
        bool(first_law.first_law_satisfied),
        bool(first_law.smarr_satisfied),
        "max_residual",
        float(
            jnp.maximum(
                jnp.abs(first_law.first_law_residual),
                jnp.abs(first_law.smarr_residual),
            )
        ),
    )


if __name__ == "__main__":
    main()
