#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    grid = phx.equations.SpectralFrequencyGrid(
        jnp.asarray((1.0e12, 2.0e12, 3.0e12)),
        jnp.asarray((0.5e12, 1.0e12, 0.5e12)),
    )
    axes = (jnp.asarray((200.0, 1000.0)), jnp.asarray((1.0e4, 1.0e6)))
    absorption = phx.equations.RadiationCoefficientTable(
        *axes,
        grid,
        2.0 * jnp.ones((2, 2, 3)),
        phx.equations.RadiationCoefficientRole.ABSORPTION,
        provenance="synthetic qualification coefficient",
    ).evaluate(jnp.asarray(500.0), jnp.asarray(1.0e5))
    transport = phx.equations.RadiationCoefficientTable(
        *axes,
        grid,
        3.0 * jnp.ones((2, 2, 3)),
        phx.equations.RadiationCoefficientRole.TRANSPORT,
        provenance="synthetic qualification coefficient",
    ).evaluate(jnp.asarray(500.0), jnp.asarray(1.0e5))
    means = phx.equations.radiation_means(jnp.asarray(500.0), absorption, transport, grid)
    print(
        json.dumps(
            {
                "planck_absorption": float(means.planck_absorption),
                "rosseland_transport": float(means.rosseland_transport),
                "successful": bool(means.successful),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
