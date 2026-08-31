#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def run():
    state = jnp.asarray((1.0, 0.0, 0.0, 2.5))
    normal = jnp.asarray((1.0, 0.0))
    exterior = phx.equations.fem.DGSEMCharacteristicBoundaryPlan(
        "slip-wall"
    ).exterior_state(state, state, normal)
    sensor = phx.equations.fem.TroubledCellEvidence(
        jnp.asarray((0.1, 2.0)),
        jnp.asarray((0.0, 0.5)),
        jnp.asarray((0.0, 0.2)),
    )
    coefficients = jnp.asarray(((1.0, 0.2, 0.01), (1.0, 0.5, 0.4)))
    limited = phx.equations.fem.ConservativeModalLimiter().apply(
        coefficients, sensor.troubled
    )
    return {
        "wall_mass_flux": float(exterior[1]),
        "troubled_cells": int(jnp.count_nonzero(sensor.troubled)),
        "mean_preserved": bool(jnp.allclose(limited[:, 0], coefficients[:, 0])),
    }


if __name__ == "__main__":
    print(run())
