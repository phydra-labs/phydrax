#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def run():
    lon = jnp.linspace(0.0, 0.2, 7)
    lat = jnp.linspace(-0.4, 0.4, 5)
    z = jnp.linspace(-100.0, 0.0, 5)
    geometry = phx.discretization.LatitudeLongitudeHydrostaticGridPlan(
        lon,
        lat,
        z,
        jnp.full((6, 4), 100.0),
    ).prepare()
    ocean = phx.applications.ocean.HydrostaticPrimitiveEquationPlan(
        geometry,
        eos=phx.applications.ocean.NonlinearSeawaterPolynomialEOS(),
        mixing=phx.applications.ocean.HydrostaticMixingPlan(
            "ri", maximum_coefficient=1.0e-3
        ),
    ).prepare()
    state = ocean.initialize_state(
        jnp.zeros((6, 4)),
        tracers={
            "absolute_salinity": jnp.full((6, 4, 4), 35.0),
            "conservative_temperature": jnp.broadcast_to(
                jnp.linspace(4.0, 12.0, 4), (6, 4, 4)
            ),
        },
    )
    view = ocean.view(state)
    return {
        "geometry": geometry.horizontal_coordinate,
        "finite_density": bool(jnp.all(jnp.isfinite(view.density))),
        "minimum_area": float(jnp.min(geometry.cell_area)),
        "coriolis_range": (
            float(jnp.min(geometry.coriolis)),
            float(jnp.max(geometry.coriolis)),
        ),
    }


if __name__ == "__main__":
    print(run())
