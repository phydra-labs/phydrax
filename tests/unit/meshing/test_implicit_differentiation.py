#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _fixed_lattice_plan() -> tuple[Any, Any]:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0),
        0.75,
        feature_id="fixed-lattice-implicit-sphere",
    ).compile()
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(9) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[-1.4, -1.4, -1.4], [1.4, 1.4, 1.4]]))
    plan = phx.geometry.discover_implicit_surface(
        geometry,
        grid,
        policy=phx.geometry.ImplicitSurfacePolicy(
            projection=phx.geometry.ImplicitProjectionPolicy(trust_fraction=0.45),
        ),
        source_id="fixed-lattice-implicit-sphere",
    )
    return geometry, plan


def test_fixed_lattice_implicit_surface_volume_observable_differentiates() -> None:
    geometry, plan = _fixed_lattice_plan()
    radius_index = geometry.schema.index(
        phx.geometry.ParameterId("fixed-lattice-implicit-sphere", "radius")
    )

    def enclosed_volume(radius: Any) -> Any:
        state = geometry.state.replace_at(radius_index, radius)
        realization = plan.realize(state)
        corners = realization.vertices[realization.faces]
        return jnp.sum(
            jnp.sum(
                corners[:, 0] * jnp.cross(corners[:, 1], corners[:, 2]), axis=-1
            )
        ) / 6.0

    radius = jnp.asarray(0.75)
    step = 1.0e-4
    derivative = jax.grad(enclosed_volume)(radius)
    finite_difference = (
        enclosed_volume(radius + step) - enclosed_volume(radius - step)
    ) / (2.0 * step)

    assert jnp.isfinite(derivative)
    np.testing.assert_allclose(derivative, finite_difference, rtol=1.0e-5)
