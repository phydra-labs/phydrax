#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def test_cell_centered_dirichlet_neumann_and_robin_ghost_relations():
    values = jnp.asarray([1.0, 2.0, 3.0])
    dirichlet = phx.discretization.CellGhostBoundary(
        0,
        "dirichlet",
        "dirichlet",
        0.5,
    )
    neumann = phx.discretization.CellGhostBoundary(
        0,
        "neumann",
        "neumann",
        0.5,
    )
    robin = phx.discretization.CellGhostBoundary(
        0,
        "robin",
        "robin",
        0.5,
        lower_alpha=2.0,
        lower_beta=1.0,
        upper_alpha=2.0,
        upper_beta=1.0,
    )

    assert jnp.allclose(
        dirichlet.fill(values, 5.0, 7.0),
        jnp.asarray([9.0, 1.0, 2.0, 3.0, 11.0]),
    )
    assert jnp.allclose(
        neumann.fill(values, -2.0, 4.0),
        jnp.asarray([2.0, 1.0, 2.0, 3.0, 5.0]),
    )
    robin_values = robin.fill(values, 4.0, 8.0)
    assert jnp.allclose(
        0.5 * 2.0 * (values[0] + robin_values[0]) + (values[0] - robin_values[0]) / 0.5,
        4.0,
    )
    assert jnp.allclose(
        0.5 * 2.0 * (values[-1] + robin_values[-1])
        + (robin_values[-1] - values[-1]) / 0.5,
        8.0,
    )


def test_periodic_ghosts_wrap_without_physical_boundary_data():
    runtime = phx.discretization.CellGhostBoundary(
        0,
        "periodic",
        "periodic",
        1.0,
    )

    result = runtime.fill(jnp.asarray([1.0, 2.0, 3.0]), jnp.nan, jnp.nan)

    assert jnp.allclose(result, jnp.asarray([3.0, 1.0, 2.0, 3.0, 1.0]))


def test_nodal_runtime_sets_only_dirichlet_boundary_entities():
    runtime = phx.discretization.NodalBoundaryRuntime(
        0,
        "dirichlet",
        "dirichlet",
    )
    values = jnp.arange(12.0).reshape((3, 4))

    result = runtime.apply_state(
        values,
        jnp.asarray([10.0, 11.0, 12.0, 13.0]),
        -1.0,
    )

    assert jnp.allclose(result[0], jnp.asarray([10.0, 11.0, 12.0, 13.0]))
    assert jnp.allclose(result[1], values[1])
    assert jnp.allclose(result[-1], -1.0)


def test_singular_robin_ghost_relation_is_rejected():
    with pytest.raises(ValueError, match="singular"):
        phx.discretization.CellGhostBoundary(
            0,
            "robin",
            "robin",
            1.0,
            lower_alpha=2.0,
            lower_beta=1.0,
        )
