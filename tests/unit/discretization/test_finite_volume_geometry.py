#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _cell_grid(shape, *, periodic=None):
    periodic = (False,) * len(shape) if periodic is None else tuple(periodic)
    return phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic[axis])
            for axis, count in enumerate(shape)
        ),
        axis_names=tuple("xyz"[: len(shape)]),
    ).prepare(jnp.stack((jnp.zeros(len(shape)), jnp.ones(len(shape)))))


def test_structured_finite_volume_has_exact_cell_and_face_geometry():
    grid = _cell_grid((4, 3))
    discretization = phx.discretization.FiniteVolumePlan(
        grid,
        component_names=("density", "energy"),
    ).prepare()

    assert discretization.cell_shape == (4, 3)
    assert discretization.state_shape == (4, 3, 2)
    assert tuple(layout.shape for layout in discretization.face_layouts) == (
        (5, 3),
        (4, 4),
    )
    np.testing.assert_allclose(discretization.cell_volumes, 1.0 / 12.0)
    np.testing.assert_allclose(discretization.face_measures[0], 1.0 / 3.0)
    np.testing.assert_allclose(discretization.face_measures[1], 1.0 / 4.0)
    np.testing.assert_allclose(jnp.sum(discretization.cell_volumes), 1.0)
    assert discretization.cell_space.representation == "cell_average"
    assert all(space.representation == "flux_moment" for space in discretization.face_spaces)


def test_periodic_faces_are_unique_and_one_dimensional_measure_is_one():
    grid = _cell_grid((7,), periodic=(True,))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()

    assert discretization.face_layouts[0].shape == (7,)
    np.testing.assert_allclose(discretization.face_measures[0], jnp.ones((7,)))
    np.testing.assert_allclose(discretization.cell_volumes, jnp.full((7,), 1.0 / 7.0))


def test_interval_quadrature_weights_define_nonuniform_cell_edges():
    axis = phx.discretization.AxisDiscretization(
        nodes=jnp.asarray([0.1, 0.45, 0.85]),
        quad_weights=jnp.asarray([0.2, 0.5, 0.3]),
        basis="uniform",
        periodic=False,
        primary_entity="interval",
        bounds=jnp.asarray([0.0, 1.0]),
    )
    grid = phx.discretization.PreparedTensorGrid((axis,), axis_names=("x",))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()

    np.testing.assert_allclose(grid.structured_axes[0].point_coordinates, [0.0, 0.2, 0.7, 1.0])
    np.testing.assert_allclose(discretization.cell_volumes, [0.2, 0.5, 0.3])
    np.testing.assert_allclose(discretization.cell_centers[:, 0], [0.1, 0.45, 0.85])


def test_nonuniform_cell_axis_rejects_inconsistent_centers():
    axis = phx.discretization.AxisDiscretization(
        nodes=jnp.asarray([0.1, 0.4]),
        quad_weights=jnp.asarray([0.2, 0.8]),
        basis="uniform",
        periodic=False,
        primary_entity="interval",
        bounds=jnp.asarray([0.0, 1.0]),
    )
    with pytest.raises(ValueError, match="cell centers"):
        phx.discretization.PreparedTensorGrid((axis,), axis_names=("x",))


def test_finite_volume_rejects_point_primary_support_and_duplicate_components():
    point_grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(8),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    with pytest.raises(ValueError, match="interval-primary"):
        phx.discretization.FiniteVolumePlan(point_grid)

    with pytest.raises(ValueError, match="component_names"):
        phx.discretization.FiniteVolumePlan(
            _cell_grid((4,)), component_names=("u", "u")
        )
