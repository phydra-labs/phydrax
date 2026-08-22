#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def test_bounded_cell_axis_has_distinct_cell_and_face_entities():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    cells = grid.cells()
    faces = grid.faces("x")
    pressure = grid.field_space("pressure", entity_layout=cells)
    velocity = grid.field_space("velocity", entity_layout=faces)

    assert grid.shape == (4,)
    assert cells.shape == (4,)
    assert faces.shape == (5,)
    assert jnp.allclose(
        cells.coordinates_by_axis[0], jnp.asarray([0.125, 0.375, 0.625, 0.875])
    )
    assert jnp.allclose(faces.coordinates_by_axis[0], jnp.linspace(0.0, 1.0, 5))
    assert jnp.allclose(jnp.sum(cells.measure), 1.0)
    assert jnp.allclose(jnp.sum(faces.measure), 1.0)
    assert pressure.vector_space.shape == (4,)
    assert velocity.vector_space.shape == (5,)
    assert pressure.vector_space.space_id != velocity.vector_space.space_id
    assert jnp.count_nonzero(faces.lower_boundary_masks[0]) == 1
    assert jnp.count_nonzero(faces.upper_boundary_masks[0]) == 1


def test_periodic_cell_axis_quotients_point_entities_to_cell_count():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))

    assert grid.cells().shape == (8,)
    assert grid.faces("x").shape == (8,)
    assert not jnp.any(grid.faces("x").lower_boundary_masks[0])
    assert not jnp.any(grid.faces("x").upper_boundary_masks[0])


def test_two_dimensional_face_layouts_have_axis_specific_shapes():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(3),
            phx.discretization.UniformCellAxisSpec(5),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, -1.0], [1.0, 1.0]]))

    assert grid.cells().shape == (3, 5)
    assert grid.faces("x").shape == (4, 5)
    assert grid.faces("y").shape == (3, 6)
    assert grid.vertices().shape == (4, 6)
    assert jnp.allclose(jnp.sum(grid.cells().measure), 2.0)


def test_unresolved_grid_location_cannot_manufacture_a_field_space():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    unresolved = phx.discretization.GridLocation(("x",), ((1, 3),))

    with pytest.raises(ValueError, match="does not resolve"):
        grid.field_space("invalid", location=unresolved)
