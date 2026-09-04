#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import pytest

from phydrax.discretization import FourierAxisSpec, TensorGridPlan, UniformAxisSpec
from phydrax.geometry import RigidFrame
from phydrax.optics.wave import (
    ideal_square_law,
    integrate_intensity,
    IntensityPlane,
    PlaneFieldSpace,
    ScalarPlaneField,
    TangentialPlaneField,
)


def _finite_space(shape=(5, 7)):
    grid = TensorGridPlan(
        tuple(UniformAxisSpec(size) for size in shape),
        axis_names=("u", "v"),
    ).prepare(jnp.asarray([[-1.0, -2.0], [1.0, 2.0]]))
    return PlaneFieldSpace(grid, RigidFrame.identity(3), "finite-window")


def test_plane_space_validates_dimension_topology_and_field_shapes():
    one_dimensional = TensorGridPlan((UniformAxisSpec(5),), axis_names=("u",)).prepare(
        jnp.asarray([[-1.0], [1.0]])
    )
    with pytest.raises(ValueError, match="exactly two-dimensional"):
        PlaneFieldSpace(
            one_dimensional,
            RigidFrame.identity(3),
            "finite-window",
        )

    periodic_grid = TensorGridPlan(
        (FourierAxisSpec(4), FourierAxisSpec(6)),
        axis_names=("u", "v"),
    ).prepare(jnp.asarray([[-1.0, -1.0], [1.0, 1.0]]))
    with pytest.raises(ValueError, match="requires grid-axis periodicity"):
        PlaneFieldSpace(periodic_grid, RigidFrame.identity(3), "finite-window")

    space = _finite_space()
    with pytest.raises(ValueError, match="must have shape"):
        ScalarPlaneField(space, jnp.ones((5, 6)), 2.0, 0.0)
    with pytest.raises(ValueError, match="must have shape"):
        TangentialPlaneField(space, jnp.ones((5, 7)), 2.0, 0.0)
    with pytest.raises(ValueError, match="must have shape"):
        IntensityPlane(space, jnp.ones((5, 7, 1)), 2.0, 0.0)


def test_plane_space_composes_grid_coordinates_and_rigid_frame():
    grid = TensorGridPlan(
        (UniformAxisSpec(3), UniformAxisSpec(4)),
        axis_names=("u", "v"),
    ).prepare(jnp.asarray([[0.0, -1.0], [2.0, 2.0]]))
    frame = RigidFrame(jnp.eye(3), jnp.asarray([1.0, 2.0, 3.0]))
    space = PlaneFieldSpace(grid, frame, "finite-window")

    assert space.coordinate_axes[0] is grid.primary_entity_layout.coordinates_by_axis[0]
    assert space.coordinate_axes[1] is grid.primary_entity_layout.coordinates_by_axis[1]
    assert space.transverse_coordinates.shape == (3, 4, 2)
    assert jnp.allclose(
        space.world_points[..., :2],
        space.transverse_coordinates + jnp.asarray([1.0, 2.0]),
    )
    assert jnp.all(space.world_points[..., 2] == 3.0)


def test_square_law_scalar_tangential_parity_and_physical_integration():
    space = _finite_space()
    scalar = ScalarPlaneField(space, (1.0 + 2.0j) * jnp.ones(space.shape), 7.0, 0.25)
    tangential = TangentialPlaneField(
        space,
        jnp.stack((scalar.values, jnp.zeros_like(scalar.values)), axis=-1),
        7.0,
        0.25,
    )

    scalar_intensity = ideal_square_law(scalar)
    tangential_intensity = ideal_square_law(tangential)
    assert jnp.allclose(scalar_intensity.values, 5.0)
    assert jnp.allclose(tangential_intensity.values, scalar_intensity.values)
    assert scalar_intensity.angular_frequency == 7.0
    assert scalar_intensity.longitudinal_coordinate == 0.25

    # The finite grid uses tensor-product trapezoid weights over a 2-by-4 window.
    assert jnp.allclose(integrate_intensity(scalar_intensity), 40.0)
