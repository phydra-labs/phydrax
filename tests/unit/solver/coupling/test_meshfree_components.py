# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Meshfree residual, constraint, reconstruction and physical capacity behavior."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import (
    PointCloudPlan,
    prepare_point_cloud_field_reconstruction,
)
from phydrax.discretization._point_cloud_view import PointCloudReconstruction
from phydrax.geometry import Rectangle
from phydrax.linalg import ArraySpace, ConstraintMap, FunctionLinearOperator
from phydrax.solver.coupling._meshfree_components import MeshfreeComponent
from phydrax.sparse import SparseCoordinateOperator


def prepared_cloud_component(
    *,
    name: str = "bulk",
    reconstruction: PointCloudReconstruction = "polynomial",
    constrained: bool = False,
) -> MeshfreeComponent:
    axis = np.linspace(0.0, 1.0, 7)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.ravel(), y.ravel()), axis=-1)
    measures = np.linspace(0.5, 1.5, points.shape[0]) / points.shape[0]
    cloud = PointCloudPlan(points, measures, neighbors=20).prepare()
    view = prepare_point_cloud_field_reconstruction(
        cloud,
        support_geometry=Rectangle((0.5, 0.5), (1.0, 1.0)).compile(),
        radius=0.65,
        capacity=49,
        reconstruction=reconstruction,
    )
    full = cloud.field_spaces[0].vector_space
    weights = sum(pair[1] for pair in cloud.derivative_weights)
    native = SparseCoordinateOperator(
        cloud.relation,
        -jnp.asarray(measures)[:, None] * weights,
        source=full,
        target=full,
    )
    offset = np.zeros(points.shape[0])
    constraint = None
    free = None
    if constrained:
        free = jnp.asarray(np.flatnonzero(points[:, 0] > 0.0), dtype=jnp.int32)
        reduced = ArraySpace((free.size,))
        prolongation = FunctionLinearOperator(
            lambda value: jnp.zeros((points.shape[0],)).at[free].set(value),
            source=reduced,
            target=full,
            transpose_action=lambda value: value[free],
        )
        constraint = ConstraintMap(full, reduced, prolongation)
        offset[points[:, 0] == 0] = points[points[:, 0] == 0, 1] ** 2
    return MeshfreeComponent(
        cloud,
        native,
        view,
        measures,
        name=name,
        owner_id=cloud.prepared_id,
        constraint=constraint,
        lift=offset,
        free_rows=free,
        nullspace=None if constrained else np.ones((points.shape[0], 1)),
    )


def test_native_residual_and_capacity_retain_physical_measures() -> None:
    component = prepared_cloud_component()
    points = np.asarray(component.owner.points)
    values = jnp.asarray(np.sum(points**2, axis=-1))
    rows = component.residual((values,), None)[0]
    np.testing.assert_allclose(rows, -4 * component.mass_diagonal, atol=2e-11)
    mass = component.prepare_capacity("concentration", coefficient=2.5).operator(None)
    np.testing.assert_allclose(
        mass.mv(values), 2.5 * component.mass_diagonal * values, atol=1e-14
    )
    query = component.prepare_field_reconstruction("concentration").prepare_query(
        np.asarray([[0.25, 0.45], [0.72, 0.64]])
    )
    np.testing.assert_allclose(
        query.apply(values), [0.25**2 + 0.45**2, 0.72**2 + 0.64**2], atol=2e-11
    )


def test_dirichlet_lift_and_row_pullback_are_applied_once() -> None:
    component = prepared_cloud_component(constrained=True)
    points = np.asarray(component.owner.points)
    field = component.fields[0]
    free = field.free_rows
    assert free is not None
    values = jnp.asarray(np.sum(points**2, axis=-1))
    state = (values[free],)
    np.testing.assert_allclose(
        component.expand("concentration", state, None), values, atol=1e-14
    )
    np.testing.assert_allclose(
        component.residual(state, None)[0],
        (-4 * component.mass_diagonal)[free],
        atol=2e-11,
    )
    operator = component.linear_operator(None)
    direction = jnp.sin(jnp.arange(free.size, dtype=jnp.float64))
    np.testing.assert_allclose(
        operator.mv((direction,))[0],
        component.residual((state[0] + direction,), None)[0]
        - component.residual(state, None)[0],
        atol=2e-11,
    )
    dual = jnp.cos(jnp.arange(free.size, dtype=jnp.float64))
    np.testing.assert_allclose(
        jnp.vdot(operator.mv((direction,))[0], dual),
        jnp.vdot(direction, operator.transpose_mv((dual,))[0]),
        atol=2e-11,
    )


def test_capacity_and_false_kernel_declarations_refuse() -> None:
    component = prepared_cloud_component()
    with pytest.raises(ValueError, match="positive"):
        component.prepare_capacity("concentration", coefficient=-1.0)
    with pytest.raises(ValueError, match="kernel"):
        MeshfreeComponent(
            component.owner,
            component.native_operator,
            component.reconstruction,
            component.mass_diagonal,
            name="wrong",
            owner_id=component.owner_id,
            nullspace=np.asarray(component.owner.points)[:, :1] ** 2,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        MeshfreeComponent(
            component.owner,
            component.native_operator,
            component.reconstruction,
            np.zeros(component.mass_diagonal.shape),
            name="wrong",
            owner_id=component.owner_id,
        )
