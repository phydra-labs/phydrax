#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

from phydrax.constraints import (
    ContinuousDirichletBoundaryConstraint,
    ContinuousNeumannBoundaryConstraint,
    ContinuousRobinBoundaryConstraint,
)
from phydrax.domain import Boundary, Interval1d, ProductStructure


def test_dirichlet_boundary_constraint_zero_when_satisfied():
    geom = Interval1d(0.0, 1.0)
    component = geom.component({"x": Boundary()})
    structure = ProductStructure((("x",),))

    u = geom.Function()(2.0)
    c = ContinuousDirichletBoundaryConstraint(
        "u",
        component,
        target=2.0,
        num_points=8,
        structure=structure,
    )
    loss = c.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss, 0.0)


def test_neumann_boundary_constraint_zero_when_satisfied():
    geom = Interval1d(0.0, 1.0)
    component = geom.component({"x": Boundary()})
    structure = ProductStructure((("x",),))

    u = geom.Function()(0.0)
    c = ContinuousNeumannBoundaryConstraint(
        "u",
        component,
        target=0.0,
        num_points=8,
        structure=structure,
    )
    loss = c.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss, 0.0)


def test_robin_boundary_constraint_zero_when_satisfied():
    geom = Interval1d(0.0, 1.0)
    component = geom.component({"x": Boundary()})
    structure = ProductStructure((("x",),))

    u = geom.Function()(0.0)
    c = ContinuousRobinBoundaryConstraint(
        "u",
        component,
        dirichlet_coeff=1.0,
        neumann_coeff=1.0,
        target=0.0,
        num_points=8,
        structure=structure,
    )
    loss = c.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss, 0.0)
