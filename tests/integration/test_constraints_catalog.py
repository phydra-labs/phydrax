#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from phydrax.constraints import (
    ContinuousConvectionBoundaryConstraint,
    ContinuousDirichletBoundaryConstraint,
    ContinuousElasticFoundationBoundaryConstraint,
    ContinuousElasticSymmetryBoundaryConstraint,
    ContinuousElectricSurfaceChargeBoundaryConstraint,
    ContinuousHeatFluxBoundaryConstraint,
    ContinuousImpedanceBoundaryConstraint,
    ContinuousInitialConstraint,
    ContinuousInterfaceNormalBContinuityConstraint,
    ContinuousInterfaceNormalDJumpConstraint,
    ContinuousInterfaceTangentialEContinuityConstraint,
    ContinuousInterfaceTangentialHJumpConstraint,
    ContinuousMagneticSurfaceCurrentBoundaryConstraint,
    ContinuousNeumannBoundaryConstraint,
    ContinuousNoPenetrationBoundaryConstraint,
    ContinuousNormalDisplacementBoundaryConstraint,
    ContinuousPECBoundaryConstraint,
    ContinuousPMCBoundaryConstraint,
    ContinuousRobinBoundaryConstraint,
    ContinuousSlipWallBoundaryConstraint,
    ContinuousSymmetryVelocityBoundaryConstraint,
    ContinuousTractionBoundaryConstraint,
    DiscreteConvectionBoundaryConstraint,
    DiscreteDisplacementBoundaryConstraint,
    DiscreteElectricSurfaceChargeBoundaryConstraint,
    DiscreteHeatFluxBoundaryConstraint,
    DiscreteInterfaceNormalBContinuityConstraint,
    DiscreteInterfaceNormalDJumpConstraint,
    DiscreteInterfaceTangentialEContinuityConstraint,
    DiscreteInterfaceTangentialHJumpConstraint,
    DiscreteMagneticSurfaceCurrentBoundaryConstraint,
    DiscreteNoPenetrationBoundaryConstraint,
    DiscreteNormalDisplacementBoundaryConstraint,
    DiscretePECBoundaryConstraint,
    DiscretePMCBoundaryConstraint,
    DiscreteRobinBoundaryConstraint,
    DiscreteTractionBoundaryConstraint,
    DiscreteZeroNormalGradientVelocityBoundaryConstraint,
    FunctionalConstraint,
)
from phydrax.domain import (
    Boundary,
    Cube,
    FixedStart,
    Interval1d,
    ProductStructure,
    QuadratureBatch,
    Square,
    TimeInterval,
)


def _uniform_quadrature(batch: Any) -> QuadratureBatch:
    weights_by_axis = {}
    for block, axis in zip(
        batch.structure.blocks, batch.structure.axis_names, strict=True
    ):
        ref_label = block[0]
        field = batch.points[ref_label]
        leaves = jtu.tree_leaves(field, is_leaf=lambda x: isinstance(x, cx.Field))
        leaf_field = leaves[0]
        n = int(leaf_field.named_shape[axis])
        weights_by_axis[axis] = cx.Field(jnp.full((n,), 1.0 / float(n)), dims=(axis,))
    return QuadratureBatch(batch, weights_by_axis=weights_by_axis)


def _assert_zero_loss(constraint, functions, *, atol=1e-5):
    key = jr.key(0)
    kwargs = {}
    inner = constraint
    if isinstance(inner, FunctionalConstraint):
        batch = inner.sample(key=key)
        kwargs["quadrature"] = _uniform_quadrature(batch)
    loss_fn = eqx.filter_jit(lambda k: constraint.loss(functions, key=k, **kwargs))
    val = loss_fn(key)
    assert jnp.allclose(val, 0.0, atol=atol)


def test_functional_boundary_and_initial_constraints():
    geom = Interval1d(0.0, 1.0)
    component = geom.component({"x": Boundary()})
    structure = ProductStructure((("x",),))

    @geom.Function("x")
    def u(x):
        return 0.0

    functions = {"u": u}
    constraints = [
        ContinuousDirichletBoundaryConstraint(
            "u",
            component,
            target=0.0,
            num_points=8,
            structure=structure,
        ),
        ContinuousNeumannBoundaryConstraint(
            "u",
            component,
            target=0.0,
            num_points=8,
            structure=structure,
        ),
        ContinuousRobinBoundaryConstraint(
            "u",
            component,
            dirichlet_coeff=1.0,
            neumann_coeff=0.0,
            target=0.0,
            num_points=8,
            structure=structure,
        ),
    ]

    for constraint in constraints:
        _assert_zero_loss(constraint, functions)

    domain = geom @ TimeInterval(0.0, 1.0)
    initial_component = domain.component({"t": FixedStart()})
    init_structure = ProductStructure((("x",),))

    @domain.Function("x", "t")
    def u_xt(x, t):
        return 0.0

    init_constraint = ContinuousInitialConstraint(
        "u",
        initial_component,
        func=0.0,
        num_points=8,
        structure=init_structure,
    )
    _assert_zero_loss(init_constraint, {"u": u_xt})


def test_cfd_constraints_continuous_and_discrete():
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component({"x": Boundary()})
    structure = ProductStructure((("x",),))

    @geom.Function("x")
    def u(x):
        return jnp.array([0.0, 0.0])

    @geom.Function("x")
    def p(x):
        return 0.0

    functions = {"u": u, "p": p}
    wall_velocity = jnp.array([0.0, 0.0])
    inflow_velocity = jnp.array([0.0, 0.0])

    continuous = [
        ContinuousNeumannBoundaryConstraint(
            "u",
            component,
            num_points=8,
            structure=structure,
        ),
        ContinuousDirichletBoundaryConstraint(
            "p",
            component,
            target=0.0,
            num_points=8,
            structure=structure,
        ),
        ContinuousDirichletBoundaryConstraint(
            "u",
            component,
            target=inflow_velocity,
            num_points=8,
            structure=structure,
        ),
        ContinuousNeumannBoundaryConstraint(
            "p",
            component,
            num_points=8,
            structure=structure,
        ),
        ContinuousSymmetryVelocityBoundaryConstraint(
            "u",
            component,
            num_points=8,
            structure=structure,
        ),
        ContinuousDirichletBoundaryConstraint(
            "u",
            component,
            target=wall_velocity,
            num_points=8,
            structure=structure,
        ),
        ContinuousNoPenetrationBoundaryConstraint(
            "u",
            component,
            wall_velocity=wall_velocity,
            num_points=8,
            structure=structure,
        ),
        ContinuousSlipWallBoundaryConstraint(
            "u",
            "p",
            component,
            viscosity=1.0,
            num_points=8,
            structure=structure,
        ),
    ]

    for constraint in continuous:
        _assert_zero_loss(constraint, functions)

    points = {"x": jnp.array([[-1.0, 0.0], [1.0, 0.0]], dtype=float)}
    discrete = [
        DiscreteNoPenetrationBoundaryConstraint(
            "u",
            component,
            points=points,
            wall_normal_velocity=0.0,
        ),
        DiscreteZeroNormalGradientVelocityBoundaryConstraint(
            "u",
            component,
            points=points,
        ),
    ]
    for constraint in discrete:
        _assert_zero_loss(constraint, functions)


def test_solid_constraints_continuous_and_discrete():
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component({"x": Boundary()})
    structure = ProductStructure((("x",),))

    @geom.Function("x")
    def u(x):
        return jnp.array([0.0, 0.0])

    functions = {"u": u}
    zeros_vec = jnp.array([0.0, 0.0])

    continuous = [
        ContinuousDirichletBoundaryConstraint(
            "u",
            component,
            target=zeros_vec,
            num_points=8,
            structure=structure,
        ),
        ContinuousTractionBoundaryConstraint(
            "u",
            component,
            lambda_=1.0,
            mu=1.0,
            traction=zeros_vec,
            num_points=8,
            structure=structure,
        ),
        ContinuousNormalDisplacementBoundaryConstraint(
            "u",
            component,
            normal_displacement=0.0,
            num_points=8,
            structure=structure,
        ),
        ContinuousElasticFoundationBoundaryConstraint(
            "u",
            component,
            lambda_=1.0,
            mu=1.0,
            stiffness=1.0,
            foundation_displacement=zeros_vec,
            num_points=8,
            structure=structure,
        ),
        ContinuousElasticSymmetryBoundaryConstraint(
            "u",
            component,
            lambda_=1.0,
            mu=1.0,
            num_points=8,
            structure=structure,
        ),
    ]
    for constraint in continuous:
        _assert_zero_loss(constraint, functions)

    points = {"x": jnp.array([[-1.0, 0.0], [1.0, 0.0]], dtype=float)}
    disp_values = jnp.zeros((2, 2), dtype=float)
    discrete = [
        DiscreteDisplacementBoundaryConstraint(
            "u",
            component,
            points=points,
            displacement_values=disp_values,
        ),
        DiscreteTractionBoundaryConstraint(
            "u",
            component,
            points=points,
            values=disp_values,
            lambda_=1.0,
            mu=1.0,
        ),
        DiscreteNormalDisplacementBoundaryConstraint(
            "u",
            component,
            points=points,
            values=jnp.zeros((2,), dtype=float),
        ),
    ]
    for constraint in discrete:
        _assert_zero_loss(constraint, functions)


def test_thermal_constraints_continuous_and_discrete():
    geom = Interval1d(0.0, 1.0)
    component = geom.component({"x": Boundary()})
    structure = ProductStructure((("x",),))

    @geom.Function("x")
    def temp(x):
        return 0.0

    functions = {"T": temp}

    continuous = [
        ContinuousHeatFluxBoundaryConstraint(
            "T",
            component,
            k=1.0,
            flux=0.0,
            num_points=8,
            structure=structure,
        ),
        ContinuousConvectionBoundaryConstraint(
            "T",
            component,
            h=1.0,
            k=1.0,
            ambient_temp=0.0,
            num_points=8,
            structure=structure,
        ),
    ]
    for constraint in continuous:
        _assert_zero_loss(constraint, functions)

    points = {"x": jnp.array([[0.0], [1.0]], dtype=float)}
    discrete = [
        DiscreteRobinBoundaryConstraint(
            "T",
            component,
            points=points,
            values=jnp.zeros((2,), dtype=float),
            dirichlet_coeff=1.0,
            neumann_coeff=0.0,
        ),
        DiscreteHeatFluxBoundaryConstraint(
            "T",
            component,
            points=points,
            values=jnp.zeros((2,), dtype=float),
            k=1.0,
        ),
        DiscreteConvectionBoundaryConstraint(
            "T",
            component,
            points=points,
            ambient_values=jnp.zeros((2,), dtype=float),
            h=1.0,
            k=1.0,
        ),
    ]
    for constraint in discrete:
        _assert_zero_loss(constraint, functions)


def test_em_constraints_continuous_and_discrete():
    geom = Cube(center=(0.0, 0.0, 0.0), side=2.0)
    component = geom.component({"x": Boundary()})
    structure = ProductStructure((("x",),))

    @geom.Function("x")
    def e(x):
        return jnp.array([0.0, 0.0, 0.0])

    @geom.Function("x")
    def h(x):
        return jnp.array([0.0, 0.0, 0.0])

    @geom.Function("x")
    def e1(x):
        return jnp.array([0.0, 0.0, 0.0])

    @geom.Function("x")
    def e2(x):
        return jnp.array([0.0, 0.0, 0.0])

    @geom.Function("x")
    def h1(x):
        return jnp.array([0.0, 0.0, 0.0])

    @geom.Function("x")
    def h2(x):
        return jnp.array([0.0, 0.0, 0.0])

    functions = {"E": e, "H": h, "E1": e1, "E2": e2, "H1": h1, "H2": h2}

    continuous = [
        ContinuousPECBoundaryConstraint(
            "E",
            component,
            num_points=6,
            structure=structure,
        ),
        ContinuousImpedanceBoundaryConstraint(
            "H",
            "E",
            component,
            admittance=1.0,
            num_points=6,
            structure=structure,
        ),
        ContinuousPMCBoundaryConstraint(
            "H",
            component,
            num_points=6,
            structure=structure,
        ),
        ContinuousElectricSurfaceChargeBoundaryConstraint(
            "E",
            component,
            epsilon=1.0,
            surface_charge=0.0,
            num_points=6,
            structure=structure,
        ),
        ContinuousMagneticSurfaceCurrentBoundaryConstraint(
            "H",
            component,
            surface_current=0.0,
            num_points=6,
            structure=structure,
        ),
        ContinuousInterfaceTangentialEContinuityConstraint(
            "E1",
            "E2",
            component,
            num_points=6,
            structure=structure,
        ),
        ContinuousInterfaceNormalDJumpConstraint(
            "E1",
            "E2",
            component,
            epsilon1=1.0,
            epsilon2=1.0,
            surface_charge=0.0,
            num_points=6,
            structure=structure,
        ),
        ContinuousInterfaceTangentialHJumpConstraint(
            "H1",
            "H2",
            component,
            surface_current=0.0,
            num_points=6,
            structure=structure,
        ),
        ContinuousInterfaceNormalBContinuityConstraint(
            "H1",
            "H2",
            component,
            mu1=1.0,
            mu2=1.0,
            num_points=6,
            structure=structure,
        ),
    ]
    for constraint in continuous:
        _assert_zero_loss(constraint, functions)

    points = {
        "x": jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    }
    zeros_vec = jnp.zeros((3, 3), dtype=float)
    zeros_scalar = jnp.zeros((3,), dtype=float)

    discrete = [
        DiscretePECBoundaryConstraint("E", component, points=points),
        DiscretePMCBoundaryConstraint("H", component, points=points),
        DiscreteElectricSurfaceChargeBoundaryConstraint(
            "E",
            component,
            points=points,
            surface_charge_values=zeros_scalar,
            epsilon=1.0,
        ),
        DiscreteMagneticSurfaceCurrentBoundaryConstraint(
            "H",
            component,
            points=points,
            surface_current_values=zeros_vec,
        ),
        DiscreteInterfaceTangentialEContinuityConstraint(
            "E",
            component,
            points=points,
            tangential_values=zeros_vec,
        ),
        DiscreteInterfaceNormalDJumpConstraint(
            "E",
            component,
            points=points,
            values=zeros_scalar,
            epsilon=1.0,
        ),
        DiscreteInterfaceTangentialHJumpConstraint(
            "H",
            component,
            points=points,
            Ks_values=zeros_vec,
        ),
        DiscreteInterfaceNormalBContinuityConstraint(
            "H",
            component,
            points=points,
            values=zeros_scalar,
            mu=1.0,
        ),
    ]
    for constraint in discrete:
        _assert_zero_loss(constraint, functions)
