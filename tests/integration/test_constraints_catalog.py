#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
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
)
from phydrax.domain import (
    Boundary,
    FixedStart,
    Interval1d,
    SampleLayout,
    TimeInterval,
)


def _assert_zero_loss(constraint, functions, *, atol=1e-5):
    key = jr.key(0)
    loss_fn = eqx.filter_jit(lambda k: constraint.loss(functions, key=k))
    value = loss_fn(key)
    assert jnp.allclose(value, 0.0, atol=atol)


def test_functional_boundary_and_initial_constraints():
    geom = Interval1d(0.0, 1.0)
    component = geom.component({"x": Boundary()})
    structure = SampleLayout((("x",),))

    @geom.Function("x")
    def u(x):
        return 0.0

    functions = {"u": u}
    constraints = [
        ContinuousDirichletBoundaryConstraint(
            "u",
            component,
            target=0.0,
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        ContinuousNeumannBoundaryConstraint(
            "u",
            component,
            target=0.0,
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        ContinuousRobinBoundaryConstraint(
            "u",
            component,
            dirichlet_coeff=1.0,
            neumann_coeff=0.0,
            target=0.0,
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
    ]

    for constraint in constraints:
        _assert_zero_loss(constraint, functions)

    domain = geom @ TimeInterval(0.0, 1.0)
    initial_component = domain.component({"t": FixedStart()})
    init_structure = SampleLayout((("x",),))

    @domain.Function("x", "t")
    def u_xt(x, t):
        return 0.0

    init_constraint = ContinuousInitialConstraint(
        "u",
        initial_component,
        func=0.0,
        sampling=phx.domain.PointSampling(8, layout=init_structure),
    )
    _assert_zero_loss(init_constraint, {"u": u_xt})


def test_cfd_constraints_continuous_and_discrete():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    component = geom.component({"x": Boundary()})
    structure = SampleLayout((("x",),))

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
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        ContinuousDirichletBoundaryConstraint(
            "p",
            component,
            target=0.0,
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        ContinuousDirichletBoundaryConstraint(
            "u",
            component,
            target=inflow_velocity,
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        ContinuousNeumannBoundaryConstraint(
            "p",
            component,
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        ContinuousSymmetryVelocityBoundaryConstraint(
            "u",
            component,
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        ContinuousDirichletBoundaryConstraint(
            "u",
            component,
            target=wall_velocity,
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        ContinuousNoPenetrationBoundaryConstraint(
            "u",
            component,
            wall_velocity=wall_velocity,
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        ContinuousSlipWallBoundaryConstraint(
            "u",
            "p",
            component,
            viscosity=1.0,
            sampling=phx.domain.PointSampling(8, layout=structure),
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
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    component = geom.component({"x": Boundary()})
    structure = SampleLayout((("x",),))

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
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        ContinuousTractionBoundaryConstraint(
            "u",
            component,
            lambda_=1.0,
            mu=1.0,
            traction=zeros_vec,
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        ContinuousNormalDisplacementBoundaryConstraint(
            "u",
            component,
            normal_displacement=0.0,
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        ContinuousElasticFoundationBoundaryConstraint(
            "u",
            component,
            lambda_=1.0,
            mu=1.0,
            stiffness=1.0,
            foundation_displacement=zeros_vec,
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        ContinuousElasticSymmetryBoundaryConstraint(
            "u",
            component,
            lambda_=1.0,
            mu=1.0,
            sampling=phx.domain.PointSampling(8, layout=structure),
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
    structure = SampleLayout((("x",),))

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
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        ContinuousConvectionBoundaryConstraint(
            "T",
            component,
            h=1.0,
            k=1.0,
            ambient_temp=0.0,
            sampling=phx.domain.PointSampling(8, layout=structure),
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


def test_thermal_constraints_use_physical_outward_flux_sign():
    geom = Interval1d(0.0, 1.0)
    component = geom.component({"x": Boundary()})
    structure = SampleLayout((("x",),))
    conductivity = 2.0
    convection = 4.0

    @geom.Function("x")
    def temperature(x):
        return x[0] ** 2

    @geom.Function("x")
    def outward_flux(x):
        return -2.0 * conductivity * x[0]

    @geom.Function("x")
    def ambient_temperature(x):
        return x[0] ** 2 + 2.0 * conductivity * x[0] / convection

    points = {"x": jnp.array([[0.0], [1.0]], dtype=float)}
    constraints = [
        ContinuousHeatFluxBoundaryConstraint(
            "T",
            component,
            k=conductivity,
            flux=outward_flux,
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        ContinuousConvectionBoundaryConstraint(
            "T",
            component,
            h=convection,
            k=conductivity,
            ambient_temp=ambient_temperature,
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        DiscreteHeatFluxBoundaryConstraint(
            "T",
            component,
            points=points,
            values=jnp.array([0.0, -2.0 * conductivity]),
            k=conductivity,
        ),
        DiscreteConvectionBoundaryConstraint(
            "T",
            component,
            points=points,
            ambient_values=jnp.array([0.0, 1.0 + 2.0 * conductivity / convection]),
            h=convection,
            k=conductivity,
        ),
    ]

    for constraint in constraints:
        _assert_zero_loss(constraint, {"T": temperature})


def test_em_constraints_continuous_and_discrete():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Cube(center=(0.0, 0.0, 0.0), side=2.0).compile()
    )
    component = geom.component({"x": Boundary()})
    structure = SampleLayout((("x",),))

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
            sampling=phx.domain.PointSampling(6, layout=structure),
        ),
        ContinuousImpedanceBoundaryConstraint(
            "H",
            "E",
            component,
            admittance=1.0,
            sampling=phx.domain.PointSampling(6, layout=structure),
        ),
        ContinuousPMCBoundaryConstraint(
            "H",
            component,
            sampling=phx.domain.PointSampling(6, layout=structure),
        ),
        ContinuousElectricSurfaceChargeBoundaryConstraint(
            "E",
            component,
            epsilon=1.0,
            surface_charge=0.0,
            sampling=phx.domain.PointSampling(6, layout=structure),
        ),
        ContinuousMagneticSurfaceCurrentBoundaryConstraint(
            "H",
            component,
            surface_current=0.0,
            sampling=phx.domain.PointSampling(6, layout=structure),
        ),
        ContinuousInterfaceTangentialEContinuityConstraint(
            "E1",
            "E2",
            component,
            sampling=phx.domain.PointSampling(6, layout=structure),
        ),
        ContinuousInterfaceNormalDJumpConstraint(
            "E1",
            "E2",
            component,
            epsilon1=1.0,
            epsilon2=1.0,
            surface_charge=0.0,
            sampling=phx.domain.PointSampling(6, layout=structure),
        ),
        ContinuousInterfaceTangentialHJumpConstraint(
            "H1",
            "H2",
            component,
            surface_current=0.0,
            sampling=phx.domain.PointSampling(6, layout=structure),
        ),
        ContinuousInterfaceNormalBContinuityConstraint(
            "H1",
            "H2",
            component,
            mu1=1.0,
            mu2=1.0,
            sampling=phx.domain.PointSampling(6, layout=structure),
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
