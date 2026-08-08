#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.domain import Boundary, FixedStart, Interval1d, SampleLayout, TimeInterval


def _continuous_term(condition, num_samples):
    source = phx.integration.per_step(
        phx.integration.mean_over(condition.on),
        phx.integration.MonteCarloPlan(num_samples),
    )
    return phx.terms.ResidualPenalty(condition, source)


def _fixed_term(condition, points, structure):
    layout = structure.canonicalize(condition.on.domain.labels)
    axis_names = layout.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    batch = phx.domain.PointBatch(
        {"x": cx.Field(jnp.asarray(points["x"], dtype=float), dims=(axis, None))},
        layout,
    )
    realization = phx.integration.from_samples(
        phx.integration.mean_over(condition.on),
        batch,
    )
    return phx.terms.ResidualPenalty(condition, phx.integration.fixed(realization))


def _assert_zero_loss(term, functions, *, atol=1e-5):
    key = jr.key(0)
    loss_fn = eqx.filter_jit(lambda k: term.loss(functions, key=k))
    value = loss_fn(key)
    assert jnp.allclose(value, 0.0, atol=atol)


def test_functional_boundary_and_initial_constraints():
    geom = Interval1d(0.0, 1.0)
    component = geom.component({"x": Boundary()})

    @geom.Function("x")
    def u(x):
        return 0.0

    functions = {"u": u}
    conditions = [
        phx.conditions.Dirichlet("u", component, target=0.0),
        phx.conditions.Neumann("u", component, target=0.0),
        phx.conditions.Robin(
            "u",
            component,
            dirichlet_coefficient=1.0,
            neumann_coefficient=0.0,
            target=0.0,
        ),
    ]

    for condition in conditions:
        _assert_zero_loss(_continuous_term(condition, 8), functions)

    domain = geom @ TimeInterval(0.0, 1.0)
    initial_component = domain.component({"t": FixedStart()})

    @domain.Function("x", "t")
    def u_xt(x, t):
        return 0.0

    initial = phx.conditions.Initial("u", initial_component, target=0.0)
    _assert_zero_loss(_continuous_term(initial, 8), {"u": u_xt})


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

    continuous_conditions = [
        phx.conditions.Neumann("u", component),
        phx.conditions.Dirichlet("p", component, target=0.0),
        phx.conditions.Dirichlet("u", component, target=inflow_velocity),
        phx.conditions.Neumann("p", component),
        phx.conditions.cfd.SymmetryVelocity("u", component),
        phx.conditions.Dirichlet("u", component, target=wall_velocity),
        phx.conditions.cfd.NoPenetration(
            "u",
            component,
            wall_velocity=wall_velocity,
        ),
        phx.conditions.cfd.SlipWall(
            "u",
            "p",
            component,
            viscosity=1.0,
        ),
    ]

    for condition in continuous_conditions:
        _assert_zero_loss(_continuous_term(condition, 8), functions)

    points = {"x": jnp.array([[-1.0, 0.0], [1.0, 0.0]], dtype=float)}
    discrete_conditions = [
        phx.conditions.cfd.NoPenetration(
            "u",
            component,
            wall_normal_velocity=0.0,
        ),
        phx.conditions.cfd.ZeroNormalGradientVelocity("u", component),
    ]
    for condition in discrete_conditions:
        _assert_zero_loss(_fixed_term(condition, points, structure), functions)


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

    continuous_conditions = [
        phx.conditions.Dirichlet("u", component, target=zeros_vec),
        phx.conditions.solids.Traction(
            "u",
            component,
            lambda_=1.0,
            mu=1.0,
            traction=zeros_vec,
        ),
        phx.conditions.solids.NormalDisplacement("u", component, target=0.0),
        phx.conditions.solids.ElasticFoundation(
            "u",
            component,
            lambda_=1.0,
            mu=1.0,
            stiffness=1.0,
            foundation_displacement=zeros_vec,
        ),
        phx.conditions.solids.ElasticSymmetry(
            "u",
            component,
            lambda_=1.0,
            mu=1.0,
        ),
    ]
    for condition in continuous_conditions:
        _assert_zero_loss(_continuous_term(condition, 8), functions)

    points = {"x": jnp.array([[-1.0, 0.0], [1.0, 0.0]], dtype=float)}
    discrete_conditions = [
        phx.conditions.Dirichlet("u", component, target=zeros_vec),
        phx.conditions.solids.Traction(
            "u",
            component,
            lambda_=1.0,
            mu=1.0,
            traction=zeros_vec,
        ),
        phx.conditions.solids.NormalDisplacement("u", component, target=0.0),
    ]
    for condition in discrete_conditions:
        _assert_zero_loss(_fixed_term(condition, points, structure), functions)


def test_thermal_constraints_continuous_and_discrete():
    geom = Interval1d(0.0, 1.0)
    component = geom.component({"x": Boundary()})
    structure = SampleLayout((("x",),))

    @geom.Function("x")
    def temp(x):
        return 0.0

    functions = {"T": temp}

    continuous_conditions = [
        phx.conditions.thermal.HeatFlux(
            "T",
            component,
            conductivity=1.0,
            flux=0.0,
        ),
        phx.conditions.thermal.Convection(
            "T",
            component,
            heat_transfer_coefficient=1.0,
            conductivity=1.0,
            ambient_temperature=0.0,
        ),
    ]
    for condition in continuous_conditions:
        _assert_zero_loss(_continuous_term(condition, 8), functions)

    points = {"x": jnp.array([[0.0], [1.0]], dtype=float)}
    discrete_conditions = [
        phx.conditions.Robin(
            "T",
            component,
            dirichlet_coefficient=1.0,
            neumann_coefficient=0.0,
            target=0.0,
        ),
        phx.conditions.thermal.HeatFlux(
            "T",
            component,
            conductivity=1.0,
            flux=0.0,
        ),
        phx.conditions.thermal.Convection(
            "T",
            component,
            heat_transfer_coefficient=1.0,
            conductivity=1.0,
            ambient_temperature=0.0,
        ),
    ]
    for condition in discrete_conditions:
        _assert_zero_loss(_fixed_term(condition, points, structure), functions)


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
    continuous_conditions = [
        phx.conditions.thermal.HeatFlux(
            "T",
            component,
            conductivity=conductivity,
            flux=outward_flux,
        ),
        phx.conditions.thermal.Convection(
            "T",
            component,
            heat_transfer_coefficient=convection,
            conductivity=conductivity,
            ambient_temperature=ambient_temperature,
        ),
    ]
    discrete_conditions = [
        phx.conditions.thermal.HeatFlux(
            "T",
            component,
            conductivity=conductivity,
            flux=outward_flux,
        ),
        phx.conditions.thermal.Convection(
            "T",
            component,
            heat_transfer_coefficient=convection,
            conductivity=conductivity,
            ambient_temperature=ambient_temperature,
        ),
    ]

    for condition in continuous_conditions:
        _assert_zero_loss(_continuous_term(condition, 8), {"T": temperature})
    for condition in discrete_conditions:
        _assert_zero_loss(
            _fixed_term(condition, points, structure),
            {"T": temperature},
        )


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

    continuous_conditions = [
        phx.conditions.electromagnetics.PEC("E", component),
        phx.conditions.electromagnetics.Impedance(
            "H",
            "E",
            component,
            admittance=1.0,
        ),
        phx.conditions.electromagnetics.PMC("H", component),
        phx.conditions.electromagnetics.ElectricSurfaceCharge(
            "E",
            component,
            permittivity=1.0,
            surface_charge=0.0,
        ),
        phx.conditions.electromagnetics.MagneticSurfaceCurrent(
            "H",
            component,
            surface_current=0.0,
        ),
        phx.conditions.electromagnetics.InterfaceTangentialEContinuity(
            "E1",
            "E2",
            component,
        ),
        phx.conditions.electromagnetics.InterfaceNormalDJump(
            "E1",
            "E2",
            component,
            permittivity_1=1.0,
            permittivity_2=1.0,
            surface_charge=0.0,
        ),
        phx.conditions.electromagnetics.InterfaceTangentialHJump(
            "H1",
            "H2",
            component,
            surface_current=0.0,
        ),
        phx.conditions.electromagnetics.InterfaceNormalBContinuity(
            "H1",
            "H2",
            component,
            permeability_1=1.0,
            permeability_2=1.0,
        ),
    ]
    for condition in continuous_conditions:
        _assert_zero_loss(_continuous_term(condition, 6), functions)

    points = {
        "x": jnp.array(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float,
        )
    }
    discrete_conditions = [
        phx.conditions.electromagnetics.PEC("E", component),
        phx.conditions.electromagnetics.PMC("H", component),
        phx.conditions.electromagnetics.ElectricSurfaceCharge(
            "E",
            component,
            permittivity=1.0,
            surface_charge=0.0,
        ),
        phx.conditions.electromagnetics.MagneticSurfaceCurrent(
            "H",
            component,
            surface_current=jnp.zeros((3,)),
        ),
        phx.conditions.electromagnetics.InterfaceTangentialEContinuity(
            "E1",
            "E2",
            component,
        ),
        phx.conditions.electromagnetics.InterfaceNormalDJump(
            "E1",
            "E2",
            component,
            permittivity_1=1.0,
            permittivity_2=1.0,
            surface_charge=0.0,
        ),
        phx.conditions.electromagnetics.InterfaceTangentialHJump(
            "H1",
            "H2",
            component,
            surface_current=jnp.zeros((3,)),
        ),
        phx.conditions.electromagnetics.InterfaceNormalBContinuity(
            "H1",
            "H2",
            component,
            permeability_1=1.0,
            permeability_2=1.0,
        ),
    ]
    for condition in discrete_conditions:
        _assert_zero_loss(_fixed_term(condition, points, structure), functions)
