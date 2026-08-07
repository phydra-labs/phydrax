#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.constraints import (
    AveragePressureBoundaryConstraint,
    CFDBoundaryFlowRateConstraint,
    CFDKineticEnergyFluxBoundaryConstraint,
    ContinuousIntegralBoundaryConstraint,
    ContinuousIntegralInitialConstraint,
    ContinuousIntegralInteriorConstraint,
    ContinuousODEConstraint,
    DiscreteODEConstraint,
    DiscreteTimeDataConstraint,
    EMBoundaryChargeConstraint,
    EMPoyntingFluxBoundaryConstraint,
    InitialODEConstraint,
    MagneticFluxZeroConstraint,
    SolidTotalReactionBoundaryConstraint,
)
from phydrax.domain import Interval1d, SampleLayout, TimeInterval
from phydrax.operators.differential import div, dt


def _jit_loss(constraint, functions):
    loss_fn = eqx.filter_jit(lambda k: constraint.loss(functions, key=k))
    return loss_fn(jr.key(0))


def test_continuous_ode_constraint_zero():
    time = TimeInterval(0.0, 1.0)

    @time.Function("t")
    def u(t):
        return t**2

    @time.Function("t")
    def target(t):
        return 2.0 * t

    def operator(f):
        return dt(f, var="t") - target

    structure = SampleLayout((("t",),))
    constraint = ContinuousODEConstraint(
        "u",
        time,
        operator,
        sampling=phx.domain.PointSampling(64, layout=structure),
    )
    loss = _jit_loss(constraint, {"u": u})
    assert loss < 1e-6


def test_discrete_ode_constraint_zero():
    time = TimeInterval(0.0, 1.0)

    @time.Function("t")
    def u(t):
        return t**2

    @time.Function("t")
    def target(t):
        return 2.0 * t

    def operator(f):
        return dt(f, var="t") - target

    times = jnp.linspace(0.0, 1.0, 8)
    constraint = DiscreteODEConstraint("u", time, operator, times=times)
    loss = _jit_loss(constraint, {"u": u})
    assert loss < 1e-6


def test_initial_ode_constraints_zero():
    time = TimeInterval(0.0, 1.0)

    @time.Function("t")
    def u(t):
        return t**2

    c0 = InitialODEConstraint("u", time, func=0.0, time_derivative_order=0)
    c1 = InitialODEConstraint("u", time, func=0.0, time_derivative_order=1)
    c2 = InitialODEConstraint(
        "u",
        time,
        func=2.0,
        time_derivative_order=2,
        time_derivative_backend="jet",
    )
    assert _jit_loss(c0, {"u": u}) < 1e-6
    assert _jit_loss(c1, {"u": u}) < 1e-6
    assert _jit_loss(c2, {"u": u}) < 1e-6


def test_discrete_time_data_constraint_zero():
    time = TimeInterval(0.0, 1.0)

    @time.Function("t")
    def u(t):
        return t**2

    times = jnp.linspace(0.0, 1.0, 6)
    values = times**2
    constraint = DiscreteTimeDataConstraint("u", time, times=times, values=values)
    loss = _jit_loss(constraint, {"u": u})
    assert loss < 1e-6


def test_integral_constraints_1d_zero_loss():
    geom = Interval1d(0.0, 1.0)
    structure = SampleLayout((("x",),))

    @geom.Function("x")
    def u(x):
        return 1.0

    @geom.Function("x")
    def v(x):
        return jnp.array([0.0])

    @geom.Function("x")
    def p(x):
        return 0.0

    @geom.Function("x")
    def D(x):
        return jnp.array([0.0])

    @geom.Function("x")
    def B(x):
        return jnp.array([0.0])

    functions = {"u": u, "v": v, "p": p, "D": D, "B": B}

    constraints = [
        ContinuousIntegralInteriorConstraint(
            "u",
            geom,
            lambda f: f,
            sampling=phx.domain.PointSampling(32, layout=structure),
            equal_to=1.0,
        ),
        ContinuousIntegralBoundaryConstraint(
            "u",
            geom,
            lambda f, n: f,
            sampling=phx.domain.PointSampling(8, layout=structure),
            equal_to=2.0,
        ),
        ContinuousIntegralInteriorConstraint(
            "v",
            geom,
            lambda f: div(f, var="x"),
            sampling=phx.domain.PointSampling(32, layout=structure),
        ),
        EMBoundaryChargeConstraint(
            "D",
            geom,
            total_free_charge=0.0,
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        MagneticFluxZeroConstraint(
            "B",
            geom,
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        CFDBoundaryFlowRateConstraint(
            "v",
            geom,
            flow_rate=0.0,
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        CFDKineticEnergyFluxBoundaryConstraint(
            "v",
            geom,
            target_total_power=0.0,
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        SolidTotalReactionBoundaryConstraint(
            "v",
            geom,
            lambda_=1.0,
            mu=1.0,
            target_reaction=jnp.array([0.0]),
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
        AveragePressureBoundaryConstraint(
            "p",
            geom,
            mean_pressure=0.0,
            sampling=phx.domain.PointSampling(8, layout=structure),
        ),
    ]

    for constraint in constraints:
        assert _jit_loss(constraint, functions) < 1e-6


def test_boundary_integral_resolves_relabeled_geometry_in_product_domain():
    space = Interval1d(0.0, 1.0).relabel("space")
    time = TimeInterval(0.0, 1.0)
    domain = space @ time
    structure = SampleLayout((("space",), ("t",)))

    @domain.Function("space", "t")
    def u(space_coordinate, time_coordinate):
        del space_coordinate, time_coordinate
        return 1.0

    constraint = ContinuousIntegralBoundaryConstraint(
        "u",
        domain,
        lambda value, normal: value,
        sampling=phx.domain.PointSampling((8, 8), layout=structure),
        equal_to=2.0,
    )

    assert _jit_loss(constraint, {"u": u}) < 1e-6


def test_integral_initial_constraint_zero():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    structure = SampleLayout((("x",),))

    @domain.Function("x", "t")
    def u(x, t):
        return 1.0

    constraint = ContinuousIntegralInitialConstraint(
        "u",
        domain,
        lambda f: f,
        sampling=phx.domain.PointSampling(32, layout=structure),
        equal_to=1.0,
    )
    loss = _jit_loss(constraint, {"u": u})
    assert loss < 1e-6


def test_poynting_flux_constraint_zero():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Cube(center=(0.0, 0.0, 0.0), side=2.0).compile()
    )
    structure = SampleLayout((("x",),))

    @geom.Function("x")
    def E(x):
        return jnp.array([0.0, 0.0, 0.0])

    @geom.Function("x")
    def H(x):
        return jnp.array([0.0, 0.0, 0.0])

    constraint = EMPoyntingFluxBoundaryConstraint(
        "E",
        "H",
        geom,
        target_total_power=0.0,
        sampling=phx.domain.PointSampling(6, layout=structure),
    )
    loss = _jit_loss(constraint, {"E": E, "H": H})
    assert loss < 1e-6
