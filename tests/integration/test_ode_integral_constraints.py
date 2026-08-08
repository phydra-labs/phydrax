#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.domain import Boundary, FixedStart, Interval1d, SampleLayout, TimeInterval
from phydrax.operators.differential import div, dt


def _jit_loss(term, functions):
    loss_fn = eqx.filter_jit(lambda k: term.loss(functions, key=k))
    return loss_fn(jr.key(0))


def _continuous_residual(condition, num_samples):
    return phx.terms.ResidualPenalty(
        condition,
        phx.integration.per_step(
            phx.integration.mean_over(condition.on),
            phx.integration.MonteCarloPlan(num_samples),
        ),
    )


def _fixed_time_source(condition, times):
    structure = SampleLayout((("t",),)).canonicalize(condition.on.domain.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    batch = phx.domain.PointBatch(
        {"t": cx.Field(jnp.asarray(times), dims=(axis_names[0],))},
        structure,
    )
    realization = phx.integration.from_samples(
        phx.integration.mean_over(condition.on),
        batch,
    )
    return phx.integration.fixed(realization)


def _moment_term(condition, plan):
    return phx.terms.MomentPenalty(
        condition,
        phx.integration.per_step(phx.integration.over(condition.on), plan),
    )


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

    condition = phx.conditions.Residual("u", time.component(), operator)
    constraint = _continuous_residual(condition, 64)
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
    condition = phx.conditions.Residual("u", time.component(), operator)
    constraint = phx.terms.ResidualPenalty(
        condition,
        _fixed_time_source(condition, times),
    )
    loss = _jit_loss(constraint, {"u": u})
    assert loss < 1e-6


def test_initial_ode_constraints_zero():
    time = TimeInterval(0.0, 1.0)

    @time.Function("t")
    def u(t):
        return t**2

    initial = time.component({"t": FixedStart()})
    plan = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule())
    source = phx.integration.per_step(phx.integration.mean_over(initial), plan)
    c0 = phx.terms.ResidualPenalty(
        phx.conditions.Initial("u", initial, target=0.0, order=0),
        source,
    )
    c1 = phx.terms.ResidualPenalty(
        phx.conditions.Initial("u", initial, target=0.0, order=1),
        source,
    )
    c2 = phx.terms.ResidualPenalty(
        phx.conditions.Initial(
            "u",
            initial,
            target=2.0,
            order=2,
            backend="jet",
        ),
        source,
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
    observed = time.Function("t")(lambda t: jnp.interp(t, times, values))
    condition = phx.conditions.Observation(
        "u",
        time.component(),
        observed,
    )
    constraint = phx.terms.ObservationPenalty(
        condition,
        _fixed_time_source(condition, times),
    )
    loss = _jit_loss(constraint, {"u": u})
    assert loss < 1e-6


def test_integral_constraints_1d_zero_loss():
    geom = Interval1d(0.0, 1.0)
    interior = geom.component()
    boundary = geom.component({"x": Boundary()})

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
    interior_plan = phx.integration.MonteCarloPlan(32)
    boundary_plan = phx.integration.MonteCarloPlan(8)

    conditions_and_plans = [
        (
            phx.conditions.Moment("u", interior, lambda f: f, target=1.0),
            interior_plan,
        ),
        (
            phx.conditions.Moment("u", boundary, lambda f: f, target=2.0),
            boundary_plan,
        ),
        (
            phx.conditions.Moment(
                "v",
                interior,
                lambda f: div(f, var="x"),
                target=0.0,
            ),
            interior_plan,
        ),
        (
            phx.conditions.conservation.BoundaryCharge("D", boundary, 0.0),
            boundary_plan,
        ),
        (
            phx.conditions.conservation.MagneticFlux("B", boundary),
            boundary_plan,
        ),
        (
            phx.conditions.conservation.FlowRate("v", boundary, 0.0),
            boundary_plan,
        ),
        (
            phx.conditions.conservation.KineticEnergyFlux("v", boundary, 0.0),
            boundary_plan,
        ),
        (
            phx.conditions.conservation.TotalReaction(
                "v",
                boundary,
                jnp.array([0.0]),
                lambda_=1.0,
                mu=1.0,
            ),
            boundary_plan,
        ),
        (
            phx.conditions.conservation.PressureIntegral("p", boundary, 0.0),
            boundary_plan,
        ),
    ]

    for condition, plan in conditions_and_plans:
        assert _jit_loss(_moment_term(condition, plan), functions) < 1e-6


def test_boundary_integral_resolves_relabeled_geometry_in_product_domain():
    space = Interval1d(0.0, 1.0).relabel("space")
    time = TimeInterval(0.0, 1.0)
    domain = space @ time
    boundary = domain.component({"space": Boundary()})

    @domain.Function("space", "t")
    def u(space_coordinate, time_coordinate):
        del space_coordinate, time_coordinate
        return 1.0

    condition = phx.conditions.Moment("u", boundary, lambda value: value, target=2.0)
    plan = phx.integration.ProductIntegrationPlan(
        {
            "space": phx.integration.MonteCarloPlan(8),
            "t": phx.integration.MonteCarloPlan(8),
        }
    )
    constraint = _moment_term(condition, plan)

    assert _jit_loss(constraint, {"u": u}) < 1e-6


def test_integral_initial_constraint_zero():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    initial = domain.component({"t": FixedStart()})

    @domain.Function("x", "t")
    def u(x, t):
        return 1.0

    condition = phx.conditions.Moment("u", initial, lambda f: f, target=1.0)
    constraint = _moment_term(condition, phx.integration.MonteCarloPlan(32))
    loss = _jit_loss(constraint, {"u": u})
    assert loss < 1e-6


def test_poynting_flux_constraint_zero():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Cube(center=(0.0, 0.0, 0.0), side=2.0).compile()
    )
    boundary = geom.component({"x": Boundary()})

    @geom.Function("x")
    def E(x):
        return jnp.array([0.0, 0.0, 0.0])

    @geom.Function("x")
    def H(x):
        return jnp.array([0.0, 0.0, 0.0])

    condition = phx.conditions.conservation.PoyntingFlux(
        "E",
        "H",
        boundary,
        0.0,
    )
    constraint = _moment_term(condition, phx.integration.MonteCarloPlan(6))
    loss = _jit_loss(constraint, {"E": E, "H": H})
    assert loss < 1e-6
