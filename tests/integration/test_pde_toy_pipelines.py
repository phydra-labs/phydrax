#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.discretization import FourierAxisSpec
from phydrax.domain import (
    Boundary,
    FixedStart,
    Interval1d,
    TimeInterval,
)
from phydrax.enforcement import enforce_dirichlet, EnforcementSpec, InteriorAnchors
from phydrax.operators.differential import bilaplacian, dt, laplacian
from phydrax.solver import FunctionalSolver


def test_pde_toy_steady_pipeline_zero_loss():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return 1.0

    left = geom.component({"x": Boundary()}, where={"x": lambda p: p[0] < 0.5})
    right = geom.component({"x": Boundary()}, where={"x": lambda p: p[0] >= 0.5})
    full_boundary = geom.component({"x": Boundary()})

    left_constraint = EnforcementSpec(
        phx.conditions.Dirichlet("u", left, target=1.0),
        kind="custom",
        transform=lambda f, _: enforce_dirichlet(f, full_boundary, var="x", target=1.0),
    )
    right_constraint = EnforcementSpec(
        phx.conditions.Dirichlet("u", right, target=1.0),
        kind="custom",
        transform=lambda f, _: enforce_dirichlet(f, full_boundary, var="x", target=1.0),
    )

    anchors = {"x": jnp.array([[0.25], [0.75]], dtype=float)}
    values = jnp.array([1.0, 1.0], dtype=float)
    interior = InteriorAnchors("u", points=anchors, values=values)

    pde_condition = phx.conditions.Residual(
        "u",
        geom.component(),
        lambda f: laplacian(f, var="x"),
    )
    pde_term = phx.terms.ResidualPenalty(
        pde_condition,
        phx.integration.per_step(
            phx.integration.mean_over(pde_condition.on),
            phx.integration.MonteCarloPlan(64),
        ),
    )

    functions = {"u": u}
    program = phx.enforcement.compile(
        functions,
        [left_constraint, right_constraint],
        interior=[interior],
        options=phx.enforcement.EnforcementOptions(num_reference=256),
    )
    solver = FunctionalSolver(
        functions=functions,
        terms=[pde_term],
        enforcement=program,
    )
    loss = solver.loss(key=jr.key(0))
    assert loss < 1e-6


def test_pde_toy_steady_pipeline_zero_loss_jet_backend():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return 1.0

    # Jet cannot be mixed with enforced boundary constraints (Boundary() enforced constraints /
    # InteriorAnchors) because the enforced pipeline traces through the MLS/BVH weight
    # computation, which uses primitives not supported by jax.experimental.jet
    # (e.g. lax.cond, softplus/logaddexp custom_jvp, and clip/min/max rules).
    pde_condition = phx.conditions.Residual(
        "u",
        geom.component(),
        lambda f: laplacian(f, var="x", backend="jet"),
    )
    pde_term = phx.terms.ResidualPenalty(
        pde_condition,
        phx.integration.per_step(
            phx.integration.mean_over(pde_condition.on),
            phx.integration.MonteCarloPlan(64),
        ),
    )

    solver = FunctionalSolver(functions={"u": u}, terms=[pde_term])
    loss = solver.loss(key=jr.key(0))
    assert loss < 1e-6


def test_pde_toy_steady_pipeline_zero_loss_basis_backend_coord_separable():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return 1.0

    left = geom.component({"x": Boundary()}, where={"x": lambda p: p[0] < 0.5})
    right = geom.component({"x": Boundary()}, where={"x": lambda p: p[0] >= 0.5})
    full_boundary = geom.component({"x": Boundary()})

    left_constraint = EnforcementSpec(
        phx.conditions.Dirichlet("u", left, target=1.0),
        kind="custom",
        transform=lambda f, _: enforce_dirichlet(f, full_boundary, var="x", target=1.0),
    )
    right_constraint = EnforcementSpec(
        phx.conditions.Dirichlet("u", right, target=1.0),
        kind="custom",
        transform=lambda f, _: enforce_dirichlet(f, full_boundary, var="x", target=1.0),
    )

    anchors = {"x": jnp.array([[0.25], [0.75]], dtype=float)}
    values = jnp.array([1.0, 1.0], dtype=float)
    interior = InteriorAnchors("u", points=anchors, values=values)

    pde_condition = phx.conditions.Residual(
        "u",
        geom.component(),
        lambda f: laplacian(
            f,
            var="x",
            backend="basis",
            basis="fourier",
            periodic=True,
        ),
    )
    batch = pde_condition.on.sample(phx.domain.GridSampling({"x": FourierAxisSpec(64)}))
    pde_term = phx.terms.ResidualPenalty(
        pde_condition,
        phx.integration.fixed(
            phx.integration.from_samples(
                phx.integration.mean_over(pde_condition.on),
                batch,
            )
        ),
    )

    functions = {"u": u}
    program = phx.enforcement.compile(
        functions,
        [left_constraint, right_constraint],
        interior=[interior],
        options=phx.enforcement.EnforcementOptions(num_reference=256),
    )
    solver = FunctionalSolver(
        functions=functions,
        terms=[pde_term],
        enforcement=program,
    )
    loss = solver.loss(key=jr.key(0))
    assert loss < 1e-6


def test_pde_toy_steady_pipeline_zero_loss_bilaplacian_jet_backend():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return 1.0

    # Jet cannot be mixed with enforced boundary constraints (Boundary() enforced constraints /
    # InteriorAnchors) because the enforced pipeline traces through the MLS/BVH weight
    # computation, which uses primitives not supported by jax.experimental.jet.
    pde_condition = phx.conditions.Residual(
        "u",
        geom.component(),
        lambda f: bilaplacian(f, var="x", backend="jet"),
    )
    pde_term = phx.terms.ResidualPenalty(
        pde_condition,
        phx.integration.per_step(
            phx.integration.mean_over(pde_condition.on),
            phx.integration.MonteCarloPlan(64),
        ),
    )

    solver = FunctionalSolver(functions={"u": u}, terms=[pde_term])
    loss = solver.loss(key=jr.key(0))
    assert loss < 1e-6


def test_pde_toy_transient_pipeline_zero_loss():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return 1.0

    left = domain.component({"x": Boundary()}, where={"x": lambda p: p[0] < 0.5})
    right = domain.component({"x": Boundary()}, where={"x": lambda p: p[0] >= 0.5})
    initial = domain.component({"t": FixedStart()})
    full_boundary = domain.component({"x": Boundary()})

    specs = [
        EnforcementSpec(
            phx.conditions.Dirichlet("u", left, target=1.0),
            kind="custom",
            transform=lambda f, _: enforce_dirichlet(
                f, full_boundary, var="x", target=1.0
            ),
        ),
        EnforcementSpec(
            phx.conditions.Dirichlet("u", right, target=1.0),
            kind="custom",
            transform=lambda f, _: enforce_dirichlet(
                f, full_boundary, var="x", target=1.0
            ),
        ),
        EnforcementSpec(phx.conditions.Initial("u", initial, target=1.0, order=0)),
        EnforcementSpec(phx.conditions.Initial("u", initial, target=0.0, order=1)),
        EnforcementSpec(phx.conditions.Initial("u", initial, target=0.0, order=2)),
    ]

    anchors = {
        "x": jnp.array([[0.25], [0.75]], dtype=float),
        "t": jnp.array([0.4, 0.6], dtype=float),
    }
    values = jnp.array([1.0, 1.0], dtype=float)
    interior = InteriorAnchors("u", points=anchors, values=values)

    pde_time_condition = phx.conditions.Residual(
        "u",
        domain.component(),
        lambda f: dt(f, var="t"),
    )
    pde_space_condition = phx.conditions.Residual(
        "u",
        domain.component(),
        lambda f: laplacian(f, var="x"),
    )
    plan = phx.integration.MonteCarloPlan(64)
    pde_time = phx.terms.ResidualPenalty(
        pde_time_condition,
        phx.integration.per_step(
            phx.integration.mean_over(pde_time_condition.on),
            plan,
        ),
    )
    pde_space = phx.terms.ResidualPenalty(
        pde_space_condition,
        phx.integration.per_step(
            phx.integration.mean_over(pde_space_condition.on),
            plan,
        ),
    )

    functions = {"u": u}
    program = phx.enforcement.compile(
        functions,
        specs,
        interior=[interior],
        options=phx.enforcement.EnforcementOptions(num_reference=256),
    )
    solver = FunctionalSolver(
        functions=functions,
        terms=[pde_time, pde_space],
        enforcement=program,
    )
    loss = solver.loss(key=jr.key(0))
    assert loss < 1e-6


def test_pde_toy_transient_pipeline_zero_loss_jet_backend():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return 1.0

    # Jet cannot be mixed with enforced boundary constraints (Boundary() enforced constraints /
    # InteriorAnchors) because the enforced pipeline traces through the MLS/BVH weight
    # computation, which uses primitives not supported by jax.experimental.jet.
    pde_time_condition = phx.conditions.Residual(
        "u",
        domain.component(),
        lambda f: dt(f, var="t"),
    )
    pde_space_condition = phx.conditions.Residual(
        "u",
        domain.component(),
        lambda f: laplacian(f, var="x", backend="jet"),
    )
    plan = phx.integration.MonteCarloPlan(64)
    pde_time = phx.terms.ResidualPenalty(
        pde_time_condition,
        phx.integration.per_step(
            phx.integration.mean_over(pde_time_condition.on),
            plan,
        ),
    )
    pde_space = phx.terms.ResidualPenalty(
        pde_space_condition,
        phx.integration.per_step(
            phx.integration.mean_over(pde_space_condition.on),
            plan,
        ),
    )

    solver = FunctionalSolver(
        functions={"u": u},
        terms=[pde_time, pde_space],
    )
    loss = solver.loss(key=jr.key(0))
    assert loss < 1e-6


def test_pde_toy_transient_enforced_initial_targets_dt2_zero_jet_backend():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return 1.0 + t**2

    initial = domain.component({"t": FixedStart()})
    specs = [
        EnforcementSpec(phx.conditions.Initial("u", initial, target=1.0, order=0)),
        EnforcementSpec(phx.conditions.Initial("u", initial, target=0.0, order=1)),
        EnforcementSpec(phx.conditions.Initial("u", initial, target=0.0, order=2)),
    ]

    initial_condition = phx.conditions.Initial(
        "u",
        initial,
        target=0.0,
        order=2,
        backend="jet",
    )
    initial_term = phx.terms.ResidualPenalty(
        initial_condition,
        phx.integration.per_step(
            phx.integration.mean_over(initial_condition.on),
            phx.integration.MonteCarloPlan(64),
        ),
    )

    functions = {"u": u}
    program = phx.enforcement.compile(functions, specs)
    solver = FunctionalSolver(
        functions=functions,
        terms=[initial_term],
        enforcement=program,
    )
    loss = solver.loss(key=jr.key(0))
    assert loss < 1e-6
