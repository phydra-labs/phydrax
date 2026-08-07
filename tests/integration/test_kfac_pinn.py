#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.constraints import enforce_dirichlet, FunctionalConstraint
from phydrax.domain import Boundary, PointSampling, SampleLayout
from phydrax.operators.differential import laplacian, partial_n
from phydrax.solver import FunctionalSolver, SingleFieldEnforcedConstraint


def _model(domain, in_size, key):
    network = phx.nn.MLP(
        in_size=in_size,
        out_size="scalar",
        hidden_sizes=(2,),
        activation=jnp.tanh,
        rwf=False,
        key=key,
    )
    labels = domain.labels
    return domain.Model(*labels)(network)


def _constraint(
    component,
    operator,
    constraint_vars,
    *,
    points,
    key,
    weight=1.0,
):
    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=constraint_vars,
        sampling=PointSampling(points, layout=SampleLayout((component.domain.labels,))),
        sampling_mode="fixed",
        fixed_batch_key=key,
        reduction="mean",
        weight=weight,
    )


def _train_and_assert_decrease(solver, *, seed, steps=1):
    initial = solver.loss(key=jr.key(seed + 100))
    trained = solver.solve(
        num_iter=steps,
        optim=phx.optim.kfac(damping=1e-2),
        seed=seed,
        jit=False,
        keep_best=False,
        log_every=0,
    )
    final = trained.loss(key=jr.key(seed + 100))
    assert jnp.isfinite(final)
    assert final < initial
    return trained


def test_kfac_trains_soft_poisson_pinn():
    domain = phx.domain.Interval1d(-1.0, 1.0)
    u = _model(domain, 1, jr.key(0))

    @domain.Function("x")
    def forcing(x):
        return (jnp.pi**2) * jnp.sin(jnp.pi * x[0])

    interior = _constraint(
        domain.component(),
        lambda field: laplacian(field, var="x") + forcing,
        "u",
        points=5,
        key=jr.key(1),
    )
    boundary_component = domain.component({"x": Boundary()})
    boundary = _constraint(
        boundary_component,
        lambda field: field,
        "u",
        points=4,
        key=jr.key(2),
        weight=5.0,
    )
    solver = FunctionalSolver(functions={"u": u}, constraints=(interior, boundary))

    _train_and_assert_decrease(solver, seed=3)


def test_kfac_trains_hard_dirichlet_poisson_pinn():
    domain = phx.domain.Interval1d(-1.0, 1.0)
    raw = _model(domain, 1, jr.key(4))
    boundary_component = domain.component({"x": Boundary()})
    hard = SingleFieldEnforcedConstraint(
        "u",
        boundary_component,
        lambda field: enforce_dirichlet(
            field,
            boundary_component,
            var="x",
            target=0.0,
        ),
    )
    interior = _constraint(
        domain.component(),
        lambda field: laplacian(field, var="x") + 1.0,
        "u",
        points=5,
        key=jr.key(5),
    )
    solver = FunctionalSolver(
        functions={"u": raw},
        constraints=interior,
        constraint_terms=(hard,),
        boundary_weight_num_reference=32,
        boundary_weight_key=jr.key(6),
    )

    trained = _train_and_assert_decrease(solver, seed=7)
    boundary_batch = boundary_component.sample(
        PointSampling(4, layout=SampleLayout((("x",),))),
        key=jr.key(8),
    )
    values = trained.ansatz_functions()["u"](boundary_batch).data
    assert jnp.allclose(values, 0.0, atol=1e-8)


def test_kfac_trains_heat_equation_pinn():
    space = phx.domain.Interval1d(0.0, 1.0)
    time = phx.domain.TimeInterval(0.0, 1.0)
    domain = space @ time
    u = _model(domain, 2, jr.key(9))

    @domain.Function("x", "t")
    def forcing(x, t):
        return (jnp.pi**2 - 1.0) * jnp.sin(jnp.pi * x[0]) * jnp.exp(-t)

    residual = _constraint(
        domain.component(),
        lambda field: (
            partial_n(field, var="t", order=1) - laplacian(field, var="x") - forcing
        ),
        "u",
        points=4,
        key=jr.key(10),
    )
    solver = FunctionalSolver(functions={"u": u}, constraints=residual)

    _train_and_assert_decrease(solver, seed=11)


def test_kfac_trains_nonlinear_burgers_pinn():
    space = phx.domain.Interval1d(0.0, 1.0)
    time = phx.domain.TimeInterval(0.0, 1.0)
    domain = space @ time
    u = _model(domain, 2, jr.key(12))

    @domain.Function("x", "t")
    def forcing(x, t):
        return 1.0 + x[0] + t

    residual = _constraint(
        domain.component(),
        lambda field: (
            partial_n(field, var="t", order=1)
            + field * partial_n(field, var="x", order=1)
            - 0.1 * partial_n(field, var="x", order=2)
            - forcing
        ),
        "u",
        points=4,
        key=jr.key(13),
    )
    solver = FunctionalSolver(functions={"u": u}, constraints=residual)

    _train_and_assert_decrease(solver, seed=14)


def test_kfac_trains_coupled_field_residuals():
    domain = phx.domain.Interval1d(-1.0, 1.0)
    u = _model(domain, 1, jr.key(15))
    v = _model(domain, 1, jr.key(16))

    @domain.Function("x")
    def first_forcing(x):
        return 2.0 + x[0]

    @domain.Function("x")
    def second_forcing(x):
        return x[0] ** 2 - x[0]

    first = _constraint(
        domain.component(),
        lambda u_field, v_field: laplacian(u_field, var="x") + v_field - first_forcing,
        ("u", "v"),
        points=4,
        key=jr.key(17),
    )
    second = _constraint(
        domain.component(),
        lambda u_field, v_field: u_field - v_field - second_forcing,
        ("u", "v"),
        points=4,
        key=jr.key(18),
    )
    solver = FunctionalSolver(
        functions={"u": u, "v": v},
        constraints=(first, second),
    )

    _train_and_assert_decrease(solver, seed=19)


def test_kfac_trains_inverse_physical_scalar_block():
    domain = phx.domain.Interval1d(0.0, 1.0)
    u = _model(domain, 1, jr.key(20))
    coefficient = domain.Parameter(0.5)

    @domain.Function("x")
    def state_target(x):
        return x[0]

    @domain.Function("x")
    def equation_target(x):
        return 2.0 * x[0]

    state_data = _constraint(
        domain.component(),
        lambda field: field - state_target,
        "u",
        points=5,
        key=jr.key(21),
    )
    equation = _constraint(
        domain.component(),
        lambda field, parameter: parameter * field - equation_target,
        ("u", "coefficient"),
        points=5,
        key=jr.key(22),
    )
    solver = FunctionalSolver(
        functions={"u": u, "coefficient": coefficient},
        constraints=(state_data, equation),
    )
    initial_coefficient = float(coefficient.func())

    trained = _train_and_assert_decrease(solver, seed=23, steps=2)
    final_coefficient = float(trained.functions["coefficient"].func())
    assert not jnp.isclose(final_coefficient, initial_coefficient)
