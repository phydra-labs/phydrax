#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

import phydrax as phx


def test_adaptive_integral_functional_returns_raw_signed_value():
    time = phx.domain.ScalarInterval(0.0, 1.0, label="t")
    density = time.Function()(-2.0)
    objective = phx.objectives.AdaptiveIntegralFunctional(
        component=time.component(),
        integrand=density,
        label="negative_energy",
    )
    solver = phx.solver.FunctionalSolver(
        functions={"density": density},
        constraints=(),
        objectives=(objective,),
    )

    assert jnp.allclose(solver.loss(), -2.0, atol=1e-12)


def test_adaptive_integral_functional_from_operator_trains_parameter():
    time = phx.domain.ScalarInterval(0.0, 1.0, label="t")
    parameter = time.Parameter(2.0)
    objective = phx.objectives.AdaptiveIntegralFunctional.from_operator(
        component=time.component(),
        operator=lambda value: (value - 1.0) ** 2,
        objective_vars="u",
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": parameter},
        constraints=(),
        objectives=(objective,),
    )

    initial_loss = solver.loss()
    trained = solver.solve(
        num_iter=2,
        optim=optax.sgd(0.1),
        keep_best=False,
        jit=True,
        log_every=0,
    )

    assert jnp.allclose(initial_loss, 1.0, atol=1e-12)
    assert trained.loss() < initial_loss
    assert jnp.allclose(trained.loss(), 0.4096, rtol=1e-8, atol=1e-10)


def test_adaptive_integral_functional_gradient_matches_analytic_value():
    time = phx.domain.ScalarInterval(0.0, 1.0, label="t")

    def loss(scale):
        @time.Function("t")
        def density(t):
            return scale * t**2

        objective = phx.objectives.AdaptiveIntegralFunctional(
            component=time.component(),
            integrand=density,
        )
        return objective.loss({"density": density})

    assert jnp.allclose(jax.grad(loss)(2.0), 1.0 / 3.0, rtol=1e-9, atol=1e-11)
    scales = jnp.asarray((1.0, 2.0, 3.0))
    batched = jax.jit(jax.vmap(loss))(scales)
    assert jnp.allclose(batched, scales / 3.0, rtol=1e-9, atol=1e-11)


def test_adaptive_integral_functional_rejects_complex_result():
    time = phx.domain.ScalarInterval(0.0, 1.0, label="t")
    density = time.Function()(1.0 + 2.0j)
    objective = phx.objectives.AdaptiveIntegralFunctional(
        component=time.component(),
        integrand=density,
    )

    with pytest.raises(TypeError, match="requires a real scalar integrand"):
        objective.loss({"density": density})


def test_adaptive_integral_functional_never_accepts_failed_quadrature():
    time = phx.domain.ScalarInterval(0.0, 1.0, label="t")

    @time.Function("t")
    def discontinuity(t):
        return jnp.where(t < 0.123, 1.0, 0.0)

    objective = phx.objectives.AdaptiveIntegralFunctional(
        component=time.component(),
        integrand=discontinuity,
        quadrature=phx.operators.AdaptiveQuadratureConfig(
            absolute_tolerance=1e-14,
            relative_tolerance=1e-14,
            max_intervals=1,
            throw=False,
        ),
    )

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="did not converge",
    ):
        jax.block_until_ready(objective.loss({"density": discontinuity}))
