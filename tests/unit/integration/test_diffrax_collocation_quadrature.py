import jax.numpy as jnp

import phydrax as phx
from phydrax.domain import Interval1d


def test_diffrax_collocation_plan_uses_fixed_capacity_lifecycle_and_identity():
    domain = Interval1d(0.0, 1.0)
    component = domain.component({"x": phx.domain.Interior()})
    target = phx.integration.over(component)
    plan = phx.integration.DiffraxCollocationQuadraturePlan(
        jnp.asarray([0.25, 0.75, jnp.nan]),
        jnp.asarray([0.5, 0.5, jnp.nan]),
        active=jnp.asarray([True, True, False]),
        solver_id="solver:diffrax:Tsit5:test",
        max_collocation=3,
        throw=False,
    )

    @domain.Function("x")
    def integrand(x):
        return x[0]

    realization = phx.integration.materialize(target, plan)
    estimate = phx.integration.reduce(integrand, realization)
    assert jnp.allclose(estimate.value.data, 0.5)
    assert estimate.diagnostics.solver_id == "solver:diffrax:Tsit5:test"
    assert int(estimate.diagnostics.active_collocation) == 2


def test_diffrax_collocation_solver_failure_is_typed_not_converged():
    plan = phx.integration.DiffraxCollocationQuadraturePlan(
        jnp.asarray([0.5]),
        jnp.asarray([1.0]),
        solver_id="solver:diffrax:failed",
        solver_successful=False,
        throw=False,
    )
    assert not bool(plan.solver_successful)
