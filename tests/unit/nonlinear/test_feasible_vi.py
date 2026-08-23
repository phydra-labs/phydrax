#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax as phx


nl = phx.nonlinear


def _termination():
    return nl.NonlinearTermination(
        absolute_residual=1e-10,
        relative_residual=0.0,
        maximum_steps=30,
    )


def test_preserve_box_never_exposes_domain_restricted_operator_to_outside_state():
    def operator(state, args):
        del args
        return jnp.where(state < 0.0, jnp.nan, state - 1.0)

    problem = nl.VariationalInequalityProblem(
        operator,
        nl.Bounds(0.0, jnp.inf),
        problem_id="nonnegative-domain",
    )
    result = nl.SemismoothNewton(feasibility="preserve-box").solve(
        problem,
        jnp.asarray([-2.0]),
        termination=_termination(),
    )

    assert bool(result.successful)
    assert jnp.all(result.state >= 0.0)
    assert jnp.allclose(result.state, jnp.asarray([1.0]), atol=1e-9)
    assert bool(result.certificate.feasible)
    assert bool(result.certificate.certified)
    assert "feasibility=preserve-box" in result.provenance.notes


def test_prepared_vi_refresh_reuses_plan_and_rejects_bound_topology_change():
    problem = nl.VariationalInequalityProblem(
        lambda state, target: state - target,
        nl.Bounds(0.0, jnp.inf),
        problem_id="prepared-vi",
    )
    method = nl.SemismoothNewton(feasibility="preserve-box")
    prepared = nl.prepare_variational_inequality(
        problem,
        jnp.asarray([0.5]),
        method=method,
        termination=_termination(),
        args=jnp.asarray([1.0]),
    )
    first = nl.solve_prepared_variational_inequality(prepared)
    refreshed = nl.refresh_variational_inequality(
        prepared,
        problem,
        first.state,
        args=jnp.asarray([2.0]),
    )
    second = nl.solve_prepared_variational_inequality(refreshed)

    assert bool(first.successful)
    assert bool(second.successful)
    assert jnp.allclose(second.state, jnp.asarray([2.0]), atol=1e-8)
    assert refreshed.nonlinear.linear_plan_id == prepared.nonlinear.linear_plan_id
    assert int(refreshed.numeric_version) == 1

    changed_topology = nl.VariationalInequalityProblem(
        lambda state, target: state - target,
        nl.Bounds(-jnp.inf, jnp.inf),
        problem_id="prepared-vi",
    )
    with pytest.raises(ValueError, match="bound topology"):
        nl.refresh_variational_inequality(
            prepared,
            changed_topology,
            first.state,
            args=jnp.asarray([2.0]),
        )


def test_prepared_feasible_vi_supports_filtered_jit_solve():
    problem = nl.VariationalInequalityProblem(
        lambda state, target: state - target,
        nl.Bounds(0.0, 1.0),
        problem_id="jitted-feasible-vi",
    )
    prepared = nl.prepare_variational_inequality(
        problem,
        jnp.asarray([0.0]),
        method=nl.SemismoothNewton(feasibility="preserve-box"),
        termination=_termination(),
        args=jnp.asarray([2.0]),
    )

    @eqx.filter_jit
    def solve(current):
        return nl.solve_prepared_variational_inequality(current).state

    assert jnp.allclose(solve(prepared), jnp.asarray([1.0]))
