#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg
nl = phx.nonlinear


def _termination(*, maximum_steps=12):
    return nl.NonlinearTermination(
        absolute_residual=1e-7,
        relative_residual=1e-7,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=maximum_steps,
    )


@pytest.mark.parametrize("method", (nl.NewtonKrylov(), nl.NewtonTrustRegion()))
def test_prepared_newton_refresh_reuses_linear_plan_and_updates_numerics(method):
    problem = nl.NonlinearSystemProblem(
        lambda state, target: state - target,
        problem_id=f"prepared-{method.method_id}",
    )
    prepared = nl.prepare_nonlinear(
        problem,
        jnp.asarray([0.0]),
        method=method,
        termination=_termination(),
        args=jnp.asarray([2.0]),
    )

    first = nl.solve_prepared_nonlinear(prepared)
    refreshed = nl.refresh_nonlinear(
        prepared,
        problem,
        first.state,
        args=jnp.asarray([3.0]),
    )
    second = nl.solve_prepared_nonlinear(refreshed)

    assert isinstance(first, nl.NonlinearResult)
    assert isinstance(second, nl.NonlinearResult)
    assert bool(first.successful)
    assert bool(second.successful)
    assert jnp.allclose(first.state, jnp.asarray([2.0]), atol=1e-6)
    assert jnp.allclose(second.state, jnp.asarray([3.0]), atol=1e-6)
    assert prepared.linear_plan_id == refreshed.linear_plan_id
    assert prepared.linear_template_id == refreshed.linear_template_id
    assert int(refreshed.numeric_version) == int(prepared.numeric_version) + 1
    assert int(refreshed.linear_refresh_state.numeric_version) > int(
        prepared.linear_refresh_state.numeric_version
    )
    assert int(first.diagnostics.setup_refreshes) == 1
    assert int(second.diagnostics.setup_refreshes) == 0
    assert first.provenance.linear_plan_id == second.provenance.linear_plan_id


@pytest.mark.parametrize("method", (nl.NewtonKrylov(), nl.NewtonTrustRegion()))
def test_prepared_newton_refresh_and_solve_follow_existing_jit_pattern(method):
    problem = nl.NonlinearSystemProblem(
        lambda state, target: state - target,
        problem_id=f"prepared-jit-{method.method_id}",
    )
    prepared = nl.prepare_nonlinear(
        problem,
        jnp.asarray([0.0]),
        method=method,
        termination=_termination(),
        args=jnp.asarray([1.0]),
    )

    def staged(target):
        refreshed = nl.refresh_nonlinear(
            prepared,
            problem,
            jnp.zeros_like(target),
            args=target,
        )
        result = nl.solve_prepared_nonlinear(refreshed)
        return (
            result.state,
            result.residual,
            result.status,
            refreshed.numeric_version,
            refreshed.linear_refresh_state.numeric_version,
        )

    state, residual, status, version, linear_version = jax.jit(staged)(jnp.asarray([4.0]))

    assert int(status) == int(nl.NonlinearStatus.SUCCESS)
    assert jnp.allclose(state, jnp.asarray([4.0]), atol=1e-6)
    assert jnp.allclose(residual, jnp.zeros(1), atol=1e-6)
    assert int(version) == 1
    assert int(linear_version) > int(prepared.linear_refresh_state.numeric_version)


def test_prepared_nonlinear_solve_accepts_per_call_termination_budget():
    problem = nl.NonlinearSystemProblem(
        lambda state, target: state**2 - target,
        problem_id="prepared-per-call-budget",
    )
    prepared = nl.prepare_nonlinear(
        problem,
        jnp.asarray([1.0]),
        method=nl.NewtonKrylov(),
        termination=_termination(maximum_steps=1),
        args=jnp.asarray([4.0]),
    )

    limited = nl.solve_prepared_nonlinear(prepared)
    completed = nl.solve_prepared_nonlinear(
        prepared, termination=_termination(maximum_steps=12)
    )

    assert int(limited.status) == int(nl.NonlinearStatus.MAXIMUM_STEPS_REACHED)
    assert bool(completed.successful)
    assert jnp.allclose(completed.state, jnp.asarray([2.0]), atol=1e-5)
    assert limited.provenance.linear_plan_id == completed.provenance.linear_plan_id


def test_prepared_nonlinear_refresh_rejects_changed_spaces():
    problem = nl.NonlinearSystemProblem(
        lambda state, target: state - target,
        problem_id="prepared-space-identity",
    )
    prepared = nl.prepare_nonlinear(
        problem,
        jnp.zeros(1),
        termination=_termination(),
        args=jnp.ones(1),
    )

    with pytest.raises(ValueError, match="state space"):
        nl.refresh_nonlinear(
            prepared,
            problem,
            jnp.zeros(2),
            args=jnp.ones(2),
        )


def test_prepared_nonlinear_refresh_rejects_changed_linear_structure():
    space = la.PyTreeSpace(jnp.zeros(1))
    problem = nl.NonlinearSystemProblem(
        lambda state, args: args["scale"] * state - 1.0,
        state_space=space,
        residual_space=space,
        problem_id="prepared-structure-identity",
    )

    def operator(state, args):
        del state
        scale = args["scale"]
        return la.FunctionLinearOperator(
            lambda vector: scale * vector,
            source=space,
            target=space,
            operator_id=args["operator_id"],
        )

    method = nl.NewtonKrylov(
        jacobian_policy=nl.JacobianPolicy("explicit", operator=operator)
    )
    prepared = nl.prepare_nonlinear(
        problem,
        jnp.zeros(1),
        method=method,
        termination=_termination(),
        args={"scale": jnp.asarray(1.0), "operator_id": "jacobian-a"},
    )

    with pytest.raises(ValueError, match="preserve problem_id|symbolic"):
        nl.refresh_nonlinear(
            prepared,
            problem,
            jnp.zeros(1),
            args={"scale": jnp.asarray(2.0), "operator_id": "jacobian-b"},
        )


def test_prepared_nonlinear_rejects_unsupported_methods_and_is_public():
    public = {
        "PreparedNonlinearSolve",
        "prepare_nonlinear",
        "refresh_nonlinear",
        "solve_prepared_nonlinear",
    }
    assert public <= set(nl.__all__)
    assert public <= set(vars(nl))

    problem = nl.NonlinearSystemProblem(lambda state, args: state)
    with pytest.raises(ValueError, match="only NewtonKrylov and NewtonTrustRegion"):
        nl.prepare_nonlinear(
            problem,
            jnp.zeros(1),
            method=nl.NonlinearGMRES(
                nl.FunctionNonlinearUpdate(lambda state, args: state)
            ),
            termination=_termination(),
        )
