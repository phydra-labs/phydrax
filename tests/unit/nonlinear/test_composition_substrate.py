#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax as phx


nl = phx.nonlinear
la = phx.linalg


def _termination(**kwargs):
    return nl.NonlinearTermination(
        absolute_residual=kwargs.pop("absolute_residual", 1e-9),
        relative_residual=0.0,
        maximum_steps=kwargs.pop("maximum_steps", 20),
        **kwargs,
    )


def test_function_update_distinguishes_application_from_root_convergence_and_refresh():
    problem = nl.NonlinearSystemProblem(
        lambda state, target: state - target,
        problem_id="finite-update",
    )
    update = nl.FunctionNonlinearUpdate(
        lambda state, target: state + 0.5 * (target - state),
        update_id="half-correction",
    )
    prepared = nl.prepare_nonlinear_update(
        problem,
        jnp.asarray([0.0]),
        update,
        args=jnp.asarray([2.0]),
    )
    result, next_prepared = nl.apply_prepared_nonlinear_update(
        prepared,
        jnp.asarray([0.0]),
        args=jnp.asarray([2.0]),
    )

    assert bool(result.applied)
    assert jnp.allclose(result.state, jnp.asarray([1.0]))
    assert float(result.diagnostics.final_residual_norm) == pytest.approx(1.0)
    assert result.status == int(nl.NonlinearUpdateStatus.APPLIED)
    assert next_prepared.plan.plan_id == prepared.plan.plan_id

    refreshed = nl.refresh_nonlinear_update(
        next_prepared,
        problem,
        result.state,
        args=jnp.asarray([3.0]),
    )
    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert int(refreshed.numeric_version) == 1


def test_update_budget_rejects_before_callable_proposal():
    problem = nl.NonlinearSystemProblem(lambda state, target: state - target)
    update = nl.FunctionNonlinearUpdate(lambda state, target: target)
    prepared = nl.prepare_nonlinear_update(
        problem,
        jnp.asarray([0.0]),
        update,
        args=jnp.asarray([2.0]),
    )
    result, _ = nl.apply_prepared_nonlinear_update(
        prepared,
        jnp.asarray([0.0]),
        args=jnp.asarray([2.0]),
        control=nl.NonlinearUpdateControl(maximum_residual_evaluations=1),
    )

    assert not bool(result.applied)
    assert result.status == int(nl.NonlinearUpdateStatus.BUDGET_EXHAUSTED)
    assert jnp.allclose(result.state, jnp.asarray([0.0]))


def test_additive_multiplicative_and_optimal_compositions_preserve_physical_residual():
    problem = nl.NonlinearSystemProblem(lambda state, target: state - target)
    half = nl.FunctionNonlinearUpdate(
        lambda state, target: state + 0.5 * (target - state),
        update_id="half",
    )
    quarter = nl.FunctionNonlinearUpdate(
        lambda state, target: state + 0.25 * (target - state),
        update_id="quarter",
    )
    initial = jnp.asarray([0.0])
    target = jnp.asarray([2.0])

    additive = nl.CompositeNonlinearUpdate(
        (half, quarter),
        kind="additive",
        weights=(1.0, 2.0),
    )
    additive_prepared = nl.prepare_nonlinear_update(
        problem,
        initial,
        additive,
        args=target,
    )
    additive_result, _ = nl.apply_prepared_nonlinear_update(
        additive_prepared,
        initial,
        args=target,
    )
    assert bool(additive_result.applied)
    assert jnp.allclose(additive_result.state, target)
    assert jnp.allclose(additive_result.residual, jnp.zeros_like(target))
    assert len(additive_result.components) == 2

    multiplicative = nl.CompositeNonlinearUpdate(
        (half, half),
        kind="multiplicative",
    )
    multiplicative_prepared = nl.prepare_nonlinear_update(
        problem,
        initial,
        multiplicative,
        args=target,
    )
    multiplicative_result, _ = nl.apply_prepared_nonlinear_update(
        multiplicative_prepared,
        initial,
        args=target,
    )
    assert jnp.allclose(multiplicative_result.state, jnp.asarray([1.5]))
    assert jnp.allclose(multiplicative_result.residual, jnp.asarray([-0.5]))

    optimal = nl.CompositeNonlinearUpdate(
        (quarter, half),
        kind="residual-optimal",
        regularization=0.0,
    )
    optimal_prepared = nl.prepare_nonlinear_update(
        problem,
        initial,
        optimal,
        args=target,
    )
    optimal_result, _ = nl.apply_prepared_nonlinear_update(
        optimal_prepared,
        initial,
        args=target,
    )
    assert bool(optimal_result.applied)
    assert float(optimal_result.diagnostics.final_residual_norm) <= 1.0


def test_typed_ngmres_and_richardson_solve_and_raw_callable_is_rejected():
    problem = nl.NonlinearSystemProblem(lambda state, target: state - target)
    exact = nl.FunctionNonlinearUpdate(lambda state, target: target)

    ngmres = nl.NonlinearGMRES(exact)
    ngmres_result = ngmres.solve(
        problem,
        jnp.asarray([0.0]),
        args=jnp.asarray([2.0]),
        termination=_termination(maximum_steps=5),
    )
    richardson_result = nl.NonlinearRichardson(exact).solve(
        problem,
        jnp.asarray([0.0]),
        args=jnp.asarray([2.0]),
        termination=_termination(maximum_steps=5),
    )

    assert bool(ngmres_result.successful)
    assert bool(richardson_result.successful)
    with pytest.raises(TypeError, match="AbstractNonlinearUpdate"):
        nl.NonlinearGMRES(lambda state, args: state)


def _subdomain(index, target):
    space = la.ArraySpace((1,), dtype=jnp.float64)
    return nl.NonlinearSubdomain(
        lambda state: state[index : index + 1],
        lambda residual: residual[index : index + 1],
        lambda correction: (
            jnp.zeros((2,), dtype=correction.dtype).at[index].set(correction[0])
        ),
        lambda local, global_state, args: local - args[index : index + 1],
        nl.FunctionNonlinearUpdate(
            lambda local, context: context[1][index : index + 1],
            update_id=f"exact-local-{index}",
        ),
        state_space=space,
        residual_space=space,
        subdomain_id=f"block-{index}",
    )


def test_nonlinear_schwarz_gauss_seidel_and_aspin_certify_global_system():
    target = jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    problem = nl.NonlinearSystemProblem(
        lambda state, args: state - args,
        problem_id="block-root",
    )
    subdomains = (_subdomain(0, target), _subdomain(1, target))
    initial = jnp.zeros((2,), dtype=jnp.float64)

    for update in (
        nl.NonlinearAdditiveSchwarz(subdomains),
        nl.NonlinearMultiplicativeSchwarz(subdomains),
        nl.NonlinearGaussSeidel(subdomains),
    ):
        prepared = nl.prepare_nonlinear_update(problem, initial, update, args=target)
        result, _ = nl.apply_prepared_nonlinear_update(
            prepared,
            initial,
            args=target,
        )
        assert bool(result.applied)
        assert jnp.allclose(result.state, target)
        assert jnp.allclose(result.residual, jnp.zeros_like(target))

    aspin = nl.ASPIN(nl.NonlinearAdditiveSchwarz(subdomains))
    aspin_result = aspin.solve(
        problem,
        initial,
        args=target,
        termination=_termination(maximum_steps=20),
    )
    assert bool(aspin_result.successful)
    assert jnp.allclose(aspin_result.state, target, atol=1e-8)
    assert aspin_result.provenance.method_id == "aspin"
    assert not aspin_result.diagnostics.counts_complete


def test_prepared_function_update_follows_filtered_jit_pattern():
    problem = nl.NonlinearSystemProblem(lambda state, target: state - target)
    prepared = nl.prepare_nonlinear_update(
        problem,
        jnp.asarray([0.0]),
        nl.FunctionNonlinearUpdate(lambda state, target: target),
        args=jnp.asarray([2.0]),
    )

    @eqx.filter_jit
    def apply(current, state, target):
        result, next_current = nl.apply_prepared_nonlinear_update(
            current,
            state,
            args=target,
        )
        return result.state, next_current

    state, next_prepared = apply(
        prepared,
        jnp.asarray([0.0]),
        jnp.asarray([2.0]),
    )
    assert jnp.allclose(state, jnp.asarray([2.0]))
    assert next_prepared.plan.plan_id == prepared.plan.plan_id
