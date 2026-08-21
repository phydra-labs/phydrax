import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax.nonlinear as nl


def _termination(*, maximum_steps=20):
    return nl.NonlinearTermination(
        absolute_residual=1e-11,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=maximum_steps,
    )


def test_implicit_root_result_retains_native_evidence_and_analytic_derivatives():
    problem = nl.NonlinearSystemProblem(
        lambda state, target: state**2 - target,
        problem_id="implicit-square",
    )

    def root(target):
        result = nl.implicit_root_result(
            problem,
            jnp.asarray([1.0, 1.0]),
            termination=_termination(),
            args=target,
        )
        return result.state

    target = jnp.asarray([1.0, 4.0])
    value, tangent = jax.jvp(root, (target,), (jnp.ones_like(target),))
    gradient = jax.grad(lambda argument: jnp.sum(root(argument)))(target)
    result = nl.implicit_root_result(
        problem,
        jnp.asarray([1.0, 1.0]),
        termination=_termination(),
        args=target,
    )

    assert result.successful
    assert result.provenance.problem_id == "implicit-square"
    assert result.diagnostics.iterations > 0
    assert jnp.allclose(value, jnp.asarray([1.0, 2.0]), rtol=1e-9, atol=1e-11)
    assert jnp.allclose(tangent, jnp.asarray([0.5, 0.25]), rtol=1e-8, atol=1e-10)
    assert jnp.allclose(gradient, tangent, rtol=1e-8, atol=1e-10)


def test_implicit_root_result_recomputes_differentiable_auxiliary_at_root():
    problem = nl.NonlinearSystemProblem(
        lambda state, target: (state**2 - target, target * state),
        has_aux=True,
        problem_id="implicit-auxiliary",
    )

    def observable(target):
        result = nl.implicit_root_result(
            problem,
            jnp.asarray(1.0),
            termination=_termination(),
            args=target,
        )
        return result.auxiliary

    value, gradient = jax.value_and_grad(observable)(jnp.asarray(4.0))

    assert jnp.allclose(value, 8.0, rtol=1e-9, atol=1e-11)
    assert jnp.allclose(gradient, 3.0, rtol=1e-8, atol=1e-10)


def test_prepared_implicit_root_refresh_preserves_symbolic_linear_identity():
    problem = nl.NonlinearSystemProblem(
        lambda state, target: state**2 - target,
        problem_id="prepared-implicit-square",
    )
    prepared = nl.prepare_nonlinear(
        problem,
        jnp.asarray([1.0, 1.0]),
        termination=_termination(),
        args=jnp.asarray([1.0, 1.0]),
    )
    refreshed = nl.refresh_nonlinear(
        prepared,
        problem,
        jnp.asarray([2.0, 3.0]),
        args=jnp.asarray([4.0, 9.0]),
    )
    result = nl.implicit_root_result(refreshed)

    assert result.successful
    assert jnp.allclose(result.state, jnp.asarray([2.0, 3.0]), atol=1e-11)
    assert refreshed.linear_template_id == prepared.linear_template_id
    assert refreshed.linear_plan_id == prepared.linear_plan_id
    assert refreshed.numeric_version == prepared.numeric_version + 1
    assert result.provenance.linear_plan_id == prepared.linear_plan_id


def test_failed_implicit_root_remains_inspectable_and_checked_root_raises():
    problem = nl.NonlinearSystemProblem(
        lambda state, _: state**2 - 2.0,
        problem_id="failed-implicit-root",
    )
    termination = _termination(maximum_steps=1)
    result = nl.implicit_root_result(
        problem,
        jnp.asarray(10.0),
        termination=termination,
    )

    assert not result.successful
    assert result.status == int(nl.NonlinearStatus.MAXIMUM_STEPS_REACHED)
    assert jnp.isfinite(result.state)
    with pytest.raises(
        eqx.EquinoxRuntimeError, match="Implicit nonlinear root solve failed"
    ):
        nl.implicit_root(problem, jnp.asarray(10.0), termination=termination)
