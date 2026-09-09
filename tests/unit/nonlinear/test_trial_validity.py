import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax.linalg as la
import phydrax.nonlinear as nl


def _positive_log(state, _):
    checked = eqx.error_if(state, jnp.any(state <= 0.0), "invalid residual evaluated")
    return jnp.log(checked)


def _problem():
    return nl.NonlinearSystemProblem(
        _positive_log,
        trial_validity=lambda state, _: jnp.all(state > 0.0),
        trial_validity_id="strict-positive-log-v1",
    )


@pytest.mark.parametrize("method", (nl.NewtonKrylov(), nl.NewtonTrustRegion()))
def test_invalid_initial_root_never_evaluates_residual_or_jacobian(method):
    result = jax.jit(
        lambda state: method.solve(
            _problem(),
            state,
            termination=nl.NonlinearTermination(),
        )
    )(jnp.asarray([-1.0]))
    assert result.status == int(nl.NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE)
    assert result.diagnostics.domain_failures == 1
    assert result.diagnostics.residual_evaluations == 0
    assert result.diagnostics.jacobian_preparations == 0
    assert jnp.array_equal(result.state, jnp.asarray([-1.0]))


def test_newton_shortens_out_of_domain_trials_without_evaluating_them():
    result = jax.jit(
        lambda initial: nl.NewtonKrylov().solve(
            _problem(),
            initial,
            termination=nl.NonlinearTermination(
                absolute_residual=1e-10,
                relative_residual=0.0,
                maximum_steps=30,
            ),
        )
    )(jnp.asarray([10.0]))
    assert result.successful
    assert result.diagnostics.domain_failures > 0
    assert result.diagnostics.nonfinite_trials == 0
    assert jnp.allclose(result.state, jnp.ones(1), atol=1e-9)


def test_explicit_jacobian_is_not_called_on_rejected_initial_state():
    space = la.ArraySpace((1,), dtype=jnp.float64)

    def jacobian(state, _):
        checked = eqx.error_if(state, jnp.any(state <= 0.0), "invalid Jacobian evaluated")
        return la.DenseLinearOperator(jnp.diag(1.0 / checked), source=space, target=space)

    problem = nl.NonlinearSystemProblem(
        _positive_log,
        state_space=space,
        residual_space=space,
        trial_validity=lambda state, _: jnp.all(state > 0.0),
        trial_validity_id="strict-positive-log-v1",
    )
    method = nl.NewtonKrylov(
        jacobian_policy=nl.JacobianPolicy("explicit", operator=jacobian)
    )
    result = jax.jit(
        lambda state: method.solve(
            problem,
            state,
            termination=nl.NonlinearTermination(),
        )
    )(jnp.asarray([-1.0]))
    assert result.status == int(nl.NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE)
    assert result.diagnostics.domain_failures == 1


def test_prepared_refresh_refuses_a_changed_domain_contract():
    problem = _problem()
    prepared = nl.prepare_nonlinear(problem, jnp.asarray([1.0]))
    changed = nl.NonlinearSystemProblem(
        _positive_log,
        problem_id=problem.problem_id,
        trial_validity=lambda state, _: jnp.all(state > 0.5),
        trial_validity_id="strict-positive-log-v2",
    )
    with pytest.raises(ValueError, match="trial_validity_id"):
        nl.refresh_nonlinear(prepared, changed, jnp.asarray([1.0]))


@pytest.mark.parametrize("explicit", (False, True))
def test_mapped_newton_never_evaluates_invalid_residual_or_jacobian_lane(explicit):
    space = la.ArraySpace((1,), dtype=jnp.float64)

    def jacobian(state, _):
        state = eqx.error_if(
            state, jnp.any(state <= 0.0), "invalid mapped Jacobian evaluated"
        )
        return la.DenseLinearOperator(jnp.diag(1.0 / state), source=space, target=space)

    problem = nl.NonlinearSystemProblem(
        _positive_log,
        state_space=space,
        residual_space=space,
        trial_validity=lambda state, _: jnp.all(state > 0.0),
        trial_validity_id="mapped-strict-positive-log-v1",
    )
    method = nl.NewtonKrylov(
        jacobian_policy=(
            nl.JacobianPolicy("explicit", operator=jacobian)
            if explicit
            else nl.JacobianPolicy()
        ),
    )
    result = jax.jit(
        jax.vmap(
            lambda state: method.solve(
                problem,
                state,
                termination=nl.NonlinearTermination(
                    absolute_residual=1e-10,
                    relative_residual=0.0,
                    maximum_steps=30,
                ),
            )
        )
    )(jnp.asarray([[-1.0], [10.0]]))
    assert jnp.array_equal(
        result.status,
        jnp.asarray(
            [
                int(nl.NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE),
                int(nl.NonlinearStatus.SUCCESS),
            ]
        ),
    )
    assert result.diagnostics.domain_failures[0] == 1
    assert result.diagnostics.residual_evaluations[0] == 0
    assert result.diagnostics.jacobian_preparations[0] == 0
    assert result.diagnostics.domain_failures[1] > 0
    assert jnp.allclose(result.state[:, 0], jnp.asarray([-1.0, 1.0]), atol=1e-9)


def test_mapped_domain_guard_preserves_jvp_and_transpose_in_both_transform_orders():
    problem = _problem()
    states = jnp.asarray([[-1.0], [4.0]])
    mapped = jax.vmap(problem.residual)
    values, tangent = jax.jit(lambda x: jax.jvp(mapped, (x,), (jnp.ones_like(x),)))(
        states
    )
    reverse_after_map = jax.jit(jax.grad(lambda x: jnp.sum(mapped(x))))(states)
    map_after_reverse = jax.jit(
        jax.vmap(jax.grad(lambda x: jnp.sum(problem.residual(x))))
    )(states)
    assert jnp.allclose(values[:, 0], jnp.asarray([0.0, jnp.log(4.0)]))
    assert jnp.allclose(tangent[:, 0], jnp.asarray([0.0, 0.25]))
    assert jnp.allclose(reverse_after_map, tangent)
    assert jnp.allclose(map_after_reverse, tangent)

    nested_states = jnp.asarray([[[-1.0], [1.0]], [[4.0], [-2.0]]])
    nested = jax.vmap(jax.vmap(problem.residual))
    nested_gradient = jax.jit(jax.grad(lambda x: jnp.sum(nested(x))))(nested_states)
    assert jnp.allclose(nested_gradient, jnp.asarray([[[0.0], [1.0]], [[0.25], [0.0]]]))
