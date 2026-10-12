from collections.abc import Callable
from math import exp, log
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array

import phydrax.linalg as la
import phydrax.nonlinear as nl


def _positive_log(state: Any, _: Any) -> Any:
    checked = eqx.error_if(state, jnp.any(state <= 0.0), "invalid residual evaluated")
    return jnp.log(checked)


def _problem() -> Any:
    return nl.NonlinearSystemProblem(
        _positive_log,
        trial_validity=lambda state, _: jnp.all(state > 0.0),
        trial_validity_id="strict-positive-log",
    )


def test_trial_validity_scenario_1() -> None:
    for method in (nl.NewtonKrylov(), nl.NewtonTrustRegion()):
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
    problem = _problem()
    prepared = nl.prepare_nonlinear(problem, jnp.asarray([1.0]))
    changed = nl.NonlinearSystemProblem(
        _positive_log,
        problem_id=problem.problem_id,
        trial_validity=lambda state, _: jnp.all(state > 0.5),
        trial_validity_id="strict-log-above-half",
    )
    with pytest.raises(ValueError, match="trial_validity_id"):
        nl.refresh_nonlinear(prepared, changed, jnp.asarray([1.0]))


def test_explicit_jacobian_is_not_called_on_rejected_initial_state() -> None:
    space = la.ArraySpace((1,), dtype=jnp.float64)

    def jacobian(state: Any, _: Any) -> Any:
        checked = eqx.error_if(state, jnp.any(state <= 0.0), "invalid Jacobian evaluated")
        return la.DenseLinearOperator(jnp.diag(1.0 / checked), source=space, target=space)

    problem = nl.NonlinearSystemProblem(
        _positive_log,
        state_space=space,
        residual_space=space,
        trial_validity=lambda state, _: jnp.all(state > 0.0),
        trial_validity_id="strict-positive-log",
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


def test_mapped_newton_never_evaluates_invalid_residual_or_jacobian_lane() -> None:
    for explicit in (False, True):
        space = la.ArraySpace((1,), dtype=jnp.float64)

        def jacobian(state: Any, _: Any) -> Any:
            state = eqx.error_if(
                state, jnp.any(state <= 0.0), "invalid mapped Jacobian evaluated"
            )
            return la.DenseLinearOperator(
                jnp.diag(1.0 / state), source=space, target=space
            )

        problem = nl.NonlinearSystemProblem(
            _positive_log,
            state_space=space,
            residual_space=space,
            trial_validity=lambda state, _: jnp.all(state > 0.0),
            trial_validity_id="mapped-strict-positive-log",
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


def test_trial_validity_scenario_2() -> None:
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
    for method in (
        nl.NewtonKrylov(),
        nl.NewtonTrustRegion(trust_region=nl.RootTrustRegion(initial_radius=100.0)),
    ):
        result = method.solve(
            _problem(),
            jnp.asarray([10.0]),
            termination=nl.NonlinearTermination(
                absolute_residual=0.0,
                relative_residual=0.0,
                maximum_steps=4,
                maximum_evaluations=2,
            ),
        )

        assert int(result.diagnostics.residual_evaluations) == 2
        assert int(result.diagnostics.domain_failures) > 0
        assert not jnp.array_equal(result.state, jnp.asarray([10.0]))


def _constitutive_problem(
    coefficient: Array, kind: Literal["local-root", "eigenspectral", "mixed"]
) -> nl.NonlinearSystemProblem:
    root_plan = nl.LocalRootPlan(plan_id="guarded-positive-local-root")
    dimension = 4 if kind == "mixed" else 2
    vector_plan = nl.VectorLocalRootPlan(
        dimension, plan_id="guarded-positive-vector-local-root"
    )
    space = la.ArraySpace((2,), dtype=coefficient.dtype)

    def residual(state: Array, _: None) -> Array:
        state = eqx.error_if(
            state, jnp.any(state <= 0.0), "invalid constitutive branch evaluated"
        )
        temperature, pressure = state
        values = jnp.log(state)
        if kind in ("local-root", "mixed"):
            root = root_plan.solve(
                lambda value: value**2 - coefficient * temperature * pressure,
                jnp.asarray(1.0, dtype=state.dtype),
            )
            values = values + jnp.asarray([root, -root])
            targets = jnp.tile(state, dimension // 2) + jnp.arange(
                dimension, dtype=state.dtype
            )
            local_space = la.ArraySpace((dimension,), dtype=state.dtype)
            local_problem = nl.NonlinearSystemProblem(
                lambda value, _: value**2 - coefficient * targets,
                state_space=local_space,
                residual_space=local_space,
                trial_validity=lambda value, _: jnp.all(value > 0.0),
                trial_validity_id="positive-nested-return-map",
            )
            vector = vector_plan.solve(
                local_problem.residual,
                jnp.ones(dimension, dtype=state.dtype),
            )
            values = values + jnp.asarray([jnp.sum(vector[::2]), jnp.sum(vector[1::2])])
        if kind in ("eigenspectral", "mixed"):
            matrix = jnp.asarray(
                [[temperature, coefficient], [coefficient, pressure + 3.0]]
            )
            eigenvalues, _ = jnp.linalg.eigh(matrix)
            spectral = jnp.sum(jnp.log(eigenvalues))
            values = values + jnp.asarray([spectral, 2.0 * spectral])
        return values

    return nl.NonlinearSystemProblem(
        residual,
        state_space=space,
        residual_space=space,
        trial_validity=lambda state, _: jnp.all(state > 0.0),
        trial_validity_id="positive-temperature-pressure",
        problem_id=f"guarded-{kind}-constitutive",
    )


def _constitutive_reference(
    state: Array,
    coefficient: Array,
    kind: Literal["local-root", "eigenspectral", "mixed"],
) -> Array:
    temperature, pressure = state
    values = jnp.log(state)
    if kind in ("local-root", "mixed"):
        root = jnp.sqrt(coefficient * temperature * pressure)
        values = values + jnp.asarray([root, -root])
        dimension = 4 if kind == "mixed" else 2
        targets = jnp.tile(state, dimension // 2) + jnp.arange(
            dimension, dtype=state.dtype
        )
        vector = jnp.sqrt(coefficient * targets)
        values = values + jnp.asarray([jnp.sum(vector[::2]), jnp.sum(vector[1::2])])
    if kind in ("eigenspectral", "mixed"):
        spectral = jnp.log(temperature * (pressure + 3.0) - coefficient**2)
        values = values + jnp.asarray([spectral, 2.0 * spectral])
    return values


@pytest.mark.parametrize("kind", ["local-root", "eigenspectral", "mixed"])
def test_guarded_native_preparation_preserves_constitutive_jvp_and_vjp(
    kind: Literal["local-root", "eigenspectral", "mixed"],
) -> None:
    coefficient = jnp.asarray(0.4, dtype=jnp.float64)
    state = jnp.asarray([2.0, 4.0], dtype=jnp.float64)
    direction = jnp.asarray([0.3, -0.2], dtype=jnp.float64)
    covector = jnp.asarray([-0.5, 0.7], dtype=jnp.float64)
    problem = _constitutive_problem(coefficient, kind)
    reference = lambda value: _constitutive_reference(value, coefficient, kind)
    jacobian = jax.jacfwd(reference)(state)
    linearization = la.prepare_linearization(problem.residual, state)
    prepared = nl.prepare_nonlinear(problem, state)
    assert jnp.allclose(linearization.primal, reference(state), rtol=1e-10)
    assert jnp.allclose(
        jax.jit(lambda value: linearization.jvp(value))(direction), jacobian @ direction
    )
    assert jnp.allclose(
        jax.jit(lambda value: linearization.vjp(value))(covector), jacobian.T @ covector
    )
    assert jnp.allclose(prepared.jacobian.operator.mv(direction), jacobian @ direction)
    assert jnp.allclose(
        prepared.jacobian.operator.transpose_mv(covector), jacobian.T @ covector
    )


def test_rematerialized_guarded_constitutive_hessian_action() -> None:
    with jax.enable_x64():
        coefficient = jnp.asarray(0.4, dtype=jnp.float64)
        point = jnp.asarray([2.0, 4.0], dtype=jnp.float64)
        direction = jnp.asarray([0.3, -0.2], dtype=jnp.float64)
        problem = _constitutive_problem(coefficient, "mixed")
        residual = jax.checkpoint(problem.residual)
        reference = lambda value: _constitutive_reference(value, coefficient, "mixed")
        actual = jax.jit(
            lambda value: jax.jvp(
                jax.grad(lambda state: jnp.sum(residual(state))),
                (value,),
                (direction,),
            )[1]
        )(point)
        expected = jax.jvp(
            jax.grad(lambda state: jnp.sum(reference(state))),
            (point,),
            (direction,),
        )[1]
        assert jnp.allclose(actual, expected, rtol=1e-8, atol=1e-10)


def test_guarded_constitutive_auxiliary_tree_preserves_primal_evidence() -> None:
    with jax.enable_x64():
        coefficient = jnp.asarray(0.4, dtype=jnp.float64)
        point = jnp.asarray([2.0, 4.0], dtype=jnp.float64)
        covector = jnp.asarray([-0.5, 0.7], dtype=jnp.float64)
        constitutive = _constitutive_problem(coefficient, "mixed")

        def residual(state: Array, _: None) -> tuple[Array, dict[str, Array]]:
            return constitutive.residual(state), {
                "temperature_sum": jnp.sum(state),
                "physical_domain": jnp.all(state > 0.0),
            }

        problem = nl.NonlinearSystemProblem(
            residual,
            has_aux=True,
            trial_validity=lambda state, _: jnp.all(state > 0.0),
            trial_validity_id="positive-auxiliary-constitutive",
        )
        prepared = la.prepare_linearization(problem.evaluate, point, has_aux=True)
        jacobian = jax.jacfwd(
            lambda state: _constitutive_reference(state, coefficient, "mixed")
        )(point)
        assert prepared.auxiliary["temperature_sum"] == jnp.sum(point)
        assert prepared.auxiliary["physical_domain"]
        assert jnp.allclose(prepared.vjp(covector), jacobian.T @ covector)
        values, auxiliary = jax.jit(jax.vmap(problem.evaluate))(
            jnp.stack((point, -point))
        )
        assert jnp.allclose(
            values[0], _constitutive_reference(point, coefficient, "mixed")
        )
        assert jnp.array_equal(values[1], jnp.zeros_like(point))
        assert jnp.array_equal(
            auxiliary["temperature_sum"], jnp.asarray([6.0, 0.0], dtype=point.dtype)
        )
        assert jnp.array_equal(
            auxiliary["physical_domain"], jnp.asarray([True, False], dtype=jnp.bool_)
        )


def test_guarded_mixed_closure_captures_parameters_without_executing_invalid_lanes() -> (
    None
):
    coefficient = jnp.asarray(0.4, dtype=jnp.float64)
    states = jnp.asarray(
        [[[2.0, 4.0], [-1.0, 4.0]], [[3.0, 2.0], [2.0, 0.0]]],
        dtype=jnp.float64,
    )
    direction = jnp.asarray(
        [[[0.3, -0.2], [1.0, 1.0]], [[-0.1, 0.4], [1.0, 1.0]]],
        dtype=jnp.float64,
    )
    covector = jnp.asarray(
        [[[0.7, -0.5], [0.7, -0.5]], [[0.2, 0.9], [0.2, 0.9]]],
        dtype=jnp.float64,
    )

    def mapped(values: Array, parameter: Array) -> Array:
        problem = _constitutive_problem(parameter, "mixed")
        return jax.vmap(jax.vmap(problem.residual))(values)

    def actions(values: Array, parameter: Array) -> tuple[Array, Array, Array]:
        linearization = la.prepare_linearization(lambda x: mapped(x, parameter), values)
        return (
            linearization.primal,
            linearization.jvp(direction),
            linearization.vjp(covector),
        )

    primal, tangent, reverse = jax.jit(actions)(states, coefficient)
    problem = _constitutive_problem(coefficient, "mixed")
    mapped_jacobians = jax.jit(jax.vmap(jax.vmap(jax.jacfwd(problem.residual))))(states)
    expected_parameter = jnp.zeros_like(coefficient)
    for row, column in ((0, 0), (1, 0)):
        state = states[row, column]
        function: Callable[[Array], Array] = lambda value: _constitutive_reference(
            value, coefficient, "mixed"
        )
        jacobian = jax.jacfwd(function)(state)
        assert jnp.allclose(primal[row, column], function(state), rtol=1e-10)
        assert jnp.allclose(tangent[row, column], jacobian @ direction[row, column])
        assert jnp.allclose(reverse[row, column], jacobian.T @ covector[row, column])
        assert jnp.allclose(mapped_jacobians[row, column], jacobian)
        expected_parameter = expected_parameter + jax.grad(
            lambda parameter: (
                covector[row, column] @ _constitutive_reference(state, parameter, "mixed")
            )
        )(coefficient)
    assert jnp.array_equal(primal[:, 1], jnp.zeros((2, 2), dtype=states.dtype))
    assert jnp.array_equal(tangent[:, 1], jnp.zeros((2, 2), dtype=states.dtype))
    assert jnp.array_equal(reverse[:, 1], jnp.zeros((2, 2), dtype=states.dtype))
    assert jnp.array_equal(
        mapped_jacobians[:, 1], jnp.zeros((2, 2, 2), dtype=states.dtype)
    )
    parameter_gradient = jax.jit(
        jax.grad(lambda parameter: jnp.sum(mapped(states, parameter) * covector))
    )(coefficient)
    assert jnp.allclose(parameter_gradient, expected_parameter)


def test_guarded_derivatives_emit_scientific_effects_once_in_lane_order() -> None:
    observed: list[float] = []

    def record(value: Array) -> None:
        observed.append(float(value))

    def residual(state: Array, _: None) -> Array:
        state = eqx.error_if(
            state, jnp.any(state <= 0.0), "invalid effectful branch evaluated"
        )
        jax.debug.callback(record, state[0], ordered=True)
        return jnp.log(state)

    space = la.ArraySpace((1,), dtype=jnp.float64)
    problem = nl.NonlinearSystemProblem(
        residual,
        state_space=space,
        residual_space=space,
        trial_validity=lambda state, _: jnp.all(state > 0.0),
        trial_validity_id="positive-effectful-constitutive",
    )
    states = jnp.asarray([[[2.0], [-1.0]], [[3.0], [0.0]]], dtype=jnp.float64)
    mapped = jax.vmap(jax.vmap(problem.residual))
    values, tangent = jax.jit(lambda x: jax.jvp(mapped, (x,), (jnp.ones_like(x),)))(
        states
    )
    tangent.block_until_ready()
    assert observed == [2.0, 3.0]
    assert jnp.allclose(tangent, jnp.asarray([[[0.5], [0.0]], [[1.0 / 3.0], [0.0]]]))
    assert jnp.array_equal(values[:, 1], jnp.zeros((2, 1), dtype=states.dtype))
    observed.clear()
    gradient = jax.jit(jax.grad(lambda x: jnp.sum(mapped(x))))(states)
    gradient.block_until_ready()
    assert observed == [2.0, 3.0]
    assert jnp.allclose(gradient, tangent)


def test_guarded_vector_root_refuses_singular_implicit_derivative() -> None:
    plan = nl.VectorLocalRootPlan(2, plan_id="singular-guarded-return-map")
    space = la.ArraySpace((2,), dtype=jnp.float64)

    def residual(state: Array, _: None) -> Array:
        return plan.solve(
            lambda local: local**2 - (state - 1.0),
            jnp.zeros_like(state),
        ) + jnp.ones_like(state)

    problem = nl.NonlinearSystemProblem(
        residual,
        state_space=space,
        residual_space=space,
        trial_validity=lambda state, _: jnp.all(state > 0.0),
        trial_validity_id="positive-singular-return-map",
    )
    state = jnp.ones(2, dtype=jnp.float64)
    forward = eqx.filter_jit(
        lambda value: jax.jvp(problem.residual, (value,), (jnp.ones_like(value),))[1]
    )
    reverse = eqx.filter_jit(jax.grad(lambda value: jnp.sum(problem.residual(value))))
    _, diagnostic = plan.solve_with_diagnostics(
        lambda local: local**2 - (state - 1.0), jnp.zeros_like(state)
    )
    assert not diagnostic.converged
    assert jnp.all(jnp.isnan(forward(state)))
    assert jnp.all(jnp.isnan(reverse(state)))
    result = jax.jit(
        lambda value: nl.NewtonKrylov().solve(
            problem, value, termination=nl.NonlinearTermination(maximum_steps=2)
        )
    )(state)
    assert result.status == int(nl.NonlinearStatus.NONFINITE_EVALUATION)
    assert not result.successful
    assert jnp.array_equal(result.state, state)


def test_vector_local_root_terminal_auxiliary_preserves_higher_derivatives() -> None:
    plan = nl.VectorLocalRootPlan(
        1, maximum_steps=12, tolerance=1.0e-12, plan_id="terminal-auxiliary"
    )
    initial = jnp.ones((1,), dtype=jnp.float64)

    def solution(parameter: Array) -> Array:
        def residual(value: Array) -> tuple[Array, tuple[Array, Array]]:
            equation = value * value - parameter
            return equation, (value**3 + parameter, parameter * value)

        root, diagnostic, auxiliary = plan.solve_with_diagnostics(
            residual, initial, has_aux=True
        )
        return jnp.concatenate(
            (
                root,
                diagnostic.residual,
                diagnostic.jacobian.reshape(-1),
                auxiliary[0],
                auxiliary[1],
            )
        )

    def reference(parameter: Array) -> Array:
        root = jnp.sqrt(parameter)
        return jnp.concatenate(
            (
                root,
                root * root - parameter,
                2.0 * root,
                root**3 + parameter,
                parameter * root,
            )
        )

    parameter = jnp.asarray([2.0], dtype=jnp.float64)
    direction = jnp.asarray([0.7], dtype=jnp.float64)
    actual, actual_jvp = jax.jvp(solution, (parameter,), (direction,))
    expected, expected_jvp = jax.jvp(reference, (parameter,), (direction,))
    assert jnp.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)
    assert jnp.allclose(actual_jvp, expected_jvp, rtol=1.0e-11, atol=1.0e-12)
    weight = jnp.asarray([0.2, 0.3, -0.4, 0.6, 0.8], dtype=jnp.float64)
    actual_gradient = jax.grad(lambda value: weight @ solution(value))
    expected_gradient = jax.grad(lambda value: weight @ reference(value))
    assert jnp.allclose(
        actual_gradient(parameter),
        expected_gradient(parameter),
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    actual_hvp = jax.jvp(actual_gradient, (parameter,), (direction,))[1]
    expected_hvp = jax.jvp(expected_gradient, (parameter,), (direction,))[1]
    assert jnp.allclose(actual_hvp, expected_hvp, rtol=1.0e-10, atol=1.0e-12)


def test_vector_local_root_terminal_auxiliary_requires_pair() -> None:
    plan = nl.VectorLocalRootPlan(1, plan_id="invalid-terminal-auxiliary")
    with pytest.raises(TypeError):
        plan.solve_with_diagnostics(  # ty: ignore[no-matching-overload]
            lambda value: value * value - 2.0,
            jnp.ones((1,), dtype=jnp.float64),
            has_aux=True,
        )


def test_scalar_root_refuses_singular_forward_and_reverse_implicit_derivatives() -> None:
    plan = nl.LocalRootPlan(plan_id="singular-scalar-return-map")
    parameter = jnp.asarray(0.0, dtype=jnp.float64)
    initial = jnp.asarray(0.0, dtype=jnp.float64)

    def solution(value: Array) -> Array:
        return plan.solve(lambda local: local**2 - value, initial)

    _, diagnostic = plan.solve_with_diagnostics(
        lambda local: local**2 - parameter, initial
    )
    assert not bool(diagnostic.converged)
    _, forward = jax.jvp(solution, (parameter,), (jnp.ones_like(parameter),))
    reverse = jax.grad(solution)(parameter)
    assert bool(jnp.isnan(forward))
    assert bool(jnp.isnan(reverse))


def _guarded_polynomial(state: Array, targets: Array) -> Array:
    checked = eqx.error_if(
        state, jnp.any(state <= 0.0), "invalid polynomial branch evaluated"
    )
    return jnp.stack((checked[0] ** 2 - targets[0], checked[1] ** 3 - targets[1]))


@pytest.mark.parametrize("storage", ["store", "rematerialize"])
def test_guarded_linearization_retains_exact_primal_and_dual_actions(
    storage: la.RematerializationPolicy,
) -> None:
    with jax.enable_x64():
        point = jnp.asarray([1.2, 1.3], dtype=jnp.float64)
        targets = jnp.asarray([4.0, 8.0], dtype=jnp.float64)
        problem = nl.NonlinearSystemProblem(
            _guarded_polynomial,
            trial_validity=lambda state, _: jnp.all(state > 0.0),
            trial_validity_id="positive-polynomial",
            problem_id="guarded-polynomial",
        )
        policy = nl.JacobianPolicy(linearization_policy=la.LinearizationPolicy(storage))
        prepared = eqx.filter_jit(
            lambda state: nl.prepare_jacobian(problem, state, policy, targets)
        )(point)
        direction = jnp.asarray([0.3, -0.4], dtype=jnp.float64)
        cotangent = jnp.asarray([-0.2, 0.5], dtype=jnp.float64)
        assert jnp.allclose(prepared.residual, jnp.asarray([-2.56, -5.803]), rtol=1e-14)
        forward = prepared.operator.mv(direction)
        reverse = prepared.operator.transpose_mv(cotangent)
        assert jnp.allclose(forward, jnp.asarray([0.72, -2.028]), rtol=1e-14)
        assert jnp.allclose(reverse, jnp.asarray([-0.48, 2.535]), rtol=1e-14)
        assert jnp.allclose(
            jnp.vdot(cotangent, forward), jnp.vdot(reverse, direction), rtol=1e-14
        )


def test_rematerialized_root_preserves_rejected_mapped_lane_and_solves_valid_lane() -> (
    None
):
    with jax.enable_x64():
        initial = jnp.asarray([[1.2, 1.3], [-1.2, 1.3]], dtype=jnp.float64)
        targets = jnp.asarray([4.0, 8.0], dtype=jnp.float64)
        problem = nl.NonlinearSystemProblem(
            _guarded_polynomial,
            trial_validity=lambda state, _: jnp.all(state > 0.0),
            trial_validity_id="positive-polynomial",
            problem_id="guarded-polynomial",
        )
        method = nl.NewtonKrylov(
            jacobian_policy=nl.JacobianPolicy(
                linearization_policy=la.LinearizationPolicy("rematerialize")
            )
        )
        results = eqx.filter_jit(
            jax.vmap(
                lambda state: method.solve(
                    problem,
                    state,
                    args=targets,
                    termination=nl.NonlinearTermination(
                        absolute_residual=1e-11, relative_residual=0.0
                    ),
                )
            )
        )(initial)
        assert bool(results.successful[0])
        assert jnp.allclose(results.state[0], jnp.asarray([2.0, 2.0]), rtol=1e-11)
        assert not bool(results.successful[1])
        assert results.status[1] == nl.NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE
        assert jnp.array_equal(results.state[1], initial[1])
        assert int(results.diagnostics.residual_evaluations[1]) == 0


@pytest.mark.parametrize("kind", ["scalar", "vector"])
def test_inactive_fractional_power_parameter_does_not_poison_root_sensitivity(
    kind: Literal["scalar", "vector"],
) -> None:
    with jax.enable_x64():
        scalar = nl.LocalRootPlan(plan_id="inactive-flux-scalar-root")
        vector = nl.VectorLocalRootPlan(1, plan_id="inactive-flux-vector-root")
        load = jnp.asarray(1.0, dtype=jnp.float64)
        flux = jnp.asarray(0.0, dtype=jnp.float64)
        exponent = jnp.asarray(0.85, dtype=jnp.float64)

        def solution(value: Array, source: Array) -> Array:
            def residual(root: Array) -> Array:
                return root - value - source**exponent

            if kind == "scalar":
                return scalar.solve(residual, value)
            return vector.solve(residual, value[None])[0]

        primal, forward = jax.jvp(
            lambda value: solution(value, flux), (load,), (jnp.ones_like(load),)
        )
        reverse = jax.grad(lambda value: solution(value, flux))(load)
        assert float(primal) == 1.0
        assert float(forward) == 1.0
        assert float(reverse) == 1.0


@pytest.mark.parametrize("kind", ["scalar", "vector"])
def test_active_undefined_source_derivative_is_not_reported_as_finite_sensitivity(
    kind: Literal["scalar", "vector"],
) -> None:
    with jax.enable_x64():
        scalar = nl.LocalRootPlan(plan_id="singular-flux-scalar-root")
        vector = nl.VectorLocalRootPlan(1, plan_id="singular-flux-vector-root")
        exponent = jnp.asarray(0.85, dtype=jnp.float64)

        def solution(flux: Array) -> Array:
            def residual(root: Array) -> Array:
                return root - 1.0 - flux**exponent

            initial = jnp.asarray(1.0, dtype=jnp.float64)
            if kind == "scalar":
                return scalar.solve(residual, initial)
            return vector.solve(residual, initial[None])[0]

        flux = jnp.asarray(0.0, dtype=jnp.float64)
        _, derivative = jax.jvp(solution, (flux,), (jnp.ones_like(flux),))
        assert not bool(jnp.isfinite(derivative))


@pytest.mark.parametrize(
    ("states", "expected_values", "expected_derivatives"),
    [
        pytest.param(
            ((4.0, -1.0), (0.0, 9.0)),
            ((log(4.0) + 2.0, 0.0), (0.0, log(9.0) + 3.0)),
            ((0.5, 0.0), (0.0, 5.0 / 18.0)),
            id="mixed",
        ),
        pytest.param(
            ((1.0, 4.0), (9.0, 16.0)),
            ((1.0, log(4.0) + 2.0), (log(9.0) + 3.0, log(16.0) + 4.0)),
            ((1.5, 0.5), (5.0 / 18.0, 3.0 / 16.0)),
            id="all-valid",
        ),
        pytest.param(
            ((-1.0, 0.0), (-4.0, -9.0)),
            ((0.0, 0.0), (0.0, 0.0)),
            ((0.0, 0.0), (0.0, 0.0)),
            id="all-refused",
        ),
    ],
)
def test_pure_guard_keeps_nan_unsafe_refused_lanes_out_of_values_and_derivatives(
    states: tuple[tuple[float, float], tuple[float, float]],
    expected_values: tuple[tuple[float, float], tuple[float, float]],
    expected_derivatives: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    with jax.enable_x64():

        def residual(state: Array, _: None) -> Array:
            return jnp.log(state) + jnp.sqrt(state)

        problem = nl.NonlinearSystemProblem(
            residual,
            trial_validity=lambda state, _: state > 0.0,
            trial_validity_id="positive-log-sqrt",
        )
        mapped = jax.vmap(jax.vmap(problem.residual))
        inputs = jnp.asarray(states, dtype=jnp.float64)
        values, tangent = jax.jit(
            lambda state: jax.jvp(mapped, (state,), (jnp.ones_like(state),))
        )(inputs)
        reverse = jax.jit(jax.grad(lambda state: jnp.sum(mapped(state))))(inputs)
        assert jnp.all(jnp.isfinite(values))
        assert jnp.all(jnp.isfinite(tangent))
        assert jnp.all(jnp.isfinite(reverse))
        assert jnp.allclose(values, jnp.asarray(expected_values, dtype=inputs.dtype))
        expected = jnp.asarray(expected_derivatives, dtype=inputs.dtype)
        assert jnp.allclose(tangent, expected)
        assert jnp.allclose(reverse, expected)


def test_pure_error_check_raises_only_for_selected_valid_lanes() -> None:
    with jax.enable_x64():

        def residual(state: Array, _: None) -> Array:
            checked = eqx.error_if(
                state,
                (state <= 0.0) | (state > 4.0),
                "valid scientific lane failed",
            )
            return jnp.log(checked) + jnp.sqrt(checked)

        problem = nl.NonlinearSystemProblem(
            residual,
            trial_validity=lambda state, _: state > 0.0,
            trial_validity_id="positive-checked-log-sqrt",
        )
        mapped = jax.vmap(jax.vmap(problem.residual))
        evaluate = eqx.filter_jit(mapped)
        inputs = jnp.asarray([[1.0, -1.0], [4.0, -4.0]], dtype=jnp.float64)
        expected = jnp.asarray([[1.0, 0.0], [log(4.0) + 2.0, 0.0]], dtype=inputs.dtype)
        assert jnp.allclose(evaluate(inputs), expected)
        derivative = eqx.filter_jit(jax.grad(lambda state: jnp.sum(mapped(state))))(
            inputs
        )
        assert jnp.allclose(
            derivative, jnp.asarray([[1.5, 0.0], [0.5, 0.0]], dtype=inputs.dtype)
        )
        refused = jnp.asarray([[-1.0, 0.0], [-4.0, -9.0]], dtype=inputs.dtype)
        assert jnp.array_equal(evaluate(refused), jnp.zeros_like(refused))
        invalid_law = jnp.asarray([[1.0, -1.0], [9.0, -4.0]], dtype=inputs.dtype)
        with pytest.raises(eqx.EquinoxRuntimeError):
            jax.block_until_ready(evaluate(invalid_law))


@pytest.mark.strict_jax
def test_shared_guard_operands_remain_dynamic_under_nested_linearization() -> None:
    with jax.enable_x64():

        def evaluate(
            point: tuple[Array, tuple[Array, ...]],
        ) -> Array:
            states, coefficients = point

            def residual(state: Array, _args: None) -> Array:
                return jnp.sum(jnp.stack(coefficients)) * jnp.log(state) + jnp.sqrt(state)

            problem = nl.NonlinearSystemProblem(
                residual,
                trial_validity=lambda state, _args: state > 0.0,
                trial_validity_id="positive-shared-log-sqrt",
            )
            return jax.vmap(jax.vmap(problem.residual))(states)

        states = jnp.asarray(
            [[4.0, -1.0, 0.0, 9.0], [-1.0, 0.0, -4.0, -9.0]],
            dtype=jnp.float64,
        )
        coefficients = tuple(
            jnp.asarray(value, dtype=jnp.float64) for value in (1.0, 3.0, 5.0)
        )
        direction = (
            jnp.asarray(
                [[0.3, -0.2, 0.4, -0.5], [0.1, 0.2, 0.3, 0.4]],
                dtype=jnp.float64,
            ),
            tuple(jnp.asarray(value, dtype=jnp.float64) for value in (0.1, 0.2, 0.3)),
        )
        covector = jnp.asarray(
            [[0.7, -0.3, 0.2, 0.9], [0.1, 0.2, 0.3, 0.4]],
            dtype=jnp.float64,
        )
        linearization = la.prepare_linearization(evaluate, (states, coefficients))
        tangent = linearization.jvp(direction)
        state_cotangent, coefficient_cotangents = linearization.vjp(covector)
        expected_value = jnp.asarray(
            [
                [9.0 * log(4.0) + 2.0, 0.0, 0.0, 9.0 * log(9.0) + 3.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=jnp.float64,
        )
        expected_tangent = jnp.asarray(
            [
                [0.75 + 0.6 * log(4.0), 0.0, 0.0, -7.0 / 12.0 + 0.6 * log(9.0)],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=jnp.float64,
        )
        expected_state_cotangent = jnp.asarray(
            [[1.75, 0.0, 0.0, 1.05], [0.0, 0.0, 0.0, 0.0]],
            dtype=jnp.float64,
        )
        expected_coefficient = 0.7 * log(4.0) + 0.9 * log(9.0)
        assert jnp.allclose(linearization.primal, expected_value, rtol=1e-13)
        assert jnp.allclose(tangent, expected_tangent, rtol=1e-13)
        assert jnp.allclose(state_cotangent, expected_state_cotangent, rtol=1e-13)
        assert jnp.allclose(
            jnp.stack(coefficient_cotangents),
            jnp.full((3,), expected_coefficient, dtype=jnp.float64),
            rtol=1e-13,
        )


@pytest.mark.strict_jax
@pytest.mark.parametrize(
    "policy",
    [la.LinearizationPolicy("store"), la.LinearizationPolicy("rematerialize")],
    ids=["store", "rematerialize"],
)
def test_unselected_undefined_shared_operand_has_zero_value_and_derivative(
    policy: la.LinearizationPolicy,
) -> None:
    with jax.enable_x64():

        def evaluate(point: tuple[Array, Array]) -> Array:
            states, coefficient = point

            def residual(state: Array, _args: None) -> Array:
                return jnp.sqrt(coefficient) + jnp.log(state)

            problem = nl.NonlinearSystemProblem(
                residual,
                trial_validity=lambda state, _args: state > 0.0,
                trial_validity_id="positive-unselected-undefined-shared",
            )
            return jax.vmap(jax.vmap(problem.residual))(states)

        states = jnp.asarray([[-1.0, 0.0], [-4.0, -9.0]], dtype=jnp.float64)
        coefficient = jnp.asarray(-4.0, dtype=jnp.float64)
        linearization = la.prepare_linearization(
            evaluate, (states, coefficient), policy=policy
        )
        tangent = linearization.jvp(
            (jnp.ones_like(states), jnp.asarray(1.0, dtype=jnp.float64))
        )
        state_cotangent, coefficient_cotangent = linearization.vjp(jnp.ones_like(states))
        assert jnp.array_equal(linearization.primal, jnp.zeros_like(states))
        assert jnp.array_equal(tangent, jnp.zeros_like(states))
        assert jnp.array_equal(state_cotangent, jnp.zeros_like(states))
        assert jnp.array_equal(coefficient_cotangent, jnp.asarray(0.0, dtype=jnp.float64))


def test_admissible_vector_local_root_globalizes_around_invalid_newton_trial() -> None:
    plan = nl.VectorLocalRootPlan(
        1, maximum_steps=40, tolerance=1.0e-12, plan_id="positive-log-root"
    )

    def root(parameter: Array) -> Array:
        def residual(value: Array) -> Array:
            checked = eqx.error_if(
                value, jnp.any(value <= 0.0), "invalid residual evaluated"
            )
            return jnp.log(checked) - parameter

        return plan.solve(
            residual,
            jnp.asarray([100.0]),
            admissible=lambda value: jnp.all(value > 0.0),
        )[0]

    parameter = jnp.asarray(1.5)
    # Analytic x = exp(parameter); the full Newton trial from 100 is negative.
    assert jnp.allclose(jax.jit(root)(parameter), exp(1.5), rtol=1.0e-11, atol=0.0)
    assert jnp.allclose(jax.grad(root)(parameter), exp(1.5), rtol=1.0e-11, atol=0.0)
