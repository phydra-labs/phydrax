#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import sys
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.sparse import EdgeRelation, SparseCoordinateOperator, SparseLinearMap


la = phx.linalg
nl = phx.nonlinear


def _termination(*, maximum_steps: Any = 12) -> Any:
    return nl.NonlinearTermination(
        absolute_residual=1e-7,
        relative_residual=1e-7,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=maximum_steps,
    )


_GLOBAL_COEFFICIENT = 2.0
_GLOBAL_PARAMETERS = {"coefficient": 2.0}
_HELPER_PARAMETERS = {"coefficient": 2.0}


def _global_float_residual(state: jax.Array, args: None) -> jax.Array:
    del args
    return _GLOBAL_COEFFICIENT * state - 6.0


def _global_container_residual(state: jax.Array, args: None) -> jax.Array:
    del args
    return _GLOBAL_PARAMETERS["coefficient"] * state - 6.0


def _helper_coefficient() -> float:
    return _HELPER_PARAMETERS["coefficient"]


def _global_helper_residual(state: jax.Array, args: None) -> jax.Array:
    del args
    return _helper_coefficient() * state - 6.0


class _GlobalHelperResidual(phx.StrictModule):
    def __call__(self, state: jax.Array, args: None) -> jax.Array:
        del args
        return _helper_coefficient() * state - 6.0


class _KeyedArithmeticResidual(phx.StrictModule):
    coefficient: jax.Array
    target: jax.Array

    def __call__(self, state: jax.Array, args: None) -> jax.Array:
        return self.coefficient * state - self.target


def _changed_arithmetic_call(
    self: _KeyedArithmeticResidual, state: jax.Array, args: None
) -> jax.Array:
    return 1.5 * self.coefficient * state - self.target


@pytest.mark.parametrize("method_type", [nl.NewtonKrylov, nl.NewtonTrustRegion])
def test_autodiff_solve_rereads_rebound_closure_arrays(
    method_type: type[nl.NewtonKrylov] | type[nl.NewtonTrustRegion],
) -> None:
    coefficients = jnp.asarray([[2.0, 3.0], [4.0, 9.0]], dtype=jnp.float64)

    def residual(state: jax.Array, args: None) -> jax.Array:
        del args
        return coefficients[0] * state - coefficients[1]

    problem = nl.NonlinearSystemProblem(residual)
    method = method_type()
    termination = nl.NonlinearTermination(
        absolute_residual=1e-12,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
    )
    first = method.solve(
        problem, jnp.ones((2,), dtype=jnp.float64), termination=termination
    )
    coefficients = jnp.asarray([[5.0, 7.0], [20.0, 35.0]], dtype=jnp.float64)
    second = method.solve(
        problem, jnp.ones((2,), dtype=jnp.float64), termination=termination
    )
    assert bool(first.successful)
    assert bool(second.successful)
    # Independent scalar roots; 64 eps bounds the two-coordinate GMRES roundoff.
    roundoff = 64 * np.finfo(np.float64).eps
    np.testing.assert_allclose(first.state, [2.0, 3.0], rtol=roundoff, atol=0.0)
    np.testing.assert_allclose(second.state, [4.0, 5.0], rtol=roundoff, atol=0.0)
    np.testing.assert_allclose(
        residual(second.state, None), np.zeros((2,), dtype=np.float64), atol=1e-12
    )


@pytest.mark.parametrize("method_type", [nl.NewtonKrylov, nl.NewtonTrustRegion])
def test_autodiff_solve_rereads_rebound_closure_float(
    method_type: type[nl.NewtonKrylov] | type[nl.NewtonTrustRegion],
) -> None:
    coefficient = 2.0

    def residual(state: jax.Array, args: None) -> jax.Array:
        del args
        return coefficient * state - 6.0

    problem = nl.NonlinearSystemProblem(residual)
    method = method_type()
    first = method.solve(
        problem, jnp.ones((1,), dtype=jnp.float64), termination=_termination()
    )
    coefficient = 3.0
    second = method.solve(
        problem, jnp.ones((1,), dtype=jnp.float64), termination=_termination()
    )
    assert bool(first.successful)
    assert bool(second.successful)
    # Independent scalar roots; 64 eps bounds GMRES and scalar arithmetic roundoff.
    roundoff = 64 * np.finfo(np.float64).eps
    np.testing.assert_allclose(first.state, [3.0], rtol=roundoff, atol=0.0)
    np.testing.assert_allclose(second.state, [2.0], rtol=roundoff, atol=0.0)


@pytest.mark.parametrize("method_type", [nl.NewtonKrylov, nl.NewtonTrustRegion])
def test_autodiff_solve_rereads_rebound_global_float(
    method_type: type[nl.NewtonKrylov] | type[nl.NewtonTrustRegion],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.modules[__name__], "_GLOBAL_COEFFICIENT", 2.0)
    problem = nl.NonlinearSystemProblem(_global_float_residual)
    method = method_type()
    first = method.solve(
        problem, jnp.ones((1,), dtype=jnp.float64), termination=_termination()
    )
    monkeypatch.setattr(sys.modules[__name__], "_GLOBAL_COEFFICIENT", 3.0)
    second = method.solve(
        problem, jnp.ones((1,), dtype=jnp.float64), termination=_termination()
    )
    assert bool(first.successful)
    assert bool(second.successful)
    # Independent scalar roots; 64 eps bounds scalar solve/arithmetic roundoff.
    roundoff = 64 * np.finfo(np.float64).eps
    np.testing.assert_allclose(first.state, [3.0], rtol=roundoff, atol=0.0)
    np.testing.assert_allclose(second.state, [2.0], rtol=roundoff, atol=0.0)
    np.testing.assert_allclose(_global_float_residual(second.state, None), 0.0, atol=1e-7)


@pytest.mark.parametrize("method_type", [nl.NewtonKrylov, nl.NewtonTrustRegion])
def test_autodiff_solve_rereads_mutated_global_container(
    method_type: type[nl.NewtonKrylov] | type[nl.NewtonTrustRegion],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(_GLOBAL_PARAMETERS, "coefficient", 2.0)
    problem = nl.NonlinearSystemProblem(_global_container_residual)
    method = method_type()
    first = method.solve(
        problem, jnp.ones((1,), dtype=jnp.float64), termination=_termination()
    )
    monkeypatch.setitem(_GLOBAL_PARAMETERS, "coefficient", 3.0)
    second = method.solve(
        problem, jnp.ones((1,), dtype=jnp.float64), termination=_termination()
    )
    assert bool(first.successful)
    assert bool(second.successful)
    # Independent scalar roots; 64 eps bounds scalar solve/arithmetic roundoff.
    roundoff = 64 * np.finfo(np.float64).eps
    np.testing.assert_allclose(first.state, [3.0], rtol=roundoff, atol=0.0)
    np.testing.assert_allclose(second.state, [2.0], rtol=roundoff, atol=0.0)
    np.testing.assert_allclose(
        _global_container_residual(second.state, None), 0.0, atol=1e-7
    )


@pytest.mark.parametrize("method_type", [nl.NewtonKrylov, nl.NewtonTrustRegion])
@pytest.mark.parametrize("callback_kind", ["function", "module", "partial"])
def test_autodiff_solve_rereads_transitive_helper_state(
    method_type: type[nl.NewtonKrylov] | type[nl.NewtonTrustRegion],
    callback_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(_HELPER_PARAMETERS, "coefficient", 2.0)
    match callback_kind:
        case "function":
            callback = _global_helper_residual
        case "module":
            callback = _GlobalHelperResidual()
        case "partial":
            callback = eqx.Partial(_GlobalHelperResidual())
        case _:
            raise AssertionError(f"Unexpected callback kind: {callback_kind}")
    problem = nl.NonlinearSystemProblem(callback)
    method = method_type()
    first = method.solve(
        problem, jnp.ones((1,), dtype=jnp.float64), termination=_termination()
    )
    monkeypatch.setitem(_HELPER_PARAMETERS, "coefficient", 3.0)
    second = method.solve(
        problem, jnp.ones((1,), dtype=jnp.float64), termination=_termination()
    )
    assert bool(first.successful)
    assert bool(second.successful)
    # Independent scalar roots; 64 eps bounds scalar solve/arithmetic roundoff.
    roundoff = 64 * np.finfo(np.float64).eps
    np.testing.assert_allclose(first.state, [3.0], rtol=roundoff, atol=0.0)
    np.testing.assert_allclose(second.state, [2.0], rtol=roundoff, atol=0.0)
    np.testing.assert_allclose(callback(second.state, None), 0.0, atol=1e-7)


def test_autodiff_preparation_rereads_rebound_module_call_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = _KeyedArithmeticResidual(
        jnp.asarray(2.0, dtype=jnp.float64), jnp.asarray(6.0, dtype=jnp.float64)
    )
    problem = nl.NonlinearSystemProblem(callback)
    state = jnp.ones((1,), dtype=jnp.float64)
    tangent = jnp.ones((1,), dtype=jnp.float64)
    policy = nl.JacobianPolicy()
    first = nl.prepare_jacobian(problem, state, policy)
    monkeypatch.setattr(_KeyedArithmeticResidual, "__call__", _changed_arithmetic_call)
    second = nl.prepare_jacobian(problem, state, policy)
    # Integer-valued affine residuals and actions are exact in float64.
    np.testing.assert_array_equal(first.residual, [-4.0])
    np.testing.assert_array_equal(first.operator.mv(tangent), [2.0])
    np.testing.assert_array_equal(first.operator.transpose_mv(tangent), [2.0])
    np.testing.assert_array_equal(second.residual, [-3.0])
    np.testing.assert_array_equal(second.operator.mv(tangent), [3.0])
    np.testing.assert_array_equal(second.operator.transpose_mv(tangent), [3.0])


def test_prepared_contracts() -> None:
    for method in (nl.NewtonKrylov(), nl.NewtonTrustRegion()):
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
    for method in (nl.NewtonKrylov(), nl.NewtonTrustRegion()):
        problem = nl.NonlinearSystemProblem(
            lambda state, target: state**2 - target,
            problem_id=f"prepared-relative-step-{method.method_id}",
        )
        prepared = nl.prepare_nonlinear(
            problem,
            jnp.asarray([8.0]),
            method=method,
            termination=_termination(maximum_steps=20),
            args=jnp.asarray([2.0]),
        )

        first, after_first = nl.step_prepared_nonlinear(prepared)
        second, after_second = nl.step_prepared_nonlinear(after_first)

        assert int(after_first.run.iteration) == 1
        assert int(after_second.run.iteration) == 2
        assert not jnp.array_equal(first.state, second.state)
        assert float(second.diagnostics.final_residual_norm) < float(
            first.diagnostics.final_residual_norm
        )


def test_prepared_newton_refresh_and_solve_follow_existing_jit_pattern() -> None:
    for method in (nl.NewtonKrylov(), nl.NewtonTrustRegion()):
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

        def staged(target: Any) -> Any:
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

        state, residual, status, version, linear_version = jax.jit(staged)(
            jnp.asarray([4.0])
        )

        assert int(status) == int(nl.NonlinearStatus.SUCCESS)
        assert jnp.allclose(state, jnp.asarray([4.0]), atol=1e-6)
        assert jnp.allclose(residual, jnp.zeros(1), atol=1e-6)
        assert int(version) == 1
        assert int(linear_version) > int(prepared.linear_refresh_state.numeric_version)


class _SparseCubicResidual(phx.StrictModule):
    stiffness: SparseLinearMap
    mass: SparseLinearMap
    boundary: jax.Array

    def __call__(self, state: jax.Array, source: jax.Array) -> jax.Array:
        interior = (
            self.stiffness.mv(state) + self.mass.mv(state**3) - self.mass.mv(source)
        )
        return jnp.where(self.boundary, state, interior)


@pytest.mark.parametrize("rematerialization", ["store", "rematerialize"])
def test_autodiff_jacobian_refresh_keeps_sparse_callback_captures_dynamic(
    rematerialization: la.RematerializationPolicy,
) -> None:
    relation = EdgeRelation(
        np.asarray([0, 0, 1, 1, 2], dtype=np.int32),
        np.asarray([0, 1, 1, 2, 2], dtype=np.int32),
        source_size=3,
        target_size=3,
    )
    stiffness_values = jnp.asarray([2.0, -1.0, 3.0, -0.5, 4.0], dtype=jnp.float64)
    mass_values = jnp.asarray([0.5, 0.1, 0.25, 0.2, 0.75], dtype=jnp.float64)
    residual = _SparseCubicResidual(
        SparseLinearMap(relation, stiffness_values),
        SparseLinearMap(relation, mass_values),
        jnp.asarray([True, False, False], dtype=jnp.bool_),
    )
    problem = nl.NonlinearSystemProblem(
        residual, problem_id="sparse-callback-jacobian-refresh"
    )
    policy = nl.JacobianPolicy(
        linearization_policy=la.LinearizationPolicy(rematerialization)
    )
    initial_state = jnp.asarray([0.2, 0.3, 0.4], dtype=jnp.float64)
    initial_source = jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float64)
    tangent = jnp.asarray([0.4, -0.7, 0.2], dtype=jnp.float64)
    cotangent = jnp.asarray([-0.3, 0.8, 0.5], dtype=jnp.float64)
    initial = nl.prepare_jacobian(problem, initial_state, policy, initial_source)
    current_dynamic, current_static = eqx.partition(initial, eqx.is_array)

    @eqx.filter_jit
    def select(
        next_problem: nl.NonlinearSystemProblem,
        state: jax.Array,
        source: jax.Array,
        refresh: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        def prepare(_: None) -> nl.PreparedJacobian:
            prepared = nl.prepare_jacobian(next_problem, state, policy, source)
            return eqx.partition(prepared, eqx.is_array)[0]

        selected_dynamic = jax.lax.cond(
            refresh, prepare, lambda _: current_dynamic, operand=None
        )
        selected = eqx.combine(selected_dynamic, current_static)
        return (
            selected.residual,
            selected.operator.mv(tangent),
            selected.operator.transpose_mv(cotangent),
        )

    next_state = 2.0 * initial_state
    next_source = 3.0 * initial_source
    next_problem = eqx.tree_at(
        lambda value: value.residual_function.stiffness.coefficients,
        problem,
        1.5 * stiffness_values,
    )
    kept = select(next_problem, next_state, next_source, jnp.asarray(False))
    refreshed = select(next_problem, next_state, next_source, jnp.asarray(True))
    warm_problem = eqx.tree_at(
        lambda value: value.residual_function.stiffness.coefficients,
        problem,
        2.0 * stiffness_values,
    )
    warm = select(warm_problem, next_state, next_source, jnp.asarray(True))

    # Independent three-row polynomial references, not sparse action internals.
    stiffness = np.asarray([[2.0, 0.0, 0.0], [-1.0, 3.0, 0.0], [0.0, -0.5, 4.0]])
    mass = np.asarray([[0.5, 0.0, 0.0], [0.1, 0.25, 0.0], [0.0, 0.2, 0.75]])
    for actions, state, source, scale in (
        (kept, initial_state, initial_source, 1.0),
        (refreshed, next_state, next_source, 1.5),
        (warm, next_state, next_source, 2.0),
    ):
        values = np.asarray(state)
        expected_residual = scale * stiffness @ values + mass @ (
            values**3 - np.asarray(source)
        )
        expected_residual[0] = values[0]
        jacobian = scale * stiffness + mass * (3.0 * values**2)[None, :]
        jacobian[0] = np.asarray([1.0, 0.0, 0.0])
        # These small polynomial actions have only float64 roundoff error.
        np.testing.assert_allclose(actions[0], expected_residual, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(
            actions[1], jacobian @ np.asarray(tangent), rtol=0.0, atol=1e-14
        )
        np.testing.assert_allclose(
            actions[2], jacobian.T @ np.asarray(cotangent), rtol=0.0, atol=1e-14
        )


@pytest.mark.parametrize("trust_region", [False, True])
def test_sparse_ilu_newton_prepares_topology_on_host_and_reuses_warm_compile(
    trust_region: bool,
) -> None:
    relation = EdgeRelation(
        np.asarray([0, 1, 2, 1, 2, 0, 1], dtype=np.int32),
        np.asarray([0, 1, 2, 0, 1, 1, 2], dtype=np.int32),
        source_size=3,
        target_size=3,
    )
    space = la.ArraySpace((3,), dtype=jnp.float64)
    setup = SparseCoordinateOperator(
        relation,
        jnp.asarray([4.0, 3.0, 2.0, -1.0, -1.0, -0.5, -0.5], dtype=jnp.float64),
        source=space,
        target=space,
        operator_id="newton-host-prepared-ilu-pattern",
    )

    def residual(state: jax.Array, rhs: jax.Array) -> jax.Array:
        return setup.mv(state) + 0.1 * state**3 - rhs

    problem = nl.NonlinearSystemProblem(
        residual,
        state_space=space,
        residual_space=space,
        problem_id="newton-host-prepared-ilu",
    )
    linear_policy = la.LinearSolvePolicy(
        la.GMRES(restart=3),
        preconditioning=la.PreconditioningPolicy(
            la.ILUPreconditionerBuilder(),
            setup_operator=setup,
            side="right",
            refresh="frozen",
        ),
    )
    method = (
        nl.NewtonTrustRegion(linear_policy=linear_policy)
        if trust_region
        else nl.NewtonKrylov(linear_policy=linear_policy)
    )
    expected_first = jnp.asarray([0.5, 1.0, 1.5], dtype=jnp.float64)
    expected_second = jnp.asarray([1.0, 1.5, 2.0], dtype=jnp.float64)
    first_rhs = residual(expected_first, jnp.zeros(3, dtype=jnp.float64))
    second_rhs = residual(expected_second, jnp.zeros(3, dtype=jnp.float64))
    prepared = nl.prepare_nonlinear(
        problem,
        jnp.zeros(3, dtype=jnp.float64),
        method=method,
        termination=_termination(),
        args=first_rhs,
    )
    first = nl.solve_prepared_nonlinear(prepared)
    assert bool(first.successful)
    np.testing.assert_allclose(first.state, expected_first, atol=1e-6)
    assert int(first.diagnostics.setup_refreshes) == 1
    traces = {"count": 0}

    def staged(rhs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        traces["count"] += 1
        refreshed = nl.refresh_nonlinear(
            prepared,
            problem,
            jnp.zeros_like(rhs),
            args=rhs,
        )
        result = nl.solve_prepared_nonlinear(refreshed)
        return result.state, result.status, result.diagnostics.setup_refreshes

    compiled = jax.jit(staged)
    cold_state, cold_status, cold_setups = compiled(first_rhs)
    warm_state, warm_status, warm_setups = compiled(second_rhs)
    np.testing.assert_allclose(cold_state, expected_first, atol=1e-6)
    np.testing.assert_allclose(warm_state, expected_second, atol=1e-6)
    assert int(cold_status) == int(nl.NonlinearStatus.SUCCESS)
    assert int(warm_status) == int(nl.NonlinearStatus.SUCCESS)
    assert int(cold_setups) == int(warm_setups) == 0
    assert traces["count"] == 1


def test_prepared_nonlinear_refresh_rejects_changed_linear_structure() -> None:
    space = la.PyTreeSpace(jnp.zeros(1))
    problem = nl.NonlinearSystemProblem(
        lambda state, args: args["scale"] * state - 1.0,
        state_space=space,
        residual_space=space,
        problem_id="prepared-structure-identity",
    )

    def operator(state: Any, args: Any) -> Any:
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
