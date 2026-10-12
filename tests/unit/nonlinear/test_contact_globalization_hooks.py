#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.nonlinear._newton import NewtonKrylov, RootLineSearch
from phydrax.nonlinear._prepared import prepare_nonlinear, refresh_nonlinear
from phydrax.nonlinear._types import (
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
)


def _linear_residual(state: Array, _: Any) -> Array:
    return state - 1.0


def _argument_bound(state: Array, direction: Array, bound: Array) -> Array:
    del state, direction
    return bound


def _quadratic_potential(state: Array, _: Any) -> Array:
    return 0.5 * jnp.sum((state - 1.0) ** 2)


def _one_step() -> NonlinearTermination:
    return NonlinearTermination(
        absolute_residual=0.0, relative_residual=0.0, maximum_steps=1
    )


@pytest.mark.strict_jax
@pytest.mark.parametrize("bound", [0.25, 1.0e-12], ids=["quarter", "below-rate-floor"])
def test_dynamic_bound_caps_every_trial(bound: float) -> None:
    def residual(state: Array, limit: Array) -> Array:
        state = eqx.error_if(state, jnp.any(state > limit), "unsafe trial evaluated")
        return state - 1.0

    problem = NonlinearSystemProblem(
        residual, step_limit=_argument_bound, step_limit_id="argument-safe-path"
    )
    result = jax.jit(
        lambda limit: NewtonKrylov().solve(
            problem,
            jnp.zeros((1,), dtype=jnp.float64),
            termination=_one_step(),
            args=limit,
        )
    )(jnp.asarray(bound, dtype=jnp.float64))
    np.testing.assert_allclose(result.state, [bound], rtol=1.0e-12, atol=0.0)
    assert result.diagnostics.accepted_steps == 1
    assert result.diagnostics.rejected_steps == 0


@pytest.mark.strict_jax
@pytest.mark.parametrize(
    "bound",
    [0.0, -0.1, 1.1, np.nan, np.inf, -np.inf],
    ids=["zero", "negative", "above-one", "nan", "positive-inf", "negative-inf"],
)
def test_invalid_bound_refuses_without_trial_evaluation(bound: float) -> None:
    def residual(state: Array, _: Any) -> Array:
        state = eqx.error_if(state, jnp.any(state != 0.0), "refused direction evaluated")
        return state - 1.0

    problem = NonlinearSystemProblem(
        residual, step_limit=_argument_bound, step_limit_id="argument-bound"
    )
    initial = jnp.zeros((1,), dtype=jnp.float64)
    result = jax.jit(
        lambda limit: NewtonKrylov().solve(
            problem, initial, termination=_one_step(), args=limit
        )
    )(jnp.asarray(bound, dtype=jnp.float64))
    assert not result.successful
    np.testing.assert_array_equal(result.state, initial)
    assert result.diagnostics.accepted_steps == 0
    assert result.diagnostics.rejected_steps == 1
    assert result.diagnostics.final_step_norm == 0.0


def _exponential_residual(state: Array, _: Any) -> Array:
    return jnp.exp(state) - 1.0


def _exponential_potential(state: Array, _: Any) -> Array:
    return jnp.sum(jnp.exp(state) - state)


@pytest.mark.strict_jax
@pytest.mark.parametrize("potential", [False, True], ids=["residual", "potential"])
def test_declared_merit_and_directional_jvp_select_armijo_rate(potential: bool) -> None:
    problem = NonlinearSystemProblem(
        _exponential_residual,
        potential=_exponential_potential if potential else None,
        potential_id="exponential-energy" if potential else None,
    )
    initial = jnp.asarray([-2.0], dtype=jnp.float64)
    result = NewtonKrylov().solve(problem, initial, termination=_one_step())
    # Independent scalar Newton direction. Energy accepts alpha=1/2 while the
    # squared residual rejects it and accepts alpha=1/4.
    rate = 0.5 if potential else 0.25
    expected = -2.0 + rate * (np.exp(2.0) - 1.0)
    np.testing.assert_allclose(result.state, [expected], rtol=1.0e-12)
    assert result.diagnostics.accepted_steps == 1
    assert result.diagnostics.rejected_steps == 0
    assert result.provenance.globalization_id == (
        "potential-armijo" if potential else "residual-armijo"
    )


@pytest.mark.strict_jax
def test_default_callbacks_preserve_existing_root_behavior() -> None:
    initial = jnp.asarray([-2.0], dtype=jnp.float64)
    method = NewtonKrylov(line_search=RootLineSearch())
    implicit = method.solve(
        NonlinearSystemProblem(_exponential_residual), initial, termination=_one_step()
    )
    explicit = method.solve(
        NonlinearSystemProblem(
            _exponential_residual,
            potential=None,
            potential_id=None,
            step_limit=None,
            step_limit_id=None,
        ),
        initial,
        termination=_one_step(),
    )
    for left, right in zip(
        jax.tree.leaves(implicit), jax.tree.leaves(explicit), strict=True
    ):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("hook", ["potential", "step_limit"], ids=["potential", "bound"])
@pytest.mark.parametrize(
    "missing_callback", [False, True], ids=["missing-id", "missing-action"]
)
def test_hook_identity_pairs_are_required(hook: str, missing_callback: bool) -> None:
    if hook == "potential":
        with pytest.raises(ValueError, match="supplied together"):
            NonlinearSystemProblem(
                _linear_residual,
                potential=None if missing_callback else _quadratic_potential,
                potential_id="energy" if missing_callback else None,
            )
    else:
        with pytest.raises(ValueError, match="supplied together"):
            NonlinearSystemProblem(
                _linear_residual,
                step_limit=None if missing_callback else _argument_bound,
                step_limit_id="bound" if missing_callback else None,
            )


@pytest.mark.parametrize("hook", ["potential", "step_limit"], ids=["potential", "bound"])
def test_bound_problem_preserves_hooks_and_refresh_refuses_identity_change(
    hook: str,
) -> None:
    problem = NonlinearSystemProblem(
        _linear_residual,
        potential=_quadratic_potential,
        potential_id="energy",
        step_limit=_argument_bound,
        step_limit_id="bound",
    )
    initial = jnp.zeros((1,), dtype=jnp.float64)
    bound = jnp.asarray(0.5, dtype=jnp.float64)
    prepared = prepare_nonlinear(problem, initial, args=bound)
    assert prepared.provenance.globalization_id == "potential-armijo"
    assert prepared.problem.potential_id == "energy"
    assert prepared.problem.step_limit_id == "bound"
    np.testing.assert_allclose(prepared.problem.potential(initial, bound), 0.5)
    np.testing.assert_allclose(prepared.problem.step_limit(initial, initial, bound), 0.5)
    changed = NonlinearSystemProblem(
        _linear_residual,
        potential=_quadratic_potential,
        potential_id="other-energy" if hook == "potential" else "energy",
        step_limit=_argument_bound,
        step_limit_id="other-bound" if hook == "step_limit" else "bound",
    )
    with pytest.raises(ValueError, match=f"{hook}_id"):
        refresh_nonlinear(prepared, changed, initial, args=bound)


@pytest.mark.strict_jax
def test_initial_rate_below_bound_is_not_enlarged() -> None:
    problem = NonlinearSystemProblem(
        _linear_residual, step_limit=_argument_bound, step_limit_id="argument-bound"
    )
    result = NewtonKrylov(line_search=RootLineSearch(initial_rate=0.125)).solve(
        problem,
        jnp.zeros((1,), dtype=jnp.float64),
        termination=_one_step(),
        args=jnp.asarray(0.5, dtype=jnp.float64),
    )
    np.testing.assert_allclose(result.state, [0.125], atol=1.0e-14)


@pytest.mark.strict_jax
def test_nonfinite_potential_refuses_direction() -> None:
    def potential(state: Array, _: Any) -> Array:
        return jnp.sum(state) * jnp.asarray(jnp.nan, dtype=state.dtype)

    problem = NonlinearSystemProblem(
        _linear_residual, potential=potential, potential_id="nonfinite-energy"
    )
    initial = jnp.zeros((1,), dtype=jnp.float64)
    result = NewtonKrylov().solve(problem, initial, termination=_one_step())
    assert not result.successful
    np.testing.assert_array_equal(result.state, initial)
    assert result.diagnostics.nonfinite_trials > 0


@pytest.mark.strict_jax
def test_diagonal_scaling_transports_both_state_and_direction_hooks() -> None:
    from phydrax.nonlinear._scaling import (
        NonlinearScaling,
        NonlinearScalingPolicy,
        scale_root,
    )

    def step_limit(state: Array, direction: Array, wall: Array) -> Array:
        return (wall - state[0]) / direction[0]

    problem = NonlinearSystemProblem(
        _linear_residual,
        potential=_quadratic_potential,
        potential_id="quadratic",
        step_limit=step_limit,
        step_limit_id="physical-wall",
    )
    initial = jnp.zeros((1,), dtype=jnp.float64)
    wall = jnp.asarray(0.25, dtype=jnp.float64)
    transformed = scale_root(
        problem,
        initial,
        args=wall,
        policy=NonlinearScalingPolicy(
            "explicit",
            explicit=NonlinearScaling(
                jnp.asarray([2.0], dtype=jnp.float64),
                jnp.asarray([3.0], dtype=jnp.float64),
            ),
        ),
    )
    solver_state = jnp.asarray([0.0625], dtype=jnp.float64)
    solver_direction = jnp.asarray([0.25], dtype=jnp.float64)
    np.testing.assert_allclose(
        transformed.problem.step_limit(solver_state, solver_direction, wall), 0.25
    )
    np.testing.assert_allclose(
        transformed.problem.potential(solver_state, wall), 0.5 * 0.875**2
    )
    assert transformed.problem.potential_id == "quadratic"
    assert transformed.problem.step_limit_id == "physical-wall"


def test_nonlinear_right_reconstruction_refuses_linear_path_bound() -> None:
    from phydrax.linalg._spaces import ArraySpace
    from phydrax.nonlinear._preconditioning import (
        FunctionRightNonlinearPreconditioner,
        RightPreconditionedSystem,
    )

    space = ArraySpace((1,), dtype=jnp.float64)
    problem = NonlinearSystemProblem(
        _linear_residual,
        state_space=space,
        residual_space=space,
        step_limit=_argument_bound,
        step_limit_id="linear-path",
    )
    reconstruction = FunctionRightNonlinearPreconditioner(
        lambda state, _: state**3, source=space, target=space
    )
    with pytest.raises(ValueError, match="nonlinear reconstruction"):
        RightPreconditionedSystem(problem, reconstruction)


def test_left_preconditioning_preserves_physical_merit_and_path_bound() -> None:
    from phydrax.linalg._spaces import ArraySpace
    from phydrax.nonlinear._preconditioning import (
        FunctionLeftNonlinearPreconditioner,
        LeftPreconditionedSystem,
    )

    space = ArraySpace((1,), dtype=jnp.float64)
    problem = NonlinearSystemProblem(
        _linear_residual,
        state_space=space,
        residual_space=space,
        potential=_quadratic_potential,
        potential_id="physical-energy",
        step_limit=_argument_bound,
        step_limit_id="physical-bound",
    )
    preconditioner = FunctionLeftNonlinearPreconditioner(
        lambda state, residual, args: 2.0 * residual,
        state_space=space,
        source=space,
        target=space,
    )
    transformed = LeftPreconditionedSystem(problem, preconditioner)
    state = jnp.zeros((1,), dtype=jnp.float64)
    bound = jnp.asarray(0.25, dtype=jnp.float64)
    np.testing.assert_allclose(transformed.problem.potential(state, bound), 0.5)
    np.testing.assert_allclose(transformed.problem.step_limit(state, state, bound), 0.25)
    assert transformed.problem.potential_id == "physical-energy"
    assert transformed.problem.step_limit_id == "physical-bound"


def test_mixed_precision_refuses_state_rounding_with_certified_bound() -> None:
    from phydrax._nonlinear_precision import NonlinearPrecisionPolicy
    from phydrax.nonlinear._execution import MixedPrecisionRootExecution

    problem = NonlinearSystemProblem(
        _linear_residual, step_limit=_argument_bound, step_limit_id="physical-bound"
    )
    execution = MixedPrecisionRootExecution(
        NonlinearPrecisionPolicy(state_dtype=jnp.float32)
    )
    with pytest.raises(ValueError, match="cannot preserve"):
        execution.solve(
            problem,
            jnp.zeros((1,), dtype=jnp.float64),
            NewtonKrylov(),
            _one_step(),
            args=jnp.asarray(0.25, dtype=jnp.float64),
        )


@pytest.mark.strict_jax
def test_indefinite_potential_refuses_ascent_newton_direction() -> None:
    def residual(state: Array, _: Any) -> Array:
        return -state

    def potential(state: Array, _: Any) -> Array:
        return -0.5 * jnp.sum(state**2)

    problem = NonlinearSystemProblem(
        residual, potential=potential, potential_id="indefinite-energy"
    )
    initial = jnp.ones((1,), dtype=jnp.float64)
    result = jax.jit(
        lambda state: NewtonKrylov().solve(problem, state, termination=_one_step())
    )(initial)
    # Newton points toward the stationary maximum; residual descent is not
    # potential descent and must not license an energy-increasing step.
    np.testing.assert_array_equal(result.state, initial)
    assert not result.successful
    assert result.diagnostics.accepted_steps == 0
    assert result.diagnostics.rejected_steps == 1
    assert result.diagnostics.nonfinite_trials == 0
    assert result.status == int(NonlinearStatus.LINE_SEARCH_FAILED)


@pytest.mark.strict_jax
def test_potential_domain_rejections_keep_recoverable_status() -> None:
    def domain(state: Array, _: Any) -> Array:
        return jnp.all(state <= 0.0)

    problem = NonlinearSystemProblem(
        _linear_residual,
        potential=_quadratic_potential,
        potential_id="domain-quadratic",
        trial_validity=domain,
        trial_validity_id="nonpositive",
    )
    initial = jnp.zeros((1,), dtype=jnp.float64)
    result = jax.jit(
        lambda state: NewtonKrylov(line_search=RootLineSearch(maximum_steps=3)).solve(
            problem, state, termination=_one_step()
        )
    )(initial)
    np.testing.assert_array_equal(result.state, initial)
    assert result.status == int(NonlinearStatus.RECOVERABLE_DOMAIN_FAILURE)
    assert result.diagnostics.domain_failures == 3
    assert result.diagnostics.nonfinite_trials == 0
    assert result.diagnostics.accepted_steps == 0
