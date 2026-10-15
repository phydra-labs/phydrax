#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax.nonlinear as nl


pytestmark = pytest.mark.strict_jax


def _affine(state: Array, args: None) -> Array:
    del args
    return jnp.asarray(
        [[0.3, 0.1], [-0.2, 0.4]], dtype=jnp.float64
    ) @ state + jnp.asarray([1.0, -0.5], dtype=jnp.float64)


def _termination(steps: int) -> nl.NonlinearTermination:
    return nl.NonlinearTermination(
        absolute_residual=1e-30,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=steps,
    )


@pytest.mark.parametrize("steps", [1, 2, 3], ids=["first", "second", "third"])
def test_affine_iterates_match_independent_irons_tuck(steps: int) -> None:
    matrix = np.asarray([[0.3, 0.1], [-0.2, 0.4]], dtype=np.float64)
    forcing = np.asarray([1.0, -0.5], dtype=np.float64)
    state = np.asarray([0.2, -0.1], dtype=np.float64)
    previous = None
    omega = 0.7
    for _ in range(steps):
        increment = matrix @ state + forcing - state
        if previous is not None:
            delta = increment - previous
            omega = np.clip(
                -omega * np.vdot(previous, delta).real / np.vdot(delta, delta).real,
                0.05,
                1.5,
            )
        state = state + omega * increment
        previous = increment
    method = nl.FixedPointIteration(
        acceleration=nl.AitkenRelaxation(initial_relaxation=0.7)
    )
    result = eqx.filter_jit(method.solve)(
        nl.FixedPointProblem(_affine),
        jnp.asarray([0.2, -0.1], dtype=jnp.float64),
        termination=_termination(steps),
    )
    np.testing.assert_allclose(result.state, state, rtol=1e-13, atol=1e-13)
    assert int(result.diagnostics.iterations) == steps
    assert int(result.diagnostics.residual_evaluations) == steps + 1
    assert int(result.diagnostics.linear_solves) == 0
    assert int(result.diagnostics.linear_iterations) == 0
    assert int(result.diagnostics.acceleration_restarts) == 0
    assert method.method_id.startswith("aitken-fixed-point:")


@pytest.mark.parametrize(
    ("previous", "increment", "omega", "expected"),
    [
        ([1.0], [0.9], 1.0, 1.5),
        ([1.0], [2.0], 1.0, 0.05),
        ([1.0], [1.0], 0.7, 0.7),
        ([1.0], [np.inf], 0.7, 0.7),
        ([1.0], [np.nan], 0.7, 0.7),
        ([1.0], [1.0], 2.0, 1.5),
        ([1.0 + 1.0j], [0.5 + 0.5j], 0.5, 1.0),
    ],
    ids=[
        "upper-clip",
        "lower-clip",
        "zero-retain",
        "infinite-retain",
        "nan-retain",
        "retained-clip",
        "hermitian-real",
    ],
)
def test_update_safeguards(
    previous: list[complex], increment: list[complex], omega: float, expected: float
) -> None:
    policy = nl.AitkenRelaxation()
    dtype = (
        jnp.complex128
        if any(isinstance(value, complex) for value in previous)
        else jnp.float64
    )
    actual = eqx.filter_jit(policy.update)(
        jnp.asarray(previous, dtype=dtype),
        jnp.asarray(increment, dtype=dtype),
        jnp.asarray(omega, dtype=jnp.float64),
    )
    assert float(actual) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("initial", "minimum", "maximum"),
    [
        (1.0, 0.0, 1.5),
        (0.01, 0.05, 1.5),
        (2.0, 0.05, 1.5),
        (1.0, 2.0, 1.5),
        (np.nan, 0.05, 1.5),
        (1.0, np.nan, 1.5),
        (1.0, 0.05, np.inf),
    ],
    ids=[
        "zero-minimum",
        "initial-below",
        "initial-above",
        "reversed",
        "nan-initial",
        "nan-minimum",
        "infinite-maximum",
    ],
)
def test_invalid_bounds_are_refused(
    initial: float, minimum: float, maximum: float
) -> None:
    with pytest.raises(ValueError):
        nl.AitkenRelaxation(
            initial_relaxation=initial,
            minimum_relaxation=minimum,
            maximum_relaxation=maximum,
        )


def test_aitken_refuses_additional_damping() -> None:
    with pytest.raises(ValueError, match="damping"):
        nl.FixedPointIteration(damping=0.5, acceleration=nl.AitkenRelaxation())


def _translation(state: Array, args: None) -> Array:
    del args
    return state + 1.0


def test_zero_secant_retention_is_reported_without_extra_work() -> None:
    result = nl.FixedPointIteration(acceleration=nl.AitkenRelaxation()).solve(
        nl.FixedPointProblem(_translation),
        jnp.zeros((1,), dtype=jnp.float64),
        termination=_termination(3),
    )
    np.testing.assert_allclose(result.state, [3.0])
    assert int(result.diagnostics.acceleration_restarts) == 2
    assert int(result.diagnostics.residual_evaluations) == 4
    assert int(result.diagnostics.linear_solves) == 0
    assert int(result.diagnostics.linear_iterations) == 0


def test_new_solve_restarts_aitken_history() -> None:
    method = nl.FixedPointIteration(
        acceleration=nl.AitkenRelaxation(initial_relaxation=0.7)
    )
    problem = nl.FixedPointProblem(_affine)
    initial = jnp.asarray([0.2, -0.1], dtype=jnp.float64)
    method.solve(problem, initial, termination=_termination(3))
    result = method.solve(problem, initial, termination=_termination(1))
    np.testing.assert_allclose(
        result.state, initial + 0.7 * (_affine(initial, None) - initial)
    )
    assert int(result.diagnostics.acceleration_restarts) == 0


@pytest.mark.parametrize(
    ("acceleration", "expected"),
    [
        (
            None,
            "fixed-point:5ab1d0ba365544a401f76494b75d7db5fbfae7b89eabfbdf409e4b8a959967fd",
        ),
        (
            nl.AndersonAcceleration(),
            "anderson-fixed-point:b66e133c3792d1af01aeb00624bda0b1843a591b636b0a0e5156abcaf37c72a3",
        ),
    ],
    ids=["unaccelerated", "anderson"],
)
def test_method_identity_is_pinned(
    acceleration: nl.AndersonAcceleration | None, expected: str
) -> None:
    result = nl.FixedPointIteration(acceleration=acceleration).solve(
        nl.FixedPointProblem(_affine),
        jnp.zeros((2,), dtype=jnp.float64),
        termination=_termination(1),
    )
    assert result.provenance.method_id == expected


def _large_affine(state: Array, args: None) -> Array:
    del args
    return 0.5 * state + 1e200


def test_nonfinite_secant_retention_is_reported_with_finite_iterates() -> None:
    result = nl.FixedPointIteration(acceleration=nl.AitkenRelaxation()).solve(
        nl.FixedPointProblem(_large_affine),
        jnp.zeros((1,), dtype=jnp.float64),
        termination=_termination(3),
    )
    np.testing.assert_allclose(result.state, [1.75e200])
    assert int(result.diagnostics.acceleration_restarts) == 2
    assert int(result.diagnostics.residual_evaluations) == 4
    assert int(result.diagnostics.linear_solves) == 0


def test_aitken_initial_bound_changes_identity_and_iterate() -> None:
    initial = jnp.zeros((2,), dtype=jnp.float64)
    problem = nl.FixedPointProblem(_affine)
    full = nl.FixedPointIteration(acceleration=nl.AitkenRelaxation())
    half = nl.FixedPointIteration(
        acceleration=nl.AitkenRelaxation(initial_relaxation=0.5)
    )
    full_result = full.solve(problem, initial, termination=_termination(1))
    half_result = half.solve(problem, initial, termination=_termination(1))
    np.testing.assert_allclose(half_result.state, 0.5 * full_result.state)
    assert full_result.provenance.method_id != half_result.provenance.method_id
