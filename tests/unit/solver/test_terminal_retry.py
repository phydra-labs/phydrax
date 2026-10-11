from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.solver import (
    CallableFixedStepMethod,
    FixedStepResult,
    retry_fixed_step,
    RobustRetryPolicy,
)


def _cooling_step(
    step_index: Array, time: Array, state: Array, dt: Array, args: None
) -> FixedStepResult:
    del step_index, time, args
    candidate = state / (1.0 + dt)
    increment = jnp.abs(candidate - state)
    accepted = increment <= 0.25
    return FixedStepResult(
        candidate_state=candidate,
        accepted_state=jnp.where(accepted, candidate, state),
        successful=accepted,
        residual=candidate - state + dt * candidate,
        iterations=jnp.asarray(1, dtype=jnp.int32),
        work=jnp.asarray(1, dtype=jnp.int32),
        transform_applied=jnp.asarray(False),
        transform_correction_norm=jnp.asarray(0.0),
        evidence={"physical_increment": increment},
    )


def _domain_step(
    step_index: Array, time: Array, state: Array, dt: Array, args: None
) -> FixedStepResult:
    del step_index, time, args
    candidate = state + 10.0 * dt
    terminal = candidate > 2.0
    return FixedStepResult(
        candidate_state=candidate,
        accepted_state=jnp.where(terminal, state, candidate),
        successful=~terminal,
        residual=jnp.asarray(0.0),
        iterations=jnp.asarray(1, dtype=jnp.int32),
        work=jnp.asarray(1, dtype=jnp.int32),
        transform_applied=jnp.asarray(False),
        transform_correction_norm=jnp.asarray(0.0),
        terminal_failure=terminal,
        evidence={"model_domain_exit": terminal},
    )


def test_retry_accepts_only_resolution_admitted_cooling_transition() -> None:
    method = CallableFixedStepMethod(
        _cooling_step, "implicit-cooling-with-increment-limit"
    )
    policy = RobustRetryPolicy(maximum_retries=4)
    initial = jnp.asarray(1.0, dtype=jnp.float64)
    result = jax.jit(
        lambda state: retry_fixed_step(
            method,
            policy,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float64),
            state,
            jnp.asarray(1.0, dtype=jnp.float64),
        )
    )(initial)
    assert bool(result.successful)
    assert not bool(result.terminal_failure)
    np.testing.assert_array_equal(
        result.attempt_executed, [True, True, True, False, False]
    )
    assert float(result.accepted_step_size) == 0.25
    np.testing.assert_allclose(result.accepted_state, 0.8, rtol=0.0, atol=1e-15)
    assert int(result.retry_count) == 2


def test_model_domain_exit_is_not_retried_into_apparent_success() -> None:
    method = CallableFixedStepMethod(_domain_step, "source-qualified-temperature-support")
    policy = RobustRetryPolicy(maximum_retries=8)
    initial = jnp.asarray(1.0, dtype=jnp.float64)
    result = jax.jit(
        lambda state: retry_fixed_step(
            method,
            policy,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float64),
            state,
            jnp.asarray(1.0, dtype=jnp.float64),
        )
    )(initial)
    assert not bool(result.successful)
    assert bool(result.terminal_failure)
    np.testing.assert_array_equal(
        result.attempt_executed,
        [True, False, False, False, False, False, False, False, False],
    )
    np.testing.assert_array_equal(result.accepted_state, initial)
    assert float(result.candidate_state) == 11.0
    assert float(result.accepted_step_size) == 0.0
    assert int(result.retry_count) == 0
