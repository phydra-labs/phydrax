#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


FrequencySystemType: TypeAlias = Literal["continuous", "discrete"]
FrequencyStatus: TypeAlias = Literal["success", "singular", "unstable", "nonfinite"]
FREQUENCY_SUCCESS = 0
FREQUENCY_SINGULAR = 1
FREQUENCY_UNSTABLE = 2
FREQUENCY_NONFINITE = 3


def frequency_status_name(value: int, /) -> FrequencyStatus:
    """Return the public name for one frequency-response status code."""

    code = int(value)
    if code == FREQUENCY_SUCCESS:
        return "success"
    if code == FREQUENCY_SINGULAR:
        return "singular"
    if code == FREQUENCY_UNSTABLE:
        return "unstable"
    if code == FREQUENCY_NONFINITE:
        return "nonfinite"
    raise ValueError(f"Unknown frequency-response status code {code}.")


class FrequencyResponseResult(StrictModule):
    """Transfer values and resolvent, stability, and conditioning diagnostics."""

    evaluation_points: Array
    angular_frequencies: Array | None
    response: Array
    state_response: Array
    resolvent: Array
    poles: Array
    stable: Array
    condition_number: Array
    pole_condition_number: Array
    singular: Array
    valid: Array
    status: Array
    system_type: FrequencySystemType = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    sample_time: float | None = eqx.field(static=True)

    @property
    def input_to_output(self) -> Array:
        return self.response

    @property
    def input_to_state(self) -> Array:
        return self.state_response

    @property
    def frequencies(self) -> Array:
        return (
            self.evaluation_points
            if self.angular_frequencies is None
            else self.angular_frequencies
        )


def _inexact(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


def _matrix(value: ArrayLike, /, *, owner: str) -> Array:
    array = _inexact(value)
    if array.ndim < 2:
        raise ValueError(f"{owner} must have at least two dimensions.")
    return array


def _validated_matrices(
    state_matrix: ArrayLike,
    input_matrix: ArrayLike,
    output_matrix: ArrayLike,
    feedthrough_matrix: ArrayLike,
    /,
) -> tuple[Array, Array, Array, Array, tuple[int, ...], int, int]:
    state = _matrix(state_matrix, owner="state_matrix")
    inputs = _matrix(input_matrix, owner="input_matrix")
    outputs = _matrix(output_matrix, owner="output_matrix")
    feedthrough = _matrix(feedthrough_matrix, owner="feedthrough_matrix")
    state_size = int(state.shape[-1])
    if state.shape[-2] != state_size:
        raise ValueError("state_matrix must end in a square matrix.")
    if inputs.shape[-2] != state_size:
        raise ValueError("input_matrix row count must equal state size.")
    if outputs.shape[-1] != state_size:
        raise ValueError("output_matrix column count must equal state size.")
    output_size = int(outputs.shape[-2])
    input_size = int(inputs.shape[-1])
    if state_size <= 0 or input_size <= 0 or output_size <= 0:
        raise ValueError("State, input, and output matrix dimensions must be positive.")
    if feedthrough.shape[-2:] != (output_size, input_size):
        raise ValueError(
            "feedthrough_matrix must have shape (..., output_size, input_size)."
        )
    batch_shape = jnp.broadcast_shapes(
        state.shape[:-2],
        inputs.shape[:-2],
        outputs.shape[:-2],
        feedthrough.shape[:-2],
    )
    if any(size <= 0 for size in batch_shape):
        raise ValueError("Frequency-response batch dimensions must be positive.")
    state = jnp.broadcast_to(state, batch_shape + (state_size, state_size))
    inputs = jnp.broadcast_to(inputs, batch_shape + (state_size, input_size))
    outputs = jnp.broadcast_to(outputs, batch_shape + (output_size, state_size))
    feedthrough = jnp.broadcast_to(feedthrough, batch_shape + (output_size, input_size))
    return (
        state,
        inputs,
        outputs,
        feedthrough,
        batch_shape,
        state_size,
        input_size,
    )


def _frequency_result(
    state_matrix: ArrayLike,
    input_matrix: ArrayLike,
    output_matrix: ArrayLike,
    feedthrough_matrix: ArrayLike,
    points: ArrayLike,
    /,
    *,
    system_type: FrequencySystemType,
    singular_rtol: float | None,
    singular_atol: float,
    stability_margin: float,
    sample_time: float | None,
    angular_frequencies: Array | None,
) -> FrequencyResponseResult:
    if system_type not in ("continuous", "discrete"):
        raise ValueError("system_type must be 'continuous' or 'discrete'.")
    if singular_atol < 0.0:
        raise ValueError("singular_atol must be non-negative.")
    if singular_rtol is not None and singular_rtol < 0.0:
        raise ValueError("singular_rtol must be non-negative or None.")
    if stability_margin < 0.0:
        raise ValueError("stability_margin must be non-negative.")
    state, inputs, outputs, feedthrough, batch_shape, state_size, _ = _validated_matrices(
        state_matrix, input_matrix, output_matrix, feedthrough_matrix
    )
    point_array = _inexact(points)
    point_shape = tuple(point_array.shape)
    point_count = int(point_array.size) if point_shape else 1
    if point_count <= 0:
        raise ValueError("evaluation_points must be non-empty.")
    complex_dtype = jnp.result_type(state, inputs, outputs, feedthrough, jnp.complex64)
    state = state.astype(complex_dtype)
    inputs = inputs.astype(complex_dtype)
    outputs = outputs.astype(complex_dtype)
    feedthrough = feedthrough.astype(complex_dtype)
    flat_points = point_array.astype(complex_dtype).reshape((point_count,))
    expanded_points = flat_points.reshape((1,) * len(batch_shape) + (point_count, 1, 1))
    identity = jnp.eye(state_size, dtype=complex_dtype)
    resolvent_matrix = expanded_points * identity - state[..., None, :, :]
    resolvent = jnp.linalg.solve(
        resolvent_matrix,
        jnp.broadcast_to(identity, batch_shape + (point_count, state_size, state_size)),
    )
    state_response = resolvent @ inputs[..., None, :, :]
    response = outputs[..., None, :, :] @ state_response + feedthrough[..., None, :, :]

    singular_values = jnp.linalg.svd(resolvent_matrix, compute_uv=False)
    largest = singular_values[..., 0]
    smallest = singular_values[..., -1]
    real_dtype = singular_values.dtype
    relative_tolerance = (
        float(jnp.finfo(real_dtype).eps * state_size)
        if singular_rtol is None
        else singular_rtol
    )
    threshold = jnp.asarray(singular_atol, dtype=real_dtype) + (
        jnp.asarray(relative_tolerance, dtype=real_dtype) * largest
    )
    singular = smallest <= threshold
    condition_number = jnp.where(smallest > 0.0, largest / smallest, jnp.inf)

    diagnostic_state = jax.lax.stop_gradient(state)
    poles, eigenvectors = jnp.linalg.eig(diagnostic_state)
    pole_condition_number = jnp.linalg.cond(eigenvectors)
    if system_type == "continuous":
        stable = jnp.all(jnp.real(poles) < -stability_margin, axis=-1)
    else:
        stable = jnp.all(jnp.abs(poles) < 1.0 - stability_margin, axis=-1)
    point_stable = jnp.broadcast_to(stable[..., None], batch_shape + (point_count,))
    finite = (
        jnp.all(jnp.isfinite(resolvent), axis=(-2, -1))
        & jnp.all(jnp.isfinite(state_response), axis=(-2, -1))
        & jnp.all(jnp.isfinite(response), axis=(-2, -1))
        & jnp.isfinite(condition_number)
    )
    status = jnp.where(
        singular,
        FREQUENCY_SINGULAR,
        jnp.where(point_stable, FREQUENCY_SUCCESS, FREQUENCY_UNSTABLE),
    )
    status = jnp.where(~finite & ~singular, FREQUENCY_NONFINITE, status).astype(jnp.int32)
    valid = status == FREQUENCY_SUCCESS

    diagnostic_shape = batch_shape + point_shape
    matrix_prefix = diagnostic_shape
    output_size = int(outputs.shape[-2])
    input_size = int(inputs.shape[-1])
    return FrequencyResponseResult(
        evaluation_points=point_array,
        angular_frequencies=angular_frequencies,
        response=response.reshape(matrix_prefix + (output_size, input_size)),
        state_response=state_response.reshape(matrix_prefix + (state_size, input_size)),
        resolvent=resolvent.reshape(matrix_prefix + (state_size, state_size)),
        poles=poles,
        stable=stable,
        condition_number=condition_number.reshape(diagnostic_shape),
        pole_condition_number=pole_condition_number,
        singular=singular.reshape(diagnostic_shape),
        valid=valid.reshape(diagnostic_shape),
        status=status.reshape(diagnostic_shape),
        system_type=system_type,
        method_id="jax-dense-direct-resolvent",
        sample_time=sample_time,
    )


def continuous_transfer_function(
    state_matrix: ArrayLike,
    input_matrix: ArrayLike,
    output_matrix: ArrayLike,
    feedthrough_matrix: ArrayLike,
    evaluation_points: ArrayLike,
    /,
    *,
    singular_rtol: float | None = None,
    singular_atol: float = 0.0,
    stability_margin: float = 0.0,
) -> FrequencyResponseResult:
    r"""Evaluate :math:`C(sI-A)^{-1}B+D` without a pseudoinverse."""

    return _frequency_result(
        state_matrix,
        input_matrix,
        output_matrix,
        feedthrough_matrix,
        evaluation_points,
        system_type="continuous",
        singular_rtol=singular_rtol,
        singular_atol=singular_atol,
        stability_margin=stability_margin,
        sample_time=None,
        angular_frequencies=None,
    )


def discrete_transfer_function(
    state_matrix: ArrayLike,
    input_matrix: ArrayLike,
    output_matrix: ArrayLike,
    feedthrough_matrix: ArrayLike,
    evaluation_points: ArrayLike,
    /,
    *,
    sample_time: float = 1.0,
    singular_rtol: float | None = None,
    singular_atol: float = 0.0,
    stability_margin: float = 0.0,
) -> FrequencyResponseResult:
    r"""Evaluate :math:`C(zI-A)^{-1}B+D` without a pseudoinverse."""

    if sample_time <= 0.0:
        raise ValueError("sample_time must be positive.")
    return _frequency_result(
        state_matrix,
        input_matrix,
        output_matrix,
        feedthrough_matrix,
        evaluation_points,
        system_type="discrete",
        singular_rtol=singular_rtol,
        singular_atol=singular_atol,
        stability_margin=stability_margin,
        sample_time=float(sample_time),
        angular_frequencies=None,
    )


def frequency_response(
    state_matrix: ArrayLike,
    input_matrix: ArrayLike,
    output_matrix: ArrayLike,
    feedthrough_matrix: ArrayLike,
    angular_frequencies: ArrayLike,
    /,
    *,
    system_type: FrequencySystemType = "continuous",
    sample_time: float = 1.0,
    singular_rtol: float | None = None,
    singular_atol: float = 0.0,
    stability_margin: float = 0.0,
) -> FrequencyResponseResult:
    """Evaluate a continuous or discrete response at angular frequencies."""

    frequencies = _inexact(angular_frequencies)
    if system_type == "continuous":
        points = 1j * frequencies
        resolved_sample_time = None
    elif system_type == "discrete":
        if sample_time <= 0.0:
            raise ValueError("sample_time must be positive.")
        points = jnp.exp(1j * frequencies * sample_time)
        resolved_sample_time = float(sample_time)
    else:
        raise ValueError("system_type must be 'continuous' or 'discrete'.")
    return _frequency_result(
        state_matrix,
        input_matrix,
        output_matrix,
        feedthrough_matrix,
        points,
        system_type=system_type,
        singular_rtol=singular_rtol,
        singular_atol=singular_atol,
        stability_margin=stability_margin,
        sample_time=resolved_sample_time,
        angular_frequencies=frequencies,
    )


def input_to_state_response(
    state_matrix: ArrayLike,
    input_matrix: ArrayLike,
    angular_frequencies: ArrayLike,
    /,
    *,
    system_type: FrequencySystemType = "continuous",
    sample_time: float = 1.0,
    singular_rtol: float | None = None,
    singular_atol: float = 0.0,
    stability_margin: float = 0.0,
) -> FrequencyResponseResult:
    """Return input-to-state response with the same resolvent diagnostics."""

    state = _matrix(state_matrix, owner="state_matrix")
    inputs = _matrix(input_matrix, owner="input_matrix")
    state_size = int(state.shape[-1])
    input_size = int(inputs.shape[-1])
    batch_shape = jnp.broadcast_shapes(state.shape[:-2], inputs.shape[:-2])
    identity = jnp.broadcast_to(
        jnp.eye(state_size, dtype=jnp.result_type(state, inputs)),
        batch_shape + (state_size, state_size),
    )
    zero = jnp.zeros(batch_shape + (state_size, input_size), dtype=identity.dtype)
    return frequency_response(
        state,
        inputs,
        identity,
        zero,
        angular_frequencies,
        system_type=system_type,
        sample_time=sample_time,
        singular_rtol=singular_rtol,
        singular_atol=singular_atol,
        stability_margin=stability_margin,
    )


def input_to_output_response(
    state_matrix: ArrayLike,
    input_matrix: ArrayLike,
    output_matrix: ArrayLike,
    feedthrough_matrix: ArrayLike,
    angular_frequencies: ArrayLike,
    /,
    *,
    system_type: FrequencySystemType = "continuous",
    sample_time: float = 1.0,
    singular_rtol: float | None = None,
    singular_atol: float = 0.0,
    stability_margin: float = 0.0,
) -> FrequencyResponseResult:
    """Return input-to-output response and explicit conditioning diagnostics."""

    return frequency_response(
        state_matrix,
        input_matrix,
        output_matrix,
        feedthrough_matrix,
        angular_frequencies,
        system_type=system_type,
        sample_time=sample_time,
        singular_rtol=singular_rtol,
        singular_atol=singular_atol,
        stability_margin=stability_margin,
    )
