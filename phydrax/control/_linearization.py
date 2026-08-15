#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._dynamics import DifferentialControlDynamics, DiscreteControlDynamics


ControlSystemType = Literal["continuous", "discrete"]
OutputFunction = Callable[[Array, Array, Array, Any], ArrayLike]


class LinearizationProvenance(StrictModule):
    """Static identity of a local input/output linearization."""

    dynamics_id: str = eqx.field(static=True)
    dynamics_method_id: str = eqx.field(static=True)
    linearization_id: str = eqx.field(static=True)
    system_type: ControlSystemType = eqx.field(static=True)


class AffineControlLinearization(StrictModule):
    r"""Batched affine local model at explicit operating points.

    For a discrete model, ``dynamics_value`` is the next state and
    ``affine_offset`` gives :math:`x^+ = A x + B u + a`. For a differential
    model it gives :math:`\dot x = A x + B u + a`. The output model is
    :math:`y = C x + D u + c`. State, control, and output axes are flattened
    only in the matrices; operating-point values retain their declared shapes.
    """

    operating_time: Array
    operating_state: Array
    operating_control: Array
    dynamics_value: Array
    output_value: Array
    state_matrix: Array
    control_matrix: Array
    affine_offset: Array
    output_matrix: Array
    feedthrough_matrix: Array
    output_offset: Array
    valid: Array
    provenance: LinearizationProvenance
    state_shape: tuple[int, ...] = eqx.field(static=True)
    control_shape: tuple[int, ...] = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)

    @property
    def A(self) -> Array:
        return self.state_matrix

    @property
    def B(self) -> Array:
        return self.control_matrix

    @property
    def C(self) -> Array:
        return self.output_matrix

    @property
    def D(self) -> Array:
        return self.feedthrough_matrix


ControlDynamics = DiscreteControlDynamics | DifferentialControlDynamics


def _physical_shape(value: tuple[int, ...], /, *, owner: str) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError(f"{owner} dimensions must be positive.")
    return shape


def _inexact(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


def _case_shape(array: Array, physical_shape: tuple[int, ...], /, *, owner: str):
    physical_rank = len(physical_shape)
    if not physical_rank:
        return array.shape
    if (
        array.ndim < physical_rank
        or tuple(array.shape[-physical_rank:]) != physical_shape
    ):
        raise ValueError(
            f"{owner} must end in shape {physical_shape}; got {array.shape}."
        )
    return array.shape[:-physical_rank]


def _linearize(
    dynamics: ControlDynamics,
    time: ArrayLike,
    state: ArrayLike,
    control: ArrayLike,
    /,
    *,
    args: Any,
    output: OutputFunction | None,
    linearization_id: str,
    system_type: ControlSystemType,
) -> AffineControlLinearization:
    if not isinstance(linearization_id, str) or not linearization_id:
        raise ValueError("linearization_id must be a non-empty string.")
    state_shape = _physical_shape(dynamics.state_shape, owner="state_shape")
    control_shape = _physical_shape(dynamics.control_shape, owner="control_shape")
    states = _inexact(state)
    controls = _inexact(control)
    times = _inexact(time)
    state_cases = _case_shape(states, state_shape, owner="state")
    control_cases = _case_shape(controls, control_shape, owner="control")
    case_shape = jnp.broadcast_shapes(state_cases, control_cases, times.shape)
    if any(size <= 0 for size in case_shape):
        raise ValueError("Linearization case dimensions must be positive.")
    states = jnp.broadcast_to(states, case_shape + state_shape)
    controls = jnp.broadcast_to(controls, case_shape + control_shape)
    times = jnp.broadcast_to(times, case_shape)

    state_size = prod(state_shape)
    control_size = prod(control_shape)
    case_count = prod(case_shape) if case_shape else 1
    flat_states = states.reshape((case_count, state_size))
    flat_controls = controls.reshape((case_count, control_size))
    flat_times = times.reshape((case_count,))
    system = dynamics.system

    def evaluate_dynamics(t, flat_state, flat_control):
        value = system.evaluate(
            t,
            flat_state.reshape(state_shape),
            args,
            inputs=flat_control.reshape(control_shape),
        )
        array = _inexact(value)
        if array.shape != state_shape:
            raise ValueError(
                f"Control dynamics output must have shape {state_shape}; got {array.shape}."
            )
        return array.reshape((state_size,))

    dynamics_value = jax.vmap(evaluate_dynamics)(flat_times, flat_states, flat_controls)
    state_matrix, control_matrix = jax.vmap(
        jax.jacfwd(evaluate_dynamics, argnums=(1, 2))
    )(flat_times, flat_states, flat_controls)
    affine_offset = (
        dynamics_value
        - oe.contract("...ij,...j->...i", state_matrix, flat_states)
        - oe.contract("...ij,...j->...i", control_matrix, flat_controls)
    )

    if output is None:
        output_shape = state_shape
        output_size = state_size
        output_value = flat_states
        output_matrix = jnp.broadcast_to(
            jnp.eye(state_size, dtype=states.dtype),
            (case_count, state_size, state_size),
        )
        feedthrough_matrix = jnp.zeros(
            (case_count, state_size, control_size), dtype=states.dtype
        )
    else:
        first_output = _inexact(
            output(
                flat_times[0],
                flat_states[0].reshape(state_shape),
                flat_controls[0].reshape(control_shape),
                args,
            )
        )
        output_shape = tuple(first_output.shape)
        output_size = prod(output_shape) if output_shape else 1

        def evaluate_output(t, flat_state, flat_control):
            value = _inexact(
                output(
                    t,
                    flat_state.reshape(state_shape),
                    flat_control.reshape(control_shape),
                    args,
                )
            )
            if value.shape != output_shape:
                raise ValueError(
                    f"output must have shape {output_shape}; got {value.shape}."
                )
            return value.reshape((output_size,))

        output_value = jax.vmap(evaluate_output)(flat_times, flat_states, flat_controls)
        output_matrix, feedthrough_matrix = jax.vmap(
            jax.jacfwd(evaluate_output, argnums=(1, 2))
        )(flat_times, flat_states, flat_controls)

    output_offset = (
        output_value
        - oe.contract("...ij,...j->...i", output_matrix, flat_states)
        - oe.contract("...ij,...j->...i", feedthrough_matrix, flat_controls)
    )
    finite_parts = (
        flat_times.reshape((case_count, 1)),
        flat_states,
        flat_controls,
        dynamics_value,
        output_value,
        state_matrix,
        control_matrix,
        affine_offset,
        output_matrix,
        feedthrough_matrix,
        output_offset,
    )
    valid = jnp.ones((case_count,), dtype=bool)
    for part in finite_parts:
        valid = valid & jnp.all(jnp.isfinite(part.reshape((case_count, -1))), axis=-1)

    method_id = dynamics.method_id
    return AffineControlLinearization(
        operating_time=times,
        operating_state=states,
        operating_control=controls,
        dynamics_value=dynamics_value.reshape(case_shape + state_shape),
        output_value=output_value.reshape(case_shape + output_shape),
        state_matrix=state_matrix.reshape(case_shape + (state_size, state_size)),
        control_matrix=control_matrix.reshape(case_shape + (state_size, control_size)),
        affine_offset=affine_offset.reshape(case_shape + (state_size,)),
        output_matrix=output_matrix.reshape(case_shape + (output_size, state_size)),
        feedthrough_matrix=feedthrough_matrix.reshape(
            case_shape + (output_size, control_size)
        ),
        output_offset=output_offset.reshape(case_shape + (output_size,)),
        valid=valid.reshape(case_shape),
        provenance=LinearizationProvenance(
            dynamics_id=dynamics.dynamics_id,
            dynamics_method_id=method_id,
            linearization_id=linearization_id,
            system_type=system_type,
        ),
        state_shape=state_shape,
        control_shape=control_shape,
        output_shape=output_shape,
    )


def linearize_discrete_dynamics(
    dynamics: DiscreteControlDynamics,
    time: ArrayLike,
    state: ArrayLike,
    control: ArrayLike,
    /,
    *,
    args: Any = None,
    output: OutputFunction | None = None,
    linearization_id: str = "jax-forward-jvp",
) -> AffineControlLinearization:
    """Linearize a discrete transition and optional output at explicit points."""

    if not isinstance(dynamics, DiscreteControlDynamics):
        raise TypeError("dynamics must be a DiscreteControlDynamics.")
    return _linearize(
        dynamics,
        time,
        state,
        control,
        args=args,
        output=output,
        linearization_id=linearization_id,
        system_type="discrete",
    )


def linearize_differential_dynamics(
    dynamics: DifferentialControlDynamics,
    time: ArrayLike,
    state: ArrayLike,
    control: ArrayLike,
    /,
    *,
    args: Any = None,
    output: OutputFunction | None = None,
    linearization_id: str = "jax-forward-jvp",
) -> AffineControlLinearization:
    """Linearize a differential vector field and optional output at explicit points."""

    if not isinstance(dynamics, DifferentialControlDynamics):
        raise TypeError("dynamics must be a DifferentialControlDynamics.")
    return _linearize(
        dynamics,
        time,
        state,
        control,
        args=args,
        output=output,
        linearization_id=linearization_id,
        system_type="continuous",
    )


def linearize_control_dynamics(
    dynamics: ControlDynamics,
    time: ArrayLike,
    state: ArrayLike,
    control: ArrayLike,
    /,
    *,
    args: Any = None,
    output: OutputFunction | None = None,
    linearization_id: str = "jax-forward-jvp",
) -> AffineControlLinearization:
    """Dispatch to the matching discrete or differential linearization."""

    if isinstance(dynamics, DiscreteControlDynamics):
        return linearize_discrete_dynamics(
            dynamics,
            time,
            state,
            control,
            args=args,
            output=output,
            linearization_id=linearization_id,
        )
    if isinstance(dynamics, DifferentialControlDynamics):
        return linearize_differential_dynamics(
            dynamics,
            time,
            state,
            control,
            args=args,
            output=output,
            linearization_id=linearization_id,
        )
    raise TypeError(
        "dynamics must be a DiscreteControlDynamics or DifferentialControlDynamics."
    )
